"""LooGLE-v2 eval driver: the served model is root and sub-LM on the real 206-row cohort (`eval_cohort.json`).

    HF_HUB_OFFLINE=1 python strategy_lab.py --arm root --max-turns 12 --max-tokens 6000 --temperature 0.2 \
        --timeout 7200 --request-timeout 7200 --concurrency C --model <name> --base-url http://<host>:<port>/v1 --out <dir>

Resumable: items already in ``rows.jsonl`` are skipped, and ``config.json`` must match the earlier run.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", BASE, ROOT / "environments/real/loogle_v2",
          ROOT / "environments/synthetic/synth_common"):
    sys.path.insert(0, str(p))

from openai import AsyncOpenAI  # noqa: E402
from synth_common.probe import prime_client  # noqa: E402
from verifiers import OpenAIChatCompletionsClient  # noqa: E402


def plain(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


class Meter(OpenAIChatCompletionsClient):
    def __init__(self, client):
        super().__init__(client.client)
        self.wrapped = client
        self.calls = []
        self.sub_transcripts = []

    async def get_response(self, prompt, model, sampling_args, tools=None, **kw):
        root = any((m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "system"
                   for m in prompt)
        chars = len(json.dumps(plain(prompt), ensure_ascii=False))
        t0 = time.monotonic()
        try:
            result = await self.wrapped.get_response(prompt, model, sampling_args, tools, **kw)
        except Exception as e:
            self.calls.append({"root": root, "input_chars": chars, "seconds": time.monotonic() - t0,
                               "usage": None, "error": f"{type(e).__name__}: {e}"})
            raise
        native = getattr(result, "response", None) or result
        self.calls.append({"root": root, "input_chars": chars, "seconds": time.monotonic() - t0,
                           "usage": plain(getattr(native, "usage", None))})
        if not root:
            self.sub_transcripts.append({"prompt_messages": plain(prompt),
                                         "response": plain(getattr(result, "message", None))})
        return result


def build_env(indices: list[int], max_iterations: int):
    import loogle_v2_real.env as L
    return L.load_environment(indices=indices, evaluation_version="2026-10-11", seed=None,
                              max_iterations=max_iterations,
                              sub_extra_body={"chat_template_kwargs": {"enable_thinking": False}})


def choose(args):
    manifest = json.loads((Path(__file__).with_name("eval_cohort.json")).read_text())
    rows = [r for r in manifest["rows"] if r["pool"] == args.pool]
    excluded = set(map(int, args.exclude_indices.split(","))) if args.exclude_indices else set()
    rows = [r for r in rows if r["idx"] not in excluded]
    if args.family != "all":
        rows = [r for r in rows if r["kind"].startswith({"finance": "fin-", "code": "code-", "game": "game-"}[args.family])]
    if args.indices:
        ids = set(map(int, args.indices.split(",")))
        return [r for r in rows if r["idx"] in ids]
    if args.per_kind:
        rng = random.Random(args.seed)
        result = []
        for kind in sorted({r["kind"] for r in rows}):
            group = [r for r in rows if r["kind"] == kind]
            rng.shuffle(group)
            result.extend(group[:args.per_kind])
        return sorted(result, key=lambda r: r["idx"])
    return rows


async def main(args):
    selected = choose(args)
    if not selected:
        raise ValueError("empty cohort")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    note = Path(args.note).read_text() if args.note else ""
    config = {**vars(args), "note_sha256": hashlib.sha256(note.encode()).hexdigest(), "cohort": selected}
    cp = out / "config.json"
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError("output config differs; choose a fresh directory")
    cp.write_text(json.dumps(config, indent=2))
    rp = out / "rows.jsonl"
    old = [json.loads(x) for x in rp.read_text().splitlines()] if rp.exists() else []
    done = {r["idx"] for r in old}
    pending = [r for r in selected if r["idx"] not in done]
    if pending:
        env = build_env([r["idx"] for r in pending], args.max_turns)
        env._user_prologue = f"<strategy_hint>\n{note}\n</strategy_hint>" if note else None
        env._max_sub_calls = args.max_sub_calls
        sem = asyncio.Semaphore(args.concurrency)
        async def one(i, row):
            async with sem:
                t0 = time.monotonic()
                client = (OpenAIChatCompletionsClient(AsyncOpenAI(base_url=args.base_url, api_key="local",
                          timeout=args.request_timeout, max_retries=1)) if args.base_url else
                          prime_client(timeout=args.request_timeout, max_retries=1))
                meter = Meter(client)
                client = meter
                try:
                    st = await asyncio.wait_for(env._run_rollout_state(
                        input=dict(env.dataset[i]), client=client, model=args.model,
                        sampling_args={"max_tokens": args.max_tokens, "temperature": args.temperature,
                                       "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}),
                        timeout=args.timeout)
                    history = plain(st.get("rlm_history") or [])
                    subs = plain(st.get("rlm_sub_llm_log") or [])
                    rec = {**row, "reward": float(st.get("reward") or 0), "final": st.get("rlm_final_answer"),
                           "iterations": st.get("rlm_iterations"), "metrics": plain(st.get("metrics") or {}),
                           "error": f"{type(st['error']).__name__}: {st['error']}" if st.get("error") is not None else None,
                           "root_observation_chars": sum(len(str(m.get("content", ""))) for m in history[2:]
                                                         if m.get("role") == "user"),
                           "sub_calls": len(subs), "sub_input_chars": sum(s.get("prompt_chars", 0) for s in subs),
                           "sub_max_input_chars": max((s.get("prompt_chars", 0) for s in subs), default=0)}
                    (out / f"trace_{row['idx']}.json").write_text(json.dumps(
                        {"history": history, "sub_calls": subs, "complete_sub_calls": meter.sub_transcripts,
                         "trajectory": plain(st.get("trajectory") or [])}, ensure_ascii=False))
                except Exception as e:
                    rec = {**row, "reward": 0., "error": f"{type(e).__name__}: {str(e)[:600]}"}
                finally:
                    await meter.wrapped.client.close()
                rec.update(seconds=time.monotonic()-t0, model=args.model, arm=args.arm, call_meter=meter.calls)
                with rp.open("a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(json.dumps({k: rec.get(k) for k in ("idx", "kind", "reward", "iterations", "sub_calls", "seconds", "error")}), flush=True)
        try:
            await asyncio.gather(*(one(i, r) for i, r in enumerate(pending)))
        finally:
            await env.teardown_rlm()
    rows = [json.loads(x) for x in rp.read_text().splitlines()]
    def summarize(rs):
        return {"n": len(rs), "reward": statistics.mean(r["reward"] for r in rs),
                "errors": sum(bool(r.get("error")) for r in rs),
                **{f"median_{k}": statistics.median(r.get(k) or 0 for r in rs)
                   for k in ("iterations", "sub_calls", "root_observation_chars", "sub_input_chars", "seconds")}}
    summary = {"all": summarize(rows), "by_kind": {k: summarize([r for r in rows if r["kind"]==k])
                                                  for k in sorted({r["kind"] for r in rows})}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=["root"], default="root")
    p.add_argument("--family", choices=["all", "finance", "code", "game"], default="all")
    p.add_argument("--pool", choices=["eval"], default="eval")
    p.add_argument("--indices", default="")
    p.add_argument("--exclude-indices", default="", help="exclude development questions from a fresh cohort")
    p.add_argument("--per-kind", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261019)
    p.add_argument("--model", default="qwen/qwen3.8-27b")
    p.add_argument("--base-url", help="explicit local OpenAI-compatible inference endpoint")
    p.add_argument("--request-timeout", type=float, default=900)
    p.add_argument("--note")
    p.add_argument("--max-turns", type=int, default=12)
    p.add_argument("--max-sub-calls", type=int, default=80)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=1200)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
