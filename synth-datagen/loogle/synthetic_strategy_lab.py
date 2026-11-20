"""Generate hint-stripped SFT candidates on loogle-strategy-synth (served root, real reader sub-calls).

    python synthetic_strategy_lab.py --base-url http://localhost:8000/v1 --model qwen38-27b --family game --n 64 \
        --start 100000 --env-config recipes/long.json --note notes/compact_v6_game_program.txt --out <dir>

One rollout per seed. A correct rollout that passes ``loogle_v2_synth.strategy.accepts`` and the hint strip becomes a
candidate; review_candidates.py does the semantic review.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "environments/synthetic/loogle_v2"
for p in (ROOT / "synth-datagen/common", ROOT / "src", BASE, BASE / "tools", ROOT / "environments/synthetic/synth_common"):
    sys.path.insert(0, str(p))

from loogle_v2_synth.strategy import (  # noqa: E402
    Budgets,
    accepts,
    load_strategy_environment,
    measurements,
)
from openai import AsyncOpenAI  # noqa: E402
from strategy_lab import Meter, plain  # noqa: E402
from synth_common.probe import prime_client  # noqa: E402
from verifiers import OpenAIChatCompletionsClient  # noqa: E402

from rlm_generate_trajectories.sft_format import HintLeak, to_sft_sample  # noqa: E402


def inference_client(a):
    if not a.base_url:
        return prime_client(timeout=a.request_timeout, max_retries=1)
    headers = {}
    if "pinference" in a.base_url:
        if not os.environ.get("PRIME_API_TEAM"):
            raise ValueError("Prime endpoint requires PRIME_API_TEAM")
        headers["X-Prime-Team-ID"] = os.environ["PRIME_API_TEAM"]
    return OpenAIChatCompletionsClient(AsyncOpenAI(
        base_url=a.base_url, api_key=os.environ.get(a.api_key_env) or "local",
        default_headers=headers, timeout=a.request_timeout, max_retries=1))


async def main(a):
    note = Path(a.note).read_text() if a.note else ""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    args = json.loads(Path(a.env_config).read_text() if a.env_config else a.env_args)
    if a.env_config and a.env_args != "{}":
        raise ValueError("choose env-config or env-args")
    source_files = [BASE / "loogle_v2_synth" / f for f in ("strategy.py", "finance.py", "code.py", "game.py", "versions.py")]
    run_args = vars(a).copy()
    config = {**run_args, "note_sha256": hashlib.sha256(note.encode()).hexdigest(), "data": "procedural-only",
              "resolved_env_args": args, "resolved_budgets": vars(Budgets(**args.get("budgets", {}))),
              "generator_version": args.get("generator_version", 2),
              "source_sha256": {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
              "sub_lm": "real", "root_teacher": a.root_mode,
              "sft_status": "mechanically-screened candidates; semantic quality review still required"}
    cp = out / "config.json"
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError("output config differs; choose a new directory")
    cp.write_text(json.dumps(config, indent=2))
    rp = out / "rows.jsonl"
    old = [json.loads(line) for line in rp.read_text().splitlines()] if rp.exists() else []
    done = {r["row"] for r in old}
    if any(k in args for k in ("note", "split", "num_examples", "family", "reward_mode", "start_seed")):
        raise ValueError("use CLI flags for note/split/family/start/count; env-args is for generator/budget knobs")
    env = load_strategy_environment(split=a.split, family=a.family, num_examples=a.n,
                                    start_seed=a.start, note=note, reward_mode="outcome", **args)
    budgets = Budgets(**args.get("budgets", {}))
    sem = asyncio.Semaphore(a.concurrency)

    async def one(i):
        async with sem:
            t0 = time.monotonic()
            meter = Meter(inference_client(a))
            client = meter
            sample = None
            try:
                extra_body = {"chat_template_kwargs": {"enable_thinking": a.thinking}}
                st = await asyncio.wait_for(env._run_rollout_state(input=dict(env.dataset[i]), client=client,
                    model=a.model, sampling_args={"max_tokens": a.max_tokens, "temperature": a.temperature,
                        "extra_body": extra_body}), timeout=a.timeout)
                good, reasons = accepts(st, budgets)
                failed_calls = sum(x.get("error") is not None for x in meter.calls)
                if failed_calls:
                    good = False
                    reasons.append("inference_failure")
                h = st["synth_hidden"]
                rec = {"row": i, "seed": st["info"]["synth_seed"], "split": a.split,
                       "family": h["family"], "kind": h["kind"], "reward": float(st.get("reward") or 0),
                       "context_chars": st.get("synth_context_chars"),
                       "failed_client_calls": failed_calls,
                       "final": st.get("rlm_final_answer"), "metrics": measurements(st),
                       "accepted": bool(good and float(st.get("reward") or 0) == 1), "reasons": reasons,
                       "error": f"{type(st['error']).__name__}: {st['error']}" if st.get("error") is not None else None}
                if rec["reward"] != 1:
                    rec["reasons"].append("incorrect")
                payload = {"history": plain(st.get("rlm_history") or []), "prompt": plain(st.get("prompt") or []),
                           "completion": plain(st.get("completion") or []),
                           "sub_calls": plain(st.get("rlm_sub_llm_log") or []),
                           "complete_sub_calls": meter.sub_transcripts}
                (out / f"trace_{i}.json").write_text(json.dumps(payload, ensure_ascii=False))
                if rec["accepted"]:
                    try:
                        sample = to_sft_sample({**payload, "example_id": rec["seed"], "reward": rec["reward"],
                            "rlm_iterations": rec["metrics"]["turns"], "rlm_sub_llm_calls": rec["metrics"]["sub_calls"]},
                            strategy_name="compact-context-v1", hint=note,
                            extra_meta={"source": "procedural-only", "split": a.split, "family": h["family"],
                                        "seed": rec["seed"],
                                        "model": a.model if a.root_mode == "sampled" else "scripted-coordinator",
                                        "note_sha256": config["note_sha256"],
                                        "root_teacher": a.root_mode, "reader_model": a.model,
                                        "inference_base_url": a.base_url or "Prime/default",
                                        "root_thinking": "provider-default" if "gpt-" in a.model else a.thinking,
                                        "reader_thinking": "provider-default" if "gpt-" in a.model else False,
                                        "root_decoder": "unconstrained",
                                        "root_token_usage": "measured" if a.root_mode == "sampled" else "unavailable-scripted; use character metrics",
                                        "context_chars": rec["context_chars"],
                                        "metrics": rec["metrics"], "quality_review": "pending"})
                        if not sample["completion"] or not any(m["role"]=="assistant" for m in sample["completion"]):
                            raise ValueError("empty SFT completion")
                    except (HintLeak, ValueError) as e:
                        rec["accepted"] = False
                        rec["reasons"].append(f"export: {e}")
            except Exception as e:
                rec = {"row": i, "reward": 0., "accepted": False, "error": f"{type(e).__name__}: {str(e)[:500]}"}
            finally:
                await meter.wrapped.client.close()
            rec["seconds"] = time.monotonic()-t0
            rec["call_meter"] = meter.calls
            (out / f"row_{i}.json").write_text(json.dumps(rec, ensure_ascii=False))
            if sample is not None:
                (out / f"candidate_{i}.json").write_text(json.dumps(sample, ensure_ascii=False))
            with rp.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False)+"\n")
            print(json.dumps({k: rec.get(k) for k in ("row", "family", "kind", "reward", "metrics", "accepted", "reasons", "error")}), flush=True)

    try:
        await asyncio.gather(*(one(i) for i in range(a.n) if i not in done))
    finally:
        await env.teardown_rlm()
    rows = [json.loads(line) for line in rp.read_text().splitlines()]
    candidates = [json.loads((out/f"candidate_{r['row']}.json").read_text()) for r in rows if r.get("accepted")]
    if a.split == "train":
        (out / "sft_candidates.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in candidates))
    summary = {"n": len(rows), "reward": statistics.mean(r["reward"] for r in rows),
               "accepted": len(candidates), "errors": sum(bool(r.get("error")) for r in rows),
               "median_turns": statistics.median(r.get("metrics",{}).get("turns",0) for r in rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen/qwen3.8-27b")
    p.add_argument("--base-url", help="explicit OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="name only; credentials are never saved")
    p.add_argument("--request-timeout", type=float, default=900)
    p.add_argument("--thinking", action="store_true", help="root thinking on; readers always use thinking off")
    p.add_argument("--root-mode", choices=["sampled"], default="sampled")
    p.add_argument("--family", choices=["finance", "code", "game", "versions"], required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--note")
    p.add_argument("--env-args", default="{}")
    p.add_argument("--env-config", help="JSON file with generator and budget arguments")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--timeout", type=float, default=1200)
    p.add_argument("--out", required=True)
    asyncio.run(main(p.parse_args()))
