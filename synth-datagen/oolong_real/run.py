"""Hinted trace generation on tabletop_synth (Qwen3.8-27B root and sub-LM, thinking off).

    python run.py --synth-seeds 36000-37500 --synth-surfaces v2 --synth-roll-frac 0.3 --arm gen_v21s2 \
        --note notes/v21.md --host localhost --ports 8000,8001 --out <dir>

One rollout per seed, each pinned to one port. Writes <out>/<arm>.jsonl (resumable) and per-rollout traces.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def plain(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x if x is None or isinstance(x, (str, int, float, bool)) else str(x)


def role_content(m):
    if isinstance(m, dict):
        return m.get("role"), m.get("content")
    return getattr(m, "role", None), getattr(m, "content", None)


async def main_async(a) -> None:
    import httpx
    from openai import AsyncOpenAI
    from synth_common.program import add_hint
    from verifiers import OpenAIChatCompletionsClient

    note = Path(a.note).read_text().strip() if a.note else ""
    lo, hi = map(int, a.synth_seeds.split("-"))
    rows = [{"id": f"synth-{s}", "seed": s} for s in range(lo, hi)]
    if a.limit:
        rows = rows[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec_path = out / f"{a.arm}.jsonl"
    tdir = out / f"{a.arm}.traces"
    tdir.mkdir(exist_ok=True)
    done = {}
    if rec_path.exists():
        for s in rec_path.read_text().splitlines():
            r = json.loads(s)
            if not r.get("error"):
                done[r["id"]] = r
    pending = [r for r in rows if r["id"] not in done]
    print(json.dumps({"arm": a.arm, "total": len(rows), "pending": len(pending), "note_chars": len(note)}), flush=True)
    ports = [p.strip() for p in a.ports.split(",")]
    natives = [AsyncOpenAI(base_url=f"http://{a.host}:{p}/v1", api_key="EMPTY",
                           timeout=httpx.Timeout(a.call_timeout), max_retries=2) for p in ports]
    sem = asyncio.Semaphore(a.concurrency)

    sub_natives = ([AsyncOpenAI(base_url=f"http://{a.host}:{p.strip()}/v1", api_key="EMPTY",
                                timeout=httpx.Timeout(a.call_timeout), max_retries=2) for p in a.sub_ports.split(",")]
                   if a.sub_ports else None)

    class Meter(OpenAIChatCompletionsClient):
        def __init__(self, native, sub_native=None):
            super().__init__(native)
            self.sub = OpenAIChatCompletionsClient(sub_native) if sub_native is not None else None
            self.n = {"root": 0, "sub": 0, "root_out_tok": 0, "sub_out_tok": 0, "root_s": 0.0, "sub_s": 0.0}

        async def get_response(self, prompt, model, sampling_args, tools=None, **kw):
            root = any(role_content(m)[0] == "system" for m in prompt or [])
            t0 = time.monotonic()
            if not root and self.sub is not None:
                r = await self.sub.get_response(prompt, model, sampling_args, tools, **kw)
            else:
                r = await super().get_response(prompt, model, sampling_args, tools, **kw)
            k = "root" if root else "sub"
            self.n[k] += 1
            self.n[k + "_s"] += time.monotonic() - t0
            u = getattr(getattr(r, "response", None), "usage", None) or getattr(r, "usage", None)
            self.n[k + "_out_tok"] += int(getattr(u, "completion_tokens", 0) or 0) if u else 0
            return r

    async def one(i, row):
        async with sem:
            t0 = time.monotonic()
            from tabletop_synth.env import load_environment as synth_env
            tid = None
            if a.synth_roll_frac and random.Random(row["seed"]).random() < a.synth_roll_frac:
                from oolong_real_synth import gen as G
                tid = random.Random(row["seed"] + 1).choice(sorted(t for t in G.SINGLE_W if t.startswith("r_")
                                                                   and t in G.MULTI_W))
            env = synth_env(num_examples=1, seed=row["seed"], tid=tid, surfaces=a.synth_surfaces, max_iterations=a.max_iterations,
                            max_sub_calls=a.max_sub_calls, sub_max_tokens=a.sub_max_tokens,
                            sub_extra_body=NO_THINK, default_answer_at_limit=False)
            info = env.dataset[0]["info"]
            info = info if isinstance(info, dict) else json.loads(info)
            row = dict(row, surfaces=a.synth_surfaces, context_window_id=f"synth-{row['seed']}", question=info["question"],
                       question_type=info["qtype"], answer=info["gold"])
            inp = dict(env.dataset[0])
            if note:
                info = inp["info"] if isinstance(inp["info"], dict) else json.loads(inp["info"])
                info = dict(info)
                info["root_prompt"] = add_hint(info["root_prompt"], note)
                inp["info"] = info
            meter = Meter(natives[i % len(natives)], sub_natives[i % len(sub_natives)] if sub_natives else None)
            rec = {k: row[k] for k in ("id", "context_window_id", "question", "question_type", "answer")}
            if row.get("surfaces"):
                rec["surfaces"] = row["surfaces"]
            rec.update(arm=a.arm, port=ports[i % len(ports)])
            try:
                st = await asyncio.wait_for(env._run_rollout_state(
                    input=inp, client=meter, model=a.model,
                    sampling_args={"max_tokens": a.root_max_tokens, "temperature": a.temperature,
                                   "extra_body": NO_THINK}), a.timeout)
                err = st.get("error") or ("infra: " + st["stop_condition"] if st.get("stop_condition") in
                                           ("has_error", "repl_worker_dead") else None)
                hist = st.get("rlm_history") or []
                sub_log = st.get("rlm_sub_llm_log") or []
                root_chars = sum(len(str(role_content(m)[1] or "")) for m in hist if role_content(m)[0] == "assistant")
                rec.update(reward=float(st.get("reward") or 0.0), final=st.get("rlm_final_answer"),
                           error=f"{type(err).__name__}: {err}" if err is not None else None,
                           stop=st.get("stop_condition"), turns=st.get("rlm_iterations"),
                           sub_calls=st.get("rlm_sub_llm_calls", 0), root_chars=root_chars,
                           sub_errors=sum(str(e.get("response") or "").startswith("Error") for e in sub_log))
                (tdir / f"{row['id']}.json").write_text(json.dumps({
                    "row": rec, "history": plain(hist),
                    "sub_log": [{"prompt": str(e.get("prompt") or ""), "response": str(e.get("response") or "")}
                                for e in sub_log]}, ensure_ascii=False))
            except Exception as exc:  # noqa: BLE001
                rec.update(reward=0.0, error=f"{type(exc).__name__}: {str(exc)[:300]}")
            finally:
                await env.teardown_rlm()
            rec.update(seconds=round(time.monotonic() - t0, 1), **{k: round(v, 1) if isinstance(v, float) else v
                                                                   for k, v in meter.n.items()})
            with rec_path.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[row["id"]] = rec
            rs = [d["reward"] for d in done.values()]
            print(json.dumps({k: rec.get(k) for k in ("id", "reward", "answer", "final", "turns", "sub_calls",
                                                      "seconds", "error")}, ensure_ascii=False)[:400]
                  + f"  | {len(done)}/{len(rows)} mean {statistics.mean(rs):.3f}", flush=True)

    await asyncio.gather(*(one(i, r) for i, r in enumerate(pending)))
    for n in natives:
        await n.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-seeds", required=True, help="lo-hi: tabletop_synth seeds")
    ap.add_argument("--synth-surfaces", default="v1", help="tabletop_synth surface set (v2 = wider formats)")
    ap.add_argument("--synth-roll-frac", type=float, default=0.0, help="share of seeds forced to a roll-question template")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--ports", default="8300,8301,8302,8303")
    ap.add_argument("--sub-ports", default="", help="send sub-calls to these ports instead (default: same as root)")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-iterations", type=int, default=30)
    ap.add_argument("--max-sub-calls", type=int, default=600)
    ap.add_argument("--root-max-tokens", type=int, default=16384)
    ap.add_argument("--sub-max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--call-timeout", type=float, default=900)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
