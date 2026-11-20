"""Run a synthetic env with a real root model and record the rollouts.

    python -m synth_common.probe --env <pkg>:load_environment --env-args '{"split":"train"}' --model <m> --n 8 --out <dir>
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from verifiers import OpenAIChatCompletionsClient


def prime_client(timeout: float = 900.0, max_retries: int | None = None) -> OpenAIChatCompletionsClient:
    key = os.environ.get("PRIME_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = os.environ.get("OPENAI_BASE_URL", "https://api.pinference.ai/api/v1")
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
    if os.environ.get("PRIME_API_TEAM"):
        headers["X-Prime-Team-ID"] = os.environ["PRIME_API_TEAM"]
    return OpenAIChatCompletionsClient(AsyncOpenAI(api_key=key, base_url=base, default_headers=headers,
                                                   timeout=timeout,
                                                   max_retries=max_retries if max_retries is not None
                                                   else int(os.environ.get("PRIME_MAX_RETRIES", "3"))))


def load_env(spec: str, kwargs: dict) -> Any:
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "load_environment")(**kwargs)


def _content(m: Any) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    if isinstance(c, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return c or ""


def _role(m: Any) -> str:
    return (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) or "?"


def record(state: dict, row_idx: int, elapsed: float, trunc: int) -> dict:
    hist = state.get("rlm_history") or []
    turns = [{"role": _role(m), "content": _content(m)[:trunc]} for m in hist[2:]]
    subs = []
    for e in (state.get("rlm_sub_llm_log") or [])[:400]:
        p = e.get("prompt") or ""
        subs.append({"prompt_chars": e.get("prompt_chars"), "prompt_head": p[:600], "prompt_tail": p[-300:],
                     "response": (e.get("response") or "")[:1200]})
    hidden = state.get("synth_hidden") or {}
    return {
        "row": row_idx,
        "seed": (state.get("info") or {}).get("synth_seed"),
        "reward": float(state.get("reward") or 0.0),
        "metrics": {k: v for k, v in (state.get("metrics") or {}).items()},
        "iterations": int(state.get("rlm_iterations") or 0),
        "sub_calls": int(state.get("rlm_sub_llm_calls") or 0),
        "final_answer": (state.get("rlm_final_answer") or "")[:4000] if state.get("rlm_final_answer") else None,
        "gold": hidden.get("gold"),
        "elapsed_s": round(elapsed, 1),
        "root_prompt": ((state.get("info") or {}).get("root_prompt") or "")[:4000],
        "turns": turns,
        "sub_calls_log": subs,
        "error": state.get("error") and str(state.get("error"))[:500],
    }


async def main_async(a: argparse.Namespace) -> None:
    env = load_env(a.env, json.loads(a.env_args))
    n = min(a.n, len(env.dataset))
    rows = list(range(a.start, a.start + n))
    client = prime_client()
    sampling: dict[str, Any] = {"max_tokens": a.max_tokens, "temperature": a.temperature}
    if a.no_think:
        sampling["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(a.concurrency)
    recs: list[dict] = []

    async def one(i: int) -> None:
        async with sem:
            t0 = time.time()
            try:
                state = await env._run_rollout_state(input=dict(env.dataset[i]), client=client, model=a.model,
                                                     sampling_args=sampling)
                r = record(state, i, time.time() - t0, a.trunc)
            except Exception as e:  # noqa: BLE001
                r = {"row": i, "reward": 0.0, "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "elapsed_s": round(time.time() - t0, 1), "metrics": {}, "iterations": 0, "sub_calls": 0}
            recs.append(r)
            m = r.get("metrics") or {}
            print(f"[row {i}] reward={r['reward']:.3f} iters={r.get('iterations')} subs={r.get('sub_calls')} "
                  f"cov={m.get('synth_required_coverage', 0):.2f} overlong={m.get('synth_overlong', 0):.0f} "
                  f"t={r['elapsed_s']}s {('ERR ' + r['error']) if r.get('error') else ''}", flush=True)
            with open(out / "rollouts.jsonl", "a") as f:
                f.write(json.dumps(r, default=str) + "\n")

    await asyncio.gather(*(one(i) for i in rows))
    try:
        await env.teardown_rlm()
    except Exception:  # noqa: BLE001
        pass
    ok = [r for r in recs if not r.get("error")]
    keys = sorted({k for r in ok for k in (r.get("metrics") or {})})
    summ = {
        "env": a.env, "env_args": json.loads(a.env_args), "model": a.model, "n": len(recs), "errors": len(recs) - len(ok),
        "reward_mean": statistics.mean(r["reward"] for r in recs) if recs else 0.0,
        "elapsed_mean_s": statistics.mean(r["elapsed_s"] for r in recs) if recs else 0.0,
        "metrics_mean": {k: statistics.mean(float((r.get("metrics") or {}).get(k) or 0) for r in ok) for k in keys}
        if ok else {},
    }
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, help="module:function, e.g. obliq_synth:load_environment")
    ap.add_argument("--env-args", default="{}")
    ap.add_argument("--model", default="qwen/qwen3.6-27b")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--trunc", type=int, default=12000, help="chars kept per root turn / REPL output in the trace")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    asyncio.run(main_async(a))


if __name__ == "__main__":
    sys.exit(main())
