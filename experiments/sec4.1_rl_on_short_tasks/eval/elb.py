"""ExpertLongBench held-out eval on one shard of rows, scored by the env's checklist judge (gpt-4.1-mini).

    python elb.py --task T01LegalMDS --ids 0-24 --base-url <url>/v1 --model <served name> --tag <tag> --out-dir <dir>

Orchestrator prompt, 40 turns, T 0, root 8192 tokens, thinking off. Writes <out-dir>/<tag>.jsonl when the shard
finishes (no resume inside a shard).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient


def _ids(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["T01LegalMDS", "T03MaterialSEG", "T04EduPAE", "T06HealthCNG"])
    ap.add_argument("--ids", required=True, help="e.g. 0-24 or 3,5,9")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="served model name (root and sub-calls)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-iterations", type=int, default=40)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    import expertlongbench_real as E

    harness = {"orchestrator_prompt": True, "min_iterations": 2}
    env = E.load_environment(task=a.task, num_examples=-1, max_iterations=a.max_iterations,
                             sub_extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **harness)
    ids = _ids(a.ids)
    sub = OpenAIChatCompletionsClient(AsyncOpenAI(api_key="EMPTY", base_url=a.base_url, timeout=1800, max_retries=2))
    t0 = time.time()

    async def go() -> list[dict]:
        sem = asyncio.Semaphore(a.concurrency)

        async def one(i: int) -> dict:
            async with sem:
                try:
                    return await env._run_rollout_state(input=dict(env.dataset[i]), client=sub, model=a.model,
                                                        sampling_args={
                        "max_tokens": 8192, "temperature": 0.0,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
                except Exception as e:  # noqa: BLE001
                    return {"error": repr(e)[:500]}

        return list(await asyncio.gather(*(one(i) for i in ids)))

    try:
        states = asyncio.run(go())
    finally:
        try:
            asyncio.run(env.teardown_rlm())
        except Exception:  # noqa: BLE001
            pass

    a.out_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, s in zip(ids, states, strict=True):
        info = json.loads(env.dataset[i]["info"])
        log = s.get("rlm_sub_llm_log") or []
        recs.append({
            "task": a.task, "example_id": i, "item_id": info.get("item_id"),
            "reward": float(s.get("reward") or 0.0), "field_scores": s.get("elb_field_scores"),
            "parsed_fields": s.get("elb_parsed_fields"), "judge_error": s.get("elb_judge_error"),
            "context_chars": len(info["context"]), "sub_calls": int(s.get("rlm_sub_llm_calls") or 0),
            "iterations": int(s.get("rlm_iterations") or 0), "final_answer": s.get("rlm_final_answer"),
            "gold": info.get("gold_checklist"), "error": s.get("error") and str(s.get("error"))[:300],
            "repl_tail": [str(m.get("content") if isinstance(m, dict) else m)[:6000] for m in (s.get("rlm_history") or [])],
            "sub_log": [{"prompt_chars": e.get("prompt_chars"), "prompt_head": (e.get("prompt") or "")[:400],
                         "response": (e.get("response") or "")[:8000]} for e in log[:120]],
        })

        def _m(m):
            if not isinstance(m, dict):
                m = {"role": getattr(m, "role", "?"), "content": getattr(m, "content", str(m)),
                     "reasoning_content": getattr(m, "reasoning_content", None)}
            c = m.get("content")
            return {"role": m.get("role"), "content": c if isinstance(c, str) else json.dumps(c, default=str),
                    "reasoning_content": m.get("reasoning_content") or ""}
        recs[-1]["messages"] = [_m(m) for m in (s.get("rlm_history") or [])]
        recs[-1]["stop_condition"] = s.get("stop_condition")
        recs[-1]["subcalls"] = log
    with open(a.out_dir / f"{a.tag}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    for r in recs:
        print(f"[{a.task} eid={r['example_id']}] reward={r['reward']:.3f} subs={r['sub_calls']} iters={r['iterations']} "
              f"chars={r['context_chars']} parsed={r['parsed_fields']} err={r['error'] or r['judge_error']}")
    print(json.dumps({"task": a.task, "tag": a.tag, "n": len(recs),
                      "reward_mean": round(statistics.mean(r["reward"] for r in recs), 4) if recs else 0.0,
                      "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    sys.exit(main())
