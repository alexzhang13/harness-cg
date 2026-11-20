"""Re-judge ExpertLongBench rows whose judge call failed (`judge_error`).

    export PRIME_API_KEY=...   # or export RLM_JUDGE_BASE_URL / RLM_JUDGE_API_KEY for another judge endpoint
    python elb_rejudge.py <out>/elb/<tag>.jsonl [...] [--concurrency 8]

Writes <rows>.rejudged.jsonl next to each input, which aggregate.py prefers.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import expertlongbench_real.env as E
from rlm_train.judge_client import build_judge_client


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="+")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    client, model = build_judge_client(model="openai/gpt-4.1-mini")
    score = E._make_score(client, model, max_value_chars=2000)
    sem = asyncio.Semaphore(a.concurrency)

    async def one(r: dict) -> dict:
        if not r.get("judge_error"):
            return r
        info = {"gold_checklist": r.get("gold") or {}, "task": r.get("task"), "item_id": r.get("item_id")}
        state: dict = {"rlm_final_answer": r.get("final_answer") or ""}
        async with sem:
            reward = await score(info, state)
        out = dict(r)
        out.update(reward=float(reward), field_scores=state.get("elb_field_scores"), parsed_fields=state.get("elb_parsed_fields"),
                   judge_error=state.get("elb_judge_error"), rejudged=True)
        return out

    for p in a.rows:
        rows = [json.loads(l) for l in open(p)]
        outs = await asyncio.gather(*(one(r) for r in rows))
        still = sum(1 for o in outs if o.get("judge_error"))
        dst = Path(p).with_suffix(".rejudged.jsonl")
        dst.write_text("".join(json.dumps(o, ensure_ascii=False) + "\n" for o in outs))
        print(f"{p}: {len(outs)} rows -> {dst.name}; rows still with judge_error: {still}")
        if still:
            print("   NOTE: judge still failing; the .rejudged file is not a valid replacement yet", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
