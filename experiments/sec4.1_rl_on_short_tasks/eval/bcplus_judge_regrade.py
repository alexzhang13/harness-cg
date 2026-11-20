"""BrowseComp-Plus answer-judge regrade (used for the Qwen3.8-27B rows of Table main-composition).

    export PRIME_API_KEY=...              # judge routing (Prime Inference), or RLM_JUDGE_BASE_URL / _API_KEY
    python bcplus_judge_regrade.py <OUT> [--concurrency 16]

Asks gpt-4.1-mini the env's answer_judge prompt for every row with a non-empty answer and substring score < 1, and
caches the verdicts in <OUT>/bcplus/judge_cache.json. `aggregate.py --bcplus judged` raises a "yes" row to 1.
"""
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

import browsecomp_plus_real.env as E
from rlm_train.judge_client import build_judge_client

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate  # noqa: E402


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dead-as-zero", action="store_true", help="same selection flag as aggregate.py")
    a = ap.parse_args()
    sel = aggregate.select_run_eval(a.out / "bcplus" / "cache", "bcplus", a.dead_as_zero)
    cache_path = a.out / "bcplus" / "judge_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client, model = build_judge_client(model="openai/gpt-4.1-mini")
    sem = asyncio.Semaphore(a.concurrency)

    async def judge(k, row):
        r = row.rec
        final, gold = r.get("final_answer"), r.get("answer")
        if str(k) in cache or row.reward >= 1.0 or not str(final or "").strip() or not gold:
            return
        q = str(r.get("root_prompt", "")).split("\n\n" + E._ROOT_PROMPT_SUFFIX)[0]
        prompt = E._ANSWER_JUDGE_PROMPT.format(question=q, response=str(final)[:8000], correct_answer=gold)
        async with sem:
            try:
                resp = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], max_completion_tokens=16)
                cache[str(k)] = (resp.choices[0].message.content or "").strip().lower()[:8]
            except Exception as e:
                print("judge error", k, repr(e)[:100])

    await asyncio.gather(*(judge(k, row) for k, row in sel.valid.items()))
    cache_path.write_text(json.dumps(cache, indent=1))
    rows = list(sel.valid.values())
    if not rows:
        print("no valid BC+ rows")
        return
    env = [r.reward for r in rows]
    strict = [aggregate.strict_bcplus(r.reward, r.rec.get("final_answer"), r.rec.get("answer")) for r in rows]
    judged = [1.0 if (r.reward >= 1.0 or str(cache.get(str(r.key), "")).startswith("yes")) else r.reward for r in rows]
    print(f"n={len(rows)} invalid={len(sel.invalid)}  env={statistics.mean(env):.3f}  strict={statistics.mean(strict):.3f}  "
          f"judged={statistics.mean(judged):.3f}  upgrades={sum(1 for r in rows if r.reward < 1 and str(cache.get(str(r.key), '')).startswith('yes'))}")


if __name__ == "__main__":
    asyncio.run(main())
