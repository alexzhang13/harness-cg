"""Score each pool task on its own (the single-task numbers quoted with Table recursion-composition).

    python run_solo.py --base-url URL --model M --reps 3 --temperature 0 --sub-thinking --out runs/solo/M

Each task in an rlm.RLM (depth 1, 30 turns); one JSON row per task and repetition.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks
from rlm import RLM
from tasks import say


def rollout(pool_id, a, rep):
    it = tasks.item(pool_id)
    kw = dict(backend="openai", backend_kwargs={"model_name": a.model, "base_url": a.base_url, "api_key": "x", "timeout": 3600,
              "sampling_args": {"temperature": a.temperature, "max_tokens": a.max_tokens,
                                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}},
              environment="local", max_depth=1, max_iterations=a.max_iterations)
    if a.sub_thinking:
        kw["sub_sampling_args"] = {"max_tokens": 24576, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    t0 = time.time(); err = resp = None
    try:
        resp = RLM(**kw).completion(it.get("context_docs") or it["context"], root_prompt=it["question"]).response
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    score = float(it["_mod"].score(it, resp)) if resp else 0.0
    return {"pool_id": pool_id, "env": it["env"], "rep": rep, "answer": resp, "score": score, "error": err, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-iterations", type=int, default=30); ap.add_argument("--sub-thinking", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only the first N pool tasks"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ids = list(range(len(tasks.POOL)))[: a.limit or None]
    fn = os.path.join(a.out, f"solo_{a.model}_{int(time.time())}.jsonl")
    jobs = [(p, r) for p in ids for r in range(a.reps)]
    say(len(jobs), "rollouts ->", fn)
    scores = []
    with ProcessPoolExecutor(a.workers) as ex, open(fn, "a") as f:
        futs = {ex.submit(rollout, p, a, r): (p, r) for p, r in jobs}
        for fu in as_completed(futs):
            r = fu.result(); f.write(json.dumps(r) + "\n"); f.flush(); scores.append(r["score"])
            say(f"task {r['pool_id']:3d} rep {r['rep']} {r['env']:10s} score={r['score']:.2f} {r['seconds']}s {r['error'] or ''}")
    say(f"mean {sum(scores) / max(1, len(scores)):.3f} over {len(scores)} rollouts")


if __name__ == "__main__":
    main()
