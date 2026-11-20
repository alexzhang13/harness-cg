"""Teacher levels (generate_teacher_levels.py) -> SFT samples {messages, pool_ids, depth, n, trunc}.

    python build_sft.py runs/teacher_levels runs/sft_gated.jsonl [--stop-share 1.0]

Keeps levels that passed the gate or stopped cleanly. Each kept level gives the whole dialogue plus copies cut after
each unlock turn and after the hand-off turn.
"""
import json, random, sys, collections

def build(rows, stop_share=1.0, seed=0, stop_dup=3, retry_dup=2):
    rng = random.Random(seed); out = []; kept = collections.Counter()
    for r in rows:
        m = r["messages"]
        if len(m) < 4 or m[-1]["role"] != "assistant" or 'answer["ready"] = True' not in m[-1]["content"]: kept["dropped_no_submit"] += 1; continue
        a = [i for i, x in enumerate(m) if x["role"] == "assistant"]
        unlock_i = [i for i in a if "result = unlock(" in m[i]["content"]]
        if not unlock_i: kept["dropped_no_unlock"] += 1; continue
        if any('answer["ready"]' in m[i]["content"] for i in a[:-1]): kept["dropped_early_submit"] += 1; continue
        if r.get("literal_fallback"): kept["dropped_literal_fallback"] += 1; continue
        args = r.get("submitted") or []
        if len(args) != len(set(args)): kept["dropped_resubmitted_same_answer"] += 1; continue
        if not r.get("accepted"):
            if rng.random() > stop_share: kept["dropped_exhausted"] += 1; continue
            kept["clean_stop"] += 1; dup = stop_dup
        else:
            kept["accepted"] += 1; dup = retry_dup if r.get("attempts") == 2 else 1
        base = {"pool_ids": r["pool_ids"], "depth": r["depth"], "n": r["n"]}
        for _ in range(dup): out.append({"messages": m, **base, "trunc": 0})
        for i in unlock_i:
            if i < len(m) - 1: out.append({"messages": m[:i + 1], **base, "trunc": 1})
        for i in a:
            if "handoff(result)" in m[i]["content"] and i < len(m) - 1: out.append({"messages": m[:i + 1], **base, "trunc": 2})
    return out, kept

if __name__ == "__main__":
    src, dst = sys.argv[1:3]; share = float(sys.argv[sys.argv.index("--stop-share") + 1]) if "--stop-share" in sys.argv else 1.0
    rows = [json.loads(l) for l in open(f"{src}/levels.jsonl")]
    out, kept = build(rows, share)
    with open(dst, "w") as f:
        for o in out: f.write(json.dumps(o) + "\n")
    print(f"levels {len(rows)} -> samples {len(out)}; {dict(kept)}")
