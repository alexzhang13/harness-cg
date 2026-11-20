"""PROCESS.md section 7 filters for hinted outlier-1M traces (real root + real sub-LM, generator-built rows).
python build_sft.py <out_dir> <rollouts.jsonl> [...]  ->  <out_dir>/kept.jsonl (build_turn_mix input), report.json
"""
import collections, json, re, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "synth-datagen/common"))
from rlm_generate_trajectories.sft_format import HINT_REFERENCE_RE  # noqa: E402

NOTE = (ROOT / "synth-datagen/ctcbench/notes/ctc-outlier-1m-k-train.md").read_text()
EXTRA_LEAK = re.compile(r"strategy_hint|\b(the|this|my) (note|hint|guide|instructions?)\b|\bas (told|given|provided) (above|in)\b|"
                        r"\bturn[- ][123]\b(?!/)|\bthe (strategy|plan|method|procedure|recipe)\b|\b(following|follow) the\b|"
                        r"\b(run|running|execute|executing) the (turn|block|strategy|program)\b|"
                        r"\bthe (provided|given) (code|program|block)\b", re.I)
CODE = re.compile(r"```repl\n(.*?)```", re.S)
SIZE_PRIOR = re.compile(r"\b(3|three)\s+(windows?|passages?|items?|excerpts?|segments?|chunks?)\b|exactly\s+(3|three)\b|"
                        r"combinations\([^)]*,\s*3\s*\)|len\([^)]*\)\s*==\s*3\b|==\s*3\s*\]|size\s*(==|of)\s*3\b|triples?\b|\btriplets?\b", re.I)
RAW_PRINT = re.compile(r"print\(\s*context\s*\[|print\(\s*context\s*\)|context\[\s*:\s*\d+\s*\]\s*\)")


def grams(text, n=6):
    w = re.findall(r"\w+", text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


NOTE_G = grams("\n".join(CODE.findall(NOTE)))


def judge(rec):
    why = []
    hist = rec.get("history") or []
    msgs = [m for m in hist if not (m.get("role") == "user" and "<strategy_hint>" in (m.get("content") or ""))]
    asst = [m.get("content") or "" for m in msgs if m.get("role") == "assistant"]
    if rec.get("error"):
        why.append("error")
    if rec.get("reward", 0) < 1.0:
        why.append("not_exact")
    if rec.get("stop") != "has_final_answer":
        why.append("not_submitted")
    if not asst or len(asst) > 15:
        why.append("turns")
    allt = "\n".join(asst)
    if "<tool_call>" in allt:
        why.append("tool_call")
    code = "\n".join(c for a in asst for c in CODE.findall(a))
    first_code = next((c for a in asst for c in CODE.findall(a)), "")
    if "llm_query" not in first_code:
        why.append("no_turn1_probe")
    vis = "\n".join(asst) + "\n" + "\n".join(m.get("content") or "" for m in msgs if m.get("role") == "user" and (m.get("content") or "").startswith("REPL"))
    if SIZE_PRIOR.search(vis):
        why.append("size_prior_visible")
    if RAW_PRINT.search(code):
        why.append("raw_context_print")
    stages = [next((k for k, a in enumerate(asst) if s in a), None) for s in ("NAME = ", "GROUP = ", "RECON = ")]
    if None in stages or stages != sorted(stages):
        why.append("stage_order")
    cov = len(grams(code) & NOTE_G) / max(1, len(grams(code)))
    if cov < .3:
        why.append("low_coverage")
    tb = sum((m.get("content") or "").count("Traceback") for m in msgs if m.get("role") == "user")
    blocks = [c.strip() for a in asst for c in CODE.findall(a)]
    if max(collections.Counter(blocks).values() or [0]) >= 3 or tb >= 4:
        why.append("loop")
    recovery = ("investigate(" in code) + (sum("llm_query(f\"\"\"Below are a few slices" in c or "slices taken from" in c for c in blocks) > 1) + \
        (tb >= 1 and rec.get("reward", 0) >= 1)
    prose = [len(CODE.sub("", a).strip()) for a in asst if CODE.search(a)]
    if not prose or min(prose) < 40:
        why.append("no_rationale")
    for a in asst:
        txt = CODE.sub("", a)
        if HINT_REFERENCE_RE.search(txt) or EXTRA_LEAK.search(txt) or "strategy_hint" in a:
            why.append("hint_reference"); break
    return why, msgs, {"turns": len(asst), "root_chars": sum(map(len, asst)), "coverage": round(cov, 3),
                       "sub_calls": rec.get("sub_calls"), "tracebacks": tb, "recovery": int(recovery > 0)}


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for p in sys.argv[2:] for l in open(p)]
    kept, reasons, prof_all, prof_kept, seen = [], collections.Counter(), [], [], collections.Counter()
    for rec in recs:
        why, msgs, prof = judge(rec)
        prof_all.append(prof)
        key = (rec["gen_row"]["source"], rec["gen_row"]["row"])
        if not why and seen[key] >= 1:
            why = ["dup_row"]
        for w in why:
            reasons[w] += 1
        if why:
            continue
        seen[key] += 1
        prof_kept.append(prof)
        kept.append({"messages": msgs, "env": "ctc-outlier-1m-k-train", "uid": f"{Path(key[0]).name}#{key[1]}#r{rec['r']}",
                     "task": "ctc-outlier-1m", "meta": {"gen_row": rec["gen_row"], "surface": rec.get("surface"), "steering": "gold-outcome-filter", **prof}})
    with open(out / "kept.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    agg = lambda P, k: round(statistics.mean(p[k] for p in P), 1) if P else None
    rep = {"rollouts": len(recs), "kept": len(kept), "yield": round(len(kept) / max(1, len(recs)), 3),
           "exact": sum(1 for r in recs if r.get("reward", 0) >= 1), "mean_reward": round(statistics.mean(r.get("reward", 0) for r in recs), 3) if recs else None,
           "reject_reasons": dict(reasons.most_common()),
           "all": {k: agg(prof_all, k) for k in ("turns", "root_chars", "coverage", "sub_calls", "recovery")},
           "kept_turns": dict(sorted(collections.Counter(p["turns"] for p in prof_kept).items())),
           "kept_profile": {k: agg(prof_kept, k) for k in ("turns", "root_chars", "coverage", "sub_calls", "recovery")}}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
