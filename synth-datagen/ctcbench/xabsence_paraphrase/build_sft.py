"""Filter hinted xabsence_paraphrase traces into hint-stripped SFT rows.

    python build_sft.py <note.md> <out_dir> <rollout jsonl globs...>

Keeps exact, submitted traces within MAX_TURNS that run the three program blocks in order, with no errors and no
hint references."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import argparse, collections, glob, json, os, re, statistics as S, sys
WT = _REPO + "/synth-datagen/common"
sys.path.insert(0, WT)
from rlm_generate_trajectories.sft_format import hint_reference, task_text_of
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contam import contaminated
ap = argparse.ArgumentParser(); ap.add_argument("note"); ap.add_argument("out"); ap.add_argument("globs", nargs="+")
ap.add_argument("--floor", type=float, default=0.999); ap.add_argument("--max-turns", type=int, default=5)
ap.add_argument("--min-cov", type=float, default=0.8)
ap.add_argument("--sig", default="llm_query(|fmt = lambda|weak = sorted(|def scan(", help="|-separated program-order signatures")
ap.add_argument("--loop-n", type=int, default=3, help="a block or error repeated this many times = a loop")
ap.add_argument("--max-subs", type=int, default=0, help="drop rollouts with more sub-calls than this (0 = no limit)")
ap.add_argument("--scan23", action="store_true", help="drop rollouts with any 13-gram hit vs all 23 ctc eval families")
a = ap.parse_args()
NOTE = open(a.note).read()
def grams(t, n=6):
    w = re.findall(r"\S+", t); return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
NG = grams("\n".join(re.findall(r"```repl\n(.*?)```", NOTE, re.S)))
SIG = a.sig.split("|")
MARK = re.compile(r"(?i)strategy_hint|<hint>|</hint>|How to solve this task|## Rules")
rows = [json.loads(l) for g in a.globs for f in glob.glob(g) for l in open(f)]
def _errs(h):
    return [re.findall(r"(\w+Error: [^\n]{0,120})|(CHECK FAILED[^\n]{0,120})", str(m.get("content")))
            for m in h if m.get("role") == "user" and str(m.get("content")).startswith("REPL output")]
def loop(asst, h):
    codes = [c.strip() for m in asst for c in re.findall(r"```repl\n(.*?)```", str(m.get("content")), re.S)]
    if codes and max(collections.Counter(codes).values()) >= a.loop_n:
        return True
    msgs = [x for e in _errs(h) for t in e for x in t if x]
    return any(v >= a.loop_n for v in collections.Counter(msgs).values())
SUBCALL = r"\b(scan|llm_query|llm_query_batched|pick|yes_no|look)\("
OVER_ALL = r"for\s+\w+(?:\s*,\s*\w+)?\s+in\s+(?:C|list\(C\)|C\.items\(\)|C\.keys\(\)|context\.split\([^)]*\)|lines)\b"
def runaway(code):
    """a sub-call evaluated once per claim: inside a comprehension / for-loop over ALL items (C, its keys/items, lines)"""
    lines = code.splitlines()
    for k, ln in enumerate(lines):
        if not re.search(OVER_ALL, ln):
            continue
        if re.search(SUBCALL, ln):
            return True
        if ln.rstrip().endswith(":"):
            ind = len(ln) - len(ln.lstrip())
            for b in lines[k + 1:]:
                if b.strip() and len(b) - len(b.lstrip()) <= ind:
                    break
                if re.search(SUBCALL, b):
                    return True
    return False


def recovered(h):
    return any("CHECK FAILED" in str(m.get("content")) or "Traceback" in str(m.get("content"))
               for m in h if m.get("role") == "user" and str(m.get("content")).startswith("REPL output"))
why, keep = collections.Counter(), []
for r in rows:
    h = r.get("rlm_history") or []
    msgs = [m for m in h if not (m.get("role") == "user" and "<strategy_hint>" in str(m.get("content")))]
    while msgs and msgs[-1].get("role") != "assistant":
        msgs.pop()
    asst = [m for m in msgs if m["role"] == "assistant"]
    txt = "\n".join(str(m.get("content") or "") for m in asst)
    code = "\n".join(re.findall(r"```repl\n(.*?)```", txt, re.S))
    g = grams(code); cov = len(g & NG) / len(g) if g else 0
    def _first(rx):
        m = re.search(rx, txt); return m.start() if m else -1
    p_probe, p_parse = _first(r"llm_query\("), _first(r"fmt\s*=\s*lambda|def fmt\(|\bC\s*=\s*\{|answer\[\"content\"\]\s*=")
    later = [x for x in (_first(r"weak\s*=\s*sorted\(|yes_no\("), _first(r"def scan\(|scan\(")) if x >= 0]
    pos = [p_probe, p_parse] + ([min(later)] if later else [-1])
    repl_out = "\n".join(str(m.get("content")) for m in h if m.get("role") == "user" and str(m.get("content")).startswith("REPL output"))
    tt = task_text_of(msgs)
    reason = (None if True else None)
    if r.get("error"): reason = "error"
    elif (r.get("reward") or 0) < a.floor: reason = "not_exact"
    elif r.get("stop_condition") != "has_final_answer": reason = "not_submitted"
    elif (r.get("rlm_iterations") or 99) > a.max_turns: reason = "too_many_turns"
    elif "<tool_call>" in txt: reason = "tool_call"
    elif a.max_subs and (r.get("rlm_sub_llm_calls") or 0) > a.max_subs: reason = "too_many_subcalls"
    elif runaway(code): reason = "subcall_per_claim"
    elif loop(asst, h): reason = "loop"
    elif re.search(r"print\([^\n]*\b(context|lines)\s*\[|print\(\s*(repr\(\s*)?(context|line|lines)\s*[\)\[,]", code): reason = "prints_context"
    elif "llm_query(" not in str(asst[0].get("content")) or not (r.get("rlm_sub_llm_log") or [{}])[0].get("prompt", "").startswith("Samples of a text"):
        reason = "no_turn1_probe"
    elif min(pos) < 0 or pos != sorted(pos): reason = "program_order"
    elif cov < a.min_cov: reason = "low_note_coverage"
    elif any(hint_reference(str(m.get("content") or ""), tt) for m in asst): reason = "hint_reference"
    elif any(MARK.search(str(m.get("content") or "")) for m in msgs): reason = "hint_marker"
    elif contaminated(h) or contaminated(r.get("rlm_sub_llm_log") or []): reason = "dropped_abstract_sentence"
    elif a.scan23 and __import__("scan23").is_hit(r): reason = "eval_13gram_hit_23fam"
    why[reason or "KEPT"] += 1
    if reason: continue
    keep.append({"messages": [{k: v for k, v in m.items() if k in ("role", "content", "reasoning_content")} for m in msgs],
                 "env": "ctc-xabsence-paraphrase-synth-v51", "surface_mode": r.get("surface_mode"), "style": r.get("style"), "id_map": r.get("id_map"), "uid": (("xabpara-surf-%s-%s" % (r["synth"].get("seed"), r["synth"].get("row_id"))) if (r.get("synth") or {}).get("row_id") is not None else
                         "xabpara-synth-%s-%s" % ((r.get("synth") or {}).get("synth_seed"), json.loads((r.get("synth") or {}).get("synth_params") or "{}").get("row"))),
                 "task": "xabsence_paraphrase", "meta": {"reward": r["reward"], "turns": r.get("rlm_iterations"),
                 "sub_calls": r.get("rlm_sub_llm_calls"), "root_chars": len(txt), "note_cov": round(cov, 3),
                 "recovery": recovered(h)},
                 "sub_calls": r.get("rlm_sub_llm_log")})
seen, uniq = set(), []
for k in keep:
    key = k["uid"]
    if key in seen: why["dup"] += 1; continue
    seen.add(key); uniq.append(k)
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "kept_traj.jsonl"), "w") as f:
    for k in uniq: f.write(json.dumps({x: y for x, y in k.items() if x != "sub_calls"}) + "\n")
with open(os.path.join(a.out, "kept_subcalls.jsonl"), "w") as f:
    for k in uniq: f.write(json.dumps({"uid": k["uid"], "sub_calls": k["sub_calls"]}) + "\n")
st = {"rollouts": len(rows), "kept": len(uniq), "reasons": dict(why),
      "kept_profile": {m: S.median(k["meta"][m] for k in uniq) for m in ("turns", "sub_calls", "root_chars", "note_cov")} if uniq else {},
      "reward_mean_all": S.mean((r.get("reward") or 0) for r in rows) if rows else 0,
      "kept_turns_hist": dict(sorted(collections.Counter(k["meta"]["turns"] for k in uniq).items())),
      "kept_recovery_share": round(S.mean(k["meta"]["recovery"] for k in uniq), 3) if uniq else 0,
      "kept_by_mode": dict(collections.Counter("native" if k.get("surface_mode") == "native" else "varied" for k in uniq))}
json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1); print(json.dumps(st, indent=1))
