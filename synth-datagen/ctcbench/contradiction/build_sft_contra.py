"""Filter hinted contradiction_iid 1M traces into hint-stripped SFT rows.

    python build_sft_contra.py <out_dir> <rollout jsonl globs...> [--floor F]

Keeps traces with F1 >= floor that submit within 15 turns, start with a probe sub-call, don't loop, print raw text or
mention the hint. Traces that recover from errors are kept and their recovery events labelled.
"""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import argparse, collections, glob, json, os, re, statistics as S, sys
WT = _REPO + "/synth-datagen/common"
sys.path.insert(0, WT)
from rlm_generate_trajectories.sft_format import hint_reference, task_text_of
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("globs", nargs="+")
ap.add_argument("--floor", type=float, default=0.25); ap.add_argument("--max-turns", type=int, default=15)
a = ap.parse_args()
MARK = re.compile(r"(?i)strategy_hint|<hint>|</hint>|How to solve this task|## Rules|\bStage \d")
PRINT_TEXT = re.compile(r"print\([^\n]*(\bcontext\s*[\[\)]|\bC\[ids?\[|\bC\[[a-z_]+\]\s*[,)]|repr\(\s*context)")
rows = [json.loads(l) for g in a.globs for f in glob.glob(g) for l in open(f)]
why, keep, rec_ct = collections.Counter(), [], collections.Counter()
for r in rows:
    h = r.get("rlm_history") or []
    msgs = [m for m in h if not (m.get("role") == "user" and "<strategy_hint>" in str(m.get("content")))]
    while msgs and msgs[-1].get("role") != "assistant":
        msgs.pop()
    asst = [m for m in msgs if m["role"] == "assistant"]
    txt = "\n".join(str(m.get("content") or "") for m in asst)
    blocks = [re.findall(r"```repl\n(.*?)```", str(m.get("content") or ""), re.S) for m in asst]
    code = "\n".join(c for b in blocks for c in b)
    outs = [str(m.get("content")) for m in h if m.get("role") == "user" and str(m.get("content")).startswith("REPL output")]
    subs = r.get("rlm_sub_llm_log") or []
    probes = [s for s in subs if str(s.get("prompt", "")).startswith("Samples of")]
    ev = set()
    tb = [i for i, o in enumerate(outs) if "Traceback" in o]
    if tb and any("Traceback" not in o for o in outs[tb[0] + 1:]): ev.add("traceback_fixed")
    if len(probes) > 1: ev.add("reprobe")
    if code.count("C = {}") > 1 or len(re.findall(r"CHECK parse", "\n".join(outs))) > 1: ev.add("parse_fix")
    if re.search(r"(failed|missing|retry|re-?ask|again)", "\n".join(c for b in blocks[1:] for c in b), re.I) and \
       len(re.findall(r"llm_query_batched\(", code)) > 4: ev.add("rerun_failed")
    loops = max(collections.Counter(str(m.get("content")) for m in asst).values()) if asst else 0
    tt = task_text_of(msgs)
    reason = None
    if r.get("error"): reason = "error"
    elif r.get("stop_condition") != "has_final_answer": reason = "not_submitted"
    elif (r.get("reward") or 0) < a.floor: reason = "below_floor"
    elif len(asst) > a.max_turns: reason = "too_many_turns"
    elif loops >= 3: reason = "loop"
    elif "<tool_call>" in txt: reason = "tool_call"
    elif not asst or "llm_query(" not in str(asst[0].get("content")) or not subs or not str(subs[0].get("prompt", "")).startswith("Samples of"):
        reason = "no_turn1_probe"
    elif PRINT_TEXT.search(code): reason = "prints_text"
    elif any(hint_reference(str(m.get("content") or ""), tt) for m in asst): reason = "hint_reference"
    elif any(MARK.search(str(m.get("content") or "")) for m in msgs): reason = "hint_marker"
    why[reason or "KEPT"] += 1
    if reason: continue
    for e in ev: rec_ct[e] += 1
    syn = r.get("synth") or {}
    keep.append({"messages": [{k: v for k, v in m.items() if k in ("role", "content", "reasoning_content")} for m in msgs],
                 "env": "ctc-contradiction-iid-1m-synth-v2", "task": "contradiction_iid", "surface_mode": r.get("surface_mode"),
                 "style": r.get("style"), "id_map": r.get("id_map"), "uid": f"contra1m-{syn.get('split')}-{syn.get('seed')}-{syn.get('row_id')}",
                 "meta": {"reward": r["reward"], "turns": len(asst), "sub_calls": r.get("rlm_sub_llm_calls"),
                          "recovery_events": sorted(ev)}, "sub_calls": subs})
seen, uniq = set(), []
for k in keep:
    if k["uid"] in seen: why["dup"] += 1; continue
    seen.add(k["uid"]); uniq.append(k)
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "kept_traj.jsonl"), "w") as f:
    for k in uniq: f.write(json.dumps({x: y for x, y in k.items() if x != "sub_calls"}) + "\n")
with open(os.path.join(a.out, "kept_subcalls.jsonl"), "w") as f:
    for k in uniq: f.write(json.dumps({"uid": k["uid"], "sub_calls": k["sub_calls"]}) + "\n")
st = {"rollouts": len(rows), "kept": len(uniq), "reasons": dict(why), "floor": a.floor,
      "turns_hist": dict(sorted(collections.Counter(k["meta"]["turns"] for k in uniq).items())),
      "kept_with_recovery": sum(1 for k in uniq if k["meta"]["recovery_events"]), "recovery_events": dict(rec_ct),
      "reward_mean_all": S.mean((r.get("reward") or 0) for r in rows) if rows else 0,
      "reward_mean_kept": S.mean(k["meta"]["reward"] for k in uniq) if uniq else 0}
json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1); print(json.dumps(st, indent=1))
