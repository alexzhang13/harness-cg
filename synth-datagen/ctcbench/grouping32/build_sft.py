"""Filter hinted grouping traces into hint-stripped SFT rows.

    python build_sft.py <note.md> <out_dir> <rollout jsonl globs...> [--floor F]

Keeps traces that reach the reward floor, submit within 15 turns, start with a probe sub-call, don't print raw
context and don't mention the hint. Traces that recover from errors are kept.
"""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import argparse, collections, glob, json, os, re, statistics as S, sys
WT = _REPO + "/synth-datagen/common"
sys.path.insert(0, WT)
from rlm_generate_trajectories.sft_format import hint_reference, task_text_of
HERE = os.path.dirname(os.path.abspath(__file__))
REF_VOCAB = set(open(os.path.join(HERE, "..", "train_vocab.txt")).read().split("|"))
MARK = re.compile(r"(?i)strategy_hint|<hint>|</hint>|How to solve this task|## Rules|## Stage")
BLOCK = re.compile(r"```repl\n(.*?)```", re.S)
def asst_msgs(h):
    msgs = [m for m in h if not (m.get("role") == "user" and "<strategy_hint>" in str(m.get("content")))]
    while msgs and msgs[-1].get("role") != "assistant": msgs.pop()
    return msgs
_CTX = {}
def context_grams(r, n=12):
    """12-word spans of the row's rendered context (rebuilt from the stored row id / seed / surface mode)."""
    syn = r.get("synth") or {}; key = (syn.get("source"), syn.get("seed"), syn.get("row_id"), r.get("surface_mode"))
    if key not in _CTX:
        sys.path.insert(0, HERE); import surf_env  # noqa: F401  (sets sys.path)
        import compose
        from ctcbench_synth import surface as SF
        split = str(syn.get("source", "synth-train")).split("-", 1)[1]
        ex = compose.compose_v3(split, syn["row_id"], seed=syn["seed"]); mode = r.get("surface_mode")
        ctx = SF.render(ex, "grouping", mode=mode, seed=(syn["seed"] * 100003 + syn["row_id"]) if mode != "native" else None).context
        w = re.findall(r"\w+", ctx.lower()); _CTX[key] = {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    return _CTX[key]
def lexical_floor(r):
    """Lexical exact-K F1 with no sub-calls (a kept trace must at least match this)."""
    sys.path.insert(0, HERE); import surf_env  # noqa: F401
    import compose, gz_core as G
    from ctcbench_synth import surface as SF
    syn = r.get("synth") or {}; split = str(syn.get("source", "synth-train")).split("-", 1)[1]
    ex = compose.compose_v3(split, syn["row_id"], seed=syn["seed"])
    items = {str(k): (d["title"], d["text"]) for k, d in enumerate(ex["documents"], 1)}
    ids, lex = G.lexical(items); groups = G.assemble(ids, lex, {}, len(ex["gold_doc_indices"]))
    ans = json.dumps({"groups": [{"doc_ids": [int(x) for x in g]} for g in groups]})
    return SF.score(ans, ex, "grouping", {"family": "grouping", "kind": "partition", "native": True, "ids": {}})["primary"]
def raw_print(r, h, n=12):
    """REPL outputs after the turn-1 probe that reproduce >= 12 consecutive words of the context (paper text)."""
    G = context_grams(r, n)
    outs = [str(m.get("content")) for m in h if m.get("role") == "user" and str(m.get("content")).startswith("REPL output")][1:]
    for o in outs:
        w = re.findall(r"\w+", o.lower())
        if any(" ".join(w[i:i + n]) in G for i in range(len(w) - n + 1)): return True
    return False
def recovery(h, sub_log):
    """Count visible recovery events (fixed errors, re-run checks, extra probes)."""
    outs = [str(m.get("content")) for m in h if m.get("role") == "user" and str(m.get("content")).startswith("REPL output")]
    ev = []
    for k, o in enumerate(outs[:-1]):
        if "Traceback" in o or re.search(r"\b\w*Error:", o): ev.append("repl_error")
    def again(pat): return sum(bool(re.search(pat, o)) for o in outs) > 1
    if again(r"items \| K"): ev.append("reparse")
    if again(r"papers in proposed groups per pass"): ev.append("reask_proposals")
    if again(r"groups \(K"): ev.append("recluster")
    probes = [s for s in (sub_log or []) if "[... at character" in str(s.get("prompt", ""))]
    if len(probes) > 1: ev.append("reprobe")
    if any("fine-grained indexer concept" in str(s.get("prompt", "")) and "Below are" not in str(s.get("prompt", ""))[:40] for s in (sub_log or [])):
        ev.append("split_subcall")
    return ev
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("note"); ap.add_argument("out"); ap.add_argument("globs", nargs="+")
    ap.add_argument("--floor", type=float, default=0.0); ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--row-floor", action="store_true", help="also require reward >= the zero-call lexical exact-K F1 on the same row")
    a = ap.parse_args()
    rows = [json.loads(l) for g in a.globs for f in glob.glob(g) for l in open(f)]
    why, keep = collections.Counter(), []
    for r in rows:
        h = r.get("rlm_history") or []; msgs = asst_msgs(h); asst = [m for m in msgs if m["role"] == "assistant"]
        txt = "\n".join(str(m.get("content") or "") for m in asst); blocks = [b for m in asst for b in BLOCK.findall(str(m.get("content") or ""))]
        code = "\n".join(blocks); tt = task_text_of(msgs)
        vm = re.search(r'VOCAB = "([^"]*)"', code); vocab = set(vm.group(1).split("|")) if vm else set()
        loop = any(blocks[i] == blocks[i + 1] == blocks[i + 2] for i in range(len(blocks) - 2))
        sl = r.get("rlm_sub_llm_log") or []
        reason = None
        if r.get("error"): reason = "error"
        elif r.get("stop_condition") != "has_final_answer": reason = "not_submitted"
        elif (r.get("reward") or 0) < a.floor: reason = "below_floor"
        elif a.row_floor and (r.get("reward") or 0) < lexical_floor(r): reason = "below_row_lexical"
        elif (r.get("rlm_iterations") or 99) > a.max_turns: reason = "too_many_turns"
        elif loop: reason = "loop"
        elif "<tool_call>" in txt: reason = "tool_call"
        elif any(re.search(r"print\([^\n]*\bcontext\s*[\[\)]|print\(\s*repr\(\s*context|print\(\s*P\s*\[|print\(\s*P\s*\)", ln)
                 and not re.search(r"ask_about\(|llm_query", ln) for ln in code.splitlines()): reason = "prints_context"
        elif raw_print(r, h): reason = "raw_print"
        elif not asst or "llm_query(" not in str(asst[0].get("content")) or not (sl and "[... at character" in str(sl[0].get("prompt", ""))): reason = "no_turn1_probe"
        elif len(vocab & REF_VOCAB) < 0.99 * len(REF_VOCAB) or len(vocab - REF_VOCAB) > 8: reason = "vocab_mismatch"
        elif any(hint_reference(str(m.get("content") or ""), tt) for m in asst): reason = "hint_reference"
        elif any(MARK.search(str(m.get("content") or "")) for m in msgs): reason = "hint_marker"
        why[reason or "KEPT"] += 1
        if reason: continue
        ev = recovery(h, sl); syn = r.get("synth") or {}
        keep.append({"messages": [{k: v for k, v in m.items() if k in ("role", "content", "reasoning_content")} for m in msgs],
                     "env": "ctc-grouping32-synth-v1", "surface_mode": r.get("surface_mode"), "style": r.get("style"), "id_map": r.get("id_map"),
                     "uid": f"grp32-synth-{syn.get('seed')}-{syn.get('row_id')}-{r.get('surface_mode')}", "task": "grouping",
                     "meta": {"reward": r["reward"], "turns": r.get("rlm_iterations"), "sub_calls": r.get("rlm_sub_llm_calls"),
                              "recovery": ev, "root_chars": len(txt)}, "sub_calls": sl})
    seen, uniq = set(), []
    for k in keep:
        if k["uid"] in seen: why["dup"] += 1; continue
        seen.add(k["uid"]); uniq.append(k)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "kept_traj.jsonl"), "w") as f:
        for k in uniq: f.write(json.dumps({x: y for x, y in k.items() if x != "sub_calls"}) + "\n")
    with open(os.path.join(a.out, "kept_subcalls.jsonl"), "w") as f:
        for k in uniq: f.write(json.dumps({"uid": k["uid"], "sub_calls": k["sub_calls"]}) + "\n")
    tc = collections.Counter(k["meta"]["turns"] for k in uniq)
    st = {"rollouts": len(rows), "kept": len(uniq), "floor": a.floor, "reasons": dict(why),
          "turns_hist": dict(sorted(tc.items())), "recovery_share": (sum(bool(k["meta"]["recovery"]) for k in uniq) / len(uniq)) if uniq else 0,
          "recovery_events": dict(collections.Counter(e for k in uniq for e in k["meta"]["recovery"])),
          "kept_reward_mean": S.mean(k["meta"]["reward"] for k in uniq) if uniq else 0,
          "reward_mean_all": S.mean((r.get("reward") or 0) for r in rows) if rows else 0}
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1); print(json.dumps(st, indent=1))
if __name__ == "__main__": main()
