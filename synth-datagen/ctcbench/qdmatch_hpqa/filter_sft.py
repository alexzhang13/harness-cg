"""PROCESS.md section 7 filters for hinted qdmatch_hpqa traces -> kept whole trajectories (note stripped)."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(_REPO)
sys.path[:0] = [str(ROOT / "synth-datagen/common"), str(Path(__file__).resolve().parent)]
import importlib  # noqa: E402
import os  # noqa: E402
_NT = importlib.import_module(os.environ.get("QD_TURNS", "note_turns_v4"))
TURNS = list(_NT.TURNS) + list(getattr(_NT, "FIXES", {}).values())
HEADER = re.compile(r"(?m)^\[\d+\] (?:Query|Document):")
_TOK = None


def ntok(s: str) -> int:
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
        except Exception:  # noqa: BLE001
            _TOK = False
    return len(_TOK.encode(s)) if _TOK else len(s) // 4
from rlm_generate_trajectories.sft_format import HINT_REFERENCE_RE  # noqa: E402

STAGES = [r"hs\s*=\s*list\(HDR", r"\bfound\b[^=\n]*=\s*screen\(", r"\bextra\b[^=\n]*=\s*screen\(", r"chains\s*=\s*\{\}", r'answer\["content"\]\s*=\s*final']
PROSE_LEAK = re.compile(r"\b(note|hint|instruction|instructions|as written|copy (?:the|this) block|the rules?)\b", re.I)
FENCE = re.compile(r"```.*?(?:```|$)", re.S)


def grams(text: str, n: int = 6) -> set:
    t = re.findall(r"\w+|[^\w\s]", text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


NOTE_GRAMS = grams("\n".join(TURNS))


def is_prologue(m: dict) -> bool:
    c = m.get("content") or ""
    return m.get("role") == "user" and ("<strategy_hint>" in c or "Harness facts that decide" in c)


def check(rec: dict, floor: float, max_turns: int) -> tuple[str | None, dict]:
    msgs = rec.get("messages") or []
    asst = [m.get("content") or "" for m in msgs if m.get("role") == "assistant"]
    users = [m.get("content") or "" for m in msgs if m.get("role") == "user"]
    code = "\n".join("\n".join(re.findall(r"```repl\n(.*?)```", a, re.S)) for a in asst)
    prose = "\n".join(FENCE.sub("", a) for a in asst)
    pos = [next((k for k, a in enumerate(asst) if re.search(s, a)), None) for s in STAGES]
    order_ok = all(p is not None for p in pos) and pos == sorted(pos)
    cov = len(grams(code) & NOTE_GRAMS) / max(1, len(NOTE_GRAMS))
    code1 = "\n".join(re.findall(r"```repl\n(.*?)```", asst[0], re.S)) if asst else ""
    probe_t1 = bool(re.search(r"llm_query(?:_batched)?\(", code1)) and "context[" in code1
    repl_out = [u for u in users if u.startswith("REPL output")]
    raw_print = bool(re.search(r"print\([^)\n]*\bcontext\b", code)) or any(len(HEADER.findall(u)) >= 3 for u in repl_out)
    outs_by_turn = {k: m.get("content") or "" for k, m in enumerate(msgs)}
    rec_ev, prev_out, unmotivated = 0, "", 0
    parses = 0
    for m in msgs:
        c = m.get("content") or ""
        if m.get("role") == "user" and c.startswith("REPL output"):
            prev_out = c
        elif m.get("role") == "assistant":
            code_t = "\n".join(re.findall(r"```repl\n(.*?)```", c, re.S))
            parses += bool(re.search(r"hs\s*=\s*list\(HDR", code_t))
            is_fix = bool(re.search(r"should match the START|CHECK (screen|verify|checks) redo", code_t)) or \
                (parses >= 2 and re.search(r"hs\s*=\s*list\(HDR", code_t)) or ("Traceback" in prev_out and code_t)
            evid = ("PROBLEM" in prev_out or "Traceback" in prev_out or
                    bool(re.search(r"unparseable replies [1-9]|still unparseable [1-9]|'(?:unparseable|bad[ _]ids|quote[^']*|format[^']*)': [1-9]", prev_out)))
            if is_fix and evid:
                rec_ev += 1
            elif is_fix and not re.search(r"hs\s*=\s*list\(HDR", code_t):
                unmotivated += 1
    blocks = [re.sub(r"\s+", " ", b) for a in asst for b in re.findall(r"```repl\n(.*?)```", a, re.S)]
    errs = [u.strip().splitlines()[-1] for u in users if "Traceback" in u]
    loop = (max(collections.Counter(blocks).values(), default=0) >= 3) or (max(collections.Counter(errs).values(), default=0) >= 3)
    prof = {"turns": len(asst), "recovery_events": rec_ev, "unmotivated_repairs": unmotivated, "loop": loop, "sub_calls": rec.get("sub_calls", 0), "root_chars": sum(map(len, asst)),
            "root_tokens": sum(ntok(a) for a in asst), "tokens_per_turn": sum(ntok(a) for a in asst) / max(1, len(asst)),
            "repl_out_tokens": sum(ntok(u) for u in repl_out), "probe_t1": probe_t1, "raw_print": raw_print,
            "prose_chars": len(prose), "stage_pos": pos, "order_ok": order_ok, "cov": round(cov, 3),
            "tracebacks": sum(u.count("Traceback") for u in users), "reward": rec.get("reward", 0)}
    if rec.get("error"):
        return "error", prof
    if rec.get("stop") != "has_final_answer":
        return "not_submitted", prof
    if rec.get("reward", 0) < floor:
        return "below_floor", prof
    if len(asst) > max_turns:
        return "too_many_turns", prof
    if any("<tool_call>" in a for a in asst):
        return "tool_call_format", prof
    if loop:
        return "loop", prof
    if unmotivated:
        return "unmotivated_repair", prof
    if not probe_t1:
        return "no_turn1_probe", prof
    if raw_print:
        return "raw_context_print", prof
    if not order_ok:
        return "stage_order", prof
    if cov < 0.3:
        return "low_note_coverage", prof
    if any("<strategy_hint>" in a or "Harness facts" in a for a in asst):
        return "hint_text_leak", prof
    if HINT_REFERENCE_RE.search(prose) or PROSE_LEAK.search(prose):
        return "hint_reference", prof
    return None, prof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--floor", type=float, default=0.8)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--per-row", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reasons, profs, kept, per_row = collections.Counter(), [], [], collections.Counter()
    for run in a.runs:
        for line in open(run):
            rec = json.loads(line)
            why, prof = check(rec, a.floor, a.max_turns)
            profs.append({**prof, "kept": why is None})
            rk = (Path(rec.get("rows") or "rows/gen256_s2027.jsonl").stem, rec["gen_row"])
            if why is None and per_row[rk] >= a.per_row:
                why = "per_row_cap"
            reasons[why or "kept"] += 1
            if why:
                continue
            per_row[rk] += 1
            prompt = [m for m in (rec.get("prompt") or []) if not is_prologue(m)]
            body = [m for m in rec["messages"] if not is_prologue(m)]
            kept.append({"messages": prompt + body, "env": "ctc-qdmatch-hpqa-256k-synthgen",
                         "uid": f"qd256-{rk[0]}-{rk[1]}-{per_row[rk]}",
                         "task": f"{rk[0]}-{rk[1]}", "reward": rec["reward"], "turns": prof["turns"],
                         "sub_calls": prof["sub_calls"], "root_chars": prof["root_chars"],
                         "surface_mode": (rec.get("surface") or {}).get("mode"), "id_map": (rec.get("surface") or {}).get("id_map"),
                         "recovery_events": prof["recovery_events"]})
    with open(out / "kept.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    def summ(rows, k):
        v = [r[k] for r in rows]
        return {"mean": round(statistics.mean(v), 1), "median": statistics.median(v)} if v else None

    k = [p for p in profs if p["kept"]]
    rep_modes = collections.Counter(r["surface_mode"] for r in kept)
    tk = collections.Counter(p["turns"] for p in profs if p["kept"])
    rep = {"reasons": dict(reasons), "n": len(profs), "kept": len(kept),
           "kept_turn_hist": dict(sorted(tk.items())),
           "kept_by_surface": dict(rep_modes), "kept_with_recovery": sum(1 for p in profs if p["kept"] and p["recovery_events"] > 0),
           "all": {x: summ(profs, x) for x in ("turns", "sub_calls", "root_tokens", "tokens_per_turn", "repl_out_tokens", "prose_chars", "reward")},
           "kept_profile": {x: summ(k, x) for x in ("turns", "sub_calls", "root_tokens", "tokens_per_turn", "repl_out_tokens", "prose_chars", "reward")}}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
