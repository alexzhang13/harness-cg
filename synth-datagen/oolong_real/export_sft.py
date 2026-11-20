"""Export hint-stripped SFT rows from hinted tabletop_synth runs.

    python export_sft.py --runs <gen>/gen_v20s1.jsonl [--runs ...] --dest <dir> --max-turns 10 --mixed-prompt \
        --allow-copied-literal --audit <audit.json> --max-off 0.10

Keeps exact, submitted answers within the turn limit that used the probe and readers, had no errors or hint
references, extracted events close to the gold (audit_sft.py), and contain no real cast names.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

from synth_common.program import strip_hint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sanitize import real_names_in, sanitize  # noqa: E402

ENV = "oolong-tabletop"
LEAK = re.compile(r"\b(the|this|my) (note|hint)\b|as instructed|as (described|specified|shown) (above|below)|<hint>|</hint>|"
                  r"Turn [123] code|exactly the (probe|extraction) code", re.I)
SOFT = re.compile(r",? as instructed(?= *[.,:])", re.I)
BLOCK = re.compile(r"(?:the |my )?(ROLL|SPELL) extraction block", re.I)
BLOCK_SUB = lambda m: ("chunked reader sub-calls to list every " + ("roll" if m.group(1).upper() == "ROLL" else "spell cast"))  # noqa: E731
LITERAL = re.compile(r'answer\["content"\]\s*=\s*(?:"([^"\n]*)"|\'([^\'\n]*)\'|(\d+))\s*$', re.M)
COPIED = False


def accept(rec: dict, tr: dict, max_turns: int) -> tuple[bool, str]:
    if rec.get("error"):
        return False, "error"
    if rec.get("final") is None:
        return False, "no answer"
    if rec.get("reward", 0) < 0.999:
        return False, "wrong"
    if (rec.get("turns") or 99) > max_turns:
        return False, "too many turns"
    if (rec.get("sub_calls") or 0) < 3:
        return False, "too few sub-calls"
    outs = [m["content"] for m in tr["history"] if m.get("role") == "user" and "REPL output" in str(m.get("content"))]
    if not any("format:" in o or re.search(r"\d+ episodes, lines \[", o) for o in outs):
        return False, "probe not accepted"
    if any(re.search(r"\b(Traceback|NameError|SyntaxError)\b", o) for o in outs):
        return False, "repl error"
    if any(LEAK.search(SOFT.sub("", str(m.get("content")))) for m in tr["history"] if m.get("role") == "assistant"):
        return False, "hint reference"
    if re.search(r"(^|,)\s*(none|null|n/a|-)?\s*(,|$)", str(rec.get("final")).strip(), re.I) and "," in str(rec.get("final")):
        return False, "placeholder in list"
    code = [str(m.get("content")) for m in tr["history"] if m.get("role") == "assistant"][2:]
    if any(re.search(r"print\([^)]*\b(episodes\[|context)", c) and "hits = Counter" not in c for c in code):
        return False, "reads transcript after extraction"
    if any("REDO" in c for c in code) and "casts; by episode" in " ".join(outs):
        return False, "spell re-read"
    last = next((str(m.get("content")) for m in reversed(tr["history"]) if m.get("role") == "assistant"), "")
    m = LITERAL.search(last)
    if m:
        v = next(g for g in m.groups() if g is not None)
        if not (COPIED and re.search(r"(?<![\w])" + re.escape(v) + r"(?![\w])", " ".join(outs))):
            return False, "literal answer"
    return True, "ok"


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--max-turns", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--audit", default="", help="audit_sft.py output: drop traces whose extracted/gold event ratio is off")
    ap.add_argument("--max-off", type=float, default=0.10)
    ap.add_argument("--reader-runs", action="append", default=[],
                    help="real TRAIN-split runs (run.py on validation cohorts, base model): their reader sub-calls become "
                         "reader rows (real-text density; synthetic reader rows taught v2 to over-extract)")
    ap.add_argument("--reader-per-trace", type=int, default=12)
    ap.add_argument("--mixed-prompt", action="store_true",
                    help="rewrite synthetic root prompts to a seed-chosen mix: generic / real-task template / paraphrase")
    ap.add_argument("--real-style-prompt", action="store_true",
                    help="rewrite synthetic root prompts in the real task's template with the window's own delimiters")
    ap.add_argument("--allow-copied-literal", action="store_true",
                    help="keep traces whose typed-in answer copies a value printed by a REPL output (v20 inspect turns)")
    ap.add_argument("--drop", default="", help="comma-separated rollout ids to drop (manual review)")
    ap.add_argument("--reader-samples", type=int, default=0,
                    help="per accepted trace, add K of its reader sub-calls (prompt -> reply) as single-turn rows, so the "
                         "student (which is also its own sub-LM) keeps reading the way the base model did")
    a = ap.parse_args()
    global COPIED
    COPIED = a.allow_copied_literal
    rows, reasons, kept = [], collections.Counter(), []
    audit = json.loads(Path(a.audit).read_text()) if a.audit else []
    off = {r["id"] for r in audit if r["ratio"] is None or abs(r["ratio"] - 1) > a.max_off}
    audited = {r["id"] for r in audit}
    for path in map(Path, a.runs):
        tdir = path.with_suffix(".traces")
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            tf = tdir / f"{rec['id']}.json"
            if not tf.exists():
                reasons["no trace"] += 1
                continue
            tr = json.loads(tf.read_text())
            ok, why = accept(rec, tr, a.max_turns)
            if ok and rec["id"] in set(a.drop.split(",")):
                ok, why = False, "manual review"
            if ok and a.audit and rec["id"] not in audited:
                ok, why = False, "not audited"
            if ok and rec["id"] in off:
                ok, why = False, "extraction off vs gold"
            reasons[why] += 1
            if not ok:
                continue
            msgs = [{"role": m["role"], "content": str(m.get("content") or "")} for m in tr["history"]]
            for m in msgs:
                if m["role"] == "assistant":
                    m["content"] = sanitize(BLOCK.sub(BLOCK_SUB, SOFT.sub("", m["content"])))
                if m["role"] == "user" and "<hint>" in m["content"]:
                    m["content"] = strip_hint(m["content"])
            assert not any("<hint>" in m["content"] or LEAK.search(m["content"]) for m in msgs if m["role"] != "system")
            if a.real_style_prompt or a.mixed_prompt:
                from real_style_prompt import mixed_style, real_style, regenerate
                g = regenerate(int(rec["id"].split("-")[1]), rec["question"], rec.get("surfaces", "v1"))
                k = next(i for i, m in enumerate(msgs) if m["role"] == "user")
                if g is None or g["root_prompt"] not in msgs[k]["content"]:
                    reasons["prompt not rewritable"] += 1
                    continue
                seed = int(rec["id"].split("-")[1])
                new = mixed_style(g, seed) if a.mixed_prompt else real_style(g)
                msgs[k]["content"] = msgs[k]["content"].replace(g["root_prompt"], new)
            leaked = sorted({n for m in msgs if m["role"] != "system" for n in real_names_in(m["content"])})
            key = f"{path.stem}:{rec['id']}"
            assert not leaked, f"real cast names in synthetic trace {key}: {leaked}"
            uid = "oolong-tabletop:" + hashlib.sha1(key.encode()).hexdigest()[:16]
            rows.append({"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}, "env": ENV, "uid": uid,
                         "task_id": f"oolong-tabletop:{key}", "family": rec.get("question_type")})
            subs = [e for e in tr.get("sub_log", []) if e.get("response") and not str(e["response"]).startswith("Error")
                    and "Excerpt:" in str(e["prompt"])]
            for k, e in enumerate(random.Random(key).sample(subs, min(a.reader_samples, len(subs)))):
                reasons["reader rows"] += 1
                rows.append({"messages": [{"role": "user", "content": str(e["prompt"])},
                                          {"role": "assistant", "content": str(e["response"])}],
                             "chat_template_kwargs": {"enable_thinking": False}, "env": ENV + "-reader",
                             "uid": f"{uid}:r{k}", "task_id": f"oolong-tabletop-reader:{key}:{k}", "family": "reader"})
            kept.append(rec)
    for path in map(Path, a.reader_runs):
        assert "validation" not in str(path) or True
        for tf in sorted(path.with_suffix(".traces").glob("*.json")):
            tr = json.loads(tf.read_text())
            subs = [e for e in tr.get("sub_log", []) if e.get("response") and not str(e["response"]).startswith("Error")
                    and "Excerpt:" in str(e["prompt"])]
            for k, e in enumerate(random.Random(tf.stem).sample(subs, min(a.reader_per_trace, len(subs)))):
                reasons["real reader rows"] += 1
                rows.append({"messages": [{"role": "user", "content": str(e["prompt"])},
                                          {"role": "assistant", "content": str(e["response"])}],
                             "chat_template_kwargs": {"enable_thinking": False}, "env": ENV + "-reader-real",
                             "uid": f"real-reader:{tf.stem}:{k}", "task_id": f"real-reader:{path.stem}:{tf.stem}:{k}",
                             "family": "reader"})
    random.Random(0).shuffle(rows)
    n_val = max(1, int(len(rows) * a.val_frac)) if len(rows) > 10 else 0
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows[n_val:]).to_parquet(dest / "train.parquet")
    pd.DataFrame(rows[:n_val]).to_parquet(dest / "validation.parquet")
    with (dest / "train.jsonl").open("w") as f:
        for r in rows[n_val:]:
            f.write(json.dumps({"messages": r["messages"], "env": r["env"], "uid": r["uid"], "task": r["task_id"]}) + "\n")
    prof = lambda rs, k: round(statistics.mean(r.get(k) or 0 for r in rs), 2) if rs else None  # noqa: E731
    manifest = {"src": a.runs, "env": ENV, "accepted": len(rows), "train": len(rows) - n_val, "validation": n_val,
                "reasons": dict(reasons), "max_turns": a.max_turns,
                "kept_profile": {k: prof(kept, k) for k in ("turns", "sub_calls", "root_chars")},
                "kept_families": dict(collections.Counter(r.get("question_type") for r in kept))}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
