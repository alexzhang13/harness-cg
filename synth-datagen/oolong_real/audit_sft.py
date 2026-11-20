"""Compare the events in accepted traces against the generator's event tables (no model calls).

    python audit_sft.py --runs <gen>/<arm>.jsonl --max-turns 10 --allow-copied-literal --out <audit.json>
"""
import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_sft  # noqa: E402
from export_sft import accept  # noqa: E402
from tabletop_synth.generate import generate  # noqa: E402

LOCAL = {"s_kth", "s_first1", "s_first2", "s_first3", "s_last1", "s_last2", "s_last3", "s_spell", "s_nchars",
         "s_first_each", "s_last_each", "s_first_each_ep", "s_last_each_ep", "s_char_first_each_ep",
         "s_char_last_each_ep", "r_nat20", "r_nat1", "r_crits", "s_upcast_list", "s_upcast_n", "s_cantrip"}

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--max-turns", type=int, default=5)
ap.add_argument("--allow-copied-literal", action="store_true")
ap.add_argument("--out", default="outputs/synth-datagen/oolong_real/gen/audit_sft.json")
a = ap.parse_args()
export_sft.COPIED = a.allow_copied_literal
path = Path(a.runs)
rows, by_scope = [], collections.defaultdict(list)
for line in path.read_text().splitlines():
    rec = json.loads(line)
    tf = path.with_suffix(".traces") / f"{rec['id']}.json"
    if not tf.exists():
        continue
    tr = json.loads(tf.read_text())
    ok, _ = accept(rec, tr, a.max_turns)
    if not ok:
        continue
    seed = int(rec["id"].split("-")[1])
    surf = rec.get("surfaces", "v1")
    g = generate(seed, surfaces=surf)
    if g["question"] != rec.get("question"):
        import random

        from oolong_real_synth import gen as G
        tid = random.Random(seed + 1).choice(sorted(t for t in G.SINGLE_W if t.startswith("r_") and t in G.MULTI_W))
        g = generate(seed, tid=tid, surfaces=surf)
    if g["question"] != rec.get("question"):
        continue
    kind = "SPELL" if "spell" in g["question"].lower() else "ROLL"
    gold_n = len(g["meta"]["spells"] if kind == "SPELL" else g["meta"]["rolls"])
    outs = [m["content"] for m in tr["history"] if m["role"] == "user" and "REPL output" in str(m["content"])]
    ext = next((int(m.group(1) or m.group(2)) for o in outs
                for m in [re.search(r"unparsed; (\d+) events|chunks, (\d+) (?:rolls|casts);", o)] if m), None)
    eps = next((int(m.group(1) or m.group(2)) for o in outs
                for m in [re.search(r"episodes: (\d+) lines|(\d+) episodes, lines \[", o)] if m), None)
    scope = "local" if g["qtype"] in LOCAL else "global"
    r = {"id": rec["id"], "qtype": g["qtype"], "kind": kind, "gold_events": gold_n, "extracted": ext,
         "ratio": (ext / gold_n) if ext is not None and gold_n else None, "eps_ok": eps == g["meta"]["n_episodes"],
         "scope": scope}
    rows.append(r)
    by_scope[scope].append(r)
print("accepted traces audited:", len(rows))
print("probe episode count correct:", sum(r["eps_ok"] for r in rows), "/", len(rows))
for scope, rs in by_scope.items():
    rat = [r["ratio"] for r in rs if r["ratio"] is not None]
    within = lambda t, rat=rat: sum(abs(x - 1) <= t for x in rat)  # noqa: E731
    print(f"{scope:6s} n={len(rs)} extracted/gold median {statistics.median(rat):.3f} min {min(rat):.3f} max "
          f"{max(rat):.3f} | within 2% {within(0.02)} 5% {within(0.05)} 10% {within(0.10)} | >10% off {len(rat) - within(0.10)}")
for kind in ("ROLL", "SPELL"):
    rat = [r["ratio"] for r in rows if r["kind"] == kind and r["ratio"] is not None]
    if rat:
        print(f"{kind:5s} n={len(rat)} extracted/gold median {statistics.median(rat):.3f}, >10% off {sum(abs(x - 1) > .1 for x in rat)}")
worst = sorted((r for r in rows if r["ratio"] is not None), key=lambda r: -abs(r["ratio"] - 1))[:8]
print("worst:", [(r["id"], r["qtype"], r["extracted"], r["gold_events"]) for r in worst])
Path(a.out).write_text(json.dumps(rows, indent=1))
