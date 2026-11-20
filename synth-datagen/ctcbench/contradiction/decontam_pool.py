"""Filter the contradiction seed pool through the audit index and write clean_pool.json.gz.

Next: extra_fillers.py, then build_rows_v3.py."""
import gzip, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit"))
import common
IX = common.Index()
def hit(s):
    if IX.exact(common.ekey(s)): return True
    return bool(IX.ghits(common.grams(s)))
seed = common.seed_pool("contradiction.seed.jsonl.gz")
f = gzip.open(seed, "rt"); f.readline(); pool = json.loads(f.readline())
fill = {}; nh = 0
for i, (k, v) in enumerate(pool["fillers"].items()):
    if any(hit(s) for s in v): nh += 1
    else: fill[k] = v
    if i % 2000 == 0: print("fillers", i, "kept", len(fill), flush=True)
pairs = [p for p in pool["pairs"] if not hit(p["claim"]) and not hit(p["contradiction"])]
print("filler abstracts kept %d/%d (%d sentences); pairs kept %d/%d" % (len(fill), len(pool["fillers"]), sum(len(v) for v in fill.values()), len(pairs), len(pool["pairs"])))
D = os.path.join(common.DATAGEN, "contradiction", "dev1m")
os.makedirs(D, exist_ok=True)
json.dump({"fillers": fill, "pairs": pairs, "fams": IX.fams}, gzip.open(os.path.join(D, "clean_pool.json.gz"), "wt"))
