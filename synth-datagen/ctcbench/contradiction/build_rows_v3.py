"""Build contradiction rows (dev and train) with the benchmark's build_example from the decontaminated pool.

    python build_rows_v3.py dev 40 ; python build_rows_v3.py train 450"""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import gzip, json, os, random, sys
sys.path.insert(0, __import__("os").environ.get("CTC_REPO_SRC", "ctc/src"))
from ctc.data.sources import pubmed as source
from ctc.tasks.contradiction.sources.pubmed import build_example
D = os.path.join(os.path.join(os.environ.get("CTC_DATAGEN_DIR") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../outputs/synth-datagen/ctcbench")), "contradiction"), "dev1m")
split, n = sys.argv[1], int(sys.argv[2]); N_DEV_PAIRS = 3 * 40
pool = json.load(gzip.open(os.path.join(D, "clean_pool.json.gz"), "rt"))
pairs = [source.ClaimPair(p["claim"], p["contradiction"], str(p.get("abstract_id", "")), p.get("mode", "realistic")) for p in pool["pairs"]]
random.Random(20261022).shuffle(pairs)
pairs = pairs[:N_DEV_PAIRS] if split == "dev" else pairs[N_DEV_PAIRS:]
fill = {**{k: tuple(v) for k, v in pool["fillers"].items()}, **{"x" + k: tuple(v) for k, v in pool["fillers_extra"].items()}}
corpus = source.PubMedPool(pairs=tuple(pairs), fillers=fill, provenance={"pairs": "seed pool, all-family decontaminated"})
out = os.path.join(D, f"rows_v3_{split}.jsonl")
with open(out, "w") as fo:
    for i in range(n):
        ex = build_example(random.Random((7 if split == "dev" else 9) * 100000 + i), index=i, corpus=corpus, num_docs=23147, num_pairs=3)
        if ex is None: print("pool exhausted at", i); break
        fo.write(json.dumps({"dev_id": i, "split": split, "documents": ex["documents"], "gold_doc_indices": ex["gold_doc_indices"]}) + "\n")
print(split, "rows written", i + 1, out)
