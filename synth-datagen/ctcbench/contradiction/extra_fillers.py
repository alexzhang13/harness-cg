"""Add more filler abstracts (PubMedQA, after the seed pool's 20,000) that pass the audit index to clean_pool.json.gz.
Needs $CTC_REPO_SRC and `datasets`."""
import gzip, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit"))
sys.path.insert(0, os.environ.get("CTC_REPO_SRC", "ctc/src"))
import common
from ctc.data.sources.pubmed import split_sentences
from datasets import load_dataset
IX = common.Index()
hit = lambda s: bool(IX.exact(common.ekey(s))) or bool(IX.ghits(common.grams(s)))
ds = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
order = list(range(len(ds))); random.Random(42).shuffle(order)
P = os.path.join(common.DATAGEN, "contradiction", "dev1m", "clean_pool.json.gz")
pool = json.load(gzip.open(P, "rt")); extra = {}; n = 0
for i in order[20000:80000]:
    sents = [s for c in ds[i]["context"]["contexts"] if c for s in split_sentences(c)]
    n += 1
    if sents and not any(hit(s) for s in sents): extra[str(i)] = sents
    if n % 5000 == 0: print(n, "kept", len(extra), sum(len(v) for v in extra.values()), flush=True)
pool["fillers_extra"] = extra
json.dump(pool, gzip.open(P, "wt"))
print("extra abstracts kept %d/%d, sentences %d" % (len(extra), n, sum(len(v) for v in extra.values())))
