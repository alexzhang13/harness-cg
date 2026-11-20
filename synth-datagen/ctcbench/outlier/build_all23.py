"""Build decontaminated 1M outlier rows (dev or train) with the CTC-Bench generator.

Same code path as the generator, except the majority-run sampling cap is raised from 400 to 4,000 so the 1M rung
is reachable. Runs whose passages match any eval passage of the 23 families (exact or 13-gram) are rejected, and
articles from eval outlier rows are excluded. Run `all23.py build` first; needs $CTC_REPO_SRC.

    python build_all23.py <out.jsonl.gz> <n> <seed> <split> [exclude_outlier_titles_from.jsonl.gz]
"""
import gzip, json, os, random, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "audit"))
sys.path.insert(0, os.environ.get("CTC_REPO_SRC", "ctc/src"))
import common  # noqa: E402
from ctc.data import seeds, ladders
from ctc.tasks.outlier.sources import wiki100w as W
from ctc.tasks.outlier.generate import TOPIC_QUERY, assemble, prefix_to_size

out_path, n, seed, split = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
pool_path = common.seed_pool("outlier.seed.jsonl.gz")
corpus = seeds.load(pool_path, "outlier")


def _sample_run_nocopy(self, n_chunks, rng, *, exclude_titles=None, single_only=False, max_attempts=64):
    """W.sample_run without copying exclude_titles (the copy dominated runtime)."""
    exclude = exclude_titles if exclude_titles is not None else set()
    candidates = self._candidates(n_chunks)
    for _ in range(max_attempts if candidates else 0):
        index = candidates[rng.randrange(len(candidates))]
        title, bodies = self.articles[index]
        if title in exclude:
            continue
        start = rng.randint(0, len(bodies) - n_chunks)
        from ctc.data.sources.wiki100w import Chunk as _Chunk
        return [_Chunk(title, b) for b in bodies[start: start + n_chunks]]
    if single_only:
        return None
    return self._compose(n_chunks, rng, exclude)


type(corpus).sample_run = _sample_run_nocopy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import all23 as _A  # noqa: E402
_G, _X = _A.load()
REJECTED = set()


def _dirty(text):
    return _A.dirty(text, _G, _X)


def clean_run(n_chunks, rng, exclude, single_only, tries=200):
    # rejected articles are added to `exclude` in place
    for _ in range(tries):
        run = corpus.sample_run(n_chunks, rng, exclude_titles=exclude, single_only=single_only)
        if run is None:
            return None
        bad = {c.title for c in run if _dirty(c.body)}
        if not bad:
            return run
        REJECTED.update(bad); exclude |= bad
    return None
size = 7209  # passages in the published r1m rows (docs_for_rung gives 7,050 here)
print("pool articles", len(corpus.articles), "1m docs", size, flush=True)
norm = lambda t: " ".join(t.split())
body2title = {}
for title, bodies in corpus.articles:
    for b in bodies:
        body2title.setdefault(norm(b), title)
eval_gold_titles, eval_gold_texts = set(), set()
EV = os.path.join(common.eval_dir(), "outlier")
for p in sorted(Path(EV).glob("*.parquet")):
    for ex in pq.read_table(p, columns=["documents", "gold_doc_indices"]).to_pylist():
        for d in ex["documents"]:                      # all eval passages, not only gold
            t = norm(d["text"])
            eval_gold_texts.add(t)
            if t in body2title:
                eval_gold_titles.add(body2title[t])
print("eval titles (any passage)", len(eval_gold_titles), "eval passages", len(eval_gold_texts), flush=True)
dev_titles = set()
if len(sys.argv) > 5:
    dev_titles = {json.loads(l)["dev_meta"]["outlier_title"] for l in gzip.open(sys.argv[5], "rt")}
print("dev outlier titles", len(dev_titles), flush=True)
EXCL_OUT = eval_gold_titles | dev_titles


def ladder(rng, num_outliers=3, min_run=4, max_run=14, attempts_cap=4000):
    floor = num_outliers + 1
    outlier = clean_run(num_outliers, rng, EXCL_OUT, True)
    if outlier is None:
        return None
    runs, used, placed, attempts = [], W._titles(outlier) | eval_gold_titles, 0, 0
    while placed < size - num_outliers and attempts < attempts_cap:
        attempts += 1
        run = clean_run(rng.randint(max(min_run, floor), max(max_run, floor)), rng, used, True)
        if run is None:
            continue
        used |= W._titles(run); runs.append(run); placed += len(run)
    if placed < size - num_outliers:
        return None
    majority = prefix_to_size(runs, size - num_outliers, floor)
    ex = assemble([(c.title, c.body) for c in majority], [(c.title, c.body) for c in outlier], source=W.SOURCE_TAG,
                  query=TOPIC_QUERY, rng=random.Random(f"{split}1m-clean:{seed}:{rng.random()}"), titles=False,
                  meta={"minority_label": outlier[0].title, "num_outliers": num_outliers,
                        "num_categories": len({c.title for c in majority}) + 1, "mode": "mixed", "rung": "1m"})
    return ex, outlier[0].title


rows, dropped, idx = [], 0, 0
with gzip.open(out_path, "wt") as f:
    while len(rows) < n:
        idx += 1
        r = ladder(random.Random(f"{seed}:{split}1m-clean:{idx}"))
        if r is None:
            continue
        ex, title = r
        if title in eval_gold_titles or title in dev_titles or any(norm(d["text"]) in eval_gold_texts for d in ex["documents"]):
            dropped += 1
            continue
        ex["dev_meta"] = {"outlier_title": title, "seed": seed, "index": idx, "split": split, "clean": "all23-families-exact+13gram"}
        rows.append(ex); f.write(json.dumps(ex) + "\n"); f.flush()
        print(len(rows), len(ex["documents"]), title, "rejected articles so far", len(REJECTED), flush=True)
print("done", len(rows), "dropped (contaminated)", dropped, flush=True)
