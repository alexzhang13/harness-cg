"""Build ``ctcbench_synth/sidecars/xabsence_para_pool.json.gz``: PubMed abstracts (sentence lists) with one
paraphrase per sentence, for the synthetic ``xabsence_paraphrase`` family (``ctcbench_synth/xabsence_para.py``).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ctcbench_synth" / "sidecars" / "xabsence_para_pool.json.gz"
CACHE = HERE.parent / "ctcbench_synth" / "sidecars" / ".xabsence_para_cache.jsonl"

STYLE = {
    "free": "Restate each sentence the way a different author would report the same finding: a different sentence "
            "structure and different words throughout (synonyms; say technical terms in other words where that is "
            "natural, e.g. 'quality of life' for 'QoL', 'survival' for 'longevity'), but keep all numbers and the names "
            "of specific genes, proteins, drugs, places and instruments. Keep EVERY fact.",
    "anchor": "Rewrite each sentence in different words, as a second author restating the same finding: replace the "
              "ordinary words and phrasing with synonyms and restructure the sentence, but keep technical terms, "
              "names, numbers and abbreviations as they are (spell out an abbreviation only occasionally). Keep "
              "EVERY fact, number, unit, entity and the exact finding.",
    "moderate": "Rewrite each sentence as a careful scientific writer would: keep EVERY fact, number, unit, entity and "
                "the exact finding, but change the wording and the sentence structure (synonyms, reordering, active/passive).",
    "heavy": "Rewrite each sentence so that it shares as few content words with the original as possible: use synonyms "
             "for every word that has one, spell out abbreviations, restructure the sentence. Keep EVERY fact, number, "
             "unit, entity and the exact finding.",
}
PROMPT = """{style} Do not add or drop information. Each rewrite is one sentence (or as many as the original has).

{items}

Reply with one JSON object mapping each number to its rewrite, e.g. {{"1": "...", "2": "..."}}. No other text."""


def norm(s: str) -> str:
    return " ".join(str(s).lower().split())


EVAL_CACHE = HERE.parent / "ctcbench_synth" / "sidecars" / ".xabsence_para_eval_norm.txt.gz"


def eval_texts() -> set[str]:
    """Normalised texts of every eval-ladder sentence to exclude (cached locally after the first, slow, load)."""
    if EVAL_CACHE.exists():
        with gzip.open(EVAL_CACHE, "rt", encoding="utf-8") as f:
            return set(f.read().split("\n")) - {""}
    out = _eval_texts_hf()
    with gzip.open(EVAL_CACHE, "wt", encoding="utf-8") as f:
        f.write("\n".join(sorted(out)))
    return out


def _eval_texts_hf() -> set[str]:
    sys.path.insert(0, str(HERE.parents[2] / "real" / "CTCBench"))
    from huggingface_hub import HfApi
    from ctcbench_real import env as E
    files = HfApi().list_repo_files(E.HF_DATASET, repo_type="dataset")
    out: set[str] = set()
    for f in files:
        parts = f.split("/")
        if len(parts) == 3 and parts[1] in ("xabsence_paraphrase", "contradiction_iid") and f.endswith(".parquet"):
            for ex in E._hf_rung(parts[1], parts[2][:-8]):
                out.update(norm(d["text"]) for d in ex.get("documents") or [])
    return out


def seed_abstracts() -> dict[str, list[str]]:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("PrasannSinghal/ctc-seed-pools", "contradiction.seed.jsonl.gz", repo_type="dataset")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        f.readline()
        return json.loads(f.readline())["fillers"]


def jparse(text: str):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-abstracts", type=int, default=1000)
    ap.add_argument("--heavy", type=float, default=0.4, help="share of abstracts in the 'heavy' style")
    ap.add_argument("--base-style", default="free", choices=["free", "anchor", "moderate"],
                    help="style of the other abstracts")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--per-call", type=int, default=20)
    ap.add_argument("--skip", type=int, default=0, help="skip this many eligible abstracts (hash order) first")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from openai import OpenAI
    cli = OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=600)

    ev = eval_texts()
    absx = seed_abstracts()
    keep, dropped, skipped = [], 0, 0
    for aid in sorted(absx, key=lambda k: hashlib.sha1(f"xabpara|{k}".encode()).hexdigest()):
        sents = [s.strip() for s in absx[aid] if 25 <= len(s.strip()) <= 400]
        if len(sents) < 4:
            continue
        if any(norm(s) in ev for s in absx[aid]):
            dropped += 1
            continue
        if skipped < a.skip:
            skipped += 1
            continue
        keep.append((aid, sents))
        if len(keep) >= a.n_abstracts:
            break
    print(f"abstracts kept {len(keep)}, dropped for eval overlap {dropped}, eval texts {len(ev)}", flush=True)

    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            r = json.loads(line)
            cache[r["key"]] = r["out"]
    jobs = []
    for aid, sents in keep:
        h = int(hashlib.sha1(f"style|{aid}".encode()).hexdigest()[:8], 16) / 16**8
        style = "heavy" if h < a.heavy else a.base_style
        jobs.append((aid, style, sents))
    batches = []
    for aid, style, sents in jobs:
        for k in range(0, len(sents), a.per_call):
            chunk = sents[k:k + a.per_call]
            key = hashlib.sha1(json.dumps([style, chunk]).encode()).hexdigest()
            batches.append((key, style, chunk))

    def call(b):
        key, style, chunk = b
        if key in cache:
            return key, cache[key]
        items = "\n".join(f"{n + 1}. {s}" for n, s in enumerate(chunk))
        for attempt in range(3):
            try:
                r = cli.chat.completions.create(
                    model=a.model, temperature=0.7, max_tokens=6000,
                    messages=[{"role": "user", "content": PROMPT.format(style=STYLE[style], items=items)}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                d = jparse(r.choices[0].message.content) or {}
                out = [str(d.get(str(n + 1), "")).strip() for n in range(len(chunk))]
                return key, out
            except Exception as e:  # noqa: BLE001
                print("retry", attempt, e, flush=True)
                time.sleep(5)
        return key, [""] * len(chunk)

    todo = [b for b in batches if b[0] not in cache]
    print(f"{len(batches)} calls, {len(todo)} to do", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex, CACHE.open("a") as fc:
        for n, (key, out) in enumerate(ex.map(call, todo)):
            cache[key] = out
            fc.write(json.dumps({"key": key, "out": out}) + "\n")
            if n % 50 == 0:
                print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s", flush=True)

    abstracts = []
    bad = 0
    by_aid: dict = {}
    for key, style, chunk in batches:
        by_aid.setdefault(id(chunk), None)
    k = 0
    for aid, style, sents in jobs:
        paras = []
        for j in range(0, len(sents), a.per_call):
            chunk = sents[j:j + a.per_call]
            paras += cache[hashlib.sha1(json.dumps([style, chunk]).encode()).hexdigest()]
        pairs = []
        for s, p in zip(sents, paras):
            if not p or norm(p) == norm(s) or len(p) < 15 or norm(p) in ev:
                bad += 1
                continue
            pairs.append([s, p])
        if len(pairs) >= 3:
            split = "val" if int(hashlib.sha1(f"split|{aid}".encode()).hexdigest()[:8], 16) % 10 == 0 else "train"
            abstracts.append({"id": aid, "style": style, "split": split, "pairs": pairs})
        k += 1
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump({"source": "PrasannSinghal/ctc-seed-pools contradiction.seed fillers (PubMed abstracts)",
                   "paraphraser": f"{a.model} T=0.7 thinking off; heavy share {a.heavy}",
                   "eval_exclusion": {"ladders": ["xabsence_paraphrase", "contradiction_iid"], "eval_texts": len(ev),
                                      "abstracts_dropped": dropped},
                   "dropped_pairs": bad, "abstracts": abstracts}, f)
    print(f"wrote {out_path}: {len(abstracts)} abstracts, {sum(len(x['pairs']) for x in abstracts)} pairs, "
          f"{bad} bad pairs dropped", flush=True)


if __name__ == "__main__":
    main()
