"""Build ``bestanswer_disjoint_manifest.json``, question-disjoint splits of Ada-LEval BestAnswer.

The same questions appear at several lengths, so splits are pinned by question id: ``eval`` (50 of the 199 long
questions), ``train`` (the 798 short-only questions), ``train_long`` (the other 149).

    python -m ada_leval_real.build_bestanswer_disjoint_manifest [--data-dir DIR] [--out PATH] [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request

from ada_leval_real.env import _BA_URL, _DEFAULT_MANIFEST, _data_dir

TRAIN_SETTING = "2k"
EVAL_SETTING = "32k"
N_EVAL = 50
SEED = 42


def _qids(setting: str, data_dir: str) -> list[str]:
    """Question ids of ``stackselect_<setting>.json`` in file order, deduplicated."""
    path = os.path.join(data_dir, f"stackselect_{setting}.json")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        urllib.request.urlretrieve(_BA_URL.format(setting=setting), path + ".part")
        os.replace(path + ".part", path)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    seen, out = set(), []
    for r in rows:
        q = str(r["question_id"])
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def build(data_dir: str) -> dict:
    short_set = set(_qids(TRAIN_SETTING, data_dir))
    long_set = set(_qids(EVAL_SETTING, data_dir))
    assert long_set <= short_set, "expected long questions to be a subset of short"

    rng = random.Random(SEED)
    eval_qids = sorted(rng.sample(sorted(long_set), N_EVAL))
    train_qids = sorted(short_set - long_set)
    train_long = sorted(long_set - set(eval_qids))
    assert not (set(train_qids) & set(eval_qids)) and not (set(train_long) & set(eval_qids))

    # kept verbatim so the output matches the packaged file
    return {
        "train": train_qids,
        "eval": eval_qids,
        "meta": {
            "task": "ada_leval_bestanswer",
            "train_setting": TRAIN_SETTING,
            "eval_setting": EVAL_SETTING,
            "n_train": len(train_qids),
            "n_eval": len(eval_qids),
            "eval_pool": len(long_set),
            "seed": SEED,
            "note": "train = short-only questions (<=16k, not in the >=32k set); "
                    "eval = 50 sampled from the 199 long-set questions. Disjoint by construction.",
            "train_long_note": (
                f"{len(train_long)} long-pool ids ({EVAL_SETTING}+ data) = all {len(long_set)} "
                f"minus the {len(eval_qids)} reserved in `eval`. Added for the "
                "train-mix-large ada-leval 8k-32k band, because `train` holds "
                "SHORT-pool ids (1k-16k) that do not exist in the 32k+ data — "
                "intersecting them yields 0 rows. `train`/`eval` are unchanged."
            ),
        },
        "train_long": train_long,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="default: $ADALEVAL_DATA_DIR")
    ap.add_argument("--out", default=_DEFAULT_MANIFEST)
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    a = ap.parse_args()
    text = json.dumps(build(_data_dir(a.data_dir)), indent=2) + "\n"
    if a.check:
        with open(a.out, encoding="utf-8") as f:
            same = f.read() == text
        print(f"{'IDENTICAL' if same else 'DIFFERENT'}: {a.out}")
        return 0 if same else 1
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
