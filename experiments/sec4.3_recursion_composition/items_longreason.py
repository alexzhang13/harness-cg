"""LongReason 128k pool tasks. key = index into the 794-row split shuffled with seed 42 (as longreason_real does).
score() = longreason_real's option-letter parse + exact match."""
from datasets import load_dataset

from longreason_real.env import HF_DATASET, _option_letters, _parse_letter

_DS = None


def _ds():
    global _DS
    if _DS is None:
        _DS = load_dataset(HF_DATASET, split="128k").shuffle(seed=42)
    return _DS


def load_items(keys):
    ds = _ds()
    out = {}
    for k in keys:
        r = ds[int(k)]
        inquiry = r["final_inquiry"].strip()
        out[k] = {"env": "longreason", "key": k, "kind": "mc", "gold": r["answer"].strip().upper(),
                  "options": _option_letters(inquiry), "context": r["background"].strip(),
                  "question": ("Answer the multiple-choice question below. The scenario it refers to is embedded somewhere inside the long "
                               "`context` passage, surrounded by unrelated filler text. Locate it first, then reason.\n\n"
                               f"{inquiry}\n\nRespond with the single option letter only.")}
    return out


def score(item, answer):
    pred = _parse_letter(str(answer), item["options"] or None)
    return 1.0 if pred and pred == item["gold"] else 0.0
