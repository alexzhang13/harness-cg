"""CorpusQA (`corpusqa-real`): aggregation questions over a multi-document corpus (128k or 1M tokens).

`context` is the list of documents; the question and the upstream output requirements go into the root prompt.
Scoring by gold type: set-F1 over entity names for lists, relative tolerance for numbers, exact match for strings.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

HF_DATASET = "Tongyi-Zhiwen/CorpusQA"

# set name -> file in the HF repo
SET_FILES = {
    "128k": "128k_4domains.jsonl",
    "1m": "1m_4domains.jsonl",
}

DOMAINS = ("financial_en", "financial_zh", "education_en", "real_estate_en")

_DOC_SPLIT_RE = re.compile(r"^#\s*Document\s+\d+\s*:", re.MULTILINE)

_PUNCT = str.maketrans("", "", string.punctuation)


def _norm_text(s: Any) -> str:
    s = str(s if s is not None else "").strip().lower()
    s = s.translate(_PUNCT)
    return re.sub(r"\s+", " ", s).strip()


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_number(s: Any) -> float | None:
    """Parse numbers like ``"1,234"``, ``"$5.6"``, ``"77.25%"``."""
    if isinstance(s, bool):
        return None
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s or "").strip()
    if not txt:
        return None
    m = _NUM_RE.search(txt.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _parse_list(pred: str) -> list[str]:
    """Pull a list of items out of a free-form answer."""
    txt = (pred or "").strip()
    m = re.search(r"\[.*\]", txt, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except json.JSONDecodeError:
            pass
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if len(lines) > 1:
        out = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in lines]
        return [o for o in out if o]
    if "," in txt:
        return [p.strip() for p in txt.split(",") if p.strip()]
    return [txt] if txt else []


# shorter name must be at least this long for containment to count
_MIN_CONTAINMENT_LEN = 5


def _info_len(s: str) -> int:
    """Length with CJK characters counted double."""
    return sum(2 if _is_cjk(ch) else 1 for ch in s)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0xF900 <= o <= 0xFAFF
        or 0x3040 <= o <= 0x30FF
    )


def _entity_equal(a: str, b: str, mode: str) -> bool:
    """Same entity? ``containment`` also accepts one name inside the other (short vs full legal names)."""
    if a == b:
        return True
    if mode != "containment":
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if _info_len(short) < _MIN_CONTAINMENT_LEN:
        return False
    return short in long


def _set_f1(pred_items: list[str], gold_items: list[str],
            mode: str = "containment") -> float:
    """Set-F1 with one-to-one greedy assignment between predictions and gold."""
    gold = [g for g in (_norm_text(x) for x in gold_items) if g]
    pred = [p for p in (_norm_text(x) for x in pred_items) if p]
    gold = list(dict.fromkeys(gold))
    pred = list(dict.fromkeys(pred))
    if not gold:   # an empty gold list is a real answer ("none match")
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0

    unmatched = list(gold)
    tp = 0
    for p in pred:
        for i, g in enumerate(unmatched):
            if _entity_equal(p, g, mode):
                tp += 1
                unmatched.pop(i)
                break
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(gold)
    return 2 * precision * recall / (precision + recall)


def _make_score(num_tolerance: float, entity_match: str = "containment"):
    async def _score(info, state: vf.State, **_kw: Any) -> float:
        meta = json.loads(info) if isinstance(info, str) else info
        gold = meta.get("gold_answer")
        kind = meta.get("answer_kind") or "str"
        final = state.get("rlm_final_answer") or state.get("final_answer") or ""
        pred = final if isinstance(final, str) else str(final)
        if not pred.strip():
            return 0.0

        if kind == "list":
            gold_items = gold if isinstance(gold, list) else [gold]
            return _set_f1(_parse_list(pred), [str(g) for g in gold_items],
                           mode=entity_match)

        if kind == "number":
            g = _to_number(gold)
            p = _to_number(pred)
            if g is None or p is None:
                return 0.0
            if g == 0:
                return 1.0 if abs(p) <= num_tolerance else 0.0
            return 1.0 if abs(p - g) / abs(g) <= num_tolerance else 0.0

        gs, ps = _norm_text(gold), _norm_text(pred)
        if not gs:
            return 0.0
        if gs == ps:
            return 1.0
        g, p = _to_number(gold), _to_number(pred)
        if g is not None and p is not None:
            if g == 0:
                return 1.0 if abs(p) <= num_tolerance else 0.0
            return 1.0 if abs(p - g) / abs(g) <= num_tolerance else 0.0
        return 1.0 if re.search(rf"(?<!\w){re.escape(gs)}(?!\w)", ps) else 0.0

    return _score


def _split_documents(user_content: str) -> list[str]:
    """Split the prepared user message back into per-document strings."""
    parts = _DOC_SPLIT_RE.split(user_content or "")
    docs = [p.strip() for p in parts if p and p.strip()]
    return docs or ([user_content.strip()] if user_content.strip() else [])


_TASK_BLOCK_RE = re.compile(r"^#\s*Question:\s*$", re.MULTILINE)
_OUTPUT_REQ_RE = re.compile(r"^Output requirements:", re.MULTILINE)


def _split_task_block(last_doc: str) -> tuple[str, str | None]:
    """(last document without the task block, ``Output requirements`` text or None)."""
    starts = [m.start() for m in _TASK_BLOCK_RE.finditer(last_doc)]
    if not starts:
        return last_doc, None
    doc, block = last_doc[:starts[-1]].rstrip(), last_doc[starts[-1]:]
    m = _OUTPUT_REQ_RE.search(block)
    return doc, (block[m.start():].strip() if m else None)


def _classify_answer(ans: Any) -> str:
    if isinstance(ans, list):
        return "list"
    if isinstance(ans, bool):
        return "str"
    if isinstance(ans, (int, float)):
        return "number"
    return "str"


def _load_rows(set_name: str) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_DATASET, SET_FILES[set_name], repo_type="dataset")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _build_dataset(
    *, set_name: str, domain: str | None, num_examples: int, seed: int | None,
    max_doc_chars: int | None,
) -> Dataset:
    raw = _load_rows(set_name)
    if domain:
        raw = [r for r in raw if r.get("domain") == domain]
    if seed is not None:
        import random

        random.Random(seed).shuffle(raw)
    if num_examples and num_examples > 0:
        raw = raw[:num_examples]

    rows: list[dict] = []
    for i, r in enumerate(raw):
        question = (r.get("question") or "").strip()
        gold = r.get("answer")
        if not question or gold is None:
            continue
        msgs = r.get("prompt") or []
        user_content = ""
        system_content = ""
        for m in msgs:
            if m.get("role") == "user":
                user_content = m.get("content") or ""
            elif m.get("role") == "system":
                system_content = m.get("content") or ""
        docs = _split_documents(user_content)
        output_req = None
        if docs:
            docs[-1], output_req = _split_task_block(docs[-1])
        if max_doc_chars is not None:
            docs = [d[:max_doc_chars] for d in docs]
        if not docs:
            continue

        kind = _classify_answer(gold)
        if output_req:
            shape = (f"{output_req}\n\nSubmit the final answer through `answer[\"content\"]`, in exactly the format "
                     "required above (the \"The answer is:\" prefix is optional).")
        else:   # a row without the upstream output requirements
            shape = {
                "list": (
                    "Answer with a JSON array of the matching entity names, e.g. "
                    '["Name A", "Name B"]. Include every match and nothing else.'
                ),
                "number": "Answer with the single number only — no units, no commas, no prose.",
                "str": "Answer with the value only — no explanation.",
            }[kind]

        sep = "\n\n" if output_req else " "
        info = {
            "context": docs,
            "root_prompt": (
                f"{system_content.strip() or 'You are a professional data analyst.'}\n\n"
                f"Question: {question}\n\n"
                f"`context` is a list of {len(docs)} documents (the source corpus); "
                f"the answer requires aggregating across them.{sep}"
                f"{shape}"
            ),
            "gold_answer": gold,
            "answer_kind": kind,
            "question": question,
            "domain": r.get("domain") or "",
            "corpusqa_set": set_name,
            "n_docs": len(docs),
            "item_id": r.get("id") or "",
        }
        if output_req:
            info["output_requirements"] = output_req
        rows.append({
            "example_id": i,
            "prompt": [{"role": "user", "content": "Answer the question about the corpus."}],
            "answer": json.dumps(gold, ensure_ascii=False),
            "info": json.dumps(info, ensure_ascii=False),
        })
    return Dataset.from_list(rows)


def load_environment(
    *,
    set_name: str = "128k",
    domain: str | None = None,
    num_examples: int = -1,
    seed: int = 42,
    max_doc_chars: int | None = None,
    num_tolerance: float = 1e-4,
    entity_match: str = "containment",
    max_iterations: int = 30,
    sub_max_tokens: int = 4096,
    min_subcall: int = 1,
    **kwargs: Any,
) -> vf.Environment:
    """Load CorpusQA as an RLMTrainEnv. set_name: ``"128k"`` or ``"1m"``."""
    if set_name not in SET_FILES:
        raise ValueError(
            f"set_name={set_name!r} unknown. Choose from {tuple(SET_FILES)}."
        )
    if domain and domain not in DOMAINS:
        raise ValueError(f"domain={domain!r} unknown. Choose from {DOMAINS}.")

    dataset = _build_dataset(
        set_name=set_name, domain=domain, num_examples=num_examples,
        seed=seed, max_doc_chars=max_doc_chars,
    )
    if entity_match not in ("containment", "exact"):
        raise ValueError(
            f"entity_match={entity_match!r} unknown. Use 'containment' or 'exact'."
        )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_make_score(num_tolerance, entity_match),
        weight=1.0, min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment", "SET_FILES", "DOMAINS"]
