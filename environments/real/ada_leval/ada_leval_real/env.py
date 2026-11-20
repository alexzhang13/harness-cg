"""Ada-LEval BestAnswer (`ada-leval-real`): pick the accepted answer to a Stack Overflow question among N candidates.

Train and eval questions are pinned by ``bestanswer_disjoint_manifest.json``, since the same questions appear at
several lengths: ``train`` (798 questions only at <= 16k), ``eval`` (50 of the 199 at >= 32k), ``train_long`` (the
other 149).
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from importlib.resources import files
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

SETTINGS = ("8k", "16k", "32k", "128k")
_BA_URL = "http://opencompass.openxlab.space/utils/AdaLEval/stackselect_{setting}.json"

_DEFAULT_MANIFEST = str(files("ada_leval_real") / "bestanswer_disjoint_manifest.json")

# Above 32k the stored label doesn't match the candidate order, so the gold is found by the accepted answer's
# text (from the 32k file).
_LABEL_RELIABLE_SETTINGS = frozenset({"8k", "16k", "32k"})
_GOLD_REF_SETTING = "32k"


def _data_dir(data_dir: str | None = None) -> str:
    return data_dir or os.environ.get("ADALEVAL_DATA_DIR") or os.path.join(
        os.path.expanduser("~"), ".cache", "rlm_train_real", "ada_leval")

_BA_ROOT_PROMPT = (
    "You are given a question, followed by a list of candidate answers each "
    "marked A1, A2, A3, and so on — there can be hundreds of them. Exactly one "
    "candidate is the accepted answer to the question; every other candidate is "
    "a real answer to a different, unrelated question, included only as a "
    "distractor. Answer with ONLY the label of the single best candidate, e.g. A17."
)


def _ensure_ba_data(setting: str, data_dir: str) -> str:
    """Local path to ``stackselect_<setting>.json``, downloaded on first use."""
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; expected one of {SETTINGS}")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"stackselect_{setting}.json")
    if not os.path.exists(path):
        tmp = path + ".part"
        urllib.request.urlretrieve(_BA_URL.format(setting=setting), tmp)
        os.replace(tmp, path)
    return path


def _build_ba_prompt(question: str, all_answers: list[str]) -> str:
    """Render the question + labeled candidate answers (A1..AN in list order)."""
    lines = [f"Question:\n{question}\n", "Candidate answers:"]
    for i, ans in enumerate(all_answers, 1):
        lines.append(f"A{i}: {ans}")
    return "\n".join(lines)


def _manifest_qids(manifest_path: str | None, manifest_split: str) -> set[str]:
    """Question ids of one manifest split (packaged manifest when ``manifest_path`` is None)."""
    with open(manifest_path or _DEFAULT_MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest_split not in manifest:
        raise ValueError(f"split {manifest_split!r} not in manifest (have {list(manifest)})")
    return {str(q) for q in manifest[manifest_split]}


def _label_index(label: Any) -> int:
    """Parse an ``A<k>`` label to a 0-based index into ``all_answers``."""
    m = _BA_LABEL_RE.search(str(label))
    if not m:
        raise ValueError(f"unparseable BestAnswer label {label!r}")
    return int(m.group(1)) - 1


def _recover_gold_map(qids: set[str] | None, data_dir: str) -> dict[str, str]:
    """{question_id: accepted-answer TEXT} from the reliable reference setting."""
    path = _ensure_ba_data(_GOLD_REF_SETTING, data_dir)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    out: dict[str, str] = {}
    for item in raw:
        qid = str(item["question_id"])
        if qids is not None and qid not in qids:
            continue
        if qid in out:
            continue
        out[qid] = item["all_answers"][_label_index(item["answer"])].strip()
    return out


def _build_ba_dataset(
    setting: str,
    num_examples: int,
    data_dir: str,
    include_qids: set[str] | None = None,
) -> Dataset:
    path = _ensure_ba_data(setting, data_dir)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    gold_texts = (
        None if setting in _LABEL_RELIABLE_SETTINGS
        else _recover_gold_map(include_qids, data_dir)
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        qid = str(item["question_id"])
        if include_qids is not None and qid not in include_qids:
            continue
        if qid in seen:
            continue
        seen.add(qid)
        context = _build_ba_prompt(item["question"], item["all_answers"])
        if gold_texts is None:
            gold = str(item["answer"])
        else:
            text = gold_texts.get(qid)
            matches = (
                [i for i, a in enumerate(item["all_answers"]) if a.strip() == text]
                if text is not None else []
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"BestAnswer setting={setting!r} qid={qid}: could not uniquely "
                    f"recover gold (ref_text={'present' if text else 'MISSING'}, "
                    f"matches={len(matches)}). Reference setting {_GOLD_REF_SETTING} "
                    f"must contain every eval question."
                )
            gold = f"A{matches[0] + 1}"
        root_prompt = f"{_BA_ROOT_PROMPT}\n\nQuestion:\n{item['question']}"
        meta = {
            "answer": gold,
            "context": context,
            "root_prompt": root_prompt,
            "question_id": qid,
            "setting": setting,
            "n_choices": len(item["all_answers"]),
        }
        rows.append({
            "example_id": len(rows),
            "prompt": [{"role": "user", "content": root_prompt}],
            "answer": gold,
            "info": json.dumps(meta),
        })
        if num_examples > 0 and len(rows) >= num_examples:
            break
    if not rows:
        raise RuntimeError(
            f"BestAnswer setting={setting!r}: built 0 rows "
            f"(include_qids={None if include_qids is None else len(include_qids)})."
        )
    return Dataset.from_list(rows)


_BA_LABEL_RE = re.compile(r"[Aa]\s*0*(\d+)")


def _coerce_label(pred: Any) -> str | None:
    """Normalize ``"A69"``, ``"a 69"``, ``"Answer: A69"`` or ``"69"`` to ``A69``."""
    if pred is None:
        return None
    s = str(pred).strip()
    if not s:
        return None
    m = _BA_LABEL_RE.search(s)
    if m:
        return f"A{int(m.group(1))}"
    m2 = re.fullmatch(r"0*(\d+)", s)
    if m2:
        return f"A{int(m2.group(1))}"
    return None


async def _score_ba(info: Any, state: vf.State, **_kwargs: Any) -> float:
    meta = json.loads(info) if isinstance(info, str) else (info or {})
    gold = _coerce_label(meta.get("answer"))
    pred = _coerce_label(state.get("rlm_final_answer"))
    return 1.0 if (pred is not None and gold is not None and pred == gold) else 0.0


def load_environment(
    *,
    task: str = "bestanswer",
    setting: str = "16k",
    num_examples: int = -1,
    data_dir: str | None = None,
    manifest_path: str | None = None,
    manifest_split: str | None = None,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 1,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    if task != "bestanswer":
        raise ValueError(f"unknown task {task!r}; expected 'bestanswer'")
    # A manifest split without a path reads the packaged manifest.
    include = (
        _manifest_qids(manifest_path, manifest_split)
        if (manifest_path or manifest_split) else None
    )
    dataset = _build_ba_dataset(
        setting=setting, num_examples=num_examples, data_dir=_data_dir(data_dir),
        include_qids=include,
    )

    rubric = rlm_train.RLMTrainRubric(
        correctness=_score_ba, weight=1.0,
        min_iterations=min_iterations, min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment", "SETTINGS"]
