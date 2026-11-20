"""CTC-Core: a fixed evaluation split over CTC-Bench cells (task x rung), frozen in ``core_v1.json``.

Per cell, rows are ordered by a hash of the question key; the first 100 are ``test`` and the next 20 ``dev`` (all 125
are ``test`` at each family's longest rung). Also registers ``xabsence_paraphrase``, which is not in the 22-row suite.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import rlm_train
import verifiers as vf
from datasets import Dataset

from . import env as E

# eval-only rows outside the 22-row suite roster
EXTRA_ROSTER: dict[str, E.Row] = {
    "xabsence_paraphrase": E.Row("xabsence", "high", "O(N^2)", E._L32, None,
                                 "eval-only (not in the 22-row suite roster): PubMed paraphrase build"),
}

CORE_V1 = Path(__file__).with_name("core_v1.json")


def row_of(task: str) -> E.Row:
    return EXTRA_ROSTER[task] if task in EXTRA_ROSTER else E.ROSTER[E.resolve_task(task)]


def spec_of(task: str):
    return E._registry.get(row_of(task).spec)


@lru_cache(maxsize=4)
def manifest(version: str = "v1") -> dict:
    return json.loads((CORE_V1 if version == "v1" else Path(version)).read_text())


def core_rows(*, version: str = "v1", part: str = "test", cells: str | list[str] | None = None,
              start_id: int = 0) -> list[dict]:
    """Rows of the selected cells (e.g. ``"outlier@r128k"``, a tier like ``"32k"``, or None for all)."""
    man = manifest(version)
    want = cells.split(",") if isinstance(cells, str) else cells
    out = []
    for cell in man["cells"]:
        name = f"{cell['task']}@{cell['rung']}"
        if want and name not in want and cell["tier"] not in want:
            continue
        spec, data = spec_of(cell["task"]), E._hf_rung(cell["task"], cell["rung"])
        row = row_of(cell["task"])
        for r in cell[part]:
            ex = data[r["hf_row"]]
            context, task_text, question = E.render(ex, spec)
            meta = {
                "context": context,
                "root_prompt": f"{task_text}\n\n{E.submit_hint(task_text)}",
                "ctc_task": cell["task"], "ctc_spec": spec.name, "ctc_class": row.ctc_class,
                "ctc_metric": spec.primary_metric, "n_docs": len(ex.get("documents") or []),
                "ctc_core": {"version": man["version"], "cell": name, "tier": cell["tier"], "part": part,
                             "hf_row": r["hf_row"], **{k: v for k, v in r.items() if k != "hf_row"}},
                "example": E._slim(ex),
            }
            out.append({"example_id": start_id + len(out), "prompt": [{"role": "user", "content": question}],
                        "answer": E.gold_target(ex, spec), "info": json.dumps(meta, ensure_ascii=False)})
    if not out:
        raise RuntimeError(f"CTC-Core {version}: no rows for cells={cells!r} part={part!r}")
    return out


def load_environment(*, core: str = "v1", part: str = "test", cells: str | list[str] | None = None,
                     shuffle_seed: int | None = None, max_iterations: int = 20, sub_max_tokens: int = 4096,
                     min_iterations: int = 1, min_subcall: int = 0,
                     **kwargs: Any) -> vf.Environment:
    """RLMTrainEnv over CTC-Core rows; scoring is the same as ``env.load_environment``."""
    rows = core_rows(version=core, part=part, cells=cells)
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(rows)
        for i, r in enumerate(rows):
            r["example_id"] = i
    rubric = rlm_train.RLMTrainRubric(correctness=E._score, weight=1.0,
                                      min_iterations=min_iterations, min_subcall=min_subcall)
    rubric.add_metric(E.ctc_primary_raw)
    rubric.add_metric(E.ctc_parse_ok)
    return rlm_train.RLMTrainEnv(dataset=Dataset.from_list(rows), max_iterations=max_iterations,
                                 sub_sampling_args={"max_tokens": sub_max_tokens}, rubric=rubric, **kwargs)
