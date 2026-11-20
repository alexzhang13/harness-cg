"""GoLongRL (`golongrl-real`): long-context RLVR rows from the ``table`` (row-multiset IoU) and ``summary`` (ROUGE-L)
families. Rows whose own gold scores below 0.999 are dropped.
"""

from __future__ import annotations

import ast
import json
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

from . import metrics as M
from .parse import ParseError, parse_row

REPO_ID = "Kwai-Klear/GoLongRL"
_SHARD_FMT = "data/train-{i:05d}-of-00011.parquet"
N_SHARDS = 11

_LBP = M.TASK_METRIC_CONFIG

TASKS: dict[str, dict[str, Any]] = {
    "table": {
        "ability": "Multi-table structured extraction",
        "metric": "iou",
        "modes": ("multitableqa_pretraining",),
    },
    "summary": {
        "ability": "Long document summarization",
        "metric": "rouge_l",
        "modes": (
            "T4.1 Global-Coverage Constrained Summary",
            "T4.2 Query-Focused Summary",
        ),
    },
}

ALL_TASKS = tuple(TASKS)

_MODE_TO_TASK = {m: t for t, spec in TASKS.items() for m in spec["modes"]}
assert set(_MODE_TO_TASK) == set(_LBP) | {
    "multitableqa_pretraining",
}, "TASKS mode lists drifted from metrics.TASK_METRIC_CONFIG"


_QUOTED = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")

_TRUNCATED_LIST_HEAD = re.compile(r"\[\s*['\"]")

_CONTAINER_METRICS = frozenset({"iou"})


def _coerce_literal(text: str, *, salvage: bool = True) -> Any | None:
    """Recover a list/dict the model wrote to ``answer["content"]``."""
    s = text.strip()
    if not s or s[0] not in "[{":
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            value = loader(s)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(value, (list, dict)):
            return value
    if salvage and _TRUNCATED_LIST_HEAD.match(s):
        items = [(a or b) for a, b in _QUOTED.findall(s)]
        if items:
            return items
    return None


def _to_answer_text(final: Any, *, salvage: bool = True) -> str:
    """Coerce the final answer into the newline-delimited text the metrics expect."""
    if final is None:
        return ""
    if isinstance(final, str):
        recovered = _coerce_literal(final, salvage=salvage)
        return final if recovered is None else _to_answer_text(recovered, salvage=salvage)
    if isinstance(final, (list, tuple)):
        out = []
        for x in final:
            if isinstance(x, (list, tuple)):
                out.append(" ".join(str(y) for y in x))
            elif isinstance(x, dict):
                out.append(json.dumps(x, ensure_ascii=False))
            else:
                out.append(str(x))
        return "\n".join(s.strip() for s in out if str(s).strip())
    if isinstance(final, dict):
        return json.dumps(final, ensure_ascii=False)
    return str(final)


def make_score():
    async def golongrl_reward(answer: str, state: vf.State, **_kw: Any) -> float:
        gold = json.loads(answer) if isinstance(answer, str) else (answer or {})
        final = state.get("rlm_final_answer")
        if final is None:
            final = state.get("final_answer")
        metric = M.metric_name(gold["data_source"], gold["reward_mode"])
        return M.compute_score(
            data_source=gold["data_source"],
            reward_mode=gold["reward_mode"],
            prediction=_to_answer_text(final, salvage=metric in _CONTAINER_METRICS),
            doc_ids=gold.get("doc_ids"),
            golden_label=gold.get("golden_label", ""),
            language=gold.get("language", ""),
        )

    return golongrl_reward


async def golongrl_answered(state: vf.State, **_kw: Any) -> float:
    """Did the rollout submit a non-empty answer?"""
    final = state.get("rlm_final_answer")
    return 1.0 if _to_answer_text(final, salvage=False).strip() else 0.0


def _resolve_modes(task: str | list[str] | tuple[str, ...]) -> set[str]:
    tasks = [task] if isinstance(task, str) else list(task)
    modes: set[str] = set()
    for t in tasks:
        if t not in TASKS:
            raise ValueError(f"Unknown task {t!r}. Choose from {ALL_TASKS}")
        modes.update(TASKS[t]["modes"])
    return modes


def _shard_path(i: int) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO_ID, _SHARD_FMT.format(i=i), repo_type="dataset")


def _index_shards(
    shards: list[int], modes: set[str], split: str, language: str | None
) -> list[tuple[int, int]]:
    """Cheap pass 1: (shard, row) for every row matching the tag filters."""
    import pyarrow.parquet as pq

    cols = [
        "data_source",
        "extra_info.reward_mode",
        "extra_info.split",
        "reward_model.ground_truth.summary",
    ]
    index: list[tuple[int, int]] = []
    for s in shards:
        table = pq.read_table(_shard_path(s), columns=cols)
        rm = table.column("reward_mode").to_pylist()
        sp = table.column("split").to_pylist()
        lg = table.column("summary").to_pylist()
        for i in range(len(rm)):
            if rm[i] not in modes:
                continue
            if split and sp[i] != split:
                continue
            if language is not None and (lg[i] or "") != language:
                continue
            index.append((s, i))
    return index


def _fetch_rows(index: list[tuple[int, int]]):
    """Pass 2: lazily yield the full rows named by ``index``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    by_shard: dict[int, set[int]] = {}
    for s, i in index:
        by_shard.setdefault(s, set()).add(i)
    for s, rows in by_shard.items():
        pf = pq.ParquetFile(_shard_path(s))
        off = 0
        for rg in range(pf.num_row_groups):
            n = pf.metadata.row_group(rg).num_rows
            lo, hi, off = off, off + n, off + n
            local = sorted(r - lo for r in rows if lo <= r < hi)
            if not local:
                continue
            table = pf.read_row_group(rg)
            yield from table.take(pa.array(local)).to_pylist()


def _gold_is_scorable(p: dict[str, Any]) -> bool:
    """Does the row's own gold answer max out the row's own metric?"""
    gold = p["gold"]
    text = M.gold_answer_text(
        data_source=p["data_source"],
        reward_mode=p["reward_mode"],
        doc_ids=gold["doc_ids"],
        golden_label=gold["golden_label"],
    )
    if not text.strip():
        return False
    return (
        M.compute_score(
            data_source=p["data_source"],
            reward_mode=p["reward_mode"],
            prediction=text,
            doc_ids=gold["doc_ids"],
            golden_label=gold["golden_label"],
            language=p["language"],
        )
        >= 0.999
    )


def build_dataset(
    *,
    task: str | list[str] | tuple[str, ...] = "table",
    language: str | None = None,
    min_context_chars: int = 200,
    num_examples: int = -1,
    seed: int = 42,
) -> Dataset:
    split = "train"
    max_root_prompt_chars = 8000
    modes = _resolve_modes(task)
    shard_list = list(range(N_SHARDS))

    index = _index_shards(shard_list, modes, split, language)
    if seed is not None:
        random.Random(seed).shuffle(index)

    rows: list[dict] = []
    drops: dict[str, int] = {}
    seen = 0
    target = num_examples if num_examples and num_examples > 0 else len(index)

    def _consume(chunk: list[tuple[int, int]]) -> None:
        nonlocal seen
        for rec in _fetch_rows(chunk):
            seen += 1
            try:
                p = parse_row(
                    rec,
                    max_root_prompt_chars=max_root_prompt_chars,
                    min_context_chars=min_context_chars,
                )
            except ParseError as e:
                drops[str(e)] = drops.get(str(e), 0) + 1
                continue
            if not _gold_is_scorable(p):
                drops["gold_not_scorable"] = drops.get("gold_not_scorable", 0) + 1
                continue
            info = {
                "context": p["context"],
                "root_prompt": p["root_prompt"],
                "context_kind": p["context_kind"],
                "context_chars": p["context_chars"],
                "data_source": p["data_source"],
                "reward_mode": p["reward_mode"],
                "golongrl_task": _MODE_TO_TASK[p["reward_mode"]],
                "metric": p["metric"],
                "language": p["language"],
                "style": p["style"],
            }
            gold = {
                "data_source": p["data_source"],
                "reward_mode": p["reward_mode"],
                "language": p["language"],
                **p["gold"],
            }
            rows.append(
                {
                    "prompt": [{"role": "user", "content": p["question"][:4000]}],
                    "answer": json.dumps(gold, ensure_ascii=False),
                    "info": json.dumps(info, ensure_ascii=False),
                }
            )
            if len(rows) >= target:
                return

    pos, chunk = 0, max(4 * target, 32)
    while len(rows) < target and pos < len(index):
        _consume(index[pos : pos + chunk])
        pos += chunk
        chunk *= 2

    if not rows:
        raise ValueError(
            f"No GoLongRL examples for task={task!r} language={language!r} "
            f"min_context_chars={min_context_chars} (scanned {seen} candidate rows, drops={drops})"
        )
    random.Random(seed).shuffle(rows)
    for i, r in enumerate(rows):
        r["example_id"] = i
    return Dataset.from_list(rows)


def load_environment(
    *,
    task: str | list[str] | tuple[str, ...] = "table",
    language: str | None = None,
    min_context_chars: int = 200,
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load GoLongRL wired through ``RLMTrainEnv``."""
    dataset = build_dataset(
        task=task,
        language=language,
        min_context_chars=min_context_chars,
        num_examples=num_examples,
        seed=seed,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=make_score(),
        weight=1.0,
        min_subcall=min_subcall,
    )
    rubric.add_metric(golongrl_answered)
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["ALL_TASKS", "TASKS", "build_dataset", "load_environment", "make_score"]
