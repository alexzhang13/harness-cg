"""BABILong (`babilong-real`): bAbI facts hidden in PG19 text, with the official per-task prompts.

Rows whose gold doesn't score 1.0 under the metric are dropped. ``metric="wordbound"`` is the official
``compare_answers`` with whole-word label matching ("no" no longer matches "I don't know").
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train
from babilong_real._upstream import (
    DEFAULT_PROMPTS,
    TASK_LABELS,
    TASK_NAMES,
    TASKS,
    preprocess_output,
)


LENGTHS: tuple[str, ...] = (
    "0k", "1k", "2k", "4k", "8k", "16k", "32k",
    "64k", "128k", "256k", "512k", "1M", "10M",
)

_LENGTH_INDEX = {name: i for i, name in enumerate(LENGTHS)}

LENGTH_TOKENS: dict[str, int] = {
    "0k": 0, "1k": 1_000, "2k": 2_000, "4k": 4_000, "8k": 8_000,
    "16k": 16_000, "32k": 32_000, "64k": 64_000, "128k": 128_000,
    "256k": 256_000, "512k": 512_000, "1M": 1_000_000, "10M": 10_000_000,
}


@dataclass(frozen=True)
class _Source:
    name: str
    repo: str
    kind: str
    nominal_rows: int
    note: str


SOURCES: dict[str, _Source] = {
    "eval": _Source(
        "eval", "RMT-team/babilong", "json", 100,
        "official 100-sample eval set; the only source with 256k/512k/1M/10M",
    ),
    "train": _Source(
        "train", "RMT-team/babilong-train-5k-samples", "json", 5000,
        "official 5000-sample TRAIN split; qa1-qa10 up to 64k; disjoint from `eval`",
    ),
}

_DEFAULT_STREAM_THRESHOLD = 512 * 1024 * 1024


@lru_cache(maxsize=8)
def _repo_files(repo: str) -> tuple[str, ...]:
    from huggingface_hub import list_repo_files

    return tuple(list_repo_files(repo, repo_type="dataset"))


@lru_cache(maxsize=8)
def _availability(source: str) -> dict[str, tuple[str, ...]]:
    """``{task: (length, ...)}`` actually present in the source repo."""
    src = SOURCES[source]
    files = _repo_files(src.repo)
    out: dict[str, set[str]] = {}
    pat = re.compile(r"^data/(qa\d+)/(.+)\.json$")
    for f in files:
        m = pat.match(f)
        if m and m.group(2) in _LENGTH_INDEX:
            out.setdefault(m.group(1), set()).add(m.group(2))
    return {
        t: tuple(sorted(ls, key=lambda L: _LENGTH_INDEX[L]))
        for t, ls in sorted(out.items(), key=lambda kv: int(kv[0][2:]))
    }


def _files_for(source: str, task: str, length: str) -> list[str]:
    return [f"data/{task}/{length}.json"]


def _iter_json_array(fp: Any, *, limit: int | None = None, chunk_size: int = 1 << 20) -> Iterator[dict]:
    """Incrementally yield the top-level objects of a JSON array."""
    decoder = json.JSONDecoder()
    inc = codecs.getincrementaldecoder("utf-8")()
    buf = ""
    pos = 0
    eof = False
    started = False
    emitted = 0
    read_size = chunk_size

    def _pump() -> bool:
        """Append one chunk; return False at EOF."""
        nonlocal buf, eof, read_size
        if eof:
            return False
        raw = fp.read(read_size)
        if not raw:
            buf += inc.decode(b"", final=True)
            eof = True
            return False
        buf += inc.decode(raw)
        return True

    while True:
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if pos < len(buf):
                break
            if not _pump():
                return
        ch = buf[pos]
        if not started:
            if ch != "[":
                raise ValueError(f"expected a JSON array, found {ch!r}")
            started = True
            pos += 1
            continue
        if ch == "]":
            return
        while True:
            try:
                obj, end = decoder.raw_decode(buf, pos)
                break
            except json.JSONDecodeError:
                if not _pump():
                    raise
                read_size = min(read_size * 2, 64 << 20)
        yield obj
        emitted += 1
        buf = buf[end:]
        pos = 0
        read_size = chunk_size
        if limit is not None and emitted >= limit:
            return


def _iter_source_rows(
    source: str,
    task: str,
    length: str,
    *,
    limit: int | None,
) -> Iterator[dict]:
    """Yield ``{input, question, target}`` dicts for one (task, length) bucket."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    src = SOURCES[source]
    emitted = 0
    for rel in _files_for(source, task, length):
        remaining = None if limit is None else limit - emitted
        if remaining is not None and remaining <= 0:
            return
        use_stream = _remote_size(src.repo, rel) > _DEFAULT_STREAM_THRESHOLD and not _in_cache(src.repo, rel)
        if use_stream:
            fs = HfFileSystem()
            with fs.open(f"datasets/{src.repo}/{rel}", "rb") as fp:
                for row in _iter_json_array(fp, limit=remaining):
                    emitted += 1
                    yield row
        else:
            path = hf_hub_download(src.repo, rel, repo_type="dataset")
            with open(path, "rb") as fp:
                for row in _iter_json_array(fp, limit=remaining):
                    emitted += 1
                    yield row


@lru_cache(maxsize=4096)
def _remote_size(repo: str, rel: str) -> int:
    from huggingface_hub import HfApi

    try:
        info = HfApi().repo_info(repo, repo_type="dataset", files_metadata=True)
        for s in info.siblings:
            if s.rfilename == rel:
                return int(s.size or 0)
    except Exception:  # noqa: BLE001
        return 0
    return 0


def _in_cache(repo: str, rel: str) -> bool:
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo, rel, repo_type="dataset", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


_WORD_BOUND_CACHE: dict[str, re.Pattern[str]] = {}


def _label_re(label: str) -> re.Pattern[str]:
    pat = _WORD_BOUND_CACHE.get(label)
    if pat is None:
        pat = _WORD_BOUND_CACHE[label] = re.compile(rf"(?<!\w){re.escape(label)}(?!\w)")
    return pat


def compare_answers_wordbound(target: str, output: str, question: str, task_labels: list[str]) -> bool:
    """``compare_answers`` with word-boundary label matching."""
    output = preprocess_output(output)
    target = target.lower()
    labels = {label.lower() for label in task_labels}

    q = question.lower()
    labels_in_question = {label for label in labels if _label_re(label).search(q)}
    labels_in_output = {label for label in labels if _label_re(label).search(output)}
    labels_in_output = labels_in_output - labels_in_question

    if "," in target and len(target) > 3:
        subtargets = target.split(",")
        if all(t in labels_in_output for t in subtargets) and len(labels_in_output) == len(subtargets):
            return True
    else:
        if target in labels_in_output and len(labels_in_output) == 1:
            return True
    return False


METRICS = {
    "wordbound": compare_answers_wordbound,
}


def _final_answer_text(final: Any) -> str:
    """Coerce ``state["rlm_final_answer"]`` (str | list | dict | None) to text."""
    if final is None:
        return ""
    if isinstance(final, str):
        return final
    if isinstance(final, (list, tuple, set)):
        return ",".join(str(x).strip() for x in final if str(x).strip())
    if isinstance(final, dict):
        for key in ("content", "answer", "target", "final_answer"):
            if key in final:
                return _final_answer_text(final[key])
        return json.dumps(final)
    return str(final)


def score_row(meta: dict[str, Any], final: Any) -> float:
    """Official BABILong accuracy for one row: 1.0 if correct else 0.0."""
    fn = METRICS[meta.get("metric", "wordbound")]
    return 1.0 if fn(meta["target"], _final_answer_text(final), meta["question"], meta["task_labels"]) else 0.0


async def _score(info: Any, state: vf.State, **_kw: Any) -> float:
    meta = json.loads(info) if isinstance(info, str) else info
    final = state.get("rlm_final_answer")
    if final is None:
        final = state.get("final_answer")
    return score_row(meta, final)


DEFAULT_REPL_GUIDANCE = (
    "The text described above is not in this message. It has already been loaded "
    "into the REPL variable `context` (a Python `str`): the facts are short, plain "
    "declarative sentences scattered through unrelated book prose. Work over "
    "`context` in the REPL to find them, then submit your answer with "
    '`answer["content"] = "..."` and `answer["ready"] = True`. Submit only the '
    "answer in the format required above — no reasoning, no extra sentences, "
    "since anything else in the answer is scored as a wrong answer. "
    '`answer["content"]` must be a **plain string**, never a Python list or dict: '
    "if the answer has more than one part, write the parts in one string separated "
    "by a comma and no space (e.g. `\"milk,football\"`, `\"s,w\"`), exactly as the "
    "examples above show."
)


def label_set_note(task: str) -> str:
    """The closed-label-set sentence appended when ``state_label_set=True``."""
    labels = list(TASK_LABELS[task])
    return (
        f"Your answer must use only values from this closed set of {len(labels)}: "
        f"{', '.join(labels)}. Anything outside it — including any place or name "
        "that appears in the surrounding book prose — is scored as wrong."
    )


def build_root_prompt(
    task: str,
    question: str,
    *,
    state_label_set: bool = False,
) -> str:
    """Assemble the short root prompt from the official per-task templates."""
    tpl = DEFAULT_PROMPTS[task]
    examples = tpl["examples"]
    examples = re.sub(r"\s*<context>\s*$", "", examples).rstrip()
    parts = [tpl["instruction"].strip()]
    if examples:
        parts.append(examples)
    parts.append(tpl["post_prompt"].strip())
    parts.append(DEFAULT_REPL_GUIDANCE.strip())
    if state_label_set:
        parts.append(label_set_note(task))
    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(parts)


MAX_ROOT_PROMPT_CHARS = 4000


@dataclass
class BuildStats:
    source: str
    pairs: list[tuple[str, str]] = field(default_factory=list)
    raw_seen: int = 0
    kept: int = 0
    dropped_unscorable: int = 0
    dropped_malformed: int = 0
    unscorable_targets: dict[str, int] = field(default_factory=dict)

    @property
    def drop_rate(self) -> float:
        considered = self.raw_seen
        return 0.0 if considered <= 0 else 1.0 - self.kept / considered

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "pairs": [f"{t}/{L}" for t, L in self.pairs],
            "raw_seen": self.raw_seen,
            "kept": self.kept,
            "dropped_unscorable": self.dropped_unscorable,
            "dropped_malformed": self.dropped_malformed,
            "unscorable_targets": self.unscorable_targets,
            "drop_rate": round(self.drop_rate, 6),
        }


def _resolve_tasks(task: str | list[str] | None, avail: dict[str, tuple[str, ...]]) -> list[str]:
    if task is None or (isinstance(task, str) and task.lower() == "all"):
        return list(avail)
    names = task if isinstance(task, (list, tuple)) else [t.strip() for t in str(task).split(",")]
    out = []
    for n in names:
        if n not in TASKS:
            raise ValueError(f"Unknown babilong task {n!r}; choose from {TASKS} or 'all'")
        out.append(n)
    return out


def _resolve_lengths(length: str | list[str] | None, tasks: list[str], avail: dict[str, tuple[str, ...]]) -> list[str]:
    if length is None or (isinstance(length, str) and length.lower() == "all"):
        present: set[str] = set()
        for t in tasks:
            present |= set(avail.get(t, ()))
        return sorted(present, key=lambda L: _LENGTH_INDEX[L])
    names = length if isinstance(length, (list, tuple)) else [x.strip() for x in str(length).split(",")]
    out = []
    for n in names:
        if n not in _LENGTH_INDEX:
            raise ValueError(f"Unknown babilong length {n!r}; choose from {LENGTHS} or 'all'")
        out.append(n)
    return out


def _build_dataset(
    *,
    source: str,
    task: str | list[str] | None,
    length: str | list[str] | None,
    num_examples: int,
    per_pair_examples: int,
    metric: str,
    state_label_set: bool,
) -> tuple[Dataset, BuildStats]:
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; choose from {tuple(SOURCES)}")
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {tuple(METRICS)}")

    avail = _availability(source)
    tasks = _resolve_tasks(task, avail)
    lengths = _resolve_lengths(length, tasks, avail)

    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for L in lengths:
        for t in tasks:
            if L in avail.get(t, ()):
                pairs.append((t, L))
            else:
                missing.append(f"{t}/{L}")
    if not pairs:
        raise ValueError(
            f"No (task, length) pairs available for source={source!r}, task={task!r}, "
            f"length={length!r}. Availability: { {t: ls for t, ls in avail.items() if t in tasks} }"
        )

    stats = BuildStats(source=source, pairs=pairs)

    if per_pair_examples and per_pair_examples > 0:
        quota = per_pair_examples
    elif num_examples and num_examples > 0:
        quota = -(-num_examples // len(pairs))
    else:
        quota = 0

    def _rows_for(t: str, L: str) -> list[dict]:
        labels = list(TASK_LABELS[t])
        kept: list[dict] = []
        it = _iter_source_rows(source, t, L, limit=None)
        try:
            for idx, raw in enumerate(it):
                stats.raw_seen += 1
                haystack = raw.get("input")
                question = raw.get("question")
                target = raw.get("target")
                if not isinstance(haystack, str) or not haystack.strip():
                    stats.dropped_malformed += 1
                    continue
                if not isinstance(question, str) or not question.strip():
                    stats.dropped_malformed += 1
                    continue
                if not isinstance(target, str) or not target.strip():
                    stats.dropped_malformed += 1
                    continue
                n_chars = len(haystack)
                target = target.strip()
                question = question.strip()
                meta_for_check = {
                    "target": target, "question": question,
                    "task_labels": labels, "metric": metric,
                }
                gold_ok = score_row(meta_for_check, target) == 1.0
                if not gold_ok:
                    stats.unscorable_targets[target] = stats.unscorable_targets.get(target, 0) + 1
                    stats.dropped_unscorable += 1
                    continue
                root_prompt = build_root_prompt(t, question, state_label_set=state_label_set)
                assert root_prompt, f"empty root_prompt for {t}"
                assert len(root_prompt) <= MAX_ROOT_PROMPT_CHARS, (
                    f"root_prompt for {t} is {len(root_prompt)} chars "
                    f"(> {MAX_ROOT_PROMPT_CHARS}); the haystack must not leak in"
                )
                if n_chars >= 400:
                    assert haystack[:200] not in root_prompt, (
                        f"haystack prefix leaked into root_prompt for {t}/{L}"
                    )
                assert question in root_prompt

                info = {
                    "context": haystack,
                    "root_prompt": root_prompt,
                    "babilong_task": t,
                    "task_name": TASK_NAMES[t],
                    "length": L,
                    "nominal_tokens": LENGTH_TOKENS[L],
                    "source": source,
                    "source_repo": SOURCES[source].repo,
                    "source_index": idx,
                    "question": question,
                    "target": target,
                    "task_labels": labels,
                    "metric": metric,
                    "gold_scorable": gold_ok,
                    "context_chars": n_chars,
                }
                kept.append({
                    "example_id": len(kept),
                    "prompt": [{"role": "user", "content": question}],
                    "answer": target,
                    "info": json.dumps(info),
                })
                stats.kept += 1
                if quota and len(kept) >= quota:
                    break
        finally:
            it.close()
        return kept

    per_pair = {pair: _rows_for(*pair) for pair in pairs}

    rows: list[dict] = []
    depth = 0
    while True:
        added = False
        for pair in pairs:
            bucket = per_pair[pair]
            if depth < len(bucket):
                rows.append(bucket[depth])
                added = True
                if num_examples and num_examples > 0 and len(rows) >= num_examples:
                    added = False
                    break
        if not added:
            break
        depth += 1
        if num_examples and num_examples > 0 and len(rows) >= num_examples:
            break

    if not rows:
        raise ValueError(
            f"No babilong rows survived filtering: source={source!r} task={task!r} "
            f"length={length!r} stats={stats.as_dict()}"
        )

    for i, r in enumerate(rows):
        r["example_id"] = i
    return Dataset.from_list(rows), stats


def load_environment(
    *,
    source: str = "eval",
    task: str | list[str] | None = "qa1",
    length: str | list[str] | None = "32k",
    seed: int = 42,
    num_examples: int = -1,
    per_pair_examples: int = -1,
    metric: str = "wordbound",
    state_label_set: bool = False,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load BABILong wired through ``RLMTrainEnv``."""
    dataset, stats = _build_dataset(
        source=source,
        task=task,
        length=length,
        num_examples=num_examples,
        per_pair_examples=per_pair_examples,
        metric=metric,
        state_label_set=state_label_set,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score, weight=1.0, min_subcall=min_subcall,
    )
    env = rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )
    env.babilong_build_stats = stats.as_dict()
    return env


__all__ = [
    "LENGTHS",
    "LENGTH_TOKENS",
    "SOURCES",
    "TASKS",
    "TASK_NAMES",
    "TASK_LABELS",
    "METRICS",
    "build_root_prompt",
    "label_set_note",
    "compare_answers_wordbound",
    "load_environment",
    "score_row",
]
