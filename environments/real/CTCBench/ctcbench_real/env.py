"""CTC-Bench (`ctcbench-real`): each of the 22 suite rows is a sub-environment, picked with ``task=``.

Prompt, parser and scorer come from the vendored ``ctc`` package (``_vendor/``). The numbered corpus goes into
``context``; the task text and question go into the root prompt. Train rows that overlap the published eval ladders
are dropped at load time; ``split_version="v1"`` applies the stricter frozen split in ``split_v1.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

from ._vendor.ctc.eval.stopping import STOP_PRESETS
from ._vendor.ctc.eval.stopping import apply as _apply_stop
from ._vendor.ctc.format import registry as _registry
from ._vendor.ctc.format.prompts import GROUPING_INSTRUCTION
from ._vendor.ctc.tasks import load_all as _load_all_specs
from ._vendor.ctc.tasks._grouping import make_grouping_spec

HF_DATASET = "PrasannSinghal/ctc-suite-eval"
# Generator-built train rows live under $CTCBENCH_TRAIN_DIR (default ~/ctcbench_data/train); see build_train.sh.
DEFAULT_TRAIN_DIR = os.path.expanduser("~/ctcbench_data/train")
TRAIN_DIR_ENV = "CTCBENCH_TRAIN_DIR"

_load_all_specs()
# `grouping` is not in the vendored canonical set; build it from the grouping_labeled factory.
if "grouping" not in _registry.names():
    _registry.register(
        make_grouping_spec(
            name="grouping",
            description="Partition abstracts into their (unnamed) field clusters.",
            instruction=GROUPING_INSTRUCTION,
            rungs=("2k", "4k", "8k", "16k", "32k"),
            query_builder=lambda ex: (
                f"{GROUPING_INSTRUCTION}\n\n{ex['queries'][0]}" if ex.get("queries") else GROUPING_INSTRUCTION
            ),
            sources=("openalex",),
        )
    )

_L32 = ("r2k", "r4k", "r8k", "r16k", "r32k")
_FULL = _L32 + ("r64k", "r128k", "r256k", "r512k", "r1m")


@dataclass(frozen=True)
class Row:
    """One suite row. ``train`` is the generator dir, or None to split the published ladder in half."""

    spec: str
    ctc_class: str
    complexity: str
    rungs: tuple[str, ...]
    train: str | None
    note: str = ""


# The 22-row suite roster, keyed by HF config name.
ROSTER: dict[str, Row] = {
    "fiqa": Row("retrieval", "low", "O(N)", _FULL, "fiqa"),
    "nq": Row("retrieval", "low", "O(N)", _FULL, "nq"),
    "hotpotqa": Row("retrieval", "low", "O(N)", _FULL, "hotpotqa"),
    "qdmatch_fiqa": Row("qdmatch", "high", "O(N^2)", _FULL[:-1], None, "no upstream generator: half-split"),
    "qdmatch_nq": Row("qdmatch", "high", "O(N^2)", _FULL, "qdmatch_nq"),
    "qdmatch_hpqa": Row("qdmatch", "high", "O(N^2)", _FULL[:-2], "qdmatch_hpqa"),
    "outlier_amzn": Row("outlier", "low", "O(N)", _FULL, "outlier_review"),
    "outlier": Row("outlier", "high", "O(NM)", _FULL, "outlier"),
    "outlier_fixedM": Row("outlier", "low", "O(N)", _FULL[:-1], None, "K pinned at 3; half-split"),
    "oolong": Row("oolong", "low", "O(N)", _FULL, None, "already covered by environments/real/oolong"),
    "grouping": Row("grouping", "high", "O(NM)", _L32, "grouping_labeled", "train rows from grouping_labeled"),
    "absence_gutenberg": Row("absence", "high", "O(N^2)", _L32[:-1], "absence"),
    "xabsence": Row("xabsence", "high", "O(N^2)", _L32, "xabsence",
                    "eval = Gutenberg exact-copy build; train generator = PubMed exact twins"),
    "rerank": Row("rerank", "low", "O(N)", _FULL[:-1], "rerank"),
    "msmarco": Row("retrieval", "low", "O(N)", _FULL[:-1], "rerank", "same MS MARCO rows as rerank, top-1 gold"),
    "reorder": Row("reorder", "high", "O(N^2)", _L32[:-1], "reorder"),
    "obliq_twitter": Row("retrieval", "low", "O(N)", _FULL, None, "already covered by environments/real/obliq"),
    "niah": Row("retrieval", "low", "O(N)", _FULL, None, "no upstream generator: half-split"),
    "contradiction_iid": Row("contradiction", "high", "O(N^2)", _FULL, "contradiction"),
    "strmatch": Row("strmatch", "high", "O(N^2)", _L32, "strmatch"),
    "textgroups": Row("textgroups", "high", "O(N^3)", _L32, "textgroups"),
    "scifact": Row("retrieval", "low", "O(N)", _L32, None,
                   "the published ladder is the whole SciFact test set, so generator rows all collide: half-split"),
}
ALIASES = {"hpqa": "hotpotqa", "absence": "absence_gutenberg", "contradiction": "contradiction_iid",
           "obliq": "obliq_twitter", "outlier_fixedm": "outlier_fixedM", "outlier_review": "outlier_amzn"}
ALL_TASKS = tuple(ROSTER)

# Specs whose question alone identifies an item (for the contamination check).
_QUERY_KEYED = {"retrieval", "rerank", "oolong"}

_SUBMIT = (
    "The REPL variable `context` holds the benchmark input exactly as it is given to a long-context model: "
    "the corpus followed by the task statement. When you are done, set `answer[\"content\"]` to ONE string "
    "that is your final answer in exactly the answer format stated above (it is read by the benchmark's own "
    "parser, so no extra words or lines). {brackets}Then set `answer[\"ready\"] = True`."
)
_BRACKETS = ("Copy that format literally: keep every label, square bracket and comma around the ids (`[12]`, "
             "not `12`), since an id written without its brackets is not read. ")
# Formats answered as "Label: [id1], [id2]". Only these get the bracket hint; it would break JSON answers.
_LABEL_LINE_RE = re.compile(r"^[A-Z][\w ]*: .*\[id\d*\]", re.M)


def submit_hint(task_text: str) -> str:
    """Submission paragraph for the root prompt."""
    keep = bool(_LABEL_LINE_RE.search(task_text))
    return _SUBMIT.format(brackets=_BRACKETS if keep else "")


def _hf_retry(fn, *, what: str, tries: int = 4):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"HF load failed after {tries} tries: {what}") from last


def resolve_task(task: str) -> str:
    t = ALIASES.get(task, task)
    if t not in ROSTER:
        raise ValueError(f"unknown CTC task {task!r}; expected one of {ALL_TASKS} (aliases {sorted(ALIASES)})")
    return t


def _norm_rung(r: str) -> str:
    r = str(r).strip().lower()
    return r if r.startswith("r") else "r" + r


def _normalize(row: dict) -> dict:
    """Parquet rows carry ``meta`` as a JSON string; the specs read it as a mapping."""
    for k in ("meta", "_meta"):
        v = row.get(k)
        if isinstance(v, str) and v[:1] in "{[":
            try:
                row[k] = json.loads(v)
            except json.JSONDecodeError:
                pass
    return row


@lru_cache(maxsize=256)
def _hf_rung(subset: str, rung: str) -> tuple[dict, ...]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = _hf_retry(
        lambda: hf_hub_download(HF_DATASET, f"data/{subset}/{rung}.parquet", repo_type="dataset"),
        what=f"{subset}/{rung}",
    )
    return tuple(_normalize(r) for r in pq.read_table(path).to_pylist())


def _flat_ints(x: Any) -> list[int]:
    if isinstance(x, (list, tuple)):
        return [i for v in x for i in _flat_ints(v)]
    try:
        return [int(x)]
    except (TypeError, ValueError):
        return []


def _gold_of(ex: dict, spec) -> Any:
    return ex.get(spec.extra.get("gold_field", "gold_doc_indices"))


def _unit_keys(ex: dict, spec) -> set[str]:
    """Keys identifying the underlying question across rungs (question text and gold document texts)."""
    docs = ex.get("documents") or []
    if spec.name == "reorder":
        idx = list(range(len(docs)))
    else:
        base = spec.gold_index_base
        idx = [i - base for i in _flat_ints(_gold_of(ex, spec))]
    texts = sorted({docs[i]["text"] for i in idx if 0 <= i < len(docs)})
    q = (ex.get("queries") or [""])[0] if ex.get("queries") else ""
    keys = {"g:" + hashlib.sha1("\x1e".join([q[:500]] + texts).encode()).hexdigest()}
    if spec.name in _QUERY_KEYED and q:
        keys.add("q:" + hashlib.sha1(q.encode()).hexdigest())
    return keys


def _half(ex: dict, spec) -> int:
    """0 = train half, 1 = eval half; all rungs of a question land on the same side."""
    k = min(k for k in _unit_keys(ex, spec) if k.startswith("g:"))
    return int(k[2:10], 16) % 2


@lru_cache(maxsize=32)
def _eval_keys(subset: str) -> frozenset[str]:
    spec = _registry.get(ROSTER[subset].spec)
    keys: set[str] = set()
    for rung in ROSTER[subset].rungs:
        if rung in _L32:
            for ex in _hf_rung(subset, rung):
                keys |= _unit_keys(ex, spec)
    return frozenset(keys)


def _split_prompt(prompt: str) -> tuple[str, str]:
    """(instruction header, ``### Input:`` body) of an alpaca-shaped CTC prompt."""
    _, _, rest = prompt.partition("### Instruction:\n")
    instr, _, rest = rest.partition("\n\n### Input:\n")
    body = rest.rsplit("\n\n### Response:", 1)[0]
    return instr.strip(), body


def render(ex: dict, spec) -> tuple[str, str, str]:
    """(context, task text, question) for one unified-format example."""
    instr, body = _split_prompt(spec.build_prompt(ex, query_position="after"))
    _, ask = _split_prompt(spec.build_prompt({**ex, "documents": []}, query_position="after"))
    if spec.name == "absence" and ex.get("queries"):
        ask = ask.replace(ex["queries"][0], "").strip() + (
            "\n\n(The second version of the corpus is the text that follows the numbered items in `context`.)"
        )
    unified = spec.unified or instr.startswith("You will be asked to do a long-context processing task")
    task = ask.strip() if unified else f"{instr}\n\n{ask.strip()}"
    question = (ex.get("queries") or [""])[0] if ex.get("queries") and spec.name != "absence" else ""
    return body, task.strip(), (question or ask.strip().split("\n")[0])[:2000]


def score_answer(answer: Any, ex: dict, spec) -> dict[str, float]:
    """Score like olmo-eval's CTCScorer. Returns all metrics plus ``parse_ok`` and ``primary``."""
    if answer is None:
        text = ""
    elif isinstance(answer, (list, tuple)):
        text = json.dumps(list(answer))
    else:
        text = str(answer)
    cleaned = _apply_stop(text.strip(), STOP_PRESETS[spec.stop])
    parsed = spec.parse(cleaned, int(ex.get("n_docs") or len(ex.get("documents") or [])))
    if spec.extra.get("score_takes_example"):
        gold = ex
    else:
        gold = _gold_of(ex, spec) or ex
    scored = {k: float(v) for k, v in spec.score(parsed, gold).items()}
    scored["parse_ok"] = float(parsed is not None)
    scored["primary"] = scored.get(spec.primary_metric, 0.0)
    return scored


def gold_target(ex: dict, spec) -> str:
    """Gold answer in the format the model must write (1-based ids, as shown in the prompt)."""
    import importlib

    if spec.name == "rerank" and ex.get("ce_scores"):
        from ._vendor.ctc.tasks.rerank.spec import TOP_K, _ce_pos_ref

        ref = _ce_pos_ref(ex, TOP_K)
        return ("Scored set -- each of these ids must be among the first 10 ids of the ranking, in any order "
                "(the rest of the ranking is not scored): " + ", ".join(f"[{i}]" for i in ref))
    mod = "grouping_labeled" if spec.name == "grouping" else spec.name
    try:
        build = importlib.import_module(f"{__package__}._vendor.ctc.tasks.{mod}.spec").build_target
        return str(build(ex))
    except Exception:  # noqa: BLE001
        return json.dumps(_gold_of(ex, spec) if _gold_of(ex, spec) is not None else ex.get("answers"))


def _slim(ex: dict) -> dict:
    """The example without document texts (scorers only need the count)."""
    out = {k: v for k, v in ex.items() if k != "documents"}
    out["n_docs"] = len(ex.get("documents") or [])
    return out


def _train_rows(subset: str, train_dir: str, rungs: tuple[str, ...]) -> list[dict]:
    row, spec = ROSTER[subset], _registry.get(ROSTER[subset].spec)
    if row.train is None:
        return [ex for r in rungs if r in row.rungs for ex in _hf_rung(subset, r) if _half(ex, spec) == 0]
    path = Path(train_dir) / row.train / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"CTC train split for {subset!r} not built: {path} missing. Run environments/real/CTCBench/build_train.sh "
            f"{row.train} (or set ${TRAIN_DIR_ENV})."
        )
    with open(path, encoding="utf-8") as f:
        exs = [_normalize(json.loads(line)) for line in f if line.strip()]
    bad = _eval_keys(subset)
    return [ex for ex in exs if not (_unit_keys(ex, spec) & bad)]


SPLIT_FILES = {"v1": Path(__file__).with_name("split_v1.json")}


@lru_cache(maxsize=4)
def _split_file(version: str) -> dict:
    if version not in SPLIT_FILES:
        raise ValueError(f"split_version must be 'legacy' or one of {sorted(SPLIT_FILES)}, got {version!r}")
    return json.loads(SPLIT_FILES[version].read_text())


def _versioned(exs: list[dict], subset: str, spec, split: str, version: str) -> list[dict]:
    """Apply a frozen split file: drop listed rows, then assign train / val by group."""
    sf = _split_file(version)
    if subset in sf["retired"]:
        raise ValueError(f"CTC task {subset!r} is retired under split_version={version!r}: {sf['retired'][subset]}")
    if subset not in sf["tasks"]:
        raise ValueError(f"CTC task {subset!r} has no split_version={version!r} entry (not a train task)")
    t = sf["tasks"][subset]
    drop, val = set(t["drop"]), set(t["val"])
    uid = lambda ex: min(k for k in _unit_keys(ex, spec) if k.startswith("g:"))  # noqa: E731
    out = [ex for ex in exs if uid(ex) not in drop and ((uid(ex) in val) == (split == "val"))]
    unknown = sum(1 for ex in exs if uid(ex) not in drop) - t["n_train"] - t["n_val"]
    if unknown:   # the train build differs from the one the split file was built from
        raise RuntimeError(f"CTC {subset!r}: {unknown:+d} train rows vs split_{version}.json; rebuild it")
    return out


GROUPING32_FILE = Path(__file__).with_name("grouping32_v1.json")


def frozen_ids(name: str, *, task: str, split: str, rungs: tuple[str, ...], seed: int, num_examples: int) -> list[int]:
    """example_ids of a frozen selection; ``"grouping32_test"`` = the 60 rows in ``grouping32_v1.json``."""
    if name != "grouping32_test":
        raise ValueError(f"unknown frozen id list {name!r}; expected 'grouping32_test'")
    man = json.loads(GROUPING32_FILE.read_text())
    want = (man["task"], man["split"], tuple(man["rungs"]), man["seed"], -1)
    if (task, split, rungs, seed, num_examples) != want:
        raise ValueError(f"ids={name!r} requires task/split/rungs/seed/num_examples = {want}")
    return list(man["test"])


def build_rows(
    task: str,
    *,
    split: str = "train",
    split_version: str = "legacy",
    num_examples: int = -1,
    seed: int = 42,
    rungs: str | list[str] = "r8k,r16k,r32k",
    train_dir: str | None = None,
    start_id: int = 0,
) -> list[dict]:
    subset = resolve_task(task)
    row, spec = ROSTER[subset], _registry.get(ROSTER[subset].spec)
    rung_list = tuple(_norm_rung(r) for r in (rungs.split(",") if isinstance(rungs, str) else rungs))
    if split == "val" and split_version == "legacy":
        raise ValueError("split='val' needs split_version='v1'")
    if split in ("train", "val"):
        exs = _train_rows(subset, train_dir or os.environ.get(TRAIN_DIR_ENV) or DEFAULT_TRAIN_DIR, rung_list)
        if split_version != "legacy":
            exs = _versioned(exs, subset, spec, split, split_version)
    elif split == "eval":
        exs = [ex for r in rung_list if r in row.rungs for ex in _hf_rung(subset, r)
               if row.train is not None or _half(ex, spec) == 1]
    else:
        raise ValueError(f"split must be 'train', 'val' or 'eval', got {split!r}")
    # all-singleton gold clusters have no pairs, so even the gold answer scores 0
    exs = [ex for ex in exs if spec.primary_metric != "pairwise_f1"
           or any(len(g) > 1 for g in (_gold_of(ex, spec) or []))]
    random.Random(seed).shuffle(exs)
    if num_examples > 0:
        exs = exs[:num_examples]
    out = []
    for ex in exs:
        context, task_text, question = render(ex, spec)
        meta = {
            "context": context,
            "root_prompt": f"{task_text}\n\n{submit_hint(task_text)}",
            "ctc_task": subset,
            "ctc_spec": spec.name,
            "ctc_class": row.ctc_class,
            "ctc_metric": spec.primary_metric,
            "n_docs": len(ex.get("documents") or []),
            "example": _slim(ex),
        }
        out.append({
            "example_id": start_id + len(out),
            "prompt": [{"role": "user", "content": question}],
            "answer": gold_target(ex, spec),
            "info": json.dumps(meta, ensure_ascii=False),
        })
    if not out:
        raise RuntimeError(f"CTC task {subset!r} split={split!r} rungs={rung_list}: 0 rows")
    return out


def _metrics(info: Any, state: vf.State) -> dict[str, float]:
    cached = state.get("_ctc_metrics")
    if cached is not None:
        return cached
    meta = json.loads(info) if isinstance(info, str) else (info or {})
    spec = _registry.get(meta["ctc_spec"])
    m = score_answer(state.get("rlm_final_answer"), meta["example"], spec)
    state["_ctc_metrics"] = m
    return m


async def _score(info: Any, state: vf.State, **_kw: Any) -> float:
    return max(0.0, min(1.0, _metrics(info, state)["primary"]))


async def ctc_primary_raw(info: Any, state: vf.State, **_kw: Any) -> float:
    return _metrics(info, state)["primary"]


async def ctc_parse_ok(info: Any, state: vf.State, **_kw: Any) -> float:
    return _metrics(info, state)["parse_ok"]


def load_environment(
    *,
    task: str = "strmatch",
    split: str = "train",
    split_version: str = "legacy",
    num_examples: int = -1,
    seed: int = 42,
    rungs: str | list[str] = "r8k,r16k,r32k",
    train_dir: str | None = None,
    ids: str | list[int] | None = None,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_iterations: int = 1,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load CTC-Bench rows as an RLMTrainEnv.

    task: a ``ROSTER`` name, a comma-separated list, or ``"all"``. split: ``"train"``, ``"val"`` (v1 only) or
    ``"eval"``. rungs: e.g. ``"r8k,r16k,r32k"``. ids: a list of example_ids or ``"grouping32_test"``.
    """
    tasks = ALL_TASKS if task == "all" else tuple(resolve_task(t.strip()) for t in task.split(","))
    per_task = kwargs.pop("per_task_examples", None)
    rows: list[dict] = []
    for t in tasks:
        rows.extend(build_rows(t, split=split, split_version=split_version,
                               num_examples=num_examples if per_task is None else per_task, seed=seed, rungs=rungs,
                               train_dir=train_dir, start_id=len(rows)))
    if len(tasks) > 1:
        random.Random(seed).shuffle(rows)
        for i, r in enumerate(rows):
            r["example_id"] = i
    if ids is not None:
        if isinstance(ids, str):
            rung_list = tuple(_norm_rung(r) for r in (rungs.split(",") if isinstance(rungs, str) else rungs))
            ids = frozen_ids(ids, task=",".join(tasks), split=split, rungs=rung_list, seed=seed,
                             num_examples=num_examples if per_task is None else per_task)
        keep = {int(i) for i in ids}
        rows = [r for r in rows if r["example_id"] in keep]
    rubric = rlm_train.RLMTrainRubric(correctness=_score, weight=1.0,
                                      min_iterations=min_iterations, min_subcall=min_subcall)
    rubric.add_metric(ctc_primary_raw)
    rubric.add_metric(ctc_parse_ok)
    return rlm_train.RLMTrainEnv(
        dataset=Dataset.from_list(rows),
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["ALL_TASKS", "ROSTER", "build_rows", "frozen_ids", "load_environment", "render", "score_answer"]
