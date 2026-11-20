"""LoongRL / KeyChain-15K (`loongrl-real`): multi-hop QA over ~120 Wikipedia passages.

``variant="keychain"`` hides the question at the end of a UUID key-value chain starting from ``start_uuid``, among
decoy chains. Train / eval rows are split by a stable hash; rows whose gold can't be found in the passages are dropped.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import string
from collections import Counter
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

_REPO = "OldKingMeister/LoongRL-Train-Data"

SOURCES = ("2wikimqa", "hotpotqa", "musique")
VARIANTS = ("plain", "keychain")

_DIRS: dict[tuple[str, str], str] = {
    (src, "plain"): f"{src}_qwen_filtered_start_idx0_end_idx2500_seq16384"
    for src in SOURCES
}
_DIRS.update(
    {
        (src, "keychain"): (
            f"{src}_filtered_distractor_256_start_idx2500_end_idx5000_seq16384"
        )
        for src in SOURCES
    }
)

_PLAIN_PREFIX = "The following are given passages.\n"
_PLAIN_QCUE = "\n\nQuestion: "

_KC_PREFIX = "Please read the following text.\n"
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_KC_TAIL_RE = re.compile(
    r"\n\nIn the context above, there is one correct question to answer\. "
    r"The correct question can only be found by following the correct consecutive "
    r"chain of key:value pairs encoded with UUID strings "
    r"\(e\.g\., f81d4fae-7dec-11d0-a765-00a0c91e6bf6\), "
    r"starting from (" + _UUID + r")\.$"
)

_PASSAGE_SPLIT_RE = re.compile(r"\n\n(?=Passage \d+:\n)")
_PAIR_RE = re.compile(r'\{"(' + _UUID + r')": "(.*?)"\}\.')

_EXPECTED_PAIRS = 64
_EXPECTED_CHAINS = 16


class ParseError(ValueError):
    """A row did not match its variant's upstream template exactly."""


def parse_plain(content: str, question: str) -> dict[str, Any]:
    """Split an upstream ``custom_longcontextqa_`` prompt into haystack + question."""
    if not content.startswith(_PLAIN_PREFIX):
        raise ParseError("missing 'The following are given passages.' prefix")
    suffix = _PLAIN_QCUE + question
    if not content.endswith(suffix):
        raise ParseError("prompt does not end with '\\n\\nQuestion: <input_question>'")
    body = content[len(_PLAIN_PREFIX) : len(content) - len(suffix)]
    passages = _PASSAGE_SPLIT_RE.split(body)
    if not passages or not passages[0].startswith("Passage 1:\n"):
        raise ParseError("haystack does not start with 'Passage 1:'")
    if "\n\n".join(passages) != body:
        raise ParseError("passage split is not loss-free")
    return {"passages": passages, "question": question}


def parse_keychain(content: str, question: str) -> dict[str, Any]:
    """Split an upstream ``custom_longcontext_needle_qa_`` prompt."""
    if not content.startswith(_KC_PREFIX):
        raise ParseError("missing 'Please read the following text.' prefix")
    tail = _KC_TAIL_RE.search(content)
    if tail is None:
        raise ParseError("missing/altered trailing KeyChain instruction")
    start_uuid = tail.group(1)
    body = content[len(_KC_PREFIX) : tail.start()]

    passages = _PASSAGE_SPLIT_RE.split(body)
    if not passages or not passages[0].startswith("Passage 1:\n"):
        raise ParseError("haystack does not start with 'Passage 1:'")
    if "\n\n".join(passages) != body:
        raise ParseError("passage split is not loss-free")

    pairs = _PAIR_RE.findall(body)
    if len(pairs) != _EXPECTED_PAIRS:
        raise ParseError(f"expected {_EXPECTED_PAIRS} UUID pairs, found {len(pairs)}")
    links = dict(pairs)
    if len(links) != _EXPECTED_PAIRS:
        raise ParseError("duplicate UUID keys in chain")

    hops = _walk_chain(links, start_uuid)
    if not hops or hops[-1] != question:
        raise ParseError("UUID chain from start_uuid does not terminate at the question")

    terminals = [v for v in links.values() if not re.fullmatch(_UUID, v)]
    if len(terminals) != _EXPECTED_CHAINS:
        raise ParseError(
            f"expected {_EXPECTED_CHAINS} question-valued terminals, got {len(terminals)}"
        )
    return {
        "passages": passages,
        "start_uuid": start_uuid,
        "chain_hops": len(hops),
        "num_pairs": len(links),
        "num_decoy_questions": len(terminals) - 1,
    }


def _walk_chain(links: dict[str, str], start: str, limit: int = 32) -> list[str]:
    """Follow ``start -> links[start] -> ...`` and return the visited values."""
    seen: list[str] = []
    cur = start
    for _ in range(limit):
        nxt = links.get(cur)
        if nxt is None:
            break
        seen.append(nxt)
        cur = nxt
    return seen


_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Upstream ``normalize_answer``: lower, strip punctuation, drop articles."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def sub_em(prediction: str, ground_truth: str) -> float:
    """Upstream ``sub_em``: normalised substring containment in either direction."""
    gt = normalize_answer(ground_truth)
    pred = normalize_answer(prediction)
    if not gt or not pred:
        return 0.0
    return 1.0 if (gt in pred or pred in gt) else 0.0


def exact_match(prediction: str, ground_truth: str) -> float:
    """Upstream ``exact_match_score``."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def token_f1(prediction: str, ground_truth: str) -> float:
    """Upstream ``f1_score`` (token F1 with the yes/no/noanswer short-circuit)."""
    pred = normalize_answer(prediction)
    gt = normalize_answer(ground_truth)
    if pred in ("yes", "no", "noanswer") and pred != gt:
        return 0.0
    if gt in ("yes", "no", "noanswer") and pred != gt:
        return 0.0
    pt, gtt = pred.split(), gt.split()
    common = Counter(pt) & Counter(gtt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gtt)
    return 2 * precision * recall / (precision + recall)


_METRICS = {
    "sub_em": sub_em,
    "em": exact_match,
    "f1": token_f1,
}
_OVERFLOW_GUARDED = {"sub_em", "f1"}


def _to_text(final: Any) -> str:
    """Coerce ``state["rlm_final_answer"]`` (str | list | dict | None) to text."""
    if final is None:
        return ""
    if isinstance(final, str):
        s = final.strip()
        if s[:1] in ("{", "[") and s[-1:] in ("}", "]"):
            for loader in (json.loads, ast.literal_eval):
                try:
                    return _to_text(loader(s))
                except (ValueError, SyntaxError, TypeError):
                    continue
        return s
    if isinstance(final, (int, float, bool)):
        return str(final)
    if isinstance(final, dict):
        for key in ("answer", "content", "final_answer", "text", "value"):
            v = final.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return json.dumps(final, ensure_ascii=False)
    if isinstance(final, (list, tuple)):
        items = [_to_text(x) for x in final if _to_text(x)]
        if len(items) == 1:
            return items[0]
        return ", ".join(items)
    return str(final).strip()


def score_answer(
    prediction: Any,
    gold: Any,
    *,
    metric: str = "sub_em",
    answer_max_chars: int | None = 128,
) -> float:
    """Score one prediction against the gold answer list (max over golds)."""
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {sorted(_METRICS)}, got {metric!r}")
    fn = _METRICS[metric]
    pred = _to_text(prediction)
    if not pred:
        return 0.0
    golds = gold if isinstance(gold, (list, tuple)) else [gold]
    best = 0.0
    for g in golds:
        g = _to_text(g)
        if not g:
            continue
        if (
            answer_max_chars is not None
            and metric in _OVERFLOW_GUARDED
            and len(pred) - len(g) > answer_max_chars
        ):
            continue
        best = max(best, float(fn(pred, g)))
    return best


_KEYCHAIN_ROOT_PROMPT = (
    "find the one real question hidden in `context`, then answer it.\n\n"
    "`context` is a dict:\n"
    '  context["passages"]   -> list of ~100 Wikipedia passage strings (the haystack)\n'
    '  context["start_uuid"] -> the UUID your chain starts from\n\n'
    "Scattered through the passage text are JSON key-value pairs written as\n"
    '  {"<uuid>": "<value>"}.\n'
    "Each <value> is either another UUID (the next link) or a question string (a\n"
    "chain terminus). Exactly ONE chain, beginning at context[\"start_uuid\"],\n"
    "terminates in the real question you must answer. The other chains terminate\n"
    "in decoy questions — ignore them.\n\n"
    "So: extract the pairs from the passage text in the REPL (a regex over\n"
    "context[\"passages\"] works well), follow the chain from the start UUID to\n"
    "recover the real question, then answer that question from the passages. It is\n"
    "a multi-hop question, so you will usually need to combine facts from two or\n"
    "more passages.\n\n"
    "Submit ONLY the short answer span — a name, date, number, or yes/no — e.g.\n"
    '  answer["content"] = "Hergé"; answer["ready"] = True\n'
    "Do not restate the question and do not explain your reasoning in the answer."
)

_PLAIN_ROOT_PROMPT_TEMPLATE = (
    "answer the question below using the passages held in `context`.\n\n"
    "Question: {question}\n\n"
    "`context` is a dict:\n"
    '  context["passages"] -> list of ~120 Wikipedia passage strings (the haystack)\n'
    '  context["question"] -> the same question, verbatim\n\n'
    "Answer the question using the passages. It is a multi-hop question, so you\n"
    "will usually need to locate and combine facts from two or more passages;\n"
    "search context[\"passages\"] in the REPL rather than reading it all.\n\n"
    "Submit ONLY the short answer span — a name, date, number, or yes/no — e.g.\n"
    '  answer["content"] = "Hergé"; answer["ready"] = True\n'
    "Do not restate the question and do not explain your reasoning in the answer."
)

_ROOT_PROMPT_MAX_CHARS = 4000


def _answer_in_context(golds: list[str], body: str) -> bool:
    """Is some gold span recoverable from the haystack?"""
    low = body.lower()
    if any(g.lower() in low for g in golds if g):
        return True
    norm_body = normalize_answer(body)
    return any(normalize_answer(g) in norm_body for g in golds if g)


def _holdout_bucket(source: str, variant: str, index: int) -> float:
    """Stable per-row value in [0, 1), independent of ``seed`` and ``num_examples``."""
    key = f"{source}/{variant}/{index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(1 << 64)


def _resolve(name: str, allowed: tuple[str, ...], what: str) -> list[str]:
    if name == "all":
        return list(allowed)
    if name in allowed:
        return [name]
    raise ValueError(f"{what} must be one of {allowed + ('all',)}, got {name!r}")


def _build_dataset(
    *,
    source: str,
    variant: str,
    split: str,
    holdout_frac: float,
    num_examples: int,
    seed: int,
) -> tuple[Dataset, dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    if split not in ("train", "eval", "all"):
        raise ValueError(f"split must be 'train', 'eval' or 'all', got {split!r}")
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    sources = _resolve(source, SOURCES, "source")
    variants = _resolve(variant, VARIANTS, "variant")

    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"kept": 0, "dropped": 0, "drops": Counter(), "buckets": Counter()}

    for src in sources:
        for var in variants:
            keychain = var == "keychain"
            path = hf_hub_download(
                _REPO, f"{_DIRS[(src, var)]}/train.parquet", repo_type="dataset"
            )
            cols = ["index", "length", "data_source", "prompt", "reward_model", "extra_info"]
            records = pq.read_table(path, columns=cols).to_pylist()

            for rec in records:
                idx = int(rec["index"])
                bucket = _holdout_bucket(src, var, idx)
                is_eval = bucket < holdout_frac
                if split == "train" and is_eval:
                    continue
                if split == "eval" and not is_eval:
                    continue

                content = rec["prompt"][0]["content"]
                question = rec["extra_info"]["input_question"]
                golds = list(rec["reward_model"]["ground_truth"])
                if not golds or not question:
                    stats["dropped"] += 1
                    stats["drops"][f"{src}/{var}:empty-gold-or-question"] += 1
                    continue

                try:
                    parsed = (
                        parse_keychain(content, question)
                        if keychain
                        else parse_plain(content, question)
                    )
                except ParseError as exc:
                    stats["dropped"] += 1
                    stats["drops"][f"{src}/{var}:{exc}"] += 1
                    continue

                passages = parsed["passages"]
                body = "\n\n".join(passages)

                if normalize_answer(golds[0]) not in ("yes", "no"):
                    if not _answer_in_context(golds, body):
                        stats["dropped"] += 1
                        stats["drops"][f"{src}/{var}:answer-not-in-context"] += 1
                        continue

                if keychain:
                    context: dict[str, Any] = {
                        "task": "keychain",
                        "passages": passages,
                        "start_uuid": parsed["start_uuid"],
                        "num_passages": len(passages),
                    }
                    root_prompt = _KEYCHAIN_ROOT_PROMPT
                    visible_question = (
                        "Follow the UUID key-value chain starting from "
                        f"{parsed['start_uuid']} to find the hidden question, then "
                        "answer it from the passages."
                    )
                else:
                    context = {
                        "task": "plain_multihop",
                        "passages": passages,
                        "question": question,
                        "num_passages": len(passages),
                    }
                    root_prompt = _PLAIN_ROOT_PROMPT_TEMPLATE.format(question=question)
                    visible_question = question

                assert len(root_prompt) < _ROOT_PROMPT_MAX_CHARS, (
                    f"root_prompt too long ({len(root_prompt)} chars) — it must "
                    "never carry the haystack"
                )
                assert passages, "context must be non-empty"

                info = {
                    "context": context,
                    "root_prompt": root_prompt,
                    "loongrl_source": src,
                    "loongrl_variant": var,
                    "loongrl_upstream_index": idx,
                    "loongrl_upstream_data_source": rec["data_source"],
                    "loongrl_upstream_length": int(rec["length"]),
                    "loongrl_split": "eval" if is_eval else "train",
                    "loongrl_num_passages": len(passages),
                    "loongrl_context_chars": len(body),
                    "loongrl_question": question,
                    "loongrl_gold": golds,
                }
                if keychain:
                    info["loongrl_chain_hops"] = parsed["chain_hops"]
                    info["loongrl_num_decoy_questions"] = parsed["num_decoy_questions"]

                rows.append(
                    {
                        "example_id": f"{src}/{var}/{idx}",
                        "prompt": [{"role": "user", "content": visible_question}],
                        "answer": json.dumps(golds, ensure_ascii=False),
                        "info": json.dumps(info, ensure_ascii=False),
                    }
                )
                stats["kept"] += 1
                stats["buckets"][f"{src}/{var}"] += 1

    if not rows:
        raise ValueError(
            f"No LoongRL examples for source={source!r}, variant={variant!r}, "
            f"split={split!r}"
        )

    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]
    stats["returned"] = len(rows)
    stats["drops"] = dict(stats["drops"])
    stats["buckets"] = dict(stats["buckets"])
    return Dataset.from_list(rows), stats


def load_environment(
    *,
    source: str = "all",
    variant: str = "keychain",
    split: str = "train",
    holdout_frac: float = 0.1,
    num_examples: int = -1,
    seed: int = 42,
    metric: str = "sub_em",
    answer_max_chars: int | None = 128,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load LoongRL / KeyChain-15K wired through ``RLMTrainEnv``."""
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {sorted(_METRICS)}, got {metric!r}")

    dataset, stats = _build_dataset(
        source=source,
        variant=variant,
        split=split,
        holdout_frac=holdout_frac,
        num_examples=num_examples,
        seed=seed,
    )

    async def correctness(answer: str, state: vf.State, **_kw: Any) -> float:
        gold = json.loads(answer) if isinstance(answer, str) else answer
        final = state.get("rlm_final_answer")
        if final is None:
            final = state.get("final_answer")
        return score_answer(
            final, gold, metric=metric, answer_max_chars=answer_max_chars
        )

    correctness.__name__ = f"loongrl_{metric}"

    rubric = rlm_train.RLMTrainRubric(
        correctness=correctness, weight=1.0, min_subcall=min_subcall
    )

    for aux in ("sub_em", "em", "f1"):
        if aux == metric:
            continue

        def _mk(name: str):
            async def _aux(answer: str, state: vf.State, **_kw: Any) -> float:
                gold = json.loads(answer) if isinstance(answer, str) else answer
                final = state.get("rlm_final_answer") or state.get("final_answer")
                return score_answer(
                    final, gold, metric=name, answer_max_chars=answer_max_chars
                )

            _aux.__name__ = f"loongrl_{name}"
            return _aux

        rubric.add_reward_func(_mk(aux), weight=0.0)

    env = rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )
    env.loongrl_stats = stats
    return env


__all__ = [
    "load_environment",
    "score_answer",
    "sub_em",
    "exact_match",
    "token_f1",
    "normalize_answer",
    "parse_plain",
    "parse_keychain",
    "ParseError",
    "SOURCES",
    "VARIANTS",
]
