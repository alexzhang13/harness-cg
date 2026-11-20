"""PrefEval classification (`prefeval-real`): answer an MCQ consistently with a preference stated earlier in a long chat.

Differences from upstream: a fixed assistant acknowledgment after an explicit preference, a random preference
position and resampled filler turns per row.
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

from . import _upstream as up
from ._data import (
    MAX_INTER_TURNS,
    PREF_FORMS,
    load_distractor_messages,
    load_mcq_topic,
    load_pref_topic,
)

TASKS: tuple[str, ...] = ("classification",)
PREF_POSITIONS: tuple[str, ...] = ("random",)
DISTRACTOR_SAMPLING: tuple[str, ...] = ("random",)

ALL_TOPICS = up.ALL_TOPICS

DEFAULT_ASSISTANT_ACK = (
    "Thanks for letting me know — I've made a note of that and I'll keep it in "
    "mind whenever I make suggestions for you."
)

_CONTEXT_BLURB = (
    "`context` is our conversation history: a list of {'role', 'content'} dicts, "
    "oldest turn first, alternating user/assistant. Somewhere in it I have made a "
    "personal preference known — sometimes stated outright, sometimes only implied "
    "by what I picked or said in passing. Most of the history is unrelated chatter. "
    "Search the history in the REPL, work out the relevant preference, then answer "
    "in a way that respects it."
)

_MCQ_PROTOCOL = (
    'Submit ONLY the letter of your choice as a plain string, e.g. '
    'answer["content"] = "B"; answer["ready"] = True'
)


def preference_turns(pref_form: str, record: dict, *, assistant_ack: str) -> list[dict]:
    """The preference-bearing message block for one record."""
    if pref_form == "explicit":
        pref = (record.get("preference") or "").strip()
        if not pref:
            raise _ParseError("explicit record has no `preference`")
        return [
            {"role": "user", "content": pref},
            {"role": "assistant", "content": assistant_ack},
        ]

    conv = record.get("conversation")
    if not isinstance(conv, dict) or not conv:
        raise _ParseError(f"{pref_form} record has no usable `conversation` dict")

    if pref_form == "implicit_persona":
        try:
            keys = sorted(conv, key=int)
        except (TypeError, ValueError) as e:
            raise _ParseError(
                f"persona conversation keys are not integer-like: {list(conv)[:5]}"
            ) from e
        out: list[dict] = []
        for k in keys:
            turn = conv[k]
            if not isinstance(turn, dict):
                raise _ParseError(f"persona turn {k!r} is {type(turn).__name__}, expected dict")
            u = str(turn.get("user") or "").strip()
            a = str(turn.get("assistant") or "").strip()
            if not u or not a:
                raise _ParseError(f"persona turn {k!r} missing user/assistant text")
            out.append({"role": "user", "content": u})
            out.append({"role": "assistant", "content": a})
        if not out:
            raise _ParseError("persona conversation produced no turns")
        return out

    raise ValueError(f"pref_form must be one of {PREF_FORMS}, got {pref_form!r}")


class _ParseError(ValueError):
    """A record we refuse to render (counted as a drop, never guessed at)."""


def distractor_turns(inter_turns: int, *, rng: random.Random) -> list[dict]:
    """``inter_turns`` filler turn-pairs sampled (without replacement) from the upstream LMSYS pool."""
    pool = load_distractor_messages()
    n_pool_pairs = len(pool) // 2
    if inter_turns > n_pool_pairs:
        raise ValueError(
            f"inter_turns={inter_turns} needs {inter_turns * 2} filler messages but "
            f"the upstream pool has only {len(pool)} ({MAX_INTER_TURNS} turns)."
        )
    pairs = [(pool[2 * i], pool[2 * i + 1]) for i in range(n_pool_pairs)]
    chosen = rng.sample(pairs, inter_turns)
    out: list[dict] = []
    for u, a in chosen:
        out.append(dict(u))
        out.append(dict(a))
    return out


def build_context(
    pref_block: list[dict],
    filler: list[dict],
    *,
    pref_position: str,
    rng: random.Random,
) -> tuple[list[dict], int]:
    """Splice the preference block into the filler stream."""
    n_pairs = len(filler) // 2
    if pref_position == "random":
        offset = rng.randint(0, n_pairs)
    else:
        raise ValueError(f"pref_position must be one of {PREF_POSITIONS}, got {pref_position!r}")
    cut = offset * 2
    return filler[:cut] + pref_block + filler[cut:], cut


def build_root_prompt(*, question: str, options: list[str]) -> str:
    """Compose the short prompt the root model sees (upstream zero-shot MCQ format)."""
    if not options:
        raise ValueError("classification task needs options")
    q = question + up.get_mcq_question_format(options)
    protocol = _MCQ_PROTOCOL

    return f"{_CONTEXT_BLURB}\n\nQuestion: {q}\n\n{protocol}"


_LETTER_ONLY_RE = re.compile(r"^\(?([ABCD])\)?[\.\):\s]*$", re.IGNORECASE)
_LABELLED_RE = re.compile(
    r"(?:answer|choice|option|recommendation|pick)\s*(?:is|:|=|->)?\s*"
    r"[*_\"'\[(]{0,3}([ABCD])\b",
    re.IGNORECASE,
)
_LINE_START_RE = re.compile(r"^\s*[*_\"'(]{0,2}([ABCD])[\)\.\:]", re.MULTILINE)
_BARE_RE = re.compile(r"\b([ABCD])\b")
_ANSWER_KEYS = ("content", "answer", "choice", "letter", "option", "final_answer")


def parse_choice(final: Any) -> str | None:
    """Coerce ``state["rlm_final_answer"]`` into one of ``A``/``B``/``C``/``D``."""
    if final is None:
        return None
    if isinstance(final, dict):
        for key in _ANSWER_KEYS:
            if key in final and final[key] is not None:
                got = parse_choice(final[key])
                if got:
                    return got
        return None
    if isinstance(final, (list, tuple)):
        for item in final:
            got = parse_choice(item)
            if got:
                return got
        return None

    text = final if isinstance(final, str) else str(final)
    stripped = text.strip()
    if not stripped:
        return None

    got = up.extract_choice(stripped)
    if got:
        return got

    m = _LETTER_ONLY_RE.match(stripped)
    if m:
        return m.group(1).upper()

    if stripped[:1] in "{[":
        for loader in (json.loads, ast.literal_eval):
            try:
                obj = loader(stripped)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(obj, (dict, list, tuple)):
                got = parse_choice(obj)
                if got:
                    return got
            break

    matches = list(_LABELLED_RE.finditer(stripped))
    if matches:
        return matches[-1].group(1).upper()

    matches = list(_LINE_START_RE.finditer(stripped))
    if matches:
        return matches[-1].group(1).upper()

    if len(stripped) <= 40:
        matches = list(_BARE_RE.finditer(stripped))
        if matches:
            return matches[-1].group(1).upper()
    return None


def _meta(info: Any) -> dict:
    return json.loads(info) if isinstance(info, str) else (info or {})


def make_classification_reward():
    async def prefeval_mcq_accuracy(answer: Any, info: Any, state: vf.State, **_kw: Any) -> float:
        """Upstream classification accuracy: extracted letter == the correct letter."""
        meta = _meta(info)
        gold = (answer if isinstance(answer, str) else str(answer or "")).strip().upper()
        final = state.get("rlm_final_answer")
        if final is None:
            final = state.get("final_answer")
        letter = parse_choice(final)
        state["prefeval_parsed_letter"] = letter or ""
        state["prefeval_parse_ok"] = bool(letter)
        state["prefeval_gold_letter"] = gold
        state["prefeval_topic"] = meta.get("prefeval_topic", "")
        return 1.0 if (letter is not None and letter == gold) else 0.0

    async def prefeval_parse_rate(state: vf.State, **_kw: Any) -> float:
        return 1.0 if state.get("prefeval_parse_ok") else 0.0

    return prefeval_mcq_accuracy, [prefeval_parse_rate]


def build_dataset(
    *,
    task: str = "classification",
    pref_form: str = "explicit",
    inter_turns: int = 10,
    pref_position: str = "random",
    distractor_sampling: str = "random",
    num_examples: int = -1,
    seed: int = 42,
    stats: dict[str, Any] | None = None,
) -> Dataset:
    """Assemble one (pref_form, inter_turns) slice over all 20 topics."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    if pref_form not in PREF_FORMS:
        raise ValueError(f"pref_form must be one of {PREF_FORMS}, got {pref_form!r}")
    if pref_position not in PREF_POSITIONS:
        raise ValueError(f"pref_position must be one of {PREF_POSITIONS}, got {pref_position!r}")
    if distractor_sampling not in DISTRACTOR_SAMPLING:
        raise ValueError(
            f"distractor_sampling must be one of {DISTRACTOR_SAMPLING}, "
            f"got {distractor_sampling!r}"
        )
    if inter_turns < 0:
        raise ValueError(f"inter_turns must be >= 0, got {inter_turns}")

    topic_list = list(ALL_TOPICS)

    candidates: list[tuple[str, int]] = []
    for topic in topic_list:
        records = load_pref_topic(pref_form, topic)
        mcq = load_mcq_topic(topic)
        if len(mcq) != len(records):
            raise RuntimeError(
                f"{topic}: mcq_options has {len(mcq)} rows but {pref_form} has "
                f"{len(records)} — upstream index alignment broke"
            )
        candidates.extend((topic, i) for i in range(len(records)))

    if num_examples and num_examples > 0 and num_examples < len(candidates):
        random.Random(seed).shuffle(candidates)
        candidates = candidates[:num_examples]

    drops: dict[str, int] = {}
    rows: list[dict] = []
    for topic, idx in candidates:
        records = load_pref_topic(pref_form, topic)
        mcq = load_mcq_topic(topic)
        rng = random.Random(f"{seed}|{pref_form}|{task}|{topic}|{idx}")
        filler = distractor_turns(inter_turns, rng=rng)
        try:
            row = _build_row(
                task=task, pref_form=pref_form, topic=topic, idx=idx,
                record=records[idx], mcq_record=mcq[idx],
                filler=filler, inter_turns=inter_turns,
                pref_position=pref_position,
                distractor_sampling=distractor_sampling, rng=rng,
            )
        except _ParseError as e:
            key = f"{pref_form}:{type(e).__name__}:{e}"
            drops[key] = drops.get(key, 0) + 1
            continue
        rows.append(row)

    if not rows:
        raise ValueError(
            f"no prefeval rows for task={task!r} pref_form={pref_form!r} "
            f"topics={topic_list} (drops={drops})"
        )
    for i, row in enumerate(rows):
        row["example_id"] = i

    if stats is not None:
        stats["rows"] = len(rows)
        stats["drops"] = drops
        stats["drop_count"] = sum(drops.values())
        stats["topics"] = topic_list
        stats["inter_turns"] = inter_turns
        stats["context_chars"] = [
            sum(len(m["content"]) for m in json.loads(r["info"])["context"]) for r in rows
        ]
        stats["root_prompt_chars"] = [len(json.loads(r["info"])["root_prompt"]) for r in rows]
    return Dataset.from_list(rows)


ROOT_PROMPT_CHAR_CEILING = 8_000


def _build_row(
    *,
    task: str,
    pref_form: str,
    topic: str,
    idx: int,
    record: dict,
    mcq_record: dict | None,
    filler: list[dict],
    inter_turns: int,
    pref_position: str,
    distractor_sampling: str,
    rng: random.Random,
) -> dict:
    """Render one benchmark record into a verifiers row."""
    question = str(record.get("question") or "").strip()
    preference = str(record.get("preference") or "").strip()
    if not question:
        raise _ParseError("record has no `question`")
    if not preference:
        raise _ParseError("record has no `preference`")

    pref_block = preference_turns(pref_form, record, assistant_ack=DEFAULT_ASSISTANT_ACK)
    context, pref_start = build_context(
        pref_block, filler, pref_position=pref_position, rng=rng
    )

    raw_options = list((mcq_record or {}).get("classification_task_options") or [])
    if len(raw_options) != 4 or any(not str(o).strip() for o in raw_options):
        raise _ParseError(
            f"classification_task_options is {len(raw_options)} non-empty entries, expected 4"
        )
    aligned_option = str(raw_options[0]).strip()
    options, correct_index = up.shuffle_options([str(o).strip() for o in raw_options], rng)
    correct_letter = "ABCD"[correct_index]
    answer = correct_letter

    root_prompt = build_root_prompt(question=question, options=options)

    if not context:
        raise _ParseError("empty context")
    if len(root_prompt) > ROOT_PROMPT_CHAR_CEILING:
        raise _ParseError(f"root_prompt is {len(root_prompt)} chars (> {ROOT_PROMPT_CHAR_CEILING})")
    if question not in root_prompt:
        raise _ParseError("question did not survive into root_prompt")
    if preference.lower() in root_prompt.lower():
        raise _ParseError("preference leaked into root_prompt")
    expected_msgs = len(pref_block) + inter_turns * 2
    if len(context) != expected_msgs:
        raise _ParseError(f"context has {len(context)} messages, expected {expected_msgs}")
    for i, m in enumerate(context):
        expected_role = "user" if i % 2 == 0 else "assistant"
        if m["role"] != expected_role:
            raise _ParseError(
                f"context message {i} is {m['role']!r}, expected {expected_role!r} "
                "(user/assistant alternation broken)"
            )
    if pref_form == "explicit" and preference != context[pref_start]["content"]:
        raise _ParseError("explicit preference not recoverable at its recorded index")

    info: dict[str, Any] = {
        "context": context,
        "root_prompt": root_prompt,
        "question": question,
        "prefeval_task": task,
        "prefeval_pref_form": pref_form,
        "prefeval_topic": topic,
        "prefeval_domain": up.TOPIC_DOMAIN[topic],
        "prefeval_row_index": idx,
        "prefeval_inter_turns": inter_turns,
        "prefeval_turn_distance": inter_turns + 2,
        "prefeval_prompt_method": "zero-shot",
        "prefeval_pref_position": pref_position,
        "prefeval_distractor_sampling": distractor_sampling,
        "prefeval_pref_start_index": pref_start,
        "prefeval_pref_block_len": len(pref_block),
        "prefeval_context_messages": len(context),
        "prefeval_preference": preference,
        "prefeval_explanation": str(record.get("explanation") or ""),
        "prefeval_persona": str(record.get("persona") or ""),
        "prefeval_options": options,
        "prefeval_correct_index": correct_index,
        "prefeval_correct_letter": correct_letter,
        "prefeval_aligned_option": aligned_option,
    }
    return {
        "example_id": 0,
        "prompt": [{"role": "user", "content": question}],
        "answer": answer,
        "info": json.dumps(info),
    }


def load_environment(
    *,
    task: str = "classification",
    pref_form: str = "explicit",
    inter_turns: int = 10,
    pref_position: str = "random",
    distractor_sampling: str = "random",
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load PrefEval (classification) wired through ``RLMTrainEnv``."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")

    correctness, metrics = make_classification_reward()

    stats: dict[str, Any] = {}
    dataset = build_dataset(
        task=task, pref_form=pref_form, inter_turns=inter_turns,
        pref_position=pref_position, distractor_sampling=distractor_sampling,
        num_examples=num_examples, seed=seed, stats=stats,
    )

    rubric = rlm_train.RLMTrainRubric(
        correctness=correctness, weight=1.0, min_subcall=min_subcall,
    )
    for m in metrics:
        rubric.add_metric(m)
    env = rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )
    env.prefeval_build_stats = stats  # type: ignore[attr-defined]
    return env


__all__ = [
    "load_environment",
    "build_dataset",
    "build_root_prompt",
    "build_context",
    "preference_turns",
    "distractor_turns",
    "parse_choice",
    "make_classification_reward",
    "ALL_TOPICS",
    "TASKS",
    "PREF_FORMS",
    "PREF_POSITIONS",
    "DISTRACTOR_SAMPLING",
    "MAX_INTER_TURNS",
    "DEFAULT_ASSISTANT_ACK",
    "ROOT_PROMPT_CHAR_CEILING",
]
