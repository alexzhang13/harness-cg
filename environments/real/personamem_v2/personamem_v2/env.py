"""PersonaMem-v2 (`personamem-v2-real`): pick the reply that fits a persona implied by a long chat history (4-way MCQ).

Rows whose revealing snippet is missing from the history are dropped.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

logger = logging.getLogger(__name__)

_REPO = "bowen-upenn/PersonaMem-v2"
# pinned dataset revision
_REVISION = os.environ.get("PERSONAMEM_V2_REVISION", "ed956dea41521fc4499acbc63f966e0fd3c053ba")

_CSV_FILES: dict[str, str] = {
    "train": "benchmark/text/train.csv",
    "val": "benchmark/text/val.csv",
    "benchmark": "benchmark/text/benchmark.csv",
}

CONTEXT_SIZES = ("32k", "128k")

PREF_TYPES = (
    "stereotypical_pref",
    "anti_stereotypical_pref",
    "neutral_preferences",
    "therapy_background",
    "health_and_medical_conditions",
    "sensitive_info",
    "ask_to_forget",
)

ANSWERABLE_PREF_TYPES = tuple(t for t in PREF_TYPES if t != "sensitive_info")

QUESTION_TYPE_GROUPS: dict[str, tuple[str, ...]] = {
    "all": ANSWERABLE_PREF_TYPES,
}

_LETTERS = "ABCD"

_MAX_ROOT_PROMPT_CHARS = 16_000

_SNAPSHOT_THRESHOLD = 150

_RECALL_NUDGE = (
    " Please recall my related preferences from our conversation history to "
    "give personalized responses."
)


_LETTER_PATTERNS = (
    r"\\boxed\{\(?([A-Da-d])\)?\}",
    r"boxed\{\(?([A-Da-d])\)?\}",
    r"final\s+answer\s*(?:is)?\s*[:=]?\s*\[?\(?([A-Da-d])\)?\]?",
    r"\banswer\s*(?:is)?\s*[:=]\s*\[?\(?([A-Da-d])\)?\]?",
    r"the\s+answer\s+is\s*\[?\(?([A-Da-d])\)?\]?",
    r"option\s*\(?([A-Da-d])\)?\b",
    r"^\s*\(?([A-Da-d])\)?\s*[.):\-]",
    r"\b([A-Da-d])\.\s*$",
)


def _normalize(text: str) -> str:
    """Aggressive normalization for the option-text fallback."""
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " ").replace("\u2013", " ")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _coerce_answer_text(final: Any) -> str:
    """``state['rlm_final_answer']`` may be a str / dict / list / None."""
    if final is None:
        return ""
    if isinstance(final, str):
        return final
    if isinstance(final, dict):
        for key in ("letter", "answer", "choice", "content", "final_answer"):
            if key in final and final[key] is not None:
                return _coerce_answer_text(final[key])
        return json.dumps(final, default=str)
    if isinstance(final, (list, tuple)):
        if len(final) == 1:
            return _coerce_answer_text(final[0])
        return " ".join(_coerce_answer_text(x) for x in final)
    return str(final)


def extract_letter(final: Any, options: dict[str, str] | None = None) -> str:
    """Extract the chosen option letter (``"A"``..``"D"``) from a final answer."""
    text = _coerce_answer_text(final).strip()
    if not text:
        return ""

    bare = text.strip().strip("*`'\"[](). \n\t")
    if len(bare) == 1 and bare.upper() in _LETTERS:
        return bare.upper()

    for pat in _LETTER_PATTERNS:
        matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return str(matches[-1]).upper()

    if options:
        norm_ans = _normalize(text)
        if norm_ans:
            exact = [k for k, v in options.items() if _normalize(v) == norm_ans]
            if len(exact) == 1:
                return exact[0]
            contained = [
                k for k, v in options.items()
                if _normalize(v) and _normalize(v) in norm_ans
            ]
            if len(contained) == 1:
                return contained[0]
    return ""


def score_mcq(final: Any, gold: dict[str, Any]) -> float:
    """1.0 iff the extracted letter equals the gold letter, else 0.0."""
    options = gold.get("options") or {}
    letter = extract_letter(final, options=options)
    if not letter:
        return 0.0
    return 1.0 if letter == str(gold.get("letter", "")).upper() else 0.0


async def _score(answer: str, state: vf.State, **_kw: Any) -> float:
    gold = json.loads(answer) if isinstance(answer, str) else answer
    final = state.get("rlm_final_answer")
    if final is None:
        final = state.get("final_answer")
    return score_mcq(final, gold)


def _parse_user_query(raw: Any) -> str | None:
    """``content`` of ``user_query`` (a Python dict literal, sometimes JSON), or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    obj: Any = None
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            obj = None
    if isinstance(obj, dict):
        content = obj.get("content")
        return str(content).strip() if content else None
    if isinstance(obj, str) and obj.strip():
        return obj.strip()
    return None


def _parse_incorrect(raw: Any) -> list[str] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    obj: Any = None
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return None
    if not isinstance(obj, (list, tuple)):
        return None
    out = [str(x).strip() for x in obj if str(x).strip()]
    return out or None


def _parse_snippet(raw: Any) -> list[str]:
    """Message contents that reveal the preference (must survive truncation)."""
    if raw is None:
        return []
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []
    obj: Any = None
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return []
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, (list, tuple)):
        return []
    out = []
    for m in obj:
        if isinstance(m, dict) and m.get("content"):
            out.append(str(m["content"]).strip())
        elif isinstance(m, str) and m.strip():
            out.append(m.strip())
    return out


def _option_order_seed(persona_id: Any, query: str, salt: int) -> int:
    """Deterministic per-row option order."""
    key = f"{persona_id}|{query}|{salt}".encode()
    return int(hashlib.md5(key).hexdigest()[:16], 16)


def _build_options(
    correct: str, incorrect: list[str], *, persona_id: Any, query: str, salt: int,
) -> tuple[dict[str, str], str]:
    opts = [correct] + list(incorrect)
    random.Random(_option_order_seed(persona_id, query, salt)).shuffle(opts)
    mapping = {_LETTERS[i]: opts[i] for i in range(len(opts))}
    letter = next(k for k, v in mapping.items() if v == correct)
    return mapping, letter


def _fetch_histories(links: list[str], *, snapshot_threshold: int) -> dict[str, str]:
    """Resolve repo-relative chat-history paths to local files."""
    from huggingface_hub import hf_hub_download, snapshot_download

    uniq = sorted(set(links))
    if len(uniq) >= snapshot_threshold:
        dirs = sorted({os.path.dirname(p) for p in uniq})
        patterns = [f"{d}/*" for d in dirs]
        logger.info(
            "personamem_v2: %d distinct histories -> snapshot_download(%s)",
            len(uniq), patterns,
        )
        root = snapshot_download(_REPO, repo_type="dataset", revision=_REVISION, allow_patterns=patterns)
        return {p: os.path.join(root, p) for p in uniq}
    return {
        p: hf_hub_download(_REPO, p, repo_type="dataset", revision=_REVISION)
        for p in uniq
    }


_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\\\n]+")


def _message_text(content: Any) -> str:
    """Coerce one message's ``content`` to plain text."""
    if content is None:
        return ""
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" or "text" in item:
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append("[image]")
        content = "\n".join(p for p in parts if p)
    elif not isinstance(content, str):
        content = str(content)
    if "base64," in content:
        content = _DATA_URI_RE.sub("[image]", content)
    return content.strip()


def _load_history(path: str) -> tuple[str | None, list[dict[str, str]], dict]:
    """Split a chat-history file into (persona system blob, messages, metadata)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        raw, meta = data, {}
    else:
        raw = data.get("chat_history") or data.get("conversations") or []
        meta = data.get("metadata") or {}
    persona: str | None = None
    msgs: list[dict[str, str]] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", ""))
        content = _message_text(m.get("content"))
        if not content:
            continue
        if role == "system" and persona is None and not msgs:
            persona = content
            continue
        msgs.append({"role": role, "content": content})
    return persona, msgs, meta


_ROOT_TEMPLATE = (
    "The user just sent you this message:\n"
    "---\n{query}\n---\n\n"
    "Exactly one of the four candidate replies below is the *most personalized* "
    "reply for THIS user. The others are fluent but reference preferences the "
    "user does not hold, contradict a preference the user later revised, or "
    "ignore the user's implicit signals.\n\n"
    "{options}\n\n"
    "`context` is a dict:\n"
    "  - `context['messages']`: the user's full prior chat history with you, a "
    "list of {{'turn': int, 'role': 'user'|'assistant', 'content': str}} dicts, "
    "oldest first ({n_messages} messages). The user's relevant preference is "
    "revealed only IMPLICITLY somewhere in there (e.g. mentioned in passing "
    "inside an email they asked you to reword), and it may have been revised or "
    "revoked later in the history — the LATEST state wins. Some stretches of the "
    "history are unrelated math/coding chatter and can be ignored.\n"
    "  - `context['persona']`: {persona_line}\n\n"
    "Search the history for the evidence, then submit ONLY the letter of the "
    "best reply as your final answer, e.g. `answer[\"content\"] = \"B\"`."
)

_PERSONA_LINE_ON = (
    "the explicit persona profile you were given for this user (demographics, "
    "hobbies, speaking style). It does NOT contain the preference being tested"
)


def _build_root_prompt(query: str, options: dict[str, str], n_messages: int) -> str:
    q = query + _RECALL_NUDGE
    opts = "\n".join(f"{k}. {options[k]}" for k in sorted(options))
    return _ROOT_TEMPLATE.format(
        query=q,
        options=opts,
        n_messages=n_messages,
        persona_line=_PERSONA_LINE_ON,
    )


def _resolve_question_types(question_type: str | list[str] | None) -> tuple[str, ...]:
    if question_type is None:
        return PREF_TYPES
    if isinstance(question_type, str):
        wanted: list[str] = [question_type]
    else:
        wanted = list(question_type)
    out: list[str] = []
    for w in wanted:
        if w in QUESTION_TYPE_GROUPS:
            out.extend(QUESTION_TYPE_GROUPS[w])
        elif w in PREF_TYPES:
            out.append(w)
        else:
            raise ValueError(
                f"Unknown question_type {w!r}. Choose from "
                f"{sorted(set(PREF_TYPES) | set(QUESTION_TYPE_GROUPS))}"
            )
    return tuple(dict.fromkeys(out))


_CSV_CACHE: dict[str, Any] = {}


def _read_csv(*, split: str):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    if split not in _CSV_FILES:
        raise ValueError(f"Unknown split {split!r}. Choose from {tuple(_CSV_FILES)}")
    key = split
    hit = _CSV_CACHE.get(key)
    if hit is not None:
        return hit
    path = hf_hub_download(_REPO, _CSV_FILES[split], repo_type="dataset", revision=_REVISION)
    df = pd.read_csv(path)
    _CSV_CACHE.clear()
    _CSV_CACHE[key] = df
    return df


def build_rows(
    *,
    split: str = "train",
    context_size: str = "32k",
    question_type: str | list[str] | None = "all",
    num_examples: int = -1,
    seed: int | None = 42,
) -> tuple[list[dict], dict[str, int]]:
    """Build the verifiers rows plus a drop-reason histogram."""
    if context_size not in CONTEXT_SIZES:
        raise ValueError(f"context_size must be one of {CONTEXT_SIZES}; got {context_size!r}")

    df = _read_csv(split=split)
    wanted_types = _resolve_question_types(question_type)

    tok_col = f"total_tokens_in_chat_history_{context_size}"
    dist_col = f"distance_from_related_snippet_to_query_{context_size}"
    link_col = f"chat_history_{context_size}_link"

    mask = df["pref_type"].isin(wanted_types)
    records = df[mask].to_dict("records")

    order = list(range(len(records)))
    if seed is not None:
        random.Random(seed).shuffle(order)

    drops: dict[str, int] = {}

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    staged: list[dict] = []
    for idx in order:
        r = records[idx]
        query = _parse_user_query(r["user_query"])
        if not query:
            _drop("bad_user_query")
            continue
        correct = r["correct_answer"]
        if not isinstance(correct, str) or not correct.strip():
            _drop("empty_correct_answer")
            continue
        correct = correct.strip()
        incorrect = _parse_incorrect(r["incorrect_answers"])
        if incorrect is None:
            _drop("bad_incorrect_answers")
            continue
        if len(incorrect) != 3:
            _drop(f"n_distractors_{len(incorrect)}")
            continue
        if any(_normalize(x) == _normalize(correct) for x in incorrect):
            _drop("duplicate_option")
            continue
        link = str(r[link_col])
        if not link or link.lower() == "nan":
            _drop("missing_history_link")
            continue
        staged.append({
            "row": r,
            "query": query,
            "correct": correct,
            "incorrect": incorrect,
            "link": link,
        })

    if not staged:
        raise ValueError(
            f"No PersonaMem-v2 rows survived filtering for split={split!r}, "
            f"question_type={question_type!r}, context_size={context_size!r}"
        )

    target = num_examples if num_examples and num_examples > 0 else len(staged)
    batch_size = max(16, min(len(staged), target + 16))

    cache: dict[str, tuple[str | None, list[dict[str, str]], dict]] = {}
    local: dict[str, str] = {}
    rows: list[dict] = []
    for pos, s in enumerate(staged):
        if len(rows) >= target:
            break
        link = s["link"]
        if link not in local:
            batch = [x["link"] for x in staged[pos: pos + batch_size]]
            local.update(_fetch_histories(batch, snapshot_threshold=_SNAPSHOT_THRESHOLD))
        if link not in local:
            _drop("history_not_fetched")
            continue
        if link not in cache:
            try:
                cache[link] = _load_history(local[link])
            except Exception as exc:  # noqa: BLE001
                logger.warning("personamem_v2: unreadable history %s: %s", link, exc)
                cache[link] = (None, [], {})
        persona_blob, msgs, meta = cache[link]
        if not msgs:
            _drop("empty_history")
            continue
        s["snippet"] = _parse_snippet(s["row"]["related_conversation_snippet"])
        present = {m["content"].strip() for m in msgs}
        if not s["snippet"]:
            _drop("no_snippet_column")
            continue
        if not all(t in present for t in s["snippet"]):
            _drop("snippet_absent_from_history")
            continue

        indexed = [
            {"turn": i, "role": m["role"], "content": m["content"]}
            for i, m in enumerate(msgs)
        ]
        context = {
            "persona": persona_blob,
            "messages": indexed,
            "n_messages": len(indexed),
        }
        options, letter = _build_options(
            s["correct"], s["incorrect"],
            persona_id=s["row"]["persona_id"], query=s["query"],
            salt=0,
        )
        root_prompt = _build_root_prompt(s["query"], options, len(indexed))
        if len(root_prompt) > _MAX_ROOT_PROMPT_CHARS:
            _drop("root_prompt_too_long")
            continue

        r = s["row"]
        info = {
            "context": context,
            "root_prompt": root_prompt,
            "question": s["query"],
            "options": options,
            "gold_letter": letter,
            "persona_id": int(r["persona_id"]),
            "context_size": context_size,
            "split": split,
            "pref_type": str(r["pref_type"]),
            "conversation_scenario": str(r["conversation_scenario"]),
            "topic_query": str(r["topic_query"]),
            "topic_preference": str(r["topic_preference"]),
            "who": str(r["who"]),
            "updated": bool(r["updated"]),
            "preference": str(r["preference"]),
            "prev_pref": None if str(r["prev_pref"]) == "nan" else str(r["prev_pref"]),
            "chat_history_link": link,
            "history_tokens": int(meta.get("final_token_count") or r[tok_col] or 0),
            "history_chars": sum(len(m["content"]) for m in indexed),
            "snippet_distance_tokens": int(r[dist_col]) if r[dist_col] == r[dist_col] else -1,
            "metric": "mcq_accuracy",
        }
        rows.append({
            "example_id": len(rows),
            "prompt": [{"role": "user", "content": s["query"]}],
            "answer": json.dumps({"letter": letter, "options": options}),
            "info": json.dumps(info),
        })

    if not rows:
        raise ValueError(
            "PersonaMem-v2: every candidate row survived filtering but all "
            f"dropped during assembly; drops={drops}."
        )
    if drops:
        logger.info("personamem_v2[%s/%s]: drops=%s", split, context_size, drops)
    return rows, drops


def _build_dataset(**kwargs: Any) -> Dataset:
    rows, _drops = build_rows(**kwargs)
    return Dataset.from_list(rows)


def load_environment(
    *,
    split: str = "train",
    context_size: str = "32k",
    question_type: str | list[str] | None = "all",
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 1,
    **kwargs: Any,
) -> vf.Environment:
    """Load PersonaMem-v2 wired through RLMTrainEnv."""
    dataset = _build_dataset(
        split=split,
        context_size=context_size,
        question_type=question_type,
        num_examples=num_examples,
        seed=seed,
    )
    rubric = rlm_train.RLMTrainRubric(
        correctness=_score, weight=1.0, min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = [
    "load_environment",
    "build_rows",
    "extract_letter",
    "score_mcq",
    "PREF_TYPES",
    "ANSWERABLE_PREF_TYPES",
    "QUESTION_TYPE_GROUPS",
    "CONTEXT_SIZES",
]
