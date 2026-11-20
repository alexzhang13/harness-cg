"""Split a raw GoLongRL row into ``(context, root_prompt)``."""

from __future__ import annotations

from typing import Any

from .metrics import TASK_METRIC_CONFIG, metric_name


_CONTEXT_TAIL_CUES = (
    "Question:",
    "question:",
    "问题：",
    "问题:",
    "【问题】",
    "请记住以上全部对话记录，回答问题。",
    "请阅读以下文字，回答下面的问题。",
    "以下是问题：",
    "###",
)

_CONTEXT_HEAD_CUES = (
    "请阅读以下文字，回答下面的问题。",
    "<text>",
)

_OUTPUT_EXAMPLE_CUES = ("【输出示例】", "Output example:", "输出示例：", "【输出示例】")

_INLINE_SPLIT_CUES = ("问题：", "问题:", "Question:")


def _strip_context_edges(text: str) -> str:
    """Peel instruction scaffolding off both ends of the payload."""
    prev = None
    while prev != text:
        prev = text
        text = text.strip()
        for cue in _CONTEXT_TAIL_CUES:
            if text.endswith(cue):
                text = text[: -len(cue)]
                break
        for cue in _CONTEXT_HEAD_CUES:
            if text.startswith(cue):
                text = text[len(cue) :]
                break
        if text.endswith("</text>"):
            text = text[: -len("</text>")]
    return text.strip()


def _split_output_example(text: str) -> tuple[str, str]:
    """Move a trailing ``Output example:`` / ``【输出示例】`` block out of the payload."""
    best = -1
    for cue in _OUTPUT_EXAMPLE_CUES:
        i = text.rfind(cue)
        if i > best:
            best = i
    if best >= 0 and len(text) - best <= 600:
        return text[:best].rstrip(), text[best:].strip()
    return text, ""


_ANSWER_SHAPE: dict[str, str] = {
    "iou": (
        'a JSON string of the form \'{"columns": ["col", ...], "data": [["val", ...], '
        '...]}\' holding exactly the rows the query selects — build it with '
        "`json.dumps(...)` rather than assigning a Python dict. It is scored by "
        "IoU over the row multiset, so spurious rows cost you."
    ),
    "rouge_l": (
        "the summary text itself as a single block, with no preamble and no "
        "bullet scaffolding. Scored as ROUGE-L against reference summaries."
    ),
}

_CONTEXT_BLURB = {
    "text": "`context` is a string of {n:,} characters holding the full source payload.",
    "chat": (
        "`context` is a list of {n:,} chat messages, each a dict with keys "
        "'role' and 'content', holding the conversation history in order."
    ),
}


def build_root_prompt(
    *,
    question: str,
    trailing_instruction: str,
    context_kind: str,
    context_size: int,
    metric: str,
) -> str:
    blurb = _CONTEXT_BLURB[context_kind].format(n=context_size)
    shape = _ANSWER_SHAPE[metric]
    if (context_kind == "text" and context_size >= 40_000) or (
        context_kind == "chat" and context_size >= 40
    ):
        blurb += (
            " It is far too long to read in full — index it, search it, slice it, "
            "and delegate reading the parts you need to sub-LLM calls."
        )
    else:
        blurb += (
            " It is small enough to inspect directly in the REPL, but still slice "
            "and search rather than dumping it all into one turn."
        )
    parts = [
        blurb,
        "",
        "TASK (verbatim from the benchmark):",
        question.strip(),
    ]
    if trailing_instruction:
        parts.append(trailing_instruction.strip())
    parts += [
        "",
        'Submit your final answer by writing it to `answer["content"]` inside a '
        '`repl` block and setting `answer["ready"] = True`. The value must be '
        + shape,
        "The `[Answer]` marker and any 'output example' formatting mentioned above "
        "are optional here — the grader reads your submitted value directly. Put "
        "your reasoning in the REPL, never in the answer.",
    ]
    return "\n".join(parts)


class ParseError(Exception):
    """Raised (and counted) when a row cannot be split confidently."""


def parse_row(
    row: dict[str, Any],
    *,
    max_root_prompt_chars: int = 8000,
    min_context_chars: int = 200,
) -> dict[str, Any]:
    """Split one raw GoLongRL parquet row."""
    data_source = row["data_source"]
    extra = row.get("extra_info") or {}
    reward_mode = extra.get("reward_mode") or ""
    gt = (row.get("reward_model") or {}).get("ground_truth") or {}
    doc_ids = list(gt.get("doc_ids") or [])
    golden_label = gt.get("golden_label") or ""
    language = gt.get("summary") or ""

    if data_source == "longbench_pro" and reward_mode not in TASK_METRIC_CONFIG:
        raise ParseError("unknown_reward_mode")
    metric = metric_name(data_source, reward_mode)

    messages = list(row.get("prompt") or [])
    if not messages:
        raise ParseError("no_messages")
    question = (extra.get("question") or "").strip()
    if not question:
        raise ParseError("no_question")

    last = messages[-1].get("content") or ""
    trailing = ""

    if len(messages) > 2:
        history = [
            {"role": m.get("role") or "user", "content": m.get("content") or ""}
            for m in messages[:-1]
        ]
        if not history:
            raise ParseError("empty_chat_history")
        context: Any = history
        context_kind = "chat"
        context_chars = sum(len(m["content"]) for m in history)
        style = "chat"
    else:
        first, last_idx = last.find(question), last.rfind(question)
        if first < 0:
            raise ParseError("question_not_in_prompt")

        if last == question:
            cut = max(last.rfind(cue) for cue in _INLINE_SPLIT_CUES)
            if cut <= 0:
                raise ParseError("inline_no_split_marker")
            head, question = last[:cut], last[cut:].strip()
            style = "inline"
        elif last_idx + len(question) == len(last):
            head, style = last[:last_idx], "suffix"
        elif first == 0:
            head, style = last[len(question) :], "prefix"
        else:
            head, trailing, style = (
                last[:last_idx],
                last[last_idx + len(question) :],
                "suffix",
            )

        head, moved = _split_output_example(head) if style == "prefix" else (head, "")
        if moved:
            trailing = (trailing + "\n" + moved).strip()
        context = _strip_context_edges(head)
        context_kind = "text"
        context_chars = len(context)

    if context_chars < min_context_chars:
        raise ParseError("context_too_short")

    root_prompt = build_root_prompt(
        question=question,
        trailing_instruction=trailing,
        context_kind=context_kind,
        context_size=len(context),
        metric=metric,
    )
    if len(root_prompt) > max_root_prompt_chars:
        raise ParseError("root_prompt_too_long")

    return {
        "context": context,
        "context_kind": context_kind,
        "context_chars": context_chars,
        "root_prompt": root_prompt,
        "question": question,
        "style": style,
        "metric": metric,
        "data_source": data_source,
        "reward_mode": reward_mode,
        "language": language,
        "gold": {"doc_ids": doc_ids, "golden_label": golden_label},
    }


__all__ = ["ParseError", "build_root_prompt", "parse_row"]
