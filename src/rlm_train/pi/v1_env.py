"""make_pi_env: turn a benchmark's dataset builder and grader into a verifiers-v1 Pi env, vf.Env(taskset, HostPi).

The context goes to context.txt in pi's sandbox and the reward grades pi's stdout.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import verifiers as vf

from .host_pi import HostPi

PI_CONTEXT_FILENAME = "context.txt"

# Prepended to the user query; pi's default system prompt is left unchanged.
PI_QUERY_PREAMBLE = (
    "IMPORTANT — READ THIS FIRST. The data for this task is in a file named "
    "`context.txt` in your working directory, and it is almost certainly LARGER "
    "THAN YOUR ENTIRE CONTEXT WINDOW — it will NOT fit. If you read it in full "
    "(`cat context.txt`, opening it in an editor, or any unbounded read), the request "
    "will FAIL with a context-overflow error and you will score ZERO. Do not do this. "
    "Never load the whole file — or a large chunk of it — into your context.\n\n"
    "Work carefully within your window instead:\n"
    "  1. First check the file's size: `wc -l context.txt` and `wc -c context.txt`.\n"
    "  2. Find what you need with `grep -n` / `rg` (search by keyword), and read only "
    "small ranges with `sed -n 'START,ENDp' context.txt`, `head`, `tail`, or `awk` — "
    "never more than a couple thousand lines at a time.\n"
    "  3. For counting or aggregation, compute the answer with a shell command "
    "(`grep -c`, `awk`, `sort | uniq -c`) rather than reading raw data into your reply.\n\n"
    "You are an autonomous coding agent running NON-INTERACTIVELY: there is no human "
    "to talk to. Do not greet, do not ask for clarification, do not wait for input — "
    "begin working immediately and finish the task entirely on your own.\n\n"
    "When finished, put your final answer on its own line at the very end, in the "
    "exact OUTPUT FORMAT stated below."
)

# Appended to the preamble when the sub-agent extension is enabled.
PI_SUBAGENT_NOTE = (
    "You also have a `subagent` tool. It hands a sub-task to a helper agent that has "
    "its OWN separate context window, and returns only that agent's short reply to you "
    "— so the raw text the helper reads never enters your window. Two helpers are "
    "available: `reader` (read/search a slice of the file and report back what it "
    "found) and `worker` (carry out a self-contained sub-task, e.g. score or aggregate "
    "a batch). Use it when it helps; you can also keep using the shell tools directly."
)


# Task paragraphs added with PI_PROMPT_GUIDE=1 and PI_PROMPT_TASK=<key> (pi_runs/settings.py TASK_PI_ENV): the data
# format and a strategy that fits the limits.
PI_TASK_GUIDES = {
    "adaleval": (
        "About this task: `context.txt` holds hundreds of candidate answers, each starting with a label line `A<number>:`. "
        "Exactly one candidate is the accepted answer to the question; the others answer unrelated questions, so searching "
        "the file for words of the question only finds the question itself and you must compare the candidates with the "
        "question. Do not read the candidates yourself: one of your tool results shows only about 6 KB, a few "
        "candidates. Helpers can read much more per call (about 90 KB, roughly 2,000 lines, in one `sed -n 'A,Bp' "
        "context.txt`). Count the lines (`wc -l`), cut the file into slices of about 2,000 lines (a 14,000-line file gives "
        "7 or 8 slices, a short file gives one), and send every slice to a `reader` helper in parallel mode: one "
        "`subagent` call with a `tasks` list of up to 8 entries, each entry `{agent: \"reader\", task: ...}` where the "
        "task names the line range, contains the full text of the question, tells the helper to read its whole range in "
        "one `sed` call, and asks for the label of the candidate in the range that best answers the question plus a "
        "one-line reason. Then answer with the best label returned."
    ),
    "oolong-pairs": (
        "About this task: `context.txt` is a list of records, one per line, each with a user ID; you must classify every "
        "record by the meaning of its question and then list all pairs of user IDs that satisfy the condition. You have a "
        "large output budget for this task (about 12,000 tokens per message in training and 131,000 in evaluation), so "
        "write the complete pair list in your final answer, without abbreviating or repeating it. Never print the records "
        "or a classification of each record in a message: keep intermediate results in files (write them with scripts "
        "and look at them with `wc` or `head`). Classify records over the whole file: either with a script (keywords or "
        "patterns with `grep` or `awk`, checked on a sample) or with helpers, which can read much more per call than "
        "you (up to about 90 KB, so a helper can read a large part of `context.txt`, even all of it, in one `sed` call): "
        "give each helper a line range together with the label definitions and ask for one `id label` line per record, "
        "several ranges per `subagent` call. Then compute the pairs with a script, never by hand."
    ),
}


def pi_harness_preamble(*, subagents: bool) -> str:
    """Short harness explanation (PI_PROMPT_VARIANT=short), quoting the limits from the PI_* variables."""
    turns = int(os.environ.get("PI_MAX_TURNS") or 30)
    calls = int(os.environ.get("PI_SUBAGENT_MAX_CALLS") or 50)
    child_turns = int(os.environ.get("PI_SUBAGENT_MAX_TURNS") or 10)
    text = (
        "You are an autonomous agent running non-interactively: no human will reply, so never ask "
        "questions. The task data is in `context.txt` in your working directory. It is far larger than "
        "your context window, so never read it whole: check its size, search it with `grep`, read small "
        "ranges with `sed -n`, `head` or `awk`, and compute counts and aggregates with shell commands.\n\n"
        f"Hard limit: you have at most {turns} turns (model calls) in total. When they are used up the "
        "run ends with no answer and you score zero, so give your final answer before then."
    )
    guide = str(os.environ.get("PI_PROMPT_GUIDE", "")).lower() in {"1", "true", "yes"}
    if guide:
        # `read` returns a few KB per call, too little to page through a large file within the turn budget
        text += (
            "\n\nHow to read the file: the `read` tool returns only about 6 KB per call whatever `limit` you pass, "
            "so paging through `context.txt` with it cannot finish within your turn budget. Use the shell instead: "
            "run `wc -lc context.txt` first, then `grep -n` for keywords and `sed -n 'A,Bp'` for short ranges of under "
            "100 lines. When your turns are nearly used up, each tool result carries a turn-budget notice; as soon as "
            "it tells you to answer, write your final answer without further tool calls."
        )
    task_guide = PI_TASK_GUIDES.get(str(os.environ.get("PI_PROMPT_TASK", "")).lower()) if guide else None
    if task_guide:
        text += "\n\n" + task_guide
    if subagents:
        text += (
            f"\n\nYou also have a `subagent` tool. Each call runs a helper agent on a sub-task in its own "
            f"separate context window and returns only its reply. A helper gets at most {child_turns} "
            f"turns, and you have only {calls} sub-agent calls in total for the whole task. Use them "
            "wisely: delegate sub-tasks that are worth a helper (for example reading or scoring a slice "
            "of the file), put several items into one call instead of making many small calls, and use "
            "the shell directly whenever it is enough."
        )
        if guide:
            text += " The available helper agents are `reader` and `worker`; use exactly these names."
    return text + "\n\nPut your final answer at the very end, in the exact OUTPUT FORMAT given below."


def _subagents_enabled() -> bool:
    """True if the sub-agent tool is enabled."""
    return str(os.environ.get("PI_SUBAGENT_DISABLE", "")).lower() not in {"1", "true", "yes"}


PI_ANSWER_HEADER = (
    "OUTPUT FORMAT (this is exactly how your answer is scored — follow it "
    "precisely and put it at the very end of your response):\n"
)


def answer_format_footer(answer_format: str) -> str:
    """The output-format block appended to the instruction."""
    return PI_ANSWER_HEADER + answer_format.strip()


def load_backing_env(module_name: str, repo_relpath: str) -> Any:
    """Import <module_name>.env, falling back to repo_relpath in this checkout."""
    try:
        return importlib.import_module(f"{module_name}.env")
    except Exception:
        pass
    repo_root = Path(__file__).resolve().parents[3]
    env_file = repo_root / repo_relpath
    if not env_file.exists():
        raise ModuleNotFoundError(
            f"cannot import backing env {module_name!r}: not installed and "
            f"{env_file} does not exist"
        )
    pkg_parent = str(env_file.parent.parent)
    if pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    return importlib.import_module(f"{module_name}.env")


def compose_instruction(
    *,
    root_prompt: str,
    question: str,
    answer_format: str,
    context_filename: str = PI_CONTEXT_FILENAME,
) -> str:
    """Preamble + task + output-format footer."""
    _ = context_filename
    if str(os.environ.get("PI_PROMPT_VARIANT", "")).lower() == "short":
        preamble = pi_harness_preamble(subagents=_subagents_enabled())
    else:
        preamble = PI_QUERY_PREAMBLE
        if _subagents_enabled():
            preamble = f"{preamble}\n\n{PI_SUBAGENT_NOTE}"
    parts: list[str] = [preamble]
    rp = (root_prompt or "").strip()
    if rp:
        parts.append(rp)
    q = (question or "").strip()
    if q and q not in rp:
        parts.append(f"Question:\n{q}")
    parts.append(answer_format_footer(answer_format))
    return "\n\n".join(parts)


def _row_question(row: dict) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        content = prompt[0].get("content") if isinstance(prompt[0], dict) else None
        if isinstance(content, str):
            return content
    return ""


def row_to_task(row: dict, *, answer_format: str, context_filename: str = PI_CONTEXT_FILENAME) -> dict:
    """One backing-env row -> Pi task dict."""
    info = row.get("info")
    meta = json.loads(info) if isinstance(info, str) else dict(info or {})
    context = meta.get("context", "")
    if isinstance(context, (list, tuple)):
        context = "\n\n".join(str(c) for c in context)
    instruction = compose_instruction(
        root_prompt=str(meta.get("root_prompt", "")),
        question=_row_question(row),
        answer_format=answer_format,
        context_filename=context_filename,
    )
    return {
        "example_id": row.get("example_id"),
        "instruction": instruction,
        "prompt": [{"role": "user", "content": instruction}],
        "context": context,
        "answer_format": answer_format_footer(answer_format),
        "answer": row.get("answer"),
        "info": json.dumps({k: v for k, v in meta.items() if k != "context"}),
    }


def _rows_of(build: Callable[[], Iterable[dict]] | Iterable[dict]) -> list[dict]:
    rows = build() if callable(build) else build
    out: list[dict] = []
    for r in rows:
        out.append(dict(r) if not isinstance(r, dict) else r)
    return out


def make_source(
    build_rows: Callable[[], Iterable[dict]] | Iterable[dict],
    *,
    answer_format: str,
    context_filename: str = PI_CONTEXT_FILENAME,
) -> Callable[[], list[dict]]:
    """Taskset source; the dataset is built on first call."""

    def source() -> list[dict]:
        return [
            row_to_task(r, answer_format=answer_format, context_filename=context_filename)
            for r in _rows_of(build_rows)
        ]

    return source


ScoreFn = Callable[[str, Any, dict], Any]


# Printed by the readcap extension (PI_TURN_CAP_MODE=hard) when it ends a run at the turn cap.
PI_TURN_CAP_SENTINEL = "__PI_TURN_CAP_HIT__"


def _pi_stdout(state: Any) -> str:
    cmd = state.get("command") if isinstance(state, dict) else None
    return str((cmd or {}).get("stdout") or "")


def turn_cap_hit(state: Any) -> bool:
    return PI_TURN_CAP_SENTINEL in _pi_stdout(state)


def _pi_output(state: Any) -> str:
    if turn_cap_hit(state):
        return ""
    return _pi_stdout(state)


PI_ANSWER_MARKER = "final answer:"


def last_marked_answer(output: str, marker: str = PI_ANSWER_MARKER) -> str:
    """Text after the last `marker` in pi's stdout, else the last non-empty line, else everything."""
    text = str(output or "")
    idx = text.lower().rfind(marker.lower())
    if idx != -1:
        return text[idx + len(marker):].strip()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else text.strip()


def _meta_of(task: Any) -> dict:
    info = task.get("info") if isinstance(task, dict) else None
    if isinstance(info, str):
        try:
            return json.loads(info)
        except Exception:
            return {}
    return dict(info or {})


def pi_reward(score_fn: ScoreFn, *, weight: float = 1.0, name: str = "reward") -> Any:
    """Wrap an (output, gold, meta) -> float grader as a reward."""

    @vf.reward(weight=weight)
    async def _reward(task: Any, state: Any, **_kw: Any) -> float:
        try:
            res = score_fn(_pi_output(state), task.get("answer"), _meta_of(task))
            if inspect.isawaitable(res):
                res = await res
            return float(res)
        except Exception:
            return 0.0

    _reward.__name__ = name
    return _reward


def pi_metric(score_fn: ScoreFn, *, name: str) -> Any:
    """Weight-0 metric."""
    return pi_reward(score_fn, weight=0.0, name=name)


def _state_counter_metric(state_key: str, *, name: str) -> Any:
    """Weight-0 metric reading a counter HostPi wrote to state."""

    @vf.reward(weight=0.0)
    async def _metric(state: Any, **_kw: Any) -> float:
        try:
            return float(state.get(state_key) or 0.0)
        except Exception:
            return 0.0

    _metric.__name__ = name
    return _metric


def pi_step_split_metrics() -> list[Any]:
    """pi_root_steps / pi_subagent_steps."""
    from .host_pi import ROOT_STEPS_KEY, SUBAGENT_STEPS_KEY

    return [
        _state_counter_metric(ROOT_STEPS_KEY, name="pi_root_steps"),
        _state_counter_metric(SUBAGENT_STEPS_KEY, name="pi_subagent_steps"),
    ]


def pi_turn_cap_metric() -> Any:
    """pi_turn_cap_hit: 1.0 if the turn cap ended the run."""

    @vf.reward(weight=0.0)
    async def pi_turn_cap_hit(state: Any, **_kw: Any) -> float:
        return 1.0 if turn_cap_hit(state) else 0.0

    return pi_turn_cap_hit


def make_pi_env(
    *,
    build_rows: Callable[[], Iterable[dict]] | Iterable[dict],
    score_fn: ScoreFn,
    answer_format: str,
    reward_name: str = "reward",
    eval_build_rows: Callable[[], Iterable[dict]] | Iterable[dict] | None = None,
    aux_metrics: Sequence[tuple[str, ScoreFn]] = (),
    harness_kwargs: dict | None = None,
    context_filename: str = PI_CONTEXT_FILENAME,
) -> vf.Environment:
    """Build vf.Env(taskset, HostPi)."""
    import os

    _skip = (os.environ.get("VF_PI_SKIP_BACKEND_INSTALL") or "").strip().lower()
    if _skip not in {"1", "true", "yes"}:
        from .process_sandbox import install_local_process_backend

        install_local_process_backend()

    rewards = [pi_reward(score_fn, name=reward_name)]
    rewards += [pi_metric(fn, name=nm) for nm, fn in aux_metrics]
    rewards += pi_step_split_metrics()
    rewards += [pi_turn_cap_metric()]

    taskset = vf.Taskset(
        source=make_source(build_rows, answer_format=answer_format, context_filename=context_filename),
        eval_source=(
            make_source(eval_build_rows, answer_format=answer_format, context_filename=context_filename)
            if eval_build_rows is not None
            else None
        ),
        rewards=rewards,
    )
    harness = HostPi(context_path=context_filename, **(harness_kwargs or {}))
    return vf.Env(taskset=taskset, harness=harness)
