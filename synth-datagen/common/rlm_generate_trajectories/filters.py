"""Heuristic filters + decomposition signature for generated rollouts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .strategies import Strategy

CODE_RE = re.compile(r"```(?:repl|python)?\n(.*?)```", re.S)
CALL_RE = re.compile(r"\bllm_query(?:_batched)?\s*\(")
SUBCALL_ERR_RE = re.compile(r"^Error:|Error: |exceeds maximum context|longer than the maximum model length|"
                            r"sub-?llm.*(?:error|failed)|Proxy request failed", re.I | re.M)
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)|^\w+Error: ", re.M)
STDOUT_TRUNC_RE = re.compile(r"\.\.\. \+ \[\d+ chars\.\.\.\]")
MAX_REPL_OUTPUT_CHARS = 15_000
CONTEXT_READ_RE = re.compile(r"\bcontext\b\s*(?:\[|\.|\)|,|$)|\(\s*context\b|\bin\s+context\b|"
                             r"\bcontext\s*\)|finditer\(.*context|\blen\(\s*context", re.M)
THINK_LEAK_RE = re.compile(r"</?think>|</?tool_call>")
CONTEXT_FEED_RE = re.compile(
    r"\bcontext\b|chunk|\.join\(|"
    r"\+\s*\w+\s*\[|\]\s*\+\s*|"
    r"\{(?:snippet|window|passage|excerpt|relevant|matches|hits|sentences|lines|evidence|candidates|section|"
    r"para|found|filtered|turns?|items?|docs?|text|content|extracted|combined|selected|batch|group|record|rows?|"
    r"entries|messages|facts|prompt_parts|steps?|ev|event|body|block|segment|page|table|data)\w*[\[\]\w.\'\"]*\}",
    re.I)


_Q = r"""[rbfu]*(["'])((?:(?!\1).){1,200})\1"""
SEARCH_CALL_RE = re.compile(r"\bre\.(?:search|findall|finditer|match|fullmatch|compile)\(\s*" + _Q)
FIND_CALL_RE = re.compile(r"\.(?:find|rfind|index|count|startswith|endswith)\(\s*" + _Q)
IN_LITERAL_RE = re.compile(_Q + r"\s+(?:not\s+)?in\s+[\w.\[\]()]+")
KEYWORD_LIST_RE = re.compile(
    r"\b\w*(?:keyword|term|pattern|marker|cue|query|queries|word|needle|trigger|phrase)s?\w*\s*=\s*[\[\(\{](.*?)[\]\)\}]",
    re.S | re.I)
STR_LIT_RE = re.compile(_Q)
CTX_BIND_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*+(?!(?:len|sum|max|min|int|float|round|bool|type|llm_query|llm_query_batched)\()"
    r"(?![^\n]*\.(?:find|rfind|index|rindex|count|start|end)\()(?=[^\n]*\bcontext\b)[^\n]*$", re.M)
STRIPPED_CTX_RE = re.compile(r"^<(?:context string \((\d+) chars\)|\d+ context docs|context) stripped from rollout log>$")
PRINT_RE = re.compile(r"\bprint\((.*)\)\s*$", re.M)
SIZE_CHECK_RE = re.compile(r"\blen\(|\bsum\(\s*len\(|\.__len__\(|\bmax_chars\b|\bMAX_\w*CHARS\b|\bbudget\b", re.I)
_BOUNDED_ARG_RE = re.compile(r"^\s*(?:f?[\"'].*[\"']\s*,?\s*)*(?:len|type|sorted\(set|set|round|int|float|sum|max|min)\(|"
                             r"\[\s*:\s*(?:\d{1,3})\s*\]|\[\s*-\s*\d{1,3}\s*:\s*\]")


_MARKER_RE = re.compile(r"^\s*(?:\[[A-Z _]{2,30}\]|[A-Z][\w ]{0,24}\s*\d*\s*:)\s*$")
_REPLY_NAMES = re.compile(r"^(?:r|rr|res|resp|reply|replies|response|responses|ans|answer_text|out|outs|result|results|v|x|lab|label|labels)$")


def search_patterns(code: str) -> set[str]:
    pats: set[str] = set()
    for rx in (SEARCH_CALL_RE, FIND_CALL_RE, IN_LITERAL_RE):
        for m in rx.finditer(code):
            lit = m.group(2).strip()
            if rx is IN_LITERAL_RE:
                target = re.match(r"\s*(?:not\s+)?in\s+([\w]+)", code[m.end(2) + 1:])
                if target and _REPLY_NAMES.match(target.group(1)):
                    continue
            if len(lit) < 2 or re.fullmatch(r"[\s\W]*", lit) or _MARKER_RE.match(lit):
                continue
            pats.add(lit.lower())
    for m in KEYWORD_LIST_RE.finditer(code):
        for lm in STR_LIT_RE.finditer(m.group(1)):
            if len(lm.group(2).strip()) >= 2:
                pats.add(lm.group(2).strip().lower())
    return pats


def _code_of_print_arg(arg: str) -> str:
    """The code inside a print argument: plain string literals dropped, f-string `{...}` fields kept."""
    out = []
    for m in re.finditer(r"""([rbfuRBFU]*)("[^"\n]*"|'[^'\n]*')|([^"']+)""", arg):
        if m.group(3) is not None:
            out.append(m.group(3))
        elif "f" in (m.group(1) or "").lower():
            out.append(" ".join(re.findall(r"\{([^{}]*)\}", m.group(2))))
    return " ".join(out)


_BOUNDED_FN_RE = re.compile(r"\b(?:len|type|round|int|float|sum|max|min|bool|any|all|isinstance)\(")


def _strip_bounded_calls(s: str) -> str:
    """Remove whole size/count/type calls, nested parentheses included (`len(EDGE.findall(g))`)."""
    while True:
        m = _BOUNDED_FN_RE.search(s)
        if not m:
            return s
        depth, j = 0, m.end() - 1
        while j < len(s):
            depth += {"(": 1, ")": -1}.get(s[j], 0)
            if depth == 0:
                break
            j += 1
        s = s[:m.start()] + " " + s[j + 1:]


def _raw_print_in(code: str, ctx_vars: set[str]) -> bool:
    """A print that emits context material rather than a size/count/short preview."""
    for m in PRINT_RE.finditer(code):
        arg = _code_of_print_arg(m.group(1))
        names = set(re.findall(r"\b([A-Za-z_]\w*)\b", arg))
        if not (names & (ctx_vars | {"context"})):
            continue
        rest = re.sub(r"\b\w+(?:\[[^\]]*\])*(?:\.\w+)*\.(?:start|end|span|count|find|rfind|index|rindex|startswith|endswith|"
                      r"isdigit|isalpha|isupper|islower)\((?:[^()]|\([^()]*\))*\)|\.__name__", "", arg)
        rest = _strip_bounded_calls(rest)
        rest = re.sub(r"\b\w+(?:\[[^\]:]*\])*\[\s*(\d+)\s*:\s*(\d+)\s*\]",
                      lambda mm: "" if 0 <= int(mm.group(2)) - int(mm.group(1)) <= 9999 else mm.group(0), rest)
        rest = re.sub(r"\b\w+(?:\[[^\]:]*\])*\[\s*(\w+)\s*:\s*\1\s*\+\s*\d{1,4}\s*\]", "", rest)
        rest = re.sub(r"\b\w+(?:\[[^\]:]*\])*\[\s*:\s*\d{1,4}\s*\]|\b\w+(?:\[[^\]:]*\])*\[\s*-\s*\d{1,4}\s*:\s*\]", "", rest)
        if set(re.findall(r"\b([A-Za-z_]\w*)\b", rest)) & (ctx_vars | {"context"}):
            return True
    return False


def verbatim_echo_chars(outputs: list[str], context: Any, min_line: int = 40) -> int:
    """Characters of REPL output that are verbatim lines of the context (exact, when the context is known)."""
    if isinstance(context, (list, dict)):
        import json as _json
        ctx = _json.dumps(context, ensure_ascii=False)
    else:
        ctx = str(context or "")
    if not ctx:
        return 0
    n = 0
    for out in outputs:
        for line in out.splitlines():
            t = line.strip().strip("'\"")
            if len(t) >= min_line and t[:200] in ctx:
                n += len(t)
    return n


def code_blocks(msgs: list[dict]) -> list[tuple[int, str]]:
    """(message index, code) for every repl block in assistant messages."""
    out = []
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant":
            for b in CODE_RE.findall(m.get("content") or ""):
                out.append((i, b))
    return out


def _metric(row: dict, key: str, default=0):
    v = row.get(key)
    if v is None:
        v = (row.get("metrics") or {}).get(key)
    return default if v is None else v


@dataclass
class Signature:
    """Which decomposition families a rollout visibly exhibits."""
    families: set[str] = field(default_factory=set)
    n_call_sites: int = 0
    n_batched_sites: int = 0
    n_context_fed_sites: int = 0
    n_subcall_errors: int = 0
    n_tracebacks: int = 0
    n_large_outputs: int = 0
    think_leak: bool = False
    n_single_sites: int = 0
    max_sub_prompt_chars: int = 0
    mean_sub_prompt_chars: float = 0.0
    max_output_chars: int = 0
    total_output_chars: int = 0
    reads_context: bool = False
    unterminated_fence: bool = False
    subcall_errors_from_log: bool = False
    answer_from_subcall: bool = False
    n_search_patterns: int = 0
    raw_echo_chars: int = 0
    raw_echo_exact: bool = False
    size_checked: bool = False
    chunk_fanout: bool = False
    carries_state: bool = False
    context_chars: int = 0
    first_block_answers: bool = False
    first_block_delegates: bool = False
    presupposed_markers: list = field(default_factory=list)
    submit_block_lines: int = 0
    submit_block_calls: bool = False
    submit_guarded: bool = False
    max_block_lines: int = 0

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["families"] = sorted(self.families)
        return d


def _messages(row: dict) -> list[dict]:
    """The rollout's chat messages, however the row spells them."""
    msgs = list(row.get("prompt") or []) + list(row.get("completion") or [])
    return msgs or list(row.get("messages") or [])


def signature(row: dict, soft_cap: int = MAX_REPL_OUTPUT_CHARS) -> Signature:
    msgs = _messages(row)
    sig = Signature()
    blocks = code_blocks(msgs)
    llm_vars: set[str] = set()
    prev_results_used = 0
    for i, b in blocks:
        if CONTEXT_READ_RE.search(b):
            sig.reads_context = True
        if not CALL_RE.search(b):
            continue
        sig.n_call_sites += 1
        if "llm_query_batched" in b:
            sig.n_batched_sites += 1
        else:
            sig.n_single_sites += 1
        if CONTEXT_FEED_RE.search(b):
            sig.n_context_fed_sites += 1
        for v in re.findall(r"(\w+)\s*=\s*llm_query", b):
            llm_vars.add(v)
        if any(re.search(rf"\{{[^}}]*\b{re.escape(v)}\b[^}}]*\}}|\+\s*{re.escape(v)}\b", b) for v in llm_vars if v):
            prev_results_used += 1
        if re.search(r"^\s*(for|while)\b.*:\s*$", b, re.M) and re.search(r"llm_query\(", b) and \
           re.search(r"(state|prev|belief|current|so_far|history|memo)\w*\s*=", b):
            sig.families.add("stateful-loop")
        if re.search(r"verif|double.?check|check (?:my|the|this|that)|is (?:my|this) (?:answer|logic|reasoning) correct", b, re.I):
            sig.families.add("verify-with-subcall")
        if re.search(r"score|rank|relevan", b, re.I) and "llm_query_batched" in b:
            sig.families.add("rank-candidates")
        if re.search(r"classif|label|categor", b, re.I) and "llm_query_batched" in b:
            sig.families.add("classify-count")
        if "llm_query_batched" in b and re.search(r"chunk|split|range\(0,\s*len", b):
            sig.families.add("chunk-fanout-aggregate")
        if re.search(r"re\.(?:search|findall|finditer)|\.find\(|\.index\(|keyword|in .*\.lower\(\)", b) and \
           not re.search(r"llm_query_batched", b):
            sig.families.add("retrieve-then-read")
        if not (row.get("sub_calls")) and i + 1 < len(msgs) and msgs[i + 1].get("role") == "user":
            if SUBCALL_ERR_RE.search(msgs[i + 1].get("content") or ""):
                sig.n_subcall_errors += 1
    if prev_results_used >= 1 and sig.n_call_sites >= 2:
        sig.families.add("dag-subquestions")
    if sig.n_call_sites == 0 and blocks:
        sig.families.add("direct-python")
    for m in msgs:
        if m.get("role") == "assistant":
            c_ = m.get("content") or ""
            if THINK_LEAK_RE.search(c_):
                sig.think_leak = True
            if c_.count("```") % 2 == 1:
                sig.unterminated_fence = True
        if m.get("role") != "user":
            continue
        c = m.get("content") or ""
        if TRACEBACK_RE.search(c):
            sig.n_tracebacks += 1
        if c.startswith("REPL output"):
            sig.max_output_chars = max(sig.max_output_chars, len(c))
            sig.total_output_chars += len(c)
            if STDOUT_TRUNC_RE.search(c) or len(c) > soft_cap:
                sig.n_large_outputs += 1
    log = row.get("sub_calls") or []
    if log:
        sizes = [int(e.get("prompt_chars") or 0) for e in log]
        sig.max_sub_prompt_chars = max(sizes)
        sig.mean_sub_prompt_chars = sum(sizes) / len(sizes)
        sig.n_subcall_errors = sum(1 for e in log if SUBCALL_ERR_RE.search(str(e.get("response") or "")[:200]))
        sig.subcall_errors_from_log = True
    all_code = "\n".join(b for _, b in blocks)
    sig.n_search_patterns = len(search_patterns(all_code))
    ctx_vars: set[str] = set()
    raw_msgs: set[int] = set()
    first_call_seen = False
    for i, b in blocks:
        ctx_vars |= set(CTX_BIND_RE.findall(b))
        if _raw_print_in(b, ctx_vars):
            raw_msgs.add(i)
        if not first_call_seen and CALL_RE.search(b):
            first_call_seen = True
            pre = b[:CALL_RE.search(b).start()]
            earlier = "\n".join(c for j, c in blocks if j < i)
            sig.size_checked = bool(SIZE_CHECK_RE.search(earlier + "\n" + pre))
    outputs = [m.get("content") or "" for m in msgs if m.get("role") == "user" and (m.get("content") or "").startswith("REPL output")]
    info = row.get("info") or {}
    ctx = info.get("context")
    stub = STRIPPED_CTX_RE.match(ctx) if isinstance(ctx, str) else None
    if stub:
        ctx = None
        info = {**info, "context_chars": int(stub.group(1) or 0) or info.get("context_chars")}
    code_level = sum(len(msgs[i + 1].get("content") or "") for i in raw_msgs
                     if i + 1 < len(msgs) and msgs[i + 1].get("role") == "user")
    from rlm_train import doc_store
    if doc_store.is_ref(ctx):
        exact = doc_store.verbatim_chars(ctx, (ln for out in outputs for ln in out.splitlines()))
        sig.raw_echo_chars = exact
        sig.raw_echo_exact = True
        sig.context_chars = doc_store.total_chars(ctx)
    elif ctx:
        exact = verbatim_echo_chars(outputs, ctx)
        sig.raw_echo_chars = max(exact, code_level)
        sig.raw_echo_exact = exact >= code_level
        sig.context_chars = len(ctx) if isinstance(ctx, str) else len(json.dumps(ctx))
    else:
        sig.raw_echo_chars = code_level
        sig.context_chars = int(info.get("context_chars") or 0)
    sig.chunk_fanout = bool(sig.families & {"chunk-fanout-aggregate", "classify-count", "structured-extraction"})
    sig.carries_state = "stateful-loop" in sig.families or prev_results_used >= 1
    code_lines = lambda b: [ln for ln in b.splitlines() if ln.strip() and not ln.strip().startswith("#")]  # noqa: E731
    if blocks:
        first_msg = blocks[0][0]
        first = "\n".join(b for i, b in blocks if i == first_msg)
        sig.first_block_answers = bool(re.search(r"answer\[\s*['\"](?:content|ready)['\"]\s*\]\s*=", first))
        sig.first_block_delegates = bool(CALL_RE.search(first)) and bool(re.search(r"\bcontext\b", first))
        ctx0 = (row.get("info") or {}).get("context")
        if not sig.first_block_delegates and isinstance(ctx0, (dict, list)):
            second_msg = next((i for i, _ in blocks if i != first_msg), None)
            second = "\n".join(b for i, b in blocks if i == second_msg) if second_msg is not None else ""
            sig.first_block_delegates = bool(CALL_RE.search(second))
        prompt_text = "\n".join(m.get("content") or "" for m in msgs[:first_msg] if m.get("role") in ("user", "system"))
        lits = re.findall(r"context(?:\[[^\]]*\])?\.(?:split|find|rfind|index|rindex|partition|rpartition|count|startswith)"
                          r"\(\s*" + _Q, first)
        lits += re.findall(r"\bre\.(?:split|search|findall|finditer|match|compile)\(\s*" + _Q, first)
        seen = set()
        for _, lit in lits:
            core = re.sub(r"\\[nrtsdwbSDW]|[\^$()\[\]*+?{}|.\\]", " ", lit).strip()
            if len(core) >= 3 and not re.fullmatch(r"[\s\W\d]*", core) and core not in prompt_text and core not in seen:
                seen.add(core)
                sig.presupposed_markers.append(core[:40])
        submit = [b for _, b in blocks if re.search(r"answer\[\s*['\"]ready['\"]\s*\]\s*=\s*True", b)]
        if submit:
            sig.submit_block_lines = len(code_lines(submit[-1]))
            sig.submit_block_calls = bool(CALL_RE.search(submit[-1]))
            sig.submit_guarded = _ready_under_if(submit[-1])
        sig.max_block_lines = max(len(code_lines(b)) for _, b in blocks)
    ans_blocks = [b for _, b in blocks if 'answer["content"]' in b or "answer['content']" in b]
    if ans_blocks and llm_vars:
        sig.answer_from_subcall = any(
            re.search(rf"answer\[.content.\]\s*=\s*[^\n]*\b{re.escape(v)}\b", ans_blocks[-1]) for v in llm_vars if v)
    return sig


@dataclass
class Verdict:
    ok: bool
    reasons: list[str]
    signature: Signature
    interesting: bool = False


def live_slice(row: dict, live_from: int) -> dict:
    """The row restricted to turns >= ``live_from`` (1-based); a replayed prefix is masked in training, so skip it."""
    comp = list(row.get("completion") or [])
    seen, start = 0, len(comp)
    for i, m in enumerate(comp):
        if m.get("role") == "assistant":
            seen += 1
            if seen == live_from:
                start = i
                break
    out = dict(row, completion=comp[start:])
    if row.get("sub_calls"):
        out["sub_calls"] = [e for e in row["sub_calls"] if int(e.get("step") or live_from) >= live_from]
    return out


PROGRAMMATIC_RATIONALE_RE = re.compile(
    r"(programmatic(ally)?|pure(ly)? (python|code)|(in|with) (plain )?(python|code)( alone)?|deterministic(ally)?|"
    r"exact(ly)? (in|with) (python|code)|no (need for )?sub-?calls?|(do(es)?n.?t|not) need (a |any )?sub-?calls?|"
    r"without (any )?sub-?calls?|code (alone )?(is enough|suffices|computes)|computes? (this|it|the answer) exactly)", re.I)


def _ready_under_if(block: str) -> bool:
    """True if answer["ready"] = True is set under an `if`."""
    lines = block.splitlines()
    for i, ln in enumerate(lines):
        if re.search(r"answer\[\s*['\"]ready['\"]\s*\]\s*=\s*True", ln):
            ind = len(ln) - len(ln.lstrip())
            for prev in reversed(lines[:i]):
                if not prev.strip():
                    continue
                pind = len(prev) - len(prev.lstrip())
                if pind < ind:
                    if re.match(r"\s*(if|elif)\b", prev):
                        return True
                    ind = pind
                    if pind == 0:
                        break
            return False
    return False


def _all_prose(row: dict) -> str:
    out = []
    for m in row.get("completion") or row.get("messages") or []:
        if (m.get("role") if isinstance(m, dict) else None) == "assistant":
            out.append(re.sub(r"```.*?```", " ", m.get("content") or "", flags=re.S))
            out.append(m.get("reasoning_content") or "")
    return "\n".join(out)


def check(row: dict, st: Strategy, rubric: str = "v1") -> Verdict:
    """Cheap acceptance heuristics. ``ok`` means "send to the judge"."""
    live_from = int(row.get("_reverify_live_from") or 1) if rubric == "reverify" else 1
    branch = live_from > 1
    sig = signature(live_slice(row, live_from) if branch else row, getattr(st, "max_repl_output_chars", MAX_REPL_OUTPUT_CHARS))
    if branch and not sig.size_checked:
        sig.size_checked = signature(row, getattr(st, "max_repl_output_chars", MAX_REPL_OUTPUT_CHARS)).size_checked
    reasons: list[str] = []
    reward = float(row.get("reward") or 0.0)
    if row.get("error"):
        reasons.append(f"rollout error: {str(row['error'])[:80]}")
    if reward < st.min_reward:
        reasons.append(f"reward {reward:.3f} < {st.min_reward}")
    if not _metric(row, "rlm_has_final_answer", 0):
        reasons.append("no final answer")
    if row.get("is_truncated"):
        submitted = bool(_metric(row, "rlm_has_final_answer", 0)) and row.get("stop_condition") == "has_final_answer"
        if not submitted or sig.unterminated_fence:
            why = "unterminated code fence" if sig.unterminated_fence else "no answer submitted"
            reasons.append(f"truncated ({row.get('stop_condition')}): {why}")
    iters = int(_metric(row, "rlm_iterations", 0))
    if iters < 2 and not (sig.n_call_sites or sig.reads_context):
        reasons.append(f"only {iters} iteration(s) and no block reads `context` or delegates (a guess, not a solve)")
    calls = int(_metric(row, "rlm_sub_llm_calls", 0))
    if calls < st.min_sub_calls:
        reasons.append(f"{calls} sub-call(s) < min {st.min_sub_calls}")
    if st.max_sub_calls is not None and calls > st.max_sub_calls:
        reasons.append(f"{calls} sub-call(s) > max {st.max_sub_calls}")
    if st.require_subcall_context and st.min_sub_calls > 0 and sig.n_call_sites and sig.n_context_fed_sites == 0:
        reasons.append("no sub-call is fed context-derived text")
    if sig.n_subcall_errors > st.max_subcall_errors:
        reasons.append(f"{sig.n_subcall_errors} sub-call error reply/replies")
    if sig.n_tracebacks > (max(st.max_repl_errors, 3) if rubric == "reverify" else st.max_repl_errors):
        reasons.append(f"{sig.n_tracebacks} REPL tracebacks")
    if sig.n_large_outputs:
        reasons.append(f"{sig.n_large_outputs} large/truncated REPL output(s) (max {sig.max_output_chars} chars)")
    if sig.total_output_chars > st.max_total_output_chars:
        reasons.append(f"{sig.total_output_chars} chars of REPL output in total (> {st.max_total_output_chars}): context crawling")
    if sig.think_leak:
        reasons.append("thinking/tool-call tags (<think>, <tool_call>) in an assistant turn")
    calls_total = int(_metric(row, "rlm_sub_llm_calls", 0))
    if sig.n_single_sites >= st.max_sequential_single_sites and sig.n_batched_sites == 0 and calls_total >= st.max_sequential_single_sites:
        reasons.append(f"{sig.n_single_sites} sequential single llm_query sites, none batched (parallelise)")
    if sig.max_sub_prompt_chars > st.max_sub_prompt_chars:
        reasons.append(f"a sub-call prompt of {sig.max_sub_prompt_chars} chars (> {st.max_sub_prompt_chars})")
    if sig.n_call_sites == 0 and not st.python_exact and sig.context_chars > st.very_short_chars and \
            not (branch and int(_metric(row, "rlm_sub_llm_calls", 0)) > 0):
        reasons.append(f"no sub-call on a {sig.context_chars}-char context (> {st.very_short_chars}; not python-exact)")
    if sig.n_search_patterns > st.max_search_patterns:
        reasons.append(f"{sig.n_search_patterns} distinct search patterns (> {st.max_search_patterns}): guess-and-grep")
    if sig.raw_echo_chars > st.max_raw_echo_chars:
        how = "verbatim context" if sig.raw_echo_exact else "raw-variable prints"
        reasons.append(f"{sig.raw_echo_chars} chars of {how} in REPL output (> {st.max_raw_echo_chars})")
    if sig.max_sub_prompt_chars > st.unsized_prompt_chars and not sig.size_checked:
        reasons.append(f"a {sig.max_sub_prompt_chars}-char sub-prompt with no size check before the first sub-call")
    if rubric in ("v2", "reverify"):
        tiny = sig.context_chars and sig.context_chars <= 1500
        prog = bool(getattr(st, "python_exact", False))
        if sig.first_block_answers and not branch and not prog:
            reasons.append("v2: the first block assigns the answer (no probe turn)")
        if not tiny and not sig.first_block_delegates and not branch and not prog \
                and not getattr(st, "probe_from_prompt_ok", False):
            reasons.append("v2: the first block does not ask a sub-call to describe part of the context")
        if prog and not branch and not PROGRAMMATIC_RATIONALE_RE.search(_all_prose(row)):
            reasons.append("programmatic: no statement that a programmatic strategy suffices (and why)")
        if sig.presupposed_markers and not branch:
            reasons.append(f"v2: the first block cuts the context on unseen markers {sig.presupposed_markers[:3]}")
        if not sig.submit_guarded and (sig.submit_block_calls or (sig.submit_block_lines > 10 and not prog)):
            reasons.append(f"v2: the submit block computes ({sig.submit_block_lines} lines, sub-call={sig.submit_block_calls})"
                           " instead of submitting an already-checked value")
        if iters < 3 and not tiny and not branch and not prog:
            reasons.append(f"v2: {iters} turns -- no room for probe, compute and check")
        if sig.max_block_lines > int(getattr(st, "max_block_lines", 60) or 60):
            reasons.append(f"v2: a {sig.max_block_lines}-line block (one job per block)")
    if rubric != "reverify":
        from .reverify import repeated_blocks, turns_of
        reps = repeated_blocks(turns_of(row.get("completion") or []))
        if reps:
            r = reps[0]
            reasons.append(f"repeat: turn {r['turn']} repeats turn {r['of']} ({r['kind']}, sim {r['sim']})")
    if rubric == "reverify":
        from .reverify import mechanical_reasons
        reasons += mechanical_reasons(row, live_from=int(row.get("_reverify_live_from") or 1),
                                      reasoning_needles=tuple(row.get("_reverify_needles") or ()))
    if rubric == "collect":
        reasons.append("collect-only: prefix pool row")
    ok = not reasons
    interesting = (
        reward < st.min_reward and not row.get("error") and sig.n_subcall_errors == 0 and sig.n_large_outputs == 0 and not sig.think_leak
        and sig.n_tracebacks <= st.max_repl_errors and sig.n_call_sites >= 1
        and bool(sig.families) and not (sig.families & set(st.families))
        and bool(_metric(row, "rlm_has_final_answer", 0))
    )
    return Verdict(ok=ok, reasons=reasons, signature=sig, interesting=interesting)

_GENERIC_IDENTS = frozenset("""
answer content ready context print range len str int float list dict set tuple sorted enumerate zip
import from return def for while if elif else try except finally with as in not and or None True False
llm_query llm_query_batched rlm_query re match search findall finditer group start end split join strip
lower upper replace append extend items keys values get sum max min abs round sort reverse format
prompt prompts response responses result results text line lines word words data value values idx index
i j k n m x y z s t f g b c d e p q r v w a o u l h
""".split())


def hint_identifiers(hint: str) -> set[str]:
    """Distinctive identifiers appearing in a hint's own code blocks."""
    idents: set[str] = set()
    for block in CODE_RE.findall(hint):
        for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]{3,}", block):
            if tok not in _GENERIC_IDENTS and not tok.startswith("__"):
                idents.add(tok)
    return idents


def hint_compliance(row: dict, hint: str) -> float:
    """Share of the hint's distinctive identifiers that appear in the rollout's own code (0.0-1.0)."""
    idents = hint_identifiers(hint)
    if not idents:
        return -1.0
    code = "\n".join(b for _, b in code_blocks(_messages(row)))
    present = {t for t in idents if re.search(rf"\b{re.escape(t)}\b", code)}
    return len(present) / len(idents)
