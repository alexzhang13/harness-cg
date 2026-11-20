"""RLMTrainEnv: rlm.RLM.completion at depth 1 as a verifiers MultiTurnEnv, one REPL subprocess per rollout.

Only root turns are trajectory steps; sub-LLM calls go through a localhost proxy and are logged as metrics. The loop
runs in get_prompt_messages: execute the previous turn's repl blocks, append the output, add the next turn prompt.
enable_sub_llm=False gives LM + REPL (no sub-LLM tools, untruncated REPL output).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Callable

import verifiers as vf
from verifiers.types import Messages, State

from rlm.utils.parsing import find_code_blocks


from rlm_train import doc_store
from rlm_train.prompts import (
    CODE_ONLY_SYSTEM_PROMPT,
    ORCHESTRATOR_PROMPT,
    RLM_SYSTEM_PROMPT,
    build_rlm_system_prompt,
    build_short_user_prompt,
)
from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.repl.base import ExecResult, ReplBackend
from rlm_train.repl.subprocess import SubprocessReplBackend
from rlm_train.rubric import RLMTrainRubric

logger = logging.getLogger(__name__)


def _strip_leaked_reasoning(text: str) -> str:
    """Drop leaked reasoning (up to the last </think>) so code drafted there is not executed."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").lstrip("\n")

# Per-block cap on REPL output shown to the root model (as in rlm.utils.parsing.format_iteration).
_MAX_REPL_OUTPUT_CHARS = 20_000

# Limits of the sub-call log in state["rlm_sub_llm_log"] (entries, chars); *_REPLAY when info["rlm_replay_log"] is set.
_SUB_LOG_MAX_ENTRIES = 400
_SUB_LOG_MAX_CHARS = 6_000
_SUB_LOG_MAX_ENTRIES_REPLAY = 4_000
_SUB_LOG_MAX_RESPONSE_CHARS_REPLAY = 200_000


def _record_sub_call(state: dict, meta: dict) -> None:
    """Proxy sink: count the sub-call and keep a bounded transcript of it in the state."""
    state["rlm_sub_llm_calls"] = int(state.get("rlm_sub_llm_calls") or 0) + 1
    # token counts are summed separately from the (truncated) log
    usage = meta.get("usage") or {}
    prompt_tokens = meta.get("prompt_tokens", usage.get("prompt_tokens"))
    completion_tokens = meta.get("completion_tokens", usage.get("completion_tokens"))
    if prompt_tokens is not None:
        state["rlm_sub_llm_prompt_tokens"] = (
            int(state.get("rlm_sub_llm_prompt_tokens") or 0) + int(prompt_tokens))
    if completion_tokens is not None:
        state["rlm_sub_llm_completion_tokens"] = (
            int(state.get("rlm_sub_llm_completion_tokens") or 0) + int(completion_tokens))
    prompt = meta.get("prompt")
    response = meta.get("response")
    p_chars = len(prompt) if isinstance(prompt, str) else len(str(prompt or ""))
    r_chars = len(response) if isinstance(response, str) else len(str(response or ""))
    state["rlm_sub_llm_prompt_chars"] = int(state.get("rlm_sub_llm_prompt_chars") or 0) + p_chars
    state["rlm_sub_llm_response_chars"] = int(state.get("rlm_sub_llm_response_chars") or 0) + r_chars
    log = state.setdefault("rlm_sub_llm_log", [])
    full = bool(state.get("rlm_replay_log"))
    if len(log) < (_SUB_LOG_MAX_ENTRIES_REPLAY if full else _SUB_LOG_MAX_ENTRIES):
        p_text = prompt if isinstance(prompt, str) else str(prompt or "")
        r_text = response if isinstance(response, str) else str(response or "")
        entry = {"i": len(log), "prompt_chars": p_chars, "response_chars": r_chars,
                 "prompt": p_text[:_SUB_LOG_MAX_CHARS],
                 "response": r_text[:_SUB_LOG_MAX_RESPONSE_CHARS_REPLAY if full else _SUB_LOG_MAX_CHARS]}
        if prompt_tokens is not None:
            entry["prompt_tokens"] = int(prompt_tokens)
        if completion_tokens is not None:
            entry["completion_tokens"] = int(completion_tokens)
        if meta.get("model") is not None:
            entry["model"] = meta["model"]
        if full:
            import hashlib
            entry["prompt_sha1"] = hashlib.sha1(p_text.encode("utf-8", "surrogatepass")).hexdigest()
            entry["step"] = state.get("rlm_replay_cur_step") or (
                int(state.get("rlm_replay_offset") or 0) + int(state.get("rlm_n_processed") or 0) + 1)
            if meta.get("batch_id") is not None:
                entry["batch_id"], entry["batch_index"] = meta["batch_id"], meta.get("batch_index")
        log.append(entry)
        if full:
            _write_snapshot(state, state.get("rlm_history") or [])


def _write_snapshot(state: dict, history: list) -> None:
    """If info["rlm_snapshot_path"] is set, save the rollout so far as a replay spec after every turn, so a killed
    rollout can be resumed with info["rlm_replay"]."""
    info = state.get("info")
    path = info.get("rlm_snapshot_path") if isinstance(info, dict) else None
    if not path:
        return
    def role(m: Any) -> Any:
        return getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)

    prefix, outputs = [], []
    for i, m in enumerate(history):
        if role(m) != "assistant":
            continue
        prefix.append(_strip_leaked_reasoning(_msg_text(m)))
        nxt = history[i + 1] if i + 1 < len(history) else None
        outputs.append(_msg_text(nxt) if nxt is not None and role(nxt) == "user" else None)
    spec = {"version": 1, "prefix": prefix, "expected_outputs": outputs,
            "sub_calls": list(state.get("rlm_sub_llm_log") or []), "hint": None}
    import json
    import os
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(spec, fh)
        os.replace(tmp, path)
    except OSError:
        pass


class RLMTrainEnv(vf.MultiTurnEnv):
    """RLM harness (depth 1), or LM + REPL with enable_sub_llm=False.

    Rows need info["context"] (the REPL `context`), info["root_prompt"] and `answer`.

    sub_llm_fn / sub_llm_fn_batched: programmed sub-LLM replies, called as (prompt(s), state); None falls through
        to the model.
    user_prologue: extra user message after the metadata message (task hints).
    max_sub_calls / max_rollout_seconds: end the rollout with no answer when exceeded.
    default_answer_at_limit: at the turn limit, ask once more for a final answer (eval only).
    """

    def __init__(
        self,
        backend_factory: Callable[[], ReplBackend] | None = None,
        max_iterations: int = 30,
        sub_model: str | None = None,
        sub_sampling_args: dict[str, Any] | None = None,
        sub_extra_body: dict[str, Any] | None = None,
        custom_system_prompt: str | None = None,
        enable_sub_llm: bool = True,
        rubric: vf.Rubric | None = None,
        sub_llm_fn: Callable[[str, Any], Any] | None = None,
        sub_llm_fn_batched: Callable[[list[str], Any], Any] | None = None,
        user_prologue: str | None = None,
        orchestrator_prompt: bool = False,
        max_sub_calls: int | None = None,
        max_rollout_seconds: float | None = None,
        default_answer_at_limit: bool = False,
        code_block_feedback: bool | None = None,
        **kwargs: Any,
    ):
        if "max_turns" in kwargs:
            raise ValueError("Use `max_iterations` instead of `max_turns` for RLMTrainEnv")
        if orchestrator_prompt and not enable_sub_llm:
            raise ValueError("orchestrator_prompt=True needs enable_sub_llm=True (the orchestrator prompt is about delegating to sub-LLMs)")
        rubric = rubric or RLMTrainRubric()
        super().__init__(
            max_turns=max_iterations,
            rubric=rubric,
            **kwargs,
        )
        self._backend_factory = backend_factory or (lambda: SubprocessReplBackend())
        self._max_iterations = max_iterations
        self._default_answer_at_limit = default_answer_at_limit
        self._sub_model = sub_model
        self._sub_sampling_args = dict(sub_sampling_args or {"max_tokens": 4096})
        if sub_extra_body:
            merged = dict(self._sub_sampling_args.get("extra_body") or {})
            merged.update(sub_extra_body)
            self._sub_sampling_args["extra_body"] = merged
        self._enable_sub_llm = enable_sub_llm
        # LM + REPL has no sub-LLM to offload long text to, so its REPL output is not truncated.
        self._max_repl_output_chars = _MAX_REPL_OUTPUT_CHARS if enable_sub_llm else None
        if custom_system_prompt is not None:
            base_system_prompt = custom_system_prompt
        elif enable_sub_llm:
            base_system_prompt = RLM_SYSTEM_PROMPT
        else:
            base_system_prompt = CODE_ONLY_SYSTEM_PROMPT
        if orchestrator_prompt:
            base_system_prompt = f"{base_system_prompt}\n\n{ORCHESTRATOR_PROMPT}"
        self._system_prompt = base_system_prompt
        self._user_prologue = user_prologue
        self._sub_llm_fn = sub_llm_fn
        self._sub_llm_fn_batched = sub_llm_fn_batched
        self._max_sub_calls = max_sub_calls
        self._max_rollout_seconds = max_rollout_seconds
        # A turn with no runnable ```repl block gets a "No REPL output: ..." user message when on; when off (default,
        # the Section 3 setting) the next message is just the turn prompt. Unset -> env var RLM_TRAIN_CODE_BLOCK_FEEDBACK=1.
        if code_block_feedback is None:
            code_block_feedback = os.environ.get("RLM_TRAIN_CODE_BLOCK_FEEDBACK", "0") == "1"
        self._code_block_feedback = code_block_feedback
        self._proxy: SubLLMProxy | None = None
        self._proxy_lock: asyncio.Lock | None = None

    async def _ensure_proxy(self) -> SubLLMProxy:
        if self._proxy_lock is None:
            self._proxy_lock = asyncio.Lock()
        async with self._proxy_lock:
            if self._proxy is None:
                proxy = SubLLMProxy()
                await proxy.start()
                self._proxy = proxy
            return self._proxy

    async def _teardown_proxy(self) -> None:
        lock = self._proxy_lock or asyncio.Lock()
        async with lock:
            if self._proxy is not None:
                await self._proxy.stop()
                self._proxy = None

    def _build_user_iter(self, iteration: int) -> dict[str, str]:
        return build_short_user_prompt(iteration=iteration, max_iterations=self._max_iterations)

    async def setup_state(self, state: State) -> None:
        await super().setup_state(state)

        info = state.get("info") or {}
        context_payload = info.get("context")
        if context_payload is None:
            raise ValueError(
                "RLMTrainEnv requires `info['context']` on each dataset row -- "
                "this is the payload exposed as the `context` REPL variable. "
                "If your task has no separate context, set it to the same string "
                "as the query."
            )
        root_prompt: str | None = info.get("root_prompt")

        rollout_id = f"rlm_{uuid.uuid4().hex[:12]}"
        proxy = await self._ensure_proxy()

        # prefix replay (rlm_train.replay): replayed turns get their recorded sub-call replies
        replay_spec = info.get("rlm_replay") if isinstance(info, dict) else None
        fake_query, fake_query_batched, replay_stub = self._sub_llm_fn, self._sub_llm_fn_batched, None
        if replay_spec:
            from rlm_train.replay import ReplayStub
            replay_stub = ReplayStub(replay_spec.get("sub_calls"), fake_query, fake_query_batched)
            fake_query, fake_query_batched = replay_stub.single, replay_stub.batched
        if replay_spec or (isinstance(info, dict) and info.get("rlm_replay_log")):
            state["rlm_replay_log"] = True

        proxy.register(
            rollout_id,
            ClientHandle(
                client=state["client"],
                model=self._sub_model or state["model"],
                sampling_args=self._sub_sampling_args,
                record_call=lambda meta: _record_sub_call(state, meta),
                fake_query=fake_query,
                fake_query_batched=fake_query_batched,
                state_ref=state,
                max_sub_calls=self._max_sub_calls,
                max_concurrent=int(os.environ.get("RLM_TRAIN_SUB_CONCURRENCY") or 16),
            ),
        )

        backend = self._backend_factory()
        await backend.start(
            proxy_url=proxy.url,
            rollout_id=rollout_id,
            depth=1 if self._enable_sub_llm else 0,
        )
        await backend.load_context(context_payload)

        base = build_rlm_system_prompt(
            system_prompt=self._system_prompt,
            query_metadata=doc_store.query_metadata(context_payload),
            root_prompt=root_prompt,
            sub_llm=self._enable_sub_llm,
        )

        state["rlm_rollout_id"] = rollout_id
        state["rlm_backend"] = backend
        state["rlm_root_prompt"] = root_prompt
        state["rlm_history"] = list(base)
        state["rlm_n_processed"] = 0
        state["rlm_iterations"] = 0
        state["rlm_repl_calls"] = 0
        state["rlm_sub_llm_calls"] = 0
        state["rlm_final_answer"] = None
        state["rlm_start_time"] = time.monotonic()
        state["rlm_final_repl_outputs"] = []

        # [system, metadata, (prologue,) turn_1, assistant_1, repl_1, turn_2, ...]
        prologue = self._user_prologue
        extra = info.get("rlm_prologue_extra") if isinstance(info, dict) else None
        if extra:
            prologue = f"{prologue}\n\n{extra}" if prologue else extra
        if prologue:
            state["rlm_history"].append(
                {"role": "user", "content": prologue}
            )

        state["rlm_history"].append(self._build_user_iter(0))
        state["prompt"] = list(state["rlm_history"])

        if replay_spec:
            from rlm_train.replay import replay_prefix
            await replay_prefix(self, state, replay_spec, replay_stub)

    async def _fold_pending_steps(self, state: State) -> bool:
        """Run the repl blocks of steps not yet in rlm_history; True once a step submits an answer."""
        history: list = state["rlm_history"]
        backend: ReplBackend = state["rlm_backend"]
        n_done = len(state["trajectory"])
        n_processed = int(state.get("rlm_n_processed") or 0)
        offset = int(state.get("rlm_replay_offset") or 0)

        while n_processed < n_done:
            step = state["trajectory"][n_processed]
            assistant_msg = _last_assistant(step["completion"])
            assistant_text = _strip_leaked_reasoning(_msg_text(assistant_msg))

            outputs: list[dict[str, Any]] = []
            final_from_answer: str | None = None
            code_blocks = list(find_code_blocks(assistant_text))
            for code in code_blocks:
                try:
                    result = await backend.execute(code)
                except Exception as e:  # noqa: BLE001
                    # dead worker: stop the rollout
                    proc = getattr(backend, "_proc", None)
                    if proc is not None and getattr(proc, "returncode", None) is not None:
                        state["rlm_worker_dead"] = True
                    outputs.append({
                        "code": code, "stdout": "", "stderr": f"Worker error: {e}",
                        "final_answer": None, "locals_keys": [],
                    })
                    continue
                outputs.append(_pack_exec(code, result))
                state["rlm_repl_calls"] = int(state.get("rlm_repl_calls") or 0) + 1
                if result.final_answer is not None and final_from_answer is None:
                    final_from_answer = result.final_answer

            repl_msgs = _format_repl_outputs(outputs, self._max_repl_output_chars)
            if not code_blocks and self._code_block_feedback:
                repl_msgs = _missing_code_block_message(assistant_text)
            history.append(assistant_msg)
            history.extend(repl_msgs)
            state["rlm_n_processed"] = n_processed + 1
            state["rlm_iterations"] = offset + n_processed + 1
            n_processed += 1

            if final_from_answer is not None:
                state["rlm_final_answer"] = final_from_answer
                state["final_answer"] = final_from_answer   # alias read by some rubrics
                state["final_env_response"] = repl_msgs
                state["rlm_final_repl_outputs"] = repl_msgs
                return True

        return False

    async def get_prompt_messages(self, state: State) -> Messages:
        """rlm_history after running the previous turn's code, plus the next turn prompt (kept in history)."""
        if not state["trajectory"]:
            if state.get("rlm_replay_offset") is not None:
                return _normalize_for_api(list(state["rlm_history"]))
            return list(state["prompt"])

        history: list = state["rlm_history"]
        if await self._fold_pending_steps(state):
            return _normalize_for_api(history)
        offset = int(state.get("rlm_replay_offset") or 0)
        n_processed = int(state.get("rlm_n_processed") or 0)

        _write_snapshot(state, history)

        history.append(self._build_user_iter(offset + n_processed))
        return _normalize_for_api(history)

    async def env_response(self, messages: Messages, state: State, **kwargs: Any) -> Messages | str:
        """Unused; the loop is in get_prompt_messages."""
        return []

    @vf.stop(priority=10)
    async def final_turn_flush(self, state: State) -> bool:
        """Run the code of the last allowed turn, so an answer submitted there counts."""
        trajectory = state.get("trajectory") or []
        if not trajectory or state.get("rlm_final_answer") is not None:
            return False
        offset = int(state.get("rlm_replay_offset") or 0)
        if offset + len(trajectory) < self._max_iterations:
            return False
        if int(state.get("rlm_n_processed") or 0) < len(trajectory) and state.get("rlm_backend") is not None:
            await self._fold_pending_steps(state)
        if state.get("rlm_final_answer") is None and self._default_answer_at_limit:
            await self._default_answer(state)
        return False

    async def _default_answer(self, state: State) -> None:
        """As rlm.RLM._default_answer: one extra call asking for a final answer (not added to the trajectory)."""
        prompt = _normalize_for_api(list(state["rlm_history"]) + [{
            "role": "assistant",
            "content": "Please provide a final answer to the user's question based on the information provided.",
        }])
        try:
            resp = await self.get_model_response(state, prompt)
            text = _msg_text(resp.message) if resp is not None else ""
        except Exception as e:  # noqa: BLE001
            state["rlm_default_answer_error"] = repr(e)[:200]
            return
        if text and text.strip():
            state["rlm_final_answer"] = text
            state["final_answer"] = text
            state["rlm_default_answer"] = True

    @vf.stop
    async def has_final_answer(self, state: State) -> bool:
        return state.get("rlm_final_answer") is not None

    @vf.stop
    async def repl_worker_dead(self, state: State) -> bool:
        """The REPL worker process died (watchdog kill or crash)."""
        return bool(state.get("rlm_worker_dead"))

    @vf.stop
    async def replay_turn_budget(self, state: State) -> bool:
        """Turn limit for a replayed rollout, whose live turns start at turn k."""
        offset = int(state.get("rlm_replay_offset") or 0)
        return offset > 0 and offset + len(state["trajectory"]) >= self._max_iterations

    @vf.stop
    async def sub_call_cap_exceeded(self, state: State) -> bool:
        """Stop (no answer) once max_sub_calls is exceeded."""
        cap = self._max_sub_calls
        if cap is None or cap <= 0:
            return False
        count = int(state.get("rlm_sub_llm_calls") or 0)
        if count > cap:
            state["is_truncated"] = True
            return True
        return False

    @vf.stop
    async def rollout_timeout_exceeded(self, state: State) -> bool:
        """Stop (no answer) once max_rollout_seconds has passed."""
        budget = self._max_rollout_seconds
        if budget is None or budget <= 0:
            return False
        start = state.get("rlm_start_time")
        if start is None:
            return False
        if time.monotonic() - start > budget:
            state["is_truncated"] = True
            return True
        return False

    @vf.cleanup
    async def cleanup_rlm(self, state: State) -> None:
        backend = state.get("rlm_backend")
        if backend is not None:
            try:
                await backend.stop()
            except Exception:
                logger.exception("backend stop failed")
            state["rlm_backend"] = None
        rollout_id = state.get("rlm_rollout_id")
        if rollout_id and self._proxy is not None:
            self._proxy.unregister(rollout_id)

        # Drop the context payload from the saved rollout record.
        info = state.get("info")
        if isinstance(info, dict) and "context" in info:
            ctx = info["context"]
            if isinstance(ctx, (list, tuple)):
                summary = f"<{len(ctx)} context docs stripped from rollout log>"
            elif isinstance(ctx, str):
                summary = f"<context string ({len(ctx)} chars) stripped from rollout log>"
            else:
                summary = "<context stripped from rollout log>"
            state["info"] = {**info, "context": summary}

    @vf.teardown
    async def teardown_rlm(self) -> None:
        await self._teardown_proxy()


def _normalize_for_api(msgs: list) -> list:
    """Copy of the messages with no None assistant `content` (uses reasoning_content, else "")."""
    out: list = []
    for m in msgs:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "assistant":
            out.append(m)
            continue
        content = (m.get("content") if isinstance(m, dict)
                   else getattr(m, "content", None))
        if content is not None and content != "":
            out.append(m)
            continue
        reasoning = (m.get("reasoning_content") if isinstance(m, dict)
                     else getattr(m, "reasoning_content", None))
        if isinstance(m, dict):
            new = dict(m)
        else:
            try:
                new = m.model_dump()
            except AttributeError:
                new = {"role": role, "content": content}
        new["content"] = reasoning if reasoning else ""
        out.append(new)
    return out


def _last_assistant(completion: Any) -> Any:
    """Return the last assistant message in a completion (Pydantic or dict)."""
    if not completion:
        return {"role": "assistant", "content": ""}
    if isinstance(completion, list):
        for m in reversed(completion):
            r = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
            if r == "assistant":
                return m
        return completion[-1]
    return completion


def _msg_text(msg: Any) -> str:
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else None)
            if t:
                parts.append(str(t))
        return "".join(parts)
    return content or ""


def _pack_exec(code: str, result: ExecResult) -> dict[str, Any]:
    return {
        "code": code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "locals_keys": result.locals_keys,
        "final_answer": result.final_answer,
        "stdout_truncated_chars": result.stdout_truncated_chars,
        "stderr_truncated_chars": result.stderr_truncated_chars,
    }


_UNCLOSED_FENCE_MSG = (
    "No REPL output: your message ended in the middle of a ```repl block, so "
    "the block never closed and nothing ran. Your turn has a hard output "
    "limit and you exceeded it. Send a MUCH shorter block -- a few lines that "
    "print one intermediate result -- and build up what you need over several "
    "turns. Do not resend the block you just wrote."
)

_NO_FENCE_MSG = (
    "No REPL output: your message contained no ```repl block, so nothing ran. "
    "Put the code you want executed inside a ```repl fenced block."
)


def _missing_code_block_message(assistant_text: str) -> list[dict[str, str]]:
    """Feedback for a turn that produced no runnable code block."""
    opens = assistant_text.count("```repl")
    closes = assistant_text.count("```") - opens
    body = _UNCLOSED_FENCE_MSG if opens and closes < opens else _NO_FENCE_MSG
    return [{"role": "user", "content": body}]


def _format_repl_outputs(
    outputs: list[dict[str, Any]], max_chars: int | None = _MAX_REPL_OUTPUT_CHARS
) -> list[dict[str, str]]:
    """Format an assistant turn's REPL outputs as a single user message."""
    if not outputs:
        return []
    parts: list[str] = []
    multi = len(outputs) > 1
    for i, o in enumerate(outputs):
        body = _format_one(o, max_chars)
        header = f"REPL output (block {i + 1}):" if multi else "REPL output:"
        parts.append(f"{header}\n{body}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def _format_one(o: dict[str, Any], max_chars: int | None = _MAX_REPL_OUTPUT_CHARS) -> str:
    # add back what the worker clipped, so the "+ [N chars...]" count is right
    cut_out = int(o.get("stdout_truncated_chars") or 0)
    cut_err = int(o.get("stderr_truncated_chars") or 0)
    stdout, stderr = o.get("stdout") or "", o.get("stderr") or ""
    if max_chars is None:
        if cut_out:
            stdout += f"... + [{cut_out} chars...]"
        if cut_err:
            stderr += f"... + [{cut_err} chars...]"
    parts: list[str] = []
    if stdout:
        parts.append(f"\n{stdout}")
    if stderr:
        parts.append(f"\n{stderr}")
    if o.get("locals_keys"):
        parts.append(f"REPL variables: {list(o['locals_keys'])}\n")
    body = "\n\n".join(parts) if parts else "No output"
    if max_chars is not None:
        full_len = len(body) + cut_out + cut_err
        if full_len > max_chars:
            body = body[:max_chars] + f"... + [{full_len - max_chars} chars...]"
    return body
