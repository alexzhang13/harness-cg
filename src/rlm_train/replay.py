"""Prefix replay: resume a recorded rollout at turn k.

info["rlm_replay"] = {"prefix", "expected_outputs", "sub_calls", "hint"}: the prefix turns are re-run in a fresh REPL
with recorded sub-call replies, then the model continues. state["rlm_replay_report"] says how well it matched.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

from rlm.utils.parsing import find_code_blocks

from rlm_train.proxy import _flatten_prompt, _maybe_await

REPLAY_SPEC_VERSION = 1
_NOT_RUN_PREFIX = "No REPL output: your message contained no"
_VOLATILE_RE = re.compile(r"rlm_worker_[A-Za-z0-9_]+\.py|rlm-repl-[A-Za-z0-9_-]+")


def outputs_equivalent(got: str, expected: str) -> bool:
    """Compare REPL replies, ignoring worker file and scratch dir names."""
    return got == expected or _VOLATILE_RE.sub("<w>", got) == _VOLATILE_RE.sub("<w>", expected)


def prompt_sha1(prompt: Any) -> str:
    text = prompt if isinstance(prompt, str) else str(prompt or "")
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


class ReplayStub:
    """Answers the prefix's sub-calls from the record."""

    def __init__(self, entries: list[dict] | None, orig_single=None, orig_batched=None):
        self.entries = [dict(e) for e in (entries or [])]
        self.used = [False] * len(self.entries)
        self.orig_single = orig_single
        self.orig_batched = orig_batched
        self.active = True
        self.leftover = False
        self.served = 0
        self.live = 0
        self.truncated_served = 0
        self.unmatched: list[str] = []

    def _matches(self, e: dict, p: str) -> bool:
        sha = e.get("prompt_sha1")
        if sha:
            return prompt_sha1(p) == sha
        rp = str(e.get("prompt") or "")
        n = e.get("prompt_chars")
        if n is not None and int(n) != len(p):
            return False
        return p[: len(rp)] == rp

    def _find(self, p: str, taken: set[int]) -> int | None:
        for i, e in enumerate(self.entries):
            if not self.used[i] and i not in taken and self._matches(e, p):
                return i
        return None

    def _take(self, i: int) -> str:
        self.used[i] = True
        self.served += 1
        e = self.entries[i]
        resp = str(e.get("response") or "")
        rc = e.get("response_chars")
        if rc is not None and int(rc) > len(resp):
            self.truncated_served += 1
        return resp

    async def single(self, prompt: Any, state_ref: Any = None):
        if self.active or self.leftover:
            p = _flatten_prompt(prompt)
            i = self._find(p, set())
            if i is not None:
                return self._take(i)
            if self.active:
                self.live += 1
                self.unmatched.append(p[:200])
        if self.orig_single is not None:
            return await _maybe_await(self.orig_single(prompt, state_ref))
        return None

    def _find_batch(self, ps: list[str]) -> list[int] | None:
        """A recorded batch whose prompts match `ps` position by position (needed when a batch repeats a prompt)."""
        groups: dict[Any, list[int]] = {}
        for i, e in enumerate(self.entries):
            if e.get("batch_id") is not None and not self.used[i]:
                groups.setdefault(e["batch_id"], []).append(i)
        for idx in groups.values():
            idx = sorted(idx, key=lambda i: int(self.entries[i].get("batch_index") or 0))
            if len(idx) == len(ps) and all(self._matches(self.entries[i], p) for i, p in zip(idx, ps, strict=True)):
                return idx
        return None

    async def batched(self, prompts: list, state_ref: Any = None):
        if self.active:
            idx_b = self._find_batch([_flatten_prompt(p) for p in prompts])
            if idx_b is not None:
                return [self._take(i) for i in idx_b]
            taken: set[int] = set()
            idx: list[int] = []
            for pr in prompts:
                i = self._find(_flatten_prompt(pr), taken)
                if i is None:
                    break
                taken.add(i)
                idx.append(i)
            if len(idx) == len(prompts):
                return [self._take(i) for i in idx]
            if self.orig_batched is None:
                return None
        elif self.leftover:
            ps = [_flatten_prompt(p) for p in prompts]
            taken: set[int] = set()
            idx = []
            for p in ps:
                i = self._find(p, taken)
                if i is None:
                    break
                taken.add(i)
                idx.append(i)
            if len(idx) == len(ps):
                return [self._take(i) for i in idx]
            if self.orig_batched is None:
                return None
        if self.orig_batched is not None:
            return await _maybe_await(self.orig_batched(prompts, state_ref))
        return None

    def report(self) -> dict[str, Any]:
        return {"served": self.served, "live_fallback": self.live, "truncated_served": self.truncated_served,
                "unused": sum(1 for u in self.used if not u), "unmatched_prompts": self.unmatched[:5]}


async def replay_prefix(env: Any, state: dict, spec: dict, stub: ReplayStub | None) -> None:
    """Run the prefix turns and install them as history."""
    from rlm_train.env import (
        _format_repl_outputs,
        _missing_code_block_message,
        _pack_exec,
        _strip_leaked_reasoning,
    )

    history: list = state["rlm_history"]
    backend = state["rlm_backend"]
    prefix = [str(t or "") for t in (spec.get("prefix") or [])]
    expected = list(spec.get("expected_outputs") or [])
    mismatches: list[dict[str, Any]] = []
    not_run: list[int] = []
    final: str | None = None
    done = 0
    for t, raw in enumerate(prefix, start=1):
        done = t
        state["rlm_replay_cur_step"] = t
        text = _strip_leaked_reasoning(raw)
        codes = list(find_code_blocks(text))
        exp_t = expected[t - 1] if t <= len(expected) else None
        if codes and isinstance(exp_t, str) and exp_t.startswith(_NOT_RUN_PREFIX):
            not_run.append(t)
            codes = []
        outputs: list[dict[str, Any]] = []
        for code in codes:
            try:
                result = await backend.execute(code)
            except Exception as e:  # noqa: BLE001
                outputs.append({"code": code, "stdout": "", "stderr": f"Worker error: {e}",
                                "final_answer": None, "locals_keys": []})
                continue
            outputs.append(_pack_exec(code, result))
            state["rlm_repl_calls"] = int(state.get("rlm_repl_calls") or 0) + 1
            if result.final_answer is not None and final is None:
                final = result.final_answer
        repl_msgs = _format_repl_outputs(outputs, env._max_repl_output_chars)
        if not codes and env._code_block_feedback:
            repl_msgs = _missing_code_block_message(text)
        if t in not_run:
            repl_msgs = [{"role": "user", "content": exp_t}]
        history.append({"role": "assistant", "content": text})
        history.extend(repl_msgs)
        if t <= len(expected) and expected[t - 1] is not None:
            got = repl_msgs[0]["content"] if repl_msgs else ""
            if not outputs_equivalent(got, expected[t - 1]):
                at = next((j for j, (a, b) in enumerate(zip(got, expected[t - 1], strict=False)) if a != b),
                          min(len(got), len(expected[t - 1])))
                g_n, e_n = _VOLATILE_RE.sub("<w>", got), _VOLATILE_RE.sub("<w>", expected[t - 1])
                kind = "order_only" if sorted(g_n) == sorted(e_n) else "content"
                mismatches.append({"step": t, "kind": kind, "at_char": at, "got": got[max(0, at - 80): at + 160],
                                   "expected": expected[t - 1][max(0, at - 80): at + 160],
                                   "got_chars": len(got), "expected_chars": len(expected[t - 1])})
        if final is not None:
            state["rlm_final_answer"] = final
            state["final_answer"] = final
            state["final_env_response"] = repl_msgs
            state["rlm_final_repl_outputs"] = repl_msgs
            break
        history.append(env._build_user_iter(t))
    state.pop("rlm_replay_cur_step", None)
    if stub is not None:
        stub.active = False
        stub.leftover = True
    hint = spec.get("hint")
    if hint and final is None:
        history.append({"role": "user", "content": str(hint)})
    off = done
    state["rlm_replay_offset"] = off
    state["rlm_iterations"] = off
    state["rlm_replay_report"] = {
        "version": REPLAY_SPEC_VERSION, "from_step": off + 1, "prefix_turns": off,
        "outputs_checked": min(len(expected), off), "output_mismatches": mismatches, "not_run_turns": not_run,
        "deterministic": not mismatches, "final_answer_in_prefix": final is not None,
        "hint_inserted": bool(hint) and final is None, **(stub.report() if stub is not None else {}),
    }
