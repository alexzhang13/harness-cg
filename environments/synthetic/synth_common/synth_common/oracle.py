"""Base class for a synthetic env's fake sub-LM (the thing that answers `llm_query` / `llm_query_batched`)."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

_TOKEN_CHARS = 4.0


@dataclass
class ReadModel:
    budget_chars: int = 60_000
    reliable_items: int = 30
    miss_rate: float = 0.5
    base_err: float = 0.02
    load_err: float = 0.10
    agg_capacity: int = 8
    chatty: float = 0.2
    min_cov: float = 0.6


def stable_hash(*parts: Any) -> int:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest()[:8], "big")


def rng_for(*parts: Any) -> random.Random:
    return random.Random(stable_hash(*parts))


def overlong_error(prompt: str, budget_chars: int) -> str:
    """Same wording as `rlm_train.worker._llm_query` when the proxy reports a too-large prompt."""
    n_tok = int(len(prompt) / _TOKEN_CHARS)
    max_tok = int(budget_chars / _TOKEN_CHARS)
    return (
        "Error: sub-LLM prompt exceeded the endpoint's request-size limit "
        f"(prompt was {len(prompt):,} chars). Shorten or chunk the prompt — summarize the context first, or split it "
        "across multiple llm_query calls. Underlying error: This model's maximum context length is "
        f"{max_tok} tokens. However, your request has {n_tok} input tokens. Please reduce the length of the input."
    )


_PREAMBLES = [
    "Sure — here is what I found.",
    "Here are the results:",
    "Okay, going through the text you provided:",
    "Based on the passage(s) above:",
]
_CLOSINGS = [
    "Let me know if you need anything else.",
    "Hope this helps.",
    "(Only the material you provided was used.)",
]


def wants_json(instruction: str) -> bool:
    return bool(re.search(r"\bjson\b", instruction, re.I))


class FakeSubLM:
    """Subclass and implement ``respond``. Pass the instance as ``sub_llm_fn`` and ``.batched`` as ``sub_llm_fn_batched``."""

    def __init__(self, read: ReadModel | None = None):
        self.read = read or ReadModel()

    def __call__(self, prompt: str, state: Any) -> str:
        prompt = "" if prompt is None else str(prompt)
        st = _acct(state)
        st["calls"] += 1
        st["prompt_chars"] += len(prompt)
        st["max_prompt_chars"] = max(st["max_prompt_chars"], len(prompt))
        if len(prompt) > self.read.budget_chars:
            st["overlong"] += 1
            return overlong_error(prompt, self.read.budget_chars)
        from .envkit import (
            oracle_view,
        )

        hidden = oracle_view(state)
        if hidden is None:
            return "Error: no synthetic payload for this rollout."
        log_call(st, state, prompt, hidden.get("_index"))
        try:
            out = self.respond(prompt, hidden, st)
        except Exception as e:  # noqa: BLE001  an oracle bug must not kill the rollout
            st["oracle_errors"] += 1
            return f"I could not process that request ({type(e).__name__})."
        st["response_chars"] += len(out)
        return out

    def batched(self, prompts: list[str], state: Any) -> list[str]:
        st = _acct(state)
        st["batches"] += 1
        st["max_batch"] = max(st["max_batch"], len(prompts))
        return [self(p, state) for p in prompts]

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:  # pragma: no cover
        raise NotImplementedError

    def given(self, prompt: str, hidden: dict) -> list[str]:
        """Item ids present (coverage >= min_cov) in prompt order."""
        return hidden["_index"].present(prompt, self.read.min_cov)

    def instruction(self, prompt: str, hidden: dict) -> str:
        return hidden["_index"].residual(prompt)

    def load(self, prompt: str) -> float:
        return min(1.0, len(prompt) / self.read.budget_chars)

    def read_items(self, prompt: str, ids: list[str], st: dict, salt: Any = "") -> tuple[list[str], list[str]]:
        """Split ``ids`` (in prompt order) into (read, skipped) under the load model. Records reads in ``st``."""
        n = len(ids)
        read, skipped = [], []
        k = self.read.reliable_items
        for pos, iid in enumerate(ids):
            if n > k and pos >= k // 2:
                over = (n - k) / n
                p = min(0.95, self.read.miss_rate * over * (0.5 + pos / n))
                if rng_for(salt, prompt, iid, "miss").random() < p:
                    skipped.append(iid)
                    continue
            read.append(iid)
        reads = st.setdefault("read_ids", {})
        for iid in read:
            reads[iid] = reads.get(iid, 0) + 1
        st["items_read"] = len(reads)
        st["items_skipped"] += len(skipped)
        return read, skipped

    def err_prob(self, prompt: str) -> float:
        return self.read.base_err + self.read.load_err * self.load(prompt)

    def flip(self, prompt: str, iid: str, salt: Any = "") -> bool:
        """True when this item's judgement in this prompt comes back wrong."""
        return rng_for(salt, prompt, iid, "err").random() < self.err_prob(prompt)

    def chat(self, prompt: str, body: str, salt: Any = "") -> str:
        """Sometimes add a chatty preamble / closing, like real sub-LMs."""
        r = rng_for(salt, prompt, "chat")
        if r.random() < self.read.chatty:
            body = r.choice(_PREAMBLES) + "\n" + body
        if r.random() < self.read.chatty / 2:
            body = body + "\n" + r.choice(_CLOSINGS)
        return body

    def render(self, prompt: str, instruction: str, pairs: list[tuple[str, Any]], salt: Any = "") -> str:
        """Render (item id, value) pairs as JSON if the instruction asks for JSON, else one ``id: value`` line each."""
        if wants_json(instruction):
            body = json.dumps({k: v for k, v in pairs}, ensure_ascii=False)
            if rng_for(salt, prompt, "fence").random() < self.read.chatty:
                body = "```json\n" + body + "\n```"
            return self.chat(prompt, body, salt)
        lines = [f"{k}: {v if not isinstance(v, (list, dict)) else json.dumps(v, ensure_ascii=False)}" for k, v in pairs]
        return self.chat(prompt, "\n".join(lines), salt)


def _acct(state: Any) -> dict:
    if state is None:
        state = {}
    st = state.get("synth")
    if st is None:
        st = {"calls": 0, "batches": 0, "max_batch": 0, "prompt_chars": 0, "response_chars": 0,
              "max_prompt_chars": 0, "overlong": 0, "refusals": 0, "oracle_errors": 0,
              "items_read": 0, "items_skipped": 0, "agg_requests": 0}
        state["synth"] = st
    return st


def log_call(st: dict, state: Any, prompt: str, index: Any) -> None:
    """Record a sub-call (turn, carries context?, instruction text) for the delegation / re-verify checks."""
    log = st.setdefault("call_log", [])
    if len(log) >= 1000:
        return
    try:
        cov = index.coverage(prompt) if index is not None else {}
        ctx = sum(round(c * index.n_segs.get(i, 0)) for i, c in cov.items()) >= 2
        residual = index.residual(prompt)[:600] if index is not None else prompt[:600]
    except Exception:  # noqa: BLE001
        ctx, residual = False, ""
    step = int((state or {}).get("rlm_n_processed") or 0) + 1
    log.append({"step": step, "ctx": ctx, "residual": residual})


def refuse_aggregate(st: dict) -> str:
    st["refusals"] += 1
    return ("I can't determine that from the material provided — I only see the text you included in this message. "
            "Send me the relevant passages (and only those) and ask about each one.")


def mentions_any(text: str, words: Iterable[str]) -> list[str]:
    low = text.lower()
    return [w for w in words if w.lower() in low]


__all__ = [
    "FakeSubLM", "ReadModel", "mentions_any", "overlong_error", "refuse_aggregate", "rng_for", "stable_hash",
    "wants_json",
]
