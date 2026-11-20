"""A verifiers Client that samples from the current Tinker policy.

tito=True reproduces prime-rl's RendererClient (RLM, LM + REPL, Transformer): a later turn reuses the previous
turn's tokens when its messages extend them, and sampled tokens and logprobs are returned. tito=False reproduces
vLLM chat completions (Pi). Prompt and completion length limits follow vLLM; unset sampling fields take the model's
generation_config. Logprobs are at T = 1, as in Tinker's training forward pass.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any

import tinker
from renderers import ToolCallParseStatus
from verifiers.clients import renderer_client as _rc
from verifiers.clients.client import Client
from verifiers.errors import OverlongPromptError

from .model import PAPER_GENERATION_CONFIG

logger = logging.getLogger(__name__)

if os.environ.get("RLM_MULTI_USER_TAIL") == "1":
    # as verifiers patch 0001: several trailing user messages still extend the previous turn's tokens
    def _multi_user_tail(messages: list) -> bool:
        if not messages:
            return False
        roles = [_rc._message_role(m) for m in messages]
        return roles[-1] == "user" and all(r in ("tool", "user") for r in roles[:-1])

    _rc._is_valid_incremental_tail = _multi_user_tail

_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg)


class Policy:
    """Current sampling weights and their version (optimizer steps). update() swaps them mid-rollout, like a live
    LoRA reload in vLLM."""

    def __init__(self, sampling_client: Any, version: int = 0, max_concurrency: int = 1024):
        self.sampling_client = sampling_client
        self.version = version
        self._sem = asyncio.Semaphore(max_concurrency)

    def update(self, sampling_client: Any, version: int) -> None:
        self.sampling_client = sampling_client
        self.version = version

    async def sample(self, prompt_ids: list[int], *, max_tokens: int, temperature: float, top_p: float,
                     top_k: int, stop: list[int]) -> tuple[list[int], list[float]]:
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k,
                                       stop=list(stop))
        client = self.sampling_client
        async with self._sem:
            resp = await client.sample(prompt=tinker.ModelInput.from_ints(list(prompt_ids)), num_samples=1,
                                       sampling_params=params)
        seq = resp.sequences[0]
        tokens = list(seq.tokens)
        logprobs = [float(x) for x in (seq.logprobs or [])]
        if len(logprobs) != len(tokens):
            raise RuntimeError(f"Tinker returned {len(logprobs)} logprobs for {len(tokens)} tokens")
        return tokens, logprobs


def resolve_sampling(sampling_args: dict[str, Any] | None, *, tito: bool,
                     generation_config: dict[str, Any] = PAPER_GENERATION_CONFIG) -> dict[str, Any]:
    """(max_tokens, temperature, top_p, top_k) as vLLM would apply them."""
    args = dict(sampling_args or {})
    extra = dict(args.pop("extra_body", None) or {})
    if tito:  # top-level fields override extra_body
        params = {**extra, **{k: v for k, v in args.items() if v is not None}}
    else:  # extra_body is merged last
        params = {**{k: v for k, v in args.items() if v is not None}, **extra}
    max_tokens = params.get("max_tokens") or params.get("max_completion_tokens")
    for key, neutral in (("min_p", 0.0), ("repetition_penalty", 1.0), ("min_tokens", 0)):
        if params.get(key) not in (None, neutral):
            _warn_once(key, f"sampling field {key}={params[key]} is not supported by Tinker and is ignored")
    return {
        "max_tokens": None if max_tokens is None else int(max_tokens),
        "temperature": float(params.get("temperature", generation_config.get("temperature", 1.0))),
        "top_p": float(params.get("top_p", generation_config.get("top_p", 1.0))),
        "top_k": int(params.get("top_k", generation_config.get("top_k", -1))),
    }


class TinkerClient(_rc.RendererClient):
    """RendererClient whose generation call samples from Tinker."""

    def __init__(self, policy: Policy, renderer: Any, *, max_model_len: int, max_context: int, tito: bool = True):
        Client.__init__(self, policy)
        self._renderer = renderer
        self._pool_size = 1
        self.policy = policy
        self.max_model_len = int(max_model_len)
        self.max_context = int(max_context)
        self.tito = tito
        self._stop_ids = list(renderer.get_stop_token_ids())

    def setup_client(self, config):  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:
        return None

    async def get_native_response(self, prompt, model: str, sampling_args, tools=None, **kwargs) -> dict[str, Any]:
        sp = resolve_sampling(sampling_args, tito=self.tito)
        prompt_ids = None
        if self.tito:
            bridged = await _rc._get_incremental_prompt_ids(renderer=self._renderer, prompt=prompt,
                                                            state=kwargs.get("state"), tools=tools)
            if bridged is not None:
                prompt_ids = list(bridged.token_ids)
        if prompt_ids is None:
            prompt_ids = list(self._renderer.render(prompt, tools=tools, add_generation_prompt=True).token_ids)
        limit = min(self.max_model_len, self.max_context)
        budget = limit - len(prompt_ids)
        if budget <= 0:
            raise OverlongPromptError(f"prompt of {len(prompt_ids)} tokens exceeds max_model_len={limit}")
        max_tokens = budget if sp["max_tokens"] is None else min(sp["max_tokens"], budget)
        completion_ids, logprobs = await self.policy.sample(
            prompt_ids, max_tokens=max_tokens, temperature=sp["temperature"], top_p=sp["top_p"], top_k=sp["top_k"],
            stop=self._stop_ids)
        parsed = self._renderer.parse_response(completion_ids, tools=tools)
        finish = "stop" if completion_ids and completion_ids[-1] in self._stop_ids else "length"
        if finish == "stop" and any(tc.status == ToolCallParseStatus.OK for tc in parsed.tool_calls or []):
            finish = "tool_calls"
        return {
            "request_id": uuid.uuid4().hex,
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "completion_logprobs": logprobs,
            "content": parsed.content,
            "reasoning_content": parsed.reasoning_content,
            "tool_calls": parsed.tool_calls,
            "finish_reason": finish,
            "routed_experts": None,
            "multi_modal_data": None,
        }
