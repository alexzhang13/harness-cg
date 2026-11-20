"""Run a synthetic env with a scripted root instead of an LM (used to test envs)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from verifiers import OpenAIChatCompletionsClient

Script = Sequence[str] | Callable[[list], str]


def repl(code: str, prose: str = "") -> str:
    """Wrap code as an assistant turn."""
    return (prose + "\n" if prose else "") + "```repl\n" + code.strip("\n") + "\n```"


class ScriptedClient(OpenAIChatCompletionsClient):
    """A verifiers client whose 'model' replays a fixed script. One instance per rollout."""

    def __init__(self, script: Script):
        super().__init__(AsyncOpenAI(api_key="scripted", base_url="http://127.0.0.1:9/v1"))
        self._script = script
        self._turn = 0

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):  # noqa: ANN001
        if callable(self._script):
            text = self._script(prompt)
        else:
            text = self._script[min(self._turn, len(self._script) - 1)]
        self._turn += 1
        return ChatCompletion(
            id=f"scripted-{self._turn}", object="chat.completion", created=int(time.time()), model=model,
            choices=[Choice(index=0, finish_reason="stop",
                            message=ChatCompletionMessage(role="assistant", content=text))],
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )


def _is_root_prompt(prompt: Any) -> bool:
    """Root turns carry the RLM system prompt; sub-calls are a bare user message."""
    for m in prompt or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "system":
            return True
    return False


class HybridClient(ScriptedClient):
    """Scripted root with a real sub-LM, to check a reference program on a real env."""

    def __init__(self, script: Script, sub_client: Any):
        super().__init__(script)
        self._sub = sub_client

    async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):  # noqa: ANN001
        if _is_root_prompt(prompt):
            return await super().get_response(prompt, model, sampling_args, tools, **kwargs)
        return await self._sub.get_response(prompt, model, sampling_args, tools, **kwargs)


async def run_rollout(env: Any, row_idx: int, client: Any, model: str = "scripted",
                      sampling_args: dict | None = None) -> dict:
    row = dict(env.dataset[row_idx])
    return await env._run_rollout_state(input=row, client=client, model=model,
                                        sampling_args=sampling_args or {"max_tokens": 4096})


def run_script(env: Any, script: Script, rows: Sequence[int] = (0,)) -> list[dict]:
    """Run ``script`` on each row; returns the final states (reward in state['reward'], metrics in state['metrics'])."""

    async def go() -> list[dict]:
        out = []
        for i in rows:
            out.append(await run_rollout(env, i, ScriptedClient(script)))
        return out

    try:
        return asyncio.run(go())
    finally:
        try:
            asyncio.run(env.teardown_rlm())
        except Exception:  # noqa: BLE001
            pass


def run_program_on_real(env: Any, script: Script, rows: Sequence[int], sub_model: str, concurrency: int = 4) -> list[dict]:
    """Run a scripted reference program on a real env with a real sub-LM (needs ``PRIME_API_KEY``)."""
    from .probe import prime_client

    sub = prime_client()

    async def go() -> list[dict]:
        sem = asyncio.Semaphore(concurrency)

        async def one(i: int) -> dict:
            async with sem:
                return await run_rollout(env, i, HybridClient(script, sub), model=sub_model)

        return list(await asyncio.gather(*(one(i) for i in rows)))

    try:
        return asyncio.run(go())
    finally:
        try:
            asyncio.run(env.teardown_rlm())
        except Exception:  # noqa: BLE001
            pass


def mean_reward(states: list[dict]) -> float:
    return sum(float(s.get("reward") or 0.0) for s in states) / max(1, len(states))


__all__ = ["HybridClient", "ScriptedClient", "mean_reward", "repl", "run_program_on_real", "run_rollout", "run_script"]
