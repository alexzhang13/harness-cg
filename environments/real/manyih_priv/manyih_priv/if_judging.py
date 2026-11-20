"""Port of ManyIH-Bench's instruction-following judging."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from rlm_train.judge_client import resolve_judge_endpoint

logger = logging.getLogger(__name__)

JudgeFn = Callable[[str], Awaitable[str]]


def execute_function(function: str, response: str) -> tuple[Any, str | None]:
    """Execute a benchmark-authored ``check_following(response)`` snippet."""
    try:
        if not response:
            return None, "Empty response"

        local_vars: dict[str, Any] = {"response": response}

        import_statements = re.findall(
            r"^\s*(import\s+\S+|from\s+\S+\s+import\s+\S+)", function, re.MULTILINE
        )
        for statement in import_statements:
            exec(statement, globals())  # noqa: S102

        exec(function, globals(), local_vars)  # noqa: S102

        if "check_following" in local_vars:
            result = local_vars["check_following"](local_vars["response"])
            return result, None
        return None, "check_following not defined"
    except Exception as e:  # noqa: BLE001
        error_msg = f"Error executing function: {e}, the function is: {function}"
        logger.error(error_msg)
        return None, error_msg


def code_checker(function: str, response: str) -> tuple[Any, str | None]:
    return execute_function(function, response)


async def llm_checker(judge: JudgeFn, response: str, prompt: str) -> tuple[str | None, str | None]:
    """Render the per-constraint judge prompt and return the judge's verdict text."""
    try:
        if "{response}" not in prompt:
            prompt += "\n\nHere is model response: {response}"
        prompt = prompt.replace("{response}", response)
        result = await judge(prompt)
        if result and "</think>" in result:
            result = result.split("</think>")[1].strip()
        return result, None
    except Exception as e:  # noqa: BLE001
        logger.error(f"LLM checker error: {e}")
        return None, str(e)


async def evaluate_constraints(
    constraints: list[dict[str, Any]],
    response: str,
    judge: JudgeFn,
) -> list[Any]:
    """Score each active constraint; returns a parallel list of True/False/None."""
    scores: list[Any] = []
    for item in constraints:
        cur: Any = response
        skipped = False

        for e in item.get("evaluation") or []:
            etype = e.get("type")
            if etype == "llm_conditional_check":
                condition_response, _err = await llm_checker(judge, response, e["exec"])
                if condition_response and "YES" in condition_response:
                    continue
                cur = None
                skipped = True
                break
            if etype == "llm":
                cur, _err = await llm_checker(judge, response, e["exec"])
                if not cur:
                    cur = None
                    break
            elif etype == "code":
                cur, err = await asyncio.to_thread(code_checker, e["exec"], cur)
                if err:
                    logger.error(f"Code checker error: {err}")
                    cur = None

        if skipped:
            scores.append(None)
            continue

        if isinstance(cur, str):
            if "YES" in cur:
                cur = True
            elif "NO" in cur:
                cur = False
            else:
                logger.warning(f"Unexpected response format: {cur}")
                cur = None

        scores.append(cur)
    return scores


def csr_isr(scores: list[Any]) -> tuple[float, float]:
    """Single-sample CSR and ISR from a list of True/False/None constraint scores."""
    non_null = [s for s in scores if s is not None]
    if not non_null:
        return 0.0, 0.0
    csr = sum(1 for s in non_null if s is True) / len(non_null)
    isr = 1.0 if all(non_null) else 0.0
    return csr, isr


def make_judge(
    state: dict[str, Any],
    *,
    timeout_s: float = 60.0,
    max_tokens: int = 256,
    retries: int = 3,
) -> JudgeFn:
    """Build an async judge callable bound to the resolved endpoint."""
    endpoint = resolve_judge_endpoint(
        state=state,
        provider_fallback=False,
        api_key_default="dummy",
        strict=False,
    )
    base_url, model = endpoint.base_url, endpoint.model

    async def judge(prompt: str) -> str:
        if not base_url or not model:
            return ""
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        if str(model).startswith("openai/gpt-5"):
            body["reasoning_effort"] = "minimal"
        headers = endpoint.headers
        for attempt in range(retries + 1):
            try:
                async with httpx.AsyncClient() as http:
                    resp = await http.post(
                        f"{base_url}/chat/completions",
                        json=body,
                        headers=headers,
                        timeout=timeout_s,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    return (data["choices"][0]["message"]["content"] or "").strip()
            except Exception as e:  # noqa: BLE001
                if attempt < retries:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                logger.error(f"Judge call failed after {retries + 1} attempts: {type(e).__name__}: {e}")
                return ""
        return ""

    return judge


__all__ = [
    "JudgeFn",
    "code_checker",
    "csr_isr",
    "evaluate_constraints",
    "execute_function",
    "llm_checker",
    "make_judge",
]
