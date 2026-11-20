"""Shared LM-as-judge endpoint / client resolution."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

PRIME_BASE_URL = "https://api.pinference.ai/api/v1"
OPENAI_DIRECT_BASE_URL = "https://api.openai.com/v1"
TEAM_HEADER = "X-Prime-Team-ID"

class JudgeEndpointError(RuntimeError):
    """No usable judge endpoint."""


def _is_prime(base_url: str | None) -> bool:
    return bool(base_url) and "pinference" in str(base_url)


def _is_openai_direct(base_url: str | None) -> bool:
    """True for the OpenAI API (base_url None is its default)."""
    return base_url is None or "api.openai.com" in str(base_url)


def _env_first(prefixes: Sequence[str], suffix: str) -> str | None:
    for prefix in prefixes:
        value = os.environ.get(f"{prefix}_{suffix}")
        if value:
            return value
    return None


def _adjust_model_prefix(model: str | None, base_url: str | None) -> str | None:
    if not model:
        return model
    model = str(model)
    if _is_prime(base_url):
        return model if "/" in model else f"openai/{model}"
    if _is_openai_direct(base_url) and model.startswith("openai/"):
        return model.split("/", 1)[1]
    return model


@dataclass(frozen=True)
class JudgeEndpoint:
    """A resolved judge endpoint."""

    base_url: str | None
    api_key: str
    model: str | None
    team_id: str | None = None
    source: str = "unresolved"

    @property
    def is_prime(self) -> bool:
        return _is_prime(self.base_url)

    @property
    def default_headers(self) -> dict[str, str] | None:
        """default_headers for AsyncOpenAI."""
        return {TEAM_HEADER: self.team_id} if self.team_id else None

    @property
    def headers(self) -> dict[str, str]:
        """Headers for a raw httpx request."""
        out = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.team_id:
            out[TEAM_HEADER] = self.team_id
        return out

    @property
    def client_base_url(self) -> str:
        """base_url for AsyncOpenAI."""
        return self.base_url or OPENAI_DIRECT_BASE_URL

    @property
    def resolved(self) -> bool:
        return bool(self.base_url) and bool(self.model)


def resolve_judge_endpoint(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    default_model: str | None = None,
    env_prefixes: Sequence[str] = ("RLM_JUDGE",),
    state: dict | None = None,
    provider_fallback: bool = True,
    api_key_default: str | None = None,
) -> JudgeEndpoint:
    """Resolve the judge endpoint from arguments, RLM_JUDGE_* variables, state, then provider keys."""
    prefixes = tuple(env_prefixes)
    source = "explicit" if (base_url or api_key or model) else "unresolved"

    if base_url is None:
        env_base = _env_first(prefixes, "BASE_URL")
        if env_base:
            base_url, source = env_base, "env"
    if not api_key:
        env_key = _env_first(prefixes, "API_KEY")
        if env_key:
            api_key = env_key
            if source == "unresolved":
                source = "env"
    if model is None:
        env_model = _env_first(prefixes, "MODEL")
        if env_model:
            model = env_model
            if source == "unresolved":
                source = "env"
        else:
            model = default_model

    if base_url is None and not api_key and provider_fallback:
        openai_key = os.environ.get("OPENAI_API_KEY")
        prime_key = os.environ.get("PRIME_API_KEY")
        if openai_key:
            api_key, base_url, source = openai_key, None, "OPENAI_API_KEY"
        elif prime_key:
            api_key, base_url, source = prime_key, PRIME_BASE_URL, "PRIME_API_KEY"

    if state is not None and (base_url is None or model is None):
        client = state.get("client")
        if client is not None:
            try:
                if base_url is None:
                    base_url = str(client._client.base_url).rstrip("/")  # type: ignore[attr-defined]
                    if not api_key:
                        api_key = client._client.api_key  # type: ignore[attr-defined]
                    source = "policy_client"
                if model is None:
                    model = state.get("model")
            except Exception:  # noqa: BLE001
                pass

    if not api_key:
        if _is_prime(base_url):
            api_key = os.environ.get("PRIME_API_KEY")
        elif _is_openai_direct(base_url):
            api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = api_key_default or "EMPTY"

    model = _adjust_model_prefix(model, base_url)

    # team header only if PRIME_API_TEAM is set
    team_id = (os.environ.get("PRIME_API_TEAM") or None) if _is_prime(base_url) else None

    return JudgeEndpoint(
        base_url=base_url,
        api_key=api_key,
        model=model,
        team_id=team_id,
        source=source,
    )


def build_judge_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    default_model: str | None = None,
    env_prefixes: Sequence[str] = ("RLM_JUDGE",),
    state: dict | None = None,
    provider_fallback: bool = True,
    api_key_default: str | None = None,
) -> tuple[Any, str]:
    """(AsyncOpenAI client, model name) for the judge."""
    endpoint = resolve_judge_endpoint(
        api_key=api_key,
        base_url=base_url,
        model=model,
        default_model=default_model,
        env_prefixes=env_prefixes,
        state=state,
        provider_fallback=provider_fallback,
        api_key_default=api_key_default,
    )
    if endpoint.base_url is None and endpoint.api_key in ("EMPTY", "dummy"):
        raise JudgeEndpointError(
            "no judge endpoint resolved: set "
            f"{'/'.join(f'{p}_API_KEY' for p in env_prefixes)} (+ _BASE_URL, "
            "_MODEL), OPENAI_API_KEY, or PRIME_API_KEY — or pass an explicit "
            "judge client. OPENAI_BASE_URL is intentionally ignored."
        )
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=endpoint.api_key,
        base_url=endpoint.client_base_url,
        default_headers=endpoint.default_headers,
    )
    return client, str(endpoint.model) if endpoint.model else ""


__all__ = [
    "OPENAI_DIRECT_BASE_URL",
    "PRIME_BASE_URL",
    "TEAM_HEADER",
    "JudgeEndpoint",
    "JudgeEndpointError",
    "build_judge_client",
    "resolve_judge_endpoint",
]
