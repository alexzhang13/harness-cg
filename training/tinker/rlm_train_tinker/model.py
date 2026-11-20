"""Base model, tokenizer and chat renderer (prime-rl's `renderers` package).

Tinker does not serve Qwen3-30B-A3B-Instruct-2507, so the default is Qwen3.6-35B-A3B with thinking off (--base-model
overrides it).
"""

from __future__ import annotations

import functools
import inspect

# prime-rl `[model] name` -> Tinker base model.
TINKER_BASE_MODELS = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "Qwen/Qwen3.6-35B-A3B:peft:262144",
}

# generation_config of Qwen3-30B-A3B-Instruct-2507, which vLLM used for unset sampling fields (inline evals,
# sub-calls). Applied explicitly so the substitute model samples the same way.
PAPER_GENERATION_CONFIG = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def tinker_base_model(config_model: str, override: str | None = None) -> str:
    if override:
        return override
    if config_model not in TINKER_BASE_MODELS:
        raise KeyError(f"no Tinker base model for {config_model!r}; pass --base-model")
    return TINKER_BASE_MODELS[config_model]


def tokenizer_id(base_model: str) -> str:
    """Strip the :peft:<n> suffix."""
    return base_model.split(":")[0]


def max_context(base_model: str) -> int:
    """Context limit from the :peft:<n> suffix (default 65,536)."""
    parts = base_model.split(":")
    return int(parts[-1]) if len(parts) == 3 and parts[1] == "peft" else 65536


@functools.lru_cache(maxsize=4)
def get_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


@functools.lru_cache(maxsize=4)
def get_renderer(name: str, enable_thinking: bool = False):
    """prime-rl renderer, thinking disabled."""
    import renderers

    tok = get_tokenizer(name)
    r = renderers.create_renderer(tok, "auto")
    cls = type(r)
    if "enable_thinking" in inspect.signature(cls.__init__).parameters:
        r = cls(tok, enable_thinking=enable_thinking)
    return r
