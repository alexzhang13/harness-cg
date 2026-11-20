"""Per-turn SFT loss mask for prime-rl's SFT trainer (installed by sitecustomize when RLM_SFT_LAST_TURN_ONLY=1).

The loss covers only the final assistant message; a sample whose history render is not a token prefix of the full
render is skipped.
"""
from __future__ import annotations


def install() -> None:
    from prime_rl.trainer.sft import data as D
    from prime_rl.utils import chat_template as C

    def last_turn_mask(tokenizer, messages, *, role_to_mask, tools=None, chat_template_kwargs=None,
                       collapse_consecutive_tool_messages=False, processor=None):
        if not messages or messages[-1].get("role") != "assistant":
            raise C.IncrementalTokenizationError("per-turn sample must end with an assistant message")
        kw = dict(tools=tools, chat_template_kwargs=chat_template_kwargs, processor=processor)
        prefix = C.render_messages(tokenizer, messages[:-1], add_generation_prompt=True, **kw)
        full = C.render_messages(tokenizer, messages, add_generation_prompt=False, **kw)
        if full[: len(prefix)] != prefix:
            raise C.IncrementalTokenizationError("history+generation prompt is not a token prefix of the full render")
        return full, [False] * len(prefix) + [True] * (len(full) - len(prefix))

    D.build_incremental_token_mask = last_turn_mask
    print("[sft] per-turn SFT: loss on the last assistant turn only", flush=True)
