"""Task identity helpers: ``task_sha1`` (the content hash rows are keyed by) and ``metadata_text``."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _info_dict(info: Any) -> dict:
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except json.JSONDecodeError:
            return {}
    return info if isinstance(info, dict) else {}


def task_sha1(info: Any) -> str:
    """Content hash of a task: root prompt + context (stable across subsampling, unlike example_id)."""
    d = _info_dict(info)
    key = {"root_prompt": d.get("root_prompt"), "context": d.get("context")}
    if key["root_prompt"] is None and key["context"] is None:
        key = {k: d.get(k) for k in sorted(d) if k.startswith("synth_")}
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(blob.encode("utf-8", "surrogatepass")).hexdigest()


def metadata_text(info: Any, sub_llm: bool = True) -> str:
    """The metadata user message RLMTrainEnv renders for this dataset row (messages[1] of a rollout)."""
    from rlm_train import doc_store
    from rlm_train.prompts import build_rlm_system_prompt
    d = _info_dict(info)
    if d.get("context") is None:
        return ""
    return build_rlm_system_prompt(system_prompt="", query_metadata=doc_store.query_metadata(d.get("context")),
                                   root_prompt=d.get("root_prompt"), sub_llm=sub_llm)[1]["content"]


def _ctx_str(info: Any) -> str:
    c = _info_dict(info).get("context")
    return c if isinstance(c, str) else json.dumps(c, ensure_ascii=False, default=str)


BEFORE, AFTER = "before", "after"


def leak_needles(loc: dict) -> list[str]:
    """Fragments of the note that must not reappear in the saved sample."""
    out = []
    for k in ("what_went_wrong", "what_to_do_instead", "evidence"):
        s = (loc.get(k) or "").strip()
        if len(s) >= 40:
            out.append(s[:60])
    return out

