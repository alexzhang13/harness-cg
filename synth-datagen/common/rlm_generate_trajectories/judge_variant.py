"""Select the judge-prompt variant: ``RLM_JUDGE_PROBE_FROM_PROMPT_FLAG=1`` adds the ``probe_from_prompt_ok`` flag
and its rubric paragraph (used for the LongReason streams, not BrowseComp-Plus)."""

from __future__ import annotations

import os
from pathlib import Path

PROBE_FROM_PROMPT_FLAG = os.environ.get("RLM_JUDGE_PROBE_FROM_PROMPT_FLAG") == "1"
PROBE_FROM_PROMPT_RUBRIC_LINE = "   If STREAM FLAGS say probe_from_prompt_ok (the task prompt itself states the layout of `context`, e.g. '---\\ndoc_id: <id>\\n---\\n<text>' or 'a -> b' edge lines), trusting that stated layout and confirming it with a short measured print (counts, one record) is a correct probe -> true; a description sub-call is still fine. Markers the prompt states count as evidence for parse_from_evidence.\n"


def rubric_text(path: Path) -> str:
    """A rubric file's text, with the probe_from_prompt_ok paragraph when the flag variant is selected."""
    text = path.read_text()
    if PROBE_FROM_PROMPT_FLAG and path.name == "30_probing_and_parsing.md":
        lines = text.splitlines(keepends=True)
        text = "".join(lines[:9] + [PROBE_FROM_PROMPT_RUBRIC_LINE] + lines[9:])
    return text
