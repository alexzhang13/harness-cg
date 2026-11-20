"""Model-sampled input files that cannot be regenerated exactly, from HF a1zhang/harness-cg-frozen-inputs (pinned)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ID = "a1zhang/harness-cg-frozen-inputs"
REVISION = os.environ.get("HARNESS_CG_FROZEN_REVISION", "90141c80efffbdce45756dfec7dff1afec00770b")


def frozen_path(local: str | os.PathLike, path_in_repo: str) -> Path:
    """`local` if it exists, else download `path_in_repo`."""
    local = Path(local)
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO_ID, path_in_repo, repo_type="dataset", revision=REVISION))
