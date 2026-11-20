"""Wikipedia corpus resolution for FanOutQA."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_SETUP_HINT = """\
FanOutQA needs the upstream Wikipedia cache (English Wikipedia @ 2023-11-20).
One-time setup (~1.5 GB download, 8.8 GB on disk, <2 min on a fast link):

    mkdir -p ~/.cache/fanoutqa && cd ~/.cache/fanoutqa
    curl -L -O https://datasets.mechanus.zhu.codes/fanoutqa/wikicache.tar.gz
    tar xzf wikicache.tar.gz          # -> ~/.cache/fanoutqa/wikicache/
    uv run python -m fanoutqa_real.prepare      # HTML -> .md for every evidence page

Point ``FANOUTQA_WIKICACHE_DIR`` at the ``wikicache`` directory if you put it
somewhere else."""


def wikicache_dir() -> Path:
    """The resolved wikicache directory (may not exist)."""
    explicit = os.environ.get("FANOUTQA_WIKICACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("FANOUTQA_CACHE_DIR")
    if base:
        return Path(base).expanduser() / "wikicache"
    from fanoutqa.wiki import WIKI_CACHE_DIR

    return Path(WIKI_CACHE_DIR)


def require_wikicache() -> Path:
    """Resolve the wikicache directory or raise with the setup instructions."""
    path = wikicache_dir()
    if not path.is_dir():
        raise FileNotFoundError(f"FanOutQA wikicache not found at {path}.\n\n{_SETUP_HINT}")
    for name in os.listdir(path):
        if name.endswith(("-dated.md", "-dated.html")):
            return path
    raise FileNotFoundError(
        f"FanOutQA wikicache at {path} contains no '*-dated.md' / '*-dated.html' files.\n\n{_SETUP_HINT}"
    )


def md_path(cache: Path, pageid: int | str) -> Path:
    return cache / f"{pageid}-dated.md"


def html_path(cache: Path, pageid: int | str) -> Path:
    return cache / f"{pageid}-dated.html"


def materialize_md(cache: Path, pageid: int | str, *, write: bool = True) -> str | None:
    """Return the markdown body for ``pageid`` using only the local cache."""
    mdp = md_path(cache, pageid)
    if mdp.exists():
        try:
            return mdp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            pass
    htmlp = html_path(cache, pageid)
    if not htmlp.exists():
        return None
    from fanoutqa.utils import markdownify

    text = markdownify(htmlp.read_text(encoding="utf-8"))
    if write:
        try:
            mdp.write_text(text, encoding="utf-8")
        except OSError:
            pass
    return text


def doc_text(evidence: Any, *, cache: Path) -> str | None:
    """Body of one ``fanoutqa.models.Evidence`` from the local cache (``None`` if absent)."""
    return materialize_md(cache, evidence.pageid)


__all__ = [
    "wikicache_dir",
    "require_wikicache",
    "md_path",
    "html_path",
    "materialize_md",
    "doc_text",
]
