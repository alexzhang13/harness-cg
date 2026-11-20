"""Corpus-scale ``context`` by reference (doc stores)."""

from __future__ import annotations

import json
import mmap
import os
import shutil
import tempfile
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

MARKER = "__rlm_doc_store__"
FORMAT = "rlm-doc-store-v1"
LOCAL_ROOT_ENV = "RLM_DOC_STORE_LOCAL_ROOT"
DEFAULT_LOCAL_ROOT = "/tmp/rlm_doc_stores"


def is_ref(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get(MARKER), dict)


def make_ref(path: str | os.PathLike, idx: list[int], chars: int) -> dict:
    return {MARKER: {"path": str(path), "idx": [int(i) for i in idx], "chars": int(chars)}}


def build_store(out_dir: str | os.PathLike, items: Iterable[tuple[str, str]]) -> dict:
    """Write (key, text) items as a store (temp dir, then rename)."""
    out = Path(out_dir)
    tmp = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    keys, offsets, chars = [], [0], []
    with open(tmp / "docs.bin", "wb") as f:
        for key, text in items:
            b = text.encode("utf-8", "surrogatepass")
            f.write(b)
            keys.append(str(key))
            offsets.append(offsets[-1] + len(b))
            chars.append(len(text))
    meta = {"format": FORMAT, "n": len(keys), "keys": keys, "offsets": offsets, "chars": chars}
    (tmp / "index.json").write_text(json.dumps(meta))
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return meta


def local_mirror(path: str | os.PathLike) -> Path:
    return Path(os.environ.get(LOCAL_ROOT_ENV) or DEFAULT_LOCAL_ROOT) / Path(path).name


def resolve_dir(path: str | os.PathLike) -> Path:
    """The node-local copy if complete, else the shared path."""
    p = Path(path)
    loc = local_mirror(p)
    try:
        if (loc / "index.json").exists() and (loc / "docs.bin").stat().st_size == (p / "docs.bin").stat().st_size:
            return loc
    except OSError:
        pass
    return p


def stage_local(path: str | os.PathLike) -> Path:
    """Copy a store to local disk once."""
    p = Path(path)
    loc = local_mirror(p)
    if resolve_dir(p) == loc:
        return loc
    try:
        loc.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(loc.parent).free < 3 * (p / "docs.bin").stat().st_size:
            return p
        tmp = Path(tempfile.mkdtemp(prefix=loc.name + ".tmp-", dir=loc.parent))
        shutil.copy2(p / "index.json", tmp / "index.json")
        shutil.copy2(p / "docs.bin", tmp / "docs.bin")
        try:
            os.replace(tmp, loc)
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
    except OSError:
        return p
    return resolve_dir(p)


@lru_cache(maxsize=4)
def load_index(path: str) -> dict:
    return json.loads((Path(path) / "index.json").read_text())


class _Store:
    def __init__(self, path: str):
        d = resolve_dir(path)
        self.meta = load_index(str(d))
        self._f = open(d / "docs.bin", "rb")
        self._mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ) if self.meta["offsets"][-1] else b""

    def doc(self, i: int) -> str:
        o = self.meta["offsets"]
        return self._mm[o[i]:o[i + 1]].decode("utf-8", "surrogatepass")


@lru_cache(maxsize=4)
def open_store(path: str) -> _Store:
    return _Store(path)


def materialize(payload: Any, limit: int | None = None) -> Any:
    """The list[str] behind a ref (first `limit` docs); other payloads unchanged."""
    if not is_ref(payload):
        return payload
    r = payload[MARKER]
    st = open_store(r["path"])
    idx = r["idx"] if limit is None else r["idx"][:limit]
    return [st.doc(i) for i in idx]


def verbatim_chars(payload: Any, lines: Iterable[str], min_line: int = 40, max_checks: int = 400) -> int:
    """Characters of `lines` found verbatim in the store (mmap search)."""
    mm = open_store(payload[MARKER]["path"])._mm
    n = checks = 0
    for line in lines:
        t = line.strip().strip("'\"")
        if len(t) < min_line:
            continue
        checks += 1
        if checks > max_checks:
            break
        if mm.find(t[:200].encode("utf-8", "surrogatepass")) >= 0:
            n += len(t)
    return n


def total_chars(payload: Any) -> int:
    r = payload[MARKER]
    if "chars" in r:
        return int(r["chars"])
    c = load_index(str(resolve_dir(r["path"])))["chars"]
    return sum(c[i] for i in r["idx"])


def query_metadata(payload: Any):
    """QueryMetadata for a payload; for a ref, without loading the docs."""
    from rlm_train.prompts import QueryMetadata
    if not is_ref(payload):
        return QueryMetadata(payload)
    md = QueryMetadata([""])
    r = payload[MARKER]
    md.context_type = "list"
    md.context_lengths = [0] * len(r["idx"]) if r["idx"] else [0]
    md.context_total_length = total_chars(payload)
    return md
