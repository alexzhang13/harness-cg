"""Which context items did the root actually put into a sub-call prompt?"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable

_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_WS = re.compile(r"\s+")
_PREFIX = re.compile(r"^(?:[-*•]\s*|\[[^\]]{1,24}\]\s*|\(?[A-Za-z_]{0,12}\s?#?\d{1,7}[\]):.]\s*|[A-Za-z_]{1,12}:\s+)")

MIN_SEG = 24
_CTX_MIN = 3


def segments(text: str, min_len: int = MIN_SEG) -> list[str]:
    """Sentence/line segments of ``text``, whitespace-normalized, shorter than ``min_len`` dropped."""
    out = []
    for piece in _SPLIT.split(text):
        s = _WS.sub(" ", piece).strip()
        if len(s) >= min_len:
            out.append(s)
    return out


def _variants(seg: str) -> Iterable[str]:
    yield seg
    cur = seg
    for _ in range(2):
        stripped = _PREFIX.sub("", cur, count=1)
        if stripped == cur or len(stripped) < MIN_SEG:
            break
        yield stripped
        cur = stripped
    t = seg.strip("\"'`")
    if t != seg and len(t) >= MIN_SEG:
        yield t


class SegmentIndex:
    """Maps sentence segments to item ids; answers "which items does this prompt contain, and how fully"."""

    def __init__(self, items: Iterable[tuple[str, str]], min_len: int = MIN_SEG,
                 context_texts: Iterable[str] | None = None):
        self.min_len = min_len
        seg2item: dict[str, str] = {}
        dup: set[str] = set()
        ctx: set[str] = set()
        item_chars = 0
        for iid, text in items:
            item_chars += len(text)
            pieces = segments(text, _CTX_MIN)
            ctx.update(pieces)
            for s in {p for p in pieces if len(p) >= min_len}:
                prev = seg2item.get(s)
                if prev is not None and prev != iid:
                    dup.add(s)
                else:
                    seg2item[s] = iid
        for s in dup:
            seg2item.pop(s, None)
        for text in context_texts or ():
            if item_chars < 0.95 * len(text):
                ctx.update(segments(text, _CTX_MIN))
        self.ctx_segs = ctx
        self.seg2item = seg2item
        self.n_segs: dict[str, int] = dict(Counter(seg2item.values()))

    def __len__(self) -> int:
        return len(self.n_segs)

    def coverage(self, prompt: str) -> dict[str, float]:
        """item id -> fraction of its (unique) segments present in ``prompt``. Items absent from the prompt are omitted."""
        hits: dict[str, set[str]] = defaultdict(set)
        for s in segments(prompt, self.min_len):
            for v in _variants(s):
                iid = self.seg2item.get(v)
                if iid is not None:
                    hits[iid].add(v)
                    break
        return {iid: len(v) / self.n_segs[iid] for iid, v in hits.items()}

    def present(self, prompt: str, min_cov: float = 0.6) -> list[str]:
        """Ids of items whose coverage is at least ``min_cov``, in order of first appearance in the prompt."""
        cov = self.coverage(prompt)
        ok = {i for i, c in cov.items() if c >= min_cov}
        order: list[str] = []
        seen: set[str] = set()
        for s in segments(prompt, self.min_len):
            for v in _variants(s):
                iid = self.seg2item.get(v)
                if iid is not None:
                    if iid in ok and iid not in seen:
                        seen.add(iid)
                        order.append(iid)
                    break
        return order

    def residual(self, prompt: str) -> str:
        """``prompt`` with every piece of context text removed (indexed or not): what is left is the root's instruction."""
        keep = []
        for piece in _SPLIT.split(prompt):
            s = _WS.sub(" ", piece).strip()
            if not s:
                continue
            if any(v in self.seg2item or v in self.ctx_segs for v in _variants(s)):
                continue
            stripped = _PREFIX.sub("", s, count=1)
            if stripped != s and stripped in self.ctx_segs:
                continue
            keep.append(s)
        return "\n".join(keep)
