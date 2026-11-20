"""Shared code for the CTC fake sub-LMs: find the items in a prompt and their labels, read the instruction, and
format a reply the way the instruction asked."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from synth_common.oracle import FakeSubLM, rng_for, wants_json
from synth_common.segindex import _SPLIT, _WS, _variants

_KW = (r"\b(?:document|doc|passage|claim|item|paper|abstract|article|chunk|sentence|query|question|entry|"
       r"snippet|excerpt|para(?:graph)?|candidate|statement|id)")
_P_KW = re.compile(_KW + r"\s*[#:]?\s*[\[(]?\s*(\d{1,7})(?=\s*[\])]|\s*[:.,*]|\s|$)", re.I)
_LS = r"(?:^|\n)[ \t>*_\-•]*(?:\*\*)?"
_P_BRACKET = re.compile(_LS + r"[\[(<{]\s*(#?[A-Za-z]{0,2}\s?\d{1,7}|[A-Za-z])\s*[\])>}]")
_P_ALNUM = re.compile(_LS + r"([A-Za-z]{1,3}\s?\d{1,5})(?:\*\*)?\s*[:.)\]\-]")
_P_NUM = re.compile(_LS + r"#?(\d{1,7})(?:\*\*)?\s*[:.)\]\-]")
_P_LETTER = re.compile(_LS + r"([A-Z])(?:\*\*)?\s*[:.)]")
_P_XML = re.compile(r"\bid\s*=\s*[\"']?([A-Za-z]{0,2}\d{1,7})", re.I)
_P_KWLETTER = re.compile(r"(?:^|\n)[ \t>*_\-•]*(?:\*\*)?(?i:claim|passage|statement|document|doc|text|sentence|item|"
                         r"paper|abstract|option|candidate)\s+([A-Z]{1,2})(?:\*\*)?\s*[:.)]")
_PATTERNS = (_P_KW, _P_XML, _P_KWLETTER, _P_BRACKET, _P_ALNUM, _P_NUM, _P_LETTER)


def _clean_label(s: str) -> str:
    s = s.strip().lstrip("#").replace(" ", "")
    return s


def label_value(lab: str) -> Any:
    """How a label is written in a JSON reply: plain integers as ints, anything else as a string."""
    return int(lab) if lab.isdigit() else lab


def find_label(header: str) -> str | None:
    return find_label_span(header)[0]


def find_label_span(header: str) -> tuple[str | None, int]:
    """(label, offset in ``header`` where the label's line/match starts); (None, len(header)) if none."""
    for pat in _PATTERNS:
        ms = list(pat.finditer(header))
        if ms:
            return _clean_label(ms[-1].group(1)), ms[-1].start()
    return None, len(header)


@dataclass
class Seen:
    iid: str
    label: str
    pos: int
    end: int
    cov: float
    full: bool = True


@dataclass
class View:
    prompt: str
    items: list[Seen]
    instr: str
    residual: str = ""
    by_iid: dict[str, Seen] = field(default_factory=dict)
    by_label: dict[str, Seen] = field(default_factory=dict)

    @property
    def low(self) -> str:
        return self.instr.lower()

    def ids(self) -> list[str]:
        return [s.iid for s in self.items]


_ITEM_PREFIX = re.compile(
    r"^[\s>*_\-•#]*(?:\*\*)?(?:(?:[A-Za-z]{1,12}\s*)?[\[(#]?\s*[A-Za-z]{0,3}\s?\d{1,7}\s*[\])]?\s*(?:\*\*)?\s*[:.)\-]\s*|"
    r"[\[(][^\])]{1,24}[\])]\s*[:.\-]?\s*|[A-Za-z]{1,12}\s+[A-Z]{1,2}\s*(?:\*\*)?\s*[:.)]\s+|"
    r"[A-Za-z]{1,12}\s*[:.]\s+)(?:\*\*)?\s*")
_TYPE_PREFIX = re.compile(r"^(?:Query|Question|Document|Doc|Passage|Claim|Abstract|Text|Paper)\s*[:.\-]\s*", re.I)


def _my_variants(s: str):
    seen = set()
    for v in _variants(s):
        if v not in seen:
            seen.add(v)
            yield v
    r = s.rstrip(".:;, ")
    if r != s and r not in seen:
        seen.add(r)
        yield r
    t = s
    for _ in range(3):
        t2 = _TYPE_PREFIX.sub("", _ITEM_PREFIX.sub("", t, count=1), count=1).strip().strip("\"'`*")
        if t2 == t or len(t2) < 12:
            break
        t = t2
        if t not in seen:
            seen.add(t)
            yield t


def _split_region(prompt: str, a: int, b: int, out: list) -> None:
    pos = a
    for m in _SPLIT.finditer(prompt, a, b):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < b:
        out.append((pos, b))


_TITLE_OPEN = re.compile(r"\(Title:\s*")


def _pieces(prompt: str) -> list[tuple[int, int]]:
    """Sentence pieces of ``prompt``; "(Title: ...)" blocks are kept separate."""
    regions, pos = [], 0
    for m in _TITLE_OPEN.finditer(prompt):
        a = m.start()
        if a < pos:
            continue
        depth, end = 0, None
        for k in range(a, min(len(prompt), a + 2000)):
            c = prompt[k]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = k
                    break
        if end is None:
            continue
        regions.append((pos, a))
        regions.append((m.end(), end))
        pos = end + 1
    regions.append((pos, len(prompt)))
    out: list[tuple[int, int]] = []
    for a, b in regions:
        if b > a:
            _split_region(prompt, a, b, out)
    return out


def _norm_head(t: str, n: int = 32) -> str:
    return _WS.sub(" ", t).strip()[:n]


def build_view(prompt: str, index, min_cov: float = 0.6, texts: dict | None = None, partial: float = 0.0,
               occurrences: bool = False) -> View:
    """Locate indexed items in ``prompt`` with the labels the root gave them. O(len(prompt))."""
    pieces = _pieces(prompt)
    keep_mask = []
    runs: list[dict] = []
    seg2item = index.seg2item
    min_len = index.min_len
    for a, b in pieces:
        s = _WS.sub(" ", prompt[a:b]).strip()
        iid = None
        if len(s) >= min(min_len, 12):
            for v in _my_variants(s):
                if len(v) < min_len:
                    continue
                iid = seg2item.get(v)
                if iid is not None:
                    if runs and runs[-1]["iid"] == iid and v not in runs[-1]["segs"]:
                        runs[-1]["b"] = b
                        runs[-1]["segs"].add(v)
                    else:
                        k = s.find(v)
                        runs.append({"iid": iid, "a": a, "b": b, "segs": {v}, "prefix": s[:k] if k > 0 else ""})
                    break
        keep_mask.append(iid is None)
    if not occurrences:
        merged: dict[str, dict] = {}
        for r in runs:
            m = merged.get(r["iid"])
            if m is None:
                merged[r["iid"]] = {**r, "segs": set(r["segs"])}
            else:
                m["b"] = max(m["b"], r["b"])
                m["segs"] |= r["segs"]
        runs = sorted(merged.values(), key=lambda r: r["a"])
    floor = min(min_cov, partial) if partial > 0 else min_cov
    seen: list[Seen] = []
    prev_end = 0
    for r in runs:
        iid = r["iid"]
        cov = len(r["segs"]) / index.n_segs[iid]
        if cov < floor:
            continue
        st = r["a"]
        lo = max(prev_end, st - 400)
        header = None
        if texts is not None and iid in texts:
            head = _norm_head(texts[iid])
            if len(head) >= 12:
                p = prompt.find(head, lo, st + len(head) + 1)
                if p >= 0:
                    header = prompt[max(prev_end, p - 200):p]
        if header is None:
            header = prompt[lo:st] + r["prefix"]
        seen.append(Seen(iid=iid, label=find_label(header) or "", pos=st, end=r["b"], cov=cov, full=cov >= min_cov))
        prev_end = max(prev_end, r["b"])
    used: set[str] = set()
    for k, s_ in enumerate(seen):
        if not s_.label or (s_.label in used and not occurrences):
            s_.label = str(k + 1) if str(k + 1) not in used else f"#{k + 1}"
        used.add(s_.label)
    residual = "\n".join(_WS.sub(" ", prompt[a:b]).strip() for (a, b), keep in zip(pieces, keep_mask, strict=True)
                         if keep and prompt[a:b].strip())
    if seen:
        head_end = seen[0].pos
        hdr = prompt[max(0, head_end - 300):head_end]
        lab, off = find_label_span(hdr)
        if lab is not None:
            head_end = max(0, head_end - len(hdr) + off)
        tail_start = max(x.end for x in seen)
        instr = (prompt[:head_end].strip() + "\n" + prompt[tail_start:].strip()).strip()
    else:
        instr = residual
    v = View(prompt=prompt, items=seen, instr=instr, residual=residual)
    for x in seen:
        v.by_iid.setdefault(x.iid, x)
        v.by_label.setdefault(x.label, x)
    return v


_ID_KEYS = ("id", "doc", "doc_id", "docid", "document", "passage", "passage_id", "claim", "claim_id", "n", "item",
            "paper", "pid", "index", "label_id", "number", "query", "query_id", "qid", "pair")


def _first_json_example(instr: str) -> Any | None:
    """The first JSON-ish example the instruction shows (placeholders like <id> made parseable)."""
    for m in re.finditer(r"[\[{]", instr):
        start = m.start()
        depth, end = 0, None
        for k in range(start, min(len(instr), start + 600)):
            c = instr[k]
            if c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
        if end is None:
            continue
        frag = instr[start:end]
        if len(frag) < 2 or re.fullmatch(r"\[\s*#?[A-Za-z]{0,2}\d{1,7}\s*\]|\(\s*\d{1,7}\s*\)", frag):
            continue
        fixed = re.sub(r'"<[^<>"]{0,60}>"', '"X"', frag)
        fixed = re.sub(r"<[^<>]{0,60}>", '"X"', fixed)
        alt = r'("[^"]*"|-?\d+(?:\.\d+)?|true|false|null)'
        fixed = re.sub(alt + r"(?:\s*(?:or|/|\|)\s*" + alt + r")+", r"\1", fixed)
        fixed = re.sub(r'(?<=[\[{,:\s])([A-Za-z_][\w ]{0,30}?)(?=\s*[,\]}:])',
                       lambda mm: mm.group(1) if mm.group(1) in ("true", "false", "null") else f'"{mm.group(1)}"', fixed)
        fixed = fixed.replace("...", "").replace("…", "")
        fixed = re.sub(r",\s*([\]}])", r"\1", fixed)
        fixed = re.sub(r",\s*,", ",", fixed)
        try:
            return json.loads(fixed)
        except Exception:  # noqa: BLE001
            continue
    return None


@dataclass
class Schema:
    kind: str = "dict"
    id_key: str = "id"
    val_key: str = "label"
    extra: dict = field(default_factory=dict)


def detect_schema(instr: str, default: str = "dict") -> Schema:
    if not wants_json(instr) and not re.search(r"[\[{]\s*[\[{\"'<\d]", instr):
        return Schema(kind="lines")
    ex = _first_json_example(instr)
    if isinstance(ex, list) and ex:
        e0 = ex[0]
        if isinstance(e0, list):
            return Schema(kind="pairs")
        if isinstance(e0, dict):
            keys = list(e0.keys())
            idk = next((k for k in keys if k.lower() in _ID_KEYS), keys[0] if keys else "id")
            valk = next((k for k in keys if k != idk), "label")
            return Schema(kind="objects", id_key=idk, val_key=valk, extra={k: v for k, v in e0.items()
                                                                          if k not in (idk, valk)})
        return Schema(kind="list")
    if isinstance(ex, dict) and ex:
        return Schema(kind="dict", extra={"keys": list(ex.keys())})
    if re.search(r"json\s+(?:list|array)|list of (?:the )?(?:ids|numbers|labels|document)", instr, re.I):
        return Schema(kind="list")
    return Schema(kind=default)


def render_selection(labels: list[str], instr: str, empty_word: str = "NONE") -> str:
    """A set of chosen items: JSON list of labels when JSON is wanted, else a short line."""
    vals = [label_value(lb) for lb in labels]
    if wants_json(instr) or re.search(r"\[\s*(?:\d|id|<)", instr):
        return json.dumps(vals)
    if not labels:
        return empty_word
    return ", ".join(f"[{lb}]" if lb.isdigit() else lb for lb in labels)


def render_per_item(pairs: list[tuple[str, Any]], instr: str) -> str:
    sch = detect_schema(instr)
    if sch.kind == "lines":
        out = []
        for lb, v in pairs:
            vv = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
            out.append(f"{lb}: {vv}")
        return "\n".join(out)
    if sch.kind == "pairs":
        return json.dumps([[label_value(lb), v] for lb, v in pairs], ensure_ascii=False)
    if sch.kind == "objects":
        return json.dumps([{sch.id_key: label_value(lb), sch.val_key: v} for lb, v in pairs], ensure_ascii=False)
    if sch.kind == "list":
        return json.dumps([v for _, v in pairs], ensure_ascii=False)
    return json.dumps({str(lb): v for lb, v in pairs}, ensure_ascii=False)


def note_call(st: dict, prompt: str, n_items: int, n_corpus: int) -> None:
    """Per-call stats for the program checks."""
    st["max_items_per_call"] = max(int(st.get("max_items_per_call", 0)), n_items)
    if n_corpus and n_items >= max(60, 0.5 * n_corpus):
        st["whole_corpus_calls"] = int(st.get("whole_corpus_calls", 0)) + 1
    if wants_json(prompt):
        st["json_prompts"] = int(st.get("json_prompts", 0)) + 1


def bump(st: dict, key: str, k: int = 1) -> None:
    st[key] = int(st.get(key, 0)) + k


_TEXTS: OrderedDict = OrderedDict()


def texts_of(hidden: dict) -> dict:
    """{item id: text} for the rollout (cached per rollout payload)."""
    items = hidden.get("items") or []
    key = id(items)
    hit = _TEXTS.get(key)
    if hit is not None and hit[0] is items:
        _TEXTS.move_to_end(key)
        return hit[1]
    d = {str(i): t for i, t in items}
    _TEXTS[key] = (items, d)
    while len(_TEXTS) > 256:
        _TEXTS.popitem(last=False)
    return d


class CTCOracle(FakeSubLM):
    """Base class: ``respond`` builds the View and dispatches to ``answer(view, hidden, st)``."""

    partial: float = 0.2
    occurrences: bool = False

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        view = build_view(prompt, hidden["_index"], self.read.min_cov, texts=texts_of(hidden), partial=self.partial,
                          occurrences=self.occurrences)
        note_call(st, prompt, len({x.iid for x in view.items if x.full}), len(hidden.get("items") or []))
        body = self.answer(view, hidden, st)
        return self.chat(prompt, body)

    def answer(self, view: View, hidden: dict, st: dict) -> str:  # pragma: no cover
        raise NotImplementedError

    def rng(self, *parts: Any):
        return rng_for(*parts)

    def read_view(self, view: View, st: dict, salt: Any = "") -> tuple[list[Seen], list[Seen]]:
        """Load model: items past ``reliable_items`` may be silently skipped."""
        ids = [s.iid for s in view.items]
        read, skipped = self.read_items(view.prompt, ids, st, salt)
        r = set(read)
        return [s for s in view.items if s.iid in r], [s for s in view.items if s.iid not in r]


__all__ = ["CTCOracle", "Schema", "Seen", "View", "build_view", "bump", "note_call", "texts_of", "detect_schema", "find_label", "label_value",
           "render_per_item", "render_selection"]
