"""Fake sub-LM for varied-surface periodic corpora."""

from __future__ import annotations

import json
import re
import statistics
from collections import OrderedDict
from typing import Any

from synth_common.oracle import rng_for

from .periodic import MONTHS

CALIBRATION: dict[str, Any] = {
    "select": {
        "wrong_adj": 0.0064,
        "wrong_measure": 0.098,
        "wrong_measure_by_section": {"hash": 0.010, "caps": 0.096, "plain": 0.211},
        "col_off": 0.028,
        "col_off_up": 0.714,
        "col_other": 0.032,
        "col_off_multi": 0.015,
        "col_other_multi": 0.020,
        "mult_wrong": 0.34,
        "mult_wrong_by_scale": {"1000": 0.259, "1000000": 0.495},
        "missing_series": 0.0143,
        "json_fail": 0.0056,
        "fence": 0.389,
    },
    "read": {
        "series_err_exact": 0.015,
        "series_err_inexact": 0.082,
        "series_err_table": 0.45,
        "series_err_right": 0.98,
        "cell_err": 0.0005,
        "rows_dropped": 0.010,
        "empty_present": 0.008,
        "absent_empty": 0.025,
        "json_fail": 0.0009,
        "fence": 0.0049,
        "str_values": 0.0,
    },
    "other": {"fence": 0.42},
}
VARIED_BUDGET_CHARS = 200_000
LOAD_FREE_CHARS = 12_000

_NON = re.compile(r"[^a-z0-9%]+")
_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?(?![\w])")
MON3 = {m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)}
_MON_RE = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
_TRANS_HEAD = re.compile(_MON_RE + r"\s+(\d{4})")
_LINE_CELL = re.compile(r"([A-Z][a-z]+)\s+(\d{4}):\s*([^;]+?)\s*(?:;|\.?$)")
_ARR2 = re.compile(r"\[\s*\[.*?\]\s*\]", re.S)
_JSON_ROW = re.compile(r'\[\s*"(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")+\s*\]')
_ARR1 = re.compile(r'\[(?:\s*"[^"\n]*"\s*,?\s*){3,}\]')
WEAK_SUBS = {"total", "all", "all classes", "overall", "small", "medium", "large", "single", "paired", "multiple",
             "light", "standard", "heavy"}
WEAK_GROUPS = {"category", "sector", "area"}
SING_SKIP = {"returns", "transfers", "listings", "filings"}


def norm(t: str) -> str:
    t = t.lower().replace("&amp;", "&").replace("’", "").replace("'", "")
    return " ".join(_NON.sub(" ", t).split())


def _singular(noun: str) -> str | None:
    w = noun.split()
    if w[0] in SING_SKIP:
        return None
    if w[0].endswith("ies"):
        w[0] = w[0][:-3] + "y"
    elif w[0].endswith("s") and not w[0].endswith("ss"):
        w[0] = w[0][:-1]
    else:
        return None
    return " ".join(w)


STOP = set("jan feb mar apr may jun jul aug sep oct nov dec january february march april june july august september "
           "october november december monday tuesday wednesday thursday friday the of and a an at to in for".split())


def caption_key(cap: str) -> str:
    """Table identity as the probe-v3 program computes it: caption words and numbers, no dates."""
    return " ".join(w for w in re.findall(r"[a-z0-9]+", cap.lower()) if w not in STOP and not re.fullmatch(r"\d{3,}", w))


class View:
    """Per-corpus lexicon (display name -> internal key) and printed-row index (printed cells -> period)."""

    def __init__(self, hidden: dict):
        sf = hidden["surface"]
        self.sf = sf
        self.scale = float(hidden["scale"])
        self.cols = [tuple(c) for c in hidden["cols"]]
        self.disp = [(sf["g"].get(g, g), sf["s"].get(s, s)) for g, s in self.cols]
        self.idx = {d: i for i, d in enumerate(self.disp)}
        self.groups = list(dict.fromkeys(G for G, _ in self.disp))
        self.national = self.disp[0][0]
        self.reports = hidden["reports"]
        self.tables: OrderedDict[tuple[str, str], dict] = OrderedDict()
        for rep in self.reports:
            for tid, t in rep["tables"].items():
                k = (t["m"], t["adj"])
                info = self.tables.setdefault(k, {"tid": tid, "caption": t["caption"], "rows": {}, "layouts": set()})
                info["layouts"].add(t["layout"])
                dec = hidden["decimals"][f"{t['m']}:{t['adj']}"]
                for ym, row in t["rows"].items():
                    info["rows"].setdefault(tuple("S" if v is None else f"{v:,.{dec}f}" for v in row), ym)
        self.dec = {k: hidden["decimals"][f"{k[0]}:{k[1]}"] for k in self.tables}
        self.by_caption = {v["caption"]: k for k, v in self.tables.items()}
        self.measures = list(dict.fromkeys(m for m, _ in self.tables))
        self.noun = {m: sf["m"][m]["noun"] for m in self.measures}
        self.adjw = dict(sf["adj"])
        self.unit = sf["unit"]
        self.mark = sf["style"]["revmark"]
        self.disp_label = {}
        for i, (G, S) in enumerate(self.disp):
            self.disp_label[f"{G} - {S}"] = i
            self.disp_label[f"{G}, {S}"] = i
        syn: dict[str, tuple] = {}

        def add(phrase: str, payload: tuple, override: bool = False) -> None:
            p = norm(phrase)
            if p and (override or p not in syn):
                syn[p] = payload
        cap_style = sf["style"]["caption"]
        for (m, adj), info in self.tables.items():
            add(info["caption"], ("tab", (m, adj)), True)
            add(caption_key(info["caption"]), ("tab", (m, adj)), True)
            tn = sf["tnum"][m]
            letter, k12 = ("a", 1) if adj == "SA" else ("b", 2)
            add({"table": f"table {tn}{letter}", "upper": f"table {tn} {letter}", "series": f"series s{tn} {adj}",
                 "exhibit": f"exhibit {tn} {k12}", "panel": f"panel {chr(64 + tn)}{k12}"}[cap_style], ("tab", (m, adj)), True)
            add({"table": f"table {tn}", "upper": f"table {tn}", "series": f"series s{tn}", "exhibit": f"exhibit {tn}",
                 "panel": f"panel {chr(64 + tn)}"}[cap_style], ("mid", m))
        if cap_style == "series":
            for m in self.measures:
                add(f"s{sf['tnum'][m]}", ("mid", m))
        for m in self.measures:
            add(self.noun[m], ("m", m), True)
            sg = _singular(self.noun[m])
            if sg:
                add(sg, ("m", m))
        for a, w in self.adjw.items():
            add(w, ("adj", a), True)
        opp = {norm(w): a for a, w in self.adjw.items()}
        for a, ws in (("NSA", ["not seasonally adjusted", "not adjusted", "non adjusted", "unadjusted", "nsa"]),
                      ("SA", ["seasonally adjusted", "adjusted", "sa"])):
            for w in ws:
                if opp.get(norm(w), a) == a:
                    add(w, ("adj", a))
        for G in self.groups:
            add(G, ("g", G), True)
        for _, S in self.disp:
            if norm(S) not in syn or syn[norm(S)][0] != "g":
                add(S, ("s", S))
        self.syn = syn
        alts = sorted(syn, key=len, reverse=True)
        self._re = re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(a) for a in alts) + r")(?![a-z0-9])")

    def spans(self, text: str) -> list[tuple[int, int, tuple]]:
        nt = norm(text)
        return [(m.start(), m.end(), self.syn[m.group(0)]) for m in self._re.finditer(nt)]

    def group_default(self, G: str) -> int | None:
        if norm(G) in WEAK_GROUPS:
            return None
        cand = [i for i, (g, _s) in enumerate(self.disp) if g == G]
        tot = [i for i in cand if norm(self.disp[i][1]) in {"total", "all", "all classes", "overall"}]
        return (tot or cand or [None])[0]

    def _strong_case(self, S: str, text: str) -> bool:
        if re.search(r"[\"'‘“]" + re.escape(S) + r"[\"'’”]", text, re.I):
            return True
        if S == S.lower():
            return False
        for m in re.finditer(r"(?<![A-Za-z])" + re.escape(S) + r"(?![a-z])", text):
            before = text[:m.start()].rstrip(" \"'‘“")
            if before and before[-1] not in ".!?\n:":
                return True
        return False

    def find_cols(self, text: str) -> list[int]:
        sp = self.spans(text)
        groups = [(a, b, p[1]) for a, b, p in sp if p[0] == "g"]
        subs = [(a, b, p[1]) for a, b, p in sp if p[0] == "s"]
        used: set[int] = set()
        out: list[tuple[int, int]] = []
        for a, b, S in subs:
            best = None
            for ga, gb, G in groups:
                if (G, S) not in self.idx:
                    continue
                d = a - gb if 0 <= a - gb <= 40 else ((ga - b) + 5 if 0 <= ga - b <= 25 else None)
                if d is not None and (best is None or d < best[0]):
                    best = (d, G, ga)
            if best:
                out.append((a, self.idx[(best[1], S)]))
                used.add(best[2])
                continue
            if norm(S) in WEAK_SUBS and not self._strong_case(S, text):
                continue
            owners = [G for G in self.groups if (G, S) in self.idx]
            if owners:
                out.append((a, self.idx[(self.national if self.national in owners else owners[0], S)]))
        named_groups = {self.disp[ci][0] for _, ci in out}
        for ga, gb, G in groups:
            if any(0 <= a - gb <= 6 for a, _b, _S in subs):
                continue
            if ga not in used and G not in named_groups:
                ci = self.group_default(G)
                if ci is not None:
                    out.append((ga, ci))
        return list(dict.fromkeys(ci for _, ci in sorted(out)))

    def fuzzy_col(self, key: str) -> int | None:
        """Column for an abbreviated key like "Combined_total", or None if ambiguous."""
        words = _key_words(key)
        hit = self.find_cols(" ".join(words))
        if len(hit) == 1:
            return hit[0]
        sc = []
        for i, (G, S) in enumerate(self.disp):
            g = sum(1 for w in words for u in norm(G).split() if len(w) >= 3 and u.startswith(w[:6]))
            s_ = sum(1 for w in words for u in norm(S).split() if (len(w) >= 3 and u.startswith(w[:6])) or w == u)
            if s_ or (g and norm(S) in {"total", "all", "all classes", "overall"}):
                sc.append((g + s_ + (0.5 if s_ == len(norm(S).split()) else 0), i))
        sc.sort(reverse=True)
        if sc and (len(sc) == 1 or sc[0][0] > sc[1][0]):
            return sc[0][1]
        return None

    def parse_series(self, text: str) -> dict:
        """A series description in any wording -> {m, adj, tab, cols} (None where the text does not say)."""
        sp = self.spans(text)
        tab = next((p[1] for _, _, p in sp if p[0] == "tab"), None)
        m = tab[0] if tab else next((p[1] for _, _, p in sp if p[0] in ("mid", "m")), None)
        adj = tab[1] if tab else next((p[1] for _, _, p in sp if p[0] == "adj"), None)
        return {"m": m, "adj": adj, "tab": tab, "cols": self.find_cols(text)}

    def header_path(self, ci: int) -> str:
        G, S = self.disp[ci]
        return f"{G} / {S}"

    def series_name(self, m: str, adj: str, ci: int) -> str:
        G, S = self.disp[ci]
        where = G if norm(S) in {"total", "all", "all classes", "overall"} and norm(G) not in WEAK_GROUPS else (
            S if norm(G) in WEAK_GROUPS else f"{G}, {S}")
        return f"{self.adjw[adj]} {self.noun[m]} for {where}"


_VIEWS: OrderedDict[str, View] = OrderedDict()


def view_for(hidden: dict) -> View:
    key = hidden.get("gen_key") or str(id(hidden["reports"]))
    v = _VIEWS.get(key)
    if v is None:
        v = _VIEWS[key] = View(hidden)
        while len(_VIEWS) > 32:
            _VIEWS.popitem(last=False)
    else:
        _VIEWS.move_to_end(key)
    return v


def is_varied(hidden: dict | None) -> bool:
    return bool(hidden) and hidden.get("family") == "periodic" and (hidden.get("surface") or {}).get("mode") == "varied" \
        and "cols" in hidden


def _row_cells(line: str) -> tuple[str, list[str], bool] | None:
    """(label, non-empty value cells, last cell complete?) of a table row in any of the three layouts."""
    s = line.strip()
    if "<t" in s:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", s, re.S)
        if len(cells) < 2:
            return None
        cut = not s.endswith("</tr>")
        return cells[0].strip(), [c.strip() for c in cells[1:] if c.strip()], not cut
    if s.startswith("|"):
        parts = [p.strip() for p in s.strip("|").split("|")]
        whole = s.endswith("|")
        if not whole:
            parts = parts[:-1]
        if len(parts) < 2:
            return None
        return parts[0], [p for p in parts[1:] if p], True
    parts = re.split(r"\s{2,}", s)
    if len(parts) < 3:
        return None
    return parts[0], parts[1:], True


def _match(info: dict, cells: list[str], whole: bool) -> tuple[str, int] | None:
    """(period, readable cell count) of a printed row; a cut-off row matches as a prefix."""
    if not cells:
        return None
    ym = info["rows"].get(tuple(cells))
    if ym:
        return ym, len(cells)
    for full, ym in info["rows"].items():
        if len(cells) <= len(full) and list(full[:len(cells) - 1]) == cells[:-1] and full[len(cells) - 1].startswith(cells[-1]):
            n = len(cells) if full[len(cells) - 1] == cells[-1] else len(cells) - 1
            return (ym, n) if n > 0 else None
    return None


class Seg:
    __slots__ = ("key", "pos", "cells", "layout", "rows")

    def __init__(self, key: tuple[str, str] | None, pos: int):
        self.key, self.pos, self.layout = key, pos, "rows"
        self.cells: dict[str, dict[int, str]] = {}
        self.rows: list[str] = []

    def put(self, ym: str, ci: int, cell: str) -> None:
        if ym not in self.cells:
            self.rows.append(ym)
        self.cells.setdefault(ym, {})[ci] = cell


def visible(prompt: str, v: View) -> tuple[list[Seg], list[tuple[int, tuple[str, str]]]]:
    """Table segments readable in the prompt, matched to the hidden ledger. Returns (segments, caption positions)."""
    caps: list[tuple[int, tuple[str, str]]] = []
    for cap, k in v.by_caption.items():
        at = prompt.find(cap)
        while at >= 0:
            caps.append((at, k))
            at = prompt.find(cap, at + len(cap))
    caps.sort()
    segs: list[Seg] = []
    cur = Seg(None, 0)
    ci_cap = 0
    trans: list[str] | None = None
    pos = 0
    for line in prompt.split("\n"):
        pos += len(line) + 1
        while ci_cap < len(caps) and caps[ci_cap][0] < pos:
            if cur.cells:
                segs.append(cur)
            cur = Seg(caps[ci_cap][1], caps[ci_cap][0])
            trans = None
            ci_cap += 1
        s = line.strip()
        if not s or len(s) < 6:
            continue
        if s.startswith("|") and len(_TRANS_HEAD.findall(s)) >= 3:
            trans = [f"{y}-{MON3[mo.lower()]:02d}" for mo, y in _TRANS_HEAD.findall(s)]
            continue
        if trans is not None and s.startswith("|"):
            parts = [p.strip() for p in s.strip("|").split("|")]
            ci = v.disp_label.get(parts[0])
            if ci is not None:
                cur.layout = "transposed"
                for ym, cell in zip(trans, parts[1:] if s.endswith("|") else parts[1:-1], strict=False):
                    if cell:
                        cur.put(ym, ci, cell)
            continue
        if " -- " in s:
            lab, _, rest = s.partition(" -- ")
            ci = v.disp_label.get(lab.strip())
            if ci is not None:
                cur.layout = "lines"
                for mo, y, cell in _LINE_CELL.findall(rest):
                    if mo[:3].lower() in MON3 and cell.strip():
                        cur.put(f"{y}-{MON3[mo[:3].lower()]:02d}", ci, cell.strip())
                continue
        if '["' in s:
            arr = [[c for c in re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(0))] for m in _JSON_ROW.finditer(s)]
            rows_ = [(a[0], [c for c in a[1:] if c.strip()], True) for a in arr if len(a) >= 3]
        else:
            rc = _row_cells(s)
            rows_ = [rc] if rc else []
        for _lab, cells, whole in rows_:
            keys = [cur.key] if cur.key in v.tables else list(v.tables)
            for k in keys:
                hit = _match(v.tables[k], cells, whole)
                if hit:
                    if cur.key is None:
                        cur.key = k
                    for c in range(hit[1]):
                        cur.put(hit[0], c, cells[c])
                    break
    if cur.cells:
        segs.append(cur)
    return [s for s in segs if s.key is not None], caps


def releases_of(seg: Seg, v: View) -> list[int]:
    """Releases whose printed table matches every readable cell of a segment."""
    m, adj = seg.key
    out = []
    for r, rep in enumerate(v.reports):
        t = next((t for t in rep["tables"].values() if t["m"] == m and t["adj"] == adj), None)
        if t is None:
            continue
        ok = True
        for ym, cs in seg.cells.items():
            row = t["rows"].get(ym)
            if row is None:
                ok = False
                break
            dec = v.dec[seg.key]
            if any(("S" if row[ci] is None else f"{row[ci]:,.{dec}f}") != cell for ci, cell in cs.items() if ci < len(row)):
                ok = False
                break
        if ok:
            out.append(r)
    return out


REQ_WORDS = re.compile(r"\b(return|extract|give|need|read|report|list|provide|want|output|find|identify|choose|select|"
                       r"pick|which|what|tell|map|plan|locate|copy|transcribe)\b", re.I)
DESC_WORDS = re.compile(r"\b(columns are|grouped|structure|layout|consists?|header rows?|headers? (are|is)|"
                        r"sub-?categor(y|ies) (are|of)|each with)\b", re.I)
NEG_WORDS = re.compile(r"\b(do not|don't|not include|exclude|ignore|skip|except|other than|rather than|instead of)\b", re.I)


def instruction_lines(resid: str, v: View, prompt: str = "") -> list[str]:
    """The root's own words, one sentence per line, without pasted tables, captions or JSON dumps."""
    cat = []
    for ln in prompt.split("\n"):
        for c in v.by_caption:
            if c in ln and (len(ln) - len(c) < 70 or ln.count('":') >= 3):
                cat.append(ln)
                break
    keep = []
    for line in resid.split("\n"):
        s = line.strip()
        if not s or s.startswith(("<tr", "<table", "</table", "|")) or s in v.by_caption:
            continue
        if any(s in ln for ln in cat):
            continue
        nums = len(_NUM.findall(s))
        if nums >= 4 and (s.count('":') >= 3 or not REQ_WORDS.search(s)):
            continue
        keep.append(line)
    return keep


def _clean(s: str) -> str:
    return _ARR1.sub(" ", _ARR2.sub(" ", s))


def json_template(prompt: str) -> Any:
    """The JSON shape the prompt asks for (placeholders like <int> made parseable), or None."""
    for m in re.finditer(r"\bJSON\b[^{\n]{0,80}?(\{)", prompt, re.I):
        i, depth, q = m.start(1), 0, False
        for j in range(i, min(len(prompt), i + 4000)):
            ch = prompt[j]
            if ch == '"' and prompt[j - 1] != "\\":
                q = not q
            elif not q and ch in "{[":
                depth += 1
            elif not q and ch in "}]":
                depth -= 1
                if depth == 0:
                    raw = prompt[i:j + 1]
                    fixed = re.sub(r'(?<!")<([^<>"]*)>(?!")', lambda x: json.dumps("<" + x.group(1) + ">"), raw)
                    fixed = re.sub(r":\s*(number|int|integer|float|value|string|str)\b(?![\"\w])", r': "<\1>"', fixed)
                    try:
                        return json.loads(fixed)
                    except Exception:  # noqa: BLE001
                        keys = re.findall(r'"([^"]{1,40})"\s*:', raw)
                        return {k: "<?>" for k in keys} if keys else None
        break
    m = re.search(r"\bJSON\b[^.\n]{0,60}?\bkeys?\b\s*:?\s*((?:[\"'`]\w{1,40}[\"'`][\s,]*(?:and|or)?\s*){1,12})", prompt, re.I)
    if m:
        return {k: "<?>" for k in re.findall(r"[\"'`](\w{1,40})[\"'`]", m.group(1))}
    return None


def _is_period_key(k: str) -> bool:
    return bool(re.fullmatch(r"(YYYY-?MM|\d{4}-\d{2}|<?(month|period|date|ym)>?)", k.strip(), re.I))


READ_RE = re.compile(r"yyyy-?mm|\b(every|each|all) (the )?months?\b|\bmonth by month\b|\bper month\b|\bextract\b|"
                     r"\btranscribe\b|\bread (off|out)?\b.*\bvalues?\b|\bvalues? (for|of|in) (the |each |every |all )|"
                     r"\bcopy\b|\blist the values\b|\bmonthly (values|figures|numbers)\b", re.I)
ASK_RE = r"\b(what|which|how|explain|mean|meaning|means|indicate\w*|tell me|in what)\b"


def _asks(low: str, words: str) -> bool:
    """A sentence that asks a question about ``words`` (not an instruction that merely mentions them)."""
    return any(re.search(words, s) and re.search(ASK_RE, s) for s in re.split(r"(?<=[.?!])\s+|\n", low))


def intent(prompt: str, instr: str, tpl: Any, segs: list, caps: list, question_seen: bool) -> str:
    low = instr.lower()
    if isinstance(tpl, dict) and tpl and (all(_is_period_key(k) for k in tpl) or all(
            isinstance(x, dict) and x and all(_is_period_key(k) for k in x) for x in tpl.values())):
        return "read"
    keys = " ".join(_tpl_keys(tpl)).lower() if tpl is not None else ""
    if re.search(r"index|cell|column|key|table|header|caption", keys) and len({k for _, k in caps}) >= 2:
        return "select"
    reading = bool(READ_RE.search(low))
    if not reading:
        if _asks(low, r"\b(revis\w*|prelim\w*|marks?|marked|asterisks?|flags?|flagged)\b"):
            return "revision"
        if _asks(low, r"\b(reference (period|month)|release date|published|issued|(month|period|date)s? (does|is|of) (this|the) "
                      r"(release|document|bulletin|report)|cover(s|ed)?)\b"):
            return "dating"
        if _asks(low, r"\b(units?|multiplier|scale|thousands?|millions?)\b") and \
                not re.search(r"\b(which|what) (table|column|cell)", low):
            return "unit"
    if re.search(r"\b(which|what|choose|select|identify|pick|decide|determine|locate|plan|map)\b[^.\n]{0,80}\b("
                 r"tables?|columns?|cells?|series|captions?)\b|\bwhere (is|are|can)\b|\bcolumn index\b|"
                 r"\b(index|indices|position|locator)\b[^.\n]{0,40}\b(cells?|columns?|value)", low) and not reading:
        return "select" if (caps or segs) else "other"
    if reading:
        return "read"
    if re.search(r"\b(describe|structure|layout|what tables|list (the|all) tables|summari[sz]e|overview)\b", low):
        return "describe"
    if question_seen or re.search(r"\b(answer|sum|total of|average|mean|median|maximum|minimum|highest|lowest|how many|"
                                  r"count|which month|ratio)\b", low):
        return "aggregate"
    if segs and re.search(r"\b(value|values|figures?|numbers?)\b", low):
        return "read"
    return "other"


def _tpl_keys(tpl: Any) -> list[str]:
    out = []
    if isinstance(tpl, dict):
        for k, x in tpl.items():
            out.append(str(k))
            out += _tpl_keys(x)
    elif isinstance(tpl, list):
        for x in tpl:
            out += _tpl_keys(x)
    return out


class VariedReader:
    """Answers one prompt on a varied periodic corpus. ``oracle`` supplies chat() and its ReadModel."""

    def __init__(self, oracle: Any, cal: dict | None = None):
        self.o = oracle
        self.cal = cal or CALIBRATION

    def respond(self, prompt: str, resid: str, hidden: dict, st: dict) -> str:
        v = view_for(hidden)
        segs, caps = visible(prompt, v)
        lines = instruction_lines(resid, v, prompt)
        q = hidden.get("question") or ""
        qseen = bool(q) and _seen(q, prompt)
        tpl = json_template(prompt)
        nq = norm(q)
        own = [ln for ln in lines if not (qseen and len(norm(ln)) > 8 and norm(ln) in nq)]
        kind = intent(prompt, "\n".join(own), tpl, segs, caps, qseen)
        st.setdefault("varied_requests", {})
        st["varied_requests"][kind] = st["varied_requests"].get(kind, 0) + 1
        self._credit(segs, v, st)
        names_seen = sum(1 for G, S in v.disp if S in prompt or G in prompt) >= 2
        if not segs and not caps and not _has_doc_text(prompt, resid) and not (kind == "select" and names_seen):
            if tpl is not None:
                return self._wrap_json(prompt, json.dumps(_empty(tpl, _asked_month(lines))), self.cal["read"])
            from synth_common.oracle import refuse_aggregate
            st.setdefault("refusals", 0)
            return refuse_aggregate(st)
        if kind == "select":
            return self.select(prompt, own if qseen else lines, tpl, segs, caps, v, hidden, qseen, st)
        if kind == "read":
            return self.read(prompt, lines, tpl, segs, v, st)
        if kind == "aggregate":
            return self.aggregate(prompt, segs, v, hidden, qseen, lines, st)
        if kind == "unit":
            return self.o.chat(prompt, self._unit_text(prompt, v, tpl))
        if kind == "dating":
            return self.o.chat(prompt, self._dating(prompt, segs, v))
        if kind == "revision":
            return self.o.chat(prompt, self._revision(segs, v))
        return self.o.chat(prompt, self._describe(prompt, segs, caps, v, tpl, brief=kind == "other"))

    def _credit(self, segs: list[Seg], v: View, st: dict) -> None:
        reads = st.setdefault("read_ids", {})
        for s in segs:
            rs = releases_of(s, v)
            if len(rs) == 1:
                iid = f"R{rs[0]}:{v.tables[s.key]['tid']}"
                reads[iid] = reads.get(iid, 0) + 1
        st["items_read"] = len(reads)

    def _needed(self, lines: list[str], v: View, hidden: dict, qseen: bool, st: dict) -> list[tuple[str, str, int, str]]:
        """[(m, adj, column, name)] series the request asks about."""
        if qseen:
            return [(m, a, ci, v.series_name(m, a, ci)) for m, a, ci in hidden["spec"]["series"]]
        out: list[tuple[str, str, int, str]] = []
        per, g_meas, g_cols, g_adj = [], [], [], None
        for ln in lines:
            s = _clean(ln)
            if (DESC_WORDS.search(s) and len(v.find_cols(s)) >= 3) or any(c in s for c in v.by_caption) or NEG_WORDS.search(s):
                continue
            sp = v.spans(s)
            meas = list(dict.fromkeys((p[1][0] if p[0] == "tab" else p[1]) for _, _, p in sp if p[0] in ("tab", "mid", "m")))
            tabadj = next((p[1][1] for _, _, p in sp if p[0] == "tab"), None)
            adj = tabadj or next((p[1] for _, _, p in sp if p[0] == "adj"), None)
            cols = v.find_cols(s)
            g_adj = g_adj or adj
            if meas and cols:
                per.append((meas, adj, cols))
            elif meas:
                g_meas += [m for m in meas if m not in g_meas]
            elif cols:
                g_cols += [c for c in cols if c not in g_cols]
        if not per and (g_meas or g_cols):
            per.append((g_meas, None, g_cols))
        elif g_meas:
            per.append((g_meas, None, g_cols or per[0][2]))
        for meas, adj, cols in per:
            a = adj or g_adj
            if not meas or a is None or not cols:
                st["guesses"] = st.get("guesses", 0) + 1
            for m in meas or [v.measures[0]]:
                for ci in cols or [0]:
                    out.append((m, a or "SA", ci, v.series_name(m, a or "SA", ci)))
        return list(dict.fromkeys(out))

    def _target(self, lines: list[str], segs: list[Seg], v: View, prompt: str, st: dict) -> tuple[Any, list[int], bool]:
        """(table key, columns, named exactly?) a READ asks for. Quoted names first, descriptions / negations last."""
        cand = []
        for k, ln in enumerate(lines):
            s = _clean(ln)
            p = v.parse_series(s)
            score = (2 if REQ_WORDS.search(s) else 0) + (1 if re.search(r"['\"‘“]", s) else 0) + \
                (1 if re.search(r"\bcolumns?\b", s, re.I) else 0) - (3 if NEG_WORDS.search(s) else 0) - \
                (3 if DESC_WORDS.search(s) and len(p["cols"]) >= 3 else 0)
            cand.append((score, -k, p, s))
        cand.sort(key=lambda x: (-x[0], -x[1]))
        tab = next((p["tab"] for _, _, p, _ in cand if p["tab"]), None)
        m = next((p["m"] for _, _, p, _ in cand if p["m"]), None)
        adj = next((p["adj"] for _, _, p, _ in cand if p["adj"]), None)
        hit = next(((p["cols"], s) for sc, _, p, s in cand if p["cols"] and sc >= 0), None)
        key = tab or ((m, adj) if m and adj else None)
        seen = list(dict.fromkeys(s.key for s in segs))
        if key is None and seen:
            pool = [k for k in seen if (m is None or k[0] == m) and (adj is None or k[1] == adj)] or seen
            key = pool[0]
        if hit:
            cols, s = hit
            G, S = v.disp[cols[0]]
            ns, g, c = norm(s), re.escape(norm(G)), re.escape(norm(S))
            exact = bool(re.search(rf"(?<![a-z0-9]){g} (?:[a-z]{{0,6}} )?{c}(?![a-z0-9])|(?<![a-z0-9]){c} (?:column )?"
                                   rf"(?:under|of|in|for) (?:the )?{g}(?![a-z0-9])", ns))
            return key, cols, exact
        mi = re.search(r"\b(?:column|cell|value)\s*(?:index|no\.?|number|#)?\s*(\d{1,2})\b", "\n".join(lines), re.I)
        if not mi:
            st["guesses"] = st.get("guesses", 0) + 1
        return key, [int(mi.group(1))] if mi and int(mi.group(1)) < len(v.cols) else [0], False

    def select(self, prompt, lines, tpl, segs, caps, v, hidden, qseen, st) -> str:
        cal = self.cal["select"]
        shown = list(dict.fromkeys([k for _, k in caps] + [s.key for s in segs]))
        ids = self._table_ids(prompt, caps, v)
        need = self._needed(lines, v, hidden, qseen, st)
        instr = "\n".join(lines).lower()
        base = 1 if re.search(r"1-based|one-based|start(ing)? (at|from|with) 1\b|numbered 1\b", instr) else 0
        if re.search(r"(including|counting) the (row )?label|raw (column|cell) index|label (column|cell) (is|=|as) (index )?0", instr):
            base += 1
        p_meas = cal.get("wrong_measure_by_section", {}).get(v.sf["style"]["section"], cal["wrong_measure"])
        lay = {s.key: s.layout for s in segs}
        facts = []
        for j, (m, a, ci, name) in enumerate(need):
            r = rng_for(prompt, j, "select")
            if r.random() < cal["missing_series"] and len(need) > 1:
                continue
            km, ka = m, a
            x = r.random()
            if x < cal["wrong_adj"]:
                ka = "NSA" if a == "SA" else "SA"
            elif x < cal["wrong_adj"] + p_meas:
                others = [mm for mm in v.measures if mm != m and (not shown or (mm, a) in shown)]
                if others:
                    km = r.choice(others)
            if shown and (km, ka) not in shown:
                alt = [k for k in shown if k[0] == km] or [k for k in shown if k[1] == ka] or shown
                km, ka = alt[0]
                st["guesses"] = st.get("guesses", 0) + 1
            cj = ci
            y = r.random()
            c_off, c_oth = (cal["col_off_multi"], cal["col_other_multi"]) if len(need) > 1 else (cal["col_off"], cal["col_other"])
            if y < c_off:
                step = 1 if r.random() < cal["col_off_up"] else -1
                if 0 <= ci + step < len(v.cols):
                    cj = ci + step
            elif y < c_off + c_oth and len(v.cols) > 3:
                cj = r.choice([c for c in range(len(v.cols)) if c not in (ci, ci - 1, ci + 1)])
            cap = v.tables[(km, ka)]["caption"] if (km, ka) in v.tables else ""
            facts.append({"name": name, "table_id": ids.get((km, ka), cap), "caption": cap, "index": cj + base,
                          "header": v.header_path(cj), "adj": v.adjw[ka], "noun": v.noun[km],
                          "orientation": _orient(lay.get((km, ka), "rows")),
                          "why": f"'{v.noun[km]}' is the measure of the section above '{cap}', whose caption says "
                                 f"'{v.adjw[ka]}'; '{v.disp[cj][1]}' is value cell {cj + base} under '{v.disp[cj][0]}'"})
        top = self._top(prompt, v, hidden, qseen)
        if tpl is not None or re.search(r"\bjson\b", "\n".join(lines), re.I):
            obj = _fill(tpl if isinstance(tpl, (dict, list)) else _DEFAULT_SELECT, facts, top)
            body = json.dumps(obj, indent=2 if rng_for(prompt, "indent").random() < 0.6 else None)
            return self._wrap_json(prompt, body, cal)
        out = []
        for f in facts:
            out.append(f"{f['name']}: table '{f['caption']}'" + (f" (key {f['table_id']!r})" if f["table_id"] != f["caption"] else "")
                       + f", value cell {f['index']} ({f['header']}).")
        out.append(f"Values are printed in {v.unit}: multiply by {top['multiplier']:,} for full units.")
        return self.o.chat(prompt, "\n".join(out))

    def _top(self, prompt: str, v: View, hidden: dict, qseen: bool) -> dict:
        cal = self.cal["select"]
        p = cal.get("mult_wrong_by_scale", {}).get(str(int(v.scale)), cal["mult_wrong"])
        mult = 1 if rng_for(prompt, "mult").random() < p else int(v.scale)
        return {"multiplier": mult, "unit": v.unit, "period_rows": self._period_rows(v), "non_period": self._non_period(v),
                "rules": " ".join(_rule_sentences(hidden.get("question") or "")) if qseen else ""}

    def _table_ids(self, prompt: str, caps: list, v: View) -> dict:
        """Name the prompt uses for each visible table (its key/id field, else its caption)."""
        out: dict = {}
        prev = 0
        for pos, k in caps:
            window = prompt[max(prev, pos - 600):pos]
            hits = re.findall(r'"(?:key|id|table_id|table_key|ref)"\s*:\s*("[^"]*"|\d+)', window)
            if hits and k not in out:
                out[k] = json.loads(hits[-1])
            else:
                line = prompt[prompt.rfind("\n", 0, pos) + 1:pos]
                mm = re.match(r"^\s*[\[(#]?\s*([A-Za-z]?\d{1,3})\s*[\])]?\s*[:.)\-]\s*$", line)
                if mm and k not in out:
                    out[k] = mm.group(1)
            prev = pos
        return out

    def _period_rows(self, v: View) -> str:
        mk = v.mark
        if v.sf["style"]["rows"] == "ym":
            return (f"every row is labelled '<Month> <Year>'; rows of a year revised in this release carry [{mk}]; "
                    "the newest month is flagged (p), the two before it (r)")
        return (f"a year row (e.g. '2020', '2019 {mk}' when that year was revised in this release) followed by month-name "
                "rows; the newest month is flagged (p), the two before it (r)")

    def _non_period(self, v: View) -> list[str]:
        trap = v.sf.get("trap") or []
        out = list(trap[:2]) + ["<Mon. YYYY> from <Mon. YYYY>"] + list(trap[3:4])
        if len(trap) > 2:
            out.append(f"<year> {trap[2]}")
        if v.sf.get("annual", "none") != "none" and v.sf.get("annual_word"):
            out.append(f"{v.sf['annual_word']} <year>")
        return out

    def _wrap_json(self, prompt: str, body: str, cal: dict) -> str:
        r = rng_for(prompt, "json-format")
        if r.random() < cal.get("json_fail", 0.0):
            body = body.rstrip().rstrip("}").rstrip() + ",\n  ...\n"
        if r.random() < cal.get("fence", 0.0):
            body = "```json\n" + body + "\n```"
        return body

    def read(self, prompt, lines, tpl, segs, v, st) -> str:
        """Values of the requested columns for visible rows, with the real 27B's error types (wrong column, dropped rows)."""
        cal = self.cal["read"]
        r = rng_for(prompt, "read")
        key, cols, exact = self._target(lines, segs, v, prompt, st)
        sub = None
        by_col = None
        if isinstance(tpl, dict) and len(tpl) == 1 and _is_period_key(next(iter(tpl))) and isinstance(next(iter(tpl.values())), dict):
            sub = {k: v.fuzzy_col(k) for k in next(iter(tpl.values()))}
            cols = [c for c in sub.values() if c is not None] or cols
        elif isinstance(tpl, dict) and tpl and all(isinstance(x, dict) and x and all(_is_period_key(k) for k in x)
                                                   for x in tpl.values()):
            by_col = {k: v.fuzzy_col(k) for k in tpl}
            cols = [c for c in by_col.values() if c is not None] or cols
        mine = [s for s in segs if s.key == key]
        others = [s for s in segs if s.key != key]
        if not mine:
            if not others or r.random() < cal["absent_empty"]:
                return self._wrap_json(prompt, "{}", cal) if _wants_json(lines, tpl) else self.o.chat(
                    prompt, "N/A -- the excerpt does not contain that table.")
            mine, others = others[:1], others[1:]
        shift = 0
        rs = rng_for(norm("\n".join(lines)), "series")
        if rs.random() < (cal["series_err_exact"] if exact else cal["series_err_inexact"]):
            if others and rs.random() < cal["series_err_table"]:
                mine = [others[0]]
            else:
                right = [d for d in range(1, len(v.cols)) if max(cols) + d < len(v.cols)]
                left = [-d for d in range(1, len(v.cols)) if min(cols) - d >= 0]
                pool = right if right and (rs.random() < cal["series_err_right"] or not left) else left
                shift = rs.choice(pool) if pool else 0
        load = max(0.0, (len(prompt) - LOAD_FREE_CHARS) / max(1, VARIED_BUDGET_CHARS - LOAD_FREE_CHARS))
        boost = 1.0 + 8.0 * load * load
        convert = bool(re.search(r"full units|full (number|value)s?|multiply|convert|not in (thousands|millions)|"
                                 r"absolute (number|units)|unscaled|actual (number|units|values?)", "\n".join(lines), re.I))
        rows = list(dict.fromkeys(ym for seg in mine for ym in seg.rows))
        drop: set[str] = set()
        if rows and r.random() < cal["rows_dropped"] * boost:
            if r.random() < 0.25:
                drop = set(rows)
            else:
                k = max(1, int(len(rows) * r.uniform(0.3, 0.7)))
                drop = set(rows[:k] if r.random() < 0.6 else rows[-k:])
        if r.random() < cal["empty_present"]:
            drop = set(rows)
        res: OrderedDict[int, OrderedDict[str, Any]] = OrderedDict()
        cells_of: dict[str, dict[int, str]] = {}
        for ci in cols:
            src = min(max(0, ci + shift), len(v.cols) - 1)
            out: OrderedDict[str, Any] = OrderedDict()
            for seg in mine:
                dec = v.dec[seg.key]
                for j, ym in enumerate(seg.rows):
                    cell = seg.cells[ym].get(src)
                    if cell is None or ym in drop:
                        continue
                    cells_of[ym] = seg.cells[ym]
                    rr = rng_for(prompt, ym, ci, "cell")
                    if rr.random() < cal["cell_err"] * boost:
                        nb = [seg.cells[ym].get(c) for c in (src - 1, src + 1)] + \
                             [seg.cells[x].get(src) for x in seg.rows[max(0, j - 1):j + 2] if x != ym]
                        nb = [x for x in nb if x]
                        if nb:
                            cell = rr.choice(nb)
                    out[ym] = _value(cell, dec, v.scale if convert else 1.0)
            res[ci] = out
        if not _wants_json(lines, tpl):
            txt = []
            for ci, out in res.items():
                if len(res) > 1:
                    txt.append(f"{v.header_path(ci)}:")
                txt += [f"{ym}: {'S' if x is None else x}" for ym, x in out.items()]
            return self.o.chat(prompt, "\n".join(txt) if txt else "N/A -- no month rows of that column are visible.")
        if sub is not None:
            obj: Any = {ym: {k: (res.get(c) or {}).get(ym) for k, c in sub.items()}
                        for ym in dict.fromkeys(y for o in res.values() for y in o)}
        elif by_col is not None:
            obj = {k: dict(res.get(c) or {}) for k, c in by_col.items()}
        elif isinstance(tpl, dict) and tpl and not all(_is_period_key(k) for k in tpl):
            asked = _asked_month(lines)
            facts = []
            for ci, out in res.items():
                ym = asked if asked in out else (next(iter(out)) if asked is None and len(out) == 1 else asked)
                facts.append({"header": v.header_path(ci), "noun": v.noun[key[0]] if key else "", "index": ci,
                              "values": dict(out), "month": ym, "value": out.get(ym) if ym else None,
                              "row_cells": [c for _, c in sorted((cells_of.get(ym) or {}).items())] if ym else [],
                              "caption": v.tables[key]["caption"] if key in v.tables else "", "table_id": "",
                              "adj": v.adjw[key[1]] if key else "", "name": v.header_path(ci),
                              "orientation": _orient(mine[0].layout),
                              "why": f"read from the '{v.disp[ci][1]}' cells under '{v.disp[ci][0]}'"})
            top = {"multiplier": int(v.scale), "unit": v.unit if v.unit in prompt else "", "period_rows": self._period_rows(v),
                   "non_period": self._non_period(v), "rules": ""}
            obj = _fill(tpl, facts, top)
        else:
            obj = next(iter(res.values())) if len(res) == 1 else {v.header_path(c): o for c, o in res.items()}
        if r.random() < cal["str_values"]:
            obj = json.loads(json.dumps(obj), parse_int=str, parse_float=str)
        body = json.dumps(obj, indent=2 if r.random() < 0.5 else None)
        return self._wrap_json(prompt, body, cal)

    def aggregate(self, prompt, segs, v, hidden, qseen, lines, st) -> str:
        """Approximate aggregate over the visible excerpts; slips by up to 12% past ``agg_capacity`` months."""
        st["agg_requests"] = st.get("agg_requests", 0) + 1
        from .answers import answer_periodic
        r = rng_for(prompt, "agg")
        spec = hidden["spec"] if qseen else None
        series = [tuple(x) for x in spec["series"]] if spec else [
            (m, a, ci) for m, a, ci, _ in self._needed(lines, v, hidden, False, st)][:1]
        mult = v.scale if r.random() < 0.5 else 1.0
        S = []
        for m, a, ci in series:
            vals: dict[str, float] = {}
            for seg in segs:
                if seg.key == (m, a):
                    for ym in seg.rows:
                        cell = seg.cells[ym].get(ci)
                        x = _value(cell, v.dec[seg.key], mult) if cell else None
                        if isinstance(x, (int, float)) and (ym not in vals or r.random() < 0.5):
                            vals[ym] = float(x)
            S.append(dict(sorted(vals.items())))
        if not S or not S[0]:
            nums = [float(x.replace(",", "")) for x in _NUM.findall(prompt[:20000]) if "%" not in x and x.replace(",", "").replace(".", "").isdigit()]
            if not nums:
                return self.o.chat(prompt, "I can't determine that from the text provided.")
            return self.o.chat(prompt, f"Based on the figures shown, the answer is approximately {statistics.median(nums) * r.uniform(0.7, 1.3):,.2f}.")
        ans = None
        if spec:
            try:
                ans = answer_periodic(spec, S, hidden["domain"])
            except Exception:  # noqa: BLE001
                ans = None
        if ans is None:
            ans = sum(S[0].values())
        n = len(S[0])
        exact = n <= self.o.read.agg_capacity
        if isinstance(ans, list):
            if not exact and ans:
                ans = r.sample(ans, max(1, len(ans) // 2)) + r.sample(list(S[0]), min(2, len(S[0])))
            return self.o.chat(prompt, "The answer is: " + json.dumps(sorted(set(ans))))
        if isinstance(ans, (int, float)):
            x = float(ans) * (1.0 if exact else r.uniform(0.88, 1.12))
            return self.o.chat(prompt, f"The answer is approximately {x:,.2f} (from the {n} months shown).")
        return self.o.chat(prompt, f"The answer is: {ans}")

    def _unit_text(self, prompt: str, v: View, tpl: Any) -> str:
        shown = v.unit in prompt
        if isinstance(tpl, dict):
            return json.dumps(_fill(tpl, [], {"multiplier": int(v.scale), "unit": v.unit if shown else None,
                                              "period_rows": self._period_rows(v), "non_period": self._non_period(v),
                                              "rules": ""}))
        if not shown:
            return "The excerpt does not state the unit; such releases usually print values in thousands."
        return f"The values are printed in {v.unit!r}: multiply the printed figures by {int(v.scale):,} for full units."

    def _dating(self, prompt: str, segs: list[Seg], v: View) -> str:
        rel = []
        for s in segs:
            rs = releases_of(s, v)
            if len(rs) == 1:
                rel.append(v.reports[rs[0]]["report"])
        m = re.search(r"(?:Issued|issued|Published:?|published)\s+([A-Z][a-z]+ \d{1,2}, \d{4})", prompt)
        when = f"; it was issued {m.group(1)}" if m else ""
        if rel:
            ym = max(rel)
            return f"The release covers {MONTHS[int(ym[5:]) - 1]} {ym[:4]} (its newest month, flagged (p)){when}."
        mm = re.search(r"(?:period|through|Reference period:|--)\s*(\d{4}-\d{2}|[A-Z][a-z]+ \d{4})", prompt)
        if mm:
            return f"The release is for {mm.group(1)}{when}."
        return "The excerpt does not show which period it covers" + (when or "") + "."

    def _revision(self, segs: list[Seg], v: View) -> str:
        mk = v.mark
        yrs = []
        for s in segs:
            rs = releases_of(s, v)
            if len(rs) == 1:
                yrs += v.reports[rs[0]].get("revised", [])
        s = (f"The newest month is marked (p) (preliminary) and the two months before it (r) (revised); a year whose "
             f"rows were revised in this release's annual revision is marked {mk!r}.")
        if yrs:
            s += f" Here that is {sorted(set(yrs))[0]}."
        return s

    def _describe(self, prompt, segs, caps, v, tpl, brief: bool) -> str:
        keys = list(dict.fromkeys([k for _, k in caps] + [s.key for s in segs]))
        parts = []
        for k in keys[:12]:
            m, a = k
            info = v.tables[k]
            parts.append(f"'{info['caption']}': {v.noun[m]}, {v.adjw[a]}")
        cols = ", ".join(f"{G} ({', '.join(S for g2, S in v.disp if g2 == G)})" for G in v.groups)
        if not parts:
            return "The text contains no statistical table, only narrative text."
        s = "The excerpt contains " + ("these tables: " if len(parts) > 1 else "the table ") + "; ".join(parts) + "."
        if not brief:
            s += (f" Columns are grouped as {cols}; the first column after the row label is {v.header_path(0)}. "
                  f"Rows: {self._period_rows(v)}. Values are in {v.unit}. Rows that are not single months: "
                  + ", ".join(self._non_period(v)) + ".")
        if isinstance(tpl, dict):
            return json.dumps({k2: s for k2 in tpl})
        return s


_DEFAULT_SELECT = {"series": [{"name": "", "table": "", "value_index": 0, "header_path": ""}], "multiplier": 0,
                   "unit_text": ""}


def _orient(layout: str) -> str:
    return {"transposed": "rows are series, columns are months", "lines": "one line per series, months in the text"}.get(
        layout, "rows are months, columns are series")


def _asked_month(lines: list[str]) -> str | None:
    """The single period a request names ("the row for 2015-12", "the May 2023 value"), if any."""
    t = "\n".join(lines)
    m = re.findall(r"\b((?:19|20)\d\d)-(\d\d)\b", t)
    if len(set(m)) == 1:
        return f"{m[0][0]}-{m[0][1]}"
    m2 = re.findall(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+((?:19|20)\d\d)\b", t)
    if len(set(m2)) == 1:
        return f"{m2[0][1]}-{MONTHS.index(m2[0][0]) + 1:02d}"
    return None


def _empty(tpl: Any, month: str | None) -> Any:
    """A template answered with nothing found (the excerpt holds no such table / row)."""
    if isinstance(tpl, dict):
        return {k: (month if re.fullmatch(r"(month|period|date|ym)", str(k), re.I) and month else _empty(x, month))
                for k, x in tpl.items() if not _is_period_key(str(k))}
    if isinstance(tpl, list):
        return []
    return None


def _key_words(key: str) -> list[str]:
    return norm(re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace("_", " ")).split()


def _fill(tpl: Any, facts: list[dict], top: dict) -> Any:
    """Fill a JSON template key by key, matching each key to the series it names."""
    def named(key: str, cur: dict | None) -> dict | None:
        kw = [w for w in _key_words(key) if len(w) >= 4]
        best, bs = cur, 0
        for f in facts:
            toks = set(norm(f"{f.get('noun', '')} {f.get('header', '')}").split())
            sc = sum(1 for w in kw for u in toks if len(u) >= 4 and (u.startswith(w[:5]) or w.startswith(u[:5])))
            if sc > bs:
                best, bs = f, sc
        return best

    def val(key: str, ph: Any, f: dict | None) -> Any:
        k = "_".join(_key_words(key))
        src = f if f is not None else (facts[0] if facts else {})
        if re.search(r"non_?period|exclude|skip|ignore|other_?rows", k):
            return top["non_period"]
        if re.search(r"orient|direction|layout", k):
            return src.get("orientation")
        if re.search(r"row_?cells|^cells$|^row$|neighbo", k):
            return src.get("row_cells")
        if re.search(r"^(values|data|series_values|months|monthly)$", k):
            return src.get("values")
        if re.search(r"^(month|period|date|ym)$", k) and "month" in src:
            return src.get("month")
        if re.search(r"period|row_?(label|format)|date_?format", k):
            return top["period_rows"]
        if re.search(r"header|path|column_?(name|label)|^column$|^col$", k):
            return src.get("header")
        if re.search(r"index|idx|cell|position|locator|col_?(no|num)|column_?(no|num)", k):
            return src.get("index")
        if re.search(r"^(value|number|printed|figure|printed_value)$", k):
            return src.get("value")
        if re.search(r"mult|scale|factor", k):
            return top["multiplier"]
        if "unit" in k:
            return top["unit"]
        if re.search(r"key|table|caption|^id$|_id$", k):
            return src.get("caption") if "caption" in k else src.get("table_id")
        if "adj" in k:
            return src.get("adj")
        if re.search(r"measure|noun|variable|metric", k):
            return src.get("noun")
        if re.search(r"why|reason|evidence|justif|explan|quote|note", k):
            return src.get("why")
        if re.search(r"rule|definition|condition", k):
            return top["rules"]
        if re.search(r"name|series|what|label|description", k):
            return src.get("name")
        return None if isinstance(ph, (int, float)) else ""

    def walk(t: Any, f: dict | None) -> Any:
        if isinstance(t, list):
            if t and isinstance(t[0], dict):
                return [walk(t[0], fx) for fx in facts] if f is None else [walk(t[0], f)]
            return t
        if not isinstance(t, dict):
            return t
        out = {}
        for key, x in t.items():
            if isinstance(x, list) and x and isinstance(x[0], dict):
                out[key] = walk(x, f)
            elif isinstance(x, dict):
                out[key] = walk(x, named(str(key), f))
            else:
                out[key] = val(str(key), x, named(str(key), f))
        return out
    if isinstance(tpl, dict) and len(facts) > 1 and not any(isinstance(x, (list, dict)) for x in tpl.values()) and \
            any(re.search(r"index|key|table|header", str(k), re.I) for k in tpl) and \
            len({id(named(str(k), None)) for k in tpl if named(str(k), None) is not None}) < 2:
        return [walk(tpl, f) for f in facts]
    return walk(tpl, None)


def _rule_sentences(q: str) -> list[str]:
    sents = [s.strip() for s in re.split(r"(?<=[.?!])\s+", q) if s.strip()]
    return [s for s in sents if not s.endswith("?")][:4]


def _wants_json(lines: list[str], tpl: Any) -> bool:
    return tpl is not None or bool(re.search(r"\bjson\b", "\n".join(lines), re.I))


def _value(cell: str | None, dec: int, mult: float) -> Any:
    if cell is None or cell in ("S", "X", "(S)", "(X)"):
        return None
    try:
        x = float(cell.replace(",", "").replace("±", "").rstrip("%"))
    except ValueError:
        return None
    x *= mult
    return int(round(x)) if (dec == 0 or mult != 1.0) and abs(x - round(x)) < 1e-6 else round(x, max(dec, 1))


def _seen(q: str, prompt: str) -> bool:
    """The question (most of its word 4-grams) is in the prompt."""
    w = norm(q).split()
    if len(w) < 8:
        return norm(q) in norm(prompt)
    grams = {" ".join(w[i:i + 4]) for i in range(len(w) - 3)}
    np_ = norm(prompt)
    return sum(1 for g in grams if g in np_) >= 0.6 * len(grams)


def _has_doc_text(prompt: str, resid: str) -> bool:
    return len(prompt) - len(resid) > 200


__all__ = ["CALIBRATION", "VARIED_BUDGET_CHARS", "VariedReader", "View", "caption_key", "is_varied", "view_for",
           "visible"]
