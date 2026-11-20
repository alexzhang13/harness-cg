"""Value types for the synthetic ExpertLongBench env: name pools, renderers (several surface forms per value) and the
normalizers the deterministic scorer uses. Pure functions, no state."""

from __future__ import annotations

import datetime as dt
import re

FIRST = ["Alicia", "Marcus", "Denise", "Harold", "Priya", "Tomas", "Gloria", "Wendell", "Rosa", "Kenji", "Lorraine",
         "Darnell", "Ingrid", "Victor", "Shonda", "Miguel", "Esther", "Clayton", "Naomi", "Russell", "Fatima", "Brendan",
         "Yolanda", "Stuart", "Mei", "Orlando", "Beatrice", "Gideon", "Tamara", "Ellis", "Constance", "Rafael", "Odessa",
         "Warren", "Lucinda", "Desmond", "Hannah", "Arturo", "Celeste", "Malcolm", "Ruth", "Emeka", "Sylvia", "Grant",
         "Imani", "Leopold", "Corinne", "Dmitri", "Paulette", "Augustin"]
LAST = ["Abernathy", "Okafor", "Lindqvist", "Marchetti", "Haverford", "Nakamura", "Delacroix", "Whitcomb", "Ostrowski",
        "Pemberton", "Quintero", "Rasmussen", "Sorensen", "Thibodeaux", "Underhill", "Valdivia", "Wexler", "Yarbrough",
        "Zielinski", "Ashworth", "Brannigan", "Castellano", "Dunmore", "Eberhardt", "Fairbanks", "Galloway", "Holloway",
        "Iverson", "Jablonski", "Kowalczyk", "Lachance", "Montgomery", "Nightingale", "Oyelaran", "Prescott", "Quimby",
        "Rutherford", "Stanislaw", "Treadwell", "Umberger", "Vanterpool", "Winslow", "Xiong", "Youngblood", "Zamora",
        "Beauregard", "Cartwright", "Drummond", "Esparza", "Fitzwilliam", "Granger", "Hargrove", "Ingersoll", "Jessup",
        "Kirkpatrick", "Lomax", "Merriweather", "Northcutt", "Pendergast", "Ravenscroft", "Albright", "Bellamy",
        "Corcoran", "Dalrymple", "Ellsworth", "Fennimore", "Gutierrez", "Hathaway", "Isley", "Jorgensen", "Kessler",
        "Lindgren", "Mulroney", "Nakashima", "Ortega", "Petrakis", "Rourke", "Sandoval", "Tolliver", "Ulrich",
        "Vasquez", "Whitaker", "Yamamoto", "Zeller", "Ambrose", "Blackwood", "Crenshaw", "Delgado", "Eastman",
        "Fontaine", "Gallagher", "Hendricks", "Imbriano", "Kaczmarek"]
CITIES = ["Riverton", "Maple Falls", "Port Jessamine", "Carver City", "Elk Hollow", "Bramblewood", "Harlow Springs",
          "Ironbridge", "Lakemont", "Stonecrest", "Willow Bend", "Fort Callum", "Greyhaven", "Ashby", "Oakridge Point",
          "Summerton", "Pine Bluff Crossing", "Northgate", "Redmond Heights", "Copper Valley"]
COUNTIES = ["Fulmer", "Jasper", "Calloway", "Brandt", "Okeechobee", "Tillman", "Garrison", "Halstead", "Mercer",
            "Pruitt", "Sheridan", "Vance", "Whitfield", "Yancey", "Dorsey", "Kendrick", "Lassiter", "Ramsey"]
STATES = [("Georgia", 11), ("Texas", 5), ("California", 9), ("New York", 2), ("Illinois", 7), ("Ohio", 6),
          ("Pennsylvania", 3), ("Massachusetts", 1), ("Louisiana", 5), ("Michigan", 6), ("Arizona", 9), ("Colorado", 10),
          ("Florida", 11), ("Virginia", 4), ("Washington", 9), ("Missouri", 8), ("Alabama", 11), ("Tennessee", 6),
          ("Oregon", 9), ("Wisconsin", 7), ("Maryland", 4), ("Minnesota", 8), ("Nevada", 9), ("Kentucky", 6)]
ORDINAL = {1: "First", 2: "Second", 3: "Third", 4: "Fourth", 5: "Fifth", 6: "Sixth", 7: "Seventh", 8: "Eighth",
           9: "Ninth", 10: "Tenth", 11: "Eleventh"}
NUMWORD = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
           10: "ten", 11: "eleven", 12: "twelve"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
_MON = {m.lower(): i + 1 for i, m in enumerate(MONTHS)}
_MON.update({m[:3].lower(): i + 1 for i, m in enumerate(MONTHS)})
_MON["sept"] = 9


def iso(d: dt.date) -> str:
    return d.isoformat()


def from_iso(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def fmt_date(d: dt.date, style: int) -> str:
    style %= 4
    if style == 0:
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == 1:
        return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if style == 2:
        return d.isoformat()
    return f"{MONTHS[d.month - 1][:3]}. {d.day}, {d.year}"


_DATE_RES = [
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b"), "mdy"),
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), "ymd"),
    (re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"), "Mdy"),
    (re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Za-z]{3,9}),?\s+(\d{4})\b"), "dMy"),
]


def find_dates(s: str) -> list[str]:
    """Every date in ``s`` (several formats) as ISO strings, in order of appearance."""
    out: list[tuple[int, str]] = []
    for rx, kind in _DATE_RES:
        for m in rx.finditer(s or ""):
            try:
                if kind == "mdy":
                    y = int(m.group(3))
                    y += 2000 if y < 50 else (1900 if y < 100 else 0)
                    d = dt.date(y, int(m.group(1)), int(m.group(2)))
                elif kind == "ymd":
                    d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                elif kind == "Mdy":
                    mon = _MON.get(m.group(1).lower())
                    if not mon:
                        continue
                    d = dt.date(int(m.group(3)), mon, int(m.group(2)))
                else:
                    mon = _MON.get(m.group(2).lower())
                    if not mon:
                        continue
                    d = dt.date(int(m.group(3)), mon, int(m.group(1)))
            except ValueError:
                continue
            out.append((m.start(), d.isoformat()))
    out.sort()
    seen, res = set(), []
    for _, x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


_TITLES = {"dr", "hon", "honorable", "judge", "mr", "ms", "mrs", "jr", "sr", "ii", "iii", "iv", "esq", "md", "phd",
           "magistrate", "district", "chief", "the", "monitor", "special", "master", "u", "s", "us", "of"}


def name_tokens(s: str) -> set[str]:
    toks = re.findall(r"[a-z]+", (s or "").lower())
    return {t for t in toks if len(t) > 1 and t not in _TITLES}


def fmt_person(p: list, style: int, title: str = "") -> str:
    first, mid, last = p
    style %= 3
    if style == 0:
        core = f"{first} {mid}. {last}" if mid else f"{first} {last}"
    elif style == 1:
        core = f"{first} {last}"
    else:
        core = f"{first} {mid}. {last}" if mid else f"{first} {last}"
        core = core.upper() if title == "" else core
    return f"{title} {core}".strip()


def person_match(pred: str, gold: list) -> bool:
    t = name_tokens(pred)
    return gold[0].lower() in t and gold[2].lower() in t


def fmt_amount(a: int, style: int) -> str:
    style %= 3
    if style == 0:
        return f"${a:,}"
    if style == 1 and a % 10_000 == 0:
        return f"${a / 1_000_000:g} million"
    return f"{a:,} dollars"


def find_amounts(s: str) -> list[float]:
    out = []
    for m in re.finditer(r"\$?\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(million|m\b|thousand|k\b)?", s or "", re.I):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit in ("million", "m"):
            v *= 1_000_000
        elif unit in ("thousand", "k"):
            v *= 1_000
        if v >= 1000:
            out.append(v)
    return out


_WORDNUM = {w: n for n, w in NUMWORD.items()}
_WORDNUM.update({"eighteen": 18, "twenty-four": 24, "thirty-six": 36, "a": 1, "an": 1, "half": 0.5})


def fmt_duration(months: int, style: int) -> str:
    style %= 3
    if months % 12 == 0:
        y = months // 12
        if style == 0:
            return f"{NUMWORD.get(y, y)} ({y}) years"
        if style == 1:
            return f"{y} years"
        return f"{months} months"
    return f"{months} months" if style != 1 else f"{NUMWORD.get(months, months)} months"


def find_durations(s: str) -> list[float]:
    out = []
    for m in re.finditer(r"\b(\d+(?:\.\d+)?|[a-z]+(?:-[a-z]+)?)\s*(?:\(\d+\)\s*)?[- ]?(years?|months?)\b", (s or "").lower()):
        tok, unit = m.group(1), m.group(2)
        try:
            n = float(tok)
        except ValueError:
            n = _WORDNUM.get(tok)
            if n is None:
                continue
        out.append(n * 12 if unit.startswith("year") else n)
    return out


def fmt_caseno(c: list, style: int, ini: str = "") -> str:
    div, yy, num = c
    style %= 3
    if style == 0:
        return f"{div}:{yy:02d}-cv-{num:05d}" + (f"-{ini}" if ini else "")
    if style == 1:
        return f"{div}:{yy:02d}-cv-{num}"
    return f"No. {div}:{yy:02d}-CV-{num:05d}"


def find_casenos(s: str) -> set[str]:
    out = set()
    for m in re.finditer(r"(\d)\s*:\s*(\d{2})\s*[-–]\s*c[vr]\s*[-–]\s*0*(\d{1,5})", s or "", re.I):
        out.add(f"{m.group(1)}:{m.group(2)}-cv-{int(m.group(3))}")
    return out


def fmt_cite(c: list) -> str:
    kind = c[0]
    if kind == "wl":
        return f"{c[1]} WL {c[2]}"
    return f"{c[1]} {c[2]} {c[3]}"


def find_cites(s: str) -> set[str]:
    out = set()
    for m in re.finditer(r"(\d{4})\s*W\.?\s?L\.?\s*(\d{4,8})", s or ""):
        out.add(f"wl{m.group(1)}-{int(m.group(2))}")
    for m in re.finditer(r"(\d{1,4})\s+F\.\s*(Supp\.\s*)?(\d(?:d|th))?\s*(\d{1,4})\b", s or ""):
        rep = "fsupp" if m.group(2) else "f"
        out.add(f"{int(m.group(1))}{rep}{(m.group(3) or '').lower()}-{int(m.group(4))}")
    return out


def cite_key(c: list) -> str:
    return next(iter(find_cites(fmt_cite(c))))


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def token_overlap(pred: str, gold: str) -> float:
    g = set(norm(gold).split()) - {"the", "of", "and", "for", "at", "in"}
    p = set(norm(pred).split())
    return len(g & p) / max(1, len(g))


_WRAP: dict[int, re.Pattern] = {}


def wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap (PDF-like short lines); regex-based, ~10x faster than a Python word loop on 1M-char files."""
    rx = _WRAP.get(width)
    if rx is None:
        rx = _WRAP[width] = re.compile(rf"(\S.{{0,{width - 1}}})(?: +|$)")
    return [x.rstrip() for x in rx.findall(text)]
