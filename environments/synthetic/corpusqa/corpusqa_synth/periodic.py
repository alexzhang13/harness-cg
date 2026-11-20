"""Periodic-report family: a corpus of monthly statistical releases (one document per release)."""

from __future__ import annotations

import math
import random
import statistics
from typing import Any

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ABBR = ["Jan.", "Feb.", "Mar.", "Apr.", "May", "June", "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."]
WEEKDAYS = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"]

DOMAINS: dict[str, dict[str, Any]] = {
    "housing": {
        "release": "MONTHLY NEW RESIDENTIAL CONSTRUCTION",
        "agency": "U.S. Census Bureau and the U.S. Department of Housing and Urban Development",
        "unit": "Thousands of Units. Detail may not add to total because of rounding.",
        "scale": 1000,
        "groups": [("United States", ["Total", "1 unit", "2 to 4 units", "5 units or more"]),
                   ("Northeast", ["Total", "1 unit"]), ("Midwest", ["Total", "1 unit"]),
                   ("South", ["Total", "1 unit"]), ("West", ["Total", "1 unit"])],
        "group_kind": "region",
        "shares": {"1 unit": 0.68, "2 to 4 units": 0.035, "5 units or more": 0.285,
                   "Northeast": 0.09, "Midwest": 0.14, "South": 0.53, "West": 0.24},
        "measures": [
            {"key": "permits", "title": "New Privately-Owned Housing Units Authorized in Permit-Issuing Places",
             "q": "housing units authorized by building permits", "head": "Building Permits",
             "lead": "Privately-owned housing units authorized by building permits", "flow": True, "level": (1250, 1750)},
            {"key": "authorized", "title": "New Privately-Owned Housing Units Authorized, but Not Started, at End of Period",
             "q": "housing units authorized but not started", "head": "Authorized but Not Started",
             "lead": "Privately-owned housing units authorized but not started", "flow": False, "level": (250, 300),
             "q_alt": ["housing authorizations", "housing units authorized"]},
            {"key": "starts", "title": "New Privately-Owned Housing Units Started", "q": "housing starts",
             "head": "Housing Starts", "lead": "Privately-owned housing starts", "flow": True, "level": (1150, 1650)},
            {"key": "under_construction", "title": "New Privately-Owned Housing Units Under Construction at End of Period",
             "q": "housing units under construction", "head": "Under Construction",
             "lead": "Privately-owned housing units under construction", "flow": False, "level": (1450, 1800)},
            {"key": "completions", "title": "New Privately-Owned Housing Units Completed", "q": "housing completions",
             "head": "Housing Completions", "lead": "Privately-owned housing completions", "flow": True,
             "level": (1150, 1650)},
        ],
        "suppressed": {("starts", "2 to 4 units"), ("completions", "2 to 4 units"),
                       ("under_construction", "2 to 4 units"), ("authorized", "2 to 4 units")},
        "bench_month": 4,
        "nsa_decimals": 1,
        "annualized": True,
        "window": 13,
    },
    "retail": {
        "release": "ADVANCE MONTHLY SALES FOR RETAIL AND FOOD SERVICES",
        "agency": "U.S. Census Bureau",
        "unit": "Millions of Dollars. Estimates are shown in millions of dollars and are based on data from the Monthly Retail Trade Survey.",
        "scale": 1_000_000,
        "groups": [("Kind of business", ["Retail and food services, total", "Motor vehicle and parts dealers",
                                         "Food services and drinking places", "Nonstore retailers",
                                         "General merchandise stores", "Gasoline stations"])],
        "group_kind": "kind of business",
        "shares": {"Motor vehicle and parts dealers": 0.19, "Food services and drinking places": 0.13,
                   "Nonstore retailers": 0.17, "General merchandise stores": 0.105, "Gasoline stations": 0.08},
        "measures": [
            {"key": "sales", "title": "Estimated Monthly Sales for Retail and Food Services, by Kind of Business",
             "q": "sales", "head": "Retail Sales", "lead": "Advance estimates of U.S. retail and food services sales",
             "flow": True, "level": (640_000, 760_000)},
            {"key": "inventories", "title": "Estimated Monthly Retail Inventories, by Kind of Business",
             "q": "end-of-month inventories", "head": "Retail Inventories",
             "lead": "Estimated end-of-month retail inventories", "flow": False, "level": (760_000, 840_000)},
            {"key": "ecommerce", "title": "Estimated Monthly E-commerce Sales, by Kind of Business",
             "q": "e-commerce sales", "head": "E-commerce", "lead": "Estimated retail e-commerce sales",
             "flow": True, "level": (90_000, 130_000)},
        ],
        "suppressed": {("ecommerce", "Gasoline stations")},
        "bench_month": 5,
        "nsa_decimals": 0,
        "annualized": False,
        "window": 7,
    },
    "jobs": {
        "release": "JOB OPENINGS AND LABOR TURNOVER",
        "agency": "U.S. Bureau of Labor Statistics",
        "unit": "Levels in thousands. Detail may not add to totals because of independent seasonal adjustment.",
        "scale": 1000,
        "groups": [("Total nonfarm", ["Total"]),
                   ("Industry", ["Construction", "Manufacturing", "Retail trade", "Health care"]),
                   ("Region", ["Northeast", "South", "Midwest", "West"])],
        "group_kind": "region",
        "shares": {"Construction": 0.045, "Manufacturing": 0.075, "Retail trade": 0.10, "Health care": 0.18,
                   "Northeast": 0.17, "South": 0.39, "Midwest": 0.20, "West": 0.24},
        "measures": [
            {"key": "openings", "title": "Job openings levels by industry and region", "q": "job openings",
             "head": "Job Openings", "lead": "The number of job openings", "flow": False, "level": (7_200, 9_400)},
            {"key": "hires", "title": "Hires levels by industry and region", "q": "hires", "head": "Hires",
             "lead": "The number of hires", "flow": True, "level": (5_200, 6_400)},
            {"key": "quits", "title": "Quits levels by industry and region", "q": "quits", "head": "Quits",
             "lead": "The number of quits", "flow": True, "level": (3_000, 3_900)},
            {"key": "layoffs", "title": "Layoffs and discharges levels by industry and region",
             "q": "layoffs and discharges", "head": "Layoffs and Discharges", "lead": "Layoffs and discharges",
             "flow": True, "level": (1_450, 1_900)},
        ],
        "suppressed": set(),
        "bench_month": 3,
        "nsa_decimals": 0,
        "annualized": False,
        "window": 5,
    },
}

ADJ_TITLE = {
    ("SA", True, True): "Seasonally adjusted annual rate", ("SA", True, False): "Seasonally adjusted",
    ("SA", False, True): "Seasonally adjusted", ("SA", False, False): "Seasonally adjusted",
    ("NSA", True, True): "Not seasonally adjusted", ("NSA", True, False): "Not seasonally adjusted",
    ("NSA", False, True): "Not seasonally adjusted", ("NSA", False, False): "Not seasonally adjusted",
}

FILLER = [
    "Estimates in this release are based on a sample survey and are subject to both sampling and nonsampling error.",
    "Changes from the previous month that are not statistically significant are marked with an asterisk.",
    "Seasonal adjustment removes the effects of normal seasonal variation, holiday and trading-day differences.",
    "Preliminary estimates for the most recent month are revised in each of the following two releases.",
    "Revisions to earlier months reflect late survey responses and updated seasonal factors.",
    "The 90 percent confidence interval includes the true change with a probability of about 90 percent.",
    "Average relative standard errors are computed over the latest six-month period.",
    "Year-to-date totals reflect revisions that are not distributed to individual months.",
    "Data users who create their own estimates using these tables should cite the agency as the source of the inputs.",
    "Nonresponse is handled by imputation, using patterns observed for respondents in the same cell.",
    "The sample was redesigned and new sample units were phased in over several months.",
    "Annual benchmark revisions are published with the release for the first month of each calendar year.",
    "A description of the survey methodology, including the estimation of totals, is available online.",
    "Detail may not add to totals because of independent rounding and seasonal adjustment of components.",
    "An estimate is suppressed when it does not meet publication standards for reliability.",
    "Tests for identifiable and stable seasonality determine whether a series is seasonally adjusted.",
    "Coverage of the survey includes all permit-issuing places as well as a sample of non-permit areas.",
    "Users should exercise caution when comparing month-to-month changes because of sampling variability.",
]

SEASONAL = {
    "housing": [0.78, 0.80, 1.00, 1.08, 1.12, 1.12, 1.08, 1.07, 1.00, 1.02, 0.92, 0.85],
    "retail": [0.90, 0.88, 1.01, 0.99, 1.04, 1.01, 1.01, 1.02, 0.96, 1.00, 1.02, 1.17],
    "jobs": [0.95, 0.97, 1.02, 1.03, 1.05, 1.06, 1.04, 1.02, 0.99, 0.98, 0.96, 0.92],
}
REV_SD = {0: 0.04, 1: 0.02, 2: 0.008}


def ym_add(ym: tuple[int, int], k: int) -> tuple[int, int]:
    t = ym[0] * 12 + (ym[1] - 1) + k
    return (t // 12, t % 12 + 1)


def ym_str(ym: tuple[int, int]) -> str:
    return f"{ym[0]:04d}-{ym[1]:02d}"


def columns(dom: dict) -> list[tuple[str, str]]:
    return [(g, s) for g, subs in dom["groups"] for s in subs]


def _fmt(v: float, dec: int) -> str:
    return f"{v:,.{dec}f}"


class Corpus:
    """Truth series and every report's printed values. ``printed[(rep_idx, mkey, adj)][ym] = [value per column]``."""

    def __init__(self, rng: random.Random, dom_key: str, n_reports: int, measures: list[dict], window: int,
                 start: tuple[int, int], surf: Any = None):
        self.dom_key = dom_key
        self.dom = surf.dom if surf is not None else DOMAINS[dom_key]
        self.surf = surf
        self.bench_month = surf.bench if surf is not None else self.dom.get("bench_month")
        self.cols = columns(self.dom)
        self.measures = measures
        self.window = window
        self.reports = [ym_add(start, i) for i in range(n_reports)]
        first = ym_add(self.reports[0], -(window - 1))
        self.months = [ym_add(first, i) for i in range(n_reports + window - 1)]
        seas = SEASONAL[dom_key]
        self.truth: dict[tuple[str, str], dict[tuple[int, int], list[float]]] = {}
        for m in measures:
            lo, hi = m["level"]
            base = rng.uniform(lo, hi)
            trend = rng.uniform(-0.006, 0.006)
            ar = 0.0
            col_noise = [0.0] * len(self.cols)
            col_bias = [rng.uniform(0.85, 1.15) for _ in self.cols]
            sa_rows: dict[tuple[int, int], list[float]] = {}
            nsa_rows: dict[tuple[int, int], list[float]] = {}
            for t, ym in enumerate(self.months):
                ar = 0.6 * ar + rng.gauss(0, 0.035)
                total = base * (1 + trend) ** t * (1 + ar)
                srow, nrow = [], []
                for ci, (g, s) in enumerate(self.cols):
                    col_noise[ci] = 0.5 * col_noise[ci] + rng.gauss(0, 0.04)
                    share = self.dom["shares"].get(s, 1.0) * self.dom["shares"].get(g, 1.0)
                    if (g, s) != self.cols[0]:
                        share *= col_bias[ci]
                    v = total * share * (1 + col_noise[ci])
                    srow.append(v)
                    sf = seas[ym[1] - 1] * (1 + rng.gauss(0, 0.03))
                    nv = v * sf
                    if m["flow"] and self.dom["annualized"]:
                        nv = nv / 12.0
                    nrow.append(nv)
                sa_rows[ym] = srow
                nsa_rows[ym] = nrow
            self.truth[(m["key"], "SA")] = sa_rows
            self.truth[(m["key"], "NSA")] = nsa_rows
        self.rev_rng = random.Random(rng.random())
        self._rows: dict = {}
        self._cells: dict = {}
        self._tables: dict = {}
        self._supp: dict = {}

    def decimals(self, adj: str, m: dict) -> int:
        if adj == "NSA" and m["flow"]:
            return self.dom["nsa_decimals"]
        return 0

    def vintage(self, rep: tuple[int, int], ym: tuple[int, int]) -> int:
        """Number of annual revisions month ``ym`` has had by release ``rep``."""
        bm = self.bench_month
        if not bm:
            return 0
        return sum(1 for b in range(ym[0] + 1, rep[0] + 1) if (b, bm) <= rep)

    def revised_years(self, rep: tuple[int, int]) -> set[int]:
        """Years whose rows carry "(r)" in release ``rep``: those revised by this year's annual revision."""
        bm = self.bench_month
        return {rep[0] - 1} if bm and rep[1] >= bm else set()

    def printed_row(self, m: dict, adj: str, ym: tuple[int, int], age: int, vint: int = 0) -> list[float | None]:
        """Printed values of a month row at a given age (0 = preliminary, >= 3 = final) and revision vintage."""
        a = min(age, 3)
        key = (m["key"], adj, ym, a, vint)
        row = self._rows.get(key)
        if row is None:
            dec = self.decimals(adj, m)
            supp = self._supp.get(m["key"])
            if supp is None:
                supp = self._supp[m["key"]] = [(m["key"], s) in self.dom["suppressed"] for _, s in self.cols]
            g = self.rev_rng.gauss
            if vint:
                prev = self.printed_row(m, adj, ym, age, vint - 1)
                row = [None if v is None else round(v * (1 + g(0, 0.012)), dec) for v in prev]
            else:
                truth = self.truth[(m["key"], adj)][ym]
                sd = REV_SD.get(a, 0.0)
                row = [None if supp[ci] else round(truth[ci] * (1 + (g(0, sd) if sd else 0.0)), dec)
                       for ci in range(len(self.cols))]
            self._rows[key] = row
        return row

    def cells(self, m: dict, adj: str, ym: tuple[int, int], age: int, vint: int = 0) -> list[str]:
        key = (m["key"], adj, ym, min(age, 3), vint)
        c = self._cells.get(key)
        if c is None:
            dec = self.decimals(adj, m)
            c = self._cells[key] = ["S" if v is None else f"{v:,.{dec}f}" for v in self.printed_row(m, adj, ym, age, vint)]
        return c

    def printed_table(self, r: int, m: dict, adj: str) -> dict[tuple[int, int], list[float | None]]:
        """Table values (printed units) for measure m / adj in report r."""
        key = (r, m["key"], adj)
        t = self._tables.get(key)
        if t is None:
            rep = self.reports[r]
            t = {}
            for k in range(self.window - 1, -1, -1):
                ym = ym_add(rep, -k)
                t[ym] = self.printed_row(m, adj, ym, k, self.vintage(rep, ym))
            self._tables[key] = t
        return t


def _month_label(ym: tuple[int, int], rep: tuple[int, int]) -> str:
    age = (rep[0] * 12 + rep[1]) - (ym[0] * 12 + ym[1])
    flag = " (p)" if age == 0 else (" (r)" if age in (1, 2) else "")
    return MONTHS[ym[1] - 1] + flag


def _cell(v: float | None, dec: int) -> str:
    return "S" if v is None else _fmt(v, dec)


def _table_rows(corp: Corpus, rep: tuple[int, int], rows: dict, dec: int, adj: str, rng: random.Random, m: dict):
    """Yield ('year', label) / ('month', label, cells) / ('extra', label, cells) in print order."""
    out = []
    year = None
    nrows = len(rows)
    revised = corp.revised_years(rep)
    sf = corp.surf
    varied = sf is not None and not sf.native
    ym_rows = varied and sf.style["rows"] == "ym"
    annual: dict[int, list[str]] = {}
    if varied and sf.annual != "none" and adj == "NSA" and m["flow"]:
        for y in {ym[0] for ym in rows}:
            yr = [rows.get((y, mo)) for mo in range(1, 13)]
            if all(r is not None for r in yr):
                annual[y] = ["S" if any(r[ci] is None for r in yr) else _fmt(sum(r[ci] for r in yr), dec)
                             for ci in range(len(corp.cols))]
    annual_rows = bool(annual) and (sf.annual == "row" or ym_rows)
    for k, ym in enumerate(rows):
        cells = corp.cells(m, adj, ym, nrows - 1 - k, corp.vintage(rep, ym))
        if ym_rows:
            lab = f"{MONTHS[ym[1] - 1]} {ym[0]}" + _month_label(ym, rep)[len(MONTHS[ym[1] - 1]):]
            out.append(("month", lab + (f" [{sf.style['revmark']}]" if ym[0] in revised else ""), cells))
            if annual_rows and ym[1] == 12 and ym[0] in annual:
                out.append(("extra", f"{sf.annual_word} {ym[0]}", annual[ym[0]]))
            continue
        if ym[0] != year:
            year = ym[0]
            lab = (sf.year_label(year, year in revised) if varied else
                   str(year) + (" (r)" if year in revised else ""))
            out.append(("year", lab, annual.get(year) if varied and sf.annual == "yearrow" else None))
        out.append(("month", _month_label(ym, rep), cells))
        if annual_rows and ym[1] == 12 and ym[0] in annual:
            out.append(("extra", f"{sf.annual_word} {ym[0]}", annual[ym[0]]))
    ncol = len(corp.cols)
    rse, chg, ytd_lab, ci = sf.trap if varied else ("Average RSE (%)¹", "Percent Change²", "Year to date",
                                                    "90 percent confidence interval³")
    out.append(("extra", rse, [("X" if v is None else str(rng.randint(2, 30))) for v in next(iter(rows.values()))]))
    prev = ym_add(rep, -1)
    last, before = rows[rep], rows.get(prev)
    if before is not None:
        pc = [("S" if (a is None or b is None or b == 0) else f"{(a - b) / b * 100:.1f}%") for a, b in zip(last, before, strict=False)]
        out.append(("extra", chg, [""] * ncol))
        out.append(("extra", f"{ABBR[rep[1] - 1]} {rep[0]} from {ABBR[prev[1] - 1]} {prev[0]}", pc))
        out.append(("extra", ci, [("X" if c == "S" else f"±{rng.uniform(3, 30):.1f}") for c in pc]))
    if adj == "NSA" and rng.random() < 0.6:
        ytd = [("S" if v is None else _fmt(v * rng.uniform(3.5, 5.5), dec)) for v in last]
        out.insert(0, ("extra", f"{rep[0]} {ytd_lab}", ytd))
    return out


def render_table(corp: Corpus, style: str, caption: str, header_note: str, rows: list, rng: random.Random) -> str:
    sf = corp.surf
    if sf is not None and not sf.native:
        groups = [(sf.gname(g), [sf.sname(x) for x in subs]) for g, subs in corp.dom["groups"]]
        cols = [(sf.gname(g), sf.sname(x)) for g, x in corp.cols]
    else:
        cols = corp.cols
        groups = corp.dom["groups"]
    lines = [caption, ""]
    if style == "html" and sf is not None and not sf.native:
        lines.extend(_html_varied(sf, groups, cols, rows))
    elif style == "html":
        h1 = "<tr><td rowspan=\"2\">Period</td>" + "".join(f"<td colspan=\"{len(s)}\">{g}</td>" for g, s in groups) + "</tr>"
        subs = [s for _, s in cols]
        if rng.random() < 0.3 and len(subs) > 9:
            j = rng.randrange(5, len(subs) - 1)
            subs = subs[:j] + [subs[j] + "&lt;fci&gt;" + subs[j + 1]] + subs[j + 2:]
        h2 = "<tr>" + "".join(f"<td>{s}</td>" for s in subs) + "</tr>"
        body = []
        for kind, lab, cells in rows:
            if kind == "year":
                body.append("<tr><td>" + lab + "</td>" + "<td></td>" * len(cols) + "</tr>")
            else:
                body.append("<tr><td>" + lab + "</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
        lines.append("<table>" + h1)
        lines.append(h2)
        lines.extend(body)
        lines.append("</table>")
    elif style == "pipe":
        lines.append("| Period | " + " | ".join(f"{g}" + " |" * (len(s) - 1) for g, s in groups) + " |")
        lines.append("| --- |" + " --- |" * len(cols))
        lines.append("| | " + " | ".join(s for _, s in cols) + " |")
        for kind, lab, cells in rows:
            if kind == "year" and not cells:
                lines.append(f"| {lab} |" + " |" * len(cols))
            else:
                lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    else:
        w = 12
        lines.append("Period".ljust(26) + "".join(g[:w * len(s) - 1].ljust(w * len(s)) for g, s in groups))
        lines.append(" " * 26 + "".join(s[:w - 1].rjust(w) for _, s in cols))
        for kind, lab, cells in rows:
            if kind == "year" and not cells:
                lines.append(lab)
            else:
                lines.append(lab.ljust(26) + "".join(c.rjust(w) for c in cells))
    lines.append("")
    lines.append(header_note)
    return "\n".join(lines)


def _html_varied(sf: Any, groups: list, cols: list, rows: list) -> list[str]:
    """HTML table with a two-row grouped header; some tables get OCR-style empty cells that break positional parsing."""
    h, rr = sf.html, sf.rr
    tag, lab = h["cell"], h["period_word"]

    def hc(x: str, attr: str = "") -> str:
        return f"<{tag}{attr}>{x}</{tag}>"
    rowspan = h["lead"] == "rowspan"
    h1 = [hc(lab, ' rowspan="2"')] if rowspan else [hc("")]
    h1 += [hc(g, f' colspan="{len(s)}"') if len(s) > 1 else hc(g) for g, s in groups]
    h2 = ([] if rowspan else [hc(lab)]) + [hc(x) for _, x in cols]
    trail = 0
    if rr.random() < h["noise"]:
        n1, n2 = rr.choice([(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1)])
        for row, n, lo in ((h1, n1, 1), (h2, n2, 0 if rowspan else 1)):
            for _ in range(n):
                row.insert(rr.randint(lo, len(row)), hc(""))
        trail = rr.choice([1, 1, 2, 3])
    pad = "<td></td>" * trail
    out = ["<table>", "<tr>" + "".join(h1) + "</tr>", "<tr>" + "".join(h2) + "</tr>"]
    for _kind, label, cells in rows:
        cells = cells if cells else [""] * len(cols)
        out.append("<tr><td>" + label + "</td>" + "".join(f"<td>{c}</td>" for c in cells) + pad + "</tr>")
    out.append("</table>")
    return out


def _display_cols(corp: Corpus) -> list[tuple[str, str]]:
    sf = corp.surf
    return corp.cols if sf is None or sf.native else [(sf.gname(g), sf.sname(x)) for g, x in corp.cols]


def render_deviant(corp: Corpus, kind: str, caption: str, header_note: str, rep: tuple[int, int], rows: dict,
                   dec: int) -> str:
    """A layout the corpus' usual parser will not read: months as columns, or one prose line per column."""
    lines = [caption, ""]
    yms = list(rows)
    if kind == "transposed":
        lines.append("| Series | " + " | ".join(f"{ABBR[ym[1] - 1]} {ym[0]}" + (" p" if ym == rep else "") for ym in yms) + " |")
        lines.append("| --- |" + " --- |" * len(yms))
        for ci, (g, s) in enumerate(_display_cols(corp)):
            lines.append(f"| {g} - {s} | " + " | ".join(_cell(rows[ym][ci], dec) for ym in yms) + " |")
    else:
        for ci, (g, s) in enumerate(_display_cols(corp)):
            parts = [f"{MONTHS[ym[1] - 1]} {ym[0]}: {_cell(rows[ym][ci], dec)}" for ym in yms]
            lines.append(f"{g}, {s} -- " + "; ".join(parts) + ".")
    lines.append("")
    lines.append(header_note)
    return "\n".join(lines)


def _release_date(rep: tuple[int, int], rng: random.Random) -> tuple[str, str]:
    nxt = ym_add(rep, 1)
    day = rng.randint(14, 26)
    return f"{MONTHS[nxt[1] - 1]} {day}, {nxt[0]}", f"{WEEKDAYS[rng.randrange(5)]}, {MONTHS[nxt[1] - 1].upper()} {day}, {nxt[0]}"


def _summary(corp: Corpus, r: int, rep: tuple[int, int], rng: random.Random, facts: dict | None = None) -> str:
    """Headline prose with preliminary numbers that a naive regex will pick up."""
    scale = corp.dom["scale"]
    sf = corp.surf
    varied = sf is not None and not sf.native
    out = []
    for m in corp.measures:
        tbl = corp.printed_table(r, m, "SA")
        cur = tbl[rep][0]
        prev = tbl.get(ym_add(rep, -1), [None])[0]
        yago = tbl.get(ym_add(rep, -12), [None])[0]
        if cur is None:
            continue
        rate = "a seasonally adjusted annual rate of" if (m["flow"] and corp.dom["annualized"]) else "a seasonally adjusted level of"
        lead = m["lead"]
        if varied:
            aw = sf.adjword("SA")
            rate, lead = f"{'an' if aw[0] in 'aeiou' else 'a'} {aw} level of", sf.mname(m, "lead")
        if varied:
            s = _summary_varied(sf, lead, aw, rep, cur * scale, prev and prev * scale, yago and yago * scale)
        else:
            s = f"{lead} in {MONTHS[rep[1] - 1]} were at {rate} {_fmt(cur * scale, 0)}."
        if prev and not varied:
            pc = (cur - prev) / prev * 100
            s += (f" This is {abs(pc):.1f} percent ({'±'}{rng.uniform(2, 15):.1f} percent) {'above' if pc >= 0 else 'below'} "
                  f"the revised {MONTHS[ym_add(rep, -1)[1] - 1]} estimate of {_fmt(prev * scale, 0)}")
            if yago:
                pc2 = (cur - yago) / yago * 100
                s += f" and is {abs(pc2):.1f} percent {'above' if pc2 >= 0 else 'below'} the {MONTHS[rep[1] - 1]} {rep[0] - 1} rate of {_fmt(yago * scale, 0)}"
            s += "."
        v1 = tbl[rep][1] if len(corp.cols) > 1 else None
        if v1 is not None:
            c1 = sf.sname(corp.cols[1][1]) if varied else corp.cols[1][1].lower()
            s += f" The {MONTHS[rep[1] - 1]} figure for {c1} was {_fmt(v1 * scale, 0)}."
        if facts is not None:
            f = {ym_str(rep): {"0": cur * scale}}
            if v1 is not None:
                f[ym_str(rep)]["1"] = v1 * scale
            if prev:
                f[ym_str(ym_add(rep, -1))] = {"0": prev * scale}
            facts[m["key"]] = f
        out.append(f"# {sf.mname(m, 'head') if varied else m['head']}\n\n{s}")
    return "\n\n".join(out)


def _summary_varied(sf: Any, lead: str, aw: str, rep: tuple[int, int], cur: float, prev: float | None,
                    yago: float | None) -> str:
    """Headline prose, worded per corpus."""
    rr, mo, pm = sf.rr, MONTHS[rep[1] - 1], MONTHS[ym_add(rep, -1)[1] - 1]

    def chg(a: float, b: float) -> str:
        pc = (a - b) / b * 100
        return f"{'up' if pc >= 0 else 'down'} {abs(pc):.1f}%"
    form = rr.randrange(3)
    if form == 0:
        s = f"{lead} came to {_fmt(cur, 0)} in {mo} ({aw})."
        if prev:
            s += f" That is {chg(cur, prev)} from {pm}, now put at {_fmt(prev, 0)}"
            s += f", and {chg(cur, yago)} from twelve months before ({_fmt(yago, 0)})." if yago else "."
    elif form == 1:
        s = f"{mo} {rep[0]}: {lead.lower()} {_fmt(cur, 0)} ({aw})"
        if prev:
            s += f"; {pm} revised to {_fmt(prev, 0)}"
        if yago:
            s += f"; one year earlier {_fmt(yago, 0)}"
        s += "."
    else:
        s = f"The {aw} estimate of {lead.lower()} for {mo} is {_fmt(cur, 0)}"
        if prev:
            s += f", against a revised {_fmt(prev, 0)} for {pm}"
        s += "."
        if yago:
            s += f" Year on year the change is {(cur - yago) / yago * 100:+.1f}%."
    return s


def build_corpus(rng: random.Random, dom_key: str, n_reports: int, target_chars: int, style: str,
                 deviant_frac: float, shuffle: bool, end_year: int | None = None, surface: str = "native") -> dict:
    """``surface``: "native" or "varied" (restyled per corpus from a separate RNG, so the data is unchanged)."""
    from .surface import Surface

    dom = DOMAINS[dom_key]
    n_meas = len(dom["measures"])
    k = n_meas if n_reports <= 30 else max(2, n_meas - (1 if n_reports <= 60 else 2))
    measures = rng.sample(dom["measures"], k)
    keys = [m["key"] for m in measures]
    if "permits" in keys and "authorized" not in keys and rng.random() < 0.7:
        measures[next(i for i, m in enumerate(measures) if m["key"] != "permits")] = \
            next(m for m in dom["measures"] if m["key"] == "authorized")
    measures = sorted(measures, key=lambda m: dom["measures"].index(m))
    start = (rng.randint(2014, 2021), rng.randint(1, 12))
    if end_year is not None:
        start = (end_year - (n_reports + start[1]) // 12 - rng.randint(0, 2), start[1])
    surf = Surface(random.Random(rng.random()) if surface == "varied" else None, dom_key, dom, surface)
    if not surf.native:
        style = surf.table_style
    corp = Corpus(rng, dom_key, n_reports, measures, surf.window, start, surf)
    docs, items, report_meta = [], [], []
    n_dev = int(round(deviant_frac * n_reports))
    deviant = set(rng.sample(range(n_reports), n_dev)) if n_dev else set()
    if deviant and rng.random() < 0.5:
        deviant.add(n_reports - 1)
    bare_captions = rng.random() < 0.7
    for r, rep in enumerate(corp.reports):
        rel_short, rel_long = _release_date(rep, rng)
        head_facts: dict = {}
        head = (surf.title(rep, rel_short, rng) + _summary(corp, r, rep, rng, head_facts)) if not surf.native else (
                f"# FOR RELEASE AT 10:00 AM EDT, {rel_long}\n\n# {dom['release']}, {MONTHS[rep[1] - 1].upper()} {rep[0]}\n\n"
                f"Release Number: CB{rep[0] % 100:02d}-{rng.randint(10, 199)}\n\n"
                f"{rel_short} - The {dom['agency']} announced the following statistics for {MONTHS[rep[1] - 1]} {rep[0]}:\n\n"
                + _summary(corp, r, rep, rng, head_facts))
        sections = [("head", head)]
        tables_meta = {}
        for m in corp.measures:
            tn = surf.tnum_of(m)
            sec_title = f"# {m['title']} ({dom['unit']})" if surf.native else surf.section(m)
            for adj, letter in (("SA", "a"), ("NSA", "b")):
                tid = f"T{tn}{letter}"
                mi = tn - 1
                dec = corp.decimals(adj, m)
                rows = corp.printed_table(r, m, adj)
                cap = (f"Table {tn}{letter} - {ADJ_TITLE[(adj, m['flow'], dom['annualized'])]}" if bare_captions else
                       f"Table {tn}{letter} - {ADJ_TITLE[(adj, m['flow'], dom['annualized'])]} — "
                       f"{m['head']}, {MONTHS[rep[1] - 1]} {rep[0]} release")
                note = (f"p Preliminary r Revised S Does not meet publication standards X Not applicable. "
                        f"Source: {dom['agency']}, {dom['release'].title()}, {rel_short}.")
                if not surf.native:
                    cap = surf.caption(tn, letter, adj)
                    note = f"{surf.note_line()} Source: {surf.agency}, {surf.release.title()}, {rel_short}."
                if r in deviant and adj == ("SA" if mi % 2 == 0 else "NSA") or (r in deviant and rng.random() < 0.5):
                    text = render_deviant(corp, rng.choice(["transposed", "lines"]), cap, note, rep, rows, dec)
                    lay = "deviant"
                else:
                    text = render_table(corp, style, cap, note, _table_rows(corp, rep, rows, dec, adj, rng, m), rng)
                    lay = style
                body = (sec_title + "\n\n" + text) if adj == "SA" else text
                sections.append((tid, body))
                tables_meta[tid] = {"m": m["key"], "adj": adj, "layout": lay, "caption": cap,
                                    "rows": {ym_str(ym): v for ym, v in rows.items()}}
        used = sum(len(t) for _, t in sections)
        notes = ["# EXPLANATORY NOTES" if surf.native else "Notes on methods"]
        frng = random.Random(rng.random())
        while used < target_chars:
            para = " ".join(frng.sample(FILLER, 6))
            notes.append(para)
            used += len(para) + 2
        if len(notes) > 1:
            sections.append(("notes", "\n\n".join(notes)))
        doc = "\n\n".join(t for _, t in sections)
        docs.append(doc)
        for sid, t in sections:
            if sid != "notes":
                items.append((f"R{r}:{sid}", t))
        report_meta.append({"report": ym_str(rep), "tables": tables_meta, "head": head_facts,
                            "revised": sorted(corp.revised_years(rep))})
    order = list(range(n_reports))
    if shuffle:
        rng.shuffle(order)
    docs = [docs[i] for i in order]
    return {"corp": corp, "docs": docs, "items": items, "reports": report_meta, "order": order, "measures": measures,
            "deviant": sorted(deviant), "surface": surf.to_dict(), "style": style}


def latest_series(meta: dict, mkey: str, adj: str, col: int, scale: float) -> dict[str, float]:
    """Gold rule: iterate reports oldest -> newest, later prints overwrite; keep months inside the report range."""
    lo, hi = meta["reports"][0]["report"], meta["reports"][-1]["report"]
    out: dict[str, float] = {}
    for rep in meta["reports"]:
        for t in rep["tables"].values():
            if t["m"] == mkey and t["adj"] == adj:
                for ym, row in t["rows"].items():
                    v = row[col]
                    if lo <= ym <= hi and v is not None:
                        out[ym] = v * scale
    return dict(sorted(out.items()))


def series_phrase(dom: dict, m: dict, adj: str, col: tuple[str, str], rng: random.Random, surf: Any = None) -> str:
    if surf is not None and not surf.native:
        noun, where, aw = surf.mname(m, "q"), surf.col_label(col), surf.adjword(adj)
        return rng.choice([f"{aw} {noun} for {where}", f"{aw} {noun} ({where})", f"{noun} for {where}, {aw}",
                           f"{noun} of {where} ({aw} table)", f"{where} {noun}, {aw} figures"])
    g, s = col
    annual = m["flow"] and dom["annualized"]
    if adj == "SA":
        a = rng.choice(["seasonally adjusted annual rate for", "seasonally adjusted annual rate of"]) if annual else \
            rng.choice(["seasonally adjusted", "seasonally adjusted number of"] if dom is DOMAINS["housing"] or dom is DOMAINS["jobs"] else ["seasonally adjusted"])
    else:
        a = rng.choice(["not seasonally adjusted", "not seasonally adjusted number of"]) if dom is not DOMAINS["retail"] else "not seasonally adjusted"
    if dom is DOMAINS["housing"]:
        cat = {"Total": "total", "1 unit": "single-family", "2 to 4 units": "two-to-four-unit",
               "5 units or more": "five-or-more-unit"}[s]
        where = rng.choice(["in the United States", "nationwide"]) if g == "United States" else f"in the {g}"
        noun = m["q"]
        if m["key"] == "starts" and s != "Total":
            noun = "housing starts"
        elif m["key"] == "permits":
            noun = rng.choice([m["q"], m["q"], "building permits", "housing permits"])
        elif m.get("q_alt") and rng.random() < 0.6:
            rng.choice(m["q_alt"])
        return f"{a} {cat} {noun} {where}".replace("  ", " ")
    if dom is DOMAINS["retail"]:
        who = "total retail and food services" if s.endswith("total") else s.lower()
        return f"{a} {m['q']} of {who}"
    if g == "Total nonfarm":
        where = "for total nonfarm employment"
    elif g == "Industry":
        where = f"in the {s.lower()} industry"
    else:
        where = f"in the {s} region"
    return f"{a} {m['q']} {where}"


VQ: dict[str, list[str]] = {
    "agg_sum": ["What do the monthly {p} add up to over the period?", "Sum the monthly {p} over the period; what is the total?",
                "Give the period total of the {p}.", "Adding every month together, how much {p} was there in all?"],
    "agg_mean": ["What is the mean monthly value of the {p}?", "Compute the arithmetic mean of the monthly {p}.",
                 "On average, how large was a month of {p}?", "Average the monthly {p} over the period."],
    "agg_median": ["Report the median of the monthly {p}.", "What is the middle value (the median) of the monthly {p}?",
                   "Sort the monthly {p} and give the median."],
    "agg_max": ["What is the single highest monthly figure for the {p}?", "Report the largest monthly value of the {p}.",
                "How high did the {p} go in its best month?"],
    "agg_min": ["What is the single lowest monthly figure for the {p}?", "Report the smallest monthly value of the {p}.",
                "How low did the {p} go in its weakest month?"],
    "arg_hi": ["Which month had the highest {p}?", "Name the month in which the {p} reached its maximum.",
               "When (year and month) did the {p} hit its high point?"],
    "arg_lo": ["Which month had the lowest {p}?", "Name the month in which the {p} reached its minimum.",
               "When (year and month) did the {p} hit its low point?"],
    "count_above": ["Count the months in which the {p} was above {thr}.", "For how many months did the {p} come in over {thr}?",
                    "How many monthly observations of the {p} lie above {thr}?"],
    "count_below": ["Count the months in which the {p} was below {thr}.", "For how many months did the {p} come in under {thr}?",
                    "How many monthly observations of the {p} lie below {thr}?"],
    "months_cmp": ["Which months saw the {p1} exceed the {p2}?", "Give every month in which the {p1} was larger than the {p2}.",
                   "In which months did the {p1} come out above the {p2}?"],
    "topk_ratio": ["For every month divide the {p1} by the {p2}. Which {k} months give the largest quotients?",
                   "Rank the months by the quotient {p1} / {p2} and return the top {k}.",
                   "Month by month, compute {p1} over {p2}; report the {k} months where this quotient is biggest."],
    "mean_ratio:mean_of_ratios": ["Take the monthly quotient of the {p1} over the {p2} and average it across the months.",
                                  "Average, over the months, the per-month ratio {p1} : {p2}.",
                                  "What is the mean, across months, of the {p1} divided by the {p2}?"],
    "mean_ratio:ratio_of_totals": ["Divide the period total of the {p1} by the period total of the {p2}.",
                                   "What is the ratio of the summed {p1} to the summed {p2} over the period?",
                                   "Add up the {p1} and the {p2} over the months and give the first sum relative to the second."],
    "months_mom": ["Which months saw the {p1} go up from the prior month while the {p2} went down from the prior month?",
                   "List the months where, relative to the month before, the {p1} rose and the {p2} fell.",
                   "In which months did the {p1} gain on the previous month at the same time as the {p2} lost ground?"],
    "mom_abs_mean": ["On average, by how much did the {p} move from one month to the next, ignoring the sign of the move?",
                     "Compute the absolute change of the {p} between each pair of consecutive months and average those changes.",
                     "What is the mean size (absolute value) of the month-to-month moves in the {p}?"],
    "count_up": ["Count the month-to-month rises in the {p}: how many months were higher than the month before?",
                 "For how many months was the {p} above its value of the preceding month?",
                 "How many times did the {p} go up from one month to the next?"],
    "group_hi": ["Of the {kinds} {opts}, which shows the largest mean {adj} {noun} over the period? Give only its name.",
                 "Averaging the {adj} {noun} over the months, which {kind} ({opts}) comes out on top? Name it only."],
    "group_lo": ["Of the {kinds} {opts}, which shows the smallest mean {adj} {noun} over the period? Give only its name.",
                 "Averaging the {adj} {noun} over the months, which {kind} ({opts}) comes out at the bottom? Name it only."],
    "streak:up": ["How long was the longest unbroken run of monthly rises in the {p}?",
                  "Find the longest stretch during which the {p} rose every month. How long is it?"],
    "streak:down": ["How long was the longest unbroken run of monthly declines in the {p}?",
                    "Find the longest stretch during which the {p} fell every month. How long is it?"],
    "prev_avg": ["In which months was the {p} above the mean of its {k} previous months?",
                 "List each month whose {p} beat the average of the {k} months right before it."],
    "peaks:peak": ["How many months were local highs of the {p}, higher than each of the {wm} immediately before and after?",
                   "Count the months in which the {p} was above all of its neighbours within {wn} on either side."],
    "peaks:trough": ["How many months were local lows of the {p}, lower than each of the {wm} immediately before and after?",
                     "Count the months in which the {p} was below all of its neighbours within {wn} on either side."],
    "cross_ma:above": ["List the months in which the {p} went from at-or-below its {k}-month moving average to above it.",
                       "Which months mark an upward pass of the {p} through its {k}-month moving average?"],
    "cross_ma:below": ["List the months in which the {p} went from at-or-above its {k}-month moving average to below it.",
                       "Which months mark a downward pass of the {p} through its {k}-month moving average?"],
    "filtered:above": ["What is the {stat} {p1} over the months in which the {p2} was above {thr}?",
                       "Take the months where the {p2} came in over {thr}; what is the {stat} of the {p1} in those months?"],
    "filtered:below": ["What is the {stat} {p1} over the months in which the {p2} was below {thr}?",
                       "Take the months where the {p2} came in under {thr}; what is the {stat} of the {p1} in those months?"],
    "topq": ["Select the months that rank in the top {pct}% by {p1}. Over those months, what is the {agg} of the {p2} to the {p3}?",
             "Keep the {pct}% of months in which the {p1} is largest; for them, give the {agg} of the {p2} to the {p3}."],
}
IN_PERIOD = ["Only months inside the period count as neighbours.", "Neighbouring months must also lie inside the period.",
             "Months before or after the period are not used as neighbours."]
PERIOD_FRAMES = ["Consider only the months {lo} through {hi}.", "Use the monthly observations dated {lo} to {hi}, both ends included.",
                 "The period of interest runs from {lo} to {hi} inclusive.", "Restrict the analysis to the months between {lo} and {hi} (inclusive).",
                 "Only the months from {lo} to {hi} (inclusive) count."]
REVISION_RULES = ["A month printed in several releases takes its value from the most recently published of them.",
                  "Releases revise earlier months: always keep the figure from the newest release that prints a month.",
                  "Where releases disagree about a month, the later release supersedes the earlier one.",
                  "Treat the figure in the latest release covering a month as that month's value."]


def _vq(rng: random.Random, key: str, **kw: Any) -> str:
    return rng.choice(VQ[key]).format(**kw)


def frame_question(rng: random.Random, body: str, lo: str, hi: str) -> str:
    """Varied question wording for the period ``lo``..``hi`` (YYYY-MM) and revision rule."""
    f = rng.choice(["iso", "slash", "name"])

    def show(ym: str) -> str:
        return {"iso": ym, "slash": ym.replace("-", "/"), "name": f"{MONTHS[int(ym[5:]) - 1]} {ym[:4]}"}[f]
    per = rng.choice(PERIOD_FRAMES).format(lo=show(lo), hi=show(hi))
    rule = rng.choice(REVISION_RULES)
    return " ".join(rng.choice([[per, body, rule], [body, per, rule], [per, rule, body]]))


def _fmt_thr(x: float) -> str:
    return f"{x:,.0f}"


def make_question(rng: random.Random, built: dict, dom_key: str) -> dict:
    dom = DOMAINS[dom_key]
    corp: Corpus = built["corp"]
    scale = dom["scale"]
    cols = corp.cols
    meas = built["measures"]
    surf = corp.surf
    varied = surf is not None and not surf.native

    def pick_series(avoid=None):
        for _ in range(50):
            m = rng.choice(meas)
            adj = rng.choice(["SA", "NSA"])
            ci = rng.randrange(len(cols)) if rng.random() < 0.7 else 0
            if (m["key"], cols[ci][1]) in dom["suppressed"]:
                continue
            if avoid and (m["key"], adj, ci) == avoid:
                continue
            return m, adj, ci
        return meas[0], "SA", 0

    def vals(m, adj, ci):
        return latest_series(built, m["key"], adj, ci, scale)

    tmpl = rng.choices(["agg", "arg", "count_thr", "months_cmp", "topk_ratio", "mean_ratio", "mom_abs_mean",
                        "count_up", "group_max", "months_mom",
                        "streak", "prev2_avg", "peaks", "cross_ma", "filtered_mean", "topq_ratio"],
                       weights=[3, 2, 2, 1.5, 1, 1.2, 1, 1, 0.8, 1,
                                1, 1, 0.8, 0.7, 1.2, 1])[0]
    m, adj, ci = pick_series()
    s1 = vals(m, adj, ci)
    p1 = series_phrase(dom, m, adj, cols[ci], rng, surf)
    months = list(s1)
    spec: dict[str, Any] = {"tmpl": tmpl, "series": [[m["key"], adj, ci]]}
    if tmpl in ("streak", "prev2_avg", "peaks", "cross_ma", "filtered_mean", "topq_ratio"):
        return _convention_question(rng, built, dom_key, tmpl, m, adj, ci, s1, p1, spec, vals) or \
            make_question(rng, built, dom_key)
    if tmpl == "agg":
        op = rng.choice(["sum", "mean", "median", "max", "min"])
        x = {"sum": sum, "mean": statistics.mean, "median": statistics.median, "max": max, "min": min}[op](s1.values())
        name = {"sum": "total sum of the", "mean": "average", "median": "median value for the", "max": "maximum value for the",
                "min": "minimum value for the"}[op]
        q = _vq(rng, f"agg_{op}", p=p1) if varied else rng.choice([f"What is the {name} {p1} across all available reports?",
                                                                    f"Across all available reports, what was the {name} {p1}?"])
        spec["op"] = op
        return dict(q=q, kind="number", gold=round(x, 2), spec=spec)
    if tmpl == "arg":
        hi = rng.random() < 0.5
        k = (max if hi else min)(s1, key=s1.get)
        if sum(1 for v in s1.values() if v == s1[k]) > 1:
            return make_question(rng, built, dom_key)
        q = _vq(rng, "arg_hi" if hi else "arg_lo", p=p1) if varied else rng.choice([
            f"Find the month with the {'largest' if hi else 'smallest'} {p1}.",
            f"In which month and year was the {p1} the {'highest' if hi else 'lowest'} among all reports?"])
        spec["hi"] = hi
        return dict(q=q, kind="list", gold=[k], spec=spec)
    if tmpl == "count_thr":
        sv = sorted(set(s1.values()))
        if len(sv) < 4:
            return make_question(rng, built, dom_key)
        j = rng.randrange(1, len(sv) - 1)
        thr = math.floor((sv[j - 1] + sv[j]) / 2 / (scale / 10)) * (scale / 10) if scale >= 1000 else (sv[j - 1] + sv[j]) / 2
        if thr in s1.values():
            thr += scale / 20
        above = rng.random() < 0.6
        n = sum(1 for v in s1.values() if (v > thr if above else v < thr))
        q = (_vq(rng, "count_above" if above else "count_below", p=p1, thr=_fmt_thr(thr)) if varied else
             f"How many months show the {p1} {'exceeding' if above else 'below'} {_fmt_thr(thr)}?")
        spec.update(thr=thr, above=above)
        return dict(q=q, kind="number", gold=float(n), spec=spec)
    if tmpl in ("months_cmp", "topk_ratio", "mean_ratio", "months_mom"):
        cands = []
        for m2 in meas:
            for cj in range(len(cols)):
                if (m2["key"], cj) == (m["key"], ci) or (m2["key"], cols[cj][1]) in dom["suppressed"]:
                    continue
                if m2["key"] == m["key"] or cj == ci:
                    cands.append((m2, cj))
        if not cands:
            return make_question(rng, built, dom_key)
        rng.shuffle(cands)
        for m2, cj in cands[:12]:
            s2 = vals(m2, adj, cj)
            common = [k for k in months if k in s2]
            frac = sum(1 for k in common if s1[k] > s2[k]) / max(1, len(common))
            if tmpl not in ("months_cmp", "months_mom") or 0.1 <= frac <= 0.9:
                break
        else:
            return make_question(rng, built, dom_key)
        p2 = series_phrase(dom, m2, adj, cols[cj], rng, surf)
        spec["series"].append([m2["key"], adj, cj])
        if tmpl == "months_cmp":
            gold = [k for k in common if s1[k] > s2[k]]
            q = _vq(rng, "months_cmp", p1=p1, p2=p2) if varied else f"List all months where the {p1} was greater than the {p2}."
            return dict(q=q, kind="list", gold=gold, spec=spec)
        if tmpl == "topk_ratio":
            kk = rng.choice([3, 3, 5])
            r = {k: s1[k] / s2[k] for k in common if s2[k]}
            top = sorted(r, key=lambda k: -r[k])[:kk]
            q = _vq(rng, "topk_ratio", p1=p1, p2=p2, k=kk) if varied else (
                f"Calculate the ratio of the {p1} to the {p2} for each month. "
                f"Return only the {kk} months with the highest ratios.")
            spec["k"] = kk
            return dict(q=q, kind="list", gold=top, spec=spec)
        if tmpl == "mean_ratio":
            share = m2["key"] == m["key"] and cols[ci][0] == cols[cj][0] and cj == _group_total(cols, cj) != ci
            r = [s1[k] / s2[k] for k in common if s2[k]]
            spec["share"] = share
            if varied:
                from .answers import answer_periodic
                spec["fmt"] = rng.choice(["pct", "ratio"])
                spec["agg"] = rng.choice(["mean_of_ratios", "mean_of_ratios", "ratio_of_totals"])
                q = _vq(rng, f"mean_ratio:{spec['agg']}", p1=p1, p2=p2) + " " + rng.choice(
                    ["Express the result as a percentage with two decimals (e.g. 37.25%).",
                     "Give it as a percentage, two decimals, with the % sign."] if spec["fmt"] == "pct" else
                    ["Express the result as a plain decimal ratio with two decimals (e.g. 0.37), not a percentage.",
                     "Give it as a decimal number rounded to two places (no % sign)."])
                gold = answer_periodic(spec, [s1, s2], dom_key)
                return dict(q=q, kind="str" if spec["fmt"] == "pct" else "number", gold=gold, spec=spec)
            said = (rng.random() < 0.4 if share else False) if surf is None or surf.native else share
            q = (f"What is the average ratio of the {p1} to the {p2}, calculated for each month and then averaged?"
                 + (" Give the result as a percentage." if said else ""))
            if surf is not None and not surf.native and not share:
                q += " Give the result as a plain number."
            if share:
                return dict(q=q, kind="str", gold=f"{statistics.mean(r) * 100:.2f}%", spec=spec)
            return dict(q=q, kind="number", gold=round(statistics.mean(r), 2), spec=spec)
        gold = []
        for a, b in zip(common, common[1:], strict=False):
            if s1[b] > s1[a] and s2[b] < s2[a]:
                gold.append(b)
        if not gold:
            return make_question(rng, built, dom_key)
        q = _vq(rng, "months_mom", p1=p1, p2=p2) + " " + rng.choice(IN_PERIOD) if varied else (
            f"List all months where the {p1} increased compared to the previous month, while the {p2} "
            "decreased compared to the previous month.")
        return dict(q=q, kind="list", gold=gold, spec=spec)
    if tmpl == "mom_abs_mean":
        d = [abs(s1[b] - s1[a]) for a, b in zip(months, months[1:], strict=False)]
        q = (_vq(rng, "mom_abs_mean", p=p1) + " " + rng.choice(IN_PERIOD) if varied else
             f"What is the average month-over-month change, in absolute value, for the {p1}?")
        return dict(q=q, kind="number", gold=round(statistics.mean(d), 2), spec=spec)
    if tmpl == "count_up":
        n = sum(1 for a, b in zip(months, months[1:], strict=False) if s1[b] > s1[a])
        q = (_vq(rng, "count_up", p=p1) + " " + rng.choice(IN_PERIOD) if varied else
             f"In how many months did the {p1} increase compared to the previous month?")
        return dict(q=q, kind="number", gold=float(n), spec=spec)
    if dom_key == "housing":
        members = [i for i, (g, s) in enumerate(cols) if g != "United States" and s == "Total"]
        def label(i):
            return cols[i][0]
        kind_word, opts = "region", "Northeast, Midwest, South, West"
    elif dom_key == "retail":
        members = [i for i, (g, s) in enumerate(cols) if not s.endswith("total") and (m["key"], s) not in dom["suppressed"]]
        def label(i):
            return cols[i][1]
        kind_word, opts = "kind of business", ", ".join(cols[i][1] for i in members)
    else:
        members = [i for i, (g, s) in enumerate(cols) if g == "Region"]
        def label(i):
            return cols[i][1]
        kind_word, opts = "region", "Northeast, South, Midwest, West"
    noun = m["q"]
    if surf is not None and not surf.native:
        def label(i):
            g, x = cols[i]
            return surf.gname(g) if dom_key == "housing" else surf.sname(x)
        kind_word = {"housing": "area", "retail": "category", "jobs": "area"}[dom_key]
        opts, noun = ", ".join(label(i) for i in members), surf.mname(m, "q")
        spec["labels"] = {str(i): label(i) for i in members}
    avgs = {i: statistics.mean(vals(m, adj, i).values()) for i in members}
    best = max(avgs, key=avgs.get)
    lo = rng.random() < 0.3
    if lo:
        best = min(avgs, key=avgs.get)
    adj_w = surf.adjword(adj) if surf is not None else ("seasonally adjusted" if adj == "SA" else "not seasonally adjusted")
    q = (_vq(rng, "group_lo" if lo else "group_hi", kind=kind_word, opts=opts,
             kinds={"area": "areas", "category": "categories"}.get(kind_word, kind_word + "s"), adj=adj_w, noun=noun) if varied else
         f"Across all reports, which {kind_word} ({opts}) has the {'lowest' if lo else 'highest'} average {adj_w} "
         f"{noun}? Return only the name.")
    spec.update(members=members, lo=lo)
    spec["series"] = [[m["key"], adj, i] for i in members]
    return dict(q=q, kind="str", gold=label(best), spec=spec)


def _group_total(cols: list[tuple[str, str]], ci: int) -> int:
    """Index of the total column of column ``ci``'s group (itself when it is the total or the group has none)."""
    g = cols[ci][0]
    for j, (g2, s2) in enumerate(cols):
        if g2 == g and (s2 == "Total" or s2.endswith("total")):
            return j
    return ci


def _convention_varied(rng: random.Random, built: dict, dom_key: str, tmpl: str, m: dict, adj: str, ci: int,
                       p1: str, spec: dict, vals) -> dict | None:
    """Convention questions with the counting rule stated in the question."""
    from .answers import answer_periodic

    dom = DOMAINS[dom_key]
    cols = built["corp"].cols
    meas = built["measures"]
    surf = built["corp"].surf
    S = [vals(m, adj, ci)]
    if tmpl == "streak":
        d = spec["dir"] = rng.choice(["up", "up", "down"])
        unit = spec["count"] = rng.choice(["steps", "months"])
        word, rel = ("rises", "a<b<c<d") if d == "up" else ("declines", "a>b>c>d")
        q = _vq(rng, f"streak:{d}", p=p1) + " " + (
            f"Measure the length as the number of consecutive month-to-month {word} (values {rel} are 3 {word})."
            if unit == "steps" else
            f"Measure the length in months, counting the month the run starts from (values {rel} span 4 months).")
        q += " " + rng.choice(IN_PERIOD)
        kind = "number"
    elif tmpl == "prev2_avg":
        k = spec["k"] = rng.choice([2, 2, 3])
        q = (_vq(rng, "prev_avg", p=p1, k=k) + f" Only months inside the period serve as previous months, so the first "
             f"{k} months of the period are not evaluated.")
        kind = "list"
    elif tmpl == "peaks":
        pk = spec["kind"] = rng.choice(["peak", "peak", "trough"])
        w = spec["w"] = rng.choice([1, 1, 2])
        q = (_vq(rng, f"peaks:{pk}", p=p1, wm="month" if w == 1 else f"{w} months",
                 wn="one month" if w == 1 else f"{w} months") + " Neighbours must lie inside the period, so the "
             + ("first and last month cannot qualify." if w == 1 else f"first and last {w} months cannot qualify."))
        kind = "number"
    elif tmpl == "cross_ma":
        k = spec["k"] = rng.choice([3, 3, 4])
        incl = spec["incl"] = rng.random() < 0.5
        d = spec["dir"] = rng.choice(["above", "above", "below"])
        ma = (f"the mean of that month and the {k - 1} months before it" if incl else
              f"the mean of the {k} months before it (the month itself excluded)")
        q = (_vq(rng, f"cross_ma:{d}", p=p1, k=k) + f" A month's moving average is {ma}, using in-period months only. "
             f"A month counts when the series was {'at or below' if d == 'above' else 'at or above'} its moving average "
             f"in the previous month and is strictly {d} it in this month.")
        kind = "list"
    elif tmpl == "filtered_mean":
        cands = [(m2, cj) for m2 in meas for cj in range(len(cols))
                 if (m2["key"], cols[cj][1]) not in dom["suppressed"] and (m2["key"], cj) != (m["key"], ci)
                 and (m2["key"] == m["key"] or cj == ci)]
        if not cands:
            return None
        m2, cj = rng.choice(cands)
        s2 = vals(m2, adj, cj)
        sv = sorted(s2.values())
        if len(sv) < 6:
            return None
        lo_v, hi_v = sv[len(sv) // 5], sv[(4 * len(sv)) // 5]
        step = dom["scale"] * (10 if dom["scale"] >= 1000 else 1)
        thr = math.floor(rng.uniform(lo_v, hi_v) / step) * step
        if thr in s2.values() or thr <= 0:
            return None
        above = rng.random() < 0.6
        stat = spec["stat"] = rng.choice(["mean", "mean", "median"])
        p2 = series_phrase(dom, m2, adj, cols[cj], rng, surf)
        spec["series"].append([m2["key"], adj, cj])
        spec.update(thr=thr, above=above)
        S.append(s2)
        q = _vq(rng, f"filtered:{'above' if above else 'below'}", p1=p1, p2=p2, thr=_fmt_thr(thr),
                stat={"mean": "average", "median": "median"}[stat])
        kind = "number"
    else:
        others = [cj for cj in range(len(cols)) if cj != ci and (m["key"], cols[cj][1]) not in dom["suppressed"]]
        if len(others) < 2:
            return None
        cj, ck = rng.sample(others, 2)
        pct = spec["pct"] = rng.choice([10, 20, 25, 30])
        spec["q"] = pct / 100
        cnt = spec["count"] = rng.choice(["floor", "round", "ceil"])
        agg = spec["agg"] = rng.choice(["mean_of_ratios", "mean_of_ratios", "ratio_of_totals"])
        p2 = series_phrase(dom, m, adj, cols[cj], rng, surf)
        p3 = series_phrase(dom, m, adj, cols[ck], rng, surf)
        spec["series"] += [[m["key"], adj, cj], [m["key"], adj, ck]]
        S += [vals(m, adj, cj), vals(m, adj, ck)]
        how = {"floor": "rounded down", "round": "rounded to the nearest whole number (halves round up)",
               "ceil": "rounded up"}[cnt]
        aw = {"mean_of_ratios": "average of the monthly ratios", "ratio_of_totals": "ratio of the totals"}[agg]
        q = (_vq(rng, "topq", pct=pct, p1=p1, p2=p2, p3=p3, agg=aw) + f" The number of months kept is {pct}% of the "
             f"months in the period, {how}." + (" Compute the ratio for each kept month, then average." if
                                               agg == "mean_of_ratios" else
                                               " Sum each series over the kept months, then divide."))
        kind = "number"
    gold = answer_periodic(spec, S, dom_key)
    if gold is None or (kind == "list" and not gold):
        return None
    return dict(q=q, kind=kind, gold=gold, spec=spec)


def _convention_question(rng: random.Random, built: dict, dom_key: str, tmpl: str, m: dict, adj: str, ci: int,
                         s1: dict, p1: str, spec: dict, vals) -> dict | None:
    """Questions that depend on a counting convention. Native: fixed, unstated rules; varied (default): rules stated.
    Returns None to request another draw."""
    from .answers import answer_periodic

    dom = DOMAINS[dom_key]
    cols = built["corp"].cols
    meas = built["measures"]
    surf = built["corp"].surf
    S = [s1]
    if surf is not None and not surf.native:
        return _convention_varied(rng, built, dom_key, tmpl, m, adj, ci, p1, spec, vals)
    if tmpl == "streak":
        q = rng.choice([f"What is the length, in months, of the longest period of consecutive months during which the "
                        f"{p1} increased each month?",
                        f"Identify the length, in number of months, of the longest continuous period where the {p1} "
                        "consistently increased. Return only this number."])
        kind = "number"
    elif tmpl == "prev2_avg":
        q = f"Find all months where the {p1} was higher than the average of the two preceding months."
        kind = "list"
    elif tmpl == "peaks":
        q = (f"How many months were peaks for the {p1}, defined as months where the value is higher than both the "
             "preceding and the succeeding month?")
        kind = "number"
    elif tmpl == "cross_ma":
        q = f"List the months in which the {p1} crossed above its 3-month moving average."
        kind = "list"
    elif tmpl == "filtered_mean":
        cands = [(m2, cj) for m2 in meas for cj in range(len(cols))
                 if (m2["key"], cols[cj][1]) not in dom["suppressed"] and (m2["key"], cj) != (m["key"], ci)
                 and (m2["key"] == m["key"] or cj == ci)]
        if not cands:
            return None
        m2, cj = rng.choice(cands)
        s2 = vals(m2, adj, cj)
        sv = sorted(s2.values())
        if len(sv) < 6:
            return None
        lo_v, hi_v = sv[len(sv) // 5], sv[(4 * len(sv)) // 5]
        step = dom["scale"] * (10 if dom["scale"] >= 1000 else 1)
        thr = math.floor(rng.uniform(lo_v, hi_v) / step) * step
        if thr in s2.values() or thr <= 0:
            return None
        above = rng.random() < 0.6
        p2 = series_phrase(dom, m2, adj, cols[cj], rng, surf)
        spec["series"].append([m2["key"], adj, cj])
        spec.update(thr=thr, above=above)
        S.append(s2)
        q = (f"What is the average {p1} during months where the {p2} "
             f"{'exceeded' if above else 'was below'} {_fmt_thr(thr)}?")
        kind = "number"
    else:
        others = [cj for cj in range(len(cols)) if cj != ci and (m["key"], cols[cj][1]) not in dom["suppressed"]]
        if len(others) < 2:
            return None
        cj, ck = rng.sample(others, 2)
        frac = rng.choice([0.2, 0.25])
        p2 = series_phrase(dom, m, adj, cols[cj], rng, surf)
        p3 = series_phrase(dom, m, adj, cols[ck], rng, surf)
        spec["series"] += [[m["key"], adj, cj], [m["key"], adj, ck]]
        spec["q"] = frac
        S += [vals(m, adj, cj), vals(m, adj, ck)]
        q = (f"Among the top {int(frac * 100)}% of months with the highest {p1}, what was the average ratio of the "
             f"{p2} to the {p3}?")
        kind = "number"
    gold = answer_periodic(spec, S, dom_key)
    if gold is None or (kind == "list" and not gold):
        return None
    return dict(q=q, kind=kind, gold=gold, spec=spec)
