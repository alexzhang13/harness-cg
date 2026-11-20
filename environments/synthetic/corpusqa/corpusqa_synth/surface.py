"""Surface layer of the periodic family: what a corpus LOOKS like, drawn per corpus, independent of its data."""

from __future__ import annotations

import random
from typing import Any

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
ABBR3 = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

FLOW_NOUNS = ["dispatches", "arrivals", "consignments", "registrations", "bookings", "inspections", "clearances",
              "renewals", "withdrawals", "placements", "referrals", "deliveries", "returns", "filings", "admissions",
              "transfers", "landings", "sailings", "listings", "enrolments"]
STOCK_NOUNS = ["orders on hand", "open cases", "backlog", "units in storage", "pending applications",
               "active licences", "vessels in port", "outstanding claims"]
PLACES = ["Northgate", "Lowmere", "Eastvale", "Saltmarsh", "Highfell", "Westbrook", "Redcliff", "Ambermoor",
          "Greyhaven", "Sunford", "Ironbridge", "Coldwater", "Marrowby", "Thistledown", "Brackenhurst", "Oakhollow"]
NATIONAL = ["All districts", "Network total", "Combined territory", "Whole system", "All areas"]
CLASS_SETS = [["small", "medium", "large"], ["Class I", "Class II", "Class III"], ["single", "paired", "multiple"],
              ["tier A", "tier B", "tier C"], ["light", "standard", "heavy"], ["short-haul", "mid-haul", "long-haul"]]
KINDS = ["Hardware", "Textiles", "Provisions", "Fuel", "Mail order", "Instruments", "Garden supplies", "Optics",
         "Leather goods", "Stationery", "Ceramics", "Spices"]
SECTORS = ["Forestry", "Shipbuilding", "Printing", "Milling", "Weaving", "Masonry", "Brewing", "Tanning"]
TOTAL_WORDS = ["Total", "All", "All classes", "Overall"]
ADJ_WORDS = [("seasonally adjusted", "not seasonally adjusted"), ("adjusted", "unadjusted"), ("smoothed", "raw"),
             ("calendar-adjusted", "as recorded"), ("trend-adjusted", "actual")]
TOPICS = ["Freight Activity", "Market Transactions", "Service Registrations", "Network Traffic", "Trade Movements",
          "Facility Usage", "Port Operations", "Licensing Activity"]
AGENCIES = ["Office of Regional Statistics", "Bureau of Network Accounts", "Statistical Service of Arden",
            "Commission for Market Data", "Northern Data Authority", "Institute of Trade Records"]
SERIES_WORDS = ["MONTHLY", "PERIODIC", "BULLETIN OF", "MONTHLY BULLETIN:"]
UNIT_TEXT = {1000: ["Thousands", "figures in thousands", "in 000s", "unit: thousand"],
             1_000_000: ["Millions", "figures in millions of credits", "in millions", "unit: million credits"]}
TRAP_LABELS = [("Average error (%)", "Change", "Year to date", "Confidence band"),
               ("Mean RSE", "Monthly change", "Cumulative YTD", "Error margin"),
               ("Sampling error %", "Pct. chg.", "Jan-to-date", "90% interval")]
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII"]

AXES = {"title": ["hash", "ref", "iso", "sentence"], "caption": ["table", "upper", "series", "exhibit", "panel"],
        "section": ["hash", "caps", "plain"], "rows": ["yearrows", "ym"], "revmark": ["(r)", "revised", "*"]}
TABLE_STYLES = (["html", "pipe", "fixed"], [0.7, 0.15, 0.15])
PERIOD_WORDS = ["Period", "Month", "Reference month", "Date", "", "Months"]
ANNUAL_WORDS = ["Annual total", "Total for year", "Calendar-year total", "Year total"]
ANNUAL_STYLES = ["none", "row", "yearrow"]
NOISE_RATES = [0.0, 0.3, 0.6, 0.9]


def _ncols_classes(dom_key: str) -> int:
    return 3 if dom_key == "housing" else 0


def _vary_columns(r: random.Random, dom_key: str, dom: dict) -> dict:
    """Copy of the domain with a per-corpus subset and order of column groups (unlike the eval's fixed layout)."""
    groups = []
    for g, subs in dom["groups"]:
        subs = list(subs)
        if dom_key == "housing" and g == "United States" and r.random() < 0.4:
            subs.remove("2 to 4 units")
        if dom_key == "retail":
            kinds = subs[1:]
            r.shuffle(kinds)
            subs = subs[:1] + kinds[:r.randint(3, len(kinds))]
        if dom_key == "jobs" and g in ("Industry", "Region"):
            r.shuffle(subs)
            subs = subs[:r.randint(2, len(subs))]
        groups.append((g, subs))
    if dom_key == "housing":
        regions = groups[1:]
        r.shuffle(regions)
        groups = groups[:1] + regions[:r.randint(2, len(regions))]
    out = dict(dom)
    out["groups"] = groups
    return out


class Surface:
    """Display names + structure styles of one corpus. ``native`` = the original generator's surface."""

    def __init__(self, rng: random.Random | None, dom_key: str, dom: dict, mode: str = "native"):
        self.mode, self.dom_key, self.dom = mode, dom_key, dom
        self.native = mode == "native"
        if self.native:
            self.window, self.bench = dom["window"], dom.get("bench_month")
            return
        assert rng is not None
        r = rng
        self.dom = dom = _vary_columns(r, dom_key, dom)
        self.style = {ax: r.choice(opts) for ax, opts in AXES.items()}
        self.window = r.randint(7, 18)
        self.bench = r.randint(1, 12)
        flows = iter(r.sample(FLOW_NOUNS, len(FLOW_NOUNS)))
        stocks = iter(r.sample(STOCK_NOUNS, len(STOCK_NOUNS)))
        self.m = {}
        for m in dom["measures"]:
            noun = next(flows) if m["flow"] else next(stocks)
            self.m[m["key"]] = {"noun": noun, "title": f"{noun[0].upper()}{noun[1:]}", "head": noun.title(),
                                "lead": f"{noun[0].upper()}{noun[1:]}"}
        order = [m["key"] for m in dom["measures"]]
        r.shuffle(order)
        self.tnum = {k: i + 1 for i, k in enumerate(order)}
        self.adj = dict(zip(("SA", "NSA"), r.choice(ADJ_WORDS), strict=True))
        places = iter(r.sample(PLACES, len(PLACES)))
        classes = r.choice(CLASS_SETS)
        kinds = iter(r.sample(KINDS, len(KINDS)))
        sectors = iter(r.sample(SECTORS, len(SECTORS)))
        total = r.choice(TOTAL_WORDS)
        self.g, self.s = {}, {}
        for g, subs in dom["groups"]:
            if g in ("United States", "Total nonfarm"):
                self.g[g] = r.choice(NATIONAL)
            elif g in ("Northeast", "Midwest", "South", "West"):
                self.g[g] = next(places)
            elif g == "Kind of business":
                self.g[g] = "Category"
            elif g == "Industry":
                self.g[g] = "Sector"
            elif g == "Region":
                self.g[g] = "Area"
            else:
                self.g[g] = g
            for s in subs:
                if s in self.s:
                    continue
                if s in ("Total", "Retail and food services, total"):
                    self.s[s] = total
                elif s in ("1 unit", "2 to 4 units", "5 units or more"):
                    self.s[s] = classes[["1 unit", "2 to 4 units", "5 units or more"].index(s)]
                elif g == "Kind of business":
                    self.s[s] = next(kinds)
                elif g == "Industry":
                    self.s[s] = next(sectors)
                elif g == "Region":
                    self.s[s] = next(places)
                else:
                    self.s[s] = s
        self.release = f"{r.choice(SERIES_WORDS)} {r.choice(TOPICS).upper()}".replace("  ", " ")
        self.agency = r.choice(AGENCIES)
        self.unit = r.choice(UNIT_TEXT[dom["scale"]])
        self.trap = r.choice(TRAP_LABELS)
        self.table_style = r.choices(*TABLE_STYLES)[0]
        self.html = {"cell": r.choice(["td", "td", "th"]),
                     "lead": r.choice(["rowspan", "blank"]),
                     "period_word": r.choice(PERIOD_WORDS),
                     "noise": r.choice(NOISE_RATES)}
        self.annual = r.choice(ANNUAL_STYLES)
        self.annual_word = r.choice(ANNUAL_WORDS)
        self.rr = random.Random(r.random())

    def gname(self, g: str) -> str:
        return g if self.native else self.g.get(g, g)

    def sname(self, s: str) -> str:
        return s if self.native else self.s.get(s, s)

    def mname(self, m: dict, field: str) -> str:
        return m[field] if self.native else self.m[m["key"]][{"q": "noun"}.get(field, field)]

    def adjword(self, adj: str) -> str:
        return ("seasonally adjusted" if adj == "SA" else "not seasonally adjusted") if self.native else self.adj[adj]

    def tnum_of(self, m: dict) -> int:
        return self.dom["measures"].index(m) + 1 if self.native else self.tnum[m["key"]]

    def col_label(self, col: tuple[str, str]) -> str:
        """How a question names one column (group / category)."""
        g, s = self.gname(col[0]), self.sname(col[1])
        if self.dom_key == "retail" or (self.dom_key == "jobs" and col[0] != "Total nonfarm"):
            return s
        return g if s in TOTAL_WORDS else f"{g}, {s}"

    def title(self, rep: tuple[int, int], rel_short: str, rng: random.Random) -> str:
        mo, y = MONTHS[rep[1] - 1], rep[0]
        t = self.style["title"]
        if t == "ref":
            return (f"{self.release.title()}\nReference period: {mo} {y}\nPublished: {rel_short}\n"
                    f"Release No. {y % 100:02d}/{rng.randint(10, 199)}\n\n")
        if t == "iso":
            return f"=== {self.release.title()} | period {y:04d}-{rep[1]:02d} ===\nIssued {rel_short} by the {self.agency}.\n\n"
        if t == "sentence":
            return (f"{self.agency}\n{self.release.title()}\n\nThis release, issued {rel_short}, presents estimates "
                    f"through {mo} {y}.\n\n")
        return f"# {self.release} -- {mo} {y}\n\nIssued {rel_short}. {self.agency}.\n\n"

    def caption(self, tn: int, letter: str, adj: str) -> str:
        c, aw = self.style["caption"], self.adjword(adj)
        if c == "upper":
            return f"TABLE {tn}-{letter.upper()}. {aw.upper()}"
        if c == "series":
            return f"Series S{tn} ({aw})"
        if c == "exhibit":
            return f"Exhibit {tn}.{1 if letter == 'a' else 2}: {aw}"
        if c == "panel":
            return f"Panel {chr(64 + tn)}{1 if letter == 'a' else 2} -- {aw}"
        return f"Table {tn}{letter} - {aw}"

    def section(self, m: dict) -> str:
        title = self.mname(m, "title")
        s = self.style["section"]
        if s == "caps":
            return f"{title.upper()}\n[{self.unit}]"
        if s == "plain":
            return f"{title} -- {self.unit}"
        return f"# {title} ({self.unit})"

    def year_label(self, year: int, revised: bool) -> str:
        if not revised:
            return str(year)
        mark = self.style["revmark"]
        return f"{year} {mark}" if mark != "*" else f"{year}*"

    def note_line(self) -> str:
        mark = self.style["revmark"]
        what = "a year marked " + (f"'{mark}'" if mark != "*" else "with '*'") + " was revised in this release"
        return f"p preliminary; r revised; {what}; S suppressed; X not applicable."

    def to_dict(self) -> dict[str, Any]:
        if self.native:
            return {"mode": "native"}
        return {"mode": self.mode, "style": self.style, "window": self.window, "bench": self.bench, "m": self.m,
                "tnum": self.tnum, "adj": self.adj, "g": self.g, "s": self.s, "release": self.release,
                "agency": self.agency, "unit": self.unit, "table_style": self.table_style, "html": self.html,
                "annual": self.annual, "annual_word": self.annual_word, "trap": list(self.trap)}
