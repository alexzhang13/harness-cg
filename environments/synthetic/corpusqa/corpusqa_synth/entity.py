"""Entity-report family: one document per entity for one period (company quarterly filings, school data sets)."""

from __future__ import annotations

# ruff: noqa: B023  (per-entity closures are called within their own iteration)
import random
import statistics
from typing import Any

FIRST = ["Harlow", "Brightwater", "Ostrander", "Kestrel", "Marlowe", "Pinecrest", "Redfield", "Sablewood", "Tamarind",
         "Vantage", "Whitcombe", "Yarrow", "Alderton", "Bexley", "Corvane", "Dunmore", "Everly", "Fairmont", "Granger",
         "Holloway", "Ironvale", "Juniper", "Kingsley", "Larkspur", "Meridian", "Northgate", "Oakhurst", "Prescott",
         "Quarrington", "Ridgeway", "Stellan", "Thornbury", "Upland", "Verity", "Westbrook", "Ashcombe", "Calloway",
         "Danforth", "Emberly", "Foxhall", "Glenmore", "Hartwell", "Ingram", "Jessop", "Kellerman", "Lockwood"]
SECOND = ["Instruments", "Logistics", "Foods", "Holdings", "Materials", "Systems", "Industries", "Brands", "Energy",
          "Therapeutics", "Semiconductor", "Packaging", "Chemicals", "Aerospace", "Networks", "Minerals", "Robotics",
          "Pharmaceuticals", "Outdoor", "Hospitality"]
SUFFIX = [("Inc", ["Inc.", "Incorporated", ", Inc."]), ("Corp", ["Corporation", "Corp."]), ("Co", ["Company", "Co."]),
          ("Group Inc", ["Group, Inc.", "Group Incorporated"])]
PLACES = ["Ashford", "Bellmont", "Carrow", "Dellwood", "Eastvale", "Farrington", "Glenhaven", "Halston", "Ivybridge",
          "Kensal", "Lindale", "Montrose", "Norwick", "Orland", "Penhurst", "Queensbury", "Rosedale", "Stanmore",
          "Tolland", "Wexford", "Yardley", "Aldridge", "Brookfield", "Clearwater", "Denholm", "Elmsford"]
SCHOOL_FORMS = ["{p} University", "University of {p}", "{p} College", "{p} State University", "{p} Institute of Technology"]
PLACES_EXT = ([f"{pre} {p}" for pre in ("North", "South", "East", "West", "Lake", "Port", "Mount", "New", "Fort",
                                        "Saint", "Upper", "Lower") for p in PLACES]
              + [f"{p} {suf}" for suf in ("Hills", "Valley", "Springs", "Falls", "Heights", "Park") for p in PLACES])

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]

BOILER = [
    "We operate in a highly competitive environment and our results may vary significantly from period to period.",
    "Changes in interest rates and foreign currency exchange rates could adversely affect our results of operations.",
    "Our business is subject to risks arising from global economic conditions, including inflationary pressures.",
    "We may not realize the anticipated benefits of acquisitions or restructuring actions.",
    "Cybersecurity incidents could disrupt our operations and expose us to liability.",
    "We are subject to legal proceedings and regulatory investigations in the ordinary course of business.",
    "The preparation of financial statements requires management to make estimates and assumptions.",
    "Forward-looking statements are subject to risks and uncertainties that could cause actual results to differ.",
    "We undertake no obligation to update any forward-looking statement except as required by law.",
    "Our ability to attract and retain qualified personnel is important to our continued success.",
    "Supply chain disruptions may increase our costs or delay deliveries to customers.",
    "Climate-related regulation may increase our compliance costs and affect demand for our products.",
    "We rely on information technology systems that may be vulnerable to failures or interruptions.",
    "Changes in tax laws could adversely affect our effective tax rate and cash flows.",
    "Impairment of goodwill or other intangible assets could adversely affect our reported results.",
    "Fluctuations in commodity prices could affect our margins.",
]
SCHOOL_BOILER = [
    "The institution participates in federal student aid programs and reports data to IPEDS annually.",
    "Data are reported as of the official fall census date unless otherwise noted.",
    "Degree-seeking students are those enrolled in courses for credit and recognized as seeking a degree.",
    "Institutional aid includes scholarships and grants from endowment and unrestricted funds.",
    "Students who withdrew for military service or died are excluded from the cohort.",
    "Instructional faculty include those whose primary responsibility is instruction.",
    "Class sections are counted once, regardless of the number of instructors assigned.",
    "Residence hall capacity includes theme housing and cooperative housing.",
    "Tuition and fees are shown for the academic year and may change without notice.",
    "Need-based aid is awarded on the basis of demonstrated financial need.",
]


def _m(v: float, dec: int = 0, paren: bool = True) -> str:
    """Accounting format: thousands separators, negatives in parentheses."""
    s = f"{abs(v):,.{dec}f}"
    if v < 0:
        return f"({s})" if paren else f"-{s}"
    return s


CO_FIELDS: dict[str, dict[str, Any]] = {
    "revenue": {"sec": "income", "unit": "usd", "q": "total revenues",
                "labels": ["Total revenues", "Net sales", "Revenue", "Total net revenue", "Revenues"]},
    "pretax": {"sec": "income", "unit": "usd", "q": "earnings before income taxes",
               "labels": ["Income before income taxes", "Earnings before income taxes", "Income (loss) before income taxes"]},
    "tax": {"sec": "income", "unit": "usd", "q": "income tax expense",
            "labels": ["Provision for income taxes", "Income tax expense", "Income tax expense (benefit)"]},
    "net_income": {"sec": "income", "unit": "usd", "q": "consolidated net income",
                   "labels": ["Net income", "Net earnings (loss)", "Net income (loss)", "Consolidated net income"]},
    "eps_basic": {"sec": "income", "unit": "eps", "q": "basic earnings per share", "labels": ["Basic"]},
    "eps_diluted": {"sec": "income", "unit": "eps", "q": "diluted earnings per share", "labels": ["Diluted"]},
    "shares_basic": {"sec": "income", "unit": "shares", "q": "weighted-average basic shares outstanding", "labels": ["Basic"]},
    "shares_diluted": {"sec": "income", "unit": "shares", "q": "weighted-average diluted shares outstanding",
                       "labels": ["Diluted"]},
    "cash": {"sec": "balance", "unit": "usd", "q": "cash and cash equivalents", "labels": ["Cash and cash equivalents"]},
    "total_assets": {"sec": "balance", "unit": "usd", "q": "total assets", "labels": ["Total assets"]},
    "lt_debt": {"sec": "balance", "unit": "usd", "q": "long-term debt",
                "labels": ["Long-term debt", "Long-term debt, net of current portion", "Long-term debt, less current maturities"]},
    "retained": {"sec": "balance", "unit": "usd", "q": "retained earnings",
                 "labels": ["Retained earnings", "Retained earnings (accumulated deficit)", "Reinvested earnings"]},
    "equity": {"sec": "balance", "unit": "usd", "q": "total shareholders' equity",
               "labels": ["Total shareholders’ equity", "Total stockholders’ equity", "Total shareholders' equity"]},
    "cfo": {"sec": "cashflow", "unit": "usd", "q": "net cash from operating activities",
            "labels": ["Net cash provided by (used in) operating activities", "Net cash from operating activities",
                       "Cash flows from operating activities"]},
    "cfi": {"sec": "cashflow", "unit": "usd", "q": "net cash from investing activities",
            "labels": ["Net cash provided by (used in) investing activities", "Net cash from investing activities",
                       "Cash flows from investing activities"]},
    "cff": {"sec": "cashflow", "unit": "usd", "q": "net cash from financing activities",
            "labels": ["Net cash provided by (used in) financing activities", "Net cash from financing activities",
                       "Cash flows from financing activities"]},
    "da": {"sec": "cashflow", "unit": "usd", "q": "depreciation and amortization",
           "labels": ["Depreciation and amortization", "Depreciation, depletion and amortization"]},
}


CO_LABELS_EXT = {
    "revenue": ["Net revenues", "Total net sales", "Sales", "Operating revenues", "Net sales and revenues",
                "Total revenues, net", "Revenues, net", "Net revenue"],
    "pretax": ["Income from continuing operations before income taxes", "Earnings (loss) before income taxes",
               "Income before provision for income taxes", "Pre-tax income"],
    "tax": ["Income tax provision", "Provision for (benefit from) income taxes", "Income taxes", "Tax expense (benefit)"],
    "net_income": ["Net earnings", "NET INCOME", "Net (loss) income", "Net income including noncontrolling interests",
                   "Net income (loss) including noncontrolling interests"],
    "cash": ["Cash and equivalents", "Cash", "Cash and cash equivalents, at carrying value"],
    "total_assets": ["TOTAL ASSETS", "Total Assets"],
    "lt_debt": ["Long-term borrowings", "Debt, less current portion", "Term debt", "Long-term debt, net",
                "Notes payable, net of current portion", "Long-term obligations, less current portion",
                "Long-term debt and finance leases"],
    "retained": ["Retained earnings (deficit)", "Accumulated earnings (deficit)", "Earnings retained in the business"],
    "equity": ["Total equity attributable to the Company", "Total stockholders’ equity (deficit)",
               "Total shareholders’ investment", "TOTAL STOCKHOLDERS’ EQUITY"],
    "cfo": ["Net cash (used in) provided by operating activities", "Net cash flows from operating activities",
            "Cash provided by (used for) operations", "Net cash provided by (used in) operating activities of continuing operations"],
    "cfi": ["Net cash (used in) provided by investing activities", "Net cash flows from investing activities",
            "Cash provided by (used for) investing activities", "Investing activities, net"],
    "cff": ["Net cash (used in) provided by financing activities", "Net cash flows from financing activities",
            "Cash provided by (used for) financing activities", "Financing activities, net"],
    "da": ["Depreciation", "Depreciation and amortization expense", "Amortization and depreciation",
           "Depreciation, amortization and accretion"],
}


def _company_values(rng: random.Random) -> dict[str, dict[str, float]]:
    """Internally consistent quarter (cur) and comparative (prior) figures, in full dollars / shares."""
    size = 10 ** rng.uniform(8.6, 10.6)
    out: dict[str, dict[str, float]] = {}
    for per in ("cur", "prior"):
        g = 1.0 if per == "cur" else rng.uniform(0.82, 1.12)
        rev = size * g
        margin = rng.uniform(-0.12, 0.28) if rng.random() < 0.85 else rng.uniform(-0.4, -0.12)
        pretax = rev * margin
        tax = pretax * rng.uniform(0.12, 0.27) if pretax > 0 else pretax * rng.uniform(-0.05, 0.2)
        ni = pretax - tax
        nci = ni * rng.uniform(0.0, 0.08) if rng.random() < 0.45 else 0.0
        shares_b = size / rng.uniform(4, 60) if per == "cur" else out["cur"]["shares_basic"] * rng.uniform(0.97, 1.05)
        shares_d = shares_b * (1 + (rng.uniform(0.002, 0.02) if ni > 0 else 0.0))
        if per == "cur":
            assets = size * rng.uniform(2.5, 9)
            cash = assets * rng.uniform(0.02, 0.22)
            equity = assets * rng.uniform(0.12, 0.6)
            lt_debt = assets * rng.uniform(0.05, 0.4)
            retained = equity * rng.uniform(-0.3, 1.8)
        else:
            c = out["cur"]
            assets = c["total_assets"] * rng.uniform(0.9, 1.08)
            cash = c["cash"] * rng.uniform(0.6, 1.5)
            equity = c["equity"] * rng.uniform(0.9, 1.1)
            lt_debt = c["lt_debt"] * rng.uniform(0.85, 1.15)
            retained = c["retained"] - c["ni_company"] * rng.uniform(0.3, 1.0)
        cfo = ni * rng.uniform(0.3, 2.2) + size * rng.uniform(-0.08, 0.12)
        cfi = -abs(size * rng.uniform(0.02, 0.2)) if rng.random() < 0.85 else size * rng.uniform(0.01, 0.1)
        cff = -cfo * rng.uniform(0.2, 1.1) + size * rng.uniform(-0.1, 0.1)
        da = size * rng.uniform(0.02, 0.09)
        out[per] = {"revenue": rev, "pretax": pretax, "tax": tax, "net_income": ni, "ni_nci": nci,
                    "ni_company": ni - nci, "shares_basic": shares_b, "shares_diluted": shares_d,
                    "total_assets": assets, "cash": cash, "equity": equity, "nci_equity": equity * rng.uniform(0, 0.05) if nci else 0.0,
                    "lt_debt": lt_debt, "cur_debt": lt_debt * rng.uniform(0.02, 0.2), "retained": retained,
                    "cfo": cfo, "cfi": cfi, "cff": cff, "da": da}
        out[per]["eps_basic"] = out[per]["ni_company"] / shares_b
        out[per]["eps_diluted"] = out[per]["ni_company"] / shares_d
    return out


_ROOTS = ["Ash", "Bel", "Cor", "Dun", "Elm", "Fair", "Glen", "Hart", "Kel", "Lind", "Mar", "Nor", "Oak", "Pem", "Quin",
          "Ros", "Stan", "Thorn", "Wex", "Yar", "Brae", "Cal", "Dor", "Fen"]
_ENDINGS = ["ford", "ley", "ton", "wick", "more", "dale", "field", "by", "worth", "stead"]


def _first_ext() -> list[str]:
    """Extra company name words for scale >= 1m (none overlap the originals)."""
    base = {f.lower() for f in FIRST}
    return [w for w in dict.fromkeys(r + e for r in _ROOTS for e in _ENDINGS)
            if w.lower() not in base and not any(b in w.lower() or w.lower() in b for b in base)]


def _co_name(rng: random.Random, used: set[str]) -> tuple[str, str]:
    pool = FIRST if len(used) < len(FIRST) else _first_ext()
    while True:
        f, s = rng.choice(pool), rng.choice(SECOND)
        if f in used:
            continue
        used.add(f)
        short, legals = rng.choice(SUFFIX)
        legal_suffix = rng.choice(legals)
        legal = f"{f} {s}{legal_suffix}" if legal_suffix.startswith(",") else f"{f} {s} {legal_suffix}"
        return f"{f} {s} {short}", legal


class _Statement:
    """Render a statement in one of three layouts, remembering which rendered line carries which field."""

    def __init__(self, style: str, heads: list[str], dollars: bool = True, noise: random.Random | None = None):
        self.style, self.heads, self.lines, self.dollars = style, heads, [], dollars
        self.key_lines: dict[str, str] = {}
        self.noise = noise

    def _lab(self, lab: str) -> str:
        n = self.noise
        if n is None or not lab:
            return lab
        if n.random() < 0.15:
            lab += n.choice([" (1)", " (a)", " (Note 6)", "*"])
        if self.style == "html" and n.random() < 0.4:
            lab = lab.replace(" ", "&#160;", n.randint(1, 3))
        return lab

    def _val(self, v: str) -> str:
        return "—" if self.noise is not None and v in ("0", "(0)") else v

    def row(self, label: str, vals: list[str] | None, keys: list[str] | None = None):
        self.lines.append(("label" if vals is None else "row", label, vals, keys or []))

    def text(self) -> str:
        out = []
        n = self.noise
        if self.style == "pipe":
            out += ["| | " + " | ".join(self.heads) + " |", "| --- |" + " --- |" * len(self.heads)]
            for _kind, lab, vals, keys in self.lines:
                lab = self._lab(lab) if vals is not None else lab
                vals = None if vals is None else [self._val(v) for v in vals]
                if vals is not None and n is not None and keys and n.random() < 0.3:
                    vals = ["$ " + v if i == 0 else v for i, v in enumerate(vals)]
                ln = f"| {lab} |" + (" |" * len(self.heads) if vals is None else " " + " | ".join(vals) + " |")
                out.append(ln)
                for k in keys:
                    self.key_lines[k] = ln
        elif self.style == "html":
            out += ["<table>", "<tr><td></td>" + "".join(f"<td>{h}</td>" for h in self.heads) + "</tr>"]
            for _kind, lab, vals, keys in self.lines:
                lab = self._lab(lab) if vals is not None else lab
                if vals is None:
                    cells = "<td></td>" * len(self.heads)
                elif n is not None and keys and n.random() < 0.4:
                    cells = "".join(f"<td>$</td><td>{self._val(v)}</td>" for v in vals)
                else:
                    cells = "".join(f"<td>{self._val(v)}</td>" for v in vals)
                ln = f"<tr><td>{lab}</td>{cells}</tr>"
                out.append(ln)
                for k in keys:
                    self.key_lines[k] = ln
            out.append("</table>")
        else:
            out.append(" " * 58 + "".join(h.rjust(18) for h in self.heads))
            for _kind, lab, vals, keys in self.lines:
                if vals is None:
                    ln = lab
                else:
                    lab = self._lab(lab)
                    nums = "".join(("$ " + self._val(v) if self.dollars and i == 0 and lab.startswith(("Total", "Net income", "Revenue", "Net sales", "Cash and")) else self._val(v)).rjust(18)
                                   for i, v in enumerate(vals))
                    if n is not None and keys and n.random() < 0.3:
                        out.append(lab)
                        ln = " " * 58 + nums
                    else:
                        ln = lab.ljust(58) + nums
                out.append(ln)
                for k in keys:
                    self.key_lines[k] = ln
        return "\n".join(out)


def _decoys(rng: random.Random, name: str, vals: dict, dscale: float, unit_word: str, titles: list[str],
            yr: int, mo: int) -> dict[str, str]:
    """Decoy tables and repeated statement titles for long filings (scale >= 1m), as in real 10-Qs."""
    f = 1.0 / dscale
    q = [rng.uniform(0.7, 1.15) for _ in range(4)]
    qs = [f"Q{i + 1} {yr - 1}" for i in range(4)]
    sel = ("Selected Quarterly Financial Data (Unaudited)\n"
           f"({'In ' + unit_word + ', except per share amounts'})\n"
           + " " * 30 + "".join(h.rjust(14) for h in qs) + "\n"
           + "Revenues".ljust(30) + "".join(_m(vals["prior"]["revenue"] * x * f / 4).rjust(14) for x in q) + "\n"
           + "Net income".ljust(30) + "".join(_m(vals["prior"]["net_income"] * x * f / 4).rjust(14) for x in q[::-1]) + "\n"
           + "Diluted earnings per share".ljust(30)
           + "".join(_m(vals["prior"]["eps_diluted"] * x / 4, 2).rjust(14) for x in q) + "\n"
           f"Quarterly amounts may not sum to the annual totals presented in the {titles[0].title()} due to rounding.")
    supp = ("Note 9 - Supplemental Cash Flow Information\n"
            f"(in {unit_word})" + "\n"
            + "Cash paid for interest".ljust(46) + _m(abs(vals["cur"]["lt_debt"]) * rng.uniform(0.008, 0.015) * f).rjust(16) + "\n"
            + "Cash paid for income taxes, net".ljust(46) + _m(abs(vals["cur"]["tax"]) * rng.uniform(0.6, 1.1) * f).rjust(16) + "\n"
            + "Capital expenditures incurred but not yet paid".ljust(46) + _m(abs(vals["cur"]["cfi"]) * rng.uniform(0.02, 0.1) * f).rjust(16) + "\n"
            f"See also the {titles[2].title()} for the reconciliation of cash, cash equivalents and restricted cash.")
    refs = "\n\n".join(
        rng.choice([f"The accompanying notes are an integral part of these {t.title()}.",
                    f"Amounts presented in the {t.title()} are unaudited and reflect all normal recurring adjustments.",
                    f"Certain prior-period amounts in the {t.title()} have been reclassified to conform to the current presentation.",
                    f"Refer to the {t.title()} included in Part I, Item 1 of this report."])
        for t in rng.sample(titles * 4, rng.randint(6, 12)))
    return {"selected": sel, "supp": supp, "refs": f"Basis of Presentation\n{refs}"}


def build_company(rng: random.Random, n: int, target_chars: int, period: tuple[int, int],
                  extras: bool = False, noisy: bool = False) -> dict:
    yr, mo = period
    end_day = {3: 31, 6: 30, 9: 30, 12: 31}[mo]
    period_s = f"{MONTHS[mo - 1]} {end_day}, {yr}"
    prior_s = f"{MONTHS[mo - 1]} {end_day}, {yr - 1}"
    ye_s = f"December 31, {yr - 1}"
    used: set[str] = set()
    ents, docs, items = [], [], []
    for e in range(n):
        name, legal = _co_name(rng, used)
        vals = _company_values(rng)
        dscale = rng.choice([1e6, 1e6, 1e3])
        sscale = rng.choice([1e6, 1e3]) if dscale == 1e6 else rng.choice([1e3, 1e3, 1.0])
        style = rng.choice(["fixed", "fixed", "pipe", "html"])
        unit_word = "millions" if dscale == 1e6 else "thousands"
        unit_hdr = rng.choice([f"(In {unit_word}, except per share amounts)", f"({'$' if rng.random() < 0.5 else 'Dollars'} in {unit_word})",
                               f"(in {unit_word} of U.S. dollars, except per-share data)",
                               "(000s omitted, except per share data)" if dscale == 1e3 else "(in $ millions, except per share)"])
        share_word = {1e6: "millions", 1e3: "thousands", 1.0: "actual shares"}[sscale]
        lab = {k: rng.choice(f["labels"]) for k, f in CO_FIELDS.items()}
        nz = random.Random(rng.random()) if noisy else None
        if nz is not None:
            lab = {k: nz.choice(CO_FIELDS[k]["labels"] + CO_LABELS_EXT.get(k, [])) for k in CO_FIELDS}
        third = rng.random() < 0.25
        unlabeled_ta = rng.random() < 0.2
        p = lambda k, per="cur": vals[per][k] / dscale  # noqa: E731
        ps = lambda k, per="cur": vals[per][k] / sscale  # noqa: E731
        printed: dict[str, dict[str, float]] = {"cur": {}, "prior": {}}

        def put(k, per, v, dec=0):
            printed[per][k] = round(v, dec)
            return _m(v, dec)

        inc = _Statement(style, [str(yr), str(yr - 1)], noise=nz)
        inc.row(lab["revenue"], [put("revenue", "cur", p("revenue")), put("revenue", "prior", p("revenue", "prior"))], ["revenue"])
        inc.row("Cost of revenue", [_m(p("revenue") * 0.62), _m(p("revenue", "prior") * 0.63)])
        inc.row("Operating income", [_m(p("pretax") * 1.08), _m(p("pretax", "prior") * 1.07)])
        inc.row(lab["pretax"], [put("pretax", "cur", p("pretax")), put("pretax", "prior", p("pretax", "prior"))], ["pretax"])
        inc.row(lab["tax"], [put("tax", "cur", p("tax")), put("tax", "prior", p("tax", "prior"))], ["tax"])
        inc.row(lab["net_income"], [put("net_income", "cur", p("net_income")), put("net_income", "prior", p("net_income", "prior"))], ["net_income"])
        if vals["cur"]["ni_nci"]:
            inc.row("Less: Net income attributable to noncontrolling interests", [_m(p("ni_nci")), _m(p("ni_nci", "prior"))])
            inc.row(f"Net income attributable to {name.split(' ')[0]}", [put("ni_company", "cur", p("ni_company")), put("ni_company", "prior", p("ni_company", "prior"))], ["ni_company"])
        inc.row("Earnings per share:" if rng.random() < 0.5 else "Net income per share attributable to common shareholders:", None)
        inc.row("Basic", [put("eps_basic", "cur", vals["cur"]["eps_basic"], 2), put("eps_basic", "prior", vals["prior"]["eps_basic"], 2)], ["eps_basic"])
        inc.row("Diluted", [put("eps_diluted", "cur", vals["cur"]["eps_diluted"], 2), put("eps_diluted", "prior", vals["prior"]["eps_diluted"], 2)], ["eps_diluted"])
        sd = 1 if sscale == 1e6 else 0
        inc.row(f"Weighted-average shares outstanding ({'in ' + share_word if sscale != 1.0 else 'actual'}):", None)
        inc.row("Basic", [put("shares_basic", "cur", ps("shares_basic"), sd), put("shares_basic", "prior", ps("shares_basic", "prior"), sd)], ["shares_basic"])
        inc.row("Diluted", [put("shares_diluted", "cur", ps("shares_diluted"), sd), put("shares_diluted", "prior", ps("shares_diluted", "prior"), sd)], ["shares_diluted"])
        inc_title = rng.choice(["CONDENSED CONSOLIDATED STATEMENTS OF INCOME", "Condensed Consolidated Statements of Operations",
                                "CONSOLIDATED STATEMENTS OF EARNINGS", "Consolidated Statements of Income (Unaudited)"])
        income = (f"{legal.upper()}\n{inc_title}\n{unit_hdr}\n"
                  f"Three Months Ended {MONTHS[mo - 1]} {end_day},\n" + inc.text())
        heads = [period_s, ye_s] + ([prior_s] if third else [])
        bs = _Statement(style, heads, noise=nz)
        cols = ["cur", "prior"] + (["prior"] if third else [])

        def brow(k, label, scale_v=1.0):
            vs = []
            for i, per in enumerate(cols):
                v = vals[per][k] / dscale * (scale_v if i < 2 else rng.uniform(0.93, 1.03))
                vs.append(put(k, per, v) if i < 2 else _m(v))
            bs.row(label, vs, [k])

        bs.row("Assets", None)
        brow("cash", lab["cash"])
        bs.row("Accounts receivable, net", [_m(vals["cur"]["total_assets"] / dscale * 0.11), _m(vals["prior"]["total_assets"] / dscale * 0.1)] + ([_m(vals["prior"]["total_assets"] / dscale * 0.1)] if third else []))
        bs.row("Total current assets", [_m(vals["cur"]["total_assets"] / dscale * 0.35), _m(vals["prior"]["total_assets"] / dscale * 0.34)] + ([_m(vals["prior"]["total_assets"] / dscale * 0.33)] if third else []))
        bs.row("Property, plant and equipment, net", [_m(vals["cur"]["total_assets"] / dscale * 0.3), _m(vals["prior"]["total_assets"] / dscale * 0.31)] + ([_m(vals["prior"]["total_assets"] / dscale * 0.3)] if third else []))
        brow("total_assets", "" if unlabeled_ta else lab["total_assets"])
        bs.row("Liabilities and equity", None)
        bs.row("Current portion of long-term debt", [_m(p("cur_debt")), _m(p("cur_debt", "prior"))] + ([_m(p("cur_debt", "prior") * 0.9)] if third else []))
        brow("lt_debt", lab["lt_debt"])
        brow("retained", lab["retained"])
        brow("equity", lab["equity"])
        if vals["cur"]["nci_equity"]:
            bs.row("Noncontrolling interests", [_m(p("nci_equity")), _m(p("nci_equity", "prior"))] + ([_m(p("nci_equity", "prior"))] if third else []))
            bs.row("Total equity", [_m(p("equity") + p("nci_equity")), _m(p("equity", "prior") + p("nci_equity", "prior"))] + ([_m(p("equity", "prior"))] if third else []))
        bs_title = rng.choice(["CONDENSED CONSOLIDATED BALANCE SHEETS", "Consolidated Balance Sheets", "Condensed Consolidated Balance Sheets (Unaudited)"])
        balance = f"{legal.upper()}\n{bs_title}\n{unit_hdr}\n" + bs.text()
        cf = _Statement(style, [str(yr), str(yr - 1)], noise=nz)
        cf.row("Cash flows from operating activities:", None)
        cf.row(lab["net_income"], [_m(p("net_income")), _m(p("net_income", "prior"))])
        cf.row(lab["da"], [put("da", "cur", p("da")), put("da", "prior", p("da", "prior"))], ["da"])
        cf.row(lab["cfo"], [put("cfo", "cur", p("cfo")), put("cfo", "prior", p("cfo", "prior"))], ["cfo"])
        cf.row("Capital expenditures", [_m(-abs(p("cfi")) * 0.8), _m(-abs(p("cfi", "prior")) * 0.8)])
        cf.row(lab["cfi"], [put("cfi", "cur", p("cfi")), put("cfi", "prior", p("cfi", "prior"))], ["cfi"])
        cf.row(lab["cff"], [put("cff", "cur", p("cff")), put("cff", "prior", p("cff", "prior"))], ["cff"])
        chg = p("cfo") + p("cfi") + p("cff")
        chg_p = p("cfo", "prior") + p("cfi", "prior") + p("cff", "prior")
        cf.row("Net change in cash and cash equivalents", [_m(chg), _m(chg_p)])
        cf.row("Cash and cash equivalents at beginning of period", [_m(printed["cur"]["cash"] - chg), _m(printed["prior"]["cash"] - chg_p)])
        cf.row("Cash and cash equivalents at end of period", [_m(printed["cur"]["cash"]), _m(printed["prior"]["cash"] * rng.uniform(0.9, 1.1))])
        cf_title = rng.choice(["CONDENSED CONSOLIDATED STATEMENTS OF CASH FLOWS", "Consolidated Statements of Cash Flows"])
        cashflow = f"{legal.upper()}\n{cf_title}\n{unit_hdr}\nThree Months Ended {MONTHS[mo - 1]} {end_day},\n" + cf.text()
        cover = ("UNITED STATES\nSECURITIES AND EXCHANGE COMMISSION\nWashington, D.C. 20549\n\nFORM 10-Q\n\n"
                 f"QUARTERLY REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934\n"
                 f"For the quarterly period ended {period_s}\n\nCommission File Number 001-{rng.randint(10000, 99999)}\n\n"
                 f"{legal.upper() if rng.random() < 0.6 else legal}\n(Exact name of registrant as specified in its charter)\n\n"
                 f"As of {MONTHS[mo % 12]} {rng.randint(2, 28)}, {yr}, the registrant had {_m(vals['cur']['shares_basic'] * rng.uniform(0.98, 1.01))} "
                 f"shares of common stock outstanding.")
        toc = ("TABLE OF CONTENTS\nPart I. Financial Information\nItem 1. Financial Statements\n"
               f"{inc_title.title()} {rng.randint(3, 5)}\n{bs_title.title()} {rng.randint(5, 7)}\n{cf_title.title()} {rng.randint(7, 9)}\n"
               "Notes to Condensed Consolidated Financial Statements\nItem 2. Management’s Discussion and Analysis\n"
               "Item 3. Quantitative and Qualitative Disclosures About Market Risk\nPart II. Other Information")
        rv = vals["cur"]["revenue"]
        mdna = (f"Management’s Discussion and Analysis. {name.split(' ')[0]} reported revenue of approximately "
                f"${rv / 1e9:.1f} billion for the quarter, {'up' if rv >= vals['prior']['revenue'] else 'down'} "
                f"{abs(rv / vals['prior']['revenue'] - 1) * 100:.0f}% from the prior-year period. Net income attributable to the "
                f"company was ${vals['cur']['ni_company'] / 1e6:,.0f} million, compared with ${vals['prior']['ni_company'] / 1e6:,.0f} "
                f"million a year earlier. Cash and cash equivalents were about ${vals['cur']['cash'] / 1e9:.2f} billion at quarter end.\n"
                f"Liquidity. Net cash provided by operating activities was ${vals['cur']['cfo'] / 1e6:,.1f} million and net cash from "
                f"financing activities was ${vals['cur']['cff'] / 1e6:,.1f} million, while investing activities used "
                f"${abs(vals['cur']['cfi']) / 1e6:,.1f} million.")
        segs = rng.randint(2, 4)
        shares = [rng.random() for _ in range(segs)]
        tot = sum(shares)
        notes = "Note 4 - Segment information\n" + "\n".join(
            f"Segment {chr(65 + i)} revenue {_m(p('revenue') * w / tot)}" for i, w in enumerate(shares))
        notes += (f"\nNote 7 - Debt\nTotal debt, including current maturities, was {_m(p('lt_debt') + p('cur_debt'))} "
                  f"(in {unit_word}) at {period_s}.")
        brng = random.Random(rng.random())

        def boiler(nchars):
            paras, size = [], 0
            while size < nchars:
                para = " ".join(brng.sample(BOILER, 5))
                paras.append(para)
                size += len(para) + 2
            return "\n\n".join(paras)

        dec = _decoys(rng, name, vals, dscale, unit_word, [inc_title, bs_title, cf_title], yr, mo) if extras else {}
        stmt_len = (len(income) + len(balance) + len(cashflow) + len(cover) + len(toc) + len(mdna) + len(notes)
                    + sum(map(len, dec.values())))
        pad = max(0, target_chars - stmt_len)
        pre = int(pad * rng.uniform(0.0, 0.5))
        sections = [("cover", cover), ("toc", toc)]
        if dec and rng.random() < 0.5:
            sections.append(("selected", dec["selected"]))
        if pre > 2000:
            sections.append(("forward", "FORWARD-LOOKING STATEMENTS\n" + boiler(pre)))
        if dec:
            sections.append(("refs", dec["refs"]))
        sections += [("income", income), ("balance", balance), ("cashflow", cashflow), ("notes", notes)]
        if dec:
            sections.append(("supp", dec["supp"]))
        sections.append(("mdna", mdna))
        if dec and not any(sid == "selected" for sid, _ in sections):
            sections.append(("selected", dec["selected"]))
        if pad - pre > 1000:
            sections.append(("risk", "Item 1A. Risk Factors\n" + boiler(pad - pre)))
        order = [s for s, _ in sections]
        if rng.random() < 0.3:
            i, j = order.index("income"), order.index("balance")
            sections[i], sections[j] = sections[j], sections[i]
        docs.append("\n\n".join(t for _, t in sections))
        for sid, t in sections:
            if sid not in ("forward", "risk"):
                items.append((f"E{e}:{sid}", t))
        key_lines = {**inc.key_lines, **bs.key_lines, **cf.key_lines}
        ents.append({"name": name, "legal": legal, "dscale": dscale, "sscale": sscale, "style": style,
                     "lines": key_lines, "printed": printed, "has_nci": bool(vals["cur"]["ni_nci"]), "unlabeled_ta": unlabeled_ta,
                     "third_col": third, "prose": {"revenue": f"${rv / 1e9:.1f} billion",
                                                   "cash": f"${vals['cur']['cash'] / 1e9:.2f} billion",
                                                   "ni_company": f"${vals['cur']['ni_company'] / 1e6:,.0f} million"}})
    return {"entities": ents, "docs": docs, "items": items, "period": period_s, "prior_period": prior_s}


def co_value(ent: dict, key: str, per: str = "cur") -> float:
    """Full-unit value of a field (dollars, shares, or per-share) from what was PRINTED."""
    f = CO_FIELDS[key] if key in CO_FIELDS else {"unit": "usd"}
    v = ent["printed"][per][key]
    if f["unit"] == "usd":
        return v * ent["dscale"]
    if f["unit"] == "shares":
        return v * ent["sscale"]
    return v


SC_FIELDS: dict[str, dict[str, Any]] = {
    "grand_total": {"sec": "B1", "q": "grand total enrollment of all students"},
    "ug_total": {"sec": "B1", "q": "total number of undergraduate students"},
    "grad_total": {"sec": "B1", "q": "total number of graduate students"},
    "fy_applied": {"sec": "C1", "q": "number of first-time, first-year students who applied"},
    "fy_admitted": {"sec": "C1", "q": "number of first-time, first-year students admitted"},
    "fy_enrolled": {"sec": "C1", "q": "number of first-time, first-year students enrolled"},
    "tr_applied": {"sec": "D2", "q": "number of transfer applicants"},
    "tr_admitted": {"sec": "D2", "q": "number of admitted transfer students"},
    "tr_enrolled": {"sec": "D2", "q": "number of enrolled transfer students"},
    "grad_rate": {"sec": "B4", "q": "six-year graduation rate"},
    "ft_faculty": {"sec": "I1", "q": "number of full-time instructional faculty"},
    "pt_faculty": {"sec": "I1", "q": "number of part-time instructional faculty"},
    "sections_100": {"sec": "I3", "q": "number of undergraduate class sections with 100 or more students"},
    "pct_out_state": {"sec": "F1", "q": "percentage of first-year students from out of state"},
    "pct_housing": {"sec": "F1", "q": "percentage of first-year students living in college housing"},
}


def _school_values(rng: random.Random) -> dict[str, float]:
    ug = rng.randint(1800, 32000)
    grad = int(ug * rng.uniform(0.05, 0.9))
    applied = int(ug * rng.uniform(0.6, 4.5))
    rate = rng.uniform(0.12, 0.92)
    admitted = int(applied * rate)
    enrolled = int(admitted * rng.uniform(0.12, 0.55))
    tr_app = int(ug * rng.uniform(0.03, 0.35))
    tr_adm = int(tr_app * rng.uniform(0.25, 0.85))
    tr_enr = int(tr_adm * rng.uniform(0.3, 0.7))
    ft = int((ug + grad) / rng.uniform(9, 22))
    pt = int(ft * rng.uniform(0.05, 0.9))
    return {"ug_total": ug, "grad_total": grad, "grand_total": ug + grad, "fy_applied": applied, "fy_admitted": admitted,
            "fy_enrolled": enrolled, "tr_applied": tr_app, "tr_admitted": tr_adm, "tr_enrolled": tr_enr,
            "grad_rate": round(rng.uniform(42, 96)), "grad_rate_4": 0, "ft_faculty": ft, "pt_faculty": pt,
            "sections_100": rng.randint(0, 260), "pct_out_state": round(rng.uniform(4, 88)),
            "pct_housing": round(rng.uniform(35, 99)), "pct_out_state_ug": 0, "pct_housing_ug": 0}


def build_school(rng: random.Random, n: int, target_chars: int, year: int, names_ext: bool = False) -> dict:
    """Common Data Set-like files; each school's layout differs a little, so one regex won't parse them all."""
    ents, docs, items = [], [], []
    if names_ext and n > len(PLACES):
        places = rng.sample(PLACES_EXT, n)
    else:
        places = rng.sample(PLACES, n) if n <= len(PLACES) else [rng.choice(PLACES) + f" {i}" for i in range(n)]
    for e in range(n):
        name = rng.choice(SCHOOL_FORMS).format(p=places[e])
        v = _school_values(rng)
        v["grad_rate_4"] = max(10, v["grad_rate"] - rng.randint(8, 30))
        v["pct_out_state_ug"] = max(1, v["pct_out_state"] + rng.randint(-9, 9))
        v["pct_housing_ug"] = max(5, v["pct_housing"] - rng.randint(15, 50))
        style = rng.choice(["html", "pipe", "fixed"])
        key_lines: dict[str, str] = {}

        def table(rows, heads):
            st = _Statement(style, heads, dollars=False)
            for row in rows:
                st.row(*row)
            txt = st.text()
            key_lines.update(st.key_lines)
            return txt

        def prose(text, keys):
            for k in keys:
                key_lines[k] = text
            return text

        men = int(v["fy_applied"] * rng.uniform(0.4, 0.6))
        madm = int(v["fy_admitted"] * rng.uniform(0.4, 0.6))
        menr = int(v["fy_enrolled"] * rng.uniform(0.4, 0.6))
        hdr = f"# {name} Common Data Set {year}-{(year + 1) % 100:02d}"
        A = f"{hdr}\n\n# A. General Information\nA1. Name of College/University: {name}\nA2. Source of institutional control: {rng.choice(['Public', 'Private (nonprofit)'])}"
        b1 = rng.random()
        lab_ug = rng.choice(["Total all undergraduates", "Total undergraduates", "Undergraduate total"])
        lab_gr = rng.choice(["Total all graduate and professional students", "Total graduate", "Graduate & professional total"])
        lab_gt = rng.choice(["Grand Total All Students", "GRAND TOTAL", "Total, all students"])
        if b1 < 0.5:
            B1 = (f"# B. Enrollment and Persistence\nB1. Institutional Enrollment, Fall {year}\n" + table(
                [(lab_ug, [_m(v["ug_total"])], ["ug_total"]), (lab_gr, [_m(v["grad_total"])], ["grad_total"]),
                 (lab_gt, [_m(v["grand_total"])], ["grand_total"])], ["Total"]))
        elif b1 < 0.8:
            B1 = (f"# B. Enrollment and Persistence\nB1. Institutional Enrollment, Fall {year}\n"
                  + prose(f"{lab_ug}: {_m(v['ug_total'])}", ["ug_total"]) + "\n"
                  + prose(f"{lab_gr}: {_m(v['grad_total'])}", ["grad_total"]) + "\n"
                  + prose(f"{lab_gt}: {_m(v['grand_total'])}", ["grand_total"]))
        else:
            B1 = ("# B. Enrollment and Persistence\n" + prose(
                f"In fall {year} the institution enrolled {_m(v['grand_total'])} students in all, of whom "
                f"{_m(v['ug_total'])} were undergraduates and {_m(v['grad_total'])} were graduate or professional students.",
                ["grand_total", "ug_total", "grad_total"]))
        g_rows = [(f"Initial {year - 6} cohort of first-time, full-time bachelor's-seeking students", [_m(int(v["fy_enrolled"] * 0.97))], []),
                  (f"Four-year graduation rate for {year - 6} cohort", [f"{v['grad_rate_4']}%"], ["grad_rate_4"]),
                  (rng.choice([f"Six-year graduation rate for {year - 6} cohort", f"Six-year graduation rate for {year - 6} cohort (G divided by C)"]),
                   [f"{v['grad_rate']}%"], ["grad_rate"]),
                  (f"Six-year graduation rate for {year - 7} cohort", [f"{max(30, v['grad_rate'] + rng.randint(-6, 6))}%"], [])]
        rng.shuffle(g_rows)
        B4 = "Graduation Rates (B4-B11)\n" + table(g_rows, ["Total"])
        c1 = rng.random()
        c1_head = f"# C. First-Time, First-Year Admission\nC1. First-time, first-year (freshman) students, Fall {year}\n"
        if c1 < 0.3:
            C1 = c1_head + table([("Men", [_m(men), _m(madm), _m(menr)], []),
                                  ("Women", [_m(v["fy_applied"] - men), _m(v["fy_admitted"] - madm), _m(v["fy_enrolled"] - menr)], []),
                                  ("Total", [_m(v["fy_applied"]), _m(v["fy_admitted"]), _m(v["fy_enrolled"])], ["fy_applied", "fy_admitted", "fy_enrolled"])],
                                 ["Applied", "Admitted", "Enrolled"])
        elif c1 < 0.55:
            C1 = c1_head + table([("Applied", [_m(men), _m(v["fy_applied"] - men), _m(v["fy_applied"])], ["fy_applied"]),
                                  ("Admitted", [_m(madm), _m(v["fy_admitted"] - madm), _m(v["fy_admitted"])], ["fy_admitted"]),
                                  ("Enrolled", [_m(menr), _m(v["fy_enrolled"] - menr), _m(v["fy_enrolled"])], ["fy_enrolled"])],
                                 ["Men", "Women", "Total"])
        elif c1 < 0.8:
            C1 = c1_head + "\n".join([
                f"Total first-time, first-year men who applied {_m(men)}", f"Total first-time, first-year women who applied {_m(v['fy_applied'] - men)}",
                prose(f"Total first-time, first-year (degree-seeking) who applied {_m(v['fy_applied'])}", ["fy_applied"]),
                f"Total first-time, first-year men who were admitted {_m(madm)}", f"Total first-time, first-year women who were admitted {_m(v['fy_admitted'] - madm)}",
                prose(f"Total first-time, first-year (degree-seeking) who were admitted {_m(v['fy_admitted'])}", ["fy_admitted"]),
                f"Total full-time, first-time, first-year men who enrolled {_m(menr)}",
                prose(f"Total first-time, first-year (degree-seeking) who enrolled {_m(v['fy_enrolled'])}", ["fy_enrolled"])])
        else:
            C1 = c1_head + prose(
                f"The entering class was selected from {_m(v['fy_applied'])} first-year applicants ({_m(men)} men); "
                f"{_m(v['fy_admitted'])} were offered admission and {_m(v['fy_enrolled'])} enrolled, {_m(menr)} of them men.",
                ["fy_applied", "fy_admitted", "fy_enrolled"])
        D2 = (f"# D. Transfer Admission\nD2. Degree-seeking transfer students in fall {year}\n" + table(
            [("Men", [_m(int(v["tr_applied"] * 0.45)), _m(int(v["tr_admitted"] * 0.45)), _m(int(v["tr_enrolled"] * 0.45))], []),
             ("Total", [_m(v["tr_applied"]), _m(v["tr_admitted"]), _m(v["tr_enrolled"])], ["tr_applied", "tr_admitted", "tr_enrolled"])],
            ["Applicants", "Admitted", "Enrolled"]))
        f_rows = [("Percent who are from out of state", ["pct_out_state", "pct_out_state_ug"]),
                  ("Percent who live in college-owned, -operated, or -affiliated housing", ["pct_housing", "pct_housing_ug"])]
        swap = rng.random() < 0.4
        heads = ["First-time, first-year students", "Undergraduates"]
        rows = []
        for lab, (kf, ku) in f_rows:
            vals_ = [f"{v[kf]}%", f"{v[ku]}%"]
            rows.append((lab, vals_[::-1] if swap else vals_, [kf, ku]))
        rows.append(("Average age of full-time students", [str(rng.randint(19, 23)), str(rng.randint(18, 19))][::1 if swap else -1], []))
        F1 = ("# F. Student Life\nF1. Percentages of first-time, first-year (freshman) students and all degree-seeking undergraduates\n"
              + table(rows, heads[::-1] if swap else heads))
        I1 = (f"# I. Instructional Faculty and Class Size\nI1. Instructional Faculty - Fall {year}\n" + table(
            [(rng.choice(["a. Total instructional faculty", "Instructional faculty (total)", "a. Instructional Faculty"]),
              [_m(v["ft_faculty"]), _m(v["pt_faculty"]), _m(v["ft_faculty"] + v["pt_faculty"])], ["ft_faculty", "pt_faculty"]),
             ("b. Instructional faculty who are members of minority groups", [_m(int(v["ft_faculty"] * 0.2)), _m(int(v["pt_faculty"] * 0.15)), _m(int(v["ft_faculty"] * 0.2) + int(v["pt_faculty"] * 0.15))], []),
             ("c. Instructional faculty who are women", [_m(int(v["ft_faculty"] * 0.45)), _m(int(v["pt_faculty"] * 0.5)), _m(int(v["ft_faculty"] * 0.45) + int(v["pt_faculty"] * 0.5))], [])],
            ["Full-Time", "Part-Time", "Total"]))
        sizes = ["2-9", "10-19", "20-29", "30-39", "40-49", "50-99", "100+"]
        secs = [rng.randint(100, 1500), rng.randint(300, 1800), rng.randint(200, 1200), rng.randint(50, 600),
                rng.randint(20, 300), rng.randint(20, 300), v["sections_100"]]
        subs = [rng.randint(0, 400) for _ in sizes]
        sub_first = rng.random() < 0.4
        srows = [(rng.choice(["CLASS SECTIONS", "Class Sections", "Number of class sections"]), [_m(x) for x in secs] + [_m(sum(secs))], ["sections_100"]),
                 (rng.choice(["CLASS SUB-SECTIONS", "Class Sub-sections", "Number of class sub-sections"]), [_m(x) for x in subs] + [_m(sum(subs))], [])]
        I3 = f"I3. Undergraduate class size - Fall {year}\n" + table(srows[::-1] if sub_first else srows, sizes + ["Total"])
        brng = random.Random(rng.random())
        base = len(A) + len(B1) + len(B4) + len(C1) + len(D2) + len(F1) + len(I1) + len(I3)
        pad = max(0, target_chars - base)
        filler = []
        size = 0
        while size < pad:
            para = " ".join(brng.sample(SCHOOL_BOILER, 4))
            filler.append(para)
            size += len(para) + 2
        G = "# G. Annual Expenses and H. Financial Aid\n" + "\n\n".join(filler) if filler else ""
        sections = [("A", A), ("B1", B1), ("B4", B4), ("C1", C1), ("D2", D2), ("F1", F1)]
        if G:
            sections.append(("G", G))
        sections += [("I1", I1), ("I3", I3)]
        docs.append("\n\n".join(t for _, t in sections))
        for sid, t in sections:
            if sid != "G":
                items.append((f"E{e}:{sid}", t))
        ents.append({"name": name, "legal": name, "style": style, "lines": key_lines, "printed": {"cur": dict(v), "prior": {}}})
    return {"entities": ents, "docs": docs, "items": items, "period": f"Fall {year}"}


def _num(x: float) -> float:
    return round(float(x), 2)


def make_question(rng: random.Random, built: dict, dom: str) -> dict:
    ents = built["entities"]
    n = len(ents)
    noun = "companies" if dom == "company" else "universities"
    one = "company" if dom == "company" else "university"
    if dom == "company":
        def val(e, k, per="cur"):
            return co_value(e, k, per)
        usd = ["revenue", "net_income", "cash", "total_assets", "equity", "lt_debt", "retained", "cfo", "cfi", "cff",
               "pretax", "tax", "da"]
        ratios = [("net_income", "total_assets", "Return on Assets (ROA)"), ("net_income", "revenue", "profit margin"),
                  ("lt_debt", "equity", "Debt-to-Equity ratio"), ("cash", "total_assets", "cash-to-assets ratio"),
                  ("cfo", "net_income", "'Quality of Earnings' ratio"), ("revenue", "total_assets", "asset turnover")]
        pairs = [("cfo", "net_income"), ("da", "net_income"), ("cash", "lt_debt"), ("cff", "cfo"), ("retained", "equity")]
        fq = lambda k: CO_FIELDS[k]["q"]  # noqa: E731
    else:
        def val(e, k, per="cur"):
            return float(e["printed"]["cur"][k])
        usd = ["grand_total", "ug_total", "grad_total", "fy_applied", "fy_admitted", "fy_enrolled", "tr_enrolled",
               "ft_faculty", "pt_faculty", "sections_100", "grad_rate", "pct_out_state", "pct_housing"]
        ratios = [("fy_enrolled", "fy_admitted", "admissions yield (enrolled divided by admitted first-year students)"),
                  ("fy_admitted", "fy_applied", "admission rate (admitted divided by applied first-year students)"),
                  ("grand_total", "ft_faculty", "student-to-full-time-faculty ratio"),
                  ("tr_enrolled", "ug_total", "share of enrolled transfer students in the undergraduate population"),
                  ("pt_faculty", "ft_faculty", "ratio of part-time to full-time faculty")]
        pairs = [("grad_total", "ug_total"), ("pt_faculty", "ft_faculty"), ("tr_enrolled", "fy_enrolled"),
                 ("pct_out_state", "pct_housing")]
        fq = lambda k: SC_FIELDS[k]["q"]  # noqa: E731
    names = [e["name"] for e in ents]
    tmpl = rng.choices(["agg", "arg", "topk", "filter_cmp", "count_cond", "yoy", "pct_entities", "avg_filtered"],
                       weights=[2.5, 2, 2, 2, 1.5, 1.2 if dom == "company" else 0, 1, 1.2])[0]
    spec: dict[str, Any] = {"tmpl": tmpl}
    if tmpl == "agg":
        k = rng.choice(usd)
        pct_like = k in ("grad_rate", "pct_out_state", "pct_housing")
        op = rng.choice(["mean", "median", "min", "max"] if pct_like else ["sum", "mean", "median", "min", "max"])
        xs = [val(e, k) for e in ents]
        x = {"sum": sum, "mean": statistics.mean, "median": statistics.median, "min": min, "max": max}[op](xs)
        name = {"sum": "total sum of", "mean": "average", "median": "median", "min": "minimum reported", "max": "maximum reported"}[op]
        q = rng.choice([f"What is the {name} {fq(k)} across all {noun}?", f"Calculate the {name} {fq(k)} for all {noun} in the dataset."])
        spec.update(fields=[k], op=op)
        return dict(q=q, kind="number", gold=_num(x), spec=spec)
    if tmpl in ("arg", "topk"):
        if rng.random() < 0.5:
            k1, k2, rname = rng.choice(ratios)
            sc = {i: val(e, k1) / val(e, k2) for i, e in enumerate(ents) if val(e, k2)}
            what = rng.choice([f"the highest {rname} ({fq(k1)} over {fq(k2)})",
                               f"the highest {rname}, i.e. {fq(k1)} relative to {fq(k2)}"])
            fields = [k1, k2]
        else:
            k1 = rng.choice(usd)
            sc = {i: val(e, k1) for i, e in enumerate(ents)}
            what = f"the highest {fq(k1)}"
            fields = [k1]
        lo = rng.random() < 0.3
        if lo:
            what = what.replace("highest", "lowest")
        order = sorted(sc, key=lambda i: sc[i], reverse=not lo)
        if len(set(sc.values())) < len(sc):
            return make_question(rng, built, dom)
        spec.update(fields=fields, lo=lo)
        if tmpl == "arg":
            q = rng.choice([f"Name the {one} with {what}.", f"Which {one} reports {what}?",
                            f"Among the {noun}, which one shows {what}?"])
            return dict(q=q, kind="list", gold=[names[order[0]]], spec=spec)
        kk = rng.choice([3, 3, 5, 10])
        q = rng.choice([f"List the top {kk} {noun} with {what}.", f"Identify the {kk} {noun} with {what}."])
        spec["k"] = kk
        return dict(q=q, kind="list", gold=[names[i] for i in order[:kk]], spec=spec)
    if tmpl == "filter_cmp":
        for _ in range(8):
            k1, k2 = rng.choice(pairs)
            if rng.random() < 0.5:
                k1, k2 = k2, k1
            gold = [names[i] for i, e in enumerate(ents) if val(e, k1) > val(e, k2)]
            if 0 < len(gold) < n or rng.random() < 0.05:
                break
        else:
            return make_question(rng, built, dom)
        q = f"List all {noun} where the {fq(k1)} exceeded the {fq(k2)}."
        spec.update(fields=[k1, k2])
        return dict(q=q, kind="list", gold=gold, spec=spec)
    if tmpl == "count_cond":
        if dom == "company" and rng.random() < 0.5:
            k = rng.choice(["net_income", "cfo", "cfi", "cff", "retained"])
            n_ = sum(1 for e in ents if val(e, k) < 0)
            q = f"How many {noun} reported a negative {fq(k)}?"
            spec.update(fields=[k])
            return dict(q=q, kind="number", gold=float(n_), spec=spec)
        k1, k2, rname = rng.choice(ratios)
        rs = sorted(val(e, k1) / val(e, k2) for e in ents if val(e, k2))
        thr = round(rs[len(rs) // 2] * rng.uniform(0.85, 1.15), 3 if abs(rs[len(rs) // 2]) < 1 else 1)
        n_ = sum(1 for r in rs if r < thr)
        q = f"How many {noun} have a {rname} ({fq(k1)} divided by {fq(k2)}) below {thr}?"
        spec.update(fields=[k1, k2], thr=thr)
        return dict(q=q, kind="number", gold=float(n_), spec=spec)
    if tmpl == "yoy":
        k = rng.choice(["net_income", "revenue", "shares_diluted", "eps_diluted", "pretax"])
        up = rng.random() < 0.6
        gold = [names[i] for i, e in enumerate(ents) if (val(e, k) > val(e, k, "prior")) == up and val(e, k) != val(e, k, "prior")]
        if not gold and rng.random() > 0.05:
            return make_question(rng, built, dom)
        q = rng.choice([f"Which {noun} reported {'higher' if up else 'lower'} {fq(k)} than in the same period one year "
                        "earlier?",
                        f"List the {noun} whose {fq(k)} {'grew' if up else 'shrank'} against the comparable prior-year "
                        "period."])
        spec.update(fields=[k], prior=True, up=up)
        return dict(q=q, kind="list", gold=gold, spec=spec)
    if tmpl == "pct_entities":
        k1, k2, rname = rng.choice(ratios)
        rs = [val(e, k1) / val(e, k2) for e in ents if val(e, k2)]
        thr = round(sorted(rs)[len(rs) // 2] * rng.uniform(0.8, 1.2), 3)
        pc = sum(1 for r in rs if r > thr) / len(ents) * 100
        q = f"What percentage of {noun} in the dataset have a {rname} ({fq(k1)} divided by {fq(k2)}) above {thr}?"
        spec.update(fields=[k1, k2], thr=thr)
        return dict(q=q, kind="str", gold=f"{pc:.2f}%", spec=spec)
    ka = rng.choice(usd)
    kb = rng.choice([k for k in usd if k != ka])
    med = statistics.median(val(e, kb) for e in ents)
    sel = [val(e, ka) for e in ents if val(e, kb) > med]
    if not sel:
        return make_question(rng, built, dom)
    q = f"What is the average {fq(ka)} for {noun} whose {fq(kb)} is above the dataset's median {fq(kb)}?"
    spec.update(fields=[ka, kb])
    return dict(q=q, kind="number", gold=_num(statistics.mean(sel)), spec=spec)
