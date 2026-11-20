"""Deterministic filler text (10-K prose, code comments, chat lines), built once at import."""

from __future__ import annotations

import random

_R = random.Random(20261013)

_SUBJ = ["The Company", "{CO}", "We", "Management", "Our business", "The Board of Directors", "Our operations",
         "Each reportable segment", "Our international business", "The Audit Committee", "Our supply chain",
         "Our customers", "Our workforce", "The registrant", "Our distribution network"]
_VERB = ["continued to invest in", "expects to expand", "may be adversely affected by", "relies significantly on",
         "periodically evaluates", "is exposed to", "has historically benefited from", "monitors", "seeks to reduce",
         "cannot predict the outcome of", "is subject to", "regularly reviews", "intends to fund",
         "recognized charges related to", "completed the integration of"]
_OBJ = ["changes in foreign currency exchange rates", "the timing of customer orders", "cybersecurity incidents",
        "raw material availability and pricing", "its revolving credit facility", "goodwill and intangible assets",
        "inventory levels across distribution centers", "pension and postretirement obligations", "competition",
        "litigation and regulatory proceedings", "interest rate fluctuations", "new product introductions",
        "the effective tax rate", "share repurchase authorizations", "capital expenditures for new facilities",
        "the recoverability of deferred tax assets", "the allowance for credit losses", "lease obligations",
        "marketable securities held as available-for-sale", "working capital requirements", "total assets under management",
        "the fair value of derivative instruments", "net sales in emerging markets", "operating expenses"]
_TAIL = ["during fiscal {Y}", "as of the end of fiscal {Y}", "compared with fiscal {PY}",
         "over the next twelve months", "in each of the last three fiscal years", "as described in Note {N}",
         "within Item 7 of this Annual Report", "as discussed under Risk Factors", "in recent periods",
         "for the year ended {DATE}"]
_NUMS = ["approximately {a},{b:03d} employees", "{c} manufacturing facilities in {d} countries",
         "a {p}.{q}% increase in unit volume", "${a}.{q} billion of remaining performance obligations",
         "{c} distribution centers", "an aggregate principal amount of ${a},{b:03d} million",
         "{p}.{q}% of consolidated net sales", "{a},{b:03d} active customer accounts", "{c} patents",
         "${c} million of restructuring charges", "a weighted-average remaining term of {d}.{q} years",
         "${a}.{q} billion of unused borrowing capacity", "{c}% of the outstanding shares"]
_BOILER = [
    "This Annual Report on Form 10-K contains forward-looking statements within the meaning of the Private Securities "
    "Litigation Reform Act of 1995. Words such as anticipate, believe, estimate, expect, intend and similar expressions "
    "identify forward-looking statements, which speak only as of the date they are made.",
    "The preparation of financial statements in conformity with accounting principles generally accepted in the United "
    "States requires management to make estimates and assumptions that affect the reported amounts of assets and "
    "liabilities and the disclosure of contingent assets and liabilities at the date of the financial statements.",
    "Disclosure controls and procedures are designed to ensure that information required to be disclosed in the "
    "reports filed under the Exchange Act is recorded, processed, summarized and reported within the time periods "
    "specified in the rules and forms of the Securities and Exchange Commission.",
    "Revenue is recognized when control of the promised goods or services is transferred to customers in an amount "
    "that reflects the consideration expected to be received in exchange for those goods or services.",
    "Inventories are stated at the lower of cost and net realizable value, with cost determined on a first-in, "
    "first-out basis. Reserves are established for excess and obsolete inventory based on forecasted demand.",
    "Property, plant and equipment are recorded at cost and depreciated on a straight-line basis over the estimated "
    "useful lives of the assets, which range from three to forty years. Leasehold improvements are amortized over the "
    "shorter of the lease term or the useful life.",
    "Goodwill is tested for impairment annually during the fourth quarter, or more frequently if events or changes in "
    "circumstances indicate that the carrying amount may not be recoverable.",
    "Marketable securities consist primarily of corporate debt securities, U.S. treasury securities and commercial "
    "paper classified as available-for-sale and reported at fair value, with unrealized gains and losses recorded in "
    "accumulated other comprehensive income.",
]


def _fill(s: str, r: random.Random) -> str:
    return s.format(a=r.randint(1, 98), b=r.randint(0, 999), c=r.randint(3, 480), d=r.randint(2, 60),
                    p=r.randint(1, 39), q=r.randint(0, 9), N=r.randint(2, 21), Y="{Y}", PY="{PY}", CO="{CO}",
                    DATE="{DATE}")


def _sentence(r: random.Random) -> str:
    s = f"{r.choice(_SUBJ)} {r.choice(_VERB)} {r.choice(_OBJ)} {r.choice(_TAIL)}"
    if r.random() < 0.55:
        s += ", including " + r.choice(_NUMS)
    return _fill(s + ".", r)


def _paragraph(r: random.Random) -> str:
    if r.random() < 0.15:
        return r.choice(_BOILER)
    return " ".join(_sentence(r) for _ in range(r.randint(3, 7)))


PARAS: list[str] = [_paragraph(_R) for _ in range(1600)]


_CODE_WORDS = ["config", "payload", "record", "buffer", "result", "session", "item", "batch", "token", "handle",
               "options", "state", "entry", "frame", "cache", "spec", "node", "chunk", "context", "params"]
_CODE_VERBS = ["normalize", "validate", "serialize", "merge", "flatten", "resolve", "encode", "decode", "split",
               "collect", "render", "prepare", "compute", "filter", "update", "sync", "refresh", "build", "check"]


def code_comment(r: random.Random) -> str:
    return f"# {r.choice(_CODE_VERBS).capitalize()} the {r.choice(_CODE_WORDS)} before the {r.choice(_CODE_WORDS)} is used."


def code_stmt(r: random.Random, var: str) -> str:
    w = r.choice(_CODE_WORDS)
    k = r.randint(0, 9)
    choices = [
        f"{var} = {{}}" if k < 2 else f"{var} = [{r.randint(0, 9)}, {r.randint(10, 99)}]",
        f"if {var} is None:\n        {var} = {r.randint(0, 99)}",
        f"for _{w} in range({r.randint(2, 9)}):\n        {var} = _{w}",
        f"{var} = str({var}).strip().lower()",
        f"{var} = len({var}) if isinstance({var}, (list, tuple)) else {var}",
        f"{var} = os.path.join(\"{w}\", str({var}))",
        f"{var} = json.dumps({{\"{w}\": {var}}})",
        f"logger.debug(\"{r.choice(_CODE_VERBS)} %s\", {var})",
    ]
    return r.choice(choices)


DOC_PARAS: list[str] = [_paragraph(_R).replace("{CO}", "the project").replace("{Y}", "2024").replace("{PY}", "2023")
                        .replace("{DATE}", "June 30, 2024") for _ in range(300)]

__all__ = ["DOC_PARAS", "PARAS", "code_comment", "code_stmt"]
