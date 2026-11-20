"""Finance family: synthetic multi-company 10-K filings for LooGLE-v2 cross-company comparison / trend questions."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from dataclasses import dataclass, field

from synth_common.envkit import Generated
from synth_common.oracle import (
    FakeSubLM,
    ReadModel,
    refuse_aggregate,
    rng_for,
    stable_hash,
    wants_json,
)

from .textbank import PARAS

CONTEXT_POINTER = ("[The full document is NOT reproduced here. It is available in the REPL as the `context` string "
                   "variable; query it with `llm_query` over slices.]")
INSTRUCTION = ("You are a senior financial analyst. You will be provided with certain fiscal year's annual reports for "
               "one or more companies. In the provided financial reports, any data enclosed in parentheses () within "
               "tables is to be interpreted as a negative value. The text below contains all the 10-K annual reports in "
               "the format <FILE: filename> followed by the file content. <text> {context} </text> Your task is to "
               "analyze the financial information and answer the following question: <question> {question} </question>.")

LABELS: dict[str, tuple[str, list[str]]] = {
    "Cash": ("BS", ["Cash and cash equivalents", "Cash and equivalents", "Cash and cash equivalents (Note 4)"]),
    "MarketableSecurities": ("BS", ["Marketable securities", "Short-term investments", "Short-term marketable securities",
                                    "Investment securities, current"]),
    "AccountsReceivableNet": ("BS", ["Accounts receivable, net of allowances of ${a1} and ${a0}",
                                     "Trade receivables (less allowances of ${a1} in {Y} and ${a0} in {PY})",
                                     "Accounts receivable, net", "Receivables, net (Note 6)"]),
    "InventoryNet": ("BS", ["Inventories", "Inventories, net", "Merchandise inventories", "Inventories (Note 7)"]),
    "CurrentAssets": ("BS", ["Total current assets", "Total Current Assets", "Total current assets"]),
    "PropertyPlantEquipmentNet": ("BS", ["Property, plant and equipment, net", "Property and equipment, net",
                                         "Property, plant and equipment, net of accumulated depreciation of ${d1} and ${d0}",
                                         "Net property, plant and equipment"]),
    "TotalAssets": ("BS", ["Total assets", "Total Assets", "TOTAL ASSETS"]),
    "AccountsPayable": ("BS", ["Accounts payable", "Trade accounts payable", "Accounts payable (Note 9)"]),
    "CurrentLiabilities": ("BS", ["Total current liabilities", "Total Current Liabilities"]),
    "LongTermDebt": ("BS", ["Long-term debt", "Long-term debt, net of current portion", "Long-term debt, less current maturities",
                            "Long-term borrowings"]),
    "TotalLiabilities": ("BS", ["Total liabilities", "Total Liabilities", "TOTAL LIABILITIES"]),
    "ShareholdersEquity": ("BS", ["Total {CO} stockholders' equity", "Total shareholders' equity", "Total stockholders’ equity",
                                  "Total shareowners' equity"]),
    "Revenue": ("IS", ["Net sales", "Total revenues", "Revenues", "Net revenues", "Total net sales"]),
    "CostOfGoodsSold": ("IS", ["Cost of sales", "Cost of revenues", "Cost of goods sold", "Cost of products sold"]),
    "GrossProfit": ("IS", ["Gross profit", "Gross margin"]),
    "OperatingExpenses": ("IS", ["Total operating expenses", "Total operating expenses", "Operating expenses, total"]),
    "OperatingIncomeLoss": ("IS", ["Operating income", "Income from operations", "Operating profit", "Operating income (loss)"]),
    "InterestExpense": ("IS", ["Interest expense", "Interest expense", "Interest expense (Note 11)"]),
    "IncomeTaxExpenseBenefit": ("IS", ["Provision for income taxes", "Income tax expense", "Income taxes"]),
    "NetIncome": ("IS", ["Net income", "Net earnings", "Net income"]),
    "DepreciationAmortization": ("CF", ["Depreciation and amortization", "Depreciation, depletion and amortization",
                                        "Depreciation and amortization expense"]),
    "OperatingCashFlow": ("CF", ["Net cash provided by operating activities", "Cash provided by operating activities",
                                 "Net cash from operating activities"]),
    "CapitalExpenditure": ("CF", ["Capital expenditures", "Purchases of property, plant and equipment",
                                  "Payments for property and equipment", "Additions to property, plant and equipment"]),
    "DividendsPaid": ("CF", ["Dividends paid", "Cash dividends paid", "Dividends paid to shareholders"]),
}
POSITIVE_OUTFLOWS = {"CapitalExpenditure", "DividendsPaid"}

FORMULAS: dict[str, tuple[str, bool, bool, list[str]]] = {
    "WorkingCapital": ("CurrentAssets - CurrentLiabilities", False, False,
                       ["WorkingCapital = CurrentAssets - CurrentLiabilities",
                        "Working Capital is defined as: Current Assets - Current Liabilities. This measures the difference "
                        "between a company's short-term assets and its short-term liabilities."]),
    "CurrentRatio": ("CurrentAssets / CurrentLiabilities", False, False,
                     ["CurrentRatio = CurrentAssets / CurrentLiabilities",
                      "Current Ratio is defined as: Current Assets / Current Liabilities. This measures a company's ability "
                      "to cover its short-term liabilities with its short-term assets."]),
    "QuickRatio": ("(CurrentAssets - InventoryNet) / CurrentLiabilities", False, False,
                   ["QuickRatio = (CurrentAssets - InventoryNet) / CurrentLiabilities"]),
    "FreeCashFlow": ("OperatingCashFlow - CapitalExpenditure", False, False,
                     ["Free Cash Flow is defined as: Operating Cash Flow - Capital Expenditure. This measures the cash a "
                      "company generates after accounting for capital expenditures."]),
    "GrossMargin": ("(Revenue - CostOfGoodsSold) / Revenue * 100", False, True,
                    ["Gross Margin is defined as: (Revenue - Cost of Goods Sold) / Revenue * 100%. This measures the "
                     "percentage of revenue that exceeds the cost of goods sold."]),
    "OperatingIncomeMargin": ("OperatingIncomeLoss / Revenue * 100", False, True,
                              ["Operating Income Margin is defined as: Operating Income Loss / Revenue * 100%. This "
                               "measures the percentage of revenue that remains after operating expenses are paid.",
                               "OperatingIncomeMargin = OperatingIncomeLoss / Revenue * 100% (Percentage)"]),
    "NetProfitMargin": ("NetIncome / Revenue * 100", False, True, ["NetProfitMargin = NetIncome / Revenue * 100% (Percentage)"]),
    "COGSMargin": ("CostOfGoodsSold / Revenue * 100", False, True,
                   ["COGS Margin is defined as: Cost of Goods Sold / Revenue * 100%. This measures the percentage of "
                    "revenue that is consumed by the direct costs of production."]),
    "D&AMargin": ("DepreciationAmortization / Revenue * 100", False, True,
                  ["D&A Margin is defined as: Depreciation and Amortization / Revenue * 100%. This measures the "
                   "percentage of revenue allocated to depreciation and amortization."]),
    "OperatingCashFlowRatio": ("OperatingCashFlow / CurrentLiabilities", False, False,
                               ["Operating Cash Flow Ratio is defined as: Operating Cash Flow / Current Liabilities. This "
                                "measures a company's ability to cover its short-term liabilities with cash flow."]),
    "CashToRevenueRatio": ("Cash / Revenue", False, False,
                           ["Cash to Revenue Ratio is defined as: Cash and Cash Equivalents / Revenue. This measures the "
                            "proportion of cash a company has relative to its revenue."]),
    "AccountsReceivableTurnover": ("Revenue / ((AccountsReceivableNet + AccountsReceivableNet_prev) / 2)", True, False,
                                   ["Accounts Receivable Turnover is defined as: Revenue / ((Accounts Receivable Net + "
                                    "Accounts Receivable Net from the previous year) / 2). This measures how efficiently a "
                                    "company collects its receivables."]),
    "InventoryTurnoverRatio": ("CostOfGoodsSold / ((InventoryNet + InventoryNet_prev) / 2)", True, False,
                               ["InventoryTurnoverRatio = CostOfGoodsSold / ((InventoryNet + InventoryNetPrev) / 2)"]),
    "AssetTurnoverRatio": ("Revenue / ((TotalAssets + TotalAssets_prev) / 2)", True, False,
                           ["Asset Turnover Ratio is defined as: Revenue / ((Total Assets + Total Assets from the previous "
                            "year) / 2). This measures how efficiently a company uses its assets to generate revenue."]),
    "ReturnOnAssets": ("NetIncome / ((TotalAssets + TotalAssets_prev) / 2) * 100", True, True,
                       ["ReturnOnAssets = NetIncome / ((TotalAssets + TotalAssets_Pre) / 2) * 100% (Percentage)"]),
}
QNAME = {"Cash": "Cash(CashAndCashEquivalents)"}

BASE_W = {"MarketableSecurities": 11, "CurrentAssets": 9, "LongTermDebt": 7, "CurrentLiabilities": 6, "AccountsPayable": 5,
          "PropertyPlantEquipmentNet": 5, "Cash": 5, "InterestExpense": 5, "TotalLiabilities": 5, "NetIncome": 5,
          "DepreciationAmortization": 4, "ShareholdersEquity": 4, "AccountsReceivableNet": 3, "OperatingCashFlow": 3,
          "IncomeTaxExpenseBenefit": 2, "DividendsPaid": 2, "TotalAssets": 2, "CostOfGoodsSold": 2,
          "OperatingIncomeLoss": 2, "GrossProfit": 1, "InventoryNet": 1, "OperatingExpenses": 1, "Revenue": 1,
          "CapitalExpenditure": 1}
KIND_W = {"trend": 36, "stable": 27, "count": 13, "rank": 12, "maxmin": 12, "hilo": 8}
FORMULA_P = 0.4


_ROOTS = ["Alder", "Bexley", "Calder", "Dunmore", "Elstrom", "Fairbrook", "Garrick", "Halvorsen", "Ivers", "Jarrow",
          "Kestrel", "Larchmont", "Marlow", "Norcastle", "Oakhurst", "Pellham", "Quarrington", "Ravensby", "Selwyn",
          "Thornbury", "Upworth", "Varden", "Westcott", "Yarrow", "Zennor", "Ashgrove", "Brantley", "Cobalt Ridge",
          "Delmore", "Everly", "Foxhall", "Glenrock", "Harwick", "Inglewood", "Juniper Bay", "Kingsmere", "Lockhart",
          "Merriweather", "Northvale", "Orchard Point", "Prescott", "Redfern", "Stanmore", "Tremont", "Underhill",
          "Vantage Hill", "Whitlock", "Altamira", "Bellhaven", "Crestmark"]
_INDUSTRY = ["Dynamics", "Foods", "Therapeutics", "Semiconductor", "Energy", "Logistics", "Systems", "Brands",
             "Industries", "Pharmaceuticals", "Networks", "Materials", "Health", "Software", "Motors", "Aerospace",
             "Financial", "Retail", "Communications", "Instruments", "Chemicals", "Biosciences", "Technologies",
             "Beverages", "Medical"]
_SUFFIX_Q = [("Inc.", "Inc."), (", Inc.", ", Inc."), ("Corporation", "Corporation"), ("CORP", "Corporation"),
             ("CO", "Company"), ("& Co., Inc.", "& Co., Inc."), ("PLC", "plc"), ("INC/DE", "Inc."),
             ("COMPANY", "Company"), ("CORP /NEW", "Corporation"), ("Group Incorporated", "Group Incorporated"), ("Inc", "Inc.")]
_SEGMENTS = ["Americas", "International", "Consumer", "Industrial", "Healthcare", "Services", "Products", "Europe",
             "Asia Pacific", "Commercial", "Specialty", "Energy Solutions"]


@dataclass
class Company:
    qname: str
    cover: str
    tkr: str
    fye: tuple[int, int]
    unit: str
    order: str
    dollar: bool
    nci: bool
    has_ms: bool
    has_inv: bool
    payer: bool
    selected: bool
    restricted: bool
    file_style: int
    labels: dict = field(default_factory=dict)
    segs: list = field(default_factory=list)
    vals: dict = field(default_factory=dict)


def _make_company(rng: random.Random, used: set) -> Company:
    for _ in range(100):
        root, ind = rng.choice(_ROOTS), rng.choice(_INDUSTRY)
        if (root, ind) not in used and root not in {u[0] for u in used}:
            break
    used.add((root, ind))
    sq, sc = rng.choice(_SUFFIX_Q)
    caps = rng.random() < 0.45
    joiner = "" if sq.startswith(",") else " "
    qname = f"{root} {ind}{joiner}{sq}"
    if caps:
        qname = qname.upper()
    cover = f"{'The ' if sc == 'Company' and rng.random() < 0.5 else ''}{root} {ind}{'' if sc.startswith(',') else ' '}{sc}"
    if rng.random() < 0.5:
        cover = cover.upper()
    letters = re.sub(r"[^a-z]", "", (root + ind).lower())
    tkr = (letters[:2] + letters[len(root.replace(" ", "")):][:2]) if rng.random() < 0.7 else letters[:3]
    fye = rng.choices([(12, 31), (9, 25), (6, 30), (8, 31), (10, 1)], weights=[74, 9, 8, 5, 4])[0]
    co = Company(qname=qname, cover=cover, tkr=tkr, fye=fye,
                 unit=rng.choices(["millions", "thousands", "millions1"], weights=[64, 24, 12])[0],
                 order="asc" if rng.random() < 0.22 else "desc", dollar=rng.random() < 0.7, nci=rng.random() < 0.25,
                 has_ms=rng.random() < 0.85, has_inv=rng.random() < 0.85, payer=rng.random() < 0.8,
                 selected=rng.random() < 0.6, restricted=rng.random() < 0.3, file_style=rng.choice([0, 0, 0, 1, 2]))
    co.labels = {m: rng.choice(v[1]) for m, v in LABELS.items()}
    if co.nci:
        co.labels["NetIncome"] = "Net income attributable to {CO}"
    co.segs = rng.sample(_SEGMENTS, rng.choice([2, 3]))
    return co


YEARS = list(range(2014, 2026))


def _gen_values(co: Company, rng: random.Random) -> None:
    rev = math.exp(rng.uniform(math.log(600), math.log(250_000)))
    mu, sd = rng.uniform(-0.02, 0.12), rng.uniform(0.02, 0.16)
    cogs_r, rd_r, sga_r = rng.uniform(0.25, 0.72), rng.uniform(0.0, 0.16), rng.uniform(0.08, 0.28)
    debt_r = rng.uniform(0.08, 1.0)
    ratios = {k: rng.uniform(*v) for k, v in dict(cash=(0.04, 0.35), ms=(0.01, 0.3), ar=(0.07, 0.2), inv=(0.05, 0.2),
                                                       oca=(0.02, 0.06), ppe=(0.1, 1.1), lti=(0.0, 0.3), gw=(0.0, 0.8),
                                                       intang=(0.0, 0.3), onca=(0.02, 0.1), ap=(0.08, 0.25),
                                                       accr=(0.03, 0.1), ocl=(0.01, 0.05), dt=(0.0, 0.05),
                                                       oncl=(0.02, 0.15), payout=(0.15, 0.6), tax=(0.12, 0.27)).items()}
    prev = None
    for y in YEARS:
        rev *= 1 + rng.gauss(mu, sd)
        rev = max(rev, 150.0)
        j = lambda: 1 + rng.gauss(0, 0.08)  # noqa: E731  per-line jitter
        v: dict[str, float] = {}
        v["Revenue"] = rev
        v["CostOfGoodsSold"] = rev * cogs_r * (1 + rng.gauss(0, 0.03))
        v["GrossProfit"] = v["Revenue"] - v["CostOfGoodsSold"]
        v["RD"] = rev * rd_r * j()
        v["SGA"] = rev * sga_r * j()
        v["OperatingExpenses"] = v["RD"] + v["SGA"]
        v["OperatingIncomeLoss"] = v["GrossProfit"] - v["OperatingExpenses"]
        debt = rev * debt_r * j()
        v["InterestExpense"] = debt * rng.uniform(0.02, 0.06)
        v["OtherInc"] = rev * rng.gauss(0, 0.01)
        v["Pretax"] = v["OperatingIncomeLoss"] - v["InterestExpense"] + v["OtherInc"]
        v["IncomeTaxExpenseBenefit"] = v["Pretax"] * ratios["tax"] * (1 + rng.gauss(0, 0.1))
        v["NetIncomeTotal"] = v["Pretax"] - v["IncomeTaxExpenseBenefit"]
        v["NCIShare"] = abs(v["NetIncomeTotal"]) * rng.uniform(0.02, 0.08) if co.nci else 0.0
        v["NetIncome"] = v["NetIncomeTotal"] - v["NCIShare"]
        v["Cash"] = rev * ratios["cash"] * j()
        v["MarketableSecurities"] = rev * ratios["ms"] * j() if co.has_ms else 0.0
        v["AccountsReceivableNet"] = rev * ratios["ar"] * j()
        v["InventoryNet"] = rev * ratios["inv"] * j() if co.has_inv else 0.0
        v["OtherCA"] = rev * ratios["oca"] * j()
        v["PropertyPlantEquipmentNet"] = rev * ratios["ppe"] * j()
        v["LTInvest"] = rev * ratios["lti"] * j()
        v["Goodwill"] = rev * ratios["gw"] * j()
        v["Intangibles"] = rev * ratios["intang"] * j()
        v["OtherNCA"] = rev * ratios["onca"] * j()
        v["AccountsPayable"] = v["CostOfGoodsSold"] * ratios["ap"] * j()
        v["Accrued"] = rev * ratios["accr"] * j()
        v["STDebt"] = debt * rng.uniform(0.03, 0.15)
        v["OtherCL"] = rev * ratios["ocl"] * j()
        v["LongTermDebt"] = debt - v["STDebt"]
        v["DeferredTax"] = rev * ratios["dt"] * j()
        v["OtherNCL"] = rev * ratios["oncl"] * j()
        v["DepreciationAmortization"] = v["PropertyPlantEquipmentNet"] * rng.uniform(0.08, 0.15)
        v["CapitalExpenditure"] = v["DepreciationAmortization"] * rng.uniform(0.7, 1.6)
        v["SBC"] = rev * rng.uniform(0.005, 0.03)
        v["DividendsPaid"] = max(0.0, v["NetIncome"]) * ratios["payout"] * j() if co.payer else 0.0
        v["Buybacks"] = rev * rng.uniform(0.0, 0.05)
        v["RestrictedCash"] = rev * rng.uniform(0.002, 0.02) if co.restricted else 0.0
        co.vals[y] = v
        _recompute(co, y, prev)
        prev = y


def _recompute(co: Company, y: int, prev_y: int | None, targets: dict | None = None) -> None:
    """Totals from components; ``targets`` pin a total (then its 'other' plug line absorbs the difference)."""
    v = co.vals[y]
    t = targets or {}
    if "CurrentAssets" in t:
        base = v["Cash"] + v["MarketableSecurities"] + v["AccountsReceivableNet"] + v["InventoryNet"]
        if base > 0.97 * t["CurrentAssets"]:
            f = 0.9 * t["CurrentAssets"] / base
            for k in ("Cash", "MarketableSecurities", "AccountsReceivableNet", "InventoryNet"):
                v[k] *= f
            base *= f
        v["OtherCA"] = t["CurrentAssets"] - base
    v["CurrentAssets"] = v["Cash"] + v["MarketableSecurities"] + v["AccountsReceivableNet"] + v["InventoryNet"] + v["OtherCA"]
    if "CurrentLiabilities" in t:
        base = v["AccountsPayable"] + v["Accrued"] + v["STDebt"]
        if base > 0.97 * t["CurrentLiabilities"]:
            f = 0.9 * t["CurrentLiabilities"] / base
            for k in ("AccountsPayable", "Accrued", "STDebt"):
                v[k] *= f
            base *= f
        v["OtherCL"] = t["CurrentLiabilities"] - base
    v["CurrentLiabilities"] = v["AccountsPayable"] + v["Accrued"] + v["STDebt"] + v["OtherCL"]
    nca = v["PropertyPlantEquipmentNet"] + v["LTInvest"] + v["Goodwill"] + v["Intangibles"]
    if "TotalAssets" in t:
        v["OtherNCA"] = max(0.01 * t["TotalAssets"], t["TotalAssets"] - v["CurrentAssets"] - nca)
    v["TotalAssets"] = v["CurrentAssets"] + nca + v["OtherNCA"]
    if "TotalLiabilities" in t:
        v["OtherNCL"] = max(0.01 * t["TotalLiabilities"], t["TotalLiabilities"] - v["CurrentLiabilities"] - v["LongTermDebt"]
                            - v["DeferredTax"])
    v["TotalLiabilities"] = v["CurrentLiabilities"] + v["LongTermDebt"] + v["DeferredTax"] + v["OtherNCL"]
    eq = v["TotalAssets"] - v["TotalLiabilities"]
    if "ShareholdersEquity" in t or eq < 0.08 * v["TotalAssets"]:
        want = t.get("ShareholdersEquity", 0.15 * v["TotalAssets"])
        v["OtherNCA"] += want * (1.03 if co.nci else 1.0) - eq
        v["TotalAssets"] = v["CurrentAssets"] + nca + v["OtherNCA"]
        eq = v["TotalAssets"] - v["TotalLiabilities"]
    v["NCIEquity"] = eq * 0.03 / 1.03 if co.nci else 0.0
    v["ShareholdersEquity"] = eq - v["NCIEquity"]
    v["TotalEquity"] = eq
    if "GrossProfit" in t:
        v["CostOfGoodsSold"] = v["Revenue"] - t["GrossProfit"]
    v["GrossProfit"] = v["Revenue"] - v["CostOfGoodsSold"]
    if "OperatingExpenses" in t:
        v["SGA"] = t["OperatingExpenses"] - v["RD"]
    v["OperatingExpenses"] = v["RD"] + v["SGA"]
    if "OperatingIncomeLoss" in t:
        v["SGA"] = v["GrossProfit"] - v["RD"] - t["OperatingIncomeLoss"]
        v["OperatingExpenses"] = v["RD"] + v["SGA"]
    v["OperatingIncomeLoss"] = v["GrossProfit"] - v["OperatingExpenses"]
    v["Pretax"] = v["OperatingIncomeLoss"] - v["InterestExpense"] + v["OtherInc"]
    if "NetIncome" in t:
        v["NCIShare"] = abs(t["NetIncome"]) * 0.04 if co.nci else 0.0
        v["NetIncomeTotal"] = t["NetIncome"] + v["NCIShare"]
        v["OtherInc"] = v["NetIncomeTotal"] + v["IncomeTaxExpenseBenefit"] - v["OperatingIncomeLoss"] + v["InterestExpense"]
        v["Pretax"] = v["OperatingIncomeLoss"] - v["InterestExpense"] + v["OtherInc"]
    else:
        v["NetIncomeTotal"] = v["Pretax"] - v["IncomeTaxExpenseBenefit"]
        v["NetIncome"] = v["NetIncomeTotal"] - v["NCIShare"]
    p = co.vals.get(prev_y) if prev_y is not None else None
    d_ar = -(v["AccountsReceivableNet"] - p["AccountsReceivableNet"]) if p else -0.02 * v["AccountsReceivableNet"]
    d_inv = -(v["InventoryNet"] - p["InventoryNet"]) if p else 0.0
    d_ap = (v["AccountsPayable"] - p["AccountsPayable"]) if p else 0.01 * v["AccountsPayable"]
    v["dAR"], v["dInv"], v["dAP"] = d_ar, d_inv, d_ap
    if "OperatingCashFlow" in t:
        v["OtherOp"] = t["OperatingCashFlow"] - (v["NetIncomeTotal"] + v["DepreciationAmortization"] + v["SBC"] + d_ar + d_inv + d_ap)
    else:
        v.setdefault("OtherOp", v["Revenue"] * 0.004)
    v["OperatingCashFlow"] = v["NetIncomeTotal"] + v["DepreciationAmortization"] + v["SBC"] + d_ar + d_inv + d_ap + v["OtherOp"]
    v["InvestingCF"] = -v["CapitalExpenditure"] - 0.1 * v["MarketableSecurities"]
    prev_cash = p["Cash"] + p.get("RestrictedCash", 0.0) if p else v["Cash"] * 0.9
    v["FinancingCF"] = (v["Cash"] + v["RestrictedCash"] - prev_cash) - v["OperatingCashFlow"] - v["InvestingCF"]


def _set_series(co: Company, metric: str, series: dict[int, float]) -> None:
    """Pin ``metric`` to ``series`` (year -> $M) and rebalance the statements around it."""
    direct = {"Cash", "MarketableSecurities", "AccountsReceivableNet", "InventoryNet", "PropertyPlantEquipmentNet",
              "AccountsPayable", "LongTermDebt", "Revenue", "InterestExpense", "IncomeTaxExpenseBenefit",
              "DepreciationAmortization", "CapitalExpenditure", "DividendsPaid", "CostOfGoodsSold"}
    for y in YEARS:
        if y not in series:
            continue
        t = {}
        if metric in direct:
            co.vals[y][metric] = series[y]
        else:
            t[metric] = series[y]
        _recompute(co, y, y - 1 if y - 1 in co.vals else None, t)
        if y + 1 in co.vals:
            _recompute(co, y + 1, y)


def metric_value(co: Company, metric: str, y: int) -> float | None:
    """Metric value for year y as printed in that year's own filing (the gold uses reported values)."""
    if metric in FORMULAS:
        expr, prev, _, _ = FORMULAS[metric]
        envd = {}
        for c in sorted(set(re.findall(r"[A-Za-z]+", expr))):
            if c in LABELS:
                envd[c] = _printed_value(co, co.vals[y][c])
                envd[c + "_prev"] = _printed_value(co, co.vals[y - 1][c])
        try:
            return float(eval(expr, {}, envd))  # noqa: S307  fixed expressions from FORMULAS
        except ZeroDivisionError:
            return None
    return _printed_value(co, co.vals[y][metric])


_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]


def _fy_date(co: Company, y: int) -> tuple[str, str]:
    m, d = co.fye
    if m == 9 and d == 25:
        import datetime as _dt
        x = _dt.date(y, 9, 30)
        while x.weekday() != 5:
            x -= _dt.timedelta(days=1)
        m, d = 9, x.day
    if m == 10 and d == 1:
        d = 1 + (y % 3)
    return f"{_MONTHS[m - 1]} {d}, {y}", f"{y}{m:02d}{d:02d}"


def _fmt(co: Company, x: float, dollar: bool = False, paren_positive: bool = False) -> str:
    if co.unit == "thousands":
        n = round(x * 1000)
        s = f"{abs(n):,}"
    elif co.unit == "millions1":
        n = round(x, 1)
        s = f"{abs(n):,.1f}"
    else:
        n = round(x)
        s = f"{abs(n):,}"
    if n == 0:
        s = "—"
    neg = (n < 0) != paren_positive and n != 0
    if neg:
        s = f"({s})"
    if dollar and co.dollar and n != 0:
        s = "$ " + s
    return s


def _printed_value(co: Company, x: float) -> float:
    """What the table shows, back in $M (rounding of the table unit)."""
    if co.unit == "thousands":
        return round(x * 1000) / 1000
    if co.unit == "millions1":
        return round(x, 1)
    return float(round(x))


def _unit_line(co: Company, kind: str) -> str:
    if co.unit == "thousands":
        return "(In thousands, except per share amounts)" if kind == "IS" else "(In thousands)"
    if co.unit == "millions1":
        return "(Dollars in millions, except per-share data)"
    return "(in millions, except per share data)" if kind == "IS" else "(in millions)"


class _Doc:
    """Accumulates one filing's text and its fact lines (items + hidden facts)."""

    def __init__(self, fi: int, seen: set | None = None):
        self.fi = fi
        self.parts: list[str] = []
        self.items: list[tuple[str, str]] = []
        self.facts: dict[str, dict] = {}
        self.n = 0
        self.seen = seen if seen is not None else set()

    def text(self, s: str) -> None:
        self.parts.append(s)

    def fact(self, line: str, src: str, metric: str, years: list[int], vals: list[float], scale: str = "table",
             approx: bool = False) -> None:
        iid = f"F{self.fi}.{src}.{self.n}"
        self.n += 1
        key = " ".join(line.split())
        k = 1
        while key in self.seen:
            line = line.rstrip() + f" ({k}) \n"
            key = " ".join(line.split())
            k += 1
        self.seen.add(key)
        self.parts.append(line)
        self.items.append((iid, line.strip()))
        self.facts[iid] = {"f": self.fi, "src": src, "m": metric, "y": years, "v": vals, "scale": scale, "ax": approx,
                           "lab": line.strip()[:80]}


def _lab(co: Company, metric: str, y: int, rng: random.Random) -> str:
    lab = co.labels[metric]
    v = co.vals[y]
    return lab.format(CO=co.cover.title() if co.cover.isupper() else co.cover, Y=y, PY=y - 1,
                      a1=max(1, round(v["AccountsReceivableNet"] * 0.012)), a0=max(1, round(v["AccountsReceivableNet"] * 0.010)),
                      d1=_fmt(co, v["PropertyPlantEquipmentNet"] * 0.6), d0=_fmt(co, v["PropertyPlantEquipmentNet"] * 0.55))


def _row(co: Company, label: str, xs: list[float], dollar=False, paren_positive=False) -> str:
    return " " + label + " " + " ".join(_fmt(co, x, dollar, paren_positive) for x in xs) + " \n"


def _cols(co: Company, ys: list[int]) -> list[int]:
    return list(reversed(ys)) if co.order == "asc" else ys


def _statements(doc: _Doc, co: Company, fy: int, rng: random.Random) -> None:
    date, _ = _fy_date(co, fy)
    ys3, ys2 = _cols(co, [fy, fy - 1, fy - 2]), _cols(co, [fy, fy - 1])
    V = co.vals
    hdr3 = " ".join(str(y) for y in ys3)
    hdr2 = " ".join(str(y) for y in ys2)
    cov = co.cover.title() if co.cover.isupper() else co.cover
    title = rng.choice(["Consolidated Statements of Operations", "CONSOLIDATED STATEMENTS OF INCOME",
                        "Consolidated Statements of Earnings", "Consolidated Statements of Income"])
    doc.text(f"\n{cov}\n{title}\n{_unit_line(co, 'IS')}\n\n Years ended {_MONTHS[co.fye[0] - 1]} {co.fye[1] if co.fye != (9, 25) else 'last Saturday'}, {hdr3}\n")
    f = lambda m, src="IS", dollar=False, lab=None, pp=False: doc.fact(  # noqa: E731
        _row(co, lab or _lab(co, m, fy, rng), [V[y][m] for y in ys3], dollar, pp), src, m, ys3,
        [_printed_value(co, V[y][m]) for y in ys3])
    f("Revenue", dollar=True)
    f("CostOfGoodsSold")
    f("GrossProfit")
    doc.fact(_row(co, "Research and development", [V[y]["RD"] for y in ys3]), "IS", "", ys3, [])
    doc.fact(_row(co, "Selling, general and administrative", [V[y]["SGA"] for y in ys3]), "IS", "", ys3, [])
    f("OperatingExpenses")
    f("OperatingIncomeLoss")
    f("InterestExpense")
    doc.fact(_row(co, "Other income (expense), net", [V[y]["OtherInc"] for y in ys3]), "IS", "", ys3, [])
    doc.fact(_row(co, "Income before income taxes", [V[y]["Pretax"] for y in ys3]), "IS", "", ys3, [])
    f("IncomeTaxExpenseBenefit")
    if co.nci:
        doc.fact(_row(co, "Net income", [V[y]["NetIncomeTotal"] for y in ys3]), "IS", "NetIncome", ys3,
                 [_printed_value(co, V[y]["NetIncomeTotal"]) for y in ys3], scale="table-incl-nci")
        doc.fact(_row(co, "Less: Net income attributable to noncontrolling interests", [V[y]["NCIShare"] for y in ys3]),
                 "IS", "", ys3, [])
    f("NetIncome", dollar=True)
    eps = [V[y]["NetIncome"] / max(50.0, V[y]["Revenue"] / 40) for y in ys3]
    doc.text(" Earnings per share: \n Basic " + " ".join(f"$ {x:.2f}" for x in eps) + " \n Diluted "
             + " ".join(f"$ {x * 0.99:.2f}" for x in eps) + " \n")
    doc.text("\n See accompanying Notes to Consolidated Financial Statements.\n")
    doc.text("\n" + " ".join(PARAS[rng.randrange(len(PARAS))] for _ in range(2)) + "\n")
    title = rng.choice(["Consolidated Balance Sheets", "CONSOLIDATED BALANCE SHEETS",
                        "Consolidated Statements of Financial Position"])
    doc.text(f"\n{cov}\n{title}\n{_unit_line(co, 'BS')}\n\n  at {date[:-6] if co.fye == (12, 31) else ''} {hdr2}\n  Assets\n Current assets:\n")
    g = lambda m, dollar=False, lab=None: doc.fact(  # noqa: E731
        _row(co, lab or _lab(co, m, fy, rng), [V[y][m] for y in ys2], dollar), "BS", m, ys2,
        [_printed_value(co, V[y][m]) for y in ys2])
    g("Cash", dollar=True)
    if co.has_ms:
        g("MarketableSecurities")
    g("AccountsReceivableNet")
    if co.has_inv:
        g("InventoryNet")
    doc.fact(_row(co, "Prepaid expenses and other current assets", [V[y]["OtherCA"] for y in ys2]), "BS", "", ys2, [])
    g("CurrentAssets")
    g("PropertyPlantEquipmentNet")
    if co.has_ms:
        doc.fact(_row(co, rng.choice(["Long-term marketable securities", "Marketable securities, non-current",
                                      "Long-term investments"]), [V[y]["LTInvest"] for y in ys2]), "BS",
                 "MarketableSecurities", ys2, [_printed_value(co, V[y]["LTInvest"]) for y in ys2], scale="noncurrent")
    doc.fact(_row(co, "Goodwill", [V[y]["Goodwill"] for y in ys2]), "BS", "", ys2, [])
    doc.fact(_row(co, "Intangible assets, net", [V[y]["Intangibles"] for y in ys2]), "BS", "", ys2, [])
    doc.fact(_row(co, "Other assets", [V[y]["OtherNCA"] for y in ys2]), "BS", "", ys2, [])
    g("TotalAssets", dollar=True)
    doc.text("  Liabilities and Equity\n Current liabilities:\n")
    g("AccountsPayable", dollar=True)
    doc.fact(_row(co, "Accrued liabilities", [V[y]["Accrued"] for y in ys2]), "BS", "", ys2, [])
    doc.fact(_row(co, "Short-term borrowings and current portion of long-term debt", [V[y]["STDebt"] for y in ys2]),
             "BS", "", ys2, [])
    doc.fact(_row(co, "Other current liabilities", [V[y]["OtherCL"] for y in ys2]), "BS", "", ys2, [])
    g("CurrentLiabilities")
    g("LongTermDebt")
    doc.fact(_row(co, "Deferred income taxes", [V[y]["DeferredTax"] for y in ys2]), "BS", "", ys2, [])
    doc.fact(_row(co, "Other noncurrent liabilities", [V[y]["OtherNCL"] for y in ys2]), "BS", "", ys2, [])
    g("TotalLiabilities")
    doc.text(" Commitments and contingencies (Note 15) \n Equity: \n")
    doc.fact(_row(co, "Common stock and additional paid-in capital", [V[y]["ShareholdersEquity"] * 0.31 for y in ys2]),
             "BS", "", ys2, [])
    doc.fact(_row(co, "Retained earnings", [V[y]["ShareholdersEquity"] * 0.93 for y in ys2]), "BS", "", ys2, [])
    doc.fact(_row(co, "Accumulated other comprehensive loss", [-V[y]["ShareholdersEquity"] * 0.06 for y in ys2]), "BS",
             "", ys2, [])
    doc.fact(_row(co, "Treasury stock, at cost", [-V[y]["ShareholdersEquity"] * 0.18 for y in ys2]), "BS", "", ys2, [])
    g("ShareholdersEquity")
    if co.nci:
        doc.fact(_row(co, "Noncontrolling interests", [V[y]["NCIEquity"] for y in ys2]), "BS", "", ys2, [])
        doc.fact(_row(co, "Total equity", [V[y]["TotalEquity"] for y in ys2]), "BS", "ShareholdersEquity", ys2,
                 [_printed_value(co, V[y]["TotalEquity"]) for y in ys2], scale="incl-nci")
    doc.fact(_row(co, "Total liabilities and equity", [V[y]["TotalAssets"] for y in ys2], True), "BS", "", ys2, [])
    doc.text("\n See accompanying Notes to Consolidated Financial Statements.\n")
    doc.text("\n" + " ".join(PARAS[rng.randrange(len(PARAS))] for _ in range(2)) + "\n")
    doc.text(f"\n{cov}\nConsolidated Statements of Cash Flows\n{_unit_line(co, 'CF')}\n\n Years ended {hdr3}\n"
             " Cash flows from operating activities: \n")
    doc.fact(_row(co, "Net income" if not co.nci else "Net income (including noncontrolling interests)",
                  [V[y]["NetIncomeTotal"] for y in ys3], True), "CF", "NetIncome", ys3,
             [_printed_value(co, V[y]["NetIncomeTotal"]) for y in ys3], scale="cf-total")
    doc.text(" Adjustments to reconcile net income to net cash provided by operating activities: \n")
    f("DepreciationAmortization", src="CF")
    doc.fact(_row(co, "Stock-based compensation", [V[y]["SBC"] for y in ys3]), "CF", "", ys3, [])
    doc.text(" Changes in operating assets and liabilities: \n")
    doc.fact(_row(co, rng.choice(["Accounts receivable", "Receivables"]), [V[y]["dAR"] for y in ys3]), "CF",
             "AccountsReceivableNet", ys3, [_printed_value(co, V[y]["dAR"]) for y in ys3], scale="change")
    if co.has_inv:
        doc.fact(_row(co, "Inventories", [V[y]["dInv"] for y in ys3]), "CF", "InventoryNet", ys3,
                 [_printed_value(co, V[y]["dInv"]) for y in ys3], scale="change")
    doc.fact(_row(co, "Accounts payable", [V[y]["dAP"] for y in ys3]), "CF", "AccountsPayable", ys3,
             [_printed_value(co, V[y]["dAP"]) for y in ys3], scale="change")
    doc.fact(_row(co, "Other, net", [V[y]["OtherOp"] for y in ys3]), "CF", "", ys3, [])
    f("OperatingCashFlow", src="CF")
    doc.text(" Cash flows from investing activities: \n")
    doc.fact(_row(co, _lab(co, "CapitalExpenditure", fy, rng), [-V[y]["CapitalExpenditure"] for y in ys3]), "CF",
             "CapitalExpenditure", ys3, [_printed_value(co, V[y]["CapitalExpenditure"]) for y in ys3], scale="outflow")
    if co.has_ms:
        doc.fact(_row(co, "Purchases of marketable securities", [-V[y]["MarketableSecurities"] * 0.6 for y in ys3]), "CF",
                 "MarketableSecurities", ys3, [_printed_value(co, V[y]["MarketableSecurities"] * 0.6) for y in ys3],
                 scale="flow")
    doc.fact(_row(co, "Net cash used in investing activities", [V[y]["InvestingCF"] for y in ys3]), "CF", "", ys3, [])
    doc.text(" Cash flows from financing activities: \n")
    if co.payer:
        doc.fact(_row(co, _lab(co, "DividendsPaid", fy, rng), [-V[y]["DividendsPaid"] for y in ys3]), "CF",
                 "DividendsPaid", ys3, [_printed_value(co, V[y]["DividendsPaid"]) for y in ys3], scale="outflow")
    doc.fact(_row(co, "Repurchases of common stock", [-V[y]["Buybacks"] for y in ys3]), "CF", "", ys3, [])
    doc.fact(_row(co, "Net cash provided by (used in) financing activities", [V[y]["FinancingCF"] for y in ys3]), "CF",
             "", ys3, [])
    end_lab = ("Cash, cash equivalents and restricted cash at end of year" if co.restricted
               else "Cash and cash equivalents at end of year")
    doc.fact(_row(co, end_lab, [V[y]["Cash"] + V[y]["RestrictedCash"] for y in ys3], True), "CF", "Cash", ys3,
             [_printed_value(co, V[y]["Cash"] + V[y]["RestrictedCash"]) for y in ys3],
             scale="restricted" if co.restricted else "table")
    doc.text("\n See accompanying Notes to Consolidated Financial Statements.\n")


def _bn(x: float) -> str:
    return f"${x / 1000:.1f} billion" if abs(x) >= 1000 else f"${x:,.0f} million"


def _shown(x: float, exact: bool) -> float:
    """The $M value a reader recovers from the MD&A wording."""
    if exact or abs(x) < 1000:
        return float(round(x))
    return round(x / 1000, 1) * 1000


def _mdna(doc: _Doc, co: Company, fy: int, rng: random.Random) -> None:
    V = co.vals
    cur, pr = V[fy], V[fy - 1]
    word = rng.choice(["fiscal ", "", "fiscal year "])
    exact = rng.random() < 0.5
    money = (lambda x: f"${x:,.0f} million") if exact else _bn
    s = []
    ch = (cur["Revenue"] / pr["Revenue"] - 1) * 100
    doc.fact(f"{_lab(co, 'Revenue', fy, rng)} {'increased' if ch >= 0 else 'decreased'} {abs(ch):.0f}% to "
             f"{money(cur['Revenue'])} in {word}{fy}, compared with {money(pr['Revenue'])} in {word}{fy - 1}. ",
             "MDNA", "Revenue", [fy, fy - 1], [_shown(cur["Revenue"], exact), _shown(pr["Revenue"], exact)], scale="words",
             approx=not exact)
    adj = cur["OperatingIncomeLoss"] * rng.uniform(1.06, 1.25) + 0.02 * cur["Revenue"]
    doc.fact(f"Adjusted operating income, a non-GAAP measure that excludes restructuring and acquisition-related "
             f"charges, was {money(adj)} in {word}{fy}. ", "MDNA", "OperatingIncomeLoss", [fy], [_shown(adj, exact)], scale="non-gaap",
             approx=not exact)
    liq = cur["Cash"] + cur["MarketableSecurities"] + cur["LTInvest"] * (1 if co.has_ms else 0)
    doc.fact(f"As of the end of {word}{fy}, cash, cash equivalents and marketable securities totaled {money(liq)}. ",
             "MDNA", "Cash", [fy], [_shown(liq, exact)], scale="sum", approx=not exact)
    total_debt = cur["LongTermDebt"] + cur["STDebt"]
    doc.fact(f"Total debt outstanding was {money(total_debt)} at the end of {word}{fy}, of which "
             f"{money(cur['STDebt'])} was classified as current. ", "MDNA", "LongTermDebt", [fy], [_shown(total_debt, exact)],
             scale="total-debt", approx=not exact)
    doc.fact(f"Capital expenditures were {money(cur['CapitalExpenditure'])} in {word}{fy} and are expected to be "
             f"approximately {money(cur['CapitalExpenditure'] * rng.uniform(0.9, 1.3))} in {word}{fy + 1}. ", "MDNA",
             "CapitalExpenditure", [fy], [_shown(cur["CapitalExpenditure"], exact)], scale="words", approx=not exact)
    if co.payer:
        doc.fact(f"We returned {money(cur['DividendsPaid'] + cur['Buybacks'])} to shareholders in {word}{fy} through "
                 f"dividends and share repurchases. ", "MDNA", "DividendsPaid", [fy],
                 [_shown(cur["DividendsPaid"] + cur["Buybacks"], exact)], scale="sum", approx=not exact)
    doc.text(f"\n Results of Operations \n {fy} {fy - 1} Change \n")
    for m in ("Revenue", "GrossProfit", "OperatingExpenses", "OperatingIncomeLoss", "NetIncome"):
        a, b = cur[m], pr[m]
        chg = f"{(a / b - 1) * 100:.0f} %" if b else "n/m"
        doc.fact(" " + _lab(co, m, fy, rng) + " " + _fmt(co, a) + " " + _fmt(co, b) + " " + chg + " \n", "MDT", m,
                 [fy, fy - 1], [_printed_value(co, a), _printed_value(co, b)])
    for _ in s:
        pass


def _selected(doc: _Doc, co: Company, fy: int) -> None:
    V = co.vals
    ys = [fy - 4, fy - 3, fy - 2, fy - 1, fy]
    doc.text("\nItem 6. Selected Financial Data\n" + _unit_line(co, "IS") + "\n " + " ".join(map(str, ys)) + " \n")
    rows = [("Revenue", "Net sales", lambda y: V[y]["Revenue"], "table"),
            ("NetIncome", "Net income", lambda y: V[y]["NetIncomeTotal"], "table-incl-nci" if co.nci else "table"),
            ("TotalAssets", "Total assets", lambda y: V[y]["TotalAssets"], "table"),
            ("LongTermDebt", "Long-term debt, including current portion", lambda y: V[y]["LongTermDebt"] + V[y]["STDebt"],
             "total-debt"),
            ("CurrentAssets", "Working capital", lambda y: V[y]["CurrentAssets"] - V[y]["CurrentLiabilities"], "wc")]
    for m, lab, fn, sc in rows:
        doc.fact(_row(co, lab, [fn(y) for y in ys]), "SEL", m, ys, [_printed_value(co, fn(y)) for y in ys], scale=sc)


def _notes(doc: _Doc, co: Company, fy: int, rng: random.Random) -> None:
    V = co.vals
    ys2 = _cols(co, [fy, fy - 1])
    cur = V[fy]
    doc.text("\nNote 11. Debt\n" + _unit_line(co, "BS") + "\n " + " ".join(map(str, ys2)) + " \n")
    doc.fact(_row(co, "Total debt", [V[y]["LongTermDebt"] + V[y]["STDebt"] for y in ys2]), "DEBT", "LongTermDebt", ys2,
             [_printed_value(co, V[y]["LongTermDebt"] + V[y]["STDebt"]) for y in ys2], scale="total-debt")
    doc.fact(_row(co, "Less: current portion", [-V[y]["STDebt"] for y in ys2]), "DEBT", "", ys2, [])
    doc.fact(_row(co, "Long-term debt, net of current portion", [V[y]["LongTermDebt"] for y in ys2]), "DEBT",
             "LongTermDebt", ys2, [_printed_value(co, V[y]["LongTermDebt"]) for y in ys2])
    doc.text(PARAS[rng.randrange(len(PARAS))] + "\n")
    if co.has_ms:
        doc.text("\nNote 5. Marketable Securities\n " + " ".join(map(str, ys2)) + " \n")
        doc.fact(_row(co, "Total marketable securities", [V[y]["MarketableSecurities"] + V[y]["LTInvest"] for y in ys2]),
                 "INV", "MarketableSecurities", ys2,
                 [_printed_value(co, V[y]["MarketableSecurities"] + V[y]["LTInvest"]) for y in ys2], scale="sum")
        doc.fact(_row(co, "Classified as current", [V[y]["MarketableSecurities"] for y in ys2]), "INV",
                 "MarketableSecurities", ys2, [_printed_value(co, V[y]["MarketableSecurities"]) for y in ys2])
        doc.fact(_row(co, "Classified as non-current", [V[y]["LTInvest"] for y in ys2]), "INV", "MarketableSecurities",
                 ys2, [_printed_value(co, V[y]["LTInvest"]) for y in ys2], scale="noncurrent")
    shares = [rng.uniform(0.2, 1.0) for _ in co.segs]
    tot = sum(shares)
    shares = [s / tot for s in shares]
    doc.text("\nNote 18. Segment Information\n" + _unit_line(co, "BS") + "\n " + " ".join(map(str, ys2)) + " \n")
    for m, lab in (("Revenue", "net sales"), ("TotalAssets", "total assets"), ("DepreciationAmortization",
                                                                              "depreciation and amortization")):
        for s, sh in zip(co.segs, shares, strict=False):
            doc.fact(_row(co, f"{s} {lab}", [V[y][m] * sh for y in ys2]), "SEG", m, ys2,
                     [_printed_value(co, V[y][m] * sh) for y in ys2], scale="segment")
    q = [rng.uniform(0.2, 0.3) for _ in range(4)]
    qt = sum(q)
    doc.text(f"\nQuarterly Financial Data (Unaudited) -- fiscal {fy}\n First Quarter Second Quarter Third Quarter Fourth Quarter \n")
    for m, lab in (("Revenue", _lab(co, "Revenue", fy, rng)), ("NetIncome", "Net income")):
        doc.fact(_row(co, lab, [cur[m] * x / qt for x in q]), "QTR", m, [fy] * 4,
                 [_printed_value(co, cur[m] * x / qt) for x in q], scale="quarter")
    _ = cur


def _cover(co: Company, fy: int) -> str:
    date, _ = _fy_date(co, fy)
    return (f"﻿ UNITED STATES \n SECURITIES AND EXCHANGE COMMISSION \n Washington, D.C. 20549 \n FORM 10-K \n"
            f"☒ ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934 \n"
            f" For the fiscal year ended {date} \n OR \n☐ TRANSITION REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE "
            f"SECURITIES EXCHANGE ACT OF 1934 \n Commission file number 001-{stable_hash(co.tkr) % 90000 + 10000} \n"
            f" {co.cover} \n (Exact name of registrant as specified in its charter) \n")


_TOC = ("\nTABLE OF CONTENTS\n Part I \n Item 1. Business \n Item 1A. Risk Factors \n Item 2. Properties \n"
        " Item 3. Legal Proceedings \n Part II \n Item 5. Market for Registrant's Common Equity \n Item 6. [Reserved] \n"
        " Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations \n"
        " Item 7A. Quantitative and Qualitative Disclosures About Market Risk \n Item 8. Financial Statements and "
        "Supplementary Data \n Consolidated Statements of Operations \n Consolidated Balance Sheets \n Consolidated "
        "Statements of Cash Flows \n Notes to Consolidated Financial Statements \n Part III \n Part IV \n")


def _filler(rng: random.Random, chars: int, co: Company, fy: int) -> str:
    out, n = [], 0
    while n < chars:
        p = PARAS[rng.randrange(len(PARAS))]
        out.append(p)
        n += len(p) + 2
    return "\n\n".join(out)


def render_filing(fi: int, co: Company, fy: int, rng: random.Random, size: int,
                  seen: set | None = None) -> tuple[str, str, _Doc]:
    date, ymd = _fy_date(co, fy)
    name = [f"{co.tkr}-{ymd}.txt", f"{co.tkr}-10k_{ymd}.txt", f"{co.tkr}-{ymd}x10k.txt"][co.file_style]
    doc = _Doc(fi, seen)
    sz = max(20_000, size - 30_000)
    doc.text(_cover(co, fy) + _TOC)
    doc.text("\nPART I\nItem 1. Business\n" + _filler(rng, int(sz * 0.22), co, fy))
    doc.text("\nItem 1A. Risk Factors\n" + _filler(rng, int(sz * 0.22), co, fy))
    doc.text("\nItem 2. Properties\n" + _filler(rng, int(sz * 0.03), co, fy))
    if co.selected and fy <= 2021:
        _selected(doc, co, fy)
    doc.text("\nItem 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
             + _filler(rng, int(sz * 0.08), co, fy) + "\n")
    _mdna(doc, co, fy, rng)
    doc.text("\n" + _filler(rng, int(sz * 0.08), co, fy) + "\nItem 8. Financial Statements and Supplementary Data\n"
             "Report of Independent Registered Public Accounting Firm\n We have audited the accompanying consolidated "
             "balance sheets and the related consolidated statements of operations and cash flows. \n")
    _statements(doc, co, fy, rng)
    doc.text("\nNotes to Consolidated Financial Statements\n" + _filler(rng, int(sz * 0.18), co, fy))
    _notes(doc, co, fy, rng)
    doc.text("\n" + _filler(rng, int(sz * 0.12), co, fy) + "\nSIGNATURES\n")
    text = "".join(doc.parts)
    cov = co.cover.title() if co.cover.isupper() else co.cover
    text = text.replace("{CO}", cov).replace("{Y}", str(fy)).replace("{PY}", str(fy - 1)).replace("{DATE}", date)
    return name, text, doc


def _companies_text(cos: list[Company], numbered: bool) -> str:
    if numbered:
        return ", ".join(f"{i + 1}. {c.qname}" for i, c in enumerate(cos))
    return ", ".join(c.qname for c in cos)


def _fmt_threshold(metric: str, thr: float) -> str:
    if metric in FORMULAS and FORMULAS[metric][2]:
        return f"{thr:.2f}%"
    if metric in FORMULAS and metric not in ("WorkingCapital", "FreeCashFlow"):
        return f"{thr:.2f}"
    return f"${thr:.2f}M"


def _well_separated(xs: list[float], rel: float) -> bool:
    s = sorted(xs)
    return all(abs(b - a) > rel * max(abs(a), abs(b), 1e-9) for a, b in zip(s, s[1:], strict=False))


@dataclass
class FinParams:
    filing_kchars: tuple[int, int] = (200, 500)
    shuffle_files_p: float = 0.2
    trend_disagree: float = 0.7
    trend_prior_trap: float = 0.5
    formula_p: float = FORMULA_P
    kinds: dict = field(default_factory=lambda: dict(KIND_W))


def _pick_metric(rng: random.Random, kind: str, p: FinParams, co_list: list[Company]) -> str:
    if kind in ("hilo", "rank", "maxmin", "count") and rng.random() < p.formula_p:
        return rng.choice(sorted(FORMULAS))
    names = [m for m in BASE_W if m not in ("InventoryNet",) or kind != "stable"]
    return rng.choices(names, weights=[BASE_W[m] for m in names])[0]


def _metric_ok(co: Company, m: str) -> bool:
    need = set(re.findall(r"[A-Za-z]+", FORMULAS[m][0])) & set(LABELS) if m in FORMULAS else {m}
    if "MarketableSecurities" in need and not co.has_ms:
        return False
    if "InventoryNet" in need and not co.has_inv:
        return False
    if "DividendsPaid" in need and not co.payer:
        return False
    return True


def generate_finance(info: dict, p: FinParams | None = None) -> Generated:
    p = p or FinParams()
    seed = f"fin:{info.get('synth_split')}:{info.get('synth_seed')}"
    rng = random.Random(seed)
    kind = info.get("_kind") or rng.choices(list(p.kinds), weights=list(p.kinds.values()))[0]
    for _attempt in range(60):
        used: set = set()
        n_co = {"trend": 1, "stable": 3}.get(kind) or rng.choices([3, 4, 5], weights=[2, 3, 5])[0]
        cos = [_make_company(rng, used) for _ in range(n_co)]
        for c in cos:
            _gen_values(c, rng)
        metric = _pick_metric(rng, kind, p, cos)
        if not all(_metric_ok(c, metric) for c in cos):
            continue
        q = _make_question(rng, kind, metric, cos, p)
        if q is not None:
            break
    else:  # pragma: no cover
        raise RuntimeError("finance generator failed to find a well-posed question")
    question, gold, filings = q
    if rng.random() < p.shuffle_files_p:
        rng.shuffle(filings)
    parts, items, facts, fil_meta = [], [], {}, []
    seen: set = set()
    for fi, (co, fy) in enumerate(filings):
        size = rng.randint(*p.filing_kchars) * 1000
        name, text, doc = render_filing(fi, co, fy, rng, size, seen)
        parts.append(f"\n<FILE: {name}>\n{text}")
        items += doc.items
        facts.update(doc.facts)
        fil_meta.append({"file": name, "co": co.qname, "cover": co.cover, "fy": fy, "unit": co.unit})
    context = "".join(parts)
    root_prompt = INSTRUCTION.replace("{context}", CONTEXT_POINTER).replace("{question}", question)
    required = [iid for iid, f in facts.items() if f["src"] in ("BS", "IS", "CF") and f["m"]
                and f["scale"] == "table" and f["m"] in _needed(metric)]
    hidden = {"family": "finance", "kind": kind, "metric": metric, "filings": fil_meta, "facts": facts,
              "answer_type": "QA", "options": [], "question": question}
    return Generated(context=context, root_prompt=root_prompt, items=items, gold=gold, hidden=hidden,
                     required_ids=required, index_min_len=12)


def _needed(metric: str) -> set[str]:
    if metric in FORMULAS:
        return set(re.findall(r"[A-Za-z]+", FORMULAS[metric][0])) & set(LABELS)
    return {metric}


def _qmetric(m: str) -> str:
    return QNAME.get(m, m)


def _make_question(rng: random.Random, kind: str, metric: str, cos: list[Company], p: FinParams):
    """Returns (question, gold, [(company, fy) filings]) or None when the draw is not well separated."""
    fmt_int = 'Please format your response as: "The correct answer is X" (plain integer number  e.g. 2).'
    pct = metric in FORMULAS and FORMULAS[metric][2]
    mname = _qmetric(metric) + (" (%)" if pct and kind in ("hilo", "rank", "count") and rng.random() < 0.5 else "")
    formula = ""
    if metric in FORMULAS:
        formula = " Formula: " + rng.choice(FORMULAS[metric][3])
    if kind == "trend":
        co = cos[0]
        a = rng.randint(2019, 2021)
        b = min(2024, a + rng.choice([3, 3, 4]))
        ys = list(range(a, b + 1))
        want_disagree = rng.random() < p.trend_disagree
        lvl = abs(metric_value(co, metric, a) or 0) or 100.0
        for _ in range(80):
            series, x = {}, lvl
            for y in ys:
                series[y] = x
                x = x * (1 + rng.choice([-1, 1]) * rng.uniform(0.02, 0.45))
            pairs = list(zip(ys, ys[1:], strict=False))
            rel = {pp: abs(series[pp[1]] / series[pp[0]] - 1) for pp in pairs}
            dol = {pp: abs(series[pp[1]] - series[pp[0]]) for pp in pairs}
            r_best = max(rel, key=rel.get)
            d_best = max(dol, key=dol.get)
            rs, ds = sorted(rel.values()), sorted(dol.values())
            if rs[-1] < 1.15 * rs[-2] or ds[-1] < 1.08 * ds[-2]:
                continue
            if (r_best != d_best) == want_disagree:
                break
        else:
            return None
        if rng.random() < p.trend_prior_trap:
            top = max(abs(series[y2] / series[y1] - 1) for y1, y2 in zip(ys, ys[1:], strict=False))
            jump = min(0.95, top * rng.uniform(1.4, 2.2))
            series[a - 1] = series[a] / (1 + jump) if rng.random() < 0.5 else series[a] / max(0.05, 1 - jump)
        _set_series(co, metric, series)
        vals = {y: metric_value(co, metric, y) for y in ys}
        rel = {pp: abs(vals[pp[1]] / vals[pp[0]] - 1) for pp in zip(ys, ys[1:], strict=False)}
        best = max(rel, key=rel.get)
        question = (f"Between FY{a} and FY{b}, in which year did {co.qname}'s {_qmetric(metric)} have the largest absolute "
                    f"year-over-year change? Please format your response as: \"The correct answer is XXXX-XXXX\"(year pair "
                    f"e.g. 2000-2001)")
        return question, f"{best[0]}-{best[1]}", [(co, y) for y in ys]
    if kind == "stable":
        a = rng.randint(2019, 2023)
        b = min(2024, a + rng.choice([1, 2, 2]))
        ys = list(range(a, b + 1))
        n = len(cos)
        target = rng.randrange(n)
        for _ in range(200):
            levels = [math.exp(rng.uniform(math.log(200), math.log(60_000))) for _ in range(n)]
            vols = [rng.uniform(0.01, 0.35) for _ in range(n)]
            series = [{y: lv * (1 + rng.gauss(0, vo)) for y in ys} for lv, vo in zip(levels, vols, strict=False)]
            if any(x <= 0 for s in series for x in s.values()):
                continue
            sds = [statistics.pstdev(list(s.values())) for s in series]
            if min(range(n), key=lambda i: sds[i]) != target:
                continue
            others = sorted(sds[i] for i in range(n) if i != target)
            if sds[target] > 0.8 * others[0]:
                continue
            smallest_level = min(range(n), key=lambda i: levels[i])
            if smallest_level == target and rng.random() < 0.6:
                continue
            break
        else:
            return None
        for c, s in zip(cos, series, strict=False):
            _set_series(c, metric, s)
        sd = [statistics.pstdev([metric_value(c, metric, y) for y in ys]) for c in cos]
        gold = str(min(range(n), key=lambda i: sd[i]) + 1)
        question = (f"Between {a} and {b}, among the following companies: {_companies_text(cos, True)}, which company "
                    f"had the most stable (i.e., least volatile) {_qmetric(metric)} across years (based on standard "
                    f"deviation)? Please answer using the number in front of the company name. {fmt_int}\"")
        return question, gold, [(c, y) for c in cos for y in ys]
    fy = rng.randint(2019, 2024)
    vals = [metric_value(c, metric, fy) for c in cos]
    if any(v is None for v in vals) or not _well_separated(vals, 0.03):
        return None
    filings = [(c, fy) for c in cos]
    if kind == "hilo":
        hi = rng.random() < 0.6
        pick = max(range(len(cos)), key=lambda i: vals[i]) if hi else min(range(len(cos)), key=lambda i: vals[i])
        question = (f"Among the following companies: {_companies_text(cos, True)}, which company had the "
                    f"{'highest' if hi else 'lowest'} {mname} in FY{fy}? Please answer using the number in front of the "
                    f"company name. {fmt_int}\"{formula}")
        return question, str(pick + 1), filings
    if kind == "rank":
        t = rng.randrange(len(cos))
        order = sorted(range(len(cos)), key=lambda i: -vals[i])
        question = (f"Among the following companies: {_companies_text(cos, False)}, rank them by {mname} in FY{fy} from "
                    f"highest to lowest. What is the rank of {cos[t].qname}? {fmt_int}{formula}")
        return question, str(order.index(t) + 1), filings
    if kind == "maxmin":
        if metric in FORMULAS and metric not in ("WorkingCapital", "FreeCashFlow"):
            return None
        diff = max(vals) - min(vals)
        question = (f"For {_qmetric(metric)} in fiscal year {fy}, what is the difference between the maximum and minimum "
                    f"values among the following companies: {_companies_text(cos, False)}? Please format your response "
                    f"as: \"The correct answer is X.XXM\" (USD Millions, rounded to 2 decimal places e.g. 1234.56M).{formula}")
        return question, f"${diff:.2f}M", filings
    if kind == "count":
        s = sorted(vals)
        k = rng.randint(0, len(s) - 2)
        lo, hi = s[k], s[k + 1]
        thr = lo + (hi - lo) * rng.uniform(0.3, 0.7)
        thr = float(f"{thr:.2f}")
        if abs(thr) < 1e-9 or min(abs(v - thr) for v in vals) < 0.03 * abs(thr):
            return None
        tt = _fmt_threshold(metric, thr)
        if rng.random() < 0.5:
            question = (f"From the list: {_companies_text(cos, False)} , how many companies have the metric {mname} "
                        f"greater than {tt} in FY{fy}? {fmt_int}{formula}")
        else:
            question = (f"Among the following companies: {_companies_text(cos, False)}, how many have the metric: {mname} "
                        f"greater than {tt} in FY{fy}? {fmt_int}{formula}")
        return question, str(sum(1 for v in vals if v > thr)), filings
    return None


_METRIC_TERMS: dict[str, list[str]] = {}
for _m, (_st, _labs) in LABELS.items():
    terms = {_m.lower(), re.sub(r"(?<!^)(?=[A-Z])", " ", _m).lower()}
    for _l in _labs:
        t = re.sub(r"\s*\(.*?\)|,? net of .*|,? less .*| \{.*", "", _l).lower().strip()
        if len(t) > 4 and "{" not in t:
            terms.add(t)
    _METRIC_TERMS[_m] = sorted(terms, key=len, reverse=True)
_METRIC_TERMS["Cash"] += ["cash and cash equivalents", "cashandcashequivalents"]
_METRIC_TERMS["MarketableSecurities"] += ["short-term investments", "marketable securities"]
_METRIC_TERMS["CapitalExpenditure"] += ["capex", "capital expenditure", "purchases of property"]
_METRIC_TERMS["DepreciationAmortization"] += ["d&a", "depreciation"]
_METRIC_TERMS["OperatingCashFlow"] += ["operating cash flow", "cash from operations", "operating activities"]
_METRIC_TERMS["ShareholdersEquity"] += ["stockholders' equity", "shareholders' equity", "stockholders’ equity", "total equity"]
_METRIC_TERMS["NetIncome"] += ["net earnings", "net income"]
_METRIC_TERMS["IncomeTaxExpenseBenefit"] += ["income tax", "tax expense", "provision for income taxes"]
_METRIC_TERMS["InventoryNet"] += ["inventory", "inventories"]
_METRIC_TERMS["AccountsReceivableNet"] += ["receivables", "accounts receivable"]
_METRIC_TERMS["LongTermDebt"] += ["long-term debt", "long term debt"]
_METRIC_TERMS["OperatingIncomeLoss"] += ["operating income", "income from operations"]
_METRIC_TERMS["Revenue"] += ["revenue", "net sales", "sales"]
_METRIC_TERMS["CostOfGoodsSold"] += ["cost of sales", "cost of revenue", "cogs"]

_GOOD = {"table"}
_PRIORITY = {"BS": 0, "IS": 0, "CF": 1, "DEBT": 2, "INV": 2, "MDT": 3, "MDNA": 4, "SEL": 5, "SEG": 6, "QTR": 7}
_AGG = re.compile(r"\b(highest|lowest|rank|ranking|largest|smallest|maximum|minimum|difference between|standard "
                  r"deviation|most stable|least volatile|volatil|how many|greater than|year-over-year|which company|"
                  r"which year|compare)\b", re.I)


def _detect_metrics(instr: str) -> list[tuple[str, str]]:
    low = instr.lower()
    found: list[tuple[int, str, str]] = []
    taken: list[tuple[int, int]] = []
    for m in list(FORMULAS):
        i = low.find(m.lower())
        if i >= 0:
            found.append((i, m, m))
            taken.append((i, i + len(m)))
    cands = [(t, m) for m, ts in _METRIC_TERMS.items() for t in ts]
    cands.sort(key=lambda x: -len(x[0]))
    for t, m in cands:
        for mt in re.finditer(r"(?<![a-z])" + re.escape(t) + r"(?![a-z])", low):
            s, e = mt.start(), mt.end()
            if any(not (e <= a or s >= b) for a, b in taken):
                continue
            taken.append((s, e))
            found.append((s, m, instr[s:e]))
    out, seen = [], set()
    for _, m, surf in sorted(found):
        if m not in seen:
            seen.add(m)
            out.append((m, surf))
    return out


class FinanceOracle:
    """Finance answers for ``LoogleOracle``."""

    def __init__(self, read: ReadModel, reliable_filings: int = 4, load_err: float = 0.5):
        self.read = read
        self.reliable_filings = reliable_filings
        self.load_err = load_err

    def respond(self, base: FakeSubLM, prompt: str, hidden: dict, st: dict) -> str:
        idx = hidden["_index"]
        ids = idx.present(prompt, base.read.min_cov)
        instr = base.instruction(prompt, hidden)
        instr = "\n".join(ln for ln in instr.split("\n") if len(re.findall(r"\d{1,3}(?:,\d{3})+", ln)) < 2)
        facts = hidden["facts"]
        fil = hidden["filings"]
        by_f: dict[int, list[str]] = {}
        for iid in ids:
            by_f.setdefault(facts[iid]["f"], []).append(iid)
        if not by_f:
            return refuse_aggregate(st)
        order = list(by_f)
        kept = []
        n = len(order)
        for pos, fi in enumerate(order):
            if n > self.reliable_filings and pos >= 2:
                pskip = min(0.9, 0.5 * (n - self.reliable_filings) / n * (0.5 + pos / n) * 2)
                if rng_for(prompt, fi, "skipf").random() < pskip:
                    st["items_skipped"] += 1
                    continue
            kept.append(fi)
        reads = st.setdefault("read_ids", {})
        for fi in kept:
            for iid in by_f[fi]:
                reads[iid] = reads.get(iid, 0) + 1
        st["items_read"] = len(reads)
        metrics = _detect_metrics(instr)
        years = sorted({int(y) for y in re.findall(r"(?<!\d)(20[12]\d)(?!\d)", instr)})
        load = base.load(prompt)
        p_err = base.read.base_err + self.load_err * load ** 2
        if _AGG.search(instr) and (len(kept) > 1 or not metrics):
            st["agg_requests"] += 1
            return self._aggregate(base, prompt, instr, hidden, kept, by_f, metrics, years, p_err)
        if not metrics:
            lines = []
            for fi in kept:
                srcs = sorted({facts[i]["src"] for i in by_f[fi]})
                lines.append(f"This text is from the Form 10-K of {fil[fi]['cover']} for fiscal year {fil[fi]['fy']} "
                             f"(file {fil[fi]['file']}); it contains {', '.join(_SRC_NAMES.get(s, s) for s in srcs)}. "
                             "Tell me which line items you need.")
            return base.chat(prompt, "\n".join(lines))
        to_millions = bool(re.search(r"\bmillions?\b", instr, re.I)) and bool(
            re.search(r"\b(?:in|to|as|convert\w*)\s+(?:usd\s+|us\s*\$\s*|\$\s*)?millions?\b|usd millions|\$m\b|millions of", instr, re.I))
        records = []
        for fi in kept:
            fy = fil[fi]["fy"]
            want_years = [y for y in years if fy - 2 <= y <= fy] or [fy]
            for m, surf in metrics:
                for y in want_years:
                    val = self._value(prompt, facts, by_f[fi], m, y, p_err, instr, fil)
                    rec = {"company": fil[fi]["cover"], "fiscal_year": fy, "metric": surf, "year": y}
                    if val is None:
                        rec.update(value=None, note="not found in the provided text")
                    else:
                        x, unit, label = val
                        if to_millions and unit != "millions":
                            conv = x * {"thousands": 1e-3, "billions": 1e3}.get(unit, 1.0)
                            if rng_for(prompt, fi, m, y, "noconv").random() < p_err / 2:
                                conv = x
                            rec.update(value=round(conv, 3), unit="millions", label=label)
                        else:
                            rec.update(value=x, unit=unit, label=label)
                    records.append(rec)
        return self._render(base, prompt, instr, records)

    def _value(self, prompt: str, facts: dict, ids: list[str], m: str, y: int, p_err: float, instr: str, fil: list):
        """(number as a float in the line's unit, unit word, line label) for metric m / year y, or None."""
        if m in FORMULAS:
            expr, prev, _, _ = FORMULAS[m]
            envd = {}
            unit = None
            for c in sorted(set(re.findall(r"[A-Za-z]+", expr)) & set(LABELS)):
                for yy, key in ((y, c), (y - 1, c + "_prev")):
                    if key.endswith("_prev") and not prev:
                        continue
                    r = self._value(prompt, facts, ids, c, yy, p_err, instr, fil)
                    if r is None:
                        return None
                    envd[key] = r[0] * {"thousands": 1e-3, "billions": 1e3}.get(r[1], 1.0)
                    unit = "millions"
            try:
                return round(float(eval(expr, {}, envd)), 4), unit or "millions", m  # noqa: S307
            except ZeroDivisionError:
                return None
        cands = []
        for iid in ids:
            f = facts[iid]
            if f["m"] != m or y not in f["y"]:
                continue
            k = f["y"].index(y)
            if k >= len(f["v"]):
                continue
            cands.append((self._prio(f, instr, m), iid, k))
        if not cands:
            return None
        cands.sort()
        _, iid, k = cands[0]
        f = facts[iid]
        r = rng_for(prompt, iid, y, "valerr")
        if r.random() < p_err:
            alts = [(i2, k2) for _, i2, k2 in cands[1:]]
            if len(f["y"]) > 1:
                alts.append((iid, (k + 1) % len(f["y"])))
            if alts:
                iid, k = alts[r.randrange(len(alts))]
                f = facts[iid]
        x = f["v"][k]
        unit = self._unit(f, fil)
        if f["scale"] == "outflow" and not re.search(r"positive|absolute", instr, re.I):
            x = -x
        if unit == "thousands":
            x = round(x * 1000)
        return x, unit, f["lab"]

    @staticmethod
    def _prio(f: dict, instr: str, m: str) -> tuple:
        p = _PRIORITY.get(f["src"], 9)
        sc = f["scale"]
        if sc in ("table", "outflow"):
            pass
        elif sc == "table-incl-nci":
            p += 0.5 if re.search(r"attributable", instr, re.I) else -0.5
        elif sc == "incl-nci":
            p += 0.5 if re.search(r"attributable|stockholders|shareholders", instr, re.I) else -0.2
        else:
            p += 0.3
        if m == "MarketableSecurities" and sc == "noncurrent" and not re.search(r"current", instr, re.I):
            p -= 0.1
        return (p,)

    @staticmethod
    def _unit(f: dict, fil: list) -> str:
        if f["src"] == "MDNA":
            return "millions"
        return "thousands" if fil[f["f"]]["unit"] == "thousands" else "millions"

    def _aggregate(self, base, prompt, instr, hidden, kept, by_f, metrics, years, p_err) -> str:
        fil = hidden["filings"]
        r = rng_for(prompt, "agg")
        names = [fil[fi]["cover"] for fi in kept]
        if not names:
            return refuse_aggregate({"refusals": 0})
        if len(kept) <= 2 and metrics:
            vals = []
            for fi in kept:
                v = self._value(prompt, hidden["facts"], by_f[fi], metrics[0][0], (years or [fil[fi]["fy"]])[-1], p_err,
                                instr, fil)
                vals.append(v[0] if v else None)
            if all(v is not None for v in vals):
                best = max(range(len(vals)), key=lambda i: vals[i])
                return base.chat(prompt, f"{names[best]} has the higher {metrics[0][1]} ({vals[best]:,}).")
        pick = r.choice(names)
        what = metrics[0][1] if metrics else "the requested measure"
        verdicts = [f"Based on the figures provided, {pick} stands out on {what}.",
                    f"The answer is {pick}; its {what} changed the most in absolute (dollar) terms.",
                    f"Comparing the companies, {pick} ranks first on {what}."]
        return base.chat(prompt, r.choice(verdicts))

    def _render(self, base: FakeSubLM, prompt: str, instr: str, records: list[dict]) -> str:
        if wants_json(instr):
            if re.search(r'"items"', instr):
                body = json.dumps({"items": records}, ensure_ascii=False)
            elif len(records) == 1:
                body = json.dumps(records[0], ensure_ascii=False)
            else:
                body = json.dumps(records, ensure_ascii=False)
            if rng_for(prompt, "fence").random() < base.read.chatty:
                body = "```json\n" + body + "\n```"
            return base.chat(prompt, body)
        lines = []
        for rec in records:
            if rec.get("value") is None:
                lines.append(f"{rec['company']} (FY{rec['fiscal_year']}) - {rec['metric']} for {rec['year']}: not found in the provided text")
            else:
                v = rec["value"]
                vs = f"{v:,}" if isinstance(v, (int, float)) else str(v)
                lines.append(f"{rec['company']} (FY{rec['fiscal_year']}) - {rec['metric']} for {rec['year']}: {vs} "
                             f"({rec['unit']}; line: {rec['label']})")
        return base.chat(prompt, "\n".join(lines))


_SRC_NAMES = {"BS": "a balance sheet", "IS": "an income statement", "CF": "a cash-flow statement", "DEBT": "a debt note",
              "INV": "an investments note", "MDT": "an MD&A results table", "MDNA": "MD&A narrative",
              "SEL": "a selected financial data table", "SEG": "a segment table", "QTR": "quarterly data"}

__all__ = ["FinParams", "FinanceOracle", "FORMULAS", "LABELS", "generate_finance", "metric_value"]
