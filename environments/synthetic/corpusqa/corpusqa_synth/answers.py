"""Recompute a question's answer from its spec under a chosen reading of the corpus."""

from __future__ import annotations

import statistics
from typing import Any

from .entity import CO_FIELDS
from .periodic import DOMAINS, columns


def series_by_rule(hidden: dict, mkey: str, adj: str, col: int, rule: str = "latest", scale: float | None = None) -> dict:
    """{YYYY-MM: value} within the release range under a revision rule.

    latest (the gold rule): newest print wins. earliest: oldest wins. own: each month from its own release.
    stale: newest wins but revised "(r)" rows are skipped, so the annual revision is lost."""
    reps = hidden["reports"]
    scale = hidden["scale"] if scale is None else scale
    lo, hi = reps[0]["report"], reps[-1]["report"]
    out: dict[str, float] = {}
    if rule == "stale":
        for rep in reps:
            skip = {str(y) for y in rep.get("revised") or []}
            for t in rep["tables"].values():
                if t["m"] == mkey and t["adj"] == adj:
                    dev = t.get("layout") == "deviant"
                    for ym, row in t["rows"].items():
                        if lo <= ym <= hi and row[col] is not None and (dev or ym[:4] not in skip):
                            out[ym] = row[col] * scale
        return dict(sorted(out.items()))
    if rule == "own":
        for rep in reps:
            for t in rep["tables"].values():
                if t["m"] == mkey and t["adj"] == adj:
                    row = t["rows"].get(rep["report"])
                    if row is not None and row[col] is not None:
                        out[rep["report"]] = row[col] * scale
        return dict(sorted(out.items()))
    order = reps if rule == "latest" else list(reversed(reps))
    for rep in order:
        for t in rep["tables"].values():
            if t["m"] == mkey and t["adj"] == adj:
                for ym, row in t["rows"].items():
                    if lo <= ym <= hi and row[col] is not None:
                        out[ym] = row[col] * scale
    return dict(sorted(out.items()))


def _group_label(dom_key: str, ci: int) -> str:
    g, s = columns(DOMAINS[dom_key])[ci]
    return g if dom_key == "housing" else s


def answer_periodic(spec: dict, S: list[dict], dom_key: str) -> Any:
    t = spec["tmpl"]
    s1 = S[0]
    if not s1:
        return None
    months = list(s1)
    if t == "agg":
        f = {"sum": sum, "mean": statistics.mean, "median": statistics.median, "max": max, "min": min}[spec["op"]]
        return round(f(s1.values()), 2)
    if t == "arg":
        return [(max if spec["hi"] else min)(s1, key=s1.get)]
    if t == "count_thr":
        thr = spec["thr"]
        return float(sum(1 for v in s1.values() if (v > thr if spec["above"] else v < thr)))
    if t == "mom_abs_mean":
        return round(statistics.mean(abs(s1[b] - s1[a]) for a, b in zip(months, months[1:], strict=False)), 2)
    if t == "count_up":
        return float(sum(1 for a, b in zip(months, months[1:], strict=False) if s1[b] > s1[a]))
    if t == "group_max":
        avgs = {spec["series"][j][2]: statistics.mean(s.values()) for j, s in enumerate(S) if s}
        best = (min if spec["lo"] else max)(avgs, key=avgs.get)
        return (spec.get("labels") or {}).get(str(best)) or _group_label(dom_key, best)
    v = [s1[k] for k in months]
    if t == "streak":
        up = spec.get("dir", "up") == "up"
        best = cur = 0
        for a, b in zip(v, v[1:], strict=False):
            cur = cur + 1 if (b > a if up else b < a) else 0
            best = max(best, cur)
        return float(best + 1 if spec.get("count") == "months" else best)
    if t == "prev2_avg":
        k = spec.get("k", 2)
        if k == 2:
            return [months[i] for i in range(2, len(v)) if v[i] > (v[i - 1] + v[i - 2]) / 2]
        return [months[i] for i in range(k, len(v)) if v[i] > sum(v[i - k:i]) / k]
    if t == "peaks":
        w, pk = spec.get("w", 1), spec.get("kind", "peak") == "peak"
        return float(sum(1 for i in range(w, len(v) - w)
                         if all((v[i] > v[j]) if pk else (v[i] < v[j]) for j in range(i - w, i + w + 1) if j != i)))
    if t == "cross_ma":
        k, incl, above = spec.get("k", 3), spec.get("incl", True), spec.get("dir", "above") == "above"
        ma: list[float | None] = [None] * len(v)
        for i in range(len(v)):
            win = v[i - k + 1:i + 1] if incl else v[i - k:i]
            if (i - k + 1 if incl else i - k) >= 0 and len(win) == k:
                ma[i] = sum(win) / k
        out = []
        for i in range(1, len(v)):
            if ma[i] is None or ma[i - 1] is None:
                continue
            if (v[i - 1] <= ma[i - 1] and v[i] > ma[i]) if above else (v[i - 1] >= ma[i - 1] and v[i] < ma[i]):
                out.append(months[i])
        return out
    s2 = S[1]
    if t == "filtered_mean":
        sel = [k for k in months if k in s2 and (s2[k] > spec["thr"] if spec["above"] else s2[k] < spec["thr"])]
        f = statistics.median if spec.get("stat") == "median" else statistics.mean
        return round(f(s1[k] for k in sel), 2) if sel else None
    if t == "topq_ratio":
        s3 = S[2]
        n = len(months)
        if "count" in spec:
            pct = spec["pct"]
            kk = {"floor": n * pct // 100, "round": (n * pct + 50) // 100, "ceil": -(-n * pct // 100)}[spec["count"]]
        else:
            kk = round(n * spec["q"])
        if kk < 1:
            return None
        top = sorted(months, key=lambda k: -s1[k])[:kk]
        keep = [k for k in top if k in s2 and k in s3 and s3[k]]
        if not keep:
            return None
        if spec.get("agg") == "ratio_of_totals":
            den = sum(s3[k] for k in keep)
            return round(sum(s2[k] for k in keep) / den, 2) if den else None
        return round(statistics.mean(s2[k] / s3[k] for k in keep), 2)
    common = [k for k in months if k in s2]
    if t == "months_cmp":
        return [k for k in common if s1[k] > s2[k]]
    if t == "topk_ratio":
        r = {k: s1[k] / s2[k] for k in common if s2[k]}
        return sorted(r, key=lambda k: -r[k])[:spec["k"]]
    if t == "mean_ratio":
        if spec.get("agg") == "ratio_of_totals":
            den = sum(s2[k] for k in common)
            x = sum(s1[k] for k in common) / den if den else None
        else:
            x = statistics.mean(s1[k] / s2[k] for k in common if s2[k])
        if x is None:
            return None
        pct = spec["fmt"] == "pct" if "fmt" in spec else spec.get("share", True)
        return f"{x * 100:.2f}%" if pct else round(x, 2)
    if t == "months_mom":
        return [b for a, b in zip(common, common[1:], strict=False) if s1[b] > s1[a] and s2[b] < s2[a]]
    raise ValueError(t)


def periodic_answer(hidden: dict, rule: str = "latest", unscaled: bool = False, swap_adj: bool = False) -> Any:
    spec = hidden["spec"]
    S = []
    for mkey, adj, col in spec["series"]:
        if swap_adj:
            adj = "NSA" if adj == "SA" else "SA"
        S.append(series_by_rule(hidden, mkey, adj, col, rule, scale=1.0 if unscaled else None))
    return answer_periodic(spec, S, hidden["domain"])


def entity_value(ent: dict, key: str, per: str, domain: str, unscaled: bool = False) -> float:
    if domain != "company":
        return float(ent["printed"]["cur"][key])
    v = ent["printed"][per][key]
    if unscaled:
        return v
    unit = CO_FIELDS[key]["unit"] if key in CO_FIELDS else "usd"
    return v * (ent["dscale"] if unit == "usd" else ent["sscale"] if unit == "shares" else 1.0)


def entity_answer(hidden: dict, unscaled: bool = False) -> Any:
    spec, ents, dom = hidden["spec"], hidden["entities"], hidden["domain"]
    t, f = spec["tmpl"], spec.get("fields", [])
    names = [e["name"] for e in ents]

    def val(e, k, per="cur"):
        return entity_value(e, k, per, dom, unscaled)

    if t == "agg":
        xs = [val(e, f[0]) for e in ents]
        fn = {"sum": sum, "mean": statistics.mean, "median": statistics.median, "min": min, "max": max}[spec["op"]]
        return round(float(fn(xs)), 2)
    if t in ("arg", "topk"):
        if len(f) == 2:
            sc = {i: val(e, f[0]) / val(e, f[1]) for i, e in enumerate(ents) if val(e, f[1])}
        else:
            sc = {i: val(e, f[0]) for i, e in enumerate(ents)}
        order = sorted(sc, key=lambda i: sc[i], reverse=not spec["lo"])
        return [names[order[0]]] if t == "arg" else [names[i] for i in order[:spec["k"]]]
    if t == "filter_cmp":
        return [names[i] for i, e in enumerate(ents) if val(e, f[0]) > val(e, f[1])]
    if t == "count_cond":
        if len(f) == 1:
            return float(sum(1 for e in ents if val(e, f[0]) < 0))
        return float(sum(1 for e in ents if val(e, f[1]) and val(e, f[0]) / val(e, f[1]) < spec["thr"]))
    if t == "yoy":
        k, up = f[0], spec["up"]
        return [names[i] for i, e in enumerate(ents) if (val(e, k) > val(e, k, "prior")) == up and val(e, k) != val(e, k, "prior")]
    if t == "pct_entities":
        n = sum(1 for e in ents if val(e, f[1]) and val(e, f[0]) / val(e, f[1]) > spec["thr"])
        return f"{n / len(ents) * 100:.2f}%"
    if t == "avg_filtered":
        med = statistics.median(val(e, f[1]) for e in ents)
        sel = [val(e, f[0]) for e in ents if val(e, f[1]) > med]
        return round(float(statistics.mean(sel)), 2)
    raise ValueError(t)


__all__ = ["answer_periodic", "entity_answer", "entity_value", "periodic_answer", "series_by_rule"]
