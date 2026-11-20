"""Build one synthetic CTC instance per (family, row, rung, seed): a real row, or several rows combined toward a
longer rung. Instances keep the benchmark's format, so its own render / score apply; ``facts`` feed the oracle.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cache

from .data import examples, meta, render, resolve, row_key, sidecar, text_key

EVAL_ITEMS = {
    "nq": {"64k": 412, "128k": 825, "256k": 1650, "1m": 6572},
    "hotpotqa": {"64k": 600, "128k": 1201, "256k": 2400, "1m": 9631},
    "contradiction_iid": {"s2k": 44, "s4k": 92, "64k": 1450, "128k": 2907, "256k": 5800, "1m": 23147},
    "outlier": {"s4k": 27, "32k": 220, "64k": 450, "128k": 901, "128k_boundary": 901,
                "256k": 1800, "1m": 7209},
    "outlier_fixedM": {"64k": 450, "128k": 901, "256k": 1800, "512k": 3605},
    "qdmatch_hpqa": {"64k": 872, "128k": 1744, "256k": 3488},
    "grouping": {"hik": 176},
    "reorder": {"w13": 13, "w27": 27},
}


@dataclass
class Instance:
    family: str
    ex: dict
    facts: dict
    rung: str = "native"
    context: str = ""
    root_prompt: str = ""
    items: list = field(default_factory=list)
    required: list = field(default_factory=list)


_WORD = re.compile(r"[a-z0-9]+")


def toks(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _facts_retrieval(ex: dict, family: str) -> dict:
    docs = ex["documents"]
    gold = [int(i) + 1 for i in ex.get("gold_doc_indices") or []]
    hard = [int(i) + 1 for i in ex.get("hard_neg_indices") or []]
    q = (ex.get("queries") or [""])[0]
    qt = set(toks(q))
    bridge = None
    if family == "hotpotqa" and len(gold) == 2:
        ov = [len(qt & set(toks(docs[g - 1]["text"]))) for g in gold]
        if min(ov) * 2 <= max(ov):
            bridge = gold[ov.index(min(ov))]
    return {"question": q, "answers": list(ex.get("answers") or []), "gold_ids": gold, "hard": hard, "bridge": bridge}


def _doc_entity(text: str) -> str:
    """Leading entity of a HotpotQA paragraph ("Marlene Favela (born ...) is ..." -> "Marlene Favela")."""
    m = re.match(r'\s*"?([^(,.;:"]{2,80}?)"?\s*(?:\(|,| is | was | are | were | refers | has | had )', text)
    return (m.group(1).strip() if m else " ".join(text.split()[:3])).strip()


def _facts_qdmatch(ex: dict) -> dict:
    docs = ex["documents"]
    queries = [i + 1 for i, d in enumerate(docs) if d.get("type") == "query"]
    dids = [i + 1 for i, d in enumerate(docs) if d.get("type") != "query"]
    pairs = [[int(a), int(b)] for a, b in ex.get("gold_pairs") or []]
    rel: dict[str, list[int]] = defaultdict(list)
    for a, b in pairs:
        rel[str(a)].append(b)
    ent = {str(d): _doc_entity(docs[d - 1]["text"]) for d in dids}
    return {"queries": queries, "docs": dids, "pairs": pairs, "rel": dict(rel), "ent": ent}


def _facts_contradiction(ex: dict) -> dict:
    pairs = [sorted([int(a), int(b)]) for a, b in ex.get("gold_doc_indices") or []]
    partner = {}
    for a, b in pairs:
        partner[str(a)], partner[str(b)] = b, a
    return {"pairs": pairs, "partner": partner}


def _facts_outlier(ex: dict, labels: list) -> dict:
    gold = [int(i) + 1 for i in ex.get("gold_doc_indices") or []]
    titles = sidecar("outlier_articles.json.gz").get("titles") or []
    art = {str(i + 1): int(a) for i, (a, _j) in enumerate(labels)}
    size = Counter(art.values())
    return {"gold_ids": gold, "art": art, "chunk": {str(i + 1): int(j) for i, (_a, j) in enumerate(labels)},
            "title": {str(a): titles[a] if 0 <= a < len(titles) else f"article {a}" for a in size},
            "size": {str(a): n for a, n in size.items()}}


def _facts_grouping(ex: dict) -> dict:
    labels: dict[str, str] = {}
    try:
        groups = json.loads((ex.get("answers") or ["{}"])[0]).get("groups") or []
    except (json.JSONDecodeError, AttributeError):
        groups = []
    for g in groups:
        for d in g.get("doc_ids") or []:
            labels[str(int(d))] = str(g.get("label") or "")
    if not labels:
        cl = ex.get("cluster_labels") or []
        for k, grp in enumerate(ex.get("gold_doc_indices") or []):
            for d in grp:
                labels[str(int(d) + 1)] = str(cl[k]) if k < len(cl) else f"group {k}"
    n = len(ex["documents"])
    for i in range(1, n + 1):
        labels.setdefault(str(i), f"unlabelled {i}")
    return {"label": labels, "K": len(set(labels.values())), "N": n, "level": int(ex.get("level") or 0)}


def _facts_reorder(ex: dict) -> dict:
    order = [int(i) for i in ex["gold_order"]]
    seams = ex.get("_seams")
    if seams is None:
        seams = sidecar("reorder_seams.json").get(row_key(ex)) or []
    return {"order": order, "pos": {str(d): p for p, d in enumerate(order)}, "seams": [int(s) for s in seams]}


def _compose_reorder_window(base: dict, w: int, rng: random.Random) -> dict:
    """``w`` consecutive passages of one row, reshuffled and renumbered (shorter windows give a denser reward)."""
    docs, order = base["documents"], [int(i) for i in base["gold_order"]]
    seams = sidecar("reorder_seams.json").get(row_key(base)) or []
    k0 = rng.randrange(0, max(1, len(order) - w + 1))
    src = order[k0:k0 + w]
    shown = src[:]
    rng.shuffle(shown)
    new_id = {old: i + 1 for i, old in enumerate(shown)}
    out = {k: v for k, v in base.items() if k not in ("documents", "gold_order", "answers", "_meta")}
    out["documents"] = [docs[i - 1] for i in shown]
    out["gold_order"] = [new_id[i] for i in src]
    out["answers"] = []
    out["_meta"] = {"n": w, "window": w, "start": k0}
    out["_seams"] = [s - k0 for s in seams if k0 < s < k0 + w]
    return out


def _facts_textgroups(ex: dict) -> dict:
    m = meta(ex)
    gold = [sorted(int(i) for i in g) for g in ex.get("gold_doc_indices") or []]
    return {"counts": [int(c) for c in m.get("counts") or []], "feature": m.get("feature"), "word": m.get("word"),
            "target": int(m.get("target") or 70), "group_size": int(m.get("group_size") or 3), "gold_groups": gold}


def facts_for(family: str, ex: dict, labels: list | None = None) -> dict:
    if family in ("nq", "hotpotqa"):
        return _facts_retrieval(ex, family)
    if family == "qdmatch_hpqa":
        return _facts_qdmatch(ex)
    if family == "contradiction_iid":
        return _facts_contradiction(ex)
    if family in ("outlier", "outlier_fixedM"):
        if labels is None:
            labels = sidecar("outlier_articles.json.gz")["rows"][row_key(ex)]
        return _facts_outlier(ex, labels)
    if family == "grouping":
        return _facts_grouping(ex)
    if family == "reorder":
        return _facts_reorder(ex)
    if family == "textgroups":
        return _facts_textgroups(ex)
    raise ValueError(family)


def eligible(family: str, ex: dict) -> bool:
    """Rows the synthetic env serves. Drops rows whose gold cannot be reached by any honest program."""
    if family == "textgroups":
        return meta(ex).get("feature") != "verbs"
    if family in ("outlier", "outlier_fixedM"):
        lab = (sidecar("outlier_articles.json.gz").get("rows") or {}).get(row_key(ex))
        if not lab:
            return False
        gold = {int(i) for i in ex.get("gold_doc_indices") or []}
        ga = {lab[i][0] for i in gold}
        c = Counter(a for a, _ in lab)
        return len(ga) == 1 and c[next(iter(ga))] == 3 and all(n >= 4 for a, n in c.items() if a not in ga)
    if family == "grouping":
        return True
    if family == "reorder":
        return row_key(ex) in (sidecar("reorder_seams.json") or {})
    return True


@cache
def pool(family: str, split: str) -> tuple[dict, ...]:
    fam = resolve(family)
    rows = tuple(ex for ex in examples(fam, split) if eligible(fam, ex))
    if fam == "outlier" and split == "train":
        val_rows = [ex for val_fam in ("outlier", "outlier_fixedM")
                    for ex in examples(val_fam, "val")]
        val_texts = {text_key(d.get("text", "")) for ex in val_rows
                     for d in ex.get("documents") or []}
        labels = sidecar("outlier_articles.json.gz")["rows"]
        val_articles = {a for ex in val_rows if row_key(ex) in labels
                        for a, _ in labels[row_key(ex)]}
        rows = tuple(ex for ex in rows
                     if all(text_key(d.get("text", "")) not in val_texts
                            for d in ex.get("documents") or [])
                     and not ({a for a, _ in labels[row_key(ex)]} & val_articles))
    return rows


def _norm_q(q: str) -> str:
    return " ".join(toks(q))


def _compose_retrieval(family: str, split: str, base: dict, target: int, rng: random.Random) -> dict:
    docs = list(base["documents"])
    seen = {text_key(d["text"]) for d in docs}
    bq = _norm_q((base.get("queries") or [""])[0])
    others = [ex for ex in pool(family, split) if ex is not base and _norm_q((ex.get("queries") or [""])[0]) != bq]
    rng.shuffle(others)
    filler = []
    for ex in others:
        if len(docs) + len(filler) >= target:
            break
        for d in ex["documents"]:
            k = text_key(d["text"])
            if k not in seen:
                seen.add(k)
                filler.append({"text": d["text"]})
    filler = filler[: max(0, target - len(docs))]
    gold = {int(i) for i in base.get("gold_doc_indices") or []}
    hard = {int(i) for i in base.get("hard_neg_indices") or []}
    allp = [(d, i in gold, i in hard) for i, d in enumerate(docs)] + [(d, False, False) for d in filler]
    rng.shuffle(allp)
    out = {k: v for k, v in base.items() if k not in ("documents", "gold_doc_indices", "hard_neg_indices")}
    out["documents"] = [d for d, _, _ in allp]
    out["gold_doc_indices"] = [i for i, (_, g, _) in enumerate(allp) if g]
    out["hard_neg_indices"] = [i for i, (_, _, h) in enumerate(allp) if h]
    return out


def _shrink_contradiction(base: dict, target: int, rng: random.Random) -> dict:
    """A smaller row (the 2k/4k rung sizes): the 3 gold pairs + a random subset of the row's other claims."""
    claims = [d["text"] for d in base["documents"]]
    pair_ids = [(int(a) - 1, int(b) - 1) for a, b in base["gold_doc_indices"]]
    gold = {i for p in pair_ids for i in p}
    rest = [i for i in range(len(claims)) if i not in gold]
    keep = sorted(gold) + rng.sample(rest, max(0, min(len(rest), target - len(gold))))
    rng.shuffle(keep)
    newpos = {old: k + 1 for k, old in enumerate(keep)}
    out = {k: v for k, v in base.items() if k not in ("documents", "gold_doc_indices")}
    out["documents"] = [{"text": claims[i]} for i in keep]
    out["gold_doc_indices"] = sorted(sorted([newpos[a], newpos[b]]) for a, b in pair_ids)
    return out


def _compose_contradiction(split: str, base: dict, target: int, rng: random.Random) -> dict:
    if target < len(base["documents"]):
        return _shrink_contradiction(base, target, rng)
    claims = [d["text"] for d in base["documents"]]
    pair_ids = [(int(a) - 1, int(b) - 1) for a, b in base["gold_doc_indices"]]
    seen = {text_key(c) for c in claims}
    others = [ex for ex in pool("contradiction_iid", split) if ex is not base]
    rng.shuffle(others)
    filler = []
    for ex in others:
        if len(claims) + len(filler) >= target:
            break
        bad = {int(i) - 1 for p in ex["gold_doc_indices"] for i in p}
        for i, d in enumerate(ex["documents"]):
            k = text_key(d["text"])
            if i not in bad and k not in seen:
                seen.add(k)
                filler.append(d["text"])
    filler = filler[: max(0, target - len(claims))]
    allc = [(c, i) for i, c in enumerate(claims)] + [(c, None) for c in filler]
    rng.shuffle(allc)
    newpos = {old: k + 1 for k, (_, old) in enumerate(allc) if old is not None}
    out = {k: v for k, v in base.items() if k not in ("documents", "gold_doc_indices")}
    out["documents"] = [{"text": c} for c, _ in allc]
    out["gold_doc_indices"] = sorted(sorted([newpos[a], newpos[b]]) for a, b in pair_ids)
    return out


def _compose_qdmatch(split: str, base: dict, target: int, rng: random.Random, keep_rel: float = 0.15) -> dict:
    """Base row plus filler rows. Fillers sometimes keep their relevant queries, so the count isn't always 3."""
    items = list(base["documents"])
    gp = [(int(a) - 1, int(b) - 1) for a, b in base["gold_pairs"]]
    seen = {text_key(d["text"]) for d in items}
    others = [ex for ex in pool("qdmatch_hpqa", split) if ex is not base]
    rng.shuffle(others)
    filler, extra_pairs = [], []
    for ex in others:
        if len(items) + len(filler) >= target:
            break
        keep = rng.random() < keep_rel and len(items) + len(filler) + len(ex["documents"]) <= target
        bad = set() if keep else {int(i) - 1 for p in ex["gold_pairs"] for i in p}
        local = {}
        for i, d in enumerate(ex["documents"]):
            k = text_key(d["text"])
            if i not in bad and k not in seen:
                seen.add(k)
                local[i] = len(filler)
                filler.append(d)
        if keep:
            for a, b in ex["gold_pairs"]:
                if int(a) - 1 in local and int(b) - 1 in local:
                    extra_pairs.append((local[int(a) - 1], local[int(b) - 1]))
    filler = filler[: max(0, target - len(items))]
    extra_pairs = [(a, b) for a, b in extra_pairs if a < len(filler) and b < len(filler)]
    tagged = [(d, i) for i, d in enumerate(items)] + [(d, ("f", j)) for j, d in enumerate(filler)]
    qs = [t for t in tagged if t[0].get("type") == "query"]
    ds = [t for t in tagged if t[0].get("type") != "query"]
    rng.shuffle(qs)
    rng.shuffle(ds)
    allx = qs + ds
    newpos = {old: k + 1 for k, (_, old) in enumerate(allx) if old is not None}
    out = {k: v for k, v in base.items() if k not in ("documents", "gold_pairs")}
    out["documents"] = [d for d, _ in allx]
    out["gold_pairs"] = sorted([newpos[a], newpos[b]] for a, b in gp) + \
        sorted([newpos[("f", a)], newpos[("f", b)]] for a, b in extra_pairs)
    return out


@cache
def _article_runs(split: str) -> tuple[list, dict]:
    """(single-article chunk runs, gold articles) from the outlier rows of ``split``."""
    rows = sidecar("outlier_articles.json.gz").get("rows") or {}
    chunks: dict[int, dict[int, str]] = defaultdict(dict)
    gold_arts = set()
    for fam in ("outlier", "outlier_fixedM"):
        for ex in examples(fam, split):
            lab = rows.get(row_key(ex))
            if not lab:
                continue
            for i in ex.get("gold_doc_indices") or []:
                gold_arts.add(lab[int(i)][0])
            for (a, j), d in zip(lab, ex["documents"], strict=False):
                if a >= 0:
                    chunks[a][j] = d["text"]
    runs = []
    for a, ch in chunks.items():
        js = sorted(ch)
        cur = [js[0]]
        for j in js[1:]:
            if j == cur[-1] + 1:
                cur.append(j)
            else:
                runs.append((a, [ch[x] for x in cur]))
                cur = [j]
        runs.append((a, [ch[x] for x in cur]))
    runs = [r for r in runs if len(r[1]) >= 4]
    if split == "train":
        val_articles = {a for fam in ("outlier", "outlier_fixedM")
                        for ex in examples(fam, "val")
                        for a, _ in rows.get(row_key(ex), [])}
        runs = [(a, texts) for a, texts in runs if a not in val_articles]
    return runs, {"gold_arts": sorted(gold_arts)}


@cache
def _boundary_gold_windows(split: str) -> tuple[tuple[int, tuple[str, str, str]], ...]:
    """Source-disjoint consecutive triples with a topical shift after the first pair."""
    windows = []
    for article, texts in _article_runs(split)[0]:
        words = [set(toks(t)) for t in texts]
        for j in range(len(texts) - 2):
            a, b, c = words[j:j + 3]
            ab = len(a & b) / max(1, len(a | b))
            bc = len(b & c) / max(1, len(b | c))
            if ab >= 0.15 and bc <= 0.08 and ab - bc >= 0.05:
                windows.append((article, tuple(texts[j:j + 3])))
    return tuple(windows)


def _compose_outlier(family: str, split: str, base: dict, target: int, rng: random.Random,
                     boundary_gold: bool = False) -> tuple[dict, list]:
    rows = sidecar("outlier_articles.json.gz")["rows"]
    lab = rows[row_key(base)]
    gold = [int(i) for i in base["gold_doc_indices"]]
    ga = lab[gold[0]][0]
    if boundary_gold:
        windows = _boundary_gold_windows(split)
        assert windows, f"no boundary gold windows for {split}"
        ga, gold_texts = rng.choice(windows)
        out_docs = [(text, (ga, j), True) for j, text in enumerate(gold_texts)]
    else:
        out_docs = [(base["documents"][i]["text"], (ga, lab[i][1]), True) for i in gold]
    runs, _ = _article_runs(split)
    if target < 60:
        own = defaultdict(list)
        for i, (a, j) in enumerate(lab):
            if a != ga:
                own[a].append((j, base["documents"][i]["text"]))
        runs = [(a, [t for _, t in sorted(v)]) for a, v in own.items() if len(v) >= 4]
    order = list(range(len(runs)))
    rng.shuffle(order)
    eval_sizes = family == "outlier" and target >= 60
    if eval_sizes:
        order.sort(key=lambda k: len(runs[k][1]) < 6)
    used = {ga}
    seen = {text_key(t) for t, _, _ in out_docs}
    n = len(out_docs)
    hi = 14
    inject_distant_fourth = target >= 60 and rng.random() < 0.5
    for k in order:
        if n >= target:
            break
        a, texts = runs[k]
        if a in used:
            continue
        positions = None
        if inject_distant_fourth and len(texts) >= 8 and target - n >= 4:
            starts = [start for start in range(len(texts) - 2)
                      if any(min(abs(j - x) for x in (start, start + 1, start + 2)) >= 4
                             for j in range(len(texts)))]
            if starts:
                start = rng.choice(starts)
                far = [j for j in range(len(texts))
                       if min(abs(j - x) for x in (start, start + 1, start + 2)) >= 4]
                positions = [start, start + 1, start + 2, rng.choice(far)]
        distant_positions = positions is not None
        lo = 5 if eval_sizes and len(texts) >= 5 else 4
        size = 4 if positions is not None else (rng.randint(lo, min(hi, len(texts)))
                                                if target >= 60 else len(texts))
        if target - n < size:
            size = target - n
            if size < 4:
                break
        if positions is None:
            st = rng.randint(0, len(texts) - size)
            positions = list(range(st, st + size))
        window = [texts[j] for j in positions]
        if any(text_key(t) in seen for t in window):
            continue
        if distant_positions:
            inject_distant_fourth = False
        used.add(a)
        for j, t in zip(positions, window, strict=True):
            seen.add(text_key(t))
            out_docs.append((t, (a, j), False))
        n += size
    rng.shuffle(out_docs)
    ex = {k: v for k, v in base.items() if k not in ("documents", "gold_doc_indices", "answers", "_meta", "meta")}
    ex["documents"] = [{"text": t} for t, _, _ in out_docs]
    ex["gold_doc_indices"] = [i for i, (_, _, g) in enumerate(out_docs) if g]
    ex["answers"] = ["; ".join(str(i + 1) for i in ex["gold_doc_indices"])]
    ex["_meta"] = {"num_outliers": 3, "num_categories": len(used), "mode": "mixed",
                   "composite": True, "boundary_gold": boundary_gold}
    labels = [list(lb) for _, lb, _ in out_docs]
    return ex, labels


@cache
def _grouping_hik_rows(split: str) -> tuple[dict, ...]:
    return tuple(ex for ex in pool("grouping", split) if int(ex.get("level") or 0) >= 2)


def _compose_grouping_hik(split: str, target: int, rng: random.Random) -> dict:
    """Eval-32k regime: N=176 papers, level-2/3 concepts, K ~ 0.85 N (mostly singletons, ~28 gold pairs)."""
    src = list(_grouping_hik_rows(split))
    rng.shuffle(src)
    cap: dict[str, int] = {}
    by: dict[str, list] = defaultdict(list)
    seen, total = set(), 0
    for ex in src:
        f = _facts_grouping(ex)
        idx = list(range(len(ex["documents"])))
        rng.shuffle(idx)
        for i in idx:
            d = ex["documents"][i]
            k = text_key(d["text"])
            lab = f["label"][str(i + 1)]
            if k in seen:
                continue
            if lab not in cap:
                cap[lab] = 1 if rng.random() < 0.7 else rng.choice([2, 2, 2, 3, 3, 4])
            if len(by[lab]) >= cap[lab]:
                continue
            seen.add(k)
            by[lab].append((d, lab))
            total += 1
            if total >= target:
                break
        if total >= target:
            break
    keep = [x for members in by.values() for x in members]
    rng.shuffle(keep)
    keep = keep[:target]
    labels = sorted({lab for _, lab in keep})
    gid = {lab: k for k, lab in enumerate(labels)}
    groups = defaultdict(list)
    for i, (_, lab) in enumerate(keep):
        groups[gid[lab]].append(i)
    n, k = len(keep), len(labels)
    ex = {
        "documents": [d for d, _ in keep],
        "queries": [f"Cluster these {n} papers into {k} groups."],
        "answers": [json.dumps({"groups": [{"label": labels[g], "doc_ids": [i + 1 for i in m]}
                                           for g, m in sorted(groups.items())]})],
        "gold_doc_indices": [m for _, m in sorted(groups.items())],
        "cluster_labels": labels,
        "level": 3,
        "k": k,
        "source": "openalex_grouping_composite",
        "_meta": {"n": n, "composite": True},
    }
    return ex


def build(family: str, split: str, row: int, rung: str = "native", seed: int = 0) -> Instance:
    fam = resolve(family)
    rng = random.Random(f"ctcsynth:{fam}:{split}:{row}:{rung}:{seed}")
    labels = None
    if fam == "grouping" and rung == "hik":
        ex = _compose_grouping_hik(split, EVAL_ITEMS["grouping"]["hik"], rng)
    else:
        rows = pool(fam, split)
        base = rows[row % len(rows)]
        if rung == "native":
            ex = base
        else:
            target = EVAL_ITEMS[fam][rung]
            if fam in ("nq", "hotpotqa"):
                ex = _compose_retrieval(fam, split, base, target, rng)
            elif fam == "contradiction_iid":
                ex = _compose_contradiction(split, base, target, rng)
            elif fam == "qdmatch_hpqa":
                ex = _compose_qdmatch(split, base, target, rng)
            elif fam in ("outlier", "outlier_fixedM"):
                ex, labels = _compose_outlier(fam, split, base, target, rng,
                                              boundary_gold=(rung == "128k_boundary"))
            elif fam == "reorder":
                ex = _compose_reorder_window(base, target, rng)
            else:
                raise ValueError(f"{fam} has no composite rung {rung!r}")
    facts = facts_for(fam, ex, labels)
    context, root_prompt = render(ex, fam)
    items = [(str(i + 1), (f"{d['title']}\n{d['text']}" if fam == "grouping" and d.get("title") else d["text"]))
             for i, d in enumerate(ex["documents"])]
    inst = Instance(family=fam, ex=ex, facts=facts, rung=rung, context=context, root_prompt=root_prompt,
                    items=items)
    inst.required = _required(fam, facts, items)
    return inst


def _required(fam: str, facts: dict, items: list) -> list[str]:
    """Items a correct solution must have had read by a sub-call (coverage metric)."""
    if fam in ("nq", "hotpotqa"):
        return [str(g) for g in facts["gold_ids"]]
    if fam == "qdmatch_hpqa":
        return sorted({str(x) for p in facts["pairs"] for x in p})
    if fam == "contradiction_iid":
        return sorted({str(x) for p in facts["pairs"] for x in p})
    if fam in ("outlier", "outlier_fixedM"):
        return [str(g) for g in facts["gold_ids"]]
    if fam == "textgroups":
        return []
    return [i for i, _ in items]


__all__ = ["EVAL_ITEMS", "Instance", "build", "eligible", "facts_for", "pool", "toks"]
