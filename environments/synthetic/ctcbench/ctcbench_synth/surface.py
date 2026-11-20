"""How a CTC row looks (ids, headers, layout), drawn per row independently of its content."""

from __future__ import annotations

import json
import random
import re
import string
from dataclasses import dataclass, field
from typing import Any

from ctcbench_real import env as R

from .data import spec_of

FAMILIES = ("qdmatch_hpqa", "outlier", "xabsence_paraphrase", "contradiction_iid", "grouping")
ALIASES = {"qdmatch": "qdmatch_hpqa", "contradiction": "contradiction_iid", "xabsence_para": "xabsence_paraphrase"}
KIND = {"qdmatch_hpqa": "pairs", "contradiction_iid": "pairs", "outlier": "set", "xabsence_paraphrase": "set",
        "grouping": "partition"}
NATIVE_LABEL = {"outlier": "Outliers", "xabsence_paraphrase": "Unmatched"}

HEADERS = ("bracket", "hash", "xml", "paren", "jsonl", "yaml", "table", "colon")
PRESETS = {h: {"header": h} for h in HEADERS}
SEPARATORS = ("\n\n", "\n", "\n\n---\n\n", "\n* * *\n")
ID_SCHEMES = ("int", "offset", "pad", "alnum")
LABELS = {
    "query": ["Query", "Question", "Q", "Ask"],
    "doc": ["Document", "Passage", "Snippet", "Record", "Entry"],
    "outlier": ["Document", "Passage", "Excerpt", "Segment", "Text"],
    "contradiction": ["Claim", "Statement", "Finding", "Assertion"],
    "grouping": ["Paper", "Abstract", "Document", "Article", "Record"],
}
SIDES = [("A", "B"), ("Set 1", "Set 2"), ("Left", "Right"), ("Source", "Target"), ("Corpus X", "Corpus Y")]
PREAMBLES = ["", "", "Corpus export: {n} records.", "source: archive-{k} | records: {n}", "# Collection {k} ({n} items)",
             "The following {n} entries were extracted for review."]
FOOTERS = ["", "", "(end of records)", "--- END ---", "[{n} records listed]"]

TASKS = {
    "qdmatch_hpqa": [
        "Some of the {qs} can be answered by one of the {ds}. Find every relevant pair: a {q} together with a {d} that "
        "answers it.",
        "The collection mixes {qs} and {ds}. A few {ds} answer specific {qs}. Identify all such ({q}, {d}) pairs.",
        "Match {qs} to the {ds} that answer them. Only a few pairs are relevant; list every one of them.",
    ],
    "outlier": [
        "Can you find passages that are about a different topic than the rest of these passages?",
        "Most of these {items} share their subject matter with others in the collection. Find the {items} that are "
        "about a different topic than the rest.",
        "Identify the {items} whose topic differs from the rest of the collection.",
    ],
    "xabsence_paraphrase": [
        "There are two collections of claims, {a} and {b}. Almost every claim has a paraphrase (a restatement of the "
        "same fact) somewhere in the OTHER collection. A few claims are unmatched: they have no paraphrase in the "
        "other collection. Identify the unmatched claims.",
        "Claims in {a} and {b} mostly restate each other: nearly every claim is paraphrased somewhere on the other "
        "side. Find the claims that have no paraphrase on the other side.",
        "Each claim belongs to {a} or {b}. Find every claim that is not restated anywhere in the other collection.",
    ],
    "contradiction_iid": [
        "Identify all pairs of {items} that contradict each other. A pair is contradictory if the two cannot both be "
        "true at the same time.",
        "Some {items} in this collection conflict: find every pair of {items} that cannot both be true.",
        "Find all pairs of contradicting {items}.",
    ],
    "grouping": [
        "Partition these {n} {items} into exactly {k} groups, putting together the {items} that belong to the same "
        "subject category.",
        "Sort the {items} into {k} topical categories. Every {item} goes into exactly one category.",
        "Group the {n} {items} by what they are about into {k} clusters.",
    ],
}
TITLE_STYLES = ("Title: {t}\n{a}", "{t}\n{a}", "{t}. {a}", "TITLE: {t} | ABSTRACT: {a}")


@dataclass
class Rendered:
    context: str
    task_text: str
    root_prompt: str
    answer_spec: str
    id_map: dict[str, Any]
    style: dict = field(default_factory=dict)


def _family(f: str) -> str:
    f = ALIASES.get(f, f)
    if f not in FAMILIES:
        raise ValueError(f"surface: unsupported family {f!r}; expected one of {FAMILIES}")
    return f


def _ids(n: int, scheme: str, rng: random.Random, prefix: str = "") -> list[str]:
    if scheme == "int":
        return [f"{prefix}{i + 1}" for i in range(n)]
    if scheme == "offset":
        start = rng.choice([0, 7, 100, 101, 1000, 5001])
        return [f"{prefix}{start + i}" for i in range(n)]
    if scheme == "pad":
        w = max(len(str(n)) + 1, rng.choice([3, 4, 5]))
        p = prefix or rng.choice(["p", "d", "r", "x", "#", "N-", "id"])
        return [f"{p}{i + 1:0{w}d}" for i in range(n)]
    p = prefix or rng.choice(["", "id-", "k", "ref:"])
    L = rng.choice([4, 5, 6])
    seen, out = set(), []
    while len(out) < n:
        c = "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(L))
        if c in seen or not (any(ch.isdigit() for ch in c) and any(ch.isalpha() for ch in c)):
            continue
        seen.add(c)
        out.append(p + c)
    return out


def _tokens(text: str, ids: set[str]) -> list[str]:
    """Surface ids mentioned in ``text``, in order (only strings that are ids of this row count)."""
    out = []
    for m in re.finditer(r"[A-Za-z0-9#:_\-]+", text):
        t = m.group(0).strip("-:_")
        if t in ids:
            out.append(t)
        else:
            for cand in (m.group(0), m.group(0).rstrip(":"), m.group(0).lstrip("#")):
                if cand in ids:
                    out.append(cand)
                    break
    return out


def _text(fam: str, st: dict, doc: dict) -> str:
    """An item's text: the document text, or for grouping its title joined to the abstract in the row's style."""
    if fam == "grouping" and doc.get("title"):
        return TITLE_STYLES[st["tstyle"]].format(t=doc["title"], a=doc["text"])
    return doc["text"]


def _item(header: str, sid: str, label: str, text: str, tag: str) -> str:
    text = text if header != "table" else " ".join(text.split()).replace("|", "/")
    if header == "bracket":
        return f"[{sid}] {label}: {text}" if label else f"[{sid}] {text}"
    if header == "hash":
        return f"### {label} {sid}\n{text}".replace("###  ", "### ")
    if header == "xml":
        return f'<{tag} id="{sid}"{f" kind={json.dumps(label)}" if label else ""}>\n{text}\n</{tag}>'
    if header == "paren":
        return f"{sid}) {f'({label}) ' if label else ''}{text}"
    if header == "jsonl":
        d = {"id": sid}
        if label:
            d["kind"] = label
        d["text"] = text
        return json.dumps(d, ensure_ascii=False)
    if header == "yaml":
        return f"- id: {sid}\n" + (f"  kind: {label}\n" if label else "") + "  text: " + json.dumps(text, ensure_ascii=False)
    if header == "table":
        return f"| {sid} | {label} | {text} |" if label else f"| {sid} | {text} |"
    return f"{label} {sid}: {text}" if label else f"{sid}: {text}"


def _draw(rng: random.Random, fam: str, n_docs: int, preset: dict | None) -> dict:
    header = (preset or {}).get("header") or rng.choice(HEADERS)
    if header == "table" and fam in ("outlier", "grouping"):
        header = rng.choice(["bracket", "colon", "xml"])
    sep = "\n" if header in ("jsonl", "table") else rng.choice(SEPARATORS)
    st = {"header": header, "sep": sep, "scheme": rng.choice(ID_SCHEMES), "task_pos": rng.choice(["outside", "before",
          "after"]), "preamble": rng.choice(PREAMBLES), "footer": rng.choice(FOOTERS), "k": rng.randint(2, 99),
          "tpl": rng.randrange(3), "answer_style": rng.choice(["line", "json"]), "tag": rng.choice(["item", "doc",
          "entry", "record"])}
    if fam == "qdmatch_hpqa":
        st.update(qlabel=rng.choice(LABELS["query"]), dlabel=rng.choice(LABELS["doc"]),
                  layout=rng.choice(["interleaved", "queries_first", "docs_first"]),
                  numbering=rng.choice(["shared", "separate"]))
    elif fam == "outlier":
        st.update(label=rng.choice(LABELS["outlier"] + [""]))
    elif fam == "contradiction_iid":
        st.update(label=rng.choice(LABELS["contradiction"] + [""]))
    elif fam == "grouping":
        st.update(label=rng.choice(LABELS["grouping"] + [""]), tstyle=rng.randrange(len(TITLE_STYLES)),
                  gstyle=rng.choice(["lists", "groups_obj"]))
    else:
        st.update(sides=rng.choice(SIDES), layout=rng.choice(["sections", "tagged"]),
                  numbering=rng.choice(["shared", "separate"]))
    return st


def render(row: dict, family: str, mode: str = "varied", seed: int | None = None) -> Rendered:
    """Render ``row`` in a surface style ("native", "varied" or a ``PRESETS`` key); the default seed is a hash of the row."""
    fam = _family(family)
    spec = spec_of(fam)
    docs = row["documents"]
    n = len(docs)
    if mode == "native":
        context, task_text, _q = R.render(row, spec)
        idm = {"family": fam, "kind": KIND[fam], "native": True, "ids": {str(i + 1): i + 1 for i in range(n)}}
        return Rendered(context, task_text, f"{task_text}\n\n{R.submit_hint(task_text)}", "", idm, {"mode": "native"})
    if seed is None:
        seed = int.from_bytes(" ".join(d.get("text", "")[:80] for d in docs[:20]).encode()[:8].ljust(8, b"\0"), "big")
    rng = random.Random(seed)
    st = _draw(rng, fam, n, PRESETS.get(mode) if mode != "varied" else None)
    st["mode"] = mode

    order = list(range(n))
    labels = [""] * n
    sid = [""] * n
    if fam == "qdmatch_hpqa":
        isq = [d.get("type") == "query" for d in docs]
        if st["numbering"] == "separate":
            qp, dp = (("Q", "D") if st["scheme"] in ("int", "offset") else (st["qlabel"][:1].upper() + "-", "D-"))
            if qp[0] == dp[0]:
                dp = "P-"
            qi = _ids(sum(isq), st["scheme"] if st["scheme"] != "alnum" else "pad", rng, qp)
            di = _ids(n - sum(isq), st["scheme"] if st["scheme"] != "alnum" else "pad", rng, dp)
            qit, dit = iter(qi), iter(di)
            sid = [next(qit) if q else next(dit) for q in isq]
        else:
            sid = _ids(n, st["scheme"], rng)
        labels = [st["qlabel"] if q else st["dlabel"] for q in isq]
        if st["layout"] == "queries_first":
            order = [i for i in range(n) if isq[i]] + [i for i in range(n) if not isq[i]]
        elif st["layout"] == "docs_first":
            order = [i for i in range(n) if not isq[i]] + [i for i in range(n) if isq[i]]
    elif fam == "xabsence_paraphrase":
        sa, sb = st["sides"]
        side = [sa if d.get("corpus") == "A" else sb for d in docs]
        if st["numbering"] == "separate":
            pa, pb = ("A", "B") if st["scheme"] in ("int", "offset", "alnum") else ("L", "R")
            ia = _ids(side.count(sa), "pad" if st["scheme"] == "alnum" else st["scheme"], rng, pa)
            ib = _ids(side.count(sb), "pad" if st["scheme"] == "alnum" else st["scheme"], rng, pb)
            ita, itb = iter(ia), iter(ib)
            sid = [next(ita) if s == sa else next(itb) for s in side]
        else:
            sid = _ids(n, st["scheme"], rng)
        labels = side if st["layout"] == "tagged" else [""] * n
    else:
        sid = _ids(n, st["scheme"], rng)
        labels = [st["label"]] * n
    assert len(set(sid)) == n, "surface ids must be unique"

    sep, header = st["sep"], st["header"]
    blocks = []
    if fam == "xabsence_paraphrase" and st["layout"] == "sections":
        for s in st["sides"]:
            members = [i for i in order if (docs[i].get("corpus") == "A") == (s == st["sides"][0])]
            body = sep.join(_item(header, sid[i], "", docs[i]["text"], st["tag"]) for i in members)
            if header == "table":
                body = "| id | claim |\n|---|---|\n" + body
            blocks.append(f"## {s}\n\n{body}")
        items = "\n\n".join(blocks)
    else:
        body = sep.join(_item(header, sid[i], labels[i], _text(fam, st, docs[i]), st["tag"]) for i in order)
        if header == "table":
            body = ("| id | kind | text |\n|---|---|---|\n" if any(labels) else "| id | text |\n|---|---|\n") + body
        items = body

    task_text, answer_spec = _task(fam, st, sid, row, rng)
    pre = st["preamble"].format(n=n, k=st["k"])
    foot = st["footer"].format(n=n)
    parts = [pre] if pre else []
    if st["task_pos"] == "before":
        parts.append(task_text + "\n\n" + answer_spec)
    parts.append(items)
    if foot:
        parts.append(foot)
    if st["task_pos"] == "after":
        parts.append(task_text + "\n\n" + answer_spec)
    context = "\n\n".join(parts)
    full_task = f"{task_text}\n\n{answer_spec}"
    root_prompt = f"{full_task}\n\n{R._SUBMIT.format(brackets='')}".replace(
        "the corpus followed by the task statement", "the collection of items the task refers to")
    idm = {"family": fam, "kind": KIND[fam], "native": False, "ids": {s: i + 1 for i, s in enumerate(sid)},
           "answer_style": st["answer_style"]}
    return Rendered(context, full_task, root_prompt, answer_spec, idm, st)


def _example_ids(sid: list[str], row: dict, fam: str, rng: random.Random, k: int) -> list[str]:
    gold = set()
    g = R._gold_of(row, spec_of(fam)) or []
    base = spec_of(fam).gold_index_base
    for v in (g if isinstance(g, list) else []):
        for x in (v if isinstance(v, (list, tuple)) else [v]):
            try:
                gold.add(int(x) - base)
            except (TypeError, ValueError):
                pass
    pool = [i for i in range(len(sid)) if i not in gold]
    return [sid[i] for i in rng.sample(pool, min(k, len(pool)))]


def _partition_example(sid: list[str], row: dict, rng: random.Random) -> tuple[str, str, str]:
    """Three ids for a partition example (the first two from different gold groups)."""
    grp = {}
    for gi, g in enumerate(R._gold_of(row, spec_of("grouping")) or []):
        for x in g:
            grp[int(x)] = gi
    while True:
        a, b, c = rng.sample(range(len(sid)), 3)
        if grp.get(a, -1) != grp.get(b, -2):
            return sid[a], sid[b], sid[c]


def _quote(s: str, ints: bool) -> str:
    return s if ints else json.dumps(s)


def _task(fam: str, st: dict, sid: list[str], row: dict, rng: random.Random) -> tuple[str, str]:
    ints = all(s.isdigit() for s in sid)
    ex = _example_ids(sid, row, fam, rng, 4)
    if fam == "qdmatch_hpqa":
        isq = [x.get("type") == "query" for x in row["documents"]]
        qx = [s for s in ex if isq[sid.index(s)]] + [s for i, s in enumerate(sid) if isq[i]][:2]
        dx = [s for s in ex if not isq[sid.index(s)]] + [s for i, s in enumerate(sid) if not isq[i]][:2]
        ex = [qx[0], dx[0], qx[1], dx[1]]
        q, d = st["qlabel"].lower(), st["dlabel"].lower()
        q = "query" if q in ("q", "ask") else q
        task = TASKS[fam][st["tpl"]].format(qs=q + ("s" if not q.endswith("s") else ""), ds=d + "s", q=q, d=d)
        spec = (f"Answer with a JSON list of [{q}_id, {d}_id] pairs, using the ids exactly as they appear (the "
                f"{q}'s id first), e.g. [[{_quote(ex[0], ints)}, {_quote(ex[1], ints)}], [{_quote(ex[2], ints)}, "
                f"{_quote(ex[3], ints)}]]. If no pair is relevant, answer [].")
        return task, spec
    if fam == "contradiction_iid":
        items = (st["label"] or "item").lower() + "s"
        task = TASKS[fam][st["tpl"]].format(items=items)
        spec = (f"Answer with a JSON list of pairs of ids, using the ids exactly as they appear, e.g. "
                f"[[{_quote(ex[0], ints)}, {_quote(ex[1], ints)}], [{_quote(ex[2], ints)}, {_quote(ex[3], ints)}]]. "
                "If no pair contradicts, answer [].")
        return task, spec
    if fam == "grouping":
        item = (st["label"] or "paper").lower()
        k = len(R._gold_of(row, spec_of(fam)) or [])
        task = TASKS[fam][st["tpl"]].format(n=len(sid), k=k, items=item + "s", item=item)
        a, b, c = _partition_example(sid, row, rng)
        ex = f"[[{_quote(a, ints)}, {_quote(b, ints)}], [{_quote(c, ints)}], ...]"
        if st["gstyle"] == "groups_obj":
            spec = (f'Answer with a JSON object {{"groups": [...]}} whose value is a list of exactly {k} groups, each a list '
                    f'of ids exactly as they appear, e.g. {{"groups": {ex}}}. Every id must be in exactly one group.')
        else:
            spec = (f"Answer with a JSON list of exactly {k} groups, each a list of ids exactly as they appear, e.g. {ex}. "
                    "Every id must be in exactly one group.")
        return task, spec
    if fam == "outlier":
        items = (st["label"] or "passage").lower() + "s"
        task = TASKS[fam][st["tpl"]].format(items=items)
    else:
        a, b = st["sides"]
        task = TASKS[fam][st["tpl"]].format(a=a, b=b)
    lab = rng.choice({"outlier": ["Outliers", "Different topic", "Off-topic"],
                      "xabsence_paraphrase": ["Unmatched", "Unpaired", "No paraphrase"]}[fam])
    if st["answer_style"] == "json":
        spec = (f"Answer with a JSON list of the ids, exactly as they appear, e.g. [{_quote(ex[0], ints)}, "
                f"{_quote(ex[1], ints)}].")
    else:
        spec = f"Answer with one line, ids exactly as they appear, comma-separated: \"{lab}: {ex[0]}, {ex[1]}, ...\""
    return task, spec


def to_canonical(answer: Any, id_map: dict) -> str:
    """Map an answer in surface ids to the benchmark's native answer string (unknown ids dropped)."""
    fam, kind = id_map["family"], id_map["kind"]
    text = json.dumps(answer) if isinstance(answer, (list, tuple)) else str(answer or "")
    if id_map.get("native"):
        return text
    ids = id_map["ids"]
    if kind == "pairs":
        pairs = None
        m = re.search(r"\[.*\]", text, re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                if isinstance(v, list) and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in v):
                    pairs = [[ids.get(str(a)), ids.get(str(b))] for a, b in v]
            except json.JSONDecodeError:
                pairs = None
        if pairs is None:
            toks = _tokens(text, set(ids))
            pairs = [[ids[a], ids[b]] for a, b in zip(toks[0::2], toks[1::2])]
        return json.dumps([p for p in pairs if None not in p])
    if kind == "partition":
        return json.dumps({"groups": [{"doc_ids": g} for g in _partition(text, ids)]})
    found = list(dict.fromkeys(ids[t] for t in _tokens(text, set(ids))))
    lab = NATIVE_LABEL[fam]
    return f"{lab}: " + ", ".join(f"[{i}]" for i in found) if found else f"{lab}: none"


def _partition(text: str, ids: dict) -> list[list[int]]:
    """Groups of canonical ids from a partition answer (JSON or one group per line)."""
    groups = None
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                v = v.get("groups", [])
            if isinstance(v, list):
                groups = [g.get("doc_ids", g.get("ids", [])) if isinstance(g, dict) else g for g in v]
                groups = [g if isinstance(g, list) else [g] for g in groups]
        except json.JSONDecodeError:
            groups = None
    if groups is None:
        groups = [_tokens(line, set(ids)) for line in text.splitlines()]
    seen, out = set(), []
    for g in groups:
        c = [ids[str(x)] for x in g if str(x) in ids and ids[str(x)] not in seen]
        seen.update(c)
        if c:
            out.append(c)
    return out


def gold_answer(row: dict, family: str, rendered: Rendered) -> str:
    """The gold answer written in the row's surface ids (what a perfect model would submit)."""
    fam = _family(family)
    spec = spec_of(fam)
    if rendered.id_map.get("native"):
        return R.gold_target(row, spec)
    inv = {v: k for k, v in rendered.id_map["ids"].items()}
    ints = all(k.isdigit() for k in inv.values())
    g = R._gold_of(row, spec)
    base = spec.gold_index_base
    if KIND[fam] == "partition":
        gs = [[inv[int(x) - base + 1] for x in grp] for grp in g]
        gs = [[int(x) for x in grp] for grp in gs] if ints else gs
        return json.dumps({"groups": gs} if rendered.style.get("gstyle") == "groups_obj" else gs)
    if KIND[fam] == "pairs":
        ps = [[inv[int(a) - base + 1], inv[int(b) - base + 1]] for a, b in g]
        return json.dumps([[int(a), int(b)] for a, b in ps] if ints else ps)
    ss = [inv[int(x) - base + 1] for x in g]
    if rendered.id_map.get("answer_style") == "json":
        return json.dumps([int(s) for s in ss] if ints else ss)
    return f"{NATIVE_LABEL[fam]}: " + ", ".join(ss)


def score(answer: Any, row: dict, family: str, id_map: dict) -> dict[str, float]:
    """Benchmark scorer on an answer written in surface ids."""
    from .data import score as native_score
    return native_score(to_canonical(answer, id_map), row, _family(family))


__all__ = ["FAMILIES", "HEADERS", "PRESETS", "Rendered", "gold_answer", "render", "score", "to_canonical"]
