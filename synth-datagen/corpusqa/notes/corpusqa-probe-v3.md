The documents are a corpus of periodic releases whose tables you have not seen. Do not read or print document text: Python finds every table, ONE sub-call reads the distinct tables and plans which table and value cell each needed series is, Python reads that cell from every document into a ledger (a later period's document overwrites; a sub-call reads any table Python cannot), and the answer is computed in Python. Three turns.

## Rules
- Turn 1 is exactly the code below with the question pasted. Turn 2 is exactly the LEDGER code, right after Turn 1: do not inspect tables or re-derive the plan first -- the LEDGER report shows whether every document was read and the values fit. Turn 3 computes and submits.
- Never print document text or slices of it. If the plan looks wrong (a series read in fewer documents than the corpus has, values that do not fit the series), ask ONE sub-call about the specific table (`show(...)` of it plus the question) and print its JSON -- do not inspect the table yourself.
- All arithmetic is Python. Use only the months of the question's period (also for "previous month" comparisons) and follow the question's own definitions exactly.
- Imports do not survive between turns: each block starts with its imports.
- Each turn is one short sentence of your own saying what the block does (never name these turns or code blocks), then ONE ```repl fenced block -- the only format the REPL runs.

## Turn 1 -- TABLES + PLAN
```repl
import re, json, statistics, html as _html
QUESTION = """<paste the question verbatim>"""
def clean(s): return " ".join(_html.unescape(re.sub(r"<[^>]+>", " ", s)).split())
def tables(doc):                      # every table of a document: heading above it, caption line, rows of cells
    out, spans = [], [(m.start(), m.end(), "html") for m in re.finditer(r"<table.*?</table>", doc, re.S)]
    if not spans:                     # text tables: paragraphs with >= 3 lines of >= 3 cells (pipes or 2+ spaces)
        at = 0
        for para in doc.split("\n\n"):
            ls = para.splitlines()
            if sum(ln.count("|") >= 2 or len(re.split(r"\s{2,}", ln.strip())) >= 3 for ln in ls) >= 3:
                spans.append((at, at + len(para), "text"))
            at += len(para) + 2
    for a, b, kind in spans:
        above = [ln.strip() for ln in re.sub(r"<[^>]+>", " ", doc[max(0, a - 1500): a]).splitlines() if ln.strip()]
        heads = re.findall(r"^#+\s*(.+)$", doc[max(0, a - 60000): a], re.M)
        seg = doc[a:b]
        if kind == "html":
            rows = [[clean(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
                    for r in re.findall(r"<tr[^>]*>(.*?)</tr>", seg, re.S)]
        else:
            rows = [[c.strip() for c in (ln.strip().strip("|").split("|") if "|" in ln else re.split(r"\s{2,}", ln.strip()))]
                    for ln in seg.splitlines() if ln.strip() and not re.fullmatch(r"[|\s:+-]+", ln.strip())]
        out.append({"heading": heads[-1].strip() if heads else "", "caption": above[-1][:160] if above else "",
                    "above": " / ".join(x[:160] for x in above[-3:]), "rows": rows})
    return out
STOP = set("jan feb mar apr may jun jul aug sep oct nov dec january february march april june july august september "
           "october november december monday tuesday wednesday thursday friday the of and a an at to in for".split())
def words(t):                         # a table's identity across documents: its caption's words and numbers, no dates
    return [w for w in re.findall(r"[a-z0-9]+", t["caption"].lower()) if w not in STOP and not re.fullmatch(r"\d{3,}", w)]
def key(t): return " ".join(words(t))
TAB = [tables(d) for d in context]
CAT = {}
for i, ts in enumerate(TAB):
    for t in ts:
        CAT.setdefault(key(t), []).append(i)
def show(t):                          # header rows + the first 2 rows with >= 3 numbers, value cells numbered
    isdata = [sum(bool(re.fullmatch(r"-?[\d,.]+%?", c)) for c in r[1:]) >= 3 for r in t["rows"]]
    first = isdata.index(True) if True in isdata else len(t["rows"])
    hdr = [[c for c in r if c] for r in t["rows"][:first]][:4]
    data = [r for r, d in zip(t["rows"], isdata) if d][:2]
    return {"heading": t["heading"], "caption": t["caption"], "text_above": t["above"], "header_rows": hdr,
            "first_rows": [{"label": r[0], "values": {str(k): v for k, v in enumerate(x for x in r[1:] if x)}} for r in data]}
SHOW = [dict(key=k, in_docs=len(v), **show(next(t for t in TAB[v[-1]] if key(t) == k))) for k, v in CAT.items() if len(v) > 1]
print(len(context), "documents |", sum(map(len, TAB)), "tables |", len(CAT), "distinct tables, in",
      sorted({len(v) for v in CAT.values()}), "documents")
PLAN = llm_query("""A corpus of periodic documents with tables. Below: every DISTINCT table (key, heading, caption, text above it, header
rows, and its first data rows with the value cells numbered 0, 1, 2 ... after the row label), and a question.
For every series the question needs, choose the table and the numbered value cell. Header rows can be misaligned with
the data cells (group cells spanning several columns, empty cells): decide by meaning, e.g. count the groups' categories
in order. Return ONLY JSON: {"series": [{"name": "<what the question needs>", "key": "<exact key>", "value_index": <int>,
"header_path": "<group / category>", "why": "<words that decide table and column>"}], "multiplier": <number the printed
values must be multiplied by to give full units, from the unit text>, "unit_text": "<exact unit text>",
"period_rows": "<how period rows are labelled, e.g. a year row then month names; marks on revised rows>",
"non_period_rows": ["<labels of rows that are not single periods>"], "rules": "<definitions the question states>"}
QUESTION: """ + QUESTION + "\nTABLES: " + json.dumps(SHOW)[:60000])
print(PLAN)
```

## Turn 2 -- LEDGER (Python reads every table; one batched sub-call reads tables Python cannot)
```repl
import re, json, statistics, html as _html
P = json.loads(re.search(r"\{.*\}", PLAN, re.S).group(0))
def unit(t):                          # the multiplier a table's own text states (heading, caption, lines above), else the plan's
    m = re.search(r"\b(thousand|million|billion)s?\b", (t["heading"] + " " + t["above"]).lower())
    return {"thousand": 1e3, "million": 1e6, "billion": 1e9}[m.group(1)] if m else P["multiplier"]
MON = {m[:3]: k + 1 for k, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}
NAMES = set(MON) | {"sept"} | set("january february march april june july august september october november december".split())
MARK = r"\b(?:r|p|revised|prelim\w*|preliminary|estimate)\b|[^a-z]"     # revision / footnote marks around a label
def period(label, year):              # -> ("year", y) | ("month", "YYYY-MM") | (None, None)
    t = label.strip().lower()
    m = re.match(r"^(\d{4})-(\d{2})\b(.*)$", t)
    if m and not re.sub(MARK, "", m.group(3)): return "month", f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"^(\d{4})\b(.*)$", t)
    if m and not re.sub(MARK, "", m.group(2)): return "year", int(m.group(1))
    m = re.match(r"^([a-z]+)\.?\s*(\d{4})?(.*)$", t)
    if m and m.group(1) in NAMES and (m.group(2) or year) and not re.sub(MARK, "", m.group(3)):
        return "month", f"{int(m.group(2) or year):04d}-{MON[m.group(1)[:3]]:02d}"
    return None, None
def num(c):
    c = c.replace(",", "").strip()
    return float(c) if re.fullmatch(r"-?\d+(?:\.\d+)?", c) else None
def read(t, idx):                     # {YYYY-MM: printed value} of one table's value cell idx
    out, year = {}, None
    for r in t["rows"]:
        if not r or not r[0]: continue
        kind, p = period(r[0], year)
        if kind == "year": year = p
        elif kind == "month":
            vals = [c for c in r[1:] if c]
            if len(vals) > idx and num(vals[idx]) is not None: out[p] = num(vals[idx])
    return out
def sim(k, t): a, b = set(k.split()), set(words(t)); return len(a & b) / len(a | b or {1})
def locate(k):                        # each document's table for key k: the same caption, else the one clearly closest
    out = {}
    for i, ts in enumerate(TAB):
        sc = sorted(((sim(k, t), j) for j, t in enumerate(ts)), reverse=True)
        if sc and (sc[0][0] == 1 or (sc[0][0] >= 0.8 and (len(sc) == 1 or sc[1][0] < sc[0][0]))):
            out[i] = ts[sc[0][1]]
    return out
DOCS = {s["name"]: {} for s in P["series"]}
for s in P["series"]:
    for i, t in locate(s["key"]).items():
        DOCS[s["name"]][i] = {p: v * unit(t) for p, v in read(t, s["value_index"]).items()}
CAPS = {s["key"]: next(x for x in SHOW if x["key"] == s["key"]) for s in P["series"]}
def excerpt(doc, k):                  # the text after a caption line matching key k (a table the parser could not read)
    for m in re.finditer(r"[^\n]+", doc):
        if sim(k, {"caption": m.group(0)}) >= 0.8:
            return doc[m.start(): m.start() + 6000]
    return None
MISS = [(s, i, x) for s in P["series"] for i in range(len(context)) if not DOCS[s["name"]].get(i)
        for x in [excerpt(context[i], s["key"])] if x]
if MISS:                              # one batched sub-call reads every unread table; its values keep their document
    ASK = [("Return ONLY JSON {\"YYYY-MM\": number as printed} with every month this excerpt gives for the column "
            f"{s['header_path']!r} of the table captioned {CAPS[s['key']]['caption']!r} (header rows as in other documents: "
            f"{json.dumps(CAPS[s['key']]['header_rows'])[:600]}). Months only, no totals or changes.\n\nEXCERPT:\n{x}") for s, i, x in MISS]
    for (s, i, x), r in zip(MISS, llm_query_batched(ASK)):
        try:
            got = json.loads(re.search(r"\{.*\}", r, re.S).group(0))
            m = re.search(r"\b(thousand|million|billion)s?\b", x[:300].lower())
            mult = {"thousand": 1e3, "million": 1e6, "billion": 1e9}[m.group(1)] if m else P["multiplier"]
            DOCS[s["name"]][i] = {p: float(str(v).replace(",", "")) * mult for p, v in got.items() if re.fullmatch(r"\d{4}-\d{2}", p)}
        except Exception:
            pass
    print("unread tables read by sub-call:", len(MISS), "->", sum(bool(DOCS[s["name"]].get(i)) for s, i, _ in MISS))
DATE = {i: max((p for s in DOCS for p in DOCS[s].get(i, {})), default="") for i in range(len(context))}
LEDGER = {s: {} for s in DOCS}
for i in sorted(range(len(context)), key=lambda i: DATE[i]):     # oldest -> newest: a later document overwrites
    for s in DOCS:
        for p, v in DOCS[s].get(i, {}).items(): LEDGER[s][p] = v
print("documents by data period:", DATE[min(DATE, key=DATE.get)], "..", max(DATE.values()), "| distinct:", len(set(DATE.values())))
for s, L in LEDGER.items():
    miss = [i for i in range(len(context)) if not DOCS[s].get(i)]
    print(s, "| docs read", len(context) - len(miss), "/", len(context), "missing", miss[:8], "| months", len(L),
          min(L, default=None), "..", max(L, default=None))
    print("  ", {p: round(v, 2) for p, v in sorted(L.items())})
```

## Turn 3 -- COMPUTE and SUBMIT (same block)
Compute from `LEDGER[<series name>]` (full units, keyed "YYYY-MM") restricted to the question's period (`S` below), print the deciding values, then submit: months as "YYYY-MM", lists via json.dumps, numbers with 2 decimals, percentages as "12.34%", a name as the question words it.
```repl
import json
START, END = "<YYYY-MM>", "<YYYY-MM>"           # the question's period
S = {n: {p: v for p, v in sorted(L.items()) if START <= p <= END} for n, L in LEDGER.items()}   # in-period months only
def prev(n, p):                                 # the previous month's value of series n, None if outside the period
    y, m = int(p[:4]), int(p[5:]); q = f"{y - (m == 1):04d}-{(m - 2) % 12 + 1:02d}"; return S[n].get(q)
# ... compute RESULT from S and prev(), print the deciding values; a name is returned exactly as the question words it ...
answer["content"] = json.dumps(RESULT) if isinstance(RESULT, list) else str(RESULT)
answer["ready"] = True
print(answer["content"])
```
