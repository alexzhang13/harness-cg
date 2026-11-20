"""Reference RLM programs for the 9 CTC families, as scripted root turns (```repl blocks)."""

from __future__ import annotations

from synth_common.scripted import repl

HELPERS = r'''
import re, json, math, random
from collections import Counter, defaultdict

def jparse(reply, want=None):
    """The largest JSON value (list or object) in a reply -- the outermost one; None if none parses."""
    if reply is None:
        return None
    s = re.sub(r"```(?:json)?", "", str(reply))
    best, best_len = None, -1
    i = 0
    while i < len(s):
        if s[i] not in "[{":
            i += 1
            continue
        depth, instr, esc, end = 0, False, False, None
        for j in range(i, len(s)):
            c = s[j]
            if instr:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': instr = False
                continue
            if c == '"': instr = True
            elif c in "[{": depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is not None:
            try:
                v = json.loads(s[i:end + 1])
                if (want is None or isinstance(v, want)) and end + 1 - i >= best_len:
                    best, best_len = v, end + 1 - i
                    i = end + 1
                    continue
            except Exception:
                pass
        i += 1
    return best

import unicodedata
def toks(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.findall(r"[a-z0-9]+", s.lower())

def ids_in(reply):
    v = jparse(reply, list)
    out = []
    if isinstance(v, list):
        for x in v:
            if isinstance(x, (int, float)) or (isinstance(x, str) and x.strip().lstrip("[").rstrip("]").isdigit()):
                out.append(int(str(x).strip().lstrip("[").rstrip("]")))
    return out

print("helpers ready")
'''

PARSE_DOCS = r'''
# numbered documents "Document [n]: ..." ; the task statement follows the last one after a blank line
marks = list(re.finditer(r"(?m)^Document \[(\d+)\]: ?", context))
docs = {}
for k, m in enumerate(marks):
    end = marks[k + 1].start() if k + 1 < len(marks) else len(context)
    docs[int(m.group(1))] = context[m.end():end].strip()
last = max(docs)
tail = docs[last]
cut = tail.find("\n\nQuestion:")
if cut < 0:
    cut = tail.find("\n\nCan you find")
task_tail = tail[cut:].strip() if cut >= 0 else ""
if cut >= 0:
    docs[last] = tail[:cut].strip()
question = task_tail.split("Question:", 1)[1].strip() if "Question:" in task_tail else task_tail
ids = sorted(docs)
lens = sorted(len(t) for t in docs.values())
print("docs", len(docs), "ids 1..N:", ids == list(range(1, len(ids) + 1)), "len min/med/max", lens[0], lens[len(lens)//2], lens[-1])
print("question:", question[:300])
'''

BM25 = r'''
def build_bm25(texts):
    tf = {i: Counter(toks(t)) for i, t in texts.items()}
    df = Counter()
    for c in tf.values():
        df.update(c.keys())
    n = len(tf)
    avg = sum(sum(c.values()) for c in tf.values()) / max(1, n)
    L = {i: sum(c.values()) for i, c in tf.items()}
    def score(q, i, k1=1.2, b=0.75):
        c = tf[i]; s = 0.0
        for w in set(toks(q)):
            f = c.get(w)
            if f:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * L[i] / avg))
        return s
    return score
print("bm25 ready")
'''

NQ = [
    repl(HELPERS + PARSE_DOCS + BM25),
    repl(r'''
bm = build_bm25(docs)
ranked = sorted(docs, key=lambda i: -bm(question, i))
pool_ids = ranked[:400]                       # long rungs: read the top-400 by BM25, not all N
batches = [pool_ids[k:k + 20] for k in range(0, len(pool_ids), 20)]
def rel_prompt(b):
    body = "\n\n".join(f"Document [{i}]: {docs[i][:1500]}" for i in b)
    return (f"Question: {question}\n\n{body}\n\nWhich of these documents contain the answer to the question? "
            "Reply with a JSON list of document ids, e.g. [12, 40], or [] if none.")
# two passes with different batch compositions (BM25 order, then shuffled): a document missed once is rarely missed twice
shuf = pool_ids[:]
random.Random(7).shuffle(shuf)
batches += [shuf[k:k + 20] for k in range(0, len(shuf), 20)]
prompts = [rel_prompt(b) for b in batches]
print("prompts", len(prompts), "largest", max(map(len, prompts)))
replies = llm_query_batched(prompts)
cands, unparsed = Counter(), []
for b, r in zip(batches, replies):
    v = jparse(r, list)
    if v is None:
        unparsed.append(b)
        continue
    cands.update(i for i in ids_in(r) if i in b)
if unparsed:
    replies2 = llm_query_batched([rel_prompt(b) for b in unparsed])
    for b, r in zip(unparsed, replies2):
        cands.update(i for i in ids_in(r) if i in b)
cands = sorted(cands, key=lambda i: (-cands[i], ranked.index(i)))
print("candidates", len(cands), cands[:10], "unparsed", len(unparsed))
'''),
    repl(r'''
pick = None
if len(cands) == 1:
    pick = cands[0]
elif cands:
    top = cands[:8]
    body = "\n\n".join(f"Document [{i}]: {docs[i][:2500]}" for i in top)
    r = llm_query(f"Question: {question}\n\n{body}\n\nWhich ONE of these documents answers the question? "
                  'Reply with JSON only: {"id": <document id>, "answer": "<short answer>"}')
    v = jparse(r, dict)
    if isinstance(v, dict) and str(v.get("id", "")).strip("[] ").isdigit() and int(str(v["id"]).strip("[] ")) in top:
        pick = int(str(v["id"]).strip("[] "))
    else:
        pick = top[0]
else:
    pick = ranked[0]
final = f"Relevant Document: [{pick}]"
print(final)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]

HOTPOTQA = [
    repl(HELPERS + PARSE_DOCS + BM25),
    repl(r'''
bm = build_bm25(docs)
ranked = sorted(docs, key=lambda i: -bm(question, i))
pool_ids = ranked[:500]
batches = [pool_ids[k:k + 25] for k in range(0, len(pool_ids), 25)]
def rel_prompt(b):
    body = "\n\n".join(f"Document [{i}]: {docs[i][:1500]}" for i in b)
    return (f"Question (it may need two documents: one about the thing it names, one about the thing that leads "
            f"to the answer): {question}\n\n{body}\n\nWhich of these documents contain a fact needed to answer the "
            "question (either step)? Reply with a JSON list of document ids, or [] if none.")
prompts = [rel_prompt(b) for b in batches]
print("prompts", len(prompts), "largest", max(map(len, prompts)))
replies = llm_query_batched(prompts)
cands = []
for b, r in zip(batches, replies):
    cands += [i for i in ids_in(r) if i in b]
cands = sorted(set(cands), key=lambda i: ranked.index(i))
print("candidates", len(cands), cands[:12])
'''),
    repl(r'''
# answer with citations: the documents actually used are the relevant ones (both hops)
top = cands[:6]
q2 = " ".join(docs[i][:300] for i in top[:2])
hop2 = [i for i in sorted(docs, key=lambda i: -bm(question + " " + q2, i)) if i not in top][:10]
show = top + hop2
body = "\n\n".join(f"Document [{i}]: {docs[i][:1500]}" for i in show)
r = llm_query(f"Question: {question}\n\n{body}\n\nAnswer the question using ONLY facts stated in these documents. "
              'Reply with JSON only: {"answer": "<answer or NOT ANSWERABLE>", "used": [<ids of every document whose facts you used>]}')
v = jparse(r, dict) or {}
used = [int(str(x).strip("[] ")) for x in (v.get("used") or []) if str(x).strip("[] ").isdigit() and int(str(x).strip("[] ")) in docs]
print("answer:", str(v.get("answer"))[:80], "| used", used, "| candidates", cands[:6])
'''),
    repl(r'''
chosen = list(dict.fromkeys(used))[:2]
if len(chosen) == 1:
    # one hop missing: ask which other shown document gives the second fact
    others = [i for i in show if i != chosen[0]]
    body = "\n\n".join(f"Document [{i}]: {docs[i][:1500]}" for i in [chosen[0]] + others)
    r = llm_query(f"Question: {question}\n\n{body}\n\nDocument [{chosen[0]}] gives one fact needed for the answer. "
                  "Which ONE other document gives the other fact needed (the other hop)? Reply with JSON only: a list with that one document id, or [] if none.")
    extra = [i for i in ids_in(r) if i in docs and i != chosen[0]]
    chosen += extra[:1]
if not chosen:
    chosen = cands[:2] or ranked[:2]
final = "Relevant Documents: " + ", ".join(f"[{i}]" for i in chosen)
print(final)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]

PROGRAMS = {"nq": NQ, "hotpotqa": HOTPOTQA}

QD_PARSE = r'''
marks = list(re.finditer(r"(?m)^\[(\d+)\] (Query|Document): ?", context))
queries, qdocs = {}, {}
for k, m in enumerate(marks):
    end = marks[k + 1].start() if k + 1 < len(marks) else len(context)
    text = context[m.end():end].strip()
    if k + 1 == len(marks):
        text = text.split("\n\n")[0].strip()          # the repeated task statement follows the last item
    (queries if m.group(2) == "Query" else qdocs)[int(m.group(1))] = text
allids = sorted(list(queries) + list(qdocs))
print("queries", len(queries), "documents", len(qdocs), "ids 1..N:", allids == list(range(1, len(allids) + 1)))
'''

QD_BM25 = r'''
def build_index(texts):
    tf = {i: Counter(toks(t)) for i, t in texts.items()}
    post = defaultdict(list)
    for i, c in tf.items():
        for w, f in c.items():
            post[w].append((i, f))
    n = len(tf)
    L = {i: sum(c.values()) for i, c in tf.items()}
    avg = sum(L.values()) / max(1, n)
    idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in post.items()}
    def top(q, k=10, k1=1.2, b=0.75):
        sc = defaultdict(float)
        for w in set(toks(q)):
            for i, f in post.get(w, ()):
                sc[i] += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * L[i] / avg))
        return sorted(sc.items(), key=lambda x: -x[1])[:k]
    return top
dtop = build_index(qdocs)
short = {}
best = {}
for q, qt in queries.items():
    t1 = dtop(qt, 6)
    hop = dtop(qt + " " + qdocs[t1[0][0]][:1500], 10) if t1 else []
    ids_ = [i for i, _ in t1] + [i for i, _ in hop if i not in {j for j, _ in t1}][:4]
    short[q] = ids_
    best[q] = t1[0][1] / max(1, len(set(toks(qt)))) if t1 else 0.0
# long rungs: only the queries whose best document matches them strongly are worth a sub-call
order_q = sorted(queries, key=lambda q: -best[q])
check = order_q if len(queries) <= 250 else order_q[:max(250, int(0.35 * len(order_q)))]
print("shortlists built; checking", len(check), "of", len(queries), "queries")
'''

QDMATCH = [
    repl(HELPERS + QD_PARSE),
    repl(QD_BM25),
    repl(r'''
def ans_prompt(q, ids_):
    body = "\n\n".join(f"[{i}] {qdocs[i][:1200]}" for i in ids_)
    return (f"Question: {queries[q]}\n\nDocuments:\n{body}\n\nAnswer the question using ONLY facts stated in these "
            'documents (no outside knowledge). Reply with JSON only: {"answer": "<answer or NOT ANSWERABLE>", '
            '"used": [<ids of every document whose facts you used>]}')
prompts = [ans_prompt(q, short[q]) for q in check]
print("prompts", len(prompts), "largest", max(map(len, prompts)))
replies = llm_query_batched(prompts)
found = {}
bad = []
for q, r in zip(check, replies):
    v = jparse(r, dict)
    if v is None:
        bad.append(q)
        continue
    a = str(v.get("answer", "")).strip()
    used = [int(str(x).strip("[] ")) for x in (v.get("used") or []) if str(x).strip("[] ").isdigit()]
    used = [d for d in used if d in qdocs]
    if a and "NOT ANSWERABLE" not in a.upper() and used:
        found[q] = used
if bad:
    for q, r in zip(bad, llm_query_batched([ans_prompt(q, short[q]) for q in bad])):
        v = jparse(r, dict) or {}
        used = [int(str(x).strip("[] ")) for x in (v.get("used") or []) if str(x).strip("[] ").isdigit()]
        if used and "NOT ANSWERABLE" not in str(v.get("answer", "")).upper():
            found[q] = [d for d in used if d in qdocs]
print("answerable queries", len(found), "| docs per query", Counter(len(v) for v in found.values()), "| unparsed", len(bad))
'''),
    repl(r'''
# second, differently framed pass for the strongest unanswered queries: a fact chain, every fact quoted from a document
retry = [q for q in check[: max(40, len(check) // 2)] if q not in found]
def chain_prompt(q):
    body = "\n\n".join(f"[{i}] {qdocs[i][:1200]}" for i in short[q])
    return (f"Question: {queries[q]}\n\nDocuments:\n{body}\n\nAnswer the question as a chain of facts where EVERY fact "
            "is quoted from one of these documents (no outside knowledge): first the fact about the thing the question "
            'names, then the fact that leads to the answer. Reply with JSON only: {"chain": [{"fact": "<quote>", '
            '"doc": <id>}, ...], "answer": "<answer or NOT ANSWERABLE>"}')
rs = llm_query_batched([chain_prompt(q) for q in retry]) if retry else []
added = 0
for q, r in zip(retry, rs):
    v = jparse(r, dict) or {}
    if "NOT ANSWERABLE" in str(v.get("answer", "NOT ANSWERABLE")).upper():
        continue
    ds_ = []
    for step in v.get("chain") or []:
        d = str(step.get("doc", "")).strip("[] ") if isinstance(step, dict) else ""
        if d.isdigit() and int(d) in qdocs and int(d) not in ds_:
            if " ".join(toks(str(step.get("fact", ""))))[:40] in " ".join(toks(qdocs[int(d)])):
                ds_.append(int(d))
    if ds_:
        found[q] = ds_
        added += 1
print("retried", len(retry), "| newly answerable", added)
'''),
    repl(r'''
# a two-hop question needs two documents: complete the one-document queries, drop those that stay single (look-alikes)
one = [q for q, v in found.items() if len(v) == 1]
# what has to be looked up next: ask for the bridge entity, then retrieve by it
def ent_prompt(q):
    known = found[q][0]
    return (f"Question: {queries[q]}\n\n[{known}] {qdocs[known][:1500]}\n\nThis document gives one step of the answer. "
            "What is the name of the other entity whose own document is needed to finish answering the question? "
            'Reply with JSON only: {"entity": "<name>"}')
bridge = {}
if one:
    for q, r in zip(one, llm_query_batched([ent_prompt(q) for q in one])):
        bridge[q] = str((jparse(r, dict) or {}).get("entity") or "")
def hop2_prompt(q):
    known = found[q][0]
    by_ent = [i for i, _ in dtop(bridge.get(q, ""), 6)] if bridge.get(q) else []
    by_ctx = [i for i, _ in dtop(queries[q] + " " + qdocs[known], 14)]
    cand = [i for i in dict.fromkeys(by_ent + by_ctx) if i != known][:12]
    body = "\n\n".join(f"[{i}] {qdocs[i][:1200]}" for i in [known] + cand)
    return (f"Question: {queries[q]}\n\nDocuments:\n{body}\n\nDocument [{known}] gives one fact needed to answer the "
            "question. Which ONE other document gives the other fact needed (the second hop), using only what the "
            "documents state? Reply with JSON only: a list with that document id, or [] if none of them does.")
def hop2(qs, rev=False):
    out = {}
    if not qs:
        return out
    ps = []
    for q in qs:
        p = hop2_prompt(q)
        if rev:   # same question, candidates in the reverse order
            head, _, rest = p.partition("Documents:\n")
            blocks, _, tail = rest.rpartition("\n\nDocument [")
            parts = blocks.split("\n\n")
            p = head + "Documents:\n" + "\n\n".join(parts[:1] + parts[1:][::-1]) + "\n\nDocument [" + tail
        ps.append(p)
    for q, r in zip(qs, llm_query_batched(ps)):
        extra = [i for i in ids_in(r) if i in qdocs and i != found[q][0]]
        if extra:
            out[q] = extra[0]
    return out
got = hop2(one)
got.update(hop2([q for q in one if q not in got], rev=True))
for q, d in got.items():
    found[q] = [found[q][0], d]
kept = {q: v[:2] for q, v in found.items() if len(v) >= 2}
print("one-doc queries", len(one), "| completed", len(got), "| two-doc queries", len(kept))
'''),
    repl(r'''
# CHECK (independent of how the pair was found): a quoted two-step chain must use BOTH documents
def check_prompt(q):
    body = "\n\n".join(f"[{i}] {qdocs[i][:1500]}" for i in kept[q])
    return (f"Question: {queries[q]}\n\nDocuments:\n{body}\n\nAnswer the question as a chain of facts where EVERY fact "
            "is quoted verbatim from one of these documents (no outside knowledge). Reply with JSON only: "
            '{"chain": [{"fact": "<quote>", "doc": <id>}, ...], "answer": "<answer or NOT ANSWERABLE>"}')
qs_ = sorted(kept)
verified = {}
for q, r in zip(qs_, llm_query_batched([check_prompt(q) for q in qs_]) if qs_ else []):
    v = jparse(r, dict) or {}
    cited = set()
    if "NOT ANSWERABLE" not in str(v.get("answer", "NOT ANSWERABLE")).upper():
        for step in v.get("chain") or []:
            d = str(step.get("doc", "")).strip("[] ") if isinstance(step, dict) else ""
            if d.isdigit() and int(d) in kept[q] and " ".join(toks(str(step.get("fact", ""))))[:40] in " ".join(toks(qdocs[int(d)])):
                cited.add(int(d))
    verified[q] = cited
final_q = [q for q in qs_ if len(verified[q]) == 2 or (q not in got and len(verified[q]) >= 1)]
pairs = sorted([q, d] for q in final_q for d in kept[q])
final = json.dumps(pairs)
print("checked", len(qs_), "| both docs cited", sum(len(verified[q]) == 2 for q in qs_), "| kept", len(final_q), "| pairs", len(pairs))
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]
PROGRAMS["qdmatch_hpqa"] = QDMATCH

CONTRA_PARSE = r'''
marks = list(re.finditer(r"(?m)^Claim (\d+): ?", context))
claims = {}
for k, m in enumerate(marks):
    end = marks[k + 1].start() if k + 1 < len(marks) else len(context)
    claims[int(m.group(1))] = context[m.end():end].split("\n")[0].strip()
ids = sorted(claims)
print("claims", len(claims), "ids 1..N:", ids == list(range(1, len(ids) + 1)))
'''

CONTRA_CANDS = r'''
# candidates: each claim's nearest neighbour by idf-weighted word overlap (inverted index, no all-pairs loop)
STOPW = set("a an the of in on at to for from by with and or but is are was were be been this that these those it its as which who than then there their they them we our not no".split())
T = {i: set(w for w in toks(t) if w not in STOPW) for i, t in claims.items()}
df = Counter(w for s in T.values() for w in s)
N = len(T)
idf = {w: math.log(N / df[w]) for w in df}
post = defaultdict(list)
for i, s in T.items():
    for w in s:
        if df[w] <= max(50, N * 0.01):
            post[w].append(i)
def sim(a, b):
    x, y = T[a], T[b]
    uni = sum(idf[w] for w in x | y)
    return sum(idf[w] for w in x & y) / uni if uni else 0.0
cand = set()
for i, s in T.items():
    sc = Counter()
    for w in s:
        for j in post.get(w, ()):
            if j != i:
                sc[j] += idf[w]
    top = sorted((j for j, _ in sc.most_common(8)), key=lambda j: -sim(i, j))
    if top:
        cand.add(tuple(sorted((i, top[0]))))
        if len(top) > 1 and sim(i, top[1]) >= 0.9 * sim(i, top[0]):     # near-tie: keep both
            cand.add(tuple(sorted((i, top[1]))))
cand = sorted(cand)
print("candidate pairs", len(cand))
'''

CONTRA_LABEL = r'''
def label_prompts(pairs, swap, seed, per=10):
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    out = []
    for k in range(0, len(pairs), per):
        chunk = pairs[k:k + per]
        lines = []
        for j, (a, b) in enumerate(chunk, 1):
            x, y = (b, a) if swap else (a, b)
            lines.append(f"Pair {j}:\nA: {claims[x]}\nB: {claims[y]}")
        out.append((chunk, "\n\n".join(lines) + "\n\nFor EACH pair: is B a reworded version of A in which a finding, "
                    "number, direction or significance has been changed? Label REWRITE (the same statement with a fact "
                    "altered), SAME (the same statement with the same facts) or DIFFERENT (a different statement). "
                    f"Reply with JSON only: a list of {len(chunk)} labels in pair order."))
    return out
def run_pass(pairs, swap, seed):
    """{pair: True/False} for every pair whose prompt reply parsed; unparsed prompts are re-sent once."""
    res = {}
    todo = label_prompts(pairs, swap, seed)
    for attempt in range(2):
        if not todo:
            break
        replies = llm_query_batched([p for _, p in todo])
        again = []
        for (chunk, p), r in zip(todo, replies):
            v = jparse(r, list)
            if not isinstance(v, list) or len(v) != len(chunk):
                again.append((chunk, p))
                continue
            for pr, lab in zip(chunk, v):
                res[pr] = "REWRITE" in str(lab).upper()
        todo = again
    return res
print("labeller ready")
'''

CONTRADICTION = [
    repl(HELPERS + CONTRA_PARSE),
    repl(CONTRA_CANDS + CONTRA_LABEL),
    repl(r'''
# votes: every candidate once; its positives again with A/B swapped; a third vote only where the two disagree
p1 = run_pass(cand, False, 1)
pos1 = [p for p, v in p1.items() if v]
p2 = run_pass(pos1, True, 2)
split = [p for p in pos1 if not p2.get(p)]
p3 = run_pass(split, False, 3)
votes = {p: int(p1.get(p, False)) + int(p2.get(p, False)) + int(p3.get(p, False)) for p in pos1}
print("candidates labelled", len(p1), "of", len(cand), "| pass1 positives", len(pos1), "| 2-of-3 majority",
      sum(1 for v in votes.values() if v >= 2), "| vote histogram", sorted(Counter(votes.values()).items()))
'''),
    repl(r'''
# majority pairs; one claim belongs to at most one pair (more votes, then more similar)
kept = sorted((p for p, v in votes.items() if v >= 2), key=lambda p: (-votes[p], -sim(*p)))
used, final_pairs = set(), []
for a, b in kept:
    if a in used or b in used:
        continue
    used |= {a, b}
    final_pairs.append([a, b])
# CHECK with a different framing, single-pair prompts, both orders; a pair passes when either order says YES
def chk(a, b):
    return (f"Claim A: {claims[a]}\nClaim B: {claims[b]}\n\nCould B have been produced by taking A and paraphrasing it "
            "while altering one or more of its facts (a number, a direction, a significance, an effect)? "
            'Reply with JSON only: {"answer": "YES" or "NO"}')
qs_ = [chk(a, b) for a, b in final_pairs] + [chk(b, a) for a, b in final_pairs]
rs = llm_query_batched(qs_) if qs_ else []
n = len(final_pairs)
said_yes = lambda r: str((jparse(r, dict) or {}).get("answer", r)).strip().upper().startswith("YES")
checked = [p for k, p in enumerate(final_pairs) if said_yes(rs[k]) or said_yes(rs[n + k])]
final = json.dumps(sorted(checked))
print("majority pairs", len(final_pairs), "| passed check", len(checked), "| final", final)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]
PROGRAMS["contradiction_iid"] = CONTRADICTION

CONTRADICTION_SINGLE = [
    repl(HELPERS + CONTRA_PARSE),
    repl(CONTRA_CANDS),
    repl(r'''
def ask(a, b):
    return (f"Claim A: {claims[a]}\nClaim B: {claims[b]}\n\nCould B have been produced by taking A and paraphrasing it "
            "while altering one or more of its facts (a number, a direction, a significance, an effect)? "
            'Reply with JSON only: {"answer": "YES" or "NO"}')
said_yes = lambda r: str((jparse(r, dict) or {}).get("answer", r)).strip().upper().startswith("YES")
r1 = llm_query_batched([ask(a, b) for a, b in cand])
pos = [p for p, r in zip(cand, r1) if said_yes(r)]
r2 = llm_query_batched([ask(b, a) for a, b in pos]) if pos else []
both = [p for p, r in zip(pos, r2) if said_yes(r)]
print("candidates", len(cand), "| YES one way", len(pos), "| YES both ways", len(both))
'''),
    repl(r'''
# one claim belongs to at most one pair (the more similar pair wins)
used, final_pairs = set(), []
for a, b in sorted(both, key=lambda p: -sim(*p)):
    if a in used or b in used:
        continue
    used |= {a, b}
    final_pairs.append([a, b])
final = json.dumps(sorted(final_pairs))
print("pairs", len(final_pairs), "| final", final)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]

OUTLIER_TFIDF = r'''
STOPW = set("a an the of in on at to for from by with and or but is are was were be been this that these those it its as which who than then there their they them he she his her we our not no had has have also after before during into".split())
tfc = {i: Counter(w for w in toks(t) if w not in STOPW and len(w) > 1) for i, t in docs.items()}
N = len(tfc)
dfc = Counter(w for c in tfc.values() for w in c)
idf = {w: math.log(N / dfc[w]) for w in dfc if dfc[w] <= max(3, N * 0.05)}
vec = {}
for i, c in tfc.items():
    v = {w: (1 + math.log(f)) * idf[w] for w, f in c.items() if w in idf}
    nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    vec[i] = {w: x / nrm for w, x in v.items()}
post = defaultdict(list)
for i, v in vec.items():
    for w, x in v.items():
        post[w].append((i, x))
def neighbours(i, k=8):
    sc = Counter()
    for w, x in vec[i].items():
        for j, y in post[w]:
            if j != i:
                sc[j] += x * y
    return sc.most_common(k)
nb = {i: neighbours(i, 4) for i in docs}
par = {i: i for i in docs}
def root(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x
for i, lst in nb.items():                      # mutual nearest neighbours above a strict similarity: same article
    for j, s in lst:
        if s >= 0.16 and any(jj == i for jj, _ in nb[j]):
            par[root(i)] = root(j)
comp = defaultdict(list)
for i in docs:
    comp[root(i)].append(i)
comps = sorted(comp.values(), key=lambda c: (-len(c), c[0]))
print("components", len(comps), "| size histogram", sorted(Counter(min(len(c), 10) for c in comps).items()))
'''

OUTLIER = [
    repl(HELPERS + PARSE_DOCS),
    repl(OUTLIER_TFIDF),
    repl(r'''
# one title per component; components that share a distinctive word are batched together, so the sub-model sees
# several chunks of an article side by side
def key_word(c):
    best = Counter()
    for i in c:
        for w, x in vec[i].items():
            if dfc[w] >= 2:
                best[w] += x
    return best.most_common(1)[0][0] if best else "~"
order = sorted(range(len(comps)), key=lambda k: (key_word(comps[k]), comps[k][0]))
def title_prompts(ks, per=8):
    out = []
    for z in range(0, len(ks), per):
        grp = ks[z:z + per]
        body = [f"Group C{k}:\n" + "\n".join(f"[{i}] {docs[i][:700]}" for i in comps[k][:3]) for k in grp]
        out.append((grp, "\n\n".join(body) + "\n\nEach group's passages were cut from one Wikipedia article. For each "
                    "group, give the title of the article its passages come from. Reply with JSON only: "
                    '{"C<number>": "<article title>", ...} with one entry per group.'))
    return out
tp = title_prompts(order)
print("prompts", len(tp), "largest", max(len(p) for _, p in tp))
title = {}
for attempt in range(2):
    todo = [(g, p) for g, p in tp if any(k not in title for k in g)]
    if not todo:
        break
    for (grp, _), r in zip(todo, llm_query_batched([p for _, p in todo])):
        v = jparse(r, dict) or {}
        for k in grp:
            if v.get(f"C{k}"):
                title[k] = " ".join(str(v[f"C{k}"]).lower().split())
    tp = title_prompts([k for k in order if k not in title], 4)
owner = {}
for k, c in enumerate(comps):
    for i in c:
        owner[i] = title.get(k, f"untitled {k}")
def groups_now():
    g = defaultdict(list)
    for i, t in owner.items():
        g[t].append(i)
    return g
G = groups_now()
print("titled", len(title), "of", len(comps), "| articles", len(G), "| sizes", sorted(Counter(min(len(v), 10) for v in G.values()).items()))
'''),
    repl(r'''
# repair + isolation: every article with at most 3 passages is shown with its 6 nearest outside passages;
# passages named as the same article are merged into it
def near(group, k=6):
    sc = Counter()
    for i in group:
        for j, s in neighbours(i, 12):
            if j not in group:
                sc[j] += s
    return [j for j, _ in sc.most_common(k)]
def iso_prompt(group, others):
    g = "\n".join(f"G{n}: {docs[i][:700]}" for n, i in enumerate(group, 1))
    o = "\n".join(f"O{n}: {docs[j][:700]}" for n, j in enumerate(others, 1))
    return (f"Group passages (all cut from one Wikipedia article):\n{g}\n\nOther passages:\n{o}\n\nWhich of the Other "
            "passages were cut from the SAME article as the group? Reply with JSON only: a list of O-labels, [] if none.")
def iso_round(groups):
    jobs = [(t, sorted(g), near(g)) for t, g in groups]
    reps = llm_query_batched([iso_prompt(g, o) for _, g, o in jobs]) if jobs else []
    named = {}
    for (t, g, o), r in zip(jobs, reps):
        v = jparse(r, list)
        if v is None:
            continue
        named[t] = [o[int(str(x).strip().lstrip("Oo")) - 1] for x in v
                    if str(x).strip().lstrip("Oo").isdigit() and 1 <= int(str(x).strip().lstrip("Oo")) <= len(o)]
    return named
small = [(t, g) for t, g in G.items() if len(g) <= 3]
named = iso_round(small)
# a proposed merge is confirmed by a second, pairwise question before it is applied
props = [(t, j) for t, js in named.items() for j in js if owner[j] != t]
def same_prompt(t, j, k):
    g = sorted(G[t])
    a = g[k % len(g)]
    return (f"Passage 1: {docs[a][:1200]}\n\nPassage 2: {docs[j][:1200]}\n\nWere these two passages cut from the same "
            'Wikipedia article? Reply with JSON only: {"same": "YES" or "NO"}')
# two independent pairwise confirmations (different group members) per proposed merge
conf = llm_query_batched([same_prompt(t, j, 0) for t, j in props] + [same_prompt(t, j, 1) for t, j in props]) if props else []
yes = lambda r: str((jparse(r, dict) or {}).get("same", r)).strip().upper().startswith("YES")
conf = [yes(a) and yes(b) for a, b in zip(conf[:len(props)], conf[len(props):])]
# union-find over article titles: every confirmed (group, passage) proposal joins the two articles
uf = {}
def find(x):
    while uf.get(x, x) != x:
        uf[x] = uf.get(uf[x], uf[x])
        x = uf[x]
    return x
merged = 0
for (t, j), ok in zip(props, conf):
    if ok and find(t) != find(owner[j]):
        uf[find(owner[j])] = find(t)
        merged += 1
named = {find(t): [j for (tt, j), ok in zip(props, conf) if tt == t and ok] for t in named}
for i in owner:
    owner[i] = find(owner[i])
G = groups_now()
# purify the small merged groups (4-6 passages): a member that two other members say is from another article leaves
mid = [(t, sorted(g)) for t, g in G.items() if 4 <= len(g) <= 6]
pj = [(t, m, o) for t, g in mid for m in g for o in [x for x in g if x != m][:2]]
def pp(a, b):
    return (f"Passage 1: {docs[a][:1200]}\n\nPassage 2: {docs[b][:1200]}\n\nWere these two passages cut from the same "
            'Wikipedia article? Reply with JSON only: {"same": "YES" or "NO"}')
pr = llm_query_batched([pp(m, o) for _, m, o in pj]) if pj else []
noes = Counter((t, m) for (t, m, o), r in zip(pj, pr) if not yes(r))
for (t, m), k in noes.items():
    if k >= 2:
        owner[m] = f"split {m}"
G = groups_now()
isolated = [sorted(g) for t, g in G.items() if len(g) == 3 and t in named and not named[t]]
recheck = [(t, g) for t, g in G.items() if len(g) == 3 and (t not in named or named[t] or t not in dict(small))]
named2 = iso_round(recheck)
isolated += [sorted(G[t]) for t, js in named2.items() if not js]
print("small articles checked", len(small), "| proposed merges", len(props), "| confirmed", merged, "| 3-passage articles", sum(1 for g in G.values() if len(g) == 3),
      "| isolated", len(isolated), isolated[:3])
'''),
    repl(r'''
def coh(g):
    return sum(sum(x * vec[b].get(w, 0.0) for w, x in vec[a].items()) for a in g for b in g if a < b) / 3
ranked = sorted(isolated, key=coh, reverse=True)
if len(ranked) > 1:
    # several isolated 3-passage groups: search the corpus for a 4th member of each (a fragment of a bigger article
    # has one somewhere); the scan covers every passage on short corpora, the 300 nearest ones on long corpora
    def scan_pool(g):
        if len(docs) <= 1500:
            return [i for i in ids if i not in g]
        sc = Counter()
        for i in g:
            for j, s in neighbours(i, 300):
                sc[j] += s
        return [j for j, _ in sc.most_common(300) if j not in g]
    jobs = []
    for gi, g in enumerate(ranked[:8]):
        pool_ = scan_pool(g)
        for rnd in range(2):          # two scans with different batch compositions
            if rnd:
                pool_ = pool_[:]
                random.Random(gi).shuffle(pool_)
            for z in range(0, len(pool_), 40):
                jobs.append((gi, pool_[z:z + 40]))
    reps = llm_query_batched([iso_prompt(ranked[gi], o) for gi, o in jobs]) if jobs else []
    props = []
    for (gi, o), r in zip(jobs, reps):
        for x in jparse(r, list) or []:
            k = str(x).strip().lstrip("Oo")
            if k.isdigit() and 1 <= int(k) <= len(o):
                props.append((gi, o[int(k) - 1]))
    def pair_prompt(a, j):
        return (f"Passage 1: {docs[a][:1200]}\n\nPassage 2: {docs[j][:1200]}\n\nWere these two passages cut from the "
                'same Wikipedia article? Reply with JSON only: {"same": "YES" or "NO"}')
    props = sorted(set(props))
    cf = llm_query_batched([pair_prompt(ranked[gi][m], j) for m in range(3) for gi, j in props]) if props else []
    n_p = len(props)
    has4 = {gi for k, (gi, j) in enumerate(props) if sum(yes(cf[m * n_p + k]) for m in range(3)) >= 2}
    print("fourth-member search:", len(jobs), "prompts |", len(props), "proposals | candidates with a 4th member", sorted(has4))
    ranked = [g for gi, g in enumerate(ranked[:8]) if gi not in has4] or ranked
pick, art = [], "one article"
for g in ranked[:2]:
    body = "\n\n".join(f"[{i}] {docs[i][:1500]}" for i in g)
    v = jparse(llm_query(f"{body}\n\nWere these three passages cut from the same Wikipedia article? "
                         'Reply with JSON only: {"same": "YES" or "NO", "article": "<its title>"}'), dict) or {}
    if str(v.get("same", "")).upper().startswith("Y"):
        pick, art = g, str(v.get("article") or art)
        break
if not pick and ranked:
    pick = ranked[0]
final = (f"The passages come from many articles that each recur several times; the outliers all come from '{art}'.\n"
         "Outliers: " + ", ".join(f"[{i}]" for i in sorted(pick)))
print(final)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]
PROGRAMS["outlier"] = OUTLIER
PROGRAMS["outlier_fixedM"] = OUTLIER

REORDER_PARSE = r'''
marks = list(re.finditer(r"(?m)^Passage \[(\d+)\]: ?", context))
P = {}
for k, m in enumerate(marks):
    end = marks[k + 1].start() if k + 1 < len(marks) else len(context)
    P[int(m.group(1))] = context[m.end():end].split("\n\n")[0].strip()
ids = sorted(P)
N = len(P)
print("passages", N, "ids 1..N:", ids == list(range(1, N + 1)), "len min/max", min(map(len, P.values())), max(map(len, P.values())))
'''

REORDER_V1 = [
    repl(HELPERS + REORDER_PARSE),
    repl(r'''
# forward and backward link passes, packed: 8 passages in full per prompt + the opening (or closing) words of all
H = 300 if N <= 60 else 200
def link_prompts(direction, seed):
    order_ = ids[:]
    random.Random(seed).shuffle(order_)
    out = []
    for z in range(0, N, 8):
        tg = order_[z:z + 8]
        full = "\n\n".join(f"A{k}: {P[i]}" for k, i in enumerate(tg, 1))
        if direction == "next":
            heads = "\n".join(f"[{i}] {P[i][:H]}" for i in ids)
            q = ("For each A, which ONE numbered passage comes DIRECTLY AFTER it in the original book (it picks up the "
                 "same scene, speaker or argument where A stops)? The numbered list shows each passage's opening words.")
        else:
            heads = "\n".join(f"[{i}] {P[i][-H:]}" for i in ids)
            q = ("For each A, which ONE numbered passage comes DIRECTLY BEFORE it in the original book (A continues "
                 "where that passage stops)? The numbered list shows each passage's closing words.")
        out.append((tg, f"{full}\n\nNumbered passages:\n{heads}\n\n{q} Reply with JSON only: "
                    '{"A1": <number or null>, ...}; null if none of them fits.'))
    return out
def run_links(direction, seed):
    jobs = link_prompts(direction, seed)
    res = {}
    for (tg, _), r in zip(jobs, llm_query_batched([p for _, p in jobs])):
        v = jparse(r, dict) or {}
        for k, i in enumerate(tg, 1):
            x = v.get(f"A{k}")
            if x is not None and str(x).strip("[] ").isdigit() and int(str(x).strip("[] ")) in P and int(str(x).strip("[] ")) != i:
                res[i] = int(str(x).strip("[] "))
    return res
fwd = run_links("next", 1)
bwd = run_links("prev", 2)
mutual = {a: b for a, b in fwd.items() if bwd.get(b) == a}
print("forward picks", len(fwd), "| backward picks", len(bwd), "| mutual links", len(mutual))
'''),
    repl(r'''
# head-to-head for passages without a mutual successor: A and its few candidates, all in full
has_pred = set(mutual.values())
todo = [a for a in ids if a not in mutual]
def h2h(a):
    cands = [c for c in dict.fromkeys([fwd.get(a)] + [b for b, x in bwd.items() if x == a]) if c and c not in has_pred and c != a]
    return cands[:4]
jobs = [(a, h2h(a)) for a in todo]
jobs = [(a, c) for a, c in jobs if c]
def h2h_prompt(a, cands):
    body = "\n\n".join(f"Candidate {chr(66 + k)}: {P[c]}" for k, c in enumerate(cands))
    return (f"Passage A: {P[a]}\n\n{body}\n\nWhich candidate comes DIRECTLY AFTER passage A in the original book? "
            "Reply with JSON only: {\"next\": \"<candidate letter>\"} or {\"next\": null} if none of them does.")
reps = llm_query_batched([h2h_prompt(a, c) for a, c in jobs]) if jobs else []
links = dict(mutual)
for (a, c), r in zip(jobs, reps):
    v = jparse(r, dict) or {}
    x = str(v.get("next") or "").strip().upper()
    if len(x) == 1 and 0 <= ord(x) - 66 < len(c):
        b = c[ord(x) - 66]
        if b not in set(links.values()):
            # no cycles
            y, ok = b, True
            while y in links:
                y = links[y]
                if y == a:
                    ok = False
                    break
            if ok:
                links[a] = b
starts = [i for i in ids if i not in set(links.values())]
chains = []
for s0 in starts:
    ch, x = [s0], s0
    while x in links and links[x] not in ch:
        x = links[x]
        ch.append(x)
    chains.append(ch)
seen_ = {i for c in chains for i in c}
chains += [[i] for i in ids if i not in seen_]
print("links", len(links), "| pieces", len(chains), "| piece lengths", sorted(Counter(len(c) for c in chains).items()))
'''),
    repl(r'''
# order the pieces: every pair compared once (pieces shown by their first and last passage), ranked by wins
def piece_text(c):
    if len(c) == 1:
        return P[c[0]][:1200]
    return P[c[0]][:600] + " [...] " + P[c[-1]][-600:]
K = len(chains)
pairs = [(x, y) for x in range(K) for y in range(x + 1, K)]
if K > 24:   # too many pieces for every pair three times: each piece meets 12 random others once
    rr = random.Random(5)
    pairs = sorted({tuple(sorted((x, y))) for x in range(K) for y in rr.sample(range(K), min(12, K - 1)) if x != y})
def cmp_prompt(x, y):
    return (f"Piece 1: {piece_text(chains[x])}\n\nPiece 2: {piece_text(chains[y])}\n\nThese two pieces come from one "
            "stretch of a book. Which piece comes EARLIER in the book? Reply with JSON only: {\"earlier\": 1 or 2}")
# every pair three times (both presentation orders + a repeat in the first order with the pieces' ends swapped in)
jobs = [(x, y, cmp_prompt(x, y)) for x, y in pairs]
if K <= 24:
    jobs += [(y, x, cmp_prompt(y, x)) for x, y in pairs]
    jobs += [(x, y, cmp_prompt(x, y).replace("These two pieces", "The two pieces above")) for x, y in pairs]
reps = llm_query_batched([p for _, _, p in jobs]) if jobs else []
beats = Counter()
for (x, y, _), r in zip(jobs, reps):
    e = str((jparse(r, dict) or {}).get("earlier", "")).strip()
    if e == "1":
        beats[(x, y)] += 1
    elif e == "2":
        beats[(y, x)] += 1
wins = Counter()
for x, y in pairs:
    if beats[(x, y)] > beats[(y, x)]:
        wins[x] += 1
    elif beats[(y, x)] > beats[(x, y)]:
        wins[y] += 1
rank = sorted(range(K), key=lambda k: -wins[k])
order = [i for k in rank for i in chains[k]]
final = json.dumps(order)
print("pieces", K, "| comparisons", len(pairs), "| permutation ok:", sorted(order) == ids)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]

REORDER = [
    repl(HELPERS + REORDER_PARSE + r'''
# numbered markers (page or footnote numbers) increase through the book: the only reliable long-range order signal
MARKS = {"[Pg n]": r"\[Pg\.? ?(\d+)\]", "{n}": r"\{(\d+)\}", "[n]": r"\[(\d{1,4})\]"}
page, mark_name = {}, None
for name_, pat_ in MARKS.items():
    vals_ = {i: min(int(x) for x in re.findall(pat_, t)) for i, t in P.items() if re.findall(pat_, t)}
    if len(vals_) > len(page):
        page, mark_name = vals_, name_
if len(page) < 0.15 * N:
    page = {}
print("numbered markers:", mark_name, "on", len(page), "passages", "(used)" if page else "(not used)")
'''),
    repl(r'''
# lexical neighbours (Python) + a forward pass over every passage's opening words (one batch; a closing-words
# backward pass picks the true predecessor only 8% of the time at N=108, so it is not run)
STOPW = set("a an the of in on at to for from by with and or but is are was were be been this that these those it its as he she his her they them their i you we".split())
df = Counter(w for t in P.values() for w in set(toks(t)))
idf = {w: math.log((N + 1) / (df[w] + 0.5)) for w in df}
def vec(t):
    c = Counter(w for w in toks(t) if w not in STOPW)
    v = {w: (1 + math.log(n)) * idf[w] for w, n in c.items()}
    z = math.sqrt(sum(x * x for x in v.values())) or 1
    return {w: x / z for w, x in v.items()}
def sents(t):
    return re.split(r"(?<=[.!?\"\u201d\u2019])\s+", t.strip())
FULL = {i: vec(t) for i, t in P.items()}
TAIL = {i: vec(" ".join(sents(t)[-2:])) for i, t in P.items()}
HEAD = {i: vec(" ".join(sents(t)[:2])) for i, t in P.items()}
def cos(a, b):
    return sum(x * b.get(w, 0) for w, x in a.items())
lex = {a: sorted((b for b in ids if b != a), key=lambda b: -(cos(FULL[a], FULL[b]) + cos(TAIL[a], HEAD[b])))[:4] for a in ids}
H = 300 if N <= 60 else 200
def link_prompts(seed):
    order_ = ids[:]
    random.Random(seed).shuffle(order_)
    heads = "\n".join(f"[{i}] {P[i][:H]}" for i in ids)
    out = []
    for z in range(0, N, 8):
        tg = order_[z:z + 8]
        full = "\n\n".join(f"A{k}: {P[i]}" for k, i in enumerate(tg, 1))
        out.append((tg, f"{full}\n\nNumbered passages:\n{heads}\n\nFor each A, which ONE numbered passage comes "
                    "DIRECTLY AFTER it in the original book (it picks up the same scene, speaker or argument where A "
                    "stops)? The numbered list shows each passage's opening words. Reply with JSON only: "
                    '{"A1": <number or null>, ...}; null if none of them fits.'))
    return out
def pick(v, k):
    x = v.get(f"A{k}", v.get(str(k)))
    x = str(x).strip("[] ") if x is not None else ""
    return int(x) if x.isdigit() and int(x) in P else None
fwd = {}
todo = link_prompts(1)
for attempt in range(2):
    replies = llm_query_batched([p for _, p in todo])
    again = []
    for (tg, p), r in zip(todo, replies):
        v = jparse(r, dict)
        if not isinstance(v, dict):
            again.append((tg, p))
            continue
        for k, i in enumerate(tg, 1):
            j = pick(v, k)
            if j is not None and j != i:
                fwd[i] = j
    todo = again
    if not todo:
        break
print("forward picks", len(fwd), "of", N, "| prompts still unparsed", len(todo))
'''),
    repl(r'''
# head-to-head: each passage against its few candidates IN FULL (forward pick + 4 lexical neighbours), shuffled
# labels; then one predecessor per passage and no cycles
cand = {a: [x for x in dict.fromkeys([fwd.get(a)] + lex[a]) if x and x != a][:5] for a in ids}
shown = {a: random.Random(a).sample(c, len(c)) for a, c in cand.items()}
def h2h_prompt(a, c):
    body = "\n\n".join(f"Candidate {chr(66 + k)}: {P[x]}" for k, x in enumerate(c))
    return (f"Passage A: {P[a]}\n\n{body}\n\nWhich candidate comes DIRECTLY AFTER passage A in the original book (it "
            "continues the same scene, speaker, argument or list where A stops)? Reply with JSON only: "
            '{"next": "<candidate letter>"} or {"next": null} if none of them does.')
replies = llm_query_batched([h2h_prompt(a, shown[a]) for a in ids])
choice = {}
for a, r in zip(ids, replies):
    x = str((jparse(r, dict) or {}).get("next") or "").strip().upper()[:1]
    if x and 0 <= ord(x) - 66 < len(shown[a]):
        choice[a] = shown[a][ord(x) - 66]
claims = defaultdict(list)
for a, b in choice.items():
    claims[b].append(a)
links = {}
for b, As in claims.items():
    sup = As if len(As) == 1 else [a for a in As if fwd.get(a) == b]
    if len(sup) == 1:
        links[sup[0]] = b
for a in list(links):
    x, seen_ = links.get(a), {a}
    while x is not None and x not in seen_:
        seen_.add(x)
        x = links.get(x)
    if x == a:
        del links[a]
starts = [i for i in ids if i not in set(links.values())]
chains = []
for s0 in starts:
    ch = [s0]
    while ch[-1] in links and links[ch[-1]] not in ch:
        ch.append(links[ch[-1]])
    chains.append(ch)
left_ = [i for i in ids if i not in {x for c in chains for x in c}]
chains += [[i] for i in left_]
print("h2h replies parsed", len(choice), "of", N, "| links", len(links), "| pieces", len(chains),
      "| piece lengths", sorted(Counter(len(c) for c in chains).items()))
'''),
    repl(r'''
# piece order: pieces with a numbered marker by their smallest marker; the rest after them, longest first (far-apart
# order judgements are at chance, and pieces placed last shift no position before them)
def first_mark(c):
    v = [page[i] for i in c if i in page]
    return min(v) if v else None
marked = sorted([c for c in chains if first_mark(c) is not None], key=first_mark)
unmarked = sorted([c for c in chains if first_mark(c) is None], key=len, reverse=True)
order = [i for c in marked + unmarked for i in c]
final = json.dumps(order)
print("pieces with a marker", len(marked), "| without", len(unmarked), "| permutation of 1..N:", sorted(order) == ids)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]

REORDER_JIGSAW = [
    repl(HELPERS + REORDER_PARSE + r"""
MARKS = {"[Pg n]": r"\[Pg\.? ?(\d+)\]", "{n}": r"\{(\d+)\}", "[n]": r"\[(\d{1,4})\]"}
page, mark_name = {}, None
for name_, pat_ in MARKS.items():
    vals_ = {i: min(int(x) for x in re.findall(pat_, t)) for i, t in P.items() if re.findall(pat_, t)}
    if len(vals_) > len(page):
        page, mark_name = vals_, name_
dup = [v for v, c in Counter(page.values()).items() if c > 1]
page = {i: v for i, v in page.items() if v not in dup}
if len(page) < 0.15 * N:
    page = {}
print("numbered markers:", mark_name, "on", len(page), "passages", "(used)" if page else "(not used)")
"""),
    repl(r'''
def full_prompt(a, cands):
    lab = {b: f"C{j + 1}" for j, b in enumerate(cands)}
    body = "\n\n".join(f"{lab[b]}: {P[b]}" for b in cands)
    return lab, (f"Passage A:\n{P[a]}\n\nCandidate passages:\n{body}\n\nThe passages were cut, in order, from one "
                 "stretch of a book. For EACH candidate, rate from 0 to 10 how likely it is to come IMMEDIATELY after "
                 'passage A (10 = it clearly continues A). Reply with JSON only: {"C1": <score>, ...}')
def edge_prompt(a, cands, words=70):
    lab = {b: f"C{j + 1}" for j, b in enumerate(cands)}
    tail = " ".join(P[a].split()[-words:])
    body = "\n\n".join(f"{lab[b]}: {' '.join(P[b].split()[:words])} ..." for b in cands)
    return lab, (f"End of passage A:\n... {tail}\n\nOpenings of candidate passages:\n{body}\n\nThe passages were cut, "
                 "in order, from one stretch of a book. For EACH candidate, rate from 0 to 10 how likely it is that its "
                 "opening comes IMMEDIATELY after the end of passage A (10 = it clearly continues A). Reply with JSON only: "
                 '{"C1": <score>, ...}')
def run_scores(jobs):
    """jobs: (a, labels, prompt) -> [(a, {b: score})], one entry per scored prompt; unparsed prompts asked once more."""
    out = []
    for attempt in range(2):
        replies = llm_query_batched([p for _, _, p in jobs])
        again = []
        for (a, lab, p), r in zip(jobs, replies):
            inv = {l: b for b, l in lab.items()}
            sc = {}
            for l, x in (jparse(r, dict) or {}).items():
                try:
                    if l in inv:
                        sc[inv[l]] = float(x)
                except (TypeError, ValueError):
                    pass
            if sc:
                out.append((a, sc))
            else:
                again.append((a, lab, p))
        jobs = again
        if not jobs:
            break
    return out
# stage 1: every passage against every other, 12 full candidates per prompt (~9 * N sub-calls, in two blocks)
jobs1 = []
for a in ids:
    others = [b for b in ids if b != a]
    random.Random(a).shuffle(others)
    for z in range(0, len(others), 12):
        lab, p = full_prompt(a, others[z:z + 12])
        jobs1.append((a, lab, p))
half = len(jobs1) // 2
S1 = run_scores(jobs1[:half])
print("stage 1, first half:", len(S1), "of", half, "prompts scored")
'''),
    repl(r'''
S1 += run_scores(jobs1[half:])
shortlist = defaultdict(list)
for a, sc in S1:
    shortlist[a].append(max(sc, key=lambda b: sc[b]))
print("stage 1:", len(S1), "of", len(jobs1), "prompts scored | shortlist sizes",
      sorted(Counter(len(set(v)) for v in shortlist.values()).items()))
# stage 2: each passage against its stage-1 winners in one prompt, in full and as closing -> opening words
jobs2 = []
for a in ids:
    c = list(dict.fromkeys(shortlist.get(a, [])))
    random.Random(a + 7).shuffle(c)
    if c:
        for fn in (full_prompt, edge_prompt):
            lab, p = fn(a, c)
            jobs2.append((a, lab, p))
S2 = run_scores(jobs2)
C = defaultdict(float)
for a, sc in S1:
    for b, x in sc.items():
        C[(a, b)] += 0.3 * x
for a, sc in S2:
    for b, x in sc.items():
        C[(a, b)] += x
print("stage 2:", len(S2), "of", len(jobs2), "prompts scored | scored pairs", len(C))
'''),
    repl(r'''
# markers: drop the backbone if confident links contradict it (markers that restart, e.g. per-chapter footnotes)
if page:
    conf = [(a, b) for (a, b), x in C.items() if x >= 18 and a in page and b in page]
    bad = sum(1 for a, b in conf if page[b] < page[a])
    print("confident links between marked passages:", len(conf), "| against the marker order:", bad)
    if conf and bad > 0.3 * len(conf):
        page = {}
rank_of = {x: k for k, x in enumerate(sorted(page, key=lambda i: page[i]))}
succ, pred = {}, {}
chain_of = {i: i for i in ids}
members = {i: [i] for i in ids}
first_m = {i: rank_of.get(i) for i in ids}
last_m = dict(first_m)
for (a, b), x in sorted(C.items(), key=lambda kv: -kv[1]):
    if a in succ or b in pred:
        continue
    ca, cb = chain_of[a], chain_of[b]
    if ca == cb or not (last_m[ca] is None or first_m[cb] is None or first_m[cb] == last_m[ca] + 1):
        continue
    succ[a], pred[b] = b, a
    for y in members[cb]:
        chain_of[y] = ca
    members[ca] += members.pop(cb)
    if first_m[ca] is None:
        first_m[ca] = first_m[cb]
    if last_m[cb] is not None:
        last_m[ca] = last_m[cb]
chains = []
for h in [i for i in ids if i not in pred]:
    c = [h]
    while c[-1] in succ:
        c.append(succ[c[-1]])
    chains.append(c)
marked = sorted([c for c in chains if any(y in rank_of for y in c)],
                key=lambda c: min(rank_of[y] for y in c if y in rank_of))
free = sorted([c for c in chains if not any(y in rank_of for y in c)], key=len, reverse=True)
order = [y for c in marked for y in c]
for c in free:      # a chain without a marker goes where its two ends fit best
    best, at = -1e9, len(order)
    for k in range(len(order) + 1):
        x = (C.get((order[k - 1], c[0]), 0) if k > 0 else 0) + (C.get((c[-1], order[k]), 0) if k < len(order) else 0)
        if x > best:
            best, at = x, k
    order[at:at] = c
final = json.dumps(order)
print("links", len(succ), "| chains", len(chains), "(with a marker", len(marked), ") | permutation of 1..N:",
      sorted(order) == ids)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]


def _code(block: str) -> str:
    return block.split("```repl\n", 1)[1].rsplit("\n```", 1)[0]


def _branch(jig: str, v2: str) -> str:
    import textwrap
    return repl("if JIG:\n" + textwrap.indent(_code(jig), "    ") + "\nelse:\n" + textwrap.indent(_code(v2), "    "))


REORDER_HYBRID = [
    repl(_code(REORDER_JIGSAW[0]) + '\nJIG = bool(page)\nprint("program:", "jigsaw + marker backbone" if JIG else '
         '"v2 (no markers)")'),
    _branch(REORDER_JIGSAW[1], REORDER[1]),
    _branch(REORDER_JIGSAW[2], REORDER[2]),
    _branch(REORDER_JIGSAW[3], REORDER[3]),
    REORDER_JIGSAW[4],
]
PROGRAMS["reorder"] = REORDER_HYBRID

GROUP_PARSE = r'''
marks = list(re.finditer(r"(?m)^Document \[(\d+)\]", context))
papers = {}
for k, m in enumerate(marks):
    end = marks[k + 1].start() if k + 1 < len(marks) else len(context)
    body = context[m.end():end]
    title = ""
    if body.startswith("(Title:"):
        depth = 0
        for j, ch in enumerate(body):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0:
                title, body = body[len("(Title:"):j].strip(), body[j + 1:]
                break
    papers[int(m.group(1))] = (title, body.split("\n")[0].strip())
# the task statement after the last paper: "Cluster these N papers into K groups." or "Group these documents into K
# groups." (the first pattern alone missed 14 of 25 train rows and fell back to N // 8)
mk = re.search(r"into (\d+) (?:groups|categories|clusters)", context[marks[-1].end():] if marks else context)
NP, K = len(papers), (int(mk.group(1)) if mk else max(2, len(papers) // 8))
ids = sorted(papers)
print("papers", len(papers), "ids 1..N:", ids == list(range(1, len(ids) + 1)), "| N", NP, "K", K, "N/K", round(NP / K, 2))
'''

GROUPING = [
    repl(HELPERS + GROUP_PARSE),
    repl(r'''
# free-form topic per paper, granularity set by N/K (many papers per group -> the field; few -> the specific topic)
gran = ("the department-level field (e.g. 'Medicine', 'Computer science')" if NP / K >= 8 else
        "the subfield" if NP / K >= 3 else "the specific research topic (the narrow concept it is about)")
def lab_prompts(pids, per=12):
    out = []
    for z in range(0, len(pids), per):
        chunk = pids[z:z + per]
        body = "\n\n".join(f"[{i}] {papers[i][0]}. {papers[i][1][:500]}" for i in chunk)
        out.append((chunk, f"{body}\n\nFor each paper give {gran} in 1-4 words, judged by its method and framing. "
                    'Reply with JSON only: {"<id>": "<label>", ...}'))
    return out
lab = {}
for attempt in range(2):
    todo = [i for i in ids if i not in lab]
    if not todo:
        break
    jobs = lab_prompts(todo, 12 if attempt == 0 else 6)
    for (chunk, _), r in zip(jobs, llm_query_batched([p for _, p in jobs])):
        v = jparse(r, dict) or {}
        for i in chunk:
            if v.get(str(i)):
                lab[i] = " ".join(str(v[str(i)]).split())
print("labelled", len(lab), "of", len(ids), "| distinct labels", len(set(lab.values())))
'''),
    repl(r'''
# merge labels that name the same thing; the merged names are the categories
names = sorted(Counter(lab.values()).items(), key=lambda x: -x[1])
listing = "\n".join(f'"{n}" ({c})' for n, c in names)
r = llm_query(f"These are topic labels given to {NP} papers, with counts:\n{listing}\n\nThe papers must be clustered into "
              f"exactly {K} groups. Merge labels that are synonyms or rewordings of the same topic, then (only if more "
              f"than {K} remain) merge the closest topics until {K} categories remain; never merge two different fields "
              "just to reduce the count when they are already distinct. Reply with JSON only: an object mapping every "
              'label to its category name, e.g. {"Computing": "Computer science", "Machine learning": "Computer science"}.')
ren = jparse(r, dict) or {}
canon = {i: str(ren.get(l, l)) for i, l in lab.items()}
cats = sorted(set(canon.values()))
print("categories after merging", len(cats), "| K asked", K)
'''),
    repl(r'''
# file every paper against the fixed category list, twice (shuffled), a third time where the two disagree
def file_prompts(pids, seed, per=12):
    pids = pids[:]
    random.Random(seed).shuffle(pids)
    menu = "\n".join(f"{k}. {c}" for k, c in enumerate(cats, 1))
    out = []
    for z in range(0, len(pids), per):
        chunk = pids[z:z + per]
        body = "\n\n".join(f"[{i}] {papers[i][0]}. {papers[i][1][:500]}" for i in chunk)
        out.append((chunk, f"Categories:\n{menu}\n\nPapers:\n{body}\n\nFile each paper under exactly one of the "
                    'categories above. Reply with JSON only: {"<paper id>": "<category name>", ...}'))
    return out
def file_pass(pids, seed):
    res = {}
    jobs = file_prompts(pids, seed)
    for (chunk, _), r in zip(jobs, llm_query_batched([p for _, p in jobs])):
        v = jparse(r, dict) or {}
        for i in chunk:
            c = v.get(str(i))
            if c in cats:
                res[i] = c
    return res
f1 = file_pass(ids, 1)
f2 = file_pass(ids, 2)
split = [i for i in ids if f1.get(i) != f2.get(i)]
f3 = file_pass(split, 3) if split else {}
final_cat = {}
for i in ids:
    votes = Counter(x for x in (f1.get(i), f2.get(i), f3.get(i), canon.get(i)) if x)
    final_cat[i] = votes.most_common(1)[0][0] if votes else canon.get(i, "misc")
groups = defaultdict(list)
for i in ids:
    groups[final_cat[i]].append(i)
final = json.dumps({"groups": [{"doc_ids": sorted(g)} for g in groups.values()]})
print("filing disagreements", len(split), "| groups", len(groups), "| sizes", sorted(Counter(len(g) for g in groups.values()).items())[:10])
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]
PROGRAMS["grouping"] = GROUPING

GROUP_LEX = r'''
STOP = set("the and for with that this from are was were have has been into their its our which these those using used "
           "based study studies results result paper also than between both such can may more most other new".split())
docs = {i: (papers[i][0] + " ") * 3 + papers[i][1] for i in ids}     # title x3: best of the offline sweep (.434)
words = {i: [t for t in re.findall(r"[a-z][a-z0-9\-]{2,}", docs[i].lower()) if t not in STOP] for i in ids}
df = Counter(t for i in ids for t in set(words[i]))
idf = {t: math.log(NP / c) for t, c in df.items() if c < 0.5 * NP}
V = {}
for i in ids:
    v = {t: (1 + math.log(c)) * idf[t] for t, c in Counter(words[i]).items() if t in idf}
    z = math.sqrt(sum(x * x for x in v.values())) or 1.0
    V[i] = {t: x / z for t, x in v.items()}
def cos(a, b):
    a, b = (a, b) if len(a) < len(b) else (b, a)
    return sum(x * b.get(t, 0.0) for t, x in a.items())
sim = {(a, b): cos(V[a], V[b]) for a in ids for b in ids if a < b}
link = lambda x, y: sim[(x, y) if x < y else (y, x)]
cl = {i: [i] for i in ids}
while len(cl) > K:
    a, b = max(((x, y) for x in cl for y in cl if x < y), key=lambda p: link(*p))
    na, nb = len(cl[a]), len(cl[b])
    for c in cl:
        if c not in (a, b):
            sim[(a, c) if a < c else (c, a)] = (link(a, c) * na + link(b, c) * nb) / (na + nb)
    cl[a] += cl.pop(b)
groups = [sorted(g) for g in cl.values()]
print("lexical clusters", len(groups), "| K", K, "| sizes", sorted(Counter(len(g) for g in groups).items())[:8])
'''
GROUPING_PY = [repl(HELPERS + GROUP_PARSE), repl(GROUP_LEX),
               repl('final = json.dumps({"groups": [{"doc_ids": g} for g in groups]})\n'
                    'answer["content"] = final\nanswer["ready"] = True')]
GROUPING_CLUSTER = [repl(HELPERS + GROUP_PARSE), repl(GROUP_LEX), repl(r'''
# name every cluster from up to 5 of its titles, 8 clusters per prompt
jobs = [list(range(z, min(z + 8, len(groups)))) for z in range(0, len(groups), 8)]
prompts = ["Each numbered cluster below lists titles of papers grouped together. Give each cluster a short name (2-5 "
           "words) for the research topic its papers share. Reply with JSON only: {\"<cluster number>\": \"<name>\", ...}\n\n"
           + "\n".join(f"Cluster {c}: " + "; ".join(papers[i][0] for i in groups[c][:5]) for c in job) for job in jobs]
names = {}
for job, r in zip(jobs, llm_query_batched(prompts)):
    v = jparse(r, dict) or {}
    for c in job:
        names[c] = " ".join(str(v.get(str(c)) or f"Topic {c}").split())
seen = Counter()
for c in sorted(names):
    seen[names[c]] += 1
    if seen[names[c]] > 1:
        names[c] = f"{names[c]} ({seen[names[c]]})"
cats = [names[c] for c in range(len(groups))]
print("named", len(names), "clusters | distinct names", len(set(cats)))
'''), repl(r'''
# re-file every paper against the cluster names; a paper keeps its cluster unless the filing names another one
menu = "\n".join(f"- {c}" for c in cats)
order = ids[:]
random.Random(1).shuffle(order)
chunks = [order[z:z + 12] for z in range(0, len(order), 12)]
prompts = [f"Categories:\n{menu}\n\nPapers:\n" + "\n\n".join(f"[{i}] {papers[i][0]}. {papers[i][1][:500]}" for i in ch)
           + '\n\nFile each paper under exactly one of the categories above. Reply with JSON only: {"<paper id>": "<category>"}'
           for ch in chunks]
home = {i: cats[c] for c, g in enumerate(groups) for i in g}
moved = 0
for ch, r in zip(chunks, llm_query_batched(prompts)):
    v = jparse(r, dict) or {}
    for i in ch:
        c = v.get(str(i))
        if c in cats and c != home[i]:
            home[i] = c
            moved += 1
by = defaultdict(list)
for i in ids:
    by[home[i]].append(i)
final = json.dumps({"groups": [{"doc_ids": sorted(g)} for g in by.values()]})
print("re-filed", len(ids), "| moved", moved, "| groups", len(by), "| K", K)
'''), repl('answer["content"] = final\nanswer["ready"] = True')]

TG_PARSE = r'''
marks = list(re.finditer(r"(?m)^Passage (\d+): ?", context))
P = {}
for k, m in enumerate(marks):
    end = marks[k + 1].start() if k + 1 < len(marks) else len(context)
    P[int(m.group(1))] = context[m.end():end].split("\n\n")[0].strip()
crit = re.search(r"Each passage has a value: (.+?)\. Find every group of (\d+) passages whose values add up to (\d+)", context)
what, G, T = crit.group(1), int(crit.group(2)), int(crit.group(3))
word = re.search(r'the word "([^"]+)"', what)
ids = sorted(P)
print("passages", len(P), "ids 1..N:", ids == list(range(1, len(ids) + 1)), "| value:", what, "| group size", G, "| target", T)
'''

TEXTGROUPS = [
    repl(HELPERS + TG_PARSE),
    repl(r'''
# what counts: a connector word is counted exactly in Python; a word class is decided by the sub-model, word by word
def words_of(t):
    return [w.strip(".,;:!?\"'()").lower() for w in t.split()]
if word:
    target_word = word.group(1).lower()
    is_feat = lambda w: w == target_word
    print("counting the word", repr(target_word))
else:
    cls = what.replace("the number of ", "").strip()
    vocab = sorted({w for t in P.values() for w in words_of(t) if w})
    example = {}
    for t in P.values():                      # one real sentence per word, so the word is judged in context
        for sent in re.split(r"(?<=\.)\s+", t):
            for w in words_of(sent):
                example.setdefault(w, sent)
    def classify(order_seed):
        vs = vocab[:]
        random.Random(order_seed).shuffle(vs)
        chunks = [vs[z:z + 40] for z in range(0, len(vs), 40)]
        prompts = ["\n".join(f"- {w}  (in: \"{example[w]}\")" for w in c) +
                   f"\n\nWhich of these words are {cls} in the sentence shown next to them? "
                   f"Reply with JSON only: a list of the words that are {cls}." for c in chunks]
        got = set()
        for c, r in zip(chunks, llm_query_batched(prompts)):
            got |= {str(x).lower() for x in (jparse(r, list) or []) if str(x).lower() in c}
        return got
    a1, a2 = classify(1), classify(2)
    disputed = sorted(a1 ^ a2)
    feat = a1 & a2
    if disputed:
        qs_ = [f"In the sentence \"{example[w]}\", is the word '{w}' one of the {cls}? Reply YES or NO." for w in disputed]
        for w, r in zip(disputed, llm_query_batched(qs_)):
            if str(r).strip().upper().startswith("YES"):
                feat.add(w)
    is_feat = lambda w: w in feat
    print("vocabulary", len(vocab), "| words classed as", cls, len(feat), "| disputed between two passes", len(disputed))
val = {i: sum(1 for w in words_of(t) if is_feat(w)) for i, t in P.items()}
print("values min/max", min(val.values()), max(val.values()))
'''),
    repl(r'''
# every group of G passages whose values sum to the target (exhaustive, in Python)
from itertools import combinations
by_val = defaultdict(list)
for i, v in val.items():
    by_val[v].append(i)
found = set()
if G == 3:
    for a, b in combinations(ids, 2):
        need = T - val[a] - val[b]
        for c in by_val.get(need, []):
            if c > b:
                found.add((a, b, c))
else:
    for comb in combinations(ids, G):
        if sum(val[i] for i in comb) == T:
            found.add(comb)
groups_ = sorted(sorted(g) for g in found)
print("groups found", len(groups_), groups_[:5])
'''),
    repl(r'''
# CHECK: one sub-call per group member sentence sample -- does the sub-model's count of the first sentences agree?
def first_sents(i, k=1):
    return " ".join(re.split(r"(?<=\.)\s+", P[i])[:k])
ms = sorted({i for g in groups_ for i in g})[:9]
qs_ = [f"Text: {first_sents(i)}\n\nHow many {('times does the word ' + repr(word.group(1)) + ' appear') if word else what.replace('the number of ', '')} "
       "are in this text? Reply with the number only." for i in ms]
agree = 0
for i, r in zip(ms, llm_query_batched(qs_) if qs_ else []):
    m = re.search(r"\d+", str(r))
    py = sum(1 for w in words_of(first_sents(i)) if is_feat(w))
    agree += bool(m and int(m.group()) == py)
final = json.dumps(groups_)
print("checked", len(ms), "sentence samples | agree", agree)
'''),
    repl('answer["content"] = final\nanswer["ready"] = True'),
]
PROGRAMS["textgroups"] = TEXTGROUPS


def _submit(expr: str) -> str:
    return repl(f'answer["content"] = {expr}\nanswer["ready"] = True')


GUESS = {
    "nq": [_submit('"Relevant Document: [1]"')],
    "hotpotqa": [_submit('"Relevant Documents: [1], [2]"')],
    "qdmatch_hpqa": [_submit('"[[1, 50], [2, 60], [3, 70]]"')],
    "contradiction_iid": [_submit('"[[1, 2], [3, 4], [5, 6]]"')],
    "outlier": [_submit('"Most passages share a topic.\\nOutliers: [1], [2], [3]"')],
    "outlier_fixedM": [_submit('"Most passages share a topic.\\nOutliers: [1], [2], [3]"')],
    "grouping": [repl('import re\nn = len(re.findall(r"(?m)^Document \\[\\d+\\]", context))\n'
                      'import json\nanswer["content"] = json.dumps({"groups": [{"doc_ids": list(range(1, n + 1))}]})\n'
                      'answer["ready"] = True')],
    "reorder": [repl('import re, json\nn = len(re.findall(r"(?m)^Passage \\[\\d+\\]", context))\n'
                     'answer["content"] = json.dumps(list(range(1, n + 1)))\nanswer["ready"] = True')],
    "textgroups": [_submit('"[]"')],
}

_WHOLE = r'''
task = context[-1500:]
r = llm_query(context + "\n\nAnswer the task above. Reply with the answer only, in exactly the requested format.")
print(str(r)[:500])
answer["content"] = str(r).strip()
answer["ready"] = True
'''
WHOLE = {f: [repl(_WHOLE)] for f in GUESS}

PY_ONLY = {
    "nq": [repl(HELPERS + PARSE_DOCS + BM25 + r'''
bm = build_bm25(docs)
best = max(docs, key=lambda i: bm(question, i))
answer["content"] = f"Relevant Document: [{best}]"
answer["ready"] = True
''')],
    "hotpotqa": [repl(HELPERS + PARSE_DOCS + BM25 + r'''
bm = build_bm25(docs)
top = sorted(docs, key=lambda i: -bm(question, i))[:2]
answer["content"] = "Relevant Documents: " + ", ".join(f"[{i}]" for i in top)
answer["ready"] = True
''')],
    "qdmatch_hpqa": [repl(HELPERS + QD_PARSE + QD_BM25 + r'''
pairs = sorted([q, d] for q in order_q[:3] for d in short[q][:2])
answer["content"] = json.dumps(pairs)
answer["ready"] = True
''')],
    "contradiction_iid": [repl(HELPERS + CONTRA_PARSE + CONTRA_CANDS + r'''
top = sorted(cand, key=lambda p: -sim(*p))[:3]
answer["content"] = json.dumps([list(p) for p in top])
answer["ready"] = True
''')],
    "outlier": [repl(HELPERS + PARSE_DOCS + OUTLIER_TFIDF + r'''
def coh(g):
    return sum(sum(x * vec[b].get(w, 0.0) for w, x in vec[a].items()) for a in g for b in g if a < b)
threes = [c for c in comps if len(c) == 3] or [c[:3] for c in comps if len(c) > 3]
pick = max(threes, key=coh) if threes else ids[:3]
answer["content"] = "Most passages come from several articles.\nOutliers: " + ", ".join(f"[{i}]" for i in sorted(pick))
answer["ready"] = True
''')],
    "reorder": [repl(HELPERS + REORDER_PARSE + r'''
STOPW = set("a an the of in on at to for from by with and or but is are was were be been this that these those it its as".split())
vec = {i: Counter(w for w in toks(t) if w not in STOPW) for i, t in P.items()}
def cos(a, b):
    x, y = vec[a], vec[b]
    num = sum(v * y.get(w, 0) for w, v in x.items())
    return num / (math.sqrt(sum(v * v for v in x.values())) * math.sqrt(sum(v * v for v in y.values())) or 1)
order, left = [ids[0]], set(ids[1:])
while left:
    nxt = max(left, key=lambda j: cos(order[-1], j))
    order.append(nxt)
    left.remove(nxt)
answer["content"] = json.dumps(order)
answer["ready"] = True
''')],
}
PY_ONLY["outlier_fixedM"] = PY_ONLY["outlier"]

SHORTCUT_LLM = {
    "textgroups": [repl(HELPERS + TG_PARSE + r'''
from itertools import combinations
batches = [ids[z:z + 10] for z in range(0, len(ids), 10)]
prompts = ["\n\n".join(f"Passage {i}: {P[i]}" for i in b) + f"\n\nFor each passage, give {what}. "
           'Reply with JSON only: {"<passage number>": <value>, ...}' for b in batches]
val = {}
for b, r in zip(batches, llm_query_batched(prompts)):
    v = jparse(r, dict) or {}
    for i in b:
        try:
            val[i] = int(v.get(str(i)))
        except Exception:
            val[i] = 0
groups_ = sorted(sorted(g) for g in combinations(ids, G) if sum(val[i] for i in g) == T) if G == 3 else []
answer["content"] = json.dumps(groups_[:50])
answer["ready"] = True
''')],
    "grouping": GROUPING[:2] + [repl(r'''
groups = defaultdict(list)
for i in ids:
    groups[lab.get(i, "?")].append(i)
answer["content"] = json.dumps({"groups": [{"doc_ids": sorted(g)} for g in groups.values()]})
answer["ready"] = True
''')],
    "contradiction_iid": [repl(HELPERS + CONTRA_PARSE + r'''
chunks = [ids[z:z + 120] for z in range(0, len(ids), 120)]
prompts = ["\n".join(f"Claim {i}: {claims[i]}" for i in c) + "\n\nList every pair of claims above that contradict each "
           "other. Reply with JSON only: a list of [id, id] pairs." for c in chunks]
pairs = []
for r in llm_query_batched(prompts):
    for p in jparse(r, list) or []:
        if isinstance(p, list) and len(p) == 2 and all(str(x).isdigit() for x in p):
            pairs.append(sorted(int(x) for x in p))
answer["content"] = json.dumps(pairs)
answer["ready"] = True
''')],
    "outlier": [repl(HELPERS + PARSE_DOCS + r'''
chunks = [ids[z:z + 40] for z in range(0, len(ids), 40)]
prompts = ["\n\n".join(f"Document [{i}]: {docs[i]}" for i in c) + "\n\nWhich of these passages are about a different "
           "topic than the rest? Reply with JSON only: a list of document ids." for c in chunks]
out = []
for r in llm_query_batched(prompts):
    out += [i for i in ids_in(r) if i in docs]
answer["content"] = "The outliers are about different topics.\nOutliers: " + ", ".join(f"[{i}]" for i in out)
answer["ready"] = True
''')],
    "reorder": REORDER_V1[:1] + [repl(REORDER_V1[1].split("```repl\n", 1)[1].rsplit("\n```", 1)[0].replace(
        'bwd = run_links("prev", 2)', 'bwd = {b: a for a, b in fwd.items()}')),
        repl(r'''
order, used_ = [], set()
starts = [i for i in ids if i not in set(fwd.values())] or ids[:1]
for s0 in starts + ids:
    x = s0
    while x is not None and x not in used_:
        order.append(x)
        used_.add(x)
        x = fwd.get(x)
answer["content"] = json.dumps(order)
answer["ready"] = True
''')],
}
SHORTCUT_LLM["outlier_fixedM"] = SHORTCUT_LLM["outlier"]


STRATEGY = {
    "nq": "Parse the numbered documents and the question. Rank documents by BM25 against the question in Python and "
          "keep the top few hundred (all of them on short corpora). Ask the sub-LM, in batches of ~20 documents, which "
          "documents contain the answer (JSON list of ids); run a second pass with differently composed batches and "
          "keep every document flagged in either. Show the candidates together and ask which ONE answers the question "
          "(JSON with its id and the short answer). Submit `Relevant Document: [id]`.",
    "hotpotqa": "Parse the documents and the two-hop question. BM25-rank in Python, ask batches of ~25 documents which "
                "contain a fact needed for either hop. Then show the best candidates together with the documents that "
                "best match question + top candidates (the second hop), and ask the sub-LM to answer using only those "
                "documents and cite every document it used. If only one hop is cited, ask which other shown document "
                "gives the other fact. Submit `Relevant Documents: [a], [b]`.",
    "qdmatch_hpqa": "Parse queries and documents. For every query build a BM25 shortlist in Python (its top documents "
                    "plus a second-hop expansion from the top document's text); on long corpora only check the "
                    "queries whose best document matches strongly. Per query, ask the sub-LM to answer from its "
                    "shortlist only and cite the documents used; retry unanswered strong queries with a quoted "
                    "fact-chain prompt. A two-hop query needs two documents: for one-document answers ask for the "
                    "bridge entity, retrieve by it, and ask which other document gives the second fact. CHECK every "
                    "kept query with a quoted two-step chain that must cite both documents (quotes verified in Python). "
                    "Never assume how many queries are relevant. Submit a JSON list of [query, document] pairs.",
    "contradiction_iid": "Parse the claims. In Python, pair every claim with its nearest neighbour by idf-weighted "
                         "word overlap (inverted index, no all-pairs loop): the altered copy of a claim is almost "
                         "always its nearest neighbour. Label candidate pairs in batches of ~10 with the framing 'is B "
                         "a reworded version of A with a fact changed: REWRITE / SAME / DIFFERENT' (JSON list in pair "
                         "order); relabel the positives with A/B swapped and a third time where the two disagree; keep "
                         "2-of-3 majorities, one pair per claim. CHECK each kept pair with single-pair prompts in both "
                         "orders ('could B be A paraphrased with a fact altered? YES/NO'). Submit a JSON list of pairs.",
    "outlier": "The corpus is chunks of many Wikipedia articles; the outliers are the chunks of the one article that "
               "has fewer chunks than every other. In Python, build tf-idf vectors and join mutual nearest neighbours "
               "above a strict similarity into components (pure but fragmented). Ask the sub-LM for the source "
               "article title of each component (components sharing a distinctive word batched together) and merge "
               "components with the same title. For every article with <=3 chunks, show it with its nearest outside "
               "chunks and ask which come from the same article; merge only pairwise-confirmed answers. The candidates "
               "are isolated 3-chunk articles; if several, search the corpus for a 4th member of each and drop those "
               "that have one. CHECK the survivor ('same article? YES/NO'). Submit one sentence then "
               "`Outliers: [a], [b], [c]`.",
    "grouping": "Parse papers (title + abstract) and K from the question. Choose the label granularity from N/K "
                "(field when many papers per group, subfield in between, specific topic when most groups are single "
                "papers). Label every paper in batches of ~12 (JSON id -> label). Ask one sub-call to merge synonym "
                "labels into exactly K categories. File every paper against the fixed category list twice (shuffled "
                "batches) and a third time where the two disagree; majority vote. Submit "
                '{"groups": [{"doc_ids": [...]}, ...]} with every paper exactly once.',
    "reorder": "Parse the passages and look for numbered markers ([Pg n], {n} or footnote [n]): they increase through "
               "the book and are the only reliable long-range order signal; drop numbers that occur twice. WITH markers "
               "on >= 15% of the passages: score every passage against every other -- prompts of passage A plus 12 "
               "candidates in full, a 0-10 score per candidate for 'comes IMMEDIATELY after A', JSON -- in batched "
               "calls; the scores are relative to each prompt (one candidate always gets 8-10), so keep each prompt's "
               "winner and score A against all its winners in ONE prompt, in full and as A's closing words against "
               "the candidates' opening words (the predecessor scores high in full, not there). Take links best-first "
               "on the summed scores: one successor and one predecessor per passage, no cycles, and chains keep the "
               "marker order (a chain ending at marker n continues only into the chain holding the next marker); "
               "chains without a marker go where their two ends score best. WITHOUT markers: one forward pass (8 "
               "passages in full plus the opening words of all: which comes directly after each?), then each passage "
               "against its forward pick and its 4 most similar passages, all in full (null allowed); keep a link "
               "when one passage alone claims that successor, drop cycles, pieces longest first. Never ask the sub-LM "
               "to order many passages or which of two far-apart pieces comes first: those replies are at chance. "
               "Submit the JSON list of all ids, each once.",
    "textgroups": "Parse the passages, the value definition, group size and target. A connector-word count is exact in "
                  "Python. For a word class, list the distinct words, ask the sub-LM to classify them (each word shown "
                  "in one real sentence) twice with different batching and re-ask the words the passes disagree on; "
                  "count per passage in Python. Never ask the sub-LM for passage totals. Enumerate every group whose "
                  "values sum to the target in Python, spot-check a few sentence counts, submit the JSON list of groups.",
}
STRATEGY["outlier_fixedM"] = STRATEGY["outlier"]


def program_text(fam: str) -> str:
    """The reference program as one code listing (the "program" hint tier)."""
    parts = []
    for k, t in enumerate(PROGRAMS[fam], 1):
        code = t.split("```repl\n", 1)[1].rsplit("\n```", 1)[0]
        if "def jparse(" in code:
            code = "# helpers: jparse(reply) -> outermost JSON value; toks(s) -> word tokens; ids_in(reply) -> int ids\n" + \
                code.split('print("helpers ready")', 1)[-1].lstrip("\n")
        parts.append(f"# ---- turn {k}\n{code.strip()}")
    return "\n\n".join(parts)


from .xabsence_programs import XAB_VARIANTS, XABSENCE_PARAPHRASE  # noqa: E402

PROGRAMS["xabsence_paraphrase"] = XABSENCE_PARAPHRASE
