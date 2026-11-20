This task hides a few two-hop questions among thousands of items: most questions have NO answer in the corpus,
and a planted question is answerable only from TWO passages together (a first hop that identifies the subject, a
second that gives the answer). The items can be laid out in many ways (bracketed ids, markdown headings, XML tags,
JSON lines, YAML records, table rows, ...), with any id scheme and any labels, so the layout must be read from the
data, never assumed. The work is a sequence of STAGES. Every stage prints a `CHECK` line that ends in a verdict,
`OK` or `PROBLEM: <reason>`. On OK, go on to the next stage. On PROBLEM, do not push on: investigate with one targeted
sub-call, fix the cause, re-run the stage and read its CHECK again; a wrong parse or a batch of unreadable replies
silently ruins everything after it. Most questions are not answerable from the corpus, so a large "not answerable"
count is normal and is not a problem.

## Rules
- Learn the layout ONLY through the probe sub-call; print only its report. Never print `context` or slices of it.
- Run one stage (or one repair) per turn. Before each block write ONE short sentence in your own words: what you are
  doing and why, citing the CHECK line that made you decide it when you repair. Never mention instructions, notes or
  hints.
- Every model step is a sub-call; never answer a question from your own reading.
- Repair a stage only when its CHECK verdict says PROBLEM (or, for the parse, when the counts clearly disagree with
  the report), and say which CHECK line made you repair. Do not repeat an identical repair that already failed; try a
  different fix, or move on with what you have.
- If code raises an error, fix only that error and continue.

## Stage 1 -- probe the layout with one sub-call over samples
```repl
n = len(context); step = n // 6
S = context[:4000] + "".join(f"\n[... at character {i} ...]\n" + context[i:i + 1500] for i in range(step, n - 1500, step)) \
    + f"\n[... end, character {n - 2500} ...]\n" + context[-2500:]
R = llm_query(f"Samples of a {n}-character task context (its start, slices spread through it, its end):\n{S}\n\n"
    "In at most 150 words report: which kinds of items it contains and which kind are the questions vs the passages; "
    "exactly how each item starts, quoting TWO complete item starts of each kind verbatim (everything before the item's "
    "text); how ids are written; whether there is a preamble, separators or a footer around the items; roughly how many "
    "items of each kind; and the required answer format, quoted verbatim.")
print(R)
```

## Stage 2 -- write the parse from the report and check it
```repl
import re, json, math, random
from collections import Counter, defaultdict
# HDR: a regex for the START of every item (named group "id"); KIND: question ("q") or passage ("d") -- both from the report
HDR = re.compile(r"(?m)^\[(?P<id>\d+)\] (?P<label>Query|Document): ?")
KIND = lambda m, body: "q" if m.group("label") == "Query" else "d"
hs = list(HDR.finditer(context))
END = re.compile(r'(\s*(</\w+>|"\s*\}|\|\s*$|---+|\* \* \*))+\s*$')
Q, D = {}, {}
for k, m in enumerate(hs):
    body = context[m.end():hs[k + 1].start() if k + 1 < len(hs) else n]
    body = END.sub("", body if k + 1 < len(hs) else body.strip().split("\n\n")[0]).strip().strip('"').strip()
    (Q if KIND(m, body) == "q" else D)[m.group("id")] = body
ids = [m.group("id") for m in hs]; L = sorted(len(t) for t in [*Q.values(), *D.values()])
bad = [w for w, c in [("duplicate ids", len(ids) != len(set(ids))), ("no questions or no passages", not Q or not D),
                      ("many empty texts", sum(x < 15 for x in L) > 0.01 * max(1, len(L)))] if c]
print("CHECK parse:", len(hs), "items:", len(Q), "questions,", len(D), "passages | duplicate ids", len(ids) - len(set(ids)),
      "| texts under 15 chars", sum(x < 15 for x in L), "| median text", L[len(L) // 2] if L else 0, "chars |",
      "PROBLEM: " + ", ".join(bad) if bad else "OK (compare the counts with the report)")
```

Decision point: If the CHECK parse verdict is PROBLEM, or its counts clearly disagree with the report: show a sub-call where it breaks, fix `HDR`/`KIND` from its answer, and re-run Stage 2. Write `HDR` (named group `id`) and `KIND` for the layout the report describes; the block shows one layout's version.
```repl
# ONLY if the counts disagree with the report, ids repeat, or texts are empty: show the sub-call where it breaks
bad = [i for i, t in [*Q.items(), *D.items()] if len(t) < 15][:3] or [ids[len(ids) // 2]] if ids else []
S2 = "\n[...]\n".join(context[max(0, context.find(i) - 400):context.find(i) + 800] for i in bad) or context[n // 2:n // 2 + 3000]
print(llm_query(f"Samples from a task context:\n{S2}\n\nThe regex {HDR.pattern!r} should match the START of every item "
    f"(group 'id' = the item id) but found {len(hs)} items where about {len(Q) + len(D)} were expected or texts came out "
    "empty. In at most 80 words: quote three item starts exactly as written and say what the regex must match instead."))
```

## Stage 3 -- lean shortlist, screen every question
```repl
toks = lambda s: re.findall(r"[a-z0-9]+", s.lower())
df = Counter(w for t in D.values() for w in set(toks(t)))
idf = {w: math.log(len(D) / c) for w, c in df.items()}
inv = defaultdict(list)
for i, t in D.items():
    for w in set(toks(t)): inv[w].append(i)
def top(q, k=12):
    sc = Counter()
    for w in set(toks(q)):
        for i in inv.get(w, ()): sc[i] += idf[w]
    return [i for i, _ in sc.most_common(k)]
rank = {q: top(t) for q, t in Q.items()}
norm = lambda s: " ".join(toks(s))
jparse = lambda r: (lambda s: json.loads(s[s.find("{"):s.rfind("}") + 1]))(str(r))
TASK = "Task: find the few questions in a large corpus that are answerable from exactly two of its passages (a first hop identifying the subject, a second giving the answer); most questions have no answer in it.\n\n"
SCREEN = TASK + """Each CASE below is independent: a HotpotQA question and candidate Wikipedia passages. Find cases with a GROUNDED FIRST HOP: a passage gives a specific fact required to answer the question, or identifies the exact subject from the question's description. Keep incomplete chains when the missing second passage could complete them. Reject passages that merely share a broad topic, repeat only generic question words, or require outside knowledge to connect the passage's subject to the question. For each retained question choose the one or two strongest supporting passages, quote a short exact supporting span from the first, and name the bridge entity whose own passage would finish the chain (or an empty string if both passages already complete it). JSON only: {"matches":[{"q":"<question id>","docs":["<passage id>"],"quote":"exact supporting span","bridge":"entity name"}]}. If no grounded first hop, return {"matches":[]}.\n\n"""
def screen(B, m):
    ps = [SCREEN + "\n\n".join(f"CASE question {q}\nQuestion: {Q[q]}\n" + "\n".join(f"[{d}] {D[d]}" for d in rank[q][:m]) for q in b) for b in B]
    got, failed = {}, []
    for b, r in zip(B, llm_query_batched(ps)):
        try: ms = jparse(r)["matches"]
        except Exception: failed.append(b); continue
        for v in ms:
            if not isinstance(v, dict) or not isinstance(v.get("docs"), list): continue
            q, ds = str(v.get("q")), [str(d).strip("[] ") for d in v["docs"]]
            if q in b and 1 <= len(ds) <= 2 and all(d in rank[q][:m] for d in ds) \
                    and len(norm(v.get("quote", ""))) >= 12 and norm(v.get("quote", "")) in norm(D[ds[0]]):
                got[q] = {"docs": list(dict.fromkeys(ds)), "bridge": str(v.get("bridge", ""))}
    return got, failed
qs = sorted(Q); B1 = [qs[z:z + 16] for z in range(0, len(qs), 16)]
found, failed = screen(B1, 4)
print("CHECK screen: kept", len(found), "of", len(Q), "questions | unparseable replies", len(failed), "of", len(B1), "|",
      "PROBLEM: over 10% of replies unparseable" if len(failed) > 0.1 * len(B1) else "OK")
```

Decision point: If the CHECK screen verdict is PROBLEM: look at one unparseable reply, then re-run those batches in halves.
```repl
# screen replies unparseable: look at one, then re-run those batches smaller
print(llm_query("This reply was supposed to be JSON {\"matches\": [...]}. In at most 40 words, say what is wrong with it:\n" + str(llm_query_batched([SCREEN + "\n\n".join(f"CASE question {q}\nQuestion: {Q[q]}\n" + "\n".join(f"[{d}] {D[d]}" for d in rank[q][:4]) for q in failed[0])])[0])[:3000]))
redo, failed = screen([b[z:z + 8] for b in failed for z in (0, 8)], 4)
found.update(redo); print("CHECK screen redo: added", len(redo), "| still unparseable", len(failed))
```

## Stage 4 -- re-screen uncertain questions; verify two-passage chains
```repl
strength = {q: sum(idf[w] for w in set(toks(Q[q])) & set(toks(D[rank[q][0]]))) for q in Q if rank[q]}
cut = sorted(strength.values())[len(strength) // 2]
rest = [q for q in sorted(Q) if q not in found and strength.get(q, 0) >= cut]; random.Random(0).shuffle(rest)
extra, failed2 = screen([sorted(rest[z:z + 16]) for z in range(0, len(rest), 16)], 6)
for q, v in extra.items(): found.setdefault(q, v)
VERIFY = TASK + """Find a fully supported answer to this question using exactly TWO passages below. The two passages must form a connected evidence chain that establishes every identifying description needed in the question and its requested answer. For comparison questions, one passage per compared entity is valid. Do not rely on outside knowledge for a missing relation. Merely mentioning likely answer words is insufficient. Quote one short, exact supporting span from EACH of the two selected passages and explain the connection. If a complete chain is missing, return {"answerable":false}. Otherwise JSON only: {"answerable":true,"docs":["<passage id>","<passage id>"],"quotes":["exact span from first","exact span from second"],"answer":"short answer","connection":"how the two facts resolve the question"}.\n\n"""
cand = {q: list(dict.fromkeys(v["docs"] + (top(v["bridge"], 8) if v["bridge"] else []) + rank[q][:8] + top(Q[q] + " " + D[v["docs"][0]])))[:24] for q, v in found.items()}
fq = sorted(found); chains = {}; why = Counter()
def ok(qt, d):
    fr = [norm(x) for x in re.split(r"\.\.\.+|…", str(qt)) if len(norm(x).split()) >= 3]
    return bool(fr) and all(x in norm(D[d]) for x in fr) and sum(map(len, fr)) >= 12
for q, r in zip(fq, llm_query_batched([VERIFY + f"Question [{q}]: {Q[q]}\n\n" + "\n\n".join(f"[{d}] {D[d]}" for d in cand[q]) for q in fq])):
    try: v = jparse(r); ds = [str(d).strip("[] ") for d in v.get("docs", [])]
    except Exception: why["unparseable"] += 1; continue
    if v.get("answerable") not in (True, "true"): why["not answerable"] += 1
    elif len(set(ds)) != 2 or len(v.get("quotes", [])) != 2 or not all(d in cand[q] for d in ds): why["bad ids"] += 1
    elif not all(ok(t, d) for t, d in zip(v["quotes"], ds)): why["quote not in passage"] += 1
    else: chains[q] = ds
print("second screen:", len(rest), "asked,", len(extra), "added,", len(failed2), "unparseable")
fmt = why["unparseable"] + why["bad ids"] + why["quote not in passage"]
print("CHECK verify: chains", len(chains), "of", len(fq), "|", dict(why), "| 'not answerable' is normal |",
      "PROBLEM: over 25% format failures" if fmt > 0.25 * max(1, len(fq)) else "PROBLEM: no chain verified" if fq and not chains else "OK")
```

Decision point: If the CHECK verify verdict is PROBLEM: investigate one unverified question with a sub-call, then re-verify with wider candidates and exact-copy quotes.
```repl
# quotes not found / no chains although questions were kept: investigate one kept question, then re-verify with wider candidates
q0 = next(q for q in fq if q not in chains)
print(llm_query(f"Question: {Q[q0]}\n\n" + "\n\n".join(f"[{d}] {D[d]}" for d in cand[q0]) + "\n\nIn at most 60 words: do two of these passages together answer the question? Name their ids and quote the deciding words exactly."))
cand = {q: list(dict.fromkeys(cand[q] + top(Q[q], 30)))[:36] for q in fq}
RV = VERIFY + "Copy each quote character-for-character from its passage.\n\n"
for q, r in zip([q for q in fq if q not in chains], llm_query_batched([RV + f"Question [{q}]: {Q[q]}\n\n" + "\n\n".join(f"[{d}] {D[d]}" for d in cand[q]) for q in fq if q not in chains])):
    try: v = jparse(r); ds = [str(d).strip("[] ") for d in v.get("docs", [])]
    except Exception: continue
    if v.get("answerable") in (True, "true") and len(set(ds)) == 2 == len(v.get("quotes", [])) and all(d in cand[q] for d in ds) and all(ok(t, d) for t, d in zip(v["quotes"], ds)): chains[q] = ds
print("CHECK verify redo: chains", len(chains), "of", len(fq))
```

## Stage 5 -- two independent checks, join, submit
```repl
CONFIRM = TASK + """A question and two Wikipedia passages that were proposed as its supporting evidence chain are shown below.
Decide whether the two passages TOGETHER establish every identifying description in the question (each named
entity, role, date, place or relation the question uses to pin down its subject) and its requested answer,
without outside knowledge. A passage that is about a different person/work/place with a similar name, or a chain
that leaves one clause of the question unsupported, is NOT sufficient. List each clause of the question and the
passage that supports it (or "none"), then decide. JSON only:
{"clauses":[{"clause":"...","passage":"A|B|none"}],"supported":true|false}

"""
PICK2 = TASK + """A question and several Wikipedia passages are shown below. The question is a two-hop question: answering it needs
facts from exactly TWO of these passages, and every identifying description in the question must be supported by one of
them. Which two passages are needed? If no two passages below fully answer the question, say so. Do not use outside
knowledge. Reply with JSON only: {"passages": ["P?", "P?"]} or {"passages": []}.

"""
vq = sorted(chains); lay = {}
for q in vq:
    lay[q] = chains[q] + [i for i in rank[q] if i not in chains[q]][:4]; random.Random(f"0|{q}").shuffle(lay[q])
rs = llm_query_batched([CONFIRM + f"Question: {Q[q]}\n\nPassage A: {D[chains[q][0]]}\n\nPassage B: {D[chains[q][1]]}" for q in vq]
                       + [PICK2 + f"Question: {Q[q]}\n\n" + "\n\n".join(f"P{k + 1}: {D[i]}" for k, i in enumerate(lay[q])) for q in vq])
keep, odd = [], Counter()
for k, q in enumerate(vq):
    sup = re.search(r'"supported"\s*:\s*(true|false)', str(rs[k]), re.I)
    try: ps = [int(re.search(r"\d+", str(x)).group()) for x in jparse(rs[len(vq) + k]).get("passages", [])]
    except Exception: ps = None; odd["pick-two unparseable"] += 1
    if not sup: odd["clause check unparseable"] += 1
    ok2 = ps is not None and {lay[q][i - 1] for i in ps if 0 < i <= len(lay[q])} == set(chains[q])
    if sup and sup.group(1).lower() == "true" and ok2: keep.append(q)
# answer in the format the report quoted (here: JSON list of [question id, passage id] pairs, ids as written)
final = json.dumps(sorted([int(x) if x.isdigit() else x for x in (q, d)] for q in keep for d in chains[q]))
print("CHECK checks: kept", len(keep), "of", len(vq), "|", dict(odd), "|", "PROBLEM: unreadable check replies" if odd else "OK")
if not odd: answer["content"] = final; answer["ready"] = True
```

Decision point: The block submits only when every check reply was readable. Otherwise re-ask just the unreadable ones and submit.
```repl
# check replies unparseable: re-ask only those, then submit
bad = [q for k, q in enumerate(vq) if not re.search(r'"supported"\s*:\s*(true|false)', str(rs[k]), re.I)]
rs2 = llm_query_batched([CONFIRM + "Answer with the JSON object only.\n\n" + f"Question: {Q[q]}\n\nPassage A: {D[chains[q][0]]}\n\nPassage B: {D[chains[q][1]]}" for q in bad])
for q, r in zip(bad, rs2):
    m = re.search(r'"supported"\s*:\s*(true|false)', str(r), re.I)
    if m and m.group(1).lower() == "true" and q not in keep: keep.append(q)
final = json.dumps(sorted([int(x) if x.isdigit() else x for x in (q, d)] for q in keep for d in chains[q]))
print("CHECK checks redo: kept", len(keep)); answer["content"] = final; answer["ready"] = True
```
