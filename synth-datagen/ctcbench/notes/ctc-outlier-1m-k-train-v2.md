# How to solve this task (stages with checks; usually 5-9 turns)

The input is a long collection of short items. Before writing any parser, find out what the items are and how they are
marked, with ONE probe sub-call over a few slices of `context`. The task asks for the items whose topic differs from
the rest: in collections like this every topic is a run of consecutive windows cut from one source document, and the
odd topic is the source that contributes the FEWEST windows. Lexical similarity alone cannot tell which source a window
came from at this size, so sub-calls name each window's source, groups are grown with sub-calls until no outside window
joins them, and each surviving group's full length is estimated by having a sub-call reassemble its source.

## Rules
- Format: every piece of code goes inside a ```repl fenced block. Do not wrap code in <tool_call>, <function=...> or
  <parameter=...> tags: such a turn executes nothing and costs a turn.
- Do not print raw slices of `context` or the text of items yourself (no `print(context[...])`, no `print(docs[i][:400])`).
  Read the collection only through the probe report, the condensed reports the blocks print, and `peek(ids, question)`:
  when you want to see what particular items say (a candidate group, a doubtful member), ask `peek` a
  specific question about them and print its short answer.
- Run the stage blocks below in order, each exactly as written (in stage 2 you only fill in `ITEM`). Do not write your
  own clustering, frequency tables or thresholds: the naming is expected to look noisy (thousands of distinct titles);
  stages 3-5 are what turn it into groups.
- Stages are not turns, and checks decide what happens next: if the stage-2 parse check fails, re-probe with the
  failing sample and fix `ITEM`; if the stage-4 leader check fails, run stage 5 (more than once if needed). Spend the
  extra sub-calls on the doubtful result rather than submitting it.
- Describe groups by their evidence (closed, reassembled source length, title agreement); do not guess how many
  windows the odd topic has.
- Before each block, write one or two sentences in your own words about what the step does and why, using what the
  previous output showed. Never refer to a strategy, plan, method, turn number or "block", and never say you are
  following or running something.
- If a block raises, fix that block and re-run it; do not change the method.

## Stage 1: probe the layout

```repl
import re, json, collections, itertools
n = len(context)
slices = "\n\n".join(f"--- slice at char {s:,} of {n:,} ---\n{context[s:s + 1500]}"
                     for s in [0, n // 4, n // 2, 3 * n // 4, max(0, n - 1500)])
report = llm_query(f"""Below are a few slices taken from different places in a long input (start, middle, end). Describe it so that someone can parse it with code:
1. What are the items, and how is each item delimited and numbered? Quote the exact marker / id format verbatim, and give a Python regex with a named group (?P<id>...) for the item id that matches the start of every item.
2. Is there any preamble, task statement or footer that is NOT an item? Quote how it starts.
3. Roughly how many items are there (use any count stated in the input, or the highest id visible)?
4. What does the task ask for, and in what answer format?
Answer in at most 10 short lines.

{slices}""")
print(report)
```

## Stage 2: parse, check the parse, name every item's source

Write `ITEM` from the probe report (the marker that starts each item, with the id in a named group `id`), then run the
checks. If a check fails, print the failing sample, ask the probe again with that sample, and fix `ITEM` before naming.
Naming every item is needed because no cheap filter narrows several thousand items: lexically isolated items are not
the odd topic at this size. A window often covers only a sub-topic of its source, so each item gets three candidate
source titles, and two items count as possibly from the same source when any of their titles agree.

```repl
ITEM = re.compile(r"PUT THE ITEM-START REGEX FROM THE REPORT HERE", re.M)
marks = list(ITEM.finditer(context))
items = []
for k, m in enumerate(marks):
    body = context[m.end():marks[k + 1].start() if k + 1 < len(marks) else len(context)]
    items.append((m.group("id"), body.strip().split("\n\n")[0].strip()))
ids = [i for i, _ in items]
problems = []
if len(set(ids)) != len(ids): problems.append(f"{len(ids) - len(set(ids))} duplicate ids")
if any(len(t.split()) < 20 for _, t in items): problems.append(f"{sum(len(t.split()) < 20 for _, t in items)} items under 20 words")
print(len(items), "items | first", ids[:3], "last", ids[-3:], "| problems:", problems or "none")
docs = {k: t for k, (_, t) in enumerate(items, 1)}       # work with positions 1..N; map back to the ids at the end
NAME = """Below are {n} numbered passages. Each is a window of text cut out of a longer English Wikipedia article. A window often discusses a sub-topic, a person, a place or an event that is only PART of the article it was cut from.

For EACH passage, give the 3 most likely titles of the Wikipedia ARTICLE it was cut from, broadest plausible source first (the article that would contain this text as a section), separated by " | ".

{passages}

Answer with exactly one line per passage, in the same order, in the form
<number>: <title> | <title> | <title>
and nothing else."""
def name_all(chunks):
    got = {}
    replies = llm_query_batched([NAME.format(n=len(c), passages="\n\n".join(f"{k}. {docs[i]}" for k, i in enumerate(c, 1))) for c in chunks])
    for c, r in zip(chunks, replies):
        for m in re.finditer(r"(?m)^\s*\[?(\d+)\]?\s*[:.)\-]\s*(.+?)\s*$", r):
            k = int(m.group(1))
            if 1 <= k <= len(c): got.setdefault(c[k - 1], [t.strip() for t in m.group(2).split("|") if t.strip()])
    return got
P = sorted(docs)
titles = name_all([P[s:s + 25] for s in range(0, len(P), 25)])
miss = [i for i in P if i not in titles]
titles.update(name_all([miss[s:s + 10] for s in range(0, len(miss), 10)]))
names = {i: ts[0] for i, ts in titles.items() if ts}
print(len(titles), "of", len(P), "items named; most common first titles:", collections.Counter(names.values()).most_common(5))
PEEK = """Below are numbered windows of text, each cut out of an English Wikipedia article.

{windows}

{question}
Answer in at most {lines} short lines and cite window numbers; do not copy the windows' text."""
def peek(ids, question, lines=8):   # look at specific items through a sub-call instead of printing their text
    return llm_query(PEEK.format(windows="\n\n".join(f"[{i}] {docs[i]}" for i in ids), question=question, lines=lines))
```

## Stage 3: seed groups and grow each until no outside item joins it

```repl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer(strip_accents="unicode", stop_words="english", sublinear_tf=True, max_df=.05, dtype=np.float32).fit_transform([docs[i] for i in P])
nb = {}
for s in range(0, len(P), 512):
    z = (X[s:s + 512] @ X.T).toarray()
    for off, row in enumerate(z):
        row[s + off] = -1
        top = np.argpartition(-row, 30)[:30]
        nb[P[s + off]] = [(P[j], float(row[j])) for j in sorted(top, key=lambda j: -row[j])]
STOP = set("the a an of and in on at to for by with from as is was were be".split())
words = lambda t: frozenset(w for w in re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", " ", t.lower())) if w not in STOP and len(w) > 1)
kw = {i: [words(t) for t in titles.get(i, [])] for i in P}
widx = collections.defaultdict(set)
for i in P:
    for k in kw[i]:
        for w in k: widx[w].add(i)
agree = lambda a, b: any(len(x & y) / max(1, len(x | y)) >= .5 for x in kw[a] for y in kw[b] if x and y)
same_name = {i: {j for k in kw[i] for w in k if len(widx[w]) <= 200 for j in widx[w] if j != i and agree(i, j)} for i in P}
def components(edges):
    par = {i: i for i in P}
    def root(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for a, b in edges: par[root(a)] = root(b)
    c = collections.defaultdict(list)
    for i in P: c[root(i)].append(i)
    return [tuple(sorted(m)) for m in c.values() if 2 <= len(m) <= 14]
iso = lambda g: sum(max([s for j, s in nb[i] if j not in g] or [0.0]) for i in g) / len(g)
top2 = {i: [j for j, _ in nb[i][:2]] for i in P}
seeds = sorted(components((i, j) for i in P for j in top2[i] if i in top2[j]), key=iso)[:300]
seeds += sorted(components((i, j) for i in P for j in same_name[i]), key=iso)[:100]
def outside(g, skip, k):
    order = [j for i in g for j in sorted(same_name[i])] + [nb[i][q][0] for q in range(30) for i in sorted(g)]
    seen, res = set(g) | skip, []
    for j in order:
        if j not in seen: seen.add(j); res.append(j)
        if len(res) >= k: break
    return res
GROUP = """Below is a GROUP of passages and a list of OTHER passages, all windows of text cut out of English Wikipedia articles.

GROUP:
{group}

On a line starting GROUP SAME:, say yes if all GROUP passages were cut from one and the same Wikipedia article, otherwise no.
Then decide, for each OTHER passage, whether it was cut from that SAME article -- possibly a different section of it. A passage from a different article (a related person, a sequel, a neighbouring place, a similar topic) does not count.

OTHER PASSAGES:
{others}

Reply with exactly two lines:
GROUP SAME: <yes or no>
SAME ARTICLE: <document numbers separated by commas, or none>"""
state, todo = {}, []
for g in seeds:
    f = frozenset(g)
    if f not in state: state[f] = {"same": True, "closed": False, "shown": set()}; todo.append(f)
for rnd in range(3):
    batch = [(f, outside(f, state[f]["shown"], 16)) for f in todo]
    batch = [(f, o) for f, o in batch if o]
    replies = llm_query_batched([GROUP.format(group="\n\n".join(f"Document [{i}]: {docs[i]}" for i in sorted(f)), others="\n\n".join(f"Document [{j}]: {docs[j]}" for j in o)) for f, o in batch])
    todo = []
    for (f, o), r in zip(batch, replies):
        st = state[f]; st["shown"] |= set(o)
        same = re.search(r"GROUP SAME\s*:\s*(yes|no)", r, re.I); joined = re.search(r"SAME ARTICLE\s*:\s*(.*)", r, re.I)
        new = [j for j in (int(x) for x in re.findall(r"\d+", joined.group(1) if joined else "")) if j in o]
        if not same or same.group(1).lower() == "no": st["same"] = False
        elif not new: st["closed"] = True
        elif len(f) + len(new) <= 14:
            g2 = frozenset(f | set(new))
            if g2 not in state: state[g2] = {"same": True, "closed": False, "shown": set(st["shown"])}; todo.append(g2)
closed = sorted((f for f, st in state.items() if st["same"] and st["closed"]), key=iso)
print(len(state), "groups checked,", len(closed), "closed (no outside item joined); sizes:", dict(sorted(collections.Counter(len(f) for f in closed).items())))
```

## Stage 4: reassemble the most isolated closed groups, then CHECK the leader

A closed group can still be a piece of a longer source whose other windows were never shown to the check, so each
candidate's source is reassembled from a wider pool, and the shortest source leads. Then check the leader before
believing it: if its reassembly added windows, or its members' candidate titles do not agree with each other, it is
probably a piece of a longer source -- investigate it (next stage) instead of submitting.

```repl
RECON = """Below are numbered windows of text. Each was cut out of an English Wikipedia article; consecutive windows of one article continue each other exactly (often mid-sentence).

The windows marked GROUP come from one article. Reassemble that article's continuous text: list EVERY window below (GROUP or not) that belongs to the same article, in reading order, as far as the text continues. Do not include windows from other articles, even on a related subject.

{windows}

Reply with exactly one line:
ORDER: <window numbers in reading order, separated by commas>"""
def reassemble(groups, k=40):
    pools = [sorted(f) + outside(f, set(), k) for f in groups]
    replies = llm_query_batched([RECON.format(windows="\n\n".join(f"[{n}]{' GROUP' if i in f else ''} {docs[i]}" for n, i in enumerate(pool, 1))) for f, pool in zip(groups, pools)])
    out = {}
    for f, pool, r in zip(groups, pools, replies):
        m = re.search(r"ORDER\s*:\s*(.*)", r, re.I)
        out[f] = {pool[x - 1] for x in (int(y) for y in re.findall(r"\d+", m.group(1) if m else "")) if 1 <= x <= len(pool)} | set(f)
    return out
cands = closed[:30]
source = reassemble(cands)
def title_agreement(f):
    pairs = list(itertools.combinations(sorted(f), 2))
    return sum(agree(a, b) for a, b in pairs) / max(1, len(pairs))
ranked = sorted(cands, key=lambda f: (len(source[f]), iso(f)))
for f in ranked[:5]:
    print("source", len(source[f]), "| group", len(f), "| titles agree", round(title_agreement(f), 2), "|", [names.get(i, "")[:28] for i in sorted(f)])
lead = ranked[0]
print("leader check: reassembly added", len(source[lead]) - len(lead), "windows; title agreement", round(title_agreement(lead), 2))
```

## Stage 5 (only when the leader's check failed): investigate the leader's source, then re-rank

Grow the leader with everything its reassembly found plus wider neighbours of ALL its members (by title and by text),
ask again until nothing joins, and reassemble once more. If it grew, it was a piece of a longer source: drop it and
check the next candidate the same way. Stop once a leader passes the check (or after a few candidates) and say which
evidence decided it.

```repl
def investigate(f, k=60):
    g = set(source.get(f, set())) | set(f)
    for _ in range(3):
        o = outside(frozenset(g), set(), k)
        r = llm_query(GROUP.format(group="\n\n".join(f"Document [{i}]: {docs[i]}" for i in sorted(g)), others="\n\n".join(f"Document [{j}]: {docs[j]}" for j in o)))
        joined = re.search(r"SAME ARTICLE\s*:\s*(.*)", r, re.I)
        new = [j for j in (int(x) for x in re.findall(r"\d+", joined.group(1) if joined else "")) if j in o]
        if not new: break
        g |= set(new)
    full = reassemble([frozenset(g)], k)[frozenset(g)]
    return g | full
checked = {}
for f in ranked[:4]:
    full = investigate(f)
    checked[f] = len(full)
    print("candidate", [names.get(i, "")[:24] for i in sorted(f)], "| group", len(f), "| full source after investigation", len(full))
    if len(full) == len(f) and title_agreement(f) >= .5:
        break
lead = min(checked, key=lambda f: (checked[f], iso(f)))
best = [items[i - 1][0] for i in sorted(lead)]
print("leader after investigation:", best, "| source length", checked[lead])
```

## Final stage: submit

If the stage-4 check passed, `best = [items[i - 1][0] for i in sorted(lead)]`. Write the answer in exactly the format
the task states, using the items' own ids from `best`, and submit:
`answer["content"] = <that string>; answer["ready"] = True`.
