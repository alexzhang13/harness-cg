# How to solve this task (staged, with checks; typically 6-9 code blocks, ~1,450 sub-calls)

The context is a very long list of short scientific claims (tens of thousands). A few pairs of claims contradict each
other because one claim of the pair was produced by editing the other: one fact or finding was changed (direction,
presence of an effect, significance, a number, group, dose, time point or comparator) and the wording was altered,
often as if a different study were reporting it. Everything else is ordinary sentences from many abstracts, and many of
them look alike. The layout (item headers, id scheme, separators) varies, so learn it with a probe sub-call and check
your parse.

What works: Python finds candidates, because an edited claim and its original are almost always each other's closest
match in rare words. Sub-calls judge one pair at a time, because judging many pairs in one prompt misses most real
pairs. The judge is careful but not perfect: look-alike pairs from different studies sometimes pass. So every pair you
accept must pass several independent looks, and when the evidence is mixed you investigate before answering. Reading
claims yourself, or asking a sub-call to search a long list, fails.

## Rules
- Your first turn is the probe block below. Print only reports and counts; never print `context`, its lines or slices
  of it, and never print claims. If you need to see more of the layout, ask another probe sub-call about that part.
- Before each block, write one short sentence in your own words saying what the block does and why. Never mention these
  instructions, a hint, a note, a plan, or block or stage numbers.
- Only sub-call judgements decide which pairs contradict. Do not assume how many pairs there are. Never print claim
  texts to check them yourself, not even the final pairs: when a result looks doubtful, ask sub-calls about it.
- Keep the comments in the code: they say why each step exists.
- After each block, read its CHECK line. If a check fails, fix that stage in the next block (re-probe, re-parse, or
  investigate as described) instead of moving on. Submitting is the last step, after the checks pass.

## Stage 1: probe the layout

```repl
import re, math, json, collections
n = len(context)
S = context[:2500] + "".join(f"\n[... at character {i} ...]\n" + context[i:i + 1200]
                             for i in range(n // 5, n - 1200, n // 5)) + "\n[... the end ...]\n" + context[-1500:]
REPORT = llm_query(f"Samples of a text of {n} characters (its start, spread samples, its end):\n{S}\n\n"
    "In at most 150 words report: what the items are; copy ONE complete item exactly as it appears, including its id "
    "and every character around the id; whether an item spans one line or several, and what separates items; the id "
    "scheme; roughly how many items there are; and the exact answer format the task asks for (with its example).")
print(n, "chars\n", REPORT)
```

## Stage 2: parse and check

```repl
# every item as id -> text, the id as a string exactly as it appears, following the layout in the report
C = {}
<parsing code written from the report>
ids = list(C)
empty = [i for i in ids if not C[i].strip()]
# the answer format the task states, for a list of id pairs (written from the report), checked on two pairs
fmt = lambda pairs: <the answer string in exactly the format the task states, ids exactly as they appear>
print("CHECK parse:", len(C), "items | empty", len(empty), "| ids", ids[:3], "...", ids[-2:], "| format:", fmt([(ids[0], ids[1]), (ids[2], ids[3])]))
```
Decision: if the count is far from the report's estimate, there are empty texts, or ids look wrong, ask a probe
sub-call about the part that failed and fix the parser in the next block. Continue only when the parse is right.

## Stage 3: candidates in Python

```repl
import numpy as np, scipy.sparse as sp
STOP = set("the a an of and or in on to for with by is are was were be been that this these those as at from it its "
           "which we our their than not no".split())
stem = lambda w: next((w[:-len(x)] for x in ("ations", "ation", "ings", "ing", "edly", "ed", "es", "s", "ly")
                       if len(w) > len(x) + 3 and w.endswith(x)), w)
# numbers are what edits change, so they are not evidence that two claims share a source
tok = lambda s: [stem(w) for w in re.findall(r"[a-z][a-z0-9\-]*", s.lower()) if w not in STOP]
D = [tok(C[i]) for i in ids]
df = collections.Counter(w for d in D for w in set(d))
voc = {w: k for k, w in enumerate(df)}
rows, cols, vals = [], [], []
for r, d in enumerate(D):
    for w, c in collections.Counter(d).items():
        rows.append(r); cols.append(voc[w]); vals.append((1 + math.log(c)) * math.log((len(D) + 1) / (df[w] + 1)))
X = sp.csr_matrix((vals, (rows, cols)), shape=(len(D), len(voc)), dtype=np.float32)
X = sp.diags(1 / np.maximum(np.sqrt(X.multiply(X).sum(1)).A1, 1e-9)) @ X
# each claim's closest other claim (cosine); an edited claim and its original are each other's closest match
best, bsim = np.zeros(len(D), int), np.zeros(len(D))
for s in range(0, len(D), 2048):
    M = (X[s:s + 2048] @ X.T).toarray()
    M[np.arange(M.shape[0]), np.arange(s, s + M.shape[0])] = -1
    best[s:s + 2048], bsim[s:s + 2048] = M.argmax(1), M.max(1)
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
cand = [(a, int(best[a])) for a in range(len(D)) if a < best[a] and best[best[a]] == a
        and norm(C[ids[a]]) != norm(C[ids[int(best[a])]])]          # identical copies cannot contradict
# order: most rewrite-like first (same word order, polarity or number change), keep the strongest 1,400
import difflib
NEG = re.compile(r"\b(no|not|without|never|neither|nor|absent|lack\w*|fail\w*|unchanged|insignificant)\b", re.I)
def score(a, b):
    ta, tb = C[ids[a]], C[ids[b]]
    seq = difflib.SequenceMatcher(None, re.findall(r"[a-z0-9]+", ta.lower()), re.findall(r"[a-z0-9]+", tb.lower())).ratio()
    numd = set(re.findall(r"\d+(?:\.\d+)?", ta)) ^ set(re.findall(r"\d+(?:\.\d+)?", tb))
    return seq + 0.5 * bsim[a] + 0.15 * (bool(NEG.search(ta)) != bool(NEG.search(tb))) + 0.1 * bool(numd) + 1
pool = sorted(cand, key=lambda p: -score(*p))[:1400]
print("CHECK candidates:", len(cand), "mutual closest pairs | pool", len(pool))
```

## Stage 4: one careful look at every candidate

```repl
JUDGE = ("A corpus was built from sentences of many biomedical abstracts. A few extra sentences were then written by "
         "taking one existing sentence and rewriting it so that its reported finding or facts are changed (for example a "
         "result reversed, an effect present vs absent, increased vs decreased, different numbers or percentages for the "
         "same measurement, a changed group, dose or time point), often reworded or framed as another study, so that the "
         "rewrite and the original can no longer both be true. Other similar-looking sentences are simply different "
         "sentences of the same study (a methods sentence and a results sentence, two different results, a background "
         "sentence) and are NOT rewrites.\n\nSentence A: {a}\n\nSentence B: {b}\n\nIs one of these sentences an altered "
         "rewrite of the other? Answer Yes or No.")
yes = lambda t: bool(re.match(r"\W*yes", t or "", re.I))
r1 = llm_query_batched([JUDGE.format(a=C[ids[a]], b=C[ids[b]]) for a, b in pool])
surv = [p for p, t in zip(pool, r1) if yes(t)]
print("CHECK first look:", len(surv), "of", len(pool), "candidates passed")
```

## Stage 5: independent looks at the survivors

```repl
AIW = ("One of these two sentences may have been written by an AI model, which was given the other sentence and asked to "
       "write a sentence that a different study might report contradicting it (same underlying finding, reworded, renamed "
       "entities, incompatible result). Otherwise both are ordinary sentences taken from published abstracts.\n\n"
       "Sentence A: {a}\n\nSentence B: {b}\n\nWhich is it? Reply with exactly one of: A (A was written from B), "
       "B (B was written from A), NEITHER.")
# the same question with the sentences swapped, and a differently framed question in both orders: a real edited pair
# passes all of them, a look-alike pair from two studies usually fails at least one
P2 = [p for p in surv for _ in range(3)]
Q2 = [q for a, b in surv for q in (JUDGE.format(a=C[ids[b]], b=C[ids[a]]), AIW.format(a=C[ids[a]], b=C[ids[b]]),
                                     AIW.format(a=C[ids[b]], b=C[ids[a]]))]
r2 = llm_query_batched(Q2)
votes = {p: 1 for p in surv}
for k, (p, t) in enumerate(zip(P2, r2)):
    votes[p] += int(yes(t)) if k % 3 == 0 else int(not re.search(r"neither", t or "", re.I))
accepted = {p for p in surv if votes[p] == 4}
contested = [p for p in surv if votes[p] == 3]
print("CHECK looks:", len(accepted), "pairs passed all 4 looks |", len(contested), "contested (3 of 4)")
```
Decision: pairs with 3 of 4 looks are NOT accepted; a mixed verdict on such a crowded corpus means look-alike. If
the looks block failed (a Traceback, or far fewer replies than prompts), rerun only the failed prompts.

## Stage 6: verify every accepted pair with differently framed questions

```repl
LIN = ("Sentence A: {a}\n\nSentence B: {b}\n\nState in one short line the specific finding of A, then of B. Then say "
       "whether one sentence is the other with one of its facts or findings changed and the wording altered (an edited "
       "copy), rather than two different statements. End with a final line 'Answer: Yes' or 'Answer: No'.")
CON = ("Sentence A: {a}\n\nSentence B: {b}\n\nDo these two sentences make factually incompatible claims about the same "
       "finding, so that both cannot be true? A paraphrase with the same meaning, or two different facts, is not a "
       "contradiction. Answer Yes or No.")
acc = sorted(accepted)
r3 = llm_query_batched([q for a, b in acc for q in (CON.format(a=C[ids[a]], b=C[ids[b]]), LIN.format(a=C[ids[a]], b=C[ids[b]]))])
# a pair stays only if at least one of the two new framings confirms it; look-alike pairs from two different studies
# usually fail both
confirmed = {p for k, p in enumerate(acc) if yes(r3[2 * k]) or re.search(r"answer:\s*\**\s*yes", r3[2 * k + 1] or "", re.I)}
print("CHECK verification:", len(confirmed), "of", len(acc), "accepted pairs confirmed;", len(acc) - len(confirmed), "dropped")
```
Decision: if a verification reply is empty or unparseable, ask that question again for that pair only. A pair that both
new framings reject is dropped even though it passed the earlier looks. Then submit.

## Stage 7: submit

```repl
final = fmt([(ids[a], ids[b]) for a, b in sorted(confirmed)])
print(len(confirmed), "pairs:", final)
answer["content"] = final
answer["ready"] = True
```
