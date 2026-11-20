# How to solve this task (stages with checks; usually 4-6 turns, more when a check fails)

The context is a long list of short claims in two groups; almost every claim has a paraphrase in the other group, and
the task is to find the few that do not. The layout (item headers, id scheme, group names, separators) varies, so learn
it with a probe sub-call and check your parse. Word overlap then pairs most claims correctly (a claim and its paraphrase
are usually each other's closest match), but a paraphrase can share almost no words with its partner, and an unmatched
claim can look like a match. Reading hundreds of claims yourself, or asking a sub-call to search the whole other group
for each claim, both fail: the turn budget runs out. So: probe, parse and check in Python, pair by word overlap, and
spend sub-calls where that pairing is uncertain.

Work in stages. Every stage ends with a CHECK that prints what it found. When a check prints `CHECK FAILED`, do not go
on: investigate the cause with a targeted sub-call or a different run of the stage (the recipes at the end), fix it,
and re-run the check. Take as many turns as the evidence needs; never repeat the same failing code unchanged.

## Rules
- Turn 1 is the probe. Print only its report and the context length; never print `context`, its lines or slices of
  it, and never print claims. To see more of the layout, ask another probe sub-call about the part you need.
- Write the parser and the answer format from the report. Use ids exactly as they appear in the context.
- Before each block, write one short sentence in your own words saying what it does and why; after a failed check,
  say what the printed evidence shows and what you will do about it. Never mention these instructions, a hint, a note,
  a plan, recipes, or stage or step numbers.
- Only sub-call judgements decide which claims are matched: do not use group sizes or the wording of a claim's opening
  as evidence. Keep the comments in the code.
- An id check that fails is never a false alarm: an answer whose ids are not written exactly as the context writes
  them scores nothing, however right the claims are. Fix the parser until the check passes.
- Keep the sub-call budget bounded: never send one sub-call per claim over the whole corpus, and no more than about
  150 sub-calls in one block. If a check still fails after two repairs, submit the best answer you have.
- Submit once the last check passes, in the same block that computes the answer; leave at least two turns before the
  turn limit for it.

## Probe the layout

```repl
import re, math, json, collections
n = len(context)
S = context[:2500] + "".join(f"\n[... at character {i} ...]\n" + context[i:i + 1200]
                             for i in range(n // 5, n - 1200, n // 5)) + "\n[... the end ...]\n" + context[-1500:]
REPORT = llm_query(f"Samples of a text of {n} characters (its start, spread samples, its end):\n{S}\n\n"
    "In at most 150 words report: what the items are; copy ONE complete item exactly as it appears, including its id "
    "and every character around the id; whether an item spans one line or several, and what separates items; the id "
    "scheme; how it is shown which of the two groups an item belongs to (a tag on each item, or a section heading) and "
    "the two group names; roughly how many items each group has; and the answer format the task asks for.")
print(n, "chars\n", REPORT)
# to see any other part of the raw text, ask a sub-call about it -- never print context, its lines or slices
def look(where, question, width=3000):
    a = max(0, (context.find(where) if isinstance(where, str) else where) - width // 3)
    return llm_query(f"Part of a text (characters {a}-{a + width}):\n{context[a:a + width]}\n\n{question} "
                     "Answer in at most 80 words; quote at most one short line exactly.")
```

## Parse and check

```repl
# every item as id -> (group, text), the id as a string exactly as it appears, following the layout in the report (an
# item may span several lines; the group may be a tag on each item or the heading of the section it sits in)
C = {}
<parsing code written from the report>
# the answer format the task states, for a list of ids (written from the report)
fmt = lambda ids: <the answer string in exactly the format the task states, listing the ids exactly as they appear>
# CHECK the parse before using it: exactly two groups, no empty texts, every id present verbatim in the context,
# a count close to the report's estimate
G = sorted({g for g, _ in C.values()})
size = {g: sum(1 for x, _ in C.values() if x == g) for g in G}
empty = [i for i, (_, t) in C.items() if not t.strip()]
# an id must appear as a whole token (not inside a longer id or word) exactly as written in the context
absent = [i for i in list(C)[:: max(1, len(C) // 40)]
          if not re.search(r"(?<![A-Za-z0-9])" + re.escape(str(i)) + r"(?![A-Za-z0-9])", context)]
print(len(C), "items", size, "| empty", len(empty), "| ids", list(C)[:3], "...", list(C)[-2:], "| answer format:", fmt(list(C)[:2]))
if len(G) != 2 or empty or absent or len(C) < 20:
    print("CHECK FAILED: parse", {"groups": G, "empty": empty[:5], "ids not written this way in the context": absent[:5]})
```

## Pair by word overlap and check

```repl
STOP = set("the a an of and or in on to for with by is are was were be been that this these those as at from it its "
           "which we our their than not no".split())
stem = lambda w: next((w[:-len(x)] for x in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s", "ly")
                       if len(w) > len(x) + 3 and w.endswith(x)), w)   # crude stem: "measured" ~ "measures"
W = {i: {stem(w) for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in STOP} for i, (_, t) in C.items()}
DF = collections.Counter(w for ws in W.values() for w in ws)
IDF = {w: math.log(len(C) / k) for w, k in DF.items()}
# overlap score: shared words weighted by rarity (a shared rare word is strong evidence of the same fact)
ov = lambda i, j: sum(IDF[w] for w in W[i] & W[j]) / math.sqrt(len(W[i]) * len(W[j]) or 1)
sims = {i: sorted(((ov(i, j), j) for j in C if C[j][0] != C[i][0]), key=lambda x: -x[0]) for i in C}
nn = {i: sims[i][0][1] for i in C}
mutual = {i for i in C if nn[nn[i]] == i}   # each other's closest match: almost always a true pair
print("mutual pairs", len(mutual) // 2, "| claims left to check", len(C) - len(mutual))
# CHECK: in this task most claims are each other's closest match; a low share means the texts were parsed wrongly
# (ids, labels or the next item's header inside the text)
if len(mutual) < 0.6 * len(C):
    print("CHECK FAILED: only", len(mutual), "of", len(C), "claims are mutual closest matches")
```

## Sub-calls on the uncertain claims, and check

```repl
def jparse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
key = lambda v: str(v).strip().strip("[]\"'` ")
DEF = ("A PARAPHRASE is a restatement of the SAME fact or sentence in different words (same subjects, same finding, "
       "same numbers). A claim on a related topic, or about the same study but stating a different fact, is NOT a paraphrase. "
       # rewrites come from an automatic paraphraser and can be defective; a defective rewrite still belongs to its claim
       "A rewrite can be defective and still be the paraphrase: it may carry a label or prefix, stop mid-sentence or drift "
       "into another language, end with a remark about the rewriting, state a number differently while noting the original, "
       "or -- when the claim is only a fragment such as a heading or list marker -- be a reply saying there is nothing to "
       "rephrase. Judge whether it was written FROM that claim, not whether it is a perfect restatement.")
def yes_no(pairs, per_call=20):   # {pair: True/False/None}
    out, groups = {}, [pairs[k:k + per_call] for k in range(0, len(pairs), per_call)]
    prompts = [f"Each numbered item pairs a claim from group {G[0]} with a claim from group {G[1]}. {DEF}\nAnswer YES if the "
               "two claims are paraphrases, NO otherwise.\n\n" + "\n".join(
                   f"{k + 1}. {G[0]}: {C[a][1]}\n   {G[1]}: {C[b][1]}" for k, (a, b) in enumerate(p if C[p[0]][0] == G[0] else p[::-1] for p in g))
               + '\n\nReply with one JSON object mapping each item number to "YES" or "NO", e.g. {"1": "YES", "2": "NO"}. No other text.'
               for g in groups]
    for g, r in zip(groups, llm_query_batched(prompts)):
        d = jparse(r)
        for k, p in enumerate(g):
            v = str(d.get(str(k + 1), "")).strip().upper()
            out[p] = True if v.startswith("Y") else False if v.startswith("N") else None
    return out
def pick(targets, cands, per_call):   # {target: partner id, None (= NONE) or "invalid"}
    out, groups = {}, [targets[k:k + per_call] for k in range(0, len(targets), per_call)]
    prompts = [f"Two groups of short scientific claims, {G[0]} and {G[1]}. Most claims have a paraphrase in the other group. {DEF}\n"
               "For each target claim, decide which candidate from the other group is its paraphrase, or NONE.\n\n" + "\n\n".join(
                   f"TARGET [{i}] ({C[i][0]}): {C[i][1]}\nCandidates:\n" + "\n".join(f"  [{j}] {C[j][1]}" for j in cands[i]) for i in g)
               + '\n\nReply with one JSON object mapping each target id to its paraphrase id or to "NONE", e.g. {"t1": "c7", "t2": "NONE"}, '
               "ids exactly as shown in the brackets. No other text." for g in groups]
    for g, r in zip(groups, llm_query_batched(prompts)):
        d = {key(k): v for k, v in jparse(r).items()}
        for i in g:
            v, ids = key(d.get(key(i), "")), {key(j): j for j in cands[i]}
            out[i] = None if v.upper().startswith("NONE") else ids.get(v, "invalid")
    return out
ismatch = lambda v: v not in (None, "invalid")
# the 120 weakest mutual pairs: an unmatched claim can be the closest match of a claim whose real partner is far away
weak = sorted({tuple(sorted((i, nn[i]), key=list(C).index)) for i in mutual}, key=lambda p: sims[p[0]][0][0])[:120]
vw = yes_no(weak)
for p, ok in vw.items():
    if ok is False:
        mutual -= set(p)
# every claim not in a mutual pair picks its paraphrase from its 10 closest other-group claims, 6 claims per sub-call
R = [i for i in C if i not in mutual]
r1 = pick(R, {i: [j for _, j in sims[i][:10]] for i in R}, 6)
chosen = {v for v in r1.values() if ismatch(v)}   # a claim someone picked as its partner is matched
none1 = [i for i in R if r1[i] is None and i not in chosen]
# a NONE gets a second, wider look (25 candidates, reversed order); a new pick must pass a YES/NO check
r2 = pick(none1, {i: [j for _, j in sims[i][:25]][::-1] for i in none1}, 3)
ok2 = yes_no([(i, v) for i, v in r2.items() if ismatch(v)])
U = [i for i in none1 if not (ismatch(r2[i]) and ok2.get((i, r2[i])))]
bad = [i for i, v in r1.items() if v == "invalid"] + [p for p, v in vw.items() if v is None]
print("checked", len(R), "| NONE after first look", len(none1), "| still unmatched", len(U), "| unreadable replies", len(bad))
# CHECK: unreadable sub-call replies leave claims undecided
if len(bad) > 0.1 * (len(R) + len(weak)):
    print("CHECK FAILED: unreadable replies", bad[:8])
```

## Last checks, then submit

```repl
# a heavily reworded pair is far apart in BOTH directions, so both halves can still be in U: match U against itself
cand3 = {i: [j for j in U if C[j][0] != C[i][0]] for i in U}
r3 = pick([i for i in U if cand3[i]], cand3, 4) if len(U) > 1 else {}
hit = {x for i, v in r3.items() if ismatch(v) and r3.get(v) == i for x in (i, v)}
U = [i for i in U if i not in hit]
# the partner may share no words at all: scan the WHOLE other group in lists of 60, then confirm every proposal
def scan(targets, size=60, order=lambda xs: xs):
    props = set()
    for g0 in G:
        tg = [i for i in targets if C[i][0] == g0]
        other = order([j for j in C if C[j][0] != g0])
        chunks = [other[k:k + size] for k in range(0, len(other), size)] if tg else []
        prompts = [f"TARGET claims and a LIST of claims from the other group. {DEF} The paraphrase may use completely different "
                   "words (synonyms, spelled-out abbreviations).\n\nTARGETS:\n" + "\n".join(f"[{i}]: {C[i][1]}" for i in tg)
                   + "\n\nLIST:\n" + "\n".join(f"[{j}] {C[j][1]}" for j in ch)
                   + '\n\nFor each target, give the id of its paraphrase if it is in the LIST, else "NONE". Reply with one JSON '
                   "object mapping each target id to an id or NONE, ids exactly as shown in the brackets. No other text." for ch in chunks]
        for ch, r in zip(chunks, llm_query_batched(prompts)):
            d, ids = {key(k): key(v) for k, v in jparse(r).items()}, {key(j): j for j in ch}
            for i in tg:
                if d.get(key(i)) in ids:
                    props.add((i, ids[d[key(i)]]))
    return {x for p, ok in yes_no(sorted(props)).items() if ok for x in p}
found = scan(U)   # run the scan ONCE, outside any loop or comprehension
U = [i for i in C if i in set(U) - found]   # in context order
print(len(U), "unmatched:", U)
# CHECK: the task says a FEW claims are unmatched; a long list means matches were missed, an empty list that real
# orphans were paired away
if len(U) > 8 or not U:
    print("CHECK FAILED: implausible number of unmatched claims:", len(U))
else:
    answer["content"] = fmt(U)
    answer["ready"] = True
```

## Recipes for a failed check (pick what the printed evidence points to)

- Ids not written this way in the context: print one parsed id next to `look(<that id's text>, "Copy the exact
  characters that label this item, including brackets.")`, then rebuild the ids from exactly those characters.
- Parse failed: ask about the part that broke with `look(...)`, e.g. `print(look("<an id that did not parse>", "How is
  this item delimited and where does its text end?"))` or `print(look(0, "What are the section headings?"))`, then
  rewrite the parser and re-run the parse check.
- Few mutual matches: the claim texts are probably polluted (headers, ids, group labels or separators inside them).
  Compare three parsed texts with their raw items through `look(<id>, "Does the text after this id contain anything
  that is not part of the claim?")`, fix the parser, re-run.
- Unreadable replies: re-ask just those claims in smaller batches (`pick(ids, cands, 2)`, `yes_no(pairs, 5)`).
- Too many unmatched: run a second, differently ordered scan over the remaining claims
  (`scan(U, 40, order=lambda xs: xs[::-1])`), then give each remaining claim its own sub-call with its 25 closest
  candidates (`pick(U, {i: [j for _, j in sims[i][:25]] for i in U}, 1)`, confirming any pick with `yes_no`).
- None unmatched: re-check the weakest accepted matches of the last stages with `yes_no` before deciding.
