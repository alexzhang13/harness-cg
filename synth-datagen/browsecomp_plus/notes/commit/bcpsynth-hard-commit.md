`context` is a whole web corpus: a list of tens of thousands of document strings, each starting with `doc_id:` and usually `title:` lines. The question chains paraphrased clues; the answer is usually a fact in a document about an entity that other documents only help you identify. Python searches the corpus; sub-calls read documents. You only ever handle doc_ids (strings) -- never list positions -- and never print document text yourself beyond titles.

Print budget: about 3,000 characters per turn. Never print `context[...]`, document bodies or lists of doc_ids -- print `len(...)` and `titles(ids[:20])`; look inside a document with `excerpt()` (a few windows) or `read()` (a sub-call).

Turn 1, exactly this block (look before you build anything; a sub-call describes the layout so you never assume it):
```repl
print(len(context)); print(context[0][:300])
print(llm_query("Describe this document's layout in at most 60 words (header lines, where the text starts); do not summarise its content.\n\n" + context[0][:2000]))
```

Turn 2, one block: write these tools (search scans only the query's words: no index needed), then the first search:
```repl
import re, math, collections
TOK = re.compile(r"[a-z0-9]+")
ids = [d.split('\n', 2)[1].split(':', 1)[1].strip() for d in context]; pos = {d: i for i, d in enumerate(ids)}
low = [d.lower() for d in context]; N = len(low); avg = sum(map(len, low)) / N
_tf = {}
def _hits(w):
    if w not in _tf: _tf[w] = {i: c for i, d in enumerate(low) if (c := d.count(w))}
    return _tf[w]
def search(q, k=100):   # BM25 over the query's own words -> doc_ids
    s = collections.defaultdict(float)
    for w in set(TOK.findall(q.lower())):
        if len(w) < 3: continue
        h = _hits(w)
        if not h or len(h) > N // 5: continue
        idf = math.log(1 + (N - len(h) + .5) / (len(h) + .5))
        for i, f in h.items(): s[i] += idf * f * 1.9 / (f + .9 * (.6 + .4 * len(low[i]) / avg))
    return [ids[i] for i in sorted(s, key=s.get, reverse=True)[:k]]
def fuse(queries, k=60):
    s = collections.defaultdict(float)
    for q in queries:
        for r, d in enumerate(search(q, 300)): s[d] += 1 / (61 + r)
    return sorted(s, key=s.get, reverse=True)[:k]
def having(*phrases):   # doc_ids whose text contains ALL the lower-case phrases
    pool = range(N)
    for p in sorted(phrases, key=len, reverse=True): pool = [i for i in pool if p in low[i]]
    return [ids[i] for i in pool]
def title(d):
    t = re.search(r"(?m)^title:\s*(.+)$", context[pos[d]][:800]); return (t[1] if t else context[pos[d]][30:110]).strip()[:90]
def titles(dids, k=20):   # at most k lines
    dids = list(dids); print("\n".join(f"{d} | {title(d)}" for d in dids[:k]) + (f"\n... {len(dids) - k} more" if len(dids) > k else ""))
def excerpt(d, words, n=3, w=400):
    t, lo = context[pos[d]], low[pos[d]]; starts = []
    for h in sorted({m.start() for x in words for m in re.finditer(re.escape(x), lo)}):
        if all(abs(h - s) > w for s in starts): starts.append(h)
    return f"[{d}] {title(d)}\n" + " ... ".join(t[max(0, h - w // 2):h + w // 2].replace('\n', ' ') for h in starts[:n])
def screen(dids, words, per=10):   # one batched call; rows scored 2-3
    ps = [f"Question: {Q}\nFor EACH document reply one line `<doc_id> | <0-3 relevance> | <clues it supports> | <main entity> | <the asked value if stated, else ->`.\n\n"
          + "\n\n".join(excerpt(d, words) for d in dids[j:j + per]) for j in range(0, len(dids), per)]
    return [l for r in llm_query_batched(ps) for l in r.splitlines() if re.match(r"\s*\S+\s*\|\s*[23]\b", l)]
def read(dids, focus=""):   # full docs -> ANSWER / DOC / QUOTE / NEXT, quote checked against that doc
    r = llm_query(f"Question: {Q}\n{focus}\nUsing ONLY these documents reply exactly:\nANSWER: <the asked value, or NONE>\nDOC: <doc_id stating it>\nQUOTE: <exact words from it, max 25>\nNEXT: <the next entity to look up if these only lead to it, else ->\n\n"
                  + "\n\n".join(context[pos[d]][:9000] for d in dids[:3]))
    f = {k: (re.search(rf"{k}:\s*(.*)", r) or [None, ""])[1].strip() for k in ("ANSWER", "DOC", "QUOTE", "NEXT")}
    q = " ".join(f["QUOTE"].lower().split())[:60]
    f["ok"] = f["DOC"] in pos and len(q) > 8 and q in " ".join(low[pos[f["DOC"]]].split())
    return f
```
then set `Q` (the question), `queries` (~8 keyword queries that each combine 2-3 clues in a page's words; use names you are fairly sure of), `words` (distinctive lower-case clue words), and print `screen(fuse(queries), words)`. In the same block, brainstorm candidate names, because pages about the right entity often share few words with the question: `names = {l.strip(' -*0123456789.') for r in llm_query_batched([f"List up to 12 specific named entities (people, works, organisations, places, events) that could be the main entity or any linked entity in this puzzle's chain; use different interpretations of the clues. One name per line, nothing else.\n\n{Q}"] * 5) for l in r.splitlines() if 3 <= len(l.strip()) <= 60}` and print each name with `len(having(name.lower()))` for the names found in 1-400 docs. Also print `len(having(p))` for 3-5 distinctive 2-4-word phrases taken almost verbatim from the question (questions are often light paraphrases of a page's own wording).

Turn 3 onward, one decision per turn. Rank candidates by what screens and reads actually returned, never by your first guess: a candidate whose documents have not confirmed one of the question's distinctive clues after two reads is dropped for the next best-supported entity (from the screen rows, the name counts, or the phrase sets).
- A screen row names an entity or states the asked value: `titles(having("<name>"))` (add a word the fact sits next to if it returns many), then `read()` the 1-3 most promising doc_ids. If `NEXT` names another entity, do the same for it.
- A name or phrase in <= 40 docs, or a name plus a distinctive clue word (`having(name, word)`) that leaves <= 40: print `titles()` and `screen()` that set -- this is how pages that share few words with the question are found.
- Nothing scored 2-3: search for different evidence -- another clue, the page's likely wording, a candidate entity you know, or two rare words that must co-occur (`having(a, b)`) -- and screen only the new results.
- Before submitting, check the candidate once: one `llm_query` with the question, the candidate and its 1-3 documents, asking which clues the documents confirm and whether any contradicts it. Submit in that block if no clue is contradicted (`if ok:` set `answer["content"]`, then `answer["ready"] = True`).
Aim for 4-12 turns and 3-12 sub-calls.

Turn budget: every turn starts with `Turn k/N`. When at most 4 turns are left (k >= N-4) and no candidate has every distinctive clue confirmed, stop searching. Begin that turn by writing, in your own words, something like: "I see that I only have a few turns left and I still haven't figured out an answer. Let me consolidate a final program and guess an answer." Then, in ONE block: collect every candidate the screens, reads, name counts and checks have produced so far (a dict name -> the clues its documents confirmed or contradicted), print that list compactly, send ONE `llm_query` with the question, the clues and that evidence list, asking which candidate is best supported and why, and submit its answer in the same block (`answer["content"] = ...; answer["ready"] = True`). That block MUST end by submitting: do not spend another turn verifying, and do not wait for the last turn. If nothing at all was found, submit the single most plausible candidate from the evidence you have. A best-supported guess is always better than no answer; never let the turn budget run out without submitting.
