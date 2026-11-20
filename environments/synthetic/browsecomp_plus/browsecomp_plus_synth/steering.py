"""Steering for the synthetic BrowseComp-Plus env: hints (strategy / program) and program-level checks.

Off by default. ``load_environment(hint="strategy"|"program", hint_prob=p)`` appends a hint to the root prompt;
``reward_mode="gated"|"shaped"`` puts the checks into the reward (synth_common.program); they are always logged.

| check            | hard | passes when                                                                                 |
|------------------|------|---------------------------------------------------------------------------------------------|
| search_rounds    | yes  | >= 2 root turns run a corpus search                                                         |
| batched_screen   | yes  | >= 2 document-bearing llm_query_batched prompts, <= 8 documents per prompt, >= 8 read        |
| final_verified   | yes  | every clue is supported, for the target's own chain, by a fact the sub-LM confirmed (YES)   |
|                  |      | or that the root printed into its own REPL output                                           |
| no_agg_requests  | yes  | the sub-LM was never asked for "the answer"                                                 |
| not_whole_corpus | yes  | at most max(400, 2% of the corpus) distinct documents went through sub-calls                |
| no_overlong      | no   | no sub-call over the window                                                                 |
"""

from __future__ import annotations

import re

from synth_common.program import (
    Check,
    HintSpec,
    acct,
    no_aggregate_requests,
    no_overlong,
    root_code,
)

from .questions import Clue, _resolve_path
from .world import get_world

STRATEGY = """\
`context` is a list of tens of thousands of documents (a whole corpus), each \
`---\\ndoc_id: N\\n---\\n---\\ntitle: ...\\ndate: ...\\n---\\n<text>` (text cut at 12k characters). Solve it in three blocks:
1. Block 1, search: decode the question into numbered clues, each with 2-6 literal forms a page would print (proper \
names exactly as given, the specific words behind a paraphrase -- "a woodwind player" -> oboist, bassoonist, \
clarinettist --, the years of a narrow range, numbers in several print formats). In the same block: lower-case \
`context` once, build a BM25 index once, count documents per form over the WHOLE list (never a slice like \
`range(50000)`), and rank documents by fusing (reciprocal rank) BM25 of each question sentence, BM25 of your forms, \
and the clues whose rare forms a document contains. Print ids and titles of the top ranks, never document text. If \
the question asks for a value about the entity (a year, a place, a title) rather than the entity, write that neutral \
question as `ASKED`.
2. Block 2, read and decide: send the top 10 ranked documents, 2 per prompt, in ONE llm_query_batched call with the \
numbered clue list, asking exactly one line per document: `<doc_id> | clues: <numbers it states> | contradicted: \
<numbers> | entity: <who or what it is about> | quote: <exact words for its most specific clue>`. Tally entities \
(ignoring any with a contradicted clue); if the leader is weak, read its own documents (search its name) in the same \
block. Stop rule, in code: once the leader has 60% of the clues confirmed and is ahead of the runner-up, ask `ASKED` over its document if \
set (`ANSWER: ... | QUOTE: ...`) and submit, in that same block. A clue counts when a sub-call line lists it: never \
re-check it by printing documents.
3. Block 3, only if block 2 did not submit: new forms for the open clues, rank again, read the next unread top \
documents and the leader's own documents, then submit the entity with the most confirmed clues. No fourth search.
Never send the whole question to a sub-call and ask for the answer. Submit only the short name or value asked for \
(two facts: separated by a semicolon)."""

PROGRAM_SETUP = r'''import re, math, collections
import numpy as np
from array import array
LOW = [d.lower() for d in context]
N = len(LOW)
def _did(d):
    m = re.search(r"doc_id:\s*(\S+)", d[:80]); return m[1] if m else ""
DID = [_did(d) for d in context]; POS = {k: j for j, k in enumerate(DID)}
def title(j):
    m = re.search(r"(?m)^title:\s*(.+)$", context[j][:600]); return m[1].strip()[:90] if m else ""
def hits(*terms):                                   # positions of documents containing ALL terms (substring)
    return [j for j in range(N) if all(t in LOW[j] for t in terms)]
TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on at to for from by with and or is was were be been it its this that as which who whom "
           "whose what when where how than then their they he she his her not no also into about after before".split())
VOC, T, D, F, DL = {}, array("i"), array("i"), array("f"), np.zeros(N, np.float32)
for j, d in enumerate(LOW):                         # BM25 index, built once (~1 min)
    c = collections.Counter(TOK.findall(d)); DL[j] = sum(c.values())
    T.extend(VOC.setdefault(t, len(VOC)) for t in c); D.extend([j] * len(c)); F.extend(c.values())
T, D, F = np.frombuffer(T, np.int32), np.frombuffer(D, np.int32), np.frombuffer(F, np.float32)
o = np.argsort(T, kind="stable"); D, F = D[o], F[o]
START = np.searchsorted(T[o], np.arange(len(VOC) + 1)); AVG = float(DL.mean()); del T, o
def bm25(text, k1=0.9, b=0.4):
    s = np.zeros(N, np.float32)
    for t in set(TOK.findall(text.lower())) - STOP:
        i = VOC.get(t)
        if i is None: continue
        lo, hi = START[i], START[i + 1]
        if hi - lo > 0.05 * N: continue             # words in >5% of documents carry no signal
        d, f = D[lo:hi], F[lo:hi]
        s[d] += math.log(1 + (N - (hi - lo) + .5) / (hi - lo + .5)) * f * (k1 + 1) / (f + k1 * (1 - b + b * DL[d] / AVG))
    return s
def top(s, k=1000):
    k = min(k, N); i = np.argpartition(-s, k - 1)[:k]; return [int(j) for j in i[np.argsort(-s[i])]]
def rrf(lists, k=60):
    sc = collections.Counter()
    for L in lists:
        for r, j in enumerate(L[:1000]): sc[j] += 1 / (k + r + 1)
    return [j for j, _ in sc.most_common()]'''

PROGRAM_RANK = r'''Q = """<the question, pasted verbatim>"""
CLUES = [("<clue 1 in the question's words>", ["<literal form>", "..."]), ...]   # your decoding
ASKED = None   # or, if the question asks for a value about the entity: "In what year did {} ...?" ({} = the entity)
DF = {}
def clue_rank():                                    # clue coverage, rare forms weighing more
    for _, fs in CLUES:
        for f in fs:
            if f not in DF: DF[f] = len(hits(f.lower()))
    sc = collections.Counter()
    for _, fs in CLUES:
        best = {}
        for f in fs:
            if 0 < DF[f] < 0.03 * N:
                for j in hits(f.lower()): best[j] = max(best.get(j, 0), math.log(N / DF[f]))
        for j, w in best.items(): sc[j] += w
    return [j for j, _ in sc.most_common(1000)]
def rank():
    sent = [x for x in re.split(r"(?<=[.;?!])\s+", Q) if len(x.split()) >= 3]
    forms = " ".join(f for _, fs in CLUES for f in fs)
    return rrf([top(bm25(x)) for x in sent] + [top(bm25(forms))] * 2 + [clue_rank()] * 2)
RANK = rank()
print(DF)                                           # 0 hits: printed differently?  thousands: too generic
for j in RANK[:12]: print(DID[j], "|", title(j))'''

PROGRAM_READ = r'''READ = {}                                           # doc_id -> (clues stated, clues contradicted, entity)
def read(js, per=2):
    rule = ("Clues:\n" + "\n".join(f"{n}. {c}" for n, (c, _) in enumerate(CLUES, 1)) + "\nFor EACH document below "
            "reply exactly one line `<doc_id> | clues: <numbers of the clues it states, or 0> | contradicted: <numbers, "
            "or none> | entity: <the person / work / organisation it is about> | quote: <exact words for its most "
            "specific clue>`.\n\n")
    js = [j for j in js if DID[j] not in READ]
    for r in llm_query_batched([rule + "\n\n".join(context[j] for j in js[i:i + per]) for i in range(0, len(js), per)]):
        for line in r.splitlines():
            m = re.match(r"\W*(\d+)\W*\|\s*clues?:\s*([^|]*)\|\s*contradicted:\s*([^|]*)\|\s*entity:\s*([^|]*)", line)
            if m and m[1] in POS:
                READ[m[1]] = ({int(x) for x in re.findall(r"\d+", m[2]) if x != "0"},
                              {int(x) for x in re.findall(r"\d+", m[3])}, m[4].strip())
def tally():
    ent = collections.defaultdict(set)
    for yes, no, e in READ.values():
        if e and e.upper() != "NONE" and not no: ent[e].update(yes)
    return sorted(ent.items(), key=lambda kv: -len(kv[1]))
def own_docs(e, k):                                 # the entity's own documents, in rank order
    h = set(hits(e.lower())); return [j for j in RANK if j in h][:k]
def submit_best(force=False):                       # the stop rule
    global ENT, BEST
    ENT = tally(); BEST = ENT[0][0] if ENT else None
    print(len(CLUES), "clues |", [(e, sorted(c)) for e, c in ENT[:5]])
    lead = len(ENT[0][1]) if ENT else 0
    clear = lead >= max(2, math.ceil(0.6 * len(CLUES))) and lead > (len(ENT[1][1]) if len(ENT) > 1 else 0)
    if not BEST or not (force or clear):            # clear = 60% of the clues and ahead of the runner-up
        return False
    final = BEST
    if ASKED:                                       # the asked value, from the entity's best document
        j = POS[max((d for d, (y, n, e) in READ.items() if e == BEST), key=lambda d: len(READ[d][0]))]
        r = llm_query("Using only the document below, answer: " + ASKED.format(BEST) + "\nReply exactly one line "
                      "`ANSWER: <short answer, or NONE> | QUOTE: <exact words that state it>`.\n\n" + context[j])
        m = re.search(r"ANSWER:\s*([^|\n]+)", r)
        final = m[1].strip() if m and m[1].strip().upper() != "NONE" else final
    answer["content"] = final; answer["ready"] = True
    print("submitted:", final)
    return True
read(RANK[:10])
ENT = tally()
if ENT and len(ENT[0][1]) < max(2, math.ceil(0.6 * len(CLUES))):   # weak lead: read the leader's own documents
    read(own_docs(ENT[0][0], 4))
submit_best()'''

PROGRAM_RETRY = r'''CLUES += [("<an open clue, reworded>", ["<new literal form>", "..."])]   # new forms for the open clues only
RANK = rank()
read([j for j in RANK if DID[j] not in READ][:8] + (own_docs(ENT[0][0], 3) if ENT else []))
submit_best(force=True)                             # the entity with the most confirmed clues, whatever it has'''

PROGRAM = ("# --- block 1 (turn 1): setup + decode + all searches + fused rank\n" + PROGRAM_SETUP + "\n" + PROGRAM_RANK
           + "\n# --- block 2 (turn 2): read the top 10 with ONE batched sub-call round, tally, stop rule\n"
           + PROGRAM_READ
           + "\n# --- block 3 (turn 3, only if block 2 did not submit): one retry, then submit the best\n"
           + PROGRAM_RETRY)


class ReplHintSpec(HintSpec):
    """HintSpec with the program tier fenced as ```repl (models copy the fence they see)."""

    def text(self, tier: str) -> str:
        out = super().text(tier)
        return out.replace("```python\n", "```repl\n") if tier == "program" else out


HINTS = ReplHintSpec(strategy=STRATEGY, program=PROGRAM)


_SEARCH_RX = re.compile(
    r"\bin\s+(?:LOW|low|lowered|lower_docs|docs_lower|ctx_low|context_lower|lc|LC)\b|"
    r"\.lower\(\)\s*(?:for|\))|for\s+\w+(?:\s*,\s*\w+)?\s+in\s+(?:enumerate\()?(?:context|LOW|low|lowered)\b|"
    r"\b(?:hits|search|grep|find_docs?|lookup|match_docs?)\w*\s*\(")


def _search_rounds(s: dict) -> float:
    return 1.0 if sum(1 for c in root_code(s) if _SEARCH_RX.search(c)) >= 2 else 0.0


def _batched_screen(s: dict) -> float:
    a = acct(s)
    ok = a.get("batched_doc_prompts", 0) >= 2 and a.get("max_docs", 99) <= 8 and a.get("items_read", 0) >= 8
    return 1.0 if ok else 0.0


def _repl_facts(W, s: dict) -> set[int]:
    """Fact ids whose sentences the root printed into its own REPL output (the root read them itself)."""
    out: set[int] = set()
    for m in (s.get("rlm_history") or [])[2:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "user":
            continue
        c = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
        if "REPL output" not in c:
            continue
        for iid in W.index.present(c.lower(), 0.6):
            out.update(W.sent_facts.get(iid, ()))
    return out


def clue_support(s: dict) -> float:
    """Fraction of the question's clues supported, on the target's own chain, by confirmed or printed facts."""
    h = s.get("synth_hidden") or {}
    if not h.get("clues"):
        return 0.0
    from .oracle import BCPOracle
    W = get_world(h["world"], h["scale"])
    seen = set(acct(s).get("yes_fids") or []) | _repl_facts(W, s)
    judge = BCPOracle()
    ok = 0
    for c in h["clues"]:
        cl = Clue(**c)
        ents = {h["target"]} if cl.role == "self" else set(_resolve_path(W, h["target"], cl.role))
        facts = [f for f in seen if W.facts[f][0] in ents]
        if judge.verdict(W, facts, cl)[0] == "YES":
            ok += 1
    return ok / len(h["clues"])


def _not_whole_corpus(s: dict) -> float:
    h = s.get("synth_hidden") or {}
    n = len(get_world(h["world"], h["scale"]).low) if h.get("world") else 20_000
    return 1.0 if acct(s).get("items_read", 0) <= max(400, int(0.02 * n)) else 0.0


CHECKS = [
    Check("search_rounds", _search_rounds, hard=True, doc=">= 2 root turns run a corpus search"),
    Check("batched_screen", _batched_screen, hard=True,
          doc="a shortlist read through llm_query_batched, <= 8 docs per prompt, >= 8 docs"),
    Check("final_verified", clue_support, hard=True,
          doc="every clue confirmed for the target's chain (sub-call YES or printed evidence)"),
    no_aggregate_requests(hard=True),
    Check("not_whole_corpus", _not_whole_corpus, hard=True, doc="<= max(400, 2% of the corpus) docs read"),
    no_overlong(hard=False),
]

__all__ = ["CHECKS", "HINTS", "PROGRAM", "PROGRAM_RANK", "PROGRAM_READ", "PROGRAM_RETRY", "PROGRAM_SETUP", "STRATEGY",
           "clue_support"]
