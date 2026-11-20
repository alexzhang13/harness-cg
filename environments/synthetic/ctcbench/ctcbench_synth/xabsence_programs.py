"""Reference programs for the synthetic ``xabsence_paraphrase`` family (scripted root, ```repl turns)."""

from __future__ import annotations

from synth_common.scripted import repl

PROG_SHA256 = "12a2dc49c0744b5ade310ac9697bcc24e16f84a3523fa91ca57d5704b1f91ef3"
V5_CFG = {'K': 10, 'K2': 25, 'QPC': 6, 'verify_second': True, 'residual': 'mutual', 'chunk_scan': 60, 'verify_mutual': 120, 'confirm_picks': 'second'}

XAB_CORE = r'''
import collections, json, math, re
STOP = set("the a an of and or in on to for with by is are was were be been that this these those as at from it its "
           "which we our their than not no".split())


def stem(w):
    for suf in ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def toks(s):
    return [stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


def parse_claims(context):
    out = {}
    for m in re.finditer(r"(?m)^\[(\d+)\] ([AB]): (.*)$", context):
        out[int(m.group(1))] = (m.group(2), m.group(3).strip())
    return out


def tfidf(claims):
    T = {i: collections.Counter(toks(t)) for i, (_, t) in claims.items()}
    df = collections.Counter(w for c in T.values() for w in c)
    N = len(T)
    V = {}
    for i, c in T.items():
        v = {w: (1 + math.log(n)) * math.log((N + 1) / (df[w] + 1) + 1) for w, n in c.items()}
        z = math.sqrt(sum(x * x for x in v.values())) or 1.0
        V[i] = {w: x / z for w, x in v.items()}
    return V


def cross_sims(claims, V):
    """sims[i] = sorted [(cos, j)] over the other corpus, best first."""
    inv = collections.defaultdict(list)
    for j, v in V.items():
        for w, x in v.items():
            inv[w].append((j, x))
    sims = {}
    for i, v in V.items():
        acc = collections.defaultdict(float)
        for w, x in v.items():
            for j, y in inv[w]:
                if claims[j][0] != claims[i][0]:
                    acc[j] += x * y
        others = [j for j in claims if claims[j][0] != claims[i][0]]
        sims[i] = sorted(((acc.get(j, 0.0), j) for j in others), reverse=True)
    return sims


def jparse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


PROMPT = """Two corpora of short scientific claims, A and B. Most claims in one corpus have a PARAPHRASE in the other: a restatement of the SAME fact or sentence in different words (same subjects, same finding, same numbers). A claim on a related topic, or about the same study but stating a different fact, is NOT a paraphrase.

For each target claim below, look at its candidates from the other corpus and decide which candidate is its paraphrase, or NONE if no candidate restates the same fact.

{items}

Reply with one JSON object mapping each target id to the id of its paraphrase or to "NONE", e.g. {{"12": 407, "15": "NONE"}}. No other text."""


def item_block(i, claims, cands):
    side, text = claims[i]
    lines = [f"TARGET [{i}] ({side}): {text}", "Candidates:"]
    lines += [f"  [{j}] {claims[j][1]}" for j in cands]
    return "\n".join(lines)


def ask(lm, claims, targets, cand_of, qpc):
    """returns {target: partner id or None(=NONE)}, plus usage list."""
    prompts, groups = [], []
    for k in range(0, len(targets), qpc):
        g = targets[k:k + qpc]
        groups.append(g)
        prompts.append(PROMPT.format(items="\n\n".join(item_block(i, claims, cand_of[i]) for i in g)))
    outs = lm.batch(prompts)
    res, usage = {}, []
    for g, (text, u) in zip(groups, outs):
        usage.append(u)
        d = jparse(text) or {}
        for i in g:
            v = d.get(str(i), d.get(i))
            if isinstance(v, str) and v.strip().upper().startswith("NONE"):
                res[i] = None
            else:
                try:
                    v = int(str(v).strip("[] "))
                    res[i] = v if v in cand_of[i] else "invalid"
                except (TypeError, ValueError):
                    res[i] = "invalid"
    return res, usage


PAIRS = """Each line below pairs a claim from corpus A with a claim from corpus B. For each pair, answer YES if the two claims are PARAPHRASES (a restatement of the SAME fact or sentence in different words: same subjects, same finding, same numbers) and NO otherwise (related topic, same study but a different fact, or different statements).

{items}

Reply with one JSON object mapping each pair number to "YES" or "NO", e.g. {{"1": "YES", "2": "NO"}}. No other text."""

CHUNK = """TARGET claims (each from corpus A or B) and a LIST of claims from the other corpus. Most claims have exactly one PARAPHRASE in the other corpus: a restatement of the SAME fact or sentence in different words (same subjects, same finding, same numbers). A claim on a related topic, or about the same study but stating a different fact, is NOT a paraphrase. The paraphrase may use completely different words (synonyms, spelled-out abbreviations).

TARGETS:
{targets}

LIST:
{chunk}

For each target, give the id of its paraphrase if it is in the LIST, else "NONE". Reply with one JSON object, e.g. {{"12": 407, "15": "NONE"}}. No other text."""


def yesno(lm, claims, pairs, per_call):
    """{(a, b): True/False/None} over unordered pairs, plus usage."""
    prompts, groups = [], []
    for k in range(0, len(pairs), per_call):
        g = pairs[k:k + per_call]
        groups.append(g)
        items = "\n".join(f"{n + 1}. A: {claims[a][1]}\n   B: {claims[b][1]}"
                          for n, (a, b) in enumerate((p if claims[p[0]][0] == "A" else p[::-1]) for p in g))
        prompts.append(PAIRS.format(items=items))
    res, usage = {}, []
    for g, (text, u) in zip(groups, lm.batch(prompts)):
        usage.append(u)
        d = jparse(text) or {}
        for n, p in enumerate(g):
            v = str(d.get(str(n + 1), "")).strip().upper()
            res[p] = True if v.startswith("Y") else False if v.startswith("N") else None
    return res, usage


def chunk_scan(lm, claims, targets, chunk_size):
    """every target vs the WHOLE other corpus, in chunks; returns {target: [found ids]}, usage."""
    found, usage, prompts, meta = collections.defaultdict(list), [], [], []
    for side in ("A", "B"):
        tg = [i for i in targets if claims[i][0] == side]
        if not tg:
            continue
        other = [j for j in claims if claims[j][0] != side]
        for k in range(0, len(other), chunk_size):
            ch = other[k:k + chunk_size]
            prompts.append(CHUNK.format(targets="\n".join(f"[{i}] ({side}): {claims[i][1]}" for i in tg),
                                        chunk="\n".join(f"[{j}] {claims[j][1]}" for j in ch)))
            meta.append((tg, set(ch)))
    for (tg, ch), (text, u) in zip(meta, lm.batch(prompts)):
        usage.append(u)
        d = jparse(text) or {}
        for i in tg:
            try:
                v = int(str(d.get(str(i), "NONE")).strip("[] "))
            except ValueError:
                continue
            if v in ch:
                found[i].append(v)
    return found, usage


DEFAULT = dict(K=10, K2=25, QPC=6, verify_second=True)


def run_program(context, lm, cfg=DEFAULT, trace=None):
    claims = parse_claims(context)
    V = tfidf(claims)
    sims = cross_sims(claims, V)
    nn = {i: sims[i][0][1] for i in claims}
    mutual = {i for i in claims if nn[nn[i]] == i}
    usage0, rejected = [], []
    if cfg.get("verify_mutual"):
        # an orphan can be the mutual nearest neighbour of a claim whose real paraphrase is lexically far:
        # check the weakest mutual pairs and send the rejected ones to the shortlist
        mp = sorted({(min(i, nn[i]), max(i, nn[i])) for i in mutual}, key=lambda p: sims[p[0]][0][0])
        vm, usage0 = yesno(lm, claims, mp[:cfg["verify_mutual"]], cfg.get("PPC", 20))
        rejected = [p for p, ok in vm.items() if ok is False]
        for p in rejected:
            mutual -= set(p)
    R = sorted(i for i in claims if i not in mutual)
    cand1 = {i: [j for _, j in sims[i][:cfg["K"]]] for i in R}
    r1, u1 = ask(lm, claims, R, cand1, cfg["QPC"])
    # a claim chosen as partner by some other shortlisted claim is matched
    usage = list(usage0) + list(u1)
    if cfg.get("confirm_picks") == "both":
        # a pick from a candidate list is a guess; keep it only if an independent pairwise question agrees
        picks = sorted({(min(i, v), max(i, v)) for i, v in r1.items() if isinstance(v, int)})
        vp, us = yesno(lm, claims, picks, cfg.get("PPC", 20))
        usage += us
        r1 = {i: (v if not isinstance(v, int) or vp.get((min(i, v), max(i, v))) else None) for i, v in r1.items()}
    chosen = {v for v in r1.values() if isinstance(v, int)}
    none1 = [i for i in R if r1[i] is None and i not in chosen]
    if cfg.get("verify_second") and none1:
        cand2 = {i: [j for _, j in sims[i][:cfg["K2"]]][::-1] for i in none1}   # wider list, reversed order
        r2, u2 = ask(lm, claims, none1, cand2, max(1, cfg["QPC"] // 2))
        usage += u2
        if cfg.get("confirm_picks"):
            picks = sorted({(min(i, v), max(i, v)) for i, v in r2.items() if isinstance(v, int)})
            vp, us = yesno(lm, claims, picks, cfg.get("PPC", 20))
            usage += us
            r2 = {i: (v if not isinstance(v, int) or vp.get((min(i, v), max(i, v))) else None) for i, v in r2.items()}
        unmatched = [i for i in none1 if r2[i] is None]
    else:
        r2, unmatched = {}, none1
    r3, removed = {}, []
    if cfg.get("residual") and len(unmatched) > 1:
        # a heavily reworded pair is lexically far in BOTH directions, so both halves land in the NONE set: match the
        # NONE set against itself (each A target vs every B leftover, each B target vs every A leftover)
        U = list(unmatched)
        cand3 = {i: [j for j in U if claims[j][0] != claims[i][0]] for i in U}
        tg = [i for i in U if cand3[i]]
        r3, u3 = ask(lm, claims, tg, cand3, cfg.get("QPC3", 4))
        usage += u3
        hit = set()
        for i, v in r3.items():
            if isinstance(v, int):
                hit |= {i, v} if cfg.get("residual") == "either" else ({i, v} if r3.get(v) == i else set())
        removed = sorted(hit)
        unmatched = [i for i in U if i not in hit]
    scan, confirmed = {}, []
    if cfg.get("chunk_scan") and unmatched:
        # the paraphrase may share no words with its target: scan the whole other corpus in short lists, then
        # confirm every proposed partner with an independent pairwise question
        scan, us = chunk_scan(lm, claims, unmatched, cfg["chunk_scan"])
        usage += us
        props = sorted({(min(i, j), max(i, j)) for i, js in scan.items() for j in js})
        if props:
            vp, us = yesno(lm, claims, props, cfg.get("PPC", 20))
            usage += us
            confirmed = [p for p, ok in vp.items() if ok]
        gone = {i for p in confirmed for i in p}
        unmatched = [i for i in unmatched if i not in gone]
    if trace is not None:
        trace.update(n_claims=len(claims), n_mutual=len(mutual), shortlist=R, r1=r1, none1=none1, r2=r2,
                     r3=r3, removed=removed, rejected_mutual=rejected, scan=dict(scan), confirmed=confirmed,
                     unmatched=unmatched)
    ans = "Unmatched: " + ", ".join(f"[{i}]" for i in sorted(unmatched))
    return ans, usage


import types

def _batch(prompts):
    outs = llm_query_batched(list(prompts))
    return [(o or "", {"prompt": 0, "completion": 0, "finish": "stop", "cached": False}) for o in outs]

lm = types.SimpleNamespace(batch=_batch)   # the REPL has no class statement; prog.py only needs lm.batch
claims_ = parse_claims(context)
print("claims", len(claims_), "A", sum(1 for s, _ in claims_.values() if s == "A"))
'''

XAB_SIMPLE = r'''
def run_simple(name, context, lm):
    claims = parse_claims(context)
    V = tfidf(claims)
    sims = cross_sims(claims, V)
    if name == "lex3":
        un = sorted(claims, key=lambda i: sims[i][0][0])[:3]
    elif name == "full":
        nn = {i: sims[i][0][1] for i in claims}
        mutual = {i for i in claims if nn[nn[i]] == i}
        R = sorted(i for i in claims if i not in mutual)
        cand = {i: [j for j in claims if claims[j][0] != claims[i][0]] for i in R}
        r, _ = ask(lm, claims, R, cand, 6)
        chosen = {v for v in r.values() if isinstance(v, int)}
        un = [i for i in R if r[i] is None and i not in chosen]
    elif name == "naive":
        pairs = [(i, sims[i][0][1]) for i in sorted(claims)]
        outs = lm.batch([PAIRS.format(items=f"1. A: {claims[a if claims[a][0] == 'A' else b][1]}\n   B: "
                                           f"{claims[b if claims[a][0] == 'A' else a][1]}") for a, b in pairs])
        un = [a for (a, b), (text, u) in zip(pairs, outs)
              if not str((jparse(text) or {}).get("1", "")).strip().upper().startswith("Y")]
    return "Unmatched: " + ", ".join(f"[{i}]" for i in sorted(un))
'''


def program(cfg: dict) -> list[str]:
    return [repl(XAB_CORE), repl(f"""
ans, usage = run_program(context, lm, {cfg!r})
print(ans, "| sub-calls", len(usage))
answer["content"] = ans
answer["ready"] = True
""")]


def simple(name: str) -> list[str]:
    return [repl(XAB_CORE + XAB_SIMPLE), repl(f"""
ans = run_simple({name!r}, context, lm)
print(ans)
answer["content"] = ans
answer["ready"] = True
""")]


XAB_VARIANTS = {
    "v5": program(V5_CFG),
    "v3b": program({**V5_CFG, "confirm_picks": None}),
    "v3a": program({k: v for k, v in V5_CFG.items() if k not in ("verify_mutual", "confirm_picks")}),
    "v4": program({**V5_CFG, "confirm_picks": "both"}),
    "v2": program({"K": 10, "K2": 25, "QPC": 6, "verify_second": True, "residual": "mutual"}),
    "v1": program({"K": 10, "K2": 25, "QPC": 6, "verify_second": True}),
    "full": simple("full"),
    "naive": simple("naive"),
    "lex3": simple("lex3"),
}
XABSENCE_PARAPHRASE = XAB_VARIANTS["v5"]
