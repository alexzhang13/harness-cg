"""compose_varied (surf_env) + generator-defect injection on matched pairs' rewrites. Patch with install()."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import os, random, sys
R = _REPO
sys.path[:0] = [R + '/environments/real/CTCBench', R + '/environments/synthetic/ctcbench', R + '/environments/synthetic/synth_common', R + '/src',
                __import__("os").path.dirname(__import__("os").path.abspath(__file__))]
import surf_env as X
import defects
def compose_defect(split, row, seed, decoys=2, per_abstract=3):
    from ctcbench_synth import xabsence_para as XP
    rng = random.Random(f"xabpara-surf:{split}:{row}:{seed}")
    n_pairs = rng.randint(300, 345); n_orph = rng.choice([1, 2, 3, 3, 4, 5, 6])
    abstracts = list(XP.pool(split)); rng.shuffle(abstracts)
    n_base = n_pairs + n_orph - n_orph * decoys
    sents, seen = [], set()
    for a in abstracts:
        for o, p in rng.sample(list(a["pairs"]), min(per_abstract, len(a["pairs"]))):
            o, p = X.PREFIX.sub("", o), X.PREFIX.sub("", p)
            if XP._norm(o) in seen or XP._norm(p) in seen: continue
            seen.update((XP._norm(o), XP._norm(p))); sents.append((a["id"], o, p))
        if len(sents) >= n_base: break
    sents = sents[:n_base]
    orph = rng.sample(range(len(sents)), n_orph); side_of = {k: rng.choice("AB") for k in orph}
    for k in orph:
        aid, o, p = sents[k]
        sents += [(a2, X.PREFIX.sub("", o2), X.PREFIX.sub("", p2)) for a2, o2, p2 in XP._decoys(split, o if side_of[k] == "A" else p, aid, decoys, seen, rng)]
    matched = [k for k in range(len(sents)) if k not in side_of]
    pairs = [[sents[k][1], sents[k][2]] for k in matched]
    drng = random.Random(f"xabpara-defect:{split}:{row}:{seed}")
    cnt = defects.inject(pairs, drng) if drng.random() < float(os.environ.get("DEFECT_SHARE", "1")) else {"clean": True}
    for k, (o, p) in zip(matched, pairs): sents[k] = (sents[k][0], o, p)
    A, B = [], []
    for k, (_aid, o, p) in enumerate(sents):
        if k in side_of: (A if side_of[k] == "A" else B).append((o if side_of[k] == "A" else p, True))
        else: A.append((o, False)); B.append((p, False))
    rng.shuffle(A); rng.shuffle(B)
    docs = [{"corpus": "A", "text": t} for t, _ in A] + [{"corpus": "B", "text": t} for t, _ in B]
    gold = [i for i, (_, o) in enumerate(A + B) if o]
    return {"documents": docs, "queries": [], "gold_doc_indices": gold, "num_pairs": len(sents) - n_orph,
            "num_unmatched": n_orph, "source": "xabsence_synth_pubmed_surface_defects", "defects": cnt}
def install(): X.compose_varied = compose_defect
