"""Synthetic grouping@32k rows in the eval shape, from the decontaminated grouping seed pool.

Matches the eval rung's structure (176 papers, mostly singletons and pairs, groups share one OpenAlex concept).
Papers are split 90/10 by id hash into train / dev pools.
"""
import ast, gzip, hashlib, json, os, random, collections
OUT = os.path.join(os.environ.get("CTC_DATAGEN_DIR") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../outputs/synth-datagen/ctcbench")), "grouping32")
QUERY = ["Partition the following abstracts into {k} categories.", "Group these documents into {k} groups.",
         "Cluster these {n} papers into {k} groups."]
def _split_of(pid): return 'dev' if int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16) % 10 == 0 else 'train'
def render_abs(a):
    w = (a or '').split(); return ' '.join(w) if len(w) <= 120 else ' '.join(w[:120]) + ' ...'
_POOL = {}
def pool(split):
    if split in _POOL: return _POOL[split]
    src = __import__('huggingface_hub').hf_hub_download('PrasannSinghal/ctc-seed-pools', 'grouping_labeled.seed.jsonl.gz', repo_type='dataset', revision='a2cbd4c1f0c900fbc8b2941f365b44e25baeead6')
    f = gzip.open(src, 'rt'); f.readline(); papers = json.loads(f.readline())['papers']
    bad = set(json.load(open(f'{OUT}/pool_decontam.json'))['contaminated'])
    train_titles = {d['title'] for l in open((__import__('os').environ.get('CTCBENCH_TRAIN_DIR') or __import__('os').path.expanduser('~/ctcbench_data/train')) + '/grouping_labeled/train.jsonl') for d in json.loads(l)['documents']}
    out = []
    for p in papers:
        if p['id'] in bad or p['title'] in train_titles or _split_of(p['id']) != split or not p.get('title') or len((p.get('abstract') or '').split()) < 40: continue
        c = p['concepts']
        if not isinstance(c, dict):
            try: c = ast.literal_eval(c)
            except Exception: continue
        out.append({'id': p['id'], 'title': p['title'], 'text': render_abs(p['abstract']), 'c': {int(k): v for k, v in c.items()}})
    by = {lv: collections.defaultdict(list) for lv in (2, 3)}
    for i, p in enumerate(out):
        for lv in (2, 3):
            if p['c'].get(lv): by[lv][p['c'][lv]].append(i)
    _POOL[split] = (out, by); return _POOL[split]
def sizes(rng, K, N=176):
    M = N - K; g4 = 1 if rng.random() < 0.13 else 0; s4 = rng.choice([4, 4, 4, 5]) if g4 else 0
    g3 = min(10, sum(rng.random() < 0.38 for _ in range(5)))
    pairs = M - 2 * g3 - (s4 - 1) * g4
    while pairs < 0: g3 -= 1; pairs += 2
    S = [s4] * g4 + [3] * g3 + [2] * pairs; S += [1] * (K - len(S))
    assert len(S) == K and sum(S) == N; return S
def compose(split, k_row, seed=0, N=176):
    rng = random.Random(f'grp32:{split}:{seed}:{k_row}'); out, by = pool(split)
    lv = 2 if rng.random() < 0.3 else 3
    K = rng.randint(132, 150) if lv == 2 else rng.randint(133, 167)
    S = sizes(rng, K, N); used_c, used_p, groups = set(), set(), []
    names = list(by[lv]); rng.shuffle(names)
    for s in sorted(S, reverse=True):
        for _ in range(5000):
            c = names[rng.randrange(len(names))]
            if c in used_c: continue
            cand = [i for i in by[lv][c] if i not in used_p]
            if len(cand) >= s: break
        used_c.add(c); pick = rng.sample(cand, s); used_p.update(pick); groups.append((c, pick))
    order = [i for _, g in groups for i in g]; rng.shuffle(order); pos = {i: n for n, i in enumerate(order)}
    docs = [{'title': out[i]['title'], 'text': out[i]['text']} for i in order]
    gold = [sorted(pos[i] for i in g) for _, g in groups]
    return {'documents': docs, 'queries': [rng.choice(QUERY).format(k=K, n=N)], 'gold_doc_indices': gold,
            'cluster_labels': [c for c, _ in groups], 'level': lv, 'k': K, 'source': 'grouping_synth_openalex_decontam',
            'synth': {'split': split, 'seed': seed, 'row': k_row, 'paper_ids': [out[i]['id'] for i in order]}}
if __name__ == '__main__':
    r = compose('dev', 0); print(r['k'], r['level'], len(r['documents']), collections.Counter(len(g) for g in r['gold_doc_indices']), r['queries'], r['documents'][0]['title'][:80])
    print({s: len(pool(s)[0]) for s in ('train', 'dev')})


def corpus(split, cid, size=600):
    """A 600-paper corpus drawn at random from the clean pool (the eval rows are samples of one such corpus)."""
    out, _ = pool(split); rng = random.Random(f'grp32corpus:{split}:{cid}')
    return rng.sample(range(len(out)), size)
def compose_v2(split, k_row, seed=0, N=176, rows_per_corpus=8):
    """Sample 176 papers like the eval rows (<= 5 per concept), rejecting draws whose concept count is out of range."""
    rng = random.Random(f'grp32v2:{split}:{seed}:{k_row}'); out, _ = pool(split)
    C = [i for i in corpus(split, (seed, k_row // rows_per_corpus)) if out[i]['c'].get(3) and out[i]['c'].get(2)]
    lv = 2 if rng.random() < 0.3 else 3; lo, hi = (132, 150) if lv == 2 else (133, 167)
    for _ in range(2000):
        order = C[:]; rng.shuffle(order); cnt = collections.Counter(); pick = []
        cap = {c: rng.choice([2, 3, 3, 4, 5]) for c in {out[i]['c'][lv] for i in order}}
        for i in order:
            c = out[i]['c'][lv]
            if cnt[c] < cap[c]: cnt[c] += 1; pick.append(i)
            if len(pick) == N: break
        K = len({out[i]['c'][lv] for i in pick})
        if len(pick) == N and lo <= K <= hi: break
    else:
        raise RuntimeError('no K in range')
    rng.shuffle(pick); pos = {i: n for n, i in enumerate(pick)}; gro = collections.defaultdict(list)
    for i in pick: gro[out[i]['c'][lv]].append(pos[i])
    names = list(gro)
    return {'documents': [{'title': out[i]['title'], 'text': out[i]['text']} for i in pick],
            'queries': [rng.choice(QUERY).format(k=K, n=N)], 'gold_doc_indices': [sorted(gro[c]) for c in names],
            'cluster_labels': names, 'level': lv, 'k': K, 'source': 'grouping_synth_openalex_decontam_v2',
            'synth': {'split': split, 'seed': seed, 'row': k_row, 'paper_ids': [out[i]['id'] for i in pick]}}


def compose_v3(split, k_row, seed=0, N=176, a_multi=2.0, a_single=1.0):
    """Like v1, but grouped concepts are drawn ~ frequency^2 and singletons ~ frequency, as in the eval rows."""
    rng = random.Random(f'grp32v3:{split}:{seed}:{k_row}'); out, by = pool(split)
    lv = 2 if rng.random() < 0.3 else 3
    K = rng.randint(132, 150) if lv == 2 else rng.randint(133, 167)
    S = sizes(rng, K, N); names = list(by[lv]); f = [len(by[lv][c]) for c in names]
    wm = [x ** a_multi for x in f]; ws = [x ** a_single for x in f]
    used_c, used_p, groups = set(), set(), []
    for s in sorted(S, reverse=True):
        w = wm if s > 1 else ws
        for _ in range(10000):
            c = rng.choices(names, weights=w)[0]
            if c in used_c: continue
            cand = [i for i in by[lv][c] if i not in used_p]
            if len(cand) >= s: break
        used_c.add(c); pick = rng.sample(cand, s); used_p.update(pick); groups.append((c, pick))
    order = [i for _, g in groups for i in g]; rng.shuffle(order); pos = {i: n for n, i in enumerate(order)}
    return {'documents': [{'title': out[i]['title'], 'text': out[i]['text']} for i in order],
            'queries': [rng.choice(QUERY).format(k=K, n=N)], 'gold_doc_indices': [sorted(pos[i] for i in g) for _, g in groups],
            'cluster_labels': [c for c, _ in groups], 'level': lv, 'k': K, 'source': 'grouping_synth_openalex_decontam_v3',
            'synth': {'split': split, 'seed': seed, 'row': k_row, 'paper_ids': [out[i]['id'] for i in order]}}
