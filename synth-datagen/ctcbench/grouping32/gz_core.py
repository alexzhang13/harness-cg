"""GZ program core (the code the note's blocks contain), for the scripted reference and tests.
items: {id(str): (title, abstract)}; K: requested groups."""
import re, math, json, collections
import numpy as np
STOP = set('the and for with that this from are was were have has been into their its our which these those using used '
           'based study studies results result paper also than between both such can may more most other new'.split())
def stem(w): return re.sub(r'(ies|es|s)$', '', w) if len(w) > 3 else w
def lexical(items):
    ids = list(items); n = len(ids)
    words = {i: [t for t in re.findall(r'[a-z][a-z0-9\-]{2,}', ((items[i][0] + ' ') * 3 + items[i][1]).lower()) if t not in STOP] for i in ids}
    df = collections.Counter(t for i in ids for t in set(words[i]))
    vec = {}
    for i in ids:
        v = {t: (1 + math.log(c)) * math.log(n / df[t]) for t, c in collections.Counter(words[i]).items() if df[t] < n * .5}
        z = math.sqrt(sum(x * x for x in v.values())) or 1; vec[i] = {t: x / z for t, x in v.items()}
    sim = np.eye(n)
    for a in range(n):
        for b in range(a):
            u, v = vec[ids[a]], vec[ids[b]]
            if len(u) > len(v): u, v = v, u
            sim[a, b] = sim[b, a] = sum(w * v.get(t, 0) for t, w in u.items())
    return ids, sim
def phrase_bonus(items, ids, sim, vocab, bonus=0.1):
    V = {' '.join(stem(w) for w in re.findall(r'[a-z0-9]+', re.sub(r'\(.*?\)', '', x.lower()))) for x in vocab.split('|')} - {''}
    L = max(len(v.split()) for v in V)
    def found(text):
        t = [stem(w) for w in re.findall(r'[a-z0-9]+', text.lower())]
        return {' '.join(t[j:j + k]) for k in range(1, L + 1) for j in range(len(t) - k + 1)} & V
    ph = {i: (found(items[i][0]), found(items[i][0] + ' ' + items[i][1])) for i in ids}
    df = collections.Counter(p for i in ids for p in ph[i][1])
    s = sim.copy()
    for a in range(len(ids)):
        for b in range(a):
            A, B = ph[ids[a]], ph[ids[b]]; sh = A[1] & B[1]
            if sh: s[a, b] = s[b, a] = s[a, b] + bonus * max((1 + (p in A[0]) + (p in B[0])) / df[p] for p in sh)
    return s
def proposal_prompt(items, order, K):
    n = len(order); M = n - K
    body = '\n'.join(f'[{i}] {items[i][0]} -- {items[i][1][:400]}' for i in order)
    return (f'Below are {n} scientific papers. An automatic indexer (OpenAlex) tagged every paper with ONE fine-grained '
            f'concept, and the papers fall into exactly {K} concept groups. So only about {M} papers share a concept with '
            f'another paper; most papers are alone. Find the groups of 2 or more papers that most likely received the SAME '
            f'concept. The concept is often a specific subject, material, method or phenomenon (e.g. "Photocatalysis", '
            f'"Dark energy", "Dysbiosis"), not a broad field. List up to {M + 10} groups, strongest first.\n\n{body}\n\n'
            'Return JSON only: {"groups": [{"concept": "<name>", "ids": ["<paper id>", ...]}, ...]}')
def votes(replies, ids):
    pos = set(ids); v = collections.Counter()
    for rep in replies:
        m = re.search(r'\{.*\}', rep or '', re.S); seen = set()
        try: gs = json.loads(m.group(0)).get('groups', []) if m else []
        except Exception: gs = []
        for g in gs:
            x = sorted({str(i) for i in (g.get('ids') or []) if str(i) in pos})
            for a, p in enumerate(x):
                for q in x[a + 1:]:
                    if (p, q) not in seen: seen.add((p, q)); v[(p, q)] += 1
    return v
def assemble(ids, s, v, K, R=3):
    from scipy.cluster.hierarchy import linkage, cut_tree
    from scipy.spatial.distance import squareform
    idx = {i: n for n, i in enumerate(ids)}; s = s.copy(); b0 = s.copy()
    for (p, q), c in v.items():
        a, b = idx[p], idx[q]; s[a, b] = s[b, a] = b0[a, b] * (1 + c / R) + 0.05 * c / R
    d = np.maximum(0, 1 - s); np.fill_diagonal(d, 0)
    lab = cut_tree(linkage(squareform(d, checks=False), method='average'), n_clusters=[K]).ravel()
    return [[i for i, l in zip(ids, lab) if l == c] for c in sorted(set(lab))]
