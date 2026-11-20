"""Drop grouping seed-pool papers whose title or abstract hits the audit index; writes pool_decontam.json for
compose.py. Run with PYTHONHASHSEED=0."""
import sys, gzip, json, os
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'audit'))
from common import Index, grams, ekey, seed_pool, DATAGEN
OUT = os.path.join(DATAGEN, 'grouping32')
IX = None
def init():
    global IX; IX = Index()
def render_abs(a):
    w = a.split(); return a if len(w) <= 120 else ' '.join(w[:120]) + ' ...'
def scan(p):
    hits = {}
    for fld, t in (('title', p['title'] or ''), ('abstract', p['abstract'] or ''), ('rendered', render_abs(p['abstract'] or ''))):
        e = IX.exact(ekey(t))
        if e and fld != 'title' or (e and len(t.split()) >= 6): hits[fld + ':exact'] = sorted(e)
        h = grams(t); g = IX.ghits(h)
        if g: hits[fld + ':13g'] = sorted(f'{f}|{lv}' for (f, lv) in g)
    return p['id'], hits
if __name__ == '__main__':
    src = seed_pool('grouping_labeled.seed.jsonl.gz')
    f = gzip.open(src, 'rt'); f.readline(); papers = json.loads(f.readline())['papers']
    with Pool(16, initializer=init) as P: res = P.map(scan, papers, chunksize=500)
    bad = {i: h for i, h in res if h}
    os.makedirs(OUT, exist_ok=True)
    json.dump({'source': src, 'n_papers': len(papers), 'n_contaminated': len(bad), 'contaminated': bad}, open(f'{OUT}/pool_decontam.json', 'w'))
    import collections
    print(len(papers), 'papers;', len(bad), 'contaminated;', collections.Counter(k for h in bad.values() for k in h).most_common(8))
    print(collections.Counter(x.split('|')[0] for h in bad.values() for v in h.values() for x in v).most_common(10))
