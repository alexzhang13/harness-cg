"""Add one-sentence sketches for every item, and decoys (same story, different technique) as hard negatives.

    python tools/fix_corpus.py <ports file> <corpus.json>      (in place, after rewrite_corpus.py)"""
import json, re, sys, random, concurrent.futures as cf
sys.path.insert(0, __file__.rsplit('/',1)[0]); import gen_corpus as G
G.PORTS[:] = [l.strip() for l in open(sys.argv[1]) if l.strip()]
F=sys.argv[2]; C=json.load(open(F)); T=C['techniques']; I=C['items']
fam={}
for t in T: fam.setdefault(t['family'],[]).append(t['id'])
SK="""For each numbered problem, write ONE short sentence (max 30 words) saying how to solve it -- the concrete key step.
{items}
Output exactly one line per problem: [k] <sentence>"""
def sketches(chunk):
    o=G.llm(SK.format(items="\n".join(f"[{j+1}] {I[k]['text']}" for j,k in enumerate(chunk))),2500,0.3); out={}
    for m in re.finditer(r'\[(\d+)\]\s*(.+)',o):
        j=int(m.group(1))-1
        if 0<=j<len(chunk): out[chunk[j]]=m.group(2).strip()[:240]
    return out
DEC="""Original problem (solved by "{a}"):
{text}

Write a NEW problem that keeps the SAME setting, characters, objects and as much of the same vocabulary as possible, but whose solution requires a DIFFERENT technique: "{b}" -- {bdesc}. It must NOT be solvable by "{a}". Do not name either technique. Then one short sentence (max 30 words) on how to solve the new problem.
Format exactly:
P: <new problem>
S: <solution sentence>"""
def decoy(job):
    k,b=job; a=T[I[k]['tech']]['name']
    o=G.llm(DEC.format(a=a,text=I[k]['text'],b=T[b]['name'],bdesc=T[b]['desc']),1200,0.8)
    m=re.search(r'P:\s*(.+?)\s*\nS:\s*(.+)',o,re.S)
    return (k,b,m.group(1).strip(),m.group(2).strip()[:240]) if m and len(m.group(1))>40 else None
idx=list(range(len(I))); rng=random.Random(5); rng.shuffle(idx)
with cf.ThreadPoolExecutor(96) as ex:
    S={}
    for o in ex.map(sketches,[idx[k:k+15] for k in range(0,len(idx),15)]): S.update(o)
    for k,s in S.items(): I[k]['sketch']=s
    print('sketches',len(S),'of',len(I),flush=True)
    jobs=[]
    for k in idx[:int(0.6*len(idx))]:
        others=[x for x in fam[T[I[k]['tech']]['family']] if x!=I[k]['tech']]
        if others: jobs.append((k,rng.choice(others)))
    D=[d for d in ex.map(decoy,jobs) if d]
n=len(I)
for j,(k,b,p,s) in enumerate(D):
    I.append({'iid':f'{b}_d{j}','tech':b,'text':p,'sketch':s,'decoy_of':I[k]['iid']})
json.dump(C,open(F,'w')); print('decoys',len(D),'items now',len(I))
