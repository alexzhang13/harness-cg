"""Rewrite each item into a different setting with the same technique, to cut lexical overlap within a technique.

    python tools/rewrite_corpus.py <ports file> <corpus.json>      (in place, after gen_corpus.py)"""
import json, re, sys, random, concurrent.futures as cf
sys.path.insert(0, __file__.rsplit('/',1)[0]); import gen_corpus as G
G.PORTS[:] = [l.strip() for l in open(sys.argv[1]) if l.strip()]
F=sys.argv[2]; C=json.load(open(F)); I=C['items']
SET=["a hospital","a space station","a medieval guild","a bakery","a jazz band","a soccer league","a beehive","a train network",
     "a library","an online game","a vineyard","a fishing fleet","a theater","a coral reef","a courier company","a chess club",
     "a weather station","a school timetable","a mining colony","a film festival","a robot factory","a hiking club","a stock exchange","a zoo"]
P="""Rewrite each numbered problem so that it is set in {setting} (or another setting far from the original), with completely different objects, names, units and numbers, and avoid reusing the original's content words. Keep the underlying mathematical/logical structure EXACTLY, so the same solution technique applies. Then give a one-sentence solution sketch for the REWRITTEN problem.
{items}
Format, one block per problem in order:
[k] P: <rewritten problem>
S: <solution sketch>"""
def run(chunk):
    r=random.Random(chunk[0])
    items="\n".join(f"[{j+1}] {I[k]['text']}" for j,k in enumerate(chunk))
    o=G.llm(P.format(setting=r.choice(SET),items=items),5000,0.9)
    out={}
    for m in re.finditer(r'\[(\d+)\]\s*P:\s*(.+?)\s*\nS:\s*(.+?)(?=\n\s*\[\d+\]|\Z)',o,re.S):
        j=int(m.group(1))-1
        if 0<=j<len(chunk) and len(m.group(2))>40: out[chunk[j]]=(m.group(2).strip(),m.group(3).strip())
    return out
idx=list(range(len(I))); random.Random(3).shuffle(idx)
chunks=[idx[k:k+12] for k in range(0,len(idx),12)]
done={}
with cf.ThreadPoolExecutor(96) as ex:
    for o in ex.map(run,chunks): done.update(o)
for k,(p,s) in done.items():
    I[k]['orig_text']=I[k]['text']; I[k]['text']=p; I[k]['sketch']=s
json.dump(C,open(F,'w')); print('rewritten',len(done),'of',len(I))
