"""Build the trick_synth corpus with the base 27B: techniques, ~25 problems each, sketches and fingerprints.

    python tools/gen_corpus.py <ports file> <out corpus.json>

Then run rewrite_corpus.py and fix_corpus.py on the same file."""
import json, re, sys, urllib.request, itertools, concurrent.futures as cf
PORTS=[l.strip() for l in open(sys.argv[1]) if l.strip()]; OUT=sys.argv[2]; rr=itertools.count()
def llm(p, mt=4000, T=0.9):
    for _ in range(4):
        url=f"http://{PORTS[next(rr)%len(PORTS)]}/v1/chat/completions"
        b={"model":"qwen38-27b","messages":[{"role":"user","content":p}],"max_tokens":mt,"temperature":T,"chat_template_kwargs":{"enable_thinking":False}}
        try:
            j=json.loads(urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(b).encode(),headers={"Content-Type":"application/json"}),timeout=900).read())
            return j['choices'][0]['message'].get('content') or ''
        except Exception as e: err=e
    return ''
FAMILIES=["algorithmic / programming", "combinatorics and counting", "number theory and divisibility", "geometry and spatial reasoning",
          "optimization and resource allocation", "probability and expectation", "games and strategy", "inequalities and bounds",
          "sequences, sums and recurrences", "logic puzzles and invariants", "physics-style estimation and conservation", "data and string processing"]
CAT="""List 12 DISTINCT core problem-solving techniques (the decisive 'aha' move) in the family "{fam}". Each must be a reusable trick that works across many unrelated surface stories (e.g. "Telescoping: rewrite each term as a difference so the sum collapses", "Two pointers moving inward over a sorted array", "Invariant: find a quantity no move can change", "Complementary counting"). Avoid overlapping techniques.
Output exactly 12 lines: <short name> :: <one-sentence domain-independent description>"""
PROB="""Technique (HIDDEN, never name it or its key words): {name} -- {desc}

Write 25 short, self-contained problems (2-4 sentences each) that are each SOLVED by exactly this technique, but whose surface stories are as different as possible: vary the setting (warehouses, sports leagues, music playlists, chemistry labs, board games, city planning, ecology, finance, social networks, cooking, travel, ...), the objects and the numbers. Do NOT mention the technique, its name or its telltale vocabulary. After each problem give a one-sentence solution sketch that shows HOW the technique applies to that specific problem.
Format exactly, 25 times:
P: <problem>
S: <solution sketch>"""
PARA="""Technique: {name} -- {desc}
Write 6 different one-sentence descriptions of this technique as an expert would write a "fingerprint" of a solution: domain-independent, varied wording, sometimes naming it and sometimes only describing the move. Output 6 lines, nothing else."""
def catalogue():
    outs=list(cf.ThreadPoolExecutor(12).map(lambda f: (f,llm(CAT.format(fam=f),1500,0.7)), FAMILIES))
    T=[]; seen=set()
    for fam,o in outs:
        for line in o.splitlines():
            if '::' in line:
                n,d=[x.strip(' -*0123456789.') for x in line.split('::',1)]
                if n and d and n.lower() not in seen: seen.add(n.lower()); T.append({'name':n,'desc':d,'family':fam})
    return T
def problems(t):
    o=llm(PROB.format(name=t['name'],desc=t['desc']),6000,0.9)
    items=re.findall(r'P:\s*(.+?)\s*\nS:\s*(.+?)(?=\nP:|\Z)', o, re.S)
    return [{'text':p.strip(),'sketch':s.strip()} for p,s in items if len(p)>40]
def paras(t):
    o=llm(PARA.format(name=t['name'],desc=t['desc']),800,0.8)
    L=[l.strip(' -*0123456789.') for l in o.splitlines() if len(l.strip())>20]
    return L[:6] or [t['desc']]
if __name__=='__main__':
    T=catalogue(); print('techniques',len(T),flush=True)
    with cf.ThreadPoolExecutor(96) as ex:
        P=list(ex.map(problems,T)); Q=list(ex.map(paras,T))
    items=[]
    for k,(t,ps,fp) in enumerate(zip(T,P,Q)):
        t['id']=k; t['fingerprints']=fp
        for j,p in enumerate(ps): items.append({'iid':f'{k}_{j}','tech':k,'text':p['text'],'sketch':p['sketch']})
    json.dump({'techniques':T,'items':items},open(OUT,'w'))
    print('items',len(items),'per tech',sorted(len(p) for p in P)[:5],'...',sorted(len(p) for p in P)[-3:])
