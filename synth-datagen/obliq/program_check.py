"""Score fixed candidate programs on trick_synth with the fake sub-LM; describe -> flag should rank first, as on
the real task.

    python program_check.py [n_seeds=10] [split=train]"""
import sys, re, math, collections, statistics as st
if {'-h', '--help'} & set(sys.argv[1:]):
    sys.exit(print(__doc__))
from trick_synth import env as E, oracle as O
N=int(sys.argv[1]) if len(sys.argv)>1 else 10; SPLIT=sys.argv[2] if len(sys.argv)>2 else 'train'
orc=O.TrickOracle()
TOK=re.compile(r'[a-z]+')
def tfidf_rank(docs,q):
    tf={i:collections.Counter(TOK.findall(t.lower())) for i,t in docs.items()}; df=collections.Counter(w for c in tf.values() for w in c); n=len(docs)
    idf={w:math.log(n/c) for w,c in df.items()}; qv=collections.Counter(TOK.findall(q.lower()))
    def s(c): return sum(qv[w]*c[w]*idf.get(w,0)**2 for w in qv)/(1+math.sqrt(sum((x*idf.get(w,0))**2 for w,x in c.items())))
    return sorted(docs,key=lambda i:-s(tf[i]))
def scores(reply, labs):
    out={}
    for line in reply.splitlines():
        m=re.match(r'\s*\[?([A-Za-z0-9]+)\]?\s*[:.)]\s*(\d+)',line)
        if m and m.group(1) in labs: out[labs[m.group(1)]]=int(m.group(2))
    return out
def run(seed):
    g=E.generate({'synth_split':SPLIT,'synth_seed':seed,'synth_params':'{}'})
    docs={e.split('\n')[1].split(': ',1)[1]: e.split('\n---\n',1)[1] for e in g.context}
    q=g.root_prompt.split('Given the following problem:\n\n')[1].split('\n\nFind other problems')[0]
    hidden=g.hidden; stt=collections.defaultdict(int); calls=collections.Counter()
    def llm(p,tag): calls[tag]+=1; return orc.respond(p,hidden,stt)
    code=tfidf_rank(docs,q); res={'code':code}
    sc={}
    for k in range(0,len(code),30):
        b=code[k:k+30]; labs={i:i for i in b}
        p=f"Query problem:\n{q}\n\nScore each problem 0-10 for whether it uses the same core technique. One line per doc: <id>: <score>\n\n"+"\n\n".join(f"{i}: {docs[i]}" for i in b)
        sc.update(scores(llm(p,'score_all'),labs))
    res['score_all']=sorted(docs,key=lambda i:(-sc.get(i,-1)))
    yn={}
    for k in range(0,300,10):
        b=code[k:k+10]; p=f"Query problem:\n{q}\n\nFor each problem: does it require the SAME core technique? Answer YES or NO per line: <id>: YES/NO\n\n"+"\n\n".join(f"{i}: {docs[i]}" for i in b)
        for line in llm(p,'yesno').splitlines():
            m=re.match(r'\s*([A-Za-z0-9]+):\s*(YES|NO)',line)
            if m and m.group(1) in docs: yn[m.group(1)]=m.group(2)=='YES'
    res['yesno_top300']=[i for i in code[:300] if yn.get(i)]+[i for i in code if not yn.get(i)]
    desc={}
    for k in range(0,len(code),40):
        b=code[k:k+40]; p="For each numbered problem, work out how you would solve it, then output one line: [k] SKETCH: <1-2 sentences> | FINGERPRINT: <one sentence naming the core trick>\n\n"+"\n\n".join(f"[{j+1}] {docs[i]}" for j,i in enumerate(b))
        for line in llm(p,'map').splitlines():
            m=re.match(r'\s*\[(\d+)\]\s*SKETCH:(.*?)\|\s*FINGERPRINT:(.*)',line)
            if m and 1<=int(m.group(1))<=len(b): desc[b[int(m.group(1))-1]]=(m.group(2).strip(),m.group(3).strip())
    qr=llm(f"Work out how to solve this problem, then output SKETCH: <2-3 sentences> and FINGERPRINT: <one sentence naming the core trick>:\n\n{q}",'map')
    qf=re.search(r'FINGERPRINT:(.*)',qr); qs=re.search(r'SKETCH:(.*)',qr); qf=qf.group(1).strip() if qf else ''; qs=qs.group(1).strip() if qs else ''
    fl=set()
    for k in range(0,len(code),60):
        b=code[k:k+60]
        p=f"Query solution idea: {qs}\nQuery core trick:\n{qf}\n\nWhich candidates could share the SAME core trick, even partly? Be generous: missing one is worse than listing extra. Reply FLAG: <numbers>\n\n"+"\n".join(f"{j+1}. META: {desc.get(i,('',''))[1]} | IDEA: {desc.get(i,('',''))[0]}" for j,i in enumerate(b))
        r=llm(p,'flag'); m=re.search(r'FLAG:\s*([\d,\s]+)',r)
        if m: fl|={b[int(x)-1] for x in re.findall(r'\d+',m.group(1)) if 1<=int(x)<=len(b)}
    res['map_flag']=[i for i in code if i in fl]+[i for i in code if i not in fl]
    out={k:E._ndcg(v,g.gold) for k,v in res.items()}
    out['desc_cov']=len(desc)/len(docs); out['flagged']=len(fl); out['flag_rec']=sum(g_ in fl for g_ in g.gold)/max(1,len(g.gold)); out['calls']=sum(calls.values())
    return out
outs=[run(s) for s in range(N)]
for k in outs[0]:
    print(SPLIT,k,round(st.mean(o[k] for o in outs),3))
