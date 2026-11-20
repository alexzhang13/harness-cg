"""STaR samples for the reasoning sub-call: resample each procedure prompt K times with thinking and grade by the
gold letter. ``LR_PORTS`` / ``LR_MODEL`` pick the server.

    python star_gen.py <out.jsonl> <K> <prompts.jsonl | pipeline dir ...>"""
import sys, json, glob, re, os, concurrent.futures as cf, urllib.request, itertools
from pathlib import Path
if {'-h', '--help'} & set(sys.argv[1:]):
    sys.exit(print(__doc__))
PORTS = os.environ.get('LR_PORTS', 'localhost:8000').split(','); _rr = itertools.count()
REPO = Path(__file__).resolve().parents[2]
from rlm_train.frozen import frozen_path
ITEMS_FILE = str(frozen_path(os.environ.get('LONGREASON_EXPANDED_ITEMS') or REPO / 'environments/synthetic/longreason/longreason_expanded/data/expanded_items.jsonl',
                             'longreason/expanded_items.jsonl'))
ITEMS = {}
for l in open(ITEMS_FILE):
    it = json.loads(l); ITEMS[it['question'].strip()[:150]] = it
LET = re.compile(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\**\(?([A-H])\b")
def call(prompt):
    body = {"model": os.environ.get("LR_MODEL", "qwen38-27b"), "messages": [{"role": "user", "content": prompt}], "max_tokens": 24000, "temperature": 0.8,
            "chat_template_kwargs": {"enable_thinking": True}}
    for _ in range(4):
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://{PORTS[next(_rr) % len(PORTS)]}/v1/chat/completions",
                    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600).read())
            m = j['choices'][0]['message']; return (m.get('reasoning_content') or m.get('reasoning') or ''), (m.get('content') or ''), j['choices'][0].get('finish_reason')
        except Exception as e: err = e
    return '', f'ERR {err}', 'error'
out, K, dirs = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
prompts = {}
if dirs and dirs[0].endswith('.jsonl'):
    for l in open(dirs[0]):
        r = json.loads(l); prompts[r['id']] = (r['prompt'], r['gold'])
    dirs = []
for d in dirs:
    for f in glob.glob(f"{d}/gen*/longreason-expanded-train/subcalls.jsonl"):
        for l in open(f):
            try: s = json.loads(l)
            except Exception: continue
            for c in s.get('sub_calls', []):
                p = c.get('prompt') or ''
                if 'QUESTION:' not in p or 'Answer: <letter>' not in p: continue
                q = p.split('QUESTION:\n', 1)[1].strip()[:150]
                it = ITEMS.get(q)
                if it and it['id'] not in prompts: prompts[it['id']] = (p, it['answer'])
print(f"{len(prompts)} distinct items with a procedure sub-call prompt", flush=True)
done = set()
if os.path.exists(out): done = {(json.loads(l)['id'], json.loads(l)['k']) for l in open(out)}
jobs = [(i, k) for i in prompts for k in range(K) if (i, k) not in done]
def run(j):
    i, k = j; p, gold = prompts[i]; rs, ct, fin = call(p); f = LET.findall(ct)
    pred = f[-1] if f else None
    return {'id': i, 'k': k, 'gold': gold, 'pred': pred, 'correct': pred == gold, 'finish': fin, 'prompt': p, 'reasoning': rs, 'content': ct}
with open(out, 'a') as fo, cf.ThreadPoolExecutor(160) as ex:
    for o in ex.map(run, jobs): fo.write(json.dumps(o) + '\n'); fo.flush()
