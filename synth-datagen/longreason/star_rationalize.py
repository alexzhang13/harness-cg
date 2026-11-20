"""STaR rationalization for items no sample solved: add the gold letter as a hint, keep replies that reach it
without revealing it. The training row uses the original prompt.

    python star_rationalize.py <star.jsonl> <out.jsonl> [tries=2]"""
import sys, json, re, collections, concurrent.futures as cf
import os
if {'-h', '--help'} & set(sys.argv[1:]):
    sys.exit(print(__doc__))
star, out = sys.argv[1], sys.argv[2]; tries = int(sys.argv[3]) if len(sys.argv) > 3 else 2
import urllib.request, itertools
PORTS = os.environ.get('LR_PORTS', 'localhost:8000').split(','); _rr = itertools.count()
def call(prompt):
    body = {"model": os.environ.get("LR_MODEL", "qwen38-27b"), "messages": [{"role": "user", "content": prompt}], "max_tokens": 24000, "temperature": 0.8,
            "chat_template_kwargs": {"enable_thinking": True}}
    j = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://{PORTS[next(_rr) % len(PORTS)]}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600).read())
    m = j['choices'][0]['message']; return (m.get('reasoning_content') or m.get('reasoning') or ''), (m.get('content') or ''), j['choices'][0].get('finish_reason')
HINT = ("\n\n(For this exercise you are told that the correct answer is {g}. Work through the passages and the procedure as if solving "
        "from scratch so that a reader can see WHY {g} is right and each other option is wrong. Never mention that you were told the "
        "answer, and do not refer to this note.)")
LEAK = re.compile(r"told|given (that )?the (correct )?answer|the note|hint|exercise|we know the answer|answer is supposed", re.I)
LET = re.compile(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\**\(?([A-H])\b")
by = collections.defaultdict(list)
for l in open(star): r = json.loads(l); by[r['id']].append(r)
zero = [(i, v[0]['prompt'], v[0]['gold']) for i, v in by.items() if not any(x['correct'] for x in v)]
print(len(zero), 'items at 0/K', flush=True)
def run(j):
    i, p, g = j
    for t in range(tries):
        try: rs, ct, fin = call(p + HINT.format(g=g))
        except Exception: continue
        f = LET.findall(ct)
        if f and f[-1] == g and fin == 'stop' and not LEAK.search(ct):
            body = ct.rsplit('Answer:', 1)[0].strip()
            if len(body) < 300: continue
            return {'id': i, 'gold': g, 'prompt': p, 'reasoning': body, 'content': f"Answer: {g}", 'rationalized': True}
    return None
with open(out, 'w') as fo, cf.ThreadPoolExecutor(128) as ex:
    n = 0
    for o in ex.map(run, zero):
        if o: fo.write(json.dumps(o) + '\n'); n += 1
print('kept', n)
