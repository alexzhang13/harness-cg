"""Assemble the SFT set: program-v1 root traces plus STaR sub-call rows (correct samples, rationalizations, and
some easy items).

    python assemble_v2.py <root.jsonl> <star.jsonl> <rat.jsonl> <out.jsonl> [other_cap=100] [n_easy=300]"""
import sys, json, random, collections
if {'-h', '--help'} & set(sys.argv[1:]):
    sys.exit(print(__doc__))
root_f, star_f, rat_f, out = sys.argv[1:5]; cap = int(sys.argv[5]) if len(sys.argv) > 5 else 100; n_easy = int(sys.argv[6]) if len(sys.argv) > 6 else 300
rng = random.Random(0); rows = []; C = collections.Counter()
R = [json.loads(l) for l in open(root_f)]; rng.shuffle(R); n = collections.Counter()
for r in R:
    if r['procedure'] == 'OTHER' and n['OTHER'] >= cap: continue
    n[r['procedure']] += 1; rows.append({k: r[k] for k in ('messages', 'env', 'uid', 'task')}); C['root'] += 1
by = collections.defaultdict(list)
for l in open(star_f): s = json.loads(l); by[s['id']].append(s)
def sub(s, tag):
    return {'messages': [{'role': 'user', 'content': s['prompt']}, {'role': 'assistant', 'content': s['content'], 'reasoning_content': s['reasoning']}],
            'env': 'longreason-expanded-sub', 'uid': f"lrsub-{s['id']}-{tag}", 'task': 'longreason_sub',
            'chat_template_kwargs': {'enable_thinking': True}}
easy = []
for i, v in by.items():
    good = [s for s in v if s['correct'] and s['finish'] == 'stop']
    k = sum(s['correct'] for s in v)
    if 0 < k < len(v):
        for j, s in enumerate(good[:2]): rows.append(sub(s, f'mixed{j}')); C['sub_mixed'] += 1
    elif k == len(v) and good: easy.append(good[0])
for s in rng.sample(easy, min(n_easy, len(easy))): rows.append(sub(s, 'easy')); C['sub_easy'] += 1
for l in open(rat_f):
    s = json.loads(l); rows.append(sub(s, 'rat')); C['sub_rationalized'] += 1
rng.shuffle(rows)
with open(out, 'w') as fo:
    for r in rows: fo.write(json.dumps(r) + '\n')
print(dict(C), 'root procedures', dict(n))
