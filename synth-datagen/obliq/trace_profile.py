"""Summarise root trajectories (turns, sub-calls, step sequence, errors) to compare against real runs.

    python trace_profile.py NAME=<pipeline gen dir | probe dir | rollouts.jsonl> [...]"""
import json, re, sys, os, glob, statistics as st, collections

STEPS = [
    ('submit', r'answer\[\s*["\']ready["\']\s*\]\s*=\s*True'),
    ('flag', r'FLAG|flagged'),
    ('rank', r'tf-?idf|idf|1\s*/\s*\(\s*60|rrf|fused|rank'),
    ('map', r'SKETCH.*FINGERPRINT|FINGERPRINT.*SKETCH'),
    ('parse', r'docs\s*=|split\(chr\(10\)\)|doc_id'),
    ('inspect', r'print\(.*context|len\(context\)|SHOW_VARS'),
]
def step(code):
    for lab, rx in STEPS:
        if re.search(rx, code, re.I | re.S): return lab
    return 'other'
def steps(code):
    """ALL program steps a block performs, in the order they appear in its code (compact v7 blocks do rank+flag+submit
    in one block; step() alone labels that block 'submit' and hides the rest)."""
    hits = []
    for lab, rx in STEPS:
        m = re.search(rx, code, re.I | re.S)
        if m: hits.append((m.start(), lab))
    return [lab for _, lab in sorted(hits)] or ['other']
def step_seq(blocks):
    seq = [lab for c in blocks for lab in steps(c)]
    return [s for k, s in enumerate(seq) if k == 0 or s != seq[k - 1]]
def in_program_order(dd):
    return all(x in dd for x in ('map', 'rank', 'flag', 'submit')) and dd.index('map') < dd.index('flag') <= len(dd) - 1

def sub_kind(p):
    pl = p.lower()
    if 'flag' in pl and ('meta:' in pl or 'idea:' in pl): return 'flag'
    if re.search(r'\[\d+\]', p) and 'sketch' in pl: return 'map'
    if 'sketch' in pl or 'fingerprint' in pl: return 'query'
    return 'other'
def code_of(t): return '\n'.join(re.findall(r'```repl\n(.*?)```', t, re.S))

def load(spec):
    name, path = spec.split('=', 1); rows = []
    if os.path.isdir(path) and os.path.exists(os.path.join(path, 'subcalls.jsonl')):
        subs = {json.loads(l)['key']: json.loads(l)['sub_calls'] for l in open(os.path.join(path, 'subcalls.jsonl'))}
        for f in ['accepted.jsonl', 'rejected.jsonl']:
            for l in open(os.path.join(path, f)):
                j = json.loads(l); m = j.get('meta', j); key = j.get('key') or m.get('key')
                rows.append({'msgs': j.get('messages') or m.get('messages'), 'reward': float(m['reward']), 'subs': subs.get(key, [])})
    else:
        fs = [path] if path.endswith('.jsonl') else glob.glob(os.path.join(path, '**', 'rollouts.jsonl'), recursive=True)
        for f in fs:
            for l in open(f):
                j = json.loads(l)
                if 'history' in j: rows.append({'msgs': j['history'], 'reward': j['reward'], 'subs': j.get('sub_call_log') or [{'prompt': x.get('prompt_head', '') + x.get('prompt_tail', '')} for x in j.get('subs', [])], 'n_subs': j.get('sub_calls')})
    return name, rows

def profile(rows):
    P = collections.defaultdict(list); seqs = collections.Counter()
    for r in rows:
        A = [m['content'] for m in r['msgs'] if m['role'] == 'assistant' and isinstance(m['content'], str)]
        U = [m['content'] for m in r['msgs'] if m['role'] == 'user' and isinstance(m['content'], str) and m['content'].startswith('REPL output')]
        dedup = step_seq([code_of(a) for a in A])
        seqs[' > '.join(dedup)] += 1
        P['reward'].append(r['reward']); P['turns'].append(len(A))
        P['errors'].append(sum(bool(re.search(r'Traceback|Error:', u)) for u in U))
        P['submitted'].append(any(step(code_of(a)) == 'submit' for a in A))
        P['program_order'].append(in_program_order(dedup))
        code = '\n'.join(code_of(a) for a in A)
        P['root_query_gloss'].append(bool(re.search(r'(core (reasoning|trick|idea)|aha|insight)[^\n]{0,40}[:=]\s*["\'f]?[^\n]{40,}', code, re.I)))
        if r['subs']:
            k = collections.Counter(sub_kind(x['prompt']) for x in r['subs'])
            for kk in ('map', 'flag', 'query', 'other'): P['sub_' + kk].append(k[kk])
            fl = [x for x in r['subs'] if sub_kind(x['prompt']) == 'flag']
            P['flag_items_total'].append(sum(len(re.findall(r'(?m)^\s*\d+\.\s*META:', x['prompt'])) for x in fl))
        P['sub_calls'].append(r.get('n_subs') if r.get('n_subs') is not None else len(r['subs']))
    return P, seqs

if __name__ == '__main__':
    res = {}
    for spec in sys.argv[1:]:
        name, rows = load(spec); res[name] = profile(rows); print(f'{name}: {len(rows)} trajectories')
    keys = ['reward', 'turns', 'sub_calls', 'sub_map', 'sub_flag', 'sub_query', 'sub_other', 'flag_items_total', 'errors', 'submitted', 'program_order', 'root_query_gloss']
    print(f"\n{'feature':18s}" + ''.join(f'{n:>22s}' for n in res))
    for k in keys:
        row = []
        for n, (P, _) in res.items():
            v = P.get(k) or []
            if not v: row.append(f"{'-':>22s}"); continue
            if isinstance(v[0], bool): row.append(f'{sum(v)/len(v):>21.0%} ')
            else: row.append(f'{st.median(v):>10.1f} (mean {st.mean(v):6.1f})')
        print(f'{k:18s}' + ''.join(row))
    for n, (_, seqs) in res.items():
        print(f'\n{n}: top step sequences')
        for s, c in seqs.most_common(5): print(f'  {c:3d}  {s}')
