"""Filter note-v7 traces into hint-stripped SFT rows.

Keeps submitted traces with reward >= MIN within MAX_TURNS that follow map -> rank -> flag -> submit in order and don't
leak the hint. Published set: MIN 0.05, MAX_TURNS 10.

    python build_sft_v7.py <out.jsonl> <min_reward> <max_turns> <real_gen_dir> <rollout dirs...>"""
import argparse, sys, os, json, re, glob, collections, importlib.util
H = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('tp', os.path.join(H, 'trace_profile.py')); tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
LEAK = re.compile(r"\b(the|my|this|strategy) (note|hint)\b|<hint>|as (instructed|suggested|the note says)", re.I)
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('out'); ap.add_argument('min_reward', type=float); ap.add_argument('max_turns', type=int)
ap.add_argument('real_gen_dir', help='real base+note pipeline gen dir for the profile comparison (printed only; may be absent)')
ap.add_argument('rollout_dirs', nargs='+')
args = ap.parse_args()
out, MIN, MAXT, real = args.out, args.min_reward, args.max_turns, args.real_gen_dir
C = collections.Counter(); kept = []; kept_rows = []
for d in args.rollout_dirs:
    for f in glob.glob(os.path.join(d, '**', 'rollouts.jsonl'), recursive=True):
        for l in open(f):
            r = json.loads(l)
            if 'history' not in r: C['no_history'] += 1; continue
            C['rollouts'] += 1; h = r['history']
            A = [m['content'] for m in h if m['role'] == 'assistant']
            dd = tp.step_seq([tp.code_of(a) for a in A])
            if not r.get('final_answer'): C['drop_no_answer'] += 1; continue
            if r['reward'] < MIN: C['drop_low_reward'] += 1; continue
            if len(A) > MAXT: C['drop_too_many_turns'] += 1; continue
            if any('<tool_call>' in a or '<function=' in a for a in A): C['drop_format_loop'] += 1; continue
            if not tp.in_program_order(dd): C['drop_program_order'] += 1; continue
            codes = [tp.code_of(a) for a in A]
            ft = next((k for k, c in enumerate(codes) if 'flag' in tp.steps(c)), None)
            if ft is None or any(re.search(r'llm_query', c) and not re.search(r'FLAG', c) for c in codes[ft + 1:]):
                C['drop_post_flag_work'] += 1; continue
            if any(LEAK.search(a) for a in A): C['drop_hint_leak'] += 1; continue
            msgs = [m for m in h if '<hint>' not in m['content']]
            if len(msgs) != len(h) - 1: C['drop_hint_strip'] += 1; continue
            kept.append({'messages': msgs, 'env': 'trick-synth-train', 'uid': f"trick-v7-{r.get('seed')}", 'task': 'trick_synth', 'reward': r['reward'],
                         'probe_t1': bool(re.search(r'llm_query', tp.code_of(A[0]) if A else '')), 'turns': len(A)})
            kept_rows.append(r); C['kept'] += 1
with open(out, 'w') as fo:
    for k in kept: fo.write(json.dumps(k) + '\n')
print(dict(C))
tmp = out + '.kept_rollouts.jsonl'
with open(tmp, 'w') as fo:
    for r in kept_rows: fo.write(json.dumps(r) + '\n')
import subprocess
subprocess.run([sys.executable, os.path.join(H, 'trace_profile.py'), f'REAL_base+note={real}', f'KEPT={tmp}'])
