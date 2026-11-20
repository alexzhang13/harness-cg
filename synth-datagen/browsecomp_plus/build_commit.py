"""Commit-near-cap rows from the hard generation run (specs/commit.toml).

Keeps rollouts that submitted in their last 4 turns after saying they were running out of turns. ``train_from_turn``
marks the first such turn, so SFT trains only the closing turns.

    python build_commit.py <commit pipeline dir> <out.jsonl>
"""
import collections
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

RUN, OUT = Path(sys.argv[1]), sys.argv[2]
SAY = re.compile(r"(few|only \w+|\d+|last|remaining|final|one) (more )?turns? (left|remaining)|my (final|last) turn|"
                 r"running out of turns|turn budget|at turn \d+ ?/ ?\d+|must submit (on|in|this)|consolidat\w*|"
                 r"best[- ]supported (guess|answer|candidate)|guess an answer", re.I)
out, why = [], collections.Counter()
for f in glob.glob(f'{RUN}/gen*/*/*.jsonl'):
    if not f.endswith(('accepted.jsonl', 'rejected.jsonl')):
        continue
    for l in open(f):
        o = json.loads(l); m = o.get('meta') or {}
        g = lambda k: o.get(k) if o.get(k) is not None else m.get(k)
        msgs = o.get('messages') or ((o.get('prompt') or []) + (o.get('completion') or []))
        a_idx = [i for i, x in enumerate(msgs) if x.get('role') == 'assistant']
        n = len(a_idx)
        tb = [re.match(r"Turn \d+/(\d+)", str(x.get('content', ''))) for x in msgs if x.get('role') == 'user']
        BUDGET = next((int(t[1]) for t in tb if t), 15)
        if str(g('stop_condition')) != 'has_final_answer':
            why['no final answer (hit the cap)'] += 1; continue
        if n < BUDGET - 4:
            why['submitted early (not a commit example)'] += 1; continue
        hit = next((t for t in range(max(0, n - 5), n) if SAY.search(str(msgs[a_idx[t]].get('content', '')))), None)
        if hit is None:
            why['late submit without the consolidation statement'] += 1; continue
        why['KEPT'] += 1
        out.append(dict(messages=msgs, train_from_turn=hit + 1, n_turns=n, budget=BUDGET, reward=float(g('reward') or 0),
                        final_answer=str(g('final_answer') or ''), key=str(g('key')), task_sha1=str(g('task_sha1')),
                        n_sub_calls=int(float(g('rlm_sub_llm_calls') or 0)), source=Path(f).relative_to(RUN).parts[0]))
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
with open(OUT, 'w') as fh:
    for r in out:
        fh.write(json.dumps(r) + '\n')
print(dict(why))
if out:
    print(f"kept {len(out)} | correct guesses {sum(r['reward'] >= 1 for r in out)} | statement at turn median "
          f"{st.median(r['train_from_turn'] for r in out)} | budgets {dict(collections.Counter(r['budget'] for r in out))} | -> {OUT}")
