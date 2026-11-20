"""Filter LongReason program-v1 traces into hint-free SFT rows.

Keeps traces with reward 1 that run the locate code in turn 1, make exactly one sub-call carrying a note procedure
over all scenario paragraphs, and stay within MAX_TURNS (4 for the published set); at most 2 per task.

    python build_sft_lr.py <out.jsonl> <max_turns> <pipeline dir> [<pipeline dir> ...]"""
import argparse, json, glob, re, collections
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
from rlm_train.frozen import frozen_path
ITEMS_FILE = str(frozen_path(os.environ.get('LONGREASON_EXPANDED_ITEMS') or REPO / 'environments/synthetic/longreason/longreason_expanded/data/expanded_items.jsonl',
                             'longreason/expanded_items.jsonl'))
LEAK = re.compile(r"\b(the|my|this) (note|hint|instructions)\b|<hint>|as (instructed|suggested)|the procedure (from|in) the", re.I)
NOTE_MARKERS = ("How to solve this task (2 turns", "## Procedures by question type")
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('out'); ap.add_argument('max_turns', type=int); ap.add_argument('run_dirs', nargs='+')
args = ap.parse_args()
out, MAXT, dirs = args.out, args.max_turns, args.run_dirs

NOTE = str(Path(__file__).resolve().parent / 'notes/longreason-expanded-train.md')
def _g(t, n=6):
    w = re.findall(r'[a-z]+', t.lower()); return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}
PROCS = {name: _g(p) for name, p in re.findall(r"### (.+?)\nUse for: .*?\n```text\n(.*?)```", open(NOTE).read(), re.S)}
def proc_fidelity(code):
    """(library procedure type, text coverage): >.8 verbatim, .3-.8 edited, <.3 custom."""
    m = re.search(r'PROCEDURE = """(.*?)"""', code, re.S)
    if not m: return None, 0.0
    G = _g(m.group(1)); best = max(PROCS, key=lambda k: len(G & PROCS[k]) / max(1, len(PROCS[k])))
    return best, len(G & PROCS[best]) / max(1, len(PROCS[best]))
ITEMS = {}
for _l in open(ITEMS_FILE):
    _it = json.loads(_l); ITEMS[_it['question'].strip()[:200]] = _it
def passage_recall(msgs, code):
    """Do the lines passed to the sub-call contain every scenario paragraph? None if unknown."""
    q = msgs[1]['content'].split('then reason.', 1)[-1].split('Respond with the single option letter only.')[0].strip()
    it = ITEMS.get(q[:200]); out1 = next((m['content'] for m in msgs if m['role'] == 'user' and m['content'].startswith('REPL output')), '')
    printed = {int(i): t for i, t in re.findall(r"^\[(\d+)\] (.*)$", out1, re.M)}
    m = re.search(r'passages = .*?for i in (\[[\d,\s]+\]|cands)', code, re.S)
    if not (it and printed and m): return None
    sel = list(printed) if m.group(1) == 'cands' else [int(x) for x in re.findall(r'\d+', m.group(1))]
    got = " ".join(printed.get(i, '') for i in sel).lower()
    return sum(p.lower()[:100] in got for p in it['paragraphs']) / len(it['paragraphs'])
def codes(msgs):
    return [c for m in msgs if m['role'] == 'assistant' for c in re.findall(r"```repl\n(.*?)```", m['content'] if isinstance(m['content'], str) else '', re.S)]

C = collections.Counter(); kept = []; seen = set()
for d in dirs:
    for f in glob.glob(f"{d}/gen*/longreason-expanded-train/accepted.jsonl") + glob.glob(f"{d}/gen*/longreason-expanded-train/rejected.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if 'messages' not in r or r['key'] in seen: continue
            seen.add(r['key']); C['rollouts'] += 1
            msgs = r['messages']; A = [m['content'] for m in msgs if m['role'] == 'assistant']; K = codes(msgs)
            if r['stop_condition'] != 'has_final_answer': C['drop_no_submit'] += 1; continue
            if r['reward'] < 1: C['drop_wrong'] += 1; continue
            if len(A) > MAXT: C['drop_too_many_turns'] += 1; continue
            if not K or not all(s in K[0] for s in ('key_terms', 'score_lines', 'extra')): C['drop_no_locate'] += 1; continue
            if r['rlm_sub_llm_calls'] != 1 or not any('PROCEDURE' in c and 'llm_query' in c for c in K): C['drop_subcall_shape'] += 1; continue
            code = "\n".join(K); ptype, cov = proc_fidelity(code)
            if cov < 0.3: C['drop_custom_procedure'] += 1; continue
            rec = passage_recall(msgs, code)
            if rec is None: C['drop_passages_unparsed'] += 1; continue
            if rec < 1: C['drop_passages_missing_fact'] += 1; continue
            if any('<tool_call>' in a for a in A): C['drop_format'] += 1; continue
            if any(LEAK.search(a) for a in A): C['drop_hint_leak'] += 1; continue
            if any(any(mk in (m['content'] if isinstance(m['content'], str) else '') for mk in NOTE_MARKERS) for m in msgs): C['drop_note_in_messages'] += 1; continue
            kept.append({'messages': msgs, 'env': 'longreason-expanded', 'uid': f"lrx-{r['task_sha1']}-{r['key']}", 'task': 'longreason_expanded',
                         'reward': r['reward'], 'turns': len(A), 'procedure': ptype, 'proc_cov': round(cov, 2)})
            C['kept'] += 1
per = collections.Counter(); final = []
for k in kept:
    t = k['uid'].split('-')[1]
    if per[t] < 2: final.append(k); per[t] += 1
with open(out, 'w') as fo:
    for k in final: fo.write(json.dumps(k) + '\n')
C['written'] = len(final); C['tasks'] = len(per)
print(dict(C))
print('turns', collections.Counter(k['turns'] for k in final)); print('procedure', collections.Counter(k['procedure'] for k in final))
