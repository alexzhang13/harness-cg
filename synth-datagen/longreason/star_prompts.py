"""Build the turn-2 sub-call prompts of program v1 for longreason-expanded train items, without a root model.

    python star_prompts.py <out.jsonl> [max_items] [items.jsonl]"""
import sys, os, json, re, io, contextlib, concurrent.futures as cf
from pathlib import Path
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
if {'-h', '--help'} & set(sys.argv[1:]):
    sys.exit(print(__doc__))
from star_common import LOCATE_CODE, classify_fine
from longreason_expanded import build_context
REPO = Path(__file__).resolve().parents[2]
from rlm_train.frozen import frozen_path
ITEMS_FILE = str(frozen_path(os.environ.get('LONGREASON_EXPANDED_ITEMS') or REPO / 'environments/synthetic/longreason/longreason_expanded/data/expanded_items.jsonl',
                             'longreason/expanded_items.jsonl'))
NOTE = open(H + '/notes/longreason-expanded-train.md').read()
PROC = {name: body.strip() for name, body in re.findall(r"### (.+?)\nUse for: .*?\n```text\n(.*?)```", NOTE, re.S)}
FINE2PROG = {'SUMMARY': 'SUMMARY', 'SUPPORT': 'SUPPORT', 'ATTITUDE': 'OTHER', 'ASSUMPTION': 'ASSUMPTION', 'ARGUMENT': 'OTHER', 'DEDUCE': 'DEDUCE',
             'PUZZLE': 'PUZZLE or CALC', 'ORDER': 'ORDER', 'CALC': 'PUZZLE or CALC', 'REFERENCE': 'REFERENCE', 'NOTMENTION': 'NOT MENTIONED', 'NOTIMPLY': 'NOT IMPLIED'}
HEAD = ("Below are passages taken from a long document. Most are unrelated filler, and some may describe a different but "
        "similarly named place, organisation or event: use ONLY the passages about the scenario in the question.")
out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
src = sys.argv[3] if len(sys.argv) > 3 else ITEMS_FILE
items = [json.loads(l) for l in open(src)]
train = [it for k, it in enumerate(items) if k % 10 != 9][:n]
def one(it):
    q = it['question'].strip(); ctx = build_context(it)
    ns = {'context': ctx}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(LOCATE_CODE.format(question=q, top=14), ns)
    passages = "\n\n".join(f"[{i}] {ns['lines'][i][:1500]}" for i in ns['cands'])
    got = passages.lower(); rec = sum(p.lower()[:100] in got for p in it['paragraphs']) / len(it['paragraphs'])
    fine = classify_fine(q); pt = FINE2PROG[fine]
    prompt = (f"{HEAD}\n\nPASSAGES:\n{passages}\n\nQUESTION:\n{q}\n\n{PROC[pt]}\n\nEnd with a final line of the form 'Answer: <letter>'.")
    return {'id': it['id'], 'gold': it['answer'], 'fine': fine, 'ptype': pt, 'recall': rec, 'prompt': prompt}
with open(out, 'w') as fo, cf.ThreadPoolExecutor(16) as ex:
    for o in ex.map(one, train): fo.write(json.dumps(o) + '\n')
import collections
rows = [json.loads(l) for l in open(out)]
print(len(rows), collections.Counter(r['ptype'] for r in rows), 'full recall', sum(r['recall'] >= 1 for r in rows))
