"""Expand reasoning MCQs not in LongReason (ReClor, LogiQA 2.0) into LongReason-style paragraphs with Qwen3.8-27B.

Each passage becomes several paragraphs that restate the setting; the question is reworded to name it, options are
kept. Filler is added at load time. Sources are read from $LONGREASON_SOURCES_DIR.

    python gen_expanded.py <out.jsonl> <n> [seed] [servers=host:port,...]
"""
import sys, os, json, re, random, glob, collections, urllib.request, concurrent.futures as cf
from pathlib import Path
import pyarrow.parquet as pq

SRC = os.environ.get('LONGREASON_SOURCES_DIR', str(Path.home() / '.cache/longreason_sources'))
LETTERS = 'ABCDEFGH'

def grams(t, n=5):
    w = re.findall(r'[a-z0-9]+', t.lower()); return {' '.join(w[i:i + n]) for i in range(len(w) - n + 1)}

def load_sources():
    b = glob.glob(str(Path.home() / '.cache/huggingface/hub/datasets--lz1bytedance--LongReason/snapshots/*/data'))[0]
    orig = [r['question'] for r in pq.read_table(f'{b}/original-00000-of-00001.parquet', columns=['question']).to_pylist()]
    idx = collections.defaultdict(set)
    for i, o in enumerate(orig):
        for g in grams(o): idx[g].add(i)
    def near_dup(text):
        G = grams(text); c = collections.Counter(i for g in G for i in idx.get(g, ()))
        return bool(c) and c.most_common(1)[0][1] / max(1, len(G)) >= 0.2
    items = []
    for f in ['train.json', 'val.json']:
        for r in json.load(open(f'{SRC}/{f}')):
            items.append({'source': 'reclor', 'src_id': r['id_string'], 'qtype_src': '', 'passage': r['context'], 'question_src': r['question'],
                          'options': r['answers'], 'answer': LETTERS[r['label']]})
    for f in ['train', 'dev', 'test']:
        for l in open(f'{SRC}/MRC/{f}.txt'):
            r = json.loads(l)
            items.append({'source': 'logiqa2', 'src_id': f"{f}-{r['id']}", 'qtype_src': ','.join(k for k, v in r.get('type', {}).items() if v),
                          'passage': r['text'], 'question_src': r['question'], 'options': r['options'], 'answer': LETTERS[r['answer']]})
    if os.environ.get('LRX_SOURCES') == 'puzzles':
        PZ = re.compile(r"\b(who|whom|which (person|one of)|order|rank|seat|sequence|how many|minimum|maximum|at least|at most|in total)\b", re.I)
        items = [it for it in items if it['source'] == 'logiqa2' and PZ.search(it['question_src'])]
        for r in pq.read_table(f'{SRC}/aqua/raw/train-00000-of-00001.parquet').to_pylist():
            opts = [re.sub(r'^[A-E]\)\s*', '', o).strip() for o in r['options']]
            if len(opts) != 5 or len(r['question']) < 120: continue
            items.append({'source': 'aqua', 'src_id': str(abs(hash(r['question'])) % 10**12), 'qtype_src': 'math', 'passage': r['question'].rsplit('?', 1)[0] if '?' in r['question'] else r['question'],
                          'question_src': (r['question'].rsplit('.', 1)[-1].strip() if '?' in r['question'] else 'What is the answer?'), 'options': opts, 'answer': r['correct'].strip()})
        aq = [it for it in items if it['source'] == 'aqua']; lq = [it for it in items if it['source'] != 'aqua']
        items = lq + random.Random(1).sample(aq, min(len(aq), len(lq)))
    keep = [it for it in items if not near_dup(it['passage'] + ' ' + it['question_src'])]
    return keep, len(items) - len(keep)

EXPAND = """You are rewriting a short reasoning-test item into a longer narrative, the way a long-document benchmark does.

PASSAGE:
{passage}

QUESTION: {question}

Do this:
1. SETTING: give the scenario a short fictional setting that the passage does not already have, e.g. a fictional town, kingdom, company, university or festival ("the town of Brindale", "Corvane Institute"). If the passage already names people, places or organisations, keep those names and add the setting around them.
2. PARAGRAPHS: split the passage into its sentences (or tight pairs of sentences). Write ONE paragraph of 90-140 words for each, in the original order. Each paragraph must (a) restate the setting by name, (b) contain the content of its source sentence(s) with every fact, number, name, quantifier and logical relation preserved, and (c) add a few sentences of natural narrative elaboration and commentary around it, as a journalist or storyteller would. Do not add new facts that bear on the question, and do not state the answer.
3. QUESTION: reword the question stem so that it names the setting instead of saying "the passage"/"the above" (e.g. "Which of the following best summarizes the findings about traffic in the town of Brindale?"). Keep its logical type exactly (most / must / except / not / assumption / weaken...).

Return JSON only: {{"setting": "...", "paragraphs": ["...", "..."], "question": "..."}}"""

def llm(servers, prompt, max_tokens=3000, temperature=0.7):
    body = {"model": "qwen38-27b", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False}}
    for k in range(4):
        url = f"http://{servers[(hash(prompt) + k) % len(servers)]}/v1/chat/completions"
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=1800).read())
            return j['choices'][0]['message']['content'] or ''
        except Exception as e: err = e
    return f'ERR {err}'

def expand(servers, it):
    rep = llm(servers, EXPAND.format(passage=it['passage'], question=it['question_src']))
    m = re.search(r'\{.*\}', rep, re.S)
    try: j = json.loads(m.group(0))
    except Exception: return None
    paras = [p.strip().replace('\n', ' ') for p in j.get('paragraphs', []) if isinstance(p, str) and len(p.strip()) > 80]
    setting = str(j.get('setting', '')).strip(); q = str(j.get('question', '')).strip()
    if not (1 <= len(paras) <= 10 and setting and q): return None
    head = setting.split()[-1].lower()
    if sum(head in p.lower() for p in paras) < len(paras) - 1: return None
    opts = "\n".join(f"{LETTERS[i]}. {o.strip()}" for i, o in enumerate(it['options']))
    return {**it, 'setting': setting, 'paragraphs': paras, 'question': f"{q}\n{opts}"}

if __name__ == '__main__':
    out, n = sys.argv[1], int(sys.argv[2]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    servers = (sys.argv[4] if len(sys.argv) > 4 else '127.0.0.1:8000').split(',')
    items, dropped = load_sources(); print(f'sources {len(items)} (dropped {dropped} near-duplicates of LongReason originals)', flush=True)
    rng = random.Random(seed); rng.shuffle(items)
    done = {json.loads(l)['src_id'] for l in open(out)} if Path(out).exists() else set()
    todo = [it for it in items if it['src_id'] not in done][: n - len(done)]
    with open(out, 'a') as fo, cf.ThreadPoolExecutor(64) as ex:
        ok = 0
        for r in ex.map(lambda it: expand(servers, it), todo):
            if r: r['id'] = f"{r['source']}-{r['src_id']}"; fo.write(json.dumps(r) + '\n'); fo.flush(); ok += 1
        print(f'expanded {ok}/{len(todo)}')
