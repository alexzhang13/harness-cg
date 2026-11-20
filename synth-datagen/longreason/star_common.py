"""Parts of the LongReason program-v1 code needed by the STaR prompt builder: the turn-1 locate code, the
question-type classifier prompt, and the classification call. ``LR_PORTS`` / ``LR_MODEL`` pick the server."""
import itertools
import json
import os
import re
import urllib.request

PORTS = os.environ.get('LR_PORTS', 'localhost:8000').split(','); _rr = itertools.count()


def llm(prompt, think, max_tokens, temperature=0.0):
    body = {"model": os.environ.get("LR_MODEL", "qwen38-27b"), "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "chat_template_kwargs": {"enable_thinking": think}}
    for _ in range(4):
        url = f"http://{PORTS[next(_rr) % len(PORTS)]}/v1/chat/completions"
        try:
            j = json.loads(urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=3600).read())
            m = j['choices'][0]['message']; c = m.get('content') or ''
            if not c and think: c = (m.get('reasoning_content') or m.get('reasoning') or '')[-4000:]
            return c, j['usage']['completion_tokens']
        except Exception as e: err = e
    return f'ERR {err}', 0


LOCATE_CODE = r'''
import re, collections
question = {question!r}
lines = [l.strip() for l in context.split("\n") if l.strip()]
low = [l.lower() for l in lines]
print(f"{{len(context):,}} chars, {{len(lines):,}} lines")
STOP = set("""a an the in on at of for to by with from as if when which what who whom whose why how where is are was
were be been being based according given assuming suppose considering among each all both following this that these
those it its they their there then than but and or not no yes option options question answer true false mr mrs ms dr
after before during also only can could would should will may might must does do did has have had one two three four
five six first second third context scenario passage text information statement statements regarding about involving
most least likely best inferred infer conclude concluded correct incorrect describe described implied implies refer
refers referring mentioned many much more some any other same such into within between under over up down out off so
very just main mainly primarily title he she his her him them we us our you your me my supported""".split())
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")

def key_terms(q):
    stem = re.split(r"\n\s*A[.)]\s", q)[0]
    out = [m.strip(" ,.?") for m in re.findall(r'["“]([^"”]{{2,60}})["”]', stem)]
    toks = [re.sub(r"(?:'s|’s)$", "", t.strip("'’")) for t in WORD.findall(stem)]
    run = []
    for t in toks + ["."]:
        if t[:1].isupper() and t.lower() not in STOP:
            run.append(t)
        else:
            if run:
                out.append(" ".join(run))
                if len(run) > 1:
                    out += [w for w in run if len(w) >= 3]
            run = []
    lw = [t.lower() for t in toks]
    out += [f"{{a}} {{b}}" for a, b in zip(lw, lw[1:]) if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2]
    out += [t for t in lw if t not in STOP and len(t) >= 5]
    return list(dict.fromkeys(t for t in out if t))

def score_lines(terms, weight=1.0, sc=None):
    sc = sc if sc is not None else collections.defaultdict(float)
    for t in terms:
        h = [i for i, l in enumerate(low) if t.lower() in l]
        if not h or len(h) > 60:
            continue
        w = weight * (1.0 if (t[:1].isupper() or " " in t) else 0.5) / len(h) ** 0.5
        for i in h:
            sc[i] += w
    return sc

terms = key_terms(question)
sc = score_lines(terms)
ranked = sorted(sc, key=lambda i: -sc[i])
strong = [i for i in ranked[:8] if sc[i] >= 0.4 * sc[ranked[0]]] if ranked else []
# snowball: names / long words that the strongest candidates share with only a few other lines
known = {{t.lower() for t in terms}}
extra = set()
for i in strong:
    for t in WORD.findall(lines[i]):
        t = re.sub(r"(?:'s|’s)$", "", t)
        tl = t.lower()
        if tl in known or tl in STOP or len(t) < 4 or (not t[:1].isupper() and len(t) < 6):
            continue
        n = sum(1 for l in low if tl in l)
        if 2 <= n <= 6:
            extra.add(t)
sc = score_lines(sorted(extra), weight=0.7, sc=sc)
cands = sorted(sorted(sc, key=lambda i: -sc[i])[:{top}])
print("terms:", terms[:12], "| snowball:", sorted(extra)[:12])
for i in cands:
    print(f"[{{i}}] {{lines[i][:1500]}}\n")
'''


CLASSIFY_FINE = """Classify this multiple-choice question by what it asks you to do. Reply with exactly one label from the list.
SUMMARY    - best summarizes the passage / main idea / what the passage mainly emphasizes or is mainly about / title
SUPPORT    - which view the passage supports / what an idea or study suggests or implies / what can be inferred overall / what an example illustrates / purpose of a passage
ATTITUDE   - the author's (or the passage's) attitude, view or opinion about something
ASSUMPTION - an assumption or premise an argument depends on
ARGUMENT   - weaken / strengthen / explain a discrepancy / flaw / evaluate an argument (including "... except")
DEDUCE     - what must be true / what follows / which conclusion can be drawn / which statements (I, II, III) must be true / conditional rules
PUZZLE     - several people's statements or guesses, who is telling the truth, who did what, matching people to things
ORDER      - the correct order or sequence of steps or events
CALC       - a computation with numbers, percentages, dates or counts
REFERENCE  - what a word, phrase or statement ("this", "it", "such a situation", a quoted phrase) refers to or means
NOTMENTION - which option the passage does NOT mention / introduce / cover
NOTIMPLY   - which option is NOT implied / cannot be inferred / is inconsistent with the passage

QUESTION:
{question}

Label:"""
# the routing table's keys, in its order (the regex alternation below depends on it)
LABELS = ['SUMMARY', 'SUPPORT', 'ATTITUDE', 'ASSUMPTION', 'ARGUMENT', 'DEDUCE', 'PUZZLE', 'ORDER', 'CALC', 'REFERENCE',
          'NOTMENTION', 'NOTIMPLY']


def classify_fine(q):
    rep, _ = llm(CLASSIFY_FINE.format(question=q), False, 10)
    m = re.search("|".join(LABELS), rep.upper()); return m.group(0) if m else 'DEDUCE'
