"""Generator-defect injection for xabsence paraphrase train rows: rewrites (corpus-B side of matched pairs) carry the kinds
of defects an LLM paraphraser leaves behind. Own templates only; rates drawn per row around the observed generator rates."""
import re
LABELS = ["Paraphrase:", "Paraphrased:", "Rewritten:", "Restatement:", "Reworded:", "Here is the rewrite:", "Rephrased version:",
          "Paraphrased claim -", "Rewrite:", "New wording:"]
TAILS = ["此处的措辞需要再调整，以保持原意。", "其余部分保持不变。", "（以下为改写后的句子）", "，并在不同条件下进行了比较。",
         "（ここは言い換えが必要です）", "の結果が得られた。", "— далее следует перефразировать без потери смысла.",
         "и это требует уточнения.", "para mantener el sentido original de la frase.", "y se mantiene el significado.",
         "unter Beibehaltung der ursprünglichen Bedeutung.", "의 결과를 유지하도록 다시 작성함.", "en conservant le sens initial."]
SHORT = ["1.", "2.", "3.", "(i)", "(ii)", "Methods.", "Aims.", "Design.", "Setting.", "Results:", "Objective.", "Background:",
         "Main outcome measures.", "Participants.", "Conclusions:", "Interventions.", "Purpose."]
REFUSE = ['The text "{s}" does not contain a claim that can be restated; please share the full sentence you would like paraphrased.',
          'I am unable to paraphrase "{s}" because it is only a fragment without any factual content. Could you send the complete claim?',
          'There seems to be nothing to rephrase here: "{s}" is a heading rather than a biomedical statement. Please provide the claim itself.',
          'It looks like the input "{s}" is incomplete, so there is no fact to restate. Kindly supply the whole sentence.',
          'Sorry, "{s}" is just a label, not a claim, so it cannot be paraphrased without inventing information.']
NOTES = [" (Note: the wording above was adjusted to differ from the source sentence while keeping its meaning.)",
         " (Note: an earlier draft of this rewrite mixed in another language; this version is in English.)",
         " [Rewrite note: technical terms were kept as in the original.]",
         " (This restatement keeps the original numbers unchanged.)",
         " (Note: abbreviations were spelled out where possible.)"]
FIXNUM = [", although this figure should read {n} to match the source statement.",
          " -- note that the original claim reports {n}, and the number here should be corrected accordingly.",
          " (the source gives {n}; the value above is a transcription slip)."]
WORDNUM = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
def _num_change(o, p, rng):
    m = re.search(r'\b(\d+(?:\.\d+)?)\b', p)
    if m:
        v = m.group(1); new = str(int(float(v)) + rng.choice([-2, -1, 1, 2, 3])) if '.' not in v else str(round(float(v) * 1.5, 1))
        if new == v or new.startswith('-'): return None
        return p[:m.start()] + new + p[m.end():] + rng.choice(FIXNUM).format(n=v)
    m = re.search(r'\b(' + '|'.join(WORDNUM) + r')\b', p, re.I)
    if m:
        alt = rng.choice([w for w in WORDNUM if w != m.group(1).lower()])
        return p[:m.start()] + alt + p[m.end():] + rng.choice(FIXNUM).format(n=m.group(1).lower())
    return None
def inject(pairs, rng):
    """pairs: list of [orig, para] for MATCHED pairs (mutated in place). Returns a count dict."""
    n = len(pairs); cnt = {}
    r_label, r_tail, r_note = rng.uniform(0.02, 0.07), rng.uniform(0.005, 0.025), rng.uniform(0.0, 0.004)
    idx = list(range(n)); rng.shuffle(idx); used = set()
    def take(k):
        out = [i for i in idx if i not in used][:k]; used.update(out); return out
    for i in take(int(round(r_tail * n))):
        o, p = pairs[i]; w = p.split(); cut = max(3, int(len(w) * rng.uniform(0.4, 0.85)))
        pairs[i][1] = ' '.join(w[:cut]) + rng.choice(['', ' ']) + rng.choice(TAILS); cnt['tail'] = cnt.get('tail', 0) + 1
    for i in take(rng.choice([0, 0, 1, 1, 2])):
        s = rng.choice(SHORT); pairs[i] = [s, rng.choice(REFUSE).format(s=s)]; cnt['refusal'] = cnt.get('refusal', 0) + 1
    for i in [i for i in idx if i not in used][:40]:
        if cnt.get('num', 0) >= rng.choice([0, 1, 1]): break
        q = _num_change(*pairs[i], rng)
        if q: pairs[i][1] = q; used.add(i); cnt['num'] = cnt.get('num', 0) + 1
    for i in take(int(round(r_note * n))):
        pairs[i][1] = pairs[i][1].rstrip() + rng.choice(NOTES); cnt['note'] = cnt.get('note', 0) + 1
    for i in [i for i in idx if i not in used]:
        if rng.random() < r_label:
            pairs[i][1] = rng.choice(LABELS) + ' ' + pairs[i][1]; cnt['label'] = cnt.get('label', 0) + 1
    return cnt
