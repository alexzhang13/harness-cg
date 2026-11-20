# How to solve this task (2 turns, 1 sub-call)

The scenario is a handful of paragraphs scattered through ~470k characters of unrelated filler; every scenario paragraph is
one whole line of `context`. Finding all of them is easy with a multi-term line search. Most of the difficulty is the
REASONING: the scenario paragraphs wrap each fact in narrative commentary, and the right answer depends on reading the
question type's logic precisely. So: locate in code, then give ONE thinking sub-call the located lines plus a procedure
written for this question's type.

## Rules
- Turn 1 is exactly the locate code below (paste the full question, options included). No other search, no reading of `context` by hand.
- Turn 2 is exactly ONE `llm_query` with the procedure for the question's type, then submit. Use it even when the answer looks
  obvious from the printed lines: the scenario paragraphs wrap each fact in commentary that misleads a quick reading.
- Do not answer from your own reading, and do not make extra sub-calls. If no letter was parsed, submit the letter the printed
  reply argues for.

## Turn 1: locate (run this code, with the question pasted in)

```repl
import re, collections
question = """<paste the full question with its options>"""
lines = [l.strip() for l in context.split("\n") if l.strip()]
low = [l.lower() for l in lines]
print(f"{len(context):,} chars, {len(lines):,} lines")
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
    out = [m.strip(" ,.?") for m in re.findall(r'["“]([^"”]{2,60})["”]', stem)]
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
    out += [f"{a} {b}" for a, b in zip(lw, lw[1:]) if a not in STOP and b not in STOP and len(a) > 2 and len(b) > 2]
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
known = {t.lower() for t in terms}
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
cands = sorted(sorted(sc, key=lambda i: -sc[i])[:14])
print("terms:", terms[:12], "| snowball:", sorted(extra)[:12])
for i in cands:
    print(f"[{i}] {lines[i][:1500]}\n")
```

## Turn 2: one sub-call with the procedure for this question's type, then submit

Decide the question type yourself from its wording (table below), and copy that type's procedure verbatim into PROCEDURE.

```repl
HEAD = ("Below are passages taken from a long document. Most are unrelated filler, and some may describe a different but "
        "similarly named place, organisation or event: use ONLY the passages about the scenario in the question.")
PROCEDURE = """<the procedure for this question's type, copied from below>"""
passages = "\n\n".join(f"[{i}] {lines[i][:1500]}" for i in cands)
reply = llm_query(f"{HEAD}\n\nPASSAGES:\n{passages}\n\nQUESTION:\n{question}\n\n{PROCEDURE}\n\nEnd with a final line of the form 'Answer: <letter>'.")
found = re.findall(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\**\(?([A-H])\b", reply)
print(reply[-800:])
if found:
    answer["content"] = found[-1]
    answer["ready"] = True
else:
    print("NO LETTER PARSED")
```

If no letter was parsed, read the printed reply and submit the letter it argues for in the next turn.

## Procedures by question type

### SUMMARY
Use for: best summarizes / main idea / mainly emphasizes / mainly about / title.
```text
This question asks which view the passage supports, i.e. its point. Work in steps.
STEP 1 - List the claims the passage makes and label each as CENTRAL (what the passage argues for, emphasises, keeps returning to, or introduces with "instead", "rather", "but", "the key is", "emphasize") or SUBSIDIARY (concessions, qualifications, background, "this doesn't mean...", side advice).
STEP 2 - State the passage's central point in one sentence, including the goal or reason it attaches to it.
STEP 3 - Evaluate EVERY option: a literally stated but subsidiary point is a worse answer than the option that captures the central point; among options that capture it, prefer the one that also includes the goal/reason the passage attaches, as long as every part of it is in the passage.
```

### SUPPORT
Use for: which view is supported / what an idea or study suggests or implies / what can be inferred overall / what an example illustrates.
```text
This question asks what an idea, argument or passage suggests or implies. Writers and paraphrasers routinely soften a thesis with hedges ("might actually be", "questioned whether", "may"); the hedge is about presentation, not about which position is being argued. Do this:
1. State the position the idea argues FOR and the belief it argues AGAINST, in one sentence each, ignoring hedges.
2. For every option, check whether it (a) states that position, (b) states the opposite, (c) overstates it beyond what the idea claims, or (d) narrows or distorts it (keeps only part of what the idea says, e.g. one half of a contrast).
3. Choose the option of type (a). An option of type (d) is not a "safer" version of the thesis; it is a misreading.
```

### ASSUMPTION
Use for: an assumption the argument depends on.
```text
This is an argument-analysis question. Work in steps.
STEP 1 - Reconstruct THE ARGUMENT under discussion as EVIDENCE -> CONCLUSION, in one or two sentences each. The argument is the inference that someone in the scenario draws or that the passage reports being drawn (e.g. "group X did better, so doing X is what produces / is required for the result"); if the passage only questions or scrutinises a conclusion, that questioned conclusion IS the argument's conclusion. The narrator's own caution, advice or calls for scrutiny are commentary, not the argument.
STEP 2 - Identify the gap between the evidence and the conclusion (what must also hold for the conclusion to follow, especially for words like "must", "only", "the definitive way").
STEP 3 - Evaluate EVERY option: an assumption is something the argument needs (negating it would break the argument). An option that contradicts or undermines the argument's conclusion cannot be an assumption it relies on.
```

### DEDUCE
Use for: what must be true / what follows / which conclusion can be drawn / which statements must be true / conditional rules.
```text
Do this:
1. Write the scenario as precise statements: each rule as "IF condition THEN consequence", including what the rule or arrangement itself depended on (what had to be in place for it to apply, e.g. "as long as", "provided that", "came with", "the proposal was that ..."), and each reported fact.
2. Check the facts against the rules. If a rule's trigger happened but its consequence visibly did not, the facts are only consistent if something the rule depended on was NOT actually in place. Identify which precondition must have failed; that conclusion is what "must be true".
3. Evaluate every option. An option that relies on a person, group or category the text never mentions does not follow. An option that just repeats a rule's consequence is wrong if a fact contradicts that consequence.
4. Choose the option that must be true.
```

### PUZZLE or CALC
Use for: who is telling the truth, who did what, matching; any computation with numbers, dates, counts.
```text
Work in this order.
1. Translate every premise about the scenario into a plain logical statement (All X are Y / No X is Y / Some X are Y / if-then). Narrative wording is a paraphrase of such a statement: read it as the closest standard statement (e.g. "several of the club members ended up at the lecture" = "some club members attended the lecture"); do not discard a premise because it is phrased loosely.
2. For EVERY option, decide whether the premises entail it, and show the chain.
3. If more than one option is entailed, prefer the one that follows by combining two or more premises over one that merely repeats a single premise: "must also be true" asks for a consequence of the premises, not a premise.
```

### ORDER
Use for: the correct order or sequence of steps or events.
```text
This question asks for an order of steps or events. The order in which the passages or paragraphs appear is NOT evidence of the order of the steps: documents often discuss steps topic by topic. Do this:
1. For each step, quote any explicit temporal cue in the text (before, after, first, begins with, finally, once ... then) and what it says the step comes before or after. Note which part of the process (e.g. studying beforehand vs. sitting the exam) each cue is about.
2. Check every option against each explicit cue and against the necessary real-world precedence (a thing must exist before it is checked or copied).
3. Choose the option that violates none of these; if every option violates something, choose the one consistent with the necessary precedences of the process the question asks about.
```

### REFERENCE
Use for: what a word, phrase or statement ("this", "it", "such a situation") refers to.
```text
This question asks what a referring expression (such as "this situation", "it", "such a move") refers to in someone's statement. Work in this order.
1. Find the speaker's reported statement in the relevant passages. If the exact wording is not there, look for the same or a synonymous referring expression attached to that speaker's words (e.g. "this situation" ~ "this possibility" ~ "such a move" ~ "this development").
2. Resolve it the way a reader resolves a demonstrative: its antecedent is the concrete thing the text introduced just before it (usually in the preceding clause or sentence), not the general topic of the passage. In a statement of the form "if X, then Y may happen", an expression standing for what may happen or develop refers to Y, not to the condition X.
3. Check EVERY option against that antecedent and choose the one that names it.
```

### NOT MENTIONED
Use for: which option the text does NOT mention / introduce / cover.
```text
This question asks which topic the text does NOT cover. Long texts often name a topic, or wrap it in generic commentary ("is a complex area of study", "remains promising", "researchers are examining it"), without saying anything specific about it. Do this:
1. For EVERY option, write down the specific information a reader actually learns about that topic from the relevant passages (concrete facts, items, causes, numbers). Quote it. If all you can quote is the topic's name, a statement that it is being studied or is unknown, or vague evaluation, write "NO SPECIFIC CONTENT".
2. Apply the same standard to every option; do not require exact wording for one option and accept paraphrase for another.
3. The answer is the option with no (or the least) specific content.
```

### NOT IMPLIED
Use for: which option is NOT implied / cannot be inferred / inconsistent.
```text
Work in this order.
1. Note the question's logical type. If it asks what is NOT stated / NOT implied / NOT introduced / an EXCEPTION, the correct option is the one the passage does not support; every other option should be supported.
2. For EVERY option, quote the passage sentence(s) that support it, or write NONE. A claim stated in different words, or a direct consequence of what is stated, counts as supported. Mark separately anything the option adds that the passage never asserts (an extra actor, an intensifier like "excessively", a positive or negative evaluation).
3. Judge each option on whether its CENTRAL content is supported, using the base claims of the passage rather than passing commentary.
4. Choose the option with the weakest support.
```

### OTHER
Use for: weaken / strengthen / explain / flaw, the author's attitude, anything else.
```text
Reason step by step using only the relevant passages, then end with a final line of the form 'Answer: <letter>'.
```
