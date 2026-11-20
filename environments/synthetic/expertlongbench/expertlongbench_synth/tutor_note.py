"""Tutor (T04) notes."""

TUTOR_NOTE_V1S = r'''This tutoring-assessment form is graded on annotation conventions, so a fixed three-turn program does best. Python keeps the form self-consistent, one sub-call reads the dialogue turn by turn and judges the last answer, and one sub-call writes the feedback. The tutoring conventions behind it:
- The last answer is "Correct response" only if it fully and specifically answers every part of what the tutor last asked. It is "Partially correct response" if part of it is right but something is wrong, missing or vague. It is "Incorrect response" if nothing asked for is right, which includes restating the question or a generic remark.
- A correct answer has no mistake: the four mistake fields are N/A, Repeated mistake is No, and the Sub-question Transition names the next sub-question. A not-correct answer always has a mistake identified, a rectification and both types, and the Transition is N/A.
- "Repeated mistake" means the student errs again right after corrective feedback. It is Yes exactly when the last answer is not correct AND the student's previous answer was not fully accepted either. An earlier mistake that the student has since fixed does not count.
- Feedback is scaffolded. On a first slip (Repeated = No) it is Indirect: a guiding question that points at the gap without giving the answer away. On a repeated mistake (Repeated = Yes) it is Direct: state the correct fact, then ask a short follow-up.

## How to run it
- Run the three turns below IN ORDER, each as ONE ```repl block, copying the code exactly.
- Do not judge or rewrite the values by hand. The program already applies the conventions.
- If a turn raises an error, fix only that line and re-run that turn.
- Explain what you are doing in your own words, as an expert tutor assessing the dialogue. Never mention, quote or refer to a note, a hint, rules or instructions.

## Turn 1: read the dialogue.
```repl
print(len(context), "chars")
print(context[:12000])
```

## Turn 2: list every student answer with the tutor's reaction, judge the last answer, and derive Evaluation and Repeated in Python.
```repl
import re, json

def parse_json(reply):
    r = (reply or "").strip()
    cands = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", r, re.S)]
    a, b = r.find("{"), r.rfind("}")
    if a != -1 and b > a:
        cands.append(r[a:b + 1])
    for c in cands:
        try:
            o = json.loads(c)
            if isinstance(o, dict):
                return o
        except Exception:
            pass
    return {}

DIALOGUE = "\n\n=== DIALOGUE ===\n" + context + "\n=== END ==="
VERDICT = ("give your own expert verdict on it: \"correct\" only if it fully and specifically answers what the "
  "tutor asked; on the right track but vague, incomplete or imprecise is \"partial\"; a wrong central claim or a "
  "non-answer is \"incorrect\". Put the verdict in \"last_verdict\" (and as the last turn's verdict).\n")
led = parse_json(llm_query(
  "Below is a tutoring dialogue. List EVERY student turn in order (including the first one and the last one).\n"
  "For each student turn give:\n- \"text\": the first 12 words of the student turn\n"
  "- \"subq\": the index (1, 2, 3, ...) of the sub-question the student is answering; it increases only when the tutor\n"
  "  moves on to a NEW sub-question\n"
  "- \"verdict\": how the TUTOR'S NEXT TURN reacts to it: \"correct\" (fully accepts it), \"partial\" (accepts part, asks for\n"
  "  more / adds a correction), \"incorrect\" (rejects it), \"none\" (not an answer, e.g. the problem statement or \"I don't\n"
  "  know\"). For the LAST student turn there is no tutor reaction yet: " + VERDICT +
  'Return ONLY JSON: {"turns": [{"text": "...", "subq": 1, "verdict": "..."}, ...], "last_verdict": "..."}'
  + DIALOGUE))
turns = [t for t in led.get("turns") or [] if isinstance(t, dict)]
for t in turns:
    print(t.get("subq"), t.get("verdict"), "|", t.get("text"))
last = str(led.get("last_verdict") or (turns[-1].get("verdict", "") if turns else "")).lower()
EV = ("Partially correct response" if "partial" in last else
      "Incorrect response" if ("incorrect" in last or "none" in last) else
      "Correct response" if "correct" in last else "Incorrect response")
earlier = [t for t in turns[:-1] if str(t.get("verdict", "")).lower() != "none"]
prev_not_accepted = bool(earlier) and not str(earlier[-1].get("verdict", "")).lower().startswith("correct")
REP = "Yes" if (EV != "Correct response" and prev_not_accepted) else "No"
print("Evaluation:", EV, "| previous answer not accepted:", prev_not_accepted, "| Repeated mistake:", REP)
```

## Turn 3: write the feedback for the branch taken, assemble the form, and submit.
```repl
if EV == "Correct response":
    tr = llm_query("You are an expert tutor. In this dialogue the student's last response is correct." + DIALOGUE +
                   "\n\nName the NEXT sub-question of the original problem the tutor should move on to, as one short "
                   "phrase or sentence (what it asks). Return only that phrase.")
    vals = {"Mistake identification": "N/A", "Mistake rectification": "N/A", "Mistake identification type": "N/A",
            "Mistake rectification type": "N/A", "Sub-question Transition": (tr or "").strip().strip('"')}
else:
    if REP == "Yes":
        TYPE, style = "Direct", ("The student has now erred again right after the tutor's feedback, so the feedback is "
            "DIRECT: state the correct answer or fact explicitly (1-2 sentences), then optionally one short follow-up question.")
    else:
        TYPE, style = "Indirect", ("This is the student's first slip on this point, so the feedback is INDIRECT: a short "
            "guiding question or hint that points at the problem WITHOUT giving the correct answer away.")
    fb = parse_json(llm_query(
        "You are an expert tutor. The student's last response in this dialogue is "
        + EV.lower().replace(" response", "") + "." + DIALOGUE +
        "\n\nWrite:\n- identification: ONE sentence about the student's last response stating precisely what is wrong "
        "or missing (e.g. 'The student incorrectly states that X.' or 'The student's response does not explain Y.');\n"
        "- rectification: the tutor's own next words TO the student that address it. " + style +
        '\nReturn ONLY JSON: {"identification": "...", "rectification": "..."}'))
    vals = {"Mistake identification": str(fb.get("identification", "")).strip() or "N/A",
            "Mistake rectification": str(fb.get("rectification", "")).strip() or "N/A",
            "Mistake identification type": TYPE, "Mistake rectification type": TYPE, "Sub-question Transition": "N/A"}
form = {"Evaluation of the student response": EV, "Repeated mistake": REP, **vals}
print(json.dumps(form, indent=1, ensure_ascii=False))
answer["content"] = json.dumps(form, ensure_ascii=False)
answer["ready"] = True
```'''


TUTOR_NOTE_C6 = r'''This tutoring-assessment form is graded on annotation conventions, so a fixed two-turn program does best. Python keeps the form self-consistent, and one sub-call reads the dialogue turn by turn, judges the last answer and drafts the feedback. The tutoring conventions behind it:
- The last answer is "Correct response" only if it fully and specifically answers every part of what the tutor last asked. It is "Partially correct response" if part of it is right but something is wrong, missing or vague. It is "Incorrect response" if nothing asked for is right, which includes restating the question or a generic remark.
- A correct answer has no mistake: the four mistake fields are N/A, Repeated mistake is No, and the Sub-question Transition names the next sub-question. A not-correct answer always has a mistake identified, a rectification and both types, and the Transition is N/A.
- "Repeated mistake" means the student errs again right after corrective feedback. It is Yes exactly when the last answer is not correct AND the student's previous answer was not fully accepted either. An earlier mistake that the student has since fixed does not count.
- Feedback is scaffolded. On a first slip (Repeated = No) it is Indirect: a guiding question that points at the gap without giving the answer away. On a repeated mistake (Repeated = Yes) it is Direct: state the correct fact, then ask a short follow-up.

## How to run it
- Turn 1 is exactly the code below: its ledger sub-call is how you look at the context, so do not print `context` first. Two turns in all, each ONE ```repl block, copying the code exactly.
- Do not judge or rewrite the values by hand. The program already applies the conventions.
- If a turn raises an error, fix only that line and re-run it.
- Keep your prose to one line per turn, in your own words as an expert tutor. Never mention a note, a hint, rules or instructions.

## Turn 1: one ledger sub-call reads the dialogue turn by turn, judges the last answer and drafts the feedback for both scaffold levels; Python derives Evaluation, Repeated and the types and picks the feedback.
```repl
import json
J = lambda r: json.loads(r[r.find("{"):r.rfind("}") + 1])
led = J(llm_query("Below is a tutoring dialogue. List EVERY student turn in order (including the first and the last). For each give "
  '"text": its first 12 words; "subq": the index of the sub-question it answers (increases only when the tutor moves on to a NEW '
  'sub-question); "verdict": how the TUTOR\'S NEXT TURN reacts to it: "correct" (fully accepts it), "partial" (accepts part, asks for '
  'more / adds a correction), "incorrect" (rejects it), "none" (not an answer, e.g. the problem statement or "I don\'t know"). The LAST '
  'student turn has no reaction yet: give your own expert verdict: "correct" only if it fully and specifically answers what the tutor '
  'asked; on the right track but vague, incomplete or imprecise is "partial"; a wrong central claim or a non-answer is "incorrect". '
  'Then, as an expert tutor: if the last verdict is correct, "next": the next sub-question of the original problem, one short phrase; '
  'otherwise "identification": one sentence stating precisely what is wrong or missing in the last response, "hint": your next words '
  'to the student, a short guiding question pointing at the problem without giving the answer away, and "tell": your next words '
  'stating the correct answer or fact explicitly, then optionally a short follow-up question. Return ONLY JSON: {"turns": [{"text": '
  '"...", "subq": 1, "verdict": "..."}, ...], "last_verdict": "...", "next": "...", "identification": "...", "hint": "...", "tell": "..."}'
  "\n\n=== DIALOGUE ===\n" + context + "\n=== END ==="))
v = [str(t.get("verdict", "")).lower() for t in led.get("turns", [])]
last = str(led.get("last_verdict") or (v or [""])[-1]).lower()
EV = "Partially correct response" if "partial" in last else "Correct response" if last.startswith("correct") else "Incorrect response"
prev = [x for x in v[:-1] if x != "none"]
REP = "Yes" if EV[0] != "C" and prev and not prev[-1].startswith("correct") else "No"
T = "Direct" if REP == "Yes" else "Indirect"
form = {"Evaluation of the student response": EV, "Repeated mistake": REP, "Mistake identification": led.get("identification") or "N/A",
        "Mistake rectification": led.get("tell" if REP == "Yes" else "hint") or "N/A", "Mistake identification type": T,
        "Mistake rectification type": T, "Sub-question Transition": "N/A"}
if EV[0] == "C":
    form = {k: "N/A" for k in form} | {"Evaluation of the student response": EV, "Repeated mistake": "No",
                                       "Sub-question Transition": led.get("next") or "N/A"}
print(json.dumps(form, indent=1))
```

## Turn 2: submit.
```repl
answer["content"] = json.dumps(form, ensure_ascii=False)
answer["ready"] = True
```'''

TUTOR_NOTE = TUTOR_NOTE_C6
