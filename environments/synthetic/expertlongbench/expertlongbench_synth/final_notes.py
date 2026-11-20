"""The final concise ELB notes for the short-record kinds, served as per-row hints (tier ``note`` of the mix env)."""

from __future__ import annotations

import json

C10_HEAD = ("You are an experienced clinician. Write this note in two short turns of code. You never print the dialogue: "
            "a probe sub-call reads it and characterises the visit, a drafting sub-call fills your note template using "
            "that report, and Python checks and submits. Turn 1 is exactly the probe code below (it is how you look at "
            "the context). Keep prose to one line per turn and never refer to this text.")

C10_PROBE = r'''import json, re
J = lambda r: json.loads(r[r.find("{"):r.rfind("}") + 1])
p = J(llm_query('Read this doctor-patient dialogue. Return ONLY JSON: {"visit": "new complaint" / "chronic follow-up" / '
  '"results discussion" / "annual exam" / "post-op follow-up", "chief": the ONE new presenting symptom with its side, or null, '
  '"problems": every problem the clinician addresses}\n\n' + context))
print(p)'''

C10_ITEMS = [
    ("Onset", "when, as a time ('3 weeks ago')"), ("Duration", ""), ("Location", "body site with side"),
    ("Character", "one phrase: the quality with its site"), ("Radiation", ""), ("Severity", "n/10 only"),
    ("Temporal Factor", "only a recurring clock/calendar pattern, else N/A"), ("Alleviating and Aggravating Factors", ""),
    ("Reason for Patient Office Visit or Hospitalization", "2-6 words: main complaint with side"),
    ("Patient Reason for the Visit", "one full sentence: main complaint with side, what brought it on or what they "
                                     "want, other concerns raised"),
    ("Patient Age", ""), ("Patient Sex", ""), ("Vital Signs", ""),
    ("Physical Examination Findings", "every finding, under the body part named"),
    ("Other Objective Data", "reported results"),
    ("Diagnosis and Clinical Impression", "EVERY problem, numbered, with side and the clinician's reasoning"),
    ("Treatment and Management Plan", "full plan per problem: meds with dose, tests, referrals, follow-up"),
    ("Review of Systems (ROS)", "never N/A: '<System>: Reports X. Denies Y.' listing EVERY symptom the patient reports or "
                                "denies, with side, under its organ system (Constitutional = fever, chills, fatigue, "
                                "weight, sleep, appetite); no history, mechanism or exam"),
    ("Relevant Medical History", "condition names"), ("Surgical History", ""), ("Family History", ""),
    ("Drugs", "tobacco / alcohol / recreational drug use, never medications"),
    ("Activities", "the patient's regular exercise / sports / hobbies, with detail; not the injury, plans or dislikes"),
    ("Home and Environment", "home exposure or stressor; not who they live with"),
    ("Employment", ""), ("Eating habits", ""), ("Education", ""), ("Sexuality", ""), ("Suicide/Depression", ""),
]
C10_SYMPTOM = [n for n, _c in C10_ITEMS[:8]]

C10_AUDIT = {
    "Severity": r'if not re.search(r"\d+ ?(/|out of) ?10", note["Severity"]): note["Severity"] = "N/A"',
    "Drugs": r'if not re.search(r"(?i)smok|tobacco|cigar|vap|alcohol|drink|beer|wine|marijuana|cannabis|cocaine|recreational", note["Drugs"]): note["Drugs"] = "N/A"',
    "Temporal Factor": r'if not re.search(r"(?i)night|morning|evening|winter|season|menstru|weekend", note["Temporal Factor"]): note["Temporal Factor"] = "N/A"',
}

C10_DRAFT_HEAD = r'''F = re.findall(r'"([^"]+)": "\.\.\."', T)
r = llm_query("Visit: " + json.dumps(p) + "\nFill this clinical-note template from the dialogue below (comments say what a field "
  "holds; N/A when unsupported). Return ONLY the JSON.\n" + T + "\n\n" + context)
note = {f: m[1] if (m := re.search(re.escape(f) + r'"?\s*:\s*"(.*?)"\s*[,}\n]', r, re.S)) else "N/A" for f in F}
note = {f: "N/A" if re.match(r"(?i)\s*(n/?a\b|none\b|not (mentioned|stated|discussed)|unknown|$)", v) else v for f, v in note.items()}'''

C10_TAIL = r'''print(json.dumps(note, indent=0))
answer["content"] = json.dumps(note)
answer["ready"] = True'''


def c10_template(fields: list[str] | None = None) -> str:
    """c10's T string for a row's fields (one field per line, c10's comment after it)."""
    have = set(fields) if fields else {n for n, _c in C10_ITEMS}
    rows = [(n, c) for n, c in C10_ITEMS if n in have]
    lines = []
    for i, (n, c) in enumerate(rows):
        end = "}" if i == len(rows) - 1 else ","
        lines.append(("{" if i == 0 else "") + f'{json.dumps(n)}: "..."{end}' + (f" # {c}" if c else ""))
    return "\n".join(lines)


def c10_turn2(fields: list[str] | None = None) -> str:
    have = set(fields) if fields else {n for n, _c in C10_ITEMS}
    k = sum(1 for n in C10_SYMPTOM if n in have)
    code = "T = '''" + c10_template(fields) + "'''\n" + C10_DRAFT_HEAD + "\n"
    if k:
        code += ('if p.get("visit") != "new complaint" or not p.get("chief"):\n'
                 f'    note |= {{f: "N/A" for f in F[:{k}]}}\n')
    code += "".join(C10_AUDIT[n] + "\n" for n in C10_AUDIT if n in have)
    return code + C10_TAIL


def visit_note_c10(fields: list[str] | None = None) -> str:
    """The FINAL visit note (c10) for a row with checklist ``fields`` (None = c10's full template)."""
    return (C10_HEAD + "\n\nTurn 1: probe the visit (copy exactly; do not print `context`).\n```repl\n" + C10_PROBE
            + "\n```\n\nTurn 2: draft, check, submit (copy exactly).\n```repl\n" + c10_turn2(fields) + "\n```")


RECIPE_NOTE_C5 = r'''You are a materials scientist; fill this rationale checklist the way you always do, in two short turns of code. Only the Temperature and Heating Method field is explained, from materials knowledge; the other five are paper-specific reasons, so bare N/A, even when the procedure names a gas, a time or a purpose. Start each turn with one plain sentence of your own saying what the block does, and never mention, quote or refer to this text (no "hint", "instructions", "rules", "template", "note").

Turn 1:
```repl
print(context)
```

Turn 2: write F (the six checklist field names, verbatim, in order), the Temperature value (keep its comment) and EXCESS, and submit in the same block.
```repl
import json
F = ["<field 1>", "<field 2>", "<field 3>", "<field 4>", "<field 5>", "<field 6>"]
form = dict.fromkeys(F, "N/A")
# 1-2 sentences from materials knowledge, never N/A: why the MAIN reaction / firing / annealing temperature (and a
# non-furnace method: microwave, plasma, flash or rapid heating, autoclave, hot pressing, arc melting, high
# pressure, molten salt) suits THIS target and these precursors -- name that step, then one specific mechanism
# (the phase that forms or crystallises there, a precursor that decomposes / melts / volatilises, a decomposition
# or competing-phase limit it stays below, a metastable phase it traps, a flux it melts, what the method enables),
# with known values; not a drying, precipitation, ramp or cooling step, no generic filler
form[F[3]] = "..."
EXCESS = None   # a precursor added in a stated excess and its volatile element, e.g. ("Li2CO3", "lithium"); else None
if EXCESS:
    form[F[2]] = f"The excess {EXCESS[0]} compensates the loss of the volatile {EXCESS[1]} during the high-temperature treatment."
answer["content"] = json.dumps(form)
answer["ready"] = True
print(form)
```'''


def note_version_of(kind: str, hint: str) -> str:
    """Which note a row's hint text is (for adherence checks and the export's ``program`` column)."""
    h = hint or ""
    if kind == "visit":
        return "t06_note_c10" if '"chief"' in h and "llm_query" in h else ("t06_note_v11c" if "VISIT =" in h else "t06_other")
    if kind == "tutor":
        return "t04_note_c6" if '"tell"' in h else ("t04_note_v1s" if "parse_json" in h else "t04_other")
    if kind == "recipe":
        return "t03_note_c5" if "dict.fromkeys(F" in h else ("t03_note_v3e" if "form = {" in h else "t03_other")
    return "unknown"
