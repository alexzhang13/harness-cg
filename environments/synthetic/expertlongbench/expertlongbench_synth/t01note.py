"""The T01 docket program as an LM-root note (tier ``note``): the general form of the real Qwen3.8-27B winners."""

from __future__ import annotations

import json

TURN1_V4 = r'''
import re, json
NA_WORDS = {"n/a", "na", "none", "not applicable", "not mentioned", ""}
SOFT_NA = NA_WORDS | {"unknown", "not stated", "not specified", "not provided", "not available", "not found",
                      "no information", "not discussed", "not addressed", "null", "-"}

def is_na(v):
    s = str(v if v is not None else "").strip().lower().rstrip(".")
    return s in SOFT_NA or s.startswith("n/a") or s.startswith("not mentioned") or s.startswith("not stated")

def parse_json(reply):
    """First JSON object in a sub-LM reply (fenced or bare); None if there is none."""
    r = (reply or "").strip()
    cands = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", r, re.S)]
    a, b = r.find("{"), r.rfind("}")
    if a != -1 and b > a:
        cands.append(r[a:b + 1])
    for c in cands:
        try:
            o = json.loads(c)
        except Exception:
            continue
        if isinstance(o, dict):
            return o
    return None

def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def quote_ok(quote, source_norm):
    """True when a sub-LM quote really comes from the source. Tolerates whitespace / OCR / punctuation drift, quotes
    stitched from several places with '...', and two-column PDF text where a sentence is interleaved with the other
    column: any '...'-piece with >= 60% of its word 3-grams in the source, or >= 50% over the whole quote."""
    def frac(t):
        w = norm(t).split()
        if len(w) < 3:
            return 0.0
        grams = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
        return sum(g in source_norm for g in grams) / len(grams)
    pieces = [p for p in re.split(r"\.\.\.|…|\[\.\.\.\]", str(quote)) if len(norm(p).split()) >= 3]
    return bool(pieces) and (max(frac(p) for p in pieces) >= 0.6 or frac(quote) >= 0.5)

MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                           "september", "october", "november", "december"])}

def find_dates(s):
    """All dates in s as (y, m, d): MM/DD/YYYY, YYYY-MM-DD, 'Month D, YYYY'."""
    out = []
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", s):
        y = int(m.group(3)); y += 2000 if y < 50 else (1900 if y < 100 else 0)
        out.append((y, int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", s):
        out.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"\b([A-Z][a-z]+)\.? (\d{1,2}),? (\d{4})\b", s):
        if m.group(1).lower() in MONTHS:
            out.append((int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2))))
    return out

def get_ci(d, key):
    """Case/whitespace-insensitive dict lookup (sub-LMs re-case field names)."""
    if key in d:
        return d[key]
    k = norm(key)
    for kk, vv in d.items():
        if norm(kk) == k:
            return vv
    return None

def docket_headers(text):
    out = []
    for m in re.finditer(r"DOCKET FOR CASE #:\s*(\S+)", text):
        blk = text[m.start():m.start() + 3000]
        cut = re.search(r"\n\s*(Plaintiff|Petitioner|Defendant|Respondent|Movant|In Re)\b", blk)
        blk = blk[:cut.start()] if cut else blk[:1500]
        d = {"case": m.group(1).replace("\u2212", "-"), "quote": " ".join(blk.split())[:600]}
        for k, pat in [("filed", r"Date Filed:\s*(\d\d/\d\d/\d{4})"), ("terminated", r"Date Terminated:\s*(\d\d/\d\d/\d{4})"),
                       ("judge", r"Assigned to:\s*(.+?)(?:\s{2,}|\n|Date |Jury)"),
                       ("referred", r"Referred to:\s*(.+?)(?:\s{2,}|\n|Jury|Date )"), ("cause", r"Cause:\s*(.+?)\n")]:
            mm = re.search(pat, blk)
            d[k] = mm.group(1).strip() if mm else None
        rel = re.search(r"Related Cases?:\s*(.*?)(?:Nature of Suit|Jurisdiction|Cause:|Case in other|$)", blk, re.S)
        d["related"] = re.findall(r"\d:\d\d[-\u2212][a-z]{2}[-\u2212]\d+", rel.group(1).replace("\u2212", "-")) if rel else []
        oth = re.search(r"Case in other\s*court:\s*(.*?)(?:Cause:|Plaintiff|Defendant|Nature of Suit|\n\s*\n|$)", blk, re.S)
        d["other_court"] = re.findall(r"\d:\d\d[-\u2212][a-zA-Z]{2}[-\u2212]\d+|(?<![\d:])\d{2}[-\u2212]\d{3,5}\b", oth.group(1)) if oth else []
        d["other_court"] = list(dict.fromkeys(x.replace("\u2212", "-") for x in d["other_court"]))
        out.append(d)
    return out

def clean_judge(s):
    s = re.sub(r"\b(Jury Demand|Date Terminated|Demand).*$", "", s or "")
    s = re.sub(r"\b(Honorable|Hon\.|Chief|Senior|District|Magistrate|U\.S\.|United States|Judge)\b", " ", s)
    return " ".join(s.replace("Jam es", "James").split()).strip(" ,")

HDR = docket_headers(context)
for h in HDR:
    print({k: v for k, v in h.items() if k != "quote"})

def plaintiff_blocks(text, limit=6000):
    """The plaintiffs' party/counsel listings of PACER docket sheets ('Plaintiff' ... 'represented by' ...)."""
    out, seen = [], set()
    for m in re.finditer(r"(?m)^\s*(Plaintiff|Petitioner)s?\s*$", text):
        seg = text[m.end():m.end() + 3000]
        cut = re.search(r"(?m)^\s*(Defendant|Respondent|Intervenor|Amicus|Movant|Interested Party)s?\b", seg)
        seg = seg[:cut.start()] if cut else seg
        if "represented by" not in seg:
            continue
        seg = re.sub(r"(?im)^.*(Email|Fax|Suite|Floor|P\.?O\.? Box|\d{3}[-.)]\s?\d{3}-\d{4}|ATTORNEY TO BE NOTICED|LEAD ATTORNEY|PRO HAC VICE|See above for address|Designation).*$\n?", "", seg)
        seg = " ".join(seg.split())
        k = seg[:200]
        if k not in seen:
            seen.add(k)
            out.append(seg)
        if sum(len(x) for x in out) > limit:
            break
    return "\n".join(out)[:limit]

PCOUNSEL = plaintiff_blocks(context)
print("plaintiff party/counsel listing:", len(PCOUNSEL), "chars;", PCOUNSEL[:300])
print(len(context), "chars")
print(context[:1500])
'''

TURN2_BODY = r'''
FIELD_LINES = "\n".join(f'- "{f}": {d}' for f, d in FIELDS.items())

def split_chunks(text, size):
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            k = text.rfind("\n", i + size // 2, j)
            if k > i:
                j = k
        out.append(text[i:j])
        i = j
    return out

def extract_prompt(part, label):
    return (f"You are reading {label} of a court case file (docket sheet and/or court documents).\n"
            "For each field below, report ONLY what THIS PART states. If this part does not state it, use \"N/A\".\n"
            "For every non-N/A value add \"quote\": a short verbatim quote (under 200 characters) copied exactly from "
            "this part that supports it. Keep names, dates and categories short; give descriptive fields complete, specific values (up to 120 words: who, what statute, what relief, what the court held) and include dates (MM/DD/YYYY) for events.\n\n"
            f"Fields:\n{FIELD_LINES}\n\n"
            'Return ONLY a JSON object {"<field>": {"value": "...", "quote": "..."}, ...} with every field above as a key.\n\n'
            f"=== {label} ===\n{part}\n=== END OF {label} ===")

parts = split_chunks(context, 40000)
records = []                      # (part text, parsed reply)
todo = [(p, f"PART {k + 1} OF {len(parts)}") for k, p in enumerate(parts)]
for rnd in range(3):              # unparseable / failed replies are re-asked on halves
    replies = llm_query_batched([extract_prompt(p, lab) for p, lab in todo])
    retry = []
    for (p, lab), r in zip(todo, replies):
        obj = parse_json(r)
        if obj is not None and len(obj) >= len(FIELDS) // 2:
            records.append((p, obj))
        elif len(p) > 4000:
            h = split_chunks(p, len(p) // 2 + 1)
            retry += [(x, f"{lab} (piece {i + 1} of {len(h)})") for i, x in enumerate(h)]
        else:
            records.append((p, obj or {}))
    print(f"round {rnd}: asked {len(todo)}, parsed {len(todo) - len(retry)}, re-asking {len(retry)} halves")
    todo = retry
    if not todo:
        break
print(len(parts), "parts;", len(records), "records")
'''

TURN3 = r'''
cands = {f: [] for f in FIELDS}
dropped = 0
for p, obj in records:
    pn = norm(p)
    for f in FIELDS:
        v = get_ci(obj, f)
        if isinstance(v, dict):
            val, q = v.get("value", ""), v.get("quote", "")
        else:
            val, q = v, ""
        if isinstance(val, list):
            val = "; ".join(str(x) for x in val)
        val = str(val or "").strip()
        if is_na(val):
            continue
        if not quote_ok(q, pn):          # unsupported value = a sub-LM guess, not evidence
            dropped += 1
            continue
        cands[f].append((val, str(q).strip()))
print("dropped (quote not in source):", dropped)
for f in FIELDS:
    print(f"{len(cands[f]):3d}  {f}: {[v[:60] for v, _ in cands[f][:3]]}")
'''

TURN4_BODY_V4 = r'''
R = lambda k: ROLE.get(k) if ROLE.get(k) in FIELDS else None
EXTRA = {}
if PCOUNSEL and R("counsel"):
    EXTRA[R("counsel")] = ("The docket sheet's plaintiff party/counsel listing (attorney names and their firms or "
                           "organisations):\n" + PCOUNSEL)
if HDR and R("judge"):
    EXTRA[R("judge")] = "Docket sheet header(s):\n" + "\n".join(h["quote"] for h in HDR)[:3000]
    for h in HDR:
        if h.get("judge") and clean_judge(h["judge"]):
            cands[R("judge")].insert(0, (clean_judge(h["judge"]), "[docket sheet header: assigned judge] " + h["quote"]))
if R("decree_dates") and R("rulings"):
    cands[R("decree_dates")] = cands[R("decree_dates")] + cands[R("rulings")]

def reduce_prompt(f):
    lines = "\n".join(f"{i + 1}. {v}\n   quote: \"{q[:300]}\"" for i, (v, q) in enumerate(cands[f][:60]))
    extra = ("\n\n" + EXTRA[f]) if f in EXTRA else ""
    return (f"You are finalizing one field of a structured summary of a court case.\n"
            f"Field: \"{f}\" -- {FIELDS[f]}\n\n"
            f"Findings extracted from different parts of the case file, each with its verbatim supporting quote:\n{lines or '(none)'}"
            f"{extra}\n\n"
            "Write the final value of this field for the whole case:\n"
            "- combine complementary findings (for lists: the distinct items that match the definition, chronological, "
            "separated by '; ', at most 12);\n"
            "- when findings conflict, follow the field definition, else prefer the most specific / latest-dated one;\n"
            "- drop findings whose quote does not actually support this field; answer exactly N/A if none does.\n"
            "Return only the value (no preamble, no field name). Be complete and specific (names, statutes, dates, "
            f"holdings), at most {40 if f == R('judge') else 250} words.")

final, need = {}, []
for f in FIELDS:
    vals = list(dict.fromkeys(v for v, _ in cands[f]))
    if f in EXTRA:
        need.append(f)                          # docket-sheet evidence: always reduce with it
    elif not vals:
        final[f] = "N/A"
    elif len(vals) == 1:
        final[f] = vals[0]
    else:
        need.append(f)
# Python owns what the docket sheet states exactly.
if R("filing_date"):
    fd = sorted((int(h["filed"][6:]), int(h["filed"][:2]), int(h["filed"][3:5])) for h in HDR if h.get("filed")) or \
         sorted(d for v, q in cands[R("filing_date")] for d in find_dates(v) if 1950 < d[0] < 2030)
    if fd:
        y, m, d = fd[0]
        final[R("filing_date")] = f"{m:02d}/{d:02d}/{y}"
        need = [f for f in need if f != R("filing_date")]
if HDR and R("related"):
    rel = list(dict.fromkeys(c for h in HDR for c in h["other_court"] + h["related"]))
    final[R("related")] = "; ".join(rel) if rel else "N/A"
    need = [f for f in need if f != R("related")]
outs = llm_query_batched([reduce_prompt(f) for f in need]) if need else []
for f, o in zip(need, outs):
    o = re.sub(r"^(final value|value)\s*:\s*", "", (o or "").strip().strip('"').strip(), flags=re.I)
    final[f] = "N/A" if (not o or is_na(o)) else o
for f in FIELDS:
    if is_na(final[f]):
        final[f] = "N/A"
for f in ROLE.get("yes_no") or []:
    if f in final and final[f] == "N/A":
        final[f] = "No"
print(json.dumps(final, indent=1, ensure_ascii=False)[:6000])
answer["content"] = json.dumps({f: final[f] for f in FIELDS}, ensure_ascii=False)
answer["ready"] = True
'''

TURN1_V5 = TURN1_V4.split("def plaintiff_blocks(")[0] + r'''ORG_RE = re.compile(r"(LLP|L\.L\.P|LLC|L\.L\.C|PLLC|P\.C\.|\bPC\b|P\.A\.|\bLaw\b|Legal|Liberties|Union|Center|Centre|Project|"
                    r"Foundation|Office|Services|Fund|Committee|Clinic|Institute|Council|Alliance|Coalition|Society|Firm|"
                    r"Associates|Group|Defender|Attorney General|Department|University|School|Justice|Rights|Advocates|&)", re.I)
JUNK_RE = re.compile(r"(Email|@|Fax|Suite|Floor|P\.?O\.? Box|\d{3}\s*[-\u2212.)]\s*\d{3}\s*[-\u2212]\s*\d{4}|ATTORNEY TO BE NOTICED|"
                     r"LEAD ATTORNEY|PRO HAC VICE|See above for address|Designation|TERMINATED|Bar Status|\b[A-Z]{2} \d{5}\b|"
                     r"^\s*\d+ .*\b(Street|St\.|Ave|Avenue|Blvd|Road|Drive|Boulevard|Way|Place|Plaza)\b|Case:|As of:)", re.I)

def plaintiff_counsel(text):
    """The organisations of the plaintiffs' attorneys in the PACER party listings ('Plaintiff' ... 'represented by'
    ...), over EVERY plaintiff block, with how many attorney listings name each; plus whether a block says PRO SE."""
    orgs, pro_se, seen = {}, False, set()
    for m in re.finditer(r"(?m)^\s*(Plaintiff|Petitioner)s?\s*$", text):
        seg = text[m.end():m.end() + 15000]
        cut = re.search(r"(?m)^\s*(V\.|Defendant|Respondent|Intervenor|Amicus|Movant|Interested Party)s?\b", seg)
        seg = seg[:cut.start()] if cut else seg[:4000]
        pro_se = pro_se or bool(re.search(r"\bPRO SE\b", seg))
        key = " ".join(seg.split())[:300]
        if "represented by" not in seg or key in seen:
            continue
        seen.add(key)
        for line in seg.splitlines():
            line = " ".join(line.split("represented by")[-1].split())[:90]
            if len(line) >= 4 and ORG_RE.search(line) and not JUNK_RE.search(line):
                orgs[line] = orgs.get(line, 0) + 1
    if not orgs and not pro_se:
        return ""
    top = sorted(orgs.items(), key=lambda kv: -kv[1])[:25]
    return ("Organisations of the PLAINTIFFS' attorneys in the docket sheet's party listings (number of attorney "
            "listings): " + "; ".join(f"{o} ({n})" for o, n in top) +
            ("\nA plaintiff party listing says PRO SE." if pro_se else ""))

PCOUNSEL = plaintiff_counsel(context)
print("plaintiff counsel:", PCOUNSEL[:600])
print(len(context), "chars")
print(context[:1500])
'''

TURN4_BODY_V5 = r'''
R = lambda k: ROLE.get(k) if ROLE.get(k) in FIELDS else None
EXTRA = {}
if PCOUNSEL and R("counsel"):
    EXTRA[R("counsel")] = PCOUNSEL
if HDR and R("judge"):
    EXTRA[R("judge")] = "Docket sheet header(s):\n" + "\n".join(h["quote"] for h in HDR)[:3000]
    for h in HDR:
        if h.get("judge") and clean_judge(h["judge"]):
            cands[R("judge")].insert(0, (clean_judge(h["judge"]), "[docket sheet header: assigned judge] " + h["quote"]))
if R("decree_dates") and R("rulings"):
    cands[R("decree_dates")] = cands[R("decree_dates")] + cands[R("rulings")]
# Outcome evidence: court rulings that approve / enter a settlement, consent decree, injunction or judgment. The
# settlement date is the court's approval date, and a yes/no field is decided from what the court actually did.
OUTCOME = [c for c in cands[R("rulings")] if re.search(r"settle|consent|approv|injunct|judgment|decree|dismiss|retain",
                                                     c[0] + c[1], re.I)] if R("rulings") else []
if R("settlement_date"):
    cands[R("settlement_date")] = cands[R("settlement_date")] + OUTCOME[:30]
YES_NO = [f for f in (ROLE.get("yes_no") or []) if f in FIELDS]
for f in YES_NO:
    cands[f] = cands[f] + OUTCOME[:30]

def reduce_prompt(f):
    lines = "\n".join(f"{i + 1}. {v}\n   quote: \"{q[:300]}\"" for i, (v, q) in enumerate(cands[f][:60]))
    extra = ("\n\n" + EXTRA[f]) if f in EXTRA else ""
    return (f"You are finalizing one field of a structured summary of a court case.\n"
            f"Field: \"{f}\" -- {FIELDS[f]}\n\n"
            f"Findings extracted from different parts of the case file, each with its verbatim supporting quote:\n{lines or '(none)'}"
            f"{extra}\n\n"
            "Write the final value of this field for the whole case:\n"
            "- combine complementary findings (for lists: the distinct items that match the definition, chronological, "
            "separated by '; ', at most 12);\n"
            "- when findings conflict, follow the field definition, else prefer the most specific / latest-dated one;\n"
            "- drop findings whose quote does not actually support this field; answer exactly N/A if none does.\n"
            "Return only the value (no preamble, no field name). Be complete and specific (names, statutes, dates, "
            f"holdings), at most {6 if f in YES_NO else 40 if f in (R('judge'), R('settlement_date')) else 60 if f == R('counsel') else 250} words.")

final, need = {}, []
for f in FIELDS:
    vals = list(dict.fromkeys(v for v, _ in cands[f]))
    if f in EXTRA or f in YES_NO:
        need.append(f)                          # docket-sheet / outcome evidence: always reduce with it
    elif not vals:
        final[f] = "N/A"
    elif len(vals) == 1:
        final[f] = vals[0]
    else:
        need.append(f)
# Python owns what the docket sheet states exactly.
if R("filing_date"):
    fd = sorted((int(h["filed"][6:]), int(h["filed"][:2]), int(h["filed"][3:5])) for h in HDR if h.get("filed")) or \
         sorted(d for v, q in cands[R("filing_date")] for d in find_dates(v) if 1950 < d[0] < 2030)
    if fd:
        y, m, d = fd[0]
        final[R("filing_date")] = f"{m:02d}/{d:02d}/{y}"
        need = [f for f in need if f != R("filing_date")]
if HDR and R("related"):
    rel = list(dict.fromkeys(c for h in HDR for c in h["other_court"] + h["related"]))
    final[R("related")] = "; ".join(rel) if rel else "N/A"
    need = [f for f in need if f != R("related")]
outs = llm_query_batched([reduce_prompt(f) for f in need]) if need else []
for f, o in zip(need, outs):
    o = re.sub(r"^(final value|value)\s*:\s*", "", (o or "").strip().strip('"').strip(), flags=re.I)
    final[f] = "N/A" if (not o or is_na(o)) else o
for f in FIELDS:
    if is_na(final[f]):
        final[f] = "N/A"
print(json.dumps(final, indent=1, ensure_ascii=False)[:6000])
answer["content"] = json.dumps({f: final[f] for f in FIELDS}, ensure_ascii=False)
answer["ready"] = True
'''

FIELDS_PLACEHOLDER = '''FIELDS = {   # EVERY checklist field from the prompt, verbatim, each with a one-line definition written from the
             # case-summary conventions above (what belongs in it for THIS case, and when it is N/A)
    "<checklist field 1>": "<definition>",
    "<checklist field 2>": "<definition>",
}'''
ROLE_PLACEHOLDER_V5 = '''ROLE = {   # which checklist field (verbatim) plays each role; None when the checklist has no such field
    "filing_date": "<field for the original filing date>",
    "judge": "<field for the presiding judge>",
    "counsel": "<field for the plaintiffs' counsel / type of counsel>",
    "related": "<field for related case numbers>",
    "decree_dates": "<field listing the dates of the court's decrees / rulings>",
    "rulings": "<field describing the court's rulings>",
    "settlement_date": "<field for the date of settlement>",
    "yes_no": ["<every field that is a yes/no question>"],
}'''
ROLE_PLACEHOLDER_V4 = '''ROLE = {   # which checklist field (verbatim) plays each role; None when the checklist has no such field
    "filing_date": "<field for the original filing date>",
    "judge": "<field for the presiding judge>",
    "counsel": "<field for the plaintiffs' counsel / type of counsel>",
    "related": "<field for related case numbers>",
    "decree_dates": "<field listing the dates of the court's decrees / rulings>",
    "rulings": "<field describing the court's rulings>",
    "yes_no": ["<every field that is a yes/no question>"],
}'''

INTRO_V4 = """\
A court case file of 100k-1M+ characters is far too long to read yourself in the turn budget, and the checklist fields \
are spread over dozens of documents (complaint, opinions, orders, settlement, the PACER docket sheet). The reliable way \
is a fixed four-turn program: Python parses the docket-sheet header, ONE batched sub-call fan-out extracts every field \
with a verbatim quote from each ~40k-character part, Python keeps only values whose quote is really in that part, and \
one small batched reduce call per contested field writes the final value from the quoted evidence. The field \
definitions follow the usual case-summary conventions listed below."""

HOW = """\
## How to run it
- Run the four turns below in order, each as ONE ```repl block, copying the code as given; only FIELDS (Turn 2) and \
ROLE (Turn 4) are yours to write, with every checklist field name copied verbatim.
- Do not print or read parts of the file yourself, do not re-check values with extra sub-calls, and do not rewrite \
the reduce output by hand: the program already verifies quotes.
- If a turn raises an error, fix only that line and re-run that turn; never restart from scratch.
- Not every case file contains a PACER docket sheet. If Turn 1 finds no docket header (HDR == []) or no plaintiff \
counsel listing (PCOUNSEL == ""), that is normal: go straight on to Turn 2 unchanged; the fan-out finds those fields \
in the documents. Never switch to searching or reading the file yourself.
- Turn 4 submits; submit by turn 6 at the latest. Unsupported fields get the bare "N/A" (the code does this)."""

CONVENTIONS_V4 = """\
## Case-summary conventions (write FIELDS from these)
- Filing date: the date the ORIGINAL complaint or petition was filed (the docket sheet's "Date Filed"; when a \
transferred case carries several docket sheets, the earliest), MM/DD/YYYY.
- Cause of action: the vehicle AND the claims ("Section 1983 suit alleging <conduct>, in violation of <provisions>"); \
complete, never the vehicle alone.
- Statutory / constitutional basis: every statute and constitutional provision the plaintiffs claim was violated; \
not jurisdiction, venue or fee statutes (28 U.S.C. 1331, 1343, 2201, 42 U.S.C. 1988).
- Remedy sought: every kind of relief requested (declaratory, injunctive, damages, attorneys' fees, ...).
- Counsel: the PLAINTIFFS' lawyers only, never the defendants': name each public-interest, civil-rights, legal-aid, \
clinic or government organisation, then "and private counsel" if private firms also appear; "Private counsel" if \
only private firms; "Pro se" if self-represented.
- Judge: the trial-court judge the case is assigned to (a magistrate judge only when the case itself is assigned to \
them) and any later reassignment ("X, later reassigned to Y"); not appellate panels, not judges of other cases.
- Opinions: bare reporter / Westlaw citations of opinions issued IN THIS case (same caption), never the precedents \
they cite.
- Class action or individual: "Class Action" if brought on behalf of a class (and not denied certification), else \
"Individual".
- Related cases: the case numbers the docket sheet lists under "Related Cases" or "Case in other court"; without a \
docket sheet, the cases a notice or order explicitly calls related. Never case numbers merely cited in briefs or \
opinions (precedents, other litigation). N/A when none.
- Consolidated cases: case numbers an order consolidated into this action; a denied consolidation is N/A.
- Decree / settlement durations: only a term a decree or settlement states (as last extended); an order in force \
"until further order" states no duration, so N/A.
- Decree terms: the obligations a decree or injunction IMPOSED; a denial imposes no terms, so N/A when the court only \
denied relief.
- Monitor, settlement date and amount: only what an order or agreement actually states; the latest (successor \
monitor, amended fund); not nominees, demands or fee awards; N/A when the case has none.
- Court-enforced settlement (yes/no): "Yes" if a settlement or consent decree is approved and retained by the court, \
otherwise "No".
- Parties: "Plaintiff: ...; Defendants: ..." (who they are, briefly).
- Dates of decrees: dates of EVERY substantive decision (consent decrees, injunctions granted or denied, judgments, \
dismissals, class-certification rulings, appellate decisions), date + a few words each; not scheduling or \
administrative orders.
- Court rulings / important filings: every substantive one, chronological, each as "On <date>, the court <outcome>" \
/ "On <date>, <party> filed <filing>"; not extensions of time, appearances, service or page-limit orders.
- Trials: bench or jury trials actually held (dates, outcome); motion hearings and oral arguments are not trials; \
N/A if none.
- Amended / extended / reassigned / substituted values (fund, decree term, deadline, judge, monitor, lead counsel): \
the latest. Lists of dates (monitor reports, motions to enforce): every date.
- Descriptive fields (facts, filings, rulings, terms, appeal): complete and specific, chronological, with dates."""

INTRO_V5 = INTRO_V4.replace("100k-1M+", "100k-3M")

CONVENTIONS_V5 = """\
## Case-summary conventions (write FIELDS from these)
- Filing date: the date the ORIGINAL complaint or petition was filed (the docket sheet's "Date Filed"; when a \
transferred case carries several docket sheets, the earliest), MM/DD/YYYY.
- Cause of action: the vehicle AND the claims ("Section 1983 suit alleging <conduct>, in violation of <provisions>"); \
complete, never the vehicle alone.
- Statutory / constitutional basis: every statute and constitutional provision the plaintiffs claim was violated; \
not jurisdiction, venue or fee statutes (28 U.S.C. 1331, 1343, 2201, 42 U.S.C. 1988).
- Remedy sought: every kind of relief requested (declaratory, injunctive, damages, attorneys' fees, ...).
- Counsel: the PLAINTIFFS' lawyers only, never the defendants' (government law offices such as an attorney general, \
county attorney or city law department count only when the government is the plaintiff). Name each public-interest, \
civil-rights, legal-aid or clinic organisation, then "and private counsel (<firm names>)" if private firms also \
appear; "Private counsel (<firm names>)" if only private firms; "Pro se" if self-represented ("Pro se, later \
represented by ..." when counsel appeared later).
- Judge: the trial-court judge the case is assigned to (a magistrate judge only when the case itself is assigned to \
them) and any later reassignment ("X, later reassigned to Y"); not appellate panels, not judges of other cases.
- Opinions: bare reporter / Westlaw citations of opinions issued IN THIS case (same caption), never the precedents \
they cite.
- Class action or individual: "Class Action" if the complaint is brought on behalf of a class (even if certification \
was never granted); otherwise "Individual, described as <how many and what kind of plaintiffs, e.g. three detainees>", \
without listing names.
- Parties: "Plaintiff: <how many and who they are, as a group>, represented by <counsel>; Defendants: <the \
institutions and officials by role>" -- describe, do not list every name.
- Related cases: the case numbers the docket sheet lists under "Related Cases" or "Case in other court"; without a \
docket sheet, the cases a notice or order explicitly calls related. Never case numbers merely cited in briefs or \
opinions (precedents, other litigation). N/A when none.
- Consolidated cases: case numbers an order consolidated into this action; a denied consolidation is N/A.
- Decrees: a decree is a court order that GRANTS relief (consent decree, injunction granted, judgment ordering the \
defendants to act, court-approved settlement). Decree terms: the obligations a decree IMPOSED; a denial imposes no \
terms, so N/A when the court only denied relief.
- Decree / settlement durations: only a stated term (as last extended); an order in force "until further order" \
states no duration, so N/A.
- Monitor: the person APPOINTED by the court, decree or settlement to oversee compliance (the latest / successor); \
not nominees, expert witnesses or inspectors; N/A when none was appointed. Monitor reports: that monitor's reports, \
every date.
- Date of settlement: the date the court APPROVED or entered the settlement or consent decree (the final approval \
order), not the date the agreement was signed or filed; only when no court approval exists, the date it was signed. \
N/A when the case did not settle.
- Settlement amount: what the agreement requires, as last amended; not demands or fee awards.
- Court-enforced (yes/no): "Yes" if the court entered or approved a consent decree, settlement or injunction that it \
retains jurisdiction to enforce; "No" if the case ended in a private settlement or dismissal without court \
enforcement, or the plaintiffs obtained no court-ordered relief; "N/A" only when the case is still pending with no \
outcome.
- Dates of decrees: dates of EVERY substantive decision (consent decrees, injunctions granted or denied, judgments, \
dismissals, class-certification rulings, appellate decisions), date + a few words each; not scheduling or \
administrative orders.
- Court rulings: every substantive one, chronological, each as "On <date>, the court <outcome>"; not extensions of \
time or page-limit orders.
- Important filings: the key pleadings and motions (complaint, amended complaints, motions for an injunction, to \
dismiss, class certification, to enforce, appeals), each as "On <date>, <who> filed <what>, which the court \
<outcome> on <date>"; not routine briefing (responses, replies), scheduling, status reports, appearances or \
extensions of time.
- Trials: bench or jury trials actually held (dates, outcome); motion hearings and oral arguments are not trials; \
N/A if none.
- Amended / extended / reassigned / substituted values (fund, decree term, deadline, judge, monitor, lead counsel): \
the latest. Lists of dates (monitor reports, motions to enforce): every date.
- Descriptive fields (facts, rulings, terms, appeal): complete and specific, chronological, with dates."""

TURN1, TURN4_BODY, ROLE_PLACEHOLDER, CONVENTIONS, INTRO = TURN1_V5, TURN4_BODY_V5, ROLE_PLACEHOLDER_V5, CONVENTIONS_V5, INTRO_V5

OWN_WORDS = ("In your reasoning, state each rule as your own judgement about the field (\"a denial imposes no "
             "terms\"); never mention, quote or refer to these instructions, a note or a hint.")


def _turn_header(n: int, title: str) -> str:
    return f"## Turn {n}: {title}"


TITLES = ["Inspect the case file; parse the docket-sheet header and the plaintiffs' counsel listing in Python.",
          "Fan out one field-bundle extraction per 40k-char part (every field, value + verbatim quote).",
          "Keep only values whose quote really occurs in the part; collect candidates per field.",
          "Reduce per field over the verified candidates (docket-sheet evidence where it is authoritative) and submit."]


NOTES = ("v4", "v5")
DEFAULT_NOTE = "v5"


def _parts(version: str) -> tuple[str, str, str, str]:
    if version == "v4":
        return TURN1_V4, TURN4_BODY_V4, ROLE_PLACEHOLDER_V4, CONVENTIONS_V4
    if version == "v5":
        return TURN1_V5, TURN4_BODY_V5, ROLE_PLACEHOLDER_V5, CONVENTIONS_V5
    raise ValueError(f"note version {version!r} not in {NOTES}")


def code_turns(fields_block: str = FIELDS_PLACEHOLDER, role_block: str | None = None,
               version: str = DEFAULT_NOTE) -> list[str]:
    t1, t4, role_ph, _ = _parts(version)
    return [t1.strip("\n"), fields_block + "\n" + TURN2_BODY.strip("\n"), TURN3.strip("\n"),
            (role_block or role_ph) + "\n" + t4.strip("\n")]


def note_text(version: str = DEFAULT_NOTE) -> str:
    """The general docket note: ``v5`` (default; general form of notes/t01_note_v5.md) or ``v4``."""
    conv = _parts(version)[3]
    body = [INTRO_V5 if version == "v5" else INTRO_V4, HOW, conv]
    for i, (t, c) in enumerate(zip(TITLES, code_turns(version=version), strict=True)):
        body.append(_turn_header(i + 1, t) + "\n```repl\n" + c + "\n```")
    body.append(OWN_WORDS)
    return "\n\n".join(body)


NOTE_DEFS_V4 = {
    "filing_date": "the date the ORIGINAL complaint was filed (the docket sheet's Date Filed; the earliest when there "
                   "are several docket sheets), MM/DD/YYYY",
    "judge": "the trial-court judge the case is assigned to (a magistrate judge only when the case itself is assigned "
             "to them) and any later reassignment ('X, later reassigned to Y'); not appellate or other cases' judges",
    "class_action": "'Class Action' if brought on behalf of a class and not denied certification, else 'Individual'",
    "counsel": "the organisations representing the PLAINTIFFS only, never the defendants' lawyers",
    "counsel_type": "the PLAINTIFFS' lawyers only, never the defendants': each public-interest / legal-aid / "
                    "civil-rights organisation, then 'and private counsel' if private firms also appear",
    "statutes": "every statute and constitutional provision the plaintiffs claim was violated; not jurisdiction or "
                "fee statutes (28 U.S.C. 1331, 1343, 2201, 42 U.S.C. 1988)",
    "remedies": "every kind of relief the plaintiffs request",
    "cause": "the vehicle AND the claims: '<vehicle> suit alleging <conduct>, in violation of <provisions>', complete",
    "settlement_date": "the date the court approved the settlement; N/A when there is none",
    "settlement_amount": "the settlement fund as last amended (not demands or fee awards); N/A when none",
    "court_enforced": "'Yes' if a settlement or consent decree is approved and retained by the court, otherwise 'No'",
    "decree_duration": "the term a decree states (as last extended); an order in force 'until further order' states "
                       "no duration, so N/A",
    "decree_terms": "the obligations a decree or injunction IMPOSED; a denial imposes no terms, so N/A when the court "
                    "only denied relief",
    "monitor": "the person appointed monitor (the latest / successor); not nominees; N/A when none",
    "monitor_reports": "the filing date of every monitor report; N/A when none",
    "decree_dates": "dates of EVERY substantive decision (decrees, injunctions granted or denied, judgments, "
                    "dismissals, class rulings, appellate decisions), date + a few words each",
    "appeal": "appeals taken in this case: which court and the outcome; N/A when none",
    "trial": "bench or jury trials actually held (date, outcome); motion hearings and oral arguments are not trials; "
             "N/A if none",
    "related": "the case numbers the docket sheet lists under Related Cases / Case in other court (without a sheet, "
               "cases explicitly noticed as related); never case numbers merely cited; N/A when none",
    "consolidated": "case numbers an order consolidated into this action; a denied consolidation is N/A",
    "disputes": "the date of every motion to enforce the settlement or decree; N/A when none",
    "plaintiffs": "the named plaintiffs (people), not their lawyers",
    "defendants": "every defendant",
    "opinions": "citations of opinions issued IN THIS case, never the precedents they cite",
    "facility": "the facility or program the case is about (not one a plaintiff was once held in)",
    "deadline": "the compliance deadline as last extended",
    "lead_counsel": "the plaintiffs' current lead counsel (after any substitution)",
    "rulings": "every substantive ruling, chronological, each as 'On <date>, the court <outcome>'; not extensions of "
               "time or page-limit orders",
    "filings": "every major filing, chronological, each as 'On <date>, <party> filed <filing>'; not appearances or "
               "service",
    "parties": "'Plaintiff: ...; Defendants: ...' (who they are, briefly)",
}
NOTE_DEFS_V5 = dict(NOTE_DEFS_V4)
NOTE_DEFS_V5.update({
    "class_action": "'Class Action' if the complaint is brought on behalf of a class (even if certification was never "
                    "granted); otherwise 'Individual, described as <how many and what kind of plaintiffs>', without "
                    "listing names",
    "counsel_type": "the PLAINTIFFS' lawyers only, never the defendants' (government law offices such as an attorney "
                    "general or county / city attorney count only when the government is the plaintiff): each "
                    "public-interest / legal-aid / civil-rights organisation, then 'and private counsel (<firm "
                    "names>)'; 'Pro se, later represented by ...' when they began self-represented",
    "parties": "'Plaintiff: <how many and who they are, as a group>, represented by <counsel>; Defendants: <the "
               "institutions and officials by role>' -- describe, do not list every name",
    "settlement_date": "the date the court APPROVED or entered the settlement or consent decree (final approval order "
                       "/ decree entry), not the date it was signed or filed; only when no court approval exists, the "
                       "signing date; N/A when the case did not settle",
    "court_enforced": "'Yes' if the court entered or approved a consent decree, settlement or injunction it retains "
                      "jurisdiction to enforce; 'No' if the case ended in a private settlement or dismissal, or with no "
                      "court-ordered relief; 'N/A' only when the case is still pending with no outcome",
    "decree_terms": "the obligations a decree (an order that GRANTS relief) imposed; a denial imposes no terms, so N/A "
                    "when the court only denied relief",
    "monitor": "the person APPOINTED by the court or decree to oversee compliance (the latest / successor); not "
               "nominees, experts or inspectors; N/A when none",
    "filings": "the key pleadings and motions, chronological, each as 'On <date>, <who> filed <what>, which the court "
               "<outcome> on <date>'; not routine briefing (responses, replies), scheduling, status reports, "
               "appearances or extensions",
})
NOTE_DEFS = NOTE_DEFS_V5
V3_DEFS = dict(NOTE_DEFS_V4)
V3_DEFS.update({
    "filing_date": "date the ORIGINAL complaint/petition was filed (the docket's 'Date Filed'), MM/DD/YYYY",
    "judge": "the district judge presiding over THIS case (if it was reassigned: 'X, later reassigned to Y'); not "
             "magistrate judges, not appellate judges, not judges of other cases",
    "counsel_type": "the PLAINTIFFS' lawyers: name every public-interest, legal-aid or civil-rights organisation and "
                    "government office representing them; 'Private counsel' when only private law firms",
    "counsel": "organisations representing the plaintiffs",
    "cause": "the claims brought and the statute / vehicle they are brought under (e.g. 'Section 1983 suit alleging "
             "...', habeas petition)",
    "statutes": "statutes / constitutional provisions the plaintiffs claim were violated",
    "decree_duration": "stated duration / term / termination condition of a consent decree or injunction",
    "decree_terms": "key obligations imposed by the court's decrees / injunctions",
    "related": "docket numbers of OTHER cases explicitly described as related to this one (not this case's own number)",
    "trial": "trials or evidentiary hearings actually held (dates, outcome)",
    "rulings": "the court's rulings on major motions, with dates and outcomes",
    "filings": "EVERY major filing with its date, chronological: complaint, amended complaints, motions (injunction, "
               "dismissal, summary judgment, class certification, enforcement, fees) and notices of appeal",
    "decree_dates": "dates of EVERY substantive decision of the court in this case: consent decrees, TRO / preliminary "
                    "/ permanent injunctions (granted or denied), judgments, dismissals, summary-judgment and "
                    "class-certification rulings, appellate decisions -- not scheduling or administrative orders; "
                    "date + a few words each",
})
ROLE_KEYS = {"filing_date": "filing_date", "judge": "judge", "counsel": "counsel_type", "related": "related",
             "decree_dates": "decree_dates", "rulings": "rulings"}


def fields_block(names_keys: list[tuple[str, str]], defs: dict) -> str:
    d = {n: defs.get(k, "") for n, k in names_keys}
    return "FIELDS = " + json.dumps(d, ensure_ascii=False, indent=1)


def role_block(names_keys: list[tuple[str, str]], header: bool = True, settlement: bool = False) -> str:
    """ROLE as a root following the note writes it; ``settlement=True`` adds note v5's settlement_date role."""
    by_key = {k: n for n, k in names_keys}
    roles = ROLE_KEYS if header else {k: v for k, v in ROLE_KEYS.items() if k in ("filing_date", "decree_dates",
                                                                                 "rulings")}
    if settlement:
        roles = {**roles, "settlement_date": "settlement_date"}
    role = {r: by_key.get(k) for r, k in roles.items()}
    if role.get("counsel") is None and header:
        role["counsel"] = by_key.get("counsel")
    role["yes_no"] = [n for n, k in names_keys if k == "court_enforced"]
    return "ROLE = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in role.items()) + "}"


TURN1_NO_HEADER = TURN1_V4.split("def docket_headers(text):")[0] + 'HDR, PCOUNSEL = [], ""\nprint(len(context), "chars")\nprint(context[:1500])\n'


C6_INTRO = """\
A court case file of 100k-3M characters is far too long to read yourself, and the checklist fields are spread over \
many documents (complaint, opinions, orders, settlement, the PACER docket sheet). The reliable way is a fixed \
three-turn program: Python pulls the docket-sheet header and the plaintiffs' counsel listing, and a cheap probe \
sub-call reads samples of the file and reports its structure (you never print the file); ONE batched fan-out of 4-16 \
large parts (~60k+ characters each) extracts notes on every field, with the field definitions sent once inside that \
prompt; then one small batched call per contested field writes its final value from all the notes. The probe's case \
overview goes into every extraction and reduce prompt, so each sub-call knows which case it is reading. Field \
definitions follow the case-summary (Clearinghouse) conventions below."""

C6_RULES = """\
## Rules
- Three turns, each ONE ```repl block, copying the code below; you write only FIELDS (Turn 2) and X (Turn 3). Keep \
your own code and prose short: one plain sentence before each block.
- Turn 1 is this code from the start: do NOT print, slice or grep the file yourself (the probe reads it for you), do \
not add sub-calls, do not edit the values the program returns.
- Many case files have no PACER docket sheet: empty H / P / FD / REL is normal. Go straight on to Turn 2 unchanged; \
the fan-out finds those fields in the documents. Never switch to printing, slicing or searching the file yourself, \
and never inspect the notes before Turn 3.
- If a turn raises an error, fix only that line and re-run it. Turn 3 submits."""

C6_CONVENTIONS = """\
## Conventions (write each as a SHORT clause in FIELDS, no placeholders in angle brackets)
- Filing date: original complaint's date, MM/DD/YYYY. Judge: the trial judge the case is assigned to, "X, later \
reassigned to Y"; no appellate or magistrate judges unless assigned. Opinions: citations of opinions issued IN this \
case only. Related / consolidated: other cases' docket numbers.
- Cause of action: one sentence, the vehicle and the claims ("Section 1983 suit alleging ...", "habeas petition under \
28 U.S.C. 2241 challenging ..."). Statutory basis: only the statutes and constitutional provisions plaintiffs claim \
were violated (not jurisdiction, venue, procedure or fee statutes). Remedy: every kind of relief requested.
- Counsel: plaintiffs' lawyers only (a government office only when the government sues): name the public-interest, \
legal-aid, clinic or government organisations, then "and private counsel (firm names)"; or "Private counsel (firm \
names)"; or "Pro se, later represented by ...".
- Class: "Class Action" if brought for a class, even if never certified; else "Individual, described as" how many and \
what kind. Parties: "Plaintiff:" the group, "represented by" counsel; "Defendants:" institutions and officials by role.
- Decree: an order that GRANTS relief (consent decree, injunction granted, judgment ordering action, approved \
settlement); denials and dismissals are not. Duration: a stated term only ("five years", "until <date>"), else N/A. \
Settlement date: the date the court approved / entered it.
- Court-enforced: "Yes" if the court entered or approved a decree, settlement or injunction it retains jurisdiction \
to enforce; "No" if a private settlement, voluntary dismissal, or no court-ordered relief; "N/A" only while pending; \
at most 6 words.
- Monitor: appointed by the court or decree to oversee compliance (not experts or party inspectors). Monitor reports: \
that monitor's reports or the status reports defendants were ordered to file.
- Dates of decrees: every substantive ruling, date + a few words. Filings: every key pleading and motion, "On <date>, \
<who> filed <what>" + its outcome, no routine entries.
- Factual basis: a narrative summary (who the plaintiffs are, what happened to them, what policy or conduct they \
challenge), not a timeline.
- Other descriptive fields (rulings, terms, appeal, trials, disputes): complete, specific, chronological sentences \
with dates."""

TURN1_C6 = r'''import re, json
H = "\n".join(context[m.start():m.start() + 2000] for m in re.finditer(r"DOCKET FOR CASE #", context))[:6000]
B = [re.split(r"(?m)^\s*(?:V\.|Defendants?|Respondents?)\b", context[m.end():m.end() + 8000])[0]
     for m in re.finditer(r"(?m)^\s*(?:Plaintiff|Petitioner)s?\s*$", context)]
P = "\n".join(dict.fromkeys(l.strip() for b in B for l in b.splitlines() if l.strip()))[:20000]
FD = sorted(re.findall(r"Date Filed:\s*(\d\d)/(\d\d)/(\d{4})", H), key=lambda d: d[::-1])
REL = list(dict.fromkeys(re.findall(r"\d:\d\d-[a-z]{2}-\d+|\b\d\d-\d{3,5}\b", " ".join(re.findall(r"(?:Related Cases?|Case in "
      r"other\s*court):(.*?)(?:Nature of Suit|Cause:|Plaintiff|Defendant|\n\s*\n)", H.replace("\u2212", "-"), re.S)))))
S = context[:15000] + "".join(f"\n[... at character {i} ...]\n" + context[i:i + 3000]
                              for i in range(len(context) // 6, len(context) - 3000, len(context) // 6))
OV = llm_query(f"Samples of a court case file of {len(context)} characters (its start, then spread samples):\n{S}\n\n"
    "In at most 120 words report: the case (caption, court, docket number, filing year); whether it contains a PACER "
    "docket sheet; which kinds of documents appear (complaint, opinions, orders, consent decree, settlement, appeals, "
    "monitor reports ...); the date range; and the outcome if visible.")
print(OV, "\n", len(context), "chars;", len(B), "plaintiff blocks; filed", FD[:2], "related", REL)'''

FIELDS_PLACEHOLDER_C6 = ('FIELDS = {"<checklist field, verbatim>": "<short convention clause or \'\'>"}   '
                         '# EVERY checklist field')

TURN2_C6_BODY = r'''n = min(16, max(4, -(-len(context) // 60000))); L = -(-len(context) // n)
ask = lambda t: (f"Case overview: {OV}\n\nBelow is one part of this court case file. For each field it gives "
    "information on, write one line '<field name>: <value>' (specific: names, statutes, holdings, dates MM/DD/YYYY; "
    "names / dates / categories short, descriptive fields up to 120 words with every event and date; lists as 'a; b; c'). Skip fields this part "
    "says nothing about.\n\nFields:\n" + "\n".join(f"- {f}: {d}" for f, d in FIELDS.items()) +
    f"\n\n=== PART ===\n{t}\n=== END ===")
K = {f.lower(): f for f in FIELDS}
N = {f: [] for f in FIELDS}
for r in llm_query_batched([ask(context[i * L:(i + 1) * L]) for i in range(n)]):
    for l in r.splitlines():
        k, _, v = l.strip(" -*").partition(":")
        if k.strip(' *"').lower() in K and v.strip(' *"') and not v.strip(' *"').upper().startswith("N/A"):
            N[K[k.strip(' *"').lower()]].append(v.strip(' *"'))
print(n, "parts", {f[:25]: len(v) for f, v in N.items()})'''

TURN3_C6_HEAD = r'''RUL = "\n".join(N["<court rulings field>"] + N["<dates of decrees field>"])
X = {"<counsel field>": P, "<judge field>": H, "<settlement date field>": RUL,
     "<court-enforced field>": RUL}'''

TURN3_C6_BODY = r'''red = lambda f: (f'Case: {OV}\n\nCase-summary field "{f}" ({FIELDS[f] or "as named"}). Notes on it from different '
    "parts of the case file:\n" + "\n".join("- " + v for v in N[f]) + (f"\n\nAlso relevant:\n{X[f]}" if X.get(f) else "")
    + "\n\nWrite the field's final value for the whole case: merge the notes, keeping every distinct item, event "
    "and date that fits the definition (chronological, '; '-separated); on conflicts follow the definition, else the "
    "most specific / latest. Return only the value, at most 250 words (N/A only if the definition excludes every note).")
need = [f for f in FIELDS if len(set(N[f])) > 1 or X.get(f)]
out = dict(zip(need, llm_query_batched([red(f) for f in need])))
final = {f: (out.get(f) or (N[f] or ["N/A"])[0]).strip() for f in FIELDS}'''

TURN3_C6_TAIL = r'''if FD:
    final["<filing date field>"] = "/".join(FD[0])
if H:
    final["<related cases field>"] = "; ".join(REL) or "N/A"
answer["content"] = json.dumps(final, ensure_ascii=False)
answer["ready"] = True
print(final)'''

C6_TITLES = ["docket header + counsel listing in Python; a probe sub-call reports the file's structure.",
             "FIELDS, then one fan-out over large parts (the overview goes into every prompt).",
             "reduce each contested field over all its notes (header / counsel / rulings evidence where it decides) "
             "and submit."]
C6_OWN_WORDS = ("Write that sentence in your own words, as the expert summarising the case; never mention, quote or "
                "refer to these instructions, a note, a hint or rules.")


def code_turns_c6(fields_block: str = FIELDS_PLACEHOLDER_C6, turn3_head: str = TURN3_C6_HEAD,
                  turn3_tail: str = TURN3_C6_TAIL) -> list[str]:
    return [TURN1_C6, fields_block + "\n" + TURN2_C6_BODY, turn3_head + "\n" + TURN3_C6_BODY + "\n" + turn3_tail]


def note_text_c6() -> str:
    """The general docket note c6 (= notes/t01_note_c6.md, code verbatim) + the own-words line."""
    body = [C6_INTRO, C6_RULES + "\n- " + C6_OWN_WORDS, C6_CONVENTIONS]
    for i, (t, c) in enumerate(zip(C6_TITLES, code_turns_c6(), strict=True)):
        body.append(_turn_header(i + 1, t) + "\n```repl\n" + c + "\n```")
    return "\n\n".join(body)


C6_DEFS = {
    "filing_date": "original complaint's date, MM/DD/YYYY",
    "judge": "the trial judge the case is assigned to, 'X, later reassigned to Y'; no appellate or magistrate judges "
             "unless assigned",
    "class_action": "'Class Action' if brought for a class, even if never certified; else 'Individual, described as' "
                    "how many and what kind",
    "counsel": "plaintiffs' lawyers only: the organisations representing the plaintiffs",
    "counsel_type": "plaintiffs' lawyers only (a government office only when the government sues): the "
                    "public-interest, legal-aid, clinic or government organisations, then 'and private counsel (firm "
                    "names)'; or 'Pro se, later represented by ...'",
    "statutes": "only the statutes and constitutional provisions plaintiffs claim were violated (not jurisdiction, "
                "venue, procedure or fee statutes)",
    "remedies": "every kind of relief requested",
    "cause": "one sentence, the vehicle and the claims ('Section 1983 suit alleging ...')",
    "settlement_date": "the date the court approved / entered it",
    "settlement_amount": "the settlement fund as last amended",
    "court_enforced": "'Yes' if the court entered or approved a decree, settlement or injunction it retains "
                      "jurisdiction to enforce; 'No' if a private settlement, voluntary dismissal, or no court-ordered "
                      "relief; 'N/A' only while pending; at most 6 words",
    "decree_duration": "a stated term only ('five years', 'until <date>'), else N/A",
    "decree_terms": "what an order that GRANTS relief requires; denials and dismissals are not decrees",
    "monitor": "appointed by the court or decree to oversee compliance (not experts or party inspectors)",
    "monitor_reports": "that monitor's reports or the status reports defendants were ordered to file",
    "decree_dates": "every substantive ruling, date + a few words",
    "appeal": "complete, specific, chronological sentences with dates",
    "trial": "complete, specific, chronological sentences with dates",
    "related": "other cases' docket numbers",
    "consolidated": "other cases' docket numbers",
    "disputes": "complete, specific, chronological sentences with dates",
    "plaintiffs": "",
    "defendants": "",
    "opinions": "citations of opinions issued IN this case only",
    "facility": "",
    "deadline": "",
    "lead_counsel": "",
    "rulings": "complete, specific, chronological sentences with dates",
    "filings": "every key pleading and motion, 'On <date>, <who> filed <what>' + its outcome, no routine entries",
    "parties": "'Plaintiff:' the group, 'represented by' counsel; 'Defendants:' institutions and officials by role",
}


def fields_block_c6(names_keys: list[tuple[str, str]]) -> str:
    return "FIELDS = " + json.dumps({n: C6_DEFS.get(k, "") for n, k in names_keys}, ensure_ascii=False, indent=1)


def turn3_c6(names_keys: list[tuple[str, str]]) -> tuple[str, str]:
    """Turn 3's head (RUL / X) and tail with the placeholders filled as a root following c6 does; roles the checklist
    lacks are dropped (their lines / entries are left out)."""
    by = {k: n for n, k in names_keys}
    rul = [by[k] for k in ("rulings", "decree_dates") if k in by]
    head = "RUL = \"\\n\".join(" + (" + ".join(f"N[{json.dumps(n)}]" for n in rul) if rul else "[]") + ")\n"
    xs = [(by.get("counsel_type") or by.get("counsel"), "P"), (by.get("judge"), "H"),
          (by.get("settlement_date"), "RUL"), (by.get("court_enforced"), "RUL")]
    head += "X = {" + ", ".join(f"{json.dumps(n)}: {v}" for n, v in xs if n) + "}"
    tail = ""
    if "filing_date" in by:
        tail += f"if FD:\n    final[{json.dumps(by['filing_date'])}] = \"/\".join(FD[0])\n"
    if "related" in by:
        tail += f"if H:\n    final[{json.dumps(by['related'])}] = \"; \".join(REL) or \"N/A\"\n"
    tail += 'answer["content"] = json.dumps(final, ensure_ascii=False)\nanswer["ready"] = True\nprint(final)'
    return head, tail
