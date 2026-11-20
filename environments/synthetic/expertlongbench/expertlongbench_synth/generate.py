"""Procedural case files for the synthetic ExpertLongBench env."""

from __future__ import annotations

import datetime as dt
import gc
import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from synth_common import Generated

from .textkit import (
    CITIES,
    COUNTIES,
    FIRST,
    LAST,
    ORDINAL,
    STATES,
    fmt_amount,
    fmt_caseno,
    fmt_cite,
    fmt_date,
    fmt_duration,
    fmt_person,
    iso,
    wrap,
)

DOCKET_FIELDS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "filing_date": ("date", "earliest", ["Filing Date", "Date the Original Complaint Was Filed"],
                    ["filing date", "date filed"]),
    "judge": ("person", "latest", ["First and Last name of Judge", "Presiding Judge (after any reassignment)"],
              ["presiding judge", "name of judge", "name of the judge", "judge"]),
    "class_action": ("enum", "latest", ["Class Action or Individual Plaintiffs", "Class Action or Individual Suit"],
                     ["class action"]),
    "counsel": ("orgs", "union", ["Plaintiffs' Counsel (organizations)", "Organizations Representing the Plaintiffs"],
                ["counsel", "representing the plaintiffs"]),
    "statutes": ("statutes", "union", ["Statutory or Constitutional Basis for the Claims",
                                       "Laws the Plaintiffs Say Were Violated"],
                 ["statutory", "constitutional basis", "laws"]),
    "remedies": ("remedies", "union", ["Remedy Sought", "Relief Requested by the Plaintiffs"],
                 ["remedy", "relief requested", "relief sought"]),
    "settlement_date": ("date", "single", ["Date of Settlement", "Date the Court Approved the Settlement"],
                        ["date of settlement", "settlement date", "approved the settlement", "settlement approval",
                         "approval date"]),
    "settlement_amount": ("amount", "latest", ["Settlement Amount (as last amended)", "Total Settlement Fund"],
                          ["settlement amount", "settlement fund", "fund amount", "amount of the settlement"]),
    "court_enforced": ("yesno", "single", ["Whether the Settlement is Court-enforced or Not",
                                           "Does the Court Retain Jurisdiction to Enforce the Settlement"],
                       ["court enforced", "retain jurisdiction"]),
    "decree_duration": ("duration", "latest", ["How Long Decrees will Last",
                                               "Term of the Consent Decree (as last extended)"],
                        ["how long decree", "term of the consent decree", "decree will last", "decrees will last",
                         "decree term", "duration of the decree"]),
    "monitor": ("person", "latest", ["Name of the Monitor", "Monitor Currently Appointed"],
                ["name of the monitor", "monitor currently", "monitor name", "who is the monitor", "appointed monitor"]),
    "monitor_reports": ("dates", "union", ["Monitor Reports (filing dates)", "Dates of the Monitor's Reports"],
                        ["monitor reports", "monitor s reports", "monitor report"]),
    "decree_dates": ("dates", "union", ["Dates of All Decrees", "Dates of Injunctions, Decrees and Judgments Entered"],
                     ["dates of all decrees", "decrees and judgments", "decree dates", "dates of decrees",
                      "dates of injunctions"]),
    "appeal": ("appeal", "latest", ["Appeal (court and outcome)", "Appeal"], ["appeal", "appellate"]),
    "trial": ("date", "single", ["Trial Date", "Date the Trial Began"], ["trial date", "trial began", "trial"]),
    "related": ("casenos", "union", ["Related Cases Listed by Their Case Code Number", "Related Case Numbers"],
                ["related case"]),
    "consolidated": ("casenos", "union", ["Consolidated Cases Noted", "Case Numbers Consolidated Into This Action"],
                     ["consolidated"]),
    "disputes": ("dates", "union", ["Disputes Over Settlement Enforcement (dates of motions)",
                                    "Motions to Enforce the Settlement or Decree (dates)"],
                 ["disputes over settlement", "motions to enforce", "motion to enforce", "enforcement disputes"]),
    "plaintiffs": ("persons", "union", ["Named Plaintiffs", "Plaintiffs (named individuals)"], ["named plaintiffs"]),
    "defendants": ("entities", "union", ["Defendants", "Who are the Defendants"], ["defendants"]),
    "opinions": ("cites", "union", ["All Reported Opinions Cited with Shortened Bluebook Citation",
                                    "Citations of Opinions Issued in This Case"],
                 ["reported opinions", "opinions issued", "bluebook"]),
    "facility": ("entity", "single", ["Facility or Program at Issue", "Institution at the Center of the Case"],
                 ["facility", "institution"]),
    "deadline": ("date", "latest", ["Compliance Deadline (as last extended)", "Deadline for Defendants' Compliance"],
                 ["compliance deadline", "deadline"]),
    "lead_counsel": ("entity", "latest", ["Current Lead Counsel for Plaintiffs", "Lead Counsel for the Plaintiffs"],
                     ["lead counsel", "counsel changes", "substitution of counsel"]),
    "counsel_type": ("counsel_type", "union", ["Type of Counsel", "Kind of Lawyers Representing the Plaintiffs"],
                     ["type of counsel", "kind of lawyers", "kind of counsel"]),
    "cause": ("cause", "union", ["Cause of Action", "Claims and the Legal Vehicle They Are Brought Under"],
              ["cause of action", "legal vehicle"]),
    "rulings": ("events", "union", ["Court Rulings", "The Court's Rulings on Major Motions"],
                ["court rulings", "rulings on major motions"]),
    "filings": ("events", "union", ["Note Important Filings", "Major Filings in the Case"],
                ["important filings", "major filings"]),
    "decree_terms": ("terms", "union", ["Significant Terms of Decrees", "Obligations Imposed by the Court's Decrees"],
                     ["terms of decrees", "obligations imposed"]),
    "parties": ("parties", "union", ["Parties to the Case", "Who the Plaintiffs and Defendants Are"],
                ["parties to the case"]),
}
T01_KEYS = ("counsel_type", "cause", "rulings", "filings", "decree_terms")
T01V5_KEYS = ("parties",)
V5_PARAMS = ("t01v5", "p_roster", "p_pro_se", "p_approval_missing", "p_ended_no_relief")
PLAINTIFF_KIND = [("Juvenile Detention", "youth", "youths"), ("Jail", "detainee", "detainees"),
                  ("State Hospital", "patient", "patients"), ("School District", "student", "students"),
                  ("Department of Corrections", "prisoner", "prisoners"), ("Police Department", "resident", "residents"),
                  ("Social Services", "benefit applicant", "benefit applicants"), ("Housing Authority", "tenant", "tenants")]
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
             "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
GOV_LAW_OFFICE = re.compile(r"Attorney General|County Attorney|City Attorney|Law Department", re.I)

CLINIC_FIELDS: dict[str, tuple[str, str, list[str], list[str]]] = {
    "age": ("int", "single", ["Patient Age"], ["age"]),
    "sex": ("enum", "single", ["Patient Sex"], ["sex", "gender"]),
    "reason": ("entity", "single", ["Reason for the Visit"], ["reason for", "chief complaint"]),
    "onset": ("days", "latest", ["Onset (how long ago symptoms began)"], ["onset"]),
    "location": ("entity", "latest", ["Location of Symptoms"], ["location"]),
    "severity": ("int", "latest", ["Severity (0-10, latest reported)"], ["severity"]),
    "meds": ("words", "union", ["Current Medications"], ["medications", "current medication"]),
    "allergies": ("words", "union", ["Allergies"], ["allerg"]),
    "surgical": ("words", "union", ["Surgical History"], ["surgical", "surgeries"]),
    "family": ("words", "union", ["Family History"], ["family history"]),
    "smoking": ("enum", "latest", ["Smoking Status"], ["smoking", "tobacco"]),
    "occupation": ("entity", "single", ["Employment"], ["employment", "occupation", "job"]),
    "bp": ("bp", "latest", ["Blood Pressure (latest reading)"], ["blood pressure"]),
    "diagnosis": ("entity", "single", ["Diagnosis and Clinical Impression"], ["diagnosis", "impression"]),
    "prescribed": ("words", "union", ["Medication Prescribed Today"], ["prescribed"]),
    "followup": ("days", "single", ["Follow-up Interval"], ["follow up", "follow-up"]),
}

STATUTES = {
    "42usc1983": (["42 U.S.C. § 1983", "Section 1983 of Title 42"], "claim"),
    "ada": (["Title II of the Americans with Disabilities Act", "the ADA, 42 U.S.C. § 12132"], "claim"),
    "rehab504": (["Section 504 of the Rehabilitation Act", "29 U.S.C. § 794"], "claim"),
    "titlevii": (["Title VII of the Civil Rights Act of 1964", "42 U.S.C. § 2000e-2"], "claim"),
    "fha": (["the Fair Housing Act", "42 U.S.C. § 3604"], "claim"),
    "amend8": (["the Eighth Amendment", "the Eighth Amendment's ban on cruel and unusual punishment"], "claim"),
    "amend14dp": (["the Due Process Clause of the Fourteenth Amendment", "Fourteenth Amendment due process"], "claim"),
    "amend14ep": (["the Equal Protection Clause", "Fourteenth Amendment equal protection"], "claim"),
    "amend1": (["the First Amendment", "the First Amendment's free speech guarantee"], "claim"),
    "amend4": (["the Fourth Amendment", "the Fourth Amendment's bar on unreasonable seizures"], "claim"),
    "idea": (["the Individuals with Disabilities Education Act", "20 U.S.C. § 1400"], "claim"),
    "apa": (["the Administrative Procedure Act", "5 U.S.C. § 706"], "claim"),
    "28usc1331": (["28 U.S.C. § 1331"], "jurisdiction"),
    "28usc1343": (["28 U.S.C. § 1343"], "jurisdiction"),
    "28usc2201": (["28 U.S.C. § 2201"], "jurisdiction"),
    "42usc1988": (["42 U.S.C. § 1988"], "jurisdiction"),
}
CAUSE_CODES = {"42usc1983": "42:1983 Civil Rights Act", "ada": "42:12132 Americans with Disabilities Act",
               "rehab504": "29:794 Rehabilitation Act", "titlevii": "42:2000e Job Discrimination (Employment)",
               "fha": "42:3604 Fair Housing Act", "idea": "20:1400 Individuals with Disabilities Education Act",
               "apa": "05:706 Administrative Procedure Act"}
REMEDIES = {
    "declaratory": ["a declaratory judgment", "declaratory relief"],
    "injunctive": ["preliminary and permanent injunctive relief", "an injunction"],
    "damages": ["compensatory damages", "money damages"],
    "punitive": ["punitive damages"],
    "fees": ["reasonable attorneys' fees and costs", "attorney's fees"],
}
CONDUCT = ["denial of adequate medical care", "chronic overcrowding", "use of solitary confinement on juveniles",
           "failure to provide sign-language interpreters", "racially discriminatory hiring practices",
           "unlawful strip searches", "denial of special-education services", "retaliation against protesters",
           "inaccessible polling places", "delays in processing benefit applications", "excessive force by guards",
           "unsanitary living conditions", "denial of mental-health treatment", "discriminatory rental practices"]
FACILITY_T = ["{c} County Jail", "{city} Juvenile Detention Center", "{st} State Hospital at {city}",
              "{city} Unified School District", "{st} Department of Corrections {l} Unit", "{city} Police Department",
              "{c} County Department of Social Services", "{city} Housing Authority"]
DEF_T = ["{c} County", "Sheriff {p} of {c} County", "{st} Department of Corrections", "City of {city}",
         "Director {p} of the {st} Department of Health", "{city} Housing Authority", "Warden {p}",
         "{st} Board of Education", "Commissioner {p}", "{c} County Board of Commissioners"]
PUBLIC_ORGS = ["{st} Civil Liberties Union Foundation", "Southern Center for {w} Rights", "{st} Legal Aid Society",
               "National {w} Law Project", "{city} Legal Defense Fund", "{st} Disability Rights Center",
               "Center for {w} Justice", "{st} Appleseed Center for Law and Justice"]
PRIVATE_ORGS = ["{a} {b} & {c} LLP", "{a} & {b} PLLC", "Law Offices of {p}", "{a}, {b} & {c} P.C."]
DEFENSE_ORGS = ["Office of the {st} Attorney General", "{c} County Attorney's Office", "{a} & {b} LLC",
                "City Attorney of {city}"]
WORDS = ["Human", "Prisoners'", "Disability", "Youth", "Immigrant", "Housing", "Voting", "Workers'", "Civil"]
CONTENT_FILL = [
    "Defendants dispute the characterization of {topic} offered by {who} in the declaration dated {d}.",
    "According to the {ord} quarterly audit of {fac}, {n} complaints were logged between {d} and {d2}.",
    "Witness {who} testified at deposition on {d} that staff at {fac} rarely documented {topic}.",
    "The parties exchanged {n} pages of records concerning {topic} under the protective order entered on {d}.",
    "Exhibit {n} to the declaration of {who} summarizes {topic} during the period ending {d}.",
    "{who}, a {role} at {fac}, stated in an affidavit sworn on {d} that {topic} had worsened since {d2}.",
    "Records produced on {d} show that {n} residents of {fac} filed grievances concerning {topic}.",
    "In a letter dated {d}, counsel for {side} asked {who} to preserve all documents relating to {topic}.",
    "The inspection of {fac} conducted on {d} by {who} identified {n} deficiencies related to {topic}.",
    "Plaintiffs' expert {who} opined in a report dated {d} that {topic} fell below accepted standards.",
]
KEYWORD_NOISE = [
    "Staff at {fac} were instructed to monitor {topic} every two hours, according to the log produced on {d}.",
    "A cardiac monitor in the {fac} infirmary malfunctioned on {d}, which {who} reported to maintenance.",
    "The monitoring sheets for {topic} kept by {who} between {d} and {d2} are incomplete.",
    "Residents of the old {city} settlement district were transferred to {fac} on {d}.",
    "Foundation settlement in the east wing of {fac} was noted in the engineering report dated {d}.",
    "Under the internal grievance policy, a resident may appeal a denial of {topic} within 14 days; {n} appeals were filed by {d}.",
    "The appeal process for disciplinary findings at {fac} was revised on {d} by {who}.",
    "A trial period for the new {topic} schedule began at {fac} on {d} and ended on {d2}.",
    "Administrative Law Judge {who} heard an unrelated benefits matter involving a resident of {fac} on {d}.",
    "Staff were expected to use professional judgment regarding {topic}, according to {who}'s affidavit of {d}.",
    "The internal deadline for answering grievances about {topic} was ten business days as of {d}.",
    "{who}, a licensed counselor, met with residents of {fac} for group counseling on {d}.",
    "General counsel for the agency circulated a memorandum on {topic} dated {d}.",
    "A training class on {topic} for {n} employees of {fac} was held on {d}.",
    "Funding requests for {topic} at {fac} totaled ${n},000 in the budget submitted on {d}.",
    "The facility's maintenance fund paid for repairs related to {topic} on {d}.",
    "Visiting hours at {fac} were extended on {d}, and the extension was reported by {who}.",
    "Consolidation of the two kitchens at {fac} was completed on {d}, according to {who}.",
    "Several complaints related to {topic} were consolidated into one internal review on {d}.",
    "Enforcement of the visiting rules at {fac} was described by {who} in a statement dated {d}.",
    "Amendments to the {fac} resident handbook concerning {topic} took effect on {d}.",
    "Some staff were reassigned to the north unit of {fac} on {d} after {topic} complaints rose.",
    "The handbook defines short-term housing at {fac} as a stay of less than {n} days, per the edition of {d}.",
    "The proposed revision of the {topic} policy, circulated on {d}, was never adopted by {fac}.",
    "{who} described the classification of residents at {fac} during a meeting held on {d}.",
]
LEGAL_FILL = [
    "To survive a motion to dismiss, a complaint must contain sufficient factual matter accepted as true; see "
    "{case1}, {cite1}.",
    "The Court must view the evidence in the light most favorable to the non-moving party, as explained in "
    "{case1}, {cite1}.",
    "A plaintiff seeking preliminary relief must show a likelihood of success on the merits, see {case1}, {cite1}, "
    "and the balance of equities considered in {case2}.",
    "Defendants rely on {case1}, {cite1}, but that decision addressed a materially different record from {d}.",
    "As the {cir} Circuit explained in {case1}, {cite1}, deliberate indifference requires more than negligence.",
    "Qualified immunity shields officials unless the right was clearly established, see {case1}, {cite1}.",
    "The standing inquiry focuses on the plaintiffs at the time the complaint was filed, see {case1}, {cite1}.",
    "Courts in this district have applied {case1}, {cite1}, to claims arising from facts like those alleged on {d}.",
]
TOPICS = ["sick-call response times", "staffing ratios", "classification of detainees", "use-of-force reports",
          "medication distribution", "intake screening", "grievance processing", "interpreter requests",
          "recreation schedules", "suicide-watch checks", "transport delays", "translation of notices",
          "maintenance of ventilation systems", "accessibility of entrances", "caseworker visits"]
ROLES = ["nurse supervisor", "shift lieutenant", "case manager", "facility administrator", "staff psychologist",
         "maintenance director", "intake officer", "program coordinator"]


T01_CHAR_QUANTILES = [(0.00, 107_889), (0.05, 129_195), (0.10, 147_371), (0.20, 163_667), (0.30, 190_683),
                      (0.40, 220_799), (0.50, 249_589), (0.60, 323_979), (0.70, 442_441), (0.80, 590_683),
                      (0.90, 740_755), (0.95, 1_030_541), (1.00, 3_363_588)]


def sample_chars(rng: random.Random, P: dict) -> int:
    """Target context size for a docket row. ``chars="t01"`` (default) follows T01's length distribution;
    ``chars=[lo, hi]`` samples uniformly."""
    spec = P.get("chars", "t01")
    cap = int(P.get("max_chars", 1_200_000))
    if isinstance(spec, (list, tuple)):
        return min(cap, rng.randint(int(spec[0]), int(spec[1])))
    u = rng.random()
    for (p0, c0), (p1, c1) in zip(T01_CHAR_QUANTILES, T01_CHAR_QUANTILES[1:], strict=False):
        if u <= p1:
            return min(cap, int(c0 + (c1 - c0) * (u - p0) / (p1 - p0)))
    return cap


@dataclass
class Builder:
    rng: random.Random
    facts: list = field(default_factory=list)

    def fact(self, key: str, kind: str, val: Any, date: dt.date, text: str, item: str = "", trap: str = "") -> dict:
        f = {"id": len(self.facts), "key": key, "kind": kind, "val": val, "date": iso(date),
             "text": " ".join(text.split()), "item": item}
        if trap:
            f["trap"] = trap
        self.facts.append(f)
        return f

    used: set = field(default_factory=set)

    def person(self) -> list:
        """A fresh person; key people (created first) get distinct last names, so 'first + last' identifies them."""
        r = self.rng
        last = r.choice(LAST)
        for _ in range(12):
            if last not in self.used:
                break
            last = r.choice(LAST)
        self.used.add(last)
        return [r.choice(FIRST), r.choice("ABCDEFGHJKLMNPRSTW") if r.random() < 0.6 else None, last]

    def extra(self) -> list:
        """A background person (witness, reporter, expert): no uniqueness bookkeeping, so filler stays cheap."""
        r = self.rng
        return [r.choice(FIRST), r.choice("ABCDEFGHJKLMNPRSTW") if r.random() < 0.6 else None, r.choice(LAST)]

    def date_after(self, d: dt.date, lo: int, hi: int) -> dt.date:
        return d + dt.timedelta(days=self.rng.randint(lo, hi))

    def ini(self) -> str:
        return "".join(self.rng.choice("abcdefghjklmnprstw") for _ in range(3))


def _pfmt(b: Builder, p: list, title: str = "") -> str:
    return fmt_person(p, b.rng.randrange(2), title)


def _v5_party_blocks(b: Builder, r3: random.Random, items: list, out: list, all_pls: list, named: list, p_orgs: list,
                     lead_final: str, filing: dt.date, city: str, st: str, roster: bool) -> int:
    """t01v5 plaintiff party blocks. ``roster``: every plaintiff (named + the rest of a long caption) repeats the whole
    attorney roster of the organisations representing them (3-10 attorneys, two-column PACER text, U+2212 phone
    numbers); the lead organisation represents everyone, a later organisation only the later-listed plaintiffs, so a
    reader of the first few blocks misses it. Else (pro se): each plaintiff block says PRO SE, counsel appear later in
    notices of appearance. Returns the number of blocks."""
    nb = 0
    if not roster:
        for p in named:
            blk = f"Plaintiff\n{fmt_person(p, 0)}\nPRO SE"
            nb += 1
            iid = f"PB{nb}"
            items.append((iid, blk))
            ftxt = f"{fmt_person(p, 0)} PRO SE"
            b.fact("plaintiffs", "true", p, filing, ftxt, iid)
            b.fact("counsel_type", "true", "Pro se", filing, ftxt, iid)
            b.fact("parties", "distractor", "name:" + fmt_person(p, 1), filing, ftxt, iid, trap="names")
            out.append(blk)
        return nb
    team = {org: [[r3.choice(FIRST), r3.choice("ABCDEFGHJKLMNPRSTW") if r3.random() < 0.5 else None, r3.choice(LAST)]
                  for _ in range(r3.randint(1, 4 if org == p_orgs[0] else 3))] for org in p_orgs}
    starts = {org: (0 if k == 0 else r3.randint(len(all_pls) // 2, len(all_pls) - 1)) for k, org in enumerate(p_orgs)}
    addr = {org: (f"{r3.randint(10, 999)} {r3.choice(LAST)} Street", f"{city}, {r3.choice(['AL', 'GA', 'TX', 'OH', 'NY', 'CA', 'IL', 'WA'])} "
                  f"{r3.randint(10000, 99999)}") for org in p_orgs}
    for i, p in enumerate(all_pls):
        reps = [(org, a) for org in p_orgs if i >= starts[org] for a in team[org]]
        if len(reps) < 3:
            for _ in range(3 - len(reps)):
                a = [r3.choice(FIRST), None, r3.choice(LAST)]
                team[p_orgs[0]].append(a)
                reps.append((p_orgs[0], a))
        lines = ["Plaintiff"]
        for j, (org, a) in enumerate(reps):
            nm = fmt_person(a, 0)
            if j == 0:
                lines.append(f"{fmt_person(p, 0):<38}represented by {nm}")
            else:
                lines.append(f"{'':<38}{nm}")
            lines.append(org)
            if j > 0 and r3.random() < 0.4:
                lines.append("(See above for address)")
            else:
                lines += [addr[org][0], addr[org][1],
                          f"({r3.randint(200, 999)}) {r3.randint(200, 999)}\u2212{r3.randint(1000, 9999)}",
                          f"Email: {a[0][0].lower()}{a[2].lower()}@{re.sub(r'[^a-z]', '', org.lower())[:12]}.org"]
            if j == 0:
                lines.append("LEAD ATTORNEY")
            lines.append("ATTORNEY TO BE NOTICED")
            lines.append("")
        blk = "\n".join(lines).rstrip()
        nb += 1
        iid = f"PB{nb}"
        items.append((iid, blk))
        ftxt0 = f"{fmt_person(p, 0)} represented by {fmt_person(reps[0][1], 0)} {reps[0][0]}"
        b.fact("plaintiffs", "true", p, filing, ftxt0, iid)
        b.fact("parties", "distractor", "name:" + fmt_person(p, 1), filing, ftxt0, iid, trap="names")
        if i == 0:
            b.fact("lead_counsel", "true" if lead_final == p_orgs[0] else "stale", p_orgs[0], filing, ftxt0, iid)
        for org in dict.fromkeys(o for o, _ in reps):
            a = next(x for o, x in reps if o == org)
            ftxt = f"{fmt_person(a, 0)} {org}"
            b.fact("counsel", "true", org, filing, ftxt, iid)
            b.fact("counsel_type", "true", org, filing, ftxt, iid)
        out.append(blk)
    return nb


def gen_docket(rng: random.Random, P: dict) -> Generated:
    b = Builder(rng)
    t01 = bool(P.get("t01", True))
    r2 = random.Random(f"elb-t01:{rng.getstate()[1][:6]}")
    v5 = t01 and bool(P.get("t01v5", True))
    r3 = random.Random(f"elb-t01v5:{rng.getstate()[1][:6]}")

    def X(*fx):
        return list(fx) if t01 else []

    def X5(*fx):
        return list(fx) if v5 else []
    st, cir = rng.choice(STATES)
    city, county = rng.choice(CITIES), rng.choice(COUNTIES)
    filing = dt.date(rng.randint(1996, 2019), rng.randint(1, 12), rng.randint(1, 28))
    div = rng.randint(1, 4)
    caseno = [div, filing.year % 100, rng.randint(100, 9999)]
    jud_ini = "".join(rng.choice("ABCDEFGHJKLMNPRSTW") for _ in range(3))
    upd = float(P.get("update_rate", 0.7))
    dis = float(P.get("distractor_rate", 0.75))
    target = sample_chars(rng, P)
    extra = int(min(4.8, target / 250_000))

    judge1, mag = b.person(), b.person()
    reassigned = rng.random() < upd * 0.7
    judge2 = b.person() if reassigned else None
    mag_case = t01 and r2.random() < float(P.get("p_magistrate", 0.08))
    no_sheet = t01 and r2.random() < float(P.get("p_no_sheet", 0.10))
    transfer = t01 and not no_sheet and r2.random() < float(P.get("p_transfer", 0.12))
    if mag_case:
        reassigned, judge2 = False, None
    roster = v5 and not no_sheet and r3.random() < float(P.get("p_roster", 0.45))
    pro_se = v5 and not no_sheet and not roster and r3.random() < float(P.get("p_pro_se", 0.10))
    approval_missing = v5 and r3.random() < float(P.get("p_approval_missing", 0.12))
    p_no_relief_end = float(P.get("p_ended_no_relief", 0.4))
    plaintiffs = [b.person() for _ in range(rng.randint(1, 4))]
    fac = rng.choice(FACILITY_T).format(c=county, city=city, st=st, l=rng.choice(["Ellis", "Hughes", "Ferguson"]))
    other_fac = rng.choice(FACILITY_T).format(c=rng.choice(COUNTIES), city=rng.choice(CITIES), st=st, l="Wynne")
    if other_fac == fac:
        other_fac = f"{rng.choice(CITIES)} Regional Center"
    defendants = list(dict.fromkeys(
        rng.choice(DEF_T).format(c=county, city=city, st=st, p=fmt_person(b.person(), 1))
        for _ in range(rng.randint(2, 4))))
    pub = [rng.choice(PUBLIC_ORGS).format(st=st, city=city, w=rng.choice(WORDS)) for _ in range(rng.randint(0, 2))]
    priv = [rng.choice(PRIVATE_ORGS).format(a=rng.choice(LAST), b=rng.choice(LAST), c=rng.choice(LAST),
                                            p=fmt_person(b.person(), 1)) for _ in range(rng.randint(0 if pub else 1, 1))]
    p_orgs = list(dict.fromkeys(pub + priv))
    d_orgs = list(dict.fromkeys(rng.choice(DEFENSE_ORGS).format(st=st, c=county, city=city, a=rng.choice(LAST),
                                                                b=rng.choice(LAST)) for _ in range(rng.randint(1, 2))))
    conduct = rng.choice(CONDUCT)
    claim_keys = rng.sample([k for k, v in STATUTES.items() if v[1] == "claim"], rng.randint(1, 3))
    jur_keys = rng.sample([k for k, v in STATUTES.items() if v[1] == "jurisdiction"], 2)
    remedy_keys = rng.sample(list(REMEDIES), rng.randint(2, 4))
    class_alleged = rng.random() < 0.5
    cert_granted = class_alleged and rng.random() < 0.7
    has_settlement = rng.random() < 0.5
    has_decree = rng.random() < 0.5
    has_monitor = (has_settlement or has_decree) and rng.random() < 0.55
    has_trial = (not has_settlement) and rng.random() < 0.35
    has_appeal = rng.random() < 0.4
    has_related = rng.random() < 0.35
    has_consol = rng.random() < 0.3
    has_disputes = (has_settlement or has_decree) and rng.random() < 0.45
    has_pi = rng.random() < 0.5
    n_opinions = min(7, rng.choice([0, 1, 1, 2, 2, 3]) + extra)

    entries: list[tuple[dt.date, int, str, list]] = []
    seq = [0]

    def entry(d: dt.date, text: str, facts: list | None = None):
        seq[0] += 1
        entries.append((d, seq[0], text, facts or []))

    end = filing + dt.timedelta(days=rng.randint(500, 3000))
    span = (end - filing).days

    def when(lo: float, hi: float) -> dt.date:
        return filing + dt.timedelta(days=int(span * rng.uniform(lo, hi)))

    def judge_at(d: dt.date) -> list:
        if transfer and d < tdate:
            return judge0
        return judge2 if (reassigned and d >= reassign_date) else judge1

    def jtitle(d: dt.date) -> str:
        return "Magistrate Judge" if (mag_case and judge_at(d) is judge1) else "Judge"

    def when2(lo: float, hi: float) -> dt.date:
        return filing + dt.timedelta(days=int(span * r2.uniform(lo, hi)))

    def when3(lo: float, hi: float) -> dt.date:
        return filing + dt.timedelta(days=int(span * r3.uniform(lo, hi)))

    reassign_date = when(0.15, 0.6) if reassigned else end
    tdate = judge0 = None
    if transfer:
        tdate = when2(0.04, 0.12) + dt.timedelta(days=1)
        last0 = r2.choice([x for x in LAST if x not in b.used])
        b.used.add(last0)
        judge0 = [r2.choice(FIRST), None, last0]
        st0 = r2.choice([s for s, _ in STATES if s != st])
        caseno0 = [r2.randint(1, 4), filing.year % 100, r2.randint(100, 9999)]
        ini0 = "".join(r2.choice("ABCDEFGHJKLMNPRSTW") for _ in range(3))
    rul: list = []
    fil: list = []

    def RUL(d: dt.date, stem: str, phrase: str, kind: str = "true") -> tuple:
        if kind == "true":
            rul.append([iso(d), stem])
        return ("rulings", kind, [iso(d), stem, phrase])

    fil_last: list = [None]

    def FIL(d: dt.date, stem: str, phrase: str, kind: str = "true", trap: str = "") -> tuple:
        v = [iso(d), stem, phrase]
        if kind == "true":
            fil.append([iso(d), stem])
            fil_last[0] = (fil[-1], v)
        return ("filings", kind, v, trap) if trap else ("filings", kind, v)

    def OUT(ref, d: dt.date, stem: str, phrase: str):
        """t01v5: the court's ruling on a filed motion becomes part of that filing's gold ("On <date>, X moved to
        dismiss, which the court denied on <date>"); the filing fact's value carries the outcome for the oracle."""
        if v5 and ref is not None:
            ref[0].extend([iso(d), stem])
            ref[1].extend([iso(d), stem, phrase])
    caption = f"{plaintiffs[0][2]}, et al. v. {defendants[0]}, et al." if len(plaintiffs) > 1 else \
        f"{plaintiffs[0][2]} v. {defendants[0]}"
    short_cap = f"{plaintiffs[0][2]} v. {defendants[0].split(' of ')[-1]}"
    sign = lambda d: f"Signed by {jtitle(d)} {fmt_person(judge_at(d), rng.randrange(2))} on {fmt_date(d, 0)}"  # noqa: E731
    sign2 = lambda d: f"Signed by {jtitle(d)} {fmt_person(judge_at(d), r2.randrange(2))} on {fmt_date(d, 0)}"  # noqa: E731

    pls_s = ", ".join(fmt_person(p, 1) for p in plaintiffs)
    entry(filing, rng.choice([f"COMPLAINT against {', '.join(defendants)} filed by {pls_s}.",
                              f"{'CLASS ACTION ' if class_alleged else ''}COMPLAINT with Jury Demand against "
                              f"{defendants[0]} et al. filed by {pls_s}. Filing fee $350, receipt number "
                              f"{rng.randint(10**6, 10**7)}."]),
          [("filing_date", "true", iso(filing))] + X(FIL(filing, "complaint", "Plaintiffs filed the complaint")))
    if rng.random() < dis:
        d = when(0.05, 0.25)
        entry(d, f"AMENDED COMPLAINT against {', '.join(defendants)} filed by {pls_s}, adding claims under "
                 f"{STATUTES[claim_keys[0]][0][0]}.",
              [("filing_date", "distractor", iso(d))] + X(FIL(d, "amend", "Plaintiffs filed an amended complaint")))
    if transfer:
        entry(tdate - dt.timedelta(days=1), f"ORDER granting Motion to Transfer Venue to the District of {st}. "
                                            f"{sign2(tdate - dt.timedelta(days=1))}.",
              [RUL(tdate - dt.timedelta(days=1), "transfer", f"transferred the case to the District of {st}")])
        entry(tdate, f"CASE TRANSFERRED IN from the District of {st0}; Case Number {fmt_caseno(caseno0, 1)}. Original "
                     f"file, certified copy of transfer order and docket sheet received.",
              [("filing_date", "distractor", iso(tdate), "transfer"), ("related", "true", caseno0)])
    has_mtd = t01 and r2.random() < 0.45
    if has_mtd:
        d = when2(0.03, 0.12)
        entry(d, f"MOTION to Dismiss for Failure to State a Claim by {defendants[0]}.",
              [FIL(d, "dismiss", "Defendants moved to dismiss the complaint")])
        mtd_ref = fil_last[0]
        d2 = when2(0.13, 0.3)
        if r2.random() < 0.5:
            dh = d2 - dt.timedelta(days=r2.randint(5, 30))
            entry(dh, f"MINUTE ENTRY for Motion Hearing held on {fmt_date(dh, 0)} before {jtitle(dh)} "
                      f"{fmt_person(judge_at(dh), 1)} re Motion to Dismiss. Argument heard; matter taken under "
                      f"advisement.", [("trial", "distractor", iso(dh), "hearing")])
        if r2.random() < 0.5:
            entry(d2, f"ORDER granting in part and denying in part Motion to Dismiss; Count II is dismissed. {sign2(d2)}.",
                  [RUL(d2, "grant", "granted in part and denied in part the motion to dismiss")])
            OUT(mtd_ref, d2, "grant", "granted in part")
        else:
            entry(d2, f"ORDER denying Motion to Dismiss. {sign2(d2)}.", [RUL(d2, "deny", "denied the motion to dismiss")])
            OUT(mtd_ref, d2, "deny", "denied")
    if reassigned:
        entry(reassign_date, rng.choice([
            f"ORDER OF REASSIGNMENT: Case reassigned to Judge {fmt_person(judge2, 0)} for all further proceedings. "
            f"Judge {fmt_person(judge1, 1)} no longer assigned to case.",
            f"NOTICE of reassignment: pursuant to General Order {rng.randint(10, 99)}-{rng.randint(1, 30)}, this case "
            f"is reassigned from Judge {fmt_person(judge1, 0)} to Judge {fmt_person(judge2, 1)}."]),
            [("judge", "true", judge2), ("judge", "stale", judge1)])
    rel_nos = []
    if has_related:
        for _ in range(rng.randint(1, 2 + extra // 2)):
            rc = [rng.randint(1, 4), rng.randint(filing.year - 4, filing.year) % 100, rng.randint(100, 9999)]
            rel_nos.append(rc)
            d = when(0.0, 0.2)
            entry(d, rng.choice([
                f"NOTICE OF RELATED CASE by Plaintiffs: this action is related to {fmt_caseno(rc, rng.randrange(3))}, "
                f"{rng.choice(LAST)} v. {defendants[-1]}.",
                f"ORDER: The Court finds that case {fmt_caseno(rc, rng.randrange(3))} is related to this action "
                f"and shall be assigned to the same judge. {sign(d)}."]), [("related", "true", rc)])
    con_nos = []
    if has_consol:
        cc = [div, (filing.year + rng.randint(0, 1)) % 100, rng.randint(100, 9999)]
        con_nos.append(cc)
        d = when(0.05, 0.4)
        entry(d, rng.choice([
            f"ORDER consolidating Case No. {fmt_caseno(cc, rng.randrange(3))} with this action for all purposes; all "
            f"further filings shall be made in the lead case. {sign(d)}.",
            f"MEMORANDUM ORDER granting motion to consolidate: {fmt_caseno(cc, rng.randrange(3))} is consolidated "
            f"into the above-captioned action. {sign(d)}."]), [("consolidated", "true", cc)])
    if rng.random() < dis * 0.6:
        xc = [div, filing.year % 100, rng.randint(100, 9999)]
        d = when(0.05, 0.5)
        entry(d, f"ORDER denying Motion to Consolidate this action with Case No. {fmt_caseno(xc, 0)}, which "
                 f"involves different parties. {sign(d)}.", [("consolidated", "distractor", xc)])

    class_date = None
    if class_alleged:
        dm = when(0.1, 0.25)
        fil_last[0] = None
        entry(dm, f"MOTION to Certify Class by {pls_s}.", X(FIL(dm, "class", "Plaintiffs moved for class certification")))
        class_ref = fil_last[0]
        class_date = when(0.26, 0.45)
        cert_doc = rng.random() < 0.5
        OUT(class_ref, class_date, "grant" if cert_granted else "deny", "granted" if cert_granted else "denied")
        if cert_doc:
            entry(class_date, f"MEMORANDUM OPINION AND ORDER re Motion to Certify Class. {sign(class_date)}.")
        elif cert_granted:
            entry(class_date, rng.choice([
                f"ORDER granting Motion to Certify Class; the Court certifies a class of all persons subjected to "
                f"{conduct} at {fac}. {sign(class_date)}.",
                f"MEMORANDUM OPINION AND ORDER: class certification is GRANTED under Rule 23(b)(2). "
                f"{sign(class_date)}."]),
                [("class_action", "true", "Class Action")] + X(RUL(class_date, "grant", "certified the class")))
        else:
            entry(class_date, f"ORDER denying Motion to Certify Class; the named plaintiffs may proceed on their "
                              f"individual claims. {sign(class_date)}.",
                  [("class_action", "distractor", "Individual", "cert_denied") if v5 else ("class_action", "true", "Individual")]
                  + X(RUL(class_date, "deny", "denied class certification")))

    decree_dates: list = []
    extra_terms: list = []
    pi_granted = False
    if has_pi:
        d = when(0.02, 0.15)
        fil_last[0] = None
        entry(d, f"MOTION for Preliminary Injunction by {pls_s}.",
              [("decree_dates", "distractor", iso(d))] + X(FIL(d, "injunct", "Plaintiffs moved for a preliminary injunction")))
        pi_ref = fil_last[0]
        d2 = when(0.16, 0.3)
        pi_granted = rng.random() < 0.6
        if t01 and r2.random() < 0.7:
            dh = d2 - dt.timedelta(days=r2.randint(3, 20))
            entry(dh, f"MINUTE ENTRY for Motion Hearing held on {fmt_date(dh, 0)} before {jtitle(dh)} "
                      f"{fmt_person(judge_at(dh), 1)} re Motion for Preliminary Injunction. Witnesses sworn; argument "
                      f"heard.", [("trial", "distractor", iso(dh), "hearing")])
        if pi_granted:
            txt = rng.choice([
                f"ORDER granting in part Motion for Preliminary Injunction; Defendants are enjoined from continuing "
                f"the {conduct}. {sign(d2)}.",
                f"PRELIMINARY INJUNCTION entered against Defendants regarding {conduct}. {sign(d2)}."])
            fx = [("decree_dates", "true", iso(d2))] + X(RUL(d2, "grant", "granted a preliminary injunction"),
                                                          ("decree_terms", "true", f"conduct:{conduct}")) \
                + X5(("court_enforced", "true", "Yes"))
            OUT(pi_ref, d2, "grant", "granted")
            if t01 and r2.random() < 0.7:
                txt = txt[:-1] + " The injunction shall remain in effect until further order of the Court."
                fx.append(("decree_duration", "distractor", "until further order", "until_further"))
            entry(d2, txt, fx)
            decree_dates.append(iso(d2))
        else:
            entry(d2, f"ORDER denying Motion for Preliminary Injunction. {sign(d2)}.",
                  X(RUL(d2, "deny", "denied the motion for a preliminary injunction"),
                    ("decree_terms", "distractor", "deny", "denial")))
            OUT(pi_ref, d2, "deny", "denied")

    if has_pi or has_decree:
        for k in range(extra):
            d = when(0.3 + 0.6 * k / max(1, extra), 0.3 + 0.6 * (k + 0.8) / max(1, extra))
            t1_ = rng.choice(TOPICS)
            opts = [f"AMENDED PRELIMINARY INJUNCTION entered regarding {t1_}. {sign(d)}.",
                    f"PARTIAL FINAL JUDGMENT entered on Count {['II', 'III', 'IV'][k % 3]}. {sign(d)}."]
            t3_ = rng.choice(TOPICS)
            opts.append(f"SUPPLEMENTAL REMEDIAL ORDER entered as an injunction concerning {t3_}. {sign(d)}.")
            ix = rng.randrange(3)
            fx = [("decree_dates", "true", iso(d))]
            if t01:
                fx.append(RUL(d, "enter", ["entered an amended preliminary injunction", "entered partial final judgment",
                                           "entered a supplemental remedial order"][ix]))
                if ix != 1:
                    fx.append(("decree_terms", "true", f"remedial:{t1_ if ix == 0 else t3_}"))
                    extra_terms.append(f"remedial:{t1_ if ix == 0 else t3_}")
            entry(d, opts[ix], fx)
            decree_dates.append(iso(d))
            if rng.random() < dis:
                dp = d - dt.timedelta(days=rng.randint(5, 30))
                entry(dp, f"[PROPOSED] ORDER on remedies lodged by {rng.choice(['Plaintiffs', 'Defendants'])}.",
                      [("decree_dates", "distractor", iso(dp))])

    settle_date = amount = amount_final = None
    enforced = None
    if has_settlement:
        if rng.random() < dis:
            d = when(0.4, 0.55)
            entry(d, f"MINUTE ENTRY for Settlement Conference held before Magistrate Judge {fmt_person(mag, 1)} on "
                     f"{fmt_date(d, 0)}.", [("settlement_date", "distractor", iso(d)), ("judge", "distractor", mag)])
        d = when(0.56, 0.62)
        entry(d, rng.choice(["NOTICE of Settlement in Principle by All Parties.",
                             "JOINT MOTION for Preliminary Approval of Settlement Agreement by All Parties."]),
              [("settlement_date", "distractor", iso(d))])
        settle_date = when(0.63, 0.7)
        signed = settle_date - dt.timedelta(days=r3.randint(14, 60)) if v5 else None
        if v5:
            dn_ = signed + dt.timedelta(days=r3.randint(1, 6))
            entry(dn_, f"NOTICE of Filing of Executed Settlement Agreement (executed {fmt_date(signed, 0)}) and JOINT "
                       f"MOTION for Final Approval by All Parties.",
                  [("settlement_date", "true", iso(signed)) if approval_missing else
                   ("settlement_date", "distractor", iso(signed), "signed")]
                  + [FIL(dn_, "approv", "the parties moved for final approval of the settlement")])
            appr_ref = fil_last[0]
        enforced = "Yes" if rng.random() < 0.6 else "No"
        if v5:
            enforced = "Yes" if r3.random() < 0.3 else "No"
        amend_date = None
        if rng.random() < 0.75:
            amount = rng.randint(4, 400) * 25_000
            amount_final = amount + rng.randint(2, 40) * 25_000 if rng.random() < upd else amount
        fund = (f", which establishes a Settlement Fund of {fmt_amount(amount, rng.randrange(3))}"
                if amount is not None and rng.random() < 0.3 else "")
        appr_txt = rng.choice([
            f"ORDER granting final approval of the Settlement Agreement{fund}; the Agreement is approved as fair, "
            f"reasonable and adequate. {sign(settle_date)}.",
            f"FINAL ORDER AND JUDGMENT approving the parties' Settlement Agreement{fund}. {sign(settle_date)}."])
        if not approval_missing:
            entry(settle_date, appr_txt,
                  [("settlement_date", "true", iso(settle_date))]
                  + ([("settlement_amount", "true" if amount_final == amount else "stale", amount)] if fund else [])
                  + X(RUL(settle_date, "approv", "approved the settlement agreement")))
            if v5:
                OUT(appr_ref, settle_date, "approv", "approved")
        if amount is not None and amount_final != amount:
            amend_date = when(0.72, 0.8)
            entry(amend_date, f"ORDER approving First Amendment to Settlement Agreement. {sign(amend_date)}.")

    duration = duration_final = None
    decree_date = None
    if has_decree:
        if rng.random() < dis:
            d = when(0.6, 0.66)
            entry(d, "[PROPOSED] CONSENT DECREE lodged by All Parties.", [("decree_dates", "distractor", iso(d))])
        decree_date = when(0.67, 0.72)
        duration = rng.choice([24, 36, 48, 60])
        duration_final = duration + rng.choice([12, 24]) if rng.random() < upd else duration
        term_txt = (f"; the Decree remains in effect for {fmt_duration(duration, rng.randrange(3))}"
                    if rng.random() < 0.3 else "")
        entry(decree_date, rng.choice([f"CONSENT DECREE entered{term_txt}. {sign(decree_date)}.",
                                       f"PERMANENT INJUNCTION AND CONSENT DECREE entered against Defendants{term_txt}. "
                                       f"{sign(decree_date)}.", f"ORDER entering Consent Decree{term_txt}. {sign(decree_date)}."]),
              [("decree_dates", "true", iso(decree_date))]
              + ([("decree_duration", "true" if duration_final == duration else "stale", duration)] if term_txt else [])
              + X(RUL(decree_date, "enter", "entered the consent decree"))
              + X5(("court_enforced", "true", "Yes"),
                   ("settlement_date", "distractor" if has_settlement else "true", iso(decree_date))))
        decree_dates.append(iso(decree_date))
        ext_doc = None
        if duration_final != duration and rng.random() < 0.7:
            ext_doc = when(0.85, 0.95)
            entry(ext_doc, f"ORDER granting Plaintiffs' motion to extend the Consent Decree. {sign(ext_doc)}.",
                  X(RUL(ext_doc, "extend", "extended the consent decree")))
        elif duration_final != duration:
            ed = when(0.85, 0.95)
            entry(ed, rng.choice([
                f"ORDER extending Consent Decree: the Consent Decree shall remain in effect for a total of "
                f"{fmt_duration(duration_final, rng.randrange(3))} from its effective date. {sign(ed)}.",
                f"ORDER granting Plaintiffs' motion to extend the decree; the term of the Consent Decree is now "
                f"{fmt_duration(duration_final, rng.randrange(3))}. {sign(ed)}."]),
                [("decree_duration", "true", duration_final)] + X(RUL(ed, "extend", "extended the consent decree")))

    monitor_final = None
    report_dates: list = []
    monitors: list = []
    if has_monitor:
        if rng.random() < dis:
            nominee = b.person()
            d = when(0.66, 0.73)
            entry(d, f"NOTICE by Plaintiffs nominating {fmt_person(nominee, 0, 'Dr.')} to serve as Monitor.",
                  [("monitor", "distractor", nominee)])
        m1 = b.person()
        md = when(0.74, 0.78)
        monitors.append((md, m1))
        if rng.random() < upd * 0.6:
            m2 = b.person()
            md2 = when(0.86, 0.9)
            monitors.append((md2, m2))
        monitor_final = monitors[-1][1]
        appoint_doc = rng.random() < 0.5
        if appoint_doc:
            entry(md, f"ORDER regarding compliance oversight under the decree. {sign(md)}.")
        else:
            entry(md, rng.choice([
                f"ORDER APPOINTING MONITOR: The Court appoints {fmt_person(m1, 0, 'Dr.')} as the independent Monitor. "
                f"{sign(md)}.",
                f"ORDER: {fmt_person(m1, 1)} shall serve as Compliance Monitor under the decree. {sign(md)}."]),
                [("monitor", "true" if len(monitors) == 1 else "stale", m1)]
                + (X(("decree_terms", "true", "monitor")) if has_decree else []))
        succ_doc = len(monitors) > 1 and rng.random() < 0.7
        if len(monitors) > 1 and not succ_doc:
            entry(monitors[1][0], f"ORDER: Upon the resignation of Monitor {fmt_person(m1, 1)}, the Court appoints "
                                  f"{fmt_person(monitors[1][1], 0)} as successor Monitor. {sign(monitors[1][0])}.",
                  [("monitor", "true", monitors[1][1]), ("monitor", "stale", m1)])
        elif len(monitors) > 1:
            entry(monitors[1][0], f"ORDER regarding the resignation of the Monitor. {sign(monitors[1][0])}.")
        n_rep = rng.randint(1, 3 + extra)
        for k in range(n_rep):
            rd = when(0.79 + 0.19 * k / n_rep, 0.79 + 0.19 * (k + 0.8) / n_rep)
            cur = [m for (dd, m) in monitors if dd <= rd][-1]
            report_dates.append(iso(rd))
            if rng.random() < 0.5:
                entry(rd, rng.choice([f"MONITOR'S {ORDINAL[k + 1].upper()} REPORT by Monitor {fmt_person(cur, rng.randrange(2))}.",
                                      f"REPORT of the Independent Monitor ({ORDINAL[k + 1].lower()} reporting period) "
                                      f"filed by {fmt_person(cur, rng.randrange(2))}."]),
                      [("monitor_reports", "true", iso(rd)), ("monitor", "true" if cur == monitor_final else "stale", cur)])
            else:
                entry(rd, rng.choice([f"MONITOR'S {ORDINAL[k + 1].upper()} REPORT filed.",
                                      f"Semi-annual compliance report of the Monitor ({ORDINAL[k + 1].lower()}) filed."]),
                      [("monitor_reports", "true", iso(rd))])
        if rng.random() < dis:
            d = when(0.8, 0.97)
            entry(d, f"STATUS REPORT by {defendants[0]} regarding compliance with the Consent Decree.",
                  [("monitor_reports", "distractor", iso(d))])
    elif rng.random() < dis * 0.5 and (has_decree or has_settlement):
        nominee = b.person()
        d = when(0.6, 0.8)
        entry(d, f"ORDER denying Plaintiffs' motion to appoint {fmt_person(nominee, 0)} as Monitor. {sign(d)}.",
              [("monitor", "distractor", nominee)])

    deadline_final = None
    deadline_docs: list = []
    if has_settlement or has_decree:
        base_d = decree_date or settle_date
        dl = base_d + dt.timedelta(days=rng.randint(120, 400))
        deadline_final = dl
        txt = (f"ORDER setting compliance deadline: Defendants shall complete the remedial measures required by the "
               f"{'Consent Decree' if has_decree else 'Settlement Agreement'} by {fmt_date(dl, rng.randrange(4))}. "
               f"{sign(base_d + dt.timedelta(days=3))}.")
        n_ext = rng.randint(1, 2 + extra) if rng.random() < upd else 0
        entry(base_d + dt.timedelta(days=3), txt, [("deadline", "true" if n_ext == 0 else "stale", iso(dl))]
              + (X(("decree_terms", "true", "deadline")) if has_decree else []))
        prev = dl
        for k in range(n_ext):
            od = prev - dt.timedelta(days=rng.randint(5, 40))
            new = prev + dt.timedelta(days=rng.randint(60, 240))
            last = k == n_ext - 1
            if rng.random() < 0.3:
                entry(od, f"ORDER granting Defendants' motion to extend the compliance deadline from "
                          f"{fmt_date(prev, 0)} to {fmt_date(new, rng.randrange(4))}. {sign(od)}.",
                      [("deadline", "true" if last else "stale", iso(new))])
            else:
                entry(od, f"ORDER granting Defendants' motion for an extension of time to comply. {sign(od)}.")
                deadline_docs.append((od, prev, new, last))
            prev = new
        deadline_final = prev
    lead_final = p_orgs[0]
    subst = None
    if rng.random() < upd * 0.5:
        new_org = rng.choice(PUBLIC_ORGS + PRIVATE_ORGS).format(st=st, city=city, w=rng.choice(WORDS), a=rng.choice(LAST),
                                                                 b=rng.choice(LAST), c=rng.choice(LAST),
                                                                 p=fmt_person(b.person(), 1))
        if new_org not in p_orgs:
            sd = when(0.3, 0.8)
            subst = (sd, new_org)
            lead_final = new_org
            if rng.random() < 0.3:
                entry(sd, f"NOTICE of Substitution of Counsel: {new_org} substitutes for {p_orgs[0]} as lead counsel "
                          f"for Plaintiffs.", [("lead_counsel", "true", new_org), ("lead_counsel", "stale", p_orgs[0])])
            else:
                entry(sd, "ORDER granting motion to substitute counsel for Plaintiffs.")

    dispute_dates: list = []
    if has_disputes:
        for _ in range(rng.randint(1, 2 + extra)):
            d = when(0.8, 0.98)
            dispute_dates.append(iso(d))
            entry(d, rng.choice([
                f"MOTION to Enforce Settlement Agreement by {pls_s}.",
                f"MOTION for Order to Show Cause why Defendants should not be held in contempt of the "
                f"{'Consent Decree' if has_decree else 'Settlement Agreement'} by Plaintiffs."]),
                [("disputes", "true", iso(d))] + X(FIL(d, "enforce", "Plaintiffs moved to enforce the "
                                                    + ("consent decree" if has_decree else "settlement agreement"))))
    if rng.random() < dis * 0.5:
        d = when(0.2, 0.6)
        entry(d, f"MOTION to Enforce Subpoena directed to {rng.choice(LAST)} Consulting by {defendants[0]}.",
              [("disputes", "distractor", iso(d))])

    trial_date = None
    trial_winner = None
    if has_trial:
        set_d = when(0.5, 0.6)
        first_setting = when(0.62, 0.7)
        entry(set_d, f"ORDER setting Bench Trial for {fmt_date(first_setting, 1)}. {sign(set_d)}.",
              [("trial", "distractor", iso(first_setting))])
        trial_date = when(0.75, 0.8)
        if rng.random() < dis:
            vd = when(0.61, 0.62)
            entry(vd, f"ORDER vacating trial setting of {fmt_date(first_setting, 0)}; trial reset at the request of "
                      f"Defendants. {sign(vd)}.", [("trial", "distractor", iso(first_setting))])
        entry(trial_date, rng.choice([
            f"MINUTE ENTRY for proceedings held before {jtitle(trial_date)} {fmt_person(judge_at(trial_date), 1)}: Bench Trial (Day 1) "
            f"held on {fmt_date(trial_date, 0)}.",
            f"JURY TRIAL begun on {fmt_date(trial_date, 1)} before {jtitle(trial_date)} {fmt_person(judge_at(trial_date), 0)}."]),
            [("trial", "true", iso(trial_date)), ("judge", "true" if judge_at(trial_date) == (judge2 or judge1)
                                                  else "stale", judge_at(trial_date))])
        jd = when(0.82, 0.86)
        trial_winner = rng.choice(['Plaintiffs', 'Defendants'])
        entry(jd, f"FINAL JUDGMENT entered in favor of {trial_winner}. {sign(jd)}.",
              [("decree_dates", "true", iso(jd))] + X(RUL(jd, "judgment", "entered final judgment after trial"))
              + X5(("court_enforced", "true", "Yes" if trial_winner == "Plaintiffs" else "No")))
        decree_dates.append(iso(jd))

    appeal = usca = None
    usca_in_hdr = False
    if has_appeal:
        d = when(0.3, 0.88)
        party = rng.choice(["Defendants", "Plaintiffs"])
        outcome = rng.choice(["AFFIRMED", "REVERSED", "VACATED and REMANDED", "DISMISSED"])
        appeal = {"circuit": cir, "outcome": outcome.split()[0].lower()}
        fil_last[0] = None
        entry(d, f"NOTICE OF APPEAL to the U.S. Court of Appeals for the {ORDINAL[cir]} Circuit by {party}.",
              [("appeal", "true", {"circuit": cir})] + X(FIL(d, "appeal", f"{party} filed a notice of appeal")))
        app_ref = fil_last[0]
        usca = [d.year % 100, rng.randint(10000, 99999)]
        usca_in_hdr = t01 and not no_sheet and r2.random() < 0.75
        entry(d + dt.timedelta(days=rng.randint(2, 9)), f"USCA Case Number {usca[0]:02d}-{usca[1]} for Notice of "
                                                        f"Appeal filed by {party}.",
              X(("related", "true" if usca_in_hdr else "distractor", ["usca", usca[0], usca[1]])))
        md_ = d + dt.timedelta(days=rng.randint(200, 500))
        _ost = {"AFFIRMED": "affirm", "REVERSED": "revers", "VACATED": "vacat", "DISMISSED": "dismiss"}[outcome.split()[0]]
        OUT(app_ref, md_, _ost, outcome.split()[0].lower())
        entry(md_, f"MANDATE of USCA (No. {usca[0]:02d}-{usca[1]}): the judgment of the district court is {outcome}.",
              [("appeal", "true", {"outcome": appeal["outcome"]})]
              + X(RUL(md_, {"AFFIRMED": "affirm", "REVERSED": "revers", "VACATED": "vacat", "DISMISSED": "dismiss"}[outcome.split()[0]],
                    f"the court of appeals {outcome.lower()} the judgment")))

    pro_se_apps: list = []
    if pro_se:
        for org in p_orgs:
            atty = [r3.choice(FIRST), None, r3.choice(LAST)]
            da = when3(0.05, 0.35)
            pro_se_apps.append((da, atty, org))
            entry(da, f"NOTICE of Appearance by {fmt_person(atty, 0)} of {org} on behalf of Plaintiff "
                      f"{fmt_person(plaintiffs[0], 0)}.",
                  [("counsel", "true", org), ("counsel_type", "true", org)]
                  + ([("lead_counsel", "true" if lead_final == org else "stale", org)] if org == p_orgs[0] else []))
    ce_gold = None
    pending = False
    if v5:
        approved = has_settlement and not approval_missing
        if has_decree or (approved and enforced == "Yes") or (not has_settlement and (pi_granted or trial_winner == "Plaintiffs")):
            ce_gold = "Yes"
        elif approved or trial_winner == "Defendants":
            ce_gold = "No"
        elif has_settlement:
            ce_gold, pending = "N/A", True
        elif r3.random() < p_no_relief_end:
            ce_gold = "No"
            dd = when3(0.86, 0.97)
            sg = f"Signed by {jtitle(dd)} {fmt_person(judge_at(dd), r3.randrange(2))} on {fmt_date(dd, 0)}"
            if r3.random() < 0.5:
                entry(dd, f"ORDER granting Stipulation of Voluntary Dismissal; this action is DISMISSED with prejudice, "
                          f"each party to bear its own fees and costs. {sg}.",
                      [("court_enforced", "true", "No"), RUL(dd, "dismiss", "dismissed the case on the parties' stipulation")])
            else:
                entry(dd, f"ORDER granting Defendants' Motion for Summary Judgment; judgment is entered for Defendants on "
                          f"all claims. {sg}.",
                      [("court_enforced", "true", "No"), RUL(dd, "grant", "granted summary judgment for the defendants")])
        else:
            ce_gold, pending = "N/A", True
    extra_pls: list = []
    if roster:
        extra_pls = [[r3.choice(FIRST), r3.choice("ABCDEFGHJKLMNPRSTW") if r3.random() < 0.6 else None, r3.choice(LAST)]
                     for _ in range(r3.randint(4, 12))]
        if len(p_orgs) == 1:
            o2 = r3.choice(PUBLIC_ORGS).format(st=st, city=city, w=r3.choice(WORDS))
            if o2 not in p_orgs:
                pub.append(o2)
                p_orgs.append(o2)
    all_pls = plaintiffs + extra_pls
    kind_s, kind_p = next(((a, b_) for k_, a, b_ in PLAINTIFF_KIND if k_ in fac), ("resident", "residents"))
    n_pl = len(all_pls)
    group = (f"{NUM_WORDS[n_pl] if n_pl < len(NUM_WORDS) else n_pl} {kind_p if n_pl > 1 else kind_s}")

    n_target = int(rng.randint(*P.get("n_entries", (60, 360))) * max(1.0, target / 400_000))
    fill_t = [
        lambda d: (f"NOTICE of Appearance by {fmt_person(b.extra(), 1)} on behalf of "
                   f"{rng.choice(['Plaintiffs', defendants[0]])}.", X(FIL(d, "appearance", "a notice of appearance was filed", "distractor", "routine" if v5 else ""))),
        lambda d: (f"MOTION for Extension of Time to File Answer by {rng.choice(defendants)}.",
                   X5(FIL(d, "extension", "a motion for an extension of time was filed", "distractor", "routine"))),
        lambda d: (f"ORDER granting Motion for Extension of Time. Signed by Magistrate Judge {fmt_person(mag, 1)} on "
                   f"{fmt_date(d, 0)}.", ([("judge", "distractor", mag)] if rng.random() < 0.3 else [])
                   + X(RUL(d, "grant", "granted an extension of time", "distractor"))),
        lambda d: (f"SCHEDULING ORDER: Discovery due by {fmt_date(d + dt.timedelta(days=120), 0)}. {sign(d)}.",
                   [("judge", "true" if judge_at(d) == (judge2 or judge1) else "stale", judge_at(d))]),
        lambda d: (f"MINUTE ENTRY for proceedings held before Magistrate Judge {fmt_person(mag, 0)}: Status "
                   f"Conference held on {fmt_date(d, 0)}.", []),
        lambda d: (f"TRANSCRIPT of Proceedings held on {fmt_date(d - dt.timedelta(days=rng.randint(3, 30)), 0)}. "
                   f"Court Reporter: {fmt_person(b.extra(), 1)}.", []),
        lambda d: (f"STIPULATED PROTECTIVE ORDER regarding records of {fac}. Signed by Magistrate Judge "
                   f"{fmt_person(mag, 1)} on {fmt_date(d, 0)}.", []),
        lambda d: (f"MOTION to Compel Production of {rng.choice(TOPICS)} records by "
                   f"{rng.choice(['Plaintiffs', 'Defendants'])}.", []),
        lambda d: (f"RESPONSE in Opposition to Motion to Compel filed by {rng.choice(['Plaintiffs', 'Defendants'])}.",
                   X5(FIL(d, "response", "a response in opposition to the motion to compel was filed", "distractor",
                          "routine"))),
        lambda d: (f"JOINT STATUS REPORT by All Parties regarding {rng.choice(TOPICS)}.",
                   X5(FIL(d, "status", "the parties filed a joint status report", "distractor", "routine"))),
        lambda d: (f"DEPOSITION of {fmt_person(b.extra(), 1)} taken on {fmt_date(d, 0)} filed by "
                   f"{rng.choice(['Plaintiffs', 'Defendants'])}.", []),
        lambda d: (f"NOTICE of Filing Exhibit {rng.randint(1, 60)} to Declaration of {fmt_person(b.extra(), 1)}.", []),
        lambda d: (f"ORDER granting Motion for Leave to File Excess Pages. {sign(d)}.",
                   X(RUL(d, "grant", "granted leave to file excess pages", "distractor"))),
        lambda d: (f"CERTIFICATE OF SERVICE by {rng.choice(['Plaintiffs', 'Defendants'])} re discovery responses "
                   f"served on {fmt_date(d, 0)}.", []),
        lambda d: (f"NOTICE of Change of Address by {fmt_person(b.extra(), 1)}.", []),
    ]
    while len(entries) < n_target:
        d = filing + dt.timedelta(days=rng.randint(1, span))
        txt, fx = rng.choice(fill_t)(d)
        entry(d, txt, fx)
    entries.sort(key=lambda e: (e[0], e[1]))

    items: list[tuple[str, str]] = []
    out: list[str] = []
    n_facts0 = len(b.facts)
    term = end
    hdr_lines = [
        f"Case: {fmt_caseno(caseno, 0, jud_ini)}   As of: {fmt_date(end + dt.timedelta(days=40), 0)} "
        f"{rng.randint(1, 12)}:{rng.randint(10, 59)} PM EST   1 of {rng.randint(8, 40)}",
        "U.S. District Court",
        f"District of {st} ({city})",
        f"CIVIL DOCKET FOR CASE #: {fmt_caseno(caseno, 0, jud_ini)}",
    ]
    cap_line = f"{caption} Date Filed: {fmt_date(tdate or filing, 0)}"
    asg_line = f"Assigned to: Judge {fmt_person(judge2 or judge1, 0) if rng.random() < 0.5 and reassigned else fmt_person(judge1, 0)}"
    asg_judge = judge2 if (reassigned and "Judge " + fmt_person(judge2, 0) in asg_line) else judge1
    if mag_case:
        asg_line = f"Assigned to: Magistrate Judge {fmt_person(judge1, 0)}"
    if not pending:
        asg_line += f" Date Terminated: {fmt_date(term, 0)}"
    ref_line = f"Referred to: Magistrate Judge {fmt_person(mag, 0)}"
    cause_key, cause_txt = next(((k, CAUSE_CODES[k]) for k in claim_keys if k in CAUSE_CODES),
                                ("28usc1331", "28:1331 Federal Question: Other Civil Rights"))
    cause_line = (f"Cause: {cause_txt}   Nature of Suit: {rng.choice(['550 Prisoner: Civil Rights', '440 Civil Rights: Other', '446 Americans with Disabilities - Other', '443 Civil Rights: Accommodations'])}"
                  f"   Jurisdiction: Federal Question")
    hdr_related: list = []
    xref_lines: list[str] = []
    if t01:
        if rel_nos:
            xref_lines.append("Related Cases: " + ", ".join(fmt_caseno(rc, 0) for rc in rel_nos))
            hdr_related += [list(rc) for rc in rel_nos]
        other = []
        if transfer:
            other.append(f"District of {st0}, {fmt_caseno(caseno0, 0, ini0)}")
            hdr_related.append(list(caseno0))
        if usca_in_hdr:
            other.append(f"{ORDINAL[cir]} Circuit, {usca[0]:02d}-{usca[1]}")
            hdr_related.append(["usca", usca[0], usca[1]])
        if other:
            xref_lines.append("Case in other court: " + "; ".join(other))
    hdr_text = "\n".join(hdr_lines + [cap_line, asg_line] + ([] if mag_case else [ref_line]) + xref_lines + [cause_line])
    items.append(("HDR", hdr_text))
    if transfer:
        b.fact("filing_date", "distractor", iso(tdate), tdate, cap_line, "HDR", trap="transfer")
    else:
        b.fact("filing_date", "true", iso(filing), filing, cap_line, "HDR")
    if not pending:
        b.fact("filing_date", "distractor", iso(term), term, asg_line, "HDR")
    b.fact("judge", "true" if asg_judge == (judge2 or judge1) else "stale", asg_judge,
           (tdate or filing) if asg_judge == judge1 else reassign_date, asg_line, "HDR")
    if not mag_case:
        b.fact("judge", "distractor", mag, filing, ref_line, "HDR")
    b.fact("statutes", "true" if cause_key in claim_keys else "distractor", cause_key, filing, cause_line, "HDR")
    if t01:
        b.fact("cause", "true" if cause_key in claim_keys else "distractor", f"claim:{cause_key}", filing, cause_line, "HDR")
        for ln in xref_lines:
            for rc in hdr_related:
                txt_rc = f"{rc[1]:02d}-{rc[2]}" if rc[0] == "usca" else fmt_caseno(rc, 0)
                if txt_rc in ln:
                    b.fact("related", "true", rc, filing, ln, "HDR")
    out.append(hdr_text)

    nb = 0
    if roster or pro_se:
        nb = _v5_party_blocks(b, r3, items, out, all_pls, plaintiffs, p_orgs, lead_final, filing, city, st, roster)
    for i, p in enumerate([] if (roster or pro_se) else plaintiffs):
        org = p_orgs[i % len(p_orgs)]
        atty = b.person()
        blk = (f"Plaintiff\n{fmt_person(p, 0)} represented by {fmt_person(atty, 0)}\n{org}\n"
               f"{rng.randint(10, 999)} {rng.choice(LAST)} Street, {city}, {st}\nLEAD ATTORNEY\nATTORNEY TO BE NOTICED")
        nb += 1
        iid = f"PB{nb}"
        items.append((iid, blk))
        ftxt = f"{fmt_person(p, 0)} represented by {fmt_person(atty, 0)} {org}"
        b.fact("plaintiffs", "true", p, filing, ftxt, iid)
        if v5:
            b.fact("parties", "distractor", "name:" + fmt_person(p, 1), filing, ftxt, iid, trap="names")
        b.fact("counsel", "true", org, filing, ftxt, iid)
        if t01:
            b.fact("counsel_type", "true", org, filing, ftxt, iid)
        if i == 0:
            b.fact("lead_counsel", "true" if lead_final == org else "stale", org, filing, ftxt, iid)
        b.fact("plaintiffs", "distractor", atty, filing, ftxt, iid)
        out.append(blk)
    for org in ([] if (roster or pro_se) else p_orgs[len(plaintiffs):]):
        atty = b.person()
        blk = (f"Plaintiff\n{fmt_person(plaintiffs[0], 0)} represented by {fmt_person(atty, 0)}\n{org}\n"
               f"(See above for address)\nATTORNEY TO BE NOTICED")
        nb += 1
        items.append((f"PB{nb}", blk))
        b.fact("counsel", "true", org, filing, f"represented by {fmt_person(atty, 0)} {org}", f"PB{nb}")
        if t01:
            b.fact("counsel_type", "true", org, filing, f"represented by {fmt_person(atty, 0)} {org}", f"PB{nb}")
        out.append(blk)
    def_counsel = []
    for i, dfn in enumerate(defendants):
        org = d_orgs[i % len(d_orgs)]
        atty = b.person()
        def_counsel.append((dfn, atty, org))
        blk = (f"Defendant\n{dfn} represented by {fmt_person(atty, 0)}\n{org}\n"
               f"{rng.randint(10, 999)} Capitol Avenue, {city}, {st}\nLEAD ATTORNEY\nATTORNEY TO BE NOTICED")
        nb += 1
        iid = f"PB{nb}"
        items.append((iid, blk))
        ftxt = f"Defendant {dfn} represented by {fmt_person(atty, 0)} {org}"
        b.fact("defendants", "true", dfn, filing, ftxt, iid)
        if v5:
            b.fact("parties", "true", "def:" + dfn, filing, ftxt, iid)
        b.fact("counsel", "distractor", org, filing, ftxt, iid)
        if t01:
            b.fact("counsel_type", "distractor", org, filing, ftxt, iid,
                   trap="gov_defense" if (v5 and GOV_LAW_OFFICE.search(org)) else "defense")
        out.append(blk)
    out.append("Date Filed # Docket Text")

    def render_entries(ents: list, cno: list, ini: str, tag: str, sink: list):
        page = 1
        for k, (d, _, text, fx) in enumerate(ents):
            num = k + 1
            line = f"{fmt_date(d, 0)}  {num}  {text} ({b.ini()})"
            iid = f"{tag}{num}"
            items.append((iid, line))
            for t in fx:
                b.fact(t[0], t[1], t[2], d, line, iid, trap=t[3] if len(t) > 3 else "")
            sink.append(line)
            if num % 45 == 0:
                page += 1
                sink.append(f"\nCase: {fmt_caseno(cno, 0, ini)}   As of: {fmt_date(end + dt.timedelta(days=40), 0)}   "
                            f"{page} of {page + rng.randint(1, 20)}")

    if transfer:
        render_entries([e for e in entries if e[0] >= tdate], caseno, jud_ini, "E", out)
        h0 = [f"Case: {fmt_caseno(caseno0, 0, ini0)}   As of: {fmt_date(tdate + dt.timedelta(days=5), 0)} "
              f"{r2.randint(1, 12)}:{r2.randint(10, 59)} AM EST   1 of {r2.randint(2, 6)}",
              "U.S. District Court", f"District of {st0}", f"CIVIL DOCKET FOR CASE #: {fmt_caseno(caseno0, 0, ini0)}"]
        cap0 = f"{caption} Date Filed: {fmt_date(filing, 0)}"
        asg0 = f"Assigned to: Judge {fmt_person(judge0, 0)} Date Terminated: {fmt_date(tdate, 0)}"
        hdr0 = "\n".join(h0 + [cap0, asg0, cause_line])
        sheet0 = ["\n" + hdr0]
        items.append(("HDR0", hdr0))
        b.fact("filing_date", "true", iso(filing), filing, cap0, "HDR0")
        b.fact("judge", "stale", judge0, filing, asg0, "HDR0")
        sheet0.append("Date Filed # Docket Text")
        render_entries([e for e in entries if e[0] < tdate], caseno0, ini0, "T", sheet0)
        out = sheet0 + ["\n"] + out if r2.random() < 0.5 else out + sheet0
    else:
        render_entries(entries, caseno, jud_ini, "E", out)
    if no_sheet:

        del b.facts[n_facts0:]
        items.clear()
        out.clear()

    docs: list[tuple[str, dt.date, list[tuple[str, list]]]] = []
    W = rng.choice([88, 96, 104])

    def fill_par(n_sent: int, legal: bool) -> str:
        sents = []
        for _ in range(n_sent):
            d = filing + dt.timedelta(days=rng.randint(-400, span))
            d2 = d + dt.timedelta(days=rng.randint(10, 200))
            if legal:
                c1 = [rng.randint(100, 999), rng.choice(["F.3d", "F.4th", "F. Supp. 2d", "F. Supp. 3d"]),
                      rng.randint(1, 1500)]
                yr = rng.randint(1985, max(1986, filing.year))
                cite1 = f"{c1[0]} {c1[1]} {c1[2]} ({rng.choice(['1st', '2d', '5th', '7th', '9th', '11th'])} Cir. {yr})"
                t = rng.choice(LEGAL_FILL).format(case1=f"{rng.choice(LAST)} v. {rng.choice(LAST)}", cite1=cite1,
                                                  case2=f"{rng.choice(LAST)} v. {rng.choice(COUNTIES)} County",
                                                  d=fmt_date(d, 1), cir=ORDINAL[rng.randint(1, 11)])
            elif rng.random() < 0.45:
                t = rng.choice(KEYWORD_NOISE).format(
                    topic=rng.choice(TOPICS), who=fmt_person(b.extra(), rng.randrange(2)), d=fmt_date(d, rng.randrange(4)),
                    d2=fmt_date(d2, rng.randrange(4)), fac=fac, n=rng.randint(3, 900), city=city)
            else:
                t = rng.choice(CONTENT_FILL).format(
                    topic=rng.choice(TOPICS), who=fmt_person(b.extra(), rng.randrange(2)), d=fmt_date(d, rng.randrange(4)),
                    d2=fmt_date(d2, rng.randrange(4)), fac=fac, n=rng.randint(3, 900), ord=ORDINAL[rng.randint(1, 4)].lower(),
                    role=rng.choice(ROLES), side=rng.choice(["Plaintiffs", "Defendants"]))
            sents.append(t)
        return " ".join(sents)

    pars: list[tuple[str, list]] = []
    pn = [0]

    def num() -> int:
        pn[0] += 1
        return pn[0]

    pars.append((f"{num()}. Plaintiffs bring this civil rights action against {', '.join(defendants)} to challenge the "
                 f"{conduct} at {fac}, which Defendants operate.",
                 [("facility", "true", fac)] + X(("cause", "true", f"conduct:{conduct}"))))
    if v5:
        pars.append((f"{num()}. {'Plaintiffs are' if n_pl > 1 else 'Plaintiff is'} {group} at {fac}"
                     f"{', identified in the caption' if extra_pls else ''}.", [("parties", "true", "group:" + group)]))
    for p in plaintiffs:
        t = (f"{num()}. Plaintiff {fmt_person(p, rng.randrange(2))} is a resident of {rng.choice(CITIES)} who was "
             f"subjected to the {conduct} beginning in {MONTHS_SHORT[rng.randint(0, 11)]} {filing.year - rng.randint(0, 2)}.")
        pars.append((t, [("plaintiffs", "true", p)] + X5(("parties", "distractor", "name:" + fmt_person(p, 1), "names"))))
    for dfn in defendants:
        pars.append((f"{num()}. Defendant {dfn} is responsible for the policies and practices described in this "
                     f"Complaint concerning {rng.choice(TOPICS)}.", [("defendants", "true", dfn)]
                    + X5(("parties", "true", "def:" + dfn))))
    if rng.random() < dis:
        pars.append((f"{num()}. Before {fmt_date(filing - dt.timedelta(days=rng.randint(30, 600)), 1)}, Plaintiff "
                     f"{fmt_person(plaintiffs[0], 1)} was briefly housed at {other_fac}, which is not a party to this "
                     f"action.", [("facility", "distractor", other_fac)]))
    jt = (f"{num()}. This Court has jurisdiction under {STATUTES[jur_keys[0]][0][0]} and "
          f"{STATUTES[jur_keys[1]][0][0]}.")
    pars.append((jt, [("statutes", "distractor", k) for k in jur_keys]
                 + X(*[("cause", "distractor", f"claim:{k}") for k in jur_keys])))
    if class_alleged:
        pars.append((f"{num()}. Plaintiffs bring this action on behalf of themselves and a class of all persons "
                     f"subjected to the {conduct} at {fac}, pursuant to Federal Rule of Civil Procedure 23.",
                     [("class_action", "true" if (cert_granted or v5) else "stale", "Class Action")]))
    elif v5:
        pars.append((f"{num()}. Plaintiff{'s' if n_pl > 1 else ''}, {group} at {fac}, bring{'' if n_pl > 1 else 's'} "
                     f"this action individually and seek{'' if n_pl > 1 else 's'} relief only for the harms personally "
                     f"suffered there.", [("class_action", "true", "Individual")]))
    else:
        pars.append((f"{num()}. Plaintiff{'s' if len(plaintiffs) > 1 else ''} bring{'' if len(plaintiffs) > 1 else 's'} "
                     f"this action individually and seek{'' if len(plaintiffs) > 1 else 's'} relief only for the harms "
                     f"personally suffered at {fac}.", [("class_action", "true", "Individual")]))
    for _ in range(rng.randint(3, 9)):
        pars.append((f"{num()}. " + fill_par(rng.randint(2, 4), False), []))
    for i, k in enumerate(claim_keys):
        pars.append((f"{num()}. COUNT {['I', 'II', 'III'][i]}: Defendants' {conduct} violates "
                     f"{rng.choice(STATUTES[k][0])}.", [("statutes", "true", k)] + X(("cause", "true", f"claim:{k}"))))
    rem = ", ".join(rng.choice(REMEDIES[k]) for k in remedy_keys)
    pars.append((f"WHEREFORE, Plaintiffs respectfully request that the Court award {rem}, and such other relief as "
                 f"is just.", [("remedies", "true", k) for k in remedy_keys]))
    if no_sheet:
        for i, org in enumerate(p_orgs):
            atty = b.person()
            pars.append((f"Respectfully submitted, /s/ {fmt_person(atty, 0)}, {org}, Counsel for Plaintiffs.",
                         [("counsel", "true", org), ("counsel_type", "true", org)]
                         + ([("lead_counsel", "true" if lead_final == org else "stale", org)] if i == 0 else [])))
        ans = [(f"ANSWER of Defendants to the Complaint in {short_cap}.", []),
               (fill_par(r2.randint(2, 3), True), [])]
        for dfn, atty, org in def_counsel:
            ans.append((f"Respectfully submitted, /s/ {fmt_person(atty, 0)}, {org}, Counsel for Defendant {dfn}.",
                        [("defendants", "true", dfn), ("counsel", "distractor", org),
                         ("counsel_type", "distractor", org,
                          "gov_defense" if (v5 and GOV_LAW_OFFICE.search(org)) else "defense")]
                        + X5(("parties", "true", "def:" + dfn))))
        docs.append(("ANSWER", when2(0.02, 0.08), ans))
        for d, _, text, fx in entries:
            keep = [t for t in fx if t[1] == "true" or (len(t) > 3 and t[3])]
            if keep:
                title = re.split(r"[.;:]", text)[0][:70].upper()
                docs.append((title, d, [(f"{fmt_date(d, 1)}: {text}", keep)]))
    docs.append(("COMPLAINT", filing, pars))

    cites: list = []
    for k in range(n_opinions):
        od = when(0.1 + 0.85 * k / n_opinions, 0.1 + 0.85 * (k + 0.9) / n_opinions)
        c = ["wl", od.year, rng.randint(100000, 9999999)] if rng.random() < 0.6 else \
            ["rep", rng.randint(100, 999), rng.choice(["F. Supp. 2d", "F. Supp. 3d"]), rng.randint(1, 1500)]
        cites.append(c)
        ps: list[tuple[str, list]] = [(f"{short_cap}, {fmt_cite(c)}", [("opinions", "true", c)]),
                                      (f"United States District Court, D. {st}. Decided {fmt_date(od, 1)}.", [])]
        if t01 and r2.random() < 0.6:
            claims_txt = ", ".join(STATUTES[kk][0][r2.randrange(len(STATUTES[kk][0]))] for kk in claim_keys)
            ps.append((f"Plaintiffs allege that Defendants' {conduct} at {fac} violates {claims_txt}.",
                       [("cause", "true", f"conduct:{conduct}")] + [("cause", "true", f"claim:{kk}") for kk in claim_keys]
                       + [("statutes", "true", kk) for kk in claim_keys]))
        for _ in range(rng.randint(3, 10)):
            ps.append((fill_par(rng.randint(2, 4), True), []))
        ps.append((f"IT IS SO ORDERED this {od.day}th day of {MONTHS_LONG[od.month - 1]}, {od.year}. "
                   f"{fmt_person(judge_at(od), 0).upper()}, UNITED STATES {'MAGISTRATE' if jtitle(od) != 'Judge' else 'DISTRICT'} JUDGE",
                   [("judge", "true" if judge_at(od) == (judge2 or judge1) else "stale", judge_at(od))]))
        docs.append(("MEMORANDUM OPINION AND ORDER", od, ps))

    if has_settlement:
        sd_ = signed if v5 else settle_date
        ps = [(f"SETTLEMENT AGREEMENT between Plaintiffs and {defendants[0]}, executed in {MONTHS_LONG[sd_.month - 1]} "
               f"{sd_.year}.", [])]
        if v5:
            ps.append((f"This Agreement was signed by all parties on {fmt_date(signed, 1)} and shall be submitted to the "
                       f"Court for approval.", [("settlement_date", "true", iso(signed)) if approval_missing else
                                                 ("settlement_date", "distractor", iso(signed), "signed")]))
        if amount is not None:
            ps.append((f"{rng.randint(1, 9)}. Defendants shall pay the total sum of {fmt_amount(amount, rng.randrange(3))} "
                       f"into a Settlement Fund administered by {rng.choice(LAST)} Claims Services.",
                       [("settlement_amount", "true" if amount_final == amount else "stale", amount)]))
            if rng.random() < dis:
                demand = amount * rng.randint(2, 5)
                ps.append((f"{rng.randint(10, 19)}. Plaintiffs' initial settlement demand of {fmt_amount(demand, 0)}, "
                           f"made on {fmt_date(when(0.3, 0.5), 1)}, was rejected by Defendants.",
                           [("settlement_amount", "distractor", demand)]))
            if rng.random() < dis:
                fees = rng.randint(4, 60) * 10_000
                ps.append((f"{rng.randint(20, 29)}. Separately, Defendants shall pay attorneys' fees and costs of "
                           f"{fmt_amount(fees, 0)} to Plaintiffs' counsel.", [("settlement_amount", "distractor", fees)]))
        if enforced == "Yes":
            t = rng.choice(["The Court shall retain jurisdiction over this matter to enforce the terms of this Agreement "
                            "until its expiration.",
                            "The parties consent to the continuing jurisdiction of the Court to enforce this Agreement."])
        else:
            t = rng.choice(["This Agreement is a private settlement and is not a consent decree; the Court shall not "
                            "retain jurisdiction to enforce it.",
                            "The parties agree that the Court will not retain jurisdiction; any breach shall be "
                            "remedied in a separate action for breach of contract."])
        ps.append((f"{rng.randint(30, 39)}. " + t, [("court_enforced", "true", enforced)]))
        for _ in range(rng.randint(2, 6)):
            ps.append((f"{rng.randint(40, 90)}. " + fill_par(rng.randint(2, 3), False), []))
        docs.append(("SETTLEMENT AGREEMENT", signed if v5 else settle_date, ps))
        if amend_date is not None:
            docs.append(("FIRST AMENDMENT TO SETTLEMENT AGREEMENT", amend_date, [
                (f"FIRST AMENDMENT TO SETTLEMENT AGREEMENT in {short_cap}, dated {fmt_date(amend_date, 1)}.", []),
                (f"1. Pursuant to this First Amendment, dated {fmt_date(amend_date, 1)}, the Settlement Fund "
                 f"established by the Agreement is increased to {fmt_amount(amount_final, rng.randrange(3))}.",
                 [("settlement_amount", "true", amount_final)]),
                ("2. " + fill_par(rng.randint(2, 3), False), [])]))

    if has_decree:
        ps = [(f"CONSENT DECREE in {short_cap}, entered {fmt_date(decree_date, 1)}.", [])]
        ps.append((f"{rng.randint(1, 9)}. Term. This Decree shall remain in effect for "
                   f"{fmt_duration(duration, rng.randrange(3))} from the Effective Date, unless extended by the Court.",
                   [("decree_duration", "true" if duration_final == duration else "stale", duration)]))
        if rng.random() < dis:
            alt = duration + rng.choice([-12, 12, 24]) if duration > 12 else duration + 12
            ps.append((f"{rng.randint(10, 19)}. Defendants had proposed a term of {fmt_duration(alt, 1)}, which "
                       f"Plaintiffs rejected during negotiations.", [("decree_duration", "distractor", alt)]))
        if t01:
            ps.append((f"{r2.randint(10, 19)}. Defendants shall end the {conduct} at {fac} and adopt written policies "
                       f"to prevent its recurrence" + (", subject to review by the Monitor." if has_monitor else "."),
                       [("decree_terms", "true", f"conduct:{conduct}")]
                       + ([("decree_terms", "true", "monitor")] if has_monitor else [])))
        if has_monitor:
            ps.append((f"{rng.randint(20, 29)}. The Monitor shall file a written report with the Court every six "
                       f"months, beginning no later than {fmt_date(when(0.75, 0.8), 1)}.", []))
        for _ in range(rng.randint(2, 6)):
            ps.append((f"{rng.randint(30, 90)}. " + fill_par(rng.randint(2, 3), False), []))
        docs.append(("CONSENT DECREE", decree_date, ps))
        if ext_doc is not None:
            docs.append(("ORDER EXTENDING CONSENT DECREE", ext_doc, [
                (f"ORDER EXTENDING CONSENT DECREE in {short_cap}.", []),
                (f"For the reasons stated on the record, the motion is GRANTED: as of {fmt_date(ext_doc, 1)}, the "
                 f"Consent Decree shall remain in effect for a total of {fmt_duration(duration_final, rng.randrange(3))} "
                 f"from its effective date.", [("decree_duration", "true", duration_final)]),
                (fill_par(rng.randint(2, 3), True), [])]))

    if has_monitor and appoint_doc:
        m1d, m1p = monitors[0]
        docs.append(("ORDER ON COMPLIANCE OVERSIGHT", m1d, [
            (f"ORDER ON COMPLIANCE OVERSIGHT in {short_cap}, entered {fmt_date(m1d, 1)}.", []),
            (rng.choice([
                f"Having considered the parties' submissions, the Court designates {fmt_person(m1p, 0, 'Dr.')} as the "
                f"Independent Reviewer, who shall serve as the Monitor contemplated by the decree.",
                f"{fmt_person(m1p, 0)} is hereby appointed Compliance Expert (the \"Monitor\") and shall have access to "
                f"all records and staff at {fac}."]),
             [("monitor", "true" if len(monitors) == 1 else "stale", m1p)]
             + (X(("decree_terms", "true", "monitor")) if has_decree else [])),
            (fill_par(rng.randint(2, 4), True), [])]))
    if class_alleged and cert_doc:
        ps = [(f"{short_cap} -- MEMORANDUM OPINION AND ORDER ON CLASS CERTIFICATION, {fmt_date(class_date, 1)}.", [])]
        for _ in range(rng.randint(2, 5)):
            ps.append((fill_par(rng.randint(2, 4), True), []))
        ps.append(((f"Because the proposed class satisfies Rule 23(a) and (b)(2), the motion is GRANTED and the Court "
                    f"certifies a class of all persons subjected to the {conduct} at {fac}.") if cert_granted else
                   ("Because common questions do not predominate and the named plaintiffs' claims are not typical, the "
                    "motion for class certification is DENIED; the named plaintiffs may proceed individually."),
                   [("class_action", "distractor", "Individual", "cert_denied") if (v5 and not cert_granted) else
                    ("class_action", "true", "Class Action" if cert_granted else "Individual")]
                   + X(RUL(class_date, "grant" if cert_granted else "deny",
                           "certified the class" if cert_granted else "denied class certification"))))
        docs.append(("MEMORANDUM OPINION AND ORDER ON CLASS CERTIFICATION", class_date, ps))
    for od, prev, new, last in deadline_docs:
        docs.append(("ORDER EXTENDING COMPLIANCE DEADLINE", od, [
            (f"ORDER EXTENDING COMPLIANCE DEADLINE in {short_cap}, entered {fmt_date(od, 1)}.", []),
            (f"Good cause having been shown, the deadline for Defendants to complete the required remedial measures is "
             f"extended from {fmt_date(prev, 1)} to {fmt_date(new, rng.randrange(4))}.",
             [("deadline", "true" if last else "stale", iso(new))]),
            (fill_par(rng.randint(2, 3), True), [])]))
    if subst is not None and lead_final == subst[1] and not any(
            f["key"] == "lead_counsel" and f["kind"] == "true" for f in b.facts):
        sd, new_org = subst
        docs.append(("NOTICE OF SUBSTITUTION OF COUNSEL", sd, [
            (f"NOTICE OF SUBSTITUTION OF COUNSEL for Plaintiffs in {short_cap}, dated {fmt_date(sd, 1)}.", []),
            (f"Effective {fmt_date(sd, 1)}, {new_org} replaces {p_orgs[0]} as lead counsel for all Plaintiffs; "
             f"{p_orgs[0]} withdraws from the representation.",
             [("lead_counsel", "true", new_org), ("lead_counsel", "stale", p_orgs[0])]),
            (fill_par(rng.randint(2, 3), False), [])]))
    if has_monitor and succ_doc:
        md2, m2 = monitors[1]
        docs.append(("ORDER APPOINTING SUCCESSOR MONITOR", md2, [
            (f"ORDER APPOINTING SUCCESSOR MONITOR in {short_cap}.", []),
            (f"Effective {fmt_date(md2, 1)}, {fmt_person(m2, 0, 'Dr.')} replaces {fmt_person(monitors[0][1], 1)} as the "
             f"Monitor under the decree and shall exercise all of the Monitor's powers.",
             [("monitor", "true", m2), ("monitor", "stale", monitors[0][1])]),
            (fill_par(rng.randint(2, 3), True), [])]))
    if has_monitor:
        for k, rd_s in enumerate(report_dates[:2 + extra // 2]):
            rd = dt.date.fromisoformat(rd_s)
            cur = [m for (dd, m) in monitors if dd <= rd][-1]
            ps = [(f"MONITOR'S {ORDINAL[k + 1].upper()} REPORT, submitted on {fmt_date(rd, 1)} by "
                   f"{fmt_person(cur, 0, 'Dr.')}, Monitor.",
                   [("monitor_reports", "true", rd_s), ("monitor", "true" if cur == monitor_final else "stale", cur)])]
            for _ in range(rng.randint(2, 6)):
                ps.append((fill_par(rng.randint(2, 4), False), []))
            docs.append(("MONITOR REPORT", rd, ps))

    if has_appeal and rng.random() < 0.6:
        ad = when(0.6, 0.95)
        c = ["rep", rng.randint(100, 999), rng.choice(["F.3d", "F.4th"]), rng.randint(1, 1500)]
        cites.append(c)
        ps = [(f"{short_cap}, {fmt_cite(c)} ({ORDINAL[cir]} Cir. {ad.year})",
               [("opinions", "true", c)]),
              (f"United States Court of Appeals, {ORDINAL[cir]} Circuit. Argued and decided {fmt_date(ad, 1)}.", [])]
        for _ in range(rng.randint(2, 6)):
            ps.append((fill_par(rng.randint(2, 4), True), []))
        ps.append((f"For these reasons, the judgment of the district court is {appeal['outcome'].upper()}.",
                   [("appeal", "true", {"outcome": appeal["outcome"]})]))
        docs.append(("APPELLATE OPINION", ad, ps))
    if rng.random() < dis * 0.7:
        oc = rng.randint(1, 11)
        ps = [(f"NOTICE OF SUPPLEMENTAL AUTHORITY by {rng.choice(['Plaintiffs', 'Defendants'])}.", []),
              (f"In {rng.choice(LAST)} v. {rng.choice(COUNTIES)} County, the {ORDINAL[oc]} Circuit recently "
               f"affirmed an injunction on similar facts.", [("appeal", "distractor", {"circuit": oc, "outcome": "affirmed"})]),
              (fill_par(rng.randint(2, 4), True), [])]
        docs.append(("NOTICE OF SUPPLEMENTAL AUTHORITY", when(0.2, 0.9), ps))
    if has_related or rng.random() < dis * 0.5:
        xc = [rng.randint(1, 4), rng.randint(90, 99) if rng.random() < 0.3 else rng.randint(0, 19), rng.randint(100, 9999)]
        ps = [(f"DECLARATION of {fmt_person(b.extra(), 0)} in support of Plaintiffs' motion.", []),
              (f"I previously served as an expert witness in {rng.choice(LAST)} v. {rng.choice(LAST)}, "
               f"{fmt_caseno(xc, 0)} ({rng.choice(['D. Md.', 'N.D. Ill.', 'S.D. Tex.', 'E.D. Pa.'])}), which concerned "
               f"{rng.choice(TOPICS)}.", [("related", "distractor", xc)]),
              (fill_par(rng.randint(3, 5), False), [])]
        docs.append(("DECLARATION", when(0.1, 0.6), ps))

    if t01:
        hosts = [x for x in docs if x[0] != "COMPLAINT"] or docs
        for _ in range(r2.randint(1, 2 + extra)):
            xc = [r2.randint(1, 4), r2.randint(0, 19) if r2.random() < 0.7 else r2.randint(90, 99), r2.randint(100, 9999)]
            ref = fmt_caseno(xc, r2.randrange(3))
            ref = ref if ref.startswith("No.") else "No. " + ref
            par = r2.choice([
                f"See {r2.choice(LAST)} v. {r2.choice(COUNTIES)} County, {ref} "
                f"({r2.choice(['D. Md.', 'N.D. Ill.', 'S.D. Tex.', 'E.D. Pa.', 'W.D. Wash.'])}), where a consent decree "
                f"addressed {r2.choice(TOPICS)}.",
                f"Defendants point to the settlement approved in {r2.choice(LAST)} v. {r2.choice(LAST)}, {ref}, which "
                f"concerned {r2.choice(TOPICS)} at another facility.",
                f"The same counsel litigated {r2.choice(LAST)} v. {r2.choice(COUNTIES)} County, {ref}, a case about "
                f"{r2.choice(TOPICS)} that is not before this Court."])
            host = r2.choice(hosts)
            host[2].insert(r2.randint(1, len(host[2])), (par, [("related", "distractor", xc, "cited")]))

    cur_len = sum(len(x) for x in out) + sum(len(p) for _, _, ps in docs for p, _ in ps)
    k = 0
    while cur_len < target:
        k += 1
        legal = rng.random() < 0.4
        ps = [(f"EXHIBIT {k} TO {'MEMORANDUM' if legal else 'DECLARATION'} OF {fmt_person(b.extra(), 0).upper()}", [])]
        for _ in range(rng.randint(4, 12)):
            ps.append((fill_par(rng.randint(3, 6), legal), []))
        docs.append((f"EXHIBIT {k}", filing + dt.timedelta(days=rng.randint(30, span)), ps))
        cur_len += sum(len(p) for p, _ in ps)

    rng.shuffle(docs)
    dn = 0
    for title, dd, ps in docs:
        dn += 1
        n_pages = max(1, sum(len(p) for p, _ in ps) // 2600 + 1)
        out.append(f"\nCase {fmt_caseno(caseno, 0, jud_ini)}   Document {rng.randint(20, 400)}   Filed "
                   f"{fmt_date(dd, 0)}   Page 1 of {n_pages}\n{title}")
        for j, (p, fx) in enumerate(ps):
            iid = f"D{dn}.{j}"
            text = "\n".join(wrap(p, W))
            items.append((iid, text))
            for t in fx:
                b.fact(t[0], t[1], t[2], dd, p, iid, trap=t[3] if len(t) > 3 else "")
            out.append(text)
    context = "\n".join(out)

    present_keys = {
        "filing_date": iso(filing), "judge": judge2 or judge1,
        "class_action": "Class Action" if cert_granted else "Individual", "counsel": p_orgs,
        "statutes": sorted(claim_keys), "remedies": sorted(remedy_keys),
        "settlement_date": iso(settle_date) if settle_date else "N/A",
        "settlement_amount": amount_final if amount_final is not None else "N/A",
        "court_enforced": enforced or ("No" if t01 else "N/A"),
        "decree_duration": duration_final if duration_final is not None else "N/A",
        "monitor": monitor_final or "N/A", "monitor_reports": sorted(report_dates) or "N/A",
        "decree_dates": sorted(set(decree_dates)) or "N/A", "appeal": appeal or "N/A",
        "trial": iso(trial_date) if trial_date else "N/A",
        "related": [list(x) for x in rel_nos] or "N/A", "consolidated": [list(x) for x in con_nos] or "N/A",
        "disputes": sorted(dispute_dates) or "N/A", "plaintiffs": plaintiffs, "defendants": defendants,
        "opinions": cites or "N/A", "facility": fac,
        "deadline": iso(deadline_final) if deadline_final else "N/A", "lead_counsel": lead_final,
    }
    if t01:
        present_keys["related"] = (hdr_related if not no_sheet else [list(x) for x in rel_nos]) or "N/A"
        terms = ((["conduct:" + conduct] if (pi_granted or has_decree) else [])
                 + (["monitor"] if (has_decree and has_monitor) else []) + (["deadline"] if has_decree else [])
                 + list(dict.fromkeys(extra_terms)))
        present_keys.update({
            "counsel_type": {"public": [o for o in p_orgs if o in pub], "private": [o for o in p_orgs if o not in pub]},
            "cause": ["conduct:" + conduct] + [f"claim:{k}" for k in sorted(claim_keys)],
            "rulings": [list(x) for x in sorted({tuple(x) for x in rul})] or "N/A",
            "filings": [list(x) for x in sorted({tuple(x) for x in fil})] or "N/A",
            "decree_terms": terms or "N/A",
        })
    if t01:
        have = {(f["item"], f["date"]) for f in b.facts if f["key"] == "decree_dates" and f["kind"] == "true"}
        for f in list(b.facts):
            if f["key"] == "rulings" and f["kind"] == "true" and f["val"][1] != "transfer" and (f["item"], f["date"]) not in have:
                have.add((f["item"], f["date"]))
                b.fact("decree_dates", "true", f["val"][0], dt.date.fromisoformat(f["date"]), f["text"], f["item"])
        present_keys["decree_dates"] = sorted({ev[0] for ev in rul if ev[1] != "transfer"} | set(decree_dates)) or "N/A"
    if v5:
        for f in b.facts:
            if f["key"] == "court_enforced" and f["kind"] in ("true", "distractor") and not f.get("trap"):
                f["kind"] = "true" if f["val"] == ce_gold else "distractor"
        present_keys["court_enforced"] = ce_gold
        present_keys["class_action"] = "Class Action" if class_alleged else "Individual"
        present_keys["plaintiffs"] = all_pls
        present_keys["counsel"] = p_orgs
        present_keys["counsel_type"] = {"public": [o for o in p_orgs if o in pub],
                                        "private": [o for o in p_orgs if o not in pub], "pro_se": pro_se}
        present_keys["parties"] = {"group": group, "n": n_pl, "kind": kind_s,
                                   "defendants": defendants, "names": [fmt_person(p, 1) for p in all_pls]}
        present_keys["settlement_date"] = (iso(settle_date) if not approval_missing else iso(signed)) if has_settlement \
            else (iso(decree_date) if has_decree else "N/A")
    lo, hi = P.get("n_fields", (17, 24))
    keys = [k for k in DOCKET_FIELDS if k not in T01_KEYS and k not in T01V5_KEYS]
    always = ["filing_date", "judge", "decree_dates"]
    rest = [k for k in keys if k not in always]
    chosen = always + rng.sample(rest, max(0, min(len(rest), rng.randint(lo, hi) - len(always))))
    order = [k for k in keys if k in chosen]
    if rng.random() < 0.5:
        rng.shuffle(order)
    if t01:
        for k, p in (("counsel_type", 1.0), ("cause", 1.0), ("rulings", 1.0), ("filings", 0.7), ("decree_terms", 0.7),
                     ("related", 0.85), ("trial", 0.6), ("decree_duration", 0.6), ("statutes", 0.6)):
            if k not in order and r2.random() < p:
                order.insert(r2.randint(0, len(order)), k)
    v5_added: set = set()
    if v5:
        for k, p in (("parties", 0.85), ("class_action", 0.8), ("court_enforced", 0.8), ("settlement_date", 0.7)):
            if k not in order and r3.random() < p:
                order.insert(r3.randint(0, len(order)), k)
                v5_added.add(k)
    names = {k: (r2.choice(DOCKET_FIELDS[k][2]) if k in T01_KEYS else
                 r3.choice(DOCKET_FIELDS[k][2]) if (k in T01V5_KEYS or k in v5_added) else None) for k in order}
    fields = [{"key": k, "name": names[k] or rng.choice(DOCKET_FIELDS[k][2]), "type": DOCKET_FIELDS[k][0],
               "rule": DOCKET_FIELDS[k][1], "gold": present_keys[k]} for k in order]
    if v5:
        for f in fields:
            if f["key"] == "class_action" and f["gold"] == "Individual":
                f["desc"] = {"group": group, "n": n_pl, "kind": kind_s, "names": [fmt_person(p, 1) for p in all_pls]}
    return _finish("docket", context, items, b.facts, fields,
                   "You are a legal analyst. Read the court docket in `context` and produce a structured summary of "
                   "the case.", P)


MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]
MONTHS_SHORT = MONTHS_LONG


DRUGS = ["lisinopril", "metformin", "atorvastatin", "levothyroxine", "amlodipine", "sertraline", "omeprazole",
         "albuterol", "gabapentin", "losartan", "hydrochlorothiazide", "montelukast", "escitalopram", "pantoprazole",
         "meloxicam", "tamsulosin"]
NEW_DRUGS = ["naproxen 500 mg twice daily", "prednisone 20 mg daily", "amoxicillin 875 mg twice daily",
             "cyclobenzaprine 10 mg at bedtime", "ibuprofen 600 mg three times daily", "fluticasone nasal spray daily",
             "cephalexin 500 mg four times daily", "ondansetron 4 mg as needed"]
COMPLAINTS = [("knee pain", "right knee", "knee osteoarthritis"), ("lower back pain", "lower back", "lumbar strain"),
              ("headaches", "forehead", "tension-type headache"), ("cough", "chest", "acute bronchitis"),
              ("shoulder pain", "left shoulder", "rotator cuff tendinopathy"), ("heartburn", "upper abdomen", "GERD"),
              ("ankle swelling", "left ankle", "ankle sprain"), ("sore throat", "throat", "streptococcal pharyngitis"),
              ("wrist pain", "right wrist", "carpal tunnel syndrome"), ("dizziness", "head", "benign positional vertigo")]
ALLERGENS = ["penicillin", "sulfa drugs", "latex", "codeine", "shellfish", "aspirin"]
SURGERIES = ["appendectomy", "gallbladder removal", "knee arthroscopy", "C-section", "tonsillectomy", "hernia repair"]
FAMILY = ["diabetes", "breast cancer", "heart disease", "stroke", "high blood pressure", "colon cancer"]
JOBS = ["warehouse supervisor", "middle-school teacher", "bus driver", "dental hygienist", "software tester",
        "line cook", "accountant", "electrician"]


def gen_clinic(rng: random.Random, P: dict) -> Generated:
    b = Builder(rng)
    lines: list[tuple[str, str, list]] = []
    D0 = dt.date(2020, 1, 1)
    sex = rng.choice(["female", "male"])
    age = rng.randint(19, 84)
    name = rng.choice(FIRST)
    comp, loc, dx = rng.choice(COMPLAINTS)
    onset1 = rng.randint(2, 20)
    onset_final = onset1 + rng.choice([2, 3, 7]) if rng.random() < 0.5 else onset1
    sev1 = rng.randint(3, 7)
    sev_final = min(10, sev1 + rng.randint(1, 3)) if rng.random() < 0.6 else sev1
    meds = rng.sample(DRUGS, rng.randint(0, 3))
    allergies = rng.sample(ALLERGENS, rng.randint(1, 2)) if rng.random() < 0.45 else []
    surg = rng.sample(SURGERIES, 1) if rng.random() < 0.35 else []
    fam = rng.sample(FAMILY, rng.randint(1, 2)) if rng.random() < 0.4 else []
    smoke = rng.choice(["never", "former", "current"]) if rng.random() < 0.6 else None
    job = rng.choice(JOBS) if rng.random() < 0.45 else None
    bp1 = f"{rng.randint(118, 160)}/{rng.randint(72, 98)}"
    bp_final = f"{rng.randint(112, 150)}/{rng.randint(70, 92)}" if rng.random() < 0.5 else bp1
    has_bp = rng.random() < 0.6
    rx = rng.choice(NEW_DRUGS) if rng.random() < 0.7 else None
    fu = rng.choice([7, 14, 28, 42, 90]) if rng.random() < 0.65 else None
    t = [0]

    def say(sp: str, text: str, fx: list | None = None):
        t[0] += 1
        lines.append((sp, text, fx or []))

    def chit():
        say("doctor", rng.choice(["okay .", "mm-hmm , go on .", "got it .", "alright , and then ?", "i see ."]))

    say("doctor", f"hi {name} , good to see you . so you're a {age}-year-old {'woman' if sex == 'female' else 'man'} "
                  f"coming in today for {comp} , is that right ?",
        [("age", "true", age), ("sex", "true", sex), ("reason", "true", comp)])
    say("patient", f"yeah , that's right . the {comp} has really been bothering me .")
    say("doctor", f"when did the {comp} start ?")
    say("patient", f"um , i think it was about {onset1} days ago , give or take .", [("onset", "true" if onset1 == onset_final else "stale", onset1)])
    if onset_final != onset1:
        chit()
        say("patient", f"actually , wait , now that i think about it , it started {onset_final} days ago , the "
                       f"weekend of my cousin's wedding .", [("onset", "true", onset_final)])
    say("doctor", "and where exactly do you feel it ?")
    say("patient", f"mostly in my {loc} , it kind of stays there .", [("location", "true", loc)])
    spouse = "husband" if sex == "female" else "wife"
    if rng.random() < 0.6:
        part = rng.choice(["neck", "hip", "elbow"])
        say("patient", f"my {rng.choice([spouse, 'sister', 'brother'])} had something similar in the {part} last "
                       f"year , but that went away .", [("location", "distractor", part)])
    say("doctor", "on a scale of zero to ten , how bad is it ?")
    say("patient", f"i'd say it's a {sev1} most days .", [("severity", "true" if sev1 == sev_final else "stale", sev1)])
    if sev_final != sev1:
        say("patient", f"although , honestly , the last couple of days it's been more like a {sev_final} .",
            [("severity", "true", sev_final)])
    say("doctor", "are you taking any medications right now ?")
    if meds:
        say("patient", f"i take {', '.join(meds[:-1]) + ' and ' + meds[-1] if len(meds) > 1 else meds[0]} every day .",
            [("meds", "true", m) for m in meds])
    else:
        say("patient", "no , nothing regular .")
    if rng.random() < 0.6:
        other = rng.choice([d for d in DRUGS if d not in meds])
        say("patient", f"my {rng.choice([spouse, 'mother'])} takes {other} but i don't .",
            [("meds", "distractor", other)])
    if rng.random() < 0.4:
        stopped = rng.choice([d for d in DRUGS if d not in meds])
        say("patient", f"i used to be on {stopped} but my last doctor stopped it two years ago .",
            [("meds", "distractor", stopped)])
    say("doctor", "any allergies to medications ?")
    if allergies:
        say("patient", f"yes , i'm allergic to {' and '.join(allergies)} .", [("allergies", "true", a) for a in allergies])
    else:
        say("patient", "no known allergies .")
    if rng.random() < 0.5:
        say("doctor", "have you had any surgeries ?")
        if surg:
            say("patient", f"just a {surg[0]} back in {rng.randint(1990, 2018)} .", [("surgical", "true", surg[0])])
        else:
            rel_s = rng.choice(SURGERIES)
            say("patient", f"no , never . my {rng.choice(['brother', 'dad'])} had a {rel_s} though .",
                [("surgical", "distractor", rel_s)])
    elif surg:
        say("patient", f"oh , and i had a {surg[0]} a while back , in case that matters .", [("surgical", "true", surg[0])])
    if fam or rng.random() < 0.5:
        say("doctor", "does anything run in your family ?")
        if fam:
            say("patient", f"my {rng.choice(['mother', 'father'])} had {' and '.join(fam)} .",
                [("family", "true", f) for f in fam])
        else:
            say("patient", "not that i know of .")
    if smoke is not None:
        say("doctor", "do you smoke ?")
        say("patient", {"never": "no , i've never smoked .", "former": f"i quit about {rng.randint(2, 20)} years ago .",
                        "current": f"yeah , about {rng.choice(['half a pack', 'a pack', 'five cigarettes'])} a day ."}[smoke],
            [("smoking", "true", smoke)])
    if job:
        say("doctor", "what do you do for work ?")
        say("patient", f"i'm a {job} , so i'm on my feet a lot .", [("occupation", "true", job)])
    elif rng.random() < 0.4:
        rel_j = rng.choice(JOBS)
        say("patient", f"my {rng.choice(['son', 'daughter'])} is a {rel_j} and keeps telling me to rest .",
            [("occupation", "distractor", rel_j)])
    qa = [(f"and is the {comp} worse in the morning or at night ?", "it's usually worse at night , i think ."),
          ("have you tried anything for it at home ?", f"i tried ice and a heating pad on the {loc} ."),
          ("any fevers or chills ?", "no fevers , no chills ."), ("how is your sleep ?", "sleep's been okay , not great ."),
          (f"does anything make the {comp} better ?", "resting helps a little ."),
          ("any numbness or tingling ?", "no , none of that ."),
          ("how is your appetite ?", "about the same as usual ."),
          ("are you able to keep up with your daily activities ?", "mostly , but i've cut back on walking .")]
    for q, a_ in rng.sample(qa, rng.randint(2, 6)):
        say("doctor", q)
        say("patient", a_)
    if has_bp:
        say("doctor", f"your blood pressure today was {bp1.replace('/', ' over ')} when you came in .",
            [("bp", "true" if bp1 == bp_final else "stale", bp1)])
        if bp_final != bp1:
            say("doctor", f"we rechecked it after you rested and it's {bp_final.replace('/', ' over ')} now .",
                [("bp", "true", bp_final)])
    say("doctor", f"on exam you have some tenderness in the {loc} .")
    say("doctor", f"so my impression is that this is {dx} .", [("diagnosis", "true", dx)])
    if rx:
        say("doctor", f"i'm going to start you on {rx} .", [("prescribed", "true", rx.split()[0])])
    if fu:
        say("doctor", f"let's follow up in {str(fu // 7) + ' weeks' if fu < 90 else '3 months'} "
                      f"and see how you're doing .", [("followup", "true", fu)])
    say("patient", "sounds good , thank you .")

    items, out = [], []
    for i, (sp, text, fx) in enumerate(lines):
        line = f"[{sp}] {text}"
        iid = f"L{i + 1}"
        items.append((iid, line))
        out.append(line)
        for key, kind, val in fx:
            b.fact(key, kind, val, D0 + dt.timedelta(days=i), line, iid)
    gold_all = {
        "age": age, "sex": sex, "reason": comp, "onset": onset_final, "location": loc, "severity": sev_final,
        "meds": meds or "N/A", "allergies": allergies or "N/A", "surgical": surg or "N/A", "family": fam or "N/A",
        "smoking": smoke or "N/A", "occupation": job or "N/A", "bp": bp_final if has_bp else "N/A", "diagnosis": dx,
        "prescribed": [rx.split()[0]] if rx else "N/A", "followup": fu if fu else "N/A",
    }
    lo, hi = P.get("n_fields_short", (10, 16))
    chosen = rng.sample(list(CLINIC_FIELDS), rng.randint(lo, hi))
    order = [k for k in CLINIC_FIELDS if k in chosen]
    fields = [{"key": k, "name": CLINIC_FIELDS[k][2][0], "type": CLINIC_FIELDS[k][0], "rule": CLINIC_FIELDS[k][1],
               "gold": gold_all[k]} for k in order]
    return _finish("clinic", "\n".join(out), items, b.facts, fields,
                   "You are a clinician. Read the doctor-patient dialogue in `context` and produce the structured "
                   "clinical note it supports.", P)


ROOT_TAIL = ("Write 'N/A' for a field the source genuinely does not support — do not invent content. Submit a single "
             "JSON object mapping each field name above to its value, and nothing else.")


def root_prompt(desc: str, names: list[str]) -> str:
    """Byte-for-byte the real env's root prompt shape (expertlongbench_real._build_dataset)."""
    field_list = "\n".join(f"  - {n}" for n in names)
    return (f"{desc}\n\nFill in the following expert checklist. Every field must be present.\n\n{field_list}\n\n"
            + ROOT_TAIL)


def _finish(variant: str, context: str, items: list, facts: list, fields: list, desc: str, P: dict) -> Generated:
    keys = {f["key"] for f in fields}
    facts = [f for f in facts if f["key"] in keys]
    req = sorted({f["item"] for f in facts if f["kind"] == "true" and f["item"]})
    item_lines = {ln.strip() for _, t in items for ln in t.split("\n")}
    noise = sorted({ln.strip() for ln in context.split("\n") if ln.strip() and ln.strip() not in item_lines})
    hidden = {"variant": variant, "facts": json.loads(json.dumps(facts)), "fields": json.loads(json.dumps(fields)),
              "noise_lines": noise, "context_chars": len(context)}
    return Generated(context=context, root_prompt=root_prompt(desc, [f["name"] for f in fields]), items=items,
                     gold={f["name"]: f["gold"] for f in fields}, hidden=hidden, required_ids=req, index_min_len=20)


def generate(info: dict) -> Generated:
    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        return _generate(info)
    finally:
        if gc_was_on:
            gc.enable()


def _generate(info: dict) -> Generated:
    P = json.loads(info.get("synth_params") or "{}") if isinstance(info.get("synth_params"), str) else \
        dict(info.get("synth_params") or {})
    seed_p = {k: v for k, v in P.items() if k not in V5_PARAMS}
    rng = random.Random(f"elb-synth:{info['synth_split']}:{info['synth_seed']}:{json.dumps(seed_p, sort_keys=True)}")
    variant = P.get("variant")
    if variant is None:
        variant = "clinic" if rng.random() < float(P.get("p_short", 0.25)) else "docket"
    return gen_clinic(rng, P) if variant == "clinic" else gen_docket(rng, P)
