"""Short expert checklists (visit / tutor / recipe forms, like ELB T06 / T04 / T03)."""

from __future__ import annotations

import json
import random
import re
from typing import Any

from synth_common.envkit import Generated
from synth_common.oracle import FakeSubLM, ReadModel, rng_for
from synth_common.program import (
    Check,
    HintSpec,
    add_hint,
    answer_not_copied,
    final_answer,
    root_code,
)

from .final_notes import RECIPE_NOTE_C5, visit_note_c10
from .generate import COMPLAINTS, FAMILY, JOBS, NEW_DRUGS, SURGERIES, root_prompt
from .textkit import FIRST
from .tutor_note import TUTOR_NOTE, TUTOR_NOTE_V1S

NA_TOKENS = {"n/a", "na", "none", "not applicable", "not mentioned", ""}


def is_na(v: Any) -> bool:
    return str(v if v is not None else "").strip().lower() in NA_TOKENS


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _has(cand_n: str, kw: str) -> bool:
    return f" {_norm(kw)} " in f" {cand_n} "


VISIT_FIELDS = {
    "age": ("Patient Age", "scalar"), "sex": ("Patient Sex", "scalar"),
    "visit_reason": ("Reason for Patient Office Visit or Hospitalization", "phrase"),
    "reason": ("Patient Reason for the Visit", "sentence"), "location": ("Location", "scalar"),
    "onset": ("Onset", "trap"), "duration": ("Duration", "scalar"),
    "character": ("Character", "narr"), "factors": ("Alleviating and Aggravating Factors", "narr"),
    "radiation": ("Radiation", "trap"), "temporal": ("Temporal Factor", "trap"), "severity": ("Severity", "trap"),
    "drugs": ("Drugs", "trap"), "vitals": ("Vital Signs", "narr"), "exam": ("Physical Examination Findings", "narr"),
    "objective": ("Other Objective Data", "narr"), "diagnosis": ("Diagnosis and Clinical Impression", "narr"),
    "plan": ("Treatment and Management Plan", "narr"), "pmh": ("Relevant Medical History", "narr"),
    "surgical": ("Surgical History", "narr"), "family": ("Family History", "narr"),
    "employment": ("Employment", "trap"), "activities": ("Activities", "trap"),
    "home": ("Home and Environment", "trap"), "eating": ("Eating habits", "trap"),
    "ros": ("Review of Systems (ROS)", "ros"),
}
SYMPTOM_KEYS = ("onset", "duration", "location", "character", "factors", "radiation", "severity", "temporal")
SYMPTOM_NAMES = tuple(VISIT_FIELDS[k][0] for k in SYMPTOM_KEYS)

TRAP_DEFS = {
    "drugs": r"recreational|alcohol|tobacco|substance|smok|not (?:prescri|medic)|never (?:prescri|medic)|excluding medic",
    "severity": r"numeric|0\s*-\s*10|0 to 10|zero to ten|out of 10|/10|rating|scale",
    "temporal": r"pattern|time of day|worse (?:at|in) |not (?:the )?onset|only the onset|timing pattern|cyclical",
    "radiation": r"spread|radiat\w* (?:to|down|into)|denial|denies|unless",
    "employment": r"(?:patient'?s?|own|their) (?:job|work|occupation)|not (?:a )?relative|family member'?s",
    "onset": r"as a time|not (?:the )?mechanism|when .{0,40}started",
    "activities": r"activit|exercise|hobb|sport",
    "home": r"lives with|living|household|home",
    "eating": r"diet|eating",
    "pmh": r"condition|histor|diagnos",
}
_NEG = r"\bnot\b|\bnever\b|\bn/?a\b|exclud|\bisn'?t\b|\baren'?t\b"
SUB_EXCLUDE = {
    "onset": ("neg", r"onset"),
    "constant": ("neg", r"constant|intermittent|comes and goes"),
    "worsening": ("neg", r"getting worse|worsening|progressi"),
    "chronic": ("neg", r"chronic"),
    "activity": ("neg", r"activit|trigger"),
    "mechanism": ("neg", r"mechanism|injur|how .{0,20}happen"),
    "restriction": ("neg", r"restriction|plans?\b"),
    "absence": ("neg", r"absence|no prior"),
    "history": ("neg", r"histor"),
    "spouse": ("pos", r"live\w* with|living with|passing mention"),
    "liveswith": ("neg", r"live\w* with|who (?:they|the patient) lives?"),
    "endofday": ("neg", r"end of (?:the )?day|when tired"),
    "afteract": ("neg", r"after (?:an? )?activit|after exercise|after being active"),
    "passing": ("pos", r"usual diet|bears on"),
}
REGROUP_RE = r"(?<!never )(?<!not )(?:group\w*|organi[sz]ed) by (?:body |organ )?system"
EXAM_SYS = ["Musculoskeletal", "Respiratory", "Cardiovascular", "Extremities", "Neurological"]
CHRONIC_RE = r"one presenting symptom|no new presenting symptom|n/?a when the visit is a|n/?a (?:for|on) (?:a )?chronic follow"
TRAP_LIVE_RATE = {"onset": 0.5}

PATTERNS = ["worse at night", "worse in the morning", "worse in the evening", "worse on weekends",
            "worse in the winter"]
SUBSTANCES = [("drink about two beers on weekends", ["beers"]), ("smoke half a pack of cigarettes a day", ["cigarettes"]),
              ("vape every day", ["vape"]), ("smoke marijuana a few times a week", ["marijuana"]),
              ("have a glass of wine most nights", ["wine"])]
ADJ = ["pretty bad", "really rough", "kind of mild", "honestly terrible", "not too bad", "pretty intense"]
CHARS = {"knee pain": ["aching", "stiffness after sitting", "a grinding feeling"],
         "lower back pain": ["dull ache", "tightness", "sharp twinges when bending"],
         "headaches": ["pressure", "band-like tightness", "throbbing"],
         "cough": ["dry", "hacking", "tickle in the throat"],
         "shoulder pain": ["deep ache", "catching when lifting", "weakness reaching overhead"],
         "heartburn": ["burning", "sour taste", "pressure behind the breastbone"],
         "ankle swelling": ["puffiness", "tightness of the skin", "a dull ache"],
         "sore throat": ["scratchy", "painful swallowing", "raw feeling"],
         "wrist pain": ["tingling in the fingers", "numbness in the thumb", "aching"],
         "dizziness": ["spinning", "lightheadedness", "unsteadiness when turning"]}
RADIATE_TO = {"knee pain": "down the shin", "lower back pain": "down the left leg", "headaches": "into the neck",
              "cough": "into the back", "shoulder pain": "down the arm", "heartburn": "up into the throat",
              "ankle swelling": "up the calf", "sore throat": "into the ears", "wrist pain": "up the forearm",
              "dizziness": "into the neck"}
CC_SYS = {"knee pain": ("Musculoskeletal", "knee"), "lower back pain": ("Musculoskeletal", "back"),
          "headaches": ("Neurological", "headaches"), "cough": ("Respiratory", "cough"),
          "shoulder pain": ("Musculoskeletal", "shoulder"), "heartburn": ("Gastrointestinal", "heartburn"),
          "ankle swelling": ("Musculoskeletal", "ankle"), "sore throat": ("HENT", "throat"),
          "wrist pain": ("Musculoskeletal", "wrist"), "dizziness": ("Neurological", "dizziness")}
SYS_ALIAS = {"Constitutional": "General", "Respiratory": "Pulmonary", "Neurological": "Neuro", "Gastrointestinal": "GI",
             "Musculoskeletal": "MSK", "HENT": "ENT", "Cardiovascular": "Cardiac", "Genitourinary": "GU"}
EXAMS = [("tenderness to palpation over the {loc}", ["palpation"]), ("full range of motion with pain at the end", ["range of motion"]),
         ("no swelling or redness", ["redness"]), ("mild swelling of the {loc}", ["mild swelling"]),
         ("normal strength", ["normal strength"]), ("lungs clear to auscultation", ["auscultation"]),
         ("regular heart rhythm", ["heart rhythm"])]
CHRONIC_EXAMS = [("lungs clear to auscultation", ["auscultation"]), ("regular heart rhythm", ["heart rhythm"]),
                 ("no swelling of the legs", ["swelling of the legs"]), ("normal sensation in both feet", ["sensation"])]
TESTS = [("x-ray shows no fracture", ["x ray", "fracture"]), ("blood work shows a hemoglobin a1c of 7.2", ["a1c", "7 2"]),
         ("the strep test came back negative", ["strep", "negative"]), ("urinalysis is normal", ["urinalysis", "normal"])]
ACTS = [("go to yoga twice a week", ["yoga"]), ("walk the dog every morning", ["dog"]),
        ("play pickleball on weekends", ["pickleball"]), ("work in my garden", ["garden"])]
ROS_NEG = [("Constitutional", "fever"), ("Respiratory", "shortness of breath"), ("Neurological", "numbness"),
           ("Gastrointestinal", "nausea"), ("Cardiovascular", "chest pain"), ("Genitourinary", "painful urination")]
REASONS = ["given the {why}", "because of the {why}", "which fits with the {why}"]
MED_CONDS = [("lisinopril", "high blood pressure"), ("amlodipine", "high blood pressure"), ("metformin", "type 2 diabetes"),
             ("atorvastatin", "high cholesterol"), ("levothyroxine", "hypothyroidism"), ("sertraline", "depression"),
             ("omeprazole", "acid reflux"), ("albuterol", "asthma")]
ONSET_EVENTS = [("after moving furniture", "moving furniture"), ("while i was raking leaves", "raking leaves"),
                ("after a long flight", None), ("out of nowhere one morning", None), ("after a cold", None),
                ("after shoveling snow", "shoveling snow")]
HOMES = [("i live alone with my two cats", ["cats"], "true"), ("i live with my daughter and her two kids", ["daughter"], "trap"),
         ("there's mold in our basement that we can't get rid of", ["mold"], "true")]
DIETS = [("i eat a lot of fast food", ["fast food"]), ("i eat a lot of salty snacks", ["salty"]),
         ("i've cut out sugar and most carbs", ["sugar"])]
NUM = ["two", "three", "four", "five", "six", "seven", "eight"]


def _fact(key, kind, text, kw, item="", **kx):
    return {"key": key, "kind": kind, "text": text, "kw": kw, "item": item, **kx}


def _cc_phrase(comp: str, loc: str) -> tuple[str, list[str]]:
    """The 2-6 word primary-complaint phrase with its side ('right knee pain'), and its keywords."""
    head = comp.split()[0]
    if loc.split()[-1] == head and len(loc.split()) > 1:
        phrase = f"{loc} {' '.join(comp.split()[1:])}".strip()
        side = loc.split()[0]
        return phrase, [head, side] if side in ("left", "right") else [head]
    return comp, [comp.split()[-1] if comp.split()[-1] not in ("pain",) else head]


def _cond_kw(c: str) -> str:
    return c.split()[-1]


def gen_visit(rng: random.Random, P: dict) -> Generated:
    """T06-like visit: acute (60%, one presenting symptom) or chronic follow-up (40%, all symptom fields N/A)."""
    lines: list[tuple[str, str, list]] = []
    sex = rng.choice(["female", "male"])
    age = rng.randint(19, 84)
    name = rng.choice(FIRST).lower()
    chronic = rng.random() < float(P.get("p_chronic", 0.4))
    spouse = "husband" if sex == "female" else "wife"
    xr = random.Random(f"visit-v3:{sex}:{age}:{name}:{chronic}")

    def say(sp, text, fx=()):
        lines.append((sp, text, list(fx)))

    say("doctor", f"hi {name} , how are you doing today ?")
    if rng.random() < 0.8:
        say("doctor", f"and you're {age} now , right ?", [_fact("age", "true", str(age), [str(age)])])
        say("patient", "yes .")
    if rng.random() < 0.75:
        say("doctor", f"{'ma am' if sex == 'female' else 'sir'} , let's go through it .",
            [_fact("sex", "true", sex, [sex])])
    seen, pairs = set(), []
    for m, c in rng.sample(MED_CONDS, len(MED_CONDS)):
        if c not in seen:
            seen.add(c)
            pairs.append((m, c))
    if chronic:
        pairs = pairs[:rng.randint(2, 3)]
    else:
        pairs = pairs[:rng.randint(1, 3)] if rng.random() < 0.8 else []
    meds, conds = [m for m, _ in pairs], [c for _, c in pairs]
    phrase_avoid: list[str] = []
    exam_list: list[tuple[str, list]] = []
    loc = "chest"
    if chronic:
        c0 = conds[0]
        phrase = f"Follow-up of {c0}"
        phrase_kw = ["follow", _cond_kw(c0)]
        phrase_avoid = [_cond_kw(c) for c in conds[1:] if _cond_kw(c) != _cond_kw(c0)]
        say("patient", f"i'm doing okay . i'm here for a follow-up on my {c0} and my {conds[1]} .",
            [_fact("reason", "true", phrase, phrase_kw), _fact("visit_reason", "true", phrase, phrase_kw),
             _fact("reason", "trap", "Follow up", ["follow"], sub="bare"),
             _fact("reason", "trap", "Follow-up of " + ", ".join(conds), ["follow"], sub="list")])
        for i, (m, c) in enumerate(pairs):
            y = rng.randint(2, 15)
            say("doctor", f"how has your {c} been ?")
            fx = [_fact("pmh", "true", f"{c} on {m}", [_cond_kw(c)]), _fact("drugs", "trap", m, [m])]
            if i == 0:
                fx += [_fact("onset", "trap", f"diagnosed {y} years ago", [f"{y} years"], sub="chronic"),
                       _fact("duration", "trap", f"{y} years", [f"{y} years"], sub="chronic"),
                       _fact("temporal", "trap", f"chronic, for {y} years", ["years"], sub="chronic")]
            say("patient", f"pretty steady . i was diagnosed about {y} years ago and i take {m} for it .", fx)
        if rng.random() < 0.6:
            say("patient", "overall i feel fine , maybe a little tired by the evening .",
                [_fact("character", "trap", "a little tired", ["tired"], sub="chronic")])
        if rng.random() < 0.5:
            say("patient", f"it gets worse when i forget my {meds[0]} .",
                [_fact("factors", "trap", f"worse when missing {meds[0]}", [meds[0]], sub="chronic")])
        if rng.random() < 0.3:
            say("patient", "my readings are all over the place , it comes and goes .",
                [_fact("temporal", "trap", "comes and goes", ["comes and goes"], sub="constant")])
        worse = None
        onset_act = None
    else:
        comp, loc, dx = rng.choice(COMPLAINTS)
        phrase, phrase_kw = _cc_phrase(comp, loc)
        weeks = rng.randint(1, 8)
        ev_txt, onset_act = rng.choice(ONSET_EVENTS)
        say("patient", f"i'm okay . i'm here because of this {comp} .",
            [_fact("reason", "true", phrase, phrase_kw), _fact("visit_reason", "true", phrase, phrase_kw),
             _fact("reason", "trap", comp, phrase_kw[:1], sub="noside")])
        say("doctor", "when did it start ?")
        fx = [_fact("onset", "true", f"{weeks} weeks ago", [f"{weeks} weeks ago"]),
              _fact("onset", "trap", ev_txt, ev_txt.split()[-2:], sub="mechanism"),
              _fact("duration", "true", f"{weeks} weeks", [f"{weeks} weeks"]),
              _fact("temporal", "trap", ev_txt, ev_txt.split()[-2:], sub="onset")]
        if onset_act:
            fx.append(_fact("activities", "trap", onset_act, onset_act.split()[-1:], sub="mechanism"))
        say("patient", f"about {weeks} weeks ago , {ev_txt} .", fx)
        if CC_SYS[comp][0] == "Musculoskeletal" and xr.random() < 0.5:
            say("doctor", "did you fall or feel a pop when it happened ?")
            say("patient", "no , no fall and no pop .",
                [_fact("ros", "trap", "Denies fall or pop", ["pop"], sub="mechanism", sys="Musculoskeletal",
                       verb="Denies", sym="fall or pop")])
        say("doctor", f"where is the {comp} exactly ?")
        say("patient", f"mostly in my {loc} .", [_fact("location", "true", loc, loc.split()[-1:])])
        feats = rng.sample(CHARS[comp], 2 if rng.random() < 0.5 else 3)
        say("doctor", "how would you describe it ?")
        say("patient", f"it's like a {feats[0]} , " + " , and ".join(f"with {f}" for f in feats[1:]) + " .",
            [_fact("character", "true", f, f.split()[-1:]) for f in feats])
        worse = rng.choice(["climbing stairs", "lying down", "lifting", "talking for a long time", "standing all day"])
        better = rng.choice(["rest", "ice", "ibuprofen", "a heating pad"])
        say("doctor", "anything make it better or worse ?")
        say("patient", f"{worse} makes it worse , and {better} helps a bit .",
            [_fact("factors", "true", f"worse with {worse}", worse.split()[:1]),
             _fact("factors", "true", f"better with {better}", better.split()[-1:]),
             _fact("temporal", "trap", f"worse with {worse}", worse.split()[:1], sub="activity")])
        r = rng.random()
        if r < 0.4:
            say("patient", rng.choice(["it's pretty much constant .", "it comes and goes ."]),
                [_fact("temporal", "trap", "constant / comes and goes", ["constant"], sub="constant")])
        elif r < 0.75:
            say("patient", "and it's been getting worse over the last couple of weeks .",
                [_fact("temporal", "trap", "getting worse", ["worse"], sub="worsening")])
        say("doctor", "how bad is it ?")
        adj = rng.choice(ADJ)
        sev = rng.randint(3, 9) if rng.random() < 0.35 else None
        say("patient", f"it's {adj} ." + (f" if i had to put a number on it , maybe a {sev} out of 10 ." if sev else ""),
            [_fact("severity", "trap", adj, adj.split()[-1:])] + ([_fact("severity", "true", f"{sev}/10", [f"{sev}"])] if sev else []))
        x3 = xr.random()
        if x3 < 0.2:
            say("patient", "it's usually worse at the end of the day when i'm tired .",
                [_fact("temporal", "trap", "worse at the end of the day", ["end", "day"], sub="endofday")])
        elif x3 < 0.4:
            say("patient", "it flares up after i've been active .",
                [_fact("temporal", "trap", "worse after activity", ["active"], sub="afteract")])
        if rng.random() < 0.12:
            pat = rng.choice(PATTERNS)
            say("patient", f"it's definitely {pat} .", [_fact("temporal", "true", pat, pat.split()[-2:])])
        r = rng.random()
        if r < 0.15:
            to = RADIATE_TO[comp]
            say("patient", f"sometimes it shoots {to} .", [_fact("radiation", "true", to, to.split()[-1:])])
        elif r < 0.5:
            say("doctor", "does it spread anywhere ?")
            say("patient", "no , it stays right there .", [_fact("radiation", "trap", "does not spread", ["spread", "stays"])])
        if pairs:
            say("doctor", "what medicines are you on ?")
            say("patient", "i take " + " and ".join(f"{m} for my {c}" for m, c in pairs) + " .",
                [_fact("pmh", "true", f"{c} on {m}", [_cond_kw(c)]) for m, c in pairs]
                + [_fact("drugs", "trap", m, [m]) for m in meds])
            phrase_avoid = [_cond_kw(c) for c in conds]
        elif rng.random() < 0.8:
            say("doctor", "any problems like this before ?")
            say("patient", "no , i've never had any problems like this , no prior injuries .",
                [_fact("pmh", "trap", "no prior injuries", ["prior"], sub="absence"),
                 _fact("ros", "trap", "Denies prior injuries", ["prior"], sub="history", sys="Musculoskeletal",
                       verb="Denies", sym="prior injuries")])
        if rng.random() < 0.3:
            say("patient", f"i also took some tylenol for the {comp} last week .", [_fact("drugs", "trap", "tylenol", ["tylenol"])])
        if rng.random() < 0.4:
            say("patient", "i haven't been able to pick up my grandkids since .",
                [_fact("activities", "trap", "unable to pick up grandkids", ["grandkids"], sub="restriction")])
    if rng.random() < 0.3:
        s, kw = rng.choice(SUBSTANCES)
        say("doctor", "do you drink or smoke ?")
        say("patient", f"well , i {s} .", [_fact("drugs", "true", s, kw)])
    if rng.random() < 0.3:
        sg = rng.choice(SURGERIES)
        say("patient", f"i had a {sg} years ago .", [_fact("surgical", "true", sg, sg.split()[-1:])])
    if rng.random() < 0.35:
        fm = rng.choice(FAMILY)
        say("patient", f"my {rng.choice(['mother', 'father', 'brother'])} has {fm} .", [_fact("family", "true", fm, fm.split()[-1:])])
    if rng.random() < 0.5:
        job = rng.choice(JOBS)
        say("doctor", "what do you do for work ?")
        say("patient", f"i'm a {job} .", [_fact("employment", "true", job, job.split()[-1:])])
    if rng.random() < 0.35:
        rj = rng.choice(JOBS)
        say("patient", f"my {rng.choice(['daughter', 'son', spouse])} is a {rj} , "
                       f"so i get a lot of advice .", [_fact("employment", "trap", rj, rj.split()[-1:])])
    if rng.random() < 0.45:
        a, kw = rng.choice(ACTS)
        say("patient", f"i usually {a} .", [_fact("activities", "true", a, kw)])
    if rng.random() < 0.42:
        h, kw, hk = rng.choice(HOMES)
        say("patient", f"{h} .", [_fact("home", hk, h.replace("i live", "lives").replace("we can't", "they can't"), kw,
                                        **({"sub": "liveswith"} if hk == "trap" else {}))])
    if rng.random() < 0.35:
        say("patient", f"my {spouse} drove me here today .", [_fact("home", "trap", f"{spouse} drove patient", [spouse], sub="spouse")])
    if rng.random() < 0.2:
        d, kw = rng.choice(DIETS)
        say("patient", f"{d} .", [_fact("eating", "true", d.replace("i eat", "eats").replace("i've cut", "has cut"), kw)])
    if rng.random() < 0.3:
        say("patient", "i grabbed a sandwich on the way here .", [_fact("eating", "trap", "had a sandwich", ["sandwich"], sub="passing")])
    ros_facts = []
    if not chronic:
        sysn, kw = CC_SYS[comp]
        ros_facts.append((sysn, "Reports", comp, kw))
    for sysn, sym in rng.sample(ROS_NEG, rng.randint(1, 3)):
        say("doctor", f"any {sym} ?")
        say("patient", rng.choice([f"no , no {sym} .", f"no {sym} , no .", f"nope , no {sym} at all ."]), [_fact("ros", "true", f"{sysn}: Denies {sym}", [sysn.lower(), sym.split()[-1]],
                                                     sys=sysn, verb="Denies", sym=sym)])
    if rng.random() < (0.8 if chronic else 0.6):
        bp = f"{rng.randint(112, 150)}/{rng.randint(70, 94)}"
        hr = rng.randint(60, 98)
        say("doctor", f"your blood pressure is {bp.replace('/', ' over ')} and your heart rate is {hr} .",
            [_fact("vitals", "true", f"BP {bp}", bp.split("/")[:1]), _fact("vitals", "true", f"HR {hr}", [str(hr)])])
    pool = CHRONIC_EXAMS if chronic else EXAMS
    exam_list = [(e.format(loc=loc), k) for e, k in rng.sample(pool, rng.randint(2, 3))]
    say("doctor", "on exam i see " + " , ".join(e for e, _ in exam_list) + " .",
        [_fact("exam", "true", e, e.split()[-2:]) for e, _ in exam_list]
        + [_fact("ros", "trap", e, k, sub="exam") for e, k in exam_list])
    if rng.random() < (0.7 if chronic else 0.5):
        t, kw = rng.choice(TESTS)
        say("doctor", f"and your {t} .", [_fact("objective", "true", t, kw)])
    if chronic:
        for i, (m, c) in enumerate(pairs):
            st = rng.choice(["well controlled", "stable", "not well controlled"])
            evd = rng.choice(["readings", "labs", "symptoms"])
            say("doctor", f"your {c} is {st} , given the {evd} , so keep the {m} going .",
                [_fact("diagnosis", "true", f"{c}, {st}", [_cond_kw(c), st.split()[-1]]),
                 _fact("diagnosis", "true", f"given the {evd}", [evd])] if i == 0 else
                [_fact("diagnosis", "true", f"{c}, {st}", [_cond_kw(c), st.split()[-1]])])
        fu = rng.choice([3, 6])
        say("doctor", f"we'll continue the {meds[0]} , repeat labs before the next visit , and follow up in {fu} months .",
            [_fact("plan", "true", f"continue {meds[0]}", [meds[0]]), _fact("plan", "true", "repeat labs", ["labs"]),
             _fact("plan", "true", f"follow up in {fu} months", [f"{fu} months"])])
    else:
        why = rng.choice([f"{feats[0]}", f"pain with {worse}", "exam findings", "story you're describing"])
        reason = rng.choice(REASONS).format(why=why)
        say("doctor", f"so i think this is {dx} , {reason} .",
            [_fact("diagnosis", "true", dx, dx.split()[-1:]), _fact("diagnosis", "true", reason, why.split()[-1:])])
        if conds and rng.random() < 0.5:
            say("doctor", f"your {conds[0]} looks well controlled , keep the {meds[0]} going .",
                [_fact("diagnosis", "true", f"{conds[0]}, well controlled", ["controlled"])])
        rx = rng.choice(NEW_DRUGS)
        plan = [_fact("plan", "true", rx, rx.split()[:2])]
        txt = f"i'm going to start {rx}"
        if rng.random() < 0.6:
            plan.append(_fact("plan", "true", "physical therapy referral", ["physical therapy"]))
            txt += " and send you to physical therapy"
        if rng.random() < 0.4:
            plan.append(_fact("plan", "true", "avoid heavy lifting", ["heavy lifting"]))
            plan.append(_fact("activities", "trap", "avoid heavy lifting", ["lifting"], sub="restriction"))
            txt += " , try to avoid heavy lifting for now"
        fu = rng.choice([2, 4, 6])
        plan.append(_fact("plan", "true", f"follow up in {fu} weeks", [f"{fu} weeks"]))
        say("doctor", txt + f" , and let's follow up in {fu} weeks .", plan)
    say("patient", "sounds good , thanks .")

    items, out, facts = [], [], []
    for i, (sp, text, fx) in enumerate(lines):
        line = f"[{sp}] {text}"
        iid = f"L{i + 1}"
        items.append((iid, line))
        out.append(line)
        for f in fx:
            facts.append({**f, "item": iid, "line": line})
    for sysn, verb, sym, kw in ros_facts:
        ln = next(f for f in facts if f["key"] == "reason" and f["kind"] == "true")
        facts.append(_fact("ros", "true", f"{sysn}: {verb} {sym}", [sysn.lower(), kw], item=ln["item"], line=ln["line"],
                           sys=sysn, verb=verb, sym=sym))
    lo, hi = P.get("n_fields_visit", (14, 22))
    keys = list(VISIT_FIELDS)
    must = ["drugs", "severity", "temporal", "onset", "reason", "ros", "diagnosis", "plan"]
    rest = [k for k in keys if k not in must]
    chosen = set(must) | set(rng.sample(rest, max(0, min(len(rest), rng.randint(lo, hi) - len(must)))))
    fields = []
    for k in keys:
        if k not in chosen:
            continue
        name, kind = VISIT_FIELDS[k]
        tf = [f for f in facts if f["key"] == k and f["kind"] == "true"]
        fd = {"key": k, "name": name, "kind": kind, "gold": "; ".join(f["text"] for f in tf) if tf else "N/A"}
        if kind == "phrase":
            fd["avoid"] = phrase_avoid
        fields.append(fd)
    desc = ("You are a clinician. Read the doctor-patient dialogue in `context` and produce the structured clinical "
            "note it supports.")
    g = _finish("visit", "\n".join(out), items, facts, fields, desc)
    g.hidden["chronic"] = int(chronic)
    g.hidden["visit_type"] = "chronic follow-up" if chronic else "new complaint"
    return g


TUTOR_FIELDS = ["Evaluation of the student response", "Repeated mistake", "Mistake identification",
                "Mistake rectification", "Sub-question Transition", "Mistake rectification type",
                "Mistake identification type"]
T04_SPEC = ("Field values: \"Evaluation of the student response\" is one of \"Correct response\", \"Incorrect response\", "
            "\"Partially correct response\"; \"Mistake identification type\" and \"Mistake rectification type\" are each "
            "\"Direct\" or \"Indirect\" (N/A when there is no mistake); \"Repeated mistake\" is \"Yes\" or \"No\".")
SYL = ["ka", "lo", "mi", "ren", "tu", "va", "sel", "dor", "pha", "quin", "zor", "bel", "nyx", "tor", "eli", "gan"]
PROPS = {"color": ["amber", "violet", "teal", "crimson", "ivory", "ochre"],
         "habitat": ["tidal pools", "high plateaus", "cave lakes", "salt flats", "cloud forests", "glacier edges"],
         "diet": ["lichen", "krill", "seeds", "fungi", "nectar", "algae"],
         "cycle": ["a 3-day cycle", "a 9-day cycle", "a 12-day cycle", "a 20-day cycle", "a 40-day cycle", "a 2-day cycle"]}
VAGUE = {"amber": "a warm color", "violet": "a purplish color", "teal": "a bluish color", "crimson": "a reddish color",
         "ivory": "a pale color", "ochre": "an earthy color", "tidal pools": "somewhere wet", "high plateaus": "up high",
         "cave lakes": "somewhere dark", "salt flats": "somewhere dry", "cloud forests": "in the forest",
         "glacier edges": "somewhere cold", "lichen": "plants", "krill": "small animals", "seeds": "plant parts",
         "fungi": "whatever grows", "nectar": "something sweet", "algae": "water plants", "a 3-day cycle": "a short cycle",
         "a 9-day cycle": "a cycle of some days", "a 12-day cycle": "a cycle of under two weeks",
         "a 20-day cycle": "a longish cycle", "a 40-day cycle": "a long cycle", "a 2-day cycle": "a very short cycle"}
AMBIG_FORMS = ("question", "vague", "hedged", "restate")
ACCEPT = ["Exactly!", "Yes, that's right.", "Great, that's correct!"]
SOFT = ["You're on the right track, but", "That's a good start! However,", "That's true, but"]
REJECT = ["Not quite -", "Actually, that's not right -", "That's a common misconception -"]
GUIDE = ["Think about", "Look back at what the problem says about", "Reconsider"]


def _word(rng):
    return (rng.choice(SYL) + rng.choice(SYL) + rng.choice(["", "a", "on", "is"])).capitalize()


def gen_tutor(rng: random.Random, P: dict) -> Generated:
    """T04-like tutoring dialogue. Repeated = Yes iff the last answer is wrong and the previous one was not accepted."""
    species, seen = [], set()
    while len(species) < 4:
        w = _word(rng)
        if w not in seen:
            seen.add(w)
            species.append(w)
    keys = rng.sample(list(PROPS), 2)
    table = {s: {k: v for k, v in zip(keys, vals, strict=False)} for s, vals in
             zip(species, zip(*[rng.sample(PROPS[k], 4) for k in keys], strict=False), strict=False)}
    intro = "; ".join(f"the {s} has " + " and ".join(f"{k} {v}" for k, v in table[s].items()) for s in species)
    subs = rng.sample(species, rng.randint(3, 4))
    cur = rng.randint(1, len(subs) - 1) if rng.random() < 0.85 else 0
    lines: list[tuple[str, str]] = []
    turns: list[dict] = []

    def answer(s, status):
        """(text, attribute at fault or None, form), including ambiguous forms (question-phrased, hedged, generic)."""
        (ka, va), (kb, vb) = list(table[s].items())
        x = rng.random()
        if status == "correct":
            if x < 0.35:
                return f"Is it that the {s} has {ka} {va} and {kb} {vb}?", None, "question"
            return f"The {s} has {ka} {va} and {kb} {vb}.", None, ""
        if status == "partial":
            if x < 0.35:
                return f"The {s} has {ka} {va}, I think.", (kb, vb), "omit"
            if x < 0.7:
                return f"The {s} has {VAGUE[va]} and {kb} {vb}.", (ka, va), "vague"
            return f"Maybe the {s} has {ka} {va}? And I guess {VAGUE[vb]} for the {kb}.", (kb, vb), "hedged"
        if x < 0.35:
            return f"The {s} has a {ka} and a {kb} that suit the way it lives.", (ka, va), "restate"
        other = rng.choice([x for x in species if x != s])
        return f"The {s} has {ka} {table[other][ka]} and {kb} {vb}.", (ka, va), "wrong"

    def ask(s):
        return f"what are the {keys[0]} and {keys[1]} of the {s}?"

    lines.append(("Student", f"Q. Researchers catalogued four species: {intro}. Describe the {' and '.join(keys)} of each "
                             f"species the tutor asks about."))
    turns.append({"status": "none", "subq": 0, "reaction": "none"})
    lines.append(("Tutor", f"Let's break the problem into subproblems. First: {ask(subs[0])}"))
    for qi in range(cur + 1):
        s = subs[qi]
        is_cur = qi == cur
        n_att = rng.choices([1, 2, 3], weights=[65, 25, 10] if is_cur else [30, 50, 20])[0]
        n_prior = (n_att - 1) if is_cur else n_att
        for k in range(n_prior):
            last_on_subq = (not is_cur) and k == n_prior - 1
            st = rng.choices(["correct", "partial", "incorrect"], weights=[75, 15, 10])[0] if last_on_subq else \
                rng.choice(["partial", "incorrect"])
            text, key, _m = answer(s, st)
            lines.append(("Student", text))
            turns.append({"status": st, "subq": qi + 1, "reaction": "accept" if st == "correct" else
                          ("soft" if st == "partial" else "reject")})
            (ka, va), (kb, vb) = list(table[s].items())
            if st == "correct":
                lines.append(("Tutor", f"{rng.choice(ACCEPT)} Let's move on to the next subproblem: {ask(subs[qi + 1])}"))
                continue
            lead = rng.choice(SOFT if st == "partial" else REJECT)
            if last_on_subq:
                lines.append(("Tutor", f"{lead} the {s} has {ka} {va} and {kb} {vb}. Next: {ask(subs[qi + 1])}"))
            else:
                lines.append(("Tutor", f"{lead} {rng.choice(GUIDE).lower()} the {key[0]} of the {s}. Can you try again?"))
    s = subs[cur]
    status = rng.choices(["correct", "partial", "incorrect"], weights=[23, 39, 38])[0]
    last, key, mk = answer(s, status)
    lines.append(("Student", last))
    turns.append({"status": status, "subq": cur + 1, "reaction": "last"})
    (ka, va), (kb, vb) = list(table[s].items())
    nxt = subs[cur + 1] if cur + 1 < len(subs) else None
    answers = [t for t in turns[:-1] if t["status"] != "none"]
    prev_bad = bool(answers) and answers[-1]["status"] != "correct"
    any_bad = any(t["status"] != "correct" for t in answers)
    repeated = "Yes" if (status != "correct" and prev_bad) else "No"
    typ = "Direct" if repeated == "Yes" else "Indirect"
    if status == "correct":
        gold = {"Evaluation of the student response": "Correct response", "Repeated mistake": "No",
                "Mistake identification": "N/A", "Mistake rectification": "N/A",
                "Sub-question Transition": f"Move on to the {nxt}" if nxt else "Summarize the four species",
                "Mistake rectification type": "N/A", "Mistake identification type": "N/A"}
        kws = {"Sub-question Transition": [nxt.lower()] if nxt else ["summarize"]}
    else:
        if mk == "omit":
            mi = f"The student's response does not give the {key[0]}; the {s} has {key[0]} {key[1]}."
        elif mk in ("vague", "hedged"):
            mi = f"The student's {key[0]} is too vague; the {s} has {key[0]} {key[1]}."
        elif mk == "restate":
            mi = f"The student only restates the question without giving the {key[0]}; the {s} has {key[0]} {key[1]}."
        else:
            mi = f"The student incorrectly states the {key[0]} of the {s}; in fact it has {key[0]} {key[1]}."
        gold = {"Evaluation of the student response": "Partially correct response" if status == "partial" else "Incorrect response",
                "Repeated mistake": repeated, "Mistake identification": mi,
                "Mistake rectification": (f"The {s} has {ka} {va} and {kb} {vb}. Can you now describe the next one?"
                                          if typ == "Direct" else
                                          f"Think about the {key[0]} of the {s}: what does the problem say about it?"),
                "Sub-question Transition": "N/A", "Mistake rectification type": typ, "Mistake identification type": typ}
        kws = {"Mistake identification": [key[0], key[1]],
               "Mistake rectification": [key[1]] if typ == "Direct" else ["?", key[0]]}
    items, out = [], []
    for i, (sp, text) in enumerate(lines):
        line = f"{sp}: {text}"
        iid = f"L{i + 1}"
        items.append((iid, line))
        out.append(line)
    stu = [i for i, (sp, _t) in enumerate(lines) if sp == "Student"]
    for t, li in zip(turns, stu, strict=False):
        t["text"] = " ".join(lines[li][1].split()[:12])
    hidden_t = {"status": status, "repeated": repeated, "type": typ, "gold": gold, "kws": kws, "species": s,
                "attrs": [[ka, va], [kb, vb]], "key": list(key) if key else None, "mistake": mk, "next": nxt,
                "ambig": int(mk in AMBIG_FORMS),
                "last_item": f"L{len(lines)}", "last_text": last, "turns": turns, "prev_bad": int(prev_bad), "any_bad": int(any_bad)}
    fields = [{"key": n, "name": n, "kind": "tutor", "gold": gold[n]} for n in TUTOR_FIELDS]
    desc = ("You are an expert tutor. Read the tutoring dialogue in `context` and assess the student's most recent "
            "response.")
    g = _finish("tutor", "\n".join(out), items, [], fields, desc, spec=T04_SPEC)
    g.hidden["tutor"] = hidden_t
    return g


RECIPE_FIELDS = [("struct", "Selection of Precursors [level] Structural Considerations", 0.24),
                 ("handling", "Selection of Precursors [level] Handling Precursor Reactivity", 0.48),
                 ("phys", "Selection of Precursors [level] Physical and Chemical Properties of Precursors", 0.36),
                 ("temp", "Synthesis Conditions [level] Temperature and Heating Method", 0.78),
                 ("atm", "Synthesis Conditions [level] Atmosphere", 0.22),
                 ("dur", "Synthesis Conditions [level] Duration", 0.14)]
T03_DESC = ("You are a materials scientist. Read the synthesis procedure in `context` and explain the rationale behind "
            "each experimental choice.")
T03_SPEC = ("Fill a field only when the procedure states, or clearly implies, the reason for that choice; if it gives no "
            "reason, write N/A for that field.")

FAMILIES = {
    "solid_state": dict(
        targets=[("BaTiO3", ["BaCO3 (99.9%)", "TiO2 (anatase)"]), ("SrFeO3", ["SrCO3", "Fe2O3"]),
                 ("CaMnO3", ["CaCO3", "MnO2"]), ("LaCoO3", ["La2O3 (pre-dried)", "Co3O4"])],
        steps=["Weigh the precursors in the stoichiometric ratio.",
               ("Ball-mill the powders in ethanol for {tm} h.", "phys", "Ball-mill the powders in ethanol for {tm} h to reduce the particle size."),
               "Dry the slurry at {Td} °C overnight.",
               ("Calcine the powder at {T} °C for {t} h in air with a heating rate of 5 °C/min.", "atm",
                "Calcine the powder at {T} °C for {t} h in air to keep the full oxygen content, heating at 5 °C/min."),
               "Regrind, press into pellets and sinter at {T2} °C for {t2} h."],
        T=(900, 1100), mech=[("{T} °C is high enough for the carbonate to decompose and for solid-state diffusion between "
                              "the oxides to form the single-phase perovskite", [["diffusion"], ["carbonate", "decompos"]])],
        wrong="Drying at {Td} °C removes the ethanol before calcination.", excess=None),
    "li_cathode": dict(
        targets=[("LiNi0.8Co0.1Mn0.1O2", ["LiOH·H2O", "Ni0.8Co0.1Mn0.1(OH)2"]), ("LiCoO2", ["Li2CO3", "Co3O4"]),
                 ("LiNi0.5Mn1.5O4", ["Li2CO3", "NiO", "MnO2"])],
        steps=["Mix the transition-metal precursor with a {x}% excess of the lithium salt in an agate mortar.",
               "Pre-calcine at 500 °C for 5 h.",
               ("Calcine at {T} °C for {t} h under flowing O2.", "atm",
                "Calcine at {T} °C for {t} h under flowing O2 to keep nickel in the 3+ state."),
               ("Cool naturally to room temperature.", "dur", "Cool slowly to room temperature to avoid cracking of the particles.")],
        T=(750, 900), mech=[("{T} °C crystallises the ordered lithium transition-metal oxide while staying below the "
                             "temperature where lithium volatilises and cation mixing grows",
                             [["lithium", "volatil"], ["lithium", "loss"], ["cation mixing"]])],
        wrong="The 500 °C pre-calcination decomposes the lithium salt before the main firing.", excess="lithium"),
    "sulfide": dict(
        targets=[("Li6PS5Cl", ["Li2S", "P2S5", "LiCl"]), ("Li3PS4", ["Li2S", "P2S5"]), ("Na3PS4", ["Na2S", "P2S5"])],
        steps=[("Inside an Ar-filled glovebox weigh the precursors.", "handling",
                "Inside an Ar-filled glovebox weigh the precursors, because the sulfides react with moisture."),
               "Ball-mill at 500 rpm for {tm} h.",
               ("Seal the milled powder in a quartz ampoule under vacuum.", "atm",
                "Seal the milled powder in a quartz ampoule under vacuum to avoid sulfur loss."),
               "Anneal at {T} °C for {t} h, then cool naturally."],
        T=(450, 550), mech=[("{T} °C crystallises the conducting phase from the amorphous milled precursor while staying "
                             "below the temperature where sulfur is lost", [["crystalli"], ["amorphous"], ["sulfur", "loss"]])],
        wrong="Ball-milling at 500 rpm mixes the powders before annealing.", excess=None),
    "hydrothermal": dict(
        targets=[("ZnO nanorods", ["Zn(NO3)2·6H2O", "hexamethylenetetramine"]), ("MoS2 nanosheets", ["Na2MoO4·2H2O", "thiourea"]),
                 ("anatase TiO2 nanoparticles", ["titanium butoxide", "acetic acid"])],
        steps=["Dissolve the precursors in {V} mL deionised water and stir for 30 min.",
               ("Transfer the solution to a Teflon-lined autoclave and heat at {T} °C for {t} h.", "dur",
                "Transfer the solution to a Teflon-lined autoclave and heat at {T} °C for {t} h to complete the growth."),
               "Wash the product with water and ethanol, then dry at {Td} °C."],
        T=(140, 220), mech=[("the sealed autoclave keeps water liquid above its boiling point, so the raised pressure and "
                             "solubility let the crystalline product nucleate at only {T} °C",
                             [["pressure"], ["solubil"], ["autoclave", "boiling"]])],
        wrong="Drying at {Td} °C removes the residual solvent from the product.", excess=None),
    "rapid": dict(
        targets=[("Ni-doped CeO2", ["Ce(NO3)3·6H2O", "Ni(NO3)2·6H2O", "urea"]), ("turbostratic graphene", ["carbon black"]),
                 ("high-entropy alloy nanoparticles", ["mixed metal chlorides on carbon"])],
        steps=["Mix the precursors and load them into a quartz tube.",
               ("Apply a flash Joule-heating pulse that reaches {T} °C within 1 s.", "handling",
                "Apply a flash Joule-heating pulse that reaches {T} °C within 1 s, so the reaction finishes before the precursors segregate."),
               "Let the sample cool under ambient conditions and collect the powder."],
        T=(1000, 2800), mech=[("the rapid heating reaches {T} °C in about a second, so the product nucleates kinetically "
                               "and grain growth and phase separation are suppressed",
                               [["grain growth"], ["rapid", "nucleat"], ["kinetic"], ["phase separation"]])],
        wrong="Cooling under ambient conditions stops the reaction.", excess=None),
    "nitridation": dict(
        targets=[("TiN", ["TiO2 (P25)"]), ("GaN", ["Ga2O3"]), ("Ta3N5", ["Ta2O5"])],
        steps=["Place the oxide powder in an alumina boat.",
               ("Heat to {T} °C under flowing NH3 (200 sccm) for {t} h.", "atm",
                "Heat to {T} °C under flowing NH3 (200 sccm) for {t} h as the nitrogen source."),
               "Cool to room temperature under NH3 at 10 °C/min."],
        T=(700, 1000), mech=[("{T} °C is high enough for ammonia to dissociate into reactive nitrogen that replaces the "
                              "oxygen, yet below where the nitride decomposes",
                              [["ammonia", "dissociat"], ["nh3", "dissociat"], ["ammonia", "decompos"], ["nitride", "decompos"]])],
        wrong="Cooling at 10 °C/min under NH3 prevents cracking.", excess=None),
    "reduction": dict(
        targets=[("oxygen-deficient TiO2-x", ["TiO2 (P25)"]), ("Ni/Al2O3 catalyst", ["Ni(NO3)2-impregnated Al2O3"]),
                 ("WO3-x nanowires", ["WO3 nanowires"])],
        steps=["Load the powder in the center of a tube furnace.", "Dry at {Td} °C for 1 h.",
               ("Heat at {T} °C for {t} h in 5% H2/Ar.", "atm", "Heat at {T} °C for {t} h in 5% H2/Ar to create oxygen vacancies."),
               "Cool to room temperature under Ar."],
        T=(400, 650), mech=[("{T} °C lets hydrogen remove lattice oxygen to reduce the oxide while staying below the "
                             "temperature where the particles sinter", [["vacanc"], ["sinter"], ["reduc", "oxygen"]])],
        wrong="Drying at {Td} °C removes adsorbed water first.", excess=None),
    "sealed_tube": dict(
        targets=[("Bi2Te3", ["Bi shot", "Te lumps"]), ("SnSe", ["Sn granules", "Se shot"]), ("Cu2Se", ["Cu powder", "Se shot"])],
        steps=["Weigh the elements (with a {x}% excess of the chalcogen) in a glovebox.",
               ("Load them into a carbon-coated quartz tube.", "struct",
                "Load them into a carbon-coated quartz tube, cut with a bevel to let the vapour reach the metal."),
               "Seal the tube under 10^-4 Torr.", "Heat to {T} °C over 10 h, hold {t} h, then cool at 2 °C/h.",
               ("Anneal at {Ta} °C for 2 days.", "dur", "Anneal at {Ta} °C for 2 days to homogenise the ingot.")],
        T=(600, 950), mech=[("{T} °C is above the melting points of the elements so the melt reacts homogeneously, while "
                             "the sealed tube holds the volatile chalcogen vapour", [["melting point"], ["vapor pressure"],
                                                                                     ["vapour pressure"], ["melt", "volatil"]])],
        wrong="Cooling at 2 °C/h lets large crystals grow.", excess="chalcogen"),
    "molten_salt": dict(
        targets=[("LaFeO3", ["La(NO3)3·6H2O", "Fe(NO3)3·9H2O"], "NaCl-KCl"), ("SrTiO3", ["SrCO3", "TiO2"], "KCl"),
                 ("Bi4Ti3O12", ["Bi2O3", "TiO2"], "Na2SO4-K2SO4")],
        steps=[("Mix the precursors with {salt} in a 1:1 weight ratio.", "phys",
                "Mix the precursors with {salt} in a 1:1 weight ratio to act as a flux."),
               "Heat at {T} °C for {t} h in an alumina crucible.",
               "Wash out the salt with hot deionised water and dry at {Td} °C."],
        T=(700, 900), mech=[("{T} °C is above the melting point of the salt, so the molten flux dissolves the precursors "
                             "and speeds their diffusion", [["flux"], ["eutectic"], ["molten salt"], ["salt", "melt"]])],
        wrong="Washing with hot water and drying at {Td} °C removes the salt.", excess=None),
    "arc_melt": dict(
        targets=[("TiNiSn", ["Ti rod", "Ni slug", "Sn shot"]), ("CoCrFeNi", ["Co", "Cr", "Fe", "Ni pieces"]),
                 ("Mn3Ga", ["Mn flakes", "Ga pieces"])],
        steps=[("Arc-melt the metals under argon, flipping and remelting 5 times.", "handling",
                "Arc-melt the metals under argon, flipping and remelting 5 times to ensure homogeneity."),
               "Seal the button in an evacuated quartz tube.",
               ("Anneal at {T} °C for {d} days, then quench in water.", "dur",
                "Anneal at {T} °C for {d} days to reach equilibrium, then quench in water.")],
        T=(700, 1000), mech=[("the anneal at {T} °C stays below the solidus but is high enough for atomic diffusion to "
                              "homogenise the cast alloy and form the ordered phase", [["homogeni"], ["order"], ["solidus"]])],
        wrong="Quenching in water freezes in the high-temperature state.", excess=None),
}
PAPER = {
    "struct": [("the precursor's layered framework is retained topotactically in the product", [["topotact"]]),
               ("an isostructural precursor lowers the nucleation barrier of the target", [["isostructur"]]),
               ("the precursor's open channels let the second reagent diffuse in", [["channel"]])],
    "handling": [("keeping the reactants apart prevents a high-melting binary phase that blocks the reaction", [["binary phase"]]),
                 ("adding the reducing agent last avoids premature reduction", [["premature"]]),
                 ("strict stoichiometry avoids an unwanted competing phase", [["competing phase"]])],
    "phys": [("the low melting point of one precursor lets it wet the other powder", [["wet"]]),
             ("the nitrate's high solubility gives an atomically mixed solution", [["atomically mixed"]]),
             ("the fine precursor particles shorten the diffusion distances", [["diffusion distance"]])],
    "atm": [("flowing gas sweeps away the CO2 released by the carbonate", [["sweep"]]),
            ("the low oxygen partial pressure keeps the metal in its lower oxidation state", [["partial pressure"]]),
            ("the reactor vents the hydrogen evolved during heating", [["vent"]])],
    "dur": [("a long hold is needed because the ordering transition is slow", [["slow"]]),
            ("a short hold limits grain growth", [["grain growth"]]),
            ("the hold lets the intermediate phase convert fully", [["intermediate"]])],
}
EXCESS_GOLD = ("the {x}% excess compensates the loss of the volatile {el} during the high-temperature treatment",
               [["compensat"], ["volatil", "loss"]])
CONFAB = {"struct": "The precursors were chosen because their structures are compatible with the target.",
          "handling": "The precursors are stable and easy to handle under the stated conditions.",
          "phys": "The precursors have suitable purity and reactivity for this synthesis.",
          "atm": "The atmosphere prevents unwanted side reactions with air.",
          "dur": "The duration ensures that the reaction goes to completion."}
GENERIC_TEMP = ["The temperature was chosen to ensure complete reaction and uniform heating.",
                "The heating conditions are suitable for this material and give a good-quality product."]


def gen_recipe(rng: random.Random, P: dict) -> Generated:
    fam_name = rng.choice(sorted(FAMILIES))
    fam = FAMILIES[fam_name]
    tg = rng.choice(fam["targets"])
    target, precs = tg[0], tg[1]
    salt = tg[2] if len(tg) > 2 else ""
    lo, hi = fam["T"]
    T = rng.randrange(lo, hi + 1, 10 if hi - lo < 1000 else 50)
    vals = {"T": T, "T2": T + rng.choice([150, 200, 250]), "t": rng.choice([2, 4, 6, 8, 10, 12, 24]),
            "t2": rng.choice([4, 6, 10, 12]), "tm": rng.choice([2, 4, 6, 12]), "Td": rng.choice([60, 80, 100, 120]),
            "Ta": max(200, T - rng.choice([150, 200, 300])), "d": rng.choice([3, 5, 7, 14]), "V": rng.choice([30, 40, 60]),
            "x": rng.choice([3, 5, 8, 10]), "salt": salt}
    excess = fam["excess"] if fam["excess"] and (fam_name == "li_cathode" or rng.random() < 0.5) else None
    steps, cues = [], {}
    for st in fam["steps"]:
        if isinstance(st, tuple):
            plain, slot, cued = st
            if rng.random() < 0.3 and slot not in cues:
                txt = cued.format(**vals)
                cues[slot] = txt
            else:
                txt = plain.format(**vals)
        else:
            txt = st.format(**vals)
            if "excess" in txt and not excess:
                txt = re.sub(r" \(with a \d+% excess of the chalcogen\)| with a \d+% excess of the lithium salt", "", txt)
        steps.append(txt)
    lines = [f"Target Material: {target}", "", "Precursors: "] + [f"- {p}" for p in precs] + ["", "Synthesis Steps: "]
    lines += [f"{i + 1}. {s}" for i, s in enumerate(steps)]
    mech_text, mech_alts = rng.choice(fam["mech"])
    mech_text = mech_text.format(**vals)
    rc = {"family": fam_name, "mech": mech_text[0].upper() + mech_text[1:] + ".", "wrong": fam["wrong"].format(**vals),
          "cues": cues, "excess": None, "gold_text": {}}
    if excess:
        estep = next(s for s in steps if "excess" in s)
        rc["excess"] = {"el": excess, "step": estep, "x": vals["x"]}
    fields = []
    for key, name, rate in RECIPE_FIELDS:
        gold, alts = "N/A", []
        if key == "temp":
            if rng.random() < rate:
                gold, alts = rc["mech"], mech_alts
        elif key == "phys" and excess:
            if rng.random() < 0.6:
                t, alts = EXCESS_GOLD
                gold = t.format(x=vals["x"], el=excess)
                gold = gold[0].upper() + gold[1:] + "."
        else:
            r_ = rate if key != "phys" else 0.34
            if rng.random() < r_:
                t, alts = rng.choice(PAPER[key])
                gold = t[0].upper() + t[1:] + "."
        fields.append({"key": key, "name": name, "kind": "recipe", "gold": gold, "alts": alts})
        rc["gold_text"][key] = gold
    context = "\n".join(lines)
    items = [(f"L{i + 1}", ln) for i, ln in enumerate(lines) if ln.strip()]
    g = _finish("recipe", context, items, [], fields, T03_DESC, spec=T03_SPEC)
    g.hidden["recipe"] = rc
    return g


def _finish(variant, context, items, facts, fields, desc, spec: str = "") -> Generated:
    rp = root_prompt(desc, [f["name"] for f in fields])
    if spec:
        rp = rp.replace("do not invent content.", "do not invent content. " + spec)
    hidden = {"variant": variant, "facts": json.loads(json.dumps(facts)), "fields": json.loads(json.dumps(fields))}
    req = sorted({f["item"] for f in facts if f.get("kind") == "true" and f.get("item")}) or None
    return Generated(context=context, root_prompt=rp, items=items, gold={f["name"]: f["gold"] for f in fields},
                     hidden=hidden, required_ids=req, index_min_len=8)


GENERATORS = {"visit": gen_visit, "tutor": gen_tutor, "recipe": gen_recipe}


def generate(info: dict) -> Generated:
    P = json.loads(info.get("synth_params") or "{}") if isinstance(info.get("synth_params"), str) else \
        dict(info.get("synth_params") or {})
    hint, hint_prob = P.pop("hint", "none"), float(P.pop("hint_prob", 1.0))
    rng = random.Random(f"elb-forms:{info['synth_split']}:{info['synth_seed']}:{json.dumps(P, sort_keys=True)}")
    kind = P.get("kind")
    if not kind:
        if rng.random() < float(P.get("p_recipe", 0.15)):
            kind = "recipe"
        else:
            kind = "tutor" if rng.random() < float(P.get("p_tutor", 0.5)) else "visit"
    g = GENERATORS[kind](rng, P)
    if hint not in ("none", "", None) and rng_for("forms-hint", info["synth_split"], info["synth_seed"]).random() < hint_prob:
        spec = HINTS_BY_KIND[kind]
        if kind == "visit":
            txt = spec.text(hint, [f["name"] for f in g.hidden["fields"]])
        elif kind == "tutor":
            txt = spec.text(hint)
        else:
            txt = spec.text(hint, [f["name"] for f in g.hidden["fields"]] if "fields" in g.hidden else None)
        g.root_prompt = add_hint(g.root_prompt, txt)
        g.hidden["hinted"] = 1
    return g


BARE_NA_RE = r"exactly n/?a|bare (?:token )?n/?a|never .?n/?a \(|not .?n/?a \(reason|only the token"
COMPLETE_RE = r"complete|every |all (?:the )?(?:relevant )?(?:details|findings|facts)|full |in detail|specific"
STRUCT_RE = r"(?:if|when)\b.{0,80}?\bcorrect\b.{0,200}?\bn/?a\b"
LEDGER_RE = r"every student (?:turn|answer)|each student (?:turn|answer)|ledger"
PREV_RULE_RE = r"previous answer|right after|errs? again|again right after|immediately before"
ANY_RULE_RE = r"earlier|already|before|previous"
CUE_RE = (r"only (?:if|when) the procedure (?:itself )?(?:states|gives|specifies)|states the purpose|states,? or clearly "
          r"implies,? the reason|states or clearly implies")


def _field_segs(low: str, fields: list[dict]) -> dict[str, str]:
    """Each field's definition: the instruction text from its name up to the next field name."""
    pos = {}
    for f in fields:
        n = f["name"].lower()
        for pat in (f'"{n}"', f"'{n}'", f"- {n}", n):
            i = low.find(pat)
            if i != -1:
                pos[f["name"]] = (i, len(pat))
                break
    order = sorted(pos.items(), key=lambda kv: kv[1][0])
    segs = {}
    for j, (name, (i, ln)) in enumerate(order):
        end = order[j + 1][1][0] if j + 1 < len(order) else min(len(low), i + 700)
        segs[name] = low[i + ln:end]
    return segs


DRAFT_MIN_FIELDS = 8
DRAFT = {"ros_deny_keep": 0.8, "reason_noside": 0.15, "single_dx": 0.35, "pmh_drop": 0.25, "act_miss": 0.15,
         "act_trap": 0.12, "meds_as_drugs": 0.2, "json_break": 0.15,
         "probe_chronic_as_new": 0.06, "probe_new_as_chronic": 0.03}


def _excluded(sub: str, text: str) -> bool:
    mode, rx = SUB_EXCLUDE[sub]
    if mode == "pos":
        return bool(re.search(rx, text))
    return any(re.search(rx, c) and re.search(_NEG, c) for c in re.split(r"[.;\n(]", text))


class FormReader(FakeSubLM):
    def __init__(self, read: ReadModel | None = None, trap_rate: float = 0.9, obey_rate: float = 0.95,
                 terse_keep: float = 0.5, full_keep: float = 0.92, reason_na: float = 0.35, live_rate: float = 0.65):
        super().__init__(read or ReadModel(budget_chars=80_000, chatty=0.2))
        self.trap_rate, self.obey_rate, self.live_rate = trap_rate, obey_rate, live_rate
        self.terse_keep, self.full_keep, self.reason_na = terse_keep, full_keep, reason_na

    def requested(self, instr: str, fields: list[dict]) -> list[dict]:
        low = instr.lower()
        return [f for f in fields if f["name"].lower() in low]

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        ids = set(self.given(prompt, hidden))
        instr = self.instruction(prompt, hidden)
        low = instr.lower()
        if not ids:
            return "I don't see the record in your message -- please include the text you want me to read."
        if hidden["variant"] == "tutor":
            return self._tutor(prompt, low, hidden, ids, instr)
        want = self.requested(instr, hidden["fields"])
        if hidden["variant"] == "recipe":
            return self._recipe(prompt, low, hidden, want)
        if hidden["variant"] == "visit" and len(want) < 5 and re.search(r'"visit"\s*:', low) and '"chief"' in low:
            return self._visit_probe(prompt, hidden)
        drafter = hidden["variant"] == "visit" and len(want) >= DRAFT_MIN_FIELDS
        facts = [f for f in hidden["facts"] if f["item"] in ids]
        if not want:
            return "Summary: " + " ".join(sorted({f["text"] for f in facts if f["kind"] == "true"})[:12])
        complete = bool(re.search(COMPLETE_RE, low))
        bare = bool(re.search(BARE_NA_RE, low))
        quotes = "quote" in low
        segs = _field_segs(low, want)
        chronic = bool(hidden.get("chronic"))
        out = {}
        for f in want:
            r = rng_for(prompt, f["key"])
            key = f["key"]
            seg = segs.get(f["name"], "")
            text = seg if len(seg.strip(' "\':-\n')) > 3 else low
            tf = [x for x in facts if x["key"] == key and x["kind"] == "true"]
            xf = [x for x in facts if x["key"] == key and x["kind"] == "trap"]
            val, ev = None, None
            if key in SYMPTOM_KEYS and chronic:
                if not (re.search(CHRONIC_RE, low) and r.random() < self.obey_rate) and xf and r.random() < 0.85:
                    val, ev = "; ".join(x["text"] for x in xf), xf[0]
            elif f["kind"] in ("phrase", "sentence"):
                val, ev = self._phrase(r, text, chronic, hidden["facts"], sentence=f["kind"] == "sentence")
            elif f["kind"] == "ros":
                rx = [x for x in hidden["facts"] if x["key"] == "ros" and x["kind"] == "trap"]
                val, ev = self._ros(r, text, tf, [x for x in rx if x.get("sub") == "exam"], complete,
                                    [x for x in rx if x.get("sub") != "exam" and x["item"] in ids],
                                    deny_keep=DRAFT["ros_deny_keep"] if drafter else 1.0)
            elif xf and key in TRAP_DEFS:
                defined = bool(re.search(TRAP_DEFS[key], text))
                live = [x for x in xf if not (_excluded(x["sub"], text) if x.get("sub") else defined)]
                if not live:
                    if r.random() < self.obey_rate:
                        if tf:
                            val, ev = tf[0]["text"], tf[0]
                    elif r.random() < 0.5:
                        val, ev = "; ".join(x["text"] for x in xf), xf[-1]
                    elif tf:
                        val, ev = tf[0]["text"], tf[0]
                elif r.random() < (TRAP_LIVE_RATE.get(key, self.live_rate) if defined else
                                   (TRAP_LIVE_RATE.get(key, self.trap_rate))):
                    pick = (tf + live) if tf and r.random() < 0.5 else live
                    val, ev = "; ".join(x["text"] for x in pick), pick[-1]
                elif tf:
                    val, ev = tf[0]["text"], tf[0]
            elif f["kind"] in ("narr", "trap"):
                keep = self.full_keep if complete else self.terse_keep
                kept = [x for x in tf if r.random() < keep] or (tf[:1] if tf else [])
                if f["kind"] == "trap":
                    kept = tf[:1]
                if kept:
                    val, ev = "; ".join(x["text"] for x in kept), kept[0]
                    if key == "exam" and re.search(REGROUP_RE, text) and r.random() < 0.7:
                        val = "; ".join(f"{r.choice(EXAM_SYS)}: {x['text']}" for x in kept)
            else:
                if tf and r.random() < 0.97:
                    val, ev = tf[0]["text"], tf[0]
            if drafter:
                val = self._draft_terse(rng_for(prompt, key, "draft"), key, val, tf, xf, hidden, chronic)
            if val is None:
                val = "N/A" if bare or r.random() >= self.reason_na else r.choice(
                    ["N/A (not discussed)", "Not mentioned in the dialogue", "None reported", "N/A - not stated"])
            out[f["name"]] = {"value": val, "quote": ev["line"].split("] ", 1)[-1] if ev else ""} if quotes else val
        body = self._render(prompt, low, out)
        if drafter and "json" in low and rng_for(prompt, "jsonbreak").random() < DRAFT["json_break"]:
            body = body.replace('"Review of Systems (ROS)": "', '"Review of Systems (ROS): "', 1)
        return body

    def _draft_terse(self, r, key, val, tf, xf, hidden, chronic):
        """Per-field drafting errors of the sub-LM, calibrated to the real model."""
        facts = hidden["facts"]
        if key in ("visit_reason", "reason") and val and r.random() < DRAFT["reason_noside"]:
            traps = {x.get("sub"): x for x in facts if x["key"] == "reason" and x["kind"] == "trap"}
            t = traps.get("bare") if chronic else traps.get("noside")
            return t["text"] if t else val
        if key == "diagnosis" and len(tf) >= 3 and r.random() < DRAFT["single_dx"]:
            return tf[0]["text"]
        if key == "pmh" and len(tf) >= 2:
            kept = [x for i, x in enumerate(tf) if i == 0 or r.random() >= DRAFT["pmh_drop"]]
            return "; ".join(x["text"] for x in kept)
        if key == "activities":
            if tf and r.random() < DRAFT["act_miss"]:
                return None
            if not tf and xf and r.random() < DRAFT["act_trap"]:
                return xf[0]["text"]
        if key == "drugs" and r.random() < DRAFT["meds_as_drugs"]:
            meds = [x["text"] for x in xf if not x.get("sub")]
            if meds:
                return "; ".join(([val] if val else []) + meds)
        return val

    def _visit_probe(self, prompt, hidden):
        """Probe reply {"visit", "chief", "problems"}, with calibrated misreads."""
        r = rng_for(prompt, "visit-probe")
        facts = hidden["facts"]
        chronic = bool(hidden.get("chronic"))
        gold = next((x for x in facts if x["key"] == "reason" and x["kind"] == "true"), None)
        conds = list(dict.fromkeys(x["text"].split(" on ")[0] for x in facts if x["key"] == "pmh" and x["kind"] == "true"))
        if chronic:
            if r.random() < DRAFT["probe_chronic_as_new"]:
                visit, chief = "new complaint", "fatigue"
            else:
                visit, chief = ("results discussion" if r.random() < 0.05 else "chronic follow-up"), None
            problems = conds
        else:
            visit = "chronic follow-up" if r.random() < DRAFT["probe_new_as_chronic"] else "new complaint"
            noside = next((x["text"] for x in facts if x["key"] == "reason" and x.get("sub") == "noside"), None)
            chief = (noside if noside and r.random() < 0.1 else (gold["text"] if gold else None))
            chief = None if visit != "new complaint" else chief
            problems = ([gold["text"]] if gold else []) + conds
        body = json.dumps({"visit": visit, "chief": chief, "problems": problems}, ensure_ascii=False)
        x = r.random()
        if x < 0.3:
            body = "```json\n" + body + "\n```"
        return ("Here is the JSON:\n" + body) if r.random() < self.read.chatty else body

    def _phrase(self, r, text, chronic, facts, sentence=False):
        gold = next(x for x in facts if x["key"] == "reason" and x["kind"] == "true")
        if sentence and re.search(r"one (?:full )?sentence|own concern|own words", text) and r.random() < self.obey_rate:
            extra = next((x["text"] for x in facts if x["key"] == "onset" and x["kind"] == "trap"), "")
            return (f"{gold['text'][0].upper() + gold['text'][1:]}" + (f", which began {extra}" if extra and not chronic
                    else "") + "; wants it checked and treated."), gold
        traps = {x.get("sub"): x for x in facts if x["key"] == "reason" and x["kind"] == "trap"}
        listed = (gold["text"] + " and " + ", ".join(x["text"] for x in facts if x["key"] == "diagnosis"
                                                     and x["kind"] == "true")) if not chronic else traps["list"]["text"]
        if re.search(r"never (?:a )?bare|say what is followed|what is followed up", text) and r.random() < self.obey_rate:
            return gold["text"], gold
        if chronic and re.search(r"'follow up'", text):
            return "Follow up", gold
        if re.search(r"with (?:its )?side|2-6 word|primary complaint|main complaint", text) and r.random() < self.obey_rate:
            return (gold["text"], gold) if not chronic or r.random() < 0.65 else ("Follow up", gold)
        if re.search(r"plus any follow|every problem|all (?:the )?problems", text) and r.random() < 0.8:
            return listed, gold
        x = r.random()
        if x < 0.3:
            return (traps["bare"]["text"] if chronic else traps["noside"]["text"]), gold
        if x < 0.5:
            return listed, gold
        return gold["text"], gold

    def _ros(self, r, text, tf, exam, complete, other=(), deny_keep=1.0):
        if not re.search(r"never n/?a\b", text) and r.random() < 0.3:
            return r.choice(["None", "N/A", "No symptoms reported"]), None
        std = bool(re.search(r"constitutional|<system>|organ system|system name|by (?:body )?system", text))
        keep = self.full_keep if complete else self.terse_keep
        kept = [x for x in tf if r.random() < keep * (deny_keep if x.get("verb") == "Denies" else 1.0)] or tf[:1]
        const_rule = bool(re.search(r"constitutional\s*(?:=|is|means|covers|:)\s*(?:only\s*)?(?:fever|systemic)", text))
        short_rule = bool(re.search(r"few words|\b\d-\d words|no .{0,30}detail", text))
        kept = [dict(x) for x in kept]
        for x in kept:
            if x["verb"] == "Reports":
                if not const_rule and r.random() < 0.25:
                    x["sys"] = "Constitutional"
                if not short_rule and r.random() < 0.4:
                    x["sym"] = x["sym"] + " that started a few weeks ago and gets worse with use"
        for x in other:
            if not _excluded(x["sub"], text) and r.random() < 0.5:
                kept.append(x)
        parts = [f"{x['sys'] if std or r.random() >= 0.6 else SYS_ALIAS.get(x['sys'], x['sys'])}: {x['verb']} {x['sym']}"
                 for x in kept]
        no_exam = bool(re.search(r"never exam|not exam|no exam|subjective", text)) or any(
            re.search(r"exam", c) and re.search(_NEG + r"|\bno\b", c) for c in re.split(r"[.;\n(]", text))
        if exam and not no_exam and r.random() < 0.5:
            parts.append("Exam: " + ", ".join(x["text"] for x in exam))
        return "; ".join(parts), (kept[0] if kept else None)

    def _render(self, prompt, low, out):
        r = rng_for(prompt, "render")
        if "json" in low:
            body = json.dumps(out, ensure_ascii=False, indent=1)
            body = f"```json\n{body}\n```" if r.random() < 0.5 else body
            return ("Here is the note.\n" + body) if r.random() < self.read.chatty else body
        return "\n".join(f"{k}: {v['value'] if isinstance(v, dict) else v}" for k, v in out.items())

    def _recipe(self, prompt, low, hidden, want):
        rc = hidden["recipe"]
        if not want:
            return "This procedure synthesises the target by a standard route; the conditions look reasonable."
        segs = _field_segs(low, want)
        cue_only = bool(re.search(CUE_RE, low))
        bare = bool(re.search(BARE_NA_RE, low))
        cue_shape = '"cue"' in low
        out = {}
        for f in want:
            r = rng_for(prompt, "recipe", f["key"])
            key, seg = f["key"], segs.get(f["name"], "")
            val, cue = None, ""
            gold_txt = rc["gold_text"][key]
            if key == "temp":
                always = bool(re.search(r"always fill|never n/?a|always explain", seg or low))
                if always or not cue_only:
                    focus = bool(re.search(r"main (?:reaction|firing|step)|ignore drying", low) and re.search(r"mechanism", low))
                    p = 0.5 if focus else (0.35 if always else 0.25)
                    x = r.random()
                    if x < p:
                        val = rc["mech"]
                    elif x < p + (1 - p) / 2:
                        val = rc["wrong"]
                    else:
                        val = r.choice(GENERIC_TEMP)
            else:
                cued = rc["cues"].get(key)
                ex = rc["excess"] if key == "phys" else None
                ex_asked = bool(re.search(r"excess", seg))
                if ex and (ex_asked or (not cue_only and r.random() < 0.5)):
                    val = f"The excess of the precursor compensates the loss of volatile {ex['el']} during firing."
                    cue = ex["step"]
                elif cue_only or ex_asked:
                    if cued and cue_only:
                        clause = re.split(r" to | because | so | as | for ", cued, maxsplit=1)[-1].rstrip(".")
                        val, cue = f"The procedure does this {clause}.", cued
                elif r.random() < 0.2:
                    val = CONFAB[key]
                    if cued and r.random() < 0.7:
                        val, cue = cued, cued
                if val and not is_na(gold_txt) and r.random() < (0.2 if cue else 0.15):
                    val = val + " " + gold_txt
            if val is None:
                val = "N/A" if bare or r.random() >= 0.1 else r.choice(["N/A (not stated)", "Not stated in the procedure"])
            out[f["name"]] = {"value": val, "cue": cue} if cue_shape else val
        return self._render(prompt, low, out)

    def _judge(self, r, status, strict, ambig=False):
        """Simulated verdict on the last answer; errors concentrate on the ambiguous answer forms."""
        x = r.random()
        if status == "partial":
            p = (0.25 if strict else 0.4) if ambig else (0.03 if strict else 0.08)
            return "correct" if x < p else "partial"
        if status == "incorrect":
            pp = 0.4 if ambig else 0.15
            return "partial" if x < pp else ("correct" if x < pp + 0.02 else "incorrect")
        pc = (0.55 if strict else 0.4) if ambig else 0.12
        return "partial" if x < pc else "correct"

    def _tutor(self, prompt, low, hidden, ids, instr=""):
        t = hidden["tutor"]
        if t["last_item"] not in ids and _norm(t["last_text"]) not in _norm(prompt):
            return "I need the end of the dialogue (the student's last response) to assess it."
        r = rng_for(prompt, "tutor")
        strict = bool(re.search(r"fully and specifically|strict", low))
        status = self._judge(r, t["status"], strict, bool(t.get("ambig")))
        told = re.search(r"(?<!if )(?<!when )(?:the )?student'?s (?:last|latest) response (?:in this dialogue )?is "
                         r"(partially correct|incorrect|correct)\b", low)
        if told:
            status = {"partially correct": "partial", "incorrect": "incorrect", "correct": "correct"}[told.group(1)]
        label = {"correct": "Correct response", "partial": "Partially correct response", "incorrect": "Incorrect response"}[status]
        defines_partial = bool(re.search(r"accepts part|right track|soft", low))
        view = []
        for i, tt in enumerate(t["turns"][:-1]):
            v = tt["status"]
            if v == "partial" and rng_for(prompt, "turn", i).random() < (0.1 if defines_partial else 0.25):
                v = "correct"
            view.append(v)
        answers = [v for v in view if v != "none"]
        prev_bad = bool(answers) and answers[-1] != "correct"
        any_bad = any(v != "correct" for v in answers)
        if re.search(LEDGER_RE, low):
            turns = [{"text": tt["text"], "subq": tt["subq"], "verdict": v} for tt, v in zip(t["turns"][:-1], view, strict=False)]
            turns.append({"text": t["turns"][-1]["text"], "subq": t["turns"][-1]["subq"], "verdict": status})
            obj = {"turns": turns}
            asked = set(re.findall(r'"([a-z_]+)"\s*:', instr))
            if asked & {"last_verdict", "identification", "hint", "tell", "next"}:
                ind, dire = self._content(r, low, t, status, "Indirect"), self._content(r, low, t, status, "Direct")
                if "last_verdict" in asked:
                    obj["last_verdict"] = status
                if status == "correct":
                    if "next" in asked:
                        obj["next"] = ind["tr"]
                else:
                    for k, v in (("identification", ind["mi"]), ("hint", ind["mr"]), ("tell", dire["mr"])):
                        if k in asked:
                            obj[k] = v
            return json.dumps(obj, ensure_ascii=False)
        if status == "correct":
            rep = "No"
        elif re.search(PREV_RULE_RE, low):
            rep = "Yes" if prev_bad else "No"
        elif re.search(ANY_RULE_RE, low):
            rep = "Yes" if any_bad else "No"
        else:
            rep = "Yes" if any_bad and r.random() < 0.5 else "No"
        if re.search(r"scaffold|first mistake|first slip", low):
            typ = "Direct" if rep == "Yes" else "Indirect"
        else:
            typ = "Direct" if r.random() < 0.5 else "Indirect"
        if re.search(r"feedback is direct|state the correct answer", low):
            typ = "Direct"
        elif re.search(r"feedback is indirect|without giving", low):
            typ = "Indirect"
        want = [f["name"] for f in hidden["fields"] if f["name"].lower() in low]
        keys = re.findall(r'"([A-Za-z_][A-Za-z_ ]{1,40})"\s*:', instr)
        content = self._content(r, low, t, status, typ)
        if len(want) <= 1 and keys:
            return json.dumps({k: self._tutor_key(k, t, status, label, rep, typ, content) for k in keys}, ensure_ascii=False)
        if len(want) <= 1:
            if re.search(r"next sub-question|next part|move on|transition", low):
                return content["tr"]
            return f"Label: {label}"
        structured = bool(re.search(STRUCT_RE, low, re.S))
        if status == "correct":
            vals = {"Mistake identification": "N/A", "Mistake rectification": "N/A", "Mistake identification type": "N/A",
                    "Mistake rectification type": "N/A", "Sub-question Transition": content["tr"]}
        else:
            vals = {"Mistake identification": content["mi"], "Mistake rectification": content["mr"],
                    "Mistake identification type": typ, "Mistake rectification type": typ, "Sub-question Transition": "N/A"}
        vals = {"Evaluation of the student response": label, "Repeated mistake": rep, **vals}
        if not structured:
            if r.random() < 0.55:
                vals["Repeated mistake"] = r.choice(["True", "False", "N/A"])
            for k in ("Mistake identification type", "Mistake rectification type"):
                if r.random() < 0.55:
                    vals[k] = r.choice(["Conceptual error", "no_mistake", "None", "Factual correction"])
            if status == "correct" and r.random() < 0.3:
                vals["Mistake identification"] = "false"
            if r.random() < 0.85:
                vals["Sub-question Transition"] = r.choice(["Yes", "True", "false", "No transition yet.",
                                                            "Yes, the tutor moves to the next subproblem."])
        return self._render(prompt, low, {k: v for k, v in vals.items() if k in want})

    def _content(self, r, low, t, status, typ):
        """Mistake identification / rectification / transition text."""
        sp = t["species"]
        (ka, va), (kb, vb) = t["attrs"]
        key = t["key"] or [ka, va]
        if status != "correct" and t["status"] == "correct":
            key = [kb, vb]
        idea = bool(re.search(r"correct idea|what the correct|in fact|precisely what is wrong", low)) or r.random() < 0.7
        mi = (f"The student's {key[0]} is {'wrong' if status == 'incorrect' else 'missing or imprecise'}"
              + (f"; the {sp} has {key[0]} {key[1]}." if idea else "."))
        if typ == "Direct":
            mr = f"The {sp} has {ka} {va} and {kb} {vb}. Can you describe the next one?"
        elif re.search(r"name the concept|guid|hint", low) or r.random() < 0.7:
            mr = f"Think about the {key[0]} of the {sp}: what does the problem say about it?"
        else:
            mr = "Can you try that again?"
        tr = f"Let's move on to the {t['next']}." if t["next"] else "Let's summarize what we found."
        return {"mi": mi, "mr": mr, "tr": tr}

    def _tutor_key(self, key, t, status, label, rep, typ, content):
        """Value for one requested JSON key of a free-form tutor request."""
        k = key.lower()
        if "type" in k:
            return typ if status != "correct" else "N/A"
        if re.search(r"eval|label|verdict|correct", k):
            return label
        if "repeat" in k:
            return rep
        if re.search(r"identif|error|wrong|mistake", k):
            return "N/A" if status == "correct" else content["mi"]
        if re.search(r"rectif|feedback|hint|correction", k):
            return "N/A" if status == "correct" else content["mr"]
        if re.search(r"transition|next", k):
            return content["tr"]
        if "question" in k:
            (ka, _va), (kb, _vb) = t["attrs"]
            return f"What are the {ka} and {kb} of the {t['species']}?"
        if "response" in k:
            return "(the student's last response)"
        return f"The student's answer about the {t['species']} is {status}."


def parse_answer(pred: str) -> dict[str, str]:
    txt = (pred or "").strip()
    cands = [m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", txt, re.S)]
    a, b = txt.find("{"), txt.rfind("}")
    if a != -1 and b > a:
        cands.append(txt[a:b + 1])
    cands.append(txt)
    for c in cands:
        try:
            o = json.loads(c)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(o, dict):
            return {str(k).strip().lower(): ("" if v is None else str(v)) for k, v in o.items()}
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\s*[-*]?\s*([^:]{2,80}?)\s*:\s*(.+)$", line)
        if m:
            out[m.group(1).strip().strip('"').lower()] = m.group(2).strip()
    return out


def _has_stem(cand_n: str, stem: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(_norm(stem)), cand_n) is not None


def field_score(f: dict, cand: str, hidden: dict) -> float:
    gold = f["gold"]
    if is_na(gold):
        return 1.0 if is_na(cand) else 0.0
    if is_na(cand):
        return 0.0
    cn = _norm(cand)
    if hidden["variant"] == "recipe":
        return 1.0 if any(all(_has_stem(cn, s) for s in alt) for alt in f.get("alts") or []) else 0.0
    if hidden["variant"] == "tutor":
        t = hidden["tutor"]
        name = f["name"]
        if name in ("Evaluation of the student response", "Repeated mistake", "Mistake rectification type",
                    "Mistake identification type"):
            return 1.0 if _norm(gold) == cn or (name.startswith("Evaluation") and _norm(gold).split()[0] == cn.split()[0]) else 0.0
        if name == "Mistake rectification":
            k0, k1 = t.get("key") or t["attrs"][0]
            return 1.0 if _has(cn, k1) or (_has(cn, k0) and "?" in cand) else 0.0
        kws = t["kws"].get(name, [])
        return 1.0 if all((k in cand) if k == "?" else _has(cn, k) for k in kws) else 0.0
    facts = [x for x in hidden["facts"] if x["key"] == f["key"]]
    tf = [x for x in facts if x["kind"] == "true"]
    if f["kind"] == "phrase":
        ok = all(_has(cn, k) for k in tf[0]["kw"]) and not any(_has(cn, k) for k in f.get("avoid", []))
        return 1.0 if ok and len(cn.split()) <= 8 else 0.0
    if f["kind"] == "sentence":
        return 1.0 if all(_has(cn, k) for k in tf[0]["kw"]) and len(cn.split()) <= 25 else 0.0
    if f["kind"] == "ros":
        if any(all(_has(cn, k) for k in x["kw"]) for x in facts if x["kind"] == "trap" and x.get("sub") == "exam"):
            return 0.0
        return ros_score(cand, tf)
    if f["key"] == "exam" and re.search(EXAM_RELABEL_RE, cand):
        return 0.0
    hit = sum(1 for x in tf if all(_has(cn, k) for k in x["kw"]))
    need = 1.0 if f["kind"] in ("scalar", "trap") else 0.7
    return 1.0 if tf and hit / len(tf) >= need - 1e-9 else 0.0


_ROS_SYS = sorted({*SYS_ALIAS, *SYS_ALIAS.values(), "HEENT", "Skin", "Psychiatric", "Eyes", "Endocrine",
                   "Hematologic", "Allergic"}, key=len, reverse=True)
EXAM_RELABEL_RE = r"(?<![A-Za-z])(?:" + "|".join(re.escape(x) for x in _ROS_SYS) + r")\s*:"
_POS_RE = r"(?:reports?|endorses?|positive for|complains? of|admits? to|notes?|with)\b"
_NEG_RE = r"(?:denies|denied|negative for|no|without|not)\b"
ROS_HIST_RE = r"prior|previous|history|before|past|injur"
ROS_MECH_RE = r"\bpops?\b|\bfalls?\b|\bfell\b|trauma|twist"
ROS_MAX_WORDS = 6


def ros_items(cand: str) -> list[tuple[str, str, str]]:
    """'<System>: Reports X, Y. Denies Z.' -> [(system, '+'/'-', symptom phrase)] (dict / list reprs flattened)."""
    txt = re.sub(r"[{}\[\]\"']", " ", str(cand))
    parts = re.split(r"([A-Za-z/]+)\s*:", txt)
    out = []
    for hdr, seg in zip(parts[1::2], parts[2::2], strict=False):
        pol = "+"
        for clause in re.split(r"[.;\n]", seg):
            c = clause.strip(" ,-")
            if not c:
                continue
            m = re.match(_NEG_RE, c, re.I)
            if m:
                pol, c = "-", c[m.end():]
            else:
                m = re.match(_POS_RE, c, re.I)
                if m:
                    pol, c = "+", c[m.end():]
            for it in re.split(r",|\band\b|\bor\b|\bnor\b|/", c):
                it = re.sub(r"^\s*(?:no|any)\s+", "", it.strip(" ,-")).strip()
                if it:
                    out.append((hdr.strip(), pol, it))
    return out


def ros_score(cand: str, tf: list[dict]) -> float:
    """Per-symptom ROS check: chief complaint positive under its organ system, asked-about denials present."""
    items = ros_items(cand)
    hit = 0
    for x in tf:
        kw = x["kw"][-1]
        pos = x["verb"] == "Reports"
        hit += any((h.lower() == x["sys"].lower() or not pos) and p == ("+" if pos else "-") and _has(_norm(it), kw)
                   and len(it.split()) <= ROS_MAX_WORDS for h, p, it in items)
    bad = 0
    bad += any(p == "-" and re.search(ROS_HIST_RE, it, re.I) for _h, p, it in items)
    bad += any(re.search(ROS_MECH_RE, it, re.I) for _h, p, it in items)
    cc = [x for x in tf if x["verb"] == "Reports" and x["sys"] != "Constitutional"]
    bad += any(h.lower() in ("constitutional", "general") and _has(_norm(it), x["kw"][-1])
               for x in cc for h, _p, it in items)
    return hit / max(1, len(tf) + bad)


def score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    if not (final or "").strip():
        return 0.0
    pred = parse_answer(final)
    scores = {}
    for f in hidden["fields"]:
        scores[f["name"]] = field_score(f, pred.get(f["name"].lower(), ""), hidden)
    if isinstance(state, dict):
        state["form_field_scores"] = scores
        state["form_parsed_fields"] = len(pred)
        na = [f for f in hidden["fields"] if is_na(f["gold"])]
        state["form_na_credit"] = sum(scores[f["name"]] for f in na) / len(na) if na else None
        state["form_reason_na"] = sum(1 for v in pred.values() if not is_na(v) and re.match(r"\s*(n/?a\b|not (mentioned|discussed|stated)|none reported)", v.lower()))
    return sum(scores.values()) / max(1, len(scores))


HINTS = HintSpec(
    strategy=("This is a convention task, not a reading task. (1) Print the record. (2) In a REPL block write DEFS: one "
              "line per checklist field saying what belongs in it for THIS kind of record and exactly when it is N/A. "
              "Read field names the way an expert form means them: a clinical 'Drugs' field is tobacco / alcohol / "
              "recreational use, never medications; 'Severity' is a numeric rating; 'Temporal Factor' is a time pattern, "
              "not the onset or what makes it worse; 'Employment' is the patient's own job. (3) Draft every value "
              "(yourself, or one sub-call that gets DEFS and returns a verbatim quote per value). (4) Audit each draft "
              "against its definition and set it to N/A when it does not satisfy it (an adjective is not a rating; a "
              "medication is not a recreational drug). Narrative fields need complete, specific values: every finding, "
              "each diagnosis WITH its reasoning, the full plan. (5) When one judgement decides which fields exist, "
              "enforce it in Python both ways: a fully correct answer has no mistake fields, no mistake types and "
              "Repeated mistake = No, and does have a transition; a partially correct or incorrect answer MUST have the "
              "mistake identified, a corrective rectification, both types, and transition N/A. 'Repeated mistake' is Yes "
              "when the student's previous answer was also not accepted (an error again right after feedback). Mistake identification is one "
              "sentence about the student ('The student states X, but ...'); the rectification is the tutor's own next "
              "words TO the student, not a description of them. Scaffold the feedback: on a FIRST mistake those words "
              "are a guiding question that does not give the answer away and both types are Indirect; on a REPEATED "
              "mistake they state the correct answer and both types are Direct. (6) Every decline is the "
              "bare token N/A (never 'N/A (reason)'); submit json.dumps of a dict with every field."),
    program=(
        "print(context)                                   # turn 1: read the record\n"
        "DEFS = {\n"
        "  '<field>': '<what belongs here for this record type>; N/A when <condition>',   # one line per field\n"
        "}\n"
        "draft = {f: '<value from the record>' for f in FIELDS}             # or one llm_query(record + DEFS, quotes)\n"
        "for f in FIELDS:                                 # audit each value against its definition\n"
        "    if not satisfies(draft[f], DEFS[f]):         # e.g. adjective in a numeric-rating field\n"
        "        draft[f] = 'N/A'\n"
        "# structure: one judgement decides which fields exist\n"
        "if draft.get('Evaluation of the student response') == 'Correct response':\n"
        "    for f in MISTAKE_FIELDS: draft[f] = 'N/A'\n"
        "elif 'Evaluation of the student response' in draft:\n"
        "    assert all(draft[f] != 'N/A' for f in MISTAKE_FIELDS); draft['Sub-question Transition'] = 'N/A'\n"
        "NA = {'n/a', 'na', 'none', 'not applicable', 'not mentioned', ''}\n"
        "final = {f: ('N/A' if str(draft.get(f, '')).strip().lower() in NA or str(draft.get(f, '')).lower().startswith('n/a') "
        "else draft[f]) for f in FIELDS}\n"
        "answer['content'] = json.dumps(final)\n"
        "answer['ready'] = True"),
)


OWN_WORDS = ("Reason in your own words as the domain expert: never mention, quote or refer to these instructions or "
             "to a hint; state each rule as your own reasoning about the field.")
_COMMON_HEAD = ("This is a convention task, not a reading task. (1) Print the record. (2) In a REPL block write DEFS: one "
                "line per checklist field saying what belongs in it for THIS kind of record and exactly when it is N/A. ")
_COMMON_TAIL = ("Every decline is the bare token N/A (never 'N/A (reason)'); submit json.dumps of a dict with every "
                "field. ")
VISIT_STRATEGY = (_COMMON_HEAD +
                  "(3) Draft every value. (4) Audit each draft against its definition and set it to N/A when it does "
                  "not satisfy it. (5) Every decline is the bare token N/A (never 'N/A (reason)', 'None', 'not "
                  "discussed', or a sentence explaining an absence); submit json.dumps of a dict whose keys are the "
                  "checklist field names copied exactly (e.g. 'Review of Systems (ROS)', not a shortened name), one per "
                  "field.\n\nRead the fields the way a clinical note means them.\n"
                  "- The symptom fields (Onset, Duration, Location, Character, Radiation, Severity, Temporal Factor, "
                  "Alleviating and Aggravating Factors) describe ONE presenting symptom: the chief complaint. Never list "
                  "several problems in them. When the visit is a follow-up of chronic conditions, a results discussion or "
                  "an annual exam with no new presenting symptom, they are N/A.\n"
                  "- Onset: WHEN that symptom started, as a time ('3 weeks ago', 'yesterday evening'), not the mechanism "
                  "of injury. Duration: how long it has lasted, short ('3 weeks'). Location: the body site only, a few "
                  "words. Character: a short descriptive phrase for the complaint.\n"
                  "- Severity is a numeric rating ('6/10'); an adjective is not a rating, so N/A. Radiation is where pain "
                  "spreads; a denial is N/A.\n"
                  "- Temporal Factor is rare: only a recurring cyclical pattern the patient states (worse at night, in "
                  "winter, around menses). Constant / intermittent, getting worse, chronic, triggered by an activity, or "
                  "the onset story are NOT temporal factors -> N/A.\n"
                  "- Alleviating and Aggravating Factors: only what the patient says makes that symptom better or worse.\n"
                  "- Both visit-reason fields are a short phrase (2-6 words) naming the primary complaint or condition "
                  "addressed, with its side ('Left ankle pain', 'Follow-up of CHF exacerbation', 'Annual exam'); never a "
                  "bare 'Follow up' -- say what is followed up. Not a sentence and not every problem.\n"
                  "- Patient Age: a bare number, only when stated. Patient Sex: female / male when stated or clear from "
                  "pronouns or titles, bare word.\n"
                  "- Vital Signs: measured values or the clinician's explicit statement about them. Other Objective Data: "
                  "reported results of labs, imaging and tests (not tests merely ordered).\n"
                  "- Physical Examination Findings: every exam finding. Diagnosis and Clinical Impression: each diagnosis "
                  "WITH the reasoning the doctor gives. Treatment and Management Plan: the full plan per problem "
                  "(medications, tests, referrals, advice, follow-up).\n"
                  "- Review of Systems is never N/A: the chief complaint as a positive and every symptom the patient "
                  "denies, grouped as '<System>: Reports X. Denies Y.' under standard system names (Constitutional, "
                  "HENT, Cardiovascular, Respiratory, Gastrointestinal, Genitourinary, Musculoskeletal, Neurological, "
                  "...); each symptom in a few words, subjective symptoms only, never exam findings.\n"
                  "- Relevant Medical History: the names of past and chronic conditions, short; a statement of absence "
                  "('no prior injury') is N/A.\n"
                  "- Social fields are N/A unless the patient's own habit is described: 'Drugs' is tobacco / alcohol / "
                  "recreational use, never medications; 'Eating habits' is the patient's usual diet when it bears on "
                  "the problem (not a passing mention of a meal); 'Activities' is the patient's regular exercise, sports "
                  "or hobbies (not how an injury happened, not restrictions or plans); 'Home and Environment' is who the "
                  "patient lives with or an exposure at home (not a passing mention of a relative); 'Employment' is the "
                  "patient's own job.\n" + OWN_WORDS)
TUTOR_STRATEGY = ("This is a convention task, not a reading task. (1) Print the dialogue. (2) In a REPL block write a "
                  "LEDGER: one line per student answer, in order, with how the tutor's NEXT turn treated it: correct "
                  "(fully accepted and moved on), partial (accepted part, then corrected, sharpened or asked for more), "
                  "incorrect (rejected), none (not an answer). (3) Judge the student's LAST response yourself, the way "
                  "this tutor judged the earlier ones: 'Correct response' only if it fully and specifically answers "
                  "what the tutor last asked; a response that is on the right track but vague, incomplete or imprecise "
                  "is 'Partially correct response'; a response whose central claim is wrong, or that does not answer, "
                  "is 'Incorrect response'. (4) That judgement decides which fields exist; enforce it in Python both "
                  "ways: a fully correct answer has the four mistake fields N/A, Repeated mistake = No, and a "
                  "Sub-question Transition naming the next sub-question of the problem in a short phrase; a partially "
                  "correct or incorrect answer MUST have the mistake identified, a rectification, both types, and "
                  "Transition N/A. (5) 'Repeated mistake' = Yes exactly when the last response is not fully correct AND "
                  "the student's PREVIOUS answer in the ledger was also not fully accepted (the student errs again right "
                  "after corrective feedback); otherwise No. An earlier mistake that the student already fixed does not "
                  "count. (6) Mistake identification says precisely what is wrong or missing AND what the correct idea "
                  "is, in 1-2 sentences. The rectification is the tutor's own next words TO the student, not a "
                  "description of them, and it carries real content. Scaffold it: on a first mistake (Repeated = No) "
                  "the words name the concept to reconsider and guide toward it, ending with a guiding question, "
                  "without handing over the answer, and both types are Indirect; on a repeated mistake (Repeated = Yes) "
                  "the words state the correct answer explicitly, then one short question that continues, and both "
                  "types are Direct. (7) " + _COMMON_TAIL + OWN_WORDS)
RECIPE_STRATEGY_V2 = (
    "How an expert fills this materials-synthesis rationale checklist. The procedure is short: read it yourself (no "
    "sub-calls needed) and submit in your second turn.\n"
    "- The procedure almost never states its reasons; the expert's rationales come from materials-science knowledge, "
    "and the expert explains only the choices that matter most.\n"
    "- 'Temperature and Heating Method': ALWAYS fill (never N/A). In 1-2 sentences say why the MAIN reaction / firing "
    "/ annealing temperature(s) suit THIS target and precursors, and, if the heating method is not a plain furnace "
    "(microwave, plasma, flash or rapid heating, autoclave, hot pressing, arc melting, molten salt), what that method "
    "enables. Name ONE specific mechanism (the phase that forms or crystallises there, a precursor that decomposes / "
    "melts / volatilises, a decomposition or competing-phase limit it stays below, a metastable phase it kinetically "
    "traps, a flux it melts), with known values when you know them. Ignore drying, precipitation, ramp and cooling "
    "rates; no generic filler.\n"
    "- 'Physical and Chemical Properties of Precursors': fill ONLY when the procedure states an excess of a precursor "
    "(e.g. '10% excess Li2CO3'): the excess compensates the loss of the volatile element during high-temperature "
    "treatment. Otherwise N/A.\n"
    "- Every other field (Structural Considerations, Handling Precursor Reactivity, Atmosphere, Duration) is exactly "
    "N/A, even when the procedure lists an atmosphere, a time or a purpose clause: the expert's reasons for those come "
    "from the paper, and a guessed reason is wrong more often than N/A is.\n"
    "- Submit with answer[\"content\"] = json.dumps({...every field...}); answer[\"ready\"] = True. Never rebind "
    "`answer` itself.\n" + OWN_WORDS)
_V3E_RECIPE = [
    ("struct", ["why a precursor suits the target's crystal structure: a paper-specific reason, so N/A"], "N/A"),
    ("handling", ["how the precursors' reactivity / air sensitivity is managed: a paper-specific reason, so N/A"], "N/A"),
    ("phys", ["a precursor property that drives a choice: N/A, except a stated excess (EXCESS below)"], "N/A"),
    ("temp", ["1-2 sentences from materials knowledge, never N/A: why the MAIN reaction / firing / annealing temperature (and a",
              "non-furnace method: microwave, plasma, flash or rapid heating, autoclave, hot pressing, arc melting, high",
              "pressure, molten salt) suits THIS target and these precursors -- name that step, then one specific mechanism",
              "(the phase that forms or crystallises there, a precursor that decomposes / melts / volatilises, a decomposition",
              "or competing-phase limit it stays below, a metastable phase it traps, a flux it melts, what the method enables),",
              "with known values; not a drying, precipitation, ramp or cooling step, no generic filler"], "..."),
    ("atm", ["the gas / vacuum choice: a paper-specific reason, so N/A even when the procedure names the gas, its role or a "
             "purpose"], "N/A"),
    ("dur", ["the hold times: a paper-specific reason, so N/A even when the procedure lists the times or says what they "
             "achieve"], "N/A"),
]
_V3E_HEAD = ("You are a materials scientist; fill this rationale checklist the way you always do, in two turns of code. "
             "The comments below are the standard meaning of each field, which you know: write them as your own code. "
             "Start each turn with one plain sentence of your own saying what the block does (one line, no lists), and "
             "never mention, quote or refer to this text (no \"hint\", "
             "\"instructions\", \"rules\", \"template\", \"guidelines\", \"note\").")


def recipe_note_hint(fields: list[str] | None = None) -> str:
    """Recipe hint for ``fields`` (None = all)."""
    names = {k: n for k, n, _r in RECIPE_FIELDS}
    have = set(fields) if fields else set(names.values())
    body = []
    for key, comment, val in _V3E_RECIPE:
        if names[key] not in have:
            continue
        body += ["    # " + c for c in comment] + [f"    {json.dumps(names[key])}: {json.dumps(val)},"]
    phys = names["phys"]
    excess = ("EXCESS = None   # a precursor added in a stated excess and its volatile element, e.g. (\"Li2CO3\", "
              "\"lithium\"); else None\n"
              "if EXCESS:\n"
              f"    form[{json.dumps(phys)}] = (\n"
              "        f\"The excess {EXCESS[0]} compensates the loss of the volatile {EXCESS[1]} during the "
              "high-temperature treatment.\")\n") if phys in have else ""
    code2 = ("import json\nform = {\n" + "\n".join(body) + "\n}\n" + excess +
             "answer[\"content\"] = json.dumps(form)\nanswer[\"ready\"] = True\nprint(answer[\"content\"])")
    return (_V3E_HEAD + "\n\nTurn 1:\n```repl\nprint(context)\n```\n\n"
            "Turn 2: fill `form`, keeping the comments (they say what each field holds), and submit in the same block.\n"
            "```repl\n" + code2 + "\n```")


RECIPE_STRATEGY = recipe_note_hint()


class RecipeHintSpec(HintSpec):
    """Hint text for a tier; note / strategy / program use the final c5 note."""

    def text(self, tier: str, fields: list[str] | None = None) -> str:
        if tier in ("strategy", "program", "note", "c5"):
            return RECIPE_NOTE_C5
        if tier == "v3e":
            return recipe_note_hint(fields)
        if tier in ("v2", "strategy_v2", "v8", "strategy_v8"):
            return HintSpec.text(self, "strategy")
        if tier in ("program_v2", "program_v8"):
            return HintSpec.text(self, "program")
        return HintSpec.text(self, tier)


_VISIT_PROGRAM = HINTS.program.split("# structure:")[0] + (
    "NA = {'n/a', 'na', 'none', 'not applicable', 'not mentioned', ''}\n"
    "final = {f: ('N/A' if str(draft.get(f, '')).strip().lower() in NA or str(draft.get(f, '')).lower().startswith('n/a') "
    "else draft[f]) for f in FIELDS}\n"
    "answer['content'] = json.dumps(final)\n"
    "answer['ready'] = True")
_RECIPE_PROGRAM = (
    "print(context)                                   # turn 1: read the procedure\n"
    "final = {f: 'N/A' for f in FIELDS}               # rationales for most choices come from the paper\n"
    "temp = [f for f in FIELDS if 'Temperature' in f][0]\n"
    "final[temp] = '<why the MAIN firing step / method suits this target: one specific mechanism, known values>'\n"
    "phys = [f for f in FIELDS if 'Physical and Chemical' in f]\n"
    "if phys and re.search(r'excess', context): final[phys[0]] = '<the excess compensates the volatile element loss>'\n"
    "answer['content'] = json.dumps(final)\n"
    "answer['ready'] = True")
_V3_GROUPS = [
    ("# symptom fields: about CHIEF only, never several problems; all N/A when the visit has no new presenting symptom", [
        ("Onset", "when it started, as a time ('3 weeks ago'), not how it happened"),
        ("Duration", "how long it has lasted ('3 weeks')"),
        ("Location", "body site, a few words ('right knee')"),
        ("Character", "one short phrase: the quality with its site ('sharp pain in the lower right back', 'pain and "
                      "swelling', 'pulsing headache'); not a list of symptoms, not what worsens it"),
        ("Radiation", "where the pain spreads; a denial is N/A"),
        ("Severity", "a numeric rating ('6/10') only"),
        ("Temporal Factor", "rare: only a recurring clock/calendar pattern the patient states, in their words ('worse "
                            "in the morning, better by afternoon'); constant / intermittent / worsening / end of day / "
                            "after activity are N/A"),
        ("Alleviating and Aggravating Factors", "what the patient says makes it better / worse")]),
    ("# visit", [
        ("Reason for Patient Office Visit or Hospitalization", "2-6 words naming the primary complaint / condition, "
         "with side ('Left ankle pain', 'Follow-up of CHF exacerbation', 'Abnormal lab results', 'Annual exam'); no "
         "cause, no second problem"),
        ("Patient Reason for the Visit", "the patient's own concern in one sentence: main complaint with side, what "
         "brought it on or what they want, other concerns they raise; never a bare 'Follow up'"),
        ("Patient Age", "bare number only when stated"),
        ("Patient Sex", "female / male")]),
    ("# objective", [
        ("Vital Signs", "measured values or the clinician's explicit statement about vitals"),
        ("Other Objective Data", "reported lab / imaging / test results (not tests ordered)"),
        ("Physical Examination Findings", "every exam finding the clinician states, in their words, under the body "
                                          "part / system they name (never relabelled by organ system), normal "
                                          "findings included"),
        ("Diagnosis and Clinical Impression", "each diagnosis WITH the clinician's reasoning"),
        ("Treatment and Management Plan", "the full plan: meds with dose, tests, referrals, advice, follow-up")]),
    ("# pertinent ROS, never N/A: '<System>: Reports X. Denies Y.'; CHIEF's symptoms under their organ system (pain of a\n"
     "    # joint/back/limb -> Musculoskeletal) + symptoms the clinician asked about; each 1-4 words; no history, mechanism,\n"
     "    # treatment response or exam findings; Constitutional = fever, chills, fatigue, weight, sleep, appetite", [
        ("Review of Systems (ROS)", "")]),
    ("# history / social: N/A unless the patient's own condition or habit is described", [
        ("Relevant Medical History", "names of past / chronic conditions and pertinent prior episodes, short; not "
                                     "today's complaint; 'no prior injury' / 'no significant history' is N/A"),
        ("Surgical History", "past operations with when, short"),
        ("Family History", "conditions of blood relatives, with who"),
        ("Drugs", "tobacco / alcohol / recreational use, never medications"),
        ("Activities", "the patient's regular exercise, sports, hobbies; N/A for the injury mechanism, restrictions, "
                       "plans, wishes, dislikes"),
        ("Home and Environment", "a home exposure or situation that bears on the problem (smoke, pets, a move, "
                                 "renovation, a stressor); merely who they live with is N/A"),
        ("Employment", "the patient's own job"),
        ("Eating habits", "usual diet when it bears on the problem; a passing mention of a meal is N/A")]),
]
_V3_AUDIT = [
    ("Severity", 'setna("Severity", lambda v: not re.search(r"\\d+\\s*(/|out of)\\s*10", v))'),
    ("Temporal Factor", 'setna("Temporal Factor", lambda v: not re.search(r"(?i)night|morning|evening|nocturnal|winter|'
                        'summer|season|menstru|period|monthly|weekend", v))'),
    ("Radiation", 'setna("Radiation", lambda v: bool(re.match(r"(?i)\\s*(no|denies|does not|without)\\b", v)))'),
    ("Relevant Medical History", 'setna("Relevant Medical History", lambda v: bool(re.match(r"(?i)\\s*(no|denies|none)\\b", v)))'),
    ("Drugs", 'setna("Drugs", lambda v: not re.search(r"(?i)smok|tobacco|cigar|vap|alcohol|drink|beer|wine|marijuana|'
              'cannabis|cocaine|recreational|substance", v))'),
]
_V3_HEAD = ("You are an experienced clinician; write this note the way you always do, in three turns of code. The "
            "comments and checks below are ordinary documentation practice that you know: write them as your own "
            "code. Keep your prose to a line or two per turn and never mention, quote or refer to this text (no "
            "\"hint\", \"instructions\", \"rules\", \"template\", \"guidelines\").")


def visit_note_hint(fields: list[str] | None = None) -> str:
    """Visit hint for ``fields`` (None = all)."""
    have = set(fields) if fields else {n for n, _k in VISIT_FIELDS.values()}
    W = 27
    body = []
    for head, items in _V3_GROUPS:
        rows = [(n, c) for n, c in items if n in have]
        if not rows:
            continue
        body.append("    " + head)
        for n, c in rows:
            k = json.dumps(n) + ":"
            body.append(f"    {k} \"...\"," + (" " * max(1, W - len(k) - 7)) + (f"# {c}" if c else "").rstrip())
            body[-1] = body[-1].rstrip()
    sym = [n for n, _c in _V3_GROUPS[0][1] if n in have]
    audit = [ln for n, ln in _V3_AUDIT if n in have]
    code2 = ("import json, re\n"
             "FIELDS = [...]   # the checklist field names, copied exactly from the task, in order\n"
             "VISIT = \"...\"    # one of: \"new complaint\", \"chronic follow-up\", \"results discussion\", \"annual exam\", "
             "\"post-op follow-up\"\n"
             "CHIEF = \"...\"    # the ONE new presenting symptom of a \"new complaint\" visit; None for every other visit type\n"
             "note = {\n" + "\n".join(body) + "\n}\n"
             "NA = \"N/A\"\n"
             "def absent(v):\n"
             "    return not str(v).strip() or bool(re.match(r\"(?i)\\s*(n/?a\\b|none\\b|unknown|not (mentioned|stated|"
             "discussed|reported|applicable))\", str(v)))\n"
             "note = {f: (NA if absent(note.get(f, \"\")) else str(note[f]).strip()) for f in FIELDS}\n"
             "def setna(f, cond):\n"
             "    if f in note and note[f] != NA and cond(note[f]):\n"
             "        note[f] = NA\n"
             + (f"SYMPTOM = {json.dumps(sym)}\n"
                "if CHIEF is None or VISIT != \"new complaint\":\n"
                "    for f in SYMPTOM: setna(f, lambda v: True)\n" if sym else "")
             + "".join(a + "\n" for a in audit)
             + "print(\"visit:\", VISIT, \"| chief:\", CHIEF, \"| ROS:\", note.get(\"Review of Systems (ROS)\"))\n"
               "print(json.dumps(note, indent=1))")
    return (_V3_HEAD + "\n\nTurn 1:\n```repl\nprint(context)\n```\n\n"
            "Turn 2: fill `note` from the dialogue (keep the comments; they say what each field holds), then run the "
            "audit in the same block.\n```repl\n" + code2 + "\n```\n\n"
            "Turn 3: read the printed note against the dialogue, fix any value in code (`note[f] = ...`), then submit:\n"
            "```repl\nanswer[\"content\"] = json.dumps(note)\nanswer[\"ready\"] = True\n```")


class VisitHintSpec(HintSpec):
    """Hint text for a tier; note / strategy / program use the final c10 note."""

    def text(self, tier: str, fields: list[str] | None = None) -> str:
        if tier in ("strategy", "program", "note", "c10"):
            return visit_note_c10(fields)
        if tier in ("v11c", "v3"):
            return visit_note_hint(fields)
        if tier in ("v8", "strategy_v8"):
            return HintSpec.text(self, "strategy")
        if tier == "program_v8":
            return HintSpec.text(self, "program")
        return HintSpec.text(self, tier)


class TutorHintSpec(HintSpec):
    """Hint text for a tier; note / strategy / program use the final c6 note."""

    def text(self, tier: str, fields: list[str] | None = None) -> str:
        if tier in ("strategy", "program", "note", "c6"):
            return TUTOR_NOTE
        if tier == "v1s":
            return TUTOR_NOTE_V1S
        if tier in ("v8", "strategy_v8"):
            return HintSpec.text(self, "strategy")
        if tier == "program_v8":
            return HintSpec.text(self, "program")
        return HintSpec.text(self, tier)


HINTS_BY_KIND = {"visit": VisitHintSpec(strategy=VISIT_STRATEGY, program=_VISIT_PROGRAM),
                 "tutor": TutorHintSpec(strategy=TUTOR_STRATEGY, program=HINTS.program),
                 "recipe": RecipeHintSpec(strategy=RECIPE_STRATEGY_V2, program=_RECIPE_PROGRAM)}


def _defined_fields(s: dict) -> float | None:
    code = "\n".join(root_code(s)).lower()
    hid = s.get("synth_hidden") or {}
    if hid.get("variant") != "visit":
        return None
    traps = [f for f in hid.get("fields", []) if f["kind"] == "trap"]
    if not traps:
        return None
    return sum(1.0 for f in traps if re.search(TRAP_DEFS[f["key"]], code)) / len(traps)


def _sub_log(s: dict) -> list[dict]:
    return [e for e in (s.get("rlm_sub_llm_log") or s.get("subcalls") or []) if isinstance(e, dict)]


def _probe_entry(s: dict) -> dict | None:
    """The c10 visit probe sub-call."""
    for e in _sub_log(s):
        p = str(e.get("prompt") or "")
        if '"chief"' in p and re.search(r'\{"visit"\s*:', p) and "Review of Systems" not in p:
            return e
    return None


def _json_obj(r: str) -> dict:
    try:
        o = json.loads(r[r.find("{"):r.rfind("}") + 1])
        return o if isinstance(o, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _visit_type(s: dict) -> float | None:
    """Logged only: did the program act on the right visit type? None if it can't be read."""
    hid = s.get("synth_hidden") or {}
    if hid.get("variant") != "visit" or "visit_type" not in hid:
        return None
    code = "\n".join(root_code(s))
    m = re.findall(r"^\s*VISIT\s*=\s*[\"']([^\"']+)[\"']", code, re.M)
    if not m:
        e = _probe_entry(s)
        if e is None:
            return None
        o = _json_obj(str(e.get("response") or ""))
        new = hid["visit_type"] == "new complaint"
        ok_v = (str(o.get("visit") or "").strip().lower() == "new complaint") == new
        ok_c = bool(o.get("chief")) == new
        return (ok_v + ok_c) / 2
    chief = re.findall(r"^\s*CHIEF\s*=\s*(.+)$", code, re.M)
    ok_v = m[-1].strip().lower() == hid["visit_type"]
    if not chief:
        return float(ok_v)
    ok_c = (chief[-1].strip().startswith("None")) == (hid["visit_type"] != "new complaint")
    return (ok_v + ok_c) / 2


def _hint_text(s: dict) -> str:
    for m in (s.get("rlm_history") or [])[:2]:
        c = str((m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or "")
        if "<hint>" in c:
            return c.split("<hint>", 1)[1].split("</hint>", 1)[0]
    return ""


def _hint_code_lines(hint: str) -> set[str]:
    return {ln.strip() for b in re.findall(r"```repl\n(.*?)```", hint, re.S) for ln in b.splitlines() if ln.strip()}


_LOOK_RE = re.compile(r"print\(\s*context\b|\bcontext\s*\[|\bcontext\.(?:find|count|index|split)\(")


def follows_note_parts(s: dict) -> dict[str, bool] | None:
    """Did the trace follow the row's note (turn and sub-call limits, no direct reads of `context`)? None if unhinted."""
    from .final_notes import note_version_of
    hid = s.get("synth_hidden") or {}
    kind = hid.get("variant")
    hint = _hint_text(s)
    if kind not in GENERATORS or not hint:
        return None
    ver = note_version_of(kind, hint)
    turns = int(s.get("rlm_iterations") or 0)
    code_turns = root_code(s)
    code = "\n".join(code_turns)
    own = _hint_code_lines(hint)
    looked = any(_LOOK_RE.search(ln) and ln.strip() not in own for ln in code.splitlines())
    subs = [str(e.get("prompt") or "") for e in _sub_log(s)]
    if ver == "t06_note_c10":
        return {"probe": _probe_entry(s) is not None,
                "draft": any("Review of Systems" in p and "Fill this clinical-note template" in p for p in subs),
                "json_dumps": "json.dumps" in code, "no_look": not looked, "turns_le_3": 0 < turns <= 3}
    if ver == "t04_note_c6":
        return {"combined_ledger": any(re.search(LEDGER_RE, p.lower()) and '"tell"' in p for p in subs),
                "subs_le_2": len(subs) <= 2, "no_look": not looked, "turns_le_3": 0 < turns <= 3}
    if ver == "t03_note_c5":
        return {"no_subcalls": not subs, "fromkeys": "dict.fromkeys(F" in code, "json_dumps": "json.dumps" in code,
                "turns_le_3": 0 < turns <= 3}
    if ver == "t06_note_v11c":
        return {"visit_set": bool(re.search(r"^\s*VISIT\s*=", code, re.M)), "turns_le_4": 0 < turns <= 4}
    if ver == "t04_note_v1s":
        return {"ledger": any(re.search(LEDGER_RE, p.lower()) for p in subs), "turns_le_5": 0 < turns <= 5}
    return {"turns_le_3": 0 < turns <= 3}


def _follows_note(s: dict) -> float | None:
    parts = follows_note_parts(s)
    return None if parts is None else (1.0 if all(parts.values()) else 0.0)


HINT_MENTION_RE = re.compile(
    r"\bhints?\b|\binstructions?\b|\binstructed\b|\bguidelines?\b|\bthe rules?\b|\bthese rules\b|\bmy rules\b|"
    r"\bthe definitions? (say|says|state|states|requires?)\b|\bas (specified|told|directed)\b|\bthe template\b|"
    r"\bper the (rule|definition|convention|template|guidance)s?\b|\bthe guidance\b|\bthe note says\b", re.I)


def root_prose(s: dict) -> str:
    out = []
    for m in (s.get("rlm_history") or [])[2:]:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "assistant":
            c = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
            out.append(re.sub(r"```.*?(?:```|$)", " ", c, flags=re.S))
    return "\n".join(out)


def _no_hint_mention(s: dict) -> float | None:
    """1 if the root never refers to the hint (such traces read oddly once the hint is stripped). Logged only."""
    if not (s.get("rlm_history") or [])[2:]:
        return None
    return 0.0 if HINT_MENTION_RE.search(root_prose(s)) else 1.0


def _json_answer(s: dict) -> float:
    fa = final_answer(s)
    try:
        return 1.0 if isinstance(json.loads(fa), dict) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _bare_na(s: dict) -> float:
    pred = parse_answer(final_answer(s))
    return 0.0 if any(re.match(r"\s*(n/?a\s*[\(\-:]|not (mentioned|discussed|stated)|none reported|unknown)", v.lower())
                      for v in pred.values()) else 1.0


CHECKS = [
    Check("json_answer", _json_answer, hard=True, doc="answer json.loads to a dict"),
    Check("na_token", _bare_na, hard=True, doc="no 'N/A (reason)' / 'Not mentioned in ...' values"),
    Check("defines_traps", _defined_fields, hard=False, doc="root code defines every trap field (visit rows)"),
    Check("visit_type", _visit_type, hard=False, weight=0.0, doc="root VISIT / CHIEF match the row's visit type (logged)"),
    Check("no_hint_mention", _no_hint_mention, hard=False, weight=0.0,
          doc="root prose never cites a hint / rules / instructions (logged; filter SFT traces on it)"),
    Check("follows_note", _follows_note, hard=False, weight=0.0,
          doc="trace adherence to the row's own note (final: visit c10 probe + draft, tutor c6 one combined call, "
              "recipe c5 no sub-calls; <= 3 turns, no root look at `context`); logged, SFT filter"),
    answer_not_copied(),
]


__all__ = ["FormReader", "generate", "score"]
