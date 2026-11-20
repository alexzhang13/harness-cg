"""BrowseComp-style questions over a synthetic world: a multi-clue oblique description of one target entity."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

from . import lexicon as L
from .world import World

ROLE_TYPE = {"self": None, "birthplace": "town", "school": "org", "spouse": "person", "works": "work",
             "creator": "person", "illustrator": "person", "publisher": "org", "shown_at": "org", "region": "region",
             "natives": "person", "founder": "person", "located": "town"}


@dataclass
class Clue:
    text: str
    role: str
    attr: str
    op: str
    args: list
    kind: str
    forms: list[str] = field(default_factory=list)


def resolve(W: World, eid: int, role: str) -> list[int]:
    e = W.ents[eid]
    if role == "self":
        return [eid]
    if role == "birthplace":
        return [e["town"]]
    if role == "school":
        return [e["school"]]
    if role == "spouse":
        return [e["spouse"]] if e.get("spouse") is not None else []
    if role == "works":
        return list(e.get("works") or [])
    if role in ("creator", "illustrator", "publisher", "founder"):
        return [e[role]] if e.get(role) is not None else []
    if role == "shown_at":
        return [e["shown"][0]] if e.get("shown") else []
    if role == "region":
        return [e["region"]]
    if role == "natives":
        return list(e.get("natives") or [])
    if role == "located":
        return [e["town"]]
    return []


def _in(v, lo, hi) -> bool:
    return v is not None and lo <= v <= hi


def value_holds(attr: str, op: str, args: list, v) -> bool | None:
    """Does KG value ``v`` of ``attr`` satisfy the clue? None = the value does not exist."""
    if v is None:
        return None
    if op == "cat":
        return L.TERM_CATEGORY.get(v) == args[0]
    if op == "theme":
        return L.THEME_CATEGORY.get(v) == args[0]
    if op == "eq":
        return v == args[0]
    if attr == "born":
        y, m, d = v
        if op == "range":
            return args[0] <= y <= args[1]
        if op == "month":
            return None if m is None else m == args[0]
        if op == "day":
            return None if d is None else d == args[0]
    if attr in ("prize", "exhib", "member"):
        obj = v[0]
        if obj != args[0]:
            return False
        if op == "obj_month":
            return v[2] == args[1]
        if op == "obj_range":
            return args[1] <= v[1] <= args[2]
        return True
    if attr == "spouse":
        return args[0] <= v[1] <= args[1]
    if attr == "nworks":
        n, y1, y2 = v
        if op == "nworks":
            return n == args[0] and y1 == args[1] and y2 == args[2]
        return args[0] <= n <= args[1]
    if attr == "shown":
        return args[0] <= v[2] <= args[1]
    if attr == "stolen":
        return args[0] <= v[0] <= args[1]
    if attr == "pop":
        return args[0] <= v[0] <= args[1]
    if attr == "landmark":
        return args[0] <= v[1] <= args[1]
    if attr == "gap":
        return v >= args[0]
    if op == "range":
        return args[0] <= v <= args[1]
    return False


def kg_value(W: World, eid: int, attr: str):
    e = W.ents[eid]
    if attr == "exhib":
        return e.get("exhib") or None
    if attr == "member":
        return e.get("member") or None
    if attr == "prize":
        z = e.get("prizes") if e["type"] == "person" else ([e["prize"]] if e.get("prize") else None)
        return z or None
    if attr == "spouse":
        return (e["spouse"], e["married"]) if e.get("spouse") is not None else None
    if attr == "year":
        return e.get("year")
    if attr == "gap":
        return (e["closed"] - e["founded"]) if e.get("closed") else None
    return e.get(attr)


def holds(W: World, eid: int, c: Clue) -> bool:
    for x in resolve(W, eid, c.role):
        v = kg_value(W, x, c.attr)
        if c.attr in ("exhib", "member", "prize") and isinstance(v, list):
            if any(value_holds(c.attr, c.op, c.args, vv) for vv in v):
                return True
        elif value_holds(c.attr, c.op, c.args, v):
            return True
    return False


def _yr_range(rng: random.Random, y: int, width: tuple[int, int] = (3, 9)) -> tuple[int, int, str]:
    r = rng.random()
    if r < 0.3:
        lo = y // 10 * 10
        return lo, lo + 9, f"in the {lo}s"
    if r < 0.5:
        lo = y // 10 * 10
        part = (y - lo)
        if part <= 3:
            return lo, lo + 3, f"in the early {lo}s"
        if part >= 7:
            return lo + 7, lo + 9, f"in the late {lo}s"
        return lo + 4, lo + 6, f"in the mid-{lo}s"
    w = rng.randint(*width)
    lo = y - rng.randint(0, w)
    if rng.random() < 0.25:
        return lo + 1, lo + w - 1, f"between {lo} and {lo + w} (exclusive)"
    return lo, lo + w, f"between {lo} and {lo + w}"


def _ordinal_century(c: int) -> str:
    n = c // 100 + 1
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _cr(rng: random.Random, y: int) -> tuple[int, int, str]:
    c = y // 100 * 100
    off = y - c
    if rng.random() < 0.3:
        return c, c + 99, f"in the {_ordinal_century(c)} century"
    if off <= 33:
        return c, c + 33, f"in the early {c}s"
    if off <= 66:
        return c + 34, c + 66, f"in the mid-{c}s"
    return c + 67, c + 99, f"in the late {c}s"


def _num_words(n: int) -> str:
    return L.NUMWORDS[n] if 0 <= n < len(L.NUMWORDS) else str(n)


SUBJ = {
    ("person", "self"): "this person", ("work", "creator"): "its creator", ("work", "illustrator"):
    "the illustrator of its first edition", ("person", "spouse"): "this person's spouse",
    ("org", "founder"): "its founder", ("town", "natives"): "someone born there",
}


def _subj(ttype: str, role: str, work_kind: str | None = None) -> str:
    if role == "creator" and work_kind:
        return {"painting": "the painter", "etching": "the artist", "symphony": "the composer", "opera": "the composer",
                "song cycle": "the composer"}.get(work_kind, "its author")
    return SUBJ.get((ttype, role), "this person")


def person_clues(W: World, pid: int, rng: random.Random, S: str, role: str) -> list[Clue]:
    p = W.ents[pid]
    out: list[Clue] = []
    Sp = S + ("'" if S.endswith("s") else "'s")
    cat = L.TERM_CATEGORY[p["prof"]]
    forms = [t.lower() for t in L.PROFESSIONS[cat][0]]
    if role not in ("creator", "illustrator"):
        out.append(Clue(rng.choice([f"{S} worked as {cat}.", f"{S} earned a living as {cat}."]), role, "prof", "cat",
                        [cat], "hyper", forms))
    y, m, d = p["born"]
    lo, hi, ph = _yr_range(rng, y)
    out.append(Clue(f"{S} was born {ph}.", role, "born", "range", [lo, hi], "range"))
    if rng.random() < 0.5:
        out.append(Clue(rng.choice([f"{S} was born in the month of {L.MONTHS[m - 1]}.",
                                    f"{Sp} birthday fell in {L.MONTHS[m - 1]}."]), role, "born", "month", [m], "range"))
    else:
        out.append(Clue(f"{S} was born on the {L.ORDINALS[d] if d < len(L.ORDINALS) else str(d) + 'th'} day of a month.",
                        role, "born", "day", [d], "read"))
    for par in ("father", "mother"):
        pc = L.TERM_CATEGORY[p[par]]
        out.append(Clue(f"{Sp} {par} was {pc}.", role, par, "cat", [pc], "hyper",
                        [t.lower() for t in L.PROFESSIONS[pc][0]]))
    n = p["nsib"]
    if n == 0:
        out.append(Clue(f"{S} had no siblings.", role, "nsib", "eq", [0], "read"))
    else:
        out.append(Clue(rng.choice([f"{S} had {_num_words(n)} siblings." if n > 1 else f"{S} had exactly one sibling.",
                                    f"{S} was one of {n + 1} children."]), role, "nsib", "eq", [n], "read"))
    if p["sibdeath"]:
        if rng.random() < 0.5:
            out.append(Clue(f"{S} lost a sibling in {p['sibdeath']}.", role, "sibdeath", "eq", [p["sibdeath"]], "read"))
        else:
            lo, hi, ph = _yr_range(rng, p["sibdeath"], (4, 10))
            out.append(Clue(f"{S} lost a sibling {ph}.", role, "sibdeath", "range", [lo, hi], "read"))
    if p["died"]:
        lo, hi, ph = _yr_range(rng, p["died"])
        out.append(Clue(f"{S} died {ph}.", role, "died", "range", [lo, hi], "range"))
    if p["spouse"] is not None:
        lo, hi, ph = _yr_range(rng, p["married"])
        out.append(Clue(f"{S} married {ph}.", role, "spouse", "range", [lo, hi], "range"))
        sc = L.TERM_CATEGORY[W.ents[p["spouse"]]["prof"]]
        out.append(Clue(f"{S} married {sc}.", "spouse" if role == "self" else role + ".spouse", "prof", "cat", [sc],
                        "hyper", [t.lower() for t in L.PROFESSIONS[sc][0]]))
    sch = W.ents[p["school"]]
    out.append(Clue(f"{S} was educated at {sch['name']}.", role, "school", "eq", [p["school"]], "literal",
                    [sch["name"].lower().removeprefix("the ")]))
    if p["prizes"]:
        z, zy = rng.choice(p["prizes"])
        lo, hi, ph = _yr_range(rng, zy)
        out.append(Clue(f"{S} received {W.ents[z]['name']} {ph}.", role, "prize", "obj_range", [z, lo, hi], "literal",
                        [W.ents[z]["name"].lower().removeprefix("the ")]))
    if p["exhib"]:
        g, gy = rng.choice(p["exhib"])
        out.append(Clue(f"{S} exhibited at {W.ents[g]['name']}.", role, "exhib", "obj", [g], "literal",
                        [W.ents[g]["name"].lower().removeprefix("the ")]))
    if p["member"]:
        s, sy, sm = rng.choice(p["member"])
        out.append(Clue(f"{S} was elected to {W.ents[s]['name']} in the month of {L.MONTHS[sm - 1]}.", role, "member",
                        "obj_month", [s, sm], "literal", [W.ents[s]["name"].lower().removeprefix("the ")]))
    if p["nworks"]:
        n, y1, y2 = p["nworks"]
        noun = {"a writer": "books", "a visual artist": "paintings", "a composer": "scores"}[p["cat"]]
        out.append(Clue(f"Between {y1} and {y2}, {S} produced {n} {noun} under their own name.", role, "nworks", "nworks",
                        [n, y1, y2], "read"))
    lo, hi, ph = _yr_range(rng, p["debut"])
    out.append(Clue(rng.choice([f"{S} turned professional {ph}.", f"{Sp} career began {ph}."]), role, "debut", "range",
                    [lo, hi], "range"))
    t = W.ents[p["town"]]
    sub = "birthplace" if role == "self" else role + ".birthplace"
    lo, hi, ph = _cr(rng, t["founded"])
    out.append(Clue(f"{S} was born in a town founded {ph}.", sub, "founded", "range", [lo, hi], "range"))
    if t["river"]:
        out.append(Clue(f"{S} was born in a town on the banks of {t['river']}.", sub, "river", "eq", [t["river"]], "literal",
                        [t["river"].lower().removeprefix("the ")]))
    if role == "self" and p["works"]:
        w = W.ents[rng.choice(p["works"])]
        tc = L.THEME_CATEGORY[w["theme"]]
        out.append(Clue(f"{S} created {_a_kind(w['kind'])} about {tc}.", "works", "theme", "theme", [tc], "hyper",
                        [x.lower() for x in L.THEMES[tc]]))
    return out


def _a_kind(kind: str) -> str:
    return ("an " if kind[0] in "aeiou" else "a ") + kind


def work_clues(W: World, wid: int, rng: random.Random) -> list[Clue]:
    w = W.ents[wid]
    out: list[Clue] = []
    lo, hi, ph = _yr_range(rng, w["year"])
    verb = "first published" if w["publisher"] is not None else "completed"
    out.append(Clue(f"It was {verb} {ph}.", "self", "year", "range", [lo, hi], "range"))
    tc = L.THEME_CATEGORY[w["theme"]]
    out.append(Clue(rng.choice([f"It is about {tc}.", f"It centres on {tc}."]), "self", "theme", "theme", [tc], "hyper",
                    [x.lower() for x in L.THEMES[tc]]))
    if w["reception"] == "poor":
        out.append(Clue(rng.choice(["It was not well received at first.", "Its initial reception disappointed its maker."]),
                        "self", "reception", "eq", ["poor"], "read"))
    else:
        out.append(Clue("It was well received from the start.", "self", "reception", "eq", ["good"], "read"))
    if w["shown"]:
        g, gy, mo = w["shown"]
        lo = max(1, mo - rng.randint(0, 1))
        out.append(Clue(f"At its first exhibition it was on show for {lo} to {lo + 1 + rng.randint(0, 1)} months.", "self",
                        "shown", "months", [lo, lo + 2], "read"))
        gal = W.ents[g]
        lo, hi, ph = _yr_range(rng, gal["founded"])
        out.append(Clue(f"It was first shown at a venue founded {ph}.", "shown_at", "founded", "range", [lo, hi], "range"))
        if gal["closed"]:
            gap = gal["closed"] - gal["founded"]
            k = max(1, gap // 10)
            out.append(Clue(f"The venue where it was first shown closed more than {_num_words(k)} decade"
                            f"{'s' if k > 1 else ''} after it opened.", "shown_at", "gap", "gap", [k * 10], "range"))
    if w["stolen"]:
        sy, ry = w["stolen"]
        lo, hi, ph = _yr_range(rng, sy)
        out.append(Clue(f"It was stolen {ph} and later recovered.", "self", "stolen", "range", [lo, hi], "read"))
    if w["prize"]:
        z, zy = w["prize"]
        out.append(Clue(f"It won {W.ents[z]['name']}.", "self", "prize", "obj", [z], "literal",
                        [W.ents[z]["name"].lower().removeprefix("the ")]))
    if w["publisher"] is not None:
        pub = W.ents[w["publisher"]]
        if rng.random() < 0.4:
            out.append(Clue(f"It was published by {pub['name']}.", "self", "publisher", "eq", [w["publisher"]], "literal",
                            [pub["name"].lower().removeprefix("the ")]))
        lo, hi, ph = _yr_range(rng, pub["founded"])
        out.append(Clue(f"Its publisher was founded {ph}.", "publisher", "founded", "range", [lo, hi], "range"))
    S = _subj("work", "creator", w["kind"])
    out += person_clues(W, w["creator"], rng, S, "creator")
    if w["illustrator"] is not None:
        out += person_clues(W, w["illustrator"], rng, "the illustrator of its first edition", "illustrator")
    return out


def town_clues(W: World, tid: int, rng: random.Random) -> list[Clue]:
    t = W.ents[tid]
    out: list[Clue] = []
    lo, hi, ph = _cr(rng, t["founded"])
    out.append(Clue(f"It was founded {ph}.", "self", "founded", "range", [lo, hi], "range"))
    if t["river"]:
        if rng.random() < 0.5:
            out.append(Clue(f"It lies on the banks of {t['river']}.", "self", "river", "eq", [t["river"]], "literal",
                            [t["river"].lower().removeprefix("the ")]))
    pop, cy = t["pop"]
    lo = max(1000, int(pop * rng.uniform(0.75, 0.95)) // 1000 * 1000)
    hi = int(pop * rng.uniform(1.05, 1.3)) // 1000 * 1000 + 1000
    out.append(Clue(f"It had between {lo:,} and {hi:,} inhabitants at its {cy} census.", "self", "pop", "range", [lo, hi],
                    "read"))
    if t["landmark"]:
        ln, ly = t["landmark"]
        kind = ln.split()[-1].lower()
        lo2, hi2, ph = _cr(rng, ly)
        out.append(Clue(f"It is home to a {kind} built {ph}.", "self", "landmark", "range", [lo2, hi2], "read"))
    reg = W.ents[t["region"]]
    lo, hi, ph = _yr_range(rng, reg["formed"], (5, 15))
    out.append(Clue(f"The region it belongs to was formed {ph}.", "region", "formed", "range", [lo, hi], "range"))
    nat = t["natives"]
    if nat:
        p = W.ents[rng.choice(nat)]
        cat = L.TERM_CATEGORY[p["prof"]]
        out.append(Clue(f"One person born there worked as {cat}.", "natives", "prof", "cat", [cat], "hyper",
                        [x.lower() for x in L.PROFESSIONS[cat][0]]))
        if p["prizes"]:
            z, _ = p["prizes"][0]
            out.append(Clue(f"A person born there received {W.ents[z]['name']}.", "natives", "prize", "obj", [z], "literal",
                            [W.ents[z]["name"].lower().removeprefix("the ")]))
    return out


def org_clues(W: World, oid: int, rng: random.Random) -> list[Clue]:
    o = W.ents[oid]
    out: list[Clue] = []
    lo, hi, ph = _yr_range(rng, o["founded"])
    out.append(Clue(f"It was founded {ph}.", "self", "founded", "range", [lo, hi], "range"))
    if o["closed"]:
        lo, hi, ph = _yr_range(rng, o["closed"])
        out.append(Clue(f"It closed {ph}.", "self", "closed", "range", [lo, hi], "range"))
        k = max(1, (o["closed"] - o["founded"]) // 10)
        out.append(Clue(f"It closed more than {_num_words(k)} decade{'s' if k > 1 else ''} after it was founded.", "self",
                        "gap", "gap", [k * 10], "range"))
    t = W.ents[o["town"]]
    lo, hi, ph = _cr(rng, t["founded"])
    out.append(Clue(f"It was based in a town founded {ph}.", "located", "founded", "range", [lo, hi], "range"))
    if t["river"]:
        out.append(Clue(f"It was based in a town on the banks of {t['river']}.", "located", "river", "eq", [t["river"]],
                        "literal", [t["river"].lower().removeprefix("the ")]))
    if o["founder"] is not None:
        out += person_clues(W, o["founder"], rng, "its founder", "founder")
    return out


INTRO = {
    "person": ["I am trying to identify a person from the following clues.", "Identify the individual described below.",
               "There is a person who fits all of the following details."],
    "work": ["I am looking for {kind} that fits the following clues.", "Identify {kind} from these details.",
             "There is {kind} matching every one of the following clues."],
    "town": ["Name this town based on the following clues.", "I am trying to work out which town is described here."],
    "org": ["I am looking for {kind} that fits the following clues.", "Identify {kind} described below."],
}
_KIND_GENERIC = {"novel": "a book", "poetry collection": "a book", "children's book": "a book", "play": "a stage work",
                 "painting": "an artwork", "etching": "an artwork", "symphony": "a musical work", "opera": "a musical work",
                 "song cycle": "a musical work"}
_ORG_GENERIC = {"gallery": "an exhibition venue", "press": "a publishing house", "school": "a school",
                "society": "a society", "prize": "an award"}


@dataclass
class Question:
    target: int
    ttype: str
    clues: list[Clue]
    ask: str
    ask_text: str
    gold: str
    gold_entity: int
    text: str
    decoys: list[int]
    chain: list[int]
    n_alike: int = 0

    def to_hidden(self) -> dict:
        return {"target": self.target, "ttype": self.ttype, "clues": [asdict(c) for c in self.clues], "ask": self.ask,
                "ask_text": self.ask_text, "gold": self.gold, "gold_entity": self.gold_entity, "decoys": self.decoys,
                "chain": self.chain, "n_alike": self.n_alike}


def _asks(W: World, eid: int, rng: random.Random) -> list[tuple[str, str, str, int]]:
    """(ask kind, ask sentence, gold, gold entity) options for a target."""
    e = W.ents[eid]
    out = []
    t = e["type"]
    if t == "person":
        out += [("name", rng.choice(["What is this person's name?", "Who is this person?",
                                     "Can you tell me the name of this individual?"]), e["name"], eid)] * 3
        y, m, d = e["born"]
        out.append(("born_my", "In which month and year was this person born?", f"{L.MONTHS[m - 1]} {y}", eid))
        out.append(("birthplace", "In which town was this person born?", W.ents[e["town"]]["name"], e["town"]))
        if e["spouse"] is not None:
            out.append(("spouse", "What was the name of this person's spouse?", W.ents[e["spouse"]]["name"], e["spouse"]))
        out.append(("name_year", "Give this person's name and year of birth, separated by a semicolon.",
                    f"{e['name']}; {y}", eid))
    elif t == "work":
        out += [("name", rng.choice(["What is its title?", "What is the title of this work?",
                                     "Please provide the title."]), e["name"], eid)] * 3
        out.append(("creator", "Who created it?", W.ents[e["creator"]]["name"], e["creator"]))
        if e["illustrator"] is not None:
            out.append(("illustrator", "Who illustrated its first edition?", W.ents[e["illustrator"]]["name"], e["illustrator"]))
        if e["shown"] and W.ents[e["shown"][0]]["closed"]:
            g = e["shown"][0]
            out += [("venue_closed", "In what year did the venue where it was first shown close?",
                     str(W.ents[g]["closed"]), g)] * 2
        if e["publisher"] is not None:
            out.append(("publisher", "Which house first published it?", W.ents[e["publisher"]]["name"], e["publisher"]))
        out.append(("name_year", "Give its title and the year it appeared, separated by a semicolon.",
                    f"{e['name']}; {e['year']}", eid))
    elif t == "town":
        out += [("name", rng.choice(["What is the name of this town?", "Which town is it?"]), e["name"], eid)] * 3
        if e["landmark"]:
            out.append(("landmark", "What is the name of the landmark it is home to?", e["landmark"][0], eid))
    else:
        out += [("name", "What is its name?", e["name"], eid)] * 3
        if e["founder"] is not None:
            out.append(("founder", "Who founded it?", W.ents[e["founder"]]["name"], e["founder"]))
        if e["closed"]:
            out.append(("closed", "In what year did it close?", str(e["closed"]), eid))
    return out


def _realised(W: World, eid: int, c: Clue) -> bool:
    """Some document about an entity the clue is on contains one of the clue's literal forms verbatim."""
    if not c.forms:
        return False
    for x in resolve(W, eid, c.role.split(".")[-1]) if "." not in c.role else _resolve_path(W, eid, c.role):
        for di in W.ent_docs.get(x, [])[:6]:
            low = W.low[di]
            if any(f in low for f in c.forms):
                return True
    return False


def _resolve_path(W: World, eid: int, role: str) -> list[int]:
    cur = [eid]
    for r in role.split("."):
        cur = [y for x in cur for y in resolve(W, x, r)]
    return cur


def holds_path(W: World, eid: int, c: Clue) -> bool:
    if "." not in c.role:
        return holds(W, eid, c)
    base = c.role.split(".")
    cur = [eid]
    for r in base:
        cur = [y for x in cur for y in resolve(W, x, r)]
    cc = Clue(c.text, "self", c.attr, c.op, c.args, c.kind, c.forms)
    return any(holds(W, x, cc) for x in cur)


def make_question(W: World, rng: random.Random, n_clues: tuple[int, int] = (5, 7), ttype: str | None = None,
                  tries: int = 160, max_alike: int = 60, min_alike: int = 6, p_rare: float = 0.85,
                  p_no_literal: float = 0.3) -> Question:
    """At most one literal (proper-name) clue per question, so a name search can't single out the target directly."""
    weights = [("person", 0.42), ("work", 0.38), ("town", 0.1), ("org", 0.1)]
    for _ in range(tries):
        tt = ttype or rng.choices([w[0] for w in weights], [w[1] for w in weights])[0]
        pool = W.by_type[tt]
        eid = rng.choice(pool)
        e = W.ents[eid]
        if tt == "org" and e["kind"] == "prize":
            continue
        if tt == "person":
            cands = person_clues(W, eid, rng, "this person", "self")
        elif tt == "work":
            cands = work_clues(W, eid, rng)
        elif tt == "town":
            cands = town_clues(W, eid, rng)
        else:
            cands = org_clues(W, eid, rng)
        max_lit = 0 if rng.random() < p_no_literal else 1
        anchors = [c for c in cands if c.kind in ("literal", "hyper") and _realised(W, eid, c)
                   and (c.kind == "hyper" or max_lit)]
        if not anchors:
            continue
        rng.shuffle(anchors)
        anchors.sort(key=lambda c: 0 if c.kind == "literal" else 1)
        chosen = [anchors[0]]
        alike = sorted(_match_set(W, tt, pool, anchors[0]))
        for c in anchors[1:]:
            if len(alike) <= 40 or len(chosen) >= 3:
                break
            if any(c.attr == x.attr and c.role == x.role for x in chosen) or c.kind == "literal":
                continue
            nxt = [x for x in alike if holds_path(W, x, c)]
            if len(nxt) >= 3:
                chosen.append(c)
                alike = nxt
        if not (min_alike <= len(alike) <= max_alike):
            continue
        n_lit = sum(1 for c in chosen if c.kind == "literal")
        rest = [c for c in cands if c not in chosen and not (c.kind in ("literal", "hyper") and c.role == "self"
                                                               and len(chosen) >= 2)]
        rng.shuffle(rest)
        rest.sort(key=lambda c: 0 if (c.role != "self" and rng.random() < 0.5) else 1)
        lo_k, hi_k = n_clues
        target_k = rng.randint(lo_k, hi_k)
        alive = list(alike)
        per_role: dict[str, int] = {}
        for c in chosen:
            per_role[c.role.split(".")[0]] = per_role.get(c.role.split(".")[0], 0) + 1
        for c in rest:
            if len(chosen) >= target_k and len(alive) == 1:
                break
            if len(chosen) >= hi_k:
                break
            r0 = c.role.split(".")[0]
            if per_role.get(r0, 0) >= 3 or any(c.attr == x.attr and c.role == x.role and c.op == x.op for x in chosen):
                continue
            if c.kind == "literal" and n_lit >= max_lit:
                continue
            ok = [x for x in alive if holds_path(W, x, c)]
            if not ok or (len(ok) == len(alive) and len(alive) > 1 and len(chosen) >= lo_k):
                continue
            if len(ok) == 1 and len(alive) > 2 and len(chosen) < lo_k - 1:
                continue
            chosen.append(c)
            per_role[r0] = per_role.get(r0, 0) + 1
            n_lit += c.kind == "literal"
            alive = ok
        if len(chosen) < lo_k or alive != [eid]:
            continue
        if any(x != eid and all(holds_path(W, x, c) for c in chosen) for x in alike):
            continue
        decoys = [x for x in alike if x != eid and sum(1 for c in chosen if not holds_path(W, x, c)) == 1]
        n_alike = len(alike) - 1
        rare = rare_anchor(W, eid, chosen)
        if rare is None and rng.random() < p_rare:
            continue
        asks = _asks(W, eid, rng)
        ask, ask_text, gold, gold_eid = rng.choice(asks)
        chain = [eid]
        for c in chosen:
            if c.role != "self":
                cc = Clue(c.text, "self", c.attr, c.op, c.args, c.kind, c.forms)
                chain += [y for y in _resolve_path(W, eid, c.role) if holds(W, y, cc) and y not in chain]
        if gold_eid not in chain:
            chain.append(gold_eid)
        rng.shuffle(chosen)
        kind = _KIND_GENERIC.get(e.get("kind", ""), "a work") if tt == "work" else _ORG_GENERIC.get(e.get("kind", ""), "an institution")
        intro = rng.choice(INTRO[tt]).format(kind=kind)
        if rng.random() < 0.5:
            body = " ".join(_cap(c.text) for c in chosen)
            text = f"{intro} {body} {ask_text}"
        else:
            body = "\n".join(f"- {_cap(c.text)}" for c in chosen)
            text = f"{intro}\n{body}\n{ask_text}"
        return Question(eid, tt, chosen, ask, ask_text, gold, gold_eid, text, decoys[:40], chain, n_alike)
    raise RuntimeError("could not generate a question")


_MATCH: dict[tuple, frozenset] = {}


def _match_set(W: World, tt: str, pool: list[int], c: Clue) -> frozenset:
    """Entities of type ``tt`` satisfying clue ``c``, cached per world (anchor clues recur across questions)."""
    key = (W.wid, W.scale, tt, c.role, c.attr, c.op, json.dumps(c.args))
    v = _MATCH.get(key)
    if v is None:
        v = frozenset(x for x in pool if holds_path(W, x, c))
        if len(_MATCH) > 4096:
            _MATCH.clear()
        _MATCH[key] = v
    return v


_POOLS: dict[tuple[str, str], object] = {}


def form_pool(W: World, form: str):
    """Documents containing a literal form (the root's own REPL search): a set if <= 200, else the count."""
    key = (W.wid + W.scale, form)
    v = _POOLS.get(key)
    if v is None:
        hit = [j for j, d in enumerate(W.low) if form in d]
        v = set(hit) if len(hit) <= 200 else len(hit)
        _POOLS[key] = v
    return v


def rare_anchor(W: World, eid: int, chosen: list[Clue], cap: int = 60) -> Clue | None:
    """A searchable clue whose documents (<= cap) include an article of the target or of the entity it is about."""
    for c in chosen:
        if c.kind not in ("literal", "hyper") or not c.forms:
            continue
        docs: set[int] = set()
        size = 0
        for f in c.forms:
            v = form_pool(W, f)
            if isinstance(v, int):
                size = cap + 1
                break
            docs |= v
            size = len(docs)
        if size > cap:
            continue
        ents = [eid] if c.role == "self" else _resolve_path(W, eid, c.role)
        arts = {d for e in ents for d in W.ent_docs.get(e, []) if W.doc_subject[d] == e}
        if docs & arts:
            return c
    return None


def _count_upto(W: World, pool: list[int], c: Clue, k: int) -> int:
    n = 0
    for x in pool:
        if holds_path(W, x, c):
            n += 1
            if n >= k:
                break
    return n


def _cap(s: str) -> str:
    return s[0].upper() + s[1:]


__all__ = ["Clue", "Question", "holds", "holds_path", "kg_value", "make_question", "resolve", "value_holds"]
