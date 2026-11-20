"""A synthetic encyclopedia: a random knowledge graph rendered as ~24k BrowseComp-Plus-style documents.

Every fact sentence is recorded with the facts it states, so the oracle answers from the graph, but only for
sentences pasted into a sub-call. People come in clusters of near-twins, so every target has decoys. A world is
built once per process and written once per machine to a doc store under ``$BCP_SYNTH_STORE_ROOT``.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synth_common.segindex import SegmentIndex

from . import lexicon as L

GEN_VERSION = "w6"


SCALES = {
    "full": (900, 1000, 50, 100, 100, 220, 130, 75, 4000),
    "medium": (320, 360, 20, 36, 36, 80, 48, 26, 1400),
    "small": (110, 130, 10, 14, 14, 28, 16, 10, 450),
}


@dataclass
class World:
    wid: str
    scale: str
    low: list[str] = field(default_factory=list)
    doc_ids: list[str] = field(default_factory=list)
    doc_kind: list[str] = field(default_factory=list)
    doc_subject: list[int] = field(default_factory=list)
    ents: list[dict] = field(default_factory=list)
    by_type: dict[str, list[int]] = field(default_factory=dict)
    name_index: dict[str, int] = field(default_factory=dict)
    facts: list[tuple] = field(default_factory=list)
    fact_doc: list[int] = field(default_factory=list)
    fact_sent: list[str] = field(default_factory=list)
    ent_facts: dict[int, list[int]] = field(default_factory=dict)
    ent_docs: dict[int, list[int]] = field(default_factory=dict)
    sent_facts: dict[str, tuple] = field(default_factory=dict)
    index: SegmentIndex | None = None
    titles: list[str] = field(default_factory=list)
    store_path: str = ""
    store_chars: list[int] = field(default_factory=list)

    def doc(self, i: int) -> str:
        """Original-case text of document ``i`` (read from the doc store; only the lower-cased copy stays in memory)."""
        from rlm_train import doc_store
        return doc_store.open_store(self.store_path).doc(i)
    build_s: float = 0.0
    sent_hashes: set = field(default_factory=set)


def _word(rng: random.Random, parts: int = 2) -> str:
    s = rng.choice(L.SYL_A)
    if parts >= 3:
        s += rng.choice(L.SYL_B)
    s += rng.choice(L.SYL_C)
    return s.capitalize()


def _unique(rng: random.Random, n: int, make, taken: set[str]) -> list[str]:
    out = []
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        w = make(rng)
        if w.lower() in taken:
            continue
        taken.add(w.lower())
        out.append(w)
    return out


_TITLE_A = ["The", "A", "The", "Songs of the", "Letters from the", "The Last", "Under the", "Beyond the", "The Silent",
            "The Little", "Winter at the", "The Long", "Night of the", "Return to the", "The Hollow", "The Iron"]
_TITLE_N = ["Lantern", "Harbour", "Orchard", "Tide", "Weir", "Bell", "Meadow", "Mirror", "Quarry", "Ferry", "Garden",
            "Tower", "Lighthouse Keeper", "Glasshouse", "Saltmarsh", "Clockmaker", "Ropewalk", "Granary", "Chapel",
            "Ledger", "Wintering", "Crossing", "Pilgrim", "Embers", "Vineyard", "Weaver", "Signal", "Kingfisher",
            "Ashes", "Bridge", "Mill", "Shore", "Archive", "Comet", "Orchid", "Falcon", "Shepherd", "Lamplighter"]
_TITLE_ADJ = ["Amber", "Crimson", "Quiet", "Distant", "Northern", "Sunken", "Painted", "Broken", "Golden", "Patient",
              "Hidden", "Burning", "Seventh", "Lost", "Bitter", "Gentle", "Wandering", "Frozen", "Salt", "Velvet"]


def _title(rng: random.Random, place: str) -> str:
    r = rng.random()
    if r < 0.35:
        return f"{rng.choice(_TITLE_A)} {rng.choice(_TITLE_ADJ)} {rng.choice(_TITLE_N)}"
    if r < 0.6:
        return f"{rng.choice(_TITLE_A)} {rng.choice(_TITLE_N)} of {place}"
    if r < 0.8:
        return f"{rng.choice(_TITLE_ADJ)} {rng.choice(_TITLE_N)}"
    return f"{rng.choice(['A Summer in', 'The Road to', 'Evenings in', 'Memories of', 'A Winter in'])} {place}"


def _kg(rng: random.Random, scale: str) -> tuple[list[dict], dict[str, list[int]]]:
    n_clusters, n_towns, n_regions, n_gal, n_press, n_school, n_soc, n_prize, _ = SCALES[scale]
    taken: set[str] = set()
    ents: list[dict] = []
    by: dict[str, list[int]] = {k: [] for k in ("person", "work", "town", "org", "region")}

    def add(e: dict) -> int:
        e["id"] = len(ents)
        ents.append(e)
        by[e["type"]].append(e["id"])
        return e["id"]

    rivers = _unique(rng, max(30, n_towns // 7), lambda r: _word(r, 2), taken)
    for nm in _unique(rng, n_regions, lambda r: _word(r, 3), taken):
        add({"type": "region", "name": f"{nm} Valley" if rng.random() < 0.4 else nm, "formed": rng.randint(1850, 2015)})
    regions = by["region"]
    for nm in _unique(rng, n_towns, lambda r: _word(r, rng.choice((2, 3))), taken):
        founded = rng.randint(1080, 1890)
        fort = None
        if rng.random() < 0.35:
            fort = (f"{_word(rng, 2)} {rng.choice(['Fort', 'Castle', 'Abbey', 'Tower'])}",
                    rng.randint(max(1000, founded - 200), min(1900, founded + 300)))
            taken.add(fort[0].lower())
        add({"type": "town", "name": nm, "founded": founded, "region": rng.choice(regions),
             "river": f"the {rng.choice(rivers)}" if rng.random() < 0.7 else None,
             "pop": (rng.randint(2, 180) * 1000 + rng.randint(0, 999), rng.choice((2001, 2011, 2021))),
             "landmark": fort, "natives": []})
    towns = by["town"]

    for kind, n in (("gallery", n_gal), ("press", n_press), ("school", n_school), ("society", n_soc),
                    ("prize", n_prize)):
        for stem in _unique(rng, n, lambda r: _word(r, rng.choice((2, 3))), taken):
            name = rng.choice(L.ORG_KINDS[kind]).format(x=stem)
            founded = rng.randint(1700, 1990)
            closed = None
            if kind in ("gallery", "press", "society", "school") and rng.random() < 0.45:
                closed = founded + rng.randint(4, 110)
                if closed > 2019:
                    closed = None
            add({"type": "org", "kind": kind, "name": name, "founded": founded, "closed": closed,
                 "town": rng.choice(towns), "founder": None, "members": []})
    orgs = {k: [o for o in by["org"] if ents[o]["kind"] == k] for k in ("gallery", "press", "school", "society", "prize")}

    cats = list(L.PROFESSIONS)
    creative_cats = list(L.CREATIVE)
    used_names: set[str] = set()
    surnames = _unique(rng, max(400, n_clusters), lambda r: _word(r, rng.choice((2, 2, 3))), taken)

    def person_name(gender: str, surname: str) -> str:
        for _ in range(40):
            first = rng.choice(L.FIRST_F if gender == "f" else L.FIRST_M)
            nm = f"{first} {surname}"
            if nm.lower() not in used_names:
                used_names.add(nm.lower())
                return nm
        nm = f"{rng.choice(L.FIRST_F if gender == 'f' else L.FIRST_M)} {rng.choice(L.FIRST_M)} {surname}"
        used_names.add(nm.lower())
        return nm

    for _ in range(n_clusters):
        cat = rng.choice(creative_cats) if rng.random() < 0.62 else rng.choice(cats)
        center = rng.randint(1790, 1985)
        region = rng.choice(regions)
        rtowns = [t for t in towns if ents[t]["region"] == region] or towns
        fcat, mcat = rng.choice(cats), rng.choice(cats)
        gal, press = rng.choice(orgs["gallery"]), rng.choice(orgs["press"])
        school, soc = rng.choice(orgs["school"]), rng.choice(orgs["society"])
        prize = rng.choice(orgs["prize"])
        sur_pool = rng.sample(surnames, 3)
        for _ in range(rng.randint(5, 8)):
            g = rng.choice("fm")
            terms = L.PROFESSIONS[cat][0]
            by_ = max(1700, min(1995, center + rng.randint(-7, 7)))
            p = {"type": "person", "gender": g, "name": person_name(g, rng.choice(sur_pool)), "cat": cat,
                 "prof": rng.choice(terms), "born": (by_, rng.randint(1, 12), rng.randint(1, 28)),
                 "town": rng.choice(rtowns) if rng.random() < 0.75 else rng.choice(towns),
                 "father": rng.choice(L.PROFESSIONS[fcat if rng.random() < 0.7 else rng.choice(cats)][0]),
                 "mother": rng.choice(L.PROFESSIONS[mcat if rng.random() < 0.6 else rng.choice(cats)][0]),
                 "nsib": rng.choice((0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7)), "sibdeath": None, "spouse": None,
                 "married": None, "school": school if rng.random() < 0.55 else rng.choice(orgs["school"]),
                 "prizes": [], "exhib": [], "member": [], "works": [], "illustrated": [], "died": None,
                 "nworks": None, "debut": by_ + rng.randint(16, 30)}
            if p["nsib"] and rng.random() < 0.45:
                p["sibdeath"] = by_ + rng.randint(3, 60)
            if rng.random() < 0.72 and by_ < 1960:
                dy = by_ + rng.randint(28, 96)
                if dy <= 2022:
                    p["died"] = dy
            if rng.random() < 0.35:
                lo = by_ + rng.randint(18, 45)
                if lo < 2015:
                    p["prizes"].append((prize if rng.random() < 0.6 else rng.choice(orgs["prize"]), lo))
            if cat == "a visual artist" or rng.random() < 0.08:
                for _ in range(rng.randint(1, 3)):
                    y = by_ + rng.randint(20, 55)
                    if y <= 2020:
                        p["exhib"].append((gal if rng.random() < 0.7 else rng.choice(orgs["gallery"]), y))
            if rng.random() < 0.35:
                p["member"].append((soc if rng.random() < 0.6 else rng.choice(orgs["society"]),
                                    by_ + rng.randint(20, 50), rng.randint(1, 12)))
            if cat in L.CREATIVE:
                y1 = by_ + rng.randint(19, 32)
                p["nworks"] = (rng.randint(3, 30), y1, y1 + rng.randint(5, 25))
                p["press"] = press if rng.random() < 0.65 else rng.choice(orgs["press"])
            add(p)
    people = by["person"]
    for pid in people:
        ents[ents[pid]["town"]]["natives"].append(pid)

    singles = [p for p in people if rng.random() < 0.5]
    rng.shuffle(singles)
    for a, b in zip(singles[::2], singles[1::2], strict=False):
        pa, pb = ents[a], ents[b]
        if pa["spouse"] is None and pb["spouse"] is None and abs(pa["born"][0] - pb["born"][0]) < 15:
            y = max(pa["born"][0], pb["born"][0]) + rng.randint(19, 40)
            pa["spouse"], pb["spouse"], pa["married"], pb["married"] = b, a, y, y

    illustrators = [p for p in people if ents[p]["cat"] == "a visual artist"]
    title_places = [ents[t]["name"] for t in towns]
    for pid in people:
        p = ents[pid]
        if p["cat"] not in L.CREATIVE:
            continue
        for _ in range(rng.choice((1, 1, 2, 2, 3))):
            kind = rng.choice(L.CREATIVE[p["cat"]])
            y = p["born"][0] + rng.randint(21, 58)
            if y > 2021:
                continue
            for _ in range(20):
                t = _title(rng, rng.choice(title_places))
                if t.lower() not in taken:
                    break
            taken.add(t.lower())
            tcat = rng.choice(list(L.THEMES))
            w = {"type": "work", "kind": kind, "name": t, "year": y, "creator": pid, "illustrator": None,
                 "publisher": None, "theme": rng.choice(L.THEMES[tcat]), "reception": "poor" if rng.random() < 0.3
                 else "good", "shown": None, "stolen": None, "prize": None}
            if kind in ("novel", "poetry collection", "children's book", "play"):
                w["publisher"] = p.get("press") or rng.choice(orgs["press"])
                if kind != "play" and rng.random() < 0.55 and illustrators:
                    ill = rng.choice(illustrators)
                    if ill != pid:
                        w["illustrator"] = ill
            if kind in ("painting", "etching") and rng.random() < 0.6:
                gal = p["exhib"][0][0] if p["exhib"] and rng.random() < 0.6 else rng.choice(orgs["gallery"])
                w["shown"] = (gal, y + rng.randint(0, 12), rng.randint(1, 7))
            if kind == "painting" and rng.random() < 0.12 and y < 2000:
                sy = rng.randint(max(y + 10, 1900), 2012)
                w["stolen"] = (sy, sy + rng.randint(1, 18))
            if rng.random() < 0.1:
                w["prize"] = (rng.choice(orgs["prize"]), y + rng.randint(0, 2))
            wid = add(w)
            p["works"].append(wid)
            if w["illustrator"] is not None:
                ents[w["illustrator"]]["illustrated"].append(wid)

    used: dict[int, list[int]] = {}
    for pid in people:
        p = ents[pid]
        used.setdefault(p["school"], []).append(p["born"][0] + 10)
        for z, y in p["prizes"]:
            used.setdefault(z, []).append(y)
        for g, y in p["exhib"]:
            used.setdefault(g, []).append(y)
        for s_, y, _m in p["member"]:
            used.setdefault(s_, []).append(y)
    for wid_ in by["work"]:
        w = ents[wid_]
        if w["publisher"] is not None:
            used.setdefault(w["publisher"], []).append(w["year"])
        if w["shown"]:
            used.setdefault(w["shown"][0], []).append(w["shown"][1])
        if w["prize"]:
            used.setdefault(w["prize"][0], []).append(w["prize"][1])
    for oid in by["org"]:
        o = ents[oid]
        ys = used.get(oid)
        if ys:
            first, last = min(ys), max(ys)
            o["founded"] = first - rng.randint(1, 45)
            if o["closed"] is not None:
                c = last + rng.randint(1, 40)
                o["closed"] = c if c <= 2019 else None
    for oid in by["org"]:
        o = ents[oid]
        cands = [p for p in rng.sample(people, min(40, len(people)))
                 if ents[p]["born"][0] + 25 <= o["founded"] <= ents[p]["born"][0] + 70]
        if cands and rng.random() < 0.8:
            o["founder"] = cands[0]
    for pid in people:
        for g, _y in ents[pid]["exhib"]:
            ents[g]["members"].append(pid)
        for s, _y, _m in ents[pid]["member"]:
            ents[s]["members"].append(pid)
    return ents, by


class _Doc:
    __slots__ = ("title", "kind", "subject", "paras", "facts", "own")

    def __init__(self, title: str, kind: str, subject: int):
        self.title, self.kind, self.subject = title, kind, subject
        self.paras: list[list[str]] = [[]]
        self.facts: list[tuple[str, list[tuple]]] = []
        self.own: list[str] = []

    def say(self, sent: str, *facts: tuple) -> None:
        sent = sent[0].upper() + sent[1:]
        if len(sent) < 30:
            sent = sent[:-1] + ", according to local records."
        self.paras[-1].append(sent)
        if facts:
            self.facts.append((sent, list(facts)))
        else:
            self.own.append(sent)

    def pad(self, sents: list[str]) -> None:
        self.paras[-1].extend(sents)

    def para(self) -> None:
        if self.paras[-1]:
            self.paras.append([])


def _pro(p: dict) -> tuple[str, str, str]:
    return ("she", "her", "her") if p["gender"] == "f" else ("he", "his", "him")


def _a(word: str) -> str:
    return ("an " if word[0].lower() in "aeiou" else "a ") + word


def _num(rng: random.Random, n: int) -> str:
    return L.NUMWORDS[n] if n < len(L.NUMWORDS) and rng.random() < 0.6 else str(n)


def _plural(w: str) -> str:
    if w.endswith("y") and w[-2:] not in ("ay", "ey", "oy"):
        return w[:-1] + "ies"
    if w.endswith(("s", "sh", "ch", "x")):
        return w + "es"
    return w + "s"


def _ord(d: int) -> str:
    return f"{d}{'th' if 10 <= d % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def _date(rng: random.Random, d: int, m: int, y: int) -> str:
    r = rng.random()
    if r < 0.4:
        return f"{d} {L.MONTHS[m - 1]} {y}"
    if r < 0.75:
        return f"{L.MONTHS[m - 1]} {d}, {y}"
    return f"the {_ord(d)} of {L.MONTHS[m - 1]} {y}"


_OWN = [
    "{P} kept a notebook of walks along {R}, filled with sketches of {thing}s.",
    "A letter from {y} describes the delight of {P} at a visit to {T} in the {season}.",
    "Friends remembered the habit {P} had of rising before dawn to work by lamplight in {T}.",
    "In later years {P} spoke warmly of a {season} spent near {T}.",
    "{P} corresponded for decades with a cousin who lived in {T}.",
    "The family home of {P} in {T} was later converted into a small {bldg}.",
    "Several letters written by {P} are preserved in the archive at {T}.",
    "An anecdote has {P} buying a second-hand {thing} from a market stall in {T} in {y}.",
    "Contemporaries described {P} as reserved but generous with younger colleagues in {T}.",
    "{P} travelled widely, and an account of a journey to {T} appeared in {y}.",
    "A portrait of {P} by an unknown hand hangs in the town hall of {T}.",
    "Among the acquaintances of {P} in {T} was a {term} who later became a close friend.",
    "In the {season} of {y} {P} rented rooms above a {bldg} in {T}.",
    "A walking stick said to have belonged to {P} is displayed in a {bldg} in {T}.",
]
_PLACE_OWN = [
    "Visitors to {N} often remark on the {thing}s sold at the {season} market.",
    "A guidebook of {y} praised {N} for its quiet streets and its old {bldg}.",
    "The archive of {N} holds letters about a dispute over a {thing} in {y}.",
    "In {y} a storm damaged the {bldg} associated with {N}.",
    "Local accounts from {y} mention a {term} who settled near {N} and opened a {bldg}.",
    "A photograph of {N} taken in the {season} of {y} shows crowds near the {bldg}.",
]
_BANK = [
    "In {y} the council debated the cost of repairing the {bldg} near {R}.",
    "The {season} of {y} was long remembered in {T} for its hard frosts and a late harvest.",
    "Travellers on the road to {T} often stopped at an inn kept by a retired {term}.",
    "{R} flooded in {y}, damaging mills, bridges and several {bldgs} along its banks.",
    "Market records from {y} list the prices of wool, salt and {things} in some detail.",
    "A collection of {things} assembled over many years was sold at auction in {y}.",
    "The parish registers survive from {y} onwards, with gaps during the years of unrest.",
    "By {y} the population of the surrounding district had grown by nearly {n} per cent.",
    "A small museum in {T} displays {things}, maps and photographs of the old quarter.",
    "Contemporary newspapers devoted several columns to the opening of a new {bldg} in {y}.",
    "The local {term} was said to know every family in the valley by name.",
    "A railway line reached the area in {y}, bringing visitors and new trades.",
    "Evening lectures on history and natural science drew large audiences in the {season}.",
    "The surviving correspondence is scattered across three archives and has never been fully catalogued.",
    "Accounts differ on the details, and some of the stories told later are plainly embellished.",
    "A fire in {y} destroyed part of the old town, including a {bldg} and two warehouses.",
    "The {season} fair attracted traders from as far away as {T}.",
    "Older residents still recall the great storm of {y} and the damage it caused along {R}.",
    "A society for the preservation of old buildings was formed in {y} after a long campaign.",
    "The roads were poor until {y}, and most goods travelled by cart or by river barge.",
    "{A_term} from {T} wrote a short memoir of the period, published privately in {y}.",
    "School attendance rose sharply after {y}, when a second schoolroom was added.",
    "The harvest of {y} was so poor that grain had to be brought in from {T}.",
    "The surrounding hills were planted with orchards and the occasional vineyard.",
    "Disputes over water rights on {R} were settled only after years of argument.",
    "The local brass band gave its first concert in {y} and played every {season} for decades.",
    "A map of {y} shows the old market square, the {bldg} and a cluster of workshops.",
    "Many families emigrated in the hard years after {y}, and some later returned.",
    "A photograph from {y} shows a crowded street outside the {bldg} on market day.",
    "The archives also hold an inventory of {things} left by {a_term} who died in {y}.",
    "In the {season} of {y} an exhibition of local crafts filled the hall for a fortnight.",
    "A ferry crossed {R} until a stone bridge was finished in {y}.",
    "The theme of the {theme} recurs in local carvings, songs and inn signs.",
    "A visiting {term} described the place in {y} as quiet, prosperous and rather dull.",
    "A cholera outbreak in {y} led to the building of new wells and a drainage scheme.",
    "Tourism grew slowly, helped by a guidebook of {y} that praised the walks along {R}.",
    "Records from {y} mention a long quarrel between two families over a {thing}.",
    "The surrounding countryside is known for its hedges, sheep pastures and old stone walls.",
]
_THINGS = ["barometer", "telescope", "writing desk", "mandolin", "sea chest", "atlas", "pocket watch", "lantern",
           "chess set", "fiddle", "travelling clock", "spyglass", "herbarium", "violin case", "map case", "music box",
           "printing block", "weather vane", "hand loom", "brass compass"]
_BLDGS = ["museum", "library", "guest house", "school", "music room", "bakery", "chapel", "granary", "mill", "town hall"]
_SEASONS = ["spring", "summer", "autumn", "winter"]


class _Renderer:
    def __init__(self, ents: list[dict], rng: random.Random, towns: list[int], scale: str):
        self.E, self.rng, self._towns = ents, rng, towns
        self.docs: list[_Doc] = []
        self.terms = list(L.TERM_CATEGORY)
        themes = [t for ts in L.THEMES.values() for t in ts]
        bank: set[str] = set()
        n_bank = {"full": 40_000, "medium": 16_000, "small": 6_000}[scale]
        guard = 0
        while len(bank) < n_bank and guard < n_bank * 4:
            guard += 1
            t = ents[rng.choice(towns)]
            term, thing, bldg = rng.choice(self.terms), rng.choice(_THINGS), rng.choice(_BLDGS)
            sent = rng.choice(_BANK).format(
                y=rng.randint(1750, 2018), bldg=bldg, bldgs=_plural(bldg), season=rng.choice(_SEASONS), thing=thing,
                things=_plural(thing), T=t["name"], R=t.get("river") or "the river", term=term, a_term=_a(term),
                A_term=_a(term).capitalize(), n=rng.randint(3, 60), theme=rng.choice(themes))
            bank.add(sent[0].upper() + sent[1:])
        self.bank = sorted(bank)
        rng.shuffle(self.bank)

    def nm(self, eid: int) -> str:
        return self.E[eid]["name"]

    def pad(self, d: _Doc, lo: int = 4, hi: int = 8) -> None:
        r = self.rng
        k = r.randint(lo, hi)
        i = r.randrange(len(self.bank))
        d.pad([self.bank[(i + 7919 * j) % len(self.bank)] for j in range(k)])

    def own(self, d: _Doc, p: dict | None, k: int = 1) -> None:
        rng, E = self.rng, self.E
        for _ in range(k):
            t = E[rng.choice(self._towns)]
            kw = dict(R=t.get("river") or f"the road to {t['name']}", thing=rng.choice(_THINGS), y=rng.randint(1780, 2015),
                      T=t["name"], season=rng.choice(_SEASONS), bldg=rng.choice(_BLDGS), term=rng.choice(self.terms))
            if p is not None:
                d.say(rng.choice(_OWN).format(P=p["name"], **kw))
            else:
                d.say(rng.choice(_PLACE_OWN).format(N=d.title, **kw))

    def person(self, pid: int) -> None:
        rng, E = self.rng, self.E
        p = E[pid]
        pro, pos, obj = _pro(p)
        nm = p["name"]
        by_, bm, bd = p["born"]
        town = E[p["town"]]
        d = _Doc(nm, "article", pid)
        life = f"{by_}–{p['died']}" if p["died"] else f"born {by_}"
        lf = [(pid, "born", (by_, None, None))] + ([(pid, "died", p["died"])] if p["died"] else [])
        r = rng.random()
        if r < 0.35:
            d.say(f"{nm} ({life}) was {_a(p['prof'])} from {town['name']}.", (pid, "prof", p["prof"]),
                  (pid, "town", p["town"]), *lf)
        elif r < 0.7:
            d.say(f"{nm} ({life}) was known chiefly as {_a(p['prof'])}, and spent much of {pos} life away from "
                  f"{pos} native {town['name']}.", (pid, "prof", p["prof"]), (pid, "town", p["town"]), *lf)
        else:
            d.say(f"{nm} ({life}) is remembered today mostly through the recollections of friends and a handful of "
                  f"surviving papers.", *lf)
            self.pad(d, 2, 6)
            d.say(rng.choice([f"By training and by temperament {nm} was {_a(p['prof'])}.",
                              f"{nm} {rng.choice(L.PROFESSIONS[p['cat']][1])}, and never seriously considered another "
                              f"career.",
                              f"It was as {_a(p['prof'])} that {nm} made a name in and around {town['name']}."]),
                  (pid, "prof", p["prof"]))
        self.pad(d, 6, 12)
        d.para()
        bf = (pid, "born", p["born"])
        r = rng.random()
        if r < 0.4:
            riv = town["river"] and rng.random() < 0.5
            d.say(f"{nm} was born on {_date(rng, bd, bm, by_)} in {town['name']}" +
                  (f", a town on {town['river']}." if riv else "."), bf, (pid, "town", p["town"]),
                  *([(p["town"], "river", town["river"])] if riv else []))
        elif r < 0.7:
            d.say(f"{nm} came into the world in {town['name']} on {_date(rng, bd, bm, by_)}.", bf, (pid, "town", p["town"]))
        else:
            d.say(f"The birth of {nm}, on {_date(rng, bd, bm, by_)}, was recorded in the parish book of {town['name']}.",
                  bf, (pid, "town", p["town"]))
        self.pad(d, 6, 14)
        d.para()
        fam = []
        fa, mo = p["father"], p["mother"]
        r = rng.random()
        if r < 0.3:
            fam.append((f"{nm}'s father was {_a(fa)} and {pos} mother {_a(mo)}.", (pid, "father", fa), (pid, "mother", mo)))
        elif r < 0.5:
            fam.append((f"{nm} was the child of {_a(fa)} and {_a(mo)} in {town['name']}.", (pid, "father", fa),
                        (pid, "mother", mo)))
        else:
            fp = rng.choice(L.PROFESSIONS[L.TERM_CATEGORY[fa]][1])
            mp = rng.choice(L.PROFESSIONS[L.TERM_CATEGORY[mo]][1])
            if rng.random() < 0.5:
                fam.append((f"The father of {nm} {fp}, while the mother worked as {_a(mo)}.", (pid, "father", fa),
                            (pid, "mother", mo)))
            else:
                fam.append((f"In the household where {nm} grew up the father {fp}.", (pid, "father", fa)))
                fam.append((f"The mother of {nm} {mp} for most of her working life.", (pid, "mother", mo)))
        n = p["nsib"]
        r = rng.random()
        if r < 0.3:
            fam.append((f"{nm} grew up with {_num(rng, n)} {'sibling' if n == 1 else 'siblings'}." if n else
                        f"{nm} was an only child.", (pid, "nsib", n)))
        elif r < 0.6 and n + 1 < len(L.ORDINALS):
            fam.append((f"{nm} was the {L.ORDINALS[rng.randint(1, n + 1)]} of {_num(rng, n + 1)} children." if n else
                        f"{nm} grew up as the only child of the household.", (pid, "nsib", n)))
        elif r < 0.8:
            fam.append((f"The household included {_num(rng, n)} other children besides {nm}." if n else
                        f"{nm} had no brothers or sisters.", (pid, "nsib", n)))
        else:
            fam.append((f"There were {_num(rng, n + 1)} children in the family of {nm}, counting {obj}." if n else
                        f"No other children were born to the parents of {nm}.", (pid, "nsib", n)))
        if p["sibdeath"]:
            sib = rng.choice(["brother", "sister"])
            fam.append((rng.choice([f"The death of {nm}'s {sib} in {p['sibdeath']} affected {obj} deeply.",
                                    f"{nm} lost a younger {sib} in {p['sibdeath']}, an event {pro} rarely spoke about.",
                                    f"{nm} buried a {sib} in {p['sibdeath']}, the first of several family losses."]),
                        (pid, "sibdeath", p["sibdeath"])))
        rng.shuffle(fam)
        for sent, *fs in fam:
            d.say(sent, *fs)
            if rng.random() < 0.5:
                self.pad(d, 2, 6)
        self.pad(d, 4, 10)
        d.para()
        sc = E[p["school"]]
        d.say(rng.choice([f"{nm} was educated at {sc['name']}.", f"{nm} attended {sc['name']} as a pupil.",
                          f"The schooling of {nm} took place at {sc['name']}.",
                          f"For several years {nm} was a pupil at {sc['name']}."]), (pid, "school", p["school"]))
        self.pad(d, 4, 10)
        r = rng.random()
        if r < 0.35:
            d.say(f"{nm} began working as {_a(p['prof'])} in {p['debut']}.", (pid, "debut", p["debut"]))
        elif r < 0.6:
            d.say(f"{nm} turned professional in {p['debut']}.", (pid, "debut", p["debut"]))
        elif r < 0.8:
            d.say(f"The professional career of {nm} began in earnest in {p['debut']}.", (pid, "debut", p["debut"]))
        self.own(d, p)
        self.pad(d, 4, 10)
        d.para()
        if p["nworks"]:
            n, y1, y2 = p["nworks"]
            noun = {"a writer": "books", "a visual artist": "paintings", "a composer": "scores"}[p["cat"]]
            d.say(rng.choice([f"Between {y1} and {y2} {nm} published {_num(rng, n)} {noun} under {pos} own name.",
                              f"From {y1} to {y2} {nm} completed {_num(rng, n)} {noun}.",
                              f"In the years {y1} to {y2} no fewer than {_num(rng, n)} {noun} by {nm} appeared."]),
                  (pid, "nworks", p["nworks"]))
        for wid in p["works"]:
            w = E[wid]
            d.say(rng.choice([f"Among the works of {nm} is '{w['name']}' ({w['year']}).",
                              f"The {w['kind']} '{w['name']}' by {nm} appeared in {w['year']}.",
                              f"In {w['year']} {nm} completed '{w['name']}', {_a(w['kind'])}."]),
                  (pid, "created", wid), (wid, "year", w["year"]), (wid, "creator", pid))
            if rng.random() < 0.5:
                self.pad(d, 2, 4)
        for wid in p["illustrated"]:
            w = E[wid]
            d.say(f"{nm} provided the illustrations for {self.nm(w['creator'])}'s '{w['name']}' ({w['year']}).",
                  (pid, "illustrated", wid), (wid, "illustrator", pid), (wid, "year", w["year"]))
        self.pad(d, 4, 10)
        d.para()
        for g, y in p["exhib"]:
            d.say(rng.choice([f"{nm} exhibited at {self.nm(g)} in {y}.", f"Work by {nm} was shown at {self.nm(g)} in {y}.",
                              f"In {y} {nm} showed several pictures at {self.nm(g)}."]), (pid, "exhib", (g, y)))
        for s_, y, m in p["member"]:
            d.say(f"In {L.MONTHS[m - 1]} {y} {nm} was elected a member of {self.nm(s_)}.", (pid, "member", (s_, y, m)))
        for z, y in p["prizes"]:
            d.say(rng.choice([f"{nm} received {self.nm(z)} in {y}.", f"In {y} {nm} was awarded {self.nm(z)}."]),
                  (pid, "prize", (z, y)))
        self.pad(d, 4, 12)
        if p["spouse"] is not None:
            d.para()
            sp = E[p["spouse"]]
            if rng.random() < 0.6:
                d.say(f"In {p['married']} {nm} married {sp['name']}, {_a(sp['prof'])}.",
                      (pid, "spouse", (p["spouse"], p["married"])), (p["spouse"], "prof", sp["prof"]))
            else:
                d.say(f"{nm} married {sp['name']} in {p['married']}.", (pid, "spouse", (p["spouse"], p["married"])))
            self.pad(d, 4, 10)
        self.own(d, p, rng.randint(1, 2))
        if p["died"]:
            d.para()
            age = p["died"] - by_ - (1 if rng.random() < 0.4 else 0)
            dt = E[rng.choice(self._towns)]["name"]
            d.say(rng.choice([f"{nm} died in {dt} in {p['died']}.", f"{nm} died in {p['died']} at the age of {age}.",
                              f"{nm} spent {pos} last years in {dt}, where {pro} died in {p['died']}.",
                              f"The death of {nm} in {p['died']} was noted in the newspapers of {dt}."]),
                  (pid, "died", p["died"]))
            self.pad(d, 2, 6)
        self.docs.append(d)
        if rng.random() < 0.25:
            self.person_secondary(pid)

    def person_secondary(self, pid: int) -> None:
        rng, E = self.rng, self.E
        p = E[pid]
        nm = p["name"]
        town = E[p["town"]]
        kind = rng.choice(["Obituary", "Local history", "Profile", "Centenary notice"])
        d = _Doc(f"{kind}: {nm}", "secondary", pid)
        d.say(f"This {kind.lower()} recalls the life of {nm}, who was born in {town['name']} in {p['born'][0]}.",
              (pid, "town", p["town"]), (pid, "born", (p["born"][0], None, None)))
        self.pad(d, 6, 12)
        d.para()
        if rng.random() < 0.5:
            d.say(f"{nm} {rng.choice(L.PROFESSIONS[p['cat']][1])}, though few outside {town['name']} knew it.",
                  (pid, "prof", p["prof"]))
        else:
            d.say(f"Readers will remember {nm} as {_a(p['prof'])} of unusual talent.", (pid, "prof", p["prof"]))
        self.pad(d, 4, 10)
        if rng.random() < 0.5:
            fp = rng.choice(L.PROFESSIONS[L.TERM_CATEGORY[p["father"]]][1])
            d.say(f"The father of {nm} {fp}.", (pid, "father", p["father"]))
        else:
            mp = rng.choice(L.PROFESSIONS[L.TERM_CATEGORY[p["mother"]]][1])
            d.say(f"The mother of {nm} {mp}.", (pid, "mother", p["mother"]))
        if p["sibdeath"] and rng.random() < 0.6:
            d.say(f"Grief came early: a sibling of {nm} died in {p['sibdeath']}.", (pid, "sibdeath", p["sibdeath"]))
        self.pad(d, 4, 10)
        d.para()
        for wid in p["works"][:2]:
            w = E[wid]
            d.say(f"{nm} is best remembered for '{w['name']}', which came out in {w['year']}.", (pid, "created", wid),
                  (wid, "year", w["year"]), (wid, "creator", pid))
        self.own(d, p, rng.randint(1, 2))
        self.pad(d, 4, 10)
        if p["died"]:
            d.say(f"{nm} died in {p['died']}, mourned in {town['name']}.", (pid, "died", p["died"]))
        self.docs.append(d)

    def work(self, wid: int) -> None:
        rng, E = self.rng, self.E
        w = E[wid]
        c = E[w["creator"]]
        t = w["name"]
        d = _Doc(t, "article", wid)
        verb = {"painting": "painted", "etching": "made"}.get(w["kind"], "written" if w["kind"] in
                                                             ("novel", "poetry collection", "children's book", "play")
                                                             else "composed")
        lead = f"'{t}' is {_a(w['kind'])} {verb} by {c['name']}"
        if w["publisher"] is not None:
            d.say(f"{lead}, first published in {w['year']} by {self.nm(w['publisher'])}.", (wid, "creator", w["creator"]),
                  (wid, "year", w["year"]), (wid, "publisher", w["publisher"]))
        else:
            d.say(f"{lead} and completed in {w['year']}.", (wid, "creator", w["creator"]), (wid, "year", w["year"]))
        self.pad(d, 6, 12)
        d.para()
        th = w["theme"]
        d.say(rng.choice([f"'{t}' centres on {_a(th)} and the people whose lives turn around it.",
                          f"The central image of '{t}' is {_a(th)}, returned to again and again.",
                          f"Critics have read the {th} at the heart of '{t}' as a portrait of its maker.",
                          f"Everything in '{t}' is organised around a single motif, {_a(th)}."]), (wid, "theme", th))
        self.pad(d, 6, 12)
        d.para()
        if w["illustrator"] is not None:
            d.say(rng.choice([f"The first edition of '{t}' carried illustrations by {self.nm(w['illustrator'])}.",
                              f"{self.nm(w['illustrator'])} provided the drawings for the first edition of '{t}'."]),
                  (wid, "illustrator", w["illustrator"]))
            self.pad(d, 2, 6)
        if w["reception"] == "poor":
            d.say(rng.choice([f"Early reviewers of '{t}' were unkind, and {c['name']} was said to be disappointed.",
                              f"'{t}' sold poorly at first and was dismissed by most critics of the day.",
                              f"The first audiences were cool towards '{t}', a reception that wounded {c['name']}.",
                              f"Few people had a good word for '{t}' when it first appeared, to the dismay of "
                              f"{c['name']}."]), (wid, "reception", "poor"))
        else:
            d.say(rng.choice([f"'{t}' was an immediate success with readers and critics alike.",
                              f"Reviewers praised '{t}' warmly from the start.",
                              f"The first audiences of '{t}' received it with enthusiasm.",
                              f"From its first appearance '{t}' was admired, and {c['name']} was much in demand."]),
                  (wid, "reception", "good"))
        self.pad(d, 4, 10)
        d.para()
        if w["shown"]:
            g, y, mo = w["shown"]
            d.say(rng.choice([f"'{t}' was first shown at {self.nm(g)} in {y}, where it hung for {_num(rng, mo)} months.",
                              f"Its first public showing came in {y} at {self.nm(g)}; '{t}' stayed on the walls for "
                              f"{_num(rng, mo)} months."]), (wid, "shown", w["shown"]))
        if w["stolen"]:
            sy, ry = w["stolen"]
            d.say(rng.choice([f"'{t}' was stolen in {sy} and only recovered in {ry}.",
                              f"Thieves removed '{t}' from its frame in {sy}; it was found again in {ry}."]),
                  (wid, "stolen", w["stolen"]))
        if w["prize"]:
            z, y = w["prize"]
            d.say(f"In {y} '{t}' won {self.nm(z)}.", (wid, "prize", w["prize"]))
        self.own(d, c)
        self.pad(d, 4, 10)
        self.docs.append(d)
        if rng.random() < 0.15:
            r = _Doc(f"Review: {t}", "secondary", wid)
            r.say(f"Revisiting '{t}', the {w['kind']} {c['name']} produced in {w['year']}, is a strange pleasure.",
                  (wid, "creator", w["creator"]), (wid, "year", w["year"]))
            self.pad(r, 4, 10)
            r.say(f"The {th} that dominates '{t}' has lost none of its force.", (wid, "theme", th))
            if w["illustrator"] is not None:
                r.say(f"The drawings by {self.nm(w['illustrator'])} remain the chief charm of '{t}'.",
                      (wid, "illustrator", w["illustrator"]))
            self.own(r, c, 1)
            self.pad(r, 6, 12)
            self.docs.append(r)

    def town(self, tid: int) -> None:
        rng, E = self.rng, self.E
        t = E[tid]
        tn = t["name"]
        d = _Doc(tn, "article", tid)
        reg = E[t["region"]]
        if t["river"]:
            d.say(f"{tn} is a town in the {reg['name']} region, on the banks of {t['river']}.",
                  (tid, "region", t["region"]), (tid, "river", t["river"]))
        else:
            d.say(f"{tn} is a market town in the {reg['name']} region.", (tid, "region", t["region"]))
        self.pad(d, 6, 12)
        d.para()
        d.say(rng.choice([f"The settlement of {tn} was founded in {t['founded']}.",
                          f"Records date the founding of {tn} to {t['founded']}.",
                          f"{tn} received its first charter in {t['founded']}."]), (tid, "founded", t["founded"]))
        self.pad(d, 4, 10)
        pop, cy = t["pop"]
        d.say(f"At the {cy} census {tn} had {pop:,} inhabitants.", (tid, "pop", t["pop"]))
        self.pad(d, 4, 8)
        d.para()
        if t["landmark"]:
            ln, y = t["landmark"]
            d.say(f"{tn} is home to {ln}, built in {y}.", (tid, "landmark", t["landmark"]))
        if rng.random() < 0.5:
            d.say(f"The {reg['name']} region, to which {tn} belongs, was created in {reg['formed']}.",
                  (t["region"], "formed", reg["formed"]))
        self.pad(d, 4, 10)
        nat = t["natives"][:6]
        if nat:
            d.para()
            d.say(f"Notable people born in {tn} include " + ", ".join(self.nm(x) for x in nat) + ".",
                  *[(x, "town", tid) for x in nat])
        self.own(d, None, 1)
        self.pad(d, 4, 8)
        self.docs.append(d)

    def region(self, rid: int) -> None:
        E = self.E
        r = E[rid]
        d = _Doc(r["name"], "article", rid)
        d.say(f"The {r['name']} region was formed in {r['formed']} as an administrative district.",
              (rid, "formed", r["formed"]))
        self.pad(d, 6, 12)
        tw = [t for t in self._towns if E[t]["region"] == rid][:25]
        if tw:
            d.say(f"Towns in the {r['name']} region include " + ", ".join(E[t]["name"] for t in tw) + ".",
                  *[(t, "region", rid) for t in tw])
        self.docs.append(d)

    def org(self, oid: int) -> None:
        rng, E = self.rng, self.E
        o = E[oid]
        on = o["name"]
        d = _Doc(on, "article", oid)
        desc = {"gallery": "an exhibition venue", "press": "a publishing house", "school": "a school",
                "society": "a members' society", "prize": "an annual award"}[o["kind"]]
        if o["kind"] == "prize":
            d.say(f"{on} is {desc}, established in {o['founded']} and presented in {E[o['town']]['name']}.",
                  (oid, "founded", o["founded"]), (oid, "located", o["town"]))
        else:
            d.say(f"{on} is {desc} in {E[o['town']]['name']}, founded in {o['founded']}.",
                  (oid, "founded", o["founded"]), (oid, "located", o["town"]))
        self.pad(d, 6, 12)
        if o["founder"] is not None:
            f = E[o["founder"]]
            d.say(f"{on} was established by {f['name']}, {_a(f['prof'])}.", (oid, "founder", o["founder"]),
                  (o["founder"], "prof", f["prof"]))
        d.para()
        if o["closed"]:
            d.say(rng.choice([f"{on} closed its doors in {o['closed']}.", f"{on} was dissolved in {o['closed']}.",
                              f"After years of decline {on} shut for good in {o['closed']}."]), (oid, "closed", o["closed"]))
        self.pad(d, 4, 10)
        mem = o["members"][:12]
        if mem:
            verb = {"gallery": "Artists who exhibited at", "society": "Members of"}.get(o["kind"], "Names associated with")
            d.say(f"{verb} {on} included " + ", ".join(self.nm(x) for x in mem) + ".")
        self.own(d, None, 1)
        self.pad(d, 4, 8)
        self.docs.append(d)

    def lists_and_noise(self, n_noise: int, by: dict[str, list[int]]) -> None:
        rng, E = self.rng, self.E
        people, works = by["person"], by["work"]
        made = 0
        years: dict[int, list[int]] = {}
        for w in works:
            years.setdefault(E[w]["year"], []).append(w)
        for y, ws in sorted(years.items()):
            if made >= n_noise // 4:
                break
            d = _Doc(f"Works of {y}", "list", -1)
            for w in ws[:30]:
                ww = E[w]
                d.say(f"'{ww['name']}', {_a(ww['kind'])} by {E[ww['creator']]['name']} ({y}).", (w, "year", y),
                      (w, "creator", ww["creator"]))
            self.docs.append(d)
            made += 1
        births: dict[int, list[int]] = {}
        for p in people:
            births.setdefault(E[p]["born"][0], []).append(p)
        for y, ps in sorted(births.items()):
            if made >= n_noise // 2:
                break
            d = _Doc(f"Births in {y}", "list", -1)
            for p in ps[:30]:
                pp = E[p]
                d.say(f"{pp['name']}, {pp['prof']}, born in {E[pp['town']]['name']} ({y}).", (p, "born", (y, None, None)),
                      (p, "prof", pp["prof"]), (p, "town", pp["town"]))
            self.docs.append(d)
            made += 1
        topics = ["Harbour news", "Market report", "Weather notes", "Society column", "Letters to the editor",
                  "Travel notes", "Parish notes", "Auction notices", "Concert listings", "Gardening notes"]
        while made < n_noise:
            d = _Doc(f"{rng.choice(topics)}, {rng.randint(1850, 2020)}", "noise", -1)
            for _ in range(rng.randint(3, 7)):
                p = E[rng.choice(people)]
                t = E[rng.choice(self._towns)]
                d.say(rng.choice([
                    f"{p['name']} was seen at the {t['name']} fair buying {_a(rng.choice(_THINGS))}.",
                    f"A {rng.choice(self.terms)} from {t['name']} has opened new premises near the station.",
                    f"The {rng.choice(self.terms)}s of {t['name']} met on {rng.choice(L.MONTHS)} {rng.randint(1, 28)} "
                    f"to discuss rents.",
                    f"Visitors to {t['name']} are advised that the old bridge is closed until {rng.choice(_SEASONS)}.",
                    f"An auction of furniture from a house in {t['name']} raised {rng.randint(20, 900)} pounds.",
                    f"A lecture on the {rng.choice(_THINGS)} was given in {t['name']} by {p['name']}.",
                ]))
                self.pad(d, 4, 10)
            self.docs.append(d)
            made += 1


def _render_doc(doc_id: str, title: str, date: str, paras: list[list[str]]) -> str:
    body = "\n\n".join(" ".join(p) for p in paras if p)
    return f"---\ndoc_id: {doc_id}\n---\n---\ntitle: {title}\ndate: {date}\n---\n{title}\n\n{body}"


def build_world(wid: str, scale: str = "full") -> tuple[World, list[str]]:
    """The world plus its rendered documents in original case (written to the store, then dropped from memory)."""
    from synth_common.segindex import segments as _segs

    t0 = time.time()
    rng = random.Random(f"bcp-synth:{GEN_VERSION}:{wid}:{scale}")
    ents, by = _kg(rng, scale)
    R = _Renderer(ents, rng, by["town"], scale)
    for pid in by["person"]:
        R.person(pid)
    for wid_ in by["work"]:
        R.work(wid_)
    for tid in by["town"]:
        R.town(tid)
    for rid in by["region"]:
        R.region(rid)
    for oid in by["org"]:
        R.org(oid)
    R.lists_and_noise(SCALES[scale][-1], by)
    docs = R.docs
    rng.shuffle(docs)

    W = World(wid=wid, scale=scale, ents=ents, by_type=by)
    W.name_index = {e["name"].lower(): e["id"] for e in ents}
    ids = rng.sample(range(10_000, 999_999), len(docs))
    items: list[tuple[str, str]] = []
    rendered: list[str] = []
    hashes: set[int] = set()
    for b in R.bank:
        for sg in _segs(b.lower(), 8):
            hashes.add(hash(sg))
    for di, d in enumerate(docs):
        did = str(ids[di])
        date = f"{rng.randint(2003, 2024)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        text = _render_doc(did, d.title, date, d.paras)
        rendered.append(text)
        W.low.append(text.lower())
        W.doc_ids.append(did)
        W.doc_kind.append(d.kind)
        W.doc_subject.append(d.subject)
        W.titles.append(d.title)
        for k, (sent, fs) in enumerate(d.facts):
            fids = []
            for (e, a, v) in fs:
                fid = len(W.facts)
                W.facts.append((e, a, v))
                W.fact_doc.append(di)
                W.fact_sent.append(sent)
                W.ent_facts.setdefault(e, []).append(fid)
                dl = W.ent_docs.setdefault(e, [])
                if not dl or dl[-1] != di:
                    dl.append(di)
                fids.append(fid)
            iid = f"{di}:{k}"
            W.sent_facts[iid] = tuple(fids)
            items.append((iid, sent.lower()))
        for k2, sent in enumerate(d.own):
            items.append((f"{di}:x{k2}", sent.lower()))
    for _iid, sl in items:
        for sg in _segs(sl, 8):
            hashes.add(hash(sg))
    W.index = SegmentIndex(items)
    W.sent_hashes = hashes
    W.build_s = time.time() - t0
    return W, rendered


def store_dir(W: World) -> Path:
    h = hashlib.sha1(("\x1f".join(W.low[:50]) + str(len(W.low))).encode()).hexdigest()[:10]
    root = Path(os.environ.get("BCP_SYNTH_STORE_ROOT") or os.environ.get("RLM_DOC_STORE_LOCAL_ROOT") or "/tmp/rlm_doc_stores")
    return root / f"bcp-synth-{GEN_VERSION}-{W.scale}-{W.wid}-{h}"


def ensure_store(W: World, rendered: list[str] | None) -> str:
    """Write the world's docs as an rlm_train doc store once per machine (lock-guarded); return its path."""
    from rlm_train import doc_store

    d = store_dir(W)
    d.parent.mkdir(parents=True, exist_ok=True)
    if not (d / "index.json").exists():
        with open(d.parent / (d.name + ".lock"), "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not (d / "index.json").exists():
                doc_store.build_store(d, zip(W.doc_ids, rendered, strict=False))
    W.store_path = str(d)
    W.store_chars = [len(x) for x in W.low]
    return W.store_path


_WORLDS: dict[tuple[str, str], World] = {}


def get_world(wid: str, scale: str = "full") -> World:
    key = (wid, scale)
    W = _WORLDS.get(key)
    if W is None:
        W, rendered = build_world(wid, scale)
        ensure_store(W, rendered)
        del rendered
        _WORLDS[key] = W
    return W


__all__ = ["GEN_VERSION", "SCALES", "World", "build_world", "ensure_store", "get_world", "store_dir"]
_ = Any
