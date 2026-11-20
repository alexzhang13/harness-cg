"""One episode: scenes -> (speaker, text) lines + the roll / cast tables a careful annotator would write."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from . import vocab as V
from ._borrow import W

DM = "@DM"
SPOKEN = dict(W.SPOKEN, **{"Hit Dice": ["hit dice"]})


@dataclass
class Actor:
    player: str
    char: str
    cls: str
    spells: list
    features: list
    weapon: str
    mods: dict
    atk: int
    pet: str | None = None
    pet_kind: str | None = None


def base_of(spell: str):
    if spell not in V.LEVEL:
        return "Feature"
    return "Cantrip" if V.LEVEL[spell] == 0 else V.LEVEL[spell]


class Episode:
    def __init__(self, rng: random.Random, actors: list[Actor], k: int, numwords: float = 0.25):
        self.r, self.actors, self.k, self.numwords = rng, actors, k, numwords
        self.lines: list[tuple[str, str]] = []
        self.rolls: list[dict] = []
        self.spells: list[dict] = []
        self.last_cast: dict[str, tuple] = {}
        self.adv = 0
        self.monster = "the bandit"

    def say(self, spk: str, text: str) -> int:
        text = re.sub(r"\b([Aa]) (8\d*|11|18|eight\w*|eleven)\b", r"\1n \2", " ".join(text.split()))
        self.lines.append((spk, text))
        return len(self.lines) - 1

    def N(self, v: int) -> str:
        return W.NUM_WORDS[v] if 0 <= v <= 40 and self.r.random() < self.numwords else str(v)

    def roll(self, li, a: Actor, rtype, total, nat, cue, how, char=None, cue_li=None, **kw) -> None:
        self.rolls.append(dict(li=li, char=char or a.char, player=a.player, type=rtype, total=total, nat=nat,
                               cue=str(cue), cue_li=li if cue_li is None else cue_li, how=how, **kw))

    def cast(self, li, a: Actor, spell, cue, how, level=None, **kw) -> None:
        b = base_of(spell)
        c = b if level is None or not isinstance(b, int) else level
        self.spells.append(dict(li=li, char=a.char, player=a.player, spell=spell, base=b, cast=c, cue=str(cue),
                                how=how, **kw))
        self.last_cast[a.char] = (spell, level, li)

    def fmt(self, t: str, a: Actor | None = None) -> str:
        r = self.r
        others = [x.char for x in self.actors if a is None or x is not a] or ["everyone"]
        npc = r.choice(W.NPCS)
        return t.format(place=r.choice(W.PLACES), adj=r.choice(W.ADJ), sound=r.choice(W.SOUNDS), npc=npc,
                        npc_cap=npc[0].upper() + npc[1:], npc2=r.choice(W.NPCS), obj=r.choice(W.OBJECTS),
                        pc=r.choice(others), m=self.monster, w=a.weapon if a else "blade")

    def spoken(self, spell: str) -> str:
        r = self.r
        u = r.random()
        if spell in V.SHORT and u < 0.35:
            return V.SHORT[spell]
        if spell in W.SPELL_ALIASES and u < 0.6:
            return r.choice(W.SPELL_ALIASES[spell])
        return spell.lower() if u < 0.75 else spell

    def pick(self, k=None) -> Actor:
        return self.r.choice(self.actors)

    def d20_say(self, nat: int, t: int, mod: int):
        """-> (text, cue, recorded natural, how)."""
        r, N = self.r, self.N
        if nat == 20:
            return r.choice([("Natural 20!", "20", "nat"), (f"Nat 20! So {t}.", "20", "nat"),
                             ("Natural twenty, oh my god.", "twenty", "nat")]) + (20,)
        if nat == 1:
            return r.choice([("Natural 1.", "1", "nat"), ("Nat 1. Of course.", "1", "nat"),
                             ("Oh no. Natural one.", "one", "nat")]) + (1,)
        u = r.random()
        if u < 0.10:
            return r.choice([(f"{t}, the die says {nat}.", str(t), "stated_nat"),
                             (f"Natural {nat}, so {t} total.", str(t), "stated_nat")]) + (nat,)
        if u < 0.22 and mod > 0:
            T = N(t)
            return f"{N(nat)} plus {N(mod)} is {T}.", T, "arith", nat
        T = N(t)
        tmpl = r.choice(["{T}.", "That's a {T}.", "{T} for me.", "Uh, {T}?", "I got a {T}.", "{T}. Is that good?",
                         "That is {T} for me.", "Okay, {T}."])
        return tmpl.format(T=T), T, "word" if not T.isdigit() else "bare", None

    def mod(self, a: Actor, rtype: str) -> int:
        return a.atk if rtype in ("Attack", "Spell Attack") else a.mods.get(rtype, 2)

    def d20(self, a: Actor, rtype: str, *, special=True, char=None, prefix="") -> int:
        """The player's answer to a d20 request (the type is on the request line). Returns the anchor line."""
        r = self.r
        m = self.mod(a, rtype)
        u = r.random() if special else 1.0
        if u < 0.12:
            return self.adv_roll(a, rtype, char=char)
        check_like = rtype not in ("Attack", "Spell Attack", "Initiative")
        if u < 0.15 and check_like:
            text = r.choice(["Success.", "That's a success.", "Made it.", "That fails.", "Failed it.", "Nope. Fail."])
            li = self.say(a.player, text)
            self.roll(li, a, rtype, None, None, text.rstrip("."), "outcome", char=char)
            return li
        nat = r.randint(1, 20)
        if u < 0.18 and nat not in (1, 20):
            t0 = nat + m - r.randint(1, 3)
            li = self.say(a.player, f"{prefix}{t0}.")
            fix = nat + m
            self.say(DM, r.choice(["Okay.", "Mm-hmm.", f"{t0}?"]))
            self.say(a.player, r.choice([f"Wait, no, {fix}. I forgot my bonus.", f"Sorry, {fix}, I added wrong."]))
            self.roll(li, a, rtype, fix, None, str(t0), "correction", char=char)
            return li
        if u < 0.20 and check_like:
            t0 = r.randint(1, 6) + m
            li0 = self.say(a.player, f"{prefix}{self.N(t0)}. Ugh.")
            self.roll(li0, a, rtype, t0, None, self.lines[li0][1].split(".")[0].replace(prefix, "", 1), "reroll",
                      char=char)
            self.say(DM, r.choice(["Do you want to use your luck point?", "You have that reroll feature, right?"]))
            t1 = r.randint(1, 20) + m
            li1 = self.say(a.player, f"Yes! Rerolling. {t1}.")
            self.roll(li1, a, rtype, t1, None, str(t1), "reroll", char=char)
            return li1
        text, cue, how, natrec = self.d20_say(nat, nat + m, m)
        li = self.say(a.player, prefix + text)
        self.roll(li, a, rtype, nat + m, natrec, cue, how, char=char)
        return li

    def adv_roll(self, a: Actor, rtype: str, char=None) -> int:
        """Advantage / disadvantage: two rows; often only the kept value is spoken."""
        r, m = self.r, self.mod(a, rtype)
        word = "advantage" if r.random() < 0.7 else "disadvantage"
        n1, n2 = r.randint(1, 20), r.randint(1, 20)
        keep, drop = (max(n1, n2), min(n1, n2)) if word == "advantage" else (min(n1, n2), max(n1, n2))

        def tok(n):
            return ("natural 20", 20) if n == 20 else ("a natural 1", 1) if n == 1 else (self.N(n + m), None)
        kt, knat = tok(keep)
        self.adv += 1
        if r.random() < 0.5:
            dt, dnat = tok(drop)
            text = r.choice([f"With {word}: {dt} and {kt}. So {kt}.", f"{kt} and {dt} with {word}, so {kt}."])
            li = self.say(a.player, text)
            self.roll(li, a, rtype, keep + m, knat, kt, "adv_kept", char=char, pair=self.adv)
            self.roll(li, a, rtype, drop + m, dnat, dt, "adv_drop", char=char, pair=self.adv)
        else:
            text = r.choice([f"With {word}, {kt}.", f"I have {word}... {kt}.", f"Rolling with {word}. {kt}!"])
            li = self.say(a.player, text)
            self.roll(li, a, rtype, keep + m, knat, kt, "adv_kept", char=char, pair=self.adv)
            self.roll(li, a, rtype, None, None, word, "adv_drop", char=char, pair=self.adv)
        return li

    def dmg(self, a: Actor, total: int, dtype: str = "slashing", char=None, rtype="Damage") -> int:
        r, N = self.r, self.N
        T = N(total)
        if rtype == "Healing":
            text = r.choice([f"{T} hit points.", f"They get {T} back.", f"That's {T} healing."])
        else:
            a1 = r.randint(1, max(1, total - 1))
            text = r.choice([f"{T} damage.", f"{a1} plus {total - a1}, {T} damage.", f"That's {T} {dtype}.",
                             f"{T} points of {dtype} damage.", f"Um, {T}?"])
        li = self.say(a.player, text)
        self.roll(li, a, rtype, total, None, T, "dmg" if rtype == "Damage" else "heal", char=char)
        return li

    def request(self, a: Actor, rtype: str, char=None) -> int:
        r, who = self.r, char or a.char
        forms = SPOKEN.get(rtype, [rtype.lower()])
        s = r.choice([f for f in forms if f.startswith(("a ", "an "))] or forms)
        lead = self.fmt(r.choice(W.DM_NARR)) + " " if r.random() < 0.3 else ""
        return self.say(DM, lead + r.choice([f"{who}, make {s}.", f"Can I get {s} from {who}?", f"Give me {s}, {who}.",
                                             f"{who}, go ahead and roll {s}.", f"All right {who}, roll me {s}."]))

    def filler(self, n: int, table_talk: bool = False) -> None:
        r = self.r
        for _ in range(n):
            if table_talk:
                a = self.pick()
                self.say(a.player, r.choice(W.CHATTER + W.OOC + V.OOC2) + " " + r.choice(W.CHATTER))
                continue
            u = r.random()
            if u < 0.36:
                t = self.fmt(r.choice(W.DM_NARR + V.DM_NARR2 * 2))
                if r.random() < 0.45:
                    t += " " + self.fmt(r.choice(W.DM_NARR))
                self.say(DM, t)
            elif u < 0.66:
                a = self.pick()
                t = r.choice(W.PC_TALK + V.PC_TALK2)
                if r.random() < 0.5:
                    t = r.choice(W.CHATTER) + " " + t
                if r.random() < 0.3:
                    t += " " + r.choice(W.PC_TALK)
                self.say(a.player, self.fmt(t, a))
            elif u < 0.80:
                a = self.pick()
                self.say(a.player, r.choice(W.CHATTER) + " " + self.fmt(r.choice(W.PC_TALK), a))
            elif u < 0.90:
                self.say(DM, r.choice(W.DM_SHORT) + " " + self.fmt(r.choice(W.DM_NARR)))
            else:
                self.say(self.pick().player, r.choice(W.OOC + V.OOC2 * 2))

    def distractor(self) -> None:
        r, a = self.r, self.pick()
        c = r.randint(0, 9)
        if c == 0:
            self.say(a.player, f"My AC is {r.randint(12, 19)}, does that hit me?")
        elif c == 1:
            self.say(a.player, f"I'm at {r.randint(3, 60)} hit points, for the record.")
        elif c == 2:
            self.say(DM, f"{r.choice(W.NPCS).capitalize()} counts out {r.randint(20, 300)} gold pieces for you.")
        elif c == 3:
            self.say(a.player, f"If I roll a {r.choice([15, 18, 20])} on this, do I get to see through the door?")
            self.say(DM, "Let's find out when you actually roll.")
        elif c == 4:
            s = r.choice([x for x in a.spells if base_of(x) != "Feature"] or ["Fireball"])
            self.say(a.player, f"Should I cast {self.spoken(s)}? No, never mind, I'll save it.")
        elif c == 5:
            ss = r.sample(a.spells, min(3, len(a.spells))) if len(a.spells) >= 2 else ["Fireball", "Haste"]
            self.say(a.player, f"I've got {', '.join(ss[:-1])} and {ss[-1]} prepared today, just so you know.")
        elif c == 6:
            b = self.pick()
            s = r.choice(b.spells) if b.spells else "Bless"
            self.say(a.player, f"{b.char}, you should cast {self.spoken(s)} on us before we go in!")
            self.say(b.player, "Maybe. Let's see what's in there first.")
        elif c == 7:
            self.say(a.player, f"These dice rolled {r.choice(['three', 'four', 'two'])} ones before the game, it's bad.")
        elif c == 8:
            self.say(DM, f"It's about {r.choice([30, 60, 90, 120])} feet to the far wall, for those who care.")
        else:
            self.say(a.player, "What was my initiative again?")
            self.say(DM, f"You had a {self.N(r.randint(4, 22))}.")

    def explore(self) -> None:
        r = self.r
        self.filler(r.randint(2, 6))
        for _ in range(r.randint(1, 3)):
            a = self.pick()
            u = r.random()
            rtype = (r.choices(W.SKILLS, W.SKILL_W)[0] if u < 0.75 else r.choice(W.ABILITY_CHECKS) if u < 0.85
                     else r.choice(W.SAVES[:3]) if u < 0.95 else "Thieves' Tools")
            if r.random() < 0.15 and a.pet and rtype == "Perception":
                self.say(DM, f"Is {a.pet} looking too? Roll for the {a.pet_kind}.")
                n = r.randint(3, 20)
                li = self.say(a.player, f"{a.pet} rolls perception... {self.N(n)}.")
                self.roll(li, a, "Perception", n, None, a.pet, "pet", char=a.pet)
            elif r.random() < 0.2:
                self.say(a.player, f"Can I roll {r.choice(SPOKEN.get(rtype, [rtype.lower()]))}?")
                self.say(DM, r.choice(["Go for it.", "Sure.", "Yeah, roll it."]))
                self.d20(a, rtype)
            elif r.random() < 0.12 and len(self.actors) > 2:
                s = r.choice(SPOKEN.get(rtype, [rtype.lower()]))
                self.say(DM, r.choice([f"Everybody, give me {s}.", f"Let's see {s} from all of you."]))
                for b in r.sample(self.actors, r.randint(2, len(self.actors))):
                    self.d20(b, rtype, special=r.random() < 0.5)
            else:
                guide = [b for b in self.actors if "Guidance" in b.spells and b is not a]
                if guide and r.random() < 0.3:
                    g = guide[0]
                    li = self.say(g.player, V.PASSING["Guidance"][0].format(s="Guidance", pc=a.char))
                    self.cast(li, g, "Guidance", "Guidance", "passing")
                self.request(a, rtype)
                self.d20(a, rtype)
            self.say(DM, self.fmt(r.choice(W.DM_NARR)))
            if r.random() < 0.3:
                self.distractor()
        if r.random() < 0.45:
            self.passing_cast()

    def passing_cast(self) -> None:
        r = self.r
        cands = [(a, s) for a in self.actors for s in a.spells if s in V.PASSING or s in V.RITUALS]
        if not cands:
            return
        a, s = r.choice(cands)
        sp = self.spoken(s)
        if s in V.RITUALS:
            text = r.choice([f"I'd like to ritual cast {sp}.", f"I'll cast {sp} as a ritual, it takes ten minutes.",
                             f"Ritual casting {sp} while everyone else takes a breather."])
        else:
            others = [b.char for b in self.actors if b is not a] or [a.char]
            text = r.choice(V.PASSING[s]).format(s=sp, pc=r.choice(others))
        li = self.say(a.player, text)
        self.cast(li, a, s, sp, "ritual" if s in V.RITUALS else "passing")
        if s == "Find Familiar" and a.pet:
            self.say(DM, f"{a.pet} the {a.pet_kind} reappears in a puff of smoke.")
        self.say(DM, r.choice(["Okay, done.", "Sure, that works.", "Go ahead."]) + " " + self.fmt(r.choice(W.DM_NARR)))

    def social(self) -> None:
        r = self.r
        self.filler(r.randint(2, 5))
        a = self.pick()
        rtype = r.choice(["Persuasion", "Deception", "Insight", "Intimidation", "Persuasion"])
        self.request(a, rtype)
        self.d20(a, rtype)
        if r.random() < 0.4:
            self.say(DM, f"{r.choice(W.NPCS).capitalize()} rolls a {r.randint(5, 22)} on their deception. Hmm.")
        self.filler(r.randint(1, 4))

    def rest(self) -> None:
        r = self.r
        self.say(DM, "All right, you take a short rest. Anyone spending hit dice?")
        for a in r.sample(self.actors, r.randint(1, min(3, len(self.actors)))):
            if r.random() < 0.5:
                x, y = r.randint(2, 11), r.randint(2, 11)
                li = self.say(a.player, f"Two hit dice: {self.N(x)} and {self.N(y)}.")
                self.roll(li, a, "Hit Dice", x, None, str(x) if str(x) in self.lines[li][1] else W.NUM_WORDS[x],
                          "multi")
                self.roll(li, a, "Hit Dice", y, None, str(y) if str(y) in self.lines[li][1] else W.NUM_WORDS[y],
                          "multi")
            else:
                x = r.randint(2, 11)
                li = self.say(a.player, f"One hit die, I got {self.N(x)} back.")
                self.roll(li, a, "Hit Dice", x, None, self.lines[li][1].split("got ")[1].split(" ")[0], "single")
        if r.random() < 0.5:
            self.passing_cast()
        self.filler(r.randint(1, 4))

    def combat(self) -> None:
        r = self.r
        self.monster = r.choice(W.MONSTERS)
        self.say(DM, self.fmt(f"Out of {r.choice(['the dark', 'the trees', 'the doorway'])}, {self.monster} lunges "
                              "toward you. Everybody, roll initiative!"))
        order = r.sample(self.actors, len(self.actors))
        for a in order:
            if r.random() < 0.85:
                self.d20(a, "Initiative", special=r.random() < 0.3,
                         prefix=r.choice(["", "", "Initiative, "]))
        pets = [a for a in self.actors if a.pet and a.cls == "ranger"]
        for a in pets:
            n = r.randint(2, 22)
            li = self.say(a.player, f"And {a.pet} got a {self.N(n)}.")
            self.roll(li, a, "Initiative", n, None, a.pet, "pet", char=a.pet)
        self.say(DM, f"{self.monster.capitalize()} gets a {r.randint(4, 22)}. Okay, top of the round: "
                     f"{order[0].char}, you're first.")
        down: list[Actor] = []
        for _rnd in range(r.randint(3, 4)):
            for a in order:
                if a in down:
                    self.death_save(a, down)
                    continue
                if r.random() < 0.2:
                    self.filler(1, True)
                    continue
                self.say(DM, r.choice([f"{a.char}, you're up.", f"{a.char}, what are you doing?",
                                       f"That brings us to {a.char}."]))
                if a.spells and r.random() < 0.65:
                    self.spell_turn(a)
                else:
                    self.weapon_turn(a)
                if r.random() < 0.35:
                    self.monster_turn(down)
                if r.random() < 0.25:
                    self.filler(r.randint(1, 3), True)
        self.say(DM, f"With that last blow, {self.monster} crumples. The fight is over.")
        self.filler(r.randint(1, 3))

    def weapon_turn(self, a: Actor) -> None:
        r = self.r
        if a.cls in V.MARTIAL and r.random() < 0.3:
            self.say(a.player, self.fmt(r.choice(V.MARTIAL[a.cls]), a))
        if r.random() < 0.2 and a.cls in ("fighter", "monk", "ranger", "barbarian", "paladin"):
            x, y = r.randint(1, 19) + a.atk, r.randint(1, 19) + a.atk
            li = self.say(a.player, f"Two attacks with my {a.weapon}: {x} and {y}.")
            self.roll(li, a, "Attack", x, None, str(x), "multi")
            self.roll(li, a, "Attack", y, None, str(y), "multi")
            hits = 1
        else:
            self.say(a.player, f"I attack {self.monster} with my {a.weapon}.")
            self.d20(a, "Attack")
            hits = 1 if r.random() < 0.65 else 0
            tot = self.rolls[-1]["total"]
            if r.random() < 0.3 and tot is not None:
                self.say(DM, f"{tot}? " + ("That hits." if hits else "That misses."))
            elif not hits:
                self.say(DM, "That misses.")
        if hits:
            self.say(DM, r.choice(["Roll damage.", "That hits. Roll damage.", "Hits. Go ahead and roll damage."]))
            self.dmg(a, r.randint(3, 18), r.choice(["slashing", "piercing", "bludgeoning"]))
        if a.pet and a.cls == "ranger" and r.random() < 0.4:
            n = r.randint(3, 20) + 4
            li = self.say(a.player, f"{a.pet} bites {self.monster} too, that's {self.N(n)} to hit.")
            self.roll(li, a, "Attack", n, None, a.pet, "pet", char=a.pet)

    def cast_line(self, a: Actor, spell: str) -> tuple[int, int | None]:
        """Say a cast (several surface forms). Returns (anchor line, stated cast level or None)."""
        r = self.r
        if base_of(spell) == "Feature":
            li = self.say(a.player, self.fmt(r.choice(V.FEATURES[a.cls][spell]), a))
            self.cast(li, a, spell, spell.split(": ")[-1].split()[0], "feature")
            return li, None
        prev = self.last_cast.get(a.char)
        if prev and prev[0] == spell and len(self.lines) - prev[2] < 60 and r.random() < 0.6:
            li = self.say(a.player, r.choice(["I'm going to do that again.", "Same spell again, on the other one.",
                                              "I do it again.", "Again! Same as last turn."]))
            self.cast(li, a, spell, "again" if "again" in self.lines[li][1].lower() else "Same", "recast",
                      level=prev[1], ref_li=prev[2])
            return li, prev[1]
        sp, lvl, b = self.spoken(spell), None, base_of(spell)
        if spell in V.ATTACK_SPELLS or spell in V.SAVE_SPELLS:
            tgt = self.monster
        else:
            tgt = r.choice([x.char for x in self.actors if x is not a] or ["the party"])
        u = r.random()
        if isinstance(b, int) and b < 5 and u < 0.15:
            lvl = b + r.randint(1, 2)
            text = r.choice([f"I cast {sp} at {W.ORDINAL[lvl]} level.", f"{sp}, using a {W.ORDINAL[lvl]}-level slot.",
                             f"Upcasting {sp} to level {lvl}."])
            how = "upcast"
        elif spell in W.VERB_SPELLS and 0.15 <= u < 0.32:
            sp = W.VERB_SPELLS[spell]
            text = f"I'm going to {sp} {r.choice(['him', 'her', 'it', 'that thing'])}."
            how = "verb"
        elif 0.32 <= u < 0.37:
            self.say(a.player, "Okay, I'm going to cast... hold on, where is it on my sheet...")
            self.say(DM, "Take your time.")
            text, how = f"{sp[0].upper() + sp[1:]}! On {tgt}.", "multiline"
        elif 0.37 <= u < 0.43:
            li = self.say(DM, f"{a.char} raises a hand and casts {sp} at {tgt}.")
            self.cast(li, a, spell, sp, "dm_narrated")
            self.say(a.player, "Yep. That.")
            return li, None
        elif 0.43 <= u < 0.48:
            li = self.say(a.player, f"I cast {sp}.")
            self.say(DM, "Where are you aiming it?")
            self.say(a.player, f"Right at {tgt}, {sp} right there.")
            self.cast(li, a, spell, sp, "named_twice")
            return li, None
        else:
            text = r.choice([f"I cast {sp} on {tgt}.", f"I'm going to use {sp}.", f"{sp}, at {tgt}.",
                             f"Let's do {sp} this turn."])
            how = "direct"
        li = self.say(a.player, text)
        self.cast(li, a, spell, sp, how, level=lvl)
        return li, lvl

    def spell_turn(self, a: Actor) -> None:
        r = self.r
        pool = [s for s in a.spells + a.features if s not in V.RITUALS and s not in V.REACTIONS
                and s not in V.PASSING and s not in ("Disguise Self", "Find Steed")]
        if not pool:
            return self.weapon_turn(a)
        s = r.choice(pool)
        prev = self.last_cast.get(a.char)
        if prev and prev[0] in pool and r.random() < 0.3:
            s = prev[0]
        li, lvl = self.cast_line(a, s)
        if s == "Scorching Ray":
            xs = [r.randint(1, 20) + a.atk for _ in range(3)]
            li = self.say(a.player, f"Three rays: {xs[0]}, {xs[1]}, and {xs[2]}.")
            for x in xs:
                self.roll(li, a, "Spell Attack", x, None, str(x), "multi")
            self.say(DM, "Two of those hit. Roll damage for both.")
            self.dmg(a, r.randint(8, 22), "fire")
        elif s in V.ATTACK_SPELLS:
            self.say(DM, r.choice(["Roll to hit.", "Go ahead and make a spell attack.", "Make your spell attack."]))
            self.d20(a, "Spell Attack")
            if r.random() < 0.65:
                self.say(DM, "That hits. Roll damage.")
                self.dmg(a, r.randint(2, 20), W.DAMAGE_TYPE.get(s, "force"))
        elif s in V.SAVE_SPELLS:
            sv = r.choice(["wisdom", "dexterity", "constitution"])
            self.say(DM, f"{self.monster.capitalize()} rolls a {r.randint(3, 24)} on the {sv} save. "
                         + r.choice(["It fails.", "That saves."]))
            if s in W.DAMAGE_TYPE and r.random() < 0.7:
                self.say(DM, "Roll damage.")
                self.dmg(a, r.randint(2, 12) if base_of(s) == "Cantrip" else r.randint(6, 40), W.DAMAGE_TYPE[s])
        elif s in V.HEAL_SPELLS:
            self.say(DM, "Roll your healing.")
            self.dmg(a, r.randint(3, 16), rtype="Healing")
        else:
            self.say(DM, r.choice(["Okay, that's in effect.", "Got it.", "Noted, that's up."]))

    def monster_turn(self, down: list[Actor]) -> None:
        r = self.r
        t = self.pick()
        u = r.random()
        if u < 0.2:
            npcs = ["Hold Person", "Fireball", "Charm Person", "Darkness", "Lightning Bolt"]
            ns = r.choice(npcs)
            self.say(DM, f"{self.monster.capitalize()} starts casting {ns}, aimed at {t.char}.")
            cs = [b for b in self.actors if "Counterspell" in b.spells and b not in down]
            if cs:
                b = cs[0]
                li = self.say(b.player, "Counterspell! As a reaction.")
                self.cast(li, b, "Counterspell", "Counterspell", "reaction")
            else:
                self.say(DM, "Make a wisdom saving throw.")
                self.d20(t, "Wisdom Save")
            return
        hit = r.randint(12, 26)
        self.say(DM, f"{self.monster.capitalize()} swings at {t.char}. That's a {hit} to hit.")
        rx = [s for s in t.spells if s in ("Shield", "Hellish Rebuke")]
        if rx and r.random() < 0.6:
            s = rx[0]
            li = self.say(t.player, r.choice([f"{s}! As a reaction.", f"Reaction, I cast {s.lower()}."]))
            self.cast(li, t, s, s if s in self.lines[li][1] else s.lower(), "reaction")
            self.say(DM, "Okay, noted.")
            return
        self.say(t.player, f"My AC is {r.randint(12, 19)}, so that hits.")
        dmg = r.randint(4, 20)
        self.say(DM, f"You take {dmg} damage.")
        if r.random() < 0.12 and t not in down:
            self.say(DM, f"That drops you to zero, {t.char}. You're unconscious.")
            down.append(t)

    def death_save(self, a: Actor, down: list) -> None:
        r = self.r
        self.say(DM, r.choice([f"{a.char}, make a death saving throw.", f"Death save, {a.char}."]))
        nat = r.randint(1, 20)
        if nat == 20:
            li = self.say(a.player, "Natural 20! I'm back up!")
            self.roll(li, a, "Death Save", 20, 20, "20", "nat")
            down.remove(a)
            return
        said = "Natural 1" if nat == 1 else self.N(nat)
        li = self.say(a.player, f"{said}. " + ("Success." if nat >= 10 else "That's a failure."))
        self.roll(li, a, "Death Save", nat, nat, said, "death")
        heal = [b for b in self.actors if b not in down and any(s in V.HEAL_SPELLS for s in b.spells)]
        if heal and r.random() < 0.6:
            b = heal[0]
            s = next(s for s in b.spells if s in V.HEAL_SPELLS)
            li = self.say(b.player, f"I cast {s.lower()} on {a.char}!")
            self.cast(li, b, s, s.lower(), "direct")
            self.dmg(b, r.randint(3, 12), rtype="Healing")
            down.remove(a)

    def recap(self, prev_rolls: list[dict], prev_spells: list[dict]) -> None:
        r = self.r
        parts = []
        if prev_rolls:
            x = r.choice(prev_rolls)
            parts.append(f"{x['char']} rolled a {x['total'] or 'decent number'} on that {x['type'].lower()} roll")
        if prev_spells:
            x = r.choice(prev_spells)
            parts.append(f"{x['char']} cast {x['spell']}")
        self.say(DM, "Welcome back, everyone. Last time, " + (" and ".join(parts) or "you rested") + ", and you all "
                     "ended the night at " + r.choice(W.PLACES) + ".")

    def build(self, target: int, prev_rolls=(), prev_spells=()) -> None:
        r = self.r
        self.say(DM, r.choice(["Hello everyone, and welcome to tonight's game!", "Okay, let's get started, everybody.",
                               "Are we recording? Great. Welcome back to the table."]))
        if self.k > 0:
            self.recap(list(prev_rolls), list(prev_spells))
        self.filler(r.randint(5, 15))
        broke = False
        while len(self.lines) < target:
            u = r.random()
            if u < 0.45:
                self.filler(r.randint(6, 16))
            elif u < 0.72:
                self.explore()
            elif u < 0.84:
                self.social()
            elif u < 0.905:
                self.combat()
            elif u < 0.935:
                self.rest()
            else:
                self.distractor()
            if not broke and len(self.lines) > target / 2:
                self.say(DM, "Let's take a quick break, we'll be back in a few minutes.")
                self.filler(2)
                broke = True
        self.say(DM, "And that's where we'll end it for tonight. Thank you all for playing!")
