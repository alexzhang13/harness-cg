"""Procedural D&D session transcripts with ground-truth roll / spell events (the synthetic OOLONG-real world)."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

from synth_common.segindex import _SPLIT

from . import world as W

INDEX_MIN_LEN = 6

SINGLEDOC_PROMPT = (
    "The following lines contains a single episode transcript of a Dungeons and Dragons game played by a group of players. "
    "The episode transcript is delimited by [START OF EPISODE] and [END OF EPISODE]. "
    "The transcript is followed by a question about the game statistics. "
    "Answer the question based on the transcript. "
    "Do not try to guess, estimate or approximate the result. "
    "Do not ask the user for clarification or follow-up. "
    "Do step-by-step reasoning if needed. "
    "Return the final answer in \\boxed{{}}. "
)
MULTIDOC_PROMPT = (
    "The following lines contains multiple episode transcripts of a Dungeons and Dragons game played by a group of players. "
    "Each episode transcript is delimited by [START OF EPISODE] and [END OF EPISODE]. "
    "The transcripts are followed by a question about the game statistics. "
    "Answer the question based on the transcripts. "
    "Do not try to guess, estimate or approximate the result. "
    "Do not ask the user for clarification or follow-up. "
    "Do step-by-step reasoning if needed. "
    "Return the final answer in \\boxed{{}}. "
)

ANSWER_FORMAT = (
    "Answer format: set answer[\"content\"] to the final answer only -- an integer for counts and percentages, the "
    "exact name for a single item (a roll type or a spell), or a comma-separated list when the question asks for a "
    "list or several items tie. No explanation (\\boxed{} is accepted but not needed)."
)


def root_prompt_for(question: str, n_episodes: int) -> str:
    what = "one episode transcript" if n_episodes == 1 else f"{n_episodes} episode transcripts"
    return (
        "`context` is the full input of a long-context question about a Dungeons and Dragons game: instructions, the "
        f"player -> character mapping, then {what} (one utterance per line, `Player: text`), each between "
        "[START OF EPISODE] and [END OF EPISODE]. Answer this question about the game statistics:\n\n"
        f"{question}\n\n{ANSWER_FORMAT}"
    )


def build_context(n_episodes: int, mapping_lines: list[str], episode_texts: list[str]) -> str:
    head = SINGLEDOC_PROMPT if n_episodes == 1 else MULTIDOC_PROMPT
    p2c = ("The following lines contain the mapping between player names and character names.\n"
           + "\n".join(mapping_lines) + "\n\nUse this mapping when answering the questions below.\n\n")
    head = head + "\n\n" + p2c
    return head + "\n\n".join(f"[START OF EPISODE]\n{t}\n[END OF EPISODE]" for t in episode_texts)


@dataclass
class PC:
    player: str
    char: str
    cls: str
    spells: list[str]
    weapon: str
    mods: dict[str, int]
    atk: int
    familiar: str | None = None


def _char_name(rng: random.Random, used: set[str]) -> str:
    for _ in range(50):
        n = rng.choice(W._SYL_A) + rng.choice(W._SYL_B) + rng.choice(W._SYL_C)
        if n not in used and len(n) >= 4:
            used.add(n)
            return n
    n = f"{rng.choice(W._SYL_A)}{rng.choice(W._SYL_B)}{len(used)}"
    used.add(n)
    return n


def make_cast(rng: random.Random) -> tuple[str, list[PC]]:
    names = rng.sample(W.PLAYER_NAMES, 8)
    dm = rng.choice(W.DM_NAMES)
    n_pc = rng.choice([4, 5, 5, 6, 6, 7])
    used: set[str] = set()
    classes = rng.sample(W.CASTER_CLASSES, max(2, n_pc // 2 + rng.randint(0, 1)))
    classes += rng.sample(W.MARTIAL_CLASSES, min(len(W.MARTIAL_CLASSES), n_pc - len(classes))) if n_pc > len(classes) else []
    while len(classes) < n_pc:
        classes.append(rng.choice(W.CASTER_CLASSES))
    rng.shuffle(classes)
    pcs = []
    fam_done = False
    for i in range(n_pc):
        cls = classes[i]
        pool = W.CLASSES[cls]["spells"]
        spells = rng.sample(pool, min(len(pool), rng.randint(5, 9))) if pool else []
        mods = {k: rng.randint(-1, 9) for k in W.SKILLS + W.ABILITY_CHECKS + W.SAVES + W.TOOLS}
        mods["Initiative"] = rng.randint(-1, 6)
        pc = PC(player=names[i], char=_char_name(rng, used), cls=cls, spells=spells,
                weapon=rng.choice(W.CLASSES[cls]["weapons"]), mods=mods, atk=rng.randint(4, 11))
        if cls in ("wizard", "warlock") and not fam_done and rng.random() < 0.6:
            pc.familiar = rng.choice(W.FAMILIAR_NAMES)
            fam_done = True
        pcs.append(pc)
    return dm, pcs


@dataclass
class Episode:
    lines: list[str] = field(default_factory=list)
    rolls: list[dict] = field(default_factory=list)
    spells: list[dict] = field(default_factory=list)
    distractors: dict[int, dict] = field(default_factory=dict)
    spell_refs: dict[int, int] = field(default_factory=dict)
    type_lines: dict[int, int] = field(default_factory=dict)


class _Gen:
    """Builds one window (all its episodes share the segment-uniqueness bookkeeping)."""

    def __init__(self, rng: random.Random, dm: str, pcs: list[PC], numwords: float = 0.25):
        self.rng = rng
        self.dm = dm
        self.pcs = pcs
        self.numwords = numwords
        self.seg_seen: Counter = Counter()
        self.protected: set[str] = set()
        self.ep: Episode | None = None
        self.lingering: list = []
        self.last_cast: dict = {}
        self.names = ({pc.char for pc in pcs} | {pc.player for pc in pcs} | {s.split(" ")[0] for s in W.SPELL_LEVEL}
                      | {"Bonus", "Two", "First", "Swinging"} - {"Bonus", "Two", "First", "Swinging"})

    @staticmethod
    def _segs(line: str) -> list[str]:
        """segments(line, INDEX_MIN_LEN) for normalised one-line text."""
        out = []
        for p in _SPLIT.split(line):
            p = p.strip()
            if len(p) >= INDEX_MIN_LEN:
                out.append(p)
        return out

    def _emit(self, speaker: str, make, required: bool) -> int | None:
        """Append a line with at least one new sentence, so the oracle can locate it. Non-unique filler is dropped."""
        r = self.rng
        line, segs, fresh, text = "", [], [], ""
        for attempt in range(40 if required else 4):
            text = make()
            if attempt >= (8 if required else 1):
                text = text.rstrip() + r.choice(W.TAILS[4:])
            if attempt >= (20 if required else 3):
                text = r.choice(W.LEADS[4:]) + text
            line = " ".join(f"{speaker}: {text}".split())
            segs = self._segs(line)
            if any(x in self.protected for x in segs):
                continue
            fresh = [x for x in segs if self.seg_seen[x] == 0]
            if fresh:
                break
        else:
            if not required:
                return None
            line = f"{speaker}: {text} (that's {r.randint(100, 999)} by my count)"
            segs = self._segs(line)
            fresh = segs[-1:]
        for x in segs:
            self.seg_seen[x] += 1
        self.protected.add(max(fresh, key=len))
        self.ep.lines.append(line)
        return len(self.ep.lines) - 1

    def say(self, speaker: str, text: str) -> int | None:
        """Filler / connective line."""
        return self._emit(speaker, lambda: text, required=False)

    def say_key(self, speaker: str, make) -> int:
        """Evidence line (event anchor / distractor / type line)."""
        return self._emit(speaker, make, required=True)

    def num(self, n: int) -> str:
        if 0 <= n <= 40 and self.rng.random() < self.numwords:
            return W.NUM_WORDS[n]
        return str(n)

    def tail(self) -> str:
        return self.rng.choice(W.TAILS)

    def lead(self) -> str:
        return self.rng.choice(W.LEADS)

    def led(self, core: str) -> str:
        """Optional spoken lead-in ("Uh, ", "Okay, ") with the first word lower-cased when it is not a name."""
        ld = self.rng.choice(W.LEADS)
        if ld and core[:1].isupper() and not core.startswith(("I ", "I'", "Nat")) and core.split(" ")[0] not in self.names:
            core = core[0].lower() + core[1:]
        return ld + core

    def roll(self, line: int, pc: PC, rtype: str, total: int | None, nat: int | None, type_line: int | None = None,
             char: str | None = None) -> None:
        self.ep.rolls.append({"line": line, "char": char or pc.char, "player": pc.player, "type": rtype,
                              "total": total, "nat": nat})
        if type_line is not None and type_line != line:
            self.ep.type_lines.setdefault(line, type_line)

    def spell(self, line: int, pc: PC, name: str, cast: int | None = None, ref_line: int | None = None) -> None:
        base = W.SPELL_LEVEL[name]
        if cast is None:
            cast = base
        self.ep.spells.append({"line": line, "char": pc.char, "player": pc.player, "spell": name,
                               "base": "Cantrip" if base == 0 else base, "cast": "Cantrip" if base == 0 else cast})
        if ref_line is not None:
            self.ep.spell_refs[line] = ref_line
        self.last_cast[pc.char] = (name, line, cast)

    def recast(self, pc: PC) -> bool:
        """'Same thing again.' -- a new cast of the PC's previous spell, named only on an earlier line."""
        prev = self.last_cast.get(pc.char)
        if not prev or prev[1] >= len(self.ep.lines) or len(self.ep.lines) - prev[1] > 60:
            return False
        name, ref, lvl = prev
        r = self.rng
        idx = self.say_key(pc.player, lambda: self.led(r.choice([
            "I'm going to do that again.", "Same spell again, on the other one.", "Again! I cast it again.",
            "I do the exact same thing as last turn.", "Cast it one more time.", "Same as before, I cast that again."]))
            + self.tail())
        self.spell(idx, pc, name, lvl if isinstance(lvl, int) else None, ref_line=ref)
        return True

    def distract(self, line: int, kind: str, fake: dict | None = None) -> None:
        self.ep.distractors[line] = {"kind": kind, "fake": fake}

    def fill(self, n: int) -> None:
        r = self.rng
        for _ in range(n):
            u = r.random()
            if u < 0.30:
                self.say(self.dm, self.narr())
            elif u < 0.62:
                pc = r.choice(self.pcs)
                self.say(pc.player, self.led(self.pc_talk(pc)))
            elif u < 0.92:
                c = r.choice(W.CHATTER)
                if r.random() < 0.5:
                    c = c + " " + r.choice(W.CHATTER)
                self.say(r.choice(self.pcs).player, c)
            else:
                self.say(self.dm, r.choice(W.DM_SHORT))

    def narr(self) -> str:
        r = self.rng
        return " ".join(self._narr1() for _ in range(r.choice([1, 1, 2, 2, 3, 4])))

    def _narr1(self) -> str:
        r = self.rng
        npc = r.choice(W.NPCS)
        return r.choice(W.DM_NARR).format(place=r.choice(W.PLACES), adj=r.choice(W.ADJ), sound=r.choice(W.SOUNDS),
                                          npc=npc, npc_cap=npc[0].upper() + npc[1:], npc2=r.choice(W.NPCS),
                                          obj=r.choice(W.OBJECTS))

    def pc_talk(self, pc: PC) -> str:
        r = self.rng
        other = r.choice([p for p in self.pcs if p is not pc] or self.pcs)
        return r.choice(W.PC_TALK).format(npc=r.choice(W.NPCS), obj=r.choice(W.OBJECTS), place=r.choice(W.PLACES),
                                          pc=other.char)

    def d20_text(self, total: int, nat: int, say_nat: bool, cue: str = "") -> str:
        """Player line stating a d20 result. ``cue`` (e.g. 'perception') names the roll type in the line itself."""
        r = self.rng
        T = self.num(total)
        if nat == 20:
            core = r.choice(["Natural 20!", "Nat 20!", "Natural twenty!", f"Natural 20, that's {T}.",
                             f"Nat 20, so {T}.", "I rolled a natural 20!"])
        elif nat == 1:
            core = r.choice(["Natural one.", "Nat 1.", "Natural 1. Of course.", "I rolled a 1. Natural.",
                             f"Natural one, so {T}."])
        elif say_nat:
            mod = total - nat
            sign = f"plus {mod}" if mod >= 0 else f"minus {-mod}"
            core = r.choice([f"{self.num(nat)} {sign}, {T}.", f"Rolled a {nat}, so {T}.", f"{T}. Natural {nat}.",
                             f"{T}, I rolled a {nat}.", f"Natural {nat}, {T} total.", f"{T}, the die says {nat}.",
                             f"An {nat} on the die, {T}." if nat in (8, 11, 18) else f"A {nat} on the die, {T}.",
                             f"{self.num(nat)} before modifiers, so {T}.", f"{T} with a {nat} on the d20."])
        else:
            core = r.choice([f"{T}.", f"That's a {T}.", f"{T} total.", f"I got a {T}.", f"{T}, I think.",
                             f"That is {T} for me.", f"{T}!"])
        if cue:
            core = r.choice([f"{cue[0].upper() + cue[1:]}, {core[0].lower() + core[1:]}", f"{core.rstrip('.!')} on {cue}."])
        return self.led(core) + self.tail()

    def d20(self, pc: PC, rtype: str) -> tuple[int, int, bool]:
        nat = self.rng.randint(1, 20)
        mod = pc.mods.get(rtype, pc.atk if "Attack" in rtype else self.rng.randint(0, 6))
        say_nat = nat in (1, 20) or self.rng.random() < 0.25
        return nat + mod, nat, say_nat

    def check(self, pc: PC, rtype: str, *, asked: bool = True, group_line: int | None = None) -> None:
        """One skill / ability / save / tool check by ``pc``."""
        r = self.rng
        spoken = r.choice(W.SPOKEN.get(rtype, [rtype.lower()]))
        type_line = group_line
        if group_line is None and asked:
            if r.random() < 0.75:
                type_line = self.say_key(self.dm, lambda: r.choice([
                    f"{pc.char}, make {spoken}.", f"Go ahead and roll {spoken}.", f"Roll {spoken} for me.",
                    f"I need {spoken} from you, {pc.char}.", f"Make {spoken}.", f"Let me see {spoken}."]))
            else:
                type_line = self.say_key(pc.player, lambda: r.choice([
                    f"Can I make {spoken}?", f"Can I roll {spoken}?", f"I want to roll {spoken}.",
                    f"I'm going to try {spoken}."]))
                if r.random() < 0.5:
                    self.say(self.dm, r.choice(["Sure.", "Go for it.", "Yeah, go ahead.", "You can."]))
        adv = r.random() < 0.08 and rtype not in ("Initiative", "Death Save")
        cue = r.choice(W.SPOKEN.get(rtype, [rtype.lower()]))
        cue = cue.split(" check")[0].replace("a ", "", 1) if cue.startswith(("a ", "an ")) else cue.split(" check")[0]
        in_line = (not asked and group_line is None) or r.random() < 0.15
        if adv:
            t1, n1, s1 = self.d20(pc, rtype)
            t2, n2, s2 = self.d20(pc, rtype)
            hi, lo = (t1, t2) if t1 >= t2 else (t2, t1)
            word = r.choice(["advantage", "disadvantage"])
            keep = hi if word == "advantage" else lo
            idx = self.say_key(pc.player, lambda: self.led(r.choice([
                f"With {word}: {self.num(t1)} and {self.num(t2)}. So {self.num(keep)}.",
                f"I have {word}, so {t1} and {t2}, I'll take the {keep}.",
                f"{t1} and {t2} with {word}.",
                f"First roll {t1}, second roll {t2}, {word}."])) + (f" That's {cue}." if in_line else ""))
            self.roll(idx, pc, rtype, t1, n1 if s1 and n1 in (1, 20) else None, type_line if not in_line else None)
            self.roll(idx, pc, rtype, t2, n2 if s2 and n2 in (1, 20) else None, type_line if not in_line else None)
        else:
            total, nat, say_nat = self.d20(pc, rtype)
            idx = self.say_key(pc.player, lambda: self.d20_text(total, nat, say_nat, cue if in_line else ""))
            self.roll(idx, pc, rtype, total, nat if say_nat else None, type_line if not in_line else None)
        if r.random() < 0.6:
            self.say(self.dm, r.choice(["Okay.", "Cool.", "All right.", "Mm-hmm.", "Got it."]) + " " + self.narr())

    def familiar_check(self, pc: PC) -> None:
        r = self.rng
        rtype = r.choice(["Perception", "Stealth", "Perception"])
        total, nat, say_nat = self.d20(pc, rtype)
        idx = self.say_key(pc.player, lambda: r.choice([
            f"{pc.familiar} is going to look around. {rtype} for {pc.familiar}: {self.num(total)}.",
            f"I'll have {pc.familiar} check it out. {pc.familiar} rolls {rtype.lower()}, {self.num(total)}.",
            f"Sending {pc.familiar} ahead. {self.num(total)} on {pc.familiar}'s {rtype.lower()}."]))
        self.roll(idx, pc, rtype, total, nat if say_nat and nat in (1, 20) else None, None, char=pc.familiar)

    def scene_explore(self) -> None:
        r = self.rng
        self.fill(r.randint(6, 16))
        for _ in range(r.randint(2, 4)):
            if r.random() < 0.3:
                rtype = r.choices(W.SKILLS[:3] + ["Dexterity Save", "Constitution Save"], [5, 4, 2, 1, 1])[0]
                spoken = W.SPOKEN[rtype][1] if rtype in W.SKILLS else W.SPOKEN[rtype][0]
                gl = self.say_key(self.dm, lambda spoken=spoken: r.choice([f"Everyone make {spoken}.", f"I need {spoken} from everybody.",
                                                              f"All of you, roll {spoken}.", f"Let's have {spoken} from the whole group."]))
                for pc in r.sample(self.pcs, r.randint(2, len(self.pcs))):
                    self.check(pc, rtype, group_line=gl)
            else:
                pc = r.choice(self.pcs)
                if pc.familiar and r.random() < 0.15:
                    self.familiar_check(pc)
                else:
                    rtype = r.choices(W.SKILLS + W.ABILITY_CHECKS[:3] + W.TOOLS, W.SKILL_W + [1.8, 0.7, 1.1, 0.6])[0]
                    self.check(pc, rtype, asked=r.random() < 0.85)
            self.fill(r.randint(5, 14))
            if r.random() < 0.3:
                self.distract_any()
        for _ in range(r.choice([0, 1, 1, 2])):
            self.utility_spell()
        if r.random() < 0.5:
            self.spell_chatter()

    def spell_chatter(self) -> None:
        """Players talking about spells they have, want or used before (no cast)."""
        r = self.rng
        casters = [p for p in self.pcs if p.spells]
        if not casters:
            return
        pc = r.choice(casters)
        sp = r.sample(pc.spells, min(3, len(pc.spells)))
        if len(sp) >= 2 and r.random() < 0.4:
            d = self.say_key(pc.player, lambda: self.led(r.choice([
                f"I've got {', '.join(self.sp(x) for x in sp[:-1])} and {self.sp(sp[-1])} prepared today.",
                f"Spell list check: {', '.join(sp)}.", f"So my options are {' or '.join(self.sp(x) for x in sp)}."])))
            self.distract(d, "spell_talk", {"spell": {"char": pc.char, "player": pc.player, "spell": sp[0]}})
        else:
            self.distract_spell_talk(pc)

    def distract_spell_talk(self, pc: PC) -> None:
        r = self.rng
        sp = r.choice(pc.spells)
        other = r.choice(self.pcs)
        d = self.say_key(pc.player, lambda: self.led(r.choice([
            f"Should I cast {self.sp(sp)}? No, never mind.", f"I still have {self.sp(sp)} prepared if we need it.",
            f"I could cast {self.sp(sp)}, but it's a waste of a slot.", f"Remind me, does {sp} need concentration?",
            f"I'm out of slots, so no {self.sp(sp)} tonight.", f"{other.char}, can you cast {sp}? I can't.",
            f"Last time I cast {sp} it went terribly.", f"If they come back I'll cast {self.sp(sp)}.",
            f"Is {sp} a bonus action or an action?", f"I really want to cast {sp} on that guy someday.",
            f"Wait, does {self.sp(sp)} work on undead?", f"I'm saving {self.sp(sp)} for the boss."])))
        self.distract(d, "spell_talk", {"spell": {"char": pc.char, "player": pc.player, "spell": sp}})

    def scene_social(self) -> None:
        r = self.rng
        self.fill(r.randint(8, 18))
        for _ in range(r.randint(1, 2)):
            pc = r.choice(self.pcs)
            self.check(pc, r.choice(["Persuasion", "Deception", "Insight", "Intimidation", "Insight"]))
            self.fill(r.randint(6, 14))
        if r.random() < 0.5:
            self.utility_spell()
        if r.random() < 0.4:
            self.spell_chatter()
        if r.random() < 0.5:
            self.distract_any()

    def utility_spell(self) -> None:
        r = self.rng
        casters = [p for p in self.pcs if p.spells]
        if not casters:
            return
        pc = r.choice(casters)
        utility = [s for s in pc.spells if s not in W.ATTACK_SPELLS and s not in W.SAVE_SPELLS] or pc.spells
        name = r.choice(utility)
        lvl, txt = self.cast_level_text(name)
        idx = self.say_key(pc.player, lambda: self.cast_line(name, txt, r.choice(self.pcs).char))
        self.spell(idx, pc, name, lvl)
        if name in W.HEAL_SPELLS:
            amt = r.randint(4, 16)
            hidx = self.say_key(pc.player, lambda: r.choice([f"That's {self.num(amt)} hit points back.",
                                                              f"Heals for {self.num(amt)}.",
                                                              f"{self.num(amt)} healing."]))
            self.roll(hidx, pc, "Healing", amt, None)
        self.say(self.dm, r.choice(["Okay.", "Done.", "You feel the magic settle.", "Sure."]) + " " + self.narr())

    def sp(self, name: str) -> str:
        """How a player says the spell's name (aliases, lower case, 'the X spell')."""
        r = self.rng
        u = r.random()
        if u < 0.35 and name in W.SPELL_ALIASES:
            return r.choice(W.SPELL_ALIASES[name])
        if u < 0.55:
            return name.lower()
        if u < 0.65:
            return f"the {name} spell"
        return name

    def cast_line(self, name: str, txt: str, target: str) -> str:
        r = self.rng
        forms = [f"I'm going to cast {self.sp(name)}{txt}.", f"I cast {self.sp(name)}{txt}.",
                 f"Can I cast {self.sp(name)}{txt} real quick?", f"I'll use {self.sp(name)}{txt} here.",
                 f"Let me throw {self.sp(name)}{txt} on {target}.", f"{name}{txt}, on {target}.",
                 f"Okay, {self.sp(name)}{txt} it is.", f"I'm doing {self.sp(name)}{txt}."]
        if name in W.VERB_SPELLS:
            v = W.VERB_SPELLS[name]
            forms += [f"I'm going to {v} {target}{txt}.", f"I'll {v} {target}{txt}.", f"{v.capitalize()} {target}{txt}!"]
        return self.led(r.choice(forms)) + self.tail()

    def cast_level_text(self, name: str) -> tuple[int, str]:
        base = W.SPELL_LEVEL[name]
        if base == 0 or self.rng.random() > 0.15:
            if base > 0 and self.rng.random() < 0.15:
                return base, f" at {W.ORDINAL[base]} level"
            return base, ""
        lvl = min(5, base + self.rng.randint(1, 2))
        return lvl, self.rng.choice([f" at {W.ORDINAL[lvl]} level", f", using a {W.ORDINAL[lvl]}-level slot",
                                     f" upcast to level {lvl}"])

    def scene_combat(self) -> None:
        r = self.rng
        mons = r.sample(W.MONSTERS, r.randint(1, 2))
        self.lingering = []
        self.say(self.dm, self.narr())
        il = self.say_key(self.dm, lambda: r.choice(["Everyone roll initiative.", "Roll initiative!",
                                                      "Okay, I need everyone to roll initiative.",
                                                      f"{mons[0][0].upper() + mons[0][1:]} lunges. Roll initiative."]))
        for pc in r.sample(self.pcs, len(self.pcs)):
            if r.random() < 0.9:
                self.check(pc, "Initiative", group_line=il)
        if r.random() < 0.3:
            pc = r.choice(self.pcs)
            v = r.randint(5, 25)
            self.say_key(self.dm, lambda: f"{pc.char}, what was your initiative again?")
            d = self.say_key(pc.player, lambda: r.choice([f"{self.num(v)}.", f"It was {v}.", f"I had a {v}."]))
            self.distract(d, "recap", {"roll": {"char": pc.char, "player": pc.player, "type": "Initiative", "total": v}})
        for _ in range(r.randint(2, 3)):
            order = r.sample(self.pcs, len(self.pcs))
            for pc in order:
                if r.random() < 0.3:
                    continue
                if pc.spells and (pc.cls not in ("paladin", "ranger", "rogue") or r.random() < 0.35) and r.random() < 0.85:
                    if not (r.random() < 0.25 and self.recast(pc)):
                        self.cast_in_combat(pc, r.choice(mons))
                else:
                    self.attacks(pc, r.choice(mons))
                if r.random() < 0.35:
                    self.npc_turn(r.choice(mons))
                if self.lingering and r.random() < 0.2:
                    self.ongoing(r.choice(mons))
                self.fill(r.randint(0, 4))
        if r.random() < 0.15:
            pc = r.choice(self.pcs)
            self.say(self.dm, f"{pc.char}, you're unconscious. Make a death saving throw.")
            self.check(pc, "Death Save", group_line=len(self.ep.lines) - 1)
        self.say(self.dm, f"{mons[-1][0].upper() + mons[-1][1:]} crumples to the ground. The fight is over.")
        self.fill(r.randint(2, 5))

    def ongoing(self, mon: str) -> None:
        """A spell already running does something again: NOT a new cast (may carry a roll)."""
        r = self.rng
        pc, name = r.choice(self.lingering)
        if name == "Spiritual Weapon" and r.random() < 0.7:
            t, nat, sn = self.d20(pc, "Spell Attack")
            idx = self.say_key(pc.player, lambda: self.led(r.choice([
                f"My spiritual weapon swings at {mon} again, {t} to hit.", f"Bonus action, the spiritual weapon attacks: {t}.",
                f"Spiritual Weapon goes again. {t} to hit."])))
            self.roll(idx, pc, "Spell Attack", t, nat if nat in (1, 20) else None)
            self.distract(idx, "ongoing_spell", {"spell": {"char": pc.char, "player": pc.player, "spell": name}})
            return
        d = self.say_key(pc.player, lambda: self.led(r.choice([
            f"I'm still concentrating on {self.sp(name)}.", f"{name} is still up, right?",
            f"I move the {name} onto {mon}.", f"Don't forget {mon} is under my {self.sp(name)}.",
            f"Plus the {name} damage, I guess."])))
        self.distract(d, "ongoing_spell", {"spell": {"char": pc.char, "player": pc.player, "spell": name}})

    def attacks(self, pc: PC, mon: str) -> None:
        r = self.rng
        n = 1 if pc.cls in ("rogue",) else r.choice([1, 2, 2, 3] if pc.cls == "fighter" else [1, 1, 2])
        self.say(pc.player, r.choice([f"I'm going to attack {mon}.", f"I run up to {mon}.", f"I go after {mon}.",
                                      f"I move toward {mon} with my {pc.weapon}."]))
        if n >= 2 and r.random() < 0.3:
            a = [self.d20(pc, "Attack") for _ in range(2)]
            idx = self.say_key(pc.player, lambda: self.led(r.choice([
                f"Two attacks: {a[0][0]} and {a[1][0]}.", f"First swing {a[0][0]}, second swing {a[1][0]}.",
                f"{a[0][0]} and {a[1][0]} to hit."])))
            for t, nat, _ in a:
                self.roll(idx, pc, "Attack", t, nat if nat in (1, 20) else None)
            hits = [t for t, _, _ in a if t >= 14]
            if hits:
                self.say(self.dm, r.choice(["Both hit.", "Those hit.", "That's a hit, roll damage."]))
                self.damage(pc, len(hits))
            else:
                self.say(self.dm, "Both miss.")
            return
        for k in range(n):
            t, nat, _ = self.d20(pc, "Attack")
            idx = self.say_key(pc.player, lambda t=t, k=k, nat=nat: self.led(r.choice([
                f"That's a {self.num(t)} to hit.", f"{t} to hit.", f"Does a {t} hit?", f"{W.ORDINAL[k + 1].capitalize()} attack, {t}.",
                f"Swinging with my {pc.weapon}, {t}.", f"{t} on the attack."] + (
                ["Natural 20 to hit!", f"Nat 20 on the attack, {t}!"] if nat == 20 else []) + (
                ["Natural one on the attack.", "Nat 1. Missed."] if nat == 1 else []))) + self.tail())
            self.roll(idx, pc, "Attack", t, nat if nat in (1, 20) else None)
            if r.random() < 0.15:
                e = self.say_key(self.dm, lambda t=t: r.choice([f"{t}? That hits.", f"A {t} hits.", f"{t} just hits."]))
                self.distract(e, "echo", {"roll": {"char": pc.char, "player": pc.player, "type": "Attack", "total": t}})
            elif t >= 14 or nat == 20:
                self.say(self.dm, r.choice(["That hits.", "That'll hit.", "Yep, hits.", "Go ahead and roll damage."]))
            else:
                self.say(self.dm, r.choice(["That misses.", "No, it misses.", "It deflects off the armor."]))
                continue
            if t >= 14 or nat == 20:
                self.damage(pc, 1, crit=nat == 20)

    def damage(self, pc: PC, n: int = 1, crit: bool = False, spell: str | None = None) -> None:
        r = self.rng
        dtype = W.DAMAGE_TYPE.get(spell or "", r.choice(["slashing", "piercing", "bludgeoning"]))
        for _ in range(n):
            amt = r.randint(3, 16) * (2 if crit else 1) + (r.randint(4, 20) if spell in ("Fireball", "Lightning Bolt", "Cone of Cold") else 0)
            a = r.randint(1, max(1, amt - 1))
            idx = self.say_key(pc.player, lambda amt=amt, a=a: self.led(r.choice([
                f"{self.num(amt)} damage.", f"That's {amt} points of {dtype} damage.", f"{amt} {dtype}.",
                f"{a} plus {amt - a}, {amt} damage.", f"{self.num(amt)} {dtype} damage.",
                f"Crit damage, {amt}!" if crit else f"That'll be {amt} damage."])) + self.tail())
            self.roll(idx, pc, "Damage", amt, None)

    def cast_in_combat(self, pc: PC, mon: str) -> None:
        r = self.rng
        att = [x for x in pc.spells if x in W.ATTACK_SPELLS]
        name = r.choice(att) if att and r.random() < 0.5 else r.choice(pc.spells)
        lvl, txt = self.cast_level_text(name)
        if name in W.ATTACK_SPELLS:
            t, nat, sn = self.d20(pc, "Spell Attack")
            if r.random() < 0.5:
                idx = self.say_key(pc.player, lambda: self.led(r.choice([
                    f"I cast {self.sp(name)}{txt} at {mon}. That's a {t} to hit.", f"{self.sp(name)}{txt} at {mon}, {t} to hit.",
                    f"I'm going to throw {self.sp(name)}{txt} at {mon}: {t}.", f"{t} to hit with {self.sp(name)}{txt}."])))
                self.spell(idx, pc, name, lvl)
                self.roll(idx, pc, "Spell Attack", t, nat if nat in (1, 20) else None)
            else:
                idx = self.say_key(pc.player, lambda: self.led(r.choice([
                    f"I'm going to cast {self.sp(name)}{txt} on {mon}.", f"{self.sp(name)}{txt}, targeting {mon}.",
                    f"Can I hit {mon} with {self.sp(name)}{txt}?"])))
                self.spell(idx, pc, name, lvl)
                aidx = self.say_key(pc.player, lambda: self.d20_text(t, nat, nat in (1, 20), "the spell attack" if r.random() < 0.4 else ""))
                self.roll(aidx, pc, "Spell Attack", t, nat if nat in (1, 20) else None, type_line=idx)
            if t >= 13 or nat == 20:
                self.say(self.dm, r.choice(["That hits.", "It strikes true.", "Roll damage."]))
                self.damage(pc, 1 if name != "Scorching Ray" else r.randint(1, 3), crit=nat == 20, spell=name)
            else:
                self.say(self.dm, "It goes wide.")
        else:
            idx = self.say_key(pc.player, lambda: self.cast_line(name, txt, mon) if r.random() < 0.6 else self.led(r.choice([
                f"Bonus action, {self.sp(name)}{txt}.", f"I'm going to cast {self.sp(name)}{txt}, centered on {mon}.",
                f"{name}{txt}!"])) + self.tail())
            self.spell(idx, pc, name, lvl)
            if name in W.LINGERING:
                self.lingering.append((pc, name))
            if name in W.SAVE_SPELLS:
                s = r.randint(3, 24)
                d = self.say_key(self.dm, lambda: r.choice([f"{mon[0].upper() + mon[1:]} rolls a {s} on the save.",
                                                             f"That's a {s} for {mon}.", f"(rolling) {s}. It fails.",
                                                             f"{mon[0].upper() + mon[1:]} gets a natural {min(s, 20)}."]))
                self.distract(d, "dm_roll", {"roll": {"char": "?", "player": self.dm, "type": "Wisdom Save", "total": s}})
                if name in W.DAMAGE_TYPE:
                    self.damage(pc, 1, spell=name)
            elif name in W.HEAL_SPELLS:
                amt = r.randint(4, 18)
                hidx = self.say_key(pc.player, lambda: r.choice([f"{self.num(amt)} hit points.", f"That's {amt} healing.",
                                                                  f"They get {amt} back."]))
                self.roll(hidx, pc, "Healing", amt, None)
            else:
                self.say(self.dm, r.choice(["Okay, that's up.", "Done. What else?", "Got it. Anything else?"]))

    def npc_turn(self, mon: str) -> None:
        r = self.rng
        tgt = r.choice(self.pcs)
        u = r.random()
        if u < 0.45:
            t = r.randint(5, 26)
            d = self.say_key(self.dm, lambda: r.choice([
                f"{mon[0].upper() + mon[1:]} swings at {tgt.char}. That's a {t} to hit.",
                f"{mon[0].upper() + mon[1:]} attacks {tgt.char}: {t}.", f"Against {tgt.char}, {t} to hit.",
                f"(rolls) {t} against {tgt.char}."]))
            self.distract(d, "dm_roll", {"roll": {"char": "?", "player": self.dm, "type": "Attack", "total": t}})
            ac = r.randint(12, 19)
            a = self.say_key(tgt.player, lambda: r.choice([f"My AC is {ac}.", f"That hits, my armor class is {ac}.",
                                                           f"AC {ac}, so it hits.", f"Oh no, {ac} is my AC."]))
            self.distract(a, "ac", {"roll": {"char": tgt.char, "player": tgt.player, "type": "Attack", "total": ac}})
            dmg = r.randint(4, 22)
            self.say(self.dm, f"You take {dmg} points of {r.choice(['slashing', 'piercing', 'bludgeoning', 'acid'])} damage.")
            if r.random() < 0.4:
                hp = r.randint(3, 60)
                h = self.say_key(tgt.player, lambda: r.choice([f"I'm at {hp} hit points.", f"Down to {hp}.",
                                                               f"{hp} hit points left."]))
                self.distract(h, "hp", {"roll": {"char": tgt.char, "player": tgt.player, "type": "Damage", "total": hp}})
        elif u < 0.8:
            st = r.choices(W.SAVES[:3], [3, 2, 3])[0]
            gl = self.say_key(self.dm, lambda: r.choice([f"{mon[0].upper() + mon[1:]} breathes out a cloud. {tgt.char}, make {r.choice(W.SPOKEN[st])}.",
                                                          f"{tgt.char}, I need {r.choice(W.SPOKEN[st])}.",
                                                          f"Make {r.choice(W.SPOKEN[st])}, {tgt.char}."]))
            self.check(tgt, st, group_line=gl)
        elif u < 0.9 and any(x in ("Shield", "Counterspell", "Hellish Rebuke") for x in tgt.spells):
            sp = r.choice([x for x in tgt.spells if x in ("Shield", "Counterspell", "Hellish Rebuke")])
            self.say(self.dm, f"{mon[0].upper() + mon[1:]} lunges at {tgt.char}.")
            lvl, txt = self.cast_level_text(sp)
            idx = self.say_key(tgt.player, lambda: self.led(r.choice([f"Reaction: {sp}{txt}!", f"I cast {sp}{txt} as a reaction.",
                                                                       f"Can I {sp}{txt}? Yes, I'm doing it."])))
            self.spell(idx, tgt, sp, lvl)
            if sp == "Hellish Rebuke":
                self.damage(tgt, 1, spell=sp)
            else:
                self.say(self.dm, r.choice(["It works.", "The attack fizzles against the magic.", "Nice."]))
        else:
            sp = r.choice(["Hold Person", "Counterspell", "Shield", "Fireball", "Darkness", "Misty Step"])
            d = self.say_key(self.dm, lambda: r.choice([f"{mon[0].upper() + mon[1:]} casts {sp}.",
                                                         f"The caster behind {mon} uses {sp}.",
                                                         f"In response, {mon} casts {sp}!"]))
            self.distract(d, "npc_spell", {"spell": {"char": "?", "player": self.dm, "spell": sp}})
            if sp in ("Hold Person", "Fireball"):
                st = "Wisdom Save" if sp == "Hold Person" else "Dexterity Save"
                gl = self.say_key(self.dm, lambda: f"{tgt.char}, make {r.choice(W.SPOKEN[st])}.")
                self.check(tgt, st, group_line=gl)

    def scene_rest(self) -> None:
        r = self.rng
        self.say(self.dm, r.choice(["You find a quiet spot and take a short rest.", "You settle in for a short rest.",
                                    "An hour passes as you rest by the fire."]))
        for pc in r.sample(self.pcs, r.randint(0, 2)):
            k = r.choice([1, 1, 1, 2])
            vals = [r.randint(1, 10) + r.randint(0, 3) for _ in range(k)]
            if k == 1:
                idx = self.say_key(pc.player, lambda vals=vals: r.choice([f"I'll spend a hit die. {self.num(vals[0])}.",
                                                                 f"One hit die: {vals[0]} hit points.",
                                                                 f"Hit die, that's {vals[0]}."]))
            else:
                idx = self.say_key(pc.player, lambda vals=vals: r.choice([f"Two hit dice: {vals[0]} and {vals[1]}.",
                                                                 f"I'm spending two hit dice, {vals[0]} and {vals[1]}."]))
            for v in vals:
                self.roll(idx, pc, "Hit Dice", v, None)
        self.fill(r.randint(2, 6))
        pc = r.choice(self.pcs)
        g = r.randint(5, 250)
        d = self.say_key(pc.player, lambda: r.choice([f"I have {g} gold left.", f"That's {g} gold, total.",
                                                       f"We split the {g} gold pieces?"]))
        self.distract(d, "money", {"roll": {"char": pc.char, "player": pc.player, "type": "Other", "total": g}})
        for _ in range(r.choice([1, 1, 2])):
            self.utility_spell()
        self.fill(r.randint(2, 6))

    def distract_any(self) -> None:
        r = self.rng
        pc = r.choice(self.pcs)
        u = r.random()
        if u < 0.08 and any(p.spells for p in self.pcs):
            c = r.choice([p for p in self.pcs if p.spells])
            sp = r.choice(c.spells)
            d = self.say_key(pc.player, lambda: r.choice([f"{c.char}, you should cast {self.sp(sp)}!",
                                                          f"Wait, {c.char} has {self.sp(sp)}, right?",
                                                          f"Do you still have {self.sp(sp)}, {c.char}?"]))
            self.distract(d, "spell_talk", {"spell": {"char": c.char, "player": c.player, "spell": sp}})
        elif u < 0.25:
            v = r.choice([15, 18, 20, 20, 10, 25])
            rtype = r.choice(["Athletics", "Acrobatics", "Persuasion", "Stealth"])
            d = self.say_key(pc.player, lambda: r.choice([
                f"If I roll a {v} on this, can I make the jump?", f"What would I need, like a {v}?",
                f"Hypothetically, a {v} on {rtype.lower()} would do it, right?", f"I'd need a natural {v} for that."]))
            self.distract(d, "hypothetical", {"roll": {"char": pc.char, "player": pc.player, "type": rtype, "total": v}})
            self.say(self.dm, r.choice(["Probably.", "Maybe.", "We'll see.", "Something like that."]))
        elif u < 0.45:
            rtype = r.choice(W.SKILLS[:6])
            sp = r.choice(W.SPOKEN[rtype])
            self.say_key(self.dm, lambda: r.choice([f"Make {sp}, {pc.char}.", f"{pc.char}, roll {sp}."]))
            d = self.say_key(pc.player, lambda: r.choice(["Actually, never mind, I don't do it.",
                                                          "Wait, no, I change my mind. No roll.",
                                                          "Hold on, I'm not going in there after all."]))
            self.distract(d, "unmade", None)
            self.say(self.dm, r.choice(["Okay, no roll then.", "Fair enough, you hang back.", "All right."]))
        elif u < 0.65:
            v = r.choice([1, 20, 3, 2])
            d = self.say_key(pc.player, lambda: r.choice([
                f"These dice rolled three {'ones' if v == 1 else str(v) + 's'} before the show.",
                f"I swear this die only rolls {v}s.", f"Remember when I rolled a natural {v} last month?",
                f"New dice tonight. They better roll {v}s."]))
            self.distract(d, "ooc_dice", {"roll": {"char": pc.char, "player": pc.player, "type": "Other", "total": v}})
        elif u < 0.85 and pc.spells:
            self.distract_spell_talk(pc)
        else:
            v = r.randint(5, 120)
            d = self.say_key(pc.player, lambda: r.choice([f"How far is it? Like {v} feet?", f"There are {v} of them?",
                                                          f"I move {v} feet toward the door."]))
            self.distract(d, "number", {"roll": {"char": pc.char, "player": pc.player, "type": "Other", "total": v}})

    def scene_break(self) -> None:
        r = self.rng
        self.say(self.dm, r.choice(["All right, let's take a quick break. We'll be back in a few minutes.",
                                    "And on that note, we're going to take a break.",
                                    "Let's pause there for a break, everybody."]))
        for _ in range(r.randint(2, 5)):
            self.say(r.choice(self.pcs).player, r.choice(W.OOC))
        self.distract_any()
        self.say(self.dm, "And welcome back, everybody. Let's pick up where we left off.")

    def recap(self, prev_rolls: list[dict], prev_spells: list[dict]) -> None:
        r = self.rng
        self.say(self.dm, r.choice(["Hello, everyone, and welcome to tonight's episode!",
                                    "Good evening, everybody, and welcome back!",
                                    "Hey, everyone. Welcome to the game."]))
        self.fill(r.randint(1, 3))
        self.say(self.dm, r.choice(["Before we jump in, a quick recap of last time.", "So, last we left off:",
                                    "Let's recap where we were."]))
        pc = r.choice(self.pcs)
        v = r.choice([20, 1, r.randint(2, 19)])
        rt = r.choice(["Perception", "Persuasion", "Stealth", "Attack"])
        d = self.say_key(self.dm, lambda: r.choice([f"Last time, {pc.char} rolled a natural {v} on {rt.lower()}.",
                                                     f"You may remember {pc.char}'s {v} on that {rt.lower()} check.",
                                                     f"{pc.char} rolled a {v} to sneak past the guards last episode."]))
        self.distract(d, "recap", {"roll": {"char": pc.char, "player": pc.player, "type": rt, "total": v}})
        casters = [p for p in self.pcs if p.spells]
        if casters:
            c = r.choice(casters)
            sp = r.choice(c.spells)
            d = self.say_key(self.dm, lambda: r.choice([f"{c.char} cast {sp} to get you all out of there.",
                                                         f"Thanks to {c.char}'s {sp}, you survived the ambush."]))
            self.distract(d, "recap_spell", {"spell": {"char": c.char, "player": c.player, "spell": sp}})
        self.fill(r.randint(2, 5))
        self.say(self.dm, "And that's where we pick up tonight.")

    def episode(self, target_lines: int, first: bool) -> Episode:
        self.ep = Episode()
        self.last_cast = {}
        r = self.rng
        self.recap([], []) if (not first or r.random() < 0.7) else self.fill(4)
        broke = False
        while len(self.ep.lines) < target_lines:
            u = r.random()
            if u < 0.40:
                self.scene_explore()
            elif u < 0.57:
                self.scene_social()
            elif u < 0.69:
                self.scene_combat()
            elif u < 0.78:
                self.scene_rest()
            elif u < 0.83 and not broke and len(self.ep.lines) > target_lines // 3:
                self.scene_break()
                broke = True
            else:
                self.fill(r.randint(10, 25))
                if r.random() < 0.5:
                    self.distract_any()
        while len(self.ep.spells) < 3:
            self.utility_spell()
        self.say(self.dm, r.choice(["And that's where we'll end tonight. Thank you all for watching!",
                                    "We'll pick it up there next week. Have a wonderful night, everyone.",
                                    "Good game, everybody. See you next time."]))
        return self.ep


def _mode(values: list) -> list:
    c = Counter(values)
    if not c:
        return []
    m = max(c.values())
    return sorted(str(v) for v, n in c.items() if n == m)


def _least_more_than_one(values: list) -> list:
    c = Counter(values)
    c = {k: n for k, n in c.items() if n > 1}
    if not c:
        return []
    m = min(c.values())
    return sorted(str(v) for v, n in c.items() if n == m)


def _least(values: list) -> list:
    c = Counter(values)
    if not c:
        return []
    m = min(c.values())
    return sorted(str(v) for v, n in c.items() if n == m)


def _first_by(rows: list[dict], key: str, val: str, last: bool = False) -> dict:
    groups: dict[str, str] = {}
    for x in rows:
        if last or x[key] not in groups:
            groups[x[key]] = x[val]
    return groups


def _join(items: list) -> str:
    return ", ".join(str(x) for x in items)


@dataclass
class Q:
    tid: str
    question: str
    gold: str
    family: str
    kind: str
    req: dict = field(default_factory=dict)


SINGLE_W = {
    "r_total": 2, "r_char": 4, "r_player": 3, "r_type": 4, "r_nat": 3, "r_pct": 3, "r_mc_type": 2, "r_lc_type": 2,
    "r_mc_nat": 0, "r_lc_nat": 2, "r_crits": 2, "r_nat20": 2, "r_nat1": 2,
    "s_total": 2, "s_char": 4, "s_player": 3, "s_spell": 4, "s_first1": 2, "s_first2": 2, "s_first3": 2,
    "s_last1": 2, "s_last2": 2, "s_last3": 2, "s_first_each": 2, "s_last_each": 2, "s_nchars": 4, "s_mc": 2,
    "s_lc": 2, "s_multi": 1, "s_cantrip": 2, "s_upcast_n": 2, "s_upcast_list": 1,
}
MULTI_W = {
    "r_cum": 5, "r_cum_char": 8, "r_total": 2, "r_char": 3, "r_player": 2, "r_type": 3, "r_nat": 3, "r_pct": 3,
    "r_mc_type": 2, "r_lc_type": 2, "r_mc_nat": 0, "r_lc_nat": 2, "r_crits": 2, "r_nat20": 2, "r_nat1": 2,
    "s_cum": 3, "s_kth": 9, "s_first_each_ep": 2, "s_last_each_ep": 2, "s_char_first_each_ep": 2,
    "s_char_last_each_ep": 2, "s_total": 2, "s_char": 3, "s_player": 2, "s_spell": 3, "s_nchars": 3, "s_mc": 2,
    "s_lc": 2, "s_cantrip": 2, "s_upcast_n": 2, "s_upcast_list": 1,
}


def make_question(rng: random.Random, tid: str, rolls: list[dict], spells: list[dict], n_eps: int,
                  p2c: dict[str, str]) -> Q | None:
    """Question in OOLONG-real's wording with its gold, computed like upstream prepare_cr_questions.py. None if degenerate."""
    single = n_eps == 1
    scope = "in this episode" if single else "across all episodes"
    c2p = {c: p for p, c in p2c.items() if c != "DM"}
    rchars = sorted({x["char"] for x in rolls})
    schars = sorted({x["char"] for x in spells})
    rtypes = sorted({x["type"] for x in rolls})
    snames = sorted({x["spell"] for x in spells})
    nats = [x["nat"] for x in rolls if x["nat"] is not None]

    def q(question, gold, family, kind, **req):
        return Q(tid, question, str(gold), family, kind, req)

    if tid == "r_total":
        return q("Total number of rolls in this episode?" if single else "Total number of rolls across all the episodes?",
                 len(rolls), "rolls", "count")
    if tid in ("r_char", "r_player"):
        cands = [c for c in rchars if tid == "r_char" or c in c2p]
        if not cands:
            return None
        c = rng.choice(cands)
        n = sum(1 for x in rolls if x["char"] == c)
        who = f"character {c}" if tid == "r_char" else f"player {c2p[c]}"
        return q(f"Total number of rolls by the {who} {scope}?", n, "rolls", "count", char=c)
    if tid == "r_type":
        t = rng.choice(rtypes)
        return q(f"Total number of rolls of type {t} {scope}?", sum(1 for x in rolls if x["type"] == t), "rolls", "count")
    if tid == "r_nat":
        v = rng.randint(1, 20)
        return q(f"Number of rolls of natural value {v} {scope}?", sum(1 for n in nats if n == v), "rolls", "count")
    if tid == "r_pct":
        v = rng.choice([1, 20, rng.randint(1, 20), rng.randint(1, 20)])
        pct = round(sum(1 for n in nats if n == v) / max(1, len(rolls)) * 100)
        return q(f"In this episode, what percentage of rolls were of value {v}? round to the nearest integer." if single else
                 f"Across all episodes, what percentage of rolls were of value {v}? round to the nearest integer.",
                 pct, "rolls", "count")
    if tid == "r_mc_type":
        return q(f"What is the most common roll type {scope}? Return a comma separated list.",
                 _join(_mode([x["type"] for x in rolls])), "rolls", "list")
    if tid == "r_lc_type":
        g = _least_more_than_one([x["type"] for x in rolls])
        return None if not g else q(f"What is the least common roll type {scope}? Only include types with more than "
                                    "one roll. Return a comma separated list.", _join(g), "rolls", "list")
    if tid == "r_mc_nat":
        g = _mode(nats)
        return None if not g else q(f"What is the most common natural roll value {scope}? Return a comma separated list.",
                                    _join(g), "rolls", "list")
    if tid == "r_lc_nat":
        g = _least_more_than_one(nats)
        return None if not g else q(f"What is the least common natural roll value {scope}? Only include values with "
                                    "more than one roll. Return a comma separated list.", _join(g), "rolls", "list")
    if tid in ("r_crits", "r_nat20", "r_nat1"):
        vals = {"r_crits": (1, 20), "r_nat20": (20,), "r_nat1": (1,)}[tid]
        n = sum(1 for v in nats if v in vals)
        text = {
            ("r_crits", True): "What is the count of Crits? (natural rolls of value 1 or 20)?",
            ("r_nat20", True): "What is the count of Nat20s (natural rolls of value 20)?",
            ("r_nat1", True): "What is the count of Nat1s (natural rolls of value 1)?",
            ("r_crits", False): "What is the total count of Crits across all episodes? (natural rolls of value 1 or 20)?",
            ("r_nat20", False): "What is the total count of Nat20s across all episodes? (natural rolls of value 20)?",
            ("r_nat1", False): "What is the total count of Nat1s across all episodes? (natural rolls of value 1)?",
        }[(tid, single)]
        return q(text, n, "rolls", "count")
    if tid in ("r_cum", "r_cum_char"):
        if single:
            return None
        k = rng.randint(1, n_eps)
        rows = [x for x in rolls if x["ep"] < k]
        if tid == "r_cum":
            return q(f"What is the cummulative total of rolls by the end of episode {k}? Count the number of rolls and "
                     "not the values of the rolls.", len(rows), "rolls", "count", max_ep=k - 1)
        c = rng.choice(rchars)
        return q(f"What is the cummulative total of rolls by the character {c} at the end of episode {k}? Count the "
                 "number of rolls and not the values of the rolls.", sum(1 for x in rows if x["char"] == c), "rolls", "count",
                 char=c, max_ep=k - 1)
    if not spells:
        return None
    if tid == "s_total":
        return q("How many spells were cast during this episode?" if single else "How many spells were cast across all episodes?",
                 len(spells), "spells", "count")
    if tid in ("s_char", "s_player"):
        cands = [c for c in schars if tid == "s_char" or c in c2p]
        if not cands:
            return None
        c = rng.choice(cands)
        who = f"character {c}" if tid == "s_char" else f"player {c2p[c]}"
        return q(f"How many spells were cast by the {who} {scope}?", sum(1 for x in spells if x["char"] == c),
                 "spells", "count", char=c)
    if tid == "s_spell":
        s = rng.choice(snames)
        return q(f"How many {s} spells were cast {'during this episode' if single else 'across all episodes'}?",
                 sum(1 for x in spells if x["spell"] == s), "spells", "count")
    if tid == "s_nchars":
        s = rng.choice(snames)
        return q(f"How many characters cast {s} spell {scope}?", len({x["char"] for x in spells if x["spell"] == s}),
                 "spells", "count")
    if tid in ("s_first1", "s_first2", "s_first3", "s_last1", "s_last2", "s_last3"):
        if not single:
            return None
        k = int(tid[-1])
        seq = [x["spell"] for x in spells]
        if len(seq) < k:
            return None
        part = seq[:k] if "first" in tid else seq[-k:]
        word = {1: "", 2: " two", 3: " three"}[k]
        fl = "first" if "first" in tid else "last"
        text = f"What is the {fl} spell cast in this episode?" if k == 1 else f"What are the {fl}{word} spells cast in this episode?"
        return q(text, _join(part), "spells", "name" if k == 1 else "list")
    if tid in ("s_first_each", "s_last_each"):
        if not single:
            return None
        g = _first_by(spells, "char", "spell", last=tid == "s_last_each")
        fl = "first" if tid == "s_first_each" else "last"
        return q(f"What is the {fl} spell cast by each character in this episode? Return a comma separated list and "
                 "retain the order of spells as they appear in the episode.", _join(g[c] for c in sorted(g)), "spells", "list")
    if tid == "s_mc":
        return q(f"What is the most common spell {scope}? Return a comma separated list.",
                 _join(_mode([x["spell"] for x in spells])), "spells", "list")
    if tid == "s_lc":
        return q(f"What is the least common spell {scope}? Only include spells that were cast at least once. Return a "
                 "comma separated list.", _join(_least([x["spell"] for x in spells])), "spells", "list")
    if tid == "s_multi":
        g = [s for s in dict.fromkeys(x["spell"] for x in spells) if len({x["char"] for x in spells if x["spell"] == s}) > 1]
        return None if not g else q("Which spells were cast by more than one character in this episode? Return a comma "
                                    "separated list.", _join(g), "spells", "list")
    if tid == "s_cantrip":
        return q(f"What is the total number of cantrip spells cast {'in this episode' if single else 'across all episodes'}?",
                 sum(1 for x in spells if x["base"] == "Cantrip"), "spells", "count")
    def numeric_level(value):
        if isinstance(value, bool) or value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    up = [x for x in spells if numeric_level(x["base"]) is not None
          and numeric_level(x["cast"]) is not None
          and numeric_level(x["cast"]) > numeric_level(x["base"])]
    if tid == "s_upcast_n":
        return q("In this episode, how many times was a spell cast at a level higher than its base level?" if single else
                 "Across all episodes, how many times was a spell cast at a level higher than its base level?",
                 len(up), "spells", "count")
    if tid == "s_upcast_list":
        g = list(dict.fromkeys(x["spell"] for x in up))
        return None if not g else q(
            ("In this episode, which spells were cast at a level higher than their base level?" if single else
             "Across all episodes, which spells were cast at a level higher than their base level?")
            + " Return a comma separated list of unique spells.", _join(g), "spells", "list")
    if single:
        return None
    eps_with = sorted({x["ep"] for x in spells})
    if tid == "s_cum":
        k = rng.randint(1, n_eps)
        return q(f"What is the cummulative total of spells cast by the end of episode {k}?",
                 sum(1 for x in spells if x["ep"] < k), "spells", "count", max_ep=k - 1)
    if tid == "s_kth":
        j = rng.choice([1, 2, 3])
        k = rng.randint(1, n_eps)
        seq = [x["spell"] for x in spells if x["ep"] == k - 1]
        if len(seq) < j:
            return None
        return q(f"What is the {W.ORDINAL[j]} spell cast in the episode {k}?", seq[j - 1], "spells", "name", ep=k - 1)
    if tid in ("s_first_each_ep", "s_last_each_ep"):
        g = _first_by(spells, "ep", "spell", last=tid == "s_last_each_ep")
        fl = "first" if tid == "s_first_each_ep" else "last"
        return q(f"List the {fl} spell cast in each episode? Return a comma separated list.", _join(g[e] for e in eps_with),
                 "spells", "list")
    if tid in ("s_char_first_each_ep", "s_char_last_each_ep"):
        c = rng.choice(schars)
        rows = [x for x in spells if x["char"] == c]
        g = _first_by(rows, "ep", "spell", last=tid == "s_char_last_each_ep")
        fl = "first" if tid == "s_char_first_each_ep" else "last"
        return q(f"List the {fl} spell cast by the character {c} in each episode? Return a comma separated list.",
                 _join(g[e] for e in sorted(g)), "spells", "list", char=c)
    return None


@dataclass
class World:
    context: str
    question: Q
    n_eps: int
    items: list[tuple[str, str]]
    hidden: dict
    required: list[str]


def generate_world(seed_key: str, *, n_episodes: int | None = None, ep_lines: tuple[int, int] = (1000, 1700),
                   p_single: float = 0.3, max_episodes: int = 4, tid: str | None = None,
                   numwords: float = 0.25) -> World:
    rng = random.Random(seed_key)
    dm, pcs = make_cast(rng)
    if n_episodes is None:
        n_episodes = 1 if rng.random() < p_single else rng.randint(2, max(2, max_episodes))
    g = _Gen(rng, dm, pcs, numwords=numwords)
    eps = [g.episode(rng.randint(*ep_lines), first=(i == 0)) for i in range(n_episodes)]

    p2c = {dm: "DM"}
    for pc in pcs:
        p2c[pc.player] = pc.char
    mapping_lines = [f"{p} plays the character {c}." for p, c in p2c.items()]

    items: list[tuple[str, str]] = []
    rolls: list[dict] = []
    spells: list[dict] = []
    ev_by_line: dict[str, list] = {}
    distractors: dict[str, dict] = {}
    type_line: dict[str, str] = {}
    spell_ref: dict[str, str] = {}
    speaker: dict[str, str] = {}
    texts = []
    base = 0
    for e, ep in enumerate(eps):
        for i, line in enumerate(ep.lines):
            lid = f"L{base + i}"
            items.append((lid, line))
            speaker[lid] = line.split(":", 1)[0]
        for x in ep.rolls:
            lid = f"L{base + x['line']}"
            rec = dict(x, line=lid, ep=e, kind="roll")
            rolls.append(rec)
            ev_by_line.setdefault(lid, []).append(rec)
        for x in ep.spells:
            lid = f"L{base + x['line']}"
            rec = dict(x, line=lid, ep=e, kind="spell")
            spells.append(rec)
            ev_by_line.setdefault(lid, []).append(rec)
        for li, d in ep.distractors.items():
            distractors[f"L{base + li}"] = d
        for li, tl in ep.type_lines.items():
            type_line[f"L{base + li}"] = f"L{base + tl}"
        for li, rl in ep.spell_refs.items():
            spell_ref[f"L{base + li}"] = f"L{base + rl}"
        texts.append("\n".join(ep.lines))
        base += len(ep.lines)
    order = {lid: n for n, (lid, _) in enumerate(items)}
    rolls.sort(key=lambda x: order[x["line"]])
    spells.sort(key=lambda x: order[x["line"]])

    weights = SINGLE_W if n_episodes == 1 else MULTI_W
    question = None
    for _ in range(50):
        t = tid or rng.choices(list(weights), list(weights.values()))[0]
        question = make_question(rng, t, rolls, spells, n_episodes, p2c)
        if question is not None:
            break
    assert question is not None, "no valid question"
    context = build_context(n_episodes, mapping_lines, texts)
    hidden = {
        "events": ev_by_line, "distractors": distractors, "type_line": type_line, "spell_ref": spell_ref,
        "speaker": speaker,
        "p2c": p2c, "dm": dm, "familiars": {pc.familiar: pc.char for pc in pcs if pc.familiar},
        "n_rolls": len(rolls), "n_spells": len(spells), "tid": question.tid, "family": question.family, "req": question.req,
        "kind": question.kind, "chars": sorted({pc.char for pc in pcs} | {pc.familiar for pc in pcs if pc.familiar}),
        "roll_types": sorted({x["type"] for x in rolls}), "spell_names": sorted({x["spell"] for x in spells}),
    }
    rq = question.req
    pool = [x for x in (rolls if question.family == "rolls" else spells)
            if ("char" not in rq or x["char"] == rq["char"]) and ("max_ep" not in rq or x["ep"] <= rq["max_ep"])
            and ("ep" not in rq or x["ep"] == rq["ep"])]
    required = sorted({x["line"] for x in pool}, key=lambda s: order[s])
    return World(context=context, question=question, n_eps=n_episodes, items=items, hidden=hidden, required=required)


def tables(world: World) -> tuple[list[dict], list[dict]]:
    """(rolls, spells) in transcript order -- for tests / debugging."""
    rolls, spells = [], []
    for lid, _ in world.items:
        for ev in world.hidden["events"].get(lid, []):
            (rolls if ev["kind"] == "roll" else spells).append(ev)
    return rolls, spells


__all__ = ["ANSWER_FORMAT", "INDEX_MIN_LEN", "MULTIDOC_PROMPT", "SINGLEDOC_PROMPT", "World", "build_context",
           "generate_world", "make_question", "root_prompt_for", "tables", "math"]
