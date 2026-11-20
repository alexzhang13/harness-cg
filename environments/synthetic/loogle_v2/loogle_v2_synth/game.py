"""Game family: synthetic match logs for LooGLE-v2 Game questions, in the eval's log grammar and wording."""

from __future__ import annotations

import random
import re
from collections import Counter

from synth_common.envkit import Generated
from synth_common.oracle import FakeSubLM, refuse_aggregate, rng_for, wants_json

INSTRUCTION = ("Now, I'm going to present you a textual description of a Crafter game and a question, please read the "
               "text and answer the question. Crafter is a survival game in which a traveler explores a jungle, gathering "
               "materials and crafting tools to survive. The text I provide shows the traveler's trajectory in this game. "
               "The text is as follows:\n<text>\n{context}\n</text>\nAfter reading the text, please answer the following "
               "question:\nQuestion: {question} The options are as follows:\n{options}\nPlease select the correct option and "
               "only response a single sentence with the format as follows: \"The correct answer is (insert answer here).\".")
CONTEXT_POINTER = ("[The full document is NOT reproduced here. It is available in the REPL as the `context` string "
                   "variable; query it with `llm_query` over slices.]")

GENERIC = ["T Spawn", "CT Spawn", "A Site", "B Site", "Mid"]
COMMON_POOL = ["A Site", "B Site", "CT Spawn", "T Spawn", "T Side", "Mid", "Catwalk", "Long A", "Short A", "Connector",
               "Lower Tunnels", "Upper Tunnels", "Window", "Ramp", "Stairs", "Pit"]


def map_vocab(rng: random.Random, mapname: str, like: list | None = None) -> dict:
    """A map's callouts: shared names plus 1-2 rare ones. ``like`` makes it differ from another map by one or two names."""
    if like:
        common = list(like)
        for _ in range(rng.randint(1, 2)):
            if rng.random() < 0.5 and len(common) > 4:
                common.pop(rng.randrange(len(common)))
            else:
                extra = [c for c in COMMON_POOL if c not in common]
                if rng.random() < 0.5 and len(common) > 4:
                    common[rng.randrange(len(common))] = rng.choice(extra)
                else:
                    common.append(rng.choice(extra))
    else:
        common = rng.sample(COMMON_POOL, rng.randint(5, 7))
    own = rng.sample(MAPS[mapname], rng.randint(1, 2))
    w = {c: rng.uniform(0.5, 2.0) for c in common}
    tot = sum(w.values())
    for o in own:
        w[o] = tot * rng.uniform(0.03, 0.06)
    return w
MAPS = {
    "Harborline": ["Dockside", "Crane Yard", "Fish Market", "Pier Four", "Boathouse", "Customs", "Seawall", "Lighthouse",
                   "Warehouse Row", "Net Loft", "Tide Gate", "Ferry Ramp", "Chandlery", "Breakwater"],
    "Quarrystone": ["Gravel Pit", "Conveyor", "Blast Shed", "Upper Ledge", "Lower Ledge", "Crusher", "Haul Road",
                    "Drill Tower", "Slag Heap", "Pump House", "Rail Spur", "Scale House", "Cutface", "Silo"],
    "Old Mercado": ["Fountain", "Bell Tower", "Arcade", "Spice Alley", "Cathedral Steps", "Plaza", "Tannery", "Viaduct",
                    "Cistern", "Market Hall", "Clock Arch", "Olive Court", "Weaver Lane", "Granary"],
    "Frostvale": ["Ice Bridge", "Cabin", "Ridge", "Ski Lift", "Lodge", "Frozen Lake", "Pine Gap", "Snow Shed", "Overlook",
                  "Boiler Room", "Cable Car", "Avalanche Wall", "Hut", "Glacier Cut"],
    "Dunerest": ["Oasis", "Sandpit", "Ruins", "Long Dune", "Short Dune", "Caravan", "Well", "Temple Gate", "Obelisk",
                 "Bazaar", "Dry Canal", "Palm Row", "Tent Camp", "Mirage"],
    "Ironworks": ["Foundry", "Smelter", "Catwalk", "Coke Ovens", "Rolling Mill", "Gantry", "Slag Pit", "Boiler",
                  "Pattern Shop", "Ladle Bay", "Tool Crib", "Mill Yard", "Vent Shaft", "Furnace"],
    "Canal Quarter": ["Lock Gate", "Gondola Dock", "Bridge of Sighs", "Glass Works", "Courtyard", "Palazzo", "Rialto",
                      "Back Canal", "Mask Shop", "Library", "Water Stairs", "Garden", "Loggia", "Belfry"],
    "Overpass": ["Toll Booth", "Underpass", "Ramp", "Billboard", "Parking Deck", "Storm Drain", "Rest Stop", "Junction",
                 "Sound Wall", "Gas Station", "Median", "Service Road", "Culvert", "Overlook Deck"],
}
WEAPONS = ["AK-47", "M4A1", "AWP", "USP-S", "Glock-18", "MP9", "Desert Eagle", "Galil AR", "FAMAS", "P250", "MAC-10",
           "Knife", "HE Grenade", "Incendiary Grenade", "SG 553", "AUG"]
_NAMES = ["siwut", "Kiruzuz", "CYQOJA", "Aubytude", "nofe", "jocon", "TEJ", "po3yfimpu", "nalox", "mevaw", "pu",
          "aumanas", "yaqiwegafa", "BIHYR", "dodob", "hidi", "nelevy", "sukaneju", "AOFETO", "goho", "Wapicyh",
          "MAS7H", "NazoM", "jum", "linuq", "mokuhi", "xikihi", "PAB", "WOJAM2", "Cecudywut", "huru", "KEJOk",
          "Nybep", "go1ej", "cequ", "rewu", "Janyze", "Tyvydob", "wuruh", "sobise", "yiqely", "Aozatah", "w3kopykok",
          "iU", "972", "dyfavem", "zetap", "kuhyhi", "b6p", "fovuqo", "Socej", "fijejI", "muzud", "rupu", "bivad",
          "rexuZ", "Jaqikeso", "totyji"]
CHAT = ["gl hf", "rotate {loc}", "they are {loc}", "eco this round", "nice", "save", "planted?", "one {loc}", "go {loc}",
        "defuse it", "wp", "hfhf", "lurk {loc}", "push {loc} now", "bomb {loc}", "I planted last time lol"]
KIND_W = {"map": 21, "plant": 6, "defuse": 2, "site": 2, "rule": 4}
TRAIN_KIND_W = {"map": 10, "plant": 4, "defuse": 2, "site": 2, "rule": 8}


def _pos_line(rng, p, side, loc_choice):
    hp = rng.choice([100, 100, 100, rng.randint(1, 99)])
    money = rng.choice([0, 150, 650, 1750, 2350, 3100, 4450, 5550, 16000, rng.randint(0, 16000)])
    base = f" Player {p} of team {side} now has {hp} points of health left. The player's remaining money is {money}."
    if loc_choice is None:
        return base
    loc, moved = loc_choice
    return base + (f" And now we discover that the player walked into {loc}." if moved else
                   f" And the player is still at {loc}.")


def _match(rng: random.Random, mapname: str, n_rounds: int, other_maps: list[str], players: tuple[list, list],
           plant_bias: dict | None = None, defuse_bias: dict | None = None, snapshots: tuple[int, int] = (3, 8),
           index_pos: bool = True, vocab: dict | None = None):
    """Returns (lines, items, events, round_info). items: (id, text) for fact lines."""
    t_team, ct_team = list(players[0]), list(players[1])
    vocab = vocab or {x: 1.0 for x in MAPS[mapname] + GENERIC}
    locs, lw = list(vocab), list(vocab.values())
    lines: list[str] = []
    items: list[tuple[str, str]] = []
    events: dict = {"plants": [], "defuses": [], "kills": Counter(), "locs": Counter(), "rounds": []}
    score = {"T": 0, "CT": 0}
    plant_bias = plant_bias or {}
    defuse_bias = defuse_bias or {}
    k = 0

    def fact(line: str, tag: str, data) -> None:
        nonlocal k
        iid = f"G{mapname[:3]}{len(lines)}"
        lines.append(line)
        items.append((iid, line.strip()))
        events.setdefault("_facts", {})[iid] = (tag, data)
        k += 1

    for r in range(1, n_rounds + 1):
        if r == n_rounds // 2 + 1:
            t_team, ct_team = ct_team, t_team
        def snapshot(t_team=t_team, ct_team=ct_team) -> None:
            lines.append("Let's take a look at the conditions and the position of some players.")
            for side, team in (("T", t_team), ("CT", ct_team)):
                for p in team:
                    if rng.random() < 0.55:
                        loc = rng.choices(locs, weights=lw)[0]
                        events["locs"][loc] += 1
                        line = _pos_line(rng, p, side, (loc, rng.random() < 0.6))
                        if index_pos:
                            fact(line, "pos", loc)
                        else:
                            lines.append(line)
                    else:
                        lines.append(_pos_line(rng, p, side, None))
            lines.append("")

        n_snap = rng.randint(snapshots[0], snapshots[1])
        snapshot()
        alive = {"T": set(t_team), "CT": set(ct_team)}
        planted, site, planter = False, None, None
        reason = None
        steps = rng.randint(4, 12)
        plant_step = rng.randint(1, steps - 1) if rng.random() < 0.55 else None
        p_snap = (n_snap - 1) / steps
        for s in range(steps):
            while rng.random() < p_snap:
                snapshot()
                p_snap *= 0.5
            p_snap = (n_snap - 1) / steps
            if rng.random() < 0.06:
                p = rng.choice(t_team + ct_team)
                msg = rng.choice(CHAT).format(loc=rng.choice(MAPS[rng.choice(other_maps)] if other_maps and rng.random() < 0.6
                                                           else COMMON_POOL))
                side = "T" if p in t_team else "CT"
                fact(f"Player [{side}]{p} sent an message. The content of the message is '{msg}'", "chat", msg)
                lines.append("")
            if s == plant_step and alive["T"]:
                w = [plant_bias.get(p, 1.0) for p in sorted(alive["T"])]
                planter = rng.choices(sorted(alive["T"]), weights=w)[0]
                site = rng.choice(["A", "B"])
                planted = True
                events["plants"].append((planter, site, r))
                fact(f"[T]{planter} has planted the bomb at {site}. The clock is ticking!", "plant", (planter, site))
                lines.append("")
                continue
            att_side = rng.choice(["T", "CT"])
            vic_side = "CT" if att_side == "T" else "T"
            if not alive[att_side] or not alive[vic_side]:
                break
            a = rng.choice(sorted(alive[att_side]))
            v = rng.choice(sorted(alive[vic_side]))
            wpn = rng.choice(WEAPONS)
            dmg = rng.randint(20, 140)
            lines.append(f"Oops! Player [{att_side}]{a} dealt {dmg} points of damage to player [{vic_side}]{v} with a {wpn}. "
                         f"Player [{vic_side}]{v} now has 0 points of health left.")
            lines.append("")
            fact(f"Oops! Player [{att_side}]{a} took down player [{vic_side}]{v} with a {wpn}. It seems the game is so easy "
                 f"for player [{att_side}]{a}. It {'was a headshot!' if rng.random() < 0.4 else 'wasn' + chr(39) + 't a headshot.'} "
                 f"It wasn't a wallbang.", "kill", (a, v))
            lines.append("")
            events["kills"][a] += 1
            alive[vic_side].discard(v)
            if not alive[vic_side]:
                if vic_side == "T" and planted:
                    break
                reason = "elimination"
                winner = att_side
                break
        if reason is None:
            if planted:
                p_def = 0.75 if not alive["T"] else 0.45
                if rng.random() < p_def and alive["CT"]:
                    if rng.random() < 0.5:
                        snapshot()
                    w = [defuse_bias.get(p, 1.0) for p in sorted(alive["CT"])]
                    d = rng.choices(sorted(alive["CT"]), weights=w)[0]
                    events["defuses"].append((d, site, r))
                    fact(f"Fantastic! [CT]{d} successfully defused the bomb at {site}. Crisis averted!", "defuse", (d, site))
                    lines.append("")
                    reason, winner = "defused", "CT"
                else:
                    for v in sorted(alive["CT"])[:rng.randint(0, 2)]:
                        lines.append(f"Oops! Player ?  dealt {rng.randint(3, 99)} points of damage to player [CT]{v} with a C4. "
                                     f"Player [CT]{v} now has {rng.randint(0, 90)} points of health left.")
                        lines.append("")
                    reason, winner = "bomb exploded", "T"
            else:
                if rng.random() < 0.5:
                    snapshot()
                reason, winner = "time ran out", "CT"
        score[winner] += 1
        events["rounds"].append((winner, reason))
        fact(f"The round has ended. Side {winner} won! The score is now T: {score['T']} - CT: {score['CT']}.", "end",
             (r, winner, reason))
        lines.append("")
    return lines, items, events


def _players(rng: random.Random) -> tuple[list, list]:
    ps = rng.sample(_NAMES, 10)
    return ps[:5], ps[5:]


def generate_game(info: dict, rounds: tuple[int, int] = (24, 40), kinds: dict | None = None) -> Generated:
    rng = random.Random(f"game:{info.get('synth_split')}:{info.get('synth_seed')}")
    kinds = kinds or dict(TRAIN_KIND_W)
    kind = info.get("_kind") or rng.choices(list(kinds), weights=list(kinds.values()))[0]
    maps = list(MAPS)
    items: list[tuple[str, str]] = []
    facts: dict = {}
    if kind == "map":
        ref = rng.choice(maps)
        others = rng.sample([m for m in maps if m != ref], 3)
        opts = others + [ref]
        rng.shuffle(opts)
        sections = [("reference map", ref)] + [(f"map option {i + 1}", m) for i, m in enumerate(opts)]
        vocabs = {ref: map_vocab(rng, ref)}
        ref_common = [c for c in vocabs[ref] if c in COMMON_POOL]
        for m in others:
            vocabs[m] = map_vocab(rng, m, like=ref_common)
        chunks = []
        for si, (label, m) in enumerate(sections):
            lines, its, ev = _match(rng, m, rng.randint(*rounds) // 2, [x for x in maps if x != m], _players(rng),
                                    snapshots=(3, 6), index_pos=True, vocab=vocabs[m])
            its = [(f"S{si}.{iid}", t) for iid, t in its]
            for old, (tag, data) in ev["_facts"].items():
                facts[f"S{si}.{old}"] = [tag, data if not isinstance(data, tuple) else list(data), si]
            items += its
            chunks.append(f"### Below is a game played on the {label}.\n<start of the game>\n" + "\n".join(lines))
        context = "\n".join(chunks)
        question = "Please determine which map is the same as the reference map."
        options = [f"{'ABCD'[i]}. Map option {i + 1}" for i in range(4)]
        gold = "ABCD"[opts.index(ref)]
        qinfo = {"ref": ref, "options": opts, "vocab": {m: sorted(v) for m, v in vocabs.items()}}
    else:
        players = _players(rng)
        allp = players[0] + players[1]
        plant_bias, defuse_bias = {}, {}
        if kind == "plant":
            star = rng.choice(allp)
            plant_bias = {p: rng.uniform(0.5, 1.5) for p in allp}
            plant_bias[star] = rng.uniform(2.5, 5.0)
        if kind == "defuse":
            star = rng.choice(allp)
            defuse_bias = {p: rng.uniform(0.5, 1.5) for p in allp}
            defuse_bias[star] = rng.uniform(3.0, 6.0)
        for _ in range(30):
            m = rng.choice(maps)
            lines, its, ev = _match(rng, m, rng.randint(*rounds), [x for x in maps if x != m], players,
                                    plant_bias, defuse_bias, snapshots=(8, 14), index_pos=False)
            ok = True
            if kind in ("plant", "defuse"):
                cnt = Counter(p for p, _, _ in ev["plants" if kind == "plant" else "defuses"])
                top = cnt.most_common(2)
                ok = bool(top) and (len(top) == 1 or top[0][1] > top[1][1]) and top[0][1] >= 2
            elif kind == "site":
                cnt = Counter(s for _, s, _ in ev["plants"])
                ok = abs(cnt.get("A", 0) - cnt.get("B", 0)) >= 2
            elif kind == "rule":
                ok = len(set(ev["rounds"])) >= 3
            if ok:
                break
        for old, (tag, data) in ev["_facts"].items():
            facts[old] = [tag, data if not isinstance(data, tuple) else list(data), 0]
        items = its
        context = "\n".join(lines)
        if kind in ("plant", "defuse"):
            evs = ev["plants" if kind == "plant" else "defuses"]
            cnt = Counter(p for p, _, _ in evs)
            top = cnt.most_common(1)[0][0]
            frag = ev["kills"].most_common(3)
            pool = [p for p in allp if p != top]
            distract = [p for p, _ in frag if p != top][:1] + [p for p, _ in cnt.most_common(4)[1:3]]
            cands = []
            for p in distract + rng.sample(pool, len(pool)):
                if p not in cands and p != top:
                    cands.append(p)
            optp = cands[:3] + [top]
            rng.shuffle(optp)
            options = [f"{'ABCD'[i]}. {p}" for i, p in enumerate(optp)]
            gold = "ABCD"[optp.index(top)]
            verb = "planted the bomb" if kind == "plant" else "defused the bomb"
            role = "planting the bomb" if kind == "plant" else "defusing the bomb"
            question = (f"Usually, the player who {verb} for the most times is the one who is responsible for {role}. Based "
                        f"on the description, please determine which of the following players were responsible for {role} "
                        f"during the match?")
            qinfo = {"counts": dict(cnt)}
        elif kind == "site":
            cnt = Counter(s for _, s, _ in ev["plants"])
            best = "A" if cnt.get("A", 0) > cnt.get("B", 0) else "B"
            options = ["A. Bombsite A", "B. Bombsite B"]
            gold = best
            question = "Which bombsite was the most frequently targeted by the players?"
            qinfo = {"counts": dict(cnt)}
        else:
            rounds_ = ev["rounds"]
            n = len(rounds_)
            for _ in range(200):
                rN = rng.randrange(n)
                same = [i for i in range(n) if i != rN and rounds_[i] == rounds_[rN]]
                diff = [i for i in range(n) if rounds_[i] != rounds_[rN]]
                if same and len(diff) >= 3:
                    break
            ans = rng.choice(same)
            trap = [i for i in diff if rounds_[i][0] == rounds_[rN][0]]
            dl = (rng.sample(trap, min(2, len(trap))) if trap else [])
            dl += [i for i in rng.sample(diff, len(diff)) if i not in dl][: 3 - len(dl)]
            opts = dl + [ans]
            rng.shuffle(opts)
            options = [f"{'ABCD'[i]}. Round {o + 1}" for i, o in enumerate(opts)]
            gold = "ABCD"[opts.index(ans)]
            question = f"Which round has the same victory condition as round {rN + 1}."
            qinfo = {"round": rN + 1, "conditions": {str(o + 1): list(rounds_[o]) for o in opts + [rN]}}
    root_prompt = (INSTRUCTION.replace("{context}", CONTEXT_POINTER).replace("{question}", question)
                   .replace("{options}", "\n".join(options)))
    hidden = {"family": "game", "kind": kind, "answer_type": "MC", "options": options, "question": question,
              "facts": facts, "q": qinfo}
    req = [iid for iid, f in facts.items() if f[0] in ({"map": "pos", "plant": "plant", "defuse": "defuse", "site": "plant",
                                                         "rule": "end"}[kind])]
    return Generated(context=context, root_prompt=root_prompt, items=items, gold=gold, hidden=hidden, required_ids=req,
                     index_min_len=20)


_AGG = re.compile(r"\b(how many|count|most|which (?:player|map|round|bombsite|option)|same as|answer|responsible|"
                  r"frequent|compare|similar)\b", re.I)


class GameOracle:
    """Lists the events / locations in the lines it is given; will not count or decide over many lines."""

    def respond(self, base: FakeSubLM, prompt: str, hidden: dict, st: dict) -> str:
        ids = base.given(prompt, hidden)
        instr = base.instruction(prompt, hidden)
        if not ids:
            return refuse_aggregate(st)
        read, _ = base.read_items(prompt, ids, st)
        facts = hidden["facts"]
        if _AGG.search(instr) and len(read) > base.read.agg_capacity:
            st["agg_requests"] += 1
            r = rng_for(prompt, "agg")
            opts = hidden.get("options") or ["A", "B", "C", "D"]
            return base.chat(prompt, f"From the log excerpt, the most likely answer is {r.choice(opts)}.")
        want = set()
        low = instr.lower()
        for tag, words in (("plant", ("plant",)), ("defuse", ("defus",)), ("kill", ("kill", "took down", "frag")),
                           ("pos", ("location", "position", "where", "walked", "place", "callout", "area")),
                           ("end", ("round", "won", "winner", "end")), ("chat", ("chat", "message"))):
            if any(w in low for w in words):
                want.add(tag)
        if not want:
            want = {"plant", "defuse", "end", "pos"}
        pairs = []
        for iid in read:
            tag, data, _ = facts[iid]
            if tag not in want:
                continue
            if tag == "pos":
                val = data
            elif tag in ("plant", "defuse"):
                val = f"{data[0]} {'planted' if tag == 'plant' else 'defused'} at site {data[1]}"
            elif tag == "kill":
                val = f"{data[0]} killed {data[1]}"
            elif tag == "end":
                val = f"round {data[0]} won by {data[1]}"
            else:
                val = f"chat: {data}"
            if base.flip(prompt, iid, tag):
                continue
            pairs.append((iid, val))
        if not pairs:
            return base.chat(prompt, "None of the requested events appear in the provided text.")
        if wants_json(instr):
            return base.render(prompt, instr, pairs)
        return base.chat(prompt, "\n".join(f"- {v}" for _, v in pairs))


__all__ = ["GameOracle", "generate_game"]
