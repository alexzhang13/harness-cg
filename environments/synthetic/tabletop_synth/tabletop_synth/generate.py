"""Window generator: cast -> episodes -> surface rendering -> one question with gold (OOLONG-real wording/semantics)."""

from __future__ import annotations

import datetime
import json
import random
import re

from . import vocab as V
from ._borrow import G, W
from .episode import DM, Actor, Episode

ANSWER_FORMAT = G.ANSWER_FORMAT


def root_prompt_for(question: str, n_eps: int, surfaces: str = "v1") -> str:
    what = "one transcript" if n_eps == 1 else f"{n_eps} transcripts"
    if surfaces == "v2":
        what = "one recorded session" if n_eps == 1 else f"{n_eps} recorded sessions"
        return ("`context` is the full input of a question about a tabletop role-playing game session log "
                f"({what}, plus whatever notes accompany it; the layout varies). Answer this question about the "
                f"game statistics:\n\n{question}\n\n{ANSWER_FORMAT}")
    return ("`context` is the full input of a question about a tabletop role-playing game session log: instructions, "
            f"who plays which character, then {what}. Answer this question about the game statistics:\n\n"
            f"{question}\n\n{ANSWER_FORMAT}")


def _char_name(rng: random.Random, used: set) -> str:
    while True:
        n = rng.choice(W._SYL_A) + rng.choice(W._SYL_B) + rng.choice(W._SYL_C)
        if n not in used and len(n) >= 4:
            used.add(n)
            return n


def _actor(rng: random.Random, player: str, cls: str, used: set, pets: set) -> Actor:
    pool = list(dict.fromkeys(W.CLASSES[cls]["spells"] + V.CLASS_EXTRA.get(cls, [])))
    spells = rng.sample(pool, min(len(pool), rng.randint(6, 10))) if pool else []
    feats = list(V.FEATURES.get(cls, {}))
    mods = {k: rng.randint(-1, 9) for k in W.SKILLS + W.ABILITY_CHECKS + W.SAVES + W.TOOLS}
    mods["Initiative"] = rng.randint(-1, 6)
    a = Actor(player, _char_name(rng, used), cls, spells, rng.sample(feats, min(len(feats), 1)),
              rng.choice(W.CLASSES[cls]["weapons"]), mods, rng.randint(4, 11))
    if cls in V.PET_KINDS and rng.random() < 0.6:
        a.pet = rng.choice([p for p in V.PET_NAMES if p not in pets])
        a.pet_kind = rng.choice(V.PET_KINDS[cls])
        pets.add(a.pet)
        if cls != "ranger" and "Find Familiar" not in a.spells:
            a.spells.append("Find Familiar")
    return a


def make_cast(rng: random.Random, n_eps: int) -> dict:
    n_pc = rng.choice([4, 5, 5, 6])
    names = rng.sample(V.PLAYERS, n_pc + 1)
    if rng.random() < 0.35:
        pair = rng.choice(V.NEAR_PAIRS)
        names[0], names[1] = pair
    classes = rng.sample(W.CASTER_CLASSES, rng.randint(2, min(4, n_pc))) + rng.sample(W.MARTIAL_CLASSES, 2)
    classes = (classes + [rng.choice(W.CASTER_CLASSES) for _ in range(n_pc)])[:n_pc + 1]
    rng.shuffle(classes)
    used, pets = set(), set()
    regulars = [_actor(rng, names[i], classes[i], used, pets) for i in range(n_pc)]
    guest = _actor(rng, names[n_pc], classes[n_pc], used, pets) if rng.random() < 0.35 else None
    guest_eps = set(range(n_eps)) if guest and n_eps == 1 else (
        set(rng.sample(range(n_eps), rng.randint(1, n_eps - 1))) if guest else set())
    stale = None
    if rng.random() < 0.3:
        i = rng.randrange(n_pc)
        old_cls = rng.choice([c for c in W.CASTER_CLASSES + W.MARTIAL_CLASSES if c != regulars[i].cls])
        stale = {"idx": i, "actor": _actor(rng, regulars[i].player, old_cls, used, pets),
                 "until": rng.randint(1, n_eps)}
    return {"dm": rng.choice(V.DM_NAMES), "regulars": regulars, "guest": guest, "guest_eps": guest_eps,
            "stale": stale}


def roster(cast: dict, k: int) -> list[Actor]:
    out = list(cast["regulars"])
    st = cast["stale"]
    if st and k < st["until"]:
        out[st["idx"]] = st["actor"]
    if cast["guest"] and k in cast["guest_eps"]:
        out.append(cast["guest"])
    return out


def pick_surface(rng: random.Random) -> dict:
    def ch(d):
        return rng.choices(list(d), list(d.values()))[0]
    return {"delim": ch(V.DELIM_W), "speaker": ch(V.SPEAKER_W), "header": ch(V.HEADER_W), "dm_label": ch(V.DM_LABEL_W),
            "instr": rng.randrange(len(V.INSTR)), "map_intro": rng.randrange(len(V.MAP_INTRO)),
            "map_outro": rng.randrange(len(V.MAP_OUTRO))}


def speaker_line(style: str, label: str, text: str) -> str:
    return {"colon": f"{label}: {text}", "upper": f"{label.upper()}: {text}", "bracket": f"[{label}] {text}",
            "dash": f"{label} - {text}"}[style]


def header_lines(sf: dict, dm: str, mapping: list[tuple[str, str]], n_eps: int) -> list[str]:
    what = "a single episode transcript" if n_eps == 1 else f"{n_eps} episode transcripts"
    each = "The transcript is" if n_eps == 1 else "Each transcript is"
    if sf["delim"] in ("session", "hash"):
        each = "The transcript" if n_eps == 1 else "Each transcript"
    delim = V.DELIM_DESC[sf["delim"]][0].format(each=each)
    out = [V.INSTR[sf["instr"]].format(what=what, delim=delim).replace("{}", "{}"), "", V.MAP_INTRO[sf["map_intro"]]]
    role = "DM" if sf["dm_label"] != "GM" else "GM"
    rows = [(dm, role)] + mapping
    h = sf["header"]
    if h == "table":
        out += ["| Player | Character |", "|---|---|"] + [f"| {p} | {c} |" for p, c in rows]
    else:
        fmt = {"plays": "{p} plays the character {c}.", "paren": "{c} (played by {p})", "arrow": "{p} -> {c}"}[h]
        out += [fmt.format(p=p, c=c) for p, c in rows]
    if V.MAP_OUTRO[sf["map_outro"]]:
        out += ["", V.MAP_OUTRO[sf["map_outro"]]]
    return out + [""]


def pick_surface_v2(rng: random.Random) -> dict:
    def ch(d):
        return rng.choices(list(d), list(d.values()))[0]
    return {"delim": ch(V.DELIM_W2), "speaker": ch(V.SPEAKER_W2), "header": ch(V.HEADER_W2),
            "dm_label": ch(V.DM_LABEL_W), "instr": rng.randrange(len(V.INSTR2)),
            "map_intro": rng.randrange(len(V.MAP_INTRO2)), "map_outro": rng.randrange(len(V.MAP_OUTRO)),
            "title_fmt": rng.randrange(2), "date_fmt": rng.choice(V.DATE_FMT),
            "date0": rng.randrange(19000, 20000)}


def header_lines_v2(sf: dict, dm: str, mapping: list[tuple[str, str]], n_eps: int, rng: random.Random) -> list[str]:
    what = "a single episode transcript" if n_eps == 1 else f"{n_eps} episode transcripts"
    if sf["delim"] in V.VERBLESS_DESC:
        each = "The transcript" if n_eps == 1 else "Each transcript"
    else:
        each = "The transcript is" if n_eps == 1 else "Each transcript is"
    delim = V.DELIM_DESC2[sf["delim"]][0].format(each=each)
    out = [V.INSTR2[sf["instr"]].format(what=what, delim=delim), "", V.MAP_INTRO2[sf["map_intro"]]]
    role = "DM" if sf["dm_label"] != "GM" else "GM"
    rows = [(dm, role)] + mapping
    h = sf["header"]
    if h == "table":
        out += ["| Player | Character |", "|---|---|"] + [f"| {p} | {c} |" for p, c in rows]
    elif h == "json":
        out += json.dumps(dict(rows), indent=2).split("\n")
    elif h == "bullet":
        out += [f"- {p} \u2014 {c}" for p, c in rows]
    elif h == "prose":
        cl = [rng.choice(V.PROSE_CLAUSE).format(p=p, c=c) for p, c in mapping]
        out.append(f"Tonight {', '.join(cl[:-1])}, and {cl[-1]}. {dm} is running the game as the {role}.")
    else:
        fmt = {"plays": "{p} plays the character {c}.", "paren": "{c} (played by {p})", "arrow": "{p} -> {c}"}[h]
        out += [fmt.format(p=p, c=c) for p, c in rows]
    if V.MAP_OUTRO[sf["map_outro"]]:
        out += ["", V.MAP_OUTRO[sf["map_outro"]]]
    return out + [""]


def delim_lines_v2(sf: dict, k: int, rng: random.Random) -> tuple[str, str | None]:
    op, cl = V.DELIMS2[sf["delim"]]
    a, b = rng.choice(V.TITLE_A), rng.choice(V.TITLE_B)
    date = datetime.date.fromordinal(datetime.date(1970, 1, 1).toordinal() + sf["date0"] + 7 * k)
    kw = dict(k=k + 1, K=V.NUM_CAPS[k], title=f"The {a} {b}" if sf["title_fmt"] == 0 else f"{a} and {b}",
              date=date.strftime(sf["date_fmt"]))
    return op.format(**kw), (cl.format(**kw) if cl else None)


def _wrap(text: str, cues: list[str], rng: random.Random) -> list[str]:
    """Split a multi-sentence utterance into 2-3 lines at sentence ends, unless that would cut a cue in two."""
    parts = re.split(r"(?<=[.!?]) (?=\S)", text)
    if len(parts) < 2 or rng.random() >= V.WRAP_P:
        return [text]
    cuts = sorted(rng.sample(range(1, len(parts)), min(len(parts) - 1, rng.randint(1, 2))))
    pieces = [" ".join(parts[i:j]) for i, j in zip([0] + cuts, cuts + [len(parts)], strict=True)]
    return pieces if all(any(c in pc.lower() for pc in pieces) for c in cues) else [text]


def render_episode_v2(sf: dict, ep: Episode, ros: list[Actor], dm: str, dm_label: str, role: str,
                      rng: random.Random):
    """-> (rendered lines, start offset of each utterance, {(utterance, cue): offset of the line holding the cue})."""
    p2c_ep = {a.player: a.char for a in ros}
    cues: dict[int, list[str]] = {}
    for x in ep.rolls + ep.spells:
        cues.setdefault(x.get("cue_li", x["li"]), []).append(str(x["cue"]).lower())
    st, t = sf["speaker"], rng.randrange(0, 900)
    body, starts, cue_off = [], [], {}
    for ui, (spk, text) in enumerate(ep.lines):
        label = dm_label if spk == DM else spk
        if st == "charname":
            label = f"{dm} ({role})" if spk == DM else f"{spk} ({p2c_ep[spk]})"
        pieces = _wrap(text, cues.get(ui, []), rng) if st == "wrap" else [text]
        t += rng.randint(2, 30)
        first = {"stamp": f"[{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}] {label}: {pieces[0]}",
                 "bold": f"**{label}:** {pieces[0]}", "quote": f"> {label}: {pieces[0]}",
                 "charname": f"{label}: {pieces[0]}", "wrap": f"{label}: {pieces[0]}"}.get(st)
        starts.append(len(body))
        for c in cues.get(ui, []):
            cue_off[(ui, c)] = len(body) + next(i for i, pc in enumerate(pieces) if c in pc.lower())
        body += [first if first is not None else speaker_line(st, label, pieces[0])] + pieces[1:]
    return body, starts, cue_off


def make_q(rng: random.Random, tid: str, rolls: list[dict], spells: list[dict], n_eps: int, p2c: dict):
    """A question from the OOLONG-real templates; natural-roll questions only ask about values the text states."""
    if tid in ("r_nat", "r_pct"):
        single = n_eps == 1
        nats = [x["nat"] for x in rolls if x["nat"] is not None]
        v = rng.choice(sorted(set(nats) | {1, 20}))
        n = sum(1 for x in nats if x == v)
        scope = "in this episode" if single else "across all episodes"
        if tid == "r_nat":
            return G.Q(tid, f"Number of rolls of natural value {v} {scope}?", str(n), "rolls", "count", {})
        pct = round(n / max(1, len(rolls)) * 100)
        text = (f"In this episode, what percentage of rolls were of value {v}? round to the nearest integer." if single
                else f"Across all episodes, what percentage of rolls were of value {v}? round to the nearest integer.")
        return G.Q(tid, text, str(pct), "rolls", "count", {})
    return G.make_question(rng, tid, rolls, spells, n_eps, p2c)


def pick_question(seed, rolls, spells, n_eps, p2c, tid=None):
    rng = random.Random(f"tabletop:{seed}:question")
    weights = G.SINGLE_W if n_eps == 1 else G.MULTI_W
    for _ in range(60):
        t = tid or rng.choices(list(weights), list(weights.values()))[0]
        q = make_q(rng, t, rolls, spells, n_eps, p2c)
        if q is not None:
            return q
    raise RuntimeError(f"no valid question for seed {seed}")


def generate(seed, *, n_episodes: int | None = None, ep_lines: tuple[int, int] = (700, 1400),
             tid: str | None = None, numwords: float = 0.25, surfaces: str = "v1") -> dict:
    """``surfaces="v2"`` renders the same window as v1 with more varied formatting (separate RNG)."""
    if surfaces not in ("v1", "v2"):
        raise ValueError(f"surfaces must be 'v1' or 'v2', got {surfaces!r}")
    rng = random.Random(f"tabletop:{seed}")
    n_eps = n_episodes or rng.choices([1, 2, 3], [0.3, 0.35, 0.35])[0]
    cast = make_cast(rng, n_eps)
    sf = pick_surface(rng)
    rng2 = random.Random(f"tabletop:{seed}:surface-v2")
    if surfaces == "v2":
        sf = pick_surface_v2(rng2)
    dm = cast["dm"]
    dm_label = dm if sf["dm_label"] == "name" else sf["dm_label"]
    header_cast = list(cast["regulars"]) + ([cast["guest"]] if cast["guest"] else [])
    p2c = {dm: "DM", **{a.player: a.char for a in header_cast}}

    eps, prev_r, prev_s = [], [], []
    for k in range(n_eps):
        ep = Episode(rng, roster(cast, k), k, numwords=numwords)
        ep.build(rng.randint(*ep_lines), prev_r, prev_s)
        prev_r, prev_s = ep.rolls, ep.spells
        eps.append(ep)

    v2 = surfaces == "v2"
    mapping = [(a.player, a.char) for a in header_cast]
    lines = header_lines_v2(sf, dm, mapping, n_eps, rng2) if v2 else header_lines(sf, dm, mapping, n_eps)
    rolls, spells, ep_meta = [], [], []
    op, cl = V.DELIMS[sf["delim"]] if not v2 else (None, None)
    for k, ep in enumerate(eps):
        ros = roster(cast, k)
        if v2:
            op_k, cl_k = delim_lines_v2(sf, k, rng2)
            role = "GM" if sf["dm_label"] == "GM" else "DM"
            body, starts, cue_off = render_episode_v2(sf, ep, ros, dm, dm_label, role, rng2)
        else:
            op_k, cl_k = op.format(k=k + 1), cl.format(k=k + 1) if cl else None
            body = [speaker_line(sf["speaker"], dm_label if spk == DM else spk, text) for spk, text in ep.lines]
            starts, cue_off = range(len(body)), {}
        if k:
            lines.append("")
        lines.append(op_k)
        base = len(lines)
        lines += body
        if cl_k:
            lines.append(cl_k)
        for rows, out in ((ep.rolls, rolls), (ep.spells, spells)):
            for x in rows:
                y = {kk: v for kk, v in x.items() if kk not in ("li", "cue_li", "ref_li")}
                cli = x.get("cue_li", x["li"])
                y.update(ep=k, line=base + starts[x["li"]],
                         cue_line=base + cue_off.get((cli, str(x["cue"]).lower()), starts[cli]))
                if "ref_li" in x:
                    y["ref_line"] = base + starts[x["ref_li"]]
                out.append(y)
        em = {"roster": {a.player: a.char for a in ros}, "first_line": base, "n_lines": len(body),
              "stale": [a.player for a in ros if p2c.get(a.player) not in (None, a.char)]}
        if v2:
            em["utt_start"] = [base + i for i in starts]
        ep_meta.append(em)
    context = "\n".join(lines)
    q = pick_question(seed, rolls, spells, n_eps, p2c, tid)
    stale = cast["stale"]
    meta = {
        "n_episodes": n_eps, "surface": sf, "dm": dm, "dm_label": dm_label, "p2c": p2c, "episodes": ep_meta,
        "cast": [{"player": a.player, "char": a.char, "cls": a.cls, "spells": a.spells, "features": a.features,
                  "pet": a.pet} for a in header_cast + ([stale["actor"]] if stale else [])],
        "stale_char": stale["actor"].char if stale else None, "rolls": rolls, "spells": spells, "family": q.family,
        "kind": q.kind,
    }
    return {"context": context, "question": q.question, "answer": q.gold, "qtype": q.tid, "meta": meta,
            "root_prompt": root_prompt_for(q.question, n_eps, surfaces), "seed": seed}
