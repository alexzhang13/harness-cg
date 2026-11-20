"""Rewrite a tabletop_synth root prompt in the real task's template, or a seeded mix of styles, so the prompt
wording is not a cue for the program (export_sft.py --mixed-prompt)."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "environments/synthetic/tabletop_synth"), str(ROOT / "environments/synthetic/oolong_real")]
from oolong_real_synth import gen as G  # noqa: E402
from tabletop_synth import vocab as V  # noqa: E402
from tabletop_synth.generate import generate  # noqa: E402

SPEAKER = {"colon": "`Player: text`", "upper": "`PLAYER: text`", "bracket": "`[Player] text`", "dash": "`Player - text`"}


def regenerate(seed: int, question: str, surfaces: str = "v1") -> dict | None:
    g = generate(seed, surfaces=surfaces)
    if g["question"] != question:
        tid = random.Random(seed + 1).choice(sorted(t for t in G.SINGLE_W if t.startswith("r_") and t in G.MULTI_W))
        g = generate(seed, tid=tid, surfaces=surfaces)
    return g if g["question"] == question else None


def real_style(g: dict) -> str:
    s, n = g["meta"]["surface"], g["meta"]["n_episodes"]
    if s["delim"] == "real" and s["speaker"] == "colon":
        return G.root_prompt_for(g["question"], n)
    what = "one episode transcript" if n == 1 else f"{n} episode transcripts"
    if s["delim"] not in V.DELIMS or s["speaker"] not in SPEAKER:
        return ("`context` is the full input of a long-context question about a Dungeons and Dragons game: instructions, "
                f"the player -> character mapping, then {what}. Answer this question about the game statistics:\n\n"
                f"{g['question']}\n\n{G.ANSWER_FORMAT}")
    op, cl = V.DELIMS[s["delim"]]
    op = op.replace("{k}", "1")
    bounds = f"each between {op} and {cl.replace('{k}', '1')} lines" if cl else f"each starting with a line such as {op}"
    return ("`context` is the full input of a long-context question about a Dungeons and Dragons game: instructions, the "
            f"player -> character mapping, then {what} (one utterance per line, {SPEAKER[s['speaker']]}), {bounds}. "
            f"Answer this question about the game statistics:\n\n{g['question']}\n\n{G.ANSWER_FORMAT}")


PARAPHRASES = [
    "`context` holds a long log of a tabletop role-playing campaign: a short instruction paragraph, a list of which player "
    "runs which character, then {what}{fmt}. Use it to answer this question about the game statistics:\n\n{q}\n\n{af}",
    "You are given `context`: the record of {what} of a tabletop RPG (D&D-style){fmt}, preceded by instructions and the "
    "player-to-character list. Question about the game statistics:\n\n{q}\n\n{af}",
]


def mixed_style(g: dict, seed: int) -> str:
    """Seeded choice of the synthetic prompt, the real-task template or a paraphrase."""
    r = random.Random(seed * 7 + 3).random()
    if r < 1 / 3:
        return g["root_prompt"]
    if r < 2 / 3:
        return real_style(g)
    s, n = g["meta"]["surface"], g["meta"]["n_episodes"]
    what = "one session transcript" if n == 1 else f"{n} session transcripts"
    fmt = "" if random.Random(seed).random() < 0.5 or s["speaker"] not in SPEAKER else f" (lines look like {SPEAKER[s['speaker']]})"
    return random.Random(seed + 11).choice(PARAPHRASES).format(what=what, fmt=fmt, q=g["question"], af=G.ANSWER_FORMAT)
