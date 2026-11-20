"""One synthetic ELB env mixing docket rows (env.py, T01-like) and short form rows (forms.py, T06 / T04 / T03-like).

With ``hint="note"`` each row gets its kind's final note; ``kind`` restricts to one row kind.

    load_environment(split="train", num_examples=4000, p_docket=0.275, form={"p_recipe": 0.345, "p_tutor": 0.54})
"""

from __future__ import annotations

import json
import random
from typing import Any

from synth_common.envkit import Generated, make_env
from synth_common.oracle import FakeSubLM, ReadModel, rng_for
from synth_common.program import HintSpec, add_hint

from . import env as docket_env
from . import forms, steering, t01note
from .generate import generate as docket_generate
from .steering import HINTS as DOCKET_HINTS
from .steering import DocketHintSpec

# docket note for hint="note" (c6); "note_v5" / "note_v4" select earlier notes
DOCKET_NOTES = {"t01_note_v4": DOCKET_HINTS.text("note_v4"), "t01_note_v5": DOCKET_HINTS.text("note_v5"),
                "t01_note_c6": t01note.note_text_c6()}
DOCKET_NOTE_VERSION = "t01_note_c6"
DOCKET_NOTE: str = DOCKET_NOTES[DOCKET_NOTE_VERSION]


def load_docket_note(path: str) -> str:
    """Read a docket note from a text/markdown file."""
    with open(path) as fh:
        return fh.read().strip()


def generate(info: dict) -> Generated:
    P = json.loads(info.get("synth_params") or "{}") if isinstance(info.get("synth_params"), str) else \
        dict(info.get("synth_params") or {})
    rng = random.Random(f"elb-mix:{info['synth_split']}:{info['synth_seed']}")
    hint, hint_prob = P.get("hint", "none"), float(P.get("hint_prob", 1.0))
    if rng.random() < float(P.get("p_docket", 0.4)):
        g = docket_generate({**info, "synth_params": json.dumps({**P.get("docket", {}), "variant": "docket"})})
        if hint not in ("none", "", None) and rng_for("mix-hint", info["synth_split"], info["synth_seed"]).random() < hint_prob:
            g.root_prompt = add_hint(g.root_prompt, DOCKET_HINT.text(hint))
            g.hidden["hinted"] = 1
        return g
    fp = dict(P.get("forms", {}))
    if hint not in ("none", "", None):
        fp.update(hint="strategy" if hint in ("note", "note_v4", "note_v5") else hint, hint_prob=hint_prob)
    return forms.generate({**info, "synth_params": json.dumps(fp)})


class MixOracle(FakeSubLM):
    def __init__(self, docket: FakeSubLM, form: FakeSubLM):
        super().__init__(docket.read)
        self.docket, self.form = docket, form

    def respond(self, prompt: str, hidden: dict, st: dict) -> str:
        if hidden.get("variant") in forms.GENERATORS:
            return self.form.respond(prompt, hidden, st)
        return self.docket.respond(prompt, hidden, st)


def score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    if hidden.get("variant") in forms.GENERATORS:
        return forms.score(final, gold, hidden, state)
    return docket_env.score(final, gold, hidden, state)


HINTS = HintSpec(
    strategy=DOCKET_HINTS.strategy + "\n\nShort records (a dialogue or a form): " + forms.HINTS.strategy,
    program=DOCKET_HINTS.program + "\n\n# --- short records ---\n" + forms.HINTS.program,
)
DOCKET_HINT = DocketHintSpec(strategy=DOCKET_HINTS.strategy + " " + forms.OWN_WORDS, program=DOCKET_HINTS.program,
                             note=DOCKET_NOTE, note_v4=DOCKET_NOTES["t01_note_v4"],
                             note_v5=DOCKET_NOTES["t01_note_v5"])


def _by_variant(name: str):
    """Apply the docket or the forms version of a check, by row kind."""
    d = next(c for c in steering.CHECKS if c.name == name)
    f = next(c for c in forms.CHECKS if c.name == name)

    def fn(s: dict):
        hid = s.get("synth_hidden") or {}
        return (f if hid.get("variant") in forms.GENERATORS else d).fn(s)
    return forms.Check(name, fn, hard=f.hard, weight=f.weight, doc=f"{f.doc} | docket rows: {d.doc}")


CHECKS = [_by_variant("json_answer"), _by_variant("na_token")] + \
    [c for c in forms.CHECKS if c.name in ("visit_type", "no_hint_mention")] + \
    [_by_variant("follows_note")]
KINDS = ("docket", *forms.GENERATORS)


def load_environment(split: str = "train", num_examples: int = 4000, p_docket: float = 0.4,
                     docket: dict | None = None, form: dict | None = None, max_iterations: int = 40,
                     hint: str = "none", hint_prob: float = 1.0, reward_mode: str = "outcome",
                     program_weight: float = 0.2, gate_floor: float = 0.0, kind: str | None = None, **kwargs: Any):
    """``docket`` / ``form`` are generator params for the two row types. ``kind`` (one of KINDS) serves one kind only."""
    if kind:
        if kind not in KINDS:
            raise ValueError(f"kind={kind!r} not in {KINDS}")
        if kind == "docket":
            p_docket = 1.0
        else:
            p_docket, form = 0.0, {**(form or {}), "kind": kind}
    params = {"p_docket": p_docket, "docket": {"chars": "t01", "max_chars": 1_200_000, "update_rate": 0.7,
                                               "distractor_rate": 0.75, "n_fields": [17, 24],
                                               **(docket or {})},
              "forms": {"p_recipe": 0.15, **(form or {})}}
    if hint not in ("none", "", None):
        params.update(hint=hint, hint_prob=hint_prob)
    docket_oracle = docket_env.CaseFileOracle(ReadModel(budget_chars=400_000, base_err=0.01, load_err=0.04, chatty=0.2),
                                              attn_chars=160_000)
    kwargs.setdefault("hints", HINTS)
    kwargs.setdefault("checks", CHECKS)
    return make_env(generator=generate, oracle=MixOracle(docket_oracle, forms.FormReader()), score=score, split=split,
                    num_examples=num_examples, params=params, max_iterations=max_iterations,
                    extra_metrics=docket_env.EXTRA_METRICS, hint="none", reward_mode=reward_mode,
                    program_weight=program_weight, gate_floor=gate_floor, **kwargs)


__all__ = ["CHECKS", "DOCKET_NOTE", "DOCKET_NOTES", "DOCKET_NOTE_VERSION", "HINTS", "KINDS", "MixOracle", "generate", "load_docket_note",
           "load_environment", "score"]
