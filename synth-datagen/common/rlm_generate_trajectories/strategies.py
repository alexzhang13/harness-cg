"""Per-stream conditioning for SFT generation: the stream's note (``$RLM_V2_NOTES_DIR/<stream>.md``) is injected
between ``<strategy_hint>`` markers so it can be stripped later, together with the stream's acceptance thresholds.
``RLM_NOTE_PREAMBLE=1`` adds ``NOTE_PREAMBLE`` before the note.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

HARNESS_FOOTER = """\
Harness facts that decide whether your work counts:
- A ```repl``` block you write is executed at the START of your next turn, so a block written on your LAST
  allowed turn never runs and nothing is submitted. Submit at least two turns before the cap.
- You get exactly ONE submission and cannot revise it. The moment `answer["ready"]` becomes True in a block
  that then finishes, that answer is taken and the rollout ends, so never submit a placeholder intending to
  improve it. Equally, do not submit so late that the cap arrives first: decide the turn you will submit on,
  and submit the best answer you have then.
- Assign the content BEFORE flipping the flag, in that order and in the same block:
  `answer["content"] = final; answer["ready"] = True`. Flipping the flag first submits an empty string.
- Every user turn ends with `Turn k/N`, so you always know your budget. Use it as a hard deadline: keep a
  variable holding your best answer so far, updated as you learn, and when you reach turn N-3 submit that
  value in that same block whatever its quality. Running out of turns scores zero, and an imperfect
  submitted answer almost always scores more than zero, so on the deadline turn submitting beats
  investigating further.
- Only `print(...)` output comes back, and a reply over ~20K characters is truncated: print counts, lengths
  and short previews, never a document or a full list.
- A sub-call whose prompt is too large returns an error string rather than an answer; keep each prompt to a
  chunk you chose deliberately, and check the first reply parses before issuing the rest.
- If a block raises, every name it defined is lost, so the next block cannot use a function you defined in
  the one that failed. Keep each block self-contained: define and use in the same block, and guard risky
  parsing with try/except inside it rather than splitting across turns.
"""

HINT_OPEN = "<strategy_hint>"
HINT_CLOSE = "</strategy_hint>"


NOTE_PREAMBLE = """\
These instructions are private scaffolding that will be removed from the transcript: never mention them, never call
a step "prescribed", "specified" or "as instructed", and never refer to a hint, note or template -- explain each step
by what you observed and why it is the next step. Every turn contains exactly one ```repl``` block.
"""


_RL_ONLY_ARGS = {"gate_subcall", "min_subcall"}


@dataclass
class Strategy:
    name: str
    env_id: str
    env_args: dict[str, Any]
    hint: str
    families: tuple[str, ...]
    min_reward: float = 1.0
    min_sub_calls: int = 1
    max_sub_calls: int | None = None
    require_subcall_context: bool = True
    max_repl_errors: int = 1
    max_subcall_errors: int = 0
    max_sequential_single_sites: int = 6
    max_sub_prompt_chars: int = 60_000
    max_repl_output_chars: int = 15_000
    max_total_output_chars: int = 30_000
    context_causal: bool = False
    search_ok: bool = False
    python_exact: bool = False
    judge_note: str = ""
    partial_credit_judge: bool = False
    probe_from_prompt_ok: bool = False
    very_short_chars: int = 12_000
    max_search_patterns: int = 6
    max_block_lines: int = 60
    max_raw_echo_chars: int = 2_500
    unsized_prompt_chars: int = 8_000
    stage: int = 1
    prologue_prefix: str | None = None
    split: str = "train"
    ratio: float = 0.0
    notes: str = ""

    hint_mode: str = "note"

    @property
    def hint_text(self) -> str:
        """The hint body actually injected: [NOTE_PREAMBLE +] the stream's note + HARNESS_FOOTER."""
        if self.hint_mode != "note":
            raise ValueError(f"hint_mode {self.hint_mode!r}: only 'note' is supported")
        if os.environ.get("RLM_NOTE_PREAMBLE") == "1":
            return f"{NOTE_PREAMBLE.strip()}\n\n{self.hint.strip()}\n\n{HARNESS_FOOTER.strip()}"
        return f"{self.hint.strip()}\n\n{HARNESS_FOOTER.strip()}"

    @property
    def prologue(self) -> str:
        """The exact user-prologue text injected into the env."""
        body = f"{HINT_OPEN}\n{self.hint_text}\n{HINT_CLOSE}"
        return f"{self.prologue_prefix.rstrip()}\n\n{body}" if self.prologue_prefix else body

    def rollout_env_args(self) -> dict[str, Any]:
        args = {k: v for k, v in self.env_args.items() if k not in _RL_ONLY_ARGS}
        args["user_prologue"] = self.prologue
        return args

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["families"] = list(self.families)
        return d


# one note per stream, read once at import
_NOTES_DIR = Path(os.environ.get("RLM_V2_NOTES_DIR") or Path(__file__).resolve().parent / "v2_notes")
V2_NOTES: dict[str, str] = ({p.stem: p.read_text().strip() for p in _NOTES_DIR.glob("*.md")}
                            if _NOTES_DIR.is_dir() else {})


def load_streams(config: Path, section: str = "train") -> list[dict[str, Any]]:
    cfg = tomllib.load(open(config, "rb"))
    return list(cfg["orchestrator"][section]["env"])


def build_strategies(config: Path, section: str = "train", overrides_json: Path | None = None,
                     hint_mode: str = "note", min_sub_calls_override: int | None = None) -> dict[str, Strategy]:
    """Strategies for every stream of ``section`` in ``config``."""
    extra: dict[str, dict[str, Any]] = {}
    if overrides_json and Path(overrides_json).exists():
        extra = json.loads(Path(overrides_json).read_text())
    out: dict[str, Strategy] = {}
    for s in load_streams(config, section):
        env_id, name, args = s["id"], s["name"], dict(s.get("args", {}))
        args.pop("user_prologue", None)
        spec = {k: v for k, v in extra.get(name, {}).items() if k in ("min_reward", "families", "notes")}
        spec["hint"] = V2_NOTES.get(name, "")
        spec.update(s.get("sft", {}))
        if spec.get("search_ok") and "max_search_patterns" not in spec:
            spec["max_search_patterns"] = 25
        if spec.get("python_exact"):
            spec["min_sub_calls"] = 0
            spec.setdefault("max_search_patterns", 20)
        hint = spec.pop("hint", "")
        fam = tuple(spec.pop("families", ()))
        spec["min_sub_calls"] = 0
        if min_sub_calls_override is not None:
            spec["min_sub_calls"] = int(min_sub_calls_override)
        out[name] = Strategy(
            name=name, env_id=env_id, env_args=args, hint=hint, families=fam,
            ratio=float(s.get("ratio", 0.0)), split="eval" if section == "eval" else "train",
            hint_mode=hint_mode, **spec,
        )
    return out


def strip_hint_text(text: str) -> str:
    """Remove the marked hint block from a user message; '' if nothing remains."""
    out = re.sub(rf"\s*{re.escape(HINT_OPEN)}.*?{re.escape(HINT_CLOSE)}\s*", "", text, flags=re.S)
    return out.strip()
