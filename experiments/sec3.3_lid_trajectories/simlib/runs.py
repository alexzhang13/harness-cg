"""Panels, run layout ($PRIME_RL_DIR/outputs/<family>/<method>/<task>) and checkpoint choice for the §3.3 analysis.

The checkpoint is the eval step with the highest mean reward (earliest on ties); for OBLIQ that is nDCG@50.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = Path(os.environ.get("PRIME_RL_DIR", Path.home() / "prime-rl")) / "outputs"

# (slug, family, task, RLM method, title, train -> eval, family label). The RLM series is the orchestrator-prompt run
# (strategy-gen `rlm` configs already use it).
PANELS = [
    ("graphwalks", "length-gen", "graphwalks", "rlm_orchestrator_prompt", "GraphWalks", "<128K → >1M", "Length generalization"),
    ("longbenchpro", "length-gen", "longbenchpro", "rlm_orchestrator_prompt", "LongBench-Pro", "32K → 256K", "Length generalization"),
    ("oolong", "length-gen", "oolong", "rlm_orchestrator_prompt", "OOLONG", "32K → 256K", "Length generalization"),
    ("oolong-pairs", "length-gen", "oolong-pairs", "rlm_orchestrator_prompt", "OOLONG-Pairs", "8K → 32K", "Length generalization"),
    ("adaleval", "length-gen", "adaleval", "rlm_orchestrator_prompt", "Ada-LEval", "8K → 128K", "Length generalization"),
    ("oolong_trec_to_spam", "strategy-gen", "oolong_trec_to_spam", "rlm", "OOLONG", "trec → spam",
     "Strategy generalization"),
    ("obliq_writing_to_math", "strategy-gen", "obliq_writing_to_math", "rlm", "OBLIQ-Bench", "writing → math",
     "Strategy generalization"),
    ("obliq_twitter_to_wildchat", "strategy-gen", "obliq_twitter_to_wildchat", "rlm", "OBLIQ-Bench",
     "twitter → wildchat", "Strategy generalization"),
]
PANEL = {p[0]: p for p in PANELS}
# Harness keys used in result tables; "rlm" resolves to the panel's RLM method.
HARNESSES = ("rlm", "lm_repl", "pi", "transformer")
# Transformer tasks evaluated offline; their eval prompts come from regen_baseline_eval.py
OFFLINE_TRANSFORMER = ("graphwalks", "longbenchpro", "oolong", "oolong-pairs")


def method_for(slug: str, harness: str) -> str:
    return PANEL[slug][3] if harness == "rlm" else harness


def run_id(slug: str, harness: str) -> str:
    """<family>/<method>/<task>, the prefix of trajectory uids."""
    _, family, task, *_ = PANEL[slug]
    return f"{family}/{method_for(slug, harness)}/{task}"


def run_dir(outputs: Path, slug: str, harness: str) -> Path:
    _, family, task, *_ = PANEL[slug]
    return Path(outputs) / family / method_for(slug, harness) / task


def _steps_with(rollouts: Path, filename: str) -> list[int]:
    if not rollouts.is_dir():
        return []
    out = []
    for d in rollouts.iterdir():
        if d.name.startswith("step_") and (d / filename).exists():
            out.append(int(d.name.split("_", 1)[1]))
    return sorted(out)


def _curve_from_inline(rollouts: Path, steps: list[int]) -> list[dict]:
    rows = []
    for s in steps:
        f = rollouts / f"step_{s}" / "eval_rollouts.jsonl"
        rewards, n_err = [], 0
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") or int(rec.get("num_turns", 0) or 0) == 0:
                n_err += 1
            rewards.append(float(rec.get("reward", 0.0) or 0.0))
        if rewards:
            rows.append({"step": s, "reward": sum(rewards) / len(rewards),
                         "n": len(rewards), "n_error": n_err})
    return rows


def entry(outputs: Path, slug: str, harness: str, max_eval_step: int | None = None) -> dict | None:
    """Rollout dir, available steps and selected eval step (only steps <= max_eval_step); None if missing."""
    task = PANEL[slug][2]
    method = method_for(slug, harness)
    rollouts = run_dir(outputs, slug, harness) / "run_default" / "rollouts"
    eval_steps = _steps_with(rollouts, "eval_rollouts.jsonl")
    if max_eval_step is not None:
        eval_steps = [s for s in eval_steps if s <= max_eval_step]
    train_steps = _steps_with(rollouts, "train_rollouts.jsonl")
    if not (eval_steps or train_steps):
        return None
    curve = _curve_from_inline(rollouts, eval_steps) if eval_steps else []
    best_step, best_score = None, None
    if curve:
        best = max(curve, key=lambda r: r["reward"])
        best_step, best_score = best["step"], round(best["reward"], 4)
    return {
        "run_id": run_id(slug, harness),
        "task": task,
        "method": method,
        "rollout_root": str(rollouts),
        "eval": {
            "best_step": best_step,
            "best_score": best_score,
            "available_traj_steps": eval_steps,
            "curve": curve,
        },
        "train": {"available_steps": train_steps},
    }


# file with the offline long-split evals of each Transformer run
LENGTH_TRANSFORMER_EVAL = {
    "mrcr": "offline_long_eval_lora.jsonl", "graphwalks": "offline_long_eval_lora.jsonl",
    "longbenchpro": "offline_long_eval_lora.jsonl", "oolong": "offline_long_eval_lora_answeronly.jsonl",
    "oolong-pairs": "offline_long_eval_lora.jsonl", "adaleval": "inline",
}


def offline_eval_curve(outputs: Path, slug: str) -> dict[int, float]:
    """{step: reward} from the offline evals, skipping errored ones."""
    path = run_dir(outputs, slug, "transformer") / LENGTH_TRANSFORMER_EVAL[PANEL[slug][2]]
    out = {}
    if path.exists():
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if r.get("n_error", 0) == 0:
                    out[int(r["step"])] = r["reward"]
    return out
