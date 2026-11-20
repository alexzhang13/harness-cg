"""Evaluate the saved checkpoints of a Tinker run on its config's eval envs.

Same as the inline eval. Results go to <log_dir>/evals/. Evals longer than Tinker's context window need the
exported adapters and the prime-rl offline evaluators.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import tinker

from . import config as C
from .client import Policy, TinkerClient
from .envs import load_eval_env
from .export import checkpoints
from .model import get_renderer, max_context, tokenizer_id
from .train import evaluate_env

logger = logging.getLogger(__name__)


async def evaluate_run(log_dir: Path, *, steps: list[int] | None = None, eval_envs: list[str] | None = None,
                       eval_num_examples: int | None = None) -> None:
    meta = json.loads((log_dir / "run.json").read_text())
    overrides = dict(meta.get("overrides") or {})
    overrides.update({"eval_envs": eval_envs, "eval_num_examples": eval_num_examples})
    cfg = C.apply_overrides(C.from_toml(meta["config"]), **overrides)
    base = meta["base_model"]
    renderer = get_renderer(tokenizer_id(base))
    service = tinker.ServiceClient()
    envs = [load_eval_env(spec) for spec in cfg.eval_envs]
    out_dir = log_dir / "evals"
    for ck in sorted(checkpoints(log_dir), key=lambda c: c["step"]):
        step = int(ck["step"])
        if steps is not None and step not in steps:
            continue
        sampler = await service.create_sampling_client(model_path=ck["sampler_path"])
        client = TinkerClient(Policy(sampler, step), renderer, max_model_len=cfg.max_model_len,
                              max_context=max_context(base), tito=cfg.use_renderer)
        for le in envs:
            res = await evaluate_env(le, client, base, out_dir / f"step_{step}" / f"{le.name}.jsonl")
            with open(out_dir / f"{le.name}.jsonl", "a") as fh:
                fh.write(json.dumps({"step": step, **res}) + "\n")
            logger.info("eval %s step %d: avg=%.4f (n=%d, failed=%d)", le.name, step, res["avg"], res["n"],
                        res["failed"])
    for le in envs:
        await le.env._teardown()
