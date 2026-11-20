"""rlm-train-tinker {show,train,eval,export}.

    rlm-train-tinker show <config.toml>                      resolved config + settings Tinker cannot match
    rlm-train-tinker train <config.toml> --log-dir <dir>     train (resumes from <dir>/checkpoints.jsonl)
    rlm-train-tinker eval <log-dir> [--steps 0,10]           evaluate saved checkpoints on Tinker
    rlm-train-tinker export <log-dir> [--steps 10,20]        download checkpoints as PEFT adapters
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

from . import config as C
from .model import tinker_base_model


def _csv(text: str | None) -> list[str] | None:
    if text is None:
        return None
    return [x for x in text.split(",") if x and x != "none"]


def _ints(text: str | None) -> list[int] | None:
    return None if text is None else [int(x) for x in text.split(",") if x]


def _add_overrides(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("overrides (smoke tests)")
    g.add_argument("--max-steps", type=int)
    g.add_argument("--batch-size", type=int)
    g.add_argument("--rollouts-per-example", type=int)
    g.add_argument("--max-inflight-rollouts", type=int)
    g.add_argument("--train-envs", help="comma list of train env names to keep")
    g.add_argument("--eval-envs", help="comma list of eval env names to keep ('none' for no inline eval)")
    g.add_argument("--eval-num-examples", type=int)
    g.add_argument("--eval-interval", type=int)
    g.add_argument("--env-arg", action="append", default=[], metavar="KEY=JSON",
                   help="set an env arg on every env that already has it (repeatable)")


def _overrides(a: argparse.Namespace) -> dict:
    return {"max_steps": a.max_steps, "batch_size": a.batch_size, "rollouts_per_example": a.rollouts_per_example,
            "max_inflight_rollouts": a.max_inflight_rollouts, "train_envs": _csv(a.train_envs),
            "eval_envs": _csv(a.eval_envs), "eval_num_examples": a.eval_num_examples,
            "eval_interval": a.eval_interval, "env_args": a.env_arg}


def _load(path: str, overrides: dict) -> C.RunConfig:
    return C.apply_overrides(C.from_toml(path), **overrides)


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "huggingface_hub", "urllib3", "aiohttp.access", "tinker", "datasets"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rlm-train-tinker", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("config")
    _add_overrides(s)
    t = sub.add_parser("train")
    t.add_argument("config")
    t.add_argument("--log-dir", required=True)
    t.add_argument("--seed", type=int, default=0, help="LoRA init and example sampling seed")
    t.add_argument("--base-model", default=None, help="Tinker base model (default: model.TINKER_BASE_MODELS)")
    t.add_argument("--no-inline-eval", action="store_true")
    t.add_argument("--ckpt-interval", type=int, default=10)
    t.add_argument("--wandb-project", default=None)
    t.add_argument("--log-level", default="info")
    _add_overrides(t)
    e = sub.add_parser("eval", help="evaluate the saved checkpoints of a run on Tinker")
    e.add_argument("log_dir")
    e.add_argument("--steps", default=None)
    e.add_argument("--eval-envs", default=None)
    e.add_argument("--eval-num-examples", type=int, default=None)
    e.add_argument("--log-level", default="info")
    x = sub.add_parser("export", help="download checkpoints as PEFT adapters into <log-dir>/run_default/broadcasts")
    x.add_argument("log_dir")
    x.add_argument("--steps", default=None)
    x.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if a.cmd == "show":
        cfg = _load(a.config, _overrides(a))
        print(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
        return
    _setup_logging(getattr(a, "log_level", "info"))
    if a.cmd == "train":
        from .train import Run

        overrides = _overrides(a)
        cfg = _load(a.config, overrides)
        base = tinker_base_model(cfg.model_name, a.base_model)
        log_dir = Path(a.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        meta = {"config": str(Path(a.config).resolve()), "base_model": base, "seed": a.seed, "overrides": overrides,
                "argv": sys.argv[1:]}
        (log_dir / "run.json").write_text(json.dumps(meta, indent=2))
        for note in cfg.notes:
            logging.getLogger(__name__).info("not matched on Tinker: %s", note)
        asyncio.run(Run(cfg, log_dir, base_model=base, seed=a.seed, inline_eval=not a.no_inline_eval,
                        ckpt_interval=a.ckpt_interval, wandb_project=a.wandb_project).run())
        return
    if a.cmd == "eval":
        from .evaluate import evaluate_run

        asyncio.run(evaluate_run(Path(a.log_dir), steps=_ints(a.steps), eval_envs=_csv(a.eval_envs),
                                 eval_num_examples=a.eval_num_examples))
        return
    if a.cmd == "export":
        from .export import export_run

        for d in export_run(a.log_dir, steps=_ints(a.steps), force=a.force):
            print(d)


if __name__ == "__main__":
    main()
