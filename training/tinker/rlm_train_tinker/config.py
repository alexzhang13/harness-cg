"""Read a prime-rl config TOML into a Tinker run config, with prime-rl@eaf95997f defaults.

Settings with no Tinker equivalent are listed in RunConfig.notes.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .loss import LossConfig

SUBSTITUTED_VARS = ("REPO", "ADALEVAL_DATA_DIR", "OOLONG_REAL_DATA_DIR", "INIT_WEIGHTS", "SFT_DATA_DIR")
DEFAULT_FILTERS = [
    {"type": "gibberish", "enforce": False},
    {"type": "repetition", "enforce": False, "window": 3000, "prob_threshold": 0.99},
    {"type": "zero_advantage", "enforce": True},
]
_MLP_TARGETS = {"gate_proj", "up_proj", "down_proj"}
_ATTN_TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj"}


@dataclass
class EnvSpec:
    id: str
    name: str
    args: dict[str, Any]
    sampling_args: dict[str, Any]
    max_retries: int = 0
    ratio: float | None = None
    # eval envs only
    num_examples: int = -1
    rollouts_per_example: int = 1
    interval: int = 100


@dataclass
class RunConfig:
    source: str
    model_name: str
    max_steps: int
    seq_len: int
    max_model_len: int
    lr: float
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    max_norm: float = 1.0
    lora_rank: int = 32
    lora_alpha: float = 64.0
    train_attn: bool = True
    train_mlp: bool = False
    batch_size: int = 64
    rollouts_per_example: int = 4
    max_inflight_rollouts: int | None = None
    max_async_level: int = 1
    max_off_policy_steps: int = 8
    max_error_reschedule_attempts: int = 3
    use_renderer: bool = True
    filters: list[dict[str, Any]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_FILTERS))
    buffer: dict[str, Any] = field(default_factory=dict)
    length_penalty: dict[str, Any] | None = None
    loss: LossConfig = field(default_factory=LossConfig)
    train_envs: list[EnvSpec] = field(default_factory=list)
    eval_envs: list[EnvSpec] = field(default_factory=list)
    eval_base_model: bool = True
    output_dir: str = ""
    wandb_project: str | None = None
    wandb_name: str | None = None
    notes: list[str] = field(default_factory=list)


def _substitute(text: str) -> str:
    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in SUBSTITUTED_VARS:
            return m.group(0)
        if not os.environ.get(name):
            raise KeyError(f"config needs ${name}; export it first")
        return os.environ[name]

    return re.sub(r"\$\{([A-Z_]+)\}", repl, text)


def _train_sampling_args(d: dict[str, Any]) -> dict[str, Any]:
    """As prime-rl TrainSamplingConfig.to_sampling_args, plus the orchestrator's extra_body defaults."""
    args: dict[str, Any] = {"temperature": float(d.get("temperature", 1.0)), "top_p": 1.0, "logprobs": True}
    mct = d.get("max_completion_tokens", d.get("max_tokens"))
    if mct is not None:
        args["max_completion_tokens"] = int(mct)
    if d.get("seed") is not None:
        args["seed"] = d["seed"]
    extra = dict(d.get("extra_body") or {})
    if d.get("min_tokens"):
        extra["min_tokens"] = d["min_tokens"]
    if d.get("repetition_penalty", 1.0) != 1.0:
        extra["repetition_penalty"] = d["repetition_penalty"]
    extra.setdefault("top_k", -1)
    extra.setdefault("min_p", 0.0)
    extra.setdefault("return_token_ids", True)
    args["extra_body"] = extra
    return args


def _eval_sampling_args(d: dict[str, Any]) -> dict[str, Any]:
    """As prime-rl EvalSamplingConfig.to_sampling_args (set fields only)."""
    args: dict[str, Any] = {}
    for key in ("temperature", "top_p", "seed"):
        if d.get(key) is not None:
            args[key] = d[key]
    mct = d.get("max_completion_tokens", d.get("max_tokens"))
    if mct is not None:
        args["max_completion_tokens"] = int(mct)
    extra = dict(d.get("extra_body") or {})
    for key in ("top_k", "min_p", "min_tokens", "repetition_penalty"):
        if d.get(key) is not None:
            extra[key] = d[key]
    if extra:
        args["extra_body"] = extra
    return args


def _merge_sampling(group: dict[str, Any], env: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(group)
    for k, v in (env or {}).items():
        merged[k] = {**dict(merged.get(k) or {}), **v} if k == "extra_body" else v
    return merged


def load_toml(path: str | Path) -> dict[str, Any]:
    return tomllib.loads(_substitute(Path(path).read_text()))


def from_toml(path: str | Path) -> RunConfig:
    d = load_toml(path)
    orch = d.get("orchestrator", {})
    trainer = d.get("trainer", {})
    tmodel = trainer.get("model", {})
    optim = trainer.get("optim", {})
    lora = tmodel.get("lora", {})
    notes: list[str] = []

    model_name = d.get("model", {}).get("name", "")
    targets = set(lora.get("target_modules") or (_ATTN_TARGETS | _MLP_TARGETS))
    moe = "-A3B" in model_name
    cfg = RunConfig(
        source=str(path),
        model_name=model_name,
        max_steps=int(d.get("max_steps", orch.get("max_steps", 0)) or 0),
        seq_len=int(d.get("seq_len", orch.get("seq_len", tmodel.get("seq_len", 2048)))),
        max_model_len=int(d.get("inference", {}).get("model", {}).get("max_model_len", 32768)),
        lr=float(optim.get("lr", 1e-6)),
        weight_decay=float(optim.get("weight_decay", 0.01)),
        betas=(float(optim.get("betas1", 0.9)), float(optim.get("betas2", 0.999))),
        max_norm=float(optim.get("max_norm", 1.0)),
        lora_rank=int(lora.get("rank", 32)),
        lora_alpha=float(lora.get("alpha", 64.0)),
        train_attn=bool(targets & _ATTN_TARGETS),
        # On the MoE model prime-rl's MLP target names match nothing, so the paper's LoRA is attention-only.
        train_mlp=bool(targets & _MLP_TARGETS) and not moe,
        batch_size=int(orch.get("batch_size", 128)),
        rollouts_per_example=int(orch.get("rollouts_per_example", 1)),
        max_inflight_rollouts=orch.get("max_inflight_rollouts"),
        max_async_level=int(orch.get("max_async_level", d.get("max_async_level", 1))),
        max_off_policy_steps=int(orch.get("max_off_policy_steps", 8)),
        max_error_reschedule_attempts=int(orch.get("max_error_reschedule_attempts", 3)),
        use_renderer=bool(orch.get("use_renderer", True)),
        filters=[dict(f) for f in orch.get("filters", [])] or copy.deepcopy(DEFAULT_FILTERS),
        buffer=dict(orch.get("buffer") or {}),
        length_penalty=dict(orch.get("advantage", {}).get("length_penalty") or {}) or None,
        output_dir=d.get("output_dir", ""),
        wandb_project=d.get("wandb", {}).get("project"),
        wandb_name=d.get("wandb", {}).get("name"),
    )
    loss = trainer.get("loss", {}) or {}
    cfg.loss = LossConfig(dppo_mask_low=float(loss.get("dppo_mask_low", 0.2)),
                          dppo_mask_high=float(loss.get("dppo_mask_high", 0.2)),
                          adv_tau=float(loss.get("adv_tau", 1.0)), kl_tau=float(loss.get("kl_tau", 1e-3)))
    if not lora:
        raise ValueError(f"{path}: full fine-tuning config (no [trainer.model.lora]); Tinker trains LoRA only")

    tr = orch.get("train", {})
    group_sampling = dict(tr.get("sampling") or {})
    for e in tr.get("env", []):
        cfg.train_envs.append(EnvSpec(
            id=e["id"], name=e.get("name") or e["id"].split("@")[0], args=dict(e.get("args") or {}),
            sampling_args=_train_sampling_args(_merge_sampling(group_sampling, e.get("sampling"))),
            max_retries=int(e.get("max_retries", tr.get("max_retries", 0))), ratio=e.get("ratio")))
    ratios = [e.ratio for e in cfg.train_envs]
    if any(r is not None for r in ratios) and any(r is None for r in ratios):
        raise ValueError(f"{path}: set `ratio` on every train env or on none")

    ev = orch.get("eval")
    if ev:
        cfg.eval_base_model = bool(ev.get("eval_base_model", True))
        group_sampling = dict(ev.get("sampling") or {})
        for e in ev.get("env", []):
            cfg.eval_envs.append(EnvSpec(
                id=e["id"], name=e.get("name") or e["id"].split("@")[0], args=dict(e.get("args") or {}),
                sampling_args=_eval_sampling_args(_merge_sampling(group_sampling, e.get("sampling"))),
                max_retries=int(e.get("max_retries", ev.get("max_retries", 0))),
                num_examples=int(e.get("num_examples", ev.get("num_examples", -1))),
                rollouts_per_example=int(e.get("rollouts_per_example", ev.get("rollouts_per_example", 1))),
                interval=int(e.get("interval", ev.get("interval", 100)))))

    if cfg.lora_alpha != cfg.lora_rank:
        notes.append(f"LoRA alpha {cfg.lora_alpha:g} / rank {cfg.lora_rank} (scale {cfg.lora_alpha / cfg.lora_rank:g}): "
                     "Tinker's LoRA scale is fixed at 1 (alpha = rank)")
    if moe and targets & _MLP_TARGETS:
        notes.append("LoRA on attention only: prime-rl's MLP targets matched no expert module of the MoE paper model")
    if any(e.sampling_args.get("temperature", 1.0) != 1.0 for e in cfg.train_envs):
        notes.append("train temperature != 1: Tinker returns T=1 logprobs, so the importance ratio and KL use T=1 "
                     "on both sides (prime-rl: logits / T)")
    cfg.notes = notes
    return cfg


def apply_overrides(cfg: RunConfig, **kw: Any) -> RunConfig:
    """Command-line overrides (--max-steps etc.); None is ignored."""
    cfg = copy.deepcopy(cfg)
    if kw.get("train_envs"):
        keep = set(kw["train_envs"])
        cfg.train_envs = [e for e in cfg.train_envs if e.name in keep]
        missing = keep - {e.name for e in cfg.train_envs}
        if missing:
            raise KeyError(f"unknown train env name(s): {sorted(missing)}")
    if kw.get("eval_envs") is not None:
        keep = set(kw["eval_envs"])
        cfg.eval_envs = [e for e in cfg.eval_envs if e.name in keep]
    if kw.get("eval_num_examples") is not None:
        for e in cfg.eval_envs:
            e.num_examples = int(kw["eval_num_examples"])
    if kw.get("eval_interval") is not None:
        for e in cfg.eval_envs:
            e.interval = int(kw["eval_interval"])
    for key in ("max_steps", "batch_size", "rollouts_per_example", "max_inflight_rollouts"):
        if kw.get(key) is not None:
            setattr(cfg, key, int(kw[key]))
    for spec in kw.get("env_args") or []:  # key=<json> for envs that already set key
        key, _, raw = spec.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        for e in cfg.train_envs + cfg.eval_envs:
            if key in e.args:
                e.args[key] = value
    if cfg.batch_size % cfg.rollouts_per_example:
        raise ValueError("batch_size must be a multiple of rollouts_per_example")
    return cfg
