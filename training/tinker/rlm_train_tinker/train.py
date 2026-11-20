"""prime-rl's `rl` loop (orchestrator + trainer) on Tinker, following prime-rl@eaf95997f.

Rollout groups stay in flight and sample from the latest weights; groups older than max_off_policy_steps are
replaced. A step takes the first finished groups once the policy is within max_async_level, then runs advantages,
filters and one trainer step (DPPO + KL loss, AdamW). Inline eval and checkpoints (every 10 steps, resumable) match
prime-rl.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import statistics
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import tinker

from . import algo
from .client import Policy, TinkerClient
from .config import RunConfig
from .envs import LoadedEnv, load_eval_env, load_train_env, run_group, run_one
from .loss import INPUT_RANKS, make_dppo_kl_loss
from .model import get_renderer, get_tokenizer, max_context, tokenizer_id
from .samples import build_datums, rollout_samples

logger = logging.getLogger(__name__)

MAX_EMPTY_BATCH_ATTEMPTS = 3


def _plain(o: Any) -> Any:
    return o.model_dump(exclude_none=True) if hasattr(o, "model_dump") else str(o)


def _jsonl(path: Path, row: dict[str, Any], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as fh:
        fh.write(json.dumps(row, default=_plain) + "\n")


def rollout_record(r: dict[str, Any]) -> dict[str, Any]:
    """Rollout as saved to disk (no token-level trajectory)."""
    rec = {k: v for k, v in r.items() if k not in ("trajectory", "prompt")}
    traj = r.get("trajectory") or []
    rec["num_steps"] = len(traj)
    rec["completion_tokens"] = sum(len((s.get("tokens") or {}).get("completion_ids") or []) for s in traj)
    return rec


def summarize(rollouts: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    if not rollouts:
        return {}
    rewards = [float(r["reward"]) for r in rollouts]
    out = {f"{prefix}/reward": statistics.fmean(rewards), f"{prefix}/n": float(len(rollouts))}
    for k in sorted({k for r in rollouts for k in (r.get("metrics") or {})}):
        vals = [float(v) for r in rollouts if isinstance(v := (r.get("metrics") or {}).get(k), (int, float))]
        if vals:
            out[f"{prefix}/{k}"] = statistics.fmean(vals)
    out[f"{prefix}/is_truncated"] = statistics.fmean(float(bool(r.get("is_truncated"))) for r in rollouts)
    out[f"{prefix}/num_turns"] = statistics.fmean(float(len(r.get("trajectory") or [])) for r in rollouts)
    for k, v in Counter(str(r.get("stop_condition")) for r in rollouts).items():
        out[f"{prefix}/stop/{k}"] = v / len(rollouts)
    return out


class Run:
    def __init__(self, cfg: RunConfig, log_dir: str | Path, *, base_model: str, seed: int = 0,
                 inline_eval: bool = True, ckpt_interval: int = 10, wandb_project: str | None = None):
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.base_model = base_model
        self.seed = seed
        self.inline_eval = inline_eval
        self.ckpt_interval = ckpt_interval
        self.wandb_project = wandb_project
        self.rng = random.Random(seed)
        self.ckpt_file = self.log_dir / "checkpoints.jsonl"
        self._wandb = None
        self._last_eval: dict[str, int] = {}

    async def setup(self) -> None:
        cfg = self.cfg
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tok_name = tokenizer_id(self.base_model)
        self.renderer = get_renderer(tok_name)
        self.vocab_size = int(get_tokenizer(tok_name).vocab_size)
        # main thread: Environment.__init__ installs signal handlers
        self.train_envs: dict[str, LoadedEnv] = {}
        for spec in cfg.train_envs:
            le = load_train_env(spec, cfg.seq_len, cfg.buffer.get("seed"))
            self.train_envs[spec.name] = le
            logger.info("train env %s (%s): %d examples", spec.name, spec.id, len(le.rows))
        self.eval_envs: list[LoadedEnv] = []
        if self.inline_eval:
            for spec in cfg.eval_envs:
                le = load_eval_env(spec)
                self.eval_envs.append(le)
                logger.info("eval env %s (%s): %d examples", spec.name, spec.id, len(le.rows))
        self.buffer = algo.Buffer({n: le.rows for n, le in self.train_envs.items()},
                                  {n: le.spec.ratio for n, le in self.train_envs.items()}, cfg.buffer, self.rng)
        self.service = tinker.ServiceClient()
        self.tc = await self.service.create_lora_training_client(
            base_model=self.base_model, rank=cfg.lora_rank, seed=self.seed,
            train_mlp=cfg.train_mlp, train_attn=cfg.train_attn, train_unembed=False)
        self.loss_handle = await self.service.create_custom_loss(make_dppo_kl_loss(cfg.loss), input_ranks=INPUT_RANKS)
        self.start_step = 0
        last = self._last_checkpoint()
        if last is not None:
            logger.info("resuming from step %d (%s)", last["step"], last["state_path"])
            await self.tc.load_state_with_optimizer(last["state_path"])
            self.start_step = int(last["step"])
            for env_name, ids in (last.get("evicted") or {}).items():
                for i in ids:
                    self.buffer.pools.get(env_name, {}).pop(i, None)
            sampler = await self.tc.create_sampling_client(last["sampler_path"])
        else:
            sampler = await self._save_checkpoint(0, with_state=False)
        self.policy = Policy(sampler, version=self.start_step)
        self.client = self._client(self.policy)
        if self.wandb_project:
            import wandb

            self._wandb = wandb.init(project=self.wandb_project, name=f"{cfg.wandb_name or 'run'}-tinker-s{self.seed}",
                                     dir=str(self.log_dir), config={"source": cfg.source, "base_model": self.base_model})

    def _client(self, policy: Policy) -> TinkerClient:
        return TinkerClient(policy, self.renderer, max_model_len=self.cfg.max_model_len,
                            max_context=max_context(self.base_model), tito=self.cfg.use_renderer)

    def _last_checkpoint(self) -> dict[str, Any] | None:
        if not self.ckpt_file.exists():
            return None
        rows = [json.loads(line) for line in self.ckpt_file.read_text().splitlines() if line.strip()]
        rows = [r for r in rows if r.get("state_path")]
        return rows[-1] if rows else None

    async def _save_checkpoint(self, step: int, with_state: bool = True):
        name = f"step_{step:06d}"
        saved = await self.tc.save_weights_for_sampler(name=name)
        rec: dict[str, Any] = {"step": step, "sampler_path": saved.path, "time": time.time()}
        if with_state:
            rec["state_path"] = (await self.tc.save_state(name=name)).path
            evicted = {}
            for env_name, le in self.train_envs.items():
                live = self.buffer.pools[env_name]
                gone = [r["example_id"] for r in le.rows if r["example_id"] not in live]
                if gone:
                    evicted[env_name] = gone
            rec["evicted"] = evicted
        _jsonl(self.ckpt_file, rec)
        logger.info("checkpoint step %d: %s", step, saved.path)
        return await self.tc.create_sampling_client(saved.path)

    async def _producer(self) -> None:
        cfg = self.cfg
        rpe = cfg.rollouts_per_example
        max_groups = max(1, int(cfg.max_inflight_rollouts or cfg.batch_size) // rpe)
        active: dict[asyncio.Task, tuple[int, str, Any]] = {}
        while not self._stop:
            # finished groups count as in flight until a batch takes them, as in prime-rl
            for t, (v0, _, _) in list(active.items()):
                if self.policy.version - v0 > cfg.max_off_policy_steps and not t.done():
                    t.cancel()
                    self._stats["cancelled_stale_groups"] += 1
            fresh = [g for g in self._groups if self.policy.version - g[0]["policy_version_start"] <= cfg.max_off_policy_steps]
            if len(fresh) < len(self._groups):
                self._stats["cancelled_stale_groups"] += len(self._groups) - len(fresh)
                self._groups = deque(fresh)
            while len(active) + len(self._groups) < max_groups:
                env_name, example = self.buffer.sample_example()
                t = asyncio.create_task(run_group(self.train_envs[env_name], example, rpe, self.client,
                                                  self.base_model, cfg.max_error_reschedule_attempts))
                active[t] = (self.policy.version, env_name, example["example_id"])
            if not active:
                await asyncio.sleep(0.5)
                continue
            done, _ = await asyncio.wait(list(active), timeout=5.0, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                v0, env_name, example_id = active.pop(t)
                if t.cancelled():
                    continue
                if t.exception() is not None:
                    logger.error("group failed: %r", t.exception())
                    self._stats["failed_groups"] += 1
                    continue
                group = t.result()
                if group is None:
                    self._stats["dropped_groups"] += 1
                    continue
                if not self.buffer.accept(env_name, example_id, group):
                    self._stats["difficulty_filtered_groups"] += 1
                    continue
                for r in group:
                    r["policy_version_start"] = v0
                self._groups.append(group)
                self._groups_ready.set()
        for t in active:
            t.cancel()

    async def _collect_batch(self) -> list[list[dict[str, Any]]]:
        need = self.cfg.batch_size // self.cfg.rollouts_per_example
        while len(self._groups) < need:
            if self._producer_task.done():
                self._producer_task.result()
                raise RuntimeError("rollout producer stopped")
            self._groups_ready.clear()
            try:
                await asyncio.wait_for(self._groups_ready.wait(), timeout=10)
            except TimeoutError:
                pass
        return [self._groups.popleft() for _ in range(need)]

    async def _train_step(self, step: int, rollouts: list[dict[str, Any]], prev: asyncio.Task | None) -> None:
        if prev is not None:
            await prev
        cfg = self.cfg
        t0 = time.time()
        samples = [s for r in rollouts if not r["is_filtered"] for s in rollout_samples(r, cfg.seq_len)]
        datums, n_loss = build_datums(samples)
        metrics: dict[str, Any] = {"step": step, "train/num_samples": len(samples), "train/num_datums": len(datums),
                                   "train/loss_tokens": n_loss}
        if samples:
            metrics["train/sample_tokens_mean"] = statistics.fmean(s.num_tokens for s in samples)
        if datums:
            fb = self.tc.forward_backward_custom(datums, self.loss_handle)
            op = self.tc.optim_step(tinker.AdamParams(
                learning_rate=cfg.lr, beta1=cfg.betas[0], beta2=cfg.betas[1], eps=1e-8,
                weight_decay=cfg.weight_decay, grad_clip_norm=cfg.max_norm))
            out = await fb
            await op
            for k, v in (out.metrics or {}).items():
                metrics[f"train/{k}"] = v
            n = metrics.get("train/custom/loss_tokens:sum") or 0
            if n:
                metrics["train/masked_fraction"] = metrics.get("train/custom/masked_tokens:sum", 0) / n
                metrics["train/mismatch_kl"] = metrics.get("train/custom/mismatch_kl_sum:sum", 0) / n
        else:
            logger.warning("step %d: no trainable tokens; optimizer step skipped", step)
        version = step + 1
        if version % self.ckpt_interval == 0 or version == cfg.max_steps:
            sampler = await self._save_checkpoint(version)
        else:
            sampler = await self.tc.save_weights_and_get_sampling_client(name=f"live_{version:06d}")
        self.policy.update(sampler, version)
        metrics["time/train_step_s"] = time.time() - t0
        self._log(metrics)

    def _log(self, metrics: dict[str, Any]) -> None:
        _jsonl(self.log_dir / "metrics.jsonl", metrics)
        if self._wandb is not None:
            self._wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, step=int(metrics["step"]))

    def _due_evals(self, version: int) -> list[LoadedEnv]:
        due = []
        for le in self.eval_envs:
            last = self._last_eval.get(le.name, -1)
            if version == last:
                continue
            if (version == 0 and self.cfg.eval_base_model) or (version > 0 and version % le.spec.interval == 0):
                due.append(le)
        return due

    async def _evaluate(self, le: LoadedEnv, version: int, sampler: Any) -> None:
        self._last_eval[le.name] = version
        res = await evaluate_env(le, self._client(Policy(sampler, version)), self.base_model,
                                 self.log_dir / "rollouts" / f"step_{version}" / f"eval_{le.name}.jsonl")
        self._log({"step": version, **{f"eval/{le.name}/{k}": v for k, v in res.items()}})
        logger.info("eval %s at step %d: avg@%d=%.4f (n=%d, failed=%d)", le.name, version,
                    le.spec.rollouts_per_example, res["avg"], res["n"], res["failed"])

    async def run(self) -> None:
        await self.setup()
        cfg = self.cfg
        self._stop = False
        self._groups: deque[list[dict[str, Any]]] = deque()
        self._groups_ready = asyncio.Event()
        self._stats: Counter = Counter()
        producer = self._producer_task = asyncio.create_task(self._producer())
        prev: asyncio.Task | None = None
        evals: list[asyncio.Task] = []
        try:
            for step in range(self.start_step, cfg.max_steps):
                for le in self._due_evals(self.policy.version):
                    evals.append(asyncio.create_task(self._evaluate(le, self.policy.version, self.policy.sampling_client)))
                while self.policy.version < step - cfg.max_async_level:
                    await asyncio.sleep(0.5)
                t0 = time.time()
                for attempt in range(MAX_EMPTY_BATCH_ATTEMPTS):
                    groups = await self._collect_batch()
                    for g in groups:
                        algo.compute_advantages(g, cfg.length_penalty)
                    rollouts = [r for g in groups for r in g]
                    fcounts = algo.apply_filters(rollouts, cfg.filters, self.vocab_size)
                    if any(not r["is_filtered"] for r in rollouts):
                        break
                    logger.warning("step %d: all %d rollouts filtered (attempt %d)", step, len(rollouts), attempt + 1)
                else:
                    raise RuntimeError(f"all rollouts filtered out at step {step} in {MAX_EMPTY_BATCH_ATTEMPTS} attempts")
                lag = [step - r["policy_version_start"] for r in rollouts]
                m: dict[str, Any] = {
                    "step": step, "time/collect_s": time.time() - t0, "policy_version": self.policy.version,
                    "off_policy/mean": statistics.fmean(lag), "off_policy/max": max(lag),
                    "filters/filtered_fraction": statistics.fmean(float(r["is_filtered"]) for r in rollouts),
                    **{f"filters/{k}": v for k, v in fcounts.items()},
                    **{f"sched/{k}": v for k, v in self._stats.items()},
                    **{f"buffer/evicted_{k}": v for k, v in self.buffer.evicted.items()},
                    **summarize(rollouts, "train"),
                }
                if len(self.train_envs) > 1:
                    for name in self.train_envs:
                        m.update(summarize([r for r in rollouts if r["env_name"] == name], f"train/{name}"))
                self._log(m)
                for r in rollouts:
                    _jsonl(self.log_dir / "rollouts" / f"step_{step}" / "train_rollouts.jsonl", rollout_record(r))
                logger.info("step %d: reward %.4f, filtered %.2f, off-policy %.2f", step, m.get("train/reward", 0.0),
                            m["filters/filtered_fraction"], m["off_policy/mean"])
                prev = asyncio.create_task(self._train_step(step, rollouts, prev))
            if prev is not None:
                await prev
            for le in [le for le in self.eval_envs if self._last_eval.get(le.name) != self.policy.version]:
                evals.append(asyncio.create_task(self._evaluate(le, self.policy.version, self.policy.sampling_client)))
            if evals:
                await asyncio.gather(*evals)
        finally:
            self._stop = True
            producer.cancel()
            try:
                await producer
            except (asyncio.CancelledError, Exception):
                pass
            for le in [*self.train_envs.values(), *self.eval_envs]:
                try:
                    await le.env._teardown()
                except Exception:  # noqa: BLE001
                    logger.exception("teardown of %s failed", le.name)
            if self._wandb is not None:
                self._wandb.finish()


async def evaluate_env(le: LoadedEnv, client: TinkerClient, model: str, out_path: Path,
                       max_concurrency: int = 64) -> dict[str, Any]:
    """As prime-rl EvalEnv.evaluate; the average skips errored rollouts."""
    sem = asyncio.Semaphore(max_concurrency)

    async def one(row: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            try:
                return await run_one(le, row, client, model, le.spec.sampling_args)
            except Exception as e:  # noqa: BLE001
                logger.exception("eval rollout raised")
                return {"example_id": row.get("example_id"), "env_name": le.name, "reward": 0.0,
                        "error": {"error_chain_repr": repr(e)}, "trajectory": []}

    t0 = time.time()
    outs = await asyncio.gather(*[one(row) for row in le.rows for _ in range(le.spec.rollouts_per_example)])
    ok = [o for o in outs if o.get("error") is None]
    res: dict[str, Any] = {"avg": statistics.fmean(float(o["reward"]) for o in ok) if ok else float("nan"),
                           "n": len(outs), "failed": len(outs) - len(ok), "wall_s": time.time() - t0}
    res.update({k.removeprefix("eval/"): v for k, v in summarize(ok, "eval").items() if k != "eval/reward"})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for o in outs:
            fh.write(json.dumps(rollout_record(o), default=_plain) + "\n")
    return res
