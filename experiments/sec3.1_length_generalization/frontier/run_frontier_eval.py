"""Frontier reference line of Fig. length-gen: a hosted model (GPT-6 Astra) in the RLM harness, no orchestrator prompt.

    PRIME_API_KEY=... PRIME_API_TEAM=... python run_frontier_eval.py --task all --model openai/gpt-6-astra

Each task uses the long eval split and harness settings of configs/rlm/<task>.toml; root and sub-calls use the same
model. Results go to results/<model>/<task>.json (resumable); collect_baseline.py builds baseline.json.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from verifiers import OpenAIChatCompletionsClient

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / "results"
RUN_LOCK = HERE / ".eval.lock"

EVAL_CONFIG = {
    "mrcr":         "configs/rlm/mrcr.toml",
    "graphwalks":   "configs/rlm/graphwalks.toml",
    "longbenchpro": "configs/rlm/longbenchpro.toml",
    "oolong":       "configs/rlm/oolong.toml",
    "oolong_pairs": "configs/rlm/oolong-pairs.toml",
    "adaleval_bestanswer": "configs/rlm/adaleval.toml",
}

PANEL_KEY = {
    "mrcr":         "MRCRv2",
    "graphwalks":   "GraphWalks",
    "longbenchpro": "LongBenchPro",
    "oolong":       "OOLONG",
    "oolong_pairs": "OOLONG-Pairs",
    "adaleval_bestanswer": "AdaLEval-BestAnswer",
}

TASK_ARGS: dict[str, dict[str, Any]] = {
    "mrcr": {
        "module": "mrcr_v2_real",
        "max_tokens": 4096,
        "args": dict(
            needle_count=8, context_range="2m-4m", num_examples=50, seed=42,
            max_iterations=20, sub_max_tokens=4096,
        ),
    },
    "graphwalks": {
        "module": "graphwalks_real",
        "max_tokens": 4096,
        "args": dict(
            split="256k_to_1mil", num_examples=50, seed=42,
            max_iterations=20, sub_max_tokens=4096,
        ),
    },
    "longbenchpro": {
        "module": "longbenchpro_real",
        "max_tokens": 4096,
        "args": dict(
            language="English", token_length="256k", thinking=False,
            num_examples=-1, seed=42, max_iterations=20, sub_max_tokens=4096,
        ),
    },
    "oolong": {
        "module": "oolong_real",
        "max_tokens": 4096,
        "args": dict(
            dataset_name="trec_coarse", min_ctx=262144, max_ctx=262144,
            exclude_numeric=False, num_examples=50, seed=42,
            max_iterations=20, sub_max_tokens=4096,
        ),
    },
    "oolong_pairs": {
        "module": "oolong_pairs_real",
        "max_tokens": 4096,
        "args": dict(
            context_length=32768, num_examples=20, seed=42,
            max_iterations=30, sub_max_tokens=4096,
            max_sub_calls=1000,
        ),
    },
    "adaleval_bestanswer": {
        "module": "ada_leval_real",
        "max_tokens": 4096,
        "args": dict(
            task="bestanswer", setting="128k", num_examples=50,
            manifest_split="eval",
            max_iterations=20, sub_max_tokens=4096,
            min_iterations=2, min_subcall=0,
        ),
    },
}

# MRCR (slowest) near the end
ALL_TASKS = ["oolong_pairs", "oolong", "graphwalks", "longbenchpro", "mrcr",
             "adaleval_bestanswer"]


def build_client(base_url: str | None = None, request_timeout: float = 1800.0,
                 max_retries: int = 6) -> OpenAIChatCompletionsClient:
    """Client for Prime Inference (PRIME_API_KEY) or OPENAI_BASE_URL."""
    api_key = os.environ.get("PRIME_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set PRIME_API_KEY (or OPENAI_API_KEY).")
    resolved = base_url or os.environ.get("OPENAI_BASE_URL")
    if resolved is None and os.environ.get("PRIME_API_KEY"):
        resolved = "https://api.pinference.ai/api/v1"
    kw: dict[str, Any] = {"api_key": api_key, "timeout": request_timeout,
                          "max_retries": max_retries}
    if resolved:
        kw["base_url"] = resolved
    team = os.environ.get("PRIME_API_TEAM")
    if team and resolved and "pinference" in resolved:
        kw["default_headers"] = {"X-Prime-Team-ID": team}
    return OpenAIChatCompletionsClient(AsyncOpenAI(**kw))


def _usage(state: dict, key: str) -> int:
    u = state.get("usage") or {}
    try:
        return int(u.get(key) or 0)
    except Exception:
        return 0


def _prime_standard_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Rough cost estimate (no caching or long-context pricing)."""
    rates = {"openai/gpt-6-astra": (10.0, 50.0)}  # USD per million tokens
    price_in, price_out = rates.get(model, (None, None))
    if price_in is None:
        return None
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


async def one_rollout(env, idx, row, client, model, max_tokens, timeout):
    t0 = time.time()
    try:
        state = await asyncio.wait_for(
            env._run_rollout_state(
                input=row, client=client, model=model,
                sampling_args={"max_tokens": max_tokens},
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return {"idx": idx, "error": "timeout", "t": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "t": round(time.time() - t0, 1)}
    root_in = _usage(state, "input_tokens")
    root_out = _usage(state, "output_tokens")
    iterations = int(state.get("rlm_iterations") or 0)
    if iterations == 0 and root_in + root_out == 0:
        return {"idx": idx,
                "error": "upstream model failure: no root iteration or token usage; "
                         "inspect provider logs (will not score as zero)",
                "t": round(time.time() - t0, 1)}
    return {
        "idx": idx,
        "reward": float(state.get("reward", 0.0)),
        "answer": 1 if state.get("rlm_final_answer") else 0,
        "sub": int(state.get("rlm_sub_llm_calls") or 0),
        "iters": iterations,
        "repl": int(state.get("rlm_repl_calls") or 0),
        "trunc": 1 if state.get("is_truncated") else 0,
        "root_out_tok": root_out,
        "root_in_tok": root_in,
        "sub_in_tok": int(state.get("rlm_sub_llm_prompt_tokens") or 0),
        "sub_out_tok": int(state.get("rlm_sub_llm_completion_tokens") or 0),
        "t": round(time.time() - t0, 1),
    }


def _load_existing(path: Path, overwrite: bool) -> dict[int, dict]:
    if overwrite or not path.exists():
        return {}
    try:
        prev = json.loads(path.read_text())
    except Exception:
        return {}
    out = {}
    for r in prev.get("rollouts", []):
        if "error" not in r and r.get("idx") is not None:
            out[int(r["idx"])] = r
    return out


def summarize(rollouts: list[dict]) -> dict:
    ok = [r for r in rollouts if "error" not in r]
    errs = [r for r in rollouts if "error" in r]
    timeouts = sorted(r["idx"] for r in errs if r.get("error") == "timeout")
    other_err = sorted(r["idx"] for r in errs if r.get("error") != "timeout")
    n = len(ok)
    if not n:
        return {"n": 0, "errors": len(errs),
                "timeout_idxs": timeouts, "other_error_idxs": other_err}
    f = lambda k: sum(r[k] for r in ok) / n  # noqa: E731
    return {
        "n": n,
        "errors": len(errs),
        "timeout_idxs": timeouts,
        "other_error_idxs": other_err,
        "mean_reward": round(f("reward"), 4),
        "answer_rate": round(f("answer"), 3),
        "zero_sub_rate": round(sum(1 for r in ok if r["sub"] == 0) / n, 3),
        "mean_sub": round(f("sub"), 2),
        "mean_iters": round(f("iters"), 2),
        "trunc_rate": round(f("trunc"), 3),
        "mean_root_out_tok": round(f("root_out_tok")),
        "mean_root_in_tok": round(f("root_in_tok")),
        "root_input_tokens_total": sum(r.get("root_in_tok", 0) for r in ok),
        "root_output_tokens_total": sum(r.get("root_out_tok", 0) for r in ok),
        "sub_input_tokens_total": sum(r.get("sub_in_tok", 0) for r in ok),
        "sub_output_tokens_total": sum(r.get("sub_out_tok", 0) for r in ok),
        "fatal_upstream": any(str(r.get("error", "")).startswith("upstream model failure")
                               for r in errs),
    }


def _write(path: Path, task: str, model: str, spec: dict, rollouts: list[dict]):
    rollouts = sorted(rollouts, key=lambda r: r["idx"])
    payload = {
        "task": task,
        "panel_key": PANEL_KEY[task],
        "model": model,
        "hint": "none",
        "eval_config": EVAL_CONFIG[task],
        "env_args": spec["args"],
        "max_tokens": spec["max_tokens"],
        "summary": summarize(rollouts),
        "rollouts": rollouts,
    }
    if model == "openai/gpt-6-astra":
        s = payload["summary"]
        total_in = s["root_input_tokens_total"] + s["sub_input_tokens_total"]
        total_out = s["root_output_tokens_total"] + s["sub_output_tokens_total"]
        s["estimated_prime_cost_usd_standard_uncached"] = round(
            _prime_standard_cost(model, total_in, total_out), 4)
        s["cost_note"] = (
            "Usage-based estimate at Prime standard Astra rates ($10/M input, "
            "$50/M output); excludes cached-input discounts and any >272K "
            "single-request long-context multiplier."
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


async def run_task(task: str, model: str, concurrency: int, timeout: float,
                   overwrite: bool, client, results_dir: Path,
                   limit: int | None = None) -> dict:
    spec = TASK_ARGS[task]
    load = importlib.import_module(spec["module"]).load_environment
    print(f"\n===== task={task}  model={model}  args={spec['args']} =====",
          file=sys.stderr, flush=True)
    env = load(**spec["args"])
    n = len(env.dataset)
    print(f"[{task}] dataset has {n} datapoints (identical to eval-long split)",
          file=sys.stderr, flush=True)

    if limit is not None:
        n = min(n, limit)
        print(f"[{task}] --limit {limit}: SMOKE ONLY (subset != the eval set)",
              file=sys.stderr, flush=True)

    out_path = results_dir / f"{task}.json"
    done = _load_existing(out_path, overwrite)
    if done:
        print(f"[{task}] resuming: {len(done)}/{n} already scored",
              file=sys.stderr, flush=True)
    rollouts: list[dict] = list(done.values())
    todo = [i for i in range(n) if i not in done]
    if not todo:
        print(f"[{task}] all {n} datapoints already cached — nothing to run",
              file=sys.stderr, flush=True)
        try:
            await env.teardown_rlm()
        except Exception as e:
            print(f"[{task}] teardown warning: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
        return summarize(rollouts)

    lock = asyncio.Lock()
    sem = asyncio.Semaphore(max(1, concurrency))

    async def worker(idx: int):
        async with sem:
            r = await one_rollout(env, idx, dict(env.dataset[idx]), client,
                                  model, spec["max_tokens"], timeout)
        async with lock:
            rollouts.append(r)
            if "error" in r:
                print(f"  [{idx}] ERR {r['error']}  t={r['t']}s",
                      file=sys.stderr, flush=True)
            else:
                print(f"  [{idx}] reward={r['reward']:.3f} ans={r['answer']} "
                      f"sub={r['sub']} iters={r['iters']} trunc={r['trunc']} "
                      f"inTok={r['root_in_tok']} subTok={r['sub_in_tok']}/"
                      f"{r['sub_out_tok']} t={r['t']}s",
                      file=sys.stderr, flush=True)
            _write(out_path, task, model, spec, rollouts)

    try:
        await asyncio.gather(*(worker(i) for i in todo))
    finally:
        try:
            await env.teardown_rlm()
        except Exception as e:
            print(f"[{task}] teardown warning: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
    s = summarize(rollouts)
    note = ""
    if s.get("timeout_idxs"):
        note += f"  TIMEOUTS={s['timeout_idxs']}"
    if s.get("other_error_idxs"):
        note += f"  ERRORS={s['other_error_idxs']}"
    if note:
        note += "  (re-run the same command to resume & retry these)"
    print(f"[{task}] DONE  mean_reward={s.get('mean_reward')}  "
          f"n={s.get('n')}  errors={s.get('errors')}  -> {out_path}{note}",
          file=sys.stderr, flush=True)
    return s


async def amain():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all",
                   help=f"one of {ALL_TASKS} or 'all'")
    p.add_argument("--model", default="openai/gpt-6-astra")
    p.add_argument("--concurrency", type=int, default=4,
                   help="concurrent datapoints per task")
    p.add_argument("--timeout", type=float, default=0.0,
                   help="per-rollout wall-clock cap in s; 0 (default) = NO cap, "
                        "run every rollout to completion (avoids premature timeouts)")
    p.add_argument("--request-timeout", type=float, default=1800.0,
                   help="per-HTTP-call timeout in s (SDK default 600 is too tight "
                        "for a slow reasoning model on a long completion)")
    p.add_argument("--overwrite", action="store_true",
                   help="ignore existing results and re-run from scratch")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="directory for resumable result JSON files (default: results/<model name>)")
    p.add_argument("--base-url", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="SMOKE ONLY: cap datapoints (subset != the real eval set)")
    args = p.parse_args()

    rollout_timeout = args.timeout if args.timeout and args.timeout > 0 else None

    if args.results_dir is None:
        args.results_dir = RESULTS_ROOT / args.model.split("/")[-1]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(args.base_url, request_timeout=args.request_timeout)
    print(f"per-rollout timeout: {'OFF (run to completion)' if rollout_timeout is None else f'{rollout_timeout}s'}"
          f"  |  per-HTTP-call timeout: {args.request_timeout}s",
          file=sys.stderr, flush=True)
    tasks = ALL_TASKS if args.task == "all" else [args.task]
    for t in tasks:
        if t not in TASK_ARGS:
            raise SystemExit(f"unknown task {t!r}; known: {ALL_TASKS} or 'all'")

    with RUN_LOCK.open("a") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise SystemExit(f"another frontier eval holds {RUN_LOCK}; refusing overlap") from e
        summaries = {}
        for t in tasks:
            summaries[t] = await run_task(
                t, args.model, args.concurrency, rollout_timeout, args.overwrite,
                client, args.results_dir, args.limit)
            if summaries[t].get("fatal_upstream"):
                print(f"\nStopping suite after upstream failure in {t}; "
                      "no further benchmark calls will be sent.", file=sys.stderr, flush=True)
                break

    print("\n## Frontier baseline summary (gray dotted line value = mean_reward)\n")
    print("| task | panel | n | mean_reward | answer% | zero-sub% | trunc% | timeouts | errors |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t, s in summaries.items():
        nto = len(s.get("timeout_idxs") or [])
        noe = len(s.get("other_error_idxs") or [])
        if not s.get("n"):
            print(f"| {t} | {PANEL_KEY[t]} | 0 | all-error | | | | {nto} | {noe} |")
            continue
        print(f"| {t} | {PANEL_KEY[t]} | {s['n']} | "
              f"{s['mean_reward']:.3f} | {s['answer_rate']:.0%} | "
              f"{s['zero_sub_rate']:.0%} | {s['trunc_rate']:.0%} | {nto} | {noe} |")
    incomplete = {t: s for t, s in summaries.items()
                  if (s.get("timeout_idxs") or s.get("other_error_idxs"))}
    if incomplete:
        print("\n!! Some datapoints did not complete (recorded, not scored):")
        for t, s in incomplete.items():
            print(f"   {t}: timeouts={s.get('timeout_idxs')} errors={s.get('other_error_idxs')}")
        print("   Re-run the SAME command to resume — completed datapoints are "
              "cached/skipped, only these are retried.")


if __name__ == "__main__":
    asyncio.run(amain())
