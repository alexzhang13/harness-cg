"""Evaluate one served model (base model or a LoRA adapter) on an env's eval split and append a summary row.

    python offline_long_eval.py --task oolong --args-json '{"min_ctx": 262144, ...}' --num-examples 50 \
        --base-url http://localhost:8011/v1 --model step_50 --step 50 --out curve.jsonl

Default env singleturn-baseline (--env-id for another). Rollouts are cached, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone


def _g(o, key, default=None):
    """Get a field from a dict or an object."""
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _question_from_info(info) -> str:
    """The question from a rollout's info, without the context ("" on failure)."""
    try:
        meta = json.loads(info) if isinstance(info, str) else info
        rp = (meta or {}).get("root_prompt", "") or ""
        tail = "\n".join([l for l in rp.splitlines() if l.strip()][-3:])
        return tail[-600:]
    except Exception:
        return ""


def _jsonable(x):
    """JSON-serializable copy (completions hold pydantic messages)."""
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, list):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    for attr in ("model_dump", "dict"):
        m = getattr(x, attr, None)
        if callable(m):
            try:
                return _jsonable(m())
            except Exception:
                break
    return str(x)


def cfg_hash(task, env_args, num_examples, rpe, maxtok, temp) -> str:
    """Short hash of the settings that affect a score."""
    payload = json.dumps(
        {"task": task, "env_args": env_args, "num_examples": num_examples,
         "rpe": rpe, "maxtok": maxtok, "temp": temp},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def build_keys(inputs, rpe: int):
    """[(sample_key, input)] in run order."""
    keyed = []
    seen: dict = defaultdict(int)
    for inp in inputs:
        eid = _g(inp, "example_id")
        occ = seen[eid]
        seen[eid] += 1
        key = str(eid) if rpe == 1 else f"{eid}#{occ}"
        keyed.append((key, inp))
    return keyed


def load_cache(path: str) -> dict:
    """{sample_key: record}; later lines win."""
    cache: dict = {}
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "key" in rec:
                    cache[rec["key"]] = rec
    return cache


def is_done(rec: dict | None) -> bool:
    """Done = scored without error."""
    return bool(rec) and not rec.get("error")


def summarize(keyed, cache: dict, step, task, model, env_args) -> dict:
    """Summary row for the step."""
    recs = [cache.get(k) for k, _ in keyed]
    rewards = [(r.get("reward") if r and r.get("reward") is not None and not r.get("error") else 0.0)
               for r in recs]
    n_trunc = sum(1 for r in recs if r and r.get("is_truncated"))
    n_overlong = sum(1 for r in recs if r and r.get("stop_condition") == "prompt_too_long")
    n_err = sum(1 for r in recs if (not r) or r.get("error"))

    metric_means: dict = {}
    for r in recs:
        for k, v in ((r or {}).get("metrics") or {}).items():
            metric_means.setdefault(k, [])
            if v is not None:
                metric_means[k].append(v)
    metric_means = {k: (sum(v) / len(v) if v else 0.0) for k, v in metric_means.items()}

    return {
        "step": step,
        "task": task,
        "model": model,
        "n": len(rewards),
        "reward": statistics.mean(rewards) if rewards else 0.0,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "n_truncated": n_trunc,
        "n_prompt_too_long": n_overlong,
        "n_error": n_err,
        "metrics": metric_means,
        "args": env_args,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def _prime_headers(base_url: str) -> dict:
    """X-Prime-Team-ID header for Prime Inference, if PRIME_API_TEAM is set."""
    team = os.environ.get("PRIME_API_TEAM")
    if team and "pinference" in (base_url or ""):
        return {"X-Prime-Team-ID": team}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="singleturn-baseline", help="verifiers env id to load")
    ap.add_argument("--task", default=None, help="singleturn-baseline task name (required for that env id)")
    ap.add_argument("--args-json", default="{}", help="long-variant env args (JSON)")
    ap.add_argument("--num-examples", type=int, default=50)
    ap.add_argument("--rollouts-per-example", type=int, default=1)
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--max-completion-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="negative = send none (the server's default: the model's generation_config)")
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--step", type=int, default=-1)
    ap.add_argument("--label", default=None, help="name of this slice in the curve file and cache key")
    ap.add_argument("--out", required=True, help="JSONL curve file to append to")
    ap.add_argument("--cache-dir", default=None,
                    help="per-rollout cache dir (default <out_dir>/rollout_cache)")
    ap.add_argument("--request-timeout", type=float, default=900.0, help="per-request timeout in seconds")
    ap.add_argument("--chunk-size", type=int, default=0,
                    help="run the rollouts in chunks of this size, flushing the cache after each (0 = one chunk)")
    ap.add_argument("--shard", default=None,
                    help="I/N: evaluate only every N-th input starting at I (give each shard its own --label)")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore any existing per-rollout cache (still writes one)")
    a = ap.parse_args()

    import verifiers as vf
    from openai import AsyncOpenAI
    from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient

    env_args = json.loads(a.args_json)
    env_args.pop("num_examples", None)
    if a.env_id == "singleturn-baseline":
        if not a.task:
            ap.error("--task is required for env-id=singleturn-baseline")
        env = vf.load_environment(
            a.env_id, task=a.task, num_examples=a.num_examples, **env_args
        )
    else:
        env = vf.load_environment(a.env_id, num_examples=a.num_examples, **env_args)

    inputs = env._get_eval_inputs(a.num_examples, a.rollouts_per_example)
    keyed = build_keys(inputs, a.rollouts_per_example)
    if a.shard:
        _si, _sn = (int(x) for x in a.shard.split("/"))
        keyed = keyed[_si::_sn]
        print(f"[shard] {a.shard}: {len(keyed)} rollouts", flush=True)

    label = a.label or (a.task if a.env_id == "singleturn-baseline" else a.env_id)
    h = cfg_hash(label, env_args, a.num_examples, a.rollouts_per_example,
                 a.max_completion_tokens, a.temperature)
    cache_dir = a.cache_dir or os.path.join(os.path.dirname(a.out) or ".", "rollout_cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", a.model)
    cache_path = os.path.join(cache_dir, f"{safe_model}__{h}.jsonl")

    cache = {} if a.no_cache else load_cache(cache_path)
    todo = [(k, inp) for k, inp in keyed if not is_done(cache.get(k))]
    print(f"[cache] {cache_path}: {len(keyed) - len(todo)}/{len(keyed)} cached, "
          f"{len(todo)} to run (model={a.model} step={a.step})")

    CHUNK = a.chunk_size if a.chunk_size and a.chunk_size > 0 else (len(todo) or 1)
    chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
    if len(chunks) > 1:
        print(f"[cache] {len(todo)} rollouts in {len(chunks)} chunks of {CHUNK}",
              flush=True)
    for _ci, chunk in enumerate(chunks, 1):
        print(f"[cache] chunk {_ci}/{len(chunks)} ({len(chunk)} rollouts)", flush=True)
        client = OpenAIChatCompletionsClient(
            AsyncOpenAI(base_url=a.base_url, api_key=a.api_key,
                        timeout=a.request_timeout, max_retries=2,
                        default_headers=_prime_headers(a.base_url))
        )
        sampling_args = {
            "max_tokens": a.max_completion_tokens,
            "extra_body": {"enable_thinking": False},
        }
        if a.temperature >= 0:
            sampling_args["temperature"] = a.temperature
        res = env.generate_sync(
            [inp for _, inp in chunk],
            client=client,
            model=a.model,
            sampling_args=sampling_args,
            max_concurrent=a.max_concurrent,
        )
        outs = res["outputs"] if isinstance(res, dict) else res.outputs

        outs_by_eid: dict = defaultdict(deque)
        for o in outs:
            outs_by_eid[_g(o, "example_id")].append(o)
        with open(cache_path, "a") as cf:
            for k, inp in chunk:
                eid = _g(inp, "example_id")
                if not outs_by_eid[eid]:
                    print(f"[cache] WARN no output for key={k} (eid={eid}); leaving uncached")
                    continue
                o = outs_by_eid[eid].popleft()
                rec = {
                    "key": k,
                    "example_id": eid,
                    "step": a.step,
                    "model": a.model,
                    "cfg": h,
                    "reward": _g(o, "reward"),
                    "is_truncated": bool(_g(o, "is_truncated", False)),
                    "stop_condition": _g(o, "stop_condition"),
                    "error": _g(o, "error"),
                    "metrics": _g(o, "metrics") or {},
                    "completion": _jsonable(_g(o, "completion")),
                    "answer": _jsonable(_g(o, "answer")),
                    "question": _question_from_info(_g(inp, "info")),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                cf.write(json.dumps(rec) + "\n")
                cache[k] = rec

    rec = summarize(keyed, cache, a.step, label, a.model, env_args)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
