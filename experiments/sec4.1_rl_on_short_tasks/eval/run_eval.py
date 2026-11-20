"""Evaluate one served model on a verifiers env, one cached record per rollout.

    python run_eval.py --env-id longreason-real --args-json '{"split": "128k", "max_iterations": 40}' \
        --num-examples -1 --base-url http://127.0.0.1:18431/v1 --model <served name> \
        --max-completion-tokens 4096 --temperature 0 --chat-template-kwargs '{"enable_thinking": false}' \
        --out <dir>/summary.jsonl --cache-dir <dir>/cache

Each finished rollout is cached right away, so a rerun only runs examples without an error-free record. A summary row
is appended to --out. --max-completion-tokens 0 sends no max_tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import statistics
from collections import defaultdict, deque
from datetime import datetime, timezone

SAVE_SUBCALLS = os.environ.get("RUN_EVAL_SAVE_SUBCALLS") == "1"
ROOT_PROMPT_MAX_CHARS = 20_000


def _g(o, key, default=None):
    """Get a field from a dict or an object."""
    if isinstance(o, dict):
        return o.get(key, default)
    return getattr(o, key, default)


def _info(info) -> dict:
    try:
        meta = json.loads(info) if isinstance(info, str) else info
        return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _question_from_info(info) -> str:
    """The question from a rollout's info, without the context ("" on failure)."""
    try:
        rp = _info(info).get("root_prompt", "") or ""
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
         "rpe": rpe, "maxtok": maxtok, "temp": temp, "window": None},
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
    """{sample_key: newest record} over `path` and its per-writer shards. A record with `error` means not done."""
    cache: dict = {}
    if not path:
        return cache
    base = path[:-len(".jsonl")] if path.endswith(".jsonl") else path
    shards = sorted(pathlib.Path(base).parent.glob(pathlib.Path(base).name + "*.jsonl"))
    stamp: dict = {}
    seq = 0
    for shard in shards:
        try:
            with open(shard) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    k = rec.get("key")
                    if k is None:
                        continue
                    seq += 1
                    cur = (str(rec.get("ts") or ""), seq)
                    if k not in cache or cur >= stamp[k]:
                        cache[k] = rec
                        stamp[k] = cur
        except OSError:
            continue
    return cache


def is_done(rec: dict | None) -> bool:
    """Done = scored without error."""
    return bool(rec) and not rec.get("error")


def summarize(keyed, cache: dict, task, model, env_args) -> dict:
    """Summary row over the requested set (missing or errored rollouts count 0)."""
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env-id", required=True, help="verifiers env id, e.g. longreason-real")
    ap.add_argument("--args-json", default="{}", help="env loader args (JSON)")
    ap.add_argument("--num-examples", type=int, default=50)
    ap.add_argument("--rollouts-per-example", type=int, default=1)
    ap.add_argument("--base-url", default="http://localhost:8011/v1")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--max-completion-tokens", type=int, default=4096, help="root max_tokens (0 = uncapped)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--out", required=True, help="JSONL summary file to append to")
    ap.add_argument("--cache-dir", default=None,
                    help="per-rollout cache dir (default <out_dir>/rollout_cache)")
    ap.add_argument("--request-timeout", type=float, default=900.0,
                    help="Per-HTTP-request timeout in seconds (default 900).")
    ap.add_argument("--ids", default=None,
                    help="comma list of example_ids to run; applied after the deterministic input list is built, "
                         "and part of the cache key")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help="JSON object passed as extra_body.chat_template_kwargs, e.g. '{\"enable_thinking\": false}'. "
                         "The rollout cache does not key on it: use a fresh --cache-dir when changing it.")
    a = ap.parse_args()

    import verifiers as vf
    from openai import AsyncOpenAI
    from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient

    env_args = json.loads(a.args_json)
    env_args.pop("num_examples", None)
    env = vf.load_environment(a.env_id, num_examples=a.num_examples, **env_args)

    inputs = env._get_eval_inputs(a.num_examples, a.rollouts_per_example)
    if a.ids:
        want = {int(x) for x in a.ids.split(",") if x.strip()}
        inputs = [inp for inp in inputs if int(_g(inp, "example_id")) in want]
        got = {int(_g(inp, "example_id")) for inp in inputs}
        if want - got:
            raise SystemExit(f"--ids not found in {a.env_id}: {sorted(want - got)[:10]}")
        env_args = {**env_args, "_ids": sorted(want)}
    keyed = build_keys(inputs, a.rollouts_per_example)

    label = a.env_id
    h = cfg_hash(label, env_args, a.num_examples, a.rollouts_per_example,
                 a.max_completion_tokens, a.temperature)
    cache_dir = a.cache_dir or os.path.join(os.path.dirname(a.out) or ".", "rollout_cache")
    os.makedirs(cache_dir, exist_ok=True)
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", a.model)
    cache_path = os.path.join(cache_dir, f"{safe_model}__{h}.jsonl")
    _writer_id = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    write_path = os.path.join(cache_dir, f"{safe_model}__{h}.{_writer_id}.jsonl")

    cache = load_cache(cache_path)
    todo = [(k, inp) for k, inp in keyed if not is_done(cache.get(k))]
    print(f"[cache] {cache_path}: {len(keyed) - len(todo)}/{len(keyed)} cached, "
          f"{len(todo)} to run (model={a.model})")
    print(f"[cache] writing shard {os.path.basename(write_path)}", flush=True)

    client = OpenAIChatCompletionsClient(
        AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=a.request_timeout, max_retries=2)
    )
    # vLLM ignores the top-level key; chat_template_kwargs is what takes effect
    sampling_args = {
        "max_tokens": a.max_completion_tokens,
        "temperature": a.temperature,
        "extra_body": {"enable_thinking": False},
    }
    if not a.max_completion_tokens:
        sampling_args.pop("max_tokens")
    if a.chat_template_kwargs:
        try:
            ctk = json.loads(a.chat_template_kwargs)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--chat-template-kwargs is not valid JSON: {exc}")
        if not isinstance(ctk, dict):
            raise SystemExit("--chat-template-kwargs must be a JSON object")
        sampling_args["extra_body"]["chat_template_kwargs"] = ctk
        print(f"[cache] chat_template_kwargs={ctk}", flush=True)

    keys_by_eid: dict = defaultdict(deque)
    inp_by_key: dict = {}
    for _k, _inp in todo:
        keys_by_eid[_g(_inp, "example_id")].append(_k)
        inp_by_key[_k] = _inp

    written = 0

    def _persist(_all_outputs, new_outputs, _metadata) -> None:
        """Cache a finished rollout. Never raises: an exception here would abort all rollouts in flight."""
        try:
            _persist_outputs(new_outputs)
        except Exception as exc:  # noqa: BLE001
            print(f"[cache] WARN could not persist rollout(s): {exc!r}",
                  flush=True)

    def _persist_outputs(new_outputs) -> None:
        nonlocal written
        for o in new_outputs:
            eid = _g(o, "example_id")
            if not keys_by_eid.get(eid):
                print(f"[cache] WARN output for unknown example_id={eid}; "
                      "not cached", flush=True)
                continue
            k = keys_by_eid[eid].popleft()
            inp = inp_by_key[k]
            rec = {
                "key": k,
                "example_id": eid,
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
                "root_prompt": str(_info(_g(inp, "info")).get("root_prompt") or "")[:ROOT_PROMPT_MAX_CHARS],
                "final_answer": _g(o, "rlm_final_answer"),
                "token_usage": _jsonable(_g(o, "token_usage")),
                "timing": _jsonable(_g(o, "timing")),
                "sub_prompt_chars": _g(o, "rlm_sub_llm_prompt_chars"),
                "sub_response_chars": _g(o, "rlm_sub_llm_response_chars"),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            if SAVE_SUBCALLS:
                rec["subcalls"] = _jsonable(_g(o, "rlm_sub_llm_log"))
            cf.write(json.dumps(rec) + "\n")
            cf.flush()
            cache[k] = rec
            written += 1

    if todo:
        with open(write_path, "a") as cf:
            env.generate_sync(
                [inp for _, inp in todo],
                client=client,
                model=a.model,
                sampling_args=sampling_args,
                max_concurrent=a.max_concurrent,
                on_progress=[_persist],
                state_columns=["rlm_sub_llm_prompt_chars", "rlm_sub_llm_response_chars", "rlm_final_answer"]
                + (["rlm_sub_llm_log"] if SAVE_SUBCALLS else []),
            )
        missing = [k for k, _ in todo if k not in cache]
        print(f"[cache] persisted {written}/{len(todo)} rollout(s) this run"
              + (f"; {len(missing)} produced no output" if missing else ""),
              flush=True)

    rec = summarize(keyed, cache, label, a.model, env_args)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
