"""Evaluate saved Pi LoRA adapters on a task's long eval split (Pi curves of Fig. length-gen; see pi_runs/EVAL.md).

Adapters are loaded on a running vLLM server as step_N (`base` = no adapter). Work is split into units of
--unit-size examples that several servers claim through <out>/claims/; results go to <out>/results/<ckpt>/.
Needs node/pi on PATH and the PI_* variables (pi_runs/slurm/eval_server.sbatch sets both).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import verifiers as vf
from verifiers.types import ClientConfig

ENV_ID = "pi-mrcr-v2-v1"
TASKS = {
    "mrcr": {"id": "pi-mrcr-v2-v1", "n": 50,
             "args": {"needle_count": 8, "context_range": "2m-4m", "seed": 42, "max_turns": 30}},
    "longbenchpro": {"id": "pi-longbenchpro-v1", "n": 115,
                     "args": {"max_turns": 30, "language": "English", "token_length": "256k", "thinking": False,
                              "num_examples": -1, "seed": 42}},
    # as in pi_runs/eval_specs/<task>.toml
    "adaleval": {"id": "pi-ada-leval-v1", "n": 50,
                 "args": {"max_turns": 30, "task": "bestanswer", "setting": "128k", "num_examples": 50,
                          "manifest_split": "eval"}},
    "oolong-pairs": {"id": "pi-oolong-pairs-v1", "n": 20,
                     "args": {"max_turns": 30, "context_length": 32768, "num_examples": 20, "seed": 42}},
}
ENV_INFO: dict = {}
SAMPLING = {
    "temperature": 0.7,
    "max_tokens": 4096,
    "extra_body": {"enable_thinking": False, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
}


def job_alive(jobid: str) -> bool:
    jobid = jobid.split(":")[0]
    if not jobid or jobid == "local":
        return True
    try:
        out = subprocess.run(
            ["squeue", "-h", "-j", jobid, "-o", "%T"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
        return bool(out) and out.split()[0] in {"RUNNING", "PENDING", "COMPLETING", "CONFIGURING"}
    except Exception:
        return True


def try_claim(claims: Path, unit: str, me: str, driver_id: str = "") -> bool:
    d = claims / unit
    try:
        d.mkdir(parents=True)
        (d / "owner").write_text(me)
        if driver_id:
            (d / "driver").write_text(driver_id)
        return True
    except FileExistsError:
        try:
            owner = (d / "owner").read_text().strip()
        except Exception:
            return False
        try:
            drv = (d / "driver").read_text().strip()
        except Exception:
            drv = ""
        if driver_id and drv:
            if drv == driver_id:
                return True
        elif owner == me:
            return True
        if not job_alive(owner):
            (d / "owner").write_text(me)
            if driver_id:
                (d / "driver").write_text(driver_id)
            return True
        return False


def http_json(url: str, payload: dict | None = None, timeout: int = 120):
    req = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode()
    return json.loads(body) if body.strip().startswith(("{", "[")) else body


def ensure_adapter(base_url: str, name: str, path: str) -> None:
    root = base_url.rsplit("/v1", 1)[0]
    ids = {m["id"] for m in http_json(f"{base_url}/models")["data"]}
    if name in ids:
        return
    http_json(f"{root}/v1/load_lora_adapter", {"lora_name": name, "lora_path": path})
    print(f"[eval] loaded adapter {name} <- {path}", flush=True)


def discover(lora_root: Path, steps: list[int] | None, watch_every: int) -> list[str]:
    names: list[str] = []
    if steps is not None:
        names += [f"step_{s}" for s in steps]
    else:
        found = []
        for p in lora_root.glob("step_*"):
            try:
                n = int(p.name.split("_")[1])
            except ValueError:
                continue
            if n % watch_every == 0 and (p / "adapter_model.safetensors").exists():
                found.append(n)
        names += [f"step_{n}" for n in sorted(found)]
    return names


def _strip_tokens(obj):
    """Replace token id / logprob arrays by their length."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, list) and len(v) > 8 and all(isinstance(x, (int, float)) for x in v[:8]):
                out[k + "_len"] = len(v)
            else:
                out[k] = _strip_tokens(v)
        return out
    if isinstance(obj, list):
        return [_strip_tokens(x) for x in obj]
    return obj


def _mkdir(d: Path) -> None:
    """mkdir -p that tolerates a concurrent create (NFS)."""
    for _ in range(5):
        try:
            d.mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            if d.is_dir():
                return
            time.sleep(1)
    d.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: list) -> None:
    import gzip

    _mkdir(path.parent)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    opener = (lambda p: gzip.open(p, "wt")) if path.name.endswith(".gz") else (lambda p: open(p, "w"))
    with opener(tmp) as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    tmp.rename(path)


async def run_unit(env, inputs, args, ckpt: str, shard: int, res_dir: Path) -> None:
    lo, hi = shard * args.unit_size, min((shard + 1) * args.unit_size, len(inputs))
    sub = inputs[lo:hi]
    client = ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=args.base_url,
        api_key_var="OPENAI_API_KEY",
    )
    t0 = time.time()
    sem = asyncio.Semaphore(args.max_concurrent)

    async def one(inp):
        async with sem:
            try:
                o = await env.generate(
                    [inp],
                    client=client,
                    model=ckpt,
                    sampling_args=SAMPLING,
                    max_concurrent=1,
                    max_retries=2,
                    state_columns=["trajectory", "command"],
                )
                return dict(o["outputs"][0])
            except Exception as e:
                print(f"[eval] rollout error {ckpt} ex={inp.get('example_id')}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                return {
                    "example_id": inp.get("example_id"),
                    "prompt": inp.get("prompt"),
                    "completion": None,
                    "reward": 0.0,
                    "error": {"error": type(e).__name__, "error_chain_str": str(e)[:2000], "driver_caught": True},
                    "info": inp.get("info"),
                    "answer": inp.get("answer"),
                    "is_completed": False,
                    "is_truncated": False,
                    "stop_condition": "driver_exception",
                    "metrics": {},
                }

    raw = list(await asyncio.gather(*[one(i) for i in sub]))
    _mkdir(res_dir)

    rows, side = [], []
    for r in raw:
        traj = r.pop("trajectory", None)
        cmd = r.pop("command", None)
        rows.append({"ckpt": ckpt, "shard": shard, **r})
        side.append(
            {
                "ckpt": ckpt,
                "shard": shard,
                "example_id": r.get("example_id"),
                "trajectory": _strip_tokens(traj),
                "command": cmd,
            }
        )
    _write_jsonl(res_dir / f"shard_{shard}.jsonl", rows)
    side_dir = res_dir.parent.parent / "sidecar" / ckpt
    _mkdir(side_dir)
    _write_jsonl(side_dir / f"shard_{shard}.jsonl.gz", side)
    manifest = {
        "ckpt": ckpt,
        "shard": shard,
        "example_range": [lo, hi],
        "n": len(rows),
        "base_url": args.base_url,
        "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "finished_at": time.strftime("%F %T"),
        "wall_seconds": round(time.time() - t0),
        "sampling_args": SAMPLING,
        "env": ENV_INFO,
        "pi_env": {k: v for k, v in os.environ.items() if k.startswith("PI_") or k.startswith("VF_")},
    }
    (side_dir / f"shard_{shard}.manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    rew = [float(r.get("reward") or 0.0) for r in rows]
    errs = sum(1 for r in rows if r.get("error"))
    print(
        f"[eval] DONE {ckpt} shard {shard} n={len(rows)} mean={sum(rew)/max(1,len(rew)):.3f} "
        f"errors={errs} {time.time()-t0:.0f}s",
        flush=True,
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="http://host:port/v1 of the vLLM server")
    ap.add_argument("--lora-root", required=True, help="dir with step_N/ adapter copies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="", help="comma list of steps; empty = auto-scan multiples of 10")
    ap.add_argument("--watch", action="store_true", help="keep rescanning for new adapters")
    ap.add_argument("--exit-after-idle", type=int, default=0, help="seconds idle before exit when --watch (0 = never)")
    ap.add_argument("--unit-size", type=int, default=10)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--task", default="mrcr", choices=sorted(TASKS), help="which true-eval env to run (default mrcr)")
    ap.add_argument("--n-examples", type=int, default=None, help="default: the task's original eval size")
    ap.add_argument("--smoke", action="store_true", help="tiny: 2-needle 32k-64k (the training split), 1 example, no claims")
    ap.add_argument("--smoke-ckpt", default="step_10")
    ap.add_argument("--only", default="", help="run exactly these units, e.g. step_40:1,step_70:3 (ignores other drivers' claims; exits when done)")
    args = ap.parse_args()

    out = Path(args.out)
    claims, results = out / "claims", out / "results"
    claims.mkdir(parents=True, exist_ok=True)
    me = os.environ.get("SLURM_JOB_ID", "local")
    me = os.environ.get("EVAL_OWNER") or me
    driver_id = f"{me}:{os.uname().nodename}:{os.getpid()}"
    steps = [int(s) for s in args.steps.split(",") if s.strip()] or None
    lora_root = Path(args.lora_root)
    os.environ.setdefault("OPENAI_API_KEY", "x")

    task = TASKS[args.task]
    # per-task output budget (settings.EVAL_MAX_TOKENS)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "pi_runs"))
    import settings as _settings

    SAMPLING["max_tokens"] = _settings.EVAL_MAX_TOKENS.get(args.task, SAMPLING["max_tokens"])
    if args.n_examples is None:
        args.n_examples = task["n"]
    env_args = dict(task["args"])
    if args.task == "mrcr":
        env_args["num_examples"] = args.n_examples
    if args.smoke:
        assert args.task == "mrcr", "--smoke is only defined for mrcr"
        env = vf.load_environment(ENV_ID, needle_count=2, context_range="32k-64k", num_examples=1, seed=42, max_turns=30)
        ENV_INFO.update({"id": ENV_ID, "smoke": True})
    else:
        env = vf.load_environment(task["id"], **env_args)
        ENV_INFO.update({"id": task["id"], "task": args.task, "n_examples": args.n_examples, **env_args})
    inputs = env._get_eval_inputs(args.n_examples if not args.smoke else 1, 1)
    inputs = list(inputs)
    n_shards = (len(inputs) + args.unit_size - 1) // args.unit_size
    print(f"[eval] env loaded: {len(inputs)} examples -> {n_shards} shards/ckpt; job={me}", flush=True)

    if args.only:
        for tok in args.only.split(","):
            ckpt, sh = tok.strip().split(":")
            shard = int(sh)
            if ckpt != "base":
                ensure_adapter(args.base_url, ckpt, str(lora_root / ckpt))
            print(f"[eval] START (only) {ckpt} shard {shard}", flush=True)
            await run_unit(env, inputs, args, ckpt, shard, results / ckpt)
        return

    idle_since = time.time()
    while True:
        did = False
        for ckpt in ([args.smoke_ckpt] if args.smoke else discover(lora_root, steps, 10)):
            for shard in range(n_shards):
                if (results / ckpt / f"shard_{shard}.jsonl").exists():
                    continue
                if not args.smoke and not try_claim(claims, f"{ckpt}__{shard}", me, driver_id):
                    continue
                if ckpt != "base":
                    ensure_adapter(args.base_url, ckpt, str(lora_root / ckpt))
                print(f"[eval] START {ckpt} shard {shard}", flush=True)
                try:
                    await run_unit(env, inputs, args, ckpt, shard, results / ckpt)
                except Exception as e:
                    print(f"[eval] FAIL {ckpt} shard {shard}: {type(e).__name__}: {e}", flush=True)
                    d = claims / f"{ckpt}__{shard}"
                    for f in d.glob("*"):
                        f.unlink()
                    d.rmdir()
                did = True
                idle_since = time.time()
                if args.smoke:
                    return
                break
            if did:
                break
        if not did:
            if not args.watch:
                print("[eval] no more units", flush=True)
                return
            if args.exit_after_idle and time.time() - idle_since > args.exit_after_idle:
                print("[eval] idle -> exit", flush=True)
                return
            await asyncio.sleep(120)


if __name__ == "__main__":
    asyncio.run(main())
