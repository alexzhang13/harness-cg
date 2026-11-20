"""Live check of the Pi limits on a vLLM server: three rollouts told never to stop (loop on bash, spawn sub-agents,
sub-agent loops on bash) must stop at the turn and sub-agent limits. Exits 1 if a limit is not enforced.

    python verify_caps_live.py --base-url http://localhost:8000/v1 --model base --out DIR   # PI_PROMPT_VARIANT unset
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import verifiers as vf
from verifiers.types import ClientConfig

import settings as S
from rlm_train.pi import make_pi_env

SAMPLING = {"temperature": 0.7, "max_tokens": 4096, "extra_body": {"enable_thinking": False, "top_p": 0.8, "top_k": 20, "min_p": 0.0}}

SCENARIOS = {
    "root-turns": (
        "This is a tool-use probe; ignore the data file. Call the `bash` tool with the command `echo ping`, exactly one "
        "call per turn, and keep doing that. Never give a final answer and never stop; you will be stopped from outside."
    ),
    "subagent-calls": (
        "This is a tool-use probe; ignore the data file. Call the `subagent` tool in PARALLEL mode with a `tasks` array "
        "of 8 items, each {\"agent\": \"worker\", \"task\": \"Reply with the single word OK.\"}. Do this again on every "
        "turn, with no pause. Never give a final answer and never stop; you will be stopped from outside."
    ),
    "child-turns": (
        "This is a tool-use probe; ignore the data file. Call the `subagent` tool exactly once with agent `worker` and the "
        "task: 'Call the bash tool with the command `echo x`, one call per turn, and keep going. Never give a final "
        "answer and never stop.' Then tell me what the helper returned."
    ),
}


def run_one(name: str, question: str, args) -> dict:
    row = {
        "example_id": 0,
        "prompt": [{"role": "user", "content": question}],
        "answer": "probe",
        "info": json.dumps({"context": "probe context: nothing to read.\n", "root_prompt": ""}),
    }
    env = make_pi_env(build_rows=[row], score_fn=lambda output, gold, meta: 0.0, answer_format="any text",
                      reward_name="probe", harness_kwargs={"max_turns": 30})
    inputs = list(env._get_eval_inputs(1, 1))
    client = ClientConfig(client_type="openai_chat_completions", api_base_url=args.base_url, api_key_var="OPENAI_API_KEY")

    async def go():
        o = await env.generate(inputs, client=client, model=args.model, sampling_args=SAMPLING, max_concurrent=1,
                               max_retries=0, state_columns=["trajectory", "command"])
        return dict(o["outputs"][0])

    t0, err = time.time(), None
    try:
        r = asyncio.run(go())
    except Exception as e:  # noqa: BLE001
        err, r = f"{type(e).__name__}: {str(e)[:400]}", {}
    traj = r.get("trajectory") or []
    sub = [s for s in traj if any("subagent" in k.lower() for k in ((s.get("extras") or {}).get("headers") or {}))]
    cmd = r.get("command") or {}
    stdout = cmd.get("stdout") or ""
    return {
        "scenario": name, "wall_s": round(time.time() - t0), "driver_error": err, "rollout_error": r.get("error"),
        "reward": r.get("reward"), "returncode": cmd.get("returncode"),
        "root_steps": len(traj) - len(sub), "subagent_steps": len(sub),
        "sentinel": "__PI_TURN_CAP_HIT__" in stdout,
        # the refusal is a tool result, so it shows up in later prompts
        "budget_refusals": sum(1 for s in traj if "Sub-agent budget exhausted" in json.dumps(s.get("prompt"), default=str)),
        "_stdout": stdout,
    }


def check(res: dict, turns: int, calls: int, child_turns: int) -> list[str]:
    bad = []
    if res["driver_error"] or res["rollout_error"]:
        bad.append(f"rollout errored: {res['driver_error'] or res['rollout_error']}")
    if res["root_steps"] > turns:
        bad.append(f"root made {res['root_steps']} model calls > {turns}")
    n = res["scenario"]
    if n == "root-turns":
        if res["root_steps"] < turns:
            bad.append(f"cap not reached ({res['root_steps']} < {turns}): the model stopped by itself, probe inconclusive")
        if not res["sentinel"]:
            bad.append("no cap sentinel in pi output")
        if res["returncode"] not in (0, "0"):
            bad.append(f"pi exited with {res['returncode']}, not 0")
    elif n == "subagent-calls":
        if res["subagent_steps"] > calls * child_turns:
            bad.append(f"{res['subagent_steps']} sub-agent steps > {calls} x {child_turns}")
        if not res["budget_refusals"]:
            bad.append("budget never refused a call: probe inconclusive or budget not enforced")
    elif n == "child-turns":
        if res["subagent_steps"] > child_turns:
            bad.append(f"sub-agent made {res['subagent_steps']} model calls > {child_turns}")
        if res["subagent_steps"] < child_turns:
            bad.append(f"sub-agent stopped at {res['subagent_steps']} < {child_turns}: probe inconclusive")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.environ.setdefault("OPENAI_API_KEY", "x")
    turns = int(os.environ.get("PI_MAX_TURNS", S.PI_ENV["PI_MAX_TURNS"]))
    calls = int(os.environ.get("PI_SUBAGENT_MAX_CALLS", S.PI_ENV["PI_SUBAGENT_MAX_CALLS"]))
    child = int(os.environ.get("PI_SUBAGENT_MAX_TURNS", S.PI_ENV["PI_SUBAGENT_MAX_TURNS"]))
    print(f"limits: root turns {turns}, sub-agent calls {calls}, turns per sub-agent {child}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, q in SCENARIOS.items():
        if args.only and name != args.only:
            continue
        res = run_one(name, q, args)
        (out / f"{name}.pi_stdout.txt").write_text(res.pop("_stdout"))
        problems = check(res, turns, calls, child)
        res["problems"] = problems
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
        print(("FAIL " if problems else "PASS ") + json.dumps(res, default=str), flush=True)
        failed |= bool(problems)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
