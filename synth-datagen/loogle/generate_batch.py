"""Run one batch recipe: generation, quality review, and the SFT bundle (resumable).

    python generate_batch.py --config recipes/expand_batch.json --out <batch dir>

The published sets use <out>/sft_sampled.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]


async def main(a):
    config = json.loads(a.config.read_text())
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    frozen = out / "batch_config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        raise ValueError("batch config changed; use a fresh output directory")
    frozen.write_text(json.dumps(config, indent=2))
    jobs = config["jobs"]
    if len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job names")
    results = {}

    async def command(name, phase, argv):
        print(json.dumps({"job": name, "phase": phase, "status": "started"}), flush=True)
        with (out / "logs" / f"{name}_{phase}.log").open("a") as log:
            process = await asyncio.create_subprocess_exec(sys.executable, *map(str, argv), cwd=ROOT,
                                                          stdout=log, stderr=asyncio.subprocess.STDOUT)
            rc = await process.wait()
        print(json.dumps({"job": name, "phase": phase, "returncode": rc}), flush=True)
        return rc

    async def worker(endpoint, assigned):
        for job in assigned:
            argv = [BASE / "synthetic_strategy_lab.py", "--base-url", endpoint,
                    "--model", config["model"], "--root-mode", job["root_mode"],
                    "--family", job["family"], "--n", job["n"], "--start", job["start"],
                    "--env-config", BASE / "recipes" / (job["profile"] + ".json"),
                    "--concurrency", config.get("concurrency_per_endpoint", 4),
                    "--max-tokens", config.get("max_tokens", 8192),
                    "--temperature", config.get("temperature", 0.8),
                    "--timeout", config.get("rollout_timeout", 1200), "--out", out / job["name"]]
            if job.get("note"):
                argv += ["--note", BASE / "notes" / job["note"]]
            results[job["name"]] = {"generation_returncode": await command(job["name"], "generate", argv)}

    endpoints = config["endpoints"]
    await asyncio.gather(*(worker(endpoint, jobs[i::len(endpoints)]) for i, endpoint in enumerate(endpoints)))
    review_sem = asyncio.Semaphore(2)

    async def review(job):
        name = job["name"]
        if results[name]["generation_returncode"]:
            return
        async with review_sem:
            results[name]["review_returncode"] = await command(name, "review", [
                BASE / "review_candidates.py", out / name, "--concurrency", 4,
                "--judge-model", config.get("judge_model", "openai/gpt-5.4-mini"),
                "--max-turns", job.get("review_max_turns", 6)])
        results[name]["summary"] = json.loads((out / name / "summary.json").read_text())
        reviewed = out / name / "sft_reviewed.jsonl"
        results[name]["reviewed_accepted"] = len(reviewed.read_text().splitlines()) if reviewed.exists() else 0

    await asyncio.gather(*(review(job) for job in jobs))
    (out / "batch_results.json").write_text(json.dumps(results, indent=2))
    codes = []
    for teacher in ("all", "sampled", "compiled"):
        target = out / ("sft" if teacher == "all" else f"sft_{teacher}")
        codes.append(await command("bundle_" + teacher, "export", [BASE / "bundle_reviewed.py", out,
                                    "--out", target, "--root-teacher", teacher]))
    if any(codes) or any(r["generation_returncode"] or r.get("review_returncode", 1) for r in results.values()):
        raise RuntimeError("one or more batch phases failed; inspect batch_results.json and logs")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    asyncio.run(main(p.parse_args()))
