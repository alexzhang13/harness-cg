"""Bundle the reviewed rows of every run dir under <batch dir> into <out>/train.jsonl (deduplicated).

    python bundle_reviewed.py <batch dir> --out <batch dir>/sft_sampled --root-teacher sampled
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "environments/synthetic/loogle_v2"
sys.path[:0] = [str(ROOT / "src"), str(BASE), str(ROOT / "environments/synthetic/synth_common"), str(Path(__file__).resolve().parent)]
from loogle_datagen.quality import (  # noqa: E402
    prepare_reasoning_export,
    quality_passes,
    require_rollout_models,
)
from loogle_v2_synth.strategy import load_strategy_environment  # noqa: E402


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main(a):
    if (a.root / "EXCLUDED_FROM_SFT.json").exists():
        raise ValueError("this experiment is explicitly excluded from SFT")
    selected, sources, duplicates = {}, [], []
    for path in sorted(a.root.glob("*/sft_reviewed.jsonl")):
        config = json.loads((path.parent/"config.json").read_text())
        if config.get("data") != "procedural-only" or config.get("split") != "train":
            raise ValueError(f"non-training or non-procedural source: {path}")
        args = config.get("resolved_env_args") or json.loads(config.get("env_args") or "{}")
        args.setdefault("generator_version", config.get("generator_version", 1))
        env = load_strategy_environment(family=config["family"], num_examples=1, **args)
        sources.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text().splitlines():
            sample = json.loads(line)
            sample = prepare_reasoning_export(sample, drop_reasoning=a.drop_reasoning)
            meta = sample["meta"]
            if a.root_teacher != "all" and meta.get("root_teacher", "sampled") != a.root_teacher:
                continue
            if meta.get("source") != "procedural-only" or meta.get("split") != "train":
                raise ValueError("candidate provenance mismatch")
            require_rollout_models(meta, a.require_rollout_model)
            verdict = meta.get("quality_review")
            if not isinstance(verdict, dict) or not quality_passes(verdict,
                    recovery=bool(config.get("recovery_curriculum")), continuation=bool(config.get("replay_prefix")),
                    focus=config.get("focus_behavior", "backtrack")):
                raise ValueError("candidate lacks a passing quality review")
            generated = env._generator({"synth_seed": meta["seed"], "synth_split": "train", "synth_params": "{}"})
            visible = "\n".join(m["content"] for m in sample["prompt"])
            if generated.root_prompt not in visible:
                raise ValueError("reconstructed task does not match the saved root prompt")
            found = re.search(r"Your context is a str of (\d+) total characters", visible)
            expected = meta.get("context_chars") or (int(found[1]) if found else None)
            if expected is None or len(generated.context) != expected:
                raise ValueError(f"cannot reconstruct exact context size for {path}, seed={meta['seed']}")
            context_hash, task_hash = sha(generated.context), sha(generated.root_prompt)
            meta.update(context_sha256=context_hash, task_sha256=task_hash,
                        context_hash_origin="regenerated from recorded seed/knobs; visible size verified",
                        context_chars=len(generated.context), source_run=str(path.parent),
                        generator_version=args["generator_version"])
            if any("<strategy_hint>" in m["content"] or "</strategy_hint>" in m["content"] for m in sample["messages"]):
                raise ValueError("hint leak")
            key = (context_hash, task_hash)
            if key in selected:
                duplicates.append({"context_sha256": context_hash, "task_sha256": task_hash})
            def rank(s):
                m = s["meta"]["metrics"]
                return (m["turns"], m["observation_chars"], s["meta"].get("root_teacher") == "compiled")
            if key not in selected or rank(sample) < rank(selected[key]):
                selected[key] = sample
    rows = list(selected.values())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/"train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    report = {"n": len(rows), "families": dict(Counter(r["meta"]["family"] for r in rows)),
              "root_teachers": dict(Counter(r["meta"].get("root_teacher", "sampled") for r in rows)),
              "root_decoders": dict(Counter(r["meta"].get("root_decoder", "unconstrained") for r in rows)),
              "source_files": sources, "duplicate_tasks": duplicates,
              "root_teacher_filter": a.root_teacher,
              "reasoning_policy": "drop" if a.drop_reasoning else "preserve; reject privileged reasoning",
              "required_rollout_models": a.require_rollout_model,
              "status": "reviewed synthetic training data; no model training or transfer result"}
    (a.out/"manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root-teacher", choices=["all", "sampled", "compiled"], default="all")
    p.add_argument("--drop-reasoning", action="store_true")
    p.add_argument("--require-rollout-model", action="append", default=[])
    main(p.parse_args())
