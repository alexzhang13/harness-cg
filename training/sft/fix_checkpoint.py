"""Make a prime-rl SFT weight checkpoint of Qwen3.8-27B loadable by vLLM (in place, idempotent).

    python fix_checkpoint.py --base Qwen/Qwen3.8-27B <run_dir>/weights/step_N [...]     # or: --run-dir <run_dir>

Fixes the nested tensor names and adds the processor configs; writes <step>/VLLM_READY when done.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

from safetensors.torch import load_file, save_file

RENAMES = [("model.language_model.language_model.language_model.", "model.language_model."),
           ("model.language_model.visual.", "model.visual.")]


def base_dir(name: str) -> Path:
    return Path(glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{name.replace('/', '--')}/snapshots/*/"))[0])


def rename(k: str) -> str:
    if k.endswith("lm_head.weight"):
        return "lm_head.weight"
    for a, b in RENAMES:
        if k.startswith(a):
            return b + k[len(a):]
    return k


def fix(d: Path, base: str) -> None:
    BASE = base_dir(base)
    if (d / "VLLM_READY").exists() or not (d / "STABLE").exists():
        return
    idx_p = d / "model.safetensors.index.json"
    idx = json.loads(idx_p.read_text())
    base_keys = set(json.loads((BASE / "model.safetensors.index.json").read_text())["weight_map"])
    new_map = {rename(k): v for k, v in idx["weight_map"].items()}
    missing = {k for k in base_keys - set(new_map) if not k.startswith("mtp.")}
    extra = set(new_map) - base_keys
    if missing or extra:
        sys.exit(f"[fix] {d}: key mismatch after rename: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}")
    for shard in sorted(set(idx["weight_map"].values())):
        p = d / shard
        t = load_file(str(p))
        if any(rename(k) != k for k in t):
            tmp = p.with_suffix(f".{os.getpid()}.tmp")
            save_file({rename(k): v for k, v in t.items()}, str(tmp), metadata={"format": "pt"})
            os.replace(tmp, p)
    idx["weight_map"] = new_map
    idx_p.write_text(json.dumps(idx, indent=2))
    for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
        if not (d / f).exists():
            shutil.copy(BASE / f, d / f)
    (d / "VLLM_READY").touch()
    print(f"[fix] {time.strftime('%H:%M:%S')} {d} ready ({len(new_map)} tensors)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--run-dir")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--base", default=None, help="HF base model (default: config.json _name_or_path, else Qwen/Qwen3.8-27B)")
    a = ap.parse_args()
    while True:
        dirs = [Path(x) for x in a.dirs] + (sorted(Path(a.run_dir, "weights").glob("step_*")) if a.run_dir else [])
        for d in dirs:
            base = a.base
            if base is None:
                try:
                    base = json.loads((d / "config.json").read_text()).get("_name_or_path") or "Qwen/Qwen3.8-27B"
                except Exception:  # noqa: BLE001
                    base = "Qwen/Qwen3.8-27B"
            fix(d, base)
        if not a.loop:
            break
        time.sleep(20)


if __name__ == "__main__":
    main()
