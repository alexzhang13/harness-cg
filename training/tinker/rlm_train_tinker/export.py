"""Download a run's Tinker checkpoints as PEFT LoRA adapters.

Written to <run_dir>/run_default/broadcasts/step_N/ like prime-rl, so the offline evaluators can serve them on the
Tinker base model.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def checkpoints(run_dir: Path) -> list[dict]:
    path = run_dir / "checkpoints.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export_run(run_dir: str | Path, steps: list[int] | None = None, force: bool = False) -> list[Path]:
    run_dir = Path(run_dir)
    out_root = run_dir / "run_default" / "broadcasts"
    written = []
    for ck in checkpoints(run_dir):
        step = int(ck["step"])
        if not ck.get("state_path") or (steps is not None and step not in steps):
            continue
        dest = out_root / f"step_{step}"
        if (dest / "adapter_model.safetensors").exists() and not force:
            written.append(dest)
            continue
        with tempfile.TemporaryDirectory(dir=run_dir) as tmp:
            subprocess.run([sys.executable, "-m", "tinker.cli", "checkpoint", "download", ck["state_path"],
                            "--output", tmp], check=True, stdout=subprocess.DEVNULL)
            (src,) = [p for p in Path(tmp).iterdir() if p.is_dir()]
            if not (src / "adapter_model.safetensors").exists():
                raise RuntimeError(f"{ck['state_path']}: no adapter_model.safetensors in the download")
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dest)
        logger.info("step %d -> %s", step, dest)
        written.append(dest)
    return written
