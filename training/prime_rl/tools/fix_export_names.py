"""Make a prime-rl full fine-tuning export of Qwen3.8-27B loadable by transformers / vLLM.

    python fix_export_names.py <run>/weights/step_N <base model snapshot dir> <out dir>

Renames the nested tensor names to the HF layout and copies the tokenizer and processor files. Prints the change of
one tensor vs the base model, to check the export is not the base weights.
"""
import json, re, shutil, sys
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, base, out = (Path(p) for p in sys.argv[1:4])
out.mkdir(parents=True, exist_ok=True)

def norm(n: str) -> str:
    n = re.sub(r"^(model\.language_model\.)(language_model\.)+", r"\1", n)
    return n.replace("model.language_model.visual.", "model.visual.")

idx = json.load(open(src / "model.safetensors.index.json"))
new_map, sample = {}, None
for shard in sorted(set(idx["weight_map"].values())):
    tensors = {}
    with safe_open(src / shard, "pt") as f:
        for k in f.keys():
            tensors[norm(k)] = f.get_tensor(k)
    save_file(tensors, out / shard, metadata={"format": "pt"})
    for k in tensors:
        new_map[k] = shard
    if sample is None:
        cand = [k for k in tensors if "mlp.down_proj.weight" in k]
        sample = (cand[0], shard) if cand else None
    print("wrote", shard, len(tensors), flush=True)
json.dump({"metadata": idx.get("metadata", {}), "weight_map": new_map}, open(out / "model.safetensors.index.json", "w"), indent=1)
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    shutil.copy(src / f, out / f)
for f in ("preprocessor_config.json", "video_preprocessor_config.json", "vocab.json", "merges.txt"):
    shutil.copy(base / f, out / f)
bidx = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
if sample is None:
    sys.exit("no sample tensor for the sanity check")
k, shard = sample
with safe_open(out / shard, "pt") as f:
    a = f.get_tensor(k).float()
with safe_open(base / bidx[k], "pt") as f:
    b = f.get_tensor(k).float()
print(f"sanity {k}: |trained-base|/|base| = {((a-b).norm()/b.norm()).item():.2e} (0 would mean the export is just the base)")
missing = sorted(set(bidx) - set(new_map))
print("tensors", len(new_map), "base", len(bidx), "missing vs base:", len(missing), missing[:3])
