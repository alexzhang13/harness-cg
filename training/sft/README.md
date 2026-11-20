# Supervised fine-tuning on RLM traces

Turns whole RLM trajectories into per-turn training rows and trains them with prime-rl's `sft` entry point.

| File | Purpose |
|---|---|
| `build_source_rows.py` | synthetic-trace dataset snapshot -> whole trajectories `{messages, meta}` |
| `build_turn_mix.py` | whole trajectories -> one row per trained assistant turn (history as context) |
| `patches/` | loaded through `PYTHONPATH`: no-split sequence packing, loss on the last assistant turn only |
| `run_sft.sh` | launch `sft @ <config>` with the patches |
| `fix_checkpoint.py` | make a saved Qwen3.8-27B checkpoint loadable by vLLM |
| `qwen3.8_nothink_chat_template.jinja` | chat template used for tokenization and training (thinking off unless requested) |

The recipe for RLM-27B is in `experiments/sec4.2_synthetic_environments/sft/`.
