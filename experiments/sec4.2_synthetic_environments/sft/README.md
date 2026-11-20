# RLM-27B: SFT on synthetic-environment traces (§4.2, Table main-composition "RLM-27B (synthetic)")

Traces: https://huggingface.co/datasets/a1zhang/synthetic-rlm-data-clean (8,015 root trajectories and 1,465
LongReason sub-call rows, hints already stripped; one config per benchmark). How they were generated:
`synth-datagen/`. Trained model: https://huggingface.co/a1zhang/RLM-27B.

```bash
# 1. dataset snapshot -> whole-trajectory rows (7,520 rows: 6,933 root + 587 sub-call)
huggingface-cli download a1zhang/synthetic-rlm-data-clean --repo-type dataset --local-dir $SNAP
python training/sft/build_source_rows.py $SNAP $SFT_DATA_DIR/source.jsonl \
  --exclude-descriptor corpusqa-synth-survey-v2 --exclude-descriptor longreason-expanded-star-v3 \
  --exclude-descriptor trick_synth-v6 --exclude-descriptor loogle-synth-targeted-backtrack \
  --exclude-descriptor loogle-synth-targeted-deadline --selected-fastest-only best-of-3

# 2. per-turn rows (44,827 samples), tokenized with the Qwen3.8-27B no-think template
SFT_TOK_MODEL=Qwen/Qwen3.8-27B python training/sft/build_turn_mix.py --input $SFT_DATA_DIR/source.jsonl \
  --out $SFT_DATA_DIR/synth_full_clean_v2_turns --turns-per-traj 100 --batch 18 --reverify --reasoning keep

# 3. train: full fine-tuning, fp32 master weights, lr 5e-6, batch 8 x 32768 tokens, 1 epoch = 1074 updates, 8 GPUs
SFT_DATA_DIR=$SFT_DATA_DIR bash training/sft/run_sft.sh experiments/sec4.2_synthetic_environments/sft/configs/rlm-27b.toml

# 4. make the final checkpoint servable
python training/sft/fix_checkpoint.py --base Qwen/Qwen3.8-27B $PRIME_RL_DIR/outputs/sft/rlm-27b/weights/step_1074
```

RLM-27B is trained on the subset these steps build from the full dataset. Steps 1 and 2 need
prime-rl's Python environment (step 2 imports its tokenization helpers).
Evaluate with `experiments/sec4.1_rl_on_short_tasks/eval/`.
