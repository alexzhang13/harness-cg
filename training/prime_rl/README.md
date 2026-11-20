# prime-rl backend

The trainer every paper run used. Details (pinned commit, what each patch does, shared RL settings, adapters):
`../README.md`.

| Path | Purpose |
|---|---|
| `install.sh` | clone prime-rl at the pinned commit, apply `patches/`, install this repo and every env package (`PRIME_RL_SFT_GATED_DIR=...` also builds the gated-SFT checkout) |
| `launch.sh` | run one experiment config (`<config.toml>`): substitutes `${REPO}` etc., sources `$PRIME_RL_DIR/.env`, resumes from the latest checkpoint |
| `patches/` | prime-rl and verifiers patches (`sft_gated/` for the section 4.3 SFT runs) |
| `seed/` | `sitecustomize.py` that seeds torch in every prime-rl process (`run_train.sh` `SEED`) |
| `offline_eval/` | long-split evaluation of saved LoRA adapters with a vLLM server we control (YaRN windows, temperature 0) |
| `tools/fix_export_names.py` | rename full-fine-tuning weight exports so vLLM can load them |
