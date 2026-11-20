# loongrl-real

LoongRL / KeyChain-15K long-context multi-hop QA as an `RLMTrainEnv` environment. `variant="plain"` states the
question; `variant="keychain"` hides it at the end of a UUID key-value chain inside the passages.
Reward: upstream `sub_em` (normalised substring containment); `em` / `f1` logged with weight 0.

Data: HF `OldKingMeister/LoongRL-Train-Data` (one parquet per source/variant, downloaded on first use). Train/eval
rows are separated by a stable hash with `holdout_frac`.

Arguments used by the paper (`source="musique"`, `seed=42`, `metric="sub_em"`):
- train: `variant="plain", split="train", holdout_frac=0.1`
- validation: `variant="keychain"` (default `split="train"`, `holdout_frac=0.1`), `num_examples=20`
