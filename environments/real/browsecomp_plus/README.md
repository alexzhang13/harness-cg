# browsecomp-plus-real

BrowseComp-Plus (`Tevatron/browsecomp-plus`: 830 encrypted queries; `Tevatron/browsecomp-plus-corpus`: 100,195
documents; both auto-downloaded from the HF Hub) as an `RLMTrainEnv`. Reward = part-wise substring match of the gold.

- Splits (query-id lists in `browsecomp_plus_real/data/`): `"test"` (150), `"train"` (`train.txt` + `dev2.txt`, 680),
  `"all"` (830), `"ids:<path>"` (one query id per line).
- `docs_per_query="all"` serves the whole corpus as a doc-store reference, built on first use (about 0.8 GB) under
  `$BCP_DOC_STORE_ROOT` (default `~/.cache/rlm_doc_stores`); otherwise gold/evidence docs padded with random docs.
- Held-out eval: `split="all", docs_per_query="all", num_examples=-1, orchestrator_prompt=true, min_subcall=0, max_iterations=40,
  sub_max_tokens=65536, answer_judge=false`. `answer_judge=true` is the prompt used for upgrade-only regrading.
- RL validation: `split="train", docs_per_query="all", require_answerable=true, num_examples=25, seed=7`.
- Broad-mixture RL: train `split="train", docs_per_query=50, num_examples=500, seed=42`; validation
  `split="test", docs_per_query=200, num_examples=20, seed=42`.
