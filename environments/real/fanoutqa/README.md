# fanoutqa-real

FanOutQA (evidence-provided setting, dev split) as an `RLMTrainEnv` environment: `context` is the list of the
question's Wikipedia evidence articles. Reward: loose accuracy (fraction of gold strings found in the answer).

Setup: `fanoutqa==1.1.1` plus the spaCy model `en_core_web_sm` (install the 3.8.0 wheel from the
explosion/spacy-models GitHub releases), and the frozen Wikipedia cache:
`mkdir -p ~/.cache/fanoutqa && cd ~/.cache/fanoutqa && curl -L -O
https://datasets.mechanus.zhu.codes/fanoutqa/wikicache.tar.gz && tar xzf wikicache.tar.gz`, then
`python -m fanoutqa_real.prepare --fill-network` (HTML to Markdown; fetches the few pages missing from the
tarball). Set `FANOUTQA_WIKICACHE_DIR` if the cache lives elsewhere.

Arguments used by the paper (`split="dev", setting="evidence", context_format="documents", metric="loose"`, `seed=42`):
- train: `max_context_chars=400000, num_examples=-1`
- validation: `min_context_chars=800000, num_examples=20`
