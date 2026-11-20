# LooGLE-v2

**Real task.** Answer questions over very long real documents: financial filings, code repositories, game logs and
software version histories.

**Synthetic task** (`loogle-strategy-synth`). The same four domains, generated: fictional 10-K filings, Python
repositories with call graphs, game logs and version trees, with LooGLE-v2's question types, answer formats and
scoring. Sub-calls go to the same model (thinking off).

**Notes.**
- `notes/compact_v6_{finance,code,game,versions}_program.txt`: one compact program per domain (readers extract
  facts, Python aggregates, submit as soon as the result is computed).
- `notes/evidence_repair_reader_first_v1.txt`: a reader-first variant (the `reader-first` rows).

```bash
L=synth-datagen/loogle; O=outputs/synth-datagen/loogle   # needs PRIME_API_KEY (judge)
python $L/generate_batch.py --config $L/recipes/expand_batch.json --out $O/compact
python $L/generate_batch.py --config $L/recipes/reader_first_batch.json --out $O/reader_first
```
Each batch runs `synthetic_strategy_lab.py` per job, has a judge review the candidates (`review_candidates.py`:
delegation, efficiency, groundedness) and bundles the kept ones into `<out>/sft_sampled/train.jsonl`.
