# CorpusQA

**Real task.** Answer an aggregation question (e.g. a total or trend) over a 1M-token corpus of statistical reports.

**Synthetic task** (`corpusqa-synth`). Generated corpora of periodic monthly releases (adjusted / not-adjusted table
pairs, revised prints, newest release wins) with a CorpusQA-style question and an exact gold. Surfaces are varied and
decontaminated against the eval set. Sub-calls go to the same model (thinking off).

**Note.** `notes/corpusqa-probe-v3.md`: catalogue the distinct tables, make one PLAN sub-call, build the ledger in
Python, fall back to a sub-call for tables the parser cannot read.

```bash
G=synth-datagen/corpusqa; O=outputs/synth-datagen/corpusqa
export PYTHONPATH=src:environments/synthetic/synth_common:environments/synthetic/corpusqa
python $G/generate_sft.py run --base-url http://localhost:8000/v1 --start 31000 --n 500 \
  --hint-file $G/notes/corpusqa-probe-v3.md --out $O/run          # odd seeds 1M, even seeds 128k
python $G/generate_sft.py export --out $O/run --program probe --max-turns 6 --dest $O/export
```
The published rows come from several such runs (seeds 30060-33300).
