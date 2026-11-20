# CTC-Bench (5 cells)

**Real task.** CTC-Bench measures corpus-level tasks at growing context lengths: find the outlier article, detect a
contradiction, group texts, match questions to documents, spot what a paraphrased corpus is missing.

**Synthetic task** (`ctcbench-synth`). Rows built with CTC-Bench's own generator from its *train* pools, recombined up
to the long eval lengths and rendered in varied formats. The SFT traces were generated with real 27B sub-calls (a plain RLM
environment over these rows); RL training answers sub-calls with fake sub-LMs calibrated to the real 27B. Every row is checked against all 23 eval families (exact text and 13-gram hits) and dropped on any overlap.

**Notes** (one program per cell; two cells have two versions, both used):

| Cell | Note (`notes/`) | Generation | Filter |
|---|---|---|---|
| qdmatch (HotpotQA, 256k) | `ctc-qdmatch-hpqa-256k-train.md` | `qdmatch_hpqa/run_dev.py note` | `qdmatch_hpqa/filter_sft.py` |
| xabsence paraphrase (32k) | `ctc-xabsence-paraphrase-32k-train-v7.md`, `-v51.md` | `xabsence_paraphrase/gen.py` | `xabsence_paraphrase/build_sft.py` |
| contradiction (1M) | `ctc-contradiction-1m-train-v2.md` | `contradiction/gen_contra.py` | `contradiction/build_sft_contra.py` |
| outlier (1M) | `ctc-outlier-1m-k-train-v2.md`, `ctc-outlier-1m-k-train.md` | `outlier/rollouts_steer.py note` | `outlier/build_sft_k.py` |
| grouping (32k) | `ctc-grouping-32k-train-v3.md` | `grouping32/gen.py` | `grouping32/build_sft.py` |

## Building the rows

Needs the CTC-Bench generator (https://github.com/PrasannS/corpustaskcomplexity, commit `2e16b61`; set
`CTC_REPO_SRC=<clone>/ctc/src`). Work dirs go under `$CTC_DATAGEN_DIR` (default `outputs/synth-datagen/ctcbench`).
Run from this directory with `PYTHONHASHSEED=0`:

```bash
python audit/build_index.py 32                 # eval-overlap index over all 23 families (~2 GB)
python contradiction/decontam_pool.py && python contradiction/extra_fillers.py
python contradiction/build_rows_v3.py train 1350
python grouping32/scan_pool.py                 # decontaminated pool for grouping32/compose.py
python outlier/all23.py build
python outlier/build_all23.py train1m_all23_s7301.jsonl.gz 300 7301 train dev1m_all23_s1301.jsonl.gz
python qdmatch_hpqa/filter_pool_v2.py - pool23/qdmatch_hpqa.seed.jsonl.gz pool_23fam_report.json
PYTHONPATH=$CTC_REPO_SRC python qdmatch_hpqa/gen_build_pinned.py build --task qdmatch_hpqa --split train \
  --seed 4001 --rungs 256k --pool pool23/qdmatch_hpqa.seed.jsonl.gz --train 300 --out built_s4001
```
The routers spread requests over the teacher ports in `$CTC_PORTS_POOL`.
