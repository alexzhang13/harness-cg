# Environments

Every benchmark is a separately installable package (a verifiers environment; most wrap `rlm_train.RLMTrainEnv`).
`training/prime_rl/install.sh` installs all of them; to install one by hand: `uv pip install -e environments/<group>/<package>`.
Each package's README gives its env id, arguments, data source and grader. Data downloads from Hugging Face on first
use at the revisions the paper used, except where a package README says otherwise (Ada-LEval: `ADALEVAL_DATA_DIR`;
OOLONG-real: `OOLONG_REAL_DATA_DIR`).

## Sections 3.1 / 3.2: length and strategy generalization

| Package | Benchmark | Used in |
|---|---|---|
| `real/mrcr_v2` | MRCRv2 multi-needle recall | length-gen |
| `real/graphwalks` | GraphWalks | length-gen |
| `real/longbenchpro` | LongBench-Pro (English) | length-gen |
| `real/oolong` | OOLONG-synth | length-gen; strategy-gen (trec -> spam) |
| `real/oolong_pairs` | OOLONG-Pairs | length-gen |
| `real/ada_leval` | Ada-LEval BestAnswer | length-gen |
| `real/obliq` | OBLIQ-Bench retrieval | strategy-gen (writing -> math, twitter -> wildchat); held-out suite |
| `singleturn` | the base Transformer baseline (whole context in one prompt) | both |
| `pi/pi_*_v1` | the same benchmarks for the Pi baseline (`rlm_train.pi.make_pi_env`) | both |

## Section 4.1: held-out evaluation suite

`real/browsecomp_plus` (BrowseComp-Plus), `real/obliq`, `real/corpusqa` (CorpusQA), `real/loogle_v2` (LooGLE-v2),
`real/expertlongbench` (ExpertLongBench), `real/longreason` (LongReason), `real/CTCBench` (CTC-Bench), and the
OOLONG-real D&D task (`synthetic/oolong_real`, `real_task`). Driven by `experiments/sec4.1_rl_on_short_tasks/eval/`.

## Section 4.2: synthetic environments

| Package | Synthetic counterpart of |
|---|---|
| `synthetic/browsecomp_plus` | BrowseComp-Plus |
| `synthetic/corpusqa` | CorpusQA |
| `synthetic/ctcbench` | CTC-Bench |
| `synthetic/expertlongbench` | ExpertLongBench |
| `synthetic/longreason` | LongReason |
| `synthetic/loogle_v2` | LooGLE-v2 |
| `synthetic/oolong_real`, `synthetic/tabletop_synth` | OOLONG-real (D&D session logs) |
| `synthetic/trick` | OBLIQ |
| `synthetic/synth_common` | shared library (scripted sub-calls, surfaces) |
| `rl_staging` | the RL training and validation streams built from the packages above |

## Appendix: broad task mixture

`real/babilong`, `real/fanoutqa`, `real/golongrl`, `real/longcot`, `real/longcot_mini`, `real/longproc`, `real/longrlvr`,
`real/loongrl`, `real/manyih_priv`, `real/nocha`, `real/personamem_v2`, `real/personamem_v3`, `real/prefeval`, together
with the section 3 and held-out packages above (`experiments/appendix/broad_mixture/configs/train-mix-large.toml`).
