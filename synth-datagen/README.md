# synth-datagen: SFT traces from synthetic tasks (§4.2)

RLM-27B is Qwen3.8-27B fine-tuned on RLM trajectories collected on **synthetic stand-ins** for eight long-context
benchmarks. Each stand-in is a procedurally generated task with the benchmark's shape (same question style, answer
format and scorer) but none of its data, so the traces teach a *strategy* without touching the evaluation set.

The released traces are [`a1zhang/synthetic-rlm-data-clean`](https://huggingface.co/datasets/a1zhang/synthetic-rlm-data-clean);
training reads them directly (`training/sft/build_source_rows.py`), so nothing here needs to be re-run to train.

## Recipe (same for every task)

1. **Note.** A short program note (in each task's `notes/`) describes a good RLM strategy for the task: what to probe,
   what to delegate to sub-calls, what to compute in Python, when to submit.
2. **Generate.** Base Qwen3.8-27B (thinking off) runs the synthetic env as an RLM with the note as a hint.
3. **Filter.** Keep traces that are correct and follow the note's program (few turns, no raw dumps of `context`).
4. **Strip.** Remove the hint, so the SFT row shows the strategy without being told it.

## Tasks

| Directory | Real benchmark | Synthetic env | Synthetic task |
|---|---|---|---|
| [`browsecomp_plus/`](browsecomp_plus) | BrowseComp-Plus | `browsecomp-plus-synth` | find one entity in a generated ~27k-document encyclopedia |
| [`corpusqa/`](corpusqa) | CorpusQA | `corpusqa-synth` | aggregate statistics over generated periodic report corpora |
| [`ctcbench/`](ctcbench) | CTC-Bench (5 cells) | `ctcbench-synth` | CTC train rows, recombined toward the long eval lengths |
| [`expertlongbench/`](expertlongbench) | ExpertLongBench T01/T03/T04/T06 | `expertlongbench-synth` | fill an expert checklist from generated records |
| [`longreason/`](longreason) | LongReason | `longreason-expanded` | reasoning MCQs (not from LongReason) expanded into 128k of filler |
| [`loogle/`](loogle) | LooGLE-v2 | `loogle-strategy-synth` | questions over generated 10-Ks, repos, game logs, version trees |
| [`obliq/`](obliq) | OBLIQ-Bench (math) | `trick-synth` | rank generated problems by shared solution trick |
| [`oolong_real/`](oolong_real) | OOLONG-Real (D&D) | `tabletop-synth` | count events in generated tabletop session logs |

`common/` holds the shared generation pipeline (used by BrowseComp-Plus and LongReason) and the overlap filter that
cut the clean set.

## Running

Serve Qwen/Qwen3.8-27B as `qwen38-27b` with the qwen3 reasoning parser
(`REASONING_PARSER=qwen3 bash training/serving/serve_vllm.sh`); scripts default to `localhost:8000`/`8001`. Judges
(BrowseComp-Plus, LongReason, LooGLE) use Prime Inference (`PRIME_API_KEY`). Sampling is not
bit-reproducible, so a rerun gives a comparable set, not the same rows.
