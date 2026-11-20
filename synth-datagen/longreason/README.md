# LongReason

**Real task.** Answer a reasoning multiple-choice question whose premises are scattered through 128k tokens of filler.

**Synthetic task** (`longreason-expanded`). The same construction from MCQs that are *not* in LongReason (ReClor,
LogiQA 2.0; near-duplicates removed): each passage is expanded by base Qwen3.8-27B into setting-restating paragraphs
and scattered through DCLM filler. Prompt and scoring are LongReason's; sub-calls go to a real model.

**Note.** `notes/longreason-expanded-train.md`: locate the premise paragraphs with code, then one sub-call that reasons
over all of them.

Two kinds of rows:
- **Root rows**: whole RLM trajectories following the note.
- **Sub-call rows** (STaR): the reasoning sub-call on its own, sampled K=4 times; items no sample solves are
  rationalized with the gold letter as a hint, which is then dropped.

```bash
L=synth-datagen/longreason; O=outputs/synth-datagen/longreason
python -m rlm_generate_trajectories.pipeline run $L/specs/gen_v1.toml          # root rollouts
python $L/build_sft_lr.py $O/root.jsonl 4 $O/gen_v1                            # filter root rows
python $L/star_prompts.py $O/star_prompts.jsonl                                # sub-call prompts
python $L/star_gen.py $O/star.jsonl 4 $O/star_prompts.jsonl                    # K=4 samples
python $L/star_rationalize.py $O/star.jsonl $O/star_rat.jsonl
python $L/assemble_v2.py $O/root.jsonl $O/star.jsonl $O/star_rat.jsonl $O/sft.jsonl 100 300
```
Expanded items are downloaded from HF `a1zhang/harness-cg-frozen-inputs` (`longreason/`) unless
`$LONGREASON_EXPANDED_ITEMS` is set. Thinking sub-calls need the server's `--reasoning-parser qwen3`.
