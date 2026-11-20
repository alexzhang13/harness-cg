# BrowseComp-Plus

**Real task.** Answer a multi-constraint question by finding one entity in a ~100k-document web corpus.

**Synthetic task** (`browsecomp-plus-synth`). The same kind of question over a procedurally generated ~27k-document
encyclopedia built from a random knowledge graph (regions, towns, organisations, people, works), with near-twin decoys.
Sub-calls are answered by a programmed graph oracle with a limited read window and occasional judgement errors.
Reward is the real env's answer match.

**Notes.**
- `notes/train/bcpsynth-train.md`: the main program (search the corpus in Python, read candidate documents with
  sub-calls, never print document text).
- `notes/commit/bcpsynth-hard-commit.md`: same program for hard tasks, stressing committing to an answer before the
  turn cap (the `commit-near-cap` rows).

```bash
for s in synthgen k3 k3b commit; do
  python -m rlm_generate_trajectories.pipeline run    synth-datagen/browsecomp_plus/specs/$s.toml
  python -m rlm_generate_trajectories.pipeline export synth-datagen/browsecomp_plus/specs/$s.toml
done
python synth-datagen/browsecomp_plus/build_commit.py outputs/synth-datagen/browsecomp_plus/commit commit_rows.jsonl
```
`k3`/`k3b` sample three rollouts per task and keep the fewest-turns correct one. Env args and acceptance thresholds:
`envs.toml`.
