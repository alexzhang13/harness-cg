# common

- `rlm_generate_trajectories/`: spec-driven generation pipeline (BrowseComp-Plus and LongReason root rows). Each
  work item runs hinted rollouts, applies mechanical filters, has an LLM judge check program adherence, strips the
  hint and writes `accepted.jsonl` / `subcalls.jsonl`.
- `run_local.sh`: how the pipeline launches a work item.
- `make_clean_subset.py filter <full set dir> <out dir>`: the `clean AND synthetic` filter that cut
  `a1zhang/synthetic-rlm-data-clean` from `a1zhang/synthetic-rlm-data`.

```bash
export PYTHONPATH=src:synth-datagen/common:$(ls -d environments/synthetic/*/ | tr '\n' ':')
export PRIME_API_KEY=...                                           # judge credentials
python -m rlm_generate_trajectories.pipeline run    <spec.toml>    # resumable; loops until the target is reached
python -m rlm_generate_trajectories.pipeline status <spec.toml>
python -m rlm_generate_trajectories.pipeline export <spec.toml>    # -> <dir>/export/{train,subcalls}.jsonl
```

A spec (TOML) names the env config and streams, the teacher servers and sampling, the judge, and per-stream
overrides; see the specs under `browsecomp_plus/specs/` and `longreason/specs/`.
