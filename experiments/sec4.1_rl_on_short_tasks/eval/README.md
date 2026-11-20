# Held-out evaluation suite (Table eval-envs, Table main-composition, statistics tables)

Eight benchmarks, 3,524 items, each run in its own harness; the model under test is the root and the sub-LM.
Used by §4.1, §4.2 and the broad-mixture appendix.

```bash
# serve the model (one or more vLLM servers, 262,144-token window) behind the router
MODEL=<hf id or dir> SERVED=<name> POOL=pool.json bash training/serving/serve_vllm.sh
LB_POOL=pool.json LB_USAGE_DIR=usage python training/serving/lb_proxy.py --listen 18431
# run, score, statistics
MODEL=<name> BASE_URL=http://127.0.0.1:18431/v1 ACCT_TAG=<name> OUT=runs/<name> bash run_suite.sh
python aggregate.py runs/<name> [--bcplus strict|judged]
python stats.py runs/<name> [--usage-dir usage --usage-tag <name>]
```
`PYTHONPATH` for `aggregate.py` / `stats.py` / `bcplus_judge_regrade.py`: `src` and `environments/real/browsecomp_plus`.

| Benchmark (n) | Env / driver | Harness |
|---|---|---|
| BrowseComp-Plus (830) | `browsecomp-plus-real` via `run_eval.py`; `split="all"`, `docs_per_query="all"` | orchestrator prompt, 40 turns, T 0.8, root 16,384 tok, sub 65,536 tok, 4 h/rollout, sub-call concurrency 24 |
| OBLIQ math (151) | `obliq-real` via `run_eval.py`; seed 74, pool 200,000, opaque ids, nDCG@50 | orchestrator prompt, 20 turns, T 0.8, root 16,384, sub 65,536, 4 h/rollout, sub concurrency 48 |
| CorpusQA (81) | `corpusqa-real` via `run_eval.py`; `1m`, `real_estate_en` | orchestrator prompt, 30 turns, T 0, root 8,192 |
| LooGLE-v2 (206) | `environments/synthetic/loogle_v2/tools/strategy_lab.py --arm root` | 12 turns, T 0.2, 6,000 tok/turn, <= 80 sub-calls, 2 h/item |
| OOLONG-Real D&D (577) | `oolong_dnd.py` (frozen ids `oolong_dnd_3ep_577_ids.json`) | 40 turns, T 0, root / reader 4,096, <= 600 sub-calls, 2 h/item |
| LongReason (794) | `longreason-real` via `run_eval.py`; `split="128k"` | 40 turns, T 0, root 4,096; sub-calls think (24,576 tok) unless `LR_SUB_THINKING=0` |
| ExpertLongBench (350) | `elb.py`; T01 / T03 / T04 / T06 = 100 / 50 / 100 / 100 | orchestrator prompt, 40 turns, T 0, root 8,192; GPT-4.1-mini checklist judge |
| CTC-Bench (535) | `ctc-core-env` (4 core cells, `part="test"`), `ctcbench-real` grouping (`ids="grouping32_test"`) via `run_eval.py` | 30 turns, uncapped root and sub-calls; core T 0.7, grouping T 0 + orchestrator prompt; sub concurrency 8 |

Thinking is off for every root call and every sub-call (`chat_template_kwargs: {"enable_thinking": false}`), except
LongReason's sub-calls (above) and `OOLONG_THINKING=low`. REPL timeouts: `RLM_TRAIN_EXEC_TIMEOUT_S=7200`,
`RLM_TRAIN_SUBCALL_TIMEOUT_S=3600`, `RLM_TRAIN_REPL_REQUEST_TIMEOUT_S=14400` (CTC: 14400 / 10800 / 30000).
Data: each env package's README (BC+ needs its document store, see `environments/real/browsecomp_plus`).
ELB and `bcplus_judge_regrade.py` need judge credentials (`PRIME_API_KEY`, or `RLM_JUDGE_BASE_URL` / `RLM_JUDGE_API_KEY`).

**Scoring** (`aggregate.py`): first valid attempt per item; infra failures (dead REPL worker, rollout timeout,
request error; failed sub-calls for OBLIQ; judge errors for ELB) are invalid, listed, and made rerunnable with
`aggregate.py OUT --invalidate` followed by the same `run_suite.sh` command; turn-cap zeros count. CTC is the mean
of its five cells. ELB judge errors: `elb_rejudge.py OUT/elb/*.jsonl`.

**Paper rows.** Qwen3.8-27B base: `LR_SUB_THINKING=0`; RLM-27B and the RL rows: `LR_SUB_THINKING=1`. BC+ for the
27B rows: `bcplus_judge_regrade.py OUT` then `--bcplus judged`; for the 30B rows: `--bcplus strict`. The 30B rows
counted dead REPL workers as 0 (`--dead-as-zero`) and report router-measured tokens (`--usage-dir`); the 27B rows
report per-trajectory tokens. Thinking flags have no effect on Qwen3-30B-A3B-Instruct-2507.
