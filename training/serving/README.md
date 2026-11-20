# Serving

| Script | Purpose |
|---|---|
| `serve_vllm.sh` | serve one model with vLLM, one server per tensor-parallel GPU group, registered in a pool file (LoRA adapters, YaRN and the reasoning parser are options; see its header) |
| `lb_proxy.py` | least-loaded router over the pool: one OpenAI-compatible endpoint for all servers |
| `serve_and_run.sh` | both of the above, then a command with `BASE_URL` / `MODEL` exported (held-out suite, gated chains, SFT data generation) |

```bash
MODEL=a1zhang/RLM-27B SERVED=rlm-27b TOKENIZER=Qwen/Qwen3.8-27B TP=2 \
  bash training/serving/serve_and_run.sh bash experiments/sec4.3_recursion_composition/eval.sh harness
```
