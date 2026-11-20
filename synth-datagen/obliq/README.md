# OBLIQ-Bench (math)

**Real task.** Given a query problem, rank a large pool of problems by whether they share its core solution technique
(scored by nDCG@50).

**Synthetic task** (`trick-synth`). The same ranking over ~2,500-3,100 generated problems, each written around one of
125 hidden techniques. Sub-calls are answered by a fake sub-LM calibrated to Qwen3.8-27B on OBLIQ math, with 15%
faulty replies.

**Notes.** `notes/map_flag_v7a.md` and `notes/map_flag_v7b.md`, two phrasings of one program: sketch every problem's
solution in batched sub-calls, rank by similarity in code, flag the top candidates with one more pass, submit.

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 OUT=outputs/synth-datagen/obliq bash synth-datagen/obliq/run_gen.sh
O=outputs/synth-datagen/obliq
python synth-datagen/obliq/build_sft_v7.py $O/sft.jsonl 0.05 10 none $O/trick_gen9 $O/trick_gen10 $O/trick_probe/pilot_v7*
```
The problem corpus is downloaded from HF `a1zhang/harness-cg-frozen-inputs` (see `environments/synthetic/trick`).
