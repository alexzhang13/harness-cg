# oolong-real-synth (support package)

No env id of its own. `oolong_real_synth.real_task` loads the real OOLONG-real D&D task (`oolongbench/oolong-real`,
config `dnd`; the ~9 GB `dnd/{validation,test}.jsonl` files are streamed) as an `RLMTrainEnv` with a real sub-LM; it
is used by the OOLONG-Real eval driver and by `rlstage-oolong-real-val` (RL validation: 25 windows of <= 3 episodes
from `$OOLONG_REAL_DATA_DIR/dnd/validation.jsonl`, seed 0). `gen.py`, `world.py` and `scoring.py` (question templates,
vocabulary, paper metric) are shared with `tabletop_synth`.

Download: `huggingface-cli download oolongbench/oolong-real --repo-type dataset --revision 6bc9ef04866fcf005c9749b70649be69dd37fffb --include "dnd/*" --local-dir
$OOLONG_REAL_DATA_DIR`.
