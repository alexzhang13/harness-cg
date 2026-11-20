# longcot-real

LongCoT long-horizon reasoning (logic / cs / chemistry / chess / math) as an `RLMTrainEnv` environment. Questions
and grading come from the upstream `longcot` package (binary `longcot.verify`; its math / chemistry LLM fallbacks
use the repo judge endpoint when `RLM_JUDGE_*` or a provider key is set).

Setup: `uv pip install "longcot @ git+https://github.com/LongHorizonReasoning/longcot.git"` (not on PyPI).

Used by the paper only through `longcot-mini-real` (`difficulty="easy"`).
