# longcot-mini-real

The `easy` difficulty of LongCoT through `longcot-real` (`difficulty` is pinned; every other argument is forwarded).

Setup: install `longcot-real` and the upstream verifier package
`uv pip install "longcot @ git+https://github.com/LongHorizonReasoning/longcot.git"` (not on PyPI).

Arguments used by the paper (validation stream of the broad mixture): `max_examples=20, seed=42,
max_iterations=20, sub_max_tokens=4096`. `max_examples` is not a `longcot_real` argument and is forwarded to
`RLMTrainEnv` unchanged; the orchestrator's `num_examples=20` bounds the stream.
