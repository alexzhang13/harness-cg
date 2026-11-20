# personamem-v3

PersonaMem-v3 `hidden_persona_recommendation` (deterministic slate ranking) as an `RLMTrainEnv` environment.
`context` holds the user's time-masked cross-app history, calendar, test moment and candidate slate; the model
submits a ranking of all candidates. Reward: upstream graded NDCG@5.

Data: HF `bowen-upenn/PersonaMem-v3` (`backend/<persona>/*.json`, fetched per persona; or a local copy via
`PERSONAMEM_V3_BACKEND_DIR`). Licence CC-BY-NC-4.0 (non-commercial).

Arguments used by the paper: `task="hidden_persona_recommendation", history_event_cap=128, num_examples=1500,
seed=42` (train only).
