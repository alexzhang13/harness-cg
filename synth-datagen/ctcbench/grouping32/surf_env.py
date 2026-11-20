"""Synthetic grouping rows (compose_v3, decontaminated pool) in surface modes as an RLMTrainEnv with REAL 27B sub-calls.
Scoring: surface.score (answer in the row's own ids -> benchmark scorer). Gate rows come from the dev paper pool, train
rows from the train pool (disjoint papers)."""
from __future__ import annotations
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import json, sys
R = _REPO
sys.path[:0] = [f"{R}/environments/real/CTCBench", f"{R}/environments/synthetic/ctcbench",
                f"{R}/environments/synthetic/synth_common", f"{R}/src", __import__("os").path.dirname(__import__("os").path.abspath(__file__))]
FAM = "grouping"
def build_rows(split: str, mode_of, n: int, seed: int = 0, start: int = 0) -> list[dict]:
    import compose
    from ctcbench_synth import surface as S
    out = []
    for k in range(start, start + n):
        ex = compose.compose_v3(split, k, seed=seed); mode = mode_of(k)
        r = S.render(ex, FAM, mode=mode, seed=(seed * 100003 + k) if mode != "native" else None)
        meta = {"context": r.context, "root_prompt": r.root_prompt, "surface_mode": mode, "style": r.style,
                "id_map": r.id_map, "row": ex, "source": f"synth-{split}", "row_id": k, "seed": seed}
        out.append({"example_id": k, "prompt": [{"role": "user", "content": r.task_text.split("\n")[0][:2000]}],
                    "answer": S.gold_answer(ex, FAM, r), "info": json.dumps(meta, ensure_ascii=False)})
    return out
def _metrics(info, state):
    from ctcbench_synth import surface as S
    m = state.get("_g_metrics")
    if m is None:
        meta = json.loads(info) if isinstance(info, str) else info
        m = S.score(state.get("rlm_final_answer") or "", meta["row"], FAM, meta["id_map"]); state["_g_metrics"] = m
    return m
async def _score(info, state, **_kw) -> float:
    return max(0.0, min(1.0, _metrics(info, state)["primary"]))
def load_environment(rows: list[dict], max_iterations: int = 15, **kwargs):
    import rlm_train
    from datasets import Dataset
    rubric = rlm_train.RLMTrainRubric(correctness=_score, weight=1.0, min_iterations=1, min_subcall=0)
    return rlm_train.RLMTrainEnv(dataset=Dataset.from_list(rows), code_block_feedback=True, max_iterations=max_iterations,
                                 sub_sampling_args={"max_tokens": None}, rubric=rubric,
                                 sub_extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kwargs)
