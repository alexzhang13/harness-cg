"""contradiction_iid 1M rows (v3, all-family decontaminated, ~185 injected duplicate fillers) in varied SURFACES
(ctcbench_synth.surface) as an RLMTrainEnv with REAL 27B sub-calls. Scoring: surface.score (row ids -> benchmark)."""
from __future__ import annotations
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import json, os, random, sys
R = _REPO
sys.path[:0] = [f"{R}/environments/real/CTCBench", f"{R}/environments/synthetic/ctcbench",
                f"{R}/environments/synthetic/synth_common", f"{R}/src", os.path.dirname(os.path.abspath(__file__))]
FAM = "contradiction_iid"
D = os.path.join(os.path.join(os.environ.get("CTC_DATAGEN_DIR") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../outputs/synth-datagen/ctcbench")), "contradiction"), "dev1m")

def raw_rows(split, start, n, ndup=185):
    out = []
    for k, line in enumerate(open(os.path.join(D, f"rows_v3_{split}.jsonl"))):
        if k < start: continue
        if len(out) >= n: break
        ex = json.loads(line); rng = random.Random(ex["dev_id"] * 7 + (3 if split == "dev" else 11))
        docs = [dict(d) for d in ex["documents"]]
        gpos = {i - 1 for p in ex["gold_doc_indices"] for i in p}
        free = [i for i in range(len(docs)) if i not in gpos]
        tgt = rng.sample(free, ndup); src = rng.sample([i for i in free if i not in set(tgt)], ndup)
        for t, s in zip(tgt, src): docs[t] = {"text": docs[s]["text"]}
        out.append((ex["dev_id"], {"documents": docs, "gold_doc_indices": ex["gold_doc_indices"], "queries": [], "answers": [],
                                   "source": "pubmed_perturbation_v3"}))
    return out

def build_rows(split, mode_of, start=0, n=20, seed=0):
    from ctcbench_synth import surface as S
    out = []
    for k, (rid, ex) in enumerate(raw_rows(split, start, n)):
        mode = mode_of(k)
        r = S.render(ex, FAM, mode=mode, seed=(seed * 100003 + rid) if mode != "native" else None)
        meta = {"context": r.context, "root_prompt": r.root_prompt, "surface_mode": mode, "style": r.style,
                "id_map": r.id_map, "row": ex, "split": split, "row_id": rid, "seed": seed}
        out.append({"example_id": k, "prompt": [{"role": "user", "content": r.task_text.split("\n")[0][:2000]}],
                    "answer": S.gold_answer(ex, FAM, r), "info": json.dumps(meta, ensure_ascii=False)})
    return out

def _metrics(info, state):
    from ctcbench_synth import surface as S
    m = state.get("_cs_metrics")
    if m is None:
        meta = json.loads(info) if isinstance(info, str) else info
        m = S.score(state.get("rlm_final_answer") or "", meta["row"], FAM, meta["id_map"]); state["_cs_metrics"] = m
    return m

async def _score(info, state, **_kw):
    return max(0.0, min(1.0, _metrics(info, state)["primary"]))

def load_environment(rows, max_iterations=15, **kwargs):
    import rlm_train
    from datasets import Dataset
    rubric = rlm_train.RLMTrainRubric(correctness=_score, weight=1.0, min_iterations=1, min_subcall=0)
    return rlm_train.RLMTrainEnv(dataset=Dataset.from_list(rows), code_block_feedback=True, max_iterations=max_iterations,
                                 sub_sampling_args={"max_tokens": None}, rubric=rubric, max_sub_calls=4000,
                                 sub_extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kwargs)
