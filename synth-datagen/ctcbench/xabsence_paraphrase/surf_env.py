"""xabsence_paraphrase rows in varied SURFACES (ctcbench_synth.surface) as an RLMTrainEnv with REAL 27B sub-calls."""
from __future__ import annotations
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")

import json
import random
import re
import sys

R = _REPO
sys.path[:0] = [f"{R}/environments/real/CTCBench", f"{R}/environments/synthetic/ctcbench",
                f"{R}/environments/synthetic/synth_common", f"{R}/src"]
FAM = "xabsence_paraphrase"
PREFIX = re.compile(r"^\s*(?:paraphrase|paraphrased|rephrased|restated|rewritten|rewrite)\s*:\s*", re.I)


def compose_varied(split: str, row: int, seed: int, decoys: int = 2, per_abstract: int = 3) -> dict:
    from ctcbench_synth import xabsence_para as XP
    rng = random.Random(f"xabpara-surf:{split}:{row}:{seed}")
    n_pairs = rng.randint(300, 345)
    n_orph = rng.choice([1, 2, 3, 3, 4, 5, 6])
    abstracts = list(XP.pool(split))
    rng.shuffle(abstracts)
    n_base = n_pairs + n_orph - n_orph * decoys
    sents, seen = [], set()
    for a in abstracts:
        prs = rng.sample(list(a["pairs"]), min(per_abstract, len(a["pairs"])))
        for o, p in prs:
            o, p = PREFIX.sub("", o), PREFIX.sub("", p)
            if XP._norm(o) in seen or XP._norm(p) in seen:
                continue
            seen.update((XP._norm(o), XP._norm(p)))
            sents.append((a["id"], o, p))
        if len(sents) >= n_base:
            break
    sents = sents[:n_base]
    orph = rng.sample(range(len(sents)), n_orph)
    side_of = {k: rng.choice("AB") for k in orph}
    for k in orph:
        aid, o, p = sents[k]
        sents += [(a2, PREFIX.sub("", o2), PREFIX.sub("", p2))
                  for a2, o2, p2 in XP._decoys(split, o if side_of[k] == "A" else p, aid, decoys, seen, rng)]
    A, B = [], []
    for k, (_aid, o, p) in enumerate(sents):
        if k in side_of:
            (A if side_of[k] == "A" else B).append((o if side_of[k] == "A" else p, True))
        else:
            A.append((o, False))
            B.append((p, False))
    rng.shuffle(A)
    rng.shuffle(B)
    docs = [{"corpus": "A", "text": t} for t, _ in A] + [{"corpus": "B", "text": t} for t, _ in B]
    gold = [i for i, (_, o) in enumerate(A + B) if o]
    return {"documents": docs, "queries": [], "gold_doc_indices": gold, "num_pairs": len(sents) - n_orph,
            "num_unmatched": n_orph, "source": "xabsence_synth_pubmed_surface"}


def build_rows(source: str, mode_of, n: int = -1, split: str = "train", seed: int = 0, part: str = "dev") -> list[dict]:
    """mode_of(k) -> surface mode for row k ("native", "varied" or a preset)."""
    from ctcbench_synth import surface as S
    if source == "real":
        from ctcbench_real import core as C
        from ctcbench_real import env as E
        cell = [c for c in C.manifest()["cells"] if c["task"] == FAM][0]
        hf = E._hf_rung(FAM, "r32k")
        exs = [(r["hf_row"], dict(hf[r["hf_row"]])) for r in cell[part]][: n if n > 0 else None]
    else:
        exs = [(k, compose_varied(split, k, seed)) for k in range(n)]
    out = []
    for k, (rid, ex) in enumerate(exs):
        mode = mode_of(k)
        r = S.render(ex, FAM, mode=mode, seed=(seed * 100003 + rid) if mode != "native" else None)
        meta = {"context": r.context, "root_prompt": r.root_prompt, "surface_mode": mode, "style": r.style,
                "id_map": r.id_map, "row": ex, "source": source, "row_id": rid, "seed": seed}
        out.append({"example_id": k, "prompt": [{"role": "user", "content": r.task_text.split("\n")[0][:2000]}],
                    "answer": S.gold_answer(ex, FAM, r), "info": json.dumps(meta, ensure_ascii=False)})
    return out


def _metrics(info, state):
    from ctcbench_synth import surface as S
    m = state.get("_xs_metrics")
    if m is None:
        meta = json.loads(info) if isinstance(info, str) else info
        m = S.score(state.get("rlm_final_answer") or "", meta["row"], FAM, meta["id_map"])
        state["_xs_metrics"] = m
    return m


async def _score(info, state, **_kw) -> float:
    return max(0.0, min(1.0, _metrics(info, state)["primary"]))


def load_environment(rows: list[dict], max_iterations: int = 20, **kwargs):
    import rlm_train
    from datasets import Dataset
    rubric = rlm_train.RLMTrainRubric(correctness=_score, weight=1.0, min_iterations=1, min_subcall=0)
    return rlm_train.RLMTrainEnv(dataset=Dataset.from_list(rows), code_block_feedback=True, max_iterations=max_iterations,
                                 sub_sampling_args={"max_tokens": None}, rubric=rubric,
                                 sub_extra_body={"chat_template_kwargs": {"enable_thinking": False}}, **kwargs)
