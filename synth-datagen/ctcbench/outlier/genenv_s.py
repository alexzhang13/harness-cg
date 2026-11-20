"""RLMTrainEnv over decontaminated outlier rows rendered through ctcbench_synth.surface, with an optional note."""
import gzip, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / p) for p in ("src", "environments/real/CTCBench", "environments/synthetic/synth_common",
                                        "environments/synthetic/ctcbench")]
import rlm_train  # noqa: E402
import verifiers as vf  # noqa: E402
from datasets import Dataset  # noqa: E402
from ctcbench_synth import surface as S  # noqa: E402

NOTE = Path(os.environ.get("OUTLIER_NOTE", str(ROOT / "synth-datagen/ctcbench/notes/ctc-outlier-1m-k-train.md")))


def note_prologue(path=NOTE):
    sys.path.insert(0, str(ROOT / "synth-datagen/common"))
    from rlm_generate_trajectories.strategies import HARNESS_FOOTER, HINT_CLOSE, HINT_OPEN
    return f"{HINT_OPEN}\n{Path(path).read_text().strip()}\n\n{HARNESS_FOOTER.strip()}\n{HINT_CLOSE}"


def mode_for(k, mix):
    """mix: 'native', a preset/varied name, 'gen' (25% native / 75% varied by row index), or 'a,b,c' cycled."""
    if mix == "gen":
        return "native" if k % 4 == 0 else "varied"
    ms = mix.split(",")
    return ms[k % len(ms)]


def build(path, idx=None, mix="native", seed0=0):
    rows = []
    with gzip.open(path, "rt") as f:
        for k, line in enumerate(f):
            if idx is not None and k not in idx:
                continue
            ex = json.loads(line)
            mode = mode_for(k, mix)
            r = S.render(ex, "outlier", mode=mode, seed=seed0 + k)
            slim = {a: b for a, b in ex.items() if a != "documents"}; slim["n_docs"] = len(ex["documents"])
            meta = {"context": r.context, "root_prompt": r.root_prompt, "ctc_task": "outlier",
                    "gen_row": {"source": str(path), "row": k, "mode": mode, "seed": seed0 + k, **(ex.get("dev_meta") or {})},
                    "surface": {"id_map": r.id_map, "style": r.style}, "example": slim}
            rows.append({"example_id": len(rows), "prompt": [{"role": "user", "content": r.task_text[:2000]}],
                         "answer": S.gold_answer(ex, "outlier", r), "info": json.dumps(meta, ensure_ascii=False)})
    return rows


def _meta(info):
    return json.loads(info) if isinstance(info, str) else info


async def surface_score(info, state, **_):
    m = _meta(info)
    sc = S.score(state.get("rlm_final_answer"), m["example"], "outlier", m["surface"]["id_map"])
    return max(0.0, min(1.0, sc["primary"]))


def make_env(path, idx=None, note=False, mix="native", max_iterations=20, **kw):
    rows = build(path, idx, mix)
    rubric = rlm_train.RLMTrainRubric(correctness=surface_score, weight=1.0)
    return rlm_train.RLMTrainEnv(dataset=Dataset.from_list(rows), code_block_feedback=True, max_iterations=max_iterations,
                                 sub_sampling_args={"max_tokens": None}, rubric=rubric,
                                 user_prologue=note_prologue() if note else None, **kw)
