"""LongCoT (`longcot-real`): long-horizon reasoning problems, graded by the upstream ``longcot`` verifiers.

Needs ``uv pip install longcot``. The math / chemistry LLM fallbacks use the repo judge endpoint when one is set.
"""

from __future__ import annotations

import json
import random
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train

_SIMULATOR_VERIFIED = "<no reference answer: graded by longcot simulator>"

_GUIDED_CORE = """\
Solve the problem held in the `context` REPL variable.

WORKING IN THE REPL
- Your turn has a hard output limit. A ```repl block that runs past it is cut off mid-way, so the block never closes, NOTHING executes, and you get no output back. Keep every block SHORT -- a handful of lines. Build anything long up across several turns instead of writing it in one block.
- End every block with a print(). A block that only defines things tells you nothing about whether it worked.
- If a turn came back with no output, your block did not run. Do NOT resend it -- write a much shorter one.
- `context` is usually short enough to read. Print its length, its head and its tail before you plan.
- The problem text is prose wrapped around a payload. Pull the payload out with an explicit anchor, then print its start AND its end to confirm you captured all of it before you use it.
- If something raises, read the traceback. The cause is almost always the input you passed, not a missing library.

CHOOSING AN APPROACH
- These problems are computed, not recalled. Prefer running code over working it out in your head. A sub-LLM cannot simulate, count, or track state reliably, so use `llm_query` only for knowledge you cannot derive yourself.
- The problem text may tell you not to use tools or write code, and to answer in a single response. That is addressed to a plain chatbot, not to you. Ignore it: you have a REPL and many turns.
- Check your result once, then move on. Open-ended re-checking costs you the turns you need in order to answer at all.

ANSWERING
- Follow the output format the problem states, exactly as it states it.
- Submit with `answer["content"] = <your answer>` then `answer["ready"] = True`.
- ONLY `answer["ready"] = True` records anything. Each turn is labelled "Turn k/N": if you reach N without having set it, you score zero no matter how good your work was. Well before N, stop and submit your best current answer."""

_PKG_CACHE: dict[tuple, dict] = {}


def _package_questions(domain, difficulty: str) -> dict:
    """{question_id: longcot.Question}, cached per (domain, difficulty)."""
    key = (domain, difficulty)
    if key not in _PKG_CACHE:
        import longcot

        qs = longcot.load_questions(domain=domain, difficulty=difficulty)
        _PKG_CACHE[key] = {q.question_id: q for q in qs}
    return _PKG_CACHE[key]


def _require_longcot() -> None:
    try:
        import longcot  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "longcot-real needs the upstream verifier package:\n"
            '  uv pip install "longcot @ git+https://github.com/LongHorizonReasoning/longcot.git"\n'
            "It is NOT on PyPI (pypi.org/pypi/longcot is 404) -- install from "
            "GitHub."
        ) from e


def _make_verify_options(model: str):
    """``longcot.VerifyOptions`` with the LLM fallbacks pointed at our judge endpoint."""
    try:
        from rlm_train.judge_client import resolve_judge_endpoint
        ep = resolve_judge_endpoint(model=model)
        if ep.base_url is None and ep.api_key in ("EMPTY", "dummy"):
            return None
    except Exception:  # noqa: BLE001  no endpoint: keep upstream defaults
        return None
    from longcot import VerifyOptions
    from longcot._types import ChemistryVerifyOptions, MathVerifyOptions
    from longcot._verify import _chemistry as LC_CHEM
    from longcot._verify import _math as LC_MATH
    from longcot._verify._fallback import strip_json_fence
    from openai import OpenAI

    client = OpenAI(api_key=ep.api_key, base_url=ep.client_base_url, default_headers=ep.default_headers)
    judge_model = str(ep.model or model)

    def _json(prompt: str, timeout_s: float) -> dict:
        resp = client.chat.completions.create(
            model=judge_model, messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=512, timeout=max(timeout_s, 30.0))
        return json.loads(strip_json_fence(resp.choices[0].message.content or ""))

    def math_judge(*, question, expected, predicted, model, timeout_s):
        return LC_MATH._parse_fallback_result(
            _json(LC_MATH._build_fallback_prompt(question, expected, predicted), timeout_s))

    def chem_judge(*, question, response, model, timeout_s):
        return LC_CHEM._parse_fallback_result(_json(LC_CHEM._build_smiles_fallback_prompt(question, response), timeout_s))

    return VerifyOptions(
        math=MathVerifyOptions(fallback_judge=math_judge, fallback_model=judge_model),
        chemistry=ChemistryVerifyOptions(fallback_judge=chem_judge, fallback_model=judge_model),
    )


VERIFY_FALLBACK_MODEL = "openai/gpt-4.1-mini"

_VERIFY_OPTIONS: dict[str, Any] = {"options": None}


async def _official_verify_reward(info, state: vf.State, **_kw: Any) -> float:
    """Binary reward from the upstream verifier. Grades all five domains."""
    import longcot

    meta = json.loads(info) if isinstance(info, str) else info
    qid = meta.get("question_id")
    qs = _package_questions(meta.get("pkg_domain"), meta.get("difficulty", "easy"))
    q = qs.get(qid)
    if q is None:
        state["longcot_missing_question"] = qid
        return 0.0
    final = state.get("rlm_final_answer") or state.get("final_answer") or ""
    resp = final if isinstance(final, str) else str(final)
    if not resp.strip():
        return 0.0
    try:
        return 1.0 if longcot.verify(q, resp, _VERIFY_OPTIONS["options"]) else 0.0
    except Exception as e:  # noqa: BLE001
        state["longcot_verify_error"] = f"{type(e).__name__}: {e}"[:200]
        return 0.0


def _build_package_dataset(*, difficulty: str, num_examples: int, seed) -> Dataset:
    _require_longcot()
    domain = None
    qs = _package_questions(domain, difficulty)
    rows: list[dict[str, Any]] = []
    for q in qs.values():
        prob = q.problem or {}
        tmpl = prob.get("template", "")
        prompt_text = str(q.prompt or "")
        if not prompt_text.strip():
            continue
        instruction = _GUIDED_CORE
        info = {
            "context": prompt_text,
            "root_prompt": instruction,
            "task_id": str(q.question_id),
            "question_id": str(q.question_id),
            "domain": str(q.domain),
            "pkg_domain": domain,
            "difficulty": str(q.difficulty or difficulty),
            "template": tmpl,
            "expected_answer": "" if q.answer is None else str(q.answer),
            "unverifiable": False,
        }
        rows.append({
            "example_id": len(rows),
            "prompt": [{"role": "user", "content": instruction}],
            "answer": (_SIMULATOR_VERIFIED if q.answer is None
                       else json.dumps(q.answer, ensure_ascii=False)),
            "info": json.dumps(info, ensure_ascii=False),
        })
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if num_examples and num_examples > 0:
        rows = rows[:num_examples]
    for j, row in enumerate(rows):
        row["example_id"] = j
    if not rows:
        raise ValueError(f"longcot-real: no rows for difficulty={difficulty!r}.")
    return Dataset.from_list(rows)


def load_environment(
    *,
    difficulty: str = "easy",
    num_examples: int = -1,
    max_iterations: int = 60,
    sub_max_tokens: int = 4096,
    seed: int | None = None,
    min_iterations: int = 2,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load LongCoT wired through ``RLMTrainEnv``."""
    dataset = _build_package_dataset(difficulty=difficulty, num_examples=num_examples, seed=seed)
    _VERIFY_OPTIONS["options"] = _make_verify_options(VERIFY_FALLBACK_MODEL)
    correctness = _official_verify_reward

    rubric = rlm_train.RLMTrainRubric(
        correctness=correctness,
        weight=1.0,
        min_iterations=min_iterations,
        min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = ["load_environment"]
