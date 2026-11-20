"""LLM review of candidates from synthetic_strategy_lab.py; ``loogle_datagen.quality.quality_passes`` decides.

    python review_candidates.py <run dir> --concurrency 4 --judge-model openai/gpt-5.4-mini --max-turns 6
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(Path(__file__).resolve().parent)]
from loogle_datagen.quality import quality_passes, review_json_schema  # noqa: E402
from loogle_datagen.recovery import acknowledges_and_prevents  # noqa: E402

from rlm_train.judge_client import build_judge_client  # noqa: E402

RUBRIC = """Review an RLM training trajectory. Correctness was independently scored against generated gold.
We want transferable behavior: bounded evidence/strategy sub-calls, substantial work per turn, full data held in
Python, compact root observations, computation/reduction in Python, bounded repairs, prompt submission.
Sub-calls may interpret a representative source sample and devise the strategy; do not demand reading calls when
Python can parse the corpus. Do reject a greeting or an irrelevant/ignored planning call, repeated manual slice
reading, full reply dumps, avoidable extra verification or loops, fabricated inputs, or ungrounded hardcoded values.
The code may contain literal numbers read from actual prior source output; that is allowed, but prefer computation
over hand transcription. Do not require any particular library, parser, code length, exact program or separate
submission turn. Do not accept merely because it got the correct answer.
Return JSON only with these boolean fields: useful_delegation, efficient_turns, efficient_context, grounded,
and a short reason. Use false if the visible evidence is insufficient to establish a criterion."""

RECOVERY_RUBRIC = """
This run teaches error recovery and submission under a deadline. A recovered error is allowed: do not reject
solely for a traceback. If an execution or extraction error occurs, the next root action must explicitly own the
error, explain the failed assumption and say how it will avoid repeating it. Require a concrete change that
addresses the failure, reuse of validated work, and successful completion without the same error recurring.
Empty/all-zero extraction or inconsistent boundaries must trigger a changed approach, not blind retry.
Set recovery_behavior=false for a rote apology followed by the same broken approach. If no error occurs,
recovery_behavior=true is allowed, but the example is a completion demonstration, not a recovery demonstration.
Require prompt submission once an answer is available; near the turn limit the root must stop optional searching
and submit its best supported answer in executable answer-dict code. Do not reward invented evidence or an
unsupported default. This is a hard budget, not a reason to deliberately wait until the deadline.
Also return boolean fields recovery_behavior and timely_submission. Both must be true for acceptance.
Return observed_failure=true if an execution error, invalid reader output, empty extraction, or a wrong
assumption actually occurred, otherwise false. Return changed_approach=true only when a visible concrete repair
addresses that observed failure. Do not label a happy-path example as recovery just because the hint asks for it.
If sampled_prefix_replay is supplied, earlier sampled actions are context-only; assess the efficiency and repair
of the NEW continuation, not whether the already-failed prefix could have been written better.
For an observed failure, provide failure_evidence_quote (an exact short quote of the actual failed attempt or its
feedback) and acknowledgment_and_prevention_quote (an exact short quote from a root ASSISTANT action explicitly
owning the mistake and stating how it will avoid repeating it). A source/task mislabel corrected before any
failed attempt is not an observed failure. Generic format identification is not owning a mistake.
"""

CONTINUATION_RUBRIC = """
This is a focused continuation dataset: the entire original prefix is context-only and excluded from training
targets. The new target teaches recovery, not initial task routing. Return appropriate_delegation=true when
new subcalls are useful OR no new call is needed for a local repair (e.g. correcting an undefined variable).
Do not demand a gratuitous subcall for such a fix. Keep useful_delegation truthful (false if there is no useful
subcall); that field is not the acceptance gate for this focused continuation stream. A useless call still makes
appropriate_delegation=false. A real observed failure, explicit acknowledgment/prevention, changed approach,
grounded correctness and timely completion remain mandatory.
Your JSON must include ALL fields: useful_delegation, appropriate_delegation, efficient_turns, efficient_context,
grounded, recovery_behavior, timely_submission, observed_failure, changed_approach, failure_evidence_quote,
acknowledgment_and_prevention_quote, reason.
"""

DEADLINE_RUBRIC = """
FOCUS OVERRIDE: This is a near-deadline completion continuation, not necessarily an error-recovery case.
The replayed unfinished prefix consumes all but two root turns. No new ordinary happy-path rollout qualifies.
An observed failure and changed approach are NOT required when the prefix was clean. Set observed_failure=false
truthfully in that case. Set deadline_finish=true only if the NEW continuation uses the remaining evidence and
variables to finish in the next executed action, submitting through the answer dict instead of more optional
exploration or verification. Grounded correctness, efficient context/turns and appropriate delegation remain
mandatory. If it does encounter an error, explicit ownership/prevention and a changed approach are still required.
Return ALL continuation fields plus deadline_finish (boolean). This override replaces the earlier statement that
a real observed failure is mandatory. The full replayed prefix remains context-only, with no loss.
"""

FINAL_TURN_RUBRIC = """
FINAL-TURN ADMISSION OVERRIDE: the actual final executed turn is a valid submission opportunity. Prefer immediate
submission, but deadline_finish may be true for one decisive extraction/repair action followed by computation and
submission within the two remaining turns. Reject optional checking/continued exploration that wastes the budget.
Do not reject solely because a correct executable submission uses the final allowed turn. There is no extra
bailout call and no extension of the original limit. All grounding, efficiency and error-prevention rules remain.
"""


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


async def main(a):
    config = json.loads((a.run / "config.json").read_text())
    if config.get("data") != "procedural-only" or config.get("split") != "train":
        raise ValueError("only procedural training runs may produce SFT data")
    recovery_mode = config.get("recovery_curriculum", False)
    continuation_mode = bool(config.get("replay_prefix"))
    focus = config.get("focus_behavior", "backtrack")
    rubric = RUBRIC + (RECOVERY_RUBRIC if recovery_mode else "") + (CONTINUATION_RUBRIC if continuation_mode else "")
    if continuation_mode and focus == "deadline":
        rubric += DEADLINE_RUBRIC
    if config.get("allow_final_turn_submission"):
        rubric += FINAL_TURN_RUBRIC
    client, model = build_judge_client(model=a.judge_model, base_url=a.judge_base_url,
                                      **({"api_key": "local"} if a.judge_base_url else {}))
    review_path = a.run / "quality_reviews.json"
    previous = {v["row"]: v for v in json.loads(review_path.read_text())} if review_path.exists() else {}
    verdicts, accepted = {}, {}
    sem = asyncio.Semaphore(a.concurrency)

    async def one(path):
        sample = json.loads(path.read_text())
        meta, m = sample["meta"], sample["meta"]["metrics"]
        if meta.get("source") != "procedural-only" or meta.get("split") != "train":
            raise ValueError("candidate provenance mismatch")
        row = path.stem.split("_")[-1]
        reason = []
        if m["turns"] > a.max_turns:
            reason.append("turn_budget")
        if m["observation_chars"] > a.max_observation_chars:
            reason.append("observation_budget")
        trace = json.loads((a.run / f"trace_{row}.json").read_text())
        signature = hashlib.sha256(json.dumps({"sample": sample, "trace": trace, "rubric": rubric,
            "model": model, "max_turns": a.max_turns,
            "judge_base_url": a.judge_base_url, "judge_no_thinking": a.judge_no_thinking,
            "judge_json_schema": a.judge_json_schema, "judge_max_tokens": a.judge_max_tokens,
            "max_observation_chars": a.max_observation_chars}, sort_keys=True).encode()).hexdigest()
        cached = previous.get(int(row), {})
        verdict = {"row": int(row), "accepted": False, "model": model,
                   "mechanical_reasons": reason, "evidence_sha256": signature}
        if cached.get("evidence_sha256") == signature and not (a.retry_errors and cached.get("error")):
            verdict = cached
        elif not reason:
            transcript = [{"role": x["role"], "content": x["content"]} for x in sample["messages"]]
            subcalls = trace.get("complete_sub_calls") or trace["sub_calls"]
            request = json.dumps({"metrics": m, "root_teacher": meta.get("root_teacher", "sampled"),
                                  "sampled_prefix_replay": meta.get("sampled_prefix_replay"),
                                  "root_token_usage": meta.get("root_token_usage", "measured"),
                                  "root": transcript, "sub_calls": subcalls}, ensure_ascii=False)
            try:
                kw = {"model": model, "messages": [{"role": "system", "content": rubric},
                                                    {"role": "user", "content": request}],
                      "max_completion_tokens": a.judge_max_tokens}
                if re.search(r"(?:gpt-5|o[134])", model):
                    kw["reasoning_effort"] = "minimal"
                if a.judge_no_thinking:
                    kw["temperature"] = 0
                    kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                if a.judge_json_schema:
                    if not a.judge_base_url:
                        raise ValueError("structured review is only configured for an explicit local endpoint")
                    kw.setdefault("extra_body", {})["structured_outputs"] = {"json": review_json_schema()}
                async with sem:
                    response = await client.chat.completions.create(**kw)
                text = response.choices[0].message.content or ""
                verdict["response_text"] = text[:6000]
                verdict["finish_reason"] = response.choices[0].finish_reason
                verdict["response_reasoning_chars"] = len(getattr(response.choices[0].message, "reasoning_content", "") or "")
                verdict["usage"] = response.usage.model_dump() if response.usage else None
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
                jd = json.loads(raw)
                verdict["judge"] = jd
                verdict["usage"] = response.usage.model_dump() if response.usage else None
                verdict["accepted"] = quality_passes(jd, recovery=recovery_mode, continuation=continuation_mode, focus=focus)
                if recovery_mode:
                    verdict["accepted"] &= isinstance(jd.get("observed_failure"), bool)
                    if jd.get("observed_failure"):
                        verdict["accepted"] &= jd.get("changed_approach") is True
                        quote = jd.get("acknowledgment_and_prevention_quote", "").strip()
                        if len(quote) >= 2 and quote[0] == quote[-1] and quote[0] in "\"'":
                            quote = quote[1:-1]
                        assistant_texts = [m["content"] for m in transcript if m["role"] == "assistant"]
                        evidence = jd.get("failure_evidence_quote", "")
                        verdict["accepted"] &= bool(acknowledges_and_prevents(quote) and any(quote in t for t in assistant_texts)
                            and evidence and any(evidence in m["content"] for m in transcript))
            except Exception as e:
                verdict["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        if verdict["accepted"]:
            sample["meta"]["quality_review"] = {"model": model, **verdict["judge"]}
            accepted[int(row)] = sample
        verdicts[int(row)] = verdict
        print(json.dumps(verdict), flush=True)
        atomic_write(review_path, json.dumps([verdicts[k] for k in sorted(verdicts)], indent=2))
        atomic_write(a.run / "sft_reviewed.jsonl", "".join(
            json.dumps(accepted[k], ensure_ascii=False)+"\n" for k in sorted(accepted)))
    try:
        await asyncio.gather(*(one(path) for path in sorted(a.run.glob("candidate_*.json"))))
        if not verdicts:
            atomic_write(a.run / "sft_reviewed.jsonl", "")
            atomic_write(review_path, "[]")
    finally:
        await client.close()
    print(f"reviewed={len(verdicts)} accepted={len(accepted)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    p.add_argument("--judge-base-url", help="explicit local judge endpoint; avoids external fallback")
    p.add_argument("--judge-no-thinking", action="store_true")
    p.add_argument("--judge-json-schema", action="store_true")
    p.add_argument("--judge-max-tokens", type=int, default=1800)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--max-observation-chars", type=int, default=8000)
    args = p.parse_args()
    with (args.run / ".review.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        asyncio.run(main(args))
