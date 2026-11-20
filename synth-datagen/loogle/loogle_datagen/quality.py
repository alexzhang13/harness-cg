"""Keep full delegation examples distinct from context-only-prefix repair targets."""
import copy


def review_json_schema():
    booleans = ("useful_delegation", "appropriate_delegation", "efficient_turns", "efficient_context", "grounded",
                "recovery_behavior", "timely_submission", "observed_failure", "changed_approach", "deadline_finish")
    strings = ("failure_evidence_quote", "acknowledgment_and_prevention_quote", "reason")
    properties = {key: {"type": "boolean"} for key in booleans}
    properties.update({key: {"type": "string"} for key in strings})
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def require_rollout_models(meta, allowed):
    if allowed and (meta.get("model") not in allowed or meta.get("reader_model") not in allowed):
        raise ValueError("root/child rollout model violates the requested model policy")


def prepare_reasoning_export(sample, *, drop_reasoning=False):
    if sample["meta"].get("reasoning_hint_leak") and not drop_reasoning:
        raise ValueError("privileged hint leaked into reasoning; export must drop reasoning")
    if not drop_reasoning:
        return sample
    clean = copy.deepcopy(sample)
    removed = sum(bool(m.get("reasoning_content")) for m in clean["messages"])
    for key in ("messages", "prompt", "completion"):
        for message in clean.get(key, []):
            message.pop("reasoning_content", None)
    clean["meta"]["reasoning_export_policy"] = "drop"
    clean["meta"]["reasoning_turns_removed"] = removed
    return clean


def quality_passes(review, *, recovery=False, continuation=False, focus="backtrack"):
    keys = ["efficient_turns", "efficient_context", "grounded",
            "appropriate_delegation" if continuation else "useful_delegation"]
    if recovery:
        keys += ["recovery_behavior", "timely_submission"]
    if continuation:
        keys += ["deadline_finish"] if focus == "deadline" else ["observed_failure", "changed_approach"]
    return all(review.get(key) is True for key in keys)
