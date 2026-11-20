"""Opt-in recovery curriculum checks; never changes REPL execution semantics."""
from __future__ import annotations

import ast
import re


def message_value(message, key):
    return message.get(key, "") if isinstance(message, dict) else getattr(message, key, "")


def acknowledges_and_prevents(content):
    owns = re.search(r"\b(?:I|my|we)\b.{0,100}\b(?:error|mistake|incorrect|wrong|failed)\b", content, re.I | re.S)
    prevention = re.search(r"\b(?:avoid|next time|from now on|won't repeat|will not repeat|not repeat)\b", content, re.I)
    return bool(owns and prevention)


def failed_assistant_turns(messages):
    """Bad actions stay visible as context but must not become loss targets."""
    turn, failed = -1, set()
    for message in messages:
        role, content = message_value(message, "role"), message_value(message, "content") or ""
        if role == "assistant":
            turn += 1
        elif role == "user" and content.startswith("REPL output") and "Traceback (most recent call last)" in content:
            if turn >= 0:
                failed.add(turn)
    return sorted(failed)


def recovery_checks(messages, turns, turn_limit, *, reserve_turn=True):
    """Mechanical checks complement semantic review, which must verify actual repair."""
    reasons, pending, signatures = [], False, set()
    error_count = 0
    failed_turns = failed_assistant_turns(messages)
    assistant_turn = -1
    for message in messages:
        role, content = message_value(message, "role"), message_value(message, "content") or ""
        if role == "user" and content.startswith("REPL output") and "Traceback (most recent call last)" in content:
            error_count += 1
            matches = re.findall(r"^([A-Za-z]+Error:.*)$", content, re.M)
            signature = matches[-1] if matches else content.split("REPL variables:")[0]
            if signature in signatures:
                reasons.append("repeated_same_error")
            signatures.add(signature)
            pending = True
        elif role == "assistant":
            assistant_turn += 1
            if pending:
                if not acknowledges_and_prevents(content):
                    reasons.append("missing_error_acknowledgment_or_prevention")
                pending = False
            blocks = re.findall(r"```repl\s*\n(.*?)```", content, re.S)
            if not blocks or re.search(r"<tool_call\b|<function\b", content):
                reasons.append("invalid_repl_protocol")
            for block in blocks:
                try:
                    ast.parse(block)
                except SyntaxError:
                    if assistant_turn not in failed_turns:
                        reasons.append("invalid_python")
    if pending:
        reasons.append("unrecovered_final_error")
    if turns >= turn_limit and reserve_turn:
        reasons.append("no_submission_reserve")
    if turns > turn_limit:
        reasons.append("exceeded_turn_limit")
    return {"repl_error_events": error_count, "failed_target_turns": failed_turns,
            "turn_limit": turn_limit, "submitted_with_reserve": turns < turn_limit,
            "within_turn_limit": turns <= turn_limit,
            "mechanical_reasons": sorted(set(reasons))}
