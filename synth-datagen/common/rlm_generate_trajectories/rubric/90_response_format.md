RESPOND WITH ONE JSON OBJECT AND NOTHING ELSE:
{"decomposition": "<tag>", "description": "<one sentence>", "subcalls_appropriate": true|false,
 "grounded": true|false, "clean_root": true|false, "parallel": true|false, "concise_subprompts": true|false,
 "recovers": true|false, "size_checked": true|false, "search_spam": true|false, "causal_ok": true|false,
 "blind_probe": true|false, "parse_from_evidence": true|false, "verified_before_submit": true|false,
 "check_independent": true|false, "reads_subcalls_critically": true|false, "legible_steps": true|false,
 "condensed_output": true|false,
 "improve": "<max 200 chars: the single most important change that would make this trajectory acceptable>",
 "wasted_turns": <int>, "quality": <1-5>, "novel": true|false,
 "reason": "<max 200 chars>"}
