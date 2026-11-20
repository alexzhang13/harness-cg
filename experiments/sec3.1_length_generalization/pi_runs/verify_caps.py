"""Check that the saved training rollouts of a run respect the Pi limits.

    python verify_caps.py <run_dir> [--max-turns 30] [--max-calls 100] [--child-turns 10]

Checks root turns, sub-agent turns and calls, and that capped rollouts scored 0. Exits 1 on a violation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import settings as S


def subagent_tasks(messages: list[dict]) -> tuple[int, bool]:
    """(sub-agent tasks requested, whether a 'budget exhausted' reply appears). Pass prompt and completion."""
    asked, refused, seen = 0, False, set()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            call = json.loads(tc) if isinstance(tc, str) else tc
            if call.get("name") != "subagent" or call.get("id") in seen:
                continue
            seen.add(call.get("id"))
            args = call.get("arguments")
            args = json.loads(args) if isinstance(args, str) else (args or {})
            asked += len(args["tasks"]) if args.get("tasks") else len(args["chain"]) if args.get("chain") else 1
        if m.get("role") == "tool" and "Sub-agent budget exhausted" in json.dumps(m.get("content")):
            refused = True
    return asked, refused


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--max-turns", type=int, default=int(S.PI_ENV["PI_MAX_TURNS"]))
    ap.add_argument("--max-calls", type=int, default=int(S.PI_ENV["PI_SUBAGENT_MAX_CALLS"]))
    ap.add_argument("--child-turns", type=int, default=int(S.PI_ENV["PI_SUBAGENT_MAX_TURNS"]))
    a = ap.parse_args()

    files = sorted(Path(a.run).glob("run_default/rollouts/step_*/train_rollouts.jsonl"))
    n = viol = cap_hits = refused_n = skipped = 0
    max_root = max_sub = max_asked = 0
    for f in files:
        for line in open(f):
            r = json.loads(line)
            n += 1
            root, sub = int(r.get("pi_root_steps") or 0), int(r.get("pi_subagent_steps") or 0)
            asked, refused = subagent_tasks([*(r.get("prompt") or []), *(r.get("completion") or [])])
            spawned = min(asked, a.max_calls)
            max_root, max_sub, max_asked = max(max_root, root), max(max_sub, sub), max(max_asked, asked)
            problems = []
            if root > a.max_turns:
                problems.append(f"root steps {root} > {a.max_turns}")
            truncated = bool(r.get("is_truncated"))
            skipped += int(truncated and sub > 0)
            if sub > spawned * a.child_turns and not truncated:
                problems.append(f"sub-agent steps {sub} > {spawned} x {a.child_turns}")
            if asked > a.max_calls and not refused:
                problems.append(f"{asked} sub-agent tasks asked, none refused")
            if float(r.get("pi_turn_cap_hit") or 0) and float(r.get("reward") or 0) != 0:
                problems.append("cap hit but reward != 0")
            cap_hits += int(float(r.get("pi_turn_cap_hit") or 0) > 0)
            refused_n += int(refused)
            if problems:
                viol += 1
                print(f"VIOLATION {f.parent.name} example {r.get('example_id')}: {'; '.join(problems)}")
    print(f"{n} rollouts in {len(files)} steps | max root steps {max_root} (cap {a.max_turns}) | max sub-agent steps {max_sub} "
          f"| max sub-agent tasks asked {max_asked} (budget {a.max_calls}) | cap hits {cap_hits} | budget refusals {refused_n} "
          f"| sub-agent check skipped for {skipped} truncated samples | violations {viol}")
    sys.exit(1 if viol else 0)


if __name__ == "__main__":
    main()
