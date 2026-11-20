"""Derived view of the store: per-example verdict, task instances, per-env progress. Pure reads, no judge calls."""

from __future__ import annotations

import collections
import re
import time
from typing import Any

from . import identity
from .spec import Spec
from .store import Store

SOURCES = ("generated", "branch-before", "branch-after", "branch-reverify_fail", "branch-reverify_falsepass",
           "branch-unrepeat", "branch-repair")


def load_examples(store: Store, spec: Spec, env: str | None = None) -> list[dict]:
    sql = ("SELECT e.*, p.bucket AS pbucket, r.condensed_output AS rc, r.reason AS rc_reason FROM examples e "
           "LEFT JOIN processed p ON p.run_dir=e.run_dir AND p.env=e.env AND p.key=e.key "
           "LEFT JOIN rechecks r ON r.env=e.env AND r.uid=e.uid")
    args: tuple = ()
    if env:
        sql += " WHERE e.env=?"
        args = (env,)
    import json
    manual = manual_rejections(spec)
    out = []
    for r in store.q(sql, args):
        d = dict(r)
        d["gates"] = json.loads(d["gates"]) if d.get("gates") else {}
        d["heuristic_reasons"] = json.loads(d["heuristic_reasons"]) if d.get("heuristic_reasons") else []
        d["verdict"] = verdict(d)
        if d.get("era") == "prefix_only" and d["verdict"] == "good":
            d["verdict"] = "prefix-only"
        mr = manual.get((d["env"], d["uid"]))
        if mr is not None and d["verdict"] == "good":
            d["verdict"], d["manual_reason"] = "manual-reject", mr
        need = int(spec.env_param(d["env"], "require_sub_calls"))
        if need and d["verdict"] == "good" and (d.get("n_sub_calls") or 0) < need:
            d["verdict"] = "superseded"
        d["group"] = identity.group_key(spec.count_by, d.get("task_sha1"), d.get("meta_sha1"))
        d["task_key"] = identity.task_key(d.get("task_sha1"), d.get("meta_sha1"))
        out.append(d)
    _depths(out)
    return out


def manual_rejections(spec: Spec) -> dict[tuple[str, str], str]:
    """Rows rejected by manual review (<dir>/manual_reject/*.jsonl); they no longer count or get exported."""
    import json
    out: dict[tuple[str, str], str] = {}
    d = spec.dir / "manual_reject"
    for f in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
        for line in f.read_text().splitlines():
            try:
                o = json.loads(line)
                out[(o["env"], o["uid"])] = str(o.get("reason") or f.stem)
            except Exception:  # noqa: BLE001
                continue
    return out


def verdict(ex: dict) -> str:
    if ex.get("bucket") != "accepted" or (ex.get("pbucket") and ex["pbucket"] != "accepted"):
        return "rejected"
    if ex.get("hint_leak"):
        return "leak"
    if ex.get("invalid"):
        return "invalid"
    g = ex.get("gates") or {}
    cond = g.get("condensed_output")
    if cond is None:
        if ex.get("rc") is None:
            return "pending-recheck"
        return "good" if ex["rc"] else "recheck-failed"
    return "good" if cond is True else "rejected"


def _depths(exs: list[dict]) -> None:
    by = {(e["env"], e["uid"]): e for e in exs}

    def depth(e: dict, seen: int = 0) -> int:
        if e.get("source") == "generated" or seen > 8:
            return 0
        p = by.get((e["env"], e.get("parent_uid")))
        return 1 + (depth(p, seen + 1) if p else 0)
    for e in exs:
        e["depth"] = depth(e)


def reason_key(s: str) -> str:
    return re.sub(r"\d+(\.\d+)?", "N", str(s))[:70]


def rejection_reasons(ex: dict) -> list[str]:
    v = ex["verdict"]
    if v == "good":
        return []
    if v == "pending-recheck":
        return ["pending condensed_output re-check"]
    if v == "recheck-failed":
        return [f"judge: condensed_output = False (re-check){': ' + ex['rc_reason'] if ex.get('rc_reason') else ''}"]
    if v == "manual-reject":
        return [f"manual review: {ex.get('manual_reason')}"]
    if v == "superseded":
        return [f"superseded: {ex.get('n_sub_calls') or 0} sub-calls < require_sub_calls (delegation required)"]
    if v == "leak":
        return [f"hint leak: {ex.get('hint_leak')}"]
    if v == "invalid":
        return [f"invalid: {ex.get('invalid')}"]
    out = list(ex.get("heuristic_reasons") or [])
    if ex.get("judge_reason"):
        out.append(ex["judge_reason"])
    return out or ["(no reason recorded)"]


def tasks_of(exs: list[dict]) -> dict[str, dict]:
    """group key -> task instance aggregate."""
    t: dict[str, dict] = {}
    for e in exs:
        g = e.get("group")
        if not g or e.get("era") == "prefix_only":
            continue
        a = t.setdefault(g, {"group": g, "question": e.get("question") or "", "context_chars": e.get("context_chars"),
                             "task_keys": set(), "n": 0, "good": 0, "by_source": collections.Counter(),
                             "good_by_source": collections.Counter(), "gen_attempts": 0, "uids": []})
        a["task_keys"].add(e["task_key"])
        a["n"] += 1
        a["uids"].append(e["uid"])
        a["by_source"][e["source"]] += 1
        if e["source"] == "generated":
            a["gen_attempts"] += 1
        if e["verdict"] == "good":
            a["good"] += 1
            a["good_by_source"][e["source"]] += 1
    return t


def env_done(spec: Spec, env: str, exs: list[dict], tasks: dict[str, dict] | None = None) -> tuple[bool, str, int]:
    """(done, why, capped good count). Capped good = sum over tasks of min(good, max_examples_per_task)."""
    tasks = tasks if tasks is not None else tasks_of(exs)
    per = int(spec.env_param(env, "examples_per_task"))
    cap = int(spec.env_param(env, "max_examples_per_task"))
    want = int(spec.env_param(env, "good_examples"))
    budget = int(spec.env_param(env, "max_examples_per_env"))
    n_tasks = sum(1 for a in tasks.values() if a["good"] >= per)
    capped = sum(min(a["good"], cap) for a in tasks.values())
    if n_tasks >= spec.env_target(env) and (want <= 0 or capped >= want):
        return True, "target", capped
    own = sum(1 for e in exs if e.get("era") != "prefix_only")
    if budget > 0 and own >= budget:
        return True, f"budget ({own} >= {budget} examples)", capped
    return False, "", capped


def env_progress(spec: Spec, env: str, exs: list[dict], now: float | None = None) -> dict[str, Any]:
    now = now or time.time()
    target = spec.env_target(env)
    per = int(spec.env_param(env, "examples_per_task"))
    tasks = tasks_of(exs)
    done_tasks = [g for g, a in tasks.items() if a["good"] >= per]
    by_source = collections.Counter(e["source"] for e in exs)
    good_by_source = collections.Counter(e["source"] for e in exs if e["verdict"] == "good")
    verdicts = collections.Counter(e["verdict"] for e in exs)
    reasons = collections.Counter(reason_key(r) for e in exs if e["verdict"] != "good" for r in rejection_reasons(e))
    gates: dict[str, list[int]] = {}
    for e in exs:
        for g, v in (e.get("gates") or {}).items():
            if v is None:
                continue
            ok = (v is False) if g == "search_spam" else (v is True)
            p = gates.setdefault(g, [0, 0])
            p[0] += int(ok)
            p[1] += 1
    win = 3 * 3600
    recent = [e for e in exs if (e.get("ts") or 0) >= now - win]
    first_good: dict[str, float] = {}
    for e in exs:
        if e["verdict"] == "good" and e.get("group"):
            first_good[e["group"]] = min(first_good.get(e["group"], 1e18), e.get("ts") or now)
    new_tasks_recent = sum(1 for ts in first_good.values() if ts >= now - win)
    rate = new_tasks_recent / (win / 3600)
    done, done_why, capped = env_done(spec, env, exs, tasks)
    remaining = max(0, target - len(done_tasks))
    return {
        "env": env, "target": target, "examples_per_task": per,
        "tasks_done": len(done_tasks), "tasks_seen": len(tasks), "done": done, "done_why": done_why,
        "good_capped": capped, "good_target": int(spec.env_param(env, "good_examples")),
        "examples": len(exs), "good": verdicts.get("good", 0), "verdicts": dict(verdicts),
        "by_source": dict(by_source), "good_by_source": dict(good_by_source),
        "top_reasons": reasons.most_common(6), "gates": {g: {"pass": p, "n": n} for g, (p, n) in gates.items()},
        "rate": {"examples_per_h": round(len(recent) / (win / 3600), 2),
                 "good_per_h": round(sum(e["verdict"] == "good" for e in recent) / (win / 3600), 2),
                 "new_tasks_per_h": round(rate, 2)},
        "eta_h": 0.0 if remaining == 0 else (round(remaining / rate, 1) if rate > 0 else None),
    }
