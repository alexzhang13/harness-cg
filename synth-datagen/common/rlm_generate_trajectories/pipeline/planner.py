"""Plan the next work for one env from the store: ``catalog`` (list task instances once) or ``generate`` (tasks with
no good example yet, fewest attempts first)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from dataclasses import dataclass, field
from typing import Any

from . import identity
from . import model as M
from .model import tasks_of
from .spec import Spec
from .store import Store

FAIL_WINDOW_S = 1800
FAIL_LIMIT = 3


@dataclass
class Plan:
    env: str
    kind: str
    writer: str
    needs_port: bool
    payload: dict[str, Any] = field(default_factory=dict)
    why: str = ""


def recent_failures(store: Store, env: str, kind: str, now: float) -> int:
    rows = store.q("SELECT status, ended FROM work WHERE env=? AND kind=? AND ended IS NOT NULL ORDER BY ended DESC LIMIT ?",
                   (env, kind, FAIL_LIMIT))
    n = 0
    for r in rows:
        if r["status"] == "failed" and (r["ended"] or 0) >= now - FAIL_WINDOW_S:
            n += 1
        else:
            break
    return n


def catalog_rows(store: Store, spec: Spec, env: str) -> list[dict]:
    load_n = int(spec.env_param(env, "load_num_examples"))
    return [dict(r) for r in store.q("SELECT * FROM catalog WHERE env=? AND load_n=? ORDER BY idx", (env, load_n))]


def plan_env(spec: Spec, store: Store, env: str, exs: list[dict], busy_writers: set[str],
             now: float | None = None) -> tuple[list[Plan], dict[str, Any]]:
    """(plans in priority order, notes for status). ``exs`` = model.load_examples(...) for this env."""
    now = now or time.time()
    notes: dict[str, Any] = {}
    plans: list[Plan] = []
    cap = int(spec.env_param(env, "max_examples_per_task"))
    tasks = tasks_of(exs)
    notes["done"], why, notes["good_capped"] = M.env_done(spec, env, exs, tasks)
    if notes["done"] and why != "target":
        notes["stalled"] = why
    want_more = int(spec.env_param(env, "good_examples")) > 0

    cat = catalog_rows(store, spec, env)
    if not cat and "catalog" not in busy_writers and recent_failures(store, env, "catalog", now) < FAIL_LIMIT:
        plans.append(Plan(env, "catalog", "catalog", False, {"build": True}, "build the task catalogue"))
    if notes["done"] and spec.stop_when_done:
        return plans, notes

    from .spec import GEN_WRITERS
    gen_plans: list[Plan] = []
    n_writers = max(1, min(len(GEN_WRITERS), int(spec.env_param(env, "gen_writers") or 1)))
    free = [w for w in GEN_WRITERS[:n_writers] if w not in busy_writers]
    if free and cat and recent_failures(store, env, "generate", now) < FAIL_LIMIT:
        held: set[str] = set()
        for w in store.active_work():
            if w["env"] == env and w["kind"] == "generate":
                try:
                    raw = str(w["item"] or "{}")
                    it = json.loads(raw) if raw.lstrip().startswith("{") else json.loads(Path(raw).read_text())
                    held |= {t["task_sha1"] for t in (it.get("tasks") or [])}
                except (OSError, ValueError, TypeError, KeyError):
                    pass
        by_group: dict[str, dict] = {}
        for c in cat:
            g = identity.group_key(spec.count_by, c["task_sha1"], c["meta_sha1"])
            by_group.setdefault(g, c)
        max_att = int(spec.env_param(env, "max_attempts_per_task"))
        pool = []
        for g, c in by_group.items():
            a = tasks.get(g) or {}
            good = a.get("good", 0)
            if (good >= (cap if want_more else 1)) or a.get("gen_attempts", 0) >= max_att or c["task_sha1"] in held:
                continue
            pool.append((good > 0, a.get("gen_attempts", 0), c["idx"], c, g))
        pool.sort(key=lambda x: (x[0], x[1], x[2]))
        notes["catalog_tasks"] = len(by_group)
        notes["untried_or_retryable"] = len(pool)
        n = int(spec.env_param(env, "tasks_per_item"))
        gen_prior = {}
        for e in exs:
            if e["source"] == "generated" and e["run_tag"] in GEN_WRITERS and e.get("task_sha1"):
                gen_prior[e["task_sha1"]] = gen_prior.get(e["task_sha1"], 0) + 1
        for wi, writer in enumerate(free):
            chosen = pool[wi * n:(wi + 1) * n]
            if not chosen:
                break
            gen_plans.append(Plan(env, "generate", writer, True,
                            {"tasks": [{"task_sha1": c["task_sha1"], "attempt": gen_prior.get(c["task_sha1"], 0)}
                                       for _, _, _, c, _ in chosen],
                             "rollouts_per_task": int(spec.env_param(env, "rollouts_per_task"))},
                            f"{len(chosen)} task instances ({sum(1 for x in pool if not x[0])} without a good example left, "
                            f"{sum(1 for x in pool if x[0])} below max_examples_per_task)"))
        if not gen_plans and not pool:
            notes["stalled"] = (f"catalogue exhausted: all {len(by_group)} task instances are good or at "
                                f"max_attempts_per_task={max_att}; raise load_num_examples / max_attempts")
    plans += gen_plans
    return plans, notes
