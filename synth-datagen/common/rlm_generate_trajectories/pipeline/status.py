"""`pipeline status`: per-env progress, rejection reasons, running items, servers, rates. No judge calls."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import model as M
from .spec import Spec
from .store import Store


def _tail(path: str | None, n: int = 1) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4000))
            lines = [x for x in f.read().decode(errors="replace").splitlines() if x.strip()]
        return " | ".join(lines[-n:])[-240:]
    except OSError:
        return ""


def build_status(spec: Spec, store: Store, *, health: dict | None = None, notes: dict | None = None,
                 exs_all: list[dict] | None = None) -> dict[str, Any]:
    now = time.time()
    exs_all = exs_all if exs_all is not None else M.load_examples(store, spec)
    by_env: dict[str, list[dict]] = {}
    for e in exs_all:
        by_env.setdefault(e["env"], []).append(e)
    active = store.active_work()
    envs = {}
    for env in spec.envs():
        p = M.env_progress(spec, env, by_env.get(env, []), now)
        cat = store.q("SELECT COUNT(*) AS n FROM catalog WHERE env=? AND load_n=?",
                      (env, int(spec.env_param(env, "load_num_examples"))))[0]["n"]
        p["catalog_tasks"] = cat
        p["notes"] = (notes or {}).get(env, {})
        p["running"] = [{"id": w["id"], "kind": w["kind"], "server": w["server"], "age_min": round((now - w["started"]) / 60, 1),
                         "last_log": _tail(w["log"])} for w in active if w["env"] == env]
        envs[env] = p
    recent = store.q("SELECT id, env, kind, status, rc, ended, stats FROM work WHERE ended IS NOT NULL ORDER BY ended DESC LIMIT 12")
    tot_done = sum(1 for p in envs.values() if p["done"])
    return {
        "spec": str(spec.path), "name": spec.name, "dir": str(spec.dir), "generated_at": now,
        "fingerprint": spec.fingerprint(), "count_by": spec.count_by,
        "last_ingest": store.get_meta("last_ingest"),
        "totals": {"envs": len(envs), "envs_done": tot_done, "examples": len(exs_all),
                   "good": sum(e["verdict"] == "good" for e in exs_all),
                   "tasks_done": sum(min(p["tasks_done"], p["target"]) for p in envs.values()),
                   "tasks_target": sum(p["target"] for p in envs.values())},
        "envs": envs,
        "servers": health or {},
        "active": [{"id": w["id"], "env": w["env"], "kind": w["kind"], "server": w["server"],
                    "age_min": round((now - w["started"]) / 60, 1)} for w in active],
        "recent": [dict(r) for r in recent],
    }


def render(st: dict[str, Any]) -> str:
    L = []
    t = st["totals"]
    L.append(f"pipeline {st['name']}  ({st['dir']})")
    li = st.get("last_ingest")
    L.append(f"envs done {t['envs_done']}/{t['envs']} · task instances {t['tasks_done']}/{t['tasks_target']} · "
             f"examples {t['examples']} (good {t['good']}) · counted by {st['count_by']} · "
             f"ingested {'%.0f min ago' % ((time.time() - li) / 60) if li else 'never'}")
    L.append("")
    hdr = f"{'env':34s} {'tasks':>7s} {'cat':>4s} {'ex':>5s} {'good':>4s}  {'gen/bb/ba (good)':22s} {'new/h':>5s} {'eta_h':>5s}  running / note"
    L.append(hdr)
    L.append("-" * len(hdr))
    for env, p in sorted(st["envs"].items(), key=lambda kv: (kv[1]["done"], kv[1]["tasks_done"] - kv[1]["target"])):
        bs, gs = p["by_source"], p["good_by_source"]
        src = "/".join(str(bs.get(s, 0)) for s in M.SOURCES) + " (" + "/".join(str(gs.get(s, 0)) for s in M.SOURCES) + ")"
        run = ", ".join(f"{r['kind']}@{r['server']} {r['age_min']:.0f}m" for r in p["running"])
        note = p["notes"].get("stalled") or ""
        pend = p["verdicts"].get("pending-recheck", 0)
        if pend:
            note = (note + "; " if note else "") + f"{pend} pending recheck"
        mark = "✓" if p["done"] else " "
        eta = "-" if p["eta_h"] is None else f"{p['eta_h']:.1f}"
        L.append(f"{mark}{env[:33]:33s} {p['tasks_done']:>3d}/{p['target']:<3d} {p['catalog_tasks']:>4d} {p['examples']:>5d} "
                 f"{p['good']:>4d}  {src:22s} {p['rate']['new_tasks_per_h']:>5.1f} {eta:>5s}  {run or note}")
    L.append("")
    L.append("top rejection reasons per env (count):")
    for env, p in sorted(st["envs"].items()):
        if p["done"] or not p["top_reasons"]:
            continue
        L.append(f"  {env[:34]:34s} " + "; ".join(f"{n}x {k[:48]}" for k, n in p["top_reasons"][:3]))
    L.append("")
    sv = st.get("servers") or {}
    if sv:
        L.append("servers: " + "  ".join(f"{k}={'up' if v['up'] else 'DOWN'}" for k, v in sv.items()))
    L.append(f"active items: {len(st['active'])}" + ("" if not st["active"] else " -- " + ", ".join(
        f"{a['kind']}:{a['env']}@{a['server']}" for a in st["active"][:12])))
    rec = st.get("recent") or []
    fails = [r for r in rec if r["status"] == "failed"]
    if fails:
        L.append("recent failures: " + ", ".join(f"{r['id']} rc={r['rc']}" for r in fails[:6]))
    return "\n".join(L)


def write(spec: Spec, st: dict) -> Path:
    p = spec.dir / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, indent=1, default=str))
    return p
