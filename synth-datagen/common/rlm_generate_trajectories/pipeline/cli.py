"""CLI: python -m rlm_generate_trajectories.pipeline <command> <spec> [options]

  run      loop until every env reaches its target (resumable; adopts running items; --once = one pass)
  status   per-env progress / rejection reasons / running items / servers / rates (+ status.json)
  ingest   incremental ingestion only (pipeline outputs -> state.sqlite)
  plan     print what the next pass would launch (no launch)
  tasks    task instances of one env with their examples
  export   write the SFT set (+ sidecar sub-calls, manifest)
  stop     stop the orchestrator (and, with --items, its running work items)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import model as M
from .ingest import ingest
from .spec import load
from .store import Store


def _orchestrator_live(spec) -> bool:
    lock = spec.dir / "orchestrator.lock"
    try:
        d = json.loads(lock.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return time.time() - float(d.get("heartbeat", 0)) < max(180, 4 * spec.poll_s)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m rlm_generate_trajectories.pipeline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("spec")
    p.add_argument("--once", action="store_true", help="one pass (ingest, plan, launch) and exit")
    p.add_argument("--dry-run", action="store_true", help="plan and print launches, start nothing")
    p.add_argument("--max-items", type=int, default=None, help="launch at most N items in total, then exit when idle")
    p.add_argument("--envs", default=None, help="comma list: only these envs")
    p = sub.add_parser("status")
    p.add_argument("spec")
    p.add_argument("--json", action="store_true")
    p.add_argument("--ingest", action="store_true", help="ingest first even if an orchestrator is running")
    p.add_argument("--no-ingest", action="store_true")
    p.add_argument("--no-servers", action="store_true", help="skip the server health probe")
    p = sub.add_parser("ingest")
    p.add_argument("spec")
    p = sub.add_parser("plan")
    p.add_argument("spec")
    p.add_argument("--envs", default=None)
    p = sub.add_parser("tasks")
    p.add_argument("spec")
    p.add_argument("env")
    p = sub.add_parser("export")
    p.add_argument("spec")
    p.add_argument("--out", default=None)
    p = sub.add_parser("stop")
    p.add_argument("spec")
    p.add_argument("--items", action="store_true", help="also SIGTERM the running work items")
    a = ap.parse_args(argv)
    spec = load(a.spec)
    envs = [e.strip() for e in a.envs.split(",")] if getattr(a, "envs", None) else None

    if a.cmd == "run":
        from .orchestrator import Orchestrator
        Orchestrator(spec, dry_run=a.dry_run, max_items=a.max_items, envs=envs).run(once=a.once or a.dry_run)
        return
    if a.cmd == "stop":
        from .orchestrator import stop_items
        for m in stop_items(spec, orchestrator=True, items=a.items) or ["nothing to stop"]:
            print(m)
        return
    store = Store(spec.db_path)
    try:
        if a.cmd == "ingest":
            print(json.dumps(ingest(spec, store)))
        elif a.cmd == "status":
            from .orchestrator import server_health
            from .status import build_status, render, write
            if not a.no_ingest and (a.ingest or not _orchestrator_live(spec)):
                ingest(spec, store)
            health = {} if a.no_servers else server_health(spec.servers)
            prev = {}
            try:
                prev = json.loads((spec.dir / "status.json").read_text())
            except (OSError, json.JSONDecodeError):
                pass
            notes = {e: (v.get("notes") or {}) for e, v in (prev.get("envs") or {}).items()}
            st = build_status(spec, store, health=health, notes=notes)
            st["orchestrator_live"] = _orchestrator_live(spec)
            if not _orchestrator_live(spec):
                write(spec, st)
            print(json.dumps(st, indent=1, default=str) if a.json else render(st))
            if not st["orchestrator_live"]:
                print("\n(orchestrator not running on this spec)")
        elif a.cmd == "plan":
            from .planner import plan_env
            ingest(spec, store)
            exs = M.load_examples(store, spec)
            busy: dict[str, set] = {}
            for w in store.active_work():
                busy.setdefault(w["env"], set()).add(w["writer"])
            for env in spec.envs():
                if envs and env not in envs:
                    continue
                ps, notes = plan_env(spec, store, env, [e for e in exs if e["env"] == env], busy.get(env, set()))
                print(f"{env}: " + ("; ".join(f"{p.kind}({p.why})" for p in ps) or "nothing")
                      + (f"   [{notes.get('stalled')}]" if notes.get("stalled") else ""))
        elif a.cmd == "tasks":
            exs = M.load_examples(store, spec, a.env)
            for g, t in sorted(M.tasks_of(exs).items(), key=lambda kv: (-kv[1]["good"], -kv[1]["n"])):
                src = " ".join(f"{k}={v}" for k, v in t["by_source"].items())
                print(f"{g[:14]}  good {t['good']:2d}/{t['n']:3d}  {src:40s} ctx={t['context_chars']}  {t['question'][:90]}")
        elif a.cmd == "export":
            from .export import export
            if not _orchestrator_live(spec):
                ingest(spec, store)
            m = export(spec, store, a.out)
            print(json.dumps({k: m[k] for k in ("rows", "holdout", "rows_by_env", "rows_by_source", "dropped")}, indent=1))
    finally:
        store.close()


if __name__ == "__main__":
    main(sys.argv[1:])

