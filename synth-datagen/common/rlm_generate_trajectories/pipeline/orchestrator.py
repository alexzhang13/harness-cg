"""Pipeline run loop: ingest results, reap finished items, check servers, plan, and launch items into free slots.

Items run as separate processes in their own sessions, so a restarted orchestrator adopts them by pid.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from . import model as M
from .ingest import ingest
from .planner import Plan, plan_env
from .spec import REPO, Server, Spec

RUN_LOCAL = REPO / "synth-datagen/common/run_local.sh"
from .store import Store


def probe_live(servers: list[Server], health: dict[str, dict[str, Any]]) -> dict[str, float]:
    """running + waiting requests per server from vLLM /metrics; unreachable servers are left out."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    def one(s: Server) -> tuple[str, float | None]:
        node = (health.get(s.id) or {}).get("node") or s.host
        if not node:
            return s.id, None
        try:
            txt = urllib.request.urlopen(f"http://{node}:{s.port}/metrics", timeout=3).read().decode()
        except Exception:  # noqa: BLE001
            return s.id, None
        tot = 0.0
        for line in txt.splitlines():
            if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
                try:
                    tot += float(line.rsplit(" ", 1)[1])
                except ValueError:
                    pass
        return s.id, tot
    with ThreadPoolExecutor(max_workers=8) as ex:
        return {k: v for k, v in ex.map(one, servers) if v is not None}


def server_health(servers: list[Server]) -> dict[str, dict[str, Any]]:
    out = {}
    for s in servers:
        up = False
        try:
            with urllib.request.urlopen(f"http://{s.host}:{s.port}/health", timeout=4) as r:
                up = r.status == 200
        except Exception:  # noqa: BLE001
            up = False
        out[s.id] = {"port": s.port, "node": s.host, "up": up, "slots": s.slots}
    return out


def pid_alive(pid: int | None, needle: str) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        return needle in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return True


INGEST_SETTLE_S = 90

class Orchestrator:
    def __init__(self, spec: Spec, *, dry_run: bool = False, max_items: int | None = None, envs: list[str] | None = None):
        self.ended_at: dict[tuple[str, str], float] = {}
        self.spec, self.dry_run, self.max_items = spec, dry_run, max_items
        self.only_envs = envs
        spec.dir.mkdir(parents=True, exist_ok=True)
        for d in ("work", "logs", "catalog", "hints"):
            (spec.dir / d).mkdir(exist_ok=True)
        self.store = Store(spec.db_path)
        self.procs: dict[str, subprocess.Popen] = {}
        self.launched = 0
        self.lock = spec.dir / "orchestrator.lock"
        snap = spec.dir / "specs" / f"{spec.fingerprint()}.toml"
        snap.parent.mkdir(exist_ok=True)
        if not snap.exists():
            snap.write_text(Path(spec.path).read_text())
        (spec.dir / "spec.toml").write_text(Path(spec.path).read_text())
        self.spec_snapshot = snap

    def acquire(self) -> None:
        if self.lock.exists():
            try:
                d = json.loads(self.lock.read_text())
            except json.JSONDecodeError:
                d = {}
            fresh = time.time() - float(d.get("heartbeat", 0)) < max(180, 4 * self.spec.poll_s)
            if fresh and d.get("host") == socket.gethostname() and pid_alive(d.get("pid"), "pipeline"):
                raise SystemExit(f"another orchestrator is running (pid {d.get('pid')} on {d.get('host')}); "
                                 f"`pipeline stop` it or wait")
            if fresh and d.get("host") != socket.gethostname():
                raise SystemExit(f"orchestrator heartbeat from {d.get('host')} is fresh; refusing to start a second one")
        self.heartbeat()

    def heartbeat(self, extra: dict | None = None) -> None:
        tmp = self.lock.with_suffix(".tmp")
        tmp.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "heartbeat": time.time(),
                                   "spec": str(self.spec.path), **(extra or {})}))
        tmp.replace(self.lock)

    def release(self) -> None:
        try:
            d = json.loads(self.lock.read_text())
            if d.get("pid") == os.getpid():
                self.lock.unlink()
        except (OSError, json.JSONDecodeError):
            pass

    def reap(self) -> list[str]:
        """Mark finished items; returns ids that ended this pass."""
        ended = []
        for w in self.store.active_work():
            wid = w["id"]
            p = self.procs.get(wid)
            rc = p.poll() if p is not None else (None if pid_alive(w["pid"], wid) else -1)
            if rc is None:
                continue
            res_f = self.spec.dir / "work" / f"{wid}.result.json"
            stats = {}
            if res_f.exists():
                try:
                    stats = json.loads(res_f.read_text()).get("stats") or {}
                except json.JSONDecodeError:
                    pass
            ok = res_f.exists() and (rc in (0, -1))
            self.store.finish_work(wid, "done" if ok else "failed", rc, stats)
            self.ended_at[(w["env"], w["writer"])] = time.time()
            self.procs.pop(wid, None)
            ended.append(wid)
            print(f"[orch] {wid} {'done' if ok else 'FAILED'} rc={rc} {json.dumps(stats)[:300]}", flush=True)
        self.store.commit()
        return ended

    def _item_json(self, plan: Plan, wid: str, server: Server | None) -> dict:
        return {"id": wid, "kind": plan.kind, "env": plan.env, "spec": str(self.spec_snapshot),
                "spec_source": str(self.spec.path), "pipeline_dir": str(self.spec.dir), "name": self.spec.name, "writer": plan.writer,
                "base_url": server.local_url if server else None, "server": server.id if server else None,
                "lb_ports": sorted({x.port for x in self.spec.servers if server and x.host == server.host}),
                "lb_host": server.host if server else "",
                "load_n": int(self.spec.env_param(plan.env, "load_num_examples")), "why": plan.why,
                "spec_fingerprint": self.spec.fingerprint(), **plan.payload}

    def launch(self, plan: Plan, server: Server | None) -> str:
        wid = f"{time.strftime('%m%d-%H%M%S')}-{plan.kind}-{plan.env}"[:120]
        n = 0
        while (self.spec.dir / "work" / f"{wid}.json").exists():
            n += 1
            wid = f"{wid.rsplit('~', 1)[0]}~{n}"
        item = self._item_json(plan, wid, server)
        item_path = self.spec.dir / "work" / f"{wid}.json"
        if not self.dry_run:
            item_path.write_text(json.dumps(item))
        log = self.spec.dir / "logs" / f"{wid}.log"
        mod = "rlm_generate_trajectories.pipeline.worker"
        cmd = ["bash", str(RUN_LOCAL), mod, str(item_path)]
        env = dict(os.environ, RLM_TRAIN_DIR=str(REPO), PIPELINE_PYTHON=self.spec.python,
                   **{k: str(v) for k, v in (self.spec.env_param(plan.env, "env_vars") or {}).items()})
        if self.dry_run:
            print(f"[orch] DRY-RUN would launch {wid}: {' '.join(cmd)}  ({plan.why})", flush=True)
            return wid
        with open(log, "ab") as lf:
            p = subprocess.Popen(cmd, cwd=str(REPO), stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.procs[wid] = p
        self.store.add_work({"id": wid, "env": plan.env, "kind": plan.kind, "writer": plan.writer, "pid": p.pid,
                             "server": server.id if server else "local",
                             "item": str(item_path), "log": str(log)})
        self.store.commit()
        self.launched += 1
        print(f"[orch] launched {wid} on {server.id if server else 'local'}: {plan.why}", flush=True)
        return wid

    def step(self) -> dict[str, Any]:
        spec, store = self.spec, self.store
        now = time.time()
        self.reap()
        counts = ingest(spec, store)
        health = server_health(spec.servers)
        active = store.active_work()
        load: dict[str, int] = {s.id: 0 for s in spec.servers}
        busy: dict[str, set[str]] = {}
        for w in active:
            busy.setdefault(w["env"], set()).add(w["writer"])
            if (w["server"] in load and w["kind"] == "generate"
                    and not (spec.dir / "work" / f"{w['id']}.judging").exists()):
                load[w["server"]] += 1
        exs_all = M.load_examples(store, spec)
        by_env: dict[str, list[dict]] = {}
        for e in exs_all:
            by_env.setdefault(e["env"], []).append(e)
        envs = [e for e in spec.envs() if not self.only_envs or e in self.only_envs]
        plans: list[tuple[int, Plan]] = []
        env_notes, progress = {}, {}
        for env in envs:
            exs = by_env.get(env, [])
            progress[env] = M.env_progress(spec, env, exs, now)
            ps, notes = plan_env(spec, store, env, exs, busy.get(env, set()), now)
            env_notes[env] = notes
            pr = progress[env]
            deficit = (max(0, (pr.get("good_target") or 0) - (pr.get("good_capped") or 0))
                       + 1000 * max(0, pr["target"] - pr["tasks_done"])
                       + 100_000 * int(spec.env_param(env, "priority") or 0))
            plans += [(deficit, p) for p in ps]
        plans.sort(key=lambda x: -x[0])
        up_servers = [s for s in spec.servers if health.get(s.id, {}).get("up")]
        live = probe_live(up_servers, health)
        started = []
        node_items = sum(1 for w in active if w["kind"] == "catalog")
        for _, p in plans:
            if self.max_items is not None and self.launched >= self.max_items:
                break
            if p.writer in busy.get(p.env, set()):
                continue
            if time.time() - self.ended_at.get((p.env, p.writer), 0.0) < INGEST_SETTLE_S:
                continue
            if p.needs_port:
                free = [s for s in up_servers if load[s.id] < s.slots]
                if not free:
                    continue
                s = min(free, key=lambda s: (live.get(s.id, 0.0) / 64 + load[s.id] / s.slots, s.id))
                wid = self.launch(p, s)
                load[s.id] += 1
                live[s.id] = live.get(s.id, 0.0) + 24
            else:
                if not up_servers or node_items >= spec.max_node_items:
                    continue
                node_items += 1
                wid = self.launch(p, None)
            busy.setdefault(p.env, set()).add(p.writer)
            started.append(wid)
        all_done = all(progress[e]["done"] for e in envs)
        self.heartbeat({"ingest": counts, "started": started})
        from .status import build_status
        st = build_status(spec, store, health=health, notes=env_notes, exs_all=exs_all)
        (spec.dir / "status.json").write_text(json.dumps(st, indent=1, default=str))
        return {"all_done": all_done, "started": started, "active": len(store.active_work()), "ingest": counts}

    def run(self, once: bool = False) -> None:
        self.acquire()
        stop = {"flag": False}

        def _sig(*_):
            stop["flag"] = True
        signal.signal(signal.SIGTERM, _sig)
        signal.signal(signal.SIGINT, _sig)
        try:
            while not stop["flag"]:
                r = self.step()
                print(f"[orch {time.strftime('%H:%M:%S')}] active={r['active']} started={len(r['started'])} "
                      f"ingest={r['ingest']}", flush=True)
                if once or (r["all_done"] and r["active"] == 0 and self.spec.stop_when_done):
                    if r["all_done"]:
                        print("[orch] every env reached its target", flush=True)
                    break
                if self.max_items is not None and self.launched >= self.max_items and r["active"] == 0:
                    break
                for _ in range(int(self.spec.poll_s)):
                    if stop["flag"]:
                        break
                    time.sleep(1)
                    if any(p.poll() is not None for p in self.procs.values()):
                        break
        finally:
            self.release()
            self.store.close()
            print("[orch] exiting; running items keep running and are adopted by the next `pipeline run`", flush=True)


def stop_items(spec: Spec, *, orchestrator: bool = True, items: bool = False) -> list[str]:
    msgs = []
    lock = spec.dir / "orchestrator.lock"
    if orchestrator and lock.exists():
        try:
            d = json.loads(lock.read_text())
            if d.get("host") == socket.gethostname() and pid_alive(d.get("pid"), "pipeline"):
                os.kill(int(d["pid"]), signal.SIGTERM)
                msgs.append(f"sent SIGTERM to orchestrator pid {d['pid']}")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            msgs.append(f"orchestrator: {e}")
    if items:
        store = Store(spec.db_path)
        for w in store.active_work():
            if pid_alive(w["pid"], w["id"]):
                try:
                    os.killpg(os.getpgid(int(w["pid"])), signal.SIGTERM)
                    msgs.append(f"SIGTERM {w['id']} (pid {w['pid']})")
                except OSError as e:
                    msgs.append(f"{w['id']}: {e}")
        store.close()
    return msgs

