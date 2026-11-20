"""The state store: one SQLite file (``<pipeline dir>/state.sqlite``) indexing every example."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, size INT, mtime_ns INT, offset INT, tail TEXT);
CREATE TABLE IF NOT EXISTS examples(
  env TEXT, uid TEXT, run_tag TEXT, run_dir TEXT, key TEXT, era TEXT,
  source TEXT, parent_uid TEXT, error_turn INT, live_from INT, category TEXT,
  task_sha1 TEXT, meta_sha1 TEXT, question TEXT, context_chars INT,
  bucket TEXT, reward REAL, stop TEXT, n_turns INT, n_sub_calls INT, quality INT, derived TEXT,
  gates TEXT, heuristic_reasons TEXT, judge_reason TEXT, hint_sha TEXT, hint_leak TEXT, invalid TEXT,
  has_messages INT, file TEXT, off INT, len INT, ts REAL,
  PRIMARY KEY(env, uid));
CREATE INDEX IF NOT EXISTS ex_task ON examples(env, meta_sha1);
CREATE INDEX IF NOT EXISTS ex_parent ON examples(env, parent_uid);
CREATE INDEX IF NOT EXISTS ex_file ON examples(file);
CREATE TABLE IF NOT EXISTS processed(run_dir TEXT, env TEXT, key TEXT, bucket TEXT, ts REAL, PRIMARY KEY(run_dir, env, key));
CREATE TABLE IF NOT EXISTS subcall_idx(file TEXT, key TEXT, off INT, len INT, n INT, PRIMARY KEY(file, key));
CREATE TABLE IF NOT EXISTS hints(sha TEXT PRIMARY KEY, text TEXT);
CREATE TABLE IF NOT EXISTS catalog(env TEXT, load_n INT, task_sha1 TEXT, meta_sha1 TEXT, context_sha1 TEXT, idx INT,
  question TEXT, context_chars INT, PRIMARY KEY(env, load_n, task_sha1));
CREATE TABLE IF NOT EXISTS resolved(env TEXT, uid TEXT, task_sha1 TEXT, how TEXT, PRIMARY KEY(env, uid));
CREATE TABLE IF NOT EXISTS rechecks(env TEXT, uid TEXT, condensed_output INT, reason TEXT, model TEXT, ts REAL,
  PRIMARY KEY(env, uid));
CREATE TABLE IF NOT EXISTS locations(env TEXT, uid TEXT, loc TEXT, ts REAL, PRIMARY KEY(env, uid));
CREATE TABLE IF NOT EXISTS branch_tried(env TEXT, uid TEXT, work_id TEXT, planned INT, ts REAL, PRIMARY KEY(env, uid));
CREATE TABLE IF NOT EXISTS work(id TEXT PRIMARY KEY, env TEXT, kind TEXT, writer TEXT, status TEXT, pid INT,
  server TEXT, started REAL, ended REAL, rc INT, item TEXT, log TEXT, stats TEXT);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""

EX_COLS = ("env", "uid", "run_tag", "run_dir", "key", "era", "source", "parent_uid", "error_turn", "live_from",
           "category", "task_sha1", "meta_sha1", "question", "context_chars", "bucket", "reward", "stop", "n_turns",
           "n_sub_calls", "quality", "derived", "gates", "heuristic_reasons", "judge_reason", "hint_sha", "hint_leak",
           "invalid", "has_messages", "file", "off", "len", "ts")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def q(self, sql: str, args: Iterable[Any] = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, tuple(args)).fetchall()

    def x(self, sql: str, args: Iterable[Any] = ()) -> None:
        self.db.execute(sql, tuple(args))

    def commit(self) -> None:
        self.db.commit()

    def get_meta(self, k: str, default: Any = None) -> Any:
        r = self.q("SELECT v FROM meta WHERE k=?", (k,))
        return json.loads(r[0]["v"]) if r else default

    def set_meta(self, k: str, v: Any) -> None:
        self.x("INSERT OR REPLACE INTO meta(k, v) VALUES(?, ?)", (k, json.dumps(v)))

    def upsert_example(self, ex: dict[str, Any]) -> None:
        vals = [ex.get(c) for c in EX_COLS]
        for i, c in enumerate(EX_COLS):
            if c in ("gates", "heuristic_reasons") and vals[i] is not None and not isinstance(vals[i], str):
                vals[i] = json.dumps(vals[i])
        self.x(f"INSERT OR REPLACE INTO examples({','.join(EX_COLS)}) VALUES({','.join('?' * len(EX_COLS))})", vals)

    def examples(self, env: str | None = None, where: str = "", args: Iterable[Any] = ()) -> list[dict]:
        sql = "SELECT * FROM examples"
        conds, a = [], list(args)
        if env:
            conds.append("env=?")
            a.insert(0, env)
        if where:
            conds.append(where)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        out = []
        for r in self.q(sql, a):
            d = dict(r)
            d["gates"] = json.loads(d["gates"]) if d.get("gates") else {}
            d["heuristic_reasons"] = json.loads(d["heuristic_reasons"]) if d.get("heuristic_reasons") else []
            out.append(d)
        return out

    def read_line(self, file: str, off: int, length: int) -> dict | None:
        try:
            with open(file, "rb") as f:
                f.seek(off)
                return json.loads(f.read(length))
        except (OSError, json.JSONDecodeError):
            return None

    def read_example(self, ex: dict) -> dict | None:
        return self.read_line(ex["file"], ex["off"], ex["len"])

    def read_subcalls(self, ex: dict) -> dict | None:
        sub = str(Path(ex["file"]).parent / "subcalls.jsonl")
        r = self.q("SELECT off, len FROM subcall_idx WHERE file=? AND key=?", (sub, ex["key"]))
        return self.read_line(sub, r[0]["off"], r[0]["len"]) if r else None

    def subcall_pointer(self, ex: dict) -> tuple[str, int, int] | None:
        sub = str(Path(ex["file"]).parent / "subcalls.jsonl")
        r = self.q("SELECT off, len FROM subcall_idx WHERE file=? AND key=?", (sub, ex["key"]))
        return (sub, r[0]["off"], r[0]["len"]) if r else None

    def add_work(self, w: dict[str, Any]) -> None:
        self.x("INSERT OR REPLACE INTO work(id, env, kind, writer, status, pid, server, started, ended, rc, item, log, stats) "
               "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (w["id"], w["env"], w["kind"], w["writer"], w.get("status", "running"), w.get("pid"), w.get("server"),
                w.get("started", time.time()), w.get("ended"), w.get("rc"), w.get("item"), w.get("log"),
                json.dumps(w.get("stats") or {})))

    def active_work(self) -> list[dict]:
        return [dict(r) for r in self.q("SELECT * FROM work WHERE status='running'")]

    def finish_work(self, wid: str, status: str, rc: int | None, stats: dict | None = None) -> None:
        self.x("UPDATE work SET status=?, rc=?, ended=?, stats=? WHERE id=?",
               (status, rc, time.time(), json.dumps(stats or {}), wid))
