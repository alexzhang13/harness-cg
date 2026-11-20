"""Incremental ingestion of generator-layout run dirs (and the pipeline's own side files) into the store."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import identity
from .spec import Source, Spec
from .store import Store

GATES = ("blind_probe", "parse_from_evidence", "verified_before_submit", "check_independent",
         "reads_subcalls_critically", "legible_steps", "condensed_output", "subcalls_appropriate", "grounded",
         "size_checked", "search_spam", "concise_subprompts", "causal_ok", "recovers")
SMALL_FILES = ("processed.jsonl", "error_locations.jsonl")
_TAIL = 4096


def _tail_sum(f, offset: int) -> str:
    start = max(0, offset - _TAIL)
    f.seek(start)
    return hashlib.sha1(f.read(offset - start)).hexdigest()


def read_new(store: Store, path: Path, *, whole: bool = False) -> tuple[bool, Iterator[tuple[int, int, dict]]] | None:
    """None if unchanged; else (reset, iterator of (offset, length, obj)) and the cursor is advanced once the
    iterator is exhausted. ``reset`` = the file must be treated as new (rewritten, or ``whole``)."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    p = str(path)
    rec = store.q("SELECT size, mtime_ns, offset, tail FROM files WHERE path=?", (p,))
    rec = dict(rec[0]) if rec else None
    if rec and rec["size"] == st.st_size and rec["mtime_ns"] == st.st_mtime_ns:
        return None
    f = open(path, "rb")
    start, reset = 0, True
    if rec and not whole and st.st_size >= rec["offset"] and rec["offset"] > 0 and _tail_sum(f, rec["offset"]) == rec["tail"]:
        start, reset = rec["offset"], False

    def it() -> Iterator[tuple[int, int, dict]]:
        f.seek(start)
        off = start
        try:
            for line in f:
                if not line.endswith(b"\n"):
                    break
                n = len(line)
                if line.strip():
                    try:
                        yield off, n, json.loads(line)
                    except json.JSONDecodeError:
                        pass
                off += n
            store.x("INSERT OR REPLACE INTO files(path, size, mtime_ns, offset, tail) VALUES(?,?,?,?,?)",
                    (p, st.st_size, st.st_mtime_ns, off, _tail_sum(f, off)))
        finally:
            f.close()
    return reset, it()


def _hint_leak(obj: dict, replay: dict | None) -> str | None:
    from ..sft_format import hint_reference
    from ..strategies import HINT_CLOSE, HINT_OPEN
    needles: list[str] = []
    if replay and isinstance(replay.get("location"), dict):
        from ..replay import leak_needles
        needles = leak_needles(replay["location"])
    from ..sft_format import task_text_of
    task_text = task_text_of(obj.get("messages") or [])
    for m in obj.get("messages") or []:
        c = str(m.get("content") or "")
        if HINT_OPEN in c or HINT_CLOSE in c or "strategy_hint" in c:
            return "hint marker in transcript"
        if any(n and n in c for n in needles):
            return "branch-note fragment in transcript"
        if m.get("role") == "assistant":
            r = hint_reference(c, task_text)
            if r:
                return f"assistant refers to the stripped hint ({r.group(0)!r})"
    return None


def example_from_row(obj: dict, *, bucket: str, env: str, src: Source, file: str, off: int, length: int,
                     tag_of_run: dict[str, str], hint_sha_default: str | None = None) -> dict[str, Any]:
    m = obj.get("meta") if bucket == "accepted" and isinstance(obj.get("meta"), dict) else obj
    key = m.get("key") or obj.get("key") or f"line@{off}"
    replay = m.get("replay") if isinstance(m.get("replay"), dict) else None
    judge = m.get("judge") if isinstance(m.get("judge"), dict) else {}
    q = judge.get("quality") if isinstance(judge.get("quality"), dict) else {}
    msgs = obj.get("messages")
    meta_text = identity.metadata_message(msgs)
    ex: dict[str, Any] = {
        "env": env, "uid": f"{src.tag}:{key}", "run_tag": src.tag, "run_dir": str(src.dir), "key": key, "era": src.era,
        "source": "generated", "bucket": bucket, "reward": m.get("reward"), "stop": m.get("stop_condition"),
        "n_turns": m.get("rlm_iterations"), "n_sub_calls": m.get("rlm_sub_llm_calls"), "quality": q.get("quality"),
        "derived": (judge.get("derived") or {}).get("verdict") if isinstance(judge.get("derived"), dict) else None,
        "gates": {g: q.get(g) for g in GATES if g in q}, "heuristic_reasons": list(obj.get("heuristic_reasons") or []),
        "judge_reason": obj.get("judge_reason"), "task_sha1": m.get("task_sha1"),
        "meta_sha1": identity.sha1(meta_text) if meta_text else None, "question": identity.question_snippet(meta_text),
        "context_chars": identity.context_chars(meta_text) or ((m.get("info") or {}).get("context_chars")
                                                               if isinstance(m.get("info"), dict) else None),
        "has_messages": int(bool(msgs)), "file": file, "off": off, "len": length, "hint_sha": hint_sha_default,
        "ts": None,
    }
    pl = m.get("pipeline") if isinstance(m.get("pipeline"), dict) else {}
    if pl.get("hint_sha1"):
        ex["hint_sha"] = pl["hint_sha1"]
    if replay:
        mode = replay.get("mode") or "before"
        ex["source"] = f"branch-{mode}"
        sk = str(replay.get("source_key") or "")
        if ":" in sk.split("@")[0]:
            ex["parent_uid"] = sk
        else:
            run = str(Path(str(replay.get("source_run") or "")).resolve()) if replay.get("source_run") else ""
            ex["parent_uid"] = f"{tag_of_run.get(run, '?' + Path(run).name)}:{sk}"
        ex["error_turn"] = replay.get("error_turn") or replay.get("from_step")
        ex["live_from"] = replay.get("from_step")
        loc = replay.get("location") if isinstance(replay.get("location"), dict) else {}
        ex["category"] = replay.get("category") or loc.get("category")
        if replay.get("hint"):
            ex["hint_sha"] = identity.sha1(replay["hint"])
            ex["_hint_text"] = replay["hint"]
    if bucket == "accepted":
        ex["hint_leak"] = _hint_leak(obj, replay)
        from ..export_sft import validate
        ex["invalid"] = validate(obj)
    return ex


class Ingestor:
    def __init__(self, spec: Spec, store: Store):
        self.spec, self.store = spec, store
        self.env_names = list(spec.envs())
        self.tag_of_run = {str(s.dir.resolve()): s.tag for s in spec.all_sources()}
        self._strategy_hints: dict[tuple[str, str], str | None] = {}
        self.counts: dict[str, int] = {}

    def _bump(self, k: str, n: int = 1) -> None:
        self.counts[k] = self.counts.get(k, 0) + n

    def _run_hint_sha(self, src: Source, env: str) -> str | None:
        """Hint of a source run's generated rows, from that run's strategies.json (latest launch; audit only)."""
        k = (str(src.dir), env)
        if k in self._strategy_hints:
            return self._strategy_hints[k]
        sha = None
        f = src.dir / "strategies.json"
        try:
            d = json.loads(f.read_text()).get(env)
            if d:
                from ..strategies import Strategy
                d = dict(d)
                d["families"] = tuple(d.get("families") or ())
                text = Strategy(**d).hint_text
                sha = identity.sha1(text)
                self.store.x("INSERT OR IGNORE INTO hints(sha, text) VALUES(?, ?)", (sha, text))
        except Exception:  # noqa: BLE001
            sha = None
        self._strategy_hints[k] = sha
        return sha

    def run(self) -> dict[str, int]:
        t0 = time.time()
        for src in self.spec.all_sources():
            if not src.dir.is_dir():
                continue
            for env in self.env_names:
                d = src.dir / env
                if d.is_dir():
                    self._ingest_stream_dir(src, env, d)
        self._ingest_side_files()
        self.store.set_meta("last_ingest", time.time())
        self.store.commit()
        self.counts["seconds"] = round(time.time() - t0, 2)
        return self.counts

    def _ingest_stream_dir(self, src: Source, env: str, d: Path) -> None:
        s = self.store
        got = read_new(s, d / "processed.jsonl", whole=True)
        if got:
            _, it = got
            s.x("DELETE FROM processed WHERE run_dir=? AND env=?", (str(src.dir), env))
            for _, _, o in it:
                if o.get("key"):
                    s.x("INSERT OR REPLACE INTO processed(run_dir, env, key, bucket, ts) VALUES(?,?,?,?,?)",
                        (str(src.dir), env, o["key"], o.get("bucket"), o.get("ts")))
        ts_of = {r["key"]: r["ts"] for r in s.q("SELECT key, ts FROM processed WHERE run_dir=? AND env=?", (str(src.dir), env))}
        pbucket = {r["key"]: r["bucket"] for r in s.q("SELECT key, bucket FROM processed WHERE run_dir=? AND env=?",
                                                        (str(src.dir), env))}
        hint_sha = self._run_hint_sha(src, env)
        for fname, bucket in (("accepted.jsonl", "accepted"), ("rejected.jsonl", "rejected"), ("interesting.jsonl", "interesting")):
            path = d / fname
            got = read_new(s, path)
            if not got:
                continue
            reset, it = got
            if reset:
                s.x("DELETE FROM examples WHERE file=?", (str(path),))
            for off, n, o in it:
                ex = example_from_row(o, bucket=bucket, env=env, src=src, file=str(path), off=off, length=n,
                                      tag_of_run=self.tag_of_run, hint_sha_default=hint_sha)
                ex["ts"] = ts_of.get(ex["key"]) or time.time()
                cur = s.q("SELECT bucket, file FROM examples WHERE env=? AND uid=?", (env, ex["uid"]))
                if cur and cur[0]["file"] != str(path):
                    want = pbucket.get(ex["key"])
                    keep_old = (cur[0]["bucket"] == want) if want else (cur[0]["bucket"] != "accepted")
                    if keep_old:
                        continue
                ht = ex.pop("_hint_text", None)
                if ht:
                    s.x("INSERT OR IGNORE INTO hints(sha, text) VALUES(?, ?)", (ex["hint_sha"], ht))
                r = s.q("SELECT task_sha1 FROM resolved WHERE env=? AND uid=?", (env, ex["uid"]))
                if not ex.get("task_sha1") and r:
                    ex["task_sha1"] = r[0]["task_sha1"]
                s.upsert_example(ex)
                self._bump(f"rows:{src.tag}")
        sub = d / "subcalls.jsonl"
        got = read_new(s, sub)
        if got:
            reset, it = got
            if reset:
                s.x("DELETE FROM subcall_idx WHERE file=?", (str(sub),))
            for off, n, o in it:
                if o.get("key"):
                    s.x("INSERT OR REPLACE INTO subcall_idx(file, key, off, len, n) VALUES(?,?,?,?,?)",
                        (str(sub), o["key"], off, n, o.get("n")))
        got = read_new(s, d / "error_locations.jsonl", whole=True)
        if got:
            _, it = got
            for _, _, o in it:
                if o.get("key") and isinstance(o.get("location"), dict):
                    s.x("INSERT OR IGNORE INTO locations(env, uid, loc, ts) VALUES(?,?,?,?)",
                        (env, f"{src.tag}:{o['key']}", json.dumps(o["location"]), o.get("ts")))
        got = read_new(s, d / "attempts.jsonl")
        if got:
            _, it = got
            for _, _, o in it:
                s.x("INSERT OR REPLACE INTO branch_tried(env, uid, work_id, planned, ts) VALUES(?,?,?,?,?)",
                    (env, o["uid"], o.get("work_id"), o.get("planned"), o.get("ts")))

    def _ingest_side_files(self) -> None:
        s, root = self.store, self.spec.dir
        for f in sorted((root / "catalog").glob("*.n*.jsonl")) if (root / "catalog").is_dir() else []:
            env, _, load = f.name[: -len(".jsonl")].rpartition(".n")
            got = read_new(s, f, whole=True)
            if not got:
                continue
            _, it = got
            s.x("DELETE FROM catalog WHERE env=? AND load_n=?", (env, int(load)))
            for _, _, o in it:
                s.x("INSERT OR REPLACE INTO catalog(env, load_n, task_sha1, meta_sha1, context_sha1, idx, question, context_chars) "
                    "VALUES(?,?,?,?,?,?,?,?)", (env, int(load), o["task_sha1"], o["meta_sha1"], o.get("context_sha1"),
                                                o.get("idx"), o.get("question"), o.get("context_chars")))
            self._bump("catalog_files")
        for f in sorted((root / "catalog").glob("*.resolved.jsonl")) if (root / "catalog").is_dir() else []:
            env = f.name[: -len(".resolved.jsonl")]
            got = read_new(s, f)
            if not got:
                continue
            for _, _, o in got[1]:
                if not o.get("task_sha1"):
                    s.x("INSERT OR IGNORE INTO resolved(env, uid, task_sha1, how) VALUES(?,?,?,?)",
                        (env, o["uid"], None, o.get("how")))
                    continue
                s.x("INSERT OR REPLACE INTO resolved(env, uid, task_sha1, how) VALUES(?,?,?,?)",
                    (env, o["uid"], o["task_sha1"], o.get("how")))
                s.x("UPDATE examples SET task_sha1=? WHERE env=? AND uid=? AND task_sha1 IS NULL",
                    (o["task_sha1"], env, o["uid"]))
        for f in sorted((root / "rechecks").glob("*.jsonl")) if (root / "rechecks").is_dir() else []:
            env = f.stem
            got = read_new(s, f)
            if not got:
                continue
            for _, _, o in got[1]:
                co = o.get("condensed_output")
                s.x("INSERT OR REPLACE INTO rechecks(env, uid, condensed_output, reason, model, ts) VALUES(?,?,?,?,?,?)",
                    (env, o["uid"], None if co is None else int(bool(co)), o.get("reason"), o.get("model"), o.get("ts")))
        for f in sorted((root / "locations").glob("*.jsonl")) if (root / "locations").is_dir() else []:
            env = f.stem
            got = read_new(s, f)
            if not got:
                continue
            for _, _, o in got[1]:
                if isinstance(o.get("location"), dict):
                    s.x("INSERT OR REPLACE INTO locations(env, uid, loc, ts) VALUES(?,?,?,?)",
                        (env, o["uid"], json.dumps(o["location"]), o.get("ts")))
        hd = root / "hints"
        if hd.is_dir():
            known = {r["sha"] for r in s.q("SELECT sha FROM hints")}
            for f in hd.glob("*.txt"):
                if f.stem not in known:
                    s.x("INSERT OR IGNORE INTO hints(sha, text) VALUES(?, ?)", (f.stem, f.read_text()))


def ingest(spec: Spec, store: Store) -> dict[str, int]:
    return Ingestor(spec, store).run()


def file_mtime(path: Path) -> float | None:
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None
