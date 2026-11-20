"""`pipeline export`: write every good example to ``<dir>/export/{train,subcalls}.jsonl`` with the spec's caps."""

from __future__ import annotations

import collections
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

from . import model as M
from .ingest import _hint_leak
from .spec import REPO, Spec
from .store import Store


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def export(spec: Spec, store: Store, out: Path | None = None) -> dict[str, Any]:
    from ..export_sft import validate
    out = Path(out or (spec.dir / "export"))
    out.mkdir(parents=True, exist_ok=True)
    exs = [e for e in M.load_examples(store, spec) if e["verdict"] == "good" and e["source"] in spec.export_sources
           and e["env"] in spec.envs()]
    ctx_of = {r["task_sha1"]: r["context_sha1"] for r in store.q("SELECT task_sha1, context_sha1 FROM catalog")}
    cat_of: dict[str, set] = collections.defaultdict(set)
    for r in store.q("SELECT env, task_sha1 FROM catalog"):
        cat_of[r["env"]].add(r["task_sha1"])
    rng = random.Random(spec.export_seed)
    rng.shuffle(exs)
    exs.sort(key=lambda e: (-(e.get("quality") or 0), e["source"] != "generated"))
    dropped: collections.Counter = collections.Counter()
    seen: set[str] = set()
    n_task: collections.Counter = collections.Counter()
    n_ctx: collections.Counter = collections.Counter()
    n_env: collections.Counter = collections.Counter()
    rows: list[tuple[dict, dict, dict | None]] = []
    for e in exs:
        if cat_of.get(e["env"]) and e.get("task_sha1") and e["task_sha1"] not in cat_of[e["env"]]:
            dropped["task left the catalogue (e.g. CTC split v1 leak)"] += 1
            continue
        r = store.read_example(e)
        if r is None:
            dropped["unreadable row"] += 1
            continue
        why = validate(r) or _hint_leak(r, (r.get("meta") or {}).get("replay") if isinstance((r.get("meta") or {}).get("replay"), dict) else None)
        if spec.rubric == "reverify":
            from ..reverify import broken_code_turns
            bt = broken_code_turns(r.get("messages") or [])
            r.setdefault("meta", {})["broken_code"], r["meta"]["broken_code_turns"] = bool(bt), bt
        if why:
            dropped[f"invalid: {why}"[:80]] += 1
            continue
        d = hashlib.sha256(json.dumps(r.get("messages"), sort_keys=True).encode()).hexdigest()
        if d in seen:
            dropped["duplicate transcript"] += 1
            continue
        tk = (e["env"], e["group"])
        if spec.export_max_per_task and n_task[tk] >= spec.export_max_per_task:
            dropped["over max_per_task"] += 1
            continue
        ck = (e["env"], ctx_of.get(e.get("task_sha1") or "") or e["group"])
        if spec.export_max_per_context and n_ctx[ck] >= spec.export_max_per_context:
            dropped["over max_per_context"] += 1
            continue
        if spec.export_max_per_env and n_env[e["env"]] >= spec.export_max_per_env:
            dropped["over max_per_env"] += 1
            continue
        seen.add(d)
        n_task[tk] += 1
        n_ctx[ck] += 1
        n_env[e["env"]] += 1
        meta = {"uid": e["uid"], "env": e["env"], "task": e["group"], "task_sha1": e.get("task_sha1"),
                "source": e["source"], "parent_uid": e.get("parent_uid"), "run": e["run_tag"], "era": e.get("era"),
                "reward": e.get("reward"), "quality": e.get("quality"), "n_turns": e.get("n_turns"),
                "n_sub_calls": e.get("n_sub_calls")}
        rm = r.get("meta") if isinstance(r.get("meta"), dict) else {}
        meta["reasoning_turns"] = sum(1 for m in r.get("messages") or [] if m.get("reasoning_content"))
        meta["reasoning_hint_leak"] = rm.get("reasoning_hint_leak")
        if rm.get("reverify") is not None:
            meta["reverify"] = rm["reverify"]
        if "broken_code" in rm:
            meta["broken_code"], meta["broken_code_turns"] = rm["broken_code"], rm.get("broken_code_turns") or []
        rows.append(({"messages": r["messages"], "prompt": r.get("prompt"), "completion": r.get("completion"), "meta": meta,
                      "broken_code": bool((meta or {}).get("broken_code"))},
                     e, store.read_subcalls(e)))
    train, hold = [], []
    by_env: dict[str, list] = collections.defaultdict(list)
    for x in rows:
        by_env[x[1]["env"]].append(x)
    for xs in by_env.values():
        nh = int(len(xs) * spec.export_holdout)
        hold += xs[:nh]
        train += xs[nh:]
    rng.shuffle(train)

    def write(path: Path, xs: list) -> None:
        with open(path, "w") as f:
            for row, _, _ in xs:
                f.write(json.dumps(row) + "\n")
    write(out / "train.jsonl", train)
    if hold:
        write(out / "holdout.jsonl", hold)
    with open(out / "subcalls.jsonl", "w") as f:
        for _, e, sub in train + hold:
            if sub:
                f.write(json.dumps({"uid": e["uid"], "env": e["env"], "n": sub.get("n"), "sub_calls": sub.get("sub_calls")}) + "\n")
    manifest = {
        "spec": str(spec.path), "fingerprint": spec.fingerprint(), "git": _git_rev(), "ts": time.time(),
        "rows": len(train), "holdout": len(hold),
        "caps": {"max_per_task": spec.export_max_per_task, "max_per_context": spec.export_max_per_context,
                 "max_per_env": spec.export_max_per_env},
        "rows_by_env": dict(collections.Counter(e["env"] for _, e, _ in train)),
        "rows_by_source": dict(collections.Counter(e["source"] for _, e, _ in train)),
        "tasks_by_env": {env: len({e["group"] for _, e, _ in train if e["env"] == env}) for env in by_env},
        "dropped": dict(dropped),
        "k_mix": None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
