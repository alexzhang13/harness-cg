"""Run one pipeline work item (generate or catalog). Started by the orchestrator via ``run_local.sh``.

    python -m rlm_generate_trajectories.pipeline.worker <pipeline_dir>/work/<id>.json

Writes ``<dir>/work/<id>.result.json`` when finished. Both kinds are idempotent.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from . import identity
from .spec import Spec
from .spec import load as load_spec


def _info(row: dict) -> dict:
    info = row.get("info")
    if isinstance(info, str):
        try:
            return json.loads(info)
        except json.JSONDecodeError:
            return {}
    return info or {}


def _dataset_rows(env: Any, st: Any) -> list[dict]:
    ds = env.get_dataset(n=-1) if st.split == "train" else env.get_eval_dataset(n=-1)
    return ds.to_list() if hasattr(ds, "to_list") else list(ds)


def gen_args(spec: Spec, base_url: str, config: Path, section: str, env: str | None = None) -> Any:
    """The generator's argparse Namespace, filled from the spec."""
    from ..generator import build_parser
    a = build_parser().parse_args([
        "--config", str(config), "--section", section, "--base-url", base_url, "--model", spec.model,
        "--hint-mode", spec.hint_mode, "--rubric", spec.rubric, "--temperature", str(spec.temperature),
        "--max-completion-tokens", str(spec.max_completion_tokens),
        "--chat-template-kwargs", json.dumps(spec.chat_template_kwargs), "--max-concurrent", str(spec.env_param(env, "max_concurrent") if env else spec.max_concurrent),
        "--chunk-size", str(spec.env_param(env, "chunk_size") if env else spec.chunk_size),
        "--request-timeout", str(spec.env_param(env, "request_timeout") if env else spec.request_timeout),
        "--judge-model", spec.judge_model, "--judge-concurrency", str(spec.judge_concurrency),
        "--min-quality", str(spec.min_quality), "--keep-messages"]
        + (["--judge-all"] if spec.judge_all else []))
    if a.rubric == "reverify":
        a.strict = True
    return a


def _judge(spec: Spec, model: str | None = None):
    from ..judge import Judge, JudgeCfg
    return Judge(JudgeCfg.from_env(model or spec.judge_model, None), spec.judge_concurrency)


def _strategy(spec: Spec, env: str):
    st = spec.strategies().get(env)
    if st is None:
        raise SystemExit(f"{env}: not in the spec's configs")
    cfg, _ = spec.envs()[env]
    return st, cfg


def _root_thinks(spec: Spec) -> bool:
    """Whether sub-calls get enable_thinking=false (root thinks, or [teacher] sub_no_think is set)."""
    return bool(spec.chat_template_kwargs.get("enable_thinking")) or spec.sub_no_think


def _save_hint(spec: Spec, text: str) -> str:
    sha = identity.sha1(text)
    p = spec.dir / "hints" / f"{sha}.txt"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(text)
        tmp.replace(p)
    return sha


async def run_generate(spec: Spec, item: dict) -> dict:
    from .. import generator as g
    from ..tasks import load_env
    env_name = item["env"]
    st, cfg = _strategy(spec, env_name)
    a = gen_args(spec, item["base_url"], cfg.path, cfg.section, env_name)
    load_n = int(item["load_n"])
    hint_sha = _save_hint(spec, st.hint_text)
    out = g.TaskWriter(spec.writer_dir(item.get("writer") or "gen"), st.name)
    env = load_env(st, num_examples=load_n, with_hint=True, sub_no_think=_root_thinks(spec), sub_think=bool(spec.env_param(env_name, "sub_think")))
    by_sha = {}
    for r in _dataset_rows(env, st):
        by_sha.setdefault(identity.task_sha1_of_info(_info(r)), r)
    stats: dict[str, int] = defaultdict(int)
    keyed, extra = [], {}
    for occ in range(int(item.get("rollouts_per_task", 1))):
        for t in item["tasks"]:
            row = by_sha.get(t["task_sha1"])
            if row is None:
                stats["task_missing"] += 1
                continue
            key = f"{t['task_sha1'][:16]}#a{t['attempt']}#r{occ}"
            if key in out.done:
                stats["already_done"] += 1
                att = int(t["attempt"]) + 1
                while f"{t['task_sha1'][:16]}#a{att}#r{occ}" in out.done:
                    att += 1
                if att - int(t["attempt"]) > 64:
                    continue
                key = f"{t['task_sha1'][:16]}#a{att}#r{occ}"
                stats["attempt_bumped"] += 1
            live = {}
            if spec.env_param(env_name, "live_snapshots"):
                ld = spec.dir / "live" / item["id"]
                ld.mkdir(parents=True, exist_ok=True)
                live = {"rlm_snapshot_path": str(ld / (key.replace("/", "_") + ".json"))}
            keyed.append((key, g._with_info(row, rlm_replay_log=True, **live)))
            extra[key] = {"leak_needles": [],
                          "meta": {"task_sha1": t["task_sha1"], "row_hint": False,
                                   "pipeline": {"work_id": item["id"], "hint_sha1": hint_sha,
                                                "attempt": int(key.rsplit("#a", 1)[1].split("#", 1)[0])}}}
    stats["todo"] = len(keyed)
    print(f"[{env_name}] {len(keyed)} rollouts on {len(item['tasks'])} task instances", flush=True)
    if not keyed:
        return dict(stats)
    judge = _judge(spec)
    client, sampling = g.make_client(a)
    chunk_n = a.chunk_size or len(keyed)
    try:
        for ci in range(0, len(keyed), chunk_n):
            chunk = keyed[ci: ci + chunk_n]
            res = await env.generate([inp for _, inp in chunk], client=client, model=a.model, sampling_args=sampling,
                                     max_concurrent=a.max_concurrent,
                                     state_columns=["rlm_sub_llm_log", "rlm_sub_llm_prompt_chars", "rlm_sub_llm_response_chars", "rlm_final_answer"])
            outs = res["outputs"] if isinstance(res, dict) else res.outputs
            by_eid: dict = defaultdict(deque)
            for o in outs:
                by_eid[g._g(o, "example_id")].append(o)
            pending = []
            for key, inp in chunk:
                qd = by_eid.get(g._g(inp, "example_id"))
                if not qd:
                    stats["missing"] += 1
                    continue
                pending.append((key, g.restore_context(g._row(qd.popleft()), inp)))
            if ci + chunk_n >= len(keyed):
                _mark_judging(spec, item)
            await g.evaluate_rows(pending, st, a, judge, out, stats, extra=extra)
            print(f"[{env_name}] chunk {ci // chunk_n + 1}: {dict(stats)}", flush=True)
    finally:
        await judge.aclose()
    return dict(stats)


def _mark_judging(spec: Spec, item: dict) -> None:
    """Mark the item as judging-only, so the orchestrator frees its server slot."""
    try:
        (spec.dir / "work" / f"{item['id']}.judging").write_text(str(time.time()))
    except OSError:
        pass


async def run_catalog(spec: Spec, item: dict) -> dict:
    from .. import replay
    from ..tasks import load_env
    env_name = item["env"]
    st, _ = _strategy(spec, env_name)
    load_n = int(item["load_n"])
    stats: dict[str, int] = defaultdict(int)
    env = load_env(st, num_examples=load_n, with_hint=True, sub_no_think=_root_thinks(spec), sub_think=bool(spec.env_param(env_name, "sub_think")))
    dest = spec.dir / "catalog" / f"{env_name}.n{load_n}.jsonl"
    if item.get("build") or not dest.exists():
        lines, seen = [], set()
        for i, r in enumerate(_dataset_rows(env, st)):
            info = _info(r)
            sha = identity.task_sha1_of_info(info)
            if sha in seen:
                stats["duplicate_content"] += 1
                continue
            seen.add(sha)
            meta = replay.metadata_text(info)
            lines.append({"task_sha1": sha, "meta_sha1": identity.sha1(meta), "context_sha1": identity.context_sha1_of_info(info),
                          "idx": i, "question": identity.question_snippet(meta), "context_chars": identity.context_chars(meta)})
        if not lines:
            raise RuntimeError(f"{env_name}: 0 task instances at num_examples={load_n} (synthetic env? set "
                               f"[envs.overrides.\"{env_name}\"] load_num_examples)")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text("".join(json.dumps(x) + "\n" for x in lines))
        tmp.replace(dest)
        stats["tasks"] = len(lines)
        print(f"[{env_name}] catalogue: {len(lines)} task instances at num_examples={load_n}", flush=True)
    return dict(stats)


KINDS = {"generate": run_generate, "catalog": run_catalog}


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    item_path = Path(argv[0])
    item = json.loads(item_path.read_text())
    spec = load_spec(item["spec"])
    if item.get("pipeline_dir"):
        spec.dir, spec.name = Path(item["pipeline_dir"]), item.get("name", spec.name)
    t0 = time.time()
    if len(item.get("lb_ports") or []) > 1:
        os.environ["RLM_LB_PORTS"] = ",".join(map(str, item["lb_ports"]))
        if item.get("lb_host"):
            os.environ["RLM_LB_HOST"] = item["lb_host"]
    print(f"[worker {item['id']}] {item['kind']} {item['env']} on {os.uname().nodename} base_url={item.get('base_url')} "
          f"lb_ports={item.get('lb_ports')}", flush=True)
    stats = asyncio.run(KINDS[item["kind"]](spec, item))
    result = {"id": item["id"], "stats": stats, "seconds": round(time.time() - t0, 1), "ts": time.time()}
    tmp = item_path.with_name(item["id"] + ".result.tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(item_path.with_name(item["id"] + ".result.json"))
    print(f"[worker {item['id']}] done {json.dumps(stats)} in {result['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
