"""Hinted SFT trace generation on the synthetic ELB mix, one row kind per run (no real ELB data is read).

    python gen_synth.py --kind visit --start 48 --n 600 --servers http://localhost:8000/v1 --per-server 5 --out <dir>

Uses the ELB eval harness settings with each kind's final note as the hint. Writes <out>/<kind>/rollouts.jsonl and
skips rows already done.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from verifiers import OpenAIChatCompletionsClient

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def client_for(base_url: str, timeout: float) -> OpenAIChatCompletionsClient:
    return OpenAIChatCompletionsClient(AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=base_url,
                                                   timeout=timeout, max_retries=2))


def _get(m: Any, k: str) -> Any:
    return m.get(k) if isinstance(m, dict) else getattr(m, k, None)


def _text(c: Any) -> str:
    if isinstance(c, list):
        return "".join((_get(p, "text") or "") for p in c)
    return c or ""


def msg_dict(m: Any) -> dict:
    """role / content / reasoning_content of one history message."""
    rc = _get(m, "reasoning_content") or _get(m, "reasoning") or ""
    return {"role": _get(m, "role") or "?", "content": _text(_get(m, "content")), "reasoning_content": str(rc or "")}


def record(state: dict, row: int, kind: str, server: str, elapsed: float, sampling: dict) -> dict:
    info = state.get("info") or {}
    hidden = dict(state.get("synth_hidden") or {})
    gold = hidden.pop("gold", None)
    for k in ("items", "_index_min_len", "_index_factory", "required_ids"):
        hidden.pop(k, None)
    return {
        "row": row, "kind": kind, "split": info.get("synth_split"), "seed": info.get("synth_seed"),
        "synth_params": info.get("synth_params"),
        "reward": float(state.get("reward") or 0.0),
        "metrics": dict(state.get("metrics") or {}),
        "iterations": int(state.get("rlm_iterations") or 0),
        "sub_calls": int(state.get("rlm_sub_llm_calls") or 0),
        "stop_condition": state.get("stop_condition"),
        "final_answer": state.get("rlm_final_answer"),
        "hinted": int((hidden or {}).get("hinted") or 0),
        "gold": gold, "hidden": hidden,
        "messages": [msg_dict(m) for m in (state.get("rlm_history") or [])],
        "subcalls": [{"prompt": e.get("prompt") or "", "response": e.get("response") or "",
                      "prompt_chars": e.get("prompt_chars"), "response_chars": e.get("response_chars")}
                     for e in (state.get("rlm_sub_llm_log") or [])],
        "server": server, "sampling": sampling, "elapsed_s": round(elapsed, 1),
        "error": state.get("error") and str(state.get("error"))[:500],
    }


def done_rows(path: Path) -> set[int]:
    out: set[int] = set()
    if path.exists():
        for ln in open(path):
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001  torn last line
                continue
            if not r.get("error"):
                out.add(int(r["row"]))
    return out


async def main_async(a: argparse.Namespace) -> None:
    from expertlongbench_synth import mix

    env = mix.load_environment(split=a.split, num_examples=a.start + a.n, kind=a.kind, hint="note", hint_prob=1.0,
                               max_iterations=a.max_iterations, orchestrator_prompt=True,
                               sub_extra_body=NO_THINK)
    assert env._sub_llm_fn is not None and env._sub_llm_fn_batched is not None, "sub-calls must go to the fake oracle"
    out = Path(a.out) / a.kind
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rollouts.jsonl"
    rows = [i for i in range(a.start, a.start + a.n) if i % a.shards == a.shard]
    skip = done_rows(path)
    todo = [i for i in rows if i not in skip]
    print(f"{a.kind}: {len(rows)} rows in shard, {len(rows) - len(todo)} done, {len(todo)} to run", flush=True)
    sampling: dict[str, Any] = {"max_tokens": a.max_tokens, "temperature": a.temperature, "extra_body": dict(NO_THINK)}
    (out / f"settings.shard{a.shard}.json").write_text(json.dumps({**vars(a), "sampling": sampling,
                                                                    "hint": "note", "hint_prob": 1.0,
                                                                    "docket_note": mix.DOCKET_NOTE_VERSION,
                                                                    "forms_notes": {"visit": "t06_note_c10",
                                                                                    "tutor": "t04_note_c6",
                                                                                    "recipe": "t03_note_c5"}},
                                                                   indent=1))
    q: asyncio.Queue[int] = asyncio.Queue()
    for i in todo:
        q.put_nowait(i)
    lock = asyncio.Lock()
    recs: list[dict] = []

    async def worker(server: str, client: Any) -> None:
        while True:
            try:
                i = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            t0 = time.time()
            try:
                state = await env._run_rollout_state(input=dict(env.dataset[i]), client=client, model=a.model,
                                                     sampling_args=sampling)
                r = record(state, i, a.kind, server, time.time() - t0, sampling)
            except Exception as e:  # noqa: BLE001
                r = {"row": i, "kind": a.kind, "reward": 0.0, "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "server": server, "elapsed_s": round(time.time() - t0, 1)}
            async with lock:
                recs.append(r)
                with open(path, "a") as f:
                    f.write(json.dumps(r, default=str) + "\n")
            m = r.get("metrics") or {}
            print(f"[{a.kind} row {i}] reward={r['reward']:.3f} iters={r.get('iterations')} subs={r.get('sub_calls')} "
                  f"stop={r.get('stop_condition')} hint_ment={m.get('prog_no_hint_mention')} t={r['elapsed_s']}s "
                  f"{server.split('//')[-1]} {('ERR ' + r['error']) if r.get('error') else ''}", flush=True)

    clients = {s: client_for(s, a.request_timeout) for s in a.servers}
    await asyncio.gather(*(worker(s, c) for s, c in clients.items() for _ in range(a.per_server)))
    try:
        await env.teardown_rlm()
    except Exception:  # noqa: BLE001
        pass
    ok = [r for r in recs if not r.get("error")]
    c = Counter(r.get("stop_condition") for r in ok)
    if ok:
        print(json.dumps({"kind": a.kind, "n": len(recs), "errors": len(recs) - len(ok),
                          "reward": round(statistics.mean(r["reward"] for r in ok), 3),
                          "turns": round(statistics.mean(r["iterations"] for r in ok), 2),
                          "sub_calls": round(statistics.mean(r["sub_calls"] for r in ok), 2),
                          "stop": dict(c)}), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["docket", "visit", "tutor", "recipe"])
    ap.add_argument("--split", default="train", choices=["train"],
                    help="train only: val/test seeds are the probe / eval ranges and never become SFT data")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--servers", nargs="+", required=True, help="OpenAI base URLs, e.g. http://localhost:8000/v1")
    ap.add_argument("--per-server", type=int, default=12, help="concurrent rollouts per server")
    ap.add_argument("--model", default="qwen38-27b")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-iterations", type=int, default=40)
    ap.add_argument("--request-timeout", type=float, default=1800.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.per_server > 24:
        sys.exit("--per-server > 24: over the per-server concurrency budget")
    asyncio.run(main_async(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
