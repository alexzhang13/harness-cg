"""Rollouts on generator-built 256k qdmatch rows (never CTC eval rows)."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(_REPO)
QD = ROOT / "synth-datagen/ctcbench/qdmatch_hpqa"
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / p) for p in ("src", "environments/synthetic/synth_common", "environments/real/CTCBench", "environments/synthetic/ctcbench")] + [str(QD), str(HERE)]
for k in ("RLM_TRAIN_REPL_REQUEST_TIMEOUT_S", "RLM_TRAIN_EXEC_TIMEOUT_S", "RLM_TRAIN_SUBCALL_TIMEOUT_S"):
    os.environ.setdefault(k, "10000")
os.environ.setdefault("RLM_TRAIN_SUB_CONCURRENCY", "48")
from openai import AsyncOpenAI  # noqa: E402
from verifiers import OpenAIChatCompletionsClient  # noqa: E402

from synth_common.scripted import HybridClient, repl, run_rollout  # noqa: E402

SERVERS = os.environ.get("QD_SERVERS", "http://localhost:8000/v1").split(",")
NOTE = os.environ.get("QD_NOTE", str(ROOT / "synth-datagen/ctcbench/notes/ctc-qdmatch-hpqa-256k-train.md"))
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}


def prologue() -> str:
    sys.path.insert(0, str(ROOT / "synth-datagen/common"))
    from rlm_generate_trajectories.strategies import HARNESS_FOOTER, HINT_CLOSE, HINT_OPEN
    return f"{HINT_OPEN}\n{Path(NOTE).read_text().strip()}\n\n{HARNESS_FOOTER.strip()}\n{HINT_CLOSE}"


def surface_mode(i: int) -> str:
    m = os.environ.get("QD_SURFACE", "")
    if m == "mix":
        return "native" if i % 4 == 0 else "varied"
    return m


def gen_rows(path: str) -> list[dict]:
    """Rows from a decontaminated generator file (same rendering as qd_dev256.dev_rows)."""
    from ctcbench_real import core as C
    from ctcbench_real import env as E
    spec = C.spec_of("qdmatch_hpqa")
    out = []
    for line in open(path):
        ex = json.loads(line)
        mode = surface_mode(len(out))
        if mode:
            from ctcbench_synth import surface as SF
            r = SF.render(ex, "qdmatch_hpqa", mode=mode, seed=len(out))
            context, root_prompt, question = r.context, r.root_prompt, r.task_text.split("\n")[0][:2000]
            surf = {"mode": mode, "id_map": r.id_map, "style": r.style}
        else:
            context, task_text, question = E.render(ex, spec)
            root_prompt, surf = f"{task_text}\n\n{E.submit_hint(task_text)}", None
        meta = {"context": context, "root_prompt": root_prompt, "surface": surf,
                "ctc_task": "qdmatch_hpqa", "ctc_spec": spec.name, "ctc_class": "high", "ctc_metric": spec.primary_metric,
                "n_docs": len(ex["documents"]),
                "ctc_core": {"version": Path(path).stem, "cell": "qdmatch_hpqa@r256k-gen", "tier": "max",
                             "part": "gen", "hf_row": ex["_gen_row"]},
                "example": E._slim(ex)}
        out.append({"example_id": len(out), "prompt": [{"role": "user", "content": question}],
                    "answer": E.gold_target(ex, spec), "info": json.dumps(meta, ensure_ascii=False)})
    return out


def waiting(base_url: str) -> float:
    import urllib.request
    try:
        txt = urllib.request.urlopen(base_url.replace("/v1", "/metrics"), timeout=5).read().decode()
        return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith("vllm:num_requests_waiting{"))
    except Exception:  # noqa: BLE001
        return 1.0


async def surface_score(info, state, **_kw) -> float:
    meta = json.loads(info) if isinstance(info, str) else info
    if not meta.get("surface"):
        from ctcbench_real import env as E
        return await E._score(info, state)
    from ctcbench_synth import surface as SF
    m = SF.score(state.get("rlm_final_answer"), meta["example"], "qdmatch_hpqa", meta["surface"]["id_map"])
    return max(0.0, min(1.0, m["primary"]))


def make_env(mode: str):
    import rlm_train
    from datasets import Dataset
    from ctcbench_real import env as E
    if not os.environ.get("QD_ROWS"):
        raise SystemExit("set QD_ROWS to a decontaminated generator rows file (the dev-row mode is not part of datagen)")
    rows = gen_rows(os.environ["QD_ROWS"])
    rubric = rlm_train.RLMTrainRubric(correctness=surface_score, weight=1.0, min_iterations=1, min_subcall=0)
    kw = {"user_prologue": prologue()} if mode == "note" else {}
    return rlm_train.RLMTrainEnv(dataset=Dataset.from_list(rows), code_block_feedback=True, max_iterations=20, rubric=rubric,
                                 sub_sampling_args={"max_tokens": None, "extra_body": NOTHINK}, **kw)


def msgs(st, key: str = "completion") -> list:
    out = []
    for m in st.get(key) or []:
        d = m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else dict(m))
        out.append({"role": d.get("role"), "content": d.get("content")})
    return out


def main(mode: str, out: str, a: int, b: int, par: int = 2) -> None:
    env = make_env(mode)
    done = {json.loads(l)["idx"] for l in open(out)} if Path(out).exists() else set()
    if os.environ.get("QD_ROUTE"):
        from router import POOL_STATE, RoutedClient
        subs = [RoutedClient()]
        SERVERS[:] = ["routed"]
    else:
        subs = [OpenAIChatCompletionsClient(AsyncOpenAI(api_key="local", base_url=u, timeout=10000)) for u in SERVERS]
    fh = open(out, "a")
    if mode == "scripted":
        import importlib
        TURNS = importlib.import_module(os.environ.get("QD_TURNS", "note_turns")).TURNS
        program = [repl(t) for t in TURNS]

    async def go():
        sems = [asyncio.Semaphore(par) for _ in subs]

        async def one(i):
            if i in done:
                return
            k = i % len(subs)
            async with sems[k]:
                if Path(os.environ.get("QD_STOP_FILE", "STOP_LAUNCHING")).exists():
                    return
                extra = [f for f in os.environ.get("QD_DONE_FILES", "").split(",") if f and Path(f).exists()]
                if any(json.loads(l)["idx"] == i for f in extra for l in open(f)):
                    return
                while os.environ.get("QD_ROUTE") and not await asyncio.to_thread(POOL_STATE.any_free):
                    await asyncio.sleep(30)
                while not os.environ.get("QD_ROUTE") and os.environ.get("QD_GUARD") and await asyncio.to_thread(waiting, SERVERS[k]) > 0:
                    await asyncio.sleep(30)
                t0 = time.time()
                client = HybridClient(program, subs[k]) if mode == "scripted" else subs[k]
                try:
                    st = await run_rollout(env, i, client, model="qwen38-27b",
                                           sampling_args={"max_tokens": None, "temperature": 0.8, "extra_body": NOTHINK})
                except Exception as e:  # noqa: BLE001
                    print("idx", i, "rollout error", repr(e)[:300], flush=True)
                    return
            info = json.loads(env.dataset[i]["info"])
            m = msgs(st)
            rec = {"idx": i, "gen_row": info["ctc_core"]["hf_row"], "rows": os.environ.get("QD_ROWS", "dev256"), "mode": mode,
                   "surface": info.get("surface"), "reward": float(st.get("reward") or 0),
                   "sub_calls": int(st.get("rlm_sub_llm_calls") or 0), "final": st.get("rlm_final_answer"),
                   "stop": str(st.get("stop_condition") or ""), "error": str(st.get("error") or "")[:300],
                   "turns": sum(1 for x in m if x["role"] == "assistant"),
                   "root_chars": sum(len(x["content"] or "") for x in m if x["role"] == "assistant"),
                   "seconds": time.time() - t0, "server": SERVERS[k], "prompt": msgs(st, "prompt"), "messages": m}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print("idx", i, "reward", round(rec["reward"], 3), "turns", rec["turns"], "sub", rec["sub_calls"],
                  "chars", rec["root_chars"], "stop", rec["stop"], "sec", round(rec["seconds"]), flush=True)

        order = list(range(a, min(b, len(env.dataset))))
        if os.environ.get("QD_ONLY"):
            order = [i for i in order if i in {int(x) for x in os.environ["QD_ONLY"].split(",")}]
        if os.environ.get("QD_REVERSE"):
            order.reverse()
        await asyncio.gather(*(one(i) for i in order))

    try:
        asyncio.run(go())
    finally:
        try:
            asyncio.run(env.teardown_rlm())
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 2)
