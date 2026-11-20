"""Hinted grouping@32k rollouts: real 27B root with the note as a <strategy_hint> prologue, thinking off, no output
cap. One process per port (--shard k/n). Writes one JSONL line per rollout with the history and sub-call log."""
import argparse, json, os, sys
from datetime import datetime, timezone
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(os.environ.get("CTC_DATAGEN_DIR") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../outputs/synth-datagen/ctcbench")), "grouping32")
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
WT = os.path.join(R, "synth-datagen/common")
sys.path[:0] = [os.path.join(R, "environments/real/CTCBench"), os.path.join(R, "environments/synthetic/ctcbench"),
                os.path.join(R, "environments/synthetic/synth_common"), os.path.join(R, "src")]
ap = argparse.ArgumentParser()
ap.add_argument("--gate", action="store_true")
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--modes", default="native")
ap.add_argument("--part", default="dev")
ap.add_argument("--split", default="train")
ap.add_argument("--n", type=int, default=-1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--note", default="")
ap.add_argument("--port", default="pool")
ap.add_argument("--shard", default="0/1")
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--max-iterations", type=int, default=15)
ap.add_argument("--out", required=True)
a = ap.parse_args()

prologue = None
if a.note:
    sys.path.insert(0, WT)
    from rlm_generate_trajectories.strategies import HARNESS_FOOTER, HINT_OPEN, HINT_CLOSE
    prologue = f"{HINT_OPEN}\n{open(a.note).read().strip()}\n\n{HARNESS_FOOTER.strip()}\n{HINT_CLOSE}"
os.environ.setdefault("RLM_TRAIN_SUB_CONCURRENCY", "48")
for k in ("RLM_TRAIN_EXEC_TIMEOUT_S", "RLM_TRAIN_REPL_REQUEST_TIMEOUT_S", "RLM_TRAIN_SUBCALL_TIMEOUT_S"):
    os.environ.setdefault(k, "10000")
common = dict(max_iterations=a.max_iterations, user_prologue=prologue)
sys.path.insert(0, HERE)
import surf_env as X
ms = a.modes.split(",")
mode_of = (lambda k: "native" if k % 4 == 0 else "varied") if a.modes == "mix" else (lambda k: ms[k % len(ms)])
if a.gate:
    rows = []
    for mi, m in enumerate(ms):
        rr = X.build_rows("dev", lambda k, m=m: m, n=a.n, seed=a.seed)
        for r in rr: r["example_id"] = mi * 1000 + r["example_id"]
        rows += rr
else:
    rows = X.build_rows(a.split, mode_of, n=a.n, seed=a.seed, start=a.start)
env = X.load_environment(rows, **common)
k, n = map(int, a.shard.split("/"))
inputs = env._get_eval_inputs(-1, 1)
inputs = [x for x in inputs if int(x["example_id"]) % n == k]
done = set()
if os.path.exists(a.out):
    for line in open(a.out):
        try: done.add(json.loads(line)["example_id"])
        except Exception: pass
inputs = [x for x in inputs if int(x["example_id"]) not in done]
print(f"{len(inputs)} to run ({len(done)} done) on {a.port}", flush=True)
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
import router
client = OpenAIChatCompletionsClient(router.async_openai(7200))
sampling = {"temperature": a.temperature, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
cols = ["rlm_history", "rlm_sub_llm_log", "rlm_sub_llm_calls", "rlm_iterations", "rlm_final_answer",
        "rlm_sub_llm_prompt_chars", "rlm_sub_llm_response_chars", "info"]

def J(x):
    if x is None or isinstance(x, (str, int, float, bool)): return x
    if isinstance(x, list): return [J(i) for i in x]
    if isinstance(x, dict): return {kk: J(v) for kk, v in x.items()}
    for m in ("model_dump", "dict"):
        f = getattr(x, m, None)
        if callable(f):
            try: return J(f())
            except Exception: break
    return str(x)

def persist(_all, new, _meta):
    try:
        with open(a.out, "a") as f:
            for o in new:
                g = (lambda key, d=None: o.get(key, d)) if isinstance(o, dict) else (lambda key, d=None: getattr(o, key, d))
                info = g("info"); info = json.loads(info) if isinstance(info, str) else (info or {})
                rec = {"example_id": g("example_id"), "reward": g("reward"), "metrics": J(g("metrics")),
                       "stop_condition": g("stop_condition"), "error": J(g("error")), "is_truncated": g("is_truncated"),
                       "answer": g("answer"), "completion": J(g("completion")), "token_usage": J(g("token_usage")),
                       "hinted": bool(prologue), "env": "grouping_synth",
                       "synth": {k2: v for k2, v in info.items() if k2 not in ("context", "row")}, "row_paper_ids": ((info.get("row") or {}).get("synth") or {}).get("paper_ids"),
                       "surface_mode": info.get("surface_mode"), "style": info.get("style"), "id_map": info.get("id_map"),
                       **{c: J(g(c)) for c in cols if c != "info"},
                       "ts": datetime.now(timezone.utc).isoformat()}
                f.write(json.dumps(rec) + "\n"); f.flush()
    except Exception as e:
        print("persist error", repr(e), flush=True)

if inputs:
    env.generate_sync(inputs, client=client, model="qwen38-27b", sampling_args=sampling, max_concurrent=a.concurrency,
                      on_progress=[persist], state_columns=cols)
print("done", flush=True)
