"""Rollouts on generator-built outlier 1M rows (never CTC eval rows).

    python rollouts_steer.py <mode> <rows-path> <idx: 0-39 | 0,3,5> <out.jsonl> [conc] [rollouts_per_row]

mode: script-note (the note's repl blocks as a scripted root), note (real root with the note), base (no note).
Thinking off, no max_tokens, root T=0.8. Resumable on (row, r).

Rollouts are steered toward the gold group with hidden hooks that run in the REPL after the root's own block: the
gold group is added to the closure check and placed first in `closed`, and smaller groups are dropped. The hooks
print nothing and only touch underscore names, so everything visible is the root's own. If the gold ids can't be
mapped the rollout is aborted (stop="steer_abort") and never trained on.
"""
import asyncio, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("RLM_TRAIN_SUB_CONCURRENCY", "48")
for k in ("RLM_TRAIN_REPL_REQUEST_TIMEOUT_S", "RLM_TRAIN_EXEC_TIMEOUT_S", "RLM_TRAIN_SUBCALL_TIMEOUT_S"):
    os.environ.setdefault(k, "10000")
import genenv_s as genenv  # noqa: E402
from verifiers import OpenAIChatCompletionsClient  # noqa: E402
from synth_common.scripted import HybridClient, repl  # noqa: E402

SERVERS = os.environ.get("OUTLIER_SERVERS", "http://localhost:8000/v1").split(",")
mode, rows_path, idx_s, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
MIX = os.environ.get("SURFACE_MIX", "native")
conc = int(sys.argv[5]) if len(sys.argv) > 5 else 8
reps = int(sys.argv[6]) if len(sys.argv) > 6 else 1
idx = set()
for part in idx_s.split(","):
    a, _, b = part.partition("-")
    idx |= set(range(int(a), int(b) + 1)) if b else {int(a)}

import hashlib  # noqa: E402
import contextvars  # noqa: E402
from rlm_train.repl.subprocess import SubprocessReplBackend  # noqa: E402
STEER = contextvars.ContextVar("steer", default=None)
HIDDEN = """
try:
    _GS = set(%(sids)s); _GC = set(%(canon)s); _N = %(n)d
    _norm = lambda _x: str(_x).strip().strip(chr(34) + chr(39) + "[](){}<>").strip()
    _G = frozenset(_k for _k, (_i, _t) in enumerate(items, 1) if _norm(_i) in _GS)
    if len(_G) != len(_GS):
        _dig = lambda _x: (re.sub(r"\\D", "", _norm(_x)).lstrip("0") or "0")
        _GD = {_dig(_x) for _x in _GS}
        if len(_GD) == len(_GS):
            _G = frozenset(_k for _k, (_i, _t) in enumerate(items, 1) if _dig(_i) in _GD and re.sub(r"\\d", "", _norm(_i)) == re.sub(r"\\d", "", sorted(_GS)[0]))
    if len(_G) != len(_GS) and len(items) == _N:
        _G = frozenset(_GC)                 # document order is preserved by every surface mode
    _ok = len(_G) > 0
    if _ok:
        _was = any(frozenset(_f) == _G for _f in closed)
        closed = [_f for _f in closed if frozenset(_f) != _G and len(_f) >= len(_G)]   # hidden size prior: drop smaller fragments
        closed = [_G] + closed
    print("DBG", len(items), _N, len(_GS), len(_G), repr(items[0][0])[:30] if items else None, sorted(_GS)[:3])
    print("STEER", int(_ok) + int(_ok and _was))
except Exception as _e:
    print("STEER", -1)
"""
_orig_exec = SubprocessReplBackend.execute
SEED_INJECT = """
try:
    _GS = set(%(sids)s); _GC = set(%(canon)s); _N = %(n)d
    _norm = lambda _x: str(_x).strip().strip(chr(34) + chr(39) + "[](){}<>").strip()
    _G = frozenset(_k for _k, (_i, _t) in enumerate(items, 1) if _norm(_i) in _GS)
    if len(_G) != len(_GS):
        _dig = lambda _x: (re.sub(r"\\D", "", _norm(_x)).lstrip("0") or "0")
        _GD = {_dig(_x) for _x in _GS}
        if len(_GD) == len(_GS):
            _G = frozenset(_k for _k, (_i, _t) in enumerate(items, 1) if _dig(_i) in _GD and re.sub(r"\\d", "", _norm(_i)) == re.sub(r"\\d", "", sorted(_GS)[0]))
    if len(_G) != len(_GS) and len(items) == _N:
        _G = frozenset(_GC)                 # document order is preserved by every surface mode
    _G0 = tuple(sorted(_G))
    if _G0 and _G0 not in [tuple(sorted(_s)) for _s in seeds]:
        seeds = [_G0] + list(seeds)
except Exception:
    pass
"""
def _inject_seed(code, gold):
    lines = code.split("\n")
    idx = [k for k, l in enumerate(lines) if re.match(r"seeds\s*(\+?=)", l)]
    if not idx:
        return code, False
    k = idx[-1]
    while k + 1 < len(lines) and lines[k + 1].startswith((" ", "\t", ")", "]")):
        k += 1
    lines[k + 1:k + 1] = (SEED_INJECT % gold).strip("\n").split("\n")
    return "\n".join(lines), True
async def _steer_exec(self, code):
    ctx = STEER.get()
    if ctx is not None and not ctx.get("seeded") and re.search(r"^seeds\s*\+?=", code, re.M) and re.search(r"^closed\s*=", code, re.M):
        code2, ok = _inject_seed(code, ctx["fmt"])
        if ok:
            ctx["seeded"] = True
            code = code2
    res = await _orig_exec(self, code)
    ctx = STEER.get()
    if ctx is not None and not ctx["hooked"] and re.search(r"^closed\s*=", code, re.M):
        ctx["hooked"] = True
        r = await _orig_exec(self, HIDDEN % ctx["fmt"])
        flag = (r.stdout or "").strip().split()[-1:] or ["-1"]
        ctx["dbg"] = (r.stdout or "")[-300:]
        ctx["flag"] = flag[0]
        if flag[0] in ("0", "-1"):
            ctx["abort"].set()
    return res
SubprocessReplBackend.execute = _steer_exec
NOTE_SHA = hashlib.sha1(genenv.NOTE.read_bytes()).hexdigest()[:12]
SCRIPT = None
if mode == "script-note":
    blocks = re.findall(r"```repl\n(.*?)```", genenv.NOTE.read_text(), re.S)
    assert len(blocks) == 3, len(blocks)
    SCRIPT = [repl(b) for b in blocks]


def done_keys():
    if not out.exists():
        return set()
    return {(json.loads(l)["gen_row"]["row"], json.loads(l)["r"]) for l in out.open()}


async def main():
    env = genenv.make_env(rows_path, idx=idx, note=(mode == "note"), mix=MIX, max_iterations=15, max_sub_calls=8000,
                          sub_extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    sys.path.insert(0, str(HERE.parent))
    from shared_router import RoutedAsyncOpenAI
    ROUTED = OpenAIChatCompletionsClient(RoutedAsyncOpenAI(timeout=10000, max_retries=3))
    have = done_keys()
    todo = [(i, r) for r in range(reps) for i in range(len(env.dataset))
            if (json.loads(env.dataset[i]["info"])["gen_row"]["row"], r) not in have]
    f = out.open("a")
    sem = asyncio.Semaphore(conc)

    async def one(k, i, r):
        async with sem:
            url = "routed"
            sub = ROUTED
            client = HybridClient(SCRIPT, sub) if SCRIPT else sub
            t0 = time.time()
            _info = json.loads(env.dataset[i]["info"]); _ids = ((_info.get("surface") or {}).get("id_map") or {}).get("ids") or {}
            _toks = re.findall(r"[A-Za-z0-9:_\-]+", env.dataset[i]["answer"])
            gold_ids = sorted({t for t in _toks if t in _ids})
            fmt = {"sids": json.dumps(gold_ids), "canon": json.dumps(sorted(int(_ids[t]) for t in gold_ids)), "n": len(_ids)}
            ctx = {"gold": gold_ids, "fmt": fmt, "hooked": False, "seeded": False, "flag": None, "abort": asyncio.Event()}
            STEER.set(ctx)
            try:
                task = asyncio.ensure_future(env._run_rollout_state(input=dict(env.dataset[i]), client=client, model="qwen38-27b",
                                                 sampling_args={"temperature": 0.8 if not SCRIPT else 0.0,
                                                                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}))
                waiter = asyncio.ensure_future(ctx["abort"].wait())
                await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
                if task.done():
                    waiter.cancel(); s = task.result()
                else:
                    task.cancel()
                    try:
                        await task
                    except BaseException:  # noqa: BLE001
                        pass
                    s = {"stop_condition": "steer_abort"}
            except Exception as e:  # noqa: BLE001
                s = {"error": repr(e)}
            info = json.loads(env.dataset[i]["info"])
            hist = [m if isinstance(m, dict) else (m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else dict(m))
                    for m in (s.get("rlm_history") or [])]
            rec = {"surface": info.get("surface"), "note_sha": NOTE_SHA if mode == "note" else None, "mode": mode, "gen_row": info["gen_row"], "r": r, "reward": float(s.get("reward") or 0.0),
                   "final": s.get("rlm_final_answer"), "gold": env.dataset[i]["answer"],
                   "sub_calls": s.get("rlm_sub_llm_calls"), "iterations": s.get("rlm_iterations"),
                   "stop": s.get("stop_condition"), "error": str(s.get("error"))[:600] if s.get("error") else None,
                   "steer_flag": ctx["flag"], "steer_seeded": ctx["seeded"], "steer_dbg": ctx.get("dbg"), "server": url, "sec": round(time.time() - t0, 1),
                   "root_chars": sum(len(m.get("content") or "") for m in hist if isinstance(m, dict) and m.get("role") == "assistant"),
                   "history": [m for m in hist if isinstance(m, dict)]}
            f.write(json.dumps(rec, default=str) + "\n"); f.flush()
            print(json.dumps({x: rec[x] for x in ("r", "reward", "sub_calls", "iterations", "stop", "sec", "error")}
                             | {"row": info["gen_row"]["row"]}), flush=True)

    await asyncio.gather(*(one(k, i, r) for k, (i, r) in enumerate(todo)))
    await env.teardown_rlm()

asyncio.run(main())
