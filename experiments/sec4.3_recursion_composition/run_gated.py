#!/usr/bin/env python3
"""Gated chains (Table recursion-composition): solve task A to unlock B, then C, then D.

Each level is an rlm.RLM on one task, with two REPL tools: unlock(answer) scores the answer and returns the next task
(2 attempts per task), handoff(pkg) starts a fresh RLM level on an unlocked task.

    --conds gflat    unlock only, protocol instruction -> "RLM harness"
            ghint    unlock + handoff, with hint   -> "+ hint"
            gplain   unlock + handoff, no hint     -> "+ SFT"

Chain score = mean task score of the last answer per task (unreached tasks 0). One JSON row per chain.
"""
import argparse, glob, json, os, sys, threading, time
from concurrent.futures import ProcessPoolExecutor, as_completed
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import tasks
from gated_chain import GatedChain
from rlm import RLM
from rlm.logger import RLMLogger
from tasks import slim, say

_TEMPLATE = ("```repl\nmy_answer = <your final answer, as a string in the format the question asks>\npkg = unlock(my_answer)\nprint(pkg if not isinstance(pkg, dict) else {k: (v if k in ('stage', 'of') else '...') for k, v in pkg.items()})\n```")
# The "+ hint" protocol, shown before the task question; REMINDER is repeated after it.
HINT = ("[MANDATORY PROTOCOL - read before starting] You are solving a gated sequence of tasks. Your answer is NOT accepted by submitting it in `answer[\"content\"]`. "
        "Work out your answer as you normally would, then, in a repl cell, run EXACTLY this (fill in the answer):\n" + _TEMPLATE + "\n"
        "Case 1: `pkg` is a dict with a 'question' and 'context': the next tasks are in it. Do NOT solve them yourself. Run `down = handoff(pkg)` in the next cell. A fresh level solves the rest.\n"
        "Case 2: `pkg` starts with 'LOCKED': your answer was wrong; recompute it and call unlock again (one more attempt).\n"
        "Case 3: `pkg` == 'DONE': nothing is left.\n"
        "Only AFTER unlock (and handoff, if there was a pkg) has been called, finish with `answer[\"content\"] = my_answer; answer[\"ready\"] = True`. Never set `answer[\"ready\"]` before calling `unlock`.")
REMINDER = ("\n\n[REMINDER] Before you finish: you must have run `pkg = unlock(my_answer)` and, if pkg is a dict, `down = handoff(pkg)`. Submitting without calling `unlock` leaves the remaining tasks unsolved and scores zero.")
UNLOCK_DESC_STRONG = ("REQUIRED. unlock(answer): you MUST call this with your final answer (a string in the format the question asks) BEFORE submitting `answer['ready']`. Returns a dict "
                      "{'stage','of','question','context'} (the next task: pass it to handoff, do not solve it yourself), 'DONE', or 'LOCKED: ...' (wrong; 2 attempts per task).")
HANDOFF_DESC_STRONG = "REQUIRED whenever unlock returned a dict. handoff(pkg): pass that dict; a fresh RLM level solves the remaining tasks. Never solve the next task yourself."
UNLOCK_DESC = ("unlock(answer): submit your answer to the CURRENT task (a string in the format the question asks). Returns the NEXT task as a dict "
               "{'stage','of','question','context'} if the answer is correct, 'DONE' after the last task, or 'LOCKED: ...' if it is wrong (2 attempts per task). Later tasks are hidden until then.")
HANDOFF_DESC = "handoff(pkg): solve the task package returned by unlock (and every task after it) in a fresh RLM level; returns a list with that level's result and everything below it."
FLAT_UNLOCK_DESC = UNLOCK_DESC + " When it returns a task dict, work on that task next (its 'context' is the data for it; bind it as `context_0 = pkg['context']; context = context_0`)."
# The "RLM harness" (gflat) instruction, shown before the task question.
FLAT_PROTOCOL = ("[MANDATORY PROTOCOL] Your answer is NOT accepted through `answer[\"content\"]` until the whole sequence is done. You must solve several tasks one after another in THIS session; the later tasks stay hidden until the current one is answered correctly. Work out your answer to the task below as you normally would, then check it with `res = unlock(my_answer)` (my_answer = a string in the format the question asks). If `res` is a dict, it is the NEXT task: `res['question']` is its question and `res['context']` its data (bind `context_0 = res['context']; context = context_0`). Solve it the same way and call `unlock` again. If `res` starts with 'LOCKED', your answer was wrong: recompute and call `unlock` once more. When `res` == 'DONE' you are finished: set `answer[\"content\"] = my_answer; answer[\"ready\"] = True`. Do not set `answer[\"ready\"]` before `unlock` has returned 'DONE': that ends the session with the remaining tasks unsolved.\n"
                 "Use exactly this after you have worked out your answer to a task:\n"
                 "```repl\n"
                 "my_answer = <your final answer to the CURRENT task, as a string in the format its question asks>\n"
                 "res = unlock(my_answer)\n"
                 "if isinstance(res, dict):      # the next task\n"
                 "    print(res['stage'], res['of']); print(res['question'])\n"
                 "    context_0 = res['context']; context = context_0     # data for the next task\n"
                 "else:\n"
                 "    print(res)\n"
                 "```")


def fake_items(seed, n=4):
    """Chain of generated tasks (fake_world.py)."""
    import random, fake_world as FW
    rng = random.Random(seed)
    return [FW.make_item(rng) for _ in range(n)]


def fake_loaders(items):
    import fake_world as FW
    return [(lambda it=it: {"env": "fake", "question": it["question"], "context": it["context"], "score": lambda a, it=it: FW.score(it, a)}) for it in items]


def level_root(a, cond, pkg, n):
    """Root prompt of one level (question, optional progress header, and the gflat / ghint instruction)."""
    import header as HD
    q = HD.with_header(pkg["question"], pkg["stage"] - 1, n, getattr(a, "header_pos", "post")) if getattr(a, "header", "none") != "none" else pkg["question"]
    if cond == "gflat": return FLAT_PROTOCOL + "\n\n" + q
    if cond != "ghint": return q
    return HINT + "\n\n" + q + REMINDER


def pool_loaders(pool_ids):
    def mk(pid):
        def load():
            it = tasks.item(pid)
            ctx = it.get("context_docs") or it["context"]
            return {"env": it["env"], "question": it["question"], "context": ctx, "score": lambda a, it=it: it["_mod"].score(it, a)}
        return load
    return [mk(p) for p in pool_ids]


class GRun:
    def __init__(self, a, cond, chain):
        self.a, self.cond, self.chain = a, cond, chain
        self.levels = []; self.lock = threading.Lock(); self.tool_calls = {"unlock": 0, "handoff": 0}

    def kw(self, logger):
        a = self.a
        return dict(backend="openai", backend_kwargs={"model_name": a.model, "base_url": a.base_url, "api_key": "x", "timeout": 3600,
                    "sampling_args": {"temperature": a.temperature, "max_tokens": a.max_tokens, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}},
                    **({"sub_sampling_args": {"max_tokens": 24576, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}} if a.sub_thinking else {}),
                    environment="local", max_depth=1, max_iterations=a.max_iterations, logger=logger, **({"max_timeout": a.timeout} if a.timeout else {}))

    def level(self, pkg, idx):
        stage = pkg["stage"] - 1
        base_unlock = self.chain.unlock_flat if self.cond == "gflat" else self.chain.unlock_for(stage)
        def unlock(answer):
            with self.lock: self.tool_calls["unlock"] += 1
            return base_unlock(answer)
        strong = self.cond == "ghint"
        tools = {"unlock": {"tool": unlock, "description": FLAT_UNLOCK_DESC if self.cond == "gflat" else (UNLOCK_DESC_STRONG if strong else UNLOCK_DESC)}}
        if self.cond != "gflat":
            def handoff(p):
                with self.lock: self.tool_calls["handoff"] += 1
                if not isinstance(p, dict) or "question" not in p or "context" not in p or "stage" not in p:
                    return "handoff expects the task package returned by unlock"
                return self.level(p, idx + 1)
            tools["handoff"] = {"tool": handoff, "description": HANDOFF_DESC_STRONG if strong else HANDOFF_DESC}
        logger = RLMLogger() if self.a.save_traj else None
        t0 = time.time()
        root = level_root(self.a, self.cond, pkg, self.chain.n)
        r = RLM(**self.kw(logger), custom_tools=tools).completion(pkg["context"], root_prompt=root)
        with self.lock:
            self.levels.append({"idx": idx, "stage": stage, "seconds": round(time.time() - t0, 1), "traj": slim(r.metadata) if logger else None})
        return [r.response] + [None] * 0


def rollout(cond, pool_ids, a, tag):
    t0 = time.time(); err = None
    synth = isinstance(pool_ids, dict)
    if synth:
        import fake_world as FW
        if getattr(a, "fake_reader", False): FW.install_fake_reader()
        loaders = fake_loaders(fake_items(pool_ids["fake"], pool_ids.get("n", 4)))
    else:
        loaders = pool_loaders(pool_ids)
    chain = GatedChain(loaders, max_attempts=a.attempts)
    run = GRun(a, cond, chain)
    try:
        run.level(chain.package(0), 0)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    st = chain.stats()
    return {"cond": cond, "tag": tag, "pool_ids": pool_ids, "envs": ["fake"] * pool_ids.get("n", 4) if synth else [tasks.POOL[p]["env"] for p in pool_ids], **st, "all": st["all_unlocked"], "scores_true": st["scores"],
            "children": max(0, len(run.levels) - 1), "tool_calls": run.tool_calls, "gate_log": chain.log, "levels": sorted(run.levels, key=lambda l: l["idx"]),
            "error": err, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--conds", default="gplain"); ap.add_argument("--chains", default=""); ap.add_argument("--chains-file", default="")
    ap.add_argument("--reps", type=int, default=1); ap.add_argument("--rep-offset", type=int, default=0); ap.add_argument("--resume", action="store_true"); ap.add_argument("--index-from", type=int, default=0); ap.add_argument("--index-to", type=int, default=10**9)
    ap.add_argument("--workers", type=int, default=20); ap.add_argument("--temperature", type=float, default=0.0); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-iterations", type=int, default=20); ap.add_argument("--timeout", type=float, default=0); ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--header", choices=["none", "post", "pre"], default="none", help="progress header (header.py) after/before the task question"); ap.add_argument("--header-pos", default="post")
    ap.add_argument("--fake", type=int, default=0, help="run N chains of fake tasks (seeds --fake-first..) instead of pool chains"); ap.add_argument("--fake-first", type=int, default=0)
    ap.add_argument("--fake-reader", action="store_true", help="fake tasks: llm_query is answered by the scripted reader")
    ap.add_argument("--sub-thinking", action="store_true"); ap.add_argument("--save-traj", action="store_true"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.header in ("post", "pre"): a.header_pos = a.header
    if a.fake: chain_list = [{"fake": s} for s in range(a.fake_first, a.fake_first + a.fake)]
    else: chain_list = json.load(open(a.chains_file))["chains"] if a.chains_file else [[int(x) for x in ch.split(",")] for ch in a.chains.split(";")]
    jobs = [(c, s, f"ch{i}r{r}") for i, s in enumerate(chain_list) if a.index_from <= i < a.index_to for c in a.conds.split(",") for r in range(a.rep_offset, a.rep_offset + a.reps)]
    if a.resume:
        done = set()
        for f in glob.glob(os.path.join(a.out, "gated_*.jsonl")):
            for l in open(f):
                try: r = json.loads(l); done.add((r["cond"], r["tag"]))
                except Exception: pass
        jobs = [j for j in jobs if (j[0], j[2]) not in done]; say(len(done), "already done; resuming")
    fn = os.path.join(a.out, f"gated_{a.model}_{int(time.time())}.jsonl")
    say(len(jobs), "rollouts ->", fn)
    with ProcessPoolExecutor(a.workers) as ex, open(fn, "a") as f:
        futs = {ex.submit(rollout, c, s, a, t): (c, s, t) for c, s, t in jobs}
        for fu in as_completed(futs):
            try: r = fu.result()
            except Exception as e:
                c, s, t = futs[fu]; r = {"cond": c, "tag": t, "pool_ids": s, "scores": [0.0] * len(s), "mean": 0.0, "depth": 0, "all": False, "error": f"crash {type(e).__name__}: {e}", "seconds": 0}
            f.write(json.dumps(r) + "\n"); f.flush()
            say(f"{r['cond']:7s} {r['tag']:10s} ids={r['pool_ids']} scores={[round(x, 2) for x in r['scores']]} depth={r.get('depth')} attempts={r.get('attempts')} tools={r.get('tool_calls')} {r['seconds']}s {r.get('error') or ''}")


if __name__ == "__main__":
    main()
