#!/usr/bin/env python3
"""Teacher levels for the gated-chain SFT corpus: chains of 1-4 fake tasks (fake_world.py) behind a GatedChain.

    python generate_teacher_levels.py --base-url URL --model rlm-27b --chains 950 --header post \
        --teacher-hint-share 0.6 --out runs/teacher_levels

Each level is an rlm.RLM run by the teacher (RLM-27B, T 0.3) with scripted sub-call replies. Its submit is
rewritten into an unlock call, followed by scripted handoff / submit turns depending on the gate's reply. A share of
levels gets a work-style note, removed from the saved prompts. Writes levels.jsonl and chains.jsonl; resumable.
"""
import argparse, json, os, random, re, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
from gated_chain import GatedChain
import fake_world as FW
import run_gated as G
from tasks import slim, say
from rlm import RLM
from rlm.core.rlm import RLMIteration
from rlm.core.types import CodeBlock
from rlm.logger import RLMLogger
from rlm.utils.parsing import find_code_blocks

TEACHER_HINT = ("Work style for this task: after at most one quick look at the data, write ONE complete, self-contained program in a single repl block: parse everything you need, "
                "compute the answer, print the evidence, and set the answer at the end of that same block. Helper functions and comments are fine; a thorough program of 30 or more lines is expected.")
PHASED_HINT = ("Work style for this task: (1) take one quick look at the data to learn its structure; (2) if the question needs several kinds of facts, make ONE llm_query call to interpret it "
               "(what to extract and how); (3) then write one large, complete program that builds a full table of every relevant record from ALL of the data (with helper functions and checks that every "
               "record was read or reported as unparsed), computes the answer from that table, prints the evidence, and sets the answer at the end of that same block. Aim for 60 or more lines in that final program.")
TEACHER_HINTS = {"one": TEACHER_HINT, "phased": PHASED_HINT}
STATUS = ('print(result if isinstance(result, str) else f"unlocked: next task received (stage {result[\'stage\']} of {result[\'of\']})")')
HAND_PROSE = ["The answer was accepted and the next task is unlocked. I'll hand it to a fresh level.",
              "That unlocked the next task. I'll pass it on to a fresh level rather than solving it here.",
              "Accepted. The next task is now available; handing it off."]
FINAL_PROSE = ["Everything below this level is done; submitting my own answer.",
               "The rest of the sequence has been handled. Submitting my answer for this task.",
               "Nothing more to do here; submitting my answer."]
STOP_PROSE = ["The gate is closed, so there is nothing further to unlock. Submitting my answer for this task.",
              "No attempts are left; I'll submit my answer and stop."]


def unlock_code(candidate): return f"result = unlock({candidate!r})\n{STATUS}"
def final_code(candidate): return f'answer["content"] = {candidate!r}\nanswer["ready"] = True'
FINAL_VAR = 'answer["content"] = my_answer\nanswer["ready"] = True'


def rewrite_submit(code):
    """Rewrite the submit in the model's code into `result = unlock(my_answer)`; None if not recognised."""
    c = re.sub(r"answer\[['\"]content['\"]\]", "my_answer", code)
    c = re.sub(r"(?m)^[ \t]*answer\[['\"]ready['\"]\][ \t]*=[ \t]*True[ \t]*(#.*)?\n?", "", c)
    if "answer[" in c or re.search(r"(?m)^\s*answer\s*=", c) or "answer.update" in c or "my_answer" not in c: return None
    try: compile(c, "<rewrite>", "exec")
    except SyntaxError: return None
    return c.rstrip() + "\n" + unlock_code_var()


def unlock_code_var(): return "result = unlock(my_answer)\n" + STATUS


class TeacherRLM(RLM):
    """RLM whose submit goes through the gate. Set `_rng` before use."""
    _rng = None

    def _forced_iteration(self, prompt, environment, response):
        t0 = time.perf_counter()
        blocks = [CodeBlock(code=c, result=environment.execute_code(c)) for c in find_code_blocks(response)]
        return RLMIteration(prompt=prompt, response=response, code_blocks=blocks, iteration_time=time.perf_counter() - t0)

    def _completion_turn(self, prompt, lm_handler, environment):
        queue = self.__dict__.setdefault("_queue", [])
        if queue:
            prose, code = queue.pop(0)
            return self._forced_iteration(prompt, environment, f"{prose}\n\n```repl\n{code}\n```")
        it = super()._completion_turn(prompt, lm_handler, environment)
        cand = next((b.result.final_answer for b in it.code_blocks if getattr(b.result, "final_answer", None) is not None), None)
        if cand is None:
            return it
        self._candidate = str(cand)
        prose = it.response.split("```repl")[0].strip() or "I have my answer; submitting it to the gate."
        blocks = find_code_blocks(it.response); rewritten = None
        for k in range(len(blocks) - 1, -1, -1):
            if "answer[" in blocks[k]:
                rw = rewrite_submit(blocks[k]); rewritten = (k, rw); break
        if rewritten and rewritten[1]:
            k, rw = rewritten; new_blocks = list(blocks); new_blocks[k] = rw; new_blocks = new_blocks[:k + 1]
            body = "\n\n".join(f"```repl\n{b}\n```" for b in new_blocks)
            self._literal_fallback = getattr(self, "_literal_fallback", False)
            forced = self._forced_iteration(prompt, environment, f"{prose}\n\n{body}")
            self._have_var = True
        else:
            self._literal_fallback = True; self._have_var = False
            forced = self._forced_iteration(prompt, environment, f"{prose}\n\n```repl\n{unlock_code(self._candidate)}\n```")
        out = "".join((b.result.stdout or "") for b in forced.code_blocks)
        rng = self._rng or random
        if "unlocked: next task received" in out:
            queue += [(rng.choice(HAND_PROSE), "down = handoff(result)\nprint('handoff returned', len(down), 'result(s)')"), (rng.choice(FINAL_PROSE), FINAL_VAR if self._have_var else final_code(self._candidate))]
        elif "LOCKED: wrong answer" in out:
            pass
        elif "LOCKED" in out:
            queue.append((rng.choice(STOP_PROSE), FINAL_VAR if self._have_var else final_code(self._candidate)))
        else:
            queue.append((rng.choice(FINAL_PROSE), FINAL_VAR if self._have_var else final_code(self._candidate)))
        return forced


class TeacherRun(G.GRun):
    """GRun with TeacherRLM levels; one dialogue row per level."""
    def __init__(self, a, chain, items, rng):
        super().__init__(a, "gplain", chain); self.items, self.rng, self.rows = items, rng, []

    def level(self, pkg, idx):
        stage = pkg["stage"] - 1
        unlock_b = self.chain.unlock_for(stage)
        def unlock(answer):
            with self.lock: self.tool_calls["unlock"] += 1
            return unlock_b(answer)
        def handoff(p):
            with self.lock: self.tool_calls["handoff"] += 1
            return self.level(p, idx + 1) if isinstance(p, dict) and "question" in p and "stage" in p else "handoff expects the task package returned by unlock"
        tools = {"unlock": {"tool": unlock, "description": G.UNLOCK_DESC}, "handoff": {"tool": handoff, "description": G.HANDOFF_DESC}}
        logger = RLMLogger(); t0 = time.time()
        rlm = TeacherRLM(**self.kw(logger), custom_tools=tools); rlm._rng = self.rng
        mix = getattr(self.a, "hint_mix", None) or ({"one": self.a.teacher_hint_share} if getattr(self.a, "teacher_hint_share", 0.0) else {})
        u, hinted, acc = self.rng.random(), None, 0.0
        for k, p in mix.items():
            acc += p
            if u < acc: hinted = k; break
        root = G.level_root(self.a, "gplain", pkg, self.chain.n) + (("\n\n" + TEACHER_HINTS[hinted]) if hinted else "")
        r = rlm.completion(pkg["context"], root_prompt=root)
        meta = slim(r.metadata); its = meta["iterations"]; last = its[-1]
        msgs = [dict(m) for m in last["prompt"]] + [{"role": "assistant", "content": last["response"]}]
        for m in msgs:
            for h in TEACHER_HINTS.values(): m["content"] = m["content"].replace("\n\n" + h, "")
        item = self.items[stage]
        with self.lock:
            self.rows.append({"messages": msgs, "depth": idx, "n": len(self.items), "trunc": 0, "stage": stage, "kind": item.get("family") or item["kind"], "answer": getattr(rlm, "_candidate", None),
                              "gold": item["gold"], "turns": len(its), "seconds": round(time.time() - t0, 1), "literal_fallback": getattr(rlm, "_literal_fallback", False), "teacher_hint": hinted or False,
                              "submitted": [e[2] for e in self.chain.log if e[0] == stage]})
        return [r.response]


def one_chain(cid, a):
    rng = random.Random(a.seed * 100003 + cid)
    n = rng.choices([1, 2, 3, 4], weights=[1, 2, 3, 4])[0]
    FW.install_fake_reader(); items = G.fake_items(a.seed * 100003 + cid, n); loaders = G.fake_loaders(items)
    chain = GatedChain(loaders, max_attempts=a.attempts)
    run = TeacherRun(a, chain, items, rng); err = None
    try: run.level(chain.package(0), 0)
    except Exception as e: err = f"{type(e).__name__}: {e}"
    st = chain.stats()
    for r in run.rows:
        r.update(pool_ids=[a.seed * 1000000 + cid], accepted=r["stage"] in chain.accepted, attempts=chain.attempts[r["stage"]], chain_depth=st["depth"], chain_n=n)
    return {"cid": cid, "n": n, "rows": run.rows, "error": err, "depth": st["depth"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--chains", type=int, default=450); ap.add_argument("--first", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=24); ap.add_argument("--temperature", type=float, default=0.3); ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--max-iterations", type=int, default=24); ap.add_argument("--attempts", type=int, default=2); ap.add_argument("--timeout", type=float, default=0)
    ap.add_argument("--teacher-hint-share", type=float, default=0.0, help="share of levels whose teacher run gets the one-program work-style hint (stripped from the saved prompts)")
    ap.add_argument("--teacher-hint-mix", default="", help='e.g. "one=0.3,phased=0.4": shares of levels that get each teacher-only hint (rest: none)')
    ap.add_argument("--header", choices=["none", "post", "pre"], default="none"); ap.add_argument("--header-pos", default="post")
    ap.add_argument("--sub-thinking", action="store_true", default=True); ap.add_argument("--save-traj", action="store_true", default=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.header in ("post", "pre"): a.header_pos = a.header
    a.hint_mix = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.teacher_hint_mix.split(",") if kv} if a.teacher_hint_mix else None
    os.makedirs(a.out, exist_ok=True)
    done = set(); fn = os.path.join(a.out, "levels.jsonl"); cf = os.path.join(a.out, "chains.jsonl")
    if os.path.exists(cf): done = {json.loads(l)["cid"] for l in open(cf)}
    todo = [c for c in range(a.first, a.first + a.chains) if c not in done]; say(len(todo), "chains ->", a.out)
    with ProcessPoolExecutor(a.workers) as ex, open(fn, "a") as f, open(cf, "a") as cfh:
        futs = {ex.submit(one_chain, c, a): c for c in todo}
        for fu in as_completed(futs):
            try: r = fu.result()
            except Exception as e: r = {"cid": futs[fu], "n": 0, "rows": [], "error": f"crash {type(e).__name__}: {e}", "depth": 0}
            for row in r["rows"]: f.write(json.dumps(row) + "\n")
            f.flush(); cfh.write(json.dumps({"cid": r["cid"], "n": r["n"], "levels": len(r["rows"]), "depth": r["depth"], "error": r["error"]}) + "\n"); cfh.flush()
            say(f"chain {r['cid']:5d} n={r['n']} depth={r['depth']} levels={len(r['rows'])} {r['error'] or ''}")


if __name__ == "__main__":
    main()
