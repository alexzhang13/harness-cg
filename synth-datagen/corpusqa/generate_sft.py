"""Generate hinted rollouts on synthetic CorpusQA corpora (real root and sub-LM) and export hint-stripped SFT rows.

    python generate_sft.py run --base-url http://localhost:8000/v1 --start 31000 --n 500 --hint-file <note> --out <dir>
    python generate_sft.py export --out <run dir> --program probe --max-turns 6 --dest <export dir>

`run` resumes by skipping finished seeds. `export` keeps rollouts that pass the program filter and strips the hint.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from pathlib import Path
from typing import Any

NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}
ENV_NAME = "corpusqa-synth-periodic"
HARD_ANSWER_CHECKS = ("answer_format", "not_preliminary", "not_stale", "not_other_adjustment", "not_unscaled")


_TOOLCALL = re.compile(r"<tool_call>\s*<function=repl>\s*(.*?)\s*</function>\s*(?:</tool_call>)?", re.S)
_PARAM = re.compile(r"</?parameter(?:=\w+)?>")


def fence_client(base: Any):
    """Client that rewrites `<tool_call><function=repl>CODE</function>` turns into ```repl fences.

    Qwen3.8 sometimes emits that markup, which the harness would not execute. `rewrites` counts it per rollout."""
    from verifiers import OpenAIChatCompletionsClient

    class FenceClient(OpenAIChatCompletionsClient):
        def __init__(self):
            super().__init__(base)
            self.rewrites = 0

        async def get_native_response(self, prompt, model, sampling_args, tools=None, **kw):  # noqa: ANN001
            r = await super().get_native_response(prompt, model, sampling_args, tools, **kw)
            try:
                msg = r.choices[0].message
                c = msg.content or ""
                if "```repl" not in c and _TOOLCALL.search(c):
                    msg.content = _TOOLCALL.sub(lambda m: "```repl\n" + _PARAM.sub("", m.group(1)).strip() + "\n```", c)
                    self.rewrites += 1
            except (AttributeError, IndexError):
                pass
            return r

    return FenceClient()


def _msg(m: Any) -> dict:
    if isinstance(m, dict):
        return {"role": m.get("role"), "content": m.get("content") if isinstance(m.get("content"), str) else str(m.get("content") or "")}
    return {"role": getattr(m, "role", None), "content": str(getattr(m, "content", "") or "")}


def _envs(n: int, start: int, hint_file: str = "", split: str = "train", hinted: bool = True, surface: str = "varied",
          allow_eval_shaped: bool = False, sub_lm: str = "real"):
    import corpusqa_synth
    kw = dict(split=split, family="periodic", hint="program" if hinted else "none", hint_prob=1.0, sub_lm=sub_lm,
              max_iterations=20, surface=surface, allow_eval_shaped=allow_eval_shaped)
    if hint_file and hinted:
        from synth_common.program import HintSpec
        kw.update(hint="strategy", hints=HintSpec(strategy=Path(hint_file).read_text().strip()))
    small = corpusqa_synth.load_environment(num_examples=start + n, **kw)
    big = corpusqa_synth.load_environment(num_examples=start + n, scale="1m", **kw)
    return small, big


async def run(a: argparse.Namespace) -> None:
    from openai import AsyncOpenAI

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "rollouts.jsonl"
    done = {json.loads(line)["key"] for line in raw.open()} if raw.exists() else set()
    small, big = _envs(a.n, a.start, a.hint_file, a.split, not a.no_hint, a.surface, a.allow_eval_shaped, a.sub_lm)
    bases = [AsyncOpenAI(base_url=u, api_key="EMPTY", timeout=a.timeout) for u in a.base_url]
    sampling = {"max_tokens": a.max_tokens, "temperature": a.temperature, "extra_body": NO_THINK}
    sem = asyncio.Semaphore(a.concurrency)
    todo = [(i, "1m" if i % 2 else "128k") for i in range(a.start, a.start + a.n)]
    todo = [(i, sc) for i, sc in todo if f"{sc}:{i}" not in done]
    print(f"{len(todo)} rollouts to run ({len(done)} done) on {len(bases)} endpoint(s)", flush=True)

    async def one(j: int, seed: int, scale: str) -> None:
        env = big if scale == "1m" else small
        async with sem:
            t0 = time.time()
            rec: dict[str, Any] = {"key": f"{scale}:{seed}", "seed": seed, "scale": scale}
            client = fence_client(bases[j % len(bases)])
            try:
                st = await asyncio.wait_for(env._run_rollout_state(
                    input=dict(env.dataset[seed]), client=client, model=a.model,
                    sampling_args=sampling), timeout=a.rollout_timeout)
                m = {k: v for k, v in (st.get("metrics") or {}).items() if isinstance(v, (int, float))}
                hid = st.get("synth_hidden") or {}
                rec.update(reward=float(st.get("reward") or 0.0), outcome=m.get("outcome", 0.0), metrics=m,
                           iterations=int(st.get("rlm_iterations") or 0), sub_calls=int(st.get("rlm_sub_llm_calls") or 0),
                           final=str(st.get("rlm_final_answer") or ""), has_answer=st.get("rlm_final_answer") is not None,
                           domain=hid.get("domain"), tmpl=(hid.get("spec") or {}).get("tmpl"),
                           gold=(hid.get("gold") or {}).get("value"),
                           messages=[_msg(x) for x in (st.get("rlm_history") or [])],
                           sub_log=[{"prompt": str(e.get("prompt") or "")[:4000], "response": str(e.get("response") or "")[:4000]}
                                    for e in (st.get("rlm_sub_llm_log") or [])])
            except Exception as e:  # noqa: BLE001
                rec.update(reward=0.0, outcome=0.0, error=f"{type(e).__name__}: {str(e)[:300]}")
            rec["wall_s"] = round(time.time() - t0, 1)
            rec["fence_rewrites"] = client.rewrites
            with raw.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            print(json.dumps({k: rec.get(k) for k in ("key", "domain", "tmpl", "outcome", "iterations", "sub_calls",
                                                      "wall_s", "fence_rewrites", "error")}), flush=True)

    await asyncio.gather(*(one(j, i, sc) for j, (i, sc) in enumerate(todo)))
    for e in (small, big):
        try:
            await e.teardown_rlm()
        except Exception:  # noqa: BLE001
            pass


SUB_KINDS = {"spec": ("Map this question", "Map the question"), "check": ("Does this computation",),
             "survey": ("Describe its STRUCTURE",), "map": ("structure reports on excerpts",),
             "confirm": ("which of these tables holds it", "Below is every table of one document"),
             "fallback": ("Below is one table", "Return ONLY JSON {\"YYYY-MM\"", "{\"YYYY-MM\": value as printed}"),
             "plan": ("A corpus of periodic documents with tables",)}
LEAK = re.compile(r"\b(the|this|my) (note|hint)\b|as instructed|<hint>|</hint>", re.I)
NAMES_LEAK = re.compile(r"\b[Tt]urn [1-4] (code|block)\b|\b(TABLES|LEDGER|COMPUTE)\b|\bPLAN (code|step|block)\b")


def prose(text: str) -> str:
    """An assistant turn without its code blocks (program names in code are variables, not citations)."""
    return re.sub(r"```.*?(```|$)", " ", text, flags=re.S)


def sub_kinds(r: dict) -> dict:
    out = {k: 0 for k in SUB_KINDS} | {"other": 0}
    for c in r.get("sub_log") or []:
        k = next((k for k, pre in SUB_KINDS.items() if any(x in c["prompt"] for x in pre)), "other")
        out[k] += 1
    return out


def _failed_checks(r: dict) -> list[str]:
    """Hard answer checks of a stored rollout (answer_format recomputed with the current steering code)."""
    from corpusqa_synth.steering import answer_format
    m = dict(r.get("metrics") or {})
    g = r.get("gold")
    kind = "list" if isinstance(g, list) else "number" if isinstance(g, (int, float)) and not isinstance(g, bool) else "str"
    m["prog_answer_format"] = answer_format({"rlm_final_answer": r.get("final", ""), "synth_hidden": {
        "family": "periodic", "spec": {"tmpl": r.get("tmpl")}, "gold": {"kind": kind, "value": g}}})
    return [c for c in HARD_ANSWER_CHECKS if m.get(f"prog_{c}", 1.0) < 0.999]


def accept_probe(r: dict, max_turns: int) -> tuple[bool, str]:
    """Probe-v3 filter: `accept` plus the PLAN sub-call, a printed LEDGER report, no raw table text, no hint references."""
    ok, why = accept(r, max_turns)
    if not ok:
        return ok, why
    if not sub_kinds(r)["plan"]:
        return False, "no PLAN sub-call"
    outs = [x["content"] for x in r["messages"] if x["role"] == "user" and x["content"].startswith("REPL output")]
    if not any("documents by data period:" in o for o in outs):
        return False, "no LEDGER report"
    if any(re.search(r"<t[dr][ >]|^\s*\|.*\|.*\|", o, re.M) for o in outs):
        return False, "raw table text printed"
    asst = [x["content"] for x in r["messages"] if x["role"] == "assistant"]
    if any(LEAK.search(a) or "<tool_call>" in a for a in asst):
        return False, "hint reference / tool-call markup"
    if any(NAMES_LEAK.search(prose(a)) for a in asst):
        return False, "names the hint's turns / blocks"
    return True, "ok"


def profile(rs: list[dict]) -> dict:
    import statistics as st
    if not rs:
        return {}
    ks = [sub_kinds(r) for r in rs]
    return {"n": len(rs), "turns_median": st.median(r.get("iterations", 0) for r in rs),
            "turns_mean": round(st.mean(r.get("iterations", 0) for r in rs), 2),
            "sub_calls_mean": round(st.mean(r.get("sub_calls", 0) for r in rs), 2),
            **{f"with_{k}": round(sum(x[k] > 0 for x in ks) / len(rs), 2) for k in SUB_KINDS | {"other": 0}}}


def accept(r: dict, max_turns: int) -> tuple[bool, str]:
    if r.get("error"):
        return False, "error"
    if not r.get("has_answer"):
        return False, "no answer"
    if r.get("outcome", 0.0) < 1.0:
        return False, "wrong"
    if r.get("sub_calls", 0) < 1:
        return False, "no sub-call"
    if r.get("iterations", 99) > max_turns:
        return False, "too many turns"
    bad = _failed_checks(r)
    if bad:
        return False, "check:" + ",".join(bad)
    return True, "ok"


def export(a: argparse.Namespace) -> None:
    import pandas as pd
    from synth_common.program import strip_hint

    out = Path(a.out)
    rows, reasons = [], {}
    allr = [json.loads(line) for line in (out / "rollouts.jsonl").open()]
    kept = []
    for r in allr:
        ok, why = {"probe": accept_probe}.get(a.program, accept)(r, a.max_turns)
        reasons[why] = reasons.get(why, 0) + 1
        if not ok:
            continue
        kept.append(r)
        msgs = [dict(m) for m in r["messages"]]
        for m in msgs:
            if m["role"] == "user" and "<hint>" in m["content"]:
                m["content"] = strip_hint(m["content"])
        assert not any("<hint>" in m["content"] for m in msgs)
        uid = "corpusqa-synth:" + hashlib.sha1(r["key"].encode()).hexdigest()[:16]
        rows.append({"messages": msgs, "chat_template_kwargs": {"enable_thinking": False}, "env": ENV_NAME,
                     "uid": uid, "task_id": f"corpusqa-synth:{r['key']}"})
    random.Random(0).shuffle(rows)
    n_val = max(1, int(len(rows) * a.val_frac)) if len(rows) > 10 else 0
    dest = Path(a.dest or f"outputs/synth-datagen/corpusqa/export-{out.name}")
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows[n_val:]).to_parquet(dest / "train.parquet")
    with (dest / "train.jsonl").open("w") as f:
        for r in rows[n_val:]:
            f.write(json.dumps({"messages": r["messages"], "env": r["env"], "uid": r["uid"], "task": r["task_id"]}) + "\n")
    pd.DataFrame(rows[:n_val]).to_parquet(dest / "validation.parquet")
    manifest = {"src": str(out / "rollouts.jsonl"), "env": ENV_NAME, "accepted": len(rows), "train": len(rows) - n_val,
                "validation": n_val, "reasons": reasons, "max_turns": a.max_turns,
                "hard_answer_checks": list(HARD_ANSWER_CHECKS), "program_filter": a.program,
                "profile_all": profile(allr), "profile_kept": profile(kept)}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base-url", action="append", required=True)
    r.add_argument("--model", default="qwen38-27b")
    r.add_argument("--n", type=int, default=200)
    r.add_argument("--start", type=int, default=0)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--max-tokens", type=int, default=6000)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--timeout", type=float, default=900)
    r.add_argument("--rollout-timeout", type=float, default=3600)
    r.add_argument("--surface", default="varied", choices=["native", "varied"],
                   help="periodic corpora's surface: varied (default) = invented keys + restyled structure "
                        "(corpusqa_synth/surface.py); native = eval-shaped, needs --allow-eval-shaped")
    r.add_argument("--allow-eval-shaped", action="store_true",
                   help="permit --surface native (reproduction only: it copies the eval; docs/synthetic_audit/corpusqa.md)")
    r.add_argument("--split", default="train", help="generator split (val = held-out seeds for adoption probes)")
    r.add_argument("--no-hint", action="store_true", help="unhinted rollouts (adoption probe of a trained model)")
    r.add_argument("--hint-file", default="", help="note file used as the hint (default: corpusqa_synth/hint_periodic.md)")
    r.add_argument("--sub-lm", default="real", choices=["real", "oracle"],
                   help="who answers sub-calls: the served model (real) or the calibrated fake sub-LM (oracle)")
    r.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", required=True)
    e.add_argument("--dest", default="")
    e.add_argument("--max-turns", type=int, default=12)
    e.add_argument("--val-frac", type=float, default=0.05)
    e.add_argument("--program", choices=["hint", "probe"], default="probe",
                   help="procedure filter: hint (base checks only) or probe (the probe-v3 program)")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    else:
        export(a)


if __name__ == "__main__":
    main()
