"""Filter hinted synthetic ELB rollouts (gen_synth.py) and export hint-stripped SFT rows.

    python export_synth_sft.py --gen <gen dir> [<gen dir> ...] --out <export.jsonl> [--elb-contam]

Keeps rollouts that submitted cleanly, reached the reward floor, passed the hard checks, followed their note, never
mention the hint, and strip byte-exactly. ``--elb-contam`` drops rows sharing a 12-gram with real ELB items.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from expertlongbench_synth import forms, mix
from synth_common.program import strip_hint

_HINT_EXACT_RE = re.compile(r"\n*<hint>.*?</hint>", re.S)


def strip_hint_exact(text: str) -> str:
    """Byte-exact hint removal. synth_common.program.strip_hint replaces the block with ONE newline, which turns the
        metadata message's "<root prompt>
    """
    return _HINT_EXACT_RE.sub("", text)


def unhinted_metadata_prefix(r: dict, pre_v5: bool = False) -> str:
    """'Answer the following: <unhinted root prompt>"""
    P = json.loads(r["synth_params"])
    P.pop("hint", None)
    P.pop("hint_prob", None)
    if pre_v5:
        P["docket"] = {**P.get("docket", {}), "t01v5": False}
    g = mix.generate({"synth_split": r["split"], "synth_seed": r["seed"], "synth_params": json.dumps(P)})
    return f"Answer the following: {g.root_prompt}\n\n"

MIN_REWARD = {"docket": 0.85, "visit": 0.85, "tutor": 0.85, "recipe": 0.66}
CITE_RE = re.compile(r"\bthe (?:hint|instructions?|template|guidelines?|guidance)\b|\bas instructed\b|"
                     r"\bthe note (?:says|said|states|tells|requires|instructs|asks|specifies)\b|"
                     r"\b(?:per|following|according to) the (?:hint|note|instructions?|template|rules?|guidelines?|conventions?|program)\b|"
                     r"\bthe rules? (?:say|says|state|states|require|requires)\b|\bI was told\b|\b(?:provided|prescribed|given) (?:program|code|template|turn)\b",
                     re.I)
FENCE_RE = re.compile(r"```.*?(?:```|$)", re.S)
TOOL_CALL_RE = re.compile(r"</?tool_call>|<function=[\w.-]+>|<parameter=[\w.-]+>")
WATCHDOG_RE = re.compile(r"parent watchdog|Worker error:|worker SIGKILLed|Connection lost")


def tool_call_turns(msgs: list[dict]) -> int:
    """Assistant turns (after the system + metadata messages) that emit a native tool call instead of ```repl."""
    return sum(1 for m in msgs[2:] if m["role"] == "assistant" and TOOL_CALL_RE.search(m["content"] or ""))


def hit_watchdog(r: dict) -> bool:
    if r.get("stop_condition") == "repl_worker_dead":
        return True
    return any(m["role"] == "user" and WATCHDOG_RE.search(m["content"] or "") for m in (r.get("messages") or [])[2:])
WORD_RE = re.compile(r"[a-z0-9]+")


def note_of(kind: str, r: dict) -> str:
    """The note a row carried, read off its hint."""
    if kind == "docket":
        from expertlongbench_synth.steering import note_version
        return f"t01_note_{note_version({'rlm_history': r['messages']})}"
    from expertlongbench_synth.final_notes import note_version_of
    return note_version_of(kind, hint_of(r["messages"]))


def check_state(r: dict) -> dict:
    """State-like dict for running the env checks on a saved rollout."""
    return {"synth_hidden": r["hidden"], "rlm_history": r["messages"], "rlm_sub_llm_log": r.get("subcalls") or [],
            "rlm_iterations": r.get("iterations")}


def words(s: str) -> list[str]:
    return WORD_RE.findall(s.lower())


def ngrams(ws: list[str], n: int) -> set[int]:
    return {hash(tuple(ws[i:i + n])) for i in range(len(ws) - n + 1)}


def hint_of(msgs: list[dict]) -> str:
    for m in msgs:
        if m["role"] in ("system", "user") and "<hint>" in m["content"]:
            return m["content"].split("<hint>", 1)[1].split("</hint>", 1)[0]
    return ""


def hint_prose(hint: str) -> list[str]:
    """The note's prose lines (outside ``` fences), >= 8 words."""
    out, inside = [], False
    for ln in hint.splitlines():
        if ln.strip().startswith("```"):
            inside = not inside
            continue
        if not inside and len(words(ln)) >= 8:
            out.append(ln.strip())
    return out


def hint_code_lines(hint: str) -> set[str]:
    out, inside = set(), False
    for ln in hint.splitlines():
        if ln.strip().startswith("```"):
            inside = not inside
            continue
        if inside and ln.strip():
            out.add(ln.strip())
    return out


def adherence(r: dict) -> tuple[bool, str]:
    """Did the trace follow its note? Uses the env's own follows_note checks, plus visit-type and recipe-field checks."""
    kind, m = r["kind"], r.get("metrics") or {}
    if kind == "docket":
        return m.get("prog_follows_note") == 1.0, "follows_note"
    s = check_state(r)
    parts = forms.follows_note_parts(s)
    if parts is None:
        return False, "follows_note_unhinted"
    bad = [k for k, v in parts.items() if not v]
    if bad:
        return False, "follows_note:" + bad[0]
    if kind == "visit":
        v = forms._visit_type(s)
        return v == 1.0, "visit_type" if v is not None else "visit_type_unset"
    if kind == "tutor":
        return True, "tutor_program"
    if kind == "recipe":
        ans = forms.parse_answer(r.get("final_answer") or "")
        na = {k: forms_is_na(v) for k, v in ans.items()}
        excess = bool((r["hidden"].get("recipe") or {}).get("excess"))
        ok = bool(ans)
        for k, isna in na.items():
            if "temperature" in k.lower():
                ok &= not isna
            elif "physical and chemical" in k.lower():
                ok &= (not isna) == excess
            else:
                ok &= isna
        return ok, "recipe_program"
    return False, "unknown_kind"


def forms_is_na(v: str) -> bool:
    return str(v).strip().lower() in {"n/a", "na", ""}


def root_prose(msgs: list[dict]) -> str:
    return "\n".join(FENCE_RE.sub(" ", m["content"]) for m in msgs[2:] if m["role"] == "assistant")


def load_elb_ngrams(n: int = 12) -> set[int]:
    from huggingface_hub import hf_hub_download
    out: set[int] = set()
    for fn in ("T01LegalMDS.jsonl", "T03MaterialSEG.jsonl", "T04EduPAE.jsonl", "T06HealthCNG.jsonl"):
        for ln in open(hf_hub_download("launch/ExpertLongBench", fn, repo_type="dataset")):
            x = json.loads(ln)
            txt = " ".join([str(x.get("input") or ""), str(x.get("raw_human_reference") or ""),
                            " ".join(str(v) for v in (x.get("human_reference_checklist") or {}).values())])
            out |= ngrams(words(txt), n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", nargs="+", required=True, help="gen_synth.py --out dirs (with <kind>/rollouts.jsonl)")
    ap.add_argument("--kinds", nargs="+", default=list(mix.KINDS))
    ap.add_argument("--min-reward", nargs="*", default=[], help="kind=x overrides of MIN_REWARD")
    ap.add_argument("--drop-hint-prose", action="store_true", help="drop (not just log) note-prose echoes")
    ap.add_argument("--elb-contam", action="store_true", help="drop rows sharing a 12-gram with real ELB T01/T03/T04/T06")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    thr = dict(MIN_REWARD, **{k: float(v) for k, v in (x.split("=") for x in a.min_reward)})
    elb = load_elb_ngrams() if a.elb_contam else None
    drops: dict[str, Counter] = defaultdict(Counter)
    stats: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seen: set[str] = set()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with open(out, "w") as fh:
        for g in a.gen:
            for kind in a.kinds:
                p = Path(g) / kind / "rollouts.jsonl"
                if not p.exists():
                    continue
                for ln in open(p):
                    r = json.loads(ln)
                    uid = f"elbsynth:{kind}:{r.get('split')}:{r.get('seed')}"
                    if r.get("error") or uid in seen:
                        continue
                    seen.add(uid)
                    D, S = drops[kind], stats[kind]
                    D["total"] += 1
                    msgs, m = r["messages"], r.get("metrics") or {}
                    prose = root_prose(msgs)
                    hint = hint_of(msgs)
                    ment = bool(forms.HINT_MENTION_RE.search(prose))
                    S["reward"].append(r["reward"])
                    S["turns"].append(r["iterations"])
                    S["subs"].append(r["sub_calls"])
                    S["hint_mention"].append(ment)
                    adh, _adh_name = adherence(r)
                    S["adherent"].append(adh)
                    S["reasoning_slip"].append(any(x["reasoning_content"] for x in msgs if x["role"] == "assistant"))
                    reason = None
                    S["tool_call"].append(tool_call_turns(msgs) > 0)
                    if hit_watchdog(r):
                        reason = "repl_watchdog"
                    elif r.get("stop_condition") != "has_final_answer" or r.get("final_answer") is None:
                        reason = "not_submitted"
                    elif tool_call_turns(msgs):
                        reason = "tool_call"
                    elif not r.get("hinted") or not hint:
                        reason = "unhinted"
                    elif r["reward"] < thr[kind]:
                        reason = "reward"
                    elif m.get("prog_json_answer") != 1.0:
                        reason = "json_answer"
                    elif m.get("prog_na_token") != 1.0:
                        reason = "na_token"
                    elif not adh:
                        reason = "adherence:" + _adh_name
                    elif ment:
                        reason = "hint_mention"
                    if reason is None:
                        own = hint_code_lines(hint)
                        for x in msgs[2:]:
                            if x["role"] != "assistant":
                                continue
                            for line in x["content"].splitlines():
                                if line.strip() not in own and CITE_RE.search(line):
                                    reason = "cites_hint"
                                    break
                            if reason:
                                break
                    pw = ngrams(words(prose), 10)
                    echo = any(ngrams(words(h), 10) & pw for h in hint_prose(hint))
                    S["echo"].append(echo)
                    if reason is None and echo and a.drop_hint_prose:
                        reason = "hint_prose"
                    if reason is not None:
                        D[reason] += 1
                        continue
                    clean = [{**x, "content": strip_hint_exact(x["content"]) if x["role"] in ("system", "user") else x["content"]}
                             for x in msgs]
                    while clean and clean[-1]["role"] != "assistant":
                        clean.pop()
                    if not (clean[1]["content"].startswith(unhinted_metadata_prefix(r)) or (
                            kind == "docket" and clean[1]["content"].startswith(unhinted_metadata_prefix(r, pre_v5=True)))):
                        D["strip_mismatch"] += 1
                        continue
                    assert strip_hint(msgs[1]["content"]).replace("\n", "") == clean[1]["content"].replace("\n", "")
                    blob = "\n".join(x["content"] for x in clean)
                    pblob = "\n".join(x["content"] for x in clean if x["role"] in ("system", "user"))
                    if "<hint>" in blob or "</hint>" in blob or any(pl in pblob for pl in hint_prose(hint)):
                        D["leak"] += 1
                        continue
                    if elb is not None:
                        inp = ngrams(words("\n".join(x["content"] for x in clean if x["role"] != "assistant")), 12)
                        out_ng = ngrams(words("\n".join(x["content"] for x in clean if x["role"] == "assistant")), 12)
                        if (out_ng - inp) & elb:
                            D["elb_text"] += 1
                            continue
                    subs = r.get("subcalls") or []
                    row = {
                        "messages": clean, "env": f"expertlongbench-synth-{kind}", "uid": uid, "task": kind,
                        "synthetic": True, "descriptor": f"expertlongbench-synth-{kind}-note", "program": note_of(kind, r),
                        "teacher_model": "qwen38-27b", "reward": r["reward"], "n_turns": r["iterations"],
                        "n_sub_calls": r["sub_calls"],
                        "sub_prompt_chars": sum(len(s["prompt"]) for s in subs),
                        "sub_response_chars": sum(len(s["response"]) for s in subs),
                        "repl_output_chars": sum(len(x["content"]) for x in clean[3:] if x["role"] == "user"),
                        "subcalls": subs,
                        "meta": {"env": f"expertlongbench-synth-{kind}", "uid": uid, "task": kind, "split": r.get("split"),
                                 "seed": r.get("seed"), "gen_dir": g, "sampling": r.get("sampling"),
                                 "note": note_of(kind, r), "metrics": m},
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    D["kept"] += 1
                    n_out += 1
    print(f"{'kind':8s} {'n':>4s} {'reward':>6s} {'turns':>5s} {'subs':>5s} {'adher':>5s} {'ment':>5s} {'slip':>5s} {'echo':>5s} "
          f"{'kept':>5s}  drops")
    for kind in a.kinds:
        D, S = drops[kind], stats[kind]
        if not D["total"]:
            continue
        mean = lambda v: statistics.mean(float(x) for x in v) if v else 0.0  # noqa: E731
        rest = {k: v for k, v in D.items() if k not in ("total", "kept")}
        print(f"{kind:8s} {D['total']:4d} {mean(S['reward']):6.3f} {mean(S['turns']):5.2f} {mean(S['subs']):5.1f} "
              f"{mean(S['adherent']):5.2f} {mean(S['hint_mention']):5.2f} {mean(S['reasoning_slip']):5.2f} {mean(S['echo']):5.2f} "
              f"{D['kept']:3d}/{D['total']:<3d} {dict(rest)}  (min_reward {thr[kind]})")
    print(f"-> {out} ({n_out} rows)")


if __name__ == "__main__":
    main()
