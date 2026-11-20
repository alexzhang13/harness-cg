"""Synthetic CorpusQA: each seed generates a corpus of periodic statistical releases or entity reports, plus a
question with an exact gold. Sub-calls go to ``CorpusOracle`` (``sub_lm="oracle"``) or the root's own model ("real").
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from synth_common import Generated, ReadModel, make_env

from . import entity as E
from . import periodic as P
from .oracle import CorpusOracle
from .scoring import root_prompt, score_answer
from .steering import CHECKS, hints_for

_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_WS = re.compile(r"\s+")
NOTE = "Please note that if different values appear for the same data across multiple reports, the most recent report shall prevail."
AUDIT_DOC = "docs/synthetic_audit/corpusqa.md"
EVAL_SHAPED_MSG = (f'surface="native" reproduces the CorpusQA eval releases and question wording (OVERLAP verdict, {AUDIT_DOC}); '
                   'it is not for training. Use the default surface="varied", or pass allow_eval_shaped=True for '
                   'reproduction / oracle tests only.')


def _surface(params: dict) -> str:
    surface = params.get("surface") or "varied"
    if surface == "native" and not params.get("allow_eval_shaped"):
        raise ValueError(EVAL_SHAPED_MSG)
    return surface


SCALES: dict[str, dict[str, tuple[tuple[int, int], tuple[int, int]]]] = {
    "1m": {"periodic": ((30, 90), (45_000, 60_000)), "company": ((20, 30), (150_000, 260_000)),
           "school": ((28, 40), (85_000, 120_000))},
    "4m": {"periodic": ((100, 160), (70_000, 110_000)), "company": ((55, 85), (180_000, 300_000)),
           "school": ((100, 140), (100_000, 140_000))},
    "10m": {"periodic": ((240, 330), (100_000, 140_000)), "company": ((120, 170), (220_000, 320_000)),
            "school": ((260, 340), (110_000, 150_000))},
}


def _scaled(rng: random.Random, scale: str, kind: str) -> tuple[int, int]:
    (nlo, nhi), (clo, chi) = SCALES[scale][kind]
    return rng.randint(nlo, nhi), int(rng.uniform(clo, chi))


def _n_docs(rng: random.Random, lo: int, hi: int) -> int:
    r = rng.random()
    if r < 0.45:
        return rng.randint(lo, min(hi, 12))
    if r < 0.8:
        return rng.randint(min(hi, 13), min(hi, 40))
    return rng.randint(min(hi, 41), hi)


def generate(info: dict) -> Generated:
    params = json.loads(info.get("synth_params") or "{}")
    split, seed = info["synth_split"], info["synth_seed"]
    rng = random.Random(f"corpusqa:{split}:{seed}")
    fam = params.get("family") or ("periodic" if rng.random() < float(params.get("p_periodic", 0.6)) else "entity")
    lo, hi = int(params.get("min_docs", 4)), int(params.get("max_docs", 90))
    scale = params.get("scale") or "128k"
    big = scale != "128k"
    gen_key = f"{split}:{seed}:{info.get('synth_params')}"
    if fam == "periodic":
        surface = _surface(params)
        dom = rng.choice(params.get("domains") or ["housing", "retail", "jobs"])
        r = rng.random()
        n = rng.randint(lo, min(hi, 13)) if r < 0.15 else (rng.randint(min(hi, 14), min(hi, 40)) if r < 0.7 else rng.randint(min(hi, 41), hi))
        total = rng.uniform(200_000, 1_600_000)
        per_doc = int(min(60_000, max(12_000, total / n, 130_000 / n)))
        if big:
            n, per_doc = _scaled(rng, scale, "periodic")
        style = rng.choice(params.get("styles") or ["html", "pipe", "fixed"])
        dev = params.get("deviant_frac")
        dev = rng.choice([0.0, 0.0, 0.1, 0.2, 0.3]) if dev is None else float(dev)
        built = P.build_corpus(rng, dom, n, per_doc, style, dev, rng.random() < float(params.get("p_shuffle", 0.3)),
                               end_year=2025 if n > 90 else None, surface=surface)
        style = built["style"]
        q = P.make_question(rng, built, dom)
        if surface == "native":
            lo_ym, hi_ym = built["reports"][0]["report"].replace("-", "/"), built["reports"][-1]["report"].replace("-", "/")
            question = f"Within the period from {lo_ym} to {hi_ym}, {q['q'][0].upper() + q['q'][1:]}  {NOTE}"
        else:
            question = P.frame_question(rng, q["q"][0].upper() + q["q"][1:], built["reports"][0]["report"],
                                        built["reports"][-1]["report"])
        corp = built["corp"]
        decimals = {f"{m['key']}:{a}": corp.decimals(a, m) for m in built["measures"] for a in ("SA", "NSA")}
        wanted = {(s[0], s[1]) for s in q["spec"]["series"]}
        required = [f"R{r}:{tid}" for r, rep in enumerate(built["reports"]) for tid, t in rep["tables"].items()
                    if (t["m"], t["adj"]) in wanted]
        hidden = {"family": "periodic", "domain": dom, "reports": built["reports"], "decimals": decimals,
                  "order": built["order"], "style": style, "deviant": built["deviant"], "spec": q["spec"],
                  "gen_key": gen_key, "scale": P.DOMAINS[dom]["scale"], "surface": built["surface"],
                  "cols": [list(c) for c in corp.cols], "question": question}
    else:
        dom = rng.choice(params.get("entity_domains") or ["company", "school"])
        n = _n_docs(rng, lo, min(hi, int(params.get("max_entities", 40))))
        if dom == "company":
            per_doc = int(rng.uniform(25_000, 110_000) if n <= 8 else rng.uniform(15_000, 45_000))
            if big:
                n, per_doc = _scaled(rng, scale, "company")
            built = E.build_company(rng, n, max(per_doc, 130_000 // n), (rng.randint(2019, 2025), rng.choice([3, 6, 9])),
                                    extras=big, noisy=bool(params.get("noisy")))
        else:
            per_doc = int(rng.uniform(15_000, 60_000) if n <= 10 else rng.uniform(10_000, 30_000))
            if big:
                n, per_doc = _scaled(rng, scale, "school")
            built = E.build_school(rng, n, max(per_doc, 130_000 // n), rng.randint(2016, 2023), names_ext=big)
        q = E.make_question(rng, built, dom)
        question = q["q"]
        home = E.CO_FIELDS if dom == "company" else E.SC_FIELDS
        required = [f"E{e}:{home[f]['sec']}" for e in range(n) for f in dict.fromkeys(q["spec"].get("fields", []))
                    if f in home]
        line_index: dict[str, list] = {}
        shared: set[str] = set()
        for e, ent in enumerate(built["entities"]):
            for k, ln in ent.get("lines", {}).items():
                for piece in _SPLIT.split(ln):
                    seg = _WS.sub(" ", piece).strip()
                    if len(seg) >= 8 and any(ch.isdigit() for ch in seg):
                        hit = line_index.setdefault(seg, [e, []])
                        if hit[0] != e:
                            shared.add(seg)
                        elif k not in hit[1]:
                            hit[1].append(k)
        for seg in shared:
            line_index.pop(seg, None)
        hidden = {"family": "entity", "domain": dom, "entities": built["entities"], "spec": q["spec"], "gen_key": gen_key,
                  "line_index": line_index}
    docs = list(built["docs"])
    gold = {"kind": q["kind"], "value": q["gold"]}
    return Generated(context=docs, root_prompt=root_prompt(question, len(docs)), items=built["items"],
                     gold=gold, hidden=hidden, required_ids=required, index_min_len=20,
                     index_factory="corpusqa_synth.oracle:make_index")


def score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    return score_answer(final, gold["value"], gold["kind"])


def load_environment(split: str = "train", num_examples: int = 2000, family: str | None = None,
                     p_periodic: float = 0.6, domains: list[str] | None = None,
                     entity_domains: list[str] | None = None, min_docs: int = 4, max_docs: int = 90, scale: str = "128k",
                     noisy_layout: bool = False,
                     deviant_frac: float | None = None, budget_chars: int = 60_000, reliable_items: int = 6,
                     agg_capacity: int = 8, max_iterations: int = 30, max_sub_calls: int | None = 300,
                     hint: str = "none", hint_prob: float = 1.0, reward_mode: str = "outcome",
                     program_weight: float = 0.2, gate_floor: float = 0.0, checks: list | None = None,
                     sub_lm: str | None = None, sub_max_tokens: int = 4096,
                     surface: str = "varied", allow_eval_shaped: bool = False, **kw: Any):
    """Synthetic CorpusQA. family: None (mix), "periodic" or "entity". surface: "varied" or "native" (eval-shaped,
    needs ``allow_eval_shaped=True``). scale: one of ``SCALES``. sub_lm: "oracle" or "real"."""
    if scale not in ("128k", *SCALES):
        raise ValueError(f"scale must be 128k | {' | '.join(SCALES)}, got {scale!r}")
    params = {"family": family, "p_periodic": p_periodic, "domains": domains, "entity_domains": entity_domains,
              "min_docs": min_docs, "max_docs": max_docs, "deviant_frac": deviant_frac,
              "scale": None if scale == "128k" else scale, "noisy": True if noisy_layout else None,
              "surface": surface, "allow_eval_shaped": True if surface == "native" and allow_eval_shaped else None}
    params = {k: v for k, v in params.items() if v is not None}
    if surface not in ("native", "varied"):
        raise ValueError(f"surface must be native | varied, got {surface!r}")
    if surface == "native" and not allow_eval_shaped:
        raise ValueError(EVAL_SHAPED_MSG)
    if sub_lm is None:
        sub_lm = "oracle"
    if sub_lm not in ("oracle", "real"):
        raise ValueError(f"sub_lm must be oracle | real, got {sub_lm!r}")
    oracle = CorpusOracle(ReadModel(budget_chars=budget_chars, reliable_items=reliable_items, miss_rate=0.6,
                                    base_err=0.0005, load_err=0.08, agg_capacity=agg_capacity, chatty=0.2, min_cov=0.6))
    env = make_env(generator=generate, oracle=oracle, score=score, split=split, num_examples=num_examples,
                   params=params, max_iterations=max_iterations, max_sub_calls=max_sub_calls,
                   hints=kw.pop("hints", None) or hints_for(family), hint=hint, hint_prob=hint_prob,
                   checks=CHECKS if checks is None else checks, reward_mode=reward_mode,
                   program_weight=program_weight, gate_floor=gate_floor, **kw)
    if sub_lm == "real":
        env._sub_llm_fn = env._sub_llm_fn_batched = None
        env._sub_sampling_args = {"max_tokens": sub_max_tokens,
                                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return env
