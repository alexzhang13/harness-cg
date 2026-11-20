"""Synthetic BrowseComp-Plus: find one entity in a ~27k-document generated encyclopedia.

The seed picks a world (a cached corpus served as a doc-store reference) and a question. Sub-calls go to
``BCPOracle``. Reward is the real env's substring match, zeroed for answers that hedge across several candidates.

    load_environment(split="train", num_examples=2000)
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
from typing import Any

from synth_common import Generated, ReadModel, make_env
from synth_common.envkit import _metric

from .oracle import BCPOracle
from .questions import make_question
from .steering import CHECKS, HINTS
from .world import get_world

ROOT_PROMPT_SUFFIX = (
    "Docs are capped at ~2k tokens each and one `llm_query` call holds ~16k input tokens, so batch only a few docs per "
    'sub-call. When you have your final answer, set `answer["content"]` to it and `answer["ready"] = True` inside a '
    "`repl` block."
)

DEFAULTS = {"scale": "full", "n_worlds": 2, "n_val_worlds": 1, "n_clues": [4, 6], "p_rare": 1.0}


def _world_id(split: str, seed: int, p: dict) -> str:
    if split == "train":
        return f"train{seed % max(1, int(p['n_worlds']))}"
    return f"{split}{seed % max(1, int(p['n_val_worlds']))}"


def generate(info: dict) -> Generated:
    p = dict(DEFAULTS)
    p.update(json.loads(info.get("synth_params") or "{}"))
    split, seed = info["synth_split"], int(info["synth_seed"])
    wid = _world_id(split, seed, p)
    W = get_world(wid, p["scale"])
    rng = random.Random(f"bcp-q:{split}:{seed}")
    q = make_question(W, rng, tuple(p["n_clues"]), p_rare=float(p["p_rare"]))
    order = list(range(len(W.low)))
    rng.shuffle(order)
    from rlm_train import doc_store
    ctx = doc_store.make_ref(W.store_path, order, sum(W.store_chars))
    hidden = {"world": wid, "scale": p["scale"], **q.to_hidden()}
    req = []
    for e in q.chain:
        arts = [d for d in W.ent_docs.get(e, []) if W.doc_subject[d] == e and W.doc_kind[d] == "article"]
        req += [W.doc_ids[d] for d in arts[:1]]
    return Generated(context=ctx, root_prompt=f"{q.text}\n\n{ROOT_PROMPT_SUFFIX}", items=[], gold=q.gold,
                     hidden=hidden, required_ids=req, index=W.index, index_key=f"bcp-synth:{wid}:{p['scale']}")


_NON_DECOMPOSING = str.maketrans({"ı": "i", "ł": "l", "ø": "o", "đ": "d", "ð": "d", "þ": "th", "ß": "ss", "æ": "ae",
                                  "œ": "oe"})
_WORD_NUMBERS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}
_WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(_WORD_NUMBERS) + r")\b")


def normalize(text: str) -> str:
    s = str(text).lower().translate(_NON_DECOMPOSING)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"[^0-9a-z]+", " ", s)
    s = _WORD_NUMBER_RE.sub(lambda m: _WORD_NUMBERS[m.group(1)], s)
    return " ".join(s.split())


def _part_in(part: str, pred: str) -> bool:
    if part.replace(" ", "").isdigit():
        return re.search(r"(?<!\d)" + re.escape(part) + r"(?!\d)", pred) is not None
    return part in pred


def substring_score(pred: str, gold: str) -> float:
    parts = [n for n in (normalize(c) for c in str(gold).split(";")) if n]
    if not parts:
        return 0.0
    p = normalize(pred)
    return sum(1 for x in parts if _part_in(x, p)) / len(parts)


def candidate_names(pred: str, hidden: dict) -> int:
    """How many distinct entity names of the answer's type the final answer mentions (hedging detector)."""
    W = get_world(hidden["world"], hidden["scale"])
    low = normalize(pred)
    words = low.split()
    gold_t = W.ents[hidden["gold_entity"]]["type"]
    found = set()
    i = 0
    while i < len(words):
        for n in (7, 6, 5, 4, 3, 2, 1):
            if i + n > len(words):
                continue
            k = " ".join(words[i:i + n])
            e = W.name_index.get(k) or W.name_index.get("the " + k)
            if e is not None and (n > 1 or W.ents[e]["type"] == "town"):
                if W.ents[e]["type"] == gold_t:
                    found.add(e)
                i += n
                break
        else:
            i += 1
    return len(found)


def score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    if len(final) > 400:
        return 0.0
    if hidden.get("ask") in ("name", "creator", "illustrator", "spouse", "founder", "publisher", "birthplace",
                             "name_year") and candidate_names(final, hidden) >= 2:
        return 0.0
    return substring_score(final, str(gold))


def _hidden(state: Any) -> dict:
    return (state or {}).get("synth_hidden") or {}


def _target_read(st: dict, state: Any) -> float:
    req = _hidden(state).get("required_ids") or []
    return 1.0 if req and req[0] in (st.get("read_ids") or {}) else 0.0


EXTRA_METRICS = [
    _metric("bcp_alike", lambda st, s: _hidden(s).get("n_alike", 0)),
    _metric("bcp_n_clues", lambda st, s: len(_hidden(s).get("clues") or [])),
    _metric("bcp_hop_clues", lambda st, s: sum(1 for c in (_hidden(s).get("clues") or []) if c["role"] != "self")),
    _metric("bcp_target_read", _target_read),
    _metric("bcp_agg_requests", lambda st, s: st.get("agg_requests", 0)),
    _metric("bcp_answer_candidates", lambda st, s: candidate_names(str((s or {}).get("rlm_final_answer") or ""),
                                                                   _hidden(s)) if (s or {}).get("rlm_final_answer") else 0),
]


def load_environment(split: str = "train", num_examples: int = 2000, scale: str = "full", n_worlds: int = 2,
                     n_val_worlds: int = 1, n_clues: tuple[int, int] = (4, 6), max_iterations: int = 20,
                     max_sub_calls: int = 120, budget_chars: int = 64_000, reliable_items: int = 8,
                     agg_capacity: int = 4, base_err: float = 0.03, load_err: float = 0.10, p_rare: float = 1.0,
                     hint: str = "none", hint_prob: float = 1.0, reward_mode: str = "outcome",
                     program_weight: float = 0.2, gate_floor: float = 0.0, **kwargs: Any):
    """scale: "full" (~27k docs), "medium" or "small". p_rare: share of questions with one rare, searchable clue.
    budget_chars ... load_err configure the oracle's read model."""
    params = {"scale": scale, "n_worlds": n_worlds, "n_val_worlds": n_val_worlds, "n_clues": list(n_clues),
              "p_rare": p_rare}
    oracle = BCPOracle(ReadModel(budget_chars=budget_chars, reliable_items=reliable_items, miss_rate=0.5,
                                 base_err=base_err, load_err=load_err, agg_capacity=agg_capacity, chatty=0.2,
                                 min_cov=0.6))
    return make_env(generator=generate, oracle=oracle, score=score, split=split, num_examples=num_examples,
                    params=params, max_iterations=max_iterations, max_sub_calls=max_sub_calls,
                    extra_metrics=EXTRA_METRICS, rubric_kwargs={"min_iterations": 2}, hints=HINTS, hint=hint,
                    hint_prob=hint_prob, checks=CHECKS, reward_mode=reward_mode, program_weight=program_weight,
                    gate_floor=gate_floor, **kwargs)


__all__ = ["ROOT_PROMPT_SUFFIX", "generate", "load_environment", "score", "substring_score"]
