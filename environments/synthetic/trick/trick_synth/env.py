"""trick_synth: OBLIQ-style ranking over generated problems, gold = problems solved by the query's hidden technique."""
from __future__ import annotations
import hashlib, json, math, random, re
from typing import Any
from synth_common.envkit import Generated, make_env
from synth_common.segindex import SegmentIndex
from . import data as D
from .oracle import TrickOracle, TrickReadModel, _norm

FRAMING = ("This is an analogical retrieval task. The query is a full problem. Find other problems in the candidate "
           "pool that are solved by the SAME core technique -- the same decisive \"aha\" move -- even when the surface "
           "story, setting and vocabulary are completely different.")
_ANSWER_PROTOCOL = (
    "This is a RANKING task: the candidate pool mixes a handful of truly relevant documents with many irrelevant "
    "distractors, and you are scored by where the relevant ones land in your ranking (nDCG@50 / recall@k). When you "
    "are done, inside a `repl` block set `answer[\"content\"]` to a list of candidate `doc_id` strings ordered from MOST "
    "to LEAST relevant (e.g. answer[\"content\"] = [\"id_a\", \"id_b\", ...]). Use the BARE id -- the text after "
    "'doc_id:' in each candidate, with NO 'doc_id:' prefix. Then set answer[\"ready\"] = True. Rank ALL candidates, "
    "best first -- put the ones you are unsure about lower rather than dropping them, so borderline positives still "
    "count toward recall@50 and recall@100.")

def root_prompt(query: str, n: int) -> str:
    return (f"{FRAMING}\n\nQuery:\nGiven the following problem:\n\n{query}\n\nFind other problems that require the same "
            f"core technique -- the same eureka insight -- even if they come from completely different settings.\n\n"
            f"The REPL variable `context` is a list of {n} candidate documents, each rendered over multiple lines as "
            f"'---\\ndoc_id: <id>\\n---\\n<text>' (so `e.split(chr(10))[1].split(': ', 1)[1]` is the id of entry `e`). "
            f"Rank them by relevance to the query above. {_ANSWER_PROTOCOL}")

DEFAULTS = {"pool_size": [2500, 3100], "gold_k": [1, 2, 3, 5, 7, 10, 13, 20, 30]}
_EMPTY = SegmentIndex([])

def _unique_keys(alias, rows, I, q=None, W=50, step=10, max_off=160):
    """Per pool item, the earliest short window of its text that no other item contains (decoys share openings)."""
    texts = {a: _norm(I[r]["text"]) for a, r in zip(alias, rows)}
    allt = list(texts.values()) + ([_norm(I[q]["text"])] if q is not None else [])
    blob = "\x00".join(allt)
    out = []
    for a, t in texts.items():
        key = None
        for off in range(0, min(max_off, max(1, len(t) - W)) + 1, step):
            w = t[off:off + W]
            if len(w) < 30:
                break
            if blob.count(w) == 1:
                key = (w, off); break
        out.append((a,) + (key or (t[:W], 0)))
    return out

def generate(info: dict) -> Generated:
    p = json.loads(info.get("synth_params") or "{}")
    split, seed = info["synth_split"], int(info["synth_seed"])
    C = D.corpus(); I, T = C["I"], C["T"]
    rng = random.Random(f"trick|{split}|{seed}")
    techs = [t["id"] for t in T if (t["id"] in C["val"]) == (split != "train") and len(C["by_tech"][t["id"]]) >= 4]
    qt = rng.choice(techs)
    members = list(C["by_tech"][qt]); rng.shuffle(members)
    q = members[0]
    k = min(len(members) - 1, rng.choice(p.get("gold_k", DEFAULTS["gold_k"])))
    gold_rows = members[1:1 + k]
    others = [i for i in range(len(I)) if I[i]["tech"] != qt]
    linked = {I[i]["iid"] for i in [q] + gold_rows}
    hard = [i for i in others if I[i].get("decoy_of") in linked or (I[i].get("iid") in {I[g].get("decoy_of") for g in [q] + gold_rows})]
    if "decoy_tech" in I[q]:
        lookalike = I[q]["decoy_tech"]
        hard = [i for i in others if I[i]["tech"] == lookalike or I[i].get("decoy_tech") == lookalike][:300]
    ps = p.get("pool_size", DEFAULTS["pool_size"])
    n = rng.randint(*ps) if isinstance(ps, list) else int(ps)
    rest = [i for i in others if i not in set(hard)]
    rows = gold_rows + hard + rng.sample(rest, max(0, min(len(rest), n - len(gold_rows) - len(hard))))
    rng.shuffle(rows)
    salt = f"{split}:{seed}"
    alias = ["d" + hashlib.sha1(f"{salt}|{r}".encode()).hexdigest()[:10] for r in rows]
    context = [f"---\ndoc_id: {a}\n---\n{I[r]['text']}" for a, r in zip(alias, rows)]
    gold = {alias[i]: 1 for i, r in enumerate(rows) if r in set(gold_rows)}
    hidden = {"key": f"{salt}", "qtech": qt, "qiid": q,
              "tech": {a: I[r]["tech"] for a, r in zip(alias, rows)},
              "sketch": {a: I[r]["sketch"] for a, r in zip(alias, rows)},
              "_row": {a: r for a, r in zip(alias, rows)},
              "_keys": _unique_keys(alias, rows, I, q),
              "_qkey": _norm(I[q]["text"])[:60],
              "_qwords": sorted(set(re.findall(r"[a-z]{4,}", I[q]["text"].lower())))}
    return Generated(context=context, root_prompt=root_prompt(I[q]["text"], len(rows)), items=[], gold=gold,
                     hidden=hidden, required_ids=list(gold), index=_EMPTY, index_key="trick-empty")

def _ndcg(rank, rel, k=50):
    dcg = sum(rel.get(c, 0) / math.log2(i + 2) for i, c in enumerate(rank[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]; idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0

def _parse(final: Any) -> list[str]:
    import ast
    if isinstance(final, (list, tuple)): L = list(final)
    else:
        s = str(final or "").strip()
        try: L = json.loads(s)
        except Exception:
            try: L = ast.literal_eval(s)
            except Exception: L = [x for x in s.splitlines() if x.strip()]
    return [str(x).strip().strip("\"',[](){} ").replace("doc_id:", "").strip() for x in (L if isinstance(L, list) else [])]

def score(final: str, gold: Any, hidden: dict, state: Any) -> float:
    return _ndcg(_parse(final), gold or {})

def load_environment(split: str = "train", num_examples: int = 2000, pool_size=None, read: dict | None = None,
                     max_iterations: int = 20, max_sub_calls: int | None = 400, hint: str = "none",
                     hint_prob: float = 1.0, **kwargs: Any):
    params = {} if pool_size is None else {"pool_size": pool_size}
    from .oracle import PRESETS
    read = dict(read or {}); preset = read.pop("preset", None)
    oracle = TrickOracle(TrickReadModel(**{**(PRESETS[preset] if preset else {}), **read}))
    return make_env(generator=generate, oracle=oracle, score=score, split=split, num_examples=num_examples,
                    params=params, max_iterations=max_iterations, max_sub_calls=max_sub_calls, hint=hint,
                    hint_prob=hint_prob, **kwargs)
