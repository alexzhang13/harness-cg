"""Task pool of the gated chains (data/pool.json): item(pool_id) loads a task from its benchmark env, scored by
that benchmark's scorer (items_longreason.py, items_loogle.py, items_cqa.py).
"""
import json
import os
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
POOL = json.load(open(os.environ.get("GATED_POOL") or (H / "data" / "pool.json")))
_ADAPT = {"longreason": "items_longreason", "loogle": "items_loogle", "corpusqa": "items_cqa"}
_ITEMS = {}


def item(pool_id):
    p = POOL[pool_id]
    if pool_id not in _ITEMS:
        mod = __import__(_ADAPT[p["env"]])
        it = mod.load_items([p["key"]])[p["key"]]
        it["_mod"] = mod
        it["pool_id"] = pool_id
        _ITEMS[pool_id] = it
    return _ITEMS[pool_id]


def say(*a):
    sys.__stdout__.write(" ".join(map(str, a)) + "\n")
    sys.__stdout__.flush()


def slim(m):
    """Keep only the last prompt of each RLM level; drop REPL variable snapshots."""
    its = m.get("iterations", [])
    for k, it in enumerate(its):
        if k < len(its) - 1:
            it.pop("prompt", None)
        for cb in it.get("code_blocks", []):
            (cb.get("result") or {}).pop("locals", None)
            for c in (cb.get("result") or {}).get("rlm_calls", []) or []:
                if c.get("metadata"):
                    slim(c["metadata"])
    return m
