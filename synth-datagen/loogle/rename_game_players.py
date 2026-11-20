"""Replace player handles in synthetic Game data that were copied from real LooGLE-v2 match logs."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

OLD = ["sirah", "Kristou", "CYPHER", "Anlelele", "nexa", "jkaem", "TMB", "pr1metapz", "nawwk", "malta", "pz",
       "apocdud", "yourwombat", "BRACE", "damyo", "hazr", "nettik", "sterling", "ADDICT", "gump", "Wicadia", "MAJ3R",
       "NertZ", "jks", "lukiz", "matios", "xureba", "PKL", "WOODY7", "Ceruttera", "hfah", "KENSi", "Norwi", "gr1ks",
       "clax", "rain", "Jaques", "Twistzz", "woxic", "skullz", "yuurih", "Aleksib", "w0nderful", "iM", "910", "degster",
       "zorte", "kyxsan", "b1t", "frozen", "Spinx", "flameZ", "mezii", "ropz", "broky", "rigoN", "Jimpphat", "torzsi"]
UP, LO, DI = "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz", "0123456789"
CONS = "bcdfghjklmnprstvwxz"


def _like(name: str, rng: random.Random) -> str:
    """Random handle with the same per-character class as `name` (vowel/consonant alternation keeps it pronounceable)."""
    out = []
    for i, ch in enumerate(name):
        pool = DI if ch.isdigit() else (UP if ch.isupper() else LO)
        if pool is not DI:
            pool = [c for c in pool if (c.lower() in "aeiouy") == (i % 2 == 1)] or list(pool)
        out.append(rng.choice(pool))
    return "".join(out)


def _orders_match(old: list[str], new: dict[str, str]) -> bool:
    for key in (None, str.lower):
        if [new[o] for o in sorted(old, key=key)] != sorted(new.values(), key=key):
            return False
    return True


def build_map_ranked(corpus: str, seed: int = 20261022) -> dict[str, str]:
    """Each new name keeps the old name's FIRST character. Names whose first letters differ (case-insensitively) then keep
    their relative order under both codepoint and case-insensitive sorting; within each first-letter group the names are
    redrawn until both orders hold. 'pr1metapz' and 'ropz' are drawn as <first char + random> + new('pz')."""
    rng = random.Random(seed)
    tokens = set(re.findall(r"[A-Za-z0-9_]+", corpus))
    groups: dict[str, list[str]] = {}
    for o in OLD:
        groups.setdefault(o[0].lower(), []).append(o)
    keys = sorted(groups, key=lambda k: (k != "p", k))
    sub = {"pr1metapz": "pr1meta", "ropz": "ro"}

    def draw(o: str, new: dict[str, str]) -> str:
        if o in sub:
            return o[0] + _like(sub[o], rng)[1:] + new["pz"]
        return o[0] + _like(o, rng)[1:]

    for _ in range(2000):
        new: dict[str, str] = {}
        for k in keys:
            olds = sorted(groups[k], key=lambda o: o in sub)
            for _t in range(20000):
                cand: dict[str, str] = {}
                for o in olds:
                    cand[o] = draw(o, {**new, **cand})
                vals = list(cand.values())
                if len({v.lower() for v in vals}) == len(vals) and \
                        [cand[o] for o in sorted(olds)] == sorted(vals) and \
                        [cand[o] for o in sorted(olds, key=str.lower)] == sorted(vals, key=str.lower) and \
                        not any(v in tokens or v.lower() in tokens for v in vals):
                    new.update(cand)
                    break
            else:
                break
        else:
            vals = list(new.values())
            if len({v.lower() for v in vals}) != len(vals) or not _orders_match(OLD, new):
                continue
            bad = any(a != b and a in b and not (a == new["pz"] and b in (new["pr1metapz"], new["ropz"]))
                      for a in vals for b in vals)
            if bad:
                continue
            return new
    raise SystemExit("could not find an order-preserving map")


def pattern(names) -> re.Pattern:
    alt = "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
    return re.compile(r"(?<![A-Za-z0-9_])(?<![0-9][,.])(" + alt + r")(?![A-Za-z0-9_])(?![.,][0-9])")


NUMERIC = {"910"}


def is_game_record(s: str, pat: re.Pattern) -> bool:
    """Only records that mention at least one non-numeric handle are Game records; '910' alone is a number."""
    return any(g.group(1) not in NUMERIC for g in pat.finditer(s))


def rename_obj(x, pat, m, skip_keys=("uid", "task", "task_id", "example_id", "source_uid")):
    if isinstance(x, str):
        return pat.sub(lambda g: m[g.group(1)], x)
    if isinstance(x, list):
        return [rename_obj(v, pat, m) for v in x]
    if isinstance(x, dict):
        return {k: (v if k in skip_keys else rename_obj(v, pat, m)) for k, v in x.items()}
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--map", type=Path, required=True, help="JSON name map; created if missing (reuse it for all files)")
    ap.add_argument("--suffix", default=".renamed")
    a = ap.parse_args()
    if a.map.exists():
        m = json.loads(a.map.read_text())
    else:
        corpus = "\n".join(p.read_text() for p in a.inputs)
        m = build_map_ranked(corpus)
        a.map.write_text(json.dumps(m, indent=1) + "\n")
    pat, newpat = pattern(m), pattern(m.values())
    for p in a.inputs:
        out = p.with_name(p.stem + a.suffix + p.suffix)
        n = hit = 0
        with p.open() as fi, out.open("w") as fo:
            for line in fi:
                r = json.loads(line)
                s0 = json.dumps(r)
                assert not newpat.search(s0), f"new name already present in {p}"
                r2 = rename_obj(r, pat, m) if is_game_record(s0, pat) else r
                s2 = json.dumps(r2)
                assert not is_game_record(s2, pat), f"old name left in {p}"
                n += 1
                hit += s2 != s0
                fo.write(json.dumps(r2, ensure_ascii=False) + "\n")
        print(f"{p} -> {out}: {n} rows, {hit} renamed", file=sys.stderr)


if __name__ == "__main__":
    main()
