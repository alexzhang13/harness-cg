"""Code family: synthetic Python repositories -> LooGLE-v2 Call Graph Analysis (and a few Version Control) questions."""

from __future__ import annotations

import json
import random
import re
from collections import deque
from dataclasses import dataclass, field

from synth_common.envkit import Generated
from synth_common.oracle import FakeSubLM, refuse_aggregate, rng_for, wants_json

from .textbank import DOC_PARAS, code_comment, code_stmt

INSTRUCTION = ("You are a senior Python expert. You will now receive a full source code of a Python project. The text "
               "below contains all the source code in the format '<File>: relative_path\n', followed by the content of "
               "each file.\n<text> {context} </text>\nYour task is to analyze the call graph of the project and answer "
               "the following multiple choice question.\n<question> {question} </question>\nThe options are as follows:\n"
               "<option> {options} </option>\nThere is only one correct option. You must ensure that your answer is one "
               "of the given option letters.\nPlease select the correct option and only response a single sentence with "
               "the format as follows:\n\"The correct answer is (insert answer here)\"")
CONTEXT_POINTER = ("[The full document is NOT reproduced here. It is available in the REPL as the `context` string "
                   "variable; query it with `llm_query` over slices.]")

_PKG = ["tessera", "loomkit", "parcelo", "vireo", "quillon", "harbinger", "mosaicflow", "ferrix", "lumen_ops", "corvid",
        "nimbusdb", "sprocket", "talonai", "brightlane", "kestrelml", "orchidpipe"]
_SUB = ["core", "utils", "io", "models", "services", "cli", "adapters", "engines", "storage", "api", "plugins", "tasks"]
_MOD = ["engine", "registry", "loader", "writer", "parser", "helpers", "client", "server", "session", "schema", "cache",
        "pipeline", "config", "events", "router", "worker", "index", "tokenizer", "metrics", "auth", "scheduler",
        "transform", "formatter", "validators", "builder", "resolver", "monitor", "exporter", "importer", "planner"]
_VERB = ["build", "load", "save", "parse", "render", "fetch", "compute", "prepare", "resolve", "dispatch", "collect",
         "refresh", "normalize", "validate", "encode", "decode", "merge", "apply", "register", "emit", "flush", "sync",
         "init", "scan", "index", "rank", "score", "split", "pack", "unpack", "route", "plan", "check", "update"]
_NOUN = ["index", "cache", "payload", "record", "batch", "token", "session", "config", "schema", "message", "event",
         "frame", "chunk", "manifest", "entry", "result", "report", "state", "graph", "query", "plan", "snapshot",
         "summary", "metadata", "buffer", "request", "response", "handler", "options", "context"]
_COMMON = ["load", "save", "run", "process", "validate", "build", "update", "render", "parse", "close", "reset", "prepare"]
_CLS = ["Manager", "Builder", "Client", "Handler", "Engine", "Store", "Service", "Controller", "Adapter", "Registry",
        "Pipeline", "Worker", "Planner", "Scheduler", "Formatter", "Resolver", "Monitor", "Exporter", "Reader", "Writer"]
_CLSPRE = ["Data", "Prompt", "Vector", "Batch", "Remote", "Local", "Async", "Stream", "Model", "Task", "Index", "Query",
           "Event", "Cache", "Config", "Session", "Graph", "Report", "Token", "Plugin"]


@dataclass
class Fn:
    qid: str
    name: str
    mod: int
    cls: str | None = None
    calls: list = field(default_factory=list)
    traps: list = field(default_factory=list)


@dataclass
class Mod:
    path: str
    dotted: str
    classes: dict = field(default_factory=dict)
    fns: list = field(default_factory=list)


def _bfs(adj: list[list[int]], s: int) -> dict[int, int]:
    d = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in d:
                d[v] = d[u] + 1
                q.append(v)
    return d


def _build_repo(rng: random.Random, n_mod: int):
    pkg = rng.choice(_PKG)
    root = rng.choice(["src/" + pkg, pkg])
    mods: list[Mod] = []
    used = set()
    for _i in range(n_mod):
        for _ in range(50):
            sub, m = rng.choice(_SUB), rng.choice(_MOD)
            if (sub, m) not in used:
                break
        used.add((sub, m))
        path = f"{root}/{sub}/{m}.py"
        dotted = path[:-3].replace("src/", "", 1).replace("/", ".")
        mods.append(Mod(path=path, dotted=dotted))
    fns: list[Fn] = []
    cls_names = set()
    for mi, mod in enumerate(mods):
        n_cls = rng.choice([0, 1, 1, 2])
        for _ in range(n_cls):
            for _ in range(50):
                cn = rng.choice(_CLSPRE) + rng.choice(_CLS)
                if cn not in cls_names:
                    break
            cls_names.add(cn)
            mod.classes[cn] = None
            names = set()
            for _ in range(rng.randint(2, 6)):
                nm = rng.choice(_COMMON) if rng.random() < 0.25 else (("_" if rng.random() < 0.3 else "") +
                                                                     f"{rng.choice(_VERB)}_{rng.choice(_NOUN)}")
                if nm in names:
                    continue
                names.add(nm)
                fns.append(Fn(qid=f"{mod.path}::{cn}.{nm}", name=nm, mod=mi, cls=cn))
                mod.fns.append(len(fns) - 1)
        for _ in range(rng.randint(2, 6)):
            nm = rng.choice(_COMMON) if rng.random() < 0.2 else f"{rng.choice(_VERB)}_{rng.choice(_NOUN)}"
            if any(fns[j].name == nm and fns[j].cls is None for j in mod.fns):
                continue
            fns.append(Fn(qid=f"{mod.path}::{nm}", name=nm, mod=mi))
            mod.fns.append(len(fns) - 1)
    all_cls = [(mi, cn) for mi, m in enumerate(mods) for cn in m.classes]
    for mi, cn in all_cls:
        if rng.random() < 0.3:
            cands = [(mj, c2) for mj, c2 in all_cls if mj != mi and mods[mj].classes.get(c2) is None]
            if cands:
                mods[mi].classes[cn] = rng.choice(cands)
    return pkg, root, mods, fns


def _method_of(fns: list[Fn], mods: list[Mod], mi: int, cn: str, name: str) -> int | None:
    """Resolve self.name() in class (mi, cn), following one level of base classes."""
    seen = 0
    while cn is not None and seen < 4:
        for j in mods[mi].fns:
            if fns[j].cls == cn and fns[j].name == name:
                return j
        base = mods[mi].classes.get(cn)
        if not base:
            return None
        mi, cn = base
        seen += 1
    return None


def _wire_calls(rng: random.Random, mods: list[Mod], fns: list[Fn]) -> list[list[int]]:
    n = len(fns)
    order = list(range(n))
    rng.shuffle(order)
    level = {f: i for i, f in enumerate(order)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for u in range(n):
        fu = fns[u]
        k = rng.choices([0, 1, 2, 3, 4], weights=[2, 4, 4, 2, 1])[0]
        cands = [v for v in range(n) if level[v] < level[u] and v != u]
        if not cands:
            continue
        for _ in range(k):
            v = rng.choice(cands)
            fv = fns[v]
            if v in adj[u]:
                continue
            if fu.cls and fv.mod == fu.mod and fv.cls == fu.cls:
                style = "self"
            elif fu.cls and fv.cls and _method_of(fns, mods, fu.mod, fu.cls, fv.name) == v:
                style = "self"
            elif fv.cls is not None:
                if fv.cls != fu.cls or fv.mod != fu.mod:
                    style = "clsattr"
                else:
                    style = "self"
            elif fv.mod == fu.mod:
                style = "local"
            else:
                style = rng.choice(["from", "from", "alias", "modattr"])
            if style == "self" and not fu.cls:
                continue
            if style == "self" and _method_of(fns, mods, fu.mod, fu.cls, fv.name) != v:
                continue
            if style == "local" and fv.cls is None and any(fns[j].name == fv.name and fns[j].cls is None and j != v
                                                           for j in mods[fu.mod].fns):
                continue
            fu.calls.append((v, style))
            adj[u].append(v)
    return adj


def _naive_adj(fns: list[Fn]) -> list[list[int]]:
    """What name-based grep sees: u -> every function whose NAME u's body mentions as a call, or as a trap name."""
    by_name: dict[str, list[int]] = {}
    for i, f in enumerate(fns):
        by_name.setdefault(f.name, []).append(i)
    out = []
    for f in fns:
        names = {fns[v].name for v, _ in f.calls} | set(f.traps)
        out.append(sorted({j for nm in names for j in by_name.get(nm, [])}))
    return out


def _render(rng: random.Random, mods: list[Mod], fns: list[Fn], file_chars: tuple[int, int]) -> tuple[str, list, dict]:
    parts = []
    items: list[tuple[str, str]] = []
    fn_src: dict[str, str] = {}
    for mi, mod in enumerate(mods):
        imports: dict[str, str] = {}
        alias_for: dict[tuple[int, str], str] = {}
        lines_by_fn: dict[int, list[str]] = {}
        for j in mod.fns:
            fu = fns[j]
            ind = "        " if fu.cls else "    "
            body = []
            if rng.random() < 0.5:
                trap_pool = [f2.name for f2 in fns if f2.name != fu.name]
                tname = rng.choice(trap_pool)
                fu.traps.append(tname)
                body.append(f'{ind}"""{rng.choice(["Delegates to", "Mirrors", "Kept in sync with", "Replaces the old"])} '
                            f'{tname}() for the {rng.choice(["cache", "index", "session", "batch"])} path."""')
            var = rng.choice(["result", "data", "out", "value", "payload", "items", "rec", "cfg"])
            init = rng.choice(["None", "{}", "[]", "0", 'kwargs.get("' + var + '")'])
            body.append(f"{ind}{var} = {init}")
            calls = list(fu.calls)
            rng.shuffle(calls)
            for v, style in calls:
                fv = fns[v]
                mv = mods[fv.mod]
                for _ in range(rng.randint(0, 2)):
                    body.append(ind + code_stmt(rng, var).replace("\n    ", "\n" + ind))
                if style == "self":
                    call = f"self.{fv.name}"
                elif style == "local":
                    call = fv.name
                elif style == "clsattr":
                    if fv.mod != mi:
                        imports[f"from {mv.dotted} import {fv.cls}"] = ""
                    call = f"{fv.cls}.{fv.name}"
                elif style == "from":
                    imports[f"from {mv.dotted} import {fv.name}"] = ""
                    call = fv.name
                elif style == "alias":
                    al = alias_for.setdefault((fv.mod, fv.name), f"_{fv.name}_{mv.dotted.split('.')[-1]}")
                    imports[f"from {mv.dotted} import {fv.name} as {al}"] = ""
                    call = al
                else:
                    al = mv.dotted.split(".")[-1] + rng.choice(["", "_mod", "_lib"])
                    alias_for.setdefault((fv.mod, "__mod__"), al)
                    al = alias_for[(fv.mod, "__mod__")]
                    if rng.random() < 0.5:
                        imports[f"import {mv.dotted} as {al}"] = ""
                    else:
                        pkg, _, last = mv.dotted.rpartition(".")
                        imports[f"from {pkg} import {last} as {al}" if al != last else f"from {pkg} import {last}"] = ""
                    call = f"{al}.{fv.name}"
                body.append(f"{ind}{var} = {call}({var})")
            if rng.random() < 0.35:
                tname = rng.choice([f2.name for f2 in fns])
                if tname not in {fns[v].name for v, _ in fu.calls}:
                    fu.traps.append(tname)
                    body.append(f"{ind}# TODO: call {tname}() once the {rng.choice(['schema', 'cache', 'index'])} is migrated")
            if rng.random() < 0.4:
                body.append(f"{ind}self.client.{rng.choice(['send', 'get', 'emit', 'publish', 'commit'])}({var})" if fu.cls
                            else f"{ind}logger.info(\"{rng.choice(_VERB)} done: %s\", {var})")
            body.append(f"{ind}return {var}")
            sig = f"self, {var}=None, **kwargs" if fu.cls else f"{var}=None, **kwargs"
            head = f"    def {fu.name}({sig}):" if fu.cls else f"def {fu.name}({sig}):"
            lines_by_fn[j] = [head] + body
        out = [f'"""{mod.dotted}: {rng.choice(DOC_PARAS)[:160]}"""', "import os", "import json", "import logging"]
        out += sorted(imports)
        out += ["", "logger = logging.getLogger(__name__)", ""]
        for _cn, base in mod.classes.items():
            if base:
                bm, bc = base
                if bm != mi:
                    imp = f"from {mods[bm].dotted} import {bc}"
                    if imp not in out:
                        out.insert(4, imp)
        emitted_cls = set()
        for j in mod.fns:
            fu = fns[j]
            if fu.cls and fu.cls not in emitted_cls:
                emitted_cls.add(fu.cls)
                base = mod.classes.get(fu.cls)
                out += ["", "", f"class {fu.cls}({base[1] if base else 'object'}):",
                        f'    """{rng.choice(DOC_PARAS)[:120]}"""', ""]
            elif not fu.cls:
                out += ["", ""]
            src = lines_by_fn[j]
            if rng.random() < 0.3:
                out.append(("    " if fu.cls else "") + code_comment(rng))
            out += src
            fn_src[fu.qid] = "\n".join(src)
            items.append((fu.qid, "\n".join(src)))
        target = rng.randint(*file_chars) * 1000
        k = 0
        text = "\n".join(out)
        pad = []
        while len(text) + sum(len(p) for p in pad) < target and k < 400:
            nm = f"_{rng.choice(_VERB)}_{rng.choice(_NOUN)}_{k}"
            v = rng.choice(["x", "y", "acc"])
            pad.append("\n\ndef " + nm + f"({v}=None):\n    " + code_comment(rng) + "\n    "
                       + "\n    ".join(code_stmt(rng, v) for _ in range(rng.randint(3, 9))) + f"\n    return {v}")
            k += 1
        parts.append(f"<File>: {mod.path}\n" + text + "".join(pad) + "\n")
    extra = [("README.md", "\n\n".join(DOC_PARAS[rng.randrange(len(DOC_PARAS))] for _ in range(rng.randint(10, 40)))),
             ("setup.py", "from setuptools import setup, find_packages\n\nsetup(name=\"pkg\", "
                          f"version=\"0.{rng.randint(1, 9)}.{rng.randint(0, 30)}\", packages=find_packages())\n"),
             ("docs/usage.md", "\n\n".join(DOC_PARAS[rng.randrange(len(DOC_PARAS))] for _ in range(rng.randint(5, 25))))]
    for path, body in extra:
        parts.insert(rng.randrange(len(parts) + 1), f"<File>: {path}\n{body}\n")
    return "\n\n\n".join(parts), items, fn_src


def _question(rng: random.Random, fns: list[Fn], adj: list[list[int]], kind: str):
    n = len(fns)
    dist = [_bfs(adj, s) for s in range(n)]
    nav = _naive_adj(fns)
    ndist = [_bfs(nav, s) for s in range(n)]
    K = rng.choice([2, 3, 4])
    if kind == "cannot":
        targets = list(range(n))
        rng.shuffle(targets)
        for t in targets:
            anc = {s: dist[s][t] for s in range(n) if s != t and t in dist[s]}
            can = [s for s, d in anc.items() if d + 1 <= K]
            exact = [s for s in can if anc[s] + 1 == K]
            deep = [s for s, d in anc.items() if d + 1 == K + 1]
            if len(can) < 3 or not deep or (K > 2 and not exact):
                continue
            tricky = [s for s in deep if t in ndist[s] and ndist[s][t] + 1 <= K]
            ans = rng.choice(tricky) if tricky and rng.random() < 0.85 else rng.choice(deep)
            others = ([rng.choice(exact)] if exact else [])
            others += rng.sample([s for s in can if s not in others], 3 - len(others))
            opts = others + [ans]
            rng.shuffle(opts)
            letters = "ABCD"
            options = [f"{letters[i]}. {fns[o].qid}" for i, o in enumerate(opts)]
            q = (f"In the given codebase, which of the following functions **cannot** directly or indirectly invoke "
                 f"`{fns[t].qid}` within a call stack depth of {K}? (e.g. A->B->C has a call stack depth of 3)")
            gold = letters[opts.index(ans)]
            info = {"K": K, "target": fns[t].qid, "depths": {fns[o].qid: dist[o][t] + 1 for o in opts},
                    "answer_tricky": ans in tricky}
            return q, options, gold, info, [fns[o].qid for o in opts] + [fns[t].qid]
        return None
    pairs = [(s, t) for s in range(n) for t, d in dist[s].items() if d >= 1]
    shallow = [(s, t) for s, t in pairs if dist[s][t] + 1 <= K]
    exact = [(s, t) for s, t in shallow if dist[s][t] + 1 == K]
    deep = [(s, t) for s, t in pairs if dist[s][t] + 1 == K + 1]
    if len(shallow) < 3 or not deep or (K > 2 and not exact):
        return None
    tricky = [(s, t) for s, t in deep if t in ndist[s] and ndist[s][t] + 1 <= K]
    ans = rng.choice(tricky) if tricky and rng.random() < 0.85 else rng.choice(deep)
    others = [rng.choice(exact)] if exact else []
    pool = [p for p in shallow if p not in others and p[0] != ans[0]]
    if len(pool) < 3 - len(others):
        return None
    others += rng.sample(pool, 3 - len(others))
    opts = others + [ans]
    rng.shuffle(opts)
    letters = "ABCD"
    options = [f"{letters[i]}. ({fns[s].qid} , {fns[t].qid})" for i, (s, t) in enumerate(opts)]
    q = (f"Which of the following function pairs has a call stack depth of **more than {K}**? (e.g. if the call trace for "
         f"the function pair (A , C) is A->B->C, the call stack depth is 3.) \n")
    gold = letters[opts.index(ans)]
    info = {"K": K, "depths": {f"{fns[s].qid} , {fns[t].qid}": dist[s][t] + 1 for s, t in opts},
            "answer_tricky": ans in tricky}
    return q, options, gold, info, [fns[x].qid for p in opts for x in p]


def generate_code(info: dict, n_mod: tuple[int, int] = (16, 40), file_kchars: tuple[int, int] = (12, 45),
                  kinds: dict | None = None) -> Generated:
    rng = random.Random(f"code:{info.get('synth_split')}:{info.get('synth_seed')}")
    kinds = kinds or {"cannot": 30, "pair": 30}
    kind = info.get("_kind") or rng.choices(list(kinds), weights=list(kinds.values()))[0]
    for _ in range(40):
        pkg, root, mods, fns = _build_repo(rng, rng.randint(*n_mod))
        adj = _wire_calls(rng, mods, fns)
        context, items, fn_src = _render(rng, mods, fns, file_kchars)
        q = _question(rng, fns, adj, kind)
        if q is not None:
            break
    else:  # pragma: no cover
        raise RuntimeError("code generator failed")
    question, options, gold, qinfo, required = q
    root_prompt = (INSTRUCTION.replace("{context}", CONTEXT_POINTER).replace("{question}", question)
                   .replace("{options}", "\n".join(options)))
    callees = {fns[u].qid: [fns[v].qid for v, _ in fns[u].calls] for u in range(len(fns))}
    hidden = {"family": "code", "kind": f"cg-{kind}", "answer_type": "MC", "options": options, "question": question,
              "callees": callees, "q": qinfo}
    return Generated(context=context, root_prompt=root_prompt, items=items, gold=gold, hidden=hidden,
                     required_ids=required, index_min_len=14)


_AGG = re.compile(r"\b(cannot|which of the following|call stack depth|depth of|more than \d|reach|reachable|shortest|"
                  r"path from|indirectly|answer)\b", re.I)


class CodeOracle:
    """Reads function definitions it is given; lists direct repo callees; does not answer the question."""

    def respond(self, base: FakeSubLM, prompt: str, hidden: dict, st: dict) -> str:
        ids = base.given(prompt, hidden)
        instr = base.instruction(prompt, hidden)
        if not ids:
            return refuse_aggregate(st)
        read, _ = base.read_items(prompt, ids, st)
        if _AGG.search(instr) and len(read) > base.read.agg_capacity:
            st["agg_requests"] += 1
            r = rng_for(prompt, "agg")
            letter = r.choice("ABCD")
            return base.chat(prompt, f"Looking at the call relationships in the code provided, the correct answer is ({letter}).")
        calls = hidden["callees"]
        names = {q.split("::")[1].split(".")[-1]: q for q in calls}
        pairs = []
        for qid in read:
            outs = list(calls.get(qid, []))
            if base.flip(prompt, qid, "drop") and outs:
                outs.pop(rng_for(prompt, qid, "d").randrange(len(outs)))
            if base.flip(prompt, qid, "add"):
                nm = rng_for(prompt, qid, "a").choice(sorted(names))
                outs.append(names[nm])
            pairs.append((qid, outs))
        if wants_json(instr):
            return base.render(prompt, instr, pairs)
        lines = [f"{q} calls: {', '.join(o) if o else '(no project functions)'}" for q, o in pairs]
        return base.chat(prompt, "\n".join(lines))


__all__ = ["CodeOracle", "generate_code"]
_ = json
