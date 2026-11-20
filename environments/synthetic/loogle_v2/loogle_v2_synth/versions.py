"""Procedural version comparisons, including add/delete/modify and end-delimiter traps."""
from __future__ import annotations

import json
import random

from synth_common.envkit import Generated


def generate_versions(info: dict, n_files: int = 18, file_chars: int = 60000) -> Generated:
    if n_files < 6 or file_chars < 200:
        raise ValueError("versions needs at least six files and 200 characters/file")
    r = random.Random(f"versions:{info['synth_split']}:{info['synth_seed']}")
    salt = r.randrange(10**8)
    files = {}
    for i in range(n_files):
        path = f"src/module_{r.randrange(10000):04d}/record_{salt}_{i}.py"
        head = f"# Generated module {i}; release token {r.randrange(10**9)}.\n"
        filler = "".join(f"VALUE_{k} = {r.randrange(10**6)}\n" for k in range(100))
        files[path] = (head + filler * (file_chars // len(filler) + 1))[:file_chars] + "\n"
    previous, current = dict(files), dict(files)
    names = list(files)
    r.shuffle(names)
    deleted, modified = names[:2], names[2:4]
    for p in deleted:
        del current[p]
    for p in modified:
        i = len(current[p]) // 2
        current[p] = current[p][:i] + f"\nUPDATED_VALUE = {r.randrange(10**9)}\n" + current[p][i:]
    added = [f"tests/new_{salt}_{i}.py" for i in range(2)]
    for p in added:
        current[p] = "# A new generated test.\n" + "assert True\n" * (file_chars // 12)
    current[names[4]] += " \n\n"
    def render(fs):
        items = list(fs.items())
        r.shuffle(items)
        return "\n\n\n".join(f"<File>: {p}\n{body}" for p, body in items)
    context = "<PREV_VERSION>\n" + render(previous) + "\n</PREV_VERSION>\n<CURR_VERSION>\n" + render(current) + "\n</CURR_VERSION>"
    gold = sorted(deleted + modified + added)
    question = ("Which files have been added, deleted or modified between the two versions? "
                "Ignore trailing whitespace at the end of each file. The complete versions are in the REPL "
                "context string inside <PREV_VERSION> and <CURR_VERSION>; each file starts with <File>: path. "
                "Submit The changed files are: followed by a JSON list of paths through answer['content'].")
    return Generated(context=context, root_prompt=question, items=[], gold=json.dumps(gold),
                     hidden={"family": "versions", "kind": "version-control", "answer_type": "SA", "options": []})
