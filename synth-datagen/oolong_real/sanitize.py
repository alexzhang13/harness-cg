"""Real-name guard for synthetic traces: ``sanitize`` swaps real names in note examples for invented ones;
``real_names_in`` lists real cast names found in a text (export_sft.py rejects those rows).

    python sanitize.py --build-names      (after prepare.py)
    python sanitize.py <note in> <note out>"""
import os
import re
from pathlib import Path

REPLACE = [
    ("# (Vex'ahlia also counts as \"Vex\")", "# (a two-part name like Kael'thanis also counts as \"Kael\")"),
    ("`Natural 20 for Fjord and a 19 for Yasha`", "`Natural 20 for Brisa and a 19 for Tovin`"),
    ("`Sam: 15.`", "`Ren: 15.`"),
    ("by the character Jester in episode 2", "by the character Orrin in episode 2"),
    ('same(x["char"], "Jester")', 'same(x["char"], "Orrin")'),
]
DATA = Path(os.environ.get("OOLONG_LMPROG_DATA") or Path(__file__).resolve().parents[2] / "outputs/synth-datagen/oolong_real/data")
NAMES = DATA / "real_cast_names.txt"


def sanitize(text: str) -> str:
    for a, b in REPLACE:
        text = text.replace(a, b)
    return text


def real_names_in(text: str) -> list[str]:
    names = [n for n in NAMES.read_text().split("\n") if n and n not in ("Mark",)]
    return [n for n in names if re.search(r"(?<![\w'])" + re.escape(n) + r"(?![\w'])", text)]


def build_names() -> list[str]:
    """Every player and character name in the real windows' cast headers (`X plays the character Y.`), both splits."""
    real = set()
    for f in (NAMES.parent / "contexts").glob("*.txt"):
        for p, c in re.findall(r"^(.+?) plays the character (.+?)\.$", f.read_text()[:4000], re.M):
            real |= {p.strip(), c.strip()}
    return sorted(real - {"DM"})


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--build-names"]:
        NAMES.write_text("\n".join(build_names()))
        print(NAMES, len(build_names()), "names")
        raise SystemExit
    src, dst = map(Path, sys.argv[1:3])
    out = sanitize(src.read_text())
    left = real_names_in(out)
    assert not left, left
    dst.write_text(out)
    print(dst, "written; real names left:", left)
