"""Render the staged, check-driven note (v4) from note_turns_v4."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import note_turns_v4 as N

HEAD = """This task hides a few two-hop questions among thousands of items: most questions have NO answer in the corpus,
and a planted question is answerable only from TWO passages together (a first hop that identifies the subject, a
second that gives the answer). The items can be laid out in many ways (bracketed ids, markdown headings, XML tags,
JSON lines, YAML records, table rows, ...), with any id scheme and any labels, so the layout must be read from the
data, never assumed. The work is a sequence of STAGES. Every stage prints a `CHECK` line that ends in a verdict,
`OK` or `PROBLEM: <reason>`. On OK, go on to the next stage. On PROBLEM, do not push on: investigate with one targeted
sub-call, fix the cause, re-run the stage and read its CHECK again; a wrong parse or a batch of unreadable replies
silently ruins everything after it. Most questions are not answerable from the corpus, so a large "not answerable"
count is normal and is not a problem.

## Rules
- Learn the layout ONLY through the probe sub-call; print only its report. Never print `context` or slices of it.
- Run one stage (or one repair) per turn. Before each block write ONE short sentence in your own words: what you are
  doing and why, citing the CHECK line that made you decide it when you repair. Never mention instructions, notes or
  hints.
- Every model step is a sub-call; never answer a question from your own reading.
- Repair a stage only when its CHECK verdict says PROBLEM (or, for the parse, when the counts clearly disagree with
  the report), and say which CHECK line made you repair. Do not repeat an identical repair that already failed; try a
  different fix, or move on with what you have.
- If code raises an error, fix only that error and continue.
"""

STAGES = [
    ("Stage 1 -- probe the layout with one sub-call over samples", N.T1, None, None),
    ("Stage 2 -- write the parse from the report and check it", N.TURNS[1],
     "If the CHECK parse verdict is PROBLEM, or its counts clearly disagree with the report: "
     "show a sub-call where it breaks, fix `HDR`/`KIND` from its answer, and re-run Stage 2. Write `HDR` (named group "
     "`id`) and `KIND` for the layout the report describes; the block shows one layout's version.", N.FIXES["parse"]),
    ("Stage 3 -- lean shortlist, screen every question", N.T3,
     "If the CHECK screen verdict is PROBLEM: look at one unparseable reply, then re-run those batches in halves.",
     N.FIXES["screen"]),
    ("Stage 4 -- re-screen uncertain questions; verify two-passage chains", N.T4,
     "If the CHECK verify verdict is PROBLEM: investigate one unverified question with a sub-call, then re-verify "
     "with wider candidates and exact-copy quotes.", N.FIXES["verify"]),
    ("Stage 5 -- two independent checks, join, submit", N.T5,
     "The block submits only when every check reply was readable. Otherwise re-ask just the unreadable ones and "
     "submit.", N.FIXES["checks"]),
]

if __name__ == "__main__":
    parts = [HEAD.strip()]
    for title, code, decide, fix in STAGES:
        parts.append(f"## {title}\n```repl\n{code.strip()}\n```")
        if decide:
            parts.append(f"Decision point: {decide}\n```repl\n{fix.strip()}\n```")
    if len(sys.argv) != 2 or sys.argv[1].startswith('-'):
        sys.exit(print(__doc__))
    Path(sys.argv[1]).write_text("\n\n".join(parts) + "\n")
    print("wrote", sys.argv[1], len(Path(sys.argv[1]).read_text()), "chars")
