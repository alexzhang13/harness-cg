"""Progress header telling the model where it is in the chain, e.g. progress_header(stage=2, n=4) ->
"[SEQUENCE] You need to solve 4 tasks in order: A, then B, then C, then D. So far you have solved: A and B. ..."
"""
LABELS = "ABCDEFGH"

def _join(xs):
    return "none" if not xs else xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]

def progress_header(stage, n):
    """stage: 0-based index of the current task; n: number of tasks."""
    order = ", then ".join(LABELS[:n])
    solved = list(LABELS[:stage])
    what = f"{n} tasks in order: {order}" if n > 1 else "1 task: A"
    return (f"[SEQUENCE] You need to solve {what}. So far you have solved: {_join(solved)}. "
            f"The task above is task {LABELS[stage]}.")

def with_header(question, stage, n, pos="post"):
    h = progress_header(stage, n)
    return f"{h}\n\n{question}" if pos == "pre" else f"{question}\n\n{h}"

if __name__ == "__main__":
    for s in range(4): print(progress_header(s, 4))
    print(progress_header(0, 1))
