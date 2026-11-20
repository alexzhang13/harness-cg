5. STRUCTURE, ERRORS AND REASONING

Judge the VISIBLE text of each assistant turn: its prose and code. That is what the student sees and is trained on,
and the student runs without hidden thinking. The teacher also runs without thinking, so most turns carry no hidden
reasoning; when a turn does carry some (shown as "[reasoning]" before the visible text), it may add detail, but it
never substitutes for the visible prose: a decision stated only in [reasoning] counts as NOT stated. The four
reasoning_* fields below keep their names but are judged on the visible prose.

5a. legible_steps: each block does one job (probe / parse / compute / check / replan / submit; a check block that ends
    with `if ok:` submitting is ONE job, the preferred way to submit), and the prose before
    it says what the last output showed and what this block is for. false for one long block that parses,
    filters, delegates and checks at once, or for illegible code.
5b. recovers: true if code errors (tracebacks) were diagnosed from the traceback's failing line and fixed. false if
    the same error appears twice in a row, or a "fix" leaves the failing line unchanged. Errors are allowed; blind
    retries are not.
5c. reasoning_present: every assistant turn has visible prose before its code that says what the last output
    showed and what this block does next (a bare code block with no prose is false). Hidden reasoning is NOT
    required and does not count.
5d. reasoning_consistent: the code matches the visible prose (and the hidden reasoning, if any). false if the prose decides one thing and
    the code does another (e.g. prose "the parse is wrong, use the dated header" but code that re-prints the
    same window), or if the prose claims a fix the code does not contain.
5e. reasoning_clean: neither the visible prose nor any hidden reasoning ever refers to a hint, note, private instruction, earlier attempt, gold
    answer, grader or "expected" answer, and never states the answer before the outputs support it.
5f. reasoning_decides: at each failed check, the VISIBLE prose contains the actual decision: why the check failed,
    what is now known, which hypothesis or method comes next, and why. This is the core of what the student must
    learn.
5g. wasted_turns: turns that neither gained information nor tested a hypothesis (repeats, re-prints, dead ends
    that were never acknowledged).
5h. programmatic_rationale: only meaningful when STREAM FLAGS say python_exact. true if the root said, in its prose or
    reasoning before relying on code for the solve, that a programmatic strategy is sufficient for this task and why (the structure
    is exact: an edge list, numbered turns, literal strings, a key chain, a line diff), so it chose code on purpose. Sub-calls
    that describe the layout or read meaning are communication and do not replace that statement. false if it never said so,
    or if the task needs reading for meaning. Not python_exact -> true. A concise programmatic solution (few turns, compact prints, one programmatic check) is the
    preferred shape when STREAM FLAGS say python_exact; do not mark its steps illegible or its work redundant for being short.
