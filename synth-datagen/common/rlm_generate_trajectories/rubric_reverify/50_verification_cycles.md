4. VERIFICATION CYCLES (this replaces the v2 verification rules for this dataset)

A CHECK is any step whose purpose is to test a result, a parse, an intermediate value or a candidate answer, by
asking "is this right?" rather than computing something new. A CYCLE is:
- an attempt (a plan executed to produce a result);
- then a check of that result;
- if the check FAILS (disagrees, finds a contradiction, the result fits no option, two sources conflict, a count or
  invariant is off, the evidence for a candidate is missing): a replan and a new attempt, which starts the next
  cycle.

Annotate every check in the trajectory, in order, as one entry of `cycles`:
  {"check_turn": <assistant turn of the check>,
   "independent": true|false,       could this check have disagreed? (different method, invariant, quotes from the
                                    text, a neutral re-ask of exactly the contested items). false = re-runs the
                                    same code or prompt, restates the answer, re-extracts the same line, or uses a
                                    LEADING sub-call prompt that states the expected conclusion.
   "result": "pass" | "fail",       what the check's output actually showed (not what the model claimed).
   "read_correctly": true|false,    did the model read the check's outcome as it is? false = calls a failed check
                                    "confirmed", ignores a disagreement, or treats a pass as a failure.
   "error_turns": [<turns>],        the turns that CAUSED a failed check: the wrong parse/plan/computation/
                                    reading. [] for a pass.
   "error_kind": "parse" | "missing_constraint" | "misread_question" | "wrong_method" | "bad_subcall_reply" |
                 "computation" | "wrong_candidate" | "other" | null,
   "replan_turn": <first turn of the new plan> | null,
   "replan_grounded": true|false|null,  does the new plan follow from what is ESTABLISHED in the outputs (facts,
                                    constraints, eliminated candidates), rather than starting over blindly?
   "replan_new": true|false|null}       is it a different hypothesis or method from EVERY earlier failed plan (not a
                                    re-run, a renamed variable, or a wider version of the same search)?

`n_failed_checks` is the number of cycles whose check FAILED and was read correctly. A trajectory that never had to
recover has `n_failed_checks = 0`.

Whole-trajectory fields:
4a. checks_independent: every check is independent. The LAST check before the submit MUST be.
    If STREAM FLAGS say python_exact, a programmatic check IS independent when it tests the result by a different
    method than the one that produced it (a recount with another construction, an invariant or `assert` that could
    fail, re-deriving the answer from the raw context): a sub-call check is not required. A purely programmatic
    solution with one such passing check before the submit is complete -- do not demand extra cycles.
4b. failure_acknowledged: after every failed check, the next turn says plainly that the check failed and what
    disagreed, citing the printed output (not "let me double-check" with no diagnosis). true if no check failed.
4c. no_repeat_after_fail: after a failed check, the model NEVER re-runs the same check, sub-call or print, and never
    re-reads the same region hoping for a different result. A narrower, different re-ask is fine.
4d. inventory_before_replan: before each new plan, the model BACKTRACKS: it states (in its visible prose) what is
    already established, which step went wrong, and what the failure rules out. Not a full recap; just enough that
    the new plan visibly builds on the sound part and discards what was derived from the faulty step.
4e. replan_from_information: each new plan is chosen BECAUSE of the failure. It names a concrete explanation of the
    disagreement (a constraint it missed, a different reading of the question, a parse that dropped items, a
    sub-call reply that was wrong, a method that cannot work here) and tests it.
    Examples of good replans:
    - "18 fits no option, so the question must intend a stricter condition; the options match the derangement
      count, 9."
    - "The two candidate statements conflict; the later dated one is the operative one."
    - "The parse found 0 of 412 records, so the delimiter is wrong."
4f. replan_is_new: every new plan differs from all earlier failed plans in hypothesis or method.
4g. reverified_before_submit: the result of the LAST plan was checked independently, and that check passed, before
    the submit.
    Exception: when the turn budget is nearly spent (the last ~3 turns) or after 5 failed checks, the model may
    submit the best-supported answer without a passing check, if it says so explicitly in one line ("committing
    to X: best supported by A and B; check C still disagrees because ...").
4h. false_pass: true if ANY check reported "pass" (or was read as a pass) on a result that was in fact wrong. This
    is the failure mode we are removing; it is never acceptable.
4i. commit_when_verified: once a check that could have disagreed has passed, the model submits in that turn or the
    next. It does not re-verify the same thing, reprint the answer, or start a new search. true if submitted
    within one turn of the passing check.
4k. no_redundant_work: the model never redoes work whose result it already has -- re-computing a value held in a
    variable, re-parsing the same region, re-asking a sub-call that was already answered, re-printing an output it
    already saw -- unless it first changed what that work depends on and says what changed. A replan that reuses
    established variables is the target; one that starts over from scratch when most results still hold is not.
    false also when two or more consecutive turns gained no new information (a loop), even if the code differs.
4j. reads_subcalls_critically: false if the root relied on a sub-call reply that was visibly wrong, overreaching or
    unparsed, without noticing. true if it noticed and corrected such replies, or if none occurred.

Failed checks are GOOD data here when they are handled well. Do not penalise a trajectory for making a mistake
that an independent check then caught. Penalise only how the failure was handled (4b–4g) and false passes (4h).
