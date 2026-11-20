4j. verified_before_submit: true if the answer was computed in one turn, checked in a LATER block after the
   root had looked at the result, and submitted either by that check block itself under its pass condition
   (`if ok: answer[...] = ...`) or in a short block after it. false if the answer is computed and submitted in
   the same block with no check, or submitted without looking at / checking the result, or submitted despite a
   failed or ignored check (an unconditional submit after a check that printed a failure counts as ignored).
4k. check_independent: true if the check could have disagreed with the first result: a different method,
   an invariant, grounding quotes in the text, a narrower re-ask of exactly the items the answer depends on,
   or one-item-per-call re-decision of contested items -- and the root confirmed the check's own replies
   parsed. false if the "check" re-runs the same code/prompt, only restates the answer, or was skipped.
4l. reads_subcalls_critically: false if the root relied on a sub-call reply that was visibly wrong,
   overreaching (e.g. a description window that "answers" the task) or unparsed, without noticing. true if it
   noticed and corrected such replies, or if no such reply occurred.
