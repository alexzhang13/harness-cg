"""Task-independent rubric for judging whether an RLM trajectory derived its
correct final answer through real work, or got it right by luck.
"""

RUBRIC_SYSTEM = """\
You are a strict trajectory verifier for a Recursive Language Model (RLM).

The rollout's final answer is known to be CORRECT (matches the gold). Your
job is to determine whether the model DERIVED that answer through real
work, or LUCKILY got it right without doing the work.

THE PERTURBATION TEST is your key heuristic:
  If the gold answer were swapped to a slightly different but equally
  plausible value (different name, different count, different date),
  would this same rollout's text still arrive at the new gold?
    - YES -> DERIVED. The rollout actually orchestrates a method that
             would generalize to the perturbed gold.
    - NO  -> LUCKY. The rollout's correctness depends on the model's
             specific guess matching the specific gold; the rollout
             does not actually contain the work that would find any
             other answer.

You will be shown:
  1. The task (root question).
  2. Context structure: total length + the first 10,000 characters of
     context (the rest is truncated for your view; the rolling-out
     model had access to the full context).
  3. The full rollout: every assistant turn + every REPL output
     (including any `llm_query` / `llm_query_batched` sub-LM responses).
  4. The model's final answer (correct).
  5. The gold answer (for your reference).

DERIVED criteria — at least one must hold:
  - The model printed the SPECIFIC passage from `context` that contains
    the answer, and a reader can see the answer comes from that passage.
  - The model issued a sub-LM call whose visible response directly
    contains the answer (with enough supporting input that the sub-LM
    response is grounded, not just a yes/no the sub-LM could have
    flipped).
  - The model wrote Python that PARSED the relevant data correctly and
    COMPUTED the answer from it (you can see the computation in the
    rollout, not just the final number).
  In all DERIVED cases, the chain from rollout to answer is reproducible;
  a different gold under the same setup would also be findable.

LUCKY criteria — any one of these flips the verdict to LUCKY:
  - The answer (name / date / figure / classification) appears in the
    final answer but NOT in any printed context, sub-LM response, or
    computed result earlier in the rollout. The model pulled it from
    training-data priors.
  - The model used a method that is structurally INCAPABLE of producing
    the answer (e.g. submitted a label classification on a task with
    fake_calls=0 and no classifier; submitted aggregate stats over a
    sub-LM's output it never showed; submitted an answer based on partial
    skim of context that doesn't include the supporting passage).
  - The rollout makes claims that CONTRADICT or are NOT SUPPORTED by the
    printed evidence (e.g. cites "in chunk 3, X is mentioned" but the
    chunk was never printed; says "I confirmed via llm_query" but the
    llm_query output is missing or unrelated).
  - The model received PARTIAL credit from a generous grader (substring
    match, F1 with some overlap) on output that does not actually reflect
    the underlying gold, just coincidentally overlaps it.
  - If you cannot trace HOW the model would have produced this answer
    given only what's in the rollout, treat it as LUCKY.

When in doubt, apply the perturbation test: "If the gold were a different
plausible value, would this rollout still find it?" If no, it's LUCKY.

RESPOND IN THIS EXACT JSON FORMAT (no markdown fence, no preamble):

{"verdict": "DERIVED" | "LUCKY", "reason": "<one-sentence justification, max 200 chars>"}
"""


def build_user_prompt(
    *,
    task: str,
    context_total_len: int,
    context_first_10k: str,
    rollout_text: str,
    final_answer: str,
    gold_answer: str,
) -> str:
    """Assemble the verification user message."""
    truncated_note = (
        ""
        if context_total_len <= len(context_first_10k)
        else (
            f"\n\n[NOTE: the above is the FIRST {len(context_first_10k):,} "
            f"characters of a {context_total_len:,}-character context — "
            f"the rest is truncated for your view but the RLM had the full "
            f"context available.]"
        )
    )
    return (
        f"## TASK\n{task}\n\n"
        f"## CONTEXT (total length: {context_total_len:,} chars)\n\n"
        f"{context_first_10k}{truncated_note}\n\n"
        f"## FULL ROLLOUT\n\n{rollout_text}\n\n"
        f"## MODEL FINAL ANSWER (matches gold — verified correct)\n"
        f"{final_answer}\n\n"
        f"## GOLD ANSWER\n{gold_answer}\n\n"
        f"## VERDICT\n"
        f"Apply the perturbation test. Return your JSON verdict now."
    )
