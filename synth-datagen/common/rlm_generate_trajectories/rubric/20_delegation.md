2. subcalls_appropriate: true only if every sub-call was given the material it needed (a chunk,
   the retrieved passage, the items to label) and its reply was actually parsed/used downstream.
   A sub-call with no context, a sub-call asked to do arithmetic/counting Python should do, or a
   sub-call whose reply is ignored -> false. A sub-call that was unnecessary because Python already
   had the answer -> false. If the trajectory made NO sub-calls and the task is one Python can do
   exactly (literal search, counting, parsing, graph traversal, verbatim copying) -> true. If it made
   NO sub-calls on a context that has to be READ for meaning (more than a few thousand characters of
   prose, dialogue or documents) and the STREAM FLAGS do not say python_exact -> false: the root should
   have offloaded that reading to sub-calls instead of doing it itself or grepping around it.
4a. parallel: true if independent sub-calls were issued together with `llm_query_batched` rather than one by one
   in sequence (a loop of single `llm_query` calls over items that do not depend on each other -> false;
   a genuinely sequential chain where each call needs the previous reply -> true).
4c. concise_subprompts: true if the sub-call prompts were short and clean -- a focused instruction plus only the
   material needed (a chunk, a record list), with a fixed reply format -- not the whole context or a rambling brief.
4e. size_checked: true if, before sending material to a sub-call, the root knew how big that prompt would be
   (it printed/checked len() of the chunk or chose chunk sizes deliberately) OR every sub-prompt is obviously
   small (a few thousand characters). False if it passed a variable of unknown size (a whole section, a joined
   list, an unbounded retrieval result) straight into a sub-call. No sub-calls -> true.
4f. search_spam: true if the root fired many ad-hoc literal searches / keyword lists / regexes at the context
   hoping to hit the answer (a guessed vocabulary tuned to this task, retried with variations), instead of
   parsing the context's structure or delegating the reading. A few structural patterns (splitting on record
   headers, one field regex) are NOT spam. If STREAM FLAGS say search_ok, narrowing a huge corpus with search
   is expected: only flag blind, repeated guessing.
4g. causal_ok: only meaningful when STREAM FLAGS say context_causal (later parts depend on earlier ones:
   state that changes over time, a chain of keys, an order). true if the decomposition respected that: a
   sequential scan carrying a running state/summary into each next call, chunks given the earlier state they
   need, or a final pass that reconciles per-chunk findings in order (latest state wins). false if independent
   sub-calls each saw one chunk and their answers were merged as if order did not matter. Not causal -> true.
