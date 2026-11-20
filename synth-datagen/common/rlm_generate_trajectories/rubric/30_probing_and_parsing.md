4h. blind_probe: the root had not seen `context` when it started. true if its FIRST turn only measured the
   context and asked a sub-call to describe a bounded piece of it (what it is, how it is laid out, exact
   markers) and printed that description -- looking at more only if needed. false if the first turn already
   parses, filters, scores or answers; hard-codes a marker/split pattern/field name it had not seen yet; or
   "probes" by printing a raw slice of the context instead of delegating a description. If the whole context
   is only a few hundred characters, reading it directly is fine. If `context` is a dict or list, printing only
   its keys, lengths and value types in turn 1 and delegating the description of a sample element in turn 2
   is the expected probe; printing raw sample elements instead of delegating is not.
   If STREAM FLAGS say python_exact (the answer is exactly computable in Python: literal search, counting, parsing, graph traversal, verbatim copying), the preferred probe is still a description sub-call (measure the context, let llm_query describe a window, confirm a quoted marker with a print of a few hundred characters at most) -> true. A first turn that only measures and prints a bounded slice or its structure (lengths, a few hundred characters, the separator lines) to find the exact markers is also acceptable -> true; printing large windows to read the context yourself is not.
4i. parse_from_evidence: true if the first parse used markers the description had reported, and printed a
   consistency check (counts vs anything stated, contiguity, leftovers, first/last item) -- and the root acted
   on anything that disagreed. false if it parsed blind or never checked the parse.
   If STREAM FLAGS say python_exact, markers the root saw in its own measured print count as evidence (no description
   sub-call needed); the consistency check of the parse is still required.
