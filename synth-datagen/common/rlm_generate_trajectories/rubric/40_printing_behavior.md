4n. condensed_output (PRINTING BEHAVIOUR): true if what the root PRINTS is condensed: statistics and counts, a
   few lines that confirm a specific belief, or sub-call summaries that the root explicitly asked to be concise.
   false only for a REDUNDANT, BULKY dump -- output whose size is out of proportion to what the root needed to learn --
   even under any character limit. Each issue below has to be bulky (roughly 20+ lines or dozens of items, or a large
   block repeated) to count:
   P1 per-item print loop: one line per item over many items, where a count + a few examples would do;
   P2 echoed sub-call reply: a whole long multi-line reply printed back (print(reply)) instead of parsed counts or the
      lines that matter;
   P3 long inline list: a list or dict of dozens of ids, titles, turn numbers or lengths;
   P4 raw context read by the root: whole paragraphs, sections, records, or many "filtered" / [:300] slices of them;
   P5 re-print: a large block that was already shown (the question, the parsed items, the same list, a long reply);
   P6 prompt echo: its own long sub-call prompts or previews of them (len() is enough);
   P7 diagnostic dump: dozens of mismatches / exceptions / missing ids instead of a count + the first few;
   P8 long description: multi-paragraph descriptions or reports printed in full when a short summary was enough.
   FINE -- must not fail this criterion: a compact parse check (a count, a range, the handful of non-data lines such
   as the instruction or question line), a small histogram or distribution, up to ~5 short example or preview lines,
   a few candidate lines, and printing a short final answer once more when submitting.
4b. clean_root: did the root keep task-specific PROSE out of its own context? One short orientation
   excerpt of the payload on an early turn (a couple of thousand characters, to see the format) is the
   expected probing step and is fine. Set false only for: paging through the payload turn after turn, a
   whole document or record list printed, dozens of sub-call replies echoed verbatim, or long task-specific
   prose pasted into comments. Computed output -- counts, ids, labels, scores, parsed `FIELD=value` lines,
   the final answer -- is never a cleanliness problem at any length. Total output size is already measured
   objectively before you see the rollout; this flag is one input to `quality`, not a veto.
