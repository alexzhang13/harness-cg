4m. legible_steps: true if each block does one job (probe / parse / compute / check / submit), is short
   and readable, and the prose before each block says what the last output showed and what this block is for.
   false for one long block that parses, filters, delegates and aggregates at once, or illegible code.
4d. recovers: true if the program did not error repeatedly; a single error that was diagnosed and fixed on the
   next turn is fine, repeated tracebacks or blind retries are not.
4. wasted_turns: number of turns spent on errors, retries, or dead ends.
