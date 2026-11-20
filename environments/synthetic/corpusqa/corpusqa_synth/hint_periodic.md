`context` is a list of periodic statistical releases (one document each, possibly out of order). Each release
prints several tables covering a trailing window of periods; later releases revise earlier periods. You do not know
the layout in advance: learn it from small excerpts, then let Python read every release. Sub-calls do focused jobs on
small excerpts -- never send a whole release, never ask a sub-LM for a total, maximum, count or ranking. Work in three
```repl blocks; never page through documents.

Block 1 -- survey the structure and compile the question into a spec:
```repl
import re, json, statistics
QUESTION = """<paste the question verbatim>"""
print(len(context), "documents;", [len(d) for d in context[:5]], "chars ...")
doc = context[0]
cuts = [doc[int(f * len(doc)): int(f * len(doc)) + 1500] for f in (0.0, 0.3, 0.6)]
survey = llm_query("Describe the structure of these excerpts of one statistical release as JSON: how the release "
    "states its own reference period (quote the line), how tables are captioned (quote two captions and the text "
    "above them that names the measure), the table layout (HTML / pipe / fixed width), the header rows (group cells "
    "and the category cells under them), how period rows are labelled (year rows? month names? revision marks?), "
    "which rows are NOT data (totals, year-to-date, changes, error rows), and the unit statement.\n\n" +
    "\n---\n".join(cuts))
print(survey)
spec = llm_query("Map the question to series in these tables. QUESTION: " + QUESTION + "\nSTRUCTURE: " + survey +
    '\nReturn ONLY JSON: {"series": [{"role": "...", "caption": "<quoted caption>", "column": "<group> / '
    '<category> header words"}], "period": ["YYYY-MM", "YYYY-MM"], "rules": "<every definition the question states: '
    'counting, rounding, neighbours, output form>", "output": "month list | number | percentage | name"}')
print(spec)
```
Compare the spec with the question's exact words before block 2; where they differ, the question decides.

Block 2 -- build the ledger in Python from EVERY release, for the needed tables only:
- Date each release by the reference period it states (cross-check: the newest period its tables print), not by
  list position or an issue date.
- One parser over all releases: ledger[(table, column)][YYYY-MM] = value, releases oldest -> newest, the newest
  release overwriting.
- Map columns by header MEANING: a group cell spans several category cells; take the non-empty category cells in
  order under their group. OCR noise is common: header rows can carry empty cells that line up with nothing, and data
  rows can end in empty cells -- drop empty header cells and trailing empty data cells, then check that every data
  row has exactly one value per category.
- Period rows: a year row (possibly with a revision mark) sets the year for the month rows under it, unless each row
  carries its own year. Skip rows that are not one period's value (totals, year-to-date, changes, error / interval
  rows); a suppressed cell is missing, not zero.
- Multiply by the unit stated with the table (thousands -> x1000, millions -> x1e6) when you store the value.
- Coverage: every release must contribute its whole window for every needed table; print the releases that did not.
  Tables your parser cannot read go to the sub-LM in ONE llm_query_batched call, one prompt per table holding just
  that table: 'Return ONLY JSON {"YYYY-MM": value as printed} for the column <header words>.' Re-ask the missing
  ones once and merge them by release date.

Block 3 -- compute, check, submit:
- Restrict every series to the question's period (inclusive) first.
- Follow the question's own definitions exactly -- how a run's length is counted, how a top share of months is
  rounded, what a moving average includes, which months count as neighbours, whether a ratio is averaged per month
  or taken over totals, and whether the result is a percentage or a plain number. Never substitute a convention the
  question did not state.
- Then ONE verification sub-call with only what decides the answer -- the question, one sentence saying what you
  computed (series, period, rule), at most 40 lines of the deciding values, and the result:
  llm_query("Does this computation answer the question as defined? Reply OK, or name the ONE concrete mismatch
  (wrong table/column, period edge, rule, units, output format).\nQUESTION: ...\nCOMPUTED: ...\nVALUES:
  ...\nRESULT: ...")
  Fix a concrete mismatch once; ignore a vague one.
- Format and submit in the same block: months as "YYYY-MM"; lists via json.dumps; numbers with 2 decimals, no
  commas or units; percentages as "12.34%" only when the question asks for a percentage; a name only.
  answer["content"] = ...; answer["ready"] = True
