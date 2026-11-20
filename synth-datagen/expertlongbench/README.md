# ExpertLongBench (T01 / T03 / T04 / T06)

**Real task.** Fill an expert checklist (court-docket summary, recipe rationale, tutoring report, clinical visit note)
from a long source document.

**Synthetic task** (`expertlongbench-synth`). The same checklists over procedurally generated records: long dockets
(T01-like, 108k-1.2M chars) and short forms (visit / tutor / recipe, like T06 / T04 / T03). The sub-LM is a
programmed reader that sometimes omits, misreads or confabulates; the reward is a deterministic per-field matcher. No
real ELB rows are used.

**Notes.** One final note per record kind, shipped inside the env (`expertlongbench_synth.final_notes`): docket
`t01_note_c6`, recipe `t03_note_c5`, tutor `t04_note_c6`, visit `t06_note_c10`.

```bash
E=synth-datagen/expertlongbench; O=outputs/synth-datagen/expertlongbench
export PYTHONPATH=src:environments/synthetic/synth_common:environments/synthetic/expertlongbench:environments/real/expertlongbench
for k in "docket 1008" "visit 648" "tutor 488" "recipe 488"; do set -- $k   # the published seed ranges
  python $E/gen_synth.py --kind $1 --start 0 --n $2 --servers http://localhost:8000/v1 --per-server 5 --out $O/gen
done
python $E/export_synth_sft.py --gen $O/gen --elb-contam --out $O/sft.jsonl
```
