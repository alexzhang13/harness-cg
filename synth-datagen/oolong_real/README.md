# OOLONG-Real (D&D)

**Real task.** Answer counting and ordering questions (rolls, spells, who did what) over transcripts of real Dungeons
& Dragons sessions.

**Synthetic task** (`tabletop-synth`). Invented tabletop session logs (1-3 episodes of 700-1,400 lines) rendered from
hidden roll and spell event tables, with an OOLONG-Real-style question and an exact gold. Sub-calls go to the real
model.

**Notes.** `notes/v20.md` and `notes/v21.md`, two versions of one program: one sub-call learns the log's format,
reader sub-calls list every event chunk by chunk, Python does all counting, then answer (usually 4-5 turns).

```bash
export PYTHONPATH=src:environments/synthetic/synth_common:environments/synthetic/oolong_real:environments/synthetic/tabletop_synth
D=synth-datagen/oolong_real; O=outputs/synth-datagen/oolong_real/gen
python $D/prepare.py --episodes 3 && python $D/sanitize.py --build-names   # real cast names to reject
python $D/run.py --synth-seeds 36000-37500 --synth-surfaces v2 --synth-roll-frac 0.3 --arm gen_v21 \
  --note $D/notes/v21.md --host localhost --ports 8000,8001 --out $O      # likewise with v20.md
python $D/audit_sft.py --runs $O/gen_v21.jsonl --max-turns 10 --allow-copied-literal --out $O/audit.json
python $D/export_sft.py --runs $O/gen_v21.jsonl --dest $O/../sft --max-turns 10 --mixed-prompt \
  --allow-copied-literal --audit $O/audit.json --max-off 0.10
```
`audit_sft.py` checks the extracted events against the generator's tables; `export_sft.py` filters and strips the hint.
`sanitize.py` needs the real dataset at `$OOLONG_REAL_DATA_DIR`.
