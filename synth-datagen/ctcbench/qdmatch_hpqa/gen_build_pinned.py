"""ctc-data build with the 256k qdmatch_hpqa document count pinned to the published eval rows' 3,488 items (1,744 questions + 1,744 passages)."""
import sys
from ctc.data import ladders, build
orig = ladders.docs_for_rung
def pinned(task, rung):
    return 3488 if task == "qdmatch_hpqa" else orig(task, rung)
ladders.docs_for_rung = pinned
for m in list(sys.modules.values()):
    if getattr(m, "docs_for_rung", None) is orig:
        m.docs_for_rung = pinned
from ctc.data.cli import main
sys.exit(main())
