"""Read-only imports from the sibling ``oolong_real_synth`` package (phrase banks, question templates, scoring)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from oolong_real_synth import gen as G
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "oolong_real"))
    from oolong_real_synth import gen as G

from oolong_real_synth import world as W
from oolong_real_synth.scoring import score_synth

__all__ = ["G", "W", "score_synth"]
