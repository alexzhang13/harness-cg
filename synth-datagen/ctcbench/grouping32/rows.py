"""Gate/train rows: synthetic grouping (compose_v3) rendered in a surface mode."""
_REPO = __import__("os").path.abspath(__file__ + "/../../../..")
import sys
R = _REPO
sys.path[:0] = [f'{R}/environments/real/CTCBench', f'{R}/environments/synthetic/ctcbench', f'{R}/environments/synthetic/synth_common', f'{R}/src']
import compose
from ctcbench_synth import surface as S
FAM = 'grouping'
def gate_rows(n=12, modes=('native', 'varied', 'xml', 'yaml')):
    out = []
    for mi, mode in enumerate(modes):
        for k in range(n):
            ex = compose.compose_v3('dev', k, seed=0)
            r = S.render(ex, FAM, mode=mode, seed=(1000 + k) if mode != 'native' else None)
            out.append((f'{mode}:{k}', ex, r))
    return out
