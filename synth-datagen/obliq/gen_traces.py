"""Hinted trace generation in trick_synth with a real root: ``synth_common.probe`` plus the FULL ``rlm_history`` of each
rollout (system prompt, metadata message, hint, every turn untruncated) and the full sub-call log, the input of
build_sft_v7.py. Same CLI as ``python -m synth_common.probe`` (see run_gen.sh for the published runs)."""
import sys

from synth_common import probe as P

_orig = P.record


def record_full(state, row_idx, elapsed, trunc):
    out = _orig(state, row_idx, elapsed, trunc)
    out["history"] = [{"role": P._role(m), "content": P._content(m)} for m in (state.get("rlm_history") or [])]
    # full sub-call prompts/replies, for the trace-match profile (trace_profile.py); never trained on
    out["sub_call_log"] = [{"prompt": e.get("prompt") or "", "response": e.get("response") or ""} for e in (state.get("rlm_sub_llm_log") or [])]
    return out


P.record = record_full
if __name__ == "__main__":
    sys.argv[0] = "probe"; P.main()
