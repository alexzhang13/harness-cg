"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-oolong-real-val."""
from rl_staging import load_oolong_real_val as load_environment

__all__ = ["load_environment"]
