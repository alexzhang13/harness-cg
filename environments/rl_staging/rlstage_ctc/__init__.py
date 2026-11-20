"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-ctc."""
from rl_staging import load_ctc as load_environment

__all__ = ["load_environment"]
