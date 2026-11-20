"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-longreason."""
from rl_staging import load_longreason as load_environment

__all__ = ["load_environment"]
