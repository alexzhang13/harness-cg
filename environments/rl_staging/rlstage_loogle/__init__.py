"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-loogle."""
from rl_staging import load_loogle as load_environment

__all__ = ["load_environment"]
