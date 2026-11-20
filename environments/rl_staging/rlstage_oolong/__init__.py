"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-oolong."""
from rl_staging import load_oolong as load_environment

__all__ = ["load_environment"]
