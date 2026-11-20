"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-bcp."""
from rl_staging import load_bcp as load_environment

__all__ = ["load_environment"]
