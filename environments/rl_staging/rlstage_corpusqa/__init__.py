"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-corpusqa."""
from rl_staging import load_corpusqa as load_environment

__all__ = ["load_environment"]
