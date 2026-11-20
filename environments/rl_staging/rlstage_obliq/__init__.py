"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-obliq."""
from rl_staging import load_obliq as load_environment

__all__ = ["load_environment"]
