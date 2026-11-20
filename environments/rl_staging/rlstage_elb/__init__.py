"""Verifiers imports an env id as a module (hyphens -> underscores): shim for rlstage-elb."""
from rl_staging import load_elb as load_environment

__all__ = ["load_environment"]
