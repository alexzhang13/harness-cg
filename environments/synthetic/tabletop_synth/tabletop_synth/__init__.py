"""Synthetic tabletop (D&D) session logs for distilling the OOLONG-real roll / spell program. See ../README.md."""

from .env import build_dataset, load_environment
from .generate import generate, root_prompt_for

__all__ = ["build_dataset", "generate", "load_environment", "root_prompt_for"]
