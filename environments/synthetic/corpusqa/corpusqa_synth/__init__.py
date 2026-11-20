"""Fast synthetic training env for CorpusQA (see ../DESIGN.md and ../PROGRAM.md)."""

from .env import generate, load_environment, score

__all__ = ["generate", "load_environment", "score"]
