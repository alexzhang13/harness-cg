"""LongCoT-mini: the ``easy`` difficulty of LongCoT (507 rows), via ``longcot_real``."""

from __future__ import annotations

from typing import Any

from longcot_real import load_environment as _load_longcot


def load_environment(**kwargs: Any):
    """``longcot_real`` pinned to ``difficulty="easy"``."""
    if "difficulty" in kwargs and kwargs["difficulty"] not in (None, "easy"):
        raise ValueError(
            "longcot-mini-real pins difficulty='easy'; pass a different "
            "difficulty to the full longcot-real env instead."
        )
    kwargs.pop("difficulty", None)
    return _load_longcot(difficulty="easy", **kwargs)


__all__ = ["load_environment"]
