"""Abstract REPL backend. The backend hosts one `worker.py` process per rollout."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecResult:
    """Result of one repl block."""

    stdout: str = ""
    stderr: str = ""
    final_answer: str | None = None
    execution_time: float = 0.0
    locals_keys: list[str] = field(default_factory=list)
    stdout_truncated_chars: int = 0
    stderr_truncated_chars: int = 0


class ReplBackend(ABC):
    """REPL session of one rollout."""

    @abstractmethod
    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None: ...

    @abstractmethod
    async def load_context(self, payload: Any, index: int | None = None) -> int: ...

    @abstractmethod
    async def execute(self, code: str) -> ExecResult: ...

    @abstractmethod
    async def stop(self) -> None: ...
