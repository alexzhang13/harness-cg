"""Subprocess REPL backend: one `worker.py` child process per rollout, JSONL over its stdin/stdout."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from rlm_train.repl.base import ExecResult, ReplBackend


class WorkerStartupError(RuntimeError):
    pass


class WorkerProtocolError(RuntimeError):
    pass


def _request_number(rid: Any) -> int | None:
    """"r12" -> 12, else None."""
    if isinstance(rid, str) and rid.startswith("r") and rid[1:].isdigit():
        return int(rid[1:])
    return None


class _OversizedLine(WorkerProtocolError):
    """A worker line over the stream limit (already discarded)."""


class _BadLine(WorkerProtocolError):
    """A non-JSON worker line (e.g. code writing to fd 1)."""


class SubprocessReplBackend(ReplBackend):
    _STREAM_LIMIT = 16 * 1024 * 1024  # max bytes of one worker reply line

    def __init__(
        self,
        python: str | None = None,
        worker_module: str = "rlm_train.worker",
        startup_timeout: float | None = None,
        request_timeout: float | None = None,
    ):
        self._python = python or sys.executable
        self._worker_module = worker_module
        if startup_timeout is None:
            startup_timeout = float(os.environ.get("RLM_TRAIN_WORKER_STARTUP_TIMEOUT_S", "120"))
        self._startup_timeout = startup_timeout
        # kill a worker that has not answered within this long
        if request_timeout is None:
            request_timeout = float(os.environ.get("RLM_TRAIN_REPL_REQUEST_TIMEOUT_S", "1200"))
        self._request_timeout = request_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._req_counter = 0
        self._lock = asyncio.Lock()
        self._stderr_task: asyncio.Task | None = None
        self._stderr_buf: list[str] = []
        self._scratch: str | None = None
        self._launch_copy: str | None = None

    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None:
        env = os.environ.copy()
        env["RLM_TRAIN_PROXY_URL"] = proxy_url
        env["RLM_TRAIN_ROLLOUT_ID"] = rollout_id
        env["RLM_TRAIN_DEPTH"] = str(depth)
        env.setdefault("PYTHONUNBUFFERED", "1")

        argv = self._worker_argv()
        # run a temp copy of the worker, deleted once it is up, so model code cannot overwrite the real file
        self._launch_copy = None
        if len(argv) == 1 and os.environ.get("RLM_TRAIN_WORKER_LAUNCH", "copy") == "copy":
            import shutil
            import tempfile
            fd, copy_path = tempfile.mkstemp(prefix="rlm_worker_", suffix=".py")
            os.close(fd)
            shutil.copyfile(argv[0], copy_path)
            self._launch_copy = copy_path
            argv = [copy_path]
        self._proc = await asyncio.create_subprocess_exec(
            self._python, "-u", *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=self._STREAM_LIMIT,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        try:
            init = await asyncio.wait_for(self._read_line(), timeout=self._startup_timeout)
        except TimeoutError as e:
            await self.stop()
            raise WorkerStartupError(
                f"Worker did not produce init line in {self._startup_timeout}s; "
                f"stderr={self._stderr_text()!r}"
            ) from e
        except WorkerProtocolError as e:
            rc = None
            try:
                rc = await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                pass
            await asyncio.sleep(0.05)
            err = self._stderr_text()
            await self.stop()
            raise WorkerStartupError(
                f"Worker exited during startup (returncode={rc}, argv={self._worker_argv()}, "
                f"python={self._python}); stderr={err!r}"
                + (" -- exit 0 with no output: the worker source may be corrupted or not run main()" if rc == 0 and not err else "")
            ) from e
        if not init.get("ok"):
            await self.stop()
            raise WorkerStartupError(f"Worker init failed: {init.get('error')}")
        self._scratch = init.get("scratch")
        self._unlink_launch_copy()

    def _unlink_launch_copy(self, final: bool = False) -> None:
        copy_path = getattr(self, "_launch_copy", None)
        if final:
            self._launch_copy = None
        if copy_path:
            try:
                os.unlink(copy_path)
            except OSError:
                pass

    def _worker_argv(self) -> list[str]:
        """Start the worker by path, so it does not import rlm_train."""
        if os.environ.get("RLM_TRAIN_WORKER_LAUNCH", "copy") != "module":
            try:
                import importlib.util
                spec = importlib.util.find_spec(self._worker_module)
                origin = getattr(spec, "origin", None) if spec else None
                if origin and origin.endswith(".py") and os.path.isfile(origin):
                    return [origin]
            except Exception:  # noqa: BLE001
                pass
        return ["-m", self._worker_module]

    async def stop(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.returncode is None:
                try:
                    await self._send({"id": "_shutdown", "type": "shutdown"})
                    await asyncio.wait_for(self._read_line(), timeout=2.0)
                except (TimeoutError, BrokenPipeError, ConnectionResetError, WorkerProtocolError):
                    pass
        finally:
            if self._proc.returncode is None:
                try:
                    self._proc.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
            if self._stderr_task and not self._stderr_task.done():
                self._stderr_task.cancel()
                try:
                    await self._stderr_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._proc = None
            self._stderr_task = None
            self._unlink_launch_copy(final=True)
            scratch = getattr(self, "_scratch", None)
            self._scratch = None
            if scratch and os.path.basename(scratch).startswith("rlm-repl-") and os.environ.get("RLM_TRAIN_REPL_KEEP_SCRATCH") != "1":
                import shutil
                shutil.rmtree(scratch, ignore_errors=True)

    async def load_context(self, payload: Any, index: int | None = None) -> int:
        result = await self._request({"type": "load_context", "payload": payload, "index": index})
        return int(result.get("index", 0))

    async def execute(self, code: str) -> ExecResult:
        result = await self._request({"type": "exec", "code": code})
        return ExecResult(
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            final_answer=result.get("final_answer"),
            execution_time=float(result.get("execution_time") or 0.0),
            locals_keys=list(result.get("locals_keys") or []),
            stdout_truncated_chars=int(result.get("stdout_truncated_chars") or 0),
            stderr_truncated_chars=int(result.get("stderr_truncated_chars") or 0),
        )

    async def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._proc is None:
            raise WorkerProtocolError("worker not started")
        async with self._lock:
            self._req_counter += 1
            rid = f"r{self._req_counter}"
            payload = {"id": rid, **payload}
            await self._send(payload)
            try:
                resp = await asyncio.wait_for(self._read_reply(rid), timeout=self._request_timeout)
            except TimeoutError as e:
                await self._kill_worker()
                raise WorkerProtocolError(
                    f"Request {payload.get('type')} exceeded parent watchdog "
                    f"({self._request_timeout:g}s); worker SIGKILLed."
                ) from e
        if resp.get("id") != rid:
            raise WorkerProtocolError(f"id mismatch: expected {rid!r}, got {resp.get('id')!r}")
        if not resp.get("ok"):
            raise WorkerProtocolError(f"worker error: {resp.get('error')}")
        return resp

    async def _kill_worker(self) -> None:
        """SIGKILL the worker (watchdog timeout)."""
        if self._proc is None or self._proc.returncode is not None:
            return
        try:
            self._proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        except TimeoutError:
            pass

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        line = (json.dumps(payload) + "\n").encode("utf-8")
        self._proc.stdin.write(line)
        await self._proc.stdin.drain()

    async def _read_reply(self, rid: str) -> dict[str, Any]:
        """Reply to request `rid`, skipping stale and non-JSON lines."""
        want = _request_number(rid)
        while True:
            try:
                resp = await self._read_line()
            except _BadLine as e:
                print(f"[repl] skipping a non-JSON worker line while waiting for {rid}: {str(e)[:200]}",
                      file=sys.stderr)
                continue
            got = resp.get("id")
            if got == rid:
                return resp
            n = _request_number(got)
            if n is not None and want is not None and n < want:
                print(f"[repl] dropping a stale worker reply {got!r} while waiting for {rid!r}", file=sys.stderr)
                continue
            raise WorkerProtocolError(f"id mismatch: expected {rid!r}, got {got!r}")

    async def _read_line(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        line = await self._read_raw_line()
        if not line:
            stderr = self._stderr_text()
            raise WorkerProtocolError(f"worker closed stdout; stderr={stderr!r}")
        try:
            return json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise _BadLine(f"bad worker line: {line[:500]!r}") from e

    async def _read_raw_line(self) -> bytes:
        """One stdout line (b"" at EOF). An oversized line is read to its end and raises _OversizedLine."""
        stream = self._proc.stdout
        try:
            return await stream.readuntil(b"\n")
        except asyncio.IncompleteReadError as e:
            return e.partial
        except asyncio.LimitOverrunError as e:
            dropped = 0
            pending = e.consumed
            while True:
                chunk = await stream.read(max(1, pending))
                if not chunk:
                    break
                dropped += len(chunk)
                try:
                    dropped += len(await stream.readuntil(b"\n"))
                    break
                except asyncio.IncompleteReadError as e2:
                    dropped += len(e2.partial)
                    break
                except asyncio.LimitOverrunError as e2:
                    pending = e2.consumed
            raise _OversizedLine(
                f"worker reply exceeded the {self._STREAM_LIMIT:,}-byte line limit ({dropped:,} bytes) and was "
                f"discarded; print a smaller slice"
            ) from None

    async def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        kept = 0
        while True:
            chunk = await self._proc.stderr.read(65536)
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            self._stderr_buf.append(text)
            kept += len(text)
            while kept > 65536 and len(self._stderr_buf) > 1:
                kept -= len(self._stderr_buf.pop(0))

    def _stderr_text(self) -> str:
        return "".join(self._stderr_buf)
