"""verifiers-v1 SandboxClient without containers: each sandbox is a temp dir used as cwd and HOME, commands run
with bash -lc on the host. No filesystem isolation.

VF_LOCAL_SANDBOX_NODE_BIN (node/pi dir), VF_LOCAL_SANDBOX_BASE, VF_LOCAL_SANDBOX_KEEP, VF_LOCAL_SANDBOX_ALLOW_NET.
"""

from __future__ import annotations

import asyncio
import logging
import os
import posixpath
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _CommandResult:
    exit_code: int
    stdout: str | None
    stderr: str | None


@dataclass
class _SandboxRecord:
    id: str


async def _run(
    args: list[str],
    *,
    timeout: int | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
        env=env,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timed out after {timeout}s: {' '.join(args[:3])}..."
    rc = proc.returncode if proc.returncode is not None else -1
    return rc, out_b.decode(errors="replace"), err_b.decode(errors="replace")


_ROOTS: dict[str, Path] = {}   # sandbox id -> root dir

DEFAULT_NODE_BIN = os.path.expanduser("~/node/bin")


class LocalProcessSandboxClient:
    """Sandboxes as host subprocesses in temp dirs."""

    def __init__(self) -> None:
        self._node_bin = os.environ.get("VF_LOCAL_SANDBOX_NODE_BIN", DEFAULT_NODE_BIN)
        self._keep = _truthy(os.environ.get("VF_LOCAL_SANDBOX_KEEP"))
        self._base = os.environ.get("VF_LOCAL_SANDBOX_BASE") or None

    async def create(self, request: object) -> _SandboxRecord:
        base_name = str(_attr(request, "name", None) or "vf-proc")
        sandbox_id = f"{base_name}-{uuid.uuid4().hex[:8]}"
        root = Path(tempfile.mkdtemp(prefix=f"{sandbox_id}-", dir=self._base))
        _ROOTS[sandbox_id] = root
        logger.info("local process sandbox: %s -> %s", sandbox_id, root)
        return _SandboxRecord(id=sandbox_id)

    async def wait_for_creation(self, sandbox_id: str) -> object:
        return None

    async def delete(self, sandbox_id: str) -> object:
        root = _ROOTS.pop(sandbox_id, None)
        if root is None:
            return None
        if self._keep:
            logger.info("VF_LOCAL_SANDBOX_KEEP set; leaving %s", root)
            return None
        await asyncio.to_thread(shutil.rmtree, root, ignore_errors=True)
        return None

    async def aclose(self) -> object:
        return None

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _CommandResult:
        root = self._root(sandbox_id)
        cwd = self._resolve(root, working_dir) if working_dir else root
        cwd.mkdir(parents=True, exist_ok=True)
        run_env = self._build_env(root, env)
        rc, out, err = await _run(
            ["bash", "-lc", command],
            timeout=timeout,
            cwd=str(cwd),
            env=run_env,
        )
        return _CommandResult(exit_code=rc, stdout=out, stderr=err)

    async def run_background_job(
        self,
        sandbox_id: str,
        command: str,
        *,
        timeout: int | None = None,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _CommandResult:
        return await self.execute_command(
            sandbox_id,
            command,
            timeout=timeout,
            working_dir=working_dir,
            env=env,
        )

    async def upload_bytes(
        self,
        sandbox_id: str,
        file_path: str,
        file_bytes: bytes,
        *,
        filename: str | None = None,
    ) -> object:
        dest = self._resolve(self._root(sandbox_id), file_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(dest.write_bytes, file_bytes)
        return None

    async def upload_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        *,
        timeout: int | None = None,
    ) -> object:
        dest = self._resolve(self._root(sandbox_id), file_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, local_file_path, dest)
        return None

    async def download_file(
        self,
        sandbox_id: str,
        file_path: str,
        local_file_path: str,
        *,
        timeout: int | None = None,
    ) -> object:
        src = self._resolve(self._root(sandbox_id), file_path)
        if not src.exists():
            raise FileNotFoundError(f"sandbox file not found: {file_path}")
        Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, src, local_file_path)
        return None

    async def read_file(self, sandbox_id: str, path: str) -> object:
        src = self._resolve(self._root(sandbox_id), path)
        if not src.exists():
            raise FileNotFoundError(f"sandbox file not found: {path}")
        return await asyncio.to_thread(src.read_text)

    def _root(self, sandbox_id: str) -> Path:
        root = _ROOTS.get(sandbox_id)
        if root is None:
            raise KeyError(f"unknown sandbox id {sandbox_id!r}")
        return root

    def _resolve(self, root: Path, guest_path: str) -> Path:
        """Guest path -> path under the sandbox root."""
        rel = posixpath.normpath(guest_path).lstrip("/")
        if rel in ("", "."):
            return root
        resolved = (root / rel).resolve()
        root_resolved = root.resolve()
        if root_resolved != resolved and root_resolved not in resolved.parents:
            raise ValueError(f"guest path {guest_path!r} escapes sandbox root")
        return resolved

    def _build_env(self, root: Path, extra: dict[str, str] | None) -> dict[str, str]:
        env = dict(os.environ)
        env["HOME"] = str(root)
        tmp = root / ".tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(tmp)
        path = env.get("PATH", "")
        if self._node_bin and self._node_bin not in path.split(os.pathsep):
            env["PATH"] = f"{self._node_bin}{os.pathsep}{path}"
        # crude network lockdown: dead HTTP(S) proxy, localhost exempt
        if not _truthy(os.environ.get("VF_LOCAL_SANDBOX_ALLOW_NET")):
            env["PI_OFFLINE"] = "1"
            dead = "http://127.0.0.1:9"
            for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
                env[k] = dead
            for k in ("no_proxy", "NO_PROXY"):
                env[k] = "127.0.0.1,localhost,::1"
        else:
            env.setdefault("PI_OFFLINE", "0")
        if extra:
            env.update({str(k): str(v) for k, v in extra.items()})
        return env


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in {"0", "false", "no", ""}


def _attr(obj: object, name: str, default: object) -> object:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def install_local_process_backend() -> None:
    """Make verifiers-v1 use LocalProcessSandboxClient."""
    import prime_sandboxes

    if getattr(prime_sandboxes, "_vf_local_backend", None) == "process":
        return
    prime_sandboxes.AsyncSandboxClient = LocalProcessSandboxClient  # type: ignore[attr-defined]
    prime_sandboxes._vf_local_backend = "process"  # type: ignore[attr-defined]
    logger.info("verifiers v1 sandbox backend -> LocalProcessSandboxClient")
