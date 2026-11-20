"""REPL worker process. Speaks JSONL over stdio ({"id", "type": exec | load_context | shutdown} -> {"id", "ok", ...}).

The namespace matches rlm's LocalREPL: `context`, `llm_query[_batched]`, `rlm_query[_batched]` (same as llm_query at
depth 1), `answer` and `SHOW_VARS`. Sub-LLM calls are POSTed to RLM_TRAIN_PROXY_URL. Stdlib only.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import resource
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any

# Same as rlm.environments.local_repl._SAFE_BUILTINS, so trained models behave the same under rlm.
_SAFE_BUILTINS = {
    "print": print, "len": len, "str": str, "int": int, "float": float,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "bool": bool,
    # needed as a hashable visited-set key in search code
    "frozenset": frozenset,
    "Ellipsis": Ellipsis, "NotImplemented": NotImplemented,
    "aiter": aiter, "anext": anext,
    "type": type, "isinstance": isinstance, "issubclass": issubclass,
    "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
    "sorted": sorted, "reversed": reversed, "range": range,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "any": any, "all": all, "pow": pow, "divmod": divmod,
    "chr": chr, "ord": ord, "hex": hex, "bin": bin, "oct": oct,
    "repr": repr, "ascii": ascii, "format": format, "hash": hash, "id": id,
    "iter": iter, "next": next, "slice": slice, "callable": callable,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr, "delattr": delattr,
    "dir": dir, "vars": vars,
    "bytes": bytes, "bytearray": bytearray, "memoryview": memoryview,
    "complex": complex, "object": object, "super": super, "property": property,
    "staticmethod": staticmethod, "classmethod": classmethod,
    "__import__": __import__, "open": open,
    "Exception": Exception, "BaseException": BaseException,
    "ValueError": ValueError, "TypeError": TypeError, "KeyError": KeyError,
    "IndexError": IndexError, "AttributeError": AttributeError,
    "FileNotFoundError": FileNotFoundError, "OSError": OSError, "IOError": IOError,
    "RuntimeError": RuntimeError, "NameError": NameError, "ImportError": ImportError,
    "StopIteration": StopIteration, "AssertionError": AssertionError,
    "NotImplementedError": NotImplementedError, "ArithmeticError": ArithmeticError,
    # all builtin exception classes, so `except SomeError:` clauses work
    "ZeroDivisionError": ZeroDivisionError, "OverflowError": OverflowError,
    "FloatingPointError": FloatingPointError, "RecursionError": RecursionError,
    "MemoryError": MemoryError, "BufferError": BufferError,
    "ModuleNotFoundError": ModuleNotFoundError,
    "EOFError": EOFError, "TimeoutError": TimeoutError,
    "UnboundLocalError": UnboundLocalError, "SyntaxError": SyntaxError,
    "IndentationError": IndentationError, "TabError": TabError,
    "SystemError": SystemError, "ReferenceError": ReferenceError,
    "StopAsyncIteration": StopAsyncIteration, "GeneratorExit": GeneratorExit,
    "EnvironmentError": EnvironmentError, "ConnectionError": ConnectionError,
    "ConnectionAbortedError": ConnectionAbortedError,
    "ConnectionRefusedError": ConnectionRefusedError,
    "ConnectionResetError": ConnectionResetError,
    "BrokenPipeError": BrokenPipeError, "BlockingIOError": BlockingIOError,
    "ChildProcessError": ChildProcessError, "FileExistsError": FileExistsError,
    "IsADirectoryError": IsADirectoryError,
    "NotADirectoryError": NotADirectoryError,
    "PermissionError": PermissionError, "ProcessLookupError": ProcessLookupError,
    "InterruptedError": InterruptedError,
    "UnicodeError": UnicodeError, "UnicodeDecodeError": UnicodeDecodeError,
    "UnicodeEncodeError": UnicodeEncodeError,
    "UnicodeTranslateError": UnicodeTranslateError,
    "ExceptionGroup": ExceptionGroup, "BaseExceptionGroup": BaseExceptionGroup,
    "UserWarning": UserWarning,
    "DeprecationWarning": DeprecationWarning,
    "PendingDeprecationWarning": PendingDeprecationWarning,
    "RuntimeWarning": RuntimeWarning, "SyntaxWarning": SyntaxWarning,
    "FutureWarning": FutureWarning, "ImportWarning": ImportWarning,
    "UnicodeWarning": UnicodeWarning, "BytesWarning": BytesWarning,
    "ResourceWarning": ResourceWarning, "EncodingWarning": EncodingWarning,
    "LookupError": LookupError, "Warning": Warning,
    "input": None, "eval": None, "exec": None, "compile": None,
    "globals": None, "locals": None,
}

RESERVED_TOOL_NAMES = frozenset({
    "llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched",
    "SHOW_VARS", "answer", "context", "history",
})

# Sub-LLM functions, bound only at depth >= 1. At depth 0 (LM + REPL) calling them is a NameError.
_SUB_LLM_NAMES = frozenset({
    "llm_query", "llm_query_batched", "rlm_query", "rlm_query_batched",
})


class _AnswerDict(dict):
    """The `answer` dict (as in rlm's _AnswerDict): calls on_ready when `ready` first becomes truthy."""

    def __init__(self, on_ready=None):
        super().__init__()
        super().__setitem__("content", "")
        super().__setitem__("ready", False)
        self._on_ready = on_ready

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key == "ready" and value and self._on_ready is not None:
            try:
                self._on_ready(self.get("content", ""))
            except Exception:
                pass


class Worker:
    def __init__(
        self,
        proxy_url: str,
        rollout_id: str,
        depth: int = 1,
        exec_timeout_s: float | None = None,
    ):
        self.proxy_url = proxy_url.rstrip("/")
        self.rollout_id = rollout_id
        self.depth = depth
        # Soft per-block timeout (SIGALRM, pure-Python loops only). The host kills the worker at 2x this value.
        # <= 0 disables it.
        if exec_timeout_s is None:
            try:
                exec_timeout_s = float(os.environ.get("RLM_TRAIN_EXEC_TIMEOUT_S", "600"))
            except ValueError:
                exec_timeout_s = 600.0
        self.exec_timeout_s = exec_timeout_s
        self._lock = threading.Lock()  # serializes stdout capture
        self._last_final_answer: str | None = None
        self._context_count = 0
        self.globals: dict[str, Any] = {}
        self.locals: dict[str, Any] = {}
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        self.globals = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals = {}
        self.globals["SHOW_VARS"] = self._show_vars
        if self.depth >= 1:
            self.globals["llm_query"] = self._llm_query
            self.globals["llm_query_batched"] = self._llm_query_batched
            # depth 1: rlm_query is llm_query.
            self.globals["rlm_query"] = self._llm_query
            self.globals["rlm_query_batched"] = self._llm_query_batched
        self.locals["answer"] = _AnswerDict(on_ready=self._capture_answer)

    def _restore_scaffold(self) -> None:
        for name in RESERVED_TOOL_NAMES:
            # never bound at depth 0
            if name in _SUB_LLM_NAMES and self.depth < 1:
                continue
            if name == "llm_query":
                self.globals["llm_query"] = self._llm_query
            elif name == "llm_query_batched":
                self.globals["llm_query_batched"] = self._llm_query_batched
            elif name == "rlm_query":
                self.globals["rlm_query"] = self._llm_query
            elif name == "rlm_query_batched":
                self.globals["rlm_query_batched"] = self._llm_query_batched
            elif name == "SHOW_VARS":
                self.globals["SHOW_VARS"] = self._show_vars
            elif name == "answer":
                current = self.locals.get("answer")
                # the model rebound `answer` to a plain dict: capture it here and re-wrap
                if not isinstance(current, _AnswerDict):
                    replacement = _AnswerDict(on_ready=self._capture_answer)
                    if isinstance(current, dict):
                        for k, v in current.items():
                            dict.__setitem__(replacement, k, v)
                        if current.get("ready") and self._last_final_answer is None:
                            self._last_final_answer = str(current.get("content", ""))
                    self.locals["answer"] = replacement
            elif name == "context" and "context_0" in self.locals:
                self.locals["context"] = self.locals["context_0"]

    def _capture_answer(self, content: Any) -> None:
        self._last_final_answer = str(content)

    def _show_vars(self) -> str:
        available = {
            k: type(v).__name__
            for k, v in self.locals.items()
            if not k.startswith("_") and k != "answer"
        }
        if not available:
            return "No variables created yet. Use ```repl``` blocks to create variables."
        return f"Available variables: {available}"

    def _proxy_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.proxy_url}/rollout/{self.rollout_id}/{path.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # per-sub-call wall clock (s)
        try:
            _sub_timeout = float(os.environ.get("RLM_TRAIN_SUBCALL_TIMEOUT_S", "600"))
        except ValueError:
            _sub_timeout = 600.0
        try:
            with urllib.request.urlopen(req, timeout=_sub_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = ""
            too_large = e.code == 413 or "Entity Too Large" in (detail or "") or "Entity Too Large" in (e.reason or "")
            return {
                "error": f"HTTP {e.code}: {detail or e.reason}",
                "too_large": too_large,
            }
        except Exception as e:
            return {"error": f"Proxy request failed: {e}"}

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        # `model` is accepted for compatibility but ignored: sub-calls always go to the configured model.
        del model  # noqa: F841
        result = self._proxy_post("llm_query", {"prompt": prompt, "depth": self.depth})
        if "error" in result and result["error"]:
            if result.get("too_large"):
                return (
                    "Error: sub-LLM prompt exceeded the endpoint's request-size limit "
                    f"(prompt was {len(prompt):,} chars). Shorten or chunk the prompt — "
                    "summarize the context first, or split it across multiple llm_query "
                    f"calls. Underlying error: {result['error']}"
                )
            return f"Error: {result['error']}"
        return result.get("response", "")

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str]:
        if not prompts:
            return []
        del model  # noqa: F841
        result = self._proxy_post(
            "llm_query_batched", {"prompts": list(prompts), "depth": self.depth}
        )
        if "error" in result and result["error"]:
            if result.get("too_large"):
                total = sum(len(p) for p in prompts)
                longest = max(len(p) for p in prompts)
                msg = (
                    "Error: sub-LLM batched request exceeded the endpoint's request-size "
                    f"limit ({len(prompts)} prompts, total {total:,} chars, longest "
                    f"{longest:,} chars). Either split into smaller batches (fewer prompts "
                    "per call) or shorten individual prompts (summarize/chunk the context). "
                    f"Underlying error: {result['error']}"
                )
                return [msg] * len(prompts)
            return [f"Error: {result['error']}"] * len(prompts)
        responses = result.get("responses")
        if not isinstance(responses, list) or len(responses) != len(prompts):
            return ["Error: malformed batched response"] * len(prompts)
        return [r if isinstance(r, str) else f"Error: {r}" for r in responses]

    def load_context(self, payload: Any, index: int | None = None) -> int:
        if index is None:
            index = self._context_count
        if isinstance(payload, dict) and isinstance(payload.get("__rlm_doc_store__"), dict):
            payload = _materialize_doc_store(payload["__rlm_doc_store__"])
        var = f"context_{index}"
        self.locals[var] = payload
        if index == 0:
            self.locals["context"] = payload
        self._context_count = max(self._context_count, index + 1)
        return index

    @contextmanager
    def _capture_output(self):
        with self._lock:
            old_out, old_err = sys.stdout, sys.stderr
            out_buf, err_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = out_buf, err_buf
                yield out_buf, err_buf
            finally:
                sys.stdout, sys.stderr = old_out, old_err

    def _exec_with_timeout(self, code: str, ns: dict[str, Any]) -> None:
        timeout_s = self.exec_timeout_s
        if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
            exec(code, ns, ns)  # noqa: S102 - intentional REPL
            return

        def _on_alarm(signum, frame):  # noqa: ARG001
            raise TimeoutError(
                f"```repl``` block exceeded {timeout_s:g}s execution timeout"
            )

        old_handler = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)
        try:
            exec(code, ns, ns)  # noqa: S102 - intentional REPL
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def execute(self, code: str) -> dict[str, Any]:
        start = time.perf_counter()
        with self._capture_output() as (out_buf, err_buf):
            try:
                combined = {**self.globals, **self.locals}
                self._exec_with_timeout(code, combined)
                for k, v in combined.items():
                    if k not in self.globals and not k.startswith("_"):
                        self.locals[k] = v
                self._restore_scaffold()
                stdout = out_buf.getvalue()
                stderr = err_buf.getvalue()
            except BaseException as e:  # noqa: BLE001 - mirror rlm: even SystemExit gets caught
                stdout = out_buf.getvalue()
                stderr = err_buf.getvalue() + f"\n{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                if tb and tb.strip() and tb not in stderr:
                    stderr = stderr + "\n" + tb
        final_answer = self._last_final_answer
        self._last_final_answer = None
        # Same filter as rlm.utils.parsing.format_execution_result (rendered as `REPL variables: [...]`), so
        # `answer` is included here although SHOW_VARS hides it.
        simple_keys = [
            k for k, v in self.locals.items()
            if not k.startswith("_")
            and k not in ("__builtins__", "__name__", "__doc__")
            and isinstance(v, (str, int, float, bool, list, dict, tuple))
        ]
        stdout, stdout_cut = _clip_stream(stdout)
        stderr, stderr_cut = _clip_stream(stderr)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "final_answer": final_answer,
            "execution_time": time.perf_counter() - start,
            "locals_keys": simple_keys,
            # chars cut by _clip_stream; the host adds them back to its count
            "stdout_truncated_chars": stdout_cut,
            "stderr_truncated_chars": stderr_cut,
        }


# Max stdout/stderr chars per exec reply; None = no limit. A reply over the host's 16 MiB line
# limit is discarded with an error asking for a smaller print.
_MAX_STREAM_CHARS: int | None = None


def _clip_stream(s: str) -> tuple[str, int]:
    """Return (s cut to _MAX_STREAM_CHARS, number of chars dropped)."""
    if _MAX_STREAM_CHARS is None or len(s) <= _MAX_STREAM_CHARS:
        return s, 0
    return s[:_MAX_STREAM_CHARS], len(s) - _MAX_STREAM_CHARS


# JSONL stdio loop
def _send(obj: dict[str, Any]) -> None:
    line = json.dumps(obj, ensure_ascii=False)
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-url", default=os.environ.get("RLM_TRAIN_PROXY_URL", ""))
    parser.add_argument("--rollout-id", default=os.environ.get("RLM_TRAIN_ROLLOUT_ID", ""))
    parser.add_argument("--depth", type=int, default=int(os.environ.get("RLM_TRAIN_DEPTH", "1")))
    args = parser.parse_args()

    if not args.proxy_url or not args.rollout_id:
        _send({"id": "_init", "ok": False, "error": "missing proxy-url or rollout-id"})
        return

    # Run in a fresh scratch dir so files written by model code stay out of the repo.
    # RLM_TRAIN_REPL_SCRATCH sets the parent dir, RLM_TRAIN_REPL_KEEP_SCRATCH=1 keeps it.
    scratch = tempfile.mkdtemp(prefix=f"rlm-repl-{args.rollout_id[:24]}-", dir=os.environ.get("RLM_TRAIN_REPL_SCRATCH") or None)
    os.chdir(scratch)
    sys.argv[0] = os.path.join(scratch, "repl_worker.py")
    sys.modules["__main__"].__file__ = sys.argv[0]

    # Memory cap (RLIMIT_DATA), so a runaway allocation is a MemoryError in the cell instead of killing the worker.
    # RLM_TRAIN_REPL_MAX_MEM_GB, default 64, 0 disables.
    _cap_gb = float(os.environ.get("RLM_TRAIN_REPL_MAX_MEM_GB", "64") or 0)
    if _cap_gb > 0:
        _cap = int(_cap_gb * 1024**3)
        try:
            _soft, _hard = resource.getrlimit(resource.RLIMIT_DATA)
            _hard_new = _cap if _hard == resource.RLIM_INFINITY else min(_cap, _hard)
            resource.setrlimit(resource.RLIMIT_DATA, (_hard_new, _hard_new))
        except (ValueError, OSError) as e:  # run uncapped
            print(f"[worker] could not set RLIMIT_DATA={_cap_gb}GB: {e!r}", file=sys.stderr)

    worker = Worker(proxy_url=args.proxy_url, rollout_id=args.rollout_id, depth=args.depth)
    _send({"id": "_init", "ok": True, "scratch": scratch, "mem_cap_gb": _cap_gb})
    global _SCRATCH
    _SCRATCH = scratch
    try:
        _serve(worker)
    finally:
        _cleanup_scratch()


_SCRATCH: str | None = None


def _materialize_doc_store(ref: dict) -> list[str]:
    """Load a context passed by reference to a doc store (see rlm_train.doc_store), preferring a node-local copy."""
    import mmap
    path = ref["path"]
    loc = os.path.join(os.environ.get("RLM_DOC_STORE_LOCAL_ROOT") or "/tmp/rlm_doc_stores", os.path.basename(path.rstrip("/")))
    try:
        if os.path.getsize(os.path.join(loc, "docs.bin")) == os.path.getsize(os.path.join(path, "docs.bin")) \
                and os.path.exists(os.path.join(loc, "index.json")):
            path = loc
    except OSError:
        pass
    with open(os.path.join(path, "index.json")) as f:
        offsets = json.load(f)["offsets"]
    with open(os.path.join(path, "docs.bin"), "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return [mm[offsets[i]:offsets[i + 1]].decode("utf-8", "surrogatepass") for i in ref["idx"]]


def _cleanup_scratch() -> None:
    global _SCRATCH
    if _SCRATCH and os.environ.get("RLM_TRAIN_REPL_KEEP_SCRATCH") != "1":
        try:
            os.chdir("/")
        except OSError:
            pass
        shutil.rmtree(_SCRATCH, ignore_errors=True)
    _SCRATCH = None


def _serve(worker: "Worker") -> None:
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError as e:
            _send({"id": "?", "ok": False, "error": f"bad json: {e}"})
            continue

        rid = req.get("id", "?")
        kind = req.get("type")

        if kind == "exec":
            try:
                result = worker.execute(req.get("code", ""))
                _send({"id": rid, "ok": True, **result})
            except BaseException as e:  # noqa: BLE001
                _send({"id": rid, "ok": False, "error": f"exec failed: {e}\n{traceback.format_exc()}"})
        elif kind == "load_context":
            try:
                idx = worker.load_context(req.get("payload"), req.get("index"))
                _send({"id": rid, "ok": True, "index": idx})
            except BaseException as e:  # noqa: BLE001
                _send({"id": rid, "ok": False, "error": f"load_context failed: {e}"})
        elif kind == "shutdown":
            # clean up before acking: the parent kills us right after
            _cleanup_scratch()
            _send({"id": rid, "ok": True})
            return
        else:
            _send({"id": rid, "ok": False, "error": f"unknown type: {kind!r}"})


if __name__ == "__main__":
    main()
