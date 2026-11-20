"""Generation stage: filter finished hinted rollouts, judge them, and write accepted (hint-stripped) / rejected rows."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from . import filters, sft_format
from .judge import Judge, accept as judge_accept, interesting as judge_interesting
from .strategies import Strategy


def _g(o, key, default=None):
    return o.get(key, default) if isinstance(o, dict) else getattr(o, key, default)


def _jsonable(x):
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, list):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    for attr in ("model_dump", "dict"):
        m = getattr(x, attr, None)
        if callable(m):
            try:
                return _jsonable(m())
            except Exception:  # noqa: BLE001
                break
    return str(x)


def _strip_reasoning(msgs: list | None) -> list:
    """Assistant turns as the harness executed them: leaked reasoning (up to the last </think>) removed."""
    from rlm_train.env import _strip_leaked_reasoning
    out = []
    for m in msgs or []:
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m = {**m, "content": _strip_leaked_reasoning(m["content"])}
        out.append(m)
    return out


def _row(o) -> dict[str, Any]:
    """Flatten a verifiers RolloutOutput into the saved-row shape used everywhere else."""
    metrics = _jsonable(_g(o, "metrics")) or {}
    row = {
        "example_id": _g(o, "example_id"),
        "prompt": _jsonable(_g(o, "prompt")),
        "completion": _strip_reasoning(_jsonable(_g(o, "completion"))),
        "answer": _jsonable(_g(o, "answer")),
        "info": _jsonable(_g(o, "info")) or {},
        "reward": _g(o, "reward"),
        "error": _jsonable(_g(o, "error")),
        "is_truncated": bool(_g(o, "is_truncated", False)),
        "stop_condition": _g(o, "stop_condition"),
        "metrics": metrics,
    }
    for k in ("rlm_iterations", "rlm_sub_llm_calls", "rlm_has_final_answer", "rlm_repl_calls"):
        row[k] = metrics.get(k)
    row["sub_calls"] = _jsonable(_g(o, "rlm_sub_llm_log")) or []
    row["final_answer"] = _jsonable(_g(o, "rlm_final_answer"))
    row["rlm_sub_llm_prompt_chars"] = _g(o, "rlm_sub_llm_prompt_chars") or 0
    row["rlm_sub_llm_response_chars"] = _g(o, "rlm_sub_llm_response_chars") or 0
    return row


def _with_messages(base: dict, row: dict, a: argparse.Namespace) -> dict:
    if getattr(a, "keep_messages", False):
        base = dict(base)
        base["messages"] = sft_format.strip_hint(list(row.get("prompt") or [])) + \
            [sft_format._plain(m) for m in (row.get("completion") or [])]
        base["answer_gold"] = row.get("answer")
    return base


def _with_info(row: dict, **kv: Any) -> dict:
    """Copy of a dataset row with extra ``info`` keys (``info`` may be a dict or a JSON string)."""
    row = dict(row)
    info = row.get("info")
    as_str = isinstance(info, str)
    info = dict(json.loads(info) if as_str else (info or {}))
    info.update(kv)
    row["info"] = json.dumps(info) if as_str else info
    return row


def _slim_info(row: dict) -> dict:
    """Drop the giant context from what we persist (keep the question)."""
    info = dict(row.get("info") or {})
    ctx = info.pop("context", None)
    info["context_chars"] = len(ctx) if isinstance(ctx, str) else (len(json.dumps(ctx)) if ctx is not None else 0)
    return info


class TaskWriter:
    def __init__(self, out_dir: Path, task: str):
        self.dir = out_dir / task
        self.dir.mkdir(parents=True, exist_ok=True)
        self.accepted = self.dir / "accepted.jsonl"
        self.rejected = self.dir / "rejected.jsonl"
        self.interesting = self.dir / "interesting.jsonl"
        self.processed = self.dir / "processed.jsonl"
        self.subcalls = self.dir / "subcalls.jsonl"
        self.done: set[str] = set()
        self.n_accepted = 0
        if self.processed.exists():
            for line in self.processed.read_text().splitlines():
                try:
                    d = json.loads(line)
                    self.done.add(d["key"])
                    self.n_accepted += int(d.get("accepted", 0))
                except Exception:  # noqa: BLE001
                    pass

    def _append(self, path: Path, obj: dict) -> None:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")

    def record(self, key: str, *, accepted: bool, bucket: str, payload: dict, sub_calls: list | None = None) -> None:
        if sub_calls:
            self._append(self.subcalls, {"key": key, "bucket": bucket, "n": len(sub_calls), "sub_calls": sub_calls})
        if bucket == "accepted":
            self._append(self.accepted, payload)
            self.n_accepted += 1
        elif bucket == "interesting":
            self._append(self.interesting, payload)
        else:
            self._append(self.rejected, payload)
        self._append(self.processed, {"key": key, "accepted": int(accepted), "bucket": bucket, "ts": time.time()})
        self.done.add(key)


async def evaluate_rows(pending: list[tuple[str, dict]], st: Strategy, a: argparse.Namespace, judge: Judge | None,
                        out: TaskWriter, stats: dict[str, int], extra: dict[str, dict] | None = None) -> None:
    """Heuristics -> judge -> record, for finished rollouts ``pending = [(key, row)]``."""
    extra = extra or {}

    def _meta(key: str) -> dict:
        return (extra.get(key) or {}).get("meta") or {}

    def _needles(key: str) -> tuple:
        return tuple((extra.get(key) or {}).get("leak_needles") or ())

    judge_jobs = []
    for key, row in pending:
        if a.rubric == "reverify":
            rp = _meta(key).get("replay") or {}
            row["_reverify_live_from"] = int(rp.get("from_step") or 1)
            row["_reverify_needles"] = _needles(key)
            if judge is not None:
                from .reverify import sanitize_reasoning
                row["_reasoning_rewritten"] = await sanitize_reasoning(
                    judge, row, needles=_needles(key) + ((st.hint_text[:80],) if st.hint_text else ()),
                    live_from=row["_reverify_live_from"])
                stats["reasoning_rewritten_turns"] += row["_reasoning_rewritten"]
        v = filters.check(row, st, rubric=a.rubric)
        base = {"key": key, "example_id": row["example_id"], "reward": row["reward"],
                "stop_condition": row["stop_condition"], "rlm_sub_llm_calls": row["rlm_sub_llm_calls"],
                "rlm_iterations": row["rlm_iterations"], "signature": v.signature.as_dict(),
                "sub_prompt_chars": row.get("rlm_sub_llm_prompt_chars"), "sub_response_chars": row.get("rlm_sub_llm_response_chars"),
                "heuristic_reasons": v.reasons, "question": str((row.get("info") or {}).get("root_prompt") or "")[-400:],
                "final_answer": row.get("final_answer")}
        if _meta(key):
            base.update(_meta(key))
        if v.ok and judge is not None:
            judge_jobs.append(("accept", key, row, v, base))
        elif v.ok:
            stats["accepted"] += 1
            out.record(key, accepted=True, bucket="accepted", sub_calls=row.get("sub_calls"),
                       payload=sft_format.to_sft_sample(row, strategy_name=st.name, hint=st.hint_text,
                                                        extra_meta={"signature": base["signature"], "info": _slim_info(row), "key": key,
                                                                    "final_answer": row.get("final_answer"), **_meta(key)},
                                                        extra_needles=_needles(key)))
        elif a.save_interesting and v.interesting and judge is not None:
            judge_jobs.append(("interesting", key, row, v, base))
        elif a.judge_all and judge is not None and row.get("rlm_has_final_answer"):
            judge_jobs.append(("verdict_only", key, row, v, base))
        else:
            stats["rejected_heuristic"] += 1
            out.record(key, accepted=False, bucket="rejected", payload=_with_messages(base, row, a), sub_calls=row.get("sub_calls"))
    if judge_jobs:
        results = await asyncio.gather(*[
            judge.judge(row, expected_families=st.families, hint=st.hint_text, run_derived=(kind in ("accept", "verdict_only")),
                        st=st, sig=v.signature, rubric=a.rubric)
            for kind, _, row, _, _ in judge_jobs])
        for (kind, key, row, v, base), jd in zip(judge_jobs, results):
            base["judge"] = jd
            if kind == "accept":
                ok, why = judge_accept(jd, min_quality=a.min_quality, require_derived=not a.no_derived_check,
                                           n_sub_calls=int(row.get("rlm_sub_llm_calls") or 0), min_sub_calls=st.min_sub_calls,
                                           strict=a.strict, context_causal=st.context_causal, rubric=a.rubric,
                                           python_exact=bool(getattr(st, "python_exact", False)))
                if ok:
                    try:
                        payload = sft_format.to_sft_sample(row, strategy_name=st.name, hint=st.hint_text, judge=jd,
                                                           extra_meta={"signature": base["signature"], "info": _slim_info(row), "key": key,
                                                                    "final_answer": row.get("final_answer"), **_meta(key)},
                                                           extra_needles=_needles(key))
                    except sft_format.HintLeak as e:
                        stats["hint_leak"] += 1
                        base["heuristic_reasons"] = [f"hint leak: {e}"]
                        out.record(key, accepted=False, bucket="rejected", payload=base)
                        continue
                    if a.rubric == "reverify":
                        if (payload.get("meta") or {}).get("reasoning_hint_leak"):
                            stats["reasoning_leak"] += 1
                            base["heuristic_reasons"] = ["reverify: reasoning refers to the hint/note"]
                            out.record(key, accepted=False, bucket="rejected", payload=_with_messages(base, row, a),
                                       sub_calls=row.get("sub_calls"))
                            continue
                        from .reverify import reverify_meta
                        rp = _meta(key).get("replay") or {}
                        payload.setdefault("meta", {})["reverify"] = {
                            **reverify_meta(jd), "source": f"branch-{rp['mode']}" if rp.get("mode") else "generated",
                            "live_from": int(rp.get("from_step") or 1), "note_used": bool(rp.get("hint")),
                            "reasoning_rewritten_turns": int(row.get("_reasoning_rewritten") or 0)}
                    stats["accepted"] += 1
                    out.record(key, accepted=True, bucket="accepted", payload=payload, sub_calls=row.get("sub_calls"))
                else:
                    stats["rejected_judge"] += 1
                    base["judge_reason"] = why
                    out.record(key, accepted=False, bucket="rejected", payload=_with_messages(base, row, a), sub_calls=row.get("sub_calls"))
            elif kind == "verdict_only":
                stats["rejected_heuristic"] += 1
                out.record(key, accepted=False, bucket="rejected", payload=_with_messages(base, row, a), sub_calls=row.get("sub_calls"))
            else:
                if judge_interesting(jd, min_quality=a.interesting_min_quality):
                    stats["interesting"] += 1
                    payload = dict(base)
                    payload["messages"] = sft_format.strip_hint(list(row.get("prompt") or [])) + \
                        [sft_format._plain(m) for m in (row.get("completion") or [])]
                    payload["answer_gold"] = row.get("answer")
                    out.record(key, accepted=False, bucket="interesting", payload=payload, sub_calls=row.get("sub_calls"))
                else:
                    stats["rejected_heuristic"] += 1
                    out.record(key, accepted=False, bucket="rejected", payload=_with_messages(base, row, a), sub_calls=row.get("sub_calls"))


class PortLBTransport:
    """httpx transport that spreads localhost requests over several vLLM ports on the same node (RLM_LB_PORTS)."""

    def __init__(self, ports: list[int]):
        import collections

        import httpx
        self.ports = ports
        self.host = os.environ.get("RLM_LB_HOST") or "localhost"
        self.inner = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=2000, max_keepalive_connections=400))
        self.sticky: collections.OrderedDict[str, int] = collections.OrderedDict()
        self.load: dict[int, float] = {p: 0.0 for p in ports}
        self.load_at = 0.0
        self.mine: dict[int, int] = {p: 0 for p in ports}
        self.metrics_client = None
        self.lock = None

    def _key(self, body: bytes) -> str | None:
        try:
            msgs = json.loads(body).get("messages") or []
            return hashlib.md5(json.dumps(msgs[:2], sort_keys=True)[:4000].encode()).hexdigest()
        except Exception:  # noqa: BLE001
            return None

    async def _refresh(self) -> None:
        import asyncio

        import httpx
        if self.lock is None:
            self.lock = asyncio.Lock()
        async with self.lock:
            if time.time() - self.load_at < 5:
                return
            await self._fetch(httpx)
            self.load_at = time.time()

    async def _fetch(self, httpx) -> None:  # noqa: ANN001
        if self.metrics_client is None:
            self.metrics_client = httpx.AsyncClient(timeout=3.0)
        for p in self.ports:
            try:
                text = (await self.metrics_client.get(f"http://{self.host}:{p}/metrics")).text
                tot = 0.0
                for line in text.splitlines():
                    if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
                        tot += float(line.rsplit(" ", 1)[-1])
                self.load[p] = tot
            except Exception:  # noqa: BLE001
                pass

    async def _port(self, body: bytes) -> int | None:
        key = self._key(body)
        if key is None:
            return None
        if key in self.sticky:
            self.sticky.move_to_end(key)
            return self.sticky[key]
        await self._refresh()
        port = min(self.ports, key=lambda p: (self.load[p] + self.mine[p], p))
        self.load[port] += 1
        self.sticky[key] = port
        if len(self.sticky) > 20000:
            self.sticky.popitem(last=False)
        return port

    async def handle_async_request(self, request):  # noqa: ANN001, ANN201
        port = None
        if request.url.host in ("localhost", "127.0.0.1", self.host) and request.url.path.endswith("/chat/completions"):
            try:
                port = await self._port(request.content or b"")
            except Exception:  # noqa: BLE001
                port = None
            if port is not None:
                request.url = request.url.copy_with(port=port)
                self.mine[port] += 1
        try:
            return await self.inner.handle_async_request(request)
        finally:
            if port is not None:
                self.mine[port] -= 1

    async def aclose(self) -> None:
        if self.metrics_client is not None:
            await self.metrics_client.aclose()
        await self.inner.aclose()


def _lb_http_client(a: argparse.Namespace):
    ports = [int(p) for p in os.environ.get("RLM_LB_PORTS", "").split(",") if p.strip()]
    if len(ports) < 2 or not any(h in a.base_url for h in ("localhost", "127.0.0.1")):
        return None
    import httpx

    class _T(PortLBTransport, httpx.AsyncBaseTransport):
        pass
    return httpx.AsyncClient(transport=_T(ports), timeout=a.request_timeout)


def make_client(a: argparse.Namespace):
    """(verifiers client, root sampling args) for the generator's OpenAI-compatible server."""
    from openai import AsyncOpenAI
    from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient

    headers = {}
    team = os.environ.get("PRIME_API_TEAM")
    if team and "pinference" in a.base_url:
        headers["X-Prime-Team-ID"] = team
    api_key = a.api_key if a.api_key != "EMPTY" or "pinference" not in a.base_url else os.environ.get("PRIME_API_KEY", "EMPTY")
    http_client = _lb_http_client(a)
    client = OpenAIChatCompletionsClient(AsyncOpenAI(base_url=a.base_url, api_key=api_key,
                                                     timeout=a.request_timeout, max_retries=2,
                                                     default_headers=headers or None,
                                                     **({"http_client": http_client} if http_client else {})))
    sampling = {"max_tokens": a.max_completion_tokens, "temperature": a.temperature,
                "extra_body": {"chat_template_kwargs": json.loads(a.chat_template_kwargs)}}
    return client, sampling


def restore_context(row: dict, inp: Any) -> dict:
    """Put the real context back into info (cleanup replaced it with a stub) so filters and the judge can see it."""
    inp_info = _g(inp, "info") or {}
    if isinstance(inp_info, str):
        try:
            inp_info = json.loads(inp_info)
        except json.JSONDecodeError:
            inp_info = {}
    real_ctx = inp_info.get("context") if isinstance(inp_info, dict) else None
    if real_ctx is not None:
        row["info"] = {**(row.get("info") or {}), "context": real_ctx}
    return row


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """The argument namespace evaluate_rows / make_client read (filled from the spec by pipeline.worker.gen_args)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=add_help)
    ap.add_argument("--config", type=Path, required=True, help="env-stream TOML the stream list/args come from")
    ap.add_argument("--section", default="train", choices=["train", "eval", "stage2"])
    ap.add_argument("--rubric", default="reverify", choices=["reverify"])
    ap.add_argument("--judge-all", action="store_true",
                    help="judge every rollout that submitted, including heuristic rejects (verdicts are recorded, not used to accept)")
    ap.add_argument("--strict", action="store_true",
                    help="stricter acceptance: judge gates on size_checked / search_spam / causal_ok / concise_subprompts")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--model", default=None, help="served model name on the vLLM server")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-tokens", type=int, default=4096)
    ap.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}',
                    help="JSON passed as extra_body.chat_template_kwargs on every root request (default: thinking off)")
    ap.add_argument("--max-concurrent", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=32, help="rollouts per generate() call (flush granularity)")
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--hint-mode", default="note", choices=["note"])
    ap.add_argument("--judge-model", default=None, help="default $RLM_JUDGE_MODEL")
    ap.add_argument("--judge-concurrency", type=int, default=8)
    ap.add_argument("--min-quality", type=int, default=4)
    ap.add_argument("--no-derived-check", action="store_true", help="skip the DERIVED/LUCKY rubric")
    ap.add_argument("--save-interesting", action="store_true",
                    help="judge low-reward rollouts with an unexpected decomposition; keep NOVEL ones")
    ap.add_argument("--interesting-min-quality", type=int, default=3)
    ap.add_argument("--keep-messages", action="store_true",
                    help="also store the (hint-stripped) messages of REJECTED rollouts, for inspection/report rendering")
    return ap
