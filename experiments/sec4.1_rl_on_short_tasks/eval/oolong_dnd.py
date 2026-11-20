"""OOLONG-Real D&D held-out eval (577 questions over ten 3-episode context windows), resumable.

    python oolong_dnd.py --out <dir> --base-url <url>/v1 --model <served name> \
        --root-max-tokens 4096 --reader-max-tokens 4096 --chat-template-kwargs '{"enable_thinking": false}' \
        --timeout 7200 --concurrency 48

Questions and windows are fixed in oolong_dnd_3ep_577_ids.json (checked by SHA-256). Each question is a one-row
RLM env (40 turns, T 0, <= 600 sub-calls). Writes rows.jsonl (errored rows are rerun on resume) and traces/.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

HERE = Path(__file__).resolve().parent
IDS_FILE = HERE / "oolong_dnd_3ep_577_ids.json"


def plain(x):
    if hasattr(x, "model_dump"):
        return x.model_dump(mode="json")
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def data_path(a) -> Path:
    if a.data is not None:
        return a.data
    from huggingface_hub import hf_hub_download
    frozen = json.loads(a.ids_file.read_text())
    return Path(hf_hub_download(frozen["dataset"], f"{frozen['config']}/{frozen['split']}.jsonl",
                                repo_type="dataset", revision=frozen["revision"]))


def prepare(a, out):
    """Rebuild the fixed rows and context windows from the upstream file."""
    manifest = out / "manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    from huggingface_hub import hf_hub_download

    frozen = json.loads(a.ids_file.read_text())
    want_rows = {rid for rid, _ in frozen["rows"]}
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(hf_hub_download(a.tokenizer, "tokenizer.json"))
        measure = f"{a.tokenizer} tokens, add_special_tokens=False"
    except ImportError:
        # keep the recorded token counts
        tok, measure = None, "frozen Qwen3.8-27B token counts (tokenizers not installed)"
        print(json.dumps({"phase": "index", "warning": "tokenizers not installed; using the frozen token counts"}))
    windows, rows = {}, {}
    with data_path(a).open() as source:
        for line in source:
            r = json.loads(line)
            wid = r["context_window_id"]
            if wid in frozen["windows"] and wid not in windows:
                context = r["context_window_text"]
                sha = hashlib.sha256(context.encode()).hexdigest()
                if sha != frozen["windows"][wid]["sha256"]:
                    raise ValueError(f"context window {wid}: SHA-256 differs from the frozen selection")
                tokens = (len(tok.encode(context, add_special_tokens=False).ids) if tok is not None
                          else frozen["windows"][wid]["tokens"])
                if tokens != frozen["windows"][wid]["tokens"]:
                    print(json.dumps({"phase": "index", "warning": "token count differs from the frozen count",
                                      "id": wid, "tokens": tokens, "frozen": frozen["windows"][wid]["tokens"]}))
                windows[wid] = {"id": wid, "tokens": tokens, "chars": len(context), "episodes": r["episodes"],
                                "eligible": True, "sha256": sha}
                (out / "contexts" / f"{wid}.txt").write_text(context)
                print(json.dumps({"phase": "index", **windows[wid]}), flush=True)
            if r["id"] in want_rows:
                rows[r["id"]] = {k: v for k, v in r.items() if k != "context_window_text"}
    missing = [rid for rid, _ in frozen["rows"] if rid not in rows] + [w for w in frozen["windows"] if w not in windows]
    if missing:
        raise ValueError(f"{len(missing)} frozen rows / windows not found in the data file, e.g. {missing[:3]}")
    result = {"dataset": frozen["dataset"], "revision": frozen["revision"], "config": frozen["config"],
              "split": frozen["split"], "length_measure": measure,
              "windows": windows, "rows": [rows[rid] for rid, _ in frozen["rows"]]}
    write_json(manifest, result)
    return result


async def evaluate(a, out, manifest):
    import httpx
    from openai import AsyncOpenAI
    from verifiers import OpenAIChatCompletionsClient
    from oolong_real_synth.real_task import load_environment

    class Meter(OpenAIChatCompletionsClient):
        """Records every model call (root calls have a system message, sub-calls do not)."""
        def __init__(self, native):
            super().__init__(native)
            self.calls = []

        async def get_response(self, prompt, model, sampling_args, tools=None, **kwargs):
            root = any(getattr(m, "role", None) == "system" for m in prompt)
            begin = time.monotonic()
            rec = {"root": root, "model": model}
            try:
                result = await super().get_response(prompt, model, sampling_args, tools, **kwargs)
                rec["response"] = plain(getattr(result, "message", None))
                rec["usage"] = plain(getattr(getattr(result, "response", None), "usage", None))
                if not root:
                    rec["prompt"] = plain(prompt)
                return result
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                rec["seconds"] = time.monotonic() - begin
                self.calls.append(rec)

    selected = manifest["rows"][:a.limit] if a.limit > 0 else manifest["rows"]
    records_path = out / "rows.jsonl"
    records = {}
    if records_path.exists():
        for line in records_path.read_text().splitlines():
            r = json.loads(line)
            records[r["id"]] = r
    pending = [r for r in selected if r["id"] not in records or records[r["id"]].get("error")]
    semaphore = asyncio.Semaphore(a.concurrency)
    ctk = json.loads(a.chat_template_kwargs)

    def status():
        completed = [records[r["id"]] for r in selected if r["id"] in records]
        errors = sum(r.get("error") is not None for r in completed)
        summary = {"total": len(selected), "completed": len(completed), "errors": errors,
                   "mean_reward": statistics.mean(r["reward"] for r in completed) if completed else None,
                   "full_reward": sum(r["reward"] == 1 for r in completed),
                   "submitted": sum(r.get("final") is not None for r in completed),
                   "by_episodes": {str(n): {"n": len(rs), "mean_reward": statistics.mean(r["reward"] for r in rs)}
                                   for n in sorted({len(r["episodes"]) for r in completed})
                                   if (rs := [r for r in completed if len(r["episodes"]) == n])}}
        write_json(out / "status.json", summary)
        return summary

    status()
    print(json.dumps({"phase": "evaluate", "pending": len(pending), "total": len(selected)}), flush=True)
    native = AsyncOpenAI(base_url=a.base_url, api_key="EMPTY", timeout=httpx.Timeout(600), max_retries=2)

    async def one(row):
        async with semaphore:
            started = time.monotonic()
            context = (out / "contexts" / f"{row['context_window_id']}.txt").read_text()
            env = load_environment(rows=[row | {"context_window_text": context}], max_iterations=40,
                                   max_sub_calls=600, sub_max_tokens=a.reader_max_tokens,
                                   sub_extra_body={"chat_template_kwargs": ctk},
                                   default_answer_at_limit=False)
            meter = Meter(native)
            rec = dict(row)
            try:
                state = await asyncio.wait_for(env._run_rollout_state(
                    input=dict(env.dataset[0]), client=meter, model=a.model,
                    sampling_args={"max_tokens": a.root_max_tokens, "temperature": 0.0,
                                   "extra_body": {"chat_template_kwargs": ctk}}), a.timeout)
                error = state.get("error")
                rec.update(reward=float(state.get("reward") or 0), final=state.get("rlm_final_answer"),
                           error=f"{type(error).__name__}: {error}" if error is not None else None,
                           stop_condition=state.get("stop_condition"),
                           iterations=state.get("rlm_iterations"), sub_calls=state.get("rlm_sub_llm_calls", 0),
                           metrics=plain(state.get("metrics") or {}))
                write_json(out / "traces" / f"{row['id']}.json", {
                    "history": plain(state.get("rlm_history") or []),
                    "sub_calls": plain(state.get("rlm_sub_llm_log") or []), "calls": meter.calls,
                    "trajectory": plain(state.get("trajectory") or [])})
            except Exception as exc:
                rec.update(reward=0.0, error=f"{type(exc).__name__}: {exc}")
                write_json(out / "traces" / f"{row['id']}.json", {"calls": meter.calls, "error": rec["error"]})
            finally:
                await env.teardown_rlm()
            rec.update(seconds=time.monotonic() - started, model=a.model,
                       context_tokens=manifest["windows"][row["context_window_id"]]["tokens"])
            with records_path.open("a") as output:
                output.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records[row["id"]] = rec
            print(json.dumps({"phase": "result", **rec, "progress": status()}), flush=True)

    try:
        await asyncio.gather(*(one(row) for row in pending))
    finally:
        await native.close()
    print(json.dumps({"phase": "finished", **status()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=None, help="local dnd/test.jsonl (default: HF download)")
    p.add_argument("--ids-file", type=Path, default=IDS_FILE)
    p.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B", help="HF tokenizer for the window token counts")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base-url", default="http://127.0.0.1:18431/v1")
    p.add_argument("--model", default="qwen38-27b")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--root-max-tokens", type=int, default=4096)
    p.add_argument("--reader-max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=7200, help="per-question wall clock (s)")
    p.add_argument("--limit", type=int, default=0, help="0 means all frozen questions")
    p.add_argument("--prepare-only", action="store_true", help="only build manifest.json and contexts/")
    p.add_argument("--chat-template-kwargs", default='{"enable_thinking": false}',
                   help="root AND reader chat_template_kwargs")
    a = p.parse_args()
    # sub-calls use the root's model and server, so no RLM_TRAIN_SUB_* overrides
    overrides = [k for k in os.environ if k.startswith("RLM_TRAIN_SUB_")]
    if overrides:
        raise RuntimeError(f"Unset foreign submodel overrides before running: {overrides}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    for directory in ["contexts", "traces"]:
        (out / directory).mkdir(exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()
              if k not in {"limit", "prepare_only", "concurrency", "data", "ids_file"}}
    config.update(dataset="oolongbench/oolong-real", dataset_config="dnd", split="test",
                  root_mode="unhinted RLM", sub_model="same as root", max_iterations=40,
                  max_completion_tokens=a.root_max_tokens, temperature=0.0,
                  chat_template_kwargs=json.loads(a.chat_template_kwargs), max_sub_calls=600,
                  default_answer_at_limit=False)
    frozen = out / "config.json"
    if frozen.exists() and json.loads(frozen.read_text()) != config:
        raise ValueError("Configuration changed; use a fresh output directory")
    write_json(frozen, config)
    manifest = prepare(a, out)
    if not a.prepare_only:
        asyncio.run(evaluate(a, out, manifest))


if __name__ == "__main__":
    main()
