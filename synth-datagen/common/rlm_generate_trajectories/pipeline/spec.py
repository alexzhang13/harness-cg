"""Pipeline spec: one TOML file naming everything a run depends on. Relative paths resolve against the repo root."""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
PYTHON = sys.executable


GEN_WRITERS = ("gen", "gen2", "gen3", "gen4")

def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


@dataclass
class EnvConfig:
    path: Path
    section: str = "train"
    streams: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class Server:
    port: int
    slots: int = 3
    host: str = "localhost"

    @property
    def id(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def local_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class Source:
    dir: Path
    tag: str
    era: str = "v2"
    note: str = ""


@dataclass
class Spec:
    path: Path
    name: str
    dir: Path
    python: str
    configs: list[EnvConfig]
    env_overrides: dict[str, dict[str, Any]]
    env_tiers: dict[str, dict[str, Any]] = field(default_factory=dict)
    distinct_tasks: int = 10
    examples_per_task: int = 1
    max_examples_per_task: int = 3
    good_examples: int = 0
    require_sub_calls: int = 0
    max_examples_per_env: int = 0
    count_by: str = "meta"
    model: str = "qwen38-27b"
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: {"enable_thinking": True})
    temperature: float = 0.8
    max_completion_tokens: int = 8192
    max_concurrent: int = 12
    chunk_size: int = 12
    request_timeout: float = 900.0
    sub_no_think: bool = False
    hint_mode: str = "note"
    servers: list[Server] = field(default_factory=list)
    judge_model: str = "openai/gpt-5.6-sol"
    judge_concurrency: int = 6
    rubric: str = "reverify"
    judge_all: bool = True
    min_quality: int = 4
    load_num_examples: int = -1
    tasks_per_item: int = 8
    live_snapshots: bool = False
    gen_writers: int = 1
    rollouts_per_task: int = 1
    max_attempts_per_task: int = 4
    seed: int = 0
    priority: int = 0
    sub_think: bool = False
    env_vars: dict[str, Any] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    source_globs: list[dict[str, Any]] = field(default_factory=list)
    export_max_per_task: int = 3
    export_max_per_context: int = 0
    export_max_per_env: int = 0
    export_holdout: float = 0.0
    export_seed: int = 0
    export_sources: list[str] = field(default_factory=lambda: ["generated"])
    env_args: dict[str, Any] = field(default_factory=dict)
    poll_s: float = 30.0
    max_node_items: int = 4
    stop_when_done: bool = True

    @property
    def pipeline_dir(self) -> Path:
        return self.dir

    @property
    def db_path(self) -> Path:
        return self.dir / "state.sqlite"

    def writer_dir(self, writer: str) -> Path:
        """gen / branch: generator-layout output dirs (<dir>/<env>/{accepted,rejected,processed,subcalls}.jsonl)."""
        return self.dir / writer

    def env_param(self, env: str, key: str) -> Any:
        ov = self.env_overrides.get(env) or {}
        if key in ov:
            return ov[key]
        for tier in self.env_tiers.values():
            if env in tier.get("envs", ()) and key in tier:
                return tier[key]
        return getattr(self, key)

    def env_tier(self, env: str) -> str:
        return next((name for name, t in self.env_tiers.items() if env in t.get("envs", ())), "")

    def env_target(self, env: str) -> int:
        return int(self.env_param(env, "distinct_tasks"))

    def all_sources(self) -> list[Source]:
        """Spec sources, dirs matching [[source_globs]] (re-expanded each call) and the pipeline's writer dirs."""
        out = list(self.sources)
        seen = {str(s.dir.resolve()) for s in out}
        for g in self.source_globs:
            for d in sorted(REPO.glob(g["pattern"])):
                if d.is_dir() and str(d.resolve()) not in seen:
                    out.append(Source(d, g.get("tag_prefix", "g") + d.name, g.get("era", "v2"), "glob " + g["pattern"]))
                    seen.add(str(d.resolve()))
        return out + [Source(self.writer_dir("gen"), "gen", "v2", "pipeline generation"),
                      *[Source(self.writer_dir(w), w, "v2", "pipeline generation (extra writer)") for w in GEN_WRITERS[1:]]]

    def source_by_dir(self) -> dict[str, Source]:
        return {str(s.dir.resolve()): s for s in self.all_sources()}

    def fingerprint(self) -> str:
        d = asdict(self)
        d.pop("path", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def envs(self) -> dict[str, tuple[EnvConfig, dict]]:
        """{stream: (config, raw toml stream entry)} across every config, in config order (first wins)."""
        out: dict[str, tuple[EnvConfig, dict]] = {}
        for c in self.configs:
            cfg = tomllib.load(open(c.path, "rb"))
            for s in cfg["orchestrator"][c.section]["env"]:
                n = s["name"]
                if (c.streams and n not in c.streams) or n in c.exclude or n in out:
                    continue
                out[n] = (c, s)
        return out

    def strategies(self) -> dict[str, Any]:
        """{stream: Strategy} with the spec's hint mode (hint text = the stream's note + footer)."""
        from ..strategies import build_strategies
        out: dict[str, Any] = {}
        envs = self.envs()
        for c in self.configs:
            reg = build_strategies(c.path, c.section, None, self.hint_mode)
            for n, st in reg.items():
                if n in envs and envs[n][0] is c and n not in out:
                    glob = {k: v for k, v in self.env_args.items() if not isinstance(v, dict)}
                    st.env_args.update(glob)
                    st.env_args.update(self.env_args.get(n) if isinstance(self.env_args.get(n), dict) else {})
                    out[n] = st
        return out


def load(path: str | Path) -> Spec:
    path = _abs(path)
    raw = tomllib.load(open(path, "rb"))
    p = raw.get("pipeline", {})
    t = raw.get("targets", {})
    te = raw.get("teacher", {})
    j = raw.get("judge", {})
    g = raw.get("generate", {})
    e = raw.get("export", {})
    o = raw.get("orchestrator", {})
    envs = raw.get("envs", {})
    configs = [EnvConfig(_abs(c["path"]), c.get("section", "train"), list(c.get("streams", [])), list(c.get("exclude", [])))
               for c in envs.get("configs", [])]
    if not configs:
        raise ValueError(f"{path}: [[envs.configs]] is empty")
    servers = []
    for s in te.get("servers", []):
        for port in s.get("ports", [s.get("port", 8000)]):
            servers.append(Server(int(port), int(s.get("slots_per_port", s.get("slots", 3))),
                                  str(s.get("host", "localhost"))))
    sources = [Source(_abs(s["dir"]), s["tag"], s.get("era", "v2"), s.get("note", "")) for s in raw.get("sources", [])]
    kw_globs = list(raw.get("source_globs", []))
    tags = [s.tag for s in sources]
    if len(set(tags)) != len(tags) or set(GEN_WRITERS) & set(tags):
        raise ValueError(f"{path}: source tags must be unique and not 'gen*': {tags}")
    kw: dict[str, Any] = {}

    def take(section: dict, mapping: dict[str, str]) -> None:
        for src, dst in mapping.items():
            if src in section:
                kw[dst] = section[src]

    take(t, {"distinct_tasks": "distinct_tasks", "examples_per_task": "examples_per_task",
             "max_examples_per_task": "max_examples_per_task", "count_by": "count_by",
             "good_examples": "good_examples", "max_examples_per_env": "max_examples_per_env",
             "require_sub_calls": "require_sub_calls"})
    take(te, {"model": "model", "chat_template_kwargs": "chat_template_kwargs", "temperature": "temperature",
              "max_completion_tokens": "max_completion_tokens", "max_concurrent": "max_concurrent", "sub_no_think": "sub_no_think",
              "chunk_size": "chunk_size", "request_timeout": "request_timeout", "hint_mode": "hint_mode"})
    take(j, {"model": "judge_model", "concurrency": "judge_concurrency", "rubric": "rubric", "judge_all": "judge_all",
             "min_quality": "min_quality"})
    take(g, {"load_num_examples": "load_num_examples", "tasks_per_item": "tasks_per_item",
             "rollouts_per_task": "rollouts_per_task", "max_attempts_per_task": "max_attempts_per_task", "seed": "seed"})
    take(e, {"max_per_task": "export_max_per_task", "max_per_context": "export_max_per_context",
             "max_per_env": "export_max_per_env", "holdout": "export_holdout", "seed": "export_seed",
             "sources": "export_sources"})
    take(o, {"poll_s": "poll_s", "stop_when_done": "stop_when_done", "max_node_items": "max_node_items"})
    spec = Spec(path=path, name=p.get("name", path.stem), dir=_abs(p.get("dir", f"outputs/sft_pipeline/{path.stem}")),
                python=p.get("python", PYTHON), configs=configs, env_overrides=dict(envs.get("overrides", {})), env_tiers=dict(envs.get("tiers", {})),
                servers=servers, sources=sources, source_globs=kw_globs, env_args=dict(envs.get("env_args", {})), **kw)
    if spec.count_by not in ("meta", "task"):
        raise ValueError(f"count_by must be meta|task, got {spec.count_by!r}")
    if raw.get("branch", {}).get("modes"):
        raise ValueError("[branch] modes: prefix-replay branching is not part of this port")
    return spec
