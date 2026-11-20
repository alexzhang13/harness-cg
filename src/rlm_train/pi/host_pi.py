"""HostPi: verifiers' vf.Pi harness run as a host subprocess instead of in a container.

The interception endpoint is reached on localhost, node and pi come from PATH (install_node_pi.sh), the task is the
prompt and the context is context.txt in the sandbox. The turncap, readcap, compaction and subagent extensions are
loaded with -e; settings come from the PI_* environment variables.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

import verifiers as vf
from verifiers.v1.packages.harnesses.pi import (
    pi_endpoint_config,
    pi_models_json,
)
from verifiers.v1.utils.endpoint_utils import Endpoint

EXTENSIONS_DIR = Path(__file__).resolve().parent / "extensions"

# Sandbox paths (HOME is the sandbox root).
MODELS_JSON_PATH = "/.pi/agent/models.json"      # provider config
AGENTS_DIR_PATH = "/.pi/agent/agents"            # sub-agent definitions (*.md)
SETTINGS_JSON_PATH = "/.pi/agent/settings.json"  # written only when compaction is on
OUTPUT_FORMAT_PATH = "pi/output_format.txt"      # answer-format footer restored after a compaction

DEFAULT_PI_CONTEXT_WINDOW = 65536       # set PI_CONTEXT_WINDOW to the server's max_model_len
DEFAULT_PI_COMPACT_KEEP_RECENT = 8192   # tokens of recent history kept verbatim by a compaction


def compaction_config() -> dict[str, int] | None:
    """Compaction settings, or None if PI_COMPACT_AT_TOKENS is unset. pi compacts above contextWindow - reserveTokens."""
    raw = (os.environ.get("PI_COMPACT_AT_TOKENS") or "").strip()
    if not raw:
        return None
    trigger = int(raw)
    window = int(os.environ.get("PI_CONTEXT_WINDOW") or DEFAULT_PI_CONTEXT_WINDOW)
    keep_recent = int(
        os.environ.get("PI_COMPACT_KEEP_RECENT_TOKENS") or DEFAULT_PI_COMPACT_KEEP_RECENT
    )
    reserve = window - trigger
    if reserve <= 0:
        raise ValueError(
            f"PI_COMPACT_AT_TOKENS={trigger} must be strictly less than "
            f"PI_CONTEXT_WINDOW={window} (pi compacts at window - reserveTokens, "
            "so a trigger at or above the window can never fire)."
        )
    if keep_recent >= trigger:
        raise ValueError(
            f"PI_COMPACT_KEEP_RECENT_TOKENS={keep_recent} must be smaller than "
            f"PI_COMPACT_AT_TOKENS={trigger}, else a compaction reclaims nothing "
            "and pi re-compacts on every turn."
        )
    return {
        "context_window": window,
        "trigger": trigger,
        "reserve_tokens": reserve,
        "keep_recent_tokens": keep_recent,
    }


# Sub-agents use a second models.json entry that sends this header, to tell their turns from root turns.
PI_SUBAGENT_MODEL_ID = "model-subagent"
PI_SUBAGENT_HEADER = "x-pi-subagent"

ROOT_STEPS_KEY = "pi_root_steps"
SUBAGENT_STEPS_KEY = "pi_subagent_steps"


def _is_subagent_request(headers: Mapping[str, object]) -> bool:
    """True if the request came from a sub-agent."""
    for key, value in headers.items():
        if str(key).lower() == PI_SUBAGENT_HEADER:
            return str(value).strip().lower() not in {"", "0", "false", "no"}
    return False


def _bump(state: object, key: str) -> None:
    if not isinstance(state, dict):
        return
    state[key] = int(state.get(key) or 0) + 1


def keep_only_root_pi_steps(
    step: object, state: object, headers: Mapping[str, object]
) -> bool:
    """keep_trajectory_step filter: root turns only."""
    _ = step
    if _is_subagent_request(headers):
        _bump(state, SUBAGENT_STEPS_KEY)
        return False
    _bump(state, ROOT_STEPS_KEY)
    return True


def count_all_pi_steps(
    step: object, state: object, headers: Mapping[str, object]
) -> bool:
    """Keep every step, counting root and sub-agent steps."""
    _ = step
    if _is_subagent_request(headers):
        _bump(state, SUBAGENT_STEPS_KEY)
    else:
        _bump(state, ROOT_STEPS_KEY)
    return True


ROLLOUT_TIMEOUT_FACTOR = 2


def rollout_timeout_seconds(command_timeout: int) -> float | None:
    """Rollout wall-clock cap in seconds, or None."""
    raw = (os.environ.get("PI_ROLLOUT_TIMEOUT") or "").strip()
    if raw:
        seconds = float(raw)
        return None if seconds <= 0 else seconds
    return float(command_timeout * ROLLOUT_TIMEOUT_FACTOR)


def _models_json_file(state: object) -> str:
    """models.json pointing pi at the interception endpoint (resolved per rollout, after it is registered)."""
    text = pi_models_json(pi_endpoint_config(state))
    cfg = compaction_config()
    if cfg is None:
        return text
    data = json.loads(text)
    try:
        models = data["providers"]["verifiers"]["models"]
        if not models:
            raise KeyError("models")
        for model in models:
            model["contextWindow"] = cfg["context_window"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "HostPi: cannot set contextWindow — verifiers' pi_models_json no "
            f"longer has providers.verifiers.models ({exc!r}). Without it pi "
            "assumes a 128000-token window and auto-compaction never fires."
        ) from exc
    return json.dumps(data, indent=2)


def _settings_json_file(state: object) -> str:
    """pi's settings.json with the compaction settings only."""
    _ = state
    cfg = compaction_config()
    if cfg is None:
        return "{}"
    return json.dumps(
        {
            "compaction": {
                "enabled": True,
                "reserveTokens": cfg["reserve_tokens"],
                "keepRecentTokens": cfg["keep_recent_tokens"],
            }
        },
        indent=2,
    )
def _marked_models_json_file(state: object) -> str:
    """models.json plus the marked sub-agent model entry."""
    config = json.loads(_models_json_file(state))
    models = config["providers"]["verifiers"]["models"]
    root = dict(models[0])
    models.append(
        {
            **root,
            "id": PI_SUBAGENT_MODEL_ID,
            "headers": {PI_SUBAGENT_HEADER: "1"},
        }
    )
    return json.dumps(config, indent=2)


def _output_format_file(task: object, state: object) -> str:
    """The task's output-format footer, restored by the compaction extension after a summary (may be empty)."""
    _ = state
    if isinstance(task, dict):
        return str(task.get("answer_format") or "")
    return str(getattr(task, "answer_format", "") or "")


def _deployed_ext(name: str) -> str:
    """Path of a deployed extension, or "" if pi is not found."""
    pi_bin = shutil.which("pi")
    if not pi_bin:
        return ""
    for parent in Path(pi_bin).resolve().parents:
        if (parent / "package.json").is_file() and (parent / "examples").is_dir():
            return str(parent / "examples" / "extensions" / name / "index.ts")
    return ""


def _deployed_subagent_ext() -> str:
    """Deployed sub-agent extension."""
    return _deployed_ext("subagent-rlm")


def _deployed_compaction_ext() -> str:
    """Deployed compaction extension."""
    return _deployed_ext("compaction-rlm")


def _static_file(text: str):
    """Fixed file content (sub-agent definitions)."""

    def _value(state: object) -> str:
        _ = state
        return text

    return _value


def _context_file(task: object, state: object) -> str:
    """The task's context, written to context.txt in the sandbox ("" if none)."""
    _ = state
    if isinstance(task, dict):
        return str(task.get("context") or "")
    return str(getattr(task, "context", "") or "")


class HostPi(vf.Pi):
    def __init__(
        self,
        *,
        context_path: str | None = "context.txt",
        train_root_only: bool | None = None,
        pi_max_turns: int | None = None,
        **kwargs: object,
    ) -> None:
        # pi_max_turns overrides PI_MAX_TURNS for this env
        self._pi_max_turns = None if pi_max_turns is None else int(pi_max_turns)
        if train_root_only is None:
            train_root_only = str(
                os.environ.get("PI_TRAIN_ROOT_ONLY", "")
            ).strip().lower() in {"1", "true", "yes"}
        self.train_root_only = bool(train_root_only)
        kwargs.setdefault("agent_workdir", ".")
        kwargs.setdefault("instruction_path", "pi/instruction.txt")
        kwargs.setdefault("system_prompt_path", "pi/system.txt")
        kwargs.setdefault("log_path", "logs/pi.txt")
        self._context_path = context_path

        sandbox = kwargs.pop("sandbox", None)
        base_sandbox: dict[str, object] = {
            "workdir": ".",
            "network_access": False,
            "command_timeout": int(os.environ.get("PI_COMMAND_TIMEOUT", "1800")),
        }
        if isinstance(sandbox, dict):
            base_sandbox.update(sandbox)
        kwargs["sandbox"] = base_sandbox
        self._command_timeout = int(base_sandbox["command_timeout"])  # type: ignore[arg-type]
        self._rollout_timeout = rollout_timeout_seconds(self._command_timeout)

        super().__init__(**kwargs)

        self.endpoint = Endpoint(use_tunnel=False)

        program = dict(self.program) if isinstance(self.program, dict) else {}

        # no install step; models.json is written as a program file below
        program["setup"] = []
        program.pop("tools", None)
        program.pop("bindings", None)
        program.pop("channels", None)

        files = dict(program.get("files") or {})
        files[MODELS_JSON_PATH] = _models_json_file
        if self._context_path:
            files[self._context_path] = _context_file
        self._compaction = compaction_config()
        if self._compaction is not None:
            files[SETTINGS_JSON_PATH] = _settings_json_file
            files[OUTPUT_FORMAT_PATH] = _output_format_file
        program["files"] = files

        import shlex

        # readcap loads from this package; compaction and subagent import pi internals, so they load from the
        # copies deployed inside the pi package.
        _exts_dir = EXTENSIONS_DIR
        _readcap = os.environ.get("PI_READCAP_EXT") or str(
            _exts_dir / "readcap" / "index.ts"
        )
        _inject = ""
        _disabled = str(os.environ.get("PI_READCAP_DISABLE", "")).lower() in {"1", "true", "yes"}
        if _readcap and os.path.exists(_readcap) and not _disabled:
            _inject = f"-e {shlex.quote(_readcap)} "
            os.environ["PI_READCAP_EXT"] = _readcap   # inherited by sub-agent processes

        _turncap = os.environ.get("PI_TURNCAP_EXT") or str(_exts_dir / "turncap" / "index.ts")
        if str(os.environ.get("PI_TURNCAP_DISABLE", "")).lower() not in {"1", "true", "yes"}:
            if not os.path.exists(_turncap):
                raise RuntimeError(f"HostPi: the turn-cap extension is missing ({_turncap!r}); set PI_TURNCAP_DISABLE=1 to run uncapped")
            _inject += f"-e {shlex.quote(_turncap)} "
            os.environ["PI_TURNCAP_EXT"] = _turncap   # inherited by sub-agent processes

        if self._compaction is not None:
            _guard_off = str(os.environ.get("PI_COMPACTGUARD_DISABLE", "")).lower() in {
                "1",
                "true",
                "yes",
            }
            _guard = os.environ.get("PI_COMPACT_EXT") or _deployed_compaction_ext()
            if not (_guard and os.path.exists(_guard)):
                raise RuntimeError(
                    "HostPi: compaction is enabled (PI_COMPACT_AT_TOKENS="
                    f"{self._compaction['trigger']}) but the compaction extension is not deployed"
                    + (f" (looked for {_guard!r})" if _guard else " (could not locate the pi package)")
                    + ". Run `bash src/rlm_train/pi/install_node_pi.sh` to deploy it — it imports "
                    "pi internals, so it must live inside the pi package tree. Without it NOTHING "
                    "compacts: pi's own threshold path only fires after the agent loop has already "
                    "ended (see the extension header), so the run would silently be a plain pi-mono "
                    "baseline under a -compact32k name."
                )
            _inject += f"-e {shlex.quote(_guard)} "
            os.environ["PI_OUTPUT_FORMAT_FILE"] = OUTPUT_FORMAT_PATH
            os.environ["PI_COMPACT_KEEP_RECENT_TOKENS"] = str(
                self._compaction["keep_recent_tokens"]
            )
            os.environ["PI_COMPACT_RESERVE_TOKENS"] = str(self._compaction["reserve_tokens"])
            if _guard_off:
                os.environ["PI_COMPACTGUARD_DISABLE"] = "1"

        _subagent = os.environ.get("PI_SUBAGENT_EXT") or _deployed_subagent_ext()
        _sub_disabled = str(os.environ.get("PI_SUBAGENT_DISABLE", "")).lower() in {"1", "true", "yes"}
        if not _sub_disabled and not (_subagent and os.path.exists(_subagent)):
            raise RuntimeError(
                "HostPi: the pi sub-agent extension is not deployed"
                + (f" (looked for {_subagent!r})" if _subagent else " (could not locate the pi package)")
                + ". Run `bash src/rlm_train/pi/install_node_pi.sh` to deploy it — it must live "
                "inside the pi package tree or node cannot resolve its imports. "
                "Set PI_SUBAGENT_DISABLE=1 to run single-agent on purpose."
            )
        self._subagents_enabled = not _sub_disabled
        if self._subagents_enabled:
            _inject += f"-e {shlex.quote(_subagent)} "
            _agents_dir = Path(
                os.environ.get("PI_SUBAGENT_AGENTS_DIR")
                or (_exts_dir / "subagent" / "agents-baseline")
            )
            _files = dict(program.get("files") or {})
            _n_agents = 0
            for _md in sorted(_agents_dir.glob("*.md")):
                _files[f"{AGENTS_DIR_PATH}/{_md.name}"] = _static_file(
                    _md.read_text(encoding="utf-8")
                )
                _n_agents += 1
            if not _n_agents:
                raise RuntimeError(
                    f"HostPi: sub-agents enabled but no agent definitions found in {_agents_dir}. "
                    "The subagent tool would load with zero agents and every delegation would fail; "
                    "set PI_SUBAGENT_AGENTS_DIR or PI_SUBAGENT_DISABLE=1."
                )
            # sub-agents use the marked model entry unless PI_SUBAGENT_MODEL is set
            _sub_model = os.environ.get("PI_SUBAGENT_MODEL")
            if _sub_model and _sub_model != PI_SUBAGENT_MODEL_ID:
                if self.train_root_only:
                    raise RuntimeError(
                        f"HostPi: train_root_only=True needs children on the marked model "
                        f"{PI_SUBAGENT_MODEL_ID!r}, but PI_SUBAGENT_MODEL={_sub_model!r} "
                        "overrides it — their requests would be indistinguishable from the "
                        "root's and would be TRAINED ON. Unset PI_SUBAGENT_MODEL."
                    )
            else:
                os.environ["PI_SUBAGENT_MODEL"] = PI_SUBAGENT_MODEL_ID
                _files[MODELS_JSON_PATH] = _marked_models_json_file
            program["files"] = _files
        elif self.train_root_only:
            raise RuntimeError(
                "HostPi: train_root_only=True with PI_SUBAGENT_DISABLE=1 is a plain "
                "single-agent run — every turn is a root turn. Drop one of the two."
            )

        self.keep_trajectory_step = (
            keep_only_root_pi_steps if self.train_root_only else count_all_pi_steps
        )

        # prompt text, not a file attachment
        cmd = program.get("command")
        instr = shlex.quote(str(kwargs.get("instruction_path")))
        if isinstance(cmd, list) and cmd and isinstance(cmd[-1], str):
            new_script = cmd[-1].replace(
                f"-p @{instr}", f'{_inject}-p "$(cat {instr})"'
            )
            if new_script == cmd[-1]:
                raise RuntimeError(
                    "HostPi: could not find `-p @<instruction>` in the pi command to "
                    "inline; vf.Pi's build_pi_run_script may have changed."
                )
            if self._pi_max_turns is not None:
                new_script = f"export PI_MAX_TURNS={self._pi_max_turns}\n{new_script}"
            program["command"] = [*cmd[:-1], new_script]

        self.program = program
        self._program = self.compile_program(self.program)

    async def run(self, task, state=None):  # type: ignore[override]
        """Run under the wall-clock cap; copy runtime.sampling_args to the state top level, where prime-rl reads it."""
        if self._rollout_timeout is None:
            state = await super().run(task, state)
        else:
            try:
                state = await asyncio.wait_for(
                    super().run(task, state), timeout=self._rollout_timeout
                )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"HostPi: rollout exceeded {self._rollout_timeout:g}s "
                    f"(PI_ROLLOUT_TIMEOUT, default {ROLLOUT_TIMEOUT_FACTOR}x the "
                    f"{self._command_timeout}s sandbox command_timeout) and was "
                    "abandoned. The agent command is separately bounded, so this "
                    "means the rollout stopped progressing outside it — raising "
                    "rather than hanging the eval barrier. PI_ROLLOUT_TIMEOUT=0 "
                    "restores the unbounded behaviour."
                ) from exc
        if not state.get("sampling_args"):
            runtime_sampling = (state.get("runtime") or {}).get("sampling_args")
            if runtime_sampling:
                state["sampling_args"] = dict(runtime_sampling)
        return state
