"""PersonaMem-v3 (`personamem-v3-real`): rank a recommendation slate from a user's time-masked cross-app history.

Only the deterministically scored ``hidden_persona_recommendation`` family is ported. Reward is upstream's graded
NDCG@5. Data licence: CC-BY-NC-4.0.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

import rlm_train
from personamem_v3 import _upstream as up

_REPO = "bowen-upenn/PersonaMem-v3"
# pinned dataset revision
_REVISION = os.environ.get("PERSONAMEM_V3_REVISION", "7b00a090b35b7293e6efeeb19494207f32b5a9ee")

APPS: tuple[str, ...] = ("instagram", "facebook", "threads", "chatbot", "ai_studio")

_PERSONA_FILES: tuple[str, ...] = (
    "test.json",
    "calendar.json",
) + tuple(f"{a}.json" for a in APPS)


RANKING_TASKS: tuple[str, ...] = ("hidden_persona_recommendation",)

_CANDIDATE_FIELDS: dict[str, tuple[str, ...]] = {
    "hidden_persona_recommendation": ("title", "hashtags"),
}

_CANDIDATE_BANNED: frozenset[str] = frozenset({
    "source_timestamp", "source_interaction_type", "source_object_id",
})


class _Backend:
    """Lazy per-persona reader over ``backend/{persona_id}/``."""

    def __init__(self, repo: str = _REPO) -> None:
        self._repo = repo
        self._local_dir = os.environ.get("PERSONAMEM_V3_BACKEND_DIR")
        self._events: dict[tuple[str, str], list[dict]] = {}
        self._calendar: dict[str, list[dict]] = {}
        self._tests: dict[str, list[dict]] = {}
        self._personas: list[str] | None = None


    def _path(self, persona_id: str, filename: str) -> str:
        if self._local_dir:
            return os.path.join(self._local_dir, persona_id, filename)
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            self._repo, f"backend/{persona_id}/{filename}", repo_type="dataset", revision=_REVISION
        )

    def personas(self) -> list[str]:
        """Persona ids, numerically sorted."""
        if self._personas is None:
            if self._local_dir:
                ids = [
                    d for d in os.listdir(self._local_dir)
                    if os.path.isdir(os.path.join(self._local_dir, d))
                ]
            else:
                from huggingface_hub import list_repo_files

                files = list_repo_files(self._repo, repo_type="dataset", revision=_REVISION)
                ids = sorted({
                    m.group(1) for m in (re.match(r"backend/([^/]+)/test\.json$", f) for f in files) if m
                })
            self._personas = sorted(ids, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        return self._personas

    def _load_json(self, persona_id: str, filename: str) -> Any:
        path = self._path(persona_id, filename)
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            return json.load(fh)

    def tests(self, persona_id: str) -> list[dict]:
        if persona_id not in self._tests:
            doc = self._load_json(persona_id, "test.json") or []
            self._tests[persona_id] = doc if isinstance(doc, list) else []
        return self._tests[persona_id]

    def raw_events(self, persona_id: str, app: str) -> list[dict]:
        key = (persona_id, app)
        if key not in self._events:
            doc = self._load_json(persona_id, f"{app}.json")
            if isinstance(doc, dict):
                for k in ("events", "interactions", "sessions"):
                    if isinstance(doc.get(k), list):
                        doc = doc[k]
                        break
            self._events[key] = [e for e in doc if isinstance(e, dict)] if isinstance(doc, list) else []
        return self._events[key]

    def calendar_mods(self, persona_id: str) -> list[dict]:
        if persona_id not in self._calendar:
            doc = self._load_json(persona_id, "calendar.json")
            mods = doc.get("modifications") if isinstance(doc, dict) else None
            self._calendar[persona_id] = list(mods) if isinstance(mods, list) else []
        return self._calendar[persona_id]

    def drop_persona(self, persona_id: str) -> None:
        """Free one persona's parsed JSON (~4 MB per app stream)."""
        for app in APPS:
            self._events.pop((persona_id, app), None)
        self._calendar.pop(persona_id, None)
        self._tests.pop(persona_id, None)


    def history(
        self,
        persona_id: str,
        t_test: int,
        *,
        cap: int | None = None,
    ) -> list[dict]:
        """The agent-visible, time-masked, chronological cross-app timeline (DMs excluded)."""
        timeline: list[tuple[int, str, dict]] = []
        for app in APPS:
            for e in self.raw_events(persona_id, app):
                if e.get("is_dm"):
                    continue
                ts = e.get("source_timestamp", 0) or 0
                if ts >= t_test:
                    continue
                compact = up.compact_event(up.strip_event(e), strip_preferences=False)
                timeline.append((ts, app, {"app": app, **compact}))
        timeline.sort(key=lambda x: (x[0], x[1]))
        out = [t[2] for t in timeline]
        if cap is not None and cap > 0:
            out = out[-cap:]
        return out

    def calendar(self, persona_id: str, t_test: int) -> list[dict]:
        return up.calendar_entries(up.calendar_state(self.calendar_mods(persona_id), t_test))


_INT_RE = re.compile(r"-?\d+")


def _coerce_int_list(obj: Any) -> list[int] | None:
    """Pull a ranking out of whatever the model submitted."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        out: list[int] = []
        for x in obj:
            if isinstance(x, bool):
                return None
            if isinstance(x, int):
                out.append(x)
            elif isinstance(x, float) and float(x).is_integer():
                out.append(int(x))
            elif isinstance(x, str) and _INT_RE.fullmatch(x.strip()):
                out.append(int(x.strip()))
            else:
                return None
        return out
    if isinstance(obj, dict):
        for key in ("ranked_indexes", "ranked_indices", "ranking", "ranked",
                    "indexes", "indices", "content", "answer", "final_answer"):
            if key in obj:
                got = _coerce_int_list(obj[key])
                if got is not None:
                    return got
        return None
    if not isinstance(obj, str):
        return None

    text = obj.strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            got = _coerce_int_list(parse(text))
        except Exception:  # noqa: BLE001
            got = None
        if got is not None:
            return got
    fences = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    for block in reversed(fences):
        got = _coerce_int_list(block.strip())
        if got is not None:
            return got
    for m in reversed(list(re.finditer(r"\[([\s\d,\-]*)\]", text))):
        inner = m.group(1).strip()
        if not inner:
            continue
        toks = [t.strip() for t in inner.split(",") if t.strip()]
        if toks and all(_INT_RE.fullmatch(t) for t in toks):
            return [int(t) for t in toks]
    return None


def _normalize_ranking(raw: Any, n: int) -> tuple[list[int] | None, bool]:
    """``(ranking, was_exact_permutation)``."""
    got = _coerce_int_list(raw)
    if got is None:
        return None, False
    seen: set[int] = set()
    ranking: list[int] = []
    for i in got:
        if 0 <= i < n and i not in seen:
            seen.add(i)
            ranking.append(i)
    exact = len(ranking) == n == len(got)
    if not ranking:
        return None, False
    ranking += [i for i in range(n) if i not in seen]
    return ranking, exact


METRICS: tuple[str, ...] = ("official",)


def score_ranking(label: dict, ranking: list[int], metric: str = "official") -> float:
    """Graded NDCG@5 of a ranking (target grade 2, filler 1, hard negatives -2)."""
    task = label["task"]
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}; got {metric!r}")
    if task == "hidden_persona_recommendation":
        hard = label.get("hard_negative_idxs") or []
        filler = up.NDCG_REL_FILLER
        if not hard:
            filler = 0.0
        return up.graded_ndcg_at_k(
            ranking, {label["held_out_idx"]}, hard, 5, filler_grade=filler
        )
    raise ValueError(f"unsupported task {task!r}")


async def _score(answer: str, state: vf.State, **_kw: Any) -> float:
    label = json.loads(answer) if isinstance(answer, str) else answer
    metric = label.get("metric_variant", "official")
    n = int(label["n_candidates"])
    final = state.get("rlm_final_answer")
    if final is None:
        final = state.get("final_answer")
    ranking, exact = _normalize_ranking(final, n)
    state["pm3_task"] = label["task"]
    state["pm3_parse_ok"] = ranking is not None
    state["pm3_exact_permutation"] = exact
    if ranking is None:
        state["pm3_ranking"] = None
        return 0.0
    state["pm3_ranking"] = ranking
    return float(score_ranking(label, ranking, metric))


_CONTEXT_DOC = """The variable `context` is a dict describing ONE user's cross-platform digital
life, already time-masked to strictly before the test moment:

  context["history"]    list of {} engagement events, oldest first. Each is a dict
                       with some of: app, t (unix s), when, action, type
                       (implicit_negative / implicit_positive / explicit_positive /
                       explicit_negative / feed_visible), hashtags, title, caption,
                       desc, conversation, user_message, author_id, location,
                       is_ad, is_trending, inferred_prefs.
  context["calendar"]   the user's calendar entries as of the test moment.
  context["moment"]     when/where the test moment is.
  context["candidates"] the {} candidate items to rank, in slate order 0..{}.
"""

_SUBMIT_DOC = """Work in the REPL: index the history (hashtag counts by interaction polarity,
recency, per-app behaviour, self-authored posts, @ai comments) instead of
printing all {} events.

Submit a ranking of ALL {} candidate indexes, most likely first, as a Python
list of ints:

    answer["content"] = [{}]
    answer["ready"] = True

It must be a permutation of range({}). A partial list is scored as if the
missing indexes were appended in ascending order; no list at all scores 0."""


def _root_prompt(task: str, inst: dict, n_hist: int, n_cand: int) -> str:
    doc = _CONTEXT_DOC.format(n_hist, n_cand, n_cand - 1)
    example = ", ".join(str(i) for i in range(min(3, n_cand))) + ", ..."
    submit = _SUBMIT_DOC.format(n_hist, n_cand, example, n_cand)

    if task == "hidden_persona_recommendation":
        query = (inst.get("query_text") or "").strip()
        hour = int(inst.get("anchor_hour_utc") or 8)
        if query:
            moment = (inst.get("moment") or "").strip()
            moment_line = f" (moment: {moment})" if moment else ""
            framing = (
                f"It is {inst.get('day_label')} {hour:02d}:00 UTC{moment_line}. The user just "
                f"asked their agent: \"{query}\"\nThe {n_cand} candidates are items already in "
                f"the user's feeds. Rank them in the order the user is most likely to want to "
                f"see right now, given the moment and their history."
            )
        else:
            framing = (
                f"It is {inst.get('day_label')} {hour:02d}:00 UTC. The recommendation system has "
                f"proposed {n_cand} candidate feed items. Rank them in the order this user is "
                f"most likely to positively engage with, given their history."
            )
        head = "# Task: personalized social-media content recommendation"
    else:  # pragma: no cover
        raise ValueError(task)

    return f"{head}\n\n{framing}\n\n{doc}\n{submit}"


def _render_candidates(task: str, candidates: list[dict]) -> list[dict]:
    keep = set(_CANDIDATE_FIELDS[task])
    keep -= _CANDIDATE_BANNED
    out: list[dict] = []
    for i, c in enumerate(candidates):
        item = {"idx": i}
        for k in sorted(keep):
            if k in c and c[k] not in (None, "", [], {}):
                item[k] = c[k]
        out.append(item)
    return out


def _label_for(task: str, inst: dict) -> dict | None:
    """Extract the scorer-side slate labels; ``None`` if the row is unusable."""
    n = len(inst.get("candidates") or [])
    if n < 2:
        return None
    base = {"task": task, "n_candidates": n}
    if task == "hidden_persona_recommendation":
        held = inst.get("held_out_idx")
        if not isinstance(held, int) or isinstance(held, bool) or not (0 <= held < n):
            return None
        hard = [i for i in (inst.get("hard_negative_idxs") or []) if isinstance(i, int) and 0 <= i < n]
        if held in hard:
            return None
        base.update(held_out_idx=held, hard_negative_idxs=sorted(set(hard)))
    else:
        return None
    return base


def _upstream_ranking(field: Any, n: int) -> list[int] | None:
    """Parse one of upstream's ``Ranked indexes: [...]`` strings."""
    text = field.get("text") if isinstance(field, dict) else field
    ranking, exact = _normalize_ranking(text, n)
    return ranking if (ranking is not None and exact) else None


def _gold_ranking(task: str, inst: dict, label: dict) -> list[int] | None:
    """The optimal ranking: upstream's ``example_response``."""
    return _upstream_ranking(inst.get("example_response"), label["n_candidates"])


def _inferior_ranking(task: str, inst: dict, label: dict) -> list[int] | None:
    """Upstream's deliberately wrong ranking (``inferior_response``)."""
    return _upstream_ranking(inst.get("inferior_response"), label["n_candidates"])


def _moment(inst: dict, t_test: int, history: list[dict]) -> dict:
    geo = inst.get("geo_context") or {}
    city = geo.get("city_at_t_test")
    if not city:
        for ev in reversed(history):
            loc = ev.get("location")
            if isinstance(loc, dict) and loc.get("city"):
                city = loc["city"]
                break
    out: dict[str, Any] = {"t_test": t_test, "city": city}
    if inst.get("day_label"):
        out["date_utc"] = inst["day_label"]
    if inst.get("anchor_hour_utc") is not None:
        out["hour_utc"] = inst["anchor_hour_utc"]
    if "same_as_intent_city" in geo:
        out["in_a_city_matching_a_past_interest"] = geo["same_as_intent_city"]
    return out


_DROP_REASONS = (
    "no_labels",
    "no_gold",
    "empty_history",
)

_MAX_ROOT_PROMPT_CHARS = 2600


def _new_stat() -> dict[str, int]:
    return {"seen": 0, "kept": 0, **{r: 0 for r in _DROP_REASONS}}


def _build_dataset(
    *,
    tasks: tuple[str, ...],
    personas: list[str],
    backend: _Backend,
    history_event_cap: int | None,
    num_examples: int,
    seed: int | None,
) -> tuple[Dataset, dict[str, dict[str, int]]]:
    import random

    rng = random.Random(seed)
    stats: dict[str, dict[str, int]] = {t: _new_stat() for t in tasks}
    rows: list[dict] = []

    persona_order = list(personas)
    if seed is not None:
        rng.shuffle(persona_order)

    for pos, pid in enumerate(persona_order):
        limit: int | None = None
        if num_examples and num_examples > 0:
            remaining = num_examples - len(rows)
            if remaining <= 0:
                break
            personas_left = len(persona_order) - pos
            limit = -(-remaining // personas_left)
        try:
            persona_rows = _rows_for_persona(
                pid=pid, tasks=tasks, backend=backend, stats=stats,
                history_event_cap=history_event_cap, limit=limit, seed=seed,
            )
        finally:
            backend.drop_persona(pid)
        if num_examples and num_examples > 0:
            persona_rows = persona_rows[: num_examples - len(rows)]
        rows.extend(persona_rows)

    if seed is not None:
        rng.shuffle(rows)

    if not rows:
        raise ValueError(
            "No PersonaMem-v3 rows survived the filters "
            f"(tasks={tasks}, personas={len(personas)}). Drop counts: {stats}"
        )
    for i, r in enumerate(rows):
        r["example_id"] = i
    return Dataset.from_list(rows), stats


def _rows_for_persona(
    *,
    pid: str,
    tasks: tuple[str, ...],
    backend: _Backend,
    stats: dict[str, dict[str, int]],
    history_event_cap: int | None,
    limit: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Build this persona's rows, at most ``limit`` of them."""
    import random

    eligible: list[tuple[dict, dict, dict, list[int]]] = []
    for test_row in backend.tests(pid):
        task = test_row.get("task_type")
        if task not in tasks:
            continue
        st = stats[task]
        st["seen"] += 1
        inst = test_row.get("instance_full") or {}

        label = _label_for(task, inst)
        if label is None:
            st["no_labels"] += 1
            continue
        gold = _gold_ranking(task, inst, label)
        if gold is None:
            st["no_gold"] += 1
            continue
        eligible.append((test_row, inst, label, gold))

    if seed is not None:
        random.Random(f"{seed}:{pid}").shuffle(eligible)

    out: list[dict] = []
    hist_cache: dict[int, list[dict]] = {}
    for test_row, inst, label, gold in eligible:
        if limit is not None and len(out) >= limit:
            break
        task = test_row["task_type"]
        st = stats[task]
        n_cand = label["n_candidates"]

        t_test = int(inst.get("t_test") or inst.get("source_timestamp") or test_row.get("ts") or 0)
        if t_test <= 0:
            st["empty_history"] += 1
            continue
        if t_test not in hist_cache:
            hist_cache[t_test] = backend.history(pid, t_test, cap=history_event_cap)
        history = hist_cache[t_test]
        if not history:
            st["empty_history"] += 1
            continue

        candidates = _render_candidates(task, inst.get("candidates") or [])
        context: dict[str, Any] = {
            "moment": _moment(inst, t_test, history),
            "candidates": candidates,
            "history": history,
        }
        context["calendar"] = backend.calendar(pid, t_test)
        context_json = json.dumps(context, ensure_ascii=False)
        ctx_chars = len(context_json)

        root_prompt = _root_prompt(task, inst, len(history), n_cand)
        if len(root_prompt) > _MAX_ROOT_PROMPT_CHARS:
            raise AssertionError(
                f"root_prompt for task={task} query={test_row.get('query_id')} is "
                f"{len(root_prompt)} chars (cap {_MAX_ROOT_PROMPT_CHARS}); the root "
                "instruction must stay short — did an upstream field grow?"
            )
        for c in candidates:
            bad = [k for k in c if k.startswith("_") or k in _CANDIDATE_BANNED]
            if bad:
                raise AssertionError(
                    f"candidate leak fields {bad} survived rendering for task={task}"
                )

        label = dict(label)
        label["gold_ranking"] = gold
        label["inferior_ranking"] = _inferior_ranking(task, inst, label)
        label["metric_variant"] = "official"
        label["max_reward"] = 1.0
        label["upstream_example_ranking"] = _upstream_ranking(inst.get("example_response"), n_cand)
        label["upstream_inferior_ranking"] = _upstream_ranking(inst.get("inferior_response"), n_cand)

        info = {
            "context": context,
            "root_prompt": root_prompt,
            "personamem_task": task,
            "persona_id": pid,
            "query_id": test_row.get("query_id") or inst.get("instance_id"),
            "instance_id": inst.get("instance_id"),
            "t_test": t_test,
            "n_history_events": len(history),
            "n_candidates": n_cand,
            "context_chars": ctx_chars,
            "history_apps": list(APPS),
            "metric": _METRIC_NAME[task],
            "metric_variant": "official",
            "max_reward": label["max_reward"],
            "lag_bucket": inst.get("lag_bucket"),
            "phase": inst.get("phase"),
            "query_kind": "moment" if (inst.get("query_text") or "").strip() else "proactive",
            "license": "cc-by-nc-4.0",
            "non_commercial": True,
            "source": f"{_REPO}:backend/{pid}/test.json",
        }
        out.append({
            "prompt": [{"role": "user", "content": _short_question(task, inst, n_cand)}],
            "answer": json.dumps(label),
            "info": json.dumps(info),
        })
        st["kept"] += 1
    return out


_METRIC_NAME: dict[str, str] = {
    "hidden_persona_recommendation": "graded_ndcg@5",
}


def _short_question(task: str, inst: dict, n_cand: int) -> str:
    """A one-line human-readable question for the (unused) ``prompt`` column."""
    query = (inst.get("query_text") or "").strip()
    if query:
        return f"The user asked: {query!r} — rank the {n_cand} candidate feed items for them."
    return f"Rank the {n_cand} candidate feed items this user is most likely to engage with."


def load_environment(
    *,
    task: str = "hidden_persona_recommendation",
    history_event_cap: int | None = None,
    num_examples: int = -1,
    seed: int = 42,
    max_iterations: int = 20,
    sub_max_tokens: int = 4096,
    min_subcall: int = 0,
    **kwargs: Any,
) -> vf.Environment:
    """Load PersonaMem-v3's hidden-persona recommendation family through RLMTrainEnv."""
    tasks_t = tuple(dict.fromkeys(t.strip() for t in str(task).split(",") if t.strip()))
    for t in tasks_t:
        if t not in RANKING_TASKS:
            raise ValueError(f"Unknown PersonaMem-v3 task {t!r}. Choose from {RANKING_TASKS}.")

    backend = _Backend()
    persona_ids = backend.personas()
    if not persona_ids:
        raise ValueError("No personas found")

    dataset, _stats = _build_dataset(
        tasks=tasks_t,
        personas=persona_ids,
        backend=backend,
        history_event_cap=history_event_cap,
        num_examples=num_examples,
        seed=seed,
    )

    rubric = rlm_train.RLMTrainRubric(
        correctness=_score, weight=1.0, min_subcall=min_subcall,
    )
    return rlm_train.RLMTrainEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        sub_sampling_args={"max_tokens": sub_max_tokens},
        rubric=rubric,
        **kwargs,
    )


__all__ = [
    "load_environment",
    "RANKING_TASKS",
    "METRICS",
    "score_ranking",
]
