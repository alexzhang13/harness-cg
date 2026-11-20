"""Rollouts -> training samples -> Tinker datums, as in prime-rl@eaf95997f.

A step whose prompt extends an open sample is merged into it, otherwise it starts a new one. Samples are cut at
seq_len.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tinker


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_mask: list[bool]
    completion_logprobs: list[float]
    advantage: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)

    @property
    def num_loss_tokens(self) -> int:
        return sum(1 for m in self.completion_mask if m)


def interleave_rollout(output: dict[str, Any]) -> list[Sample] | None:
    trajectory = output.get("trajectory") or []
    if not trajectory:
        return None
    has_error = output.get("error") is not None
    steps = []
    for step in trajectory:
        tokens = step.get("tokens")
        if tokens is None:
            return None
        steps.append(tokens)

    def make(t: dict[str, Any]) -> Sample:
        mask = [False] * len(t["completion_ids"]) if has_error else [bool(m) for m in t["completion_mask"]]
        return Sample(list(t["prompt_ids"]), list(t["completion_ids"]), mask, list(t["completion_logprobs"]))

    first = steps[0]
    active: list[tuple[list[int], Sample]] = [(list(first["prompt_ids"]) + list(first["completion_ids"]), make(first))]
    for t in steps[1:]:
        prompt_ids = list(t["prompt_ids"])
        matched = next((i for i, (prefix, _) in enumerate(active) if prompt_ids[: len(prefix)] == prefix), None)
        if matched is None:
            active.append((prompt_ids + list(t["completion_ids"]), make(t)))
            continue
        prefix, sample = active[matched]
        new_prompt = prompt_ids[len(prefix):]
        sample.completion_ids.extend(new_prompt)
        sample.completion_mask.extend([False] * len(new_prompt))
        sample.completion_logprobs.extend([0.0] * len(new_prompt))
        sample.completion_ids.extend(t["completion_ids"])
        sample.completion_mask.extend([False] * len(t["completion_ids"]) if has_error
                                      else [bool(m) for m in t["completion_mask"]])
        sample.completion_logprobs.extend(t["completion_logprobs"])
        active[matched] = (prompt_ids + list(t["completion_ids"]), sample)
    return [s for _, s in active]


def truncate(sample: Sample, seq_len: int) -> Sample:
    if sample.num_tokens <= seq_len:
        return sample
    p = len(sample.prompt_ids)
    if p >= seq_len:
        return Sample(sample.prompt_ids[:seq_len], [], [], [], sample.advantage)
    keep = seq_len - p
    return Sample(list(sample.prompt_ids), sample.completion_ids[:keep], sample.completion_mask[:keep],
                  sample.completion_logprobs[:keep], sample.advantage)


def rollout_samples(output: dict[str, Any], seq_len: int) -> list[Sample]:
    samples = interleave_rollout(output) or []
    adv = float(output.get("advantage") or 0.0)
    out = []
    for s in samples:
        s.advantage = adv
        out.append(truncate(s, seq_len))
    return out


def build_datums(samples: list[Sample]) -> tuple[list[tinker.Datum], int]:
    n_loss = sum(s.num_loss_tokens for s in samples)
    if n_loss == 0:
        return [], 0
    inv = 1.0 / n_loss
    datums = []
    for s in samples:
        if not s.num_loss_tokens:
            continue  # no loss tokens
        toks = list(s.prompt_ids) + list(s.completion_ids)
        mask = [0.0] * len(s.prompt_ids) + [1.0 if m else 0.0 for m in s.completion_mask]
        lps = [0.0] * len(s.prompt_ids) + [float(x) for x in s.completion_logprobs]
        n = len(toks) - 1
        datums.append(tinker.Datum(
            model_input=tinker.ModelInput.from_ints(toks[:-1]),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=toks[1:], dtype="int64", shape=[n]),
                "inference_logprobs": tinker.TensorData(data=lps[1:], dtype="float32", shape=[n]),
                "advantages": tinker.TensorData(data=[float(s.advantage)] * n, dtype="float32", shape=[n]),
                "loss_mask": tinker.TensorData(data=mask[1:], dtype="float32", shape=[n]),
                "inv_loss_scale": tinker.TensorData(data=[inv], dtype="float32", shape=[]),
            },
        ))
    return datums, n_loss
