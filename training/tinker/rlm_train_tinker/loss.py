"""prime-rl ``default_loss_fn`` (DPPO-binary TV mask + Kimi-K2.5 KL) as a Tinker custom loss.

Per token (``prime_rl/trainer/rl/loss.py:116-165`` @ eaf95997f):

    log_ratio = lp_train - lp_inf;  ratio = exp(log_ratio)
    d         = exp(lp_train) - exp(lp_inf)
    masked    = where(A > 0, d > dppo_mask_high, d < -dppo_mask_low)
    loss      = sum( -(mask & ~masked) * adv_tau * A * ratio + kl_tau * mask * log_ratio**2 )

Each datum carries inv_loss_scale = 1 / (loss tokens in the step), so the sum over datums is prime-rl's token mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossConfig:
    dppo_mask_low: float = 0.2
    dppo_mask_high: float = 0.2
    adv_tau: float = 1.0
    kl_tau: float = 1e-3


def make_dppo_kl_loss(cfg: LossConfig):
    lo, hi, adv_tau, kl_tau = float(cfg.dppo_mask_low), float(cfg.dppo_mask_high), float(cfg.adv_tau), float(cfg.kl_tau)

    def prime_rl_dppo_kl(logprobs, inference_logprobs, advantages, loss_mask, inv_loss_scale):
        log_ratio = logprobs - inference_logprobs
        ratio = torch.exp(log_ratio)
        probs_diff = torch.exp(logprobs) - torch.exp(inference_logprobs)
        invalid_high = (probs_diff > hi).float()
        invalid_low = (probs_diff < -lo).float()
        positive = (advantages > 0).float()
        masked = positive * invalid_high + (1.0 - positive) * invalid_low
        keep = loss_mask * (1.0 - masked)
        pg = keep * (adv_tau * advantages) * ratio
        kl = loss_mask * log_ratio * log_ratio
        loss = (-pg + kl_tau * kl).sum() * inv_loss_scale
        mismatch_kl = ratio - log_ratio - 1.0
        n = loss_mask.sum()
        metrics = {
            "custom/loss_tokens:sum": n,
            "custom/masked_tokens:sum": (loss_mask * masked).sum(),
            "custom/masked_low_tokens:sum": (loss_mask * (1.0 - positive) * invalid_low).sum(),
            "custom/masked_high_tokens:sum": (loss_mask * positive * invalid_high).sum(),
            "custom/mismatch_kl_sum:sum": (loss_mask * mismatch_kl).sum(),
        }
        return loss, metrics

    return prime_rl_dppo_kl


INPUT_RANKS = {"logprobs": 1, "inference_logprobs": 1, "advantages": 1, "loss_mask": 1, "inv_loss_scale": 0}


def reference_loss(lp_train: torch.Tensor, lp_inf: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor,
                   loss_scale: float, cfg: LossConfig = LossConfig()) -> torch.Tensor:
    """prime-rl's default_loss_fn with boolean masks, for comparison."""
    mask_b = mask.bool()
    log_ratio = lp_train - lp_inf
    ratio = torch.exp(log_ratio)
    d = torch.exp(lp_train) - torch.exp(lp_inf)
    inv_hi = d > cfg.dppo_mask_high
    inv_lo = d < -cfg.dppo_mask_low
    pos = adv > 0
    is_masked = torch.where(pos, inv_hi, inv_lo)
    keep = mask_b & ~is_masked
    pg = keep * (cfg.adv_tau * adv) * ratio
    kl = mask_b * log_ratio**2
    return (-pg + cfg.kl_tau * kl).sum() / loss_scale
