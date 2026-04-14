"""DADG gate loss — KL to divergence-derived soft targets + entropy floor.

The gate never sees gradient from the downstream KD loss; its supervision
is purely this loss. That's what breaks the collapse loop.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def divergence_to_target(
    divergences: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """(B, 3) per-branch divergence → (B, 3) soft target distribution.

    Low divergence = easy branch = high target weight.
    """
    return F.softmax(-divergences / tau, dim=-1)


def dadg_gate_loss(
    gate_output: torch.Tensor,
    divergences: torch.Tensor,
    tau: float = 1.0,
    entropy_floor: float = 0.05,
    entropy_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DADG gate training loss.

    Args:
        gate_output: (B, 3) softmax weights from DADGGate(x).
        divergences: (B, 3) per-branch divergences (already normalized).
        tau: softmax temperature for target construction.
        entropy_floor: per-weight minimum. If a weight dips below, a
            hinge penalty `(floor - w).clamp(min=0)` pushes it back.
        entropy_weight: multiplier on the floor penalty term.

    Returns:
        total_loss, log_dict with components for logging.
    """
    target = divergence_to_target(divergences, tau=tau).detach()
    # KL(gate || target). Use log-softmax-free path: both are valid
    # probability vectors already.
    kl = (gate_output * ((gate_output + EPS).log() - (target + EPS).log())).sum(dim=-1)
    kl_loss = kl.mean()

    # Per-weight floor hinge — prevents any single branch from going to 0
    # even if its divergence is much higher than the others.
    floor_violation = (entropy_floor - gate_output).clamp(min=0.0)
    floor_loss = floor_violation.sum(dim=-1).mean()

    total = kl_loss + entropy_weight * floor_loss
    return total, {
        "gate/kl_loss": kl_loss.detach(),
        "gate/floor_loss": floor_loss.detach(),
        "gate/total_loss": total.detach(),
        "gate/w_feat_mean": gate_output[:, 0].mean().detach(),
        "gate/w_attn_mean": gate_output[:, 1].mean().detach(),
        "gate/w_loc_mean": gate_output[:, 2].mean().detach(),
        "gate/w_feat_min": gate_output[:, 0].min().detach(),
    }
