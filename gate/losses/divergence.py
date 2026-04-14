"""Per-branch teacher-student divergence for DADG gate supervision.

Produces a (B, 3) tensor of divergences `d = [d_feat, d_attn, d_loc]` per
image. These are converted to soft gate targets elsewhere via
`softmax(-d / τ)` — low divergence = easy-to-align branch = high weight.

Design notes
------------
* Each metric is computed per-sample then normalized within the batch
  (min-max on each branch axis) so the three metrics share a comparable
  scale before being softmax-ed into a target distribution.
* Teacher tensors must be detached BEFORE calling these functions; we do
  not wrap with `detach()` here in case the caller wants raw values.
* All functions accept tensors of shape (B, ...) and return (B,).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-8


def feature_divergence(
    student_feat: torch.Tensor, teacher_feat: torch.Tensor
) -> torch.Tensor:
    """Cosine distance averaged over channel and spatial dims.

    student_feat / teacher_feat: (B, C, H, W). Returns (B,).
    """
    s = student_feat.flatten(1)
    t = teacher_feat.flatten(1)
    cos = F.cosine_similarity(s, t, dim=1).clamp(-1.0, 1.0)
    return 1.0 - cos  # in [0, 2], smaller = more aligned


def attention_divergence(
    student_attn: torch.Tensor, teacher_attn: torch.Tensor
) -> torch.Tensor:
    """KL divergence between normalized spatial attention maps.

    Inputs: (B, H, W) or (B, 1, H, W) non-negative attention maps.
    """
    if student_attn.dim() == 4:
        student_attn = student_attn.squeeze(1)
        teacher_attn = teacher_attn.squeeze(1)
    B = student_attn.size(0)
    s = student_attn.reshape(B, -1)
    t = teacher_attn.reshape(B, -1)
    s = s / (s.sum(dim=1, keepdim=True) + EPS)
    t = t / (t.sum(dim=1, keepdim=True) + EPS)
    # KL(teacher || student) — measure how hard student is to match teacher
    return (t * ((t + EPS).log() - (s + EPS).log())).sum(dim=1)


def localization_divergence(
    student_boxes: torch.Tensor,
    teacher_boxes: torch.Tensor,
    reduction: str = "l1",
) -> torch.Tensor:
    """Localization branch divergence from bbox regression outputs.

    student_boxes / teacher_boxes: (B, N, 4) xyxy or xywh aligned per-sample.
    Reduction 'l1' returns per-sample mean |Δ|; 'iou_gap' returns 1 - mean IoU.
    Both shapes must be pre-matched by the caller (typically via hungarian
    or anchor assignment already done during distillation loss computation).
    """
    if reduction == "l1":
        return (student_boxes - teacher_boxes).abs().mean(dim=(1, 2))
    if reduction == "iou_gap":
        iou = _pairwise_iou(student_boxes, teacher_boxes)
        return 1.0 - iou.mean(dim=1)
    raise ValueError(f"unknown reduction: {reduction}")


def _pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (B, N, 4) xyxy. Returns (B, N) IoU per pair."""
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + EPS
    return inter / union


def minmax_normalize(d: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Min-max normalize to [0, 1] along `dim` (default: across batch)."""
    d_min = d.min(dim=dim, keepdim=True).values
    d_max = d.max(dim=dim, keepdim=True).values
    return (d - d_min) / (d_max - d_min + EPS)


def stack_divergences(
    d_feat: torch.Tensor, d_attn: torch.Tensor, d_loc: torch.Tensor
) -> torch.Tensor:
    """Stack three per-sample divergences into (B, 3), each per-branch
    min-max normalized across the batch so the scales are comparable before
    `softmax(-d/τ)` is applied in gate_loss."""
    d_feat_n = minmax_normalize(d_feat, dim=0)
    d_attn_n = minmax_normalize(d_attn, dim=0)
    d_loc_n = minmax_normalize(d_loc, dim=0)
    return torch.stack([d_feat_n, d_attn_n, d_loc_n], dim=1)
