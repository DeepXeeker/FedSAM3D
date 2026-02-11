from __future__ import annotations
from typing import Sequence
import torch
import torch.nn.functional as F

def dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # prob/target: [B, D, H, W]
    prob = prob.contiguous().view(prob.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    inter = (prob * target).sum(dim=1)
    denom = prob.sum(dim=1) + target.sum(dim=1)
    return 1.0 - (2.0 * inter + eps) / (denom + eps)

def masked_dice_bce_loss(
    probs: torch.Tensor,
    gt: torch.Tensor,
    availability_mask: Sequence[int],
) -> torch.Tensor:
    """Masked supervised loss:
      sum_{c=1..C} I_k(c) * (Dice + BCE)

    probs: [B, C, D, H, W] sigmoid probabilities
    gt:    [B, 1, D, H, W] integer labels in dataset space (after mapping to global ids)
    availability_mask: length C, binary indicators
    """
    B, C, D, H, W = probs.shape
    gt = gt.long()
    total = 0.0
    denom = 0.0
    for c in range(1, C + 1):
        if availability_mask[c - 1] == 0:
            continue
        target = (gt[:, 0] == c).float()
        p = probs[:, c - 1]
        d = dice_loss(p, target).mean()
        b = F.binary_cross_entropy(p, target)
        total = total + (d + b)
        denom += 1.0
    if denom == 0:
        return torch.zeros([], device=probs.device, dtype=probs.dtype)
    return total / denom
