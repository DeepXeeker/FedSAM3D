from __future__ import annotations
from typing import Dict, Sequence
import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric

def compute_metrics_batch(
    probs: torch.Tensor,     # [B,C,D,H,W]
    gt: torch.Tensor,        # [B,1,D,H,W] global ids
    availability_mask: Sequence[int],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute mean Dice and mean HD95 across available classes for this batch."""
    B, C, D, H, W = probs.shape
    y_pred = (probs >= threshold).float()

    classes = [c for c in range(1, C+1) if availability_mask[c-1] == 1]
    if len(classes) == 0:
        return {"dice": float("nan"), "hd95": float("nan")}

    y_true = torch.stack([(gt[:,0] == c).float() for c in classes], dim=1)  # [B,Ca,D,H,W]
    y_pred_a = torch.stack([y_pred[:, c-1] for c in classes], dim=1)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd_metric = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")

    dice = dice_metric(y_pred=y_pred_a, y=y_true).item()
    hd95 = hd_metric(y_pred=y_pred_a, y=y_true).item()
    return {"dice": float(dice), "hd95": float(hd95)}
