from __future__ import annotations
from typing import Dict, Any
import torch
from monai.inferers import SlidingWindowInferer
from .utils import to_device
from ..metrics import compute_metrics_batch

def evaluate_model(model, loader, availability_mask, device, patch_size) -> Dict[str, float]:
    model.eval()
    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=1, overlap=0.5, mode="gaussian")
    meters = {"dice": [], "hd95": []}

    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            img = batch["image"]  # [B,1,D,H,W]
            gt = batch["label"]   # [B,1,D,H,W]
            probs = inferer(img, model)  # [B,C,D,H,W]
            m = compute_metrics_batch(probs, gt, availability_mask)
            if not (m["dice"] != m["dice"]):  # not nan
                meters["dice"].append(m["dice"])
                meters["hd95"].append(m["hd95"])

    out = {k: float(sum(v) / max(1, len(v))) for k, v in meters.items()}
    return out
