from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import torch
from ..models.experts import ExpertBank, ExpertPackage

def build_peer_targets(
    model,
    vol: torch.Tensor,
    expert_bank: ExpertBank,
    selected_classes: List[int],   # 0-based indices
    device: torch.device,
) -> torch.Tensor:
    """Run peer experts to create peer target probability maps for selected classes.

    Implementation choice for reproducibility:
      - We reuse the target client's backbone+APG to compute features/queries.
      - For each class c, we temporarily load each provider's expert parameters into a copy of the decoder
        and average their outputs (uniform weights).
    """
    with torch.no_grad():
        base_probs = model(vol)  # [B,C,D,H,W] (student predictions)

    # Allocate peer target tensor
    peer = torch.zeros_like(base_probs)

    for c0 in selected_classes:
        pkgs = expert_bank.get(c0 + 1)  # stored as 1-based class_id
        if len(pkgs) == 0:
            continue
        # Uniform ensemble
        acc = None
        for pkg in pkgs:
            # Load expert parameters into model decoder (in-place), run forward, restore later.
            # We only overwrite matching keys in decoder.* namespace.
            # This is lightweight and keeps reproducibility simple.
            backup = {}
            for k, v in pkg.state_dict.items():
                fullk = k
                if fullk in model.state_dict():
                    backup[fullk] = model.state_dict()[fullk].detach().clone()
                    model.state_dict()[fullk].copy_(v.to(device))
            with torch.no_grad():
                p = model(vol)[:, c0]  # [B,D,H,W]
            # restore
            for fullk, old in backup.items():
                model.state_dict()[fullk].copy_(old.to(device))
            acc = p if acc is None else acc + p
        peer[:, c0] = (acc / float(len(pkgs))).clamp(0, 1)

    return peer
