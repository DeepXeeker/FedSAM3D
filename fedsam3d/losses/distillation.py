from __future__ import annotations
from typing import Sequence, List, Dict, Optional
import torch
import torch.nn.functional as F

def global_kd_loss(
    student: torch.Tensor,   # [B,C,D,H,W]
    teacher: torch.Tensor,   # [B,C,D,H,W]
    availability_mask: Sequence[int],
    tau: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Global consistency distillation (GKD) with confidence gating.

    Applies only on missing classes (I_k(c)=0).
    Mask M(v)=1 if max_{c in missing} teacher_c(v) >= tau else 0.
    Uses BCE between teacher soft target and student prob, masked by M.
    """
    B, C, D, H, W = student.shape
    missing = [i for i in range(C) if availability_mask[i] == 0]
    if len(missing) == 0:
        return torch.zeros([], device=student.device, dtype=student.dtype)

    with torch.no_grad():
        t_missing = teacher[:, missing]  # [B, Cm, D, H, W]
        conf = t_missing.max(dim=1).values  # [B, D, H, W]
        mask = (conf >= tau).float()

    s_missing = student[:, missing]
    # BCE with soft targets
    loss = F.binary_cross_entropy(s_missing, t_missing, reduction="none")  # [B,Cm,D,H,W]
    loss = loss * mask.unsqueeze(1)
    return loss.mean()

def peer_kd_loss(
    student: torch.Tensor,         # [B,C,D,H,W]
    peer_targets: torch.Tensor,    # [B,C,D,H,W]  (only some classes populated)
    selected_missing: List[int],   # class indices (0-based) being distilled
    tau_p: float,
) -> torch.Tensor:
    """Peer-guided distillation (PKD) with confidence gating.

    For selected missing classes, form mask M^P(v)=1 if max_c peer_c(v) >= tau_p else 0.
    BCE between peer soft targets and student.
    """
    if len(selected_missing) == 0:
        return torch.zeros([], device=student.device, dtype=student.dtype)

    p = peer_targets[:, selected_missing]  # [B,m,D,H,W]
    with torch.no_grad():
        conf = p.max(dim=1).values
        mask = (conf >= tau_p).float()
    s = student[:, selected_missing]
    loss = F.binary_cross_entropy(s, p, reduction="none") * mask.unsqueeze(1)
    return loss.mean()
