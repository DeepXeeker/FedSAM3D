from __future__ import annotations
from typing import Dict, List, Tuple
import torch

def normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    s = sum(raw.values())
    if s <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}
    return {k: v / s for k, v in raw.items()}

def aggregate_state_dicts(state_dicts: Dict[str, Dict[str, torch.Tensor]], weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """Weighted average of client state_dicts (trainable subset)."""
    keys = next(iter(state_dicts.values())).keys()
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for cid, sd in state_dicts.items():
            w = weights[cid]
            v = sd[k].float()
            acc = v.mul(w) if acc is None else acc.add(v.mul(w))
        out[k] = acc
    return out
