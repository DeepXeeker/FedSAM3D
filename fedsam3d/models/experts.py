from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import torch
import torch.nn as nn

@dataclass
class ExpertPackage:
    """Lightweight per-class expert parameters uploaded by a client.
    We store decoder query projection and mask projection rows for a class.
    """
    class_id: int
    state_dict: Dict[str, torch.Tensor]
    provider: str

class ExpertBank:
    """Holds a set of peer experts keyed by class id."""
    def __init__(self):
        self.experts: Dict[int, List[ExpertPackage]] = {}

    def add(self, pkg: ExpertPackage) -> None:
        self.experts.setdefault(pkg.class_id, []).append(pkg)

    def get(self, class_id: int) -> List[ExpertPackage]:
        return self.experts.get(class_id, [])

    def clear(self) -> None:
        self.experts.clear()
