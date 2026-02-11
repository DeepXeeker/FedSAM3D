"""Paper-aligned organ vocabulary and dataset-to-global label mappings.

Global vocabulary follows a common BTCV/Synapse 13-organ abdomen label set.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

# Global organ vocabulary (1..13) with 0 reserved for background.
GLOBAL_ORGANS: List[str] = [
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "portal_vein_and_splenic_vein",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland",
]

NAME_TO_GLOBAL_ID = {name: i + 1 for i, name in enumerate(GLOBAL_ORGANS)}

@dataclass(frozen=True)
class LabelMap:
    """Maps dataset-specific integer labels -> one or more global class ids.
    Each dataset label can map to multiple global ids (e.g., KiTS kidney -> L/R kidney).
    """
    dataset_name: str
    mapping: Dict[int, List[int]]  # dataset_label_value -> list[global_ids]

    def availability_mask(self) -> List[int]:
        """Binary availability mask I_k(c) for c=1..C."""
        C = len(GLOBAL_ORGANS)
        avail = [0] * C
        for _, gids in self.mapping.items():
            for g in gids:
                if 1 <= g <= C:
                    avail[g - 1] = 1
        return avail

def dataset_to_global_mapping(dataset_name: str) -> LabelMap:
    """Return dataset-specific label mapping aligned with the paper setup.

    - MSD tasks typically use labels {0: background, 1: organ, 2: tumor}. We keep organ only.
    - KiTS19 uses labels {0: background, 1: kidney, 2: tumor}. We keep kidney only.
      KiTS19 does not separate left/right kidneys, while BTCV does.
      We supervise BOTH kidney classes with the same kidney mask (best-effort alignment).
    """
    dn = dataset_name.lower()

    if dn == "btcv_multi":
        # BTCV uses 1..13 mapping matching GLOBAL_ORGANS order.
        return LabelMap(dataset_name, {i: [i] for i in range(1, 14)})

    if dn == "msd_spleen":
        return LabelMap(dataset_name, {1: [NAME_TO_GLOBAL_ID["spleen"]]})

    if dn == "msd_liver":
        return LabelMap(dataset_name, {1: [NAME_TO_GLOBAL_ID["liver"]]})

    if dn == "msd_pancreas":
        return LabelMap(dataset_name, {1: [NAME_TO_GLOBAL_ID["pancreas"]]})

    if dn == "kits19_kidney":
        rk = NAME_TO_GLOBAL_ID["right_kidney"]
        lk = NAME_TO_GLOBAL_ID["left_kidney"]
        return LabelMap(dataset_name, {1: [rk, lk]})

    raise ValueError(f"Unknown dataset_name: {dataset_name}")
