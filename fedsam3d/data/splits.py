from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
import random

def make_patient_split(case_ids: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    case_ids = list(case_ids)
    rng.shuffle(case_ids)
    n_val = int(round(len(case_ids) * val_ratio))
    val_ids = case_ids[:n_val]
    train_ids = case_ids[n_val:]
    return train_ids, val_ids

def save_split_json(split_path: Path, train_ids: List[str], val_ids: List[str], test_ids: List[str] | None = None) -> None:
    payload: Dict[str, List[str]] = {"train": train_ids, "val": val_ids}
    if test_ids is not None:
        payload["test"] = test_ids
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2))

def load_split_json(split_path: Path) -> Dict[str, List[str]]:
    return json.loads(split_path.read_text())
