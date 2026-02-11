from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import monai
from monai.data import Dataset, DataLoader
from .transforms import build_preprocess_transforms, build_train_aug_transforms, build_val_transforms
from .label_maps import dataset_to_global_mapping

def _load_case_list(list_path: Path) -> Dict[str, Any]:
    return json.loads(list_path.read_text())

def build_client_loaders(
    dataset_name: str,
    data_root: str,
    target_spacing,
    patch_size,
    batch_size: int,
    num_workers: int,
    intensity_cfg: Dict[str, float],
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """Build MONAI loaders for a single client.

    Expects:
      datasets/<name>/splits/split.json with keys train/val (and optional test)
      datasets/<name>/preprocessed/imagesTr + labelsTr (or a user-defined JSON list)
    """
    root = Path(data_root)
    split_path = root / "splits" / "split.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}. Create it or run preprocessing.")

    split = _load_case_list(split_path)
    def mk_items(case_ids):
        items = []
        for cid in case_ids:
            items.append({
                "image": str(root / "preprocessed" / "images" / f"{cid}.nii.gz"),
                "label": str(root / "preprocessed" / "labels" / f"{cid}.nii.gz"),
                "case_id": cid,
            })
        return items

    train_items = mk_items(split["train"])
    val_items = mk_items(split["val"])

    pre = build_preprocess_transforms(target_spacing, intensity_cfg)
    aug = build_train_aug_transforms(patch_size)
    val_t = build_val_transforms(patch_size)

    train_ds = Dataset(train_items, transform=monai.transforms.Compose([pre, aug]))
    val_ds = Dataset(val_items, transform=monai.transforms.Compose([pre, val_t]))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    lm = dataset_to_global_mapping(dataset_name)
    meta = {
        "dataset_name": dataset_name,
        "label_map": lm,
        "availability_mask": lm.availability_mask(),
    }
    return train_loader, val_loader, meta
