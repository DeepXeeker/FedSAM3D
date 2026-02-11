from __future__ import annotations
from typing import Sequence, Dict
import monai
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, NormalizeIntensityd, CropForegroundd,
    SpatialPadd, RandSpatialCropd, RandFlipd, RandAffined,
    RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
    EnsureTyped
)

def build_preprocess_transforms(
    target_spacing: Sequence[float],
    intensity: Dict[str, float],
) -> Compose:
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=target_spacing, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=float(intensity["a_min"]),
            a_max=float(intensity["a_max"]),
            b_min=float(intensity["b_min"]),
            b_max=float(intensity["b_max"]),
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"]),
    ])

def build_train_aug_transforms(patch_size: Sequence[int]) -> Compose:
    return Compose([
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandAffined(
            keys=["image", "label"],
            prob=0.3,
            rotate_range=(0.10, 0.10, 0.10),
            scale_range=(0.10, 0.10, 0.10),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
        EnsureTyped(keys=["image", "label"]),
    ])

def build_val_transforms(patch_size: Sequence[int]) -> Compose:
    return Compose([
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        EnsureTyped(keys=["image", "label"]),
    ])
