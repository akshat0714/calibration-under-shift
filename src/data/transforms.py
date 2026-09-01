"""Train-only augmentation and deterministic evaluation transforms."""

from __future__ import annotations

from collections.abc import Sequence

from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERPOLATION_MODES = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


def preprocessing_from_model_config(model_config: dict) -> dict:
    """Validate and return the preprocessing contract saved with a checkpoint."""

    mean = tuple(float(value) for value in model_config.get("mean", IMAGENET_MEAN))
    std = tuple(float(value) for value in model_config.get("std", IMAGENET_STD))
    interpolation = str(model_config.get("interpolation", "bilinear")).lower()
    crop_pct = float(model_config.get("crop_pct", 0.875))
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("model mean/std must contain three values and std must be positive")
    if interpolation not in INTERPOLATION_MODES:
        raise ValueError(f"interpolation must be one of {sorted(INTERPOLATION_MODES)}")
    if not 0 < crop_pct <= 1:
        raise ValueError("crop_pct must be in (0, 1]")
    return {
        "mean": mean,
        "std": std,
        "interpolation": interpolation,
        "crop_pct": crop_pct,
    }


def build_transform(
    train: bool,
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
    interpolation: str = "bilinear",
    crop_pct: float = 0.875,
):
    """Build conservative microscopy transforms.

    Random augmentation is present only when ``train=True``. Evaluation preserves
    aspect ratio by resizing the shorter side followed by a center crop.
    """

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if interpolation not in INTERPOLATION_MODES:
        raise ValueError(f"interpolation must be one of {sorted(INTERPOLATION_MODES)}")
    if not 0 < crop_pct <= 1:
        raise ValueError("crop_pct must be in (0, 1]")
    interpolation_mode = INTERPOLATION_MODES[interpolation]
    normalize = transforms.Normalize(mean=mean, std=std)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.85, 1.0),
                    interpolation=interpolation_mode,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    resize_to = round(image_size / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(resize_to, interpolation=interpolation_mode, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
