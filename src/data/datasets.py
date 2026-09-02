"""Manifest-backed image datasets with explicit split provenance."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

REQUIRED_COLUMNS = {"path", "label", "split"}
ALLOWED_SPLITS = {"train", "val", "calibration", "test", "consensus_test"}


def load_manifest(path: str | Path, fold: int | None = None) -> pd.DataFrame:
    """Load and validate a split manifest.

    A manifest is immutable experiment input. Splits are not inferred here.
    Cross-validation manifests may contain one row per sample and fold. Callers must
    select exactly one fold before constructing a dataset.
    """

    manifest_path = Path(path)
    frame = pd.read_csv(manifest_path)
    return _validate_manifest_frame(frame, source=str(manifest_path), fold=fold)


def _validate_manifest_frame(
    frame: pd.DataFrame,
    source: str = "in-memory manifest",
    fold: int | None = None,
) -> pd.DataFrame:
    """Apply identical leakage/schema checks to file and in-memory manifests."""

    frame = frame.copy()
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"manifest {source} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"manifest is empty: {source}")
    if fold is not None:
        if "fold" not in frame.columns:
            raise ValueError("fold was requested but the manifest has no 'fold' column")
        frame = frame.loc[frame["fold"] == fold].copy()
        if frame.empty:
            raise ValueError(f"fold {fold} has no rows in {source}")
    elif "fold" in frame.columns and frame["fold"].nunique() > 1:
        raise ValueError("cross-validation manifest requires an explicit fold")

    unknown_splits = set(frame["split"].astype(str)) - ALLOWED_SPLITS
    if unknown_splits:
        raise ValueError(f"unknown split names: {sorted(unknown_splits)}")
    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    if numeric_labels.isna().any() or (numeric_labels < 0).any():
        raise ValueError("labels must be non-negative integers")
    if not np.allclose(numeric_labels, numeric_labels.astype(int)):
        raise ValueError("labels must be integers")
    frame["label"] = numeric_labels.astype(int)

    if frame["path"].isna().any() or frame["path"].astype(str).str.strip().eq("").any():
        raise ValueError("sample paths must be non-empty")
    duplicates = frame.groupby("path")["split"].nunique()
    leaked = duplicates[duplicates > 1]
    if not leaked.empty:
        examples = leaked.index[:3].tolist()
        raise ValueError(f"sample paths occur in multiple splits. Examples include {examples}")
    if frame["path"].duplicated().any():
        examples = frame.loc[frame["path"].duplicated(keep=False), "path"].head(3).tolist()
        raise ValueError(
            f"duplicate sample paths occur in one manifest view. Examples include {examples}"
        )
    if "patient_id" in frame.columns:
        if frame["patient_id"].isna().any():
            raise ValueError("patient_id must be non-missing when the column is present")
        patient_splits = frame.groupby("patient_id")["split"].nunique()
        leaking_patients = patient_splits[patient_splits > 1]
        if not leaking_patients.empty:
            examples = leaking_patients.index[:3].tolist()
            raise ValueError(f"patient leakage occurs across splits. Examples include {examples}")
    return frame.reset_index(drop=True)


def assert_fit_split(frame: pd.DataFrame, expected: str = "calibration") -> None:
    """Reject post-hoc fitting data that is not exclusively calibration data."""

    actual = set(frame["split"].astype(str).unique())
    if actual != {expected}:
        raise ValueError(f"expected only split={expected!r}, received {sorted(actual)}")


class ManifestImageDataset(Dataset):
    """Read RGB images listed in a validated manifest.

    Corruption is deliberately injected between image decoding and the evaluation
    transform. The constructor rejects corruption on the training split.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str,
        data_root: str | Path = ".",
        transform: Callable[[Image.Image], Any] | None = None,
        corruption: Callable[[Image.Image, int], Image.Image | np.ndarray] | None = None,
        fold: int | None = None,
    ) -> None:
        if isinstance(manifest, pd.DataFrame):
            frame = _validate_manifest_frame(manifest, fold=fold)
        else:
            frame = load_manifest(manifest, fold=fold)
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"manifest contains no rows for split={split!r}")
        if split == "train" and corruption is not None:
            raise ValueError("corruptions are evaluation-only and cannot be applied to train")
        self.split = split
        self.data_root = Path(data_root)
        self.transform = transform
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["path"]))
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        if self.corruption is not None:
            corrupted = self.corruption(image, index)
            image = (
                corrupted.convert("RGB")
                if isinstance(corrupted, Image.Image)
                else Image.fromarray(np.asarray(corrupted, dtype=np.uint8))
            )
        if self.transform is not None:
            image = self.transform(image)
        sample: dict[str, Any] = {
            "image": image,
            "label": int(row["label"]),
            "path": str(row["path"]),
            "split": str(row["split"]),
        }
        for column in ("sample_id", "patient_id", "fold"):
            if column in row and pd.notna(row[column]):
                sample[column] = row[column]
        return sample


def labels_for_split(manifest: str | Path, split: str, fold: int | None = None) -> np.ndarray:
    """Return manifest labels without decoding images."""

    frame = load_manifest(manifest, fold=fold)
    selected = frame.loc[frame["split"] == split, "label"]
    if selected.empty:
        raise ValueError(f"no labels for split={split!r}")
    return selected.to_numpy(dtype=np.int64)
