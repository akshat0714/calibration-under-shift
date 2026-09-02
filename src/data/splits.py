"""Leakage-resistant, deterministic split-manifest generation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split

STANDARD_FRACTIONS = {"train": 0.60, "val": 0.15, "calibration": 0.10, "test": 0.15}


def _distribution_score(labels: np.ndarray, selected: np.ndarray, target_fraction: float) -> float:
    classes = np.unique(labels)
    overall = np.array([(labels == value).mean() for value in classes])
    chosen = labels[selected]
    local = np.array([(chosen == value).mean() for value in classes])
    size_error = abs(len(selected) / len(labels) - target_fraction)
    return float(np.abs(local - overall).mean() + 2.0 * size_error)


def _best_group_holdout(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
    label_col: str,
    group_col: str,
    trials: int = 256,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the best of deterministic group holdouts by class/size balance."""

    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    if frame[group_col].nunique() < 4:
        raise ValueError("at least four distinct groups are required")
    labels = frame[label_col].to_numpy()
    splitter = GroupShuffleSplit(n_splits=trials, test_size=fraction, random_state=seed)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for keep_idx, hold_idx in splitter.split(frame, labels, groups=frame[group_col]):
        score = _distribution_score(labels, hold_idx, fraction)
        if best is None or score < best[0]:
            best = (score, keep_idx, hold_idx)
    assert best is not None
    return frame.iloc[best[1]].copy(), frame.iloc[best[2]].copy()


def grouped_stratified_manifest(
    metadata: pd.DataFrame,
    fractions: Mapping[str, float] = STANDARD_FRACTIONS,
    seed: int = 2025,
    label_col: str = "label",
    group_col: str = "patient_id",
    path_col: str = "path",
) -> pd.DataFrame:
    """Create patient-disjoint train/val/calibration/test rows.

    GroupShuffleSplit cannot natively stratify heterogeneous patient groups. The
    implementation samples many seeded group-only candidates at each step and retains
    the candidate with the closest class and size distribution.
    """

    required = {label_col, group_col, path_col}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")
    expected = {"train", "val", "calibration", "test"}
    values = np.asarray(list(fractions.values()), dtype=float)
    if (
        set(fractions) != expected
        or not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError(f"fractions must contain {sorted(expected)}, be positive, and sum to one")
    if metadata[path_col].duplicated().any():
        raise ValueError("input metadata contains duplicate paths")

    remaining = metadata.copy()
    assignments: list[pd.DataFrame] = []
    remaining_fraction = 1.0
    for offset, name in enumerate(("test", "calibration", "val")):
        relative = fractions[name] / remaining_fraction
        remaining, selected = _best_group_holdout(
            remaining,
            relative,
            seed + offset,
            label_col,
            group_col,
        )
        selected["split"] = name
        assignments.append(selected)
        remaining_fraction -= fractions[name]
    remaining["split"] = "train"
    assignments.append(remaining)
    output = pd.concat(assignments, ignore_index=True)
    output = output.rename(columns={path_col: "path", label_col: "label", group_col: "patient_id"})
    _assert_disjoint(output, group_col="patient_id")
    return output.sort_values(["split", "patient_id", "path"]).reset_index(drop=True)


def stratified_manifest(
    metadata: pd.DataFrame,
    fractions: Mapping[str, float] = {
        "train": 0.70,
        "val": 0.10,
        "calibration": 0.10,
        "test": 0.10,
    },
    seed: int = 2025,
    label_col: str = "label",
    path_col: str = "path",
) -> pd.DataFrame:
    """Create a four-way sample-level stratified split for non-grouped data."""

    expected = {"train", "val", "calibration", "test"}
    values = np.asarray(list(fractions.values()), dtype=float)
    if (
        set(fractions) != expected
        or not np.isfinite(values).all()
        or (values <= 0).any()
        or not np.isclose(values.sum(), 1.0)
    ):
        raise ValueError(f"fractions must contain {sorted(expected)}, be positive, and sum to one")
    work = metadata.copy().rename(columns={path_col: "path", label_col: "label"})
    if work["path"].duplicated().any():
        raise ValueError("input metadata contains duplicate paths")
    remaining = work
    chunks: list[pd.DataFrame] = []
    remaining_fraction = 1.0
    for offset, name in enumerate(("test", "calibration", "val")):
        relative = fractions[name] / remaining_fraction
        keep, chosen = train_test_split(
            remaining,
            test_size=relative,
            random_state=seed + offset,
            stratify=remaining["label"],
        )
        chosen = chosen.copy()
        chosen["split"] = name
        chunks.append(chosen)
        remaining = keep
        remaining_fraction -= fractions[name]
    remaining = remaining.copy()
    remaining["split"] = "train"
    chunks.append(remaining)
    return pd.concat(chunks, ignore_index=True).sort_values(["split", "label", "path"])


def cross_validation_manifest(
    metadata: pd.DataFrame,
    folds: int = 5,
    calibration_fraction: float = 0.10,
    seed: int = 2025,
    label_col: str = "label",
    path_col: str = "path",
) -> pd.DataFrame:
    """Create stratified CV folds with disjoint train/val/calibration/test roles.

    For each outer fold, the held-out fold is test. A second stratified split of
    the remaining rows creates validation and calibration subsets. Neither is
    reused for model fitting.
    """

    if not 0 < calibration_fraction < 0.25:
        raise ValueError("calibration_fraction must be between zero and 0.25")
    work = metadata.copy().rename(columns={path_col: "path", label_col: "label"})
    if work["path"].duplicated().any():
        raise ValueError("input metadata contains duplicate paths")
    outer = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    outputs: list[pd.DataFrame] = []
    for fold, (development_idx, test_idx) in enumerate(outer.split(work, work["label"])):
        development = work.iloc[development_idx]
        test = work.iloc[test_idx].copy()
        # Allocate 15% of total data to validation and 10% to calibration.
        nontrain_fraction = min(0.40, 0.25 / (1.0 - 1.0 / folds))
        train, held = train_test_split(
            development,
            test_size=nontrain_fraction,
            random_state=seed + fold,
            stratify=development["label"],
        )
        calibration_relative = calibration_fraction / 0.25
        val, calibration = train_test_split(
            held,
            test_size=calibration_relative,
            random_state=seed + 100 + fold,
            stratify=held["label"],
        )
        for name, frame in (
            ("train", train),
            ("val", val),
            ("calibration", calibration),
            ("test", test),
        ):
            part = frame.copy()
            part["split"] = name
            part["fold"] = fold
            outputs.append(part)
    return pd.concat(outputs, ignore_index=True).sort_values(["fold", "split", "path"])


def kromp_high_quality_label(
    frame: pd.DataFrame,
    expansion_col: str,
    icm_col: str,
    te_col: str,
    min_expansion: int = 3,
    expansion_offset: int = 0,
) -> pd.Series:
    """Map Gardner grades to the prespecified binary quality task."""

    expansion = pd.to_numeric(frame[expansion_col], errors="coerce") + expansion_offset

    def normalize_grade(value) -> str:
        text = str(value).strip().upper()
        numeric_mapping = {
            "0": "A",
            "0.0": "A",
            "1": "B",
            "1.0": "B",
            "2": "C",
            "2.0": "C",
            "3": "ND",
            "3.0": "ND",
        }
        return numeric_mapping.get(text, text)

    icm = frame[icm_col].map(normalize_grade)
    te = frame[te_col].map(normalize_grade)
    invalid_expansion = expansion.isna() | (expansion % 1 != 0) | ~expansion.between(1, 6)
    missing = expansion.isna() | ~icm.isin({"A", "B", "C"}) | ~te.isin({"A", "B", "C"})
    invalid = invalid_expansion | missing
    if invalid.any():
        raise ValueError(f"invalid or missing Gardner grades in {int(invalid.sum())} rows")
    return ((expansion >= min_expansion) & icm.isin({"A", "B"}) & te.isin({"A", "B"})).astype(int)


def _assert_disjoint(frame: pd.DataFrame, group_col: str | None = None) -> None:
    if frame.groupby("path")["split"].nunique().max() > 1:
        raise AssertionError("path leakage across splits")
    if group_col is not None and frame.groupby(group_col)["split"].nunique().max() > 1:
        raise AssertionError(f"{group_col} leakage across splits")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strategy", choices=("grouped", "stratified", "cv"), required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--group-col", default="patient_id")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = pd.read_csv(args.metadata)
    if args.strategy == "grouped":
        manifest = grouped_stratified_manifest(
            metadata,
            seed=args.seed,
            path_col=args.path_col,
            label_col=args.label_col,
            group_col=args.group_col,
        )
    elif args.strategy == "cv":
        manifest = cross_validation_manifest(
            metadata,
            folds=args.folds,
            seed=args.seed,
            path_col=args.path_col,
            label_col=args.label_col,
        )
    else:
        manifest = stratified_manifest(
            metadata, seed=args.seed, path_col=args.path_col, label_col=args.label_col
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"wrote {len(manifest)} rows to {args.output}")


if __name__ == "__main__":
    main()
