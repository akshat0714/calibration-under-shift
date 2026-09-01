"""Verify an isolated release reproduction against the committed reference results."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.run_grid import KEY_COLUMNS
from experiments.run_stage2_matrix import EXPECTED_DETAIL_JSONS, EXPECTED_TOTAL_ROWS

VALUE_ATOL = 1e-6
VALUE_RTOL = 1e-5
EXPECTED_FIGURES = {
    "f1_headline",
    *(
        f"f1_appendix_{name}"
        for name in (
            "defocus_blur",
            "motion_blur",
            "gaussian_noise",
            "shot_noise",
            "jpeg",
            "resample",
            "illumination",
        )
    ),
    "f2_reliability",
    "f3_risk_coverage",
    "f4_failure_detection_auroc",
    "f5_conformal",
    "f6_corruption_grid",
    "f7_attribution_grid",
    "f7_attribution_stability_accuracy",
}
FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"/Users/"),
    re.compile(r"/home/[^/]+/"),
    re.compile(r"/content/"),
    re.compile(r"[A-Za-z]:\\"),
)


class ReproductionError(RuntimeError):
    """Raised when a release reproduction does not match the frozen result."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_file(root: Path, raw_path: Any, *, label: str) -> Path:
    text = str(raw_path)
    path = Path(text)
    if path.is_absolute():
        raise ReproductionError(f"{label} contains an absolute path: {text}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ReproductionError(f"{label} escapes the repository: {text}") from error
    if not resolved.is_file():
        raise ReproductionError(f"{label} is missing: {text}")
    return resolved


def _reject_host_paths(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "path" and isinstance(item, str) and Path(item).is_absolute():
                raise ReproductionError(f"{label} contains an absolute path: {item}")
            _reject_host_paths(item, label=label)
    elif isinstance(value, list | tuple):
        for item in value:
            _reject_host_paths(item, label=label)
    elif isinstance(value, str) and any(
        pattern.search(value) for pattern in FORBIDDEN_PATH_PATTERNS
    ):
        raise ReproductionError(f"{label} contains a host-specific path: {value}")


def verify_split_integrity(root: Path) -> dict[str, Any]:
    """Reconfirm the frozen SMIDS split and rotating HuSHeM folds against raw data."""

    smids = pd.read_csv(root / "data/splits/smids.csv")
    if len(smids) != 3_000 or not smids["path"].is_unique:
        raise ReproductionError("SMIDS split must contain 3,000 unique paths")
    expected_smids = {"train": 2_100, "val": 300, "calibration": 300, "test": 300}
    if smids["split"].value_counts().to_dict() != expected_smids:
        raise ReproductionError("SMIDS split counts differ from the frozen 70/10/10/10 split")
    smids_root = root / "data/raw/smids/files"
    missing_smids = [path for path in smids["path"] if not (smids_root / path).is_file()]
    if missing_smids:
        raise ReproductionError(f"SMIDS raw files are missing; first={missing_smids[0]}")

    hushem = pd.read_csv(root / "data/splits/hushem.csv")
    if len(hushem) != 1_080 or set(hushem["fold"]) != set(range(5)):
        raise ReproductionError("HuSHeM manifest must contain five complete 216-image folds")
    expected_hushem = {
        0: {"train": 118, "val": 32, "calibration": 22, "test": 44},
        1: {"train": 118, "val": 33, "calibration": 22, "test": 43},
        2: {"train": 118, "val": 33, "calibration": 22, "test": 43},
        3: {"train": 118, "val": 33, "calibration": 22, "test": 43},
        4: {"train": 118, "val": 33, "calibration": 22, "test": 43},
    }
    for fold, expected in expected_hushem.items():
        selected = hushem.loc[hushem["fold"] == fold]
        if len(selected) != 216 or not selected["path"].is_unique:
            raise ReproductionError(f"HuSHeM fold {fold} is not a disjoint 216-image partition")
        if selected["split"].value_counts().to_dict() != expected:
            raise ReproductionError(f"HuSHeM fold {fold} role counts differ from the frozen split")
    test_counts = hushem.loc[hushem["split"] == "test", "path"].value_counts()
    if len(test_counts) != 216 or not test_counts.eq(1).all():
        raise ReproductionError("every HuSHeM image must serve as outer test data exactly once")
    hushem_root = root / "data/raw/hushem/files"
    missing_hushem = [
        path for path in hushem["path"].unique() if not (hushem_root / path).is_file()
    ]
    if missing_hushem:
        raise ReproductionError(f"HuSHeM raw files are missing; first={missing_hushem[0]}")

    return {
        "smids": {"unique_images": 3_000, "roles": expected_smids},
        "hushem": {"unique_images": 216, "folds": 5, "outer_test_appearances": 1},
    }


def _read_metrics(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        low_memory=False,
        dtype={"seed": "string", "fold": "string", "evaluation_git_revision": "string"},
    )


def _normalized_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted((set(KEY_COLUMNS) | {"value", "evaluation_git_revision"}) - set(frame))
    if missing:
        raise ReproductionError(f"metrics are missing columns: {missing}")
    output = frame.copy()
    output["seed"] = output["seed"].fillna("")
    output["fold"] = output["fold"].fillna("")
    output["severity"] = output["severity"].astype(int)
    keys = output[KEY_COLUMNS].copy()
    if keys.duplicated(keep=False).any():
        raise ReproductionError("metrics contain duplicate tidy keys")
    numeric = pd.to_numeric(output["value"], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ReproductionError("metrics contain missing or non-finite values")
    output["value"] = numeric
    return output.sort_values(KEY_COLUMNS, kind="stable").reset_index(drop=True)


def verify_metrics(reference_path: Path, reproduced_path: Path) -> dict[str, Any]:
    reference = _normalized_metrics(_read_metrics(reference_path))
    reproduced = _normalized_metrics(_read_metrics(reproduced_path))
    if len(reference) != EXPECTED_TOTAL_ROWS or len(reproduced) != EXPECTED_TOTAL_ROWS:
        raise ReproductionError(
            f"metrics row count must be {EXPECTED_TOTAL_ROWS}; "
            f"reference={len(reference)}, reproduced={len(reproduced)}"
        )
    if set(reproduced["dataset"]) != {"smids", "hushem"}:
        raise ReproductionError("reproduced metrics must contain only SMIDS and HuSHeM")
    forbidden = reproduced.astype(str).apply(
        lambda column: column.str.contains("pilot|synthetic_demo|kromp", case=False, regex=True)
    )
    if forbidden.any().any():
        raise ReproductionError("Kromp, pilot, or synthetic-demo rows entered scientific metrics")

    pd.testing.assert_frame_equal(
        reference[KEY_COLUMNS], reproduced[KEY_COLUMNS], check_dtype=False, check_names=True
    )
    provenance_columns = [
        column
        for column in reference.columns
        if column not in {*KEY_COLUMNS, "value", "evaluation_git_revision"}
    ]
    pd.testing.assert_frame_equal(
        reference[provenance_columns],
        reproduced[provenance_columns],
        check_dtype=False,
        check_names=True,
    )
    if not np.allclose(
        reference["value"].to_numpy(),
        reproduced["value"].to_numpy(),
        atol=VALUE_ATOL,
        rtol=VALUE_RTOL,
    ):
        difference = np.abs(reference["value"].to_numpy() - reproduced["value"].to_numpy())
        index = int(np.argmax(difference))
        raise ReproductionError(
            "reproduced metric values exceed the frozen numerical tolerance; "
            f"max_abs_error={difference[index]:.9g}, key={reference.iloc[index][KEY_COLUMNS].to_dict()}"
        )
    reference_revisions = sorted(reference["evaluation_git_revision"].dropna().unique().tolist())
    reproduced_revisions = sorted(reproduced["evaluation_git_revision"].dropna().unique().tolist())
    if len(reference_revisions) != 1 or len(reproduced_revisions) != 1:
        raise ReproductionError(
            "reference and reproduced metrics each require one committed revision"
        )
    return {
        "rows": len(reproduced),
        "reference_revision": reference_revisions[0],
        "reproduction_revision": reproduced_revisions[0],
        "value_atol": VALUE_ATOL,
        "value_rtol": VALUE_RTOL,
        "max_abs_error": float(
            np.max(np.abs(reference["value"].to_numpy() - reproduced["value"].to_numpy()))
        ),
    }


def verify_thresholds(reference_path: Path, reproduced_path: Path) -> dict[str, Any]:
    reference = (
        pd.read_csv(reference_path)
        .sort_values(["dataset", "model", "signal_method", "signal_metric"], kind="stable")
        .reset_index(drop=True)
    )
    reproduced = (
        pd.read_csv(reproduced_path)
        .sort_values(["dataset", "model", "signal_method", "signal_metric"], kind="stable")
        .reset_index(drop=True)
    )
    if len(reference) != 16 or len(reproduced) != 16:
        raise ReproductionError("threshold analysis must contain exactly 16 signal rows")
    identity = ["dataset", "model", "signal_method", "signal_metric"]
    exact = [
        "signal_crossing_severity",
        "accuracy_drop_severity",
        "early_warning_gap",
        "signal_is_earlier",
    ]
    pd.testing.assert_frame_equal(
        reference[identity + exact], reproduced[identity + exact], check_dtype=False
    )
    for column in ("clean_signal", "clean_accuracy"):
        if not np.allclose(
            reference[column].to_numpy(),
            reproduced[column].to_numpy(),
            atol=VALUE_ATOL,
            rtol=VALUE_RTOL,
            equal_nan=True,
        ):
            raise ReproductionError(f"threshold baseline column differs beyond tolerance: {column}")
    paired = (
        reproduced["signal_crossing_severity"].notna()
        & reproduced["accuracy_drop_severity"].notna()
    )
    earlier = reproduced.loc[paired, "signal_is_earlier"].fillna(False).astype(bool)
    signal_never = int(reproduced["signal_crossing_severity"].isna().sum())
    if int(paired.sum()) != 10 or int(earlier.sum()) != 0 or signal_never != 6:
        raise ReproductionError(
            "approved null changed: expected 10 paired crossings, 0 early, and 6 signal-never"
        )
    return {"rows": 16, "paired_crossings": 10, "early_crossings": 0, "signal_never": 6}


def verify_details(root: Path, details_dir: Path, revision: str) -> dict[str, Any]:
    paths = sorted(details_dir.glob("*.json"))
    if len(paths) != EXPECTED_DETAIL_JSONS:
        raise ReproductionError(
            f"expected {EXPECTED_DETAIL_JSONS} evaluation details, found {len(paths)}"
        )
    calibration_records = 0
    for path in paths:
        detail = json.loads(path.read_text(encoding="utf-8"))
        _reject_host_paths(detail, label=str(path.relative_to(root)))
        if detail.get("evaluation_git_revision") != revision:
            raise ReproductionError(f"detail revision mismatch: {path.name}")
        fit = detail.get("calibration_fit", {})
        required = (
            ("ensemble_aps",)
            if detail.get("kind") == "deep_ensemble_evaluation"
            else (
                "temperature",
                "vector_scaling",
                "aps",
            )
        )
        for method in required:
            if fit.get(method, {}).get("split") != "calibration":
                raise ReproductionError(f"{path.name} did not fit {method} on calibration only")
            calibration_records += 1
    return {"json_files": len(paths), "calibration_only_fit_records": calibration_records}


def verify_figure_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _reject_host_paths(manifest, label=str(manifest_path.relative_to(root)))
    figures = manifest.get("figures", {})
    if set(figures) != EXPECTED_FIGURES:
        raise ReproductionError(
            f"final figure set is not exact; missing={sorted(EXPECTED_FIGURES - set(figures))}, "
            f"extra={sorted(set(figures) - EXPECTED_FIGURES)}"
        )
    verified_hashes = 0

    def walk(value: Any, label: str) -> None:
        nonlocal verified_hashes
        if isinstance(value, Mapping):
            if "path" in value and "sha256" in value:
                path = _portable_file(root, value["path"], label=label)
                observed = _sha256(path)
                if observed != value["sha256"]:
                    raise ReproductionError(f"manifest SHA-256 mismatch for {value['path']}")
                verified_hashes += 1
            for key, item in value.items():
                walk(item, f"{label}.{key}")
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                walk(item, f"{label}[{index}]")

    walk(manifest, "figure_manifest")
    return {"figures": len(figures), "verified_hash_entries": verified_hashes}


def verify_source_guardrails(root: Path) -> dict[str, bool]:
    grid = (root / "experiments/run_grid.py").read_text(encoding="utf-8")
    evaluator = (root / "src/evaluate.py").read_text(encoding="utf-8")
    trainer = (root / "src/train.py").read_text(encoding="utf-8")
    checks = {
        "clean_calibration_inference": 'infer("calibration", "clean", 0)' in grid,
        "calibration_rows_labeled": 'np.repeat("calibration"' in grid,
        "training_corruption_rejected": "corruptions cannot be applied to the training split"
        in evaluator,
        "training_uses_train_transform": "build_transform(True" in trainer,
    }
    if not all(checks.values()):
        raise ReproductionError(f"source guardrail grep failed: {checks}")
    return checks


def verify_reproduction(
    *,
    root: Path,
    reproduction_root: Path,
    reference_metrics: Path,
    reference_thresholds: Path,
    timings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    reproduction_root = reproduction_root.resolve()
    expected_root = (root / "results/reproduction").resolve()
    if reproduction_root != expected_root:
        raise ReproductionError(
            f"reproduction output must be isolated at {expected_root}, found {reproduction_root}"
        )
    split_summary = verify_split_integrity(root)
    metrics_summary = verify_metrics(reference_metrics, reproduction_root / "metrics.csv")
    threshold_summary = verify_thresholds(
        reference_thresholds, reproduction_root / "thresholds.csv"
    )
    revision = metrics_summary["reproduction_revision"]
    detail_summary = verify_details(root, reproduction_root / "evaluation_details", revision)
    figure_summary = verify_figure_manifest(
        root, reproduction_root / "figure_data/final_figure_manifest.json"
    )
    source_guardrails = verify_source_guardrails(root)
    result = {
        "status": "passed",
        "metrics": metrics_summary,
        "thresholds": threshold_summary,
        "details": detail_summary,
        "figures": figure_summary,
        "splits": split_summary,
        "source_guardrails": source_guardrails,
        "timings": dict(timings or {}),
        "reference": {
            "metrics_sha256": _sha256(reference_metrics),
            "thresholds_sha256": _sha256(reference_thresholds),
        },
    }
    output = reproduction_root / "verification.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--reproduction-root", type=Path, default=Path("results/reproduction"))
    parser.add_argument("--reference-metrics", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--reference-thresholds", type=Path, default=Path("results/thresholds.csv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    result = verify_reproduction(
        root=args.root,
        reproduction_root=args.reproduction_root,
        reference_metrics=args.reference_metrics,
        reference_thresholds=args.reference_thresholds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
