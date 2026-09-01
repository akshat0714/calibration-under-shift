"""Generate the prespecified Checkpoint-3 analysis artifacts.

This module is intentionally outcome-agnostic.  It validates a complete Stage-2
metric grid, independently recomputes the frozen threshold analysis, and refuses
to render artifacts when the supplied ``thresholds.csv`` disagrees.  Primary and
secondary outputs are kept separate so an exploratory pattern cannot silently
become the headline claim.

Run ``python -m experiments.checkpoint3 --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes

from experiments.analyze import threshold_analysis, validate_complete_grid
from src.viz.figures import DEFAULT_DPI

CANONICAL_REPLICATES: tuple[tuple[str, str, int], ...] = (
    ("smids", "resnet50", 5),
    ("smids", "xception", 3),
    ("smids", "mobilenet_v3_large", 3),
    ("hushem", "resnet50", 5),
)
CANONICAL_REPLICATE_IDENTITIES: Mapping[tuple[str, str], frozenset[tuple[str, str]]] = {
    ("smids", "resnet50"): frozenset((str(seed), "") for seed in range(2025, 2030)),
    ("smids", "xception"): frozenset((str(seed), "") for seed in range(2025, 2028)),
    ("smids", "mobilenet_v3_large"): frozenset((str(seed), "") for seed in range(2025, 2028)),
    ("hushem", "resnet50"): frozenset(("2025", str(fold)) for fold in range(5)),
}
CANONICAL_CORRUPTIONS = (
    "defocus_blur",
    "motion_blur",
    "gaussian_noise",
    "shot_noise",
    "jpeg",
    "resample",
    "illumination",
)
CANONICAL_SEVERITIES = (1, 2, 3, 4, 5)
FROZEN_THRESHOLDS: Mapping[str, float] = {
    "accuracy_absolute_drop": 0.05,
    "ece_relative_increase": 2.0,
    "ece_minimum_absolute_increase": 0.02,
    "entropy_relative_increase": 2.0,
    "entropy_minimum_absolute_increase": 0.05,
    "selective_risk_relative_increase": 2.0,
    "selective_risk_minimum_absolute_increase": 0.02,
    "conformal_target_coverage": 0.90,
    "conformal_allowed_shortfall": 0.05,
}

KEY_COLUMNS = (
    "dataset",
    "model",
    "seed",
    "fold",
    "corruption",
    "severity",
    "method",
    "metric",
    "value",
)
REPLICATE_COLUMNS = ("dataset", "model", "seed", "fold")
THRESHOLD_KEY_COLUMNS = ("dataset", "model", "signal_method", "signal_metric")
THRESHOLD_VALUE_COLUMNS = (
    "clean_signal",
    "signal_crossing_severity",
    "clean_accuracy",
    "accuracy_drop_severity",
    "early_warning_gap",
)


@dataclass(frozen=True)
class SignalSpec:
    method: str
    metric: str
    label: str


PRIMARY_SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec("raw_softmax", "ece", "Raw-softmax ECE"),
    SignalSpec(
        "raw_softmax",
        "mean_predictive_entropy",
        "Raw-softmax predictive entropy",
    ),
    SignalSpec(
        "raw_softmax",
        "risk_at_80_coverage",
        "Raw-softmax risk at 80% coverage",
    ),
    SignalSpec("aps", "conformal_coverage", "APS empirical coverage"),
)

BASE_REQUIRED_METRICS: Mapping[str, tuple[str, ...]] = {
    "raw_softmax": (
        "accuracy",
        "ece",
        "mean_predictive_entropy",
        "risk_at_80_coverage",
    ),
    "temperature": ("ece",),
    "aps": ("conformal_coverage", "conformal_mean_set_size"),
    "energy": ("risk_at_80_coverage",),
}
ENSEMBLE_REQUIRED_METRICS: Mapping[str, tuple[str, ...]] = {
    "deep_ensemble": ("mean_mutual_information", "risk_at_80_coverage"),
    "ensemble_aps": ("conformal_coverage", "conformal_mean_set_size"),
}

_GROUP_ORDER = {
    (dataset, model): index for index, (dataset, model, _count) in enumerate(CANONICAL_REPLICATES)
}
_SIGNAL_ORDER = {
    (signal.method, signal.metric): index for index, signal in enumerate(PRIMARY_SIGNALS)
}
_PRIMARY_STYLE = "#0072B2"
_SECONDARY_STYLE = "#D55E00"
_COVERAGE_STYLE = "#009E73"
_SHADE_STYLE = "#CC79A7"
_ACCURACY_MARKER_STYLE = "#222222"


def _identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _normalize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(KEY_COLUMNS).difference(metrics.columns))
    if missing:
        raise ValueError(f"metrics table is missing columns: {missing}")
    frame = metrics.copy()
    for column in ("dataset", "model", "corruption", "method", "metric"):
        if frame[column].isna().any():
            raise ValueError(f"metrics table has missing {column} identifiers")
        frame[column] = frame[column].astype(str)
    frame["seed"] = frame["seed"].map(_identifier)
    frame["fold"] = frame["fold"].map(_identifier)
    severity = pd.to_numeric(frame["severity"], errors="coerce")
    if severity.isna().any() or not np.allclose(severity, np.round(severity)):
        raise ValueError("metrics severity values must be finite integers")
    frame["severity"] = severity.astype(int)
    values = pd.to_numeric(frame["value"], errors="coerce")
    invalid = frame["value"].notna() & values.isna()
    if invalid.any():
        raise ValueError("metrics values must be numeric")
    frame["value"] = values.astype(float)
    return frame


def _normalize_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    required = set(THRESHOLD_KEY_COLUMNS) | set(THRESHOLD_VALUE_COLUMNS) | {"signal_is_earlier"}
    missing = sorted(required.difference(thresholds.columns))
    if missing:
        raise ValueError(f"threshold table is missing columns: {missing}")
    frame = thresholds.copy()
    for column in THRESHOLD_KEY_COLUMNS:
        if frame[column].isna().any():
            raise ValueError(f"threshold table has missing {column} identifiers")
        frame[column] = frame[column].astype(str)
    for column in THRESHOLD_VALUE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["signal_is_earlier"] = frame["signal_is_earlier"].map(_as_bool)
    return _sort_thresholds(frame)


def _as_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean value in thresholds.csv: {value!r}")


def _sort_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_group_order"] = [
        _GROUP_ORDER.get((row.dataset, row.model), len(_GROUP_ORDER))
        for row in ordered.itertuples(index=False)
    ]
    ordered["_signal_order"] = [
        _SIGNAL_ORDER.get((row.signal_method, row.signal_metric), len(_SIGNAL_ORDER))
        for row in ordered.itertuples(index=False)
    ]
    return (
        ordered.sort_values(["_group_order", "_signal_order"], kind="stable")
        .drop(columns=["_group_order", "_signal_order"])
        .reset_index(drop=True)
    )


def _expected_conditions(protocol: Mapping[str, Any]) -> set[tuple[str, int]]:
    aggregation = protocol["aggregation"]
    corruptions = [str(value) for value in aggregation["device_corruptions"]]
    severities = [int(value) for value in aggregation.get("severities", range(1, 6))]
    if len(corruptions) != len(set(corruptions)) or len(severities) != len(set(severities)):
        raise ValueError("frozen protocol contains duplicate corruptions or severities")
    return {("clean", 0)} | {
        (corruption, severity) for corruption in corruptions for severity in severities
    }


def _validate_frozen_protocol(protocol: Mapping[str, Any]) -> None:
    aggregation = protocol.get("aggregation", {})
    corruptions = tuple(str(value) for value in aggregation.get("device_corruptions", ()))
    severities = tuple(int(value) for value in aggregation.get("severities", ()))
    if corruptions != CANONICAL_CORRUPTIONS:
        raise ValueError(
            "analysis protocol does not match the frozen seven-corruption order; "
            f"observed={corruptions}"
        )
    if severities != CANONICAL_SEVERITIES:
        raise ValueError(
            f"analysis protocol does not match frozen severities 1 through 5; observed={severities}"
        )
    observed_thresholds = protocol.get("thresholds", {})
    for name, expected in FROZEN_THRESHOLDS.items():
        if name not in observed_thresholds or not np.isclose(
            float(observed_thresholds[name]), expected
        ):
            raise ValueError(
                f"analysis protocol threshold {name!r} differs from frozen value {expected}"
            )


def _base_replicate_keys(metrics: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    selected = metrics.loc[
        (metrics["method"] == "raw_softmax")
        & (metrics["metric"] == "accuracy")
        & (metrics["seed"] != "ensemble")
    ]
    return list(
        selected[list(REPLICATE_COLUMNS)].drop_duplicates().itertuples(index=False, name=None)
    )


def _condition_map(
    metrics: pd.DataFrame,
    method: str,
    metric: str,
    group_columns: Sequence[str],
) -> dict[tuple[Any, ...], list[tuple[str, int]]]:
    selected = metrics.loc[(metrics["method"] == method) & (metrics["metric"] == metric)]
    return {
        tuple(key) if isinstance(key, tuple) else (key,): list(
            zip(group["corruption"], group["severity"], strict=True)
        )
        for key, group in selected.groupby(list(group_columns), dropna=False, sort=False)
    }


def _validate_metric_conditions(
    metrics: pd.DataFrame,
    protocol: Mapping[str, Any],
    method: str,
    metric: str,
    keys: Sequence[tuple[Any, ...]],
    group_columns: Sequence[str],
) -> None:
    expected = _expected_conditions(protocol)
    conditions = _condition_map(metrics, method, metric, group_columns)
    for key in keys:
        pairs = conditions.get(tuple(key), [])
        relevant = [pair for pair in pairs if pair in expected]
        if len(relevant) != len(set(relevant)):
            raise ValueError(f"duplicate {method}/{metric} conditions for replicate {key}")
        actual = set(relevant)
        if actual != expected:
            missing = sorted(expected - actual)
            raise ValueError(
                f"incomplete {method}/{metric} grid for replicate {key}; missing={missing}"
            )
        selected = metrics.loc[
            (metrics["method"] == method)
            & (metrics["metric"] == metric)
            & metrics[list(group_columns)].eq(pd.Series(key, index=group_columns)).all(axis=1)
            & metrics[["corruption", "severity"]].apply(tuple, axis=1).isin(expected)
        ]
        if not np.isfinite(selected["value"]).all():
            raise ValueError(f"non-finite primary value in {method}/{metric} for replicate {key}")


def _validate_canonical_replicates(metrics: pd.DataFrame) -> None:
    base = metrics.loc[
        (metrics["method"] == "raw_softmax")
        & (metrics["metric"] == "accuracy")
        & (metrics["corruption"] == "clean")
        & (metrics["severity"] == 0)
        & (metrics["seed"] != "ensemble")
    ]
    counts = (
        base.groupby(["dataset", "model"], sort=False)[list(REPLICATE_COLUMNS)]
        .apply(lambda group: len(group.drop_duplicates()))
        .to_dict()
    )
    expected = {(dataset, model): count for dataset, model, count in CANONICAL_REPLICATES}
    if set(counts) != set(expected):
        raise ValueError(
            "metrics contain an unexpected canonical dataset/backbone matrix; "
            f"expected={sorted(expected)}, observed={sorted(counts)}"
        )
    mismatched = {
        key: (counts[key], expected[key]) for key in expected if counts[key] != expected[key]
    }
    if mismatched:
        raise ValueError(f"canonical replicate counts do not match Stage 1: {mismatched}")
    identities = {
        (dataset, model): frozenset(
            group[["seed", "fold"]].drop_duplicates().itertuples(index=False, name=None)
        )
        for (dataset, model), group in base.groupby(["dataset", "model"], sort=False)
    }
    identity_mismatches = {
        key: {
            "missing": sorted(CANONICAL_REPLICATE_IDENTITIES[key] - identities[key]),
            "extra": sorted(identities[key] - CANONICAL_REPLICATE_IDENTITIES[key]),
        }
        for key in CANONICAL_REPLICATE_IDENTITIES
        if identities.get(key) != CANONICAL_REPLICATE_IDENTITIES[key]
    }
    if identity_mismatches:
        raise ValueError(
            f"canonical seed/fold identities do not match Stage 1: {identity_mismatches}"
        )


def _validate_provenance(metrics: pd.DataFrame) -> None:
    required = {"checkpoint", "corruption_protocol_sha256", "n_samples"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"metrics table is missing provenance columns: {missing}")
    if (
        metrics["checkpoint"].isna().any()
        or (metrics["checkpoint"].astype(str).str.len() == 0).any()
    ):
        raise ValueError("metrics table has missing checkpoint provenance")
    digests = metrics["corruption_protocol_sha256"].dropna().astype(str)
    if len(digests) != len(metrics) or (digests.str.len() == 0).any():
        raise ValueError("metrics table has missing corruption-protocol provenance")
    if digests.nunique() != 1:
        raise ValueError("metrics table mixes multiple corruption protocols")
    samples = pd.to_numeric(metrics["n_samples"], errors="coerce")
    if samples.isna().any() or (samples <= 0).any() or not np.allclose(samples, np.round(samples)):
        raise ValueError("metrics n_samples provenance must contain positive integers")


def _ensemble_keys(metrics: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    selected = metrics.loc[
        (metrics["method"] == "deep_ensemble")
        & (metrics["metric"] == "accuracy")
        & (metrics["seed"] == "ensemble")
    ]
    return list(
        selected[list(REPLICATE_COLUMNS)].drop_duplicates().itertuples(index=False, name=None)
    )


def _validate_ensembles(metrics: pd.DataFrame, protocol: Mapping[str, Any]) -> None:
    ensemble_rows = metrics.loc[metrics["method"].isin(ENSEMBLE_REQUIRED_METRICS)]
    if (ensemble_rows["dataset"] == "hushem").any():
        raise ValueError("HuSHeM must not be represented as a same-fold deep ensemble")
    keys = _ensemble_keys(metrics)
    for method, metric_names in ENSEMBLE_REQUIRED_METRICS.items():
        for metric in metric_names:
            _validate_metric_conditions(
                metrics,
                protocol,
                method,
                metric,
                keys,
                REPLICATE_COLUMNS,
            )
    resnet_keys = [key for key in keys if key[:2] == ("smids", "resnet50")]
    if len(resnet_keys) != 1:
        raise ValueError("canonical SMIDS ResNet50 deep ensemble is missing or duplicated")
    checkpoint = metrics.loc[
        (metrics["dataset"] == "smids")
        & (metrics["model"] == "resnet50")
        & (metrics["seed"] == "ensemble")
        & (metrics["method"] == "deep_ensemble"),
        "checkpoint",
    ].astype(str)
    member_counts = checkpoint.map(_checkpoint_member_count).unique()
    if set(member_counts) != {5}:
        raise ValueError(
            "canonical SMIDS ResNet50 ensemble must contain five checkpoint members; "
            f"observed={sorted(member_counts)}"
        )


def _checkpoint_member_count(checkpoint: str) -> int:
    return len({item for item in str(checkpoint).split("|") if item})


def _validate_threshold_agreement(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> pd.DataFrame:
    observed = _normalize_thresholds(thresholds)
    expected = _normalize_thresholds(threshold_analysis(metrics, dict(protocol)))
    canonical_keys = {
        (dataset, model, signal.method, signal.metric)
        for dataset, model, _count in CANONICAL_REPLICATES
        for signal in PRIMARY_SIGNALS
    }
    observed_keys = set(observed[list(THRESHOLD_KEY_COLUMNS)].itertuples(index=False, name=None))
    expected_keys = set(expected[list(THRESHOLD_KEY_COLUMNS)].itertuples(index=False, name=None))
    if observed_keys != canonical_keys or expected_keys != canonical_keys:
        raise ValueError(
            "threshold table does not contain the canonical 16 primary rows; "
            f"observed={len(observed_keys)}, recomputed={len(expected_keys)}"
        )
    merged = observed.merge(
        expected,
        on=list(THRESHOLD_KEY_COLUMNS),
        suffixes=("_observed", "_expected"),
        validate="one_to_one",
    )
    for column in THRESHOLD_VALUE_COLUMNS:
        left = merged[f"{column}_observed"].to_numpy(dtype=float)
        right = merged[f"{column}_expected"].to_numpy(dtype=float)
        if not np.allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True):
            raise ValueError(f"thresholds.csv disagrees with frozen recomputation for {column}")
    if (
        merged["signal_is_earlier_observed"].tolist()
        != merged["signal_is_earlier_expected"].tolist()
    ):
        raise ValueError("thresholds.csv disagrees with frozen recomputation for earlier flags")
    return observed


def validate_checkpoint3_inputs(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized inputs after enforcing the complete canonical contract."""

    _validate_frozen_protocol(protocol)
    normalized = _normalize_metrics(metrics)
    _validate_provenance(normalized)
    validate_complete_grid(normalized, dict(protocol))
    _validate_canonical_replicates(normalized)
    base_keys = _base_replicate_keys(normalized)
    for method, metric_names in BASE_REQUIRED_METRICS.items():
        for metric in metric_names:
            _validate_metric_conditions(
                normalized,
                protocol,
                method,
                metric,
                base_keys,
                REPLICATE_COLUMNS,
            )
    _validate_ensembles(normalized, protocol)
    verified_thresholds = _validate_threshold_agreement(normalized, thresholds, protocol)
    return normalized, verified_thresholds


def _replicate_trajectory(
    metrics: pd.DataFrame,
    protocol: Mapping[str, Any],
    method: str,
    metric: str,
) -> pd.DataFrame:
    selected = metrics.loc[(metrics["method"] == method) & (metrics["metric"] == metric)].copy()
    corruptions = {str(value) for value in protocol["aggregation"]["device_corruptions"]}
    device = selected.loc[selected["corruption"].isin(corruptions)]
    shifted = (
        device.groupby([*REPLICATE_COLUMNS, "severity"], dropna=False, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "replicate_value"})
    )
    clean = (
        selected.loc[(selected["corruption"] == "clean") & (selected["severity"] == 0)]
        .groupby(list(REPLICATE_COLUMNS), dropna=False, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "replicate_value"})
    )
    clean["severity"] = 0
    return pd.concat([clean, shifted], ignore_index=True).sort_values(
        ["dataset", "model", "seed", "fold", "severity"], kind="stable"
    )


def _summarize_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["dataset", "model", "severity", "mean", "std", "n_replicates"])
    return (
        frame.groupby(["dataset", "model", "severity"], dropna=False)["replicate_value"]
        .agg(mean="mean", std="std", n_replicates="count")
        .reset_index()
    )


def _clean_stats(metrics: pd.DataFrame, method: str, metric: str) -> pd.DataFrame:
    trajectory = _replicate_trajectory(
        metrics, {"aggregation": {"device_corruptions": []}}, method, metric
    )
    clean = trajectory.loc[trajectory["severity"] == 0]
    return (
        clean.groupby(["dataset", "model"], dropna=False)["replicate_value"]
        .agg(mean="mean", std="std", n_replicates="count")
        .reset_index()
    )


def _status(signal_crossing: Any, accuracy_crossing: Any) -> str:
    signal_value = float(signal_crossing) if pd.notna(signal_crossing) else np.nan
    accuracy_value = float(accuracy_crossing) if pd.notna(accuracy_crossing) else np.nan
    signal_observed = np.isfinite(signal_value)
    accuracy_observed = np.isfinite(accuracy_value)
    if not signal_observed and not accuracy_observed:
        return "neither_crossed"
    if not signal_observed:
        return "signal_did_not_cross"
    if not accuracy_observed:
        return "accuracy_did_not_cross"
    return "earlier" if signal_value < accuracy_value else "same_or_later"


def build_main_table(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical 16-row primary table with replicate-aware clean summaries."""

    accuracy = _clean_stats(metrics, "raw_softmax", "accuracy").rename(
        columns={
            "mean": "clean_accuracy_mean",
            "std": "clean_accuracy_std",
            "n_replicates": "accuracy_n_replicates",
        }
    )
    rows: list[dict[str, Any]] = []
    signal_labels = {(item.method, item.metric): item.label for item in PRIMARY_SIGNALS}
    for threshold in _sort_thresholds(thresholds).itertuples(index=False):
        signal_stats = _clean_stats(metrics, threshold.signal_method, threshold.signal_metric)
        signal_row = signal_stats.loc[
            (signal_stats["dataset"] == threshold.dataset)
            & (signal_stats["model"] == threshold.model)
        ]
        accuracy_row = accuracy.loc[
            (accuracy["dataset"] == threshold.dataset) & (accuracy["model"] == threshold.model)
        ]
        if len(signal_row) != 1 or len(accuracy_row) != 1:
            raise ValueError("clean summary is missing or duplicated for a primary threshold row")
        signal_record = signal_row.iloc[0]
        accuracy_record = accuracy_row.iloc[0]
        signal_n = int(signal_record["n_replicates"])
        accuracy_n = int(accuracy_record["accuracy_n_replicates"])
        if signal_n != accuracy_n:
            raise ValueError(
                f"signal and accuracy replicate counts differ for {threshold.dataset}/{threshold.model}"
            )
        if not np.isclose(float(signal_record["mean"]), float(threshold.clean_signal)):
            raise ValueError("clean signal mean disagrees with verified thresholds")
        if not np.isclose(
            float(accuracy_record["clean_accuracy_mean"]), float(threshold.clean_accuracy)
        ):
            raise ValueError("clean accuracy mean disagrees with verified thresholds")
        rows.append(
            {
                "analysis_tier": "primary/prespecified",
                "dataset": threshold.dataset,
                "model": threshold.model,
                "signal_method": threshold.signal_method,
                "signal_metric": threshold.signal_metric,
                "signal_label": signal_labels[(threshold.signal_method, threshold.signal_metric)],
                "n_replicates": signal_n,
                "clean_signal_mean": float(signal_record["mean"]),
                "clean_signal_std": float(signal_record["std"]),
                "signal_crossing_severity": threshold.signal_crossing_severity,
                "clean_accuracy_mean": float(accuracy_record["clean_accuracy_mean"]),
                "clean_accuracy_std": float(accuracy_record["clean_accuracy_std"]),
                "accuracy_drop_severity": threshold.accuracy_drop_severity,
                "early_warning_gap": threshold.early_warning_gap,
                "status": _status(
                    threshold.signal_crossing_severity,
                    threshold.accuracy_drop_severity,
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 16:
        raise ValueError(f"main table must contain 16 rows, found {len(frame)}")
    return frame


def _per_corruption_table(metrics: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    selected_metrics = (
        ("raw_softmax", "accuracy"),
        ("raw_softmax", "ece"),
        ("raw_softmax", "mean_predictive_entropy"),
        ("raw_softmax", "risk_at_80_coverage"),
        ("aps", "conformal_coverage"),
        ("aps", "conformal_mean_set_size"),
    )
    corruptions = {str(item) for item in protocol["aggregation"]["device_corruptions"]}
    parts: list[pd.DataFrame] = []
    for method, metric in selected_metrics:
        selected = metrics.loc[
            (metrics["seed"] != "ensemble")
            & (metrics["method"] == method)
            & (metrics["metric"] == metric)
            & metrics["corruption"].isin(corruptions)
        ]
        summary = (
            selected.groupby(["dataset", "model", "corruption", "severity"], dropna=False)["value"]
            .agg(mean="mean", std="std", n_replicates="count")
            .reset_index()
        )
        summary["method"] = method
        summary["metric"] = metric
        parts.append(summary)
    output = pd.concat(parts, ignore_index=True)
    output.insert(0, "analysis_tier", "secondary/exploratory")
    return output.sort_values(
        ["dataset", "model", "method", "metric", "corruption", "severity"],
        kind="stable",
    ).reset_index(drop=True)


def _ece_transfer_table(metrics: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    raw = _replicate_trajectory(metrics, protocol, "raw_softmax", "ece").rename(
        columns={"replicate_value": "raw_ece"}
    )
    temperature = _replicate_trajectory(metrics, protocol, "temperature", "ece").rename(
        columns={"replicate_value": "temperature_ece"}
    )
    paired = raw.merge(
        temperature,
        on=[*REPLICATE_COLUMNS, "severity"],
        validate="one_to_one",
    )
    paired["temperature_minus_raw"] = paired["temperature_ece"] - paired["raw_ece"]
    summary = (
        paired.groupby(["dataset", "model", "severity"], dropna=False)
        .agg(
            raw_ece_mean=("raw_ece", "mean"),
            raw_ece_std=("raw_ece", "std"),
            temperature_ece_mean=("temperature_ece", "mean"),
            temperature_ece_std=("temperature_ece", "std"),
            paired_difference_mean=("temperature_minus_raw", "mean"),
            paired_difference_std=("temperature_minus_raw", "std"),
            n_replicates=("temperature_minus_raw", "count"),
        )
        .reset_index()
    )
    summary.insert(0, "analysis_tier", "secondary/exploratory")
    summary.insert(
        4,
        "corruption_scope",
        np.where(summary["severity"] == 0, "clean", "mean_of_device_corruptions"),
    )
    return summary


def _summary_for(
    metrics: pd.DataFrame,
    protocol: Mapping[str, Any],
    method: str,
    metric: str,
) -> pd.DataFrame:
    return _summarize_replicates(_replicate_trajectory(metrics, protocol, method, metric))


def _selective_table(metrics: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    selectors = (
        ("energy", "Energy score"),
        ("deep_ensemble", "Deep-ensemble predictive entropy"),
    )
    outputs: list[pd.DataFrame] = []
    for method, score_label in selectors:
        baseline_method = "deep_ensemble" if method == "deep_ensemble" else "raw_softmax"
        shifted_accuracy = _summary_for(metrics, protocol, baseline_method, "accuracy").rename(
            columns={
                "mean": "unselective_accuracy_mean",
                "std": "unselective_accuracy_std",
                "n_replicates": "unselective_n_replicates",
            }
        )
        clean_accuracy = shifted_accuracy.loc[
            shifted_accuracy["severity"] == 0,
            [
                "dataset",
                "model",
                "unselective_accuracy_mean",
                "unselective_accuracy_std",
                "unselective_n_replicates",
            ],
        ].rename(
            columns={
                "unselective_accuracy_mean": "clean_accuracy_mean",
                "unselective_accuracy_std": "clean_accuracy_std",
                "unselective_n_replicates": "clean_n_replicates",
            }
        )
        summary = _summary_for(metrics, protocol, method, "risk_at_80_coverage")
        summary = summary.loc[summary["severity"].isin([3, 4])].copy()
        if summary.empty:
            continue
        summary["retained_accuracy_mean"] = 1.0 - summary["mean"]
        summary["retained_accuracy_std"] = summary["std"]
        summary = summary.rename(columns={"n_replicates": "selector_n_replicates"})
        summary["selector_method"] = method
        summary["selector_score"] = score_label
        summary = summary.merge(
            shifted_accuracy.loc[shifted_accuracy["severity"].isin([3, 4])].drop(
                columns=["mean", "std"], errors="ignore"
            ),
            on=["dataset", "model", "severity"],
            how="left",
            validate="many_to_one",
        )
        summary = summary.merge(
            clean_accuracy,
            on=["dataset", "model"],
            how="left",
            validate="many_to_one",
        )
        summary["gain_vs_unselective"] = (
            summary["retained_accuracy_mean"] - summary["unselective_accuracy_mean"]
        )
        summary["difference_from_clean"] = (
            summary["retained_accuracy_mean"] - summary["clean_accuracy_mean"]
        )
        outputs.append(summary)
    if not outputs:
        raise ValueError("selective secondary analysis has no severity-3/4 rows")
    output = pd.concat(outputs, ignore_index=True)
    output.insert(0, "analysis_tier", "secondary/exploratory")
    columns = [
        "analysis_tier",
        "dataset",
        "model",
        "severity",
        "selector_method",
        "selector_score",
        "selector_n_replicates",
        "retained_accuracy_mean",
        "retained_accuracy_std",
        "unselective_n_replicates",
        "unselective_accuracy_mean",
        "unselective_accuracy_std",
        "clean_n_replicates",
        "clean_accuracy_mean",
        "clean_accuracy_std",
        "gain_vs_unselective",
        "difference_from_clean",
    ]
    return (
        output[columns]
        .sort_values(["dataset", "model", "selector_method", "severity"], kind="stable")
        .reset_index(drop=True)
    )


def _conformal_table(metrics: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for method in ("aps", "ensemble_aps"):
        coverage = _summary_for(metrics, protocol, method, "conformal_coverage").rename(
            columns={
                "mean": "coverage_mean",
                "std": "coverage_std",
                "n_replicates": "coverage_n_replicates",
            }
        )
        size = _summary_for(metrics, protocol, method, "conformal_mean_set_size").rename(
            columns={
                "mean": "set_size_mean",
                "std": "set_size_std",
                "n_replicates": "set_size_n_replicates",
            }
        )
        if coverage.empty and size.empty:
            continue
        merged = coverage.merge(
            size,
            on=["dataset", "model", "severity"],
            validate="one_to_one",
        )
        merged["method"] = method
        outputs.append(merged)
    output = pd.concat(outputs, ignore_index=True)
    output.insert(0, "analysis_tier", "secondary/exploratory")
    return output.sort_values(["dataset", "model", "method", "severity"], kind="stable")


def _ensemble_mi_table(metrics: pd.DataFrame, protocol: Mapping[str, Any]) -> pd.DataFrame:
    summary = _summary_for(metrics, protocol, "deep_ensemble", "mean_mutual_information")
    members = (
        metrics.loc[
            (metrics["method"] == "deep_ensemble")
            & (metrics["metric"] == "mean_mutual_information")
            & (metrics["corruption"] == "clean"),
            ["dataset", "model", "checkpoint"],
        ]
        .drop_duplicates()
        .assign(ensemble_members=lambda frame: frame["checkpoint"].map(_checkpoint_member_count))
        .drop(columns="checkpoint")
    )
    output = summary.merge(members, on=["dataset", "model"], validate="many_to_one")
    if (output["dataset"] == "hushem").any():
        raise ValueError("HuSHeM ensemble MI must remain unavailable by design")
    output.insert(0, "analysis_tier", "secondary/exploratory")
    return output.rename(
        columns={"mean": "mi_mean", "std": "mi_std", "n_replicates": "n_ensembles"}
    ).sort_values(["dataset", "model", "severity"], kind="stable")


def _severity_text(value: Any) -> str:
    if pd.isna(value):
        return "missing"
    number = float(value)
    return f"S{int(number)}" if number.is_integer() else f"S{number:g}"


def _group_scope(groups: Iterable[tuple[str, str]]) -> str:
    ordered = sorted(set(groups), key=lambda item: _GROUP_ORDER[item])
    canonical = {(dataset, model) for dataset, model, _count in CANONICAL_REPLICATES}
    if set(ordered) == canonical:
        return "all four dataset–backbone groups"
    return ", ".join(f"{dataset.upper()}/{model}" for dataset, model in ordered)


def _grouped_finding_details(frame: pd.DataFrame, status: str) -> str:
    selected = frame.loc[frame["status"] == status]
    if selected.empty:
        return ""
    labels = {(signal.method, signal.metric): signal.label for signal in PRIMARY_SIGNALS}
    grouping = [
        "signal_method",
        "signal_metric",
        "signal_crossing_severity",
        "accuracy_drop_severity",
    ]
    parts: list[str] = []
    for key, group in selected.groupby(grouping, dropna=False, sort=False):
        method, metric, signal_crossing, accuracy_crossing = key
        label = labels[(str(method), str(metric))]
        scope = _group_scope(zip(group["dataset"], group["model"], strict=True))
        if status == "signal_did_not_cross":
            parts.append(
                f"{label} (signal missing vs accuracy {_severity_text(accuracy_crossing)}) "
                f"in {scope}"
            )
            continue
        if status == "accuracy_did_not_cross":
            parts.append(
                f"{label} ({_severity_text(signal_crossing)} vs accuracy missing) in {scope}"
            )
            continue
        if status == "neither_crossed":
            parts.append(f"{label} (both crossings missing) in {scope}")
            continue
        parts.append(
            f"{label} ({_severity_text(signal_crossing)} vs accuracy "
            f"{_severity_text(accuracy_crossing)}) in {scope}"
        )
    return "; ".join(parts)


def _primary_finding_sentence(main_table: pd.DataFrame) -> str:
    earlier = main_table.loc[main_table["status"] == "earlier"]
    same_or_later = main_table.loc[main_table["status"] == "same_or_later"]
    evaluable = len(earlier) + len(same_or_later)
    if evaluable == 0:
        interpretation = "the prespecified hypothesis was not evaluable"
    elif len(earlier) == evaluable:
        interpretation = "the prespecified hypothesis was supported in every evaluable comparison"
    elif len(earlier) == 0:
        interpretation = "the prespecified hypothesis was not supported in any evaluable comparison"
    else:
        interpretation = "the prespecified result was mixed across evaluable comparisons"
    earlier_detail = _grouped_finding_details(main_table, "earlier")
    sentence = (
        "Under the frozen equal-corruption protocol, "
        f"{len(earlier)} of {evaluable} evaluable reliability comparisons crossed before the "
        "five-percentage-point raw-accuracy drop"
    )
    if earlier_detail:
        sentence += f": {earlier_detail}"
    status_labels = {
        "same_or_later": "same or later",
        "signal_did_not_cross": "signal did not cross",
        "accuracy_did_not_cross": "accuracy did not cross",
        "neither_crossed": "neither crossed",
    }
    remainder = [
        f"{label}: {_grouped_finding_details(main_table, status)}"
        for status, label in status_labels.items()
        if (main_table["status"] == status).any()
    ]
    if remainder:
        sentence += "; " + "; ".join(remainder)
    return f"{sentence}; {interpretation}."


def _secondary_ece_sentence(ece_transfer: pd.DataFrame) -> str:
    if ece_transfer.empty:
        raise ValueError("secondary ECE table is empty")
    severity = int(ece_transfer["severity"].max())
    selected = ece_transfer.loc[ece_transfer["severity"] == severity].copy()
    observed_groups = set(selected[["dataset", "model"]].itertuples(index=False, name=None))
    expected_groups = {(dataset, model) for dataset, model, _count in CANONICAL_REPLICATES}
    if observed_groups != expected_groups or len(selected) != len(expected_groups):
        raise ValueError("secondary ECE table lacks one canonical group at maximum severity")
    tolerance = 1e-12
    difference = selected["paired_difference_mean"].astype(float)
    categories = (
        (difference < -tolerance, "lower"),
        (difference > tolerance, "higher"),
        (difference.abs() <= tolerance, "equal"),
    )
    comparisons: list[str] = []
    for mask, label in categories:
        subset = selected.loc[mask]
        if subset.empty:
            continue
        scope = _group_scope(zip(subset["dataset"], subset["model"], strict=True))
        comparisons.append(f"{label} in {scope}")
    lower = float(difference.min())
    upper = float(difference.max())
    return (
        "In secondary analysis, clean-calibration-fitted temperature scaling had mean ECE "
        f"{', '.join(comparisons)} at S{severity}; paired temperature-minus-raw differences "
        f"ranged from {lower:+.3f} to {upper:+.3f}."
    )


def finding_text(main_table: pd.DataFrame, ece_transfer: pd.DataFrame) -> str:
    """Return exactly two deterministic sentences from primary and secondary tables."""

    return f"{_primary_finding_sentence(main_table)} {_secondary_ece_sentence(ece_transfer)}\n"


def _divergence_interval(
    signal_crossing: Any, accuracy_crossing: Any
) -> tuple[float, float] | None:
    if pd.isna(signal_crossing) or pd.isna(accuracy_crossing):
        return None
    signal = float(signal_crossing)
    accuracy = float(accuracy_crossing)
    return (signal, accuracy) if signal < accuracy else None


def _trajectory_for_group(
    metrics: pd.DataFrame,
    protocol: Mapping[str, Any],
    dataset: str,
    model: str,
    method: str,
    metric: str,
) -> pd.DataFrame:
    summary = _summary_for(metrics, protocol, method, metric)
    return summary.loc[(summary["dataset"] == dataset) & (summary["model"] == model)].copy()


def _plot_trajectory(
    ax: Axes,
    summary: pd.DataFrame,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    marker: str = "o",
) -> None:
    if summary.empty:
        return
    ordered = summary.sort_values("severity")
    x = ordered["severity"].to_numpy(dtype=float)
    y = ordered["mean"].to_numpy(dtype=float)
    std = ordered["std"].to_numpy(dtype=float)
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label=label,
    )
    estimable = np.isfinite(std)
    if estimable.any():
        ax.fill_between(
            x,
            y - std,
            y + std,
            where=estimable,
            color=color,
            alpha=0.14,
            linewidth=0,
        )


def _crossing_marker(
    ax: Axes,
    summary: pd.DataFrame,
    crossing: Any,
    *,
    color: str,
    label: str,
) -> None:
    if pd.isna(crossing):
        return
    value = float(crossing)
    match = summary.loc[np.isclose(summary["severity"].astype(float), value)]
    ax.axvline(value, color=color, linestyle=":", linewidth=1.1, alpha=0.9)
    if not match.empty:
        ax.scatter(
            [value],
            [float(match.iloc[0]["mean"])],
            color=color,
            marker="X",
            s=48,
            zorder=5,
            label=label,
        )


def _shade_divergence(ax: Axes, signal_crossing: Any, accuracy_crossing: Any) -> bool:
    interval = _divergence_interval(signal_crossing, accuracy_crossing)
    if interval is None:
        return False
    ax.axvspan(*interval, color=_SHADE_STYLE, alpha=0.10, linewidth=0)
    return True


def _threshold_row(
    main_table: pd.DataFrame,
    dataset: str,
    model: str,
    method: str,
    metric: str,
) -> pd.Series:
    selected = main_table.loc[
        (main_table["dataset"] == dataset)
        & (main_table["model"] == model)
        & (main_table["signal_method"] == method)
        & (main_table["signal_metric"] == metric)
    ]
    if len(selected) != 1:
        raise ValueError(f"primary row missing for {dataset}/{model}/{method}/{metric}")
    return selected.iloc[0]


def _signal_cutoff(row: pd.Series, protocol: Mapping[str, Any]) -> float:
    thresholds = protocol["thresholds"]
    baseline = float(row["clean_signal_mean"])
    metric = str(row["signal_metric"])
    if metric == "ece":
        return max(
            baseline * float(thresholds["ece_relative_increase"]),
            baseline + float(thresholds["ece_minimum_absolute_increase"]),
        )
    if metric == "mean_predictive_entropy":
        return max(
            baseline * float(thresholds["entropy_relative_increase"]),
            baseline + float(thresholds["entropy_minimum_absolute_increase"]),
        )
    if metric == "risk_at_80_coverage":
        return max(
            baseline * float(thresholds["selective_risk_relative_increase"]),
            baseline + float(thresholds["selective_risk_minimum_absolute_increase"]),
        )
    if metric == "conformal_coverage":
        return float(thresholds["conformal_target_coverage"]) - float(
            thresholds["conformal_allowed_shortfall"]
        )
    raise ValueError(f"unknown primary signal cutoff: {metric}")


def _style_axis(ax: Axes, row_index: int, column_index: int, dataset: str) -> None:
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.1, 5.1)
    ax.set_xticks(range(6), ["Clean", "1", "2", "3", "4", "5"])
    if row_index == 0:
        ax.set_title(dataset.upper(), fontsize=12, fontweight="bold")
    if row_index == 3:
        ax.set_xlabel("Corruption severity (ordinal)")
    if column_index > 0:
        ax.tick_params(axis="y", labelleft=True)


def _legend(ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels, strict=True):
        if label:
            unique.setdefault(label, handle)
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=7, frameon=False, loc="best")


def plot_checkpoint3_headline(
    metrics: pd.DataFrame,
    main_table: pd.DataFrame,
    protocol: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Render a threshold-annotated ResNet50 headline without outcome tuning."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5}):
        fig, axes = plt.subplots(4, 2, figsize=(13.0, 11.2), sharex=True)
        for column_index, dataset in enumerate(("smids", "hushem")):
            model = "resnet50"
            accuracy = _trajectory_for_group(
                metrics, protocol, dataset, model, "raw_softmax", "accuracy"
            )
            ece = _trajectory_for_group(metrics, protocol, dataset, model, "raw_softmax", "ece")
            temperature_ece = _trajectory_for_group(
                metrics, protocol, dataset, model, "temperature", "ece"
            )
            entropy = _trajectory_for_group(
                metrics,
                protocol,
                dataset,
                model,
                "raw_softmax",
                "mean_predictive_entropy",
            )
            coverage = _trajectory_for_group(
                metrics, protocol, dataset, model, "aps", "conformal_coverage"
            )
            ece_row = _threshold_row(main_table, dataset, model, "raw_softmax", "ece")
            entropy_row = _threshold_row(
                main_table,
                dataset,
                model,
                "raw_softmax",
                "mean_predictive_entropy",
            )
            coverage_row = _threshold_row(main_table, dataset, model, "aps", "conformal_coverage")
            accuracy_crossing = ece_row["accuracy_drop_severity"]
            signal_rows = (ece_row, entropy_row, coverage_row)
            finite_signals = [
                float(row["signal_crossing_severity"])
                for row in signal_rows
                if pd.notna(row["signal_crossing_severity"])
            ]
            earliest_signal = min(finite_signals) if finite_signals else np.nan

            accuracy_ax = axes[0, column_index]
            _plot_trajectory(
                accuracy_ax,
                accuracy,
                label="Raw-softmax accuracy (primary)",
                color=_PRIMARY_STYLE,
            )
            accuracy_cutoff = float(ece_row["clean_accuracy_mean"]) - float(
                protocol["thresholds"]["accuracy_absolute_drop"]
            )
            accuracy_ax.axhline(
                accuracy_cutoff,
                color=_ACCURACY_MARKER_STYLE,
                linestyle="--",
                linewidth=1.1,
                label="Clean accuracy − 0.05",
            )
            _crossing_marker(
                accuracy_ax,
                accuracy,
                accuracy_crossing,
                color=_ACCURACY_MARKER_STYLE,
                label="Accuracy crossing",
            )
            _shade_divergence(accuracy_ax, earliest_signal, accuracy_crossing)
            accuracy_ax.set_ylabel("Accuracy")
            accuracy_ax.set_ylim(0.0, 1.0)

            ece_ax = axes[1, column_index]
            _plot_trajectory(
                ece_ax,
                ece,
                label="Raw-softmax ECE (primary)",
                color=_PRIMARY_STYLE,
            )
            _plot_trajectory(
                ece_ax,
                temperature_ece,
                label="Temperature ECE (secondary)",
                color=_SECONDARY_STYLE,
                linestyle="--",
                marker="s",
            )
            ece_ax.axhline(
                _signal_cutoff(ece_row, protocol),
                color=_PRIMARY_STYLE,
                linestyle="--",
                linewidth=1.0,
                label="Frozen raw-ECE cutoff",
            )
            _crossing_marker(
                ece_ax,
                ece,
                ece_row["signal_crossing_severity"],
                color=_PRIMARY_STYLE,
                label="Raw-ECE crossing",
            )
            if pd.notna(accuracy_crossing):
                ece_ax.axvline(
                    float(accuracy_crossing),
                    color=_ACCURACY_MARKER_STYLE,
                    linestyle="-.",
                    linewidth=0.9,
                )
            _shade_divergence(ece_ax, ece_row["signal_crossing_severity"], accuracy_crossing)
            ece_ax.set_ylabel("Expected calibration error")
            ece_ax.set_ylim(bottom=0.0)

            entropy_ax = axes[2, column_index]
            _plot_trajectory(
                entropy_ax,
                entropy,
                label="Raw predictive entropy (primary)",
                color=_PRIMARY_STYLE,
            )
            ensemble_mi = _trajectory_for_group(
                metrics,
                protocol,
                dataset,
                model,
                "deep_ensemble",
                "mean_mutual_information",
            )
            if ensemble_mi.empty:
                entropy_ax.text(
                    0.98,
                    0.93,
                    "No same-fold deep ensemble by design",
                    ha="right",
                    va="top",
                    transform=entropy_ax.transAxes,
                    color="#555555",
                    fontsize=7.5,
                )
            else:
                _plot_trajectory(
                    entropy_ax,
                    ensemble_mi,
                    label="Deep-ensemble MI (secondary)",
                    color=_SECONDARY_STYLE,
                    linestyle="--",
                    marker="D",
                )
            entropy_ax.axhline(
                _signal_cutoff(entropy_row, protocol),
                color=_PRIMARY_STYLE,
                linestyle="--",
                linewidth=1.0,
                label="Frozen entropy cutoff",
            )
            _crossing_marker(
                entropy_ax,
                entropy,
                entropy_row["signal_crossing_severity"],
                color=_PRIMARY_STYLE,
                label="Entropy crossing",
            )
            if pd.notna(accuracy_crossing):
                entropy_ax.axvline(
                    float(accuracy_crossing),
                    color=_ACCURACY_MARKER_STYLE,
                    linestyle="-.",
                    linewidth=0.9,
                )
            _shade_divergence(
                entropy_ax,
                entropy_row["signal_crossing_severity"],
                accuracy_crossing,
            )
            entropy_ax.set_ylabel("Uncertainty (nats)")
            entropy_ax.set_ylim(bottom=0.0)

            coverage_ax = axes[3, column_index]
            _plot_trajectory(
                coverage_ax,
                coverage,
                label="APS coverage (primary)",
                color=_COVERAGE_STYLE,
            )
            coverage_ax.axhline(
                _signal_cutoff(coverage_row, protocol),
                color=_COVERAGE_STYLE,
                linestyle="--",
                linewidth=1.0,
                label="Frozen 0.85 cutoff",
            )
            _crossing_marker(
                coverage_ax,
                coverage,
                coverage_row["signal_crossing_severity"],
                color=_COVERAGE_STYLE,
                label="APS crossing",
            )
            if pd.notna(accuracy_crossing):
                coverage_ax.axvline(
                    float(accuracy_crossing),
                    color=_ACCURACY_MARKER_STYLE,
                    linestyle="-.",
                    linewidth=0.9,
                )
            _shade_divergence(
                coverage_ax,
                coverage_row["signal_crossing_severity"],
                accuracy_crossing,
            )
            coverage_ax.set_ylabel("Empirical coverage")
            coverage_ax.set_ylim(0.0, 1.0)

            for row_index, ax in enumerate(axes[:, column_index]):
                _style_axis(ax, row_index, column_index, dataset)
                _legend(ax)

        fig.suptitle(
            "Checkpoint 3 draft: reliability under simulated device shift\n"
            "ResNet50 primary trajectories; temperature scaling and ensemble MI are secondary",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        corruption_count = len(protocol["aggregation"]["device_corruptions"])
        fig.text(
            0.5,
            0.012,
            "Lines are means and bands are ±1 sample SD across seed/fold replicates after equal "
            f"weighting of the {corruption_count} corruptions within each replicate. Shading "
            "appears only "
            "between an observed primary-signal crossing and a later observed accuracy crossing.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.95))
        fig.savefig(
            output,
            dpi=DEFAULT_DPI,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Software": "calibration-under-shift checkpoint3"},
        )
        plt.close(fig)
    return output


def _format_number(value: Any, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def _format_crossing(value: Any) -> str:
    if pd.isna(value):
        return "—"
    number = float(value)
    return f"{number:g}"


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    title: str,
    note: str,
    formatters: Mapping[str, Any] | None = None,
) -> str:
    formatters = dict(formatters or {})
    lines = [f"# {title}", "", note, ""]
    headers = [label for _column, label in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in frame.itertuples(index=False):
        values: list[str] = []
        record = row._asdict()
        for column, _label in columns:
            value = record[column]
            formatter = formatters.get(column)
            rendered = formatter(value) if formatter is not None else _escape_markdown(value)
            values.append(rendered)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _main_markdown(main_table: pd.DataFrame) -> str:
    display = main_table.copy()
    display["clean_signal"] = [
        f"{_format_number(mean)} ± {_format_number(std)}"
        for mean, std in zip(display["clean_signal_mean"], display["clean_signal_std"], strict=True)
    ]
    display["clean_accuracy"] = [
        f"{_format_number(mean)} ± {_format_number(std)}"
        for mean, std in zip(
            display["clean_accuracy_mean"], display["clean_accuracy_std"], strict=True
        )
    ]
    return _markdown_table(
        display,
        (
            ("dataset", "Dataset"),
            ("model", "Backbone"),
            ("signal_label", "Prespecified signal"),
            ("n_replicates", "n"),
            ("clean_signal", "Clean signal, mean ± SD"),
            ("signal_crossing_severity", "Signal crossing"),
            ("clean_accuracy", "Clean accuracy, mean ± SD"),
            ("accuracy_drop_severity", "Accuracy crossing"),
            ("early_warning_gap", "Gap"),
            ("status", "Status"),
        ),
        title="Checkpoint 3 primary results table",
        note=(
            "Primary/prespecified. Corruptions are averaged equally within each seed/fold before "
            "replicate means and sample standard deviations are computed. Missing crossings remain "
            "missing; the gap is an ordinal severity-level difference."
        ),
        formatters={
            "signal_crossing_severity": _format_crossing,
            "accuracy_drop_severity": _format_crossing,
            "early_warning_gap": _format_crossing,
        },
    )


def _secondary_markdown(frame: pd.DataFrame, title: str) -> str:
    columns = [(column, column.replace("_", " ").title()) for column in frame.columns]
    formatters = {
        column: _format_number
        for column in frame.columns
        if column.endswith(("_mean", "_std")) or column in {"mean", "std"}
    }
    return _markdown_table(
        frame,
        columns,
        title=title,
        note=(
            "Secondary/exploratory. This table is not part of the frozen primary threshold "
            "decision and must be labeled accordingly in any interpretation."
        ),
        formatters=formatters,
    )


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_table_pair(
    output_dir: Path,
    stem: str,
    frame: pd.DataFrame,
    markdown: str,
) -> tuple[Path, Path]:
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    _atomic_csv(csv_path, frame)
    _atomic_text(markdown_path, markdown)
    return csv_path, markdown_path


def generate_checkpoint3_artifacts(
    metrics_path: str | Path,
    thresholds_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path = "results/checkpoint3",
    figure_path: str | Path | None = "results/figures/checkpoint3_headline.png",
) -> dict[str, Path]:
    """Validate canonical inputs and write every Checkpoint-3 artifact."""

    metrics_source = Path(metrics_path)
    thresholds_source = Path(thresholds_path)
    protocol_source = Path(protocol_path)
    if (
        not metrics_source.is_file()
        or not thresholds_source.is_file()
        or not protocol_source.is_file()
    ):
        raise FileNotFoundError("metrics, thresholds, and frozen protocol files must all exist")
    metrics_input = pd.read_csv(metrics_source)
    thresholds_input = pd.read_csv(thresholds_source)
    with protocol_source.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    metrics, thresholds = validate_checkpoint3_inputs(
        metrics_input,
        thresholds_input,
        protocol,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    main_table = build_main_table(metrics, thresholds)
    outputs["main_table_csv"], outputs["main_table_markdown"] = _write_table_pair(
        destination,
        "checkpoint3_main_table",
        main_table,
        _main_markdown(main_table),
    )
    ece_transfer = _ece_transfer_table(metrics, protocol)
    finding = finding_text(main_table, ece_transfer)
    outputs["finding"] = destination / "checkpoint3_finding.txt"
    _atomic_text(outputs["finding"], finding)

    secondary_specs = (
        (
            "secondary_per_corruption",
            _per_corruption_table(metrics, protocol),
            "Checkpoint 3 secondary per-corruption breakdown",
        ),
        (
            "secondary_ece_transfer",
            ece_transfer,
            "Checkpoint 3 secondary raw-versus-temperature ECE",
        ),
        (
            "secondary_selective",
            _selective_table(metrics, protocol),
            "Checkpoint 3 secondary selective retained accuracy",
        ),
        (
            "secondary_conformal",
            _conformal_table(metrics, protocol),
            "Checkpoint 3 secondary conformal coverage and set size",
        ),
        (
            "secondary_ensemble_mi",
            _ensemble_mi_table(metrics, protocol),
            "Checkpoint 3 secondary deep-ensemble mutual information",
        ),
    )
    for stem, frame, title in secondary_specs:
        outputs[f"{stem}_csv"], outputs[f"{stem}_markdown"] = _write_table_pair(
            destination,
            stem,
            frame,
            _secondary_markdown(frame, title),
        )

    if figure_path is not None:
        outputs["headline_figure"] = plot_checkpoint3_headline(
            metrics,
            main_table,
            protocol,
            figure_path,
        )
    manifest_path = destination / "checkpoint3_manifest.json"
    manifest = {
        "analysis_tiers": {
            "primary": [
                key
                for key in ("main_table_csv", "main_table_markdown", "finding", "headline_figure")
                if key in outputs
            ],
            "secondary_exploratory": sorted(key for key in outputs if key.startswith("secondary_")),
        },
        "inputs": {
            "metrics": {"path": str(metrics_source), "sha256": _sha256(metrics_source)},
            "thresholds": {
                "path": str(thresholds_source),
                "sha256": _sha256(thresholds_source),
            },
            "protocol": {"path": str(protocol_source), "sha256": _sha256(protocol_source)},
        },
        "outputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in sorted(outputs.items())
        },
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    outputs["manifest"] = manifest_path
    return outputs


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--thresholds", type=Path, default=Path("results/thresholds.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/analysis_protocol.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/checkpoint3"))
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/figures/checkpoint3_headline.png"),
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="write tables only; the final unshaded F1 is generated separately",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    outputs = generate_checkpoint3_artifacts(
        args.metrics,
        args.thresholds,
        args.protocol,
        args.output_dir,
        None if args.no_figure else args.figure,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


if __name__ == "__main__":
    main()
