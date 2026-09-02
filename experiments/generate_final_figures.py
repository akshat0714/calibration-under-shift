"""Generate the final provenance-traced figure suite.

F1, F4, and F5 are computed directly from ``results/metrics.csv``. F2 and
F3 require bin- and curve-level geometry that is not reconstructible from the
scalar tidy rows, so they consume the paired saved detail JSONs. The
generator verifies every detail-level ECE, AURC, and risk-at-80 scalar against
its matching ``metrics.csv`` row before plotting and writes the plotted data
plus source hashes to ``results/figure_data``. F7 combines the validated
attribution summary with raw accuracy from the exact matching
checkpoint and registers the hashed qualitative attribution grid.

All per-severity summaries weight device corruptions equally within each
seed/fold replicate before summarizing across replicates. Reliability-bin
geometry is explicitly pooled by bin count and is labeled as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from experiments.analysis_artifacts import validate_analysis_inputs
from src.viz.figures import DEFAULT_DPI

DEVICE_CORRUPTIONS: tuple[str, ...] = (
    "defocus_blur",
    "motion_blur",
    "gaussian_noise",
    "shot_noise",
    "jpeg",
    "resample",
    "illumination",
)
GROUPS: tuple[tuple[str, str], ...] = (
    ("smids", "resnet50"),
    ("smids", "xception"),
    ("smids", "mobilenet_v3_large"),
    ("hushem", "resnet50"),
)
REPLICATE_COLUMNS: tuple[str, ...] = ("dataset", "model", "seed", "fold")


@dataclass(frozen=True)
class SignalSpec:
    method: str
    metric: str
    label: str
    color: str
    marker: str
    linestyle: str


SIGNALS: tuple[SignalSpec, ...] = (
    SignalSpec("raw_softmax", "accuracy", "Accuracy", "#222222", "o", "-"),
    SignalSpec("raw_softmax", "ece", "ECE", "#0072B2", "s", "--"),
    SignalSpec(
        "raw_softmax",
        "mean_predictive_entropy",
        "Predictive entropy",
        "#D55E00",
        "^",
        "-.",
    ),
    SignalSpec(
        "raw_softmax",
        "risk_at_80_coverage",
        "Selective risk @ 80%",
        "#CC79A7",
        "D",
        ":",
    ),
    SignalSpec("aps", "conformal_coverage", "APS coverage", "#009E73", "P", "-"),
)

_METHOD_LABELS: Mapping[str, str] = {
    "raw_softmax": "Max-softmax",
    "temperature": "Temperature-scaled",
    "energy": "Energy",
    "mahalanobis": "Mahalanobis",
    "mc_dropout": "MC dropout",
    "deep_ensemble": "Deep ensemble",
    "aps": "APS",
    "ensemble_aps": "Ensemble APS",
}
_GROUP_LABELS: Mapping[tuple[str, str], str] = {
    ("smids", "resnet50"): "SMIDS · ResNet50",
    ("smids", "xception"): "SMIDS · Xception",
    ("smids", "mobilenet_v3_large"): "SMIDS · MobileNetV3",
    ("hushem", "resnet50"): "HuSHeM · ResNet50",
}
_COLORS: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#000000",
)
_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "P", "X")
_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":", (0, (5, 2)), (0, (1, 1)))
_FULL_REVISION = re.compile(r"[0-9a-f]{40}")

_STYLE: Mapping[str, Any] = {
    "axes.axisbelow": True,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlesize": 10,
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "grid.alpha": 0.20,
    "grid.color": "#666666",
    "grid.linewidth": 0.55,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "lines.linewidth": 1.8,
    "savefig.dpi": DEFAULT_DPI,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_committed_revision(revision: Any, root: Path, *, label: str) -> str:
    """Require one full hexadecimal SHA that resolves to a local commit."""

    value = str(revision)
    if _FULL_REVISION.fullmatch(value) is None:
        raise ValueError(f"{label} must be one full 40-hex git revision, found {value!r}")
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{value}^{{commit}}"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"{label} does not resolve to a committed revision: {value}")
    return value


def _resolve_repo_path(value: Any, root: Path, *, label: str) -> Path:
    text = str(value)
    if not text or text.startswith("external:"):
        raise ValueError(f"{label} must be a repository path, found {text!r}")
    candidate = Path(text)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes the repository: {text!r}") from exc
    return resolved


def _file_manifest_entry(path: Path, root: Path, *, label: str) -> dict[str, str]:
    resolved = _resolve_repo_path(path, root, label=label)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(resolved),
    }


def _assert_portable_manifest_paths(value: Any) -> None:
    """Reject absolute paths anywhere under a manifest ``path`` key."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "path" and isinstance(item, str) and Path(item).is_absolute():
                raise ValueError(f"final figure manifest contains an absolute path: {item}")
            _assert_portable_manifest_paths(item)
    elif isinstance(value, list | tuple):
        for item in value:
            _assert_portable_manifest_paths(item)


def _identifier(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "calibration-under-shift final-figure generator"},
    )
    plt.close(fig)
    return path


def _format_severity_axis(ax: Axes) -> None:
    ax.set_xlim(-0.10, 5.10)
    ax.set_xticks(range(6), ["Clean", "1", "2", "3", "4", "5"])
    ax.set_xlabel("Corruption severity")


def _figure_legend(
    fig: Figure,
    axes: Iterable[Axes],
    *,
    y: float,
    columns: int,
) -> None:
    unique: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            if label and not label.startswith("_"):
                unique.setdefault(label, handle)
    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, y),
            ncol=min(columns, len(unique)),
            handlelength=2.5,
            columnspacing=1.3,
        )


def _replicate_values(
    metrics: pd.DataFrame,
    method: str,
    metric: str,
    *,
    corruption: str | None,
) -> pd.DataFrame:
    """Return clean and shifted replicate values with equal corruption weighting."""

    selected = metrics.loc[
        (metrics["method"] == method)
        & (metrics["metric"] == metric)
        & (metrics["seed"] != "ensemble")
    ].copy()
    clean = (
        selected.loc[(selected["corruption"] == "clean") & (selected["severity"] == 0)]
        .groupby(list(REPLICATE_COLUMNS), dropna=False, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "replicate_value"})
    )
    clean["severity"] = 0
    if corruption is None:
        shifted_source = selected.loc[selected["corruption"].isin(DEVICE_CORRUPTIONS)]
        shifted = (
            shifted_source.groupby([*REPLICATE_COLUMNS, "severity"], dropna=False, as_index=False)[
                "value"
            ]
            .mean()
            .rename(columns={"value": "replicate_value"})
        )
    else:
        if corruption not in DEVICE_CORRUPTIONS:
            raise ValueError(f"unknown device corruption: {corruption}")
        shifted = selected.loc[
            selected["corruption"] == corruption, [*REPLICATE_COLUMNS, "severity", "value"]
        ].rename(columns={"value": "replicate_value"})
    return pd.concat([clean, shifted], ignore_index=True)


def _summarize_replicates(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["dataset", "model", "severity"], dropna=False)["replicate_value"]
        .agg(mean="mean", std="std", n_replicates="count")
        .reset_index()
    )


def _signal_cutoff(
    clean: float,
    method: str,
    metric: str,
    protocol: Mapping[str, Any],
) -> float:
    thresholds = protocol["thresholds"]
    if method == "raw_softmax" and metric == "accuracy":
        return clean - float(thresholds["accuracy_absolute_drop"])
    if method == "raw_softmax" and metric == "ece":
        return max(
            clean * float(thresholds["ece_relative_increase"]),
            clean + float(thresholds["ece_minimum_absolute_increase"]),
        )
    if method == "raw_softmax" and metric == "mean_predictive_entropy":
        return max(
            clean * float(thresholds["entropy_relative_increase"]),
            clean + float(thresholds["entropy_minimum_absolute_increase"]),
        )
    if method == "raw_softmax" and metric == "risk_at_80_coverage":
        return max(
            clean * float(thresholds["selective_risk_relative_increase"]),
            clean + float(thresholds["selective_risk_minimum_absolute_increase"]),
        )
    if method == "aps" and metric == "conformal_coverage":
        return float(thresholds["conformal_target_coverage"]) - float(
            thresholds["conformal_allowed_shortfall"]
        )
    raise ValueError(f"no frozen cutoff for {method}/{metric}")


def _observed_crossing(summary: pd.DataFrame, cutoff: float, *, decreases: bool) -> float:
    shifted = summary.loc[summary["severity"] > 0].sort_values("severity")
    crossed = (
        shifted.loc[shifted["mean"] < cutoff]
        if decreases
        else shifted.loc[shifted["mean"] > cutoff]
    )
    return float(crossed.iloc[0]["severity"]) if not crossed.empty else float("nan")


def build_lockstep_data(
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    corruption: str | None = None,
) -> pd.DataFrame:
    """Build clean- and threshold-normalized primary trajectories.

    Zero is the clean baseline and one is the signal's frozen degradation
    threshold.  This monotone rescaling does not alter any threshold crossing.
    """

    rows: list[pd.DataFrame] = []
    for spec in SIGNALS:
        summary = _summarize_replicates(
            _replicate_values(metrics, spec.method, spec.metric, corruption=corruption)
        )
        for dataset, model in GROUPS:
            group = summary.loc[
                (summary["dataset"] == dataset) & (summary["model"] == model)
            ].sort_values("severity")
            if len(group) != 6:
                raise ValueError(
                    f"incomplete lockstep trajectory for {dataset}/{model}/{spec.metric}"
                )
            clean_row = group.loc[group["severity"] == 0]
            if len(clean_row) != 1:
                raise ValueError("clean baseline is missing or duplicated")
            clean = float(clean_row.iloc[0]["mean"])
            cutoff = _signal_cutoff(clean, spec.method, spec.metric, protocol)
            denominator = cutoff - clean
            if np.isclose(denominator, 0.0):
                raise ValueError(f"zero normalization denominator for {spec.metric}")
            part = group.copy()
            part["analysis_tier"] = (
                "primary/prespecified" if corruption is None else "secondary/exploratory"
            )
            part["corruption_scope"] = (
                "equal_mean_of_7_device_corruptions" if corruption is None else corruption
            )
            part["method"] = spec.method
            part["metric"] = spec.metric
            part["signal_label"] = spec.label
            part["clean_mean"] = clean
            part["frozen_cutoff"] = cutoff
            part["normalized_mean"] = (part["mean"] - clean) / denominator
            part["normalized_std"] = part["std"] / abs(denominator)
            decreases = cutoff < clean
            observed = _observed_crossing(group, cutoff, decreases=decreases)
            if corruption is None:
                if spec.metric == "accuracy":
                    candidates = thresholds.loc[
                        (thresholds["dataset"] == dataset) & (thresholds["model"] == model),
                        "accuracy_drop_severity",
                    ].drop_duplicates()
                else:
                    candidates = thresholds.loc[
                        (thresholds["dataset"] == dataset)
                        & (thresholds["model"] == model)
                        & (thresholds["signal_method"] == spec.method)
                        & (thresholds["signal_metric"] == spec.metric),
                        "signal_crossing_severity",
                    ]
                if len(candidates) != 1:
                    raise ValueError("verified primary crossing is missing or duplicated")
                frozen = float(candidates.iloc[0]) if pd.notna(candidates.iloc[0]) else np.nan
                if not ((np.isnan(observed) and np.isnan(frozen)) or np.isclose(observed, frozen)):
                    raise ValueError(
                        f"lockstep crossing disagrees with thresholds.csv for "
                        f"{dataset}/{model}/{spec.metric}"
                    )
                crossing = frozen
            else:
                crossing = observed
            part["crossing_severity"] = crossing
            rows.append(part)
    output = pd.concat(rows, ignore_index=True)
    columns = (
        "analysis_tier",
        "dataset",
        "model",
        "corruption_scope",
        "severity",
        "method",
        "metric",
        "signal_label",
        "n_replicates",
        "mean",
        "std",
        "clean_mean",
        "frozen_cutoff",
        "normalized_mean",
        "normalized_std",
        "crossing_severity",
    )
    return output.loc[:, columns].sort_values(
        ["dataset", "model", "method", "metric", "severity"], kind="stable"
    )


def plot_lockstep(
    data: pd.DataFrame,
    output_path: Path,
    *,
    corruption: str | None = None,
) -> Path:
    """Plot threshold-normalized trajectories without outcome-region shading."""

    with plt.rc_context(_STYLE):
        fig, axes_array = plt.subplots(2, 2, figsize=(11.3, 7.7), sharex=True, sharey=True)
        axes = list(axes_array.ravel())
        for ax, (dataset, model) in zip(axes, GROUPS, strict=True):
            group = data.loc[(data["dataset"] == dataset) & (data["model"] == model)]
            for spec in SIGNALS:
                series = group.loc[
                    (group["method"] == spec.method) & (group["metric"] == spec.metric)
                ].sort_values("severity")
                ax.plot(
                    series["severity"],
                    series["normalized_mean"],
                    color=spec.color,
                    marker=spec.marker,
                    linestyle=spec.linestyle,
                    markersize=4.6,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    label=spec.label,
                )
                crossing = series["crossing_severity"].iloc[0]
                if pd.notna(crossing):
                    point = series.loc[np.isclose(series["severity"], float(crossing))]
                    ax.scatter(
                        [float(crossing)],
                        [float(point.iloc[0]["normalized_mean"])],
                        marker="X",
                        s=54,
                        color=spec.color,
                        edgecolor="white",
                        linewidth=0.5,
                        zorder=8,
                    )
            ax.axhline(
                0.0,
                color="#777777",
                linewidth=0.8,
                linestyle=":",
                label="Clean baseline" if ax is axes[0] else "_clean",
            )
            ax.axhline(
                1.0,
                color="#222222",
                linewidth=1.1,
                linestyle="--",
                label="Frozen crossing threshold" if ax is axes[0] else "_threshold",
            )
            accuracy = group.loc[group["metric"] == "accuracy"]
            crossing = accuracy["crossing_severity"].iloc[0]
            if pd.notna(crossing):
                ax.axvline(
                    float(crossing),
                    color="#222222",
                    linewidth=0.9,
                    linestyle=(0, (2, 2)),
                    alpha=0.75,
                )
                ax.text(
                    float(crossing) + 0.04,
                    0.94,
                    f"accuracy S{float(crossing):g}",
                    fontsize=7,
                    color="#222222",
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="top",
                )
            ax.set_title(_GROUP_LABELS[(dataset, model)], loc="left", fontweight="bold")
            _format_severity_axis(ax)
            ax.set_ylabel("Threshold-normalized degradation")
        finite_normalized = data["normalized_mean"].to_numpy(dtype=float)
        finite_normalized = finite_normalized[np.isfinite(finite_normalized)]
        upper = max(2.0, float(finite_normalized.max(initial=1.0)) * 1.08)
        axes[0].set_ylim(-0.65, upper)
        scope = (
            "equal-corruption mean (primary/prespecified)"
            if corruption is None
            else f"{corruption.replace('_', ' ')} (secondary/exploratory)"
        )
        headline = (
            "Frozen aggregate reliability signals did not lead the accuracy drop"
            if corruption is None
            else "Per-corruption threshold-normalized trajectories"
        )
        fig.suptitle(
            f"{headline}\nClean = 0 and each frozen degradation threshold = 1 · {scope}",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        _figure_legend(fig, axes, y=0.044, columns=4)
        fig.text(
            0.5,
            0.010,
            "X marks the first observed crossing. The plot has no shading, and a missing X "
            "indicates that the signal never crossed. Values are means after equal corruption "
            "weighting within each seed/fold.",
            ha="center",
            va="bottom",
            fontsize=7.7,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.115, 0.99, 0.91))
        return _save(fig, output_path)


def _metric_index(metrics: pd.DataFrame) -> dict[tuple[str, ...], float]:
    index: dict[tuple[str, ...], float] = {}
    for row in metrics.itertuples(index=False):
        key = (
            str(row.dataset),
            str(row.model),
            _identifier(row.seed),
            _identifier(row.fold),
            str(row.corruption),
            str(int(row.severity)),
            str(row.method),
            str(row.metric),
        )
        if key in index:
            raise ValueError(f"duplicate metrics.csv lookup key: {key}")
        index[key] = float(row.value)
    return index


def _detail_identity(payload: Mapping[str, Any]) -> tuple[str, str, str, str]:
    checkpoint = payload.get("checkpoint", {})
    dataset = checkpoint.get("dataset", payload.get("dataset"))
    model = checkpoint.get("model", payload.get("model"))
    seed = checkpoint.get("seed", "ensemble")
    fold = checkpoint.get("fold", payload.get("fold"))
    return str(dataset), str(model), _identifier(seed), _identifier(fold)


def _condition_key(
    identity: tuple[str, str, str, str],
    condition: Mapping[str, Any],
    method: str,
    metric: str,
) -> tuple[str, ...]:
    dataset, model, seed, fold = identity
    return (
        dataset,
        model,
        seed,
        fold,
        str(condition["corruption"]),
        str(int(condition["severity"])),
        method,
        metric,
    )


def _assert_detail_scalar(
    lookup: Mapping[tuple[str, ...], float],
    identity: tuple[str, str, str, str],
    condition: Mapping[str, Any],
    method: str,
    metric: str,
    observed: float,
) -> None:
    key = _condition_key(identity, condition, method, metric)
    if key not in lookup:
        raise ValueError(f"detail scalar has no matching metrics.csv row: {key}")
    expected = lookup[key]
    if not np.isclose(float(observed), expected, rtol=1e-11, atol=1e-12):
        raise ValueError(
            f"detail scalar disagrees with metrics.csv for {key}: "
            f"detail={observed}, metrics={expected}"
        )


def _load_base_details(
    details_dir: Path,
    metrics: pd.DataFrame,
) -> tuple[list[tuple[Path, Mapping[str, Any]]], list[dict[str, str]]]:
    revisions = set(metrics["evaluation_git_revision"].dropna().astype(str))
    if len(revisions) != 1:
        raise ValueError(f"metrics.csv must contain one evaluation revision, found {revisions}")
    loaded: list[tuple[Path, Mapping[str, Any]]] = []
    provenance: list[dict[str, str]] = []
    for path in sorted(details_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if str(payload.get("evaluation_git_revision")) not in revisions:
            raise ValueError(f"detail evaluation revision disagrees with metrics.csv: {path}")
        provenance.append(
            {
                "path": path.as_posix(),
                "sha256": _sha256(path),
                "kind": str(payload.get("kind")),
                "evaluation_git_revision": str(payload.get("evaluation_git_revision")),
            }
        )
        if payload.get("kind") == "checkpoint_evaluation":
            loaded.append((path, payload))
    if len(loaded) != 16:
        raise ValueError(f"expected 16 base-checkpoint detail JSONs, found {len(loaded)}")
    return loaded, provenance


def extract_reliability_data(
    details: Sequence[tuple[Path, Mapping[str, Any]]],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Pool saved reliability bins after scalar-ECE cross-checks."""

    lookup = _metric_index(metrics)
    contributions: list[dict[str, Any]] = []
    selected_severities = {0, 3, 5}
    for path, payload in details:
        identity = _detail_identity(payload)
        dataset, model, seed, fold = identity
        for condition in payload["conditions"]:
            severity = int(condition["severity"])
            corruption = str(condition["corruption"])
            if severity not in selected_severities:
                continue
            if severity == 0 and corruption != "clean":
                continue
            if severity > 0 and corruption not in DEVICE_CORRUPTIONS:
                continue
            for method in ("raw_softmax", "temperature"):
                detail = condition["methods"][method]
                _assert_detail_scalar(
                    lookup,
                    identity,
                    condition,
                    method,
                    "ece",
                    detail["metrics"]["ece"],
                )
                for row in detail["reliability_bins"]:
                    contributions.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "seed": seed,
                            "fold": fold,
                            "corruption": corruption,
                            "severity": severity,
                            "method": method,
                            "bin": int(row["bin"]),
                            "lower": float(row["lower"]),
                            "upper": float(row["upper"]),
                            "count": int(row["count"]),
                            "confidence": row["confidence"],
                            "accuracy": row["accuracy"],
                            "source_json": path.as_posix(),
                        }
                    )
    raw = pd.DataFrame(contributions)
    if raw.empty:
        raise ValueError("no reliability bins were extracted")
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "model", "severity", "method", "bin"]
    for key, group in raw.groupby(group_columns, sort=False, dropna=False):
        dataset, model, severity, method, bin_index = key
        bounds = group[["lower", "upper"]].drop_duplicates()
        if len(bounds) != 1:
            raise ValueError("reliability-bin edges differ across evaluation details")
        count = int(group["count"].sum())
        populated = group.loc[group["count"] > 0].copy()
        if count:
            confidence = float(
                np.average(populated["confidence"].astype(float), weights=populated["count"])
            )
            accuracy = float(
                np.average(populated["accuracy"].astype(float), weights=populated["count"])
            )
        else:
            confidence = np.nan
            accuracy = np.nan
        ece_summary = _summary_metric(
            metrics,
            method,
            "ece",
        )
        scalar = ece_summary.loc[
            (ece_summary["dataset"] == dataset)
            & (ece_summary["model"] == model)
            & (ece_summary["severity"] == severity)
        ]
        if len(scalar) != 1:
            raise ValueError("mean ECE summary is missing or duplicated")
        rows.append(
            {
                "analysis_tier": "secondary/exploratory",
                "dataset": dataset,
                "model": model,
                "corruption_scope": "clean" if severity == 0 else "pooled_7_device_corruptions",
                "severity": int(severity),
                "method": method,
                "bin": int(bin_index),
                "lower": float(bounds.iloc[0]["lower"]),
                "upper": float(bounds.iloc[0]["upper"]),
                "count": count,
                "confidence": confidence,
                "accuracy": accuracy,
                "mean_ece_from_metrics": float(scalar.iloc[0]["mean"]),
                "std_ece_from_metrics": float(scalar.iloc[0]["std"]),
                "n_replicates": int(scalar.iloc[0]["n_replicates"]),
                "source_condition_count": int(
                    group[["seed", "fold", "corruption"]].drop_duplicates().shape[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group_columns, kind="stable").reset_index(drop=True)


def plot_reliability_panels(data: pd.DataFrame, output_path: Path) -> Path:
    severities = (0, 3, 5)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(4, 3, figsize=(11.3, 12.0), sharex=True, sharey=True)
        for row_index, (dataset, model) in enumerate(GROUPS):
            for column_index, severity in enumerate(severities):
                ax = axes[row_index, column_index]
                ax.plot(
                    [0.0, 1.0],
                    [0.0, 1.0],
                    color="#777777",
                    linestyle=":",
                    linewidth=1.0,
                    label="Perfect calibration" if row_index == column_index == 0 else "_perfect",
                )
                for method_index, method in enumerate(("raw_softmax", "temperature")):
                    series = data.loc[
                        (data["dataset"] == dataset)
                        & (data["model"] == model)
                        & (data["severity"] == severity)
                        & (data["method"] == method)
                    ].sort_values("bin")
                    populated = series.loc[series["count"] > 0]
                    ece = float(series.iloc[0]["mean_ece_from_metrics"])
                    label = (
                        "Raw" if method == "raw_softmax" else "Temperature"
                    ) + f" · mean ECE {ece:.3f}"
                    ax.plot(
                        populated["confidence"],
                        populated["accuracy"],
                        color=_COLORS[method_index],
                        marker=_MARKERS[method_index],
                        linestyle=_LINESTYLES[method_index],
                        markersize=4.1,
                        markerfacecolor="white",
                        markeredgewidth=0.9,
                        label=label,
                    )
                ax.set_xlim(-0.01, 1.01)
                ax.set_ylim(-0.01, 1.01)
                ax.set_xticks(np.linspace(0.0, 1.0, 6))
                ax.set_yticks(np.linspace(0.0, 1.0, 6))
                if row_index == 0:
                    ax.set_title(
                        "Clean" if severity == 0 else f"Severity {severity}", fontweight="bold"
                    )
                if column_index == 0:
                    ax.set_ylabel(f"{_GROUP_LABELS[(dataset, model)]}\nObserved accuracy")
                if row_index == len(GROUPS) - 1:
                    ax.set_xlabel("Predicted confidence")
                ax.legend(loc="upper left", fontsize=6.3)
        fig.suptitle(
            "Reliability diagrams under device corruption · secondary/exploratory",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        fig.text(
            0.5,
            0.010,
            "Reliability curves pool bin counts from the paired saved evaluation-detail JSONs. "
            "Mean ECE values come from metrics.csv after equal corruption weighting within each "
            "seed/fold. Temperature scaling was fitted on clean calibration data only.",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.04, 0.99, 0.965))
        return _save(fig, output_path)


def extract_risk_coverage_data(
    details: Sequence[tuple[Path, Mapping[str, Any]]],
    metrics: pd.DataFrame,
    *,
    coverage_grid: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Aggregate raw-softmax risk curves hierarchically after scalar checks."""

    lookup = _metric_index(metrics)
    grid = np.asarray(
        coverage_grid if coverage_grid is not None else np.linspace(0.0, 1.0, 101),
        dtype=float,
    )
    if grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all():
        raise ValueError("coverage grid must be a finite one-dimensional sequence")
    if not np.all(np.diff(grid) > 0) or grid[0] < 0 or grid[-1] > 1:
        raise ValueError("coverage grid must be strictly increasing in [0, 1]")
    contributions: list[dict[str, Any]] = []
    for path, payload in details:
        identity = _detail_identity(payload)
        dataset, model, seed, fold = identity
        for condition in payload["conditions"]:
            severity = int(condition["severity"])
            corruption = str(condition["corruption"])
            if severity == 0 and corruption != "clean":
                continue
            if severity > 0 and corruption not in DEVICE_CORRUPTIONS:
                continue
            detail = condition["methods"]["raw_softmax"]
            for metric in ("aurc", "risk_at_80_coverage"):
                _assert_detail_scalar(
                    lookup,
                    identity,
                    condition,
                    "raw_softmax",
                    metric,
                    detail["metrics"][metric],
                )
            curve = detail["risk_coverage"]
            coverage = np.asarray(curve["coverage"], dtype=float)
            risk = np.asarray(curve["risk"], dtype=float)
            if coverage.shape != risk.shape or coverage.ndim != 1 or len(coverage) < 2:
                raise ValueError(f"invalid risk-coverage curve in {path}")
            if not np.isfinite(coverage).all() or not np.isfinite(risk).all():
                raise ValueError(f"non-finite risk-coverage curve in {path}")
            if np.any(np.diff(coverage) < 0):
                raise ValueError(f"non-monotone coverage grid in {path}")
            interpolated = np.interp(grid, coverage, risk)
            for target, value in zip(grid, interpolated, strict=True):
                contributions.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "fold": fold,
                        "corruption": corruption,
                        "severity": severity,
                        "coverage": float(target),
                        "risk": float(value),
                        "source_json": path.as_posix(),
                    }
                )
    frame = pd.DataFrame(contributions)
    if frame.empty:
        raise ValueError("no risk-coverage curves were extracted")
    within = (
        frame.groupby([*REPLICATE_COLUMNS, "severity", "coverage"], dropna=False, as_index=False)[
            "risk"
        ]
        .mean()
        .rename(columns={"risk": "replicate_risk"})
    )
    summary = (
        within.groupby(["dataset", "model", "severity", "coverage"], dropna=False)["replicate_risk"]
        .agg(mean_risk="mean", std_risk="std", n_replicates="count")
        .reset_index()
    )
    aurc = _summary_metric(metrics, "raw_softmax", "aurc").rename(
        columns={"mean": "mean_aurc", "std": "std_aurc"}
    )
    risk80 = _summary_metric(metrics, "raw_softmax", "risk_at_80_coverage").rename(
        columns={"mean": "mean_risk_at_80", "std": "std_risk_at_80"}
    )
    summary = summary.merge(
        aurc[["dataset", "model", "severity", "mean_aurc", "std_aurc"]],
        on=["dataset", "model", "severity"],
        validate="many_to_one",
    ).merge(
        risk80[["dataset", "model", "severity", "mean_risk_at_80", "std_risk_at_80"]],
        on=["dataset", "model", "severity"],
        validate="many_to_one",
    )
    summary.insert(0, "analysis_tier", "secondary/exploratory")
    summary.insert(
        3,
        "corruption_scope",
        np.where(summary["severity"] == 0, "clean", "equal_mean_of_7_device_corruptions"),
    )
    return summary.sort_values(
        ["dataset", "model", "severity", "coverage"], kind="stable"
    ).reset_index(drop=True)


def plot_risk_coverage_panels(data: pd.DataFrame, output_path: Path) -> Path:
    with plt.rc_context(_STYLE):
        fig, axes_array = plt.subplots(2, 2, figsize=(10.8, 7.7), sharex=True, sharey=True)
        axes = list(axes_array.ravel())
        for ax, (dataset, model) in zip(axes, GROUPS, strict=True):
            group = data.loc[(data["dataset"] == dataset) & (data["model"] == model)]
            for index, severity in enumerate(range(6)):
                series = group.loc[group["severity"] == severity].sort_values("coverage")
                ax.plot(
                    series["coverage"],
                    series["mean_risk"],
                    color=_COLORS[index],
                    linestyle=_LINESTYLES[index],
                    marker=_MARKERS[index],
                    markevery=10,
                    markersize=3.4,
                    markerfacecolor="white",
                    markeredgewidth=0.8,
                    label="Clean" if severity == 0 else f"Severity {severity}",
                )
            ax.axvline(0.8, color="#555555", linestyle=":", linewidth=1.0)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 0.70)
            ax.set_xticks(np.linspace(0.0, 1.0, 6))
            ax.set_yticks(np.linspace(0.0, 0.7, 8))
            ax.set_xlabel("Coverage (fraction retained)")
            ax.set_ylabel("Risk (error rate)")
            ax.set_title(_GROUP_LABELS[(dataset, model)], loc="left", fontweight="bold")
        fig.suptitle(
            "Selective risk–coverage by corruption severity · secondary/exploratory",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        _figure_legend(fig, axes, y=0.047, columns=6)
        fig.text(
            0.5,
            0.012,
            "Risk-coverage geometry comes from the paired saved evaluation-detail JSONs. AURC "
            "and risk-at-80 endpoints are cross-checked against metrics.csv.\nCurves use a common "
            "coverage grid and equal corruption weighting within each seed/fold. The dotted line "
            "marks 80% coverage.",
            ha="center",
            fontsize=7.6,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.12, 0.99, 0.94))
        return _save(fig, output_path)


def _summary_metric(
    metrics: pd.DataFrame,
    method: str,
    metric: str,
) -> pd.DataFrame:
    frame = _replicate_values(metrics, method, metric, corruption=None)
    return _summarize_replicates(frame)


def build_failure_auroc_data(metrics: pd.DataFrame) -> pd.DataFrame:
    methods = ("raw_softmax", "energy", "mahalanobis", "mc_dropout")
    parts: list[pd.DataFrame] = []
    for method in methods:
        summary = _summary_metric(metrics, method, "failure_detection_auroc")
        summary["method"] = method
        parts.append(summary)
    ensemble = metrics.loc[
        (metrics["seed"] == "ensemble")
        & (metrics["method"] == "deep_ensemble")
        & (metrics["metric"] == "failure_detection_auroc")
    ].copy()
    if not ensemble.empty:
        clean = ensemble.loc[(ensemble["corruption"] == "clean") & (ensemble["severity"] == 0)]
        shifted = ensemble.loc[ensemble["corruption"].isin(DEVICE_CORRUPTIONS)]
        summary = pd.concat(
            [
                clean.groupby(["dataset", "model", "severity"], as_index=False)["value"].mean(),
                shifted.groupby(["dataset", "model", "severity"], as_index=False)["value"].mean(),
            ],
            ignore_index=True,
        ).rename(columns={"value": "mean"})
        summary["std"] = np.nan
        summary["n_replicates"] = 1
        summary["method"] = "deep_ensemble"
        parts.append(summary)
    output = pd.concat(parts, ignore_index=True)
    output.insert(0, "analysis_tier", "secondary/exploratory")
    output.insert(
        3,
        "corruption_scope",
        np.where(output["severity"] == 0, "clean", "equal_mean_of_7_device_corruptions"),
    )
    return output.sort_values(
        ["dataset", "model", "method", "severity"], kind="stable"
    ).reset_index(drop=True)


def plot_failure_auroc_panels(data: pd.DataFrame, output_path: Path) -> Path:
    methods = ("raw_softmax", "energy", "mahalanobis", "mc_dropout", "deep_ensemble")
    with plt.rc_context(_STYLE):
        fig, axes_array = plt.subplots(2, 2, figsize=(10.8, 7.6), sharex=True, sharey=True)
        axes = list(axes_array.ravel())
        for ax, (dataset, model) in zip(axes, GROUPS, strict=True):
            group = data.loc[(data["dataset"] == dataset) & (data["model"] == model)]
            for index, method in enumerate(methods):
                series = group.loc[group["method"] == method].sort_values("severity")
                if series.empty:
                    continue
                ax.plot(
                    series["severity"],
                    series["mean"],
                    color=_COLORS[index],
                    marker=_MARKERS[index],
                    linestyle=_LINESTYLES[index],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    label=_METHOD_LABELS[method],
                )
            ax.axhline(0.5, color="#555555", linestyle=":", linewidth=1.0, label="Chance")
            finite = group["mean"].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            lower = min(0.45, float(finite.min(initial=0.5)) - 0.03)
            ax.set_ylim(max(0.0, lower), 1.0)
            ax.set_ylabel("Failure-detection AUROC")
            ax.set_title(_GROUP_LABELS[(dataset, model)], loc="left", fontweight="bold")
            _format_severity_axis(ax)
        fig.suptitle(
            "Per-sample failure ranking under shift · secondary/exploratory",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        _figure_legend(fig, axes, y=0.048, columns=6)
        fig.text(
            0.5,
            0.012,
            "Failure-detection AUROC measures whether each score ranks incorrect predictions "
            "above correct predictions. It is separate from the prespecified aggregate threshold "
            "analysis. Means use equal corruption weighting within each seed/fold.",
            ha="center",
            fontsize=7.6,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.12, 0.99, 0.94))
        return _save(fig, output_path)


def build_conformal_data(metrics: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for method in ("aps",):
        for metric in ("conformal_coverage", "conformal_mean_set_size"):
            summary = _summary_metric(metrics, method, metric)
            summary["method"] = method
            summary["metric"] = metric
            parts.append(summary)
    ensemble = metrics.loc[
        (metrics["seed"] == "ensemble")
        & (metrics["method"] == "ensemble_aps")
        & metrics["metric"].isin(("conformal_coverage", "conformal_mean_set_size"))
    ]
    for metric in ("conformal_coverage", "conformal_mean_set_size"):
        selected = ensemble.loc[ensemble["metric"] == metric]
        if selected.empty:
            continue
        clean = selected.loc[(selected["corruption"] == "clean") & (selected["severity"] == 0)]
        shifted = selected.loc[selected["corruption"].isin(DEVICE_CORRUPTIONS)]
        summary = pd.concat(
            [
                clean.groupby(["dataset", "model", "severity"], as_index=False)["value"].mean(),
                shifted.groupby(["dataset", "model", "severity"], as_index=False)["value"].mean(),
            ],
            ignore_index=True,
        ).rename(columns={"value": "mean"})
        summary["std"] = np.nan
        summary["n_replicates"] = 1
        summary["method"] = "ensemble_aps"
        summary["metric"] = metric
        parts.append(summary)
    output = pd.concat(parts, ignore_index=True)
    output.insert(0, "analysis_tier", "secondary/exploratory")
    output.insert(
        3,
        "corruption_scope",
        np.where(output["severity"] == 0, "clean", "equal_mean_of_7_device_corruptions"),
    )
    return output.sort_values(
        ["dataset", "model", "method", "metric", "severity"], kind="stable"
    ).reset_index(drop=True)


def plot_conformal_panels(data: pd.DataFrame, output_path: Path) -> Path:
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(4, 2, figsize=(10.8, 11.8), sharex=True)
        for row_index, (dataset, model) in enumerate(GROUPS):
            group = data.loc[(data["dataset"] == dataset) & (data["model"] == model)]
            for method_index, method in enumerate(("aps", "ensemble_aps")):
                coverage = group.loc[
                    (group["method"] == method) & (group["metric"] == "conformal_coverage")
                ].sort_values("severity")
                size = group.loc[
                    (group["method"] == method) & (group["metric"] == "conformal_mean_set_size")
                ].sort_values("severity")
                if coverage.empty or size.empty:
                    continue
                style = {
                    "color": _COLORS[method_index],
                    "marker": _MARKERS[method_index],
                    "linestyle": _LINESTYLES[method_index],
                    "markersize": 4.5,
                    "markerfacecolor": "white",
                    "markeredgewidth": 0.9,
                    "label": _METHOD_LABELS[method],
                }
                axes[row_index, 0].plot(coverage["severity"], coverage["mean"], **style)
                axes[row_index, 1].plot(size["severity"], size["mean"], **style)
            axes[row_index, 0].axhline(
                0.90, color="#555555", linestyle="--", linewidth=0.9, label="Nominal 90%"
            )
            axes[row_index, 0].axhline(
                0.85, color="#777777", linestyle=":", linewidth=0.9, label="Frozen 85% floor"
            )
            axes[row_index, 0].set_ylim(0.65, 1.02)
            axes[row_index, 0].set_ylabel(f"{_GROUP_LABELS[(dataset, model)]}\nCoverage")
            axes[row_index, 1].set_ylabel("Mean set size (classes)")
            for ax in axes[row_index]:
                _format_severity_axis(ax)
            if row_index == 0:
                axes[row_index, 0].set_title("Empirical coverage", fontweight="bold")
                axes[row_index, 1].set_title("Prediction-set size", fontweight="bold")
        fig.suptitle(
            "Conformal behavior under device corruption · secondary/exploratory",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        _figure_legend(fig, axes.ravel(), y=0.047, columns=4)
        fig.text(
            0.5,
            0.012,
            "APS thresholds were fitted once on the clean calibration split. Means use equal "
            "corruption weighting within each seed/fold. The five-member ensemble exists only "
            "for SMIDS ResNet50.",
            ha="center",
            fontsize=7.6,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.08, 0.99, 0.95))
        return _save(fig, output_path)


def _recompute_attribution_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["method", "severity"], as_index=False)
        .agg(
            n_samples=("path", "size"),
            n_valid_spearman=("spearman", "count"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            n_valid_iou=("top_percent_iou", "count"),
            iou_mean=("top_percent_iou", "mean"),
            iou_std=("top_percent_iou", "std"),
        )
        .sort_values(["method", "severity"])
        .reset_index(drop=True)
    )


def _assert_attribution_summary(
    observed: pd.DataFrame,
    per_image: pd.DataFrame,
) -> pd.DataFrame:
    columns = (
        "method",
        "severity",
        "n_samples",
        "n_valid_spearman",
        "spearman_mean",
        "spearman_std",
        "n_valid_iou",
        "iou_mean",
        "iou_std",
    )
    missing = sorted(set(columns).difference(observed.columns))
    if missing:
        raise ValueError(f"attribution summary is missing columns: {missing}")
    expected = _recompute_attribution_summary(per_image)
    selected = observed.loc[:, columns].sort_values(["method", "severity"]).reset_index(drop=True)
    if selected[["method", "severity"]].to_dict("records") != expected[
        ["method", "severity"]
    ].to_dict("records"):
        raise ValueError("attribution summary method/severity rows disagree with per-image data")
    for column in ("n_samples", "n_valid_spearman", "n_valid_iou"):
        if not np.array_equal(
            selected[column].to_numpy(dtype=int), expected[column].to_numpy(dtype=int)
        ):
            raise ValueError(f"attribution summary disagrees with per-image data for {column}")
    for column in ("spearman_mean", "spearman_std", "iou_mean", "iou_std"):
        if not np.allclose(
            selected[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=1e-11,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(f"attribution summary disagrees with per-image data for {column}")
    return selected


def _single_identity(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame:
        raise ValueError(f"attribution per-image data is missing {column}")
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"attribution per-image data must contain one {column}")
    return values.iloc[0]


def _validate_hashed_output(
    provenance: Mapping[str, Any],
    path: Path,
    root: Path,
) -> dict[str, str]:
    outputs = provenance.get("outputs", {})
    entry = outputs.get(path.name)
    if not isinstance(entry, Mapping):
        raise ValueError(f"attribution provenance does not register {path.name}")
    recorded_path = _resolve_repo_path(
        entry.get("path"), root, label=f"attribution output {path.name}"
    )
    if recorded_path != path.resolve():
        raise ValueError(f"attribution provenance path disagrees for {path.name}")
    digest = _sha256(path)
    if str(entry.get("sha256")) != digest:
        raise ValueError(f"attribution provenance hash disagrees for {path.name}")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest}


def build_attribution_accuracy_data(
    metrics: pd.DataFrame,
    *,
    attribution_dir: Path,
    repo_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Validate attribution artifacts and pair stability with checkpoint accuracy."""

    root = repo_root.resolve()
    directory = attribution_dir.resolve()
    summary_path = directory / "attribution_stability_summary.csv"
    per_image_path = directory / "attribution_stability.csv"
    samples_path = directory / "attribution_samples.csv"
    grid_path = directory / "attribution_grid.png"
    original_stability_path = directory / "attribution_stability.png"
    provenance_path = directory / "attribution_provenance.json"
    for path in (
        summary_path,
        per_image_path,
        samples_path,
        grid_path,
        original_stability_path,
        provenance_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required attribution artifact is missing at {path}")
    with provenance_path.open(encoding="utf-8") as handle:
        provenance = json.load(handle)
    evaluation_revision = _require_committed_revision(
        provenance.get("evaluation_git_revision"),
        root,
        label="attribution evaluation revision",
    )
    hashed_outputs = {
        path.name: _validate_hashed_output(provenance, path, root)
        for path in (
            summary_path,
            per_image_path,
            samples_path,
            grid_path,
            original_stability_path,
        )
    }
    per_image = pd.read_csv(per_image_path)
    summary = _assert_attribution_summary(pd.read_csv(summary_path), per_image)
    required_methods = {"gradcam", "gradcam++"}
    if set(summary["method"]) != required_methods:
        raise ValueError("F7 requires both Grad-CAM and Grad-CAM++")
    if set(summary["severity"].astype(int)) != set(range(6)) or len(summary) != 12:
        raise ValueError("F7 requires attribution stability at severities 0 through 5")

    dataset = str(_single_identity(per_image, "dataset"))
    model = str(_single_identity(per_image, "model"))
    seed = _identifier(_single_identity(per_image, "seed"))
    fold = _identifier(_single_identity(per_image, "fold"))
    run_id = str(_single_identity(per_image, "run_id"))
    checkpoint = str(_single_identity(per_image, "checkpoint"))
    checkpoint_sha256 = str(_single_identity(per_image, "checkpoint_sha256"))
    manifest_sha256 = str(_single_identity(per_image, "manifest_sha256"))
    corruption_protocol_sha256 = str(_single_identity(per_image, "corruption_protocol_sha256"))
    corruption = str(_single_identity(per_image, "corruption"))
    for label, digest in (
        ("checkpoint", checkpoint_sha256),
        ("manifest", manifest_sha256),
        ("corruption protocol", corruption_protocol_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"attribution {label} hash must contain 64 hexadecimal characters")

    checkpoint_provenance = provenance.get("checkpoint", {})
    expected_checkpoint = {
        "path": checkpoint,
        "sha256": checkpoint_sha256,
        "run_id": run_id,
        "seed": seed,
    }
    observed_checkpoint = {
        "path": str(checkpoint_provenance.get("path")),
        "sha256": str(checkpoint_provenance.get("sha256")),
        "run_id": str(checkpoint_provenance.get("run_id")),
        "seed": _identifier(checkpoint_provenance.get("seed")),
    }
    if observed_checkpoint != expected_checkpoint:
        raise ValueError("attribution checkpoint provenance disagrees with per-image data")
    checkpoint_path = _resolve_repo_path(checkpoint, root, label="attribution checkpoint")
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint_sha256:
        raise ValueError("attribution checkpoint file is missing or has the wrong SHA-256")
    dataset_provenance = provenance.get("dataset", {})
    if (
        str(dataset_provenance.get("name")) != dataset
        or str(dataset_provenance.get("manifest_sha256")) != manifest_sha256
    ):
        raise ValueError("attribution dataset provenance disagrees with per-image data")
    if str(provenance.get("corruption_protocol_sha256")) != corruption_protocol_sha256:
        raise ValueError("attribution corruption-protocol provenance disagrees")
    protocol = provenance.get("protocol", {})
    if (
        str(protocol.get("corruption")) != corruption
        or set(protocol.get("methods", [])) != required_methods
        or set(int(value) for value in protocol.get("quantitative_severities", [])) != set(range(6))
    ):
        raise ValueError("attribution protocol provenance is not the complete F7 protocol")
    if str(protocol.get("qualitative_paths_sha256")) != _sha256(samples_path):
        raise ValueError("attribution qualitative-sample hash disagrees with provenance")
    samples = pd.read_csv(samples_path)
    if len(samples) not in range(4, 7) or int(protocol.get("qualitative_sample_count", -1)) != len(
        samples
    ):
        raise ValueError("attribution qualitative-sample count is outside the fixed protocol")

    config = provenance.get("config", {})
    config_path = _resolve_repo_path(config.get("path"), root, label="attribution config")
    if not config_path.is_file() or _sha256(config_path) != str(config.get("sha256")):
        raise ValueError("attribution config is missing or has the wrong SHA-256")

    registry = provenance.get("registry")
    if not isinstance(registry, Mapping):
        raise ValueError("attribution provenance is missing the pinned checkpoint registry")
    registry_path = _resolve_repo_path(
        registry.get("path"), root, label="attribution checkpoint registry"
    )
    if not registry_path.is_file() or _sha256(registry_path) != str(registry.get("sha256")):
        raise ValueError("attribution checkpoint registry is missing or has the wrong SHA-256")
    training_revision = _require_committed_revision(
        registry.get("release_training_git_revision"),
        root,
        label="checkpoint training revision",
    )
    registry_row = registry.get("row", {})
    expected_registry = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "fold": fold,
        "run_id": run_id,
        "checkpoint": checkpoint,
    }
    observed_registry = {
        "dataset": str(registry_row.get("dataset")),
        "model": str(registry_row.get("model")),
        "seed": _identifier(registry_row.get("seed")),
        "fold": _identifier(registry_row.get("fold")),
        "run_id": str(registry_row.get("run_id")),
        "checkpoint": str(registry_row.get("checkpoint")),
    }
    if observed_registry != expected_registry:
        raise ValueError("attribution registry row disagrees with per-image identity")

    matching = metrics.loc[
        (metrics["dataset"] == dataset)
        & (metrics["model"] == model)
        & (metrics["seed"].map(_identifier) == seed)
        & (metrics["fold"].map(_identifier) == fold)
        & (metrics["run_id"].astype(str) == run_id)
        & (metrics["method"] == "raw_softmax")
        & (metrics["metric"] == "accuracy")
        & (
            ((metrics["corruption"] == "clean") & (metrics["severity"] == 0))
            | ((metrics["corruption"] == corruption) & metrics["severity"].between(1, 5))
        )
    ].copy()
    if len(matching) != 6 or set(matching["severity"].astype(int)) != set(range(6)):
        raise ValueError("metrics.csv lacks the six matching raw-accuracy conditions for F7")
    if (
        matching["checkpoint"].astype(str).nunique() != 1
        or str(matching["checkpoint"].iloc[0]) != checkpoint
    ):
        raise ValueError("F7 metrics rows do not identify the attribution checkpoint")
    if (
        matching["manifest_sha256"].astype(str).nunique() != 1
        or str(matching["manifest_sha256"].iloc[0]) != manifest_sha256
    ):
        raise ValueError("F7 metrics rows do not identify the attribution manifest")
    if (
        matching["corruption_protocol_sha256"].astype(str).nunique() != 1
        or str(matching["corruption_protocol_sha256"].iloc[0]) != corruption_protocol_sha256
    ):
        raise ValueError("F7 metrics rows use a different corruption protocol")

    rows: list[dict[str, Any]] = []
    for row in matching.sort_values("severity").itertuples(index=False):
        rows.append(
            {
                "analysis_tier": "secondary/exploratory",
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "fold": fold,
                "run_id": run_id,
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha256,
                "corruption": corruption,
                "severity": int(row.severity),
                "method": "raw_softmax",
                "metric": "accuracy",
                "mean": float(row.value),
                "std": np.nan,
                "n_samples": int(row.n_samples),
            }
        )
    for row in summary.itertuples(index=False):
        for metric, mean_column, std_column, count_column in (
            ("spearman", "spearman_mean", "spearman_std", "n_valid_spearman"),
            ("top_percent_iou", "iou_mean", "iou_std", "n_valid_iou"),
        ):
            rows.append(
                {
                    "analysis_tier": "secondary/exploratory",
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "fold": fold,
                    "run_id": run_id,
                    "checkpoint": checkpoint,
                    "checkpoint_sha256": checkpoint_sha256,
                    "corruption": corruption,
                    "severity": int(row.severity),
                    "method": str(row.method),
                    "metric": metric,
                    "mean": float(getattr(row, mean_column)),
                    "std": float(getattr(row, std_column)),
                    "n_samples": int(getattr(row, count_column)),
                }
            )
    source = {
        "evaluation_git_revision": evaluation_revision,
        "release_training_git_revision": training_revision,
        "checkpoint": expected_checkpoint,
        "corruption": corruption,
        "provenance": {
            "path": provenance_path.relative_to(root).as_posix(),
            "sha256": _sha256(provenance_path),
        },
        "outputs": hashed_outputs,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": _sha256(config_path),
        },
        "registry": {
            "path": registry_path.relative_to(root).as_posix(),
            "sha256": _sha256(registry_path),
        },
    }
    return pd.DataFrame(rows), source, Path(grid_path.relative_to(root).as_posix())


def plot_attribution_accuracy(data: pd.DataFrame, output_path: Path) -> Path:
    identity = data.iloc[0]
    dataset = str(identity["dataset"])
    model = str(identity["model"])
    corruption = str(identity["corruption"])
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.0), sharex=True)
        accuracy = data.loc[data["metric"] == "accuracy"].sort_values("severity")
        axes[0].plot(
            accuracy["severity"],
            accuracy["mean"],
            color="#222222",
            marker="o",
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="Raw-softmax accuracy",
        )
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("A  Matching checkpoint accuracy", loc="left", fontweight="bold")
        specs = (
            (axes[1], "spearman", "Spearman correlation", (-1.05, 1.05), "B"),
            (axes[2], "top_percent_iou", "Top-20% saliency-mask IoU", (-0.05, 1.05), "C"),
        )
        for axis, metric, ylabel, limits, panel in specs:
            for index, method in enumerate(("gradcam", "gradcam++")):
                series = data.loc[
                    (data["metric"] == metric) & (data["method"] == method)
                ].sort_values("severity")
                mean = series["mean"].to_numpy(dtype=float)
                std = series["std"].fillna(0).to_numpy(dtype=float)
                x = series["severity"].to_numpy(dtype=float)
                axis.plot(
                    x,
                    mean,
                    color=_COLORS[index],
                    marker=_MARKERS[index],
                    linestyle=_LINESTYLES[index],
                    markerfacecolor="white",
                    markeredgewidth=0.9,
                    label="Grad-CAM" if method == "gradcam" else "Grad-CAM++",
                )
                axis.fill_between(
                    x,
                    np.clip(mean - std, *limits),
                    np.clip(mean + std, *limits),
                    color=_COLORS[index],
                    alpha=0.12,
                    linewidth=0,
                )
            axis.set_ylim(*limits)
            axis.set_ylabel(ylabel)
            axis.set_title(
                f"{panel}  Clean-to-shift attribution stability",
                loc="left",
                fontweight="bold",
            )
        for axis in axes:
            axis.set_xlim(-0.1, 5.1)
            axis.set_xticks(range(6), ["Clean", "1", "2", "3", "4", "5"])
        axes[2].set_xlabel("Corruption severity")
        fig.suptitle(
            "Attribution stability alongside single-model accuracy · secondary/exploratory\n"
            f"{_GROUP_LABELS[(dataset, model)]} · "
            f"{corruption.replace('_', ' ')}",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        _figure_legend(fig, axes, y=0.048, columns=3)
        fig.text(
            0.5,
            0.012,
            "Attribution lines show mean ± 1 SD across the fixed test images for the released "
            "checkpoint. The accuracy curve uses the matching metrics.csv rows. This analysis "
            "is not part of the prespecified threshold decision.",
            ha="center",
            fontsize=7.6,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.10, 0.99, 0.93))
        return _save(fig, output_path)


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _manifest(
    metrics_path: Path,
    thresholds_path: Path,
    protocol_path: Path,
    detail_provenance: Sequence[Mapping[str, str]],
    attribution_provenance: Mapping[str, Any],
    data_outputs: Mapping[str, Path],
    figure_outputs: Mapping[str, Path],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    root = repo_root.resolve()
    metrics = pd.read_csv(metrics_path, usecols=["evaluation_git_revision"])
    metric_entry = _file_manifest_entry(metrics_path, root, label="metrics source")
    metric_entry["evaluation_git_revisions"] = sorted(
        metrics["evaluation_git_revision"].dropna().astype(str).unique().tolist()
    )
    normalized_details: list[dict[str, str]] = []
    for entry in detail_provenance:
        path = _resolve_repo_path(entry["path"], root, label="evaluation detail source")
        normalized = dict(entry)
        normalized["path"] = path.relative_to(root).as_posix()
        normalized_details.append(normalized)
    manifest = {
        "schema_version": 1,
        "analysis_labels": {
            "f1_headline": "primary/prespecified",
            "f1_per_corruption": "secondary/exploratory",
            "f2_reliability": "secondary/exploratory",
            "f3_risk_coverage": "secondary/exploratory",
            "f4_failure_detection_auroc": "secondary/exploratory",
            "f5_conformal": "secondary/exploratory",
            "f7_attribution_stability_accuracy": "secondary/exploratory",
            "f7_attribution_grid": "secondary/exploratory",
        },
        "source_files": {
            "metrics": {
                **metric_entry,
            },
            "thresholds": _file_manifest_entry(thresholds_path, root, label="threshold source"),
            "analysis_protocol": _file_manifest_entry(
                protocol_path, root, label="analysis protocol source"
            ),
            "evaluation_details": normalized_details,
            "attribution": dict(attribution_provenance),
        },
        "geometry_note": (
            "F2 reliability bins and F3 full risk-coverage curves come from the paired saved "
            "evaluation detail JSONs because scalar metrics.csv rows cannot reconstruct them. "
            "Generation fails unless each detail ECE, AURC, and risk-at-80 scalar agrees with "
            "its matching metrics.csv row. All aggregate claims and displayed summary scalars "
            "come from metrics.csv. F7 recomputes the attribution summary from its per-image "
            "tidy source, validates the attribution hashes and checkpoint identity, and uses the "
            "exact matching raw-accuracy rows from metrics.csv."
        ),
        "plotted_data": {
            name: _file_manifest_entry(path, root, label=f"plotted data {name}")
            for name, path in sorted(data_outputs.items())
        },
        "figures": {
            name: _file_manifest_entry(path, root, label=f"figure {name}")
            for name, path in sorted(figure_outputs.items())
        },
    }
    _assert_portable_manifest_paths(manifest)
    return manifest


def generate_final_figures(
    *,
    metrics_path: str | Path = "results/metrics.csv",
    thresholds_path: str | Path = "results/thresholds.csv",
    protocol_path: str | Path = "configs/analysis_protocol.yaml",
    details_dir: str | Path = "results/evaluation_details",
    output_dir: str | Path = "results/figures",
    data_dir: str | Path = "results/figure_data",
    attribution_dir: str | Path = "results/attribution",
) -> dict[str, Path]:
    repo_root = Path.cwd().resolve()
    metrics_source = Path(metrics_path)
    thresholds_source = Path(thresholds_path)
    protocol_source = Path(protocol_path)
    details_source = Path(details_dir)
    figures = Path(output_dir)
    data = Path(data_dir)
    attribution = Path(attribution_dir)
    raw_metrics = pd.read_csv(metrics_source)
    raw_thresholds = pd.read_csv(thresholds_source)
    with protocol_source.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    metrics, thresholds = validate_analysis_inputs(raw_metrics, raw_thresholds, protocol)
    metric_revisions = sorted(
        metrics["evaluation_git_revision"].dropna().astype(str).unique().tolist()
    )
    if len(metric_revisions) != 1:
        raise ValueError("metrics.csv must contain exactly one evaluation git revision")
    _require_committed_revision(metric_revisions[0], repo_root, label="evaluation revision")

    lockstep = build_lockstep_data(metrics, thresholds, protocol)
    per_corruption = pd.concat(
        [
            build_lockstep_data(metrics, thresholds, protocol, corruption=corruption)
            for corruption in DEVICE_CORRUPTIONS
        ],
        ignore_index=True,
    )
    details, detail_provenance = _load_base_details(details_source, metrics)
    reliability = extract_reliability_data(details, metrics)
    risk_coverage = extract_risk_coverage_data(details, metrics)
    failure_auroc = build_failure_auroc_data(metrics)
    conformal = build_conformal_data(metrics)
    attribution_accuracy, attribution_provenance, attribution_grid = (
        build_attribution_accuracy_data(
            metrics,
            attribution_dir=attribution,
            repo_root=repo_root,
        )
    )

    data_outputs = {
        "f1_headline": _write_csv(lockstep, data / "f1_headline_lockstep.csv"),
        "f1_per_corruption": _write_csv(per_corruption, data / "f1_per_corruption_lockstep.csv"),
        "f2_reliability": _write_csv(reliability, data / "f2_reliability_bins.csv"),
        "f3_risk_coverage": _write_csv(risk_coverage, data / "f3_risk_coverage.csv"),
        "f4_failure_detection_auroc": _write_csv(
            failure_auroc, data / "f4_failure_detection_auroc.csv"
        ),
        "f5_conformal": _write_csv(conformal, data / "f5_conformal.csv"),
        "f7_attribution_stability_accuracy": _write_csv(
            attribution_accuracy, data / "f7_attribution_stability_accuracy.csv"
        ),
    }
    figure_outputs: dict[str, Path] = {
        "f1_headline": plot_lockstep(lockstep, figures / "f1_headline_lockstep.png"),
        "f2_reliability": plot_reliability_panels(
            reliability, figures / "f2_reliability_diagrams.png"
        ),
        "f3_risk_coverage": plot_risk_coverage_panels(
            risk_coverage, figures / "f3_risk_coverage.png"
        ),
        "f4_failure_detection_auroc": plot_failure_auroc_panels(
            failure_auroc, figures / "f4_failure_detection_auroc.png"
        ),
        "f5_conformal": plot_conformal_panels(conformal, figures / "f5_conformal.png"),
        "f7_attribution_stability_accuracy": plot_attribution_accuracy(
            attribution_accuracy, figures / "f7_attribution_stability_accuracy.png"
        ),
        "f7_attribution_grid": attribution_grid,
    }
    appendix_dir = figures / "appendix"
    for corruption in DEVICE_CORRUPTIONS:
        selected = per_corruption.loc[per_corruption["corruption_scope"] == corruption]
        figure_outputs[f"f1_appendix_{corruption}"] = plot_lockstep(
            selected,
            appendix_dir / f"f1_lockstep_{corruption}.png",
            corruption=corruption,
        )
    corruption_grid = figures / "corruption_grid.png"
    if not corruption_grid.is_file():
        raise FileNotFoundError("registered corruption grid is missing")
    figure_outputs["f6_corruption_grid"] = corruption_grid

    manifest_path = data / "final_figure_manifest.json"
    manifest = _manifest(
        metrics_source,
        thresholds_source,
        protocol_source,
        detail_provenance,
        attribution_provenance,
        data_outputs,
        figure_outputs,
        repo_root=repo_root,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    data_return = {f"{name}_data": path for name, path in data_outputs.items()}
    return {**figure_outputs, "manifest": manifest_path, **data_return}


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--thresholds", type=Path, default=Path("results/thresholds.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/analysis_protocol.yaml"))
    parser.add_argument("--details-dir", type=Path, default=Path("results/evaluation_details"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--data-dir", type=Path, default=Path("results/figure_data"))
    parser.add_argument("--attribution-dir", type=Path, default=Path("results/attribution"))
    args = parser.parse_args(argv)
    outputs = generate_final_figures(
        metrics_path=args.metrics,
        thresholds_path=args.thresholds,
        protocol_path=args.protocol,
        details_dir=args.details_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        attribution_dir=args.attribution_dir,
    )
    for name, path in sorted(outputs.items()):
        print(f"{name}: {path}")
    return outputs


if __name__ == "__main__":
    main()
