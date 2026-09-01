"""Publication-style figures for calibration-under-shift results.

The metric plots consume the tidy table written by ``experiments/run_grid.py``.
Device-corruption summaries are hierarchical: metrics are first averaged equally
over registered corruptions within a seed/fold, then averaged across seed/fold
replicates.  Uncertainty is therefore not inflated by treating every corruption
as an independent experimental replicate.

Run ``python -m src.viz.figures --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import t as student_t

from src.metrics.calibration import reliability_diagram_data
from src.shifts.corruptions import corrupt
from src.shifts.severity import canonical_name, corruption_names

REQUIRED_METRIC_COLUMNS = frozenset(
    {
        "dataset",
        "model",
        "seed",
        "fold",
        "corruption",
        "severity",
        "method",
        "metric",
        "value",
    }
)

DEFAULT_DPI = 150

# Okabe-Ito-derived colors, paired with marker and dash changes so that no series
# identity depends on color alone.  Yellow is omitted because it is weak on white.
_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#000000",
    "#56B4E9",
)
_MARKERS = ("o", "s", "^", "D", "P", "X", "v")
_LINESTYLES = ("-", "--", "-.", ":")
_METHOD_ORDER = {
    method: index
    for index, method in enumerate(
        (
            "raw_softmax",
            "temperature",
            "vector_scaling",
            "deep_ensemble",
            "mc_dropout",
            "aps",
            "ensemble_aps",
            "energy",
            "mahalanobis",
        )
    )
}

_METRIC_CANDIDATES = {
    "accuracy": ("accuracy",),
    "ece": ("ece", "expected_calibration_error"),
    "entropy": ("mean_predictive_entropy", "predictive_entropy", "entropy"),
    "coverage": ("conformal_coverage", "coverage"),
    "set_size": ("conformal_mean_set_size", "mean_set_size", "set_size"),
    "failure_auroc": ("failure_detection_auroc", "failure_auroc"),
}

_STYLE = {
    "axes.axisbelow": True,
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.titlesize": 11,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "grid.alpha": 0.22,
    "grid.color": "#666666",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.8,
    "savefig.dpi": DEFAULT_DPI,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


def load_metrics(path: str | Path) -> pd.DataFrame:
    """Load and validate a tidy metrics CSV without changing its observations."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"metrics CSV does not exist: {source}")
    return validate_metrics(pd.read_csv(source))


def validate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy of a tidy metric frame.

    Non-finite metric values are retained here because some metrics (notably an
    AUROC on a one-class slice) are legitimately undefined.  Individual plots
    omit only those undefined observations and explicitly mark an unavailable
    panel rather than substituting a value.
    """

    if not isinstance(metrics, pd.DataFrame):
        raise TypeError("metrics must be a pandas DataFrame")
    missing = sorted(REQUIRED_METRIC_COLUMNS.difference(metrics.columns))
    if missing:
        raise ValueError(f"metrics table is missing required columns: {', '.join(missing)}")

    frame = metrics.copy()
    identifier_columns = ("dataset", "model", "seed", "corruption", "method", "metric")
    null_identifiers = [column for column in identifier_columns if frame[column].isna().any()]
    if null_identifiers:
        raise ValueError("metrics table has missing identifiers in: " + ", ".join(null_identifiers))

    severity = pd.to_numeric(frame["severity"], errors="coerce")
    invalid_severity = severity.isna() | ~np.isfinite(severity)
    if invalid_severity.any() or not np.allclose(severity, np.round(severity)):
        raise ValueError("severity values must be finite integers")
    frame["severity"] = severity.astype(int)

    values = pd.to_numeric(frame["value"], errors="coerce")
    invalid_values = frame["value"].notna() & values.isna()
    if invalid_values.any():
        examples = ", ".join(repr(item) for item in frame.loc[invalid_values, "value"].head(3))
        raise ValueError(f"metric values must be numeric; invalid values include {examples}")
    frame["value"] = values.astype(float)

    for column in ("dataset", "model", "corruption", "method", "metric"):
        frame[column] = frame[column].astype(str)
    return frame


def _as_metrics(metrics: pd.DataFrame | str | Path) -> pd.DataFrame:
    return load_metrics(metrics) if isinstance(metrics, str | Path) else validate_metrics(metrics)


def _select_metrics(
    metrics: pd.DataFrame,
    dataset: str | None = None,
    model: str | None = None,
) -> pd.DataFrame:
    selected = metrics
    if dataset is not None:
        selected = selected.loc[selected["dataset"] == str(dataset)]
    if model is not None:
        selected = selected.loc[selected["model"] == str(model)]
    if selected.empty:
        qualifiers = ", ".join(
            item
            for item in (
                f"dataset={dataset!r}" if dataset is not None else "",
                f"model={model!r}" if model is not None else "",
            )
            if item
        )
        raise ValueError(f"no metric rows match {qualifiers or 'the requested selection'}")
    return selected.copy()


def _device_corruption(name: Any) -> str | None:
    normalized = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "clean":
        return "clean"
    try:
        return canonical_name(normalized)
    except ValueError:
        # Population/prior shifts and future non-device conditions must not leak
        # into the advertised device-corruption average.
        return None


def _choose_metric(metrics: pd.DataFrame, family: str) -> str | None:
    available = set(metrics["metric"])
    return next((name for name in _METRIC_CANDIDATES[family] if name in available), None)


def _interval_label(uncertainty: str) -> str:
    normalized = uncertainty.lower().replace("_", "").replace("%", "")
    labels = {
        "95ci": "95% CI across seed/fold replicates",
        "ci95": "95% CI across seed/fold replicates",
        "sem": "±1 SEM across seed/fold replicates",
        "sd": "±1 SD across seed/fold replicates",
        "none": "mean across seed/fold replicates",
    }
    if normalized not in labels:
        raise ValueError("uncertainty must be one of: ci95, sem, sd, none")
    return labels[normalized]


def _metric_summary(
    metrics: pd.DataFrame,
    metric_name: str | None,
    uncertainty: str = "ci95",
) -> pd.DataFrame:
    """Hierarchically average a metric over corruptions, then replicates."""

    interval_key = uncertainty.lower().replace("_", "").replace("%", "")
    _interval_label(interval_key)  # Validate even when this metric is absent.
    columns = [
        "dataset",
        "model",
        "method",
        "severity",
        "mean",
        "lower",
        "upper",
        "n_replicates",
        "metric",
    ]
    if metric_name is None:
        return pd.DataFrame(columns=columns)

    subset = metrics.loc[metrics["metric"] == metric_name].copy()
    subset = subset.loc[np.isfinite(subset["value"])]
    subset["_device_corruption"] = subset["corruption"].map(_device_corruption)
    subset = subset.loc[subset["_device_corruption"].notna()]
    if subset.empty:
        return pd.DataFrame(columns=columns)

    run_columns = [
        "dataset",
        "model",
        "method",
        "severity",
        "seed",
        "fold",
    ]
    # Collapse accidental duplicate rows within one condition before weighting
    # registered corruptions equally in each experimental replicate.
    per_corruption = (
        subset.groupby(run_columns + ["_device_corruption"], dropna=False, as_index=False)["value"]
        .mean()
        .rename(columns={"value": "corruption_mean"})
    )
    per_run = (
        per_corruption.groupby(run_columns, dropna=False, as_index=False)["corruption_mean"]
        .mean()
        .rename(columns={"corruption_mean": "run_mean"})
    )
    summary_columns = ["dataset", "model", "method", "severity"]
    summary = (
        per_run.groupby(summary_columns, dropna=False)["run_mean"]
        .agg(mean="mean", std="std", n_replicates="count")
        .reset_index()
    )

    count = summary["n_replicates"].to_numpy(dtype=float)
    std = summary["std"].to_numpy(dtype=float)
    if interval_key in {"95ci", "ci95"}:
        critical = np.full_like(count, np.nan, dtype=float)
        estimable = count >= 2
        critical[estimable] = student_t.ppf(0.975, count[estimable] - 1)
        error = critical * std / np.sqrt(count)
    elif interval_key == "sem":
        error = np.where(count >= 2, std / np.sqrt(count), np.nan)
    elif interval_key == "sd":
        error = np.where(count >= 2, std, np.nan)
    else:
        error = np.full_like(count, np.nan, dtype=float)

    summary["lower"] = summary["mean"] - error
    summary["upper"] = summary["mean"] + error
    summary["metric"] = metric_name
    return summary[columns]


def summarize_over_corruptions(
    metrics: pd.DataFrame | str | Path,
    metric: str,
    *,
    dataset: str | None = None,
    model: str | None = None,
    uncertainty: str = "ci95",
) -> pd.DataFrame:
    """Public summary helper used by plots and downstream report tables."""

    selected = _select_metrics(_as_metrics(metrics), dataset=dataset, model=model)
    return _metric_summary(selected, metric, uncertainty=uncertainty)


def _method_label(method: Any) -> str:
    special = {
        "aps": "APS",
        "ece": "ECE",
        "ensemble_aps": "Ensemble APS",
        "mc_dropout": "MC dropout",
        "raw_softmax": "Raw softmax",
    }
    text = str(method)
    return special.get(text, text.replace("_", " ").strip().capitalize())


def _series_sort_key(key: tuple[str, str, str]) -> tuple[Any, ...]:
    dataset, model, method = key
    return (_METHOD_ORDER.get(method, len(_METHOD_ORDER)), method, dataset, model)


def _series_keys(summaries: Sequence[pd.DataFrame]) -> list[tuple[str, str, str]]:
    keys: set[tuple[str, str, str]] = set()
    for summary in summaries:
        if not summary.empty:
            keys.update(
                (str(row.dataset), str(row.model), str(row.method))
                for row in summary[["dataset", "model", "method"]].itertuples(index=False)
            )
    return sorted(keys, key=_series_sort_key)


def _style_map(keys: Sequence[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    styles: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, key in enumerate(keys):
        styles[key] = {
            "color": _COLORS[index % len(_COLORS)],
            "marker": _MARKERS[index % len(_MARKERS)],
            "linestyle": _LINESTYLES[(index // len(_COLORS)) % len(_LINESTYLES)],
        }
    return styles


def _series_label(
    key: tuple[str, str, str],
    keys: Sequence[tuple[str, str, str]],
) -> str:
    dataset, model, method = key
    datasets = {item[0] for item in keys}
    models = {item[1] for item in keys}
    parts: list[str] = []
    if len(datasets) > 1:
        parts.append(dataset)
    if len(models) > 1:
        parts.append(model)
    parts.append(_method_label(method))
    return " · ".join(parts)


def _format_severity_axis(ax: Axes, severities: Sequence[float]) -> None:
    values = sorted({float(value) for value in severities})
    if not values:
        values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ax.set_xticks(values)
    labels = ["Clean" if np.isclose(value, 0.0) else f"{value:g}" for value in values]
    ax.set_xticklabels(labels)
    ax.set_xlabel("Corruption severity")


def _plot_summary_axis(
    ax: Axes,
    summary: pd.DataFrame,
    *,
    keys: Sequence[tuple[str, str, str]],
    styles: Mapping[tuple[str, str, str], Mapping[str, Any]],
    title: str,
    ylabel: str,
    ylim: tuple[float | None, float | None] | None = None,
) -> None:
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_ylabel(ylabel)
    if summary.empty:
        ax.text(
            0.5,
            0.5,
            "Not available in metrics.csv",
            ha="center",
            va="center",
            color="#555555",
            transform=ax.transAxes,
        )
        _format_severity_axis(ax, [])
        if ylim is not None:
            ax.set_ylim(ylim)
        return

    all_severities: list[float] = []
    group_columns = ["dataset", "model", "method"]
    for group_key, group in summary.groupby(group_columns, sort=False, dropna=False):
        key = tuple(str(item) for item in group_key)
        ordered = group.sort_values("severity")
        x = ordered["severity"].to_numpy(dtype=float)
        y = ordered["mean"].to_numpy(dtype=float)
        lower = ordered["lower"].to_numpy(dtype=float)
        upper = ordered["upper"].to_numpy(dtype=float)
        style = styles[key]
        ax.plot(
            x,
            y,
            label=_series_label(key, keys),
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )
        estimable = np.isfinite(lower) & np.isfinite(upper)
        if estimable.any():
            ax.fill_between(
                x,
                lower,
                upper,
                where=estimable,
                color=style["color"],
                alpha=0.16,
                linewidth=0,
                interpolate=False,
            )
        all_severities.extend(x.tolist())

    _format_severity_axis(ax, all_severities)
    if ylim is not None:
        ax.set_ylim(ylim)


def _context_label(metrics: pd.DataFrame) -> str:
    combinations = metrics[["dataset", "model"]].drop_duplicates()
    if len(combinations) == 1:
        row = combinations.iloc[0]
        return f"{row['dataset']} · {row['model']}"
    datasets = sorted(metrics["dataset"].unique())
    if len(datasets) == 1:
        return f"{datasets[0]} · multiple models"
    return "Multiple datasets and models"


def _figure_legend(
    fig: Figure,
    axes: Sequence[Axes],
    *,
    bottom: float,
    max_columns: int = 4,
) -> bool:
    unique: dict[str, Any] = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            if label and not label.startswith("_"):
                unique.setdefault(label, handle)
    if not unique:
        return False
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, bottom),
        ncol=min(max_columns, len(unique)),
        handlelength=2.6,
        columnspacing=1.3,
    )
    return True


def _output_path(path: str | Path) -> Path:
    output = Path(path)
    if not output.suffix:
        output = output.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _save(fig: Figure, path: str | Path) -> Path:
    output = _output_path(path)
    fig.savefig(output, dpi=DEFAULT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_headline(
    metrics: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    dataset: str | None = None,
    model: str | None = None,
    uncertainty: str = "ci95",
) -> Path:
    """Plot accuracy, ECE, entropy, and conformal coverage against severity."""

    selected = _select_metrics(_as_metrics(metrics), dataset=dataset, model=model)
    families = ("accuracy", "ece", "entropy", "coverage")
    summaries = [
        _metric_summary(selected, _choose_metric(selected, family), uncertainty=uncertainty)
        for family in families
    ]
    keys = _series_keys(summaries)
    styles = _style_map(keys)

    with plt.rc_context(_STYLE):
        fig, axes_grid = plt.subplots(2, 2, figsize=(11.0, 7.2))
        axes = list(axes_grid.ravel())
        panel_specs = (
            ("A  Accuracy", "Accuracy", (0.0, 1.0)),
            ("B  Expected calibration error", "ECE", (0.0, None)),
            ("C  Predictive entropy", "Entropy (nats)", (0.0, None)),
            ("D  Conformal coverage", "Empirical coverage", (0.0, 1.0)),
        )
        for ax, summary, (title, ylabel, ylim) in zip(axes, summaries, panel_specs, strict=True):
            _plot_summary_axis(
                ax,
                summary,
                keys=keys,
                styles=styles,
                title=title,
                ylabel=ylabel,
                ylim=ylim,
            )

        fig.suptitle(
            f"Reliability under simulated device shift\n{_context_label(selected)}",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )
        has_legend = _figure_legend(fig, axes, bottom=0.045)
        fig.text(
            0.5,
            0.012,
            "Corruptions are equally weighted within each replicate; lines show mean and "
            + _interval_label(uncertainty)[:1].lower()
            + _interval_label(uncertainty)[1:]
            + ".",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.11 if has_legend else 0.06, 0.99, 0.92))
        return _save(fig, output_path)


def plot_failure_detection(
    metrics: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    dataset: str | None = None,
    model: str | None = None,
    uncertainty: str = "ci95",
) -> Path:
    """Plot per-sample failure-detection AUROC against corruption severity."""

    selected = _select_metrics(_as_metrics(metrics), dataset=dataset, model=model)
    summary = _metric_summary(
        selected,
        _choose_metric(selected, "failure_auroc"),
        uncertainty=uncertainty,
    )
    keys = _series_keys([summary])
    styles = _style_map(keys)
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        _plot_summary_axis(
            ax,
            summary,
            keys=keys,
            styles=styles,
            title=f"Failure detection under device shift · {_context_label(selected)}",
            ylabel="Failure-detection AUROC",
            ylim=(0.0, 1.0),
        )
        if not summary.empty:
            ax.axhline(
                0.5,
                color="#555555",
                linestyle=":",
                linewidth=1.1,
                label="Chance",
                zorder=0,
            )
        has_legend = _figure_legend(fig, [ax], bottom=0.035)
        fig.text(
            0.5,
            0.01,
            "Device corruptions are equally weighted within each seed/fold replicate.",
            ha="center",
            fontsize=8,
            color="#444444",
        )
        fig.tight_layout(rect=(0.02, 0.11 if has_legend else 0.05, 0.99, 0.98))
        return _save(fig, output_path)


def plot_conformal(
    metrics: pd.DataFrame | str | Path,
    output_path: str | Path,
    *,
    dataset: str | None = None,
    model: str | None = None,
    uncertainty: str = "ci95",
    nominal_coverage: float | None = None,
) -> Path:
    """Plot conformal empirical coverage and mean prediction-set size."""

    if nominal_coverage is not None and not 0.0 <= nominal_coverage <= 1.0:
        raise ValueError("nominal_coverage must lie in [0, 1]")
    selected = _select_metrics(_as_metrics(metrics), dataset=dataset, model=model)
    coverage = _metric_summary(
        selected, _choose_metric(selected, "coverage"), uncertainty=uncertainty
    )
    set_size = _metric_summary(
        selected, _choose_metric(selected, "set_size"), uncertainty=uncertainty
    )
    keys = _series_keys([coverage, set_size])
    styles = _style_map(keys)

    with plt.rc_context(_STYLE):
        fig, axes_array = plt.subplots(1, 2, figsize=(10.4, 4.3))
        axes = list(axes_array)
        _plot_summary_axis(
            axes[0],
            coverage,
            keys=keys,
            styles=styles,
            title="A  Empirical coverage",
            ylabel="Coverage",
            ylim=(0.0, 1.0),
        )
        if nominal_coverage is not None:
            axes[0].axhline(
                nominal_coverage,
                color="#555555",
                linestyle=":",
                linewidth=1.1,
                label=f"Nominal ({nominal_coverage:.0%})",
                zorder=0,
            )
        _plot_summary_axis(
            axes[1],
            set_size,
            keys=keys,
            styles=styles,
            title="B  Mean prediction-set size",
            ylabel="Classes per prediction set",
            ylim=(0.0, None),
        )
        fig.suptitle(
            f"Conformal prediction under device shift · {_context_label(selected)}",
            fontsize=13,
            fontweight="bold",
            y=0.99,
        )
        has_legend = _figure_legend(fig, axes, bottom=0.035)
        fig.tight_layout(rect=(0.02, 0.11 if has_legend else 0.05, 0.99, 0.91))
        return _save(fig, output_path)


def _probability_series(
    probabilities: Any,
    label: str | None,
) -> list[tuple[str, np.ndarray]]:
    if isinstance(probabilities, Mapping):
        if not probabilities:
            raise ValueError("probability mapping must not be empty")
        return [
            (str(name), np.asarray(values, dtype=float)) for name, values in probabilities.items()
        ]
    return [(label or "Predictions", np.asarray(probabilities, dtype=float))]


def plot_reliability_diagram(
    probabilities: Any,
    labels: Any,
    output_path: str | Path,
    *,
    n_bins: int = 15,
    label: str | None = None,
    title: str = "Reliability diagram",
) -> Path:
    """Plot calibration curves and confidence distributions from probabilities.

    ``probabilities`` may be one ``N x C`` array or a mapping from display label
    to arrays sharing the supplied labels.  Empty bins are left empty; no curve
    interpolation or synthetic observations are introduced.
    """

    target = np.asarray(labels)
    series = _probability_series(probabilities, label)
    if target.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")

    rows_by_series: list[tuple[str, np.ndarray, list[dict[str, float | int]]]] = []
    for name, values in series:
        if values.ndim != 2 or len(values) != len(target):
            raise ValueError("each probability array must be N x C and match labels")
        rows = reliability_diagram_data(values, target, n_bins=n_bins)
        rows_by_series.append((name, values, rows))

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(6.2, 6.4))
        grid = fig.add_gridspec(2, 1, height_ratios=(3.2, 1.0), hspace=0.08)
        reliability_ax = fig.add_subplot(grid[0])
        histogram_ax = fig.add_subplot(grid[1], sharex=reliability_ax)
        reliability_ax.plot(
            [0.0, 1.0],
            [0.0, 1.0],
            color="#555555",
            linestyle=":",
            linewidth=1.2,
            label="Perfect calibration",
            zorder=0,
        )

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        for index, (name, values, rows) in enumerate(rows_by_series):
            # Preserve NaNs for empty bins so the line visibly breaks instead of
            # implying observations in an unsupported confidence interval.
            confidence = np.asarray([row["confidence"] for row in rows], dtype=float)
            accuracy = np.asarray([row["accuracy"] for row in rows], dtype=float)
            total = sum(int(row["count"]) for row in rows)
            ece = sum(int(row["count"]) * float(row["gap"]) for row in rows if row["count"]) / total
            color = _COLORS[index % len(_COLORS)]
            marker = _MARKERS[index % len(_MARKERS)]
            linestyle = _LINESTYLES[(index // len(_COLORS)) % len(_LINESTYLES)]
            reliability_ax.plot(
                confidence,
                accuracy,
                color=color,
                marker=marker,
                linestyle=linestyle,
                markersize=5.5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                label=f"{name} (ECE {ece:.3f})",
            )
            max_probability = values.max(axis=1)
            weights = np.full(len(max_probability), 1.0 / len(max_probability))
            histogram_ax.hist(
                max_probability,
                bins=bins,
                weights=weights,
                histtype="step",
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=name,
            )

        reliability_ax.set_xlim(-0.01, 1.01)
        reliability_ax.set_ylim(-0.01, 1.01)
        reliability_ax.set_xticks(np.linspace(0.0, 1.0, 6))
        reliability_ax.set_yticks(np.linspace(0.0, 1.0, 6))
        reliability_ax.set_ylabel("Observed accuracy")
        reliability_ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        reliability_ax.legend(loc="upper left")
        reliability_ax.tick_params(axis="x", labelbottom=False)
        histogram_ax.set_xlim(-0.01, 1.01)
        histogram_ax.set_ylim(bottom=0.0)
        histogram_ax.set_xlabel("Predicted confidence")
        histogram_ax.set_ylabel("Sample\nfraction")
        return _save(fig, output_path)


def _normalize_curve_data(curve_data: Any) -> pd.DataFrame:
    if isinstance(curve_data, pd.DataFrame):
        frame = curve_data.copy()
    elif isinstance(curve_data, Mapping):
        parts: list[pd.DataFrame] = []
        for name, curve in curve_data.items():
            if isinstance(curve, pd.DataFrame):
                part = curve.copy()
            else:
                try:
                    coverage, risk = curve
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "mapping values must be (coverage, risk) pairs or DataFrames"
                    ) from exc
                part = pd.DataFrame({"coverage": coverage, "risk": risk})
            part["label"] = str(name)
            parts.append(part)
        if not parts:
            raise ValueError("curve mapping must not be empty")
        frame = pd.concat(parts, ignore_index=True)
    else:
        try:
            coverage, risk = curve_data
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "curve_data must be a DataFrame, mapping, or (coverage, risk) pair"
            ) from exc
        frame = pd.DataFrame({"coverage": coverage, "risk": risk})

    missing = {"coverage", "risk"}.difference(frame.columns)
    if missing:
        raise ValueError(f"risk-coverage data is missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("risk-coverage data must not be empty")
    frame["coverage"] = pd.to_numeric(frame["coverage"], errors="coerce")
    frame["risk"] = pd.to_numeric(frame["risk"], errors="coerce")
    if not np.isfinite(frame[["coverage", "risk"]].to_numpy(dtype=float)).all():
        raise ValueError("coverage and risk must be finite")
    if not frame["coverage"].between(0.0, 1.0).all():
        raise ValueError("coverage must lie in [0, 1]")
    if not frame["risk"].between(0.0, 1.0).all():
        raise ValueError("risk must lie in [0, 1]")
    return frame


def _risk_group_columns(frame: pd.DataFrame) -> list[str]:
    if "label" in frame.columns:
        return ["label"]
    candidates = ("dataset", "model", "method", "corruption", "severity")
    return [column for column in candidates if column in frame.columns]


def _risk_label(columns: Sequence[str], values: Sequence[Any]) -> str:
    if columns == ["label"]:
        return str(values[0])
    parts: list[str] = []
    for column, value in zip(columns, values, strict=True):
        if column == "method":
            parts.append(_method_label(value))
        elif column == "severity":
            parts.append("Clean" if float(value) == 0 else f"Severity {value}")
        elif column == "corruption" and str(value) == "clean":
            continue
        else:
            parts.append(str(value))
    return " · ".join(parts) or "Risk–coverage"


def plot_risk_coverage(
    curve_data: Any,
    output_path: str | Path,
    *,
    title: str = "Selective prediction",
) -> Path:
    """Plot one or more supplied risk-coverage curves without interpolation."""

    frame = _normalize_curve_data(curve_data)
    group_columns = _risk_group_columns(frame)
    if group_columns:
        grouper: str | list[str] = group_columns[0] if len(group_columns) == 1 else group_columns
        grouped = frame.groupby(grouper, sort=False, dropna=False)
    else:
        grouped = [((), frame)]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        for index, (raw_key, group) in enumerate(grouped):
            values = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            ordered = group.sort_values("coverage", kind="stable")
            ax.plot(
                ordered["coverage"].to_numpy(dtype=float),
                ordered["risk"].to_numpy(dtype=float),
                color=_COLORS[index % len(_COLORS)],
                marker=_MARKERS[index % len(_MARKERS)],
                markevery=max(1, len(ordered) // 10),
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linestyle=_LINESTYLES[(index // len(_COLORS)) % len(_LINESTYLES)],
                label=_risk_label(group_columns, values),
            )
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_xlabel("Coverage (fraction retained)")
        ax.set_ylabel("Risk (error rate)")
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
        ax.legend(loc="best")
        fig.tight_layout()
        return _save(fig, output_path)


def _as_rgb_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    if isinstance(image, str | Path):
        source = Path(image)
        if not source.is_file():
            raise FileNotFoundError(f"sample image does not exist: {source}")
        with Image.open(source) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("sample image must have shape HxW, HxWx3, or HxWx4")
    array = array[..., :3]
    if not np.isfinite(array).all():
        raise ValueError("sample image values must be finite")
    if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
        array = array * 255.0
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def plot_corruption_grid(
    image: Image.Image | np.ndarray | str | Path,
    output_path: str | Path,
    *,
    corruptions: Sequence[str] | None = None,
    severities: Sequence[int] = (0, 1, 2, 3, 4, 5),
    seed: int = 0,
) -> Path:
    """Render a clean sample and registered corruption severity ladders."""

    sample = _as_rgb_image(image)
    names = tuple(corruptions) if corruptions is not None else tuple(corruption_names())
    numeric_levels = np.asarray(tuple(severities), dtype=float)
    if not np.isfinite(numeric_levels).all() or not np.allclose(
        numeric_levels, np.round(numeric_levels)
    ):
        raise ValueError("severities must be integers from 0 through 5")
    levels = tuple(int(value) for value in numeric_levels)
    if not names:
        raise ValueError("at least one corruption is required")
    if not levels:
        raise ValueError("at least one severity is required")
    if any(level not in range(0, 6) for level in levels):
        raise ValueError("severities must be integers from 0 through 5")
    canonical_names = tuple(canonical_name(name) for name in names)

    with plt.rc_context(_STYLE):
        width = max(4.0, 1.85 * len(levels))
        height = max(3.2, 0.65 + 1.65 * len(canonical_names))
        fig, axes = plt.subplots(
            len(canonical_names),
            len(levels),
            figsize=(width, height),
            squeeze=False,
        )
        for row_index, name in enumerate(canonical_names):
            for column_index, severity in enumerate(levels):
                shifted = corrupt(sample, name, severity, seed=seed)
                ax = axes[row_index, column_index]
                ax.imshow(shifted)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_index == 0:
                    ax.set_title("Clean" if severity == 0 else f"Severity {severity}")
                if column_index == 0:
                    ax.set_ylabel(
                        name.replace("_", " ").title(),
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=8,
                    )
        fig.suptitle(
            "Simulated device-corruption sanity check",
            fontsize=13,
            fontweight="bold",
            y=0.995,
        )
        axes_top = 1.0 - 0.62 / height
        fig.subplots_adjust(
            left=0.17,
            right=0.995,
            top=axes_top,
            bottom=0.02,
            wspace=0.03,
            hspace=0.12,
        )
        return _save(fig, output_path)


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip()).strip("-").lower()
    return text or "unnamed"


def generate_metric_figures(
    metrics: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    dataset: str | None = None,
    model: str | None = None,
    uncertainty: str = "ci95",
    nominal_coverage: float | None = None,
) -> dict[str, Path]:
    """Generate headline, failure-detection, and conformal figures.

    When the selection contains multiple dataset/model combinations, each is
    plotted separately so unlike tasks or architectures are never averaged
    together.
    """

    selected = _select_metrics(_as_metrics(metrics), dataset=dataset, model=model)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    combinations = (
        selected[["dataset", "model"]].drop_duplicates().sort_values(["dataset", "model"])
    )
    multiple = len(combinations) > 1
    outputs: dict[str, Path] = {}
    for row in combinations.itertuples(index=False):
        subset = selected.loc[
            (selected["dataset"] == row.dataset) & (selected["model"] == row.model)
        ]
        prefix = f"{_slug(row.dataset)}-{_slug(row.model)}-" if multiple else ""
        specs = (
            ("headline", plot_headline, {}),
            ("failure_detection", plot_failure_detection, {}),
            ("conformal", plot_conformal, {"nominal_coverage": nominal_coverage}),
        )
        for name, function, extra in specs:
            key = f"{prefix}{name}" if multiple else name
            outputs[key] = function(
                subset,
                destination / f"{prefix}{name}.png",
                uncertainty=uncertainty,
                **extra,
            )
    return outputs


def generate_all_figures(
    metrics: pd.DataFrame | str | Path | None,
    output_dir: str | Path,
    *,
    probabilities: Any | None = None,
    labels: Any | None = None,
    risk_curves: Any | None = None,
    sample_image: Image.Image | np.ndarray | str | Path | None = None,
    dataset: str | None = None,
    model: str | None = None,
    uncertainty: str = "ci95",
    nominal_coverage: float | None = None,
    n_bins: int = 15,
    seed: int = 0,
) -> dict[str, Path]:
    """Generate every figure for which the caller supplied real source data."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    if metrics is not None:
        outputs.update(
            generate_metric_figures(
                metrics,
                destination,
                dataset=dataset,
                model=model,
                uncertainty=uncertainty,
                nominal_coverage=nominal_coverage,
            )
        )
    if (probabilities is None) != (labels is None):
        raise ValueError("probabilities and labels must be supplied together")
    if probabilities is not None:
        outputs["reliability"] = plot_reliability_diagram(
            probabilities,
            labels,
            destination / "reliability.png",
            n_bins=n_bins,
        )
    if risk_curves is not None:
        outputs["risk_coverage"] = plot_risk_coverage(
            risk_curves, destination / "risk_coverage.png"
        )
    if sample_image is not None:
        outputs["corruption_grid"] = plot_corruption_grid(
            sample_image, destination / "corruption_grid.png", seed=seed
        )
    if not outputs:
        raise ValueError("supply metrics and/or data for at least one figure")
    return outputs


def _load_array(path: Path, key: str | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"array file does not exist: {path}")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if key is not None and key in loaded.files:
                return np.asarray(loaded[key])
            if len(loaded.files) == 1:
                return np.asarray(loaded[loaded.files[0]])
            choices = ", ".join(loaded.files)
            raise ValueError(f"{path} contains multiple arrays ({choices}); expected key {key!r}")
        finally:
            loaded.close()
    return np.asarray(loaded)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", nargs="?", type=Path, help="tidy metrics.csv")
    parser.add_argument("--metrics", "--metrics-csv", dest="metrics_option", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument(
        "--uncertainty",
        default="ci95",
        choices=("ci95", "sem", "sd", "none"),
        help="band across seed/fold replicates",
    )
    parser.add_argument("--nominal-coverage", type=float)
    parser.add_argument("--probabilities", type=Path, help="N x C .npy/.npz array")
    parser.add_argument("--labels", type=Path, help="length-N .npy/.npz array")
    parser.add_argument("--risk-curves", type=Path, help="CSV with coverage and risk columns")
    parser.add_argument("--sample-image", type=Path)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument("--corruption-seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.metrics_csv is not None and args.metrics_option is not None:
        parser.error("provide metrics as either the positional argument or --metrics, not both")
    if (args.probabilities is None) != (args.labels is None):
        parser.error("--probabilities and --labels must be provided together")
    if not any(
        (
            args.metrics_csv,
            args.metrics_option,
            args.probabilities,
            args.risk_curves,
            args.sample_image,
        )
    ):
        parser.error("provide metrics and/or data for at least one figure")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    """Command-line entry point; returns paths as a convenience for tests."""

    args = _parse_args(argv)
    probabilities = (
        _load_array(args.probabilities, "probabilities") if args.probabilities is not None else None
    )
    labels = _load_array(args.labels, "labels") if args.labels is not None else None
    curves = pd.read_csv(args.risk_curves) if args.risk_curves is not None else None
    outputs = generate_all_figures(
        args.metrics_option or args.metrics_csv,
        args.output_dir,
        probabilities=probabilities,
        labels=labels,
        risk_curves=curves,
        sample_image=args.sample_image,
        dataset=args.dataset,
        model=args.model,
        uncertainty=args.uncertainty,
        nominal_coverage=args.nominal_coverage,
        n_bins=args.n_bins,
        seed=args.corruption_seed,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


if __name__ == "__main__":
    main()
