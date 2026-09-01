"""Apply the prespecified aggregation and early-warning threshold analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_COLUMNS = {
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


def validate_complete_grid(metrics: pd.DataFrame, protocol: dict[str, Any]) -> None:
    """Require every prespecified condition and method for each replicate."""

    missing_columns = REQUIRED_COLUMNS - set(metrics.columns)
    if missing_columns:
        raise ValueError(f"metrics table is missing columns: {sorted(missing_columns)}")
    corruptions = set(protocol["aggregation"]["device_corruptions"])
    severities = {int(value) for value in protocol["aggregation"].get("severities", range(1, 6))}
    expected = {("clean", 0)} | {
        (corruption, severity) for corruption in corruptions for severity in severities
    }
    work = metrics.copy()
    work["fold"] = work["fold"].fillna("")
    raw = work.loc[(work["method"] == "raw_softmax") & (work["metric"] == "accuracy")]
    if raw.empty:
        raise ValueError("metrics table has no raw-softmax accuracy rows")
    group_columns = ["dataset", "model", "seed", "fold"]
    base_keys = list(raw.groupby(group_columns, dropna=False).groups)
    required = protocol["aggregation"].get("required_method_metrics", {"raw_softmax": "accuracy"})
    for method, metric in required.items():
        selected = work.loc[(work["method"] == method) & (work["metric"] == metric)]
        conditions_by_key = {
            key: list(zip(group["corruption"], group["severity"].astype(int), strict=True))
            for key, group in selected.groupby(group_columns, dropna=False)
        }
        for key in base_keys:
            pairs = conditions_by_key.get(key, [])
            if len(pairs) != len(set(pairs)):
                raise ValueError(f"duplicate {method}/{metric} conditions for replicate {key}")
            actual = set(pairs) & expected
            if actual != expected:
                missing = sorted(expected - actual)
                raise ValueError(
                    f"incomplete {method}/{metric} grid for replicate {key}; missing={missing}"
                )

    ensemble_required = protocol["aggregation"].get("ensemble_method_metrics", {})
    base_replicates = raw.loc[raw["seed"].astype(str) != "ensemble"]
    for group_key, group in base_replicates.groupby(["dataset", "model", "fold"]):
        if group["seed"].nunique() < 2:
            continue
        dataset, model, fold = group_key
        for method, metric in ensemble_required.items():
            selected = work.loc[
                (work["dataset"] == dataset)
                & (work["model"] == model)
                & (work["fold"] == fold)
                & (work["seed"].astype(str) == "ensemble")
                & (work["method"] == method)
                & (work["metric"] == metric)
            ]
            actual = set(zip(selected["corruption"], selected["severity"].astype(int), strict=True))
            if actual & expected != expected:
                missing = sorted(expected - actual)
                raise ValueError(
                    f"incomplete {method}/{metric} ensemble grid for {group_key}; missing={missing}"
                )


def aggregate_device_corruptions(metrics: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    """Average corruptions within each seed before summarizing across seeds."""

    corruptions = set(protocol["aggregation"]["device_corruptions"])
    device = metrics.loc[metrics["corruption"].isin(corruptions)].copy()
    within_seed_columns = ["dataset", "model", "seed", "fold", "severity", "method", "metric"]
    within = device.groupby(within_seed_columns, dropna=False, as_index=False)["value"].mean()
    across_columns = ["dataset", "model", "severity", "method", "metric"]
    summary = (
        within.groupby(across_columns, dropna=False)["value"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary


def _clean_means(metrics: pd.DataFrame) -> pd.DataFrame:
    clean = metrics.loc[metrics["corruption"] == "clean"]
    return (
        clean.groupby(["dataset", "model", "method", "metric"], dropna=False)["value"]
        .mean()
        .rename("clean_value")
        .reset_index()
    )


def _first_crossing(
    frame: pd.DataFrame,
    comparator,
) -> float:
    ordered = frame.sort_values("severity")
    crossed = ordered.loc[comparator(ordered["mean"].to_numpy())]
    return float(crossed.iloc[0]["severity"]) if not crossed.empty else float("nan")


def threshold_analysis(metrics: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    aggregated = aggregate_device_corruptions(metrics, protocol)
    clean = _clean_means(metrics)
    joined = aggregated.merge(clean, on=["dataset", "model", "method", "metric"], how="left")
    thresholds = protocol["thresholds"]
    outputs: list[dict[str, Any]] = []
    for (dataset, model), group in joined.groupby(["dataset", "model"]):
        accuracy = group.loc[(group["method"] == "raw_softmax") & (group["metric"] == "accuracy")]
        if accuracy.empty:
            continue
        accuracy_clean = float(accuracy["clean_value"].iloc[0])
        accuracy_cutoff = accuracy_clean - float(thresholds["accuracy_absolute_drop"])
        accuracy_severity = _first_crossing(
            accuracy,
            lambda value, cutoff=accuracy_cutoff: value < cutoff,
        )
        signals = [
            (
                "raw_softmax",
                "ece",
                lambda value, baseline: (
                    value > baseline * float(thresholds["ece_relative_increase"])
                )
                & (value > baseline + float(thresholds["ece_minimum_absolute_increase"])),
            ),
            (
                "raw_softmax",
                "mean_predictive_entropy",
                lambda value, baseline: (
                    value > baseline * float(thresholds["entropy_relative_increase"])
                )
                & (value > baseline + float(thresholds["entropy_minimum_absolute_increase"])),
            ),
            (
                "raw_softmax",
                "risk_at_80_coverage",
                lambda value, baseline: (
                    value > baseline * float(thresholds["selective_risk_relative_increase"])
                )
                & (
                    value > baseline + float(thresholds["selective_risk_minimum_absolute_increase"])
                ),
            ),
            (
                "aps",
                "conformal_coverage",
                lambda value, _baseline: value
                < float(thresholds["conformal_target_coverage"])
                - float(thresholds["conformal_allowed_shortfall"]),
            ),
        ]
        for method, metric, comparator in signals:
            signal = group.loc[(group["method"] == method) & (group["metric"] == metric)]
            if signal.empty:
                continue
            baseline = float(signal["clean_value"].iloc[0])
            signal_severity = _first_crossing(
                signal,
                lambda value, compare=comparator, clean_value=baseline: compare(value, clean_value),
            )
            gap = (
                accuracy_severity - signal_severity
                if np.isfinite(accuracy_severity) and np.isfinite(signal_severity)
                else np.nan
            )
            outputs.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "signal_method": method,
                    "signal_metric": metric,
                    "clean_signal": baseline,
                    "signal_crossing_severity": signal_severity,
                    "clean_accuracy": accuracy_clean,
                    "accuracy_drop_severity": accuracy_severity,
                    "early_warning_gap": gap,
                    "signal_is_earlier": bool(gap > 0) if np.isfinite(gap) else False,
                }
            )
    return pd.DataFrame(outputs)


def write_summary(thresholds: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Prespecified threshold analysis",
        "",
        "Generated from `results/metrics.csv` using `configs/analysis_protocol.yaml`.",
        "Missing threshold crossings remain missing.",
        "",
    ]
    if thresholds.empty:
        lines.append("No complete result groups were available.")
    else:
        lines.extend(
            [
                "| Dataset | Model | Signal | Signal severity | Accuracy severity | Gap | Earlier? |",
                "|---|---|---|---:|---:|---:|:---:|",
            ]
        )
        for row in thresholds.itertuples(index=False):
            signal_severity = (
                "—"
                if pd.isna(row.signal_crossing_severity)
                else f"{row.signal_crossing_severity:g}"
            )
            accuracy_severity = (
                "—" if pd.isna(row.accuracy_drop_severity) else f"{row.accuracy_drop_severity:g}"
            )
            gap = "—" if pd.isna(row.early_warning_gap) else f"{row.early_warning_gap:g}"
            lines.append(
                f"| {row.dataset} | {row.model} | {row.signal_method}/{row.signal_metric} | "
                f"{signal_severity} | {accuracy_severity} | {gap} | "
                f"{'yes' if row.signal_is_earlier else 'no'} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/analysis_protocol.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/thresholds.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/thresholds.md"))
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    metrics = pd.read_csv(args.metrics)
    with args.protocol.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not args.allow_partial:
        validate_complete_grid(metrics, protocol)
    results = threshold_analysis(metrics, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    write_summary(results, args.summary)
    print(f"wrote {len(results)} threshold rows to {args.output}")


if __name__ == "__main__":
    main()
