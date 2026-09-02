from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.analyze import (
    aggregate_device_corruptions,
    threshold_analysis,
    validate_complete_grid,
    write_summary,
)


def _protocol():
    return {
        "aggregation": {"device_corruptions": ["defocus_blur", "jpeg"]},
        "thresholds": {
            "accuracy_absolute_drop": 0.05,
            "ece_relative_increase": 2.0,
            "ece_minimum_absolute_increase": 0.02,
            "entropy_relative_increase": 2.0,
            "entropy_minimum_absolute_increase": 0.05,
            "selective_risk_relative_increase": 2.0,
            "selective_risk_minimum_absolute_increase": 0.02,
            "conformal_target_coverage": 0.90,
            "conformal_allowed_shortfall": 0.05,
        },
    }


def _row(seed, corruption, severity, method, metric, value):
    return {
        "dataset": "demo",
        "model": "model",
        "seed": seed,
        "fold": "",
        "corruption": corruption,
        "severity": severity,
        "method": method,
        "metric": metric,
        "value": value,
    }


def test_hierarchical_average_weights_corruptions_within_seed():
    frame = pd.DataFrame(
        [
            _row(1, "defocus_blur", 1, "raw_softmax", "accuracy", 0.8),
            _row(1, "jpeg", 1, "raw_softmax", "accuracy", 1.0),
            _row(2, "defocus_blur", 1, "raw_softmax", "accuracy", 0.6),
            _row(2, "jpeg", 1, "raw_softmax", "accuracy", 0.8),
        ]
    )
    summary = aggregate_device_corruptions(frame, _protocol())
    assert summary.iloc[0]["mean"] == 0.8
    assert summary.iloc[0]["n"] == 2


def test_threshold_analysis_reports_reliability_before_accuracy():
    rows = []
    for seed in (1, 2):
        clean_values = {
            ("raw_softmax", "accuracy"): 0.90,
            ("raw_softmax", "ece"): 0.05,
            ("raw_softmax", "mean_predictive_entropy"): 0.10,
            ("raw_softmax", "risk_at_80_coverage"): 0.02,
            ("aps", "conformal_coverage"): 0.90,
        }
        for (method, metric), value in clean_values.items():
            rows.append(_row(seed, "clean", 0, method, metric, value))
        for severity in (1, 2):
            shifted_values = {
                ("raw_softmax", "accuracy"): 0.88 if severity == 1 else 0.84,
                ("raw_softmax", "ece"): 0.12,
                ("raw_softmax", "mean_predictive_entropy"): 0.22,
                ("raw_softmax", "risk_at_80_coverage"): 0.07,
                ("aps", "conformal_coverage"): 0.84,
            }
            for corruption in ("defocus_blur", "jpeg"):
                for (method, metric), value in shifted_values.items():
                    rows.append(_row(seed, corruption, severity, method, metric, value))
    result = threshold_analysis(pd.DataFrame(rows), _protocol())
    assert set(result["signal_crossing_severity"]) == {1.0}
    assert set(result["accuracy_drop_severity"]) == {2.0}
    assert result["signal_is_earlier"].all()
    assert np.allclose(result["early_warning_gap"], 1.0)


def test_complete_grid_validation_rejects_missing_condition():
    protocol = _protocol()
    protocol["aggregation"]["severities"] = [1, 2]
    rows = [_row(1, "clean", 0, "raw_softmax", "accuracy", 0.9)]
    for corruption in ("defocus_blur", "jpeg"):
        for severity in (1, 2):
            rows.append(_row(1, corruption, severity, "raw_softmax", "accuracy", 0.8))
    complete = pd.DataFrame(rows)
    validate_complete_grid(complete, protocol)
    with pytest.raises(ValueError, match="incomplete raw_softmax/accuracy grid"):
        validate_complete_grid(complete.iloc[:-1], protocol)


def test_missing_threshold_crossing_remains_not_estimable(tmp_path):
    rows = []
    for seed in (1, 2):
        clean_values = {
            ("raw_softmax", "accuracy"): 0.90,
            ("raw_softmax", "ece"): 0.05,
            ("raw_softmax", "mean_predictive_entropy"): 0.10,
            ("raw_softmax", "risk_at_80_coverage"): 0.02,
            ("aps", "conformal_coverage"): 0.90,
        }
        for (method, metric), value in clean_values.items():
            rows.append(_row(seed, "clean", 0, method, metric, value))
        for severity in (1, 2):
            shifted_values = {
                ("raw_softmax", "accuracy"): 0.88 if severity == 1 else 0.84,
                ("raw_softmax", "ece"): 0.06,
                ("raw_softmax", "mean_predictive_entropy"): 0.12,
                ("raw_softmax", "risk_at_80_coverage"): 0.03,
                ("aps", "conformal_coverage"): 0.90,
            }
            for corruption in ("defocus_blur", "jpeg"):
                for (method, metric), value in shifted_values.items():
                    rows.append(_row(seed, corruption, severity, method, metric, value))

    result = threshold_analysis(pd.DataFrame(rows), _protocol())
    assert result["signal_crossing_severity"].isna().all()
    assert result["early_warning_gap"].isna().all()
    assert result["signal_is_earlier"].isna().all()

    summary = tmp_path / "thresholds.md"
    write_summary(result, summary)
    assert "| Not reached | 2 | Not applicable | Not evaluable |" in summary.read_text(
        encoding="utf-8"
    )


def test_complete_grid_requires_ensemble_only_for_five_seed_smids_resnet50():
    protocol = _protocol()
    protocol["aggregation"].update(
        {
            "severities": [1],
            "ensemble_method_metrics": {
                "deep_ensemble": "accuracy",
                "ensemble_aps": "conformal_coverage",
            },
        }
    )

    def base_rows(dataset: str, model: str, seeds: range) -> list[dict]:
        rows = []
        for seed in seeds:
            for corruption, severity in (("clean", 0), ("defocus_blur", 1), ("jpeg", 1)):
                row = _row(seed, corruption, severity, "raw_softmax", "accuracy", 0.8)
                row.update({"dataset": dataset, "model": model})
                rows.append(row)
        return rows

    xception = pd.DataFrame(base_rows("smids", "xception", range(3)))
    validate_complete_grid(xception, protocol)
    unexpected = pd.concat(
        [
            xception,
            pd.DataFrame(
                [
                    {
                        **_row("ensemble", "clean", 0, "deep_ensemble", "accuracy", 0.8),
                        "dataset": "smids",
                        "model": "xception",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="unexpected non-prespecified"):
        validate_complete_grid(unexpected, protocol)

    resnet_rows = base_rows("smids", "resnet50", range(5))
    with pytest.raises(ValueError, match="incomplete deep_ensemble/accuracy"):
        validate_complete_grid(pd.DataFrame(resnet_rows), protocol)

    for corruption, severity in (("clean", 0), ("defocus_blur", 1), ("jpeg", 1)):
        for method, metric in (
            ("deep_ensemble", "accuracy"),
            ("ensemble_aps", "conformal_coverage"),
        ):
            row = _row("ensemble", corruption, severity, method, metric, 0.8)
            row.update({"dataset": "smids", "model": "resnet50"})
            resnet_rows.append(row)
    validate_complete_grid(pd.DataFrame(resnet_rows), protocol)
