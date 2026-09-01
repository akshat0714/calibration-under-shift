from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.analyze import (
    aggregate_device_corruptions,
    threshold_analysis,
    validate_complete_grid,
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
