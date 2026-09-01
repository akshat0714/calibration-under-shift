from __future__ import annotations

import numpy as np
import pytest

from src.metrics.calibration import (
    adaptive_calibration_error,
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from src.metrics.classification import classification_metrics
from src.metrics.selective import (
    area_under_risk_coverage,
    failure_detection_auroc,
    risk_at_coverage,
)


def test_hand_computable_calibration_metrics():
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
    labels = np.array([0, 1])
    # Both are correct; mean confidence is 0.7.
    assert expected_calibration_error(probabilities, labels, n_bins=1) == pytest.approx(0.3)
    assert adaptive_calibration_error(probabilities, labels, n_bins=1) == pytest.approx(0.3)
    assert brier_score(probabilities, labels) == pytest.approx(0.20)
    assert negative_log_likelihood(probabilities, labels) == pytest.approx(
        -np.log([0.8, 0.6]).mean()
    )


def test_perfect_top_label_calibration_has_zero_ece():
    # In the .75 confidence bin, exactly three of four top-label predictions are correct.
    probabilities = np.array([[0.75, 0.25]] * 3 + [[0.25, 0.75]])
    labels = np.array([0, 0, 0, 0])
    assert expected_calibration_error(probabilities, labels, n_bins=4) == pytest.approx(0.0)


def test_classification_metrics_binary():
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.7, 0.3]])
    labels = np.array([0, 1, 0, 0])
    result = classification_metrics(probabilities, labels)
    assert result["accuracy"] == pytest.approx(0.75)
    assert 0 <= result["macro_f1"] <= 1
    assert 0 <= result["auroc"] <= 1


def test_selective_ranking_rewards_uncertainty():
    correct = np.array([True, True, False, False])
    good_uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
    bad_uncertainty = good_uncertainty[::-1]
    assert area_under_risk_coverage(correct, good_uncertainty) < area_under_risk_coverage(
        correct, bad_uncertainty
    )
    assert risk_at_coverage(correct, good_uncertainty, 0.5) == 0.0
    assert failure_detection_auroc(correct, good_uncertainty) == 1.0


def test_probability_validation_rejects_non_normalized_rows():
    with pytest.raises(ValueError, match="sum to one"):
        expected_calibration_error([[0.2, 0.2]], [0])
