from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.conformal import fit_aps, prediction_set_metrics


def _exchangeable_probabilities(rng, labels, strength=1.5):
    logits = rng.normal(size=(len(labels), 3))
    logits[np.arange(len(labels)), labels] += strength
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def test_aps_exchangeable_coverage_is_near_target():
    rng = np.random.default_rng(7)
    calibration_labels = rng.integers(0, 3, size=2000)
    test_labels = rng.integers(0, 3, size=4000)
    calibration_probabilities = _exchangeable_probabilities(rng, calibration_labels)
    test_probabilities = _exchangeable_probabilities(rng, test_labels)
    calibrator = fit_aps(
        calibration_probabilities,
        calibration_labels,
        split=np.repeat("calibration", len(calibration_labels)),
        alpha=0.1,
    )
    metrics = prediction_set_metrics(calibrator.predict(test_probabilities), test_labels)
    assert metrics["conformal_coverage"] >= 0.88
    assert metrics["conformal_coverage"] <= 0.95


def test_aps_refuses_test_split():
    with pytest.raises(ValueError, match="calibration split only"):
        fit_aps([[0.8, 0.2]], [0], split="test", alpha=0.1)
