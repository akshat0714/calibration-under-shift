from __future__ import annotations

import numpy as np
import pytest

from src.uncertainty.temperature import fit_scaler, scaled_probabilities


def test_temperature_scaling_reduces_nll_and_preserves_argmax():
    logits = np.array([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
    labels = np.array([0, 1, 1, 0])
    result = fit_scaler(logits, labels, split="calibration", max_iter=50)
    scaled = scaled_probabilities(result.model, logits)
    assert result.after_nll <= result.before_nll + 1e-6
    assert np.array_equal(logits.argmax(axis=1), scaled.argmax(axis=1))


def test_temperature_scaling_refuses_noncalibration_data():
    with pytest.raises(ValueError, match="calibration split only"):
        fit_scaler([[1.0, 0.0]], [0], split="val")
