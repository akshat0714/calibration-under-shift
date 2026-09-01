from __future__ import annotations

import numpy as np

from src.uncertainty.ood_scores import (
    MahalanobisScorer,
    energy_score,
    shifted_input_auroc,
)


def test_energy_is_larger_for_flat_low_magnitude_logits():
    confident = energy_score([[8.0, 0.0], [0.0, 8.0]])
    flat = energy_score([[0.0, 0.0], [0.1, -0.1]])
    assert np.all(flat > confident)
    assert shifted_input_auroc(confident, flat) == 1.0


def test_mahalanobis_is_small_near_training_classes_and_large_far_away():
    features = np.array([[-1.1, 0.0], [-0.9, 0.1], [0.9, 0.0], [1.1, -0.1]])
    labels = np.array([0, 0, 1, 1])
    scorer = MahalanobisScorer.fit(features, labels, regularization=1e-2)
    near = scorer.score([[-1.0, 0.02], [1.0, -0.02]])
    far = scorer.score([[0.0, 5.0], [5.0, 5.0]])
    assert np.all(np.isfinite(near)) and np.all(np.isfinite(far))
    assert far.min() > near.max()
