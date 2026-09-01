"""Split-conformal adaptive prediction sets (APS)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.metrics.calibration import _validate


def _assert_calibration_split(split) -> None:
    values = {str(value) for value in np.atleast_1d(split)}
    if values != {"calibration"}:
        raise ValueError(
            f"conformal fitting requires calibration split only, received {sorted(values)}"
        )


def aps_scores(probabilities, labels) -> np.ndarray:
    """Cumulative probability mass through each observation's true label."""

    probs, target = _validate(probabilities, labels)
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probabilities = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probabilities, axis=1)
    rank = np.argmax(order == target[:, None], axis=1)
    return cumulative[np.arange(len(target)), rank]


def conformal_quantile(scores, alpha: float) -> float:
    """Finite-sample corrected split-conformal quantile."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    rank = int(np.ceil((len(values) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(values))
    return float(np.partition(values, rank - 1)[rank - 1])


@dataclass(frozen=True)
class APSCalibrator:
    alpha: float
    threshold: float
    calibration_size: int

    def predict(self, probabilities) -> list[list[int]]:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim != 2 or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("probabilities must be NxC rows that sum to one")
        order = np.argsort(-probs, axis=1, kind="stable")
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumulative = np.cumsum(sorted_probs, axis=1)
        sets: list[list[int]] = []
        for row_order, row_cumulative in zip(order, cumulative, strict=True):
            included = row_cumulative <= self.threshold + 1e-12
            included[0] = True  # Avoid an empty decision set at unusually small thresholds.
            sets.append([int(label) for label in row_order[included]])
        return sets


def fit_aps(probabilities, labels, split, alpha: float = 0.1) -> APSCalibrator:
    _assert_calibration_split(split)
    scores = aps_scores(probabilities, labels)
    return APSCalibrator(
        alpha=alpha,
        threshold=conformal_quantile(scores, alpha),
        calibration_size=len(scores),
    )


def prediction_set_metrics(prediction_sets, labels) -> dict[str, float]:
    target = np.asarray(labels, dtype=np.int64)
    if len(prediction_sets) != len(target):
        raise ValueError("one prediction set is required per label")
    covered = [
        int(label) in set(prediction_set)
        for prediction_set, label in zip(prediction_sets, target, strict=True)
    ]
    sizes = [len(set(prediction_set)) for prediction_set in prediction_sets]
    return {
        "conformal_coverage": float(np.mean(covered)),
        "conformal_mean_set_size": float(np.mean(sizes)),
    }
