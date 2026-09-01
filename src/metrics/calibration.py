"""Calibration metrics for binary and multiclass classifiers."""

from __future__ import annotations

import numpy as np


def _validate(probabilities, labels) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probabilities, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or target.ndim != 1 or len(probs) != len(target):
        raise ValueError("probabilities must be NxC and labels must be length N")
    if len(probs) == 0 or probs.shape[1] < 2:
        raise ValueError("at least one sample and two classes are required")
    if not np.isfinite(probs).all() or (probs < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("each probability row must sum to one")
    if (target < 0).any() or (target >= probs.shape[1]).any():
        raise ValueError("label outside probability columns")
    return probs, target


def reliability_diagram_data(
    probabilities, labels, n_bins: int = 15
) -> list[dict[str, float | int]]:
    """Return equal-width bin counts, confidence, accuracy, and gaps."""

    probs, target = _validate(probabilities, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == target
    # Values exactly equal to a boundary go in the upper bin, except confidence=1.
    bin_index = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        members = bin_index == index
        count = int(members.sum())
        mean_confidence = float(confidence[members].mean()) if count else float("nan")
        accuracy = float(correct[members].mean()) if count else float("nan")
        rows.append(
            {
                "bin": index,
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "count": count,
                "confidence": mean_confidence,
                "accuracy": accuracy,
                "gap": abs(accuracy - mean_confidence) if count else float("nan"),
            }
        )
    return rows


def expected_calibration_error(probabilities, labels, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""

    rows = reliability_diagram_data(probabilities, labels, n_bins=n_bins)
    total = sum(int(row["count"]) for row in rows)
    return float(sum(int(row["count"]) * float(row["gap"]) for row in rows if row["count"]) / total)


def adaptive_calibration_error(probabilities, labels, n_bins: int = 15) -> float:
    """Top-label ECE with approximately equal-mass confidence bins."""

    probs, target = _validate(probabilities, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == target
    ordered = np.argsort(confidence, kind="stable")
    bins = np.array_split(ordered, min(n_bins, len(ordered)))
    weighted_gap = 0.0
    for members in bins:
        if len(members):
            weighted_gap += len(members) * abs(
                float(correct[members].mean()) - float(confidence[members].mean())
            )
    return float(weighted_gap / len(target))


def brier_score(probabilities, labels) -> float:
    """Mean multiclass Brier score (sum over class dimensions)."""

    probs, target = _validate(probabilities, labels)
    one_hot = np.eye(probs.shape[1], dtype=np.float64)[target]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def negative_log_likelihood(probabilities, labels, epsilon: float = 1e-12) -> float:
    probs, target = _validate(probabilities, labels)
    true_probability = np.clip(probs[np.arange(len(target)), target], epsilon, 1.0)
    return float(-np.log(true_probability).mean())


def calibration_metrics(probabilities, labels, n_bins: int = 15) -> dict[str, float]:
    return {
        "ece": expected_calibration_error(probabilities, labels, n_bins=n_bins),
        "adaptive_ece": adaptive_calibration_error(probabilities, labels, n_bins=n_bins),
        "brier": brier_score(probabilities, labels),
        "nll": negative_log_likelihood(probabilities, labels),
    }
