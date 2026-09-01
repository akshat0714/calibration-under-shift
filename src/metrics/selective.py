"""Selective-prediction and per-sample failure-detection metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def risk_coverage_curve(
    correct,
    uncertainty,
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage and error risk after retaining least-uncertain samples."""

    is_correct = np.asarray(correct, dtype=bool)
    score = np.asarray(uncertainty, dtype=np.float64)
    if is_correct.ndim != 1 or score.ndim != 1 or len(is_correct) != len(score):
        raise ValueError("correct and uncertainty must be equal-length vectors")
    if len(score) == 0 or not np.isfinite(score).all():
        raise ValueError("uncertainty must be non-empty and finite")
    order = np.argsort(score, kind="stable")
    cumulative_errors = np.cumsum(~is_correct[order])
    retained = np.arange(1, len(score) + 1)
    coverage = retained / len(score)
    risk = cumulative_errors / retained
    return np.concatenate([[0.0], coverage]), np.concatenate([[0.0], risk])


def area_under_risk_coverage(correct, uncertainty) -> float:
    coverage, risk = risk_coverage_curve(correct, uncertainty)
    return float(np.trapezoid(risk, coverage))


def risk_at_coverage(correct, uncertainty, target_coverage: float = 0.8) -> float:
    if not 0 < target_coverage <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    coverage, risk = risk_coverage_curve(correct, uncertainty)
    index = int(np.searchsorted(coverage, target_coverage, side="left"))
    index = min(index, len(risk) - 1)
    return float(risk[index])


def failure_detection_auroc(correct, uncertainty) -> float:
    """AUROC where incorrect predictions are the positive class."""

    is_correct = np.asarray(correct, dtype=bool)
    score = np.asarray(uncertainty, dtype=np.float64)
    if is_correct.shape != score.shape:
        raise ValueError("correct and uncertainty must have matching shapes")
    failure = (~is_correct).astype(int)
    if np.unique(failure).size < 2:
        return float("nan")
    return float(roc_auc_score(failure, score))


def selective_metrics(
    probabilities,
    labels,
    uncertainty=None,
    target_coverage: float = 0.8,
) -> dict[str, float]:
    probs = np.asarray(probabilities)
    target = np.asarray(labels)
    correct = probs.argmax(axis=1) == target
    if uncertainty is None:
        uncertainty = 1.0 - probs.max(axis=1)
    return {
        "aurc": area_under_risk_coverage(correct, uncertainty),
        f"risk_at_{int(target_coverage * 100)}_coverage": risk_at_coverage(
            correct, uncertainty, target_coverage
        ),
        "failure_detection_auroc": failure_detection_auroc(correct, uncertainty),
    }
