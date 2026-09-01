"""Confidence, energy, and feature-space scores for shift and failure detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score


def max_softmax_uncertainty(logits) -> np.ndarray:
    return 1.0 - softmax(np.asarray(logits, dtype=np.float64), axis=1).max(axis=1)


def energy_score(logits, temperature: float = 1.0) -> np.ndarray:
    """Return energy where larger (less negative) values indicate more OOD-like input."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape samples x classes")
    return -temperature * logsumexp(values / temperature, axis=1)


@dataclass(frozen=True)
class MahalanobisScorer:
    class_means: np.ndarray
    precision: np.ndarray

    @classmethod
    def fit(cls, features, labels, regularization: float = 1e-4) -> MahalanobisScorer:
        values = np.asarray(features, dtype=np.float64)
        target = np.asarray(labels, dtype=np.int64)
        if values.ndim != 2 or target.shape != values.shape[:1]:
            raise ValueError("features must be NxD and labels length N")
        if not np.isfinite(values).all():
            raise ValueError("features must be finite")
        classes = np.unique(target)
        if not np.array_equal(classes, np.arange(classes.size)):
            raise ValueError("training labels must be contiguous from zero")
        means = np.stack([values[target == label].mean(axis=0) for label in classes])
        centered = np.concatenate(
            [values[target == label] - means[label] for label in classes], axis=0
        )
        # macOS Accelerate can emit spurious floating-point-status warnings during
        # otherwise finite BLAS products. Validate inputs/outputs explicitly and
        # silence only the low-level arithmetic flags within this bounded block.
        with np.errstate(all="ignore"):
            covariance = centered.T @ centered / max(1, len(centered) - len(classes))
            covariance += regularization * np.eye(covariance.shape[0])
            precision = np.linalg.pinv(covariance, hermitian=True)
        if not np.isfinite(precision).all():
            raise ValueError("regularized covariance inversion produced non-finite values")
        return cls(class_means=means, precision=precision)

    def score(self, features) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        differences = values[:, None, :] - self.class_means[None, :, :]
        distances = np.einsum("ncd,df,ncf->nc", differences, self.precision, differences)
        return distances.min(axis=1)


def shifted_input_auroc(clean_scores, shifted_scores) -> float:
    clean = np.asarray(clean_scores, dtype=np.float64)
    shifted = np.asarray(shifted_scores, dtype=np.float64)
    labels = np.concatenate([np.zeros(len(clean)), np.ones(len(shifted))])
    scores = np.concatenate([clean, shifted])
    return float(roc_auc_score(labels, scores))
