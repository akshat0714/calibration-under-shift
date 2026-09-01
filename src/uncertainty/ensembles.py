"""Predictive uncertainty decomposition for deep ensembles."""

from __future__ import annotations

import numpy as np


def _entropy(probabilities: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    clipped = np.clip(probabilities, epsilon, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=-1)


def ensemble_statistics(logits, temperature: float = 1.0) -> dict[str, np.ndarray]:
    """Summarize MxNxC logits into predictive entropy and mutual information."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2:
        raise ValueError("ensemble logits must have shape models x samples x classes")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    shifted = values / temperature
    shifted -= shifted.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    mean_probability = probabilities.mean(axis=0)
    predictive_entropy = _entropy(mean_probability)
    expected_entropy = _entropy(probabilities).mean(axis=0)
    mutual_information = np.maximum(0.0, predictive_entropy - expected_entropy)
    votes = probabilities.argmax(axis=-1)
    variation_ratio = np.empty(values.shape[1], dtype=np.float64)
    for sample in range(values.shape[1]):
        counts = np.bincount(votes[:, sample], minlength=values.shape[2])
        variation_ratio[sample] = 1.0 - counts.max() / values.shape[0]
    return {
        "probabilities": mean_probability,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
        "variation_ratio": variation_ratio,
    }
