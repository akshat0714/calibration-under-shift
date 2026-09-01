"""Quantitative attribution-stability metrics.

The clean and shifted saliency maps are compared by rank agreement (Spearman)
and overlap of their most salient pixels.  Functions accept NumPy arrays or
PyTorch tensors and return ordinary Python/NumPy values for easy serialization.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

Heatmap = np.ndarray | torch.Tensor | Sequence[float]


def _as_finite_array(heatmap: Heatmap, *, name: str) -> np.ndarray:
    if isinstance(heatmap, torch.Tensor):
        values = heatmap.detach().float().cpu().numpy()
    else:
        values = np.asarray(heatmap, dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values.astype(np.float64, copy=False)


def _matching_pair(
    clean_heatmap: Heatmap, shifted_heatmap: Heatmap
) -> tuple[np.ndarray, np.ndarray]:
    clean = _as_finite_array(clean_heatmap, name="clean_heatmap")
    shifted = _as_finite_array(shifted_heatmap, name="shifted_heatmap")
    if clean.shape != shifted.shape:
        raise ValueError(
            "clean_heatmap and shifted_heatmap must have the same shape, "
            f"got {clean.shape} and {shifted.shape}"
        )
    return clean, shifted


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return zero-based average ranks, including exact tie handling."""

    flat = values.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_values = flat[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1], True])
    ranks = np.empty(flat.size, dtype=np.float64)
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranks[order[start:stop]] = (start + stop - 1) / 2
    return ranks


def spearman_heatmap_correlation(clean_heatmap: Heatmap, shifted_heatmap: Heatmap) -> float:
    """Compute Spearman correlation between two flattened heatmaps.

    Spearman correlation is undefined for constant vectors. Constant attribution
    maps therefore return ``NaN`` rather than being credited as stable evidence.
    """

    clean, shifted = _matching_pair(clean_heatmap, shifted_heatmap)
    clean_ranks = _average_ranks(clean)
    shifted_ranks = _average_ranks(shifted)
    clean_centered = clean_ranks - clean_ranks.mean()
    shifted_centered = shifted_ranks - shifted_ranks.mean()
    denominator = float(np.linalg.norm(clean_centered) * np.linalg.norm(shifted_centered))
    if denominator == 0:
        return float("nan")
    correlation = float(np.dot(clean_centered, shifted_centered) / denominator)
    return float(np.clip(correlation, -1, 1))


def top_percent_mask(heatmap: Heatmap, *, top_fraction: float = 0.2) -> np.ndarray:
    """Select exactly ``ceil(top_fraction * pixels)`` highest-saliency pixels.

    A stable sort gives deterministic tie handling, including constant maps.
    """

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    values = _as_finite_array(heatmap, name="heatmap")
    count = max(1, int(np.ceil(values.size * top_fraction)))
    order = np.argsort(-values.ravel(), kind="stable")
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:count]] = True
    return mask.reshape(values.shape)


def top_percent_iou(
    clean_heatmap: Heatmap,
    shifted_heatmap: Heatmap,
    *,
    top_fraction: float = 0.2,
) -> float:
    """Return IoU between the top-saliency masks of two heatmaps."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    clean, shifted = _matching_pair(clean_heatmap, shifted_heatmap)
    if np.ptp(clean) == 0 or np.ptp(shifted) == 0:
        return float("nan")
    clean_mask = top_percent_mask(clean, top_fraction=top_fraction)
    shifted_mask = top_percent_mask(shifted, top_fraction=top_fraction)
    intersection = np.count_nonzero(clean_mask & shifted_mask)
    union = np.count_nonzero(clean_mask | shifted_mask)
    return float(intersection / union)


def pair_stability(
    clean_heatmap: Heatmap,
    shifted_heatmap: Heatmap,
    *,
    top_fraction: float = 0.2,
) -> dict[str, float]:
    """Summarize attribution stability for one clean/shifted image pair."""

    return {
        "spearman": spearman_heatmap_correlation(clean_heatmap, shifted_heatmap),
        "top_percent_iou": top_percent_iou(
            clean_heatmap,
            shifted_heatmap,
            top_fraction=top_fraction,
        ),
    }


def batch_stability(
    clean_heatmaps: Heatmap,
    shifted_heatmaps: Heatmap,
    *,
    top_fraction: float = 0.2,
) -> dict[str, int | float | np.ndarray]:
    """Return per-pair values and aggregate statistics for two heatmap batches.

    Inputs should have shape ``(batch, ...)``.  Two-dimensional inputs are
    treated as a batch containing one spatial heatmap.
    """

    clean = _as_finite_array(clean_heatmaps, name="clean_heatmaps")
    shifted = _as_finite_array(shifted_heatmaps, name="shifted_heatmaps")
    if clean.shape != shifted.shape:
        raise ValueError(
            f"clean_heatmaps and shifted_heatmaps must have the same shape, got {clean.shape} and {shifted.shape}"
        )
    if clean.ndim < 2:
        raise ValueError("heatmap batches must have at least two dimensions")
    if clean.ndim == 2:
        clean = clean[None]
        shifted = shifted[None]

    pairs = [
        pair_stability(clean_map, shifted_map, top_fraction=top_fraction)
        for clean_map, shifted_map in zip(clean, shifted, strict=True)
    ]
    spearman_values = np.asarray([pair["spearman"] for pair in pairs], dtype=np.float64)
    iou_values = np.asarray([pair["top_percent_iou"] for pair in pairs], dtype=np.float64)
    valid = np.isfinite(spearman_values) & np.isfinite(iou_values)

    def summary(values: np.ndarray, reducer) -> float:
        finite = values[np.isfinite(values)]
        return float(reducer(finite)) if finite.size else float("nan")

    return {
        "n_pairs": len(pairs),
        "n_valid_pairs": int(valid.sum()),
        "top_fraction": float(top_fraction),
        "spearman_per_pair": spearman_values,
        "top_percent_iou_per_pair": iou_values,
        "mean_spearman": summary(spearman_values, np.mean),
        "std_spearman": summary(spearman_values, np.std),
        "median_spearman": summary(spearman_values, np.median),
        "mean_top_percent_iou": summary(iou_values, np.mean),
        "std_top_percent_iou": summary(iou_values, np.std),
        "median_top_percent_iou": summary(iou_values, np.median),
    }


# Readable aliases for callers that prefer metric-centric naming.
spearman_correlation = spearman_heatmap_correlation
top_saliency_iou = top_percent_iou
attribution_stability = pair_stability
batch_attribution_stability = batch_stability
