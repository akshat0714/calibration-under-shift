from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from experiments.run_attribution import _stratified_indices
from src.attribution.gradcam import GradCAM, GradCAMPlusPlus, overlay_heatmap
from src.attribution.stability import (
    batch_stability,
    pair_stability,
    spearman_heatmap_correlation,
    top_percent_iou,
)


class TinyAttributionCNN(nn.Module):
    """A deterministic one-channel CNN with analytically simple saliency."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1)
            self.head.weight.copy_(torch.tensor([[1.0], [-1.0]]))

    def forward_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(inputs))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.forward_feature_map(inputs)).flatten(1)
        return self.head(pooled)


@pytest.fixture
def tiny_inputs() -> torch.Tensor:
    inputs = torch.zeros(2, 1, 8, 8)
    inputs[0, 0, 1:4, 2:5] = 1.0
    inputs[1, 0, 4:7, 3:7] = 2.0
    return inputs


@pytest.mark.parametrize("cam_type", [GradCAM, GradCAMPlusPlus])
def test_native_cams_have_input_shape_and_normalized_range(cam_type, tiny_inputs):
    model = TinyAttributionCNN()
    model.train()
    result = cam_type(model).explain(tiny_inputs)

    assert result.heatmaps.shape == (2, 8, 8)
    assert torch.isfinite(result.heatmaps).all()
    assert result.heatmaps.min().item() >= 0
    assert result.heatmaps.max().item() <= 1
    assert torch.equal(result.target_classes, torch.zeros(2, dtype=torch.long))
    assert torch.all((result.confidences >= 0) & (result.confidences <= 1))
    assert model.training  # Attribution temporarily evaluates, then restores the mode.
    assert result.heatmaps[0, 2, 3] == pytest.approx(1.0)
    assert result.heatmaps[0, 7, 7] == pytest.approx(0.0)


def test_gradcam_does_not_touch_input_or_parameter_gradients(tiny_inputs):
    model = TinyAttributionCNN()
    inputs_before = tiny_inputs.clone()
    assert all(parameter.grad is None for parameter in model.parameters())

    heatmaps = GradCAM(model)(tiny_inputs, targets=[0, 0])

    assert heatmaps.shape == (2, 8, 8)
    assert torch.equal(tiny_inputs, inputs_before)
    assert tiny_inputs.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_overlay_is_rgb_resized_and_bounded():
    image = torch.linspace(0, 1, steps=3 * 12 * 10).reshape(3, 12, 10)
    heatmap = np.zeros((6, 5), dtype=np.float32)
    heatmap[2:4, 2:4] = 1

    overlay = overlay_heatmap(image, heatmap, alpha=0.5)

    assert overlay.shape == (12, 10, 3)
    assert overlay.dtype == np.float32
    assert np.isfinite(overlay).all()
    assert overlay.min() >= 0 and overlay.max() <= 1


def test_identical_and_opposite_heatmaps_have_known_stability():
    clean = np.arange(25, dtype=np.float64).reshape(5, 5)
    opposite = clean.max() - clean

    assert spearman_heatmap_correlation(clean, clean) == pytest.approx(1.0)
    assert top_percent_iou(clean, clean) == pytest.approx(1.0)
    assert spearman_heatmap_correlation(clean, opposite) == pytest.approx(-1.0)
    assert top_percent_iou(clean, opposite) == pytest.approx(0.0)
    assert pair_stability(clean, opposite) == pytest.approx(
        {"spearman": -1.0, "top_percent_iou": 0.0}
    )


def test_batch_stability_returns_per_pair_and_summary_values():
    clean = np.arange(25, dtype=np.float64).reshape(5, 5)
    opposite = clean.max() - clean
    batch = np.stack([clean, clean])
    shifted = np.stack([clean, opposite])

    summary = batch_stability(batch, shifted)

    assert summary["n_pairs"] == 2
    assert summary["spearman_per_pair"] == pytest.approx([1.0, -1.0])
    assert summary["top_percent_iou_per_pair"] == pytest.approx([1.0, 0.0])
    assert summary["mean_spearman"] == pytest.approx(0.0)
    assert summary["mean_top_percent_iou"] == pytest.approx(0.5)


def test_stability_validates_shapes_and_fraction():
    with pytest.raises(ValueError, match="same shape"):
        top_percent_iou(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="top_fraction"):
        top_percent_iou(np.zeros((2, 2)), np.zeros((2, 2)), top_fraction=0)


def test_constant_heatmaps_are_degenerate_not_perfectly_stable():
    constant = np.zeros((4, 4))
    assert np.isnan(spearman_heatmap_correlation(constant, constant))
    assert np.isnan(top_percent_iou(constant, constant))
    summary = batch_stability(constant, constant)
    assert summary["n_valid_pairs"] == 0
    assert np.isnan(summary["mean_spearman"])


def test_qualitative_attribution_indices_are_class_stratified():
    evaluator = type(
        "Evaluator",
        (),
        {
            "manifest": pd.DataFrame(
                {
                    "path": [f"sample-{index}.png" for index in range(12)],
                    "label": [0] * 4 + [1] * 4 + [2] * 4,
                    "split": ["test"] * 12,
                }
            )
        },
    )()
    indices = _stratified_indices(evaluator, 6)
    assert evaluator.manifest.iloc[indices]["label"].value_counts().to_dict() == {0: 2, 1: 2, 2: 2}
