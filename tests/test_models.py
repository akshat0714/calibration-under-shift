from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torchvision.transforms import InterpolationMode

from src.data.transforms import build_transform, preprocessing_from_model_config
from src.models.build import build_model, set_backbone_trainable
from src.train import _run_epoch


@pytest.mark.parametrize(
    ("name", "size"),
    [("resnet50", 64), ("mobilenet_v3_large", 64), ("tiny_cnn", 32)],
)
def test_torchvision_backbones_forward_and_expose_features(name, size):
    model = build_model(name, num_classes=4, pretrained=False, dropout=0.2)
    model.eval()
    with torch.inference_mode():
        inputs = torch.randn(2, 3, size, size)
        logits = model(inputs)
        features = model.forward_features(inputs)
    assert logits.shape == (2, 4)
    assert features.shape == (2, model.feature_dim)
    assert model.gradcam_target_layer is not None


def test_freeze_leaves_only_head_trainable():
    model = build_model("resnet50", num_classes=2, pretrained=False)
    set_backbone_trainable(model, False)
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_frozen_backbone_keeps_batchnorm_statistics_fixed():
    model = build_model("tiny_cnn", num_classes=2, pretrained=False)
    set_backbone_trainable(model, False)
    batch_norm = next(
        module for module in model.backbone.modules() if isinstance(module, torch.nn.BatchNorm2d)
    )
    before = batch_norm.running_mean.clone()
    loader = [{"image": torch.randn(4, 3, 32, 32), "label": torch.tensor([0, 1, 0, 1])}]
    optimizer = torch.optim.SGD(model.head.parameters(), lr=0.01)

    _run_epoch(model, loader, torch.nn.CrossEntropyLoss(), torch.device("cpu"), optimizer)

    assert torch.equal(batch_norm.running_mean, before)


def test_model_factory_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_model("vit-gigantic", num_classes=2, pretrained=False)


def test_xception_config_matches_pinned_timm_preprocessing():
    with Path("configs/smids_xception.yaml").open(encoding="utf-8") as handle:
        model_config = yaml.safe_load(handle)["model"]
    spec = preprocessing_from_model_config(model_config)
    model = build_model("xception", num_classes=3, pretrained=False)
    pretrained = model.backbone.pretrained_cfg

    assert tuple(spec["mean"]) == tuple(pretrained["mean"]) == (0.5, 0.5, 0.5)
    assert tuple(spec["std"]) == tuple(pretrained["std"]) == (0.5, 0.5, 0.5)
    assert spec["interpolation"] == pretrained["interpolation"] == "bicubic"
    assert spec["crop_pct"] == pytest.approx(pretrained["crop_pct"])
    assert model_config["input_size"] == pretrained["input_size"][-1] == 299

    evaluation = build_transform(False, model_config["input_size"], **spec)
    assert evaluation.transforms[0].size == 333
    assert evaluation.transforms[0].interpolation == InterpolationMode.BICUBIC
    assert tuple(evaluation.transforms[-1].mean) == (0.5, 0.5, 0.5)
    train = build_transform(True, model_config["input_size"], **spec)
    assert train.transforms[0].interpolation == InterpolationMode.BICUBIC

    constant = np.zeros((340, 340, 3), dtype=np.uint8)
    constant[..., 1] = 128
    constant[..., 2] = 255
    transformed = evaluation(Image.fromarray(constant))
    expected = torch.tensor([-1.0, 2 * (128 / 255) - 1, 1.0])
    assert torch.allclose(transformed[:, 100, 100], expected, atol=1e-6)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"mean": [0.5, 0.5]}, "mean/std"),
        ({"std": [1.0, 0.0, 1.0]}, "std must be positive"),
        ({"interpolation": "nearest"}, "interpolation"),
        ({"crop_pct": 1.1}, "crop_pct"),
    ],
)
def test_preprocessing_contract_rejects_invalid_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        preprocessing_from_model_config({"backbone": "tiny_cnn", **overrides})
