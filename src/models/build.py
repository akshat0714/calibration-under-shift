"""Backbone factory for ResNet50, Xception, and MobileNetV3-Large."""

from __future__ import annotations

from collections.abc import Iterable

import timm
import torch
from torch import nn
from torchvision import models


class VisionClassifier(nn.Module):
    """Expose pooled and spatial features through one backbone-neutral interface."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float,
        adapter: str,
        name: str,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feature_dim, num_classes))
        self.adapter = adapter
        self.backbone_name = name
        self.num_classes = num_classes
        self.feature_dim = feature_dim

    def forward_feature_map(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.adapter in {"resnet", "mobilenet", "tiny"}:
            return self.backbone(inputs)
        if self.adapter == "timm":
            return self.backbone.forward_features(inputs)
        raise RuntimeError(f"unsupported model adapter: {self.adapter}")

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        feature_map = self.forward_feature_map(inputs)
        if self.adapter == "timm":
            pooled = self.backbone.forward_head(feature_map, pre_logits=True)
        else:
            pooled = self.pool(feature_map).flatten(1)
        return pooled

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Return the final spatial feature-producing module."""

        if self.adapter == "resnet":
            return self.backbone[-1]
        if self.adapter == "mobilenet":
            return self.backbone[-1]
        if self.adapter == "tiny":
            return self.backbone[-1]
        # Timm Xception exposes its final separable-convolution block here.
        for candidate in ("act4", "conv4", "blocks"):
            if hasattr(self.backbone, candidate):
                layer = getattr(self.backbone, candidate)
                if isinstance(layer, nn.Sequential):
                    return layer[-1]
                return layer
        raise RuntimeError(f"cannot locate Grad-CAM target for {self.backbone_name}")


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> VisionClassifier:
    """Construct a classifier with an explicit dropout layer before its head."""

    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    normalized = name.lower().replace("-", "_")
    if normalized in {"resnet50", "resnet_50"}:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.resnet50(weights=weights)
        backbone = nn.Sequential(*list(base.children())[:-2])
        return VisionClassifier(
            backbone, base.fc.in_features, num_classes, dropout, "resnet", "resnet50"
        )
    if normalized in {"mobilenetv3_large", "mobilenet_v3_large", "mobilenetv3"}:
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        feature_dim = base.classifier[0].in_features
        return VisionClassifier(
            base.features,
            feature_dim,
            num_classes,
            dropout,
            "mobilenet",
            "mobilenet_v3_large",
        )
    if normalized in {"xception", "legacy_xception"}:
        backbone = timm.create_model(
            "legacy_xception", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        return VisionClassifier(
            backbone,
            int(backbone.num_features),
            num_classes,
            dropout,
            "timm",
            "xception",
        )
    if normalized in {"tiny_cnn", "tiny"}:
        # CI/demo-only architecture. It is never part of the scientific backbone grid.
        backbone = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ),
        )
        return VisionClassifier(backbone, 32, num_classes, dropout, "tiny", "tiny_cnn")
    raise ValueError("unknown backbone. Choose resnet50, xception, or mobilenet_v3_large")


def set_backbone_trainable(model: VisionClassifier, trainable: bool) -> None:
    """Freeze/unfreeze the feature extractor while always training the new head."""

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(trainable)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)
