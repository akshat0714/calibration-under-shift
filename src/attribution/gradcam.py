"""Dependency-light Grad-CAM utilities for spatial image classifiers.

The implementation only depends on PyTorch and NumPy.  Models are expected to
return ``(batch, classes)`` logits and expose ``gradcam_target_layer`` (as
``VisionClassifier`` does), unless a target layer is supplied explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CAMMethod = Literal["gradcam", "gradcam++"]


@dataclass(frozen=True)
class GradCAMResult:
    """Outputs needed for both quantitative and qualitative attribution plots."""

    heatmaps: torch.Tensor
    logits: torch.Tensor
    target_classes: torch.Tensor
    confidences: torch.Tensor
    method: CAMMethod


def _normalize_method(method: str) -> CAMMethod:
    normalized = method.lower().replace("-", "").replace("_", "").replace(" ", "")
    if normalized == "gradcam":
        return "gradcam"
    if normalized in {"gradcam++", "gradcamplusplus", "plusplus"}:
        return "gradcam++"
    raise ValueError("method must be 'gradcam' or 'gradcam++'")


def _resolve_target_layer(model: nn.Module, target_layer: nn.Module | None) -> nn.Module:
    if target_layer is None:
        try:
            target_layer = model.gradcam_target_layer  # type: ignore[attr-defined]
        except AttributeError as error:
            raise ValueError(
                "target_layer is required when model does not expose gradcam_target_layer"
            ) from error
    if not isinstance(target_layer, nn.Module):
        raise TypeError("target_layer must be a torch.nn.Module")
    return target_layer


def _extract_logits(output: object) -> torch.Tensor:
    """Extract logits while tolerating the common ``(logits, ...)`` convention."""

    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(output, tuple | list) and output and isinstance(output[0], torch.Tensor):
        logits = output[0]
    elif isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        logits = output["logits"]
    else:
        raise TypeError("model must return logits as a tensor, first tuple item, or 'logits' value")
    if logits.ndim != 2:
        raise ValueError(
            f"model logits must have shape (batch, classes), got {tuple(logits.shape)}"
        )
    return logits


def _target_tensor(
    targets: int | torch.Tensor | list[int] | tuple[int, ...] | np.ndarray | None,
    logits: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_classes = logits.shape
    if targets is None:
        result = logits.argmax(dim=1)
    elif isinstance(targets, int):
        result = torch.full((batch_size,), targets, dtype=torch.long, device=logits.device)
    else:
        result = torch.as_tensor(targets, dtype=torch.long, device=logits.device)
        if result.ndim == 0:
            result = result.repeat(batch_size)
        if result.ndim != 1 or result.numel() != batch_size:
            raise ValueError("targets must contain exactly one class index per input")
    if torch.any((result < 0) | (result >= num_classes)):
        raise ValueError(f"target class indices must be in [0, {num_classes})")
    return result


def _normalize_heatmaps(heatmaps: torch.Tensor, eps: float) -> torch.Tensor:
    flat = heatmaps.flatten(start_dim=1)
    minima = flat.min(dim=1).values[:, None, None]
    maxima = flat.max(dim=1).values[:, None, None]
    ranges = maxima - minima
    normalized = (heatmaps - minima) / ranges.clamp_min(eps)
    # A spatially constant CAM contains no localizing information.
    return torch.where(ranges > eps, normalized, torch.zeros_like(normalized)).clamp_(0, 1)


class GradCAM:
    """Generate Grad-CAM or Grad-CAM++ heatmaps from the final spatial layer.

    Hooks are installed only for the duration of each call.  Parameter ``.grad``
    buffers and the caller's input tensor are left untouched, and the model's
    original train/eval mode is restored before returning.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        *,
        method: str = "gradcam",
        eps: float = 1e-8,
    ) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.model = model
        self.target_layer = _resolve_target_layer(model, target_layer)
        self.method = _normalize_method(method)
        self.eps = float(eps)

    def _weights(self, activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
        if self.method == "gradcam":
            return gradients.mean(dim=(2, 3), keepdim=True)

        # Grad-CAM++ channel weights from Chattopadhyay et al.  The widely used
        # first-gradient formulation avoids constructing expensive higher-order
        # graphs while retaining the per-pixel alpha weighting.
        gradient_squared = gradients.square()
        gradient_cubed = gradient_squared * gradients
        activation_sum = activations.sum(dim=(2, 3), keepdim=True)
        denominator = 2 * gradient_squared + activation_sum * gradient_cubed
        safe_denominator = torch.where(
            denominator.abs() > self.eps,
            denominator,
            torch.full_like(denominator, self.eps),
        )
        alpha = gradient_squared / safe_denominator
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))
        return (alpha * gradients.clamp_min(0)).sum(dim=(2, 3), keepdim=True)

    def explain(
        self,
        inputs: torch.Tensor,
        targets: int | torch.Tensor | list[int] | tuple[int, ...] | np.ndarray | None = None,
        *,
        resize: bool = True,
    ) -> GradCAMResult:
        """Return normalized heatmaps, logits, selected classes, and confidence.

        ``targets=None`` explains each sample's predicted class.  A scalar target
        applies to the whole batch. Otherwise provide one class index per sample.
        Heatmaps have shape ``(batch, height, width)`` and values in ``[0, 1]``.
        """

        if not isinstance(inputs, torch.Tensor):
            raise TypeError("inputs must be a torch.Tensor")
        if inputs.ndim != 4:
            raise ValueError(
                f"inputs must have shape (batch, channels, height, width), got {inputs.shape}"
            )
        if inputs.shape[0] == 0:
            raise ValueError("inputs must contain at least one sample")
        if not torch.is_floating_point(inputs):
            raise TypeError("inputs must have a floating-point dtype")

        captured: list[torch.Tensor] = []

        def capture_activation(
            _module: nn.Module, _arguments: tuple[object, ...], output: object
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError("Grad-CAM target layer must return a tensor")
            captured.append(output)

        handle = self.target_layer.register_forward_hook(capture_activation)
        training_states = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            # Re-enable autograd even when the caller is in inference mode.  A
            # detached clone also ensures frozen backbones still produce an
            # activation graph without modifying the user's tensor.
            with torch.inference_mode(False), torch.enable_grad():
                attribution_inputs = inputs.detach().clone().requires_grad_(True)
                logits = _extract_logits(self.model(attribution_inputs))
                if logits.shape[0] != inputs.shape[0]:
                    raise ValueError("model output batch size does not match inputs")
                if len(captured) != 1:
                    raise RuntimeError(
                        "target layer must run exactly once per model forward. "
                        f"observed {len(captured)} calls"
                    )
                activations = captured[0]
                if activations.ndim != 4:
                    raise ValueError(
                        "Grad-CAM target activations must have shape "
                        f"(batch, channels, height, width), got {tuple(activations.shape)}"
                    )
                if activations.shape[0] != inputs.shape[0]:
                    raise ValueError("target activation batch size does not match inputs")

                target_classes = _target_tensor(targets, logits)
                selected_scores = logits.gather(1, target_classes[:, None]).sum()
                gradients = torch.autograd.grad(
                    selected_scores,
                    activations,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]

                # Float32 accumulation is more stable under mixed-precision inference.
                activation_values = activations.float()
                gradient_values = gradients.float()
                weights = self._weights(activation_values, gradient_values)
                heatmaps = (weights * activation_values).sum(dim=1).relu()
                if resize and heatmaps.shape[-2:] != inputs.shape[-2:]:
                    heatmaps = F.interpolate(
                        heatmaps[:, None],
                        size=inputs.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                heatmaps = _normalize_heatmaps(heatmaps, self.eps)
                probabilities = logits.float().softmax(dim=1)
                confidences = probabilities.gather(1, target_classes[:, None])[:, 0]
        finally:
            handle.remove()
            # Restore mixed module states too (for example, MC-dropout with
            # BatchNorm still in eval mode), not merely the root flag.
            for module, training in training_states:
                module.training = training

        return GradCAMResult(
            heatmaps=heatmaps.detach(),
            logits=logits.detach(),
            target_classes=target_classes.detach(),
            confidences=confidences.detach(),
            method=self.method,
        )

    def __call__(
        self,
        inputs: torch.Tensor,
        targets: int | torch.Tensor | list[int] | tuple[int, ...] | np.ndarray | None = None,
        *,
        resize: bool = True,
    ) -> torch.Tensor:
        """Return only the normalized heatmaps from :meth:`explain`."""

        return self.explain(inputs, targets, resize=resize).heatmaps


class GradCAMPlusPlus(GradCAM):
    """Convenience subclass configured for Grad-CAM++."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | None = None,
        *,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(model, target_layer, method="gradcam++", eps=eps)


def generate_gradcam(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: int | torch.Tensor | list[int] | tuple[int, ...] | np.ndarray | None = None,
    *,
    target_layer: nn.Module | None = None,
    method: str = "gradcam",
    resize: bool = True,
) -> torch.Tensor:
    """Functional interface for generating a batch of Grad-CAM heatmaps."""

    return GradCAM(model, target_layer, method=method)(inputs, target_classes, resize=resize)


def gradcam(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: int | torch.Tensor | list[int] | tuple[int, ...] | np.ndarray | None = None,
    *,
    target_layer: nn.Module | None = None,
    resize: bool = True,
) -> torch.Tensor:
    """Generate standard Grad-CAM heatmaps."""

    return generate_gradcam(
        model,
        inputs,
        target_classes,
        target_layer=target_layer,
        method="gradcam",
        resize=resize,
    )


def gradcam_plus_plus(
    model: nn.Module,
    inputs: torch.Tensor,
    target_classes: int | torch.Tensor | list[int] | tuple[int, ...] | np.ndarray | None = None,
    *,
    target_layer: nn.Module | None = None,
    resize: bool = True,
) -> torch.Tensor:
    """Generate Grad-CAM++ heatmaps."""

    return generate_gradcam(
        model,
        inputs,
        target_classes,
        target_layer=target_layer,
        method="gradcam++",
        resize=resize,
    )


def _display_rgb(image: np.ndarray | torch.Tensor) -> np.ndarray:
    values = image.detach().cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)
    if values.ndim == 4 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    elif values.ndim == 3 and values.shape[-1] in {1, 3, 4}:
        values = values[..., :3]
        if values.shape[-1] == 1:
            values = np.repeat(values, 3, axis=2)
    elif values.ndim == 3 and values.shape[0] in {1, 3, 4}:
        values = np.moveaxis(values[:3], 0, -1)
        if values.shape[-1] == 1:
            values = np.repeat(values, 3, axis=2)
    else:
        raise ValueError("image must be grayscale, HWC RGB(A), or CHW RGB(A)")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError("image must contain finite numeric values")

    is_integer = np.issubdtype(values.dtype, np.integer)
    integer_max = float(np.iinfo(values.dtype).max) if is_integer else None
    values = values.astype(np.float32)
    lower, upper = float(values.min()), float(values.max())
    if integer_max is not None:
        values /= integer_max
    elif lower >= 0 and upper <= 1:
        pass
    elif lower >= 0 and upper <= 255:
        values /= 255.0
    elif upper > lower:
        values = (values - lower) / (upper - lower)
    else:
        values = np.zeros_like(values)
    return np.clip(values, 0, 1)


def _display_heatmap(heatmap: np.ndarray | torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    if isinstance(heatmap, torch.Tensor):
        values = heatmap.detach().float().cpu().numpy()
    else:
        values = np.asarray(heatmap, dtype=np.float32)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError("heatmap must be two-dimensional after removing singleton axes")
    if not np.isfinite(values).all():
        raise ValueError("heatmap must contain only finite values")
    if values.shape != size:
        values = (
            F.interpolate(
                torch.from_numpy(values)[None, None],
                size=size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .numpy()
            .astype(np.float32)
        )

    lower, upper = float(values.min()), float(values.max())
    if lower >= 0 and upper <= 1:
        return np.clip(values, 0, 1)
    if upper > lower:
        return ((values - lower) / (upper - lower)).astype(np.float32)
    return np.ones_like(values) if upper > 0 else np.zeros_like(values)


def _colorize(heatmap: np.ndarray, colormap: str) -> np.ndarray:
    normalized_name = colormap.lower().replace("-", "_")
    if normalized_name == "red":
        return np.stack((heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap)), axis=-1)
    if normalized_name != "jet":
        raise ValueError("colormap must be 'jet' or 'red'")
    # A compact Jet-style map avoids importing a plotting library in inference code.
    red = np.clip(1.5 - np.abs(4 * heatmap - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * heatmap - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * heatmap - 1), 0, 1)
    return np.stack((red, green, blue), axis=-1).astype(np.float32)


def overlay_heatmap(
    image: np.ndarray | torch.Tensor,
    heatmap: np.ndarray | torch.Tensor,
    *,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Blend one heatmap onto an image and return display-ready float RGB.

    The result has shape ``(height, width, 3)`` and range ``[0, 1]``.  Saliency
    controls local opacity, so zero-saliency pixels remain unchanged.
    """

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    rgb = _display_rgb(image)
    saliency = _display_heatmap(heatmap, cast(tuple[int, int], rgb.shape[:2]))
    colors = _colorize(saliency, colormap)
    local_alpha = alpha * saliency[..., None]
    return np.clip((1 - local_alpha) * rgb + local_alpha * colors, 0, 1).astype(np.float32)


def overlay_heatmaps(
    images: np.ndarray | torch.Tensor,
    heatmaps: np.ndarray | torch.Tensor,
    *,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """Vector-friendly wrapper that overlays corresponding image/heatmap pairs."""

    image_values = images.detach().cpu() if isinstance(images, torch.Tensor) else np.asarray(images)
    if isinstance(heatmaps, torch.Tensor):
        heatmap_values = heatmaps.detach().cpu()
    else:
        heatmap_values = np.asarray(heatmaps)
    if len(image_values) != len(heatmap_values):
        raise ValueError("images and heatmaps must contain the same number of samples")
    if len(image_values) == 0:
        raise ValueError("images and heatmaps must not be empty")
    return np.stack(
        [
            overlay_heatmap(image, heatmap, alpha=alpha, colormap=colormap)
            for image, heatmap in zip(image_values, heatmap_values, strict=True)
        ]
    )
