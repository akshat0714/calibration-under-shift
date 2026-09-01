"""Deterministic image corruptions that approximate low-cost device shift.

These are sensitivity-analysis models, not claims that synthetic images reproduce
any named microscope or phone. Corruptions operate in RGB at the decoded-image
stage and must only be attached to validation, calibration, or test datasets.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from numbers import Integral

import cv2
import numpy as np
from PIL import Image, ImageFilter

from src.shifts.severity import canonical_name, parameter_for


def _rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("image array must have shape HxW, HxWx3, or HxWx4")
        array = array[..., :3]
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
            array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def _pil(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.rint(array), 0, 255).astype(np.uint8))


def _defocus(array: np.ndarray, sigma: float, _rng: np.random.Generator) -> np.ndarray:
    return np.asarray(_pil(array).filter(ImageFilter.GaussianBlur(radius=sigma)))


def _motion(array: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    # Fix angle for a given seed across the severity ladder. Border reflection avoids
    # adding a black-frame artifact that is unrelated to handheld motion.
    angle = float(rng.uniform(0.0, 180.0))
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    radians = np.deg2rad(angle)
    half = (length - 1) / 2.0
    x1, y1 = center - half * np.cos(radians), center - half * np.sin(radians)
    x2, y2 = center + half * np.cos(radians), center + half * np.sin(radians)
    cv2.line(
        kernel,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        color=1.0,
        thickness=1,
        lineType=cv2.LINE_AA,
    )
    kernel /= kernel.sum()
    return cv2.filter2D(array, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)


def _gaussian_noise(array: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    normalized = array.astype(np.float32) / 255.0
    noise = rng.standard_normal(normalized.shape, dtype=np.float32)
    return np.clip(normalized + sigma * noise, 0.0, 1.0) * 255.0


def _shot_noise(array: np.ndarray, photons: float, rng: np.random.Generator) -> np.ndarray:
    normalized = array.astype(np.float32) / 255.0
    # This is a post-demosaic proxy: sample signal-dependent Poisson noise from
    # luminance, then apply the same luminance residual to all RGB channels.  It
    # avoids the rainbow speckle produced by independently sampling each channel
    # while preserving chromatic structure.  The Poisson draw is unbiased before
    # clipping; clipping at the public image boundary can shift the final mean.
    luminance_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luminance = normalized @ luminance_weights
    noisy_luminance = rng.poisson(luminance * photons).astype(np.float32) / photons
    luminance_residual = (noisy_luminance - luminance)[..., None]
    return np.clip(normalized + luminance_residual, 0.0, 1.0) * 255.0


def _jpeg(array: np.ndarray, quality: int, _rng: np.random.Generator) -> np.ndarray:
    buffer = io.BytesIO()
    _pil(array).save(buffer, format="JPEG", quality=quality, subsampling=2, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"))


def _resample(array: np.ndarray, factor: float, _rng: np.random.Generator) -> np.ndarray:
    height, width = array.shape[:2]
    low_size = (max(1, round(width / factor)), max(1, round(height / factor)))
    reduced = cv2.resize(array, low_size, interpolation=cv2.INTER_AREA)
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)


def _illumination(
    array: np.ndarray, parameter: tuple[float, float], rng: np.random.Generator
) -> np.ndarray:
    base_gamma, gain_delta = parameter
    # A seed fixes the direction of gamma and chromatic bias; only magnitude changes
    # across severity. This makes paired severity comparisons interpretable.
    bright_branch = bool(rng.integers(0, 2))
    gamma = (1.0 / base_gamma) if bright_branch else base_gamma
    direction = rng.normal(size=3)
    direction -= direction.mean()
    max_abs = np.abs(direction).max()
    if max_abs > 0:
        direction /= max_abs
    gains = 1.0 + gain_delta * direction
    normalized = array.astype(np.float32) / 255.0
    shifted = np.power(np.clip(normalized, 1e-6, 1.0), gamma) * gains[None, None, :]
    return np.clip(shifted, 0.0, 1.0) * 255.0


_IMPLEMENTATIONS: dict[str, Callable] = {
    "defocus_blur": _defocus,
    "motion_blur": _motion,
    "gaussian_noise": _gaussian_noise,
    "shot_noise": _shot_noise,
    "jpeg": _jpeg,
    "resample": _resample,
    "illumination": _illumination,
}


def _scale_spatial_parameter(name: str, parameter, shape: tuple[int, ...]):
    """Express pixel-space ladders at a 224-pixel reference short side."""

    scale = min(shape[:2]) / 224.0
    if name == "defocus_blur":
        return float(parameter) * scale
    if name == "motion_blur":
        length = max(1, int(round(float(parameter) * scale)))
        return length if length % 2 else length + 1
    return parameter


def corrupt(
    image: Image.Image | np.ndarray,
    name: str,
    severity: int,
    seed: int = 0,
) -> Image.Image:
    """Apply one registered corruption and return an RGB PIL image.

    ``severity=0`` is the clean identity condition. For severities 1--5, the same
    seed makes each condition reproducible and fixes nuisance choices such as the
    motion angle and illumination direction. Test images stay paired across the
    ladder; Poisson draws are deterministic per condition but not coupled across
    different count scales.
    """

    canonical = canonical_name(name)
    if isinstance(severity, bool) or not isinstance(severity, Integral):
        raise ValueError("severity must be an integer from 0 through 5")
    severity = int(severity)
    if severity not in range(0, 6):
        raise ValueError("severity must be an integer from 0 through 5")
    array = _rgb_array(image)
    if severity == 0:
        return _pil(array.copy())
    parameter = parameter_for(canonical, severity)
    parameter = _scale_spatial_parameter(canonical, parameter, array.shape)
    rng = np.random.default_rng(seed)
    output = _IMPLEMENTATIONS[canonical](array, parameter, rng)
    if output.shape != array.shape:
        raise AssertionError(f"corruption changed image shape from {array.shape} to {output.shape}")
    return _pil(output)


def make_corruption(name: str, severity: int, base_seed: int = 0):
    """Create a dataset callback with deterministic per-sample seeds."""

    canonical = canonical_name(name)

    def apply(image: Image.Image, index: int) -> Image.Image:
        # The large odd multiplier separates adjacent samples without Python's salted hash.
        sample_seed = (base_seed + index * 1_000_003) % (2**32)
        return corrupt(image, canonical, severity, sample_seed)

    return apply
