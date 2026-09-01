"""Versioned corruption severity ladders and their physical interpretation."""

from __future__ import annotations

CORRUPTION_PARAMETERS: dict[str, tuple[float | int | tuple[float, float], ...]] = {
    "defocus_blur": (0.5, 1.0, 2.0, 3.0, 5.0),
    "motion_blur": (3, 5, 9, 13, 17),
    "gaussian_noise": (0.02, 0.04, 0.08, 0.12, 0.18),
    "shot_noise": (60.0, 25.0, 12.0, 5.0, 3.0),
    "jpeg": (80, 60, 40, 25, 12),
    "resample": (1.25, 1.5, 2.0, 3.0, 4.0),
    "illumination": (
        (0.85, 0.05),
        (0.75, 0.10),
        (0.65, 0.15),
        (0.58, 0.20),
        (0.50, 0.25),
    ),
}

ALIASES = {
    "blur": "defocus_blur",
    "defocus": "defocus_blur",
    "gaussian_blur": "defocus_blur",
    "sensor_noise": "gaussian_noise",
    "poisson_noise": "shot_noise",
    "low_light": "shot_noise",
    "jpeg_compression": "jpeg",
    "downsample": "resample",
    "down_up_resampling": "resample",
    "gamma": "illumination",
    "white_balance": "illumination",
}

PHYSICAL_MECHANISMS = {
    "defocus_blur": "lower numerical-aperture phone optics or autofocus error",
    "motion_blur": "handheld capture without a fixed microscope stage",
    "gaussian_noise": "read noise from a small CMOS sensor",
    "shot_noise": "photon-limited acquisition under low illumination",
    "jpeg": "lossy smartphone image encoding and transfer",
    "resample": "lower magnification or sensor pixel density",
    "illumination": "light-source spectrum and phone image-signal-processor variation",
}


def canonical_name(name: str) -> str:
    """Normalize a public corruption name or raise a useful error."""

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in CORRUPTION_PARAMETERS:
        available = ", ".join(sorted(CORRUPTION_PARAMETERS))
        raise ValueError(f"unknown corruption {name!r}; choose one of: {available}")
    return normalized


def parameter_for(name: str, severity: int):
    """Return the registered parameter for a 1-indexed severity level."""

    canonical = canonical_name(name)
    if severity not in range(1, 6):
        raise ValueError("severity must be an integer from 1 through 5")
    return CORRUPTION_PARAMETERS[canonical][severity - 1]


def corruption_names() -> tuple[str, ...]:
    return tuple(CORRUPTION_PARAMETERS)
