from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.shifts.corruptions import corrupt
from src.shifts.severity import CORRUPTION_PARAMETERS, parameter_for


@pytest.fixture
def textured_image() -> Image.Image:
    y, x = np.mgrid[:96, :96]
    checker = ((x // 4 + y // 4) % 2) * 100
    rgb = np.stack(
        [
            (x * 2 + checker) % 256,
            (y * 2 + checker) % 256,
            ((x + y) + checker) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb)


@pytest.mark.parametrize("name", sorted(CORRUPTION_PARAMETERS))
@pytest.mark.parametrize("severity", range(1, 6))
def test_corruption_is_deterministic_valid_rgb(textured_image, name, severity):
    first = np.asarray(corrupt(textured_image, name, severity, seed=73))
    second = np.asarray(corrupt(textured_image, name, severity, seed=73))
    assert np.array_equal(first, second)
    assert first.shape == (96, 96, 3)
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255


def test_clean_condition_is_identity(textured_image):
    clean = np.asarray(corrupt(textured_image, "jpeg", 0, seed=1))
    assert np.array_equal(clean, np.asarray(textured_image))


@pytest.mark.parametrize("name", sorted(CORRUPTION_PARAMETERS))
def test_severity_five_distorts_more_than_one(textured_image, name):
    clean = np.asarray(textured_image).astype(np.float32)
    mild = np.asarray(corrupt(textured_image, name, 1, seed=19)).astype(np.float32)
    severe = np.asarray(corrupt(textured_image, name, 5, seed=19)).astype(np.float32)
    mild_mse = np.mean((clean - mild) ** 2)
    severe_mse = np.mean((clean - severe) ** 2)
    assert severe_mse > mild_mse, (name, mild_mse, severe_mse)


def test_registered_ladders_have_five_levels():
    assert all(len(values) == 5 for values in CORRUPTION_PARAMETERS.values())
    with pytest.raises(ValueError, match="severity"):
        parameter_for("jpeg", 6)


def test_unknown_corruption_has_actionable_error(textured_image):
    with pytest.raises(ValueError, match="choose one of"):
        corrupt(textured_image, "lens_flare", 2, seed=0)
