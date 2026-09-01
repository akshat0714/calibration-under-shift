from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from src.data.datasets import ManifestImageDataset
from src.shifts.corruptions import corrupt, make_corruption
from src.shifts.severity import (
    CORRUPTION_PARAMETERS,
    corruption_protocol_digest,
    parameter_for,
)


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


@pytest.fixture
def smooth_nonperiodic_image() -> Image.Image:
    """Natural-ish blobs and an edge avoid periodic alias coincidences."""

    y, x = np.mgrid[:224, :224]
    array = np.full((224, 224, 3), 50.0, dtype=np.float64)
    for center_x, center_y, sigma, color in (
        (55, 70, 18, (180, 80, 120)),
        (145, 130, 32, (50, 160, 200)),
        (100, 190, 12, (210, 190, 60)),
    ):
        weight = np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma**2))
        array += weight[..., None] * np.asarray(color)
    array += ((x + 2 * y) > 300)[..., None] * np.asarray((20, 40, 15))
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("name", sorted(CORRUPTION_PARAMETERS))
@pytest.mark.parametrize("severity", range(1, 6))
def test_corruption_is_deterministic_valid_rgb(textured_image, name, severity):
    first_image = corrupt(textured_image, name, severity, seed=73)
    first = np.asarray(first_image)
    second = np.asarray(corrupt(textured_image, name, severity, seed=73))
    assert np.array_equal(first, second)
    assert first_image.mode == "RGB"
    assert first.shape == (96, 96, 3)
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255


def test_clean_condition_is_identity(textured_image):
    clean = np.asarray(corrupt(textured_image, "jpeg", 0, seed=1))
    assert np.array_equal(clean, np.asarray(textured_image))


def test_corruption_callback_runs_in_spawned_data_loader_worker(tmp_path, textured_image):
    image_path = tmp_path / "sample.png"
    textured_image.save(image_path)
    dataset = ManifestImageDataset(
        pd.DataFrame([{"path": str(image_path), "label": 0, "split": "test"}]),
        split="test",
        corruption=make_corruption("jpeg", severity=2, base_seed=19),
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        multiprocessing_context="spawn",
    )

    sample = next(iter(loader))

    assert sample["image"].mode == "RGB"
    assert sample["label"] == 0


@pytest.mark.parametrize("name", sorted(CORRUPTION_PARAMETERS))
def test_severity_five_distorts_more_than_one(textured_image, name):
    clean = np.asarray(textured_image).astype(np.float32)
    mild = np.asarray(corrupt(textured_image, name, 1, seed=19)).astype(np.float32)
    severe = np.asarray(corrupt(textured_image, name, 5, seed=19)).astype(np.float32)
    mild_mse = np.mean((clean - mild) ** 2)
    severe_mse = np.mean((clean - severe) ** 2)
    assert severe_mse > mild_mse, (name, mild_mse, severe_mse)


@pytest.mark.parametrize("name", sorted(CORRUPTION_PARAMETERS))
def test_psnr_strictly_decreases_across_severity_ladder(smooth_nonperiodic_image, name):
    clean = np.asarray(smooth_nonperiodic_image).astype(np.float32)
    psnr_values = []
    for severity in range(1, 6):
        shifted = np.asarray(corrupt(smooth_nonperiodic_image, name, severity, seed=19)).astype(
            np.float32
        )
        mean_squared_error = float(np.mean((clean - shifted) ** 2))
        psnr_values.append(float(10.0 * np.log10((255.0**2) / mean_squared_error)))

    assert all(
        milder > stronger
        for milder, stronger in zip(
            psnr_values[:-1],
            psnr_values[1:],
            strict=True,
        )
    ), (name, psnr_values)


def test_shot_noise_residual_is_channel_correlated():
    clean = np.empty((96, 96, 3), dtype=np.uint8)
    clean[...] = [80, 120, 160]
    shifted = np.asarray(corrupt(clean, "shot_noise", 3, seed=73)).astype(np.float32)
    residual = shifted - clean.astype(np.float32)
    correlations = np.corrcoef(residual.reshape(-1, 3), rowvar=False)
    assert np.all(correlations[np.triu_indices(3, k=1)] > 0.99)


def test_shot_noise_is_achromatic_on_neutral_input():
    clean = np.full((128, 128, 3), 128, dtype=np.uint8)
    shifted = np.asarray(corrupt(clean, "shot_noise", 4, seed=73))
    assert np.array_equal(shifted[..., 0], shifted[..., 1])
    assert np.array_equal(shifted[..., 1], shifted[..., 2])
    assert shifted[..., 0].std() > 0


def test_shot_noise_variance_is_signal_dependent():
    clean = np.zeros((128, 384, 3), dtype=np.uint8)
    clean[:, 128:256] = 48
    clean[:, 256:] = 192
    shifted = np.asarray(corrupt(clean, "shot_noise", 3, seed=73)).astype(np.float32)
    residual = shifted[..., 0] - clean[..., 0].astype(np.float32)
    assert np.count_nonzero(residual[:, :128]) == 0
    assert residual[:, 256:].std() > residual[:, 128:256].std() > 0


def test_fixed_illumination_seed_darkens_with_green_teal_bias():
    clean = np.full((32, 32, 3), 128, dtype=np.uint8)
    shifted = np.asarray(corrupt(clean, "illumination", 5, seed=1729))
    channel_means = shifted.mean(axis=(0, 1))
    assert np.all(channel_means < 128)
    assert channel_means[0] < channel_means[2] < channel_means[1]


def test_motion_blur_support_grows_at_hushem_scale():
    impulse = np.zeros((131, 131, 3), dtype=np.uint8)
    impulse[65, 65] = 255
    supports = []
    peaks = []
    for severity in range(1, 6):
        shifted = np.asarray(corrupt(impulse, "motion_blur", severity, seed=73))[..., 0]
        supports.append(int(np.count_nonzero(shifted)))
        peaks.append(int(shifted.max()))
    assert all(a < b for a, b in zip(supports[:-1], supports[1:], strict=True))
    assert all(a > b for a, b in zip(peaks[:-1], peaks[1:], strict=True))
    assert supports[0] > 1


def test_resample_intermediate_sizes_are_distinct_at_hushem_scale():
    factors = CORRUPTION_PARAMETERS["resample"]
    widths = [round(131 / factor) for factor in factors]
    assert all(a > b for a, b in zip(widths[:-1], widths[1:], strict=True))


def test_corruption_protocol_digest_tracks_registered_ladders(monkeypatch):
    original = corruption_protocol_digest()
    monkeypatch.setitem(CORRUPTION_PARAMETERS, "motion_blur", (3, 5, 7, 9, 11))
    assert corruption_protocol_digest() != original


def test_registered_ladders_have_five_levels():
    assert all(len(values) == 5 for values in CORRUPTION_PARAMETERS.values())
    assert all(
        a < b
        for a, b in zip(
            CORRUPTION_PARAMETERS["motion_blur"][:-1],
            CORRUPTION_PARAMETERS["motion_blur"][1:],
            strict=True,
        )
    )
    assert all(
        a < b
        for a, b in zip(
            CORRUPTION_PARAMETERS["resample"][:-1],
            CORRUPTION_PARAMETERS["resample"][1:],
            strict=True,
        )
    )
    assert all(
        a > b
        for a, b in zip(
            CORRUPTION_PARAMETERS["shot_noise"][:-1],
            CORRUPTION_PARAMETERS["shot_noise"][1:],
            strict=True,
        )
    )
    with pytest.raises(ValueError, match="severity"):
        parameter_for("jpeg", 6)
    with pytest.raises(ValueError, match="integer"):
        parameter_for("jpeg", 1.0)


def test_unknown_corruption_has_actionable_error(textured_image):
    with pytest.raises(ValueError, match="choose one of"):
        corrupt(textured_image, "lens_flare", 2, seed=0)
    with pytest.raises(ValueError, match="choose one of"):
        corrupt(textured_image, "lens_flare", 0, seed=0)
    with pytest.raises(ValueError, match="integer"):
        corrupt(textured_image, "jpeg", 1.0, seed=0)
