from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.data.audit import audit_metadata
from src.data.datasets import ManifestImageDataset, assert_fit_split, load_manifest
from src.data.prepare import _infer_class, discover_decodable_images
from src.data.splits import (
    STANDARD_FRACTIONS,
    cross_validation_manifest,
    grouped_stratified_manifest,
    kromp_high_quality_label,
    stratified_manifest,
)
from src.data.transforms import build_transform


def _write_manifest(tmp_path: Path, frame: pd.DataFrame, name: str = "manifest.csv") -> Path:
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


def _balanced_metadata(classes: int = 4, samples_per_class: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"path": f"class-{label}/sample-{index:03d}.png", "label": label}
            for label in range(classes)
            for index in range(samples_per_class)
        ]
    )


def test_manifest_rejects_patient_leakage_across_splits(tmp_path):
    frame = pd.DataFrame(
        {
            "path": ["patient-7/a.png", "patient-7/b.png"],
            "label": [0, 1],
            "split": ["train", "test"],
            "patient_id": ["patient-7", "patient-7"],
        }
    )

    with pytest.raises(ValueError, match="patient leakage"):
        load_manifest(_write_manifest(tmp_path, frame))


def test_manifest_rejects_path_leakage_across_splits(tmp_path):
    frame = pd.DataFrame(
        {
            "path": ["same.png", "same.png"],
            "label": [0, 0],
            "split": ["train", "calibration"],
        }
    )

    with pytest.raises(ValueError, match="sample paths occur in multiple splits"):
        load_manifest(_write_manifest(tmp_path, frame))


def test_grouped_manifest_is_disjoint_deterministic_and_balanced():
    # Giving every patient one sample from each class makes both class and group
    # balance exactly checkable rather than dependent on a lucky random draw.
    metadata = pd.DataFrame(
        [
            {
                "path": f"patient-{patient:03d}/class-{label}.png",
                "patient_id": f"patient-{patient:03d}",
                "label": label,
            }
            for patient in range(80)
            for label in (0, 1)
        ]
    )

    first = grouped_stratified_manifest(metadata, seed=31)
    second = grouped_stratified_manifest(metadata, seed=31)

    pd.testing.assert_frame_equal(first, second)
    assert first["path"].is_unique
    assert first.groupby("patient_id")["split"].nunique().eq(1).all()
    assert set(first["split"]) == set(STANDARD_FRACTIONS)
    observed_fractions = first["split"].value_counts(normalize=True).to_dict()
    assert observed_fractions == pytest.approx(STANDARD_FRACTIONS)
    for _, part in first.groupby("split"):
        assert part["label"].value_counts(normalize=True).to_dict() == pytest.approx(
            {0: 0.5, 1: 0.5}
        )


def test_stratified_manifest_has_exact_70_10_10_10_allocation():
    metadata = _balanced_metadata()

    manifest = stratified_manifest(metadata, seed=47)

    assert manifest["path"].is_unique
    assert set(manifest["path"]) == set(metadata["path"])
    assert manifest["split"].value_counts().to_dict() == {
        "train": 280,
        "val": 40,
        "calibration": 40,
        "test": 40,
    }
    expected_per_class = {"train": 70, "val": 10, "calibration": 10, "test": 10}
    for split, count in expected_per_class.items():
        assert manifest.loc[manifest["split"] == split, "label"].value_counts().to_dict() == {
            0: count,
            1: count,
            2: count,
            3: count,
        }


def test_five_fold_manifest_has_disjoint_roles_and_rotating_test_sets():
    metadata = _balanced_metadata()

    manifest = cross_validation_manifest(metadata, folds=5, seed=59)

    assert set(manifest["fold"]) == set(range(5))
    assert len(manifest) == 5 * len(metadata)
    expected_sizes = {"train": 220, "val": 60, "calibration": 40, "test": 80}
    for fold, fold_frame in manifest.groupby("fold"):
        assert fold_frame["path"].is_unique, fold
        assert set(fold_frame["path"]) == set(metadata["path"])
        assert fold_frame["split"].value_counts().to_dict() == expected_sizes
        role_paths = {
            role: set(fold_frame.loc[fold_frame["split"] == role, "path"])
            for role in expected_sizes
        }
        for left, right in combinations(role_paths.values(), 2):
            assert left.isdisjoint(right)
        for role, count in {
            "train": 55,
            "val": 15,
            "calibration": 10,
            "test": 20,
        }.items():
            assert fold_frame.loc[
                fold_frame["split"] == role, "label"
            ].value_counts().to_dict() == {0: count, 1: count, 2: count, 3: count}

    # Every sample serves as outer-fold test data exactly once.
    test_appearances = manifest.loc[manifest["split"] == "test", "path"].value_counts()
    assert test_appearances.eq(1).all()


def test_load_manifest_validates_schema_split_names_and_integer_labels(tmp_path):
    cases = [
        (
            pd.DataFrame({"path": ["a.png"], "label": [0]}),
            "missing columns",
        ),
        (
            pd.DataFrame(columns=["path", "label", "split"]),
            "manifest is empty",
        ),
        (
            pd.DataFrame({"path": ["a.png"], "label": [0], "split": ["dev"]}),
            "unknown split names",
        ),
        (
            pd.DataFrame({"path": ["a.png"], "label": [-1], "split": ["train"]}),
            "non-negative integers",
        ),
        (
            pd.DataFrame({"path": ["a.png"], "label": [1.5], "split": ["train"]}),
            "labels must be integers",
        ),
        (
            pd.DataFrame({"path": ["a.png"], "label": ["not-a-label"], "split": ["train"]}),
            "non-negative integers",
        ),
    ]

    for index, (frame, message) in enumerate(cases):
        path = _write_manifest(tmp_path, frame, name=f"invalid-{index}.csv")
        with pytest.raises(ValueError, match=message):
            load_manifest(path)


def test_cross_validation_manifest_requires_and_filters_explicit_fold(tmp_path):
    frame = pd.DataFrame(
        {
            "path": ["a.png", "b.png", "a.png", "b.png"],
            "label": [0, 1, 0, 1],
            "split": ["train", "test", "test", "train"],
            "fold": [0, 0, 1, 1],
        }
    )
    path = _write_manifest(tmp_path, frame)

    with pytest.raises(ValueError, match="requires an explicit fold"):
        load_manifest(path)
    selected = load_manifest(path, fold=1)
    assert selected[["path", "split", "fold"]].to_dict("records") == [
        {"path": "a.png", "split": "test", "fold": 1},
        {"path": "b.png", "split": "train", "fold": 1},
    ]
    with pytest.raises(ValueError, match="fold 9 has no rows"):
        load_manifest(path, fold=9)


def test_calibration_fit_assertion_accepts_only_calibration_rows():
    assert_fit_split(pd.DataFrame({"split": ["calibration", "calibration"]}))

    for splits in (["val"], ["calibration", "test"], []):
        with pytest.raises(ValueError, match="expected only split='calibration'"):
            assert_fit_split(pd.DataFrame({"split": splits}))


def test_dataset_rejects_training_corruption_before_image_decode():
    frame = pd.DataFrame(
        {
            "path": ["does-not-need-to-exist.png"],
            "label": [0],
            "split": ["train"],
        }
    )

    with pytest.raises(ValueError, match="evaluation-only"):
        ManifestImageDataset(frame, split="train", corruption=lambda image, index: image)


def test_dataset_decodes_rgb_and_applies_corruption_before_transform(tmp_path):
    image_path = tmp_path / "gray.png"
    Image.fromarray(np.full((5, 7), 100, dtype=np.uint8)).save(image_path)
    frame = pd.DataFrame(
        {
            "path": [image_path.name],
            "label": [2],
            "split": ["test"],
            "sample_id": ["sample-1"],
        }
    )
    seen: dict[str, object] = {}

    def corruption(image, index):
        seen["corruption_input"] = (image.mode, index)
        return np.full((5, 7, 3), 23, dtype=np.uint8)

    def transform(image):
        seen["transform_input"] = (image.mode, int(np.asarray(image).mean()))
        return np.asarray(image)

    sample = ManifestImageDataset(
        frame,
        split="test",
        data_root=tmp_path,
        corruption=corruption,
        transform=transform,
    )[0]

    assert seen == {"corruption_input": ("RGB", 0), "transform_input": ("RGB", 23)}
    assert sample["image"].shape == (5, 7, 3)
    assert sample | {"image": None} == {
        "image": None,
        "label": 2,
        "path": "gray.png",
        "split": "test",
        "sample_id": "sample-1",
    }


def test_gardner_binary_quality_mapping_and_normalization():
    grades = pd.DataFrame(
        {
            "expansion": ["3", 6, 2, 3, 4, 5],
            "icm": [" a ", "B", "A", "C", "b", "A"],
            "te": ["b", "A", "A", "A", " B ", "C"],
        }
    )

    labels = kromp_high_quality_label(
        grades,
        expansion_col="expansion",
        icm_col="icm",
        te_col="te",
    )

    assert labels.dtype.kind in "iu"
    assert labels.tolist() == [1, 1, 0, 0, 1, 0]


@pytest.mark.parametrize(
    "bad_row",
    [
        {"expansion": None, "icm": "A", "te": "A"},
        {"expansion": 3, "icm": "D", "te": "A"},
        {"expansion": 3, "icm": "A", "te": "unknown"},
    ],
)
def test_gardner_mapping_rejects_missing_or_unknown_grades(bad_row):
    with pytest.raises(ValueError, match="invalid or missing Gardner grades in 1 rows"):
        kromp_high_quality_label(
            pd.DataFrame([bad_row]),
            expansion_col="expansion",
            icm_col="icm",
            te_col="te",
        )


def test_image_discovery_and_audit_are_content_based(tmp_path):
    disguised_png = tmp_path / "normal" / "sample.bmp"
    disguised_png.parent.mkdir()
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(disguised_png, format="PNG")
    duplicate = tmp_path / "normal" / "duplicate.png"
    duplicate.write_bytes(disguised_png.read_bytes())
    grayscale = tmp_path / "other" / "gray.png"
    grayscale.parent.mkdir()
    Image.new("L", (5, 4), color=80).save(grayscale)
    corrupt = tmp_path / "other" / "corrupt.jpg"
    corrupt.write_bytes(b"not an image")
    (tmp_path / "ignored.txt").write_text("not an image", encoding="utf-8")

    with pytest.raises(ValueError, match="1 unreadable image candidates"):
        discover_decodable_images(tmp_path)
    discovered = discover_decodable_images(tmp_path, strict=False)
    assert discovered == sorted([disguised_png, duplicate, grayscale])

    metadata = pd.DataFrame(
        {
            "path": [
                str(disguised_png.relative_to(tmp_path)),
                str(duplicate.relative_to(tmp_path)),
                str(grayscale.relative_to(tmp_path)),
                str(corrupt.relative_to(tmp_path)),
            ],
            "label": [0, 0, 1, 1],
            "patient_id": ["p1", "p1", "p2", "p3"],
        }
    )
    report = audit_metadata(metadata, data_root=tmp_path)

    assert report["images_listed"] == 4
    assert report["classes"] == {"0": 2, "1": 2}
    assert report["dimensions"] == {"8x6": 2, "5x4": 1}
    assert report["color_modes"] == {"RGB": 2, "L": 1}
    assert report["corrupt_files"] == ["other/corrupt.jpg"]
    assert report["duplicate_groups"] == [["normal/sample.bmp", "normal/duplicate.png"]]
    assert report["patients"] == 3
    assert report["images_per_patient"] == {"min": 1, "median": 1.0, "max": 2}


def test_class_inference_normalizes_release_directory_names():
    smids = {"non sperm": 2, "nonsperm": 2, "abnormal sperm": 1, "normal sperm": 0}

    assert _infer_class(Path("01 - Normal_Sperm/example.bmp"), smids) == 0
    assert _infer_class(Path("2-Abnormal-Sperm/sample.png"), smids) == 1
    assert _infer_class(Path("NonSperm Images/cell.jpg"), smids) == 2
    with pytest.raises(ValueError, match="could not infer exactly one class"):
        _infer_class(Path("unknown/sample.png"), smids)


def test_evaluation_transform_is_deterministic_and_train_transform_is_randomized():
    image = Image.fromarray(
        np.arange(45 * 60 * 3, dtype=np.uint16).reshape(45, 60, 3).astype(np.uint8)
    )
    evaluation = build_transform(train=False, image_size=32)

    first = evaluation(image)
    second = evaluation(image)

    assert first.shape == (3, 32, 32)
    assert torch.equal(first, second)
    train_names = [
        type(step).__name__ for step in build_transform(train=True, image_size=32).transforms
    ]
    assert train_names[:5] == [
        "RandomResizedCrop",
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomRotation",
        "ColorJitter",
    ]
    with pytest.raises(ValueError, match="image_size must be positive"):
        build_transform(train=False, image_size=0)
