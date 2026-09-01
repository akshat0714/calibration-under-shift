from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.splits import stratified_manifest
from src.evaluate import CheckpointEvaluator
from src.train import train


def _make_images(root: Path, samples_per_class: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    rows = []
    for label in range(3):
        directory = root / f"class_{label}"
        directory.mkdir(parents=True)
        for index in range(samples_per_class):
            array = rng.integers(0, 30, size=(32, 32, 3), dtype=np.uint8)
            array[6 + label * 3 : 18 + label * 3, 8:24, label] += 180
            path = directory / f"{index:03d}.png"
            Image.fromarray(array).save(path)
            rows.append({"path": str(path.relative_to(root)), "label": label})
    return pd.DataFrame(rows)


def test_train_checkpoint_and_corrupted_inference_end_to_end(tmp_path):
    data_root = tmp_path / "images"
    metadata = _make_images(data_root)
    manifest = stratified_manifest(metadata, seed=5)
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    config = {
        "dataset": {
            "name": "smoke",
            "root": str(data_root),
            "manifest": str(manifest_path),
            "num_classes": 3,
        },
        "model": {
            "backbone": "tiny_cnn",
            "pretrained": False,
            "dropout": 0.1,
            "input_size": 32,
        },
        "training": {
            "seed": 5,
            "deterministic": True,
            "device": "cpu",
            "batch_size": 8,
            "num_workers": 0,
            "head_epochs": 1,
            "finetune_epochs": 0,
            "head_lr": 0.01,
            "finetune_lr": 0.001,
            "weight_decay": 0.0,
            "patience": 1,
        },
        "evaluation": {"batch_size": 8, "num_workers": 0, "corruption_seed": 3},
        "outputs": {
            "runs_dir": str(tmp_path / "runs"),
            "checkpoints_dir": str(tmp_path / "checkpoints"),
            "cache_dir": str(tmp_path / "cache"),
        },
    }
    result = train(config)
    checkpoint = Path(result["checkpoint"])
    assert checkpoint.exists()
    evaluator = CheckpointEvaluator(
        checkpoint, config=config, device="cpu", cache_dir=tmp_path / "cache"
    )
    calibration = evaluator.infer("calibration")
    shifted_test = evaluator.infer("test", "defocus_blur", 2)
    assert calibration.logits.shape[1] == 3
    assert shifted_test.logits.shape == (len(shifted_test.labels), 3)
    assert np.allclose(shifted_test.probabilities.sum(axis=1), 1.0)

    tampered = pd.read_csv(manifest_path)
    tampered.loc[0, "label"] = (int(tampered.loc[0, "label"]) + 1) % 3
    tampered.to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="manifest differs"):
        CheckpointEvaluator(checkpoint, config=config, device="cpu")
