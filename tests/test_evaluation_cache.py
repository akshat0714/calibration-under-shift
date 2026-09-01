from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src import evaluate as evaluate_module
from src.evaluate import CheckpointEvaluator, _atomic_savez
from src.models.build import build_model


def _checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index in range(4):
        array = np.full((32, 32, 3), 32 + index * 40, dtype=np.uint8)
        path = image_root / f"sample-{index}.png"
        Image.fromarray(array).save(path)
        rows.append({"path": path.name, "label": index % 2, "split": "test"})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    config = {
        "dataset": {
            "name": "cache_test",
            "root": str(image_root),
            "manifest": str(manifest),
            "num_classes": 2,
        },
        "model": {
            "backbone": "tiny_cnn",
            "pretrained": False,
            "dropout": 0.5,
            "input_size": 32,
        },
        "training": {"seed": 17, "batch_size": 4},
        "evaluation": {"batch_size": 4, "num_workers": 0},
        "outputs": {"cache_dir": str(tmp_path / "cache")},
    }
    model = build_model("tiny_cnn", num_classes=2, pretrained=False, dropout=0.5)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "config": config,
            "model_state": model.state_dict(),
            "seed": 17,
            "run_id": "cache-test-run",
        },
        checkpoint,
    )
    return checkpoint, config


def test_atomic_savez_preserves_published_entry_on_write_failure(tmp_path, monkeypatch):
    target = tmp_path / "entry.npz"
    target.write_bytes(b"published")

    def interrupted(handle, **_arrays):
        handle.write(b"partial")
        raise RuntimeError("preempted")

    monkeypatch.setattr(evaluate_module.np, "savez_compressed", interrupted)
    with pytest.raises(RuntimeError, match="preempted"):
        _atomic_savez(target, values=np.arange(3))

    assert target.read_bytes() == b"published"
    assert list(tmp_path.glob(".entry.npz.*.tmp")) == []


def test_mc_dropout_reuses_feature_cache_and_caches_stochastic_logits(tmp_path, monkeypatch):
    checkpoint, config = _checkpoint(tmp_path)
    evaluator = CheckpointEvaluator(
        checkpoint,
        config=config,
        device="cpu",
        cache_dir=tmp_path / "cache",
    )
    inputs = next(iter(evaluator.loader("test", "clean", 0)))["image"]
    evaluator.model.eval()
    evaluator.model.head[0].train()
    with torch.random.fork_rng():
        torch.manual_seed(evaluator.seed)
        full_forward_logits = np.stack([evaluator.model(inputs).detach().numpy() for _ in range(3)])
    evaluator.model.eval()
    deterministic = evaluator.infer("test", "clean", 0)

    def backbone_must_not_run(*_args, **_kwargs):
        raise AssertionError("MC dropout recomputed deterministic backbone features")

    monkeypatch.setattr(evaluator.model, "forward_features", backbone_must_not_run)
    logits, labels, paths = evaluator.infer_mc_dropout("test", "clean", 0, passes=3)

    assert logits.shape == (3, len(deterministic.labels), 2)
    assert np.allclose(logits, full_forward_logits)
    assert np.array_equal(labels, deterministic.labels)
    assert np.array_equal(paths, deterministic.paths)
    mc_path = evaluator._mc_cache_path("test", "clean", 0, 3)
    assert mc_path.is_file()

    def head_must_not_run(*_args, **_kwargs):
        raise AssertionError("MC-dropout cache hit recomputed the head")

    monkeypatch.setattr(evaluator.model.head, "forward", head_must_not_run)
    cached_logits, cached_labels, cached_paths = evaluator.infer_mc_dropout(
        "test", "clean", 0, passes=3
    )
    assert np.array_equal(cached_logits, logits)
    assert np.array_equal(cached_labels, labels)
    assert np.array_equal(cached_paths, paths)
    assert evaluator._mc_cache_path("test", "clean", 0, 4) != mc_path
