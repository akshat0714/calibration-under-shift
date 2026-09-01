from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from experiments.run_grid import ensemble_group_indices, validate_checkpoint_matrix
from src.evaluate import CheckpointEvaluator
from src.models.build import build_model


def _config(tmp_path: Path, fold: int) -> dict:
    return {
        "dataset": {
            "name": "folded",
            "root": str(tmp_path),
            "manifest": str(tmp_path / "manifest.csv"),
            "num_classes": 2,
            "fold": fold,
            "folds": [0, 1],
        },
        "model": {
            "backbone": "tiny_cnn",
            "pretrained": False,
            "dropout": 0.1,
            "input_size": 32,
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "interpolation": "bilinear",
            "crop_pct": 0.875,
        },
        "training": {"seed": 11, "seeds": [11, 13], "batch_size": 2},
        "evaluation": {"batch_size": 2, "num_workers": 0, "corruption_seed": 7},
        "outputs": {"cache_dir": str(tmp_path / "cache")},
    }


def test_evaluator_preserves_checkpoint_fold_against_base_yaml(tmp_path):
    pd.DataFrame(
        {
            "path": ["fold-0.png", "fold-1.png"],
            "label": [0, 1],
            "split": ["test", "test"],
            "fold": [0, 1],
        }
    ).to_csv(tmp_path / "manifest.csv", index=False)
    checkpoint_config = _config(tmp_path, fold=1)
    model = build_model("tiny_cnn", num_classes=2, pretrained=False, dropout=0.1)
    checkpoint = tmp_path / "fold-1.pt"
    torch.save(
        {
            "config": checkpoint_config,
            "model_state": model.state_dict(),
            "seed": 11,
        },
        checkpoint,
    )
    base_config = _config(tmp_path, fold=0)

    evaluator = CheckpointEvaluator(checkpoint, config=base_config, device="cpu")

    assert evaluator.config["dataset"]["fold"] == 1
    assert set(evaluator.manifest["fold"]) == {1}
    incompatible = copy.deepcopy(base_config)
    incompatible["model"]["mean"] = [0.4, 0.5, 0.5]
    with pytest.raises(ValueError, match="model.mean"):
        CheckpointEvaluator(checkpoint, config=incompatible, device="cpu")


def _fake(fold: int, seed: int, digest: str = "manifest"):
    return SimpleNamespace(
        seed=seed,
        manifest_digest=digest,
        model=SimpleNamespace(backbone_name="tiny_cnn"),
        config={
            "dataset": {"name": "folded", "num_classes": 2, "fold": fold},
            "model": {"backbone": "tiny_cnn", "input_size": 32},
        },
    )


def test_ensemble_groups_same_fold_distinct_seeds_only():
    evaluators = [_fake(0, 11), _fake(1, 11), _fake(0, 13)]
    groups = {frozenset(group) for group in ensemble_group_indices(evaluators)}
    assert groups == {frozenset({0, 2}), frozenset({1})}

    with pytest.raises(ValueError, match="duplicate seeds"):
        ensemble_group_indices([_fake(0, 11), _fake(0, 11)])


def test_checkpoint_matrix_requires_configured_seed_fold_product():
    config = {
        "dataset": {"folds": [0, 1]},
        "training": {"seed": 11, "seeds": [11, 13]},
    }
    complete = [_fake(fold, seed) for seed in (11, 13) for fold in (0, 1)]
    validate_checkpoint_matrix(complete, config)
    with pytest.raises(ValueError, match="missing"):
        validate_checkpoint_matrix(complete[:-1], config)
