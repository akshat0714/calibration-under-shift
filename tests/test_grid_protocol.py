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
from src.shifts.severity import CORRUPTION_PARAMETERS


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


def test_evaluator_preserves_checkpoint_fold_and_shift_protocol(tmp_path, monkeypatch):
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
    original_cache_path = evaluator._cache_path("test", "motion_blur", 1)
    monkeypatch.setitem(CORRUPTION_PARAMETERS, "motion_blur", (3, 5, 7, 9, 11))
    changed_protocol = CheckpointEvaluator(checkpoint, config=base_config, device="cpu")
    assert changed_protocol._cache_path("test", "motion_blur", 1) != original_cache_path
    incompatible = copy.deepcopy(base_config)
    incompatible["model"]["mean"] = [0.4, 0.5, 0.5]
    with pytest.raises(ValueError, match="model.mean"):
        CheckpointEvaluator(checkpoint, config=incompatible, device="cpu")


def _fake(
    fold: int,
    seed: int,
    digest: str = "manifest",
    *,
    dataset: str = "folded",
    model: str = "tiny_cnn",
):
    return SimpleNamespace(
        seed=seed,
        manifest_digest=digest,
        model=SimpleNamespace(backbone_name=model),
        config={
            "dataset": {"name": dataset, "num_classes": 2, "fold": fold},
            "model": {"backbone": model, "input_size": 32},
        },
    )


def test_ensemble_groups_only_exact_five_member_smids_resnet50():
    evaluators = [
        *(_fake(0, seed, dataset="smids", model="resnet50") for seed in range(11, 16)),
        *(_fake(0, seed, dataset="smids", model="xception") for seed in range(11, 14)),
        *(_fake(1, seed, dataset="smids", model="resnet50") for seed in range(11, 15)),
    ]
    groups = {frozenset(group) for group in ensemble_group_indices(evaluators)}
    assert groups == {frozenset(range(5))}

    with pytest.raises(ValueError, match="duplicate seeds"):
        ensemble_group_indices([_fake(0, 11), _fake(0, 11)])


def test_checkpoint_matrix_requires_configured_seed_fold_product():
    config = {
        "dataset": {"folds": [0, 1]},
        "training": {"seed": 11, "seeds": [11, 13]},
    }
    complete = [_fake(fold, seed) for seed in (11, 13) for fold in (0, 1)]
    validate_checkpoint_matrix(complete, config)
    with pytest.raises(ValueError, match="Missing"):
        validate_checkpoint_matrix(complete[:-1], config)
