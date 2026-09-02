from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments import evaluate_clean_matrix as clean_module
from experiments.evaluate_clean_matrix import (
    clean_acceptance_failures,
    evaluate_clean_matrix,
    summarize_clean_metrics,
    validate_checkpoint_registry,
)
from src.evaluate import PredictionBundle

TEST_MEMBERS = tuple(
    sorted(
        [
            *(("smids", "resnet50", seed, None) for seed in range(2025, 2030)),
            *(("smids", "xception", seed, None) for seed in range(2025, 2028)),
            *(("smids", "mobilenet_v3_large", seed, None) for seed in range(2025, 2028)),
            *(("hushem", "resnet50", 2025, fold) for fold in range(5)),
        ]
    )
)


def _registry() -> pd.DataFrame:
    rows = []
    for index, (dataset, model, seed, fold) in enumerate(TEST_MEMBERS):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "fold": np.nan if fold is None else fold,
                "run_id": f"run-{index}",
                "checkpoint": f"checkpoint-{index}.pt",
                "best_val_macro_f1": 0.8,
            }
        )
    return pd.DataFrame(rows)


def test_registry_requires_exact_prespecified_members():
    frame = _registry()
    validated = validate_checkpoint_registry(frame)
    assert len(validated) == 16
    assert {
        (member.dataset, member.model, member.seed, member.fold) for member in validated["_member"]
    } == set(TEST_MEMBERS)

    with pytest.raises(ValueError, match="missing"):
        validate_checkpoint_registry(frame.iloc[:-1])
    with pytest.raises(ValueError, match="duplicates"):
        validate_checkpoint_registry(pd.concat([frame, frame.iloc[[0]]], ignore_index=True))

    extra = frame.copy()
    extra.loc[0, "seed"] = 9999
    with pytest.raises(ValueError, match="extra"):
        validate_checkpoint_registry(extra)


def test_registry_rejects_duplicate_checkpoint_and_noninteger_identity():
    duplicate = _registry()
    duplicate.loc[1, "checkpoint"] = duplicate.loc[0, "checkpoint"]
    with pytest.raises(ValueError, match="checkpoint paths must be unique"):
        validate_checkpoint_registry(duplicate)

    duplicate = _registry()
    duplicate.loc[1, "run_id"] = duplicate.loc[0, "run_id"]
    with pytest.raises(ValueError, match="run IDs must be unique"):
        validate_checkpoint_registry(duplicate)

    invalid = _registry()
    invalid["seed"] = invalid["seed"].astype(float)
    invalid.loc[0, "seed"] = 2025.5
    with pytest.raises(ValueError, match="seed must be an integer"):
        validate_checkpoint_registry(invalid)


def test_summary_uses_sample_standard_deviation():
    frame = pd.DataFrame(
        {
            "dataset": ["smids", "smids"],
            "model": ["resnet50", "resnet50"],
            "metric": ["accuracy", "accuracy"],
            "value": [0.8, 1.0],
        }
    )
    row = summarize_clean_metrics(frame).iloc[0]
    assert row["mean"] == pytest.approx(0.9)
    assert row["std"] == pytest.approx(np.std([0.8, 1.0], ddof=1))
    assert row["n"] == 2


def test_clean_acceptance_criteria_use_prespecified_thresholds():
    registry = validate_checkpoint_registry(_registry())
    accuracy_rows = [
        {
            "dataset": member.dataset,
            "model": member.model,
            "seed": member.seed,
            "fold": "" if member.fold is None else member.fold,
            "metric": "accuracy",
            "value": 0.9,
        }
        for member in registry["_member"]
    ]
    summary = pd.DataFrame(
        [
            {"dataset": "smids", "model": model, "metric": "macro_f1", "mean": 0.9}
            for model in ("resnet50", "xception", "mobilenet_v3_large")
        ]
        + [
            {
                "dataset": "hushem",
                "model": "resnet50",
                "metric": "accuracy",
                "mean": 0.9,
            }
        ]
    )
    metrics = pd.DataFrame(accuracy_rows)
    assert clean_acceptance_failures(metrics, summary) == []

    metrics.loc[0, "value"] = 0.25
    summary.loc[summary["model"] == "xception", "mean"] = 0.84
    summary.loc[summary["dataset"] == "hushem", "mean"] = 0.79
    failures = clean_acceptance_failures(metrics, summary)
    assert any("at or below chance" in failure for failure in failures)
    assert any("smids/xception" in failure for failure in failures)
    assert any("hushem/resnet50" in failure for failure in failures)


def test_require_cuda_rejects_cpu(monkeypatch):
    monkeypatch.setattr(
        clean_module, "resolve_device", lambda _requested: clean_module.torch.device("cpu")
    )
    monkeypatch.setattr(clean_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA execution was required"):
        clean_module._require_requested_cuda("auto", True)


def test_evaluator_runs_clean_test_only_and_writes_tidy_outputs(tmp_path, monkeypatch):
    registry = _registry()
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)
    for checkpoint in registry["checkpoint"]:
        (tmp_path / checkpoint).write_bytes(b"placeholder")
    registry["checkpoint"] = [str(tmp_path / value) for value in registry["checkpoint"]]
    registry.to_csv(registry_path, index=False)

    calls = []
    loaded_configs = []

    class FakeEvaluator:
        def __init__(self, checkpoint_path, config, device, cache_dir):
            index = int(Path(checkpoint_path).stem.split("-")[-1])
            dataset, model, seed, fold = TEST_MEMBERS[index]
            self.config = {
                "dataset": {"name": dataset, "fold": fold},
            }
            self.model = SimpleNamespace(backbone_name=model)
            self.seed = seed
            self.run_id = f"run-{index}"
            self.manifest_digest = "manifest"
            self.corruption_protocol_sha256 = "protocol"

        def infer(self, split, corruption, severity):
            calls.append((split, corruption, severity))
            return PredictionBundle(
                logits=np.asarray([[4.0, 0.0], [0.0, 4.0]]),
                labels=np.asarray([0, 1]),
                features=np.zeros((2, 1)),
                paths=np.asarray(["a.png", "b.png"]),
            )

    monkeypatch.setattr(clean_module, "CheckpointEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        clean_module,
        "load_config",
        lambda path: loaded_configs.append(path) or {},
    )
    monkeypatch.setattr(
        clean_module, "resolve_device", lambda _requested: clean_module.torch.device("cpu")
    )
    output = tmp_path / "metrics.csv"
    summary_output = tmp_path / "summary.csv"
    tidy, summary = evaluate_clean_matrix(
        registry_path,
        output,
        summary_output,
        device="cpu",
        cache_dir=tmp_path / "cache",
    )

    assert calls == [("test", "clean", 0)] * 16
    assert len(loaded_configs) == 4
    assert {"accuracy", "macro_f1"} <= set(tidy["metric"])
    assert set(tidy["split"]) == {"test"}
    assert set(tidy["corruption"]) == {"clean"}
    assert set(tidy["severity"]) == {0}
    assert len(tidy.loc[tidy["metric"] == "accuracy"]) == 16
    replicate_counts = summary.loc[summary["metric"] == "macro_f1"].set_index(["dataset", "model"])[
        "n"
    ]
    assert replicate_counts.to_dict() == {
        ("hushem", "resnet50"): 5,
        ("smids", "mobilenet_v3_large"): 3,
        ("smids", "resnet50"): 5,
        ("smids", "xception"): 3,
    }
    assert output.is_file() and summary_output.is_file()


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("identity", "does not match registry"),
        ("run_id", "does not match registry"),
    ],
)
def test_evaluator_rejects_checkpoint_provenance_mismatch(tmp_path, monkeypatch, mismatch, message):
    registry = _registry()
    registry["checkpoint"] = [str(tmp_path / checkpoint) for checkpoint in registry["checkpoint"]]
    for checkpoint in registry["checkpoint"]:
        Path(checkpoint).write_bytes(b"placeholder")
    registry_path = tmp_path / "registry.csv"
    registry.to_csv(registry_path, index=False)

    class MismatchedEvaluator:
        def __init__(self, checkpoint_path, config, device, cache_dir):
            index = int(Path(checkpoint_path).stem.split("-")[-1])
            dataset, model, seed, fold = TEST_MEMBERS[index]
            self.config = {"dataset": {"name": dataset, "fold": fold}}
            self.model = SimpleNamespace(backbone_name=model)
            self.seed = seed + (1 if mismatch == "identity" else 0)
            self.run_id = "wrong-run" if mismatch == "run_id" else f"run-{index}"

        def infer(self, *args, **kwargs):
            pytest.fail("inference must not run for mismatched checkpoint provenance")

    monkeypatch.setattr(clean_module, "CheckpointEvaluator", MismatchedEvaluator)
    monkeypatch.setattr(clean_module, "load_config", lambda _path: {})
    monkeypatch.setattr(
        clean_module, "resolve_device", lambda _requested: clean_module.torch.device("cpu")
    )
    with pytest.raises(ValueError, match=message):
        evaluate_clean_matrix(
            registry_path,
            tmp_path / "metrics.csv",
            tmp_path / "summary.csv",
            device="cpu",
        )
