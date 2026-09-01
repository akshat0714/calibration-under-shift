from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.run_grid import (
    _detail_filename,
    _provenance_path,
    _write_detail_json,
    evaluate_checkpoint,
    evaluate_deep_ensemble,
)
from src.evaluate import PredictionBundle


def _bundle(logits: list[list[float]], labels: list[int], prefix: str) -> PredictionBundle:
    values = np.asarray(logits, dtype=np.float32)
    return PredictionBundle(
        logits=values,
        labels=np.asarray(labels, dtype=np.int64),
        features=np.column_stack(
            [np.linspace(-1.0, 1.0, len(labels)), np.linspace(1.0, -1.0, len(labels))]
        ).astype(np.float32),
        paths=np.asarray([f"{prefix}-{index}.png" for index in range(len(labels))]),
    )


class _FakeEvaluator:
    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint_path = checkpoint
        self.checkpoint_sha256 = "a" * 64
        self.run_id = "toy-run"
        self.seed = 23
        self.device = "cpu"
        self.manifest_digest = "b" * 64
        self.corruption_protocol_sha256 = "c" * 64
        self.model = SimpleNamespace(backbone_name="toy_model")
        self.config = {
            "dataset": {"name": "toy", "num_classes": 2},
            "model": {"backbone": "toy_model", "input_size": 8},
            "training": {"seed": 23, "batch_size": 4},
            "evaluation": {
                "corruptions": ["jpeg"],
                "severities": [1],
                "calibration_bins": 3,
                "conformal_alpha": 0.1,
                "risk_coverage_target": 0.8,
                "mc_dropout_passes": 2,
            },
        }
        self.calibration = _bundle(
            [[2.0, 0.0], [0.0, 2.0], [1.2, 0.1], [0.2, 1.1], [0.8, 0.2], [0.1, 0.9]],
            [0, 1, 0, 1, 0, 1],
            "calibration",
        )
        self.train = _bundle(
            [
                [2.0, 0.0],
                [0.0, 2.0],
                [1.5, 0.1],
                [0.2, 1.6],
                [1.0, 0.2],
                [0.1, 1.1],
                [0.9, 0.3],
                [0.3, 0.8],
            ],
            [0, 1, 0, 1, 0, 1, 0, 1],
            "train",
        )
        self.test = _bundle(
            [[2.0, 0.0], [0.0, 2.0], [0.4, 0.7], [0.8, 0.3]],
            [0, 1, 0, 1],
            "test",
        )

    def infer(self, split: str, _corruption: str, _severity: int) -> PredictionBundle:
        return {"calibration": self.calibration, "train": self.train, "test": self.test}[split]

    def infer_mc_dropout(self, _split: str, _corruption: str, _severity: int, passes: int):
        assert passes == 2
        return (
            np.stack([self.test.logits, self.test.logits + np.asarray([0.1, -0.1])]),
            self.test.labels,
            self.test.paths,
        )


def test_checkpoint_details_include_provenance_reliability_and_risk_curves(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    rows, bundles, calibration, details = evaluate_checkpoint(_FakeEvaluator(checkpoint))

    assert set(bundles) == {("clean", 0), ("jpeg", 1)}
    assert len(calibration.labels) == 6
    assert {row["run_id"] for row in rows} == {"toy-run"}
    assert details["checkpoint"]["sha256"] == "a" * 64
    assert details["calibration_fit"]["temperature"]["split"] == "calibration"
    assert details["calibration_fit"]["mahalanobis"]["split"] == "train"
    assert len(details["conditions"]) == 2
    clean = details["conditions"][0]
    assert len(clean["methods"]["raw_softmax"]["reliability_bins"]) == 3
    assert len(clean["methods"]["raw_softmax"]["risk_coverage"]["coverage"]) == 5
    assert "risk_coverage" in clean["methods"]["energy"]
    assert "reliability_bins" in clean["methods"]["mc_dropout"]

    output = tmp_path / "details" / "toy-run.json"
    _write_detail_json(details, output)
    encoded = output.read_text(encoding="utf-8")
    assert "NaN" not in encoded
    assert json.loads(encoded)["checkpoint"]["run_id"] == "toy-run"


def test_checkpoint_provenance_paths_are_portable(tmp_path):
    repository = tmp_path / "repository"
    checkpoint = repository / "results" / "checkpoints" / "member.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    assert _provenance_path(checkpoint, repository) == "results/checkpoints/member.pt"
    assert _provenance_path(tmp_path / "elsewhere" / "member.pt", repository) == (
        "external/member.pt"
    )
    assert not Path(_provenance_path(checkpoint, repository)).is_absolute()


def test_detail_filename_sanitizes_checkpoint_provided_run_id():
    assert _detail_filename("safe-run_1") == "safe-run_1.json"
    unsafe = _detail_filename("../../outside")
    assert unsafe.startswith("outside-")
    assert unsafe.endswith(".json")
    assert "/" not in unsafe


def test_deep_ensemble_details_cover_only_the_five_member_smids_resnet_group(tmp_path):
    evaluators = []
    condition_bundles = []
    calibration_bundles = []
    for offset in range(5):
        evaluator = _FakeEvaluator(tmp_path / f"member-{offset}.pt")
        evaluator.seed = 2025 + offset
        evaluator.run_id = f"member-{offset}"
        evaluator.checkpoint_sha256 = str(offset) * 64
        evaluator.model = SimpleNamespace(backbone_name="resnet50")
        evaluator.config["dataset"].update({"name": "smids", "num_classes": 2})
        evaluator.config["model"].update({"backbone": "resnet50", "input_size": 8})
        evaluators.append(evaluator)
        condition_bundles.append({("clean", 0): evaluator.test, ("jpeg", 1): evaluator.test})
        calibration_bundles.append(evaluator.calibration)

    rows, details = evaluate_deep_ensemble(
        evaluators,
        condition_bundles,
        calibration_bundles,
    )

    assert details["member_count"] == 5
    assert len(details["members"]) == 5
    assert len(details["conditions"]) == 2
    assert {row["method"] for row in rows} == {"deep_ensemble", "ensemble_aps"}
    clean = details["conditions"][0]["methods"]["deep_ensemble"]
    assert "reliability_bins" in clean
    assert "risk_coverage" in clean
