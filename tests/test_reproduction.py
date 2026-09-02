from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import experiments.full_retrain as retrain
import experiments.reproduce_release as reproduce
import experiments.verify_reproduction as verify


def _metric_rows(revision: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": "resnet50",
                "seed": "2025",
                "fold": "" if dataset == "smids" else "0",
                "corruption": "clean",
                "severity": 0,
                "n_samples": 10,
                "run_id": f"{dataset}-run",
                "checkpoint": f"results/checkpoints/{dataset}.pt",
                "manifest_sha256": "a" * 64,
                "corruption_protocol_sha256": "b" * 64,
                "evaluation_git_revision": revision,
                "method": "raw_softmax",
                "metric": "accuracy",
                "value": value,
            }
            for dataset, value in (("smids", 0.9), ("hushem", 0.8))
        ]
    )


def test_safe_reproduction_reset_is_exact_and_removes_only_isolated_tree(tmp_path):
    repository = tmp_path / "repository"
    destination = repository / "results/reproduction"
    destination.mkdir(parents=True)
    (destination / "stale.txt").write_text("stale\n", encoding="utf-8")
    sibling = repository / "results/reference.txt"
    sibling.write_text("keep\n", encoding="utf-8")

    observed = reproduce._safe_reset_reproduction(repository)

    assert observed == destination.resolve()
    assert observed.is_dir() and not (observed / "stale.txt").exists()
    assert sibling.read_text(encoding="utf-8") == "keep\n"


def test_release_reproduction_refuses_silent_cpu(monkeypatch):
    monkeypatch.setattr(reproduce, "resolve_device", lambda requested: torch.device("cpu"))

    with pytest.raises(reproduce.ReleaseReproductionError, match="CPU execution is refused"):
        reproduce._resolved_accelerator("auto", allow_cpu=False)

    assert reproduce._resolved_accelerator("auto", allow_cpu=True) == "cpu"


def test_full_retrain_refuses_non_cuda_hosts(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        retrain.full_retrain(root=tmp_path, download=False)


def test_metric_comparison_allows_only_revision_change_within_tolerance(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "EXPECTED_TOTAL_ROWS", 2)
    reference = tmp_path / "reference.csv"
    reproduced = tmp_path / "reproduced.csv"
    _metric_rows("a" * 40).to_csv(reference, index=False)
    _metric_rows("b" * 40).to_csv(reproduced, index=False)

    summary = verify.verify_metrics(reference, reproduced)

    assert summary["rows"] == 2
    assert summary["max_abs_error"] == 0.0
    changed = pd.read_csv(reproduced, dtype={"fold": "string"})
    changed.loc[0, "value"] += 1e-3
    changed.to_csv(reproduced, index=False)
    with pytest.raises(verify.ReproductionError, match="numerical tolerance"):
        verify.verify_metrics(reference, reproduced)


def test_threshold_verifier_preserves_approved_null(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    reference = repository / "results/thresholds.csv"
    reproduced = tmp_path / "thresholds.csv"
    frame = pd.read_csv(reference)
    frame.to_csv(reproduced, index=False)

    assert verify.verify_thresholds(reference, reproduced) == {
        "rows": 16,
        "paired_crossings": 10,
        "early_crossings": 0,
        "signal_never": 6,
    }

    paired = frame["signal_crossing_severity"].notna() & frame["accuracy_drop_severity"].notna()
    frame.loc[paired.idxmax(), "signal_is_earlier"] = True
    frame.to_csv(reproduced, index=False)
    with pytest.raises(AssertionError):
        verify.verify_thresholds(reference, reproduced)


def test_figure_manifest_rejects_tampered_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "EXPECTED_FIGURES", {"f1"})
    artifact = tmp_path / "results/figures/f1.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"figure")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "figures": {
                    "f1": {
                        "path": "results/figures/f1.png",
                        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    assert verify.verify_figure_manifest(tmp_path, manifest_path)["figures"] == 1
    artifact.write_bytes(b"tampered")
    with pytest.raises(verify.ReproductionError, match="SHA-256 mismatch"):
        verify.verify_figure_manifest(tmp_path, manifest_path)


def test_source_guardrail_grep_matches_current_implementation():
    repository = Path(__file__).resolve().parents[1]

    assert all(verify.verify_source_guardrails(repository).values())


def test_demo_attribution_panels_use_evaluated_severities():
    repository = Path(__file__).resolve().parents[1]
    script = (repository / "run.sh").read_text(encoding="utf-8")
    demo_block = script.split("  --demo)", maxsplit=1)[1].split("  --corruption-grid)", maxsplit=1)[
        0
    ]
    attribution_call = demo_block.split('"$python_bin" -m experiments.run_attribution', maxsplit=1)[
        1
    ]

    assert "--severities 0 1 3" in attribution_call
    assert "--figure-severities 0 1 3" in attribution_call
