from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from experiments.generate_final_figures import (
    DEVICE_CORRUPTIONS,
    GROUPS,
    _assert_portable_manifest_paths,
    _manifest,
    build_attribution_accuracy_data,
    build_lockstep_data,
    extract_reliability_data,
    extract_risk_coverage_data,
    plot_attribution_accuracy,
    plot_lockstep,
)


def _protocol() -> dict:
    return {
        "thresholds": {
            "accuracy_absolute_drop": 0.05,
            "ece_relative_increase": 2.0,
            "ece_minimum_absolute_increase": 0.02,
            "entropy_relative_increase": 2.0,
            "entropy_minimum_absolute_increase": 0.05,
            "selective_risk_relative_increase": 2.0,
            "selective_risk_minimum_absolute_increase": 0.02,
            "conformal_target_coverage": 0.90,
            "conformal_allowed_shortfall": 0.05,
        }
    }


def _lockstep_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories = {
        ("raw_softmax", "accuracy"): [0.90, 0.85, 0.84, 0.82, 0.80, 0.78],
        ("raw_softmax", "ece"): [0.05, 0.07, 0.10, 0.11, 0.12, 0.13],
        ("raw_softmax", "mean_predictive_entropy"): [
            0.20,
            0.22,
            0.24,
            0.27,
            0.31,
            0.35,
        ],
        ("raw_softmax", "risk_at_80_coverage"): [0.05, 0.10, 0.11, 0.13, 0.15, 0.17],
        ("aps", "conformal_coverage"): [0.93, 0.91, 0.88, 0.85, 0.84, 0.82],
    }
    rows: list[dict[str, object]] = []
    for dataset, model in GROUPS:
        for seed in ("1", "2"):
            for (method, metric), values in trajectories.items():
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "fold": "",
                        "corruption": "clean",
                        "severity": 0,
                        "method": method,
                        "metric": metric,
                        "value": values[0],
                    }
                )
                for corruption in DEVICE_CORRUPTIONS:
                    for severity in range(1, 6):
                        rows.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "seed": seed,
                                "fold": "",
                                "corruption": corruption,
                                "severity": severity,
                                "method": method,
                                "metric": metric,
                                "value": values[severity],
                            }
                        )
    threshold_rows: list[dict[str, object]] = []
    crossings = {
        ("raw_softmax", "ece"): 3.0,
        ("raw_softmax", "mean_predictive_entropy"): np.nan,
        ("raw_softmax", "risk_at_80_coverage"): 2.0,
        ("aps", "conformal_coverage"): 4.0,
    }
    for dataset, model in GROUPS:
        for (method, metric), signal_crossing in crossings.items():
            threshold_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "signal_method": method,
                    "signal_metric": metric,
                    "signal_crossing_severity": signal_crossing,
                    "accuracy_drop_severity": 2.0,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows)


def test_lockstep_normalization_preserves_strict_frozen_crossings(tmp_path):
    metrics, thresholds = _lockstep_fixture()
    data = build_lockstep_data(metrics, thresholds, _protocol())
    group = data.loc[(data["dataset"] == "smids") & (data["model"] == "resnet50")]

    accuracy = group.loc[group["metric"] == "accuracy"].set_index("severity")
    assert accuracy.loc[0, "normalized_mean"] == pytest.approx(0.0)
    # Equality to the frozen cutoff normalizes to one but does not cross the
    # strict primary comparator; the first lower accuracy is severity two.
    assert accuracy.loc[1, "normalized_mean"] == pytest.approx(1.0)
    assert accuracy["crossing_severity"].iloc[0] == pytest.approx(2.0)

    ece = group.loc[group["metric"] == "ece"].set_index("severity")
    assert ece.loc[2, "normalized_mean"] == pytest.approx(1.0)
    assert ece["crossing_severity"].iloc[0] == pytest.approx(3.0)

    entropy = group.loc[group["metric"] == "mean_predictive_entropy"]
    assert entropy["crossing_severity"].isna().all()
    assert set(data["analysis_tier"]) == {"primary/prespecified"}

    output = plot_lockstep(data, tmp_path / "lockstep.png")
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.info["dpi"][0] == pytest.approx(150, abs=1)
        assert image.width > 800 and image.height > 500


def _detail_fixture() -> tuple[list[tuple[Path, dict]], pd.DataFrame]:
    identity = {
        "dataset": "smids",
        "model": "resnet50",
        "seed": "1",
        "fold": "",
    }
    rows: list[dict[str, object]] = []
    conditions: list[dict] = []
    values = {
        ("clean", 0): {
            "raw_softmax": {"ece": 0.10, "aurc": 0.12, "risk_at_80_coverage": 0.15},
            "temperature": {"ece": 0.08},
        },
        ("defocus_blur", 3): {
            "raw_softmax": {"ece": 0.20, "aurc": 0.22, "risk_at_80_coverage": 0.25},
            "temperature": {"ece": 0.18},
        },
    }
    for (corruption, severity), methods in values.items():
        for method, metrics in methods.items():
            for metric, value in metrics.items():
                rows.append(
                    {
                        **identity,
                        "corruption": corruption,
                        "severity": severity,
                        "method": method,
                        "metric": metric,
                        "value": value,
                    }
                )
        raw_metrics = methods["raw_softmax"]
        temp_metrics = methods["temperature"]
        bins = [
            {
                "bin": 0,
                "lower": 0.0,
                "upper": 0.5,
                "count": 2,
                "confidence": 0.4,
                "accuracy": 0.5,
                "gap": 0.1,
            },
            {
                "bin": 1,
                "lower": 0.5,
                "upper": 1.0,
                "count": 3,
                "confidence": 0.8,
                "accuracy": 2 / 3,
                "gap": 0.8 - 2 / 3,
            },
        ]
        conditions.append(
            {
                "corruption": corruption,
                "severity": severity,
                "n_samples": 5,
                "methods": {
                    "raw_softmax": {
                        "metrics": raw_metrics,
                        "reliability_bins": bins,
                        "risk_coverage": {
                            "coverage": [0.0, 0.5, 1.0],
                            "risk": [0.0, 0.1, 0.2 if severity == 0 else 0.3],
                        },
                    },
                    "temperature": {
                        "metrics": temp_metrics,
                        "reliability_bins": bins,
                    },
                },
            }
        )
    payload = {
        "kind": "checkpoint_evaluation",
        "checkpoint": identity,
        "evaluation_git_revision": "a" * 40,
        "conditions": conditions,
    }
    return [(Path("detail.json"), payload)], pd.DataFrame(rows)


def test_detail_geometry_is_cross_checked_and_emitted_as_tidy_data():
    details, metrics = _detail_fixture()
    reliability = extract_reliability_data(details, metrics)
    clean_raw = reliability.loc[
        (reliability["severity"] == 0)
        & (reliability["method"] == "raw_softmax")
        & (reliability["bin"] == 0)
    ].iloc[0]
    assert clean_raw["count"] == 2
    assert clean_raw["confidence"] == pytest.approx(0.4)
    assert clean_raw["mean_ece_from_metrics"] == pytest.approx(0.10)
    assert clean_raw["analysis_tier"] == "secondary/exploratory"

    curves = extract_risk_coverage_data(details, metrics, coverage_grid=[0.0, 0.5, 1.0])
    shifted_end = curves.loc[(curves["severity"] == 3) & (curves["coverage"] == 1.0)].iloc[0]
    assert shifted_end["mean_risk"] == pytest.approx(0.3)
    assert shifted_end["mean_aurc"] == pytest.approx(0.22)
    assert shifted_end["mean_risk_at_80"] == pytest.approx(0.25)


def test_detail_scalar_disagreement_fails_before_plotting():
    details, metrics = _detail_fixture()
    mismatched = deepcopy(details)
    mismatched[0][1]["conditions"][0]["methods"]["raw_softmax"]["metrics"]["ece"] = 0.101
    with pytest.raises(ValueError, match="disagrees with metrics.csv"):
        extract_reliability_data(mismatched, metrics)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _attribution_fixture(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    checkpoint = tmp_path / "results/checkpoints/smids-r50.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"released-checkpoint")
    checkpoint_relative = checkpoint.relative_to(tmp_path).as_posix()
    checkpoint_digest = _digest(checkpoint)
    manifest_digest = "b" * 64
    corruption_digest = "c" * 64
    run_id = "real-run"
    attribution = tmp_path / "results/attribution"
    attribution.mkdir(parents=True)
    per_image_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(("gradcam", "gradcam++")):
        for severity in range(6):
            for sample in range(2):
                per_image_rows.append(
                    {
                        "dataset": "smids",
                        "model": "resnet50",
                        "seed": 2025,
                        "fold": "",
                        "run_id": run_id,
                        "checkpoint": checkpoint_relative,
                        "checkpoint_sha256": checkpoint_digest,
                        "manifest_sha256": manifest_digest,
                        "corruption_protocol_sha256": corruption_digest,
                        "method": method,
                        "corruption": "defocus_blur",
                        "severity": severity,
                        "path": f"sample-{sample}.png",
                        "spearman": 1.0 - 0.10 * severity - 0.01 * method_index,
                        "top_percent_iou": 1.0 - 0.12 * severity - 0.01 * method_index,
                    }
                )
    per_image = pd.DataFrame(per_image_rows)
    summary = (
        per_image.groupby(["method", "severity"], as_index=False)
        .agg(
            n_samples=("path", "size"),
            n_valid_spearman=("spearman", "count"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            n_valid_iou=("top_percent_iou", "count"),
            iou_mean=("top_percent_iou", "mean"),
            iou_std=("top_percent_iou", "std"),
        )
        .sort_values(["method", "severity"])
    )
    per_image_path = attribution / "attribution_stability.csv"
    summary_path = attribution / "attribution_stability_summary.csv"
    samples_path = attribution / "attribution_samples.csv"
    grid_path = attribution / "attribution_grid.png"
    stability_path = attribution / "attribution_stability.png"
    per_image.to_csv(per_image_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame(
        {
            "dataset_index": range(6),
            "path": [f"sample-{index}.png" for index in range(6)],
            "label": [index % 3 for index in range(6)],
        }
    ).to_csv(samples_path, index=False)
    Image.new("RGB", (640, 480), "white").save(grid_path)
    Image.new("RGB", (640, 480), "white").save(stability_path)
    config_path = tmp_path / "configs/smids_resnet50.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dataset: {name: smids}\n", encoding="utf-8")
    registry_path = tmp_path / "results/checkpoint_registry-stage1.csv"
    pd.DataFrame(
        [
            {
                "dataset": "smids",
                "model": "resnet50",
                "seed": 2025,
                "fold": "",
                "run_id": run_id,
                "checkpoint": checkpoint_relative,
            }
        ]
    ).to_csv(registry_path, index=False)
    provenance = {
        "checkpoint": {
            "path": checkpoint_relative,
            "sha256": checkpoint_digest,
            "run_id": run_id,
            "seed": 2025,
        },
        "dataset": {"name": "smids", "manifest_sha256": manifest_digest},
        "config": {
            "path": config_path.relative_to(tmp_path).as_posix(),
            "sha256": _digest(config_path),
        },
        "registry": {
            "path": registry_path.relative_to(tmp_path).as_posix(),
            "sha256": _digest(registry_path),
            "release_training_git_revision": "e" * 40,
            "row": {
                "dataset": "smids",
                "model": "resnet50",
                "seed": 2025,
                "fold": None,
                "run_id": run_id,
                "checkpoint": checkpoint_relative,
            },
        },
        "corruption_protocol_sha256": corruption_digest,
        "evaluation_git_revision": "d" * 40,
        "protocol": {
            "corruption": "defocus_blur",
            "methods": ["gradcam", "gradcam++"],
            "quantitative_severities": list(range(6)),
            "qualitative_sample_count": 6,
            "qualitative_paths_sha256": _digest(samples_path),
        },
        "outputs": {
            path.name: {
                "path": path.relative_to(tmp_path).as_posix(),
                "sha256": _digest(path),
            }
            for path in (per_image_path, summary_path, samples_path, grid_path, stability_path)
        },
    }
    (attribution / "attribution_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    metric_rows = []
    for severity in range(6):
        metric_rows.append(
            {
                "dataset": "smids",
                "model": "resnet50",
                "seed": "2025",
                "fold": "",
                "run_id": run_id,
                "checkpoint": checkpoint_relative,
                "manifest_sha256": manifest_digest,
                "corruption_protocol_sha256": corruption_digest,
                "corruption": "clean" if severity == 0 else "defocus_blur",
                "severity": severity,
                "method": "raw_softmax",
                "metric": "accuracy",
                "n_samples": 300,
                "value": 0.90 - 0.04 * severity,
            }
        )
    return attribution, pd.DataFrame(metric_rows)


def test_f7_joins_validated_attribution_to_matching_accuracy(tmp_path, monkeypatch):
    attribution, metrics = _attribution_fixture(tmp_path)
    monkeypatch.setattr(
        "experiments.generate_final_figures._require_committed_revision",
        lambda revision, _root, *, label: str(revision),
    )
    data, source, grid = build_attribution_accuracy_data(
        metrics,
        attribution_dir=attribution,
        repo_root=tmp_path,
    )
    assert len(data.loc[data["metric"] == "accuracy"]) == 6
    assert len(data.loc[data["metric"] == "spearman"]) == 12
    assert len(data.loc[data["metric"] == "top_percent_iou"]) == 12
    assert data["analysis_tier"].eq("secondary/exploratory").all()
    assert data.loc[(data["metric"] == "accuracy") & (data["severity"] == 5), "mean"].iloc[
        0
    ] == pytest.approx(0.70)
    assert source["evaluation_git_revision"] == "d" * 40
    assert grid == Path("results/attribution/attribution_grid.png")

    output = plot_attribution_accuracy(data, tmp_path / "f7.png")
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.info["dpi"][0] == pytest.approx(150, abs=1)


def test_f7_rejects_a_tampered_attribution_output(tmp_path, monkeypatch):
    attribution, metrics = _attribution_fixture(tmp_path)
    monkeypatch.setattr(
        "experiments.generate_final_figures._require_committed_revision",
        lambda revision, _root, *, label: str(revision),
    )
    with (attribution / "attribution_stability_summary.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="hash disagrees"):
        build_attribution_accuracy_data(
            metrics,
            attribution_dir=attribution,
            repo_root=tmp_path,
        )


def test_manifest_normalizes_every_file_path_to_the_repository(tmp_path):
    metrics_path = tmp_path / "results/metrics.csv"
    thresholds_path = tmp_path / "results/thresholds.csv"
    protocol_path = tmp_path / "configs/analysis_protocol.yaml"
    detail_path = tmp_path / "results/evaluation_details/run.json"
    data_path = tmp_path / "results/figure_data/f1.csv"
    figure_path = tmp_path / "results/figures/f1.png"
    for path in (metrics_path, thresholds_path, protocol_path, detail_path, data_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    pd.DataFrame({"evaluation_git_revision": ["a" * 40]}).to_csv(metrics_path, index=False)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), "white").save(figure_path)
    manifest = _manifest(
        metrics_path.resolve(),
        thresholds_path.resolve(),
        protocol_path.resolve(),
        [
            {
                "path": detail_path.resolve().as_posix(),
                "sha256": _digest(detail_path),
                "kind": "checkpoint_evaluation",
                "evaluation_git_revision": "a" * 40,
            }
        ],
        {"provenance": {"path": "results/attribution/provenance.json"}},
        {"f1": data_path.resolve()},
        {"f1": figure_path.resolve()},
        repo_root=tmp_path,
    )

    paths: list[str] = []

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "path":
                    paths.append(str(item))
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(manifest)
    assert paths
    assert not any(Path(path).is_absolute() for path in paths)
    assert not any("/Users/" in path for path in paths)
    with pytest.raises(ValueError, match="absolute path"):
        _assert_portable_manifest_paths({"source": {"path": "/Users/example/leak.csv"}})
