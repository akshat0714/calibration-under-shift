from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from experiments.analysis_artifacts import (
    _divergence_interval,
    _ece_transfer_table,
    _status,
    build_main_table,
    finding_text,
    generate_analysis_artifacts,
    validate_analysis_inputs,
)
from experiments.analyze import threshold_analysis

GROUPS = (
    ("smids", "resnet50", tuple(range(2025, 2030)), ("",)),
    ("smids", "xception", tuple(range(2025, 2028)), ("",)),
    ("smids", "mobilenet_v3_large", tuple(range(2025, 2028)), ("",)),
    ("hushem", "resnet50", (2025,), tuple(range(5))),
)
CORRUPTIONS = (
    "defocus_blur",
    "motion_blur",
    "gaussian_noise",
    "shot_noise",
    "jpeg",
    "resample",
    "illumination",
)


def _protocol() -> dict:
    return {
        "aggregation": {
            "device_corruptions": list(CORRUPTIONS),
            "severities": [1, 2, 3, 4, 5],
            "required_method_metrics": {
                "raw_softmax": "accuracy",
                "temperature": "ece",
                "vector_scaling": "ece",
                "aps": "conformal_coverage",
                "energy": "mean_ood_score",
                "mahalanobis": "mean_ood_score",
                "mc_dropout": "accuracy",
            },
            "ensemble_method_metrics": {
                "deep_ensemble": "accuracy",
                "ensemble_aps": "conformal_coverage",
            },
        },
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
        },
    }


def _condition_value(metric: str, severity: int, replicate_offset: float) -> float:
    if metric == "accuracy":
        return (0.90 if severity == 0 else 0.88 if severity == 1 else 0.83) + replicate_offset
    if metric == "ece":
        return (0.04 if severity == 0 else 0.10) + abs(replicate_offset)
    if metric == "mean_predictive_entropy":
        return (0.20 if severity == 0 else 0.45) + abs(replicate_offset)
    if metric == "risk_at_80_coverage":
        return (0.05 if severity == 0 else 0.08 if severity == 1 else 0.12) + abs(replicate_offset)
    if metric == "conformal_coverage":
        return (0.90 if severity == 0 else 0.86 if severity == 1 else 0.84) - abs(replicate_offset)
    if metric == "conformal_mean_set_size":
        return 1.15 + 0.15 * severity + abs(replicate_offset)
    if metric == "mean_ood_score":
        return 0.10 + 0.05 * severity + abs(replicate_offset)
    if metric == "mean_mutual_information":
        return 0.01 + 0.02 * severity
    raise AssertionError(metric)


def _row(
    dataset: str,
    model: str,
    seed: int | str,
    fold: int | str,
    corruption: str,
    severity: int,
    method: str,
    metric: str,
    value: float,
    checkpoint: str,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "fold": fold,
        "corruption": corruption,
        "severity": severity,
        "n_samples": 300 if dataset == "smids" else 43,
        "checkpoint": checkpoint,
        "corruption_protocol_sha256": "a" * 64,
        "method": method,
        "metric": metric,
        "value": value,
    }


def _metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    conditions = [("clean", 0)] + [
        (corruption, severity) for corruption in CORRUPTIONS for severity in range(1, 6)
    ]
    base_metrics = {
        "raw_softmax": (
            "accuracy",
            "ece",
            "mean_predictive_entropy",
            "risk_at_80_coverage",
        ),
        "temperature": ("ece",),
        "vector_scaling": ("ece",),
        "aps": ("conformal_coverage", "conformal_mean_set_size"),
        "energy": ("mean_ood_score", "risk_at_80_coverage"),
        "mahalanobis": ("mean_ood_score",),
        "mc_dropout": ("accuracy",),
    }
    for dataset, model, seeds, folds in GROUPS:
        replicate_number = 0
        checkpoints: list[str] = []
        for seed in seeds:
            for fold in folds:
                replicate_offset = (replicate_number - (len(seeds) * len(folds) - 1) / 2) * 0.002
                replicate_number += 1
                checkpoint = f"/checkpoints/{dataset}-{model}-{seed}-{fold or 'none'}.pt"
                checkpoints.append(checkpoint)
                for corruption, severity in conditions:
                    for method, metric_names in base_metrics.items():
                        for metric in metric_names:
                            value = _condition_value(metric, severity, replicate_offset)
                            if method == "temperature" and metric == "ece":
                                value *= 0.75
                            elif method == "vector_scaling" and metric == "ece":
                                value *= 0.70
                            elif method == "energy" and metric == "risk_at_80_coverage":
                                value = max(0.01, value - 0.02)
                            rows.append(
                                _row(
                                    dataset,
                                    model,
                                    seed,
                                    fold,
                                    corruption,
                                    severity,
                                    method,
                                    metric,
                                    value,
                                    checkpoint,
                                )
                            )
        if (dataset, model) == ("smids", "resnet50"):
            ensemble_checkpoint = "|".join(checkpoints)
            for corruption, severity in conditions:
                for method, metric_names in {
                    "deep_ensemble": (
                        "accuracy",
                        "mean_mutual_information",
                        "risk_at_80_coverage",
                    ),
                    "ensemble_aps": (
                        "conformal_coverage",
                        "conformal_mean_set_size",
                    ),
                }.items():
                    for metric in metric_names:
                        value = _condition_value(metric, severity, 0.0)
                        if method == "deep_ensemble" and metric == "accuracy":
                            value += 0.03
                        rows.append(
                            _row(
                                dataset,
                                model,
                                "ensemble",
                                "",
                                corruption,
                                severity,
                                method,
                                metric,
                                value,
                                ensemble_checkpoint,
                            )
                        )
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    metrics = _metrics()
    protocol = _protocol()
    thresholds = threshold_analysis(metrics, protocol)
    metrics_path = tmp_path / "metrics.csv"
    thresholds_path = tmp_path / "thresholds.csv"
    protocol_path = tmp_path / "analysis_protocol.yaml"
    metrics.to_csv(metrics_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    protocol_path.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    return metrics_path, thresholds_path, protocol_path


def test_analysis_artifacts_generate_strict_primary_secondary_and_figure(tmp_path):
    metrics_path, thresholds_path, protocol_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "analysis"
    figure = tmp_path / "figures" / "headline.png"

    outputs = generate_analysis_artifacts(
        metrics_path,
        thresholds_path,
        protocol_path,
        output_dir,
        figure,
    )

    assert all(path.is_file() for path in outputs.values())
    main = pd.read_csv(outputs["main_table_csv"])
    assert len(main) == 16
    assert set(main["status"]) == {"earlier", "same_or_later"}
    assert set(main.groupby(["dataset", "model"]).size()) == {4}
    assert (
        main.loc[
            (main["dataset"] == "hushem") & (main["model"] == "resnet50"),
            "n_replicates",
        ]
        .eq(5)
        .all()
    )
    assert np.isfinite(main["clean_signal_std"]).all()

    finding = outputs["finding"].read_text(encoding="utf-8")
    ece_transfer = pd.read_csv(outputs["secondary_ece_transfer_csv"])
    assert finding == finding_text(main, ece_transfer)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", finding.strip())
    assert len(sentences) == 2
    assert sentences[1].startswith("In secondary analysis, ")
    assert "8 of 16 evaluable" in finding
    assert "mixed" in finding

    ensemble_mi = pd.read_csv(outputs["secondary_ensemble_mi_csv"])
    assert set(ensemble_mi["dataset"]) == {"smids"}
    assert set(ensemble_mi.loc[ensemble_mi["model"] == "resnet50", "ensemble_members"]) == {5}
    selective = pd.read_csv(outputs["secondary_selective_csv"])
    hushem_selectors = set(selective.loc[selective["dataset"] == "hushem", "selector_method"])
    assert hushem_selectors == {"energy"}
    assert set(selective["severity"]) == {3, 4}
    assert (selective["analysis_tier"] == "secondary/exploratory").all()
    ensemble_selective = selective.loc[selective["selector_method"] == "deep_ensemble"]
    assert np.allclose(ensemble_selective["unselective_accuracy_mean"], 0.86)
    assert np.allclose(ensemble_selective["clean_accuracy_mean"], 0.93)
    assert np.allclose(ensemble_selective["gain_vs_unselective"], 0.02)
    assert np.allclose(ensemble_selective["difference_from_clean"], -0.05)

    with Image.open(outputs["analysis_overview_figure"]) as rendered:
        assert rendered.format == "PNG"
        assert rendered.width > 1_000
        assert rendered.height > 1_000
        assert rendered.info["dpi"][0] == pytest.approx(150, abs=1)
        assert rendered.info["dpi"][1] == pytest.approx(150, abs=1)

    manifest = outputs["manifest"].read_text(encoding="utf-8")
    assert "secondary_exploratory" in manifest
    assert "sha256" in manifest


def test_analysis_artifacts_reject_threshold_file_that_disagrees_with_recomputation(tmp_path):
    metrics_path, thresholds_path, protocol_path = _write_inputs(tmp_path)
    thresholds = pd.read_csv(thresholds_path)
    thresholds.loc[0, "signal_crossing_severity"] = 5
    thresholds.to_csv(thresholds_path, index=False)

    with pytest.raises(ValueError, match="disagrees with frozen recomputation"):
        generate_analysis_artifacts(
            metrics_path,
            thresholds_path,
            protocol_path,
            tmp_path / "analysis",
            tmp_path / "headline.png",
        )


def test_divergence_shading_interval_requires_two_observed_ordered_crossings():
    assert _divergence_interval(1, 3) == (1.0, 3.0)
    assert _divergence_interval(3, 3) is None
    assert _divergence_interval(4, 3) is None
    assert _divergence_interval(np.nan, 3) is None
    assert _divergence_interval(1, np.nan) is None


def test_missing_accuracy_crossing_has_distinct_status_not_negative():
    metrics = _metrics()
    protocol = _protocol()
    no_accuracy_drop = (
        (metrics["dataset"] == "smids")
        & (metrics["model"] == "xception")
        & (metrics["seed"] != "ensemble")
        & (metrics["method"] == "raw_softmax")
        & (metrics["metric"] == "accuracy")
        & (metrics["severity"] > 0)
    )
    metrics.loc[no_accuracy_drop, "value"] = 0.88
    thresholds = threshold_analysis(metrics, protocol)

    normalized, verified = validate_analysis_inputs(metrics, thresholds, protocol)
    main = build_main_table(normalized, verified)
    xception = main.loc[(main["dataset"] == "smids") & (main["model"] == "xception")]

    assert xception["status"].eq("accuracy_did_not_cross").all()
    assert xception["accuracy_drop_severity"].isna().all()
    finding = finding_text(main, _ece_transfer_table(normalized, protocol))
    assert "accuracy did not cross" in finding
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", finding.strip())
    assert len(sentences) == 2
    assert sentences[1].startswith("In secondary analysis, ")


def test_crossing_statuses_distinguish_every_missing_case():
    assert _status(1, 3) == "earlier"
    assert _status(3, 3) == "same_or_later"
    assert _status(4, 3) == "same_or_later"
    assert _status(np.nan, 3) == "signal_did_not_cross"
    assert _status(1, np.nan) == "accuracy_did_not_cross"
    assert _status(np.nan, np.nan) == "neither_crossed"


def test_analysis_artifacts_reject_replaced_seed_even_when_replicate_count_matches(tmp_path):
    metrics = _metrics()
    protocol = _protocol()
    thresholds = threshold_analysis(metrics, protocol)
    replaced = metrics.copy()
    mask = (replaced["dataset"] == "smids") & (replaced["model"] == "xception")
    replaced.loc[mask & (replaced["seed"].astype(str) == "2027"), "seed"] = "9999"

    with pytest.raises(ValueError, match="seed and fold identities"):
        validate_analysis_inputs(replaced, thresholds, protocol)
