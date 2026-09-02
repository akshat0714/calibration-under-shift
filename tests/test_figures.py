from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.viz.figures import (
    generate_metric_figures,
    main,
    plot_corruption_grid,
    plot_headline,
    plot_reliability_diagram,
    plot_risk_coverage,
    summarize_over_corruptions,
)


@pytest.fixture
def tidy_metrics() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    conditions = (("clean", 0), ("defocus_blur", 1), ("jpeg", 1), ("defocus_blur", 2), ("jpeg", 2))
    for seed in (11, 29):
        seed_offset = 0.01 if seed == 29 else 0.0
        for corruption, severity in conditions:
            shift = 0.035 * severity + (0.01 if corruption == "jpeg" else 0.0)
            values_by_method = {
                "raw_softmax": {
                    "accuracy": 0.91 - shift - seed_offset,
                    "ece": 0.04 + 1.2 * shift + seed_offset,
                    "mean_predictive_entropy": 0.28 + 1.4 * shift + seed_offset,
                    "failure_detection_auroc": 0.66 + 0.08 * severity - seed_offset,
                },
                "temperature": {
                    "accuracy": 0.91 - shift - seed_offset,
                    "ece": 0.025 + shift + seed_offset,
                },
                "aps": {
                    "conformal_coverage": 0.92 - 0.035 * severity - seed_offset,
                    "conformal_mean_set_size": 1.15 + 0.30 * severity + seed_offset,
                },
            }
            for method, metric_values in values_by_method.items():
                for metric, value in metric_values.items():
                    rows.append(
                        {
                            "dataset": "toy-cells",
                            "model": "tiny-cnn",
                            "seed": seed,
                            "fold": 0,
                            "corruption": corruption,
                            "severity": severity,
                            "method": method,
                            "metric": metric,
                            "value": value,
                        }
                    )
    # This population-shift row must not enter the device-corruption average.
    rows.append(
        {
            "dataset": "toy-cells",
            "model": "tiny-cnn",
            "seed": 11,
            "fold": 0,
            "corruption": "prior_shift",
            "severity": 1,
            "method": "raw_softmax",
            "metric": "accuracy",
            "value": 0.01,
        }
    )
    return pd.DataFrame(rows)


def _assert_png(path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 1_000
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width > 200 and image.height > 150
        dpi = image.info.get("dpi")
        assert dpi is not None
        assert dpi[0] == pytest.approx(150, abs=1)
        assert dpi[1] == pytest.approx(150, abs=1)


def test_metric_figures_and_hierarchical_summary(tidy_metrics, tmp_path):
    summary = summarize_over_corruptions(tidy_metrics, "accuracy")
    severity_one = summary.loc[
        (summary["method"] == "raw_softmax") & (summary["severity"] == 1)
    ].iloc[0]
    # Each seed first averages defocus and JPEG. prior_shift is not a device corruption.
    expected = np.mean([np.mean([0.875, 0.865]), np.mean([0.865, 0.855])])
    assert severity_one["mean"] == pytest.approx(expected)
    assert severity_one["n_replicates"] == 2
    assert np.isfinite(severity_one["lower"])
    assert np.isfinite(severity_one["upper"])

    outputs = generate_metric_figures(tidy_metrics, tmp_path, nominal_coverage=0.9)
    assert set(outputs) == {"headline", "failure_detection", "conformal"}
    for output in outputs.values():
        _assert_png(output)


def test_probability_curve_and_corruption_figures(tmp_path):
    probabilities = np.array(
        [
            [0.90, 0.10],
            [0.75, 0.25],
            [0.40, 0.60],
            [0.20, 0.80],
            [0.55, 0.45],
            [0.30, 0.70],
        ]
    )
    labels = np.array([0, 0, 1, 1, 1, 0])
    reliability = plot_reliability_diagram(
        probabilities, labels, tmp_path / "reliability.png", n_bins=4
    )

    curves = pd.DataFrame(
        {
            "coverage": [0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
            "risk": [0.0, 0.0, 0.33, 0.0, 0.20, 0.33],
            "method": ["entropy"] * 3 + ["max_softmax"] * 3,
            "severity": [2] * 6,
        }
    )
    risk_coverage = plot_risk_coverage(curves, tmp_path / "risk_coverage.png")

    y, x = np.mgrid[:24, :24]
    sample = np.stack((x * 10, y * 10, (x + y) * 5), axis=-1).astype(np.uint8)
    corruption_grid = plot_corruption_grid(
        sample,
        tmp_path / "corruption_grid.png",
        corruptions=("defocus_blur", "jpeg"),
        severities=(0, 1, 5),
        seed=7,
    )
    for output in (reliability, risk_coverage, corruption_grid):
        _assert_png(output)


def test_missing_metrics_remain_explicit_and_cli_writes_outputs(tidy_metrics, tmp_path):
    accuracy_only = tidy_metrics.loc[tidy_metrics["metric"] == "accuracy"]
    missing_panels = plot_headline(accuracy_only, tmp_path / "partial.png")
    _assert_png(missing_panels)

    metrics_csv = tmp_path / "metrics.csv"
    tidy_metrics.to_csv(metrics_csv, index=False)
    cli_dir = tmp_path / "cli"
    outputs = main([str(metrics_csv), "--output-dir", str(cli_dir)])
    assert set(outputs) == {"headline", "failure_detection", "conformal"}
    for output in outputs.values():
        _assert_png(output)
