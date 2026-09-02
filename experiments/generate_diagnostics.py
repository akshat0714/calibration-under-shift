"""Regenerate per-checkpoint reliability, risk-coverage, and corruption figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import CheckpointEvaluator
from src.metrics.selective import risk_coverage_curve
from src.uncertainty.temperature import fit_scaler, scaled_probabilities
from src.utils import load_config
from src.viz.figures import (
    plot_corruption_grid,
    plot_reliability_diagram,
    plot_risk_coverage,
)


def _sample_image_path(evaluator: CheckpointEvaluator) -> Path:
    row = evaluator.manifest.loc[evaluator.manifest["split"] == "test"].iloc[0]
    path = Path(str(row["path"]))
    if not path.is_absolute():
        path = Path(evaluator.config["dataset"]["root"]) / path
    return path


def generate_diagnostics(
    evaluator: CheckpointEvaluator,
    output_dir: Path,
    corruption: str = "defocus_blur",
    severities: tuple[int, ...] = (0, 3, 5),
) -> dict[str, Path]:
    """Generate diagnostic figures using only saved checkpoint predictions.

    Temperature is fit once on the clean calibration role. Every plotted test
    condition reuses that frozen scaler. No shifted or test observation is fitted.
    """

    if not severities or severities[0] != 0 or any(level not in range(6) for level in severities):
        raise ValueError("severities must start with 0 and contain only values from 0 through 5")
    calibration = evaluator.infer("calibration", "clean", 0)
    fitted = fit_scaler(
        calibration.logits,
        calibration.labels,
        split=np.repeat("calibration", len(calibration.labels)),
        method="temperature",
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    curves: list[pd.DataFrame] = []
    for severity in severities:
        condition = "clean" if severity == 0 else corruption
        bundle = evaluator.infer("test", condition, severity)
        raw = bundle.probabilities
        calibrated = scaled_probabilities(fitted.model, bundle.logits)
        label = "clean" if severity == 0 else f"{corruption}-s{severity}"
        outputs[f"reliability_{label}"] = plot_reliability_diagram(
            {"Raw softmax": raw, "Temperature": calibrated},
            bundle.labels,
            destination / f"reliability-{label}.png",
            n_bins=int(evaluator.config.get("evaluation", {}).get("calibration_bins", 15)),
            title=f"Reliability · {label}",
        )
        correct = raw.argmax(axis=1) == bundle.labels
        coverage, risk = risk_coverage_curve(correct, 1.0 - raw.max(axis=1))
        curves.append(
            pd.DataFrame(
                {
                    "coverage": coverage,
                    "risk": risk,
                    "label": "Clean" if severity == 0 else f"Severity {severity}",
                }
            )
        )
    outputs["risk_coverage"] = plot_risk_coverage(
        pd.concat(curves, ignore_index=True),
        destination / "risk_coverage.png",
        title=f"Selective prediction · {corruption}",
    )
    outputs["corruption_grid"] = plot_corruption_grid(
        _sample_image_path(evaluator),
        destination / "corruption_grid.png",
        seed=int(evaluator.config.get("evaluation", {}).get("corruption_seed", 1729)),
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--corruption", default="defocus_blur")
    parser.add_argument("--severities", nargs="+", type=int, default=[0, 3, 5])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluator = CheckpointEvaluator(
        args.checkpoint,
        config=load_config(args.config),
        device=args.device,
    )
    outputs = generate_diagnostics(
        evaluator,
        args.output_dir,
        corruption=args.corruption,
        severities=tuple(args.severities),
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
