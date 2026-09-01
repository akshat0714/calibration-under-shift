"""Evaluate cached checkpoints over the full corruption/reliability grid."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.transforms import preprocessing_from_model_config
from src.evaluate import CheckpointEvaluator, PredictionBundle
from src.metrics.calibration import calibration_metrics, reliability_diagram_data
from src.metrics.classification import classification_metrics
from src.metrics.selective import risk_coverage_curve, selective_metrics
from src.shifts.severity import corruption_names
from src.uncertainty.conformal import fit_aps, prediction_set_metrics
from src.uncertainty.ensembles import ensemble_statistics
from src.uncertainty.ood_scores import (
    MahalanobisScorer,
    energy_score,
    max_softmax_uncertainty,
    shifted_input_auroc,
)
from src.uncertainty.temperature import fit_scaler, scaled_probabilities
from src.utils import git_revision, load_config, write_json

KEY_COLUMNS = [
    "dataset",
    "model",
    "seed",
    "fold",
    "corruption",
    "severity",
    "method",
    "metric",
]

DETAIL_SCHEMA_VERSION = 1
PRESPECIFIED_ENSEMBLE_DATASET = "smids"
PRESPECIFIED_ENSEMBLE_MODEL = "resnet50"
PRESPECIFIED_ENSEMBLE_MEMBERS = 5


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def _json_number(value: Any) -> float | int | None:
    if isinstance(value, int | np.integer):
        return int(value)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    return {name: _json_number(value) for name, value in metrics.items()}


def _reliability_details(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int
) -> list[dict[str, float | int | None]]:
    return [
        {name: _json_number(value) for name, value in row.items()}
        for row in reliability_diagram_data(probabilities, labels, n_bins=n_bins)
    ]


def _risk_coverage_details(
    probabilities: np.ndarray,
    labels: np.ndarray,
    uncertainty: np.ndarray | None = None,
) -> dict[str, list[float]]:
    if uncertainty is None:
        uncertainty = 1.0 - probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    coverage, risk = risk_coverage_curve(correct, uncertainty)
    return {"coverage": coverage.tolist(), "risk": risk.tolist()}


def _probability_details(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metrics: dict[str, float],
    n_bins: int,
    uncertainty: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "metrics": _json_metrics(metrics),
        "reliability_bins": _reliability_details(probabilities, labels, n_bins),
        "risk_coverage": _risk_coverage_details(probabilities, labels, uncertainty),
    }


def _score_details(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metrics: dict[str, float],
    uncertainty: np.ndarray,
) -> dict[str, Any]:
    return {
        "metrics": _json_metrics(metrics),
        "risk_coverage": _risk_coverage_details(probabilities, labels, uncertainty),
    }


def _write_detail_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        write_json(temporary, payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _rows(metadata: dict[str, Any], method: str, metrics: dict[str, float]) -> list[dict[str, Any]]:
    return [
        {**metadata, "method": method, "metric": name, "value": float(value)}
        for name, value in metrics.items()
    ]


def _probability_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int,
    uncertainty: np.ndarray | None = None,
    target_coverage: float = 0.8,
) -> dict[str, float]:
    metrics = classification_metrics(probabilities, labels)
    metrics.update(calibration_metrics(probabilities, labels, n_bins=n_bins))
    metrics.update(
        selective_metrics(
            probabilities,
            labels,
            uncertainty=uncertainty,
            target_coverage=target_coverage,
        )
    )
    metrics["mean_confidence"] = float(probabilities.max(axis=1).mean())
    metrics["mean_predictive_entropy"] = float(_entropy(probabilities).mean())
    return metrics


def _conditions(config: dict[str, Any]) -> list[tuple[str, int]]:
    evaluation = config.get("evaluation", {})
    corruptions = evaluation.get("corruptions", list(corruption_names()))
    severities = evaluation.get("severities", [1, 2, 3, 4, 5])
    return [("clean", 0)] + [
        (str(corruption), int(severity)) for corruption in corruptions for severity in severities
    ]


def _prior_shift(
    bundle: PredictionBundle, class_index: int, multiplier: float, seed: int
) -> PredictionBundle:
    """Resample with replacement after multiplying one class's empirical prior."""

    labels = bundle.labels
    classes, counts = np.unique(labels, return_counts=True)
    probabilities = counts.astype(np.float64)
    matching = np.where(classes == class_index)[0]
    if not len(matching):
        raise ValueError(f"prior-shift class {class_index} is absent from test data")
    probabilities[matching[0]] *= multiplier
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed)
    sampled_classes = rng.choice(classes, size=len(labels), p=probabilities)
    sampled_indices = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == value),
                size=int((sampled_classes == value).sum()),
                replace=True,
            )
            for value in classes
        ]
    )
    rng.shuffle(sampled_indices)
    return PredictionBundle(
        logits=bundle.logits[sampled_indices],
        labels=bundle.labels[sampled_indices],
        features=bundle.features[sampled_indices],
        paths=bundle.paths[sampled_indices],
    )


def evaluate_checkpoint(
    evaluator: CheckpointEvaluator,
    skip_mc_dropout: bool = False,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, int], PredictionBundle],
    PredictionBundle,
    dict[str, Any],
]:
    config = evaluator.config
    evaluation = config.get("evaluation", {})
    n_bins = int(evaluation.get("calibration_bins", 15))
    alpha = float(evaluation.get("conformal_alpha", 0.1))
    target_coverage = float(evaluation.get("risk_coverage_target", 0.8))
    evaluation_revision = git_revision()
    calibration = evaluator.infer("calibration", "clean", 0)
    temperature = fit_scaler(
        calibration.logits,
        calibration.labels,
        split=np.repeat("calibration", len(calibration.labels)),
        method="temperature",
    )
    vector = fit_scaler(
        calibration.logits,
        calibration.labels,
        split=np.repeat("calibration", len(calibration.labels)),
        method="vector",
    )
    aps = fit_aps(
        calibration.probabilities,
        calibration.labels,
        split=np.repeat("calibration", len(calibration.labels)),
        alpha=alpha,
    )
    train_bundle = evaluator.infer("train", "clean", 0)
    mahalanobis = MahalanobisScorer.fit(train_bundle.features, train_bundle.labels)
    clean = evaluator.infer("test", "clean", 0)
    clean_msp = max_softmax_uncertainty(clean.logits)
    clean_energy = energy_score(clean.logits)
    clean_mahalanobis = mahalanobis.score(clean.features)
    rows: list[dict[str, Any]] = []
    bundles: dict[tuple[str, int], PredictionBundle] = {}
    details: dict[str, Any] = {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "kind": "checkpoint_evaluation",
        "evaluation_git_revision": evaluation_revision,
        "device": str(evaluator.device),
        "checkpoint": {
            "path": str(evaluator.checkpoint_path),
            "sha256": evaluator.checkpoint_sha256,
            "run_id": evaluator.run_id,
            "dataset": config["dataset"]["name"],
            "model": evaluator.model.backbone_name,
            "seed": evaluator.seed,
            "fold": config["dataset"].get("fold"),
            "manifest_sha256": evaluator.manifest_digest,
            "corruption_protocol_sha256": evaluator.corruption_protocol_sha256,
        },
        "protocol": {
            "conditions": [
                {"corruption": corruption, "severity": severity}
                for corruption, severity in _conditions(config)
            ],
            "calibration_bins": n_bins,
            "conformal_alpha": alpha,
            "risk_coverage_target": target_coverage,
            "num_workers": int(evaluation.get("num_workers", 0)),
            "corruption_seed": int(evaluation.get("corruption_seed", 1729)),
            "prior_shift": evaluation.get("prior_shift"),
            "mc_dropout_passes": (
                None if skip_mc_dropout else int(evaluation.get("mc_dropout_passes", 30))
            ),
        },
        "calibration_fit": {
            "temperature": {
                "value": float(temperature.model.temperature.detach().cpu().item()),
                "before_nll": temperature.before_nll,
                "after_nll": temperature.after_nll,
                "iterations": temperature.iterations,
                "split": "calibration",
                "n_samples": len(calibration.labels),
            },
            "vector_scaling": {
                "scale": vector.model.log_scale.detach().exp().cpu().tolist(),
                "bias": vector.model.bias.detach().cpu().tolist(),
                "before_nll": vector.before_nll,
                "after_nll": vector.after_nll,
                "iterations": vector.iterations,
                "split": "calibration",
                "n_samples": len(calibration.labels),
            },
            "aps": {
                "alpha": aps.alpha,
                "threshold": aps.threshold,
                "calibration_size": aps.calibration_size,
                "split": "calibration",
            },
            "mahalanobis": {
                "split": "train",
                "n_samples": len(train_bundle.labels),
                "feature_dim": int(train_bundle.features.shape[1]),
                "regularization": 1e-4,
            },
        },
        "conditions": [],
    }

    all_conditions = _conditions(config)
    for corruption, severity in all_conditions:
        bundle = clean if corruption == "clean" else evaluator.infer("test", corruption, severity)
        bundles[(corruption, severity)] = bundle
        metadata = {
            "dataset": config["dataset"]["name"],
            "model": evaluator.model.backbone_name,
            "seed": evaluator.seed,
            "fold": config["dataset"].get("fold", ""),
            "corruption": corruption,
            "severity": severity,
            "n_samples": len(bundle.labels),
            "run_id": evaluator.run_id,
            "checkpoint": str(evaluator.checkpoint_path),
            "manifest_sha256": evaluator.manifest_digest,
            "corruption_protocol_sha256": evaluator.corruption_protocol_sha256,
            "evaluation_git_revision": evaluation_revision,
        }
        condition_details: dict[str, Any] = {
            "corruption": corruption,
            "severity": severity,
            "n_samples": len(bundle.labels),
            "methods": {},
        }
        raw_probabilities = bundle.probabilities
        current_msp = max_softmax_uncertainty(bundle.logits)
        raw_metrics = _probability_metrics(
            raw_probabilities,
            bundle.labels,
            n_bins,
            uncertainty=current_msp,
            target_coverage=target_coverage,
        )
        raw_metrics["mean_ood_score"] = float(current_msp.mean())
        if corruption != "clean":
            raw_metrics["shift_detection_auroc"] = shifted_input_auroc(clean_msp, current_msp)
        rows.extend(_rows(metadata, "raw_softmax", raw_metrics))
        condition_details["methods"]["raw_softmax"] = _probability_details(
            raw_probabilities,
            bundle.labels,
            raw_metrics,
            n_bins,
            uncertainty=current_msp,
        )

        temperature_probabilities = scaled_probabilities(temperature.model, bundle.logits)
        temperature_metrics = _probability_metrics(
            temperature_probabilities,
            bundle.labels,
            n_bins,
            target_coverage=target_coverage,
        )
        rows.extend(_rows(metadata, "temperature", temperature_metrics))
        condition_details["methods"]["temperature"] = _probability_details(
            temperature_probabilities,
            bundle.labels,
            temperature_metrics,
            n_bins,
        )

        vector_probabilities = scaled_probabilities(vector.model, bundle.logits)
        vector_metrics = _probability_metrics(
            vector_probabilities,
            bundle.labels,
            n_bins,
            target_coverage=target_coverage,
        )
        rows.extend(_rows(metadata, "vector_scaling", vector_metrics))
        condition_details["methods"]["vector_scaling"] = _probability_details(
            vector_probabilities,
            bundle.labels,
            vector_metrics,
            n_bins,
        )

        prediction_sets = aps.predict(raw_probabilities)
        aps_metrics = prediction_set_metrics(prediction_sets, bundle.labels)
        rows.extend(_rows(metadata, "aps", aps_metrics))
        condition_details["methods"]["aps"] = {"metrics": _json_metrics(aps_metrics)}

        current_energy = energy_score(bundle.logits)
        energy_metrics = selective_metrics(
            raw_probabilities,
            bundle.labels,
            uncertainty=current_energy,
            target_coverage=target_coverage,
        )
        energy_metrics["mean_ood_score"] = float(current_energy.mean())
        if corruption != "clean":
            energy_metrics["shift_detection_auroc"] = shifted_input_auroc(
                clean_energy, current_energy
            )
        rows.extend(_rows(metadata, "energy", energy_metrics))
        condition_details["methods"]["energy"] = _score_details(
            raw_probabilities,
            bundle.labels,
            energy_metrics,
            current_energy,
        )

        current_mahalanobis = mahalanobis.score(bundle.features)
        mahalanobis_metrics = selective_metrics(
            raw_probabilities,
            bundle.labels,
            uncertainty=current_mahalanobis,
            target_coverage=target_coverage,
        )
        mahalanobis_metrics["mean_ood_score"] = float(current_mahalanobis.mean())
        if corruption != "clean":
            mahalanobis_metrics["shift_detection_auroc"] = shifted_input_auroc(
                clean_mahalanobis, current_mahalanobis
            )
        rows.extend(_rows(metadata, "mahalanobis", mahalanobis_metrics))
        condition_details["methods"]["mahalanobis"] = _score_details(
            raw_probabilities,
            bundle.labels,
            mahalanobis_metrics,
            current_mahalanobis,
        )

        if not skip_mc_dropout:
            passes = int(evaluation.get("mc_dropout_passes", 30))
            mc_logits, mc_labels, mc_paths = evaluator.infer_mc_dropout(
                "test", corruption, severity, passes=passes
            )
            if not np.array_equal(mc_labels, bundle.labels) or not np.array_equal(
                mc_paths, bundle.paths
            ):
                raise AssertionError("MC-dropout sample order differs from deterministic inference")
            statistics = ensemble_statistics(mc_logits)
            mc_metrics = _probability_metrics(
                statistics["probabilities"],
                bundle.labels,
                n_bins,
                uncertainty=statistics["predictive_entropy"],
                target_coverage=target_coverage,
            )
            mc_metrics.update(
                {
                    "mean_mutual_information": float(statistics["mutual_information"].mean()),
                    "mean_variation_ratio": float(statistics["variation_ratio"].mean()),
                }
            )
            rows.extend(_rows(metadata, "mc_dropout", mc_metrics))
            condition_details["methods"]["mc_dropout"] = _probability_details(
                statistics["probabilities"],
                bundle.labels,
                mc_metrics,
                n_bins,
                uncertainty=statistics["predictive_entropy"],
            )
        details["conditions"].append(condition_details)

    prior_config = evaluation.get("prior_shift")
    if prior_config:
        shifted = _prior_shift(
            clean,
            class_index=int(prior_config["class_index"]),
            multiplier=float(prior_config.get("multiplier", 2.0)),
            seed=int(prior_config.get("seed", 931)),
        )
        bundles[("prior_shift", 1)] = shifted
        metadata = {
            "dataset": config["dataset"]["name"],
            "model": evaluator.model.backbone_name,
            "seed": evaluator.seed,
            "fold": config["dataset"].get("fold", ""),
            "corruption": "prior_shift",
            "severity": 1,
            "n_samples": len(shifted.labels),
            "run_id": evaluator.run_id,
            "checkpoint": str(evaluator.checkpoint_path),
            "manifest_sha256": evaluator.manifest_digest,
            "corruption_protocol_sha256": evaluator.corruption_protocol_sha256,
            "evaluation_git_revision": evaluation_revision,
        }
        prior_details: dict[str, Any] = {
            "corruption": "prior_shift",
            "severity": 1,
            "n_samples": len(shifted.labels),
            "methods": {},
        }
        prior_msp = max_softmax_uncertainty(shifted.logits)
        prior_raw_metrics = _probability_metrics(
            shifted.probabilities,
            shifted.labels,
            n_bins,
            uncertainty=prior_msp,
            target_coverage=target_coverage,
        )
        rows.extend(_rows(metadata, "raw_softmax", prior_raw_metrics))
        prior_details["methods"]["raw_softmax"] = _probability_details(
            shifted.probabilities,
            shifted.labels,
            prior_raw_metrics,
            n_bins,
            uncertainty=prior_msp,
        )

        prior_temperature_probabilities = scaled_probabilities(temperature.model, shifted.logits)
        prior_temperature_metrics = _probability_metrics(
            prior_temperature_probabilities,
            shifted.labels,
            n_bins,
            target_coverage=target_coverage,
        )
        rows.extend(_rows(metadata, "temperature", prior_temperature_metrics))
        prior_details["methods"]["temperature"] = _probability_details(
            prior_temperature_probabilities,
            shifted.labels,
            prior_temperature_metrics,
            n_bins,
        )

        prior_aps_metrics = prediction_set_metrics(
            aps.predict(shifted.probabilities), shifted.labels
        )
        rows.extend(_rows(metadata, "aps", prior_aps_metrics))
        prior_details["methods"]["aps"] = {"metrics": _json_metrics(prior_aps_metrics)}
        details["conditions"].append(prior_details)
    return rows, bundles, calibration, details


def evaluate_deep_ensemble(
    evaluators: list[CheckpointEvaluator],
    condition_bundles: list[dict[tuple[str, int], PredictionBundle]],
    calibration_bundles: list[PredictionBundle],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(evaluators) != PRESPECIFIED_ENSEMBLE_MEMBERS:
        raise ValueError(
            f"the prespecified deep ensemble requires exactly {PRESPECIFIED_ENSEMBLE_MEMBERS} "
            "members"
        )
    if len({evaluator.seed for evaluator in evaluators}) != len(evaluators):
        raise ValueError("deep-ensemble members must have distinct training seeds")
    if len({_ensemble_identity(evaluator) for evaluator in evaluators}) != 1:
        raise ValueError("deep-ensemble members must share dataset, model, manifest, and fold")
    config = evaluators[0].config
    if (
        config["dataset"]["name"] != PRESPECIFIED_ENSEMBLE_DATASET
        or evaluators[0].model.backbone_name != PRESPECIFIED_ENSEMBLE_MODEL
    ):
        raise ValueError("the prespecified deep ensemble is SMIDS ResNet50 only")
    evaluation = config.get("evaluation", {})
    n_bins = int(evaluation.get("calibration_bins", 15))
    alpha = float(evaluation.get("conformal_alpha", 0.1))
    target_coverage = float(evaluation.get("risk_coverage_target", 0.8))
    evaluation_revision = git_revision()
    reference_labels = calibration_bundles[0].labels
    reference_paths = calibration_bundles[0].paths
    for bundle in calibration_bundles[1:]:
        if not np.array_equal(bundle.labels, reference_labels) or not np.array_equal(
            bundle.paths, reference_paths
        ):
            raise ValueError("ensemble calibration sample order differs across checkpoints")
    calibration_stats = ensemble_statistics(
        np.stack([bundle.logits for bundle in calibration_bundles])
    )
    aps = fit_aps(
        calibration_stats["probabilities"],
        reference_labels,
        split=np.repeat("calibration", len(reference_labels)),
        alpha=alpha,
    )
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {
        "schema_version": DETAIL_SCHEMA_VERSION,
        "kind": "deep_ensemble_evaluation",
        "evaluation_git_revision": evaluation_revision,
        "dataset": config["dataset"]["name"],
        "model": evaluators[0].model.backbone_name,
        "fold": config["dataset"].get("fold"),
        "member_count": len(evaluators),
        "members": [
            {
                "run_id": evaluator.run_id,
                "seed": evaluator.seed,
                "checkpoint": str(evaluator.checkpoint_path),
                "checkpoint_sha256": evaluator.checkpoint_sha256,
            }
            for evaluator in evaluators
        ],
        "manifest_sha256": evaluators[0].manifest_digest,
        "corruption_protocol_sha256": evaluators[0].corruption_protocol_sha256,
        "calibration_fit": {
            "ensemble_aps": {
                "alpha": aps.alpha,
                "threshold": aps.threshold,
                "calibration_size": aps.calibration_size,
                "split": "calibration",
            }
        },
        "conditions": [],
    }
    shared_conditions = set.intersection(*(set(items) for items in condition_bundles))
    for corruption, severity in sorted(shared_conditions):
        bundles = [items[(corruption, severity)] for items in condition_bundles]
        labels, paths = bundles[0].labels, bundles[0].paths
        if any(
            not np.array_equal(bundle.labels, labels) or not np.array_equal(bundle.paths, paths)
            for bundle in bundles[1:]
        ):
            # Prior-shift resampling can differ only if configs/seeds were inconsistent;
            # ordinary device-shift grids must always be paired exactly.
            raise ValueError(f"ensemble sample order differs for {corruption} severity {severity}")
        stats = ensemble_statistics(np.stack([bundle.logits for bundle in bundles]))
        metadata = {
            "dataset": config["dataset"]["name"],
            "model": evaluators[0].model.backbone_name,
            "seed": "ensemble",
            "fold": config["dataset"].get("fold", ""),
            "corruption": corruption,
            "severity": severity,
            "n_samples": len(labels),
            "run_id": "ensemble-smids-resnet50",
            "checkpoint": "|".join(str(item.checkpoint_path) for item in evaluators),
            "manifest_sha256": evaluators[0].manifest_digest,
            "corruption_protocol_sha256": evaluators[0].corruption_protocol_sha256,
            "evaluation_git_revision": evaluation_revision,
        }
        metrics = _probability_metrics(
            stats["probabilities"],
            labels,
            n_bins,
            uncertainty=stats["predictive_entropy"],
            target_coverage=target_coverage,
        )
        metrics.update(
            {
                "mean_mutual_information": float(stats["mutual_information"].mean()),
                "mean_expected_entropy": float(stats["expected_entropy"].mean()),
                "mean_variation_ratio": float(stats["variation_ratio"].mean()),
            }
        )
        rows.extend(_rows(metadata, "deep_ensemble", metrics))
        ensemble_aps_metrics = prediction_set_metrics(aps.predict(stats["probabilities"]), labels)
        rows.extend(_rows(metadata, "ensemble_aps", ensemble_aps_metrics))
        details["conditions"].append(
            {
                "corruption": corruption,
                "severity": severity,
                "n_samples": len(labels),
                "methods": {
                    "deep_ensemble": _probability_details(
                        stats["probabilities"],
                        labels,
                        metrics,
                        n_bins,
                        uncertainty=stats["predictive_entropy"],
                    ),
                    "ensemble_aps": {"metrics": _json_metrics(ensemble_aps_metrics)},
                },
            }
        )
    return rows, details


def _ensemble_identity(evaluator: CheckpointEvaluator) -> tuple[Any, ...]:
    return (
        evaluator.config["dataset"]["name"],
        evaluator.model.backbone_name,
        int(evaluator.config["dataset"]["num_classes"]),
        int(evaluator.config["model"]["input_size"]),
        tuple(sorted(preprocessing_from_model_config(evaluator.config["model"]).items())),
        evaluator.manifest_digest,
        evaluator.config["dataset"].get("fold", ""),
    )


def ensemble_group_indices(evaluators: list[CheckpointEvaluator]) -> list[list[int]]:
    """Return only the prespecified five-member SMIDS ResNet50 ensemble."""

    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, evaluator in enumerate(evaluators):
        groups.setdefault(_ensemble_identity(evaluator), []).append(index)
    selected: list[list[int]] = []
    for identity, indices in groups.items():
        seeds = [evaluators[index].seed for index in indices]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                "checkpoint list contains duplicate seeds within one dataset/model/fold"
            )
        dataset, model = identity[:2]
        if (
            dataset == PRESPECIFIED_ENSEMBLE_DATASET
            and model == PRESPECIFIED_ENSEMBLE_MODEL
            and len(indices) == PRESPECIFIED_ENSEMBLE_MEMBERS
        ):
            selected.append(indices)
    return selected


def validate_checkpoint_matrix(
    evaluators: list[CheckpointEvaluator], config: dict[str, Any]
) -> None:
    """Require the exact configured seed/fold matrix for canonical evaluation."""

    expected_seeds = {
        int(value) for value in config["training"].get("seeds", [config["training"]["seed"]])
    }
    dataset = config["dataset"]
    expected_folds = (
        {int(value) for value in dataset["folds"]}
        if "folds" in dataset
        else {dataset.get("fold", "")}
    )
    expected = {(seed, fold) for seed in expected_seeds for fold in expected_folds}
    observed = {
        (evaluator.seed, evaluator.config["dataset"].get("fold", "")) for evaluator in evaluators
    }
    if len(observed) != len(evaluators):
        raise ValueError("checkpoint matrix contains duplicate seed/fold members")
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise ValueError(
            f"checkpoint matrix is incomplete or unexpected; missing={missing}, extra={extra}. "
            "Use --allow-partial only for exploratory noncanonical output."
        )


def _write_metrics(rows: list[dict[str, Any]], output: Path) -> None:
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument(
        "--details-dir",
        type=Path,
        default=Path("results/evaluation_details"),
        help="per-checkpoint and ensemble provenance, reliability-bin, and risk-coverage JSONs",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        help="override evaluation data-loader workers (scheduling only)",
    )
    parser.add_argument("--skip-mc-dropout", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.skip_mc_dropout and not args.allow_partial:
        raise SystemExit("--skip-mc-dropout requires --allow-partial")
    if args.num_workers is not None and args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    config = load_config(args.config)
    if args.num_workers is not None:
        config.setdefault("evaluation", {})["num_workers"] = args.num_workers
    evaluators = [
        CheckpointEvaluator(
            checkpoint,
            config=config,
            device=args.device,
            cache_dir=config.get("outputs", {}).get("cache_dir", "results/cache"),
        )
        for checkpoint in args.checkpoints
    ]
    if not args.allow_partial:
        validate_checkpoint_matrix(evaluators, config)
    ensemble_groups = ensemble_group_indices(evaluators)
    condition_bundles: list[dict[tuple[str, int], PredictionBundle]] = []
    calibration_bundles: list[PredictionBundle] = []
    rows: list[dict[str, Any]] = []
    progress_output = args.output.with_name(f"{args.output.stem}.partial{args.output.suffix}")
    for evaluator in evaluators:
        checkpoint_rows, bundles, calibration, details = evaluate_checkpoint(
            evaluator, skip_mc_dropout=args.skip_mc_dropout
        )
        condition_bundles.append(bundles)
        calibration_bundles.append(calibration)
        rows.extend(checkpoint_rows)
        _write_detail_json(details, args.details_dir / f"{evaluator.run_id}.json")
        _write_metrics(rows, progress_output)
        print(f"evaluated {evaluator.checkpoint_path}")
    for indices in ensemble_groups:
        ensemble_rows, ensemble_details = evaluate_deep_ensemble(
            [evaluators[index] for index in indices],
            [condition_bundles[index] for index in indices],
            [calibration_bundles[index] for index in indices],
        )
        rows.extend(ensemble_rows)
        _write_detail_json(
            ensemble_details,
            args.details_dir / "ensemble-smids-resnet50.json",
        )
    _write_metrics(rows, args.output)
    progress_output.unlink(missing_ok=True)
    print(f"wrote {len(rows)} tidy metric rows to {args.output}")


if __name__ == "__main__":
    main()
