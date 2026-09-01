"""Generate real-checkpoint Grad-CAM panels and attribution-stability results."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes

from experiments.evaluate_clean_matrix import validate_stage1_registry
from experiments.run_stage2_matrix import require_clean_git_revision
from scripts.release_checkpoints import (
    INTERNAL_MANIFEST,
    TRAINING_SHA,
    verify_pinned_internal_manifest,
)
from src.attribution.gradcam import GradCAM, GradCAMResult, overlay_heatmaps
from src.attribution.stability import pair_stability
from src.data.transforms import preprocessing_from_model_config
from src.evaluate import CheckpointEvaluator
from src.utils import git_revision, load_config, set_seed

CAM_METHODS = ("gradcam", "gradcam++")
DEFAULT_FIGURE_SEVERITIES = (0, 2, 4)
_METHOD_LABELS = {"gradcam": "Grad-CAM", "gradcam++": "Grad-CAM++"}
_METHOD_COLORS = {"gradcam": "#0072B2", "gradcam++": "#D55E00"}
_METHOD_MARKERS = {"gradcam": "o", "gradcam++": "s"}


def _stratified_indices(evaluator: CheckpointEvaluator, count: int) -> list[int]:
    """Select a deterministic, class-interleaved prefix of the test manifest."""

    frame = evaluator.manifest.loc[evaluator.manifest["split"] == "test"].reset_index(drop=True)
    if count <= 0 or count >= len(frame):
        return list(range(len(frame)))
    by_class = {
        label: iter(frame.index[frame["label"] == label].tolist())
        for label in sorted(frame["label"].unique())
    }
    selected: list[int] = []
    while len(selected) < count:
        added = False
        for iterator in by_class.values():
            try:
                selected.append(next(iterator))
                added = True
            except StopIteration:
                continue
            if len(selected) == count:
                break
        if not added:
            break
    return selected


def _batch_from_dataset(dataset: Any, indices: list[int], device: torch.device) -> dict[str, Any]:
    samples = [dataset[index] for index in indices]
    return {
        "image": torch.stack([sample["image"] for sample in samples]).to(device),
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "path": [str(sample["path"]) for sample in samples],
    }


def _display_images(normalized: torch.Tensor, model_config: dict[str, Any]) -> np.ndarray:
    preprocessing = preprocessing_from_model_config(model_config)
    mean = torch.tensor(preprocessing["mean"], device=normalized.device)[None, :, None, None]
    std = torch.tensor(preprocessing["std"], device=normalized.device)[None, :, None, None]
    return (normalized * std + mean).clamp(0, 1).detach().cpu().numpy()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _portable_identifier(path: str | Path, root: Path, digest: str | None = None) -> str:
    """Return a repository-relative path or a non-path external identifier."""

    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(root.resolve()).as_posix()
    except ValueError:
        suffix = f"@sha256:{digest}" if digest else ""
        return f"external:{candidate.name}{suffix}"


def _validate_severities(
    severities: Sequence[int], figure_severities: Sequence[int]
) -> tuple[list[int], list[int]]:
    quantitative = sorted(set(int(level) for level in severities))
    qualitative = list(dict.fromkeys(int(level) for level in figure_severities))
    if not quantitative or quantitative[0] != 0:
        raise ValueError("quantitative severities must include clean severity 0")
    if any(level not in range(6) for level in quantitative):
        raise ValueError("quantitative severities must be integers from 0 through 5")
    if not qualitative or qualitative[0] != 0:
        raise ValueError("figure severities must start with clean severity 0")
    if any(level not in quantitative for level in qualitative):
        raise ValueError("figure severities must be a subset of quantitative severities")
    return quantitative, qualitative


def _normalize_methods(method: str) -> tuple[str, ...]:
    if method == "both":
        return CAM_METHODS
    if method not in CAM_METHODS:
        raise ValueError("method must be 'gradcam', 'gradcam++', or 'both'")
    return (method,)


def _registry_row(
    registry_path: Path | None,
    checkpoint_path: Path,
    evaluator: CheckpointEvaluator,
    root: Path,
) -> dict[str, Any] | None:
    """Require one unique checkpoint member of the pinned Stage-1 release."""

    if registry_path is None:
        return None
    registry_path = registry_path.resolve()
    registry = validate_stage1_registry(pd.read_csv(registry_path))
    verified = verify_pinned_internal_manifest(root)
    registry_relative = registry_path.relative_to(root)
    if registry_relative not in verified:
        raise ValueError(f"approved registry is not anchored by {INTERNAL_MANIFEST}")

    def resolved(value: object) -> Path:
        candidate = Path(str(value)).expanduser()
        return (candidate if candidate.is_absolute() else root / candidate).resolve()

    selected = registry.loc[registry["checkpoint"].map(resolved) == checkpoint_path.resolve()]
    if len(selected) != 1:
        raise ValueError(
            f"checkpoint must occur exactly once in approved registry; observed {len(selected)} rows"
        )
    checkpoint_relative = checkpoint_path.resolve().relative_to(root)
    if checkpoint_relative not in verified:
        raise ValueError("selected checkpoint is not anchored by the pinned release manifest")
    row = selected.iloc[0]
    expected = {
        "dataset": str(evaluator.config["dataset"]["name"]),
        "model": str(evaluator.config["model"]["backbone"]),
        "seed": str(int(evaluator.seed)),
        "run_id": str(evaluator.run_id),
    }
    observed = {
        "dataset": str(row["dataset"]),
        "model": str(row["model"]),
        "seed": str(int(row["seed"])),
        "run_id": str(row["run_id"]),
    }
    if observed != expected:
        raise ValueError(f"registry/checkpoint identity mismatch: {observed} != {expected}")
    fields = ("dataset", "model", "seed", "fold", "run_id", "checkpoint")
    return {
        "path": registry_relative.as_posix(),
        "sha256": _sha256(registry_path),
        "release_training_git_revision": TRAINING_SHA,
        "row": {
            key: None
            if pd.isna(row[key]) or row[key] == ""
            else row[key].item()
            if hasattr(row[key], "item")
            else row[key]
            for key in fields
        },
    }


def _datasets(
    evaluator: CheckpointEvaluator, corruption: str, severities: Sequence[int]
) -> dict[int, Any]:
    return {
        severity: evaluator.loader(
            "test", "clean" if severity == 0 else corruption, severity
        ).dataset
        for severity in severities
    }


def _explain_conditions(
    generator: GradCAM,
    datasets: dict[int, Any],
    indices: list[int],
    severities: Sequence[int],
    device: torch.device,
) -> dict[int, tuple[dict[str, Any], GradCAMResult]]:
    clean_batch = _batch_from_dataset(datasets[0], indices, device)
    # Work at the native spatial CAM resolution. Upsampling a 7x7 ResNet map
    # would manufacture thousands of interpolated ranks without information.
    clean_result = generator.explain(clean_batch["image"], resize=False)
    results = {0: (clean_batch, clean_result)}
    for severity in severities[1:]:
        batch = _batch_from_dataset(datasets[severity], indices, device)
        if batch["path"] != clean_batch["path"]:
            raise AssertionError("attribution batches are not sample-aligned")
        results[severity] = (
            batch,
            generator.explain(batch["image"], clean_result.target_classes, resize=False),
        )
    return results


def _method_rows(
    evaluator: CheckpointEvaluator,
    datasets: dict[int, Any],
    corruption: str,
    severities: list[int],
    indices: list[int],
    batch_size: int,
    method: str,
    figure_index_set: set[int],
    root: Path,
) -> list[dict[str, Any]]:
    generator = GradCAM(evaluator.model, method=method)
    rows: list[dict[str, Any]] = []
    dataset_name = str(evaluator.config["dataset"]["name"])
    backbone = str(evaluator.config["model"]["backbone"])
    fold = evaluator.config["dataset"].get("fold")
    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        results = _explain_conditions(
            generator, datasets, batch_indices, severities, evaluator.device
        )
        clean_maps = results[0][1].heatmaps.detach().float().cpu().numpy()
        for severity, (batch, result) in results.items():
            maps = result.heatmaps.detach().float().cpu().numpy()
            logits = result.logits.detach().float().cpu()
            probabilities = logits.softmax(dim=1).numpy()
            targets = result.target_classes.detach().cpu().numpy()
            target_confidences = result.confidences.detach().float().cpu().numpy()
            for position, path in enumerate(batch["path"]):
                stability = pair_stability(clean_maps[position], maps[position], top_fraction=0.2)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "model": backbone,
                        "seed": int(evaluator.seed),
                        "fold": fold,
                        "run_id": evaluator.run_id,
                        "checkpoint": _portable_identifier(
                            evaluator.checkpoint_path, root, evaluator.checkpoint_sha256
                        ),
                        "checkpoint_sha256": evaluator.checkpoint_sha256,
                        "manifest_sha256": evaluator.manifest_digest,
                        "corruption_protocol_sha256": evaluator.corruption_protocol_sha256,
                        "method": method,
                        "corruption": corruption,
                        "condition": "clean" if severity == 0 else corruption,
                        "severity": severity,
                        "path": path,
                        "true_label": int(batch["label"][position]),
                        "clean_target": int(targets[position]),
                        "predicted_label": int(probabilities[position].argmax()),
                        "target_confidence": float(target_confidences[position]),
                        "predicted_confidence": float(probabilities[position].max()),
                        "spearman": stability["spearman"],
                        "top_percent_iou": stability["top_percent_iou"],
                        "top_fraction": 0.2,
                        "heatmap_valid": bool(
                            np.isfinite(stability["spearman"])
                            and np.isfinite(stability["top_percent_iou"])
                        ),
                        "heatmap_height": int(maps.shape[-2]),
                        "heatmap_width": int(maps.shape[-1]),
                        "qualitative_sample": batch_indices[position] in figure_index_set,
                    }
                )
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["method", "severity"], as_index=False)
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
        .reset_index(drop=True)
    )


def _plot_stability(
    summary: pd.DataFrame,
    output: Path,
    *,
    corruption: str,
    dataset: str,
    model: str,
) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True)
    specs = (
        ("spearman_mean", "spearman_std", "Clean-to-shift Spearman correlation", (-1.05, 1.05)),
        ("iou_mean", "iou_std", "Clean-to-shift top-20% mask IoU", (-0.05, 1.05)),
    )
    for axis, (mean_column, std_column, title, limits) in zip(axes, specs, strict=True):
        for method in CAM_METHODS:
            selected = summary.loc[summary["method"] == method]
            if selected.empty:
                continue
            x = selected["severity"].to_numpy(dtype=float)
            mean = selected[mean_column].to_numpy(dtype=float)
            std = selected[std_column].fillna(0).to_numpy(dtype=float)
            color = _METHOD_COLORS[method]
            axis.plot(
                x,
                mean,
                color=color,
                marker=_METHOD_MARKERS[method],
                label=_METHOD_LABELS[method],
            )
            axis.fill_between(
                x,
                np.clip(mean - std, limits[0], limits[1]),
                np.clip(mean + std, limits[0], limits[1]),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        axis.axhline(0, color="#666666", linewidth=0.8, alpha=0.5)
        axis.set_title(title)
        axis.set_xlabel("Defocus severity" if corruption == "defocus_blur" else "Severity")
        axis.set_xticks(sorted(summary["severity"].unique()))
        axis.set_ylim(*limits)
        axis.grid(alpha=0.22)
    axes[0].set_ylabel("Attribution stability (mean ± 1 SD)")
    axes[1].legend(frameon=False, loc="lower left")
    figure.suptitle(
        f"Attribution stability under {corruption.replace('_', ' ')} · {dataset.upper()} {model}",
        fontsize=12,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def _panel_title(method: str, severity: int, corruption: str) -> str:
    condition = "clean" if severity == 0 else f"{corruption.replace('_', ' ')} · s{severity}"
    return f"{_METHOD_LABELS[method]}\n{condition}"


def _plot_qualitative_grid(
    evaluator: CheckpointEvaluator,
    datasets: dict[int, Any],
    methods: Sequence[str],
    figure_severities: list[int],
    figure_indices: list[int],
    corruption: str,
    output: Path,
) -> Path:
    columns = [(method, severity) for method in methods for severity in figure_severities]
    results: dict[str, dict[int, tuple[dict[str, Any], GradCAMResult]]] = {}
    for method in methods:
        results[method] = _explain_conditions(
            GradCAM(evaluator.model, method=method),
            datasets,
            figure_indices,
            figure_severities,
            evaluator.device,
        )
    figure, axes = plt.subplots(
        len(figure_indices),
        len(columns),
        figsize=(2.8 * len(columns), 2.6 * len(figure_indices)),
        squeeze=False,
    )
    class_names = evaluator.config["dataset"].get("class_names", [])
    for column, (method, severity) in enumerate(columns):
        batch, result = results[method][severity]
        overlays = overlay_heatmaps(
            _display_images(batch["image"], evaluator.config["model"]), result.heatmaps
        )
        probabilities = result.logits.detach().float().cpu().softmax(dim=1).numpy()
        confidences = result.confidences.detach().float().cpu().numpy()
        targets = result.target_classes.detach().cpu().numpy()
        for row, overlay in enumerate(overlays):
            axis: Axes = axes[row, column]
            axis.imshow(overlay)
            if row == 0:
                axis.set_title(_panel_title(method, severity, corruption), fontsize=9)
            true_index = int(batch["label"][row])
            target_index = int(targets[row])
            predicted_index = int(probabilities[row].argmax())
            predicted_confidence = float(probabilities[row, predicted_index])
            target_name = (
                str(class_names[target_index])
                if target_index < len(class_names)
                else str(target_index)
            )
            predicted_name = (
                str(class_names[predicted_index])
                if predicted_index < len(class_names)
                else str(predicted_index)
            )
            axis.text(
                0.02,
                0.02,
                f"p(target={target_name})={confidences[row]:.2f}\n"
                f"pred={predicted_name} (p={predicted_confidence:.2f})",
                transform=axis.transAxes,
                fontsize=7.2,
                color="#111111",
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.8},
            )
            if column == 0:
                true_name = (
                    str(class_names[true_index])
                    if true_index < len(class_names)
                    else str(true_index)
                )
                axis.set_ylabel(f"sample {row + 1}\ntrue={true_name}", fontsize=8.5)
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle(
        "Fixed class-stratified test images · clean-predicted target fixed across severity",
        fontsize=12,
        y=0.997,
    )
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def run_attribution(
    evaluator: CheckpointEvaluator,
    corruption: str,
    severities: list[int],
    sample_count: int,
    output_dir: Path,
    stability_sample_count: int = 0,
    batch_size: int = 8,
    method: str = "gradcam",
    figure_severities: Sequence[int] = DEFAULT_FIGURE_SEVERITIES,
    registry_path: Path | None = None,
    require_clean_revision: bool = False,
    config_path: Path | None = None,
) -> pd.DataFrame:
    """Run one auditable attribution suite and return per-image tidy metrics."""

    if sample_count < 4 or sample_count > 6:
        raise ValueError("qualitative sample_count must be between 4 and 6")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path.cwd().resolve()
    evaluation_revision = (
        require_clean_git_revision(root, allowed_untracked=(output_dir,))
        if require_clean_revision
        else git_revision(root)
    )
    ordered_severities, ordered_figure_severities = _validate_severities(
        severities, figure_severities
    )
    methods = _normalize_methods(method)
    if require_clean_revision:
        if registry_path is None:
            raise ValueError("a pinned Stage-1 registry is required for a traceable Stage-4 run")
        expected_quantitative = list(range(6))
        if (
            methods != CAM_METHODS
            or ordered_severities != expected_quantitative
            or ordered_figure_severities != list(DEFAULT_FIGURE_SEVERITIES)
        ):
            raise ValueError(
                "traceable Stage-4 runs require both CAM methods, quantitative severities "
                "0–5, and qualitative severities 0/2/4"
            )
    registry = _registry_row(registry_path, evaluator.checkpoint_path, evaluator, root)
    set_seed(evaluator.seed, deterministic=True)
    datasets = _datasets(evaluator, corruption, ordered_severities)
    stability_indices = _stratified_indices(evaluator, stability_sample_count)
    figure_indices = _stratified_indices(evaluator, sample_count)
    if not stability_indices:
        raise ValueError("test split contains no samples")
    rows: list[dict[str, Any]] = []
    for cam_method in methods:
        rows.extend(
            _method_rows(
                evaluator,
                datasets,
                corruption,
                ordered_severities,
                stability_indices,
                batch_size,
                cam_method,
                set(figure_indices),
                root,
            )
        )
    frame = (
        pd.DataFrame(rows)
        .sort_values(["method", "severity", "path"], kind="stable")
        .reset_index(drop=True)
    )
    summary = _summary(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "attribution_stability.csv"
    summary_path = output_dir / "attribution_stability_summary.csv"
    samples_path = output_dir / "attribution_samples.csv"
    grid_path = output_dir / "attribution_grid.png"
    stability_path = output_dir / "attribution_stability.png"
    selected_frame = (
        evaluator.manifest.loc[evaluator.manifest["split"] == "test"]
        .reset_index(drop=True)
        .iloc[figure_indices]
    )
    sample_frame = pd.DataFrame(
        {
            "manifest_test_index": figure_indices,
            "path": selected_frame["path"].astype(str).tolist(),
            "label": selected_frame["label"].astype(int).tolist(),
        }
    )
    _atomic_csv(metrics_path, frame)
    _atomic_csv(summary_path, summary)
    _atomic_csv(samples_path, sample_frame)
    _plot_qualitative_grid(
        evaluator,
        datasets,
        methods,
        ordered_figure_severities,
        figure_indices,
        corruption,
        grid_path,
    )
    _plot_stability(
        summary,
        stability_path,
        corruption=corruption,
        dataset=str(evaluator.config["dataset"]["name"]),
        model=str(evaluator.config["model"]["backbone"]),
    )
    if config_path is None:
        recorded_config_path = evaluator.config.get("_config_path")
        config_path = (
            Path(recorded_config_path)
            if recorded_config_path and Path(recorded_config_path).is_file()
            else None
        )
    output_paths = (metrics_path, summary_path, samples_path, grid_path, stability_path)
    provenance = {
        "analysis": "Stage 4 attribution",
        "checkpoint": {
            "path": _portable_identifier(
                evaluator.checkpoint_path, root, evaluator.checkpoint_sha256
            ),
            "sha256": evaluator.checkpoint_sha256,
            "run_id": evaluator.run_id,
            "seed": evaluator.seed,
        },
        "config": {
            "path": _portable_identifier(config_path, root) if config_path else None,
            "sha256": _sha256(config_path) if config_path else None,
        },
        "dataset": {
            "name": evaluator.config["dataset"]["name"],
            "manifest_sha256": evaluator.manifest_digest,
            "test_samples_evaluated": len(stability_indices),
        },
        "registry": registry,
        "corruption_protocol_sha256": evaluator.corruption_protocol_sha256,
        "evaluation_git_revision": evaluation_revision,
        "device": str(evaluator.device),
        "torch_version": torch.__version__,
        "protocol": {
            "corruption": corruption,
            "methods": list(methods),
            "quantitative_severities": ordered_severities,
            "qualitative_severities": ordered_figure_severities,
            "qualitative_sample_count": len(figure_indices),
            "quantitative_sample_count_per_method": len(stability_indices),
            "top_saliency_fraction": 0.2,
            "target_definition": "clean predicted class, fixed for the same image at every severity",
            "heatmap_resolution": "native final spatial feature map",
            "qualitative_selection": "deterministic class-interleaved manifest order",
            "qualitative_paths_sha256": _sha256(samples_path),
        },
        "outputs": {
            path.name: {"path": _portable_identifier(path, root), "sha256": _sha256(path)}
            for path in output_paths
        },
    }
    _atomic_json(output_dir / "attribution_provenance.json", provenance)
    return frame


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        help="approved checkpoint registry; scientific runs should always provide this",
    )
    parser.add_argument("--corruption", default="defocus_blur")
    parser.add_argument("--severities", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--figure-severities", nargs="+", type=int, default=list(DEFAULT_FIGURE_SEVERITIES)
    )
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--stability-samples",
        type=int,
        default=0,
        help="quantitative sample count; 0 evaluates the complete test split",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--method", choices=["gradcam", "gradcam++", "both"], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/attribution"))
    parser.add_argument(
        "--require-clean-revision",
        action="store_true",
        help="reject tracked/unrelated untracked changes and record an exact committed SHA",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    args = _parse_args(argv)
    evaluator = CheckpointEvaluator(
        args.checkpoint, config=load_config(args.config), device=args.device
    )
    frame = run_attribution(
        evaluator,
        args.corruption,
        args.severities,
        args.samples,
        args.output_dir,
        stability_sample_count=args.stability_samples,
        batch_size=args.batch_size,
        method=args.method,
        figure_severities=args.figure_severities,
        registry_path=args.registry,
        require_clean_revision=args.require_clean_revision,
        config_path=args.config,
    )
    valid = int(frame["heatmap_valid"].sum())
    print(
        f"wrote {len(frame):,} attribution rows for {frame['path'].nunique():,} samples; "
        f"{valid:,} rows have finite stability metrics"
    )
    return frame


if __name__ == "__main__":
    main()
