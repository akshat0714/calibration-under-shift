"""Generate paired Grad-CAM panels and quantitative attribution stability."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.attribution.gradcam import GradCAM, overlay_heatmaps
from src.attribution.stability import pair_stability
from src.data.transforms import preprocessing_from_model_config
from src.evaluate import CheckpointEvaluator
from src.utils import load_config


def _stratified_indices(evaluator: CheckpointEvaluator, count: int) -> list[int]:
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


def _batch_from_dataset(dataset, indices: list[int], device: torch.device):
    samples = [dataset[index] for index in indices]
    return {
        "image": torch.stack([sample["image"] for sample in samples]).to(device),
        "label": torch.tensor([sample["label"] for sample in samples], dtype=torch.long),
        "path": [sample["path"] for sample in samples],
    }


def _display_images(normalized: torch.Tensor, model_config: dict) -> np.ndarray:
    preprocessing = preprocessing_from_model_config(model_config)
    mean = torch.tensor(preprocessing["mean"], device=normalized.device)[None, :, None, None]
    std = torch.tensor(preprocessing["std"], device=normalized.device)[None, :, None, None]
    return (normalized * std + mean).clamp(0, 1).detach().cpu().numpy()


def run_attribution(
    evaluator: CheckpointEvaluator,
    corruption: str,
    severities: list[int],
    sample_count: int,
    output_dir: Path,
    stability_sample_count: int = 0,
    batch_size: int = 8,
    method: str = "gradcam",
) -> pd.DataFrame:
    if 0 not in severities:
        raise ValueError("severities must include clean severity 0")
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample_count and batch_size must be positive")
    ordered_severities = sorted(set(severities))
    datasets = {
        severity: evaluator.loader(
            "test", "clean" if severity == 0 else corruption, severity
        ).dataset
        for severity in ordered_severities
    }
    generator = GradCAM(evaluator.model, method=method)
    stability_indices = _stratified_indices(evaluator, stability_sample_count)
    rows = []
    for offset in range(0, len(stability_indices), batch_size):
        indices = stability_indices[offset : offset + batch_size]
        clean_batch = _batch_from_dataset(datasets[0], indices, evaluator.device)
        clean_result = generator.explain(clean_batch["image"])
        fixed_targets = clean_result.target_classes
        results = {0: (clean_batch, clean_result)}
        for severity in ordered_severities[1:]:
            batch = _batch_from_dataset(datasets[severity], indices, evaluator.device)
            if batch["path"] != clean_batch["path"]:
                raise AssertionError("attribution batches are not sample-aligned")
            results[severity] = (batch, generator.explain(batch["image"], fixed_targets))
        for severity, (batch, result) in results.items():
            for index, path in enumerate(batch["path"]):
                stability = pair_stability(
                    clean_result.heatmaps[index], result.heatmaps[index], top_fraction=0.2
                )
                rows.append(
                    {
                        "path": path,
                        "severity": severity,
                        "true_label": int(batch["label"][index]),
                        "clean_target": int(fixed_targets[index]),
                        "predicted_label": int(result.logits[index].argmax()),
                        "target_confidence": float(result.confidences[index]),
                        **stability,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "attribution_stability.csv", index=False)

    figure_indices = _stratified_indices(evaluator, sample_count)
    clean_batch = _batch_from_dataset(datasets[0], figure_indices, evaluator.device)
    clean_result = generator.explain(clean_batch["image"])
    fixed_targets = clean_result.target_classes
    figure_results = {0: (clean_batch, clean_result)}
    for severity in ordered_severities[1:]:
        batch = _batch_from_dataset(datasets[severity], figure_indices, evaluator.device)
        figure_results[severity] = (batch, generator.explain(batch["image"], fixed_targets))
    figure, axes = plt.subplots(
        len(clean_batch["path"]),
        len(ordered_severities),
        figsize=(3.0 * len(ordered_severities), 2.7 * len(clean_batch["path"])),
        squeeze=False,
    )
    for column, severity in enumerate(ordered_severities):
        batch, result = figure_results[severity]
        overlays = overlay_heatmaps(
            _display_images(batch["image"], evaluator.config["model"]), result.heatmaps
        )
        for row, overlay in enumerate(overlays):
            axes[row, column].imshow(overlay)
            condition = "clean" if severity == 0 else f"{corruption} s{severity}"
            axes[row, column].set_title(
                f"{condition}\np(target)={float(result.confidences[row]):.2f}, "
                f"pred={int(result.logits[row].argmax())}",
                fontsize=9,
            )
            axes[row, column].axis("off")
    figure.tight_layout()
    panel_name = "gradcam_grid.png" if generator.method == "gradcam" else "gradcamplusplus_grid.png"
    figure.savefig(output_dir / panel_name, dpi=150, bbox_inches="tight")
    plt.close(figure)

    summary = frame.groupby("severity").agg(
        spearman_mean=("spearman", "mean"),
        spearman_std=("spearman", "std"),
        iou_mean=("top_percent_iou", "mean"),
        iou_std=("top_percent_iou", "std"),
    )
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.errorbar(
        summary.index,
        summary["spearman_mean"],
        yerr=summary["spearman_std"].fillna(0),
        marker="o",
        label="Spearman rank correlation",
    )
    axis.errorbar(
        summary.index,
        summary["iou_mean"],
        yerr=summary["iou_std"].fillna(0),
        marker="s",
        label="Top-20% saliency IoU",
    )
    axis.set(xlabel="Corruption severity", ylabel="Attribution stability", ylim=(-0.05, 1.05))
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_dir / "attribution_stability.png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--corruption", default="defocus_blur")
    parser.add_argument("--severities", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--stability-samples",
        type=int,
        default=0,
        help="quantitative sample count; 0 evaluates the complete test split",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--method", choices=["gradcam", "gradcam++"], default="gradcam")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
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
    )
    print(f"wrote attribution outputs for {frame['path'].nunique()} samples")


if __name__ == "__main__":
    main()
