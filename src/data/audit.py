"""Audit decoded image properties, labels, groups, corruption, and duplicates."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError


def audit_metadata(metadata: pd.DataFrame, data_root: str | Path = ".") -> dict[str, Any]:
    """Decode every image and return JSON-serializable audit statistics."""

    root = Path(data_root)
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    hashes: dict[str, list[str]] = {}
    corrupt: list[str] = []
    for raw_path in metadata["path"].astype(str):
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(raw_path)
            with Image.open(path) as image:
                image.load()
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
        except (OSError, UnidentifiedImageError):
            corrupt.append(raw_path)
    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    report: dict[str, Any] = {
        "images_listed": int(len(metadata)),
        "classes": {
            str(key): int(value) for key, value in metadata["label"].value_counts().items()
        },
        "dimensions": dict(dimensions.most_common()),
        "color_modes": dict(modes.most_common()),
        "corrupt_files": corrupt,
        "duplicate_groups": duplicate_groups,
    }
    if "patient_id" in metadata.columns:
        counts = metadata.groupby("patient_id").size()
        report["patients"] = int(counts.size)
        report["images_per_patient"] = {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "max": int(counts.max()),
        }
    return report


def save_sample_grid(
    metadata: pd.DataFrame,
    output: str | Path,
    data_root: str | Path = ".",
    per_class: int = 4,
) -> None:
    """Save a deterministic class-stratified sample grid."""

    classes = sorted(metadata["label"].unique())
    fig, axes = plt.subplots(
        len(classes),
        per_class,
        figsize=(2.4 * per_class, 2.4 * len(classes)),
        squeeze=False,
    )
    root = Path(data_root)
    for row_index, label in enumerate(classes):
        examples = metadata.loc[metadata["label"] == label].sort_values("path").head(per_class)
        row_axes = axes[row_index]
        for axis, (_, item) in zip(row_axes, examples.iterrows(), strict=False):
            path = Path(str(item["path"]))
            if not path.is_absolute():
                path = root / path
            with Image.open(path) as image:
                axis.imshow(image.convert("RGB"))
            axis.set_title(f"class {label}")
            axis.axis("off")
        for axis in row_axes[len(examples) :]:
            axis.axis("off")
    fig.tight_layout()
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _markdown(report: dict[str, Any], dataset: str) -> str:
    summary = f"The manifest lists {report['images_listed']} images."
    if "patients" in report:
        summary += f" It includes {report['patients']} patients."
    lines = [f"# {dataset} data audit", "", summary]
    lines.extend(["", "## Class counts", "", "| Label | Count |", "|---|---:|"])
    lines.extend(f"| {label} | {count} |" for label, count in report["classes"].items())
    lines.extend(["", "## Decode checks", ""])
    lines.append(
        f"The audit found {len(report['corrupt_files'])} corrupt or unreadable files and "
        f"{len(report['duplicate_groups'])} exact duplicate groups."
    )
    lines.append(f"Observed dimensions were `{report['dimensions']}`.")
    lines.append(f"Observed color modes were `{report['color_modes']}`.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grid", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.metadata)
    report = audit_metadata(frame, data_root=args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_markdown(report, args.dataset), encoding="utf-8")
    if args.grid:
        save_sample_grid(frame, args.grid, data_root=args.data_root)
    if report["corrupt_files"]:
        raise SystemExit(f"audit failed: {len(report['corrupt_files'])} unreadable files")


if __name__ == "__main__":
    main()
