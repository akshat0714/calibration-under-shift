"""Audit the public Kromp v3 archive without inventing a patient-level split."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.data.prepare import discover_decodable_images


def _annotation_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]
    if "Image" not in frame:
        raise ValueError(f"annotation file has no Image column: {path}")
    return frame


def audit_kromp_release(root: Path) -> tuple[dict, pd.DataFrame]:
    """Return a machine-readable image/annotation audit and source manifest."""

    images_root = root / "Images"
    paths = discover_decodable_images(images_root)
    silver = _annotation_frame(root / "Gardner_train_silver.csv")
    gold = _annotation_frame(root / "Gardner_test_gold_onlyGardnerScores.csv")
    duplicate_silver = silver.loc[silver["Image"].duplicated(keep=False)].copy()
    grade_columns = [column for column in silver if column != "Image"]
    conflicting = (
        duplicate_silver.groupby("Image")[grade_columns].nunique(dropna=False)
        if not duplicate_silver.empty
        else pd.DataFrame()
    )
    conflict_names = (
        conflicting.index[(conflicting > 1).any(axis=1)].astype(str).tolist()
        if not conflicting.empty
        else []
    )

    image_names = {path.name for path in paths}
    silver_names = set(silver["Image"].astype(str))
    gold_names = set(gold["Image"].astype(str))
    annotated = silver_names | gold_names
    hashes: dict[str, list[str]] = {}
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    rows: list[dict] = []
    for path in paths:
        relative = str(path.relative_to(root))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(relative)
        with Image.open(path) as image:
            image.load()
            dimensions[f"{image.width}x{image.height}"] += 1
            modes[image.mode] += 1
        source = (
            "gold"
            if path.name in gold_names
            else "silver"
            if path.name in silver_names
            else "unlabeled"
        )
        rows.append({"path": relative, "filename": path.name, "annotation_source": source})

    prefixes = {path.stem.rsplit("_", 1)[0] for path in paths}
    report = {
        "images": len(paths),
        "dimensions": dict(dimensions.most_common()),
        "color_modes": dict(modes.most_common()),
        "exact_duplicate_groups": [items for items in hashes.values() if len(items) > 1],
        "filename_prefix_groups_not_assumed_patients": len(prefixes),
        "silver_rows": len(silver),
        "silver_unique_images": len(silver_names),
        "silver_conflicting_images": conflict_names,
        "gold_rows": len(gold),
        "gold_unique_images": len(gold_names),
        "annotation_overlap": sorted(silver_names & gold_names),
        "images_without_gardner_annotation": sorted(image_names - annotated),
        "annotations_without_image": sorted(annotated - image_names),
    }
    return report, pd.DataFrame(rows).sort_values("path").reset_index(drop=True)


def save_kromp_sample_grid(manifest: pd.DataFrame, root: Path, output: Path) -> None:
    selected = []
    for source in ("silver", "gold", "unlabeled"):
        selected.extend(
            manifest.loc[manifest["annotation_source"] == source]
            .sort_values("path")
            .head(4)
            .to_dict("records")
        )
    figure, axes = plt.subplots(3, 4, figsize=(9.6, 7.2), squeeze=False)
    for axis, row in zip(axes.flat, selected, strict=False):
        with Image.open(root / row["path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(f"{row['annotation_source']} · {row['filename']}", fontsize=8)
        axis.axis("off")
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/kromp/files"))
    parser.add_argument("--audit", type=Path, default=Path("results/audits/kromp.json"))
    parser.add_argument("--metadata", type=Path, default=Path("data/metadata/kromp_release.csv"))
    parser.add_argument("--grid", type=Path, default=Path("results/figures/kromp_samples.png"))
    args = parser.parse_args()
    report, manifest = audit_kromp_release(args.root)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest.to_csv(args.metadata, index=False)
    save_kromp_sample_grid(manifest, args.root, args.grid)
    print(
        f"audited {report['images']} images. "
        f"{len(report['images_without_gardner_annotation'])} unlabeled. "
        f"{len(report['silver_conflicting_images'])} conflicting silver labels"
    )


if __name__ == "__main__":
    main()
