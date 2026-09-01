"""Create a tiny synthetic cell-like dataset for end-to-end engineering checks.

The generated images are not evidence for the scientific hypothesis and are never
mixed with public-data results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from src.data.splits import stratified_manifest


def _sample(label: int, rng: np.random.Generator, size: int = 96) -> Image.Image:
    background = rng.normal(225, 8, size=(size, size, 3))
    background[..., 0] += np.linspace(-8, 8, size)[None, :]
    image = Image.fromarray(np.clip(background, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image, mode="RGBA")
    cx = int(size / 2 + rng.integers(-8, 9))
    cy = int(size / 2 + rng.integers(-8, 9))
    if label == 0:  # compact round head
        rx, ry = int(rng.integers(17, 22)), int(rng.integers(20, 25))
        color = (80, 105, 155, 210)
    elif label == 1:  # tapered morphology
        rx, ry = int(rng.integers(11, 15)), int(rng.integers(25, 30))
        color = (120, 80, 145, 210)
    else:  # pyriform / irregular proxy
        rx, ry = int(rng.integers(18, 24)), int(rng.integers(20, 27))
        color = (75, 135, 120, 210)
    draw.ellipse(
        (cx - rx, cy - ry, cx + rx, cy + ry), fill=color, outline=(35, 45, 75, 230), width=2
    )
    if label == 2:
        draw.ellipse((cx - rx - 7, cy - 5, cx - rx + 8, cy + 12), fill=(75, 135, 120, 190))
    draw.ellipse(
        (cx - rx // 3, cy - ry // 3, cx + rx // 3, cy + ry // 3), fill=(180, 195, 220, 100)
    )
    return image.filter(ImageFilter.GaussianBlur(radius=0.45))


def make_demo(root: Path, manifest_path: Path, samples_per_class: int, seed: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for label in range(3):
        class_dir = root / f"class_{label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(samples_per_class):
            relative = Path(f"class_{label}") / f"sample_{index:03d}.png"
            _sample(label, rng).save(root / relative)
            rows.append(
                {"path": str(relative), "label": label, "sample_id": f"{label}-{index:03d}"}
            )
    metadata = pd.DataFrame(rows)
    manifest = stratified_manifest(metadata, seed=seed)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote {len(manifest)} synthetic rows to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/demo"))
    parser.add_argument("--manifest", type=Path, default=Path("data/splits/demo.csv"))
    parser.add_argument("--samples-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    make_demo(args.root, args.manifest, args.samples_per_class, args.seed)


if __name__ == "__main__":
    main()
