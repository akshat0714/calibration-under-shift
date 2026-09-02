"""Regenerate the fixed held-out SMIDS corruption grid and provenance sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.shifts.severity import CORRUPTION_PARAMETERS, corruption_protocol_digest
from src.utils import load_config, write_json
from src.viz.figures import plot_corruption_grid


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def generate_corruption_grid(
    config_path: str | Path = "configs/smids_resnet50.yaml",
    output_path: str | Path = "results/figures/corruption_grid.png",
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    """Render a deterministic grid from the first held-out manifest row."""

    config_source = Path(config_path)
    config = load_config(config_source)
    dataset = config["dataset"]
    manifest_path = Path(dataset["manifest"])
    manifest = pd.read_csv(manifest_path)
    required = {"path", "split"}
    missing = required - set(manifest)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if "fold" in manifest and dataset.get("fold") is not None:
        manifest = manifest.loc[manifest["fold"] == int(dataset["fold"])]
    test_rows = manifest.loc[manifest["split"] == "test"]
    if test_rows.empty:
        raise ValueError("manifest contains no held-out test row")

    sample_relative = Path(str(test_rows.iloc[0]["path"]))
    sample_path = Path(dataset["root"]) / sample_relative
    if not sample_path.is_file():
        raise FileNotFoundError(
            f"held-out sample does not exist at {sample_path}. Run the SMIDS download first"
        )
    seed = int(config.get("evaluation", {}).get("corruption_seed", 1729))
    output = Path(output_path)
    plot_corruption_grid(sample_path, output, seed=seed)

    metadata_output = Path(metadata_path) if metadata_path else output.with_suffix(".json")
    metadata = {
        "config": _portable_path(config_source),
        "manifest": _portable_path(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sample_path": str(sample_relative),
        "sample_role": "test",
        "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "corruption_seed": seed,
        "corruption_protocol_sha256": corruption_protocol_digest(),
        "parameters": json.loads(json.dumps(CORRUPTION_PARAMETERS)),
        "figure_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    write_json(metadata_output, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smids_resnet50.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/corruption_grid.png"),
    )
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    metadata = generate_corruption_grid(args.config, args.output, args.metadata)
    print(f"corruption grid: {args.output}")
    print(f"sample: {metadata['sample_path']}")
    print(f"seed: {metadata['corruption_seed']}")
    print(f"protocol: {metadata['corruption_protocol_sha256']}")


if __name__ == "__main__":
    main()
