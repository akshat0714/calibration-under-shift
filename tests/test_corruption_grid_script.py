from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from scripts.generate_corruption_grid import generate_corruption_grid
from src.shifts.severity import corruption_protocol_digest


def test_fixed_grid_records_held_out_source_and_protocol(tmp_path):
    data_root = tmp_path / "images"
    data_root.mkdir()
    for name, offset in (("first.png", 0), ("second.png", 20)):
        y, x = np.mgrid[:32, :32]
        array = np.stack((x * 6 + offset, y * 6, (x + y) * 3), axis=-1)
        Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(data_root / name)

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"path": "second.png", "label": 1, "split": "test"},
            {"path": "first.png", "label": 0, "split": "test"},
        ]
    ).to_csv(manifest_path, index=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {"root": str(data_root), "manifest": str(manifest_path)},
                "evaluation": {"corruption_seed": 1729},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "corruption_grid.png"
    sidecar = tmp_path / "corruption_grid.json"

    metadata = generate_corruption_grid(config_path, output, sidecar)

    assert output.is_file() and output.stat().st_size > 1_000
    assert metadata["sample_path"] == "second.png"
    assert metadata["sample_role"] == "test"
    assert metadata["corruption_seed"] == 1729
    assert metadata["corruption_protocol_sha256"] == corruption_protocol_digest()
    assert metadata["figure_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == metadata
