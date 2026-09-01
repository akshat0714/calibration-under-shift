from __future__ import annotations

import pandas as pd

from experiments.train_matrix import _write_registry


def test_registry_write_is_atomic_and_replaces_existing_file(tmp_path):
    path = tmp_path / "registry.csv"
    path.write_text("old\nvalue\n", encoding="utf-8")
    frame = pd.DataFrame({"seed": [2025], "checkpoint": ["checkpoint.pt"]})

    _write_registry(frame, path)

    assert pd.read_csv(path).to_dict("records") == frame.to_dict("records")
    assert not path.with_suffix(".csv.tmp").exists()
