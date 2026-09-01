from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
from pathlib import Path

import pandas as pd
import pytest
import zstandard

from scripts.release_checkpoints import (
    INTERNAL_MANIFEST_SHA256,
    ReleaseCheckpointError,
    safe_extract_archive,
    select_registry_checkpoints,
    verify_internal_manifest,
    verify_pinned_internal_manifest,
)


def _write_zstd_tar(path: Path, members: dict[str, bytes]) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as bundle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))
    path.write_bytes(zstandard.ZstdCompressor().compress(buffer.getvalue()))


def _config() -> dict:
    return {
        "dataset": {"name": "smids"},
        "model": {"backbone": "resnet50"},
        "training": {"seed": 2025, "seeds": [2025, 2026]},
    }


def test_safe_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "malicious.tar.zst"
    _write_zstd_tar(archive, {"results/checkpoint.pt": b"safe", "../escaped": b"unsafe"})

    with pytest.raises(ReleaseCheckpointError, match="unsafe archive member path"):
        safe_extract_archive(archive, tmp_path / "extract")

    assert not (tmp_path / "escaped").exists()


def test_internal_manifest_verifies_every_released_file(tmp_path):
    checkpoint = tmp_path / "results/checkpoints/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = tmp_path / "results/stage1_SHA256SUMS"
    manifest.write_text(f"{digest}  results/checkpoints/model.pt\n", encoding="utf-8")

    assert verify_internal_manifest(tmp_path) == {Path("results/checkpoints/model.pt")}

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ReleaseCheckpointError, match="SHA-256 mismatch"):
        verify_internal_manifest(tmp_path)


def test_pinned_internal_manifest_rejects_self_consistent_local_substitute(tmp_path, monkeypatch):
    checkpoint = tmp_path / "results/checkpoints/model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"locally substituted checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = tmp_path / "results/stage1_SHA256SUMS"
    manifest.write_text(f"{digest}  results/checkpoints/model.pt\n", encoding="utf-8")

    assert verify_internal_manifest(tmp_path) == {Path("results/checkpoints/model.pt")}
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() != INTERNAL_MANIFEST_SHA256
    with pytest.raises(ReleaseCheckpointError, match="manifest SHA-256 mismatch"):
        verify_pinned_internal_manifest(tmp_path)


def test_registry_selection_requires_and_orders_exact_matrix(tmp_path):
    checkpoint_dir = tmp_path / "results/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    paths = [checkpoint_dir / "seed2026.pt", checkpoint_dir / "seed2025.pt"]
    for path in paths:
        path.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {
                "dataset": "smids",
                "model": "resnet50",
                "seed": 2026,
                "fold": "",
                "checkpoint": "results/checkpoints/seed2026.pt",
            },
            {
                "dataset": "smids",
                "model": "resnet50",
                "seed": 2025,
                "fold": "",
                "checkpoint": "results/checkpoints/seed2025.pt",
            },
        ]
    ).to_csv(registry, index=False)

    selected = select_registry_checkpoints(_config(), registry, tmp_path)

    assert selected == [
        Path("results/checkpoints/seed2025.pt"),
        Path("results/checkpoints/seed2026.pt"),
    ]

    frame = pd.read_csv(registry).iloc[:1]
    frame.to_csv(registry, index=False)
    with pytest.raises(ReleaseCheckpointError, match="matrix is not exact"):
        select_registry_checkpoints(_config(), registry, tmp_path)


def test_run_sh_release_mode_uses_canonical_matrix_without_allow_partial(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    fake_python = tmp_path / "python"
    log = tmp_path / "calls.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_PYTHON_LOG"
if [[ "$*" == "-m scripts.release_checkpoints --config configs/smids_resnet50.yaml" ]]; then
  printf '%s\\n' results/checkpoints/seed2025.pt results/checkpoints/seed2026.pt
fi
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update({"CALIBRATION_PYTHON": str(fake_python), "FAKE_PYTHON_LOG": str(log)})

    completed = subprocess.run(
        [
            "bash",
            "run.sh",
            "--eval-only",
            "configs/smids_resnet50.yaml",
            "--release-checkpoints",
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    grid_call = next(line for line in calls if line.startswith("-m experiments.run_grid"))
    analysis_call = next(line for line in calls if line.startswith("-m experiments.analyze"))
    assert "results/checkpoints/seed2025.pt results/checkpoints/seed2026.pt" in grid_call
    assert "--allow-partial" not in grid_call
    assert "--allow-partial" not in analysis_call


def test_run_sh_explicit_checkpoints_keep_partial_mode(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    fake_python = tmp_path / "python"
    log = tmp_path / "calls.log"
    fake_python.write_text(
        """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_PYTHON_LOG"
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update({"CALIBRATION_PYTHON": str(fake_python), "FAKE_PYTHON_LOG": str(log)})

    completed = subprocess.run(
        [
            "bash",
            "run.sh",
            "--eval-only",
            "configs/smids_resnet50.yaml",
            "checkpoint.pt",
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    calls = log.read_text(encoding="utf-8").splitlines()
    grid_call = next(line for line in calls if line.startswith("-m experiments.run_grid"))
    analysis_call = next(line for line in calls if line.startswith("-m experiments.analyze"))
    assert "--allow-partial" in grid_call
    assert "--allow-partial" in analysis_call
