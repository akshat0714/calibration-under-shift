"""Run the complete released-checkpoint evaluation into an isolated output tree."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from experiments.analyze import threshold_analysis, validate_complete_grid, write_summary
from experiments.checkpoint3 import generate_checkpoint3_artifacts
from experiments.generate_final_figures import generate_final_figures
from experiments.run_stage2_matrix import require_clean_git_revision, run_stage2_matrix
from experiments.verify_reproduction import verify_reproduction, verify_split_integrity
from scripts.generate_corruption_grid import generate_corruption_grid
from scripts.release_checkpoints import REGISTRY, prepare_release_matrix
from src.evaluate import resolve_device

REPRODUCTION_ROOT = Path("results/reproduction")
REFERENCE_METRICS = Path("results/metrics.csv")
REFERENCE_THRESHOLDS = Path("results/thresholds.csv")
PROTOCOL = Path("configs/analysis_protocol.yaml")
ATTRIBUTION_CONFIG = Path("configs/smids_resnet50.yaml")


class ReleaseReproductionError(RuntimeError):
    """Raised when the one-command release workflow cannot run safely."""


def _run(command: list[str], *, root: Path) -> None:
    subprocess.run(command, cwd=root, check=True)


def _safe_reset_reproduction(root: Path) -> Path:
    destination = (root / REPRODUCTION_ROOT).resolve()
    expected = (root.resolve() / "results/reproduction").resolve()
    if destination != expected or destination in {root.resolve(), root.resolve() / "results"}:
        raise ReleaseReproductionError(f"refusing unsafe reproduction cleanup: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _resolved_accelerator(requested: str, *, allow_cpu: bool) -> str:
    device = resolve_device(requested)
    if device.type == "cpu" and not allow_cpu:
        raise ReleaseReproductionError(
            "release evaluation requires CUDA or Apple MPS; refusing silent CPU execution. "
            "Set CALIBRATION_ALLOW_CPU=1 only for an explicitly accepted slow diagnostic run."
        )
    return str(device)


def _hardware(device: str) -> dict[str, Any]:
    output: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        output["accelerator"] = torch.cuda.get_device_name(torch.device(device))
        output["cuda"] = torch.version.cuda
    elif device == "mps":
        output["accelerator"] = "Apple Metal Performance Shaders"
    else:
        output["accelerator"] = "CPU (explicit override)"
    return output


@contextmanager
def _timed(timings: dict[str, float], label: str) -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        timings[label] = round(time.monotonic() - started, 3)


def _select_attribution_checkpoint(root: Path) -> Path:
    registry = pd.read_csv(root / REGISTRY)
    selected = registry.loc[
        (registry["dataset"] == "smids")
        & (registry["model"] == "resnet50")
        & (pd.to_numeric(registry["seed"], errors="coerce") == 2025)
        & (registry["fold"].isna() | registry["fold"].astype(str).str.strip().eq(""))
    ]
    if len(selected) != 1:
        raise ReleaseReproductionError(
            "could not select exactly one SMIDS ResNet50 seed-2025 attribution checkpoint"
        )
    checkpoint = Path(str(selected.iloc[0]["checkpoint"]))
    resolved = (root / checkpoint).resolve()
    if not resolved.is_file():
        raise ReleaseReproductionError(f"attribution checkpoint is missing: {checkpoint}")
    return checkpoint


def reproduce_release(
    *,
    root: Path | None = None,
    device: str = "auto",
    num_workers: int | None = None,
    allow_cpu: bool = False,
    download: bool = True,
    python: str = sys.executable,
) -> dict[str, Any]:
    """Execute Stages 2–5 from the pinned release and verify the frozen result."""

    root = (root or Path.cwd()).resolve()
    if not (root / ".git").exists():
        raise ReleaseReproductionError(f"not a full git clone: {root}")
    if num_workers is not None and num_workers < 0:
        raise ReleaseReproductionError("CALIBRATION_NUM_WORKERS must be non-negative")
    resolved_device = _resolved_accelerator(device, allow_cpu=allow_cpu)
    revision = require_clean_git_revision(root, allowed_untracked=(REPRODUCTION_ROOT,))
    reference_metrics = root / REFERENCE_METRICS
    reference_thresholds = root / REFERENCE_THRESHOLDS
    if not reference_metrics.is_file() or not reference_thresholds.is_file():
        raise ReleaseReproductionError("committed reference metrics and thresholds are required")

    timings: dict[str, float] = {}
    overall_started = time.monotonic()
    _safe_reset_reproduction(root)
    relative = REPRODUCTION_ROOT

    if download:
        with _timed(timings, "download_and_extract_seconds"):
            _run(["bash", "scripts/download_data.sh", "smids"], root=root)
            _run(["bash", "scripts/download_data.sh", "hushem"], root=root)
    with _timed(timings, "split_preflight_seconds"):
        verify_split_integrity(root)

    with _timed(timings, "release_verification_seconds"):
        checkpoints = prepare_release_matrix(root)
        if len(checkpoints) != 16:
            raise ReleaseReproductionError("pinned release did not provide the 16-member matrix")
        attribution_checkpoint = _select_attribution_checkpoint(root)

    with _timed(timings, "stage2_seconds"):
        run_stage2_matrix(
            registry_path=REGISTRY,
            output=relative / "metrics.csv",
            parts_dir=relative / "stage2_parts",
            details_dir=relative / "evaluation_details",
            device=resolved_device,
            num_workers=num_workers,
            root=root,
            python=python,
        )

    with _timed(timings, "stage3_seconds"):
        metrics = pd.read_csv(root / relative / "metrics.csv", low_memory=False)
        with (root / PROTOCOL).open(encoding="utf-8") as handle:
            protocol = yaml.safe_load(handle)
        validate_complete_grid(metrics, protocol)
        thresholds = threshold_analysis(metrics, protocol)
        thresholds_path = root / relative / "thresholds.csv"
        thresholds.to_csv(thresholds_path, index=False)
        write_summary(thresholds, root / relative / "thresholds.md")
        generate_checkpoint3_artifacts(
            relative / "metrics.csv",
            relative / "thresholds.csv",
            PROTOCOL,
            relative / "checkpoint3",
            None,
        )

    with _timed(timings, "corruption_grid_seconds"):
        generate_corruption_grid(
            ATTRIBUTION_CONFIG,
            relative / "figures/corruption_grid.png",
            relative / "figures/corruption_grid.json",
        )

    with _timed(timings, "stage4_attribution_seconds"):
        _run(
            [
                python,
                "-m",
                "experiments.run_attribution",
                "--config",
                str(ATTRIBUTION_CONFIG),
                "--checkpoint",
                str(attribution_checkpoint),
                "--registry",
                str(REGISTRY),
                "--corruption",
                "defocus_blur",
                "--severities",
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "--figure-severities",
                "0",
                "2",
                "4",
                "--samples",
                "6",
                "--stability-samples",
                "0",
                "--method",
                "both",
                "--device",
                resolved_device,
                "--output-dir",
                str(relative / "attribution"),
                "--require-clean-revision",
            ],
            root=root,
        )

    with _timed(timings, "final_figures_seconds"):
        generate_final_figures(
            metrics_path=relative / "metrics.csv",
            thresholds_path=relative / "thresholds.csv",
            protocol_path=PROTOCOL,
            details_dir=relative / "evaluation_details",
            output_dir=relative / "figures",
            data_dir=relative / "figure_data",
            attribution_dir=relative / "attribution",
        )

    timings["total_before_verification_seconds"] = round(time.monotonic() - overall_started, 3)
    with _timed(timings, "verification_seconds"):
        result = verify_reproduction(
            root=root,
            reproduction_root=root / relative,
            reference_metrics=reference_metrics,
            reference_thresholds=reference_thresholds,
            timings={"hardware": _hardware(resolved_device), **timings},
        )
    timings["total_seconds"] = round(time.monotonic() - overall_started, 3)
    result["timings"] = {"hardware": _hardware(resolved_device), **timings}
    verification_path = root / relative / "verification.json"
    temporary = verification_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(verification_path)
    final_revision = require_clean_git_revision(root, allowed_untracked=(REPRODUCTION_ROOT,))
    if final_revision != revision:
        raise ReleaseReproductionError(
            f"repository revision changed during reproduction: {revision} -> {final_revision}"
        )
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("CALIBRATION_DEVICE", "auto"))
    worker_default = os.environ.get("CALIBRATION_NUM_WORKERS")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None if worker_default is None else int(worker_default),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        default=os.environ.get("CALIBRATION_ALLOW_CPU") == "1",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--skip-download", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    result = reproduce_release(
        device=args.device,
        num_workers=args.num_workers,
        allow_cpu=args.allow_cpu,
        download=not args.skip_download,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
