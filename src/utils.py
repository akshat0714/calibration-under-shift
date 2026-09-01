"""Shared configuration, reproducibility, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch without silently weakening determinism."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - older supported Torch releases
            torch.use_deterministic_algorithms(True)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and attach its source path for auditability."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must contain a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def git_revision(root: str | Path = ".") -> str:
    """Return the full current Git revision plus a dirty-tree marker."""

    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode != 0
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def stable_id(payload: dict[str, Any], length: int = 10) -> str:
    """Create a deterministic short identifier from JSON-serializable metadata."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def create_run_directory(
    config: dict[str, Any], seed: int, base_dir: str | Path = "results/runs"
) -> tuple[str, Path]:
    """Create a collision-resistant run directory and persist run provenance."""

    now = datetime.now(UTC)
    identity = {"config": config, "seed": seed, "started_at": now.isoformat()}
    run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{stable_id(identity)}"
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        **identity,
        "run_id": run_id,
        "git_revision": git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    write_json(run_dir / "run.json", metadata)
    return run_id, run_dir


def write_json(path: str | Path, payload: Any) -> None:
    """Write indented, stable JSON and create parent directories."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")
