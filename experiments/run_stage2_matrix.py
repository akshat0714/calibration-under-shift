"""Run and strictly merge the four canonical Stage-2 evaluation groups."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.analyze import validate_complete_grid
from experiments.evaluate_clean_matrix import CONFIG_PATHS, validate_stage1_registry
from experiments.run_grid import KEY_COLUMNS
from src.utils import load_config

REGISTRY_PATH = Path("results/checkpoint_registry-stage1.csv")
PROTOCOL_PATH = Path("configs/analysis_protocol.yaml")
OUTPUT_PATH = Path("results/metrics.csv")
PARTS_DIR = Path("results/stage2_parts")
DETAILS_DIR = Path("results/evaluation_details")

CANONICAL_GROUP_ORDER = (
    ("smids", "resnet50"),
    ("smids", "xception"),
    ("smids", "mobilenet_v3_large"),
    ("hushem", "resnet50"),
)
EXPECTED_GROUP_ROWS = {
    ("smids", "resnet50"): 14_565,
    ("smids", "xception"): 8_295,
    ("smids", "mobilenet_v3_large"): 8_295,
    ("hushem", "resnet50"): 14_385,
}
EXPECTED_TOTAL_ROWS = 45_540
EXPECTED_DETAIL_JSONS = 17
ENSEMBLE_DETAIL_NAME = "ensemble-smids-resnet50.json"


@dataclass(frozen=True)
class CanonicalGroup:
    dataset: str
    model: str
    config: Path
    checkpoints: tuple[Path, ...]
    output: Path
    expected_rows: int

    @property
    def identity(self) -> tuple[str, str]:
        return self.dataset, self.model


def _absolute(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _git_command(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(["git", *arguments], cwd=root, check=False, capture_output=True)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {message or 'unknown error'}")
    return completed


def _is_allowed_untracked(root: Path, raw_path: bytes, allowed: tuple[Path, ...]) -> bool:
    candidate = (root / os.fsdecode(raw_path)).resolve()
    for permitted in allowed:
        resolved = _absolute(root, permitted).resolve()
        if candidate == resolved or os.path.commonpath((resolved, candidate)) == str(resolved):
            return True
    return False


def require_clean_git_revision(root: Path, *, allowed_untracked: tuple[Path, ...] = ()) -> str:
    """Return a committed SHA and reject tracked dirt or unrelated untracked files."""

    revision = _git_command(root, "rev-parse", "--verify", "HEAD").stdout.decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40,64}", revision) is None:
        raise RuntimeError(f"evaluation revision is unversioned or invalid: {revision!r}")

    status = _git_command(root, "status", "--porcelain=v1", "--untracked-files=all", "-z").stdout
    dirty: list[str] = []
    for entry in status.split(b"\0"):
        if not entry:
            continue
        if (
            len(entry) >= 4
            and entry[:2] == b"??"
            and _is_allowed_untracked(root, entry[3:], allowed_untracked)
        ):
            continue
        dirty.append(os.fsdecode(entry))
    if dirty:
        preview = ", ".join(repr(item) for item in dirty[:5])
        raise RuntimeError(
            f"evaluation git revision is dirty or contains unversioned files: {preview}"
        )
    return revision


def _checkpoint_path(root: Path, raw_path: Any) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else root / path


def build_canonical_groups(
    registry_path: Path, parts_dir: Path, root: Path
) -> tuple[list[CanonicalGroup], set[str]]:
    """Validate the exact Stage-1 registry before launching any subprocess."""

    registry = validate_stage1_registry(pd.read_csv(registry_path))
    missing_checkpoints = sorted(
        str(path)
        for path in (_checkpoint_path(root, value) for value in registry["checkpoint"])
        if not path.is_file()
    )
    if missing_checkpoints:
        raise FileNotFoundError(f"checkpoint paths are missing: {missing_checkpoints}")

    expected_details = {f"{run_id}.json" for run_id in registry["run_id"]}
    if any(Path(name).name != name for name in expected_details):
        raise ValueError("registry run IDs must be safe detail filenames")
    expected_details.add(ENSEMBLE_DETAIL_NAME)
    if len(expected_details) != EXPECTED_DETAIL_JSONS:
        raise ValueError(
            f"expected {EXPECTED_DETAIL_JSONS} unique detail JSON names, "
            f"found {len(expected_details)}"
        )

    groups: list[CanonicalGroup] = []
    for index, identity in enumerate(CANONICAL_GROUP_ORDER, start=1):
        if identity not in CONFIG_PATHS:
            raise ValueError(f"canonical config is missing for {identity}")
        config = _absolute(root, CONFIG_PATHS[identity])
        if not config.is_file():
            raise FileNotFoundError(f"canonical config does not exist: {config}")
        selected = registry.loc[
            (registry["dataset"] == identity[0]) & (registry["model"] == identity[1])
        ]
        checkpoints = tuple(
            _checkpoint_path(root, value).resolve() for value in selected["checkpoint"]
        )
        slug = f"{identity[0]}-{identity[1].replace('_', '-')}"
        groups.append(
            CanonicalGroup(
                dataset=identity[0],
                model=identity[1],
                config=config.resolve(),
                checkpoints=checkpoints,
                output=parts_dir / f"{index:02d}-{slug}.csv",
                expected_rows=EXPECTED_GROUP_ROWS[identity],
            )
        )
    return groups, expected_details


def run_group(
    group: CanonicalGroup,
    *,
    details_dir: Path,
    device: str,
    num_workers: int | None,
    root: Path,
    python: str,
) -> None:
    group.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        python,
        "-m",
        "experiments.run_grid",
        "--config",
        str(group.config),
        "--checkpoints",
        *(str(path) for path in group.checkpoints),
        "--output",
        str(group.output),
        "--details-dir",
        str(details_dir),
        "--device",
        device,
    ]
    if num_workers is not None:
        command.extend(["--num-workers", str(num_workers)])
    subprocess.run(command, cwd=root, check=True)


def _read_group_part(group: CanonicalGroup, revision: str) -> pd.DataFrame:
    if not group.output.is_file():
        raise FileNotFoundError(f"Stage-2 group output was not written: {group.output}")
    part = pd.read_csv(
        group.output,
        low_memory=False,
        dtype={"evaluation_git_revision": "string"},
    )
    if len(part) != group.expected_rows:
        raise ValueError(
            f"{group.dataset}/{group.model} row count is {len(part)}, "
            f"expected {group.expected_rows}"
        )
    identities = set(part[["dataset", "model"]].itertuples(index=False, name=None))
    if identities != {group.identity}:
        raise ValueError(
            f"{group.dataset}/{group.model} part contains unexpected identities: "
            f"{sorted(identities)}"
        )
    if "evaluation_git_revision" not in part.columns:
        raise ValueError(f"{group.dataset}/{group.model} part lacks evaluation_git_revision")
    if part["evaluation_git_revision"].isna().any():
        raise ValueError(
            f"{group.dataset}/{group.model} part has missing evaluation_git_revision values"
        )
    revisions = set(part["evaluation_git_revision"].dropna().astype(str))
    if revisions != {revision}:
        raise ValueError(
            f"{group.dataset}/{group.model} evaluation revisions are {sorted(revisions)}, "
            f"expected only {revision}"
        )
    return part


def reject_duplicate_tidy_keys(metrics: pd.DataFrame) -> None:
    missing = sorted(set(KEY_COLUMNS) - set(metrics.columns))
    if missing:
        raise ValueError(f"merged metrics lack tidy key columns: {missing}")
    keys = metrics[KEY_COLUMNS].copy()
    keys["fold"] = keys["fold"].fillna("")
    duplicates = keys.duplicated(keep=False)
    if duplicates.any():
        examples = keys.loc[duplicates].drop_duplicates().head(5).to_dict("records")
        raise ValueError(f"merged metrics contain duplicate tidy keys: {examples}")


def validate_detail_jsons(details_dir: Path, expected_names: set[str], revision: str) -> None:
    observed_paths = sorted(details_dir.rglob("*.json")) if details_dir.is_dir() else []
    observed_names = {str(path.relative_to(details_dir)) for path in observed_paths}
    if len(observed_paths) != EXPECTED_DETAIL_JSONS or observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        extra = sorted(observed_names - expected_names)
        raise ValueError(
            f"detail JSON set is not exact; expected_count={EXPECTED_DETAIL_JSONS}, "
            f"observed_count={len(observed_paths)}, missing={missing}, extra={extra}"
        )
    for path in observed_paths:
        with path.open(encoding="utf-8") as handle:
            detail = json.load(handle)
        if detail.get("evaluation_git_revision") != revision:
            raise ValueError(
                f"detail JSON {path} has evaluation_git_revision="
                f"{detail.get('evaluation_git_revision')!r}, expected {revision}"
            )


def atomic_write_metrics(metrics: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        metrics.to_csv(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def run_stage2_matrix(
    *,
    registry_path: Path = REGISTRY_PATH,
    output: Path = OUTPUT_PATH,
    parts_dir: Path = PARTS_DIR,
    details_dir: Path = DETAILS_DIR,
    device: str = "auto",
    num_workers: int | None = None,
    root: Path = Path("."),
    python: str = sys.executable,
) -> pd.DataFrame:
    root = root.resolve()
    registry_path = _absolute(root, registry_path).resolve()
    output = _absolute(root, output).resolve()
    parts_dir = _absolute(root, parts_dir).resolve()
    details_dir = _absolute(root, details_dir).resolve()
    protocol_path = _absolute(root, PROTOCOL_PATH).resolve()
    allowed_generated = (output, parts_dir, details_dir)

    if num_workers is not None and num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    revision = require_clean_git_revision(root, allowed_untracked=allowed_generated)
    groups, expected_details = build_canonical_groups(registry_path, parts_dir, root)
    for group in groups:
        run_group(
            group,
            details_dir=details_dir,
            device=device,
            num_workers=num_workers,
            root=root,
            python=python,
        )

    parts = [_read_group_part(group, revision) for group in groups]
    metrics = pd.concat(parts, ignore_index=True)
    if len(metrics) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"merged Stage-2 row count is {len(metrics)}, expected {EXPECTED_TOTAL_ROWS}"
        )
    reject_duplicate_tidy_keys(metrics)
    if metrics["evaluation_git_revision"].isna().any():
        raise ValueError("merged metrics contain missing evaluation_git_revision values")
    revisions = set(metrics["evaluation_git_revision"].dropna().astype(str))
    if revisions != {revision}:
        raise ValueError(
            f"merged metrics must contain one clean evaluation revision; observed={revisions}"
        )
    validate_detail_jsons(details_dir, expected_details, revision)
    protocol = load_config(protocol_path)
    validate_complete_grid(metrics, protocol)
    final_revision = require_clean_git_revision(root, allowed_untracked=allowed_generated)
    if final_revision != revision:
        raise RuntimeError(
            f"evaluation revision changed during Stage 2: started {revision}, ended {final_revision}"
        )
    atomic_write_metrics(metrics, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--parts-dir", type=Path, default=PARTS_DIR)
    parser.add_argument("--details-dir", type=Path, default=DETAILS_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--num-workers",
        type=int,
        help="override evaluation data-loader workers (scheduling only)",
    )
    args = parser.parse_args()
    metrics = run_stage2_matrix(
        registry_path=args.registry,
        output=args.output,
        parts_dir=args.parts_dir,
        details_dir=args.details_dir,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(
        f"wrote {len(metrics)} validated Stage-2 metric rows to {args.output}; "
        f"detail JSONs={EXPECTED_DETAIL_JSONS}"
    )


if __name__ == "__main__":
    main()
