"""Resume or retrain the exact 16-member matrix on one CUDA GPU."""

from __future__ import annotations

import argparse
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from experiments.evaluate_clean_matrix import (
    CONFIG_PATHS,
    EXPECTED_CHECKPOINT_MEMBERS,
    CheckpointMember,
    clean_acceptance_failures,
    evaluate_clean_matrix,
    validate_checkpoint_registry,
)
from experiments.evaluate_matrix import require_clean_git_revision
from experiments.train_matrix import train_matrix
from experiments.verify_reproduction import verify_split_integrity
from src.utils import load_config

DEFAULT_REGISTRY = Path("results/checkpoint_registry-full-retrain.csv")
GROUP_ORDER = (
    ("smids", "resnet50"),
    ("smids", "xception"),
    ("smids", "mobilenet_v3_large"),
    ("hushem", "resnet50"),
)


def _identity(row: pd.Series) -> CheckpointMember:
    fold_value: Any = row.get("fold", "")
    fold = None if pd.isna(fold_value) or str(fold_value).strip() == "" else int(float(fold_value))
    return CheckpointMember(
        str(row["dataset"]).strip().lower(),
        str(row["model"]).strip().lower().replace("-", "_"),
        int(float(row["seed"])),
        fold,
    )


def _atomic_registry(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_partial_registry(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "seed",
                "fold",
                "run_id",
                "checkpoint",
                "best_val_macro_f1",
            ]
        )
    frame = pd.read_csv(path)
    required = {
        "dataset",
        "model",
        "seed",
        "fold",
        "run_id",
        "checkpoint",
        "best_val_macro_f1",
    }
    if not required.issubset(frame):
        raise ValueError(f"partial retrain registry lacks columns: {sorted(required - set(frame))}")
    identities = [_identity(row) for _, row in frame.iterrows()]
    duplicates = {member for member in identities if identities.count(member) > 1}
    if duplicates:
        raise ValueError(
            "partial retrain registry contains duplicate logical members: "
            f"{sorted(member.label() for member in duplicates)}"
        )
    unexpected = set(identities) - EXPECTED_CHECKPOINT_MEMBERS
    if unexpected:
        raise ValueError(
            "partial retrain registry contains unexpected members: "
            f"{sorted(member.label() for member in unexpected)}"
        )
    return frame


def full_retrain(
    *,
    root: Path,
    registry_path: Path = DEFAULT_REGISTRY,
    download: bool = True,
    num_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Resume missing members, then enforce the clean-test acceptance criteria."""

    root = root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("--full-retrain requires a CUDA GPU. CPU/MPS training is refused")
    require_clean_git_revision(root)
    if download:
        subprocess.run(["bash", "scripts/download_data.sh", "smids"], cwd=root, check=True)
        subprocess.run(["bash", "scripts/download_data.sh", "hushem"], cwd=root, check=True)
    verify_split_integrity(root)

    registry = root / registry_path if not registry_path.is_absolute() else registry_path
    frame = _load_partial_registry(registry)
    by_member = {_identity(row): index for index, row in frame.iterrows()}
    for identity in GROUP_ORDER:
        config = load_config(root / CONFIG_PATHS[identity])
        config["training"]["device"] = "cuda"
        if num_workers is not None:
            if num_workers < 0:
                raise ValueError("num_workers must be non-negative")
            config["training"]["num_workers"] = num_workers
            config["evaluation"]["num_workers"] = num_workers
        members = sorted(
            member
            for member in EXPECTED_CHECKPOINT_MEMBERS
            if (member.dataset, member.model) == identity
        )
        for member in members:
            existing_index = by_member.get(member)
            if existing_index is not None:
                checkpoint = Path(str(frame.loc[existing_index, "checkpoint"]))
                resolved = checkpoint if checkpoint.is_absolute() else root / checkpoint
                if resolved.is_file():
                    print(f"resume: keeping {member.label()} at {checkpoint}")
                    continue
                frame = frame.drop(index=existing_index).reset_index(drop=True)
                by_member = {_identity(row): index for index, row in frame.iterrows()}
            trained = train_matrix(
                config,
                seeds=[member.seed],
                folds=None if member.fold is None else [member.fold],
            )
            if len(trained) != 1 or _identity(trained.iloc[0]) != member:
                raise RuntimeError(
                    f"training returned the wrong logical member for {member.label()}"
                )
            frame = pd.concat([frame, trained], ignore_index=True)
            _atomic_registry(frame, registry)
            by_member = {_identity(row): index for index, row in frame.iterrows()}

    exact = validate_checkpoint_registry(frame).drop(columns="_member")
    _atomic_registry(exact, registry)
    clean_path = registry.with_name(f"{registry.stem}-clean-metrics.csv")
    summary_path = registry.with_name(f"{registry.stem}-clean-summary.csv")
    clean, summary = evaluate_clean_matrix(
        registry,
        clean_path,
        summary_path,
        device="cuda",
        require_cuda=True,
        cache_dir=root / "results/cache",
    )
    failures = clean_acceptance_failures(clean, summary)
    if failures:
        raise RuntimeError("Clean-test acceptance criteria failed.\n- " + "\n- ".join(failures))
    return exact, clean, summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    registry, clean, summary = full_retrain(
        root=Path.cwd(),
        registry_path=args.registry,
        download=not args.skip_download,
        num_workers=args.num_workers,
    )
    print(
        f"full retrain complete: members={len(registry)}, clean rows={len(clean)}, "
        f"summary rows={len(summary)}"
    )


if __name__ == "__main__":
    main()
