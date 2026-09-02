"""Evaluate the exact checkpoint matrix on clean held-out test images only."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.evaluate import CheckpointEvaluator, resolve_device
from src.metrics.classification import classification_metrics
from src.utils import load_config

REQUIRED_REGISTRY_COLUMNS = {
    "dataset",
    "model",
    "seed",
    "fold",
    "run_id",
    "checkpoint",
    "best_val_macro_f1",
}

CONFIG_PATHS = {
    ("smids", "resnet50"): Path("configs/smids_resnet50.yaml"),
    ("smids", "xception"): Path("configs/smids_xception.yaml"),
    ("smids", "mobilenet_v3_large"): Path("configs/smids_mobilenetv3.yaml"),
    ("hushem", "resnet50"): Path("configs/hushem_resnet50.yaml"),
}

CHANCE_ACCURACY = {"smids": 1.0 / 3.0, "hushem": 1.0 / 4.0}
SMIDS_MIN_MEAN_MACRO_F1 = 0.85
HUSHEM_MIN_MEAN_ACCURACY = 0.80


@dataclass(frozen=True, order=True)
class CheckpointMember:
    dataset: str
    model: str
    seed: int
    fold: int | None = None

    def label(self) -> str:
        suffix = "" if self.fold is None else f"/fold={self.fold}"
        return f"{self.dataset}/{self.model}/seed={self.seed}{suffix}"


EXPECTED_CHECKPOINT_MEMBERS = frozenset(
    {
        *(CheckpointMember("smids", "resnet50", seed) for seed in range(2025, 2030)),
        *(CheckpointMember("smids", "xception", seed) for seed in range(2025, 2028)),
        *(CheckpointMember("smids", "mobilenet_v3_large", seed) for seed in range(2025, 2028)),
        *(CheckpointMember("hushem", "resnet50", 2025, fold) for fold in range(5)),
    }
)


def _integer(value: Any, field: str, *, allow_missing: bool = False) -> int | None:
    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
        if allow_missing:
            return None
        raise ValueError(f"registry {field} must be a non-missing integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"registry {field} must be an integer, received {value!r}") from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"registry {field} must be an integer, received {value!r}")
    return int(numeric)


def _member_from_row(row: pd.Series) -> CheckpointMember:
    dataset = str(row["dataset"]).strip().lower()
    model = str(row["model"]).strip().lower().replace("-", "_")
    seed = _integer(row["seed"], "seed")
    fold = _integer(row["fold"], "fold", allow_missing=True)
    assert seed is not None
    return CheckpointMember(dataset, model, seed, fold)


def validate_checkpoint_registry(registry: pd.DataFrame) -> pd.DataFrame:
    """Normalize and require exactly the prespecified 16 logical members."""

    missing_columns = REQUIRED_REGISTRY_COLUMNS - set(registry.columns)
    if missing_columns:
        raise ValueError(f"registry is missing columns: {sorted(missing_columns)}")
    if registry.empty:
        raise ValueError("registry is empty")

    normalized = registry.copy()
    members = [_member_from_row(row) for _, row in normalized.iterrows()]
    counts = Counter(members)
    duplicates = sorted(member.label() for member, count in counts.items() if count > 1)
    observed = set(members)
    missing = sorted(member.label() for member in EXPECTED_CHECKPOINT_MEMBERS - observed)
    extra = sorted(member.label() for member in observed - EXPECTED_CHECKPOINT_MEMBERS)
    if duplicates or missing or extra:
        raise ValueError(
            "Training checkpoint matrix is not exact. "
            f"duplicates={duplicates}, missing={missing}, extra={extra}"
        )

    for column in ("run_id", "checkpoint"):
        text = normalized[column].astype(str).str.strip()
        if normalized[column].isna().any() or text.eq("").any():
            raise ValueError(f"registry {column} values must be non-missing")
        normalized[column] = text
    for column, label in (("run_id", "run IDs"), ("checkpoint", "checkpoint paths")):
        duplicates = normalized.loc[normalized[column].duplicated(keep=False), column].unique()
        if len(duplicates):
            raise ValueError(
                f"registry {label} must be unique. Duplicate values are "
                f"{sorted(str(value) for value in duplicates)}"
            )

    validation_scores = pd.to_numeric(normalized["best_val_macro_f1"], errors="coerce")
    if validation_scores.isna().any() or not validation_scores.between(0.0, 1.0).all():
        raise ValueError("registry best_val_macro_f1 values must be finite and in [0, 1]")

    normalized["dataset"] = [member.dataset for member in members]
    normalized["model"] = [member.model for member in members]
    normalized["seed"] = [member.seed for member in members]
    normalized["fold"] = ["" if member.fold is None else member.fold for member in members]
    normalized["best_val_macro_f1"] = validation_scores.astype(float)
    normalized["_member"] = members
    return normalized.sort_values(["dataset", "model", "seed", "fold"], kind="stable").reset_index(
        drop=True
    )


def _require_requested_cuda(device: str, require_cuda: bool) -> str:
    resolved = resolve_device(device)
    if require_cuda and (resolved.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA execution was required, but no CUDA device is available or selected"
        )
    return str(resolved)


def summarize_clean_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return replicate mean, sample standard deviation, and count by metric."""

    required = {"dataset", "model", "metric", "value"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"clean metrics are missing columns: {sorted(missing)}")
    return (
        metrics.groupby(["dataset", "model", "metric"], as_index=False, sort=True)["value"]
        .agg(mean="mean", std="std", n="count")
        .sort_values(["dataset", "model", "metric"], kind="stable")
        .reset_index(drop=True)
    )


def clean_acceptance_failures(metrics: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    """Return prespecified clean-performance failures without hiding the outputs."""

    failures: list[str] = []
    accuracy = metrics.loc[metrics["metric"] == "accuracy"]
    for row in accuracy.itertuples(index=False):
        chance = CHANCE_ACCURACY[str(row.dataset)]
        if float(row.value) <= chance:
            fold = "" if pd.isna(row.fold) or row.fold == "" else f" fold={int(row.fold)}"
            failures.append(
                f"{row.dataset}/{row.model} seed={int(row.seed)}{fold} "
                f"accuracy={float(row.value):.6f} is at or below chance={chance:.6f}"
            )

    smids = summary.loc[(summary["dataset"] == "smids") & (summary["metric"] == "macro_f1")]
    for row in smids.itertuples(index=False):
        if float(row.mean) < SMIDS_MIN_MEAN_MACRO_F1:
            failures.append(
                f"smids/{row.model} mean macro_f1={float(row.mean):.6f} "
                f"is below {SMIDS_MIN_MEAN_MACRO_F1:.2f}"
            )

    hushem = summary.loc[
        (summary["dataset"] == "hushem")
        & (summary["model"] == "resnet50")
        & (summary["metric"] == "accuracy")
    ]
    if len(hushem) != 1:
        failures.append("HuSHeM ResNet50 accuracy summary is missing or duplicated")
    elif float(hushem.iloc[0]["mean"]) < HUSHEM_MIN_MEAN_ACCURACY:
        failures.append(
            f"hushem/resnet50 mean accuracy={float(hushem.iloc[0]['mean']):.6f} "
            f"is below {HUSHEM_MIN_MEAN_ACCURACY:.2f}"
        )
    return failures


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def evaluate_clean_matrix(
    registry_path: str | Path,
    output_path: str | Path = "results/clean_test_metrics.csv",
    summary_path: str | Path = "results/clean_test_summary.csv",
    *,
    device: str = "auto",
    require_cuda: bool = False,
    cache_dir: str | Path = "results/cache",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the matrix, run clean-test inference, and write atomic CSV outputs."""

    resolved_device = _require_requested_cuda(device, require_cuda)
    registry = validate_checkpoint_registry(pd.read_csv(registry_path))
    config_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for _, row in registry.iterrows():
        member: CheckpointMember = row["_member"]
        identity = (member.dataset, member.model)
        if identity not in config_cache:
            config_cache[identity] = load_config(CONFIG_PATHS[identity])
        config = config_cache[identity]
        checkpoint_path = Path(row["checkpoint"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
        evaluator = CheckpointEvaluator(
            checkpoint_path,
            config=config,
            device=resolved_device,
            cache_dir=cache_dir,
        )
        observed = CheckpointMember(
            str(evaluator.config["dataset"]["name"]),
            evaluator.model.backbone_name,
            evaluator.seed,
            evaluator.config["dataset"].get("fold"),
        )
        if observed != member:
            raise ValueError(
                f"checkpoint identity {observed.label()} does not match registry {member.label()}"
            )
        if evaluator.run_id != row["run_id"]:
            raise ValueError(
                f"checkpoint run_id {evaluator.run_id!r} does not match registry {row['run_id']!r}"
            )

        bundle = evaluator.infer("test", "clean", 0)
        metrics = classification_metrics(bundle.probabilities, bundle.labels)
        metadata = {
            "dataset": member.dataset,
            "model": member.model,
            "seed": member.seed,
            "fold": "" if member.fold is None else member.fold,
            "split": "test",
            "corruption": "clean",
            "severity": 0,
            "run_id": row["run_id"],
            "checkpoint": str(checkpoint_path),
            "best_val_macro_f1": float(row["best_val_macro_f1"]),
            "n_samples": len(bundle.labels),
            "manifest_sha256": evaluator.manifest_digest,
            "corruption_protocol_sha256": evaluator.corruption_protocol_sha256,
        }
        rows.extend(
            {**metadata, "metric": metric, "value": float(value)}
            for metric, value in metrics.items()
        )

    tidy = pd.DataFrame(rows)
    summary = summarize_clean_metrics(tidy)
    _write_csv(tidy, Path(output_path))
    _write_csv(summary, Path(summary_path))
    return tidy, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/clean_test_metrics.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/clean_test_summary.csv"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--enforce-clean-criteria", action="store_true")
    parser.add_argument(
        "--enforce-sanity",
        dest="enforce_clean_criteria",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("results/cache"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tidy, summary = evaluate_clean_matrix(
        args.registry,
        args.output,
        args.summary,
        device=args.device,
        require_cuda=args.require_cuda,
        cache_dir=args.cache_dir,
    )
    print(f"wrote {len(tidy)} clean-test metric rows to {args.output}")
    print(f"wrote {len(summary)} aggregate rows to {args.summary}")
    if args.enforce_clean_criteria:
        failures = clean_acceptance_failures(tidy, summary)
        if failures:
            raise SystemExit("Clean-test acceptance criteria failed.\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
