"""Convert public release layouts into audited metadata and split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.audit import audit_metadata, save_sample_grid
from src.data.splits import (
    cross_validation_manifest,
    grouped_stratified_manifest,
    kromp_high_quality_label,
    stratified_manifest,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def discover_decodable_images(root: Path, *, strict: bool = True) -> list[Path]:
    """Inventory supported image candidates using content-aware decoding.

    Successful dataset preparation is strict: an unreadable candidate is an audit
    failure, not a file that can silently disappear from the release inventory.
    """

    images: list[Path] = []
    unreadable: list[Path] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as handle:
                handle.verify()
            images.append(path)
        except OSError:
            unreadable.append(path)
    if strict and unreadable:
        examples = [str(path.relative_to(root)) for path in unreadable[:5]]
        raise ValueError(
            f"found {len(unreadable)} unreadable image candidates; examples: {examples}"
        )
    return images


def _infer_class(path: Path, mapping: dict[str, int]) -> int:
    components = []
    for raw_component in path.parts:
        component = raw_component.lower().replace("-", " ").replace("_", " ")
        component = re.sub(r"\s+", " ", component).strip()
        component = re.sub(r"^\d+\s*", "", component).strip()
        components.append(component)
    matches = {
        value
        for component in components
        for key, value in mapping.items()
        if component == key or component.startswith(f"{key} ")
    }
    if len(matches) != 1:
        raise ValueError(f"could not infer exactly one class from path: {path}")
    return matches.pop()


def prepare_smids(raw_root: Path, output_root: Path, seed: int) -> tuple[pd.DataFrame, dict]:
    mapping = {"non sperm": 2, "nonsperm": 2, "abnormal sperm": 1, "normal sperm": 0}
    paths = discover_decodable_images(raw_root)
    metadata = pd.DataFrame(
        {
            "path": [str(path.relative_to(raw_root)) for path in paths],
            "label": [_infer_class(path.relative_to(raw_root), mapping) for path in paths],
        }
    )
    expected = {0: 1021, 1: 1005, 2: 974}
    observed = metadata["label"].value_counts().sort_index().to_dict()
    if observed != expected:
        raise ValueError(
            f"SMIDS class counts differ from release metadata: {observed} != {expected}"
        )
    manifest = stratified_manifest(metadata, seed=seed)
    report = audit_metadata(metadata, raw_root)
    _write_outputs("smids", metadata, manifest, report, raw_root, output_root)
    return manifest, report


def prepare_hushem(raw_root: Path, output_root: Path, seed: int) -> tuple[pd.DataFrame, dict]:
    mapping = {"normal": 0, "tapered": 1, "pyriform": 2, "amorphous": 3}
    paths = discover_decodable_images(raw_root)
    metadata = pd.DataFrame(
        {
            "path": [str(path.relative_to(raw_root)) for path in paths],
            "label": [_infer_class(path.relative_to(raw_root), mapping) for path in paths],
        }
    )
    expected = {0: 54, 1: 53, 2: 57, 3: 52}
    observed = metadata["label"].value_counts().sort_index().to_dict()
    if observed != expected:
        raise ValueError(f"HuSHeM class counts differ from publication: {observed} != {expected}")
    manifest = cross_validation_manifest(metadata, folds=5, seed=seed)
    report = audit_metadata(metadata, raw_root)
    _write_outputs("hushem", metadata, manifest, report, raw_root, output_root)
    return manifest, report


def prepare_kromp(
    raw_root: Path,
    output_root: Path,
    annotations_path: Path,
    patient_map_path: Path,
    filename_col: str,
    expansion_col: str,
    icm_col: str,
    te_col: str,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    """Prepare Kromp only with a genuine image-to-patient map.

    The Figshare v3 release has no patient identifier. Filename prefixes produce
    851 values, inconsistent with the paper's 837 patients, so this function never
    treats prefixes as patients. Obtain a map from the dataset authors first.
    """

    annotations = pd.read_csv(annotations_path)
    required = {filename_col, expansion_col, icm_col, te_col}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"annotation CSV is missing columns: {sorted(missing)}")
    if annotations[filename_col].isna().any():
        raise ValueError("annotation CSV contains missing filenames")
    duplicate_rows = annotations[annotations.duplicated(filename_col, keep=False)]
    conflicts = duplicate_rows.groupby(filename_col)[[expansion_col, icm_col, te_col]].nunique()
    conflicts = conflicts[(conflicts > 1).any(axis=1)]
    if not conflicts.empty:
        examples = conflicts.index[:5].tolist()
        raise ValueError(
            "Kromp annotations contain conflicting duplicate labels; resolve explicitly. "
            f"Examples: {examples}"
        )
    annotations = annotations.drop_duplicates(filename_col, keep="first").copy()
    annotations["label"] = kromp_high_quality_label(
        annotations,
        expansion_col=expansion_col,
        icm_col=icm_col,
        te_col=te_col,
        expansion_offset=1,
    )
    patient_map = pd.read_csv(patient_map_path)
    if set(patient_map.columns) < {filename_col, "patient_id"}:
        raise ValueError(f"patient map must contain {filename_col!r} and 'patient_id'")
    if (
        patient_map[filename_col].isna().any()
        or patient_map[filename_col].duplicated().any()
        or patient_map["patient_id"].isna().any()
    ):
        raise ValueError("patient map must have one non-missing patient_id per filename")
    merged = annotations.merge(
        patient_map[[filename_col, "patient_id"]],
        on=filename_col,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    unmapped = merged.loc[merged["_merge"] != "both", filename_col].tolist()
    if unmapped:
        raise ValueError(f"patient map is missing {len(unmapped)} annotated images: {unmapped[:5]}")
    merged = merged.drop(columns="_merge")
    image_candidates: dict[str, list[Path]] = {}
    for path in discover_decodable_images(raw_root):
        image_candidates.setdefault(path.name, []).append(path)
    ambiguous = {name: paths for name, paths in image_candidates.items() if len(paths) > 1}
    if ambiguous:
        raise ValueError(f"duplicate image basenames under raw root: {list(ambiguous)[:5]}")
    image_lookup = {name: paths[0] for name, paths in image_candidates.items()}
    missing_images = sorted(set(merged[filename_col]) - set(image_lookup))
    if missing_images:
        raise ValueError(f"annotations refer to {len(missing_images)} missing images")
    metadata = merged.assign(
        path=[str(image_lookup[name].relative_to(raw_root)) for name in merged[filename_col]]
    )[["path", "patient_id", "label"]]
    metadata["sha256"] = [
        hashlib.sha256((raw_root / path).read_bytes()).hexdigest() for path in metadata["path"]
    ]
    duplicate_patient_counts = metadata.groupby("sha256")["patient_id"].nunique()
    cross_patient_duplicates = duplicate_patient_counts[duplicate_patient_counts > 1]
    if not cross_patient_duplicates.empty:
        raise ValueError(
            "exact duplicate Kromp images map to different patients; resolve before splitting"
        )
    metadata = metadata.drop(columns="sha256")
    balance = metadata["label"].value_counts(normalize=True)
    if balance.min() < 0.25:
        raise ValueError(
            f"binary task falls outside prespecified 25--75% balance: {balance.to_dict()}"
        )
    manifest = grouped_stratified_manifest(metadata, seed=seed)
    report = audit_metadata(metadata, raw_root)
    _write_outputs("kromp", metadata, manifest, report, raw_root, output_root)
    return manifest, report


def _write_outputs(
    dataset: str,
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    report: dict,
    raw_root: Path,
    output_root: Path,
) -> None:
    metadata_dir = output_root / "metadata"
    splits_dir = output_root / "splits"
    figures_dir = Path("results/figures")
    audit_dir = Path("results/audits")
    for directory in (metadata_dir, splits_dir, figures_dir, audit_dir):
        directory.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_dir / f"{dataset}.csv", index=False)
    manifest.to_csv(splits_dir / f"{dataset}.csv", index=False)
    (audit_dir / f"{dataset}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    save_sample_grid(metadata, figures_dir / f"{dataset}_samples.png", raw_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("smids", "hushem", "kromp"))
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--patient-map", type=Path)
    parser.add_argument("--filename-col", default="Image")
    parser.add_argument("--expansion-col", default="EXP")
    parser.add_argument("--icm-col", default="ICM")
    parser.add_argument("--te-col", default="TE")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    default_root = Path("data/raw") / args.dataset / "files"
    raw_root = args.raw_root or default_root
    if args.dataset == "smids":
        prepare_smids(raw_root, args.output_root, args.seed)
    elif args.dataset == "hushem":
        prepare_hushem(raw_root, args.output_root, args.seed)
    else:
        if args.annotations is None or args.patient_map is None:
            raise SystemExit(
                "Kromp preparation requires --annotations and an author-verified --patient-map; "
                "the public release does not contain patient IDs."
            )
        prepare_kromp(
            raw_root,
            args.output_root,
            args.annotations,
            args.patient_map,
            args.filename_col,
            args.expansion_col,
            args.icm_col,
            args.te_col,
            args.seed,
        )
    print(f"prepared {args.dataset} under {args.output_root}")


if __name__ == "__main__":
    main()
