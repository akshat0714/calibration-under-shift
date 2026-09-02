"""Fetch, verify, and select the immutable release checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd
import zstandard

from src.utils import load_config

REPOSITORY = "akshat0714/calibration-under-shift"
RELEASE_TAG = "stage1-gcp-handoff-v1"
TRAINING_SHA = "65cee06fbf23294b358f35569d2cf2b32b46cbab"
ARCHIVE_NAME = f"calibration-stage1-{TRAINING_SHA}.tar.zst"
ARCHIVE_SHA256 = "06510f8813eb2f67b11268ebcb2761fbb616c777bf37364ca8bda2b485a0105f"
ARCHIVE_SIZE = 1_141_701_245
ARCHIVE_URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{ARCHIVE_NAME}"
INTERNAL_MANIFEST = Path("results/stage1_SHA256SUMS")
INTERNAL_MANIFEST_SHA256 = "5fee4cc901584313cd4559cb808bdb72f19878df4f0ca74f879a4f6c51fb854b"
REGISTRY = Path("results/checkpoint_registry-stage1.csv")
CANONICAL_CONFIG_PATHS = (
    Path("configs/smids_resnet50.yaml"),
    Path("configs/smids_xception.yaml"),
    Path("configs/smids_mobilenetv3.yaml"),
    Path("configs/hushem_resnet50.yaml"),
)


class ReleaseCheckpointError(RuntimeError):
    """Raised when released checkpoint material fails an integrity check."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(
    archive: Path,
    *,
    expected_sha256: str = ARCHIVE_SHA256,
    expected_size: int = ARCHIVE_SIZE,
) -> None:
    if not archive.is_file():
        raise ReleaseCheckpointError(f"release archive does not exist: {archive}")
    observed_size = archive.stat().st_size
    if observed_size != expected_size:
        raise ReleaseCheckpointError(
            f"release archive size mismatch: expected {expected_size}, observed {observed_size}"
        )
    observed_sha256 = sha256_file(archive)
    if observed_sha256 != expected_sha256:
        raise ReleaseCheckpointError(
            "release archive SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )


def _run_curl(url: str, partial: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise ReleaseCheckpointError("curl is required to download released checkpoints")
    command = [
        curl,
        "--fail",
        "--location",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-connrefused",
        "--continue-at",
        "-",
        "--output",
        str(partial),
        url,
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ReleaseCheckpointError(
            f"checkpoint download failed with curl exit code {completed.returncode}"
        )


def fetch_archive(
    archive: Path,
    *,
    url: str = ARCHIVE_URL,
    expected_sha256: str = ARCHIVE_SHA256,
    expected_size: int = ARCHIVE_SIZE,
) -> Path:
    """Fetch the fixed release asset, resuming only a dedicated partial file."""

    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        verify_archive(archive, expected_sha256=expected_sha256, expected_size=expected_size)
        print(f"using verified release archive: {archive}", file=sys.stderr)
        return archive

    partial = archive.with_name(f"{archive.name}.part")
    if partial.exists() and partial.stat().st_size >= expected_size:
        try:
            verify_archive(partial, expected_sha256=expected_sha256, expected_size=expected_size)
        except ReleaseCheckpointError:
            partial.unlink()
        else:
            partial.replace(archive)
            return archive

    print(f"downloading {url}", file=sys.stderr)
    _run_curl(url, partial)
    try:
        verify_archive(partial, expected_sha256=expected_sha256, expected_size=expected_size)
    except ReleaseCheckpointError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(archive)
    print(f"verified release archive SHA-256: {expected_sha256}", file=sys.stderr)
    return archive


def _safe_member_path(destination: Path, member_name: str) -> Path:
    member = PurePosixPath(member_name)
    if (
        member.is_absolute()
        or not member.parts
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise ReleaseCheckpointError(f"unsafe archive member path: {member_name!r}")
    if member.parts[0] != "results":
        raise ReleaseCheckpointError(
            f"release archive member is outside the results tree: {member_name!r}"
        )
    target = destination.joinpath(*member.parts)
    resolved_destination = destination.resolve()
    resolved_target = target.resolve()
    if os.path.commonpath((resolved_destination, resolved_target)) != str(resolved_destination):
        raise ReleaseCheckpointError(f"archive member escapes extraction root: {member_name!r}")
    return target


def safe_extract_archive(archive: Path, destination: Path) -> None:
    """Stream regular files from a zstd tar without delegating path handling to tar."""

    destination.mkdir(parents=True, exist_ok=True)
    with (
        archive.open("rb") as compressed,
        zstandard.ZstdDecompressor().stream_reader(compressed) as stream,
        tarfile.open(fileobj=stream, mode="r|") as bundle,
    ):
        for member in bundle:
            target = _safe_member_path(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ReleaseCheckpointError(
                    f"release archive contains a non-regular member: {member.name!r}"
                )
            source = bundle.extractfile(member)
            if source is None:
                raise ReleaseCheckpointError(
                    f"could not read release archive member: {member.name!r}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def verify_internal_manifest(root: Path) -> set[Path]:
    """Verify every archive entry anchored by the packaged SHA-256 manifest."""

    manifest = root / INTERNAL_MANIFEST
    if not manifest.is_file():
        raise ReleaseCheckpointError(f"internal checksum manifest is missing: {manifest}")
    verified: set[Path] = set()
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            expected, raw_path = raw_line.split(maxsplit=1)
        except ValueError as error:
            raise ReleaseCheckpointError(
                f"malformed internal checksum line {line_number}"
            ) from error
        raw_path = raw_path.removeprefix("*")
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ReleaseCheckpointError(f"invalid SHA-256 on internal checksum line {line_number}")
        target = _safe_member_path(root, raw_path)
        if not target.is_file():
            raise ReleaseCheckpointError(f"released artifact is missing: {raw_path}")
        observed = sha256_file(target)
        if observed != expected:
            raise ReleaseCheckpointError(
                f"released artifact SHA-256 mismatch for {raw_path}: "
                f"expected {expected}, observed {observed}"
            )
        verified.add(Path(raw_path))
    if not verified:
        raise ReleaseCheckpointError("internal checksum manifest contains no files")
    return verified


def verify_pinned_internal_manifest(root: Path) -> set[Path]:
    """Verify that the local manifest is the one carried by the pinned archive."""

    manifest = root / INTERNAL_MANIFEST
    if not manifest.is_file():
        raise ReleaseCheckpointError(f"internal checksum manifest is missing: {manifest}")
    observed = sha256_file(manifest)
    if observed != INTERNAL_MANIFEST_SHA256:
        raise ReleaseCheckpointError(
            "internal checksum manifest SHA-256 mismatch: "
            f"expected {INTERNAL_MANIFEST_SHA256}, observed {observed}"
        )
    return verify_internal_manifest(root)


def install_archive(archive: Path, root: Path) -> set[Path]:
    """Verify an extracted tree before atomically replacing its individual files."""

    cache_dir = root / "results/releases"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="stage1-extract-", dir=cache_dir) as temporary:
        extraction_root = Path(temporary)
        safe_extract_archive(archive, extraction_root)
        verified = verify_pinned_internal_manifest(extraction_root)
        extracted_files = sorted(path for path in extraction_root.rglob("*") if path.is_file())
        for source in extracted_files:
            relative = source.relative_to(extraction_root)
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
    final_verified = verify_pinned_internal_manifest(root)
    if final_verified != verified:
        raise ReleaseCheckpointError("installed release manifest differs from verified extraction")
    return final_verified


def _normalize_integer(value: Any, label: str) -> int:
    if pd.isna(value):
        raise ReleaseCheckpointError(f"{label} must be an integer, not empty")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ReleaseCheckpointError(f"{label} must be an integer: {value!r}") from error
    if not number.is_integer():
        raise ReleaseCheckpointError(f"{label} must be an integer: {value!r}")
    return int(number)


def _normalize_fold(value: Any) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return _normalize_integer(value, "registry fold")


def select_registry_checkpoints(config: dict[str, Any], registry: Path, root: Path) -> list[Path]:
    frame = pd.read_csv(registry)
    required = {"dataset", "model", "seed", "fold", "checkpoint"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ReleaseCheckpointError(
            f"release registry is missing columns: {', '.join(missing_columns)}"
        )

    dataset = str(config["dataset"]["name"])
    model = str(config["model"]["backbone"])
    selected = frame[(frame["dataset"] == dataset) & (frame["model"] == model)].copy()
    if selected.empty:
        raise ReleaseCheckpointError(
            f"release contains no checkpoints for dataset={dataset}, model={model}"
        )

    seeds = [int(value) for value in config["training"].get("seeds", [config["training"]["seed"]])]
    folds = (
        [int(value) for value in config["dataset"]["folds"]]
        if "folds" in config["dataset"]
        else [int(config["dataset"]["fold"]) if config["dataset"].get("fold") is not None else None]
    )
    expected_order = [(seed, fold) for fold in folds for seed in seeds]
    expected = set(expected_order)

    identities: list[tuple[int, int | None]] = []
    for row in selected.itertuples(index=False):
        identities.append(
            (_normalize_integer(row.seed, "registry seed"), _normalize_fold(row.fold))
        )
    observed = set(identities)
    if len(observed) != len(identities):
        raise ReleaseCheckpointError(
            f"release registry contains duplicate identities for dataset={dataset}, model={model}"
        )
    if observed != expected:
        missing = sorted(expected - observed, key=str)
        extra = sorted(observed - expected, key=str)
        raise ReleaseCheckpointError(
            f"release checkpoint matrix is not exact. Missing entries are {missing}. "
            f"Extra entries are {extra}"
        )

    rows_by_identity = {
        identity: row
        for identity, row in zip(identities, selected.itertuples(index=False), strict=True)
    }
    checkpoint_root = (root / "results/checkpoints").resolve()
    checkpoints: list[Path] = []
    for identity in expected_order:
        raw_path = Path(str(rows_by_identity[identity].checkpoint))
        if raw_path.is_absolute():
            raise ReleaseCheckpointError(f"release checkpoint path must be relative: {raw_path}")
        checkpoint = (root / raw_path).resolve()
        if os.path.commonpath((checkpoint_root, checkpoint)) != str(checkpoint_root):
            raise ReleaseCheckpointError(
                f"release checkpoint is outside results/checkpoints: {raw_path}"
            )
        if not checkpoint.is_file():
            raise ReleaseCheckpointError(f"released checkpoint is missing: {raw_path}")
        checkpoints.append(checkpoint.relative_to(root.resolve()))
    return checkpoints


def ensure_release(root: Path) -> set[Path]:
    """Fetch, install if needed, and verify the pinned checkpoint release once."""

    root = root.resolve()
    archive = root / "results/releases" / ARCHIVE_NAME
    fetch_archive(archive)
    try:
        verified = verify_pinned_internal_manifest(root)
    except ReleaseCheckpointError:
        verified = install_archive(archive, root)
    if REGISTRY not in verified:
        raise ReleaseCheckpointError(f"release registry is not anchored by {INTERNAL_MANIFEST}")
    return verified


def prepare_release_checkpoints(config_path: Path, root: Path) -> list[Path]:
    root = root.resolve()
    verified = ensure_release(root)
    checkpoints = select_registry_checkpoints(load_config(config_path), root / REGISTRY, root)
    unverified = sorted(set(checkpoints) - verified)
    if unverified:
        raise ReleaseCheckpointError(
            f"release registry selected checkpoints outside {INTERNAL_MANIFEST}: {unverified}"
        )
    return checkpoints


def prepare_release_matrix(root: Path) -> list[Path]:
    """Return the exact canonical 16-member matrix after one release verification."""

    root = root.resolve()
    verified = ensure_release(root)
    checkpoints: list[Path] = []
    for relative_config in CANONICAL_CONFIG_PATHS:
        config_path = root / relative_config
        if not config_path.is_file():
            raise ReleaseCheckpointError(f"canonical config is missing: {relative_config}")
        checkpoints.extend(
            select_registry_checkpoints(load_config(config_path), root / REGISTRY, root)
        )
    if len(checkpoints) != 16 or len(set(checkpoints)) != 16:
        raise ReleaseCheckpointError(
            "release matrix must contain exactly 16 unique checkpoint paths. "
            f"observed={len(checkpoints)}, unique={len(set(checkpoints))}"
        )
    unverified = sorted(set(checkpoints) - verified)
    if unverified:
        raise ReleaseCheckpointError(
            f"release registry selected checkpoints outside {INTERNAL_MANIFEST}: {unverified}"
        )
    return checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--all", action="store_true", help="select the canonical 16-member matrix")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        checkpoints = (
            prepare_release_matrix(args.root)
            if args.all
            else prepare_release_checkpoints(args.config, args.root)
        )
    except ReleaseCheckpointError as error:
        raise SystemExit(str(error)) from error
    for checkpoint in checkpoints:
        print(checkpoint)


if __name__ == "__main__":
    main()
