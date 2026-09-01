#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/download_data.sh [all|smids|hushem|kromp] [--download-only]"
}

dataset="${1:-all}"
mode="${2:-extract}"
if [[ "$dataset" == "-h" || "$dataset" == "--help" ]]; then
  usage
  exit 0
fi
case "$dataset" in
  all|smids|hushem|kromp) ;;
  *) usage >&2; exit 2 ;;
esac
if [[ "$mode" != "extract" && "$mode" != "--download-only" ]]; then
  usage >&2
  exit 2
fi

raw_root="data/raw"
mkdir -p "$raw_root"

download() {
  local url="$1"
  local destination="$2"
  local partial="${destination}.part"
  if [[ ! -f "$destination" ]]; then
    echo "Downloading $(basename "$destination")"
    curl --fail --location --retry 4 --retry-all-errors --continue-at - \
      --output "$partial" "$url"
    mv "$partial" "$destination"
  fi
}

verify_sha256() {
  local expected="$1"
  local path="$2"
  local actual
  if command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "$path" | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "$path" | awk '{print $1}')"
  else
    echo "Need shasum or sha256sum to verify $path" >&2
    exit 1
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch for $path" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
}

verify_md5() {
  local expected="$1"
  local path="$2"
  local actual
  if command -v md5 >/dev/null 2>&1; then
    actual="$(md5 -q "$path")"
  elif command -v md5sum >/dev/null 2>&1; then
    actual="$(md5sum "$path" | awk '{print $1}')"
  else
    echo "Need md5 or md5sum to verify $path" >&2
    exit 1
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "MD5 mismatch for $path" >&2
    echo "expected: $expected" >&2
    echo "actual:   $actual" >&2
    exit 1
  fi
}

download_smids() {
  local target_dir="$raw_root/smids"
  local archive="$target_dir/SMIDS.zip"
  mkdir -p "$target_dir"
  download \
    "https://data.mendeley.com/public-files/datasets/6xvdhc9fyb/files/f8b3773f-d9c0-4f60-bfaf-5085378b4a1f/file_downloaded" \
    "$archive"
  verify_sha256 "f46868e3a957414da55793973f75394ad2469fed48d3368e7f7f8a3aa59780a3" "$archive"
  if [[ "$mode" == "extract" && ! -f "$target_dir/.extracted" ]]; then
    unzip -q "$archive" -d "$target_dir/files"
    touch "$target_dir/.extracted"
  fi
}

download_hushem() {
  local target_dir="$raw_root/hushem"
  local archive="$target_dir/HuSHeM.rar"
  mkdir -p "$target_dir"
  download \
    "https://data.mendeley.com/public-files/datasets/tt3yj2pf38/files/cb128460-9945-43a1-a88b-7233276d5fde/file_downloaded" \
    "$archive"
  verify_sha256 "aec7c19643a298386cae2399fb225b6b382a149b78d3a6d9239e842cce95de00" "$archive"
  if [[ "$mode" == "extract" && ! -f "$target_dir/.extracted" ]]; then
    mkdir -p "$target_dir/files"
    if command -v unar >/dev/null 2>&1; then
      unar -quiet -output-directory "$target_dir/files" "$archive"
    elif command -v unrar >/dev/null 2>&1; then
      unrar x -idq -o+ "$archive" "$target_dir/files/"
    elif command -v 7z >/dev/null 2>&1; then
      7z x -y "-o$target_dir/files" "$archive" >/dev/null
    elif command -v bsdtar >/dev/null 2>&1 && bsdtar -tf "$archive" >/dev/null 2>&1; then
      bsdtar -xf "$archive" -C "$target_dir/files"
    else
      echo "HuSHeM is a RAR archive. Install unar, unrar, 7-Zip, or a RAR-capable bsdtar, then rerun." >&2
      exit 1
    fi
    touch "$target_dir/.extracted"
  fi
}

download_kromp() {
  local target_dir="$raw_root/kromp"
  local archive="$target_dir/Kromp_blastocyst_v3.zip"
  mkdir -p "$target_dir"
  download "https://ndownloader.figshare.com/files/39348899" "$archive"
  verify_md5 "d19532b4b6bc4792b44738b8930d9ad2" "$archive"
  if [[ "$mode" == "extract" && ! -f "$target_dir/.extracted" ]]; then
    unzip -q "$archive" -d "$target_dir/files"
    touch "$target_dir/.extracted"
  fi
}

case "$dataset" in
  smids) download_smids ;;
  hushem) download_hushem ;;
  kromp) download_kromp ;;
  all)
    download_smids
    download_hushem
    download_kromp
    ;;
esac

echo "Download/checksum step complete for: $dataset"
