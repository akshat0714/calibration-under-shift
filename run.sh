#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run.sh --setup
  bash run.sh --demo
  bash run.sh --corruption-grid
  bash run.sh --download <smids|hushem|kromp|all>
  bash run.sh --prepare <smids|hushem>
  bash run.sh --audit-kromp
  bash run.sh --train <config.yaml> [extra train_matrix arguments]
  bash run.sh --stage1-clean <mixed-registry.csv> [extra evaluator arguments]
  bash run.sh --eval-only <config.yaml> <checkpoint.pt> [checkpoint.pt ...]
  bash run.sh --full-smids

The demo is a synthetic engineering check, not a scientific result. --full-smids
downloads/audits SMIDS, trains the configured ResNet50 seeds from scratch, evaluates
the full grid, applies preregistered thresholds, and regenerates result figures.
EOF
}

python_bin="${CALIBRATION_PYTHON:-.venv/bin/python}"

setup_environment() {
  local setup_python=".venv/bin/python"
  if command -v uv >/dev/null 2>&1; then
    [[ -x "$setup_python" ]] || uv venv --python 3.11 .venv
    uv pip install --python "$setup_python" -r requirements.txt
  else
    if ! command -v python3.11 >/dev/null 2>&1; then
      echo "Python 3.11 is required when uv is unavailable." >&2
      exit 2
    fi
    python3.11 -m venv .venv
    "$setup_python" -m pip install --upgrade pip
    "$setup_python" -m pip install -r requirements.txt
  fi
}

require_environment() {
  if [[ "$python_bin" == */* && ! -x "$python_bin" ]] || \
     [[ "$python_bin" != */* && -z "$(command -v "$python_bin" 2>/dev/null)" ]]; then
    echo "Python executable not found: $python_bin. Run: bash run.sh --setup" >&2
    exit 2
  fi
}

latest_demo_checkpoint() {
  find results/checkpoints -type f -name 'synthetic_demo-tiny_cnn-*.pt' -print \
    | sort | tail -n 1
}

case "${1:---help}" in
  --setup)
    setup_environment
    ;;
  --demo)
    require_environment
    "$python_bin" -m scripts.make_demo_data
    "$python_bin" -m src.train --config configs/demo.yaml --smoke
    demo_checkpoint="$(latest_demo_checkpoint)"
    if [[ -z "$demo_checkpoint" ]]; then
      echo "Demo checkpoint was not created" >&2
      exit 1
    fi
    "$python_bin" -m experiments.run_grid \
      --config configs/demo.yaml \
      --checkpoints "$demo_checkpoint" \
      --output results/demo_metrics.csv \
      --allow-partial \
      --device cpu
    "$python_bin" -m experiments.analyze \
      --metrics results/demo_metrics.csv \
      --output results/demo_thresholds.csv \
      --summary results/demo_thresholds.md \
      --allow-partial
    "$python_bin" -m src.viz.figures \
      --metrics results/demo_metrics.csv \
      --output-dir results/demo_figures \
      --uncertainty sd \
      --nominal-coverage 0.9
    "$python_bin" -m experiments.generate_diagnostics \
      --config configs/demo.yaml \
      --checkpoint "$demo_checkpoint" \
      --output-dir results/demo_figures \
      --severities 0 1 3 \
      --device cpu
    "$python_bin" -m experiments.run_attribution \
      --config configs/demo.yaml \
      --checkpoint "$demo_checkpoint" \
      --output-dir results/demo_figures \
      --severities 0 1 3 \
      --samples 4 \
      --device cpu
    echo "Synthetic demo completed. Do not report its metrics as scientific results."
    ;;
  --corruption-grid)
    require_environment
    "$python_bin" -m scripts.generate_corruption_grid
    ;;
  --download)
    bash scripts/download_data.sh "${2:-all}"
    ;;
  --prepare)
    require_environment
    if [[ -z "${2:-}" ]]; then usage >&2; exit 2; fi
    "$python_bin" -m src.data.prepare "$2"
    ;;
  --audit-kromp)
    require_environment
    "$python_bin" -m src.data.kromp_audit
    ;;
  --train)
    require_environment
    if [[ -z "${2:-}" ]]; then usage >&2; exit 2; fi
    config_path="$2"
    shift 2
    "$python_bin" -m experiments.train_matrix --config "$config_path" "$@"
    ;;
  --stage1-clean)
    require_environment
    if [[ -z "${2:-}" ]]; then usage >&2; exit 2; fi
    registry_path="$2"
    shift 2
    "$python_bin" -m experiments.evaluate_clean_matrix --registry "$registry_path" "$@"
    ;;
  --eval-only)
    require_environment
    if [[ "$#" -lt 3 ]]; then usage >&2; exit 2; fi
    config_path="$2"
    shift 2
    first_checkpoint="$1"
    eval_metrics="results/eval_metrics.csv"
    eval_figures="results/eval_figures"
    "$python_bin" -m experiments.run_grid \
      --config "$config_path" --checkpoints "$@" --output "$eval_metrics" --allow-partial
    "$python_bin" -m experiments.analyze \
      --metrics "$eval_metrics" \
      --output results/eval_thresholds.csv \
      --summary results/eval_thresholds.md \
      --allow-partial
    "$python_bin" -m src.viz.figures \
      --metrics "$eval_metrics" --output-dir "$eval_figures" --uncertainty sd
    "$python_bin" -m experiments.generate_diagnostics \
      --config "$config_path" --checkpoint "$first_checkpoint" --output-dir "$eval_figures"
    "$python_bin" -m experiments.run_attribution \
      --config "$config_path" --checkpoint "$first_checkpoint" --output-dir "$eval_figures"
    ;;
  --full-smids)
    require_environment
    bash scripts/download_data.sh smids
    "$python_bin" -m src.data.prepare smids
    registry_path="results/checkpoint_registry-$(date -u +%Y%m%dT%H%M%SZ)-$$.csv"
    "$python_bin" -m experiments.train_matrix \
      --config configs/smids_resnet50.yaml \
      --registry "$registry_path"
    checkpoints=()
    while IFS= read -r checkpoint; do
      checkpoints+=("$checkpoint")
    done < <(
      "$python_bin" - "$registry_path" <<'PY'
import sys
import pandas as pd
import yaml

frame = pd.read_csv(sys.argv[1])
with open("configs/smids_resnet50.yaml", encoding="utf-8") as handle:
    expected = sorted(yaml.safe_load(handle)["training"]["seeds"])
selected = frame[(frame.dataset == "smids") & (frame.model == "resnet50")]
observed = sorted(selected.seed.astype(int).tolist())
if observed != expected:
    raise SystemExit(f"checkpoint seeds {observed} do not match configured seeds {expected}")
for checkpoint in selected.sort_values("seed").checkpoint:
    print(checkpoint)
PY
    )
    if [[ "${#checkpoints[@]}" -eq 0 ]]; then
      echo "No SMIDS ResNet50 checkpoints were registered" >&2
      exit 1
    fi
    "$python_bin" -m experiments.run_grid \
      --config configs/smids_resnet50.yaml \
      --checkpoints "${checkpoints[@]}" \
      --output results/metrics.csv
    "$python_bin" -m experiments.analyze
    "$python_bin" -m src.viz.figures \
      --metrics results/metrics.csv --output-dir results/figures --uncertainty sd
    "$python_bin" -m experiments.generate_diagnostics \
      --config configs/smids_resnet50.yaml \
      --checkpoint "${checkpoints[0]}" \
      --output-dir results/figures
    "$python_bin" -m experiments.run_attribution \
      --config configs/smids_resnet50.yaml \
      --checkpoint "${checkpoints[0]}" \
      --output-dir results/figures
    ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
