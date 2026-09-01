# Stage 1 GCP GPU handoff

Use this fallback only when the agent cannot reach the project's GPU. Do not run
the full matrix on CPU or Apple MPS. These commands train the prespecified 16
members from the immutable `stage1-gcp-handoff-v1` tag, produce clean-test
Checkpoint 2 artifacts, and package everything needed to resume the study.

## Current access blocker

The desktop shell has no Cloud SDK or CUDA runtime. Its signed-in Google Cloud
Console is blocked by Google's two-step-verification enforcement and reports that
the free trial has ended. Enable two-step verification, wait for the block to
clear, and link an active full billing account to the target project. GPUs incur
charges while running. Open Cloud Shell from that project; do not create or
download a service-account key.

## 1. Discover and enter the existing GPU VM

Run in Cloud Shell:

```bash
set -euo pipefail

gcloud auth list --filter=status:ACTIVE
GCP_PROJECT="$(gcloud config get-value project 2>/dev/null)"
test -n "$GCP_PROJECT" && test "$GCP_PROJECT" != "(unset)"
BILLING_ENABLED="$(gcloud billing projects describe "$GCP_PROJECT" \
  --format='value(billingEnabled)')"
test "$BILLING_ENABLED" = "True"

if [[ -z "$(gcloud services list --enabled \
  --filter='config.name=compute.googleapis.com' \
  --format='value(config.name)')" ]]; then
  gcloud services enable compute.googleapis.com
fi

gcloud compute instances list \
  --filter='guestAccelerators:*' \
  --format='table(name,zone.basename(),status,machineType.basename(),guestAccelerators[].acceleratorType.basename(),guestAccelerators[].acceleratorCount)'

mapfile -t GPU_ROWS < <(
  gcloud compute instances list \
    --filter='guestAccelerators:*' \
    --format='csv[no-heading](name,zone.basename(),status)'
)
if [[ "${#GPU_ROWS[@]}" -ne 1 ]]; then
  echo "Expected exactly one existing GPU VM; select the intended VM explicitly." >&2
  exit 1
fi
IFS=',' read -r GPU_INSTANCE GPU_ZONE GPU_STATUS <<<"${GPU_ROWS[0]}"
export GCP_PROJECT GPU_INSTANCE GPU_ZONE

gcloud compute instances describe "$GPU_INSTANCE" \
  --project "$GCP_PROJECT" --zone "$GPU_ZONE" \
  --format='yaml(name,status,machineType,guestAccelerators,disks,scheduling,networkInterfaces,serviceAccounts)'

if [[ "$GPU_STATUS" != "RUNNING" ]]; then
  gcloud compute instances start "$GPU_INSTANCE" \
    --project "$GCP_PROJECT" --zone "$GPU_ZONE"
fi

GPU_NAT_IP="$(gcloud compute instances describe "$GPU_INSTANCE" \
  --project "$GCP_PROJECT" --zone "$GPU_ZONE" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
if [[ -n "$GPU_NAT_IP" ]]; then
  gcloud compute ssh "$GPU_INSTANCE" \
    --project "$GCP_PROJECT" --zone "$GPU_ZONE"
else
  gcloud compute ssh "$GPU_INSTANCE" \
    --project "$GCP_PROJECT" --zone "$GPU_ZONE" --tunnel-through-iap
fi
```

The code is single-GPU. A 16 GB T4 should fit the committed batches; an L4 gives
more headroom. Keep at least 100 GB free for dependencies, datasets, checkpoints,
logs, and later inference caches.

## 2. Pin and verify the VM environment

Run the rest on the GPU VM. The tag resolves to a clean commit descending from
the approved corruption-protocol commit `0bc67ab`; each run records that SHA.

```bash
set -euo pipefail
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl unzip p7zip-full python3-pip python3-venv tmux zstd

if [[ ! -d calibration-under-shift/.git ]]; then
  git clone https://github.com/akshat0714/calibration-under-shift.git
fi
cd calibration-under-shift
test -z "$(git status --porcelain)"
git fetch origin --tags
EXPECTED_TRAINING_SHA="65cee06fbf23294b358f35569d2cf2b32b46cbab"
TAG_TRAINING_SHA="$(git rev-list -n 1 stage1-gcp-handoff-v1)"
test "$TAG_TRAINING_SHA" = "$EXPECTED_TRAINING_SHA"
git checkout --detach "$EXPECTED_TRAINING_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_TRAINING_SHA"
test -z "$(git status --porcelain)"

if ! command -v uv >/dev/null 2>&1 && ! command -v python3.11 >/dev/null 2>&1; then
  python3 -m venv scratch/bootstrap
  scratch/bootstrap/bin/python -m pip install 'uv==0.8.15'
  export PATH="$(pwd)/scratch/bootstrap/bin:$PATH"
fi
bash run.sh --setup
mkdir -p results/logs
git rev-parse HEAD | tee results/stage1_training_sha.txt
git status --porcelain | tee results/stage1_git_status.txt
test ! -s results/stage1_git_status.txt
nvidia-smi -q | tee results/stage1_nvidia_smi.txt
.venv/bin/python - <<'PY' > results/stage1_packages.txt
from importlib.metadata import distributions

for distribution in sorted(
    distributions(), key=lambda item: (item.metadata.get("Name") or "").lower()
):
    print(f"{distribution.metadata.get('Name') or 'unknown'}=={distribution.version}")
PY
.venv/bin/python - <<'PY' | tee results/stage1_cuda.txt
import platform
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; abort rather than train on CPU/MPS")
print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"cuda_devices={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"cuda_device_{index}={torch.cuda.get_device_name(index)}")
PY

.venv/bin/python -m pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

If the CUDA assertion fails, stop. `device: auto` would otherwise fall back to
CPU, which is forbidden for this matrix.

## 3. Download and verify the fixed public data

The download script verifies the published archives before extraction. Keep the
committed manifests; do not regenerate tracked figures on the VM.

```bash
bash run.sh --download smids
bash run.sh --download hushem

printf '%s  %s\n' \
  'ce3c1a90df7674579ac807bec9026f805a82e68f96ce8474b4e35c7b1c006da2' \
  'data/splits/smids.csv' \
  'b64e07da0f8aef8950595d9591ff71736c9ff1fd47682089f2030bda68c24fbe' \
  'data/splits/hushem.csv' | sha256sum --check

.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from PIL import Image

for dataset, expected in (("smids", 3000), ("hushem", 216)):
    manifest = pd.read_csv(f"data/splits/{dataset}.csv")
    root = Path(f"data/raw/{dataset}/files")
    if len(manifest) != expected or manifest["path"].nunique() != expected:
        raise SystemExit(f"unexpected {dataset} inventory")
    for relative in manifest["path"]:
        with Image.open(root / relative) as image:
            image.verify()
    print(f"{dataset}: verified {expected} manifest images")
PY

test -z "$(git status --porcelain)"
```

## 4. Train the exact 16-member matrix

Start `tmux new -s calibration-stage1`, then run this block inside it. Each call
contains one member, so its registry row is durable immediately. The helper skips
members already registered after an SSH disconnect.

```bash
set -euo pipefail
STAGE1_REGISTRY="results/checkpoint_registry-stage1.csv"
STAGE1_LOG="results/logs/stage1-training.log"

stage1_has_member() {
  .venv/bin/python - "$STAGE1_REGISTRY" "$1" "$2" "$3" "$4" <<'PY'
from pathlib import Path
import sys
import pandas as pd

path, dataset, model, seed, fold = sys.argv[1:]
if not Path(path).is_file():
    raise SystemExit(1)
frame = pd.read_csv(path)
mask = (
    frame["dataset"].eq(dataset)
    & frame["model"].eq(model)
    & frame["seed"].astype(int).eq(int(seed))
)
if fold == "none":
    mask &= frame["fold"].isna()
else:
    mask &= frame["fold"].fillna(-1).astype(int).eq(int(fold))
selected = frame.loc[mask]
if len(selected) != 1:
    raise SystemExit(1)
row = selected.iloc[0]
required = (
    Path(str(row["checkpoint"])),
    Path("results/runs") / str(row["run_id"]) / "run.json",
    Path("results/runs") / str(row["run_id"]) / "metrics.json",
    Path("results/runs") / str(row["run_id"]) / "curves.csv",
)
raise SystemExit(0 if all(item.is_file() for item in required) else 1)
PY
}

run_member() {
  local config="$1" dataset="$2" model="$3" seed="$4" fold="$5"
  if stage1_has_member "$dataset" "$model" "$seed" "$fold"; then
    echo "already complete: $dataset $model seed=$seed fold=$fold" | tee -a "$STAGE1_LOG"
    return
  fi
  local args=(--seeds "$seed")
  if [[ "$fold" != "none" ]]; then
    args+=(--folds "$fold")
  fi
  bash run.sh --train "$config" "${args[@]}" \
    --registry "$STAGE1_REGISTRY" 2>&1 | tee -a "$STAGE1_LOG"
}

for seed in 2025 2026 2027 2028 2029; do
  run_member configs/smids_resnet50.yaml smids resnet50 "$seed" none
done
for seed in 2025 2026 2027; do
  run_member configs/smids_xception.yaml smids xception "$seed" none
done
for seed in 2025 2026 2027; do
  run_member configs/smids_mobilenetv3.yaml smids mobilenet_v3_large "$seed" none
done
for fold in 0 1 2 3 4; do
  run_member configs/hushem_resnet50.yaml hushem resnet50 2025 "$fold"
done
```

Detach with `Ctrl-b`, then `d`; reconnect with
`tmux attach -t calibration-stage1`.

## 5. Produce Checkpoint 2 clean-test artifacts

The committed clean-only evaluator validates the exact 16 logical members,
rejects duplicates or missing members, and performs only uncorrupted test inference.

```bash
bash run.sh --stage1-clean \
  results/checkpoint_registry-stage1.csv \
  --device cuda \
  --require-cuda \
  --enforce-sanity
```

It writes:

- `results/stage1_clean_metrics.csv`: tidy per-member test metrics and provenance;
- `results/stage1_clean_summary.csv`: mean, sample SD, and `n` by dataset/backbone,
  from which the Checkpoint 2 table is reported.

Stop and return artifacts without starting Stage 2 if any individual run is at or
below chance accuracy (`1/3` for SMIDS, `1/4` for HuSHeM), any SMIDS backbone has
mean macro-F1 below `0.85`, or HuSHeM ResNet50 has mean cross-fold accuracy below
`0.80`.

## 6. Package and return artifacts

Build the archive from registry-referenced checkpoints and runs, so an orphan
checkpoint left by an interrupted process cannot enter the handoff:

```bash
STAGE1_SHA="$(git rev-parse HEAD)"
RETURN_LIST="results/stage1_return_files.txt"
RETURN_ARCHIVE="calibration-stage1-${STAGE1_SHA}.tar.zst"

.venv/bin/python - "$RETURN_LIST" <<'PY'
from pathlib import Path
import sys
import pandas as pd

registry_path = Path("results/checkpoint_registry-stage1.csv")
registry = pd.read_csv(registry_path)
paths = {
    registry_path,
    Path("results/stage1_clean_metrics.csv"),
    Path("results/stage1_clean_summary.csv"),
    Path("results/stage1_training_sha.txt"),
    Path("results/stage1_git_status.txt"),
    Path("results/stage1_nvidia_smi.txt"),
    Path("results/stage1_cuda.txt"),
    Path("results/stage1_packages.txt"),
    Path("results/logs/stage1-training.log"),
}
paths.update(Path(value) for value in registry["checkpoint"])
for run_id in registry["run_id"]:
    run_dir = Path("results/runs") / str(run_id)
    paths.update(path for path in run_dir.rglob("*") if path.is_file())
missing = sorted(str(path) for path in paths if not path.is_file())
if missing:
    raise SystemExit(f"missing return artifacts: {missing}")
Path(sys.argv[1]).write_text(
    "".join(f"{path}\n" for path in sorted(paths, key=str)), encoding="utf-8"
)
PY

xargs -d '\n' sha256sum < "$RETURN_LIST" > results/stage1_SHA256SUMS
printf '%s\n' results/stage1_SHA256SUMS >> "$RETURN_LIST"
tar --zstd -cf "$RETURN_ARCHIVE" -T "$RETURN_LIST"
sha256sum "$RETURN_ARCHIVE" | tee "${RETURN_ARCHIVE}.sha256"
```

Exit the VM. Back in Cloud Shell, recover the exact training SHA and archive:

```bash
STAGE1_SHA="$(gcloud compute ssh "$GPU_INSTANCE" \
  --project "$GCP_PROJECT" --zone "$GPU_ZONE" \
  --command='cd calibration-under-shift && git rev-parse HEAD')"
SCP_FLAGS=()
if [[ -z "$GPU_NAT_IP" ]]; then
  SCP_FLAGS+=(--tunnel-through-iap)
fi
gcloud compute scp \
  "$GPU_INSTANCE:~/calibration-under-shift/calibration-stage1-${STAGE1_SHA}.tar.zst" \
  "$GPU_INSTANCE:~/calibration-under-shift/calibration-stage1-${STAGE1_SHA}.tar.zst.sha256" \
  . \
  --project "$GCP_PROJECT" \
  --zone "$GPU_ZONE" \
  "${SCP_FLAGS[@]}"
sha256sum --check "calibration-stage1-${STAGE1_SHA}.tar.zst.sha256"
```

Return that archive and checksum sidecar to this task. If an existing GCS bucket
is preferred, upload both immutable files under
`gs://<existing-bucket>/calibration-under-shift/stage1/<training-sha>/`. The
eventual public checkpoint URI and checksum manifest will be wired into
`run.sh --eval-only` only after that URI is known and verified.

After the archive is safely copied, stop the VM to stop GPU billing:

```bash
gcloud compute instances stop "$GPU_INSTANCE" \
  --project "$GCP_PROJECT" --zone "$GPU_ZONE"
```

Command references: [verify project billing](https://docs.cloud.google.com/billing/docs/how-to/verify-billing-enabled),
[copy VM artifacts with `gcloud compute scp`](https://docs.cloud.google.com/sdk/gcloud/reference/compute/scp),
and [IAP TCP forwarding](https://docs.cloud.google.com/iap/docs/using-tcp-forwarding).
