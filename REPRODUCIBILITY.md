# Reproducibility record

I used the released-checkpoint workflow for two verification runs on Apple MPS. I did not use these runs as a source for scientific values. The committed `results/metrics.csv` and `results/thresholds.csv` remain the scientific records.

## New clone run

I created a new full-history clone and a new Python 3.11.15 environment from `requirements.txt`. I copied no data, checkpoints, logits, or caches into it.

I ran the following command after environment setup.

```bash
MPLBACKEND=Agg CALIBRATION_DEVICE=mps CALIBRATION_NUM_WORKERS=0 bash run.sh --eval-only
```

The run downloaded and verified SMIDS and HuSHeM. It downloaded the 1.14 GB checkpoint release and matched the pinned archive SHA-256. It installed the exact 16-model registry and completed in 42 minutes and 29 seconds.

## Exact revision run

I updated that clone to revision `5089548ddc6b6d4bff2650a0abe82479752f9b9f` and ran the same command. This run completed in 26 minutes and 55 seconds with existing verified downloads and inference caches.

The verifier reported the following results.

It found 45,540 metric rows with no duplicate keys and 17 evaluation detail files. The maximum absolute metric difference was `0.0`. The threshold output had 16 rows, 10 comparisons with both crossings, 0 earlier reliability crossings, and 6 signals that never crossed.

It also found 49 calibration-only fit records and 15 figures with 51 verified hash entries. It confirmed 3,000 unique SMIDS paths with exact train, validation, calibration, and test roles. It confirmed 216 unique HuSHeM paths across five folds with one outer test appearance per image. All artifact paths were repository-relative, and the scientific grid contained no Kromp, pilot, or synthetic-demo rows.

I did not measure T4 runtime. I therefore make no T4 timing claim.
