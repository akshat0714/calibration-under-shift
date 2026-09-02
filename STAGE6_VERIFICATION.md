# Stage 6 verification record

This record summarizes completed reproduction gates. It is evidence about the released-checkpoint workflow, not a scientific result source; scientific numbers remain in `results/metrics.csv` and `results/thresholds.csv`.

## Cold fresh-clone drill

- Date: 2026-09-01
- Evaluated revision: `42b84828a07f44bafeaaeebefd355135d9dcbbe7`
- Environment: new full-history clone; new Python 3.11.15 virtual environment installed only from `requirements.txt`; PyTorch 2.7.1; Apple Metal Performance Shaders; zero copied data, checkpoints, logits, or caches
- Command after setup: `MPLBACKEND=Agg CALIBRATION_DEVICE=mps CALIBRATION_NUM_WORKERS=0 bash run.sh --eval-only`
- Total wall time: 2,548.873 seconds (42m29s)
- Stage 2: 2,409.592 seconds
- Dataset download/extraction: 43.993 seconds
- Checkpoint download, extraction, and release verification: 50.996 seconds
- Stage 3 tables: 7.885 seconds
- Full Stage 4 attribution: 22.610 seconds
- Final figures: 12.080 seconds
- Final verification: 0.980 seconds

The cold run downloaded and checksum-verified SMIDS and HuSHeM, extracted HuSHeM with a RAR-capable `bsdtar`, downloaded the 1.14 GB release asset, matched the pinned archive SHA-256, and installed the exact 16-member registry. No external cache or artifact was copied into the clone.

## Warm-cache drill

The same revision completed the identical command in 1,606.840 seconds (26m47s) on the same Apple-MPS host. Stage 2 accounted for 1,561.705 seconds. This run started with verified data, checkpoints, and matching inference caches already present.

## Exact acceptance results

Both runs passed the same final verifier:

- 45,540 tidy metric rows and zero duplicate keys
- 17 exact evaluation-detail JSON files
- maximum absolute metric difference from the committed reference: `0.0`
- 16 threshold rows
- 10 comparisons with both crossings, 0 early warnings, and 6 signals that never crossed
- 49 recorded calibration-only temperature/vector/APS fit assertions
- 15 final figures and 51 verified manifest hash entries
- 3,000 unique SMIDS paths with exact 2100/300/300/300 roles
- 216 unique HuSHeM paths across five folds, with every image serving as outer test data once
- repository-relative artifact paths only
- no Kromp, pilot, or synthetic-demo row in the scientific grid

## Runtime boundary

No T4 timing was run as part of these two local drills. The repository therefore reports the measured Apple-MPS times and does not claim that the cold released-checkpoint workflow finishes in under 30 minutes on a T4. Runtime is a scheduling property, not part of the frozen scientific protocol.
