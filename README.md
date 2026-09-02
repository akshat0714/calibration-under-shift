# Calibration Under Shift

[![CI](https://github.com/akshat0714/calibration-under-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/akshat0714/calibration-under-shift/actions/workflows/ci.yml)

**Question tested:** Under a prespecified simulated device-shift protocol, do ECE, predictive entropy, selective risk, or APS coverage cross their degradation thresholds before raw-softmax accuracy falls by more than five percentage points?

Under the frozen equal-corruption protocol, no reliability signal crossed before the accuracy-drop threshold in any of the 10 comparisons where both crossings were observed; six further signals never crossed, so the prespecified early-warning hypothesis was not supported at these thresholds on SMIDS and HuSHeM. Clean-fitted temperature scaling transferred inconsistently: it improved mean severity-5 ECE on all three SMIDS backbones but worsened it on HuSHeM ResNet50.

![Frozen aggregate reliability signals did not lead the accuracy drop](results/figures/f1_headline_lockstep.png)

**Figure 1. Prespecified threshold-normalized degradation.** Clean is 0 and each frozen degradation threshold is 1; X marks the first crossing. Signals and accuracy share the severity axis, no outcome region is shaded, and a missing X means no crossing. Means use equal corruption weighting within each seed/fold.

## Primary result

| Dataset / backbone | Replicates | Accuracy drop | ECE | Predictive entropy | Selective risk @80% | APS coverage |
|---|---:|---:|---:|---:|---:|---:|
| SMIDS / ResNet50 | 5 seeds | S3 | S3 | — | S3 | — |
| SMIDS / Xception | 3 seeds | S3 | S3 | S5 | S3 | — |
| SMIDS / MobileNetV3-Large | 3 seeds | S2 | S2 | — | S2 | S4 |
| HuSHeM / ResNet50 | 5 folds | S3 | — | S4 | S4 | — |

All observed reliability crossings were simultaneous with or later than the accuracy crossing. An em dash means the signal never crossed; it is not replaced with severity 5. Aggregation averages the seven corruptions within each seed/fold before summarizing replicates.

### What this means

At these frozen thresholds on these datasets, standard uncertainty dashboards did not provide an earlier aggregate degradation alarm. That bounded null is not evidence that uncertainty methods are useless: secondary analyses show useful per-sample error ranking and selective prediction even when the aggregate signal supplies no threshold-level lead time. For low-cost deployment research, uncertainty dashboards alone should not be treated as degradation alarms; paired-device validation and ongoing monitoring remain necessary.

## Study design and leakage guardrails

| Dataset | Task and matrix | Frozen split design | Limitation |
|---|---|---|---|
| SMIDS | 3-class sperm images; ResNet50 ×5, Xception ×3, MobileNetV3-Large ×3 | stratified 70/10/10/10 train/validation/calibration/test | no released patient or source-field ID |
| HuSHeM | 4-class sperm-head morphology; ResNet50 across 5 folds | five stratified outer folds with disjoint train/validation/calibration/test roles | 15 patients reported, but no image-to-patient map |
| Kromp | proposed blastocyst task | would require a genuine patient-grouped split | blocked: public files lack defensible patient linkage and contain unresolved duplicate/annotation defects |

Models train on **clean training images only**. Validation macro-F1 selects each checkpoint. Scalar temperature scaling, vector scaling, and APS fit only the disjoint **clean calibration role**. Test images fit nothing, and corruptions are applied only during evaluation. The SMIDS pilot and synthetic demo are excluded from the scientific grid.

[`configs/analysis_protocol.yaml`](configs/analysis_protocol.yaml) freezes the aggregation order, severities, and decision rules. Accuracy crosses after a drop greater than 0.05; ECE, entropy, and selective risk use prespecified relative-plus-absolute increases; APS crosses below 0.85. A signal counts as early only if both crossings exist and its severity is strictly lower.

## Clean-test context

| Dataset / backbone | Accuracy, mean ± sample SD | Macro-F1, mean ± sample SD |
|---|---:|---:|
| SMIDS / ResNet50 | 89.7% ± 0.7 | 89.7% ± 0.7 |
| SMIDS / Xception | 88.8% ± 2.2 | 88.9% ± 2.1 |
| SMIDS / MobileNetV3-Large | 88.9% ± 1.6 | 89.0% ± 1.6 |
| HuSHeM / ResNet50 | 86.5% ± 7.8 | 86.5% ± 7.9 |

HuSHeM's ±7.8-point accuracy spread is highly exposed to small-*n* resolution: four test folds contain 43 images and one contains 44, so one changed classification moves fold accuracy by 2.27–2.33 percentage points. It combines learned-model, fold-composition, and coarse evaluation variability and should not be called model instability alone.

MobileNetV3-Large's clean SMIDS accuracy was descriptively close to ResNet50, 88.9% versus 89.7%. This makes the compact model relevant to memory-, latency-, and power-constrained on-device research, but does not establish equivalence or deployment readiness.

## Secondary and exploratory analyses

These analyses do not revise the primary result.

- **Per corruption:** applying the same definitions separately produced 112 correlated comparisons. Both thresholds existed in 58, and none was early; the signal never crossed in the other 54. No multiple-testing claim is made. The seven panels are in [`results/figures/appendix/`](results/figures/appendix/).
- **Failure detection:** raw-softmax per-sample failure AUROC moved from severity 1→5 as follows: SMIDS ResNet50 0.817→0.697, Xception 0.839→0.695, MobileNetV3-Large 0.807→0.657, HuSHeM ResNet50 0.815→0.704, and the SMIDS ensemble 0.836→0.732.
- **Selective prediction:** at 80% retention, energy-based selection recovered +3.6 to +5.2 accuracy points at severities 3–4 across the four single-backbone paths. Ensemble entropy recovered +5.2/+5.4 points at S3/S4. These are ranking results at a fixed review budget, not a clinical abstention policy.
- **Conformal behavior:** clean→S5 APS coverage/set size was 0.977→0.865 / 1.761→1.910 for SMIDS ResNet50; 0.958→0.760 / 1.546→1.822 for MobileNetV3-Large; 0.987→0.869 / 1.873→1.994 for Xception; and 0.958→0.942 / 2.311→2.966 for HuSHeM.
- **Temperature transfer:** raw→scaled S5 ECE was 0.274→0.251 for SMIDS ResNet50, 0.397→0.384 for MobileNetV3-Large, 0.181→0.178 for Xception, and 0.184→0.198 for HuSHeM.
- **Attribution:** the real-checkpoint Grad-CAM/Grad-CAM++ defocus analysis is explicitly exploratory and has separate provenance. See the [qualitative grid](results/attribution/attribution_grid.png) and [stability/accuracy panel](results/figures/f7_attribution_stability_accuracy.png).

Post hoc, the relative threshold is visibly baseline-dependent. HuSHeM clean ECE was 0.132, so its frozen 2× rule required ECE to exceed 0.264 as well as the 0.02 minimum increase. This observation does not revise the primary analysis; absolute-threshold and AUROC-style alarm protocols are future work that must be specified before a new evaluation.

The remaining final figures are [reliability diagrams](results/figures/f2_reliability_diagrams.png), [risk–coverage curves](results/figures/f3_risk_coverage.png), [failure-detection AUROC](results/figures/f4_failure_detection_auroc.png), and [conformal coverage/set size](results/figures/f5_conformal.png). Their tidy plotted data and hashes are registered in [`results/figure_data/final_figure_manifest.json`](results/figure_data/final_figure_manifest.json).

## Simulated device shift

These are deterministic ordinal sensitivity settings, not measurements mapped to a named phone or microscope.

| Corruption | Physical mechanism | Severity 1 → 5 at a 224 px short side |
|---|---|---|
| Defocus blur | lower-NA optics or autofocus miss | Gaussian σ 0.5, 1, 2, 3, 5 px |
| Motion blur | ideal handheld exposure-smear proxy | kernel length 5, 11, 19, 31, 45 px |
| Gaussian noise | ImageNet-C independent-channel additive baseline; deliberately least physical | σ 0.02, 0.04, 0.08, 0.12, 0.18 on [0,1] |
| Shot noise | post-demosaic proxy for photon-limited acquisition | effective luminance count scale 4,096, 1,024, 256, 64, 16 |
| JPEG | lossy encoding/transfer | quality 80, 60, 40, 25, 12 |
| Down–up resampling | spatial-resolution or sensor-density loss | factors 1.5, 2.25, 3.5, 5.5, 8; bilinear restoration |
| Gamma + white balance | stylized light-source/ISP proxy | base γ 0.85→0.50 or its reciprocal, with a per-image RGB direction |

Gaussian noise deliberately follows the ImageNet-C independent per-pixel/channel convention for comparability and is the least physically realistic corruption by design. Shot noise is the more physically motivated counterpart, using a luminance-derived Poisson residual shared across RGB channels as a post-demosaic proxy. For illumination, a seed derived deterministically for each image fixes its gamma branch and zero-mean RGB white-balance direction across all severities. The illustrated seed-1729 grid happens to darken midtones, suppress red, and raise green and blue, producing its green/teal cast; that direction is not imposed on every evaluation image.

![Seven deterministic device-corruption ladders on a held-out SMIDS image](results/figures/corruption_grid.png)

The exact parameters and fixed-seed source are recorded in [`results/figures/corruption_grid.json`](results/figures/corruption_grid.json). The natural realism test is paired capture of the same specimens on reference and low-cost devices, followed by a newly frozen evaluation.

## Relation to Shafiee Lab work

Thirumalaraju et al. showed substantial intermodel and cross-center variability despite similar average performance. The present null is consistent in spirit with that reliability concern, but tests different tasks and endpoints: here, standard signals did not provide threshold-level lead time. HuSHeM's coarse fold variability is a thematic connection, not an effect-size comparison.

Kanakasabapathy et al.'s MD-nets work framed lossy acquisition and device/domain quality as a deployment problem across clinical and smartphone imaging. This repository tests a narrower monitoring question along a simulated quality axis. Ovadia et al. found that in-distribution post-hoc calibration does not reliably transfer under shift; the mixed temperature-scaling direction here is a narrow contextual parallel, not a replication.

## Reproduce

Python 3.11 is the tested environment. A full-history clone is required because figure provenance resolves the committed training/evaluation revisions. Raw data, released checkpoints, caches, and reproduction outputs remain ignored by Git.

```bash
git clone https://github.com/akshat0714/calibration-under-shift.git
cd calibration-under-shift
bash run.sh --setup
MPLBACKEND=Agg bash run.sh --eval-only
```

The no-argument evaluation route:

1. downloads and checksum-verifies SMIDS and HuSHeM only—never Kromp;
2. rechecks the exact split roles and rotating HuSHeM test folds;
3. downloads the pinned, hash-verified [Stage-1 checkpoint release](https://github.com/akshat0714/calibration-under-shift/releases/tag/stage1-gcp-handoff-v1), verifies archive SHA-256 `06510f8813eb2f67b11268ebcb2761fbb616c777bf37364ca8bda2b485a0105f`, and installs exactly 16 checkpoints;
4. regenerates Stage 2/3, the corruption grid, full attribution audit, F1–F7, and seven per-corruption appendices under ignored `results/reproduction/`; and
5. requires 45,540 tidy rows, 17 detail JSONs, exact primary crossings, portable paths, and matching values before writing `verification.json`.

The command auto-selects CUDA or Apple MPS and refuses silent CPU execution. `CALIBRATION_NUM_WORKERS` changes scheduling only. With `CALIBRATION_DEVICE=mps` and `CALIBRATION_NUM_WORKERS=0`, a genuinely cold macOS/Apple-MPS fresh clone was measured at 42m29s end to end; a warm-cache rerun took 26m47s. T4 runtime has not been measured, so this repository does not claim the earlier <30-minute T4 target. Exact timings and gate outputs are recorded in [`STAGE6_VERIFICATION.md`](STAGE6_VERIFICATION.md).

To retrain rather than use released checkpoints:

```bash
# CUDA only; resumes completed logical members from its registry.
bash run.sh --full-retrain
```

The operational GCP procedure and shutdown guardrails are in [`RUN_ON_GCP.md`](RUN_ON_GCP.md). The synthetic `bash run.sh --demo` path remains an engineering fixture and never satisfies scientific-grid validation.

## Repository map

```text
configs/                 versioned model and frozen analysis protocols
data/metadata, splits/   audited metadata and immutable split manifests
experiments/             training, evaluation, analysis, reproduction, attribution
notebooks/               result-only walkthrough; no training
scripts/                 checksum downloads, release verification, fixed grids
src/                     data, models, shifts, metrics, uncertainty, attribution, plots
tests/                   unit, protocol, provenance, notebook, and smoke coverage
results/                 committed metrics, tables, figures, audits, and manifests
```

## Limitations

This is a public-proxy study of simulated, not paired-device, shift. SMIDS and HuSHeM lack released patient/source linkage; HuSHeM is small; Kromp remains blocked pending an author-verified patient map and label resolution. Single-source datasets do not establish held-out-center generalization. Synthetic corruptions omit a real phone's full optics, Bayer sampling, demosaicing, denoising, sharpening, tone mapping, and color pipeline.

ECE is binning-dependent, temperature scaling corrects only a global sharpness error, and APS offers marginal coverage under exchangeability rather than a per-class or shifted-distribution guarantee. The threshold-relative caveat was observed after the primary analysis. Grad-CAM is noncausal. Nothing here constitutes clinical validation, a clinical decision policy, model equivalence, or proof that uncertainty methods are generally ineffective.

## Citations and licenses

Primary context: [Thirumalaraju et al., *Fertility and Sterility*](https://doi.org/10.1016/j.fertnstert.2025.08.021), [Kanakasabapathy et al., *Nature Biomedical Engineering*](https://doi.org/10.1038/s41551-021-00733-w), [Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html), [Hendrycks & Dietterich, ICLR 2019](https://openreview.net/forum?id=HJz6tiCqYm), and [Angelopoulos & Bates](https://doi.org/10.1561/2200000101).

SMIDS, HuSHeM, and Kromp are cited with checksums, release caveats, and CC BY 4.0 terms in [`DATASETS.md`](DATASETS.md). Raw archives are not redistributed. Code is GPL-3.0-only; source-image portions of committed audit/corruption figures retain their dataset licenses.
