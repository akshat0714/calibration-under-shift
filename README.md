# Calibration Under Shift

[![CI](https://github.com/akshat0714/calibration-under-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/akshat0714/calibration-under-shift/actions/workflows/ci.yml)

**Thesis:** diagnostic-image confidence may become unreliable before accuracy visibly collapses as image quality degrades toward smartphone-microscope conditions.

This repository implements a prespecified, leakage-resistant study of calibration, uncertainty, conformal prediction, selective prediction, input-shift detection, and attribution stability under simulated diagnostic-imaging shift. It is intentionally honest about its current evidence: SMIDS and HuSHeM have been checksum-verified, audited, and split; the complete synthetic engineering workflow passes; the multi-seed public-data experiment is not yet complete, so no headline empirical claim is made here.

> **Release status:** code and data-audit candidate, not yet a sendable scientific result. Do not replace this notice with a conclusion until `results/metrics.csv`, `results/thresholds.csv`, the seeded/fold summaries, and a fresh-clone run have all been verified.

![Seven deterministic device-corruption ladders on a held-out SMIDS image](results/figures/corruption_grid.png)

## What is verified now

| Check | Outcome | Evidence |
|---|---:|---|
| Automated tests | 127 passing | shifts, metrics, scaling, conformal, data leakage, models, Grad-CAM, figures, and end-to-end smoke training |
| SMIDS release audit | 3,000/3,000 images decoded | expected 1,021/1,005/974 class counts; no corrupt or exact-duplicate files |
| HuSHeM release audit | 216/216 images decoded | expected 54/53/57/52 class counts; no corrupt or exact-duplicate files |
| Kromp release audit | 2,344/2,344 images decoded | patient map absent; 15 exact-duplicate groups; one conflicting and one missing Gardner label |
| Split/calibration guardrails | enforced in code | fixed manifests; post-hoc scalers and APS reject any role other than `calibration` |
| Public-data headline | pending | a synthetic demo is engineering evidence only and is excluded from scientific results |

The full local audit is in [`results/data_audit.md`](results/data_audit.md). Kromp is deliberately blocked: its public v3 release has no patient map, filename prefixes do not reproduce the paper's patient count, exact duplicates cross prefixes, and one silver-standard image has conflicting expansion labels. Calling a prefix split “patient-level” would be indefensible.

## Study design

Models train on **clean train images only**. Validation selects the checkpoint. A disjoint **clean calibration split** fits scalar/vector scaling and the APS threshold. The exact same held-out test images are then decoded with one deterministic corruption at severity 1–5; calibrators are not refit after shift.

The prespecified question is: does a reliability signal cross its threshold before raw-softmax accuracy loses five percentage points? [`configs/analysis_protocol.yaml`](configs/analysis_protocol.yaml) freezes the aggregation order and thresholds before the full seeded public-data grid. Missing crossings remain missing rather than being assigned severity 5.

### Tasks and splits

| Dataset | Task | Split design | Current caveat |
|---|---|---|---|
| SMIDS | 3-class normal / abnormal / non-sperm | stratified 70/10/10/10 train/val/calibration/test | no released patient or source-field ID |
| HuSHeM | 4-class head morphology | five stratified outer folds; disjoint train/val/calibration/test within each fold | 15 donors are reported but no image-to-donor map is released |
| Kromp | binary high quality: Gardner expansion ≥3 and ICM/TE in {A,B} | requires genuine patient-grouped 60/15/10/15 manifest | blocked pending author-verified patient map and label resolution |

Temperature/vector scaling and conformal thresholds are fitted only on the calibration role. Validation never tunes post-hoc calibration; test data never fit anything.

## Simulated device shift

The engine follows an ImageNet-C-style sensitivity design but maps each corruption to a low-cost imaging mechanism. These are mechanistic proxies, not a claim that synthetic severity 3 equals a particular phone or microscope.

| Corruption | Physical mechanism | Severity 1 → 5 at a 224 px short side |
|---|---|---|
| Defocus blur | lower-NA phone optics or autofocus miss | Gaussian σ 0.5, 1, 2, 3, 5 px |
| Motion blur | ideal linear exposure-smear proxy for handheld capture | 224-px-reference kernel length 5, 11, 19, 31, 45 px |
| Gaussian noise | ImageNet-C-style independent-channel additive baseline; deliberately least physical | σ 0.02, 0.04, 0.08, 0.12, 0.18 on [0,1] |
| Shot noise | luminance-correlated post-demosaic proxy for photon-limited low-light acquisition | effective luminance count scale 4,096, 1,024, 256, 64, 16 |
| JPEG | lossy smartphone encoding/transfer | quality 80, 60, 40, 25, 12 |
| Down–up resampling | spatial-resolution or sensor-density loss | area-downsample factor 1.5, 2.25, 3.5, 5.5, 8; bilinear restoration |
| Gamma + white balance | seeded global darkening and green/teal cast as a stylized light-source/ISP proxy | with protocol seed 1729: γ 1.18, 1.33, 1.54, 1.72, 2.00; severity-5 RGB gains 0.750, 1.145, 1.105 |

Severity levels are ordinal sensitivity-analysis settings, not measurements mapped to any named device or acquisition condition.

Gaussian noise deliberately retains the ImageNet-C independent per-pixel/channel
convention for benchmark comparability; it is the least physically realistic
corruption by design. Shot noise is its more physically motivated counterpart:
one luminance-derived Poisson residual is shared across RGB channels to approximate
post-demosaic channel correlation. With the fixed protocol seed, the illumination
direction suppresses red and increases green and blue, so its growing bias is
green/teal rather than an unspecified random white-balance cast.

Every corruption is `corrupt(image, name, severity, seed)`: same input and seed produce identical RGB output, severity 0 is an identity condition, and corruption callbacks are rejected on the training split.
Pixel-space blur and motion parameters are scaled by the decoded image's shorter
side relative to 224 px, so one nominal severity does not mean a different relative
kernel merely because the public SMIDS images have different native dimensions.
Regenerate the displayed held-out sample with its fixed manifest row, seed, and
machine-readable provenance sidecar using `bash run.sh --corruption-grid`.

## Methods

- **Backbones:** torchvision ResNet50, timm Xception, and MobileNetV3-Large, each with explicit pre-head dropout. A tiny CNN exists only for CI and synthetic smoke checks.
- **Calibration:** 15-bin equal-width ECE, equal-mass adaptive ECE, multiclass Brier score, NLL, scalar temperature scaling, and vector scaling.
- **Uncertainty:** five-seed deep ensembles, predictive entropy, expected entropy, mutual information, variation ratio, and 30-pass MC dropout.
- **Decision support:** risk–coverage/AURC, failure-detection AUROC, risk at 80% coverage, and APS split-conformal sets at α=0.1.
- **Input shift:** max-softmax uncertainty, energy, and class-conditional Mahalanobis distance fitted on clean train features.
- **Attribution:** class-stratified Grad-CAM panels plus full-test clean-to-shift Spearman correlation and top-20% saliency-mask IoU; constant, non-localizing CAMs are reported as undefined. The same CLI supports tested Grad-CAM++ via `--method gradcam++`.

Deep ensembles directly target the replicate-model disagreement raised by Thirumalaraju et al.; the experiment tests whether disagreement is a useful per-image warning signal rather than merely another average predictor.

## Reproduce

Python 3.11 is the tested environment. Raw data and checkpoints are ignored by Git.

```bash
# Create .venv and install exact pins.
bash run.sh --setup

# Fast, data-free engineering proof. Its metrics are not scientific output.
bash run.sh --demo

# Download, verify, decode, audit, and write the fixed SMIDS manifest.
bash run.sh --download smids
bash run.sh --prepare smids

# Full five-seed ResNet50 training + complete grid + analysis + figures.
# This is compute-intensive; the command records every checkpoint and run.
bash run.sh --full-smids
```

The resumable, CUDA-guarded command sequence for the complete 16-member Stage 1
matrix is in [`RUN_ON_GCP.md`](RUN_ON_GCP.md).

To evaluate already-trained checkpoints without retraining:

```bash
bash run.sh --eval-only configs/smids_resnet50.yaml \
  results/checkpoints/smids-resnet50-seed2025-<run>.pt \
  results/checkpoints/smids-resnet50-seed2026-<run>.pt
```

This safety-oriented path writes `results/eval_metrics.csv` and
`results/eval_figures/`; it does not overwrite the canonical full-matrix result.

Direct entry points remain available:

```bash
.venv/bin/python -m src.train --config configs/smids_resnet50.yaml --seed 2025
.venv/bin/python -m experiments.run_grid --config configs/smids_resnet50.yaml \
  --checkpoints results/checkpoints/smids-resnet50-seed*.pt
.venv/bin/python -m experiments.analyze
.venv/bin/python -m src.viz.figures --metrics results/metrics.csv \
  --output-dir results/figures --uncertainty sd
.venv/bin/python -m experiments.generate_diagnostics --config configs/smids_resnet50.yaml \
  --checkpoint results/checkpoints/<checkpoint>.pt
.venv/bin/python -m experiments.run_attribution --config configs/smids_resnet50.yaml \
  --checkpoint results/checkpoints/<checkpoint>.pt
```

Training runs write the resolved configuration, seed, Git revision, environment, curves, and metrics under `results/runs/<run_id>/`; checkpoint evaluation embeds the saved training identity in every tidy row. Inference caches are keyed by checkpoint bytes, manifest bytes, corruption-protocol digest, split, corruption, and severity.

## Repository map

```text
configs/                 versioned model and analysis protocols
data/metadata, splits/   audited metadata and immutable split manifests
experiments/             training matrix, full grid, threshold and attribution runs
notebooks/               result-only narrative walkthrough; no training
scripts/                 checksum downloads and synthetic smoke-data generator
src/data/                decoding, audit, transforms, and leakage-resistant splits
src/models/              backbone factory and feature interface
src/shifts/              deterministic device-corruption engine
src/metrics/             classification, calibration, and selective metrics
src/uncertainty/         scaling, ensembles, MC dropout, APS, and OOD scores
src/attribution/         Grad-CAM and quantitative stability
src/viz/                 publication-style figure generation
tests/                   unit and end-to-end smoke coverage
```

## Relation to Shafiee Lab work

Kanakasabapathy et al.'s MD-nets study framed lossy acquisition and device/domain quality as a deployment problem across clinical and smartphone imaging. Thirumalaraju et al. later showed substantial instability among replicate embryo-ranking models and a marked cross-center error-variance increase. This repository asks a narrower operational question: can calibration, disagreement, conformal sets, selective risk, and attribution drift expose unreliable individual predictions before aggregate accuracy makes a failure obvious?

The experimental vocabulary also follows the lab's embryo-morphology work and Kromp et al.'s public Gardner annotations, while its calibration and shift baselines follow Guo et al., Ovadia et al., Hendrycks and Dietterich, and Angelopoulos and Bates.

Primary references:

- P. Thirumalaraju et al., “Stability and reliability of artificial intelligence models in embryo selection for in vitro fertilization,” *Fertility and Sterility* 125(2), 277–286 (2026; online 2025), [doi:10.1016/j.fertnstert.2025.08.021](https://doi.org/10.1016/j.fertnstert.2025.08.021).
- M. K. Kanakasabapathy et al., “Adaptive adversarial neural networks for the analysis of lossy and domain-shifted datasets of medical images,” *Nature Biomedical Engineering* 5, 571–585 (2021), [doi:10.1038/s41551-021-00733-w](https://doi.org/10.1038/s41551-021-00733-w).
- F. Kromp et al., “An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization,” *Scientific Data* 10, 271 (2023), [doi:10.1038/s41597-023-02182-3](https://doi.org/10.1038/s41597-023-02182-3).
- C. Guo et al., “On Calibration of Modern Neural Networks,” ICML (2017), [PMLR 70:1321–1330](https://proceedings.mlr.press/v70/guo17a.html).
- Y. Ovadia et al., “Can You Trust Your Model's Uncertainty?,” NeurIPS (2019), [official proceedings](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).
- D. Hendrycks and T. Dietterich, “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,” ICLR (2019), [OpenReview](https://openreview.net/forum?id=HJz6tiCqYm).
- A. N. Angelopoulos and S. Bates, “Conformal Prediction: A Gentle Introduction,” *Foundations and Trends in Machine Learning* 16(4), 494–591 (2023), [doi:10.1561/2200000101](https://doi.org/10.1561/2200000101).

Dataset citations, licenses, exact archive hashes, and release defects are documented in [`DATASETS.md`](DATASETS.md).

## Limitations

This is a public-proxy, simulated-shift study. It is not paired clinical-microscope versus smartphone capture, does not establish that the corruption ladder matches a real phone ISP or optical transfer function, and supports no clinical claim. SMIDS and HuSHeM lack released patient linkage; HuSHeM is small; Kromp cannot currently support the promised patient-level split. A calibration or conformal result from these data does not transfer automatically to a clinic, device, site, or prevalence.

The natural next step is paired-device validation on the lab's real acquisition data, with patients grouped, center held out, corruptions compared against measured device statistics, and all calibration choices frozen before the target-center test.

## License

Code is released under GPL-3.0-only. Raw dataset archives are not redistributed. The four audit/sanity figures contain attributed thumbnails or transformations from the cited CC BY 4.0 releases; those image portions retain their source license. See [`DATASETS.md`](DATASETS.md).
