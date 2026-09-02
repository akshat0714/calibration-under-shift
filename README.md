# Calibration Under Shift

[![CI](https://github.com/akshat0714/calibration-under-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/akshat0714/calibration-under-shift/actions/workflows/ci.yml)

I built this project around one question. Do common reliability measures signal degradation under simulated device shift before classifier accuracy falls by more than five percentage points?

Under the protocol I set before evaluation, the answer was no. ECE, predictive entropy, selective risk, and APS coverage reached their degradation thresholds at the same severity as the accuracy drop or later in all 10 comparisons where both thresholds were reached. Six other signals never reached their thresholds. This result does not mean uncertainty methods are useless. It means that none provided an earlier threshold crossing under these definitions on SMIDS and HuSHeM.

![Reliability and accuracy thresholds](results/figures/f1_headline_lockstep.png)

**Figure 1.** I normalized each measure so that clean performance is 0 and its threshold is 1. X shows the first threshold crossing. A measure without an X did not cross. I averaged all seven corruptions within each seed or fold before I summarized the replicates.

## Main result

| Dataset and model | Replicates | Accuracy drop | ECE | Predictive entropy | Selective risk at 80% | APS coverage |
|---|---|---|---|---|---|---|
| SMIDS ResNet50 | 5 seeds | S3 | S3 | Not reached | S3 | Not reached |
| SMIDS Xception | 3 seeds | S3 | S3 | S5 | S3 | Not reached |
| SMIDS MobileNetV3-Large | 3 seeds | S2 | S2 | Not reached | S2 | S4 |
| HuSHeM ResNet50 | 5 folds | S3 | Not reached | S4 | S4 | Not reached |

I defined an accuracy crossing as a drop of more than five percentage points from clean performance. A reliability measure counted as earlier only when both thresholds were reached and the reliability severity was lower. I left missing crossings as missing instead of assigning them severity 5.

Temperature scaling fitted on clean calibration data transferred inconsistently in the prespecified secondary analysis. It lowered severity 5 ECE for all three SMIDS models but raised it for HuSHeM ResNet50.

## Data and models

| Dataset | What I used | Split |
|---|---|---|
| SMIDS | Three sperm morphology classes with ResNet50, Xception, and MobileNetV3-Large | Stratified train, validation, calibration, and test shares of 70, 10, 10, and 10% |
| HuSHeM | Four sperm head morphology classes with ResNet50 | Five stratified outer folds with separate train, validation, calibration, and test roles |
| Kromp | Proposed blastocyst task | Excluded because I could not verify patient linkage and found unresolved duplicate and annotation issues |

I trained every model on clean images. I selected checkpoints with validation macro F1. I fitted scalar temperature scaling, vector scaling, and APS on the separate clean calibration split. Corruptions were applied only during evaluation. The SMIDS pilot and synthetic demo are engineering checks and are not part of the reported results.

I saved the aggregation order, severity settings, and threshold rules in [`configs/analysis_protocol.yaml`](configs/analysis_protocol.yaml) before the full evaluation.

## Clean test results

| Dataset and model | Accuracy mean ± sample SD | Macro F1 mean ± sample SD |
|---|---|---|
| SMIDS ResNet50 | 89.7% ± 0.7 | 89.7% ± 0.7 |
| SMIDS Xception | 88.8% ± 2.2 | 88.9% ± 2.1 |
| SMIDS MobileNetV3-Large | 88.9% ± 1.6 | 89.0% ± 1.6 |
| HuSHeM ResNet50 | 86.5% ± 7.8 | 86.5% ± 7.9 |

HuSHeM has about 43 test images in each fold. One changed prediction moves fold accuracy by about 2.3 percentage points, so I interpret its variation cautiously. The spread includes model variation, fold composition, and the resolution of a small test set.

MobileNetV3-Large and ResNet50 had similar clean accuracy on SMIDS at 88.9% and 89.7%. I include that comparison because a compact model may be useful for future on-device work. I did not test statistical equivalence or deployment readiness.

## Secondary and exploratory results

### Individual corruptions

I repeated the same threshold analysis for each of the seven corruptions. Both thresholds were reached in 58 of 112 correlated comparisons, and none had an earlier reliability crossing. The reliability threshold was not reached in the other 54 comparisons.

### Error ranking

Raw softmax failure AUROC fell as severity increased. From severity 1 to severity 5, it changed from 0.817 to 0.697 for SMIDS ResNet50, 0.839 to 0.695 for Xception, 0.807 to 0.657 for MobileNetV3-Large, and 0.815 to 0.704 for HuSHeM ResNet50. The SMIDS ensemble changed from 0.836 to 0.732.

### Selective prediction

At 80% retention, energy selection improved retained accuracy by 3.6 to 5.2 percentage points at severities 3 and 4 across the four single-model settings. Ensemble entropy improved retained accuracy by 5.2 and 5.4 points. I treat this as an error-ranking result at a fixed review budget, not as a clinical policy.

### Conformal prediction

From clean data to severity 5, APS coverage changed from 0.977 to 0.865 for SMIDS ResNet50, 0.987 to 0.869 for Xception, 0.958 to 0.760 for MobileNetV3-Large, and 0.958 to 0.942 for HuSHeM ResNet50. Mean set size increased from 1.761 to 1.910, 1.873 to 1.994, 1.546 to 1.822, and 2.311 to 2.966 in the same order.

### Calibration transfer

At severity 5, raw and temperature-scaled ECE were 0.274 and 0.251 for SMIDS ResNet50, 0.181 and 0.178 for Xception, 0.397 and 0.384 for MobileNetV3-Large, and 0.184 and 0.198 for HuSHeM ResNet50.

### Attribution

I used Grad-CAM and Grad-CAM++ on defocus blur as an exploratory analysis. I report the [image comparison](results/attribution/attribution_grid.png) and [numerical result](results/figures/f7_attribution_stability_accuracy.png) as exploratory, separate from the main finding.

The relative threshold also depends on the clean baseline. For example, HuSHeM clean ECE was 0.132, so the 2× rule required ECE to exceed 0.264 as well as the minimum absolute increase. This is a post-hoc observation. I did not change the main analysis. In future work I would define absolute thresholds or AUROC-style detection rules before evaluation.

The remaining figures show [reliability diagrams](results/figures/f2_reliability_diagrams.png), [risk coverage curves](results/figures/f3_risk_coverage.png), [failure detection AUROC](results/figures/f4_failure_detection_auroc.png), [conformal results](results/figures/f5_conformal.png), and [individual corruption results](results/figures/appendix/). Their source files and hashes are recorded in [`results/figure_data/final_figure_manifest.json`](results/figure_data/final_figure_manifest.json).

## Simulated device shift

I used fixed severity settings from 1 through 5. They are controlled approximations and do not reproduce a specific phone or microscope.

| Corruption | Approximation | Severity settings at a 224 pixel short side |
|---|---|---|
| Defocus blur | Lower numerical aperture optics or autofocus error | Gaussian sigma values of 0.5, 1, 2, 3, and 5 pixels |
| Motion blur | Handheld exposure blur | Kernel lengths of 5, 11, 19, 31, and 45 pixels |
| Gaussian noise | ImageNet-C independent channel baseline | Sigma values of 0.02, 0.04, 0.08, 0.12, and 0.18 on the 0 to 1 range |
| Shot noise | Approximation of photon-limited acquisition after demosaicing | Effective luminance counts of 4096, 1024, 256, 64, and 16 |
| JPEG | Lossy image storage or transfer | Quality values of 80, 60, 40, 25, and 12 |
| Down and up resampling | Reduced spatial resolution or sensor density | Factors of 1.5, 2.25, 3.5, 5.5, and 8 with bilinear restoration |
| Gamma and white balance | Global illumination and color processing | Base gamma values from 0.85 to 0.50 or their reciprocals with a fixed per-image RGB direction |

I followed the ImageNet-C per-channel Gaussian noise convention so that this corruption is comparable with prior work. It is the least physically realistic corruption in the study by design. Shot noise is the physically motivated counterpart. I used a luminance-based Poisson residual shared across RGB channels as an approximation after demosaicing.

For illumination, a deterministic image seed fixes the gamma branch and RGB direction across severity. The displayed seed 1729 example darkens the image and shifts it toward green and teal. Other evaluation images can use different fixed directions.

![Seven simulated device corruptions on one held-out SMIDS image](results/figures/corruption_grid.png)

The exact settings and source image are recorded in [`results/figures/corruption_grid.json`](results/figures/corruption_grid.json).

## Run the project

I tested the repository with Python 3.11. A full-history clone is required because the provenance check verifies the recorded training and evaluation revisions.

```bash
git clone https://github.com/akshat0714/calibration-under-shift.git
cd calibration-under-shift
bash run.sh --setup
MPLBACKEND=Agg bash run.sh --eval-only
```

The evaluation command verifies the two datasets and their saved splits. It downloads the [checkpoint release](https://github.com/akshat0714/calibration-under-shift/releases/tag/stage1-gcp-handoff-v1), verifies archive SHA-256 `06510f8813eb2f67b11268ebcb2761fbb616c777bf37364ca8bda2b485a0105f`, and installs 16 checkpoints. It then regenerates the evaluation, analysis, corruption grid, attribution outputs, figures, and appendices in `results/reproduction/`. The final verification requires 45,540 metric rows, 17 detail files, the exact main result, portable paths, and matching values.

The script uses CUDA or Apple MPS and will not start an implicit CPU run. My verified Apple MPS runs took 42 minutes and 29 seconds on a fresh clone and 26 minutes and 55 seconds on the same clone after it was updated. I have not measured the T4 runtime. I recorded the checks in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

I use this command to resume or retrain all 16 models on CUDA.

```bash
bash run.sh --full-retrain
```

I use `bash run.sh --demo` only as a synthetic engineering test.

## Repository files

| Path | Contents |
|---|---|
| `configs/` | Model settings and the analysis protocol |
| `data/metadata/` and `data/splits/` | Audited metadata and saved split files |
| `experiments/` | Training, evaluation, analysis, and reproduction scripts |
| `notebooks/` | A walkthrough of the saved results |
| `scripts/` | Data download, checkpoint verification, and figure utilities |
| `src/` | Reusable project code |
| `tests/` | Unit, protocol, provenance, notebook, and integration tests |
| `results/` | Metrics, figures, audits, and provenance files |

## Limits

I used public proxy datasets and simulated image changes rather than paired-device captures. SMIDS and HuSHeM do not include image-level patient or source links. HuSHeM is small. I excluded Kromp because I could not resolve its patient linkage and annotation problems. I did not evaluate a separate clinical center.

The simulations do not reproduce the full optics and image-processing pipeline of a phone. ECE depends on binning. Temperature scaling only adjusts global confidence. APS gives marginal coverage under exchangeability and does not guarantee coverage for each class or after distribution shift. Grad-CAM is not a causal explanation.

I do not claim clinical validation, a clinical decision policy, model equivalence, or a general failure of uncertainty methods.

## Related work and licenses

Thirumalaraju et al. reported variation across models and evaluation settings even when average performance was similar. My study asks a different question about whether reliability measures cross fixed thresholds before accuracy. I use their study as context rather than as a direct comparison.

Kanakasabapathy et al. studied image acquisition and device quality in clinical and smartphone settings. My experiment is narrower because it uses simulated changes. Ovadia et al. showed that calibration fitted on the original distribution can transfer poorly under shift. My mixed temperature-scaling result is consistent with that concern but does not replicate their study.

The main references are [Thirumalaraju et al.](https://doi.org/10.1016/j.fertnstert.2025.08.021), [Kanakasabapathy et al.](https://doi.org/10.1038/s41551-021-00733-w), [Ovadia et al.](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html), [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html), [Hendrycks and Dietterich](https://openreview.net/forum?id=HJz6tiCqYm), and [Angelopoulos and Bates](https://doi.org/10.1561/2200000101).

I list dataset sources, checksums, and license terms in [`DATASETS.md`](DATASETS.md). I do not redistribute the raw archives. The code is available under GPL 3.0. Images used in the audit and corruption figures keep their original dataset licenses.
