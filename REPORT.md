# Calibration Under Shift: Prespecified Reliability Signals Under Simulated Device Degradation

## Abstract

This study asked whether standard reliability signals warn before accuracy degrades under simulated low-cost imaging shift. Sixteen models were trained only on clean SMIDS or HuSHeM images; a disjoint clean calibration split fitted post-hoc methods. The same test images were evaluated across seven corruptions and five severities. No reliability signal crossed before the five-point accuracy-drop threshold in any of the 10 comparisons where both crossings were observed; six further signals never crossed. The prespecified hypothesis was therefore not supported at these thresholds on these datasets. Clean-fitted temperature scaling improved severity-5 ECE on all three SMIDS backbones but worsened HuSHeM ResNet50. These public-proxy results support no clinical claim and motivate paired-device validation and monitoring beyond uncertainty dashboards.

## Motivation and prespecified question

Aggregate accuracy can conceal changes in probability quality, error ranking, disagreement, or prediction-set behavior. Kanakasabapathy et al. framed lossy acquisition and device/domain quality as a deployment problem [2], while Thirumalaraju et al. found substantial intermodel and cross-center variability despite similar average performance [1]. This work asks whether off-the-shelf reliability signals provide threshold-level lead time along a controlled degradation axis.

The following hypothesis was written before the full shifted grid and is preserved verbatim:

> We predict that calibration error and predictive uncertainty will degrade at lower simulated device-corruption severities than classification accuracy. We will test this by measuring accuracy, macro-F1, ECE, adaptive ECE, NLL, Brier score, uncertainty, selective risk, and conformal coverage across fixed severity ladders applied only at evaluation time. The hypothesis is falsified if prespecified reliability thresholds are not crossed before a five-percentage-point accuracy drop, consistently across datasets and seeds.

This was a local prespecification, not an external registration. Its aggregation order, thresholds, and missing-crossing rule were frozen in `configs/analysis_protocol.yaml`; the observed result did not support it.

## Methods

### Data, models, and leakage controls

SMIDS supplies a three-class sperm-image task with disjoint train, validation, calibration, and test roles [7]. HuSHeM supplies a four-class sperm-head task across five stratified outer folds [8]. Neither release includes image-level patient/source linkage, so these are public-proxy rather than patient-level evaluations.

The matrix contains five SMIDS ResNet50 seeds, three Xception seeds, three MobileNetV3-Large seeds, and HuSHeM ResNet50 across five folds. Models train on clean images; validation macro-F1 selects checkpoints. Temperature/vector scaling and APS fit only the clean calibration role. Test images fit nothing, and corruptions are evaluation-only. The five ResNet50 seeds form the SMIDS ensemble. The pilot and synthetic demo are excluded from scientific summaries.

Kromp et al. remains excluded: its public files lack defensible patient linkage and contain unresolved duplicate/annotation defects [9]. `DATASETS.md` documents the blocker; filename prefixes were not relabeled as patients.

### Simulated shift and reliability methods

Seven deterministic corruptions are applied to the same test images at severities 1–5: defocus, motion, Gaussian noise, shot noise, JPEG, resampling, and gamma/white-balance shift. These are ordinal sensitivity settings, not named-device equivalents. Gaussian noise deliberately follows the ImageNet-C independent-channel convention for comparability and is least physical by design [4]. Its more physical counterpart shares a luminance-derived Poisson residual across RGB channels. Fixed-seed illumination darkens midtones, suppresses red, and increases green/blue, producing the implemented green/teal direction.

The grid records classification, calibration, risk–coverage, failure-detection AUROC, 30-pass MC dropout, ensemble uncertainty, APS, energy, and Mahalanobis metrics. Temperature scaling uses clean calibration [5]; APS uses a corrected split-conformal quantile at nominal 90% coverage [6].

The primary analysis weights corruptions equally within each seed/fold, then summarizes replicates. Accuracy crosses more than 0.05 below clean. ECE must exceed 2× clean and a 0.02 increase; entropy, 2× and 0.05; risk at 80% coverage, 2× and 0.02. APS crosses below 0.85. A signal is earlier only when both crossings exist and its severity is lower. Missing crossings remain missing.

## Results

### Clean sanity context

| Dataset | Backbone | Clean accuracy, mean ± sample SD | Clean macro-F1, mean ± sample SD |
|---|---|---:|---:|
| SMIDS | ResNet50 | 89.7% ± 0.7 | 89.7% ± 0.7 |
| SMIDS | Xception | 88.8% ± 2.2 | 88.9% ± 2.1 |
| SMIDS | MobileNetV3-Large | 88.9% ± 1.6 | 89.0% ± 1.6 |
| HuSHeM | ResNet50 | 86.5% ± 7.8 | 86.5% ± 7.9 |

HuSHeM's ±7.8-point spread is exposed to small-*n* resolution: four folds contain 43 images and one contains 44, so one classification moves accuracy by 2.27–2.33 points. It blends model, fold-composition, and evaluation variability, not model instability alone. Its connection to Thirumalaraju et al. is contextual, not an effect-size comparison [1].

MobileNetV3-Large's SMIDS clean accuracy was descriptively close to ResNet50, 88.9% versus 89.7%. This makes it relevant to constrained on-device research, but establishes neither equivalence nor deployment readiness.

### Primary/prespecified threshold result

![Threshold-normalized primary trajectories](results/figures/f1_headline_lockstep.png)

**Figure 1.** Clean is 0 and each frozen threshold is 1; X marks the first crossing. Signals and accuracy share the severity axis, with no outcome shading. Missing X means no crossing. Means weight corruptions equally within seed/fold.

| Dataset / backbone | Replicates | Accuracy drop | ECE | Predictive entropy | Selective risk @80% | APS coverage |
|---|---:|---:|---:|---:|---:|---:|
| SMIDS / ResNet50 | 5 seeds | S3 | S3 | — | S3 | — |
| SMIDS / Xception | 3 seeds | S3 | S3 | S5 | S3 | — |
| SMIDS / MobileNetV3-Large | 3 seeds | S2 | S2 | — | S2 | S4 |
| HuSHeM / ResNet50 | 5 folds | S3 | — | S4 | S4 | — |

All 10 observed paired crossings were simultaneous or later; six signals never crossed. The hypothesis was therefore not supported at these thresholds on SMIDS and HuSHeM. An em dash means missing, not severity 5.

### Secondary/exploratory analyses

These analyses do not revise the primary claim. Per-corruption application produced 0 early warnings among 58 observed crossing pairs; signals never crossed in the other 54 of 112 correlated comparisons. No multiple-testing claim is made; trajectories are in `results/figures/appendix/`.

At severity 5, mean raw→temperature ECE was 0.274→0.251 for SMIDS ResNet50, 0.397→0.384 for MobileNetV3, 0.181→0.178 for Xception, and 0.184→0.198 for HuSHeM. Paired changes were −0.023, −0.013, −0.003, and +0.014. This mixed transfer is consistent in spirit with, but not a replication of, Ovadia et al. [3].

Failure-detection AUROC from severity 1→5 changed by 0.817→0.697 for SMIDS ResNet50, 0.839→0.695 for Xception, 0.807→0.657 for MobileNetV3, 0.815→0.704 for HuSHeM, and 0.836→0.732 for the ensemble. At 80% retention, energy selection recovered 3.6–5.2 accuracy points at severities 3–4; ensemble entropy recovered 5.2/5.4 points. These are ranking results, not a validated abstention policy.

APS clean→severity-5 coverage was 0.977→0.865 for SMIDS ResNet50, 0.958→0.760 for MobileNetV3, 0.987→0.869 for Xception, and 0.958→0.942 for HuSHeM. Set size grew from 1.761→1.910, 1.546→1.822, 1.873→1.994, and 2.311→2.966 classes. Neither quantity supplied primary lead time. The real-checkpoint Grad-CAM/++ defocus analysis in [Figure 7](results/figures/f7_attribution_stability_accuracy.png) is exploratory, not a primary alarm result.

## Interpretation and limitations

The central finding is lockstep, not lead time: curves changed, but frozen decision points did not precede accuracy. This bounded null does not make uncertainty methods useless. A score can retain per-sample ranking value without acting as an earlier aggregate alarm.

The result is consistent in spirit—not a replication—with Thirumalaraju et al.'s reliability warning [1]. Here, common signals provided no threshold-level lead time. For low-cost deployment research, uncertainty dashboards alone should not be degradation alarms; paired-device validation and monitoring remain necessary.

One post-hoc observation concerns threshold relativity. HuSHeM clean ECE was 0.132, so the frozen 2× rule required exceeding 0.264 and the 0.02 minimum increase. This can be harder than an absolute criterion when baseline ECE is high, but does not redefine this result. Absolute-threshold or AUROC-style alarm protocols are future, prespecified work.

The shifts are simulated public proxies, not paired reference/smartphone captures. The releases lack patient/source linkage and held-out-center evaluation; HuSHeM is small. Synthetic corruptions omit a phone's full optical and ISP pipeline. Kromp remains blocked pending verified linkage and labels. ECE is estimator-dependent, APS coverage is marginal and exchangeability-dependent, and Grad-CAM is noncausal. Nothing here constitutes clinical validation, a performance guarantee, or a decision policy.

Next, paired reference/low-cost captures should use patient-grouped splits, a held-out center, measured device statistics, and a newly frozen alert protocol.

## Traceability

Aggregate numbers derive from `results/metrics.csv`; crossings from `results/thresholds.csv`; definitions from `configs/analysis_protocol.yaml`. `results/figure_data/final_figure_manifest.json` registers plotted data and hashes. Attribution has separate provenance. Pilot/demo rows are excluded.

## References

1. Thirumalaraju P et al. “Stability and reliability of artificial intelligence models in embryo selection for in vitro fertilization.” *Fertility and Sterility* 125(2), 277–286 (2026; online 2025). [doi:10.1016/j.fertnstert.2025.08.021](https://doi.org/10.1016/j.fertnstert.2025.08.021).
2. Kanakasabapathy MK et al. “Adaptive adversarial neural networks for the analysis of lossy and domain-shifted datasets of medical images.” *Nature Biomedical Engineering* 5, 571–585 (2021). [doi:10.1038/s41551-021-00733-w](https://doi.org/10.1038/s41551-021-00733-w).
3. Ovadia Y et al. “Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift.” NeurIPS (2019). [Official proceedings](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).
4. Hendrycks D, Dietterich T. “Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.” ICLR (2019). [OpenReview](https://openreview.net/forum?id=HJz6tiCqYm).
5. Guo C et al. “On Calibration of Modern Neural Networks.” ICML, PMLR 70:1321–1330 (2017). [Proceedings](https://proceedings.mlr.press/v70/guo17a.html).
6. Angelopoulos AN, Bates S. “Conformal Prediction: A Gentle Introduction.” *Foundations and Trends in Machine Learning* 16(4), 494–591 (2023). [doi:10.1561/2200000101](https://doi.org/10.1561/2200000101).
7. Takidin H, Ceylan HI, Kusetogullari H. *Sperm Morphology Image Data Set (SMIDS)*, Mendeley Data v1 (2022). [doi:10.17632/6xvdhc9fyb.1](https://doi.org/10.17632/6xvdhc9fyb.1).
8. Shaker M, Monadjemi SA. *Human Sperm Head Morphology dataset (HuSHeM)*, Mendeley Data v3 (2018). [doi:10.17632/tt3yj2pf38.3](https://doi.org/10.17632/tt3yj2pf38.3).
9. Kromp F et al. “An annotated human blastocyst dataset to benchmark deep learning architectures for in vitro fertilization.” *Scientific Data* 10, 271 (2023). [doi:10.1038/s41597-023-02182-3](https://doi.org/10.1038/s41597-023-02182-3).
