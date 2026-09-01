# Results summary

## Main finding

This study asked a prespecified question: as seven simulated device corruptions become more severe, do standard reliability signals cross fixed degradation thresholds before raw-softmax accuracy falls by more than five percentage points? They did not. Among the 10 dataset/backbone/signal comparisons in which both thresholds were crossed, ECE, predictive entropy, selective risk, and APS coverage crossed at the same severity as accuracy or later; six further signals never crossed. The early-warning hypothesis was therefore **not supported at these thresholds on SMIDS and HuSHeM**. This is a bounded null result, not evidence that uncertainty methods are generally useless.

Clean-fitted temperature scaling also transferred inconsistently. In the prespecified secondary comparison at severity 5, it reduced mean ECE for every SMIDS backbone but increased it for HuSHeM ResNet50. This direction is consistent in spirit with Ovadia et al.'s finding that in-distribution post-hoc calibration need not remain effective under dataset shift, while the overall null is consistent in spirit with Thirumalaraju et al.'s warning that similar aggregate performance can conceal reliability variability. Neither paper tested this protocol, so these are contextual parallels rather than replications.

## Five numbers to remember

1. **0 of 10** observed paired threshold crossings were early warnings; **6 of 16** prespecified signals never crossed their degradation threshold.
2. The five-point accuracy-drop threshold occurred at **severity 2** for SMIDS MobileNetV3-Large and **severity 3** for SMIDS ResNet50, SMIDS Xception, and HuSHeM ResNet50.
3. Temperature-minus-raw mean ECE at severity 5 was **−0.023** for SMIDS ResNet50, **−0.013** for SMIDS MobileNetV3-Large, **−0.003** for SMIDS Xception, and **+0.014** for HuSHeM ResNet50. Negative means improvement.
4. At 80% coverage, secondary energy-based selection recovered **+3.6 to +5.2 percentage points** of accuracy at severities 3–4 across the four dataset/backbone paths. For the five-member SMIDS ResNet50 ensemble, entropy-based selection recovered **+5.2 points at severity 3** and **+5.4 points at severity 4** relative to that ensemble's unselective accuracy.
5. APS coverage from clean to severity 5 changed from **0.977 to 0.865** (SMIDS ResNet50), **0.958 to 0.760** (SMIDS MobileNetV3-Large), **0.987 to 0.869** (SMIDS Xception), and **0.958 to 0.942** (HuSHeM ResNet50); mean set size simultaneously increased in all four paths.

All severity summaries above first average the seven device corruptions within each seed/fold, then average replicates, exactly as specified in `configs/analysis_protocol.yaml`.

## Primary threshold result

| Dataset | Backbone | Signal | Signal crossing | Accuracy crossing | Interpretation |
|---|---|---|---:|---:|---|
| SMIDS | ResNet50 | ECE | 3 | 3 | same severity |
| SMIDS | ResNet50 | predictive entropy | — | 3 | signal did not cross |
| SMIDS | ResNet50 | selective risk at 80% coverage | 3 | 3 | same severity |
| SMIDS | ResNet50 | APS coverage | — | 3 | signal did not cross |
| SMIDS | Xception | ECE | 3 | 3 | same severity |
| SMIDS | Xception | predictive entropy | 5 | 3 | two severities later |
| SMIDS | Xception | selective risk at 80% coverage | 3 | 3 | same severity |
| SMIDS | Xception | APS coverage | — | 3 | signal did not cross |
| SMIDS | MobileNetV3-Large | ECE | 2 | 2 | same severity |
| SMIDS | MobileNetV3-Large | predictive entropy | — | 2 | signal did not cross |
| SMIDS | MobileNetV3-Large | selective risk at 80% coverage | 2 | 2 | same severity |
| SMIDS | MobileNetV3-Large | APS coverage | 4 | 2 | two severities later |
| HuSHeM | ResNet50 | ECE | — | 3 | signal did not cross |
| HuSHeM | ResNet50 | predictive entropy | 4 | 3 | one severity later |
| HuSHeM | ResNet50 | selective risk at 80% coverage | 4 | 3 | one severity later |
| HuSHeM | ResNet50 | APS coverage | — | 3 | signal did not cross |

An em dash means the signal did not cross; the protocol does not replace missing crossings with severity 5.

## Secondary and exploratory results

These analyses were prespecified as secondary or added as explicitly labeled diagnostics. They do not revise the primary thresholds or claim.

### Per-corruption thresholds

Applying the same threshold definitions separately to each corruption produced **0 early warnings among 58 comparisons with both crossings observed**; the signal did not cross in the other 54 signal/dataset/backbone/corruption combinations. This exploratory breakdown reinforces the averaged result, but its 112 comparisons are correlated and were not assigned a multiple-testing claim.

### Failure-detection AUROC

Per-sample error ranking remained useful at mild shift but generally weakened with severity. Raw-softmax failure-detection AUROC moved from severity 1 to severity 5 as follows: SMIDS ResNet50 **0.817→0.697**, SMIDS Xception **0.839→0.695**, SMIDS MobileNetV3-Large **0.807→0.657**, and HuSHeM ResNet50 **0.815→0.704**. The SMIDS ResNet50 ensemble declined **0.836→0.732**. Mahalanobis failure ranking was near chance for the SMIDS ResNet50/Xception paths, illustrating that a shift score is not automatically a good per-sample error score.

### Selective prediction at 80% coverage

Rejecting the 20% least-favored samples improved retained accuracy at severities 3–4, even though those scores did not supply a threshold-level early warning. Energy selection recovered **+3.6 to +5.2 points** across the four single-backbone paths; the five-model SMIDS ResNet50 ensemble recovered **+5.2/+5.4 points** at severities 3/4. This is a ranking result at a fixed review budget, not a validated clinical abstention policy.

### Conformal coverage and set size

APS behavior depended strongly on the dataset/backbone. By severity 5, coverage loss was **−0.112** for SMIDS ResNet50, **−0.198** for SMIDS MobileNetV3-Large, **−0.118** for SMIDS Xception, and only **−0.016** for HuSHeM ResNet50. Mean set size increased respectively by **+0.148**, **+0.276**, **+0.121**, and **+0.655** classes. Larger sets can preserve marginal coverage by becoming less decisive; neither coverage nor size alone is a deployment alarm.

### Temperature scaling under shift

At severity 5, mean raw→temperature-scaled ECE was **0.274→0.251** for SMIDS ResNet50, **0.397→0.384** for SMIDS MobileNetV3-Large, **0.181→0.178** for SMIDS Xception, and **0.184→0.198** for HuSHeM ResNet50. A calibrator fitted only on clean calibration data is therefore not a universal correction for shifted inputs.

## Clean sanity context

| Dataset | Backbone | Clean accuracy, mean ± sample SD | Clean macro-F1, mean ± sample SD |
|---|---|---:|---:|
| SMIDS | ResNet50 | 89.7% ± 0.7 | 89.7% ± 0.7 |
| SMIDS | Xception | 88.8% ± 2.2 | 88.9% ± 2.1 |
| SMIDS | MobileNetV3-Large | 88.9% ± 1.6 | 89.0% ± 1.6 |
| HuSHeM | ResNet50 | 86.5% ± 7.8 | 86.5% ± 7.9 |

HuSHeM's ±7.8-point clean-accuracy spread is especially exposed to small-*n* resolution: four test folds have 43 images and one has 44, so one changed classification moves a fold by 2.27–2.33 points, about 2.3 points. It mixes learned-model, fold-composition, and evaluation variability; it is not evidence of model instability alone. That caution connects to the inter-model/inter-evaluation variability theme in Thirumalaraju et al., not to a directly comparable effect size.

MobileNetV3-Large's clean SMIDS accuracy was descriptively close to ResNet50 (88.9% versus 89.7%). That makes the compact backbone relevant to on-device research where memory, latency, and power matter; it does not establish equivalence, clinical validity, or production readiness.

## Post-hoc threshold-relativity observation

The primary protocol intentionally remains frozen. A relative 2× threshold is a harder bar for a signal with a high clean baseline: for example, HuSHeM clean ECE was **0.132**, so its ECE threshold required exceeding both 0.264 and a 0.02 absolute increase. This is a post-hoc interpretation of why a crossing may remain missing, not a reason to redefine the result. Absolute-threshold protocols and AUROC-style alarm evaluations are appropriate future work and must be specified before a new evaluation.

## What was surprising, and what it means

The surprising result was lockstep rather than lead time: standard aggregate reliability curves visibly changed, but at the frozen decision points they did not alert earlier than accuracy. At the same time, per-sample ranking could still recover roughly four to five accuracy points at an 80% retention budget. Thus, “not an early alarm” and “not operationally useful” are different statements.

For low-cost diagnostic deployment, uncertainty dashboards alone should not be treated as degradation alarms. Paired-device validation on the same specimen, patient/source grouping, a held-out center, and ongoing performance monitoring remain necessary.

## Single weakest link

The study's weakest link is that the corruption ladder is simulated rather than calibrated against paired reference-microscope and smartphone captures of the same specimen. The datasets are public proxies without released patient/source linkage, but the central device-degradation claim is most directly limited by the absence of paired-device ground truth. Kromp remains excluded because its public release cannot support the required patient-level split.

Nothing here is clinical validation, a performance guarantee, or a recommended decision policy.

## Traceability and primary sources

- Primary grid: `results/metrics.csv`; primary crossings: `results/thresholds.csv`; frozen definitions and aggregation: `configs/analysis_protocol.yaml`; clean sanity matrix: `results/stage1_clean_summary.csv`.
- Thirumalaraju P et al., “Stability and reliability of artificial intelligence models in embryo selection for in vitro fertilization,” *Fertility and Sterility* 125(2), 277–286 (2026; online August 26, 2025), [doi:10.1016/j.fertnstert.2025.08.021](https://doi.org/10.1016/j.fertnstert.2025.08.021), [PubMed/PMC record](https://pubmed.ncbi.nlm.nih.gov/40876725/).
- Ovadia Y et al., “Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift,” NeurIPS 2019, [official proceedings](https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).
