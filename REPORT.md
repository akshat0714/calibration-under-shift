# Calibration Under Shift: Protocol and Interim Research Report

**Evidence status (2026-09-01):** the implementation, public-release audits, fixed SMIDS/HuSHeM manifests, synthetic end-to-end engineering check, and clean-test sanity matrix are complete. The full shifted evaluation grid is still pending, so the clean results below are interim checks rather than the prespecified headline finding; no clinically oriented conclusion is claimed.

## Prespecified hypothesis

We predict that calibration error and predictive uncertainty will degrade at lower simulated device-corruption severities than classification accuracy. We will test this by measuring accuracy, macro-F1, ECE, adaptive ECE, NLL, Brier score, uncertainty, selective risk, and conformal coverage across fixed severity ladders applied only at evaluation time. The hypothesis is falsified if prespecified reliability thresholds are not crossed before a five-percentage-point accuracy drop, consistently across datasets and seeds.

> Status: implementation and clean-test sanity matrix complete; full shifted evaluation pending. The hypothesis and thresholds were frozen before the full seeded grid. This is a local prespecification, not an externally registered protocol.

## Motivation

Deployment failure can be quiet. A diagnostic classifier may preserve enough decisions to keep aggregate accuracy apparently stable while its probabilities become overconfident, its errors become harder to identify, or independently trained models cease to agree. Those changes matter when confidence determines whether an image is accepted, re-acquired, or sent for human review.

This question is closely aligned with two threads in the Shafiee Lab's work. Kanakasabapathy et al. developed adaptive adversarial networks for lossy and domain-shifted medical-image datasets, explicitly spanning clinical and smartphone acquisition quality [1]. Thirumalaraju et al. trained 50 replicate embryo-selection networks and reported weak rank agreement (mean Kendall's W 0.3571 at MGH and 0.3410 at Cornell), critical error rates around 15%, and a cross-center error-variance increase of 46.07 percentage-points squared despite similar architectures and average performance [2]. This study asks whether calibration, ensemble disagreement, selective risk, conformal sets, input-shift scores, and attribution drift can serve as per-image warnings along a controlled device-quality axis.

## Data and task definitions

SMIDS is the primary executable public-data path: 3,000 sperm images in normal (1,021), abnormal (1,005), and non-sperm (974) classes [3]. The release was split once, stratified with seed 2025, into 70% train, 10% validation, 10% calibration, and 10% test. HuSHeM supplies 216 sperm-head images in normal (54), tapered (53), pyriform (57), and amorphous (52) classes and uses five stratified outer folds, each with disjoint training, validation, calibration, and test roles [4].

The originally proposed hero dataset, Kromp et al.'s 2,344-image blastocyst release [5], cannot currently support the promised patient-level split. All 2,344 images decode, but the local audit found 15 exact-byte duplicate groups, including cross-prefix duplicates. The public archive has no patient identifier, while filename prefixes produce 851 groups rather than the paper's 837 patients. It also contains a conflicting duplicate expansion label for `838_02.png` and no label for `846_01.png`. The preparation code therefore requires an author-verified patient map and explicit duplicate/label resolution; it does not relabel filename prefixes as patients.

All locally acquired SMIDS and HuSHeM files were decoded. Neither release has corrupt or byte-identical duplicate images. SMIDS contains 1,914 distinct dimensions and 480 PNG payloads named `.bmp`; HuSHeM contains six images that differ from its nominal 131×131 size. These anomalies are handled by content-aware decoding and deterministic resizing and are retained in the audit record.

## Methods

### Leakage controls and model training

Models train on clean training images only. Standard light geometry and color augmentation are restricted to that role. Validation macro-F1 selects checkpoints; validation data do not fit post-hoc calibration. A fourth, disjoint calibration role is the only input accepted by temperature scaling, vector scaling, and APS conformal fitting. Test images never select a model, fit a scaler, or determine a conformal threshold.

The planned matrix uses ImageNet-pretrained ResNet50 as the primary backbone across five seeds, Xception across three seeds to mirror MD-nets, and MobileNetV3-Large across three seeds as an on-device comparison. Each model has dropout immediately before its classification head. Transfer learning freezes the feature extractor for 4–5 head epochs, then fine-tunes with AdamW, a lower learning rate, cosine decay, and early stopping on validation macro-F1. Each run records the resolved YAML, seed, Git revision, environment, epoch curves, and checkpoint.

### Device and population shift

Seven deterministic corruption families are applied after image decoding and before the fixed evaluation transform: defocus blur, motion blur, Gaussian noise, luminance-correlated Poisson shot noise, JPEG compression, down–up resampling, and combined gamma/white-balance shift. Each has five parameters fixed before model results are examined. Blur sigma and motion-kernel length are defined at a 224-pixel reference short side and scaled to native image geometry, preventing SMIDS dimension variation from changing relative severity. Gaussian noise deliberately retains the ImageNet-C independent per-pixel/channel additive convention for benchmark comparability and is the least physically realistic corruption by design. Shot noise is its more physically motivated counterpart: it is sampled from image luminance and its residual is shared across RGB channels, a post-demosaic approximation that avoids implausible independent color speckle. The same held-out images and seed are used at every severity; the seed fixes motion angle and illumination direction, while Poisson draws are reproducible per condition but are not coupled across count scales. At the fixed protocol seed of 1729, the illumination transformation uses gamma values above one to darken midtones and suppresses red while increasing green and blue, producing a growing green/teal cast. A class-prior resampling condition doubles the SMIDS abnormal-class weight as a population-prevalence proxy; it is analyzed separately from the device-corruption average.

These transformations are sensitivity analyses, not a physical calibration of a particular phone. The critical validation experiment would compare their image statistics and metric ordering against paired captures of the same specimen on reference and smartphone hardware.

### Reliability and uncertainty

Classification metrics are accuracy, macro-F1, per-class recall, and one-vs-rest AUROC. Calibration metrics are 15-bin top-label ECE, equal-mass adaptive ECE, multiclass Brier score, and NLL [6]. Scalar temperature scaling is optimized with LBFGS on clean calibration logits; because division by a positive scalar preserves logit order, it cannot change predicted classes or accuracy. Vector scaling is included as a more flexible, less data-efficient comparison.

Five ResNet seeds form the deep ensemble. Mean softmax gives the ensemble prediction; predictive entropy measures total uncertainty and mutual information isolates disagreement. Thirty-pass MC dropout is the lower-cost comparison. Selective prediction ranks samples from least to most uncertain and reports the risk–coverage curve, AURC, risk at 80% coverage, and AUROC for detecting an incorrect prediction.

APS split conformal prediction uses α=0.1 and a finite-sample corrected calibration quantile [7]. Empirical coverage and mean prediction-set size are recorded. Its marginal coverage guarantee relies on exchangeability between calibration and test examples; degradation under device or population shift is therefore expected to appear as coverage loss, larger sets, or both.

Input-shift baselines are maximum-softmax uncertainty, log-sum-exp energy, and the minimum class-conditional Mahalanobis distance in penultimate feature space. Class means and a shared regularized covariance are fitted on clean training features. Grad-CAM adds a decision-strategy view, with a tested Grad-CAM++ option available for comparison; class-stratified panels are qualitative, while full-test clean and shifted maps are compared with flattened Spearman rank correlation and IoU between exact top-20% saliency masks. Constant non-localizing maps are recorded as undefined rather than perfectly stable.

### Prespecified analysis

For each severity, device corruptions are averaged equally within a seed before means and sample standard deviations are computed across seeds or folds. Accuracy degradation is the first severity more than 0.05 below clean accuracy. ECE is degraded only when it exceeds both twice its clean value and a 0.02 absolute increase; entropy similarly requires a two-fold and 0.05-nat increase; selective risk requires a two-fold and 0.02 absolute increase. Conformal coverage is degraded below 0.85. A reliability signal is “earlier” only when its first crossing has a lower severity than the accuracy crossing. Missing crossings remain missing.

## Interim results

Checkpoint 2 verified the multi-seed/fold clean-test sanity matrix. These clean-only estimates do not answer the shifted-grid hypothesis and must not be promoted into the headline figure.

HuSHeM ResNet50 clean accuracy was 86.5% ± 7.8 percentage points across five outer folds (mean ± sample SD). The apparently large fold spread is strongly exposed to small-*n* granularity: the test folds contain 43 images (44 in fold 0), so one additional misclassification moves fold accuracy by 2.27–2.33 points, or about 2.3 points. It should therefore be read as a mixture of learned-model variability, fold-composition variability, and coarse evaluation resolution—not model instability alone. This is a small-data illustration of the broader inter-model/inter-evaluation variability theme emphasized by Thirumalaraju et al. [2], not a directly comparable effect size.

On SMIDS, MobileNetV3-Large and ResNet50 were close on clean accuracy (88.9% ± 1.6 versus 89.7% ± 0.7) and macro-F1 (89.0% ± 1.6 versus 89.7% ± 0.7). This descriptive near-parity makes the compact backbone relevant to research on memory-, latency-, and power-constrained on-device diagnostic deployment, while proving neither statistical equivalence nor deployment readiness.

The remaining verified results are infrastructural:

1. SMIDS and HuSHeM archive hashes match their primary publisher records, all expected images decode, and class counts reproduce the releases.
2. Fixed manifests are present for SMIDS and all five HuSHeM folds, with a distinct calibration role.
3. The synthetic workflow completes training, clean-calibration fitting, corrupted inference, MC dropout, APS, OOD scoring, tidy metric output, threshold analysis, Grad-CAM stability, and figure generation.
4. The automated suite covers deterministic shifts, monotonic endpoint distortion, hand-computable calibration and selective metrics, exchangeable conformal coverage, calibration-only guards, split leakage, model interfaces, attribution, plotting, and end-to-end checkpoint inference.

The prespecified finding and headline figure will be added only after the full seeded shifted grid produces `results/metrics.csv`, the analysis produces `results/thresholds.csv`, and the fresh-clone reproduction drill matches those artifacts.

## Limitations

The data are public proxies, not the lab's clinical cohorts. SMIDS has no released patient or source-field lineage, and HuSHeM identifies 15 donors in aggregate but releases no image-to-donor map; image-level splitting may therefore overstate generalization. Kromp is currently unusable for the required patient-level design without author metadata. HuSHeM is small, and fold estimates will be noisy.

The shifts are simulated. Gaussian blur, ideal line-kernel motion, luminance-domain Poisson statistics, JPEG quality, bilinear resampling, and global gamma/channel gains do not capture a phone lens's modulation transfer function, raw Bayer sampling, demosaicing, denoising, sharpening, local tone mapping, or device-dependent color pipeline. The independent-channel Gaussian baseline is deliberately the least physically realistic corruption; the global green/teal white-balance and gamma transformation is also a simplified fixed-direction proxy rather than a phone ISP model. No severity is asserted to equal a clinical microscope or smartphone system.

Calibration error is estimator-dependent, particularly at small n; ECE can hide within-bin errors and adaptive ECE changes bin boundaries across conditions. Temperature scaling corrects a global sharpness error but cannot repair class-conditional or input-dependent miscalibration. APS coverage is marginal, not per-class or per-patient, and exchangeability is exactly what distribution shift threatens. Grad-CAM is a coarse post-hoc attribution and is not a causal account of a model decision.

Nothing in this study constitutes clinical validation, a performance guarantee, or a recommended decision policy.

## Next steps

The immediate next step is to run the prespecified shifted evaluation and analysis grid from the verified clean-test checkpoints. Kromp should remain blocked until its authors provide a patient map and resolve annotation discrepancies. The most informative lab-data extension is a paired acquisition experiment on the same embryo or sperm sample: reference microscope versus smartphone hardware, with patients grouped and a center held out. Corruption parameters could then be fitted to measured image statistics, while calibration and alert thresholds remain frozen before target-center evaluation.

## References

1. Kanakasabapathy MK et al. *Nature Biomedical Engineering* 5, 571–585 (2021). [doi:10.1038/s41551-021-00733-w](https://doi.org/10.1038/s41551-021-00733-w).
2. Thirumalaraju P et al. *Fertility and Sterility* 125(2), 277–286 (2026; online 2025). [doi:10.1016/j.fertnstert.2025.08.021](https://doi.org/10.1016/j.fertnstert.2025.08.021).
3. Takidin H, Ceylan HI, Kusetogullari H. *Sperm Morphology Image Data Set (SMIDS)*, Mendeley Data v1 (2022). [doi:10.17632/6xvdhc9fyb.1](https://doi.org/10.17632/6xvdhc9fyb.1).
4. Shaker M, Monadjemi SA. *Human Sperm Head Morphology dataset (HuSHeM)*, Mendeley Data v3 (2018). [doi:10.17632/tt3yj2pf38.3](https://doi.org/10.17632/tt3yj2pf38.3).
5. Kromp F et al. *Scientific Data* 10, 271 (2023). [doi:10.1038/s41597-023-02182-3](https://doi.org/10.1038/s41597-023-02182-3).
6. Guo C et al. ICML, PMLR 70:1321–1330 (2017). [Proceedings](https://proceedings.mlr.press/v70/guo17a.html).
7. Angelopoulos AN, Bates S. *Foundations and Trends in Machine Learning* 16(4), 494–591 (2023). [doi:10.1561/2200000101](https://doi.org/10.1561/2200000101).
