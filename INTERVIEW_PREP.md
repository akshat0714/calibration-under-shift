# Interview mastery notes

This file is a defense guide, not a substitute for reading the source. Before a call, open every module named below, re-derive the equations by hand, and rehearse the five-minute walkthrough twice without notes.

## Five-minute walkthrough

**0:00–0:40 — question.** “The project asks whether confidence and other reliability signals fail before top-line accuracy when diagnostic image quality moves toward low-cost smartphone conditions. That matters because aggregate accuracy cannot tell an embryologist which individual image needs re-acquisition or review.”

**0:40–1:20 — why this lab.** “MD-nets framed device/lossy domain shift. Your replicate-model study then found Kendall's W around 0.35, critical errors around 15%, and greater error variance at another center. I treated ensemble disagreement, calibration, and selective prediction as possible early-warning layers rather than another accuracy benchmark.”

**1:20–2:15 — split and experiment.** “Models see only clean training images. Validation selects the checkpoint. A disjoint calibration split alone fits temperature/vector scaling and APS. I apply seven seeded corruptions only to held-out evaluation images at severity 1–5, cache logits, and reuse the same forward passes across methods. SMIDS is fixed 70/10/10/10; HuSHeM uses five outer folds. I did not claim a patient-level Kromp split because the release has no patient map.”

**2:15–3:15 — methods.** “I compare raw and temperature-scaled ECE/NLL/Brier, a five-model ensemble and MC dropout, APS coverage/set size, risk–coverage and failure-detection AUROC, energy and Mahalanobis shift scores, and quantitative Grad-CAM stability. The key clinical translation is abstention: rank uncertain images and ask whether reviewing a bounded fraction recovers lower risk.”

**3:15–4:20 — result.** Replace this paragraph only after the full grid: state clean mean±SD, the first accuracy-drop severity, the first reliability-signal severity, the early-warning gap, and the strongest per-corruption exception. Do not use the synthetic demo or one-seed pilot as evidence.

**4:20–5:00 — limitation and next step.** “These are public proxies and simulated shifts. The independent-channel Gaussian baseline is deliberately the least physical corruption, while the fixed-direction illumination proxy is also much simpler than a real phone ISP. I would validate severity against paired captures of the same specimen on reference and smartphone hardware, group by patient, hold out a center, and freeze calibration before target-center testing.”

## Module map

| Module | What it does | Design choice to defend |
|---|---|---|
| `src/data/splits.py` | fixed sample and group split manifests | repeated group-only candidates minimize size/class imbalance without leaking a patient |
| `src/data/datasets.py` | decodes manifest rows and injects evaluation corruption | constructor refuses corruption on `train`; file and in-memory manifests share validation |
| `src/models/build.py` | ResNet50/Xception/MobileNet factory and features | explicit dropout enables MC inference; one feature interface supports OOD and attribution |
| `src/train.py` | frozen-head then low-LR fine-tuning | macro-F1 checkpointing; clean train only; run provenance saved |
| `src/shifts/corruptions.py` | seven deterministic device proxies | fixed images/seeds pair the ladder; Poisson draws remain condition-specific; no corrupted copies on disk |
| `src/metrics/calibration.py` | ECE/adaptive ECE/Brier/NLL | ECE is top-label and bin-sensitive, so it is never the sole reliability metric |
| `src/metrics/selective.py` | risk–coverage, AURC, failure AUROC | uncertainty is evaluated by ranking failures, not only by its mean |
| `src/uncertainty/temperature.py` | scalar and vector scaling | fit function rejects non-calibration roles |
| `src/uncertainty/ensembles.py` | entropy/MI decomposition | MI is predictive entropy minus expected member entropy: disagreement, not total ambiguity |
| `src/uncertainty/conformal.py` | deterministic APS | finite-sample corrected quantile; guarantee needs exchangeability |
| `src/uncertainty/ood_scores.py` | MSP, energy, Mahalanobis | Mahalanobis parameters use clean train features, never test features |
| `experiments/run_grid.py` | cached full factorial evaluation | clean calibration is fitted once and transferred unchanged across shift |
| `experiments/analyze.py` | applies prespecified thresholds | average corruptions within a seed before treating seeds as replicates |
| `src/attribution/*` | Grad-CAM/++ and stability | fixed clean class target makes heatmaps comparable across severity |

## Derivations to be able to write

### Expected calibration error

For confidence bins \(B_m\),

\[
\operatorname{ECE}=\sum_{m=1}^{M}\frac{|B_m|}{n}
\left|\operatorname{acc}(B_m)-\operatorname{conf}(B_m)\right|.
\]

Equal-width ECE gives confidence intervals identical boundaries but can have sparse high-confidence bins. Adaptive ECE sorts by confidence and forms roughly equal-mass bins, trading fixed boundaries for lower variance in occupancy. Both are estimators, not intrinsic properties of the model.

### Why scalar temperature cannot change accuracy

Temperature scaling maps logits \(z_k\) to \(z_k/T\) for \(T>0\). For any classes \(i,j\), \(z_i>z_j\) if and only if \(z_i/T>z_j/T\); the argmax is unchanged. Softmax sharpness and NLL can change, but class predictions and accuracy cannot. Vector scaling can change argmax because each class has a different scale and bias.

### Split-conformal coverage

Compute calibration nonconformity scores \(s_1,\ldots,s_n\) and use rank

\[
k=\left\lceil(n+1)(1-\alpha)\right\rceil,
\]

clipped at \(n\). The threshold is the \(k\)-th order statistic. Under exchangeability of the calibration example and a new test example, the new score's rank is uniform, giving marginal coverage at least \(1-\alpha\) up to the finite-sample correction. Device, center, or prevalence shift breaks exchangeability; coverage can fall and the nominal statement no longer applies.

### Ensemble mutual information

With ensemble member probabilities \(p_m(y\mid x)\) and their mean \(\bar p\),

\[
\operatorname{MI}=H(\bar p)-\frac{1}{M}\sum_m H(p_m).
\]

The first term is total predictive uncertainty. The second is average member ambiguity. Their difference is epistemic disagreement and is zero when all members make identical distributions.

### Energy score

The implemented energy is

\[
E(x)=-T\log\sum_k\exp(z_k(x)/T).
\]

In-distribution classifiers often produce a large logit and therefore more negative energy; larger (less negative) values are treated as more OOD-like. It is a score whose orientation must be checked, not a calibrated probability.

## Likely questions and crisp answers

**Walk me through the splits. Patient-level?**

SMIDS is a fixed stratified 70/10/10/10 image split because its release has no patient/source-field IDs. HuSHeM uses five stratified outer folds, but its 15-patient linkage is also unreleased, so I call it image-level and flag possible optimism. Kromp is not run: the public release has no patient map and filename prefixes disagree with the stated patient count. If an author map is supplied, the code creates a true group-disjoint 60/15/10/15 manifest.

**Where did calibration data come from?**

It is carved out before training and is disjoint from train, validation, and test. The exact SMIDS counts are 300 calibration images; each HuSHeM fold has 22. The fitting APIs reject any split label other than `calibration`.

**What if you fit temperature on test?**

That leaks test labels into model selection, biases NLL/ECE downward, and destroys the held-out estimate. Even though model weights do not change, the decision system has been tuned on test outcomes.

**Why adaptive ECE too?**

Equal-width bins can be nearly empty when confidence concentrates near one. Equal-mass bins stabilize occupancy and expose a different failure mode. Neither is sufficient alone, which is why I also report proper scoring rules, reliability data, and selective metrics.

**A model is 90% accurate but poorly calibrated. Which is worse?**

It depends on how confidence is used. If probabilities trigger embryo ranking, automatic acceptance, or a review threshold, miscalibration can turn a 10% error rate into misleading assurance and poor triage. If only hard labels are used, discrimination may dominate. I would specify the decision and asymmetric costs before declaring one metric “worse.”

**What is least physically realistic?**

The independent per-channel Gaussian-noise baseline, deliberately. I retained the ImageNet-C convention for benchmark comparability, but independent RGB residuals are not a faithful post-demosaic phone-sensor model. I therefore redesigned shot noise as the more physically motivated counterpart: it samples a Poisson perturbation from luminance and shares that residual across RGB channels, removing implausible rainbow speckle while preserving color structure. The illumination proxy is simplified too: at the fixed seed it darkens midtones and suppresses red while increasing green and blue, creating a green/teal cast. A real phone ISP also includes raw Bayer sampling, demosaicing, local denoising, sharpening, tone mapping, and auto-exposure. I would estimate device statistics from paired captures and test whether synthetic and real shift induce the same metric and attribution ordering.

**How would this behave on Embryoscope-versus-smartphone data?**

I expect optical blur, color/illumination, and resampling to interact rather than act independently. The current clean-fitted temperature may fail first because it cannot adapt input-conditional calibration; energy or ensemble disagreement may rank bad images better. That is a hypothesis, not a result. Paired specimens and patient/site grouping are required.

**What would a clinic do with a conformal set?**

A singleton may permit the normal workflow; a multi-class or empty/atypical result can trigger image re-acquisition or human review. The action threshold and cost must be prospectively defined, and marginal 90% coverage is not a per-patient safety guarantee.

**Why not augment training with the corruptions?**

The primary experiment isolates monitoring under unseen deployment shift. Corruption augmentation would change the scientific question to robustness training. It is an appropriate follow-up arm after the warning-signal baseline is measured.

**What did you build versus generate?**

Give the literal answer. A defensible version is: “I used AI assistance for scaffolding, test generation, and review, but I validated every invariant, traced every data caveat to primary metadata, ran the code, and can derive the methods. The Kromp blocker is an example where I rejected the requested automation because the source data could not support the claim.” Never imply hand authorship of generated code.

## Questions for Manoj and Prudhvi

1. Your replicate-model study found similar AUCs but divergent ranking and Grad-CAM strategies. Which operational failure would you prioritize first: rank instability, critical-error control, or calibrated abstention?
2. Do you have an internal patient/image map for the public Kromp release, or would you recommend a different embryo dataset for patient-grouped external work?
3. For paired clinical-versus-smartphone images, which device statistics best track the lab's 4→1 quality scale: optical resolution, signal-to-noise, compression, color, or a learned domain distance?
4. When onboarding a new fertility center, does the lab currently reserve a site-specific calibration set, use domain adaptation, or require a fully untouched center-level acceptance test?
