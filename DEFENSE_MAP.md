# Defense map

Use this as an index, not a script. For every row, be able to open the file, point to the implementation, state the tradeoff, and name the failure the tests are intended to prevent. Actual scientific results come only from `results/metrics.csv`, `results/thresholds.csv`, and the Stage 1 clean summaries; demo and pilot artifacts are engineering checks only.

**Primary statement to defend:** under the prespecified protocol, no reliability signal crossed before the five-point accuracy-drop threshold in any of the 10 comparisons where both crossings were observed; six further signals never crossed. The early-warning hypothesis was not supported at these thresholds on these datasets. This does not prove that uncertainty methods are useless.

## Protocol, configuration, and entry points

| File | Decision to defend in one line |
|---|---|
| `run.sh` | The no-argument `--eval-only` route performs the complete isolated released-checkpoint reproduction; config-specific and demo routes remain explicit so fixtures cannot replace scientific results. |
| `requirements.txt` | Exact runtime pins make the released-checkpoint path reproducible from the declared Python environment rather than depending on a developer machine. |
| `pyproject.toml` | Pytest and Ruff configuration is versioned with the code so local and CI quality gates use the same rules. |
| `.github/workflows/ci.yml` | CI exercises deterministic unit/smoke behavior without downloading private or large raw data; real-result reproduction is a separate released-checkpoint drill. |
| `.gitignore` | Raw datasets, checkpoints, caches, and transient outputs stay out of Git; only auditable summaries, provenance, and publishable artifacts belong in history. |
| `CONTRIBUTING.md` | Contributions must preserve split/calibration guards, provenance, tests, and evidence boundaries rather than optimizing only for a passing metric. |
| `LICENSE` | Repository code is GPL-3.0-only; dataset thumbnails and transformations retain their source CC BY 4.0 attribution instead of being relicensed as code. |
| `configs/analysis_protocol.yaml` | The corruption average, replicate aggregation order, four degradation thresholds, and missing-crossing rule were frozen before the full grid and are not adjusted after seeing the null. |
| `configs/smids_resnet50.yaml` | ResNet50 is the primary SMIDS path and five seeds are retained as independent training runs and the deep-ensemble members. |
| `configs/smids_xception.yaml` | Xception supplies the architecture connection to the lab's device/domain-shift work and uses three seeds without being promoted over the primary backbone. |
| `configs/smids_mobilenetv3.yaml` | MobileNetV3-Large is the compact on-device research comparison; descriptive near-parity is not called statistical equivalence. |
| `configs/hushem_resnet50.yaml` | HuSHeM uses five image-level outer folds because donor linkage is described but not released; every fold retains disjoint train/validation/calibration/test roles. |
| `configs/kromp_resnet50.yaml` | Kromp's model configuration is dormant until an author-verified patient map and label resolution exist; having a config is not authorization to train. |
| `configs/kromp_xception.yaml` | The Kromp Xception option stays blocked by the same data-integrity gate rather than bypassing it through another backbone. |
| `configs/kromp_mobilenetv3.yaml` | The Kromp compact-model option stays blocked; no filename-prefix group is relabeled as a patient. |
| `configs/demo.yaml` | The tiny synthetic task proves plumbing only and is kept structurally distinct from scientific dataset/backbone configurations. |

## Data, splits, and shift engine

| File | Decision to defend in one line |
|---|---|
| `scripts/download_data.sh` | Archives resume into `.part` files, are atomically renamed, checksum-verified, and extracted with an explicit RAR-tool preflight; raw data are never redistributed. |
| `scripts/release_checkpoints.py` | The release URL, archive size/hash, internal manifest, safe extraction, and exact ordered 16-member selection are pinned and verified once per full run. |
| `scripts/make_demo_data.py` | Synthetic images are deterministic and exist only to exercise CI/smoke plumbing without downloading a scientific dataset. |
| `src/data/prepare.py` | Preparation decodes by content, maps labels explicitly, writes deterministic metadata/splits, and fails Kromp on conflicting labels or absent patient linkage instead of inventing a defensible-looking split. |
| `src/data/audit.py` | Audits count classes, dimensions, decode failures, and byte duplicates before modeling and emit both machine-readable evidence and visual samples. |
| `src/data/kromp_audit.py` | Kromp gets release-specific checks for duplicate images, conflicting/missing annotations, and filename-prefix/patient-count disagreement. |
| `src/data/splits.py` | SMIDS is fixed image-stratified, HuSHeM is fixed stratified outer-fold CV, and genuinely grouped splitting requires an actual group field; train/validation/calibration/test disjointness is asserted. |
| `src/data/datasets.py` | Corruptions are constructor-level evaluation behavior and are refused for the training role, making clean-only training an enforceable invariant. |
| `src/data/transforms.py` | Model-specific image size/interpolation is resolved from config, while ImageNet normalization remains consistent with the pretrained weights. |
| `src/shifts/severity.py` | One versioned table defines all seven ordinal ladders and a protocol digest so cached logits and reported physics cannot drift independently. |
| `src/shifts/corruptions.py` | Deterministic corruptions operate at decoded native geometry; Gaussian noise preserves ImageNet-C independent-channel comparability, shot noise uses a shared luminance-derived residual, and illumination fixes a per-image gamma/white-balance direction across severity. |
| `scripts/generate_corruption_grid.py` | The displayed ladder uses a fixed held-out manifest row and seed, with a sidecar recording the exact source and protocol rather than a hand-picked undocumented image. |
| `data/metadata/smids.csv` | The checksum-audited 3,000-image inventory is the source of truth for SMIDS paths and labels. |
| `data/metadata/hushem.csv` | The 216-image inventory preserves published classes and observed dimension anomalies rather than silently normalizing the audit record. |
| `data/metadata/kromp_release.csv` | Kromp metadata records what the release actually contains without asserting patient identity or resolving contradictory labels by guesswork. |
| `data/splits/smids.csv` | The seed-fixed 70/10/10/10 image split reserves a calibration role before training; absent patient/source IDs remain an explicit limitation. |
| `data/splits/hushem.csv` | Five fixed outer folds provide disjoint roles within fold, while the missing image-to-donor map prevents a patient-level claim. |

## Models and training

| File | Decision to defend in one line |
|---|---|
| `src/models/build.py` | A common feature/logit interface wraps ResNet50, Xception, and MobileNetV3-Large; explicit pre-head dropout supports MC inference and penultimate features support OOD/attribution analyses. |
| `src/train.py` | Training is clean-only, uses frozen-head then lower-rate full fine-tuning, selects checkpoints by validation macro-F1, and records resolved config, seed, Git revision, curves, and environment. |
| `experiments/train_matrix.py` | The matrix enumerates only the approved seed/fold members and appends an identity-rich registry so interrupted runs can resume without conflating models. |
| `experiments/full_retrain.py` | The public full-retrain flag is CUDA-only, skips already completed logical members with existing checkpoints, atomically updates its registry after each run, and enforces the clean sanity gates at the end. |
| `experiments/evaluate_clean_matrix.py` | The clean gate evaluates every registered member, summarizes mean ± sample SD, rejects chance-level runs, and enforces the SMIDS/HuSHeM sanity thresholds before shift analysis. |
| `RUN_ON_GCP.md` | GPU setup, resumability, artifact packaging, and shutdown are written as an auditable operational procedure; it does not authorize CPU fallback or service-account keys. |

## Evaluation, calibration, uncertainty, and decision support

| File | Decision to defend in one line |
|---|---|
| `src/evaluate.py` | Checkpoint inference is cached atomically by model/data/protocol identity, exposes logits/features/MC passes, and keeps evaluation device selection separate from scientific definitions. |
| `src/utils.py` | Config loading, full Git revision capture, JSON serialization, and environment metadata are centralized so every run records the same provenance semantics. |
| `experiments/run_grid.py` | Clean calibration fits happen once per checkpoint and are frozen across clean/corrupted test cells; all methods reuse cached forward passes and write tidy rows with portable provenance. |
| `experiments/run_stage2_matrix.py` | The canonical 16-member matrix, exactly five SMIDS ResNet50 ensemble members, expected row counts, unique tidy keys, detail files, and one Git revision are validated before `metrics.csv` is written atomically. |
| `experiments/reproduce_release.py` | The one-command orchestrator isolates outputs under `results/reproduction`, refuses silent CPU use, excludes Kromp, regenerates Stages 2–5, and stops on the first failed gate. |
| `experiments/verify_reproduction.py` | The final verifier compares every tidy identity/value to the committed reference, preserves the 0/10 and 6/16 null, rechecks split/calibration guards, and verifies all figure hashes and portable paths. |
| `src/metrics/classification.py` | Accuracy is accompanied by macro-F1, class recall, and macro one-vs-rest AUROC so imbalance and class-specific failures are not hidden. |
| `src/metrics/calibration.py` | ECE, adaptive ECE, Brier, and NLL are all reported because binned calibration estimates are not intrinsic or sufficient alone. |
| `src/metrics/selective.py` | Risk–coverage, AURC, fixed 80%-coverage risk, and failure-detection AUROC evaluate ranking usefulness independently of mean uncertainty. |
| `src/uncertainty/temperature.py` | Scalar/vector scaling APIs accept calibration-labeled data only; scalar temperature preserves argmax while vector scaling may trade flexibility for small-calibration-set variance. |
| `src/uncertainty/ensembles.py` | The five independently trained ResNet50 members yield mean probabilities, predictive entropy, expected entropy, mutual information, and variation ratio; ensemble benefits are not compared as equal-cost single models. |
| `src/uncertainty/mc_dropout.py` | Thirty stochastic passes activate only dropout at inference and provide a lower-cost approximate disagreement baseline without updating weights or batch-normalization state. |
| `src/uncertainty/conformal.py` | APS uses the finite-sample corrected clean-calibration quantile at α=0.1; its marginal guarantee is stated only under exchangeability. |
| `src/uncertainty/ood_scores.py` | MSP/energy orientations are explicit, while Mahalanobis means/covariance are fitted on clean training features only and never on test features. |

## Analysis, attribution, and figures

| File | Decision to defend in one line |
|---|---|
| `experiments/analyze.py` | Device corruptions are averaged within seed/fold before replicate means/SDs; a signal is early only for a strictly lower crossing severity and missing crossings stay missing. |
| `experiments/checkpoint3.py` | The approved checkpoint audit revalidates the complete grid/provenance, materializes the primary table and labeled secondary tables, and derives finding text from data rather than prose constants. |
| `src/attribution/gradcam.py` | Grad-CAM and Grad-CAM++ share the tested hook/target interface; fixed clean-class targets make maps comparable even if the shifted prediction changes. |
| `src/attribution/stability.py` | Clean-to-shift heatmaps are compared with flattened Spearman correlation and exact top-20% mask IoU; constant non-localizing maps are undefined rather than called stable. |
| `experiments/run_attribution.py` | A fixed manifest subset and severities 0/2/4 produce confidence-labeled qualitative panels and full-test quantitative stability from real checkpoints, with run provenance. |
| `src/viz/figures.py` | Reusable plotting primitives and the synthetic/demo figure path share a consistent accessible style without being treated as scientific output. |
| `experiments/generate_final_figures.py` | F1–F7 are regenerated from validated real metrics, paired detail geometry, and attribution provenance; the headline uses an unshaded shared severity axis and every panel emits hashed tidy source data. |
| `experiments/generate_diagnostics.py` | Small diagnostic panels are generated from a real selected checkpoint for manual sanity checking, not promoted into the multi-seed primary result. |
| `notebooks/01_walkthrough.ipynb` | The narrative notebook reads committed result artifacts and keeps its fixture/demo route explicitly opt-in; it does not train or silently fall back. |

## Tests: what each file protects

| File | Decision to defend in one line |
|---|---|
| `tests/test_shifts.py` | Every corruption is deterministic, valid-range, identity at severity 0, and endpoint distortion is monotonic enough for the frozen ladder. |
| `tests/test_corruption_grid_script.py` | The published ladder and provenance sidecar regenerate from the fixed manifest row and seed. |
| `tests/test_data.py` | Label mappings, split disjointness, calibration role, decoding, and Kromp blockers fail loudly. |
| `tests/test_models.py` | All backbones obey the same logit/feature/dropout contract required by downstream methods. |
| `tests/test_temperature.py` | Calibration-role guards, positive scalar temperature, and expected calibration behavior are checked on controlled logits. |
| `tests/test_conformal.py` | APS quantiles/set construction match hand-computable cases and exchangeable synthetic coverage behaves as expected. |
| `tests/test_metrics.py` | Calibration, classification, selective, and AUROC routines match small hand-computable examples and orientation conventions. |
| `tests/test_ood.py` | Energy and Mahalanobis scoring use the documented sign/orientation and training-feature fit. |
| `tests/test_attribution.py` | Grad-CAM/++ shapes, target handling, mask thresholds, and undefined constant-map stability are enforced. |
| `tests/test_evaluation_cache.py` | Cache identity includes checkpoint, manifest, protocol, condition, and MC-pass distinctions; interrupted writes cannot masquerade as valid caches. |
| `tests/test_grid_protocol.py` | Full-grid methods/conditions, calibration-only fitting, population-shift separation, and ensemble scope are enforced. |
| `tests/test_grid_details.py` | Per-run detail JSONs retain complete, portable provenance and agree with tidy output identities. |
| `tests/test_stage2_matrix.py` | Canonical member counts, one revision, exact row count, atomic assembly, and duplicate rejection protect the scientific grid. |
| `tests/test_analysis.py` | Within-seed aggregation, strict crossing definitions, absolute-plus-relative thresholds, and missing-crossing semantics match the frozen protocol. |
| `tests/test_checkpoint3.py` | Main/secondary table derivation and honest finding text are tested so the primary null cannot be changed by prose editing. |
| `tests/test_clean_matrix.py` | Registry shape, member completeness, clean summary statistics, chance checks, and sanity gates are tested. |
| `tests/test_train_matrix.py` | Seed/fold enumeration and registry resume behavior prevent accidental pilot inclusion or duplicate members. |
| `tests/test_release_checkpoints.py` | Archive hash/size, safe extraction, pinned manifests, exact 16-member selection, and run-script dispatch prevent checkpoint substitution, path traversal, or a partial default evaluation. |
| `tests/test_reproduction.py` | Isolated-output reset, accelerator refusal, metric tolerance, frozen-null, figure-hash, source-guard, and demo-severity regressions protect the public one-command path. |
| `tests/test_notebook.py` | The notebook defaults to real results, requires explicit demo opt-in plus paths, honors an isolated results root, and CI never overwrites canonical metrics. |
| `tests/test_figures.py` | Figure generation is exercised on tidy fixture data and rejects missing required methods/metrics rather than drawing misleading empty panels. |
| `tests/test_final_figures.py` | Final F1–F7 inputs are cross-checked against canonical metrics/detail/attribution artifacts, including hash and scalar agreement, before publication panels can be written. |
| `tests/test_smoke_pipeline.py` | The synthetic path exercises train→evaluate→analyze→figure plumbing only; its outputs never satisfy real-grid identity checks. |

## Evidence and narrative files

| File | Decision to defend in one line |
|---|---|
| `results/metrics.csv` | This is the canonical 45,540-row real evaluation table; all reported shifted numbers must be reproducible from its tidy cells and frozen aggregation. |
| `results/thresholds.csv` | These 16 rows are the immutable output of the frozen primary decision rule; 10 paired crossings were same/later and six signals did not cross. |
| `results/stage1_clean_summary.csv` | Clean mean ± sample SD across the approved seeds/folds is the sole source for the clean sanity table. |
| `results/checkpoint_registry-stage1.csv` | Exactly 16 approved members link config/seed/fold/run identities to released checkpoints; synthetic/pilot runs are not members. |
| `results/data_audit.md` | Dataset counts, dimensions, duplicates, and Kromp release defects are reported before results and remain independent of model performance. |
| `results/audits/smids.json` | Machine-readable SMIDS audit makes the 3,000-image completeness and no-duplicate claims checkable. |
| `results/audits/hushem.json` | Machine-readable HuSHeM audit makes the 216-image counts and dimension exceptions checkable. |
| `results/audits/kromp.json` | Machine-readable Kromp defects justify exclusion instead of being hidden in a prose footnote. |
| `results/figures/corruption_grid.png` + `.json` | The visual ladder and its source/protocol sidecar are a single evidence pair; the grid is a sensitivity design, not a physical device calibration. |
| `results/figures/smids_samples.png` | The sample grid documents decoded SMIDS content and class appearance; it is an audit thumbnail, not a training or result figure. |
| `results/figures/hushem_samples.png` | The sample grid documents HuSHeM class content and dimension handling without implying patient independence. |
| `results/figures/kromp_samples.png` | The sample grid documents the audited release even though modeling is blocked; visual availability does not solve missing patient linkage. |
| `DATASETS.md` | Primary dataset citations, checksums, licenses, and release caveats are kept next to reproduction guidance; public availability does not erase attribution duties. |
| `DELIVERY_CHECKLIST.md` | The local checklist is a verification aid, not evidence that a gate passed; every checked item still requires command or artifact evidence. |
| `STAGE6_VERIFICATION.md` | Cold/warm fresh-clone commands, revision, timing boundary, and exact gate outputs record what was actually reproduced without inventing a T4 runtime. |
| `EMAIL.md` | The earlier generic email draft is non-authoritative; the final result-bearing message is `DELIVERY_EMAIL.md` and remains unsent. |
| `REPORT.md` | The hypothesis is stated before the null result, secondary analyses are labeled, limitations remain prominent, and every number traces to canonical artifacts. |
| `README.md` | The first screen must state the bounded null and link one-command reproduction; it must not market uncertainty as an alarm the experiment did not support. |
| `RESULTS_SUMMARY.md` | The five memorable quantities, surprises, and weakest link are a concise derivative of canonical artifacts, not an independent analysis. |
| `interview_prep.md` | Answers distinguish primary null, secondary usefulness, hypotheses, and clinical unknowns so oral shorthand does not overclaim. |
| `DELIVERY_EMAIL.md` | This is a draft only; it states the null in one sentence, preserves the Kromp exclusion, and is never sent by automation. |

## Boundaries to say out loud

- The null is “no prespecified early warning at these thresholds on these datasets,” not “uncertainty never helps.”
- Clean-fitted temperature scaling transferred inconsistently at severity 5: ECE improved for all three SMIDS backbones and worsened for HuSHeM ResNet50.
- Failure ranking and selective prediction can be useful even when an aggregate signal supplies no lead-time alarm.
- The 2× relative thresholds are harder to cross from high clean baselines (HuSHeM clean ECE was 0.132); this is a post-hoc limitation, not a revised primary analysis, and absolute/AUROC-style alarms are future protocols.
- An uncertainty dashboard alone is not a validated degradation alarm; paired-device validation and ongoing monitoring remain necessary.
- The HuSHeM ±7.8-point fold SD is affected by 43–44-image folds: one error is about 2.3 points; it mixes model and evaluation variability.
- SMIDS MobileNetV3-Large and ResNet50 are descriptively close on clean performance; equivalence was not tested.
- Gaussian noise intentionally follows the ImageNet-C per-channel convention for comparability and is the least physical corruption; luminance-correlated shot noise is the more physical counterpart.
- Illumination uses a deterministic per-image gamma/white-balance direction held across severity; only the illustrated seed-1729 grid specifically darkens toward green/teal.
- Kromp stays blocked, pilot/demo runs stay excluded, and no result constitutes clinical validation or a deployment policy.
