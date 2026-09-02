# Contributing

I use the following rules to keep experiments reproducible.

1. Store experiment settings in versioned YAML files under `configs/`.
2. Record the resolved configuration, random seed, environment, and Git revision for every run.
3. Save dataset splits as CSV manifests with separate train, validation, calibration, and test roles.
4. Keep Kromp modeling disabled until an author-verified patient map supports a patient-grouped split.
5. Train models only on clean images and apply corruptions only during evaluation.
6. Fit temperature scaling, vector scaling, and conformal thresholds only on calibration rows.
7. Report results across seeds or folds and retain negative or ambiguous findings.
8. Do not make clinical claims from public proxy data or simulated corruptions.
9. Do not commit raw data, model checkpoints, secrets, or machine-specific absolute paths.
10. Update `CORRUPTION_PROTOCOL_VERSION` after any corruption algorithm change and regenerate the fixed corruption example before evaluation.
