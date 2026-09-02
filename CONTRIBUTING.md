# Contributing

I store experiment settings in versioned YAML files under `configs/`. For every run, I record the resolved configuration, random seed, environment, and Git revision. I save dataset splits as CSV manifests with separate train, validation, calibration, and test roles.

Kromp modeling stays disabled until an author-verified patient map supports a patient-grouped split. I train models only on clean images and apply corruptions only during evaluation. I fit temperature scaling, vector scaling, and conformal thresholds only on calibration rows.

I report results across seeds or folds and retain negative or ambiguous findings. I do not make clinical claims from public proxy data or simulated corruptions.

I do not commit raw data, model checkpoints, secrets, or machine-specific absolute paths. After any corruption algorithm change, I update `CORRUPTION_PROTOCOL_VERSION` and regenerate the fixed corruption example before evaluation.
