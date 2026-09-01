# Engineering and research contract

1. Every experiment is driven by a versioned YAML file in `configs/`; experimental constants do not live only in Python source.
2. Every run records its resolved configuration, random seed, environment metadata, and Git revision under `results/runs/<run_id>/`.
3. Dataset splits are generated once with an explicit seed and saved as CSV manifests. Kromp may run only with an author-verified patient map and a patient-grouped split; it remains blocked otherwise. Calibration data never overlaps train, validation, or test data.
4. Models train only on clean images. Device corruptions are deterministic, seeded, and applied only by evaluation data loaders.
5. Temperature/vector scaling and conformal thresholds are fit only on rows whose manifest split is `calibration`; the implementation rejects any other split.
6. Report uncertainty across seeds or folds, keep negative and ambiguous results, and make no clinical claims from public proxy data or simulated corruptions.
7. Raw data, checkpoints, secrets, and machine-specific absolute paths are never committed.
8. Any corruption-algorithm change must bump `CORRUPTION_PROTOCOL_VERSION`; any numeric ladder change is captured automatically by the registry digest, which invalidates stale inference caches. Regenerate the fixed corruption grid before evaluating checkpoints so its visual record and provenance match the active protocol.
