# Release checklist

Last audited: 2026-08-31. Check an item only after verifying it in the release candidate or public repository; synthetic-demo success is not a substitute for public-data completion.

- [x] Public GitHub repository created as `calibration-under-shift`; GPL-3.0 recognized; remote URL recorded
- [x] Exact dependency pins install in Python 3.11
- [x] SMIDS and HuSHeM downloads match publisher checksums
- [x] Kromp v3 archive matches Figshare MD5 and its full release defects are audited
- [x] All local SMIDS/HuSHeM files decoded and audited; expected class counts match
- [x] Fixed manifests keep train/validation/calibration/test roles disjoint
- [x] Kromp does not claim patient grouping without an author-verified map
- [x] Temperature/vector scaling and conformal fitting reject non-calibration roles
- [x] Training dataset rejects evaluation corruption
- [x] Analysis and threshold rules are versioned and frozen before the full seeded grid
- [x] Unit and synthetic end-to-end smoke tests pass locally
- [x] Walkthrough notebook restart-and-run-all succeeds locally without training
- [ ] Full configured public-data seed/fold matrix reaches sanity targets
- [ ] `results/metrics.csv` contains the complete planned grid
- [ ] Mean±SD results and per-corruption appendix generated
- [ ] Headline, reliability, risk–coverage, failure, conformal, and attribution figures generated from real results
- [ ] README and REPORT status language replaced with the verified result, including negative findings
- [ ] Fresh clone on a clean Linux/Colab environment reproduces metrics and figures
- [x] GitHub Actions green on the public commit; badge points at the real repository
- [x] No raw data, large checkpoints, secrets, absolute paths, or synthetic metrics committed
- [x] Public repository link opens without authentication
- [ ] Five-minute walkthrough rehearsed twice; every module re-read
- [ ] Email link and wording updated, then sent on their existing thread
