# Release checklist

Last audited: 2026-09-01. Check an item only after verifying it in the release candidate or public repository; synthetic-demo success is not a substitute for public-data completion.

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
- [x] Full configured public-data seed/fold matrix reaches sanity targets
- [x] `results/metrics.csv` contains the complete 45,540-row planned grid
- [x] Mean±SD results and seven per-corruption appendices generated
- [x] Headline, reliability, risk–coverage, failure, conformal, and attribution figures generated from real results
- [x] README and REPORT state the bounded null and mixed temperature-scaling result
- [x] Fresh full-history clone in a new pinned environment reproduces metrics and figures with one evaluation command
- [ ] GitHub Actions green on the public commit; badge points at the real repository
- [x] No raw data, large checkpoints, secrets, absolute paths, or synthetic metrics committed
- [x] Public repository link opens without authentication
- [ ] Five-minute walkthrough rehearsed twice; every module re-read
- [x] Delivery email wording and repository links drafted for user review
- [ ] User has reviewed and sent the delivery email on the existing thread
