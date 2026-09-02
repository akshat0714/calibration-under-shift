# Delivery email — draft only

**Status:** Draft for Akshat's review. Not sent.

**Subject:** Re: Research Assistant application — code sample

Dear Dr. Shafiee,

Thank you again — here is my code sample: [calibration-under-shift](https://github.com/akshat0714/calibration-under-shift).

It reproduces diagnostic image-classification pipelines on the public SMIDS and HuSHeM sperm-morphology datasets, then tests whether calibration, uncertainty, and conformal methods flag unreliable predictions as simulated image quality degrades toward smartphone-microscope conditions, extending the question raised by your group's [Fertility and Sterility study of cross-center and replicate-model variability](https://doi.org/10.1016/j.fertnstert.2025.08.021). Under the prespecified protocol, no reliability signal crossed before the five-point accuracy-drop threshold in any of the 10 comparisons where both crossings were observed, and six further signals never crossed, so the early-warning hypothesis was not supported at these thresholds on these datasets; in secondary analysis, clean-fitted temperature scaling improved severity-5 ECE on all three SMIDS backbones but worsened it on HuSHeM.

The repository includes a one-command released-checkpoint reproduction path in the [README](https://github.com/akshat0714/calibration-under-shift#reproduce), with the short write-up in [REPORT.md](https://github.com/akshat0714/calibration-under-shift/blob/main/REPORT.md). One note: I audited the public Kromp et al. blastocyst release and excluded it from modeling because its public files lack the patient linkage needed for a defensible patient-level split; the release defects are documented in the [data audit](https://github.com/akshat0714/calibration-under-shift/blob/main/results/data_audit.md).

I look forward to speaking with Manoj and Prudhvi — happy to walk through any part of it.

Best,

Akshat
