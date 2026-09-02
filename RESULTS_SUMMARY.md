# Results summary

## Main finding

I tested whether standard reliability signals crossed prespecified degradation thresholds before raw-softmax accuracy fell by more than five percentage points under seven simulated device corruptions. No reliability signal crossed earlier in any of the 10 comparisons where both crossings occurred. Six other signals never crossed. The prespecified early-warning hypothesis was not supported at these thresholds on SMIDS and HuSHeM.

I also found inconsistent transfer from clean-fitted temperature scaling. It reduced mean severity-5 ECE for every SMIDS backbone and increased it for HuSHeM ResNet50. I treat the relationship to Ovadia et al. and Thirumalaraju et al. as context rather than replication.

## Key results

I observed 0 earlier reliability crossings among 10 comparisons with both thresholds reached. Six of 16 signals never reached their thresholds.

The accuracy-drop threshold occurred at severity 2 for SMIDS MobileNetV3-Large and severity 3 for SMIDS ResNet50, SMIDS Xception, and HuSHeM ResNet50.

Temperature scaling changed mean severity-5 ECE by -0.023 for SMIDS ResNet50, -0.013 for MobileNetV3-Large, -0.003 for Xception, and +0.014 for HuSHeM.

At 80% retention, energy selection increased retained accuracy by 3.6 to 5.2 percentage points at severities 3 and 4. Ensemble entropy increased retained accuracy by 5.2 and 5.4 points.

APS coverage decreased and mean set size increased from clean data to severity 5 in all four dataset and backbone settings.

I averaged the seven corruptions within each seed or fold before summarizing replicates, as specified in `configs/analysis_protocol.yaml`.

## Primary thresholds

| Dataset and backbone | Accuracy drop | ECE | Predictive entropy | Selective risk at 80% | APS coverage |
|---|---|---|---|---|---|
| SMIDS ResNet50 | S3 | S3 | Not reached | S3 | Not reached |
| SMIDS Xception | S3 | S3 | S5 | S3 | Not reached |
| SMIDS MobileNetV3-Large | S2 | S2 | Not reached | S2 | S4 |
| HuSHeM ResNet50 | S3 | Not reached | S4 | S4 | Not reached |

## Secondary and exploratory analyses

I report these analyses separately from the primary conclusion.

I applied the same thresholds per corruption. Both thresholds were reached in 58 of 112 correlated comparisons. None showed an earlier reliability crossing. The signal did not cross in the other 54 comparisons.

I measured raw-softmax failure AUROC from severity 1 to severity 5. It changed from 0.817 to 0.697 for SMIDS ResNet50, 0.839 to 0.695 for Xception, 0.807 to 0.657 for MobileNetV3-Large, 0.815 to 0.704 for HuSHeM, and 0.836 to 0.732 for the ensemble.

I measured APS coverage and set size. Coverage changed from 0.977 to 0.865 for SMIDS ResNet50, 0.958 to 0.760 for MobileNetV3-Large, 0.987 to 0.869 for Xception, and 0.958 to 0.942 for HuSHeM. Mean set size increased by 0.148, 0.276, 0.121, and 0.655 classes, respectively.

I treated the real-checkpoint Grad-CAM and Grad-CAM++ analysis as exploratory.

## Clean test results

| Dataset and backbone | Accuracy mean ± sample SD | Macro-F1 mean ± sample SD |
|---|---|---|
| SMIDS ResNet50 | 89.7% ± 0.7 | 89.7% ± 0.7 |
| SMIDS Xception | 88.8% ± 2.2 | 88.9% ± 2.1 |
| SMIDS MobileNetV3-Large | 88.9% ± 1.6 | 89.0% ± 1.6 |
| HuSHeM ResNet50 | 86.5% ± 7.8 | 86.5% ± 7.9 |

I interpret the HuSHeM spread cautiously because four test folds contain 43 images and one contains 44. One classification changes fold accuracy by about 2.3 percentage points. The spread includes model variation, fold composition, and small-sample test resolution.

MobileNetV3-Large reached 88.9% clean accuracy on SMIDS and ResNet50 reached 89.7%. This descriptive similarity is relevant to on-device diagnostic deployment research. I did not test equivalence or deployment readiness.

## Post-hoc threshold limitation

I observed that the relative threshold depends on the clean baseline. HuSHeM clean ECE was 0.132, so the 2× rule required ECE to exceed 0.264 and the minimum absolute increase. I did not revise the primary analysis. I would prespecify absolute-threshold or AUROC-based detection methods in future work.

## Interpretation

The tested reliability measures did not identify aggregate degradation earlier than accuracy under my prespecified definitions. This result does not show that uncertainty methods are generally ineffective. Per-sample ranking remained useful in secondary analysis.

I would not use uncertainty dashboards alone as degradation alarms. Paired-device validation and ongoing monitoring remain necessary.

## Main limitation

My main limitation is the absence of paired reference and low-cost device captures of the same specimen. The public datasets also lack image-level patient or source linkage. Kromp remains excluded because its public release cannot support a patient-grouped split.

I do not claim clinical validation, a performance guarantee, or a clinical decision policy.

## Traceability

`results/metrics.csv` contains the evaluated metrics. `results/thresholds.csv` contains the primary crossings. `configs/analysis_protocol.yaml` contains the prespecified definitions. `results/figure_data/final_figure_manifest.json` contains figure source paths and hashes.
