# Prespecified threshold analysis

Generated from `results/metrics.csv` using `configs/analysis_protocol.yaml`.
Missing threshold crossings remain missing.

| Dataset | Model | Signal | Signal severity | Accuracy severity | Gap | Earlier? |
|---|---|---|---:|---:|---:|:---:|
| hushem | resnet50 | raw_softmax/ece | Not reached | 3 | Not applicable | Not evaluable |
| hushem | resnet50 | raw_softmax/mean_predictive_entropy | 4 | 3 | -1 | no |
| hushem | resnet50 | raw_softmax/risk_at_80_coverage | 4 | 3 | -1 | no |
| hushem | resnet50 | aps/conformal_coverage | Not reached | 3 | Not applicable | Not evaluable |
| smids | mobilenet_v3_large | raw_softmax/ece | 2 | 2 | 0 | no |
| smids | mobilenet_v3_large | raw_softmax/mean_predictive_entropy | Not reached | 2 | Not applicable | Not evaluable |
| smids | mobilenet_v3_large | raw_softmax/risk_at_80_coverage | 2 | 2 | 0 | no |
| smids | mobilenet_v3_large | aps/conformal_coverage | 4 | 2 | -2 | no |
| smids | resnet50 | raw_softmax/ece | 3 | 3 | 0 | no |
| smids | resnet50 | raw_softmax/mean_predictive_entropy | Not reached | 3 | Not applicable | Not evaluable |
| smids | resnet50 | raw_softmax/risk_at_80_coverage | 3 | 3 | 0 | no |
| smids | resnet50 | aps/conformal_coverage | Not reached | 3 | Not applicable | Not evaluable |
| smids | xception | raw_softmax/ece | 3 | 3 | 0 | no |
| smids | xception | raw_softmax/mean_predictive_entropy | 5 | 3 | -2 | no |
| smids | xception | raw_softmax/risk_at_80_coverage | 3 | 3 | 0 | no |
| smids | xception | aps/conformal_coverage | Not reached | 3 | Not applicable | Not evaluable |
