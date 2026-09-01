# Checkpoint 3 primary results table

Primary/prespecified. Corruptions are averaged equally within each seed/fold before replicate means and sample standard deviations are computed. Missing crossings remain missing; the gap is an ordinal severity-level difference.

| Dataset | Backbone | Prespecified signal | n | Clean signal, mean ± SD | Signal crossing | Clean accuracy, mean ± SD | Accuracy crossing | Gap | Status |
|---|---|---|---|---|---|---|---|---|---|
| smids | resnet50 | Raw-softmax ECE | 5 | 0.047 ± 0.011 | 3 | 0.897 ± 0.007 | 3 | 0 | same_or_later |
| smids | resnet50 | Raw-softmax predictive entropy | 5 | 0.186 ± 0.036 | — | 0.897 ± 0.007 | 3 | — | signal_did_not_cross |
| smids | resnet50 | Raw-softmax risk at 80% coverage | 5 | 0.048 ± 0.009 | 3 | 0.897 ± 0.007 | 3 | 0 | same_or_later |
| smids | resnet50 | APS empirical coverage | 5 | 0.977 ± 0.014 | — | 0.897 ± 0.007 | 3 | — | signal_did_not_cross |
| smids | xception | Raw-softmax ECE | 3 | 0.052 ± 0.023 | 3 | 0.888 ± 0.022 | 3 | 0 | same_or_later |
| smids | xception | Raw-softmax predictive entropy | 3 | 0.282 ± 0.150 | 5 | 0.888 ± 0.022 | 3 | -2 | same_or_later |
| smids | xception | Raw-softmax risk at 80% coverage | 3 | 0.053 ± 0.017 | 3 | 0.888 ± 0.022 | 3 | 0 | same_or_later |
| smids | xception | APS empirical coverage | 3 | 0.987 ± 0.006 | — | 0.888 ± 0.022 | 3 | — | signal_did_not_cross |
| smids | mobilenet_v3_large | Raw-softmax ECE | 3 | 0.040 ± 0.006 | 2 | 0.889 ± 0.016 | 2 | 0 | same_or_later |
| smids | mobilenet_v3_large | Raw-softmax predictive entropy | 3 | 0.217 ± 0.033 | — | 0.889 ± 0.016 | 2 | — | signal_did_not_cross |
| smids | mobilenet_v3_large | Raw-softmax risk at 80% coverage | 3 | 0.054 ± 0.014 | 2 | 0.889 ± 0.016 | 2 | 0 | same_or_later |
| smids | mobilenet_v3_large | APS empirical coverage | 3 | 0.958 ± 0.011 | 4 | 0.889 ± 0.016 | 2 | -2 | same_or_later |
| hushem | resnet50 | Raw-softmax ECE | 5 | 0.132 ± 0.042 | — | 0.865 ± 0.078 | 3 | — | signal_did_not_cross |
| hushem | resnet50 | Raw-softmax predictive entropy | 5 | 0.414 ± 0.074 | 4 | 0.865 ± 0.078 | 3 | -1 | same_or_later |
| hushem | resnet50 | Raw-softmax risk at 80% coverage | 5 | 0.086 ± 0.073 | 4 | 0.865 ± 0.078 | 3 | -1 | same_or_later |
| hushem | resnet50 | APS empirical coverage | 5 | 0.958 ± 0.038 | — | 0.865 ± 0.078 | 3 | — | signal_did_not_cross |
