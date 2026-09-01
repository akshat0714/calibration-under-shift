# Checkpoint 3 secondary raw-versus-temperature ECE

Secondary/exploratory. This table is not part of the frozen primary threshold decision and must be labeled accordingly in any interpretation.

| Analysis Tier | Dataset | Model | Severity | Corruption Scope | Raw Ece Mean | Raw Ece Std | Temperature Ece Mean | Temperature Ece Std | Paired Difference Mean | Paired Difference Std | N Replicates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| secondary/exploratory | hushem | resnet50 | 0 | clean | 0.132 | 0.042 | 0.154 | 0.044 | 0.021 | 0.067 | 5 |
| secondary/exploratory | hushem | resnet50 | 1 | mean_of_device_corruptions | 0.122 | 0.018 | 0.135 | 0.037 | 0.013 | 0.028 | 5 |
| secondary/exploratory | hushem | resnet50 | 2 | mean_of_device_corruptions | 0.139 | 0.024 | 0.145 | 0.036 | 0.006 | 0.026 | 5 |
| secondary/exploratory | hushem | resnet50 | 3 | mean_of_device_corruptions | 0.152 | 0.021 | 0.169 | 0.027 | 0.018 | 0.022 | 5 |
| secondary/exploratory | hushem | resnet50 | 4 | mean_of_device_corruptions | 0.188 | 0.019 | 0.198 | 0.019 | 0.010 | 0.013 | 5 |
| secondary/exploratory | hushem | resnet50 | 5 | mean_of_device_corruptions | 0.184 | 0.019 | 0.198 | 0.030 | 0.014 | 0.031 | 5 |
| secondary/exploratory | smids | mobilenet_v3_large | 0 | clean | 0.040 | 0.006 | 0.035 | 0.007 | -0.005 | 0.003 | 3 |
| secondary/exploratory | smids | mobilenet_v3_large | 1 | mean_of_device_corruptions | 0.055 | 0.010 | 0.050 | 0.013 | -0.005 | 0.003 | 3 |
| secondary/exploratory | smids | mobilenet_v3_large | 2 | mean_of_device_corruptions | 0.139 | 0.066 | 0.130 | 0.070 | -0.009 | 0.005 | 3 |
| secondary/exploratory | smids | mobilenet_v3_large | 3 | mean_of_device_corruptions | 0.214 | 0.065 | 0.207 | 0.066 | -0.007 | 0.002 | 3 |
| secondary/exploratory | smids | mobilenet_v3_large | 4 | mean_of_device_corruptions | 0.289 | 0.091 | 0.282 | 0.096 | -0.008 | 0.005 | 3 |
| secondary/exploratory | smids | mobilenet_v3_large | 5 | mean_of_device_corruptions | 0.397 | 0.137 | 0.384 | 0.148 | -0.013 | 0.012 | 3 |
| secondary/exploratory | smids | resnet50 | 0 | clean | 0.047 | 0.011 | 0.046 | 0.009 | -0.001 | 0.009 | 5 |
| secondary/exploratory | smids | resnet50 | 1 | mean_of_device_corruptions | 0.046 | 0.005 | 0.041 | 0.006 | -0.005 | 0.006 | 5 |
| secondary/exploratory | smids | resnet50 | 2 | mean_of_device_corruptions | 0.052 | 0.003 | 0.047 | 0.009 | -0.006 | 0.007 | 5 |
| secondary/exploratory | smids | resnet50 | 3 | mean_of_device_corruptions | 0.101 | 0.010 | 0.088 | 0.014 | -0.012 | 0.011 | 5 |
| secondary/exploratory | smids | resnet50 | 4 | mean_of_device_corruptions | 0.178 | 0.016 | 0.161 | 0.018 | -0.016 | 0.016 | 5 |
| secondary/exploratory | smids | resnet50 | 5 | mean_of_device_corruptions | 0.274 | 0.020 | 0.251 | 0.030 | -0.023 | 0.018 | 5 |
| secondary/exploratory | smids | xception | 0 | clean | 0.052 | 0.023 | 0.046 | 0.010 | -0.005 | 0.013 | 3 |
| secondary/exploratory | smids | xception | 1 | mean_of_device_corruptions | 0.054 | 0.011 | 0.046 | 0.006 | -0.008 | 0.009 | 3 |
| secondary/exploratory | smids | xception | 2 | mean_of_device_corruptions | 0.068 | 0.008 | 0.060 | 0.003 | -0.009 | 0.007 | 3 |
| secondary/exploratory | smids | xception | 3 | mean_of_device_corruptions | 0.126 | 0.016 | 0.120 | 0.004 | -0.006 | 0.014 | 3 |
| secondary/exploratory | smids | xception | 4 | mean_of_device_corruptions | 0.156 | 0.008 | 0.145 | 0.008 | -0.010 | 0.004 | 3 |
| secondary/exploratory | smids | xception | 5 | mean_of_device_corruptions | 0.181 | 0.053 | 0.178 | 0.052 | -0.003 | 0.029 | 3 |
