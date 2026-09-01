# Checkpoint 3 secondary selective retained accuracy

Secondary/exploratory. This table is not part of the frozen primary threshold decision and must be labeled accordingly in any interpretation.

| Analysis Tier | Dataset | Model | Severity | Selector Method | Selector Score | Selector N Replicates | Retained Accuracy Mean | Retained Accuracy Std | Unselective N Replicates | Unselective Accuracy Mean | Unselective Accuracy Std | Clean N Replicates | Clean Accuracy Mean | Clean Accuracy Std | Gain Vs Unselective | Difference From Clean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| secondary/exploratory | hushem | resnet50 | 3 | energy | Energy score | 5 | 0.825 | 0.052 | 5 | 0.779 | 0.040 | 5 | 0.865 | 0.078 | 0.04554141318260918 | -0.040399390200054563 |
| secondary/exploratory | hushem | resnet50 | 4 | energy | Energy score | 5 | 0.737 | 0.061 | 5 | 0.701 | 0.051 | 5 | 0.865 | 0.078 | 0.03589708187714835 | -0.12822251945175522 |
| secondary/exploratory | smids | mobilenet_v3_large | 3 | energy | Energy score | 3 | 0.732 | 0.040 | 3 | 0.680 | 0.049 | 3 | 0.889 | 0.016 | 0.05226190476190473 | -0.15694444444444433 |
| secondary/exploratory | smids | mobilenet_v3_large | 4 | energy | Energy score | 3 | 0.647 | 0.078 | 3 | 0.602 | 0.072 | 3 | 0.889 | 0.016 | 0.045674603174603035 | -0.241468253968254 |
| secondary/exploratory | smids | resnet50 | 3 | deep_ensemble | Deep-ensemble predictive entropy | 1 | 0.901 | — | 1 | 0.849 | — | 1 | 0.920 | — | 0.051547619047619064 | -0.019404761904761925 |
| secondary/exploratory | smids | resnet50 | 4 | deep_ensemble | Deep-ensemble predictive entropy | 1 | 0.804 | — | 1 | 0.750 | — | 1 | 0.920 | — | 0.05369047619047629 | -0.11583333333333334 |
| secondary/exploratory | smids | resnet50 | 3 | energy | Energy score | 5 | 0.869 | 0.019 | 5 | 0.830 | 0.015 | 5 | 0.897 | 0.007 | 0.039380952380952516 | -0.027380952380952395 |
| secondary/exploratory | smids | resnet50 | 4 | energy | Energy score | 5 | 0.778 | 0.021 | 5 | 0.736 | 0.016 | 5 | 0.897 | 0.007 | 0.041857142857142926 | -0.1183333333333334 |
| secondary/exploratory | smids | xception | 3 | energy | Energy score | 3 | 0.831 | 0.018 | 3 | 0.787 | 0.017 | 3 | 0.888 | 0.022 | 0.04448412698412707 | -0.05662698412698419 |
| secondary/exploratory | smids | xception | 4 | energy | Energy score | 3 | 0.768 | 0.028 | 3 | 0.723 | 0.019 | 3 | 0.888 | 0.022 | 0.0441666666666668 | -0.12011904761904768 |
