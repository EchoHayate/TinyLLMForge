# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3874
- Mean ridge-value val R2: 0.1132
- Mean recovery val R2: 0.1138
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.5012

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 0 | 64 | 12.50% | -0.1527 | 0.1024 | 0.0862 | 0.0683 | 0.1024 | ridge_value_recovery |
| 8 | 1 | 64 | 12.50% | 0.0431 | 0.3891 | 0.3861 | 0.3382 | 0.3891 | ridge_value_recovery |
| 8 | 2 | 64 | 12.50% | -0.4107 | 0.0310 | -0.0177 | -0.0287 | 0.0310 | ridge_value_recovery |
| 8 | 3 | 64 | 12.50% | -0.4315 | 0.1278 | 0.0781 | 0.0378 | 0.1278 | ridge_value_recovery |
| 8 | 4 | 64 | 12.50% | -1.0578 | -0.1066 | -0.1426 | -0.3890 | -0.1066 | ridge_value_recovery |
| 8 | 5 | 64 | 12.50% | -0.0204 | 0.3152 | 0.2936 | 0.2527 | 0.3152 | ridge_value_recovery |
| 8 | 6 | 64 | 12.50% | -0.4101 | 0.0611 | 0.0295 | -0.0015 | 0.0611 | ridge_value_recovery |
| 8 | 7 | 64 | 12.50% | -0.1359 | 0.0415 | 0.0342 | 0.0133 | 0.0415 | ridge_value_recovery |
| 8 | 0 | 64 | 12.50% | -0.0738 | 0.1669 | 0.1424 | 0.1038 | 0.1669 | ridge_value_recovery |
| 8 | 1 | 64 | 12.50% | -0.0402 | 0.3302 | 0.3401 | 0.2975 | 0.3401 | learned_compact_values |
| 8 | 2 | 64 | 12.50% | -0.3924 | 0.0909 | 0.0286 | -0.0191 | 0.0909 | ridge_value_recovery |
| 8 | 3 | 64 | 12.50% | -0.2694 | 0.2342 | 0.1794 | 0.1160 | 0.2342 | ridge_value_recovery |
| 8 | 4 | 64 | 12.50% | -1.5872 | -0.1301 | -0.1738 | -0.4213 | -0.1301 | ridge_value_recovery |
| 8 | 5 | 64 | 12.50% | -0.4450 | 0.1315 | 0.1045 | 0.0173 | 0.1315 | ridge_value_recovery |
| 8 | 6 | 64 | 12.50% | -0.6193 | 0.0451 | 0.0238 | -0.1716 | 0.0451 | ridge_value_recovery |
| 8 | 7 | 64 | 12.50% | -0.1946 | -0.0188 | -0.0336 | -0.0788 | -0.0188 | ridge_value_recovery |
