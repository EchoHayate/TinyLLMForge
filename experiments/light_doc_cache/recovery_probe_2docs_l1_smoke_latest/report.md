# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.8270
- Mean ridge-value val R2: 0.0748
- Mean recovery val R2: 0.0748
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.9018

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0 | 64 | 12.50% | -1.0422 | -0.0423 | -0.2453 | -0.7114 | -0.1904 | -0.0423 | ridge_value_recovery |
| 1 | 1 | 64 | 12.50% | -1.8248 | -0.1046 | -0.3319 | -0.9611 | -0.3042 | -0.1046 | ridge_value_recovery |
| 1 | 2 | 64 | 12.50% | -1.3547 | -0.0506 | -0.1384 | -0.4986 | -0.1457 | -0.0506 | ridge_value_recovery |
| 1 | 3 | 64 | 12.50% | -1.0965 | -0.0081 | -0.1851 | -0.5501 | -0.1091 | -0.0081 | ridge_value_recovery |
| 1 | 4 | 64 | 12.50% | -0.2348 | 0.0995 | 0.0325 | -0.1143 | 0.0473 | 0.0995 | ridge_value_recovery |
| 1 | 5 | 64 | 12.50% | -0.6697 | 0.1231 | 0.0022 | -0.5657 | -0.0157 | 0.1231 | ridge_value_recovery |
| 1 | 6 | 64 | 12.50% | -0.9422 | 0.0083 | -0.0977 | -0.6378 | -0.1207 | 0.0083 | ridge_value_recovery |
| 1 | 7 | 64 | 12.50% | -0.4960 | 0.1250 | 0.0288 | -0.1816 | 0.0578 | 0.1250 | ridge_value_recovery |
| 1 | 0 | 64 | 12.50% | -0.6863 | 0.1592 | 0.0020 | -0.4337 | 0.0168 | 0.1592 | ridge_value_recovery |
| 1 | 1 | 64 | 12.50% | -1.2647 | -0.0997 | -0.3028 | -0.7572 | -0.2524 | -0.0997 | ridge_value_recovery |
| 1 | 2 | 64 | 12.50% | -1.6574 | -0.0441 | -0.2119 | -0.7222 | -0.2122 | -0.0441 | ridge_value_recovery |
| 1 | 3 | 64 | 12.50% | -0.9276 | -0.0309 | -0.2075 | -0.5148 | -0.1386 | -0.0309 | ridge_value_recovery |
| 1 | 4 | 64 | 12.50% | -0.0983 | 0.1409 | 0.1028 | -0.0612 | 0.0982 | 0.1409 | ridge_value_recovery |
| 1 | 5 | 64 | 12.50% | -0.3848 | 0.3326 | 0.2816 | -0.0981 | 0.2443 | 0.3326 | ridge_value_recovery |
| 1 | 6 | 64 | 12.50% | -0.4709 | 0.2628 | 0.2190 | -0.3289 | 0.1289 | 0.2628 | ridge_value_recovery |
| 1 | 7 | 64 | 12.50% | -0.0816 | 0.3255 | 0.2458 | 0.0513 | 0.2507 | 0.3255 | ridge_value_recovery |
