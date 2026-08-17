# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.2441
- Mean ridge-value val R2: 0.1274
- Mean recovery val R2: 0.1276
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.3717

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 18 | 0 | 64 | 12.50% | 0.0082 | 0.3938 | 0.3852 | 0.3389 | 0.3327 | 0.3938 | ridge_value_recovery |
| 18 | 1 | 64 | 12.50% | 0.0452 | 0.3217 | 0.3185 | 0.3155 | 0.3194 | 0.3217 | ridge_value_recovery |
| 18 | 2 | 64 | 12.50% | -0.5946 | 0.0373 | 0.0180 | -0.0465 | -0.0950 | 0.0373 | ridge_value_recovery |
| 18 | 3 | 64 | 12.50% | -0.3028 | 0.0199 | 0.0144 | -0.0198 | -0.0289 | 0.0199 | ridge_value_recovery |
| 18 | 4 | 64 | 12.50% | -0.3489 | -0.0229 | -0.0323 | -0.0618 | -0.1386 | -0.0229 | ridge_value_recovery |
| 18 | 5 | 64 | 12.50% | -0.2098 | 0.1510 | 0.1412 | 0.0973 | 0.0937 | 0.1510 | ridge_value_recovery |
| 18 | 6 | 64 | 12.50% | -0.1733 | 0.2169 | 0.2106 | 0.1658 | 0.1782 | 0.2169 | ridge_value_recovery |
| 18 | 7 | 64 | 12.50% | -0.3384 | -0.0539 | -0.0614 | -0.0666 | -0.0967 | -0.0539 | ridge_value_recovery |
| 18 | 0 | 64 | 12.50% | 0.0042 | 0.4767 | 0.4697 | 0.4083 | 0.4046 | 0.4767 | ridge_value_recovery |
| 18 | 1 | 64 | 12.50% | 0.0656 | 0.3399 | 0.3396 | 0.3286 | 0.3434 | 0.3434 | fused_residual |
| 18 | 2 | 64 | 12.50% | -0.4381 | 0.0636 | 0.0457 | -0.0273 | -0.0879 | 0.0636 | ridge_value_recovery |
| 18 | 3 | 64 | 12.50% | -0.4848 | -0.1090 | -0.1200 | -0.1790 | -0.1957 | -0.1090 | ridge_value_recovery |
| 18 | 4 | 64 | 12.50% | -0.4089 | -0.0831 | -0.0961 | -0.1597 | -0.2399 | -0.0831 | ridge_value_recovery |
| 18 | 5 | 64 | 12.50% | -0.2505 | 0.1136 | 0.1027 | 0.0602 | 0.0623 | 0.1136 | ridge_value_recovery |
| 18 | 6 | 64 | 12.50% | -0.0765 | 0.2438 | 0.2430 | 0.2155 | 0.2295 | 0.2438 | ridge_value_recovery |
| 18 | 7 | 64 | 12.50% | -0.4020 | -0.0709 | -0.0781 | -0.0919 | -0.1880 | -0.0709 | ridge_value_recovery |
