# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.4045
- Mean ridge-value val R2: 0.0992
- Mean recovery val R2: 0.0996
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.5041

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 19 | 0 | 64 | 12.50% | -0.5834 | 0.1643 | 0.1500 | 0.0592 | 0.0583 | 0.1643 | ridge_value_recovery |
| 19 | 1 | 64 | 12.50% | -0.3700 | 0.0660 | 0.0571 | -0.0146 | -0.0826 | 0.0660 | ridge_value_recovery |
| 19 | 2 | 64 | 12.50% | -0.5595 | -0.0122 | -0.0283 | -0.0984 | -0.2163 | -0.0122 | ridge_value_recovery |
| 19 | 3 | 64 | 12.50% | -0.0989 | 0.1023 | 0.0982 | 0.0912 | 0.0756 | 0.1023 | ridge_value_recovery |
| 19 | 4 | 64 | 12.50% | -0.3185 | 0.1230 | 0.1121 | 0.0798 | 0.0387 | 0.1230 | ridge_value_recovery |
| 19 | 5 | 64 | 12.50% | -0.5849 | -0.0544 | -0.0671 | -0.1473 | -0.1503 | -0.0544 | ridge_value_recovery |
| 19 | 6 | 64 | 12.50% | -0.1344 | 0.2537 | 0.2473 | 0.2125 | 0.1798 | 0.2537 | ridge_value_recovery |
| 19 | 7 | 64 | 12.50% | -0.8748 | 0.0325 | 0.0202 | -0.0394 | 0.0172 | 0.0325 | ridge_value_recovery |
| 19 | 0 | 64 | 12.50% | -0.2989 | 0.2001 | 0.1922 | 0.1888 | 0.1988 | 0.2001 | ridge_value_recovery |
| 19 | 1 | 64 | 12.50% | -0.4571 | 0.0995 | 0.0914 | 0.0292 | -0.0033 | 0.0995 | ridge_value_recovery |
| 19 | 2 | 64 | 12.50% | -0.4151 | 0.0470 | 0.0327 | -0.0582 | -0.1421 | 0.0470 | ridge_value_recovery |
| 19 | 3 | 64 | 12.50% | -0.0248 | 0.1318 | 0.1280 | 0.1054 | 0.0652 | 0.1318 | ridge_value_recovery |
| 19 | 4 | 64 | 12.50% | -0.2742 | 0.1810 | 0.1699 | 0.1316 | 0.1160 | 0.1810 | ridge_value_recovery |
| 19 | 5 | 64 | 12.50% | -0.7442 | -0.1606 | -0.1740 | -0.2361 | -0.2796 | -0.1606 | ridge_value_recovery |
| 19 | 6 | 64 | 12.50% | -0.1929 | 0.2111 | 0.1993 | 0.1673 | 0.1669 | 0.2111 | ridge_value_recovery |
| 19 | 7 | 64 | 12.50% | -0.5402 | 0.2017 | 0.2040 | 0.2049 | 0.2090 | 0.2090 | fused_residual |
