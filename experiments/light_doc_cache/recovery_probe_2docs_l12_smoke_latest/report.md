# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.2689
- Mean ridge-value val R2: 0.1575
- Mean recovery val R2: 0.1605
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.4294

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 12 | 0 | 64 | 12.50% | -0.3497 | 0.1489 | 0.1205 | 0.0832 | 0.1489 | ridge_value_recovery |
| 12 | 1 | 64 | 12.50% | -0.0227 | 0.2867 | 0.2726 | 0.2361 | 0.2867 | ridge_value_recovery |
| 12 | 2 | 64 | 12.50% | -0.4043 | -0.0124 | -0.0289 | -0.0589 | -0.0124 | ridge_value_recovery |
| 12 | 3 | 64 | 12.50% | 0.0427 | 0.2387 | 0.2404 | 0.2585 | 0.2585 | mlp_residual |
| 12 | 4 | 64 | 12.50% | -0.1585 | 0.3087 | 0.2979 | 0.2828 | 0.3087 | ridge_value_recovery |
| 12 | 5 | 64 | 12.50% | -0.2070 | 0.2273 | 0.2055 | 0.1815 | 0.2273 | ridge_value_recovery |
| 12 | 6 | 64 | 12.50% | -0.1594 | 0.2220 | 0.2023 | 0.1700 | 0.2220 | ridge_value_recovery |
| 12 | 7 | 64 | 12.50% | -0.0588 | 0.3037 | 0.2908 | 0.2658 | 0.3037 | ridge_value_recovery |
| 12 | 0 | 64 | 12.50% | -0.4733 | 0.0882 | 0.0606 | 0.0018 | 0.0882 | ridge_value_recovery |
| 12 | 1 | 64 | 12.50% | -0.1360 | 0.2313 | 0.2112 | 0.1652 | 0.2313 | ridge_value_recovery |
| 12 | 2 | 64 | 12.50% | -0.5079 | -0.1694 | -0.1859 | -0.2072 | -0.1694 | ridge_value_recovery |
| 12 | 3 | 64 | 12.50% | -0.0722 | 0.2111 | 0.2148 | 0.2389 | 0.2389 | mlp_residual |
| 12 | 4 | 64 | 12.50% | -0.2863 | 0.1301 | 0.1310 | 0.1090 | 0.1310 | learned_compact_values |
| 12 | 5 | 64 | 12.50% | -0.4512 | 0.1201 | 0.0861 | 0.0565 | 0.1201 | ridge_value_recovery |
| 12 | 6 | 64 | 12.50% | -0.7364 | -0.0427 | -0.0763 | -0.1394 | -0.0427 | ridge_value_recovery |
| 12 | 7 | 64 | 12.50% | -0.3213 | 0.2278 | 0.2042 | 0.1334 | 0.2278 | ridge_value_recovery |
