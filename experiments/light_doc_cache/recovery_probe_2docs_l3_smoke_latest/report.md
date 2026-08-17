# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.5129
- Mean ridge-value val R2: 0.1893
- Mean recovery val R2: 0.1893
- Recovery coverage above accept R2: 6.25%
- Mean recovery gain vs direct: 0.7021

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3 | 0 | 64 | 12.50% | -0.5758 | 0.3960 | 0.3516 | 0.1712 | 0.3447 | 0.3960 | ridge_value_recovery |
| 3 | 1 | 64 | 12.50% | -0.1452 | 0.2632 | 0.2389 | 0.1616 | 0.2272 | 0.2632 | ridge_value_recovery |
| 3 | 2 | 64 | 12.50% | -0.3455 | -0.0705 | -0.1105 | -0.1343 | -0.1137 | -0.0705 | ridge_value_recovery |
| 3 | 3 | 64 | 12.50% | -1.4498 | -0.0589 | -0.1461 | -0.3183 | -0.1411 | -0.0589 | ridge_value_recovery |
| 3 | 4 | 64 | 12.50% | -0.7931 | 0.0357 | -0.0454 | -0.1395 | -0.0487 | 0.0357 | ridge_value_recovery |
| 3 | 5 | 64 | 12.50% | -0.3073 | 0.0633 | 0.0185 | 0.0083 | 0.0260 | 0.0633 | ridge_value_recovery |
| 3 | 6 | 64 | 12.50% | -0.4341 | 0.2308 | 0.1727 | 0.0751 | 0.1700 | 0.2308 | ridge_value_recovery |
| 3 | 7 | 64 | 12.50% | -0.6861 | 0.1794 | 0.1357 | -0.0029 | 0.1368 | 0.1794 | ridge_value_recovery |
| 3 | 0 | 64 | 12.50% | -0.1921 | 0.5450 | 0.5173 | 0.3315 | 0.4956 | 0.5450 | ridge_value_recovery |
| 3 | 1 | 64 | 12.50% | -0.1242 | 0.3668 | 0.3319 | 0.2450 | 0.3337 | 0.3668 | ridge_value_recovery |
| 3 | 2 | 64 | 12.50% | -0.2947 | -0.0525 | -0.0793 | -0.0978 | -0.0860 | -0.0525 | ridge_value_recovery |
| 3 | 3 | 64 | 12.50% | -1.2608 | -0.0127 | -0.1113 | -0.3751 | -0.1488 | -0.0127 | ridge_value_recovery |
| 3 | 4 | 64 | 12.50% | -0.8597 | 0.0836 | -0.0304 | -0.2130 | -0.0672 | 0.0836 | ridge_value_recovery |
| 3 | 5 | 64 | 12.50% | -0.0190 | 0.2765 | 0.2450 | 0.2378 | 0.2439 | 0.2765 | ridge_value_recovery |
| 3 | 6 | 64 | 12.50% | -0.0299 | 0.4785 | 0.4479 | 0.3625 | 0.4311 | 0.4785 | ridge_value_recovery |
| 3 | 7 | 64 | 12.50% | -0.6887 | 0.3037 | 0.2615 | 0.0606 | 0.2620 | 0.3037 | ridge_value_recovery |
