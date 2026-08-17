# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3778
- Mean ridge-value val R2: 0.1757
- Mean recovery val R2: 0.1763
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.5541

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 21 | 0 | 64 | 12.50% | -0.4230 | 0.1539 | 0.1491 | 0.0858 | 0.1085 | 0.1539 | ridge_value_recovery |
| 21 | 1 | 64 | 12.50% | -0.2040 | 0.1736 | 0.1730 | 0.1598 | 0.1797 | 0.1797 | fused_residual |
| 21 | 2 | 64 | 12.50% | -0.7085 | 0.2118 | 0.2080 | 0.1041 | 0.1809 | 0.2118 | ridge_value_recovery |
| 21 | 3 | 64 | 12.50% | -0.4528 | -0.0225 | -0.0266 | -0.1103 | -0.1638 | -0.0225 | ridge_value_recovery |
| 21 | 4 | 64 | 12.50% | -0.5492 | -0.0174 | -0.0265 | -0.1102 | -0.1221 | -0.0174 | ridge_value_recovery |
| 21 | 5 | 64 | 12.50% | -0.6502 | 0.2400 | 0.2353 | 0.1103 | 0.1620 | 0.2400 | ridge_value_recovery |
| 21 | 6 | 64 | 12.50% | -0.2760 | 0.2279 | 0.2203 | 0.1360 | 0.1197 | 0.2279 | ridge_value_recovery |
| 21 | 7 | 64 | 12.50% | -0.2040 | 0.2127 | 0.2073 | 0.1393 | 0.1096 | 0.2127 | ridge_value_recovery |
| 21 | 0 | 64 | 12.50% | -0.5193 | 0.2565 | 0.2543 | 0.1129 | 0.1719 | 0.2565 | ridge_value_recovery |
| 21 | 1 | 64 | 12.50% | -0.0351 | 0.2650 | 0.2676 | 0.2250 | 0.2294 | 0.2676 | learned_compact_values |
| 21 | 2 | 64 | 12.50% | -0.5086 | 0.1750 | 0.1623 | 0.0504 | 0.1078 | 0.1750 | ridge_value_recovery |
| 21 | 3 | 64 | 12.50% | -0.4662 | -0.0112 | -0.0146 | -0.1063 | -0.1514 | -0.0112 | ridge_value_recovery |
| 21 | 4 | 64 | 12.50% | -0.3091 | 0.1045 | 0.0979 | 0.0303 | -0.0175 | 0.1045 | ridge_value_recovery |
| 21 | 5 | 64 | 12.50% | -0.3160 | 0.3904 | 0.3890 | 0.2594 | 0.2830 | 0.3904 | ridge_value_recovery |
| 21 | 6 | 64 | 12.50% | -0.1540 | 0.3375 | 0.3295 | 0.2349 | 0.2109 | 0.3375 | ridge_value_recovery |
| 21 | 7 | 64 | 12.50% | -0.2694 | 0.1140 | 0.1100 | 0.0404 | 0.0152 | 0.1140 | ridge_value_recovery |
