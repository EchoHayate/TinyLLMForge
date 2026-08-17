# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.2694
- Mean ridge-value val R2: 0.1636
- Mean recovery val R2: 0.1647
- Recovery coverage above accept R2: 6.25%
- Mean recovery gain vs direct: 0.4341

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0 | 64 | 12.50% | -0.3121 | -0.0615 | -0.0657 | -0.1094 | -0.1333 | -0.0615 | ridge_value_recovery |
| 24 | 1 | 64 | 12.50% | -0.0895 | 0.1521 | 0.1519 | 0.1542 | 0.1651 | 0.1651 | fused_residual |
| 24 | 2 | 64 | 12.50% | -0.9346 | 0.1705 | 0.1658 | -0.0754 | 0.0411 | 0.1705 | ridge_value_recovery |
| 24 | 3 | 64 | 12.50% | -0.4309 | 0.2078 | 0.2051 | 0.0538 | 0.0991 | 0.2078 | ridge_value_recovery |
| 24 | 4 | 64 | 12.50% | -0.4861 | -0.1165 | -0.1207 | -0.1772 | -0.2046 | -0.1165 | ridge_value_recovery |
| 24 | 5 | 64 | 12.50% | -0.3566 | 0.2508 | 0.2484 | 0.1499 | 0.1634 | 0.2508 | ridge_value_recovery |
| 24 | 6 | 64 | 12.50% | -0.5272 | 0.1287 | 0.1246 | 0.0013 | 0.0225 | 0.1287 | ridge_value_recovery |
| 24 | 7 | 64 | 12.50% | -0.1583 | 0.2926 | 0.2889 | 0.2064 | 0.2090 | 0.2926 | ridge_value_recovery |
| 24 | 0 | 64 | 12.50% | -0.4543 | -0.1767 | -0.1784 | -0.2123 | -0.2253 | -0.1767 | ridge_value_recovery |
| 24 | 1 | 64 | 12.50% | -0.0039 | 0.1927 | 0.1922 | 0.1710 | 0.1978 | 0.1978 | fused_residual |
| 24 | 2 | 64 | 12.50% | 0.0720 | 0.5345 | 0.5343 | 0.4610 | 0.4866 | 0.5345 | ridge_value_recovery |
| 24 | 3 | 64 | 12.50% | 0.1709 | 0.4141 | 0.4130 | 0.3722 | 0.3695 | 0.4141 | ridge_value_recovery |
| 24 | 4 | 64 | 12.50% | -0.4369 | -0.1745 | -0.1775 | -0.2442 | -0.2976 | -0.1745 | ridge_value_recovery |
| 24 | 5 | 64 | 12.50% | -0.0703 | 0.2502 | 0.2480 | 0.1850 | 0.1710 | 0.2502 | ridge_value_recovery |
| 24 | 6 | 64 | 12.50% | -0.1354 | 0.2911 | 0.2906 | 0.2057 | 0.2194 | 0.2911 | ridge_value_recovery |
| 24 | 7 | 64 | 12.50% | -0.1573 | 0.2618 | 0.2570 | 0.0862 | 0.0905 | 0.2618 | ridge_value_recovery |
