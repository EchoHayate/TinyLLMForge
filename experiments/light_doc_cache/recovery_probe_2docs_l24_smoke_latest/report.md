# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.2694
- Mean ridge-value val R2: 0.1636
- Mean recovery val R2: 0.1639
- Recovery coverage above accept R2: 6.25%
- Mean recovery gain vs direct: 0.4333

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0 | 64 | 12.50% | -0.3121 | -0.0615 | -0.0657 | -0.1113 | -0.0615 | ridge_value_recovery |
| 24 | 1 | 64 | 12.50% | -0.0895 | 0.1521 | 0.1519 | 0.1563 | 0.1563 | mlp_residual |
| 24 | 2 | 64 | 12.50% | -0.9346 | 0.1705 | 0.1658 | -0.0684 | 0.1705 | ridge_value_recovery |
| 24 | 3 | 64 | 12.50% | -0.4309 | 0.2078 | 0.2051 | 0.0486 | 0.2078 | ridge_value_recovery |
| 24 | 4 | 64 | 12.50% | -0.4861 | -0.1165 | -0.1207 | -0.1764 | -0.1165 | ridge_value_recovery |
| 24 | 5 | 64 | 12.50% | -0.3566 | 0.2508 | 0.2484 | 0.1540 | 0.2508 | ridge_value_recovery |
| 24 | 6 | 64 | 12.50% | -0.5272 | 0.1287 | 0.1246 | 0.0019 | 0.1287 | ridge_value_recovery |
| 24 | 7 | 64 | 12.50% | -0.1583 | 0.2926 | 0.2889 | 0.2050 | 0.2926 | ridge_value_recovery |
| 24 | 0 | 64 | 12.50% | -0.4543 | -0.1767 | -0.1784 | -0.2127 | -0.1767 | ridge_value_recovery |
| 24 | 1 | 64 | 12.50% | -0.0039 | 0.1927 | 0.1922 | 0.1768 | 0.1927 | ridge_value_recovery |
| 24 | 2 | 64 | 12.50% | 0.0720 | 0.5345 | 0.5343 | 0.4611 | 0.5345 | ridge_value_recovery |
| 24 | 3 | 64 | 12.50% | 0.1709 | 0.4141 | 0.4130 | 0.3737 | 0.4141 | ridge_value_recovery |
| 24 | 4 | 64 | 12.50% | -0.4369 | -0.1745 | -0.1775 | -0.2474 | -0.1745 | ridge_value_recovery |
| 24 | 5 | 64 | 12.50% | -0.0703 | 0.2502 | 0.2480 | 0.1846 | 0.2502 | ridge_value_recovery |
| 24 | 6 | 64 | 12.50% | -0.1354 | 0.2911 | 0.2906 | 0.2005 | 0.2911 | ridge_value_recovery |
| 24 | 7 | 64 | 12.50% | -0.1573 | 0.2618 | 0.2570 | 0.0879 | 0.2618 | ridge_value_recovery |
