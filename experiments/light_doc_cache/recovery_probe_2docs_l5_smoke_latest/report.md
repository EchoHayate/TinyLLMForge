# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.6739
- Mean ridge-value val R2: -0.0359
- Mean recovery val R2: -0.0359
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.6380

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 5 | 0 | 64 | 12.50% | -0.6234 | -0.0918 | -0.1469 | -0.2092 | -0.1739 | -0.0918 | ridge_value_recovery |
| 5 | 1 | 64 | 12.50% | -1.2263 | -0.0005 | -0.0442 | -0.2364 | -0.0518 | -0.0005 | ridge_value_recovery |
| 5 | 2 | 64 | 12.50% | -0.4613 | 0.2128 | 0.1828 | 0.1311 | 0.1842 | 0.2128 | ridge_value_recovery |
| 5 | 3 | 64 | 12.50% | -0.7461 | -0.0846 | -0.1820 | -0.3056 | -0.1928 | -0.0846 | ridge_value_recovery |
| 5 | 4 | 64 | 12.50% | -1.0458 | -0.1611 | -0.2368 | -0.3461 | -0.2718 | -0.1611 | ridge_value_recovery |
| 5 | 5 | 64 | 12.50% | -0.2182 | 0.0496 | 0.0173 | -0.0126 | 0.0153 | 0.0496 | ridge_value_recovery |
| 5 | 6 | 64 | 12.50% | -0.9251 | -0.1819 | -0.2507 | -0.3571 | -0.2376 | -0.1819 | ridge_value_recovery |
| 5 | 7 | 64 | 12.50% | -0.1976 | -0.0052 | -0.0328 | -0.0659 | -0.0522 | -0.0052 | ridge_value_recovery |
| 5 | 0 | 64 | 12.50% | -0.5531 | -0.1111 | -0.1763 | -0.2321 | -0.1980 | -0.1111 | ridge_value_recovery |
| 5 | 1 | 64 | 12.50% | -0.9076 | -0.0090 | -0.0614 | -0.1581 | -0.0884 | -0.0090 | ridge_value_recovery |
| 5 | 2 | 64 | 12.50% | -0.3855 | 0.2266 | 0.1856 | 0.1125 | 0.1800 | 0.2266 | ridge_value_recovery |
| 5 | 3 | 64 | 12.50% | -0.7551 | -0.0480 | -0.1407 | -0.2680 | -0.1788 | -0.0480 | ridge_value_recovery |
| 5 | 4 | 64 | 12.50% | -1.1715 | -0.1848 | -0.2540 | -0.4095 | -0.3274 | -0.1848 | ridge_value_recovery |
| 5 | 5 | 64 | 12.50% | -0.2260 | 0.0786 | 0.0455 | 0.0216 | 0.0508 | 0.0786 | ridge_value_recovery |
| 5 | 6 | 64 | 12.50% | -1.1499 | -0.2421 | -0.3452 | -0.3834 | -0.3283 | -0.2421 | ridge_value_recovery |
| 5 | 7 | 64 | 12.50% | -0.1891 | -0.0213 | -0.0494 | -0.1028 | -0.0862 | -0.0213 | ridge_value_recovery |
