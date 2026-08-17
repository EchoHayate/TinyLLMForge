# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3141
- Mean ridge-value val R2: 0.0987
- Mean recovery val R2: 0.0987
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.4128

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 17 | 0 | 64 | 12.50% | -0.7110 | 0.0435 | 0.0232 | -0.0950 | -0.1195 | 0.0435 | ridge_value_recovery |
| 17 | 1 | 64 | 12.50% | -0.0030 | 0.2409 | 0.2366 | 0.2075 | 0.2180 | 0.2409 | ridge_value_recovery |
| 17 | 2 | 64 | 12.50% | -0.2182 | 0.0099 | 0.0023 | -0.0388 | -0.0628 | 0.0099 | ridge_value_recovery |
| 17 | 3 | 64 | 12.50% | -0.6843 | -0.1562 | -0.1713 | -0.1927 | -0.1999 | -0.1562 | ridge_value_recovery |
| 17 | 4 | 64 | 12.50% | -0.1492 | 0.2662 | 0.2594 | 0.2393 | 0.2437 | 0.2662 | ridge_value_recovery |
| 17 | 5 | 64 | 12.50% | -0.1199 | 0.2513 | 0.2412 | 0.1965 | 0.1526 | 0.2513 | ridge_value_recovery |
| 17 | 6 | 64 | 12.50% | -0.0673 | 0.2130 | 0.2065 | 0.1822 | 0.1709 | 0.2130 | ridge_value_recovery |
| 17 | 7 | 64 | 12.50% | -0.4283 | -0.0675 | -0.0799 | -0.1239 | -0.2073 | -0.0675 | ridge_value_recovery |
| 17 | 0 | 64 | 12.50% | -0.6094 | 0.0984 | 0.0906 | -0.0543 | -0.0439 | 0.0984 | ridge_value_recovery |
| 17 | 1 | 64 | 12.50% | 0.0067 | 0.2743 | 0.2695 | 0.2680 | 0.2680 | 0.2743 | ridge_value_recovery |
| 17 | 2 | 64 | 12.50% | -0.2633 | -0.0075 | -0.0087 | -0.0286 | -0.0736 | -0.0075 | ridge_value_recovery |
| 17 | 3 | 64 | 12.50% | -0.7424 | -0.1134 | -0.1259 | -0.2040 | -0.2242 | -0.1134 | ridge_value_recovery |
| 17 | 4 | 64 | 12.50% | -0.1656 | 0.2463 | 0.2408 | 0.2137 | 0.2233 | 0.2463 | ridge_value_recovery |
| 17 | 5 | 64 | 12.50% | -0.2732 | 0.2237 | 0.2101 | 0.1241 | 0.1346 | 0.2237 | ridge_value_recovery |
| 17 | 6 | 64 | 12.50% | -0.1332 | 0.2004 | 0.1926 | 0.1530 | 0.1733 | 0.2004 | ridge_value_recovery |
| 17 | 7 | 64 | 12.50% | -0.4639 | -0.1448 | -0.1540 | -0.1926 | -0.2279 | -0.1448 | ridge_value_recovery |
