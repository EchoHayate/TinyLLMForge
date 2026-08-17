# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.2276
- Mean ridge-value val R2: 0.2020
- Mean recovery val R2: 0.2020
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.4297

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 13 | 0 | 64 | 12.50% | -0.0480 | 0.2131 | 0.1948 | 0.1761 | 0.1638 | 0.2131 | ridge_value_recovery |
| 13 | 1 | 64 | 12.50% | -0.1215 | 0.3575 | 0.3349 | 0.3145 | 0.3233 | 0.3575 | ridge_value_recovery |
| 13 | 2 | 64 | 12.50% | -0.0613 | 0.2108 | 0.1922 | 0.1857 | 0.1854 | 0.2108 | ridge_value_recovery |
| 13 | 3 | 64 | 12.50% | 0.0804 | 0.2736 | 0.2620 | 0.2349 | 0.2364 | 0.2736 | ridge_value_recovery |
| 13 | 4 | 64 | 12.50% | -0.6795 | 0.0684 | 0.0261 | -0.0069 | 0.0173 | 0.0684 | ridge_value_recovery |
| 13 | 5 | 64 | 12.50% | -0.1015 | 0.2115 | 0.1910 | 0.1853 | 0.1978 | 0.2115 | ridge_value_recovery |
| 13 | 6 | 64 | 12.50% | -1.0255 | -0.0743 | -0.1288 | -0.1864 | -0.1337 | -0.0743 | ridge_value_recovery |
| 13 | 7 | 64 | 12.50% | -0.2050 | 0.1804 | 0.1554 | 0.1339 | 0.1209 | 0.1804 | ridge_value_recovery |
| 13 | 0 | 64 | 12.50% | -0.0371 | 0.2354 | 0.2153 | 0.2118 | 0.2059 | 0.2354 | ridge_value_recovery |
| 13 | 1 | 64 | 12.50% | -0.1173 | 0.3190 | 0.2839 | 0.2504 | 0.2615 | 0.3190 | ridge_value_recovery |
| 13 | 2 | 64 | 12.50% | 0.0473 | 0.3359 | 0.3216 | 0.2792 | 0.2717 | 0.3359 | ridge_value_recovery |
| 13 | 3 | 64 | 12.50% | 0.1343 | 0.3328 | 0.3209 | 0.2923 | 0.2804 | 0.3328 | ridge_value_recovery |
| 13 | 4 | 64 | 12.50% | -0.5874 | 0.0309 | 0.0016 | -0.0762 | -0.0415 | 0.0309 | ridge_value_recovery |
| 13 | 5 | 64 | 12.50% | 0.0765 | 0.3334 | 0.3195 | 0.2992 | 0.2985 | 0.3334 | ridge_value_recovery |
| 13 | 6 | 64 | 12.50% | -0.9563 | -0.0516 | -0.1066 | -0.2051 | -0.1717 | -0.0516 | ridge_value_recovery |
| 13 | 7 | 64 | 12.50% | -0.0404 | 0.2554 | 0.2309 | 0.1918 | 0.1721 | 0.2554 | ridge_value_recovery |
