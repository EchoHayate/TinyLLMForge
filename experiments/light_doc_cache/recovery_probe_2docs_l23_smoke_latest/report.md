# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3865
- Mean ridge-value val R2: 0.0914
- Mean recovery val R2: 0.0914
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.4779

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 23 | 0 | 64 | 12.50% | -0.2092 | 0.2600 | 0.2563 | 0.1866 | 0.1515 | 0.2600 | ridge_value_recovery |
| 23 | 1 | 64 | 12.50% | -0.6822 | -0.0847 | -0.0904 | -0.1631 | -0.1548 | -0.0847 | ridge_value_recovery |
| 23 | 2 | 64 | 12.50% | -0.1893 | 0.2669 | 0.2621 | 0.1948 | 0.1745 | 0.2669 | ridge_value_recovery |
| 23 | 3 | 64 | 12.50% | -0.1307 | 0.0782 | 0.0773 | 0.0652 | 0.0487 | 0.0782 | ridge_value_recovery |
| 23 | 4 | 64 | 12.50% | -0.7094 | -0.0913 | -0.0992 | -0.1979 | -0.2342 | -0.0913 | ridge_value_recovery |
| 23 | 5 | 64 | 12.50% | -0.5234 | -0.0146 | -0.0182 | -0.1041 | -0.1395 | -0.0146 | ridge_value_recovery |
| 23 | 6 | 64 | 12.50% | -0.7790 | -0.0414 | -0.0439 | -0.1607 | -0.1346 | -0.0414 | ridge_value_recovery |
| 23 | 7 | 64 | 12.50% | -0.1120 | 0.3309 | 0.3278 | 0.2325 | 0.2363 | 0.3309 | ridge_value_recovery |
| 23 | 0 | 64 | 12.50% | -0.1098 | 0.3248 | 0.3205 | 0.2473 | 0.2313 | 0.3248 | ridge_value_recovery |
| 23 | 1 | 64 | 12.50% | -0.5221 | -0.1465 | -0.1536 | -0.2566 | -0.2932 | -0.1465 | ridge_value_recovery |
| 23 | 2 | 64 | 12.50% | 0.0265 | 0.3457 | 0.3429 | 0.3131 | 0.2913 | 0.3457 | ridge_value_recovery |
| 23 | 3 | 64 | 12.50% | 0.0269 | 0.1653 | 0.1644 | 0.1473 | 0.1145 | 0.1653 | ridge_value_recovery |
| 23 | 4 | 64 | 12.50% | -0.6065 | -0.1190 | -0.1269 | -0.1929 | -0.1982 | -0.1190 | ridge_value_recovery |
| 23 | 5 | 64 | 12.50% | -0.3280 | 0.0349 | 0.0321 | -0.0833 | -0.1326 | 0.0349 | ridge_value_recovery |
| 23 | 6 | 64 | 12.50% | -1.0670 | -0.1029 | -0.1153 | -0.3441 | -0.3314 | -0.1029 | ridge_value_recovery |
| 23 | 7 | 64 | 12.50% | -0.2690 | 0.2559 | 0.2496 | 0.2245 | 0.1540 | 0.2559 | ridge_value_recovery |
