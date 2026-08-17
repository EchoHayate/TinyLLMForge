# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3580
- Mean ridge-value val R2: -0.1105
- Mean recovery val R2: -0.1105
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.2475

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 27 | 0 | 64 | 12.50% | -0.3586 | -0.1001 | -0.1003 | -0.1350 | -0.1510 | -0.1001 | ridge_value_recovery |
| 27 | 1 | 64 | 12.50% | -0.2355 | -0.0466 | -0.0482 | -0.0973 | -0.1025 | -0.0466 | ridge_value_recovery |
| 27 | 2 | 64 | 12.50% | -0.2752 | -0.0809 | -0.0839 | -0.1366 | -0.1429 | -0.0809 | ridge_value_recovery |
| 27 | 3 | 64 | 12.50% | -0.2165 | -0.0517 | -0.0535 | -0.0623 | -0.0913 | -0.0517 | ridge_value_recovery |
| 27 | 4 | 64 | 12.50% | -0.6261 | -0.1050 | -0.1078 | -0.1562 | -0.1489 | -0.1050 | ridge_value_recovery |
| 27 | 5 | 64 | 12.50% | -0.5062 | -0.1326 | -0.1347 | -0.1953 | -0.2257 | -0.1326 | ridge_value_recovery |
| 27 | 6 | 64 | 12.50% | -0.2364 | -0.0657 | -0.0686 | -0.0904 | -0.1161 | -0.0657 | ridge_value_recovery |
| 27 | 7 | 64 | 12.50% | -0.1840 | -0.0618 | -0.0631 | -0.0851 | -0.0897 | -0.0618 | ridge_value_recovery |
| 27 | 0 | 64 | 12.50% | -0.3431 | -0.1465 | -0.1474 | -0.1873 | -0.2136 | -0.1465 | ridge_value_recovery |
| 27 | 1 | 64 | 12.50% | -0.2426 | -0.0958 | -0.0964 | -0.1220 | -0.1380 | -0.0958 | ridge_value_recovery |
| 27 | 2 | 64 | 12.50% | -0.3650 | -0.1559 | -0.1577 | -0.1946 | -0.2078 | -0.1559 | ridge_value_recovery |
| 27 | 3 | 64 | 12.50% | -0.3912 | -0.0855 | -0.0873 | -0.1040 | -0.1235 | -0.0855 | ridge_value_recovery |
| 27 | 4 | 64 | 12.50% | -0.6822 | -0.2322 | -0.2341 | -0.3387 | -0.3207 | -0.2322 | ridge_value_recovery |
| 27 | 5 | 64 | 12.50% | -0.4355 | -0.1946 | -0.1975 | -0.2514 | -0.2294 | -0.1946 | ridge_value_recovery |
| 27 | 6 | 64 | 12.50% | -0.4705 | -0.1136 | -0.1163 | -0.1470 | -0.1463 | -0.1136 | ridge_value_recovery |
| 27 | 7 | 64 | 12.50% | -0.1594 | -0.0997 | -0.1007 | -0.1448 | -0.1748 | -0.0997 | ridge_value_recovery |
