# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.4906
- Mean ridge-value val R2: 0.1047
- Mean recovery val R2: 0.1047
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.5953

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 14 | 0 | 64 | 12.50% | -0.0788 | 0.2856 | 0.2699 | 0.2426 | 0.2557 | 0.2856 | ridge_value_recovery |
| 14 | 1 | 64 | 12.50% | -0.5471 | 0.2938 | 0.2591 | 0.1686 | 0.2304 | 0.2938 | ridge_value_recovery |
| 14 | 2 | 64 | 12.50% | -0.6670 | 0.0435 | 0.0023 | -0.0311 | 0.0271 | 0.0435 | ridge_value_recovery |
| 14 | 3 | 64 | 12.50% | -0.4613 | 0.1321 | 0.0992 | 0.0662 | 0.0428 | 0.1321 | ridge_value_recovery |
| 14 | 4 | 64 | 12.50% | -1.4260 | -0.2225 | -0.2924 | -0.4142 | -0.3686 | -0.2225 | ridge_value_recovery |
| 14 | 5 | 64 | 12.50% | -0.2790 | 0.2202 | 0.1850 | 0.1389 | 0.1398 | 0.2202 | ridge_value_recovery |
| 14 | 6 | 64 | 12.50% | -0.0980 | 0.1628 | 0.1422 | 0.1180 | 0.0756 | 0.1628 | ridge_value_recovery |
| 14 | 7 | 64 | 12.50% | -0.3621 | -0.0227 | -0.0533 | -0.0985 | -0.1786 | -0.0227 | ridge_value_recovery |
| 14 | 0 | 64 | 12.50% | -0.2779 | 0.1829 | 0.1687 | 0.1335 | 0.1328 | 0.1829 | ridge_value_recovery |
| 14 | 1 | 64 | 12.50% | -0.7328 | 0.1716 | 0.1424 | 0.0606 | 0.0976 | 0.1716 | ridge_value_recovery |
| 14 | 2 | 64 | 12.50% | -0.6229 | 0.1092 | 0.0757 | 0.0172 | 0.0672 | 0.1092 | ridge_value_recovery |
| 14 | 3 | 64 | 12.50% | -0.4634 | 0.1103 | 0.0795 | -0.0206 | -0.0364 | 0.1103 | ridge_value_recovery |
| 14 | 4 | 64 | 12.50% | -1.1723 | -0.1411 | -0.1834 | -0.3732 | -0.3550 | -0.1411 | ridge_value_recovery |
| 14 | 5 | 64 | 12.50% | -0.1739 | 0.2710 | 0.2465 | 0.1916 | 0.1943 | 0.2710 | ridge_value_recovery |
| 14 | 6 | 64 | 12.50% | -0.1507 | 0.1047 | 0.0876 | 0.0293 | -0.0091 | 0.1047 | ridge_value_recovery |
| 14 | 7 | 64 | 12.50% | -0.3363 | -0.0257 | -0.0588 | -0.1016 | -0.1399 | -0.0257 | ridge_value_recovery |
