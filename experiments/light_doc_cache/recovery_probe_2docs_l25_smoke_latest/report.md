# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.1906
- Mean ridge-value val R2: 0.2285
- Mean recovery val R2: 0.2298
- Recovery coverage above accept R2: 6.25%
- Mean recovery gain vs direct: 0.4205

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 25 | 0 | 64 | 12.50% | -0.4411 | -0.0594 | -0.0605 | -0.0705 | -0.0388 | -0.0388 | fused_residual |
| 25 | 1 | 64 | 12.50% | -0.6658 | 0.0625 | 0.0616 | -0.0171 | -0.0001 | 0.0625 | ridge_value_recovery |
| 25 | 2 | 64 | 12.50% | -0.6941 | 0.1132 | 0.1107 | -0.1176 | -0.0681 | 0.1132 | ridge_value_recovery |
| 25 | 3 | 64 | 12.50% | -0.3783 | 0.3131 | 0.3109 | 0.1990 | 0.2630 | 0.3131 | ridge_value_recovery |
| 25 | 4 | 64 | 12.50% | -0.1322 | 0.1197 | 0.1202 | 0.0812 | 0.0718 | 0.1202 | learned_compact_values |
| 25 | 5 | 64 | 12.50% | -0.1908 | 0.3855 | 0.3859 | 0.2938 | 0.3268 | 0.3859 | learned_compact_values |
| 25 | 6 | 64 | 12.50% | -0.2210 | 0.2682 | 0.2677 | 0.1675 | 0.1943 | 0.2682 | ridge_value_recovery |
| 25 | 7 | 64 | 12.50% | -0.2868 | 0.1547 | 0.1521 | 0.0621 | 0.0566 | 0.1547 | ridge_value_recovery |
| 25 | 0 | 64 | 12.50% | -0.3427 | -0.1001 | -0.1004 | -0.1336 | -0.1095 | -0.1001 | ridge_value_recovery |
| 25 | 1 | 64 | 12.50% | -0.3252 | 0.0298 | 0.0272 | -0.0044 | -0.0159 | 0.0298 | ridge_value_recovery |
| 25 | 2 | 64 | 12.50% | 0.1155 | 0.4619 | 0.4618 | 0.3927 | 0.3602 | 0.4619 | ridge_value_recovery |
| 25 | 3 | 64 | 12.50% | -0.1384 | 0.2172 | 0.2146 | 0.1306 | 0.1390 | 0.2172 | ridge_value_recovery |
| 25 | 4 | 64 | 12.50% | 0.1082 | 0.3907 | 0.3897 | 0.3516 | 0.3515 | 0.3907 | ridge_value_recovery |
| 25 | 5 | 64 | 12.50% | 0.2002 | 0.4016 | 0.3994 | 0.3583 | 0.3549 | 0.4016 | ridge_value_recovery |
| 25 | 6 | 64 | 12.50% | 0.3345 | 0.5409 | 0.5403 | 0.5054 | 0.5009 | 0.5409 | ridge_value_recovery |
| 25 | 7 | 64 | 12.50% | 0.0079 | 0.3563 | 0.3547 | 0.3009 | 0.2966 | 0.3563 | ridge_value_recovery |
