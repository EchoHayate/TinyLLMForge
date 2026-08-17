# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.2595
- Mean ridge-value val R2: 0.1377
- Mean recovery val R2: 0.1395
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.3990

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 22 | 0 | 64 | 12.50% | -0.1628 | 0.2652 | 0.2621 | 0.2242 | 0.2276 | 0.2652 | ridge_value_recovery |
| 22 | 1 | 64 | 12.50% | 0.0758 | 0.4230 | 0.4271 | 0.3988 | 0.4146 | 0.4271 | learned_compact_values |
| 22 | 2 | 64 | 12.50% | -0.3512 | 0.0120 | 0.0076 | -0.0663 | -0.1281 | 0.0120 | ridge_value_recovery |
| 22 | 3 | 64 | 12.50% | -0.1587 | 0.1566 | 0.1550 | 0.1398 | 0.1144 | 0.1566 | ridge_value_recovery |
| 22 | 4 | 64 | 12.50% | -0.5566 | 0.0861 | 0.0826 | -0.0154 | -0.0487 | 0.0861 | ridge_value_recovery |
| 22 | 5 | 64 | 12.50% | -0.4863 | -0.0639 | -0.0691 | -0.1219 | -0.1290 | -0.0639 | ridge_value_recovery |
| 22 | 6 | 64 | 12.50% | -0.4664 | 0.1254 | 0.1255 | 0.0431 | 0.0051 | 0.1255 | learned_compact_values |
| 22 | 7 | 64 | 12.50% | -0.3883 | -0.0149 | -0.0216 | -0.0767 | -0.0871 | -0.0149 | ridge_value_recovery |
| 22 | 0 | 64 | 12.50% | -0.0244 | 0.4664 | 0.4664 | 0.4301 | 0.4480 | 0.4664 | ridge_value_recovery |
| 22 | 1 | 64 | 12.50% | 0.0831 | 0.3749 | 0.3785 | 0.3927 | 0.3997 | 0.3997 | fused_residual |
| 22 | 2 | 64 | 12.50% | -0.1666 | 0.0612 | 0.0588 | 0.0229 | 0.0154 | 0.0612 | ridge_value_recovery |
| 22 | 3 | 64 | 12.50% | -0.0444 | 0.2134 | 0.2129 | 0.2006 | 0.1666 | 0.2134 | ridge_value_recovery |
| 22 | 4 | 64 | 12.50% | -0.2176 | 0.2528 | 0.2519 | 0.1795 | 0.1074 | 0.2528 | ridge_value_recovery |
| 22 | 5 | 64 | 12.50% | -0.4032 | -0.1186 | -0.1226 | -0.1877 | -0.1882 | -0.1186 | ridge_value_recovery |
| 22 | 6 | 64 | 12.50% | -0.2289 | 0.0983 | 0.0972 | 0.0493 | 0.0407 | 0.0983 | ridge_value_recovery |
| 22 | 7 | 64 | 12.50% | -0.6551 | -0.1350 | -0.1432 | -0.1648 | -0.1379 | -0.1350 | ridge_value_recovery |
