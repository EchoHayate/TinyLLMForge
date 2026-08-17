# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 25.00%
- Mean direct val R2: 0.0936
- Mean ridge-value val R2: 0.2179
- Mean recovery val R2: 0.2295
- Recovery coverage above accept R2: 12.50%
- Mean recovery gain vs direct: 0.1359

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0 | 128 | 25.00% | -0.1442 | -0.0246 | -0.0247 | -0.0749 | -0.0923 | -0.0246 | ridge_value_recovery |
| 24 | 1 | 128 | 25.00% | 0.1318 | 0.1839 | 0.1881 | 0.2229 | 0.2222 | 0.2229 | mlp_residual |
| 24 | 2 | 128 | 25.00% | -0.0454 | 0.2653 | 0.2678 | 0.2213 | 0.2026 | 0.2678 | learned_compact_values |
| 24 | 3 | 128 | 25.00% | 0.1477 | 0.2944 | 0.2972 | 0.2577 | 0.2519 | 0.2972 | learned_compact_values |
| 24 | 4 | 128 | 25.00% | -0.2879 | -0.0795 | -0.0798 | -0.1481 | -0.1836 | -0.0795 | ridge_value_recovery |
| 24 | 5 | 128 | 25.00% | 0.0737 | 0.2773 | 0.2792 | 0.2248 | 0.2137 | 0.2792 | learned_compact_values |
| 24 | 6 | 128 | 25.00% | -0.0081 | 0.1926 | 0.1955 | 0.1449 | 0.1455 | 0.1955 | learned_compact_values |
| 24 | 7 | 128 | 25.00% | 0.3173 | 0.3937 | 0.3977 | 0.4009 | 0.3686 | 0.4009 | mlp_residual |
| 24 | 0 | 128 | 25.00% | -0.2119 | -0.1549 | -0.1509 | -0.1659 | -0.1780 | -0.1509 | learned_compact_values |
| 24 | 1 | 128 | 25.00% | 0.1763 | 0.2311 | 0.2354 | 0.2659 | 0.2784 | 0.2784 | fused_residual |
| 24 | 2 | 128 | 25.00% | 0.5287 | 0.5913 | 0.5953 | 0.6289 | 0.6112 | 0.6289 | mlp_residual |
| 24 | 3 | 128 | 25.00% | 0.4536 | 0.4984 | 0.5007 | 0.5087 | 0.4977 | 0.5087 | mlp_residual |
| 24 | 4 | 128 | 25.00% | -0.2630 | -0.1418 | -0.1402 | -0.2169 | -0.2528 | -0.1402 | learned_compact_values |
| 24 | 5 | 128 | 25.00% | 0.2883 | 0.3353 | 0.3379 | 0.3602 | 0.3197 | 0.3602 | mlp_residual |
| 24 | 6 | 128 | 25.00% | 0.2291 | 0.3099 | 0.3135 | 0.3045 | 0.2763 | 0.3135 | learned_compact_values |
| 24 | 7 | 128 | 25.00% | 0.1115 | 0.3142 | 0.3130 | 0.1825 | 0.1895 | 0.3142 | ridge_value_recovery |
