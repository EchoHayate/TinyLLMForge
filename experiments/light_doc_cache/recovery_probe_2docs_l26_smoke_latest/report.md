# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3401
- Mean ridge-value val R2: 0.2681
- Mean recovery val R2: 0.2747
- Recovery coverage above accept R2: 12.50%
- Mean recovery gain vs direct: 0.6148

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 26 | 0 | 64 | 12.50% | -1.0600 | 0.1442 | 0.1405 | -0.0347 | 0.0484 | 0.1442 | ridge_value_recovery |
| 26 | 1 | 64 | 12.50% | -0.9659 | 0.2018 | 0.1980 | 0.0249 | 0.1334 | 0.2018 | ridge_value_recovery |
| 26 | 2 | 64 | 12.50% | -0.6805 | 0.3335 | 0.3300 | 0.1616 | 0.2552 | 0.3335 | ridge_value_recovery |
| 26 | 3 | 64 | 12.50% | -0.0786 | 0.3132 | 0.3129 | 0.2801 | 0.2550 | 0.3132 | ridge_value_recovery |
| 26 | 4 | 64 | 12.50% | -0.3329 | 0.0216 | 0.0204 | -0.0319 | -0.0565 | 0.0216 | ridge_value_recovery |
| 26 | 5 | 64 | 12.50% | -0.0029 | 0.3952 | 0.4006 | 0.3881 | 0.4209 | 0.4209 | fused_residual |
| 26 | 6 | 64 | 12.50% | -0.5605 | -0.0476 | -0.0431 | -0.0573 | -0.0500 | -0.0431 | learned_compact_values |
| 26 | 7 | 64 | 12.50% | -1.1024 | 0.3711 | 0.3713 | 0.2237 | 0.2915 | 0.3713 | learned_compact_values |
| 26 | 0 | 64 | 12.50% | -0.0495 | 0.4854 | 0.4902 | 0.5117 | 0.5115 | 0.5117 | mlp_residual |
| 26 | 1 | 64 | 12.50% | -0.1021 | 0.4885 | 0.4869 | 0.4429 | 0.4596 | 0.4885 | ridge_value_recovery |
| 26 | 2 | 64 | 12.50% | 0.0497 | 0.4640 | 0.4639 | 0.4169 | 0.4293 | 0.4640 | ridge_value_recovery |
| 26 | 3 | 64 | 12.50% | 0.2276 | 0.4079 | 0.4091 | 0.4111 | 0.4149 | 0.4149 | fused_residual |
| 26 | 4 | 64 | 12.50% | -0.1632 | 0.0278 | 0.0271 | -0.0313 | -0.0405 | 0.0278 | ridge_value_recovery |
| 26 | 5 | 64 | 12.50% | 0.2735 | 0.4834 | 0.4922 | 0.5157 | 0.5182 | 0.5182 | fused_residual |
| 26 | 6 | 64 | 12.50% | -0.6572 | 0.0044 | 0.0113 | -0.0341 | 0.0004 | 0.0113 | learned_compact_values |
| 26 | 7 | 64 | 12.50% | -0.2364 | 0.1952 | 0.1935 | 0.1519 | 0.1737 | 0.1952 | ridge_value_recovery |
