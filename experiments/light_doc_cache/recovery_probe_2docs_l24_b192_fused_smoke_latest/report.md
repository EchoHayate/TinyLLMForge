# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 37.50%
- Mean direct val R2: 0.2210
- Mean ridge-value val R2: 0.2087
- Mean recovery val R2: 0.2650
- Recovery coverage above accept R2: 12.50%
- Mean recovery gain vs direct: 0.0440

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 24 | 0 | 192 | 37.50% | -0.0774 | -0.0308 | -0.0269 | -0.0495 | -0.0710 | -0.0269 | learned_compact_values |
| 24 | 1 | 192 | 37.50% | 0.2140 | 0.1584 | 0.1657 | 0.2502 | 0.2437 | 0.2502 | mlp_residual |
| 24 | 2 | 192 | 37.50% | 0.2725 | 0.2550 | 0.2623 | 0.3463 | 0.3090 | 0.3463 | mlp_residual |
| 24 | 3 | 192 | 37.50% | 0.3155 | 0.2784 | 0.2835 | 0.3525 | 0.3081 | 0.3525 | mlp_residual |
| 24 | 4 | 192 | 37.50% | -0.1737 | -0.0679 | -0.0651 | -0.1261 | -0.1544 | -0.0651 | learned_compact_values |
| 24 | 5 | 192 | 37.50% | 0.2840 | 0.2761 | 0.2822 | 0.3276 | 0.2909 | 0.3276 | mlp_residual |
| 24 | 6 | 192 | 37.50% | 0.2080 | 0.1807 | 0.1877 | 0.2467 | 0.2058 | 0.2467 | mlp_residual |
| 24 | 7 | 192 | 37.50% | 0.4339 | 0.3851 | 0.3910 | 0.4669 | 0.4091 | 0.4669 | mlp_residual |
| 24 | 0 | 192 | 37.50% | -0.1246 | -0.1616 | -0.1532 | -0.1151 | -0.1313 | -0.1151 | mlp_residual |
| 24 | 1 | 192 | 37.50% | 0.2706 | 0.2174 | 0.2253 | 0.3071 | 0.3144 | 0.3144 | fused_residual |
| 24 | 2 | 192 | 37.50% | 0.6636 | 0.5685 | 0.5754 | 0.6861 | 0.6601 | 0.6861 | mlp_residual |
| 24 | 3 | 192 | 37.50% | 0.5488 | 0.4895 | 0.4930 | 0.5621 | 0.5257 | 0.5621 | mlp_residual |
| 24 | 4 | 192 | 37.50% | -0.2145 | -0.1538 | -0.1492 | -0.1878 | -0.2272 | -0.1492 | learned_compact_values |
| 24 | 5 | 192 | 37.50% | 0.3666 | 0.3246 | 0.3294 | 0.3898 | 0.3448 | 0.3898 | mlp_residual |
| 24 | 6 | 192 | 37.50% | 0.3170 | 0.3004 | 0.3051 | 0.3343 | 0.2912 | 0.3343 | mlp_residual |
| 24 | 7 | 192 | 37.50% | 0.2318 | 0.3184 | 0.3194 | 0.2477 | 0.2311 | 0.3194 | learned_compact_values |
