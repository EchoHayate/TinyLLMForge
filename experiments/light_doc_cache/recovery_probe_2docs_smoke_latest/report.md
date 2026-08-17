# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3912
- Mean ridge-value val R2: 0.0033
- Mean recovery val R2: -0.0491
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.3421

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | MLP Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 64 | 12.50% | -0.3637 | 0.1062 | -0.0060 | -0.0060 | mlp_residual |
| 0 | 1 | 64 | 12.50% | -0.4818 | -0.1009 | -0.1625 | -0.1625 | mlp_residual |
| 0 | 2 | 64 | 12.50% | -0.2898 | 0.0626 | 0.0115 | 0.0115 | mlp_residual |
| 0 | 3 | 64 | 12.50% | -0.5058 | -0.0482 | -0.1961 | -0.1961 | mlp_residual |
| 0 | 4 | 64 | 12.50% | -0.3136 | 0.0149 | -0.0714 | -0.0714 | mlp_residual |
| 0 | 5 | 64 | 12.50% | -0.9473 | 0.0176 | -0.1453 | 0.0176 | ridge_value_recovery |
| 0 | 6 | 64 | 12.50% | -0.2847 | 0.0290 | -0.0112 | -0.0112 | mlp_residual |
| 0 | 7 | 64 | 12.50% | -0.3696 | -0.0710 | -0.1817 | -0.1817 | mlp_residual |
| 0 | 0 | 64 | 12.50% | -0.1513 | 0.0726 | -0.0120 | 0.0726 | ridge_value_recovery |
| 0 | 1 | 64 | 12.50% | -0.5327 | -0.1307 | -0.1860 | -0.1860 | mlp_residual |
| 0 | 2 | 64 | 12.50% | -0.2155 | 0.0639 | 0.0280 | 0.0280 | mlp_residual |
| 0 | 3 | 64 | 12.50% | -0.2679 | 0.0272 | -0.0938 | -0.0938 | mlp_residual |
| 0 | 4 | 64 | 12.50% | -0.2073 | 0.0170 | -0.0440 | 0.0170 | ridge_value_recovery |
| 0 | 5 | 64 | 12.50% | -0.8737 | -0.0169 | -0.2425 | -0.0169 | ridge_value_recovery |
| 0 | 6 | 64 | 12.50% | -0.1985 | 0.0304 | 0.0134 | 0.0134 | mlp_residual |
| 0 | 7 | 64 | 12.50% | -0.2558 | -0.0206 | -0.1084 | -0.0206 | ridge_value_recovery |
