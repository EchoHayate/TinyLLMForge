# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.6530
- Mean ridge-value val R2: 0.0862
- Mean recovery val R2: 0.0862
- Recovery coverage above accept R2: 6.25%
- Mean recovery gain vs direct: 0.7393

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 0 | 64 | 12.50% | -1.2377 | -0.0656 | -0.1150 | -0.1628 | -0.0656 | ridge_value_recovery |
| 10 | 1 | 64 | 12.50% | -0.2246 | 0.0936 | 0.0725 | 0.0394 | 0.0936 | ridge_value_recovery |
| 10 | 2 | 64 | 12.50% | -0.4368 | 0.0576 | 0.0202 | -0.0049 | 0.0576 | ridge_value_recovery |
| 10 | 3 | 64 | 12.50% | -0.6374 | 0.3725 | 0.3506 | 0.2086 | 0.3725 | ridge_value_recovery |
| 10 | 4 | 64 | 12.50% | -0.6198 | -0.0759 | -0.1182 | -0.1415 | -0.0759 | ridge_value_recovery |
| 10 | 5 | 64 | 12.50% | -0.1107 | 0.3059 | 0.2848 | 0.2741 | 0.3059 | ridge_value_recovery |
| 10 | 6 | 64 | 12.50% | -0.8441 | -0.0257 | -0.0848 | -0.1759 | -0.0257 | ridge_value_recovery |
| 10 | 7 | 64 | 12.50% | -1.2061 | 0.0003 | -0.0279 | -0.1603 | 0.0003 | ridge_value_recovery |
| 10 | 0 | 64 | 12.50% | -1.0903 | -0.1416 | -0.1866 | -0.2613 | -0.1416 | ridge_value_recovery |
| 10 | 1 | 64 | 12.50% | -0.3489 | 0.0744 | 0.0545 | 0.0097 | 0.0744 | ridge_value_recovery |
| 10 | 2 | 64 | 12.50% | -0.3087 | 0.1531 | 0.1267 | 0.0762 | 0.1531 | ridge_value_recovery |
| 10 | 3 | 64 | 12.50% | -0.3513 | 0.5149 | 0.5125 | 0.3854 | 0.5149 | ridge_value_recovery |
| 10 | 4 | 64 | 12.50% | -0.7042 | -0.0783 | -0.1270 | -0.1833 | -0.0783 | ridge_value_recovery |
| 10 | 5 | 64 | 12.50% | -0.1661 | 0.2100 | 0.1946 | 0.1509 | 0.2100 | ridge_value_recovery |
| 10 | 6 | 64 | 12.50% | -0.9956 | -0.0126 | -0.0794 | -0.1659 | -0.0126 | ridge_value_recovery |
| 10 | 7 | 64 | 12.50% | -1.1663 | -0.0032 | -0.0309 | -0.1953 | -0.0032 | ridge_value_recovery |
