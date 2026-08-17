# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.1400
- Mean ridge-value val R2: 0.1511
- Mean recovery val R2: 0.1511
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.2911

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11 | 0 | 64 | 12.50% | -0.0059 | 0.2972 | 0.2805 | 0.2601 | 0.2546 | 0.2972 | ridge_value_recovery |
| 11 | 1 | 64 | 12.50% | -0.3638 | 0.0199 | -0.0083 | -0.0277 | -0.0544 | 0.0199 | ridge_value_recovery |
| 11 | 2 | 64 | 12.50% | 0.1309 | 0.4335 | 0.4118 | 0.3768 | 0.3834 | 0.4335 | ridge_value_recovery |
| 11 | 3 | 64 | 12.50% | -0.2885 | 0.1515 | 0.1154 | 0.0697 | 0.0722 | 0.1515 | ridge_value_recovery |
| 11 | 4 | 64 | 12.50% | -0.1874 | -0.0112 | -0.0236 | -0.0740 | -0.1681 | -0.0112 | ridge_value_recovery |
| 11 | 5 | 64 | 12.50% | 0.0463 | 0.2531 | 0.2358 | 0.2105 | 0.2230 | 0.2531 | ridge_value_recovery |
| 11 | 6 | 64 | 12.50% | -0.0293 | 0.2449 | 0.2320 | 0.2235 | 0.2284 | 0.2449 | ridge_value_recovery |
| 11 | 7 | 64 | 12.50% | -0.2407 | -0.0668 | -0.0808 | -0.1681 | -0.1900 | -0.0668 | ridge_value_recovery |
| 11 | 0 | 64 | 12.50% | -0.0895 | 0.2677 | 0.2517 | 0.2442 | 0.2471 | 0.2677 | ridge_value_recovery |
| 11 | 1 | 64 | 12.50% | -0.2204 | 0.0354 | 0.0082 | -0.0299 | -0.0460 | 0.0354 | ridge_value_recovery |
| 11 | 2 | 64 | 12.50% | 0.0546 | 0.3594 | 0.3437 | 0.3104 | 0.3276 | 0.3594 | ridge_value_recovery |
| 11 | 3 | 64 | 12.50% | -0.3235 | 0.0940 | 0.0449 | -0.0235 | -0.0119 | 0.0940 | ridge_value_recovery |
| 11 | 4 | 64 | 12.50% | -0.1480 | 0.0109 | 0.0006 | -0.0529 | -0.1309 | 0.0109 | ridge_value_recovery |
| 11 | 5 | 64 | 12.50% | -0.1088 | 0.1984 | 0.1707 | 0.1486 | 0.1564 | 0.1984 | ridge_value_recovery |
| 11 | 6 | 64 | 12.50% | 0.0641 | 0.3149 | 0.3087 | 0.2940 | 0.2901 | 0.3149 | ridge_value_recovery |
| 11 | 7 | 64 | 12.50% | -0.5299 | -0.1855 | -0.2063 | -0.2946 | -0.3153 | -0.1855 | ridge_value_recovery |
