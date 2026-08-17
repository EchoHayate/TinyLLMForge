# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.4508
- Mean ridge-value val R2: 0.0932
- Mean recovery val R2: 0.0932
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.5440

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 20 | 0 | 64 | 12.50% | -1.2891 | -0.2865 | -0.3032 | -0.4268 | -0.2865 | ridge_value_recovery |
| 20 | 1 | 64 | 12.50% | -0.3134 | 0.0697 | 0.0663 | -0.0037 | 0.0697 | ridge_value_recovery |
| 20 | 2 | 64 | 12.50% | -0.0413 | 0.3836 | 0.3772 | 0.3363 | 0.3836 | ridge_value_recovery |
| 20 | 3 | 64 | 12.50% | -0.2478 | 0.0811 | 0.0735 | 0.0268 | 0.0811 | ridge_value_recovery |
| 20 | 4 | 64 | 12.50% | -0.5763 | 0.0278 | 0.0182 | -0.0656 | 0.0278 | ridge_value_recovery |
| 20 | 5 | 64 | 12.50% | -0.4623 | 0.0378 | 0.0264 | -0.0325 | 0.0378 | ridge_value_recovery |
| 20 | 6 | 64 | 12.50% | -0.6618 | 0.1548 | 0.1411 | 0.0327 | 0.1548 | ridge_value_recovery |
| 20 | 7 | 64 | 12.50% | -0.3776 | 0.1470 | 0.1369 | 0.0828 | 0.1470 | ridge_value_recovery |
| 20 | 0 | 64 | 12.50% | -0.8771 | -0.1966 | -0.2059 | -0.2940 | -0.1966 | ridge_value_recovery |
| 20 | 1 | 64 | 12.50% | -0.2174 | 0.0943 | 0.0914 | 0.0525 | 0.0943 | ridge_value_recovery |
| 20 | 2 | 64 | 12.50% | -0.1829 | 0.3287 | 0.3114 | 0.2439 | 0.3287 | ridge_value_recovery |
| 20 | 3 | 64 | 12.50% | -0.3027 | 0.0824 | 0.0724 | 0.0167 | 0.0824 | ridge_value_recovery |
| 20 | 4 | 64 | 12.50% | -0.3925 | 0.0684 | 0.0594 | -0.0296 | 0.0684 | ridge_value_recovery |
| 20 | 5 | 64 | 12.50% | -0.3180 | 0.1372 | 0.1261 | 0.0565 | 0.1372 | ridge_value_recovery |
| 20 | 6 | 64 | 12.50% | -0.8533 | 0.1228 | 0.1071 | -0.0210 | 0.1228 | ridge_value_recovery |
| 20 | 7 | 64 | 12.50% | -0.0992 | 0.2383 | 0.2320 | 0.2061 | 0.2383 | ridge_value_recovery |
