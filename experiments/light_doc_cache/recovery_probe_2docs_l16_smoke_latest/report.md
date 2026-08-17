# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.3204
- Mean ridge-value val R2: 0.0856
- Mean recovery val R2: 0.0856
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.4060

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 0 | 64 | 12.50% | -0.2438 | 0.0036 | -0.0093 | -0.0154 | 0.0036 | ridge_value_recovery |
| 16 | 1 | 64 | 12.50% | -0.5016 | 0.0575 | 0.0402 | -0.0158 | 0.0575 | ridge_value_recovery |
| 16 | 2 | 64 | 12.50% | -0.8930 | -0.1282 | -0.1786 | -0.2085 | -0.1282 | ridge_value_recovery |
| 16 | 3 | 64 | 12.50% | -0.4864 | 0.0837 | 0.0602 | 0.0039 | 0.0837 | ridge_value_recovery |
| 16 | 4 | 64 | 12.50% | -0.0061 | 0.1944 | 0.1854 | 0.1555 | 0.1944 | ridge_value_recovery |
| 16 | 5 | 64 | 12.50% | -0.2267 | 0.1636 | 0.1414 | 0.0943 | 0.1636 | ridge_value_recovery |
| 16 | 6 | 64 | 12.50% | 0.0626 | 0.2630 | 0.2548 | 0.2314 | 0.2630 | ridge_value_recovery |
| 16 | 7 | 64 | 12.50% | -0.2753 | 0.0441 | 0.0174 | -0.0120 | 0.0441 | ridge_value_recovery |
| 16 | 0 | 64 | 12.50% | -0.2996 | -0.0245 | -0.0450 | -0.1178 | -0.0245 | ridge_value_recovery |
| 16 | 1 | 64 | 12.50% | -0.5143 | 0.0134 | -0.0036 | -0.0526 | 0.0134 | ridge_value_recovery |
| 16 | 2 | 64 | 12.50% | -0.9556 | -0.1033 | -0.1393 | -0.2007 | -0.1033 | ridge_value_recovery |
| 16 | 3 | 64 | 12.50% | -0.4155 | 0.0808 | 0.0548 | 0.0079 | 0.0808 | ridge_value_recovery |
| 16 | 4 | 64 | 12.50% | -0.0269 | 0.1940 | 0.1803 | 0.1418 | 0.1940 | ridge_value_recovery |
| 16 | 5 | 64 | 12.50% | -0.1889 | 0.2013 | 0.1830 | 0.1650 | 0.2013 | ridge_value_recovery |
| 16 | 6 | 64 | 12.50% | 0.0174 | 0.2336 | 0.2237 | 0.1885 | 0.2336 | ridge_value_recovery |
| 16 | 7 | 64 | 12.50% | -0.1733 | 0.0924 | 0.0675 | 0.0242 | 0.0924 | ridge_value_recovery |
