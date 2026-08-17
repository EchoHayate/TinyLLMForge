# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.4187
- Mean ridge-value val R2: 0.1187
- Mean recovery val R2: 0.1187
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.5374

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | 0 | 64 | 12.50% | -0.6221 | 0.0407 | -0.0040 | -0.0347 | -0.0038 | 0.0407 | ridge_value_recovery |
| 9 | 1 | 64 | 12.50% | -0.3689 | 0.2771 | 0.2181 | 0.1239 | 0.1716 | 0.2771 | ridge_value_recovery |
| 9 | 2 | 64 | 12.50% | -0.5221 | 0.0473 | 0.0130 | -0.0196 | 0.0185 | 0.0473 | ridge_value_recovery |
| 9 | 3 | 64 | 12.50% | -0.2523 | 0.0097 | -0.0080 | -0.0374 | -0.0361 | 0.0097 | ridge_value_recovery |
| 9 | 4 | 64 | 12.50% | -0.0877 | 0.1758 | 0.1522 | 0.1216 | 0.1208 | 0.1758 | ridge_value_recovery |
| 9 | 5 | 64 | 12.50% | -0.9420 | -0.1688 | -0.2335 | -0.2957 | -0.2541 | -0.1688 | ridge_value_recovery |
| 9 | 6 | 64 | 12.50% | 0.0025 | 0.1939 | 0.1820 | 0.1773 | 0.1510 | 0.1939 | ridge_value_recovery |
| 9 | 7 | 64 | 12.50% | -0.8313 | 0.0025 | -0.0233 | -0.1085 | -0.0536 | 0.0025 | ridge_value_recovery |
| 9 | 0 | 64 | 12.50% | -0.4433 | 0.1307 | 0.0962 | -0.0026 | 0.0657 | 0.1307 | ridge_value_recovery |
| 9 | 1 | 64 | 12.50% | -0.2111 | 0.4430 | 0.4018 | 0.3163 | 0.3757 | 0.4430 | ridge_value_recovery |
| 9 | 2 | 64 | 12.50% | -0.4761 | 0.2012 | 0.1678 | 0.0658 | 0.1042 | 0.2012 | ridge_value_recovery |
| 9 | 3 | 64 | 12.50% | -0.1839 | 0.0193 | 0.0046 | -0.0420 | -0.0405 | 0.0193 | ridge_value_recovery |
| 9 | 4 | 64 | 12.50% | 0.0105 | 0.2013 | 0.1875 | 0.1582 | 0.1662 | 0.2013 | ridge_value_recovery |
| 9 | 5 | 64 | 12.50% | -1.1290 | -0.2118 | -0.2800 | -0.3581 | -0.3120 | -0.2118 | ridge_value_recovery |
| 9 | 6 | 64 | 12.50% | 0.0422 | 0.2462 | 0.2369 | 0.2281 | 0.2137 | 0.2462 | ridge_value_recovery |
| 9 | 7 | 64 | 12.50% | -0.6846 | 0.2905 | 0.2808 | 0.1938 | 0.2431 | 0.2905 | ridge_value_recovery |
