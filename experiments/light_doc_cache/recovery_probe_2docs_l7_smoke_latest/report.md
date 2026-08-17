# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.4191
- Mean ridge-value val R2: -0.0083
- Mean recovery val R2: -0.0083
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.4109

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7 | 0 | 64 | 12.50% | -0.2696 | 0.0479 | 0.0308 | -0.0108 | 0.0171 | 0.0479 | ridge_value_recovery |
| 7 | 1 | 64 | 12.50% | -0.6715 | -0.1383 | -0.1785 | -0.2100 | -0.2222 | -0.1383 | ridge_value_recovery |
| 7 | 2 | 64 | 12.50% | -0.0955 | 0.2205 | 0.2025 | 0.1770 | 0.2014 | 0.2205 | ridge_value_recovery |
| 7 | 3 | 64 | 12.50% | -0.3222 | 0.2283 | 0.1927 | 0.1532 | 0.1977 | 0.2283 | ridge_value_recovery |
| 7 | 4 | 64 | 12.50% | -0.5656 | -0.1370 | -0.1845 | -0.2083 | -0.1795 | -0.1370 | ridge_value_recovery |
| 7 | 5 | 64 | 12.50% | -0.6332 | -0.1371 | -0.1802 | -0.2005 | -0.1846 | -0.1371 | ridge_value_recovery |
| 7 | 6 | 64 | 12.50% | -0.2635 | -0.0139 | -0.0345 | -0.0497 | -0.0441 | -0.0139 | ridge_value_recovery |
| 7 | 7 | 64 | 12.50% | -0.2846 | 0.1181 | 0.0993 | 0.0328 | 0.0917 | 0.1181 | ridge_value_recovery |
| 7 | 0 | 64 | 12.50% | -0.5109 | -0.0455 | -0.0686 | -0.0765 | -0.0702 | -0.0455 | ridge_value_recovery |
| 7 | 1 | 64 | 12.50% | -0.6769 | -0.1570 | -0.2033 | -0.2503 | -0.2724 | -0.1570 | ridge_value_recovery |
| 7 | 2 | 64 | 12.50% | -0.1866 | 0.1547 | 0.1415 | 0.0705 | 0.1019 | 0.1547 | ridge_value_recovery |
| 7 | 3 | 64 | 12.50% | -0.4038 | -0.0019 | -0.0197 | -0.1126 | -0.0205 | -0.0019 | ridge_value_recovery |
| 7 | 4 | 64 | 12.50% | -0.4723 | -0.1500 | -0.1860 | -0.2241 | -0.2278 | -0.1500 | ridge_value_recovery |
| 7 | 5 | 64 | 12.50% | -0.6810 | -0.1263 | -0.1673 | -0.2550 | -0.2273 | -0.1263 | ridge_value_recovery |
| 7 | 6 | 64 | 12.50% | -0.3133 | -0.0647 | -0.0942 | -0.1376 | -0.1143 | -0.0647 | ridge_value_recovery |
| 7 | 7 | 64 | 12.50% | -0.3559 | 0.0700 | 0.0362 | -0.0205 | 0.0297 | 0.0700 | ridge_value_recovery |
