# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_NEEDS_TASK_SMOKE**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.2966
- Mean ridge-value val R2: 0.0633
- Mean recovery val R2: 0.0809
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.3775

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2 | 0 | 64 | 12.50% | -0.3421 | -0.1071 | -0.1829 | -0.3460 | -0.1898 | -0.1071 | ridge_value_recovery |
| 2 | 1 | 64 | 12.50% | -0.4009 | 0.0472 | 0.0502 | 0.0485 | 0.1003 | 0.1003 | fused_residual |
| 2 | 2 | 64 | 12.50% | -0.1838 | 0.1068 | 0.0895 | 0.0398 | 0.0908 | 0.1068 | ridge_value_recovery |
| 2 | 3 | 64 | 12.50% | -0.3177 | 0.1475 | 0.1103 | -0.0202 | 0.1174 | 0.1475 | ridge_value_recovery |
| 2 | 4 | 64 | 12.50% | -0.1320 | 0.0466 | 0.0141 | -0.0308 | 0.0238 | 0.0466 | ridge_value_recovery |
| 2 | 5 | 64 | 12.50% | -0.0372 | 0.2072 | 0.2127 | 0.2145 | 0.2781 | 0.2781 | fused_residual |
| 2 | 6 | 64 | 12.50% | -0.4917 | -0.0918 | -0.1504 | -0.2170 | -0.1255 | -0.0918 | ridge_value_recovery |
| 2 | 7 | 64 | 12.50% | -0.2521 | 0.2373 | 0.2722 | 0.1591 | 0.2510 | 0.2722 | learned_compact_values |
| 2 | 0 | 64 | 12.50% | -0.4876 | -0.1258 | -0.1961 | -0.3348 | -0.2071 | -0.1258 | ridge_value_recovery |
| 2 | 1 | 64 | 12.50% | -0.6416 | 0.0032 | 0.0064 | 0.0577 | 0.0815 | 0.0815 | fused_residual |
| 2 | 2 | 64 | 12.50% | -0.2211 | 0.0594 | 0.0481 | 0.0197 | 0.0552 | 0.0594 | ridge_value_recovery |
| 2 | 3 | 64 | 12.50% | -0.3490 | 0.0983 | 0.0397 | -0.1494 | 0.0341 | 0.0983 | ridge_value_recovery |
| 2 | 4 | 64 | 12.50% | -0.1030 | 0.0471 | 0.0242 | -0.0093 | 0.0258 | 0.0471 | ridge_value_recovery |
| 2 | 5 | 64 | 12.50% | -0.0232 | 0.2218 | 0.2163 | 0.2161 | 0.2658 | 0.2658 | fused_residual |
| 2 | 6 | 64 | 12.50% | -0.4417 | -0.0961 | -0.1727 | -0.2877 | -0.1658 | -0.0961 | ridge_value_recovery |
| 2 | 7 | 64 | 12.50% | -0.3213 | 0.2107 | 0.1832 | 0.0501 | 0.1694 | 0.2107 | ridge_value_recovery |
