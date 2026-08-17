# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.4351
- Mean ridge-value val R2: 0.1157
- Mean recovery val R2: 0.1157
- Recovery coverage above accept R2: 6.25%
- Mean recovery gain vs direct: 0.5508

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 6 | 0 | 64 | 12.50% | -0.2839 | 0.1891 | 0.1407 | 0.0851 | 0.1457 | 0.1891 | ridge_value_recovery |
| 6 | 1 | 64 | 12.50% | -1.0802 | -0.1452 | -0.2318 | -0.3125 | -0.2168 | -0.1452 | ridge_value_recovery |
| 6 | 2 | 64 | 12.50% | -0.8690 | -0.0518 | -0.1370 | -0.2180 | -0.1207 | -0.0518 | ridge_value_recovery |
| 6 | 3 | 64 | 12.50% | -0.5819 | -0.0809 | -0.1354 | -0.1547 | -0.1334 | -0.0809 | ridge_value_recovery |
| 6 | 4 | 64 | 12.50% | -0.6770 | 0.0586 | -0.0211 | -0.1320 | -0.0023 | 0.0586 | ridge_value_recovery |
| 6 | 5 | 64 | 12.50% | -0.2504 | 0.1167 | 0.0708 | 0.0558 | 0.0647 | 0.1167 | ridge_value_recovery |
| 6 | 6 | 64 | 12.50% | -0.1058 | 0.1127 | 0.0892 | 0.0727 | 0.0916 | 0.1127 | ridge_value_recovery |
| 6 | 7 | 64 | 12.50% | 0.0688 | 0.4849 | 0.4592 | 0.3715 | 0.4628 | 0.4849 | ridge_value_recovery |
| 6 | 0 | 64 | 12.50% | -0.3085 | 0.2163 | 0.1658 | 0.0801 | 0.1659 | 0.2163 | ridge_value_recovery |
| 6 | 1 | 64 | 12.50% | -1.0656 | -0.1318 | -0.2175 | -0.3105 | -0.2148 | -0.1318 | ridge_value_recovery |
| 6 | 2 | 64 | 12.50% | -0.7165 | 0.0170 | -0.0694 | -0.1466 | -0.0659 | 0.0170 | ridge_value_recovery |
| 6 | 3 | 64 | 12.50% | -0.4398 | -0.0410 | -0.0964 | -0.1253 | -0.1062 | -0.0410 | ridge_value_recovery |
| 6 | 4 | 64 | 12.50% | -0.6854 | 0.1359 | 0.0378 | -0.1318 | 0.0455 | 0.1359 | ridge_value_recovery |
| 6 | 5 | 64 | 12.50% | -0.0830 | 0.2394 | 0.2001 | 0.1823 | 0.1901 | 0.2394 | ridge_value_recovery |
| 6 | 6 | 64 | 12.50% | -0.0811 | 0.1361 | 0.1076 | 0.0956 | 0.1090 | 0.1361 | ridge_value_recovery |
| 6 | 7 | 64 | 12.50% | 0.1973 | 0.5955 | 0.5730 | 0.4727 | 0.5682 | 0.5955 | ridge_value_recovery |
