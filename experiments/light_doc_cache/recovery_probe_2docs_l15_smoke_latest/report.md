# Trainable Light Doc Cache Recovery Probe

Decision: **RECOVERY_WEAK**

- Heads: 16
- Mean budget fraction: 12.50%
- Mean direct val R2: -0.5115
- Mean ridge-value val R2: -0.0190
- Mean recovery val R2: -0.0190
- Recovery coverage above accept R2: 0.00%
- Mean recovery gain vs direct: 0.4925

| Layer | KV Head | Budget | Budget Frac | Direct Val R2 | FitV Val R2 | LearnedV Val R2 | MLP Val R2 | Fused Val R2 | Recovery Val R2 | Variant |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 15 | 0 | 64 | 12.50% | -0.6361 | -0.0122 | -0.0344 | -0.0911 | -0.1092 | -0.0122 | ridge_value_recovery |
| 15 | 1 | 64 | 12.50% | -0.8955 | -0.0799 | -0.1143 | -0.1913 | -0.1789 | -0.0799 | ridge_value_recovery |
| 15 | 2 | 64 | 12.50% | -0.2960 | 0.0161 | 0.0058 | -0.0269 | -0.0374 | 0.0161 | ridge_value_recovery |
| 15 | 3 | 64 | 12.50% | -0.2338 | 0.1444 | 0.1359 | 0.1013 | 0.0911 | 0.1444 | ridge_value_recovery |
| 15 | 4 | 64 | 12.50% | -0.4792 | 0.0658 | 0.0447 | 0.0024 | 0.0020 | 0.0658 | ridge_value_recovery |
| 15 | 5 | 64 | 12.50% | -0.5763 | 0.0454 | 0.0309 | 0.0126 | 0.0299 | 0.0454 | ridge_value_recovery |
| 15 | 6 | 64 | 12.50% | -0.5482 | -0.1064 | -0.1281 | -0.1314 | -0.1761 | -0.1064 | ridge_value_recovery |
| 15 | 7 | 64 | 12.50% | -0.5052 | -0.0111 | -0.0310 | -0.0507 | -0.0599 | -0.0111 | ridge_value_recovery |
| 15 | 0 | 64 | 12.50% | -0.7641 | -0.1961 | -0.2110 | -0.2592 | -0.2596 | -0.1961 | ridge_value_recovery |
| 15 | 1 | 64 | 12.50% | -0.6561 | -0.0623 | -0.0899 | -0.1954 | -0.2102 | -0.0623 | ridge_value_recovery |
| 15 | 2 | 64 | 12.50% | -0.6183 | -0.1493 | -0.1664 | -0.1965 | -0.1975 | -0.1493 | ridge_value_recovery |
| 15 | 3 | 64 | 12.50% | -0.2404 | 0.1518 | 0.1406 | 0.1055 | 0.0641 | 0.1518 | ridge_value_recovery |
| 15 | 4 | 64 | 12.50% | -0.4418 | 0.0295 | 0.0214 | -0.0113 | -0.0079 | 0.0295 | ridge_value_recovery |
| 15 | 5 | 64 | 12.50% | -0.3815 | 0.0636 | 0.0501 | 0.0239 | 0.0288 | 0.0636 | ridge_value_recovery |
| 15 | 6 | 64 | 12.50% | -0.4803 | -0.1445 | -0.1649 | -0.1955 | -0.2272 | -0.1445 | ridge_value_recovery |
| 15 | 7 | 64 | 12.50% | -0.4312 | -0.0589 | -0.0688 | -0.1214 | -0.1282 | -0.0589 | ridge_value_recovery |
