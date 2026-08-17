# Light Doc Cache V2 Layer Policy Comparison

Task set: `task_quality_tasks_kv_sparse_v2.json` (9 baseline-stable questions).

| Policy | Threshold | Compressed Heads | Entry Saving | Compact Acc | Agreement | Answer LogP Delta | Compact Margin | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| all_layers | 0.35 | 63 | 25.15% | 55.56% | 55.56% | -3.6045 | 0.1170 | fail |
| all_layers | 0.50 | 11 | 4.30% | 100.00% | 100.00% | -0.2323 | 3.1628 | pass |
| all_layers_mid | 0.40 | 44 | 17.43% | 66.67% | 66.67% | -4.0953 | 0.3259 | fail |
| all_layers_mid | 0.45 | 27 | 10.79% | 77.78% | 77.78% | -1.8755 | 2.5217 | fail |
| late_layers_16_27 | 0.35 | 32 | 12.80% | 88.89% | 88.89% | -2.3634 | 1.8717 | fail |
| late_layers_16_27 | 0.50 | 5 | 1.93% | 100.00% | 100.00% | -0.3994 | 3.1079 | pass |

Interpretation:
- Late-layer-only compression improves `0.35` quality versus all-layer `0.35` (88.89% vs 55.56%) but still fails the 9-task smoke.
- Late-layer `0.50` passes, but saves only about 1.93% entries, worse than all-layer `0.50` at about 4.30%.
- The current evidence points away from simple threshold/layer-filter policies and toward richer recovery or constrained optimization that directly includes task-quality gates.
