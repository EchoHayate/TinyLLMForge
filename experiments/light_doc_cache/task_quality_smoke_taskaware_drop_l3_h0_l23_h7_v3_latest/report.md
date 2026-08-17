# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_taskaware_drop_l3_h0_l23_h7`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.45 | 13 | 15 | pass | 100.00% | 100.00% | 100.00% | -0.4694 | 2.4631 | 2.3124 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.45 | 13 / 13 | 100.00% | 100.00% |

Per-task rows are in `task_rows.csv`.
