# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/task_quality_policy`
Adaptive policy file: `/data00/home/sitian/light-doc-cache-work/probe/adaptive_policy.json`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Avg Entry Saving | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.90 | 13 | 80 | 17.71% | pass | 100.00% | 100.00% | 100.00% | 0.9575 | 2.4631 | 2.6967 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.90 | 13 / 13 | 100.00% | 100.00% |

Per-task rows are in `task_rows.csv`.
