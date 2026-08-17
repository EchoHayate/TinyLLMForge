# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/task_quality_policy`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.90 | 13 | 73 | pass | 100.00% | 84.62% | 84.62% | -0.7073 | 2.4631 | 2.1812 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.90 | 13 / 13 | 84.62% | 84.62% |

Per-task rows are in `task_rows.csv`.
