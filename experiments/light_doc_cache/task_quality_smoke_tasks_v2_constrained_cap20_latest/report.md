# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_constrained_cap20_r1.0`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.45 | 9 | 20 | pass | 100.00% | 77.78% | 77.78% | -1.0685 | 3.2316 | 2.7088 |
| 0.50 | 9 | 11 | pass | 100.00% | 100.00% | 100.00% | -0.2323 | 3.2316 | 3.1628 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.45 | 9 / 9 | 77.78% | 77.78% |
| 0.50 | 9 / 9 | 100.00% | 100.00% |

Per-task rows are in `task_rows.csv`.
