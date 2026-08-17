# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_all_r1.0`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.35 | 12 | 63 | weak-baseline | 75.00% | 41.67% | 66.67% | -2.9293 | 1.8887 | -0.4485 |
| 0.50 | 12 | 11 | weak-baseline | 75.00% | 83.33% | 91.67% | -0.0113 | 1.8887 | 2.1351 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.35 | 9 / 12 | 55.56% | 55.56% |
| 0.50 | 9 / 12 | 100.00% | 100.00% |

Per-task rows are in `task_rows.csv`.
