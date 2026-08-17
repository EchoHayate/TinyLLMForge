# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_all_r1.0`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.35 | 5 | 63 | pass | 80.00% | 80.00% | 100.00% | -3.2911 | 2.4540 | 0.7557 |
| 0.50 | 5 | 11 | pass | 80.00% | 80.00% | 100.00% | 0.0311 | 2.4540 | 2.7043 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.35 | 4 / 5 | 100.00% | 100.00% |
| 0.50 | 4 / 5 | 100.00% | 100.00% |

Per-task rows are in `task_rows.csv`.
