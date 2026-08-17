# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_all_r1.0_mid`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.40 | 9 | 44 | pass | 100.00% | 66.67% | 66.67% | -4.0953 | 3.2316 | 0.3259 |
| 0.45 | 9 | 27 | pass | 100.00% | 77.78% | 77.78% | -1.8755 | 3.2316 | 2.5217 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.40 | 9 / 9 | 66.67% | 66.67% |
| 0.45 | 9 / 9 | 77.78% | 77.78% |

Per-task rows are in `task_rows.csv`.
