# Task-Level Quality Smoke Test

This is a quality-only simulation. It keeps the full KV tensor shape, but selected heads are forced to attend only to their compact bank during decode scoring.

Policy dir: `/data00/home/sitian/light-doc-cache-work/probe/runs/policy_am_qwen3_0_6b_s1536_holdout_all_r1.0`
Minimum baseline accuracy for a reliable quality smoke: `80.00%`.
The baseline gate marks rows as `weak-baseline` when the baseline is below that threshold.

| Threshold | Tasks | Heads | Baseline Gate | Baseline Acc | Compact Acc | Agreement | Mean Answer LogP Delta | Baseline Margin | Compact Margin |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 0.35 | 5 | 63 | weak-baseline | 60.00% | 40.00% | 40.00% | -0.4697 | 0.5750 | -0.5625 |
| 0.50 | 5 | 11 | weak-baseline | 60.00% | 80.00% | 80.00% | -0.0263 | 0.5750 | 0.5875 |

| Threshold | Baseline-Correct Tasks | Compact Acc on Baseline-Correct | Agreement on Baseline-Correct |
|---:|---:|---:|---:|
| 0.35 | 3 / 5 | 33.33% | 33.33% |
| 0.50 | 3 / 5 | 100.00% | 100.00% |

Per-task rows are in `task_rows.csv`.
