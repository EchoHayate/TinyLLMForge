# Task Failure Diagnostics Round4-Round7

This diagnostic aggregates local mirrored `task_rows.csv` for Round4-Round7 candidate gates.

- Progress rows scanned: 41
- Flipped/incorrect task rows: 39
- Flip table: `task_failure_diagnostics_round4_round7_20260709.tsv`
- Per-policy summary: `task_quality_summary_round4_round7_20260709.tsv`

## Most Frequent Flipped Tasks

| Count | Doc | Task |
|---:|---|---|
| 14 | first | `topk8_quality` |
| 8 | first | `route_phase3` |
| 7 | first | `sweet_spot` |
| 6 | first | `quest_decode_selection` |
| 2 | second | `tp_true_weight_split` |
| 2 | second | `smoothquant_status` |

## Candidates With Most Flips

| Count | Candidate |
|---:|---|
| 2 | `12:1` |
| 2 | `21:1` |
| 2 | `7:2` |
| 2 | `22:3` |
| 2 | `22:4` |
| 2 | `3:5` |
| 2 | `14:6` |
| 2 | `1:6` |
| 2 | `22:6` |
| 2 | `15:4` |
| 1 | `4:5` |
| 1 | `12:3` |
| 1 | `8:5` |
| 1 | `25:7` |
| 1 | `25:4` |
| 1 | `9:4` |
| 1 | `19:0` |
| 1 | `6:5` |
| 1 | `17:6` |
| 1 | `19:4` |
| 1 | `21:0` |
| 1 | `21:2` |
| 1 | `14:3` |
| 1 | `2:1` |
| 1 | `6:6` |
| 1 | `5:5` |
| 1 | `15:5` |
| 1 | `7:7` |
| 1 | `6:4` |

## Takeaways

- First-doc failures dominate late fixed-head expansion.
- Recurring fragile tasks should drive document/task-aware gating experiments: keep more heads full for policies that threaten these tasks, rather than extending a single global head list.
