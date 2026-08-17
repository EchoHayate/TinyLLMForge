# Light Doc Cache V2 Quality vs Compression Tradeoff

Task set: `task_quality_tasks_kv_sparse_v2.json` (9 baseline-stable questions, Qwen3-0.6B, `choice_scoring=text_only`).

| Threshold | Compressed Heads | Entry Saving | Baseline Acc | Compact Acc | Agreement | Answer LogP Delta | Compact Margin | Decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.35 | 63 | 25.15% | 100.00% | 55.56% | 55.56% | -3.6045 | 0.1170 | quality-fail |
| 0.40 | 44 | 17.43% | 100.00% | 66.67% | 66.67% | -4.0953 | 0.3259 | quality-fail |
| 0.45 | 27 | 10.79% | 100.00% | 77.78% | 77.78% | -1.8755 | 2.5217 | quality-fail |
| 0.50 | 11 | 4.30% | 100.00% | 100.00% | 100.00% | -0.2323 | 3.1628 | quality-pass; low saving |

## Constrained Policy Sweep

Constrained policies rank eligible compact heads by holdout `fitv_val_r2`, then apply simple global/per-layer caps and a minimum per-head saving filter. The rows below all use `min_saving_fraction=0.50`.

| Policy | Threshold | Constraint | Compressed Heads | Entry Saving | Compact Acc | Agreement | Answer LogP Delta | Compact Margin | Decision |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| constrained16_l1 | 0.35 | max 16, max 1/layer | 16 | 6.47% | 88.89% | 88.89% | -0.9532 | 2.7388 | quality-fail |
| constrained16_l1 | 0.40 | max 16, max 1/layer | 16 | 6.40% | 88.89% | 88.89% | -0.9232 | 2.5324 | quality-fail |
| constrained16_l1 | 0.45 | max 16, max 1/layer | 16 | 6.45% | 100.00% | 100.00% | -0.5539 | 3.0093 | quality-pass |
| constrained16_l1 | 0.50 | max 16, max 1/layer | 9 | 3.52% | 100.00% | 100.00% | +0.0499 | 3.0498 | quality-pass; lower saving |
| constrained18_l1 | 0.45 | max 18, max 1/layer | 17 | 6.83% | 100.00% | 100.00% | -0.7587 | 2.8619 | v2 pass; fails v3 stress |
| constrained20_l2 | 0.45 | max 20, max 2/layer | 20 | 8.04% | 77.78% | 77.78% | -1.0685 | 2.7088 | quality-fail |
| constrained24_l2 | 0.45 | max 24, max 2/layer | 23 | 9.19% | 77.78% | 77.78% | -1.7738 | 2.3487 | quality-fail |
| constrained20_l2 | 0.50 | max 20, max 2/layer | 11 | 4.30% | 100.00% | 100.00% | -0.2323 | 3.1628 | same as unconstrained 0.50 |

Interpretation:
- `0.35`, `0.40`, and `0.45` all fail the 9-task v2 quality smoke despite increasingly smaller compression sets.
- `0.50` is the first tested threshold that preserves all baseline-correct v2 tasks, but it compresses only 11 / 224 heads and saves about 4.30% KV head-token entries.
- Simple per-layer caps improve the best v2 smoke point from 11 heads / 4.30% saving to 17 heads / 6.83% saving.
- The safety boundary is still tight: allowing 20 or 23 heads at threshold `0.45` flips `quest_summary_form` and `quest_default_enable`.
- The expanded v3 stable task set exposes a further failure: the 17-head policy flips two non-A-answer tasks (`phase2_magic_number`, `sweet_spot`) and reaches only 84.62% compact accuracy.
- This is a better research direction than the raw threshold policy, but still not enough for runtime integration. Next work should use the v3 stable set as the minimum gate and explore stricter task-aware constraints or richer multi-source recovery before committing engineering work.

## V3 Stress Check

`task_quality_tasks_kv_sparse_v3_candidate.json` expanded the task set to 20 questions, including non-A correct answers. Baseline-only calibration with an empty policy reached only 65.00% baseline accuracy, so `task_quality_tasks_kv_sparse_v3_stable.json` keeps the 13 baseline-correct tasks and records 7 excluded task IDs.

| Task Set | Policy | Threshold | Heads | Entry Saving | Baseline Acc | Compact Acc | Agreement | Answer LogP Delta | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| v3_candidate | empty baseline-only | 0.99 | 0 | 0.00% | 65.00% | 65.00% | 100.00% | 0.0000 | weak-baseline; use only for filtering |
| v3_stable | constrained18_l1 | 0.45 | 17 | 6.83% | 100.00% | 84.62% | 84.62% | -0.6256 | quality-fail |
| v3_stable | taskaware_drop_l3_h0_l23_h7 | 0.45 | 15 | 6.03% | 100.00% | 100.00% | 100.00% | -0.4694 | current best smoke |
| v2_stable | taskaware_drop_l3_h0_l23_h7 | 0.45 | 15 | 6.03% | 100.00% | 100.00% | 100.00% | -0.6160 | passes v2 regression |

V3 failure rows for `constrained18_l1`:

| Task | Baseline Pred | Compact Pred | Answer | Answer LogP Delta | Compact Margin |
|---|---|---|---|---:|---:|
| phase2_magic_number | B | A | B | -0.0574 | -0.1019 |
| sweet_spot | B | A | B | -0.1323 | -0.2110 |

## Task-Aware Pair-Drop Ablation

To diagnose the v3 failure, `make_policy_ablation.py` generated leave-one-out and pair-drop variants from `constrained18_l1`. A dropped head is now accounted as a full head in `selected_budget`, so entry-saving summaries reflect the true compact set.

Leave-one-out result:
- No single dropped head repaired all v3 stable tasks.
- Several variants repaired exactly one of the two non-A failures, reaching 92.31% compact accuracy.

Pair-drop fail-only screening:
- `task_quality_tasks_v3_failonly.json` contains only `phase2_magic_number` and `sweet_spot`.
- 26 pair-drop variants fixed both fail-only tasks.
- The strongest fail-only margin candidate was `drop_l3_h0__l23_h7`.

Validated task-aware policy:

| Policy | Dropped Heads | Compressed Heads | Entry Saving | V3 Stable Compact Acc | V2 Stable Compact Acc | Mean V3 Delta | Mean V3 Compact Margin |
|---|---|---:|---:|---:|---:|---:|---:|
| taskaware_drop_l3_h0_l23_h7 | `3:0`, `23:7` | 15 | 6.03% | 100.00% | 100.00% | -0.4694 | 2.3124 |

Interpretation:
- The task-aware pair-drop policy recovers the v3 non-A failures while retaining more saving than the raw safe threshold `0.50` (6.03% vs 4.30%).
- It is still below the v2-only 17-head policy's nominal 6.83% saving, but is the first policy in this run that passes both v3 stable and v2 stable.
- This remains a single-document quality-only simulation; multi-document stress below shows it does not yet generalize.

## Second-Document Stress

The runner now supports `LOCAL_TEXT_FILE=...`, which base64-syncs a local Markdown document to the remote probe directory and verifies size/SHA256 before running. This was used to evaluate `docs/qwen3-8b-fixes.md`.

Task files:
- `task_quality_tasks_qwen3_8b_fixes_candidate.json`: 12 candidate questions.
- `task_quality_tasks_qwen3_8b_fixes_stable.json`: 6 baseline-correct questions after Qwen3-0.6B `text_only` calibration.

Baseline-only calibration:

| Document | Candidate Tasks | Baseline Acc | Baseline Gate | Stable Tasks |
|---|---:|---:|---|---:|
| qwen3-8b-fixes.md | 12 | 50.00% | weak-baseline | 6 |

Policy comparison on the second-document stable set:

| Policy | Heads | Entry Saving | Compact Acc | Agreement | Mean Delta | Compact Margin | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| all_layers_0.50 | 11 | 4.30% | 100.00% | 100.00% | +0.8972 | 3.5369 | cross-doc pass |
| constrained18_l1_0.45 | 17 | 6.83% | 83.33% | 83.33% | +0.1766 | 2.2585 | fail |
| taskaware_drop_l3_h0_l23_h7 | 15 | 6.03% | 66.67% | 66.67% | +0.3109 | 2.3160 | fail |

Second-document failures:

| Policy | Failed Task | Baseline Pred | Compact Pred | Answer | Answer LogP Delta | Compact Margin |
|---|---|---|---|---|---:|---:|
| taskaware_drop_l3_h0_l23_h7 | qwen3_8b_model_choice | B | D | B | +1.0515 | -0.1506 |
| taskaware_drop_l3_h0_l23_h7 | tp_true_weight_split | A | D | A | -1.6273 | -0.0011 |
| constrained18_l1_0.45 | qwen3_8b_model_choice | B | D | B | +0.6268 | -0.9651 |

Updated conclusion:
- The 15-head task-aware policy was overfit to the first document's v3 failures.
- The conservative 11-head all-layer `0.50` policy is currently the only tested cross-document pass.
- T1.3 should not proceed to runtime integration; next work should focus on document-adaptive quality gates or multi-source recovery.
