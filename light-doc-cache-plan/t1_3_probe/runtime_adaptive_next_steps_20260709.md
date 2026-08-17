# Runtime / Document-Adaptive Next Steps for Light Doc Cache

Date: 2026-07-09

## Current Evidence

The best strict cross-document fixed-head frontier is:

- Policy: `experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050`
- Heads: 79 / 448
- Per-head selected budget: 50%
- Total KV head-token entry saving: 17.63%
- First-doc gate: 13/13
- Second-doc gate: 6/6

Late fixed-head expansion is bottlenecked by a few fragile task decisions, not by remote instability:

- `topk8_quality`: 14 flips across Round4-Round7
- `route_phase3`: 8 flips
- `sweet_spot`: 7 flips
- `quest_decode_selection`: 6 flips
- second-doc failures are rarer but include `tp_true_weight_split` and `smoothquant_status`

A mixed-budget rescue attempt failed:

- Starting from the 79-head frontier, adding `6:6`, `5:5`, `22:6`, or `15:5` at 75% selected budget still failed first-doc gates.
- A targeted `learned_values` rescue for `15:5` also failed and was much slower than ridge.

## Interpretation

The current result is useful evidence for trained/correlation recovery-bank modeling, but it is not a KR-level 2x+ global doc-cache/runtime compression result.

The fixed global head-list frontier appears near a quality wall. Additional gains probably require one of:

1. **Document-adaptive gating**: choose compact heads per document/prompt instead of one global head list.
2. **Task-aware fallback**: detect fragile query/task regimes and keep specific heads full for those queries.
3. **Runtime recovery integration**: move from offline decode scoring simulation into real KV-cache recovery path and measure actual memory/runtime.

## Recommended Next Experiment

Implement a quality-only document-adaptive gate before runtime integration:

1. Keep the 79-head safe policy as the default compact set.
2. For each prompt/task, compute cheap risk features before decode scoring:
   - baseline margin on the uncompressed prompt for the choices,
   - task/document signature if available,
   - candidate task belongs to fragile set (`topk8_quality`, `route_phase3`, `sweet_spot`, `quest_decode_selection`),
   - compact policy size / target head groups.
3. If risk is high, selectively disable compacting a small rescue set of heads for that task, rather than globally dropping those heads.
4. Report two metrics:
   - quality gate: 13/13 and 6/6 strict accuracy/agreement,
   - effective saving: average selected KV head-token entry saving over tasks after adaptive fallback.

This gives a realistic bridge between the current offline task-quality simulation and eventual runtime integration: the policy becomes conditional, and the reported compression becomes average effective compression rather than a misleading single static head count.

## Minimal Implementation Plan

Suggested new script:

- `experiments/light_doc_cache/make_adaptive_task_policy.py`

Inputs:

- base policy rows, e.g. `policy_recovery_seed78_add_l11h3_b050/policy_rows.csv`
- task failure diagnostics, e.g. `task_failure_diagnostics_round4_round7_20260709.tsv`
- fragile task list or auto-selected top-N fragile tasks
- rescue mode: `drop_layer`, `drop_heads`, or `drop_recent_heads`

Outputs:

- a JSON adaptive policy spec describing:
  - default policy directory,
  - task-id specific head overrides,
  - average expected entry saving under the task mix.

Suggested change to `task_quality_smoke.py`:

- Add optional `--adaptive-policy-file`.
- For each task, load default `policy_heads`, then apply task-specific overrides before building banks.
- Keep old `--policy-dir` behavior unchanged.

Validation:

1. Unit/static tests in `tools/test_light_doc_cache_recovery_probe.py` for adaptive-policy parsing and task override hooks.
2. Local `py_compile` and `bash -n`.
3. Remote first-doc and second-doc quality smoke using the adaptive policy.
4. Document average effective saving and compare against the static 79-head 17.63% frontier.

## Claim Boundary

If adaptive fallback passes, phrase the result as:

> A document/task-adaptive recovery-bank simulation preserves the strict two-document task gates while achieving X% average KV head-token entry saving.

Do not phrase it as:

> 2x+ runtime doc-cache compression.

A 2x+ runtime claim still needs real KV-cache integration and runtime/memory measurements.

## Update: Adaptive Policy v1 Result

Adaptive policy v1 was implemented and remotely validated.

- Spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_drop_on_fragile_v1.json`
- Default policy: `experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l15h5`
- Fallback tasks: `topk8_quality`, `route_phase3`, `sweet_spot`, `quest_decode_selection`
- Fallback action: drop `15:5`, reverting those tasks from the 80-head default policy to the 79-head safe frontier.

Results:

| Task Set | Quality Gate | Avg Effective Saving |
|---|---|---:|
| first-doc v3 stable | 13/13 accuracy and agreement | 17.71% |
| second-doc stable | 6/6 accuracy and agreement | 17.75% |

This validates the value of task-adaptive fallback in the current quality simulation. The next useful step is to replace this hand-written v1 spec with an automatic adaptive-policy generator that consumes `task_failure_diagnostics_round4_round7_20260709.tsv`, proposes fallback heads per fragile task, and reports average effective saving for the combined task mix.

## Update: Adaptive Policy Generator

The first automatic generator is now available:

- Script: `experiments/light_doc_cache/make_adaptive_task_policy.py`
- Generated spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json`

It consumes the Round4-Round7 failure diagnostics, selects the top fragile first-doc tasks, and emits the same fallback set as the hand-written v1 policy:

- `topk8_quality`
- `route_phase3`
- `sweet_spot`
- `quest_decode_selection`

This turns the v1 adaptive result into a reproducible policy-generation flow. The next higher-leverage step is to search over candidate fallback heads and default policies automatically, rather than only generating a policy for a manually selected added head.

## Update: Adaptive Candidate Search

The first small adaptive search has now been completed.

- Result table: `experiments/light_doc_cache/adaptive_search_from79_top4_two_doc_results_20260709.tsv`
- Candidates: `6:6`, `5:5`, `22:6`, `15:5`, `7:7`, `6:4`
- Shared fallback rule: for top-4 first-doc fragile tasks, drop the newly added head and use the 79-head safe frontier.

All six candidates passed the first-doc gate after fallback, but only two candidates passed the second-doc gate:

| Added Head | First Doc Gate | Second Doc Gate | Status |
|---|---|---|---|
| `22:6` | 13/13 | 6/6 | safe alternate |
| `15:5` | 13/13 | 6/6 | preferred current frontier |

The other four candidates failed second-doc tasks:

- `6:6`: failed `tp_true_weight_split`, `gpu_utilization_semantics`, `smoothquant_status`
- `5:5`: failed `smoothquant_status`
- `7:7`: failed `gpu_utilization_semantics`, `smoothquant_status`
- `6:4`: failed `gpu_utilization_semantics`, `smoothquant_status`

Current best adaptive frontier:

- Preferred spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json`
- Alternate safe spec: `experiments/light_doc_cache/adaptive_policy_from79_add_l22h6_auto_top4_first_v1.json`
- Quality gate: 13/13 first-doc and 6/6 second-doc accuracy/agreement
- Average effective saving: 17.71% first-doc and 17.75% second-doc

## Updated Next Step

Do not keep expanding one-head global defaults blindly. The useful next step is to extend `make_adaptive_task_policy.py` so it can consume both first-doc and second-doc failure diagnostics, then emit a two-doc fallback policy:

1. Keep the 80-head default candidate.
2. Apply first-doc fragile fallback as before.
3. Add optional second-doc fallback tasks such as `tp_true_weight_split`, `gpu_utilization_semantics`, and `smoothquant_status`.
4. Recompute average effective saving across both task sets.
5. Only after the two-doc adaptive quality gate is stable, move toward runtime KV-cache recovery integration.

Claim boundary remains unchanged: this is a task-adaptive recovery-bank quality simulation, not a 2x+ runtime compression result.

## Update: Two-Document Fallback Rescue

`make_adaptive_task_policy.py` now supports two-document fallback generation:

- Multiple `--failure-diagnostics` inputs.
- `--per-doc-top-tasks first=4,second=3`.
- Adaptive search result tables with `fail_tasks` can be consumed as failure sources.

The four candidates that previously failed second-doc validation were regenerated with first-doc top-4 fallback and second-doc top-3 fallback. All four now pass both strict gates:

| Added Head | First Doc Gate | Second Doc Gate | Second Saving |
|---|---|---|---:|
| `6:6` | 13/13 | 6/6 | 17.69% |
| `5:5` | 13/13 | 6/6 | 17.69% |
| `7:7` | 13/13 | 6/6 | 17.69% |
| `6:4` | 13/13 | 6/6 | 17.69% |

Result table:

- `experiments/light_doc_cache/adaptive_search_from79_twodoc_fallback_results_20260710.tsv`

This means the quality-only frontier is no longer limited to the earlier `15:5` and `22:6` safe candidates. A family of 80-head default policies can preserve the two-document quality gates when fragile first-doc and second-doc tasks fall back to the 79-head safe frontier.

## Next Best Step

Prepare a paper-ready evidence table and then prototype runtime integration:

1. Summarize static 79-head, first-doc-only adaptive, and two-doc adaptive frontiers in one table.
2. Report quality gate, average effective saving, fallback-task count, and claim boundary.
3. Start a small runtime KV-cache integration prototype only after this table is stable.
4. Runtime prototype should measure actual KV storage/recovery and not reuse the current decode-score simulation wording.

## Update: Paper-Ready Table Complete

The frontier table is now generated and checked in as an experiment artifact:

- `experiments/light_doc_cache/paper_frontier_table_20260710/frontier_table.csv`
- `experiments/light_doc_cache/paper_frontier_table_20260710/frontier_table.md`

The next step can now move from quality-only simulation toward runtime integration.

Recommended runtime prototype scope:

1. Keep it default-off and isolated from the normal TinyLLM path.
2. Start with Qwen3-0.6B and the already validated adaptive policy specs.

## Update: Calibrated Read-Path Follow-Up

The runtime prototype has now reached a trained/calibrated recovery-bank
read-path smoke:

- Bank artifact:
  `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/multi_source_recovery_bank.json`
- Read-path artifact:
  `experiments/light_doc_cache/tinyllm_sidecar_read_path_calibrated_smoke_qwen3_0_6b_20260710/`
- Target prompt: `Light Doc Cache TinyLLM target prompt for Qwen KV recovery.`
- Mode: `calibrated_multi_correlated`
- Result: argmax preserved on the target prompt (`785 -> 785`), with
  `max_abs_logit_diff=3.890625` and `mean_abs_logit_diff=0.6842344403266907`.

Interpretation:

- This validates the default-off trained-bank read-path plumbing: JSON-loaded
  calibrated weights can recover missing sidecar entries and feed one TinyLLM
  decode logits comparison without using target missing KV values as oracle.
- It is not yet evidence for a 2x+ runtime KV-cache compression claim. The
  logical saving in this smoke is the current adaptive policy's `17.6339%`, and
  the path still materializes a restored sidecar tensor for comparison.
- The same-target `repeat_last` and `correlated_same_layer` artifacts have now
  been pulled and the final matrix has been generated:
  `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_target_qwen3_0_6b_20260710/`.
- All three modes preserve argmax on the target prompt, but the trained bank
  does not beat the simple same-layer correlated baseline in this matrix.
  `correlated_same_layer_target` has the best missing-token MSE (`11.4076`) and
  mean logit diff (`0.507468`), while `calibrated_multi_correlated_target`
  reports MSE `13.5193` and mean logit diff `0.684234`.

Next concrete step:

1. Broaden calibration beyond the one calibration/target pair, including more
   prompts and possibly per-layer/source-head selection.
2. Re-run the artifact-backed target matrix before changing attention hot path
   or physical KV allocation.
3. Only move toward runtime allocation changes once a non-oracle trained bank
   clearly beats the cheap same-layer correlated baseline on logits/error.

## Update: Multi-Prompt Calibration Smoke

The first broadened-calibration step is implemented and remotely validated.

Code changes:

- `run_tinyllm_calibrated_kv_smoke.py` accepts repeatable
  `--calibration-prompt-extra` and optional `--calibration-prompts-file`.
- Calibration KV samples are packed by prompt-prefix tokens, not by cloning the
  full preallocated cache. This fixed the initial remote OOM where three full
  `kv_cache.detach().clone()` tensors exceeded GPU memory.
- Fitting now uses a calibration-length plan, while target storage/readback uses
  the target-length plan.

Artifacts:

- `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_multiprompt/`
- `experiments/light_doc_cache/tinyllm_sidecar_read_path_calibrated_multiprompt_smoke_qwen3_0_6b_20260713/`
- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_multiprompt_target_qwen3_0_6b_20260713/`

Result:

- Calibration tokens: `31`; target tokens: `14`.
- KV-only multi-prompt bank MSE: `11.6785` vs previous single-pair `13.5483`.
- Read-path multi-prompt row: missing MSE `9.55884`, max logit diff `4.0625`,
  mean logit diff `0.538956`, argmax `785 -> 785`.
- Extended matrix: multi-prompt calibrated is now best on missing-token MSE, but
  `correlated_same_layer_target` still has the best mean logit diff (`0.507468`).

Next concrete step:

1. Improve source-head selection for the calibrated bank instead of using a
   fixed same-layer top-2 source rule.
2. Consider scoring source candidates by calibration holdout error or by
   read-path/logit sensitivity if affordable.
3. Keep attention hot path and physical KV allocation unchanged until the
   trained bank beats cheap baselines in the artifact-backed matrix.

## Update: Calibration-Fit Source Selection

Implemented and tested `--source-map calibration_fit` in
`run_tinyllm_calibrated_kv_smoke.py`. It selects retained source heads by
prefix reconstruction fit on calibration KV instead of fixed same-layer order.

Remote artifacts:

- `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_calfit/`
- `experiments/light_doc_cache/tinyllm_sidecar_read_path_calfit_smoke_qwen3_0_6b_20260713/`
- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_sourcefit_target_qwen3_0_6b_20260713/`

Result:

- KV-only MSE: `20.7961`, worse than same-layer multi-prompt `11.6785`.
- Read-path row: missing MSE `11.5086`, max logit diff `3.4375`, mean logit
  diff `0.533486`, argmax `785 -> 785`.
- Compared with multi-prompt same-layer trained bank, calibration-fit improves
  max logit diff (`4.0625 -> 3.4375`) and slightly improves mean logit diff
  (`0.538956 -> 0.533486`), but worsens missing MSE (`9.55884 -> 11.5086`).
- It still does not beat the cheap `correlated_same_layer_target` baseline on
  mean logit diff (`0.507468`).

Next concrete step:

1. Replace prefix-fit selection with held-out missing-token selection: fit on
   part of calibration tokens and select sources by predicting held-out tokens.
2. If feasible, add a small read-path/logit-aware selector over a reduced
   candidate set.
3. Do not move to hot-path integration until the trained bank beats the cheap
   baseline on the artifact-backed read-path matrix.

## Update: Calibration-Holdout Source Selection

Implemented and tested `--source-map calibration_holdout`.

Method:

- For each target compact head and candidate retained source head, fit an affine
  map on the selected calibration prefix.
- Score the candidate on held-out calibration tokens.
- Pick the lowest held-out error sources.

Remote artifacts:

- `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_holdout/`
- `experiments/light_doc_cache/tinyllm_sidecar_read_path_holdout_smoke_qwen3_0_6b_20260713/`
- `experiments/light_doc_cache/read_path_recovery_matrix_calibrated_holdout_target_qwen3_0_6b_20260713/`

Result:

- KV-only holdout bank MSE: `7.07867`.
- Read-path holdout row: missing MSE `11.7772`, max logit diff `3.10938`,
  mean logit diff `0.455011`, argmax `785 -> 785`.
- This is the first trained/calibrated row to beat the cheap same-layer
  correlated baseline on mean logit diff (`0.455011` vs `0.507468`) and max
  logit diff (`3.10938` vs `3.59375`).
- It does not beat the best MSE row: multi-prompt same-layer remains best on
  missing-token MSE (`9.55884`).

Next concrete step:

1. Confirm holdout selector on more target prompts, not just this single target
   prompt.
2. Generate a multi-target read-path matrix before any hot-path integration.
3. If stable, package the selector/bank generation flow into a reproducible
   experiment script rather than ad hoc remote commands.

## Update: Runtime Planning Prototype Started

The first default-off runtime artifact is now implemented:

- Module: `tinyvllm/engine/light_doc_cache_runtime.py`
- Tests: `tools/test_light_doc_cache_runtime.py`

Current scope:

1. Load adaptive policy specs generated by `make_adaptive_task_policy.py`.
2. Map `task_id` fallback overrides to added-head enable/drop decisions.
3. Estimate request-level KV head-token entries:
   - total entries,
   - stored entries,
   - recovered/saved entries,
   - stored/recovered equivalent KV-head counts,
   - compression ratio.
4. Keep the estimate budget-aware. A `budget75` head saves only 25% of that head's doc-token entries; `b050` saves 50%.

Validation command:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/tinyllmforge_pycache \
python3 -m pytest -q tools/test_light_doc_cache_runtime.py
```

Boundary:

- This is still not runtime KV-cache compression.
- It does not mutate `self.kv_cache`, allocate compact KV storage, or recover K/V tensors online.
- It is a safer bridge from quality-only adaptive policies to a future `ModelRunner` integration because it makes the planned storage/recovery accounting explicit.

Next concrete step:

1. Add a default-off summary hook or standalone smoke around `ModelRunner` inputs that calls `build_light_doc_cache_runtime_plan(...)` for a request.
2. Compare planned entry counts with the existing full KV cache shape from `allocate_kv_cache()`.
3. Only after the accounting matches expectations, prototype a real compressed storage layout for selected heads and measure actual memory/runtime.

## Update: Real Policy Rows Runtime Table

The planner now reads real policy directories:

- `default_policy_dir/policy_rows.csv`
- `base_safe_policy_dir/policy_rows.csv`

This matters because the current adaptive frontier is mixed-budget:

- 79-head base: 50% selected-token budget, so `39.50` KV-head equivalents are recoverable/saved.
- added `6:4` default head: 75% selected-token budget, so it adds only `0.25` recoverable/saved KV-head equivalent.
- fallback tasks drop the added head and revert to the base `39.50` equivalent.

Generated 19-task planning artifact:

- `experiments/light_doc_cache/runtime_plan_table_from79_l6h4_20260710/runtime_plan_table.md`
- Average effective planned saving: `17.70%`
- Average planned compression ratio: `1.2151x`
- Full KV bytes for audited shape `[2, 28, 6, 256, 8, 128]` fp16/bf16: `176,160,768`
- Average planned recovered/saved KV bytes: `31,188,237`
- Average planned stored KV bytes: `144,972,531`

Updated next step:

1. Keep using `make_runtime_plan_table.py` to audit policy/accounting changes.
2. Add a default-off `ModelRunner` summary path that can report the same planned metrics alongside the real allocated full KV cache shape.
3. Then implement the first real storage prototype: retain only selected-token slices for planned compact heads, keep full storage for non-compact/fallback heads, and explicitly measure allocated bytes and reconstruction latency.

Do not claim runtime compression until step 3 measures actual storage and online recovery.

## Update: Byte-Level Planning Summary

`tinyvllm/engine/light_doc_cache_runtime.py` now includes `summarize_planned_kv_storage(...)`, which maps a plan to full-cache byte accounting:

- full KV cache shape,
- full KV bytes,
- planned stored KV bytes,
- planned recovered/saved KV bytes,
- planned byte saving fraction.

This function is pure accounting and does not allocate compressed tensors. The next safe integration is to call it from a default-off `ModelRunner` summary method after `allocate_kv_cache()` has real `self.kv_cache.shape`, then compare:

1. actual allocated `self.kv_cache.numel() * element_size`;
2. planned stored bytes from the adaptive policy;
3. delta that a future compressed-layout prototype must realize physically.

The helper now also has `summarize_planned_kv_storage_from_shape(...)`, and `make_runtime_plan_table.py` accepts `--kv-cache-shape 2,L,B,block,H,D`. This is the intended interface for the next default-off `ModelRunner` hook:

```python
summarize_planned_kv_storage_from_shape(
    plan,
    full_cache_shape=tuple(self.kv_cache.shape),
    element_size_bytes=self.kv_cache.element_size(),
)
```

Keep this hook summary-only until a separate compressed storage implementation can prove that allocated bytes actually decrease.

## Update: ModelRunner Summary Wrapper

`tinyvllm/engine/model_runner.py` now has a summary-only wrapper:

```python
runner.light_doc_cache_planning_summary(plan)
```

It delegates to `build_model_runner_light_doc_cache_summary(...)` and reads only:

- `runner.kv_cache.shape`
- `runner.kv_cache.element_size()`

It does not change runtime behavior. The next validation step, before any real compressed storage work, is a remote smoke in the CUDA/model environment:

1. sync the touched files to `sitian@10.232.195.203`;
2. run remote `py_compile` for `tinyvllm/engine/model_runner.py` and `tinyvllm/engine/light_doc_cache_runtime.py`;
3. optionally instantiate a tiny Qwen3-0.6B runner and call `light_doc_cache_planning_summary(plan)` after `allocate_kv_cache()` to confirm the reported full bytes match `self.kv_cache.numel() * self.kv_cache.element_size()`.

Only after that should the work move to real compressed storage.

## Update: CPU Storage Prototype

`LightDocCacheCompressedKVStorage` is now available as an isolated CPU/numpy storage prototype. It is the first step beyond accounting:

- stores full tensors for non-compact heads;
- stores only selected prefix tokens for compact heads;
- restores the original KV shape;
- fills missing compact-head tokens with a sentinel value.

Smoke artifact:

- `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710/storage_prototype_report.md`

Toy result:

- Full tensor bytes: `57,344`
- Stored tensor bytes: `35,424`
- Saved tensor bytes: `21,920`
- Byte saving fraction: `38.23%`

Boundary:

- This proves sparse storage can physically store fewer bytes in a toy layout.
- It does not prove runtime quality because missing compact-head tokens are not recovered yet.
- The next step should be a torch-based remote smoke that uses the same storage API on tensors, followed by a recovery-fill path using the existing ridge/recovery-bank artifacts.
3. Apply policy decisions to actual KV-cache materialization/recovery, not only choice-score evaluation.
4. Measure:
   - allocated/stored KV entries or bytes,
   - recovered/fallback head counts per task,
   - decode correctness on the same task prompts,
   - wall-clock and bank/recovery overhead.
5. Keep the claim separate: runtime prototype may validate implementation feasibility, while the current 17.7% number remains the offline quality-simulation frontier until runtime memory measurements match it.

## Update: Recovery-Fill Callback

`LightDocCacheCompressedKVStorage.restore_to_full_shape(...)` now accepts `recover_missing_fn`. This turns the storage prototype into a pluggable recovery layout:

1. compact heads store only selected prefix tokens;
2. the callback receives compact/head metadata and stored tokens;
3. the callback returns missing tokens with shape `[2, missing_tokens, head_dim]`;
4. storage writes those tokens into the restored full KV shape.

The smoke now supports `--recover-mode none|fill|repeat_last|linear_tail|oracle`. Oracle mode copies original missing tokens only to validate storage layout and error accounting, producing zero missing-token error by construction. Repeat-last uses only the stored prefix tokens and repeats the last stored token into missing compact-head positions, so it is the first deterministic non-oracle recovery baseline. Linear-tail fits a toy ridge-linear trend from stored prefix tokens and extrapolates missing tokens, which verifies fitted-callback plumbing without using oracle KV values.

Latest repeat-last artifact:

- Path: `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710/storage_prototype_report.md`
- Recovery mode: `linear_tail`
- KV pattern: `nonlinear`
- Missing-token MSE: `17.65`
- Missing-token max abs error: `7.41`
- Missing compact tokens: `474`
- Byte saving fraction: `38.23%`

Boundary: this is now a nonlinear toy KV tensor, so linear-tail no longer gets misleading near-zero error from `np.arange`. It is still an adapter/API validation baseline, not evidence of model-quality KV recovery.

The next meaningful step is to connect a real prompt-derived ridge/recovery-bank adapter from `task_quality_smoke.py` style bank construction, comparing restored-token error against fill, repeat-last, linear-tail, and oracle baselines.

## Update: Real HF KV Storage Smoke

The storage prototype now has a real-model tensor smoke:

- Script: `experiments/light_doc_cache/run_real_kv_storage_smoke.py`
- Artifact: `experiments/light_doc_cache/real_kv_storage_smoke_qwen3_0_6b_20260710/real_kv_storage_report.md`
- Model: Qwen3-0.6B HF `past_key_values`
- KV cache shape: `[2, 28, 1, 16, 8, 128]`
- Recovery mode: `linear_tail`
- Byte saving fraction: `38.23%`
- Missing-token MSE: `23.19`
- Missing-token max abs error: `302`

This is stronger than the CPU toy smoke because the input tensors come from a real model forward pass. It is still offline and not in the TinyLLM runtime hot path.

Next runtime step:

1. Add a default-off TinyLLM `ModelRunner` smoke hook that exports/inspects allocated `kv_cache` after prefill without changing decode outputs.
2. Compare the HF real-KV shape/accounting against TinyLLM's allocated KV layout for the same short prompt/model.
3. Only after shape/accounting matches, prototype a runtime storage sidecar for compact heads.
4. Keep recovery-quality work separate: real ridge/recovery-bank adapter should be evaluated against fill/repeat-last/linear-tail/oracle before any accuracy claim.

## Update: TinyLLM ModelRunner KV Summary Smoke

The default-off TinyLLM accounting smoke is now implemented:

- Script: `experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py`
- Artifact: `experiments/light_doc_cache/tinyllm_kv_summary_smoke_qwen3_0_6b_20260710/tinyllm_kv_summary_report.md`
- Model: Qwen3-0.6B through TinyLLM `LLM` / `ModelRunner`
- Prompt tokens: `10`
- Allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Allocated KV cache bytes: `23,634,903,040`
- Logical full KV bytes for plan seq_len: `1,146,880`
- Planned recovered KV bytes: `202,240`
- Planned byte saving fraction: `17.63%`

This closes the shape/accounting bridge from:

1. offline policy rows,
2. CPU toy storage,
3. real HF `past_key_values`,
4. real TinyLLM `ModelRunner.kv_cache` allocation.

Next runtime step:

1. Add a default-off `ModelRunner` sidecar prototype that materializes compact-head selected tokens from the actual `kv_cache` after prefill.
2. Compare sidecar `stored_tensor_bytes` against the logical planned bytes and allocated bytes.
3. Add a readback/restore smoke that reconstructs compact heads into a temporary full-shape tensor for error evaluation.
4. Only after sidecar storage/readback is verified should attention read-path integration be attempted.

## Update: TinyLLM Sidecar Materialization / Readback Smoke

The first default-off TinyLLM sidecar storage smoke is now implemented and remotely validated:

- Script: `experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py --write-sidecar-storage`
- Artifact: `experiments/light_doc_cache/tinyllm_sidecar_storage_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_storage_report.md`
- Model: Qwen3-0.6B through TinyLLM `LLM` / `ModelRunner`
- Prompt tokens: `11`
- Allocated TinyLLM KV shape: `[2, 28, 805, 256, 8, 128]`
- Recovery mode: `linear_tail`
- Runtime policy task: `smoothquant_status`, so the adaptive fallback drops added head `6:4`
- Compact heads materialized into sidecar: `79`
- Full heads copied into sidecar: `145`

Byte accounting:

- Allocated-capacity full tensor bytes: `23,634,903,040`
- Sidecar stored tensor bytes: `1,059,328`
- Allocated-capacity saving fraction: `99.9955%`
- Logical full KV bytes for 11 prompt tokens: `1,261,568`
- Logical stored KV bytes: `1,059,328`
- Logical byte saving fraction: `16.03%`

Readback/error smoke:

- Missing compact-head tokens: `395`
- Missing-token MSE: `28.7971`
- Missing-token max abs error: `274`

Boundary:

- This is the first materialization/readback smoke from real TinyLLM `ModelRunner.kv_cache`.
- The large `99.9955%` number is relative to TinyLLM's full preallocated KV cache capacity, not a runtime memory-reduction claim.
- The more relevant prompt-level sidecar accounting is the logical `16.03%` byte saving for this 11-token plan.
- Attention still reads the original full `kv_cache`; no hot-path storage replacement, online recovery, latency saving, or end-to-end quality claim is made yet.

## Update: ModelRunner Sidecar Inspection Hook

The default-off `ModelRunner` inspection hook is now implemented:

```python
runner.light_doc_cache_materialize_sidecar(plan, evaluate_readback=True)
```

It routes through `materialize_model_runner_light_doc_cache_sidecar(...)`, stores the sidecar object on `runner.light_doc_cache_sidecar`, and returns:

- sidecar full/stored/saved byte accounting;
- logical full/stored/saved bytes for `plan.seq_len`;
- optional readback error metrics for missing compact-head tokens.

`run_tinyllm_kv_summary_smoke.py --write-sidecar-storage` now calls this `ModelRunner` wrapper when available, so the TinyLLM smoke covers the runner integration point rather than only the standalone storage helper.

Remote wrapper smoke is validated and mirrored locally:

- Artifact: `experiments/light_doc_cache/tinyllm_sidecar_storage_wrapper_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_storage_report.md`
- Prompt tokens: `11`
- KV shape: `[2, 28, 805, 256, 8, 128]`
- Logical stored KV bytes: `1,059,328`
- Logical byte saving fraction: `16.03%`
- Missing-token MSE: `28.4588`
- Missing-token max abs error: `274`

Boundary:

- The hook is default-off and never called by `ModelRunner.run(...)`.
- It does not replace `self.kv_cache`.
- It does not change slot mapping, attention reads, KV writes, or CUDA graph behavior.
- It is still an inspection/materialization hook, not runtime memory reduction.

Next runtime integration step:

1. Add a separate default-off read-path experiment that reconstructs compact heads into a temporary attention-visible buffer and compares logits/output on a tiny prompt.
2. Keep the original `kv_cache` allocation alive during this experiment; do not attempt memory lifetime/allocation changes yet.
3. Only after read-path correctness is stable should memory lifetime/allocation changes be considered.

## Update: Restored Sidecar Read-Path Smoke

The default-off read-path experiment is now implemented:

- Script: `experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py`
- Artifact: `experiments/light_doc_cache/tinyllm_sidecar_read_path_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_report.md`

It does not modify attention kernels. Instead, it:

1. runs normal TinyLLM prefill;
2. materializes the Light Doc Cache sidecar;
3. restores the sidecar into a temporary full KV tensor;
4. temporarily points each attention layer's `k_cache` / `v_cache` at the restored tensor;
5. runs one decode step and compares logits against the original full-cache read path;
6. restores all original cache pointers.

Remote result:

- Prompt tokens: `13`
- KV shape: `[2, 28, 805, 256, 8, 128]`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing compact tokens: `553`
- Missing-token MSE: `38.5860`
- Max abs logit diff: `17.5`
- Mean abs logit diff: `3.1362`
- Argmax match: `False`

Interpretation:

- The restored sidecar buffer is shape-compatible with TinyLLM's existing decode read path.
- The current toy `linear_tail` recovery is not logit-preserving on this prompt.
- This is useful negative evidence: the next recovery step should not be memory-lifetime changes, but a stronger recovery path.

Next runtime integration step:

1. Run the same read-path smoke with `--recover-mode oracle` to establish the layout/read-path upper bound.
2. If oracle logits match, integrate a trained/correlation recovery callback and compare against `fill`, `repeat_last`, and `linear_tail`.
3. Only after read-path logits are stable should attention hot-path or allocation-lifetime changes be attempted.

## Update: Oracle Read-Path Upper Bound

The same default-off read-path smoke now passes with oracle recovery:

- Artifact: `experiments/light_doc_cache/tinyllm_sidecar_read_path_oracle_smoke_qwen3_0_6b_20260710/tinyllm_sidecar_read_path_report.md`
- Prompt tokens: `13`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing compact tokens: `553`
- Missing-token MSE: `0`
- Max abs logit diff: `0`
- Mean abs logit diff: `0`
- Argmax match: `True`

Interpretation:

- Sidecar storage, restore indexing, and the temporary decode read-path pointer swap are layout-correct.
- The previous `linear_tail` read-path failure is a recovery-quality failure, not a KV layout or read-path compatibility failure.

Next runtime integration step:

1. Implement a real trained/correlation recovery callback compatible with `recover_missing_fn`.
2. Compare read-path logits for `fill`, `repeat_last`, `linear_tail`, oracle, and the trained/correlation callback.
3. Only if the trained/correlation callback gets stable logits/output should we proceed to attention hot-path integration or physical allocation-lifetime changes.

## Update: First Correlated-Head Runtime Callback

The first non-oracle correlated recovery callback is implemented:

- Runtime API: `make_correlated_head_recovery_callback(storage, source_heads=..., ridge=...)`
- Read-path smoke mode: `--recover-mode correlated`
- Test coverage: prefix-only affine/ridge fit from a retained full source head,
  missing-source validation, and source-head mapping for the read-path smoke.

Remote Qwen3-0.6B read-path result:

- Artifact: `experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_smoke_qwen3_0_6b_20260710/`
- Logical byte saving fraction: `18.99%`
- Missing compact tokens: `553`
- Missing-token MSE: `11.5488`
- Max abs logit diff: `5.21875`
- Mean abs logit diff: `0.886142`
- Argmax match: `False`

This is a clear improvement over the previous `linear_tail` read-path baseline
(`MSE 38.5860`, max logit diff `17.5`, mean logit diff `3.1362`), but it still
does not preserve logits/argmax. Treat it as runtime-compatible plumbing plus a
stronger non-oracle baseline, not as the final trained recovery module.

Updated next best step:

1. Keep `same_layer` as the default correlated source-map baseline.
2. Treat `prefix_fit` source selection as an ablation, not as the default:
   it reduced prefix reconstruction error by construction but was worse on
   decode logits for the Qwen3-0.6B read-path smoke.
3. Extend recovery from one source head to multi-source ridge/MLP using retained
   heads and stored target prefixes, with decode-logit preservation as the
   selection objective.
4. Run a small prompt suite comparing `repeat_last`, `linear_tail`,
   `correlated/same_layer`, `correlated/prefix_fit`, and `oracle` on
   missing-token error and decode logits.
5. Only after non-oracle recovery preserves logits/output should we attempt
   attention hot-path integration or physical KV allocation-lifetime changes.

## Update: Prefix-Fit Source Selection Ablation

`build_correlated_source_head_map(storage, ridge=...)` is now implemented for
controlled source-head selection experiments. It picks the retained full source
head with the lowest affine prefix reconstruction error for each compact target
head.

The TinyLLM read-path smoke exposes this explicitly:

- default: `--correlated-source-map same_layer`
- ablation: `--correlated-source-map prefix_fit`

Remote Qwen3-0.6B prefix-fit result:

- Artifact: `experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_prefixfit_smoke_qwen3_0_6b_20260710/`
- Missing-token MSE: `13.2058`
- Max abs logit diff: `6.9375`
- Mean abs logit diff: `1.10957`
- Argmax match: `False`

This is worse than the previous same-layer correlated baseline:

- Missing-token MSE: `11.5488`
- Max abs logit diff: `5.21875`
- Mean abs logit diff: `0.886142`
- Argmax match: `False`

Conclusion: a short-prefix per-head MSE selector is not enough; source
selection should be optimized for downstream decode logits or upgraded to a
multi-source trained recovery module.

## Update: Read-Path Recovery Matrix

The same-prompt Qwen3-0.6B read-path matrix is now available:

- `experiments/light_doc_cache/read_path_recovery_matrix_qwen3_0_6b_20260710/read_path_recovery_matrix.md`
- `experiments/light_doc_cache/read_path_recovery_matrix_qwen3_0_6b_20260710/read_path_recovery_matrix.csv`

| Mode | Missing MSE | Max Logit Diff | Mean Logit Diff | Argmax Match |
|---|---:|---:|---:|---|
| `repeat_last` | 13.7399 | 5.5625 | 0.787285 | False |
| `linear_tail` | 38.5860 | 17.5 | 3.13622 | False |
| `correlated_same_layer` | 11.5488 | 5.21875 | 0.886142 | False |
| `correlated_prefix_fit` | 13.2058 | 6.9375 | 1.10957 | False |
| `multi_correlated2` | 18.3975 | 6.96875 | 1.04723 | False |
| `oracle` | 0 | 0 | 0 | True |

Updated conclusion:

- Single-source non-oracle callbacks are still insufficient for argmax/logit
  preservation.
- Missing-token MSE and mean logit diff can rank methods differently, so the
  next selector/trainer must include decode-logit metrics, not only KV MSE.
- A multi-source correlated callback is now implemented and mechanically
  validated, but `multi_correlated2` is worse than the single-source same-layer
  baseline on this short prompt.
- The next concrete implementation should avoid more per-prompt short-prefix
  least-squares variants. Instead, train/calibrate recovery coefficients across
  more tokens/prompts or select recovery maps with decode-logit metrics.

## Update: Multi-Source Correlated Runtime Callback

`make_multi_source_correlated_head_recovery_callback(...)` is implemented and
available through the read-path smoke:

```bash
--recover-mode multi_correlated --multi-correlated-source-count 2
```

Toy tests show it can recover compact heads that are linear combinations of two
retained heads, including multi-dimensional head values.

Remote Qwen3-0.6B result:

- Artifact: `experiments/light_doc_cache/tinyllm_sidecar_read_path_multi_correlated2_smoke_qwen3_0_6b_20260710/`
- Missing-token MSE: `18.3975`
- Max abs logit diff: `6.96875`
- Mean abs logit diff: `1.04723`
- Argmax match: `False`

Conclusion: the implementation path is ready, but this untrained per-prompt
two-source ridge is not sufficient. The next useful work is an offline-trained
or calibrated recovery module, not attention hot-path integration.

## Update: Offline-Calibrated Recovery-Bank API

The runtime now has an explicit offline-calibrated recovery entry point:

```python
bank = fit_multi_source_recovery_bank(
    calibration_kv,
    plan,
    source_heads={(layer, target_head): [(layer, source_head_a), ...]},
    ridge=1e-6,
)
save_multi_source_recovery_bank(bank, "bank.json")
bank = load_multi_source_recovery_bank("bank.json")
callback = make_calibrated_multi_source_recovery_callback(storage, bank)
```

The fitter learns per-target/per-KV/per-dimension weights from a calibration KV
tensor. The runtime callback applies those weights using only retained source
heads in the sidecar storage, so it does not use runtime target missing KV
values.

Current validation:

- Toy test passes with different calibration/runtime KV tensors.
- This verifies the trained-recovery API shape and no-oracle apply path.
- TinyLLM calibrated KV smoke now produces a real Qwen3-0.6B bank artifact:
  `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/multi_source_recovery_bank.json`.

Next concrete step:

1. Run the read-path smoke with the generated bank:
   `--recover-mode calibrated_multi_correlated --recovery-bank-file <bank.json>`.
2. Re-run the recovery matrix and compare against `repeat_last`,
   `correlated_same_layer`, `multi_correlated2`, and `oracle`.
3. If logits still fail, expand calibration beyond one prompt pair before
   changing attention hot-path or KV allocation lifetime.

Do not move to attention hot-path or KV allocation lifetime changes until a
calibrated non-oracle bank preserves logits/output on the read-path smoke.

## Update: TinyLLM Calibrated KV Smoke

`run_tinyllm_calibrated_kv_smoke.py` now fits a calibrated bank from real
TinyLLM `ModelRunner.kv_cache` on a calibration prompt and applies it to a
target prompt.

Remote Qwen3-0.6B artifact:

- `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/`
- Bank: `multi_source_recovery_bank.json`
- Calibration tokens: `14`
- Target tokens: `14`
- Missing-token MSE: `13.5483`
- Missing-token max abs error: `221`

This is the first real TinyLLM calibrated-bank artifact. It is still KV-error
only, not a decode-logit read-path result. The next concrete step is to feed
this bank into `run_tinyllm_sidecar_read_path_smoke.py` via
`--recover-mode calibrated_multi_correlated`.
