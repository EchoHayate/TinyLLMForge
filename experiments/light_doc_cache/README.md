# Light Doc Cache Experiments

This folder contains executable artifacts for the Light Doc Cache T1.3/T4-style probes.

## Main files

- `probe_am_compact_cache.py`: attention-output recovery probe used to build compact-head policies.
- `train_recovery_probe.py`: offline trainable recovery MVP. It trains a per-head residual MLP over compact attention outputs and compares it with direct compact output plus ridge-fitted compact values.
- `make_attention_output_policy.py`: converts AM probe rows into thresholded compact/full policies.
- `task_quality_smoke.py`: task-level quality simulation that compares baseline scoring with compact-bank scoring.
- `run_recovery_probe_remote.sh`: remote Qwen3-0.6B runner for the trainable recovery MVP.
- `run_task_quality_smoke_remote.sh`: remote Qwen3-0.6B runner with script/task transfer and result mirroring.
- `task_quality_tasks_kv_sparse_v1.json`: 12-question exploratory task file.
- `task_quality_tasks_kv_sparse_v2.json`: 9-question baseline-stable task file used for the current main extended smoke.

## Trainable recovery MVP

The fixed/head-threshold policy search currently overfits across documents. The next research direction is a trained correlation/recovery module:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
LOCAL_TEXT_FILES="docs/kv-sparse-attention.md docs/qwen3-8b-fixes.md" \
OUTPUT_DIR=/data00/home/sitian/light-doc-cache-work/probe/runs/recovery_probe_qwen3_0_6b_s1536_b64_2docs \
LOCAL_OUT_DIR=experiments/light_doc_cache/recovery_probe_2docs_latest \
BUDGETS=64 \
MAX_HEADS=8 \
EPOCHS=100 \
  experiments/light_doc_cache/run_recovery_probe_remote.sh
```

Outputs:

- `recovery_head_rows.csv`: per doc/layer/KV-head metrics.
- `summary.json`: aggregate decision and recovery gains.
- `report.md`: Markdown summary.

This is intentionally an offline teacher-student probe. It does not modify runtime attention yet. The teacher target is full attention output; the compact input is produced from selected doc-cache tokens. The MVP reports direct compact R², ridge-value recovery R², residual-MLP recovery R², and the non-degrading recovery envelope. A runtime implementation should only start after multi-document recovery metrics and task-quality smokes show a stable gain.

Initial 2-document Qwen3-0.6B smoke results:

| Run | Layer Range | Heads | Budget | Mean Direct Val R² | Mean Ridge/Recovery Val R² | Decision |
|---|---|---:|---:|---:|---:|---|
| `recovery_probe_2docs_smoke_latest` | layer 0 | 16 | 64/512 | -0.3912 | -0.0491 | `RECOVERY_WEAK` |
| `recovery_probe_2docs_l10_smoke_latest` | layer 10 | 16 | 64/512 | -0.6530 | 0.0862 | `RECOVERY_WEAK` |
| `recovery_probe_2docs_l10_learnedv_smoke_latest` | layer 10 | 16 | 64/512 | -0.6530 | 0.0862 | `RECOVERY_WEAK` |

Interpretation: compact output itself is very weak at 12.5% token budget. Ridge-value recovery improves substantially over direct compact output, especially at layer 10, but the current residual MLP / learned compact-values path does not beat the ridge closed-form on holdout. The next useful change is not runtime integration; it is a stronger recovery objective/feature set, for example training against attention logits/mass plus values, adding cross-layer/head features, or training a shared multi-doc module instead of independent per-head fits.

## Current recommended stress smoke

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
POLICY_DIR=/data00/home/sitian/light-doc-cache-work/probe/runs/policy_taskaware_drop_l3_h0_l23_h7 \
THRESHOLDS=0.45 \
TASK_FILE=experiments/light_doc_cache/task_quality_tasks_kv_sparse_v3_stable.json \
OUTPUT_DIR=/data00/home/sitian/light-doc-cache-work/probe/runs/task_quality_smoke_qwen3_0_6b_s1536_taskaware_drop_l3_h0_l23_h7_v3 \
LOCAL_OUT_DIR=experiments/light_doc_cache/task_quality_smoke_taskaware_drop_l3_h0_l23_h7_v3_latest \
  experiments/light_doc_cache/run_task_quality_smoke_remote.sh
```

Latest single-document stress result: the task-aware pair-drop policy `drop_l3_h0__l23_h7` passes the expanded v3 stable task set. It compresses 15 heads, saves about `6.03%` KV head-token entries, and preserves 13/13 baseline-correct v3 tasks.

Latest multi-document stress result: `drop_l3_h0__l23_h7` does **not** generalize to `docs/qwen3-8b-fixes.md` (4/6 compact accuracy). The only policy tested so far that passes both documents is the more conservative all-layer threshold `0.50` policy with 11 heads and about `4.30%` entry saving.

## Notes

- `task_quality_smoke.py` defaults to `--choice-scoring text_only`, which passed the baseline gate in calibration.
- `--task-file` accepts either a JSON list of tasks or an object with a `tasks` list.
- Runner mirrors `report.md`, `summary.json`, `task_rows.csv`, and `tasks.json` into `LOCAL_OUT_DIR`.

## Runtime Planning Prototype

The first runtime-side artifact is now a default-off planning/metrics helper, not a hot-path KV-cache implementation:

- Module: `tinyvllm/engine/light_doc_cache_runtime.py`
- Tests: `tools/test_light_doc_cache_runtime.py`
- Table generator: `experiments/light_doc_cache/make_runtime_plan_table.py`

Example:

```bash
PYTHONPATH=$PWD python3 tinyvllm/engine/light_doc_cache_runtime.py \
  --enabled \
  --policy-file experiments/light_doc_cache/adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json \
  --task-id smoothquant_status \
  --doc-id second \
  --seq-len 2048 \
  --num-layers 28 \
  --num-kv-heads 8 \
  --base-recovered-heads 11:3 \
  --base-budget-fraction 0.5
```

This converts an adaptive policy spec into a request-level summary with:

- `recovered_heads`, `applied_added_heads`, and `dropped_added_heads`;
- stored/recovered KV head-token entries;
- stored/recovered equivalent KV-head counts after budget fractions;
- effective saving fraction and compression ratio.

Important boundary: this helper only estimates the KV entries that a future runtime path should retain/recover. It does **not** allocate a compressed KV layout, recover tensors online, or measure latency/memory. It is intended to prevent claim drift before wiring a real `ModelRunner`/attention integration.

Latest generated planning table:

- Output: `experiments/light_doc_cache/runtime_plan_table_from79_l6h4_20260710/runtime_plan_table.md`
- Policy: `adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json`
- Task files: 13 first-doc v3 stable tasks + 6 second-doc stable tasks
- Average effective planned saving: `17.70%`
- Average planned compression ratio: `1.2151x`
- Full KV bytes for the audited shape `[2, 28, 6, 256, 8, 128]` fp16/bf16: `176,160,768`
- Average planned recovered/saved KV bytes: `31,188,237`
- Average planned stored KV bytes after applying policy budgets: `144,972,531`

The planner reads real `policy_rows.csv` files, so mixed budgets are counted correctly: the 79-head base uses 50% selected-token budgets and the added `6:4` default head uses a 75% selected-token budget. Fallback tasks drop only the added head and therefore revert from `39.75` to `39.50` recovered KV-head equivalents.

`make_runtime_plan_table.py` also accepts `--kv-cache-shape 2,L,B,block,H,D`, matching the shape of `ModelRunner.kv_cache`. This keeps the current integration default-off while making the accounting directly comparable with a real allocated KV cache.

`ModelRunner` now exposes a thin summary-only wrapper:

```python
runner.light_doc_cache_planning_summary(plan)
```

It delegates to `build_model_runner_light_doc_cache_summary(...)`, reads only `runner.kv_cache.shape` and `runner.kv_cache.element_size()`, and returns planned-vs-full byte accounting. It does not change `allocate_kv_cache()`, slot mapping, attention, or KV writes.

The next default-off `ModelRunner` inspection wrapper is also available:

```python
runner.light_doc_cache_materialize_sidecar(plan, evaluate_readback=True)
```

It materializes a sidecar from the current `runner.kv_cache`, stores it on `runner.light_doc_cache_sidecar`, and returns sidecar/logical-byte accounting plus optional readback error metrics. This remains an inspection hook: attention still reads the original `kv_cache`, and the full TinyLLM allocation is not replaced.

## CPU Storage Prototype

The first actual storage prototype is isolated from `ModelRunner` and runs on CPU/numpy:

- Class: `LightDocCacheCompressedKVStorage`
- Smoke script: `experiments/light_doc_cache/run_storage_prototype_smoke.py`
- Output: `experiments/light_doc_cache/storage_prototype_smoke_l6h4_20260710/`

It stores full tensors for non-compact heads and only selected prefix tokens for compact heads, then restores the original KV shape with a pluggable missing-token callback. This proves real tensor bytes can be stored sparsely in a toy layout, but it is **not** an online recovery-quality result.

`restore_to_full_shape(recover_missing_fn=...)` now accepts an external recovery callback. The smoke supports constant fill, deterministic repeat-last, toy linear-tail, and oracle modes. Oracle mode copies missing tokens from the original full KV tensor only to verify layout and error metrics. Repeat-last and linear-tail use only stored prefix tokens, but neither is the trained attention-output recovery bank.

Latest toy smoke:

- Full tensor bytes: `57,344`
- Stored tensor bytes: `35,424`
- Saved tensor bytes: `21,920`
- Byte saving fraction: `38.23%`
- Recovery mode: `linear_tail`
- KV pattern: `nonlinear`
- Missing-token MSE: `17.65`
- Missing-token max abs error: `7.41`
- Missing compact tokens: `474`
- Boundary: linear-tail is a deterministic non-oracle toy trend baseline over stored prefix tokens. The current artifact uses a nonlinear toy KV pattern to avoid the misleading near-zero error from `np.arange`; this should still not be read as model-quality trained/ridge recovery.

## Real HF KV Storage Smoke

The first real-tensor smoke now runs on Qwen3-0.6B HuggingFace `past_key_values` and converts them into the runtime prototype shape `[2, layers, blocks, block_size, kv_heads, head_dim]`.

- Script: `experiments/light_doc_cache/run_real_kv_storage_smoke.py`
- Output: `experiments/light_doc_cache/real_kv_storage_smoke_qwen3_0_6b_20260710/`
- Remote Python that worked: `/data00/home/sitian/miniconda3/envs/py311/bin/python`

Latest remote artifact:

- Input tokens: `12`
- KV cache shape: `[2, 28, 1, 16, 8, 128]`
- Recovery mode: `linear_tail`
- Full tensor bytes: `1,835,008`
- Stored tensor bytes: `1,133,568`
- Saved tensor bytes: `701,440`
- Byte saving fraction: `38.23%`
- Missing compact tokens: `474`
- Missing-token MSE: `23.19`
- Missing-token max abs error: `302`

Boundary: this uses real model KV tensors, but it is still an offline HF `past_key_values` storage/recovery smoke. It is not wired into TinyLLM `ModelRunner`, attention kernels, slot mapping, or decode hot path, and it does not prove task quality.

## TinyLLM KV Summary Smoke

The current runtime-facing smoke now instantiates TinyLLM `LLM` / `ModelRunner`, runs a short prompt, and reads the actual allocated `model_runner.kv_cache` shape for Light Doc Cache accounting.

- Script: `experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py`
- Output: `experiments/light_doc_cache/tinyllm_kv_summary_smoke_qwen3_0_6b_20260710/`

Latest remote artifact:

- Model: Qwen3-0.6B
- Prompt tokens: `10`
- TinyLLM allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Allocated KV cache bytes: `23,634,903,040`
- Logical full KV bytes for the 10-token plan: `1,146,880`
- Planned recovered KV bytes: `202,240`
- Planned stored KV bytes: `944,640`
- Planned byte saving fraction: `17.63%`
- Planned compression ratio: `1.2141x`

Boundary: this is an allocated-cache/accounting smoke only. It reads TinyLLM's real `ModelRunner.kv_cache` allocation, but it does not yet store compact heads in a sidecar, change KV writes, change attention reads, or measure runtime memory reduction.

## TinyLLM Sidecar Storage Smoke

The next default-off prototype materializes a compressed sidecar from the actual TinyLLM `ModelRunner.kv_cache` after a short run, then restores into a temporary full-shape tensor for missing-token error evaluation.

- Script: `experiments/light_doc_cache/run_tinyllm_kv_summary_smoke.py --write-sidecar-storage`
- Output: `experiments/light_doc_cache/tinyllm_sidecar_storage_smoke_qwen3_0_6b_20260710/`

Latest remote artifact:

- Prompt tokens: `11`
- TinyLLM allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Recovery mode: `linear_tail`
- Sidecar full tensor bytes over allocated capacity: `23,634,903,040`
- Sidecar stored tensor bytes: `1,059,328`
- Allocated-capacity saving fraction: `99.9955%`
- Logical full KV bytes for the 11-token plan: `1,261,568`
- Logical stored KV bytes: `1,059,328`
- Logical byte saving fraction: `16.03%`
- Missing compact tokens: `395`
- Missing-token MSE: `28.80`
- Missing-token max abs error: `274`

Boundary: the allocated-capacity saving fraction compares sidecar storage against the whole preallocated TinyLLM KV cache and is not a runtime memory reduction claim. The logical byte saving fraction is the comparable prompt-level compression number. Attention still reads the original full `kv_cache`; no hot-path KV write/read integration has been applied.

The smoke script now routes sidecar materialization through `ModelRunner.light_doc_cache_materialize_sidecar(...)` when present, so the remote smoke covers the default-off runner integration point while still leaving the attention read path unchanged.

Wrapper smoke artifact:

- Output: `experiments/light_doc_cache/tinyllm_sidecar_storage_wrapper_smoke_qwen3_0_6b_20260710/`
- Logical full KV bytes: `1,261,568`
- Logical stored KV bytes: `1,059,328`
- Logical byte saving fraction: `16.03%`
- Missing-token MSE: `28.4588`
- Missing-token max abs error: `274`

## TinyLLM Restored Sidecar Read-Path Smoke

The next default-off smoke validates that a restored sidecar buffer can be
consumed by the existing TinyLLM decode read path without modifying attention
kernels:

- Script: `experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py`
- Output: `experiments/light_doc_cache/tinyllm_sidecar_read_path_smoke_qwen3_0_6b_20260710/`

The script runs normal prefill, materializes the sidecar, restores it into a
temporary full KV tensor, temporarily points each attention layer's
`k_cache`/`v_cache` at that restored tensor for one decode step, then restores
the original cache pointers.

Latest remote artifact:

- Prompt tokens: `13`
- TinyLLM allocated KV shape: `[2, 28, 805, 256, 8, 128]`
- Logical full KV bytes: `1,490,944`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing compact tokens: `553`
- Missing-token MSE: `38.5860`
- Max abs logit diff: `17.5`
- Mean abs logit diff: `3.1362`
- Argmax match: `False`

Boundary: this proves the restored sidecar tensor is shape-compatible with the
existing decode read path. It also shows the current toy `linear_tail` recovery
is not quality-preserving for logits on this prompt, so it should not be used
for an accuracy or runtime-compression claim. The next quality step should use
oracle read-path as a layout upper bound and then a trained/correlation recovery
module, before any memory-lifetime changes.

Oracle upper-bound read-path artifact:

- Output: `experiments/light_doc_cache/tinyllm_sidecar_read_path_oracle_smoke_qwen3_0_6b_20260710/`
- Logical full KV bytes: `1,490,944`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing compact tokens: `553`
- Missing-token MSE: `0`
- Max abs logit diff: `0`
- Mean abs logit diff: `0`
- Argmax match: `True`

Interpretation: oracle recovery proves the sidecar restore layout and temporary
read-path pointer swap can exactly reproduce the original decode logits. The
remaining gap is recovery quality, not cache indexing/layout.

Correlation-recovery read-path artifact:

- Output: `experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_smoke_qwen3_0_6b_20260710/`
- Recovery mode: `correlated`
- Source-head strategy: `same_layer`; each compact head uses a retained full
  head from the same layer when available, otherwise the first retained full
  head.
- Logical full KV bytes: `1,490,944`
- Logical stored KV bytes: `1,207,808`
- Logical byte saving fraction: `18.99%`
- Missing compact tokens: `553`
- Missing-token MSE: `11.5488`
- Missing-token max abs error: `174`
- Max abs logit diff: `5.21875`
- Mean abs logit diff: `0.886142`
- Argmax match: `False`

Interpretation: the first non-oracle correlated-head callback improves the
missing-token error and logit-diff scale versus the toy `linear_tail` baseline,
but it still changes the decode argmax. This is a useful runtime-compatible
recovery callback and a better non-oracle baseline, not yet a quality-preserving
trained recovery module. The next quality step should fit better source-head
selection and multi-head/multi-layer recovery weights from prompt/model data
before changing attention hot paths or KV allocation lifetime.

Prefix-fit source selection is also available for controlled comparison via
`--correlated-source-map prefix_fit`. It selects retained source heads by
lowest affine prefix reconstruction error, using only stored target prefixes.
On the same Qwen3-0.6B read-path smoke it was worse than the simpler same-layer
heuristic:

- Output: `experiments/light_doc_cache/tinyllm_sidecar_read_path_correlated_prefixfit_smoke_qwen3_0_6b_20260710/`
- Missing-token MSE: `13.2058`
- Max abs logit diff: `6.9375`
- Mean abs logit diff: `1.10957`
- Argmax match: `False`

Interpretation: per-head prefix MSE on a very short prompt is not a reliable
proxy for downstream decode logits. Keep `same_layer` as the default
non-oracle correlated baseline and treat `prefix_fit` as an ablation, not an
upgrade.

Recovery-mode matrix artifact:

- Generator: `experiments/light_doc_cache/make_read_path_recovery_matrix.py`
- Output: `experiments/light_doc_cache/read_path_recovery_matrix_qwen3_0_6b_20260710/`
- CSV: `read_path_recovery_matrix.csv`
- Markdown: `read_path_recovery_matrix.md`

Same-prompt Qwen3-0.6B read-path comparison:

| Mode | Missing MSE | Max Logit Diff | Mean Logit Diff | Argmax Match |
|---|---:|---:|---:|---|
| `repeat_last` | 13.7399 | 5.5625 | 0.787285 | False |
| `linear_tail` | 38.5860 | 17.5 | 3.13622 | False |
| `correlated_same_layer` | 11.5488 | 5.21875 | 0.886142 | False |
| `correlated_prefix_fit` | 13.2058 | 6.9375 | 1.10957 | False |
| `multi_correlated2` | 18.3975 | 6.96875 | 1.04723 | False |
| `oracle` | 0 | 0 | 0 | True |

Matrix interpretation: oracle remains exact, so layout/read-path are correct.
No non-oracle mode preserves argmax on this prompt. `repeat_last` has the best
mean logit diff, while `correlated_same_layer` has the best missing-token MSE
and max logit diff among non-oracle modes. `multi_correlated2` is mechanically
validated but worse than the single-source same-layer baseline here, so the next
useful recovery work should use trained coefficients from a larger calibration
set or decode-logit-aware selection rather than more per-prompt short-prefix
least squares.

Calibrated recovery-bank API:

- `fit_multi_source_recovery_bank(calibration_kv, plan, source_heads=..., ridge=...)`
- `make_calibrated_multi_source_recovery_callback(storage, bank)`
- `save_multi_source_recovery_bank(bank, path)`
- `load_multi_source_recovery_bank(path)`
- Read-path mode: `--recover-mode calibrated_multi_correlated --recovery-bank-file <bank.json>`

This is the first offline-calibrated recovery entry point. Unit tests verify
that weights fitted on one calibration KV tensor can be applied to a different
runtime KV tensor when the source-target relation is stable. The API is not yet
wired to a real calibration dataset or a remote Qwen smoke; it is the next
place to connect prompt/model calibration artifacts before claiming trained
recovery quality.

TinyLLM calibrated KV smoke:

- Script: `experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py`
- Output: `experiments/light_doc_cache/tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/`
- Bank file: `multi_source_recovery_bank.json`
- Calibration tokens: `14`
- Target tokens: `14`
- Source count: `2`
- Missing-token MSE: `13.5483`
- Missing-token max abs error: `221`
- Stored tensor bytes: `1,322,496`

Interpretation: this is the first real TinyLLM `ModelRunner.kv_cache`
calibrated-bank artifact. It validates bank fitting, JSON persistence, and
runtime application on real TinyLLM KV tensors. It is KV-error-only, not a
decode-logit read-path result. On this prompt pair it is slightly better than
`repeat_last` in missing-token MSE (`13.7399`) but worse than the same-layer
single-source correlated read-path baseline (`11.5488`, measured on the earlier
same-prompt read-path setup), so it should be treated as plumbing plus initial
calibration evidence rather than a final trained recovery gain.

## Current Tradeoff Conclusion

See `task_quality_v2_tradeoff.md` for the combined unconstrained and constrained policy sweep.

- Unconstrained thresholds `0.35`, `0.40`, and `0.45` fail the 9-task v2 quality smoke.
- Unconstrained threshold `0.50` passes quality but saves only about `4.30%` KV head-token entries.
- A constrained per-layer policy improves the v2 safe smoke point to about `6.83%` entry saving at 17 compressed heads.
- Expanded v3 stress shows that v2 was too weak: the same 17-head policy fails 13 baseline-stable questions, especially non-A-answer tasks.
- Pair-drop ablation found a task-aware 15-head policy that passes both v3 stable and v2 stable on `kv-sparse-attention.md`, but it fails the second-document `qwen3-8b-fixes.md` stable set.
- The current evidence does not justify runtime integration. This older task-aware policy sweep found only weak cross-document safe compression; the newer recovery-bank head-level search below improves the recovery-policy safe frontier to 58 heads / 12.95% total entry saving, but still remains far from global 2x+ doc-cache compression.

See also `task_quality_v2_layer_policy_compare.md` for the all-layer vs late-layer comparison. Late-layer `0.35` improves quality but still fails; late-layer `0.50` passes with only about `1.93%` entry saving.

## Recovery Sweep Update

Latest aggregate report: `recovery_sweep_summary_latest/report.md`.

Best current probe: `recovery_probe_2docs_l24_b256_fused_smoke_latest`.

- Layer start: 24
- Budget: 256 / 512 sampled tokens, approximately 2x token compression
- Heads: 16 across two documents in the smoke
- Mean direct val R²: 0.2999
- Mean FitV val R²: 0.1852
- Mean recovery val R²: 0.3085
- Mean recovery gain vs FitV: +0.1232
- Coverage above R² 0.5: 25.00%

The useful signal is that trained residual recovery starts to help at larger budgets, especially around layer 24 and 2x token compression. The limiting result is that absolute recovery R² and coverage remain too low for a runtime integration or an accuracy-lossless claim. Treat this as diagnostic evidence for the next design step: shared multi-doc/multi-head recovery or a task-level recovery adapter, not a production policy.

## Task-Level Recovery Smoke Update

A layer-24-only 2x token-budget policy was tested with `task_quality_smoke.py`:

- Policy: `policy_recovery_l24_b050`
- Compacted heads: 8 KV heads at layer 24
- Per-head selected budget: 50% of prompt tokens
- Total KV head-token entry saving: 1.79%

Results:

| Task Set | Bank Method | Local Result Dir | Compact Acc | Agreement | Mean Delta | Result |
|---|---|---|---:|---:|---:|---|
| `kv-sparse-attention` v3 stable | `learned_values` | `task_quality_smoke_recovery_l24_b050_v3_learned_values_latest` | 100.00% | 100.00% | +0.2624 | pass |
| `qwen3-8b-fixes` stable | `learned_values` | `task_quality_smoke_recovery_l24_b050_qwen8bfixes_learned_values_latest` | 100.00% | 100.00% | +0.6987 | pass |
| `kv-sparse-attention` v3 stable | `ridge` | `task_quality_smoke_recovery_l24_b050_v3_ridge_latest` | 100.00% | 100.00% | +0.2861 | pass |
| `qwen3-8b-fixes` stable | `ridge` | `task_quality_smoke_recovery_l24_b050_qwen8bfixes_ridge_latest` | 100.00% | 100.00% | +0.7324 | pass |

A true all-layer 2x task-level policy was also tested:

- Policy: `policy_recovery_all_b050`
- Compacted heads: all 224 KV heads
- Total KV head-token entry saving: 50.00%
- Result on `kv-sparse-attention` v3 stable: compact accuracy 15.38%, agreement 15.38%, mean answer logP delta -34.5383.

Conclusion: layer24 has a real task-level safe signal, but all-layer 2x compression fails badly. Current evidence supports targeted high-layer/head recovery search, not global 2x runtime integration.

## High-Layer Range Search Update

Layer-range policies were tested after the layer24-only policy passed task gates:

| Policy | Layers | Total Entry Saving | First-Doc v3 Compact Acc | Result |
|---|---:|---:|---:|---|
| `policy_recovery_l20_28_b050` | 20:28 | 14.29% | 23.08% | fail |
| `policy_recovery_l24_28_b050` | 24:28 | 7.14% | 46.15% | fail |

This confirms that layer24-only safety does not transfer to whole high-layer ranges. The next search should be head-level progressive expansion, not layer-range expansion.

## Head-List Policy Support

`make_recovery_task_policy.py` now supports exact head lists:

```bash
python3 experiments/light_doc_cache/make_recovery_task_policy.py \
  --compact-heads 24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7 \
  --budget-fraction 0.5 \
  --threshold 0.79 \
  --output-dir experiments/light_doc_cache/policy_recovery_heads_l24_all_b050
```

This enables the next search phase: progressive add-one/add-pair head expansion from the safe layer24 seed. Whole-layer expansion failed, so future searches should optimize at head granularity.

## Head-Level Progressive Search Update

Starting from the safe layer24 seed, add-one head search found a larger cross-document safe recovery-bank policy:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_add_one_from_l24_seed_top12/add_l10_h3` | 9 | `10:3` | 2.01% | 13/13 | 6/6 | safe |
| `policy_recovery_seed_l24_plus_l10h3_l8h1_b050` | 10 | `8:1` | 2.23% | 13/13 | 6/6 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_add_l20h2_b050` | 11 | `20:2` | 2.46% | 13/13 | 6/6 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_add_l4h6_b050` | 12 | `4:6` | 2.68% | 13/13 | 5/6 | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_add_l8h5_b050` | 12 | `8:5` | 2.68% | 13/13 | 5/6 | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_add_l12h4_b050` | 12 | `12:4` | 2.68% | 13/13 | 6/6 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_add_l10h5_b050` | 13 | `10:5` | 2.90% | 13/13 | 6/6 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_add_l12h7_b050` | 14 | `12:7` | 3.12% | 13/13 | 6/6 | safe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l4h5_b050` | 15 | `4:5` | 3.35% | 12/13 | not run | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l12h1_b050` | 15 | `12:1` | 3.35% | 12/13 | not run | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l16h6_b050` | 15 | `16:6` | 3.35% | 11/13 | not run | unsafe |
| `policy_recovery_seed_l24_l10h3_l8h1_l20h2_l12h4_l10h5_l12h7_add_l12h3_b050` | 15 | `12:3` | 3.35% | 12/13 | not run | unsafe |
| `policy_recovery_seed14_add_l20h7_b050` | 15 | `20:7` | 3.35% | 13/13 | 6/6 | safe |
| `policy_recovery_seed15_l20h7_add_l8h3_b050` | 16 | `8:3` | 3.57% | 13/13 | 6/6 | safe |
| `policy_recovery_seed16_add_l12h5_b050` | 17 | `12:5` | 3.79% | 13/13 | 6/6 | safe |
| `policy_recovery_seed17_add_l12h6_b050` | 18 | `12:6` | 4.02% | 12/13 | not run | unsafe |
| `policy_recovery_seed17_add_l16h5_b050` | 18 | `16:5` | 4.02% | 13/13 | 6/6 | safe |
| `policy_recovery_seed18_add_l16h4_b050` | 19 | `16:4` | 4.24% | 12/13 | not run | unsafe |
| `policy_recovery_seed18_add_l8h0_b050` | 19 | `8:0` | 4.24% | 13/13 | 6/6 | safe |
| `policy_recovery_seed19_add_l20h6_b050` | 20 | `20:6` | 4.46% | 13/13 | 6/6 | safe |
| `policy_recovery_seed20_add_l10h2_b050` | 21 | `10:2` | 4.69% | 13/13 | 6/6 | safe |
| `policy_recovery_seed21_add_l12h0_b050` | 22 | `12:0` | 4.91% | 13/13 | 6/6 | safe |
| `policy_recovery_seed22_add_l20h5_b050` | 23 | `20:5` | 5.13% | 12/13 | not run | unsafe |
| `policy_recovery_seed22_add_l20h1_b050` | 23 | `20:1` | 5.13% | 13/13 | 6/6 | safe |
| `policy_recovery_seed23_add_l10h1_b050` | 24 | `10:1` | 5.36% | 13/13 | 6/6 | safe |
| `policy_recovery_seed24_add_l16h7_b050` | 25 | `16:7` | 5.58% | 11/13 | not run | unsafe |
| `policy_recovery_seed24_add_l8h2_b050` | 25 | `8:2` | 5.58% | 12/13 | not run | unsafe |
| `policy_recovery_seed24_add_l16h3_b050` | 25 | `16:3` | 5.58% | 12/13 | not run | unsafe |
| `policy_recovery_seed24_add_l20h3_b050` | 25 | `20:3` | 5.58% | 13/13 | 6/6 | safe |
| `policy_recovery_seed25_add_l0h0_b050` | 26 | `0:0` | 5.80% | 13/13 | 6/6 | safe |
| `policy_recovery_seed26_add_l20h4_b050` | 27 | `20:4` | 6.03% | 12/13 | not run | unsafe |
| `policy_recovery_seed26_add_l8h6_b050` | 27 | `8:6` | 6.03% | 12/13 | not run | unsafe |
| `policy_recovery_seed26_add_l16h1_b050` | 27 | `16:1` | 6.03% | 12/13 | not run | unsafe |

Current best cross-document safe frontier:

- Head list: `24:0..7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0`
- Per-head selected budget: 50%
- Total entry saving: 5.80%
- Task gates: first-doc v3 stable 13/13, second-doc stable 6/6

Failure pattern after the 14-head frontier:

- `4:6` and `8:5` passed the first doc but failed the second-doc `tp_true_weight_split` task.
- `4:5`, `12:1`, and `12:3` failed first-doc `topk8_quality`.
- `16:6` failed first-doc `route_phase3` and `topk8_quality`.
- Expanding beyond top12 found more safe heads: `20:7`, `8:3`, `12:5`, `16:5`, `8:0`, `20:6`, `10:2`, `12:0`, `20:1`, `10:1`, `20:3`, and `0:0`.
- `12:6` and `16:4` failed first-doc `route_phase3`, so the current tested frontier is 18 safe heads rather than a monotonic top-R² prefix.
- `20:5` failed first-doc `quest_decode_selection`.
- `16:7` failed first-doc `prefix_cache_bug` and `topk8_quality`; `8:2` and `16:3` failed first-doc `quest_decode_selection`.
- `20:4` failed first-doc `topk8_quality`.
- `8:6` and `16:1` failed first-doc `quest_decode_selection`.

This confirms that task-safe compression is head-specific and task-sensitive. Offline R² can propose candidates, but every added head needs task-level gating and unsafe heads should be skipped rather than treated as a strict prefix. The best current recovery-bank policy improves over layer24-only, but it is still only a 5.80% total KV head-token entry saving and does not support a global 2x+ compression claim.

## Extended Head-Level Search Update

The progressive add-one search was continued from the previous 26-head safe frontier using the same recovery bank (`ridge`) and the same two task gates:

- first doc: `docs/kv-sparse-attention.md`, `task_quality_tasks_kv_sparse_v3_stable.json`
- second doc: `docs/qwen3-8b-fixes.md`, `task_quality_tasks_qwen3_8b_fixes_stable.json`
- policy: exact compact head list, per-head selected budget 50%

New results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed26_add_l8h7_b050` | 27 | `8:7` | 6.03% | 13/13, delta +0.3193 | 6/6, delta +1.8511 | safe |
| `policy_recovery_seed27_add_l4h1_b050` | 28 | `4:1` | 6.25% | 13/13, delta +0.3922 | 6/6, delta +1.8479 | safe |
| `policy_recovery_seed28_add_l0h2_b050` | 29 | `0:2` | 6.47% | 13/13, delta +0.2938 | 6/6, delta +1.7941 | safe |
| `policy_recovery_seed29_add_l0h5_b050` | 30 | `0:5` | 6.70% | 13/13, delta +0.2240 | 6/6, delta +1.5503 | safe |
| `policy_recovery_seed30_add_l0h4_b050` | 31 | `0:4` | 6.92% | 13/13, delta +0.1433 | 6/6, delta +1.4856 | safe |
| `policy_recovery_seed31_add_l0h6_b050` | 32 | `0:6` | 7.14% | 12/13, delta +0.0611 | not run | unsafe |
| `policy_recovery_seed31_add_l16h0_b050` | 32 | `16:0` | 7.14% | 12/13, delta +0.2646 | not run | unsafe |
| `policy_recovery_seed31_add_l10h7_b050` | 32 | `10:7` | 7.14% | 13/13, delta +0.0428 | 6/6, delta +1.1412 | safe |
| `policy_recovery_seed32_add_l12h2_b050` | 33 | `12:2` | 7.37% | 12/13, delta +0.4341 | not run | unsafe |
| `policy_recovery_seed32_add_l10h6_b050` | 33 | `10:6` | 7.37% | 13/13, delta +0.4107 | 6/6, delta +1.6898 | safe |
| `policy_recovery_seed33_add_l0h7_b050` | 34 | `0:7` | 7.59% | 13/13, delta +0.4742 | 6/6, delta +1.6563 | safe |
| `policy_recovery_seed34_add_l4h2_b050` | 35 | `4:2` | 7.81% | 13/13, delta +0.4964 | 6/6, delta +1.6008 | safe |
| `policy_recovery_seed35_add_l4h7_b050` | 36 | `4:7` | 8.04% | 13/13, delta +0.4686 | 6/6, delta +1.5820 | safe |
| `policy_recovery_seed36_add_l10h0_b050` | 37 | `10:0` | 8.26% | 13/13, delta +0.1315 | 6/6, delta +1.3163 | safe |
| `policy_recovery_seed37_add_l10h4_b050` | 38 | `10:4` | 8.48% | 12/13, delta +0.1232 | not run | unsafe |
| `policy_recovery_seed37_add_l4h4_b050` | 38 | `4:4` | 8.48% | 13/13, delta +0.1853 | 6/6, delta +1.3584 | safe |
| `policy_recovery_seed38_add_l0h3_b050` | 39 | `0:3` | 8.71% | 13/13, delta +0.0922 | 6/6, delta +1.4533 | safe |
| `policy_recovery_seed39_add_l16h2_b050` | 40 | `16:2` | 8.93% | 13/13, delta +0.2084 | 6/6, delta +1.4016 | safe |
| `policy_recovery_seed40_add_l8h4_b050` | 41 | `8:4` | 9.15% | 13/13, delta +0.2154 | 6/6, delta +1.2424 | safe |
| `policy_recovery_seed41_add_l4h0_b050` | 42 | `4:0` | 9.38% | 13/13, delta +0.3445 | 6/6, delta +1.3466 | safe |
| `policy_recovery_seed42_add_l4h3_b050` | 43 | `4:3` | 9.60% | 13/13, delta +0.2704 | 6/6, delta +1.6126 | safe |
| `policy_recovery_seed43_add_l0h1_b050` | 44 | `0:1` | 9.82% | 13/13, delta +0.3443 | 6/6, delta +1.6561 | safe |
| `policy_recovery_seed44_add_l20h0_b050` | 45 | `20:0` | 10.04% | 12/13, delta +1.1228 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `policy_recovery_seed43_add_l0h1_b050`
- Heads: 44
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1`
- Per-head selected budget: 50%
- Total KV head-token entry saving: 9.82%
- First-doc v3 stable: 13/13 pass, mean delta +0.3443
- Second-doc stable: 6/6 pass, mean delta +1.6561

New failure pattern:

- `0:6` failed first-doc `quest_decode_selection` (`A -> C`).
- `16:0` failed first-doc `route_phase3` (`A -> B`).
- `12:2`, `10:4`, and `20:0` failed first-doc `prefix_cache_bug`.

This expands the safe task-gated recovery-bank frontier from 26 heads / 5.80% total entry saving to 44 heads / 9.82% total entry saving. It still does not support a 2x+ global doc-cache compression claim: the gains are real but remain targeted to a sparse head set under an offline quality simulation, not an integrated runtime K/V cache recovery path.

## Round2 Full-Layer Head Search Update

After completing recovery probes for all layers, the add-one search was resumed from the 44-head safe frontier. The ordering still uses offline recovery metrics only as a candidate proposal mechanism; every accepted head must pass both task-quality gates with the ridge recovery bank.

Round2 logs:

- `head_progress_round2_20260709_192603.tsv`
- `head_progress_round2_from52_20260709_194018.tsv`
- `head_progress_round2_retry23h7_20260709_195107.tsv`

New round2 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed44_add_l6h7_b050` | 45 | `6:7` | 10.04% | 13/13, delta +0.2832 | 6/6, delta +1.7664 | safe |
| `policy_recovery_seed45_add_l3h0_b050` | 46 | `3:0` | 10.27% | 13/13, delta +0.3215 | 6/6, delta +1.7573 | safe |
| `policy_recovery_seed46_add_l26h5_b050` | 47 | `26:5` | 10.49% | 13/13, delta +0.3215 | 6/6, delta +1.8209 | safe |
| `policy_recovery_seed47_add_l18h0_b050` | 48 | `18:0` | 10.71% | 13/13, delta +0.4821 | 6/6, delta +1.7973 | safe |
| `policy_recovery_seed48_add_l25h6_b050` | 49 | `25:6` | 10.94% | 12/13, delta +0.1111 | not run | unsafe |
| `policy_recovery_seed48_add_l22h1_b050` | 49 | `22:1` | 10.94% | 13/13, delta +0.5319 | 6/6, delta +1.9941 | safe |
| `policy_recovery_seed49_add_l26h2_b050` | 50 | `26:2` | 11.16% | 13/13, delta +0.3615 | 5/6, delta +1.7897 | unsafe |
| `policy_recovery_seed49_add_l11h2_b050` | 50 | `11:2` | 11.16% | 13/13, delta +0.6035 | 6/6, delta +1.7880 | safe |
| `policy_recovery_seed50_add_l25h5_b050` | 51 | `25:5` | 11.38% | 12/13, delta +0.2425 | not run | unsafe |
| `policy_recovery_seed50_add_l22h0_b050` | 51 | `22:0` | 11.38% | 13/13, delta +0.4683 | 6/6, delta +1.8989 | safe |
| `policy_recovery_seed51_add_l9h1_b050` | 52 | `9:1` | 11.61% | 13/13, delta +0.6004 | 5/6, delta +1.4883 | unsafe |
| `policy_recovery_seed51_add_l3h6_b050` | 52 | `3:6` | 11.61% | 13/13, delta +0.4419 | 6/6, delta +1.9612 | safe |
| `policy_recovery_seed52_add_l26h3_b050` | 53 | `26:3` | 11.83% | 12/13, delta +0.7619 | not run | unsafe |
| `policy_recovery_seed52_add_l26h1_b050` | 53 | `26:1` | 11.83% | 10/13, delta +0.3498 | not run | unsafe |
| `policy_recovery_seed52_add_l26h0_b050` | 53 | `26:0` | 11.83% | 12/13, delta +0.1350 | not run | unsafe |
| `policy_recovery_seed52_add_l13h1_b050` | 53 | `13:1` | 11.83% | 13/13, delta +0.6363 | 6/6, delta +1.9674 | safe |
| `policy_recovery_seed53_add_l18h1_b050` | 54 | `18:1` | 12.05% | 11/13, delta -1.4445 | not run | unsafe |
| `policy_recovery_seed53_add_l21h5_b050` | 54 | `21:5` | 12.05% | 12/13, delta +0.9535 | not run | unsafe |
| `policy_recovery_seed53_add_l3h1_b050` | 54 | `3:1` | 12.05% | 13/13, delta +0.6570 | 6/6, delta +1.9631 | safe |
| `policy_recovery_seed54_add_l23h2_b050` | 55 | `23:2` | 12.28% | 12/13, delta +0.6697 | not run | unsafe |
| `policy_recovery_seed54_add_l25h2_b050` | 55 | `25:2` | 12.28% | 12/13, delta +0.1035 | not run | unsafe |
| `policy_recovery_seed54_add_l13h3_b050` | 55 | `13:3` | 12.28% | 13/13, delta +0.7032 | 6/6, delta +2.3966 | safe |
| `policy_recovery_seed55_add_l23h0_b050` | 56 | `23:0` | 12.50% | 12/13, delta +0.5160 | not run | unsafe |
| `policy_recovery_seed55_add_l23h7_b050` | 56 | `23:7` | 12.50% | 13/13, delta +0.5845 | 6/6, delta +2.3314 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed55_add_l23h7_b050`
- Heads: 56
- Per-head selected budget: 50%
- Total KV head-token entry saving: 12.50%
- First-doc v3 stable: 13/13 pass, mean delta +0.5845
- Second-doc stable: 6/6 pass, mean delta +2.3314
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7`

Round2 failure pattern:

- First-doc failures: `25:6`, `25:5`, `26:3`, `26:1`, `26:0`, `18:1`, `21:5`, `23:2`, `25:2`, `23:0`.
- Second-doc failures after first-doc pass: `26:2` and `9:1`, both failing `tp_true_weight_split`.
- A transient SSH/base64 transfer failure for `23:7` was retried and resolved; the retried quality result is safe.

Conclusion: full-layer recovery ranking plus strict task gates expanded the frontier from 44 heads / 9.82% to 56 heads / 12.50% total KV head-token entry saving. This is a stronger targeted recovery-bank result, but it is still not global 2x doc-cache compression: it remains a fixed sparse head-list policy under an offline quality simulation. A 2x+ claim still needs document-adaptive or shared trained recovery and runtime K/V cache integration.

## Round3 Partial Head Search Update

The next add-one batch started from the 56-head frontier and used all-layer cross-doc `recovery_val_r2` with known safe/unsafe heads excluded.

Round3 log:

- `head_progress_round3_20260709_195436.tsv`

Completed quality results before Kerberos expiry:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed56_add_l11h0_b050` | 57 | `11:0` | 12.72% | 13/13, delta +0.6066 | 6/6, delta +2.5306 | safe |
| `policy_recovery_seed57_add_l11h6_b050` | 58 | `11:6` | 12.95% | 11/13, delta +0.3431 | not run | unsafe |
| `policy_recovery_seed57_add_l13h2_b050` | 58 | `13:2` | 12.95% | 12/13, delta +0.3330 | not run | unsafe |
| `policy_recovery_seed57_add_l13h5_b050` | 58 | `13:5` | 12.95% | 13/13, delta +0.5693 | 6/6, delta +2.3385 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed57_add_l13h5_b050`
- Heads: 58
- Per-head selected budget: 50%
- Total KV head-token entry saving: 12.95%
- First-doc v3 stable: 13/13 pass, mean delta +0.5693
- Second-doc stable: 6/6 pass, mean delta +2.3385
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5`

Blocked state: later round3 rows are SSH/Kerberos command failures, not quality failures. `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist` shows the TGT expired, and `kinit -R` could not renew it. After Kerberos is refreshed, retry unresolved candidates from the 58-head seed: `2:5,17:1,17:4,14:5,3:7,2:7,17:5,14:0,26:7,21:6,14:1,19:6,1:5,11:5,1:7,13:0,9:6,5:2,13:7,25:3`.

Boundary: the latest frontier is now 58 heads / 12.95% total entry saving under strict two-document gates. This is still a sparse fixed-head offline quality result, not a global 2x+ doc-cache compression result.

## Round3 Retry Update to 68 Heads

After Kerberos was refreshed, the unresolved round3 candidates were rerun from the 58-head seed with the same strict two-document gate:

- Bank method: `ridge`
- Per-head selected budget: 50%
- First gate: `docs/kv-sparse-attention.md` + `task_quality_tasks_kv_sparse_v3_stable.json`
- Second gate: `docs/qwen3-8b-fixes.md` + `task_quality_tasks_qwen3_8b_fixes_stable.json`
- Authoritative retry logs:
  - `head_progress_round3_retry_from58_20260709_213131.tsv`
  - `head_progress_round3_retry_remaining_direct_20260709_215135.tsv`

Important log caveat: `head_progress_round3_retry_from58_20260709_213131.tsv` has stale/command-failure rows for the tail candidates after SSH instability. The direct rerun log `head_progress_round3_retry_remaining_direct_20260709_215135.tsv` is the source of truth for `13:0,9:6,5:2,13:7,25:3`.

New retry results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed58_add_l2h5_b050` | 59 | `2:5` | 13.17% | 12/13, delta +0.6885 | not run | unsafe |
| `policy_recovery_seed58_add_l17h1_b050` | 59 | `17:1` | 13.17% | 13/13, delta -0.2055 | 6/6, delta +1.9262 | safe |
| `policy_recovery_seed59_add_l17h4_b050` | 60 | `17:4` | 13.39% | 11/13, delta -0.1008 | not run | unsafe |
| `policy_recovery_seed59_add_l14h5_b050` | 60 | `14:5` | 13.39% | 12/13, delta -0.5952 | not run | unsafe |
| `policy_recovery_seed59_add_l3h7_b050` | 60 | `3:7` | 13.39% | 13/13, delta -0.0911 | 6/6, delta +1.9033 | safe |
| `policy_recovery_seed60_add_l2h7_b050` | 61 | `2:7` | 13.62% | 13/13, delta -0.1743 | 6/6, delta +1.8585 | safe |
| `policy_recovery_seed61_add_l17h5_b050` | 62 | `17:5` | 13.84% | 13/13, delta +0.4271 | 6/6, delta +2.0074 | safe |
| `policy_recovery_seed62_add_l14h0_b050` | 63 | `14:0` | 14.06% | 12/13, delta +0.4874 | not run | unsafe |
| `policy_recovery_seed62_add_l26h7_b050` | 63 | `26:7` | 14.06% | 12/13, delta +0.3042 | not run | unsafe |
| `policy_recovery_seed62_add_l21h6_b050` | 63 | `21:6` | 14.06% | 12/13, delta +0.4442 | not run | unsafe |
| `policy_recovery_seed62_add_l14h1_b050` | 63 | `14:1` | 14.06% | 13/13, delta +0.1545 | 6/6, delta +2.0307 | safe |
| `policy_recovery_seed63_add_l19h6_b050` | 64 | `19:6` | 14.29% | 12/13, delta -0.2735 | not run | unsafe |
| `policy_recovery_seed63_add_l1h5_b050` | 64 | `1:5` | 14.29% | 13/13, delta +0.1935 | 6/6, delta +1.8957 | safe |
| `policy_recovery_seed64_add_l11h5_b050` | 65 | `11:5` | 14.51% | 13/13, delta -0.0415 | 5/6, delta +1.9840 | unsafe |
| `policy_recovery_seed64_add_l1h7_b050` | 65 | `1:7` | 14.51% | 13/13, delta +0.2274 | 6/6, delta +1.8394 | safe |
| `policy_recovery_seed65_add_l13h0_b050` | 66 | `13:0` | 14.73% | 13/13, delta +0.2724 | 6/6, delta +1.7213 | safe |
| `policy_recovery_seed66_add_l9h6_b050` | 67 | `9:6` | 14.96% | 13/13, delta +0.2535 | 6/6, delta +1.9532 | safe |
| `policy_recovery_seed67_add_l5h2_b050` | 68 | `5:2` | 15.18% | 13/13, delta +0.0320 | 6/6, delta +1.3612 | safe |
| `policy_recovery_seed68_add_l13h7_b050` | 69 | `13:7` | 15.40% | 13/13, delta +0.0354 | 5/6, delta +1.2103 | unsafe |
| `policy_recovery_seed68_add_l25h3_b050` | 69 | `25:3` | 15.40% | 12/13, delta -0.4433 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed67_add_l5h2_b050`
- Heads: 68
- Per-head selected budget: 50%
- Total KV head-token entry saving: 15.18%
- First-doc v3 stable: 13/13 pass, mean delta +0.0320
- Second-doc stable: 6/6 pass, mean delta +1.3612
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2`

Round3 retry failure pattern:

- First-doc failures: `2:5`, `17:4`, `14:5`, `14:0`, `26:7`, `21:6`, `19:6`, and `25:3`.
- Second-doc failures: `11:5` failed `gpu_utilization_semantics`; `13:7` failed `tp_true_weight_split`.
- Several SSH `Connection closed by UNKNOWN port 65535` events were retried successfully and should not be counted as quality failures.

Conclusion: the task-gated recovery-bank frontier improved from 58 heads / 12.95% to 68 heads / 15.18% total KV head-token entry saving. This is a stronger sparse-head recovery result, but still not a 2x+ global doc-cache compression result: the evidence remains an offline quality simulation with fixed head lists, not runtime K/V cache recovery or document-adaptive trained gating.

## Round4 Candidate Update to 72 Heads

The next candidate batch was generated from all-layer cross-doc recovery rows after excluding the 68 safe heads and known unsafe heads. Candidate ordering used mean/min `recovery_val_r2` with light downranking for layers that had repeated task-gate failures.

- Candidate ranking file: `head_candidates_round4_from68_20260709.tsv`
- Automation log: `head_progress_round4_from68_20260709_220108.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%
- Gates: same first-doc 13-task stable set and second-doc 6-task stable set.

Round4 top-12 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed68_add_l4h6_b050` | 69 | `4:6` | 15.40% | 13/13, delta +0.0482 | 6/6, delta +1.4524 | safe |
| `policy_recovery_seed69_add_l4h5_b050` | 70 | `4:5` | 15.62% | 13/13, delta +0.1080 | 5/6, delta +1.5376 | unsafe |
| `policy_recovery_seed69_add_l12h1_b050` | 70 | `12:1` | 15.62% | 11/13, delta +0.2706 | not run | unsafe |
| `policy_recovery_seed69_add_l12h3_b050` | 70 | `12:3` | 15.62% | 12/13, delta +0.1186 | not run | unsafe |
| `policy_recovery_seed69_add_l16h6_b050` | 70 | `16:6` | 15.62% | 13/13, delta -0.0055 | 6/6, delta +1.3167 | safe |
| `policy_recovery_seed70_add_l18h6_b050` | 71 | `18:6` | 15.85% | 13/13, delta -0.0919 | 6/6, delta +1.3758 | safe |
| `policy_recovery_seed71_add_l8h5_b050` | 72 | `8:5` | 16.07% | 12/13, delta +0.2750 | not run | unsafe |
| `policy_recovery_seed71_add_l25h7_b050` | 72 | `25:7` | 16.07% | 12/13, delta -0.6519 | not run | unsafe |
| `policy_recovery_seed71_add_l6h0_b050` | 72 | `6:0` | 16.07% | 13/13, delta -0.1569 | 6/6, delta +1.5364 | safe |
| `policy_recovery_seed72_add_l25h4_b050` | 73 | `25:4` | 16.29% | 12/13, delta -0.0955 | not run | unsafe |
| `policy_recovery_seed72_add_l9h4_b050` | 73 | `9:4` | 16.29% | 12/13, delta -0.2362 | not run | unsafe |
| `policy_recovery_seed72_add_l21h1_b050` | 73 | `21:1` | 16.29% | 11/13, delta -0.3408 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed71_add_l6h0_b050`
- Heads: 72
- Per-head selected budget: 50%
- Total KV head-token entry saving: 16.07%
- First-doc v3 stable: 13/13 pass, mean delta -0.1569
- Second-doc stable: 6/6 pass, mean delta +1.5364
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0`

Round4 pattern:

- Safe additions were sparse despite strong offline recovery scores: `4:6`, `16:6`, `18:6`, and `6:0`.
- `4:5` passed first-doc but failed second-doc `tp_true_weight_split`.
- Most other failures were first-doc flips on `route_phase3`, `topk8_quality`, `sweet_spot`, or `quest_decode_selection`.
- The frontier is now 72 heads / 16.07% total KV head-token entry saving. It remains an offline fixed-head quality simulation, not a runtime 2x+ compression claim.

## Round5 Candidate Update to 74 Heads

Round5 regenerated candidates from the 72-head safe frontier, excluding new round4 failures and using stronger penalties for layers that repeatedly failed first-doc task gates.

- Candidate ranking file: `head_candidates_round5_from72_20260709.tsv`
- Automation log: `head_progress_round5_from72_20260709_221720.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%

Round5 top-12 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed72_add_l7h2_b050` | 73 | `7:2` | 16.29% | 11/13, delta -0.2060 | not run | unsafe |
| `policy_recovery_seed72_add_l22h3_b050` | 73 | `22:3` | 16.29% | 11/13, delta -0.7073 | not run | unsafe |
| `policy_recovery_seed72_add_l19h0_b050` | 73 | `19:0` | 16.29% | 12/13, delta -0.0959 | not run | unsafe |
| `policy_recovery_seed72_add_l6h5_b050` | 73 | `6:5` | 16.29% | 12/13, delta +0.0092 | not run | unsafe |
| `policy_recovery_seed72_add_l17h6_b050` | 73 | `17:6` | 16.29% | 12/13, delta -0.0703 | not run | unsafe |
| `policy_recovery_seed72_add_l22h4_b050` | 73 | `22:4` | 16.29% | 11/13, delta -0.1766 | not run | unsafe |
| `policy_recovery_seed72_add_l3h5_b050` | 73 | `3:5` | 16.29% | 11/13, delta +0.0882 | not run | unsafe |
| `policy_recovery_seed72_add_l15h3_b050` | 73 | `15:3` | 16.29% | 13/13, delta -0.2607 | 6/6, delta +1.6232 | safe |
| `policy_recovery_seed73_add_l19h4_b050` | 74 | `19:4` | 16.52% | 12/13, delta +1.0387 | not run | unsafe |
| `policy_recovery_seed73_add_l21h0_b050` | 74 | `21:0` | 16.52% | 12/13, delta -0.2035 | not run | unsafe |
| `policy_recovery_seed73_add_l21h2_b050` | 74 | `21:2` | 16.52% | 13/13, delta -0.4267 | 5/6, delta +1.5138 | unsafe |
| `policy_recovery_seed73_add_l18h5_b050` | 74 | `18:5` | 16.52% | 13/13, delta +0.2090 | 6/6, delta +1.6877 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed73_add_l18h5_b050`
- Heads: 74
- Per-head selected budget: 50%
- Total KV head-token entry saving: 16.52%
- First-doc v3 stable: 13/13 pass, mean delta +0.2090
- Second-doc stable: 6/6 pass, mean delta +1.6877
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0,15:3,18:5`

Round5 pattern:

- New safe heads: `15:3` and `18:5`.
- `21:2` passed the first document but failed second-doc `smoothquant_status`.
- The frontier is now 74 heads / 16.52% total KV head-token entry saving. Margins still show that the fixed sparse-head frontier is narrowing, and the result remains an offline task-quality simulation rather than runtime/global compression.

## Round6 Candidate Update to 78 Heads

Round6 regenerated candidates from the 74-head frontier. The first run was interrupted by intermittent SSH/remote command errors after `23:3`; the remaining candidates were retried from the verified 77-head frontier. Treat `head_progress_round6_continue_from77_20260709_223824.tsv` as the authoritative continuation for `1:6`, `2:1`, and `9:7`.

- Candidate ranking file: `head_candidates_round6_from74_20260709.tsv`
- Initial automation log: `head_progress_round6_from74_20260709_223132.tsv`
- Authoritative continuation log: `head_progress_round6_continue_from77_20260709_223824.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%

Round6 top-8 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed74_add_l14h6_b050` | 75 | `14:6` | 16.74% | 13/13, delta -0.1076 | 4/6, delta +1.3523 | unsafe |
| `policy_recovery_seed74_add_l14h3_b050` | 75 | `14:3` | 16.74% | 12/13, delta -0.0764 | not run | unsafe |
| `policy_recovery_seed74_add_l2h3_b050` | 75 | `2:3` | 16.74% | 13/13, delta +0.2269 | 6/6, delta +1.6754 | safe |
| `policy_recovery_seed75_add_l1h4_b050` | 76 | `1:4` | 16.96% | 13/13, delta +0.5610 | 6/6, delta +1.8397 | safe |
| `policy_recovery_seed76_add_l23h3_b050` | 77 | `23:3` | 17.19% | 13/13, delta +0.4796 | 6/6, delta +1.6751 | safe |
| `policy_recovery_seed77_add_l1h6_b050` | 78 | `1:6` | 17.41% | 12/13, delta +0.3075 | not run | unsafe |
| `policy_recovery_seed77_add_l2h1_b050` | 78 | `2:1` | 17.41% | 12/13, delta +0.3075 | not run | unsafe |
| `policy_recovery_seed77_add_l9h7_b050` | 78 | `9:7` | 17.41% | 13/13, delta +0.5104 | 6/6, delta +1.4734 | safe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed77_add_l9h7_b050`
- Heads: 78
- Per-head selected budget: 50%
- Total KV head-token entry saving: 17.41%
- First-doc v3 stable: 13/13 pass, mean delta +0.5104
- Second-doc stable: 6/6 pass, mean delta +1.4734
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0,15:3,18:5,2:3,1:4,23:3,9:7`

Round6 pattern:

- New safe heads: `2:3`, `1:4`, `23:3`, and `9:7`.
- `14:6` passed the first document but failed the second document badly, showing that single-document acceptance is not reliable.
- `14:3`, `1:6`, and `2:1` failed the first document despite good offline recovery ranking.
- The sparse fixed-head frontier improved from 74 heads / 16.52% to 78 heads / 17.41% total KV head-token entry saving. This is useful incremental evidence for recovery-bank search, but still far from a 2x+ global doc-cache/runtime compression claim. The next useful step is likely document-adaptive/task-aware gating or runtime recovery integration rather than only extending fixed-head greedy search.

## Round7 Conservative Candidate Update to 79 Heads

Round7 used a more conservative re-ranking from the Round6 candidate table: exclude known safe/unsafe heads, penalize layers with accumulated task-gate failures, and prefer positive cross-document minimum recovery scores. This produced a new proposal table for the 78-head frontier.

- Candidate ranking file: `head_candidates_round7_from78_20260709.tsv`
- Automation log: `head_progress_round7_from78_20260709_224340.tsv`
- Bank method: `ridge`
- Per-head selected budget: 50%

Round7 top-8 results:

| Policy | Heads | Added Head | Total Entry Saving | First Doc v3 | Second Doc Stable | Result |
|---|---:|---|---:|---|---|---|
| `policy_recovery_seed78_add_l6h6_b050` | 79 | `6:6` | 17.63% | 12/13, delta +0.6312 | not run | unsafe |
| `policy_recovery_seed78_add_l5h5_b050` | 79 | `5:5` | 17.63% | 12/13, delta +0.4275 | not run | unsafe |
| `policy_recovery_seed78_add_l11h3_b050` | 79 | `11:3` | 17.63% | 13/13, delta +0.6782 | 6/6, delta +1.4649 | safe |
| `policy_recovery_seed79_add_l22h6_b050` | 80 | `22:6` | 17.86% | 11/13, delta +1.3102 | not run | unsafe |
| `policy_recovery_seed79_add_l15h5_b050` | 80 | `15:5` | 17.86% | 12/13, delta +1.0609 | not run | unsafe |
| `policy_recovery_seed79_add_l7h7_b050` | 80 | `7:7` | 17.86% | 12/13, delta +0.7118 | not run | unsafe |
| `policy_recovery_seed79_add_l6h4_b050` | 80 | `6:4` | 17.86% | 12/13, delta +0.6025 | not run | unsafe |
| `policy_recovery_seed79_add_l15h4_b050` | 80 | `15:4` | 17.86% | 11/13, delta +0.7605 | not run | unsafe |

Current best cross-document safe frontier:

- Policy: `experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050`
- Heads: 79
- Per-head selected budget: 50%
- Total KV head-token entry saving: 17.63%
- First-doc v3 stable: 13/13 pass, mean delta +0.6782
- Second-doc stable: 6/6 pass, mean delta +1.4649
- Head list: `24:0,24:1,24:2,24:3,24:4,24:5,24:6,24:7,10:3,8:1,20:2,12:4,10:5,12:7,20:7,8:3,12:5,16:5,8:0,20:6,10:2,12:0,20:1,10:1,20:3,0:0,8:7,4:1,0:2,0:5,0:4,10:7,10:6,0:7,4:2,4:7,10:0,4:4,0:3,16:2,8:4,4:0,4:3,0:1,6:7,3:0,26:5,18:0,22:1,11:2,22:0,3:6,13:1,3:1,13:3,23:7,11:0,13:5,17:1,3:7,2:7,17:5,14:1,1:5,1:7,13:0,9:6,5:2,4:6,16:6,18:6,6:0,15:3,18:5,2:3,1:4,23:3,9:7,11:3`

Round7 pattern:

- New safe head: `11:3`.
- Seven of eight conservative candidates still failed the first document, mostly by flipping `topk8_quality`, `sweet_spot`, `quest_decode_selection`, or `route_phase3`.
- The frontier improved only from 78 heads / 17.41% to 79 heads / 17.63% total entry saving. This confirms diminishing returns for fixed global head lists; future progress should pivot toward document-adaptive/task-aware gating or an integrated recovery module rather than more blind fixed-head expansion.

## Task-Aware Failure Diagnostics and Budget Rescue

After Round7, late-round failures were aggregated from local mirrored `task_rows.csv` files to identify task-level bottlenecks.

- Diagnostic report: `task_failure_diagnostics_round4_round7_20260709.md`
- Flip rows: `task_failure_diagnostics_round4_round7_20260709.tsv`
- Per-policy summary: `task_quality_summary_round4_round7_20260709.tsv`

Most frequent flipped tasks:

| Count | Doc | Task |
|---:|---|---|
| 14 | first | `topk8_quality` |
| 8 | first | `route_phase3` |
| 7 | first | `sweet_spot` |
| 6 | first | `quest_decode_selection` |
| 2 | second | `tp_true_weight_split` |
| 2 | second | `smoothquant_status` |

This confirms that the late frontier is constrained by a small set of fragile task decisions, not by SSH instability or random command failures.

A small task-aware budget rescue experiment then kept the 79-head safe policy fixed and tried to add four failing Round7 candidates with the new head at 75% selected budget while existing safe heads stayed at 50%.

- Progress log: `taskaware_budget75_from79_progress_20260709_225838.tsv`
- Policies:
  - `policy_taskaware_budget75_from79_add_l6h6`
  - `policy_taskaware_budget75_from79_add_l5h5`
  - `policy_taskaware_budget75_from79_add_l22h6`
  - `policy_taskaware_budget75_from79_add_l15h5`

Budget-rescue results:

| Policy | Added Head | Heads | Entry Saving | First Doc v3 | Result |
|---|---|---:|---:|---|---|
| `policy_taskaware_budget75_from79_add_l6h6` | `6:6` | 80 | 17.75% | 12/13, delta +0.8239 | unsafe |
| `policy_taskaware_budget75_from79_add_l5h5` | `5:5` | 80 | 17.75% | 12/13, delta +0.5985 | unsafe |
| `policy_taskaware_budget75_from79_add_l22h6` | `22:6` | 80 | 17.75% | 11/13, delta +1.3246 | unsafe |
| `policy_taskaware_budget75_from79_add_l15h5` | `15:5` | 80 | 17.75% | 12/13, delta +1.0727 | unsafe |

The closest rescue candidate, `15:5`, was also tested with `learned_values` instead of `ridge`:

- Output: `task_quality_smoke_taskaware_budget75_from79_add_l15h5_v3_learnedv_latest`
- First-doc v3 stable: 11/13, mean delta +1.0147
- Mean bank build time: 6.3772s versus 0.3760s for ridge on the same policy.

Conclusion: increasing the new head budget to 75% and switching to learned compact values did not rescue the late fixed-head frontier. The failure mode is more likely task/document conditional sensitivity than insufficient per-head selected tokens. This strengthens the case for a real document-adaptive/task-aware gate or runtime recovery integration instead of further global fixed-head expansion.

## Adaptive Policy v1: Task-Conditional Fallback

The next step implemented a quality-only adaptive policy hook in `task_quality_smoke.py`:

- New argument: `--adaptive-policy-file`
- Default behavior remains unchanged when no adaptive policy is supplied.
- The adaptive policy can remove selected compact heads for specific task IDs before bank construction.
- Reports now include `effective_entry_saving_fraction`, which is the average task-level entry saving after task-specific fallback.
- The remote runner now supports `LOCAL_ADAPTIVE_POLICY_FILE` and verifies the transferred `adaptive_policy.json`.
- `make_adaptive_task_policy.py` can generate adaptive policy specs from task-failure diagnostics.

Adaptive policy v1:

- Policy spec: `adaptive_policy_from79_add_l15h5_drop_on_fragile_v1.json`
- Auto-generated equivalent policy: `adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json`
- Default policy: `policy_taskaware_budget75_from79_add_l15h5`
- Default behavior: 80 compressed heads, adding `15:5` at 75% budget on top of the 79-head safe frontier.
- Fragile-task fallback: for `topk8_quality`, `route_phase3`, `sweet_spot`, and `quest_decode_selection`, drop `15:5` so those tasks use the 79-head safe frontier.

Remote results:

| Output | Tasks | Quality Gate | Mean Delta | Mean Bank Build | Avg Entry Saving |
|---|---:|---|---:|---:|---:|
| `task_quality_smoke_adaptive_from79_add_l15h5_v3_latest` | 13 | 13/13 accuracy and agreement | +0.9575 | 0.2921s | 17.71% |
| `task_quality_smoke_adaptive_from79_add_l15h5_qwen8bfixes_latest` | 6 | 6/6 accuracy and agreement | +1.7281 | 0.2999s | 17.75% |

Interpretation:

- Static `policy_taskaware_budget75_from79_add_l15h5` failed the first document at 12/13.
- Adaptive fallback recovered the first document to 13/13 while preserving 6/6 on the second document.
- Average effective saving is slightly above the 79-head static frontier because only fragile first-doc tasks fall back from 80 to 79 heads.
- This is a stronger quality-simulation result than the static 79-head frontier, but it is still not runtime compression. The correct claim is: task-adaptive recovery-bank simulation preserves the strict two-document gates with about 17.7% average KV head-token entry saving.

Auto-generation command:

```bash
python3 experiments/light_doc_cache/make_adaptive_task_policy.py \
  --failure-diagnostics experiments/light_doc_cache/task_failure_diagnostics_round4_round7_20260709.tsv \
  --default-policy-dir experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l15h5 \
  --base-safe-policy-dir experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050 \
  --drop-heads 15:5 \
  --top-tasks 4 \
  --doc first \
  --output experiments/light_doc_cache/adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json
```

## Adaptive Search: 80-Head Default Policy Candidates

After validating adaptive v1, a small automated search tried six 80-head default policies. Each policy starts from the 79-head safe frontier, adds one candidate head at 75% selected-token budget, and uses the same top-4 first-doc fragile-task fallback:

- Fragile fallback tasks: `topk8_quality`, `route_phase3`, `sweet_spot`, `quest_decode_selection`
- Fallback action: drop the newly added head for those first-doc tasks.
- Full result table: `adaptive_search_from79_top4_two_doc_results_20260709.tsv`
- First-doc log: `adaptive_search_from79_top4_first_20260709.log`
- Second-doc log: `adaptive_search_from79_top4_second_20260709.log`

Remote two-document results:

| Added Head | First Doc Gate | First Mean Delta | Second Doc Gate | Second Mean Delta | Avg Effective Saving | Result |
|---|---|---:|---|---:|---:|---|
| `6:6` | 13/13 | +0.7536 | 3/6 | +1.2965 | 17.71% / 17.75% | second-doc unsafe |
| `5:5` | 13/13 | +0.6721 | 5/6 | +1.7423 | 17.71% / 17.75% | second-doc unsafe |
| `22:6` | 13/13 | +1.2329 | 6/6 | +1.4539 | 17.71% / 17.75% | safe |
| `15:5` | 13/13 | +0.9575 | 6/6 | +1.7281 | 17.71% / 17.75% | safe |
| `7:7` | 13/13 | +0.7426 | 4/6 | +1.4461 | 17.71% / 17.75% | second-doc unsafe |
| `6:4` | 13/13 | +0.6108 | 4/6 | +1.2970 | 17.71% / 17.75% | second-doc unsafe |

New best adaptive frontier by two-document gate:

- Best quality margin: `adaptive_policy_from79_add_l15h5_auto_top4_first_v1.json`, because it passes both documents and has stronger second-doc mean delta (+1.7281).
- Best first-doc margin: `adaptive_policy_from79_add_l22h6_auto_top4_first_v1.json`, because it passes both documents and has stronger first-doc mean delta (+1.2329), but lower second-doc margin (+1.4539).
- Both safe policies have the same average effective saving in the current task mix: 17.71% on the first doc and 17.75% on the second doc.

Second-doc failures are concentrated in `tp_true_weight_split`, `gpu_utilization_semantics`, and `smoothquant_status`, which were not covered by the first-doc-only fragile fallback. The next useful search is therefore a two-doc adaptive generator that can add second-doc fallback rules, rather than increasing global compression.

Claim boundary: this remains an offline task-adaptive recovery-bank quality simulation. It supports a narrow claim of strict two-document quality preservation at about 17.7% average KV head-token entry saving; it does not support a 2x+ runtime doc-cache compression claim.

## Two-Document Adaptive Fallback Rescue

The first adaptive search used only first-doc fragile tasks for fallback, so four candidates still failed the second-doc gate. The generator now supports multiple failure sources and per-document top task selection:

```bash
python3 experiments/light_doc_cache/make_adaptive_task_policy.py \
  --failure-diagnostics experiments/light_doc_cache/task_failure_diagnostics_round4_round7_20260709.tsv \
  --failure-diagnostics experiments/light_doc_cache/adaptive_search_from79_top4_two_doc_results_20260709.tsv \
  --default-policy-dir experiments/light_doc_cache/policy_taskaware_budget75_from79_add_l6h6 \
  --base-safe-policy-dir experiments/light_doc_cache/policy_recovery_seed78_add_l11h3_b050 \
  --drop-heads 6:6 \
  --per-doc-top-tasks first=4,second=3 \
  --output experiments/light_doc_cache/adaptive_policy_from79_add_l6h6_auto_top4_first_top3_second_v1.json
```

Generated specs:

- `adaptive_policy_from79_add_l6h6_auto_top4_first_top3_second_v1.json`
- `adaptive_policy_from79_add_l5h5_auto_top4_first_top3_second_v1.json`
- `adaptive_policy_from79_add_l7h7_auto_top4_first_top3_second_v1.json`
- `adaptive_policy_from79_add_l6h4_auto_top4_first_top3_second_v1.json`

Remote results:

| Added Head | First Doc Gate | First Saving | Second Doc Gate | Second Saving | Second Mean Delta | Result |
|---|---|---:|---|---:|---:|---|
| `6:6` | 13/13 | 17.71% | 6/6 | 17.69% | +1.6298 | rescued |
| `5:5` | 13/13 | 17.71% | 6/6 | 17.69% | +1.5364 | rescued |
| `7:7` | 13/13 | 17.71% | 6/6 | 17.69% | +1.4269 | rescued |
| `6:4` | 13/13 | 17.71% | 6/6 | 17.69% | +1.4672 | rescued |

Result table: `adaptive_search_from79_twodoc_fallback_results_20260710.tsv`

Interpretation:

- Two-document fallback rescued all four candidates that previously failed second-doc validation.
- The quality rescue cost is small in this task mix: second-doc average effective saving drops from 17.75% to 17.69% because 3 of 6 second-doc tasks fall back from 80 to 79 heads.
- This strengthens the evidence that failures are task/document conditional rather than global policy failures.
- The best quality-only adaptive frontier is now a set of validated 80-head defaults with per-doc fallback, not a single fixed global head list.

Claim boundary remains unchanged: this is still an offline recovery-bank simulation. Runtime compression claims require integrating recovery into real KV-cache storage/decode paths and measuring actual memory/runtime.

## Paper-Ready Frontier Table

A reusable frontier-table generator now packages the main evidence into CSV and Markdown:

- Script: `make_adaptive_frontier_table.py`
- Output directory: `paper_frontier_table_20260710`
- CSV: `paper_frontier_table_20260710/frontier_table.csv`
- Markdown: `paper_frontier_table_20260710/frontier_table.md`

The table compares:

| Frontier | First Doc Gate | Second Doc Gate | First Saving | Second Saving | Role |
|---|---|---|---:|---:|---|
| static 79-head fixed policy | 13/13 | 6/6 | 17.63% | 17.63% | fixed-head baseline |
| first-doc adaptive `15:5` | 13/13 | 6/6 | 17.71% | 17.75% | best current margin |
| two-doc adaptive family | 13/13 | 6/6 | 17.71% | 17.69% | robust per-document fallback evidence |

This is the current paper-safe claim:

> Offline task/document-adaptive recovery-bank simulation preserves strict two-document quality gates while improving average effective KV head-token entry saving from the 17.63% fixed-head frontier to about 17.7%.

Do not phrase this as runtime cache compression until the policy is integrated into real KV-cache storage/recovery and memory/runtime are measured.

## Calibrated TinyLLM Read-Path Smoke

The current trained-recovery path is a calibrated multi-source bank fitted from
real TinyLLM `ModelRunner.kv_cache` and then consumed by the default-off
restored-sidecar read-path smoke.

Artifacts:

- Bank and KV-only recovery smoke:
  `tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/`
- Read-path logits smoke:
  `tinyllm_sidecar_read_path_calibrated_smoke_qwen3_0_6b_20260710/`
- Same-target baseline read-path smokes:
  `tinyllm_sidecar_read_path_repeat_last_target_smoke_qwen3_0_6b_20260710/`
  and
  `tinyllm_sidecar_read_path_correlated_target_smoke_qwen3_0_6b_20260710/`
- Final artifact-backed matrix:
  `read_path_recovery_matrix_calibrated_target_qwen3_0_6b_20260710/`
- Bank JSON:
  `tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260710/multi_source_recovery_bank.json`

Target prompt:

```text
Light Doc Cache TinyLLM target prompt for Qwen KV recovery.
```

Read-path matrix result:

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match |
|---|---|---:|---:|---:|---:|---|
| `repeat_last_target` | baseline | 15.2284 | 217 | 4.0625 | 0.598329 | true |
| `correlated_same_layer_target` | baseline | 11.4076 | 246 | 3.59375 | 0.507468 | true |
| `calibrated_multi_correlated_target` | trained | 13.5193 | 219 | 3.890625 | 0.684234 | true |

This is useful because it exercises a trained/calibrated recovery bank through
the TinyLLM restored-sidecar read path and preserves the target-prompt argmax
(`785 -> 785`). It is still not a hot-path integration: the smoke restores a
sidecar tensor and temporarily swaps attention cache pointers for one decode
logits comparison.

Final matrix interpretation:

- The final target-prompt matrix is now backed by all three local summary JSON
  artifacts.
- All three modes preserve argmax (`785 -> 785`) at the same prompt length and
  logical saving (`14` tokens, `17.6339%`).
- On this single target prompt, the calibrated trained bank does not beat the
  simple same-layer correlated baseline: `correlated_same_layer_target` has the
  best missing-token MSE and mean logit diff.
- The provisional stdout-backed matrix remains only as historical handoff
  context in
  `read_path_recovery_matrix_calibrated_target_qwen3_0_6b_20260710_provisional/`;
  use the final matrix directory above for current analysis.

Current claim boundary:

> A trained/calibrated recovery bank can be loaded from JSON and exercised in a
> default-off TinyLLM restored-sidecar read-path logits comparison while
> preserving argmax on one target prompt at the current 17.63% logical saving,
> but this calibrated bank is not yet better than the same-layer correlated
> baseline on the artifact-backed target matrix.

Do not phrase this as 2x+ runtime doc-cache compression or physical KV memory
reduction until the recovery path is integrated into actual KV-cache storage and
measured under runtime allocation constraints.

## Multi-Prompt Calibrated Bank Follow-Up

The calibrated KV smoke now supports multiple calibration prompts:

- `--calibration-prompt-extra`, repeatable
- `--calibration-prompts-file`, one non-empty prompt per line

Implementation detail: each TinyLLM run now copies only the prompt-prefix KV
blocks instead of cloning the full preallocated `ModelRunner.kv_cache`. This
avoids the multi-prompt calibration OOM seen when each sample retained a full
`[2, layers, 500+ blocks, block_size, heads, head_dim]` clone.

Artifacts:

- KV-only multi-prompt bank:
  `tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_multiprompt/`
- Read-path smoke with that bank:
  `tinyllm_sidecar_read_path_calibrated_multiprompt_smoke_qwen3_0_6b_20260713/`
- Extended matrix:
  `read_path_recovery_matrix_calibrated_multiprompt_target_qwen3_0_6b_20260713/`

Remote Qwen3-0.6B result:

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match |
|---|---|---:|---:|---:|---:|---|
| `repeat_last_target` | baseline | 15.2284 | 217 | 4.0625 | 0.598329 | true |
| `correlated_same_layer_target` | baseline | 11.4076 | 246 | 3.59375 | 0.507468 | true |
| `calibrated_single_pair_target` | trained | 13.5193 | 219 | 3.890625 | 0.684234 | true |
| `calibrated_multiprompt_target` | trained | 9.55884 | 146 | 4.0625 | 0.538956 | true |

Interpretation:

- Multi-prompt calibration improves the trained bank materially: it now has the
  best missing-token MSE and lower mean logit diff than the single-pair bank.
- It still does not beat the same-layer correlated baseline on mean logit diff
  for this target prompt.
- Next useful improvement is better source-head selection or a calibration
  objective closer to logits/attention impact, not hot-path integration yet.

## Calibration-Fit Source Selection

The calibrated KV smoke now has a source-head selection ablation:

- `--source-map same_layer`: previous behavior; choose retained same-layer heads
  first, then fall back to other retained heads.
- `--source-map calibration_fit`: rank retained source heads by prefix
  reconstruction fit on the calibration KV tensor.

Artifacts:

- KV-only calibration-fit bank:
  `tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_calfit/`
- Read-path smoke:
  `tinyllm_sidecar_read_path_calfit_smoke_qwen3_0_6b_20260713/`
- Source-fit matrix:
  `read_path_recovery_matrix_calibrated_sourcefit_target_qwen3_0_6b_20260713/`

Remote Qwen3-0.6B result:

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match |
|---|---|---:|---:|---:|---:|---|
| `correlated_same_layer_target` | baseline | 11.4076 | 246 | 3.59375 | 0.507468 | true |
| `calibrated_multiprompt_same_layer_target` | trained | 9.55884 | 146 | 4.0625 | 0.538956 | true |
| `calibrated_multiprompt_calfit_target` | trained | 11.5086 | 211 | 3.4375 | 0.533486 | true |

Interpretation:

- `calibration_fit` is a mixed/negative result: it improves max logit diff and
  slightly improves mean logit diff versus the multi-prompt same-layer trained
  bank, but it worsens missing-token MSE and still does not beat the cheap
  same-layer correlated baseline on mean logit diff.
- Prefix reconstruction fit alone is not a reliable source selector for this
  target prompt. The next selector should use a held-out missing-token objective
  or a logit/read-path aware score.

## Calibration-Holdout Source Selection

The next source selector uses held-out calibration tokens instead of prefix-only
fit:

- `--source-map calibration_holdout`
- For each candidate retained source head, fit on the selected calibration
  prefix and score prediction error on the remaining calibration tokens.
- This is still offline/default-off and does not touch attention hot path or
  physical KV allocation.

Artifacts:

- KV-only holdout-selected bank:
  `tinyllm_calibrated_kv_smoke_qwen3_0_6b_20260713_holdout/`
- Read-path smoke:
  `tinyllm_sidecar_read_path_holdout_smoke_qwen3_0_6b_20260713/`
- Final holdout matrix:
  `read_path_recovery_matrix_calibrated_holdout_target_qwen3_0_6b_20260713/`

Remote Qwen3-0.6B result:

| Mode | Role | Missing MSE | Missing Max Abs | Max Logit Diff | Mean Logit Diff | Argmax Match |
|---|---|---:|---:|---:|---:|---|
| `correlated_same_layer_target` | baseline | 11.4076 | 246 | 3.59375 | 0.507468 | true |
| `calibrated_multiprompt_same_layer_target` | trained | 9.55884 | 146 | 4.0625 | 0.538956 | true |
| `calibrated_multiprompt_calfit_target` | trained | 11.5086 | 211 | 3.4375 | 0.533486 | true |
| `calibrated_multiprompt_holdout_target` | trained | 11.7772 | 252 | 3.10938 | 0.455011 | true |

Interpretation:

- Holdout selection is the first trained/calibrated row that beats the cheap
  `correlated_same_layer_target` baseline on mean logit diff for this target
  prompt (`0.455011` vs `0.507468`) and also improves max logit diff
  (`3.10938` vs `3.59375`).
- It is not the best missing-token MSE row; multi-prompt same-layer remains best
  on MSE (`9.55884`).
- This supports continuing source-selection/logit-aware calibration research
  before runtime hot-path integration.

## Multi-Target Calibration-Holdout Gate

The single-target `calibration_holdout` result above did not generalize. The
multi-target gate evaluates one immutable calibration bank against eight target
prompts and three recovery modes in one TinyLLM process:

- `repeat_last_target`
- `correlated_same_layer_target`
- `calibration_holdout`

Dataset:

- `read_path_multi_target_prompts_v1.json`
- Actual Qwen3-0.6B token buckets:
  - short: `31`, `36`
  - medium: `79`, `51`, `52`
  - long: `212`, `202`, `229`

Local validation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
  python3 -m pytest -q \
  tools/test_light_doc_cache_multi_target.py \
  tools/test_light_doc_cache_recovery_probe.py \
  tools/test_light_doc_cache_runtime.py

PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target.py \
  experiments/light_doc_cache/make_multi_target_read_path_report.py \
  experiments/light_doc_cache/run_tinyllm_calibrated_kv_smoke.py \
  experiments/light_doc_cache/run_tinyllm_sidecar_read_path_smoke.py

bash -n \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh
```

Remote execution:

```bash
CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203 \
TARGET_LIMIT=0 \
TAG=20260714_final_evidence \
LOCAL_OUTPUT=experiments/light_doc_cache/read_path_multi_target_qwen3_0_6b_20260714 \
  experiments/light_doc_cache/run_tinyllm_read_path_multi_target_remote.sh
```

Canonical artifact:

- `read_path_multi_target_qwen3_0_6b_20260714/`
- Calibration bank SHA256:
  `92ab5801523c85faa5e315cc229381818f960cc69940a4de6579688bd8e1fcc0`
- Remote host: `sitian@10.232.195.203` / `n232-195-203`
- GPU: `4`
- Dynamic `TINYVLLM_DIST_PORT` / `MASTER_PORT`: `60495`

Aggregate Qwen3-0.6B result:

| Mode | Completed | Argmax Match | Mean Logit Diff | Median | P90 | Worst | Mean Missing MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `repeat_last_target` | 8/8 | 5/8 | 0.637132 | 0.609053 | 0.903192 | 0.903192 | 12.2817 |
| `correlated_same_layer_target` | 8/8 | 4/8 | 0.642408 | 0.606610 | 0.862850 | 0.862850 | 6.28873 |
| `calibration_holdout` | 8/8 | 5/8 | 1.31780 | 1.22901 | 2.32133 | 2.32133 | 13.8190 |

Gate:

- [x] all eight paired targets completed
- [x] holdout argmax rate was not lower than correlated
- [ ] holdout won at least five targets: `0/8`
- [ ] mean logit diff improved by at least 5%: it regressed by `105.13%`
- [ ] worst relative regression was at most 25%: observed `322.98%`
- [ ] no correlated argmax match regressed: `repetitive` regressed
- [x] actual prompt tokens matched intended length buckets

Decision: **`NO_GO`**.

The weakest target was `repetitive`: mean logit diff increased from `0.548805`
to `2.32133`, and argmax changed from a match to a mismatch. The trained bank
won zero targets on mean logit difference. Do not tune this selector further
on the same eight targets and do not integrate it into the attention hot path
or physical KV allocation.

During the two-target smoke, repeated reuse of one `LLM` exposed a read-path
prototype bug: sidecar materialization assumed sequence KV started at physical
block zero. The smoke now packs KV according to `Sequence.block_table` and
scatters the restored blocks back to their physical slots before the temporary
read-path pointer swap.

Claim boundary:

- This is a default-off restored-sidecar next-token logit comparison.
- The approximately `17.50%` logical byte-saving fraction is accounting
  evidence only.
- It does not demonstrate physical GPU-memory reduction, serving throughput
  improvement, task-answer quality, or production-safe KV-cache integration.
- The next research branch should move to APC/shared-prefix benchmarking or
  adaptive speculative decoding rather than further fitting this selector on
  the gate targets.
