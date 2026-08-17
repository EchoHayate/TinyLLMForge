# Qwen3.5 TP4 decode phase-split completion audit

Audit date: 2026-08-11

Overall status: `INCOMPLETE`

This audit intentionally separates source-level evidence from real TP4
correctness and performance evidence. A passing unit test, a deterministic
source tar, or the earlier r621 structural result is not accepted as a
substitute for a fresh end-to-end run of the current source tree.

## Objective converted to success criteria

The user objective is to optimize Qwen3.5 decode tensor-parallel
communication and weight layout instead of spending more effort on the
already-small restore path.

Completion requires all of the following:

1. The two Qwen3.5 attention output projections use local axis-1 weight
   shards during decode.
2. Decode no longer reconstructs full projection inputs with the legacy
   replicated-weight AllGather path.
3. Decode uses a valid replacement collective and records its real
   collective type/count.
4. Long prefill retains the legacy numerical path so the decode
   optimization does not change recompute outputs.
5. All real TP4 recompute/exact-restore output-token pairs match.
6. Repeated real TP4 measurements on GPUs `2,4,5,6` support any reported
   wall/CUDA/collective improvement.
7. The resource policy remains controlled shared/non-exclusive: each GPU
   has at least 25 GiB free and at most 10 percent utilization at admission.
8. Attempt-scoped cleanup is proven clean without killing unrelated
   processes.
9. Every real run uses a fresh tag and preserves prior attempts.
10. Source, commands, receipts, comparison, limitations, and next action
    are persisted.

## Prompt-to-artifact checklist

| # | Requirement or gate | Status | Concrete evidence | Missing or weak coverage |
|---|---|---|---|---|
| 1 | Optimize decode TP AllGather/weight layout, not restore internals | PASS at source level | `tinyvllm/models/qwen35_components.py` enables `RowParallelLinear` for linear/full attention output projections; `tinyvllm/layers/linear.py` implements local decode GEMM plus `row_parallel_all_reduce` | Requires fresh real-run confirmation that bundled code is the executed code |
| 2 | Local axis-1 checkpoint shard contract | PASS | `tinyvllm/models/qwen35_checkpoint_binding.py`; `tools/test_qwen35_checkpoint_target_binding.py` = 4 passed; `tools/test_qwen35_checkpoint_assignment.py -k output_projection` = 1 passed, 8 deselected | Full assignment file has three unrelated segmented-QKV/packed-oracle failures |
| 3 | Legacy decode AllGather absent | PASS only for r621 source/run | `qwen35-tp4-decode-row-parallel-20260811-r621-attempt001/row_parallel_comparison.json`: baseline 26,880 legacy AllGather rows, candidate 0 | r621 failed token parity; current phase-split source has not run |
| 4 | Row-parallel decode collective present | PASS only for r621 source/run | Same r621 comparison: baseline 0, candidate 26,880 row-parallel AllReduce rows | Current phase-split source has not produced a decode profile |
| 5 | Reduce communication count | NOT ACHIEVED | r621 replaced 26,880 AllGather rows with 26,880 AllReduce rows | Collective type/volume improved, but operation count was not reduced; do not claim fewer collectives |
| 6 | Preserve legacy long-prefill math | PASS at unit/source level | `RowParallelLinear(preserve_dense_prefill=True)` caches full BF16 prefill weight; `forward_prefill()` uses input AllGather plus full `F.linear`; both attention shells dispatch it only for `context.is_prefill` | Needs real recompute token/logit validation |
| 7 | Decode still uses FP32 partial accumulation | PASS at unit/source level | `tools/test_qwen35_output_projection_row_parallel.py`: 3 passed, including fixed BF16 counterexample and dense-prefill oracle | Needs GPU execution evidence |
| 8 | Full-attention phase dispatch | PASS | `tools/test_qwen35_full_attention_shell.py`: 12 passed, including prefill-only `forward_prefill` dispatch | Unit evidence only |
| 9 | Linear-attention phase dispatch | PASS | `tools/test_qwen35_linear_attention_shell.py`: 15 passed, including prefill-only `forward_prefill` dispatch | Unit evidence only |
| 10 | Concrete Qwen3.5 factory enables phase split only on the two target projections | PASS | `tools/test_qwen35_concrete_component_factory.py`: 4 passed; assertions require `accumulation_dtype=torch.float32` and `preserve_dense_prefill=True` | Unit evidence only |
| 11 | Decode profile wiring rejects legacy Qwen3.5 projection path | PASS | `tools/test_decode_internal_profile_wiring.py`: 4 passed | Does not prove runtime parity/performance |
| 12 | Comparison classifier remains fail-closed | PASS | `tools/test_qwen35_decode_row_parallel_comparison.py`: 4 passed | No current candidate comparison exists |
| 13 | Current remote source contract includes every phase-split production file | PASS | `source_prep_receipt.json`; tar members include `linear.py`, both attention shells, and `qwen35_components.py` | None at source-prep level |
| 14 | Current source bundle is deterministic | PASS | Two generated tar files both SHA256 `f791f27e807e602f889345d301b72035dcd4a93d55a32adf51fd5eb3eaefb79c`; tree SHA256 `6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837` | A real runner invocation will generate a new bundle and must match this tree or explain drift |
| 15 | Real TP4 token parity | FAIL for r621 and r630; PENDING for current source | r621 comparison `output_parity=false`; r630 diagnosis: 6 of 24 pairs mismatch, all request-1 at token zero | Fresh phase-split TP4 run required |
| 16 | Real TP4 repeated performance | PENDING | r621 showed raw improvement but `NO_GO`; r630 diagnostic medians show improvement but parity failed | Fresh current-source five-repetition result required |
| 17 | GPUs fixed to `2,4,5,6` | PENDING for current source | Runner constants and prior attempts use these GPUs | Must be proven in fresh entry/worker guards |
| 18 | Minimum 25 GiB free and utilization at most 10 percent | PENDING | Runner guard encodes thresholds | New full prefill-weight residency may increase memory; fresh guard and worker guards required |
| 19 | Shared/non-exclusive, no dummy reservation, no unrelated process kill | PENDING | Runner resource policy and attempt-scoped cleanup implementation | Fresh receipts required |
| 20 | Fresh run tag and old-attempt preservation | READY | Reserved next tag: `qwen35-tp4-decode-phase-split-20260811-r631-attempt001`; r620/r621/r630 retained | Do not create/reuse the real attempt until remote execution starts |
| 21 | r630 cleanup verified clean | BLOCKED | Runner called cleanup in `finally`, but aggregation exception prevented receipt persistence | Read-only remote process check blocked by expired Kerberos |
| 22 | Current candidate cleanup verified clean | PENDING | No execution yet | Fresh attempt receipt required |
| 23 | Canonical manifest/case matrix/profile schema unchanged | PASS by scoped inspection | Current edits are limited to linear layer, two attention shells, Qwen3.5 component factory, tests, diagnosis, and handoff | Recheck scoped diff before final completion |
| 24 | Static validation | PASS | Fresh run: 47 targeted tests passed, 8 deselected; `python3 -m py_compile` PASS; scoped `git diff --check` PASS | Static evidence is not end-to-end evidence |
| 25 | Final comparison artifact | MISSING | Expected current-run `row_parallel_comparison.json` | Generate only after parity gate passes |
| 26 | Final completion audit and report | MISSING | This file is a current incomplete audit, not a completion receipt | Build final audit/report from current real-run artifacts |
| 27 | Handoff persistence | PASS for current state | `AGENT_HANDOFF_STATE.md`; r630 `residual_parity_diagnosis.md`; this source-prep directory | Append fresh-run result after execution |

## Current source-prep artifacts

```text
experiments/qwen35_hybrid_state/
  qwen35-tp4-decode-phase-split-source-20260811-r631-prep001/
    source_prep_receipt.json
    first/benchmark_source.tar
    second/benchmark_source.tar
    completion_audit.current.md
```

Current source identity:

```text
tree SHA256: 6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837
tar SHA256:  f791f27e807e602f889345d301b72035dcd4a93d55a32adf51fd5eb3eaefb79c
owned files: 93
```

## Blocking dependency

The current `sitian` Kerberos tickets for `jump-proxy-hl.byted.org` and
`10.232.195.203` were issued on 2026-07-29 and are expired. The jump host
rejects GSSAPI and direct port 22 access times out.

Until credentials are refreshed, the following required gates cannot be
completed:

- read-only r630 residual-process check
- current GPU resource guard
- fresh phase-split TP4 execution
- real token/logit parity
- real collective inventory
- repeated performance comparison
- attempt-scoped cleanup proof

## Next executable sequence after credential refresh

1. Confirm `klist` has non-expired credentials for the jump and target.
2. Read-only search remote command lines for the exact r630 tag.
3. Confirm GPUs `2,4,5,6` satisfy the controlled-shared guard.
4. Start `qwen35-tp4-decode-phase-split-20260811-r631-attempt001`.
5. Require all 12 workers to return zero.
6. Require all 24 recompute/exact-restore token pairs to match.
7. Confirm decode contains zero legacy projection AllGather rows and the
   expected row-parallel AllReduce inventory.
8. Compare repeated wall/CUDA/collective medians with r620.
9. Confirm attempt-scoped cleanup is `CLEAN`.
10. Generate `row_parallel_comparison.json`, final completion audit, final
    report, and handoff update.

The objective is not achieved until these pending real-run gates pass.
