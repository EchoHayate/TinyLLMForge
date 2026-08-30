# Lease-Sealed Persistent Decode MegaKernel Ceiling Audit

Date: 2026-08-30

## Executive Verdict

The source-bound Qwen3-0.6B BF16 TP1 batch-one qualification completed on a
strict-clean NVIDIA A100 80GB PCIe and returned:

```text
GO_PERSISTENT_DECODE_CEILING
```

The result authorizes a separate runtime design for a default-disabled,
lease-scoped persistent decode segment. It does not authorize an immediate
CUDA or Triton implementation without that design, and it is not a measured
runtime speedup.

The frozen optimistic model estimates an aggregate median TPOT ceiling of
`82.155817%` if all eligible segment time could be removed at zero cost.
Per-context optimistic ceilings are `82.736802%`, `82.155817%`, and
`81.051352%` for 256, 2,048, and 8,192 prompt tokens. These numbers establish
headroom only. They do not include persistent-kernel launch, synchronization,
state retention, scheduling, register, occupancy, or fallback cost.

## Canonical Evidence

```text
authoritative checkout:
  /Users/bytedance/dev/TinyLLMForge
Desktop alias:
  /Users/bytedance/Desktop/TinyLLMForge
branch:
  feat/kv-sparse-attention
source commit:
  9bf07719e1344ebf3865e255691b619b1ae9a3aa
source tree SHA-256:
  8aa60f9b9f8fa07a5237557673bfda034fed8328a50828ec561248790a61f3bd
run tag:
  20260830-qwen3-06b-persistent-decode-ceiling-r5
classification:
  GO_PERSISTENT_DECODE_CEILING
local compact evidence:
  artifacts/lease_sealed_persistent_decode/
    20260830-qwen3-06b-persistent-decode-ceiling-r5/
remote staging:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    persistent-decode-ceiling/staging/
    20260830-qwen3-06b-persistent-decode-ceiling-r5
remote primary:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    persistent-decode-ceiling/runs/
    20260830-qwen3-06b-persistent-decode-ceiling-r5
remote controller:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    persistent-decode-ceiling/controller-verification/
    20260830-qwen3-06b-persistent-decode-ceiling-r5
```

The selected device was physical GPU 0,
`GPU-57be086f-e967-c022-3832-93df4fc77bd0`. Both admissions observed an
NVIDIA A100 80GB PCIe at `0 MiB`, `0%` utilization, and zero compute
processes. The controller used the explicit Kerberos cache
`FILE:/Users/bytedance/krb5cc_sitian`, applied the frozen TTL guard, and wrote
all remote task data below the approved mounted `/data00/home/sitian/...`
root.

## Prompt-to-Artifact Checklist

| Requirement | Implementation file and symbol | Test and result | Artifact or verifier evidence | Consequence |
| --- | --- | --- | --- | --- |
| Source and pushed-HEAD identity | `run_lease_sealed_persistent_decode_ceiling_remote.py::{require_pushed_head,validate_source_commit,committed_source_archive}`; `profile_lease_sealed_persistent_decode_ceiling.py::build_source_manifest` | qualification suite `101 passed` | `source_manifest.json`: commit `9bf0771...`, tree `8aa60f9...`; local and remote branch heads matched before launch | `PASS` |
| Kerberos cache and TTL fail-fast | controller `validate_kerberos` calls before staging and launch | `test_validate_kerberos_uses_fixed_file_cache_and_ttl` | explicit cache was valid through 2026-08-31 06:51:02 Asia/Shanghai; no `kinit` or `krenew` | `PASS` |
| Approved mounted storage only | `remote_paths`, `build_worker_plan` | `test_remote_paths_stay_below_approved_task_root`; `test_nsys_command_is_bounded_and_mounted_only`; source leakage test | all staging, primary, controller, scratch, cache, and profiler paths are below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/` | `PASS` |
| Two strict-clean admissions | `wait_for_clean_a100`; `validate_selected_gpu_still_clean`; `run_controller` | `test_strict_clean_a100_requires_zero_everything`; `test_run_controller_rechecks_gpu_immediately_before_launch` | `gpu_admission.json`, remote `gpu_admission_second.json`, and `controller/launch_admission.json` agree on GPU 0 at 0 MiB, 0%, no process | `PASS` |
| Frozen timing inventory | `profile_lease_sealed_persistent_decode_ceiling.py::TIMING_CASES` | `test_timing_inventory_is_five_repetitions_for_three_contexts` | `timing_summary.json`: 15 rows; contexts 256/2,048/8,192; repetitions 0-4 | `PASS_15_OF_15` |
| Frozen trace inventory | structural producer and `persistent_decode_kernel_trace.py::read_decode_trace` | `test_structural_inventory_is_one_matched_case_per_context`; SQLite parser tests | `trace_inventory.json`: three contexts, 16 decode transactions and 430 kernels per context | `PASS_3_OF_3` |
| Exact output token and text identity | producer finalizer and independent verifier | `test_finalize_rejects_structural_output_mismatch`; verifier mutation tests | all five timing repetitions match the structural token IDs and text digest within each context | `PASS_EXACT` |
| Forward and commit accounting | producer row validation and ceiling classifier | runtime-failure and mutation tests | every timing and structural row has 127 target forwards and 127 committed tokens; timing totals are 1,905/1,905; structural totals are 381/381 | `PASS` |
| Kernel launch and duration coverage | `classify_kernel_rows`; `summarize_trace_coverage` | role, unknown-kernel, and coverage tests | 1,290 kernel rows; minimum launch coverage `1.0`; minimum duration coverage `1.0` | `PASS` |
| Candidate-segment reconstruction | `build_candidate_segments`; `finalize_evidence` | segment boundary, stream, signature, interval, and global-ID tests | 48 globally numbered segments; two signatures are stable across all three contexts | `PASS` |
| Profiler perturbation | `lease_sealed_persistent_decode_ceiling.py::build_ceiling_report` | median/P95 perturbation boundary tests | maximum median perturbation `1.710314%` <= `10%`; maximum P95 perturbation `1.405524%` <= `15%` | `PASS` |
| Aggregate headroom | `classify_ceiling` | `test_complete_headroom_returns_go`; aggregate-boundary test | aggregate optimistic median TPOT improvement `82.155817%` >= `5%` | `PASS` |
| Per-context headroom | `classify_ceiling` | per-context boundary test | minimum context ceiling `81.051352%` >= `3%` | `PASS` |
| Candidate CUDA share | `classify_ceiling` | candidate-share boundary test | aggregate candidate CUDA-duration share `100%` >= `4%` within the selected decode transactions | `PASS` |
| Raw trace remote-only handling | `_download_inventory_record`; `stream_and_verify_raw_traces`; compact filter | temporary-file and compact-filter tests; transient-download RED then GREEN | three SQLite traces were streamed, hash checked, and removed; no local `.sqlite` or `.nsys-rep` remains | `PASS` |
| Download resilience and integrity | `_download_inventory_record`; `download_compact_bundle`; `write_controller_receipts` | `test_download_inventory_record_retries_transient_chunk_failure`; manifest tests | every compact file is rehashed in `controller/download_manifest.json`; three attempts allowed per chunk | `PASS` |
| Independent verification | `verify_lease_sealed_persistent_decode_ceiling.py::verify_artifact_directory` | reconstruction, tamper, source-drift, manifest, non-finite, and controller-receipt tests | remote verifier and local verifier agree on classification, 15 timing rows, three contexts, 1,290 kernels, and 48 segments; local verifier also confirms three raw traces | `PASS` |
| Stop-rule compliance | ceiling design and plan | terminal-classification tests | qualification-only work stopped before any persistent CUDA/Triton kernel was implemented | `PASS` |
| Originality and claim boundary | design and this audit | documentation review | TinyLLMForge-specific lease sealing, segmentation, source binding, and verification protocol; persistent kernels and kernel fusion are not claimed as new primitives | `PASS_SCOPED` |

## Frozen Workload

```text
model: Qwen3-0.6B
precision: BF16
tensor parallelism: 1
batch size: 1
contexts: 256, 2048, 8192
generated tokens: 128
temperature: 0
ignore EOS: true
timing repetitions: 5 per context
timing rows: 15 / 15
structural rows: 3 / 3
decode transactions: 48
kernel rows: 1,290
candidate segment rows: 48
failures / fallbacks / rollbacks: 0 / 0 / 0
```

## Benefit and Cost

| Context | Baseline median TPOT | Candidate CUDA duration | Optimistic zero-cost improvement | Profile median perturbation | Profile P95 perturbation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | `2.164975 ms` | `0.679268 ms` per traced transaction | `82.736802%` | `1.710314%` | `0.000000%` |
| 2,048 | `2.450711 ms` | `0.677612 ms` per traced transaction | `82.155817%` | `1.405990%` | `1.405524%` |
| 8,192 | `2.999532 ms` | `0.684488 ms` per traced transaction | `81.051352%` | `1.609426%` | `0.381280%` |

Aggregate and coverage metrics:

```text
aggregate optimistic median TPOT improvement: 82.1558174155%
minimum per-context optimistic improvement:   81.0513518849%
aggregate candidate CUDA-duration share:     100.0000000000%
minimum classified launch ratio:               1.0000000000
minimum classified duration ratio:             1.0000000000
stable cross-context signatures:               2
```

The measured cost of producing this qualification was:

```text
uninstrumented measured E2E sum:                 5.757451201 s
profiled measured E2E sum:                       1.184095697 s
post-timing profile/export/finalize wall span: 367.855818 s
producer wall span:                           1,641.953568 s
remote .nsys-rep bytes:                         126,209,606
remote SQLite bytes:                            505,884,672
total remote raw-profiler bytes:                632,094,278
total remote raw-profiler size:                 602.812078 MiB
local compact evidence bytes:                       703,682
local compact evidence size:                      0.671083 MiB
```

The `100%` candidate share is scoped to CUDA kernel duration inside the
selected, bounded decode transaction ranges. It is not a claim that the
whole request, model, or serving stack can be eliminated. Likewise, the
`82.155817%` result is an intentionally optimistic zero-cost ceiling, not an
observed acceleration.

## Verifier Closure

The remote independent verifier reports:

```text
verified: true
classification: GO_PERSISTENT_DECODE_CEILING
timing rows: 15
structural contexts: 3
kernel rows: 1,290
segment rows: 48
source commit: 9bf07719e1344ebf3865e255691b619b1ae9a3aa
```

The local independent verifier reconstructs the same fields from the compact
bundle and additionally validates all three streamed raw SQLite digests.
The standalone local verifier also exits zero with the same classification.
The compact manifest contains 11 producer artifacts, while the controller
download manifest independently hashes all 12 compact files including
`manifest.json`.

## Attempt History

- r1 was consumed by a second-admission race before workload launch.
- r2 completed measurements but exposed timing-prompt drift.
- r3 completed the producer but exposed per-context versus global segment-ID
  numbering.
- r4 completed the producer and remote verifier, then a transient SSH close
  interrupted the first compact download. It is retained as partial and is
  not canonical.
- r5 includes the chunk-level retry fix and completed the complete producer,
  remote verifier, streamed raw-trace verification, compact download, and
  local verifier lifecycle.

## Classification and Next Boundary

Every frozen threshold passes and `failed_conditions` is empty. Therefore:

```text
PERSISTENT_DECODE_CEILING=GO_PERSISTENT_DECODE_CEILING
RUNTIME_DESIGN_AUTHORIZED=true
RUNTIME_IMPLEMENTED=false
MEASURED_RUNTIME_SPEEDUP_ESTABLISHED=false
PRODUCTION_PROMOTION_AUTHORIZED=false
NEXT_COMMAND=write docs/superpowers/specs/2026-08-30-lease-sealed-persistent-decode-megakernel-runtime-design.md before any CUDA or Triton implementation
```

The next stage must define a realizable subset of the two stable segment
signatures, execution ownership, persistent-state lifetime, synchronization,
fallback and quarantine behavior, retained-memory limits, correctness
authority, and a measured benefit/cost gate. Only that later runtime gate can
establish actual TPOT or throughput improvement.

## Claim Boundary

This qualification proves substantial theoretical headroom in the selected
Exact Greedy K8 decode transactions on one Qwen3-0.6B BF16 TP1 batch-one A100
workload. It proves complete trace coverage, stable candidate signatures,
exact output/accounting behavior, bounded profiler perturbation, and
independent evidence reconstruction.

It does not prove that a persistent kernel can realize the full ceiling, that
the implementation will improve end-to-end latency, or that the result
generalizes to Qwen3-8B, Qwen3.8-27B, tensor parallelism, batching,
non-greedy sampling, variable output lengths, other GPUs, or production
traffic. The underlying persistent-kernel and fusion concepts are prior art;
the original contribution claimed here is the TinyLLMForge-specific
lease-sealed composition and its source-bound execution and verification
protocol.
