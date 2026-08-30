# Lease-Sealed Persistent Decode MegaKernel Ceiling Audit

Date: 2026-08-30

## Executive Verdict

The corrected source-bound Qwen3-0.6B BF16 TP1 batch-one qualification
completed on a strict-clean NVIDIA A100 80GB PCIe and returned:

```text
NO_GO_PERSISTENT_DECODE_CEILING
```

The canonical run is r6. It includes CUDA Graph execution intervals from
Nsight's `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` table and treats each interval as
a `RUNTIME_OR_GRAPH` barrier. The resulting aggregate optimistic median TPOT
ceiling is only `0.960747%`, and candidate kernels account for only
`0.226829%` of traced execution duration. Both are below the frozen
implementation-authorization thresholds.

No persistent CUDA/Triton runtime implementation is authorized. The useful
outcome is negative evidence: the apparent host-side gaps in r5 were occupied
by CUDA Graph executions and are not removable candidate time.

## Superseded r5 Result

r5 reported `GO_PERSISTENT_DECODE_CEILING` and an `82.155817%` optimistic
ceiling. That result is invalid and superseded.

The r5 parser read `CUPTI_ACTIVITY_KIND_KERNEL` but omitted
`CUPTI_ACTIVITY_KIND_GRAPH_TRACE`. Inside a representative 17.619347 ms NVTX
decode transaction, the ordinary kernel table contained only 27 short fill
kernels while the graph-trace table contained eight approximately 2 ms CUDA
Graph executions. The old segment builder therefore stretched one candidate
segment across those executions and counted the occupied intervals as
eliminable internal gaps.

A read-only reparse of all three immutable r5 SQLite files with the corrected
parser produced:

```text
aggregate optimistic median TPOT improvement: 1.000027%
minimum per-context optimistic improvement:   0.814129%
aggregate candidate CUDA-duration share:      0.228078%
```

The corrected independent verifier also rejects the old r5 compact bundle
with:

```text
graph execution inventory is incomplete
```

r5 remains immutable historical evidence. It is not canonical and must not
be used to authorize implementation or advertise headroom.

## Canonical Evidence

```text
authoritative checkout:
  /Users/bytedance/dev/TinyLLMForge
Desktop alias:
  /Users/bytedance/Desktop/TinyLLMForge
branch:
  feat/kv-sparse-attention
source commit:
  23b0a5e3b243873e77362a233c38d4e9a37bda66
source tree SHA-256:
  f63aec8f5cecdd296fcdf75301a00281914900365855761a188726b30fa0c1d3
run tag:
  20260830-qwen3-06b-persistent-decode-ceiling-r6
classification:
  NO_GO_PERSISTENT_DECODE_CEILING
local compact evidence:
  artifacts/lease_sealed_persistent_decode/
    20260830-qwen3-06b-persistent-decode-ceiling-r6/
remote primary:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    persistent-decode-ceiling/runs/
    20260830-qwen3-06b-persistent-decode-ceiling-r6
remote controller:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    persistent-decode-ceiling/controller-verification/
    20260830-qwen3-06b-persistent-decode-ceiling-r6
```

The controller selected physical GPU 0,
`GPU-57be086f-e967-c022-3832-93df4fc77bd0`. Both admissions observed all
eight A100s as clean and selected GPU 0 at 0 MiB, 0% utilization, with zero
compute processes. All task data, scratch files, caches, source staging, and
profiler output remained below the approved mounted `/data00/home/sitian/`
root.

## Corrected Trace Model

Commit `23b0a5e` closes both producer and verifier failure modes:

- the parser requires and reads `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`;
- graph intervals become JSON-safe `cuda_graph_execution` rows;
- those rows classify as `RUNTIME_OR_GRAPH`;
- graph execution duration contributes to total traced duration;
- graph rows terminate candidate segments instead of becoming internal gaps;
- the independent verifier requires graph execution evidence in every decode
  transaction.

TDD and regression evidence:

```text
graph-barrier RED:
  parser omitted cuda_graph_execution and failed the new expectation
graph-barrier GREEN:
  33 passed
qualification suite:
  104 passed
adjacent Exact Burst / Phase-Stitched suite:
  243 passed, 1 deselected
known unrelated repository defect:
  dependency-light preflight references the absent function
  test_model_runner_invalidates_both_burst_graphs
```

The unrelated preflight defect is present in committed HEAD and was not
introduced or modified by this correction.

## Frozen Workload and Evidence Closure

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
execution rows: 1,671
ordinary candidate rows: 1,290
CUDA Graph execution rows: 381
candidate segment rows: 381
failures / fallbacks / rollbacks: 0 / 0 / 0
worker stages: 10 / 10 exit 0
remote verifier: PASS
streamed raw-trace verification: PASS, 3 / 3
standalone local verifier: PASS
```

The 381 graph executions are 127 executions per context. Their presence
explains the long intervals that r5 incorrectly treated as candidate gaps.

## Benefit and Cost

| Context | Baseline median TPOT | Eligible zero-cost time/token | Optimistic improvement | Candidate CUDA share | Profile median / P95 perturbation |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | `2.166331 ms` | `0.024230 ms` | `1.118496%` | `0.264442%` | `1.610125% / 0.000000%` |
| 2,048 | `2.454015 ms` | `0.023577 ms` | `0.960747%` | `0.231296%` | `1.452253% / 1.466947%` |
| 8,192 | `3.003892 ms` | `0.023589 ms` | `0.785290%` | `0.195604%` | `1.570916% / 3.745591%` |

Aggregate gate:

```text
aggregate optimistic median TPOT improvement:
  0.9607474565% < 5.0%       FAIL
minimum per-context optimistic improvement:
  0.7852899440% < 3.0%       FAIL
aggregate candidate CUDA-duration share:
  0.2268287748% < 4.0%       FAIL
minimum classified launch ratio:
  1.0 >= 0.98                PASS
minimum classified duration ratio:
  1.0 >= 0.99                PASS
maximum profiler median perturbation:
  1.6101247242% <= 10.0%     PASS
maximum profiler P95 perturbation:
  3.7455911383% <= 15.0%     PASS
stable cross-context signatures:
  2 >= 1                     PASS
```

Qualification/storage cost:

```text
remote .nsys-rep plus SQLite:
  627,392,539 bytes / approximately 598.33 MiB
local compact evidence:
  1,019,145 bytes / 0.971932 MiB
local SQLite or .nsys-rep:
  none
```

## Prompt-to-Artifact Checklist

| Requirement | Canonical evidence | Verdict |
| --- | --- | --- |
| Source and workload identity | pushed source `23b0a5e...`; tree `f63aec8...`; frozen 3-context workload | `PASS` |
| Approved mounted storage only | all remote paths below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/` | `PASS` |
| Strict-clean launch | two admissions; selected GPU 0 at 0 MiB, 0%, no process | `PASS` |
| Complete execution inventory | 1,290 candidate rows plus 381 graph executions | `PASS_CORRECTED` |
| Graph barriers | every transaction contains `RUNTIME_OR_GRAPH`; 381 separate candidate segments | `PASS` |
| Exact behavior | token IDs/text match; 127 forwards and commits per context; zero anomalies | `PASS_EXACT` |
| Coverage | launch and duration classification both 100% | `PASS` |
| Profiler perturbation | median maximum `1.610125%`; P95 maximum `3.745591%` | `PASS` |
| Aggregate headroom | `0.960747%` against `5%` | `FAIL_THRESHOLD` |
| Per-context headroom | minimum `0.785290%` against `3%` | `FAIL_THRESHOLD` |
| Candidate CUDA share | `0.226829%` against `4%` | `FAIL_THRESHOLD` |
| Independent closure | remote, streamed-raw, and local verifiers agree on r6 | `PASS` |
| Stop rule | no persistent CUDA/Triton runtime implemented | `PASS` |

## Final Classification and Boundary

```text
PERSISTENT_DECODE_CEILING_RUN=20260830-qwen3-06b-persistent-decode-ceiling-r6
PERSISTENT_DECODE_CEILING_SOURCE_COMMIT=23b0a5e3b243873e77362a233c38d4e9a37bda66
PERSISTENT_DECODE_CEILING_CLASSIFICATION=NO_GO_PERSISTENT_DECODE_CEILING
PERSISTENT_DECODE_CEILING_CORRECTNESS=PASS_EXACT
PERSISTENT_DECODE_CEILING_REMOTE_VERIFIER=PASS
PERSISTENT_DECODE_CEILING_LOCAL_VERIFIER=PASS
PERSISTENT_DECODE_RUNTIME_DESIGN_AUTHORIZED=false
PERSISTENT_DECODE_RUNTIME_IMPLEMENTED=false
PERSISTENT_DECODE_MEASURED_SPEEDUP=false
PERSISTENT_DECODE_PRODUCTION_PROMOTION=false
R5_RESULT=SUPERSEDED_INVALID_TRACE_MODEL
NEXT_COMMAND=select a different optimization target with measurable runtime headroom
```

This campaign establishes that a lease-sealed persistent decode megakernel is
not a worthwhile next implementation for this frozen workload. It does not
show that persistent kernels are universally ineffective; it shows that this
specific TinyLLMForge target leaves less than 1% optimistic TPOT headroom once
CUDA Graph execution is represented correctly.
