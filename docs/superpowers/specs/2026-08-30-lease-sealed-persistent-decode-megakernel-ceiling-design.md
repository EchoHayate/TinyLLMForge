# Lease-Sealed Persistent Decode MegaKernel Ceiling Design

**Date:** 2026-08-30

**Status:** Approved direction; this document authorizes qualification only

**Stage-1 profile:** Qwen3-0.6B, TP1, one NVIDIA A100 80GB PCIe

**Primary target:** exact-greedy batch-1 decode median and tail TPOT

## Objective

Determine whether TinyLLMForge has enough device-side decode work that could
be removed or fused by a lease-scoped persistent decode megakernel to justify
building that runtime.

The qualification must answer one narrow question:

> If every legally eligible small-kernel region and every gap internal to that
> region became free, would the resulting optimistic TPOT improvement be large
> enough to justify a real A100 implementation?

The profile is deliberately more optimistic than an implementation. It may
authorize implementation, but it cannot establish a speedup. A result below
the frozen threshold rejects the implementation direction.

The qualification must report benefit and cost together:

- benefit ceiling: eligible CUDA time, eligible launch count, internal gap
  time, and optimistic TPOT reduction;
- measurement cost: profiler perturbation, trace size, run time, and
  classification coverage;
- projected runtime cost: retained device state, supported shapes and
  dtypes, compilation/capture time, and fallback complexity.

## Motivation and Existing Evidence

Exact Greedy K8 is the current positive batch-1 decode result. It reduces
host-visible token publication and D2H frequency while preserving target
model forwards and exact greedy output.

The subsequent octet-folded CUDA Graph experiment reduced physical launches
inside eligible K8 regions by 87.5%, but improved aggregate median TPOT by
only 0.0194117368%. It also regressed maximum paired TTFT by 2.0833962547%,
so it was classified `NO_GO_CEILING`.

That result rules out a narrow hypothesis:

```text
fewer host CUDA Graph replay calls alone
    -> material exact-greedy TPOT improvement
```

It does not rule out a device-side fusion hypothesis:

```text
fewer operator/kernel boundaries
    + fewer intermediate global-memory round trips
    + persistent on-device control
    -> material exact-greedy TPOT improvement
```

Recent persistent-megakernel work motivates measuring that second hypothesis,
but does not prove it on this runtime or hardware:

- Ada-MK uses adaptive multi-scale persistent megakernels and reports strong
  low-batch inference gains on RTX 5090 and L20 GPUs.
- Mirage Persistent Kernel compiles an LLM into one persistent GPU kernel and
  reports gains on A6000 and H100, while its public results also show that the
  current approach can lose on A100.
- TokenWeave targets distributed compute/communication overlap and depends on
  Hopper-era mechanisms such as NVSHMEM and NVSHARP/Multimem; it is not the
  selected A100 path.
- NanoFlow targets high-concurrency operation-level pipelines rather than the
  current single-request TPOT objective.
- Mooncake targets disaggregated KVCache serving at cluster scale rather than
  one-GPU decode kernel boundaries.

Primary sources:

- https://arxiv.org/abs/2605.11581
- https://arxiv.org/abs/2512.22219
- https://github.com/mirage-project/mirage
- https://arxiv.org/abs/2505.11329
- https://github.com/microsoft/tokenweave
- https://arxiv.org/abs/2408.12757
- https://arxiv.org/abs/2407.00079
- https://github.com/kvcache-ai/Mooncake

## Capability Restatement Without Model Nouns

The reusable capability is:

> For a fixed-shape exact decode transaction, identify contiguous
> device-operation regions whose inputs, outputs, ownership, and failure
> boundaries permit replacement by a resident implementation, then compute a
> conservative implementation authorization from source-bound traces.

The core contract contains no checkpoint name, prompt bucket, exact-burst
width, or GPU product. Those are benchmark-profile policy.

## Considered Approaches

### A. Source-bound kernel census and zero-cost fusion ceiling

Run the unchanged exact runtime with instrumentation off for timing authority,
then replay a bounded matched case under Nsight Systems. Parse CUDA kernels,
CUDA API launches, graph identities, and NVTX decode boundaries. Classify
kernel families by generic operation role and compute an optimistic
zero-cost-fusion ceiling.

Advantages:

- no speculative megakernel implementation;
- runs on the current A100 and existing exact path;
- reuses the repository's source manifests, paired workload, remote
  controller, and independent-verifier patterns;
- can reject a low-headroom direction cheaply;
- does not require unavailable Nsight Compute counters.

Costs:

- the ceiling cannot measure actual HBM traffic;
- kernel-name classification needs fail-closed coverage checks;
- Nsight can perturb timing and therefore needs matched overhead controls;
- a positive ceiling still requires a separate implementation design.

This is the selected approach.

### B. Implement a Triton/CUDA fused segment first

Fuse RMSNorm, residual, position update, token feedback, or selected
pointwise operations before measuring their total contribution.

Advantages:

- produces executable code quickly;
- can measure actual fusion behavior.

Costs:

- may spend substantial implementation time on a sub-1% target;
- makes a negative result ambiguous between low headroom and a weak kernel;
- risks numerical, aliasing, graph-capture, and lifecycle regressions before
  the opportunity is established.

This approach is rejected before a positive ceiling.

### C. Integrate a whole-model persistent-kernel compiler

Adopt or reproduce a compiler/runtime such as Mirage Persistent Kernel.

Advantages:

- attacks kernel boundaries and global-memory traffic comprehensively;
- has published evidence on newer and workstation GPUs.

Costs:

- large compiler and kernel-runtime dependency;
- current public evidence does not establish an A100 win;
- difficult to preserve TinyLLMForge's lease, rollback, quarantine, and
  source-bound evidence contracts;
- far beyond the smallest experiment needed to answer the current question.

This approach is rejected for Stage 1.

## Qualification Architecture

### 1. Canonical uninstrumented timing arm

Reuse the existing Exact Greedy K8 production entrypoint and workload:

```text
contexts:        256, 2,048, 8,192 prompt tokens
generated:       128 tokens
policy:          Exact Greedy K8
topology:        TP1
dtype:           BF16
checkpoint:      Qwen3-0.6B
hardware:        one strict-clean A100 80GB PCIe
```

Run one warmup and at least five measured repetitions per context. Rotate
context order across repetitions. This arm is the timing authority for TTFT,
median/P95/P99 TPOT, E2E latency, throughput, peak allocated memory, and peak
reserved memory.

The arm must preserve:

- exact output-token IDs and decoded-text digest;
- target-forward and committed-token counts;
- zero fallback, failure, rollback, and quarantine;
- one final token D2H per accepted lease;
- the existing one-token graph as fallback; and
- the source-bound exact-burst identity.

### 2. Matched Nsight structural arm

Run one additional matched repetition per context under:

```text
/usr/local/bin/nsys profile
--trace=cuda,nvtx,osrt
```

The structural arm is not timing authority. It provides:

- CUDA Graph launch/API intervals;
- CUDA kernel start/end timestamps;
- kernel names;
- process/thread and correlation identities;
- NVTX request, phase, and decode-transaction boundaries; and
- trace inventory and capture duration.

The remote host currently exposes Nsight Systems 2024.7.1. Nsight Compute is
not available. Therefore the qualification must not claim measured DRAM
bytes, cache hit rate, occupancy, warp efficiency, or instruction mix.

### 3. Generic kernel-role classifier

The parser assigns every CUDA kernel to exactly one generic role:

```text
MATMUL
ATTENTION
NORMALIZATION
ELEMENTWISE
REDUCTION
INDEX_OR_STATE_UPDATE
TOKEN_SELECTION
COPY_OR_FILL
RUNTIME_OR_GRAPH
UNKNOWN
```

Classification uses normalized kernel symbols and an ordered rule table.
Rules are benchmark tooling, not core runtime policy.

The following are never eligible for the Stage-1 resident region:

- `MATMUL`;
- `ATTENTION`;
- host/device copies;
- allocator or graph-runtime work;
- any kernel classified `UNKNOWN`;
- any operation whose producer, consumer, aliasing, or mutation ownership is
  unresolved.

The following are candidate roles only:

- `NORMALIZATION`;
- `ELEMENTWISE`;
- `REDUCTION`;
- `INDEX_OR_STATE_UPDATE`;
- `TOKEN_SELECTION`.

Candidate classification does not assert that a kernel can be deleted. It
only allows its complete observed interval to enter the deliberately
optimistic zero-cost bound.

### 4. Segment reconstruction

Within each exact decode transaction, construct maximal contiguous candidate
segments. A segment ends at:

- a matmul;
- an attention kernel;
- a copy;
- a runtime/graph kernel that cannot be attributed to candidate execution;
- an unknown kernel;
- a host-visible synchronization;
- transaction publication;
- lease commit, rollback, or quarantine; or
- the end of the token step.

For every segment record:

```text
segment_id
request_id
context_bucket
repetition
logical_token_ordinal
first_kernel_start_ns
last_kernel_end_ns
kernel_count
kernel_duration_sum_ns
internal_gap_sum_ns
wall_union_ns
role_histogram
normalized_kernel_signature_sha256
```

The segment identity is role- and source-based. It must not contain Qwen,
prompt text, or one checkpoint's layer names.

### 5. Optimistic ceiling

For each measured token:

```text
eligible_zero_cost_ns =
    sum(candidate kernel durations)
    + sum(gaps internal to candidate segments)

optimistic_improvement =
    eligible_zero_cost_ns / uninstrumented_tpot_ns
```

This bound assumes every candidate segment becomes free, including its useful
math. A real fused or persistent implementation cannot do better without
also changing excluded model work.

The qualification also emits two diagnostic sub-bounds:

```text
launch_gap_only_ns
candidate_kernel_duration_ns
```

Neither sub-bound is a performance claim.

Nsight timestamps come only from the structural arm. The denominator comes
from paired uninstrumented timings. The verifier must reject any artifact
that substitutes profiled TPOT as the speedup denominator.

## Frozen Classification

The classifier returns exactly one terminal class:

```text
GO_PERSISTENT_DECODE_CEILING
NO_GO_PERSISTENT_DECODE_CEILING
INCONCLUSIVE_PROFILE_OVERHEAD
INCONCLUSIVE_TRACE_COVERAGE
INCONCLUSIVE_CORRECTNESS
INCOMPLETE_EVIDENCE
```

### GO

`GO_PERSISTENT_DECODE_CEILING` requires all of:

- aggregate optimistic median TPOT improvement at least 5.0%;
- each context's optimistic median TPOT improvement at least 3.0%;
- aggregate candidate CUDA-duration share at least 4.0%;
- at least one stable candidate segment signature present in every measured
  request and every context;
- at least 98.0% of CUDA kernel launches classified;
- at least 99.0% of CUDA kernel duration classified;
- maximum matched Nsight-versus-uninstrumented median TPOT perturbation no
  greater than 10.0%;
- maximum matched Nsight-versus-uninstrumented P95 TPOT perturbation no
  greater than 15.0%;
- exact output tokens and text digest match between timing and structural
  arms;
- source, checkpoint, hardware, runtime, workload, and graph identities are
  complete and equal;
- no fallback, failure, rollback, quarantine, or external process overlap;
  and
- all remote and local independent verifiers pass.

The 5% threshold is an implementation-authorization threshold, not an
expected speedup. It intentionally leaves room for a real implementation to
retain useful arithmetic and add residency/dispatch overhead.

### NO_GO

Return `NO_GO_PERSISTENT_DECODE_CEILING` when evidence is complete and
correct but either:

- aggregate optimistic median improvement is below 5.0%; or
- any context is below 3.0%; or
- aggregate candidate CUDA-duration share is below 4.0%; or
- no stable cross-context candidate segment exists.

No runtime implementation is authorized after `NO_GO`.

### Inconclusive

Return the corresponding `INCONCLUSIVE_*` class before evaluating headroom
when:

- profiler perturbation exceeds its ceiling;
- kernel launch/duration classification coverage is incomplete;
- timing and structural outputs differ;
- source or runtime identity is incomplete;
- the selected GPU was not strictly clean immediately before launch; or
- required raw artifacts or receipts are missing.

Thresholds are frozen before the first real trace and may not be retuned
after observing results.

## Lease and Ownership Boundary

The qualification does not change scheduler or model-runner behavior.

A later runtime, if authorized, must obey:

- the Scheduler remains the sole issuer of logical-token authority;
- the lease identifies generation, sequence, token budget, KV/block-table
  identity, and commit epoch;
- a resident segment may mutate only lease-owned static state;
- no output is scheduler-visible before normal commit;
- failure before publication leaves the existing fallback usable;
- failure after device mutation but before publication either proves complete
  rollback or quarantines the capability;
- no resident worker survives engine teardown; and
- the ordinary one-token exact path remains the complete fallback.

The name `lease-sealed` describes these ownership constraints. It does not
imply that the Stage-1 profiler creates a persistent GPU worker.

## Layer Map

### Mechanism

- generic CUDA/NVTX interval parsing;
- generic kernel-role classification;
- candidate-segment reconstruction;
- optimistic-ceiling arithmetic;
- fail-closed artifact validation.

### Adapter

- maps TinyLLMForge exact-burst request and graph identities into generic
  request, transaction, and logical-token boundaries;
- supplies exact-output and lease-counter evidence.

### Policy and configuration

- eligible/excluded role allowlist;
- 5% aggregate and 3% per-context ceiling thresholds;
- profiler-overhead limits;
- trace-coverage limits;
- strict-clean GPU admission policy.

### Benchmark profile

- Qwen3-0.6B checkpoint;
- TP1 A100;
- 256/2,048/8,192-token contexts;
- 128 generated tokens;
- one warmup and at least five measured repetitions;
- Exact Greedy K8 runtime.

## Two-Axis Genericity Verdict

```text
mechanism:   reusable candidate
integration: first adopter only
```

The parser, role schema, segment reconstruction, and ceiling arithmetic can
serve another fixed-shape decode runtime. The first adapter and hardware gate
are intentionally tied to Exact Greedy K8 and Qwen3-0.6B. Genericity is not
proven until a second caller or synthetic trace uses the same mechanism
without adding model-specific fields.

## Leakage Guard

Core qualification modules must not contain:

- `qwen`, `llama`, or another model family name;
- `k8`, `octet`, or a fixed burst width;
- prompt-bucket labels;
- checkpoint paths;
- A100-specific policy;
- hard-coded layer counts; or
- model-specific kernel symbols.

Those values belong in the adapter, rule table, or benchmark profile.

A source test must scan the generic modules and fail on prohibited terms.
A synthetic second-caller trace must prove that the generic parser and
classifier do not require the first adopter's metadata.

## Evidence and Artifacts

Use a fresh immutable run tag below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

No remote task output may be written under `/`, `/tmp`, `/private/tmp`, the
historical checkout, or a model-cache directory.

The terminal bundle contains:

```text
source_manifest.json
runtime_manifest.json
gpu_admission.json
workload_manifest.json
timing_rows.jsonl
timing_summary.json
nsys/
  *.nsys-rep
  *.sqlite
trace_inventory.json
kernel_rows.jsonl
segment_rows.jsonl
ceiling.json
remote_verification.json
manifest.json
```

The local controller adds:

```text
controller/plan.json
controller/launch_admission.json
controller/download_manifest.json
controller/local-verification.json
```

The remote and local verifiers independently reconstruct:

- expected timing and trace identities;
- token/text equality;
- kernel classification coverage;
- segment signatures;
- profiler perturbation;
- all ceiling metrics;
- final terminal classification; and
- full file hashes.

The verifier must not trust `ceiling.json` calculations.

## Remote Execution Constraints

- Use `sitian@10.232.195.203`.
- Set `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` for every SSH command.
- Never run `kinit` or `krenew`.
- Use non-persistent SSH sessions.
- Require one A100 with zero compute processes, zero MiB reported used
  memory, and zero percent utilization immediately before launch.
- Do not terminate, attach to, or reuse an external process.
- Preserve failed or interrupted immutable tags.
- Never reuse a partial tag for a new producer.
- Use the existing remote model cache; do not copy or download the model.
- Keep temporary Python caches and profiler exports below the approved remote
  task root.

## Stop Rule

The qualification is the only authorized next implementation.

If the result is not `GO_PERSISTENT_DECODE_CEILING`:

- do not add a persistent-kernel runtime flag;
- do not write Triton/CUDA production kernels;
- do not add scheduler state;
- do not run Qwen3-8B promotion;
- publish the negative or inconclusive result with its measured cost.

If the result is `GO_PERSISTENT_DECODE_CEILING`, write a separate runtime
design. That design must select one concrete segment, define its numerical
contract, retained-state budget, compilation/capture lifecycle, rollback,
quarantine, fallback, and terminal paired performance gate.

## Claim Boundary

A successful qualification proves only:

> On the frozen A100/Qwen3-0.6B/Exact-K8 workload, a complete source-bound
> trace contains enough optimistically removable candidate-region time to
> justify implementing one lease-scoped resident decode segment.

It does not prove:

- an implemented kernel exists;
- actual TPOT, throughput, TTFT, or tail improvement;
- reduced HBM bytes;
- production readiness;
- Qwen3-8B benefit;
- TP or multi-GPU benefit;
- superiority to vLLM, TensorRT-LLM, Ada-MK, or Mirage;
- academic novelty; or
- support on Hopper or Blackwell.

## Originality Boundary

Borrowed ideas:

- persistent megakernel execution;
- on-device intermediate-state reuse;
- adaptive selection among execution scales;
- operation-level scheduling and pipeline decomposition.

TinyLLMForge-specific original combination:

- use an exact-greedy scheduler lease as the authority boundary;
- preserve commit, rollback, quarantine, and one-token fallback semantics;
- use a zero-cost segment ceiling to reject implementation before kernel work;
- reconstruct candidate regions from source-bound exact-runtime traces; and
- require benefit/cost evidence and independent verification before runtime
  promotion.

The combination may be original engineering, but the underlying persistent
kernel, fusion, CUDA Graph, and profiling techniques are not claimed as
original.
