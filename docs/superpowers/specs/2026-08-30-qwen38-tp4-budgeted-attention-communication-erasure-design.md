# Qwen3.8 TP4 Peer-Reduction and Residual-Fusion Qualification Design

**Date:** 2026-08-30

**Status:** Revised approved direction; implementation plan pending

**Supersedes:** The R8/R16 projection-replication route previously recorded
in this file

**Stage-1 model:** `Qwen/Qwen3.8-27B`

**Model revision:** `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`

**Runtime topology:** BF16, tensor parallel size four

## 1. Objective

Determine whether TinyLLMForge can reduce Qwen3.8-27B TP4 decode latency by
replacing the current small-message FP32 NCCL AllReduce plus BF16 conversion
plus residual addition with one topology-qualified peer-memory transaction:

```text
rank-local FP32 projection partials
  -> peer-visible fixed staging slots
  -> deterministic four-rank FP32 sum
  -> residual add
  -> one BF16 write
```

The work begins as a default-disabled qualification path. It may enter the
model runtime only if an isolated real-shape four-GPU microgate proves enough
latency headroom, correct synchronization, bounded memory, deterministic
failure handling, and stable numerical behavior.

This design does not claim that CUDA IPC, peer-to-peer access, collective
fusion, or residual fusion is novel. The TinyLLMForge-specific contribution
is the end-to-end protocol: fixed per-layer peer slots, generation-tagged
publication, bounded peer waiting, deterministic rank-order accumulation,
residual ownership, fail-closed topology admission, and a frozen
microgate-to-E2E promotion rule.

## 2. Why the previous R8/R16 design is superseded

The previous revision proposed replacing selected attention-output
AllReduces with:

```text
BF16 input AllGather
  -> full FP32 projection on every rank
  -> no output AllReduce
```

That is not a new candidate. It restores the same fundamental
`ReplicatedWeightRowParallelLinear` execution shape that the Qwen3.5/Qwen3.8
runtime already replaced with phase-split row parallelism.

The authoritative historical evidence is:

- r620 is the valid legacy AllGather/full-GEMM baseline;
- r621 removed all 26,880 measured legacy projection AllGathers and showed
  promising speed, but failed output parity;
- r630 restored FP32 local accumulation and remained `NO_GO` because 6 of 24
  request pairs mismatched;
- r631 preserved the legacy BF16 dense prefill path while using local FP32
  GEMM plus FP32 AllReduce for decode;
- r631 passed 24 of 24 warmup-plus-measured token pairs and was classified
  `PERFORMANCE_PASS`.

r631 versus r620 measured:

| Policy | Steady decode wall | Collective CUDA |
| --- | ---: | ---: |
| recompute | 13.6684% faster | 28.5959% faster |
| exact restore | 6.7341% faster | 33.7067% faster |

The canonical 64-token request-level comparison was mixed:

| Policy | Makespan / throughput | Pooled request p50 E2E |
| --- | ---: | ---: |
| recompute | 8.7896% faster / 9.6366% higher | 4.4865% faster |
| exact restore | 2.6156% slower / 2.5490% lower | 2.5943% faster |

Therefore projection replication is rejected for this stage. Reintroducing
it under an R8/R16 mask would spend memory and compute to restore a path that
the existing runtime has already beaten. The old revision remains visible
in Git history, but it is not implementation authority.

## 3. Corrected current collective boundary

The current Qwen3.8 model path has:

```text
1 vocabulary-embedding AllReduce
+ 64 attention-output AllReduces
+ 1 greedy-token broadcast
= 66 steady-decode collectives
```

The 48 linear-attention and 16 full-attention output projections all use:

```text
rank-local BF16 activation shard
  -> rank-local FP32 GEMM
  -> FP32 AllReduce of [active_tokens, 5120]
  -> BF16 cast
  -> residual add
  -> RMSNorm
```

The MLP output path is replicated and contributes no per-layer MLP
AllReduce. The older static 130-site catalog is wrong for the implemented
model and must be replaced before any terminal gate.

For one active token, each attention collective operates on:

```text
5120 float32 values = 20,480 logical bytes
```

The residual is an immediate consumer, so `async_op=True` followed by an
immediate wait is not an optimization.

## 4. Considered approaches

### A. Fixed-slot peer reduction plus residual fusion — selected

Each TP rank owns a two-entry ring of fixed FP32 staging slots and
generation flags for every selected attention layer. The local projection
writes into the slot selected by the decode-step generation. A short publish
kernel makes the slot visible and stores the generation to a peer-visible
flag. One bounded fused kernel waits for the four expected generations,
reads the four peer slots in fixed rank order, reproduces the baseline cast
and residual-add order, and writes one BF16 output.

Benefits:

- removes the NCCL AllReduce kernel sequence for selected sites;
- removes the separate FP32-to-BF16 conversion and residual-add kernels;
- retains local row-parallel weights and GEMM work;
- adds only small fixed staging/flag state rather than full projection
  weights; and
- directly targets the measured tiny-message communication boundary.

Costs:

- supports only a proven single-node, four-GPU peer-access topology;
- introduces CUDA IPC handles, imported peer pointers, generation lifetime,
  bounded device waiting, and
  rank-failure contracts;
- fixed rank-order summation can differ numerically from NCCL reduction
  order;
- graph capture support is not assumed in Stage 1; and
- a custom CUDA extension is required because Triton does not provide the
  cross-process IPC lifecycle needed here.

### B. NCCL AllReduce with only post-reduction fusion — fallback

Keep the existing blocking AllReduce, then fuse FP32-to-BF16 conversion and
residual addition.

This has a much smaller correctness surface, but it cannot remove collective
launch latency. It is eligible only if Stage 1 shows the post-reduction
kernels themselves provide at least a two-percent transaction opportunity.

### C. Chunked projection/collective overlap — rejected for this stage

Chunking creates a legal producer/consumer wavefront but multiplies launch
counts for a payload of only 20 KiB per active token. RMSNorm still needs the
complete hidden dimension. This route requires separate evidence for a
larger-message regime and is not part of this implementation.

## 5. Qualification-first architecture

### 5.1 Python policy and lifecycle owner

Add a small runtime-neutral owner with this conceptual interface:

```python
class TP4PeerReductionGroup:
    @classmethod
    def create(
        cls,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        layer_count: int,
        max_active_tokens: int,
        hidden_size: int,
    ) -> "TP4PeerReductionGroup": ...

    def reduce_add_residual(
        self,
        *,
        layer_index: int,
        generation: int,
        local_partial: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor: ...

    def close(self) -> None: ...
```

`create()` is collective setup, not request-path work. It must:

- require `world_size == 4`;
- require four distinct CUDA devices in one host;
- verify bidirectional peer access for every selected rank pair;
- allocate fixed local FP32 slots and generation flags;
- exchange only opaque IPC handles through the existing process group;
- open peer mappings exactly once;
- validate shapes, dtypes, devices, and layer counts on every rank;
- converge any rank-local failure before advertising readiness; and
- retain a synchronous NCCL fallback without mutating global state.

`close()` must be idempotent, wait for owned work, close imported handles,
release slots and flags, and converge peer failures.

### 5.2 CUDA extension boundary

The extension exposes only lifecycle primitives and one fused operation:

```text
export_memory_handle(local_slot_or_flag) -> bytes
open_memory_handle(bytes, device) -> opaque mapping
publish_generation(local_flag, generation)
reduce4_cast_add_bf16(
    peer_slots,
    peer_flags,
    expected_generation,
    residual,
    output,
    element_count,
    timeout_clocks,
    status,
)
close_mapping(mapping)
```

The reduction kernel:

1. waits until all four flags equal the expected generation;
2. returns a nonzero device status after the frozen bounded-wait limit rather
   than spinning indefinitely;
3. reads rank 0, 1, 2, and 3 FP32 partials in that exact order;
4. performs accumulation in FP32;
5. rounds the reduced value to BF16 before the residual addition, matching
   the current high-level operation order;
6. performs the BF16 residual addition; and
7. writes the replicated BF16 output on every rank.

No atomic accumulation, unbounded spin wait, host synchronization, dynamic
allocation, or handle exchange is allowed in the timed path.

### 5.3 Fixed slot and generation protocol

For each selected layer and rank:

```text
slot ring length = 2
slot shape       = [max_active_tokens, 5120], dtype=float32
generation flags = 2 peer-visible uint64 values
```

One decode transaction is:

```text
select slot = generation mod 2
  -> local GEMM writes local layer slot
  -> publish_generation writes the expected generation
  -> launch bounded reduce4_cast_add_bf16
  -> next RMSNorm consumes fused BF16 output
```

The two-entry ring is safe only because the current engine has one
synchronous model step in flight per TP worker and the end-of-step token
broadcast prevents a rank from beginning the next step independently. The
verifier must prove that no two live transactions share one
`(layer_index, slot_index, generation)` identity. If the scheduler can
violate that invariant, the candidate fails closed rather than enlarging the
ring after seeing results.

### 5.4 Runtime integration boundary

Stage 1 does not edit Qwen3.8 model semantics. It benchmarks the exact
transaction using real shapes and four processes.

Stage 2, authorized only by a Stage-1 pass, adds an opt-in method to
`RowParallelLinear` that writes the local FP32 GEMM into a caller-owned slot.
The decoder-layer path passes the first residual into the selected attention
output projection and receives the already-reduced, residual-added BF16
tensor. Unselected layers and all prefill execution remain on the current
path.

The baseline must remain byte-for-byte reachable when the policy is absent,
disabled, unsupported, not ready, or closed.

## 6. Memory and topology budget

With `max_active_tokens = 8`:

```text
one FP32 slot       = 8 * 5120 * 4 bytes = 160 KiB
two slots per layer = 320 KiB
64 local slot rings = 20 MiB per rank
imported peer mappings reference peer allocations; they do not duplicate
their storage
```

The implementation must report actual allocated and reserved deltas. Stage 1
allows at most 48 MiB additional allocated memory per rank, including slots,
flags, status/output buffers, bookkeeping tensors, and allocator rounding.

Topology admission requires:

- one host;
- exactly four selected GPUs;
- complete directed peer-access matrix;
- no MIG mode;
- identical compute capability;
- no unexpected external compute process;
- at least 2 GiB free memory beyond the baseline preflight requirement; and
- successful IPC memory and generation-flag round trips for all 12 directed
  peer relationships.

Failure of any requirement classifies the candidate
`INELIGIBLE_TOPOLOGY`; it does not fall back silently inside a measured arm.

## 7. Stage 0: repair the evidence contract

Before timing the candidate:

1. derive the static catalog from the actual module graph;
2. require 64 attention-output sites and zero MLP-output sites;
3. require 66 total steady-decode collectives including embedding and token
   broadcast;
4. collect a real four-rank dynamic census for at least two decode steps;
5. make the handwritten 130-site fixture fail; and
6. preserve immutable r10 artifacts without rewriting them.

This stage fixes the measurement authority; it is not a performance result.

## 8. Stage 1: isolated real-shape microgate

Use four strict-clean A100 GPUs and frozen random inputs for:

```text
active_tokens = 1, 4, 8
hidden_size   = 5120
local input   = 1536
input dtype   = bfloat16
weight dtype  = float32 accumulation copy
residual      = bfloat16
```

Compare:

```text
baseline:
  local FP32 GEMM
  + blocking FP32 NCCL AllReduce
  + BF16 cast
  + residual add

candidate:
  local FP32 GEMM into fixed slot
  + generation publication
  + bounded reduce4/cast/add BF16 fused kernel
```

Use two warmups and at least 200 measured alternating pairs per active-token
count. Measure:

- complete transaction CUDA-event latency;
- host submission latency;
- P50, P90, P95, and P99;
- candidate fused-kernel duration;
- bounded-wait duration and timeout count;
- baseline NCCL and post-reduction durations;
- per-rank skew;
- allocated and reserved memory delta;
- exact token-independent tensor error;
- NaN/Inf counts; and
- cleanup state.

Stage 1 passes only if:

- median candidate transaction latency is at least 10% lower for active
  tokens 1 and 4;
- active tokens 8 does not regress by more than 2%;
- P99 does not regress by more than 3% for any shape;
- every rank produces the same candidate tensor within
  `atol=2e-4, rtol=2e-4`;
- candidate versus baseline is within `atol=2e-2, rtol=2e-3`;
- no NaN or Inf appears;
- no bounded wait times out;
- additional allocated memory is at most 48 MiB per rank; and
- every imported handle is released.

If the Stage-1 median speedup is below 5% for active tokens 1 and 4, stop the
entire direction. Results between 5% and 10% may justify only fallback B,
not model integration.

## 9. Stage 2: bounded runtime integration

If Stage 1 passes, integrate only these fixed masks:

```text
F16 = full-attention layers:
  3, 7, 11, 15, 19, 23, 27, 31,
  35, 39, 43, 47, 51, 55, 59, 63

A64 = all 64 attention-output layers
```

Run `F16` first. `A64` is admitted only if F16:

- preserves exact generated-token parity;
- reduces measured TPOT by at least 3%;
- has no workload P99 regression above 3%; and
- remains within the memory budget.

No mask may be tuned from observed layer timings in the same campaign.

Prefill always uses the existing dense-BF16-preserving path. Decode uses the
candidate only when the peer group is ready and the active-token count is no
greater than eight.

## 10. Stage 3: paired end-to-end gate

Use the frozen workload matrix:

```text
P0: prompt=256,  output=128, concurrency=1
P1: prompt=2048, output=128, concurrency=1
Q0: prompt=256,  output=128, concurrency=4
Q1: prompt=256,  output=128, concurrency=8
Q2: prompt=2048, output=128, concurrency=4
```

Compare baseline, F16, and conditionally A64 with alternating paired order,
two warmups, and at least five measured repetitions. Nsight results are
diagnostic only and cannot determine GO.

The best candidate receives
`GO_TP4_PEER_REDUCTION_RESIDUAL_FUSION` only if:

- every generated token, output length, and stop reason matches baseline;
- aggregate paired median TPOT improves by at least 5%;
- P0 and P1 median TPOT each improve by at least 3%;
- Q0, Q1, and Q2 median TPOT do not regress by more than 2%;
- aggregate output tokens/s improves by at least 3%;
- no workload P99 E2E latency regresses by more than 3%;
- every workload TTFT regresses by at most 3%;
- peak allocated and reserved memory retain at least 512 MiB headroom;
- realized memory delta is within 10% of the declared budget;
- selected attention sites contain no NCCL AllReduce;
- unselected attention sites retain the baseline AllReduce;
- no host synchronization or request-path handle exchange occurs; and
- cleanup and both verifiers pass.

## 11. Failure and fallback semantics

The peer path fails closed before request admission on unsupported topology,
IPC setup failure, rank disagreement, shape mismatch, dtype mismatch, or
budget violation.

After readiness is advertised, any rank-local runtime failure poisons the
peer group. All ranks converge the failure, stop admitting candidate work,
and return to the baseline only after the current distributed request is
aborted consistently. A rank must never continue with NCCL while another
rank continues with peer reduction for the same logical layer.

The candidate is not CUDA-Graph-compatible until a separate capture/replay
gate proves fixed pointers, generation semantics, bounded-wait behavior, and
replay safety. Graph mode must select the baseline path in this stage.

## 12. Evidence bundle

The immutable bundle contains:

```text
source_identity.json
model_manifest.json
gpu_topology.json
peer_access_matrix.json
ipc_roundtrip.jsonl
runtime_collective_catalog.json
dynamic_collective_census.jsonl
microgate_rows.jsonl
microgate_summary.json
layer_policy.json
paired_online_metrics.json
correctness.jsonl
memory_summary.json
resource_samples.jsonl
cleanup.json
classification.json
independent_verification.json
manifest.sha256
```

The independent verifier reconstructs topology admission, source identity,
the 66-site baseline sequence, selected-site substitutions, numerical
thresholds, paired statistics, memory accounting, cleanup, and terminal
classification. Empty census or timing rows, missing ranks, missing
workloads, incomplete repetitions, or missing cleanup evidence are
`INCONCLUSIVE_EVIDENCE`.

## 13. Remote execution constraints

All remote task data must remain below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
```

The controller must:

- verify Kerberos lifetime without running `kinit` or `krenew`;
- admit exactly four strict-clean GPUs;
- keep monitoring and launch authority in the local agent;
- never terminate, adopt, or clean an external GPU process;
- use a fresh immutable attempt tag;
- put extension build and cache directories below the approved remote root;
- never write task data to `/`, `/tmp`, the model cache, or an old checkout;
- keep large traces remote and download only compact evidence;
- perform a second strict-clean check immediately before launch; and
- preserve every failed or partial attempt.

## 14. Classifications and stop rules

```text
GO_TP4_PEER_REDUCTION_RESIDUAL_FUSION
NO_GO_CORRECTNESS
NO_GO_MICROGATE
NO_GO_PERFORMANCE
NO_GO_MEMORY
NO_GO_TAIL_OR_TTFT
INELIGIBLE_TOPOLOGY
INCONCLUSIVE_RESOURCE_IDENTITY
INCONCLUSIVE_EVIDENCE
```

Stop before runtime integration if:

- the topology or IPC round-trip gate fails;
- the Stage-0 sequence is not 66;
- active-token 1 or 4 median microgate speedup is below 10%;
- correctness requires weakening the frozen tolerance;
- P99 regresses beyond the threshold;
- the implementation requires host synchronization in the timed path; or
- cleanup cannot prove all IPC resources were released.

Do not revive R8/R16 replicated projections, reinterpret r630 diagnostic
timings as a pass, or describe a microgate result as end-to-end speedup.

## 15. Claim boundary

A successful terminal gate may support this claim:

> On the qualified single-node A100 TP4 topology, TinyLLMForge replaced
> selected small FP32 attention-output NCCL AllReduces and separate residual
> post-processing with a fixed-slot peer-memory reduction protocol, producing
> the measured TPOT/throughput benefit at the reported memory and topology
> cost.

It must not support claims about other GPU interconnects, TP sizes, models,
dtypes, CUDA Graph execution, multi-node operation, or general collective
replacement without separate evidence.
