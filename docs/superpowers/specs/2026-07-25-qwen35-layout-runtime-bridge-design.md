# Qwen3.5 Layout and Runtime Bridge Design

## Status

Approved for inline execution under the standing instruction to continue
without per-step confirmation.

This phase extends the model-independent hybrid request-state foundation. It
does not implement Qwen3.5 model math, load Qwen3.5 weights in TinyLLMForge,
add a Gated DeltaNet kernel, start a GPU process, or reinterpret the immutable
Qwen3.5 schema-v2 canonical `NO_GO`.

## Objective

Add two CPU-testable pieces needed before native hybrid-model execution:

1. a strict Qwen3.5 Hugging Face config adapter that derives a TP-local
   `HybridStateLayout`; and
2. an explicit scheduler-to-worker lifecycle bridge that releases an old
   generation on every rank before a reused slot generation is activated.

The result is a validated control-plane and storage-lifecycle contract. It is
not Qwen3.5 inference support and makes no correctness, quality, compression,
latency, throughput, or GPU-memory claim.

## Existing Boundaries

The completed foundation already provides:

- `HybridStateLayout` and exact byte accounting;
- generation-tagged `HybridStateLease` metadata;
- `HybridStateSlotAllocator` owned by the scheduler;
- `HybridStateTensorPool` owned by a future model runner;
- `Sequence` serialization of request id, slot id, and generation;
- transactional paired KV/state allocation and release;
- fail-closed hybrid prefix reuse.

The missing ordering boundary is important. `Scheduler.postprocess()` releases
a completed request only after `ModelRunner.run()` returns. A worker cannot
infer those releases from the next active batch because a live request may
simply be unscheduled in that step. Conversely, a scheduler may preempt a
request, release its slot, and allocate that slot to a new generation before
the next worker call. Therefore active-batch scanning is neither a release
protocol nor a safe stale-state guard.

## Alternatives

### A. Recommended: Explicit Release Events Piggybacked on `run`

The scheduler records every released lease. `LLMEngine` drains those events
immediately before each `run` RPC and sends them with the active sequences.
Each ModelRunner rank applies releases first, then activates or validates the
active leases.

Advantages:

- one ordered TP broadcast for both release and activation;
- no batch scan or hidden liveness inference;
- old generation is cleared before a reused generation is activated;
- works for completion, preemption, and same-step slot reuse;
- adds no extra steady-state synchronization call.

An explicit `release_hybrid_state(leases)` RPC remains available for idle
drain and engine shutdown when no later `run` exists.

### B. Dedicated Release RPC After Every Scheduler Postprocess

Safe but rejected as the primary path. It adds another shared-memory broadcast
and worker dispatch after every step that finishes or preempts requests. The
piggyback protocol preserves the same ordering with fewer control-plane calls.

### C. Infer Releases from the Next Active Batch

Rejected. Requests may remain live while absent from a scheduled batch, and a
slot may be reused before the next batch is inspected. This approach can
either clear live state or retain stale state.

## 1. Strict Qwen3.5 Config-to-Layout Adapter

Add a focused adapter in `tinyvllm/engine/qwen35_hybrid_state.py`:

```python
def build_qwen35_hybrid_state_layout(
    hf_config,
    *,
    tensor_parallel_size: int,
    dtype: torch.dtype,
    speculative_tokens: int = 1,
) -> HybridStateLayout:
    ...
```

`speculative_tokens` is the number of token positions represented by the
convolution cache at dispatch time. The non-speculative TinyLLMForge bridge
uses `1`; future multi-token speculative execution must pass its exact active
token width rather than silently reuse the default.

### Accepted Config Shape

The adapter reads `hf_config.text_config` when present, otherwise `hf_config`.
It requires all of the following explicit fields:

- `num_hidden_layers`;
- `layer_types`;
- `linear_num_key_heads`;
- `linear_num_value_heads`;
- `linear_key_head_dim`;
- `linear_value_head_dim`;
- `linear_conv_kernel_dim`.

It accepts only a schedule whose normalized values are exactly
`linear_attention` or `full_attention`. It does not infer the schedule from
`full_attention_interval`, class names, module names, or layer indices.

The adapter rejects:

- missing or non-integral fields;
- booleans used as integers;
- non-positive dimensions;
- a `layer_types` length different from `num_hidden_layers`;
- a schedule with no linear-attention layer;
- unsupported layer-type strings;
- unsupported state dtype;
- TP size outside `[1, 8]`;
- key or value head counts not divisible by TP size;
- convolution channel count not divisible by TP size;
- `speculative_tokens < 1`.

### TP-Local Shape Formula

For each `linear_attention` layer:

```text
local_key_heads = linear_num_key_heads / tensor_parallel_size
local_value_heads = linear_num_value_heads / tensor_parallel_size

conv_channels =
    linear_key_head_dim * linear_num_key_heads * 2
    + linear_value_head_dim * linear_num_value_heads

local_conv_channels = conv_channels / tensor_parallel_size
conv_width = linear_conv_kernel_dim - 1 + speculative_tokens

linear_convolution shape =
    (local_conv_channels, conv_width)

linear_recurrent shape =
    (local_value_heads,
     linear_value_head_dim,
     linear_key_head_dim)
```

The layout contains no component for `full_attention` layers because their
state remains in the existing paged KV cache.

For the frozen Qwen3.5-2B architecture at TP=1, BF16, and
`speculative_tokens=1`, the adapter must produce:

- linear layers `0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22`;
- convolution shape `(6144, 4)` per linear layer;
- recurrent shape `(16, 128, 128)` per linear layer;
- `10,321,920` bytes per request.

At TP=2, each rank owns `(3072, 4)` convolution state and
`(8, 128, 128)` recurrent state per linear layer. Per-rank layout bytes are
half the TP=1 bytes; this is logical layout accounting only, not a measured
GPU-memory result.

## 2. Release Event Contract

The event payload is the existing immutable `HybridStateLease`:

```python
HybridStateLease(
    slot_id: int,
    generation: int,
    request_id: int,
)
```

No separate lossy tuple or request-id-only event is introduced. Generation
and owner identity are required to reject stale or duplicated releases.

`Scheduler` gains:

```python
def drain_hybrid_state_release_events(
    self,
) -> tuple[HybridStateLease, ...]:
    ...
```

When `_release_request_storage()` validates a lease, it:

1. releases KV storage;
2. releases the allocator lease;
3. clears lease metadata on the sequence;
4. appends the exact released lease to a FIFO pending-event queue.

Events are published only after both scheduler-owned resources have released
successfully. If validation or either release fails, no event is published.
Draining returns the current FIFO tuple and atomically clears the local queue.
With no hybrid allocator, the queue remains empty.

## 3. LLMEngine Dispatch Ordering

`LLMEngine.step()` changes the worker call to:

```python
released_leases = self.scheduler.drain_hybrid_state_release_events()
token_ids = self.model_runner.call(
    "run",
    seqs,
    is_prefill,
    do_sample,
    batch_kind,
    released_leases,
)
```

The drain occurs after scheduling and immediately before the worker RPC.
This includes:

- releases created by the preceding step's postprocess;
- releases created by preemption during the current schedule;
- an old generation released before a new sequence in the same scheduled
  batch reuses that slot.

If scheduling or dispatch raises after the drain, `LLMEngine` restores the
drained events to the front of the scheduler queue before propagating the
exception. A control-plane failure must not lose a release event.

`LLMEngine.exit()` drains remaining events through
`release_hybrid_state(leases)` before broadcasting `exit`. This protects the
case where the final postprocess releases requests and no subsequent `step()`
occurs.

## 4. Rank-Local Runtime Bridge

Add `HybridStateRuntimeBridge` beside the existing pool:

```python
class HybridStateRuntimeBridge:
    def __init__(self, pool: HybridStateTensorPool):
        ...

    def prepare_batch(
        self,
        released_leases: tuple[HybridStateLease, ...],
        active_leases: tuple[HybridStateLease, ...],
    ) -> torch.Tensor:
        ...

    def release(
        self,
        released_leases: tuple[HybridStateLease, ...],
    ) -> None:
        ...
```

`prepare_batch()` performs exactly two phases:

1. release every event in FIFO order;
2. activate every active lease in sequence-row order, then return the validated
   `torch.int32` slot-id tensor.

This order is mandatory. It permits:

```text
release(slot=0, generation=1, request=7)
activate(slot=0, generation=2, request=9)
```

in one dispatch while rejecting:

- stale release after generation 2 is already active;
- duplicate release;
- active lease without a prior release when the slot is bound to another
  generation;
- duplicate active rows with conflicting lease metadata.

Repeated activation of the same live lease remains idempotent and must not
zero live state.

## 5. ModelRunner Integration

`ModelRunner.run()` gains an optional final argument:

```python
released_hybrid_state_leases: tuple[HybridStateLease, ...] = ()
```

Before preparing model inputs, it derives active leases from every sequence
with enabled hybrid metadata and calls the rank-local runtime bridge. The
returned slot-id tensor is stored as step-local metadata for future Qwen3.5
model dispatch.

The bridge is optional:

- no active hybrid metadata and no release events: preserve current Qwen3
  behavior;
- hybrid metadata or release events with no installed bridge: fail closed;
- installed bridge: release first, then activate/validate active rows.

`ModelRunner.release_hybrid_state(leases)` forwards to the same rank-local
bridge and is broadcast by the existing `call()` mechanism to every TP rank.

This phase does not instantiate a pool in production `ModelRunner.__init__`.
Pool construction must wait for a native model loader that deliberately opts
into a validated Qwen3.5 layout. Auto-detecting Qwen3.5 while the runner still
constructs `Qwen3ForCausalLM` would create a false support claim.

## 6. Failure and Retry Semantics

All lifecycle errors fail closed with deterministic exceptions:

- malformed Qwen3.5 config;
- non-divisible TP layout;
- unsupported dtype or layer type;
- stale or duplicate release event;
- wrong request owner;
- slot generation conflict;
- active hybrid metadata without a runtime bridge;
- release-event drain lost by a failed dispatch.

The bridge never guesses a schedule, silently changes TP size, clears an
unmentioned slot, or treats request absence as release.

## 7. CPU-Only Test Matrix

### Config Adapter

- canonical TP=1 BF16 component keys, shapes, bytes, and fingerprint;
- TP=2 per-rank shapes and half-byte accounting;
- FP32 byte doubling;
- `text_config` wrapper support;
- explicit full/linear schedule preservation;
- missing fields, length mismatch, unsupported layer type, invalid dtype,
  invalid speculative width, and TP divisibility rejection.

### Runtime Bridge

- release generation 1 then activate generation 2 in one call;
- release occurs before activation by observing zeroed reused state;
- same active generation is idempotent and preserves mutations;
- stale, duplicate, wrong-owner, and out-of-range events fail;
- active row ordering matches returned slot ids;
- multiple independent request slots remain isolated.

### Scheduler Event Queue

- finish publishes one exact event;
- preemption publishes one exact event;
- no allocator publishes no event;
- drain is FIFO and destructive;
- failed release publishes nothing;
- restore places failed-dispatch events before newer events.

### Engine and ModelRunner Boundary

- a lightweight fake scheduler/model runner proves `step()` forwards releases
  before the active batch;
- dispatch failure restores drained events;
- model-runner helper fails closed without a bridge;
- a CPU pool bridge processes release-before-activate without importing or
  constructing CUDA model state.

Existing foundation, sequence, scheduler, chunked-prefill, arrival-gate,
model-runner spec-verify, and CUDA-graph gate tests must remain green.

## 8. Scope and Claim Boundary

Included:

- strict config-to-layout conversion;
- exact TP-local logical state accounting;
- explicit generation-safe release events;
- rank-local release-before-activate bridge;
- dormant ModelRunner and LLMEngine plumbing;
- CPU-only behavioral tests.

Excluded:

- Qwen3.5 model class and weight loading;
- linear-attention forward math;
- causal convolution or Gated DeltaNet kernels;
- recurrent state updates;
- hybrid CUDA Graph capture;
- hybrid prefix caching;
- GPU execution and remote experiments;
- any compression, quality, speed, throughput, or memory-benefit claim.

The next production gate after this phase is a separate native Qwen3.5
model-math and kernel design, followed by remote correctness evidence. GPU0
admission remains a hard prerequisite for any model process.
