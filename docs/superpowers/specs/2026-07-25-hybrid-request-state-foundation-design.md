# Hybrid Request-State Foundation Design

## Status

Approved for inline execution under the standing instruction to continue
without per-step confirmation.

This design starts a new engine-infrastructure branch of work. It does not
change or reinterpret the completed Qwen3.5 schema-v2 canonical `NO_GO`, and it
does not replace the blocked Exact CUDA Graph production gate.

## Objective

Add the smallest model-independent foundation TinyLLMForge needs to represent
model-native hybrid attention state:

- token-growing paged KV for full-attention layers;
- fixed-size convolution state for linear-attention layers;
- fixed-size recurrent matrix state for linear-attention layers;
- request-indexed ownership, lifecycle, capacity, accounting, and stale-state
  protection.

The first implementation is CPU-testable infrastructure. It does not add a
Qwen3.5 model, load Qwen3.5 weights, implement Gated DeltaNet math, integrate a
linear-attention kernel, or claim speed, quality, or memory improvements.

## Evidence and Motivation

The frozen Qwen3.5-2B reference evidence already establishes the relevant
state shape:

- 24 decoder layers;
- 18 linear-attention layers and 6 full-attention layers;
- each linear layer has a convolution state shaped `[1, 6144, 4]`;
- each linear layer has a recurrent state shaped `[1, 16, 128, 128]`;
- the observed BF16 reference state therefore occupies `9.84375 MiB` per
  request across the 18 linear layers.

These values are evidence for the canonical model snapshot only. The engine
must calculate storage from a supplied layout and dtype rather than hard-code
Qwen3.5-2B constants.

The current engine cannot represent this state safely:

- `Sequence.block_table` owns only token-growing KV blocks;
- `BlockManager` allocates storage by token count and prefix hash;
- `ModelRunner.kv_cache` assumes homogeneous K/V tensors for every layer;
- recurrent and convolution state are request-indexed, not token-block-indexed;
- prefix-cache reuse of full-attention KV does not by itself reconstruct the
  matching linear recurrent state.

Current official serving implementations likewise treat hybrid models as a
separate cache problem. The reference model updates convolution and recurrent
state independently, while vLLM exposes dedicated Mamba/Gated-Delta state
shape and dtype calculation and restricts unsupported prefix-cache modes.

## Alternatives

### A. Recommended: Independent Request-State Slot Pool

Keep full-attention KV in the existing `BlockManager`. Add a separate,
fixed-capacity request-state slot allocator and a tensor pool whose first
dimension is the slot id.

Advantages:

- matches the real ownership and lifetime of recurrent state;
- gives fixed physical memory accounting before model integration;
- supports ragged request batches through an explicit slot-id vector;
- prevents stale state through generation-tagged leases;
- leaves the existing Qwen3 path unchanged when no layout is installed.

### B. Encode Recurrent State as Fake KV Blocks

Rejected. The recurrent matrix is fixed-size, updated in place, and has no
token-block or prefix-hash semantics. Fake blocks would waste capacity, make
prefix reuse unsound, and hide physical bytes in misleading KV accounting.

### C. Store Python Tensors Directly on `Sequence`

Rejected. `Sequence` crosses process boundaries and is pickled. Carrying GPU
tensors in it would break the current control-plane/data-plane separation,
duplicate storage, and make TP workers disagree about ownership.

## Architecture

### 1. Layout Contract

Create `tinyvllm/engine/hybrid_state.py` with immutable descriptors:

```python
@dataclass(frozen=True)
class HybridStateComponentSpec:
    layer_index: int
    role: Literal["linear_convolution", "linear_recurrent"]
    shape: tuple[int, ...]       # excludes the request-slot dimension
    dtype: torch.dtype

@dataclass(frozen=True)
class HybridStateLayout:
    components: tuple[HybridStateComponentSpec, ...]
```

The layout validates:

- non-negative, unique `(layer_index, role)` keys;
- positive shape dimensions;
- supported floating dtypes;
- at most one convolution and one recurrent component per layer.

It exposes exact `bytes_per_slot`, per-role bytes, and a deterministic
fingerprint derived from layer, role, shape, and dtype.

### 2. Scheduler-Side Lease Allocator

The scheduler control plane owns metadata only:

```python
@dataclass(frozen=True)
class HybridStateLease:
    slot_id: int
    generation: int
    request_id: int
```

`HybridStateSlotAllocator(capacity)` provides:

- `can_allocate()`;
- `allocate(request_id) -> HybridStateLease`;
- idempotence rejection for a request that already owns a slot;
- `release(lease)` with exact owner and generation validation;
- `lease_for_request(request_id)`;
- deterministic free-slot reuse;
- observation snapshots for free, used, owner, and generation state.

Each slot generation increases on every allocation. A stale lease from a
preempted or completed request must never validate after slot reuse.

### 3. ModelRunner Tensor Pool

`HybridStateTensorPool(layout, capacity, device)` preallocates one tensor per
component with shape:

```text
[capacity, *component.shape]
```

The pool does not choose slots. It consumes scheduler leases and provides:

- `activate(lease)`: validate range, reject conflicting live ownership, zero
  the row, and bind `(request_id, generation)`;
- `validate(lease)`: reject stale, unbound, or wrong-owner leases;
- `release(lease)`: validate, zero all component rows, and unbind;
- `component_tensor(layer_index, role)`: return the stable backing tensor;
- `slot_ids(leases)`: return validated slot ids for kernel dispatch;
- exact allocated logical bytes and physical storage bytes.

Zero-on-activate and zero-on-release are both required. The duplicated clear is
intentional: release protects completed-request data, while activate protects
against interrupted cleanup and future allocator/pool desynchronization.

### 4. Sequence Serialization

Extend `Sequence` with:

```python
hybrid_state_slot_id: int = -1
hybrid_state_generation: int = 0
```

These scalar fields must round-trip through `__getstate__` and `__setstate__`
for prefill and decode serialization. They are metadata only; no tensor is
stored on `Sequence`.

Backward compatibility is required for older serialized tuples that do not
contain these fields.

### 5. Scheduler Lifecycle Adapter

The first implementation adds an optional allocator dependency to
`Scheduler`. With no allocator, behavior is byte-for-byte equivalent at the
public API level.

When an allocator is present:

- admission requires both KV capacity and one free hybrid-state slot;
- first prefill allocation acquires a lease and stores its scalar identity on
  `Sequence`;
- chunked-prefill continuation and decode preserve the same lease;
- completion releases KV blocks and the hybrid lease;
- preemption releases both, because the current scheduler recomputes request
  history after readmission;
- readmission receives a new generation even if it reuses the same slot;
- allocation is transactional: a block-allocation failure rolls back a newly
  acquired lease, and a lease-allocation failure leaves blocks untouched.

All direct scheduler allocate/deallocate paths must use shared helper methods;
no lifecycle branch may call only one storage manager.

### 6. Prefix-Cache Boundary

The foundation does not implement hybrid prefix caching.

When a hybrid allocator is installed, admission of a request with
`num_cached_tokens > 0` must fail closed until a later design supplies an
aligned recurrent-state snapshot. Reusing full-attention KV without matching
linear state is semantically invalid.

The future supported choices are:

1. disable prefix reuse for hybrid models and recompute all state; or
2. cache aligned full-KV plus recurrent/convolution snapshots at the same token
   boundary.

This design does not silently choose or simulate option 2.

## Data Flow

### First Admission

1. Scheduler estimates KV blocks and checks a free state slot.
2. Scheduler allocates the hybrid lease.
3. Scheduler allocates KV blocks.
4. `Sequence` carries slot id and generation to every ModelRunner.
5. ModelRunner activates the lease before the first model forward.
6. Future native linear-attention kernels index component tensors by slot id.

### Decode

1. The same lease remains attached to the request.
2. ModelRunner validates every lease in the active batch.
3. Full-attention layers use existing block tables.
4. Linear-attention layers use the request slot vector.
5. State updates occur in the stable pool tensors.

### Finish or Preempt

1. Scheduler releases the KV allocation.
2. Scheduler releases the lease and clears the sequence metadata.
3. ModelRunner release integration, added with native model support, validates
   and zeros the corresponding tensor rows.
4. Reuse increments generation and invalidates all stale references.

## Failure Semantics

All invalid lifecycle operations raise deterministic exceptions:

- allocator exhausted;
- duplicate request allocation;
- out-of-range slot;
- stale generation;
- wrong request owner;
- double release;
- duplicate layout key;
- unsupported dtype or invalid shape;
- tensor-pool activation conflicts;
- hybrid prefix reuse without an aligned state snapshot.

No failure may silently choose another slot, preserve stale state, or fall back
to fake KV storage.

## Testing

### Layout Tests

- exact byte accounting for the frozen 18-layer Qwen3.5-2B BF16 fixture;
- FP32 fixture doubles recurrent and convolution bytes;
- fingerprint stability and mutation sensitivity;
- invalid role, shape, dtype, or duplicate key rejection.

### Allocator Tests

- deterministic allocation and reuse;
- exhaustion;
- duplicate request rejection;
- wrong-owner, stale-generation, and double-release rejection;
- generation increments after reuse;
- snapshot accounting.

### Tensor-Pool Tests

- exact tensor shapes and storage bytes;
- zero-on-activate;
- mutation followed by zero-on-release;
- stale lease cannot read or release a reused row;
- multi-request slot-id validation;
- component backing addresses remain stable.

### Sequence Tests

- prefill and decode pickle round trips preserve lease metadata;
- older tuple formats restore with the disabled sentinel.

### Scheduler Tests

- allocator-disabled behavior remains unchanged;
- state-slot exhaustion blocks admission without consuming KV blocks;
- KV allocation failure rolls back the lease;
- chunked prefill and decode preserve one lease;
- finish releases both managers;
- preemption releases both and readmission gets a new generation;
- hybrid prefix-cache reuse fails closed.

## Validation Boundary

This phase may claim only:

- TinyLLMForge has a tested, model-independent representation for
  request-indexed fixed hybrid state;
- lifecycle, slot reuse, stale-generation protection, and byte accounting pass
  local tests;
- existing Qwen3 scheduling remains unchanged when the feature is absent.

It may not claim:

- native Qwen3.5 serving;
- cached/reference equivalence;
- recurrent-state compression;
- quality retention;
- kernel speedup;
- end-to-end latency or throughput improvement;
- physical GPU-memory savings in a running model.

Those claims require separate native-model, correctness, compression, and
performance gates.

## Follow-On Gates

After this foundation passes:

1. add a Qwen3.5 model/layout adapter and weight mapping;
2. add reference-equivalent BF16/FP32 state-update kernels;
3. bind ModelRunner activation/release to real request lifecycle;
4. run cached continuation and mixed-request correctness against immutable
   official fixtures;
5. only then evaluate recurrent-state INT8/low-rank compression;
6. benchmark model-native linear-attention kernels separately for prefill,
   single-request decode, and multi-request decode.

The completed Qwen3.5 schema-v2 `NO_GO` remains immutable throughout.

## Completion Criteria

The foundation is complete when:

- every layout, allocator, tensor-pool, serialization, and scheduler test
  above exists and passes;
- no existing Qwen3 test regresses;
- feature absence leaves current scheduling behavior unchanged;
- the handoff records exact commands, proven scope, and prohibited claims;
- no README performance claim is added.
