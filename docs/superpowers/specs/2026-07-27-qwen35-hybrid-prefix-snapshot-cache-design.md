# Qwen3.5 Hybrid Prefix Snapshot Cache Design

## Objective

Build the first correctness-safe cache primitive required for lossless
Qwen3.5 shared-prefix reuse.

TinyLLMForge already reuses full-attention KV blocks by chained token hashes.
Qwen3.5 also carries fixed convolution and recurrent state through every
linear-attention layer. Reusing only the KV blocks would therefore start the
suffix from a state that does not represent the reused prefix.

The cache must bind these two histories atomically:

```text
full-attention KV block chain
+ all linear-attention convolution/recurrent state at the same token boundary
```

The first implementation is a dependency-light CPU gate. It proves identity,
publication, acquisition, rollback, eviction, and byte-accounting semantics.
It does not yet connect the cache to the scheduler, ModelRunner, GPU KV
storage, or a real Qwen3.5 checkpoint.

## Industry Direction

The design borrows narrow, proven ideas rather than copying a complete engine:

- vLLM Automatic Prefix Caching uses chained block hashes and exact token
  verification for reusable full KV blocks.
- TensorRT-LLM KV cache reuse associates reusable blocks with cache identity
  and lifecycle metadata.
- SGLang RadixAttention and HiCache show the value of hierarchical prefix
  ownership and explicit cache eviction.
- FlashInfer keeps paged KV layout and execution planning explicit.

TinyLLMForge should retain its current full-block chained hash index for this
phase. A radix tree is unnecessary until workload evidence shows that lookup
or partial-prefix structure, rather than state correctness and storage cost,
is the limiting factor.

Official references:

- `https://docs.vllm.ai/en/latest/design/prefix_caching/`
- `https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html`
- `https://docs.sglang.ai/advanced_features/hicache.html`
- `https://docs.flashinfer.ai/tutorials/kv_layout.html`

## Existing Constraint

`Scheduler._allocate_sequence()` currently rejects hybrid prefix reuse:

```text
hybrid prefix reuse requires aligned state snapshot
```

That fail-closed behavior remains unchanged in this phase. The new component
must establish the missing aligned snapshot contract before scheduler wiring
is allowed.

## Alternatives

### 1. KV-Only Prefix Reuse

Rejected. It is valid for homogeneous transformer layers but incorrect for a
hybrid model whose linear layers depend on persistent recurrent history.

### 2. Snapshot Every Full Block

Rejected as the default. A fixed-state snapshot can be several megabytes, so
duplicating it at every 256-token block can cost more memory than the KV reuse
saves.

### 3. Completion-Point Snapshot Index

Selected.

Publish a state snapshot only for explicitly completed, reusable prefix
boundaries. The first runtime integration will target shared system prompts,
documents, and conversation prefixes that are submitted as complete prompt
segments. A future measured workload may justify periodic anchor snapshots.

### 4. Radix Tree with State on Every Node

Deferred. It supports richer partial-prefix reuse but increases metadata,
ownership, pruning, and state-storage complexity before the state contract is
proven.

## Data Model

Create `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`.

### `Qwen35HybridPrefixKey`

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixKey:
    token_hash: int
    token_count: int
    terminal_block_hash: int
    block_size: int
    model_fingerprint: str
    layout_fingerprint: str
    tensor_parallel_size: int
    dtype: torch.dtype
```

The key is valid only when:

- `token_count` is positive and block aligned;
- `terminal_block_hash` identifies the full KV chain ending at that boundary;
- model, layout, TP, dtype, and block size match exactly.

`token_hash` is not trusted alone. Publication and lookup also receive the
exact prefix token tuple and compare it before returning a hit.

### `Qwen35HybridPrefixSnapshot`

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixSnapshot:
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    convolution_states: tuple[torch.Tensor, ...]
    recurrent_states: tuple[torch.Tensor, ...]
    storage_bytes: int
```

Each block identity is:

```text
(block_id, block_generation, block_hash)
```

The generation changes whenever physical block storage is reset for another
logical prefix. This prevents an old state snapshot from being paired with
new KV bytes that happen to occupy the same block ID.

State tensors are detached contiguous clones. Every tensor represents one
source request row. Their order exactly matches the transaction adapter order
and therefore the linear-layer order.

### Cache Manager

```python
class Qwen35HybridPrefixSnapshotCache:
    def __init__(
        self,
        state_transaction: Qwen35CrossLayerStateTransaction,
        *,
        max_entries: int,
        max_bytes: int,
    )

    def publish(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        lease: HybridStateLease,
    ) -> bool

    def acquire(
        self,
        key: Qwen35HybridPrefixKey,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        leases: tuple[HybridStateLease, ...],
    ) -> bool

    def invalidate_blocks(
        self,
        block_identities: tuple[tuple[int, int, int], ...],
    ) -> int

    def clear(self) -> int

    def observation_snapshot(self) -> dict[str, int]
```

## Publication

Publication is allowed only after all of the following are true:

1. the prefix token count ends at a full block boundary;
2. the full-attention KV write for every referenced block completed;
3. the heterogeneous layer stack completed;
4. the cross-layer state transaction committed;
5. every block identity is current;
6. the source lease validates against every transaction adapter.

`publish()` gathers one source request row from every linear layer, validates
every candidate snapshot, clones all tensors, computes exact storage bytes,
and only then replaces the index entry. Any failure leaves the previous entry
unchanged.

Publishing the same key and exact tokens replaces the previous snapshot and
refreshes recency. A hash collision with different tokens is stored as a
separate exact-token entry.

## Acquisition

`acquire()` is fail-closed:

1. validate key, tokens, block identities, and leases;
2. find an exact key plus exact-token match;
3. verify every current block identity including generation and hash;
4. validate every stored state tensor against every destination row;
5. expand the single cached row into the requested destination batch;
6. commit all linear-layer state through
   `Qwen35CrossLayerStateTransaction.commit()`;
7. refresh LRU recency only after successful commit;
8. return `True`.

It returns `False` without modifying state for a normal miss, token collision,
model/layout/TP/dtype mismatch, or stale block generation.

If cross-layer copy fails, the existing transaction restores every destination
state row and `acquire()` propagates the exception. A failed acquisition does
not refresh recency.

This phase does not allocate or reference-count KV blocks. Runtime integration
must reserve the matching KV chain before calling `acquire()` and release that
reservation if state restoration fails.

## Eviction

Use deterministic LRU with two hard limits:

```text
entry_count <= max_entries
storage_bytes <= max_bytes
```

After a successful publication:

1. insert or replace the entry;
2. evict the least-recently-used entries until both limits pass;
3. never retain a single entry larger than `max_bytes`;
4. report oversize publication as a non-published outcome, not partial data.

`invalidate_blocks()` removes entries that contain any exact block identity in
the invalidation set. Generation mismatch also causes lazy removal during
lookup.

This cache owns state snapshot tensors only. Existing `BlockManager` remains
the owner of KV blocks and token/hash metadata.

## Byte Accounting

`storage_bytes` is the sum of unique tensor storage bytes owned by the entry.
The cache reports:

- current entries;
- current state snapshot bytes;
- peak entries and bytes;
- publishes and replacements;
- hits and misses;
- collision misses;
- stale-block misses;
- validation failures;
- evictions by entry and byte limit;
- invalidations;
- failed restores.

These are logical cache-storage measurements. They are not GPU allocator
savings and cannot be presented as end-to-end memory reduction.

## Correctness Test Matrix

Create `tools/test_qwen35_hybrid_prefix_cache.py` with real
`HybridStateTensorPool`, adapters, and cross-layer transaction.

Required tests:

1. publish one source request row across two linear layers at a block boundary;
2. exact acquire broadcasts that snapshot into three out-of-order destination
   request rows;
3. gathered and stored tensors are clone-isolated;
4. token hash collision with different exact tokens misses;
5. model, layout, TP, dtype, block size, and token count mismatches miss;
6. stale block generation/hash misses and lazily invalidates;
7. malformed stored state fails before writes;
8. late cross-layer copy failure restores all destination rows;
9. publish failure leaves an older entry intact;
10. same-key replacement updates bytes without double counting;
11. deterministic LRU eviction by entry count;
12. deterministic LRU eviction by byte budget;
13. oversize entry is not published;
14. explicit block invalidation removes all dependent entries;
15. clear removes entries and resets current bytes while preserving cumulative
    counters;
16. FP32 and BF16 state tensors;
17. observation counters match actual events.

## Completion Gate

This phase is complete only when:

- the focused cache suite passes;
- cross-layer, packed full, packed linear, packed heterogeneous stack,
  adapter, hybrid-state runtime, and ModelRunner dependency-light regressions
  pass;
- Python 3.9 and Python 3.12 compilation passes;
- `git diff --check` passes;
- staged files remain empty;
- `AGENT_HANDOFF_STATE.md` records proof and claim boundaries.

## Claim Boundary

Passing this gate proves only an atomic CPU cache index for pairing an existing
full-KV block identity with one exact Qwen3.5 linear-layer state snapshot that
can be transactionally restored into multiple request rows.

It does not prove:

- scheduler or ModelRunner prefix reuse;
- real KV block reservation or ref-count integration;
- checkpoint, logits, token, or generation equivalence;
- GPU correctness;
- TTFT, prefill throughput, decode throughput, or hit-rate benefit;
- physical KV or recurrent-state memory savings.

The next phase is scheduler/runtime integration with transactional KV
reservation plus state restoration. Only after exact-output GPU validation may
a source-bound performance gate measure TTFT, prefill tokens/s, request
throughput, hit rate, and allocated/reserved memory.
