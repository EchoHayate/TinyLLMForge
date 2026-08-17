# Qwen3.5 Hybrid Prefix Exact Tensor Interning Design

## Objective

Reduce the physical tensor storage owned by
`Qwen35HybridPrefixSnapshotCache` without changing any cached value, prefix
identity, restoration behavior, or numerical result.

The existing cache makes an independent detached contiguous clone for every
convolution and recurrent tensor in every snapshot. Different prefixes can
therefore own byte-identical state tensors while paying for separate storage.
This gate adds cache-local content-addressed interning so byte-identical
immutable snapshot tensors share one canonical owned clone.

This is exact deduplication, not compression by approximation. It does not use
quantization, low rank, sparsity, truncation, or lossy codecs.

## Scope

Modify only:

- `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`;
- `tools/test_qwen35_hybrid_prefix_cache.py`;
- this design, its implementation plan, and `AGENT_HANDOFF_STATE.md`.

Do not change scheduler, Engine, ModelRunner, block ownership, prefix
publication, transaction semantics, model math, checkpoint loading, or the
canonical Qwen3.5 schema-v2 result.

## Alternatives

### Per-Entry Clone

This is the current behavior. It is simple but duplicates physical storage
whenever snapshots contain identical tensors.

### Digest-Only Interning

Rejected. A digest collision could alias different state values and silently
change restored model state.

### Exact Content-Addressed Interning

Selected. Use a strong digest only to select a small candidate bucket, then
require exact metadata and equality of the contiguous logical `uint8` byte
view before sharing storage. This keeps the cache bitwise exact while avoiding
an all-pairs tensor comparison.

### Approximate State Compression

Deferred. INT4/INT8, low-rank, sparse, and token-selective schemes require
separate accuracy gates and are outside this no-loss phase.

## Intern Identity

Each candidate tensor is first detached, cloned, and made contiguous so the
cache never aliases live pool storage. Its intern bucket key binds:

```text
dtype
shape
device type
device index
SHA-256 of contiguous logical tensor bytes
```

The digest is not sufficient for equality. Every canonical tensor in the
bucket is checked by comparing its contiguous logical `uint8` byte view.
Tensors share only when metadata and every logical byte are identical. This
distinguishes representations that value equality does not, including
`+0.0` and `-0.0`. A digest collision with different bytes creates a separate
canonical record in the same bucket and increments the collision counter.

Hashing a CUDA tensor requires reading its logical bytes and can synchronize
the device. This gate proves storage semantics only; it makes no publication
latency or end-to-end throughput claim. Publication computes each candidate
tensor digest once and reuses the prepared key for candidate-footprint and
intern-table lookup.

## Data Model

Add a private mutable intern record:

```python
@dataclass
class _InternedTensor:
    key: _TensorInternKey
    tensor: torch.Tensor
    refcount: int
    storage_bytes: int
```

The cache owns:

```python
self._intern_table: dict[_TensorInternKey, list[_InternedTensor]]
self._intern_records: dict[int, _InternedTensor]
```

`_intern_table` supports collision buckets. `_intern_records` resolves a
snapshot tensor object back to its record during release.

`Qwen35HybridPrefixSnapshot.convolution_states` and
`recurrent_states` reference canonical tensors. Its existing `storage_bytes`
field remains the logical referenced byte count for that entry. This preserves
the meaning of the snapshot while cache-level `current_bytes` changes to
unique physical bytes.

Canonical tensors are private immutable cache values by contract. Acquire only
reads them and transaction commit copies from them into live state rows.

## Atomic Publication

Publication remains fail-closed:

1. validate identity and source lease;
2. gather and clone every source tensor;
3. validate every clone against its layer state;
4. group exact-equal candidate tensors;
5. compute the candidate's standalone unique physical bytes;
6. reject it without changing entries, refs, counters, or recency when that
   unique footprint exceeds `max_bytes`;
7. resolve groups against the intern table with exact equality;
8. increment existing refs or install new canonical records;
9. create the complete snapshot;
10. only then remove a previous same-entry snapshot and publish the new one;
11. enforce LRU entry and unique-byte limits.

If interning fails before publication, every newly acquired reference is
rolled back and the previous entry remains unchanged.

Replacement acquires the new references before releasing the old references.
Therefore replacing an entry with identical content cannot temporarily destroy
the canonical tensor it will continue to use.

## Release and Lifecycle

Every snapshot tensor occurrence owns one intern reference. Removing,
replacing, invalidating, lazily invalidating, evicting, or clearing an entry:

1. decrements each referenced record;
2. retains the canonical tensor while `refcount > 0`;
3. removes the record from both indexes when `refcount == 0`;
4. subtracts its storage bytes from physical accounting exactly once.

Removing one of two sharing entries must retain storage. Removing the final
entry must release it.

## Byte Limits and Observability

`max_bytes`, `current_bytes`, and `peak_bytes` are based on unique canonical
tensor storage. Shared tensors are not charged twice.

The cache additionally reports:

```text
current_logical_bytes
peak_logical_bytes
deduplicated_bytes
current_interned_tensors
current_intern_references
intern_hits
intern_misses
intern_collisions
```

Definitions:

- logical bytes: sum of tensor storage referenced by all entries;
- physical bytes: sum of unique canonical tensor storage;
- deduplicated bytes: `current_logical_bytes - current_bytes`;
- intern hit: a tensor occurrence reuses an existing canonical value,
  including a duplicate later occurrence in the same publication;
- intern miss: one new canonical tensor is allocated;
- intern collision: a digest bucket contains metadata-compatible content that
  fails exact equality.

All values are cache-owned tensor accounting. They do not prove CUDA allocator,
KV-cache, process RSS, TTFT, or throughput improvement.

## Preserved Semantics

The change must not alter:

- exact key and token identity;
- block identity and generation validation;
- deterministic entry LRU order;
- miss, collision-miss, stale-block, invalidation, or clear behavior;
- cross-layer transaction validation and rollback;
- clone isolation from source pool mutation;
- FP32, BF16, and FP16 support.

## Test Matrix

Extend `tools/test_qwen35_hybrid_prefix_cache.py` with:

1. two different prefixes published from identical state retain two entries
   but own one physical snapshot;
2. matching snapshot tensors have the same object and `data_ptr()`;
3. source mutation after publication does not change canonical values;
4. removing one sharing entry retains storage and removing the final reference
   releases it;
5. partially equal snapshots deduplicate only equal tensors;
6. same-key replacement updates refs and bytes without leaks;
7. forced digest collision with unequal content never shares;
8. forced digest collision between `+0.0` and `-0.0` never shares;
9. acquire output remains exact;
10. an interning failure after one acquired reference rolls it back and
    preserves the previous entry;
11. each publication hashes each candidate tensor exactly once;
12. `max_bytes` uses unique physical bytes;
13. byte-limit eviction still occurs for genuinely different tensors;
14. clear, invalidation, LRU, FP32, and BF16 regressions remain green.

## Completion Gate

This gate is complete only when:

- every new test is observed failing before implementation;
- the focused cache suite passes;
- dependent prefix acquisition/owner/restore and state transaction suites pass;
- Python compilation passes;
- `git diff --check` passes;
- staged files remain empty;
- handoff records measured logical and unique bytes plus claim boundaries.

## Claim Boundary

Passing proves exact physical tensor deduplication inside the CPU-tested
snapshot cache implementation. It proves no numerical change because shared
values pass bitwise logical-byte equality and restore through the unchanged
transaction.

It does not prove production cache reduction until the runtime actually
publishes multiple snapshots with duplicate tensors. It does not prove GPU
allocator savings or inference speedup. Those require a later runtime-bound
workload gate with exact output equivalence and physical memory/performance
measurements.
