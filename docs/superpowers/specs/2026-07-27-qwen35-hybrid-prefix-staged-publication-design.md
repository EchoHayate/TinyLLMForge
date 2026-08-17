# Qwen3.5 Hybrid Prefix Staged Publication Design

## Objective

Add the cache-local prepare/commit/rollback primitive required before an
all-rank hybrid-prefix snapshot publication transaction can be safe.

Today `Qwen35HybridPrefixSnapshotCache.publish()` clones state and immediately
makes the entry visible. A future all-rank coordinator cannot call that method
during prepare: rank 0 could expose a reusable snapshot while another rank
rejects the same prefix.

This gate separates owned snapshot capture from visibility:

```text
prepare_publication:
  validate + clone + hold privately

commit_publication:
  intern + atomically replace/publish + enforce LRU

rollback_publication:
  drop private clones without changing visible entries
```

The existing `publish()` API remains available and becomes a local
prepare-then-commit convenience wrapper.

## Scope

Modify only:

- `tinyvllm/engine/qwen35_hybrid_prefix_cache.py`;
- `tools/test_qwen35_hybrid_prefix_cache.py`;
- this design, its plan, and `AGENT_HANDOFF_STATE.md`.

Do not add Engine, Scheduler, ModelRunner, transport, or live-step wiring in
this phase.

## Single In-Flight Contract

Each cache permits exactly one prepared publication at a time. This bounds
transient clone storage to one snapshot per rank and matches the first
all-rank coordinator, which will serialize publication tickets.

Preparing while another handle is live raises `RuntimeError` without changing
the existing prepared state. Commit or rollback consumes the handle.

## Public Handle

Add:

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixPreparedPublication:
    publication_id: int
    key: Qwen35HybridPrefixKey
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    storage_bytes: int
```

The handle contains identity and accounting only. Owned tensor clones remain
private to the cache.

Handles are cache-instance-bound and one-shot. A foreign, stale, committed, or
rolled-back handle is rejected before entry mutation.

## Prepare

```python
def prepare_publication(
    self,
    key,
    token_ids,
    block_identities,
    lease,
) -> Qwen35HybridPrefixPreparedPublication | None
```

Prepare:

1. validates identity and lease;
2. gathers one row from every linear layer;
3. makes detached contiguous owned clones;
4. validates every clone;
5. computes logical and standalone unique bytes;
6. returns `None` for a standalone oversize snapshot;
7. stores the private immutable clones only after all checks pass.

It does not:

- intern tensors into visible canonical storage;
- change entries or LRU;
- change visible physical/logical bytes;
- replace an existing same-key entry.

Source mutation after prepare cannot affect staged clones.

## Commit

```python
def commit_publication(
    self,
    prepared,
) -> bool
```

Commit validates the exact current handle, consumes the staged state, interns
its tensors, atomically replaces/publishes the entry, enforces limits, and
returns whether the new entry remains resident after LRU enforcement.

If interning fails, all acquired refs and intern counters roll back, the
previous visible entry remains unchanged, and the prepared publication remains
available for explicit rollback. A successful commit consumes the handle.

## Rollback

```python
def rollback_publication(
    self,
    prepared,
) -> None
```

Rollback validates and consumes the exact current handle, releases its private
clones, and leaves visible entries, LRU, bytes, and publish counters unchanged.

Rollback is intentionally not idempotent at the cache level. The future
participant transaction will provide ticket-level idempotency.

## Existing Publish

`publish()` becomes:

```text
prepare
if oversize: return False
commit
on commit exception: rollback if still prepared, then re-raise
```

Its existing public behavior and counters remain unchanged. Internal staging
counters are additional observability.

## Observability

Add:

```text
current_prepared_publications
current_prepared_bytes
peak_prepared_bytes
publication_prepares
publication_commits
publication_rollbacks
publication_prepare_conflicts
```

`current_prepared_bytes` is actual private clone storage and is separate from
visible cache `current_bytes`.

## Correctness Matrix

1. prepare leaves entries, visible bytes, and LRU unchanged;
2. staged tensors are source-isolated;
3. commit publishes exact state and consumes the handle;
4. rollback leaves an existing same-key entry unchanged;
5. preparing replacement is invisible until commit;
6. foreign/stale/replayed handles fail before mutation;
7. second concurrent prepare is rejected;
8. oversize prepare returns `None` with no live handle;
9. commit interning failure preserves the previous entry and allows rollback;
10. existing immediate `publish()` tests remain green;
11. exact tensor interning, collision, byte budget, and restore behavior remain
    green.

## Claim Boundary

Passing proves only a cache-local exact staged publication primitive. It does
not prove all-rank atomicity, runtime publication, cache hits, production
memory reduction, or speedup.

The next gate is an all-rank publication participant/coordinator that uses this
prepare/commit/rollback contract without wiring it into `LLMEngine.step()`.
