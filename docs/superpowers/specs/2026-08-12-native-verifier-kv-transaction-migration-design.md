# Native Verifier KV Transaction Migration Design

## Objective

Migrate the existing native speculative verifier path from caller-owned bare
reserved block IDs to `SpeculativeKVTransaction`.

The previous phase added a model-independent transaction contract in
`BlockManager`. The real native verifier still calls:

```text
reserve_append_blocks()
commit_accepted_tokens()
release_reserved_blocks()
```

directly from `tools/profile_ngram_commit.py`. This phase connects that
existing native target forward to the transaction without changing the draft
source, acceptance math, output event schema, or accepted-KV
no-rematerialization behavior.

## Scope

This phase covers only `verify_and_commit_block(..., verifier_mode="native")`.

It must:

- begin one speculative KV transaction before target execution;
- use the transaction's reserved block IDs in the proxy block table;
- acknowledge `query_len` appended KV positions only after native tail
  forward completes;
- commit accepted tokens through the transaction;
- rollback on first-target, preparation, tail-forward, acceptance, or commit
  failure;
- preserve zero/one/partial/full acceptance, EOS, budget, oracle capture, and
  finish/deallocation behavior;
- preserve `accepted_kv_rematerialization.decode_calls == 0`;
- preserve existing event fields and their meaning.

It does not:

- move orchestration out of the profiler;
- integrate scheduler batching;
- enable native spec verify with KV offload;
- modify legacy rematerialization;
- add MTP or a learned draft model;
- make performance claims.

## Alternatives

### 1. Migrate Native and Legacy Paths Together

Rejected. Legacy mode rematerializes accepted KV after target verification
and has different offload/writeback behavior. Combining both migrations would
make transaction correctness inseparable from legacy compatibility.

### 2. Wrap Bare IDs in a Profiler-Local Transaction

Rejected. It would duplicate the ownership state machine and leave
BlockManager unable to validate generations or exactly-once terminal
operations.

### 3. Native-Only BlockManager Transaction

Selected. Native verification already writes the accepted KV directly. It is
the path that benefits immediately from explicit materialization and
commit/rollback semantics.

## Lifecycle

Before the first target decode:

```python
transaction = block_manager.begin_speculative_kv_transaction(
    seq,
    proposed_token_count=len(draft_tokens),
)
reserved_blocks = list(transaction.reserved_block_ids)
```

The proxy block table remains:

```python
proxy_block_table = list(seq.block_table) + reserved_blocks
```

After the native tail forward completes, including the `query_len == 0` K=1
case:

```python
block_manager.mark_speculative_kv_materialized(
    transaction,
    query_len,
)
```

`query_len` equals `len(draft_tokens) - 1`, exactly the appended token
positions whose KV the native verifier writes. The first target decode writes
the pre-existing pending token and is not counted in the transaction's
appended materialization field.

After acceptance, EOS truncation, and output-budget truncation:

```python
block_manager.commit_speculative_kv_transaction(
    transaction,
    seq,
    accepted_tokens,
)
```

The transaction becomes `committed`. Event accounting derives
`committed_blocks` from the sequence block-table extension and
`released_blocks` from the original transaction reservation, preserving the
current JSON schema.

## Failure Handling

The profiler keeps a nullable transaction owner:

```text
None -> active transaction -> None after successful commit
```

On any exception:

- if an active native transaction remains in `reserved` or `materialized`,
  call `rollback_speculative_kv_transaction(transaction, seq)`;
- if rollback itself fails, preserve both the original phase error and the
  cleanup failure in the raised native verifier error;
- legacy mode continues releasing its bare reserved block list;
- context reset remains in `finally`.

Compatibility validation still happens before transaction begin. Unsupported
native modes therefore allocate no blocks.

## Legacy Boundary

`verifier_mode="legacy_rematerialize"` remains unchanged:

```text
reserve_append_blocks
-> legacy target forward/rematerialization
-> commit_accepted_tokens
-> release_reserved_blocks on failure
```

This keeps real KV-offload experiments and historical profiler behavior
isolated from the native transaction migration.

## Test Strategy

Extend the dependency-light native fixture in
`tools/test_ngram_speculative.py`.

The fake block manager will expose both API families:

- transaction calls for native mode;
- old bare-list calls for legacy mode.

New and updated assertions:

1. native full acceptance calls begin, mark, and transaction commit exactly
   once;
2. native K=1 marks zero materialized appended tokens;
3. native zero/one/partial/full acceptance preserves block and pending-token
   lifecycle;
4. native tail failure rolls back a reserved transaction;
5. native commit failure rolls back a materialized transaction;
6. unsupported native mode fails before transaction begin;
7. native no-rematerialization remains zero decode calls;
8. legacy mode does not call any transaction API;
9. existing n-gram speculative and block-manager transaction matrices remain
   green.

## Completion Boundary

Passing this phase proves that the existing single-sequence native profiler
uses the generic transaction contract.

It does not prove a production scheduler runtime, batching, real KV offload,
MTP, exact model parity, TP1/TP4 behavior, long-context behavior, or any TPOT,
TTFT, throughput, memory, H2D-byte, or acceptance improvement.
