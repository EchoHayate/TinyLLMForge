# Engine Speculative Prepared-Commit Wiring Design

**Date:** 2026-08-12

## Goal

Connect scheduler-selected greedy decode rows to the generic speculative batch
runtime and real ModelRunner callbacks without allowing KV ownership, sequence
metadata, scheduler queues, or draft lifecycle state to become partially
committed.

The first production slice must:

- execute selected rows through the generic draft/runtime/ModelRunner bridge;
- preserve ordinary execution for suppressed rows from the same schedule;
- commit accepted KV directly and release rejected suffix blocks;
- apply multiple verified output tokens per selected sequence;
- synchronize stateful draft indexes only after target metadata commits;
- remain default-off and fail closed for unsupported stateful model execution;
- make no end-to-end performance claim.

## Current Root Cause

The callback plumbing is no longer the blocking issue. The current
`execute_native_speculative_batch()` commits each sequence inside the runtime:

1. `commit_speculative_kv_transaction()` attaches accepted blocks;
2. it also appends accepted tokens to `Sequence`;
3. later engine commit-row construction would observe already-mutated
   completion counts;
4. a later sequence, suppressed ordinary execution, or scheduler postprocess
   failure can leave accepted KV and token metadata only partially applied.

Therefore directly replacing the selected-row guard in `LLMEngine.step()` is
unsafe.

## Considered Approaches

### 1. Inline execution in `LLMEngine.step()`

Run suppressed rows normally, call `execute_native_speculative_batch()` for
selected rows, then append fallback tokens.

This is the smallest diff, but it preserves the current per-sequence commit
loop and duplicates ownership of token mutation between BlockManager and
Scheduler. It cannot provide a credible failure boundary and is rejected.

### 2. Prepared runtime plus engine-owned atomic commit

Split speculative execution into:

- a **prepare phase** that performs target forwards, proposals, reservations,
  KV materialization, and acceptance calculation without attaching blocks or
  changing sequence tokens;
- a **commit phase** that prevalidates every selected and suppressed row, then
  applies one engine-owned KV/metadata/lifecycle commit.

This adds explicit contracts but keeps the existing generic runtime and
callback bridge. It is the recommended approach.

### 3. Move the entire speculative state machine into Scheduler

Scheduler would own callbacks, KV transactions, draft lifecycle, and model
execution.

This could centralize request state eventually, but it couples Scheduler to
ModelRunner and proposal adapters and would be a broad architectural rewrite.
It is rejected for this slice.

## Architecture

### Prepared speculative batch

Add immutable prepared rows plus a batch-owned terminal state:

```python
@dataclass(frozen=True)
class PreparedNativeSpeculativeSequence:
    sequence_id: int
    sequence: object
    proposal: DraftProposal
    first_target_token: int
    target_tokens: tuple[int, ...]
    greedy_accepted_count: int
    accepted_tokens: tuple[int, ...]
    transaction: object | None
    reserved_blocks: tuple[int, ...]
    proxy_block_table: tuple[int, ...]
    first_target_metadata: object | None
    tail_metadata: object | None
    tail_auxiliary: object | None


@dataclass
class PreparedNativeSpeculativeBatch:
    sequences: tuple[PreparedNativeSpeculativeSequence, ...]
    first_target_callback_count: int
    tail_callback_count: int
    timing_ms: dict[str, float]
    state: str = "prepared"
```

`prepare_native_speculative_batch()` owns all active transactions until one of
two terminal operations:

```python
commit_prepared_native_speculative_batch(...)
rollback_prepared_native_speculative_batch(...)
```

The prepare function may reserve and materialize private KV blocks, but it
must not:

- extend a live sequence block table;
- append accepted or fallback tokens;
- publish prefix-cache hashes;
- release rejected suffix blocks;
- synchronize or release a draft index.

The existing `execute_native_speculative_batch()` remains as a compatibility
wrapper for profiler/tests. It calls prepare plus the legacy standalone commit
path. `LLMEngine` uses the prepared API only.

### KV commit plans

BlockManager receives a non-mutating planner:

```python
prepare_speculative_kv_commit(
    transaction,
    seq,
    accepted_tokens: tuple[int, ...],
) -> SpeculativeKVCommitPlan
```

The plan contains:

- original sequence/token/block snapshots;
- committed and unused reserved block IDs;
- accepted materialized token count and materialized end;
- full-block cache publications computed from
  `original_token_ids + accepted_tokens`;
- the terminal transaction state.

The KV commit plan does **not** append tokens. Token metadata belongs to the
Scheduler postprocess commit.

All selected plans are validated before any plan mutates ownership. A batch
commit applies plans in schedule order and keeps enough allocator/hash/sequence
block-table snapshots to restore the whole batch if an injected commit failure
occurs. Physical KV bytes may remain in released scratch slots, but no live
request or prefix-cache index may reference them after rollback.

### Engine runtime installation

Add a source-agnostic runtime object:

```python
@dataclass(frozen=True)
class EngineSpeculativeRuntime:
    draft_adapter: DraftAdapter
    lifecycle: DraftLifecycle | None = None
```

Installation is explicit and idempotent only for the same object:

```python
LLMEngine.install_speculative_runtime(runtime)
```

Installation validates:

- adapter capabilities are batch-capable;
- scheduler selection is enabled;
- scheduler and adapter proposal limits are compatible;
- target hidden/logit requirements are supported by the callback bridge.

With no installed runtime, selected rows retain the current pre-execution
fail-closed error.

### Draft lifecycle

Stateful proposal sources use an optional generic lifecycle:

```python
class DraftLifecycle(Protocol):
    def register_sequence(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> None: ...

    def synchronize_verified_history(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> int: ...

    def release_sequence(self, sequence_id: int) -> None: ...
```

`SAMDraftAdapter` can satisfy this protocol directly. Stateless n-gram
execution installs no lifecycle.

Lifecycle ordering is:

1. register after `Sequence` construction and before scheduler admission;
2. synchronize every sequence that successfully emits verified tokens,
   including ordinarily executed suppressed rows;
3. release a finished sequence only after synchronization succeeds.

If synchronization fails after target commit, the engine marks the
speculative runtime poisoned and refuses later selected execution. Target
sequence state remains authoritative; it is not rolled back to match a stale
draft index.

## Step Data Flow

For a schedule containing selected and suppressed rows:

1. Scheduler returns the ordinary batch and immutable selection record.
2. Engine validates the record and builds the exact partition.
3. Engine validates the installed runtime before draining release events.
4. Suppressed rows execute through ordinary `ModelRunner.run()` using their
   original row metadata. Pending hybrid-state releases are carried exactly
   once by this call.
5. If there are no suppressed rows, pending releases are flushed through the
   existing acknowledged release path before speculative callbacks.
6. Selected rows execute `prepare_native_speculative_batch()` with:
   - one first-target RPC for the selected batch;
   - one tail RPC per distinct fixed query length;
   - private per-sequence KV transactions.
7. Engine builds selected output rows from the prepared snapshots, not from
   mutated sequences.
8. Scheduler builds a non-mutating postprocess plan covering the original
   scheduled order:
   - selected decode rows receive one or more output tokens;
   - suppressed decode/sample rows receive the ordinary single token;
   - suppressed prefill/no-sample rows preserve current prefill behavior.
9. Engine prevalidates all KV commit plans, scheduler mutations, finish
   decisions, queue membership, and lifecycle operations.
10. Engine commits the KV plan batch, then the scheduler postprocess plan.
11. Lifecycle histories synchronize from the committed `Sequence.token_ids`.
12. Finished lifecycle entries release.
13. Observability records selected output counts, accepted draft counts,
    fixed-Q group count, runtime timings, and suppression reasons.

Selected rows are always greedy decode rows. Mixed batches may contain
suppressed prefill rows, but the selected subset itself never enters prefill
or non-sampling execution.

## Scheduler Postprocess Contract

Add immutable rows:

```python
@dataclass(frozen=True)
class ScheduledOutputRow:
    sequence_id: int
    output_tokens: tuple[int, ...]
    speculative: bool
    accepted_draft_tokens: tuple[int, ...] = ()


@dataclass(frozen=True)
class PreparedSchedulerPostprocess:
    scheduled_sequence_ids: tuple[int, ...]
    rows: tuple[ScheduledOutputRow, ...]
    is_prefill: bool
    do_sample: bool
    batch_kind: str | None
    decision_now_ns: int | None
    step_end_ns: int | None
```

Preparation rejects:

- missing, extra, duplicate, or reordered sequence IDs;
- multi-token output for non-selected rows;
- output beyond remaining budget;
- tokens after an effective EOS;
- queue/status state that does not match the scheduled batch;
- selected rows that are not decode/sample/greedy rows.

Commit reuses the existing prefill, mixed, progress, finish, release, and SLO
semantics but consumes a token tuple per row. It appends selected accepted
tokens and fallback token exactly once.

## Failure Semantics

### Before prepared commit

Any ordinary ModelRunner, first-target, proposal, reservation, tail, or
validation failure:

- rolls back every active selected KV transaction;
- restores drained release events when the release-carrying ordinary call
  fails;
- leaves sequence tokens, live block tables, scheduler queues, and draft
  histories unchanged.

Ordinary suppressed GPU writes may have overwritten the current decode slot,
matching existing retry behavior, but no host token is published.

### During KV commit

An injected failure restores:

- sequence block tables;
- free/used block ownership;
- block refcounts, generations, hashes, and token metadata;
- hash indexes;
- all transaction states.

No sequence token or scheduler queue mutation has occurred yet.

### During scheduler commit

All validation occurs before KV mutation. The scheduler commit contains only
deterministic list/deque/status/token operations. Tests inject failures at
each mutation boundary and require restoration of sequence metadata, queue
membership, block ownership, progress maps, and release events.

### During lifecycle synchronization

KV and target sequence metadata stay committed. The runtime becomes poisoned,
and future selected rows fail before ModelRunner execution. This avoids
pretending a stale proposal index is safe.

## Compatibility

- Default-off execution remains byte-for-byte on the existing ordinary path.
- Existing selection records and fixed-Q callback bridge remain unchanged.
- Existing profiler callers may continue using
  `execute_native_speculative_batch()`.
- No model name or proposal-source branch enters generic engine, scheduler,
  execution, or commit code.
- Stateful recurrent/convolution model rows remain fail closed in
  `ModelRunner` until non-KV state transactions exist.
- Nonzero temperature rows remain scheduler-suppressed and ModelRunner
  guarded.

## Tests

Dependency-light tests must prove:

1. prepare performs callbacks/reservation/materialization but no live metadata
   commit;
2. rollback releases every active reservation;
3. KV commit planning does not mutate sequences;
4. batch KV commit is all-or-nothing under per-plan injected failures;
5. selected and suppressed rows preserve original schedule order;
6. pending release events are carried exactly once;
7. commit-row construction uses pre-prepare snapshots;
8. selected output tokens append exactly once;
9. mixed suppressed prefill/decode behavior matches the ordinary path;
10. EOS and output-budget finish semantics match existing Scheduler behavior;
11. SAM registers, synchronizes after commit, and releases after sync;
12. lifecycle failure poisons later speculative execution;
13. no installed runtime preserves the current guard before ModelRunner work;
14. default-off ordinary source shape and return behavior remain unchanged.

The existing 314-test focused matrix, native verifier dispatch script,
chunked-prefill script, dual-version `py_compile`, generic source scan, and
git hygiene remain mandatory.

## Promotion Boundary

Completing this design proves production code-path integration and
transactional host-state semantics. It does not prove:

- GPU numerical parity;
- TP1 or TP4 correctness;
- throughput, TTFT, or TPOT improvement;
- long-context benefit;
- support for stochastic speculative decoding;
- safety for recurrent/convolution state without a non-KV transaction.

The overall classification remains `NOT_PROMOTABLE`.
