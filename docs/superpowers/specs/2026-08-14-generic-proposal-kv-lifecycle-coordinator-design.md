# Generic Proposal-KV Lifecycle Coordinator Design

**Date:** 2026-08-14

**Status:** Selected local continuation design

## Objective

Extract proposal-KV transaction registration, batched finalize, rollback,
sequence release, and lifecycle evidence from
`Qwen35MTPProposalExecutor` into a model-independent coordinator.

The coordinator must let a future second learned proposal structure reuse the
same proposal-KV ownership contract without copying Qwen3.5-specific code. The
existing Qwen3.5 native-MTP executor must delegate to the coordinator without
changing proposal tokens, target-KV transactions, verifier selection, exact-Q
grouping, side-state selection, or offload accounting.

## Audit Result

The production target-KV path already avoids accepted-prefix replay:

```text
tail target forward
-> mark_speculative_kv_materialized(query_len)
-> prepare_speculative_kv_commit(accepted_tokens)
-> commit_speculative_kv_commit_batch()
-> scheduler commit
```

The first accepted proposal token consumes the sequence's existing pending
token. Therefore target KV materialization and proposal KV materialization
both commit `max(accepted_tokens - 1, 0)` newly written entries. No accepted
token decode replay or per-token KV rematerialization remains in the
production batch-native path.

The remaining reuse gap is proposal-source-side lifecycle ownership:

```text
Qwen35MTPProposalExecutor
  _proposal_transactions
  _batch_tickets
  _batch_ticket_transactions
  proposal authority rows
  register group proposals
  prepare/commit/rollback finalize batch
  active-transaction release guard
```

These operations depend on `ProposalKVCache`, sequence identity, epoch, and
`DraftProposal`; they do not depend on Qwen3.5 tensors or MTP math.

## Considered Approaches

### 1. Implement KV8/KV4 Offload First

Rejected for this slice. Current KV offload is explicitly fp16/bf16-only.
Adding KV8 correctly requires a storage format, scales, D2H/H2D transport
semantics, GPU dequantization, attention integration, generation identity,
and real movement validation. A local-only contract change would risk
claiming support without a real data path.

### 2. Implement the Qwen3.5 MTP CUDA Graph Backend

Rejected as the next task. It is useful for proposal latency but remains a
Qwen3.5-specific optimization and does not reduce the integration work for a
second learned model structure.

### 3. Extract a Generic Proposal-KV Lifecycle Coordinator

Selected. It moves already exercised transaction behavior behind a reusable
model-neutral API while retaining Qwen3.5 as the first production consumer.
It is locally testable and directly reduces the amount of model-specific code
required by the next learned drafter.

## Architecture

Create:

```text
tinyvllm/engine/proposal_kv_lifecycle.py
```

The file owns:

```python
@dataclass(frozen=True)
class ProposalKVRegistration:
    sequence_id: int
    sequence_epoch: int
    proposal: DraftProposal


class ProposalKVLifecycleCoordinator:
    def register_batch(
        self,
        rows: tuple[ProposalKVRegistration, ...],
    ) -> tuple[DraftProposal, ...]: ...

    def prepare_finalize_batch(
        self,
        rows: tuple[ProposalFinalizeRow, ...],
    ) -> str: ...

    def commit_finalize_batch(self, ticket_id: str) -> None: ...

    def rollback_finalize_batch(self, ticket_id: str) -> None: ...

    def assert_sequence_releasable(
        self,
        sequence_id: int,
        sequence_epoch: int,
    ) -> None: ...

    def release_sequence(
        self,
        sequence_id: int,
        sequence_epoch: int,
    ) -> None: ...

    def authority_snapshot(self) -> dict: ...
```

The coordinator is constructed with:

- one `ProposalKVCache`;
- a non-empty ticket namespace such as `"qwen35-mtp"`;
- no model module, tensor, graph runner, target-KV manager, or scheduler.

## Registration Contract

`register_batch()` accepts successful proposal rows after model-specific
execution has materialized their proposal-KV transactions.

For every non-empty proposal it validates:

- exact `DraftProposal` type;
- non-negative unique sequence IDs;
- non-negative sequence epochs;
- non-empty unique proposal transaction IDs;
- transaction existence in the configured `ProposalKVCache`;
- transaction state is `materialized`;
- transaction sequence ID and epoch match the registration row;
- `len(proposal.token_ids) == len(transaction.staged_slot_ids) + 1`;
- the transaction is not already active in the coordinator.

Empty proposals must not contain a transaction ID and are returned unchanged.

Registration is batch-atomic. If any row fails, every still-unregistered
transaction referenced by the batch that remains `reserved` or
`materialized` is aborted in reverse order. Existing registered
transactions are never aborted by a failed later batch.

The return value preserves input order and contains the original
`DraftProposal` objects.

## Finalize Contract

`prepare_finalize_batch()` validates unique sequence and transaction
identities against active registrations, then calls
`ProposalKVCache.prepare_finalize()` once per row.

If preparation fails partway:

- already prepared underlying tickets are rolled back in reverse order;
- remaining materialized transactions in the requested batch are aborted in
  reverse order;
- every requested transaction is removed from active coordinator ownership;
- lifecycle evidence records the terminal cleanup state;
- no coordinator batch ticket is published;
- the failed batch is not retryable with the same proposal transactions.

On success, the coordinator creates one namespaced batch ticket and records
the ordered underlying ticket and transaction IDs.

`commit_finalize_batch()` consumes the batch ticket exactly once, commits
underlying tickets in row order, marks evidence committed, and removes active
registrations.

`rollback_finalize_batch()` consumes the batch ticket exactly once, rolls
underlying tickets back in reverse order, marks evidence rolled back, and
removes active registrations.

The coordinator does not retry or compensate after an underlying commit has
partially succeeded. Such failures retain the current production boundary:
the Engine poisons the speculative runtime because publication is no longer
safe to retry.

## Sequence Release

`assert_sequence_releasable()` rejects:

- invalid sequence or epoch values;
- any active proposal transaction for the sequence;
- an epoch different from the proposal cache sequence state.

`release_sequence()` performs the same validation, calls
`ProposalKVCache.release_sequence()`, and records one release evidence row.

Qwen3.5 retains ownership of:

- pending target-prefill observations;
- bootstrap hidden-state execution;
- bootstrapped sequence metadata.

Its `release_sequence()` first asks the generic coordinator to validate
releasability, then validates Qwen3.5 pending/bootstrap epochs, then calls the
coordinator release and clears model-specific state. This ordering prevents
the generic cache from being released before a model-specific stale-epoch
error is detected.

## Qwen3.5 Integration

`Qwen35MTPProposalExecutor` keeps:

- target-prefill observation validation and accumulation;
- logits-free bootstrap;
- exact-Q grouping;
- eager and CUDA-graph proposal execution;
- tensor-parallel greedy token agreement;
- selected-token evidence;
- Qwen3.5 pending/bootstrap state.

It delegates:

- successful group registration;
- unregistered proposal cleanup;
- finalize prepare/commit/rollback;
- proposal transaction activity counts;
- proposal transaction authority rows;
- release guarding and proposal-cache release.

The public `ProposalExecutor` protocol and ModelRunner RPC surface do not
change.

## Error Semantics

- Model-specific proposal execution failure aborts transactions through the
  existing execution owner before registration.
- Registration validation failure aborts only new unregistered transactions.
- Finalize preparation failure rolls back prepared underlying tickets.
- Finalize commit failure propagates; Engine poisoning remains authoritative.
- Finalize rollback failure propagates; Engine poisoning remains
  authoritative.
- Sequence release with an active transaction or stale epoch fails closed.
- No error path mutates target-KV transactions, scheduler output, verifier
  acceptance, recurrent side-state selection, or target-KV residency.

## Testing

Add dependency-light coordinator tests covering:

1. mixed empty/non-empty registration with stable order;
2. duplicate sequence or transaction rejection;
3. missing, reserved, stale-sequence, and stale-epoch transaction rejection;
4. registration batch failure aborting only new unregistered transactions;
5. prepare finalize preserving accepted-token counts;
6. partial prepare failure rolling back prior tickets;
7. commit ordering and active-registration removal;
8. rollback reverse ordering and active-registration removal;
9. ticket single-consumption;
10. active-transaction and stale-epoch release rejection;
11. release of committed proposal slots;
12. tensor-free authority snapshot fields.

Update Qwen3.5 executor tests to prove:

- it constructs and uses the coordinator;
- eager and graph proposal rows still register exactly once;
- finalize and release behavior remains unchanged;
- authority snapshot counters and rows retain their current schema;
- proposal token outputs and transaction ownership are unchanged.

Run focused regressions for:

- proposal KV cache;
- generic ModelRunner proposal executor registry;
- Qwen3.5 MTP executor;
- Qwen3.5 ModelRunner integration;
- Engine speculative publication and rollback;
- native-MTP gate contracts.

## Non-Goals

This slice does not:

- add a second learned checkpoint;
- add a new model loader;
- change target-KV transaction behavior;
- add accepted-prefix replay or rematerialization;
- implement KV8/KV4;
- implement proposal-KV offload;
- implement MTP CUDA Graphs;
- change `MAX_PROPOSAL_TOKENS=4`;
- run a remote or GPU workload;
- claim parity, performance, movement, production readiness, or Phase 1
  completion.

## Completion Boundary

Passing this slice proves that proposal-KV lifecycle ownership is
model-independent and that the established Qwen3.5 native-MTP executor uses
the generic coordinator without behavior change.

It does not establish a second learned model structure. It removes a concrete
model-specific transaction duplication barrier so that the next learned
executor needs to implement only its model loading, bootstrap, and proposal
forward semantics.
