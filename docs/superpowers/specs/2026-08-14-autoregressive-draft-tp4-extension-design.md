# Autoregressive Draft TP4 Extension Design

**Date:** 2026-08-14

**Status:** Approved by user on 2026-08-14

**Extends:** `docs/superpowers/specs/2026-08-14-autoregressive-draft-model-executor-design.md`

## Objective

Extend the independent Qwen3 autoregressive draft-model executor from TP1 to
TP4 without changing the generic speculative runtime's verifier, accepted
prefix, target-KV transaction, side-state, or offload semantics.

The selected topology is:

```text
target rank 0  <->  Qwen3 draft shard 0  <->  proposal KV shard 0
target rank 1  <->  Qwen3 draft shard 1  <->  proposal KV shard 1
target rank 2  <->  Qwen3 draft shard 2  <->  proposal KV shard 2
target rank 3  <->  Qwen3 draft shard 3  <->  proposal KV shard 3
```

Each target rank privately loads the corresponding Qwen3 tensor-parallel
shard and owns rank-local proposal K/V. All ranks execute the same bootstrap,
proposal, finalize, rollback, and release state transitions. Only selected
token IDs, registration status, and normalized logical authority digests
cross ranks. Proposal K/V and physical proposal slot IDs never cross ranks.

This extension removes the current TP4 fail-closed restriction only after
the complete local TP4 contract test matrix passes. It does not by itself
establish real-checkpoint TP4 correctness or complete Phase 1.

## Existing Contract Preserved

The TP1 design remains authoritative for:

- proposal token zero being the target-produced first token;
- at most `Q - 1` learned draft forwards;
- `MAX_PROPOSAL_TOKENS=4`;
- proposal-side staged-entry count `Q - 1`;
- accepted proposal KV count `max(A - 1, 0)`;
- prompt bootstrap through the proposal-KV lifecycle;
- source-neutral executor registration;
- exact tokenizer and vocabulary compatibility;
- batch-native draft prefill and decode;
- target-hidden independence;
- tensor-free authority snapshots;
- target-only exact greedy output parity as the correctness oracle.

This extension must not change:

- verifier token selection;
- fallback token indexing;
- accepted-prefix semantics;
- target-KV prepare, commit, rollback, or release behavior;
- speculative side-state selection;
- target KV residency or offload counters;
- scheduler publication behavior;
- n-gram, SAM, or Qwen3.5 native-MTP behavior.

## Considered TP4 Approaches

### 1. Sharded Qwen3 Drafter on Every Target Rank

Selected.

Qwen3 already uses tensor-parallel vocabulary embeddings, QKV and MLP
projections, KV-head partitioning, and a parallel LM head. Constructing the
draft model inside each existing target rank therefore gives each rank its
normal Qwen3 shard without introducing a second distributed topology.

Advantages:

- draft weights remain TP-sharded;
- proposal K/V remains TP-sharded and rank-local;
- all ranks perform real draft forwards;
- only rank zero materializes full-vocabulary logits;
- the existing tensor-parallel greedy token selector broadcasts compact
  token IDs;
- memory accounting reflects one local draft shard and one local proposal
  KV shard per rank.

Costs:

- every proposal step includes one draft-model token broadcast;
- lifecycle consistency needs explicit cross-rank logical authority checks;
- registration must become all-rank and failure-atomic.

### 2. Replicated Drafter on Every Target Rank

Rejected.

Replication would simplify logits and local correctness, but would place a
complete Qwen3 model and complete proposal K/V on every rank. It would also
hide failures in the existing Qwen3 tensor-parallel path and make memory
evidence incomparable with the sharded target.

### 3. Rank-Zero-Only Drafter

Rejected.

Running the drafter only on rank zero and broadcasting proposal tokens would
avoid draft collectives, but it would not provide rank-local proposal K/V for
later draft continuation. It would also violate the requirement that every
rank execute the learned draft path and would not establish TP4 learned-draft
correctness.

## Rank-Local Ownership

Each rank owns:

- one Qwen3 draft model shard;
- one `Qwen3DraftPhysicalSlotStore`;
- one `ProposalKVCache`;
- one `Qwen3AutoregressiveDraftBackend`;
- one `AutoregressiveDraftProposalExecutor`;
- one `ProposalKVLifecycleCoordinator`;
- rank-local pending prompt and bootstrapped sequence state;
- rank-local registration candidate objects before publication.

The following values are logical and must agree across ranks:

- executor ID and capabilities;
- target and draft checkpoint composite hashes;
- target and draft tokenizer contract hashes;
- backend identity;
- sequence ID and sequence epoch;
- prompt token IDs and positions;
- `exact_q`;
- proposal token IDs;
- accepted proposal length;
- committed proposal length;
- lifecycle transition name and logical transaction state.

The following values are physical and are explicitly allowed to differ:

- proposal transaction ID strings;
- proposal physical slot IDs;
- K/V storage pointers;
- local K/V tensor shapes along sharded dimensions;
- allocator free-list order;
- rank-local timing values.

No cross-rank equality check may include a physical slot ID or storage
pointer. Those values are addresses in independent rank-local allocators.

## Tensor-Parallel Collective Boundary

Create:

```text
tinyvllm/engine/autoregressive_draft_tp.py
```

This module owns the independent drafter's small collective contract. It
does not own model execution, proposal allocation, or Engine publication.

It provides:

```python
@dataclass(frozen=True)
class AutoregressiveDraftRankRegistrationStatus:
    rank: int
    world_size: int
    success: bool
    stage: str
    error_type: str | None
    message: str | None
    target_checkpoint_sha256: str | None
    draft_checkpoint_sha256: str | None
    target_tokenizer_sha256: str | None
    draft_tokenizer_sha256: str | None
    backend_identity: str | None
    executor_id: str | None
    capabilities_sha256: str | None


class AutoregressiveDraftTensorParallelCoordinator:
    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: object,
        gather_registration_status=None,
        gather_digest=None,
    ): ...

    def collect_registration_status(
        self,
        status: AutoregressiveDraftRankRegistrationStatus,
    ) -> tuple[AutoregressiveDraftRankRegistrationStatus, ...]: ...

    def assert_logical_authority(
        self,
        *,
        stage: str,
        rows: object,
    ) -> str: ...

    def converge_stage(
        self,
        *,
        stage: str,
        rows: object,
        local_error: BaseException | None,
    ) -> str: ...
```

TP1 is a no-op topology: registration returns the one local status and
logical authority returns the local SHA-256 digest without a distributed
operation.

TP4 registration may use `torch.distributed.all_gather_object` because it
runs once during construction and must preserve structured error details.
Runtime authority does not gather Python objects. Each rank canonicalizes
the stage and logical rows, computes SHA-256, places the 32 digest bytes in a
contiguous rank-local `torch.uint8` tensor on the draft device, prefixes one
success byte, and performs one fixed-size all-gather. A success row carries
the logical payload digest. A failure row carries a digest of the stage,
exception type, and exception message while preserving the full local error
in rank-local authority. Any failure bit or unequal success digest raises the
same stage-attributed runtime error on every surviving rank.

Every fallible local lifecycle phase uses this sequence:

```text
try local phase
capture local error instead of returning early
converge_stage(stage, logical rows, local error)
if a peer failed, clean up still-owned local state
raise the common stage failure
```

This prevents a rank that completed local work from proceeding while a peer
returned early. It cannot recover a process death or a failure inside a
Qwen3 tensor-parallel collective; those are distributed process-group
failures and poison ModelRunner execution.

The canonical encoder accepts only:

- `None`;
- booleans;
- integers;
- finite floats;
- strings;
- tuples and lists;
- dictionaries with string keys sorted lexicographically;
- dataclasses recursively converted into the allowed values.

Tensors, sets, arbitrary object stringification, NaN, infinity, physical
slot IDs, transaction IDs, and storage pointers are rejected from logical
authority payloads.

## Draft Logits Contract

The existing backend protocol changes to:

```python
class AutoregressiveDraftBackend(Protocol):
    device: object
    backend_identity: str
    model_fingerprint: str
    tokenizer_fingerprint: str

    def prefill_batch(
        self,
        rows: tuple[AutoregressiveDraftPrefillRow, ...],
    ) -> None: ...

    def decode_step_batch(
        self,
        rows: tuple[AutoregressiveDraftDecodeRow, ...],
    ) -> tuple[object, ...] | None: ...
```

The return contract is topology-aware:

- TP1 rank zero returns one rank-one full-vocabulary logit tensor per row;
- TP4 rank zero returns one rank-one full-vocabulary logit tensor per row;
- TP4 non-root ranks return `None`;
- no non-root rank may return local-vocabulary logits;
- rank zero rows must have equal vocabulary width, matching dtype/device,
  finite values, and the exact input batch order.

`Qwen3ForCausalLM.compute_logits()` already delegates to
`ParallelLMHead`. Under TP4, `ParallelLMHead` gathers vocabulary shards to
rank zero and returns `None` on non-root ranks. The Qwen3 draft backend must
accept this existing behavior instead of requiring a tensor on every rank.

The executor passes:

```python
root_logits_or_none
```

to:

```python
select_tensor_parallel_greedy_tokens(
    logits,
    rank=tensor_parallel_rank,
    world_size=tensor_parallel_size,
    batch_size=batch_size,
    device=backend.device,
)
```

Rank zero computes exact `argmax`; all ranks receive one contiguous
`torch.int64[batch_size]` token tensor from rank zero. Full logits are not
broadcast.

## Bootstrap Data Flow

All ranks receive the same target prefill observation through the existing
ModelRunner command fanout.

Before allocating proposal slots, the executor builds logical bootstrap
rows containing:

```text
sequence_id
sequence_epoch
prompt_token_ids
prompt_positions
final_chunk_seen
```

The tensor-parallel coordinator compares one `bootstrap_preflight` digest.
Only after equality succeeds may any rank call `ProposalKVCache.begin()`.

Each rank then:

1. reserves rank-local prompt slots;
2. performs one batch-native Qwen3 draft prefill using its local model shard;
3. materializes local K/V for every Qwen3 layer;
4. prepares the existing bootstrap finalize encoding;
5. builds a logical `bootstrap_prepared` row with prompt count and state;
6. compares the logical digest;
7. commits all rank-local prompt slots.

Every fallible local phase converges success or failure before the next
phase. If local work or a peer rank fails before commit, each successful rank
aborts every still-owned bootstrap transaction in reverse order. A rank must
not return to normal proposal execution after a peer reports bootstrap
failure.

A post-commit `bootstrap_committed` digest records sequence, epoch, prompt
count, and committed logical length. A mismatch after commit poisons the
speculative runtime because the committed rank-local caches no longer have
a safe rollback contract.

## Proposal Data Flow

`ModelRunner.run_spec_first_target_and_proposal_batch()` already runs on all
TP ranks. The target LM head produces full logits only on rank zero, and the
existing target token selector broadcasts the first target token to every
rank. Each rank therefore constructs the same
`ModelRunnerProposalInput`.

For each exact-Q group:

1. Build `proposal_preflight` logical rows:

   ```text
   batch_index
   sequence_id
   sequence_epoch
   context_token_count
   exact_q
   first_target_token
   ```

2. Compare the preflight digest before proposal slot allocation.
3. Begin one rank-local transaction with `exact_q - 1` staged entries.
4. For every learned draft step:
   - execute the same Qwen3 shard forward on every rank;
   - require root full logits and non-root `None`;
   - broadcast selected token IDs with the existing greedy selector;
   - append the same token IDs on every rank.
5. Mark all local staged entries materialized.
6. Build `proposal_materialized` rows:

   ```text
   batch_index
   sequence_id
   sequence_epoch
   exact_q
   proposal_token_ids
   staged_entry_count
   logical_state="materialized"
   ```

7. Compare the materialized digest.
8. Register the local transaction with the local lifecycle coordinator.

Local input validation converges before entering the Qwen3 forward so all
ranks either call the same model collective sequence or call none of it.
After the forward and token selection, each rank converges its local outcome
before the next step. If a local forward, logits validation, token broadcast,
peer outcome, or materialized digest check fails, every still-owned new local
transaction is aborted in reverse order. No proposal result is published by
rank zero.

The executor continues to return proposal rows on every rank because the
registry validates them locally. `ModelRunner` continues to return the
fused proposal result only on rank zero. No Engine or Scheduler return type
changes.

## Finalize, Rollback, and Release

All lifecycle commands continue to fan out to every ModelRunner rank.

### Prepare Finalize

Before local prepare, compare `finalize_preflight` rows containing:

```text
batch_index
sequence_id
sequence_epoch
exact_q
proposal_token_ids
accepted_proposal_tokens
committed_proposal_entries=max(accepted_proposal_tokens - 1, 0)
```

After every local coordinator has prepared its ticket, compare
`finalize_prepared` rows containing the same logical fields plus
`logical_state="prepared"`. A mismatch or peer failure rolls back every
local prepared ticket and aborts any still-owned local transaction. Local
prepare errors are captured and converged before any rank returns.

### Commit Finalize

Commit uses the unchanged lifecycle coordinator. After local commit, compare
`finalize_committed` rows containing sequence, epoch, accepted length,
committed logical length, and `logical_state="committed"`.

A local commit failure or post-commit digest mismatch poisons the
speculative runtime through the existing Engine boundary. The extension
does not invent cross-rank partial commit recovery.

### Rollback Finalize

Rollback uses the unchanged local coordinator. After rollback, compare
`finalize_rolled_back` rows containing sequence, epoch, accepted length,
retained committed logical length, and `logical_state="rolled_back"`.
Rejected suffix slots are released locally; their physical IDs are not
compared. A peer rollback failure poisons the speculative runtime because
some rank-local lifecycle state may already have been released.

### Sequence Release

Before release, compare `release_preflight` rows containing sequence ID,
epoch, and committed logical length. Every rank then releases its local
proposal cache and pending draft state. A `release_complete` digest requires
zero active transactions, zero prepared tickets, and zero committed logical
entries for the released sequence on every rank. A peer release failure is a
poisoned runtime boundary rather than a partially reusable sequence.

## Failure-Atomic All-Rank Registration

Current registration builds and immediately publishes one TP1 executor. TP4
requires private construction followed by all-rank consensus.

Add to:

```text
tinyvllm/engine/autoregressive_draft_registration.py
```

```python
@dataclass(frozen=True)
class AutoregressiveDraftRegistrationCandidate:
    target_checkpoint: CheckpointFingerprint
    draft_checkpoint: CheckpointFingerprint
    target_tokenizer_contract: TokenizerContract
    draft_tokenizer_contract: TokenizerContract
    model: object
    physical_store: object
    proposal_kv_cache: object
    backend: object
    executor: object
    descriptor: object
```

`ModelRunner._maybe_register_autoregressive_draft_executor()` performs:

1. validate topology is exactly TP1 or TP4;
2. build every checkpoint and tokenizer identity locally;
3. load the rank-local Qwen3 draft shard;
4. construct the rank-local physical store, cache, backend, executor, and
   descriptor without publishing them;
5. preflight the local registry entry, descriptor ID, capabilities, and
   required lifecycle methods without mutation;
6. convert local success or failure into
   `AutoregressiveDraftRankRegistrationStatus`;
7. collect exactly `world_size` statuses;
8. require ranks `0..world_size-1` exactly once;
9. require every rank to report success;
10. require identical checkpoint, tokenizer, backend, executor ID, and
    capability identities;
11. publish the already validated local executor on every rank;
12. expose the candidate's objects on the local ModelRunner only after
    publication succeeds.

The proposal executor registry currently has no unregister operation.
Therefore every predictable failure must occur before step 11. Add a
read-only registry preflight method that performs the same validation as
`register()` without mutating `_entries`. `register()` reuses the same
validation helper so preflight and publication cannot drift.

If any rank reports a private construction failure:

- no rank calls registry `register()`;
- every rank stores an all-rank registration error with the failing rank and
  stage;
- successful private candidates are discarded;
- no rank exposes `autoregressive_draft_model`,
  `autoregressive_draft_executor`, descriptor, physical store, or
  checkpoint/tokenizer publication fields;
- the target ModelRunner remains usable without the autoregressive draft
  executor only if the existing Engine configuration treats missing
  registration as a closed speculative feature.

An unexpected process death or collective failure during publication is a
distributed ModelRunner initialization failure. It is not recovered through
registry rollback.

## Configuration

The source-neutral configuration fields remain:

```text
autoregressive_draft_enabled
autoregressive_draft_model
autoregressive_draft_backend
autoregressive_draft_max_proposal_tokens
autoregressive_draft_gpu_slot_capacity
```

No replicated/sharded mode flag is added. This extension defines the only
supported TP4 mode as sharded.

Validation permits:

```text
tensor_parallel_size in {1, 4}
autoregressive_draft_max_proposal_tokens in [1, 4]
autoregressive_draft_backend == "qwen3"
```

Any other TP size remains fail-closed. A TP4 configuration must also satisfy
the existing Qwen3 divisibility rules for vocabulary, query heads, KV heads,
and tensor-parallel linear layers. Rank-local physical store geometry is
derived from the instantiated sharded Qwen3 attention modules, never from a
global KV-head count copied to every rank.

## Authority Evidence

Extend each rank's autoregressive draft authority snapshot with:

```text
rank
world_size
registration_consensus_sha256
logical_authority_rows
logical_authority_digest_count
last_logical_authority_sha256
local_model_parameter_bytes
local_proposal_kv_bytes
local_query_heads
local_kv_heads
local_prefill_forward_count
local_decode_forward_count
```

The snapshot retains rank-local physical store evidence, including local
slot counts and storage identity. Aggregation code must interpret those as
rank-local evidence and must not require equal storage pointers or slot IDs.

The TP4 gate requires an all-rank aggregate containing exactly four rank
snapshots. It proves that:

- all four ranks registered the same logical executor identity;
- all four ranks executed real Qwen3 prefill and decode forwards;
- selected proposal tokens and logical lifecycle digests agreed;
- every rank used a distinct rank-local proposal K/V store;
- total proposal KV bytes are the sum of rank-local physical stores;
- zero live proposal slots remain after sequence release.

Timing remains rank-local. Steady-state draft latency is the maximum rank
latency for the measured stage, not the sum or rank-zero-only value.

## Local TDD Matrix

The implementation plan must add focused tests before removing either
current fail-closed message:

```text
autoregressive draft executor currently requires TP1
autoregressive draft currently requires TP1
```

Required local tests:

1. TP4 executor construction accepts ranks 0 through 3 and rejects every
   other topology.
2. TP1 behavior and all existing TP1 tests remain unchanged.
3. TP4 rank zero accepts full logits and broadcasts exact greedy token IDs.
4. TP4 non-root requires backend logits to be `None`.
5. Non-root local-vocabulary logits fail closed.
6. Malformed root logits abort every new rank-local transaction.
7. A broadcast failure aborts every new rank-local transaction.
8. Q=1 through Q=4 preserve staged-entry count `Q - 1`.
9. Mixed exact-Q batches preserve input order on all ranks.
10. Every rank executes one real backend decode call per active proposal
    step; fixture token broadcast cannot satisfy the counter.
11. Logical authority digest ignores physical transaction and slot IDs.
12. Logical authority digest rejects tensors, storage pointers, NaN, and
    unsupported objects.
13. A preflight digest mismatch prevents slot allocation.
14. A materialized digest mismatch aborts staged slots before registration.
15. Bootstrap prepare failure rolls back all local bootstrap candidates.
16. Finalize prepare mismatch rolls back every prepared local ticket.
17. One-rank local failure is converged so successful peers clean up and
    raise instead of continuing.
18. Commit failure is reported as a poisoned runtime boundary.
19. Release requires zero active transactions and leaves zero live local
    slots on every rank.
20. Registration privately constructs all four candidates before any
    registry mutation.
21. One-rank checkpoint, tokenizer, model load, store, backend, executor, or
    descriptor failure publishes on no rank.
22. One-rank identity mismatch publishes on no rank.
23. Registry preflight rejects duplicate IDs and invalid capabilities
    without mutation.
24. Registry publication occurs only after four successful matching
    statuses.
25. Qwen3 backend returns root full logits and non-root `None` under mocked
    TP4 LM-head behavior.
26. Rank-local K/V geometry uses instantiated local KV heads.
27. Proposal physical slot IDs may differ across ranks without authority
    failure.
28. Checkpoint and tokenizer consensus appears in every rank snapshot.
29. ModelRunner still returns fused proposal rows only on rank zero.
30. Generic Engine, Scheduler, verifier, target-KV, side-state, and offload
    code contains no Qwen3-specific branch.
31. Qwen3.5 native MTP, n-gram, and SAM regression tests remain green.

Tests use injected collectives and fake rank-local backends. They must not
initialize a real four-process NCCL group as part of the dependency-light
local suite.

## Real TP4 Checkpoint Gate

Local tests establish contracts only. They do not establish TP4 learned
draft correctness.

The first authorized real gate requires:

- source-attributed immutable Qwen3 draft checkpoint hashes;
- source-attributed immutable Qwen3.5 target checkpoint hashes;
- exact tokenizer and ordered token-to-ID compatibility;
- TP4 BF16;
- greedy temperature zero;
- `MAX_PROPOSAL_TOKENS=4`;
- 4K context;
- batch 1, batch 4, and a true multi-sequence case;
- target-only and learned-draft Engine runs over identical prompts;
- exact output-token parity for every sequence;
- all-rank registration consensus;
- all-rank real draft prefill and decode forward counts;
- all-rank logical proposal and lifecycle digest agreement;
- accepted-prefix proposal KV commit evidence;
- rejected-suffix rollback evidence;
- zero proposal KV leaks after release;
- no extra target forward beyond the generic speculative contract;
- separate target KV and proposal KV byte accounting;
- no claim that simulated copy counters represent real H2D movement.

The gate fails if any non-root rank has zero real draft forwards, if proposal
tokens come from fixtures, if only rank-zero authority is present, or if
physical proposal K/V is gathered across ranks.

After the 4K TP4 gate is established, 16K and 32K learned-draft campaigns
remain separate promotion work. Existing Qwen3.5 native-MTP TP4 evidence
cannot substitute for independent Qwen3 learned-draft evidence.

## Implementation Scope

Expected files:

```text
Create:
  tinyvllm/engine/autoregressive_draft_tp.py
  tools/test_autoregressive_draft_tp.py

Modify:
  tinyvllm/config.py
  tinyvllm/engine/autoregressive_draft_executor.py
  tinyvllm/engine/autoregressive_draft_registration.py
  tinyvllm/engine/model_runner.py
  tinyvllm/engine/qwen3_draft_backend.py
  tinyvllm/engine/speculative_proposal_executor.py
  tools/test_autoregressive_draft_executor.py
  tools/test_autoregressive_draft_model_runner_integration.py
  tools/test_autoregressive_draft_registration.py
  tools/test_qwen3_draft_backend.py
  tools/test_autoregressive_draft_tp1_engine_gate.py
```

The implementation plan may add a dedicated TP4 Engine gate and verifier,
but it must not modify existing real-gate artifacts or run a remote/GPU
campaign without separate authorization.

## Promotion Boundary

Passing the local implementation matrix proves:

- the independent learned-draft runtime has a TP4-capable sharded design;
- all-rank private construction and registration are failure-atomic for
  predictable errors;
- rank zero alone materializes full draft logits;
- all ranks execute synchronized proposal and proposal-KV lifecycle logic;
- physical proposal K/V remains rank-local.

It does not prove:

- a real second learned structure under TP1 or TP4;
- real TP4 Engine parity;
- 4K, 16K, or 32K learned-draft correctness;
- TPOT, TTFT, throughput, acceptance, memory, or H2D improvement;
- KV8 or KV4;
- production readiness;
- Phase 1 completion.

The independent Qwen3 draft path remains `NOT_PROMOTABLE` until the
source-attributed real-checkpoint gates pass. The full Phase 1 objective
remains `NOT_ACHIEVED` until the complete two-structure, TP1/TP4,
4K/16K/32K, batch 1/4/multi-sequence, exact greedy parity, performance, and
real KV movement matrix is established.
