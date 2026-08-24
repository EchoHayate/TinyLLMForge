# Exact-Burst Generation-Sealed Lease Identity Design

**Date:** 2026-08-24

**Status:** Approved under the standing autonomous-optimization authorization

**Stage-1 model:** Qwen3-0.6B

**Primary target:** remove repeated context-length-scaled block-identity work
from consecutive exact-greedy bursts without weakening stale-lease detection

## Objective

Replace repeated full block-table identity construction and validation with a
constant-time generation seal for an already-stable sequence layout.

The optimization must not change:

- model weights, attention, logits, argmax, or output tokens;
- CUDA Graph capture or replay count;
- token or sampled-logit D2H count and bytes;
- KV physical slots or prefix-cache publication;
- scheduler queue ordering or request fairness;
- rollback behavior; or
- the requirement that stale ownership fails closed before host metadata is
  committed.

This is a runtime-data-flow-specific original engineering design, not a claim
of academic novelty.

## Motivation and Measured Ceiling

An exact-burst lease currently embeds every `(block_id, generation)` pair in
the sequence block table. The scheduler reconstructs that tuple when granting
the lease and scans it again during commit validation and rollback checks.
The work is correct, but scales with context length even when the sequence KV
layout has not changed between consecutive K8 bursts.

A local read-only ceiling probe on the production lease-local transaction
path replaced only repeated identity validation with a no-op:

| Sequence length | Current median | Validation-free median | Ceiling improvement |
| ---: | ---: | ---: | ---: |
| 249 | 29.250 us | 27.291 us | 6.70% |
| 2,041 | 35.667 us | 28.750 us | 19.39% |
| 8,185 | 54.000 us | 29.625 us | 45.14% |

The same probe measured full identity construction and validation directly:

| Sequence length | Blocks | Build median | Validate median |
| ---: | ---: | ---: | ---: |
| 249 | 1 | 0.458 us | 0.709 us |
| 2,041 | 8 | 1.583 us | 2.583 us |
| 8,185 | 32 | 4.834 us | 8.042 us |

These are CPU microbenchmarks, not end-to-end performance evidence. They show
that the remaining transaction path contains a measurable context-scaled
component and define an upper bound for a focused implementation.

## Considered Approaches

### A. Generation-sealed block-table identity

Track mutation generations for the sequence block table and BlockManager
ownership state. Capture a validated immutable identity receipt only when
either generation changes. Consecutive leases reuse the receipt and validate
the two generations in constant time.

Benefits:

- retains explicit stale-layout and stale-ownership rejection;
- removes repeated tuple construction, row validation, and identity payload
  serialization on stable layouts;
- does not alter GPU execution or model outputs;
- naturally invalidates on allocation, release, restore, or table mutation;
- can be reused by other scheduler leases after a second caller proves the
  contract.

Costs:

- introduces a mutation-tracked list for `Sequence.block_table`;
- requires complete coverage of list mutation methods and serialization;
- adds a BlockManager ownership generation;
- changes the exact-burst lease identity schema; and
- retains an O(blocks) cold capture whenever layout or ownership changes.

This is the selected approach.

### B. Trust the pending lease and skip repeated validation

Use the pending-lease pointer and schedule generation as proof that the block
table could not have changed.

Benefits:

- smallest code change;
- maximal hot-path reduction.

Costs:

- does not detect direct list mutation;
- does not detect ownership changes outside the scheduler's expected path;
- weakens the current fail-closed contract; and
- converts an implementation assumption into correctness authority.

This approach is rejected.

### C. Recompute a compact hash on every validation

Store only a digest in the lease and hash the current table on every check.

Benefits:

- compact serialized lease;
- detects arbitrary table changes.

Costs:

- hashing remains O(blocks);
- Python iteration and serialization remain on every burst;
- it changes representation without removing the main scaling cost.

This approach is rejected.

## Capability and Layer Boundary

Without model-specific nouns, the capability is:

> A scheduler-owned mutable resource layout exposes a monotonically versioned
> identity receipt. A lease captures that receipt once and may use constant-
> time validation while both the layout generation and resource-ownership
> generation remain unchanged.

Layer assignment:

- **Mechanism:** mutation-tracked block table, ownership generation, immutable
  identity receipt, capture/cache/validate operations.
- **Adapter:** exact-greedy lease construction and commit consume the receipt.
- **Policy/config:** a default-off exact-burst flag selects the new identity
  path for the first adopter.
- **Benchmark/profile:** Qwen3-0.6B, TP1, batch 1, K8, fixed 2K/4K/8K contexts,
  repeated paired runs, exact correctness oracle.

Two-axis genericity verdict before implementation:

- mechanism: `reusable candidate`;
- integration: `first adopter only`.

The implementation must not claim a generic runtime facility until a second
synthetic or production lease caller proves the same receipt contract.

## Architecture

### 1. Mutation-tracked block table

Introduce a list-compatible block-table container owned by `Sequence`. It
preserves existing indexing, iteration, equality, slicing, append, extend,
assignment, deletion, clear, insert, pop, remove, in-place addition, reverse,
and sort behavior.

Every mutating operation first increments a non-negative, read-only
`revision`, then delegates to the underlying list operation. Advancing before
the operation is deliberately conservative: a failed or partially applied
list mutation still invalidates every prior seal. Replacing the whole table
through `Sequence.block_table = value` creates a fresh tracked table with a
strictly newer revision. The revision can only be initialized by the
container constructor and restored by Sequence's validated deserialization
path; ordinary callers cannot assign it.

Python augmented assignment on a property performs both the in-place
operation and a subsequent setter call. Therefore the setter must treat
assignment of the already-owned tracked table as an identity-preserving
no-op:

```text
sequence.block_table += values
sequence.block_table *= count
```

The in-place list operation advances the revision exactly once; the property
setter must neither replace the table nor advance it a second time. Assignment
of any different object remains whole-table replacement and advances from the
current owned revision, regardless of the incoming object's revision.

Serialization carries:

```text
plain-list block IDs
block-table revision
```

Deserialization remains backward compatible with older Sequence state tuples;
an old state receives revision zero after its block IDs are restored. The new
17-field state places the revision at index 15 and keeps the prompt token list
or decode last-token payload as the final element, preserving the existing
`state[-1]` payload convention. The serialized block-table field remains a
plain list rather than pickling the tracked-list implementation itself, so the
wire contract does not depend on list-subclass pickle behavior.

### 2. BlockManager ownership generation

Add a monotonically increasing `ownership_generation` to `BlockManager`.
Increment it whenever block ownership identity can change:

- allocating a free block;
- activating an idle cached block;
- decrementing the final reference and deallocating a block;
- restoring allocation/refcount/generation state during rollback; or
- any other path that directly changes `used_block_ids`, `free_block_ids`,
  `ref_count`, or block `generation`.

Hash publication alone does not change ownership identity and therefore does
not increment this generation.

The counter is an invalidation epoch, not a transaction counter. One logical
operation may advance it more than once if it performs multiple ownership
mutations. A rollback or partial mutation must also advance it, even when the
visible state is restored, so a seal captured before the attempted mutation
can never become valid again accidentally.

`Block.generation` is exposed through a manager-bound tracked property. Any
post-construction assignment that changes or restores a block generation
advances the owning manager's epoch. This closes direct rollback and
fault-injection paths that would otherwise bypass method-level accounting.
Initial block construction occurs before the callback is bound and does not
advance the epoch.

The counter is process-local scheduler authority. It is not a wall clock and
is never compared across ranks.

### 3. Immutable identity receipt

Add an immutable receipt containing:

```text
sequence ID
block-table revision
BlockManager ownership generation
block count
write-block index
write-block ID
write-block generation
predecessor block ID/generation when present
identity digest
```

On a cold capture, BlockManager performs the existing full ID/generation
validation, constructs the complete immutable identity once, and derives the
digest. The sequence caches the receipt together with the complete identity
for diagnostics and fallback validation.

On a hot capture, the cached receipt is reused only if:

```text
same sequence object and sequence ID
same tracked block-table object
same block-table revision
same BlockManager ownership generation
same block count
same write-block index/ID/generation
same predecessor ID/generation when present
```

Any mismatch invalidates the cache and performs a cold full capture. A failure
during the cold capture rejects the lease.

### 4. Constant-time lease validation

The exact-burst lease stores the immutable receipt rather than serializing the
full identity rows into every new lease hash.

Before commit, validation compares:

- pending lease object and lease identity;
- schedule and graph generations;
- sequence ID, length, and completion count;
- current block-table revision;
- current BlockManager ownership generation;
- block count;
- write-block ID/generation; and
- predecessor ID/generation when present.

If all values match, validation is O(1). If either generation differs, the
optimized transaction is rejected before mutation. It does not silently
refresh a stale lease.

The complete cached identity remains available for diagnostics and explicit
fallback validation, but is not scanned on the hot path.

### 5. Continuation receipt compatibility

The CUDA-graph continuation receipt currently compares the lease's complete
block-table identity across consecutive bursts. A sealed lease cannot simply
replace that tuple with an empty value: doing so would either make the receipt
invalid or collapse distinct layouts onto the same empty identity.

Both the lease and continuation receipt therefore carry exactly one identity
mode:

```text
full identity tuple, with no seal
generation seal, with an empty full identity tuple
```

Continuation matching compares the active identity representation. In sealed
mode it compares the immutable seal, including its table and ownership
generations; in baseline mode it retains the existing full-tuple comparison.
The baseline canonical lease payload and digest remain unchanged when the new
flag is disabled. A generation change causes a continuation miss and a
fail-closed scheduler validation, never a false continuation hit.

### 6. Failure and rollback semantics

The feature is fail closed:

- an untracked or malformed block table disables the optimized path;
- revision overflow is treated as fatal rather than wrapped;
- stale table or ownership generations reject before mutation;
- any mutation after journal capture forces rollback failure if the journal
  can no longer prove its bounded state;
- generic journal fallback remains available before mutation; and
- rollback failure remains terminal through
  `SchedulerPostprocessRollbackError`.

The feature remains default-disabled until its dedicated gate passes.

## Stage-1 Scope

The first adopter is limited to the existing eligible one-phase exact-greedy
K8 lease-local journal:

```text
TP1, rank 0
batch size 1
completion-only decode
temperature == 0
ignore_eos == true
non-terminal K8
one physical write block
no speculative path
no mixed batch
```

All existing ineligible cases retain their current behavior. The change does
not expand exact-burst coverage and does not alter the selected burst width.

## Verification

### Unit and fault-injection coverage

Require tests for:

- every supported block-table mutation increments revision exactly once;
- direct property augmented assignment (`sequence.block_table += ...` and
  `sequence.block_table *= ...`) preserves the tracked-table object and
  increments revision exactly once;
- whole-table replacement increments revision;
- Sequence serialization round-trips IDs and revision;
- legacy Sequence states remain readable;
- every BlockManager ownership transition increments its generation;
- hash publication alone does not increment ownership generation;
- stable consecutive captures reuse the same immutable receipt;
- table mutation invalidates the cache;
- allocation, activation, release, and rollback restoration invalidate it;
- direct write-block or interior-block generation assignment advances the
  ownership epoch and invalidates it;
- stale revisions reject exact-burst commit before token mutation;
- write-block and predecessor drift reject;
- baseline and sealed continuation receipts compare the correct identity
  representation, and generation drift cannot produce a continuation hit;
- successful commit and injected rollback preserve existing semantics; and
- the legacy full-identity path remains unchanged when the feature is off.

### CPU profile

Use production entrypoints at sequence lengths 249, 2,041, and 8,185.
Compare:

```text
full_identity
generation_sealed
```

Report:

- lease-grant median/P95;
- prepare+commit median/P95;
- identity rows visited;
- cold captures and hot reuses;
- Python allocation bytes; and
- fallback/rollback counts.

Required CPU result:

```text
8K lease lifecycle median improvement >= 30%
8K lease lifecycle P95 improvement >= 25%
aggregate median improvement >= 20%
candidate hot-path identity rows visited == 0
exactly one cold capture per stable fixture
```

### Source-bound GPU paired gate

Run Qwen3-0.6B on one strict-clean A100 with identical model, prompts, K8
graph, output length, and policy order. The only variable is the identity
mode.

Use 10 paired repetitions for each of 2K, 4K, and 8K contexts. Require:

```text
exact token/logit/argmax parity
unchanged target forwards, graph replays, and D2H
candidate stale/fallback/rollback count == 0
candidate hot identity reuse count == eligible bursts minus cold captures
8K scheduler lifecycle median/P95 improvement >= 25%
aggregate scheduler lifecycle median/P95 improvement >= 15%
aggregate TPOT median/P95 improvement >= 0.5%
TTFT/E2E/TPOT-P99/throughput regression <= 2%
allocated/reserved memory regression <= 1%
```

Both producer and independent verifier must reconstruct the same result from
60 performance rows and 24 correctness rows. Partial evidence cannot produce
a promotion result.

## Benefit and Cost Contract

A successful result may claim only:

> Generation-sealed identity removes repeated context-scaled scheduler
> identity work for stable one-phase K8 leases on the tested Qwen3-0.6B TP1
> workload.

It must report:

- CPU and GPU scheduler-lifecycle benefit;
- TPOT, TTFT, E2E, throughput, and P99 effect;
- extra Sequence and BlockManager metadata;
- cold-capture cost at layout changes;
- implementation and serialization complexity; and
- the first-adopter-only genericity boundary.

It must not claim Qwen3-8B, TP greater than one, multi-sequence, EOS-aware,
production-default, or academic novelty without separate evidence.
