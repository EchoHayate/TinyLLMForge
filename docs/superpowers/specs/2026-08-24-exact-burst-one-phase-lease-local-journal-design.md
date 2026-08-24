# Exact-Burst One-Phase Lease-Local Journal Design

**Date:** 2026-08-24

**Status:** Approved under the standing autonomous-optimization authorization

**Stage-1 model:** Qwen3-0.6B

**Primary target:** remove context-length-scaled scheduler transaction work
from the winning one-phase exact-greedy K8 path

## Objective

Reduce host-side prepare and commit latency for non-terminal one-phase K8
exact bursts without changing:

- model weights or numerical kernels;
- CUDA Graph capture or replay;
- target-model forward count;
- sampled logits, argmax decisions, or output tokens;
- D2H call count or byte count;
- KV write positions or physical block ownership; or
- the exact-burst lease and generation-validation contract.

The current one-phase K8 path commits eight already-authorized tokens
atomically. Before doing so, it still captures the generic scheduler rollback
journal. That journal copies the complete token list, snapshots scheduler
queues and every touched block, and reconstructs prefix-hash publication state
for the complete materialized block-table prefix.

The exact-burst lease has already frozen the sequence, schedule generation,
block-table identity, writable block, writable-block generation, and all eight
write positions. For an eligible non-terminal commit, the mutable state is
therefore bounded to one token-list suffix, one sequence's progress/SLO state,
and at most one newly full write block.

The selected optimization generalizes the existing split-phase lease-local
delta journal to the one-phase K8 path. It replaces only host scheduler
transaction bookkeeping. This is a runtime-data-flow-specific original
engineering design, not a claim of academic novelty.

## Evidence and Motivation

The established one-phase K8 exact-greedy path is the performance baseline to
preserve. Its source-bound Stage-1 gate reported:

```text
aggregate median TPOT improvement versus host greedy: 29.361129%
aggregate P95 TPOT improvement versus host greedy:    33.916056%
throughput improvement versus host greedy:            39.198420%
```

The later split-phase K8 design remained exact but regressed relative to this
one-phase path:

```text
aggregate median TPOT regression:  8.09%
aggregate P95 TPOT regression:    42.30%
aggregate P99 TPOT regression:    54.79%
```

The one-phase path must therefore remain the main K8 execution shape. This
optimization cannot depend on splitting the burst or adding another GPU/host
rendezvous.

The existing lease-local journal CPU profile isolates the context-scaled
generic snapshot cost:

```text
sequence length       generic median     lease-local median
249                    125.250 us          86.354 us
2041                   282.104 us          98.792 us
8185                   866.188 us         116.354 us
```

At sequence length 8,185, the generic path also issued 6,200 hash calls across
the measured samples, while the lease-local path issued zero for a partial
write block. These measurements were collected for split phases, so they do
not establish a one-phase end-to-end gain. They do establish that the generic
journal performs context-length-scaled work that the lease-local transaction
can avoid.

The preceding medium-context split-K candidate is not reused. It changed the
captured numerical execution shape and produced exact-token divergence. This
design deliberately leaves the numerical path untouched.

## Considered Approaches

### A. Generalize the lease-local delta journal to one-phase K8

Use the existing one-sequence delta journal and single-write-block publication
plan for eligible one-phase K8 rows.

Benefits:

- covers the established winning K8 path;
- removes full token-list copies;
- removes generic full-block journal capture and prefix-hash reconstruction;
- replaces `publish_full_blocks()` with at most one planned publication;
- keeps rollback and validation inside existing scheduler transaction
  boundaries; and
- has no model-numerical or GPU-execution change.

Costs:

- broadens the delta journal's eligibility predicate;
- requires explicit terminal and publication-authority fallbacks;
- adds one-phase lifecycle observability;
- requires fault-injection parity for an eight-token atomic commit; and
- retains full lease block-identity validation, which still scales with block
  count.

This is the selected approach.

### B. Pinned host token mailbox for the final eight-token transfer

Reuse a fixed pinned host buffer instead of allowing the final token tensor
conversion to allocate a new Python list.

Benefits:

- small implementation surface;
- no scheduler transaction changes; and
- no model-numerical change.

Costs:

- only eight `int64` values are transferred;
- the implicit allocation is unlikely to dominate long-context scheduler
  latency;
- ownership across graph generations and rollback would still need a contract;
  and
- expected end-to-end benefit may be below measurement noise.

This remains a later micro-optimization if profiling proves the transfer
material.

### C. Dirty-range block-table rebinding

Update only the changed block-table tail at cold graph-bind boundaries.

Benefits:

- can reduce host-to-device metadata traffic at block transitions; and
- does not change model arithmetic.

Costs:

- continuation epochs already skip most repeated binds;
- the optimization only applies at block-table changes;
- it does not remove per-burst generic scheduler snapshots; and
- coverage is lower than the selected one-phase commit optimization.

This is deferred.

## Fixed Stage-1 Scope

The fast path is eligible only when all of the following are true:

```text
feature flag enabled
TP1, rank 0
batch size 1
completion-only decode
temperature == 0
ignore_eos == true
one output row
row.exact_burst == true
row.exact_burst_phase is None
row.exact_burst_gate_only == false
active lease authorized_token_count == 8
len(row.output_tokens) == 8
sequence.status == RUNNING
batch_kind is None
is_prefill == false
do_sample == true
all eight writes remain in the lease write block
the request remains non-terminal after all eight tokens
the write block is unpublished at prepare time
```

The implementation remains default-disabled and reuses:

```text
Config.exact_greedy_decode_burst_lease_local_delta_journal
```

The flag currently requires exact burst, split-phase support, and K8. The
configuration contract must be generalized to require only exact burst and
K8. Split-phase support remains optional: when enabled, its existing
prefix/suffix rows remain eligible; when disabled, one-phase K8 can use the
same journal. This avoids adding another public option while allowing the
hardware gate to keep split execution disabled in both arms.

The fast path explicitly excludes:

- terminal K8 commits;
- one-phase ragged widths K2 through K7;
- split prefix/suffix behavior beyond the already implemented path;
- ordinary decode;
- prefill and mixed batches;
- speculative decoding;
- EOS-sensitive requests;
- TP greater than one;
- gate-only rows;
- requests whose writes cross a physical block boundary;
- stale or ambiguous predecessor-hash authority; and
- any transaction whose mutation surface is not bounded by one sequence and
  one write block.

Any uncertainty falls back to the generic journal before mutation.

## Architecture

### 1. Generalized journal identity

Rename the internal journal from:

```text
ExactBurstPhaseDeltaJournal
```

to:

```text
ExactBurstLeaseLocalDeltaJournal
```

and rename the selector from:

```text
_select_exact_burst_phase_journal()
```

to:

```text
_select_exact_burst_lease_local_journal()
```

The journal remains private to `scheduler.py`. No serialized or public API
depends on the old name.

The generalized journal serves:

- existing eligible split prefix rows;
- existing eligible split suffix rows; and
- new eligible one-phase K8 rows.

It does not become a generic scheduler transaction framework.

### 2. Eligibility and validation order

`prepare_postprocess()` first performs all existing structural validation:

- row/sequence identity and ordering;
- exact-burst single-row shape;
- decode and sampling mode;
- sequence running state;
- token type and output-budget checks;
- active lease presence;
- lease sequence identity;
- schedule generation;
- complete block-table ID and generation identity;
- write-block ID and generation;
- write positions and authorized width; and
- exact-burst result shape requirements.

Only after those checks pass may the selector attempt the lease-local path.

For a one-phase row, the selector additionally verifies:

```text
row.exact_burst_phase is None
lease.authorized_token_count == 8
len(row.output_tokens) == 8
sequence.num_completion_tokens + 8 < sequence.max_tokens
lease.last_write_position // block_size ==
    lease.first_write_position // block_size
the lease write block still occupies that block-table index
the write block hash is -1
```

The strict `< max_tokens` test is intentional. An exactly terminal commit
releases request storage, removes the sequence from the running queue, and
removes progress state. That mutation surface exceeds the delta journal and
must use the generic transaction.

### 3. Constant-size capture

The generalized journal records:

- the sequence object and token-list object identity;
- original token-list length;
- original `last_token`, `num_tokens`, and `status`;
- the lease's complete block-table identity for rollback validation;
- scheduler queue lengths and the expected single running sequence;
- prior decode-progress presence/value;
- prior SLO and adaptive-controller fields already captured by the existing
  journal;
- one `LeaseWriteBlockPublicationPlan`; and
- publication-applied and journal-state flags.

It does not copy:

- the complete sequence token list;
- every block's mutable state;
- the complete scheduler queues;
- every existing prefix-hash entry; or
- the complete materialized prefix-hash chain.

The lease's block-table identity tuple is reused by reference. The fast path
does not construct a second full block-table snapshot.

### 4. One-block publication plan

Before mutation, the selector calls:

```text
BlockManager.plan_lease_write_block_publication(...)
```

with:

- the eight output tokens;
- the lease write-block table index;
- expected block ID and generation;
- `materialized_tokens = lease.last_write_position + 1`; and
- predecessor hash authority when the write block becomes full.

If the eight tokens do not complete the write block, the plan records
`will_publish == false` and commit performs no hash work.

If the write block becomes exactly full, the planner:

- builds only that block's final token tuple;
- validates the predecessor's already-published hash authority;
- computes exactly one new block hash;
- snapshots only the affected primary/duplicate hash-index entries; and
- returns a reversible publication plan.

The commit calls:

```text
BlockManager.publish_lease_write_block(...)
```

instead of scanning `publish_full_blocks()`.

### 5. Commit data flow

The one-phase execution remains:

```text
validated exact-burst result
  -> prepare lease-local journal and publication plan
  -> append the same eight token IDs in the same order
  -> revalidate the same active lease
  -> publish zero or one write block
  -> record decode progress and SLO state
  -> leave the non-terminal sequence RUNNING
  -> commit journal
  -> record exact-burst commit and clear the lease
```

The implementation must not batch or replace the eight calls to
`Sequence.append_token()` in Stage 1. Keeping the mutation sequence identical
isolates the measured change to transaction capture/publication.

### 6. Rollback boundary

If any commit operation raises, rollback must:

1. verify token-list object identity;
2. verify the token list has not been truncated before the captured length;
3. verify the complete lease block-table identity and generations;
4. verify scheduler queue shape has not drifted;
5. undo the one-block publication if it was applied or became externally
   visible before the failure;
6. restore the affected block's prior hash and token IDs;
7. restore the affected primary and duplicate hash indexes;
8. truncate exactly the appended token suffix;
9. restore `last_token`, `num_tokens`, and `status`;
10. restore decode-progress, SLO, and adaptive-controller state; and
11. mark the journal rolled back.

The exact-burst pending lease remains owned by the existing outer transaction
until the normal post-commit accounting succeeds. A failed one-phase commit
must not silently clear or reuse the lease.

If rollback itself fails:

- preserve the original commit error;
- expose the rollback error through
  `SchedulerPostprocessRollbackError`;
- mark the prepared transaction and journal `rollback_failed`; and
- reject all later commit or rollback attempts on that object.

No best-effort rollback is allowed.

## Fallback Contract

When the feature is enabled and a one-phase exact row is inspected, record one
journal attempt. An ineligible row records one stable fallback reason and uses
the generic journal.

Required closed fallback reasons include:

```text
unsupported_burst_shape
terminal_one_phase
write_block_position_mismatch
write_block_boundary_crossed
write_block_already_published
predecessor_hash_unavailable
publication_plan_rejected
```

Existing split-phase fallback reasons remain valid. The implementation may
share a reason when the underlying failed invariant is identical, but the
gate's accepted-reason inventory must be explicit and closed.

A fallback is not a correctness failure. Any fallback in an otherwise
eligible candidate performance row is a gate failure because it means the
candidate did not exercise the intended optimization.

## Observability

Retain the aggregate lifecycle counters:

```text
lease_local_delta_journal_attempts
lease_local_delta_journal_captures
lease_local_delta_journal_commits
lease_local_delta_journal_rollbacks
lease_local_delta_journal_published_blocks
lease_local_delta_journal_fallback_counts
```

Add path-specific counters:

```text
lease_local_delta_journal_one_phase_attempts
lease_local_delta_journal_one_phase_captures
lease_local_delta_journal_one_phase_commits
lease_local_delta_journal_one_phase_rollbacks
lease_local_delta_journal_one_phase_published_blocks
lease_local_delta_journal_one_phase_fallback_counts
```

Existing split-phase counters may remain aggregate for compatibility, but
one-phase evidence must be independently attributable. Counter increments must
occur at the same lifecycle boundaries as the aggregate counters.

Benchmark rows must also report:

- generic-journal captures;
- lease-local prepare and commit duration samples;
- target forwards;
- graph replays;
- final-token D2H calls and bytes;
- sampled-logit D2H calls;
- output token IDs and sampled logits/argmax;
- TTFT, TPOT median/P95/P99, E2E, and throughput;
- CUDA peak allocated and reserved bytes; and
- fallback and rollback inventories.

## Test Strategy

Implementation follows strict RED to GREEN.

### Eligibility tests

Verify lease-local selection for non-terminal one-phase K8 rows when:

- the eight writes leave the write block partial; and
- the eight writes make the write block exactly full.

Verify generic fallback for:

- disabled feature;
- terminal one-phase K8;
- K2 through K7 ragged rows;
- split rows continue to use their existing eligibility rules;
- ordinary decode, prefill, mixed batch, and speculative rows;
- stale write-block position or generation;
- block-boundary crossing;
- already-published write block; and
- unavailable predecessor-hash authority.

### Complexity tests

At approximately 256, 2K, and 8K context lengths, assert that eligible
one-phase prepare:

- does not tuple-copy or iterate the complete token list;
- does not capture all block states;
- does not reconstruct all prior prefix hashes;
- plans at most one block publication;
- calls `compute_hash()` zero times for a partial block;
- calls `compute_hash()` at most once for a newly full block; and
- has journal state whose mutable payload does not grow with context length.

A direct CPU profile compares generic and lease-local one-phase K8 prepare and
rollback, reporting median, P95, hash calls, and positive Python allocation.

### Transaction tests

For both partial-block and full-block commits:

- successful eight-token atomic commit;
- exact token order;
- exact progress/SLO state;
- publication index correctness;
- injected failure before token append;
- injected failure after one or more token appends;
- injected failure after publication;
- injected failure during progress/SLO publication;
- complete logical-state equality after rollback;
- stale block generation after prepare;
- stale hash authority after prepare; and
- rollback-failure terminal behavior.

The disabled arm must retain generic behavior exactly.

### Adjacent regression tests

Run focused suites for:

- scheduler prepared postprocess;
- exact-greedy burst contract and statistics;
- model-runner exact-burst configuration;
- LLM-engine exact burst;
- continuation epochs;
- split-phase and ragged coalescing;
- existing lease-local journal CPU profile;
- existing lease-local gate/verifier/controller; and
- the new one-phase profile/gate/verifier/controller.

## Hardware Gate

The source-bound paired gate compares:

```text
baseline:  one-phase exact K8 + generic scheduler journal
candidate: one-phase exact K8 + lease-local scheduler journal
```

Fixed Stage-1 conditions:

- Qwen3-0.6B;
- TP1 on one strict-clean A100;
- batch size 1;
- completion-only greedy decode;
- `ignore_eos == true`;
- exact one-phase K8 enabled;
- split execution disabled for both arms;
- identical model, pushed source SHA, prompts, output lengths, and graph
  configuration;
- contexts approximately 2K, 4K, and 8K;
- ten paired repetitions per context;
- interleaved and reversed arm order;
- four correctness samples per context;
- no concurrent controller;
- remote task data only under
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`;
- strict-clean GPU admission: memory at most 1024 MiB, utilization at most 5%,
  and no compute process; and
- remote plus local independent verification.

The fixed row inventory is:

```text
performance:  2 policies x 3 contexts x 10 repetitions = 60 rows
correctness:  2 policies x 3 contexts x 4 samples     = 24 rows
```

The run tag is immutable and never reused after an attempted launch.

### Correctness and invariance gates

All must pass:

- exact output-token IDs for every pair;
- exact sampled argmax for every sampled position;
- sampled logits within the existing exact-burst tolerance;
- target forwards unchanged;
- graph replays unchanged;
- D2H calls and bytes unchanged;
- exact-burst lease widths unchanged;
- candidate generic-journal captures equal zero for every eligible burst;
- candidate one-phase attempts, captures, and commits equal eligible-burst
  count;
- candidate one-phase rollbacks and fallbacks equal zero in performance rows;
  and
- both independent verifiers pass.

### Benefit gates

The candidate must achieve:

```text
8K prepare median improvement >= 50%
8K prepare P95 improvement    >= 50%
aggregate prepare median      >= 35%
aggregate prepare P95         >= 35%
aggregate TPOT median         >= 1%
aggregate TPOT P95            >= 1%
```

The TPOT threshold is intentionally positive rather than non-regression. The
optimization is only worth carrying if removing host transaction work is
visible in the serving metric.

### Cost and non-regression gates

All must pass:

```text
TTFT regression                 <= 2%
E2E regression                  <= 2%
throughput regression           <= 2%
aggregate TPOT P99 regression   <= 2%
CUDA peak allocated regression  <= 1%
CUDA peak reserved regression   <= 1%
target forwards delta            = 0
graph replays delta               = 0
D2H calls/bytes delta             = 0
```

The report must also state:

- Python allocation change;
- code and test surface added;
- fallback inventory;
- whether the write block was partial or newly published;
- remaining complete lease block-identity validation cost; and
- any metric that is neutral or below the promotion threshold.

## Classification

Possible terminal classifications:

```text
GO_EXACT_BURST_ONE_PHASE_LEASE_LOCAL_JOURNAL
NO_GO_PERFORMANCE
NO_GO_CORRECTNESS
NO_GO_TRANSACTIONAL_SAFETY
NO_GO_EVIDENCE_INCOMPLETE
```

A GO authorizes only the default-disabled Stage-1 one-phase K8 lease-local
journal. It does not authorize:

- enabling the feature by default;
- terminal K8 commits;
- K2 through K7 ragged commits;
- TP greater than one;
- removal of the generic journal;
- removal of complete lease/block-generation validation;
- a claim of GPU-compute improvement; or
- a claim of academic novelty.

## Expected Implementation Files

The focused implementation may update:

```text
tinyvllm/config.py
tinyvllm/engine/scheduler.py
tinyvllm/engine/exact_greedy_decode_burst.py
tools/test_model_runner_spec_verify.py
tools/test_scheduler_prepared_postprocess.py
tools/test_exact_greedy_decode_burst.py
tools/profile_exact_burst_one_phase_lease_local_journal.py
tools/test_profile_exact_burst_one_phase_lease_local_journal.py
tools/exact_burst_one_phase_lease_local_journal_gate.py
tools/test_exact_burst_one_phase_lease_local_journal_gate.py
tools/exact_burst_one_phase_lease_local_journal_verify.py
tools/test_exact_burst_one_phase_lease_local_journal_verify.py
tools/run_exact_burst_one_phase_lease_local_journal_remote.py
tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py
```

The existing configuration and block-manager APIs should require no semantic
change unless TDD exposes a missing invariant. Any such change must remain
inside the one-block publication contract.

The final evidence reconciliation may update:

```text
docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
docs/superpowers/audits/2026-08-24-exact-burst-one-phase-lease-local-journal-audit.md
AGENT_HANDOFF_STATE.md
```
