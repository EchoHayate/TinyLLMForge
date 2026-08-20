# Autoregressive Draft Bounded Rollback Journal Design

**Date:** 2026-08-20

**Status:** Approved for implementation

## Goal

Reduce TP4/B4/Q4 autoregressive-draft TPOT and decode tail latency by
eliminating full-capacity Python rollback snapshots from the measured decode
path while preserving the existing transactional correctness contract.

The optimization is successful only if a fresh, source-bound, interleaved
paired campaign establishes all of the following:

```text
overall TPOT p95:              <= 105.87 ms
overall TPOT median:           <= 85.66 ms
TTFT p95 regression:           <= 3%
throughput regression:         <= 3%
exact output tokens:           pass
Proposal-KV transactions:      pass
four-rank correctness:         pass
paired stationarity:           pass
```

The implementation alone does not establish a performance improvement.

## Fixed Scope

The first optimization slice is limited to the two host-side transactional
boundaries identified by the terminal command-timeline campaign:

```text
Scheduler.prepare_postprocess()
BlockManager.commit_speculative_kv_commit_batch()
```

It must preserve:

- the existing `PreparedSchedulerPostprocess` public contract;
- Proposal-KV transaction states and ownership checks;
- all-or-nothing batch commit behavior;
- scheduler rollback after injected row, hook, allocator, or block failures;
- exact free/used block ownership and free-list order;
- prefix-cache hash publication and duplicate-hash indexes;
- hybrid-state lease and release-event behavior;
- speculative side-state and proposal-lifecycle ordering; and
- current exception propagation and runtime-poisoning boundaries.

It must not:

- change model execution, sampling, accepted-token selection, or outputs;
- add CUDA synchronization, `.item()`, worker acknowledgements, or fences;
- disable Python garbage collection in production;
- add measured-path logging or profiling;
- weaken rollback or convert failures into silent best-effort recovery;
- change the command-timeline schema or paired campaign identity;
- optimize unrelated scheduler, CUDA Graph, or Proposal-KV code; or
- claim that the isolated `speculative_prepare` CUDA anomaly is fixed.

## Evidence and Root-Cause Boundary

The immutable terminal campaign is:

```text
20260818-command-timeline-tp4-b4-q4-r23
source commit:
596e724ea87966b2ab3b47cccda08c106f9084bb
```

It completed 40 measured repeats and 160 measured requests with exact
correctness and timeline conservation. Eight of the 40 repeats contained a
large rollback-snapshot phase:

```text
snapshot spike range:          approximately 527-622 ms
affected repeats:              8 / 40
affected request samples:      32 / 160
affected sample fraction:      20%
```

Seven spikes were attributed to
`proposal_kv_prepare_commit`; one was attributed to
`scheduler_prepare_postprocess`. The spike moves between the two phases
rather than remaining attached to one engine step or one CUDA Graph mode.

The implementation explains this behavior:

- `Scheduler._capture_postprocess_snapshot()` copies every KV block, the
  complete free and used inventories, every hash index, and the complete
  hybrid-state allocator before each prepared postprocess.
- `BlockManager.commit_speculative_kv_commit_batch()` independently copies
  every KV block, the complete free and used inventories, and every hash
  index before applying a batch that touches only the selected sequences and
  their reserved blocks.
- both paths allocate nested Python tuples, lists, dicts, and sets in
  proportion to total KV capacity, not transaction size.

The proven root cause is therefore an `O(total KV capacity)` rollback
snapshot on the decode critical path.

Automatic Python garbage collection is a high-confidence explanation for
why the otherwise repeated allocation work occasionally becomes a
`0.5s+` pause and why the pause is charged to whichever snapshot crosses a
collection boundary. The r23 artifact did not record GC callbacks, so GC is
not treated as directly proven. The selected change removes the allocation
storm regardless of whether a particular pause was GC, allocator, or host
scheduling dominated.

One eager repeat separately contained approximately `1.035s` in
`speculative_prepare`, including anomalous worker/CUDA time. That is a
different boundary. It remains visible to the final paired gate and prevents
an overbroad claim.

## Considered Approaches

### Disable automatic GC during decode

This is rejected as the production optimization. It could move periodic
collection outside a request, but it leaves both full-capacity copies in
place, retains their throughput and allocation cost, and introduces memory
growth and collection-timing risk.

GC control may be used only in a separate diagnostic experiment if later
needed to explain residual variance. It is not part of this implementation.

### Convert the entire scheduler commit into a no-throw transaction

A fully validated, no-throw apply phase could eventually remove rollback
state entirely. It would require a larger redesign of prefill hooks,
sequence release, hybrid-state publication, hash publication, and error
boundaries. That is disproportionate to the localized r23 bottleneck.

### Use bounded mutation journals

Selected. Each transaction records only state that the current batch can
mutate. Validation remains ahead of mutation; commit behavior remains
all-or-nothing; failure restores the journal in reverse order.

The journal size is bounded by active transaction work:

```text
Proposal-KV:
  O(number of plans + reserved blocks + published blocks)

Scheduler:
  O(scheduled sequences + their owned blocks + touched leases/hashes)
```

Neither journal may scale with unused KV capacity.

## Architecture

### Shared journal principles

The two journals are separate types because they own different state, but
they follow the same rules:

1. validate the complete batch before the first mutation;
2. identify or lazily record every object before its first write;
3. record each object or key at most once;
4. apply the existing mutation order;
5. discard the journal after a successful commit;
6. restore in strict reverse mutation order after failure; and
7. surface the original failure unless rollback itself fails.

Rollback failure remains fatal. At the engine publication boundary, a
rollback failure must continue to poison speculative runtime state rather
than permitting a partially restored engine to serve another request.

The design does not introduce a generic transaction framework. The helpers
remain local to `block_manager.py` and `scheduler.py`.

### Proposal-KV commit journal

`commit_speculative_kv_commit_batch()` retains its current prevalidation:

- every value is a `SpeculativeKVCommitPlan`;
- recomputed plans exactly equal supplied plans;
- sequence IDs are unique;
- transaction objects are unique; and
- reserved block IDs are disjoint.

Before applying a plan, the bounded journal records:

- the original block table of each participating sequence;
- the original state of each participating transaction;
- metadata for each block in the union of:
  - `committed_block_ids`,
  - `unused_block_ids`, and
  - publication block IDs;
- preexisting primary and duplicate hash-index entries for hashes that a
  publication may add or replace;
- whether every touched block was in `used_block_ids`; and
- free-list deltas caused by releasing unused reserved blocks.

The journal must not copy:

- `self.blocks` as a whole;
- `free_block_ids` as a whole;
- `used_block_ids` as a whole; or
- either complete hash-index dictionary.

Unused reserved blocks are released through the existing mutation path.
Their free-list append order is deterministic. Rollback removes those exact
appends in reverse order, restores touched block metadata and ownership,
then restores sequence tables and transaction states.

Hash rollback restores only keys associated with touched publication
hashes and any prior hashes owned by touched blocks. Duplicate hash buckets
and primary block selection must be byte-for-byte equivalent to their
pre-commit logical state.

### Scheduler postprocess journal

`PreparedSchedulerPostprocess.snapshot` remains available to avoid changing
callers, but its value becomes a bounded scheduler rollback journal rather
than a full scheduler image.

The journal records:

- the existing per-sequence fields for scheduled sequences;
- scheduler queue order for `waiting`, `prefilling`, and `running`;
- metadata for blocks owned by scheduled sequences;
- touched prefix-cache hash keys;
- touched block free/used membership and deallocation append deltas;
- only hybrid-state leases and slots owned by scheduled sequences;
- the prior hybrid release-event length and any appended leases;
- decode-progress entries for scheduled sequences;
- `_last_slo_postprocess`;
- prefill notification membership for scheduled sequences;
- `_prefill_commit_hook_error`;
- adaptive mixed-controller scalars;
- `_consecutive_prefill_chunks`;
- SLO clock-invalid fields; and
- `_last_slo_decision_now_ns`.

Queue snapshots are allowed because they contain object references and are
bounded by admitted active sequences. The journal must not copy all KV
blocks or all hybrid-state slots.

For prefill publication, the journal covers only full blocks between the
scheduled sequence's old and new computed boundaries and their hash keys.
For decode completion, it covers the sequence's owned block table and the
hybrid-state lease that `_release_request_storage()` can release.

If a prefill commit hook raises, its external side effects are not invented
or reversed by this change. Scheduler-owned state is restored exactly as it
is today, and `_prefill_commit_hook_error` retains the raised hook failure.

### Rollback order

Proposal-KV rollback uses:

```text
free-list deltas
  -> touched used membership
  -> touched block metadata
  -> touched hash indexes
  -> sequence block tables
  -> transaction states
```

Scheduler rollback uses:

```text
hybrid release-event appends
  -> touched hybrid leases/slots
  -> block free-list deltas
  -> touched block ownership and metadata
  -> touched hash indexes
  -> sequence fields
  -> scheduler queues
  -> progress/notification/SLO/controller scalars
```

Restoration helpers must be idempotence-guarded by journal state. A committed,
rolled-back, or rollback-failed journal cannot be reused.

## Correctness and Error Handling

### Validation before mutation

All stale-plan, ownership, token, capacity, publication, and disjointness
checks remain before the mutation loop where possible. The journal is not a
substitute for validation.

### Failure atomicity

Tests must inject failures after:

- the first Proposal-KV plan has committed but before the second;
- a publication hash has been registered;
- an unused reserved block has been released;
- the first scheduler row has appended tokens;
- a sequence has been marked finished and its KV blocks released;
- a hybrid-state lease has been released;
- a prefill hash has been published; and
- a prefill commit hook has raised.

After each failure, logical authority snapshots must match the exact
pre-commit state, including free-list order and duplicate hash indexes.

### Rollback failure

Tests must separately inject rollback failure and verify:

- the original failure is retained as causal context;
- the journal enters a terminal rollback-failed state;
- speculative runtime is poisoned at the existing engine boundary; and
- no later commit or rollback reuses the journal.

## Test Strategy

Implementation follows strict RED to GREEN.

### Structural complexity tests

Use an indexable but non-iterable block collection in focused unit tests.
Valid transaction operations may index touched blocks, but any full
`for block in self.blocks` traversal fails the test.

Add equivalent guards showing that:

- Proposal-KV batch commit does not iterate all blocks;
- scheduler prepare does not iterate all blocks;
- untouched block metadata is not read or copied; and
- journal entry counts depend on touched blocks, not configured capacity.

These are deterministic complexity-contract tests, not wall-clock tests.

### Existing transactional tests

Retain and extend:

```text
tools/test_speculative_kv_transaction.py
tools/test_scheduler_prepared_postprocess.py
tools/test_engine_speculative_execution.py
tools/test_engine_speculative_runtime.py
```

Existing success, stale-plan, non-mutating prepare, rollback, commit-failure,
mixed prefill/decode, final-token, lifecycle, and runtime-poisoning tests must
remain green.

### Focused performance evidence

A CPU-only focused benchmark may report journal construction versus total KV
capacity, but it is diagnostic evidence only. It must not replace the real
TP4/B4/Q4 paired gate.

No timing threshold is added to ordinary unit tests.

### Full paired gate

After local verification and source commit, package a fresh source-bound
candidate and run baseline and candidate in the same interleaved protocol.
Failed and attempted tags remain immutable.

The campaign must preserve:

```text
TP = 4
batch = 4
Q = 4
prompt tokens = 256
output tokens = 16
temperature = 0
Proposal-KV allocator = direct
Proposal-KV offload = disabled
balanced eager/graph order
strict four-clean-GPU admission
dual verification
complete manifest
stationarity requirement
```

The baseline must use the terminal source behavior from an exact pinned
source archive or preserved source bundle. No additional local worktree may
be created. The candidate differs only by the bounded journal implementation
and required tests.

### Source-version pair protocol

The existing command-timeline runner is a single-source bundle whose internal
labels compare eager and graph positions. It is not, by itself, a
baseline-source versus candidate-source gate. The bounded-journal authority
therefore reuses two independently frozen command-timeline bundles and adds a
thin source-pair orchestration and comparison layer. It does not change the
worker, command-timeline schema, measured-path instrumentation, or model
configuration.

The two frozen revisions are:

```text
baseline:
  596e724ea87966b2ab3b47cccda08c106f9084bb

candidate:
  the committed and pushed branch HEAD used for the campaign
```

Both sources are exported directly from Git objects. The gate must not create
a checkout or worktree for either source. The source archives, manifests, and
tree hashes are distinct, immutable inputs to the same parent run tag.

Each source retains the existing eight-epoch mode schedule:

```text
epoch modes:
  eager, graph, graph, eager, graph, eager, eager, graph

measured repeats per epoch:
  5

measured repeats per source:
  40

request samples per source:
  160
```

The parent orchestrator executes corresponding epochs as eight source pairs.
Within each CUDA mode, baseline-first and candidate-first each occur twice.
Across the complete gate, each source runs first four times and second four
times:

```text
pair 0: eager  baseline -> candidate
pair 1: graph  candidate -> baseline
pair 2: graph  baseline -> candidate
pair 3: eager  candidate -> baseline
pair 4: graph  baseline -> candidate
pair 5: eager  baseline -> candidate
pair 6: eager  candidate -> baseline
pair 7: graph  candidate -> baseline
```

Before and after every member, the existing frozen four-GPU inventory must
still match exactly. Any external process, UUID change, threshold violation,
worker failure, incomplete epoch, or transport failure terminates the parent
campaign without running a replacement member under the same tag.

Each source bundle must independently complete:

```text
canonical assembly
pre-manifest verification
complete checksum manifest
archived remote verification
no-overwrite controller copy
controller verification
normalized receipt equality
```

The source-pair comparator consumes only those verified immutable bundles. It
binds both canonical artifact hashes, both source commits and tree hashes,
both manifests, both normalized verifier receipts, the parent pair schedule,
and the frozen GPU UUID set.

For corresponding epoch and repeat identities, it requires exact equality of:

- output token IDs;
- prompt and request order;
- proposal token rows and row lengths;
- accepted token rows and accepted-prefix counts;
- accepted/proposed totals and acceptance rate;
- transaction digest; and
- zero active transactions.

It also requires both underlying command-timeline artifacts to retain exact
four-rank identity correctness and timeline conservation.

The candidate aggregate is computed from all 160 request samples:

```text
TPOT median and p95:
  request-level tpot_ns

TTFT p95:
  request-level ttft_ns

throughput:
  median of the 40 measured batch token-throughput values
```

Regression uses the fresh paired baseline from the same campaign:

```text
TTFT p95 regression:
  candidate_ttft_p95 / baseline_ttft_p95 - 1

throughput regression:
  1 - candidate_median_throughput / baseline_median_throughput
```

Paired stationarity requires all sixteen source epochs to pass the existing
command-timeline stationarity checks. It also requires, separately for eager
and graph pairs, both the candidate/baseline TPOT ratio and throughput ratio
to have robust MAD divided by median `<= 0.10` and first-half versus
second-half median drift `<= 0.15`. A zero or non-finite denominator is an
artifact failure.

The source-pair artifact has its own complete manifest and two independent
verifier receipts. The verifier classification precedence is:

```text
INCONCLUSIVE_ARTIFACT
NO_GO_CORRECTNESS
INCONCLUSIVE_STATIONARITY
NO_GO_TPOT_P95
NO_GO_TPOT_MEDIAN
NO_GO_TTFT_REGRESSION
NO_GO_THROUGHPUT_REGRESSION
GO_TPOT_TAIL_OPTIMIZATION
```

`INCONCLUSIVE_ENVIRONMENT` is produced by orchestration before a complete
source-pair artifact exists. No historical r23 timing value may substitute
for the fresh paired baseline in TTFT or throughput regression calculations.

## Acceptance and Claim Rules

The optimization is accepted only when every fixed threshold in the Goal
section passes in the same campaign.

Possible outcomes are:

```text
GO_TPOT_TAIL_OPTIMIZATION
NO_GO_TPOT_P95
NO_GO_TPOT_MEDIAN
NO_GO_TTFT_REGRESSION
NO_GO_THROUGHPUT_REGRESSION
NO_GO_CORRECTNESS
INCONCLUSIVE_STATIONARITY
INCONCLUSIVE_ENVIRONMENT
INCONCLUSIVE_ARTIFACT
```

Removing the observed snapshot spikes is not by itself a `GO`. If the
isolated worker/CUDA anomaly or another boundary keeps TPOT above the fixed
thresholds, the result remains `NO_GO` or `INCONCLUSIVE`, and the next
optimization must be selected from the new evidence.

## Implementation Sequence

1. Add failing structural tests that reject full block iteration.
2. Add failing Proposal-KV rollback tests for touched-state restoration.
3. Implement the bounded Proposal-KV journal and make those tests green.
4. Add failing scheduler journal tests for decode, completion, prefill,
   hybrid-state release, hook failure, and rollback failure.
5. Implement the bounded scheduler journal and make those tests green.
6. Run the focused transactional suites and broader affected suite.
7. Review the diff for hidden full-capacity copies and unrelated changes.
8. Commit and push the implementation with exact-path staging.
9. Only under the established remote-path, GPU-admission, and campaign
   authorization rules, run a fresh paired campaign.
10. Complete dual verification, manifest verification, audit reconciliation,
    commit, and push before making a performance claim.

## Documentation and Artifact Boundaries

The authoritative checkout remains:

```text
/Users/bytedance/Desktop/TinyLLMForge
```

All remote outputs, caches, logs, diagnostics, receipts, manifests, and
scratch remain under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

No task output may be written to local or remote `/`, `/tmp`, or
`/private/tmp`. The remote source checkout
`/data00/home/sitian/tllm/TinyLLMForge` remains read-only and must not be
modified.

The retired adaptive-ngram checkout is outside scope.
