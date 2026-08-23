# Exact-Burst Lease-Local Delta Journal Design

**Date:** 2026-08-23

**Status:** Approved under the standing autonomous-optimization authorization

**Stage-1 model:** Qwen3-0.6B

**Primary target:** remove context-length-scaled scheduler snapshot and
prefix-hash work from non-terminal split-phase exact-burst publication

## Objective

Reduce host-side split-phase commit latency without changing model execution,
token cadence, target-model forward count, D2H traffic, output tokens, or GPU
KV ownership.

The current split-phase K8 path publishes four tokens at a time, but each
four-token publication uses the generic scheduler rollback journal. That
journal:

- copies the complete sequence token list;
- snapshots every block in the sequence block table;
- reconstructs the complete full-block prefix-hash chain;
- snapshots all scheduler queues; and
- later scans the complete block table again in `publish_full_blocks()`.

Those operations scale with context length even though a split-phase parent
lease authorizes eight writes inside one already known writable block.

The selected optimization replaces only the journal capture and block
publication work for eligible non-terminal split phases with a lease-local
delta journal. All scheduler authority checks remain unchanged.

The implementation alone does not establish a performance improvement. A
fresh, source-bound paired hardware gate must report both benefit and cost.

## Evidence and Root-Cause Boundary

Focused host profiling measured the current generic phase-prepare path at
three sequence lengths:

```text
sequence length       249        2041        8185
prepare median     149.916 us  818.000 us  2502.917 us
prepare p95        161.250 us  917.166 us  3540.000 us
journal blocks          16         128         512
journal hashes          15         127         511
positive allocation 39,344 B   131,968 B   500,991 B
```

At 8,192 tokens, 100 prepare-plus-rollback operations consumed approximately
`0.467 s` in cProfile:

```text
phase prepare                                      0.433 s
prepare_postprocess                                0.391 s
SchedulerPostprocessJournal.capture                0.234 s
_capture_sequence_publication_hashes               0.134 s
capture_exact_burst_publication_hashes              0.121 s
BlockManager.compute_hash: 102,200 calls            0.190 s
```

A fixed-size capture prototype was approximately `0.5 us`, but that prototype
is only design evidence. It is not an end-to-end result and is not a gate.

The source-bound Qwen3-0.6B canonical gates at commit
`c28829de592fac634b73ea784b1ac748a15f9124` establish the production
baseline:

- ragged coalescing run
  `20260823-qwen3-06b-ragged-canonical-r9` completed 45 performance rows and
  36 correctness rows, passed both verifiers, and classified
  `GO_EXACT_BURST_RAGGED_COALESCING`;
- relative to split K8, ragged coalescing improved aggregate TPOT P95 by
  `23.56%`, E2E by `2.01%`, and throughput by `2.05%`, while exact outputs
  and sampled logits matched and aggregate reserved memory was unchanged;
- split-phase run `20260823-qwen3-06b-split-canonical-r10` completed
  60 performance rows and 48 correctness rows and passed both evidence
  verifiers, but classified
  `NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE`; and
- relative to one-phase K8, split K8 regressed aggregate TPOT median by
  `8.09%`, TPOT P95 by `42.30%`, TPOT P99 by `54.79%`, E2E by `2.77%`, and
  throughput by `2.70%`, while aggregate reserved memory was unchanged and
  all 36 correctness pairs were exact.

These results do not prove that scheduler journaling causes the entire
split-phase regression. They do prove that the baseline remains
performance-negative despite exact correctness, and they make the separately
profiled context-scaled journal cost a valid next bottleneck to isolate.

The proven opportunity is narrower than a general scheduler redesign:

- split-phase exact burst has one sequence and one four-token row;
- its K8 parent lease already freezes the complete block-table identity,
  schedule generation, write block, write-block generation, physical slots,
  and sequence length;
- all eight writes remain in one writable block;
- the prefix cannot make that block full, because four suffix positions must
  remain in the same block; and
- a non-terminal suffix can make at most that one write block newly
  publishable.

The optimization therefore targets context-scaled host bookkeeping. It does
not claim to reduce GPU execution time.

## Fixed Scope

Stage 1 is limited to:

- TP1, rank 0, batch size 1;
- exact greedy completion-only decode;
- `temperature == 0`;
- `ignore_eos == true`;
- a K8 parent lease;
- split prefix and suffix rows of exactly four tokens;
- all parent writes inside one writable KV block;
- prefix publication, which is necessarily non-terminal;
- suffix publication only when the request remains non-terminal after the
  four suffix tokens; and
- a default-disabled configuration flag.

The path must preserve:

- pending-lease identity and schedule-generation validation;
- complete block-table ID and generation validation;
- write-block ID and generation validation;
- split-result and graph identity validation;
- prefix-before-suffix ordering;
- publication-ticket identity and token-transfer validation;
- exact token order and output-budget validation;
- the existing commit exception boundary;
- automatic rollback after a commit failure;
- terminal `SchedulerPostprocessRollbackError` behavior when rollback fails;
- prefix-cache duplicate-hash and primary-hash indexes;
- decode-progress, SLO, and adaptive-controller state; and
- the parent lease remaining active after prefix commit and being released
  exactly once after suffix commit.

The path must not:

- change graph replay count, target-model forward count, or D2H count;
- change split mailbox, CUDA event, or pending-suffix ownership;
- skip full parent block-table identity validation;
- introduce a global block epoch or rolling sequence hash;
- change ordinary decode, one-phase exact burst, terminal suffix, prefill,
  mixed-batch, speculative, or completion-release transactions;
- weaken rollback into best-effort recovery;
- write remote task data outside the approved mounted task root; or
- become enabled by default.

## Considered Approaches

### A. Lease-local delta journal

Record only state that an eligible non-terminal four-token split phase can
mutate, and publish at most the lease's one write block.

Advantages:

- removes full token-list copies from the eligible prepare path;
- removes full block-table journal capture;
- removes complete prefix-hash reconstruction;
- removes full block-table publication scans;
- keeps the existing validation and exception structure; and
- bounds new rollback state by one sequence, one block, and one hash key.

Costs and risks:

- introduces a second scheduler journal type;
- requires an explicit eligibility predicate and generic fallback;
- adds a single-block publication API;
- requires fault-injection parity with the generic journal; and
- does not remove the existing full block-table identity revalidation.

This is the selected approach.

### B. Sequence-owned rolling prefix-hash frontier

Maintain a persistent hash frontier on every sequence so any future
publication can continue hashing without scanning prior blocks.

This could benefit more paths, but the frontier would have to remain correct
across allocation, deallocation, prefix-cache reuse, preemption, serialization,
and rollback. Its state surface is much larger than the measured split-phase
hotspot, so it is deferred.

### C. No-throw phase commit with no rollback

Prevalidate every operation and declare the apply phase incapable of raising.

This could eventually eliminate journal state, but the current path still
touches token lists, prefix-cache indexes, SLO state, and helper calls that can
raise or be fault-injected. Proving a no-throw boundary would require a larger
transaction redesign and would weaken current failure containment if done
partially. It is rejected for Stage 1.

## Architecture

### 1. Default-off configuration

Add:

```text
Config.exact_greedy_decode_burst_lease_local_delta_journal: bool = False
```

When true, configuration validation requires:

```text
exact_greedy_decode_burst == true
exact_greedy_decode_burst_split_phase == true
exact_greedy_decode_burst_tokens == 8
```

Ragged coalescing may be either enabled or disabled. Smaller one-phase ragged
leases do not use this fast path.

The scheduler stores the normalized boolean at construction time. No runtime
environment variable or implicit auto-enable rule is added.

### 2. Eligibility predicate

`prepare_postprocess()` may select the delta journal only after all existing
row, lease, split-result, ticket, token, output-budget, sequence-state, and
block-identity checks have passed.

Eligibility requires:

```text
feature flag enabled
one sequence
one output row
row.exact_burst == true
row.exact_burst_phase in {"prefix", "suffix"}
parent lease authorized_token_count == 8
len(row.output_tokens) == 4
sequence.status == RUNNING
sequence.ignore_eos == true
batch_kind is None
is_prefill == false
do_sample == true
lease write block is still in the same block-table position
all eight lease writes remain inside that block
phase result remains non-terminal after applying its four tokens
```

The prefix is non-terminal by construction: K8 admission requires at least
eight remaining completion tokens and EOS-sensitive requests are excluded.

The suffix is eligible only when:

```text
sequence.num_completion_tokens + 4 < sequence.max_tokens
```

An exactly terminal suffix continues to use the generic journal because
completion releases all request blocks, mutates queues, may release hybrid
state, and therefore exceeds the one-block delta boundary.

Any eligibility uncertainty selects the generic journal. Fallback is not an
error and must be counted by reason in test and benchmark evidence.

### 3. Delta journal contents

Introduce a dedicated journal local to `scheduler.py`, conceptually:

```text
ExactBurstPhaseDeltaJournal
```

It records:

- the sequence object;
- original token-list length;
- original `last_token`;
- original `num_tokens`;
- original `status`;
- whether the sequence had a decode-progress entry and its prior value;
- the prior `_last_slo_postprocess` mapping;
- adaptive mixed-controller state and streak scalars;
- `_consecutive_prefill_chunks`;
- SLO invalidity fields;
- `_last_slo_decision_now_ns`;
- the expected write-block ID and generation;
- optional pre-publication metadata for that one block;
- optional prior primary and duplicate hash-index entries for one future hash;
- whether a single-block publication is expected; and
- journal lifecycle state.

It does not copy:

- the complete token list;
- the sequence block table;
- scheduler queues;
- unrelated blocks;
- the complete hash dictionaries;
- hybrid allocator state;
- prefill state; or
- release-event queues.

The omitted state is safe only because the eligibility predicate excludes
terminal release, queue mutation, prefill, mixed batches, hybrid release, and
multi-sequence work.

### 4. Fixed-size future publication calculation

For an eligible phase, at most the lease write block can become newly full.

Prefix:

- cannot fill the write block;
- records no future publication hash; and
- performs no block publication after appending tokens.

Non-terminal suffix:

- may leave the write block partial, in which case no publication occurs; or
- may exactly fill the write block, in which case one future hash is computed
  from at most one block's token IDs plus the four suffix tokens.

If the write block is not the first sequence block, its immediate predecessor
must already have a published hash, and that hash must still index the
predecessor block through the existing primary or duplicate-hash maps. If
that predecessor authority is unavailable or inconsistent, the prepare path
falls back to the generic journal rather than reconstructing the full chain.

The block size is fixed configuration state. Work proportional to one block
is treated as bounded with respect to context length.

### 5. Single-block publication API

Add a narrow `BlockManager` helper for the lease-owned write block. It:

1. validates the supplied block-table index, block ID, and generation;
2. verifies the materialized boundary makes that block fully available;
3. returns without mutation when the block remains partial;
4. rejects a missing predecessor hash instead of scanning prior blocks;
5. computes at most one block hash;
6. registers at most one cached block through the existing duplicate-hash
   index path; and
7. reports whether publication occurred.

The generic `publish_full_blocks()` API remains unchanged for all other
callers.

The helper must never infer authority from the current sequence alone. Its
expected block identity comes from the already validated parent lease and
delta journal.

### 6. Commit integration

`PreparedSchedulerPostprocess.snapshot` already accepts an object, so the
public prepared object does not change.

`commit_prepared_postprocess()` accepts either:

- `SchedulerPostprocessJournal`; or
- `ExactBurstPhaseDeltaJournal`.

Both expose:

```text
scheduled sequences
rollback(scheduler)
state
```

The existing commit prevalidation remains before mutation. The same
`_apply_prepared_decode_row()` logic appends the four tokens and updates
progress/status, but exact-burst publication is dispatched by journal type:

- generic journal -> `publish_full_blocks()`;
- delta journal -> lease-local single-block publication.

Commit statistics and split-phase state transitions remain after the
transaction commits, exactly as today.

### 7. Rollback

Delta rollback restores only the eligible mutation set, in this order:

```text
single future hash index
  -> single write-block metadata
  -> token-list suffix deletion
  -> last_token / num_tokens / status
  -> decode-progress entry
  -> SLO and adaptive-controller scalars
```

Token rollback uses:

```python
del sequence.token_ids[original_length:]
```

It must not allocate a copy of the original token list.

Before deleting, rollback verifies:

- the current token list is at least the original length;
- `sequence.token_ids` is still the exact list object captured by the
  journal; and
- no unrelated sequence or queue mutation occurred.

The last condition is enforced structurally by eligibility and by tests that
inject forbidden mutations and require rollback failure rather than silent
repair.

Journal states are:

```text
active -> committed
active -> rolled_back
active -> rollback_failed
```

A committed, rolled-back, or rollback-failed journal cannot be reused.
Rollback failure continues to raise `SchedulerPostprocessRollbackError` with
both the original commit error and rollback error.

### 8. Observability

Extend exact-burst scheduler statistics with:

```text
lease_local_delta_journal_attempts
lease_local_delta_journal_captures
lease_local_delta_journal_commits
lease_local_delta_journal_rollbacks
lease_local_delta_journal_published_blocks
lease_local_delta_journal_fallback_counts
```

Counters are scheduler-owned and contain no per-request identifiers.

An attempt is counted only for a split-phase row when the feature is enabled.
A capture proves the fast journal was selected. A commit or rollback is
counted at the corresponding terminal journal transition. Fallback reasons
must be stable strings suitable for gate validation.

Expected production fallback reasons include:

```text
terminal_suffix
write_block_position_mismatch
write_block_already_published
predecessor_hash_unavailable
unsupported_phase_shape
```

Invariant violations already rejected by existing validation are errors, not
fallbacks.

## Correctness and Error Handling

### Validation remains context-complete

The optimization does not make the entire phase prepare O(1). Existing
`_validate_pending_exact_greedy_decode_burst()` still revalidates the complete
lease block-table identity and current block generations.

The claim is narrower:

```text
journal capture and publication no longer scale with sequence context length
for eligible split phases
```

This distinction must appear in benchmark reports and final documentation.

### Publication collisions

The future hash may already map to one or more cached blocks. The delta journal
records the prior primary mapping and duplicate bucket for only that hash.
Rollback restores both exactly.

The write block's prior hash must be `-1` for a new publication. Any other
state falls back before mutation.

### Commit failure

Tests inject failures:

- after the first appended token;
- after all four appended tokens;
- immediately before single-block publication;
- immediately after hash registration;
- during decode-progress publication;
- during SLO publication; and
- during adaptive-controller reset.

After rollback, the logical scheduler snapshot must equal the exact
pre-commit snapshot, including the parent lease and split phase.

### Rollback failure

Tests separately inject delta rollback failure and verify:

- the original commit failure is retained;
- the rollback failure is exposed;
- the prepared object enters `rollback_failed`;
- the delta journal enters `rollback_failed`; and
- no later commit or rollback accepts the object.

## Test Strategy

Implementation follows strict RED to GREEN.

### Configuration tests

Verify:

- default is false;
- non-boolean values are rejected;
- exact burst is required;
- split phase is required;
- K8 is required; and
- ragged coalescing is optional.

### Eligibility tests

Verify delta selection for:

- non-terminal prefix;
- non-terminal suffix with a partial write block; and
- non-terminal suffix that fills and publishes one write block.

Verify generic fallback for:

- disabled feature;
- ordinary decode;
- one-phase exact burst;
- ragged K2-K4 one-phase commits;
- terminal suffix;
- unsupported row or batch shape;
- missing predecessor hash; and
- any uncertain write-block publication state.

### Complexity tests

Use long sequences at multiple context lengths and assert:

- the delta capture does not iterate or tuple-copy the complete token list;
- it does not iterate the complete block table after existing lease
  revalidation;
- it captures at most one block and one hash key;
- phase publication calls `compute_hash()` at most once;
- prefix publication calls it zero times; and
- generic fallback retains existing behavior.

A focused benchmark records median, P95, and positive Python allocation for
both journals at approximately 256, 2K, and 8K sequence lengths.

### Transaction tests

Reuse the existing split-phase test matrix and run it with the feature both
off and on:

- prefix then suffix success;
- correctness-trace revalidation;
- out-of-order and duplicate phases;
- ticket mismatch;
- token mismatch after prepare;
- schedule-generation drift;
- block-generation drift;
- prefix rollback;
- suffix rollback preserving the committed prefix; and
- rollback-failure terminal behavior.

Add exact logical snapshots around every injected failure.

### Adjacent regression tests

Run:

- exact-burst contract tests;
- scheduler prepared-postprocess tests;
- LLM engine split-phase tests;
- split-phase profile, gate, verifier, and remote-controller tests;
- ragged profile, gate, verifier, and remote-controller tests; and
- continuation tests.

## Hardware Gate

The hardware campaign is source-bound and uses a fresh run tag. It compares
the same split-phase K8 policy with the delta journal disabled and enabled.

The fixed row inventory is:

```text
performance:  2 policies x 3 contexts x 10 repetitions = 60 rows
correctness:  2 policies x 3 contexts x 4 sampling points = 24 rows
```

Policy order rotates and reverses by repetition and context so neither arm
owns a fixed warmup or thermal position.

Fixed conditions:

- model: Qwen3-0.6B Stage 1;
- same pushed source commit;
- same A100 GPU;
- same prompts and generated token counts;
- same CUDA Graph and split-phase settings;
- same repetition inventory;
- interleaved paired order;
- no concurrent benchmark controller;
- strict-clean GPU admission;
- exact output and sampled-logit parity; and
- remote plus local independent verification.

The producer wraps the real scheduler phase prepare and commit entrypoints to
record durations without adding production-path timing calls.

Required evidence:

```text
performance rows
correctness rows
per-phase prepare and commit samples
delta-journal lifecycle counters
target forwards and graph replays
D2H calls and bytes
TTFT, TPOT median/P95/P99, E2E, throughput
CUDA peak allocated and reserved bytes
source manifest and patch hash
remote independent verifier
local independent verifier
```

The gate returns `GO_EXACT_BURST_LEASE_LOCAL_DELTA_JOURNAL` only if:

- all exact output and sampled-logit checks pass;
- every eligible candidate phase uses the delta journal;
- baseline phases use the generic journal;
- target forwards, graph replays, D2H calls, and D2H bytes are identical;
- long-context phase-prepare median improves by at least 50%;
- long-context phase-prepare P95 improves by at least 50%;
- at each shorter context bucket, phase-prepare median and P95 do not regress
  by more than 3%;
- aggregate TPOT median and P95 do not regress by more than 3%;
- TTFT and E2E do not regress by more than 3%;
- throughput does not regress by more than 3%;
- CUDA reserved memory does not regress by more than 1%;
- no delta rollback or fallback occurs in eligible performance rows;
- paired stationarity checks pass; and
- both independent verifiers pass.

The report must publish:

- the measured phase-prepare benefit;
- any measured end-to-end TPOT or throughput change, including a neutral
  result;
- Python allocation reduction from the direct-path benchmark;
- GPU memory delta;
- fallback inventory;
- code and state-complexity cost; and
- the remaining complete block-table identity validation cost.

## Rollout and Classification

The feature remains default-disabled after implementation.

Possible final classifications:

```text
GO_EXACT_BURST_LEASE_LOCAL_DELTA_JOURNAL
NO_GO_PERFORMANCE
NO_GO_CORRECTNESS
NO_GO_TRANSACTIONAL_SAFETY
NO_GO_EVIDENCE_INCOMPLETE
```

A GO classification authorizes only the narrow Stage-1 split-phase fast path.
It does not authorize:

- use for terminal suffix commits;
- use for one-phase or ragged smaller bursts;
- removal of the generic scheduler journal;
- removal of full lease block-identity validation;
- a claim of context-independent total scheduler prepare time; or
- a claim of GPU compute improvement.

## Expected Files

The authorized implementation write set is:

```text
tinyvllm/config.py
tinyvllm/engine/scheduler.py
tinyvllm/engine/block_manager.py
tinyvllm/engine/exact_greedy_decode_burst.py
tools/test_model_runner_spec_verify.py
tools/test_scheduler_prepared_postprocess.py
tools/profile_exact_burst_lease_local_delta_journal.py
tools/test_profile_exact_burst_lease_local_delta_journal.py
tools/exact_burst_lease_local_delta_journal_gate.py
tools/test_exact_burst_lease_local_delta_journal_gate.py
tools/exact_burst_lease_local_delta_journal_verify.py
tools/test_exact_burst_lease_local_delta_journal_verify.py
tools/run_exact_burst_lease_local_delta_journal_remote.py
tools/test_run_exact_burst_lease_local_delta_journal_remote.py
docs/superpowers/plans/2026-08-23-exact-burst-lease-local-delta-journal.md
```

The final reconciliation also updates:

```text
AGENT_HANDOFF_STATE.md
docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
```
