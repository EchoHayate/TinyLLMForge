# Generic Speculative Runtime TP4 Correctness and Collective Authority

## Status

Approved Phase 1 design for the next gate after the TP1 Qwen3-0.6B
16K/32K blockwise correctness and movement campaign.

This design covers a real TP4 Engine execution authority for the generic
host-side n-gram speculative runtime. It does not claim TP4 performance,
second-model support, learned-drafter support, or Phase 1 completion.

## Goal

Establish source-bound evidence that a real `LLMEngine` configured with
`tensor_parallel_size=4`:

1. produces exactly the same greedy output tokens with the baseline and the
   generic n-gram speculative runtime;
2. executes speculative first-target and verification callbacks on ranks
   `0, 1, 2, 3`;
3. executes the TP collectives required by those callbacks on every rank;
4. receives complete four-rank speculative residency acknowledgements when
   KV offload is enabled;
5. returns complete four-rank KV summaries and cleanup receipts; and
6. fails closed if rank inventory, callback identity, collective identity,
   source identity, output parity, or cleanup is incomplete.

The gate is an execution and collective correctness authority. It is not a
throughput or latency gate.

## Existing Rank Ownership

The current implementation intentionally splits ownership:

- Rank 0 owns scheduling, generic host n-gram proposal generation, greedy
  acceptance, Scheduler publication, and final output token authority.
- `ModelRunner.call()` dispatches first-target and speculative verification
  commands to worker ranks before rank 0 executes the same command locally.
- `run_spec_first_target_batch()` and `run_spec_verify_batch()` execute the
  model forward on every rank, but only rank 0 returns token rows.
- TP collectives are therefore part of every rank's model forward even though
  token selection remains rank-0-only.
- Speculative residency `prepare`, `precommit`, `seal`, and `rollback` use
  acknowledged commands and reject missing, duplicate, malformed, or
  rank-mismatched rows.
- `kv_offload_summaries()` and `LLMEngine.exit()` already provide
  acknowledged rank inventories.
- A model-runner proposal executor remains TP1-only. This gate uses the
  generic host-side `NGramDraftAdapter`; it does not relax that restriction.

No speculative runtime object needs to be installed in worker processes.
Workers execute only the model callbacks and residency operations dispatched
by rank 0.

## Considered Approaches

### 1. Output parity plus cleanup only

Run baseline and n-gram TP4 cells, compare outputs, collect KV summaries, and
require clean shutdown.

This is the smallest change, but it cannot prove that the first-target and
verification callback paths ran on every rank. It also cannot distinguish a
correct TP callback from an accidental rank-0-only path. This approach is
rejected as insufficient authority.

### 2. Separate `torchrun` collective probe

Use a standalone four-process model harness to record collectives, then run
the Engine gate separately.

This proves that the model can execute TP collectives, but it does not bind
those collectives to the generic speculative runtime callbacks. A passing
standalone probe could coexist with a broken Engine callback bridge. This
approach remains useful as a diagnostic, not as the main authority.

### 3. Independent TP4 gate with acknowledged callback profiling

Extend the existing acknowledged decode-internal profiler so
`run_spec_first_target_batch()` and `run_spec_verify_batch()` create explicit
profile scopes. Existing `profile_collective()` call sites then record the TP
collectives under those callback scopes on every rank.

Build a new TP4-only gate, worker, verifier, and remote runner. Keep the TP1
blockwise schema and artifacts unchanged.

This is the selected approach because it binds exact output behavior,
speculative callback execution, TP collective execution, residency
acknowledgements, rank inventory, and cleanup to one real Engine campaign.

## Architecture

### Callback profile scopes

The existing `DecodeInternalProfiler` remains the per-rank recorder and the
existing Engine configure/finalize methods remain the acknowledged collection
surface.

Two ModelRunner callbacks gain explicit profiled scopes:

- `batch_kind="spec_first_target"` for
  `run_spec_first_target_batch()`;
- `batch_kind="spec_verify"` for `run_spec_verify_batch()`.

Both scopes:

- set `is_decode=True`;
- use the canonical sorted sequence IDs to build
  `request_set_sha256`;
- set `active_sequence_count` to the callback row count;
- use `dispatch="eager"` because KV offload forces eager execution in this
  campaign; and
- preserve the existing rule that only rank 0 returns token rows.

The profiler's existing context variable makes current
`profile_collective()` call sites record collectives without changing model
or layer call sites. The ordinary prefill/decode `run()` scopes continue to be
recorded as before.

The profile schema is not renamed. This is a deliberate, minimal extension of
the existing internal execution profile to include decode-adjacent
speculative callbacks.

### Engine cell worker

A new TP4 worker constructs one real Engine per policy cell with:

- `tensor_parallel_size=4`;
- `enforce_eager=True`;
- Qwen3-0.6B;
- generic host-side `NGramDraftAdapter` for the candidate cell;
- greedy sampling;
- KV offload and blockwise KV settings enabled; and
- four GPU indices supplied by the remote preflight.

The initial authority matrix is intentionally bounded:

- policies: `baseline`, `ngram`;
- context length: `4096`;
- batch sizes: `1`, `4`;
- output budget: `8`;
- n-gram size: `3`; and
- maximum proposal tokens: `4`.

This four-cell matrix proves TP4 correctness and rank participation without
duplicating the later 16K/32K performance campaign. The prompt generator and
fixed output budget reuse the generic TP1 gate conventions.

Each cell:

1. configures the acknowledged internal profiler on all four ranks;
2. runs one unrecorded warmup generation;
3. clears reusable prefix state;
4. captures per-rank KV summaries;
5. wraps `_call_speculative_residency_phase()` locally on rank 0 to retain the
   already validated acknowledgement rows returned by the production method;
6. runs one recorded generation;
7. captures the second per-rank KV summaries;
8. finalizes and collects the four-rank callback/collective profile;
9. validates the cell before returning it; and
10. always captures and validates the Engine cleanup receipt.

The wrapper does not replace residency validation. It calls the production
method and records only its successful return value. Any production
validation failure still aborts the cell.

### Orchestrator and independent verifier

The new orchestrator runs each cell in a fresh subprocess. A cell cannot reuse
an Engine, process group, shared-memory command channel, or profiler from
another cell.

The orchestrator writes the final artifact atomically only after:

- all four cells validate;
- baseline and n-gram outputs match exactly for each batch size;
- candidate runtime evidence is non-zero;
- rank and collective evidence validates; and
- every cell has a clean shutdown receipt.

The independent verifier reloads the artifact without importing worker
objects. It recomputes source hashes, prompt digests, output parity, movement
deltas, callback/collective invariants, residency acknowledgement invariants,
and cleanup invariants.

## Authority Schema

The TP4 artifact uses a new schema and claim scope. It must not reuse or
upgrade the TP1 blockwise artifact in place.

Top-level fields:

- `schema_version`;
- `classification`;
- `claim_scope`;
- `limitations`;
- `source_tree_sha256`;
- `model_manifest_sha256`;
- `world_size`;
- `gpu_indices`;
- `cells`;
- `parity`;
- `aggregate_runtime`;
- `aggregate_movement`;
- `rank_authority`; and
- `cleanup_authority`.

Each cell contains:

- policy, context length, batch size, tokenizer, dtype, and prompt digests;
- output token rows;
- rank-0 runtime counters from `last_step_observation`;
- per-rank KV movement deltas;
- successful residency phase acknowledgement rows;
- the finalized four-rank internal profile;
- rank inventory and worker acknowledgement inventory; and
- the cleanup receipt.

The artifact classification is `NOT_PROMOTABLE`. A successful campaign proves
only the stated TP4 generic runtime correctness and collective claim.

## Fail-Closed Invariants

### Output and runtime

- Every cell returns exactly one output row per prompt.
- Every output row contains exactly eight tokens.
- Baseline and n-gram outputs are byte-for-byte equal for the same prompt
  batch.
- Candidate cells have positive proposal rows, proposed tokens, accepted
  draft tokens, first-target callbacks, and verification callbacks.
- Baseline cells have zero speculative callback profile rows.

### Rank inventory

- `world_size` is exactly `4`.
- Engine rank inventory is exactly `[0, 1, 2, 3]`.
- Worker acknowledgement ranks are exactly `[1, 2, 3]`.
- KV summaries, profiles, residency rows, and cleanup rows contain each rank
  exactly once.

### Callback identity

For every candidate cell:

- each rank has at least one `spec_first_target` profile row;
- each rank has at least one `spec_verify` profile row;
- callback row counts match across ranks;
- ordered callback kinds match across ranks;
- request-set hashes, active sequence counts, and dispatch labels match
  across ranks; and
- every callback row references its own enclosing rank.

Timing values are observational and may differ across ranks.

### Collective identity

For each matching speculative callback row:

- every rank records a non-empty collective sequence;
- collective counts match across ranks;
- ordered operation names match across ranks;
- tensor shapes and dtypes match across ranks; and
- each collective references the correct rank, step index, and decode
  ordinal.

Wall-clock and CUDA durations may differ across ranks and are not pass/fail
thresholds.

### Residency acknowledgements

For candidate cells with a speculative tail callback:

- each successful phase contains rows for ranks `[0, 1, 2, 3]`;
- ticket ID, operation, status, sequence IDs, committed identities, rejected
  identities, and empty detail match the production method's contract;
- the successful path contains `prepare`, `precommit`, and `seal` in order;
- rollback is absent on the successful path; and
- rejected speculative blocks never cause host writeback.

The gate does not require a positive committed or rejected block count for
every cell. Boundary placement may legitimately leave both at zero.

### Cleanup

Every cell requires:

- `process_group_destroyed=True`;
- `rank_exit_codes=[0, 0, 0, 0]`;
- `owned_children_remaining=[]`;
- rank cleanup receipts for `[0, 1, 2, 3]`; and
- no stale campaign-owned remote processes after the runner's bounded
  terminal-state check.

## Remote Runner

The remote runner uses only:

- host `sitian@10.232.195.203`;
- `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- `ControlMaster=no`; and
- `ControlPath=none`.

Before launching the campaign it performs a real four-GPU preflight:

1. query visible GPU inventory and free memory;
2. select four distinct safe GPU indices;
3. reject duplicate, missing, or insufficient GPUs;
4. allocate fresh distributed and master ports;
5. verify no campaign-owned stale processes use those ports or GPUs; and
6. persist the selected GPU indices and preflight evidence.

The runner must not assume that the previously validated single GPU 7 path
can expand to TP4.

SSH actions remain serial. The runner uses detached remote execution only for
the long campaign and includes the bounded stale-process terminal-state
reread already proven by the TP1 runner.

## Test Strategy

Implementation follows strict RED then GREEN.

### Dependency-light tests

Add focused tests for:

- profiled speculative first-target scope on rank 0;
- profiled speculative first-target execution with `None` result on a worker
  rank;
- profiled speculative verification scope on rank 0;
- profiled speculative verification execution with `None` result on a worker
  rank;
- callback profile rank-parity validation;
- collective sequence parity validation;
- missing, duplicate, and malformed rank evidence;
- residency phase ordering and four-rank row validation;
- cleanup receipt validation;
- exact baseline/candidate token parity;
- source-hash and model-manifest binding;
- atomic artifact publication; and
- remote runner contract and stale-process terminal-state reread.

Tests that import the large ModelRunner test stubs run in isolated Python
processes so they cannot pollute other test modules.

### Real remote gate

The real campaign must produce:

- a source-bound result artifact;
- independent local verification;
- independent remote verification;
- four-rank callback and collective evidence;
- four-rank KV summary evidence;
- four-rank cleanup evidence; and
- the exact command and selected GPU inventory used for reproduction.

## Validation Boundary

A passing artifact proves:

- real Qwen3-0.6B TP4 Engine execution;
- exact greedy parity between baseline and generic host n-gram speculation for
  the bounded 4K, batch 1/4 matrix;
- all-rank speculative first-target and verification callback execution;
- all-rank TP collective execution inside those callbacks;
- complete successful-path speculative residency acknowledgements;
- complete per-rank KV summary collection; and
- complete process-group and worker cleanup.

It does not prove:

- 16K/32K TP4 performance direction;
- TP4 TPOT, throughput, or memory improvement;
- second-model portability;
- model-runner proposal execution under TP4;
- learned-drafter or MTP plus KV-offload support;
- positive committed or rejected speculative block counts in every cell;
- KV8/KV4 correctness; or
- Phase 1 completion.

## Implementation Order

1. Add RED tests for speculative callback profile scopes.
2. Extend the two ModelRunner callbacks to enter the existing profile.
3. Add RED tests for the new TP4 schema and rank/collective invariants.
4. Implement the independent gate contract and validator.
5. Implement the TP4 worker and residency evidence capture.
6. Implement the orchestrator and independent verifier.
7. Implement the serial Kerberos remote runner and four-GPU preflight.
8. Run dependency-light tests and scoped diff checks.
9. Run the real remote campaign and both verifiers.
10. Update the Phase 1 audit and `AGENT_HANDOFF_STATE.md` with the exact
    authority and remaining limitations.
