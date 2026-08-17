# Autoregressive Draft TP1 Proposal-KV Offload Authority Design

Date: 2026-08-15

Repository: `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`

## Objective

Extend the existing loaded-checkpoint
`tools/autoregressive_draft_tp1_engine_gate.py` so one explicitly enabled TP1
campaign can prove exact greedy target-versus-learned parity while recording
real Proposal-KV residency movement from the production
`ProposalKVResidencyManager` authority snapshot.

This is an authority-harness change. It does not alter verifier selection,
accepted-prefix semantics, target-KV transactions, Scheduler behavior, draft
model execution, or the default runtime configuration.

## Prompt-to-Artifact Gap

The current TP1 gate already provides:

- real target and Qwen3 draft checkpoint fingerprints;
- tokenizer compatibility;
- batch-1 and batch-4 exact greedy output parity;
- real draft-forward counts;
- proposal/target storage separation;
- terminal proposal-slot cleanup; and
- a default `performance_pass_criterion=false` boundary.

It does not currently provide:

- a way to enable learned-drafter Proposal-KV offload;
- the exact logical/GPU/CPU capacity tuple used by the run;
- H2D and D2H operation, entry, and byte deltas;
- accepted-entry copy, replay, and rematerialization deltas; or
- a fail-closed distinction between parity and real bidirectional movement.

The TP4 local gate validates four rank snapshots but does not launch or verify
a loaded TP4 engine campaign. TP4 remains outside this change.

## Considered Approaches

### 1. Extend the existing TP1 gate

Add explicit Proposal-KV offload arguments and collect nested allocator
snapshot deltas in the existing loaded gate.

Advantages:

- reuses the checkpoint, tokenizer, parity, lifecycle, and CLI contracts;
- preserves one authoritative TP1 artifact instead of creating competing
  schemas;
- directly exercises the production runtime wiring completed in the prior
  milestone.

Disadvantage:

- the existing payload schema must advance to version 2.

This is the selected approach.

### 2. Create a separate TP1 offload-only gate

This avoids changing the existing schema, but duplicates model loading,
prompt handling, parity validation, and lifecycle checks. The duplicated
authority logic would be more likely to drift.

### 3. Build the full TP1/TP4 performance campaign now

This would combine distributed launch, performance repetition, resource
isolation, and movement authority in one change. It is rejected because TP4
loaded execution and controlled performance are independent boundaries and
cannot be validated locally.

## Configuration Contract

The public Python and CLI interfaces add:

```text
proposal_kv_offload_enabled: bool = False
proposal_kv_gpu_slot_capacity: int | None = None
proposal_kv_async_copy: bool = True
proposal_kv_batch_copy: bool = True
```

The workload-derived direct capacity remains the existing
`proposal_slot_capacity`.

When offload is disabled:

```text
logical_entry_capacity = proposal_slot_capacity
gpu_slot_capacity = proposal_slot_capacity
cpu_backing_capacity = 0
allocator_mode = direct
```

When offload is enabled:

```text
logical_entry_capacity = proposal_slot_capacity
cpu_backing_capacity = proposal_slot_capacity
0 < gpu_slot_capacity < logical_entry_capacity
allocator_mode = residency
```

The caller must provide `proposal_kv_gpu_slot_capacity` in offload mode.
Supplying it in direct mode is rejected so a stale value cannot be mistaken
for an active residency limit.

The engine receives the existing production fields:

```text
autoregressive_draft_proposal_kv_offload_enabled
autoregressive_draft_logical_entry_capacity
autoregressive_draft_gpu_slot_capacity
autoregressive_draft_cpu_backing_capacity
proposal_kv_async_copy
proposal_kv_batch_copy
```

Target-only baseline engines keep all learned-drafter fields disabled.

## Evidence Contract

For each learned case, the adapter reads:

```text
executor.backend.proposal_kv_cache.entry_allocator
```

before and after execution. The payload records nonnegative deltas for:

```text
h2d_operation_count
h2d_entry_count
h2d_bytes
d2h_operation_count
d2h_entry_count
d2h_bytes
accepted_entry_copy_count
accepted_entry_replay_count
accepted_entry_rematerialization_count
```

It also records:

```text
allocator_mode
logical_entry_capacity
gpu_slot_capacity
```

The batch-1 and batch-4 deltas are summed. Capacity and allocator-mode fields
must remain identical across both cases.

## Terminal Classification

The existing exact-parity/lifecycle gate remains `gate_pass`.

The payload adds:

```text
proposal_kv_offload_enabled
real_proposal_kv_bidirectional_movement
```

`real_proposal_kv_bidirectional_movement` is true only when all of the
following are positive:

```text
h2d_entry_count
h2d_bytes
d2h_entry_count
d2h_bytes
```

It is not inferred from configuration, allocator mode, logical capacity, or
simulated copies.

Offload correctness may pass while bidirectional movement remains false. Such
an artifact is valid diagnostic evidence but does not establish the movement
promotion boundary.

The accepted-entry counters must remain:

```text
accepted_entry_copy_count = 0
accepted_entry_replay_count = 0
accepted_entry_rematerialization_count = 0
```

This protects the transactional Proposal-KV direct-commit boundary.

## Failure Behavior

Validation fails closed for:

- invalid or contradictory direct/offload capacity settings;
- allocator mode inconsistent with the requested configuration;
- negative or decreasing authority counters;
- missing movement counters;
- nonzero accepted-entry copy, replay, or rematerialization;
- exact output mismatch;
- draft-forward absence;
- extra ordinary target decode forwards;
- proposal-slot leaks; or
- proposal and target storage identity collision.

CLI failure writes the existing JSON failure envelope and exits nonzero.

## Test Boundary

Dependency-light tests use fake engines and synthetic authority snapshots to
prove:

1. default direct behavior is unchanged;
2. offload capacities reach the engine factory;
3. movement counters are computed as deltas and merged across cases;
4. parity can pass without claiming movement;
5. positive H2D and D2H establish only the movement field;
6. accepted-entry replay/rematerialization fails closed;
7. invalid capacity combinations fail before engine construction; and
8. CLI preflight records configuration without loading an engine.

No local test is treated as CUDA movement, loaded-checkpoint parity, or
performance evidence.

## Non-Goals

- TP4 launch or loaded TP4 parity;
- 4K/16K/32K campaign execution;
- performance warmup/repetition/statistical analysis;
- target-KV offload changes;
- KV4/KV8;
- heat-tier policy;
- verifier/sampling/commit fusion; or
- CUDA Graph changes.
