# Qwen3.5 TP4 32K H2D Slot-Reuse Causal Diagnostic Design

## Status

The user approved **方案 B** on 2026-08-14:

```text
per-slot CUDA event observation
+ diagnostic-only copy-stream wait control
```

This approval authorizes this local, uncommitted design document. It does not
authorize a GPU, remote, NCCL, or authority workload.

Repository constraints forbid staging, committing, pushing, switching
branches or worktrees, stashing, resetting, cleaning, or terminating unrelated
GPU processes. This document therefore remains uncommitted.

Current classifications remain:

```text
FOCUSED_H2D_GPU_DIAGNOSTIC=NOT_APPROVED
PAIRED_TRACE_REMOTE_DIAGNOSTIC=NOT_APPROVED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=NOT_ESTABLISHED
TP4_32K_EXACT_ROOT_CAUSE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

## Problem Statement

The retained Qwen3.5 native-MTP TP4/32K authority failed exact greedy parity
for batch 1. Artifact inspection then showed that ordinary baseline execution
already has a batch-shape-dependent logit drift before native verification is
needed to explain the mismatch.

The asynchronous target-KV staging path currently establishes:

```text
H2D overwrite on copy stream
  -> current stream waits for h2d_done
  -> current stream reads the staged slot
```

It does not establish the reverse dependency needed before physical-slot
reuse:

```text
prior current-stream read of old slot occupancy
  -> later H2D overwrite of the same physical slot
```

`KVOffloadMVP0._enqueue_d2h_pairs()` explicitly orders copy-stream work after
the current stream, but `_enqueue_h2d_pairs()` only waits for logical-block
D2H completion. Existing `h2d_done` and `d2h_done` dictionaries are keyed by
logical block and cannot represent the lifetime of a reused physical slot.

Static replay shows that dependency-free H2D slot reuse is reachable in both
ordinary baseline shapes:

```text
baseline:b1 first-layer hazardous overwrites: 61..65
baseline:b4 first-layer hazardous overwrites: 448
```

This is compatible with the retained batch-shape drift, but static
reachability is not CUDA causality. A real GPU diagnostic must observe the
cross-stream ordering and then test a narrowly scoped synchronization control.

## Goal

Determine whether asynchronous H2D overwrite of a reused target-KV physical
slot occurs before the previous current-stream read of that slot completes,
and whether ordering those operations removes the retained ordinary-baseline
batch-shape drift.

The diagnostic must answer:

1. Which physical slot and occupancy generation was read?
2. Which later H2D copy replaced that exact occupancy?
3. Did H2D begin before the prior read completed?
4. Does a diagnostic-only copy-stream wait remove every observed unsafe
   overlap?
5. Does the control remove the prediction-index-1 batch-shape logit drift?
6. Does the control preserve exact greedy output parity under the existing
   rule?
7. Are H2D/D2H counts, bytes, batch spans, proposal length, target-forward
   count, and authority inputs unchanged apart from synchronization?

Passing local tests proves only that the diagnostic contract is internally
consistent and disabled by default. Causal support requires a separately
authorized real TP4/32K GPU campaign.

## Non-Goals

This design does not:

- implement a production synchronization fix;
- modify verifier selection, fallback indexing, accepted-prefix semantics,
  target/proposal KV transactions, recurrent side state, Scheduler behavior,
  n-gram, SAM, or unrelated MTP behavior;
- run native MTP, first-target, verify-tail, or a paired baseline/native trace;
- create target-KV snapshots, forks, clones, or shadow forwards;
- add target forwards or replay accepted target tokens;
- change `MAX_PROPOSAL_TOKENS=4`;
- change prompt tokens, output tokens, block size, GPU-slot budget, logical
  block budget, TP size, eager execution, or blockwise window size;
- change H2D/D2H copy selection, coalescing, destinations, source tensors,
  counters, or movement accounting;
- serialize the entire device, current stream, layer, or model step before
  every H2D;
- infer causality from token equality alone;
- claim TP4/32K correctness, production readiness, or Phase 1 completion.

## Frozen Diagnostic Matrix

The diagnostic uses only the ordinary baseline policy from the failed
authority configuration:

```text
policy:                           baseline
tensor_parallel_size:             4
prompt_tokens:                 32768
max_output_tokens:                8
batch sizes:                      1 and 4
decoding:                         exact greedy
max_proposal_tokens:              4, unchanged but unused by baseline
block_size:                     256
gpu_blocks:                      68
logical_blocks:                 640
kv_offload_async_copy:          true
kv_offload_batch_copy:          true
kv_offload_writeback_on_evict: false
kv_offload_blockwise_decode:    true
kv_offload_blockwise_blocks:       8
enforce_eager:                  true
```

The four logical cells are:

```text
observe:b1
observe:b4
control:b1
control:b4
```

`observe` records timing events without adding a current-to-copy dependency.
`control` records the same events and inserts only the per-slot predecessor
wait described below.

The diagnostic must run each cell in a fresh process. A future run
authorization may choose the repetition count, but one repetition cannot be
classified as robust causal evidence. The verifier must report the actual
repetition inventory rather than silently promoting a single observation.

## Considered Approaches

### A. Observation Only

Record current-stream read completion and copy-stream H2D start events without
changing execution.

This can establish actual unsafe overlap, but it cannot show that the overlap
causes the batch-shape drift.

### B. Per-Slot Observation Plus Narrow Wait Control

Record events by physical slot occupancy. In the control cell, make the copy
stream wait for all prior read-completion events associated with the
destination occupancy immediately before overwriting that slot.

This is selected because it supplies both the observation and intervention
needed for causal support while leaving copy selection and target semantics
unchanged.

### C. Global Synchronization

Synchronize the current stream, copy stream, or whole device before every H2D.

This is rejected. It may make the workload green by broadly changing launch
geometry and overlap, cannot identify the missing edge, and would not isolate
physical-slot reuse.

## Architecture

The diagnostic has four bounded components:

1. **Rank-local slot-reuse recorder**
   - A default-disabled owner stores slot occupancy identity, pending CUDA
     timing events, and immutable drained rows.
   - It is owned by `KVOffloadMVP0`; no global singleton is introduced.
   - Activation is explicit with mode `off`, `observe`, or `control`.

2. **Blockwise read-completion hooks**
   - Decode, spec-verify, and prefill call one shared manager hook after all
     K/V reads for a staged window have been enqueued on the current stream.
   - The focused campaign exercises ordinary prefill and decode only. The hook
     contract also covers spec verify because all three consumers share the
     same slot-reuse risk, but the focused worker does not activate the native
     speculative runtime.

3. **H2D overwrite observation/control hook**
   - Eviction captures the old slot occupancy before mappings are replaced.
   - H2D enqueue records a timing-enabled start event for each coalesced span.
   - In control mode only, the copy stream waits for all unique predecessor
     read events before recording H2D start and issuing the unchanged copy.

4. **Focused worker and verifier**
   - A diagnostic-only entry point runs ordinary `baseline:b1` and
     `baseline:b4` in observe/control modes.
   - It captures compact rank-zero logits for prediction indices 0 and 1,
     rank-local slot rows from all four ranks, runtime version metadata, and
     the existing movement/cleanup evidence.
   - It does not import or activate the paired verify trace lifecycle.

## Slot Occupancy Identity

Logical block generation is necessary but insufficient because one logical
block may leave and later re-enter a different physical slot. The diagnostic
therefore assigns a monotonically increasing occupancy generation to each
physical slot.

An occupancy identity is:

```text
(physical_slot, occupancy_generation)
```

Its bound logical identity is:

```text
(logical_block, bound_generation)
```

Rules:

- every assignment of a logical block to a physical slot increments that
  slot's occupancy generation;
- an unoccupied slot has no active occupancy identity;
- moving or reloading the same logical block creates a new occupancy;
- read events are attached to the occupancy observed when the read was
  enqueued;
- an overwrite row captures both old and new occupancy identities;
- an event from an old occupancy must never satisfy a dependency for a later
  occupancy;
- cleanup, rollback, discard, and identity rebinding remove diagnostic
  references without changing production metadata.

The production `bound_generations`, `logical_to_slot`, `slot_to_logical`,
`h2d_done`, and `d2h_done` remain authoritative for runtime behavior. The
diagnostic identity is observational and must not replace them.

## Cross-Stream Event Protocol

### Read Completion

For every blockwise attention window:

1. stage required logical blocks and wait for their H2D completion as today;
2. enqueue all K/V reads from staged slots into `k_dense` and `v_dense`;
3. collect the active occupancy identity for every physical slot read;
4. record one timing-enabled `prior_read_done_event` on the current stream
   after the last K/V read for that window;
5. associate that event with every collected occupancy identity.

One event may cover multiple slots. The recorder stores event identity
separately from slot rows so the verifier can confirm deduplication.

If one occupancy is read repeatedly on the same current stream, the latest
event supersedes earlier events because stream order makes it a completion
barrier for all preceding reads. If multiple current streams ever contribute
reads, the recorder retains the latest event per stream and the overwrite
must consider all unique events. The current implementation is expected to
use one current stream; the diagnostic reports the observed stream inventory
and fails closed on an unrepresented stream.

### H2D Overwrite

Before changing the production mapping for an eviction/reassignment:

1. capture destination physical slot;
2. capture old logical identity and old occupancy generation;
3. snapshot all predecessor read events still associated with that old
   occupancy;
4. perform the existing mapping update and assign a new occupancy generation.

For each coalesced H2D span:

1. collect the predecessor read events for every destination slot in the
   span;
2. in `control` mode only, call `copy_stream.wait_event(event)` once per
   unique predecessor event;
3. record timing-enabled `h2d_start_event` on the copy stream;
4. execute the existing non-blocking H2D copy without changing source,
   destination, shape, coalescing, or counters;
5. record timing-enabled `h2d_done_event` on the copy stream;
6. retain the existing production `h2d_done` completion event behavior
   unchanged.

Diagnostic timing events are separate from production completion events.
They must not be inserted into `h2d_done` or `d2h_done`.

### Timing Resolution

The worker drains timing rows only after generation and cleanup have finished
and after the diagnostic explicitly synchronizes pending diagnostic events.
No synchronization is inserted while resolving observe-mode rows.

For each overwrite with a predecessor read event:

```text
read_done_after_h2d_start_ms =
  h2d_start_event.elapsed_time(prior_read_done_event)
```

Classification:

```text
read_done_after_h2d_start_ms > timing_epsilon_ms:
  UNSAFE_OVERLAP_OBSERVED

abs(read_done_after_h2d_start_ms) <= timing_epsilon_ms:
  ORDERING_AMBIGUOUS

read_done_after_h2d_start_ms < -timing_epsilon_ms:
  READ_COMPLETED_BEFORE_H2D
```

The timing epsilon is serialized in the artifact and fixed by the verifier.
It may not be chosen after inspecting results. A row with missing,
unqueryable, or incomplete events is invalid rather than safe.

## Diagnostic-Only Synchronization Control

The control adds exactly one kind of edge:

```text
prior current-stream slot read completion
  -> copy-stream H2D start for the replacement occupancy
```

Requirements:

- the wait is attached to the physical destination occupancy, not the incoming
  logical block;
- all unique predecessor events for a coalesced span are waited exactly once;
- no wait is added when the destination slot has no predecessor read;
- no current-stream wait, device synchronize, copy-stream synchronize, layer
  barrier, TP collective barrier, or process-group barrier is added;
- H2D/D2H pair inventories and coalesced spans remain identical between
  observe and control cells with the same batch shape;
- production counters retain their existing meaning; diagnostic wait counts
  are reported separately;
- mode `off` allocates no timing events, records no rows, inserts no waits,
  and has no output schema effect.

The control is not a production fix. Even if it removes the drift, a later
design must decide the minimal production ownership and regression matrix.

## Activation Contract

The manager exposes an explicit lifecycle equivalent to:

```text
configure_h2d_slot_reuse_diagnostic(mode)
drain_h2d_slot_reuse_diagnostic()
configure_h2d_slot_reuse_diagnostic("off")
```

Contract:

- valid modes are exactly `off`, `observe`, and `control`;
- construction defaults to `off`;
- enabling starts with empty rows and fresh diagnostic occupancy state;
- enabling while already enabled is rejected unless the same mode has no
  undrained state;
- disabling clears undrained events and rows;
- draining returns immutable, tensor-free rows and clears retained CUDA event
  references;
- failure paths disable and clear in `finally`;
- rank zero and worker ranks all record slot ordering locally;
- only rank zero records compact logits;
- no environment variable, import side effect, authority default, or config
  fallback may enable the diagnostic.

## Trace Schema

### Run Metadata

Every cell records:

```text
schema
mode
policy
batch_size
repetition
world_size
prompt_tokens
max_output_tokens
max_proposal_tokens
block_size
gpu_blocks
logical_blocks
blockwise_blocks
async_copy
batch_copy
writeback_on_evict
enforce_eager
torch_version
torch_cuda_runtime_version
nvidia_driver_version
cuda_device_names
source_tree_sha256
checkpoint_sha256
timing_epsilon_ms
```

The schema is:

```text
qwen35.tp4-32k-h2d-slot-reuse-causal-diagnostic.v1
```

Missing PyTorch, CUDA runtime, or NVIDIA driver version is a hard diagnostic
failure because the retained authority did not record them.

### Read Event Row

Each logical read association contains:

```text
rank
engine_step
attention_stage
layer_index
window_ordinal
current_stream_id
physical_slot
occupancy_generation
logical_block
bound_generation
read_event_ordinal
```

`attention_stage` is one of `decode`, `spec_verify`, or `prefill`. The focused
baseline campaign retains ordinary `prefill` and `decode` rows and rejects
`spec_verify` or any native-MTP stage.

### H2D Overwrite Row

Each destination slot contains:

```text
rank
engine_step
attention_stage
layer_index
window_ordinal
copy_batch_ordinal
copy_span_ordinal
physical_slot
old_occupancy_generation
old_logical_block
old_bound_generation
new_occupancy_generation
new_logical_block
new_bound_generation
read_event_ordinals
h2d_start_event_ordinal
h2d_done_event_ordinal
control_wait_event_ordinals
control_wait_count
timing_status
read_done_after_h2d_start_ms
```

Rows for slots without an old occupancy are marked `NO_PRIOR_OCCUPANCY` and
cannot count as safe or unsafe reuse evidence.

Rows with an old occupancy but no recorded prior read are marked
`NO_PRIOR_READ`. They remain part of the inventory but cannot support the
hazard.

### Compact Logit Row

Rank zero records only prediction indices 0 and 1:

```text
sequence_id
prediction_index
input_token_id
position
context_length
top_tokens
top_logits
top1_margin
argmax_token
```

The compaction rule is the existing deterministic rule: descending logit,
then ascending token ID. `top_k=5` is fixed.

The diagnostic may reuse `enable_step_logits_recording()` and
`last_step_logits()` from the ordinary worker. It must not activate
`enable_spec_verify_trace_recording()` or Qwen3.5 side-state trace recording.

## Observation and Control Invariants

For each batch size, observe and control must have identical:

- prompt token IDs and sequence ordering;
- target checkpoint identity and source-tree identity;
- engine and KV-offload configuration;
- output length request;
- number of target forwards;
- logical block identity inventory;
- H2D copy count and byte count;
- D2H copy count and byte count;
- H2D/D2H batch and span inventories;
- eviction count;
- peak resident block limit;
- final cleanup inventory.

The following may differ:

- CUDA event timestamps;
- diagnostic predecessor wait count;
- wall-clock timing;
- compact logits and greedy outputs, because those are the dependent
  variables under test.

Any movement-inventory difference invalidates the intervention comparison.

## Causal Decision Matrix

### Supported

The diagnostic may classify:

```text
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=SUPPORTED
```

only if all conditions hold:

1. prompt 0 has the exact same token sequence in all four cells;
2. prediction indices 0 and 1 have the exact same semantic identity in all
   four cells: `input_token_id`, `position`, and `context_length`, with
   `context_length == position + 1`;
3. observe mode records at least one `UNSAFE_OVERLAP_OBSERVED` row for the
   exact old physical-slot occupancy later overwritten by H2D;
4. control mode records the corresponding predecessor waits and zero
   `UNSAFE_OVERLAP_OBSERVED` rows;
5. observe mode reproduces the retained prediction-index-1 batch-shape logit
   drift between `b1` and prompt 0 of `b4`;
6. control mode removes that drift under a predeclared numeric comparison and
   yields the same compact top-token ordering and greedy token at index 1;
7. control mode preserves the existing exact greedy output rule for prompt 0
   across `b1` and `b4`;
8. all observation/control invariants and cleanup checks pass on every rank;
9. no native-MTP, paired trace, shadow forward, or extra target forward ran.

Even this classification does not by itself establish that every retained
native-MTP mismatch is explained, nor does it authorize a production fix.

### Rejected

The focused hypothesis is rejected for this campaign if:

- observe mode reproduces the drift but records no unsafe overlap with valid
  timing coverage; or
- control removes all observed overlap while the prediction-index-1 drift and
  exact-output mismatch remain unchanged.

The next step is then the separately approved paired target-forward and
side-state trace, not a broader synchronization change.

### Inconclusive

The result is inconclusive if:

- prompt-0 token identity differs or is missing across the four cells;
- prediction-index-0/1 semantic identity differs or is missing across the four
  cells;
- observe mode does not reproduce the retained drift;
- timing rows are missing, ambiguous, or incomplete;
- runtime/source/checkpoint identity differs;
- movement or target-forward inventories differ between observe and control;
- only global serialization makes outputs agree;
- control changes outputs without a matching observed overlap;
- only final tokens agree while compact index-1 logits remain batch-shape
  dependent; or
- one rank observes an unsupported stream or incomplete event lifecycle.

No inconclusive result may be relabeled as a pass.

## Failure Semantics

- Invalid mode, stale occupancy, generation mismatch, missing event, duplicate
  event ordinal, or unresolved event timing is a hard diagnostic failure.
- A slot mapping change that lacks an occupancy transition is a hard failure
  while diagnostics are enabled.
- Diagnostic buffer overflow is a hard failure; rows may not be silently
  sampled or dropped.
- Any extra H2D/D2H movement, target forward, proposal callback, or paired
  trace row invalidates the cell.
- Any rank failure retains failed artifacts and cannot publish a causal
  classification.
- OOM, NCCL failure, GPU contention, or missing runtime-version metadata is an
  environment or campaign failure, not evidence for or against the
  hypothesis.
- The runner must not kill unrelated GPU processes or weaken the frozen
  configuration to obtain a result.

## Local Test Strategy

Implementation planning must include CPU/dummy-event tests for:

1. default-off mode allocates no events and inserts no waits;
2. occupancy generation increments on every physical-slot reassignment;
3. a read event is bound to the occupancy active at read submission;
4. stale events cannot transfer to a replacement occupancy;
5. repeated reads on one stream retain the latest completion event;
6. reads from distinct streams retain all unique predecessor events;
7. observe mode records but never waits;
8. control mode waits once per unique predecessor event before H2D start;
9. coalesced spans preserve copy pairs and wait deduplication;
10. slots with no prior occupancy or no prior read are classified explicitly;
11. drain returns immutable tensor-free rows and releases event references;
12. disable/failure cleanup clears undrained state;
13. decode, spec-verify, and prefill place the read marker after K/V reads;
14. movement counters and existing production completion events are
    unchanged;
15. the focused worker permits only ordinary baseline cells;
16. the worker captures only prediction indices 0 and 1 with fixed `top_k=5`;
17. the verifier rejects native-MTP, paired-trace, shadow-forward, and
    extra-target-forward evidence;
18. missing PyTorch/CUDA/driver metadata fails closed;
19. movement-inventory differences invalidate observe/control comparison; and
20. causal, rejected, and inconclusive decision matrices are mutually
    exclusive.

Dummy-stream tests validate control-flow ownership only. They cannot establish
real CUDA overlap or causal correctness.

## Separately Authorized GPU Validation

No GPU validation is authorized by this document. If the user later approves
the run, the runner must:

1. use the approved adaptive-ngram source tree and frozen target checkpoint;
2. preflight Kerberos, SSH, four idle GPUs, source identity, checkpoint
   identity, and fresh ports;
3. record PyTorch, CUDA runtime, NVIDIA driver, and device metadata before
   execution;
4. run only the four focused ordinary-baseline cells;
5. retain per-rank event rows, compact index-0/1 logits, movement summaries,
   output tokens, cleanup evidence, and failed artifacts;
6. run independent remote and copied-source local verification;
7. preserve exact greedy parity and all frozen authority inputs; and
8. emit one terminal classification without modifying the original failed
   authority.

## Claim Boundary

Local implementation and tests may establish:

```text
H2D_SLOT_REUSE_DIAGNOSTIC_CONTRACT=ESTABLISHED
DEFAULT_OFF_NON_INVASIVENESS=ESTABLISHED
```

A separately authorized GPU campaign may establish one of:

```text
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=SUPPORTED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=REJECTED
TP4_32K_H2D_SLOT_REUSE_GPU_CAUSALITY=INCONCLUSIVE
```

It may not establish, without further evidence:

```text
TP4_32K_EXACT_ROOT_CAUSE=ESTABLISHED
QWEN35_NATIVE_MTP_TP4_32K_TARGET_KV_OFFLOAD_ESTABLISHED
PRODUCTION_H2D_SLOT_REUSE_FIX=ESTABLISHED
PHASE_1=ACHIEVED
PROMOTION=PROMOTABLE
```
