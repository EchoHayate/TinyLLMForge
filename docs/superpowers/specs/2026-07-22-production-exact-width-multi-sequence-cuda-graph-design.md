# Production Exact-Width Multi-Sequence CUDA Graph Design

## Status

This document designs the production admission path that follows the
independently verified diagnostic `GO` recorded for:

```text
experiments/cuda_graph/qwen3-06b-heuristic-exact-width-canonical-20260722-055943/
```

The diagnostic proved exact replay correctness and FlashAttention 2.6.3
heuristic compatibility. It did not prove production throughput, latency,
startup, memory, cache-hit, or scheduler behavior.

The existing fail-closed guard remains authoritative until the production
correctness and arrival-load gates in this design both return `GO`:

```python
multi_sequence_decode = mode == "decode" and input_ids.size(0) > 1
```

This design does not authorize a README performance claim or removal of that
guard by itself.

## Objective

Add an opt-in, bounded production path for multi-sequence decode CUDA Graphs
that:

1. reuses only an exact FlashAttention graph identity;
2. preserves the current batch-one CUDA Graph fast path;
3. avoids capturing low-reuse identities;
4. bounds graph count, static-buffer bytes, incremental CUDA reserved memory,
   and capture stalls;
5. fails closed to eager execution on every miss or unsupported condition;
6. exposes enough telemetry to explain every replay and fallback;
7. preserves exact greedy output against an eager baseline;
8. earns a production `GO` only through source-bound arrival-load evidence;
9. makes no rounded-width replay available in production.

## Non-Goals

This stage does not:

- combine CUDA Graph work with Light Doc Cache, Gist KV layer sharing, token
  sparsity, low rank, KV quantization, KV offload, Quest, KV-Cartridge, or
  Attention Matching;
- change FlashAttention source, build flags, or public split tuning;
- support arbitrary FlashAttention versions or devices;
- evict or recapture live graph entries;
- capture graphs asynchronously on another thread or CUDA stream;
- claim capacity reduction from graph replay;
- optimize prefill, speculative verification, hidden-state return, or
  embedding-input paths;
- enable rounded batch or rounded page-table-width reuse.

## Current Production Problem

`ModelRunner.capture_cudagraph()` currently captures:

```text
batch sizes: [1, 2, 4, 8, 16, ...]
page width:  ceil(max_model_len / block_size)
```

All entries share one set of maximum-sized static buffers. Runtime lookup uses
the first captured batch size greater than or equal to the active batch size.
Multi-sequence decode is therefore forced to eager execution because the
captured FlashAttention plan can disagree with the runtime plan, and a larger
zero-padded page table can change the auto-selected split.

The diagnostic established that the safe identity must include both the
effective FlashAttention split and exact page-table width. It also established
that rounded-width replay is corrupt and must remain production-disabled.

## Considered Approaches

### 1. Bounded Exact-Identity Lazy Capture

Start with no multi-sequence entries. Count eager observations of exact
identities, capture only hot identities, and stop admitting entries when any
budget is exhausted.

Advantages:

- spends memory only on observed identities;
- avoids startup capture of unused multi-sequence shapes;
- naturally measures real workload reuse;
- always has an eager fallback.

Costs:

- the request that triggers capture observes a cold capture stall;
- graph admission and telemetry are more complex;
- a hard no-eviction policy can leave later identities eager.

### 2. Startup Capture of a Fixed Allowlist

Capture selected batch and width combinations before the engine becomes ready.

Advantages:

- no capture stalls after readiness;
- simple immutable cache;
- deterministic startup inventory.

Costs:

- width combinations depend on workload context lengths;
- unused entries increase startup time and memory;
- a practical allowlist either has poor coverage or becomes too large;
- production identity drift is harder to expose from real demand.

### 3. Unbounded Lazy Capture with LRU Eviction

Capture every observed identity and evict old entries under pressure.

Advantages:

- highest theoretical coverage;
- adapts to changing workload distributions.

Costs:

- difficult CUDA Graph and graph-pool lifetime management;
- repeated capture can create latency spikes and reserved-memory fragmentation;
- eviction safety is not established in this engine;
- substantially larger correctness and operational surface.

## Decision

Use approach 1 for multi-sequence decode, while preserving the existing
batch-one startup fast path.

When the new feature is enabled, startup capture is restricted to batch one.
Batch-greater-than-one entries are admitted lazily by exact identity. This is
not rounded or hybrid identity reuse: every multi-sequence graph still has an
exact active batch and exact page-table width.

The feature is default-off. With the feature disabled, current behavior and
the current batch-greater-than-one eager guard remain unchanged.

## Configuration

Add one public enable switch and bounded operational controls:

```text
multi_sequence_cuda_graphs: bool = False
multi_sequence_cuda_graph_batch_allowlist: tuple[int, ...] = (2, 4, 8)
multi_sequence_cuda_graph_min_observations: int = 3
multi_sequence_cuda_graph_max_entries: int = 8
multi_sequence_cuda_graph_max_static_bytes: int = 64 MiB
multi_sequence_cuda_graph_max_reserved_bytes: int = 512 MiB
multi_sequence_cuda_graph_max_total_capture_ns: int = 5_000_000_000
multi_sequence_cuda_graph_max_single_capture_ns: int = 2_000_000_000
```

The byte values above are hard safety ceilings, not performance claims. The
production gate can return `NO_GO` before any ceiling is reached.

Validation rules:

- the allowlist is sorted, unique, and contains only integers greater than one;
- `min_observations`, `max_entries`, and both capture limits are positive;
- both byte budgets are positive;
- `enforce_eager=True` overrides the feature;
- incompatible decode features continue to force eager execution;
- no configuration exposes `effective_num_splits` as a tuning knob;
- no configuration permits rounded page-table widths.

The initial allowlist is deliberately narrow. A later allowlist change is a
new source and contract revision that must rerun the full production gate.

## Exact Graph Identity

Production reuses `FlashAttentionGraphIdentity` from
`tinyvllm/engine/flash_attn_split_policy.py`:

```text
graph_batch_size
active_batch_size
page_table_width
effective_num_splits
flash_attn_version
multi_processor_count
num_query_heads
num_kv_heads
head_dim
page_block_size
max_seqlen_q
```

For the first production version:

```text
graph_batch_size == active_batch_size
```

No batch rounding is allowed. The two fields remain explicit so diagnostic and
production identities share one canonical contract and a future design cannot
silently introduce rounding.

Identity construction fails closed unless all of the following hold:

- decode has exactly one query token per active sequence;
- active batch is in the configured allowlist;
- page-table width is the runtime tensor's exact positive second dimension;
- FlashAttention version is exactly `2.6.3`;
- the supported paged-decode shape and GQA invariants validate;
- the runtime SM count matches the identity input;
- the recomputed SHA-256 matches the stored cache entry.

## Components

### `ExactCudaGraphEntry`

One immutable entry owns:

- canonical identity and SHA-256;
- one `torch.cuda.CUDAGraph`;
- graph-private static input, metadata, and output tensors sized exactly for
  its active batch and page-table width;
- capture duration and pre/post allocated/reserved-memory counters;
- replay count and last replay step;
- a terminal state: `ready` or `rejected`.

Static buffers are never shared between identities. CUDA graph-pool use may be
shared because engine execution is serialized, but pool ownership and entry
lifetime must remain explicit.

### `ExactCudaGraphCache`

The cache owns:

- an immutable configuration snapshot;
- observation counts by exact identity SHA;
- ready entries by exact identity SHA;
- permanently rejected identities and rejection reasons;
- cumulative static bytes, reserved-memory delta, and capture duration;
- aggregate hit, miss, fallback, capture, and failure counters.

The first version has no eviction. Once an identity or global budget rejects an
admission, that result is stable for the process lifetime.

### `ModelRunner` Dispatch

`run_model()` keeps all existing eager conditions. Multi-sequence decode
dispatch becomes:

```text
feature disabled or incompatible condition
    -> eager

identity construction failure
    -> eager + fallback telemetry

ready exact entry
    -> copy exact-shaped inputs + replay

identity not ready
    -> eager
    -> record observation
    -> optionally admit capture at the safe post-step boundary
```

The old broad guard is not deleted. It becomes a fail-closed predicate whose
only exception is a ready entry produced by the exact cache.

## Observation and Admission

An observation counts only after a successful eager model step for the exact
identity. Failed, cancelled, unsupported, prefill, or speculative steps do not
increase heat.

With `min_observations=3`, the first three matching steps remain eager. After
the third successful eager result, the engine may capture at a serialized
post-step boundary. The triggering step's user-visible result remains the
already computed eager result; the newly captured graph is eligible only for a
later step.

Before capture, admission checks:

1. identity is not ready, capturing, or rejected;
2. active batch is allowlisted;
3. all compatibility predicates still hold;
4. entry count is below `max_entries`;
5. exact static-buffer byte estimate fits `max_static_bytes`;
6. cumulative capture time is below its budget;
7. current reserved-memory delta is below its budget.

Capture uses graph-private static buffers and dedicated scratch KV write slots.
It must not write to live request KV slots. Scratch capacity is reserved from
the engine's existing KV allocation and is excluded from scheduler allocation.
The source-bound gate freezes one scheduler-visible KV-capacity contract after
that reservation and applies the same visible capacity to baseline and
candidate processes. The candidate is invalid if it exposes fewer request KV
blocks than the paired baseline or if scratch reservation violates the frozen
capacity contract.

The capture path snapshots and restores scratch KV bytes even though the slots
are not live. This preserves the diagnostic's write-lifecycle invariant and
makes repeated verification deterministic.

After capture, the cache synchronizes, records actual duration and CUDA memory
deltas, and validates the identity again. An entry becomes `ready` only if:

- capture raised no exception;
- post-capture identity is unchanged;
- actual static bytes remain within budget;
- incremental reserved bytes remain within budget;
- single and cumulative capture durations remain within budget;
- scratch KV restoration succeeds.

Otherwise the identity is permanently rejected and runtime remains eager.
Destruction of a rejected graph is best-effort; telemetry records any retained
reserved bytes. The engine never relies on allocator reclamation to satisfy a
budget after an overshoot.

## Replay

Replay is allowed only on an exact SHA match. Before replay:

1. recompute the identity from current immutable runtime inputs;
2. check every identity field against the entry;
3. zero or overwrite every graph-private tensor region that can affect output;
4. copy exactly shaped input IDs, positions, slot mappings, context lengths,
   and block tables;
5. set the runtime context to the graph-private tensors;
6. replay and return only the active output rows;
7. reset runtime context in `finally`.

There is no slicing from a larger captured batch and no padding to a larger
page-table width.

Any lookup inconsistency, copy-shape mismatch, context failure, replay
exception, or identity drift disables that entry permanently and immediately
falls back to eager for subsequent steps. A replay exception for the current
step is not silently retried after partial KV mutation; the engine must fail
the step unless it can prove that no live KV write occurred.

## Budget Semantics

The cache enforces all of these independent ceilings:

| Budget | Default | Meaning |
|---|---:|---|
| Ready entries | 8 | Maximum exact multi-sequence graphs |
| Static graph buffers | 64 MiB | Sum of entry-owned tensor storage |
| Incremental reserved CUDA memory | 512 MiB | Peak post-feature delta from the batch-one-ready baseline |
| Single capture duration | 2 s | Maximum synchronized duration for one capture |
| Cumulative capture duration | 5 s | Maximum synchronized duration over process lifetime |

The production benchmark imposes stricter relative limits:

```text
peak reserved-memory ratio <= 1.02
initialization-duration ratio <= 1.05
```

Lazy capture time is not hidden inside initialization. It appears in request
latency, ITL, model-step, and explicit capture-stall telemetry and therefore
can independently cause `NO_GO`.

## Telemetry

Every multi-sequence decode step emits one structured dispatch event:

```text
step_id
request_ids_hash
mode
active_batch_size
page_table_width
effective_num_splits
graph_identity_sha256
feature_enabled
dispatch = eager | graph
cache_state = absent | observing | ready | rejected | budget_exhausted
observation_count
fallback_reason
capture_attempted
capture_duration_ns
capture_static_bytes
capture_allocated_delta_bytes
capture_reserved_delta_bytes
cache_ready_entries
cache_static_bytes
cache_reserved_delta_bytes
cache_total_capture_ns
source_sha256
```

Fallback reasons are a closed enum:

```text
feature_disabled
enforce_eager
unsupported_mode
incompatible_feature
batch_not_allowlisted
identity_invalid
cold_identity
entry_limit
static_byte_budget
reserved_byte_budget
single_capture_budget
total_capture_budget
scratch_unavailable
capture_failed
identity_drift
replay_disabled
```

The run summary reports:

- exact graph hits and hit rate;
- eager fallbacks by reason;
- unique observed, ready, and rejected identities;
- capture attempts, successes, failures, and durations;
- per-entry identity and replay counts;
- static, allocated, and reserved memory deltas;
- source, environment, engine, model, CUDA, GPU, PyTorch, and FlashAttention
  identity.

No production report may infer a hit from batch size alone. A hit requires a
recorded exact identity SHA and `dispatch=graph`.

## Correctness Gate

The production implementation must pass two layers before performance testing.

### Local Contract Tests

Tests cover:

- default-off behavior;
- exact allowlist validation;
- three eager observations before admission;
- exact hit on the first later eligible step;
- no batch rounding;
- no width rounding;
- identity changes when split, width, active batch, model shape, SM count, or
  FlashAttention version changes;
- unsupported identities remain eager;
- every budget independently blocks admission;
- capture failure and identity drift permanently reject an entry;
- no eviction or recapture;
- graph-private buffers are not shared;
- scratch KV slots cannot be allocated to requests;
- context reset on success and exception;
- complete fallback-reason telemetry;
- the original broad guard still fails closed without a ready exact entry.

### Source-Bound Remote Correctness

On `sitian@10.232.195.203`, using GPU 0 and unique dynamic
`TINYVLLM_DIST_PORT` and `MASTER_PORT`, run:

1. the complete 315-case diagnostic matrix against the candidate source;
2. actual `ModelRunner` dispatch for allowlisted batches `2`, `4`, and `8`;
3. non-allowlisted batches adjacent to each allowlisted batch;
4. stable-width trajectories and trajectories crossing a 256-token KV page
   boundary;
5. cold observations, capture, later replay, and every budget fallback;
6. one warmup plus five measured repetitions for each production workload.

For every compared step:

- eager and candidate output tokens must be identical;
- logits must satisfy the frozen diagnostic tolerances;
- KV bytes and hashes must match for live slots;
- graph dispatch must have an exact identity match;
- rounded replay remains independently classified
  `ROUNDED_REPLAY_CORRUPT`;
- the independent verifier recomputes all identities, hashes, summaries, and
  classifications from immutable source-bound artifacts.

Any correctness, completeness, source, identity, or verifier failure is
`NO_GO`.

## Arrival-Load Performance Gate

The performance gate compares two fresh source-bound processes:

```text
baseline:
    production feature disabled
    batch > 1 eager

candidate:
    production feature enabled
    batch-one startup graph preserved
    exact multi-sequence cache enabled
```

Both processes use the same commit, model, prompts, arrival schedule, greedy
settings, GPU, environment, capacity contract, and dynamic-port discipline.
The runner randomizes baseline/candidate order by repetition and records GPU
preflight occupancy. Runs affected by unrelated GPU occupancy are incomplete,
not losses or wins.

Workloads must include:

- stable exact identities with enough reuse to amortize capture;
- mixed batch sizes including non-allowlisted values;
- page-table-width transitions;
- short requests where capture may not amortize;
- long decode requests;
- burst arrivals;
- sustained arrivals near the measured stable service rate;
- long-prompt pressure with concurrent decode.

One warmup plus five isolated measured repetitions are required. The aggregate
report includes cold capture periods; a separate stable-exact slice begins
only after an entry is ready.

Production `GO` requires every measured case to preserve exact output and all
of these frozen thresholds:

```text
aggregate decode throughput ratio       >= 1.15
stable-exact decode throughput ratio    >= 1.25
minimum per-case request throughput     >= 0.95
maximum per-case p95 ITL ratio          <= 1.05
maximum per-case p99 ITL ratio          <= 1.10
peak reserved-memory ratio              <= 1.02
initialization-duration ratio           <= 1.05
stable-exact graph hit rate             >= 0.60
```

Additionally:

- at least one allowlisted batch and two distinct exact page widths must replay;
- at least one non-allowlisted batch must be observed falling back to eager;
- no unknown fallback reason is allowed;
- no graph replay may occur after an entry is rejected;
- all five measured repetitions must be complete;
- recorded and independently recomputed summaries must match.

These thresholds are conjunctive. A throughput gain cannot compensate for a
latency, memory, initialization, correctness, completeness, or provenance
failure.

## Classification and Rollout

The independent report returns exactly one classification:

```text
GO
NO_GO
INCOMPLETE
```

`GO` authorizes a follow-up change that:

- retains the already tested narrow exact-cache exception;
- keeps the broad batch-greater-than-one guard for every path without a
  feature-enabled ready exact entry;
- keeps the feature default-off;
- documents the opt-in configuration and canonical evidence;
- updates README only with independently verified measured claims.

`NO_GO` keeps all multi-sequence production decode eager. The implementation
may retain the narrow candidate path only if it is unreachable by default and
is clearly marked experimental. It must not be enabled by README guidance.

`INCOMPLETE` means evidence is missing or unverifiable and grants no production
authorization.

A default-on rollout is explicitly out of scope. It requires a later design
with broader model, GPU, workload, sampling, concurrency, and soak coverage.

## Artifact Contract

The canonical run directory contains:

```text
manifest.json
environment.json
source_manifest.json
dispatch_events.jsonl
capture_events.jsonl
request_metrics.jsonl
model_step_metrics.jsonl
memory_trace.jsonl
correctness_rows.jsonl
case_summaries.json
summary.json
report.md
independent_verification.json
```

The manifest binds:

- source commit and source tree SHA-256;
- exact local files copied to the remote run;
- model path and model-config SHA-256;
- runner and verifier commands;
- workload and arrival schedule hashes;
- baseline/candidate order;
- process IDs and unique ports;
- all configuration and threshold values.

The independent verifier must reject missing, duplicate, mixed-source,
reordered, non-finite, hash-inconsistent, or threshold-inconsistent evidence.

## Implementation Boundaries

Expected production changes are limited to:

```text
tinyvllm/config.py
tinyvllm/engine/model_runner.py
tinyvllm/engine/flash_attn_split_policy.py
tinyvllm/engine/exact_cuda_graph_cache.py
tools/multi_sequence_cuda_graph_contract.py
tools/run_multi_sequence_cuda_graph_production_gate_remote.py
tools/verify_multi_sequence_cuda_graph_production.py
tools/test_multi_sequence_cuda_graph_gate.py
tools/arrival_load_gate.py
AGENT_HANDOFF_STATE.md
```

The cache should be a separate module rather than further expanding
`model_runner.py`. `ModelRunner` remains responsible for preparing runtime
inputs and executing the model; the cache owns identity lifecycle, budgets,
entries, and telemetry.

The implementation must use selective staging, must not stage `experiments/`,
must not modify the remote checkout, and must not update README before an
independent production `GO`.

The candidate implementation may add a narrow feature-enabled exception to the
existing `multi_sequence_decode` predicate so the real `ModelRunner` path can
be tested. This is not removal of the guard: with default configuration, on a
cache miss, or without a ready exact entry, the predicate continues to force
eager execution exactly as before.

## Acceptance Checklist

The design is implemented only when evidence maps directly to every item:

1. feature default is off;
2. disabled behavior is unchanged;
3. batch one retains its current fast path;
4. multi-sequence identity is exact and heuristic-equivalent;
5. rounded batch and width replay are impossible;
6. three successful eager observations precede capture;
7. capture uses non-live scratch KV slots;
8. graph entries own independent static buffers;
9. all five ceilings fail closed: entry count, static bytes, reserved bytes,
   single-capture duration, and cumulative capture duration;
10. no eviction or recapture exists;
11. every dispatch has an auditable reason;
12. local contract tests pass;
13. 315-case diagnostic remains `GO`;
14. actual `ModelRunner` correctness passes;
15. source-bound arrival-load evidence passes every frozen threshold;
16. independent verification reconstructs the same result;
17. the broad production guard remains authoritative except for the
    feature-enabled ready exact entry exercised by the gate;
18. README remains unchanged until that `GO`.

## Claim Boundary

Before the production gate, the strongest valid statement is:

> Heuristic-equivalent exact-width multi-sequence CUDA Graph replay is
> correctness-compatible in the diagnostic matrix and has a bounded,
> fail-closed production design.

Only after an independent production `GO` may the project claim a measured
throughput or latency improvement, and that claim must name the exact model,
GPU, workload, source, and gate ratios.
