# Qwen3.5 TP4 32K Paired Verify Trace Design

## Status

Approved on 2026-08-14 for local implementation planning.

This design is a diagnostic-only follow-up to the failed Qwen3.5 native-MTP
TP4/32K target-KV-offload authority. It does not modify the authority
classification, weaken exact greedy parity, change proposal length, change KV
budgets, or authorize another remote run.

Repository constraints forbid staging, committing, pushing, switching
branches or worktrees, stashing, resetting, cleaning, or terminating unrelated
GPU processes. The design document therefore remains uncommitted.

## Problem Statement

The current TP4/32K batch-1 authority fails exact greedy parity:

```text
baseline:
  [220, 15, 15, 15, 15, 15, 15, 15]

native_mtp:
  [220, 15, 15, 220, 15, 15, 220, 15]
```

The same 32K prompt in batch 4 and the corresponding 16K batch-1 case produce:

```text
[220, 15, 15, 220, 15, 15, 220, 15]
```

For the first divergent native batch-1 step:

```text
proposal:       [15, 15, 2658, 8381]
accepted count: 2
fallback:       220
```

The serial batch-1 decode observation near the divergence has:

```text
token 15 logit:  18.875
token 220 logit: 18.625
top-1 margin:     0.25
```

Existing source inspection has not found evidence of:

- a fallback index error;
- verify-row splitting misalignment;
- proposal lifecycle leakage;
- rank disagreement; or
- incomplete cleanup.

The strongest remaining hypothesis is execution-geometry sensitivity. Serial
batch-1 decode uses `Q=1`, while the first native verify tail uses `Q=3`.
Q-dependent TP4 gated-RMSNorm, projection GEMM, full-attention, or another
batched target path may change near-boundary logits enough to flip the greedy
token.

## Goal

Produce a source-bound paired trace that identifies the earliest aligned
baseline/native target-forward row where logits or side-state lineage diverge.

The trace must answer:

1. Were the baseline `Q=1` row and native verify row evaluated for the same
   logical next-token prediction?
2. Did they use the same input token, absolute position, context length, and
   logical KV block coverage, and was each policy bound to the expected
   logical block generation?
3. What were the top target logits and top-1 margin for each aligned row?
4. Which first-target or tail side-state checkpoint fed the committed
   continuation?
5. Did accepted-prefix, fallback, committed-input, and selected-checkpoint
   metadata agree with the runtime selection semantics?
6. What is the first target-forward stage at which the aligned traces stop
   agreeing?

Passing the diagnostic tests establishes only that the trace is internally
consistent and non-invasive. It does not establish 32K correctness.

## Non-Goals

This design does not:

- fix the parity mismatch;
- create a target-KV snapshot, clone, or fork API;
- run a `Q=1` shadow oracle against native target KV;
- change production verifier math;
- change token sampling, accepted-prefix calculation, or fallback selection;
- change target-KV reserve, materialize, commit, rollback, or offload behavior;
- change Qwen3.5 recurrent side-state prepare/select/apply/seal behavior;
- change `MAX_PROPOSAL_TOKENS=4`;
- change the fixed target-KV contract of 68 GPU blocks, 640 logical blocks,
  and block size 256;
- move proposal KV off GPU;
- rerun remote authority before trace implementation and local validation;
- claim performance, KV8/KV4, a second learned structure, production
  readiness, or Phase 1 completion.

## Considered Approaches

### A. Paired Baseline/Native Trace

Record compact target logits and execution identity for ordinary baseline
decode, native first-target decode, and every native verify row. Join the
records after generation using semantic prediction keys.

This is the selected approach because it is sufficient to locate the first
observable divergence and does not require cloning target KV.

### B. Native `Q=1` Shadow Oracle

Run an additional serial target forward from the exact pre-verify target-KV
and recurrent-state snapshot.

This would isolate Q geometry directly, but the current block manager has no
read-only target-KV fork API. Reusing the live transaction would risk
publishing extra KV, changing offload counters, or perturbing state. This
approach is deferred unless the paired trace remains inconclusive.

### C. Immediate Production Math Change

Force the `Q>1` path to imitate the TP4 `Q=1` gated-RMSNorm or projection
geometry before collecting evidence.

This is rejected because it would encode an unproven root cause and could
hide a row-alignment, KV-identity, or side-state-selection defect.

## Architecture

The diagnostic surface has three bounded components:

1. **ModelRunner trace capture**
   - A default-disabled capture configuration stores rank-zero compact logits
     for ordinary decode, first-target decode, and verify rows.
   - Verify logits are captured before argmax and split with the existing
     `SpecVerifyBatchMetadata` row offsets.
   - Capture does not alter returned tensors, target tokens, or execution
     mode.

2. **Qwen3.5 side-state lineage capture**
   - A default-disabled observer records fingerprints at first-target and tail
     checkpoint creation and records the selected checkpoint index.
   - Fingerprints are read-only hashes of detached CPU byte representations
     plus tensor dtype and shape.
   - The observer does not retain GPU tensors and does not change checkpoint
     ownership or cloning.

3. **32K worker trace assembly**
   - The 32K worker enables capture only for an explicit diagnostic run.
  - It combines ModelRunner rows, existing engine step observations, target-KV
    identity rows, and side-state selection rows into a paired JSON artifact.
   - The ordinary 16K/32K authority path remains unchanged when diagnostics are
     disabled.

No new global singleton, environment-dependent implicit activation, or
authority fallback is introduced.

## Activation Contract

Capture is disabled by default.

The ModelRunner exposes an explicit boolean-controlled diagnostic lifecycle:

```text
enable_spec_verify_trace_recording(True)
drain_spec_verify_trace_rows()
enable_spec_verify_trace_recording(False)
```

Requirements:

- enabling twice is idempotent;
- disabling clears any undrained rows;
- draining returns immutable, tensor-free rows and clears the buffer;
- worker ranks keep no CPU logit buffer;
- rank zero is the only rank that emits compact logits;
- capture state is reset in `finally` blocks after generation failure;
- existing `enable_step_logits_recording()` and `last_step_logits()` behavior
  remains unchanged.

The 32K worker activates the lifecycle only when its diagnostic entry point
is selected. Importing the module or running the authority worker normally
must not enable capture.

## Trace Row Contract

Each target-forward row is tensor-free and contains exactly:

```text
schema
policy
batch_size
engine_step
target_forward_ordinal
stage
execution_mode
sequence_id
query_offset
query_len
row_index
prediction_index
input_token_id
position
context_length
logical_block_identities
logical_block_coverage
top_tokens
top_logits
top1_margin
argmax_token
```

Field semantics:

- `schema` is
  `qwen35.native-mtp-tp4-32k-paired-verify-trace.v1`.
- `policy` is `baseline` or `native_mtp`.
- `stage` is `ordinary_decode`, `first_target`, or `verify_tail`.
- `execution_mode` is the exact ModelRunner execution mode.
- `target_forward_ordinal` is monotonic within one policy/batch cell.
- `query_offset` and `query_len` come from the existing verify metadata;
  ordinary and first-target rows use offset zero and length one.
- `row_index` is the row index within the flattened target forward.
- `prediction_index` is the zero-based output-token position predicted by the
  row after accounting for tokens already committed before that forward.
- `input_token_id`, `position`, and `context_length` describe the target input
  that produced the row's logits. `context_length` is exactly `position + 1`
  for that causal target-input row.
- `logical_block_identities` contains the ordered `(block_id, generation)`
  pairs copied from the already-bound target-KV identity row.
- `logical_block_coverage` contains the ordered logical block ordinals and
  token spans required by the row's context. It is derived from
  `context_length` and the frozen block size, not from physical residency.
- Raw block IDs and generations are diagnostic evidence but are not
  cross-policy join keys. Independent policy cells may allocate different
  logical IDs. The assembler instead requires identical logical coverage and
  validates each raw generation against the identity row bound within its own
  policy cell.
- Physical GPU slots are not recorded as alignment keys because staging
  residency can legitimately differ.
- `top_tokens`, `top_logits`, and `top1_margin` use the existing deterministic
  logit compaction rule: descending logit, then ascending token ID.
- `argmax_token` equals `top_tokens[0]`.

The default `top_k` is five. The diagnostic rejects a smaller value and does
not serialize the full vocabulary.

## Side-State Lineage Contract

Each side-state lineage row contains exactly:

```text
schema
policy
batch_size
engine_step
sequence_id
event
checkpoint_index
committed_input_count
proposal_token_ids
accepted_token_ids
verify_input_count
fallback_target_token
fingerprint
```

Legal events are:

```text
first_target_checkpoint
tail_checkpoint
selected_checkpoint
```

Rules:

- first-target checkpoint index is exactly `1`;
- tail checkpoint indices begin at `2`;
- selected checkpoint index equals `committed_input_count`;
- `committed_input_count` equals
  `1 + min(accepted_count, verify_input_count)`;
- accepted tokens are an exact proposal prefix;
- partial acceptance records
  `fallback_target_token = target_tokens[greedy_accepted_count]`;
- checkpoint fingerprints include every layer's convolution and recurrent
  candidate in stable layer order;
- identical checkpoint tensors, shapes, and dtypes produce identical
  fingerprints;
- no raw tensor values or full tensor payloads enter the artifact.

## Pairing Semantics

Rows are paired only after each policy cell finishes.

The primary semantic key is:

```text
(
  batch_size,
  prompt_index,
  prediction_index,
  input_token_id,
  position,
  context_length,
  logical_block_coverage,
)
```

`sequence_id` is not a cross-policy key because policy cells may allocate
different runtime IDs. The worker maps each sequence ID to the stable
`prompt_index` already used by output rows.

For each baseline row, the assembler requires exactly one native row with the
same semantic key. Missing or duplicate matches are diagnostic failures, not
permission to compare by engine step or row ordinal.

After the semantic join, the assembler retains both policies' raw
`logical_block_identities` and validates:

- each row has enough ordered logical blocks to cover `context_length`;
- each identity exactly matches the generation bound for that policy cell;
- the baseline/native `logical_block_coverage` values are equal; and
- the trace does not claim raw block-ID equality across independent cells.

The paired row records:

```text
baseline_stage
native_stage
baseline_query_len
native_query_len
baseline_top_tokens
native_top_tokens
baseline_top_logits
native_top_logits
baseline_argmax_token
native_argmax_token
argmax_equal
baseline_logical_block_identities
native_logical_block_identities
logical_block_coverage_equal
shared_token_logit_deltas
first_topk_disagreement
```

`shared_token_logit_deltas` contains native minus baseline logit values only
for token IDs present in both compact top-k sets.
`first_topk_disagreement` is true when the ordered top-k token IDs differ or
any shared serialized logit differs exactly. These are observations, not
correctness thresholds. The trace must not claim full-logit equivalence from
compact rows.

## Artifact Contract

The diagnostic entry point writes one JSON document containing:

```text
schema
created_at_utc
source_manifest_sha256
target_manifest_sha256
mtp_manifest_sha256
frozen_contract
cells
first_divergence
limitations
```

Required cells are:

```text
baseline:b1
native_mtp:b1
baseline:b4
native_mtp:b4
```

Each cell includes:

- output rows;
- target-forward trace rows;
- side-state lineage rows;
- existing step observations needed to validate accepted count and fallback;
- rank cleanup summary; and
- a digest over the canonicalized cell payload.

`first_divergence` is either `null` or one paired-row summary. It is selected
by ascending prompt index and prediction index, then by target-forward
ordinal. It reports the first aligned row whose logical block coverage,
ordered top-k token IDs, shared serialized top-k logits, or argmax differs.

The limitations include:

```text
diagnostic_only
full_logits_not_captured
target_kv_shadow_not_established
root_cause_not_established
phase1_not_promotable
performance_not_established
```

## Data Flow

### Baseline

1. The worker adds the frozen prompt rows.
2. Before each `engine.step()`, it starts a target-forward ordinal.
3. Ordinary decode captures rank-zero logits and target-input identity.
4. The worker drains trace rows after synchronization.
5. Existing output and step-observation collection continues unchanged.

### Native MTP

1. First-target decode captures its `Q=1` row and first-target checkpoint
   fingerprint.
2. Proposal execution remains unchanged.
3. Verify captures every flattened `Q>1` row before argmax and splits rows
   using the existing query offsets and lengths.
4. Existing verifier logic calculates accepted prefix and fallback.
5. Existing side-state selection produces `committed_input_count`.
6. The lineage observer records the selected checkpoint index and fingerprint.
7. The worker drains all trace rows after synchronization.

### Assembly

1. Validate every row against its exact schema.
2. Bind runtime sequence IDs to prompt indices within each policy cell.
3. Join baseline and native rows by the semantic pairing key.
4. Validate checkpoint-index and fallback invariants.
5. Compute the first divergence without changing the authority result.
6. Write the diagnostic artifact under a separate diagnostic directory.

## Non-Invasiveness Invariants

With diagnostics disabled:

- no trace rows are allocated;
- no logits are copied to CPU;
- no checkpoint fingerprints are computed;
- production return values and exceptions are unchanged;
- target/proposal KV counters are unchanged;
- source-bound authority artifact schemas are unchanged.

With diagnostics enabled:

- the number and shape of target forwards are unchanged;
- logits are observed only after the production forward has completed;
- no additional target or proposal forward is issued;
- no target-KV reserve, materialize, commit, rollback, H2D, or D2H operation is
  issued;
- no side-state candidate is mutated;
- no production selection row is rewritten;
- output token IDs must exactly equal a diagnostics-disabled run for the same
  local synthetic fixture.

The remote 32K authority remains failed until a later, explicitly authorized
run passes the original exact-parity contract.

## Failure Semantics

The diagnostic fails closed when:

- trace activation is malformed;
- a verify row cannot be split by existing metadata;
- target-input identity is incomplete;
- a target-KV identity or generation is missing or disagrees with the
  policy-local binding;
- a runtime sequence ID cannot be mapped to one prompt index;
- a baseline row has zero or multiple native matches;
- top-k rows are malformed or nondeterministic;
- checkpoint indices violate first-target/tail ordering;
- selected checkpoint index differs from `committed_input_count`;
- accepted tokens are not an exact proposal prefix;
- partial-acceptance fallback disagrees with
  `target_tokens[greedy_accepted_count]`;
- any tensor reaches the serialized artifact;
- capture remains enabled after an exception; or
- enabling diagnostics changes output tokens, target-forward counts, KV
  movement counters, or cleanup inventory in local tests.

An observed logit difference is diagnostic evidence, not a test failure by
itself. Schema, alignment, lineage, and non-invasiveness violations are test
failures.

## Test Strategy

### ModelRunner Unit Tests

Extend `tools/test_model_runner_spec_verify.py` to prove:

- capture is disabled by default;
- worker ranks do not store logits;
- rank zero records first-target and verify-tail compact logits;
- verify rows preserve query offset, query length, input token, position, and
  context length;
- draining is tensor-free and clears the buffer;
- disabling clears undrained rows;
- capture resets after forward failure;
- existing `_last_step_logits_cpu` behavior is unchanged; and
- target tokens returned by `_run_spec_verify_batch()` are unchanged.

### Side-State Unit Tests

Extend the Qwen3.5 speculative-state tests to prove:

- first-target fingerprint uses checkpoint index one;
- tail fingerprints start at checkpoint index two;
- fingerprints are stable for cloned equal candidates;
- changing one candidate value changes the fingerprint;
- selected lineage uses `committed_input_count`; and
- observation does not mutate stored candidates.

### Worker and Gate Tests

Extend
`tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py` to prove:

- diagnostic activation is explicit and default-off;
- trace rows reject missing or extra fields;
- semantic pairing rejects missing and duplicate matches;
- sequence IDs are normalized through prompt indices;
- top-k ties are deterministic;
- first divergence ordering is deterministic;
- selected-checkpoint and fallback invariants are enforced;
- full tensor payloads are rejected;
- the diagnostic artifact is source- and checkpoint-bound; and
- the ordinary authority schema and source inventory remain unchanged.

### Local Regression

The implementation plan must run the focused tests first, then the existing
local regression set:

```bash
python3 -m pytest -q \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/test_qwen35_generic_speculative_tp4_32k_gate.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_model_runner_spec_verify.py
```

It must also run:

```bash
python3 -m py_compile \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py

bash -n \
  tools/run_qwen35_native_mtp_tp4_32k_target_kv_offload_remote.sh

git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/qwen35_speculative_state.py \
  tools/qwen35_native_mtp_tp4_32k_target_kv_offload_worker.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_native_mtp_tp4_32k_target_kv_offload_gate.py \
  docs/superpowers/specs/2026-08-14-qwen35-tp4-32k-paired-verify-trace-design.md
```

No remote execution belongs to this diagnostic implementation plan.

## Evidence Interpretation

The paired trace can support one of these bounded conclusions:

1. **Geometry-linked logit divergence:** aligned rows and checkpoint lineage
   agree, but `Q=1` and `Q>1` logits first diverge at a specific prediction.
2. **Input/KV identity divergence:** rows cannot be semantically paired
   because token, position, context length, logical block coverage, or
   policy-local generation binding differs.
3. **Side-state lineage divergence:** selected checkpoint or fingerprint does
   not match committed-input semantics.
4. **Insufficient compact evidence:** argmax agrees or the decisive token is
   absent from one top-k set; a separately designed full-logit or target-KV
   shadow diagnostic is required.

None of these conclusions is itself a production fix. Any fix requires a new
design grounded in the captured first-divergence evidence.
