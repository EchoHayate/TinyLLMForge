# Qwen3.5 Recurrent Full-Fidelity Capture Design

## Status

- **State:** Approved design, pending written-spec review
- **Date:** 2026-08-06
- **Selected approach:** Detached per-rank capture plus an independent closed-bundle assembler
- **Implementation status:** Not started
- **Execution restriction:** This document does not authorize implementation, SSH, remote directory creation, GPU execution, CUDA initialization, or canonical benchmark execution

## Objective

Create a production path that captures real Qwen3.5 recurrent-state tensors from a strict-P1 execution and assembles them into the immutable
`qwen35.recurrent-full-fidelity-bundle.v1` source bundle already consumed by the recurrent INT8 calibration pipeline.

The path must:

1. preserve every captured recurrent tensor as raw FP32 data;
2. cover every declared rank, workload, and all 18 linear recurrent layers;
3. avoid retaining the full capture set in GPU or CPU memory;
4. fail closed on missing, duplicate, corrupt, rebound, or incomplete artifacts;
5. leave ordinary schema-v1 benchmark behavior and artifacts unchanged when capture is disabled; and
6. keep capture evidence, calibration classification, canonical execution, and final performance claims as separate gates.

## Motivation

The existing calibration contract, calibration producer, and independent verifier already define and validate the immutable
`qwen35.recurrent-full-fidelity-bundle.v1` input. Their current tests prove that calibration works for synthetic, hand-built source bundles.
They do not provide a production producer for recurrent tensors observed during a real strict-P1 run.

The existing strict-P1 benchmark worker and assembler record cache and memory observations but do not preserve full-fidelity recurrent tensor
payloads. Consequently, no real strict-P1 artifact can currently be supplied to the calibration producer without adding a dedicated capture
path.

## Scope

### In Scope

- A default-disabled recurrent calibration capture mode.
- A narrow runtime hook at the point where the raw recurrent state is available.
- Immediate per-tensor transfer and atomic persistence.
- Per-rank staging directories and closed rank manifests.
- An independent assembler that constructs one immutable full-fidelity source bundle.
- Structural, identity, hash, dtype, shape, coverage, and path validation.
- CPU-only unit and integration tests using small tensors and fake rank/workload fixtures.
- A later real strict-P1 capture run, but only after its existing prerequisites and resource gates allow execution.

### Out of Scope

- Changing recurrent computation, cache policy, eviction, quantization, or model outputs.
- Capturing from the snapshot cache after execution.
- A generic runtime tensor-tracing framework.
- Keeping all recurrent states resident for batch publication.
- Modifying ordinary schema-v1 benchmark rows or artifacts.
- Running calibration, P2 canonical, SSH, remote setup, CUDA, or GPU work as part of design approval.
- Claiming accuracy, speed, cache, VRAM, capacity, or end-to-end performance benefits from synthetic fixtures or capture-only evidence.

## Selected Architecture

The selected design has two detached stages:

1. **Per-rank capture:** each worker writes tensors immediately into a rank-owned staging directory and closes that directory with a rank
   manifest.
2. **Independent assembly:** after all ranks have closed successfully, a separate assembler validates the rank manifests and publishes the
   canonical `qwen35.recurrent-full-fidelity-bundle.v1`.

This separates hot-path observation from bundle publication. A benchmark worker never has authority to publish a canonical source bundle, and
the assembler never reads live GPU state.

### Rejected Alternative: Snapshot-Cache Export

Exporting recurrent values from the runtime snapshot cache after execution is rejected because cache entries are subject to eviction,
quantized representation, and lifecycle rules. Such an export would not reliably preserve the raw FP32 state observed at the intended capture
boundary.

### Rejected Alternative: Generic Tensor Tracing

A general tensor-tracing subsystem is rejected because it would broaden the hot-path surface, create unnecessary configuration and lifecycle
complexity, and weaken the guarantee that ordinary benchmark behavior remains unchanged. This design adds only the specific capture hook
required by calibration.

## Enablement and Compatibility

Capture is enabled only by an explicit CLI option:

```text
--recurrent-calibration-capture-dir <path>
```

When the option is absent:

- no capture object is created;
- no tensor is detached or copied for calibration;
- no capture directory or manifest is written;
- no benchmark schema or artifact changes;
- no additional synchronization is introduced; and
- the ordinary schema-v1 execution path remains behaviorally identical.

An empty string, malformed path, pre-existing incompatible capture root, or partial prior run is an error. Capture output must never be
silently mixed with prior artifacts.

## Identity Model

Every capture belongs to one immutable run identity. The identity binds at least:

- `model_manifest_sha256`;
- `source_tree_sha256`;
- `workload_manifest_sha256`;
- `world_size`;
- the declared workload IDs; and
- the declared 18 linear recurrent layer indices.

Each rank receives the same run identity plus its unique integer rank. The rank must satisfy `0 <= rank < world_size`.

The source tree identity for the currently approved design context is:

```text
e265b3ead9d9717d92d8bc0507ac051d93ec22f8403b7929c3625ee4153ccfd7
```

This value records the reviewed context; implementation and execution plans must bind their own actual source-tree identity and must not
silently reuse this value after code changes.

## Capture Boundary

The runtime hook observes the raw recurrent state at the stable boundary after the layer has produced the state required for subsequent
runtime use and before any cache encoding, INT8 conversion, eviction, or destructive reuse can alter its representation.

The hook is observational:

- it does not replace the tensor used by the model;
- it does not modify tensor storage;
- it does not change cache ownership;
- it does not retain a reference after persistence completes; and
- a capture failure terminates the capture-enabled run rather than allowing an unverified partial run to continue as successful.

The proposed API is:

```python
capture_recurrent_state(
    *,
    run_identity,
    rank,
    workload_id,
    layer_index,
    tensor,
) -> CapturedTensorRecord
```

## Per-Tensor Data Flow

For every rank, workload, and linear recurrent layer, capture performs exactly this lifecycle:

1. validate run, rank, workload, and layer identity;
2. reject a duplicate tensor ID before publication;
3. detach the observed tensor;
4. convert it to FP32;
5. make it contiguous;
6. copy it to CPU;
7. serialize it into a temporary file in the rank staging directory;
8. flush and atomically rename the temporary file to its final relative path;
9. compute and record the SHA-256, shape, dtype, and logical byte count from the persisted representation;
10. append the immutable tensor record to rank-local capture state; and
11. immediately release temporary GPU and CPU references.

The implementation must process one tensor at a time. It must not accumulate all recurrent states in GPU memory, CPU memory, a Python list of
tensor objects, or a deferred serialization queue.

The canonical tensor ID is:

```text
rank{rank}:{workload_id}:layer{layer_index}:linear_recurrent
```

Each tensor record contains:

- `tensor_id`;
- `rank`;
- `workload_id`;
- `layer_index`;
- `relative_path`;
- `sha256`;
- `shape`;
- `dtype`; and
- `logical_bytes`.

Captured tensors must be FP32 and rank-3. Paths must be normalized, relative, remain inside the rank staging directory, and be unique under the
capture root.

## Atomic Persistence

Tensor publication uses same-filesystem temporary files and atomic rename. The final path is evidence that serialization completed; the
existence of a temporary file is not.

The capture path must:

- reject symlinks and path traversal;
- reject overwriting any final tensor or manifest;
- create files with deterministic relative names;
- hash the exact bytes that are subsequently validated;
- leave incomplete temporary files outside all manifest inventories; and
- treat fsync behavior consistently with the repository's existing artifact-publication conventions.

If a tensor cannot be fully persisted, no `CapturedTensorRecord` is committed for it.

## Rank Closure

Each rank owns an isolated staging subtree and closes it with:

```python
close_rank_capture(
    *,
    staging_dir,
    expected_workload_ids,
    expected_linear_layer_indices,
) -> RankCaptureManifest
```

Rank closure validates:

- the run identity matches the rank staging identity;
- exactly one tensor exists for every expected workload/layer pair;
- no undeclared workload or layer exists;
- all tensor IDs, paths, ranks, shapes, dtypes, byte counts, and hashes are valid;
- no temporary or untracked payload is accepted as canonical evidence; and
- the rank manifest is published atomically only after all checks pass.

A rank manifest is immutable once published. Reopening, extending, or repairing a closed rank directory in place is forbidden. Recovery
requires a new clean capture root and a new run identity.

## Full-Fidelity Bundle Assembly

The independent assembler API is:

```python
assemble_full_fidelity_bundle(
    *,
    capture_root,
    output_dir,
    model_manifest_sha256,
    source_tree_sha256,
    workload_manifest_sha256,
    world_size,
) -> FullFidelityBundle
```

The assembler:

1. opens only closed rank manifests;
2. validates that ranks are exactly `0..world_size-1`;
3. validates identical run identity across all ranks;
4. validates complete Cartesian coverage of ranks, workload IDs, and the declared 18 linear recurrent layers;
5. reopens each tensor safely and validates its exact bytes, SHA-256, dtype, rank-3 shape, and logical byte count;
6. rejects duplicate IDs, duplicate paths, extra payloads, symlinks, path traversal, and identity rebinding;
7. copies or publishes payloads into a temporary output bundle without trusting worker-supplied absolute paths;
8. constructs the canonical source-bundle document; and
9. atomically publishes the output directory only after complete validation.

The resulting bundle uses the existing immutable schema:

```text
qwen35.recurrent-full-fidelity-bundle.v1
```

Its required top-level fields remain:

- `schema_version`;
- `model_manifest_sha256`;
- `source_tree_sha256`;
- `workload_manifest_sha256`;
- `world_size`;
- `linear_layer_indices`;
- `workload_ids`; and
- `tensors`.

No schema-v2 calibration-specific result is embedded in this source bundle. Calibration remains a downstream CPU-only operation.

## Failure Behavior

Capture and assembly fail closed. The following conditions are fatal:

- capture requested without a complete run identity;
- duplicate rank, workload, layer, tensor ID, or relative path;
- undeclared or missing rank, workload, or layer;
- non-FP32 or non-rank-3 persisted payload;
- hash, shape, dtype, or byte-count mismatch;
- identity mismatch between CLI inputs, rank manifests, and source bundle;
- symlink, absolute path, traversal, or root escape;
- a pre-existing final file or output bundle;
- a rank process exiting before manifest closure;
- assembler observation of any unclosed rank;
- partial copy or publication failure; or
- any attempt to continue canonical publication from a partial capture.

Failures may leave quarantined temporary files for diagnosis, but those files are not evidence and cannot be consumed by calibration. Failed
capture roots must not be repaired or resumed in place.

## Concurrency and Resource Boundaries

- Each rank writes only under its own staging subtree.
- No two ranks may publish the same logical tensor ID or relative path.
- Rank closure is local and does not wait for or mutate another rank's files.
- Assembly starts only after worker execution has ended and all expected rank manifests exist.
- At most one tensor payload is being transformed and serialized per capture call.
- The design does not promise lower capture-time VRAM, RAM, or latency; capture mode is evidence production and may add synchronization and
  transfer overhead.
- Capture must be disabled for canonical P2 performance measurement.

## Integration Boundaries

The implementation plan may add focused components with these responsibilities:

1. **Capture contract and records:** immutable run identity, tensor record, and rank-manifest validation.
2. **Capture writer:** per-tensor transformation and atomic persistence.
3. **Rank lifecycle:** initialization, duplicate tracking, completeness checks, and closure.
4. **Assembler:** independent validation and publication of the existing full-fidelity source bundle.
5. **Worker integration:** explicit CLI enablement and the narrow runtime hook.

The runtime integration should be minimal and default to `None` or an equivalent no-op state. Capture-specific publication logic must not be
embedded into the ordinary strict-P1 assembler.

## Testing Strategy

All implementation tests before an approved real run are CPU-only.

### Contract Tests

- valid run identity, tensor record, and rank manifest;
- malformed hashes, ranks, layers, workloads, paths, shapes, and dtypes;
- duplicate tensor IDs and relative paths;
- required 18-layer coverage;
- deterministic canonical serialization.

### Writer Tests

- `detach -> FP32 -> contiguous -> CPU` conversion;
- one-at-a-time persistence and release;
- byte-exact hashing;
- atomic rename;
- failure before rename;
- refusal to overwrite;
- no record publication for incomplete payloads.

### Rank Closure Tests

- complete rank closure;
- missing and extra workload/layer pairs;
- duplicate records;
- temporary and untracked files;
- immutable closed manifests.

### Assembler Tests

- complete multi-rank assembly;
- missing, duplicate, or out-of-range ranks;
- cross-rank identity mismatch;
- payload tampering after rank closure;
- symlink and path-escape rejection;
- false-root or absolute-path rebinding rejection;
- partial output cleanup;
- compatibility with the existing calibration contract and independent verifier.

### Compatibility Tests

- capture flag absent produces no capture files;
- ordinary schema-v1 benchmark document is byte-for-byte or semantically unchanged according to its existing contract;
- capture-enabled failure cannot be reported as an ordinary successful strict-P1 run;
- existing calibration producer and verifier tests continue to pass.

### Real Validation

Real validation is a separate, explicitly authorized execution phase. It requires:

- correctness prerequisite `PASS`;
- Gate-1 `PASS`;
- a fresh v2 preflight classified `READY`;
- strict-P1 resources available under the fixed GPU `2,4,5,6` policy;
- no ownership conflict with unrelated processes; and
- explicit approval for SSH, remote path creation, and GPU execution.

`BLOCKED_RESOURCES` must stop execution without creating a remote path or launching workers.

## Gate Sequence

The evidence chain is strictly ordered:

1. Execute an approved real strict-P1 run with capture enabled.
2. Assemble the real closed `qwen35.recurrent-full-fidelity-bundle.v1`.
3. Validate the real source bundle through the independent verifier path.
4. Run the existing CPU-only recurrent INT8 calibration on that bundle.
5. Require calibration classification `PASS`.
6. Disable capture.
7. Execute canonical P2 only after all prior correctness, Gate-1, strict-P1, calibration, and fresh preflight gates pass.
8. Require the canonical independent verifier classification `GO`.
9. Only then report speed, cache, VRAM, capacity, or accuracy conclusions.

No later gate can retroactively validate an earlier incomplete artifact.

## Claim Boundary

This design and its future CPU-only tests can establish only that:

- the capture and assembly contracts are well-defined;
- synthetic fixtures are handled correctly;
- corruption and incompleteness are rejected; and
- the produced bundle is structurally compatible with the existing calibration pipeline.

They cannot establish:

- real-model numerical fidelity;
- calibration `PASS`;
- canonical `GO`;
- unchanged model accuracy;
- faster inference;
- lower cache usage;
- lower VRAM usage; or
- higher serving capacity.

Those claims require the complete real evidence chain in the gate sequence above.

## Implementation Preconditions

Implementation may begin only after the user reviews and approves this written spec and a separate implementation plan is produced and
approved. Writing this document alone does not authorize implementation.

The implementation must remain in:

```text
/Users/bytedance/dev/TinyLLMForge-adaptive-ngram
```

It must not modify `/Users/bytedance/dev/TinyLLMForge`, and it must not stage, commit, branch, stash, reset, or clean the existing worktree
unless separately authorized.
