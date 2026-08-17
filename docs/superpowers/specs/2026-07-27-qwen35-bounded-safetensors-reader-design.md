# Qwen3.5 Bounded Safetensors Reader Design

## Status

Approved under the standing inline-execution direction. This is a CPU-only
file-reader correctness gate using temporary small safetensors shards. It must
not read the real 4.5 GB checkpoint or connect production runtime.

## Goal

Materialize exactly the checkpoint sources referenced by a completed
`Qwen35CheckpointBindingPlan`, enforce an explicit byte budget, validate every
loaded tensor against immutable metadata, close every shard handle, and invoke
the transactional assignment executor only after complete materialization
succeeds.

## Public API

Create:

```text
tinyvllm/models/qwen35_checkpoint_reader.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35CheckpointMaterialization:
    source_tensors: Mapping[str, torch.Tensor]
    source_count: int
    shard_count: int
    materialized_bytes: int


@dataclass(frozen=True)
class Qwen35CheckpointLoadResult:
    materialization: Qwen35CheckpointMaterialization
    assignment: Qwen35CheckpointAssignmentResult


def materialize_qwen35_checkpoint_sources(
    binding_plan: Qwen35CheckpointBindingPlan,
    checkpoint_dir: str | Path,
    *,
    max_materialized_bytes: int,
) -> Qwen35CheckpointMaterialization:
    ...


def load_and_assign_qwen35_checkpoint(
    binding_plan: Qwen35CheckpointBindingPlan,
    checkpoint_dir: str | Path,
    *,
    max_materialized_bytes: int,
) -> Qwen35CheckpointLoadResult:
    ...
```

## Path and Budget Contract

`checkpoint_dir` must be an existing directory. Shard names come only from the
already validated binding metadata and must remain safe relative `.safetensors`
paths.

`max_materialized_bytes` must be a positive non-boolean integer. Before any
file opens, sum the exact metadata byte counts for unique requested sources:

```text
BF16 -> 2 bytes per element
F32  -> 4 bytes per element
```

Reject when the required total exceeds the budget. This prevents an accidental
real-checkpoint read in the local gate.

## Requested-Source Contract

Group unique requested source names by shard. Reject:

- duplicate source names with conflicting metadata or shard;
- a missing shard file;
- a shard path escaping `checkpoint_dir`;
- a requested source missing from the opened shard.

Extra tensors in a shard are allowed because the verified real shard also
contains visual and MTP tensors that the language-model plan intentionally
skips. The reader calls `get_tensor()` only for requested source names.

## Tensor Validation

Every loaded tensor must:

- be a CPU `torch.Tensor`;
- have the exact metadata dtype;
- have the exact metadata shape;
- consume the exact metadata byte count.

The reader does not cast, reshape, transform, TP-shard, or assign. Those
responsibilities remain in the transactional assignment executor.

Materialized source tensors are returned in a read-only `MappingProxyType` so
the container cannot be changed accidentally. Tensor values themselves remain
ordinary CPU tensors for the assignment executor.

## File Lifetime and Atomic Boundary

Each shard is opened with:

```python
with safe_open(path, framework="pt", device="cpu") as handle:
    ...
```

The handle must close on:

- success;
- missing source;
- corrupt tensor shape/dtype;
- `get_tensor()` exception.

`load_and_assign_qwen35_checkpoint()` performs:

```text
complete materialization and validation
then
transactional assignment
```

Any materialization failure occurs before destination mutation. Any assignment
failure is rolled back by the completed assignment executor.

## Test Strategy

Use `safetensors.torch.save_file()` to create temporary small shards from the
existing 27-entry two-layer fixture.

### Positive

- split requested sources across two shards;
- include unrelated extra tensors;
- materialize exactly 27 requested sources;
- verify exact values, shapes, dtypes, byte count, and immutable mapping;
- run load-and-assign at TP=1/2 and verify exact destination values through the
  assignment test helpers.

### Budget and metadata failures

Cover:

- invalid budget;
- required bytes above budget before any open;
- missing directory or shard;
- requested source missing from shard;
- wrong tensor shape or dtype;
- conflicting duplicate source contract.

Destinations remain unchanged.

### File-handle cleanup

Wrap the module-level `safe_open` with a tracking context manager that records
entry/exit and delegates to the real implementation. Assert every entered
handle exits on success and representative failures.

### Assignment boundary

Inject a destination-loader failure after successful materialization. Assert:

- every file handle is already closed before the loader runs;
- transactional rollback restores all destinations;
- the contextual assignment error propagates.

## Non-Goals

This gate does not:

- read the real 4.5 GB checkpoint;
- optimize streaming or peak memory below the explicit all-source budget;
- load GPU tensors;
- connect generic `tinyvllm.utils.loader`;
- wire ModelRunner, Engine, or Scheduler;
- execute model forward or token/logit equivalence;
- establish performance/cache/memory/compression/quality gains.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge can read only the requested Qwen3.5 sources from bounded CPU
> safetensors shards, close all file handles, validate immutable metadata, and
> cross the assignment boundary only after complete materialization succeeds.
