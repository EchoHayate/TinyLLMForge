# Qwen3.5 Transactional Checkpoint Assignment Design

## Status

Approved under the standing inline-execution direction. This is a bounded
CPU-only correctness gate over already materialized source tensors and the
completed checkpoint binding plan.

## Goal

Execute a `Qwen35CheckpointBindingPlan` against an exact in-memory mapping:

```text
checkpoint source name -> CPU torch.Tensor
```

while preserving:

- declared lossless transforms;
- existing TinyLLMForge custom TP loaders;
- exact direct-buffer TP slicing;
- packed gate/up destination slices;
- tied embedding storage;
- all-or-nothing destination mutation.

Any validation or execution failure must leave every destination tensor
exactly equal to its pre-call value.

## Scope Boundary

This gate begins after source tensors have been materialized by a caller. It
does not open files or import safetensors. It proves the transform, TP-loader,
direct-buffer, and rollback layer independently from file I/O and production
runtime wiring.

The test uses the existing complete two-layer 27-entry CPU fixture at TP=1/2.
It does not allocate the real 4.5 GB checkpoint graph.

## Public API

Create:

```text
tinyvllm/models/qwen35_checkpoint_assignment.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35CheckpointAssignmentResult:
    assigned_bindings: int
    unique_destinations: int
    source_tensors: int


def assign_qwen35_checkpoint_tensors(
    binding_plan: Qwen35CheckpointBindingPlan,
    source_tensors: Mapping[str, torch.Tensor],
) -> Qwen35CheckpointAssignmentResult:
    ...
```

The function accepts only an exact `Qwen35CheckpointBindingPlan`. It returns
only after all writes succeed.

## Source Mapping Contract

The mapping key set must exactly equal:

```python
{
    binding.load.weight.source.name
    for binding in binding_plan.bindings
}
```

Reject missing and unexpected sources. Every value must:

- be a `torch.Tensor`;
- be on CPU;
- have the exact checkpoint metadata dtype;
- have the exact checkpoint metadata shape;
- remain unmodified by assignment.

The executor does not silently cast dtype, move device, reshape, or accept a
local TP shard in place of the full checkpoint tensor.

## Transform Contract

Supported transforms:

```text
identity
squeeze_conv_channel
```

`identity` returns the source tensor unchanged.

`squeeze_conv_channel` requires:

```text
[channels, 1, kernel] -> [channels, kernel]
```

It uses `squeeze(1)` and must be lossless. Any unknown transform fails during
prevalidation.

## Execution by Loader Kind

### `custom_parameter_loader`

Pass the transformed full checkpoint tensor to the callable loader stored on
the bound destination Parameter:

```python
loader(destination, transformed)
```

For packed MLP gate/up bindings:

```python
loader(destination, transformed, packed_slot)
```

The existing loaders perform the correct rank-local behavior:

- `VocabParallelEmbedding`: axis-0 vocabulary shard;
- `ColumnParallelLinear`: axis-0 shard;
- `HeadPairedColumnParallelLinear`: contiguous complete-head-pair shard;
- `SegmentedColumnParallelLinear`: per-segment axis-0 shard and local
  concatenation;
- `MergedColumnParallelLinear`: per-slot axis-0 shard into the packed local
  destination slice;
- `RowParallelLinear`: axis-1 shard.

The executor must not duplicate those algorithms for custom parameters.

### `default_parameter_copy`

Require the transformed source shape to equal the destination shape and copy
directly under `torch.no_grad()`. This covers replicated offset RMSNorm
parameters.

### `direct_buffer_copy`

Compute the exact local tensor before copying:

```text
linear_attention.conv_weight -> transformed axis-0 shard
linear_attention.A_log       -> axis-0 shard
linear_attention.dt_bias     -> axis-0 shard
linear_attention.norm_weight -> replicated
```

The local tensor must match `binding.local_shape`, destination shape, dtype,
and device before mutation.

## Prevalidation Phase

Before snapshotting or writing any destination:

1. validate plan and mapping types;
2. validate exact source coverage;
3. validate every source tensor metadata contract;
4. apply/validate every transform;
5. validate every loader kind and callable;
6. derive direct-buffer local tensors;
7. validate packed-slot presence only for merged gate/up;
8. validate all transformed/local shapes and dtypes;
9. reject meta or non-CPU destinations for this CPU gate.

The phase publishes an immutable internal operation tuple. No destination or
source tensor may be mutated.

## Transaction and Rollback

Snapshot every unique destination tensor once:

```text
id(destination) -> destination.detach().clone()
```

This handles:

- gate and up sharing one packed destination;
- any future multiple bindings targeting one storage object.

Execute operations in binding-plan order under `torch.no_grad()`.

If an operation raises:

1. restore every unique destination from its snapshot;
2. verify no rollback write is skipped;
3. re-raise a `RuntimeError` containing the failing source and target, chained
   from the original exception.

Rollback itself is best-effort across all destinations. If any restore fails,
raise a `RuntimeError` describing rollback failure and chain the first restore
error. The function never returns a partial-success result.

## Read-Only Inputs

The executor may mutate only bound destination tensors during the transaction.
It must not:

- mutate `binding_plan` records;
- mutate source tensors;
- replace destination Parameter/Buffer objects;
- change module registrations;
- change state-pool tensors;
- open files;
- initialize distributed process groups;
- execute model forward.

## Test Strategy

### Positive TP=1/2

Build the existing concrete 27-entry two-layer CPU fixture and deterministic
full-source tensors. For every rank:

- assign all 27 bindings;
- assert exact local values for every destination/slice;
- assert gate/up packed slices;
- assert segmented Q/K/V local concatenation;
- assert head-paired query/gate local rows;
- assert row-parallel axis-1 slices;
- assert convolution squeeze plus axis-0 shard;
- assert F32 stable buffers;
- assert tied embedding/LM-head storage;
- assert sources and pool remain unchanged;
- assert result counts.

### Prevalidation failures

Cover:

- wrong plan type;
- non-mapping sources;
- missing and unexpected source;
- non-tensor source;
- wrong shape, dtype, or device;
- unsupported transform;
- invalid loader kind;
- missing custom loader;
- meta destination.

All destinations must remain unchanged.

### Mid-transaction failure

Inject a custom loader failure after earlier writes. Assert:

- the error identifies source and target;
- every unique destination equals its original snapshot;
- source tensors remain unchanged;
- tied aliases and Parameter/Buffer object identities remain unchanged.

## Non-Goals

This gate does not:

- open safetensors files;
- stream or bound peak source-memory usage;
- assign the real 320-entry/4.5 GB checkpoint;
- move tensors to GPU;
- connect ModelRunner, Engine, or Scheduler;
- run model forward or token/logit equivalence;
- establish performance, cache, memory, compression, or quality gains.

The schema-v2 canonical `NO_GO` remains unchanged.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge can transactionally apply a complete two-layer Qwen3.5
> checkpoint tensor set on CPU at TP=1/2, including the convolution transform,
> custom TP loaders, direct buffer sharding, packed gate/up slices, and full
> rollback after an injected mid-assignment failure.
