# Qwen3.5 Streamed Fresh Checkpoint Loading Design

## Status

Approved under the standing inline-execution direction. This is a CPU-only
correctness and ownership-boundary gate. It uses temporary small safetensors
shards and must not read the real 4.5 GB checkpoint, start GPU work, or connect
production ModelRunner, Engine, or Scheduler.

## Goal

Load a Qwen3.5 checkpoint into a fresh unpublished CPU model assembly while
holding at most one checkpoint source tensor at a time. A failed load may leave
the private candidate partially written, but it must never mutate or replace a
published owner. Only a completely loaded candidate may cross a separate,
one-shot publication boundary.

## Why the Existing Transaction Is Insufficient

The completed bounded reader retains all requested source tensors before
assignment. The completed transactional assignment also clones every unique
destination for rollback. For the real language-model payload this can
approach:

```text
model destinations + all source tensors + all rollback snapshots
```

The source payload alone is `4,548,144,832` bytes. The largest source is the
BF16 embedding tensor:

```text
model.language_model.embed_tokens.weight
shape=[248320, 2048]
bytes=1,017,118,720
```

Reducing only source retention would still leave the full rollback copy.
Therefore the transaction boundary moves from in-place mutation of a live
model to construction of a fresh unpublished candidate.

## Considered Approaches

### A. Stream into a live model and retain full rollback snapshots

This lowers source retention but preserves a model-sized rollback allocation.
It also risks exposing partially written live weights if the transaction
boundary is violated. Rejected.

### B. Stream into a live model and journal only changed slices

This remains proportional to the complete model because every destination is
eventually changed. Packed destinations and custom loaders also make a correct
minimal journal unnecessarily complex. Rejected.

### C. Stream into a fresh unpublished assembly and discard on failure

The candidate owns its destination storage before any source opens. Sources
are loaded, validated, assigned, and released one at a time. Failure discards
the candidate rather than restoring its destinations. Success returns an
opaque loaded candidate that can be published through a separate one-shot
slot. Selected.

## Public Loading API

Create:

```text
tinyvllm/models/qwen35_checkpoint_streaming.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35StreamedCheckpointLoadStats:
    assigned_bindings: int
    source_tensors: int
    shard_count: int
    loaded_bytes: int
    peak_source_bytes: int


@dataclass(frozen=True)
class Qwen35LoadedCheckpointCandidate:
    owner: Qwen35HybridModelOwner
    binding_plan: Qwen35CheckpointBindingPlan
    stats: Qwen35StreamedCheckpointLoadStats


def load_qwen35_fresh_checkpoint_candidate(
    candidate_factory: Callable[
        [],
        tuple[
            Qwen35PackedForCausalLM,
            Qwen35CheckpointBindingPlan,
        ],
    ],
    checkpoint_dir: str | Path,
    *,
    max_tensor_bytes: int,
) -> Qwen35LoadedCheckpointCandidate:
    ...
```

`candidate_factory` is invoked exactly once inside the load call. It must
return a new exact packed model and its exact binding plan. The loader does not
accept a caller-supplied live model as a direct argument and does not publish
the candidate during construction or assignment.

The returned owner is built only after every source has been assigned and all
file handles have closed.

## Candidate Validation

Before any shard opens:

- `candidate_factory` must be callable;
- it must return an exact two-item tuple;
- the model must be an exact `Qwen35PackedForCausalLM`;
- the binding plan must be an exact `Qwen35CheckpointBindingPlan`;
- all destinations must be CPU, non-meta tensors registered by that model;
- every unique source name must have one consistent shard and metadata
  contract;
- every required source byte count must be less than or equal to
  `max_tensor_bytes`;
- shard paths must be safe relative `.safetensors` paths below the checkpoint
  directory.

The test gate uses a factory that creates a new model object on every
invocation. Production wiring remains absent, so no production object can
accidentally pass through this API in this gate.

## One-Source-at-a-Time Data Flow

Unique source contracts are grouped by shard. Shards and source names are
processed in deterministic sorted order:

```text
build fresh candidate
validate complete candidate/source/path contract
for each required shard:
    open shard in a context manager
    for each requested source in that shard:
        materialize one CPU tensor
        validate exact shape, dtype, and byte count
        prepare every binding that consumes this source
        assign those bindings into the private candidate
        release all source and transformed references
close shard
build owner from the fully loaded candidate
return loaded candidate
```

The result contains only the owner, immutable binding plan, and scalar stats.
It never retains source tensors.

`peak_source_bytes` is the largest metadata byte count observed among loaded
sources. `loaded_bytes` is the sum for all unique loaded sources. The hard
budget is per source:

```text
source_bytes <= max_tensor_bytes
```

This proves bounded checkpoint-source retention, not total process RSS.
Destination parameters, custom-loader temporaries, safetensors internals, and
allocator behavior remain outside this scalar bound.

## Per-Source Assignment

Refactor the completed assignment module so both assignment modes share the
same validation, transform, loader, packed-slot, TP-sharding, and direct-buffer
rules.

Add an internal source-group primitive that:

- receives all bindings for exactly one source;
- validates the source once against immutable metadata;
- prevalidates every operation in the group before its first write;
- executes the existing custom/default/direct loader behavior;
- returns the number of assigned bindings;
- does not snapshot or roll back destinations.

The existing public `assign_qwen35_checkpoint_tensors()` keeps its complete
prevalidation and rollback semantics unchanged. The streaming loader uses the
internal no-rollback primitive only because the model is unpublished and
discardable.

If a custom loader fails after mutating the candidate, raise a contextual
stream-load error containing source and target. Do not attempt rollback.

## Publication Boundary

Create:

```text
tinyvllm/engine/qwen35_hybrid_model_publication.py
```

with:

```python
class Qwen35HybridModelOwnerPublicationSlot:
    @property
    def owner(self) -> Qwen35HybridModelOwner | None:
        ...

    def publish(
        self,
        candidate: Qwen35LoadedCheckpointCandidate,
    ) -> Qwen35HybridModelOwner:
        ...
```

The slot starts empty. `publish()`:

1. validates the exact loaded-candidate type and coherent owner graph;
2. rejects publication when the slot is already occupied;
3. performs one final Python reference assignment;
4. returns the published owner.

No method clears or replaces an occupied slot. A failed load yields no loaded
candidate, so it cannot call `publish()`. An occupied slot remains unchanged
after invalid or repeated publication attempts.

This is a pure CPU ownership primitive. ModelRunner's existing one-shot owner
binding is not modified or invoked.

## Failure and Discard Semantics

Failures before source assignment leave the fresh candidate untouched.
Failures during source assignment may leave only that private candidate
partially written. In every failure case:

- all entered safetensors handles close;
- no loaded candidate is returned;
- no publication occurs;
- an existing publication slot owner remains unchanged;
- no rollback snapshot is allocated.

The caller and tests may retain diagnostic references to a failed candidate,
but such references are not publishable because no
`Qwen35LoadedCheckpointCandidate` was produced.

## Test Strategy

Use the existing two-layer, 27-source CPU fixture and temporary two-shard
safetensors files.

### Streaming success

For TP=1 rank 0 and TP=2 ranks 0/1:

- the factory is invoked exactly once and creates a fresh model;
- all 27 sources and 27 bindings load with exact expected local values;
- only one source tensor is retained by the streaming loop at a time;
- every shard handle closes;
- the returned owner preserves model/layer-stack/transaction/pool identity;
- tied embedding storage remains shared;
- stats report exact source, binding, shard, total-byte, and peak-byte counts.

### Pre-open failures

Cover invalid factory, malformed factory result, non-CPU destination,
conflicting source contract, unsafe/missing shard, invalid budget, and a source
larger than the per-tensor budget. Assert no shard opens and no publication
occurs.

### Stream failures

Cover missing source, wrong shape/dtype, and an injected late custom-loader
failure. Assert balanced handle cleanup, contextual errors, no loaded result,
and no rollback restoration of the private candidate.

### Publication

- an empty slot publishes one successfully loaded candidate;
- the slot stores the exact owner object;
- invalid candidate publication leaves it empty;
- an occupied slot rejects a second candidate and preserves the first owner;
- a failed stream load leaves a pre-existing occupied slot unchanged.

## Non-Goals

This gate does not:

- read or assign the real 320-entry/4.5 GB checkpoint;
- prove total RSS or allocator peak;
- reduce the 1,017,118,720-byte largest-source requirement;
- implement tensor slicing inside one safetensors tensor;
- load GPU tensors or execute native model forward;
- compare checkpoint tokens or logits;
- wire generic loader, ModelRunner, Engine, or Scheduler;
- change the supplied state-pool ownership contract;
- establish any speed, cache, memory, compression, or quality improvement.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge can build a fresh unpublished CPU Qwen3.5 candidate, stream
> requested safetensors sources into it one tensor at a time without a
> model-sized rollback copy, discard failed candidates, and publish only a
> completely loaded owner through a separate one-shot slot.

