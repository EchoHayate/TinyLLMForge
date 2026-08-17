# Qwen3.5 Rank Checkpoint Candidate Target Factory Design

## Objective

Compose the validated Qwen3.5 concrete component assembly and checkpoint
binding planner into one rank-specific, one-shot candidate target while
preserving a caller-supplied hybrid-state pool as the sole state owner.

The factory performs no checkpoint payload read or assignment.

## Interface

Create:

```text
tinyvllm/models/qwen35_checkpoint_candidate_factory.py
```

with:

```python
@dataclass
class Qwen35PreparedCheckpointCandidateTarget:
    assembly: Qwen35ConcreteComponentAssembly
    binding_plan: Qwen35CheckpointBindingPlan
    pool: HybridStateTensorPool

    def take(self) -> tuple[
        Qwen35PackedForCausalLM,
        Qwen35CheckpointBindingPlan,
    ]:
        ...


def prepare_qwen35_checkpoint_candidate_target(
    hf_config,
    tensor_plan,
    *,
    pool,
    tensor_parallel_size,
    tensor_parallel_rank,
    build_attention_backend,
    parameter_device="meta",
) -> Qwen35PreparedCheckpointCandidateTarget:
    ...
```

## Composition

The function validates exact `HybridStateTensorPool` and
`Qwen35CheckpointTensorPlan` inputs, then calls:

```text
build_qwen35_concrete_component_assembly(...)
build_qwen35_checkpoint_binding_plan(...)
```

It verifies:

- assembly pool is the exact supplied pool;
- supplied pool layout matches the config, TP context, and its existing
  speculative convolution width;
- assembly TP size/rank match the request;
- binding-plan TP size/rank match the request;
- every binding destination is registered by the assembled model;
- every binding destination device matches `parameter_device`;
- the supplied pool storage identities and values are unchanged.

The factory never constructs another state pool.

The factory derives the existing speculative-token width from the supplied
pool's linear-convolution component shape, then rebuilds only the immutable
layout description for canonical fingerprint comparison. It does not require
`speculative_tokens=1` and does not allocate state tensors.

## One-Shot Target

`take()` returns the exact `(model, binding_plan)` tuple expected by existing
streamed and tiled checkpoint loaders.

It is one-shot. A second call fails closed. This prevents accidental reuse of a
possibly partially assigned target after a failed load.

The target stores no checkpoint directory, authorization, or payload bytes.

## Device Boundary

`parameter_device="meta"` supports complete topology/binding preflight without
checkpoint parameter allocation.

`parameter_device="cpu"` prepares a materializable target for a future
authorized local loader. This gate does not allocate the real 4.5 GB model in
tests; CPU behavior is covered with a compact config fixture.

CUDA is rejected by the existing concrete assembly factory.

## Failure Atomicity

Any assembly or binding failure returns no prepared target. The supplied pool
must retain the same tensor objects, storage pointers, values, capacity, and
layout identity.

`take()` changes only the target's consumed flag. It does not mutate model,
binding, or pool state.

## Tests

Tests prove:

- compact TP=1/2 meta and CPU composition;
- exact model/binding/pool identity;
- exact destination device;
- one-shot `take()` tuple compatibility;
- no second pool allocation;
- pool storage/value preservation;
- malformed tensor plan, TP mismatch, backend failure, and binding failure
  return no target and preserve the pool;
- a valid wider speculative-state layout is accepted without constructing a
  second pool;
- real 24-layer metadata composes all 320 bindings on meta at TP=1/2 without
  opening safetensors payloads.

## Claim Boundary

Passing proves production-shaped rank target preparation. It does not load
payloads, implement the real local loader, execute a remote worker, run model
forward, or establish speed or memory benefit.
