# Qwen3.5 Native Model Owner Binding Design

## Objective

Define the missing ownership interface between a real packed Qwen3.5 model
stack and `ModelRunner`, so the runtime bridge and hybrid-prefix restore owner
can only be installed from the model's exact existing state transaction/pool.

This phase does not change ModelRunner model selection. The current production
constructor still builds `Qwen3ForCausalLM`; therefore binding remains an
explicit fail-closed API until a later native Qwen3.5 model loader gate.

## Model Owner

Create:

```text
tinyvllm/engine/qwen35_hybrid_model_owner.py
```

It exposes:

```python
@dataclass(frozen=True)
class Qwen35HybridModelOwner:
    model: Qwen35PackedHeterogeneousLayerStack
    state_transaction: Qwen35CrossLayerStateTransaction
    pool: HybridStateTensorPool
    runtime_bridge: HybridStateRuntimeBridge

def build_qwen35_hybrid_model_owner(
    model,
) -> Qwen35HybridModelOwner
```

The factory accepts only the exact packed heterogeneous layer-stack type and
derives:

```text
state_transaction = model.state_transaction
pool = state_transaction.pool
runtime_bridge = HybridStateRuntimeBridge(pool)
```

It allocates no state tensor and copies no state. The owner keeps strong
references to the complete graph.

Validation requires:

- exact model type;
- exact transaction type;
- transaction adapters remain aligned with model linear-layer indices;
- transaction pool is the unique pool for all adapters;
- runtime bridge pool is that exact pool.

## ModelRunner Binding

Add:

```python
self.qwen35_hybrid_model_owner = None
```

Add:

```python
bind_qwen35_hybrid_model_owner(
    owner: Qwen35HybridModelOwner,
) -> None
```

Binding requires:

- exact owner type;
- `owner.model is self.model`;
- no existing different model owner;
- no existing different runtime bridge;
- no existing restore owner/participant from another pool.

On success:

```python
self.qwen35_hybrid_model_owner = owner
self.hybrid_state_runtime_bridge = owner.runtime_bridge
```

It does not automatically create the snapshot cache or Engine coordinator.
The already completed acknowledged owner factory remains the next call.

Rebinding the identical owner is idempotent.

## Convenience Binding From Current Model

Add:

```python
bind_current_qwen35_hybrid_model() -> dict
```

It calls the pure factory on `self.model`, binds the owner, and returns a
pickle-safe identity row:

```python
{
    "participant_id": int,
    "capacity": int,
    "layout_fingerprint": str,
    "bytes_per_slot": int,
    "linear_layer_indices": tuple[int, ...],
}
```

With the current `Qwen3ForCausalLM`, this fails closed before installing a
bridge. It becomes usable only after a future native model-selection gate sets
`self.model` to the packed Qwen3.5 owner type or a later approved root model
interface.

## Failure Boundary

Binding is one-shot and occurs before any hybrid request allocation. There is
no unbind or replacement API because a runtime bridge may own live slot
bindings.

No Scheduler or Engine automatic call is added in this phase.

## Correctness Tests

Create:

```text
tools/test_qwen35_native_model_owner_binding.py
```

CPU/dependency-light tests cover:

1. owner factory reuses exact model transaction/pool/storage;
2. runtime bridge references the exact pool;
3. invalid model type fails closed;
4. ModelRunner binding requires `owner.model is self.model`;
5. binding installs exact bridge and retains exact owner;
6. identical rebinding is idempotent;
7. different owner or pre-existing different bridge fails closed;
8. pre-existing restore owner/participant pool mismatch fails closed;
9. convenience binding returns an exact pickle-safe row;
10. current non-Qwen3.5 model fails before mutation;
11. Scheduler guard and `LLMEngine.step()` remain unchanged.

## Acceptance Gate

Complete only when:

- focused tests show RED then GREEN under Python 3.9 and 3.12;
- owner factory, live transaction, restore-ticket, ModelRunner methods, ack,
  and wiring tests pass;
- Qwen3.5 packed stack and hybrid CPU regressions pass;
- chunked-prefill remains 97 pass / 1 known skip / 0 fail;
- Python 3.9/3.12 `py_compile` and `git diff --check` pass;
- staged files remain empty and experiment evidence remains;
- handoff records that native model selection/checkpoint loading is still
  missing.

Allowed conclusion:

> TinyLLMForge has a fail-closed ownership interface that can bind a real
> packed Qwen3.5 model's existing state transaction and pool into ModelRunner
> without allocating duplicate state tensors.

Not established:

- ModelRunner selecting or constructing the packed Qwen3.5 model;
- checkpoint loading into that model;
- Engine startup automatically invoking the binding/configuration chain;
- Scheduler admission, GPU correctness, or performance/cache/memory/quality
  benefit.

