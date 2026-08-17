# Qwen3.5 Root Model Owner Promotion Design

## Objective

Promote the hybrid-state ownership boundary from the internal packed layer
stack to the complete `Qwen35PackedForCausalLM` root model.

After this gate, a future production ModelRunner selection can satisfy both:

```text
owner.model is ModelRunner.model
owner.layer_stack is model.layer_stack
```

while still deriving the unique transaction and pool from the exact existing
packed layer stack.

## Owner Shape

Update:

```python
@dataclass(frozen=True)
class Qwen35HybridModelOwner:
    model: Qwen35PackedForCausalLM
    layer_stack: Qwen35PackedHeterogeneousLayerStack
    state_transaction: Qwen35CrossLayerStateTransaction
    pool: HybridStateTensorPool
    runtime_bridge: HybridStateRuntimeBridge
```

The factory accepts only the exact root model type:

```python
def build_qwen35_hybrid_model_owner(
    model: Qwen35PackedForCausalLM,
) -> Qwen35HybridModelOwner
```

It derives:

```text
layer_stack = model.layer_stack
state_transaction = layer_stack.state_transaction
pool = state_transaction.pool
runtime_bridge = HybridStateRuntimeBridge(pool)
```

It creates no second pool and copies no state.

## Coherence Validation

The factory rejects:

- non-root models and root subclasses;
- a non-exact packed layer stack;
- transaction subclasses;
- transaction adapter indices that differ from stack linear-layer indices;
- adapters that do not all reference the exact transaction pool.

ModelRunner binding additionally validates:

```text
owner.model is self.model
owner.model.layer_stack is owner.layer_stack
owner.layer_stack.state_transaction is owner.state_transaction
owner.state_transaction.pool is owner.pool
owner.runtime_bridge.pool is owner.pool
```

All checks remain before mutation.

## Convenience Identity

`bind_current_qwen35_hybrid_model()` continues returning the same pickle-safe
identity row, but `linear_layer_indices` now comes from `owner.layer_stack`.

The current production `Qwen3ForCausalLM` still fails closed before mutation.

## Compatibility Boundary

The old layer-stack-only factory contract is deliberately removed. A layer
stack is not a complete ModelRunner model and retaining dual semantics would
make `owner.model is self.model` ambiguous.

No ModelRunner constructor, checkpoint loader, Engine startup, Scheduler
admission, or GPU path changes in this phase.

## Correctness Tests

Update:

```text
tools/test_qwen35_native_model_owner_binding.py
```

Tests cover:

1. exact root/stack/transaction/pool/storage identity;
2. root and transaction exact-type rejection;
3. layer-stack-only input rejection;
4. forged root/stack/transaction/pool/bridge graph rejection;
5. ModelRunner binding with `self.model` equal to the exact root;
6. identical rebinding and replacement rejection;
7. restore owner/participant pool mismatch rejection;
8. convenience identity row;
9. current Qwen3 production model fails before mutation;
10. Scheduler and Engine remain fail-closed.

## Acceptance Gate

Complete only when:

- focused RED/GREEN passes under Python 3.9 and 3.12;
- transactional root, packed stack, owner factory, restore protocol, and
  Qwen3.5 regressions pass;
- chunked-prefill remains 97 pass / 1 known skip / 0 fail;
- Python 3.9/3.12 compile and `git diff --check` pass;
- staged files remain empty and experiment evidence remains;
- handoff records production model selection/checkpoint loading still missing.

Allowed conclusion:

> TinyLLMForge can derive and bind hybrid-state ownership from a complete
> Qwen3.5 root model while reusing its exact packed layer-stack transaction and
> state pool.

Not established:

- production root-model construction or selection;
- checkpoint loading;
- startup binding/configuration;
- Scheduler admission, GPU correctness, or any performance benefit.

