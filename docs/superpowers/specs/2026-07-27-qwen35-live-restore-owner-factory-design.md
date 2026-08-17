# Qwen3.5 Live Restore Owner Factory Design

## Objective

Automatically construct the rank-local Qwen3.5 hybrid-prefix restore owner
from an already installed live `HybridStateRuntimeBridge`, then let
`LLMEngine` configure every rank through the acknowledged command channel and
install the central Engine restore coordinator.

This phase does not construct the runtime bridge itself. The current production
`ModelRunner` still instantiates `Qwen3ForCausalLM`, not the Qwen3.5 packed
heterogeneous model, so silently allocating a second pool from HF config would
duplicate state ownership and be incorrect.

Scheduler admission remains fail-closed.

## Ownership Rule

The only accepted state owner is:

```text
ModelRunner.hybrid_state_runtime_bridge.pool
```

The factory derives every other object from that exact pool:

```text
pool
  -> one Qwen35LayerStateAdapter per linear layer
  -> Qwen35CrossLayerStateTransaction
  -> Qwen35HybridPrefixSnapshotCache
  -> Qwen35HybridPrefixRestoreParticipant(rank)
```

No state tensor is allocated or copied by the factory.

## Rank-Local Factory Module

Create:

```text
tinyvllm/engine/qwen35_hybrid_prefix_owner.py
```

It exposes:

```python
@dataclass(frozen=True)
class Qwen35HybridPrefixRestoreOwner:
    pool: HybridStateTensorPool
    adapters: tuple[Qwen35LayerStateAdapter, ...]
    state_transaction: Qwen35CrossLayerStateTransaction
    snapshot_cache: Qwen35HybridPrefixSnapshotCache
    participant: Qwen35HybridPrefixRestoreParticipant
    max_entries: int
    max_bytes: int

def build_qwen35_hybrid_prefix_restore_owner(
    pool,
    *,
    participant_id,
    max_entries,
    max_bytes,
) -> Qwen35HybridPrefixRestoreOwner
```

The factory validates:

- exact pool type;
- non-negative participant rank;
- positive cache limits;
- every layout layer has exactly one `linear_convolution` and one
  `linear_recurrent` component;
- no unsupported component role exists;
- at least one complete linear layer exists;
- adapters are ordered by layer index;
- transaction/cache/participant all reference the same pool.

The returned owner keeps strong references to the full object graph.

## ModelRunner Configuration

Add:

```python
self.qwen35_hybrid_prefix_restore_owner = None
```

Add an acknowledged-call target:

```python
configure_qwen35_hybrid_prefix_restore_owner(
    max_entries,
    max_bytes,
) -> dict
```

Behavior:

1. require `hybrid_state_runtime_bridge`;
2. build the owner from `runtime_bridge.pool`;
3. install the owner's participant using the existing one-shot API;
4. retain the owner;
5. return a pickle-safe identity row.

Identity row:

```python
{
    "participant_id": int,
    "capacity": int,
    "layout_fingerprint": str,
    "bytes_per_slot": int,
    "max_entries": int,
    "max_bytes": int,
}
```

Reconfiguration with identical pool and limits is idempotent. Any different
limits, pool, participant, or owner object fails closed.

No bridge means no allocation and an explicit error.

## Engine All-Rank Factory

Add:

```python
configure_qwen35_hybrid_prefix_restore(
    *,
    max_entries,
    max_bytes,
    timeout_s,
) -> Qwen35HybridPrefixEngineRestoreCoordinator
```

It calls:

```python
call_model_runner_acknowledged(
    "configure_qwen35_hybrid_prefix_restore_owner",
    max_entries,
    max_bytes,
    timeout_s=timeout_s,
)
```

Then validates:

- exact identity-row fields and types;
- inner participant ID equals outer rank;
- complete ranks `0..world_size-1`;
- all ranks report identical capacity, layout fingerprint,
  bytes per slot, and cache limits;
- reported capacity equals
  `scheduler.hybrid_state_allocator.capacity`;
- Scheduler allocator exists;
- positive timeout/cache limits.

Only after all-rank validation does Engine construct and install:

```python
Qwen35HybridPrefixEngineRestoreCoordinator(
    self,
    self.scheduler.block_manager,
    self.scheduler.hybrid_state_allocator,
    timeout_s=timeout_s,
)
```

Malformed or inconsistent all-rank owner identity poisons the acknowledged
command collector.

Repeated Engine configuration with the same limits and timeout returns the
existing coordinator. Different settings fail closed.

## Failure Boundary

Rank-local owner creation is one-shot. If some ranks configure and another
rank fails, the acknowledged channel is poisoned and the Engine coordinator is
not installed. This phase does not add remote owner destruction because the
pool/cache/participant may already contain live state and destructive rollback
would be unsafe.

Therefore configuration must happen before Scheduler admission and before any
hybrid-prefix transaction.

## Correctness Tests

Create:

```text
tools/test_qwen35_live_restore_owner_factory.py
```

CPU/dependency-light tests cover:

1. factory derives ordered adapters from one existing pool without new tensor
   storage;
2. exact owner graph pool coherence;
3. malformed, incomplete, empty, or unsupported layouts fail closed;
4. ModelRunner without a bridge fails before allocation;
5. ModelRunner configuration installs and retains the exact participant;
6. identical ModelRunner reconfiguration is idempotent;
7. different cache limits or owner replacement fail closed;
8. identity row is exact and pickle-safe;
9. Engine TP=1 and TP>1 all-rank configuration;
10. outer rank versus inner participant identity validation;
11. cross-rank capacity/layout/bytes/cache mismatch poison;
12. allocator capacity mismatch poison;
13. Engine repeated identical configuration is idempotent;
14. Engine changed configuration fails closed;
15. Scheduler guard and `LLMEngine.step()` remain unchanged.

## Acceptance Gate

Complete only when:

- focused tests show RED then GREEN under Python 3.9 and 3.12;
- live transaction, restore-ticket, ModelRunner method, ack, and wiring tests
  pass;
- Qwen3.5/hybrid CPU regression scripts pass;
- chunked-prefill remains 97 pass / 1 known skip / 0 fail;
- Python 3.9/3.12 `py_compile` and `git diff --check` pass;
- staged files remain empty and experiment evidence remains present;
- handoff records that runtime bridge/native Qwen3.5 model construction and
  Scheduler admission are still missing.

Allowed conclusion:

> TinyLLMForge can automatically derive and configure rank-local
> hybrid-prefix restore owners from an already installed live Qwen3.5
> hybrid-state runtime pool, validate all-rank owner identity, and install the
> Engine transaction coordinator without duplicating state tensors.

Not established:

- native Qwen3.5 ModelRunner/model/checkpoint construction;
- automatic runtime bridge/pool installation;
- Scheduler admission;
- GPU/checkpoint correctness or any performance/cache/memory/quality benefit.

