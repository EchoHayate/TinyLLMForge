# ModelRunner Hybrid Prefix Restore Methods Design

## Objective

Expose the CPU-tested Qwen3.5 hybrid-prefix restore participant through real
ModelRunner acknowledged methods and add Engine-side validation of the nested
per-rank prepare results.

The runtime still does not automatically construct a Qwen3.5 state pool,
snapshot cache, or participant from a checkpoint. This phase therefore adds an
explicit installation contract and fails closed when no participant is
installed.

Scheduler hybrid-prefix admission remains disabled.

## ModelRunner Ownership

Add:

```python
self.qwen35_hybrid_prefix_restore_participant = None
```

Installation:

```python
def install_qwen35_hybrid_prefix_restore_participant(
    self,
    participant: Qwen35HybridPrefixRestoreParticipant,
) -> None
```

Validation:

- participant has the exact expected type;
- `participant.participant_id == self.rank`;
- installation is one-shot unless the identical object is supplied;
- if `hybrid_state_runtime_bridge` exists, its pool is the participant pool.

The method does not create pools, adapters, transactions, or caches.

## Restore Methods

Add acknowledged-call targets:

```python
def prepare_hybrid_prefix_restore(payload)
def validate_hybrid_prefix_restore(payload)
def commit_hybrid_prefix_restore(payload)
def rollback_hybrid_prefix_restore(payload)
```

Every method validates that a participant is installed and that
`payload.request_id`, lease identity, ticket identity, and participant-local
state are delegated to the participant implementation.

Return values are pickle-safe dictionaries:

```python
{
    "ticket_id": int,
    "participant_id": int,
    "operation": str,
    "status": str,
    "detail": str,
}
```

Prepare maps the inner participant acknowledgement directly:

- `prepared`;
- `miss`;
- `error`.

Validate, commit, and rollback return `ok` only after the participant method
returns normally. Their exceptions become outer
`ModelRunnerCommandAck(status="error")` through the existing executor.

## Engine Prepare Aggregation

Add:

```python
def prepare_model_runner_hybrid_prefix_restore(
    self,
    payload,
    *,
    timeout_s,
) -> tuple[dict, ...]
```

It calls:

```python
call_model_runner_acknowledged(
    "prepare_hybrid_prefix_restore",
    payload,
    timeout_s=timeout_s,
)
```

Then validates:

- local result and every worker outer result are dictionaries;
- results cover participant IDs exactly `0..world_size-1`;
- every result has the requested ticket ID;
- operation is exactly `prepare`;
- status is one of `prepared`, `miss`, `error`;
- detail is a string.

Malformed nested results poison the command collector and raise. Valid inner
`miss` or `error` does not poison the transport channel; it is returned to the
restore coordinator for deterministic rollback.

The returned tuple is ordered by participant ID.

## Other Engine Operations

Add a private helper:

```python
def _call_model_runner_hybrid_prefix_restore_operation(
    operation,
    payload,
    *,
    timeout_s,
) -> tuple[dict, ...]
```

It supports `validate`, `commit`, and `rollback`, calls the corresponding
acknowledged ModelRunner method, and requires every nested result status to be
`ok`. Any malformed or non-`ok` result poisons the collector and raises.

Public wrappers:

```python
validate_model_runner_hybrid_prefix_restore(...)
commit_model_runner_hybrid_prefix_restore(...)
rollback_model_runner_hybrid_prefix_restore(...)
```

This phase does not compose these methods with
`Qwen35HybridPrefixRestoreCoordinator`; that is the next Engine transaction
gate.

## Correctness Tests

Create `tools/test_model_runner_hybrid_prefix_restore_methods.py`.

Dependency-light tests cover:

1. participant installation type, rank, one-shot, and bridge-pool coherence;
2. uninstalled methods fail closed;
3. prepare returns exact pickle-safe prepared/miss/error dictionaries;
4. validate/commit/rollback call the exact participant operation and return
   `ok`;
5. participant exceptions propagate to the outer executor error path;
6. Engine prepare aggregation orders rank 0 plus worker results;
7. inner miss/error remain valid protocol results and do not poison;
8. malformed ticket/rank/operation/status/detail or missing rank poisons and
   raises;
9. validate/commit/rollback require all nested `ok`;
10. TP=1 uses the same nested-result validation;
11. `LLMEngine.step()` and Scheduler fail-closed guard remain unchanged.

## Acceptance Gate

Complete only when:

- focused tests show RED then GREEN;
- command ack/live wiring and Qwen3.5 restore regressions pass;
- chunked-prefill matrix remains 97 pass / 1 known skip / 0 fail;
- Python 3.9/3.12 compilation and `git diff --check` pass;
- staged files remain empty and experiment evidence remains present;
- handoff records that automatic checkpoint participant construction and
  Engine coordinator composition are still missing.

Allowed conclusion:

> TinyLLMForge can invoke rank-local hybrid-prefix restore participants through
> acknowledged ModelRunner methods and validate nested all-rank results on the
> Engine control plane.

Not established:

- automatic Qwen3.5 pool/cache/participant construction;
- full Engine restore-ticket transaction;
- scheduler admission;
- GPU/checkpoint correctness or performance/cache/memory benefit.

