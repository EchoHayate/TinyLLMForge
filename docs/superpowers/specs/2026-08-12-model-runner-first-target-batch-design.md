# ModelRunner First-Target Batch Design

**Date:** 2026-08-12

## Goal

Add a production-shaped `ModelRunner` callback that executes the first target
decode for multiple speculative sequences in one model forward and returns
ordered `FirstTargetResult` rows for the generic speculative batch runtime.

This slice does not wire `LLMEngine` to the runtime. It establishes the real
first-target GPU boundary needed before engine integration.

## Current Gap

`execute_native_speculative_batch()` invokes one `run_first_targets(seqs)`
callback before proposal generation. Tests currently inject synthetic target
tokens, hidden states, and metadata. `ModelRunner` has no method that satisfies
that callback.

The existing `ModelRunner.run()` path cannot be reused as the callback:

- it samples according to sequence temperature rather than requiring exact
  greedy tokens;
- it does not return per-row hidden states;
- it hides logits behind optional debug recording;
- it resets context before a callback can package row results.

The existing fixed-Q tail verifier is therefore production-shaped only on the
tail side.

## Correctness Boundary

The generic speculative runtime currently performs greedy prefix acceptance.
It does not implement stochastic speculative decoding or rejection sampling.
The first-target boundary must reject any row whose `temperature` is not
exactly zero.

The first implementation also supports KV-only model state. If
`ModelRunner.hybrid_state_runtime_bridge` is active, both the first-target and
tail verifier boundaries fail closed. Recurrent or other non-KV state needs a
separate transactional ownership design; committing KV while leaving rejected
recurrent suffix state mutated would be incorrect.

These restrictions are capability/state based. Generic speculative code must
not branch on model names or proposal-source names.

## Considered Approaches

### 1. Reuse `ModelRunner.run()` and read debug logits

This minimizes new code, but it cannot return hidden states reliably and it
uses the ordinary sampler. Debug-logit recording also copies tensors to CPU
and is not a production callback contract.

**Decision:** rejected.

### 2. Add a dedicated first-target batch method

The method validates the speculative boundary, uses `prepare_decode()` once,
runs the target model once in eager decode mode, optionally returns hidden
states and logits, packages ordered rank-0 results, and resets context in a
`finally` block.

**Decision:** selected. It is the smallest production-shaped boundary and
matches the fixed-Q tail method.

### 3. Refactor all ordinary and speculative decode execution into one shared
forward lifecycle

This could eventually reduce duplication, but it would mix engine integration,
sampling, profiling, hybrid-state ownership, and speculative semantics in one
change.

**Decision:** deferred until the callback pair is exercised by `LLMEngine`.

## Public Interface

`ModelRunner` adds:

```python
def run_spec_first_target_batch(
    self,
    seqs: tuple[Sequence, ...],
    *,
    return_hidden: bool = False,
    return_logits: bool = False,
) -> tuple[FirstTargetResult, ...] | None:
```

The method preserves input sequence order in its result rows.

For rank zero:

```text
target_token:
  greedy argmax for the row
target_hidden:
  hidden_states[row_index] when return_hidden is true, otherwise None
target_logits:
  logits[row_index] when return_logits is true, otherwise None
metadata:
  immutable callback metadata containing batch_index and execution_mode
```

Worker ranks execute the same model forward and return `None`.

## Data Flow

1. Validate that `seqs` is a non-empty tuple with unique non-negative sequence
   IDs.
2. Require `temperature == 0` for every sequence.
3. Reject active non-KV/hybrid state before decode preparation or KV mutation.
4. Call `prepare_decode(list(seqs))` exactly once.
5. Call `_kv_offload_before_forward()` once.
6. Call:

   ```python
   run_model(
       input_ids,
       positions,
       False,
       return_hidden=return_hidden,
       execution_mode="decode",
   )
   ```

   exactly once.
7. Call `_kv_offload_after_forward()` once after a successful target forward.
8. On rank zero, compute one greedy token per row and construct ordered
   `FirstTargetResult` rows.
9. Always call `reset_context()` in `finally`.

No per-sequence target forward is allowed.

## Stateful-Model Fail-Closed Rule

The callback pair must reject an active `hybrid_state_runtime_bridge`:

```text
speculative verification requires transactional non-KV state
```

The check belongs in the common speculative compatibility validation so both
first-target and tail paths enforce the same boundary.

This does not claim that hybrid models cannot support speculation. It means
the current transaction owns KV blocks only, so recurrent/convolution state
must not be mutated speculatively until snapshot/commit/rollback semantics
exist.

## Error Handling

Validation happens before `prepare_decode()`:

- non-tuple or empty sequence batch;
- invalid or duplicate sequence IDs;
- missing or non-numeric temperature;
- non-greedy temperature;
- active non-KV/hybrid state.

Forward, argmax, or packaging failures propagate unchanged. Context is reset
for preparation failures, forward failures, rank-zero result failures, and
successful execution.

KV-offload dirty marking occurs only after a successful model forward.

## Tests

Dependency-light execution tests cover:

- batch two uses one decode preparation and one target forward;
- ordered greedy target rows;
- optional row hidden/logit payloads;
- worker rank executes the forward and returns `None`;
- context reset after forward failure;
- non-greedy input fails before preparation;
- active hybrid state fails before preparation;
- the existing tail verifier also rejects active hybrid state.

AST/source tests enforce:

- exactly one `prepare_decode()` call;
- exactly one `run_model()` call;
- no call to `run()` or `_run_model_step()`;
- no forward nested in a loop or comprehension;
- `reset_context()` is inside `finally`.

## Explicit Non-Goals

- `LLMEngine` callback wiring;
- fixed-Q grouping of selected rows;
- variable-Q tail verification;
- stochastic speculative decoding;
- transactional recurrent/convolution state;
- multi-token scheduler postprocessing;
- CUDA Graph capture;
- GPU numerical parity or performance claims.

## Promotion Boundary

After this slice:

```text
production-shaped first-target ModelRunner batch:
  implemented for greedy KV-only execution
fixed-Q tail ModelRunner batch:
  implemented for greedy KV-only execution
LLMEngine callback wiring:
  not implemented
stateful-model speculative transaction:
  not implemented
end-to-end performance improvement:
  unmeasured
overall classification:
  NOT_PROMOTABLE
```
