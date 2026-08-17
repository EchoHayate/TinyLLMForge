# ModelRunner Live Acknowledgement Wiring Design

## Objective

Wire the CPU-tested command acknowledgement protocol into the actual
`LLMEngine` and `ModelRunner` process lifecycle while preserving all existing
fire-and-forget call sites.

This phase changes the real Python control path but remains CPU/static tested:
it does not instantiate a CUDA ModelRunner, start NCCL, load a checkpoint, or
enable hybrid-prefix scheduler admission.

## Existing Compatibility Contract

Existing callers use:

```python
result = model_runner.call(method_name, *args)
```

Rank 0 broadcasts the command to workers and invokes the same method locally.
Only the rank-0 return value is observable.

That behavior remains the default. Existing `run`, profiling, release, and
exit call sites do not become acknowledged automatically.

## Process Channel Ownership

For every worker rank `1..tensor_parallel_size-1`, `LLMEngine.__init__` creates:

```text
receive endpoint -> retained by LLMEngine/rank 0
send endpoint    -> passed to exactly one worker ModelRunner
```

The pipe is unidirectional. After `Process.start()`, the parent closes its copy
of the send endpoint.

`ModelRunner` constructor gains an optional `ack_sender`:

- rank 0 must receive `None`;
- a worker in TP>1 must receive a send-capable endpoint;
- TP=1 uses no pipe or collector.

`LLMEngine` creates one `ModelRunnerCommandAckCollector` from the receive
endpoints and owns endpoint cleanup.

## Shared-Memory Command Format

`ModelRunner.write_shm()` serializes a `ModelRunnerCommandEnvelope`, not the
legacy list form.

`ModelRunner.read_shm()` returns the validated envelope. For one transition
window it accepts the legacy `[method_name, *args]` form and converts it into a
fire-and-forget envelope with a worker-local compatibility command ID. This
prevents stale external tooling from crashing immediately, but rank 0 emits
only envelopes after this phase.

Every ModelRunner instance owns a monotonic command-ID counter. Only rank 0
allocates IDs for broadcast commands.

## Worker Loop

```python
def loop(self):
    while True:
        envelope = self.read_shm()
        execute_acknowledged_command(
            envelope,
            rank=self.rank,
            target=self,
            send_ack=self.ack_sender.send,
        )
        if envelope.method_name == "exit":
            break
```

For fire-and-forget envelopes, the executor sends no acknowledgement and
preserves existing method exception behavior.

For acknowledged envelopes, it sends one outer `ok` or `error`. Failure to
send the acknowledgement terminates the worker loop by propagation.

## Rank-0 APIs

### Compatible Call

`ModelRunner.call(method_name, *args)`:

1. allocates a fire-and-forget envelope;
2. broadcasts it when TP>1;
3. invokes the local method;
4. returns the local result exactly as before.

### Envelope Dispatch

Add:

```python
def dispatch_command(
    self,
    method_name: str,
    *args,
    requires_ack: bool,
) -> ModelRunnerCommandEnvelope
```

It validates and broadcasts one envelope but does not execute the local rank.
It is rank-0 only.

### Acknowledged Engine Call

Add:

```python
def call_model_runner_acknowledged(
    self,
    method_name: str,
    *args,
    timeout_s: float,
) -> tuple[object, tuple[ModelRunnerCommandAck, ...]]
```

The method:

1. dispatches one acknowledged envelope;
2. invokes the same method on rank 0;
3. collects worker acknowledgements using one deadline and
   `Process.is_alive()`;
4. returns `(local_result, worker_acks)` only after all worker outer statuses
   are `ok`.

For TP=1 it invokes locally and returns an empty worker tuple.

If rank-0 local execution raises after broadcast:

- the acknowledgement collector is explicitly poisoned;
- no later acknowledged command may reuse the channel;
- the original local exception is re-raised.

If worker collection fails, the collector poisons itself and the exception is
propagated. The local method may already have run; callers must treat the
runtime as poisoned and must not publish request metadata or reuse state
leases.

The method is not used for ordinary `run` in this phase.

## Collector Poison API

Add:

```python
def poison(self, reason: str) -> None
```

It is idempotent, requires a non-empty reason, and makes future `collect()`
calls fail closed. This supports rank-0 local failures after worker dispatch.

## Cleanup

`LLMEngine.exit()` retains the existing model shutdown call sequence in this
CPU/static phase. In a `finally` path it:

- closes every rank-0 receive endpoint;
- closes any retained parent send endpoint defensively;
- joins workers using the existing behavior.

The existing distributed barrier remains a known GPU-runtime risk. This phase
does not claim that a process crash during `ModelRunner.exit()` is recoverable.
The acknowledged restore methods added later must not depend on the exit
barrier for correctness.

## Static and Dependency-Light Tests

Create `tools/test_model_runner_live_ack_wiring.py`.

Tests load the real class methods without constructing CUDA/NCCL objects and
use fake processes, events, shared memory, pipe endpoints, and target methods.

The matrix covers:

1. `ModelRunner.call()` emits a fire-and-forget envelope and preserves the
   local return value;
2. `dispatch_command()` emits one acknowledged envelope with monotonic ID;
3. `read_shm()` accepts the new envelope and one legacy compatibility list;
4. worker loop sends an acknowledgement for an acknowledged method and none
   for fire-and-forget;
5. worker loop exits after an `exit` envelope;
6. `LLMEngine.call_model_runner_acknowledged()` returns local result plus
   ordered worker acknowledgements;
7. TP=1 returns local result plus an empty worker tuple;
8. rank-0 local exception poisons the collector and is re-raised;
9. collector failure propagates without converting it into a local success;
10. worker liveness callback maps rank `1..N` to the corresponding Process;
11. pipe creation passes one send endpoint per worker and closes the parent
    duplicate after start;
12. cleanup closes receive endpoints without changing normal return values;
13. actual source imports the acknowledgement module and does not change
    `LLMEngine.step()` or scheduler admission in this phase.

## Acceptance Gate

This phase is complete only when:

- wiring tests show an observed RED before implementation and pass after;
- command-ack protocol tests pass under Python 3.9 and 3.12;
- chunked-prefill/Engine dependency-light tests remain green except the known
  Config AST skip;
- Qwen3.5 restore-ticket/hybrid regressions remain green;
- Python 3.9 and Python 3.12 compilation pass;
- `git diff --check` passes;
- staged files remain empty and experiment evidence remains present;
- handoff records the remaining restore participant method integration.

Allowed conclusion:

> TinyLLMForge's real Engine/ModelRunner Python control path can broadcast an
> acknowledged command and collect explicit per-worker outcomes without
> changing existing fire-and-forget callers.

Not established:

- CUDA/NCCL multi-rank execution of the channel;
- crash-safe distributed exit;
- ModelRunner hybrid-prefix prepare/commit/rollback methods;
- scheduler admission or request publication;
- checkpoint correctness or performance/cache/memory improvement.

