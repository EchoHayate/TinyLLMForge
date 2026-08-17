# ModelRunner Command Acknowledgement Channel Design

## Objective

Add the dependency-light control-plane primitive required for rank 0 to prove
that every ModelRunner worker completed an acknowledged command.

The current Tensor Parallel dispatch path is one-way:

```text
rank 0 pickles [method_name, *args] into shared memory
rank 0 sets one Event per worker
worker reads and invokes the method
worker return value or exception is discarded
```

That is sufficient for synchronous model execution only because the later
collectives tend to expose missing ranks indirectly. It is insufficient for a
two-phase hybrid-prefix restore: rank 0 must distinguish all-rank prepare,
rank-local miss, rank-local exception, stale acknowledgement, timeout, and
worker death before it publishes request metadata.

This phase creates a stdlib-only command envelope, worker acknowledgement, and
rank-0 collector with real multiprocessing tests. It does not yet change
`LLMEngine`, `ModelRunner`, CUDA/NCCL startup, or scheduler admission.

## Alternatives

### 1. Reuse the Existing Shared Memory for Replies

Rejected. Multiple workers would race to overwrite one buffer, and rank 0
could not identify missing or duplicate ranks without adding another
serialization protocol and locking scheme.

### 2. Use a Distributed Collective for Control Acknowledgement

Rejected. A worker exception before entering the collective can deadlock the
remaining ranks. Control-plane error reporting must not depend on the model
data-plane collective succeeding.

### 3. One Multiprocessing Queue Shared by All Workers

Rejected for the first gate. A shared queue can work, but rank ownership,
worker closure, and per-rank liveness are less explicit. It also introduces a
feeder thread and queue shutdown semantics that complicate deterministic
failure tests.

### 4. One Unidirectional Pipe per Worker

Selected. `LLMEngine` can eventually create one pipe per spawned worker,
retain each receive endpoint at rank 0, and pass one send endpoint to the
worker. The existing shared-memory command broadcast remains unchanged.

## Module

Create `tinyvllm/engine/model_runner_command_ack.py`. The module imports only
Python standard-library packages so it can be tested without CUDA, PyTorch,
Transformers, or NCCL.

## Command Envelope

```python
@dataclass(frozen=True)
class ModelRunnerCommandEnvelope:
    command_id: int
    method_name: str
    args: tuple
    requires_ack: bool
```

`command_id` is monotonic on rank 0 and non-negative. `method_name` is a
non-empty public method name. `args` is a tuple. Every envelope must survive a
pickle round trip before dispatch.

The later runtime integration can preserve the existing fire-and-forget path
for latency-sensitive `run` calls initially, while restore
prepare/commit/rollback use `requires_ack=True`.

## Worker Acknowledgement

```python
@dataclass(frozen=True)
class ModelRunnerCommandAck:
    command_id: int
    rank: int
    status: str
    result: object = None
    error_type: str = ""
    error_detail: str = ""
```

`status` is exactly:

- `ok`: the method returned normally and `result` contains its return value;
- `error`: the method raised an ordinary `Exception`; result is `None`.

Error strings are bounded to 4096 UTF-8 characters each. Tracebacks are not
transported in this first gate. `BaseException` values such as
`KeyboardInterrupt` and `SystemExit` are not converted into success-like
messages; worker process termination is handled by collector liveness.

## Worker Executor

```python
def execute_acknowledged_command(
    envelope: ModelRunnerCommandEnvelope,
    *,
    rank: int,
    target: object,
    send_ack: Callable[[ModelRunnerCommandAck], None],
) -> object
```

The executor:

1. validates the envelope and rank;
2. resolves the exact public method;
3. invokes `method(*args)`;
4. for an acknowledged command, sends one `ok` acknowledgement containing the
   result;
5. for an ordinary method exception, sends one `error` acknowledgement and
   returns `None`;
6. for a fire-and-forget command, preserves ordinary return/exception
   behavior and sends nothing.

If acknowledgement serialization or sending fails after the method has run,
the failure is raised. The worker must not continue silently because rank 0
cannot prove the command outcome.

## Rank-0 Collector

```python
class ModelRunnerCommandAckCollector:
    def __init__(
        self,
        receivers: tuple[tuple[int, Connection], ...],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    )
```

Receiver ranks must be unique, positive, and contiguous with the expected
worker ranks supplied to `collect()`.

```python
def collect(
    self,
    command_id: int,
    *,
    expected_ranks: tuple[int, ...],
    timeout_s: float,
    is_rank_alive: Callable[[int], bool],
) -> tuple[ModelRunnerCommandAck, ...]
```

The collector:

1. validates command identity, timeout, expected ranks, and health;
2. polls every missing rank without blocking indefinitely;
3. receives and validates acknowledgements;
4. rejects wrong-rank messages on a rank-specific pipe;
5. rejects stale/future command IDs, duplicate rank acknowledgements, malformed
   values, and unpickle failures;
6. fails immediately if a missing rank is reported dead;
7. fails at the absolute deadline if any rank remains missing;
8. returns acknowledgements ordered by rank only when all expected ranks are
   present.

Any protocol violation, worker death, receive failure, timeout, or worker
`error` acknowledgement poisons the collector. A poisoned collector rejects
all future collections until the ModelRunner process group and channel are
reconstructed.

The collector never retries or guesses whether a command ran. At-most-once
command dispatch plus explicit runtime reconstruction is safer than
duplicating a stateful prepare/commit operation.

## Result Semantics

The outer command acknowledgement reports transport/execution status. A
successful restore `prepare` result can itself be a
`Qwen35HybridPrefixPrepareAck` with status `prepared`, `miss`, or `error`.

Therefore rank 0 evaluates two layers:

```text
outer ModelRunnerCommandAck.status
  proves worker command execution and transport

inner Qwen35HybridPrefixPrepareAck.status
  reports restore-protocol prepared/miss/error
```

An outer `ok` containing an inner `miss` is a clean restore miss, not a
transport failure. An outer `error`, timeout, death, stale message, or malformed
result poisons the runtime.

## Liveness and Timeout

The collector receives process liveness through `is_rank_alive(rank)`. The
later `LLMEngine` integration will implement it from the spawned worker
`Process.is_alive()` values.

Timeout uses one absolute monotonic deadline. Polling one rank may not grant a
fresh timeout to later ranks.

The CPU gate uses short real-time multiprocessing tests plus deterministic
fake-clock tests for deadline accounting.

## Cleanup

The module owns no processes and closes no pipe endpoints automatically.
`LLMEngine` will later own endpoint and process cleanup.

When the collector is poisoned:

- no further acknowledged command may be issued through it;
- outstanding restore tickets must become `rollback_failed` unless every rank
  subsequently acknowledges rollback through a reconstructed channel;
- allocator slots must not be reused under the old runtime.

## Correctness Test Matrix

Create `tools/test_model_runner_command_ack.py`.

The tests cover:

1. envelope and acknowledgement validation plus pickle round trips;
2. acknowledged success transports an exact result;
3. ordinary worker exception becomes bounded `error` acknowledgement;
4. fire-and-forget preserves return and exception behavior and sends no ack;
5. two spawned workers return acknowledgements ordered by rank despite
   opposite completion order;
6. one worker `error` poisons the collector;
7. missing rank timeout uses one absolute deadline and lists missing ranks;
8. dead worker is detected before timeout;
9. stale/future command ID, wrong rank, duplicate rank, malformed ack, closed
   pipe, and receive-unpickle failure poison the collector;
10. collector rejects duplicate receiver ranks, unknown expected ranks,
    invalid timeout, and reuse after poison;
11. an outer `ok` can carry an inner pickle-safe restore prepare
    acknowledgement without conflating status layers;
12. no CUDA, torch distributed, checkpoint, or remote server is used.

## Acceptance Gate

This phase is complete only when:

- focused tests show an observed missing-module RED and then pass;
- real spawned-process success/error/timeout/death cases pass;
- Python 3.9 and Python 3.12 `py_compile` pass;
- restore-ticket and Qwen3.5/hybrid CPU regressions remain green;
- `git diff --check` passes;
- staged files remain empty;
- no `experiments/` evidence is removed;
- handoff records the remaining live `LLMEngine`/`ModelRunner` wiring gate.

Allowed conclusion:

> TinyLLMForge has a CPU-tested, fail-closed per-worker acknowledgement
> channel contract that can distinguish all-rank command success from worker
> error, stale protocol data, timeout, and worker death.

Not established:

- live `LLMEngine` pipe creation or `ModelRunner.loop()` integration;
- NCCL/TP process correctness;
- hybrid-prefix scheduler admission;
- GPU/checkpoint correctness;
- latency, throughput, cache, memory, compression, or quality improvement.

