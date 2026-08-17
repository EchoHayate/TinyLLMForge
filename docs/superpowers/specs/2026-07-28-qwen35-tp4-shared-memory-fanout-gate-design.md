# Qwen3.5 TP4 Shared-Memory Fan-Out Gate Design

## Objective

Prove that one production ModelRunner command can fan out through one real
named POSIX shared-memory segment to three independent production worker loops,
and that three production acknowledgement pipes are collected in deterministic
rank order:

```text
rank0 shell
  -> LLMEngine.call_model_runner_acknowledged
  -> ModelRunner.dispatch_command
  -> ModelRunner.write_shm
  -> one named POSIX SharedMemory segment
  -> three multiprocessing Events
  -> three ModelRunner.read_shm calls
  -> three ModelRunner.loop children
  -> three production acknowledgement executors
  -> three one-way acknowledgement pipes
  -> ModelRunnerCommandAckCollector
  -> ordered ranks (1, 2, 3)
```

This is a construction-free CPU transport gate. It must not import or
construct `LLMEngine` or `ModelRunner`, load checkpoint metadata or payloads,
construct model/target/adapter objects, enter the scheduler, call
`LLMEngine.step()`, initialize CUDA, execute forward, or run inference.

## Immutable Prerequisite

Use the completed TP2 live shared-memory artifact:

```text
run:
  qwen35-live-shm-engine-ack-20260728-110846
artifact:
  live_shared_memory_engine_ack_dispatch_preflight.json
artifact SHA256:
  11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57
source tree:
  6cc9672dbd80c211ccd64371573fd8de463b773fc5cc3ae7286ad21c9c664572
```

The prerequisite proves the same shared-memory codec, production worker loop,
production acknowledgement executor and collector, and construction-free
method extraction at TP2. This gate must verify the prerequisite hash and its
exact 55-file source closure before using it.

## Approaches Considered

### Construct a Production TP4 Engine

Rejected. Production Engine and ModelRunner construction crosses into
distributed initialization, model construction, checkpoint and cache state,
CUDA, scheduler, and worker lifecycle behavior outside this gate.

### Create Three Independent Shared-Memory Segments

Rejected. It avoids the production fan-out invariant. `write_shm()` writes one
payload once and then sets every worker Event; the gate must exercise that
single-buffer broadcast behavior.

### One Frozen Shared-Memory Broadcast with Three Worker Loops

Selected. AST-extract and compile only:

```text
ModelRunner.write_shm
ModelRunner.read_shm
ModelRunner.loop
ModelRunner.dispatch_command
LLMEngine.call_model_runner_acknowledged
```

The parent owns a rank0 shell, one unique 1 MiB shared-memory segment, three
real Events, three acknowledgement receivers, and the production collector.
Three fresh children attach to the same segment by name and run the frozen
production `loop()` as ranks 1, 2, and 3. No production constructor runs.

## Frozen Sources

Freeze:

```text
tinyvllm/engine/model_runner.py:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849
ModelRunner.write_shm:
  f9a377bf748d5be91a3c3722850e5e486f8e7dd8157e87d3dc6d692a60be6d76
ModelRunner.read_shm:
  1266b5d20b2978b655716f9ec8b58ce0a5644b9709164a23c18b85346170054a
ModelRunner.loop:
  342bac6d01606e4834e7ed77ef3e76d59b2fc3ea617afebe2c195912159dd2bb
ModelRunner.dispatch_command:
  9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342

tinyvllm/engine/llm_engine.py:
  6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae
LLMEngine.call_model_runner_acknowledged:
  6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d

tinyvllm/engine/model_runner_command_ack.py:
  ca28babca5cc725d8c9bf0e3e057fa4b0cabfd847bf0c052c40876fbc148c61b
```

The gate may import only the production acknowledgement module. It must not
import `llm_engine.py` or `model_runner.py`.

## Fan-Out Command Contract

The acknowledged command is:

```text
report_qwen35_tp4_fanout_identity(attempt_nonce)
```

Every shell implements this public method locally. It returns an immutable
row with:

```text
participant_id
operation = "report_tp4_shared_memory_fanout_identity"
status = "ok" or "error"
attempt_nonce
detail
```

The rows are a transport oracle only. They are not checkpoint candidates and
must never be described as real per-rank checkpoint binding.

The parent validates:

- the local result is participant 0;
- worker acknowledgements are returned as ranks `(1, 2, 3)`;
- worker result participant IDs are `(1, 2, 3)`;
- all four rows carry the exact attempt nonce and operation;
- any inner row with `status="error"` rejects the transaction without
  poisoning an otherwise healthy collector.

## Shared-Memory Ownership

Each fresh outer attempt:

1. creates one unique named shared-memory segment with capacity `2**20`;
2. creates three real multiprocessing Events;
3. creates three one-way acknowledgement pipes and three readiness pipes;
4. spawns fresh rank1, rank2, and rank3 children;
5. waits until all children have attached to the same segment;
6. executes one production acknowledged fan-out call;
7. waits until every live worker has cleared its command Event before any
   cleanup overwrite;
8. sends one production fire-and-forget `exit` envelope to all Events when at
   least one worker remains alive;
9. joins all children, closes all handles, and unlinks the segment exactly
   once in the parent.

The name must include the mode, parent PID, and a random suffix and remain at
most 30 characters. The fixed production name `tinyvllm` must never be
created or unlinked.

## Attempt Matrix

Run four fresh TP4 outer processes:

```text
tp4_fanout_success_reverse_completion
tp4_fanout_rank2_inner_error
tp4_fanout_rank2_ack_exception
tp4_fanout_rank2_exit_without_ack
```

Every attempt creates one new segment and three new child processes.

## Success Semantics

The success attempt deliberately delays worker methods so acknowledgement send
order is exactly `(3, 2, 1)`. A shared completion recorder wraps each child's
pipe sender without replacing the production executor.

The production collector must still return:

```text
ack ranks:
  (1, 2, 3)
result participant IDs:
  (1, 2, 3)
```

The attempt requires one acknowledged fan-out dispatch and one separate
fire-and-forget exit dispatch. All three Events must be set, waited, and
cleared once per envelope. All children exit with code zero and are joined.

## Failure Semantics

### Rank2 Inner Error

Rank2 returns a valid `ok` acknowledgement whose result row has
`status="error"`. Ranks 1 and 3 return successful rows. The production
collector remains healthy and returns all acknowledgements in rank order, but
the parent row validator rejects the fan-out transaction. Cleanup uses one
fire-and-forget exit envelope.

### Rank2 Acknowledgement Exception

Rank2 raises the exact injected `RuntimeError`. The production executor emits
an `error` acknowledgement. The production collector fails closed and poisons
itself. After all live workers have consumed the original envelope, cleanup
uses the production dispatch/write path for a fire-and-forget exit envelope.

### Rank2 Exit Without Acknowledgement

Rank2 raises `SystemExit(9)`. The production executor does not catch
`BaseException`, so rank2 exits without an acknowledgement. The collector
observes receive failure or worker death and poisons itself. Ranks 1 and 3
receive a cleanup exit envelope; rank2 remains exited with code 9.

## Evidence

Each attempt row records:

- unique outer PID, three unique child PIDs, and rank-to-PID mapping;
- one shared-memory name/capacity and all three attach confirmations;
- exact dispatch envelopes, command IDs, payload byte counts, and nonce;
- per-rank Event set/wait/clear counts;
- per-rank read/executor counts;
- acknowledgement send order and collector return order;
- per-rank acknowledgement status, result, and error;
- collector poison state and exact failure detail;
- fan-out validation state;
- exit-envelope state and each child exit code/join state;
- segment unlink state and post-unlink attach failure.

The validator rejects PID reuse, missing ranks, reused shared-memory names,
wrong completion order, partial cleanup, or an attachable finalized segment.

## Source Closure and Artifacts

Inherit the exact 55-file prerequisite closure and add:

```text
tools/qwen35_tp4_shared_memory_fanout_preflight.py
```

Total:

```text
56 unique source files
```

Publish atomically:

```text
tp4_shared_memory_fanout_preflight.json
source_manifest.json
```

Use one SHA-bound prerequisite, deterministic staging, four fresh attempt
workers, a separate finalizer, exact remote round trip, and a unique run tag.
Never overwrite or delete failed or superseded evidence.

## Independent Verification

A standard-library-only verifier imports neither TinyLLMForge nor the gate. It
recomputes:

- prerequisite/result hashes;
- exact 56-file source closure and source tree;
- frozen file/method hashes and signatures;
- four-row ordering, four unique outer PIDs, and twelve unique child PIDs;
- four unique shared-memory names and one-mebibyte capacity;
- exact envelopes, payload bytes, nonce, and Event/read/executor counts;
- reverse completion order and ranked collector order;
- inner error versus outer acknowledgement failure;
- poison, worker-death, exit, join, unlink, and non-attachability semantics;
- local/remote inventory and result SHA equality.

Tamper tests must reject at least a modified collector return order and a
modified rank2 child exit code.

## Static Safety

Require:

- zero imports or construction of `LLMEngine` and `ModelRunner`;
- one compiled invocation site for every frozen method;
- one live `SharedMemory(create=True)` site and child/probe attach sites;
- no fixed `SharedMemory(name="tinyvllm")` call;
- one production acknowledgement collector constructor;
- production worker-loop acknowledgement executor only;
- zero checkpoint, model construction, scheduler, `step()`, CUDA, forward, or
  inference calls;
- exact worker hard rejection and schema-v2 canonical `NO_GO` unchanged.

## Claim Boundary

This gate may prove:

- one production shared-memory write fans out to three real worker loops;
- three real acknowledgement pipes are collected in deterministic rank order;
- reverse worker completion does not change returned rank order;
- inner error, worker exception, and worker death fail closed distinctly;
- all workers and the one shared-memory segment are cleaned exactly.

It does not prove:

- production Engine or ModelRunner construction;
- TP4 checkpoint loading, candidate binding, or model ownership;
- scheduler or `LLMEngine.step()` integration;
- CUDA, forward, inference, correctness, latency, throughput, cache, GPU
  memory, compression, or quality benefit.

The schema-v2 canonical `NO_GO` remains unchanged.

## Authoritative Evidence

The construction-free TP4 shared-memory fan-out gate completed with:

```text
run:
  qwen35-tp4-shm-fanout-20260728-115046
local:
  experiments/qwen35_hybrid_state/
  qwen35-tp4-shm-fanout-20260728-115046
remote:
  /data00/home/sitian/sitian-workspace01/tllm/
  qwen35-tp4-shared-memory-fanout-runs/
  qwen35-tp4-shm-fanout-20260728-115046
```

Authoritative identities:

```text
result SHA256:
  ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a
source manifest SHA256:
  2a973fe1cdd7394fd502865df19a63521729faba5aac64a997fe55e5acf24ba7
source tree SHA256:
  ec7b0dee43a06c47b72f8ac14ab26518845f57f070e6c27d394bb4c328644403
prerequisite SHA256:
  11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57
```

Fresh outer PIDs:

```text
4044174
4044794
4045431
4046029
```

Fresh child PIDs:

```text
4044369, 4044370, 4044371
4045011, 4045013, 4045015
4045662, 4045663, 4045666
4046451, 4046452, 4046454
```

Unique shared-memory names:

```text
qwen35-tp4_fanout-4044174-7975
qwen35-tp4_fanout-4044794-b149
qwen35-tp4_fanout-4045431-49f3
qwen35-tp4_fanout-4046029-4bcf
```

Every attempt used one 1 MiB segment, three real Events, three real
acknowledgement pipes, and three fresh production worker loops. The fan-out
envelope serialized to 219 bytes and the exit envelope to 154 bytes.

Mode evidence:

```text
success:
  acknowledgement send order (3, 2, 1)
  collector return order (1, 2, 3)
  result participants (1, 2, 3)
  collector healthy; all children exit 0

rank2 inner error:
  acknowledgement statuses (ok, ok, ok)
  collector return order (1, 2, 3)
  collector healthy; parent row validation rejects rank2

rank2 acknowledgement exception:
  acknowledgement statuses (ok, error, ok)
  collector poisoned; all children receive exit and exit 0

rank2 SystemExit:
  acknowledgement statuses (ok, absent, ok)
  send order (3, 1); collector poisoned
  rank2 exits 9; ranks 1 and 3 receive exit and exit 0
```

The standard-library-only independent verifier passed 539 checks locally and
the same 539 checks against the remotely staged source and artifacts. Its
three tests include targeted rejection of modified collector return ordering
and a modified rank2 exit code.

The remote inventory contained exactly 56 source files, one prerequisite, and
two result files. Local and remote artifact hashes matched. All four names
were independently non-attachable after finalization, and all four outer plus
twelve child PIDs were absent.

Regression evidence:

```text
TP4 harness:
  7 tests passed
TP4 independent verifier:
  3 tests passed
TP2 live shared-memory harness/verifier:
  6 + 3 tests passed
Engine acknowledgement transport harness/verifier:
  7 + 2 tests passed
ModelRunner command acknowledgement:
  14 tests passed
ModelRunner live acknowledgement wiring:
  11 tests passed
Engine all-rank candidate binding:
  9 tests passed
ModelRunner published candidate binding:
  4 tests passed
real checkpoint worker boundary:
  6 tests passed
manifest-bound loader configuration:
  4 tests passed remotely
```

The local matrix passed 72 tests and the adjacent remote test passed four.
Python compilation, static audit, exact worker hard-rejection AST check,
schema-v2 `NO_GO` SHA check, `git diff --check`, and staged-zero passed.

One preserved superseded remote tag exists:

```text
qwen35-tp4-shm-fanout-20260728-114743
```

Its four attempts ran, but final publication failed because the CLI passed an
output string to an inherited atomic writer requiring `Path`. A focused RED
test reproduced the type-contract failure, the source was fixed at the CLI
boundary, and the successful tag used the corrected source. The failed remote
directory was retained and had no residual worker.

## Final Claim Boundary

```text
one write / three Events / three production worker loops:
  proven
three real acknowledgement pipes:
  proven
reverse completion with deterministic rank ordering:
  proven
rank2 inner error / exception / SystemExit:
  proven fail closed with distinct semantics
four outer and twelve child cleanup:
  proven
four shared-memory segment unlinks:
  proven
LLMEngine / ModelRunner construction:
  absent
TP4 real checkpoint loading or binding:
  absent
scheduler / LLMEngine.step():
  absent
CUDA / forward / inference:
  absent
production latency / throughput / cache / GPU-memory / quality:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
