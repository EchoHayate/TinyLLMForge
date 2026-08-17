# Qwen3.5 Live Shared-Memory Engine Acknowledgement Dispatch Gate Design

## Objective

Execute the already proven TP2 real-candidate binding transaction through the
production ModelRunner shared-memory command path:

```text
Engine all-rank binding method
  -> Engine acknowledged call
  -> ModelRunner dispatch_command
  -> ModelRunner write_shm
  -> named POSIX SharedMemory
  -> multiprocessing Event
  -> ModelRunner read_shm
  -> ModelRunner loop
  -> production acknowledgement executor
  -> acknowledgement pipe and collector
  -> Engine all-rank result validation
```

This remains an explicit CPU preflight. It must not import or construct
`LLMEngine` or `ModelRunner`, load checkpoint metadata or payloads, construct a
model/target/adapter, start the scheduler, call `LLMEngine.step()`, initialize
CUDA, execute forward, or run inference.

## Immutable Prerequisite

Use the completed Engine acknowledgement transport artifact:

```text
run:
  qwen35-engine-ack-transport-20260728-102828
artifact:
  engine_ack_transport_preflight.json
artifact SHA256:
  8aeb571c3d56641e747a0d5c5e66314efe6b35b73320cb49e0340c0fe5fd42fb
source tree:
  a041ebf7653e141dd96ebe31143ba00e5634c61c1a4bec68f17e7a7c6bba5cc8
```

The prerequisite retains the exact TP2 rank0/rank1 success rows and the
rank1 bridge-conflict row. It also binds the production acknowledgement
semantics already proved with private command pipes.

## Approaches Considered

### Construct Production ModelRunner and Engine

Rejected. ModelRunner construction loads the model, initializes distributed
state, allocates KV cache, may capture CUDA graphs, and starts the worker loop.
Engine construction additionally starts workers, tokenizer, scheduler, and
other runtime state. Those paths are outside this gate.

### Shared-Memory Codec Only

Rejected as insufficient. Running `write_shm()` and `read_shm()` against one
buffer proves serialization but not the production child loop, executor,
acknowledgement collector, Engine call ordering, all-rank binding validation,
or worker-death behavior.

### Frozen Production Methods with Real POSIX Shared Memory

Selected. AST-extract and compile:

```text
ModelRunner.write_shm
ModelRunner.read_shm
ModelRunner.loop
ModelRunner.dispatch_command
LLMEngine.call_model_runner_acknowledged
LLMEngine.bind_qwen35_loaded_checkpoint_candidates
```

Create a unique named `multiprocessing.shared_memory.SharedMemory` segment for
each attempt. The parent shell owns rank0, the segment, one real Event, and the
production acknowledgement collector. One fresh child attaches to the segment
by name and executes the production `loop()` on a rank1 shell. No production
class constructor runs.

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
LLMEngine.bind_qwen35_loaded_checkpoint_candidates:
  82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c

tinyvllm/engine/model_runner_command_ack.py:
  ca28babca5cc725d8c9bf0e3e057fa4b0cabfd847bf0c052c40876fbc148c61b
```

The gate may import only the production acknowledgement module. It must not
import `llm_engine.py` or `model_runner.py`.

## Shared-Memory Ownership

Each fresh outer attempt:

1. creates one unique named shared-memory segment with capacity `2**20`;
2. creates one real multiprocessing Event;
3. creates one one-way acknowledgement pipe;
4. spawns one rank1 child;
5. waits for an explicit child-ready signal after the child attaches;
6. executes the production Engine binding method on the rank0 shell;
7. sends one production `exit` envelope for live-child cleanup when possible;
8. joins the child, closes both processes' shared-memory handles, and unlinks
   the segment exactly once in the parent.

The unique name must include the run tag, attempt mode, parent PID, and a
random suffix. The fixed production name `tinyvllm` must never be created or
unlinked by this gate.

## Attempt Matrix

Run four fresh TP2 outer processes:

```text
tp2_shm_success
tp2_shm_worker_binding_error
tp2_shm_worker_ack_exception
tp2_shm_worker_exit_without_ack
```

Every attempt starts a unique rank1 child and unique shared-memory segment.

## Success Semantics

`tp2_shm_success` requires:

- one production `dispatch_command()` call;
- one production `write_shm()` call;
- one production `read_shm()` call for the binding command;
- one exact `ModelRunnerCommandEnvelope`;
- one Event set, wait, and clear for the binding command;
- one production worker-loop executor call;
- one `ok` acknowledgement through the real acknowledgement pipe;
- one production collector call;
- exact authoritative rank0/rank1 bound rows;
- completion commit only after matching model/layout/dtype;
- exact replay returns the committed tuple with zero new binding dispatch;
- one separate fire-and-forget `exit` envelope;
- child exit code zero and no residual child or shared-memory segment.

## Failure Semantics

### Worker Binding Error

The worker target returns the authoritative rank1 bridge-conflict row. The
outer acknowledgement remains `ok`; the Engine binder rejects the inner row,
leaves completion unset, and the collector remains usable. Cleanup sends the
separate `exit` envelope.

### Worker Acknowledgement Exception

The worker target raises the exact injected `RuntimeError`. The production
executor emits an `error` acknowledgement. The collector poisons itself and
the Engine leaves completion unset. Cleanup bypasses the poisoned collector
and uses the production dispatch/write path to send a fire-and-forget `exit`
envelope.

### Worker Exit Without Acknowledgement

The worker target raises `SystemExit` after receiving the binding envelope.
The production executor does not convert `BaseException` to an acknowledgement,
so the production worker loop exits without sending an ack. The collector
observes EOF or worker death, poisons itself, and the Engine leaves completion
unset. The child must be joined and the shared-memory segment unlinked without
an exit envelope.

## Shared-Memory Evidence

Each row records:

- unique shared-memory name and capacity;
- parent and child PIDs;
- child-ready, child-collected, and segment-unlinked state;
- command IDs and exact envelopes;
- serialized payload byte count;
- Event set/wait/clear counts observed across the transaction;
- dispatch, write, read, executor, collector, and ack counts;
- acknowledgement status/error;
- binding rows and completion state;
- cleanup exit envelope evidence;
- collector poison state and exact failure detail.

The row validator must reject missing cleanup, reused names/PIDs, unexpected
command counts, or a segment that remains attachable after finalization.

## Source Closure and Artifacts

Inherit the exact 54-file prerequisite closure and add:

```text
tools/qwen35_live_shared_memory_engine_ack_dispatch_preflight.py
```

Total:

```text
55 unique source files
```

Publish atomically:

```text
live_shared_memory_engine_ack_dispatch_preflight.json
source_manifest.json
```

Use one SHA-bound prerequisite, deterministic staging, four fresh attempt
workers, a separate finalizer, exact remote round trip, and a unique run tag.
Never overwrite or delete failed or superseded evidence.

## Independent Verification

A standard-library-only verifier imports neither TinyLLMForge nor the gate. It
recomputes:

- prerequisite/result hashes;
- exact 55-file source closure and source tree;
- all frozen file and AST method hashes and signatures;
- exact four-row ordering and unique outer/child PIDs;
- unique shared-memory names and one-mebibyte capacity;
- exact envelope bytes, command IDs, Event counts, and method counts;
- exact binding rows and completion configuration;
- inner binding error versus outer acknowledgement error;
- poison, worker-death, exit-envelope, join, and unlink semantics;
- exact replay zero additional binding dispatch;
- local/remote inventory and result SHA equality.

Tamper tests must reject at least a modified shared-memory name and a modified
acknowledgement status.

## Static Safety

Require:

- zero imports or construction of `LLMEngine` and `ModelRunner`;
- one compiled invocation site for every frozen method;
- exactly two `SharedMemory(create=True)` sites and four attach/probe sites
  across the codec fixture and live transaction;
- no fixed `SharedMemory(name="tinyvllm")` call;
- one production acknowledgement collector constructor;
- production worker-loop acknowledgement executor only;
- no checkpoint metadata/read/load/adapter/target/model construction;
- no scheduler, `LLMEngine.step()`, CUDA, forward, or inference call;
- production `step()` remains loading/publication/binding-free;
- exact real-worker hard rejection remains unchanged;
- immutable schema-v2 canonical `NO_GO` remains unchanged.

## Allowed Conclusion

Passing proves that already proven real per-rank binding rows traverse the
production shared-memory command codec and synchronization path, production
worker loop, acknowledgement channel, Engine acknowledged call, and Engine
all-rank binding validator at TP2.

It does not prove production ModelRunner/Engine construction, checkpoint
loading inside Engine, multi-worker TP greater than two, scheduler integration,
`LLMEngine.step()`, CUDA, forward/inference correctness, latency, throughput,
cache savings, GPU-memory savings, compression, or model quality.

The next safe gate after this one is explicit construction-free multi-worker
shared-memory fan-out at TP4, still outside `LLMEngine.step()` and without
checkpoint loading, CUDA, forward, or inference.

## Authoritative Evidence

The source-bound remote gate completed on
`sitian@10.232.195.203`:

```text
run tag:
  qwen35-live-shm-engine-ack-20260728-110846
remote run:
  /data00/home/sitian/sitian-workspace01/tllm/
  qwen35-live-shared-memory-engine-ack-runs/
  qwen35-live-shm-engine-ack-20260728-110846
local run:
  experiments/qwen35_hybrid_state/
  qwen35-live-shm-engine-ack-20260728-110846
```

Authoritative identities:

```text
live_shared_memory_engine_ack_dispatch_preflight.json:
  11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57
source_manifest.json:
  7dea81d146b9eedf327cc2dbb8bf19b43895a83a14bc68e5e4827a23c7469ad4
source tree:
  6cc9672dbd80c211ccd64371573fd8de463b773fc5cc3ae7286ad21c9c664572
prerequisite:
  8aeb571c3d56641e747a0d5c5e66314efe6b35b73320cb49e0340c0fe5fd42fb
```

Fresh process identities:

```text
outer attempt PIDs:
  3285526, 3285730, 3285967, 3286191
rank1 child PIDs:
  3285644, 3285849, 3286089, 3286360
```

Unique shared-memory names:

```text
qwen35-tp2_shm_su-3285526-fb17
qwen35-tp2_shm_wo-3285730-302c
qwen35-tp2_shm_wo-3285967-47cd
qwen35-tp2_shm_wo-3286191-ea58
```

Every segment had capacity 1 MiB. Independent remote probes after the run
proved all four names were no longer attachable. No attempt worker or finalizer
remained alive.

Transport evidence:

```text
binding envelope payload:
  199 bytes
exit envelope payload:
  154 bytes
```

Success, worker-inner-error, and worker-ack-exception attempts each observed
two dispatch/write/read/Event-set/Event-wait/Event-clear/executor operations:
one acknowledged binding command and one fire-and-forget exit command. The
worker-death attempt observed exactly one of each binding operation, no exit
command, and child exit code 9.

Mode results:

```text
tp2_shm_success:
  ack ok, collector healthy, completion committed, replay added zero binding
  dispatches, exit sent, child exit 0
tp2_shm_worker_binding_error:
  ack ok, collector healthy, rank1 inner error, completion unset, exit sent,
  child exit 0
tp2_shm_worker_ack_exception:
  ack error, collector poisoned, completion unset, exit sent, child exit 0
tp2_shm_worker_exit_without_ack:
  ack absent, receive failure, collector poisoned, completion unset, no exit,
  child exit 9
```

The standard-library-only independent verifier:

```text
tools/verify_qwen35_live_shared_memory_engine_ack_dispatch_gate.py
```

passed 458 checks locally and the same 458 checks against the remotely staged
source and remote artifacts. Its three focused tests include independent
tamper rejection for both the shared-memory name and acknowledgement status.

The remote inventory contained exactly 55 staged source files and three
root-level JSON files: one immutable prerequisite and two results. Both local
result hashes matched the remote files exactly. Remote CLI validation passed.

Regression evidence:

```text
new shared-memory harness: 6 tests passed
new independent verifier: 3 tests passed
previous Engine acknowledgement harness: 7 tests passed
previous Engine acknowledgement verifier: 2 tests passed
ModelRunner command acknowledgement: 14 tests passed
ModelRunner live acknowledgement wiring: 11 tests passed
Engine all-rank candidate binding: 9 tests passed
ModelRunner published candidate binding: 4 tests passed
real checkpoint worker boundary: 6 tests passed
manifest-bound loader configuration: 4 tests passed remotely
```

The local nine-group matrix passed 62 tests and the remote torch-dependent
configuration suite passed another four. Python 3.9 compilation,
`git diff --check`, and staged-zero checks passed.

Static audit found exactly two shared-memory create sites and four
attach/probe sites across the codec fixture and live transaction. It found
zero fixed `tinyvllm` names, Engine/ModelRunner imports or construction,
checkpoint calls, scheduler calls, `step()` calls, CUDA calls, forward calls,
or inference calls. Production `LLMEngine.step()` remains free of candidate
loading, publication, or binding references.

The exact real worker hard rejection remains:

```text
RuntimeError: real checkpoint load worker execution is not implemented; only the local safety dry-run is authorized
```

The immutable schema-v2 authoritative verifier remains `NO_GO` with SHA256:

```text
a6b0be0be57e6df62dfbcf7b4f05936218b57c417e2ac74c97306a3bc32e1f38
```

Final claim boundary:

```text
production dispatch -> write_shm -> POSIX SharedMemory -> Event -> read_shm
-> worker loop -> acknowledgement -> collector -> Engine validator:
  proven at TP=2
completion after exact all-rank identity and replay zero-dispatch:
  proven
worker inner error / worker exception / SystemExit without acknowledgement:
  proven fail closed with distinct acknowledgement semantics
shared-memory and worker cleanup:
  proven for all four attempts
LLMEngine / ModelRunner construction:
  absent
TP greater than two fan-out:
  unproven
checkpoint loading / scheduler / LLMEngine.step():
  absent
CUDA / forward / inference:
  absent
production latency / throughput / cache or GPU-memory savings / quality:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
