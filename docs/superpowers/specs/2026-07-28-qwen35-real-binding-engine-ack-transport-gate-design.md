# Qwen3.5 Real-Binding Engine Acknowledgement Transport Gate Design

## Objective

Execute the existing production Engine acknowledgement and all-rank binding
methods over the authoritative real-candidate binding rows without importing
or constructing `LLMEngine` or `ModelRunner`.

The gate proves:

```text
authoritative per-rank real binding rows
  -> production ModelRunner command envelope
  -> real one-way multiprocessing acknowledgement pipe
  -> production acknowledgement executor
  -> production acknowledgement collector
  -> production Engine acknowledged call
  -> production Engine all-rank binding validation and completion commit
```

It remains an explicit preflight. It does not call `LLMEngine.step()`, start
the scheduler, load a checkpoint, construct a model, initialize CUDA, execute
forward, or run inference.

## Immutable Prerequisite

Use:

```text
qwen35-model-runner-published-binding-20260728-100419
model_runner_published_candidate_binding_preflight.json:
  79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a
source tree:
  0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785
```

The prerequisite must retain exactly six rows and unique PIDs. Its success
rows provide TP1 rank0 and TP2 rank0/rank1 bound results. Its conflict rows
provide exact local or worker inner-error results.

## Approaches Considered

### Fake `call_model_runner_acknowledged`

Rejected as authoritative. Existing dependency-light Engine binding tests
already inject a fake acknowledged call. That proves row validation but not
the production command envelope, acknowledgement executor, pipe transport,
collector, timeout/liveness behavior, or Engine dispatch ordering.

### Import and Construct `LLMEngine`

Rejected. `LLMEngine.__init__` enters ModelRunner construction, worker process
startup, model/runtime setup, and CUDA-sensitive paths outside this gate.

### Frozen Engine Methods with Real Acknowledgement Pipes

Selected. Freeze and AST-compile:

```text
LLMEngine.call_model_runner_acknowledged
LLMEngine.bind_qwen35_loaded_checkpoint_candidates
ModelRunner.dispatch_command
```

Use the production:

```text
ModelRunnerCommandEnvelope
ModelRunnerCommandAck
ModelRunnerCommandAckCollector
execute_acknowledged_command
```

Build private Engine and rank-local runner shells. For TP2, the rank0 shell's
production `dispatch_command()` sends the exact envelope to one fresh worker
over a private command pipe. The worker executes
`execute_acknowledged_command()` against a private target whose
`bind_published_qwen35_loaded_checkpoint_candidate()` returns the exact
authoritative rank1 row or raises an injected exception. The worker sends the
production acknowledgement through a separate one-way acknowledgement pipe.
The production collector reads that pipe.

## Frozen Sources

Freeze:

```text
tinyvllm/engine/llm_engine.py:
  6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae
LLMEngine.call_model_runner_acknowledged:
  6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d
LLMEngine.bind_qwen35_loaded_checkpoint_candidates:
  82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c

tinyvllm/engine/model_runner.py:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849
ModelRunner.dispatch_command:
  9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342

tinyvllm/engine/model_runner_command_ack.py:
  ca28babca5cc725d8c9bf0e3e057fa4b0cabfd847bf0c052c40876fbc148c61b
```

The preflight imports only `model_runner_command_ack.py`. It must never import
`llm_engine.py` or `model_runner.py`; those three methods are compiled from
frozen AST segments.

## Worker Matrix

Run six fresh standard-library-only CPU processes:

```text
(TP=1, success)
(TP=1, local_binding_error)
(TP=2, success)
(TP=2, worker_binding_error)
(TP=2, worker_ack_exception)
(TP=2, worker_exit_without_ack)
```

Each outer attempt runs in a fresh process. TP2 attempts spawn exactly one
fresh rank1 child.

## Success Semantics

TP1 success executes the production Engine all-rank binder locally with no
dispatch and commits:

```text
(model_fingerprint, layout_fingerprint, bfloat16, timeout_s)
```

TP2 success requires:

- one production `dispatch_command()` call;
- one exact envelope with `requires_ack=True`;
- one real command-pipe send and receive;
- one production worker executor call;
- one production `ok` acknowledgement;
- one production collector call;
- rank0 and rank1 exact bound rows;
- exact cross-rank model/layout/dtype identity;
- one completion commit;
- exact repeat returns the same tuple with zero new dispatch.

## Failure Semantics

### Local Binding Error

TP1 returns the authoritative rank0 conflict row. The Engine binder raises the
fixed participant failure and leaves completion unset.

### Worker Binding Error

The rank1 target returns the authoritative conflict row. The worker
acknowledgement status remains `ok` because command execution itself
succeeded. The Engine binder observes the inner row status `error`, raises
with `rank=1`, and leaves completion unset.

### Worker Acknowledgement Exception

The rank1 target raises `RuntimeError("injected worker acknowledgement
exception")`. The production executor converts it to an `error`
acknowledgement. The collector rejects it, poisons itself, and the Engine
leaves completion unset.

### Worker Exit Without Acknowledgement

The worker receives the command and exits without sending an acknowledgement.
The acknowledgement pipe becomes readable with EOF, so the production
collector fails the receive, poisons itself, and the Engine leaves completion
unset before timeout completion. No child may remain alive.

## Source Closure

Inherit all 51 prerequisite source files and add:

```text
tinyvllm/engine/llm_engine.py
tinyvllm/engine/model_runner_command_ack.py
tools/qwen35_real_binding_engine_ack_transport_preflight.py
```

Total:

```text
54 unique source files
```

`model_runner.py` is already inherited. The 51 inherited files must match the
prerequisite byte for byte.

## Artifacts

Publish:

```text
engine_ack_transport_preflight.json
source_manifest.json
```

Use one SHA-bound prerequisite transfer, six fresh outer workers, a separate
finalizer, deterministic JSON, exact remote round trip, and atomic local and
remote publication. Preserve every failed run.

## Independent Verification

A standard-library-only verifier imports neither the gate nor TinyLLMForge.
It recomputes:

- prerequisite/result hashes;
- 54-file closure and source tree;
- all frozen file and AST method hashes;
- exact six-row ordering and unique outer PIDs;
- unique child PIDs for TP2 attempts;
- exact envelope, command ID, method, args, and `requires_ack`;
- exact success rows and completion configuration;
- inner binding error versus outer acknowledgement error distinction;
- collector poison and worker liveness evidence;
- exact-repeat zero-dispatch behavior;
- no residual child;
- local/remote inventory and result SHA equality.

## Static Safety

Require:

- zero imports or construction of `LLMEngine` and `ModelRunner`;
- exactly one compiled invocation site for each frozen method;
- exactly one `ModelRunnerCommandAckCollector` constructor;
- exactly one `execute_acknowledged_command` call site;
- no checkpoint metadata/read/load/adapter/target/model construction;
- no scheduler, `LLMEngine.step()`, CUDA, forward, or inference call;
- production `step()` remains free of candidate binding;
- schema-v2 canonical `NO_GO` and real worker hard rejection remain unchanged.

## Allowed Conclusion

Passing proves that already proven real per-rank binding rows can traverse the
production acknowledgement envelope, worker executor, real one-way pipe,
collector, Engine acknowledged call, and Engine all-rank binding validator at
TP1 and TP2. It proves completion commit only after all ranks return matching
bound identities, and fail-closed behavior for local inner errors, worker
inner errors, worker acknowledgement exceptions, and worker death.

It does not prove checkpoint loading inside Engine, ModelRunner/Engine
construction, shared-memory command broadcast, scheduler integration,
`LLMEngine.step()`, CUDA, forward/inference correctness, latency, throughput,
cache savings, GPU-memory savings, compression, or model quality.

The next safe gate is explicit live shared-memory dispatch using the same
acknowledgement contract, still outside `LLMEngine.step()` and without
checkpoint loading or CUDA/forward.

## Authoritative Evidence

The source-bound remote gate completed on
`sitian@10.232.195.203`:

```text
run tag:
  qwen35-engine-ack-transport-20260728-102828
remote run:
  /data00/home/sitian/sitian-workspace01/tllm/
  qwen35-real-binding-engine-ack-runs/
  qwen35-engine-ack-transport-20260728-102828
local run:
  experiments/qwen35_hybrid_state/
  qwen35-engine-ack-transport-20260728-102828
```

Authoritative identities:

```text
engine_ack_transport_preflight.json:
  8aeb571c3d56641e747a0d5c5e66314efe6b35b73320cb49e0340c0fe5fd42fb
source_manifest.json:
  f39c81677053271965caf82457fbbea91a220359dc954442826b8e7aebfe0c59
source tree:
  a041ebf7653e141dd96ebe31143ba00e5634c61c1a4bec68f17e7a7c6bba5cc8
prerequisite:
  79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a
```

The remote inventory contained exactly 54 staged source files and three
root-level JSON files: one immutable prerequisite and two results. Both
result SHA256 values matched the two local artifacts exactly. Remote CLI
validation passed and no attempt worker or finalizer remained alive.

Fresh process identities:

```text
outer attempt PIDs:
  2661964, 2662040, 2662127, 2662213, 2662274, 2662351
TP2 child PIDs:
  2662201, 2662269, 2662337, 2662429
```

The six modes proved:

```text
tp1_success:
  no dispatch, acknowledgement absent, completion committed
tp1_local_binding_error:
  no dispatch, rank0 inner error, completion unset
tp2_success:
  one envelope/send/receive/collector, ack ok, completion committed,
  exact replay with zero additional dispatch
tp2_worker_binding_error:
  ack ok, rank1 inner row error, completion unset, collector unpoisoned
tp2_worker_ack_exception:
  ack error, collector poisoned, completion unset
tp2_worker_exit_without_ack:
  ack absent, receive failure, collector poisoned, completion unset
```

The standard-library-only verifier:

```text
tools/verify_qwen35_real_binding_engine_ack_transport_gate.py
```

passed 455 checks locally and passed the same 455 checks against the remotely
staged source and remote artifacts. Its two focused tests include direct
tamper rejection of the worker-binding acknowledgement status.

Validation evidence:

```text
focused transport:
  7 tests passed
independent verifier:
  2 tests passed
ModelRunner command acknowledgement:
  14 tests passed
ModelRunner live acknowledgement wiring:
  11 tests passed
Engine all-rank candidate binding:
  9 tests passed
ModelRunner published candidate binding:
  4 tests passed
manifest-bound loader configuration:
  4 tests passed on the remote torch environment
real checkpoint worker boundary:
  6 tests passed
```

The seven-group regression matrix therefore passed 51 tests, with the
separate worker-boundary suite passing another six. Python 3.9 compilation,
`git diff --check`, and staged-zero checks passed.

The static audit found exactly one invocation site for each frozen method,
one acknowledgement collector constructor, and one production acknowledgement
executor call. It found zero Engine or ModelRunner imports/construction,
checkpoint calls, scheduler calls, `step()` calls, CUDA calls, forward calls,
or inference calls. The production `LLMEngine.step()` body contains zero
candidate-loading, publication, or binding references.

The worker continues to reject direct execution with the exact message:

```text
RuntimeError: real checkpoint load worker execution is not implemented; only the local safety dry-run is authorized
```

The immutable schema-v2 authoritative verifier remains `NO_GO`:

```text
experiments/qwen35_hybrid_state/
qwen35-2b-hybrid-v2-canonical-interleaved-drift-20260724-155900/
independent_verification.json:
  a6b0be0be57e6df62dfbcf7b4f05936218b57c417e2ac74c97306a3bc32e1f38
```

Final claim boundary:

```text
production envelope -> worker executor -> real one-way pipes ->
collector -> Engine acknowledged call -> all-rank binding validator:
  proven at TP=1 and TP=2
completion only after matching all-rank bound identities:
  proven
local inner error / worker inner error / worker exception / worker death:
  fail closed with the expected transport distinction
LLMEngine / ModelRunner construction:
  absent
shared-memory command broadcast:
  absent
checkpoint loading / scheduler / LLMEngine.step():
  absent
CUDA / forward / inference:
  absent
production latency / throughput / cache or GPU-memory savings / quality:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
