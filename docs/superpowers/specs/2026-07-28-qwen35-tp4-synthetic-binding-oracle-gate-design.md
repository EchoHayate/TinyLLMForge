# Qwen3.5 TP4 Synthetic Binding Oracle Gate Design

## Objective

Execute the production TP4 shared-memory and acknowledgement path through the
production Engine all-rank loaded-candidate binding validator using a separate,
explicitly synthetic, independently reproducible rank-row oracle:

```text
SHA-bound synthetic oracle artifact
  -> four synthetic binding rows
  -> one production shared-memory broadcast
  -> three production worker loops
  -> three production acknowledgement pipes
  -> ranked production collector
  -> LLMEngine.call_model_runner_acknowledged
  -> LLMEngine.bind_qwen35_loaded_checkpoint_candidates
  -> success commit or provenance mismatch rejection
```

This remains a construction-free CPU gate. It must not import or construct
`LLMEngine` or `ModelRunner`, read checkpoint metadata or tensor payloads,
construct model/target/adapter objects, enter the scheduler, call
`LLMEngine.step()`, initialize CUDA, execute forward, or run inference.

## Immutable Prerequisites

Use the completed TP4 shared-memory fan-out artifact:

```text
run:
  qwen35-tp4-shm-fanout-20260728-115046
artifact:
  tp4_shared_memory_fanout_preflight.json
artifact SHA256:
  ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a
source tree:
  ec7b0dee43a06c47b72f8ac14ab26518845f57f070e6c27d394bb4c328644403
```

Use the immutable synthetic oracle:

```text
artifact:
  experiments/qwen35_hybrid_state/
  qwen35-tp4-synthetic-binding-oracle-v1/
  synthetic_binding_oracle.json
artifact SHA256:
  1fdc3d64178308d6d26242805433ee31012890f6824a13826075db1e6937431e
schema:
  qwen35.tp4-synthetic-binding-oracle.v1
provenance:
  synthetic-construction-free-oracle
claim boundary:
  not-real-checkpoint-binding
tensor payload:
  absent
```

Both prerequisites are immutable and independently hashed before any attempt.

## Oracle Construction

The oracle contains public JSON descriptors, canonicalized as UTF-8 JSON with
sorted keys and compact separators.

The canonical model descriptor hashes to:

```text
b48e29c9ea266197c56fe3e9133c65d378c682e9437d9e1299c4fa93d0241bd9
```

The canonical TP4 layout descriptor hashes to:

```text
fe2db881dc909cbd05894a4e194273f4692caed1c4df967e810330324248e2c6
```

The layout descriptor records:

```text
tensor_parallel_size:
  4
layer_pattern:
  linear_attention
  linear_attention
  linear_attention
  full_attention
```

The oracle uses `bfloat16`. Its alternate model/layout fingerprints are
derived from the same descriptors with `revision="mismatch-v1"`, not chosen
as arbitrary opaque constants.

## Approaches Considered

### Reuse Real TP2 Rows for Ranks 2 and 3

Rejected. Duplicating TP2 rows and changing participant IDs would create no
independent TP4 provenance and could be misread as real rank2/rank3 binding.

### Construct Four Real Checkpoint Candidates

Rejected. That crosses into checkpoint loading, target/model construction,
large CPU memory, and eventually CUDA-sensitive ownership beyond this gate.

### Explicit Synthetic Oracle with Production Binder

Selected. The oracle is a first-class artifact whose schema, descriptors,
fingerprints, cases, and synthetic claim boundary can be recomputed without
TinyLLMForge. The production binder consumes the resulting rows unchanged.

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

The gate imports only the acknowledgement module and the TP4 transport helper.
It never imports production Engine or ModelRunner modules.

## Attempt Matrix

Run four fresh outer processes:

```text
tp4_synthetic_binding_success
tp4_synthetic_rank2_model_mismatch
tp4_synthetic_rank2_layout_mismatch
tp4_synthetic_rank2_dtype_mismatch
```

Every attempt uses one fresh shared-memory segment, three real Events, three
real acknowledgement pipes, and three fresh worker-loop children.

## Success Semantics

All four oracle rows contain:

```text
operation:
  bind_loaded_checkpoint_candidate
status:
  bound
model_fingerprint:
  b48e29...
layout_fingerprint:
  fe2db8...
dtype:
  bfloat16
detail:
  empty
```

Workers deliberately complete in send order `(3, 2, 1)`. The production
collector returns `(1, 2, 3)`. The production Engine binder commits:

```text
(
  b48e29c9ea266197c56fe3e9133c65d378c682e9437d9e1299c4fa93d0241bd9,
  fe2db881dc909cbd05894a4e194273f4692caed1c4df967e810330324248e2c6,
  bfloat16,
  timeout_s,
)
```

An exact repeat returns the same ordered row tuple with zero new binding
dispatch.

## Mismatch Semantics

Each mismatch attempt changes only rank2 and one identity field:

```text
model mismatch:
  alternate model fingerprint
layout mismatch:
  alternate layout fingerprint
dtype mismatch:
  float16
```

All command acknowledgements remain `ok`, because every worker method returns
a valid bound row. The production collector remains healthy and returns ranks
`(1, 2, 3)`. The production Engine binder rejects the specific mismatched
field and leaves both completion fields unset. Cleanup still sends one
fire-and-forget exit envelope to all workers.

## Evidence

Each attempt records:

- exact TP4 and oracle prerequisite hashes;
- oracle provenance, claim boundary, descriptor hashes, and case name;
- one outer PID and three child PIDs;
- one shared-memory name and three attach confirmations;
- binding and exit envelopes and payload byte counts;
- Event/read/executor counts per rank;
- acknowledgement send and collector return order;
- acknowledgement statuses and exact synthetic row results;
- completion configuration/rows or mismatch detail;
- repeat zero-dispatch state;
- child exit/join and segment unlink state.

## Source Closure and Artifacts

Inherit the exact 56-file TP4 source closure and add:

```text
tools/qwen35_tp4_synthetic_binding_oracle_preflight.py
```

Total:

```text
57 unique source files
```

Stage the separate oracle artifact as a second prerequisite. Publish:

```text
tp4_synthetic_binding_oracle_preflight.json
source_manifest.json
```

Use deterministic staging, four fresh attempts, a separate finalizer, atomic
local/remote publication, and a unique run tag. Preserve failed runs.

## Independent Verification

A standard-library-only verifier imports neither TinyLLMForge nor either gate.
It recomputes:

- both prerequisite hashes and schemas;
- both canonical descriptor hashes and both alternate hashes;
- explicit synthetic provenance and `not-real-checkpoint-binding` boundary;
- 57-file source closure and source tree;
- frozen file/method hashes and signatures;
- four unique outer and twelve unique child PIDs;
- four unique shared-memory names;
- exact envelopes, payload bytes, Event/read/executor counts;
- reverse send order and ranked collector order;
- success completion and exact-repeat zero dispatch;
- the single changed rank2 field in each mismatch case;
- healthy collector and unset completion for every mismatch;
- cleanup, inventory, and local/remote hash equality.

Tamper tests must reject changed oracle provenance and a mismatch case that
changes more than its one authorized rank2 field.

## Static Safety

Require:

- zero Engine/ModelRunner import or construction;
- exact frozen method invocation sites;
- no fixed `tinyvllm` shared-memory name;
- no checkpoint metadata/payload/load calls;
- no target/adapter/model construction;
- no scheduler, `step()`, CUDA, forward, or inference calls;
- exact worker hard rejection and schema-v2 canonical `NO_GO` unchanged.

## Claim Boundary

Passing proves production TP4 transport plus production all-rank binder
behavior for an independently reproducible synthetic identity oracle. It
proves homogeneous success commit, deterministic rank ordering, mismatch
rejection, exact-repeat zero dispatch, and complete resource cleanup.

It does not prove any real rank loaded or bound a checkpoint candidate. It
does not prove Engine/ModelRunner construction, checkpoint loading, scheduler
integration, CUDA, forward, inference, correctness, latency, throughput,
cache savings, GPU-memory savings, compression, or quality.

## Authoritative Result

The gate completed on the remote server with:

```text
run:
  qwen35-tp4-synthetic-binding-20260728-122021
source tree:
  e88236ebe4f97ddecf55004e4bbcdb46a677462f183b6724031d85d8648a6de0
result:
  803c8fac331eeee82b90013e0b0872de8f079661b6dd1ba43225fb446006cce4
source manifest:
  643e8d1e24e97ee085f060559999fa1ad1b7608c7c1998c4aaeef9610cc7ccdb
```

The immutable prerequisites remained:

```text
TP4 fan-out:
  ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a
synthetic oracle:
  1fdc3d64178308d6d26242805433ee31012890f6824a13826075db1e6937431e
schema-v2 canonical verifier:
  a6b0be0be57e6df62dfbcf7b4f05936218b57c417e2ac74c97306a3bc32e1f38
schema-v2 classification:
  NO_GO
```

Fresh outer PIDs:

```text
357700, 358006, 358303, 358588
```

Fresh child PIDs:

```text
357782, 357783, 357784
358092, 358093, 358094
358383, 358384, 358385
358672, 358673, 358674
```

Unique shared-memory names:

```text
qwen35-tp4_synthet-357700-d43a
qwen35-tp4_synthet-358006-e29f
qwen35-tp4_synthet-358303-f18a
qwen35-tp4_synthet-358588-6b2f
```

All sixteen PIDs were absent after finalization. All four names were
independently non-attachable. The remote source closure contained exactly 57
files, no `__pycache__`, and no `.pyc`; the run directory contained exactly
two immutable inputs and two result files.

The homogeneous case returned acknowledgement send order `(3, 2, 1)` and
collector order `(1, 2, 3)`, committed the canonical four-row tuple, and
repeated with zero new binding dispatch. The model/layout/dtype cases changed
only the authorized rank2 field. Every acknowledgement remained `ok`, the
collector remained healthy, and the production binder rejected each mismatch
without setting either completion field.

The independent standard-library verifier passed 720 checks locally and the
same 720 checks against the remote staged source and artifacts. Its four
tests cover the real fixture plus provenance tamper, unauthorized rank2
second-field tamper, and a re-signed source-tree attack that injects a
production Engine import.

Validation:

```text
new synthetic harness: 6 tests passed
new independent verifier: 4 tests passed
TP4 fan-out harness/verifier: 7 + 3 tests passed
TP2 Engine acknowledgement harness/verifier: 7 + 2 tests passed
ModelRunner command acknowledgement: 14 tests passed
ModelRunner live acknowledgement wiring: 11 tests passed
Engine all-rank candidate binding: 9 tests passed
real checkpoint worker boundary: 6 tests passed
manifest-bound loader configuration: 4 tests passed remotely
```

The local matrix passed 69 tests. Python compilation, static safety audit,
the exact hard-rejection RuntimeError, canonical schema-v2 `NO_GO` SHA,
`git diff --check`, and staged-zero passed.

The next safe boundary is a source-bound TP4 real-candidate provenance gate:
four rank-specific candidates must be produced by the already authorized
loader path and then passed through this proven transport/binder path, still
outside scheduler, `LLMEngine.step()`, CUDA, forward, and inference. It must
first prove bounded CPU ownership and complete cleanup before any runtime or
performance claim.
