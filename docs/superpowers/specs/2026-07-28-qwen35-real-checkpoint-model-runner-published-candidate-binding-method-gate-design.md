# Qwen3.5 Real-Checkpoint ModelRunner Published-Candidate Binding Method Gate Design

## Objective

Execute the existing production
`ModelRunner.bind_published_qwen35_loaded_checkpoint_candidate()` method on a
complete approved real Qwen3.5 candidate at TP=1 and TP=2 without importing,
constructing, or initializing `ModelRunner`.

The gate proves the exact private rank-local transition:

```text
authorized complete real candidate
  -> production local publication method
  -> published-candidate outer binding method
  -> production candidate binder
  -> production owner binder
  -> owner + runtime bridge + runtime identity visibility
```

It also proves that a pre-existing incompatible runtime bridge is converted
into a bounded participant error row before any owner or runtime-identity
mutation.

The gate begins after:

```text
qwen35-model-runner-load-publish-20260728-092500
```

It does not execute ModelRunner `__init__`, Engine transport, scheduler,
cross-rank orchestration, CUDA, forward, or inference.

## Approaches Considered

### Execute Only the Outer Method with a Fake Inner Binder

Rejected as the authoritative path. Existing dependency-light tests already
cover the outer row formatting with a fake binder. That does not prove the
real owner and runtime-identity mutations on a real loaded candidate.

### Import or Construct ModelRunner

Rejected. Importing or constructing ModelRunner enters unrelated runtime and
CUDA-sensitive initialization beyond the current correctness boundary.

### Frozen Multi-Method AST Composition

Selected. Freeze `model_runner.py`, extract and compile four exact production
methods, inject only their required production globals, bind the extracted
methods onto a minimal private runner shell, and execute the outer published
candidate binder on a real candidate.

The four methods are:

```text
publish_qwen35_loaded_checkpoint_candidate(self, candidate)
bind_qwen35_hybrid_model_owner(self, owner)
bind_qwen35_loaded_checkpoint_candidate(self, candidate)
bind_published_qwen35_loaded_checkpoint_candidate(self)
```

This approach proves the complete production binding mutation while retaining
the no-import/no-construction boundary.

## Immutable Prerequisite

Use the completed load-and-publish artifact:

```text
model_runner_load_and_publish_preflight.json
SHA256:
  d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18
source tree:
  a1bf0161eeedf3c73fb176a0f26ab2156bb3d944096db187a9c83eeb98ae5cc8
```

The prerequisite must retain six fresh-process rows, exact complete-oracle
value hashes, cleanup, whole-scope collection, memory evidence, CUDA false,
and the exact 50-file source closure.

Only its three success rows are value oracles for TP `(1,0)`, `(2,0)`, and
`(2,1)`.

## Frozen Production Sources

Freeze:

```text
tinyvllm/engine/model_runner.py
file SHA256:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849

publish_qwen35_loaded_checkpoint_candidate source SHA256:
  37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f

bind_qwen35_hybrid_model_owner source SHA256:
  462e2fefe22e90e60b85c786de6a95e7eaaae31bd9b257025088cd767555ee25

bind_qwen35_loaded_checkpoint_candidate source SHA256:
  a14e6856ad74eb935116075ee6fe81516c8e212f89914e0fcd55bb39e86d63e0

bind_published_qwen35_loaded_checkpoint_candidate source SHA256:
  aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd
```

The extracted methods may receive only these production globals:

```text
Qwen35HybridModelOwner
Qwen35LoadedCheckpointCandidate
_bind_qwen35_hybrid_prefix_runtime_identity
```

The preflight must independently validate exact method arguments, globals,
call targets, mutation ordering, exception boundary, and result-row shape.
Any file, source, dependency-name, ordering, or structural drift rejects the
gate before loading.

## Candidate Preparation

Use the same approved checkpoint and authorization as the completed
load-and-publish gate:

```text
checkpoint_dir:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/
  qwen35-2b-hybrid-acquire-20260723-222004/model
model_fingerprint:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
max_tensor_bytes:
  1017118720
authorization_sha256:
  10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4
```

Each worker builds one private prepared target and invokes
`build_qwen35_authorized_checkpoint_candidate_loader(...)` once. The adapter
must call the target provider exactly once and return one exact complete
`Qwen35LoadedCheckpointCandidate`.

The candidate is published by the extracted production local publication
method, not by direct slot mutation.

## Private Runner Shell

The private shell initially exposes:

```text
rank
model = candidate.owner.model
qwen35_loaded_checkpoint_candidate_slot = production one-shot slot
hybrid_state_runtime_bridge = None or injected conflicting object
qwen35_hybrid_model_owner = None
qwen35_hybrid_prefix_restore_owner = None
qwen35_hybrid_prefix_restore_participant = None
qwen35_hybrid_prefix_publication_participant = None
qwen35_hybrid_prefix_runtime_identity = None
qwen35_hybrid_prefix_runtime_identity_owner = None
```

It receives bound versions of the three extracted binding methods. The shell
is not a ModelRunner instance and no ModelRunner constructor or initializer
executes.

## Worker Matrix

Run exactly six fresh CPU-only processes in fixed order:

```text
(TP=1, rank=0, success)
(TP=1, rank=0, injected_bridge_conflict)
(TP=2, rank=0, success)
(TP=2, rank=0, injected_bridge_conflict)
(TP=2, rank=1, success)
(TP=2, rank=1, injected_bridge_conflict)
```

## Success Transaction

Each success worker:

1. loads one exact complete real candidate through the authorized adapter;
2. publishes it once through the extracted production local publication
   method;
3. invokes the extracted outer published-candidate binding method once;
4. requires one exact candidate-binder call and one exact owner-binder call;
5. requires the fixed `bound` participant row;
6. requires `qwen35_hybrid_model_owner is candidate.owner`;
7. requires `hybrid_state_runtime_bridge is candidate.owner.runtime_bridge`;
8. requires one exact runtime identity with:
   - approved model fingerprint;
   - candidate owner pool layout fingerprint;
   - `bfloat16`;
9. requires `qwen35_hybrid_prefix_runtime_identity_owner is candidate.owner`;
10. rehashes all 320 bindings and 26 phases against the prerequisite oracle;
11. clears selected destinations, drops the whole private scope, and proves
    every tracked object is collected.

The gate invokes the outer binding method only once. Existing dependency-light
tests retain exact-repeat coverage.

## Injected Bridge Conflict

Each failure worker installs one private object as
`hybrid_state_runtime_bridge` before binding.

The production owner binder must observe that the existing bridge is not
`candidate.owner.runtime_bridge` and raise:

```text
RuntimeError: a different hybrid state runtime bridge is already installed
```

The outer published-candidate binding method must catch that error and return:

```text
participant_id: exact rank
operation: bind_loaded_checkpoint_candidate
status: error
model_fingerprint: ""
layout_fingerprint: ""
dtype: ""
detail:
  RuntimeError: a different hybrid state runtime bridge is already installed
```

The failure row must prove:

- one local publication method call;
- one outer binding method call;
- one candidate-binder call;
- one owner-binder call;
- the production slot still exposes the exact candidate;
- the injected bridge remains unchanged;
- owner, runtime identity, and identity-owner fields remain `None`;
- the complete candidate values still match the oracle;
- cleanup and whole-scope collection complete.

This is a post-publication, pre-binding-mutation atomicity test.

## Cleanup and Memory

Use the same selected-only initialization, reverse unique-object clear,
non-selected preservation, pool preservation, tensor identity, and weakref
collection contract as the completed load-and-publish gate.

Track:

```text
runner shell
production slot
injected bridge when present
candidate
owner
runtime bridge
runtime identity on success
model
pool
prepared target
```

Use the same correctness ceilings:

```text
TP=1 total/post-Torch/post-metadata:
  10485760 / 10223616 / 9961472 KiB
TP=2:
  7340032 / 7077888 / 6815744 KiB
```

These remain correctness ceilings, not performance claims.

## Source Closure

Freeze the 50-file load-and-publish closure and add:

```text
tools/qwen35_real_checkpoint_model_runner_published_candidate_binding_preflight.py
```

Total:

```text
51 unique source files
```

The inherited 50 files must match the prerequisite artifact byte for byte.
The frozen ModelRunner file and all four method source hashes remain
independently checked.

## Artifacts and Orchestration

Outputs:

```text
model_runner_published_candidate_binding_preflight.json
source_manifest.json
```

Use one SHA-bound prerequisite transfer, six fixed fresh workers, a separate
finalizer, deterministic JSON, exact remote round trip, and atomic local and
remote publication. Any worker failure prevents authoritative result
publication. Preserve every failed run directory.

Exact inventory:

```text
remote source files: 51
remote root prerequisites: 1
remote root results: 2
local result artifacts: 2
```

## Independent Verification

A standard-library-only verifier imports neither the gate nor TinyLLMForge.
It verifies:

- prerequisite and result SHA256 values;
- 51-file closure and source tree;
- exact ModelRunner file SHA;
- all four independently parsed method source hashes and structures;
- six fixed rows and unique PIDs;
- success publication, owner/bridge/identity visibility, and exact bound row;
- injected conflict error row and pristine binding fields;
- complete oracle hashes;
- cleanup, weakref collection, memory, and CUDA false;
- exact local/remote inventories and result SHA equality.

## Safety Audits

The preflight must contain:

- exactly one authorized-adapter-builder call site;
- exactly one production publication-slot constructor call site;
- exactly one invocation site for each extracted production method;
- no import of `tinyvllm.engine.model_runner`;
- no ModelRunner construction or `ModelRunner.__init__` call;
- no direct streamed-loader or `target.take()` call;
- no Engine, scheduler, step, CUDA operation, forward, or inference call;
- only two read-only `torch.cuda.is_initialized()` observations;
- no production-module modification.

The real worker hard rejection remains exact and schema-v2 canonical `NO_GO`
remains unchanged.

## Tests

Focused TDD covers:

- strict prerequisite and 51-file source inheritance;
- exact four-method file/source/dependency/structure validation;
- isolated multi-method compilation with three production globals;
- success owner/bridge/runtime-identity mutation;
- injected bridge-conflict error row and pristine binding state;
- exact value hashes, cleanup, and collection;
- row/memory schemas;
- six-process orchestration and partial-finalization rejection;
- Python 3.9 compatibility and CLI modes.

Regression includes load-and-publish, local publication, private publication,
ownership, production slot, atomic candidate binding, runtime identity,
ModelRunner binding, Engine all-rank binding, loader-core, complete
transaction, request/configuration, factory, loader, metadata, reader,
assignment, authorization, and safety scripts.

## Allowed Conclusion

Passing proves that the exact production published-candidate binding method,
candidate binder, and owner binder can atomically install one complete
real-checkpoint candidate's owner, runtime bridge, and runtime identity into a
private ModelRunner-shaped shell at TP=1 and TP=2. It also proves fail-closed
bounded error behavior for an incompatible pre-existing bridge before any
binding mutation.

It does not prove ModelRunner initialization, Engine transport, cross-rank
runtime coordination, scheduler integration, CUDA, forward/inference
correctness, latency, throughput, cache savings, GPU-memory savings,
compression, or model quality.

The next safe gate is an explicit dependency-light Engine acknowledgement
transport preflight over already proven per-rank bound rows, still absent from
`LLMEngine.step()` and without CUDA/forward.

## Authoritative Result

The gate completed with authoritative source-bound run:

```text
qwen35-model-runner-published-binding-20260728-100419
```

Exact artifact identities:

```text
source tree:
  0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785
model_runner_published_candidate_binding_preflight.json:
  79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a
source_manifest.json:
  27e1e81f84fa8c58a8df065a08598991bec8dd6e35d19a694221dde478d834f8
```

The six fresh worker PIDs were:

```text
2325799 2329647 2333989 2336542 2340270 2345004
```

Exact worker memory deltas
`total/post-Torch/post-metadata` were:

```text
TP=1 rank0 success:
  7849360 / 7506792 / 7367052 KiB
TP=1 rank0 injected bridge conflict:
  7862236 / 7519044 / 7379144 KiB
TP=2 rank0 success:
  4569140 / 4225972 / 4086168 KiB
TP=2 rank0 injected bridge conflict:
  4565276 / 4223036 / 4082772 KiB
TP=2 rank1 success:
  4567700 / 4225804 / 4085556 KiB
TP=2 rank1 injected bridge conflict:
  4567640 / 4225608 / 4085956 KiB
```

All success rows returned the exact `bound` participant row and exposed the
candidate owner, its runtime bridge, one runtime identity with the approved
model fingerprint and owner layout fingerprint, and the exact identity owner.
All conflict rows returned:

```text
RuntimeError: a different hybrid state runtime bridge is already installed
```

The injected bridge remained installed while owner, runtime identity, and
identity-owner fields remained empty. Every worker called the authorized
adapter, target provider, production publication method, outer binder,
candidate binder, and owner binder exactly once.

All six rows matched the immutable prerequisite's 320 binding hashes, 26 phase
hashes, aggregate hash, loader statistics, and 24 alias groups. The flat
prerequisite hash rows were deterministically reconstructed into the canonical
candidate-validator oracle using the frozen `PHASE_BINDING_RUNS`; focused TDD
also proved that reconstruction equals the canonical complete-transaction
artifact for all three TP/rank rows.

Cleanup cleared all 296 unique selected destinations, preserved non-selected
values, tensor identities, and pool state, and collected every tracked private
object. CUDA remained uninitialized and no model/attention forward path ran.

The standard-library-only independent verifier imported neither the gate nor
TinyLLMForge and passed:

```text
checks=2475
rows=6
unique PIDs=6
source files=51
```

It independently rehashed the complete source closure, source tree,
`model_runner.py`, all four extracted method segments, every one of the
six-by-320 binding hashes, all phase and aggregate hashes, row schemas,
binding visibility, rejection atomicity, memory ceilings, cleanup, collection,
and CUDA/forward absence. A tampered aggregate hash was rejected against the
immutable prerequisite.

Exact inventory was:

```text
remote source files: 51
remote root input artifacts: 1
remote root result artifacts: 2
local result artifacts: 2
local/remote result SHA256: exact equality
```

Focused tests passed 8/8 and verifier tests passed 2/2. Python 3.9 compilation,
local and remote CLI validation, and 22 regression groups passed. Static AST
audit found exactly one authorized-adapter-builder call site, one production
slot constructor, and one invocation site for each of the four extracted
methods; zero ModelRunner imports/construction, direct streamed-loader calls,
`target.take()` calls, Engine/scheduler calls, forward calls, inference calls,
or CUDA execution calls; and exactly two read-only
`torch.cuda.is_initialized()` observations.

The failed pre-artifact run:

```text
qwen35-model-runner-published-binding-20260728-100109
```

is preserved remotely. It failed before publication because the first
implementation passed the flattened load-and-publish row directly to the
candidate validator, which requires canonical `binding_results` and
`phase_results`. No result artifact or residual worker was produced. TDD then
added and verified the lossless oracle reconstruction before the new run tag.

Final claim boundary:

```text
exact production local publication -> published-candidate outer binder ->
candidate binder -> owner binder on complete real candidates:
  proven at TP=1 and TP=2
owner / runtime bridge / runtime identity visibility:
  proven
post-publication pre-mutation incompatible-bridge rejection:
  proven
320/26/aggregate value preservation, cleanup, and collection:
  proven
ModelRunner import / construction / initialization:
  absent
Engine transport / acknowledgement / scheduler / LLMEngine.step():
  absent
CUDA allocation or operators / forward / inference:
  absent
production latency / throughput / cache or GPU-memory savings / quality:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
