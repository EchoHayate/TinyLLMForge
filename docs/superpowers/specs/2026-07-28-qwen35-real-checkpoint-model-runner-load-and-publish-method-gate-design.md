# Qwen3.5 Real-Checkpoint ModelRunner Load-and-Publish Method Gate Design

## Objective

Execute the existing production
`ModelRunner.load_and_publish_qwen35_checkpoint_candidate(request)` method body
on complete approved real Qwen3.5 checkpoint candidates at TP=1 and TP=2,
without importing, constructing, or initializing `ModelRunner`.

The gate proves the exact production loader-to-publication transaction:

```text
bounded request validation
  -> installed authorized loader
  -> exact complete candidate
  -> one-shot local publication slot
  -> completion state
  -> published participant row
```

It also proves fail-closed behavior when publication rejects after the real
candidate has loaded but before production-slot mutation.

It begins after:

```text
qwen35-model-runner-local-publication-20260728-090014
```

It does not execute ModelRunner `__init__`, candidate binding, Engine
transport, scheduler, CUDA, forward, or inference.

## Approaches Considered

### Import or Construct ModelRunner

Rejected. Importing the module or constructing the class would transitively
enter runtime dependencies, CUDA-sensitive initialization, and unrelated
Engine state. That is beyond the current gate.

### Handwritten Equivalent Transaction

Rejected. A copied implementation could validate the idea but would not prove
the production method body.

### Frozen AST Extraction

Selected. Freeze `model_runner.py`, parse exactly one `ModelRunner` class and
one `load_and_publish_qwen35_checkpoint_candidate` method, validate the method
source and structure, compile only that method into an isolated namespace,
inject its two required production globals, and call it as an unbound method
on a minimal private runner shell.

## Immutable Prerequisites

Use:

```text
complete oracle:
  complete_checkpoint_transaction_preflight.json
  SHA256:
    7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176

ModelRunner-local publication gate:
  model_runner_local_publication_preflight.json
  SHA256:
    f8f78ae574991eb3f16aed57b4275cf76a409fa553e01597f5179c41eb158b15
  source tree:
    d3eb52326d8e9d9a744f4641877c90a41468d26f94cbe31eda5ee04fe4d2201a
```

The publication-method artifact must retain six fresh-process rows, exact
success/failure value hashes, method identity, cleanup, whole-scope
collection, and CUDA false.

## Frozen Production Method

Freeze:

```text
tinyvllm/engine/model_runner.py
file SHA256:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849

ModelRunner.load_and_publish_qwen35_checkpoint_candidate
source SHA256:
  9134c5bad8c4127714e07ffd8af56209c247a746e9f0d0ceceb60227c1358612
```

The extracted method may receive only these production globals:

```text
Qwen35LoadedCheckpointCandidate
validate_qwen35_checkpoint_candidate_load_request
```

The preflight must independently validate that the method:

- accepts exactly `self, request`;
- validates the request before loader invocation;
- reads loader, authorization, completion, request, and slot state only from
  the runner shell;
- calls the installed loader exactly once on the fresh path;
- requires exact `Qwen35LoadedCheckpointCandidate` type and fingerprint;
- calls the runner slot's `publish(candidate)` exactly once;
- writes completion request and configuration only after publication;
- catches `Exception`, not `BaseException`;
- returns the fixed published/error participant row shape;
- contains no binding, Engine, scheduler, CUDA, forward, or inference call.

Any file, source, dependency-name, ordering, or structural drift rejects the
gate before loading.

## Private Runner Shell

The private shell exposes exactly:

```text
rank
qwen35_checkpoint_candidate_loader
qwen35_checkpoint_candidate_loader_authorization_sha256
qwen35_checkpoint_candidate_load_configuration
qwen35_checkpoint_candidate_load_request
qwen35_loaded_checkpoint_candidate_slot
```

Initial completion configuration and request are `None`. The installed loader
is the existing authorized adapter bound to one private prepared target. The
authorization digest is the exact approved digest:

```text
10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4
```

The request is the exact production
`Qwen35CheckpointCandidateLoadRequest` with:

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

## Worker Matrix

Run exactly six fresh CPU-only processes in fixed order:

```text
(TP=1, rank=0, success)
(TP=1, rank=0, injected_publication_failure)
(TP=2, rank=0, success)
(TP=2, rank=0, injected_publication_failure)
(TP=2, rank=1, success)
(TP=2, rank=1, injected_publication_failure)
```

## Success Transaction

Each success worker:

1. creates metadata, pool, prepared target, exact authorized adapter, request,
   production slot, and private runner shell;
2. invokes the extracted production method exactly once;
3. requires exactly one adapter and provider call;
4. requires exactly one production-slot publish;
5. requires the exact fixed `published` row with participant rank and model
   fingerprint;
6. requires the slot to expose the exact candidate, owner, and fingerprint;
7. requires completion request identity to be the validated request;
8. requires completion configuration to equal the exact four-field tuple;
9. rehashes all loaded values against the complete oracle;
10. clears selected tensors, drops the whole private scope, and proves every
    tracked object is collected.

The method is not called a second time in this gate. Dependency-light exact
retry behavior remains covered by
`tools/test_model_runner_authorized_checkpoint_loader.py`.

## Injected Publication Failure

The failure worker installs the same real authorized loader but points the
runner shell at a private proxy slot. The proxy:

1. starts with `candidate is None`;
2. records the exact candidate identity;
3. increments one publication-attempt count;
4. raises:

   ```text
   RuntimeError("injected ModelRunner load-and-publish failure")
   ```

5. never delegates to the production slot.

The production method must catch that `RuntimeError` and return:

```text
participant_id: exact rank
operation: load_checkpoint_candidate
status: error
model_fingerprint: ""
detail:
  RuntimeError: injected ModelRunner load-and-publish failure
```

The failure row must prove:

- one production-method call;
- one real adapter/provider call;
- one proxy publication attempt;
- zero production-slot publications;
- production slot remains empty;
- completion request and configuration remain `None`;
- the real loaded candidate values match the complete oracle before cleanup;
- cleanup and whole-scope collection complete.

This is a post-load, pre-publication-mutation atomicity test.

## Cleanup and Memory

Use the same selected-only initialization, reverse unique-object clear,
non-selected preservation, pool preservation, tensor identity, and weakref
collection contract as the completed publication-method gate.

Track:

```text
runner shell
production slot
proxy slot when present
request
candidate
owner
model
pool
prepared target
```

Use the same correctness ceilings:

```text
TP=1 total/post-Torch/post-metadata:
  10485760 / 10223616 / 9961472 KiB
TP=2 total/post-Torch/post-metadata:
  7340032 / 7077888 / 6815744 KiB
```

These remain correctness ceilings, not performance claims.

## Source Closure

Freeze the 49-file publication-method closure and add:

```text
tools/qwen35_real_checkpoint_model_runner_load_and_publish_preflight.py
```

Total:

```text
50 unique source files
```

The inherited 49 files must match the publication-method artifact byte for
byte. The frozen ModelRunner file and both relevant method source hashes
remain independently checked.

## Artifacts and Orchestration

Outputs:

```text
model_runner_load_and_publish_preflight.json
source_manifest.json
```

Use two SHA-bound prerequisite transfers, six fixed fresh workers, a separate
finalizer, deterministic JSON, exact remote round trip, and atomic local and
remote publication. Any worker failure prevents authoritative result
publication. Preserve every failed run directory.

Exact inventory:

```text
remote source files: 50
remote root prerequisites: 2
remote root results: 2
local result artifacts: 2
```

## Independent Verification

A standard-library-only verifier imports neither the gate nor TinyLLMForge.
It verifies:

- both prerequisite and result SHA256 values;
- 50-file closure and source tree;
- exact ModelRunner file SHA;
- independently parsed load-and-publish method source SHA, dependencies, and
  structural order;
- six fixed rows and unique PIDs;
- exact success participant row, publication, and completion state;
- exact failure participant row, empty production slot, and empty completion
  state;
- complete oracle hashes;
- cleanup, weakref collection, memory, and CUDA false;
- exact local/remote inventories and result SHA equality.

## Safety Audits

The preflight must contain:

- exactly one authorized-adapter-builder call site;
- exactly one production publication-slot constructor call site;
- exactly one extracted load-and-publish method invocation call site;
- no import of `tinyvllm.engine.model_runner`;
- no ModelRunner construction or `ModelRunner.__init__` call;
- no direct streamed-loader, `target.take()`, candidate binding, Engine,
  scheduler, step, forward, or inference call;
- no direct production-slot publication outside the extracted method;
- only two read-only `torch.cuda.is_initialized()` observations;
- no production-module modification.

The real worker hard rejection remains exact and schema-v2 canonical `NO_GO`
remains unchanged.

## Tests

Focused TDD covers:

- strict prerequisite and 50-file source inheritance;
- exact method file/source/dependency/structure validation;
- isolated method compilation with only two production globals;
- success published row and completion state;
- injected publication failure error row and pristine completion state;
- exact value hashes, cleanup, and collection;
- row/memory schemas;
- six-process orchestration and partial-finalization rejection;
- Python 3.9 compatibility and CLI modes.

Regression includes the prior publication-method gate, private publication,
ownership, production slot, ModelRunner loader/binding, Engine binding,
loader-core, complete transaction, request/configuration, factory, loader,
metadata, reader, assignment, authorization, and safety scripts.

## Allowed Conclusion

Passing proves that the exact production
`ModelRunner.load_and_publish_qwen35_checkpoint_candidate(request)` method can
validate one approved request, invoke the existing authorized real loader,
accept a complete candidate, publish it once into the existing production
slot, and commit completion state at TP=1 and TP=2 without constructing
ModelRunner. It also proves fail-closed behavior when publication rejects
after real loading but before production-slot mutation.

It does not prove ModelRunner initialization, candidate binding, Engine
transport, cross-rank runtime coordination, scheduler integration, CUDA,
forward/inference correctness, latency, throughput, cache savings, GPU-memory
savings, compression, or model quality.

The next safe gate is a dependency-light execution of
`ModelRunner.bind_published_qwen35_loaded_checkpoint_candidate()` using a
private runner shell and the already published real candidate, still without
constructing ModelRunner or invoking Engine/CUDA/forward.

## Authoritative Result

The gate is complete. The authoritative source-bound run is:

```text
qwen35-model-runner-load-publish-20260728-092500
```

An earlier unique run remains preserved remotely:

```text
qwen35-model-runner-load-publish-20260728-092223
```

That earlier run ended before authoritative artifact publication when its SSH
ControlMaster connection closed with
`Connection closed by UNKNOWN port 65535`. Investigation found no residual
worker, OOM, or memory-ceiling failure. The ControlMaster was rebuilt and the
gate reran under the new tag without changing code or memory ceilings.

The authoritative run executed the exact AST-extracted production
`ModelRunner.load_and_publish_qwen35_checkpoint_candidate(self, request)`
method once in each of six fresh CPU-only processes:

```text
TP=1 rank0 success:
  PID 1764145
  memory 7861760/7518348/7377044 KiB
  aggregate 4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963
TP=1 rank0 injected publication failure:
  PID 1768441
  memory 7863652/7521184/7379928 KiB
  aggregate 4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963
TP=2 rank0 success:
  PID 1772552
  memory 4569936/4226532/4085132 KiB
  aggregate 4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041
TP=2 rank0 injected publication failure:
  PID 1775523
  memory 4568012/4225044/4083808 KiB
  aggregate 4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041
TP=2 rank1 success:
  PID 1778235
  memory 4569872/4226556/4084844 KiB
  aggregate 5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca
TP=2 rank1 injected publication failure:
  PID 1780765
  memory 4562500/4219244/4077532 KiB
  aggregate 5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca
```

Every row invoked one installed authorized loader and one target provider.
All six rows matched the complete oracle's exact 320 binding hashes, 26 phase
hashes, aggregate hash, and 24 alias groups. They cleared all 296 unique
selected destinations, preserved tensor identity, non-selected values, and
pool state, and collected the runner shell, slot/proxy, request, candidate,
owner, model, pool, and prepared target. CUDA remained uninitialized and no
forward or inference path executed.

Success rows returned the exact fixed `published` participant row, performed
one production-slot publication, exposed the exact candidate, owner, and
fingerprint, and committed the exact request plus four-field completion
configuration. Injected rows loaded the same complete real candidate, made one
proxy publication attempt, then returned the exact bounded error row for:

```text
RuntimeError: injected ModelRunner load-and-publish failure
```

The proxy raised before production delegation. The production slot remained
empty and both completion fields remained `None`.

Authoritative identities:

```text
source tree:
  a1bf0161eeedf3c73fb176a0f26ab2156bb3d944096db187a9c83eeb98ae5cc8
model_runner_load_and_publish_preflight.json:
  d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18
source_manifest.json:
  809fb53d0f910cd503453fba9a9f314b86ce001794d6c838a3248c0fcdcfaf33
model_runner.py:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849
extracted load-and-publish method source:
  9134c5bad8c4127714e07ffd8af56209c247a746e9f0d0ceceb60227c1358612
```

A standard-library-only independent verifier imported neither the gate nor
TinyLLMForge and passed 323 checks. It independently recomputed the 50-file
closure and source tree, parsed the exact production method AST, checked its
two globals and validation/loader/publication/completion ordering, compared
every value hash with the complete oracle, checked six unique PIDs, success
and rejection semantics, cleanup, collection, memory, CUDA false, and exact
local/remote inventories:

```text
remote source files: 50
remote root prerequisites: 2
remote root results: 2
local result artifacts: 2
local/remote result SHA256: exact equality
```

Focused tests passed 6/6, Python 3.9 compilation passed, local and remote CLI
validation passed, and 20 regression groups passed with zero failures. Static
AST audit found exactly one
`build_qwen35_authorized_checkpoint_candidate_loader` call site, one
production publication-slot constructor, and one extracted production-method
invocation. It found zero ModelRunner imports/construction, direct streamed
loader calls, `target.take()` calls, binding, Engine, scheduler, forward,
inference, or CUDA operation calls. CUDA observation was limited to two
read-only `torch.cuda.is_initialized()` calls. The real worker hard rejection
remains exact, schema-v2 canonical `NO_GO` remains unchanged,
`git diff --check` passed, and staged files remain zero.

Exact claim boundary:

```text
exact production ModelRunner request-validation -> authorized-loader ->
production-slot publication -> completion-state method at TP=1 and TP=2:
  proven
success published row, slot visibility, and completion tuple:
  proven
post-load pre-slot injected rejection with empty slot/completion:
  proven
value preservation, cleanup, and whole-scope collection:
  proven
ModelRunner construction / initialization:
  absent
candidate binding / Engine / scheduler / cross-rank runtime:
  absent
CUDA allocation or operators / forward / inference:
  absent
production latency / throughput / cache or GPU-memory savings / quality:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
