# Qwen3.5 Real-Checkpoint ModelRunner-Local Publication Method Gate Design

## Objective

Execute the existing production
`ModelRunner.publish_qwen35_loaded_checkpoint_candidate(candidate)` method body
on one complete approved real Qwen3.5 checkpoint candidate at TP=1 and TP=2,
without importing, constructing, or initializing `ModelRunner`.

The gate proves that the production method delegates exactly once to a local
publication slot, returns the exact candidate, preserves all loaded values,
and fails atomically when the local slot rejects publication.

It begins after:

```text
qwen35-private-publication-20260728-093000
```

It does not execute ModelRunner `__init__`, owner binding, Engine transport,
scheduler, CUDA, forward, or inference.

## Immutable Prerequisites

Use:

```text
complete oracle:
  complete_checkpoint_transaction_preflight.json
  SHA256:
    7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176

private publication gate:
  private_publication_slot_preflight.json
  SHA256:
    f208a799eca053e03a35aa4bfcbe66dfe6e5875b3e7b78390ded345a7c7c12b6
  source tree:
    20c87258ff71449ebb8bf15af6ba77153804c16ab88a5fb11917a4597be51440
```

The publication artifact must retain six fixed fresh-process rows, exact
success/failure hash evidence, one production slot publication per row,
complete cleanup, whole-scope collection, and CUDA false.

## Production Method Binding

Freeze:

```text
tinyvllm/engine/model_runner.py
file SHA256:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849

ModelRunner.publish_qwen35_loaded_checkpoint_candidate
AST source SHA256:
  37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f
```

The preflight reads the frozen source file, parses the `ModelRunner` class,
extracts exactly the named method AST, removes decorators, compiles only that
function into an isolated namespace, and calls it as an unbound production
method.

The preflight must not:

- import `tinyvllm.engine.model_runner`;
- construct `ModelRunner`;
- call `ModelRunner.__init__`;
- execute any other ModelRunner method;
- rewrite the method body;
- substitute a copied handwritten implementation.

The method AST must contain exactly:

```text
self.qwen35_loaded_checkpoint_candidate_slot.publish(candidate)
return candidate
```

apart from argument and formatting syntax.

## Runner Shell

Create one minimal private shell object with exactly:

```text
qwen35_loaded_checkpoint_candidate_slot
```

No rank, model, CUDA device, attention backend, runtime bridge, Engine channel,
or scheduler reference is attached.

The shell remains inside the nested transaction scope and must be collected
with the candidate graph.

## Worker Matrix

Use six fresh CPU-only processes:

```text
TP=1 rank0 success
TP=1 rank0 injected_method_failure
TP=2 rank0 success
TP=2 rank0 injected_method_failure
TP=2 rank1 success
TP=2 rank1 injected_method_failure
```

Every process receives empty `CUDA_VISIBLE_DEVICES` and fixed
`OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`.

## Candidate Acquisition

Reuse the exact approved request and existing authorized adapter:

```text
checkpoint_dir:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model
model_fingerprint:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
max_tensor_bytes:
  1017118720
authorization_sha256:
  10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4
```

The existing authorized adapter is built and invoked once per worker. The
preflight does not call `target.take()` or the streamed loader directly.

Before method execution, validate exact candidate identity, ownership graph,
fingerprint, streamed stats, 320 binding hashes, 26 phase hashes, aggregate
hash, 24 alias groups, and one-shot target consumption.

## Success Transaction

The nested success transaction:

1. creates one exact production `Qwen35HybridModelOwnerPublicationSlot`;
2. creates the minimal runner shell pointing to that slot;
3. calls the extracted production ModelRunner publication method exactly once;
4. requires the return value to be the exact candidate;
5. requires the slot candidate, owner, and fingerprint to be exact;
6. requires one underlying `slot.publish(...)` call;
7. rehashes all loaded values after method return;
8. clears selected tensors in `finally`;
9. drops runner shell, slot, candidate, owner, model, pool, and target;
10. proves all tracked weak references are dead after garbage collection.

## Injected Method Failure

Use a private proxy slot whose `publish(candidate)`:

1. records one method delegation attempt;
2. verifies the exact candidate identity;
3. raises:

   ```text
   RuntimeError("injected ModelRunner local publication failure")
   ```

4. never delegates to the production slot.

The failure row must prove:

- production method call count is one;
- proxy publish call count is one;
- production slot remains empty;
- no candidate is returned from the method;
- no publication is visible;
- loaded values remain exact before cleanup;
- cleanup and whole-scope collection complete;
- no other ModelRunner or runtime path executes.

This failure tests method-level atomicity before the underlying slot mutation.
Occupied-slot replacement rejection remains covered by the existing
production slot and ModelRunner dependency-light regression tests.

## Cleanup and Memory

Use the same selected-only initialization, reverse unique-object clear,
non-selected preservation, pool preservation, tensor identity, and weakref
collection contract as the private publication-slot gate.

Use the same correctness ceilings:

```text
TP=1 total/post-Torch/post-metadata:
  10485760 / 10223616 / 9961472 KiB
TP=2 total/post-Torch/post-metadata:
  7340032 / 7077888 / 6815744 KiB
```

These remain correctness ceilings, not performance claims.

## Source Closure

Freeze the 47-file private publication closure and add:

```text
tinyvllm/engine/model_runner.py
tools/qwen35_real_checkpoint_model_runner_local_publication_preflight.py
```

Total:

```text
49 unique source files
```

The 47 prerequisite files and production publication module must match the
publication artifact byte for byte. `model_runner.py` must match both frozen
file and method-source SHA256 values.

## Artifacts and Orchestration

Outputs:

```text
model_runner_local_publication_preflight.json
source_manifest.json
```

Use two SHA-bound prerequisite transfers, six fixed fresh workers, a separate
finalizer, deterministic JSON, exact remote round trip, and atomic local and
remote publication. Any worker failure prevents authoritative artifact
publication. Preserve all failed run directories.

Exact inventory:

```text
remote source files: 49
remote root prerequisites: 2
remote root results: 2
local result artifacts: 2
```

## Independent Verification

A standard-library-only verifier imports neither the gate nor TinyLLMForge.
It verifies:

- prerequisite and result SHA256 values;
- 49-file closure and source tree;
- exact `model_runner.py` file SHA;
- independently parsed method AST source SHA and structure;
- six fixed rows and unique PIDs;
- success return/slot identity;
- failure one-attempt/no-publication evidence;
- complete oracle hashes;
- cleanup, weakref collection, memory, and CUDA false;
- exact local/remote inventories and result SHA equality.

## Safety Audits

The preflight must contain:

- exactly one authorized-adapter-builder call site;
- exactly one production publication-slot constructor call site;
- exactly one extracted production method call site;
- no import of `tinyvllm.engine.model_runner`;
- no `ModelRunner` construction or `__init__` call;
- no direct `slot.publish(...)` call in success orchestration outside the
  extracted method;
- no direct streamed-loader or `target.take()` call;
- no binding, Engine, scheduler, step, forward, inference, or CUDA operation;
- only two read-only `torch.cuda.is_initialized()` observations;
- no production-module modification.

The real worker hard rejection remains exact and schema-v2 canonical `NO_GO`
remains unchanged.

## Tests

Focused TDD covers:

- strict two-prerequisite validation;
- frozen ModelRunner file/method AST validation;
- exact method extraction;
- success return and slot visibility;
- injected proxy failure and empty production slot;
- exact 49-file closure;
- cleanup and whole-scope collection;
- row/memory schemas;
- six-process orchestration and partial-finalization rejection;
- Python 3.9 compatibility and CLI modes.

Regression includes private publication, ownership, production slot,
ModelRunner published-candidate binding, authorized loader, Engine all-rank
binding, loader-core, complete transaction, loader/factory/metadata/reader/
assignment/authorization/safety scripts.

## Allowed Conclusion

Passing proves that the exact production ModelRunner local publication method
can accept one complete real-checkpoint private candidate, delegate once to the
existing one-shot slot, return the exact candidate, and preserve value,
ownership, cleanup, and scope-discard contracts at TP=1 and TP=2. It also
proves fail-closed behavior when the local publication dependency rejects
before slot mutation.

It does not prove ModelRunner initialization, local loader installation,
`load_and_publish_qwen35_checkpoint_candidate`, candidate binding, Engine
transport, cross-rank orchestration, scheduler integration, CUDA,
forward/inference correctness, latency, throughput, cache savings, GPU-memory
savings, compression, or model quality.

The next safe gate is a dependency-light execution of
`ModelRunner.load_and_publish_qwen35_checkpoint_candidate(request)` using a
private runner shell with the already authorized real loader and production
slot, still without constructing ModelRunner or invoking Engine/CUDA/runtime.

## Authoritative Result

The gate is complete. The sole preserved source-bound run is:

```text
qwen35-model-runner-local-publication-20260728-090014
```

Six fresh CPU-only processes executed the exact extracted production
`ModelRunner.publish_qwen35_loaded_checkpoint_candidate(candidate)` method
once each:

```text
TP=1 rank0 success:
  PID 1430521
  memory 7853696/7510864/7378412 KiB
  aggregate 4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963
TP=1 rank0 injected method failure:
  PID 1434991
  memory 7855804/7513092/7381068 KiB
  aggregate 4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963
TP=2 rank0 success:
  PID 1439369
  memory 4561704/4219020/4087192 KiB
  aggregate 4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041
TP=2 rank0 injected method failure:
  PID 1443496
  memory 4560344/4218620/4085868 KiB
  aggregate 4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041
TP=2 rank1 success:
  PID 1446432
  memory 4561120/4218448/4085916 KiB
  aggregate 5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca
TP=2 rank1 injected method failure:
  PID 1449206
  memory 4561732/4219440/4086792 KiB
  aggregate 5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca
```

Every row loaded through exactly one authorized-adapter call and one target
provider call, matched the complete oracle's 320 binding hashes, 26 phase
hashes, aggregate hash, and 24 alias groups, then cleared all 296 unique
selected destinations. Tensor identity, non-selected values, and pool state
were preserved, and the runner shell, production slot, optional proxy slot,
candidate, owner, model, pool, and target were all collected.

Success rows proved one method call, one production-slot publish, exact
candidate return identity, and exact candidate/owner/fingerprint visibility.
Injected rows proved one method call and one proxy delegation followed by the
exact:

```text
RuntimeError("injected ModelRunner local publication failure")
```

before production-slot mutation; the production slot remained empty and the
method returned no candidate.

Authoritative identities:

```text
source tree:
  d3eb52326d8e9d9a744f4641877c90a41468d26f94cbe31eda5ee04fe4d2201a
model_runner_local_publication_preflight.json:
  f8f78ae574991eb3f16aed57b4275cf76a409fa553e01597f5179c41eb158b15
source_manifest.json:
  38c1ba3d666d66f664b2477ee71be5b331ae584ae39414a938c9626a947fcfb3
model_runner.py:
  0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849
extracted method source:
  37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f
```

A standard-library-only independent verifier imported neither this gate nor
TinyLLMForge and passed 335 checks. It independently recomputed the 49-file
closure and source tree, parsed and validated the production method AST,
compared every emitted value hash with the complete oracle, checked six unique
PIDs, success and rejection semantics, cleanup, collection, memory, CUDA
false, and exact local/remote inventories:

```text
remote source files: 49
remote root prerequisites: 2
remote root results: 2
local result artifacts: 2
local/remote result SHA256: exact equality
```

Focused tests passed 5/5, Python 3.9 compilation passed, local and remote CLI
validation passed, and 19 regression groups passed with zero failures.

Static AST audit found exactly one authorized-adapter-builder call site, one
production publication-slot constructor call site, and one extracted-method
invocation call site. It found zero ModelRunner imports or construction, zero
direct streamed-loader or `target.take()` calls, zero Engine/scheduler calls,
zero forward calls, and exactly two read-only
`torch.cuda.is_initialized()` observations. The real worker hard rejection
remains exact. `git diff --check` passed and staged files remain zero.

Exact claim boundary:

```text
exact production ModelRunner local publication method on a complete
real-checkpoint private candidate at TP=1 and TP=2:
  proven
one production-slot delegation and exact candidate return on success:
  proven
pre-slot injected dependency rejection and empty production slot:
  proven
value preservation, cleanup, and whole-scope collection:
  proven
ModelRunner construction or initialization:
  absent
load_and_publish_qwen35_checkpoint_candidate:
  absent
candidate binding / Engine / scheduler / cross-rank runtime:
  absent
CUDA / forward / inference:
  absent
production latency / throughput / cache or GPU-memory savings / quality:
  unmeasured
schema-v2 canonical NO_GO:
  unchanged
```
