# Qwen3.5 Private Candidate Publication-Slot Transaction Gate Design

## Objective

Execute the existing one-shot
`Qwen35HybridModelOwnerPublicationSlot.publish(...)` boundary on a complete
approved real Qwen3.5 checkpoint candidate at TP=1 and TP=2, prove exact
private publication visibility, and prove fail-closed whole-scope discard
after both success and an injected post-publication failure.

This gate begins after the completed private candidate ownership-transfer gate:

```text
qwen35-private-ownership-20260728-090000
```

It does not install or bind the published candidate into ModelRunner, Engine,
scheduler, CUDA, forward, or inference.

## Immutable Prerequisites

Use two exact local artifacts:

```text
complete-checkpoint oracle:
  qwen35-complete-checkpoint-20260728-065128
  complete_checkpoint_transaction_preflight.json
  SHA256:
    7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176

private ownership gate:
  qwen35-private-ownership-20260728-090000
  private_candidate_ownership_preflight.json
  SHA256:
    977a20a1986ade81e2b94063287cd15e6ece2adc3c818f3e0d9589f75b1adac4
```

The ownership artifact must retain:

```text
schema:
  qwen35.real-checkpoint-private-ownership.v1
status:
  PASS
source tree:
  91f9225a6ee214049002dc12bc7a669cdfa6a0d847b03e0cc107834f96f561a0
rows:
  six fixed success/failure rows with six unique PIDs
success hash evidence:
  320 binding hashes, 26 phase hashes, one aggregate hash
failure hash evidence:
  exact first-source binding indices and hashes
```

The publication preflight must not weaken, reinterpret, or rewrite either
prerequisite.

## Chosen Boundary

Reuse the existing production slot:

```text
Qwen35HybridModelOwnerPublicationSlot
```

Each fresh worker creates one private slot, loads one fresh private candidate
through the existing authorized adapter, and calls `slot.publish(candidate)`
exactly once.

The slot remains private to one nested transaction scope. No slot reference,
candidate, owner, model, pool, target, or runtime bridge may escape that
scope. Rollback is not an in-place `clear()` operation. The only allowed
rollback is:

1. clear every selected checkpoint destination in `finally`;
2. leave the one-shot production slot API unchanged;
3. drop the entire isolated object graph;
4. run garbage collection outside the nested scope;
5. prove weak references to the slot, candidate, owner, model, pool, and
   prepared target are dead.

This matches the production slot's existing contract: it intentionally has no
`clear()` or `replace()` method.

## Rejected Alternatives

### Add `clear()` or `replace()` to the Production Slot

Rejected. Existing publication tests require a one-shot slot with no clear or
replacement API. Adding rollback mutation would weaken the ownership and
provenance boundary.

### Publish Through ModelRunner

Rejected. ModelRunner publication would combine this gate with runtime object
installation and later binding semantics. Those remain separate gates.

### Use a Synthetic Slot

Rejected. The purpose of this gate is to execute the existing production
publication boundary on a real loaded candidate.

## Worker Matrix

Use six fresh processes in fixed order:

```text
TP=1 rank0 success
TP=1 rank0 injected_post_publication_failure
TP=2 rank0 success
TP=2 rank0 injected_post_publication_failure
TP=2 rank1 success
TP=2 rank1 injected_post_publication_failure
```

Every process receives:

```text
CUDA_VISIBLE_DEVICES=
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
```

Every process constructs a fresh CPU target, pool, candidate, and publication
slot. No process or object may be reused.

## Candidate Acquisition

Use the exact request and authorization from the ownership gate:

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

The existing authorized adapter must be built once and invoked once per
worker. The preflight must not call `target.take()` or
`load_qwen35_fresh_checkpoint_candidate()` directly.

Before publication, validate:

- exact candidate type;
- exact private target owner/model/pool/binding identities;
- exact model fingerprint;
- exact streamed-loader stats;
- all 320 binding hashes;
- all 26 phase hashes;
- aggregate destination hash;
- all 24 alias groups;
- one provider call and one adapter call;
- `target._consumed: false -> true`;
- no publication yet.

## Success Transaction

The nested success transaction:

1. creates an empty exact production publication slot;
2. proves `candidate`, `owner`, and `model_fingerprint` are all `None`;
3. acquires and validates one exact private loaded candidate;
4. calls `slot.publish(candidate)` exactly once;
5. requires the returned owner to be the exact candidate owner;
6. requires slot candidate, owner, and fingerprint identity to be visible;
7. requires slot owner/model/pool/runtime-bridge graph coherence;
8. rehashes all 320 published binding views;
9. re-derives all 26 phase hashes and the aggregate hash;
10. proves publication did not mutate loaded checkpoint values;
11. records scalar/hash evidence only;
12. clears selected destinations in `finally`;
13. exits the nested scope;
14. proves the entire private publication graph is garbage-collected.

The artifact may report that publication was observed before discard. It must
not serialize or return the candidate, slot, owner, model, pool, target, or
runtime bridge.

## Injected Post-Publication Failure

The failure transaction uses the same candidate acquisition and production
slot. After `slot.publish(candidate)` returns and exact slot visibility is
recorded, raise:

```text
RuntimeError("injected private publication-slot failure")
```

The failure row must prove:

- one provider call;
- one adapter call;
- one successful slot publication call;
- exact slot candidate/owner/fingerprint visibility before injection;
- exact injected error observed;
- no object escaped the nested scope;
- selected destinations cleared in `finally`;
- non-selected rotary values unchanged;
- pool state unchanged;
- slot, candidate, owner, model, pool, and target weak references dead after
  scope exit and garbage collection;
- zero ModelRunner, Engine, scheduler, forward, inference, or CUDA execution.

No second candidate is loaded and no second publication is attempted. Existing
unit tests remain authoritative for occupied-slot replacement rejection.

## Cleanup and Scope-Discard Contract

Selected-only zero initialization remains required because production
parameter construction uses `torch.empty`.

For both modes:

- collect the 296 unique selected tensor objects;
- preserve all non-selected tensor values, including rotary buffers;
- snapshot pool identity and storage;
- clear selected tensors once in reverse unique-object order under
  `torch.no_grad()`;
- prove every selected tensor is zero;
- prove registered tensor identity is unchanged;
- prove non-selected values and pool state are unchanged;
- drop all strong references before `gc.collect()`;
- require all tracked weak references to return `None`.

Cleanup failure overrides success or injected-failure evidence and prevents
artifact publication.

## Memory Boundary

The publication slot adds only a small Python object graph to the completed
ownership transaction. Start with the same correctness ceilings:

```text
TP=1 total/post-Torch/post-metadata:
  10485760 / 10223616 / 9961472 KiB
TP=2 total/post-Torch/post-metadata:
  7340032 / 7077888 / 6815744 KiB
```

Both success and injected post-publication failure must stay within the
corresponding ceiling. Any error must report observed and allowed values. If a
live run exceeds a ceiling after successful cleanup, preserve that run,
capture exact evidence, and calibrate only to a 256-MiB boundary with at least
256 MiB headroom before using a new run tag.

These are correctness ceilings, not latency, throughput, cache, or GPU-memory
claims.

## Source Closure

Freeze the complete 45-file ownership source closure and add:

```text
tinyvllm/engine/qwen35_hybrid_model_publication.py
tools/qwen35_real_checkpoint_private_publication_slot_preflight.py
```

Total:

```text
47 unique source files
```

The ownership source closure must match its prerequisite artifact byte for
byte. The production publication module is added without modification.

## Artifacts and Orchestration

Authoritative outputs:

```text
private_publication_slot_preflight.json
source_manifest.json
```

Use:

- two exact SHA-bound prerequisite transfers;
- six fresh workers;
- one separate finalizer;
- fixed worker ordering;
- exact remote artifact round trip;
- atomic local and remote publication;
- no authoritative artifacts if any worker fails.

Remote inventory:

```text
source files: 47
root prerequisite artifacts: 2
root result artifacts: 2
```

All failed and superseded remote run directories must remain preserved.

## Independent Verification

Use a standard-library-only verifier that imports neither the gate nor
TinyLLMForge modules. It must independently verify:

- prerequisite artifact SHA256 values;
- 47 local and remote source hashes and source-tree digest;
- six fixed rows and six unique PIDs;
- exact success binding, phase, and aggregate hashes against the complete
  oracle;
- exact publication identity evidence;
- exact injected error and pre-injection visibility;
- one-shot target consumption;
- selected cleanup and non-selected/pool preservation;
- whole-scope weak-reference collection;
- exact memory deltas and ceilings;
- exact local/remote inventories;
- exact local/remote result SHA256 equality.

## Safety Audits

The preflight must contain:

- exactly one call site to
  `build_qwen35_authorized_checkpoint_candidate_loader`;
- exactly one call site to
  `Qwen35HybridModelOwnerPublicationSlot`;
- exactly one call site to `slot.publish(...)`;
- no direct call to `load_qwen35_fresh_checkpoint_candidate`;
- no direct call to `target.take()`;
- no slot `clear()` or `replace()` call;
- no ModelRunner, Engine, scheduler, or `LLMEngine.step()` call;
- no forward or inference call;
- no CUDA allocation, transfer, synchronization, or operator call;
- only read-only `torch.cuda.is_initialized()` observations;
- no production-module modification.

The production worker hard rejection remains exactly:

```text
RuntimeError: real checkpoint load worker execution is not implemented; only the local safety dry-run is authorized
```

## Tests

Focused TDD covers:

- strict two-prerequisite validation;
- exact 47-file source closure;
- empty slot state;
- exact successful publication identity and hash preservation;
- injected post-publication failure;
- selected cleanup and non-selected/pool preservation;
- whole-scope weak-reference collection;
- exact row schemas and memory diagnostics;
- six-process orchestration and partial-failure non-publication;
- local Python 3.9 orchestration compatibility;
- CLI `run`, `internal-rank-worker`, `internal-finalize`, and `validate`.

Regression includes ownership transfer, hybrid model publication, complete
transaction, loader-core, target factory, authorized adapter, streamed/tiled
loaders, metadata, reader, assignment, worker request, authorization, safety,
and unchanged ModelRunner/Engine publication-binding tests.

## Authoritative Evidence

The final source-bound run is:

```text
qwen35-private-publication-20260728-093000
```

Fresh worker evidence:

```text
TP=1 rank0 success:
  PID 1087135
  total/post-Torch/post-metadata VmHWM increment:
    7855896/7513068/7381168 KiB
  aggregate destination SHA256:
    4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963

TP=1 rank0 injected post-publication failure:
  PID 1090773
  total/post-Torch/post-metadata VmHWM increment:
    7851444/7509764/7377824 KiB
  aggregate destination SHA256:
    4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963

TP=2 rank0 success:
  PID 1094885
  total/post-Torch/post-metadata VmHWM increment:
    4562828/4219644/4087500 KiB
  aggregate destination SHA256:
    4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041

TP=2 rank0 injected post-publication failure:
  PID 1097941
  total/post-Torch/post-metadata VmHWM increment:
    4561188/4218840/4086664 KiB
  aggregate destination SHA256:
    4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041

TP=2 rank1 success:
  PID 1100492
  total/post-Torch/post-metadata VmHWM increment:
    4562008/4219328/4087292 KiB
  aggregate destination SHA256:
    5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca

TP=2 rank1 injected post-publication failure:
  PID 1102784
  total/post-Torch/post-metadata VmHWM increment:
    4563152/4220012/4088048 KiB
  aggregate destination SHA256:
    5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca
```

Every row loaded one complete real candidate through one provider call and one
authorized-adapter call, consumed its target once, created one exact production
publication slot, and called `slot.publish(candidate)` once. The returned
owner, slot candidate, slot owner, and retained model fingerprint matched the
exact private candidate.

All six rows independently matched the complete oracle's 320 binding hashes,
26 phase hashes, aggregate hash, and 24 alias groups after publication. Their
streamed-loader stats were exactly:

```text
assigned_bindings/source_tensors/shards:
  320/320/1
loaded_bytes/peak_source_bytes:
  3763655360/1017118720
```

The three injected rows observed:

```text
RuntimeError: injected private publication-slot failure
```

only after exact slot visibility was established. Success and injected rows
then cleared all 296 unique selected destinations, preserved tensor identity,
non-selected rotary values, and pool state, dropped every strong private
reference, and proved weak-reference collection of:

```text
slot
candidate
owner
model
pool
prepared target
```

No candidate was installed or bound into runtime state.

Authoritative identities:

```text
source tree:
  20c87258ff71449ebb8bf15af6ba77153804c16ab88a5fb11917a4597be51440
private_publication_slot_preflight.json:
  f208a799eca053e03a35aa4bfcbe66dfe6e5875b3e7b78390ded345a7c7c12b6
source_manifest.json:
  f505fc69f92e67ed54e8eb03bb85026da42d1a86644cfa53e48828dcca398569
production publication module:
  4ab2f928a3bbeeb632ca4180dcd496d56ac7716ac90d2a6adeb861f9c65d5b84
```

A standard-library-only independent verifier passed 401 checks. It imported
neither the gate nor TinyLLMForge modules and compared every emitted binding,
phase, and aggregate hash directly with the complete oracle and the completed
ownership prerequisite. It also verified fixed worker ordering, six unique
PIDs, exact publication/cleanup/collection evidence, memory deltas and
ceilings, source inheritance, and exact inventories:

```text
remote source files: 47
remote root prerequisites: 2
remote root results: 2
local results: 2
local/remote result SHA256: exact equality
```

The 47-file source closure inherits all 45 ownership-gate source files
byte-for-byte and adds only the unchanged production publication module and
the publication preflight. Local and remote CLI validation passed, local
Python 3.9 compilation passed, and 18 regression groups passed with zero
failures.

Static AST audit found exactly one authorized-adapter-builder call, one
production publication-slot constructor, and one `slot.publish(...)` call.
It found zero direct streamed-loader calls, zero direct `target.take()` calls,
zero slot `clear()`/`replace()` calls, only two read-only
`torch.cuda.is_initialized()` observations, and no ModelRunner, Engine,
scheduler, step, forward, inference, or CUDA operation. The production worker
hard rejection remains exact. `git diff --check` passed and staged files
remain zero.

No failed or superseded private-publication run exists. The sole preserved
remote run is the authoritative `qwen35-private-publication-20260728-093000`.

## Allowed Conclusion

Passing proves that one complete real-checkpoint candidate loaded through the
existing authorized adapter can be published exactly once into the existing
private production publication slot at TP=1 and TP=2, that the exact candidate
owner and provenance are visible without value mutation, and that both normal
completion and a real post-publication injected failure fail closed by clearing
private tensors and discarding the entire isolated publication object graph.

It does not prove ModelRunner or Engine installation, cross-rank publication,
runtime binding, scheduler integration, CUDA, forward/inference correctness,
production latency, throughput, cache savings, GPU-memory savings,
compression, or model quality. Schema-v2 canonical `NO_GO` remains unchanged.

The next safe boundary after this gate is a dependency-light
ModelRunner-local publication method preflight using the already existing
method surface, still without Engine, CUDA, forward, or inference.
