# Qwen3.5 Private Candidate Ownership Transfer Gate Design

## Objective

Execute the existing authorized prepared-target loader adapter against the
complete approved real Qwen3.5 checkpoint at TP=1 and TP=2, prove exactly one
`Qwen35PreparedCheckpointCandidateTarget.take()` ownership transfer per
attempt, and deterministically clear and discard every private candidate after
both success and injected post-transfer failure.

This is a CPU-only, source-bound preflight. It must not install or publish the
loaded candidate, call ModelRunner or Engine, allocate or execute CUDA work,
run model forward, or perform inference.

## Prerequisites

The gate consumes two immutable successful artifacts:

```text
complete checkpoint transaction:
  experiments/qwen35_hybrid_state/
  qwen35-complete-checkpoint-20260728-065128/
  complete_checkpoint_transaction_preflight.json
  SHA256:
    7ed84bd1e4e894e9ea990ff2dc631db849f805f4468a9b266bd308d690acb176

production tiled loader-core transaction:
  experiments/qwen35_hybrid_state/
  qwen35-tiled-loader-core-20260728-075700/
  tiled_loader_core_preflight.json
  SHA256:
    58df3dfa9fec11d1fd079c9473766413232bd3f928f537ac87e047e13ef65aae
```

The complete-checkpoint artifact is the immutable binding/phase/aggregate
hash oracle. The loader-core artifact proves the exact approved source tree,
checkpoint identity, target construction, production tiled load path, and
private cleanup at TP=1 and TP=2 before this gate authorizes ownership
transfer.

Both artifacts must pass exact schema, SHA256, source-tree, row, PID, binding,
phase, alias, loader-stat, cleanup, and CUDA validation before any checkpoint
payload access.

## Chosen Approach

Use the existing production composition:

```text
Qwen35CheckpointCandidateLoadRequest
  -> build_qwen35_authorized_checkpoint_candidate_loader(...)
  -> fresh CPU Qwen35PreparedCheckpointCandidateTarget provider
  -> target.take()
  -> load_qwen35_fresh_checkpoint_candidate(...)
  -> Qwen35LoadedCheckpointCandidate
```

The preflight does not add a second ownership API and does not call the
streamed loader directly. This preserves the production authorization,
request validation, exact target-type validation, CPU check, one-shot target,
streamed assignment, and loaded-candidate construction boundaries.

The adapter and all production modules remain unchanged. The gate adds only a
new tool, tests, source-bound orchestration, and result artifacts.

## Alternatives Rejected

### Call `target.take()` directly in the preflight

Rejected because it would prove only the target primitive, not the authorized
adapter composition that production bootstrap code is expected to install.

### Reuse the tiled loader-core wrapper and call `take()` afterward

Rejected because that bypasses the existing authorized adapter and returns a
different tiled candidate type. It would not prove the current
`Qwen35LoadedCheckpointCandidate` ownership boundary.

### Install the adapter into ModelRunner

Rejected for this gate. ModelRunner installation would combine ownership
transfer with publication-slot mutation and runtime state. The current gate
must keep the candidate private and discardable.

### Execute the real worker

Rejected. `tools/qwen35_real_checkpoint_load_worker.py::main()` must retain
its exact hard rejection. This preflight uses its own source-bound rank worker.

## Frozen Request

Every attempt uses the exact request:

```text
checkpoint_dir:
  /data00/home/sitian/sitian-workspace01/tllm/
  qwen35-hybrid-state-runs/
  qwen35-2b-hybrid-acquire-20260723-222004/model
model_fingerprint:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
max_tensor_bytes:
  1017118720
authorization_sha256:
  10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4
```

`max_tensor_bytes` exactly equals the largest approved source tensor, the
BF16 embedding matrix. A lower value must fail before shard payload access.

## Worker Matrix

Use six fresh processes:

```text
TP=1 rank0 success
TP=1 rank0 injected failure
TP=2 rank0 success
TP=2 rank0 injected failure
TP=2 rank1 success
TP=2 rank1 injected failure
```

Every process receives empty `CUDA_VISIBLE_DEVICES`, fixed
`OMP_NUM_THREADS=8`, fixed `MKL_NUM_THREADS=8`, and a separate private CPU
target. No process or target may be reused between attempts.

## Target Construction

Each worker:

1. validates both prerequisite artifacts;
2. reads the approved bounded metadata;
3. builds the exact 320-entry tensor plan;
4. creates the canonical TP-specific hybrid-state layout and one CPU pool;
5. snapshots pool identity, storage, values, and bindings;
6. constructs one fresh CPU prepared target;
7. snapshots all unbound rotary buffers;
8. explicitly zero-initializes only the 296 unique selected checkpoint
   destinations before payload access.

Selected-only initialization is required because production parameter
construction uses `torch.empty`. Unbound rotary `inv_freq` buffers are
deterministic nonzero values and must never be cleared or modified.

## Success Transaction

The success attempt:

1. constructs the exact bounded request;
2. builds the existing authorized adapter with a provider that may be invoked
   exactly once;
3. invokes the adapter exactly once;
4. requires the provider to return the exact prepared target;
5. requires `target._consumed` to transition `false -> true`;
6. requires the returned value to be an exact
   `Qwen35LoadedCheckpointCandidate`;
7. requires candidate owner/model, binding plan, fingerprint, and pool
   identities to match the private target;
8. validates exact streamed-loader stats;
9. hashes all 320 loaded binding views in binding order;
10. derives and compares all 26 phase hashes and the aggregate hash with the
    immutable complete-checkpoint oracle;
11. validates all 24 alias groups and non-selected rotary preservation;
12. clears every unique selected destination in reverse object order;
13. proves all selected destinations are zero, rotary values are unchanged,
    pool state is unchanged, and the candidate was never published.

The candidate and target remain private local references only. The worker
returns scalar/hash evidence, then drops all references.

## Streamed Loader Stats

Every successful row must report:

```text
TP=1:
  assigned_bindings/source_tensors/shards: 320/320/1
  loaded_bytes/peak_source_bytes: 3763655360/1017118720
TP=2 rank0/rank1:
  assigned_bindings/source_tensors/shards: 320/320/1
  loaded_bytes/peak_source_bytes: 3763655360/1017118720
```

The streamed loader materializes every full checkpoint source before applying
rank-local assignment. All 320 binding source names are visited, so
`loaded_bytes` and `peak_source_bytes` are identical at TP=1 and TP=2 even
though TP=2 destinations retain only rank-local slices.

These values are loader accounting, not process-memory or performance claims.

## Injected Failure Transaction

The failure attempt must use the same production adapter and streamed loader.
The preflight may temporarily wrap only the streamed module's internal
per-source assignment function inside the private worker process.

The wrapper:

1. delegates the first source assignment to the original function;
2. records the first assigned source name, binding count, and exact
   destination hashes;
3. raises `RuntimeError("injected ownership-transfer assignment failure")`
   before the second source assignment.

The failure row must prove:

- the adapter and provider were each called exactly once;
- `target.take()` occurred exactly once and `_consumed == true`;
- at least one real source was read and assigned before injection;
- the first assigned binding values match the complete-checkpoint oracle;
- no `Qwen35LoadedCheckpointCandidate` escaped;
- no owner was published or installed;
- the private target was cleared in `finally`;
- all selected destinations are zero afterward;
- unbound rotary values and pool state are unchanged;
- all safetensors handles closed through normal exception unwinding;
- a repeated `target.take()` is rejected.

The assignment wrapper must be restored before worker exit, including on
unexpected failure.

## Memory Boundary

The streamed loader materializes one full source tensor at a time, including
the 1,017,118,720-byte embedding source. It intentionally uses more CPU memory
than the completed 65,536-byte tiled loader-core gate.

Initial conservative VmHWM increment ceilings:

```text
TP=1 success total/post-Torch/post-metadata:
  10485760 / 10223616 / 9961472 KiB
TP=2 success total/post-Torch/post-metadata:
  7340032 / 7077888 / 6815744 KiB
all injected-failure attempts:
  must not exceed the corresponding success ceiling
```

The validator must include observed and allowed values in any ceiling error.
If a live attempt exceeds a ceiling after completing ownership cleanup, retain
that failed run and calibrate a new 256-MiB-aligned ceiling from the observed
evidence before using a new run tag.

These are correctness ceilings, not latency, throughput, cache, or GPU-memory
claims.

## Source Closure and Artifacts

Freeze the completed loader-core 44-file source closure plus:

```text
tools/qwen35_real_checkpoint_private_candidate_ownership_preflight.py
```

Total:

```text
45 files
```

Authoritative outputs:

```text
private_candidate_ownership_preflight.json
source_manifest.json
```

The source manifest binds both prerequisite artifact SHA256 values, all 45
local/remote file hashes, and one source-tree digest.

Use six fresh rank workers, one separate finalizer, exact remote round trip,
and atomic local publication. Any worker failure prevents authoritative
artifact publication. Failed and superseded remote directories remain
preserved.

## Safety Audits

The preflight must contain:

- exactly one call site to
  `build_qwen35_authorized_checkpoint_candidate_loader`;
- no direct call to `load_qwen35_fresh_checkpoint_candidate`;
- no direct call to `target.take()`;
- no candidate publication slot, ModelRunner, Engine, scheduler, or
  `LLMEngine.step()` calls;
- no forward or inference calls;
- no CUDA allocation, transfer, synchronization, or operator calls;
- only read-only `torch.cuda.is_initialized()` observations;
- no modification of the production adapter, target factory, streamed loader,
  worker, ModelRunner, or Engine.

The production worker hard rejection must remain exactly:

```text
RuntimeError: real checkpoint load worker execution is not implemented; only the local safety dry-run is authorized
```

## Tests

Focused TDD covers:

- exact two-prerequisite validation and row selection;
- success ownership transition and exact candidate identity;
- injected post-transfer failure and no candidate escape;
- selected-only initialization and reverse unique-object cleanup;
- unbound rotary and pool preservation;
- exact success/failure row schemas and memory diagnostics;
- exact 45-file source closure;
- six fresh processes and partial-failure non-publication;
- local Python 3.9 orchestration compatibility;
- CLI `run`, `internal-rank-worker`, `internal-finalize`, and `validate`.

Regression requirements include:

- tiled loader-core and complete transaction gates;
- target factory, candidate adapter, streamed loader, tiled loader, metadata,
  reader, assignment, worker request, loader construction, authorization, and
  safety gates;
- ModelRunner authorized-loader and publication/binding tests only as
  unchanged regressions, never as live runtime execution.

## Authoritative Evidence

The final source-bound run is:

```text
qwen35-private-ownership-20260728-090000
```

The earlier `qwen35-private-ownership-20260728-084300` run passed the live
transactions, but its row schema retained only hash-verification booleans and
counts. It did not serialize the measured 320 binding hashes, 26 phase hashes,
aggregate hash, or injected-failure first-source binding hashes required for
an independent comparison. That run remains preserved and is superseded, not
deleted or reinterpreted. A focused RED test reproduced the evidence gap; the
final schema serializes the measured hashes and validates their shape before
artifact publication.

Fresh worker evidence:

```text
TP=1 rank0 success:
  PID 755827
  total/post-Torch/post-metadata VmHWM increment:
    7856712/7513656/7381480 KiB
  aggregate destination SHA256:
    4b932c940e3411a38572bca57edfd473b086ef5c46eb229b50beb5af90ed4963

TP=1 rank0 injected failure:
  PID 760128
  total/post-Torch/post-metadata VmHWM increment:
    6138368/5795644/5663000 KiB
  first-source destination SHA256:
    b222b11204158144e369ae8fca02cab9cb63b0a8cde1dd59dd4d0c60690824ed

TP=2 rank0 success:
  PID 762147
  total/post-Torch/post-metadata VmHWM increment:
    4560788/4218224/4085964 KiB
  aggregate destination SHA256:
    4b64f77724aa9114839bf0b3e78edc618ba6bc092626be9c6bb81bbb733c9041

TP=2 rank0 injected failure:
  PID 764509
  total/post-Torch/post-metadata VmHWM increment:
    3304164/2961312/2829300 KiB
  first-source destination SHA256:
    d45e558c608450961514de8174a9f9455176a60b00eb651fc6f1d571535adafe

TP=2 rank1 success:
  PID 765957
  total/post-Torch/post-metadata VmHWM increment:
    4562224/4219080/4086816 KiB
  aggregate destination SHA256:
    5bbaa720483edb8e5f1df61386c427ba733aaa411ef8b006f4d39b65a1f9c3ca

TP=2 rank1 injected failure:
  PID 769291
  total/post-Torch/post-metadata VmHWM increment:
    3306240/2962728/2830568 KiB
  first-source destination SHA256:
    4c7d498a1a57731508852e1e7e23f52a3e6f708d399bcb7e6b80d7d784fa604d
```

Every success row serialized and independently matched all 320 binding
destination hashes, all 26 phase hashes, and the aggregate hash from the
complete-checkpoint oracle. Every injected-failure row serialized and matched
the exact first assigned binding index and destination hash after one real
source assignment and before the injected second-source failure. All rows
proved one provider call, one adapter call, `target._consumed: false -> true`,
selected cleanup to zero, unchanged non-selected rotary values and pool state,
no candidate publication, zero forward calls, and CUDA false.

The exact streamed-loader stats for all three success TP rows were:

```text
assigned_bindings/source_tensors/shards:
  320/320/1
loaded_bytes/peak_source_bytes:
  3763655360/1017118720
```

Authoritative hashes:

```text
source tree:
  91f9225a6ee214049002dc12bc7a669cdfa6a0d847b03e0cc107834f96f561a0
private_candidate_ownership_preflight.json:
  977a20a1986ade81e2b94063287cd15e6ece2adc3c818f3e0d9589f75b1adac4
source_manifest.json:
  bfb2201b59f5a206d0b20f3d6ce65485f94c87c2dfca0136bdd2973b19242820
```

A standard-library-only verifier that imported neither the gate nor
TinyLLMForge passed 330 checks. It compared every emitted success and failure
hash directly with the immutable complete oracle, recomputed memory deltas and
source-tree identity, required six unique PIDs in fixed order, and verified
the exact local/remote inventory:

```text
remote source files: 45
remote root prerequisites: 2
remote root results: 2
local results: 2
local/remote result SHA256: exact equality
```

The source closure inherits all 44 loader-core files byte-for-byte and adds
only this ownership preflight. Local and remote CLI validation passed, local
Python 3.9 compilation passed, and 17 regression groups passed with zero
failures. Static AST audit found exactly one authorized-adapter-builder call,
zero direct streamed-loader calls, zero direct `target.take()` calls, only two
read-only `torch.cuda.is_initialized()` observations, and no publication,
ModelRunner, Engine, scheduler, step, forward, inference, or CUDA operation.
The production worker hard rejection is unchanged. `git diff --check` passed
and staged files remain zero.

Preserved failed or superseded remote runs:

```text
qwen35-private-ownership-20260728-082000
qwen35-private-ownership-20260728-082500
qwen35-private-ownership-20260728-083300
qwen35-private-ownership-20260728-084300
```

## Allowed Conclusion

Passing proves that the existing authorized prepared-target adapter can
transfer one fresh private CPU candidate into the production streamed loader,
load the complete approved real checkpoint at TP=1 and TP=2, return the exact
loaded-candidate ownership object on success, and fail closed after a real
post-transfer partial assignment while the preflight deterministically clears
and discards all private state.

It does not prove candidate installation, publication, ModelRunner or Engine
integration, CUDA, forward/inference correctness, production latency,
throughput, cache savings, GPU-memory savings, compression, or model quality.
Schema-v2 canonical `NO_GO` remains unchanged.

The next safe boundary after this gate is a separate private
candidate-publication-slot transaction. It may publish the already loaded
candidate into an isolated one-shot slot and roll back only by discarding the
entire isolated slot owner; it must still prohibit ModelRunner, Engine, CUDA,
forward, and inference.
