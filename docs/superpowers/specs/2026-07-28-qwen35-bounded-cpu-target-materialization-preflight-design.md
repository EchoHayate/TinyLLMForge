# Qwen3.5 Bounded CPU Target Materialization Preflight Design

## Status

Approved for inline implementation and one live source-bound run on
`sitian@10.232.195.203`.

This gate authorizes construction and explicit page commitment of an empty
rank-local CPU checkpoint target. It does not authorize reading checkpoint
payload bytes, calling a loader, assigning checkpoint tensors, consuming the
prepared target, executing forward, initializing CUDA, or installing the
candidate into Engine.

## Goal

In an independent fresh process for:

```text
TP=1 rank 0
TP=2 rank 0
TP=2 rank 1
```

construct:

```text
HybridStateTensorPool(capacity=1, device="cpu")
parameter_device="cpu"
non-executing static attention backends
```

then touch every unique registered tensor with `zero_()` under
`torch.no_grad()` and prove the physical CPU allocation, binding identity,
tied embedding, zero contents, and bounded process memory without loading
checkpoint payload.

## Exact Static Budget

The preceding meta graph was counted by unique registered tensor object:

```text
TP=1:
  unique registered tensors: 302
  unique registered bytes: 3763656128
  unique 320-binding destination bytes: 3763655360
  unbound rotary buffers: 6 * 128 = 768 bytes
  state pool bytes: 10321920

TP=2, each rank:
  unique registered tensors: 302
  unique registered bytes: 1881936480
  unique 320-binding destination bytes: 1881935712
  unbound rotary buffers: 6 * 128 = 768 bytes
  state pool bytes: 5160960
```

`embed_tokens.weight` and `lm_head.weight` are the same `Parameter` object and
must be counted once.

## Non-Authoritative Touch Probe

The exact production closure plus a temporary probe ran each row in a fresh
process and zeroed every unique registered tensor.

```text
TP=1 rank0:
  VmHWM before / Torch / metadata / pool / target / touch:
  13284 / 365584 / 496104 / 506300 / 4186824 / 4187796 KiB

TP=2 rank0:
  13112 / 365812 / 496020 / 501088 / 2343048 / 2343696 KiB

TP=2 rank1:
  13300 / 366252 / 496740 / 501936 / 2343996 / 2344692 KiB
```

Every row retained all 320 CPU bindings, exact tied embedding identity,
302 unique registered tensors, zero payload/loader/assignment/forward events,
zero values after touch, and CUDA false.

After `del` and `gc.collect()`, the process RSS remained approximately
0.9-1.8 GiB because the allocator retained pages. Therefore every rank must
run in a separate process and exit immediately after emitting its row. No
same-process rank loop is allowed.

## Frozen Memory Ceilings

Ceilings are TP-specific:

```text
TP=1:
  total process VmHWM increment <= 4718592 KiB
  post-Torch VmHWM increment <= 4194304 KiB
  post-metadata VmHWM increment <= 3932160 KiB

TP=2:
  total process VmHWM increment <= 2621440 KiB
  post-Torch VmHWM increment <= 2359296 KiB
  post-metadata VmHWM increment <= 2097152 KiB
```

The post-metadata ceiling is the main materialization guard. It admits one
rank-local model plus its exact state pool and graph bookkeeping, but rejects
a second model copy, payload staging, or TP mismatch.

## Architecture

Create:

```text
tools/qwen35_real_checkpoint_cpu_materialization_preflight.py
tools/test_qwen35_real_checkpoint_cpu_materialization_preflight.py
```

The source closure contains the same frozen 32 production files, the already
verified target-preparation gate module that supplies shared deterministic
SSH/source/memory primitives, and the CPU gate module: 34 files total.
Namespace packages avoid unrelated package initialization.

Each rank worker:

1. rejects any checkpoint path other than the approved model before imports;
2. records memory before imports and after Torch;
3. reads bounded config/index/header metadata and builds the 320-entry plan;
4. allocates the exact capacity-one CPU state pool;
5. installs a fail-closed post-metadata safetensors-open guard;
6. prepares the real target with `parameter_device="cpu"`;
7. verifies exact graph, backend, pool, registration, and binding contracts;
8. deduplicates registrations by Python object identity;
9. verifies exact unique tensor bytes for the TP size;
10. zeroes every unique registered tensor under `torch.no_grad()`;
11. verifies every unique tensor is zero and all binding destinations retain
    their registered object identity;
12. records memory after target construction and after touch;
13. emits one validated JSON row and exits.

The worker does not rely on in-process release. Process exit is the release
boundary.

The orchestrator stages/hashes sources, launches one worker process per row,
validates unique PIDs, invokes a separate finalizer, and atomically publishes:

```text
cpu_materialization_preflight.json
source_manifest.json
```

A partial or failed run publishes no local authoritative directory. Remote
failed directories remain preserved.

## Exact Contract

Every row requires:

```text
metadata bytes read: 144024
payload bytes read: 0
plan loads/skips/payload bytes: 320/312/4548144832
pool capacity/device/components/bindings/nonzero: 1/cpu/36/0/0
layers/adapters/backends: 24/18/6
bindings/shared/linear/full/buffer/F32: 320/2/252/66/72/36
registered entries: 303
unique registered tensors: 302
unbound registrations: exactly six rotary.inv_freq F32[32] buffers
tied embedding: exact same object
all registrations/bindings: CPU
all unique tensors zero after touch: true
loader/assignment/model-forward/attention-forward calls: 0/0/0/0
CUDA before/after: false/false
```

The exact bytes are TP-specific values from the static budget section.

## Safety

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use only `sitian@10.232.195.203` and the approved remote Python.
- Use one fresh process per rank and never construct two rank models in one
  process.
- Set `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`,
  `OMP_NUM_THREADS=8`, and `MKL_NUM_THREADS=8`.
- Never open safetensors after bounded header parsing.
- Never call a loader, assignment, `target.take()`, or forward.
- Never overwrite or delete evidence.
- Keep the real worker `main()` hard-disabled.
- Do not modify Engine, publication, or schema-v2 canonical `NO_GO`.
- Do not stage, commit, or merge.
- Do not claim runtime speed, cache, GPU-memory, compression, or quality
  benefit.

## Allowed Conclusion

Passing proves one approved rank-local Qwen3.5 CPU target can be fully
materialized and physically committed within a bounded fresh process while
retaining exact pool, graph, tied-weight, and binding identities without
checkpoint payload, assignment, forward, or CUDA.

It does not prove checkpoint loading, loaded-weight correctness, transactional
assignment, candidate installation, inference correctness, runtime memory
retention, latency, throughput, cache savings, or model quality.
