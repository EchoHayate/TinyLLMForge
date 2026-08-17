# Qwen3.5 Meta Target-Preparation Memory Preflight Design

## Status

Approved for inline implementation and one live source-bound run on
`sitian@10.232.195.203`.

This gate authorizes only bounded checkpoint metadata reads, exact CPU hybrid
state-pool allocation, and construction of the real Qwen3.5 parameter graph on
the `meta` device. It does not authorize a checkpoint loader call, tensor
payload read, tensor assignment, model forward, CUDA initialization, Engine
integration, or publication.

## Goal

For each required rank in an independent fresh Python process:

```text
TP=1 rank 0
TP=2 rank 0
TP=2 rank 1
```

prepare the exact real-checkpoint candidate target with:

```text
HybridStateTensorPool(capacity=1, device="cpu")
parameter_device="meta"
non-executing static attention backends
```

and prove:

- the approved bounded metadata produces the exact 320-entry tensor plan;
- the pool has the exact layout, physical bytes, and no active bindings;
- the graph has 24 layers, 18 linear-state adapters, and 6 full-attention
  backend constructions;
- all 320 checkpoint binding destinations are `meta`;
- no unexpected registered CPU parameter or buffer exists;
- no payload, loader, assignment, or forward path executes;
- CUDA remains uninitialized;
- every process remains inside frozen host-memory ceilings.

## Non-Authoritative Probe

The exact 32-file production closure plus a temporary probe script was staged
outside the evidence roots. Each TP/rank case ran in a separate process.

Observed rows:

```text
TP=1 rank0:
  pool logical bytes: 10321920
  VmHWM before / after Torch / metadata / pool / target:
  13324 / 364668 / 494688 / 504940 / 509160 KiB

TP=2 rank0:
  pool logical bytes: 5160960
  VmHWM before / after Torch / metadata / pool / target:
  13324 / 364716 / 494716 / 500236 / 504480 KiB

TP=2 rank1:
  pool logical bytes: 5160960
  VmHWM before / after Torch / metadata / pool / target:
  13444 / 365032 / 495452 / 501100 / 505184 KiB
```

Every probe observed:

```text
metadata bytes read: 144024
payload bytes read: 0
plan loads/skips: 320/312
layers/adapters/backends: 24/18/6
binding count/buffer bindings: 320/72
unexpected non-meta registrations: []
loader events: []
forward events: []
CUDA initialized before/after: false/false
```

The probe is not authoritative evidence and is not published under
`experiments/`.

## Frozen Memory Ceilings

Each fresh process records six points:

```text
before production imports
after Torch import
after bounded metadata and tensor-plan construction
after exact CPU pool allocation
after meta target preparation
after immediate target/pool release plus gc.collect()
```

The formal ceilings are:

```text
total process VmHWM increment:
  <= 524288 KiB
post-Torch target-preparation VmHWM increment:
  <= 196608 KiB
post-metadata pool-plus-meta-target VmHWM increment:
  <= 32768 KiB
```

The total ceiling covers Python and Torch. The post-Torch ceiling covers
bounded metadata parsing, tensor-plan construction, the exact CPU pool, and
the meta graph. The post-metadata ceiling is the sensitive regression guard:
it allows the expected 10,321,920-byte TP=1 pool plus graph bookkeeping but
rejects accidental real CPU model-parameter allocation.

These are safety ceilings, not performance targets.

## Architecture

Create:

```text
tools/qwen35_real_checkpoint_target_preparation_preflight.py
tools/test_qwen35_real_checkpoint_target_preparation_preflight.py
```

The preflight stages the frozen 32-file production closure plus its own gate
module. It uses namespace packages so `tinyvllm/__init__.py` and unrelated
Engine paths do not execute.

The local orchestrator:

1. validates a unique run tag and absent local/remote destinations;
2. creates a deterministic source tar with normalized metadata;
3. stages exactly 33 source files and verifies every SHA256 remotely;
4. launches one remote worker process for each TP/rank row;
5. validates every row before retaining it;
6. invokes a separate finalizer that atomically writes the aggregate artifact;
7. round-trips the aggregate and source manifest;
8. atomically publishes exactly two local JSON files.

The rank worker:

1. rejects any checkpoint path other than the approved real model before
   production imports or reads;
2. imports Torch and records the initial CUDA/memory state;
3. reads only config, index, the 8-byte safetensors prefix, and JSON header;
4. builds the exact tensor plan;
5. allocates the exact capacity-one CPU hybrid-state pool;
6. replaces post-metadata safetensors opens with a fail-closed guard;
7. constructs six static `nn.Module` attention backends whose `forward`
   raises;
8. calls `prepare_qwen35_checkpoint_candidate_target(...,
   parameter_device="meta")`;
9. verifies the graph, registrations, pool identity, pool storage, and all 320
   bindings without calling `target.take()`;
10. drops all target/pool references, runs `gc.collect()`, records final
    memory/CUDA state, validates the row, and prints one JSON object.

No rank worker writes an authoritative artifact. This prevents a partial
multi-rank run from being mistaken for PASS.

## Exact Graph and Binding Contract

Every row requires:

```text
layer count: 24
linear-attention adapter count: 18
full-attention backend count: 6
binding count: 320
shared/embed/final bindings: 2
linear-layer bindings: 252
full-attention bindings: 66
buffer bindings: 72
F32 binding destinations: 36
```

Backend calls must be exactly:

```text
layer indices: 3, 7, 11, 15, 19, 23
TP=1 local query/KV heads: 8/2
TP=2 local query/KV heads: 4/1
head dimension: 256
```

Every binding destination must be the same object registered under its model
destination name and must reside on `meta`.

All registered parameters must reside on `meta`. All persistent checkpoint
buffers must reside on `meta`. The rotary `inv_freq` buffer is allowed only if
its exact name, shape, dtype, persistence, and device match the current
factory behavior; the observed real graph currently creates it on `meta`, so
the authoritative expected non-meta registration list is empty.

## Exact Pool Contract

The supplied object must be an exact `HybridStateTensorPool` with:

```text
capacity: 1
device: cpu
component count: 36
active bindings: 0
TP=1 logical/physical bytes: 10321920
TP=2 logical/physical bytes: 5160960
```

Every pool tensor must keep the same object identity, storage pointer, storage
offset, shape, dtype, device, and version across target preparation. The pool
must contain no non-zero value after construction.

## Execution Guards

The worker must report:

```text
payload_bytes_read: 0
payload_hashes_recomputed: false
loader_call_count: 0
assignment_call_count: 0
model_forward_count: 0
attention_forward_count: 0
pool_create_count: 1
backend_create_count: 6
```

Static source scans and focused tests additionally reject calls to:

- `build_qwen35_real_checkpoint_rank_loader*`;
- `Qwen35ManifestBoundCheckpointCandidateLoader.__call__`;
- checkpoint streaming or assignment functions;
- `target.take()`;
- model `run_step` or module forward;
- Engine, ModelRunner, scheduler, publication, or restore APIs.

## Artifact Contract

Schema:

```text
qwen35.real-checkpoint-target-preparation-preflight.v1
```

An authoritative PASS aggregate contains:

- exact remote target, Python, checkpoint, retained manifest, config, index,
  shard, and composite identities;
- exact source-file hashes and source-tree SHA256;
- exactly three independently produced rank rows;
- a unique process ID per row;
- exact metadata, tensor-plan, graph, binding, pool, execution, CUDA, and
  memory fields;
- `fresh_process_per_rank: true`;
- `payload_identity_source: retained_approved_manifest`.

The published local directory contains exactly:

```text
target_preparation_preflight.json
source_manifest.json
```

If staging, any rank process, validation, finalization, round trip, or local
publication fails, no local authoritative directory is created. Failed remote
run directories remain preserved.

## Safety

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use only `sitian@10.232.195.203` and the approved remote Python.
- Use `CUDA_VISIBLE_DEVICES=""`, `PYTHONDONTWRITEBYTECODE=1`, and `python -B`.
- Never overwrite or delete local or remote evidence.
- Do not invoke `tools/qwen35_real_checkpoint_load_worker.py::main()`.
- Do not call a loader, assignment, target consumption, or model forward.
- Do not read or hash tensor payload bytes.
- Do not allocate real CPU model parameters.
- Do not modify Engine, publication, or schema-v2 canonical `NO_GO`.
- Do not stage, commit, or merge.
- Do not claim production speed, cache, GPU-memory, compression, or quality
  benefit.

## Allowed Conclusion

Passing proves that the approved real checkpoint metadata can prepare every
required rank's exact capacity-one CPU state pool, real 24-layer model graph
on `meta`, and all 320 binding destinations in fresh bounded-memory processes
without payload loading, assignment, forward execution, or CUDA.

It does not prove real tensor loading, transactional assignment, CPU/GPU model
materialization, inference correctness, runtime cache restoration, production
memory savings, latency, throughput, compression ratio, or model quality.
