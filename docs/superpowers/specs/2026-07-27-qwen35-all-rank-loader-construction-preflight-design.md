# Qwen3.5 All-Rank Loader Construction Preflight Design

## Status

Approved for inline implementation and one live construction-only run on
`sitian@10.232.195.203`.

This gate does not authorize calling a loader, creating a state pool,
constructing a model, reading tensor payload bytes, initializing CUDA, or
running inference.

## Goal

Starting from the approved real checkpoint's bounded metadata bundle, construct
the manifest-bound rank loader for:

```text
TP=1 rank 0
TP=2 rank 0
TP=2 rank 1
```

and prove that construction:

- produces exact loader/configuration types for every rank;
- retains the approved manifest and exact tensor plan;
- calls neither the pool provider nor attention-backend provider;
- reads zero tensor-payload bytes;
- leaves CUDA uninitialized;
- stays within a frozen host peak-RSS increment.

## Pre-Probe Evidence

A non-authoritative import-only probe using the exact 32-file production
closure observed:

```text
provider events: []
CUDA initialized before: false
CUDA initialized after: false
VmHWM before: 364944 kB
VmHWM after: 497068 kB
VmHWM increment: 132124 kB
```

The formal gate freezes:

```text
maximum total process VmHWM increment:
  524288 kB
maximum post-Torch-import construction VmHWM increment:
  262144 kB
```

The total ceiling includes the remote Python process importing PyTorch. The
construction ceiling starts from a second measurement immediately after the
PyTorch import and isolates metadata parsing plus loader-object construction.
Both are safety ceilings, not performance targets.

## Architecture

Create:

```text
tools/qwen35_real_checkpoint_loader_construction_preflight.py
tools/test_qwen35_real_checkpoint_loader_construction_preflight.py
```

The gate stages an exact source closure containing the real worker, bounded
metadata reader, loader configuration and all transitively imported
TinyLLMForge modules. It creates namespace packages at runtime to avoid
executing `tinyvllm/__init__.py`.

The worker:

1. validates the approved checkpoint path before any import or read;
2. records `/proc/self/status` before production dependency imports;
3. imports Torch, records `torch.cuda.is_initialized()`, and captures the
   second memory measurement;
4. directly loads the bounded metadata reader and real worker module;
5. reads config/index/header metadata with zero payload bytes;
6. constructs three rank loaders with providers that raise if called;
7. verifies exact loader/configuration/manifest/tensor-plan values;
8. records `/proc/self/status` and CUDA state after construction;
9. validates the frozen record;
10. atomically writes one result JSON.

The local orchestrator stages sources, verifies hashes, invokes the worker with
`CUDA_VISIBLE_DEVICES=""`, round-trips artifacts, and atomically publishes:

```text
loader_construction_preflight.json
source_manifest.json
```

## Exact Source Closure

The source inventory is generated locally from:

```text
tools/qwen35_real_checkpoint_load_worker.py
tinyvllm/models/qwen35_checkpoint_metadata.py
```

by recursively following only local `tinyvllm.*` imports, then adding the gate
module itself. The resolved production closure observed before implementation
contains 32 files. Tests freeze the exact sorted list and reject additions,
omissions, remote bytecode, and source-hash mismatch.

## Provider and Execution Boundary

Construction receives:

```python
def forbidden_pool():
    raise AssertionError("pool provider called")

def forbidden_backend(*args, **kwargs):
    raise AssertionError("attention backend provider called")
```

The result must record:

```text
provider_events: []
loader_call_count: 0
pool_create_count: 0
backend_create_count: 0
payload_bytes_read: 0
```

The worker must not call:

- the returned loader;
- `prepare_target`;
- `create_pool`;
- `build_attention_backend`;
- model assembly;
- checkpoint streaming/assignment;
- Engine or publication APIs.

## Record Contract

Schema:

```text
qwen35.real-checkpoint-loader-construction-preflight.v1
```

PASS requires:

- exact remote user/host/Python/model identities;
- exact source hashes and source-tree identity;
- metadata bytes `144024`, payload bytes `0`;
- exact plan counts: 320 loads, 312 skips, 4,548,144,832 payload bytes;
- exact rows `(1,0)`, `(2,0)`, `(2,1)`;
- exact loader type
  `Qwen35ManifestBoundCheckpointCandidateLoader`;
- exact configuration type
  `Qwen35RankCheckpointLoaderConfiguration`;
- exact approved manifest directory and identities;
- zero provider/loader calls;
- CUDA uninitialized before and after;
- positive VmRSS/VmHWM values;
- non-negative total VmHWM increment no greater than `524288 kB`;
- non-negative post-Torch-import construction VmHWM increment no greater than
  `262144 kB`.

## Safety

- Use only the fixed `sitian` host, Python, and SSH ControlMaster.
- Use `PYTHONDONTWRITEBYTECODE=1` and `python -B`.
- Never overwrite or delete local/remote evidence.
- Do not invoke `tools/qwen35_real_checkpoint_load_worker.py::main()`.
- Do not call any constructed loader.
- Do not read or hash tensor payload bytes.
- Do not modify Engine, ModelRunner, Scheduler, publication, or schema-v2
  canonical evidence.
- Do not claim performance, cache, memory, compression, or quality benefit.

## Allowed Conclusion

Passing proves the approved metadata can construct all required rank-local
manifest-bound loader objects within a bounded host-memory increment and
without invoking providers, CUDA, payload loading, or runtime publication.

It does not prove target/model construction, tensor assignment, full-shard
integrity, inference correctness, or production performance.
