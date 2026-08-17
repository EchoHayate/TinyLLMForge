# Qwen3.5 Real Checkpoint Worker Loader Construction Design

## Objective

Implement the authorized construction-only portion of the real checkpoint
worker while preserving the existing hard execution rejection.

The worker receives already parsed and verified metadata plus dependency-
injected runtime providers. It constructs a manifest-bound rank loader but
does not invoke it or open any file.

## Interface

Extend:

```text
tools/qwen35_real_checkpoint_load_worker.py
```

with:

```python
def build_qwen35_real_checkpoint_rank_loader(
    hf_config,
    index_payload,
    shard_headers,
    *,
    checkpoint_dir,
    model_manifest_sha256,
    config_sha256,
    index_sha256,
    config_index_header_sha256,
    tensor_parallel_size,
    tensor_parallel_rank,
    create_pool,
    build_attention_backend,
    authorization_sha256,
) -> Qwen35ManifestBoundCheckpointCandidateLoader:
    ...
```

The function:

1. builds the exact `Qwen35CheckpointTensorPlan` from the supplied metadata;
2. constructs `Qwen35CheckpointManifestIdentity`;
3. constructs `Qwen35RankCheckpointLoaderConfiguration`;
4. returns `configuration.build_loader()`.

## Input Boundary

`hf_config`, `index_payload`, and `shard_headers` must already be parsed by a
future authorized execution path. The function delegates their full topology,
coverage, shape, dtype, offsets, and shard-set validation to
`build_qwen35_checkpoint_tensor_plan(...)`.

The function performs no JSON parsing, path reads, safetensors open, hashing,
CUDA initialization, process creation, network access, or publication.

## Execution Boundary

`main()` remains:

```text
raise RuntimeError("real checkpoint load worker execution is not implemented...")
```

No CLI arguments are added. Running the worker remains impossible in this
gate. The new function is import-only and dependency injected.

## Failure Semantics

Invalid metadata, manifest identity, TP context, provider, or authorization
raises before returning a loader.

`create_pool` and `build_attention_backend` are retained but not called during
construction. The returned loader creates rank state/model only when a future
authorized caller invokes it.

## Tests

Dependency-light tests prove:

- exact tensor-plan builder inputs and one invocation;
- exact manifest/rank/provider forwarding;
- no pool/backend call during construction;
- exact returned loader passthrough;
- metadata/configuration failure returns no loader;
- worker source contains no file-open, payload, CUDA, SSH, subprocess, or
  Engine wiring;
- `main()` still raises the existing execution-not-implemented error.

Existing authorization and safety-gate tests remain mandatory because the
worker is in their source-bound owned set.

## Claim Boundary

Passing proves worker-side construction of a validated loader graph from
already verified metadata.

It does not read metadata from disk, verify file hashes, open a checkpoint
payload, execute loading, initialize CUDA, run inference, publish a candidate,
or establish speed/cache/memory benefit.
