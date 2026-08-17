# Qwen3.5 TP4 Remote Authority Configuration Design

## Objective

Create a pure-local configuration path for the real TP4 correctness campaign
without requiring the 4.5 GB model payload to exist on the local workstation.
The result must remain bound to the canonical local model manifest and to an
explicit remote model directory, while preserving the existing
`build_configuration(...)` local-execution safety behavior unchanged.

This work prepares data only. It must not use SSH, inspect GPUs, load a model,
construct an Engine, invoke subprocesses, or authorize benchmark execution.

## Existing Boundary

`build_configuration(...)` currently requires:

- a regular local repository root;
- a regular local model manifest;
- an existing local model directory;
- explicit TP4 GPU, port, cache, timeout, and fingerprint values.

That contract is correct for a configuration intended to execute against local
model weights and must not be weakened.

The cached-continuation and Engine remote-plan builders use the configuration
only to verify the local manifest, workload manifest, source inventory, and
configuration identities. They then replace `model_dir`,
`model_manifest_path`, and `workload_manifest_path` with explicit remote paths
in the staged remote configuration. They do not read local model weights.

## Considered Approaches

### 1. Add a permissive flag to `build_configuration(...)`

Rejected. A flag such as `require_local_model=False` makes the safety-critical
local builder mode-dependent and creates an easy accidental bypass for local
execution callers.

### 2. Create a second standalone builder module

Viable but not preferred. It would duplicate source-tree hashing, workload
generation, atomic publication, and executor-configuration serialization,
creating avoidable drift between local and remote authority configuration.

### 3. Add a separate function in the existing builder module

Selected. `build_remote_configuration(...)` has a distinct public contract and
shares only private deterministic publication helpers with
`build_configuration(...)`. The local builder still requires an existing model
directory. The remote builder never accepts or probes a local model directory.

## Public Interface

The existing interface remains unchanged:

```python
build_configuration(
    *,
    repo_root,
    output_dir,
    model_dir,
    model_manifest_path,
    model_fingerprint,
    gpu_indices,
    dist_port,
    master_port,
    max_cache_entries,
    max_cache_bytes,
    timeout_s,
) -> dict
```

The new interface is:

```python
build_remote_configuration(
    *,
    repo_root,
    output_dir,
    model_manifest_path,
    remote_model_dir,
    model_fingerprint,
    gpu_indices,
    dist_port,
    master_port,
    max_cache_entries,
    max_cache_bytes,
    timeout_s,
) -> dict
```

Every operational value remains required. There are no hidden defaults for
GPU indices, ports, cache limits, timeout, or model fingerprint.

## Manifest Binding

`model_manifest_path` must be a non-symlink regular local file containing one
JSON object.

The manifest must contain `remote_model_dir` as a non-empty absolute POSIX
path. The explicit `remote_model_dir` argument must also be an absolute POSIX
path and must exactly equal the manifest value after lexical POSIX
normalization. No local filesystem lookup is performed for that remote path.

The emitted executor configuration uses:

- `model_dir`: the validated explicit remote model directory;
- `model_manifest_path`: the absolute local canonical manifest path;
- `model_manifest_sha256`: the exact SHA-256 of that local manifest.

Keeping the local manifest path in the preparation configuration is required:
the child plan builders hash that file before producing a remote configuration.
They subsequently replace the manifest path with the separately supplied
absolute remote manifest path.

The remote manifest path is not inferred by this builder because it is not a
field in the canonical model manifest. It remains an explicit input to the
campaign preparation builder. Existing acquisition and preflight evidence
binds the intended path to:

```text
/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json
```

## Shared Publication

Both public builders use one private publication function after completing
their mode-specific validation. The helper:

1. validates the repository root and absent output directory;
2. computes the authority-owned source inventory and source-tree SHA;
3. writes the canonical workload manifest;
4. constructs `ExecutorConfiguration` with all explicit values;
5. writes the canonical configuration and source inventory;
6. atomically renames the temporary directory into place;
7. removes temporary output on every failure.

The helper receives a validated model directory string. It does not decide
whether the path is local or remote.

## CLI

The existing CLI remains local-mode compatible. A mutually exclusive
`--remote-model-dir` option selects the remote builder; `--model-dir` selects
the existing local builder. Exactly one must be supplied.

All other arguments remain required. `--model-manifest` continues to name the
regular local manifest used for identity binding.

## Failure Behavior

The remote builder rejects:

- an existing output directory;
- a missing repository root;
- a missing, symlinked, malformed, or non-object manifest;
- a missing, empty, relative, or non-string manifest `remote_model_dir`;
- a relative explicit remote model directory;
- any mismatch between the explicit and manifest-bound remote directory;
- invalid executor configuration fields;
- missing authority-owned source files.

Every failure leaves no published output directory.

## Testing

Tests must prove:

1. the remote builder succeeds without a local model payload;
2. the emitted configuration preserves manifest, source, workload, GPU, port,
   cache, timeout, and fingerprint identities;
3. missing or mismatched manifest remote paths fail closed with no output;
4. the existing local builder still rejects a missing local model directory;
5. CLI mode selection is unambiguous;
6. the canonical Qwen3.5 manifest can produce a real local configuration;
7. that configuration can produce and independently reopen a real `READY`
   campaign preparation bundle without SSH or GPU work.

## Real Local Artifact Parameters

The artifact generation command must pass every value explicitly:

```text
gpu_indices=(0,1,2,3)
dist_port=31001
master_port=31002
max_cache_entries=8
max_cache_bytes=1073741824
timeout_s=600.0
model_fingerprint=3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

The model fingerprint is the canonical manifest SHA used by the real
checkpoint candidate evidence, not the older test-only
`qwen35-m8-authority` label.

Fresh run tags and authorization nonces must be unique within the generated
preparation bundle. The output must live under
`experiments/qwen35_hybrid_state/` and must not modify the frozen
`qwen35-tp4-source-prep-20260729-010400` directory.

## Claim Boundary

Successful completion proves only that a manifest-bound remote authority
configuration and an independently verifiable local `READY` preparation
bundle can be produced without local weights. It does not prove correctness,
accuracy preservation, speed, cache reduction, GPU-memory reduction,
compression, or quality.
