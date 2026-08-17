# Qwen3.5 Real Checkpoint Load Read-Only Preflight Design

## Status

Approved for inline implementation and one live read-only run against the
fixed remote host. This gate does not authorize a checkpoint-load worker,
payload reads, CUDA initialization, model construction, or inference.

## Goal

Turn the real-checkpoint safety harness `preflight` mode from an intentional
rejection into a source-bound remote audit that answers one question:

> Is the approved host, Python environment, immutable Qwen3.5 model snapshot,
> filesystem, memory, and GPU0 occupancy state sufficiently identified for a
> later CPU-only 8-versus-16 MiB checkpoint-load comparison?

The audit may return `READY` or `INCOMPLETE`. It must not return `GO` and must
not establish any speed, memory, cache, compression, or quality improvement.

## Fixed Identity

```text
remote target:
  sitian@10.232.195.203
SSH control path:
  /tmp/ssh-sitian-10.232.195.203
remote Python:
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
model repository:
  Qwen/Qwen3.5-2B
model revision:
  15852e8c16360a2fea060d615a32b45270f8a8fc
approved acquisition manifest:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json
approved model directory:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model
future worker CUDA_VISIBLE_DEVICES:
  ""
```

The approved sidecar already binds the exact repository, revision, model
directory, file names, sizes, and full-file SHA256 values. This preflight may
read that sidecar and non-payload JSON files. It must not recompute any
`.safetensors` hash because doing so would open the payload.

## Architecture

The existing runner remains the sole local entrypoint. `preflight` performs:

1. validate the run tag and exact local owned-source set;
2. create a deterministic tar snapshot of those source files;
3. create a unique remote run directory without deleting existing paths;
4. stream and extract the source snapshot;
5. hash every staged source file remotely and require exact local equality;
6. invoke the staged runner's internal remote-audit entrypoint;
7. download only `preflight.json` and `source_manifest.json`;
8. validate the returned record with the dependency-light contract;
9. persist the same two artifacts atomically in the local experiment path.

The `run`, `download-only`, and `verify-only` modes remain intentional
fail-closed rejections.

## Payload-Zero Boundary

The preflight script may open only:

- staged Python source files for SHA256 verification;
- the approved `model_manifest.json`;
- `config.json`;
- `model.safetensors.index.json`;
- `/proc/meminfo`;
- `/proc/self/status`;
- mount/process metadata exposed by the operating system.

For every path ending in `.safetensors`, the script may call metadata-only
operations such as `stat`, `exists`, `is_file`, `resolve`, and filesystem
queries. It must not call `open`, `read`, `read_bytes`, `mmap`, safetensors
APIs, PyTorch checkpoint APIs, or a full-file hash operation on that path.

The result records:

```text
payload_open_count: 0
payload_bytes_read: 0
payload_hashes_recomputed: false
payload_identity_source: approved_model_manifest
```

Any code path unable to prove those values fails closed as `INCOMPLETE`.

## Remote Audit Record

`preflight.json` uses
`qwen35.real-checkpoint-load-safety.v1` and records:

- exact requested and observed host/user identity;
- remote Python path and Python version;
- package presence/version for `torch`, `safetensors`, and `transformers`;
- `CUDA_VISIBLE_DEVICES` and whether CUDA was initialized;
- GPU0 name, UUID, driver, and compute-process rows;
- approved repository, revision, manifest path, and model directory;
- approved manifest SHA256;
- config and index SHA256 values;
- a deterministic composite identity over the approved sidecar identities;
- every shard's name, expected size/SHA256, observed size, inode, device,
  mode, and resolved path;
- run-root and model-root filesystem device/type/mount/source;
- free run-root bytes and the frozen required artifact allowance;
- `/proc/meminfo`, `/proc/self/status`, and required telemetry-field presence;
- staged source-tree SHA256 and per-file local/remote SHA256 values;
- all checks, failed checks, status, and failure reasons.

The composite config/index/header field is derived from the approved sidecar:
it binds config SHA256, index SHA256, and each shard's previously verified
full-file SHA256/size. It is identity evidence, not a new header or payload
read.

## Classification

`READY` requires all of:

- observed user is `sitian`;
- observed host identity resolves to the approved target;
- exact remote Python path is used;
- required packages are installed;
- approved manifest repository/revision/path identities match;
- config and index hashes match the approved sidecar;
- every declared shard exists at the approved resolved path and its stat size
  matches the sidecar;
- no undeclared `.safetensors` file exists in the approved model directory;
- `/proc` telemetry fields are available;
- run-root free space meets the frozen artifact allowance;
- staged source hashes match exactly;
- `CUDA_VISIBLE_DEVICES` is empty and CUDA remains uninitialized;
- GPU0 has no compute process;
- payload counters remain exactly zero.

Any failed or missing check produces `INCOMPLETE`, with explicit reasons.
GPU0 occupancy is observational only: the preflight neither kills processes
nor waits for the GPU to become idle.

## Safety and Non-Destructive Rules

The implementation must not contain or execute:

```text
rm -rf
pkill
killall
git reset
git clean
rsync
snapshot_download
HfApi
torch.load
safetensors.safe_open
```

It must not access the network from the remote Python audit. SSH is the only
network action. Existing run directories are never overwritten.

## Testing

Dependency-light tests must prove:

- source tar creation and local/remote hash binding;
- strict run-tag and unique-path behavior;
- exact remote command construction;
- classification of complete and occupied/mismatched fixtures;
- generated remote script contains no payload-open operation;
- `preflight` uses injected command runners in unit tests;
- `run`, `download-only`, and `verify-only` remain rejected;
- local artifact persistence is atomic;
- the live artifact passes schema validation even when its status is
  `INCOMPLETE`.

The live run must additionally verify that only the two authorized artifacts
were downloaded and that both payload counters are zero.

## Allowed Conclusion

After a successful live invocation, the only allowed conclusion is:

> A source-bound read-only preflight ran on the approved host and recorded the
> current environment/model/resource state without opening a safetensors
> payload.

`READY` only authorizes designing the worker gate. It does not authorize or
prove real loading, assignment correctness, inference correctness, or a
performance improvement.
