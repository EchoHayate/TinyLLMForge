# Qwen3.5 Zero-Payload Model Manifest Restore Design

## Status

Approved for inline implementation and one live remote restore attempt. This
gate may create only the missing approved `model_manifest.json`; it may not
open a `.safetensors` payload, load a model, initialize CUDA, or start the
checkpoint-load worker.

## Goal

Restore the immutable Qwen3.5 acquisition sidecar required by the read-only
preflight, while independently proving that the surviving model directory is
consistent with the historical approved manifest using only:

- full SHA256 of non-payload files;
- parsed `config.json`;
- parsed `model.safetensors.index.json`;
- shard path/name/count/stat metadata;
- the historical local approved manifest template.

The restored file must be byte-for-byte canonical JSON equivalent to:

```text
experiments/qwen35_hybrid_state/
  qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json
```

## Fixed Paths and Identity

```text
remote target:
  sitian@10.232.195.203
remote Python:
  /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
approved manifest path:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/
  qwen35-2b-hybrid-acquire-20260723-222004/model_manifest.json
approved model directory:
  /data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/
  qwen35-2b-hybrid-acquire-20260723-222004/model
repository:
  Qwen/Qwen3.5-2B
revision:
  15852e8c16360a2fea060d615a32b45270f8a8fc
approved manifest SHA256:
  3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0
```

## Verification Contract

Before any write, the remote script must prove:

1. the target manifest is absent, or already byte-identical to the approved
   canonical bytes;
2. the model directory exists and resolves to the fixed path;
3. every non-payload file declared by the historical manifest exists with the
   exact expected size and full SHA256;
4. `config.json` parses as JSON and identifies the expected Qwen3.5 model
   family;
5. `model.safetensors.index.json` parses as JSON;
6. every index `weight_map` target is exactly the one approved shard name;
7. the observed `.safetensors` filename set is exactly one approved shard;
8. shard stat size is exactly `4548221488` bytes;
9. payload counters remain zero.

The historical shard SHA256 is copied only as preserved provenance from the
approved manifest. The restore gate does not recompute or revalidate that
full-file payload hash.

## Write Semantics

The operation is create-if-absent and conflict-reject:

- absent target + all checks pass: write a temporary file with mode `0600`,
  `fsync`, then install it without replacing an existing target;
- identical existing target: report `ALREADY_PRESENT`, make no change;
- different existing target: report `CONFLICT`, make no change;
- any failed prerequisite: report `INCOMPLETE`, make no change.

No command may delete, truncate, replace, or rename an existing manifest.

## Restore Artifact

The unique restore run directory contains:

```text
source/<five owned source files>
restore_model_manifest.json
source_manifest.json
```

`restore_model_manifest.json` records:

- status: `RESTORED`, `ALREADY_PRESENT`, `INCOMPLETE`, or `CONFLICT`;
- exact target path and model directory;
- approved manifest SHA256 and observed final SHA256;
- per-file non-payload sizes and SHA256;
- parsed config/index evidence;
- shard stat evidence;
- checks and failure reasons;
- `payload_open_count=0`;
- `payload_bytes_read=0`;
- `payload_hashes_recomputed=false`;
- whether a write occurred.

The artifact is written atomically remotely, read back through SSH, compared,
and persisted locally.

## Follow-Up Preflight

Only `RESTORED` or `ALREADY_PRESENT` permits rerunning the existing read-only
preflight. The preflight still independently rechecks the manifest, config,
index, shard stat, source identity, CUDA state, resources, and GPU0 occupancy.

`READY` is not guaranteed: GPU0 occupancy may continue to produce
`INCOMPLETE`.

## Safety Rules

Forbidden:

```text
open/read/mmap/hash of any .safetensors path
torch.load
safetensors.safe_open
snapshot_download
HfApi
rm
unlink
replace existing manifest
worker launch
CUDA initialization
```

## Allowed Conclusion

If successful:

> The missing acquisition manifest was restored from historical approved
> provenance after all currently verifiable non-payload identities and shard
> stat metadata matched, without opening a checkpoint payload.

This does not establish current full-file shard SHA256 correctness, checkpoint
assignment correctness, inference correctness, or any performance benefit.
