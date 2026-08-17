# Qwen3.5 One-Convolution-Tile Payload Gate Design

## Status

Implemented and passed one live source-bound run on
`sitian@10.232.195.203`.

This is the first gate authorized to read real tensor payload bytes. It is
limited to one fixed layer-0 convolution tile per fresh TP/rank process. It
does not authorize any loader, full source tensor, all-binding loop,
checkpoint assignment transaction, target consumption, candidate creation,
Engine installation, CUDA, or inference.

## Selected Binding

```text
source:
  model.language_model.layers.0.linear_attn.conv1d.weight
target:
  layers.0.linear_attention.conv_weight
binding index:
  3
dtype:
  BF16
source shape:
  [6144, 1, 4]
transform:
  squeeze_conv_channel
payload-relative offsets:
  [1017133184, 1017182336)
```

The approved safetensors header is 76,648 bytes, so tensor data begins at
absolute file byte 76,656.

Rank tiles:

```text
TP=1 rank0:
  source rows [0, 6144)
  destination rows [0, 6144)
  tile shape [6144, 4]
  bytes 49152
  absolute file range [1017209840, 1017258992)

TP=2 rank0:
  source rows [0, 3072)
  destination rows [0, 3072)
  tile shape [3072, 4]
  bytes 24576
  absolute file range [1017209840, 1017234416)

TP=2 rank1:
  source rows [3072, 6144)
  destination rows [0, 3072)
  tile shape [3072, 4]
  bytes 24576
  absolute file range [1017234416, 1017258992)
```

## Architecture

Create:

```text
tools/qwen35_real_checkpoint_one_tile_payload_preflight.py
tools/test_qwen35_real_checkpoint_one_tile_payload_preflight.py
```

The gate reuses the bounded CPU materialization gate to construct and
zero-touch one empty rank target in a fresh process. It then:

1. derives the fixed tile from the real 320-binding plan with
   `max_tile_bytes=65536`;
2. rejects any deviation in binding index, source, target, dtype, transform,
   shape, slices, offsets, or absolute file range;
3. opens the approved shard with `os.open(..., O_RDONLY)` and reads exactly
   the selected tile bytes using `os.pread`;
4. decodes the first byte string as contiguous BF16 `[rows, 4]`;
5. independently reopens the shard and performs a second exact `os.pread` for
   the same range;
6. requires identical SHA256 values for production and verifier reads;
7. snapshots the selected destination slice and verifies it is all zero;
8. copies the decoded tile with the existing
   `_copy_qwen35_checkpoint_tile` production primitive;
9. verifies destination bytes and SHA256 exactly match the source tile;
10. proves every non-selected unique registered tensor remains zero;
11. restores the selected destination snapshot;
12. proves all unique registered tensors are zero after rollback;
13. emits one validated row and exits.

The two reads are intentionally separate file descriptors. Logical payload
bytes read are exactly twice the tile size:

```text
TP=1: 98304
TP=2: 49152 per rank
```

No header, config, or index bytes are counted as tensor payload.

## Source Closure

The source inventory is:

```text
32 frozen production files
checkpoint tile plan
checkpoint tile policy
checkpoint tiled loading
target-preparation gate
CPU-materialization gate
one-tile gate
```

Total: 38 files.

## Exact Row Contract

Each row retains all CPU materialization assertions and adds:

```text
selected binding index: 3
selected tile count: 1
selected source count: 1
selected shard count: 1
production payload bytes: exact tile bytes
verifier payload bytes: exact tile bytes
logical payload bytes read: 2 * tile bytes
production/verifier/source/destination SHA256: exact match
destination initially zero: true
destination changed after copy: true
non-selected tensors remained zero: true
rollback restored selected destination: true
all unique tensors zero after rollback: true
open count: 2
pread count: 2
loader/assignment/target.take/forward calls: 0
CUDA before/after: false/false
```

The process memory ceilings remain those of the CPU materialization gate plus
at most 16 MiB:

```text
TP=1 total/post-Torch/post-metadata:
  4734976 / 4210688 / 3948544 KiB
TP=2 total/post-Torch/post-metadata:
  2637824 / 2375680 / 2113536 KiB
```

## Failure and Publication

Each TP/rank row runs in an independent process. A separate finalizer
publishes only after all three rows pass. Local authoritative output contains:

```text
one_tile_payload_preflight.json
source_manifest.json
```

Any mismatch, short read, hash disagreement, copy failure, non-selected
mutation, rollback failure, memory violation, or process failure publishes no
local authoritative directory. Remote failed directories are preserved.

## Safety

- Read only the fixed absolute byte range for the selected tile.
- Never call `safe_open.get_tensor`, the full streaming loader, tiled
  candidate loader, checkpoint assignment, or real worker `main()`.
- Never call `target.take()`.
- Never read another binding or tile.
- Keep CUDA, Engine, publication, and inference absent.
- Preserve schema-v2 canonical `NO_GO`.
- Do not stage, commit, merge, overwrite, or delete evidence.
- Do not claim production performance or model quality.

## Allowed Conclusion

Passing proves the approved real shard can provide one exact TP-aware,
BF16, squeeze-transformed convolution tile whose bytes independently verify,
copy into the correct registered CPU destination slice, leave all other model
storage untouched, and roll back to the all-zero state.

It does not prove any other tensor, full-source loading, all-binding loading,
transactional checkpoint assignment, candidate installation, inference
correctness, performance, cache savings, or quality.

## Authoritative Result

Run:

```text
qwen35-one-tile-payload-20260727-204441
```

Artifacts:

```text
experiments/qwen35_hybrid_state/
qwen35-one-tile-payload-20260727-204441/
  one_tile_payload_preflight.json
  source_manifest.json
```

All three fresh processes passed. The TP=2 rank payloads concatenate
byte-for-byte to the TP=1 payload:

```text
TP=1 rank0:
  PID 2328814
  SHA256 0dbb863f97d7ac62ca2e452e0fe1487edb5d954e2380192102aa1ace8f40642a

TP=2 rank0:
  PID 2330411
  SHA256 406a3bd779dbb7a92796e386c2e7843206d399a54e2186edf7d1f7b7f974e1e0

TP=2 rank1:
  PID 2331834
  SHA256 a20c9d32c149d39d248c146c65d6ec620b32456423c4d5488b71aeb2cfcc15f4
```

An independent remote `pread` reproduced all three hashes and proved
`TP2-rank0 || TP2-rank1 == TP1-rank0`. No loader, assignment,
`target.take()`, forward, CUDA, Engine, publication, or inference executed.
