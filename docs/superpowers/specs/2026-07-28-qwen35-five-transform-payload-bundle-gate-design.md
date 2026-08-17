# Qwen3.5 Five-Transform Payload Bundle Gate Design

## Status

Implemented and passed one source-bound live run on
`sitian@10.232.195.203`.

This is the next bounded payload gate after the one-convolution-tile PASS. It
loads exactly five layer-0 tiles that cover every production tile-copy shape
family used by the current checkpoint plan:

```text
replicated
axis0
segmented_axis0
squeeze_axis0
axis1
```

It does not authorize any checkpoint loader, all-binding loop, assignment
transaction, `target.take()`, candidate installation, CUDA, Engine,
publication, forward, or inference.

## Selected Bundle

The bundle is fixed to these binding indices:

```text
3:
  model.language_model.layers.0.linear_attn.conv1d.weight
  -> layers.0.linear_attention.conv_weight
  BF16 / squeeze_conv_channel / squeeze_axis0

4:
  model.language_model.layers.0.linear_attn.dt_bias
  -> layers.0.linear_attention.dt_bias
  BF16 / identity / axis0

7:
  model.language_model.layers.0.linear_attn.in_proj_qkv.weight
  -> layers.0.linear_attention.in_proj_qkv.weight
  BF16 / identity / segmented_axis0

9:
  model.language_model.layers.0.linear_attn.norm.weight
  -> layers.0.linear_attention.norm_weight
  F32 / identity / replicated

11:
  model.language_model.layers.0.mlp.down_proj.weight
  -> layers.0.mlp.down_proj.weight
  BF16 / identity / axis1
```

All tiles come from the real 320-binding plan built with
`max_tile_bytes=65536`. Selection is by exact binding index and expected first
tile. Any source, target, dtype, transform, kind, shape, slice, offset, or
range deviation fails closed.

## Exact Read Contracts

Safetensors tensor data begins at absolute byte `76656`.

### TP=1 rank0

```text
binding 3:
  shape [6144,4]
  bytes 49152
  ranges [[1017209840,1017258992]]

binding 4:
  shape [16]
  bytes 32
  ranges [[1017258992,1017259024]]

binding 7:
  shape [16,2048]
  bytes 65536
  ranges [[1017390096,1017455632]]

binding 9:
  shape [128]
  bytes 512
  ranges [[76720,77232]]

binding 11:
  shape [5,6144]
  bytes 61440
  ranges [[1059333136,1059394576]]
```

Per pass: `176672` bytes in `5` ranges.

### TP=2 rank0

```text
binding 3:
  shape [3072,4]
  bytes 24576
  ranges [[1017209840,1017234416]]

binding 4:
  shape [8]
  bytes 16
  ranges [[1017258992,1017259008]]

binding 7:
  shape [16,2048]
  bytes 65536
  ranges [[1017390096,1017455632]]

binding 9:
  shape [128]
  bytes 512
  ranges [[76720,77232]]

binding 11:
  shape [10,3072]
  bytes 61440
  ranges:
    [[1059333136,1059339280],
     [1059345424,1059351568],
     [1059357712,1059363856],
     [1059370000,1059376144],
     [1059382288,1059388432],
     [1059394576,1059400720],
     [1059406864,1059413008],
     [1059419152,1059425296],
     [1059431440,1059437584],
     [1059443728,1059449872]]
```

Per pass: `152080` bytes in `14` ranges.

### TP=2 rank1

```text
binding 3:
  shape [3072,4]
  bytes 24576
  ranges [[1017234416,1017258992]]

binding 4:
  shape [8]
  bytes 16
  ranges [[1017259008,1017259024]]

binding 7:
  shape [16,2048]
  bytes 65536
  ranges [[1021584400,1021649936]]

binding 9:
  shape [128]
  bytes 512
  ranges [[76720,77232]]

binding 11:
  shape [10,3072]
  bytes 61440
  ranges:
    [[1059339280,1059345424],
     [1059351568,1059357712],
     [1059363856,1059370000],
     [1059376144,1059382288],
     [1059388432,1059394576],
     [1059400720,1059406864],
     [1059413008,1059419152],
     [1059425296,1059431440],
     [1059437584,1059443728],
     [1059449872,1059456016]]
```

Per pass: `152080` bytes in `14` ranges.

## Independent Read Architecture

Each rank process opens the approved shard exactly twice:

1. production descriptor;
2. verifier descriptor.

For each descriptor, it issues exactly one `os.pread` per frozen range in
bundle order and concatenates range payloads within each tile. Therefore:

```text
TP=1:
  open count 2
  pread count 10
  production bytes 176672
  verifier bytes 176672
  logical bytes 353344

TP=2 per rank:
  open count 2
  pread count 28
  production bytes 152080
  verifier bytes 152080
  logical bytes 304160
```

Short reads, overlapping ranges within one tile, unsorted ranges, byte-count
disagreement, or production/verifier hash disagreement fail closed.

The axis-1 tile is intentionally read as ten row-local column spans at TP=2.
This independently verifies the non-contiguous safetensors layout rather than
delegating correctness to `safe_open.get_slice`.

## Copy, Isolation, and Rollback

Production bytes are decoded to exact CPU tensors using each tile dtype and
shape. The gate then:

1. proves all unique registered tensors are zero;
2. snapshots all five destination slices;
3. copies tiles in bundle order with
   `_copy_qwen35_checkpoint_tile`;
4. after each copy, verifies that tile source and destination hashes match;
5. verifies destinations not yet selected remain zero;
6. verifies all registered tensors outside the five selected destination
   objects remain zero;
7. verifies the five selected destination slices match their source hashes
   simultaneously;
8. rolls back the five snapshots in reverse order;
9. proves every selected snapshot was restored;
10. proves every unique registered tensor is zero.

Selected destination objects are distinct. Any aliasing between selected
destinations fails closed.

## Source Closure and Publication

The source closure is the exact 38-file one-tile closure plus:

```text
tools/qwen35_real_checkpoint_five_transform_bundle_preflight.py
```

Total: 39 files.

Three TP/rank rows run in independent processes. A separate finalizer publishes
only after all rows pass. Authoritative output contains exactly:

```text
five_transform_bundle_preflight.json
source_manifest.json
```

Failed remote directories are preserved. No partial local authoritative
directory is published.

## Memory Ceilings

The bundle adds less than 0.35 MiB of logical read payload over the CPU target,
but retains a conservative 32 MiB margin over the one-tile ceilings:

```text
TP=1 total/post-Torch/post-metadata:
  4767744 / 4243456 / 3981312 KiB

TP=2 total/post-Torch/post-metadata:
  2670592 / 2408448 / 2146304 KiB
```

## Required Row Evidence

Each row records:

```text
selected binding indices [3,4,7,9,11]
selected kinds and dtypes
exact source/destination slices
exact range list per tile
production/verifier bytes and SHA256 per tile
source/destination SHA256 per tile
aggregate production/verifier/logical bytes
open and pread counts
selected destinations distinct
non-selected tensors remained zero
reverse rollback order
all selected snapshots restored
all unique tensors zero after rollback
loader/assignment/target.take/forward counters zero
CUDA before/after false
fresh-process memory points and bounded deltas
```

## Allowed Conclusion

Passing proves five representative real checkpoint tiles spanning F32/BF16,
replicated, TP axis-0, segmented axis-0, squeeze, and non-contiguous TP
axis-1 layouts can be independently read, decoded, copied into exact
registered CPU destinations without cross-destination mutation, and rolled
back exactly.

It does not prove any unselected tile, complete source tensor, complete layer,
all-binding checkpoint loading, assignment transaction, candidate
installation, inference correctness, production speed, cache savings,
GPU-memory savings, compression, or model quality.

## Authoritative Result

Run:

```text
qwen35-five-transform-bundle-20260727-210712
```

Artifacts:

```text
experiments/qwen35_hybrid_state/
qwen35-five-transform-bundle-20260727-210712/
  five_transform_bundle_preflight.json
  source_manifest.json
```

Fresh PIDs:

```text
TP=1 rank0: 2618752
TP=2 rank0: 2620196
TP=2 rank1: 2621492
```

An independent remote verifier reread every frozen range, reproduced all
15 tile hashes, and proved:

```text
TP2 squeeze rank0 || rank1 == TP1 squeeze
TP2 axis0 rank0 || rank1 == TP1 axis0
replicated F32 payload equal across all rows
segmented TP1 first tile == TP2 rank0 first tile
TP2 axis1 row-local column halves reconstruct TP1 axis1 rows
```

No loader, assignment, `target.take()`, forward, CUDA, Engine, publication, or
inference executed.
