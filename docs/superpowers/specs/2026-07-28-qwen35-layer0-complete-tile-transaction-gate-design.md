# Qwen3.5 Layer-0 Complete Tile Transaction Gate Design

## Status

Completed with one source-bound live run on `sitian@10.232.195.203`.

Authoritative run:

```text
qwen35-layer0-transaction-20260728-053410
```

This gate extends the five-transform bundle to every checkpoint binding whose
target starts with `layers.0.`. It remains a dedicated reversible preflight. It
does not call the production checkpoint loader, assignment transaction,
`target.take()`, candidate installation, CUDA, Engine, publication, forward,
or inference.

## Frozen Layer Contract

Layer 0 contains exactly binding indices `1..14`:

```text
1  input_layernorm.weight                 replicated
2  linear_attention.A_log                 axis0
3  linear_attention.conv_weight           squeeze_axis0
4  linear_attention.dt_bias               axis0
5  linear_attention.in_proj_a.weight       axis0
6  linear_attention.in_proj_b.weight       axis0
7  linear_attention.in_proj_qkv.weight     segmented_axis0
8  linear_attention.in_proj_z.weight       axis0
9  linear_attention.norm_weight            replicated
10 linear_attention.out_proj.weight        axis1
11 mlp.down_proj.weight                    axis1
12 mlp.gate_up_proj.weight / gate source   axis0
13 mlp.gate_up_proj.weight / up source     axis0
14 post_attention_layernorm.weight         replicated
```

There are exactly 13 unique destination tensor objects. Bindings `12` and
`13` intentionally share one `gate_up_proj.weight` destination:

```text
TP=1:
  binding 12 destination rows [0,6144)
  binding 13 destination rows [6144,12288)

TP=2:
  binding 12 destination rows [0,3072)
  binding 13 destination rows [3072,6144)
```

The slices are disjoint and exactly cover the shared destination. Any other
alias, overlap, gap, or ordering change fails closed.

## Production Tile Plan

The gate derives tiles from the exact real 320-binding plan with:

```text
max_tile_bytes = 65536
```

Expected aggregate contracts:

```text
TP=1 rank0:
  binding count 14
  unique destination objects 13
  tile count 1826
  kind counts:
    replicated 3
    axis0 900
    segmented_axis0 384
    squeeze_axis0 1
    axis1 538
  bytes per pass 117629536
  contiguous ranges per pass 1826
  two-pass pread count 3652
  logical payload bytes 235259072

TP=2 rank0/rank1:
  binding count 14
  unique destination objects 13
  tile count 917
  kind counts:
    replicated 3
    axis0 452
    segmented_axis0 192
    squeeze_axis0 1
    axis1 269
  bytes per pass 58819120
  contiguous ranges per pass 4744
  two-pass pread count 9488
  logical payload bytes 117638240
```

TP=2 has more ranges than tiles because every axis-1 tile is reconstructed
from one contiguous local-column span per source row.

## Range Derivation

Ranges are derived from approved metadata offsets, dtype width, source shape,
and the production tile source slices:

- rank-1 tensors: one contiguous range;
- full-row rank-2 tiles: one contiguous range;
- TP=2 axis-1 tiles: one range per source row;
- convolution tiles: one contiguous range after the squeezed singleton
  channel is validated.

Every derived range must remain inside the source tensor metadata interval.
Ranges within one tile must be sorted, positive, non-overlapping, and sum to
the tile byte count.

## Streaming Two-Descriptor Transaction

Each rank process opens the approved shard exactly twice, once for production
and once for independent verification. It does not retain all layer payloads.

For each tile in deterministic production-plan order:

1. derive and validate exact ranges;
2. read every range from the production descriptor;
3. read every range from the verifier descriptor;
4. require exact byte and SHA256 equality;
5. decode one CPU tensor with the tile dtype and shape;
6. copy with `_copy_qwen35_checkpoint_tile`;
7. verify source/destination SHA256 equality;
8. update per-binding rolling SHA256, bytes, tile count, and range count;
9. release the tile tensor before advancing.

The peak live payload is therefore one tile from each descriptor plus one
decoded tile, bounded by `3 * 65536` bytes.

## Coverage and Isolation

Before the first copy, snapshot the 13 unique layer-0 destination tensors and
prove all registered model tensors are zero.

During the transaction:

- tiles must be ordered by binding index and production-plan order;
- every tile destination must belong to one of the 13 layer-0 objects;
- tile destination slices for each binding must be non-overlapping and cover
  its exact local shape;
- bindings `12` and `13` must cover the shared destination exactly once;
- all non-layer-0 registered tensors must remain zero;
- completed binding bytes must equal its local destination bytes;
- the aggregate completed bytes must equal the frozen layer bytes.

After all tiles:

- all 14 binding aggregates are complete;
- every layer-0 destination object differs from its zero snapshot;
- all non-layer-0 objects remain zero;
- reverse rollback restores the 13 unique destination snapshots;
- every registered tensor is zero.

Rollback order is reverse first-seen destination-object order. The shared
`gate_up_proj` object is restored once.

## Source Closure and Publication

The source closure is the 39-file five-transform closure plus:

```text
tools/qwen35_real_checkpoint_layer0_transaction_preflight.py
```

Total: 40 files.

Three rank processes run independently. A separate finalizer publishes only
after every row passes. Authoritative output contains exactly:

```text
layer0_transaction_preflight.json
source_manifest.json
```

Failed remote runs are preserved. Partial local publication is forbidden.

## Memory Ceilings

The layer snapshot is approximately 112.2 MiB at TP=1 and 56.1 MiB at TP=2.
Allowing allocator and hash bookkeeping margin, ceilings are:

```text
TP=1 total/post-Torch/post-metadata:
  5033168 / 4508876 / 4246732 KiB

TP=2 total/post-Torch/post-metadata:
  2818048 / 2555904 / 2293760 KiB
```

## Required Evidence

Each row records:

```text
binding indices 1..14
13 unique destination objects
alias group [12,13] and exact destination slices
tile/kind/range counts
production/verifier/logical bytes
per-binding source, target, dtype, kind, local shape
per-binding rolling production/verifier/source/destination SHA256
per-binding bytes, tiles, ranges, and complete coverage
aggregate source/destination SHA256
non-layer-0 isolation
reverse unique-object rollback
all snapshots restored
all registered tensors zero
loader/assignment/target.take/forward counters zero
CUDA before/after false
fresh-process memory points and bounded deltas
```

## Allowed Conclusion

Passing proves the complete layer-0 checkpoint destination set can be loaded
tile-by-tile from the approved real shard with exact TP-aware source slicing,
all five tile layout families, shared-destination slice composition,
non-layer isolation, bounded streaming payload memory, and exact rollback.

It does not prove layer 1-23 loading, full-model checkpoint loading, the
production loader loop, assignment transaction, candidate installation,
inference correctness, production speed, cache savings, GPU-memory savings,
compression, or model quality.

## Completed Evidence

All three fresh rank processes passed:

```text
TP=1 rank0:
  PID 2959409
  tiles/ranges-per-pass/preads 1826/1826/3652
  production/verifier/logical bytes
    117629536/117629536/235259072
  aggregate SHA256
    c997ff8c44d42bee1e2083355c48fb11f2bb69a60ded7770a2def4aef1843588
  total/post-Torch/post-metadata VmHWM increment KiB
    4364272/4019828/3889640

TP=2 rank0:
  PID 2961371
  tiles/ranges-per-pass/preads 917/4744/9488
  production/verifier/logical bytes
    58819120/58819120/117638240
  aggregate SHA256
    33bf0596584cc179214da1e8bee098824bf0262c951a6ae46f413674d1248810
  total/post-Torch/post-metadata VmHWM increment KiB
    2428148/2084112/1954940

TP=2 rank1:
  PID 2962718
  tiles/ranges-per-pass/preads 917/4744/9488
  production/verifier/logical bytes
    58819120/58819120/117638240
  aggregate SHA256
    2ecd602b6d095a00781854f1090d3fc9c53285c32191b23fee6ecffa691adf30
  total/post-Torch/post-metadata VmHWM increment KiB
    2428196/2083668/1954272
```

Every row recorded exactly 14 binding results, 13 unique destination
objects, alias group `[12,13]`, two read-only shard descriptors, zero
loader/assignment/`target.take()`/forward calls, CUDA false before and after,
non-layer isolation, and rollback order:

```text
[14,12,11,10,9,8,7,6,5,4,3,2,1]
```

Independent direct-`pread` verification did not import this gate or reuse its
range helper. It reparsed the safetensors header and completed 62 checks:

```text
42 TP-local binding payload hashes reproduced
3 aggregate row hashes reproduced
3 replicated payload equalities
8 axis-0/squeeze TP reconstructions
1 segmented Q/K/V TP reconstruction
2 axis-1 row-interleave TP reconstructions
3 shared gate/up destination partitions
```

Authoritative source and artifact hashes:

```text
source tree:
  0be5c56dd5c49f4e257d14fbb478e5ae6170a0b529852c537272897b23d679fd
layer0_transaction_preflight.json:
  167d3ee5e3b0996ebab9331f17e36d1775ea775c39964d2f4c1e4de3c9820b73
source_manifest.json:
  cb3e7e4e79590eb77c3e13a74b0afc38f89e08b9ded98bbe8cdbc1e54c7aee93
```

The remote inventory is exactly the 40-file source closure plus the two
authoritative artifacts.

## Next Gate Boundary

The next safe step is a bounded heterogeneous two-layer transaction: one
complete linear-attention layer and one complete full-attention layer. It
must prove cross-layer destination isolation, layer-local shared aliases,
deterministic transaction ordering, bounded streaming reads, and reverse
rollback across both layers.

It must remain a dedicated CPU preflight. The production checkpoint loader,
assignment transaction, `target.take()`, candidate installation, CUDA,
Engine, publication, forward, and inference remain forbidden.
