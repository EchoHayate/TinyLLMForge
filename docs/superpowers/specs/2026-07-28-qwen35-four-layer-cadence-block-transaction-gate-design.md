# Qwen3.5 Four-Layer Cadence-Block Transaction Gate Design

## Status

Completed and independently verified on `sitian@10.232.195.203`.

Authoritative source-bound run:

```text
qwen35-four-layer-cadence-20260728-062455
```

The earlier run `qwen35-four-layer-cadence-20260728-061925` is preserved but
superseded because the final implementation added an explicit fail-closed
alias-partition validator and therefore changed the staged source tree.

This gate extends the proven layer-0/layer-3 heterogeneous transaction to the
first complete architecture cadence block:

```text
layer 0 linear_attention
layer 1 linear_attention
layer 2 linear_attention
layer 3 full_attention
```

It remains a reversible CPU preflight. The production checkpoint loader,
assignment transaction, `target.take()`, candidate installation, CUDA,
Engine, publication, forward, and inference remain forbidden.

## Frozen Binding Contract

Real binding-plan order is not layer-number order. The exact selected indices
are:

```text
layer 0:
  1..14
layer 1:
  15..28
layer 2:
  160..173
layer 3:
  227..237
```

The transaction preserves global binding-plan order:

```text
1..28, 160..173, 227..237
```

Totals:

```text
53 bindings
49 unique destination tensor objects
```

Intentional alias groups:

```text
[12,13]    layer 0 gate/up
[26,27]    layer 1 gate/up
[171,172]  layer 2 gate/up
[229,230]  layer 3 gate/up
```

Every alias pair must be an ordered, disjoint, gap-free partition of one
layer-local `gate_up_proj.weight`. Every other selected destination must be
unique and no selected destination may alias a later-layer tensor.

## Production Tile Contract

All tiles are derived from the exact real 320-binding plan with:

```text
max_tile_bytes = 65536
```

Frozen aggregate contracts:

```text
TP=1 rank0:
  bindings 53
  unique destinations 49
  tiles 7108
  kind counts:
    replicated 13
    axis0 3788
    segmented_axis0 1152
    squeeze_axis0 3
    axis1 2152
  bytes per pass 457755424
  contiguous ranges per pass 7108
  two-pass pread count 14216
  logical payload bytes 915510848

TP=2 rank0/rank1:
  bindings 53
  unique destinations 49
  tiles 3568
  kind counts:
    replicated 13
    axis0 1900
    segmented_axis0 576
    squeeze_axis0 3
    axis1 1076
  bytes per pass 228895376
  contiguous ranges per pass 18876
  two-pass pread count 37752
  logical payload bytes 457790752
```

Per-layer contracts:

```text
linear layer TP=1:
  14 bindings, 13 unique destinations
  1826 tiles, 1826 ranges, 117629536 bytes/pass

linear layer TP=2:
  14 bindings, 13 unique destinations
  917 tiles, 4744 ranges, 58819120 bytes/pass

full layer TP=1:
  11 bindings, 10 unique destinations
  1630 tiles, 1630 ranges, 104866816 bytes/pass

full layer TP=2:
  11 bindings, 10 unique destinations
  817 tiles, 4644 ranges, 52438016 bytes/pass
```

## Four-Layer Transaction

Each fresh rank process:

1. constructs the real CPU target without taking or installing it;
2. validates layer types `linear,linear,linear,full`;
3. selects exactly the frozen 53 bindings and their production tiles;
4. snapshots the 49 unique selected destination objects;
5. opens the approved shard exactly twice;
6. streams every tile through independent production/verifier `pread`s;
7. copies only through `_copy_qwen35_checkpoint_tile`;
8. accumulates per-binding, per-layer, and transaction hashes;
9. records a completion checkpoint before each next layer begins;
10. proves completed layers changed, all future selected layers remain zero,
    and all non-selected model tensors remain zero;
11. validates all four gate/up compositions;
12. restores 49 unique objects in reverse first-seen order;
13. proves every registered tensor is zero after rollback.

Layer completion order must be exactly:

```text
[0,1,2,3]
```

At each transition:

```text
before layer 1: layer 0 changed; layers 1-3 zero
before layer 2: layers 0-1 changed; layers 2-3 zero
before layer 3: layers 0-2 changed; layer 3 zero
```

Returning to an earlier layer or mutating a future layer fails closed.

## Streaming and Memory

The live payload remains bounded by one production tile, one verifier tile,
and one decoded tile:

```text
3 * 65536 bytes
```

Selected snapshot bytes are approximately 436.5 MiB at TP=1 and 218.3 MiB at
TP=2. The complete CPU graph is still materialized once. Conservative
ceilings are:

```text
TP=1 total/post-Torch/post-metadata:
  5505024 / 4980736 / 4718592 KiB

TP=2 total/post-Torch/post-metadata:
  3145728 / 2883584 / 2621440 KiB
```

## Independent Verification

The independent verifier must not import the gate or reuse its range helper.
It reparses the safetensors header and directly reproduces:

- 159 TP-local binding hashes;
- 12 per-layer hashes;
- 3 transaction hashes;
- replicated equality for every replicated binding;
- axis-0 and squeezed rank concatenation;
- segmented Q/K/V rank reconstruction for all three linear layers;
- axis-1 row-interleave reconstruction;
- all 12 rank-local gate/up destination partitions.

## Source Closure and Publication

The source closure is the 41-file heterogeneous two-layer closure plus:

```text
tools/qwen35_real_checkpoint_four_layer_cadence_preflight.py
```

Total: 42 files.

Authoritative output:

```text
four_layer_cadence_preflight.json
source_manifest.json
```

Three rank workers and a separate finalizer are required. Failed remote runs
are preserved and partial local publication is forbidden.

## Allowed Conclusion

Passing proves one complete Qwen3.5 architecture cadence block—three
linear-attention layers followed by one full-attention layer—can participate
in a deterministic, bounded, source-bound, reversible real-checkpoint tile
transaction at TP=1 and TP=2.

It does not prove all six cadence blocks, all 24 layers, the production loader
loop, assignment, candidate installation, inference correctness, production
speed, cache savings, GPU-memory savings, compression, or model quality.

## Completion Evidence

The authoritative run passed with three unique fresh processes:

```text
TP=1 rank0 PID 3613053
TP=2 rank0 PID 3615514
TP=2 rank1 PID 3617493
```

All rows recorded exact layer order `[0,1,2,3]`, three successful transition
checkpoints, 53 bindings, 49 unique destinations, all four alias groups,
complete coverage, no non-selected mutation, reverse rollback, zero loader /
assignment / `target.take()` / forward counters, and CUDA false before and
after.

The independent verifier imported no TinyLLMForge or gate code. It reparsed
the safetensors header and used direct `os.pread` calls to pass:

```text
159 TP-local binding hashes
12 per-layer hashes
3 transaction hashes
53 TP reconstruction checks
12 rank-local shared-destination partition checks
239 checks total
```

Authoritative source and artifacts:

```text
source tree:
  da67572b9db6324b96999b607259824459a980e837d2a275f24bce1a88100148
four_layer_cadence_preflight.json:
  c256f18ff4c528637aaa1d0be49cf0f9a26c07df1c322dba5f29224813e37967
source_manifest.json:
  db89bb9306ca13cc68b9c03d9f72e801ad418fcc0eb94b10058fa0397d72439c
```
