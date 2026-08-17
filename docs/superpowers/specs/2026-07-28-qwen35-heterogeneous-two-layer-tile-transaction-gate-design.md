# Qwen3.5 Heterogeneous Two-Layer Tile Transaction Gate Design

## Status

Completed with one source-bound live run on `sitian@10.232.195.203`.

Authoritative run:

```text
qwen35-heterogeneous-two-layer-20260728-060325
```

This gate extends the proven complete layer-0 transaction to one
linear-attention layer and one full-attention layer. It remains a dedicated
reversible CPU preflight and does not call the production checkpoint loader,
assignment transaction, `target.take()`, candidate installation, CUDA,
Engine, publication, forward, or inference.

## Layer Selection

The approved Qwen3.5-2B schedule has linear-attention layers `0,1,2` followed
by the first full-attention layer at index `3`.

Three possible increments were considered:

1. layers `0+1`: smallest conceptual increment, but both layers use the same
   linear-attention structure and add little layout diversity;
2. layers `0+3`: one complete linear-attention layer plus the first complete
   full-attention layer;
3. all 24 layers: highest coverage, but jumps directly to the full 4.5 GB
   assignment surface before cross-layer rollback is isolated.

The gate freezes option 2. It adds full-attention Q/K/V/output and norm
bindings while limiting the transaction to two layers.

## Frozen Binding Contract

Selected binding indices:

```text
layer 0:
  1..14

layer 3:
  227..237
```

Total:

```text
25 bindings
23 unique destination tensor objects
```

Intentional alias groups:

```text
[12,13]    layers.0.mlp.gate_up_proj.weight
[229,230]  layers.3.mlp.gate_up_proj.weight
```

Both pairs must be disjoint, ordered, gap-free destination-row partitions.
Every other selected destination must be unique, and no selected destination
may alias a non-selected destination.

The first-binding order for unique destination objects is:

```text
1,2,3,4,5,6,7,8,9,10,11,12,14,
227,228,229,231,232,233,234,235,236,237
```

Rollback must restore these 23 objects in exact reverse order.

## Layer-3 Full-Attention Contract

Layer 3 contains exactly:

```text
227 input_layernorm.weight
228 mlp.down_proj.weight
229 mlp.gate_up_proj.weight / gate source
230 mlp.gate_up_proj.weight / up source
231 post_attention_layernorm.weight
232 full_attention.k_norm.weight
233 full_attention.k_projection.weight
234 full_attention.output_projection.weight
235 full_attention.q_norm.weight
236 full_attention.q_projection.weight
237 full_attention.v_projection.weight
```

Full-attention Q projection is an axis-0 head-paired destination, K/V are
axis-0 destinations, output projection is axis-1, and Q/K norms are
replicated. The existing production binding and tile planners remain the
source of truth.

## Production Tile Plan

All tiles are derived from the exact 320-binding real plan with:

```text
max_tile_bytes = 65536
```

Frozen aggregate contracts:

```text
TP=1 rank0:
  bindings 25
  unique destinations 23
  aliases [12,13], [229,230]
  tiles 3456
  kind counts:
    replicated 7
    axis0 1988
    segmented_axis0 384
    squeeze_axis0 1
    axis1 1076
  bytes per pass 222496352
  contiguous ranges per pass 3456
  two-pass pread count 6912
  logical payload bytes 444992704

TP=2 rank0/rank1:
  bindings 25
  unique destinations 23
  aliases [12,13], [229,230]
  tiles 1734
  kind counts:
    replicated 7
    axis0 996
    segmented_axis0 192
    squeeze_axis0 1
    axis1 538
  bytes per pass 111257136
  contiguous ranges per pass 9388
  two-pass pread count 18776
  logical payload bytes 222514272
```

TP=2 has more ranges than tiles because each axis-1 tile is reconstructed
from one local-column range per source row.

## Streaming Transaction

Each fresh rank process:

1. constructs the real CPU target without taking or installing it;
2. selects only bindings `1..14` and `227..237`;
3. derives all selected production tiles in global production order;
4. validates the two-layer and alias contracts before opening payload;
5. snapshots the 23 selected destination objects;
6. opens the approved shard exactly twice;
7. independently `pread`s every tile range through both descriptors;
8. requires production/verifier byte equality;
9. decodes at most one tile and copies only through
   `_copy_qwen35_checkpoint_tile`;
10. verifies the destination slice hash after every copy;
11. accumulates per-binding, per-layer, and transaction hashes;
12. proves all non-selected model tensors remain zero;
13. restores all 23 unique objects in reverse first-seen order;
14. proves every registered tensor is zero after rollback.

The live payload remains bounded by one production tile, one verifier tile,
and one decoded tile: at most `3 * 65536` bytes. Snapshot memory is separate
and explicitly measured.

## Cross-Layer Isolation

After completing layer 0 and before the first layer-3 copy:

```text
all layer-0 destinations changed
all layer-3 destinations remain zero
all other model tensors remain zero
```

After completing layer 3:

```text
all selected destinations changed
all non-selected tensors remain zero
both gate/up aliases are exactly composed
```

Layer completion order must be exactly `[0,3]`. A tile returning to layer 0
after layer 3 begins fails closed.

## Independent Verification

The independent verifier must not import the gate or reuse its range helper.
It reparses the safetensors header and directly reconstructs:

- all 75 TP-local binding hashes;
- all three transaction aggregate hashes;
- TP replicated equality;
- axis-0 and squeezed rank concatenation;
- segmented Q/K/V rank reconstruction;
- axis-1 row-interleave reconstruction;
- both shared gate/up destinations;
- layer-0 and layer-3 per-layer aggregate hashes.

## Source Closure and Publication

The source closure is the completed 40-file layer-0 closure plus:

```text
tools/qwen35_real_checkpoint_heterogeneous_two_layer_preflight.py
```

Total: 41 files.

Authoritative output contains exactly:

```text
heterogeneous_two_layer_preflight.json
source_manifest.json
```

Three rank workers and a separate finalizer are required. Partial local
publication is forbidden and failed remote directories are preserved.

## Memory Ceilings

Selected destination bytes are approximately 212.2 MiB at TP=1 and
106.1 MiB at TP=2. The target still materializes the complete CPU graph, so
ceilings extend the measured layer-0 values only for the additional selected
snapshot and bookkeeping:

```text
TP=1 total/post-Torch/post-metadata:
  5242880 / 4718592 / 4456448 KiB

TP=2 total/post-Torch/post-metadata:
  2949120 / 2686976 / 2424832 KiB
```

## Fail-Closed Conditions

The gate rejects:

- any schedule other than layer 0 linear and layer 3 full attention;
- any selected binding/index/order drift;
- missing, duplicate, or out-of-order tiles;
- an alias group other than `[12,13]` or `[229,230]`;
- gate/up overlap, gap, reversed order, or incomplete composition;
- incomplete per-binding or per-layer coverage;
- any layer-3 mutation before layer 0 completes;
- any non-selected tensor mutation;
- short read, range mismatch, production/verifier mismatch, or copy mismatch;
- descriptor, byte, tile, range, memory, PID, source, or artifact drift;
- rollback order or restored-value mismatch;
- any loader, assignment, `target.take()`, forward, CUDA, Engine,
  publication, or inference execution.

## Allowed Conclusion

Passing proves one complete linear-attention layer and one complete
full-attention layer can participate in a deterministic, bounded,
source-bound, reversible real-checkpoint tile transaction for TP=1 and TP=2.
It proves the full-attention binding layouts, cross-layer isolation, two
layer-local shared destinations, and reverse multi-layer rollback.

It does not prove all 24 layers, the production loader loop, assignment,
candidate installation, inference correctness, production speed, cache
savings, GPU-memory savings, compression, or model quality.

## Completed Evidence

Fresh rank results:

```text
TP=1 rank0:
  PID 3332932
  tiles/ranges-per-pass/preads 3456/3456/6912
  production/verifier/logical bytes
    222496352/222496352/444992704
  transaction SHA256
    a5569f1f87650b9cef5313ce7ce7acf2368baf763243955b3c60c5a02a922c90
  layer 0 SHA256
    c997ff8c44d42bee1e2083355c48fb11f2bb69a60ded7770a2def4aef1843588
  layer 3 SHA256
    1b3d45a0c128533adc807732619b3768971b0244cefefc4f13824683f295ac8d
  total/post-Torch/post-metadata VmHWM increment KiB
    4468884/4123948/3994120

TP=2 rank0:
  PID 3335035
  tiles/ranges-per-pass/preads 1734/9388/18776
  production/verifier/logical bytes
    111257136/111257136/222514272
  transaction SHA256
    104ae396ac9a791ef36bfe5b58917e710d2d23190df4b89839f88b382af2b170
  layer 0 SHA256
    33bf0596584cc179214da1e8bee098824bf0262c951a6ae46f413674d1248810
  layer 3 SHA256
    76e81362cd67490661199a9bc59152b14e61e8c2803bedfc6b5a03ff9dc9102a
  total/post-Torch/post-metadata VmHWM increment KiB
    2481204/2137272/2007632

TP=2 rank1:
  PID 3336758
  tiles/ranges-per-pass/preads 1734/9388/18776
  production/verifier/logical bytes
    111257136/111257136/222514272
  transaction SHA256
    edd2138296775a7f347cff2df9921a0953b39e3e6b77c3a627df775ea9a29772
  layer 0 SHA256
    2ecd602b6d095a00781854f1090d3fc9c53285c32191b23fee6ecffa691adf30
  layer 3 SHA256
    2be84a53dd81b7182cb9b4b17b0d73800481a69b6284f60bcd0c1aec54aa9013
  total/post-Torch/post-metadata VmHWM increment KiB
    2480948/2136800/2007288
```

Every row proved:

```text
25 complete binding results
23 unique selected destination objects
alias groups [12,13] and [229,230]
layer completion order [0,3]
all layer-0 destinations changed before layer 3
all layer-3 destinations zero before its first copy
all non-selected model tensors remained zero
reverse unique-object rollback restored every snapshot
loader/assignment/target.take/forward counters zero
CUDA false before and after
```

Independent verification reparsed the safetensors header and directly
`pread` the approved shard without importing the gate. It passed 115 checks:

```text
75 TP-local binding hashes reproduced
3 transaction hashes reproduced
6 per-layer hashes reproduced
7 replicated equalities
13 axis-0/squeeze TP reconstructions
1 segmented Q/K/V TP reconstruction
4 axis-1 row-interleave TP reconstructions
6 shared gate/up destination partitions
```

Authoritative hashes:

```text
source tree:
  66ef9e8e5c12eb8b06ed419c356035773afef04cb8c7c3985320841d3f4a940e
heterogeneous_two_layer_preflight.json:
  e0cf1f0d1e48347b771aae045e22fa0b81b9dd64e50fda79d01235fddb37bad9
source_manifest.json:
  352ace9a3c34f6f4f3e97fd9ec54bd968081af36c27e14785eeb0c0a4ee4037a
```

The remote inventory is exactly the 41-file source closure plus the two
authoritative artifacts.

## Next Gate Boundary

The next safe transaction covers one complete model cadence block:
layers `0..3`, consisting of three linear-attention layers followed by one
full-attention layer. It must add two more complete linear layers while
preserving per-layer completion order, four layer-local gate/up aliases,
cross-layer isolation, bounded payload streaming, and reverse rollback.

It remains a dedicated CPU preflight. The production loader, assignment,
`target.take()`, candidate installation, CUDA, Engine, publication, forward,
and inference remain forbidden.
