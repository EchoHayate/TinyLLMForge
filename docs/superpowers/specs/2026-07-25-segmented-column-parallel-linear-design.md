# Segmented Column-Parallel Linear Design

## Status

Approved for inline execution under the standing instruction to continue
without per-step confirmation.

This is the second native Qwen3.5 model-math slice. It implements only a
generic tensor-parallel projection/loading primitive and CPU tests. It does
not add a Qwen3.5 model, load a real checkpoint, start a GPU process, or
change the immutable schema-v2 canonical `NO_GO`.

## Problem

`ColumnParallelLinear` shards one contiguous output range. That is correct for
a single logical projection, but incorrect for a fused Qwen3.5 checkpoint
tensor such as:

```text
[Q_global, K_global, V_global]
```

At TP > 1, taking one contiguous shard of the concatenated tensor can cross
logical segment boundaries. The required rank-local layout is:

```text
[Q_rank, K_rank, V_rank]
```

where every logical segment is independently sharded before concatenation.

`MergedColumnParallelLinear` already supports loading separate gate/up source
tensors into a packed local parameter, but its loader requires a segment id
and cannot consume one fused source tensor. Qwen3.5 needs both modes:

- fused checkpoint source, for example `in_proj_qkv.weight`;
- separate source tensors packed into one local parameter in future model
  mappings.

This primitive is only for checkpoint tensors whose rows are globally grouped
by logical segment. It must not be used for a head-major tensor that
interleaves logical values inside every head. In particular, official
Qwen3.5 full-attention `q_proj` stores each head as
`[query_head, query_gate_head]`; that projection requires ordinary contiguous
column sharding over complete head pairs, not segmented `[Q_all, gate_all]`
sharding.

## Interface

Add to `tinyvllm/layers/linear.py`:

```python
class SegmentedColumnParallelLinear(ColumnParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int] | tuple[int, ...],
        bias: bool = False,
    ):
        ...

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_segment_id: int | None = None,
    ):
        ...
```

`output_sizes` describes global logical output sizes in checkpoint order.
Every size must be a positive non-boolean integer divisible by TP size.

The local parameter layout is:

```text
segment 0 local shard
segment 1 local shard
...
segment N local shard
```

The inherited forward path remains unchanged and produces the same packed
local ordering.

## Loading Modes

### Fused source

When `loaded_segment_id is None`, the source must have the exact global shape:

```text
weight: [sum(output_sizes), input_size]
bias:   [sum(output_sizes)]
```

For each segment:

1. narrow the global segment;
2. narrow the current TP rank's shard inside that segment;
3. copy it to the corresponding local segment range.

The implementation must not take one contiguous shard of the full fused
source.

### Separate source

When `loaded_segment_id` is an integer, the source must have the exact global
shape for that segment:

```text
weight: [output_sizes[id], input_size]
bias:   [output_sizes[id]]
```

The loader selects the rank shard and writes only that local segment range.
This preserves compatibility with `packed_modules_mapping`, whose shared id
is forwarded to the parameter loader.

## Failure Semantics

Fail closed with `ValueError` on:

- empty `output_sizes`;
- boolean, non-integer, or non-positive segment sizes;
- a segment not divisible by TP size;
- invalid segment id, including booleans;
- unsupported parameter rank;
- input dimension mismatch;
- fused source output-size mismatch;
- separate source output-size mismatch;
- source/parameter dtype mismatch;
- source/parameter device mismatch.

The loader must validate the complete operation before the first copy so a
failed fused load cannot partially mutate the parameter.

## Test Gate

Use a dependency-light CPU script that loads `linear.py` directly and
temporarily stubs `dist.get_rank()` / `dist.get_world_size()`. It must cover:

- TP=1 fused identity;
- TP=2 and TP=4 fused loading with unequal logical segment sizes;
- rank-local order `[seg0_rank, seg1_rank, ...]`;
- separate-source loading produces the same parameter as fused loading;
- bias follows the same segmented sharding;
- inherited forward uses the assembled local weight;
- invalid constructor, id, shape, dtype, and device contracts fail;
- failed fused validation leaves the destination unchanged.

Synthetic source rows must encode both segment and global row identity so an
accidental contiguous full-tensor shard is visibly different from the
expected result.

## Claim Boundary

Passing this gate proves only rank-local segmented tensor assembly and the
existing linear forward over that assembled parameter. It does not prove:

- Qwen3.5 checkpoint-name mapping or expected-weight completeness;
- Qwen3.5 module construction or forward correctness;
- distributed process-group execution;
- quantized segmented loading;
- any latency, throughput, compression, quality, or memory improvement.

