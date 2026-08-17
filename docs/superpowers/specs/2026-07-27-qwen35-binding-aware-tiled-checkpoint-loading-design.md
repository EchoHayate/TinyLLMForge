# Qwen3.5 Binding-Aware Tiled Checkpoint Loading Design

## Status

Approved under the standing inline-execution direction. This is a CPU-only
correctness and bounded-materialization gate using temporary small
safetensors shards plus the already verified real checkpoint metadata. It must
not open the real checkpoint payload, start GPU work, or connect production
ModelRunner, Engine, or Scheduler.

## Goal

Load every Qwen3.5 checkpoint binding into a fresh unpublished CPU candidate
through `safetensors.safe_open().get_slice()`, materializing only bounded
binding-aware tiles rather than complete source tensors.

The hard source-memory contract is:

```text
every materialized tile byte count <= max_tile_bytes
```

The loader writes already TP-local tiles directly into their final destination
regions. It does not pass sliced tensors to existing full-source custom
loaders, because those loaders would attempt to shard a second time.

## Current Evidence

Safetensors `0.7.0` exposes:

```python
slice_view = handle.get_slice(name)
shape = slice_view.get_shape()
tensor = slice_view[row_start:row_end, column_start:column_end]
```

Temporary CPU evidence confirmed materialization of row, column, two-axis, and
rank-reducing slices.

The verified real Qwen3.5 language-model plan contains:

```text
bindings: 320
unique sources: 320
axis-0: 157
axis-1: 48
segmented axis-0: 18
squeeze plus axis-0: 18
replicated: 79
```

At TP=1, the largest complete local source is the embedding at
`1,017,118,720` bytes, but one embedding row is only `4,096` bytes. The largest
single destination-row tile unit across all 320 bindings is a row-parallel MLP
down-projection row:

```text
12,288 bytes at TP=1
6,144 bytes at TP=2
```

Therefore a unified row-tiled loader can bound source materialization far
below the largest source tensor without changing checkpoint values.

## Architecture

Create two focused modules:

```text
tinyvllm/models/qwen35_checkpoint_tiles.py
tinyvllm/models/qwen35_checkpoint_tiled_loading.py
```

`qwen35_checkpoint_tiles.py` is file-I/O-free. It converts one immutable
binding plan plus a byte budget into an immutable sequence of exact source and
destination slice records.

`qwen35_checkpoint_tiled_loading.py` owns fresh-candidate construction, path
validation, `safe_open`, `get_slice`, tile materialization, direct destination
writes, failure discard, owner construction, and scalar statistics.

The existing all-source transactional assignment and one-source streamed
loader remain unchanged.

## Tile Plan API

```python
@dataclass(frozen=True)
class Qwen35CheckpointTile:
    binding_index: int
    source_name: str
    shard: str
    source_tensor_shape: tuple[int, ...]
    source_slices: tuple[slice | int, ...]
    tile_shape: tuple[int, ...]
    destination: torch.Tensor
    destination_slices: tuple[slice, ...]
    destination_shape: tuple[int, ...]
    dtype: torch.dtype
    byte_count: int
    target: str
    kind: str


@dataclass(frozen=True)
class Qwen35CheckpointTilePlan:
    tiles: tuple[Qwen35CheckpointTile, ...]
    tensor_parallel_size: int
    tensor_parallel_rank: int
    binding_count: int
    source_count: int
    destination_bytes: int
    peak_tile_bytes: int


def build_qwen35_checkpoint_tile_plan(
    binding_plan: Qwen35CheckpointBindingPlan,
    *,
    max_tile_bytes: int,
) -> Qwen35CheckpointTilePlan:
    ...
```

Every real source is unique today, but the planner must reject conflicting
duplicate source contracts rather than assuming uniqueness.

`binding_index` is the binding-plan position. It distinguishes packed gate/up
bindings that share the same target string and destination object.
`source_tensor_shape` is the complete immutable checkpoint shape;
`tile_shape` is the tensor shape returned by `PySafeSlice.__getitem__`.

## Supported Binding Classes

The planner classifies by exact binding target, loader kind, transform,
destination shape, destination slice, and exact custom-loader owner type.

### Axis-0 local rows

Includes:

- vocabulary embedding;
- ordinary `ColumnParallelLinear`;
- `HeadPairedColumnParallelLinear`;
- one packed gate/up source for `MergedColumnParallelLinear`;
- direct buffers `A_log` and `dt_bias`.

For local destination row `r`, the source global row is:

```text
rank * local_rows + r
```

Packed gate/up tiles write into the binding's declared destination row slice.
All columns/trailing dimensions for each selected source row are materialized.

### Segmented axis-0 local rows

`SegmentedColumnParallelLinear` source rows contain concatenated global
segments. For each segment:

```text
source segment start = sum(global segment sizes before it)
source local start   = source segment start + rank * local segment rows
destination start    = sum(local segment rows before it)
```

Each segment is tiled independently so no tile crosses a Q/K/V boundary.
Segment sizes come from the exact bound loader owner and must agree with
metadata and destination shape.

### Axis-1 local columns

Includes:

- MLP down projection;
- linear-attention output projection;
- full-attention output projection.

All source rows belong to every rank. The source column interval is:

```text
[rank * local_columns, (rank + 1) * local_columns)
```

Rows are tiled so each materialized `[tile_rows, local_columns]` tensor stays
within budget. The tensor copies directly to the corresponding destination
rows and all local columns.

### Convolution squeeze plus axis-0

The source shape is:

```text
[global_channels, 1, kernel]
```

The destination shape is:

```text
[local_channels, kernel]
```

Each tile uses:

```python
slice_view[source_rows, 0, :]
```

Safetensors materializes the exact transformed two-dimensional tensor, so no
full source and no extra squeeze view are retained.

### Replicated

Includes offset RMSNorm parameters and
`linear_attention.norm_weight`. These are one-dimensional and small in the
real plan, but they are still divided into one-dimensional tiles so the hard
budget remains universal.

## Tile Size Calculation

`max_tile_bytes` must be a positive non-boolean integer.

For each binding class, compute the indivisible byte unit:

```text
axis-0 matrix row          = product(source_shape[1:]) * dtype_bytes
segmented axis-0 row      = product(source_shape[1:]) * dtype_bytes
axis-1 destination row    = local_columns * dtype_bytes
convolution channel row   = kernel * dtype_bytes
replicated scalar         = dtype_bytes
```

Reject the complete plan before file I/O if any indivisible unit exceeds the
budget. Otherwise:

```text
rows_per_tile = max(1, max_tile_bytes // unit_bytes)
```

Every tile record must have exact positive source/destination shapes and:

```text
byte_count <= max_tile_bytes
```

The planner verifies exact complete coverage of every destination element
associated with every binding, with no overlap except the intentional
gate/up disjoint slices of one packed destination.

## Direct Write Contract

The tiled loader validates each materialized tensor against its tile record:

- exact CPU tensor;
- exact dtype;
- exact source-derived shape;
- exact byte count;
- exact destination-slice shape.

Then under `torch.no_grad()`:

```python
destination[destination_slices].copy_(tile_tensor)
```

No custom weight loader executes. This is safe only because the tile planner
has already encoded the same TP semantics and the candidate is unpublished.

The test gate installs custom loaders that raise if called, proving tiled
loading bypasses them.

## Fresh Candidate and Failure Boundary

Public API:

```python
@dataclass(frozen=True)
class Qwen35TiledCheckpointLoadStats:
    assigned_bindings: int
    source_tensors: int
    shard_count: int
    tile_count: int
    destination_bytes: int
    materialized_bytes: int
    peak_tile_bytes: int


@dataclass(frozen=True)
class Qwen35TiledLoadedCheckpointCandidate:
    owner: Qwen35HybridModelOwner
    binding_plan: Qwen35CheckpointBindingPlan
    tile_plan: Qwen35CheckpointTilePlan
    stats: Qwen35TiledCheckpointLoadStats


def load_qwen35_fresh_checkpoint_candidate_tiled(
    candidate_factory,
    checkpoint_dir,
    *,
    max_tile_bytes: int,
) -> Qwen35TiledLoadedCheckpointCandidate:
    ...
```

The factory, candidate ownership, CPU destination, safe path, and publication
rules match the completed streamed fresh-candidate gate.

All shard paths and the complete tile plan validate before any file opens.
On any read, validation, or copy failure:

- every entered handle closes;
- no loaded candidate is returned;
- the private partially written candidate is discarded;
- no rollback snapshot is allocated;
- any existing publication slot remains unchanged.

The existing publication slot is intentionally typed to the previous loaded
candidate class. This gate does not widen it until a shared sealed candidate
protocol is designed; tiled loading proves candidate construction only.

## Statistics

`destination_bytes` is the sum of all tile byte counts and equals the total
rank-local destination payload written, counting each packed slice once.

`materialized_bytes` is the sum of all materialized tile bytes. For this exact
lossless direct-copy design it equals `destination_bytes`.

`peak_tile_bytes` is the maximum tile byte count in the plan and at runtime.
It bounds one explicit tensor returned from `PySafeSlice.__getitem__`, not
total process RSS, file mapping, allocator state, or destination memory.

## Test Strategy

### Static real-plan proof

Using only the verified real config/index/header and a meta graph:

- build TP=1/2 tile plans for all 320 bindings;
- classify exactly `157/48/18/18/79`;
- prove all source/destination coverage;
- prove every tile is within a chosen small budget;
- verify TP=1 minimum feasible budget is `12,288` bytes and TP=2 is `6,144`;
- never open a safetensors payload.

### Temporary shard exactness

Using the existing 27-source two-layer CPU fixture:

- force multiple tiles per large source with a small budget;
- load TP=1 rank 0 and TP=2 ranks 0/1;
- compare all destinations to independent full-source expectations;
- prove custom loaders never execute;
- track every requested `get_slice()` and reject `get_tensor()`;
- weak-reference every materialized tile and prove release before the next
  tile;
- verify exact stats and tied embedding storage.

### Failure matrix

Cover:

- invalid budget or unsupported binding class;
- budget below one indivisible unit;
- malformed or conflicting tile/source contract;
- wrong `get_slice().get_shape()`;
- missing source;
- materialized tile wrong shape or dtype;
- destination copy failure;
- balanced handle cleanup and no loaded candidate.

## Non-Goals

This gate does not:

- open or assign the real checkpoint payload;
- prove real total RSS, page-cache behavior, or load latency;
- publish a tiled candidate into production;
- load GPU tensors or execute model forward;
- compare checkpoint tokens or logits;
- wire generic loader, ModelRunner, Engine, or Scheduler;
- alter state-pool ownership;
- establish any speed, cache, memory, compression, or quality improvement.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge can derive exact TP-local tile slices for every verified
> Qwen3.5 checkpoint binding and load a fresh CPU candidate from temporary
> safetensors shards while bounding each explicitly materialized source tile
> by `max_tile_bytes`, without materializing complete checkpoint tensors or
> executing full-source custom loaders.

