# Qwen3.5 Five-Binding Synthetic Replay Calibration Design

## Status

Approved by the standing inline-execution direction and the completed
synthetic tile-copy handoff. This is a local CPU-only benchmark over generated
temporary safetensors payloads. It must not read a real checkpoint, start GPU
work, or change production loading/runtime wiring.

## Goal

Replay bounded synthetic representatives of all five verified Qwen3.5
checkpoint tile grammars through the existing immutable tile planner and exact
`PySafeSlice` source/destination slices, then determine whether the current
8-16 MiB policy region remains reasonable across non-axis-0 layouts.

## Representative Cases

Use TP=2 rank1 and one independent BF16 binding per case:

```text
axis0
axis1
segmented_axis0
squeeze_axis0
replicated
```

Each case has exactly 32 MiB of TP-local destination data. Global source data
may be larger where TP sharding requires it. The segmented case uses an exact
`SegmentedColumnParallelLinear` owner so the existing planner validates the
same segment contract as production binding plans.

The source values are deterministic and exactly representable in BF16. Each
destination starts with a sentinel value and is compared element-for-element
with an independently derived expected TP-local result after every warm-up and
timed replay.

## Benchmark Matrix

Evaluate:

```text
tile budgets: 4, 8, 16, 32 MiB
warm-ups: 1 per case/budget
timed repeats: 3 per case/budget
```

For each repeat:

1. reset the final destination sentinel outside timing;
2. open the temporary shard on CPU;
3. call `get_slice()` once for the case source;
4. iterate the exact `Qwen35CheckpointTilePlan.tiles`;
5. materialize each planner-provided `source_slices`;
6. copy into the exact planner-provided `destination_slices`;
7. close the handle;
8. compare the full destination against the independent expected tensor
   outside timing;
9. record elapsed time, call count, planned peak bytes, and exact verification.

The benchmark must use the public
`build_qwen35_checkpoint_tile_plan()` result without reconstructing tile
slices in the benchmark.

## Public Harness

Create:

```text
tools/benchmark_qwen35_five_binding_tile_replay.py
```

with:

```python
def run_qwen35_five_binding_tile_replay_calibration(
    *,
    tile_mib: tuple[int, ...] = (4, 8, 16, 32),
    repeats: int = 3,
) -> dict:
    ...
```

The result schema is:

```text
schema_version
environment
configuration
cases
interpretation_limits
```

Each case records source/local shapes and bytes. Each budget record contains:

```text
raw_seconds
median_seconds
min_seconds
max_seconds
tile_count
requested_tile_bytes
peak_tile_bytes
destination_checksum
exact_destination_verified
```

The CLI accepts a comma-separated budget list, repeat count, and optional
`--output-json PATH`, persisted atomically.

## Correctness and Failure Contract

The harness rejects invalid repeats, empty/non-positive budgets, or budgets
smaller than an indivisible planner unit. It requires exactly one planner
binding and one source per case, the expected tile kind, complete destination
bytes, bounded peak bytes, exact tile count, and exact full-destination
equality after every replay.

Unit tests use a reduced shape mode exposed only as a private test fixture
builder. They assert schema/accounting, all five kinds, exact destination
verification, invalid inputs, and CLI persistence. They do not assert absolute
timing or timing order.

## Evidence Artifact

Persist the default matrix to:

```text
experiments/qwen35_hybrid_state/20260727-five-binding-synthetic-tile-replay.json
```

## Interpretation

Allowed:

- compare planner-driven slice/copy overhead across the five grammar classes;
- identify whether 4 MiB remains call-heavy and whether 8-16 MiB remains near
  the diminishing-return region on this host;
- use the result to retain or revise synthetic policy constants.

Forbidden:

- real checkpoint load latency or optimal tile budget;
- cold-cache, disk, multi-shard, RSS, allocator, or page-cache conclusions;
- GPU, inference, KV-cache, compression, memory, accuracy, or quality claims.

## Allowed Conclusion

After this gate passes:

> On this host, bounded synthetic representatives of all five Qwen3.5 tile
> grammars replayed through the actual planner slices with exact destination
> correctness, providing grammar-sensitive calibration evidence for the
> 8-16 MiB policy region.

