# Qwen3.5 Synthetic Tile-Copy Calibration Design

## Status

Approved under the standing inline-execution direction. This is a local
CPU-only microbenchmark over a generated temporary safetensors tensor. It must
not read a model checkpoint, run GPU work, or claim end-to-end model loading or
inference performance.

## Goal

Measure the warm-cache CPU cost of:

```text
PySafeSlice tile materialization + copy into a preallocated destination
```

across bounded tile sizes, compare it with one full-tensor
`get_tensor()+copy_` baseline, verify complete payload correctness, and persist
machine-readable evidence that can calibrate tile-policy constants.

## Methodology

Create one contiguous BF16 tensor:

```text
shape = [16384, 2048]
payload = 64 MiB
```

Each row is filled with its row index. Save it to a temporary safetensors file.
This shape matches the real embedding row width of `2048`, while remaining
small enough for repeatable local CPU runs.

For tile budgets:

```text
1, 2, 4, 8, 16, 32, 64 MiB
```

derive complete-row tiles and, for every timed repeat:

1. allocate one full pre-sized BF16 destination;
2. open the temporary shard on CPU;
3. call `get_slice()` once;
4. materialize every row tile and `copy_` it to the exact destination rows;
5. close the handle;
6. compute a full FP32 destination checksum outside the timed region;
7. require exact checksum equality.

Run one untimed warm-up plus five timed repeats per budget.

Baseline repeats perform:

```python
destination.copy_(handle.get_tensor("synthetic.weight"))
```

with the same allocation, handle, timing boundary, and checksum.

## Public Harness

Create:

```text
tools/benchmark_qwen35_safetensors_tile_copy.py
```

with pure result records and:

```python
def run_qwen35_synthetic_tile_copy_calibration(
    *,
    rows: int = 16384,
    columns: int = 2048,
    tile_mib: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    repeats: int = 5,
) -> dict:
    ...
```

CLI options permit smaller test runs and optional:

```text
--output-json PATH
```

The JSON result contains:

```text
schema_version
environment
tensor
baseline
tile_results
interpretation_limits
```

Each timing result records all raw seconds plus median, minimum, maximum,
call count, requested bytes, actual peak tile bytes, and ratio to baseline
median.

## Correctness Contract

The harness must reject invalid shape, repeats, or tile budgets. Every tile
budget must fit at least one complete row.

For each repeat:

- every destination element is written exactly once;
- checksum equals the source checksum;
- call count equals ceiling(rows / rows_per_tile);
- actual tile bytes never exceed requested bytes;
- the final short tile is handled exactly.

Unit tests use a tiny temporary tensor and do not assert timing order or exact
seconds.

## Evidence Artifact

Persist the full default run to:

```text
experiments/qwen35_hybrid_state/20260727-synthetic-tile-copy-calibration.json
```

The artifact is local machine evidence. It must include exact Python, PyTorch,
safetensors, platform, CPU, and timestamp fields.

## Interpretation

Allowed analysis:

- quantify synthetic warm-cache call/copy overhead;
- identify where larger tiles provide diminishing returns;
- compare candidate policy budgets on this one host;
- use results as one input to a future real-load gate.

Forbidden conclusions:

- real 4.5 GB checkpoint load latency;
- cold-cache or disk throughput;
- total RSS or page-cache peak;
- GPU load/forward latency;
- inference speed, KV-cache, compression, memory, or quality benefit.

## Allowed Conclusion

After this gate passes:

> On this host and a 64 MiB synthetic BF16 tensor with 2048-column rows, the
> calibration quantifies warm-cache `get_slice()+copy_` overhead across
> bounded tile sizes while proving complete destination correctness.

