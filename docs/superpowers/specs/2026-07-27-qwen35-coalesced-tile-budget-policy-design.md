# Qwen3.5 Coalesced Tile Budget Policy Design

## Status

Approved under the standing inline-execution direction. This remains a
CPU/static and temporary-payload gate. It does not open the real checkpoint,
run GPU work, or connect production runtime.

## Goal

Select the smallest power-of-two tile budget that satisfies both:

```text
peak_tile_bytes <= max_tile_bytes
tile_count <= max_tile_count
```

for an exact `Qwen35CheckpointBindingPlan`, then use the selected immutable tile
plan to load one fresh unpublished temporary-shard candidate without rebuilding
the model or invoking the candidate factory twice.

## Why a Policy Is Required

The completed tile planner proves a very low correctness bound, but the real
plan produces too many operations at that bound:

```text
TP=1, 12 KiB: 306,449 tiles
TP=2,  6 KiB: 426,723 tiles per rank
```

Real static budget scan:

```text
budget   TP=1 tiles   TP=2 tiles
64 KiB      58,169       29,169
256 KiB     14,561        7,365
1 MiB        3,779        1,986
2 MiB        1,986        1,096
4 MiB        1,096          651
8 MiB          651          488
16 MiB         488          386
32 MiB         386          371
64 MiB         371          363
```

The marginal gain after 16 MiB is small relative to the larger explicit
materialization bound. Therefore the policy target is:

```text
max_tile_count = 512
max_tile_bytes = 16 MiB
```

The policy derives 16 MiB for TP=1 and 8 MiB for each TP=2 rank from
constraints; it does not hard-code those results.

## Public Policy API

Create:

```text
tinyvllm/models/qwen35_checkpoint_tile_policy.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35CheckpointTileBudgetEvaluation:
    max_tile_bytes: int
    tile_count: int
    peak_tile_bytes: int


@dataclass(frozen=True)
class Qwen35CheckpointTileBudgetDecision:
    tile_plan: Qwen35CheckpointTilePlan
    selected_max_tile_bytes: int
    max_tile_count: int
    evaluations: tuple[Qwen35CheckpointTileBudgetEvaluation, ...]


def select_qwen35_checkpoint_tile_budget(
    binding_plan: Qwen35CheckpointBindingPlan,
    *,
    max_tile_bytes: int,
    max_tile_count: int,
) -> Qwen35CheckpointTileBudgetDecision:
    ...
```

## Candidate Budgets

Validate positive non-boolean integer inputs.

Find the smallest indivisible tile-unit byte count implied by the complete
binding plan by attempting the existing planner. Candidate budgets are powers
of two:

```text
next_power_of_two(minimum feasible bytes)
then double until max_tile_bytes
```

If `max_tile_bytes` is not a power of two, include it as the final candidate so
the caller's exact cap is not silently reduced.

For every candidate:

1. build an exact immutable tile plan;
2. record candidate bytes, tile count, and actual peak tile bytes;
3. return immediately on the first `tile_count <= max_tile_count`.

Reject when no candidate satisfies both caps. The error reports the final tile
count and both caller caps.

The policy is deterministic and monotonic over one immutable binding plan.
It does not mutate the plan, model, destinations, or pool and performs no file
I/O.

## Policy-Driven Load API

Extend:

```text
tinyvllm/models/qwen35_checkpoint_tiled_loading.py
```

with:

```python
@dataclass(frozen=True)
class Qwen35PolicyTiledLoadedCheckpointCandidate:
    loaded: Qwen35TiledLoadedCheckpointCandidate
    decision: Qwen35CheckpointTileBudgetDecision


def load_qwen35_fresh_checkpoint_candidate_with_tile_policy(
    candidate_factory,
    checkpoint_dir,
    *,
    max_tile_bytes: int,
    max_tile_count: int,
) -> Qwen35PolicyTiledLoadedCheckpointCandidate:
    ...
```

Refactor the existing tiled loader into:

```text
validate/invoke factory once
build or receive exact tile plan
validate paths
load tiles
build owner
```

The policy-driven API invokes the candidate factory exactly once, selects a
plan from that exact binding plan, and passes the selected plan into the shared
loading implementation. It must not call the existing public tiled loader in a
way that invokes the factory a second time.

The returned nested `loaded.tile_plan` must be the exact same object as
`decision.tile_plan`.

## Test Strategy

### Pure policy

Use the two-layer fixture and synthetic caps to cover:

- exact smallest satisfying candidate;
- exact non-power-of-two final cap;
- invalid inputs;
- no satisfying candidate;
- deterministic repeated decisions;
- unchanged binding plan and destinations.

### Real static policy

Using real metadata and meta graphs:

```text
max_tile_bytes=16 MiB
max_tile_count=512
```

must select:

```text
TP=1: 16 MiB, 488 tiles
TP=2:  8 MiB, 488 tiles per rank
```

With `max_tile_bytes=8 MiB`, TP=1 must fail the 512 cap while TP=2 still
selects 8 MiB and 488 tiles. No safetensors payload may open.

### Policy-driven temporary load

- factory invoked exactly once;
- selected plan identity preserved;
- exact TP=1/2 destination values;
- get-slice-only and custom-loader bypass properties retained;
- selected tile count and peak bytes match runtime stats;
- policy failure occurs before any shard open;
- load failure still discards the candidate and closes handles.

## Non-Goals

This gate does not:

- benchmark actual safetensors load latency;
- claim 512 is globally optimal;
- open the real payload;
- prove RSS, page-cache, or allocator peaks;
- publish the tiled candidate;
- run GPU, forward, token/logit, or performance gates;
- connect ModelRunner, Engine, or Scheduler;
- establish inference speed, KV-cache, compression, memory, or quality gains.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge can deterministically choose the smallest power-of-two
> Qwen3.5 tile budget satisfying explicit peak-byte and tile-count caps, and
> use that exact plan to load one fresh temporary-shard CPU candidate with a
> single factory invocation.

