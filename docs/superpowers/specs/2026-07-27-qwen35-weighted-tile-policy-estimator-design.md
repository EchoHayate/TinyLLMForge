# Qwen3.5 Weighted Tile-Policy Estimator Design

## Status

Approved under the standing inline-execution direction. This gate combines
existing local synthetic calibration evidence with the real 320-binding static
tile plans. It must not read real checkpoint payloads, change production
policy selection, or claim measured real-load latency.

## Goal

Produce a deterministic, auditable latency proxy and Pareto comparison for
candidate Qwen3.5 tile budgets by combining:

1. per-kind synthetic replay medians from the completed five-binding artifact;
2. real static per-kind destination bytes and tile counts from the verified
   320-binding plan.

The result should clarify the predicted trade-off between 8, 16, and 32 MiB
without collapsing peak memory and latency proxy into an arbitrary weighted
scalar.

## Calibration Model

For each tile kind:

```text
axis0
axis1
segmented_axis0
squeeze_axis0
replicated
```

fit ordinary least squares over the four synthetic points:

```text
median_seconds = intercept_32mib + per_tile_seconds * tile_count
```

The synthetic case always copies 32 MiB locally. Require:

- exactly the expected four budgets `4, 8, 16, 32 MiB`;
- exact destination verification for every point;
- positive finite medians;
- non-negative fitted intercept and slope;
- finite residual metrics.

For a real static plan, estimate each kind as:

```text
estimated_kind_seconds =
    intercept_32mib * real_kind_bytes / 32 MiB
    + per_tile_seconds * real_kind_tile_count
```

Sum kinds to obtain a latency proxy. This is a model-derived score in
synthetic-host seconds, not a predicted or measured real checkpoint load time.

## Real Static Matrix

Use the existing real config/index/header and meta-model binding graph for:

```text
TP=1 rank0
TP=2 rank0
TP=2 rank1
```

Evaluate:

```text
4, 8, 16, 32 MiB
```

For every budget record:

- total tile count;
- actual peak tile bytes;
- per-kind binding count, destination bytes, and tile count;
- per-kind estimated proxy contribution;
- total estimated latency proxy;
- reduction versus 8 MiB;
- extra peak bytes versus 8 MiB.

Every safetensors payload open remains guarded and must be zero.

## Pareto Contract

A budget is dominated when another budget has:

```text
peak_tile_bytes <= candidate peak
estimated_latency_proxy <= candidate proxy
```

with at least one strict inequality.

Report the non-dominated frontier in ascending peak-byte order. Do not select
or mutate a production default. Explicitly compare:

```text
8 -> 16 MiB
16 -> 32 MiB
```

and classify each incremental proxy reduction as a percentage of the lower
budget score.

## Public Tool

Create:

```text
tools/estimate_qwen35_weighted_tile_policy.py
```

with pure helpers for:

```python
fit_qwen35_tile_kind_calibration(...)
estimate_qwen35_weighted_tile_policy(...)
```

The CLI consumes:

```text
--calibration-json PATH
--output-json PATH
```

and writes atomically.

## Evidence

Persist:

```text
experiments/qwen35_hybrid_state/20260727-weighted-tile-policy-estimator.json
```

## Interpretation Limits

Allowed:

- compare candidate budgets under one deterministic synthetic-derived proxy;
- expose which kinds contribute bytes, tiles, and modeled overhead;
- identify Pareto candidates and diminishing modeled returns;
- decide whether more real-load evidence is needed before changing policy.

Forbidden:

- call proxy seconds measured or predicted real load latency;
- claim RSS, page-cache, disk, GPU, inference, cache, compression, accuracy, or
  quality benefits;
- change `select_qwen35_checkpoint_tile_budget()` defaults or production
  wiring.

## Allowed Conclusion

After this gate passes:

> TinyLLMForge has an auditable static estimator that combines real Qwen3.5
> tile distributions with per-kind synthetic copy/call fits and reports the
> 8/16/32 MiB Pareto trade-off without reading model payloads or changing
> production policy.

