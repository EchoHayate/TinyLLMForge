# Adaptive N-Gram Speculation Gate

- Decision: **NO_GO**
- Reasons: `adaptive_vs_baseline_gate_failed, adaptive_vs_fixed_gate_failed, natural_or_transition_regression`
- Rows: `20/20`
- Source: `198cede8a3b0d201588ceb547208ada111aa77b7` (dirty=True)
- Model: `Qwen3-0___6B` at `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Host/Python: `sitian@10.232.195.203` / `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`

## Aggregate Throughput

| Policy | Median tok/s |
|---|---:|
| baseline | 34.593668 |
| fixed_k1 | 33.066012 |
| fixed_k2 | 29.439458 |
| fixed_k4 | 29.390634 |
| adaptive | 29.620451 |

## Per-Prompt Throughput

| Prompt | Baseline | Fixed K1 | Fixed K2 | Fixed K4 | Adaptive |
|---|---:|---:|---:|---:|---:|
| natural_prose | 35.315756 | 34.695717 | 34.728219 | 32.978326 | 33.538371 |
| structured_mixed | 34.802577 | 35.102501 | 33.197203 | 33.210235 | 32.105402 |
| repeated_long_context | 35.215140 | 30.130601 | 25.906034 | 26.092968 | 26.934554 |
| transition_heavy | 33.172559 | 33.787432 | 27.470392 | 28.060708 | 28.142727 |

## Audits

- Correctness: `True`
- Trajectory replay: `True`
- Adaptive exercise: `True`
- Selected levels: `[1, 2, 4]`
- Transition reasons: `['full_accept_streak', 'hold', 'promote', 'weak_acceptance', 'zero_accept']`

## Fixed Thresholds

```json
{
  "adaptive_near_best_fixed_min": -0.01,
  "adaptive_vs_baseline_min": 0.05,
  "adaptive_vs_best_fixed_min": 0.02,
  "adaptive_waste_reduction_vs_k4_min": 0.2,
  "adaptive_zero_cost_reduction_vs_k4_min": 0.15,
  "natural_transition_ratio_min": 0.95
}
```

## Claim Boundaries

This decision covers only greedy single-sequence Qwen3-0.6B runs on the recorded host and prompt bank. It does not establish ragged batched verification correctness, production batch throughput, queueing-tail latency, memory-capacity reduction, or transfer to other models.

A GO should be followed by a separate ragged batched target-verify and load-aware K=0..N design. A NO_GO should retain the correctness and measurement machinery while preferring the best measured fixed policy only in its validated regime or moving to a higher-quality draft source.
