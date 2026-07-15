# Adaptive N-Gram Speculation Gate

- Decision: **NO_GO**
- Reasons: `adaptive_vs_fixed_gate_failed`
- Rows: `140/140`
- Source: `08b122daedc8ab531a5d301f0b5a82b5cb1997e5` (dirty=False)
- Model: `Qwen3-0___6B` at `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`
- Host/Python: `sitian@10.232.195.203` / `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`

## Aggregate Throughput

| Policy | Median tok/s |
|---|---:|
| baseline | 33.815941 |
| fixed_k1 | 32.979915 |
| fixed_k2 | 32.925531 |
| fixed_k4 | 37.962906 |
| adaptive | 37.574839 |

## Per-Prompt Throughput

| Prompt | Baseline | Fixed K1 | Fixed K2 | Fixed K4 | Adaptive |
|---|---:|---:|---:|---:|---:|
| natural_prose | 33.956405 | 33.435853 | 33.271282 | 31.579000 | 32.969817 |
| structured_mixed | 33.719898 | 33.486046 | 33.077964 | 32.191697 | 33.160628 |
| repeated_long_context | 33.935897 | 33.026123 | 32.911202 | 46.905002 | 46.608547 |
| transition_heavy | 33.996681 | 32.474497 | 32.921986 | 41.950054 | 39.137848 |

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
