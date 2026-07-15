# SAM Drafter Canonical Gate

## Decision

- Decision: `NO_GO`
- Reasons: sam_vs_baseline_gate_failed, critical_prompt_regression:natural_prose, critical_prompt_regression:structured_code_like, critical_prompt_regression:repeated_long_context, critical_prompt_regression:prompt_copy_retrieval

## Environment

- Host: `sitian@10.232.195.203`
- Source commit: `b71e7ceabec211a7e1f5a4e2a942fac9a780c067`
- Source dirty: `False`
- Model: `Qwen3-0___6B:d612d47d4a84d410`

## Completeness

- Rows: `25/25`
- Correctness pass: `True`
- Trace reconciliation pass: `True`
- Policy exercise pass: `True`

## Median Throughput

- `baseline`: `32.517347` tok/s
- `ngram_fixed_k4`: `25.968690` tok/s
- `ngram_adaptive`: `25.705494` tok/s
- `sam_fixed_k16`: `28.339892` tok/s
- `sam_match_aware`: `28.627359` tok/s

## Paired Metrics

- SAM vs baseline: `-0.10721453066361797`
- SAM vs n-gram K4: `0.08438688372122316`
- Verify-attempt reduction: `0.25`
- Draft-waste reduction: `-2.1818181818181817`

## Critical Prompts

- `natural_prose`: `-0.10721453066361797`
- `structured_code_like`: `-0.14021308999320659`
- `repeated_long_context`: `-0.07400203838977637`
- `transition_heavy`: `0.059544011102800276`
- `prompt_copy_retrieval`: `-0.26751636734948503`

## Policy Exercise

- Failures: `[]`

## SAM CPU Overhead

- `sam_build_ms`: `5.052339` ms
- `sam_extension_ms`: `6.141484` ms
- `sam_lookup_ms`: `11.810966` ms

## Fixed Thresholds

```json
{
  "critical_prompt_speedup_min": -0.05,
  "draft_waste_reduction_min": 0.25,
  "sam_near_ngram_k4_min": -0.01,
  "sam_vs_baseline_min": 0.1,
  "sam_vs_ngram_k4_min": 0.03,
  "verify_attempt_reduction_min": 0.25
}
```

## Claim Boundaries

```json
{
  "greedy_only": true,
  "memory_reduction": false,
  "production_batch_throughput": false,
  "profiler_owned": true,
  "queue_tail_latency": false,
  "ragged_batched_verify": false,
  "single_sequence": true
}
```

## Next Direction

- Stop performance promotion and inspect failed evidence or thresholds.
