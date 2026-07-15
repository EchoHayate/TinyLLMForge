# SAM Drafter Canonical Gate

## Decision

- Decision: `INCOMPLETE`
- Reasons: (0, 'natural_prose'):sam_fixed_k16:output_mismatch, (0, 'natural_prose'):sam_match_aware:output_mismatch, (0, 'natural_prose'):ngram_fixed_k4:output_mismatch, (0, 'structured_code_like'):sam_fixed_k16:output_mismatch, (0, 'structured_code_like'):ngram_adaptive:output_mismatch, (0, 'structured_code_like'):ngram_fixed_k4:output_mismatch

## Environment

- Host: `sitian@10.232.195.203`
- Source commit: `aba3a17ba417c9e1cacbee226f7b87ed5e1f47c6`
- Source dirty: `False`
- Model: `Qwen3-0___6B:d612d47d4a84d410`

## Completeness

- Rows: `25/25`
- Correctness pass: `False`
- Trace reconciliation pass: `True`
- Policy exercise pass: `True`

## Median Throughput

- `baseline`: `31.364755` tok/s
- `ngram_fixed_k4`: `36.538560` tok/s
- `ngram_adaptive`: `33.158649` tok/s
- `sam_fixed_k16`: `32.397456` tok/s
- `sam_match_aware`: `57.339012` tok/s

## Paired Metrics

- SAM vs baseline: `0.7992583744152906`
- SAM vs n-gram K4: `0.30749802288440087`
- Verify-attempt reduction: `-0.4444444444444444`
- Draft-waste reduction: `-2.888888888888889`

## Critical Prompts

- `natural_prose`: `0.026962506917506568`
- `structured_code_like`: `0.8281352152283568`
- `repeated_long_context`: `1.757506046746982`
- `transition_heavy`: `0.7992583744152906`
- `prompt_copy_retrieval`: `-0.034491184842733946`

## Policy Exercise

- Failures: `[]`

## SAM CPU Overhead

- `sam_build_ms`: `6.315999` ms
- `sam_extension_ms`: `6.975610` ms
- `sam_lookup_ms`: `14.791347` ms

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

- Repair incomplete evidence and rerun without changing thresholds.
