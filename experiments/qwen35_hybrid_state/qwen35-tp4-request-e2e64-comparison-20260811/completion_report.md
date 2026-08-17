# Qwen3.5 TP4 Request-Level E2E64 Comparison

## Result

Classification: `MIXED`

The r631 decode phase-split implementation improves request latency, but it
does not produce a uniform end-to-end throughput win across both benchmark
policies.

- `recompute`: case makespan improves `8.7896%`; request and output-token
  throughput improve `9.6366%`.
- `exact_restore`: case makespan regresses `2.6156%`; request and output-token
  throughput regress `2.5490%`.
- Pooled request p50 E2E improves in both policies:
  `4.4865%` for recompute and `2.5943%` for exact restore.
- All 48 warmup-plus-measured request pairs preserve exact 64-token output
  parity between r620 and r631.

Therefore the accurate conclusion is:

> The decode optimization translates into a modest request-latency
> improvement and a material recompute-throughput improvement, but the
> exact-restore case makespan remains dominated by request-level variance and
> does not improve in this run. There is no policy-independent E2E throughput
> pass.

## Controlled Workload

Both sources used:

- workload: `w2_long_reuse`
- requests per case: `4`
- generated tokens per request: `64`
- warmup: r0 recompute and exact restore
- measured: r0-r4 recompute and exact restore
- total workers per source: `12`
- GPUs: `2,4,5,6`
- resource policy: shared and non-exclusive

The generated commands retain `--profile` and contain neither
`--generated-tokens-override` nor `--decode-internal-profile`.

Source identities:

```text
r620 source tree:
a26c543e79a9d4927fd0451d4a287363a677568a1daefe65a2a234a22f5997aa

r620 source tar:
5c39d91203d6c75a487936161bb1bb62e5487b67d4648ea2d464f84be85cd50e

r631 source tree:
6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837

r631 source tar:
f791f27e807e602f889345d301b72035dcd4a93d55a32adf51fd5eb3eaefb79c
```

## Case-Level E2E and Throughput

Case makespan is the maximum `e2e_ns` among the four requests:

```text
request throughput = 4 requests / case makespan
output throughput  = 256 output tokens / case makespan
```

| Policy | Metric | r620 | r631 | Change |
|---|---:|---:|---:|---:|
| recompute | median makespan | 6.368136 s | 5.808404 s | 8.7896% faster |
| recompute | request throughput | 0.628127 req/s | 0.688657 req/s | 9.6366% higher |
| recompute | output throughput | 40.200146 tok/s | 44.074066 tok/s | 9.6366% higher |
| exact restore | median makespan | 6.250599 s | 6.414092 s | 2.6156% slower |
| exact restore | request throughput | 0.639939 req/s | 0.623627 req/s | 2.5490% lower |
| exact restore | output throughput | 40.956077 tok/s | 39.912116 tok/s | 2.5490% lower |

Makespan population standard deviation:

```text
                         r620       r631
recompute              0.563913 s  0.485703 s
exact restore          1.391410 s  0.739813 s
```

Paired per-repetition makespan improvements:

```text
recompute:      -15.4636%, -13.1760%, +18.8977%, +15.9738%, +13.5969%
exact restore:   +9.5540%, +38.9466%,  -9.9083%, -14.5880%, -10.6380%
```

The exact-restore ratio-of-medians regression is smaller than the paired
median regression because the r620 r1 case contains a 9.421-second
straggler. The classification remains `MIXED`; neither aggregation supports
a uniform two-policy throughput pass.

## Request-Level Latency

The primary request-level figures below pool all 20 measured requests per
policy before taking p50. Decode latency is the sum of the 63
`decode_step_ns` values for each request.

| Policy | Metric | r620 p50 | r631 p50 | Change |
|---|---:|---:|---:|---:|
| recompute | request E2E | 5.590756 s | 5.339925 s | 4.4865% faster |
| recompute | TTFT | 0.678915 s | 0.690861 s | 1.7596% slower |
| recompute | decode latency | 4.817636 s | 4.647632 s | 3.5288% faster |
| exact restore | request E2E | 5.650202 s | 5.503620 s | 2.5943% faster |
| exact restore | TTFT | 0.695027 s | 0.678498 s | 2.3782% faster |
| exact restore | decode latency | 4.887756 s | 4.756247 s | 2.6906% faster |

The median-of-four-requests, then median-of-five-cases aggregation is also
preserved in `comparison.json`. It shows larger E2E improvements
(`10.1426%` recompute and `7.9331%` exact restore), but it is not presented
as the pooled single-request p50.

## Correctness and Resource Gates

Both attempts completed:

```text
workers:                    12 / 12
case directories:           12 / 12
measured cases:              10 / 10
measured request rows:       40 / 40
tokens per request:          64
decode timings per request:  63
RUN_COMPLETE:                present
attempt receipt:             DOWNLOADED
cleanup receipt:             CLEAN
```

Cross-source token parity:

```text
warmup plus measured request pairs: 48 / 48
measured request pairs:             40 / 40
classification:                     PASS
```

GPU guard evidence:

```text
                         r620          r631
READY guards            12 / 12       12 / 12
selected GPUs           2,4,5,6       2,4,5,6
minimum free bytes      55,654,219,776 56,982,765,568
maximum utilization    0%             0%
exclusive               false          false
```

Three consecutive ancestry-aware `/proc` checks found zero processes for
either exact attempt tag after cleanup.

## Verification

```text
focused runner/comparison tests: 13 passed
runner/comparison py_compile:    PASS
canonical aggregate cross-check: PASS
raw makespan/throughput recalc:  PASS
JSON parse and schema checks:    PASS
```

The existing `aggregate_profile(..., generated_tokens=64)` independently
reproduced the comparison's median makespan, TTFT, and decode values for
both sources and both policies.

## Artifacts

```text
experiments/qwen35_hybrid_state/
  qwen35-tp4-request-e2e64-r620-baseline-attempt001/
  qwen35-tp4-request-e2e64-r631-candidate-attempt001/
  qwen35-tp4-request-e2e64-comparison-20260811/
    comparison.json
    baseline_canonical_aggregate.json
    candidate_canonical_aggregate.json
    completion_audit.json
    completion_report.md
    dry_run.json
```

## Next Optimization Boundary

The decode hot path is genuinely faster, but request makespan still depends
on per-request stragglers and policy-specific TTFT behavior. The next useful
work is to profile the slowest request within each four-request case and
reduce or overlap the remaining row-parallel AllReduce schedule. Further
subdivision of restore timing is not the highest-value next step.

