# TP4 Qwen3 Batch-4 Primed Target-Then-Learned Telemetry

## Purpose

This bundle is the same-policy-primed `target,learned` companion to the
unprimed r3/r4 policy-order campaigns. Before each measured worker, the runner
executes an isolated worker with the same policy, two warmups, and one
discarded measured request batch.

Prime JSON and logs are retained separately and are excluded from the timing
and telemetry artifacts:

```text
prime-workers/target-prime-b4.json
prime-workers/learned-prime-b4.json
prime-logs/target-prime-b4.log
prime-logs/learned-prime-b4.log
```

## Workload

```text
target model:          Qwen3 1.7B
draft model:           learned Qwen3 draft
tensor parallel size:  4
GPUs:                  3,4,6,7
batch size:            4
prompt tokens:         256
output tokens:         16
max proposal tokens:   4
temperature:           0
prime per policy:      2 warmups + 1 measured
measured per policy:   2 warmups + 8 measured
policy order:          target,learned
```

The effective controls are retained in `prime-each-policy.txt`,
`policy-order.txt`, and `command.txt`.

## Verification

```text
remote tests:                    126 passed in 4.19s
campaign exit code:              0
remote status:                   0
remote timing verifier:          PASS
remote telemetry verifier:       PASS
local timing verifier:           PASS
local telemetry verifier:        PASS
manifest:                        PASS
exact greedy parity:             true
timing classification:           UNSTABLE
telemetry classification:        RUNTIME_VARIANCE_SUSPECTED
classification reasons:          []
sampler stderr:                  0 bytes
```

All six timing source hashes and all five telemetry source hashes are
identical across unprimed r3/r4 and primed r5/r6. The priming runner control is
not part of either verifier-bound source set.

## Primed Measurements

```text
target first:
  median TTFT:                   0.251904 s
  median TPOT:                   0.238367 s
  median E2E:                    3.811664 s
  median throughput:            16.802941 tok/s
  E2E range / median:           55.73%
  E2E half drift:               29.34%

learned second:
  median TTFT:                   0.190710 s
  median TPOT:                   0.330360 s
  median E2E:                    5.157472 s
  median throughput:            11.232830 tok/s
  proposal-forward median:       3011.710241 ms
  E2E range / median:           95.20%
  E2E half drift:               64.99%
```

The isolated prime requests were:

```text
target prime E2E:                2.101977 s
learned prime E2E:               7.260708 s
learned prime proposal-forward:  3442.018993 ms
```

Prime timings are retained as control evidence only. They are not included in
measured medians or stationarity classification.

## Four-Campaign Comparison

Do not merge medians across campaigns.

```text
target:
  r3 unprimed first E2E:          5.522148 s
  r4 unprimed second E2E:         3.958820 s
  r5 primed first E2E:            3.811664 s
  r6 primed second E2E:           3.932132 s

  primed second versus first:
    E2E:                         +3.16%
    TPOT:                        +2.17%

learned:
  r4 unprimed first E2E:         11.641256 s
  r3 unprimed second E2E:         9.840383 s
  r6 primed first E2E:            6.676585 s
  r5 primed second E2E:           5.157472 s

  primed second versus first:
    E2E:                        -22.75%
    TPOT:                       -22.11%
    proposal-forward:           -15.09%
```

Same-policy priming moves both policies toward lower absolute latency, but it
does not remove the learned-policy position effect. Target is near-converged
between the two primed orders; learned remains materially faster when run
second. The unique design classification is therefore:

```text
POSITION_EFFECT_REMAINS
TARGET_PRIMED_ORDER_EFFECT=NEAR_CONVERGED
LEARNED_PRIMED_ORDER_EFFECT=REMAINS
```

The absolute priming changes do not establish whether JIT, model page cache,
allocator state, CUDA runtime state, or another process boundary is the
cause.

## GPU and Host Evidence

```text
r5 target:
  samples per repeat/GPU: 6..12
  total samples:          296

r5 learned:
  samples per repeat/GPU: 10..22
  total samples:          500

SM clock:                 1410 MHz only
memory clock:             1512 MHz only
P-state:                  P0 only
measured throttle mask:   0 only
temperature:              38..44 C
```

The selected-GPU process inventory before and after the campaign retains only
the pre-existing GPU-7 PID `703088` `python3` service on GPUs 3,4,6,7. No
selected-GPU process was terminated.

The six host logs are hash-bound retention, not semantically parsed or
per-repeat aligned verifier authority. An exploratory `vmstat` aggregation is
approximately `us=59% sy=39% id=2% wa=0 st=0`; host contention remains
possible.

## Claim Boundary

Established:

```text
same-policy priming executed immediately before each measured policy
prime artifacts retained separately from measured artifacts
exact greedy parity
remote and local timing/telemetry verification
stable sampled GPU clocks, P-state, throttle state, and temperature
target primed order medians near-converged
learned primed second-position speedup remains material
POSITION_EFFECT_REMAINS
```

Not established:

```text
specific JIT, page-cache, CUDA, allocator, TP collective, or runtime cause
host contention excluded
stable performance baseline
performance promotion
4K or long-context performance
second learned model structure
Phase 1 completion
```

## Next Gate

Do not select CUDA Graph, TP collective, or metadata optimization from these
campaigns. The next controlled experiment should target the residual learned
first-position effect with per-repeat host semantic alignment or a more
specific process/JIT boundary control.

```text
SPECIFIC_RUNTIME_ROOT_CAUSE=NOT_ESTABLISHED
HOST_CONTENTION=NOT_EXCLUDED
STABLE_PERFORMANCE_BASELINE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```
