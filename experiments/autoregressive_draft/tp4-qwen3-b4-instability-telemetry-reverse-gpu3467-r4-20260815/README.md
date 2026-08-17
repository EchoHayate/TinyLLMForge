# TP4 Qwen3 Batch-4 Reverse-Order Instability Telemetry

## Purpose

This bundle is the reversed-policy companion to:

```text
experiments/autoregressive_draft/
  tp4-qwen3-b4-instability-telemetry-gpu3467-r3-20260815
```

The r3 campaign ran `target,learned`. This r4 campaign ran
`learned,target`. The two campaign medians are intentionally not merged.

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
warmup runs:           2 per policy
measured runs:         8 per policy
policy order:          learned,target
```

The actual order is retained in `policy-order.txt` and `command.txt`.

## Verification

```text
remote tests:                    125 passed in 4.41s
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

The local timing receipt verifies six source files. The local telemetry
receipt verifies five source files and six retained host logs.

All six timing source hashes and all five telemetry source hashes are
identical between r3 and r4. The runner and its contract test changed to add
validated policy-order control, but neither is part of the timing or telemetry
source-hash sets.

## Reverse-Order Result

Both orders remain unstable.

```text
r3 target first:
  median TPOT:               0.344989 s
  median E2E:                5.522148 s
  median throughput:        11.589696 tok/s
  E2E range / median:       31.68%

r4 target second:
  median TPOT:               0.245206 s
  median E2E:                3.958820 s
  median throughput:        16.170753 tok/s
  E2E range / median:       40.99%

target second versus first:
  TPOT:                     -28.92%
  E2E:                      -28.31%
  throughput:               +39.53%
```

```text
r4 learned first:
  median TPOT:               0.746706 s
  median E2E:               11.641256 s
  median throughput:         5.027431 tok/s
  proposal-forward median:   6564.098539 ms
  E2E range / median:       65.90%

r3 learned second:
  median TPOT:               0.631957 s
  median E2E:                9.840383 s
  median throughput:         5.843038 tok/s
  proposal-forward median:   5323.351506 ms
  E2E range / median:       32.29%

learned second versus first:
  TPOT:                     -15.37%
  E2E:                      -15.47%
  throughput:               +16.22%
  proposal-forward:         -18.90%
```

The same direction appears for both policies: the policy executed second is
faster than the same policy executed first in the companion campaign. The
learned proposal-forward path follows the same direction.

This is evidence for a policy-position or process-cadence effect. It is not
proof of a specific runtime root cause because there is only one campaign per
order and the retained host logs are hash-bound but not semantically parsed
or interval-aligned by the verifier.

## GPU Telemetry

```text
r3 learned:
  samples per repeat/GPU: 19..27
  total samples:          752

r3 target:
  samples per repeat/GPU: 9..15
  total samples:          388

r4 learned:
  samples per repeat/GPU: 11..41
  total samples:          860

r4 target:
  samples per repeat/GPU: 7..10
  total samples:          288
```

Across both campaigns and both policies:

```text
SM clock:            1410 MHz only
memory clock:        1512 MHz only
P-state:             P0 only
throttle mask:       0 only
temperature:         38..45 C
minimum coverage:    at least 7 samples per repeat/GPU
```

The selected-GPU process inventory before and after each campaign retains the
existing GPU-7 `python3` service. No selected-GPU process was terminated.

An exploratory, non-authoritative `vmstat` aggregation found median CPU idle
near 2%, I/O wait 0%, and steal 0% for every policy in both campaigns. The
host was highly loaded, but the current verifier does not prove whether
per-repeat host contention correlates with the timing outliers.

## Claim Boundary

Established:

```text
reverse-order campaign completed with exact parity
remote and local timing/telemetry verification passed
all verifier-bound runtime and telemetry sources match r3
sampled clocks, P-state, throttle state, and temperature remain stable
both policy orders remain timing-unstable
both policies are faster when executed second
```

Not established:

```text
host contention excluded
specific CUDA, TP collective, allocator, JIT, page-cache, or runtime root cause
stable performance baseline
performance promotion
4K or long-context performance
second learned model structure
Phase 1 completion
```

## Next Gate

Do not select a CUDA Graph, TP-authority, or metadata optimization from these
two campaigns. The next controlled gate should prime each policy with an
isolated, discarded same-policy worker process before its measured worker,
retain the prime logs separately, and rerun both policy orders. This tests
whether the second-process speedup is a repeatable process/JIT/page-cache
cadence effect rather than a policy-specific runtime optimization target.

