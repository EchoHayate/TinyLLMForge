# TP4 Qwen3 Independent-Draft Batch-4 Proposal-Forward Diagnostic

## Result

```text
status:          PASS
classification: UNSTABLE
exact parity:    true
decision:        INSTABILITY_INVESTIGATION
```

The campaign used TP4 on GPUs `3,4,6,7`, two warmups, eight measured
repeats, batch 4, 256 prompt tokens, 16 output tokens, temperature zero, and
`max_proposal_tokens=4`. The existing GPU-7 `python3` service remained
present with the same memory footprint before and after the run.

## Verification

```text
remote executor suite: 75 passed
remote verifier:       PASS
local verifier:        PASS
source files verified: 6
manifest:              PASS
```

The artifact retains every raw repeat, every rank's parent timing and six
proposal-forward details, critical-rank attribution, acceptance, memory,
output tokens, and exact parity evidence.

## Stability

```text
target E2E range / median:          66.60%
target TPOT range / median:         68.77%
learned E2E range / median:         62.67%
learned TPOT range / median:        65.96%
proposal-forward range / median:    64.98%
```

The learned medians remain negative versus target:

```text
TPOT:       +93.06%
E2E:        +82.05%
throughput: -50.64%
```

These directions are descriptive only. The variability is too large to use
this campaign as an optimization baseline.

## Proposal Attribution

Learned acceptance was invariant at `53 / 72 = 73.61%` for every measured
repeat. Peak allocated memory was invariant at `72455.08 MiB`.

```text
proposal-forward minimum: 3770.732 ms
proposal-forward median:  4648.731 ms
proposal-forward maximum: 6791.609 ms
proposal-forward/E2E correlation: 0.9466
```

Critical-rank median bucket shares:

```text
backend submit + selection collective + token readback: 94.45%
decode authority:                                    2.66%
setup + materialize/register + residual:             3.13%
```

The split between `backend_submit` and `selection_collective` changed sharply
between repeats. CUDA execution is asynchronous, so deferred completion wait
can move between those adjacent wall-clock boundaries. This is not individual
GPU-kernel duration evidence.

## Post-Run Instability Triage

Read-only recomputation over the retained rank rows narrows the instability:

```text
critical-rank sequence:                    1,0,3,0,3,1,1,1
median cross-rank proposal spread:         23.475 ms
proposal-forward median:                 4648.731 ms
critical submit+collective+readback /
  proposal-forward correlation:             0.9967
per-rank submit+collective /
  proposal-forward correlation:        0.9979-0.9989
critical submit / collective correlation:  -0.6761
```

The median cross-rank spread is only `0.51%` of the proposal-forward median.
All four ranks therefore slow down together; a single changing laggard or the
critical-rank selection does not explain the `64.98%` proposal range. The
negative submit/collective correlation confirms boundary migration, while
their sum tracks each rank's parent proposal time almost exactly.

Target and learned runs were executed as separate sequential campaigns.
Same-index target/learned correlations are therefore not paired environmental
evidence. The before/after GPU snapshots show no new process on GPUs
`3,4,6,7`, and the existing GPU-7 service retained its footprint, but those
snapshots do not capture clocks, throttling, utilization, power, temperature,
or host contention during individual repeats.

## r1 Correction

The first focused attempt incorrectly added six independently selected
per-key rank maxima and compared that sum with one rank's parent maximum.
Maxima from different ranks are not additive.

Schema v3 now preserves per-key `max_rank_ms` as non-additive evidence,
identifies the parent critical rank, records that same rank's
`critical_rank_ms`, and computes detail sum/residual only from that coherent
rank. A staggered-rank regression test covers this failure mode.

## Decision

Do **not** implement CUDA Graph, TP authority reduction, or metadata
optimization from this run. The design rule selects
`INSTABILITY_INVESTIGATION` because target and learned baselines are both
non-stationary.

The next diagnostic should record out-of-band GPU clocks, power, utilization,
temperature, and host load for every repeat, and pair or interleave
target/learned observations where practical. It must preserve exact parity
and must not add `torch.cuda.synchronize()` to the measured request path.

## Claim Boundary

Established:

- schema-v3 source-bound critical-rank proposal attribution;
- TP4 batch-4 exact greedy parity across eight repeats;
- invariant learned acceptance and peak allocation;
- proposal-forward/E2E correlation under this campaign;
- dual verifier and checksum-covered raw evidence.

Not established:

- a stable performance baseline;
- GPU kernel duration or CUDA Graph speedup;
- 4K/16K/32K performance;
- Proposal-KV offload benefit;
- a second learned model structure;
- Phase-1 promotion.
