# Qwen3 Independent-Draft TP4 Timing Pilot

## Classification

```text
artifact status: PASS
classification: PILOT_ONLY
direction: NEGATIVE
batch 1 direction: REGRESSED
batch 4 direction: REGRESSED
remote verifier: PASS
local verifier: PASS
manifest: PASS after refresh
```

`PASS` means the source-bound schema-v2 artifact is internally valid. It does
not mean the learned runtime improved performance, and it does not make the
high-variance batch-4 cell a stable optimization baseline.

## Frozen Workload

```text
target: Qwen3-1.7B
draft: independent Qwen3 draft checkpoint
tensor parallel size: 4
GPUs: 3,4,6,7
proposal allocator: direct
prompt tokens: 256
output tokens: 16
batch sizes: 1 and 4
temperature: 0
max proposal tokens: 4
warmup runs per cell: 1
measured runs per cell: 3
```

Proposal-KV capacity remains the exact workload-derived bound:

```text
batch 1: 1 * (256 + 16 + 4) = 276 slots
batch 4: 4 * (256 + 16 + 4) = 1104 slots
```

## Median Results

| Batch | Metric | Target | Learned | Learned vs target |
|---|---:|---:|---:|---:|
| 1 | TTFT | 0.301981 s | 0.264154 s | -12.53% |
| 1 | TPOT | 0.256894 s | 0.374803 s | +45.90% |
| 1 | E2E | 4.146951 s | 5.886194 s | +41.94% |
| 1 | output throughput | 3.858257 tok/s | 2.718225 tok/s | -29.55% |
| 4 | TTFT | 0.265197 s | 0.290367 s | +9.49% |
| 4 | TPOT | 0.281634 s | 0.535204 s | +90.03% |
| 4 | E2E | 4.489711 s | 8.287627 s | +84.59% |
| 4 | output throughput | 14.254814 tok/s | 6.885352 tok/s | -51.70% |

Exact output-token parity holds for all measured target/learned repeats.
Learned acceptance is unchanged from the earlier pilot:

```text
batch 1: 15 / 15 accepted, 100.00%, four speculative steps
batch 4: 53 / 72 accepted, 73.61%, six speculative steps
```

## Runtime Stage Timing

The seven runtime stages are mutually additive within each learned run.
Executor timing is a nested view inside `first_target_batch_ms`; the executor
three-stage numbers must not be added again.

Batch-1 median:

| Runtime stage | Median | Share of median E2E |
|---|---:|---:|
| first target plus proposal batch | 4019.264 ms | 68.28% |
| target verify tail batch | 935.312 ms | 15.89% |
| transactional commit metadata | 390.031 ms | 6.63% |
| reserve blocks | 1.289 ms | 0.02% |
| KV materialize | 0.092 ms | less than 0.01% |
| accept/sample | 0.055 ms | less than 0.01% |

Nested batch-1 executor median:

```text
prompt bootstrap:   350.626 ms
proposal forward:  2693.178 ms
proposal finalize:  312.353 ms
```

Batch-4 median:

| Runtime stage | Median | Share of median E2E |
|---|---:|---:|
| first target plus proposal batch | 6402.739 ms | 68.88% |
| target verify tail batch | 1288.624 ms | 13.86% |
| transactional commit metadata | 962.029 ms | 10.35% |
| reserve blocks | 4.706 ms | 0.05% |
| KV materialize | 0.217 ms | less than 0.01% |
| accept/sample | 0.146 ms | less than 0.01% |

Nested batch-4 executor median:

```text
prompt bootstrap:   335.970 ms
proposal forward:  4411.708 ms
proposal finalize:  529.206 ms
```

The dominant optimization target is therefore independent-draft
`proposal_forward`, not block reservation, KV materialization, or greedy
acceptance logic. Transaction finalization and publication are secondary but
still material, particularly at batch 4.

## Raw Variance Boundary

Batch-1 learned E2E is reasonably clustered:

```text
5929.507 ms
5886.194 ms
5744.654 ms
```

Batch-4 learned E2E rises monotonically:

```text
6082.036 ms
9295.094 ms
11607.905 ms
```

The same increase is visible on every TP rank in `proposal_forward` and in
the enclosing `first_target_batch_ms`. It is not a single observed rank-3
or GPU-7 timing outlier. GPU snapshots show the pre-existing GPU-7
`python3` service before and after the run; it was not terminated or changed.

This bundle establishes the negative direction and bottleneck family, but
the batch-4 medians must not be treated as a stable before/after optimization
baseline. An isolated repeated batch-4 diagnostic is required first.

## Memory and Movement

Median CUDA peak differences:

```text
batch 1 learned minus target:
  allocated: +287.992 MiB
  reserved:  +342.000 MiB

batch 4 learned minus target:
  allocated: +309.826 MiB
  reserved:  +270.000 MiB
```

Proposal-KV movement remains zero in every measured run:

```text
H2D bytes: 0
D2H bytes: 0
```

This is direct-allocator evidence only and is not an offload benefit.

## Validation

Fresh local validation before the pilot:

```text
performance/publication timing tests: 24 passed, 23 deselected
py_compile: PASS
bash -n: PASS
scoped git diff --check: PASS
```

Fresh remote validation against the uploaded timing source:

```text
full autoregressive draft executor: 75 passed
performance/publication timing tests: 24 passed, 23 deselected
py_compile: PASS
bash -n: PASS
```

One broader remote combination produced `121 passed, 1 failed`; the only
failure is the pre-existing
`test_partition_preserves_selected_and_suppressed_order` expectation drift.
The timing/publication selection and the full executor suite pass separately.

The first r4 launch failed during transient SSH banner exchange and retained
only `command.txt` plus `source.tar` in its separate local directory. This
successful run uses the unique `r4b` tag and does not overwrite that evidence.

## Next Optimization Step

Before changing algorithms, add source-bound substage evidence for:

```text
draft backend decode
TP greedy token selection
Proposal-KV transaction/bookkeeping
TP authority convergence
```

Then run a repeated isolated batch-4 diagnostic. If backend decode plus the
following synchronization remains dominant, the first implementation target
is an exact-shape CUDA Graph path for independent-draft decode. If metadata
or authority convergence dominates instead, optimize those paths before
graph capture.

## Claim Boundary

Established:

```text
schema-v2 TP4 timing artifact
seven additive runtime-stage totals
four-rank executor timing deltas
exact parity and acceptance evidence
negative performance direction
proposal_forward as the dominant bottleneck family
source-bound remote and local verification
```

Not established:

```text
stable batch-4 performance baseline
CUDA Graph speedup
4K, 16K, or 32K learned-drafter performance
Proposal-KV offload performance benefit
second learned model structure
Phase 1 promotion
```
