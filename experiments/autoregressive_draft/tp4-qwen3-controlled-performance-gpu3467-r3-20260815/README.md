# Qwen3 Independent-Draft TP4 Controlled Performance Pilot

## Classification

```text
artifact status: PASS
classification: PILOT_ONLY
direction: NEGATIVE
batch 1 direction: REGRESSED
batch 4 direction: REGRESSED
remote verifier: PASS
local verifier: PASS
manifest: PASS
```

`PASS` means the source-bound measurement artifact is internally valid. It
does not mean the learned runtime improved performance.

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

Proposal-KV capacity is workload-derived rather than arbitrarily oversized:

```text
batch 1: 1 * (256 + 16 + 4) = 276 slots
batch 4: 4 * (256 + 16 + 4) = 1104 slots
```

## Median Results

| Batch | Metric | Target | Learned | Learned vs target |
|---|---:|---:|---:|---:|
| 1 | TTFT | 0.291134 s | 0.308903 s | +6.10% |
| 1 | TPOT | 0.277665 s | 0.353775 s | +27.41% |
| 1 | E2E | 4.456102 s | 5.615526 s | +26.02% |
| 1 | output throughput | 3.590582 tok/s | 2.849243 tok/s | -20.65% |
| 4 | TTFT | 0.481151 s | 0.431688 s | -10.28% |
| 4 | TPOT | 0.401188 s | 0.796710 s | +98.59% |
| 4 | E2E | 6.530863 s | 12.448543 s | +90.61% |
| 4 | output throughput | 9.799624 tok/s | 4.761981 tok/s | -51.41% |

The b4 learned cell improves TTFT, but decode dominates the request and both
TPOT and end-to-end latency regress substantially.

## Acceptance and Parity

Every measured target/learned repeat has exact output-token parity.

```text
batch 1 learned:
  proposed tokens per repeat: 15
  accepted draft tokens per repeat: 15
  acceptance rate: 1.000000

batch 4 learned:
  proposed tokens per repeat: 72
  accepted draft tokens per repeat: 53
  acceptance rate: 0.736111
```

Even perfect b1 acceptance does not overcome independent-draft execution and
verification overhead in this pilot.

## Memory and Movement

Median CUDA peak differences:

```text
batch 1 learned minus target:
  allocated: +287.992 MiB
  reserved:  +342.000 MiB

batch 4 learned minus target:
  allocated: +309.826 MiB
  reserved:  +260.000 MiB
```

Proposal-KV movement is zero in every measured run:

```text
H2D bytes: 0
D2H bytes: 0
```

This is expected for the direct allocator and cannot be used as evidence of
Proposal-KV offload benefit. The earlier TP1 offload authority remains the
real Proposal-KV movement evidence.

## Source Binding

```text
result.json:
  73be67aacaa23f9eae3c1b568f57f23dc32af5410ca65a9172301c9173142381

source.tar:
  c23ea1e28ddfdd7ce65fdb95c0e33738475b299916741a0756c1aab57033680c

verify.remote.json:
  dd5958e8abec5110855c06a4a541b61f4fb6ae26a002ad9cdb54a5262f5d71f3

verify.local.json:
  dd5958e8abec5110855c06a4a541b61f4fb6ae26a002ad9cdb54a5262f5d71f3
```

Both independent verifiers recomputed all aggregates and directions from raw
worker rows and verified 14 source-file hashes.

## Diagnostic History

Two failed attempts are intentionally retained as separate bundles:

1. `tp4-qwen3-controlled-performance-gpu3467-20260815` failed before model
   load because the runner omitted the authority environment's
   `run_packages` path containing `flash_attn`.
2. `tp4-qwen3-controlled-performance-gpu3467-r2-20260815` completed both
   target cells, then the learned b1 bootstrap exhausted the incorrectly
   fixed 90-slot Proposal-KV capacity. The worker now derives and validates
   the exact workload upper bound.

No unrelated GPU process was terminated.

## Claim Boundary

Established:

```text
TP4 direct Proposal-KV controlled measurement path
four isolated target/learned cells
one warmup plus three measured runs per cell
exact greedy parity for every measured repeat
raw TTFT/TPOT/E2E/throughput distributions
distributed peak-memory rows
real acceptance rows
real direct-allocator H2D/D2H counters
source-bound remote and local verification
```

Not established:

```text
performance improvement
4K, 16K, or 32K learned-drafter performance
Proposal-KV offload performance benefit
second learned model structure
statistical significance
Phase 1 promotion
```

## Next Performance Step

Before expanding to 4K, add per-stage learned-runtime timing for draft
forward, first-target verification, verify-tail, synchronization, and
Proposal-KV transaction work. The current evidence shows that acceptance is
not the limiting metric: b1 acceptance is 100%, yet TPOT still regresses
27.41%. The next optimization must reduce draft/verification launch and
collective overhead rather than tune acceptance alone.
