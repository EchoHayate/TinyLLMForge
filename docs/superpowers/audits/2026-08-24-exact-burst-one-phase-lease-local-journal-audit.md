# Exact-Burst One-Phase Lease-Local Journal Audit

Date: 2026-08-24

## Objective

Validate a runtime-data-flow-specific original engineering design that
replaces the context-length-scaled generic scheduler rollback journal with a
lease-local constant-size transaction for eligible non-terminal one-phase K8
exact-greedy bursts.

Promotion requires all of the following:

- exact output-token, sampled-logit, and argmax parity;
- unchanged target-model forwards, graph replays, and D2H inventory;
- no candidate generic-journal captures, fallbacks, or rollbacks;
- attempts, captures, and commits equal the eligible-burst count;
- at least 50% median and P95 prepare improvement at 8K;
- at least 35% aggregate median and P95 prepare improvement;
- at least 1% aggregate median and P95 TPOT improvement;
- no TTFT, E2E, TPOT-P99, or throughput regression above 2%;
- no allocated or reserved memory regression above 1%; and
- a complete source-bound artifact with 60 performance rows, 24 correctness
  rows, and agreeing producer, remote verifier, and local verifier results.

## Prompt-to-Artifact Checklist

| Requirement | Implementation evidence | Contract evidence | GPU/artifact evidence | Status |
| --- | --- | --- | --- | --- |
| Default-disabled one-phase lease-local journal | `tinyvllm/config.py` | config/model-runner tests | r10 candidate arm explicitly enables the flag | PASS |
| Eligibility limited to non-terminal one-phase K8 | `tinyvllm/engine/scheduler.py` | scheduler selection/fallback tests | All 30 candidate performance rows record 16 attempts but only 15 captures/commits plus one `unsupported_burst_shape` fallback | FAIL |
| Constant-size token and scheduler rollback state | `ExactBurstLeaseLocalDeltaJournal` | rollback/fault-injection tests and local CPU profile | 60/60 prepare-sample rows were preserved; terminal source-bound summary is absent | PARTIAL |
| Zero-or-one write-block publication | lease write-block publication plan | commit/rollback/publication tests | Candidate rows record zero rollbacks, but incomplete terminal evidence prevents promotion | PARTIAL |
| Numerical path unchanged | eight existing token appends and existing CUDA graph | token/logit/argmax mutation tests | Available 2K/4K pairs are exact; all eight 8K correctness rows are missing | INCOMPLETE_16_OF_24 |
| Paired same-policy workload | `tools/exact_burst_one_phase_lease_local_journal_gate.py` | fixed inventory/order/schema tests | Complete 60/60 performance matrix | PASS_PERFORMANCE_INVENTORY_ONLY |
| Producer and independent verifier agree | producer gate plus independent verifier | tamper/source-drift/NaN tests | No producer summary/gate, remote verifier, or local PASS receipt exists | FAIL_INCOMPLETE |
| Source dependency closure | gate and verifier `SOURCE_FILES` | profiler dependency inventory test | Worker exited before writing `source_manifest.json` or `runner_receipt.json` | FAIL_INCOMPLETE |
| Strict-clean GPU and mounted-only runtime paths | safe remote controller | controller safety and recovery tests | r10 selected strict-clean GPU 2 under approved mounted root | PASS |
| Benefit and cost reported together | gate summary schema | threshold and classification tests | Partial diagnostics are recorded below without a promotion claim | PASS_WITH_INCOMPLETE_BOUNDARY |

## Source Boundary

- Authoritative checkout: `/Users/bytedance/Desktop/TinyLLMForge`
- Branch: `feat/kv-sparse-attention`
- r10 source commit:
  `c3890e7aa4d17b12f00d41b5df0bf8a004de6b96`
- Source patch SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Model: `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`
- Remote runtime root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/exact-burst-one-phase-lease-local-journal`
- Canonical run tag:
  `20260824-qwen3-06b-one-phase-lease-local-r10`
- Selected hardware at launch: GPU 2,
  `GPU-63c05907-407b-8240-07a0-f38872840867`,
  NVIDIA A100 80GB PCIe, 0 MiB used, 0% utilization, and no compute process.

## Local Evidence

CPU prepare-and-rollback profile:

| Sequence length | Median improvement | P95 improvement |
| ---: | ---: | ---: |
| 249 | 10.684% | 12.218% |
| 2,041 | 73.484% | 74.323% |
| 8,185 | 90.011% | 89.688% |

Latest local test evidence before r10:

```text
focused one-phase journal suite: 329 passed, 1 skipped
split-phase adjacent gate:         9 passed
ragged-coalescing adjacent gate:   8 passed
continuation-epoch adjacent gate:  9 passed
```

The skipped case is pre-existing and environment-dependent; it is not treated
as hardware evidence.

## Remote Execution History

Tags r1 through r7 are immutable transport/preflight failures and contain no
usable terminal performance evidence. r8 and r9 launched workers but stopped
before emitting a complete performance matrix:

- r8 exposed a missing gate-to-profiler context contract;
- r9 exposed a stale profiler capture-receipt schema;
- both defects were reproduced locally, fixed with RED-to-GREEN tests, and
  pushed before r10;
- no partial rows from any failed tag are included in the terminal
  classification.

r10 was the only candidate canonical artifact. The remote filesystem became
unavailable during correctness collection. After SSH access recovered on
2026-08-25, the controller observed that worker PID/PGID `2952927` had
disappeared before writing `worker.exitcode`.

The preserved artifact contains:

```text
performance rows:        60 / 60
prepare-sample rows:     60 / 60
correctness rows:        16 / 24
2K correctness rows:      8 / 8
4K correctness rows:      8 / 8
8K correctness rows:      0 / 8
logit sidecars:            8 / 12 expected policy/point pairs
worker.exitcode:          missing
summary.json:             missing
source_manifest.json:     missing
runner_receipt.json:      missing
remote verifier receipt:  missing
local verifier PASS:      absent
```

The frozen-source verifier from commit
`c3890e7aa4d17b12f00d41b5df0bf8a004de6b96` rejected the downloaded
artifact before reconstruction with:

```text
ValueError: required artifact is missing: source_manifest.json
```

The tag is immutable and will not be resumed, overwritten, or reused.

## Benefit and Cost

The following values are diagnostics from the complete 60-row performance
matrix. They are not a terminal performance claim because the correctness
matrix and source-bound receipts are incomplete.

```text
prepare median improvement:
  2K:        72.377760%
  4K:        82.514499%
  8K:        87.226748%
  aggregate: 81.372219%

prepare P95 improvement:
  2K:        20.066995%
  4K:        35.655786%
  8K:        40.445404%
  aggregate: 64.937033%

aggregate TPOT median improvement:  2.382038%
aggregate TPOT P95 improvement:     1.768253%
aggregate TPOT P99 improvement:     1.768253%
aggregate TTFT regression:          0.169831%
aggregate E2E improvement:          2.026498%
median throughput improvement:      2.068202%
peak allocated-memory reduction:    0.464710%
peak reserved-memory reduction:     1.011846%
```

The available 2K and 4K correctness pairs have exact token IDs, argmax
values, sampled logits with maximum absolute difference `0.0`, and equal
forward/replay/D2H inventories. No 8K correctness evidence exists.

Every one of the 30 candidate performance rows records:

```text
eligible bursts:          16
one-phase attempts:       16
one-phase captures:       15
one-phase commits:        15
generic journal captures: 1
fallbacks:
  unsupported_burst_shape: 1
rollbacks:                 0
target forwards/replays:   127 / 127
D2H calls/bytes:            16 / 1016
```

Consequently, even the diagnostic performance matrix misses the frozen 8K
prepare-P95 threshold of 50% and violates the zero-fallback/zero-generic-
capture lifecycle requirement. These observations do not supersede the
fixed-precedence incomplete-evidence classification.

## Final Classification

`NO_GO_EVIDENCE_INCOMPLETE`

No performance or promotion claim is authorized. r10 has 60/60 performance
rows but only 16/24 correctness rows, no worker exit receipt, no producer
summary/gate, and no remote/local verifier agreement. The preserved partial
bundle remains diagnostic evidence only.
