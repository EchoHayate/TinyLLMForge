# SLO-aware Cohort Decode Burst canonical qualification

## Result

The canonical Qwen3-0.6B/A100/TP1 qualification is a terminal **NO-GO**.
The candidate must remain disabled and must not be promoted.

The GPU worker completed all 2,340 request rows, 28,568 decision rows, and
20,407 execution rows. Its producer summary reported
`NO_GO_TAIL_LATENCY`, but the independent remote verifier then rejected the
bundle with `paired output correctness mismatch`. The producer's
`correctness_passed=true` and `verifier_agreement=true` fields are therefore
not authoritative: they cover the isolated 16-case correctness matrix and
were written before independent canonical verification.

## Source and execution boundary

- Source commit:
  `960fde15f497a95e12411576d6f9e48129b2ed2d`
- Branch at launch: `feat/kv-sparse-attention`
- Model: Qwen3-0.6B
- Hardware: one NVIDIA A100 80GB PCIe
- Tensor parallel size: 1
- Stage-1 tag: `20260914-correctness-960fde15-r1`
- Stage-2 tag: `20260914-canonical-960fde15-r1`
- Remote storage root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/slo-cohort-burst`
- Stage-2 remote raw bundle size: approximately 387 MiB

The shared checkout advanced after launch because of unrelated KV-capacity
work. All formal artifacts remain bound to `960fde15`; the current mutable
checkout is not used as source evidence.

## Correctness and lifecycle

Stage-1 passed remote and local independent verification:

- classification: `PASS_CORRECTNESS_AND_LIFECYCLE`
- exact correctness cases: 16
- manifest SHA-256:
  `0bc041fc6a6e32c9a04e0f33935026b20ad1850e9504c75403aaa017aa0a4b5b`

Stage-2 exposed 11 baseline/candidate output mismatches among 1,170 paired
requests:

- medium load: 6
- high load: 5
- repetitions: 1 in repetition 2, 7 in repetition 3, and 3 in repetition 4
- mismatched requests with their own cohort-burst execution: 3
- mismatched requests with no cohort-burst execution: 8

The independent verifier failed closed before producing a verification
receipt. The Stage-2 manifest SHA-256 is
`f90ccf2ac87f54024bf81958e1f0769698556b03d4239345bb2acbb156428f4e`.

## Root-cause evidence

The evidence supports batch-schedule-dependent greedy divergence rather than
a direct single-row burst write error:

1. The isolated Stage-1 graph/correctness matrix passed all 16 cases.
2. Eight of the 11 mismatched requests never executed a cohort burst.
3. For identical prompts, the baseline itself produced different token
   sequences across load points in 33 duplicate-prompt groups.
4. Three mismatched candidate outputs exactly equal the baseline output for
   the same prompt at another load point.
5. Cohort bursting changes dynamic batch composition and therefore changes
   the batch-shape numerical path. Greedy token identity is not invariant
   under that change for this runtime/checkpoint.

This evidence rejects global exact-output equivalence. It does not establish
cross-request KV corruption.

## Performance result

The producer-side metrics are diagnostic only because independent
verification failed, but they independently reject promotion:

| Metric | Result | Gate |
| --- | ---: | ---: |
| Aggregate throughput improvement | +0.188106% | at least +10% |
| Medium-load throughput improvement | +0.270566% | at least +10% |
| High-load throughput improvement | +0.223727% | at least +10% |
| Worst throughput regression | +1.229188% | at most +2% |
| Worst P99 ITL regression | +669.615134% | at most +3% |
| Worst P99 TTFT regression | +41.299007% | at most +5% |
| Worst P99 E2E regression | +83.584382% | at most +5% |
| Maximum host-visible gap | 4.809370 s | at most 0.040 s |
| Peak reserved-memory regression | 0% | at most +5% |
| Post-EOS wasted-forward fraction | 0.010623% | at most 10% |
| Starved requests | 0 | 0 |

Even if the paired-output failure were ignored, the candidate would still
fail tail-latency and throughput gates by large margins.

## Decision and claim boundary

Terminal decision:
`STOP_NO_GO_CORRECTNESS_AND_PERFORMANCE`.

Do not claim a throughput, latency, exactness, or production win for
SLO-aware Cohort Decode Burst. The implementation remains default-disabled.
No result here generalizes beyond Qwen3-0.6B/A100/TP1 and the frozen workload.

The complete raw evidence remains remote. This repository stores only the
compact postmortem and report; it intentionally does not duplicate the
hundreds of MiB of immutable JSONL evidence locally.
