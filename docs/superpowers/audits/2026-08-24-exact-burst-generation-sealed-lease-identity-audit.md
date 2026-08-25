# Exact-Burst Generation-Sealed Lease Identity Audit

Date: 2026-08-25

## Objective

Validate a default-disabled runtime-data-flow-specific original engineering
design that replaces repeated full block-table identity construction on stable
one-phase exact-greedy K8 leases with a fail-closed generation seal.

Promotion requires all of the following:

- exact output-token, sampled-logit, and argmax parity;
- unchanged target-model forwards, graph replays, and D2H inventory;
- zero candidate fallback, failure, and rollback;
- one cold seal capture followed by hot reuse for every remaining eligible
  burst;
- at least 25% median and P95 lifecycle improvement at 8K;
- at least 15% aggregate median and P95 lifecycle improvement;
- at least 0.5% aggregate median and P95 TPOT improvement;
- no TTFT, E2E, TPOT-P99, or throughput regression above 2%;
- no allocated or reserved memory regression above 1%; and
- a complete source-bound artifact with 60 performance rows, 24 correctness
  rows, worker exit code zero, and agreeing producer, remote verifier, and
  frozen-source local verifier classifications.

## Prompt-to-Artifact Checklist

| Requirement | Implementation evidence | Contract/test evidence | GPU/artifact evidence | Verdict |
| --- | --- | --- | --- | --- |
| Default-disabled sealed identity | `tinyvllm/config.py` | focused config and model-runner tests | Candidate arm explicitly enables the flag; baseline retains full identity | PASS |
| Mutation-tracked sequence table | `VersionedBlockTable` in `tinyvllm/engine/sequence.py` | mutation, replacement, pickle, and legacy-state tests | Candidate completes all contexts without table-revision fallback | PASS |
| Authoritative ownership generation | `BlockManager` generation and `BlockTableIdentitySeal` | allocation, release, generation drift, capture, and validation tests | 1,440 candidate validations complete with zero fallback | PASS |
| Fail-closed lease lifecycle | scheduler captures and validates the seal before mutation | stale-seal, commit, rollback, and worker-serialization tests | zero failures and zero one-phase rollbacks | PASS |
| Full identity preserved as baseline/fallback | baseline path remains unchanged | baseline counter and fallback tests | baseline identity counters are authoritative and candidate fallback count is zero | PASS |
| Numerical path unchanged | no model, graph, KV-slot, token, or sampling change | exact token/logit/argmax tests | exact output IDs and argmax; maximum sampled-logit absolute difference `0.0` | PASS |
| Execution inventory unchanged | existing K8 graph and model path | paired inventory tests | 127 target forwards, 127 graph replays, and equal D2H inventory in every pair | PASS |
| Complete paired workload | source-bound gate and frozen workload manifest | inventory/order/schema tests | 60/60 performance rows and 24/24 correctness rows | PASS |
| Source and artifact closure | source/workload manifests plus runner receipt | source drift, artifact tamper, missing row, and non-finite tests | clean patch hash, source SHA, four artifact hashes, 12 logits sidecars | PASS |
| Strict-clean GPU and approved storage | safe remote controller | admission, path, Kerberos TTL, and recovery tests | GPU 2 admitted at 3 MiB, 0%, no compute process; all remote data under `/data00/home/sitian/...` | PASS |
| Worker and dual verifier closure | controller completion protocol | verifier independence and tamper tests | worker exit 0; remote and local verifiers both `verified=true` | PASS |
| Lifecycle median thresholds | constant-time hot seal reuse | CPU profile and gate threshold tests | 8K `48.046332%`; aggregate `35.881303%` | PASS |
| Lifecycle P95 thresholds | constant-time hot seal reuse | CPU profile and gate threshold tests | 8K `7.587183%` vs 25%; aggregate `12.584104%` vs 15% | FAIL |
| TPOT median/P95 thresholds | unchanged CUDA execution path | gate threshold tests | median `0.463800%` vs 0.5%; P95 regresses `2.439467%` | FAIL |
| Protected metrics | frozen regression limits | gate threshold tests | TTFT improves `1.309128%`, E2E improves `0.219737%`, throughput improves `0.220221%`, but TPOT P99 regresses `2.439467%` | FAIL_TPOT_P99 |
| Memory limits | no additional GPU-resident state | gate threshold tests | allocated improves `0.464710%`; reserved improves `1.011846%` | PASS |
| Benefit and cost reported together | summary schema and this audit | producer/verifier reconstruction | lifecycle benefit and host-tail cost are both recorded below | PASS |

## Source and Execution Boundary

- Authoritative checkout:
  `/Users/bytedance/Desktop/TinyLLMForge`
- Branch: `feat/kv-sparse-attention`
- Frozen source commit:
  `18f2ff24d2c4fa470a2f118afada194b26f4149a`
- Source patch SHA256:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Model: `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`
- Run tag: `20260825-generation-sealed-task7-r7`
- Remote root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/exact-burst-generation-sealed-lease-identity`
- Hardware: NVIDIA A100 80GB PCIe, physical GPU 2,
  `GPU-63c05907-407b-8240-07a0-f38872840867`
- Admission snapshot: 3 MiB used, 0% utilization, no compute process
- Worker PID/PGID: `3768160 / 3768160`
- Distribution port: `24401`
- Worker exit code: `0`

The local downloaded bundle is:

```text
artifacts/exact_burst_generation_sealed_lease_identity/
  20260825-generation-sealed-task7-r7-controller/
```

Generated artifacts remain untracked and are not committed.

## Verification Closure

Terminal inventory:

```text
performance rows:       60 / 60
correctness rows:       24 / 24
logit sidecars:         12 / 12
worker exit code:       0
producer classification:       NO_GO_PERFORMANCE
remote verifier classification: NO_GO_PERFORMANCE
local verifier classification:  NO_GO_PERFORMANCE
remote verifier verified:       true
local verifier verified:        true
```

Artifact SHA256 values bound by `source_manifest.json`:

```text
performance_rows.jsonl:
  a311c216bdad6cada8df95f8a7fb232f3b786ff5d2fd45082bef34935345e548
correctness_rows.jsonl:
  96c06067d615a0fcfb5eba758e0d119b1300567029584ef55da5382fd1842d31
lifecycle_samples.jsonl:
  ea098a88209ae927b5b30d134b8971d2cef3168cd7ac9aebd06eaf24f8dbd520
workload_manifest.json:
  f81cff6ced08cd9ac313f79bdc3a15f3a8f2dd8c6cecf62f428786953d1978ac
```

The controller status is `COMPLETE`, and its download receipt is
`DOWNLOADED_AND_VERIFIED`.

Fresh post-gate verification:

```text
19 dependency-light focused/adjacent files, isolated processes:
  505 passed, 1 skipped
Torch-backed tools/test_chunked_prefill.py on the frozen remote source:
  101 passed
fresh local independent verifier:
  verified=true, NO_GO_PERFORMANCE, 60/60 + 24/24
py_compile for gate, verifier, and remote controller:
  PASS
documentation diff check:
  PASS
```

The earlier combined-process run remains a known test-isolation limitation:
`482 passed, 1 skipped, 32 failed`; the same code passed when files were
isolated. The failures were traced to legacy import/global monkeypatch
contamination and are not hidden as a globally green suite.

## Correctness and Transactional Safety

All baseline/candidate pairs have:

- exact output token IDs;
- exact sampled argmax;
- sampled-logit maximum absolute difference `0.0`;
- equal prompt digest and generated-token count;
- equal target-model forwards, graph replays, D2H calls, and D2H bytes; and
- no candidate failure, fallback, or rollback.

Every candidate performance row records:

```text
eligible bursts:              16
identity-seal cold captures:   1
identity-seal hot reuses:     15
identity-seal validations:    48
identity-seal fallbacks:       0
exact-burst failures:          0
one-phase rollbacks:           0
target forwards:             127
graph replays:               127
D2H calls / bytes:          16 / 1016
```

Across 30 candidate rows this is 30 cold captures, 450 hot reuses, and 1,440
validations with zero fallback, failure, or rollback.

## Benefit and Cost

### CPU lifecycle support

Ten recent local profiles, r17 through r26, provide supporting
microbenchmark evidence:

```text
classifications:                         9 GO / 1 NO_GO
aggregate lifecycle median improvement:
  cross-run median:                      25.617384%
  range:                                 10.883874% to 30.794402%
8K lifecycle median improvement:
  cross-run median:                      50.694998%
  range:                                 46.622785% to 61.780906%
8K lifecycle P95 improvement:
  cross-run median:                      58.049860%
  range:                                 32.846132% to 72.677378%
```

This establishes that the O(block-count) identity walk can be removed from
the synthetic hot lifecycle. It is microbenchmark evidence only.

### GPU end-to-end result

```text
lifecycle median improvement:
  2K:                                    27.912517%
  4K:                                    33.858493%
  8K:                                    48.046332%
  aggregate:                             35.881303%

lifecycle P95 improvement:
  2K:                                     5.143060%
  4K:                                     0.552084%
  8K:                                     7.587183%
  aggregate:                             12.584104%

aggregate TPOT median improvement:        0.463800%
aggregate TPOT P95 regression:            2.439467%
aggregate TPOT P99 regression:            2.439467%
aggregate TTFT improvement:               1.309128%
aggregate E2E improvement:                0.219737%
median throughput improvement:            0.220221%
peak allocated-memory improvement:        0.464710%
peak reserved-memory improvement:         1.011846%
```

The benefit is real but localized: median scheduler lifecycle falls sharply,
especially at 8K. The cost is that this bookkeeping reduction does not
translate into the frozen TPOT target, and host-side tail variation is worse.
The candidate misses four promotion checks:

1. 8K lifecycle P95 is `7.587183%`, below `25%`;
2. aggregate lifecycle P95 is `12.584104%`, below `15%`;
3. aggregate TPOT median improvement is `0.463800%`, below `0.5%`; and
4. TPOT P95/P99 regress `2.439467%`, exceeding the `2%` protection limit.

No correctness, transactional-safety, execution-inventory, or memory failure
was observed.

## Model-Agnostic Runtime Review

### Two-axis verdict

- Mechanism: `reusable candidate`
- Integration: `first adopter only`

### Layer map

- Mechanism: sequence block-table revision, block ownership generation,
  immutable identity seal, scheduler capture/validation, fail-closed fallback.
- Adapter: none; the mechanism consumes existing `Sequence`, `BlockManager`,
  and exact-burst lease contracts.
- Policy/config: the feature flag and one-phase exact-greedy eligibility
  remain outside the seal data structure.
- Benchmark/profile: Qwen3-0.6B, TP1, batch 1, K8, context buckets, repetitions,
  correctness sampling points, and promotion thresholds.

### Leakage evidence and data flow

Core runtime files contain no Qwen checkpoint name, context bucket, workload
ordering, or benchmark threshold. The producer/consumer flow is:

```text
Sequence table mutation
  -> table revision
BlockManager ownership mutation
  -> ownership generation
Scheduler lease capture
  -> immutable BlockTableIdentitySeal
worker execution
  -> structural seal projection
Scheduler/BlockManager commit validation
  -> accept unchanged generations or fail closed
```

The first production caller is still only one-phase exact-greedy K8. That
means the contract is reusable by construction but not yet proven generic by
a second caller or model.

### Recommended split

The current commits already separate the reusable core mechanism, first
adopter integration, benchmark/controller, and validation evidence. A future
promotion should add a second caller or synthetic lifecycle before describing
the integration itself as generic.

### Evidence boundary

Proven:

- mutation and ownership invalidation semantics;
- fail-closed lease validation;
- exact Qwen3-0.6B TP1 numerical and execution parity;
- complete source-bound A100 evidence; and
- scheduler-lifecycle median benefit.

Not proven:

- a TPOT or tail-latency win;
- a second caller or second model;
- TP greater than one, batching, streaming, or EOS-aware behavior;
- Qwen3-8B performance; or
- production-default safety.

## Final Classification

`NO_GO_PERFORMANCE`

The implementation is correct, source-bound, complete, independently
verified, and default-disabled. It removes substantial median scheduler
lifecycle work, but the GPU end-to-end gate does not authorize promotion
because lifecycle P95, TPOT median, TPOT P95, and TPOT P99 fail the frozen
thresholds.

The code remains useful infrastructure for later experiments, but it must not
be described as an end-to-end latency optimization on the basis of this gate.
