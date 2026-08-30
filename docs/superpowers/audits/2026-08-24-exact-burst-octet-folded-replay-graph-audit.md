# Exact-Burst Octet-Folded Replay Graph Completion Audit

Date: 2026-08-30

## Executive Verdict

The default-disabled Exact-Burst Octet-Folded Replay Graph is implemented and
has complete source-bound Qwen3-0.6B BF16 TP1 batch-one ceiling evidence.
It preserves exact outputs and the existing logical execution contract while
reducing the physical CUDA Graph launches used by each eligible K8 region from
eight to one.

The optimization is classified:

```text
NO_GO_CEILING
```

It does not proceed to the terminal gate. The folded graph improves aggregate
median TPOT by only `0.019412%`, below the frozen `1.0%` ceiling threshold, and
the worst paired TTFT regression is `2.083396%`, above the protected `2.0%`
limit. Aggregate P95 TPOT improves `0.707864%`, but that isolated result is not
enough to promote the mechanism.

## Canonical Evidence

```text
authoritative checkout:
  /Users/bytedance/dev/TinyLLMForge
Desktop alias:
  /Users/bytedance/Desktop/TinyLLMForge
branch:
  feat/kv-sparse-attention
source commit:
  b61fe9d6350aa4ada2569feff09f7d6fb7d80a6b
source patch SHA-256:
  e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
run tag:
  20260830-qwen3-06b-octet-folded-ceiling-r4
local evidence:
  artifacts/exact_burst_octet_folded_graph/
    20260830-qwen3-06b-octet-folded-ceiling-r4/
remote staging:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    exact-burst-octet-folded-graph/staging/
    20260830-qwen3-06b-octet-folded-ceiling-r4
remote primary:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    exact-burst-octet-folded-graph/runs/
    20260830-qwen3-06b-octet-folded-ceiling-r4
remote controller:
  /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
    exact-burst-octet-folded-graph/controller-verification/
    20260830-qwen3-06b-octet-folded-ceiling-r4
```

The selected device was physical GPU 0, an NVIDIA A100 80GB PCIe with UUID
`GPU-57be086f-e967-c022-3832-93df4fc77bd0`. Admission observed `0 MiB`,
`0%` utilization, and no compute processes. All remote writes stayed below
the approved mounted `/data00/home/sitian/.../command-timeline-20260818`
root.

## Prompt-to-Artifact Checklist

| Requirement | Implementation and test evidence | Hardware or artifact evidence | Verdict |
| --- | --- | --- | --- |
| Feature is default-disabled and dependency-checked | `tinyvllm/config.py`; configuration tests in `tools/test_exact_greedy_decode_burst.py` and `tools/test_model_runner_spec_verify.py` | Both measured arms differ only by the folded flag | `PASS` |
| Capture exactly eight complete token steps | `ExactGreedyDecodeBurstFoldedGraph`; capture-count, retained-output, identity, and state-progression tests | Folded receipts report eight steps per launch | `PASS` |
| K8 uses one folded launch and K16 can use two without reset | Runtime routing and multi-launch tests in `tools/test_exact_greedy_decode_burst.py` | Ceiling exercises K8; K16 remains covered by local tests but is not promoted by this NO_GO run | `PASS_SCOPED` |
| Unsupported widths fall back before replay | Eligibility and fallback tests | Every measured request has the expected seven one-token tail launches and zero unexpected fallback | `PASS` |
| No same-step retry after a folded launch failure | Quarantine and post-launch failure tests | No measured failure or quarantine occurred | `PASS` |
| Existing one-token graph remains the complete fallback | One-token-after-folded-quarantine tests | Control arm completed all 15 measured requests | `PASS` |
| Exact tokens and decoded text | Producer validation and independent verifier | All 30 performance rows and all policy pairs match | `PASS_EXACT` |
| Exact sampled logits and argmax | Four frozen sampling points per policy and context; 24 float32 sidecars | 24/24 correctness rows match byte digests and argmax | `PASS_EXACT` |
| Logical target work is unchanged | Runtime accounting tests and verifier | Each arm records 1,905 forwards and 1,905 logical replays | `PASS` |
| Physical launch reduction is explicit | Ceiling and independent verifier enforce the frozen 85% eligible-region threshold | Eligible K8 work reduces `120 -> 15` launches per request, or `87.5%`; whole-request physical launches reduce `1,905 -> 330`, or `82.677165%`, after the seven-step tails | `PASS` |
| One final token D2H per committed burst | Runtime tests and row validation | Each arm records 240 calls and 15,240 bytes across 15 measured requests | `PASS` |
| Scheduler ownership, rollback, and quarantine remain intact | Scheduler and exact-burst adjacent suites | Zero fallback, rollback, pending-lease anomaly, or quarantine | `PASS` |
| Frozen workload and execution order | Workload manifest and alternating AB/BA validation | 30/30 performance rows: two policies x three contexts x five repetitions; 24/24 correctness rows | `PASS` |
| Source and patch identity | Source manifest and independent source hash reconstruction | Commit `b61fe9d...`; empty patch digest; all observed rows agree | `PASS` |
| Strict-clean hardware and mounted-only storage | Remote-controller tests and controller manifest | A100 GPU 0 admitted clean; approximately 1.25 TB free under approved root | `PASS` |
| Complete producer and verifier closure | Independent verifier does not import producer/classifier; tamper tests cover rows, source, thresholds, physical counters, P99, workload, and sidecars | Producer exit 0, remote verifier exit 0, fresh local verifier pass; receipts byte-identical | `PASS` |
| Benefit and cost reported together | This audit and `ceiling.json` | TPOT, throughput, TTFT, E2E, launch, D2H, gap, capture, and memory data below | `PASS` |
| Stop rule | Plan Task 6 | `NO_GO_CEILING`; no terminal-gate implementation or run was created | `PASS` |

## Frozen Workload

```text
model: Qwen3-0.6B
precision: BF16
tensor parallelism: 1
batch size: 1
contexts: 256, 2048, 8192
generated tokens: 128
temperature: 0
ignore EOS: true
performance repetitions: 5 per policy and context
warmups: 2 per case
performance rows: 30 / 30
correctness rows: 24 / 24
correctness sidecars: 24 / 24
execution order: alternating AB/BA by repetition and context
```

## Benefit and Cost

Positive percentages below mean improvement; negative percentages mean
regression.

| Context | Median TPOT | P95 TPOT | P99 TPOT | TTFT | E2E | Throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | `+0.160404%` | `+0.707864%` | `+0.707864%` | `+2.280253%` | `+0.351268%` | `+0.352506%` |
| 2,048 | `+0.052136%` | `-0.009485%` | `-0.009485%` | `-0.473159%` | `+0.008799%` | `+0.008800%` |
| 8,192 | `-0.181616%` | `+1.545753%` | `+1.545753%` | `-0.280444%` | `+0.070669%` | `+0.070719%` |

The table reports the median paired improvement within each context. The
frozen aggregate and protection metrics are:

```text
aggregate median TPOT improvement:       0.0194117368%
aggregate P95 TPOT improvement:          0.7078639514%
maximum context median TPOT regression:  0.1816156952%
maximum context P95 TPOT regression:     0.0094847753%
maximum paired TPOT-P99 regression:      0.4665124143%
maximum paired TTFT regression:          2.0833962547%
maximum paired E2E regression:           0.5962847137%
minimum paired throughput improvement:  -0.5927502346%
```

The mechanism and resource accounting are:

| Metric | One-token graph | Octet-folded graph | Interpretation |
| --- | ---: | ---: | --- |
| Logical forwards | 1,905 | 1,905 | unchanged |
| Logical graph replays | 1,905 | 1,905 | unchanged |
| One-token physical launches | 1,905 | 105 | seven tail launches per request remain |
| Folded physical launches | 0 | 225 | fifteen K8 launches per request |
| Total physical launches | 1,905 | 330 | `82.677165%` whole-request reduction |
| Eligible K8 launch reduction | n/a | `87.5%` | passes frozen 85% threshold |
| Final-token D2H calls | 240 | 240 | unchanged |
| Final-token D2H bytes | 15,240 | 15,240 | unchanged |
| Maximum host-visible burst gap | 23.867218 ms | 23.890323 ms | approximately unchanged |
| Maximum capture duration | 8,086.462605 ms | 8,698.384968 ms | folded capture costs more |
| Maximum capture allocated delta | 0.872070 MiB | 6.976562 MiB | folded adds about 6.10 MiB |
| Maximum capture reserved delta | 2.000000 MiB | 8.000000 MiB | folded adds 6 MiB |
| Maximum retained static bytes | 0.873455 MiB | 6.972874 MiB | delta 6,395,704 bytes |
| Maximum peak allocated | 38.817954 GiB | 38.798283 GiB | no regression |
| Maximum peak reserved | 39.074219 GiB | 38.958984 GiB | no regression |
| Fallbacks / rollbacks / quarantines | 0 / 0 / 0 | 0 / 0 / 0 | no runtime anomaly |

The folded capture allocated and reserved ratios are
`0.017675%` and `0.020227%` of the paired baseline peaks, both below the
frozen `1%` limits. Maximum folded capture duration is `8.698385 s`, below
the `120 s` ceiling.

## Classification

Passing requirements:

- complete evidence, source identity, workload identity, and execution order;
- exact token, text, sampled-logit, and argmax parity;
- exact logical-forward and logical-replay accounting;
- exact physical launch counts and `87.5%` eligible K8 launch reduction;
- aggregate P95 TPOT improvement above `0.5%`;
- context median/P95 and paired P99 protection;
- E2E, throughput, capture-memory, retained-memory, and capture-time limits;
- zero fallback, rollback, or quarantine anomalies.

Failing requirements:

- aggregate median TPOT improves only `0.019412%`, below `1.0%`; and
- worst paired TTFT regresses `2.083396%`, above `2.0%`.

Therefore:

```text
OCTET_FOLDED_REPLAY_GRAPH_CLASSIFICATION=NO_GO_CEILING
TERMINAL_GATE_AUTHORIZED=false
PRODUCTION_PROMOTION_AUTHORIZED=false
```

## Controller Recovery

The producer completed and wrote all primary artifacts plus
`producer_exitcode=0`, but the long-lived SSH command returned nonzero after
the remote process had completed. The controller preserved the full primary
bundle and controller partial evidence before raising.

Recovery did not rerun the producer. It used the immutable r4 staging source
to run only the independent remote verifier, downloaded the controller
evidence to `controller-resume/`, and ran a fresh local independent verifier.
Both verifier receipts are byte-identical:

```text
remote verifier exit: 0
local verifier: PASS
receipt SHA-256:
  bb73798076528c5dae6d5e66d96b2bc1e030b8972d982faeb68d42f546a2ad02
```

The earlier r3 run is retained but superseded. Its classifier did not yet
make the frozen eligible-launch and TPOT protection thresholds explicit.

## Claim Boundary

This result proves that TinyLLMForge can execute eight ordered exact-greedy
decode steps in one CUDA Graph launch while preserving the measured
single-sequence correctness and accounting contracts.

It does not prove a deployable speedup. It does not establish benefit for
Qwen3-8B, Qwen3.8-27B, tensor parallelism greater than one, batching,
non-greedy sampling, variable output lengths, other hardware, or production
traffic. It also does not establish academic novelty. The mechanism remains
default-disabled and must not be promoted on the basis of this run.
