# Qwen3.8 TP4 Peer-Reduction Microgate Audit

## Executive Verdict

The isolated four-A100 microgate completed on August 31, 2026 and produced
the terminal classification:

```text
NO_GO_MICROGATE
```

The fixed-slot CUDA IPC peer-reduction path is correct, topology-eligible,
bounded in memory, timeout-free, and cleanly torn down. It nevertheless
misses the frozen performance gate by a wide margin:

- active tokens 1: median CUDA latency regresses by 9.85%;
- active tokens 4: median CUDA latency regresses by 29.98% and P99 by 40.85%;
- active tokens 8: median CUDA latency regresses by 43.73% and P99 by 4.59%.

The candidate reduces median host submission time by 12.54% to 15.53%, but
that CPU-side benefit does not compensate for the device-side peer-memory
reduction and synchronization cost on this PCIe topology. Runtime integration
is therefore forbidden by the accepted stop rule.

## Frozen Scope

- Source revision:
  `edef77db06d9d99b44a87aa8aba3628c16f02cf8`
- Source-tree SHA-256:
  `6fa1209c945264cad440930b59fbfcf9fe3c464c539e249ad823043232699f7f`
- Attempt:
  `20260830-qwen38-tp4-peer-reduction-r1`
- Remote root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`
- World size: 4
- Hidden size: 5120
- Active-token groups: 1, 4, and 8
- Warmup pairs per group: 2
- Measured pairs per group: 200
- Rank rows per pair: 4
- Total measured rows: 2400
- Arm order: alternating baseline/candidate and candidate/baseline

No model-runtime integration, A64 gate, or end-to-end claim is included in
this result.

## Launch and Resource Identity

The local controller admitted four strict-clean GPUs immediately before
launch:

| Rank | GPU index | GPU UUID |
| ---: | ---: | --- |
| 0 | 0 | `GPU-57be086f-e967-c022-3832-93df4fc77bd0` |
| 1 | 1 | `GPU-7dc22583-df04-6c76-4ba5-ea32c428c130` |
| 2 | 2 | `GPU-63c05907-407b-8240-07a0-f38872840867` |
| 3 | 3 | `GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d` |

All four selected GPUs reported 0 MiB used memory, 0% utilization, and no
compute process at admission. The worker later proved all 12 directed peer
edges with both CUDA peer access and IPC round-trip success.

The supervisor owned four worker PIDs, collected 39 resource snapshots, saw
no ownership violation, and received exit code 0 from every rank. A
post-run check again showed all eight host GPUs at 0 MiB and 0% utilization
with no compute application.

## Benefit and Cost

CUDA duration is reconstructed per pair as the maximum duration across all
four ranks. P99 uses nearest-rank selection over 200 paired measurements.

| Active tokens | Baseline median | Candidate median | Median result | Baseline P99 | Candidate P99 | P99 result |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 159.408 us | 175.104 us | 9.85% slower | 544.767 us | 474.016 us | 12.99% faster |
| 4 | 187.072 us | 243.152 us | 29.98% slower | 310.272 us | 437.023 us | 40.85% slower |
| 8 | 239.856 us | 344.736 us | 43.73% slower | 471.904 us | 493.568 us | 4.59% slower |

The frozen gate required at least 10% median speedup for active tokens 1 and
4, no more than 2% median regression for active tokens 8, and no more than 3%
P99 regression for any group. The candidate fails every median requirement
and the token-4 and token-8 P99 requirements.

Host submission has a real but insufficient benefit:

| Active tokens | Baseline host median | Candidate host median | Host result |
| ---: | ---: | ---: | ---: |
| 1 | 170.863 us | 149.445 us | 12.54% lower |
| 4 | 172.864 us | 150.696 us | 12.82% lower |
| 8 | 176.732 us | 149.279 us | 15.53% lower |

The mechanism therefore moves work away from host submission but makes the
GPU critical path materially longer. On this machine, the fixed rank
0-to-1-to-2-to-3 peer-memory reduction is not a viable replacement for the
NCCL AllReduce plus BF16 cast/residual baseline.

## Correctness, Timeout, Memory, and Cleanup

- Cross-rank maximum absolute error: 0.0
- Cross-rank maximum relative error: 0.0
- Candidate-versus-baseline maximum absolute error: 0.0
- Candidate-versus-baseline maximum relative error: 0.0
- Timed-out rows: 0
- Nonzero device-status rows: 0
- Maximum allocated delta per rank: 31,458,816 bytes, or about 30.00 MiB
- Frozen allocated-delta ceiling: 48 MiB
- Peer group closed on all four ranks: yes
- Owned children remaining: none
- Three exact-tag process scans: all empty
- Cleanup classification: `CLEAN`

Correctness, memory, timeout, and cleanup gates pass. The terminal `NO_GO`
is exclusively a performance result.

## Producer and Independent Verification

The producer classification, remote independent verifier, and locally rerun
independent verifier agree:

```text
producer:            NO_GO_MICROGATE
remote verifier:     NO_GO_MICROGATE
local verifier:      NO_GO_MICROGATE
manifest verification: PASS
```

The independent verifier checked all nine compact producer artifacts,
reconstructed the 12 directed peer edges and 2400 unique measurement rows,
recomputed the classification from raw rows, and verified the SHA-256
manifest. It did not trust `microgate_summary.json` as classification input.

## Prompt-to-Artifact Completion Checklist

| Requirement | Concrete evidence | Result |
| --- | --- | --- |
| Work only in the authoritative checkout | Source revision and Git branch are recorded in controller and bundle identities | PASS |
| Use a committed source archive | `source_identity.json` records full revision and source-tree SHA-256 | PASS |
| Keep remote task data below the approved root | Every path in `controller/plan.json` descends from the approved remote root | PASS |
| Do not reuse an existing attempt | Controller preflight required the attempt to be absent; one immutable attempt was created | PASS |
| Require four strict-clean GPUs | `controller/launch_admission.json` records four 0 MiB, 0%, process-free GPUs | PASS |
| Recheck immediately before launch | Controller execution performs the second strict-clean probe before worker start | PASS |
| Use attempt-local temp/build/cache/log paths | `TMPDIR`, `TORCH_EXTENSIONS_DIR`, `CUDA_CACHE_PATH`, XDG cache, source, raw data, and logs are under the attempt root | PASS |
| Do not terminate external processes | Plan contains no kill command; supervisor only observes owned descendants and reports no foreign process | PASS |
| Prove all directed peer edges | `peer_access_matrix.json` and `ipc_roundtrip.jsonl` contain 12 successful edges | PASS |
| Measure active-token groups 1, 4, and 8 | `microgate_rows.jsonl` contains 800 rows per group | PASS |
| Require at least 200 pairs per group | Independent reconstruction finds 200 complete four-rank pairs per group | PASS |
| Require four ranks per pair | 2400 rows have 2400 unique `(tokens, pair, rank)` identities | PASS |
| Alternate AB/BA order | Raw rows record even-pair baseline/candidate and odd-pair candidate/baseline order | PASS |
| Enforce zero timeout and valid status | Raw rows contain zero timeout and zero nonzero device status | PASS |
| Enforce numerical tolerances | All recorded cross-rank and baseline errors are exactly zero | PASS |
| Enforce at most 48 MiB allocated delta | Maximum is about 30.00 MiB | PASS |
| Require clean teardown | `cleanup.json` is `CLEAN`, all groups closed, no owned child remains, and three scans are empty | PASS |
| Assemble one compact immutable bundle | Local compact evidence is about 1.1 MiB and contains no build object or large trace | PASS |
| Run byte-independent verification | Remote and local verifier outputs match and manifest hashes verify | PASS |
| Report benefit and cost | CUDA median/P99, host submission, memory, correctness, timeout, and cleanup are reported above | PASS |
| Integrate only after a real microgate PASS | Classification is `NO_GO_MICROGATE`; `runtime_integration_authorized` is false | STOP RULE ENFORCED |
| Preserve large build data remotely | Only compact controller receipts and final bundle were downloaded | PASS |

## Final Classification

`NO_GO_MICROGATE`

The implementation is a valid and correct mechanism experiment, but it is
not a performance optimization on the tested four-A100 PCIe topology. Task 7
runtime integration, A64 qualification, and end-to-end benchmarking are not
authorized and must not be started from this result.
