# TP4 Completion-Owned Overlap Stage-0.1 Audit

**Audit date:** 2026-09-08

**Terminal attempt:** `20260908-tp4-completion-owned-overlap-stage01-r1`

**Protocol:** `completion-owned-stage01`

**Source revision:** `472d1cc84de8eea35fe55daab9ebf652100a5206`

**Source tree SHA-256:** `ef237c378f8ddfe17d06b4f74605c3736c29038798ed59c18aa66b2b35f2564e`

**Terminal classification:** `NO_GO_PERFORMANCE`

**Stage-1 integration:** prohibited

## 1. Executive conclusion

Completion ownership repaired the correctness and lifecycle failure observed in
the preceding event-only Stage-0 design. Every formal correctness check and
every lifecycle row passed, and the deliberately unsafe event-only diagnostic
reproduced premature collective visibility failures. The completion-owned arm
also produced a 100% median realized-overlap ratio for all three frozen shapes.

The mechanism nevertheless fails the frozen performance gate:

- active-token 4 median critical latency regressed `28.670041%`;
- active-token 8 median critical latency regressed `10.280405%`;
- their geometric aggregate speedup was `-19.120881%`, below the required
  `+5%`;
- active-token 1 P99 regressed `206.975043%`;
- active-token 4 P99 regressed `10.001668%`;
- host submission regressed between `99.162755%` and `127.951833%`; and
- active-token 4 and 8 improved only `7/15` and `8/15` pairs, below the
  required `11/15`.

The producer, remote independent verifier, downloaded local independent
verifier, and fresh post-seal `--check-only` verification all agree on
`NO_GO_PERFORMANCE`. Evidence integrity is `PASS`; performance qualification
is not.

This is a terminal, model-neutral microgate result. It is not Qwen3.8
end-to-end latency, throughput, TTFT, TPOT, or memory evidence. It does not
authorize Qwen integration, model-transaction changes, or edits to
`tinyvllm/layers/linear.py`.

## 2. Immutable attempt and storage identity

The formal attempt used the pushed source revision and source-tree identity
shown above. The remote attempt root was:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/attempts/20260908-tp4-completion-owned-overlap-stage01-r1
```

Source, raw evidence, controller records, caches, and temporary files remained
under the approved mounted `/data00/home/sitian` workspace. The compact
downloaded evidence is:

```text
artifacts/lease_sealed_state_commit_overlap/
  20260908-tp4-completion-owned-overlap-stage01-r1/final_bundle
```

The admission classification was `STRICT_CLEAN`. Every selected GPU was
observed at 0 MiB used memory, 0% utilization, and with no compute process.

| Rank | Physical GPU | UUID |
|---:|---:|---|
| 0 | 2 | `GPU-63c05907-407b-8240-07a0-f38872840867` |
| 1 | 3 | `GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d` |
| 2 | 6 | `GPU-c27f6fd6-8a66-7935-41fd-bd5ccdaced31` |
| 3 | 7 | `GPU-b8ffec62-b437-85f7-3f7d-2cd05bd23e16` |

All four ranks reported:

- host `n232-195-203`;
- NVIDIA A100 80GB PCIe, compute capability 8.0;
- Python 3.11.15;
- PyTorch 2.4.1+cu121;
- CUDA 12.1;
- NVIDIA driver 535.261.03;
- NCCL 2.20.5;
- FP32 collective tensors; and
- BF16 output and state tensors.

## 3. Frozen workload and evidence inventory

The workload used world size 4, active-token groups 1/4/8, hidden size 5,120,
48 linear layers, two warmup pairs, 15 measured pairs per shape, and 15
diagnostic iterations per shape.

| Evidence set | Rows |
|---|---:|
| Diagnostic rows | 180 |
| Formal paired rows | 180 |
| Projected correctness rows | 180 |
| Projected overlap rows | 180 |
| Lifecycle rows | 12 |
| Memory rank rows | 4 |

The sealed compact bundle contains 18 files:

```text
admission.json
cleanup.json
correctness_rows.jsonl
diagnostic_rows.jsonl
environment_manifest.json
gpu_rank_manifest.json
lifecycle_rows.jsonl
local_streaming_independent_verification.json
manifest.json
manifest.sha256
memory_rows.jsonl
overlap_rows.jsonl
paired_rows.jsonl
producer_result.json
remote_independent_verification.json
report.md
source_manifest.json
workload_manifest.json
```

`manifest.sha256` binds the 17 other files, including the terminal manifest
and both independent-verifier receipts.

## 4. Diagnostic control and correctness closure

The diagnostic arm intentionally retained the unsafe event-only behavior. It
failed before the collective result was safely consumable:

| Active tokens | Diagnostic rows | Event-only reduced failures | Event-only final failures | Baseline failures | Completion-owned failures |
|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 41 | 40 | 0 | 0 |
| 4 | 60 | 37 | 35 | 0 | 0 |
| 8 | 60 | 44 | 44 | 0 | 0 |
| **Total** | **180** | **122** | **119** | **0** | **0** |

This reproduces the diagnostic premise: recording and waiting on a CUDA event
after asynchronous submission is not a sufficient ownership boundary for the
returned distributed `Work`.

The completion-owned formal arm calls `Work.wait()` from the consumer-stream
context before publishing the dependent state. All 180 formal rows passed
each of the following checks:

- independent expected reduction exact;
- baseline and candidate reduced results exact;
- baseline and candidate final outputs exact;
- baseline/candidate equality;
- shadow payload exact;
- active state preserved before publish;
- published state exact;
- abort preserved old state;
- commit identity matched across ranks;
- collective wait invoked;
- collective dependency transferred;
- side-effect dependency joined;
- finite output;
- no timeout; and
- zero timed-path allocations.

All 12 standalone lifecycle rows passed the corresponding publish, abort,
identity, dependency-transfer, and completion-ownership checks.

## 5. Critical-path latency

For each measured pair, the critical latency is the maximum rank latency.
P90/P95/P99 use the frozen nearest-rank rule. “Classifier speedup” is
`1 - candidate_median / baseline_median`; positive is faster. “Median paired
speedup” is the median of the 15 individual paired speedups and is included to
show pair heterogeneity.

| Tokens | Baseline median | Candidate median | Absolute median delta | Baseline P90 | Candidate P90 | Baseline P95 | Candidate P95 | Baseline P99 | Candidate P99 | Classifier speedup | Median paired speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 401.919 us | 365.568 us | -36.351 us | 597.119 us | 669.279 us | 607.007 us | 1,863.360 us | 607.007 us | 1,863.360 us | +9.044360% | +10.828052% |
| 4 | 347.680 us | 447.360 us | +99.680 us | 609.791 us | 657.407 us | 833.151 us | 916.480 us | 833.151 us | 916.480 us | -28.670041% | -10.001668% |
| 8 | 328.703 us | 362.495 us | +33.792 us | 619.871 us | 666.912 us | 742.464 us | 681.119 us | 742.464 us | 681.119 us | -10.280405% | +17.208926% |

The active-token 8 paired median is positive despite a negative
ratio-of-medians result because the paired ratios are heterogeneous and are
not the frozen classifier statistic. The frozen classifier uses the
ratio-of-medians result.

| Tokens | P99 regression | Improving pairs |
|---:|---:|---:|
| 1 | +206.975043% | 8/15 |
| 4 | +10.001668% | 7/15 |
| 8 | -8.262353% | 8/15 |

The geometric aggregate speedup over the required active-token 4 and 8 groups
was `-19.120881%`.

## 6. Overlap realization

The collective-outstanding window begins after producer readiness and ends
when completion ownership makes the collective visible. The side-effect
interval covers the dependent state copy. Their intersection is the realized
overlap. The table reports row-level medians across 60 rank/pair rows per
shape; the gate uses the minimum-rank overlap ratio per pair and then its
median across 15 pairs.

| Tokens | Median collective-outstanding window | Median side-effect interval | Median overlap intersection | Gate median realized overlap |
|---:|---:|---:|---:|---:|
| 1 | 282.625 us | 20.993 us | 20.993 us | 100.000000% |
| 4 | 261.632 us | 19.456 us | 19.456 us | 100.000000% |
| 8 | 272.3835 us | 19.456 us | 19.456 us | 100.000000% |

All shapes exceed the frozen 20% overlap threshold. This establishes that the
dependent side-effect interval is contained in the outstanding collective
window; it does not establish a faster total critical path.

## 7. Host-submission and memory cost

Host submission is aggregated as the maximum rank per pair and then the
median across 15 pairs.

| Tokens | Baseline host median | Candidate host median | Regression |
|---:|---:|---:|---:|
| 1 | 186.206 us | 370.853 us | +99.162755% |
| 4 | 158.935 us | 332.095 us | +108.950200% |
| 8 | 161.188 us | 367.431 us | +127.951833% |

All three shapes exceed the frozen 3% host-submission regression ceiling.

| Tokens | Maximum peak allocated delta | Maximum peak reserved delta | Theoretical shadow bytes |
|---:|---:|---:|---:|
| 1 | 1,386,496 B (1.3223 MiB) | 2,097,152 B (2 MiB) | 13,025,280 B (12.4219 MiB) |
| 4 | 5,079,040 B (4.8438 MiB) | 23,068,672 B (22 MiB) | 52,101,120 B (49.6875 MiB) |
| 8 | 10,158,080 B (9.6875 MiB) | 0 B | 104,202,240 B (99.375 MiB) |

The maximum rank-level allocated delta was 10,158,080 bytes; the maximum
rank-level reserved delta was 23,068,672 bytes. Every reserved delta remained
within theoretical shadow bytes plus the frozen 64 MiB slack, and no
timed-path allocation occurred. Memory did not cause the terminal no-go.

## 8. Frozen gate decision

| Gate | Frozen requirement | Result | Status |
|---|---:|---:|---|
| Correctness and lifecycle | all exact, no timeout, all lifecycle flags true | 180/180 formal rows and 12/12 lifecycle rows pass | PASS |
| Diagnostic reproduction | at least one event-only visibility failure | 122 reduced and 119 final failures | PASS |
| Realized overlap for tokens 4/8 | at least 20% | 100% / 100% | PASS |
| Aggregate median speedup for tokens 4/8 | at least 5% | -19.120881% | **FAIL** |
| Per-shape token 4/8 median | no regression | -28.670041% / -10.280405% | **FAIL** |
| Token-1 median protection | regression at most 1% | improves 9.044360% | PASS |
| P99 protection | regression at most 3% | worst +206.975043% | **FAIL** |
| Host-submission protection | regression at most 3% | +99.162755% to +127.951833% | **FAIL** |
| Directional consistency for tokens 4/8 | at least 11/15 improving pairs | 7/15 and 8/15 | **FAIL** |
| Memory and allocation | bounded reserve; zero timed allocations | within bound; zero timed allocations | PASS |
| Cleanup | `CLEAN` | `CLEAN` | PASS |

The frozen classifier therefore returns `NO_GO_PERFORMANCE`.

## 9. Cleanup and independent verification

Cleanup was `CLEAN`:

- all four process groups were destroyed;
- streams and events were released on all ranks;
- no rank timed out;
- no owned child remained; and
- all three exact-tag scans were empty.

| Authority | Integrity status | Reconstructed classification | Rows |
|---|---|---|---:|
| Producer | terminal | `NO_GO_PERFORMANCE` | 180 formal / 180 diagnostic |
| Remote independent verifier | `PASS` | `NO_GO_PERFORMANCE` | 180 formal / 180 diagnostic |
| Local streaming independent verifier | `PASS` | `NO_GO_PERFORMANCE` | 180 formal / 180 diagnostic |
| Fresh post-seal local `--check-only` | `PASS` | `NO_GO_PERFORMANCE` | 180 formal / 180 diagnostic |

The exact post-seal command is:

```bash
python3 tools/verify_lease_sealed_state_commit_overlap.py \
  artifacts/lease_sealed_state_commit_overlap/20260908-tp4-completion-owned-overlap-stage01-r1/final_bundle \
  --check-only
```

Verifier `PASS` means the artifact hashes, inventory, identities, rows, and
classification reconstruction agree. It does not turn the reconstructed
performance no-go into a performance pass.

## 10. Prompt-to-artifact checklist

| Requirement | Evidence | Status |
|---|---|---|
| Completion ownership implemented | Runtime calls `Work.wait()` from consumer-stream context and emits ownership flags | complete |
| Unsafe control retained only as diagnostic | `diagnostic_rows.jsonl`; event-only failures reproduced | complete |
| Independent correctness oracle | expected reduction compared with both arms | complete |
| Exact formal correctness | all exactness fields pass on 180/180 rows | complete |
| Lifecycle closure | 12/12 lifecycle rows pass | complete |
| Frozen TP4 shapes and pair count | tokens 1/4/8, 15 measured pairs, four ranks | complete |
| Real overlap measured | event windows, intersection, and ratio in 180 rows | complete |
| Benefit and cost together | critical latency, tail, host, overlap, and memory in Sections 5–7 | complete |
| Strict-clean four-GPU admission | admission and UUID/rank manifests | complete |
| Immutable source identity | pushed revision and source-tree hash in every evidence family | complete |
| Safe remote storage | attempt and runtime paths below `/data00/home/sitian` | complete |
| Compact-only download | local attempt contains controller receipts plus sealed `final_bundle` | complete |
| Cleanup | process groups, streams, events, children, and tag scans clean | complete |
| Independent verification | producer plus remote, local, and post-seal reconstruction agree | complete |
| Performance qualification | frozen latency, tail, host, and directional gates | **failed** |
| Qwen3.8 evidence | no model integration or end-to-end workload was run | not established |
| Stage-1 authorization | `stage1_authorized=false` | **prohibited** |

## 11. Final claim boundary and next action

The defensible conclusion is:

```text
completion ownership fixes the Stage-0 correctness defect
the frozen microgate realizes overlap
the added completion and submission work loses on protected performance gates
the terminal result is NO_GO_PERFORMANCE
```

Do not claim a TinyLLMForge Qwen3.8 performance improvement from this result.
Do not proceed to Stage-1 or modify Qwen layers, model transactions, or
`tinyvllm/layers/linear.py`.

The frozen next action for a non-GO result is:

```text
stop this mechanism and design a larger-granularity TP path
```

Any future route requires a separately reviewed design, a new source revision,
fresh frozen gates, and a new immutable attempt tag. The r1 bundle must remain
unchanged.
