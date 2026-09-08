# Lease-Sealed State-Commit / AllReduce Overlap Stage-0 Audit

**Audit date:** 2026-09-08

**Terminal attempt:** `20260908-lease-sealed-state-commit-overlap-stage0-r5`

**Source revision:** `20afc4174ae30187f3a73246049dde4860c2146e`

**Source tree SHA-256:** `8610eb6aadba787161ddb3c882cbfd5966dc7fde24b3d035914a5f8aa937ff02`

**Terminal classification:** `NO_GO_CORRECTNESS_OR_LIFECYCLE`

**Stage-1 integration:** prohibited

## 1. Executive conclusion

The model-neutral four-GPU Lease-Sealed State-Commit / AllReduce Overlap
primitive does not qualify for Qwen3.8 integration under the frozen Stage-0
contract.

The terminal r5 attempt completed the full evidence path:

- all four workers exited zero;
- all 180 measurement rows were present;
- cleanup was `CLEAN`;
- the producer returned `NO_GO_CORRECTNESS_OR_LIFECYCLE`;
- the remote independent verifier returned `PASS` and reconstructed the same
  no-go classification; and
- the downloaded local independent verifier returned `PASS` and reconstructed
  the same no-go classification.

The failure is not a lifecycle-state failure. All 12 rank/shape lifecycle
rows passed active-state preservation, publish, abort, and commit-identity
checks. The correctness failure is in collective result visibility:

- `reduced_output_exact` failed on 12 of 180 rows; and
- `final_output_exact` failed on 176 of 180 rows.

The timing rows are retained as diagnostics only. They cannot establish a
performance win because correctness failed first. They also show zero median
realized overlap for every shape and a 67.95% to 77.87% host-submission
regression.

## 2. Attempt reconciliation

| Attempt | Frozen source | Terminal boundary |
|---|---|---|
| r1 | `68ed63b57790ebf98c228da70205acf6b8c658cd` | Environment failure after foreign GPU-process detection; cleanup `DIRTY`; no producer or verifier |
| r2 | pre-`63831cc` source | Source staging failed on SSH return code 255; no worker |
| r3 | `63831cc9795215e388a778f467b5d6ce936c7ace` | Workers and cleanup passed; assembler rejected missing runtime GPU UUID |
| r4 | `ff34614a99689bcb3ff450b8ca629c99c8d83b69` | Workers and cleanup passed; assembler rejected nested lifecycle rank wrappers |
| r5 | `20afc4174ae30187f3a73246049dde4860c2146e` | Full producer and dual-verifier path completed; terminal correctness no-go |

Each attempt remains immutable. No prior attempt was repaired or reclassified
with newer source.

## 3. Immutable execution identity and storage

Before r5 launch, local HEAD, the tracking branch, and the GitHub branch all
matched `20afc4174ae30187f3a73246049dde4860c2146e`. The tracked `tinyvllm/` and
`tools/` scope was clean.

The remote attempt root was:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/attempts/20260908-lease-sealed-state-commit-overlap-stage0-r5
```

The plan placed source, raw evidence, the final bundle, controller records,
`TMPDIR`, `XDG_CACHE_HOME`, `TORCH_EXTENSIONS_DIR`, and `CUDA_CACHE_PATH`
below the approved mounted `/data00/home/sitian` root. No task path was
placed under remote `/` or `/tmp`.

The compact downloaded evidence is:

```text
artifacts/lease_sealed_state_commit_overlap/
  20260908-lease-sealed-state-commit-overlap-stage0-r5/final_bundle
```

## 4. GPU admission and runtime identity

The controller acquired four simultaneous strict-clean A100 GPUs. Every
admission row recorded 0 MiB used memory, 0% utilization, and no compute
process.

| Rank | Physical GPU | UUID |
|---:|---:|---|
| 0 | 2 | `GPU-63c05907-407b-8240-07a0-f38872840867` |
| 1 | 3 | `GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d` |
| 2 | 4 | `GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1` |
| 3 | 6 | `GPU-c27f6fd6-8a66-7935-41fd-bd5ccdaced31` |

All ranks reported:

- NVIDIA A100 80GB PCIe;
- compute capability 8.0;
- Python 3.11.15;
- PyTorch 2.4.1+cu121;
- CUDA 12.1;
- NVIDIA driver 535.261.03;
- NCCL 2.20.5;
- FP32 collective tensors; and
- BF16 output and state tensors.

## 5. Workload and inventory

The frozen workload used:

- world size 4;
- active-token groups 1, 4, and 8;
- hidden size 5,120;
- two warmup pairs per shape;
- 15 measured AB/BA pairs per shape; and
- 180 unique `(active_tokens, pair_index, rank)` rows.

The supervisor receipt records owned PIDs `2115321`, `2115322`, `2115323`,
and `2115324`, rank exit codes `[0, 0, 0, 0]`, five resource snapshots, no
resource-identity violation, and no missing artifact.

## 6. Correctness and lifecycle

| Active tokens | Rows | Reduced-output failures | Final-output failures | Shadow failures | Lifecycle failures |
|---:|---:|---:|---:|---:|---:|
| 1 | 60 | 8 | 56 | 0 | 0 |
| 4 | 60 | 4 | 60 | 0 | 0 |
| 8 | 60 | 0 | 60 | 0 | 0 |
| **Total** | **180** | **12** | **176** | **0** | **0** |

Across all 180 rows:

- `shadow_payload_exact` was true;
- `active_state_preserved_before_publish` was true;
- `published_state_exact` was true;
- `abort_preserved_old_state` was true;
- `commit_identity_match` was true;
- `finite_output` was true; and
- `timed_out` was false.

All 12 standalone lifecycle rows also passed. Therefore the frozen classifier
correctly applies its highest-priority correctness no-go.

### Diagnostic root-cause boundary

The leading implementation-level hypothesis is incomplete ownership transfer
from the asynchronous NCCL `Work` to the stream that consumes
`candidate_result`.

The candidate records a CUDA event after calling
`dist.all_reduce(..., async_op=True)`, but the returned collective `Work`
remains the authoritative completion object. The observed pattern is
consistent with the BF16 output copy sometimes consuming the result before
the collective is globally visible: the final output fails much more often
than the later FP32 reduced-result check.

This is a diagnostic inference from the frozen source and r5 rows, not a
post-hoc reclassification and not a validated repair. Any redesign requires a
separate plan, a new source revision, and a fresh attempt.

## 7. Diagnostic benefit and measured cost

Because exact correctness failed, none of the latency values below are
eligible performance evidence. They are included to report benefit and cost
together and to show that the candidate also fails the intended overlap
mechanism.

| Tokens | Baseline median | Candidate median | Diagnostic median change | Baseline P99 | Candidate P99 | P99 change | Median overlap | Host submission change | Improving pairs |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 257.824 us | 198.623 us | 22.96% faster | 1,911.808 us | 256.319 us | 86.59% lower | 0.00% | 77.49% slower | 11/15 |
| 4 | 247.071 us | 192.351 us | 22.15% faster | 339.967 us | 241.344 us | 29.01% lower | 0.00% | 77.87% slower | 12/15 |
| 8 | 288.767 us | 186.368 us | 35.46% faster | 354.303 us | 226.400 us | 36.10% lower | 0.00% | 67.95% slower | 15/15 |

The apparent latency improvement is not trustworthy as a mechanism benefit:
the candidate output is not exact, median realized overlap is zero, and host
submission regresses substantially.

### Memory cost

| Tokens | Maximum peak allocated delta | Maximum peak reserved delta | Theoretical shadow bytes |
|---:|---:|---:|---:|
| 1 | 1,033,216 B (0.9854 MiB) | 2,097,152 B (2 MiB) | 13,025,280 B (12.4219 MiB) |
| 4 | 4,128,256 B (3.9370 MiB) | 20,971,520 B (20 MiB) | 52,101,120 B (49.6875 MiB) |
| 8 | 8,254,976 B (7.8726 MiB) | 0 B | 104,202,240 B (99.375 MiB) |

No timed-path allocation was reported, and the aggregate memory gate did not
cause the terminal classification.

## 8. Cleanup, producer, verifiers, and manifest

Cleanup was `CLEAN`:

- all four process groups were destroyed;
- streams and events were released on all ranks;
- no rank timed out;
- no owned child remained; and
- all three exact-tag scans were empty.

The evidence chain is:

| Authority | Status | Classification |
|---|---|---|
| Producer | terminal | `NO_GO_CORRECTNESS_OR_LIFECYCLE` |
| Remote independent verifier | `PASS` | `NO_GO_CORRECTNESS_OR_LIFECYCLE` |
| Local streaming independent verifier | `PASS` | `NO_GO_CORRECTNESS_OR_LIFECYCLE` |

Both verifiers checked artifact hashes and reconstructed all 180 rows. A
fresh local `--check-only` verification on 2026-09-08 also returned `PASS`
with the same classification.

## 9. Prompt-to-artifact checklist

| Requirement | Evidence | Status |
|---|---|---|
| Generic mechanism only | Runtime primitive and worker; no Qwen or `linear.py` integration | complete |
| RED then GREEN fixes | Source-staging, UUID, and lifecycle aggregation regressions; final suite `63 passed` | complete |
| Three frozen shapes | 180 unique rank/pair/shape rows | complete |
| Exact correctness | 12 reduced-output and 176 final-output failures | **failed** |
| Real overlap | Event intervals exist; median realized overlap is 0% for all shapes | **failed** |
| Benefit and cost | Section 7 reports latency, host, overlap, and memory together | complete |
| Strict-clean TP4 | Admission plus four UUID/rank rows | complete |
| Safe storage | All remote paths below approved `/data00/home/sitian` root | complete |
| Immutable source | Pushed revision and tree hash in the bundle | complete |
| Complete producer bundle | Exact terminal inventory and sealed manifests | complete |
| Independent evidence | Remote and local verifiers both `PASS` | complete |
| Clean lifecycle | Cleanup `CLEAN`; lifecycle rows all pass | complete |
| Claim boundary | Mechanism-only; no Qwen3.8 end-to-end claim | complete |
| Stage-1 authorization | Producer `stage1_authorized=false` | **prohibited** |
| Repository publication | Audit and handoff commit | complete in this publication commit |

## 10. Final claim boundary and next action

This result is a complete negative Stage-0 qualification:

```text
mechanism implementation complete
evidence pipeline complete
correctness qualification failed
real overlap qualification failed
Qwen3.8 integration prohibited
```

Do not describe r5 as a latency or throughput win. Do not modify Qwen layers,
model transactions, or `tinyvllm/layers/linear.py` from this result.

The frozen plan's next action is:

```text
stop mechanism and preserve terminal evidence
```

A future attempt is permitted only under a separately reviewed redesign that
establishes correct asynchronous collective completion ownership. It must use
a new source revision and a fresh immutable attempt tag; it must not mutate,
repair, or reclassify r5.
