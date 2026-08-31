# Cross-Request Wavefront Stage-0 Completion Audit

**Date:** 2026-08-31

**Attempt:** `20260831-cross-request-wavefront-stage0-r1`

**Measurement source revision:** `716f0e0acacc487d29b6da223080922833f5fbdf`

**Measurement tree SHA-256:** `bf52a806e1c7f88c66d00a31e69ad500d3c07326cf5dacb4a9c359db73ba9b90`

**Tolerance-fix analyzer revision:** `65b9b1d9ec8166350cd6c00e81bf6ab22da4a214`

**Terminal direction verdict:** `NO_GO_INSUFFICIENT_OVERLAP`

**Stage-1 integration:** prohibited

## 1. Executive conclusion

The isolated four-A100 real-shape gate does not support cross-request
wavefront integration.

The candidate was substantially slower than the unchanged NCCL baseline:

| Active tokens | Baseline median | Candidate median | Median change | Baseline P99 | Candidate P99 | P99 change |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 210.6395 us | 441.1355 us | +109.43% slower | 696.319 us | 1,145.696 us | +64.54% |
| 8 | 227.583 us | 482.687 us | +112.09% slower | 310.431 us | 989.856 us | +218.87% |

Realized overlap covered only `10.9409%` and `17.4066%` of the candidate
communication interval, below the frozen `20%` minimum for both shapes.
Host submission also regressed by `230.69%` and `199.52%`.

This is a terminal Stage-0 no-go. The production runtime, scheduler,
`tinyvllm/layers/linear.py`, and model files remain unchanged. No Stage-1
integration plan is authorized.

## 2. Classification reconciliation

The immutable producer bundle emitted `NO_GO_CORRECTNESS`, and both the
remote and downloaded local verifier reproduced that classification under
measurement revision `716f0e0...`.

Post-run inspection found that this was a classifier false negative:

- tokens 8 maximum candidate-versus-baseline absolute error was
  `0.00390625`, below frozen `atol=0.02`;
- maximum relative error was `0.03448275849223137`, caused by values near
  zero;
- the original classifier incorrectly required absolute and relative maxima
  to pass independently;
- cross-rank maximum absolute and relative errors were both exactly zero;
- tokens 4 candidate-versus-baseline errors were both exactly zero; and
- all NaN and Inf counts were zero.

Revision `65b9b1d9...` added a RED regression test for the observed
near-zero case and corrected the fail condition. Reconstructing the original
hash-bound rows with that revision yields:

```text
NO_GO_INSUFFICIENT_OVERLAP
```

The reconciliation does not mutate the original bundle and does not rerun
or retune the workload. It is bound to these inputs:

| Input | SHA-256 |
|---|---|
| `microgate_rows.jsonl` | `859e769a419066625fdaed6de9eba6ccc0cc363d0c1dea9428e6a81600e2002a` |
| `memory_summary.json` | `0e12fb8e8fa7f24e9c68669d125ebeacf0c61b95cfcf8c96070815ac802da16c` |
| `cleanup.json` | `817bbe59170d70e8d745aba962e74135908cf820c337b6e3e39c0d11af994585` |
| original `manifest.sha256` | `298e35ba3b7503ace3b15493d2a95efc0139c4b0f9f929c203650bef0a626cd3` |

The compact machine-readable reconciliation is
`controller/posthoc_reconciliation.json`.

## 3. Immutable execution identity

The run began from a clean tracked `tinyvllm/` and `tools/` scope. Before
launch, local HEAD, tracking HEAD, and GitHub branch SHA all matched
`716f0e0acacc487d29b6da223080922833f5fbdf`.

The source archive was produced from that exact committed revision. The
attempt root was:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/attempts/20260831-cross-request-wavefront-stage0-r1
```

Every task path, cache, source archive, log, temporary directory, raw
artifact, and final bundle remained below the approved mounted root.
No task data was written to remote `/`, `/tmp`, a model-cache directory, or
the retired checkout.

Plan-only completed without remote mutation. The immutable attempt then ran
from `2026-08-31T15:22:16+08:00` to terminal local receipt at
`2026-08-31T15:23:04+08:00`.

## 4. GPU admission and runtime

The local controller selected four strict-clean physical GPUs and checked
them again immediately before launch:

| Physical index | UUID | Memory used | Utilization | External compute processes |
|---:|---|---:|---:|---:|
| 0 | `GPU-57be086f-e967-c022-3832-93df4fc77bd0` | 0 MiB | 0% | 0 |
| 2 | `GPU-63c05907-407b-8240-07a0-f38872840867` | 0 MiB | 0% | 0 |
| 3 | `GPU-f8904cb4-f9f0-c757-df36-e6fd971b3a9d` | 0 MiB | 0% | 0 |
| 4 | `GPU-56b882d2-6e6e-adb3-80e7-95f0a9e678f1` | 0 MiB | 0% | 0 |

CUDA-visible local ranks 0 through 3 each reported:

- NVIDIA A100 80GB PCIe;
- compute capability 8.0;
- CUDA 12.1;
- PyTorch 2.4.1+cu121;
- NCCL available;
- BF16 input/output and FP32 accumulation; and
- real shape `1536 -> 5120`.

The supervisor owned PIDs `448636`, `448637`, `448638`, and `448639`.
All four exited zero. Four resource samples passed, no external-process
violation occurred, and no process was terminated or adopted by the
controller.

## 5. Inventory and order evidence

The compact row inventory is complete:

- active-token groups: 4 and 8;
- warmup pairs: 2 per shape;
- measured pairs: 300 per shape;
- ranks: 4;
- total measured rows: `2 * 300 * 4 = 2,400`;
- unique `(shape, pair, rank)` identities: 2,400;
- baseline-first rows: 1,200;
- candidate-first rows: 1,200;
- one stable cohort digest per shape;
- one stable collective-order digest across all rows;
- one output digest per shape; and
- zero rank-output digest disagreements.

The fixed collective-order digest is:

```text
09f244ffbcb71fa328a2ecdc3fc123105caa03fc25fdba0fb1a9cf029173aeff
```

## 6. Benefit and cost

### 6.1 Active tokens 4

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| CUDA P50 | 210.6395 us | 441.1355 us | +109.43% |
| CUDA P90 | 242.239 us | 489.984 us | — |
| CUDA P95 | 250.496 us | 511.391 us | — |
| CUDA P99 | 696.319 us | 1,145.696 us | +64.54% |
| Host submission P50 | 132.3185 us | 437.564 us | +230.69% |
| Host submission P99 | 620.353 us | 1,014.548 us | — |

Communication union totaled `272,365,805 ns`; realized overlap totaled
`29,799,388 ns`, or `10.940943%`.

### 6.2 Active tokens 8

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| CUDA P50 | 227.583 us | 482.687 us | +112.09% |
| CUDA P90 | 260.511 us | 525.088 us | — |
| CUDA P95 | 265.536 us | 533.760 us | — |
| CUDA P99 | 310.431 us | 989.856 us | +218.87% |
| Host submission P50 | 146.6115 us | 439.1325 us | +199.52% |
| Host submission P99 | 175.021 us | 758.620 us | — |

Communication union totaled `363,234,594 ns`; realized overlap totaled
`63,226,821 ns`, or `17.406608%`.

### 6.3 Memory cost

Maximum measured peak allocated delta was `26,296,832 bytes`
(`25.0786 MiB`), below the `128 MiB` ceiling. Maximum reserved-memory delta
was zero.

The direct preallocated tensor delta was:

- tokens 4: `491,520 bytes` (`0.46875 MiB`) per rank;
- tokens 8: `983,040 bytes` (`0.9375 MiB`) per rank.

Memory was acceptable, but it does not offset the latency, tail,
host-submission, and overlap failures.

## 7. Correctness and cleanup

| Shape | Cross-rank max abs | Cross-rank max rel | Baseline max abs | Baseline max rel | NaN | Inf |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0.00390625 | 0.03448275849 | 0 | 0 |

The tokens-8 absolute error is below the frozen absolute tolerance. The
corrected classifier therefore does not reject numerical parity.

Cleanup was `CLEAN`:

- all streams released;
- all events released;
- all four process groups destroyed;
- no timeout;
- no owned child remained; and
- three exact-tag scans were empty.

## 8. Producer, verifier, and manifest evidence

Under measurement revision `716f0e0...`:

- producer classification: `NO_GO_CORRECTNESS`;
- remote independent verifier: `PASS`, reconstructing
  `NO_GO_CORRECTNESS`;
- downloaded local independent verifier: `PASS`, reconstructing
  `NO_GO_CORRECTNESS`;
- artifact hashes verified: true; and
- measurement row count: 2,400.

This proves the original bundle is internally consistent. It does not make
the original tolerance interpretation correct. The post-hoc reconciliation
is intentionally outside the immutable producer manifest and binds itself
to the original manifest and input hashes.

## 9. Prompt-to-artifact checklist

| Requirement | Concrete evidence | Status |
|---|---|---|
| Model-neutral Stage-0 contract | `tools/cross_request_wavefront_overlap.py`; no torch/model import | complete |
| Real four-rank worker | `tools/cross_request_wavefront_microgate_worker.py`; supervisor exit codes 0/0/0/0 | complete |
| Two cohorts, fixed collective order | `cohort_policy.json`; stable row digests | complete |
| Shapes 4 and 8, real `1536 -> 5120` dimensions | `runtime_capabilities.json`; 2,400 raw rows | complete |
| Two warmups and 300 AB/BA pairs | worker schedule tests; exact row inventory | complete |
| No timed-path allocation/device-wide sync | source inspection tests | complete |
| Benefit and cost reported together | Sections 6.1 through 6.3 | complete |
| Correctness, NaN, Inf, digest evidence | raw rows and Section 7 | complete after tolerance reconciliation |
| At least 20% realized overlap | 10.94% and 17.41% | failed |
| Median speedup at least 5% / 8% | -109.43% / -112.09% speedup | failed |
| P99 regression at most 3% | +64.54% / +218.87% | failed |
| Host regression at most 10% | +230.69% / +199.52% | failed |
| Added allocation at most 128 MiB | 25.0786 MiB | passed |
| Four strict-clean GPUs twice | launch admission plus pre-launch controller check | complete |
| No external task termination/adoption | supervisor receipt and empty violations | complete |
| All remote writes under mounted root | plan and attempt path receipts | complete |
| Producer plus remote/local verifier | immutable final bundle and controller result | complete |
| Compact local evidence only | 2.96 MB row bundle plus small receipts; no profiler trace | complete |
| Stage-1 only after GO | reconciled non-GO; production runtime untouched | prohibited |

## 10. Missing or weaker evidence

The worker did not persist explicit cross-rank start-skew and completion-skew
fields, although the design listed them as diagnostic measurements. This is
a coverage gap. It does not weaken the no-go decision because the candidate
already fails the primary median, P99, host-submission, and overlap gates by
large margins.

The bundle's runtime capability rows report CUDA-visible local device
indices, not physical indices or UUIDs. Physical identity is instead bound
by the controller admission receipt. Future gates should carry the selected
physical index and UUID into each runtime row.

No real-model E2E speedup was attempted or established because Stage 0 did
not pass.

## 11. Final classification and claim boundary

The corrected terminal mechanism classification is:

```text
NO_GO_INSUFFICIENT_OVERLAP
```

Both shape medians are also below the frozen `3%` stop threshold, so the
complete cross-request wavefront direction stops here. No bounded scheduling
refinement, Stage-1 integration plan, production implementation, or Qwen3.8
E2E claim is authorized.

The supported claim is limited to:

> On one frozen four-A100 PCIe real-shape TP4 microgate, the implemented
> two-cohort NCCL wavefront realized only 10.9% to 17.4% overlap and more
> than doubled median transaction latency. The direction was rejected before
> model integration.

This audit does not claim results for an end-to-end model, other checkpoints,
other tensor-parallel sizes, NVLink systems, multi-node systems, stochastic
sampling, speculative decoding, CUDA Graphs, KV offload, or production
serving.
