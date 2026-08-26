# Context-Gated Elastic Exact-Burst Audit

**Audit date:** 2026-08-26

**Authoritative checkout:** `/Users/bytedance/Desktop/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

**Terminal source commit:** `1e7ecaf85df81c7f52de1ce8fcaef098bd30c05d`

**Terminal decision:** `GO_CONTEXT_GATED_ELASTIC_EXACT_BURST`

**Promotion:** `AUTHORIZED_WITHIN_FROZEN_STAGE_1_SCOPE_DEFAULT_DISABLED`

## Audit Rule

This audit treats checked-in source, raw benchmark rows, float32 logit
sidecars, source manifests, producer output, remote independent verification,
download receipts, and frozen-source local verification as separate evidence
layers. A partial run, a producer-only result, or a verifier run against the
live checkout is not terminal evidence.

The gate may promote only the frozen context-gated K8/K16 policy described in
`docs/superpowers/specs/2026-08-24-context-gated-elastic-exact-burst-design.md`.
It does not authorize threshold retuning, Qwen3-8B claims, multi-sequence or
tensor-parallel claims, streaming/EOS-aware claims, or default enablement.

## Prompt-to-Artifact Checklist

| Requirement | Source contract | Test evidence | Runtime artifact | Status |
| --- | --- | --- | --- | --- |
| Context threshold is exactly 2,048 tokens | `tinyvllm/engine/exact_greedy_decode_burst.py`; terminal producer and verifier | selector boundary tests plus terminal threshold tests | all eligible 256/2,048 rows select K16; all 4,096/8,192 rows select K8 | `PASS` |
| Elastic mode is default-disabled and depends on fixed K8 without split phase | `tinyvllm/config.py`; `tinyvllm/engine/llm_engine.py` | config and engine wiring tests in `tools/test_exact_greedy_decode_burst.py` and `tools/test_llm_engine_exact_greedy_decode_burst.py` | frozen source manifests bind both files | `PASS` |
| K16 requires at least sixteen output tokens and writable positions | runtime selector and lease validation | focused elastic selector/runtime tests | eligible rows request seven K16 bursts and finish with two expected K8/output-budget clips | `PASS` |
| Fixed K8 remains unchanged and reports no elastic counters | runtime plus terminal lifecycle classifier | verifier tamper test rejects fixed-K8 elastic counters | fixed-K8 summaries in all 20 fixed-policy performance rows | `PASS` |
| One shared complete-token graph owns both widths | `tinyvllm/engine/exact_greedy_decode_burst.py` and capture receipt | shared-graph CPU tests | one graph/capture receipt per row; elastic incremental capture and retained bytes are zero | `PASS` |
| Width-scoped and graph-scoped failures fail closed | runtime K16 health, graph quarantine, lease cancellation, and bounded rollback paths | focused fault-injection, stale-generation, rollback, and verifier lifecycle tests | zero failure, rollback, pending lease, or quarantine in terminal rows | `PASS` |
| Width-aware `{8,16}` transaction binds requested and authorized width | scheduler/runtime lease digest and one-phase journal | serialization, stale identity, block-boundary, continuation, and commit tests | requested/authorized histograms and committed widths reconstruct exactly | `PASS` |
| Exact 40 performance rows | profiler and terminal producer/verifier | incomplete/duplicate inventory tests | `performance_rows.jsonl`: 40 unique rows | `PASS_40_OF_40` |
| Exact 32 correctness rows and float32 sidecars | profiler and terminal producer/verifier | sidecar, row-inventory, token and derived-artifact tamper tests | `correctness_rows.jsonl` plus 32 referenced sidecars | `PASS_32_OF_32` |
| Exact token, text, argmax, and logit parity | producer and independent verifier | valid fixture and tamper tests | terminal summary plus raw correctness rows/sidecars | `PASS_EXACT` |
| One target forward and one graph replay per emitted decode token | terminal runtime-inventory checks | terminal classification tests | 2,540 forwards = 2,540 replays = 2,540 committed decode tokens | `PASS` |
| Zero intermediate token D2H and one final token D2H per burst | terminal runtime-inventory checks | terminal classification tests | zero intermediate calls; 250 final calls for 250 committed bursts; 20,320 bytes | `PASS` |
| Zero unexpected fallback, rollback, failure, or quarantine | terminal lifecycle checks | unexpected fallback/rollback/quarantine and fixed-K8 tamper tests | only expected context/output-budget K8 fallback; all exceptional counters zero | `PASS` |
| Eligible aggregate median TPOT improvement at least 2% | terminal classifier | exact-threshold classification tests | `2.848048%` | `PASS` |
| Eligible aggregate P95 TPOT improvement at least 1% | terminal classifier | exact-threshold classification tests | `2.128914%` | `PASS` |
| Per-context median/P95 TPOT regression at most 2% | terminal classifier | independent boundary tests | worst median regression `0.129540%`; worst P95 regression `0.161586%` | `PASS` |
| TTFT, E2E, and TPOT-P99 regression at most 2% | terminal classifier | independent boundary tests | worst regressions: TTFT `0.983500%`, E2E `0.153544%`, TPOT P99 `1.527992%` | `PASS` |
| Throughput regression at most 1% | terminal classifier | independent boundary test | worst per-context regression `0.153544%` | `PASS` |
| Allocated/reserved CUDA-memory regression at most 3% | terminal classifier | independent boundary tests | worst regressions: allocated `0%`, reserved `0.456300%` | `PASS` |
| Maximum selected-K16 host-visible gap at most 40 ms | terminal classifier | exact-threshold and inconsistent-width tests | maximum `37.006666 ms`; P95 `36.914469 ms` | `PASS` |
| Benefit and cost are both reported | terminal summary schema | valid synthetic reconstruction test | tables below report TPOT, throughput, TTFT, E2E, gap, capture, retained bytes, memory, fallback rate, and lifecycle counters | `PASS` |
| Source tree is the pushed branch HEAD with an empty patch | remote controller, source manifests, terminal manifest | controller source-binding tests and source tamper tests | source `1e7ecaf...`; empty-patch SHA256 `e3b0c442...`; local and origin SHA matched at launch | `PASS` |
| Producer, remote verifier, and frozen-source local verifier agree | terminal producer, independent verifier, remote controller | verifier independence/tamper tests and controller routing tests | all three report `GO_CONTEXT_GATED_ELASTIC_EXACT_BURST`, 40 + 32 rows, same source | `PASS` |
| Manifest and downloaded bundle are complete and hash-exact | terminal manifest and controller download receipt | independent manifest verifier plus fresh local inventory rehash | 42 terminal artifacts and 938 downloaded primary/controller/source entries rehashed | `PASS` |
| Remote writes stay under `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818` | remote controller path validation | strict path tests | preflight records staging, primary, and controller paths under the approved root | `PASS` |
| GPU is strict-clean immediately before launch | remote controller admission and recheck | monitor/recheck ordering tests | A100 GPU 1, 0 MiB, 0% utilization, no compute process | `PASS` |
| Kerberos has at least 5,400 seconds remaining before launch | remote controller fail-fast guard | controller lifecycle tests | preflight status `PASS`, 35,989 seconds remaining | `PASS` |
| Claim remains inside the frozen Stage-1 boundary | approved design and terminal classifier | workload-contract and source-bound verifier tests | Qwen3-0.6B, TP1, batch one, completion-only, temperature zero, ignore-EOS | `PASS_SCOPED` |

## Frozen Ceiling Checkpoint

The prerequisite ceiling is complete and authorizes the terminal gate:

```text
run tag:                       20260825-context-gated-elastic-k16-ceiling-r5
source commit:                 f4acc4aded4b91bdb2f8ae30ebf21a0c8944443f
producer classification:       CEILING_GO
remote verifier:               PASS / CEILING_GO
frozen-source local verifier:  PASS / CEILING_GO
performance rows:              24 / 24
correctness rows:              32 / 32
256 median TPOT improvement:   3.249202173719195%
2048 median TPOT improvement:  3.216560024935232%
maximum selected-K16 gap:      36,984,111 ns
```

Canonical local evidence:

```text
artifacts/context_gated_elastic_exact_burst/
  20260825-context-gated-elastic-k16-ceiling-r5/
```

The retained `recovery-stale-relative-verifier-pycache/` directory is
diagnostic recovery evidence and is not part of the source, primary, or
controller hash inventories.

## Terminal Implementation Verification

The terminal gate implementation is pushed at
`1e7ecaf85df81c7f52de1ce8fcaef098bd30c05d`.

Fresh local verification before that commit:

```text
focused producer/verifier/controller plus adjacent gate suites:
  72 passed in 1.96s
full elastic plan regression set:
  455 passed, 1 skipped in 7.10s
py_compile:
  PASS
git diff --cached --check:
  PASS
```

The local code review covered the six Task 6 files. It found and fixed:

1. malformed performance measurements being misclassified as correctness
   failures instead of being rejected;
2. incomplete independent-verifier validation of non-negative measurements,
   output-token types, and workload execution fields; and
3. fixed-K8 rows being able to carry elastic counters without invalidating the
   lifecycle gate.

The post-fix review found no remaining P0-P2 defects. The local report is:

```text
/tmp/TinyLLMForge_task6_review_vmRkHyY1/report.html
```

## Terminal Execution and Evidence Closure

The locally hosted monitor observed the externally refreshed ticket and
launched the controller. It did not run `kinit`, and all remote run data stayed
under the approved mounted root.

```text
run tag:                       20260825-context-gated-elastic-k16-terminal-r1
source commit:                 1e7ecaf85df81c7f52de1ce8fcaef098bd30c05d
source patch:                  empty
empty-patch SHA256:            e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
remote task root:              /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
Kerberos preflight:            PASS / 35,989 seconds remaining
selected GPU:                  A100 80GB GPU 1 / 0 MiB / 0% / no process
worker exit code:              0
performance rows:              40 / 40
correctness rows:              32 / 32
logit sidecars:                32 / 32
producer classification:       GO_CONTEXT_GATED_ELASTIC_EXACT_BURST
remote independent verifier:   PASS / GO_CONTEXT_GATED_ELASTIC_EXACT_BURST
frozen-source local verifier:  PASS / GO_CONTEXT_GATED_ELASTIC_EXACT_BURST
download receipt:              DOWNLOADED_AND_VERIFIED
```

A fresh local verifier reconstructed the summary and classification from raw
rows and float32 sidecars. A separate inventory pass rehashed all 938 entries
in the download receipt—43 primary, 8 controller, and 887 frozen-source
files—including every chunk, and rehashed all 42 artifacts named by the
terminal manifest.

Canonical local evidence:

```text
artifacts/context_gated_elastic_exact_burst/
  20260825-context-gated-elastic-k16-terminal-r1/
```

## Measured Benefit

Negative values in a `regression` column are improvements. Positive values in
an `improvement` column are improvements.

| Scope | TPOT median | TPOT P95 | TPOT P99 | TTFT median | E2E median | Throughput |
| --- | --- | --- | --- | --- | --- | --- |
| eligible 256+2048 | 2.371205 -> 2.303672 ms (`+2.848048%` improvement) | 2.464854 -> 2.412379 ms (`+2.128914%` improvement) | 2.557349 -> 2.537232 ms (`-0.786624%` regression) | 36.533589 -> 35.691143 ms (`-2.305949%` regression) | 330.671955 -> 319.621118 ms (`-3.341934%` regression) | 387.762108 -> 401.449946 tok/s (`-3.409600%` regression) |
| overall | 2.554675 -> 2.555060 ms (`-0.015046%` improvement) | 2.946645 -> 2.954000 ms (`-0.249606%` improvement) | 3.105961 -> 3.103142 ms (`-0.090747%` regression) | 43.392683 -> 42.200426 ms (`-2.747601%` regression) | 362.557139 -> 356.726129 ms (`-1.608301%` regression) | 353.455492 -> 359.807916 tok/s (`-1.765504%` regression) |
| 256 | 2.161301 -> 2.085549 ms (`+3.504920%` improvement) | 2.252415 -> 2.237788 ms (`+0.649411%` improvement) | 2.340364 -> 2.273681 ms (`-2.849275%` regression) | 33.349347 -> 33.350580 ms (`+0.003697%` regression) | 309.742147 -> 300.227020 ms (`-3.071951%` regression) | 413.246958 -> 426.344038 tok/s (`-3.071951%` regression) |
| 2048 | 2.413640 -> 2.339140 ms (`+3.086612%` improvement) | 2.526752 -> 2.528455 ms (`-0.067404%` improvement) | 2.559045 -> 2.598147 ms (`+1.527992%` regression) | 37.290438 -> 36.589787 ms (`-1.878903%` regression) | 345.570779 -> 335.725797 ms (`-2.848905%` regression) | 370.401688 -> 381.263523 tok/s (`-2.848905%` regression) |
| 4096 | 2.573907 -> 2.577241 ms (`-0.129540%` improvement) | 2.712304 -> 2.716687 ms (`-0.161586%` improvement) | 2.742438 -> 2.733769 ms (`-0.316111%` regression) | 47.352923 -> 47.818639 ms (`+0.983500%` regression) | 376.538247 -> 377.066833 ms (`+0.140380%` regression) | 339.938907 -> 339.462368 tok/s (`+0.140380%` regression) |
| 8192 | 2.886115 -> 2.887160 ms (`-0.036208%` improvement) | 3.101783 -> 3.095603 ms (`+0.199222%` improvement) | 3.150528 -> 3.116484 ms (`-1.080603%` regression) | 106.533364 -> 106.493674 ms (`-0.037256%` regression) | 476.920066 -> 477.652349 ms (`+0.153544%` regression) | 268.388791 -> 267.977328 tok/s (`+0.153544%` regression) |

The promotion signal is the frozen eligible aggregate, not the all-context
TPOT aggregate. At 4,096 and 8,192 tokens the candidate intentionally selects
the same K8 width as the control; their small differences are protected-noise
checks, not claimed K16 benefits.

## Measured Cost and Safety

```text
selected-K16 host-visible gap:
  P95:                              36.914469 ms
  maximum:                          37.006666 ms
  frozen maximum:                   40.000000 ms

shared graph capture duration (maximum observed):
  elastic policy process:           15.698698350 s
  fixed-K8 control process:         16.203395746 s
  elastic incremental capture:      0 ns

shared capture storage per row:
  retained static bytes:            915,760 to 915,884
  allocated delta bytes:            914,432
  reserved delta bytes:             2,097,152
  elastic incremental retained:     0 bytes
  elastic incremental allocated:    0 bytes
  elastic incremental reserved:     0 bytes

eligible peak CUDA memory:
  allocated: 41,680,460,800 -> 41,652,018,176 bytes (-0.068240% regression)
  reserved:  41,955,622,912 -> 42,014,343,168 bytes (+0.139958% regression)

overall peak CUDA memory:
  allocated: 41,680,460,800 -> 41,653,471,232 bytes (-0.064754% regression)
  reserved:  42,058,383,360 -> 42,058,383,360 bytes (0.000000% regression)

candidate width activity:
  attempts / acceptances / commits:  250 / 250 / 250
  K8 fallbacks:                      180 / 250 = 72%
  selected K16 bursts:               70
  K16 width-health quarantines:      0

execution inventory:
  committed decode tokens:           2,540
  target-model forwards:             2,540
  graph replays:                     2,540
  intermediate token D2H calls:      0
  final token D2H calls / bytes:      250 / 20,320
  failures / quarantines:             0 / 0
  journal rollbacks / one-phase:      0 / 0
```

The 72% K8 fallback rate is expected policy behavior: all 4,096/8,192-token
attempts fall back for `context_above_2048`, and the final short output tail
falls back for `output_budget_below_16`. The independent verifier rejects any
other fallback reason, so this rate is a cost of the conservative policy, not
an unclassified runtime failure.

## Terminal Result

`GO_CONTEXT_GATED_ELASTIC_EXACT_BURST`

The complete source-bound Qwen3-0.6B gate establishes a real eligible-context
TPOT median/P95 benefit while staying below the 40 ms visibility limit and
preserving exact outputs, runtime inventory, protected latency/throughput/
memory limits, and fail-closed lifecycle behavior.

This authorizes the default-disabled elastic K8/K16 policy only for the frozen
Stage-1 envelope: TP1, batch one, completion-only, temperature zero,
`ignore_eos=true`. It does not establish streaming, EOS-aware generation,
multi-sequence scheduling, tensor parallelism, Qwen3-8B benefit, threshold
retuning, or production-default readiness.
