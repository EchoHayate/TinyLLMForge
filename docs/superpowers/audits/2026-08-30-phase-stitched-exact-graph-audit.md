# Phase-Stitched Exact Graph Runtime Terminal Audit

**Date:** 2026-08-30

**Repository:** `/Users/bytedance/dev/TinyLLMForge`

**Branch:** `feat/kv-sparse-attention`

**Terminal classification:** `NO_GO_PERFORMANCE`

## Plain-Language Result

The runtime mechanism works and remains exact, but the optimization is too
narrow to promote.

Phase stitching removes a meaningful part of the visible pause between the
first and second generated tokens:

```text
aggregate token-0-to-token-1 gap improvement: 20.148586%
```

That pause is only a small fraction of the full 128-token request, however.
The resulting aggregate end-to-end improvement is only:

```text
aggregate E2E improvement: 1.047063%
required aggregate E2E improvement: 2.000000%
```

Neither prompt shape reaches the frozen 3% per-shape E2E threshold. The
feature therefore remains default-disabled and is not promoted.

## Prompt-to-Artifact Completion Checklist

| Requirement | Concrete evidence | Status |
| --- | --- | --- |
| Implement one parent transaction joining exact prefill and the first exact K8 decode burst | `tinyvllm/engine/phase_stitched_exact_graph.py`, `scheduler.py`, `model_runner.py`, and `llm_engine.py`; source-bound run reports one attempt, one success, one prefill replay, seven decode replays, and eight target forwards per stitched request | `COMPLETE` |
| Preserve exact greedy output | `summary.json`: 20/20 four-arm prompt/sample groups have identical token IDs and decoded-text hashes | `COMPLETE` |
| Keep Scheduler ownership unique across prefix and suffix publication | Regression coverage in `tools/test_phase_stitched_exact_graph.py`; source commit `a2cd73f1b83c0752e2d7ca2cbf23b1a4e073e03b` prevents a duplicate `running` reference | `COMPLETE` |
| Keep first-token LM-head indexing independent of prefill-only row indices | Regression coverage in `tools/test_phase_stitched_exact_graph.py`; source commit `8953262cac1774f429b9ac18cf67c111fb527a38` uses a temporary decode context for the one-row hidden state | `COMPLETE` |
| Count D2H work exactly once per transaction | Source commit `54ce310203446e954ebcaf52a9ddb8de225d62f1`; every r7 stitched row reports prefix/suffix calls `1/1` and bytes `8/56` | `COMPLETE` |
| Fail closed after replay begins | Gate records zero failures, quarantines, fallbacks, and pending leases; regression tests cover terminal post-replay failure behavior | `COMPLETE` |
| Use the frozen four-arm AB/BA matrix | `run_manifest.json`: eager, prefill-only, independent composition, and stitched composition; 256/2048 prompts; two rounds with reversed order; two warmups and five measurements per case | `COMPLETE` |
| Use exact frozen runtime settings | `run_manifest.json` and case results: Qwen3-0.6B, BF16, TP1, batch one, temperature zero, `ignore_eos=true`, completion-only, 128 generated tokens | `COMPLETE` |
| Admit only a strict-clean A100 | `run_manifest.json`: physical GPU 0, UUID `GPU-57be086f-e967-c022-3832-93df4fc77bd0`, 0 MiB used, 0% utilization, and no compute process | `COMPLETE` |
| Keep remote writes below the approved mounted root | Staging, runtime, primary, and controller artifacts are below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/phase-stitched-exact-graph/` | `COMPLETE` |
| Bind the run to pushed source | `run_manifest.json`: source `54ce310203446e954ebcaf52a9ddb8de225d62f1`; local and `origin/feat/kv-sparse-attention` matched before launch | `COMPLETE` |
| Produce a complete immutable case inventory | 16/16 case results and 16/16 zero exit codes; 80 measured rows total | `COMPLETE` |
| Produce manifest-bound evidence | `manifest.json` covers 19 primary files and has SHA-256 `201655637ecb6be0870bec67acffb7d51c1ae08828e86623c2321fcdeccc75fd` | `COMPLETE` |
| Run producer and independent verifier | Producer exit 0; remote verifier exit 0; resumed local verification independently reconstructs raw metrics and semantically matches the remote receipt | `COMPLETE` |
| Report benefit and cost | This audit reports E2E, first-token gap, TTFT, tail, memory, capture time, and retained graph storage | `COMPLETE` |
| Apply the frozen promotion thresholds without retuning | Gate classification is `NO_GO_PERFORMANCE`; failed checks are only aggregate and per-shape E2E gain | `COMPLETE` |

## Source and Run Authority

```text
terminal source commit:
  54ce310203446e954ebcaf52a9ddb8de225d62f1
run tag:
  20260830-qwen3-06b-phase-stitched-r7
contract SHA-256:
  3c6a3cb7db0cefa2f3b7e76f55a6673061149940b8acbbe3bb20b2a65cdcce44
manifest SHA-256:
  201655637ecb6be0870bec67acffb7d51c1ae08828e86623c2321fcdeccc75fd
producer classification:
  NO_GO_PERFORMANCE
remote verifier:
  PASS / NO_GO_PERFORMANCE
resumed local verifier:
  PASS / NO_GO_PERFORMANCE
```

Canonical local evidence:

```text
artifacts/phase_stitched_exact_graph/
  20260830-qwen3-06b-phase-stitched-r7/
```

Canonical remote evidence:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  phase-stitched-exact-graph/
```

The initial controller process completed all remote work but encountered a
transient SSH closure while fetching the terminal inventory:

```text
Connection closed by UNKNOWN port 65535
```

Read-only recovery confirmed 16/16 result files, 16/16 exit receipts,
producer exit 0, verifier exit 0, and complete remote terminal artifacts.
Only the download/local-verification phase was resumed. No benchmark worker
or run tag was recreated.

## Correctness and Mechanism

```text
four-arm exact token/text groups: 20 / 20
case results:                     16 / 16
measured rows:                    80 / 80
stitched attempt/success:          1 / 1 per measured request
prefill/decode graph replay:       1 / 7 per measured request
target-model forwards:             8 per measured request
prefix/suffix commits:             1 / 1 per measured request
prefix/suffix D2H calls:           1 / 1 per measured request
prefix/suffix D2H bytes:           8 / 56 per measured request
failure/quarantine/fallback:       0 / 0 / 0
pending leases after request:      0
preauthorized KV tokens:           7
```

The earlier r6 run incorrectly classified the mechanism as incomplete
because suffix-drain telemetry repeated the same transaction-level D2H
counters already reported by the prefix step. The worker correctly summed
step-local observations, so the duplicated observation produced `2/2` calls
and `16/112` bytes. The r7 source fixes the observation at its origin:
suffix drain now reports zero newly issued D2H calls and bytes. Runtime
execution, token publication, and CUDA work are otherwise unchanged.

## Performance

Primary comparison is stitched composition D versus independent composition
C.

| Prompt | C median E2E | D median E2E | E2E gain | C token-0→1 gap | D token-0→1 gap | Gap gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 302.966 ms | 301.706 ms | 0.415890% | 18.662 ms | 15.182 ms | 18.650420% |
| 2048 | 334.136 ms | 330.253 ms | 1.162159% | 19.580 ms | 15.720 ms | 19.713046% |
| aggregate | — | — | 1.047063% | — | — | 20.148586% |

Protected metrics all pass:

```text
256 TTFT:             5.461 ms -> 5.390 ms, 1.285430% improvement
2048 TTFT:           21.914 ms -> 19.525 ms, 10.903066% improvement
256 P95/P99 E2E:    0.835253% improvement
2048 P95/P99 E2E:   1.108896% improvement
256 peak reserved:  0.192172% lower
2048 peak reserved: 1.482554% lower
```

Frozen promotion thresholds:

```text
at least one shape E2E gain >= 3%: FAIL
aggregate E2E gain >= 2%:          FAIL
aggregate token-0→1 gap >= 10%:    PASS
TTFT regression <= 2%:             PASS
P95/P99 E2E regression <= 2%:      PASS
peak reserved regression <= 3%:    PASS
```

## Cost

The stitched arm reuses the same exact-prefill and one-token exact-burst
graphs as the independent-composition arm:

```text
prefill retained static storage: 4,764,704 bytes
prefill reserved capture delta:  41,943,040 bytes
burst retained static storage:      915,788 bytes
burst reserved capture delta:     2,097,152 bytes
```

Across the four stitched case processes:

```text
prefill capture duration: 734.090 ms to 767.006 ms
burst capture duration:   342.289 ms to 365.560 ms
```

These are startup/capture costs of the reused component graphs, not an
incremental memory penalty unique to stitching. The measured stitched peak
reserved memory is lower than the independent arm in both shapes. The real
incremental cost is runtime complexity: a parent lease, prefix-visible
publication, asynchronous suffix mailbox, and two-phase Scheduler commit.
That complexity is not justified by a 1.047% aggregate E2E gain under the
frozen workload.

## Classification and Claim Boundary

```text
PHASE_STITCHED_EXACT_GRAPH_RUN=20260830-qwen3-06b-phase-stitched-r7
PHASE_STITCHED_EXACT_GRAPH_SOURCE=54ce310203446e954ebcaf52a9ddb8de225d62f1
PHASE_STITCHED_EXACT_GRAPH_CLASSIFICATION=NO_GO_PERFORMANCE
PHASE_STITCHED_EXACT_GRAPH_CORRECTNESS=PASS_EXACT
PHASE_STITCHED_EXACT_GRAPH_MECHANISM=PASS
PHASE_STITCHED_EXACT_GRAPH_AGGREGATE_E2E_GAIN=1_047063_PERCENT
PHASE_STITCHED_EXACT_GRAPH_AGGREGATE_GAP_GAIN=20_148586_PERCENT
PHASE_STITCHED_EXACT_GRAPH_DEFAULT_ENABLED=false
PHASE_STITCHED_EXACT_GRAPH_PROMOTION=NOT_AUTHORIZED
```

This result proves that the phase boundary is removable and that doing so
reduces the first-token handoff gap. It does not establish enough whole
request benefit to promote the mechanism, does not establish a cross-engine
advantage, and does not authorize Qwen3-8B, TP greater than one,
multi-sequence, sampled decoding, or production-default claims.

The next performance candidate should attack a repeated decode cost rather
than another one-time phase boundary. The already approved
Octet-Folded Replay Graph is the direct next experiment: fold eight ordered
one-token graph steps into one physical graph launch while preserving all
logical target forwards and exact outputs.
