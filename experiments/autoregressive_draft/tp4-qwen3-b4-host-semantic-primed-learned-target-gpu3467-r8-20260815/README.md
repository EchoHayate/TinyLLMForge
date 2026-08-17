# TP4 Qwen3 B4 Host-Semantic Campaign r8

## Scope

This is the clean learned-then-target half of the repeat-aligned
host-semantic comparison.

```text
policy order:       learned,target
prime each policy:  true
world size:         4
GPU indices:        3,4,6,7
measured repeats:   8 per policy
exact greedy parity:true
campaign status:    PASS
```

Unlike r7, this campaign launched after the run-path, host-gap, and GPU-edge
alignment fixes and completed without postprocessing recovery.

## Source Identity

```text
tools/autoregressive_draft_b4_timing_diagnostic.py
  d2fc63df070602403163c87e2d3a7244d373522e38ec694482f48bb0f7a87b4b
tools/autoregressive_draft_host_sampler.py
  6245dc19c9f56cf1530181a5be9a4df606f96adc71c69ff42bf7f533d4eac986
tools/autoregressive_draft_host_semantic_diagnostic.py
  e3b1a4ed9dbfc769ab4baed2356f6d63760eefd9ab7919585a3f169fb8bf49ee
tools/autoregressive_draft_instability_telemetry.py
  b74acc04ddbb5557c65d9e983e73b96764cb272bd9a384ed6d71babe405c79fa
tools/autoregressive_draft_performance_worker.py
  fdc81278a137218b66e4057ae08d52dfa52ed30c426c394e51b171bc2893b7c5
tools/verify_autoregressive_draft_host_semantic_diagnostic.py
  110556d81c4b119ca740deb83080d53958c7885c6bfb6a609534bd6cc8ea0a8a
```

The r7 campaign and the comparison artifact bind the same six hashes.

## Coverage

```text
learned raw samples:       940
learned aligned samples:   47,42,29,44,47,31,30,25
learned maximum repeat gap:0.234627638 s

target raw samples:       2579
target aligned samples:   16,22,16,24,25,22,17,20
target maximum repeat gap:0.222141446 s

allowed maximum gap:      0.600000000 s
```

All gaps above are calculated only inside each measured repeat's bracketing
host interval.

## Timing Medians

| Policy | E2E (s) | TPOT (s) | Proposal forward (ms) |
| --- | ---: | ---: | ---: |
| learned | 6.1944397425 | 0.3971080518 | 3468.1439884007 |
| target | 3.7443748215 | 0.2380473820 | 0.0000000000 |

## Primary Host Medians

| Metric | Learned | Target |
| --- | ---: | ---: |
| context switches/s | 6189290.617425797 | 6237102.887156395 |
| CPU iowait fraction | 0.000033576805 | 0.000074123356 |
| CPU system fraction | 0.398614217406 | 0.398306005718 |
| I/O PSI some fraction | 0.015843236751 | 0.015940874065 |
| major faults/s | 0.187509227588 | 0.000000000000 |
| memory dirty max (KiB) | 297487060 | 297486840 |
| memory PSI some fraction | 0.000000000000 | 0.000000000000 |
| memory writeback max (KiB) | 60 | 44 |
| run queue mean | 42448.454393771 | 41892.563750000 |

## Verification

```text
runner exit:                    0
remote timing verifier:         PASS
local timing verifier:          PASS
remote GPU telemetry verifier:  PASS
local GPU telemetry verifier:   PASS
remote host-semantic verifier:  PASS / ALIGNED_CAMPAIGN
local host-semantic verifier:   PASS / ALIGNED_CAMPAIGN
bundle manifest:                PASS / 54 entries
```

## Cross-Order Result

The independently verified r7/r8 comparison is:

```text
classification:
  HOST_ALIGNMENT_INCONCLUSIVE

reason:
  learned E2E position effect is below 10%

learned-first versus learned-second:
  E2E:              -31.7193547971%
  TPOT:             -31.7448629878%
  proposal forward: -32.9781484940%
```

The previously observed positive learned-first slowdown did not reproduce;
the direction reversed and learned-first was faster. Two of nine primary
host metrics were worse in learned-first: CPU iowait and major faults. The
largest positive Spearman coefficient was `0.3589400087`, below the `0.6`
threshold; run queue was negatively correlated with E2E at `-0.55`.

This is not a missing-sample or source-identity classification. It does not
support or refute a host-pressure cause for a positive slowdown because no
positive slowdown was present in these paired campaigns.

## Claim Boundary

Host correlation is not causal proof, and system-wide counters do not
identify a responsible process. This campaign does not establish stable
long-context performance, Proposal-KV offload benefit, Phase 1 completion, or
promotion readiness.

