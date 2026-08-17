# TP4 Qwen3 B4 Host-Semantic Campaign r7

## Scope

This is the canonical target-then-learned half of the repeat-aligned
host-semantic comparison.

```text
policy order:       target,learned
prime each policy:  true
world size:         4
GPU indices:        3,4,6,7
measured repeats:   8 per policy
exact greedy parity:true
campaign status:    PASS
```

The raw campaign initially exited nonzero during postprocessing. The measured
worker JSON, timing artifact, GPU telemetry inputs, and host JSONL were not
rerun or modified. `recovery-provenance.json` and
`recovery-raw-inputs.sha256` bind the postprocessing-only recovery. The
original failure remains in `initial-exit-code.txt`; the canonical recovered
result is in `exit-code.txt`.

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

The r8 campaign and the comparison artifact bind the same six hashes.

## Coverage

```text
target raw samples:       1585
target aligned samples:   14,13,19,13,21,15,12,11
target maximum repeat gap:0.201811240 s

learned raw samples:       1069
learned aligned samples:   58,61,54,39,43,63,52,40
learned maximum repeat gap:0.203207182 s

allowed maximum gap:       0.600000000 s
```

All gaps above are calculated only inside each measured repeat's bracketing
host interval. Samples emitted after the final measured repeat are not used
to reject an otherwise aligned campaign.

## Timing Medians

| Policy | E2E (s) | TPOT (s) | Proposal forward (ms) |
| --- | ---: | ---: | ---: |
| target | 2.2908685005 | 0.1415256324 | 0.0000000000 |
| learned | 9.0720287193 | 0.5817995088 | 5174.6466420591 |

## Primary Host Medians

| Metric | Target | Learned |
| --- | ---: | ---: |
| context switches/s | 6208538.040880868 | 6201304.916815925 |
| CPU iowait fraction | 0.000079238772 | 0.000003701729 |
| CPU system fraction | 0.387350541149 | 0.390570009291 |
| I/O PSI some fraction | 0.015943443887 | 0.015899403023 |
| major faults/s | 0.000000000000 | 0.164828530622 |
| memory dirty max (KiB) | 297486772 | 297486882 |
| memory PSI some fraction | 0.000000000000 | 0.000000064106 |
| memory writeback max (KiB) | 78 | 82 |
| run queue mean | 41469.591228070 | 41341.092100406 |

## Verification

```text
runner exit:                         0
postprocessing recovery driver:      0
remote timing verifier:              PASS
local timing verifier:               PASS
remote GPU telemetry verifier:       PASS
local GPU telemetry verifier:        PASS
remote host-semantic verifier:       PASS / ALIGNED_CAMPAIGN
local host-semantic verifier:        PASS / ALIGNED_CAMPAIGN
bundle manifest:                     PASS / 72 entries
```

The initial postprocessing exit and initial verifier exits are retained as
`1`; the canonical source-bound recovery exits are all `0`.

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

