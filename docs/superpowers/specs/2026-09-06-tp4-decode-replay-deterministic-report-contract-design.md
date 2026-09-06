# TP4 Decode Replay Deterministic Report Contract

Date: 2026-09-06

## Context

The frozen TP4 decode replay qualification design lists `report.md` as
mandatory terminal evidence. The current assembler does not produce it, the
manifest does not hash it, and the independent verifier does not require or
validate it. That leaves a protocol blind spot: a bundle may verify while the
required human-readable conclusion is absent, or a producer-written report may
contradict the hash-bound machine evidence.

Historical bundles, including r48, remain immutable. This contract applies only
to fresh bundles produced by a new source revision.

## Decision

The assembler writes one canonical UTF-8 `report.md` with a terminal newline.
Every statement in the report is a deterministic rendering of:

- validated source identity;
- validated launch admission and claim boundary;
- validated cleanup classification;
- the classification and metrics reconstructed from raw evidence rows by
  `tp4_decode_replay_contract.classify`.

The report contains no free-form producer commentary.

The independent verifier:

1. requires `report.md` in the producer artifact inventory;
2. verifies its manifest hash;
3. independently reconstructs source, admission, cleanup, classification, and
   metrics from the underlying evidence;
4. renders the canonical expected report in verifier-owned code; and
5. requires byte-for-byte equality.

The verifier must not import the producer assembler.

## Canonical Content

The report records:

- run tag, source revision, source tree hash, model repository, and model
  revision;
- admission mode and claim boundary;
- cleanup classification;
- final classification and every failed gate;
- aggregate throughput and TPOT ratios;
- per-workload throughput, TPOT, P99 end-to-end, and TTFT ratios;
- replay coverage;
- added peak allocated and reserved bytes;
- capture duration and capture-amortization tokens;
- the fixed claim boundary that only `GO_STAGE1_JUSTIFIED` under
  `FORMAL_STRICT_CLEAN` may justify Stage 1, while `DIAGNOSTIC_ONLY` evidence
  never does.

Unavailable values are rendered as `N/A`; floating-point values use a stable,
round-trip-safe representation.

## Failure Semantics

- Missing `report.md` or a missing manifest entry is `INCOMPLETE`.
- A hash mismatch is `INCOMPLETE`.
- A rehashed report that differs from reconstructed evidence is `INCOMPLETE`
  with `report does not match reconstructed evidence`.
- The report does not change classification precedence or any frozen
  correctness, replay, performance, capture-cost, memory, admission, or cleanup
  threshold.

## Compatibility and Rollout

Existing bundles are not rewritten or upgraded. A fresh run tag and the new
source revision are required for hardware qualification. r48 remains
`INCOMPLETE` for its original capture-cost gap and remains without a
spec-compliant report.
