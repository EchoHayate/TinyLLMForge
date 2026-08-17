# Greedy Speculative Selection Design

**Date:** 2026-08-12

## Goal

Prevent the scheduler from selecting stochastic decode rows for the current
greedy-only speculative runtime, and make temperature changes detectable when
the engine validates an immutable selection record.

## Problem

The selection builder currently treats `do_sample=True` as sufficient for
speculation. In TinyLLMForge, `do_sample` means that a row should produce an
output token; the actual greedy/stochastic choice is controlled by
`Sequence.temperature`.

The generic runtime only implements greedy prefix acceptance. Selecting a row
with `temperature != 0` would either produce incorrect stochastic semantics or
fail later at the ModelRunner boundary after scheduler publication.

## Decision

Add `temperature_snapshot: float` to `SpeculativeSelectionRow`.

The builder validates a finite numeric temperature for every sequence and uses
this suppression precedence:

```text
disabled
prefill
not_sampling
non_greedy
insufficient_output_budget
selected
```

A row is selected only when its effective row temperature equals `0.0`.

`validate_speculative_selection_record()` compares the current normalized
temperature with the immutable snapshot. A mutation between scheduling and
engine consumption is rejected as:

```text
speculative selection temperature is stale
```

It also rejects any selected row whose current temperature is nonzero, even if
a malformed record contains a matching nonzero snapshot.

## Alternatives

### Reject only in ModelRunner

Already implemented as a final safety boundary, but it publishes misleading
selected observations and fails after engine work has begun.

**Rejected as the primary gate.**

### Add stochastic speculative decoding now

This requires proposal distributions, rejection sampling, residual
distribution sampling, and new parity criteria.

**Deferred.**

### Scheduler suppression plus ModelRunner defense

Suppress non-greedy rows before runtime execution and retain ModelRunner's
greedy validation as defense in depth.

**Selected.**

## Compatibility

- Existing greedy rows use `temperature_snapshot=0.0` and remain selected.
- Prefill and non-sampling precedence remains unchanged.
- Disabled selection remains the highest-precedence reason.
- Mixed batches evaluate each row's own temperature.
- The legacy scheduler tuple shape remains unchanged.

## Tests

Dependency-light selection tests cover:

- ordinary nonzero-temperature decode becomes `non_greedy`;
- mixed batches select only greedy sampling decode rows;
- invalid, boolean, NaN, and infinite temperatures fail validation;
- temperature mutation after publication is stale;
- a scheduler with speculation enabled publishes non-greedy rows as
  suppressed;
- greedy scheduler behavior and tuple identity remain unchanged.

## Non-Goals

- stochastic speculative decoding;
- engine callback execution;
- variable-Q grouping;
- multi-token metadata commit;
- performance claims.

## Result Boundary

```text
greedy selection publication:
  correct and stale-checked
stochastic speculation:
  not implemented
LLMEngine speculative execution:
  not implemented
overall classification:
  NOT_PROMOTABLE
```
