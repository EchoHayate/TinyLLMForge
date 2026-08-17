# Sequence Temperature Serialization Design

**Date:** 2026-08-12

## Goal

Preserve `Sequence.temperature` across the existing pickle/shared-memory
transport so every tensor-parallel ModelRunner rank can enforce the greedy
speculative first-target boundary consistently.

## Problem

`Sequence.__getstate__()` currently emits a 14-field state containing decode,
chunk, request, hybrid lease, and token payload data, but not temperature.
Worker ranks reconstruct a `Sequence` without a `temperature` attribute.

`run_spec_first_target_batch()` executes on every rank and validates
`temperature == 0`. Without transport support, rank zero succeeds while worker
ranks fail before the target forward.

## Schema

The new state is a 15-tuple:

```text
0  num_tokens
1  num_prompt_tokens
2  num_cached_blocks
3  block_table
4  num_computed_tokens
5  prefill_chunk_start
6  prefill_chunk_end
7  prefill_chunk_final
8  step_is_decode
9  step_do_sample
10 seq_id
11 hybrid_state_slot_id
12 hybrid_state_generation
13 temperature
14 token payload or last_token
```

The token payload remains the final element so existing prompt/decode restore
logic remains simple.

## Backward Compatibility

`__setstate__()` accepts:

- new 15-field state and restores the finite numeric temperature;
- old 14-field state and defaults temperature to `0.0`;
- old 13-field hybrid state and defaults temperature to `0.0`;
- legacy 11-field and older states and defaults temperature to `0.0`.

The default is intentionally greedy. Old workers never had production
speculative execution, and ordinary ModelRunner decode does not use
temperature on worker ranks.

Malformed new-state temperature values are rejected instead of silently
normalizing.

## Tests

- prompt round trip preserves a nonzero temperature;
- decode round trip preserves a nonzero temperature;
- explicit old 14-field state restores `temperature == 0.0`;
- existing 13/11/5-field compatibility tests retain their behavior and assert
  the greedy default;
- chunked-prefill TP worker round trip preserves temperature.

## Non-Goals

- transporting `max_tokens` or `ignore_eos`;
- changing sampling behavior;
- enabling stochastic speculative decoding;
- engine runtime wiring;
- performance claims.

## Result Boundary

```text
TP worker greedy validation input:
  available and schema-compatible
LLMEngine speculative execution:
  not implemented
overall classification:
  NOT_PROMOTABLE
```
