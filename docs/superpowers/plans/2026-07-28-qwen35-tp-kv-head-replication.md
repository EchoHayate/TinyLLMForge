# Qwen3.5 TP KV-Head Replication Implementation Plan

**Goal:** Add exact full-attention KV-head replication so the approved
`8Q/2KV` Qwen3.5 checkpoint can construct and load at TP4.

## Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, merge, create a PR, or delete evidence.
- Preserve TP1/TP2 behavior.
- Replicate only complete full-attention KV heads.
- Do not alter linear-attention head sharding.
- Do not claim performance or accuracy benefit from this gate.

### Task 1: KV Parallel Linear Contract

- [x] Write RED tests for normal sharding, TP4 replication, source-rank
  mapping, and invalid `3KV/TP4`.
- [x] Implement the dedicated KV-head parallel linear layer.
- [x] Run focused linear tests GREEN.

### Task 2: Qwen3.5 Component Construction

- [x] Write TP4 `8Q/2KV` component RED tests.
- [x] Replace only full-attention K/V projections with the dedicated layer.
- [x] Preserve Q/O projection and TP1/TP2 shapes.
- [x] Run component tests GREEN.

### Task 3: Checkpoint Binding and Assignment

- [x] Write RED tests proving rank0=rank1 and rank2=rank3 K/V payloads.
- [x] Teach binding validation the replicated local shape contract.
- [x] Load source rows by KV source rank.
- [x] Run binding/assignment/candidate tests GREEN.

### Task 4: Restore TP4 Real Provenance Gate

- [x] Update the real-candidate source contract with the authorized delta.
- [x] Run focused and adjacent regressions.
- [x] Run a new authoritative remote TP4 real-candidate tag.
- [x] Preserve all failed tags and record the new evidence.

## Completion Evidence

- Focused KV linear tests: 3 passed.
- Concrete component tests: 2 passed, including `8Q/2KV/TP4`.
- Checkpoint binding tests: 4 passed.
- Checkpoint assignment tests: 5 passed, including identical payloads for
  ranks `(0, 1)` and `(2, 3)` with distinct payloads between the groups.
- Adjacent candidate-factory and full-attention-shell regressions: 6 passed
  each.
- Authoritative real-checkpoint run:
  `qwen35-tp4-real-candidate-replay-20260728-145713`.
- Frozen source tree:
  `42dddc0eac0a6db6041d5abb71df34db4d5e7c99d3b74d69f94598a2f24eb137`.
- The earlier failed and superseded run directories remain preserved.

This closes the KV-head construction/loading blocker only. It does not claim
model-output equivalence, accuracy, latency, throughput, cache savings, or
live concurrent TP4 ownership.

