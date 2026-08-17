# Qwen3.5 Recurrent INT8 Task 10 Completion Audit

## Decision

`INVALID_ARTIFACT`

Canonical P2 preflight and execution are prohibited because the authoritative
prerequisite chain is incomplete and the previously assembled correctness
bundle no longer validates against the frozen schema-v2 contract.

## Local Gate

- All 24 Task 5-9 test entrypoints passed in fresh processes.
- The schema-v2 contract reported `578 passed in 23.43s`.
- The Task 10 `py_compile` command passed.
- The schema-v1 verifier regression exposed a stale test expectation after the
  frozen W3 shared-prefix size changed. The test now derives cache-efficiency
  expectations from the frozen contract; the production verifier was not
  changed. Its focused suite passed all 11 tests after the correction.
- The scoped `git diff --check` command still reports six pre-existing,
  unrelated trailing-whitespace lines in
  `tinyvllm/engine/model_runner.py`. They were not repaired during this audit.

Full local log:

```text
/tmp/qwen35_task10_local_gate_fresh_1785939425.log
SHA256 885642f1f5556f0dba0ec5ee38ed2ae73acca86fc14ab9eadb9241bf0a594adf
```

## Prerequisite Audit

The historical correctness prerequisite bundle is:

```text
experiments/qwen35_hybrid_state/
qwen35-tp4-performance-correctness-prerequisites-20260804-attempt67-r542/
correctness_prerequisites.json
SHA256 35b4bf092d5c4c84746b88ecd88b32bf14357a21d2923336d62653186cf352f8
```

Fresh validation returned `BLOCKED_CORRECTNESS`:

1. `tp4_root_logit execution plan identity is invalid`
2. `cached_continuation execution receipt resource guard schema fields are invalid`
3. `engine_correctness execution receipt resource guard schema fields are invalid`

An exact JSON scan under `experiments/qwen35_hybrid_state` found:

```text
authoritative calibration bindings: 0
authoritative strict-P1 bindings:   0
authoritative Gate-1 audit docs:    0
```

The required states therefore cannot be established:

```text
calibration:                     PASS  -> missing
strict P1 independent authority: GO    -> missing
Gate 1 audit:                   PASS  -> missing
correctness prerequisites:      PASS  -> invalid under current contract
```

## Safety Boundary

The v2 preflight was not invoked because prerequisite validation must precede
the GPU query. No SSH command, GPU query, CUDA work, remote directory,
source staging, worker launch, execution plan, authorization, or receipt was
created.

No correctness, accuracy, cache, capacity, memory, compression, latency,
throughput, or speed benefit is established.

## Exact Next Action

Regenerate the source-bound correctness prerequisite bundle against the
current schema-v2 identity/resource-guard contract, then independently produce
and verify:

1. a real recurrent-INT8 calibration binding classified `PASS`;
2. a strict P1 exact-restore authority binding classified `GO`;
3. a closed Gate-1 audit document classified `PASS`.

Only after all four prerequisite documents validate may a fresh read-only v2
preflight query GPUs `2,4,5,6`.
