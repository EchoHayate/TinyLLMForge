# Qwen3.5 TP4 Request-Level E2E64 Comparison Design

## Objective

Measure whether the r631 Qwen3.5 decode phase-split implementation improves
complete request latency and throughput relative to the r620 baseline under
an identical canonical workload.

The comparison must not reuse the 8-token decode-internal campaign as a
request-level proxy. Both sources run the canonical `w2_long_reuse` workload
with 64 generated tokens per request.

## Scope

The campaign covers two immutable benchmark sources:

- r620 baseline source tree:
  `a26c543e79a9d4927fd0451d4a287363a677568a1daefe65a2a234a22f5997aa`
- r631 candidate source tree:
  `6f881fae7010cc5f048100b147a72fbf27ffba0f77bc34e2e2e68388a98a2837`

Each source runs the same 12 workers:

- warmup r0 for `recompute` and `exact_restore`
- measured r0-r4 for `recompute` and `exact_restore`
- four requests per worker
- 64 output tokens per request
- GPUs `2,4,5,6`

This work adds attempt-scoped orchestration, comparison, tests, reports, and
handoff evidence. It does not modify the canonical workload manifest,
case-matrix schema, existing `profile.json` schema, or old artifacts.

## Considered Approaches

### 1. Thin wrapper over the verified r631 launcher

Import the verified attempt-scoped `launch_w2.py`, override source identity
and output paths, and derive canonical 64-token commands by deleting
`--generated-tokens-override 8` and `--decode-internal-profile`.

Advantages:

- preserves the proven GPU guard, upload bundle, sequential SSH runner,
  download, and exact-tag cleanup implementation
- keeps baseline and candidate execution structurally identical
- minimizes new remote-control code

Risk:

- the imported launcher is attempt-scoped, so the wrapper must validate every
  derived command and must not rely on mutable global state after launch

This is the selected approach.

### 2. Copy the launcher into two new attempt directories

This is operationally explicit but duplicates a large amount of safety logic
and creates a higher risk that the two copies drift.

### 3. Extend the canonical benchmark CLI

This would be reusable, but it unnecessarily changes a stable execution
surface for a one-off controlled comparison and risks altering old workflows.

## Architecture

### Dual-source orchestration

A local orchestration tool defines immutable source specifications for r620
and r631. For each source it:

1. selects a fresh attempt tag and local output directory;
2. dynamically imports the verified r631 `launch_w2.py`;
3. overrides `TAG`, `OUTPUT`, `REMOTE`, `SOURCE`, `SOURCE_TAR`, and
   `SOURCE_TAR_SHA`;
4. derives the 12 commands from the verified template;
5. removes the two short-output-only arguments;
6. validates the canonical command matrix before any remote side effect;
7. verifies that the local and remote attempt paths do not exist;
8. executes the existing launcher's upload, guarded sequential run, download,
   and exact-tag cleanup flow.

The baseline and candidate use separate fresh tags and separate attempt
directories. A failed attempt remains preserved and is never overwritten.

### Command validation

Every derived command must satisfy all of the following:

- exactly 12 commands;
- every case is `w2_long_reuse`;
- warmup and measured phase/policy/repetition matrix matches the canonical
  manifest;
- `--profile` remains present;
- `--generated-tokens-override` is absent;
- `--decode-internal-profile` is absent;
- source tree hash and attempt tag match the selected source;
- the worker therefore uses its canonical 64 generated tokens.

Any mismatch fails before SSH upload or remote execution.

### Comparison and metrics

The comparison tool reads the ten measured case directories from each
attempt and validates four request rows per case.

For each measured case:

```text
case_makespan_ns = max(request.e2e_ns)
request_throughput_rps = 4e9 / case_makespan_ns
output_token_throughput_tps = 256e9 / case_makespan_ns
request_decode_ns = sum(request.decode_step_ns)
```

For each source and policy, the report records medians across five measured
repetitions for:

- case makespan;
- request throughput;
- output-token throughput;
- request E2E latency;
- TTFT;
- request decode latency.

It also records paired per-repetition candidate-versus-baseline ratios and
dispersion using minimum, maximum, and population standard deviation.

### Correctness and classification

Output-token parity is a hard gate:

- each measured baseline request must match the corresponding candidate
  request token-for-token;
- each row must contain exactly 64 output tokens and 63 decode-step timings;
- case IDs, policies, repetitions, and request identities must align.

Classification:

- `NO_GO`: parity, schema, source identity, completeness, or cleanup fails;
- `E2E_PERFORMANCE_PASS`: parity passes and candidate median makespan improves
  by at least 5 percent for both policies;
- `MIXED`: parity passes but only one policy reaches the 5 percent threshold
  or policy directions disagree;
- `NO_MATERIAL_E2E_CHANGE`: parity passes and both policy changes are within
  plus or minus 5 percent;
- `E2E_REGRESSION`: parity passes and either policy regresses by at least
  5 percent without the other policy qualifying as an improvement.

All latency and throughput deltas are reported regardless of classification.

## Safety and Cleanup

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run
  `git clean`.
- Use only GPUs `2,4,5,6`.
- Admission requires at least `26843545600` free bytes and utilization no
  greater than `10` percent on every selected GPU.
- The run is shared and non-exclusive; unrelated low-utilization processes
  are allowed.
- Do not create dummy reservations or kill unrelated processes.
- Cleanup may match only the exact current attempt tag and its descendants.
- Preserve every successful or failed attempt.

## Verification

Before launching:

- unit tests prove command derivation and metric/classification behavior;
- a static dry-run dump proves both 12-command matrices are canonical
  64-token runs;
- local and remote freshness checks pass for both tags;
- source tar hashes match the frozen values.

After launching:

- both attempts contain `RUN_COMPLETE`, all 12 worker outputs, and ten
  measured profile/row pairs;
- measured output tokens have full cross-source parity;
- comparison report includes all six latency/throughput metrics for both
  policies;
- cleanup receipts are `CLEAN`;
- independent exact-tag process checks return zero;
- completion audit maps every requirement to concrete artifacts.

