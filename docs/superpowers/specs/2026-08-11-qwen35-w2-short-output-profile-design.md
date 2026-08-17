# Qwen3.5 W2 Short-Output Profile Design

## Goal

Run a focused TP4 `w2_long_reuse` experiment with eight generated tokens to
test whether the existing 15,360-token prefix reuse produces a stable
end-to-end benefit when the decode tail is shortened.

## Scope

The experiment keeps these properties identical to the existing 64-token
profile:

- shared prefix: 3,840 tokens;
- continuation suffix: 64 tokens;
- continuations per case: 4;
- policies: `recompute` and `exact_restore`;
- one paired warmup and five paired measured repetitions;
- fixed GPUs `2,4,5,6`;
- at least 25 GiB free and at most 10 percent utilization per GPU;
- shared-low-utilization, non-exclusive resource classification;
- attempt-scoped cleanup.

Only generated output length changes, from 64 to 8 tokens.

## Architecture

Add a profile-only generated-token override to the existing benchmark worker.
The override is rejected unless profiling is enabled, the workload is
`w2_long_reuse`, and the value is a positive integer no larger than the
canonical 64-token output. The worker copies the canonical workload payload,
changes only `spec.generated_tokens`, and uses that same effective payload for
engine execution and request validation.

The canonical workload manifest, benchmark case matrix, case-row field set,
and existing r607/r609 artifacts remain unchanged. The separate `profile.json`
records both canonical and effective generated-token counts and marks the case
as an experimental short-output variant.

The existing profile aggregator accepts an expected generated-token count. It
verifies every measured row and profile artifact against that count, reports
the count in the summary, and preserves the same restore, TTFT, decode,
makespan, and reuse metrics used by the 64-token profile.

## Interfaces

Worker:

```text
--profile
--generated-tokens-override 8
```

Python:

```text
build_engine_configuration(..., generated_tokens_override=None)
validate_benchmark_requests(..., generated_tokens=None)
run_benchmark_case(..., profiling=False, generated_tokens_override=None)
```

Profile case metadata:

```json
{
  "variant": "short_output",
  "canonical_generated_tokens": 64,
  "generated_tokens": 8
}
```

Aggregator:

```text
--generated-tokens 8
```

## Safety and Evidence Boundaries

- No canonical benchmark schema or manifest is changed.
- The short-output result is not a replacement for the 64-token authority
  artifact; it is a diagnostic experiment.
- Output-token parity is required for all 20 measured continuation pairs.
- Restore timing retains the existing boundary: `restore_prepare` combines
  rank-local work with acknowledgement transport and waiting.
- A stable speedup claim requires paired and ratio-of-medians summaries to
  agree in direction. Otherwise the result is classified as inconclusive.
- No unrelated GPU process may be killed.

## Success Criteria

1. Five measured recompute/exact-restore pairs complete with eight generated
   tokens per continuation.
2. Every pair preserves output-token parity.
3. Exact restore still reports 15,360 reused KV tokens and 256 executed
   prefill tokens per four-request case.
4. Decode share is lower than the 64-token run's 84.722 percent.
5. TTFT, decode, makespan, restore, and both makespan-ratio summaries are
   reported.
6. Cleanup is `CLEAN` with no attempt-scoped process remaining.
7. Focused tests, compilation, artifact regeneration, and `git diff --check`
   pass.

