# Task 5 Implementer Report

## Scope

Implemented only Task 5, Canonical Exact-Identity Diagnostic, from:

- approved spec:
  `docs/superpowers/specs/2026-08-18-autoregressive-draft-command-timeline-sync-debt-design.md`;
- implementation plan:
  `docs/superpowers/plans/2026-08-18-autoregressive-draft-command-timeline-sync-debt.md`;
- task brief: `.superpowers/sdd/task-5-brief.md`.

Base SHA:

```text
ebded1867081a1c6fac96331c2fe5cf8c21bb751
```

No later task, runtime optimization, GPU run, remote run, artifact generation,
review-package update, retired-checkout change, frozen-fingerprint update, or
push was performed. Completed schema-v2 payload state was neither imported nor
mutated.

## Implementation

Added the standalone pure deterministic diagnostic:

`tools/autoregressive_draft_command_timeline_diagnostic.py`

It provides the required public interfaces:

- `EpochIdentity`;
- `expected_epoch_identities()`;
- `validate_epoch_worker(worker, identity)`;
- `join_repeat_timeline(worker, repeat_index)`;
- `compute_sync_debt(repeat)`;
- `build_epoch_admission(identity, raw_inputs)`;
- `compute_paired_boundary_effects(epochs)`;
- `classify_boundary(bundle_admission, effects)`;
- `build_command_timeline_artifact(...)`; and
- `validate_command_timeline_artifact(artifact)`.

The diagnostic enforces:

- the exact eight-epoch, four-block balanced graph/eager schedule and all
  fixed TP4/B4/Q4, prompt, output, temperature, Proposal-KV, provenance,
  source, checkpoint, tokenizer, and four-GPU identities;
- graph lifecycle/resource identity and eager-zero lifecycle requirements;
- exact token, proposal-row, accepted-prefix, accepted-token, acceptance,
  transaction-digest, and active-transaction parity without padded or
  oversized logical rows;
- strict four-rank command, deferred-CUDA, engine-step, and telemetry joins
  using non-boolean identities, matching clock/boot metadata, exact repeat
  intervals, acknowledgement semantics, and fail-closed interval ordering;
- queue debt, CUDA execution, acknowledgement wait, scheduler/postprocess,
  and conservation arithmetic without double-counting;
- inclusive stationarity and localization thresholds, balanced paired
  effects, exact classification precedence, and optimization authorization
  only for verified `BOUNDARY_LOCALIZED`;
- `performance_improvement_established=false`, `phase_1_complete=false`, and
  `promotion_ready=false` for this phase;
- exact artifact top-level keys and full derived-field recomputation from
  embedded normalized epochs; and
- canonical JSON and SHA-256 behavior, finite numerics, signed 64-bit integer
  bounds, strict boolean/integer distinction, bounded collections/strings/
  nesting, safe relative paths, and deterministic ordering.

Added comprehensive independent mutation tests in:

`tools/test_autoregressive_draft_command_timeline_diagnostic.py`

Each negative changes one contract dimension. Tests include exact-boundary and
one-unit-beyond threshold cases, identity aliases, missing/duplicate/reordered
and unknown timeline evidence, negative/overlapping intervals,
acknowledgement semantic errors, parity drift, cross-epoch graph drift,
artifact-key drift, and derived-field tampering.

## RED Evidence

The mandatory focused RED was run after the Task 5 tests were written and
before the production module existed:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  -k 'schedule or identity or parity or timeline or debt or conservation or stationarity or classification'
```

Output:

```text
1 error in 0.13s
```

Collection failed because
`tools/autoregressive_draft_command_timeline_diagnostic.py` did not exist.

The first implementation runs exposed normalization and recomputation defects:

```text
68 passed, 15 failed
70 passed, 13 failed
```

After the first complete GREEN, focused self-review added seven independent
fail-closed tests before their production fixes:

```text
7 failed, 83 deselected in 0.39s
```

Those RED cases covered cross-graph-epoch identity/resource drift, CUDA
selected-sequence and batch-kind joins, rank-zero acknowledgement completion,
engine status/error detail, scheduler/command overlap, inconsistent admission
summary, and signed integer bounds.

## GREEN Evidence

First complete Task 5 GREEN:

```text
83 passed in 5.18s
```

Review-fix targeted GREEN:

```text
7 passed, 83 deselected in 0.12s
```

Final fresh Task 5 suite:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-task5-final-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py
```

Output:

```text
90 passed in 5.22s
```

Final fresh regression for the four inspected pure-helper families:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-task5-helper-final-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py
```

Output:

```text
217 passed in 6.79s
```

Python syntax verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-task5-compile-pycache \
python3 -m py_compile \
  tools/autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py
```

Output: exit code `0`, no diagnostics.

`git diff --check` passed.

## Invariant Review

- Schedule generation produces exactly the approved labels and positions.
- Every identity-bearing integer rejects boolean aliases and values outside
  the signed 64-bit bound.
- Worker normalization retains evidence required for artifact recomputation
  while semantic parity excludes graph lifecycle counters.
- Timeline joins reject missing, duplicate, reordered, unknown, negative,
  out-of-interval, overlapping, and semantically invalid evidence.
- Queue, CUDA, acknowledgement, and scheduler/postprocess components are
  disjoint and conservation is checked against both absolute and relative
  tolerances.
- Stationarity thresholds are inclusive; each one-unit-beyond mutation fails.
- Localization requires at least 60% absolute E2E explanation, at least three
  same-sign blocks, balanced order behavior, and median unexplained E2E no
  greater than 10%.
- Classification precedence is identity/correctness, timeline/conservation,
  stationarity, then localization.
- Runtime optimization remains unauthorized except for a recomputed
  `BOUNDARY_LOCALIZED` result; no performance improvement is claimed.
- Artifact validation rebuilds every derived field from normalized embedded
  epochs and rejects canonical-byte differences.
- No completed schema-v2 module or payload state is imported or modified.

## Self-Review

The Task 5 source and tests were reviewed by schedule/identity/parity,
timeline/conservation, stationarity/localization, and artifact/canonicalization
groups. Seven additional fail-closed gaps were converted into RED tests and
fixed. After the final focused and full regressions, no remaining P0-P2 finding
was identified.

## Commit

`b295e104528ca34095e3b571d46f3c4b4e8daedf`

## Review 1 Fix

Review source:
`.superpowers/sdd/task-5-review-1.md` (`Needs fixes`).

The review identified five fail-closed gaps in the initial localization and
nested-schema validation. All fixes remain inside Task 5:

- balanced `eager_graph` and `graph_eager` order groups must each contain a
  qualifying block and support one common nonzero boundary direction;
  mixed-sign groups or order reversal/sequence interaction remain stable but
  unlocalized and cannot authorize runtime optimization;
- localization requires exactly one candidate boundary; zero or multiple
  candidates produce `localized_boundary=None`,
  `stable_but_unlocalized=true`, and no authorization;
- nanosecond deltas and component values remain signed 64-bit integers, and
  the 60% explanation and 10% residual gates use exact integer/rational cross
  multiplication rather than float comparisons;
- zero E2E delta with a nonzero component emits
  `explanation_ratio=None` with a false defined flag, remains
  non-qualifying, and cannot localize; and
- nested command-timeline and command-rank snapshot schema versions pass
  through the strict non-boolean integer validator before comparison.

Per-request timing aggregation now uses the discrete lower median, preserving
integer nanoseconds and making even-cardinality aggregation conservative.
Exact paired epoch midpoint values are retained internally as rational values
for the residual threshold while canonical output remains finite.

### Review-1 RED Evidence

Focused review command:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-task5-review1-red-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  -k 'mixed_signs or order_reversal or multiple_localized or integer_threshold or integer_residual or zero_e2e or timeline_schema_versions'
```

Output:

```text
9 failed, 3 passed, 90 deselected in 0.17s
```

The nine failures were the accepted mixed-sign and reversal cases, arbitrary
multi-candidate selection, integer-type loss, one-nanosecond explanation and
residual threshold rounding, infinite zero-denominator output, and boolean
timeline/rank schema aliases. The three controls already passing were exact
threshold admission and malformed non-integer schema strings.

### Review-1 GREEN Evidence

Focused review GREEN:

```text
12 passed, 90 deselected in 0.08s
```

Artifact-focused integration regression after preserving integer timing
medians:

```text
32 passed, 70 deselected in 5.26s
```

Complete Task 5 regression:

```text
102 passed in 5.38s
```

Reused-helper regression:

```text
217 passed in 6.95s
```

Python syntax compilation completed with exit code `0` and no diagnostics.
`git diff --check` passed.

No Task 6 work, runtime optimization, GPU/remote execution, schema-v2 payload
mutation, historical artifact/review-package modification, retired-checkout
change, or push was performed.

### Review-1 Fix Commit

`SELF/HEAD`
