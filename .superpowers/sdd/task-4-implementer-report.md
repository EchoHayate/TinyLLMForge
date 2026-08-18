# Task 4 Implementer Report

## Scope

Implemented only Task 4, Deferred CUDA Identity and Worker Export, from:

- approved spec:
  `docs/superpowers/specs/2026-08-18-autoregressive-draft-command-timeline-sync-debt-design.md`;
- implementation plan:
  `docs/superpowers/plans/2026-08-18-autoregressive-draft-command-timeline-sync-debt.md`;
- task brief: `.superpowers/sdd/task-4-brief.md`.

Base SHA:

```text
8da831bb9d4622adc8bb0ed556ebd3f8830a5933
```

No later task was run. No GPU, CUDA, NCCL, checkpoint, remote, artifact,
experiment, review-package, retired-checkout, push, or frozen-fingerprint
update was performed.

## Implementation

Modified `DecodeInternalProfiler` to:

- capture the active `command_id`, `engine_step_id`, and `repeat_index` once at
  `begin_step`;
- retain that captured identity on finalized step and collective rows, even
  after the command ContextVar scope exits;
- emit explicit `None` identity outside an active command scope, preventing
  stale ContextVar reuse;
- support keyword-only
  `finalize(*, already_synchronized=False)`;
- reject non-boolean `already_synchronized` values; and
- skip its synchronization hook exactly when
  `already_synchronized=True`, while retaining the prior default behavior.

Modified model-runner and engine lifecycle wiring to:

- remember the active decode-profiler configuration;
- acknowledged-reset all ranks to fresh profilers without changing the
  configured label or enabled state;
- forward `already_synchronized` through the existing all-rank acknowledged
  finalization path;
- preserve rank inventory, acknowledgement validation, disagreement
  detection, and original exception propagation; and
- exclude configure/reset/finalize management commands from measured command
  tracing.

Modified the autoregressive-draft worker to:

- add opt-in `--command-timeline` and
  `run_policy_campaign(..., command_timeline=False)`;
- leave disabled output keys, default one-warmup/three-measured behavior, and
  the legacy injected callback keyword shape unchanged;
- configure command and decode profiling once per enabled campaign;
- complete authority and memory-reset work before timeline reset;
- reset the decode profiler immediately before request timing;
- retain every existing post-`engine.step()` synchronization;
- snapshot command, deferred CUDA, and engine-step evidence only after the
  final existing post-step synchronization;
- finalize CUDA Events with `already_synchronized=True`;
- export four command-rank snapshots, four CUDA-rank snapshots, and engine
  step rows per diagnostic repeat;
- use canonical JSON SHA-256 for the request set;
- restrict the diagnostic to learned TP4/B4/Q4 with exactly one warmup and
  five measured repeats;
- assign unique non-negative command-timeline repeat identities `0..5` while
  preserving the established public run repeat values `-1, 0..4`; and
- validate exact eager-zero or graph capture/replay/resource lifecycle
  counters.

No new `torch.cuda.synchronize`, CUDA completion fence, CUDA Event wait, or
request-path wait was added.

## RED Evidence

Initial mandatory focused RED, after tests and before Task 4 production edits:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_autoregressive_draft_performance_gate.py \
  -k 'command_timeline or already_synchronized or active_command'
```

Output:

```text
12 failed, 1 passed, 51 deselected in 0.25s
```

The failures were the missing Task 4 finalization, identity, all-rank reset,
CLI, orchestration, cardinality, and graph-counter contracts. The one pass was
the inherited configure-helper error-propagation behavior.

Focused self-review found two integration defects and each received a fresh
test-first RED:

```text
non-negative warmup timeline identity:
  1 failed in 0.16s
  KeyError: 'command_timeline_repeat_index'

disabled callback keyword compatibility:
  1 failed in 0.11s
  TypeError: unexpected keyword argument 'command_timeline'
```

The first proved that the diagnostic warmup's established public repeat `-1`
could not be passed directly to Task 3's non-negative repeat API. The second
proved that disabled mode had changed the pre-Task-4 injected callback call
shape.

## GREEN Evidence

Targeted review fixes:

```text
2 passed in 0.05s
```

Final focused Task 4 command:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-task4-focused-final-pycache \
python3 -m pytest -q \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_autoregressive_draft_performance_gate.py \
  -k 'command_timeline or already_synchronized or active_command'
```

Output:

```text
13 passed, 51 deselected in 0.20s
```

Complete planned Task 4 regression:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-task4-final-pycache \
python3 -m pytest -q \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
```

Output:

```text
117 passed in 2.87s
```

Task 1 and Task 2 regression:

```text
56 passed in 1.46s
```

Task 3 four-file regression under Homebrew Python 3.12 through a temporary
`uv` pytest environment:

```text
215 passed in 7.62s
```

Exact inherited-fingerprint regression:

```text
57 passed, 6 failed in 1.92s
```

The six failures retain the inherited shape:

- five stop at `ValueError: LLMEngine source hash is invalid`;
- one reports the pre-existing prerequisite source-closure mismatch.

Frozen fingerprint expectations were not rewritten.

Python 3.12 syntax compilation of all changed source and test files completed
with exit code `0` and no diagnostics.

`git diff --check` passed. A changed-line scan found no added
`torch.cuda.synchronize`, synchronize call, `wait_event`, completion fence,
request-path wait, or generic wait call.

## Invariant Review

- Command identity is captured at profiler step begin and cannot be replaced
  by a later or stale ContextVar value.
- `already_synchronized=True` performs zero profiler synchronization calls;
  omitted/default finalization retains one synchronization.
- Configure, reset, and finalize preserve all-rank acknowledged dispatch and
  propagate the original helper failure object.
- Disabled mode adds no timeline output and preserves result shape, defaults,
  and the pre-existing callback keyword contract.
- Enabled configuration occurs once. Each repeat resets command and CUDA
  recorders after pre-run authority/memory work and immediately before request
  timing.
- All existing post-step synchronizations remain in place. Timeline snapshots
  occur after the final one, and CUDA finalization reuses it.
- Every diagnostic run is validated for four command-rank snapshots, four
  non-empty CUDA-rank step inventories, and non-empty engine-step evidence.
- Exact eager and graph lifecycle counters are validated for all four ranks.
- Canonical JSON, TP4, B4, Q4, one warmup, and five measured repeats are
  fail-closed requirements.
- Existing immutable schema-v2 output remains unchanged while the diagnostic
  is disabled.

## Self-Review

The Task 4 diff was reviewed by profiler identity/fencing, all-rank lifecycle,
and worker orchestration groups. The review found the two RED→GREEN issues
documented above. After those fixes, no remaining P0-P2 finding was identified.
The temporary review reports were written outside the repository under
`/tmp/TinyLLMForge_task4_review.k1Wq9f/`.

## Commit

`SELF/HEAD`

## Review 1 Fix

Review source:
`.superpowers/sdd/task-4-review-1.md` (`Needs fixes`).

The review correctly identified that the initial campaign gate validated
rank cardinality but could accept empty, dropped, stale, malformed, or
unrelated timeline evidence. The fix remains inside Task 4:

- every rank command snapshot must contain rows and report zero drops;
- enabled CUDA snapshots report zero dropped steps/collectives and contain
  non-empty step evidence;
- the exported engine-step snapshot reports its dropped-step count;
- warmup is bound to timeline repeat `0`, while measured runs are bound to
  `1..5`;
- command, CUDA step/collective, and engine-step rows are bound to the
  canonical campaign request digest and expected repeat;
- row-level rank, command ID, engine-step ID, transport inventory, and
  optional selected-sequence digest relationships are reconciled;
- malformed identities, unknown IDs, inconsistent selected digests, and
  reused warmup evidence fail closed; and
- CUDA rows capture the command-trace request/selection identity at
  `begin_step`, while rows outside a command scope retain absent command
  identity.

The disabled worker path remains unchanged: it emits no command-timeline
field and retains the existing schema-v2 output shape, defaults, and callback
signature. The disabled profiler snapshot also retains its prior shape.

### Review-fix RED

Before production validation changes:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-task4-review-red-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_performance_gate.py \
  -k 'command_timeline_rejects_invalid_evidence or command_timeline_rejects_warmup_reuse'
```

Output:

```text
13 failed, 38 deselected in 0.28s
```

All 13 failures were `Failed: DID NOT RAISE <class 'ValueError'>`, directly
demonstrating the missing validation.

### Review-fix GREEN

Final focused Task 4 selection:

```text
28 passed, 51 deselected in 0.28s
```

Complete Task 4 regression:

```text
132 passed in 2.99s
```

Task 1 and Task 2 regression:

```text
56 passed in 1.55s
```

Task 3 four-file regression:

```text
215 passed in 3.30s
```

Exact inherited-fingerprint regression:

```text
57 passed, 6 failed in 1.86s
```

The inherited failure shape is unchanged: five failures stop at
`ValueError: LLMEngine source hash is invalid`, and one reports the existing
prerequisite source-closure mismatch. No frozen fingerprint was rewritten.

Python compilation completed with exit code `0`. `git diff --check` passed,
and a changed-line scan found no added CUDA synchronization, completion
fence, event wait, barrier, or request-path wait.

### Review-fix Self-Review

The final diff was checked against the review requirements for evidence
completeness, expected repeat/request identity, cross-rank transport
inventory, command/engine-step reconciliation, optional selected-sequence
digest consistency, disabled compatibility, and synchronization neutrality.
No remaining P0-P2 finding was identified.

Review-fix commit: `SELF/HEAD`.

## Review 2 Fix

Review source:
`.superpowers/sdd/task-4-review-2.md` (`Needs fixes`).

The second review identified three remaining Task 4 validation and readiness
gaps:

- rank zero's existing post-step synchronization was incorrectly broadcast as
  authorization for worker ranks to skip their own profiler completion;
- Python boolean ranks could pass exact-rank comparisons because `True == 1`;
  and
- public repeat labels were type-checked but not bound to warmup `-1` and
  measured positions `0..4`.

The fix adds an explicit `already_synchronized_rank` lifecycle argument while
preserving the existing `already_synchronized` semantics and default. Rank
zero maps the explicit rank to `already_synchronized=True`; ranks 1 through 3
map it to `False` and therefore each execute exactly one existing profiler
synchronization. The all-rank acknowledged dispatch still carries one command
and preserves the original failure object. The worker invokes this finalizer
only after request timing, after the final existing post-step synchronization,
and before deferred CUDA event rows are read.

Timeline validation now sends every command/CUDA snapshot rank and every
command/CUDA row rank through the strict non-boolean non-negative integer
validator before exact `0..3` comparison. The graph lifecycle passes both
expected labels into each run validator: public warmup `-1` maps to timeline
repeat `0`, and public measured positions `0..4` map to timeline repeats
`1..5`.

### Review-2 RED

Test-only focused command before production edits:

```bash
python3 -m pytest -q \
  tools/test_decode_internal_profile_wiring.py::test_rank_aware_profile_finalization_only_reuses_rank_zero_sync \
  tools/test_decode_internal_profile_wiring.py::test_engine_rank_aware_profile_finalization_is_acknowledged \
  tools/test_decode_internal_profile_wiring.py::test_command_timeline_profile_helpers_propagate_all_rank_failure \
  tools/test_autoregressive_draft_performance_gate.py::test_worker_command_timeline_reset_snapshot_order_and_cardinality \
  tools/test_autoregressive_draft_performance_gate.py::test_policy_campaign_command_timeline_rejects_invalid_evidence \
  tools/test_autoregressive_draft_performance_gate.py::test_policy_campaign_command_timeline_rejects_public_repeat_drift
```

Output:

```text
11 failed, 16 passed in 0.33s
```

The failures were three missing rank-aware finalization API/error-propagation
contracts, the worker still passing the all-rank boolean, five boolean-rank
acceptances, and two accepted public-repeat drifts.

### Review-2 GREEN

The same focused command after implementation:

```text
27 passed in 0.17s
```

The final complete focused Task 4 selection, including the explicit
post-measurement worker-fence ordering assertions:

```text
35 passed, 53 deselected in 0.28s
```

Complete Task 4 regression:

```text
141 passed in 2.99s
```

Task 1 and Task 2 regression:

```text
56 passed in 1.93s
```

Task 3 four-file regression:

```text
215 passed in 3.49s
```

Exact inherited-fingerprint regression:

```text
57 passed, 6 failed in 1.89s
```

The inherited failure shape remains unchanged: five failures stop at
`ValueError: LLMEngine source hash is invalid`, and one reports the existing
prerequisite source-closure mismatch. Frozen fingerprints were not rewritten.

Python compilation completed with exit code `0`. `git diff --check` passed.
A production changed-line scan found no added direct CUDA synchronization,
event/stream wait, barrier, completion fence, or request-path wait. The only
new completion behavior is the authorized worker-rank call through the
existing profiler finalizer, outside request timing and after the final
post-step synchronization.

### Review-2 Self-Review

The final diff was checked against rank-local CUDA event readiness,
acknowledged all-rank error propagation, strict rank typing, exact public and
timeline repeat labels, disabled compatibility, and synchronization placement.
No remaining P0-P2 finding was identified.

Review-2 fix commit: `SELF/HEAD`.
