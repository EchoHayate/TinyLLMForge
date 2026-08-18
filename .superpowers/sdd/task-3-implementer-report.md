# Task 3 Implementer Report

## Scope

Implemented only Task 3, Engine Step Envelope and Conservation, from:

- approved spec:
  `docs/superpowers/specs/2026-08-18-autoregressive-draft-command-timeline-sync-debt-design.md`;
- implementation plan:
  `docs/superpowers/plans/2026-08-18-autoregressive-draft-command-timeline-sync-debt.md`;
- task brief: `.superpowers/sdd/task-3-brief.md`.

Base SHA:

```text
7de6f15f357c919185b1dfe5f810089da4b80cce
```

No later task was run. No remote, GPU, CUDA, NCCL, checkpoint, or push action
was performed.

## Implementation

Created `tinyvllm/engine/engine_step_timeline.py` with:

- frozen `EngineStepTraceIdentity`;
- the fixed twelve-phase inventory;
- explicit skipped phase rows;
- one-active-step and one-active-phase enforcement;
- nested, repeated, and mismatched finish rejection;
- disabled clock-free no-op behavior;
- ContextVar scope and recorder-owned active identity;
- bounded deep-copy snapshots and bounded error/detail fields;
- bounded row retention;
- canonical `compute_step_conservation(step, command_rows)` support;
- one compatible explicit-total form for the brief tests;
- absolute/relative tolerance
  `max(2_000_000 ns, ceil(1% * step_wall_ns))`;
- command critical-path and post-local acknowledged-wait accounting without
  double-counting phase intervals;
- fail-closed malformed inventory, overlapping intervals, missing command
  data, negative duration, and over-attribution handling; and
- active ContextVar cleanup on success, operation failure, nesting rejection,
  and finish-clock failure.

Modified `LLMEngine` to:

- create/reset the step recorder with the existing command-timeline setting;
- expose begin/end repeat and step snapshot APIs;
- bind the final batch kind and canonical selected-sequence digest before
  model-runner dispatch;
- wrap exact existing scheduler, dispatch, speculative preparation,
  Scheduler prepare/commit, Proposal-KV commit, proposal lifecycle,
  side-state seal, residency precommit, and ordinary postprocess statements;
- retain the original rollback, poison, commit, seal, and exception ordering;
- finalize failed step telemetry in one outer `try/except/finally`;
- preserve the original engine exception if telemetry finalization also
  fails; and
- attach finalized identity, phases, status, and conservation status to
  `last_step_observation["command_timeline_step"]`.

No completion fence, request-path wait, CUDA Event, or CUDA synchronization was
added.

## RED Evidence

Initial mandatory focused RED:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_engine_step_timeline.py \
  tools/test_engine_speculative_execution.py \
  -k 'timeline or phase or conservation'
```

Output:

```text
14 failed, 1 passed, 27 deselected in 0.21s
```

Thirteen failures were the missing engine-step module. The remaining failure
proved the existing publication operations were not yet wrapped by Task 3
phases.

Additional test-first review fixes:

```text
engine reset clears step rows/repeat:
  1 failed, 15 deselected in 0.07s

active reset rejects before all-rank dispatch:
  1 failed, 16 deselected in 0.13s

missing command data fails closed:
  1 failed, 1 passed, 16 deselected in 0.06s

finish-clock failure cleans active ContextVar:
  1 failed, 18 deselected in 0.07s
```

Each failure was observed before the corresponding production edit.

## GREEN Evidence

Final focused Task 3 command:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_engine_step_timeline.py \
  tools/test_engine_speculative_execution.py \
  -k 'timeline or phase or conservation'
```

Output:

```text
21 passed, 27 deselected in 0.20s
```

The system `/usr/bin/python3` lacks Torch, so the exact planned four-file
regression was run with the existing Torch-capable Homebrew Python 3.12 and a
temporary pytest-only import path. No repository dependency or environment
file was changed.

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312 \
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_engine_step_timeline.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_chunked_prefill.py
```

Output:

```text
208 passed in 5.76s
```

Task 1 and Task 2 regression:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py
```

Output:

```text
56 passed in 1.29s
```

Exact inherited-fingerprint regression:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py \
  tools/test_qwen35_real_binding_engine_ack_transport_preflight.py
```

Output:

```text
57 passed, 6 failed in 1.44s
```

The six failures are inherited:

- five stop at `ValueError: LLMEngine source hash is invalid`;
- one reports the pre-existing prerequisite source-closure mismatch.

No frozen hash expectation was rewritten.

Syntax verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m py_compile \
  tinyvllm/engine/engine_step_timeline.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_engine_step_timeline.py \
  tools/test_engine_speculative_execution.py
```

Output: exit code `0`, no diagnostics.

## Invariant Review

- Task 2 dispatch reads the live Task 3 ContextVar and receives the finalized
  engine-step/repeat identity.
- Scheduler, speculative preparation, Proposal-KV publication, lifecycle
  finalization, Scheduler publication, side-state seal, residency boundaries,
  ordinary postprocess, rollback, and poison operations remain in their
  original order.
- The outer finalizer does not suppress or replace an existing engine
  exception.
- Skipped optional phases are explicit zero-duration rows.
- Canonical command-row conservation subtracts command and post-local ack
  intervals from containing phase spans before adding them as separate
  components.
- Missing or malformed required evidence yields
  `status="invalid"` and `passed=false`.
- No completion fence, `torch.cuda.synchronize()`, or new request-path wait was
  introduced.
