# Task 2 Implementer Report

## Files changed

- `.superpowers/sdd/task-2-brief.md`
- `.superpowers/sdd/progress.md`
- `.superpowers/sdd/task-2-implementer-report.md`
- `tinyvllm/engine/model_runner_command_ack.py`
- `tinyvllm/engine/model_runner.py`
- `tinyvllm/engine/llm_engine.py`
- `tools/test_model_runner_command_ack.py`
- `tools/test_model_runner_live_ack_wiring.py`

## Implementation summary

- Added optional `CommandTraceIdentity` transport to
  `ModelRunnerCommandEnvelope`, including fail-closed command, method, and
  acknowledgement identity validation.
- Wrapped worker and rank-zero method execution in the active command trace
  scope and recorded method plus worker ack-send boundaries only for traced
  envelopes with an enabled recorder.
- Installed a default-disabled rank-local recorder in every `ModelRunner`,
  with configure, reset, and snapshot lifecycle methods.
- Stamped dispatch identity only when the recorder is enabled and Task 3's
  lazy active engine-step context supplies both an engine step and repeat.
- Serialized the final traced envelope before setting worker events, then
  recorded worker wake/read timestamps before returning the envelope.
- Routed rank-zero ordinary and acknowledged execution through the exact
  dispatched envelope, preserving the same command ID and trace identity
  across rank zero and workers.
- Timed TP>1 acknowledgement collection and recorded the wait on rank zero.
  TP1 remains local-only, uses a non-acknowledged local envelope, and records
  no worker ack wait.
- Added acknowledged all-rank engine operations for configure, reset, and
  snapshot. Because management calls run outside an active measured
  step/repeat context, they remain untraced and do not contaminate returned
  measured snapshots.

## RED evidence

Initial Task 2 RED command:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py \
  -k 'timeline or traced or acknowledged_call'
```

Output:

```text
5 failed, 30 deselected in 0.21s
```

The failures were the intended missing contracts:

- `ModelRunnerCommandEnvelope` did not accept or expose `trace_identity`;
- traced executor method/ack hooks were absent;
- TP1 and TP>1 rank-zero execution bypassed the dispatched envelope; and
- engine timeline configure/reset/snapshot methods were absent.

During self-review, a focused TP1 lifecycle test was added after temporarily
restoring the prior `requires_ack=True` behavior:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_live_ack_wiring.py \
  -k 'tp1_traced_local_call_finishes_without_ack_wait'
```

RED output:

```text
1 failed, 19 deselected in 0.13s
```

The exact failure was:

```text
ValueError: cannot snapshot command timeline with unfinished rows
```

The minimum correction was to preserve TP1's pre-existing local-only
semantics by dispatching its local envelope with `requires_ack=False`.

## GREEN and regression evidence

TP1 lifecycle GREEN:

```text
1 passed, 19 deselected in 0.10s
```

Final selected Task 2 GREEN command:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py \
  -k 'timeline or traced or acknowledged_call'
```

Output:

```text
7 passed, 30 deselected in 0.25s
```

Task 1 core plus complete Task 2 command transport regression:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py
```

Output:

```text
48 passed in 0.94s
```

Exact planned regression command:

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
49 passed, 6 failed in 1.16s
```

All six failures are in the known frozen
`tools/test_qwen35_real_binding_engine_ack_transport_preflight.py`
source-fingerprint family:

- five tests stop at
  `ValueError: LLMEngine source hash is invalid`; and
- one test reports the frozen prerequisite source-closure mismatch.

Base/current comparison established:

```text
prerequisite closure mismatches at Task 2 base SHA: 23
prerequisite closure mismatches after Task 2:       23
new prerequisite mismatch paths introduced:         0
```

The frozen direct source hashes also had pre-existing drift at the Task 2
base:

```text
LLMEngine base SHA:
  actual   4a2d7f4303cf703f6f2d418656e78d87427201319e9b5edcbef00fc749450de8
  frozen   6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae

ModelRunner base SHA:
  actual   bc5f435e85fa97d83497f65b4e5901df122a02d529635db09167336e204e9eca
  frozen   0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849
```

Task 2 intentionally changes the live `LLMEngine`, `ModelRunner`, and
acknowledgement transport sources. The preflight is an immutable historical
fingerprint contract that was already red at the supplied base SHA, so Task 2
does not refresh its file or method hashes merely to turn the regression
green.

Syntax verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m py_compile \
  tinyvllm/engine/model_runner_command_ack.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py
```

Output: exit code 0 with no diagnostics.

## Invariant checks

- Trace identity is absent while the recorder is disabled, while no active
  engine-step context exists, or while no measured repeat is active.
- The final traced envelope, including
  `dispatch_published_monotonic_ns`, is serialized before any worker
  `Event.set()`.
- Worker wake and envelope-read timestamps are captured before `read_shm()`
  returns the envelope.
- Rank zero executes the exact dispatched envelope; TP>1 workers deserialize
  that same command ID and trace identity.
- TP1 remains local-only, records a complete local method row, and has null
  worker ack-wait timestamps.
- Ordinary `ModelRunner.call()` dispatch remains `requires_ack=False`.
- TP>1 acknowledged commands preserve rank-zero exception poisoning,
  worker `Exception` to error-ack conversion, `BaseException` propagation,
  bounded error detail, and ack-send failure propagation.
- Configure, reset, and snapshot use
  `call_model_runner_acknowledged()` across the complete rank inventory.
  Management calls are outside active measured contexts, so reset-before-run
  and snapshot-after-run do not add management rows to measured snapshots.
- No completion fence, `torch.cuda.synchronize()`, CUDA Event, or other new
  request-path synchronization was added by Task 2.
- No artifact, experiment, log, PID, archive, remote output, review-package
  diff, or retired adaptive-ngram path was modified or staged.

## Residual concerns

- Task 2 provides a lazy bridge to `active_engine_step_trace()`. The context
  module and repeat lifecycle are intentionally owned by Task 3, so live
  measured commands remain untraced until Task 3 activates that context.
- No GPU, CUDA, NCCL, checkpoint, SSH, remote write, or remote workload was
  run.
- The six frozen fingerprint failures remain intentionally visible and must
  not be interpreted as Task 2 GREEN.
- Independent task review is still pending and is owned by the controller.

## Commit

`SELF/HEAD`

---

## Review 1 fix

Review input:

- `.superpowers/sdd/task-2-review-1.md`

Review-fix files:

- `.superpowers/sdd/progress.md`
- `.superpowers/sdd/task-2-implementer-report.md`
- `.superpowers/sdd/task-2-review-1.md`
- `tinyvllm/engine/model_runner_command_timeline.py`
- `tinyvllm/engine/model_runner_command_ack.py`
- `tinyvllm/engine/model_runner.py`
- `tinyvllm/engine/llm_engine.py`
- `tools/test_model_runner_command_timeline.py`
- `tools/test_model_runner_command_ack.py`
- `tools/test_model_runner_live_ack_wiring.py`

### Review-fix implementation

- Excluded `configure_command_timeline`, `reset_command_timeline`, and
  `command_timeline_snapshot` from trace-context reads and trace-identity
  creation. This exclusion applies even if an enabled recorder observes a
  deliberately stale active measured step/repeat context.
- Added an explicit recorder terminal-error transition for the active
  `method`, `awaiting_ack`, `ack_send`, and `ack_wait` phases.
- Terminal rows use:
  - `status="error"`;
  - bounded `error_type`;
  - UTF-8 `error_detail` bounded to 4096 bytes, matching the existing
    acknowledgement detail limit;
  - `terminal_error_monotonic_ns`; and
  - the corresponding method, ack-send, or ack-wait finish timestamp when
    that phase had already started.
- Worker ack-send failure, rank-zero local execution failure, and rank-zero
  collector timeout/error now terminalize traced rows without replacing the
  original exception. Recorder cleanup is best-effort only so observation
  failures cannot mask transport or execution failures.
- Rank-zero local failure still poisons the acknowledgement collector before
  re-raising. Collector timeout/error poisoning remains owned by the
  collector and is unchanged.
- The lazy engine-step bridge now suppresses only
  `ModuleNotFoundError` whose `name` is exactly
  `tinyvllm.engine.engine_step_timeline`; nested dependency failures are
  re-raised.

### Review-fix RED evidence

The mandatory tests were added before review-fix production edits and run
with:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py::test_terminal_error_closes_awaiting_ack_with_bounded_detail \
  tools/test_model_runner_command_ack.py::test_traced_ack_send_failure_preserves_error_and_terminalizes_timeline \
  tools/test_model_runner_live_ack_wiring.py::test_model_runner_management_dispatch_ignores_stale_measured_trace \
  tools/test_model_runner_live_ack_wiring.py::test_model_runner_lazy_engine_step_import_only_suppresses_absent_module \
  tools/test_model_runner_live_ack_wiring.py::test_engine_traced_local_exception_terminalizes_timeline \
  tools/test_model_runner_live_ack_wiring.py::test_engine_traced_collector_failure_terminalizes_ack_wait
```

Output:

```text
6 failed in 0.37s
```

The intended failures were:

- missing `record_terminal_error`;
- worker ack-send `OSError` left `ack_send` unfinished;
- management dispatch read the trace clock under stale context;
- nested `ModuleNotFoundError` was hidden;
- rank-zero local `ValueError` left `awaiting_ack` unfinished; and
- collector timeout left the acknowledged row unfinished.

### Review-fix GREEN and regression evidence

Focused GREEN, rerun after the final test refinement:

```text
6 passed in 0.26s
```

The collector case covers both `TimeoutError` and a non-timeout
`RuntimeError` while asserting object identity of the re-raised exception.

Complete Task 1 core plus Task 2 transport regression:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py
```

Output:

```text
54 passed in 1.11s
```

Exact planned regression:

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
55 passed, 6 failed in 1.32s
```

The six failures remain the inherited frozen source-fingerprint boundary:

- five fail with
  `ValueError: LLMEngine source hash is invalid`; and
- one reports the pre-existing prerequisite source-closure mismatch.

Task 2 owns intentional changes to the live `LLMEngine`, `ModelRunner`,
acknowledgement transport, and command timeline sources. It does not own the
immutable historical preflight fingerprint contract, which was already red
at the supplied Task 2 base. No frozen hash expectation was changed.

Syntax verification:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m py_compile \
  tinyvllm/engine/model_runner_command_timeline.py \
  tinyvllm/engine/model_runner_command_ack.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_model_runner_command_timeline.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py
```

Output: exit code 0 with no diagnostics.

### Review-fix invariant checks

- Enabled stale measured context cannot stamp any of the three management
  operations, and snapshot cannot trace or block on itself.
- Worker send failure preserves and re-raises the original `OSError` while
  producing a snapshot-able terminal error row.
- Rank-zero local failure preserves the original exception, retains
  collector poisoning, and closes `awaiting_ack`.
- Collector timeout and non-timeout failure preserve the original exception
  object and close `ack_wait`.
- Existing worker `Exception` to error-ack conversion, `BaseException`
  propagation, acknowledgement send propagation, and bounded ack detail
  remain covered by the complete regression.
- Rank zero and workers still use the same command ID and trace identity.
- TP1 remains local-only with no worker ack wait.
- Ordinary `call()` commands remain `requires_ack=False`.
- Configure/reset/snapshot remain acknowledged all-rank operations.
- No completion fence, `torch.cuda.synchronize()`, CUDA Event, or other
  measured request-path synchronization was added.

### Review-fix residual concerns

- Task 3's engine-step context module is still intentionally absent. Exact
  absence remains a lazy no-op; a future nested dependency failure will now
  surface instead of being misclassified as absence.
- Terminal cleanup is deliberately best-effort in exception handlers to
  guarantee that telemetry cannot replace the original execution or
  transport exception.
- No GPU, CUDA, NCCL, checkpoint, SSH, remote write, or remote workload was
  run.
- The six frozen fingerprint failures remain intentionally visible.

### Review-fix commit

`SELF/HEAD`
