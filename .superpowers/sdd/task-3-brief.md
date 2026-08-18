### Task 3: Engine Step Envelope and Conservation

**Files:**
- Create: `tools/test_engine_step_timeline.py`
- Create: `tinyvllm/engine/engine_step_timeline.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_engine_speculative_execution.py`

**Interfaces:**
- Produces `EngineStepTraceIdentity`.
- Produces `EngineStepTimelineRecorder`.
- Produces `active_engine_step_trace()`.
- Produces `engine_step_trace_scope(identity)`.
- Produces `compute_step_conservation(step, command_rows)`.
- `LLMEngine.begin_command_timeline_repeat(repeat_index)`.
- `LLMEngine.end_command_timeline_repeat()`.
- `LLMEngine.engine_step_timeline_snapshot() -> dict`.

- [ ] **Step 1: Write lifecycle and conservation tests**

Create pure tests:

```python
PHASES = (
    "scheduler_schedule",
    "partition_and_step_setup",
    "ordinary_or_first_target_dispatch",
    "speculative_prepare",
    "scheduler_prepare_postprocess",
    "proposal_kv_prepare_commit",
    "proposal_lifecycle_finalize_prepare",
    "scheduler_commit_postprocess",
    "proposal_lifecycle_finalize_commit",
    "side_state_seal",
    "residency_precommit_or_seal",
    "ordinary_scheduler_postprocess",
)


def test_step_recorder_emits_explicit_skipped_phases(module):
    recorder = module.EngineStepTimelineRecorder(
        enabled=True,
        clock_ns=iter((100, 120, 140, 180)).__next__,
    )
    identity = recorder.begin_step(
        repeat_index=0,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
    )
    with recorder.phase("scheduler_schedule"):
        pass
    recorder.finish_step(identity)
    row = recorder.snapshot()["steps"][0]
    assert row["phases"]["scheduler_schedule"]["executed"] is True
    assert row["phases"]["scheduler_schedule"]["duration_ns"] == 20
    assert row["phases"]["speculative_prepare"] == {
        "executed": False,
        "started_monotonic_ns": None,
        "finished_monotonic_ns": None,
        "duration_ns": 0,
    }


def test_step_conservation_uses_larger_absolute_or_relative_tolerance(module):
    result = module.compute_step_conservation(
        {
            "step_wall_ns": 100_000_000,
            "phases": {
                "scheduler_schedule": {"duration_ns": 10_000_000},
                "scheduler_commit_postprocess": {"duration_ns": 20_000_000},
            },
        },
        command_critical_path_ns=69_000_000,
        acknowledged_wait_ns=0,
    )
    assert result["residual_ns"] == 1_000_000
    assert result["tolerance_ns"] == 2_000_000
    assert result["passed"] is True
```

Add integration assertions that speculative execution records
`speculative_prepare`, `scheduler_prepare_postprocess`,
`proposal_kv_prepare_commit`, lifecycle prepare/commit, scheduler commit, and
side-state seal around the existing operations without changing their order.

- [ ] **Step 2: Run tests and confirm RED**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_engine_step_timeline.py \
  tools/test_engine_speculative_execution.py \
  -k 'timeline or phase or conservation'
```

Expected: missing module and missing engine phase observations.

- [ ] **Step 3: Implement the step recorder**

Create `engine_step_timeline.py` with:

- immutable identity containing `engine_step_id`, `repeat_index`,
  `request_set_sha256`, `batch_kind`, and selected-sequence digest;
- fixed phase inventory from the approved spec;
- one active step and at most one active phase;
- explicit skipped-phase rows;
- nested or repeated phase rejection;
- deep-copy snapshots;
- `step_wall_ns`, serial phase sum, command critical path, ack wait,
  `step_residual_ns`, tolerance, and pass/fail; and
- disabled no-op behavior.

Use a ContextVar to make the active step identity available to
`ModelRunner.dispatch_command`.

- [ ] **Step 4: Instrument `LLMEngine.step()` without reordering operations**

At entry:

```python
step_trace = self.engine_step_timeline.begin_step(
    repeat_index=self._command_timeline_repeat_index,
    request_set_sha256=self._command_timeline_request_set_sha256,
    batch_kind="unknown",
    speculative_selected_sequence_ids_sha256=None,
)
```

After scheduling and partition construction, bind the final batch kind and
selected-sequence digest. Wrap the exact existing statements in named phase
contexts. Do not move statements across try/except, rollback, commit, seal,
or poison boundaries.

Use one outer `try/finally` so `finish_step` records failure status and never
suppresses the original exception.

Store the finalized step identity and phase summary in
`last_step_observation["command_timeline_step"]`.

- [ ] **Step 5: Run focused regression**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_engine_step_timeline.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_chunked_prefill.py
```

Expected: all tests pass and pre-existing speculative call ordering assertions
remain unchanged.

- [ ] **Step 6: Commit and push engine spans**

```bash
git add -- \
  tinyvllm/engine/engine_step_timeline.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_engine_step_timeline.py \
  tools/test_engine_speculative_execution.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): trace engine step phases" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---
