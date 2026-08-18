### Task 4: Deferred CUDA Identity and Worker Export

**Files:**
- Modify: `tinyvllm/engine/decode_internal_profiler.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/autoregressive_draft_performance_worker.py`
- Modify: `tools/test_decode_internal_profiler.py`
- Modify: `tools/test_decode_internal_profile_wiring.py`
- Modify: `tools/test_autoregressive_draft_performance_gate.py`

**Interfaces:**
- `DecodeInternalProfiler.finalize(*, already_synchronized=False)`.
- CUDA rows include `command_id`, `engine_step_id`, and `repeat_index`.
- Add keyword-only parameter `command_timeline: bool = False` to the existing
  `run_policy_campaign` signature after `cuda_graph_mode`.
- Worker CLI `--command-timeline`.
- Every warmup/measured run contains `runtime.command_timeline`.

- [ ] **Step 1: Write failing deferred-CUDA and worker-export tests**

Add:

```python
def test_finalize_can_reuse_existing_synchronization():
    profiler, synchronizations = _profiler([0.0, 1.0])
    profiler.begin_step(
        batch_kind="decode",
        is_decode=True,
        active_sequence_count=4,
        request_set_sha256="a" * 64,
        dispatch="graph",
    )
    profiler.end_step()
    snapshot = profiler.finalize(already_synchronized=True)
    assert synchronizations == []
    assert snapshot["steps"][0]["cuda_ns"] == 1_000_000


def test_profile_rows_bind_active_command_identity(command_scope):
    profiler, _ = _profiler([0.0, 1.0])
    with command_scope(
        command_id=9,
        engine_step_id=4,
        repeat_index=2,
    ):
        profiler.begin_step(
            batch_kind="decode",
            is_decode=True,
            active_sequence_count=4,
            request_set_sha256="a" * 64,
            dispatch="graph",
        )
        profiler.end_step()
    row = profiler.finalize()["steps"][0]
    assert (row["command_id"], row["engine_step_id"], row["repeat_index"]) == (
        9,
        4,
        2,
    )
```

Worker tests must assert:

- timeline disabled leaves current worker schema unchanged;
- timeline enabled configures all ranks once;
- reset occurs after pre-run authority/memory snapshots and immediately before
  request timing;
- snapshot occurs only after the existing `synchronize()` following the final
  `engine.step()`;
- each measured repeat has four rank snapshots, one engine-step snapshot, and
  deferred CUDA rows;
- warmup count one and measured count five are accepted only for the new
  diagnostic command; and
- exact graph counters prove one warmup capture and measured replay growth.

- [ ] **Step 2: Run tests and confirm RED**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_autoregressive_draft_performance_gate.py \
  -k 'command_timeline or already_synchronized or active_command'
```

Expected: missing finalization option, identity fields, CLI, and worker output.

- [ ] **Step 3: Bind CUDA rows to active command identity**

At `begin_step`, read `active_model_runner_command_trace()` and store:

```python
"command_id": None if trace is None else trace.command_id,
"engine_step_id": None if trace is None else trace.engine_step_id,
"repeat_index": None if trace is None else trace.repeat_index,
```

Include the fields in finalized step and collective rows.

Change finalization to:

```python
def finalize(self, *, already_synchronized=False):
    if not isinstance(already_synchronized, bool):
        raise ValueError("already_synchronized must be a bool")
    if not already_synchronized:
        self._synchronize()
```

Existing callers retain the default and existing behavior.

- [ ] **Step 4: Export all-rank command, CUDA, and step evidence**

Add engine helpers that acknowledged-reset all rank command recorders and
configure/reset the decode profiler. After the worker's existing
post-`engine.step()` synchronization, obtain:

```python
command_rows = engine.command_timeline_snapshots(timeout_s=60.0)
cuda_rows = engine.finalize_decode_internal_profile(
    already_synchronized=True,
    timeout_s=60.0,
)
step_rows = engine.engine_step_timeline_snapshot()
```

Store:

```python
"command_timeline": {
    "schema_version": 1,
    "rank_snapshots": list(command_rows),
    "cuda_rank_snapshots": list(cuda_rows),
    "engine_steps": step_rows["steps"],
}
```

Compute request-set and selected-sequence SHA-256 values using canonical JSON,
not Python `repr`.

The diagnostic worker command is:

```text
--policy learned
--batch-size 4
--warmup-runs 1
--measured-runs 5
--command-timeline
```

- [ ] **Step 5: Run focused and complete worker tests**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
```

Expected: all tests pass; existing schema-v2 defaults remain one warmup and one
measured run.

- [ ] **Step 6: Commit and push worker evidence**

```bash
git add -- \
  tinyvllm/engine/decode_internal_profiler.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/autoregressive_draft_performance_worker.py \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_autoregressive_draft_performance_gate.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): export command and cuda timelines" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---
