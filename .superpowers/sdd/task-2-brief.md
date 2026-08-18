### Task 2: Shared-Memory Command and Ack-Wait Wiring

**Files:**
- Modify: `tinyvllm/engine/model_runner_command_ack.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_model_runner_command_ack.py`
- Modify: `tools/test_model_runner_live_ack_wiring.py`

**Interfaces:**
- `ModelRunnerCommandEnvelope.trace_identity: CommandTraceIdentity | None`.
- `execute_acknowledged_command(envelope, *, rank, target, send_ack, timeline=None, clock_ns=time.monotonic_ns)`.
- `ModelRunner.configure_command_timeline(enabled, max_rows) -> dict`.
- `ModelRunner.reset_command_timeline() -> dict`.
- `ModelRunner.command_timeline_snapshot() -> dict`.
- `LLMEngine.configure_command_timeline(enabled, max_rows, timeout_s) -> dict`.
- `LLMEngine.reset_command_timeline(timeout_s) -> tuple[dict, ...]`.
- `LLMEngine.command_timeline_snapshots(timeout_s) -> tuple[dict, ...]`.

- [ ] **Step 1: Write traced transport and disabled-compatibility tests**

Extend `tools/test_model_runner_command_ack.py`:

```python
def test_traced_executor_records_method_and_ack_boundaries():
    timeline = _FakeTimeline()
    clock = iter((100, 200, 300, 400)).__next__
    envelope = ModelRunnerCommandEnvelope(
        command_id=31,
        method_name="add",
        args=(4, 5),
        requires_ack=True,
        trace_identity=make_trace_identity(command_id=31, requires_ack=True),
    )
    sent = []

    assert execute_acknowledged_command(
        envelope,
        rank=2,
        target=_Target(),
        send_ack=sent.append,
        timeline=timeline,
        clock_ns=clock,
    ) == 9
    assert timeline.events == [
        ("method_start", 31, 100),
        ("method_end", 31, 200, "ok", ""),
        ("ack_start", 31, 300),
        ("ack_end", 31, 400),
    ]


def test_untraced_envelope_preserves_existing_semantics():
    envelope = ModelRunnerCommandEnvelope(
        command_id=32,
        method_name="add",
        args=(1, 2),
        requires_ack=False,
    )
    assert envelope.trace_identity is None
    assert execute_acknowledged_command(
        envelope,
        rank=1,
        target=_Target(),
        send_ack=lambda value: None,
    ) == 3
```

Extend `tools/test_model_runner_live_ack_wiring.py` to assert:

- `dispatch_command` stamps one trace identity only when enabled;
- `write_shm` serializes the final publish timestamp before `Event.set()`;
- `read_shm` records wake/read timestamps before returning the envelope;
- rank-zero local execution uses the same command ID as workers;
- `call_model_runner_acknowledged` records ack-wait start/end around
  `collector.collect`;
- disabled mode still emits envelope equality compatible with existing tests;
- configure/reset/snapshot operations use acknowledged all-rank calls; and
- snapshot operations are excluded from the returned measured timeline by
  reset-before-run and snapshot-after-run boundaries.

- [ ] **Step 2: Run focused tests and confirm RED**

```bash
cd /Users/bytedance/Desktop/TinyLLMForge
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py \
  -k 'timeline or traced or acknowledged_call'
```

Expected: failures show missing `trace_identity`, timeline hooks, and
configure/reset/snapshot methods.

- [ ] **Step 3: Extend the command envelope and execution helper**

In `model_runner_command_ack.py`, import `CommandTraceIdentity` and add:

```python
@dataclass(frozen=True)
class ModelRunnerCommandEnvelope:
    command_id: int
    method_name: str
    args: tuple
    requires_ack: bool
    trace_identity: CommandTraceIdentity | None = None

    def __post_init__(self):
        # retain existing validation
        if (
            self.trace_identity is not None
            and self.trace_identity.command_id != self.command_id
        ):
            raise ValueError("trace identity command mismatch")
        if (
            self.trace_identity is not None
            and self.trace_identity.method_name != self.method_name
        ):
            raise ValueError("trace identity method mismatch")
        if (
            self.trace_identity is not None
            and self.trace_identity.requires_ack != self.requires_ack
        ):
            raise ValueError("trace identity acknowledgement mismatch")
```

Wrap target method execution in `command_trace_scope(trace_identity)`.
Record method and ack-send boundaries only when both trace identity and an
enabled recorder exist. Preserve:

- fire-and-forget exception propagation;
- acknowledged `Exception` conversion to error ack;
- `BaseException` propagation;
- ack-send failure propagation; and
- existing bounded error detail.

- [ ] **Step 4: Wire rank-zero dispatch, worker receive, and engine ack wait**

In `ModelRunner.__init__`, install a disabled recorder and injected
`time.monotonic_ns` clock. Add the three lifecycle methods.

`dispatch_command` must:

1. allocate `command_id`;
2. read active engine step/repeat trace context;
3. record `dispatch_started_monotonic_ns`;
4. read `dispatch_published_monotonic_ns` immediately before serializing the
   final envelope;
5. write the envelope and set worker events; and
6. record rank-zero dispatch.

`call()` and `LLMEngine.call_model_runner_acknowledged()` must invoke the local
method through the recorder with the same envelope identity.

`read_shm()` must record:

```python
event_woken_monotonic_ns = self._command_timeline_clock_ns()
n = int.from_bytes(self.shm.buf[0:4], "little")
envelope = pickle.loads(self.shm.buf[4:n + 4])
envelope_read_monotonic_ns = self._command_timeline_clock_ns()
```

Then `loop()` passes `self.command_timeline` and the clock into
`execute_acknowledged_command`.

`LLMEngine.call_model_runner_acknowledged()` records:

```python
ack_wait_started = self._clock_ns()
worker_acks = collector.collect(
    envelope.command_id,
    expected_ranks=tuple(range(1, self.model_runner.world_size)),
    timeout_s=timeout_s,
    is_rank_alive=self._is_worker_rank_alive,
)
ack_wait_finished = self._clock_ns()
self.model_runner.command_timeline.record_ack_wait(
    envelope.command_id,
    started_ns=ack_wait_started,
    finished_ns=ack_wait_finished,
)
```

TP1 remains local-only and records no worker ack wait.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py \
  tools/test_qwen35_real_binding_engine_ack_transport_preflight.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit and push transport wiring**

```bash
git add -- \
  tinyvllm/engine/model_runner_command_ack.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): trace model runner command debt" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---
