# Autoregressive Draft Command Timeline and Sync-Debt Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a default-off, source-bound command and engine-step timeline that can localize TP4/B4/Q4 graph-versus-eager E2E dispersion to worker queue debt, CUDA execution, acknowledgement wait, or scheduler/postprocess without changing runtime semantics.

**Architecture:** Add two small runtime profiler modules: one owns per-command cross-rank trace identity and timestamps, and one owns non-overlapping `LLMEngine.step()` phase spans. Wire them through the existing shared-memory command envelope, acknowledgement collector, deferred CUDA Event profiler, and autoregressive-draft performance worker. Build a new balanced eight-epoch diagnostic, independent verifier, and safe remote runner by reusing the existing paired-stability admission, host/GPU telemetry, source-bound packaging, dual-receipt, and checksum-manifest patterns without modifying immutable schema-v2 `r3`.

**Tech Stack:** Python 3.11, dataclasses, contextvars, `time.monotonic_ns`, PyTorch CUDA Events, multiprocessing shared memory/Event/Pipe, pytest 8.4.2, Bash/SSH/rsync, existing TinyLLMForge CUDA Graph and paired-stability tooling.

## Global Constraints

- The authoritative date for this plan is Tuesday, August 18, 2026.
- Modify only `/Users/bytedance/Desktop/TinyLLMForge`, whose physical path is `/Users/bytedance/dev/TinyLLMForge`.
- Never modify or package `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep branch `feat/kv-sparse-attention`.
- Preserve immutable campaign `20260817-steady-state-schema-v2-tp4-b4-q4-r3`.
- Do not stage `artifacts/`, `experiments/`, source archives, PID files, worker logs, or raw remote output.
- Stage only explicit intended source, test, spec, plan, audit, and handoff paths.
- Every commit uses `git -c core.hooksPath=/dev/null commit`.
- Every commit message ends with exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push every completed versionable slice to `origin/feat/kv-sparse-attention`.
- Keep command timeline observation default-off.
- Do not change ordinary command `requires_ack=False` behavior.
- Do not insert a completion fence into the measured path.
- Do not add `torch.cuda.synchronize()` to the measured request path.
- Resolve CUDA Events only after the worker's existing per-step or terminal synchronization.
- Preserve exact independent-Qwen3 TP4/B4/Q4 identity, prompt length 256, output length 16, temperature zero, direct Proposal-KV allocation, Proposal-KV offload disabled, and no padding or shape rounding.
- Preserve exact target/proposal/accepted-token parity, accepted-prefix commit, rejected-suffix rollback, zero active transactions, TP failure convergence, and replay-started fail-closed behavior.
- Reuse the fixed balanced schedule `eager_graph, graph_eager, graph_eager, eager_graph`.
- Use one in-process warmup and five measured batches per epoch.
- Reuse paired-stability admission thresholds `MAD / median <= 0.10` and `half_drift <= 0.15`.
- Use timing-conservation tolerance `max(2_000_000 ns, 1% of step wall time)`.
- Require one named boundary to explain at least `60%` of absolute paired E2E delta in at least three of four blocks, with the same sign in at least three blocks and median unexplained residual at most `10%` of median E2E.
- Local implementation and validation do not authorize SSH, remote writes, GPU, CUDA, NCCL, checkpoint execution, or a remote workload.
- Do not terminate, pause, signal, or otherwise modify unrelated GPU or host processes.
- A fresh remote bundle requires separate explicit user authorization after local implementation is green.
- Do not implement a runtime optimization until a verified `BOUNDARY_LOCALIZED` result exists and a separate optimization design is approved.

## File Map

- Create `tinyvllm/engine/model_runner_command_timeline.py`: trace identity, clock identity, bounded rank-local recorder, active-command context, queue/debt arithmetic, and canonical snapshots.
- Create `tinyvllm/engine/engine_step_timeline.py`: engine-step identity, named non-overlapping host phase recorder, conservation calculation, and canonical snapshots.
- Modify `tinyvllm/config.py`: add validated default-off command timeline configuration and bounded row capacity.
- Modify `tinyvllm/engine/model_runner_command_ack.py`: carry optional trace identity, record worker method and ack-send boundaries, and expose ack-collection timing hooks without changing success/error semantics.
- Modify `tinyvllm/engine/model_runner.py`: install the recorder, timestamp shared-memory publication/read, wrap rank-zero local execution, expose acknowledged configure/reset/snapshot methods, and bind CUDA profile rows to active commands.
- Modify `tinyvllm/engine/llm_engine.py`: install the step recorder, set repeat/step trace context, time named scheduler/postprocess phases, record ack wait, and expose all-rank timeline lifecycle methods.
- Modify `tinyvllm/engine/decode_internal_profiler.py`: attach command/step/repeat identity to deferred CUDA Event rows and permit finalization after an already-completed synchronization.
- Modify `tools/autoregressive_draft_performance_worker.py`: enable diagnostic-only timeline mode, reset before every warmup/measured batch, export all-rank command/CUDA/step rows after the existing synchronization, and retain exact graph/eager identity.
- Modify `tools/autoregressive_draft_cuda_graph_gate.py`: allow the new runner to reuse worker command construction with one warmup and five measured repeats without changing schema-v2 gate defaults.
- Create `tools/autoregressive_draft_command_timeline_diagnostic.py`: fixed schedule, exact identity/parity validation, timeline joins, sync-debt decomposition, conservation, stationarity, paired effects, localization, canonical artifact, and CLI.
- Create `tools/verify_autoregressive_draft_command_timeline_diagnostic.py`: source/raw-input/manifest binding, full recomputation, canonical equality, and receipt CLI.
- Create `tools/run_autoregressive_draft_command_timeline_remote.py`: immutable tag, Kerberos/GPU preflight, source archive, balanced epoch execution, telemetry ownership, canonical assembly, remote/local verification, transfer, and manifest.
- Create `tools/test_model_runner_command_timeline.py`: pure timeline schema, ordering, arithmetic, capacity, and disabled-mode tests.
- Create `tools/test_engine_step_timeline.py`: phase lifecycle, optional phases, nesting rejection, and conservation tests.
- Modify `tools/test_model_runner_command_ack.py`: traced acknowledged/non-acknowledged execution and collector timing tests.
- Modify `tools/test_model_runner_live_ack_wiring.py`: shared-memory and rank-zero wiring contracts.
- Modify `tools/test_decode_internal_profiler.py`: active command identity and no-extra-sync finalization tests.
- Modify `tools/test_decode_internal_profile_wiring.py`: production wiring assertions.
- Modify `tools/test_autoregressive_draft_performance_gate.py`: worker timeline enable/reset/snapshot and five-repeat export tests.
- Create `tools/test_autoregressive_draft_command_timeline_diagnostic.py`: exact identity, arithmetic, admission, classification, verifier, manifest, and runner tests.
- Modify `AGENT_HANDOFF_STATE.md`: append local implementation status and retain the remote authorization boundary only after local verification is green.
- Create `docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md`: prompt-to-artifact local completion audit.

---

### Task 1: Default-Off Command Timeline Core

**Files:**
- Create: `tools/test_model_runner_command_timeline.py`
- Create: `tinyvllm/engine/model_runner_command_timeline.py`
- Modify: `tinyvllm/config.py`
- Modify: `tools/test_autoregressive_draft_cuda_graph_config.py`

**Interfaces:**
- Produces `CommandTraceIdentity`.
- Produces `CommandClockIdentity`.
- Produces `ModelRunnerCommandTimelineRecorder`.
- Produces `active_model_runner_command_trace()`.
- Produces `command_trace_scope(identity)`.
- Produces `compute_command_decomposition(rows)`.
- Configures `Config.autoregressive_draft_command_timeline: bool = False`.
- Configures `Config.autoregressive_draft_command_timeline_max_rows: int = 8192`.

- [ ] **Step 1: Write failing config and pure-recorder tests**

Create `tools/test_model_runner_command_timeline.py` with dependency-light
loading and these executable contracts:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "tinyvllm" / "engine" / "model_runner_command_timeline.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "model_runner_command_timeline",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def identity(module, command_id=7, requires_ack=False):
    return module.CommandTraceIdentity(
        command_id=command_id,
        method_name="run",
        requires_ack=requires_ack,
        engine_step_id=3,
        repeat_index=2,
        request_set_sha256="a" * 64,
        batch_kind="decode",
        speculative_selected_sequence_ids_sha256="b" * 64,
        dispatch_started_monotonic_ns=1_000,
        dispatch_published_monotonic_ns=1_100,
    )


def test_disabled_recorder_is_empty_and_side_effect_free():
    module = load_module()
    recorder = module.ModelRunnerCommandTimelineRecorder.disabled(rank=2)
    recorder.record_worker_receive(
        identity(module),
        event_woken_monotonic_ns=1_200,
        envelope_read_monotonic_ns=1_250,
    )
    assert recorder.snapshot() == {
        "schema_version": 1,
        "rank": 2,
        "enabled": False,
        "clock": None,
        "rows": [],
        "dropped_rows": 0,
    }


def test_recorder_requires_strict_command_order():
    module = load_module()
    recorder = module.ModelRunnerCommandTimelineRecorder(
        rank=1,
        max_rows=8,
        clock_identity=module.CommandClockIdentity(
            boot_id="boot",
            implementation="clock_gettime(CLOCK_MONOTONIC)",
            resolution_s=1e-9,
            monotonic=True,
            adjustable=False,
        ),
    )
    recorder.record_worker_receive(
        identity(module, 1),
        event_woken_monotonic_ns=1_200,
        envelope_read_monotonic_ns=1_250,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        recorder.record_worker_receive(
            identity(module, 1),
            event_woken_monotonic_ns=1_300,
            envelope_read_monotonic_ns=1_350,
        )


def test_queue_debt_arithmetic_separates_prior_overlap():
    module = load_module()
    rows = [
        {
            "rank": 1,
            "command_id": 1,
            "dispatch_published_monotonic_ns": 100,
            "method_started_monotonic_ns": 120,
            "method_finished_monotonic_ns": 260,
            "cuda_ns": 100,
        },
        {
            "rank": 1,
            "command_id": 2,
            "dispatch_published_monotonic_ns": 200,
            "method_started_monotonic_ns": 260,
            "method_finished_monotonic_ns": 320,
            "cuda_ns": 40,
        },
    ]
    result = module.compute_command_decomposition(rows)
    assert result[1]["worker_queue_wait_ns"] == 60
    assert result[1]["queued_behind_prior_command_ns"] == 60
    assert result[1]["worker_ready_delay_ns"] == 0
    assert result[1]["worker_non_cuda_upper_bound_ns"] == 20
```

Extend `tools/test_autoregressive_draft_cuda_graph_config.py`:

```python
def test_command_timeline_defaults_off_and_validates_capacity(config_factory):
    config = config_factory()
    assert config.autoregressive_draft_command_timeline is False
    assert config.autoregressive_draft_command_timeline_max_rows == 8192

    with pytest.raises(
        ValueError,
        match="autoregressive_draft_command_timeline must be a bool",
    ):
        config_factory(autoregressive_draft_command_timeline=1)
    with pytest.raises(
        ValueError,
        match="command timeline max rows must be a positive integer",
    ):
        config_factory(autoregressive_draft_command_timeline_max_rows=0)
```

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
cd /Users/bytedance/Desktop/TinyLLMForge
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py \
  tools/test_autoregressive_draft_cuda_graph_config.py \
  -k 'command_timeline'
```

Expected: collection fails because
`tinyvllm/engine/model_runner_command_timeline.py` does not exist and Config
does not expose the new fields.

- [ ] **Step 3: Implement immutable identity, clock identity, recorder, and arithmetic**

Create `tinyvllm/engine/model_runner_command_timeline.py` with:

```python
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import copy
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time


SCHEMA_VERSION = 1
_ACTIVE_TRACE = ContextVar(
    "tinyvllm_model_runner_command_trace",
    default=None,
)


def _sha256(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


@dataclass(frozen=True)
class CommandClockIdentity:
    boot_id: str
    implementation: str
    resolution_s: float
    monotonic: bool
    adjustable: bool

    def __post_init__(self):
        if not isinstance(self.boot_id, str) or not self.boot_id:
            raise ValueError("boot_id must be non-empty")
        if not isinstance(self.implementation, str) or not self.implementation:
            raise ValueError("clock implementation must be non-empty")
        if (
            not isinstance(self.resolution_s, (int, float))
            or isinstance(self.resolution_s, bool)
            or not math.isfinite(float(self.resolution_s))
            or float(self.resolution_s) <= 0
        ):
            raise ValueError("clock resolution must be positive and finite")
        if self.monotonic is not True or not isinstance(self.adjustable, bool):
            raise ValueError("clock must be monotonic with boolean adjustability")


def read_command_clock_identity():
    info = time.get_clock_info("monotonic")
    return CommandClockIdentity(
        boot_id=Path(
            "/proc/sys/kernel/random/boot_id"
        ).read_text(encoding="utf-8").strip(),
        implementation=str(info.implementation),
        resolution_s=float(info.resolution),
        monotonic=bool(info.monotonic),
        adjustable=bool(info.adjustable),
    )


@dataclass(frozen=True)
class CommandTraceIdentity:
    command_id: int
    method_name: str
    requires_ack: bool
    engine_step_id: int | None
    repeat_index: int | None
    request_set_sha256: str | None
    batch_kind: str | None
    speculative_selected_sequence_ids_sha256: str | None
    dispatch_started_monotonic_ns: int
    dispatch_published_monotonic_ns: int

    def __post_init__(self):
        if (
            isinstance(self.command_id, bool)
            or not isinstance(self.command_id, int)
            or self.command_id < 0
        ):
            raise ValueError("command_id must be a non-negative integer")
        if not isinstance(self.method_name, str) or not self.method_name:
            raise ValueError("method_name must be non-empty")
        if not isinstance(self.requires_ack, bool):
            raise ValueError("requires_ack must be a bool")
        for name in (
            "request_set_sha256",
            "speculative_selected_sequence_ids_sha256",
        ):
            value = getattr(self, name)
            if value is not None:
                _sha256(value, name)
        if (
            self.dispatch_started_monotonic_ns < 0
            or self.dispatch_published_monotonic_ns
            < self.dispatch_started_monotonic_ns
        ):
            raise ValueError("dispatch timestamps are invalid")


def active_model_runner_command_trace():
    return _ACTIVE_TRACE.get()


@contextmanager
def command_trace_scope(identity):
    token = _ACTIVE_TRACE.set(identity)
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)
```

Implement `ModelRunnerCommandTimelineRecorder` as a bounded recorder that:

- has `disabled(rank)`;
- validates rank and `max_rows`;
- records one mutable internal row per command;
- rejects duplicate or non-increasing command IDs;
- exposes explicit methods for dispatch, worker receive, method start/end,
  ack-send start/end, and rank-zero ack-wait start/end;
- records `status` and bounded `error_type`;
- returns deep-copied canonical snapshots;
- increments `dropped_rows` rather than growing past `max_rows`; and
- rejects snapshots with an active unfinished row.

Implement `compute_command_decomposition(rows)` exactly from the approved
spec, sorting by `(rank, command_id)` and rejecting negative duration,
`cuda_ns > worker_method_wall_ns`, missing predecessors, or non-monotonic
rank-local command order.

Add the two Config fields and validation in `Config.__post_init__`.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run the Step 2 command.

Expected: all selected tests pass.

- [ ] **Step 5: Commit and push the core**

```bash
git add -- \
  tinyvllm/engine/model_runner_command_timeline.py \
  tinyvllm/config.py \
  tools/test_model_runner_command_timeline.py \
  tools/test_autoregressive_draft_cuda_graph_config.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): add command timeline core" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

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

### Task 5: Canonical Exact-Identity Diagnostic

**Files:**
- Create: `tools/autoregressive_draft_command_timeline_diagnostic.py`
- Create: `tools/test_autoregressive_draft_command_timeline_diagnostic.py`

**Interfaces:**
- `EpochIdentity`.
- `expected_epoch_identities()`.
- `validate_epoch_worker(worker, identity)`.
- `join_repeat_timeline(worker, repeat_index)`.
- `compute_sync_debt(repeat)`.
- `build_epoch_admission(identity, raw_inputs)`.
- `compute_paired_boundary_effects(epochs)`.
- `classify_boundary(bundle_admission, effects)`.
- `build_command_timeline_artifact(*, metadata, epoch_raw_inputs, input_files, source_files)`.
- `validate_command_timeline_artifact(artifact)`.

- [ ] **Step 1: Write schedule, exact identity, and parity tests**

Use fixed constants:

```python
SCHEMA_VERSION = 1
BLOCK_SCHEDULE = (
    ("eager", "graph"),
    ("graph", "eager"),
    ("graph", "eager"),
    ("eager", "graph"),
)
MEASURED_RUNS_PER_EPOCH = 5
MEASURED_RUNS_TOTAL = 40
BATCH_SIZE = 4
MAX_PROPOSAL_TOKENS = 4
PROMPT_TOKENS = 256
OUTPUT_TOKENS = 16
TEMPERATURE = 0.0
ROBUST_DISPERSION_LIMIT = 0.10
HALF_DRIFT_LIMIT = 0.15
ABSOLUTE_CONSERVATION_NS = 2_000_000
RELATIVE_CONSERVATION_LIMIT = 0.01
BOUNDARY_EXPLANATION_THRESHOLD = 0.60
BOUNDARY_BLOCK_COUNT = 3
UNEXPLAINED_E2E_LIMIT = 0.10
```

Tests must independently reject:

- wrong schedule label or position;
- non-learned policy;
- TP not equal to four;
- batch not equal to four;
- proposal limit not equal to four;
- prompt length or prompt digest mismatch;
- output length or request order mismatch;
- nonzero temperature;
- non-direct Proposal-KV allocator;
- Proposal-KV offload enabled;
- graph/eager source, checkpoint, tokenizer, or GPU UUID mismatch;
- graph capture/replay/resource drift;
- eager capture, replay, or ready entry;
- target token, proposal row, accepted-prefix, accepted-token, transaction
  digest, acceptance, or active-transaction mismatch; and
- padded or oversized logical proposal rows.

- [ ] **Step 2: Write timeline join, debt, and conservation tests**

Create a valid fixture with four rank command rows, CUDA rows, engine spans,
and request timing. Assert:

```python
repeat = diagnostic.join_repeat_timeline(worker, 0)
assert repeat["critical_rank"] == 3
assert repeat["components_ns"] == {
    "worker_queue_debt": 60_000_000,
    "worker_cuda_execution": 400_000_000,
    "ack_wait": 20_000_000,
    "scheduler_postprocess": 100_000_000,
}
assert repeat["conservation"]["passed"] is True
```

Reject:

- mismatched boot IDs or clock metadata;
- duplicate, missing, or reordered command IDs;
- unknown command references in CUDA or engine rows;
- negative queue/ack/CUDA/phase duration;
- CUDA greater than method wall time;
- overlap greater than containing intervals;
- a non-ack command with ack timestamps;
- an ack command with missing ack wait;
- unexplained step residual above tolerance; and
- timeline rows outside the repeat campaign interval.

- [ ] **Step 3: Write stationarity and classification boundary tests**

Test exact threshold inclusivity:

```python
assert stationarity_for_values(
    [100.0, 100.0, 100.0, 100.0, 110.0]
)["robust_dispersion_passed"] is True
```

Construct four blocks where queue debt explains `60%` in exactly three blocks,
same-sign count is three, and residual is exactly `10%`; expect:

```text
BOUNDARY_LOCALIZED
localized_boundary=worker_queue_debt
```

Move each threshold one unit beyond the allowed boundary and expect:

```text
PAIRED_PROTOCOL_UNSTABLE
stable_but_unlocalized=true
```

Also cover precedence:

```text
identity/parity failure -> INVALID_IDENTITY_OR_CORRECTNESS
timeline/conservation failure -> TIMELINE_INCOMPLETE_OR_NONCONSERVING
stationarity failure -> PAIRED_PROTOCOL_UNSTABLE
```

- [ ] **Step 4: Run tests and confirm RED**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  -k 'schedule or identity or parity or timeline or debt or conservation or stationarity or classification'
```

Expected: collection fails because the diagnostic module does not exist.

- [ ] **Step 5: Implement canonical diagnostic**

Reuse only pure validated helpers from:

- `autoregressive_draft_cuda_graph_contract.py` for exact graph counter and
  bounded logical-row rules;
- `autoregressive_draft_paired_stability_diagnostic.py` for stationarity and
  balanced block effect primitives;
- `autoregressive_draft_instability_telemetry.py` and
  `autoregressive_draft_host_semantic_diagnostic.py` for telemetry alignment.

Do not import or mutate completed schema-v2 payload state.

The canonical artifact must include these exact top-level keys:

```python
TOP_LEVEL_KEYS = (
    "schema_version",
    "schedule",
    "configuration",
    "provenance",
    "raw_input_files",
    "source_files",
    "epochs",
    "blocks",
    "admission",
    "effects",
    "classification",
    "localized_boundary",
    "stable_but_unlocalized",
    "runtime_optimization_authorized",
    "performance_improvement_established",
    "phase_1_complete",
    "promotion_ready",
)
```

`validate_command_timeline_artifact` recomputes all derived fields from the
embedded normalized epoch rows and rejects any mismatch.

- [ ] **Step 6: Run diagnostic tests and confirm GREEN**

Run the Step 4 command without `-k`.

Expected: all tests pass.

- [ ] **Step 7: Commit and push the diagnostic**

```bash
git add -- \
  tools/autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): classify command timeline boundaries" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 6: Independent Verifier and Manifest

**Files:**
- Create: `tools/verify_autoregressive_draft_command_timeline_diagnostic.py`
- Modify: `tools/test_autoregressive_draft_command_timeline_diagnostic.py`

**Interfaces:**
- `verify_raw_input_bindings(artifact, artifact_root)`.
- `verify_source_bindings(artifact, source_root)`.
- `verify_manifest(manifest_path, artifact_root)`.
- `verify_command_timeline_diagnostic(*, artifact_path, source_root, manifest_path=None)`.
- CLI supports `--artifact`, `--source-root`, `--manifest`, `--receipt`, and
  `--verification-location`.

- [ ] **Step 1: Write source/raw/manifest tamper tests**

Build a temporary source-bound bundle containing:

```text
command-timeline.json
result.json
workers/block-*/{eager,graph}.json
telemetry/block-*/*
source/*
source.patch
source_manifest.json
manifest.sha256
```

Test:

- successful full recomputation;
- changed raw worker byte;
- changed timeline row;
- changed source file;
- unsafe absolute or `..` path;
- duplicate manifest path;
- missing authoritative file;
- extra unlisted authoritative file;
- mismatched canonical result summary;
- changed classification;
- changed false claim-boundary field; and
- byte-equivalent remote/local semantic receipts after removing only
  `verified_at_utc`, `verification_location`, and `artifact_path`.

- [ ] **Step 2: Run verifier tests and confirm RED**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  -k 'verifier or source or raw_input or manifest or tamper or receipt'
```

Expected: missing verifier module and CLI.

- [ ] **Step 3: Implement full recomputation**

Follow the paired-stability verifier's safe path and detached-attestation
pattern, but use these detached files:

```python
DETACHED_ATTESTATION_PATHS = {
    "manifest.sha256",
    "verify.command-timeline.remote.json",
    "verify.command-timeline.remote.log",
    "verify.command-timeline.local.json",
    "verify.command-timeline.local.log",
}
```

The verifier:

1. validates canonical artifact structure;
2. verifies every raw input hash;
3. verifies every frozen source hash;
4. reloads all bound workers and telemetry;
5. calls `build_command_timeline_artifact` from raw inputs;
6. requires exact canonical equality;
7. verifies the manifest inventory when supplied; and
8. writes a receipt with classification, localized boundary, source/input
   counts and inventory digests, manifest digest/count, and verifier source
   digest.

It must force all claim-boundary booleans false.

- [ ] **Step 4: Run verifier tests and confirm GREEN**

Run the Step 2 command without `-k`.

Expected: all tests pass.

- [ ] **Step 5: Commit and push verifier**

```bash
git add -- \
  tools/verify_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): verify command timeline evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 7: Safe Source-Bound Remote Runner Contract

**Files:**
- Create: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Modify: `tools/autoregressive_draft_cuda_graph_gate.py`
- Modify: `tools/test_autoregressive_draft_command_timeline_diagnostic.py`
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- `build_epoch_schedule()`.
- `build_worker_command(*, python, worker_script, target_model, draft_model, mode, output_path)`.
- `build_source_archive(repo_root, archive_path)`.
- `classify_local_kerberos_payload(payload, *, now, minimum_lifetime_seconds=5400)`.
- `classify_gpu_preflight(rows)`.
- `run_preflight(*, run_tag, command_runner=subprocess.run)`.
- `run_bundle(*, run_tag, command_runner=subprocess.run)`.
- CLI subcommands `preflight`, `execute`, and `verify-local`.

- [ ] **Step 1: Write runner command, source closure, and safety tests**

Assert the exact schedule:

```python
[
    ("block-0", "eager", "first"),
    ("block-0", "graph", "second"),
    ("block-1", "graph", "first"),
    ("block-1", "eager", "second"),
    ("block-2", "graph", "first"),
    ("block-2", "eager", "second"),
    ("block-3", "eager", "first"),
    ("block-3", "graph", "second"),
]
```

Assert every worker command contains:

```text
--policy learned
--batch-size 4
--warmup-runs 1
--measured-runs 5
--command-timeline
--cuda-graph-mode eager|graph
```

Source closure must include:

```text
tinyvllm/
tools/autoregressive_draft_performance_worker.py
tools/autoregressive_draft_performance_gate.py
tools/autoregressive_draft_cuda_graph_contract.py
tools/autoregressive_draft_cuda_graph_gate.py
tools/autoregressive_draft_command_timeline_diagnostic.py
tools/verify_autoregressive_draft_command_timeline_diagnostic.py
tools/autoregressive_draft_paired_stability_diagnostic.py
tools/autoregressive_draft_instability_telemetry.py
tools/autoregressive_draft_host_semantic_diagnostic.py
tools/autoregressive_draft_host_sampler.py
tools/run_autoregressive_draft_command_timeline_remote.py
```

Reject symlinks and archive members outside `source/`.

Safety tests must reject or forbid:

```text
pkill
killall
fuser -k
git clean
git reset
rm -rf
sudo
```

Runner cleanup may signal only PIDs created and recorded by the runner in the
current process.

- [ ] **Step 2: Write immutable tag, Kerberos, GPU, and manifest tests**

Reuse CUDA Graph runner preflight helpers and test:

- tag must match `[A-Za-z0-9_-]+`;
- local and remote destination must not already exist;
- Kerberos principal/TGT and at least `5400 s` remaining lifetime;
- exactly four selected GPUs, each under idle memory/utilization thresholds;
- no unrelated selected-GPU compute process;
- GPU UUID set retained before and after every epoch;
- partial evidence transferred on owned-worker failure;
- no next epoch after a failed epoch;
- pre-manifest verifier must pass before manifest creation;
- remote verifier must pass after manifest;
- local verifier must pass after transfer; and
- normalized remote/local receipts must be identical.

- [ ] **Step 3: Run runner tests and confirm RED**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'runner or schedule or command or archive or kerberos or gpu or safety or immutable or manifest'
```

Expected: missing runner and five-repeat command support.

- [ ] **Step 4: Implement the local runner contract**

Base connection, Kerberos, GPU parser, archive safety, and source hash logic on
`run_autoregressive_draft_cuda_graph_gate_remote.py`.

For each epoch:

1. capture before GPU/process inventory;
2. start runner-owned GPU and host samplers;
3. execute one isolated worker in the foreground;
4. stop and reap only recorded sampler PIDs;
5. capture after GPU/process inventory;
6. preserve worker stdout/stderr and exit status;
7. stop the bundle on any nonzero worker or invariant failure; and
8. never delete partial evidence.

After all epochs:

1. assemble `command-timeline.json`;
2. write `result.json` containing canonical artifact SHA-256,
   classification, localized boundary, and false claim flags;
3. run pre-manifest verification;
4. generate `manifest.sha256` over every authoritative file except detached
   attestations;
5. run remote verification;
6. transfer to a new local artifact directory;
7. run local verification against current source; and
8. compare normalized receipts.

Do not execute the runner in this task.

- [ ] **Step 5: Run all runner contract tests and syntax checks**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m py_compile \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/autoregressive_draft_command_timeline_diagnostic.py \
  tools/verify_autoregressive_draft_command_timeline_diagnostic.py
```

Expected: all tests and compilation pass.

- [ ] **Step 6: Commit and push runner contract**

```bash
git add -- \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): add command timeline gate runner" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 8: Expanded Local Verification and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Create: `docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md`

**Interfaces:**
- Produces a prompt-to-artifact checklist covering every approved spec
  requirement.
- Records exact test commands, counts, environment limitations, commit IDs,
  and remote authorization boundary.

- [ ] **Step 1: Run dependency-light focused suite**

```bash
cd /Users/bytedance/Desktop/TinyLLMForge
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m pytest -q \
  tools/test_model_runner_command_timeline.py \
  tools/test_engine_step_timeline.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py
```

Expected: all tests pass.

- [ ] **Step 2: Run Torch-backed expanded suite in a pinned uv environment**

The system Python has pytest 8.4.2 but no Torch. Use:

```bash
UV_CACHE_DIR=/tmp/tinyllmforge-command-timeline-uv-cache \
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
uv run \
  --with pytest==8.4.2 \
  --with torch==2.7.1 \
  --with transformers==4.57.6 \
  python -m pytest -q \
    tools/test_model_runner_command_timeline.py \
    tools/test_engine_step_timeline.py \
    tools/test_model_runner_command_ack.py \
    tools/test_model_runner_live_ack_wiring.py \
    tools/test_decode_internal_profiler.py \
    tools/test_decode_internal_profile_wiring.py \
    tools/test_engine_speculative_execution.py \
    tools/test_engine_speculative_runtime.py \
    tools/test_autoregressive_draft_model_runner_integration.py \
    tools/test_autoregressive_draft_performance_gate.py \
    tools/test_autoregressive_draft_cuda_graph_gate.py \
    tools/test_autoregressive_draft_paired_stability_diagnostic.py \
    tools/test_autoregressive_draft_instability_telemetry.py \
    tools/test_autoregressive_draft_command_timeline_diagnostic.py
```

Expected: all tests pass. Record optional dependency warnings separately; do
not report missing dependencies as test success or failure.

- [ ] **Step 3: Run source, syntax, and forbidden-pattern checks**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-command-timeline-pycache \
python3 -m compileall -q \
  tinyvllm/engine/model_runner_command_timeline.py \
  tinyvllm/engine/engine_step_timeline.py \
  tinyvllm/engine/model_runner_command_ack.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/decode_internal_profiler.py \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_command_timeline_diagnostic.py \
  tools/verify_autoregressive_draft_command_timeline_diagnostic.py \
  tools/run_autoregressive_draft_command_timeline_remote.py
git diff --check
rg -n \
  'torch\\.cuda\\.synchronize|requires_ack=True|pkill|killall|fuser -k|rm -rf' \
  tinyvllm/engine/model_runner_command_timeline.py \
  tinyvllm/engine/engine_step_timeline.py \
  tools/run_autoregressive_draft_command_timeline_remote.py
```

Interpret matches manually:

- no runtime profiler module may contain `torch.cuda.synchronize`;
- the runner may contain no destructive process-wide command;
- acknowledged snapshot/configuration calls are allowed outside the measured
  request path;
- existing acknowledged lifecycle semantics may not be broadened.

- [ ] **Step 4: Build the prompt-to-artifact local audit**

Create the audit with a table containing:

```text
requirement
source implementation
focused test
expanded test
artifact/verifier coverage
status
remaining remote evidence
```

It must explicitly cover:

- paired-stability reuse;
- host/GPU telemetry reuse;
- remote/local verifier and manifest;
- exact graph/eager identity;
- command timeline;
- queue debt, CUDA, ack, scheduler/postprocess decomposition;
- no new request-path sync/fence;
- parity and Proposal-KV transaction semantics;
- timing conservation;
- stationarity/localization thresholds;
- immutable `r3`;
- source-bound closure;
- runner process ownership;
- remote execution not run; and
- runtime optimization not authorized.

Append an EOF handoff reconciliation with exact local test counts and:

```text
COMMAND_TIMELINE_LOCAL_IMPLEMENTATION=ESTABLISHED
COMMAND_TIMELINE_REMOTE_BUNDLE=NOT_RUN
BOUNDARY_LOCALIZED=NOT_ESTABLISHED
RUNTIME_OPTIMIZATION=NOT_AUTHORIZED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

- [ ] **Step 5: Commit and push local completion evidence**

```bash
git add -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "docs(runtime): audit command timeline implementation" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git status --short --untracked-files=no
```

Expected: local and origin heads match; tracked worktree is clean.

---

## STOP Gate: Separate Remote Authorization Required

Do not execute the remote runner until the user explicitly authorizes one
source-bound command-timeline bundle after reviewing the local audit.

The authorization request must name:

```text
runner:
  tools/run_autoregressive_draft_command_timeline_remote.py

schedule:
  eager_graph, graph_eager, graph_eager, eager_graph

epochs:
  8 isolated processes

per epoch:
  1 warmup + 5 measured batches

scope:
  TP4/B4/Q4, prompt 256, output 16, greedy, direct Proposal-KV,
  Proposal-KV offload disabled

remote claim:
  boundary localization only
```

After authorization, use a never-before-used tag:

```text
20260818-command-timeline-tp4-b4-q4-r1
```

First run a read-only preflight:

```bash
python3 tools/run_autoregressive_draft_command_timeline_remote.py \
  preflight \
  --run-tag 20260818-command-timeline-tp4-b4-q4-r1
```

Proceed to `execute` only if:

- local and remote tag destinations are absent;
- Kerberos has at least `5400 s` remaining;
- four GPUs are clean and the selected UUID inventory is stable;
- no unrelated selected-GPU compute process exists; and
- the source commit equals the pushed branch head.

The exact execute command is:

```bash
python3 tools/run_autoregressive_draft_command_timeline_remote.py \
  execute \
  --run-tag 20260818-command-timeline-tp4-b4-q4-r1
```

If preflight or execution fails, preserve partial evidence and do not reuse,
resume, merge, or overwrite the tag.

## Post-Run Completion Audit

Before claiming boundary localization:

1. verify all eight epochs and forty measured batches exist;
2. verify exact target/proposal/accepted-token and transaction parity;
3. verify every graph rank has one warmup capture and measured replay growth;
4. verify eager has zero graph state;
5. verify all command/CUDA/step/telemetry rows are complete and conserving;
6. verify all eight epochs pass stationarity;
7. verify remote and local receipts agree;
8. run `shasum -a 256 -c manifest.sha256`;
9. map every approved spec requirement to retained evidence; and
10. classify only as one of the four approved fail-closed classifications.

If and only if the canonical result is `BOUNDARY_LOCALIZED`, write a separate
boundary-specific optimization design and obtain approval before changing
runtime scheduling or synchronization.
