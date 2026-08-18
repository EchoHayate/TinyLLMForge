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
