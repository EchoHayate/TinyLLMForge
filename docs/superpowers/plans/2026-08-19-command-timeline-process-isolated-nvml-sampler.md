# Command-Timeline Process-Isolated NVML Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run one persistent NVML query process per frozen GPU so process-local NVML serialization cannot create a two-second four-GPU snapshot gap.

**Architecture:** Replace the four-thread sampler with one parent and four forked child processes connected by pipes. The parent broadcasts one sample command, receives one complete row from each child, and only then assigns shared timestamps and emits the group.

**Tech Stack:** Python 3.9/3.11, `multiprocessing` fork context, `ctypes`, NVML C ABI, JSON Lines, pytest 8.4.2, SSH ControlMaster, Sitian source transaction.

## Global Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not instrument or synchronize the measured request path.
- Keep all eight NVML fields, their order, and all eight query durations.
- Emit no row until all four child results are complete.
- Keep shared parent timestamps, 200 ms cadence, and strict coverage unchanged.
- Do not borrow, cache, duplicate, or synthesize telemetry.
- Keep four-clean-GPU admission and process ownership unchanged.
- Keep every remote output beneath `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Set all remote scratch and cache variables beneath that root.
- Do not signal or adopt unrelated processes.
- Preserve r16 and use fresh tag `20260818-command-timeline-tp4-b4-q4-r17`.
- Stage exact paths only.
- Commit with disabled hooks and exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Add the Failing Four-Process Contract

**Files:**
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `_build_gpu_sampler_script() -> str`
- Produces: raw-row proof `sampler_process_pid: int`

- [ ] **Step 1: Make fake shutdown evidence process-safe**

Change the fake prelude so `shutdown_path` is a directory. Import `os`, create
the directory, and atomically create one marker per process:

```python
shutdown_root = Path(SHUTDOWN_PATH)
shutdown_root.mkdir(parents=True, exist_ok=True)
(shutdown_root / str(os.getpid())).write_text("shutdown\n")
```

Change query-order evidence to a directory containing one
`<pid>.json` file per NVML process. Each file records the one device handled by
that process and its observed query sequence.

- [ ] **Step 2: Replace the thread barrier test**

Remove the `threading.Barrier` probe. Add:

```python
def test_command_timeline_direct_nvml_sampler_isolates_each_gpu_process(...):
    ...
    rows = [json.loads(sampler.stdout.readline()) for _ in range(4)]
    assert len({row["sampler_process_pid"] for row in rows}) == 4
    assert all(
        row["sampler_process_pid"] != sampler.pid
        for row in rows
    )
```

Also require the emitted GPU index and UUID order to remain `[2, 3, 4, 6]`.

- [ ] **Step 3: Update order and shutdown assertions**

After one snapshot and SIGTERM, require four query-order files. Each file must
contain one device whose sequence starts with:

```python
[
    "performance_state",
    "sm_clock",
    "memory_clock",
    "power_usage",
    "temperature",
    "utilization_rates",
    "memory_info",
    "clock_throttle_reasons",
]
```

Require four shutdown marker files after clean SIGTERM and after the injected
power-query failure.

- [ ] **Step 4: Run RED**

Run the direct-NVML focused selector with a fresh local basetemp.

Expected: FAIL because threaded rows have no `sampler_process_pid` and fake
shutdown evidence comes from one process.

---

### Task 2: Implement the Parent/Child Protocol

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Produces: `configure_nvml()`, `sample_device(nvml, device)`, `device_worker(connection, device)`, and parent-owned complete groups

- [ ] **Step 1: Replace the thread import**

Use:

```python
import multiprocessing
import os
import traceback
```

Remove `ThreadPoolExecutor`.

- [ ] **Step 2: Configure NVML per child**

Move `ctypes.CDLL` and every `.argtypes`/`.restype` assignment into:

```python
def configure_nvml():
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    # Assign every existing signature.
    return nvml
```

Change `check` and `timed_query` to receive `nvml` explicitly.

- [ ] **Step 3: Keep one sequential query chain**

Change the current helper to:

```python
def sample_device(nvml, device):
    index, gpu_uuid, handle = device
    ...
    return {
        ...,
        "sampler_process_pid": os.getpid(),
    }
```

Keep all eight calls and their order unchanged.

- [ ] **Step 4: Add the child protocol**

Implement:

```python
def device_worker(connection, device):
    initialized = False
    try:
        nvml = configure_nvml()
        check(nvml, "nvmlInit_v2", nvml.nvmlInit_v2())
        initialized = True
        index, expected_uuid = device
        handle = ctypes.c_void_p()
        check(
            nvml,
            "nvmlDeviceGetHandleByIndex_v2",
            nvml.nvmlDeviceGetHandleByIndex_v2(
                index, ctypes.byref(handle)
            ),
        )
        # Resolve and validate the UUID.
        connection.send({
            "kind": "ready",
            "gpu_index": index,
            "gpu_uuid": expected_uuid,
            "sampler_process_pid": os.getpid(),
        })
        while True:
            command = connection.recv()
            if command == "stop":
                break
            if command != "sample":
                raise ValueError("GPU sampler command is invalid")
            connection.send({
                "kind": "row",
                "row": sample_device(
                    nvml, (index, expected_uuid, handle)
                ),
            })
    except BaseException:
        try:
            connection.send({
                "kind": "error",
                "gpu_index": device[0],
                "detail": traceback.format_exc(),
            })
        except BaseException:
            pass
    finally:
        if initialized:
            shutdown_result = nvml.nvmlShutdown()
            if shutdown_result != NVML_SUCCESS:
                try:
                    connection.send({
                        "kind": "error",
                        "gpu_index": device[0],
                        "detail": "nvmlShutdown failed",
                    })
                except BaseException:
                    pass
        connection.close()
```

- [ ] **Step 5: Start and validate four children**

Use `multiprocessing.get_context("fork")`. Create one pipe and one process per
frozen device. Start all children before receiving ready messages. Validate:

```python
ready["kind"] == "ready"
ready["gpu_index"] == expected_index
ready["gpu_uuid"] == expected_uuid
```

Require four distinct positive child PIDs.

- [ ] **Step 6: Broadcast then collect**

For every sample:

```python
for connection in connections:
    connection.send("sample")
snapshot = []
for connection in connections:
    message = connection.recv()
    if message.get("kind") != "row":
        raise RuntimeError(format_child_error(message))
    snapshot.append(message["row"])
```

Check exact index/UUID order before timestamps and printing.

- [ ] **Step 7: Stop and join owned children**

In `finally`, send `stop` only to children that are alive and whose pipe has
not failed. Close parent pipe endpoints, join every child, and reject an
unexpected nonzero exit unless a signal-requested clean exit is in progress.
Do not terminate or inspect unrelated processes.

- [ ] **Step 8: Run GREEN and full local contract**

Run:

1. the new distinct-process/order/failure/SIGTERM selector;
2. the complete direct-NVML and telemetry selector;
3. all three command-timeline test files.

Expected: all pass, with the full local count increasing from 307.

- [ ] **Step 9: Review, commit, and push**

Inspect only the two changed files, run `git diff --check`, stage exactly those
paths, commit:

```text
fix(command-timeline): isolate NVML samplers by process
```

and push `feat/kv-sparse-attention`.

---

### Task 3: Remote Verification and Automatic r17

**Files:**
- Remote sync: the same two implementation files

**Interfaces:**
- Produces: synced source receipt, remote GREEN, immutable r17

- [ ] **Step 1: Atomically sync and verify**

Use the task-owned ControlMaster and source transaction. Require local,
origin, and remote source heads to match and detached hashes to equal local
SHA-256.

- [ ] **Step 2: Run remote focused and full contracts**

Use the task-local test Python and unique basetemps beneath the task root.
Preserve exact counts.

- [ ] **Step 3: Start the Mac-owned controller**

Confirm r17 primary and controller destinations are absent. Reuse the existing
Kerberos cache without refreshing. Repeatedly invoke official preflight and
immediately `exec` official r17 when `READY`.

- [ ] **Step 4: Evaluate r17**

For every reached measured repeat, require:

- at least one strict in-interval complete snapshot;
- exactly four expected UUIDs per emitted timestamp;
- exactly four distinct `sampler_process_pid` values;
- complete eight-key query durations;
- empty sampler stderr.

Preserve any failure and continue only under r18 or later.

---

### Task 4: Finish Eight Epochs and Reconcile

**Files:**
- Modify: this plan
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-concurrent-nvml-sampler.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-nvml-query-latency.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-direct-nvml-sampler.md`
- Modify: `docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Complete fresh-tag diagnosis until all epochs finish**

Use one evidence-backed TDD change per newly exposed failure. Never reuse a
tag or weaken a gate.

- [ ] **Step 2: Run both verifiers and manifest validation**

Require matching normalized receipts and all manifest hashes.

- [ ] **Step 3: Reconcile chronological evidence**

Record r15, r16, and every later tag with exact failure, test, commit, sync,
and claim boundaries.

- [ ] **Step 4: Final verification, commit, and push**

Rerun the complete local contract, `git diff --check`, local/origin parity,
and exact documentation staging before the final push.

## 2026-08-20 Final Reconciliation

Process isolation was necessary but not sufficient: r17 completed only two
epochs before block-1 graph telemetry coverage failed. The field-isolated
32-process `(GPU, query)` generation matrix replaced this architecture while
retaining process identity, positive per-query duration, empty-stderr, frozen
GPU-set, and strict measured-interval coverage requirements.

The immutable continuation was:

- r18: all eight before/after inventories completed, but no canonical result
  or manifest was produced;
- r19: an exiting task-owned process was observed as unowned;
- r20: all eight epochs completed; current-source read-only reconciliation
  classifies the retained data as `PAIRED_PROTOCOL_UNSTABLE`;
- r21: failed before epoch 0 during transport, with empty `status/`;
- r22: failed during prepare transport and created neither destination;
- r23: terminal complete authority.

Tag `20260818-command-timeline-tp4-b4-q4-r23` is bound to
`596e724ea87966b2ab3b47cccda08c106f9084bb` and completed eight epochs,
40 measured repeats, dual verification, and exact manifests. It establishes
the field-isolated telemetry architecture, not a runtime optimization:

```text
RUNTIME_CLASSIFICATION=PAIRED_PROTOCOL_UNSTABLE
BOUNDARY_LOCALIZED=NOT_ESTABLISHED
RUNTIME_OPTIMIZATION=NOT_AUTHORIZED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
```
