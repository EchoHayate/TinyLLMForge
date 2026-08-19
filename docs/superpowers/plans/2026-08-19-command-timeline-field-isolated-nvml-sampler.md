# Command-Timeline Field-Isolated NVML Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace each per-GPU sequential NVML query chain with a strict 32-process `(GPU, query)` generation matrix, prove the architecture under the unchanged real graph workload, and automatically launch fresh official tag r18 only after every diagnostic gate passes.

**Architecture:** The generated dependency-free sampler starts one persistent forked child for each frozen GPU and query identity. A parent broadcasts one generation to all 32 children, validates the complete Cartesian result matrix, reconstructs the existing four JSON rows, and emits only after all real values and durations are present. A single-epoch diagnostic reuses the official preparation, inventory, worker, telemetry attachment, and process-ownership paths without changing the measured worker.

**Tech Stack:** Python 3.9/3.11, `multiprocessing` fork context, `ctypes`, NVML C ABI, JSON Lines, pytest 8.4.2, SSH ControlMaster, Sitian source transaction.

## Global Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve r17 and every earlier tag as immutable evidence.
- Use fresh tag `20260818-command-timeline-tp4-b4-q4-r18` only after the diagnostic gate passes.
- Do not modify the measured worker, workload, warmup count, measured repeat count, repeat duration, schedule, or invariant verifier.
- Do not add synchronization or instrumentation to the measured request path.
- Preserve all eight real NVML values and a real duration for each query.
- Emit no row until all 32 results for one generation are complete.
- Keep at most one generation in flight and preserve the 200 ms target cadence.
- Preserve strict in-interval coverage, no boundary borrowing, and no synthetic duplication.
- Preserve four-clean-GPU admission and task-owned process handling.
- Do not signal, pause, adopt, or interfere with unrelated processes.
- Keep every remote artifact, cache, temporary file, test basetemp, and receipt beneath `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Set `TMPDIR`, `TMP`, `TEMP`, `PYTHONPYCACHEPREFIX`, and `XDG_CACHE_HOME` beneath that task root for every remote payload.
- Do not write task output to local or remote `/`, `/tmp`, or `/private/tmp`.
- Do not manually refresh Kerberos.
- Stage exact paths only; never use `git add -A`.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Add the Failing 32-Process Matrix Contract

**Files:**
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `_build_gpu_sampler_script() -> str`
- Produces: black-box proof of 32 distinct `(GPU, query, PID)` children and complete generation output

- [ ] **Step 1: Make fake evidence query-process specific**

Change `_fake_nvml_prelude` so each fake NVML instance records the one query
executed by its process. Add a process-safe evidence directory whose
`<pid>.json` payload has:

```python
{
    "pid": os.getpid(),
    "gpu_indices": sorted(self.query_order),
    "query_names": sorted({
        query
        for queries in self.query_order.values()
        for query in queries
    }),
    "query_intervals": self.query_intervals,
}
```

Keep one shutdown marker per PID. Add optional
`query_delays: dict[str, float]` and sleep only inside the named fake query
callback. Record:

```python
started_ns = time.monotonic_ns()
time.sleep(query_delays.get(query_name, 0.0))
finished_ns = time.monotonic_ns()
self.query_intervals.append({
    "query_name": query_name,
    "started_ns": started_ns,
    "finished_ns": finished_ns,
})
```

- [ ] **Step 2: Replace the four-process assertion**

Replace
`test_command_timeline_direct_nvml_sampler_isolates_each_gpu_process` with:

```python
def test_command_timeline_direct_nvml_sampler_isolates_each_gpu_query(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_field_process_isolation_test"
    )
    evidence_path = tmp_path / "query-processes"
    sampler = subprocess.Popen(
        _direct_nvml_sampler_command(
            runner,
            shutdown_path=tmp_path / "shutdown-markers",
            order_path=evidence_path,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert sampler.stdout is not None
    rows = [json.loads(sampler.stdout.readline()) for _ in range(4)]
    sampler.terminate()
    _stdout, stderr = sampler.communicate(timeout=10)

    assert sampler.returncode == 0, stderr
    assert [row["gpu_index"] for row in rows] == [2, 3, 4, 6]
    query_processes = {
        (gpu_index, query_name): pid
        for row in rows
        for query_name, pid in row["sampler_process_pid"].items()
        for gpu_index in [row["gpu_index"]]
    }
    assert len(query_processes) == 32
    assert len(set(query_processes.values())) == 32
    assert sampler.pid not in set(query_processes.values())
```

The output schema intentionally changes `sampler_process_pid` from one integer
per GPU row to a mapping of all eight query names to their real child PID.

- [ ] **Step 3: Add the concurrent-generation timing contract**

Add:

```python
def test_command_timeline_field_queries_run_in_one_concurrent_generation(
    tmp_path,
):
    runner = _load_command_timeline_runner(
        "command_timeline_runner_field_concurrency_test"
    )
    evidence_path = tmp_path / "query-processes"
    sampler = subprocess.Popen(
        _direct_nvml_sampler_command(
            runner,
            shutdown_path=tmp_path / "shutdown-markers",
            order_path=evidence_path,
            query_delays={
                "performance_state": 0.20,
                "sm_clock": 0.20,
                "memory_clock": 0.20,
                "power_usage": 0.20,
                "temperature": 0.20,
                "utilization_rates": 0.20,
                "memory_info": 0.20,
                "clock_throttle_reasons": 0.20,
            },
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert sampler.stdout is not None
    rows = [json.loads(sampler.stdout.readline()) for _ in range(4)]
    sampler.terminate()
    _stdout, stderr = sampler.communicate(timeout=10)

    assert sampler.returncode == 0, stderr
    assert len(rows) == 4
    intervals = [
        interval
        for path in evidence_path.glob("*.json")
        for interval in json.loads(path.read_text())["query_intervals"]
    ]
    assert len(intervals) == 32
    assert max(row["started_ns"] for row in intervals) < min(
        row["finished_ns"] for row in intervals
    )
```

The current per-GPU sequential implementation has only four query processes
and cannot produce 32 overlapping query intervals.

- [ ] **Step 4: Update order, shutdown, and failure expectations**

Require 32 query-process evidence files and 32 shutdown markers. Every
evidence file must contain exactly one GPU and one query identity. The set of
all `(GPU, query)` pairs must equal:

```python
{
    (gpu_index, query_name)
    for gpu_index in (2, 3, 4, 6)
    for query_name in EXPECTED_NVML_QUERY_NAMES
}
```

For an injected `nvmlDeviceGetPowerUsage` failure, require:

```python
assert result.stdout == ""
assert "GPU 2 query power_usage" in result.stderr
assert "nvmlDeviceGetPowerUsage" in result.stderr
assert len(list(shutdown_path.glob("*"))) == 32
```

UUID mismatch still emits no row. It may initialize all 32 children before
the parent reports startup failure, so require all started children to be
reaped and exactly 32 shutdown markers.

- [ ] **Step 5: Run RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/local-pycache-field-red \
python -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'field_queries or isolates_each_gpu_query or direct_nvml_sampler'
```

Expected: FAIL because the current sampler exposes four PIDs, preserves one
sequential eight-query chain per GPU, and writes four shutdown markers.

---

### Task 2: Implement Query-Specific Children and Generation Assembly

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Produces: `QUERY_NAMES`, `query_device(...)`, `query_worker(...)`, and four existing-schema rows assembled from a 32-cell generation
- Preserves: `_build_gpu_sampler_script() -> str` and `_start_epoch_samplers(...)`

- [ ] **Step 1: Define the frozen query identities**

In the generated script add:

```python
QUERY_NAMES = (
    "performance_state",
    "sm_clock",
    "memory_clock",
    "power_usage",
    "temperature",
    "utilization_rates",
    "memory_info",
    "clock_throttle_reasons",
)
```

Build:

```python
expected_cells = tuple(
    (index, gpu_uuid, query_name)
    for index, gpu_uuid in expected
    for query_name in QUERY_NAMES
)
```

- [ ] **Step 2: Replace `sample_device` with one-query execution**

Implement:

```python
def query_device(nvml, handle, query_name):
    started_ns = time.monotonic_ns()
    if query_name == "performance_state":
        value = ctypes.c_int()
        result = nvml.nvmlDeviceGetPerformanceState(
            handle, ctypes.byref(value)
        )
        converted = "P" + str(value.value)
        api_name = "nvmlDeviceGetPerformanceState"
    elif query_name == "sm_clock":
        value = ctypes.c_uint()
        result = nvml.nvmlDeviceGetClockInfo(
            handle, NVML_CLOCK_SM, ctypes.byref(value)
        )
        converted = value.value
        api_name = "nvmlDeviceGetClockInfo"
    elif query_name == "memory_clock":
        value = ctypes.c_uint()
        result = nvml.nvmlDeviceGetClockInfo(
            handle, NVML_CLOCK_MEM, ctypes.byref(value)
        )
        converted = value.value
        api_name = "nvmlDeviceGetClockInfo"
    elif query_name == "power_usage":
        value = ctypes.c_uint()
        result = nvml.nvmlDeviceGetPowerUsage(
            handle, ctypes.byref(value)
        )
        converted = value.value / 1000.0
        api_name = "nvmlDeviceGetPowerUsage"
    elif query_name == "temperature":
        value = ctypes.c_uint()
        result = nvml.nvmlDeviceGetTemperature(
            handle, NVML_TEMPERATURE_GPU, ctypes.byref(value)
        )
        converted = value.value
        api_name = "nvmlDeviceGetTemperature"
    elif query_name == "utilization_rates":
        value = NvmlUtilization()
        result = nvml.nvmlDeviceGetUtilizationRates(
            handle, ctypes.byref(value)
        )
        converted = {
            "gpu": value.gpu,
            "memory": value.memory,
        }
        api_name = "nvmlDeviceGetUtilizationRates"
    elif query_name == "memory_info":
        value = NvmlMemory()
        result = nvml.nvmlDeviceGetMemoryInfo(
            handle, ctypes.byref(value)
        )
        converted = value.used // (1024 * 1024)
        api_name = "nvmlDeviceGetMemoryInfo"
    elif query_name == "clock_throttle_reasons":
        value = ctypes.c_ulonglong()
        result = nvml.nvmlDeviceGetCurrentClocksThrottleReasons(
            handle, ctypes.byref(value)
        )
        converted = value.value
        api_name = "nvmlDeviceGetCurrentClocksThrottleReasons"
    else:
        raise ValueError("GPU sampler query identity is invalid")
    duration_ns = time.monotonic_ns() - started_ns
    check(nvml, api_name, result)
    return converted, duration_ns
```

- [ ] **Step 3: Implement one persistent child per cell**

Replace `device_worker` with:

```python
def query_worker(connection, cell):
    initialized = False
    nvml = None
    index, expected_uuid, query_name = cell
    try:
        nvml = configure_nvml()
        check(nvml, "nvmlInit_v2", nvml.nvmlInit_v2())
        initialized = True
        handle = ctypes.c_void_p()
        check(
            nvml,
            "nvmlDeviceGetHandleByIndex_v2",
            nvml.nvmlDeviceGetHandleByIndex_v2(
                index, ctypes.byref(handle)
            ),
        )
        uuid_buffer = ctypes.create_string_buffer(
            NVML_DEVICE_UUID_BUFFER_SIZE
        )
        check(
            nvml,
            "nvmlDeviceGetUUID",
            nvml.nvmlDeviceGetUUID(
                handle,
                uuid_buffer,
                NVML_DEVICE_UUID_BUFFER_SIZE,
            ),
        )
        observed_uuid = uuid_buffer.value.decode("utf-8")
        if observed_uuid != expected_uuid:
            raise ValueError("GPU UUID inventory changed")
        connection.send({
            "kind": "ready",
            "gpu_index": index,
            "gpu_uuid": observed_uuid,
            "query_name": query_name,
            "sampler_process_pid": os.getpid(),
        })
        while True:
            command = connection.recv()
            if command == "stop":
                break
            if (
                not isinstance(command, tuple)
                or len(command) != 2
                or command[0] != "sample"
                or isinstance(command[1], bool)
                or not isinstance(command[1], int)
                or command[1] <= 0
            ):
                raise ValueError("GPU sampler command is invalid")
            generation = command[1]
            value, duration_ns = query_device(
                nvml, handle, query_name
            )
            connection.send({
                "kind": "result",
                "generation": generation,
                "gpu_index": index,
                "gpu_uuid": observed_uuid,
                "query_name": query_name,
                "value": value,
                "query_duration_ns": duration_ns,
                "sampler_process_pid": os.getpid(),
            })
    except BaseException:
        try:
            connection.send({
                "kind": "error",
                "gpu_index": index,
                "query_name": query_name,
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
                        "gpu_index": index,
                        "query_name": query_name,
                        "detail": "nvmlShutdown failed",
                    })
                except BaseException:
                    pass
        connection.close()
```

- [ ] **Step 4: Validate the complete startup matrix**

Start every process before receiving any ready message. For each expected
cell require exact index, UUID, and query identity. Then require:

```python
ready_pids = [
    message["sampler_process_pid"]
    for message in ready_messages
]
if (
    len(ready_messages) != 32
    or len(set(ready_pids)) != 32
    or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
        for pid in ready_pids
    )
):
    raise RuntimeError("GPU sampler query processes are not isolated")
```

- [ ] **Step 5: Broadcast and validate one generation**

Use:

```python
generation = 0
while not stop_requested:
    generation += 1
    command = ("sample", generation)
    for connection in connections:
        connection.send(command)
    matrix = {}
    for cell, connection in zip(expected_cells, connections):
        message = receive_message(connection, "result")
        observed_cell = (
            message.get("gpu_index"),
            message.get("gpu_uuid"),
            message.get("query_name"),
        )
        if (
            observed_cell != cell
            or message.get("generation") != generation
            or cell in matrix
        ):
            raise RuntimeError(
                "GPU sampler generation matrix is invalid"
            )
        matrix[cell] = message
    if set(matrix) != set(expected_cells):
        raise RuntimeError(
            "GPU sampler generation matrix is incomplete"
        )
```

Do not start the next generation until this matrix has been emitted or
discarded during shutdown.

- [ ] **Step 6: Reconstruct the existing four-row schema**

For each frozen GPU build:

```python
values = {
    query_name: matrix[(index, gpu_uuid, query_name)]["value"]
    for query_name in QUERY_NAMES
}
durations = {
    query_name: matrix[
        (index, gpu_uuid, query_name)
    ]["query_duration_ns"]
    for query_name in QUERY_NAMES
}
pids = {
    query_name: matrix[
        (index, gpu_uuid, query_name)
    ]["sampler_process_pid"]
    for query_name in QUERY_NAMES
}
row = {
    "gpu_index": index,
    "gpu_uuid": gpu_uuid,
    "pstate": values["performance_state"],
    "sm_clock_mhz": values["sm_clock"],
    "memory_clock_mhz": values["memory_clock"],
    "power_w": values["power_usage"],
    "temperature_c": values["temperature"],
    "gpu_utilization_percent": values[
        "utilization_rates"
    ]["gpu"],
    "memory_utilization_percent": values[
        "utilization_rates"
    ]["memory"],
    "memory_used_mib": values["memory_info"],
    "throttle_reasons_active": values[
        "clock_throttle_reasons"
    ],
    "query_duration_ns": durations,
    "sampler_process_pid": pids,
}
```

Capture shared parent timestamps and print only after all four rows exist.

- [ ] **Step 7: Preserve owned shutdown semantics**

Send `stop` only to live owned children, close all parent pipe endpoints,
join all 32 processes, and reject unexpected nonzero exits when no signal or
active exception is present. Do not call `terminate()` on unrelated PIDs.

- [ ] **Step 8: Run focused GREEN**

Run the Task 1 selector again.

Expected: all selected tests pass, including the 32-overlapping-interval
fake-delay generation contract.

- [ ] **Step 9: Run the complete local command-timeline contract**

Run:

```bash
PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/local-pycache-field-green \
python -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_performance_gate.py
```

Expected: all tests pass; no cache output is written outside the task root.

---

### Task 3: Add a Single-Epoch Real-Workload Diagnostic Gate

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Produces: `evaluate_field_sampler_diagnostic(...) -> dict`
- Produces: `run_field_sampler_diagnostic(...) -> dict`
- Adds CLI command: `diagnose-field-sampler --run-tag <diagnostic-tag>`

- [ ] **Step 1: Add evaluator RED tests**

Build five measured intervals and complete four-GPU JSONL groups. Add one
passing case and failures for:

```text
missing in-interval group
wrong UUID set
not 32 distinct query PIDs
missing duration key
zero or boolean duration
non-empty sampler stderr
maximum complete-group gap >= shortest repeat duration
```

The passing result must equal:

```python
{
    "status": "PASS",
    "measured_repeat_count": 5,
    "complete_generation_count": expected_count,
    "maximum_generation_gap_ns": expected_gap,
    "shortest_repeat_duration_ns": expected_duration,
}
```

- [ ] **Step 2: Implement the pure evaluator**

Implement:

```python
def evaluate_field_sampler_diagnostic(
    *,
    worker: dict,
    gpu_rows: list[dict],
    gpu_uuids: list[str],
    gpu_stderr: str,
    host_stderr: str,
) -> dict:
```

Group rows by `sampled_at_unix_ns`. Every group must contain four rows, the
frozen UUID set, the exact eight duration keys per row, and 32 distinct
query-process PIDs across the group. Require positive integer durations and
empty sampler stderr. Require five measured repeats and at least one complete
group inside every repeat. Compute consecutive complete-group gaps whose
timestamps span the measured campaign window and require:

```python
maximum_generation_gap_ns < shortest_repeat_duration_ns
```

Return only the canonical PASS payload above.

- [ ] **Step 3: Add the diagnostic orchestration contract**

Test that `run_field_sampler_diagnostic`:

1. calls official preflight;
2. prepares a fresh destination from the exact source archive and patch;
3. runs `inventory-before` for schedule epoch index `2`;
4. runs only `epoch 2 graph first`;
5. runs `inventory-after`;
6. requires frozen inventory parity;
7. calls remote action `field-sampler-diagnostic-verify`;
8. copies retained evidence to the controller destination on success or
   failure;
9. never calls `assemble`, `manifest`, or either campaign verifier.

- [ ] **Step 4: Implement diagnostic orchestration**

Add:

```python
def run_field_sampler_diagnostic(
    *,
    run_tag: str,
    command_runner=subprocess.run,
    now=None,
    repo_root: Path | None = None,
    target_model: str = DEFAULT_TARGET_MODEL,
    draft_model: str = DEFAULT_DRAFT_MODEL,
) -> dict:
```

Reuse the same preflight, source archive, prepare payload, exact
`inventory-before`, `_remote_epoch`, and `inventory-after` actions as
`run_bundle`. Use schedule index `2`, whose identity is
`block-1:graph:first`. After inventory parity, call
`field-sampler-diagnostic-verify`; then create a no-overwrite controller copy.

Return:

```python
{
    "status": "PASS",
    "diagnostic_epoch": "block-1:graph:first",
    "primary_run": primary,
    "controller_run": controller,
    "diagnostic": diagnostic_receipt,
}
```

On any failure, preserve a partial controller copy and return `FAILED`.

- [ ] **Step 5: Implement the remote verification action**

`_remote_field_sampler_diagnostic_verify(tag)` loads:

```text
workers/block-1/graph.raw.json
telemetry/block-1/graph.gpu.jsonl
telemetry/block-1/graph.gpu.jsonl.stderr
telemetry/block-1/graph.host.jsonl.stderr
preflight.json
```

It invokes `evaluate_field_sampler_diagnostic`, writes the canonical receipt
exclusively to `field-sampler-diagnostic.json`, prints the receipt as compact
JSON, and returns zero only on PASS.

Register remote action `field-sampler-diagnostic-verify`.

- [ ] **Step 6: Add the CLI command**

Extend `parse_args` with:

```python
subparser = subparsers.add_parser("diagnose-field-sampler")
subparser.add_argument("--run-tag", required=True)
```

Route it to `run_field_sampler_diagnostic`. Preserve existing exit-code
semantics: PASS is zero; environment insufficiency or diagnostic failure is
two.

- [ ] **Step 7: Run focused and complete GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/local-pycache-field-diagnostic \
python -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'field_sampler_diagnostic or field_queries or direct_nvml_sampler'
```

Then rerun the complete three-file contract from Task 2.

Expected: all pass.

---

### Task 4: Review, Commit, Push, Sync, and Verify Remotely

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Produces: pushed implementation commit, detached sync receipt, remote GREEN

- [ ] **Step 1: Review the exact diff**

Run:

```bash
git diff --check -- \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git diff --stat -- \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
```

Inspect the complete focused diff. Confirm no measured worker, invariant
verifier, schedule, coverage predicate, or unrelated file changed.

- [ ] **Step 2: Commit and push exact paths**

Run:

```bash
git add \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(command-timeline): isolate NVML queries by field" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 3: Atomically sync exact files**

Use the active task-owned SSH ControlMaster and existing source transaction.
Require:

```text
local HEAD == origin/feat/kv-sparse-attention == remote source HEAD
remote runner SHA-256 == local runner SHA-256
remote test SHA-256 == local test SHA-256
```

Write the detached hash receipt beneath:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/receipts
```

- [ ] **Step 4: Run remote focused and complete contracts**

Set all five scratch/cache variables beneath the task root. Use unique
basetemps and the task-local Python. Run the focused selector from Task 3 and
the complete three-file contract from Task 2.

Expected: all pass and no output appears under remote `/tmp`.

---

### Task 5: Run the Real Diagnostic and Automatically Launch r18

**Files:**
- Remote diagnostic tag:
  `20260818-command-timeline-field-sampler-diagnostic-r1`
- Official tag:
  `20260818-command-timeline-tp4-b4-q4-r18`

**Interfaces:**
- Consumes: exact pushed/synced implementation and four clean GPUs
- Produces: immutable diagnostic receipt and, only on PASS, immutable r18

- [ ] **Step 1: Start the Mac-owned diagnostic controller**

Confirm both diagnostic primary and controller destinations are absent.
Reuse the existing Kerberos cache without refreshing it. Repeatedly invoke
official preflight and immediately execute:

```bash
python tools/run_autoregressive_draft_command_timeline_remote.py \
  diagnose-field-sampler \
  --run-tag 20260818-command-timeline-field-sampler-diagnostic-r1
```

when preflight reports `READY`.

- [ ] **Step 2: Evaluate the diagnostic receipt**

Require all ten design criteria:

1. four frozen UUIDs and eight query identities per generation;
2. 32 distinct query PIDs per generation;
3. no malformed or partial output;
4. positive real duration for every query;
5. empty GPU and host sampler stderr;
6. five measured repeats;
7. one or more complete groups inside every repeat;
8. maximum group gap below shortest repeat duration;
9. valid worker output and invariant attachment;
10. unchanged clean before/after inventory.

If any criterion fails, preserve the diagnostic and do not launch r18.

- [ ] **Step 3: Start the r18 condition controller without an approval gap**

Only after diagnostic PASS, confirm r18 primary and controller destinations
are absent. The current Mac Agent repeatedly runs official preflight and,
on `READY`, immediately replaces the polling process with:

```bash
python tools/run_autoregressive_draft_command_timeline_remote.py \
  execute \
  --run-tag 20260818-command-timeline-tp4-b4-q4-r18
```

No terminal observation or new user approval is required between READY and
execute.

- [ ] **Step 4: Follow r18 to a terminal result**

For every reached epoch require strict telemetry, 32 query PIDs per group,
empty sampler stderr, and unchanged inventory. Preserve any failure and
continue only under r19 or later after evidence-backed diagnosis.

---

### Task 6: Complete the Campaign and Reconcile Documentation

**Files:**
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-field-isolated-nvml-sampler.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-process-isolated-nvml-sampler.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-concurrent-nvml-sampler.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-nvml-query-latency.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-direct-nvml-sampler.md`
- Modify: `docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Produces: complete eight-epoch primary bundle, dual verification receipts, manifest, chronological audit, canonical handoff

- [ ] **Step 1: Require complete campaign artifacts**

Do not claim success until one official tag contains:

```text
8 completed epochs
40 measured repeats
8 worker JSON files
8 telemetry sidecars
16 clean before/after inventory snapshots
primary verification PASS
controller verification PASS
matching normalized receipts
complete checksum manifest
```

- [ ] **Step 2: Reconcile every plan chronologically**

Record r15 serialized-query evidence, r16 thread evidence, r17 process
evidence, field-isolated diagnostic evidence, and the terminal official tag.
Check every completed box only against retained artifacts and exact command
output.

- [ ] **Step 3: Update the audit and handoff**

Add a prompt-to-artifact checklist mapping:

```text
four-clean-GPU admission
Mac-owned READY-to-execute controller
strict per-repeat telemetry
four UUIDs and 32 query PIDs
eight epochs and 40 repeats
dual verifier receipts
manifest coverage
remote path containment
immutable failed tags
commit and push state
```

Classify runtime performance separately from telemetry/campaign completion.

- [ ] **Step 4: Run final verification**

Run the complete local command-timeline contract, `git diff --check`, both
remote verifiers, receipt comparison, and a manifest-to-files audit. Confirm
local HEAD, origin branch, and the remote source commit match.

- [ ] **Step 5: Commit and push reconciliation**

Stage only the plan, audit, and handoff paths changed by this reconciliation.
Commit:

```text
docs(command-timeline): reconcile field-isolated campaign
```

with the required single trailer, then push
`feat/kv-sparse-attention`.
