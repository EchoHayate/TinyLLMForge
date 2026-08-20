# Command-Timeline Direct NVML Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blocking persistent `nvidia-smi` telemetry query with a dependency-free direct-NVML sampler, then let the current Mac Agent automatically launch and verify fresh `r9` as soon as four clean GPUs are available.

**Architecture:** Keep the command-timeline worker, schedule, artifact paths, and strict telemetry admission unchanged. Generate a standalone Python sampler that loads `libnvidia-ml.so.1` through `ctypes`, validates the frozen index/UUID inventory, emits only complete four-GPU snapshots, and shuts NVML down on every exit path. After local and remote validation, use a Mac-owned polling loop that calls official `execute` immediately after official preflight reports `READY`.

**Tech Stack:** Python 3.11, `ctypes`, NVML C ABI, JSON Lines, POSIX signals, pytest 8.4.2, SSH ControlMaster, existing Sitian source transaction and verifier pipeline.

## Global Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Preserve immutable `r6`, `r7`, and `r8` primary/controller artifacts.
- Use fresh tag `20260818-command-timeline-tp4-b4-q4-r9`.
- Require four distinct GPUs with memory used at most 1024 MiB, utilization at most 5%, and no compute processes.
- Do not signal, pause, kill, adopt, or otherwise interfere with unrelated processes.
- Do not borrow boundary telemetry or duplicate snapshots across repeats.
- Do not instrument or synchronize the measured request path.
- Put all remote task output, caches, temporary files, receipts, and test basetemps under `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Every remote payload sets `TMPDIR`, `TMP`, `TEMP`, `PYTHONPYCACHEPREFIX`, and `XDG_CACHE_HOME` beneath that task root.
- Do not write task output to local or remote `/`, `/tmp`, or `/private/tmp`.
- Do not manually refresh Kerberos.
- Stage exact paths only; never use `git add -A`.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Add Failing Direct-NVML Sampler Contracts

**Files:**
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: existing `_build_gpu_sampler_script() -> str` and `_start_epoch_samplers(...)`.
- Produces: executable fake-NVML contracts for the generated sampler and frozen inventory propagation.

- [ ] **Step 1: Add a fake ctypes-compatible NVML layer**

Add test helpers whose callable functions accept `.argtypes` and `.restype`
assignment and write through ctypes output pointers:

```python
class _FakeNvmlFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *arguments):
        return self.callback(*arguments)


def _fake_nvml_prelude(*, query_failure=None, uuid_mismatch=False):
    return """
import ctypes
import signal

class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None
    def __call__(self, *arguments):
        return self.callback(*arguments)

class FakeNvml:
    ...

ctypes.CDLL = lambda _name: FakeNvml()
"""
```

The fake must implement `nvmlInit_v2`, `nvmlShutdown`,
`nvmlErrorString`, `nvmlDeviceGetHandleByIndex_v2`,
`nvmlDeviceGetUUID`, `nvmlDeviceGetPerformanceState`,
`nvmlDeviceGetClockInfo`, `nvmlDeviceGetPowerUsage`,
`nvmlDeviceGetTemperature`, `nvmlDeviceGetUtilizationRates`,
`nvmlDeviceGetMemoryInfo`, and
`nvmlDeviceGetCurrentClocksThrottleReasons`.

- [ ] **Step 2: Add the complete-snapshot test**

Execute the generated sampler after the fake prelude with:

```python
inventory = [
    {"index": 2, "uuid": "GPU-2"},
    {"index": 3, "uuid": "GPU-3"},
    {"index": 4, "uuid": "GPU-4"},
    {"index": 6, "uuid": "GPU-6"},
]
```

Terminate it after the first four emitted rows and assert:

```python
assert [row["gpu_index"] for row in rows] == [2, 3, 4, 6]
assert [row["gpu_uuid"] for row in rows] == [
    "GPU-2", "GPU-3", "GPU-4", "GPU-6"
]
assert len({row["sampled_at_unix_ns"] for row in rows}) == 1
assert len({row["sampled_at_monotonic_ns"] for row in rows}) == 1
assert all(row["pstate"] == "P0" for row in rows)
assert all(row["power_w"] == 70.0 for row in rows)
assert all(row["memory_used_mib"] == 100 for row in rows)
```

- [ ] **Step 3: Add fail-closed tests**

Add separate tests for:

```text
frozen UUID mismatch -> ValueError: GPU UUID inventory changed
nvmlDeviceGetPowerUsage failure -> RuntimeError naming that API
invalid or duplicated inventory -> ValueError: GPU inventory is invalid
```

For UUID and query failures, assert stdout contains no JSON rows.

- [ ] **Step 4: Add SIGTERM/shutdown test**

Run the fake sampler until initialization is observed, terminate the
sampler, and assert the fake `nvmlShutdown` marker is written exactly once
and the process exits with status zero.

- [ ] **Step 5: Update epoch sampler contract**

Change `_start_epoch_samplers` expectations so the GPU sampler command
receives one JSON inventory containing both `index` and `uuid`. Assert the
host sampler command and stderr-file ownership remain unchanged.

- [ ] **Step 6: Run RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/local-pycache \
python -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'direct_nvml_sampler or epoch_samplers or boundary_only_gpu_rows'
```

Expected: direct-NVML tests fail because the runner still invokes
`nvidia-smi`, and the inventory propagation test fails because
`_start_epoch_samplers` does not yet accept `gpu_uuids`.

---

### Task 2: Implement the Direct-NVML Sampler

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `gpu_indices: list[int]` and `gpu_uuids: list[str]` frozen by official preflight.
- Produces: `_build_gpu_sampler_script() -> str` and `_start_epoch_samplers(*, gpu_indices, gpu_uuids, gpu_path, host_path, source_root=None)`.

- [ ] **Step 1: Replace the generated `nvidia-smi` program**

Generate a script that defines:

```python
class NvmlUtilization(ctypes.Structure):
    _fields_ = [
        ("gpu", ctypes.c_uint),
        ("memory", ctypes.c_uint),
    ]


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]
```

Set explicit signatures for every NVML function. Use
`ctypes.c_void_p` for `nvmlDevice_t`, `ctypes.c_int` for
`nvmlReturn_t`, and the header-defined constants:

```python
NVML_SUCCESS = 0
NVML_TEMPERATURE_GPU = 0
NVML_CLOCK_SM = 1
NVML_CLOCK_MEM = 2
NVML_DEVICE_UUID_BUFFER_SIZE = 80
```

- [ ] **Step 2: Validate and freeze device identity**

Parse one JSON list of four objects. Reject non-integer/negative/duplicate
indices and empty/duplicate UUIDs. Resolve each handle by index and compare
`nvmlDeviceGetUUID` with the frozen UUID before entering the sample loop.

- [ ] **Step 3: Emit only complete snapshots**

For each handle, query every required field into fresh ctypes values. If
any call fails, raise before printing any row from that iteration. After
all four rows are complete, capture:

```python
unix_ns = time.time_ns()
monotonic_ns = time.monotonic_ns()
```

and print the four rows with the existing schema. Convert milliwatts to
watts and bytes to MiB. Format the UTC timestamp with millisecond precision.

- [ ] **Step 4: Add bounded cadence and clean shutdown**

Use a monotonic deadline:

```python
next_sample_ns = time.monotonic_ns()
while not stop_requested:
    ...
    next_sample_ns += 200_000_000
    remaining_ns = next_sample_ns - time.monotonic_ns()
    if remaining_ns > 0:
        time.sleep(remaining_ns / 1_000_000_000)
    else:
        next_sample_ns = time.monotonic_ns()
```

Install SIGTERM/SIGINT handlers that set `stop_requested`. In `finally`,
call `nvmlShutdown()` once if initialization succeeded. A shutdown failure
is fatal unless a signal-requested clean exit is already in progress.

- [ ] **Step 5: Propagate frozen UUIDs**

Add `gpu_uuids` to `_start_epoch_samplers`, validate it as a four-item
inventory, pass:

```python
json.dumps([
    {"index": index, "uuid": uuid}
    for index, uuid in zip(gpu_indices, gpu_uuids)
])
```

to the generated sampler, and update `_remote_epoch` to pass its existing
preflight-frozen `gpu_uuids`.

- [ ] **Step 6: Run focused GREEN**

Run the Task 1 RED command again.

Expected: all selected tests pass with no warning or cache output.

- [ ] **Step 7: Run full local contract**

Run:

```bash
PYTHONPYCACHEPREFIX=/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/local-pycache \
python -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_performance_gate.py
```

Expected: the complete three-file contract passes.

- [ ] **Step 8: Review and commit**

Run `git diff --check`, inspect only the two implementation files, stage
those exact paths, and commit:

```bash
git add \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(command-timeline): sample GPUs through direct NVML" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 3: Remote Validation and Cadence Smoke

**Files:**
- Remote sync source:
  `tools/run_autoregressive_draft_command_timeline_remote.py`
- Remote sync source:
  `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: pushed direct-NVML implementation commit.
- Produces: detached sync receipt, remote focused/full GREEN, and a real direct-NVML cadence receipt.

- [ ] **Step 1: Atomically sync exact files**

Use the existing source transaction engine and task-owned SSH channel.
Verify remote source commit equals local/origin and detached receipt hashes
equal local SHA-256 values.

- [ ] **Step 2: Run remote focused tests**

Set all scratch/cache variables beneath the Sitian task root and run:

```bash
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python \
  -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'direct_nvml_sampler or epoch_samplers or boundary_only_gpu_rows' \
  --basetemp /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/pytest/direct-nvml-focused
```

- [ ] **Step 3: Run remote full contract**

Run the same three files as Task 2 Step 7 with a unique basetemp under
`.../pytest/direct-nvml-full`.

- [ ] **Step 4: Run real cadence smoke**

Start only the generated sampler for the selected four currently idle GPUs,
write stdout/stderr beneath:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/validation/direct-nvml-sampler-smoke-20260819
```

Run for at least five seconds, terminate only the owned sampler, and verify:

- every timestamp group has exactly four distinct expected UUIDs;
- no JSON or NVML errors occurred;
- at least 20 complete snapshots exist;
- median and maximum inter-snapshot gaps are recorded;
- the maximum gap is below 1000 ms.

Do not start `r9` if this smoke fails.

---

### Task 4: Mac-Agent Automatic r9 Controller

**Files:**
- No repository file changes before campaign completion.
- Runtime logs and receipts: remote Sitian task root only.

**Interfaces:**
- Consumes: official `preflight` and `execute` commands at the pushed/synced commit.
- Produces: automatically launched immutable `r9` or a fail-fast Kerberos/transport receipt.

- [ ] **Step 1: Verify fresh destinations**

Confirm both `r9` paths are absent:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/runs/20260818-command-timeline-tp4-b4-q4-r9
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/controller-verification/20260818-command-timeline-tp4-b4-q4-r9
```

- [ ] **Step 2: Start the active local controller**

The current Mac Agent owns one long-running process that:

1. invokes official `preflight` every 15 seconds;
2. records bounded poll results beneath the remote task root;
3. retries transient SSH failures;
4. exits on insufficient Kerberos lifetime without refreshing credentials;
5. calls official `execute` immediately on `READY`;
6. does not print-and-wait for user input;
7. preserves execute's built-in second preflight;
8. if that second preflight reports a vanished GPU window without creating
   run destinations, returns to polling;
9. exits after execute creates the immutable campaign result.

- [ ] **Step 3: Observe without interfering**

While waiting, inspect only official preflight results and task-owned
controller state. Do not inspect or manipulate unrelated process groups
beyond the read-only GPU process inventory already used by preflight.

---

### Task 5: Verify r9 and Reconcile Canonical Records

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-19-command-timeline-direct-nvml-sampler.md`

**Interfaces:**
- Consumes: immutable r6/r7/r8/r9 artifacts and verifier receipts.
- Produces: canonical classification, chronological reconciliation, final commit, and pushed branch.

- [ ] **Step 1: Require complete campaign shape**

Verify all eight epochs exist:

```text
block-0:eager:first
block-0:graph:second
block-1:graph:first
block-1:eager:second
block-2:graph:first
block-2:eager:second
block-3:eager:first
block-3:graph:second
```

- [ ] **Step 2: Require independent agreement**

Require primary verifier PASS, controller verifier PASS, normalized receipt
agreement, and successful `manifest.sha256` validation. Read the canonical
classification only from verified artifacts.

- [ ] **Step 3: Reconcile chronology**

Record:

- r6 ownership race and commit `d975a30`;
- r7 subprocess-sampling coverage failure;
- persistent sampler commits `6b67ebb`, `2a7e8a5`, and `07bf4fc`;
- r8 successful worker plus 2129 ms `nvidia-smi` stall and strict failure;
- direct-NVML design, tests, commit, remote receipt, and cadence smoke;
- r9 GPU inventory, eight epochs, dual verifier evidence, manifest result,
  and final classification;
- explicit runtime-optimization authorization boundary.

- [ ] **Step 4: Mark this plan accurately**

Check only steps supported by preserved command/test/run evidence. Do not
mark campaign or verifier steps complete after a partial or failed r9.

- [ ] **Step 5: Final verification and push**

Run focused documentation checks, `git diff --check`, exact-path status,
and inspect the final diff. Stage only the audit, handoff, and this plan:

```bash
git add \
  docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/plans/2026-08-19-command-timeline-direct-nvml-sampler.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(command-timeline): reconcile direct NVML campaign" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Expected: local and origin heads match; remote source remains bound to the
implementation commit; unrelated untracked `artifacts/` and `experiments/`
remain untouched.

## 2026-08-20 Final Reconciliation

The direct NVML sampler was an evidence-producing intermediate design. Later
iterations retained its per-query duration telemetry but changed the execution
architecture to concurrent, process-isolated, and field-isolated collection.
The unchecked implementation steps above remain a historical plan record and
are not the source of truth for the terminal campaign.

Immutable tag `20260818-command-timeline-tp4-b4-q4-r23`, bound to
`596e724ea87966b2ab3b47cccda08c106f9084bb`, completed all eight epochs and
40 measured repeats. The retained raw telemetry contains 5,945 complete
four-GPU groups. Every complete group contains the four expected UUIDs,
32 distinct `(GPU, query)` process IDs, and eight positive query durations per
GPU row; all epoch sampler stderr files are empty.

The final verifier chain passed identity correctness and timeline
conservation, but stationarity failed. The canonical runtime classification is
`PAIRED_PROTOCOL_UNSTABLE`, with no localized boundary. Therefore direct-query
latency evidence is established, while sampler optimization and performance
improvement remain unauthorized and unestablished.
