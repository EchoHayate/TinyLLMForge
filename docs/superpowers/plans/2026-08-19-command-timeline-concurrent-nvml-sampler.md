# Command-Timeline Concurrent NVML Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Query the four frozen GPUs concurrently while preserving sequential per-GPU NVML calls, complete-snapshot emission, and strict fail-closed telemetry.

**Architecture:** Reuse one standard-library `ThreadPoolExecutor` with exactly four workers inside the generated sampler. Each worker executes one GPU's existing eight-call timed query chain in order and returns an unprinted row; the main thread materializes all four rows in frozen inventory order before assigning shared timestamps and printing.

**Tech Stack:** Python 3.11, `concurrent.futures`, `ctypes`, NVML C ABI, JSON Lines, pytest 8.4.2, SSH ControlMaster, existing Sitian source transaction and official command-timeline runner.

## Global Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not instrument or synchronize the measured request path.
- Keep all eight NVML fields and all eight `query_duration_ns` values.
- Keep calls sequential and in the existing order within each GPU.
- Emit no row until one complete four-GPU snapshot has succeeded.
- Keep the 200 ms cadence and monotonic deadline logic unchanged.
- Preserve strict in-interval telemetry coverage, with no boundary borrowing or synthetic duplication.
- Keep the official four-clean-GPU admission thresholds unchanged.
- Put every remote artifact, cache, basetemp, receipt, and run beneath `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Every remote payload sets `TMPDIR`, `TMP`, `TEMP`, `PYTHONPYCACHEPREFIX`, and `XDG_CACHE_HOME` beneath the task root.
- Do not signal, pause, kill, adopt, or interfere with unrelated processes.
- Use fresh immutable tag `20260818-command-timeline-tp4-b4-q4-r16`.
- Stage exact paths only; never use `git add -A`.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Prove Cross-GPU Concurrency and Preserved Ordering

**Files:**
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `_build_gpu_sampler_script() -> str`
- Produces: fake-NVML proofs for concurrent device chains and unchanged per-device call order

- [ ] **Step 1: Extend the fake NVML prelude**

Add optional `concurrency_probe: bool = False` and `order_path=None` arguments.
Import `json` and `threading` in the generated prelude. Give `FakeNvml`:

```python
self.query_barrier = (
    threading.Barrier(4) if concurrency_probe else None
)
self.query_order = {}
self.query_order_lock = threading.Lock()
```

Record every query by device index:

```python
def record(self, device, name):
    index = int(device.value) - 100
    with self.query_order_lock:
        self.query_order.setdefault(str(index), []).append(name)
```

In `get_pstate`, record `"performance_state"` and then wait for all four
device chains when the barrier is enabled:

```python
if self.query_barrier is not None:
    self.query_barrier.wait(timeout=1)
```

Record the remaining names as `"sm_clock"`, `"memory_clock"`,
`"power_usage"`, `"temperature"`, `"utilization_rates"`, `"memory_info"`,
and `"clock_throttle_reasons"`. In `shutdown`, write `query_order` as JSON
when `order_path` is provided, then update the existing shutdown counter.

- [ ] **Step 2: Add the concurrency test**

Launch the generated sampler with `concurrency_probe=True`, require four
nonempty JSON lines, terminate the owned sampler, and assert the four expected
GPU indices:

```python
assert [row["gpu_index"] for row in rows] == [2, 3, 4, 6]
```

The barrier is the proof: a serialized sampler cannot let all four first
queries enter simultaneously.

- [ ] **Step 3: Add the per-device order characterization**

Launch one snapshot with `order_path`, terminate the owned sampler, load the
JSON file, and require every device to have this exact prefix:

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

Use a prefix because the sampler may begin another complete query chain before
SIGTERM is observed.

- [ ] **Step 4: Strengthen the failure characterization**

In the existing injected `nvmlDeviceGetPowerUsage` failure case, continue
requiring nonzero status, a named error, and empty stdout. Also assert the
shutdown counter equals `1`.

- [ ] **Step 5: Run RED**

Run:

```bash
mkdir -p /Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/{pycache,pytest}
TMPDIR=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TEMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
PYTHONPYCACHEPREFIX=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pycache \
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'direct_nvml_sampler_queries_devices_concurrently or direct_nvml_sampler_preserves_per_device_query_order or direct_nvml_sampler_fails_closed' \
  --basetemp /Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pytest/concurrent-nvml-red
```

Expected: the concurrency test fails because the first serialized fake-NVML
query times out waiting at the four-party barrier. The ordering and existing
fail-closed characterizations may already pass.

---

### Task 2: Implement One Reused Four-Worker Executor

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: the frozen `devices: list[tuple[int, str, ctypes.c_void_p]]`
- Produces: `sample_device(device) -> dict` and complete snapshots collected through `executor.map`

- [ ] **Step 1: Import the standard-library executor in the generated script**

Add:

```python
from concurrent.futures import ThreadPoolExecutor
```

No package dependency or runtime configuration is added.

- [ ] **Step 2: Extract one sequential device chain**

Move the existing per-device ctypes allocation, eight `timed_query` calls,
and row construction into:

```python
def sample_device(device):
    index, gpu_uuid, handle = device
    pstate = ctypes.c_int()
    sm_clock = ctypes.c_uint()
    memory_clock = ctypes.c_uint()
    power = ctypes.c_uint()
    temperature = ctypes.c_uint()
    utilization = NvmlUtilization()
    memory = NvmlMemory()
    throttle = ctypes.c_ulonglong()
    query_duration_ns = {}
    timed_query(
        query_duration_ns,
        "performance_state",
        "nvmlDeviceGetPerformanceState",
        lambda: nvml.nvmlDeviceGetPerformanceState(
            handle, ctypes.byref(pstate)
        ),
    )
    timed_query(
        query_duration_ns,
        "sm_clock",
        "nvmlDeviceGetClockInfo",
        lambda: nvml.nvmlDeviceGetClockInfo(
            handle, NVML_CLOCK_SM, ctypes.byref(sm_clock)
        ),
    )
    timed_query(
        query_duration_ns,
        "memory_clock",
        "nvmlDeviceGetClockInfo",
        lambda: nvml.nvmlDeviceGetClockInfo(
            handle, NVML_CLOCK_MEM, ctypes.byref(memory_clock)
        ),
    )
    timed_query(
        query_duration_ns,
        "power_usage",
        "nvmlDeviceGetPowerUsage",
        lambda: nvml.nvmlDeviceGetPowerUsage(
            handle, ctypes.byref(power)
        ),
    )
    timed_query(
        query_duration_ns,
        "temperature",
        "nvmlDeviceGetTemperature",
        lambda: nvml.nvmlDeviceGetTemperature(
            handle, NVML_TEMPERATURE_GPU, ctypes.byref(temperature)
        ),
    )
    timed_query(
        query_duration_ns,
        "utilization_rates",
        "nvmlDeviceGetUtilizationRates",
        lambda: nvml.nvmlDeviceGetUtilizationRates(
            handle, ctypes.byref(utilization)
        ),
    )
    timed_query(
        query_duration_ns,
        "memory_info",
        "nvmlDeviceGetMemoryInfo",
        lambda: nvml.nvmlDeviceGetMemoryInfo(
            handle, ctypes.byref(memory)
        ),
    )
    timed_query(
        query_duration_ns,
        "clock_throttle_reasons",
        "nvmlDeviceGetCurrentClocksThrottleReasons",
        lambda: nvml.nvmlDeviceGetCurrentClocksThrottleReasons(
            handle, ctypes.byref(throttle)
        ),
    )
    return {
        "gpu_index": index,
        "gpu_uuid": gpu_uuid,
        "pstate": "P" + str(pstate.value),
        "sm_clock_mhz": sm_clock.value,
        "memory_clock_mhz": memory_clock.value,
        "power_w": power.value / 1000.0,
        "temperature_c": temperature.value,
        "gpu_utilization_percent": utilization.gpu,
        "memory_utilization_percent": utilization.memory,
        "memory_used_mib": memory.used // (1024 * 1024),
        "throttle_reasons_active": throttle.value,
        "query_duration_ns": query_duration_ns,
    }
```

- [ ] **Step 3: Reuse exactly four workers**

Initialize `executor = None` before the NVML `try`. After all four handles and
UUIDs are validated, create:

```python
executor = ThreadPoolExecutor(max_workers=4)
```

Replace the serialized device loop with:

```python
snapshot = list(executor.map(sample_device, devices))
```

Do not print inside `sample_device`. `executor.map` preserves frozen input
order, while `list(...)` ensures all four rows exist before timestamping.

- [ ] **Step 4: Shut down threads before NVML**

At the start of `finally`, add:

```python
if executor is not None:
    executor.shutdown(wait=True, cancel_futures=False)
```

Then retain the existing single `nvmlShutdown` call and error semantics.

- [ ] **Step 5: Run focused GREEN**

Run the Task 1 command again.

Expected: all selected tests pass.

- [ ] **Step 6: Run the direct-NVML and telemetry contract**

Run:

```bash
TMPDIR=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TEMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
PYTHONPYCACHEPREFIX=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pycache \
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'direct_nvml_sampler or epoch_samplers or boundary_only_gpu_rows' \
  --basetemp /Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pytest/concurrent-nvml-focused
```

Expected: all selected tests pass with no warning or cache output.

- [ ] **Step 7: Run the complete local command-timeline contract**

Run:

```bash
TMPDIR=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TEMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
PYTHONPYCACHEPREFIX=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pycache \
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_performance_gate.py \
  --basetemp /Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pytest/concurrent-nvml-full
```

Expected: the complete suite passes with no failures.

- [ ] **Step 8: Review, commit, and push**

Run `git diff --check` and inspect only the two implementation paths. Stage:

```bash
git add \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(command-timeline): query GPUs concurrently" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 3: Sync and Verify the Exact Remote Source

**Files:**
- Remote sync source: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Remote sync source: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: pushed implementation commit
- Produces: remote source head parity, detached SHA-256 receipt, focused GREEN, and full GREEN

- [ ] **Step 1: Atomically sync exact files**

Use the existing source transaction engine over
`/Users/bytedance/.ssh/tinyllmforge-command-timeline-20260819.sock`. Set every
remote scratch variable beneath the task root. Verify:

```text
local HEAD == origin/feat/kv-sparse-attention == remote source HEAD
local file SHA-256 == detached sync receipt SHA-256
```

- [ ] **Step 2: Run remote focused tests**

Use:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python
```

Run the same focused selector as Task 2 Step 6 with a unique remote basetemp
beneath `.../pytest/concurrent-nvml-focused`.

- [ ] **Step 3: Run remote full tests**

Run the same three-file contract as Task 2 Step 7 with a unique remote
basetemp beneath `.../pytest/concurrent-nvml-full`.

- [ ] **Step 4: Preserve validation evidence**

Record the local and remote test counts, source commit, receipt path, and exact
file hashes in the implementation plan reconciliation before launching r16.

---

### Task 4: Automatically Launch and Evaluate Fresh r16

**Files:**
- No repository implementation files before result analysis.
- Remote run destination:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/runs/20260818-command-timeline-tp4-b4-q4-r16`
- Remote controller verification:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/controller-verification/20260818-command-timeline-tp4-b4-q4-r16`

**Interfaces:**
- Consumes: official `preflight` and `execute` at the synced commit
- Produces: one immutable r16 outcome, automatically started on `READY`

- [ ] **Step 1: Validate fresh destinations and Kerberos lifetime**

Confirm both r16 destinations are absent. Check the existing Kerberos cache
without refreshing it. If its remaining lifetime cannot cover the bounded
campaign, preserve a fail-fast receipt rather than starting work that cannot
finish.

- [ ] **Step 2: Run the Mac-owned condition controller**

The current Mac Agent repeatedly invokes official `preflight`. It must:

- retain the exact four-clean-GPU classifier;
- avoid signaling or adopting any external process;
- sleep only between non-READY observations;
- stop on transport, Kerberos, or malformed-preflight failure;
- invoke official `execute --run-tag 20260818-command-timeline-tp4-b4-q4-r16`
  immediately in the same controller when preflight returns `READY`;
- keep all generated remote state under the Sitian task root.

- [ ] **Step 3: Evaluate strict telemetry first**

Require every measured repeat in every completed epoch to contain at least one
in-interval complete four-GPU snapshot. Check raw timestamp groups for exactly
four distinct expected UUIDs and retained `query_duration_ns`.

- [ ] **Step 4: Continue from the immutable result**

If r16 exposes a later strict-gate failure, preserve it, write one evidence-
backed hypothesis, add a failing regression test, and use r17 or later after a
verified fix. Do not reuse r16 or weaken any gate.

If all eight epochs complete, run both verifiers, build the manifest, and
continue to final audit reconciliation.

---

### Task 5: Reconcile Documentation and Finish the Campaign

**Files:**
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-concurrent-nvml-sampler.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-nvml-query-latency.md`
- Modify: `docs/superpowers/plans/2026-08-19-command-timeline-direct-nvml-sampler.md`
- Modify: `docs/superpowers/audits/2026-08-18-autoregressive-draft-command-timeline-local-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: immutable retries, verifier outputs, manifest, commits, and sync receipts
- Produces: chronological audit and handoff whose claims match preserved evidence

- [ ] **Step 1: Reconcile every checkbox and retry**

Record r11 through the final tag, including each exact failure, diagnostic
commit, test count, pushed commit, remote receipt, and claim boundary.

- [ ] **Step 2: Run final verification**

Run both verifiers, validate the manifest hashes, rerun the complete local
contract, run `git diff --check`, and verify local/origin parity.

- [ ] **Step 3: Commit and push exact documentation paths**

Stage only the five listed documentation files. Commit with one TRAE CLI
trailer and push `feat/kv-sparse-attention`.

- [ ] **Step 4: Report the result without overclaiming**

Separate local tests, remote tests, telemetry completeness, verifier results,
manifest integrity, and runtime classification. Do not claim a runtime
optimization result unless the final verified campaign establishes it.

## 2026-08-20 Final Reconciliation

The concurrent sampler was superseded after the campaign exposed that
thread-level concurrency did not provide the required independent query
execution evidence. The subsequent process-isolated and field-isolated plans
preserved the strict four-GPU admission and raw-duration contracts.

The immutable retry chain continued through r23. The terminal tag
`20260818-command-timeline-tp4-b4-q4-r23` used source commit
`596e724ea87966b2ab3b47cccda08c106f9084bb`, GPUs `2,4,5,6`, eight epochs,
and 40 measured repeats. Its primary and controller artifacts are
byte-identical; their receipts are equal after excluding only
`artifact_path`, `verification_location`, and `verified_at_utc`; each manifest
has exact `279/279` coverage and passes `sha256sum -c`.

The final classification is intentionally split:

```text
COMMAND_TIMELINE_LOCAL_IMPLEMENTATION=ESTABLISHED
COMMAND_TIMELINE_REMOTE_BUNDLE=COMPLETE
COMMAND_TIMELINE_TELEMETRY=ESTABLISHED
COMMAND_TIMELINE_DUAL_VERIFICATION=PASS
BOUNDARY_LOCALIZED=NOT_ESTABLISHED
RUNTIME_CLASSIFICATION=PAIRED_PROTOCOL_UNSTABLE
RUNTIME_OPTIMIZATION=NOT_AUTHORIZED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

Thus the campaign and evidence-integrity work completed, but no runtime
optimization result was established.
