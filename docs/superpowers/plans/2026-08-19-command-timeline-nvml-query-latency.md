# Command-Timeline NVML Query-Latency Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record per-API direct-NVML query durations in the immutable GPU sidecar JSONL so an active CUDA Graph campaign can identify the exact source of multi-second telemetry stalls.

**Architecture:** Keep the current single-process, sequential, fail-closed direct-NVML sampler unchanged except for monotonic timing around each existing query. Attach an eight-key `query_duration_ns` mapping to each raw GPU row; telemetry attachment and canonical verifier inputs continue selecting only identity and timestamp fields.

**Tech Stack:** Python 3.11, `ctypes`, NVML C ABI, JSON Lines, pytest 8.4.2, SSH ControlMaster, existing Sitian source transaction and official command-timeline runner.

## Global Constraints

- Use only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not instrument or synchronize the measured request path.
- Do not remove, substitute, cache, or borrow any NVML field.
- Emit only complete four-GPU snapshots.
- Keep the 200 ms cadence and monotonic deadline logic unchanged.
- Preserve strict in-interval telemetry coverage.
- Put every remote artifact, cache, basetemp, receipt, and run beneath `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Every remote payload sets `TMPDIR`, `TMP`, `TEMP`, `PYTHONPYCACHEPREFIX`, and `XDG_CACHE_HOME` beneath the task root.
- Do not signal, pause, kill, adopt, or interfere with unrelated processes.
- Use a fresh immutable run tag after validation.
- Stage exact paths only; never use `git add -A`.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Add Failing Query-Duration Contract

**Files:**
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `_build_gpu_sampler_script() -> str`
- Produces: a row-level contract for `query_duration_ns: dict[str, int]`

- [ ] **Step 1: Extend the complete-snapshot assertion**

After reading the first four rows from the fake-NVML sampler, require:

```python
expected_duration_keys = {
    "performance_state",
    "sm_clock",
    "memory_clock",
    "power_usage",
    "temperature",
    "utilization_rates",
    "memory_info",
    "clock_throttle_reasons",
}
assert all(
    set(row["query_duration_ns"]) == expected_duration_keys
    for row in rows
)
assert all(
    isinstance(duration, int)
    and not isinstance(duration, bool)
    and duration >= 0
    for row in rows
    for duration in row["query_duration_ns"].values()
)
```

- [ ] **Step 2: Run RED**

Run:

```bash
TMPDIR=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TEMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
PYTHONPYCACHEPREFIX=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pycache \
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k direct_nvml_sampler_emits_complete_snapshot \
  --basetemp /Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/nvml-query-latency-red
```

Expected: FAIL with `KeyError: 'query_duration_ns'`.

---

### Task 2: Time Every Existing NVML Query

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Test: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: the existing generated sampler and its `check(name, result)` helper
- Produces: `query_duration_ns` on every emitted GPU row

- [ ] **Step 1: Add a generated timing helper**

Add this helper immediately after `check`:

```python
def timed_query(durations, key, name, query):
    started_ns = time.monotonic_ns()
    result = query()
    durations[key] = time.monotonic_ns() - started_ns
    check(name, result)
```

The helper records elapsed time before fail-closed validation so a successful
query always has one non-negative integer duration.

- [ ] **Step 2: Time the eight existing calls**

Create `query_duration_ns={}` for each device row and replace each direct
`check(...)` call in the sample loop with `timed_query(...)`. Use lambdas that
capture the existing output objects:

```python
timed_query(
    query_duration_ns,
    "power_usage",
    "nvmlDeviceGetPowerUsage",
    lambda: nvml.nvmlDeviceGetPowerUsage(
        handle, ctypes.byref(power)
    ),
)
```

Apply the same pattern to performance state, both clock calls, temperature,
utilization, memory, and throttle reasons using the exact canonical keys from
Task 1.

- [ ] **Step 3: Emit the mapping**

Add:

```python
"query_duration_ns": query_duration_ns,
```

to the row appended to `snapshot`. Do not copy this mapping into the attached
canonical telemetry rows.

- [ ] **Step 4: Run focused GREEN**

Run the Task 1 command again.

Expected: `1 passed`.

- [ ] **Step 5: Run direct-NVML contract**

Run:

```bash
TMPDIR=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
TEMP=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819 \
PYTHONPYCACHEPREFIX=/Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/pycache \
/usr/bin/python3 -m pytest -q -p no:cacheprovider \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k 'direct_nvml_sampler or epoch_samplers or boundary_only_gpu_rows' \
  --basetemp /Users/bytedance/Library/Caches/TinyLLMForge/command-timeline-20260819/nvml-query-latency-focused
```

Expected: all selected tests pass.

- [ ] **Step 6: Run full local contract**

Run the three command-timeline test files with a unique basetemp. Expected:
the complete suite passes with no failures.

- [ ] **Step 7: Commit and push**

Stage exactly:

```bash
git add \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(command-timeline): record NVML query latency" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 3: Remote Verification and Fresh Campaign

**Files:**
- Remote sync source:
  `tools/run_autoregressive_draft_command_timeline_remote.py`
- Remote sync source:
  `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: pushed query-latency implementation
- Produces: detached sync receipt, remote test evidence, and immutable active-workload JSONL

- [ ] **Step 1: Atomically sync exact files**

Use the existing source transaction engine over the task-owned SSH
ControlMaster. Verify remote source head equals local/origin and receipt hashes
equal local SHA-256 values.

- [ ] **Step 2: Run remote focused and full contracts**

Use `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/env/test/bin/python`
with all scratch variables and basetemps under the task root. Require the same
focused and full results as Tasks 2.5 and 2.6.

- [ ] **Step 3: Start a fresh official run**

Use the next unused run tag and invoke:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
TINYLLMFORGE_SSH_CONTROL_PATH=/Users/bytedance/.ssh/tinyllmforge-command-timeline-20260819.sock \
/usr/bin/python3 \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  execute --run-tag 20260818-command-timeline-tp4-b4-q4-r15
```

Official execute must retain its built-in preflight and immutable destination
rules.

- [ ] **Step 4: Summarize retained query durations**

For each GPU UUID and canonical query key, compute count, maximum, median, and
p95 from the raw graph GPU JSONL. Also list the query durations belonging to
snapshots immediately before and after every uncovered measured interval.

Expected: one or more APIs are identified with durations that explain the
observed multi-second snapshot gap. If no single API dominates, the evidence
must instead establish aggregate serialized query cost before any concurrency
change is proposed.

- [ ] **Step 5: Preserve claim boundary**

Do not classify runtime performance or authorize sampler optimization from the
latency instrumentation alone. Use the evidence to write the next minimal
optimization design, then continue with TDD and a new immutable run tag.
