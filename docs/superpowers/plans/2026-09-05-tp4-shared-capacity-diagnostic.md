# TP4 Shared-Capacity Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the frozen Qwen3.8-27B BF16 TP4 decode-replay matrix on four lightly occupied GPUs without weakening the existing strict-clean production gate or misrepresenting shared-host measurements as formal performance evidence.

**Architecture:** Add an explicit admission mode to the controller plan. `strict_clean` retains all existing behavior; `shared_capacity` admits four stable GPUs with at most 20,480 MiB existing allocation and at most 5% utilization, records the baseline process identities, uses a 0.95 whole-device allocator ceiling together with workload-bounded KV capacity, and forces the final claim boundary to `DIAGNOSTIC_ONLY`. The local supervisor chooses and guards the same four GPUs for one minute, tolerates only the frozen baseline PIDs plus exact-tag owned PIDs, and stops only exact-tag owned work if a new foreign process, PID reuse, or unsafe baseline-memory growth appears.

**Tech Stack:** Python 3, dependency-light `unittest`/assert tests, SSH, `nvidia-smi`, vLLM-compatible engine configuration, Git.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`; `/Users/bytedance/Desktop/TinyLLMForge` is its symlink.
- Do not use `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Keep the frozen TP4/BF16/greedy Q0-Q2, five-repetition, 30-case/15-pair matrix unchanged.
- Keep strict-clean admission as the default and retain its formal classification semantics.
- `shared_capacity` requires exactly four GPUs, each at `memory_used_mib <= 20_480` and `utilization_percent <= 5` for four samples at 15-second intervals.
- `shared_capacity` uses `gpu_memory_utilization=0.95` as the whole-device
  initialization ceiling and explicitly limits `num_kvcache_blocks` to
  `Q0=8`, `Q1=16`, and `Q2=36`.
- Freeze baseline compute-process identity as `(pid, start_time_ticks)` before launch.
- Abort only this run when a new foreign PID, a reused baseline PID, or unsafe baseline-memory growth is observed; never terminate external workloads.
- Every shared-capacity result is `DIAGNOSTIC_ONLY`, regardless of measured performance. A formal GO still requires a strict-clean rerun.
- Remote cache, logs, temporary files, and artifacts must remain below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not run `kinit` or `krenew`.
- Use exact-path staging only and preserve unrelated untracked artifacts.

---

### Task 1: Freeze Admission Mode and Plan Identity

**Files:**
- Modify: `tools/test_run_tp4_decode_replay.py`
- Modify: `tools/run_tp4_decode_replay.py`

**Interfaces:**
- Consumes: `build_plan(run_tag, source_identity, selected_gpus, admission_mode)`
- Produces: a validated plan containing `admission_mode`, `claim_boundary`, and four mode-valid GPU identities

- [ ] **Step 1: Write failing controller-plan tests**

Add tests proving:

```python
shared = _plan(
    admission_mode="shared_capacity",
    selected_gpus=[
        _gpu(0),
        _gpu(1, memory=14_382),
        _gpu(5, memory=18_854),
        _gpu(7),
    ],
)
assert shared["admission_mode"] == "shared_capacity"
assert shared["claim_boundary"] == "DIAGNOSTIC_ONLY"
```

Also prove that shared mode rejects memory above 20,480 MiB, utilization above 5%, duplicate GPU identity, and unsupported admission modes; strict-clean remains the default and still rejects any foreign process.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 tools/test_run_tp4_decode_replay.py
```

Expected: failure because `build_plan` does not accept or persist `admission_mode`.

- [ ] **Step 3: Implement mode-aware plan normalization**

Add constants for `strict_clean`, `shared_capacity`, 20,480 MiB, and `DIAGNOSTIC_ONLY`. Validate GPU identity separately from mode policy, preserve strict-clean defaults, and include the mode and claim boundary in plan validation.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python3 tools/test_run_tp4_decode_replay.py
```

Expected: all controller tests pass.

### Task 2: Record and Revalidate Shared Baseline Processes

**Files:**
- Modify: `tools/test_run_tp4_decode_replay.py`
- Modify: `tools/run_tp4_decode_replay.py`

**Interfaces:**
- Consumes: selected GPU inventory containing compute-process rows with PID and used-memory fields
- Produces: mode-specific admission receipt with immutable baseline process identities and memory totals

- [ ] **Step 1: Write failing admission tests**

Add tests proving shared admission:

```python
assert receipt["admission_mode"] == "shared_capacity"
assert receipt["claim_boundary"] == "DIAGNOSTIC_ONLY"
assert receipt["baseline_compute_processes"] == [
    {
        "gpu_uuid": "GPU-0001",
        "pid": 101,
        "start_time_ticks": 9001,
        "used_memory_mib": 14_000,
    }
]
```

Also prove admission rejects a missing/unreadable start time, PID reuse between monitor and readmission, a new PID, utilization above 5%, or memory above 20,480 MiB.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 tools/test_run_tp4_decode_replay.py
```

Expected: failure because shared baseline capture and validation are absent.

- [ ] **Step 3: Implement shared admission and evidence**

Add a mode-dispatched admission method. For shared mode, re-query the frozen GPUs, obtain `/proc/<pid>/stat` start times, compare them with the monitor receipt, persist `shared_capacity_admission.json`, and retain the baseline identity for post-launch safety checks. Keep `strict_clean_admission.json` and its schema unchanged for strict mode.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python3 tools/test_run_tp4_decode_replay.py
```

Expected: all controller tests pass.

### Task 3: Apply the Shared-Capacity Worker Memory Budget

**Files:**
- Modify: `tools/test_tp4_decode_replay_worker.py`
- Modify: `tools/tp4_decode_replay_worker.py`
- Modify: `tools/run_tp4_decode_replay.py`

**Interfaces:**
- Consumes: `TINYLLMFORGE_TP4_ADMISSION_MODE`
- Produces: `gpu_memory_utilization=0.84` for strict-clean; shared-capacity
  uses a `0.95` whole-device ceiling plus workload-bounded KV blocks

- [ ] **Step 1: Write the failing worker test**

Add:

```python
with mock.patch.dict(
    os.environ,
    {"TINYLLMFORGE_TP4_ADMISSION_MODE": "shared_capacity"},
):
    config = worker.build_engine_config(arm="eager", workload="Q1")
assert config["gpu_memory_utilization"] == 0.95
assert config["num_kvcache_blocks"] == 16
```

Prove the default remains 0.84 and unknown values fail closed.

- [ ] **Step 2: Run the focused worker test and verify RED**

Run:

```bash
python3 tools/test_tp4_decode_replay_worker.py
```

Expected: failure because the worker ignores the admission mode and does not
bound shared-capacity KV allocation by workload.

- [ ] **Step 3: Implement the mode-specific budget**

Read the frozen environment variable in `build_engine_config`, retain `0.84`
for strict-clean, and use `0.95` for shared-capacity so the whole-device
ceiling can accommodate model initialization despite pre-existing global
usage. In shared mode, explicitly set workload-sized KV capacity to `Q0=8`,
`Q1=16`, and `Q2=36` blocks so the engine cannot opportunistically consume
all remaining device memory. Export the plan admission mode into the remote
worker environment.

- [ ] **Step 4: Run both focused suites and verify GREEN**

Run:

```bash
python3 tools/test_tp4_decode_replay_worker.py
python3 tools/test_run_tp4_decode_replay.py
```

Expected: both suites pass.

### Task 4: Force the Diagnostic Claim Boundary

**Files:**
- Modify: `tools/test_run_tp4_decode_replay.py`
- Modify: `tools/run_tp4_decode_replay.py`

**Interfaces:**
- Consumes: producer and verifier receipts from the unchanged frozen gate
- Produces: a controller result whose top-level classification is always `DIAGNOSTIC_ONLY` in shared mode, with the measured gate classification preserved as `diagnostic_gate_classification`

- [ ] **Step 1: Write the failing result-classification test**

Run a successful fake shared attempt and assert:

```python
assert result["classification"] == "DIAGNOSTIC_ONLY"
assert result["diagnostic_gate_classification"] == "GO_STAGE1_JUSTIFIED"
assert result["claim_boundary"] == "shared_capacity_not_formal_performance_evidence"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 tools/test_run_tp4_decode_replay.py
```

Expected: failure because the raw producer classification is still returned.

- [ ] **Step 3: Implement result wrapping**

Preserve the producer/verifier agreement check. After agreement, wrap only the controller result for shared mode; do not rewrite the frozen bundle or verifier classification.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python3 tools/test_run_tp4_decode_replay.py
```

Expected: all controller tests pass.

### Task 5: Select and Guard Shared GPUs in the Local Supervisor

**Files:**
- Modify: `.agent_runtime/test_r34_supervisor.py`
- Modify: `.agent_runtime/r34_supervisor.py`

**Interfaces:**
- Consumes: four stable GPU rows and baseline `/proc` identities
- Produces: controller launch arguments with `--admission-mode shared_capacity`, plus fail-closed runtime monitoring

- [ ] **Step 1: Write failing supervisor tests**

Add tests for:

- selecting `0,1,5,7` from a representative inventory under the 20,480 MiB/5% thresholds;
- requiring the same GPU UUIDs and same baseline `(pid, start_time_ticks)` for four samples;
- rejecting a changed baseline PID identity or newly appearing foreign PID;
- accepting exact-tag owned PIDs after launch;
- emitting `--admission-mode shared_capacity`;
- reading `shared_capacity_admission.json`;
- cleaning only exact-tag owned processes after an unsafe change.

- [ ] **Step 2: Run the supervisor test and verify RED**

Run:

```bash
cd .agent_runtime
python3 -m unittest -v test_r34_supervisor.py
```

Expected: failure because shared-capacity selection and baseline guarding are absent.

- [ ] **Step 3: Implement shared selection and guard**

Replace the strict-only key with a stable shared-capacity candidate carrying GPU identity, memory, and baseline process identity. Keep the one-minute window. After launch, compare every sample with the frozen baseline and stop only exact-tag owned PIDs on unsafe drift.

- [ ] **Step 4: Run the supervisor test and verify GREEN**

Run:

```bash
cd .agent_runtime
python3 -m unittest -v test_r34_supervisor.py
```

Expected: all supervisor tests pass.

### Task 6: Verify, Document, Commit, Push, and Launch

**Files:**
- Modify: `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Inspect: all files listed in Tasks 1-5

**Interfaces:**
- Consumes: implementation, tests, and a fresh remote GPU snapshot
- Produces: committed/pushed source plus a uniquely tagged shared-capacity diagnostic attempt

- [ ] **Step 1: Run complete focused verification**

Run:

```bash
python3 tools/test_run_tp4_decode_replay.py
python3 tools/test_tp4_decode_replay_worker.py
cd .agent_runtime && python3 -m unittest -v test_r34_supervisor.py
python3 -m py_compile tools/run_tp4_decode_replay.py tools/tp4_decode_replay_worker.py .agent_runtime/r34_supervisor.py .agent_runtime/test_r34_supervisor.py
git diff --check
```

Expected: all tests and checks pass.

- [ ] **Step 2: Perform a prompt-to-artifact completion audit**

Verify every global constraint and task requirement against actual source, tests, generated admission/result schemas, Git status, and remote paths. Treat any uncovered requirement as incomplete.

- [ ] **Step 3: Update audit and handoff**

Record the mode thresholds, one-minute window, baseline process identity,
0.95 whole-device ceiling with workload-bounded KV blocks, safety stop
semantics, and the `DIAGNOSTIC_ONLY` claim boundary.

- [ ] **Step 4: Commit exact tracked paths**

Stage only:

```text
tools/run_tp4_decode_replay.py
tools/tp4_decode_replay_worker.py
tools/test_run_tp4_decode_replay.py
tools/test_tp4_decode_replay_worker.py
docs/superpowers/plans/2026-09-05-tp4-shared-capacity-diagnostic.md
docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md
AGENT_HANDOFF_STATE.md
```

Commit with exactly one trailer:

```text
Co-authored-by: TRAE CLI <noreply@bytedance.com>
```

- [ ] **Step 5: Push and verify SHA**

Push only to `origin/feat/kv-sparse-attention`, then verify local HEAD equals the remote branch SHA.

- [ ] **Step 6: Launch a fresh diagnostic tag**

Use a fresh `r39` or later tag, verify Kerberos TTL, source identity, attempt nonexistence, four stable shared-capacity GPUs, admission receipt, and controller-start event. Monitor through completion or produce a precise terminal blocker/partial-run reconciliation without touching foreign processes.
