# TP4 Worker Schedstat Tail Diagnostic Implementation Plan

> **Execution mode:** Implement directly in the authoritative checkout with
> `superpowers:executing-plans`. Do not create a worktree or dispatch
> subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an external, verified per-rank `/proc` sampler that can falsify the TP4 host-scheduling explanation for the second spec-verify tail.

**Architecture:** A dependency-light sampler parses rank-bound Linux process files and emits canonical JSONL at 10 ms cadence. The command-timeline runner starts it once the four owned GPU process bindings are known, validates identity and campaign coverage after the worker exits, and preserves the raw trace in the immutable run manifest without changing production inference code or performance metrics.

**Tech Stack:** Python standard library, Linux `/proc`, pytest, existing command-timeline remote runner.

## Global Constraints

- Modify only `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not modify `tinyvllm/engine/model_runner.py` or any production measured path.
- Do not add CUDA synchronization, `.item()`, worker logging, acknowledgements, fences, or GC control.
- Keep every remote artifact below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not write remote task output beneath `/`, `/tmp`, or `/private/tmp`.
- Do not modify `/data00/home/sitian/tllm/TinyLLMForge`.
- Use strict RED, then minimal GREEN.
- Stage exact paths only; never use `git add -A`.
- Commit with hooks disabled and exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only `origin/feat/kv-sparse-attention`.

---

### Task 1: Dependency-light process sampler

**Files:**
- Create: `tools/autoregressive_draft_process_sampler.py`
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: JSON rank bindings with `rank`, `gpu_uuid`, `pid`, and `starttime_ticks`.
- Produces: `parse_process_sample(...) -> dict`, `collect_process_samples(...) -> list[dict]`, and a CLI that emits canonical JSONL until signaled.

- [ ] **Step 1: Write failing parser and identity tests**

Add tests that load the new module and assert:

```python
sample = module.parse_process_sample(
    binding={
        "rank": 0,
        "gpu_uuid": "GPU-0",
        "pid": 101,
        "starttime_ticks": 9001,
    },
    schedstat_text="100 200 3\n",
    stat_text=build_proc_stat(
        pid=101,
        state="S",
        utime_ticks=11,
        stime_ticks=7,
        threads=4,
        starttime_ticks=9001,
        processor=12,
        delayacct_blkio_ticks=5,
    ),
    status_text=(
        "voluntary_ctxt_switches:\t8\n"
        "nonvoluntary_ctxt_switches:\t2\n"
    ),
    wchan_text="futex_wait_queue\n",
    unix_ns=123,
    monotonic_ns=456,
)
assert sample["run_time_ns"] == 100
assert sample["runqueue_wait_ns"] == 200
assert sample["state"] == "S"
assert sample["wchan"] == "futex_wait_queue"
```

Also assert duplicate ranks/UUIDs/PIDs and changed start-time ticks fail.

- [ ] **Step 2: Run the focused tests and preserve RED**

Run:

```bash
python3 -m pytest \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k process_sampler -vv
```

Expected: FAIL because `tools/autoregressive_draft_process_sampler.py` does
not exist.

- [ ] **Step 3: Implement the minimal sampler**

Implement:

```python
PROCESS_SAMPLE_SCHEMA_VERSION = 1
DEFAULT_INTERVAL_SECONDS = 0.01

def validate_bindings(bindings: object) -> tuple[dict, ...]: ...
def parse_schedstat(text: str) -> tuple[int, int, int]: ...
def parse_proc_stat(text: str) -> dict[str, int | str]: ...
def parse_proc_status(text: str) -> tuple[int, int]: ...
def parse_process_sample(... ) -> dict: ...
def collect_process_samples(... ) -> list[dict]: ...
def run_sampler(... ) -> int: ...
```

Use the final `)` in `/proc/<pid>/stat` to isolate a command name that may
contain spaces or parentheses. Validate every required integer as a
non-negative non-boolean integer. Emit one terminal `status="exited"` row
when a process disappears, and stop after all bindings are terminal.

- [ ] **Step 4: Run focused GREEN**

Run the Step 2 command.

Expected: all `process_sampler` tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add \
  tools/autoregressive_draft_process_sampler.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(telemetry): sample TP worker scheduling" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 2: Runner lifecycle and coverage verification

**Files:**
- Modify: `tools/run_autoregressive_draft_command_timeline_remote.py`
- Modify: `tools/test_autoregressive_draft_cuda_graph_gate.py`

**Interfaces:**
- Consumes: the sampler CLI from Task 1 and the first complete mapping returned by `validate_owned_gpu_processes(...)`.
- Produces: one owned sampler process plus `<mode>.process.jsonl`, stderr, and strict `_verify_process_telemetry(...)` coverage evidence.

- [ ] **Step 1: Write failing lifecycle tests**

Add tests that require:

```python
started = []

result = runner._monitor_owned_worker(
    fake_worker,
    process_group_id=500,
    gpu_uuids=["GPU-0", "GPU-1", "GPU-2", "GPU-3"],
    on_binding=lambda binding, owned: started.append(
        (dict(binding), set(owned))
    ),
    ...
)
assert len(started) == 1
```

Add a coverage test with four ranks and one row per campaign interval. Assert
missing rank coverage, PID drift, decreasing counters, or non-empty stderr
raises `ValueError`.

- [ ] **Step 2: Run focused RED**

Run:

```bash
python3 -m pytest \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k "process_telemetry or monitor_owned_worker_starts_process_sampler" -vv
```

Expected: FAIL because `on_binding` and process telemetry verification do not
exist.

- [ ] **Step 3: Implement minimal runner integration**

Add the new tool to `SOURCE_PATHS`. Extend `_monitor_owned_worker` with an
optional `on_binding` callback invoked exactly once after the first complete
owned TP4 binding. Add `_start_process_sampler(...)` that builds bindings in
`gpu_uuids` order and records `/proc/<pid>/stat` start-time ticks before
launching:

```python
[
    REMOTE_PYTHON,
    f"{source_root}/tools/autoregressive_draft_process_sampler.py",
    "--interval-seconds",
    "0.01",
    "--bindings-json",
    json.dumps(bindings, sort_keys=True),
]
```

In `_remote_epoch`, create process telemetry/stdout handles beneath the epoch
telemetry directory, register the callback before monitoring, terminate and
reap the process sampler before finalization, and call
`_verify_process_telemetry(...)` against warmup plus five measured campaign
intervals.

- [ ] **Step 4: Run focused GREEN and affected runner tests**

Run:

```bash
python3 -m pytest \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  -k "process_sampler or process_telemetry or epoch_samplers or monitor_owned_worker" -vv
```

Then run:

```bash
python3 -m pytest \
  tools/test_autoregressive_draft_cuda_graph_gate.py -q
```

Expected: both commands PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(telemetry): bind TP schedstat traces" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 3: Local closure, push, and fresh diagnostic campaign

**Files:**
- Modify after evidence: `AGENT_HANDOFF_STATE.md`
- Modify after evidence: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

**Interfaces:**
- Consumes: committed runner and sampler from Tasks 1--2.
- Produces: one immutable four-GPU diagnostic artifact, a slow/fast per-rank scheduling matrix, and an evidence-bounded next decision.

- [ ] **Step 1: Run complete local verification**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-schedstat-pycache \
python3 -m py_compile \
  tools/autoregressive_draft_process_sampler.py \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
python3 -m tabnanny \
  tools/autoregressive_draft_process_sampler.py \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
python3 -m pytest \
  tools/test_autoregressive_draft_cuda_graph_gate.py \
  tools/test_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_verify_autoregressive_draft_command_timeline_diagnostic.py \
  tools/test_autoregressive_draft_source_pair_gate.py -q
git diff --check -- \
  tools/autoregressive_draft_process_sampler.py \
  tools/run_autoregressive_draft_command_timeline_remote.py \
  tools/test_autoregressive_draft_cuda_graph_gate.py
```

Use a repository-local or task-specific cache path if `/tmp` is disallowed by
the active environment; do not write remote artifacts there.

- [ ] **Step 2: Push exact committed head**

Verify `HEAD == origin/feat/kv-sparse-attention` after:

```bash
git push origin feat/kv-sparse-attention
```

- [ ] **Step 3: Run fresh four-clean-GPU diagnostic**

Use a new immutable tag and the existing command-timeline remote runner.
Require four GPUs with memory used `<=1024 MiB`, utilization `<=5%`, and no
compute processes. Keep all output below the configured remote root.

- [ ] **Step 4: Build the falsification matrix**

For each step-4 verify command, record:

```text
epoch, repeat, verify ordinal, batch size, query length,
selected-sequence hash, transaction digest,
per-rank method wall and CUDA duration,
per-rank CPU runtime delta, runqueue-wait delta,
state/wchan sequence, voluntary/involuntary context-switch delta
```

Classify the hypothesis as confirmed only when a rank-local scheduling or
off-CPU delta overlaps a slow command and is absent in identical fast calls.

- [ ] **Step 5: Update canonical audit and handoff**

Append the immutable tag, hashes, verifier status, matrix, hypothesis result,
and exact claim boundary. Do not claim a performance improvement from a
diagnostic-only run.

- [ ] **Step 6: Commit and push documentation**

Stage exactly the two canonical documentation files, commit with one required
co-author trailer, push `origin/feat/kv-sparse-attention`, and verify local,
tracking, and live remote heads match.
