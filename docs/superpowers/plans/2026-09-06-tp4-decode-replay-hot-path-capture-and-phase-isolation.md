# TP4 Decode Replay Hot-Path Capture and Phase Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. This
> repository task must be executed inline; do not create a worktree or
> dispatch subagents. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove one redundant model execution from exact
multi-sequence CUDA Graph capture and prevent warmup graph state from
contaminating measured TP4 decode-replay evidence.

**Architecture:** Add a complete phase reset to
`ExactCudaGraphCache`, expose it through rank-acknowledged
`ModelRunner`/`LLMEngine` methods, and invoke it at the benchmark phase
boundary. Keep lease-sealed graph identities and all frozen
qualification thresholds unchanged. The capture body will rely on the
successful eager decode that immediately precedes admission and will
execute the model only inside `torch.cuda.graph(...)`.

**Tech Stack:** Python 3.9+, PyTorch CUDA Graphs, torch.distributed,
dependency-light script tests, pytest-compatible assertions.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`.
- Do not use
  `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`, worktrees, or
  subagents.
- Use strict RED -> minimal implementation -> GREEN for every behavior.
- Stage exact paths only; never use `git add -A`, `git reset`, or
  `git clean`.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit must contain exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- Keep exact identity sealed by ordered
  `slot_id + generation + request_id`.
- Keep single capture at `2_000_000_000 ns`, total capture at
  `5_000_000_000 ns`, and replay coverage at `0.80`.
- Keep shared-capacity evidence `DIAGNOSTIC_ONLY`.
- Do not modify or supplement r48 artifacts.
- Put every fresh remote task file below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not run `kinit` or `krenew`.
- Do not terminate, adopt, or clean external GPU processes.

---

### Task 1: Add a complete exact graph cache phase reset

**Files:**

- Modify: `tinyvllm/engine/exact_cuda_graph_cache.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**

- Consumes:
  `ExactCudaGraphCache.release_ready_graphs(*, synchronize) -> int`
- Produces:
  `ExactCudaGraphCache.reset_phase(*, synchronize) -> dict`
- Receipt keys:
  `released_ready_entries`, `cleared_observations`,
  `cleared_rejections`, `summary`

- [ ] **Step 1: Write the failing complete-reset test**

Add a test that constructs ready, rejected, observed, capturing-free,
counter, byte, and capture-time state:

```python
def test_exact_cache_phase_reset_releases_graphs_and_clears_accounting():
    cache_module = load_exact_cache()
    cache = cache_module.ExactCudaGraphCache(make_cache_config())
    identity = make_identity(batch=4, width=2, splits=2)
    rejected = make_identity(batch=2, width=1, splits=2)
    calls = []

    class Graph:
        def reset(self):
            calls.append("reset")

    for _ in range(3):
        cache.observe_success(identity, estimated_static_bytes=4096)
    entry = make_entry(identity)
    entry.graph = Graph()
    cache.commit_capture(entry)
    cache.reject(
        rejected,
        "capture_failed",
        retained_reserved_bytes=2048,
    )
    cache.counters["hits"] = 7

    receipt = cache.reset_phase(
        synchronize=lambda: calls.append("synchronize"),
    )

    assert calls == ["reset", "synchronize"]
    assert receipt["released_ready_entries"] == 1
    assert receipt["cleared_observations"] == 1
    assert receipt["cleared_rejections"] == 1
    assert receipt["summary"] == {
        "ready_entries": [],
        "rejected": {},
        "capturing": [],
        "observation_counts": {},
        "static_bytes": 0,
        "reserved_delta_bytes": 0,
        "total_capture_ns": 0,
        "hits": 0,
        "misses": 0,
        "capture_attempts": 0,
        "capture_successes": 0,
        "capture_failures": 0,
    }
```

- [ ] **Step 2: Write the failing active-capture atomicity test**

```python
def test_exact_cache_phase_reset_rejects_active_capture_without_mutation():
    cache_module = load_exact_cache()
    cache = cache_module.ExactCudaGraphCache(make_cache_config())
    identity = make_identity(batch=4, width=2, splits=2)
    for _ in range(3):
        decision = cache.observe_success(
            identity,
            estimated_static_bytes=4096,
        )
    assert decision.should_capture is True
    before = cache.summary()

    with pytest.raises(RuntimeError, match="capture is active"):
        cache.reset_phase(synchronize=lambda: None)

    assert cache.summary() == before
```

- [ ] **Step 3: Run RED**

Run:

```bash
python -m pytest -q \
  tools/test_multi_sequence_cuda_graph_gate.py \
  -k 'phase_reset'
```

Expected: both tests fail because `reset_phase` does not exist.

- [ ] **Step 4: Implement the minimal cache reset**

Add:

```python
def reset_phase(self, *, synchronize) -> dict:
    if self.capturing:
        raise RuntimeError(
            "exact CUDA Graph cache reset rejected while capture is active"
        )
    cleared_observations = len(self.observation_counts)
    cleared_rejections = len(self.rejected)
    released_ready_entries = self.release_ready_graphs(
        synchronize=synchronize,
    )
    self.observation_counts.clear()
    self.rejected.clear()
    self.capturing.clear()
    self.static_bytes = 0
    self.reserved_delta_bytes = 0
    self.total_capture_ns = 0
    self.counters.clear()
    return {
        "released_ready_entries": released_ready_entries,
        "cleared_observations": cleared_observations,
        "cleared_rejections": cleared_rejections,
        "summary": self.summary(),
    }
```

- [ ] **Step 5: Run GREEN and adjacent cache tests**

Run:

```bash
python -m pytest -q tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(tp4): isolate exact graph cache phases" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 2: Remove the duplicate capture-path model execution

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Preserves:
  `ModelRunner._capture_exact_multi_sequence_graph(...) ->
  ExactCudaGraphEntry`
- Produces:
  `ModelRunner.reset_exact_cuda_graph_cache() -> dict`
- Capture receipt phases:
  `entered_capture`, `hot_path_eager_prerequisite`, `capture_begin`,
  `capture_body_completed`, `capture_end_synchronize_completed`,
  `scratch_restore_completed`

- [ ] **Step 1: Write RED assertions for one capture execution**

In the forward-protocol capture fixture, add a call counter to
`MutatingModel.__call__()` and assert:

```python
assert observed["model_calls"] == 1
assert observed["force_attention_backend"] == [True]
```

Update the expected receipt phases:

```python
assert phases == [
    "entered_capture",
    "hot_path_eager_prerequisite",
    "capture_begin",
    "capture_body_completed",
    "capture_end_synchronize_completed",
    "scratch_restore_completed",
]
```

In the lease-transaction fixture, count
`run_exact_cuda_graph_step()` calls and assert:

```python
assert runner.model.capture_step_calls == 1
```

- [ ] **Step 2: Write RED for ModelRunner reset**

Create a ready entry and stale accounting, invoke
`runner.reset_exact_cuda_graph_cache()`, and assert:

```python
assert receipt["rank"] == runner.rank
assert receipt["released_ready_entries"] == 1
assert receipt["summary"]["ready_entries"] == []
assert receipt["summary"]["rejected"] == {}
assert receipt["summary"]["observation_counts"] == {}
assert receipt["summary"]["total_capture_ns"] == 0
assert runner._exact_cuda_graph_pool is None
```

- [ ] **Step 3: Run RED**

Run:

```bash
python -m pytest -q tools/test_model_runner_spec_verify.py \
  -k 'capture_without_legacy_pool or transactional_capture or reset_exact'
```

Expected: old capture tests report two executions and the reset method
is missing.

- [ ] **Step 4: Implement hot-path capture**

Inside `_capture_exact_multi_sequence_graph()`:

- record `hot_path_eager_prerequisite` after state/KV snapshots;
- delete the uncaptured forward/transactional step;
- delete `warmup_forward_completed`;
- delete the pre-capture `torch.cuda.synchronize()`;
- delete `warmup_synchronize_completed`;
- preserve capture body, post-capture synchronization, all rollback,
  identity validation, TP MAX duration, and budget commit.

- [ ] **Step 5: Implement ModelRunner reset**

Add:

```python
def reset_exact_cuda_graph_cache(self):
    receipt = self.exact_cuda_graph_cache.reset_phase(
        synchronize=torch.cuda.synchronize,
    )
    self._exact_cuda_graph_pool = None
    return {"rank": self.rank, **receipt}
```

- [ ] **Step 6: Run GREEN and full model-runner test**

Run:

```bash
python -m pytest -q tools/test_model_runner_spec_verify.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 2**

```bash
git add \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "perf(tp4): capture exact graphs on hot path" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 3: Add all-rank reset acknowledgement and worker ordering

**Files:**

- Modify: `tinyvllm/engine/llm_engine.py`
- Create: `tools/test_exact_cuda_graph_phase_reset_wiring.py`
- Modify: `tools/tp4_decode_replay_worker.py`
- Modify: `tools/test_tp4_decode_replay_worker.py`

**Interfaces:**

- Produces:
  `LLMEngine.reset_exact_cuda_graph_cache(*, timeout_s) -> tuple[dict, ...]`
- Consumes:
  rank receipts from `ModelRunner.reset_exact_cuda_graph_cache()`

- [ ] **Step 1: Write RED for acknowledged engine reset**

Use the existing AST method-extraction pattern to load only
`LLMEngine.reset_exact_cuda_graph_cache`. Build a fake engine whose
`call_model_runner_acknowledged()` returns rank receipts.

Test:

```python
def test_engine_returns_ordered_agreeing_graph_reset_receipts():
    rows = tuple(_receipt(rank) for rank in range(4))
    engine = _Engine(rows)
    result = _reset(engine)
    assert result == rows
    assert engine.calls == [
        ("reset_exact_cuda_graph_cache", (), 5.0),
    ]
```

Add fail-closed cases for:

- missing rank;
- embedded rank mismatch;
- non-rank receipt disagreement;
- non-empty post-reset state;
- non-zero post-reset accounting.

- [ ] **Step 2: Run engine RED**

Run:

```bash
python -m pytest -q tools/test_exact_cuda_graph_phase_reset_wiring.py
```

Expected: method extraction fails because the LLMEngine method is
missing.

- [ ] **Step 3: Implement acknowledged reset validation**

Add `LLMEngine.reset_exact_cuda_graph_cache()` beside the other
acknowledged lifecycle methods. Dispatch:

```python
self.call_model_runner_acknowledged(
    "reset_exact_cuda_graph_cache",
    timeout_s=timeout_s,
)
```

Normalize receipts by rank, require complete inventory, require
identical non-rank content, validate the zero post-reset summary, and
return receipts ordered by rank.

- [ ] **Step 4: Write worker RED for fixed phase order**

Extend `_FakeEngine`:

```python
self.phase_boundary_calls = []

def clear_reusable_prefix_cache(self):
    self.phase_boundary_calls.append("clear_prefix")
    return 4

def reset_exact_cuda_graph_cache(self, *, timeout_s):
    assert timeout_s > 0
    self.phase_boundary_calls.append("reset_graph_cache")
    return tuple(_reset_receipt(rank) for rank in range(4))

def reset_decode_internal_profile(self, *, timeout_s):
    self.phase_boundary_calls.append("reset_profile")
    ...

def reset_peak_memory_stats(self, *, timeout_s):
    self.phase_boundary_calls.append("reset_peak")
    ...
```

Assert:

```python
assert engines[0].phase_boundary_calls == [
    "clear_prefix",
    "reset_graph_cache",
    "reset_profile",
    "reset_peak",
]
```

- [ ] **Step 5: Run worker RED**

Run:

```bash
python -m pytest -q tools/test_tp4_decode_replay_worker.py \
  -k run_arm_emits_complete_measured_evidence
```

Expected: phase order lacks `reset_graph_cache`.

- [ ] **Step 6: Implement worker reset call**

Insert:

```python
engine.reset_exact_cuda_graph_cache(timeout_s=float(timeout_s))
```

between prefix-cache clearing and profile reset.

- [ ] **Step 7: Run Task 3 GREEN**

Run:

```bash
python -m pytest -q \
  tools/test_exact_cuda_graph_phase_reset_wiring.py \
  tools/test_tp4_decode_replay_worker.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add \
  tinyvllm/engine/llm_engine.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py \
  tools/tp4_decode_replay_worker.py \
  tools/test_tp4_decode_replay_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(tp4): reset graph state before measurement" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 4: Wire fresh-run capture receipts into approved remote storage

**Files:**

- Modify: `tools/run_tp4_decode_replay.py`
- Modify: `tools/test_run_tp4_decode_replay.py`

**Interfaces:**

- Produces a per-case remote receipt root below:
  `<remote-run-root>/attempts/<attempt>/capture-receipts/<case-id>/`
- Exports:
  `TINYVLLM_EXACT_GRAPH_CAPTURE_RECEIPT_ROOT=<case-receipt-root>`

- [ ] **Step 1: Locate the exact worker-command environment builder**

Use:

```bash
rg -n \
  'tp4_decode_replay_worker|PYTHONDONTWRITEBYTECODE|environment|env ' \
  tools/run_tp4_decode_replay.py tools/test_run_tp4_decode_replay.py
```

Record the exact function and existing storage validation used for
remote worker commands. Do not introduce a second command builder.

- [ ] **Step 2: Write RED for receipt-root containment and export**

Extend the existing command-generation test to require:

```python
assert (
    environment["TINYVLLM_EXACT_GRAPH_CAPTURE_RECEIPT_ROOT"]
    == expected_case_receipt_root
)
assert expected_case_receipt_root.startswith(
    runner.REMOTE_ROOT + "/"
)
```

Add a negative test showing a receipt root outside `REMOTE_ROOT` is
rejected before SSH launch.

- [ ] **Step 3: Run RED**

Run:

```bash
python -m pytest -q tools/test_run_tp4_decode_replay.py \
  -k 'receipt or worker_command'
```

Expected: receipt environment is absent.

- [ ] **Step 4: Implement receipt-root wiring**

Derive the root from validated run/attempt/case identifiers and export
it only through the existing worker command environment. Create no
local Mac artifact mirror.

- [ ] **Step 5: Run GREEN**

Run:

```bash
python -m pytest -q tools/test_run_tp4_decode_replay.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  tools/run_tp4_decode_replay.py \
  tools/test_run_tp4_decode_replay.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): retain graph capture phase receipts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 5: Local regression, review, push, and remote smoke

**Files:**

- Review only the files named in Tasks 1-4.
- Create fresh remote artifacts only below the approved remote root.
- Do not modify r48.

- [ ] **Step 1: Run focused suite**

```bash
python -m pytest -q \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_run_tp4_decode_replay.py
```

Expected: all pass.

- [ ] **Step 2: Run adjacent TP4 suite**

Run the bounded runtime and TP4 evidence suite:

```bash
python -m pytest -q \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py \
  tools/test_tp4_decode_replay_contract.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_run_tp4_decode_replay.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py
```

Expected: no failures.

- [ ] **Step 3: Run static verification**

```bash
python -m py_compile \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/tp4_decode_replay_worker.py \
  tools/run_tp4_decode_replay.py
git diff --check
```

Expected: both commands succeed.

- [ ] **Step 4: Review exact diff and commit metadata**

```bash
git status --short -- \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py \
  tools/tp4_decode_replay_worker.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/run_tp4_decode_replay.py \
  tools/test_run_tp4_decode_replay.py
git log -5 --format='%H%n%B%n---'
```

Expected: only planned paths are changed; every new commit has exactly
one required trailer.

- [ ] **Step 5: Push and verify SHA**

```bash
git push origin feat/kv-sparse-attention
git rev-parse HEAD
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: local and remote SHAs match.

- [ ] **Step 6: Refresh external prerequisites without mutating them**

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist
```

Then use the existing SSH/GPU inventory functions. Do not run
`kinit`, lower TTL admission, or kill foreign tasks.

- [ ] **Step 7: Launch one fresh graph smoke**

Use a new tag derived from the final source revision. Select one Q1
graph case that exercises lease-transaction capture. Require a
currently admissible four-GPU cohort under the existing policy and
enable per-rank capture receipts under the remote attempt root.

Expected terminal evidence:

- exact output match;
- cleanup `CLEAN`;
- four rank receipts with the new phase sequence;
- measured capture-cost rows for all four ranks;
- maximum single capture `<= 2_000_000_000 ns`;
- total measured capture `<= 5_000_000_000 ns`.

- [ ] **Step 8: Apply the frozen decision**

If smoke correctness, lifecycle, and both capture budgets pass, launch
the complete fresh 30-case/15-pair gate and finish dual verification,
manifest, audit, exact-path commit, push, and remote SHA verification.

If any frozen gate fails, preserve the terminal artifact and classify
it without threshold changes. In that case, the next design target is
the Stage-1 dynamic pool-index graph protocol, not identity-field
deletion.
