# Cross-Request Wavefront Overlap Stage-0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a source-bound four-GPU real-shape microgate that
determines whether two request cohorts can hide TP4 NCCL AllReduce time behind
independent GEMM and dependent pointwise work.

**Architecture:** Keep Stage 0 outside the model runtime. A dependency-light
contract module freezes cohort construction, evidence schema, interval math,
and fail-closed classification. A real PyTorch/NCCL worker executes the
baseline and two-cohort candidate with preallocated streams, events, and
buffers; an assembler, independent verifier, and remote controller produce a
compact immutable bundle. Model integration is forbidden unless the terminal
classification is `GO_WAVEFRONT_MICROGATE`.

**Tech Stack:** Python 3.12/3.11, PyTorch distributed with NCCL, CUDA streams
and events, pytest, JSON/JSONL, SHA-256 manifests, SSH, four A100 GPUs.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`; the Desktop path is a
  symlink to this checkout.
- Stay on `feat/kv-sparse-attention` and push only to
  `origin/feat/kv-sparse-attention`.
- Do not create a worktree or use subagents; execute inline.
- Use meaningful RED, minimal implementation, and GREEN for every code task.
- Stage exact paths only. Never use broad `git add`, `git reset`, `git clean`,
  or unrelated formatting.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Stage 0 must not modify `tinyvllm/layers/linear.py`, Qwen model files, the
  scheduler, or the production collective path.
- Stage 0 supports TP4, active-token groups four and eight, BF16 inputs,
  FP32 local accumulation and collective, and BF16 residual output.
- Use two warmup pairs and at least 300 alternating measured pairs per shape.
- Require at least `5%` median improvement for active tokens four and `8%`
  for active tokens eight.
- Require realized overlap of at least `20%` of the candidate communication
  interval for both shapes.
- Allow at most `3%` P99 regression, `10%` host-submission regression, and
  `128 MiB` additional peak allocated memory per rank.
- If both shape medians improve by less than `3%`, stop the complete
  direction.
- Put every remote task file, cache, log, and temporary artifact below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not write remote task data to `/`, `/tmp`, a retired checkout, or a
  model-cache directory.
- Do not run `kinit` or `krenew`.
- Require four strict-clean GPUs twice, including immediately before launch.
- Do not terminate, adopt, or clean external GPU processes.
- Keep large traces remote and download only compact evidence.
- Report benefit and cost together. Stage 0 cannot establish model E2E
  speedup.

---

### Task 1: Freeze the model-neutral cohort and classifier contract

**Files:**

- Create: `tools/cross_request_wavefront_overlap.py`
- Create: `tools/test_cross_request_wavefront_overlap.py`

**Interfaces:**

- Produces:
  `build_balanced_cohorts(active_request_count: int) -> tuple[dict, dict]`.
- Produces:
  `cohort_digest(cohorts: Sequence[Mapping]) -> str`.
- Produces:
  `interval_union_ns(intervals) -> int`.
- Produces:
  `interval_overlap_ns(left, right) -> int`.
- Produces:
  `classify_wavefront_microgate(rows, memory, cleanup) -> dict`.
- Contains no `torch` import and no model name.

- [ ] **Step 1: Write failing cohort and interval tests**

Create tests with these exact expectations:

```python
def test_balanced_cohorts_are_contiguous_complete_and_stable():
    assert build_balanced_cohorts(4) == (
        {
            "cohort_id": 0,
            "request_indices": (0, 1),
            "active_token_count": 2,
        },
        {
            "cohort_id": 1,
            "request_indices": (2, 3),
            "active_token_count": 2,
        },
    )
    assert build_balanced_cohorts(8)[0]["request_indices"] == (0, 1, 2, 3)
    assert build_balanced_cohorts(8)[1]["request_indices"] == (4, 5, 6, 7)
    assert cohort_digest(build_balanced_cohorts(8)) == cohort_digest(
        build_balanced_cohorts(8)
    )


@pytest.mark.parametrize("count", (0, 1, 2, 3, 5, True))
def test_balanced_cohorts_reject_unsupported_counts(count):
    with pytest.raises(ValueError, match="active request count"):
        build_balanced_cohorts(count)


def test_interval_math_uses_unions_before_overlap():
    communication = ((10, 30), (25, 40))
    computation = ((0, 15), (20, 35))

    assert interval_union_ns(communication) == 30
    assert interval_overlap_ns(communication, computation) == 20
```

- [ ] **Step 2: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_request_wavefront_overlap.py -q
```

Expected: collection fails because
`tools.cross_request_wavefront_overlap` does not exist.

- [ ] **Step 3: Implement the cohort and interval primitives**

Use immutable constants and canonical JSON:

```python
WORLD_SIZE = 4
ACTIVE_TOKEN_GROUPS = (4, 8)
WARMUP_PAIR_COUNT = 2
MEASURED_PAIR_COUNT = 300
MAX_ALLOCATED_DELTA_BYTES = 128 * 1024 * 1024
MIN_MEDIAN_SPEEDUP = {4: 0.05, 8: 0.08}
STOP_MEDIAN_SPEEDUP = 0.03
MIN_OVERLAP_RATIO = 0.20
MAX_P99_REGRESSION = 0.03
MAX_HOST_SUBMISSION_REGRESSION = 0.10
CROSS_RANK_ATOL = 2e-4
CROSS_RANK_RTOL = 2e-4
BASELINE_ATOL = 2e-2
BASELINE_RTOL = 2e-3


def build_balanced_cohorts(active_request_count):
    if (
        type(active_request_count) is not int
        or active_request_count not in ACTIVE_TOKEN_GROUPS
    ):
        raise ValueError("active request count must be 4 or 8")
    split = (active_request_count + 1) // 2
    return (
        {
            "cohort_id": 0,
            "request_indices": tuple(range(split)),
            "active_token_count": split,
        },
        {
            "cohort_id": 1,
            "request_indices": tuple(
                range(split, active_request_count)
            ),
            "active_token_count": active_request_count - split,
        },
    )


def cohort_digest(cohorts):
    encoded = json.dumps(
        list(cohorts),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

Normalize intervals by sorting and merging overlapping ranges before
computing a union or intersection. Reject booleans, negative endpoints,
reversed ranges, non-finite values, and empty interval sets.

- [ ] **Step 4: Write failing classifier tests**

Construct 300 complete four-rank pairs for both shapes and cover:

```python
def test_classifier_accepts_complete_profitable_gate():
    result = classify_wavefront_microgate(
        _passing_rows(),
        memory={"maximum_allocated_delta_bytes": 64 * 1024 * 1024},
        cleanup={"classification": "CLEAN"},
    )

    assert result["classification"] == "GO_WAVEFRONT_MICROGATE"
    assert result["runtime_integration_authorized"] is True
    assert [row["active_tokens"] for row in result["shape_summaries"]] == [
        4,
        8,
    ]


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("coverage", "INCONCLUSIVE_EVIDENCE"),
        ("rank_digest", "INCONCLUSIVE_EVIDENCE"),
        ("correctness", "NO_GO_CORRECTNESS"),
        ("memory", "NO_GO_MEMORY"),
        ("tail", "NO_GO_PERFORMANCE"),
        ("overlap", "NO_GO_INSUFFICIENT_OVERLAP"),
        ("fragmentation", "NO_GO_GEMM_FRAGMENTATION"),
        ("cleanup", "INCONCLUSIVE_EVIDENCE"),
    ),
)
def test_classifier_fails_closed(mutation, expected):
    rows, memory, cleanup = _mutated_inputs(mutation)
    assert classify_wavefront_microgate(
        rows,
        memory,
        cleanup,
    )["classification"] == expected
```

Each row must include:

```python
{
    "active_tokens": 4,
    "pair_index": 17,
    "rank": 2,
    "arm_order": ["candidate", "baseline"],
    "cohort_digest": "a" * 64,
    "collective_order_digest": "b" * 64,
    "baseline_cuda_ns": 100_000,
    "candidate_cuda_ns": 90_000,
    "baseline_host_submission_ns": 20_000,
    "candidate_host_submission_ns": 21_000,
    "candidate_communication_union_ns": 40_000,
    "candidate_realized_overlap_ns": 12_000,
    "cross_rank_max_abs_error": 0.0,
    "cross_rank_max_rel_error": 0.0,
    "baseline_max_abs_error": 0.0,
    "baseline_max_rel_error": 0.0,
    "nan_count": 0,
    "inf_count": 0,
    "timed_out": False,
}
```

- [ ] **Step 5: Implement fail-closed classification**

Classification precedence is:

```text
NO_GO_CORRECTNESS
NO_GO_MEMORY
INCONCLUSIVE_EVIDENCE
NO_GO_INSUFFICIENT_OVERLAP
NO_GO_GEMM_FRAGMENTATION
NO_GO_PERFORMANCE
GO_WAVEFRONT_MICROGATE
```

Use the maximum four-rank duration per pair. Use nearest-rank P99. Require
exactly 300 complete pair identities for each active-token group, one stable
cohort digest, one stable collective-order digest, zero timeout, zero
NaN/Inf, and a clean teardown.

Classify `NO_GO_GEMM_FRAGMENTATION` when both shape median speedups are below
`3%`. Otherwise classify insufficient overlap before the general performance
gate.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_request_wavefront_overlap.py -q
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/cross_request_wavefront_overlap.py
git diff --check -- \
  tools/cross_request_wavefront_overlap.py \
  tools/test_cross_request_wavefront_overlap.py
```

Expected: all tests pass and syntax/diff checks are clean.

Commit:

```bash
git add -- \
  tools/cross_request_wavefront_overlap.py \
  tools/test_cross_request_wavefront_overlap.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): define TP4 wavefront microgate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Implement the real four-rank stream/event worker

**Files:**

- Create: `tools/cross_request_wavefront_microgate_worker.py`
- Create: `tools/test_cross_request_wavefront_microgate_worker.py`

**Interfaces:**

- Produces `build_workload_schedule()`.
- Produces `validate_measurement_row(row)`.
- Produces `WavefrontBuffers.create(...)`.
- Produces `run_worker(args)`.
- Writes rank-local rows that rank zero consolidates into:
  `microgate_rows.jsonl`, `memory_summary.json`, `cleanup.json`, and
  `runtime_capabilities.json`.

- [ ] **Step 1: Write RED tests for schedule, schema, and timed-path safety**

Use:

```python
def test_schedule_freezes_two_shapes_warmups_pairs_and_abba_order():
    schedule = build_workload_schedule()

    assert [row["active_tokens"] for row in schedule] == [4, 8]
    assert all(len(row["warmups"]) == 2 for row in schedule)
    assert all(len(row["measurements"]) == 300 for row in schedule)
    assert schedule[0]["measurements"][0]["arm_order"] == (
        "baseline",
        "candidate",
    )
    assert schedule[0]["measurements"][1]["arm_order"] == (
        "candidate",
        "baseline",
    )


def test_candidate_timed_function_contains_no_device_sync_or_allocation():
    source = inspect.getsource(_run_candidate)

    assert "torch.cuda.synchronize" not in source
    assert "empty(" not in source
    assert "zeros(" not in source
    assert "Stream(" not in source
    assert "Event(" not in source


def test_measurement_row_requires_overlap_and_order_evidence():
    row = _valid_measurement_row()
    assert validate_measurement_row(row) == row
    for field in (
        "candidate_communication_union_ns",
        "candidate_realized_overlap_ns",
        "cohort_digest",
        "collective_order_digest",
    ):
        broken = dict(row)
        broken.pop(field)
        with pytest.raises(ValueError, match=field):
            validate_measurement_row(broken)
```

- [ ] **Step 2: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_request_wavefront_microgate_worker.py -q
```

Expected: collection fails because the worker module does not exist.

- [ ] **Step 3: Implement immutable workload and preallocated resources**

Use these dimensions:

```python
ACTIVE_TOKEN_GROUPS = (4, 8)
WORLD_SIZE = 4
LOCAL_INPUT_SIZE = 1536
HIDDEN_SIZE = 5120
WARMUP_PAIR_COUNT = 2
MEASURED_PAIR_COUNT = 300
```

`WavefrontBuffers.create(torch, device, active_tokens)` allocates before
timing:

```python
return WavefrontBuffers(
    compute_streams=(
        torch.cuda.Stream(device=device),
        torch.cuda.Stream(device=device),
    ),
    communication_stream=torch.cuda.Stream(device=device),
    origin=torch.cuda.Event(enable_timing=True),
    local_started=tuple(
        torch.cuda.Event(enable_timing=True) for _ in range(2)
    ),
    local_ready=tuple(
        torch.cuda.Event(enable_timing=True) for _ in range(2)
    ),
    communication_started=tuple(
        torch.cuda.Event(enable_timing=True) for _ in range(2)
    ),
    communication_ready=tuple(
        torch.cuda.Event(enable_timing=True) for _ in range(2)
    ),
    dependent_ready=tuple(
        torch.cuda.Event(enable_timing=True) for _ in range(2)
    ),
    completed=torch.cuda.Event(enable_timing=True),
    local_partials=torch.empty(
        (active_tokens, HIDDEN_SIZE),
        dtype=torch.float32,
        device=device,
    ),
    cast_buffer=torch.empty(
        (active_tokens, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    ),
    output=torch.empty(
        (active_tokens, HIDDEN_SIZE),
        dtype=torch.bfloat16,
        device=device,
    ),
)
```

Input FP32 conversion, transposed contiguous weight, cohort slices, baseline
buffers, candidate buffers, and residuals are also created outside timed
functions.

- [ ] **Step 4: Implement baseline and candidate launch order**

The candidate function must enqueue in this order on every rank:

```python
caller = torch.cuda.current_stream(device)
buffers.origin.record(caller)

with torch.cuda.stream(buffers.compute_streams[0]):
    buffers.compute_streams[0].wait_event(buffers.origin)
    buffers.local_started[0].record()
    torch.mm(x_fp32[cohort0], weight_t, out=partial0)
    buffers.local_ready[0].record()

with torch.cuda.stream(buffers.communication_stream):
    buffers.communication_stream.wait_event(buffers.local_ready[0])
    buffers.communication_started[0].record()
    work0 = dist.all_reduce(partial0, async_op=True)
    work0.wait()
    buffers.communication_ready[0].record()

with torch.cuda.stream(buffers.compute_streams[1]):
    buffers.compute_streams[1].wait_event(buffers.origin)
    buffers.local_started[1].record()
    torch.mm(x_fp32[cohort1], weight_t, out=partial1)
    buffers.local_ready[1].record()

with torch.cuda.stream(buffers.compute_streams[0]):
    buffers.compute_streams[0].wait_event(
        buffers.communication_ready[0]
    )
    cast0.copy_(partial0)
    output0.copy_(cast0).add_(residual0)
    buffers.dependent_ready[0].record()

with torch.cuda.stream(buffers.communication_stream):
    buffers.communication_stream.wait_event(buffers.local_ready[1])
    buffers.communication_started[1].record()
    work1 = dist.all_reduce(partial1, async_op=True)
    work1.wait()
    buffers.communication_ready[1].record()

with torch.cuda.stream(buffers.compute_streams[1]):
    buffers.compute_streams[1].wait_event(
        buffers.communication_ready[1]
    )
    cast1.copy_(partial1)
    output1.copy_(cast1).add_(residual1)
    buffers.dependent_ready[1].record()

caller.wait_event(buffers.dependent_ready[0])
caller.wait_event(buffers.dependent_ready[1])
buffers.completed.record(caller)
```

Record the fixed collective order digest from:

```python
("cohort:0", "cohort:1")
```

The baseline performs one full-batch `torch.mm`, one blocking FP32
`dist.all_reduce`, BF16 `copy_`, and in-place residual add.

- [ ] **Step 5: Add correctness, interval, memory, and cleanup evidence**

After each complete arm pair, one bounded synchronization is allowed:

```python
buffers.completed.synchronize()
```

Compute event positions relative to `origin`, create half-open nanosecond
intervals for cohort GEMMs, collectives, and dependent operations, then call
the Task-1 interval helpers.

Gather candidate outputs and digest strings across all ranks. Record maximum
cross-rank and candidate-versus-baseline errors. Rank zero writes atomic
compact artifacts only after all four rank rows are present.

Always destroy the process group in `finally`. Cleanup rows contain:

```python
{
    "rank": rank,
    "streams_released": True,
    "events_released": True,
    "timed_out": False,
    "process_group_destroyed": True,
}
```

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_cross_request_wavefront_overlap.py \
  tools/test_cross_request_wavefront_microgate_worker.py -q
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/cross_request_wavefront_microgate_worker.py
git diff --check -- \
  tools/cross_request_wavefront_microgate_worker.py \
  tools/test_cross_request_wavefront_microgate_worker.py
```

Commit:

```bash
git add -- \
  tools/cross_request_wavefront_microgate_worker.py \
  tools/test_cross_request_wavefront_microgate_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): add TP4 wavefront worker" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Add compact assembly and independent verification

**Files:**

- Create: `tools/assemble_cross_request_wavefront_microgate.py`
- Create: `tools/verify_cross_request_wavefront_microgate.py`
- Create: `tools/test_assemble_cross_request_wavefront_microgate.py`
- Create: `tools/test_verify_cross_request_wavefront_microgate.py`

**Interfaces:**

- Produces:
  `assemble_bundle(output_root, source_identity, runtime_capabilities,
  cohort_policy, rows, memory, cleanup)`.
- Produces: `verify_bundle(root)`.
- Writes a nine-file producer bundle plus independent verification and a
  rewritten SHA-256 manifest.

- [ ] **Step 1: Write assembler RED tests**

The passing fixture contains 2 shapes × 300 pairs × 4 ranks = 2,400 rows.
Assert:

```python
result = assemble_bundle(output_root=tmp_path, **_inputs())

assert result["classification"] == "GO_WAVEFRONT_MICROGATE"
assert json.loads(
    (tmp_path / "classification.json").read_text()
) == {
    "schema_version": "cross-request-wavefront-classification.v1",
    "classification": "GO_WAVEFRONT_MICROGATE",
    "runtime_integration_authorized": True,
}
```

Cover every classifier failure, duplicate JSON keys, non-finite numbers,
identity drift, nonempty output directories, and missing compact artifacts.

- [ ] **Step 2: Run assembler RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_assemble_cross_request_wavefront_microgate.py -q
```

Expected: assembler module import failure.

- [ ] **Step 3: Implement atomic producer assembly**

The producer files are:

```text
source_identity.json
runtime_capabilities.json
cohort_policy.json
microgate_rows.jsonl
memory_summary.json
cleanup.json
microgate_summary.json
classification.json
manifest.sha256
```

Reject any source revision that is not 40 lowercase hexadecimal characters,
any tree hash that is not 64 lowercase hexadecimal characters, any row whose
attempt/revision differs, and any output directory that is not empty.

Call `classify_wavefront_microgate()` from the contract module. Write all
files atomically with `allow_nan=False`, then hash every producer artifact
except the manifest.

- [ ] **Step 4: Write verifier RED tests**

Require:

```python
result = verify_bundle(tmp_path)

assert result["status"] == "PASS"
assert result["producer_classification"] == (
    "GO_WAVEFRONT_MICROGATE"
)
assert result["reconstructed_classification"] == (
    "GO_WAVEFRONT_MICROGATE"
)
assert result["measurement_row_count"] == 2400
assert result["artifact_hashes_verified"] is True
```

Also prove that the verifier does not import the assembler and rejects:

- a mutated artifact hash;
- producer-classification disagreement;
- summary disagreement;
- an extra file;
- a missing row; and
- rank digest disagreement.

- [ ] **Step 5: Implement independent reconstruction**

The verifier may import only
`tools.cross_request_wavefront_overlap`. It independently:

- validates the manifest inventory and hashes;
- reloads strict JSON and JSONL;
- checks source and attempt identity;
- reconstructs classification from raw rows;
- compares exact producer summary and classification;
- writes `independent_verification.json`; and
- rewrites the manifest to include verification.

- [ ] **Step 6: Run GREEN and commit**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_cross_request_wavefront_overlap.py \
  tools/test_assemble_cross_request_wavefront_microgate.py \
  tools/test_verify_cross_request_wavefront_microgate.py
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/assemble_cross_request_wavefront_microgate.py \
  tools/verify_cross_request_wavefront_microgate.py
git diff --check -- \
  tools/assemble_cross_request_wavefront_microgate.py \
  tools/verify_cross_request_wavefront_microgate.py \
  tools/test_assemble_cross_request_wavefront_microgate.py \
  tools/test_verify_cross_request_wavefront_microgate.py
```

Commit:

```bash
git add -- \
  tools/assemble_cross_request_wavefront_microgate.py \
  tools/verify_cross_request_wavefront_microgate.py \
  tools/test_assemble_cross_request_wavefront_microgate.py \
  tools/test_verify_cross_request_wavefront_microgate.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): verify TP4 wavefront evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Add the fail-closed remote controller

**Files:**

- Create: `tools/run_cross_request_wavefront_microgate.py`
- Create: `tools/test_run_cross_request_wavefront_microgate.py`

**Interfaces:**

- Produces `build_attempt_plan(...)`.
- Produces `build_remote_worker_commands(...)`.
- Produces `capture_source_identity(...)`.
- Produces `run_attempt(...)`.
- Reuses strict-clean GPU and Kerberos parsers from
  `tools/run_qwen38_tp4_communication_profile.py`.

- [ ] **Step 1: Write controller RED tests**

Use a fresh attempt example:

```text
20260831-cross-request-wavefront-stage0-r1
```

Require:

```python
def test_plan_keeps_every_remote_path_attempt_local():
    plan = _plan()
    attempt_root = PurePosixPath(plan["attempt_root"])

    assert all(
        PurePosixPath(path).is_relative_to(
            PurePosixPath(APPROVED_REMOTE_ROOT)
        )
        for path in _absolute_paths(plan)
    )
    for value in plan["environment"].values():
        assert PurePosixPath(value).is_relative_to(attempt_root)


def test_controller_checks_kerberos_then_gpu_twice():
    events = []
    result = run_attempt(
        _plan(),
        kerberos_probe=lambda: events.append("kerberos") or {
            "classification": "PASS",
        },
        gpu_probe=lambda: events.append("gpu") or _four_clean_gpus(),
        remote_writer=lambda plan: events.append("write") or {},
        worker_runner=lambda plan: events.append("worker") or {},
        assembler=lambda plan: events.append("assemble") or {
            "classification": "GO_WAVEFRONT_MICROGATE",
        },
        remote_verifier=lambda plan: events.append("remote_verify") or {
            "status": "PASS",
        },
        downloader=lambda plan: events.append("download") or {},
        local_verifier=lambda plan: events.append("local_verify") or {
            "status": "PASS",
        },
    )

    assert result["classification"] == "GO_WAVEFRONT_MICROGATE"
    assert events == [
        "kerberos",
        "gpu",
        "write",
        "gpu",
        "worker",
        "assemble",
        "remote_verify",
        "download",
        "local_verify",
    ]
```

Also reject:

- expired Kerberos before any remote access;
- an existing or symlinked attempt path;
- fewer than four strict-clean GPUs;
- paths outside the approved root;
- `kinit`, `krenew`, or signal commands;
- missing producer/verifier agreement;
- source archives with tracked `tinyvllm/` or `tools/` changes; and
- a remote Python outside `/data00/home/sitian`.

- [ ] **Step 2: Run RED**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest \
  tools/test_run_cross_request_wavefront_microgate.py -q
```

Expected: controller module import failure.

- [ ] **Step 3: Implement attempt planning and source identity**

Use:

```python
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
PLAN_SCHEMA = "cross-request-wavefront-plan.v1"
```

Attempt-local environment must include:

```python
{
    "TMPDIR": f"{runtime_root}/tmp",
    "XDG_CACHE_HOME": f"{runtime_root}/cache/xdg",
    "TORCH_EXTENSIONS_DIR": (
        f"{runtime_root}/cache/torch-extensions"
    ),
    "CUDA_CACHE_PATH": f"{runtime_root}/cache/cuda",
}
```

Create the tracked source archive from the exact committed revision. Bind the
source identity to the tracked `tinyvllm/` and `tools/` tree SHA-256.

- [ ] **Step 4: Implement orchestration and verification agreement**

Follow this exact sequence:

```text
Kerberos TTL fail-fast
strict-clean GPU admission
fresh remote path and non-symlink proof
atomic controller receipts and tracked source upload
second strict-clean GPU admission
one four-rank worker launch
producer assembly
remote independent verifier
compact bundle download
local independent verifier
producer/remote/local classification agreement
```

SSH retry only return code 255 within the fixed retry budget. Do not retry a
worker after launch.

- [ ] **Step 5: Run GREEN and commit**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_cross_request_wavefront_overlap.py \
  tools/test_cross_request_wavefront_microgate_worker.py \
  tools/test_assemble_cross_request_wavefront_microgate.py \
  tools/test_verify_cross_request_wavefront_microgate.py \
  tools/test_run_cross_request_wavefront_microgate.py
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/run_cross_request_wavefront_microgate.py
git diff --check -- \
  tools/run_cross_request_wavefront_microgate.py \
  tools/test_run_cross_request_wavefront_microgate.py
```

Commit:

```bash
git add -- \
  tools/run_cross_request_wavefront_microgate.py \
  tools/test_run_cross_request_wavefront_microgate.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): orchestrate TP4 wavefront gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Verify the complete committed Stage-0 source

**Files:**

- Verify: `tools/cross_request_wavefront_overlap.py`
- Verify: `tools/cross_request_wavefront_microgate_worker.py`
- Verify: `tools/assemble_cross_request_wavefront_microgate.py`
- Verify: `tools/verify_cross_request_wavefront_microgate.py`
- Verify: `tools/run_cross_request_wavefront_microgate.py`
- Verify: the five matching `tools/test_*wavefront*` files.

**Interfaces:**

- Produces one clean committed source revision suitable for the remote
  source-identity guard.

- [ ] **Step 1: Run the complete Stage-0 suite**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_cross_request_wavefront_overlap.py \
  tools/test_cross_request_wavefront_microgate_worker.py \
  tools/test_assemble_cross_request_wavefront_microgate.py \
  tools/test_verify_cross_request_wavefront_microgate.py \
  tools/test_run_cross_request_wavefront_microgate.py
```

- [ ] **Step 2: Run adjacent distributed/controller regressions**

Run:

```bash
/opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_qwen38_tp4_peer_reduction.py \
  tools/test_qwen38_tp4_peer_reduction_microgate_worker.py \
  tools/test_assemble_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_verify_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_run_qwen38_tp4_peer_reduction_microgate.py \
  tools/test_qwen38_collective_reduction.py
```

- [ ] **Step 3: Run syntax, whitespace, and source guards**

Run:

```bash
/opt/homebrew/bin/python3.12 -m py_compile \
  tools/cross_request_wavefront_overlap.py \
  tools/cross_request_wavefront_microgate_worker.py \
  tools/assemble_cross_request_wavefront_microgate.py \
  tools/verify_cross_request_wavefront_microgate.py \
  tools/run_cross_request_wavefront_microgate.py
git diff --check
git status --short --untracked-files=no -- tinyvllm tools
```

Expected: tests and compilation pass, diff check is empty, and no tracked
source change remains.

- [ ] **Step 4: Verify local and remote source SHA**

Push any final focused repair commit, then require:

```bash
local_sha=$(git rev-parse HEAD)
remote_sha=$(git ls-remote \
  origin refs/heads/feat/kv-sparse-attention | awk '{print $1}')
test "$local_sha" = "$remote_sha"
```

### Task 6: Execute the real four-A100 Stage-0 gate

**Files:**

- Create compact receipts under:
  `artifacts/cross_request_wavefront/20260831-cross-request-wavefront-stage0-r1/controller/`
- Create compact terminal bundle under:
  `artifacts/cross_request_wavefront/20260831-cross-request-wavefront-stage0-r1/final_bundle/`

**Interfaces:**

- Consumes the committed source revision from Task 5.
- Produces one immutable terminal classification and two verifier results.

- [ ] **Step 1: Run plan-only locally**

Use a fresh tag:

```bash
/opt/homebrew/bin/python3.12 \
  tools/run_cross_request_wavefront_microgate.py \
  --attempt 20260831-cross-request-wavefront-stage0-r1 \
  --ssh-target sitian@10.232.195.203 \
  --plan-only
```

Inspect the receipt and require every absolute remote task path to descend
from the approved mounted root.

- [ ] **Step 2: Check Kerberos and remote admission**

Run only the controller's read-only preflight. Do not run `kinit` or
`krenew`. If the TTL guard blocks, preserve the controller receipt and stop
without remote mutation.

- [ ] **Step 3: Launch exactly one immutable attempt**

Run:

```bash
/opt/homebrew/bin/python3.12 \
  tools/run_cross_request_wavefront_microgate.py \
  --attempt 20260831-cross-request-wavefront-stage0-r1 \
  --ssh-target sitian@10.232.195.203
```

The local controller owns monitoring and continuation. Do not launch a
replacement worker under the same or another tag while the attempt is live.

- [ ] **Step 4: Require terminal evidence**

Do not classify the attempt until all are present:

```text
classification.json
microgate_summary.json
microgate_rows.jsonl
memory_summary.json
cleanup.json
independent_verification.json
manifest.sha256
local controller verifier receipt
```

Require producer, remote verifier, and local verifier to reconstruct the same
classification.

- [ ] **Step 5: Apply the stop rule**

If the result is not `GO_WAVEFRONT_MICROGATE`, do not modify model runtime
files. If both medians improve by less than `3%`, classify terminal GEMM
fragmentation and end the direction. If the result is GO, Stage 1 still
requires a new written implementation plan before integration.

### Task 7: Audit, commit compact evidence, and push

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-31-cross-request-wavefront-stage0-audit.md`
- Modify:
  `docs/superpowers/specs/2026-08-31-cross-request-wavefront-collective-overlap-design.md`
- Modify:
  `docs/superpowers/plans/2026-08-31-cross-request-wavefront-overlap-stage0.md`
- Add exact compact attempt paths from Task 6.

**Interfaces:**

- Produces the final prompt-to-artifact reconciliation.
- Produces a pushed evidence commit whose local and remote SHAs agree.

- [ ] **Step 1: Write the audit from raw terminal evidence**

The audit must include:

- immutable source/tree/attempt identities;
- selected GPU indices and UUIDs;
- exact shape and pair inventory;
- per-shape baseline/candidate median and P99;
- host-submission benefit or cost;
- communication union and realized-overlap ratios;
- numerical errors and NaN/Inf counts;
- peak allocated/reserved memory delta;
- cleanup;
- producer/remote/local verifier agreement;
- manifest verification;
- terminal classification; and
- the exact claim boundary.

- [ ] **Step 2: Update spec and plan status**

Mark Stage 0 terminal. If non-GO, explicitly prohibit Stage-1 integration.
If GO, authorize only writing the Stage-1 plan, not implementation by
implication.

- [ ] **Step 3: Run final verification**

Run the complete Stage-0 and adjacent suites from Task 5, rerun the local
independent verifier from the downloaded bundle, run:

```bash
git diff --check -- \
  docs/superpowers/specs/2026-08-31-cross-request-wavefront-collective-overlap-design.md \
  docs/superpowers/plans/2026-08-31-cross-request-wavefront-overlap-stage0.md \
  docs/superpowers/audits/2026-08-31-cross-request-wavefront-stage0-audit.md \
  artifacts/cross_request_wavefront/20260831-cross-request-wavefront-stage0-r1
```

- [ ] **Step 4: Commit exact evidence paths**

Stage only:

```text
the Stage-0 audit
the updated Stage-0 spec
the updated Stage-0 plan
the compact controller receipts
the compact final bundle
```

Commit with:

```bash
git -c core.hooksPath=/dev/null commit \
  -m "perf(runtime): qualify TP4 request wavefront" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 5: Verify pushed identity**

Require local HEAD, tracking HEAD, and GitHub branch SHA to match exactly.
