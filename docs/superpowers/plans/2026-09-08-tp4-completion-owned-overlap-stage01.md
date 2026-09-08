# TP4 Completion-Owned Overlap Stage-0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Subagents
> and worktrees are prohibited for this repository task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Correct NCCL completion ownership in the model-neutral TP4
state-commit overlap primitive, then run a source-bound four-GPU gate that
determines whether the corrected overlap is exact and measurably beneficial.

**Architecture:** Keep the existing Stage-0/v1 evidence reader and classifier
available for immutable historical bundles. Add a Stage-0.1/v2 protocol to the
same focused modules. The runtime makes the returned NCCL `Work` authoritative;
the worker separately emits untimed event-only diagnostic rows and timed
baseline/completion-owned rows; assembler and independent verifier reconstruct
the v2 result without trusting producer summaries.

**Tech Stack:** Python 3, PyTorch distributed ProcessGroupNCCL, CUDA streams
and events, pytest, JSON/JSONL evidence manifests, SSH orchestration, Git.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`.
- Stay on and push only `origin/feat/kv-sparse-attention`.
- Do not create a worktree and do not use subagents.
- Preserve all unrelated dirty and untracked files.
- Stage exact paths only; never use `git add -A`, `git reset`, `git clean`, or
  broad formatting.
- Use RED, minimal implementation, then GREEN for every behavior change.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Do not run `kinit` or `krenew`.
- Never terminate, pause, adopt, or modify foreign GPU or process workloads.
- Put all remote task files, caches, logs, artifacts, temporary data, and
  compiler output below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not write task-owned files to remote `/`, `/tmp`, or another
  root-filesystem path.
- Keep large artifacts remote; download only the compact sealed final bundle.
- Every attempt tag is immutable and must be fresh.
- World size is four; formal admission requires four strict-clean A100 GPUs.
- Minimum Kerberos lifetime at launch remains 22,560 seconds.
- Active-token groups are exactly `(1, 4, 8)`.
- The event-only arm is untimed diagnostic evidence and can never promote.
- Correctness failure has precedence over every performance result.
- No timed candidate path may allocate, call `.item()`, busy-poll, poll host
  events, or call `torch.cuda.synchronize()`.
- Do not modify Qwen model code or `tinyvllm/layers/linear.py`.
- Do not integrate with Qwen unless a fresh Stage-0.1 producer, remote
  verifier, and local verifier all return
  `GO_COMPLETION_OWNED_OVERLAP_MICROGATE`.

---

## File structure and compatibility contract

Files modified by the implementation:

- `tinyvllm/engine/collective_side_effect_overlap.py`
  - Owns safe asynchronous collective and side-effect lifecycle.
- `tools/test_collective_side_effect_overlap.py`
  - Proves exact runtime call order and failure behavior.
- `tools/lease_sealed_state_commit_overlap.py`
  - Keeps the v1 classifier and adds v2 row validation and Stage-0.1
    classification.
- `tools/test_lease_sealed_state_commit_overlap.py`
  - Tests v1 non-regression and all v2 classifier branches.
- `tools/lease_sealed_state_commit_overlap_worker.py`
  - Produces v2 diagnostics, formal rows, lifecycle rows, and memory evidence.
- `tools/test_lease_sealed_state_commit_overlap_worker.py`
  - Tests schedule, independent oracle, arm separation, and timed-path safety.
- `tools/assemble_lease_sealed_state_commit_overlap.py`
  - Adds v2 bundle assembly while preserving v1 assembly.
- `tools/test_assemble_lease_sealed_state_commit_overlap.py`
  - Tests v2 inventory, identity, diagnostic rows, and manifest.
- `tools/verify_lease_sealed_state_commit_overlap.py`
  - Dispatches by source schema and independently reconstructs v1 or v2.
- `tools/test_verify_lease_sealed_state_commit_overlap.py`
  - Tests v2 reconstruction, tampering, receipts, and v1 compatibility.
- `tools/run_lease_sealed_state_commit_overlap.py`
  - Launches the v2 worker/assembler/verifier path under the existing safety
    envelope.
- `tools/test_run_lease_sealed_state_commit_overlap.py`
  - Tests protocol identity, paths, admission, launch order, and dual verifier.
- `docs/superpowers/audits/2026-09-08-tp4-completion-owned-overlap-stage01-audit.md`
  - Records the terminal attempt and claim boundary after the GPU campaign.
- `AGENT_HANDOFF_STATE.md`
  - Receives one append-only EOF checkpoint after terminal verification.

No Stage-0 r1-r5 artifact is modified. V1 schemas remain accepted:

```python
STAGE0_SOURCE_SCHEMA = "lease-sealed-state-commit-overlap-source.v1"
STAGE01_SOURCE_SCHEMA = "tp4-completion-owned-overlap-source.v2"
```

---

### Task 1: Transfer collective completion ownership to `Work.wait()`

**Files:**

- Modify: `tinyvllm/engine/collective_side_effect_overlap.py:16-97`
- Modify: `tools/test_collective_side_effect_overlap.py:23-206`

**Interfaces:**

- Consumes: a collective callback returning an object with `wait()`.
- Produces:
  - `OverlapResources.collective_visible_event`
  - `LeaseSealedOverlapTicket.collective_waited: bool`
  - `LeaseSealedOverlapTicket.side_effect_joined: bool`
  - `LeaseSealedCollectiveSideEffect.join(ticket)`

- [ ] **Step 1: Write failing call-order tests**

Replace the old event-only join assertion with tests that record stream
context entry, collective wait, event recording, side-effect join, and state
transition:

```python
def test_join_transfers_collective_ownership_before_side_effect_join():
    events = []
    runtime = executor(events)
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )

    runtime.join(ticket)

    assert events.index(("host", "wait", "collective")) < events.index(
        ("current", "record", "collective_visible")
    )
    assert events.index(("current", "record", "collective_visible")) < (
        events.index(("current", "wait", "side_effect_ready"))
    )
    assert ticket.collective_waited is True
    assert ticket.side_effect_joined is True
    assert ticket.state == "joined"


def test_failed_collective_wait_cannot_join_seal_or_publish():
    events = []
    runtime = executor(events, wait_error=RuntimeError("wait failed"))
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )

    with pytest.raises(RuntimeError, match="wait failed"):
        runtime.join(ticket)

    assert ticket.state == "launched"
    assert ticket.collective_waited is False
    with pytest.raises(RuntimeError, match="joined"):
        runtime.seal(ticket, "identity-a")
```

Update `FakeContext` to append `("context", "enter", stream.name)` and
`("context", "exit", stream.name)`. Update `FakeWork.wait()` to append the
wait event and raise its optional injected error.

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
python3 -m pytest tools/test_collective_side_effect_overlap.py -q
```

Expected: the new tests fail because `join()` does not call
`collective_work.wait()`, does not record `collective_visible_event`, and the
ticket has no ownership flags.

- [ ] **Step 3: Implement the minimal completion-owned join**

Change the resource and ticket fields:

```python
@dataclass(frozen=True)
class OverlapResources:
    communication_stream: object
    side_effect_stream: object
    producer_ready_event: object
    collective_visible_event: object
    side_effect_ready_event: object


@dataclass
class LeaseSealedOverlapTicket:
    commit_identity: str
    local_result: object
    collective_work: object
    collective_visible_event: object
    side_effect_ready_event: object
    collective_waited: bool = False
    side_effect_joined: bool = False
    state: TicketState = "launched"
```

Remove the call that records `consumer_ready_event` immediately after
asynchronous submission. Store `collective_visible_event` on the ticket.
Replace `join()` with:

```python
def join(self, ticket: LeaseSealedOverlapTicket):
    self._require_active(ticket, "launched")
    current = self.current_stream(ticket.local_result)
    with self.stream_context(current):
        ticket.collective_work.wait()
        ticket.collective_waited = True
        ticket.collective_visible_event.record(current)
        current.wait_event(ticket.side_effect_ready_event)
        ticket.side_effect_joined = True
    ticket.state = "joined"
    return ticket.local_result
```

Keep abort fail-closed. If `collective_waited` is false, abort calls
`collective_work.wait()`; it then synchronizes only the owned side-effect
event before invoking the abort callback.

- [ ] **Step 4: Run focused runtime tests and verify GREEN**

Run:

```bash
python3 -m pytest tools/test_collective_side_effect_overlap.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the runtime ownership change**

```bash
git add -- \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/test_collective_side_effect_overlap.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "fix(tp4): own NCCL completion before overlap join" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 2: Add Stage-0.1 v2 evidence schema and classifier

**Files:**

- Modify: `tools/lease_sealed_state_commit_overlap.py:8-477`
- Modify: `tools/test_lease_sealed_state_commit_overlap.py:7-165`

**Interfaces:**

- Preserves: `validate_measurement_row()` and `classify_stage0()` for v1.
- Produces:
  - `DIAGNOSTIC_ITERATION_COUNT = 15`
  - `validate_stage01_diagnostic_row(row) -> dict`
  - `validate_stage01_measurement_row(row) -> dict`
  - `classify_stage01(rows, diagnostic_rows, memory, cleanup) -> dict`

- [ ] **Step 1: Add failing v2 fixtures and precedence tests**

Create `passing_stage01_rows()` with 180 rows keyed by
`(active_tokens, pair_index, rank)` and these v2 fields:

```python
{
    "arm_order": ["baseline", "completion_owned"],
    "baseline_critical_ns": 100_000,
    "candidate_critical_ns": 90_000,
    "baseline_host_submission_ns": 20_000,
    "candidate_host_submission_ns": 20_400,
    "collective_outstanding_window_ns": [10_000, 60_000],
    "side_effect_window_ns": [35_000, 75_000],
    "overlap_intersection_ns": 25_000,
    "expected_reduced_exact": True,
    "baseline_reduced_exact": True,
    "candidate_reduced_exact": True,
    "baseline_final_exact": True,
    "candidate_final_exact": True,
    "baseline_candidate_exact": True,
    "shadow_payload_exact": True,
    "active_state_preserved_before_publish": True,
    "published_state_exact": True,
    "abort_preserved_old_state": True,
    "commit_identity_match": True,
    "collective_wait_invoked": True,
    "collective_dependency_transferred": True,
    "side_effect_dependency_joined": True,
    "finite_output": True,
    "timed_path_allocation_count": 0,
    "timed_out": False,
}
```

Create `passing_stage01_diagnostics()` with 180 rows keyed by
`(active_tokens, diagnostic_index, rank)`. Baseline and completion-owned flags
are true on every row. Set `event_only_reduced_exact=False` and
`event_only_final_exact=False` on one row and true elsewhere.

Add parameterized mutations for:

```python
(
    ("candidate_correctness", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
    ("wait_not_invoked", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
    ("diagnostic_missing", "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"),
    ("diagnostic_not_reproduced", "INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED"),
    ("overlap", "NO_GO_INSUFFICIENT_OVERLAP"),
    ("median", "NO_GO_PERFORMANCE"),
    ("tail", "NO_GO_PERFORMANCE"),
    ("host", "NO_GO_PERFORMANCE"),
)
```

- [ ] **Step 2: Run the schema tests and capture RED**

Run:

```bash
python3 -m pytest tools/test_lease_sealed_state_commit_overlap.py -q
```

Expected: import or assertion failures for the missing v2 functions and
classification.

- [ ] **Step 3: Implement v2 validation**

Add:

```python
DIAGNOSTIC_ITERATION_COUNT = 15


def validate_stage01_diagnostic_row(row):
    if not isinstance(row, dict):
        raise ValueError("diagnostic row must be an object")
    if row.get("active_tokens") not in ACTIVE_TOKEN_GROUPS:
        raise ValueError("diagnostic active_tokens is invalid")
    if type(row.get("diagnostic_index")) is not int or row[
        "diagnostic_index"
    ] not in range(DIAGNOSTIC_ITERATION_COUNT):
        raise ValueError("diagnostic_index is invalid")
    if type(row.get("rank")) is not int or row["rank"] not in range(WORLD_SIZE):
        raise ValueError("diagnostic rank is invalid")
    flags = (
        "baseline_reduced_exact",
        "baseline_final_exact",
        "completion_owned_reduced_exact",
        "completion_owned_final_exact",
        "event_only_reduced_exact",
        "event_only_final_exact",
    )
    if any(type(row.get(name)) is not bool for name in flags):
        raise ValueError("diagnostic correctness flag is invalid")
    return dict(row)
```

Implement `validate_stage01_measurement_row()` with the v2 names above,
recompute `overlap_intersection_ns` from
`collective_outstanding_window_ns` and `side_effect_window_ns`, and require
the deterministic AB/BA order:

```python
expected_order = (
    ["baseline", "completion_owned"]
    if pair_index % 2 == 0
    else ["completion_owned", "baseline"]
)
```

- [ ] **Step 4: Implement the v2 classifier**

Implement `classify_stage01()` with this exact precedence:

```python
MIN_OVERLAP_RATIO = 0.20
MIN_AGGREGATE_SPEEDUP = 0.05
MAX_SINGLE_TOKEN_MEDIAN_REGRESSION = 0.01
MAX_P99_REGRESSION = 0.03
MAX_HOST_SUBMISSION_REGRESSION = 0.03
MIN_DIRECTIONAL_PAIR_COUNT = 11
MAX_RESERVED_SLACK_BYTES = 64 * 1024 * 1024

STAGE01_CLASSIFICATIONS = (
    "NO_GO_CORRECTNESS_OR_LIFECYCLE",
    "NO_GO_RESOURCE_IDENTITY",
    "NO_GO_MEMORY_OR_ALLOCATION",
    "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT",
    "INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED",
    "NO_GO_INSUFFICIENT_OVERLAP",
    "NO_GO_PERFORMANCE",
    "GO_COMPLETION_OWNED_OVERLAP_MICROGATE",
)
```

It must:

1. Validate complete 180-row formal and 180-row diagnostic inventories.
2. Return correctness no-go if any baseline or candidate exactness, lifecycle,
   wait-ownership, finite-output, or timeout flag fails.
3. Return environment/measurement inconclusive for malformed or incomplete
   evidence or dirty cleanup.
4. Return diagnostic inconclusive if every event-only diagnostic row is exact.
5. Reuse the frozen memory, overlap, speed, P99, host, and pair-direction
   thresholds.
6. Set `stage1_authorized=True` only for
   `GO_COMPLETION_OWNED_OVERLAP_MICROGATE`.

- [ ] **Step 5: Run v1 and v2 classifier tests**

Run:

```bash
python3 -m pytest tools/test_lease_sealed_state_commit_overlap.py -q
```

Expected: all tests pass, including unchanged v1 fixtures.

- [ ] **Step 6: Commit the v2 schema**

```bash
git add -- \
  tools/lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add completion-owned overlap classifier" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 3: Produce independent-oracle diagnostics and formal measurements

**Files:**

- Modify: `tools/lease_sealed_state_commit_overlap_worker.py:17-860`
- Modify: `tools/test_lease_sealed_state_commit_overlap_worker.py:17-285`

**Interfaces:**

- Consumes: safe runtime from Task 1 and v2 validators from Task 2.
- Produces:
  - `diagnostic_rows.rank0.jsonl` through `diagnostic_rows.rank3.jsonl`
  - `measurement_rows.rank0.jsonl` through `measurement_rows.rank3.jsonl`
  - v2 lifecycle ownership flags
  - completion-owned outstanding-window timing

- [ ] **Step 1: Write failing worker structure tests**

Add tests asserting:

```python
def test_stage01_schedule_freezes_diagnostics_and_formal_pairs():
    schedule = build_workload_schedule()
    assert all(len(row["diagnostics"]) == 15 for row in schedule)
    assert all(len(row["warmups"]) == 2 for row in schedule)
    assert all(len(row["measurements"]) == 15 for row in schedule)
    assert schedule[0]["measurements"][0]["arm_order"] == (
        "baseline",
        "completion_owned",
    )


def test_completion_owned_timed_path_has_no_forbidden_host_sync():
    source = inspect.getsource(_run_completion_owned)
    for forbidden in (
        "torch.cuda.synchronize",
        ".item(",
        ".synchronize(",
        "torch.empty",
        "torch.zeros",
        "torch.cuda.Stream",
        "torch.cuda.Event",
    ):
        assert forbidden not in source


def test_event_only_arm_is_diagnostic_only():
    source = inspect.getsource(_run_event_only_diagnostic)
    assert "collective_work.wait()" not in source
    formal = inspect.getsource(run_worker).split(
        "for pair in workload[\"measurements\"]:", 1
    )[1]
    assert "_run_event_only_diagnostic(" not in formal
```

Add a pure helper test for `build_expected_reduction()` that supplies four
small known rank tensors and checks an exact FP32 sum and BF16 cast.

- [ ] **Step 2: Run worker tests and capture RED**

Run:

```bash
python3 -m pytest tools/test_lease_sealed_state_commit_overlap_worker.py -q
```

Expected: failures for missing v2 schedule, functions, buffers, and fields.

- [ ] **Step 3: Rename timing resources and preallocate all buffers**

In `OverlapBuffers`, replace `consumer_ready_event`, `allreduce_started`, and
`allreduce_completed` with:

```python
collective_visible_event: object
state_copy_started: object
state_copy_completed: object
expected_result: object
expected_output: object
diagnostic_result: object
diagnostic_output: object
diagnostic_shadow: object
expected_rank_stack: object
```

Allocate every tensor, stream, and event in `OverlapBuffers.create()` before
warmup. Pass `collective_visible_event` into `OverlapResources`.

- [ ] **Step 4: Build an execution-independent correctness oracle**

Use deterministic rank inputs and compute the expected sum before timing:

```python
def build_expected_reduction(*, buffers, rank_inputs, torch):
    torch.stack(rank_inputs, dim=0, out=buffers.expected_rank_stack)
    torch.sum(
        buffers.expected_rank_stack,
        dim=0,
        out=buffers.expected_result,
    )
    buffers.expected_output.copy_(buffers.expected_result)
    return buffers.expected_result, buffers.expected_output
```

`expected_rank_stack` must also be preallocated. Gather deterministic local
input bytes outside warmup and formal timing. The oracle may synchronize
outside the timed path, but it may not reuse baseline or candidate output as
its expected value.

- [ ] **Step 5: Implement separate diagnostic and formal arms**

Keep the unsafe arm private to the worker:

```python
def _run_event_only_diagnostic(*, buffers, torch, dist):
    stream = torch.cuda.current_stream(buffers.local_result.device)
    buffers.diagnostic_result.copy_(buffers.local_result)
    with torch.cuda.stream(buffers.communication_stream):
        buffers.communication_stream.wait_event(buffers.producer_ready_event)
        work = dist.all_reduce(buffers.diagnostic_result, async_op=True)
        buffers.event_only_submitted_event.record(
            buffers.communication_stream
        )
    stream.wait_event(buffers.event_only_submitted_event)
    buffers.diagnostic_output.copy_(buffers.diagnostic_result)
    return work
```

The function returns the `Work` only so the untimed diagnostic harness can
retire owned work after capturing the premature output. It must never be
called by the formal measurement loop.

Rename `_run_candidate()` to `_run_completion_owned()`. It calls the safe
runtime and returns:

```python
{
    "reduced_result": result,
    "final_output": buffers.candidate_output,
    "shadow": buffers.candidate_shadow,
    "started": buffers.candidate_started,
    "completed": buffers.candidate_completed,
    "collective_visible": buffers.collective_visible_event,
    "state_copy_started": buffers.state_copy_started,
    "state_copy_completed": buffers.state_copy_completed,
    "collective_wait_invoked": ticket.collective_waited,
    "side_effect_dependency_joined": ticket.side_effect_joined,
    "host_submission_ns": time.perf_counter_ns() - submitted,
}
```

- [ ] **Step 6: Emit exact v2 rows**

For each diagnostic triplet, copy outputs before retiring event-only work,
then emit oracle comparisons for all arms. For formal rows, emit:

```python
"collective_outstanding_window_ns": _event_interval_ns(
    candidate["started"],
    buffers.producer_ready_event,
    candidate["collective_visible"],
),
"side_effect_window_ns": _event_interval_ns(
    candidate["started"],
    candidate["state_copy_started"],
    candidate["state_copy_completed"],
),
"expected_reduced_exact": bool(
    torch.equal(candidate["reduced_result"], buffers.expected_result)
),
"baseline_reduced_exact": bool(
    torch.equal(baseline["reduced_result"], buffers.expected_result)
),
"candidate_reduced_exact": bool(
    torch.equal(candidate["reduced_result"], buffers.expected_result)
),
"baseline_final_exact": bool(
    torch.equal(baseline["final_output"], buffers.expected_output)
),
"candidate_final_exact": bool(
    torch.equal(candidate["final_output"], buffers.expected_output)
),
"baseline_candidate_exact": bool(
    torch.equal(baseline["final_output"], candidate["final_output"])
),
```

Perform digest creation, `.item()`-based finite checks, all-gathered identity
checks, and event synchronization only after the timed pair completes.

- [ ] **Step 7: Run worker tests and static source checks**

Run:

```bash
python3 -m pytest tools/test_lease_sealed_state_commit_overlap_worker.py -q
python3 -m py_compile \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/lease_sealed_state_commit_overlap_worker.py
```

Expected: all tests pass and compilation exits zero.

- [ ] **Step 8: Commit worker protocol v2**

```bash
git add -- \
  tools/lease_sealed_state_commit_overlap_worker.py \
  tools/test_lease_sealed_state_commit_overlap_worker.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): measure completion-owned overlap" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 4: Assemble a v2 source-bound evidence bundle

**Files:**

- Modify: `tools/assemble_lease_sealed_state_commit_overlap.py:1-546`
- Modify: `tools/test_assemble_lease_sealed_state_commit_overlap.py:11-288`

**Interfaces:**

- Consumes: `diagnostic_rows`, formal rows, memory, lifecycle, cleanup, and v2
  source identity.
- Produces: `assemble_stage01_bundle` returning the producer-result mapping.

- [ ] **Step 1: Add failing v2 assembler tests**

Add a `passing_stage01_inputs()` fixture and assert that assembly emits exactly:

```python
STAGE01_PRODUCER_ARTIFACTS = frozenset({
    "source_manifest.json",
    "environment_manifest.json",
    "gpu_rank_manifest.json",
    "workload_manifest.json",
    "admission.json",
    "diagnostic_rows.jsonl",
    "paired_rows.jsonl",
    "correctness_rows.jsonl",
    "lifecycle_rows.jsonl",
    "memory_rows.jsonl",
    "overlap_rows.jsonl",
    "cleanup.json",
    "producer_result.json",
    "report.md",
    "manifest.sha256",
})
```

Test rejection of a missing diagnostic row, mismatched attempt identity,
non-finite timing, and a nonempty output directory.

- [ ] **Step 2: Run assembler tests and capture RED**

Run:

```bash
python3 -m pytest tools/test_assemble_lease_sealed_state_commit_overlap.py -q
```

Expected: failures for missing v2 assembler and artifact inventory.

- [ ] **Step 3: Implement v2 assembly without changing v1 behavior**

Add this exact public signature:

```python
STAGE01_MANIFEST_SCHEMA = "tp4-completion-owned-overlap-manifest.v2"


def assemble_stage01_bundle(
    *,
    output_root,
    source_identity,
    rows,
    diagnostic_rows,
    memory,
    lifecycle,
    cleanup,
):
    return _assemble_validated_bundle(
        protocol="completion-owned-stage01",
        manifest_schema=STAGE01_MANIFEST_SCHEMA,
        output_root=output_root,
        source_identity=source_identity,
        rows=rows,
        diagnostic_rows=diagnostic_rows,
        memory=memory,
        lifecycle=lifecycle,
        cleanup=cleanup,
    )
```

Extract the current v1 assembly body into `_assemble_validated_bundle` with
explicit `protocol`, `manifest_schema`, and `diagnostic_rows` parameters. For
v1, pass `protocol="lease-sealed-stage0"` and
`diagnostic_rows=None`; this must emit the byte-compatible v1 inventory. For
v2, the helper validates source schema
`tp4-completion-owned-overlap-source.v2`, applies identity to every row,
calls `classify_stage01()`, writes `diagnostic_rows.jsonl`, projects formal
correctness and overlap rows, and generates a report titled:

```text
TP4 Completion-Owned Overlap Stage-0.1
```

The workload manifest includes:

```python
{
    "protocol": "completion-owned-stage01",
    "world_size": 4,
    "active_token_groups": [1, 4, 8],
    "diagnostic_iteration_count": 15,
    "warmup_pair_count": 2,
    "measured_pair_count": 15,
    "formal_arms": ["baseline", "completion_owned"],
    "diagnostic_arms": [
        "baseline",
        "event_only",
        "completion_owned",
    ],
}
```

Dispatch `assemble_raw_attempt()` by the source schema so v1 inputs continue
to call `assemble_bundle()` and v2 inputs call `assemble_stage01_bundle()`.

- [ ] **Step 4: Run assembler and legacy regression tests**

Run:

```bash
python3 -m pytest \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py -q
```

Expected: all v1 and v2 tests pass.

- [ ] **Step 5: Commit v2 assembly**

```bash
git add -- \
  tools/assemble_lease_sealed_state_commit_overlap.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): assemble completion-owned evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 5: Independently verify v1 and v2 bundles

**Files:**

- Modify: `tools/verify_lease_sealed_state_commit_overlap.py:1-548`
- Modify: `tools/test_verify_lease_sealed_state_commit_overlap.py:9-179`

**Interfaces:**

- Consumes: sealed producer bundle with v1 or v2 source schema.
- Produces: independent receipt with reconstructed classification and artifact
  hashes.

- [ ] **Step 1: Add failing v2 verifier tests**

Add tests that:

- assemble a passing v2 bundle and reconstruct
  `GO_COMPLETION_OWNED_OVERLAP_MICROGATE`;
- mutate one diagnostic event-only flag so the known unsafe behavior is no
  longer reproduced and reconstruct
  `INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED`;
- mutate one candidate exactness flag and reconstruct correctness no-go;
- reject missing or extra v2 artifacts;
- reject hash mutation and producer-summary disagreement;
- preserve distinct remote and local receipts through terminal sealing;
- still verify a v1 fixture through the legacy path;
- confirm the verifier source does not import the assembler module.

- [ ] **Step 2: Run verifier tests and capture RED**

Run:

```bash
python3 -m pytest tools/test_verify_lease_sealed_state_commit_overlap.py -q
```

Expected: failures for missing v2 schema dispatch and reconstruction.

- [ ] **Step 3: Implement schema dispatch and v2 reconstruction**

Update `_verify_manifest()` to accept the v1 and v2 manifest schemas, validate
the schema-specific artifact inventory, verify every declared hash, and return
the manifest schema. Then read `source_manifest.json`, require the matching
source schema, and dispatch with this complete helper:

```python
def _verify_by_schema(root, source, *, receipt_name, seal_terminal):
    if source["schema_version"] == STAGE0_SOURCE_SCHEMA:
        return _verify_stage0_bundle(
            root,
            source,
            receipt_name=receipt_name,
            seal_terminal=seal_terminal,
        )
    if source["schema_version"] == STAGE01_SOURCE_SCHEMA:
        return _verify_stage01_bundle(
            root,
            source,
            receipt_name=receipt_name,
            seal_terminal=seal_terminal,
        )
    raise ValueError("source schema is unsupported")
```

The v2 verifier independently:

1. validates the exact v2 artifact inventory;
2. validates source, runtime, admission, workload, GPU/rank, and cleanup
   identity;
3. reconstructs diagnostic, formal, lifecycle, memory, correctness, and
   overlap projections;
4. calls `classify_stage01()` directly;
5. requires exact producer equality;
6. writes the selected receipt atomically;
7. seals only after a remote receipt exists;
8. rewrites the content-hash manifest without mutating existing receipts.

- [ ] **Step 4: Run verifier, assembler, and classifier suites**

Run:

```bash
python3 -m pytest \
  tools/test_lease_sealed_state_commit_overlap.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit independent v2 verification**

```bash
git add -- \
  tools/verify_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): verify completion-owned evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 6: Route the safe controller through Stage-0.1

**Files:**

- Modify: `tools/run_lease_sealed_state_commit_overlap.py:37-1572`
- Modify: `tools/test_run_lease_sealed_state_commit_overlap.py:11-700`

**Interfaces:**

- Consumes: committed source revision, fresh attempt tag, valid Kerberos
  probe, and four strict-clean GPU rows.
- Produces: remote attempt, compact v2 bundle, dual verifier receipts, and a
  terminal controller receipt.

- [ ] **Step 1: Add failing controller protocol tests**

Freeze:

```python
PLAN_SCHEMA = "tp4-completion-owned-overlap-plan.v2"
PROTOCOL = "completion-owned-stage01"
```

Tests require:

- plan embeds `PROTOCOL`;
- all remote paths and environment paths stay below the approved mount;
- worker commands include `--protocol completion-owned-stage01`;
- attempt creation rejects any existing target;
- launch rechecks the same selected four GPUs;
- Kerberos failure stops before GPU or remote access;
- worker runs before assembler, remote verifier, download, and local verifier;
- all three classifications must agree;
- only exact-tag-owned process groups may be terminated;
- local bundle download excludes raw traces.

- [ ] **Step 2: Run controller tests and capture RED**

Run:

```bash
python3 -m pytest tools/test_run_lease_sealed_state_commit_overlap.py -q
```

Expected: failures for missing Stage-0.1 protocol identity and command
arguments.

- [ ] **Step 3: Implement protocol-bound orchestration**

Add `"protocol": PROTOCOL` to plans, source identity, launch admission, and
terminal receipts. Add to every worker command:

```text
--protocol completion-owned-stage01
```

Keep:

```python
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
MINIMUM_KERBEROS_LIFETIME_SECONDS = 22_560
```

Update remote assembler and verifier invocations to use v2 schema dispatch.
Do not alter ownership checks, retry budget, fresh-path checks, second GPU
admission, or exact-tag cleanup.

- [ ] **Step 4: Run the complete six-file CPU suite**

Run:

```bash
python3 -m pytest \
  tools/test_collective_side_effect_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap_worker.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py \
  tools/test_run_lease_sealed_state_commit_overlap.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run compilation and static safety scans**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-stage01-pycache \
python3 -m py_compile \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/lease_sealed_state_commit_overlap.py \
  tools/lease_sealed_state_commit_overlap_worker.py \
  tools/assemble_lease_sealed_state_commit_overlap.py \
  tools/verify_lease_sealed_state_commit_overlap.py \
  tools/run_lease_sealed_state_commit_overlap.py

python3 - <<'PY'
from pathlib import Path
import ast

worker = Path(
    "tools/lease_sealed_state_commit_overlap_worker.py"
).read_text()
tree = ast.parse(worker)
forbidden = ("torch.cuda.synchronize", ".item(", "while not")
start = worker.index("def _run_completion_owned")
end = worker.index("\\ndef ", start + 1)
timed = worker[start:end]
for token in forbidden:
    assert token not in timed, token
assert "_run_event_only_diagnostic(" not in worker[
    worker.index('for pair in workload["measurements"]:') :
]
print("stage01_static_safety=PASS")
PY

git diff --check
```

Expected: compilation, static safety, and whitespace checks pass.

- [ ] **Step 6: Commit controller integration**

```bash
git add -- \
  tools/run_lease_sealed_state_commit_overlap.py \
  tools/test_run_lease_sealed_state_commit_overlap.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): orchestrate completion-owned gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 7: Review, publish, and freeze the executable source

**Files:**

- Inspect only the eleven implementation and test files listed in Tasks 1-6.

**Interfaces:**

- Consumes: completed CPU suite and static checks.
- Produces: one pushed source revision used by the immutable remote attempt.

- [ ] **Step 1: Perform a path-limited self-review**

Run:

```bash
git status --short -- \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/lease_sealed_state_commit_overlap.py \
  tools/lease_sealed_state_commit_overlap_worker.py \
  tools/assemble_lease_sealed_state_commit_overlap.py \
  tools/verify_lease_sealed_state_commit_overlap.py \
  tools/run_lease_sealed_state_commit_overlap.py \
  tools/test_collective_side_effect_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap_worker.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py \
  tools/test_run_lease_sealed_state_commit_overlap.py
```

Review exact diffs for completion ownership, v1 compatibility, event-only
isolation, independent oracle, classifier precedence, remote path safety, and
owned cleanup. Fix defects through focused RED/GREEN cycles and exact-path
commits.

- [ ] **Step 2: Re-run fresh verification**

Run the complete six-file pytest command and the compilation/static checks
from Task 6 again.

Expected: zero failures and zero static-safety violations.

- [ ] **Step 3: Push and verify SHA equality**

```bash
git push origin feat/kv-sparse-attention
local_sha=$(git rev-parse HEAD)
tracking_sha=$(git rev-parse origin/feat/kv-sparse-attention)
remote_sha=$(git ls-remote origin refs/heads/feat/kv-sparse-attention |
  awk '{print $1}')
test "$local_sha" = "$tracking_sha"
test "$local_sha" = "$remote_sha"
```

Expected: all three SHAs are identical. This exact SHA becomes the campaign
source revision.

---

### Task 8: Run the fresh four-GPU Stage-0.1 campaign

**Files:**

- Create remotely under the fresh attempt root only.
- Download only:
  `artifacts/lease_sealed_state_commit_overlap/${attempt_tag}/final_bundle/`

**Interfaces:**

- Consumes: pushed source SHA, at least 22,560 seconds of Kerberos lifetime,
  and four strict-clean A100 GPUs.
- Produces: immutable v2 producer bundle plus remote and local verifier
  receipts.

- [ ] **Step 1: Freeze a fresh attempt identity**

Initial tag:

```text
20260908-tp4-completion-owned-overlap-stage01-r1
```

If that exact local or remote path already exists, do not reuse or delete it.
Increment only the final numeric revision suffix and rerun the preflight with
the fresh tag.

- [ ] **Step 2: Run plan-only and dry-run admission**

Run the controller with `--plan-only`, then `--dry-run`. Record:

- Kerberos remaining lifetime;
- resolved remote mount and distinct-filesystem result;
- fresh non-symlink attempt path;
- selected physical indices and UUIDs;
- strict-clean memory, utilization, and process rows;
- source revision and source-tree SHA-256.

Expected: no worker starts during either command.

- [ ] **Step 3: Launch only after immediate second admission**

Run the same controller without `--plan-only` or `--dry-run`. The controller
must:

1. recheck Kerberos;
2. recheck the frozen GPU set;
3. create the fresh remote attempt;
4. perform immediate second admission;
5. write launch admission;
6. start four owned workers;
7. assemble the v2 bundle;
8. run the remote verifier;
9. download only the compact bundle;
10. run and seal the local verifier.

Do not treat a monitor PID, background shell, or controller-start event as run
evidence.

- [ ] **Step 4: Re-run post-seal local verification**

Run:

```bash
attempt_tag=20260908-tp4-completion-owned-overlap-stage01-r1
python3 tools/verify_lease_sealed_state_commit_overlap.py \
  "artifacts/lease_sealed_state_commit_overlap/${attempt_tag}/final_bundle" \
  --check-only
```

Expected: verifier status `PASS`, with reconstructed classification exactly
matching producer and remote verifier. `PASS` describes evidence integrity;
the separate reconstructed classification may be GO, no-go, or inconclusive.

- [ ] **Step 5: Apply the frozen stop rule**

- If correctness or lifecycle fails: stop with
  `NO_GO_CORRECTNESS_OR_LIFECYCLE`.
- If event-only failure is not reproduced: stop with
  `INCONCLUSIVE_DIAGNOSTIC_NOT_REPRODUCED`.
- If overlap is below threshold: stop with `NO_GO_INSUFFICIENT_OVERLAP`.
- If overlap exists but latency, P99, host, memory, or directional gates fail:
  stop with the applicable no-go.
- Only unanimous `GO_COMPLETION_OWNED_OVERLAP_MICROGATE` permits a later,
  separately planned Qwen integration.

---

### Task 9: Publish the terminal audit and handoff

**Files:**

- Create:
  `docs/superpowers/audits/2026-09-08-tp4-completion-owned-overlap-stage01-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md` at true EOF

**Interfaces:**

- Consumes: sealed compact final bundle and three matching classifications.
- Produces: durable terminal record and next-action boundary.

- [ ] **Step 1: Write the audit from sealed artifacts**

The audit must contain:

- exact attempt tag, source revision, tree hash, environment, and rank mapping;
- row inventory and artifact inventory;
- diagnostic-control outcome;
- exact correctness and lifecycle counts;
- per-shape baseline/candidate median, P90, P95, P99, absolute delta, and
  paired speed ratio;
- collective outstanding window, side-effect interval, overlap intersection,
  and realized overlap;
- host-submission and memory cost;
- cleanup result;
- producer, remote verifier, local verifier, and post-seal results;
- prompt-to-artifact checklist;
- final classifier and Stage-1 authorization;
- an explicit statement that Stage-0.1 is not Qwen end-to-end evidence.

- [ ] **Step 2: Append the handoff at true EOF**

Record the same source identity, terminal classification, compact bundle
path, exact verifier command, claim boundary, and one next action:

```text
GO -> write a separately reviewed Qwen3.8 Stage-1 integration plan
otherwise -> stop this mechanism and design a larger-granularity TP path
```

- [ ] **Step 3: Verify documentation against artifacts**

Run a script that loads producer and verifier JSON, extracts every numeric
value quoted in the audit, and asserts equality. Then run:

```bash
git diff --check -- \
  docs/superpowers/audits/2026-09-08-tp4-completion-owned-overlap-stage01-audit.md \
  AGENT_HANDOFF_STATE.md
```

Expected: artifact comparison and whitespace checks pass.

- [ ] **Step 4: Commit and push the terminal record**

```bash
git add -- \
  docs/superpowers/audits/2026-09-08-tp4-completion-owned-overlap-stage01-audit.md \
  AGENT_HANDOFF_STATE.md
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "docs(tp4): record completion-owned overlap result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 5: Verify final publication**

```bash
local_sha=$(git rev-parse HEAD)
tracking_sha=$(git rev-parse origin/feat/kv-sparse-attention)
remote_sha=$(git ls-remote origin refs/heads/feat/kv-sparse-attention |
  awk '{print $1}')
test "$local_sha" = "$tracking_sha"
test "$local_sha" = "$remote_sha"
```

Expected: all SHAs match, the compact bundle remains unchanged, and no
unrelated path is staged or committed.
