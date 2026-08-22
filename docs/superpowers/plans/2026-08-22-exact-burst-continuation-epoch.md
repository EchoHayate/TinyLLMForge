# Exact Burst GPU-Resident Continuation Epoch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. The user
> has prohibited subagents and additional worktrees for this repository.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse exact-burst graph state across verified consecutive K4 leases
inside one KV-block epoch, reducing setup work while preserving exact outputs,
bounded logits, scheduler ownership, and host-visible cadence.

**Architecture:** `ExactGreedyDecodeBurstGraph` owns an immutable host receipt
for the expected next device state. A complete receipt match allows replay to
continue from graph-resident state and read only the newly written history
slice; any mismatch performs the existing cold bind before mutation. A
separate source-bound four-arm gate compares current K4, K4 continuation, and
current K8 under the same Qwen3-0.6B workload.

**Tech Stack:** Python 3, PyTorch CUDA Graphs, pytest-compatible dependency-light
test scripts, JSON/JSONL evidence, SHA-256 manifests, SSH ControlMaster.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`.
- Do not create a worktree or use a subagent.
- Preserve all unrelated dirty and untracked files.
- Stage only exact task paths; never use broad `git add`, `git reset`, or
  `git clean`.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit contains exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- Keep `exact_greedy_decode_burst_continuation=False` until the complete
  source-bound gate returns `GO_EXACT_BURST_CONTINUATION_EPOCH`.
- Do not launch Qwen3-8B unless Qwen3-0.6B is formally GO.
- Remote task data must stay below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Never write remote task data to `/`, `/tmp`, `/private/tmp`, or
  `/data00/home/sitian/tllm/TinyLLMForge`.
- Do not refresh Kerberos automatically.
- Do not terminate, reset, or interfere with unrelated GPU processes.
- GPU admission requires memory `<=1024 MiB`, utilization `<=5%`, and no
  compute process.
- Use a fresh immutable run tag for every remote attempt.
- Report benefit and cost together.

---

## File Structure

**Modify**

- `tinyvllm/config.py`
  - owns the default-disabled continuation feature flag and type validation.
- `tinyvllm/engine/exact_greedy_decode_burst.py`
  - owns the continuation receipt, continuity decision, history cursor,
    invalidation, accounting, cold bind, and continuation replay.
- `tinyvllm/engine/model_runner.py`
  - supplies lazy block-table materialization and the configured continuation
    policy.
- `tinyvllm/engine/llm_engine.py`
  - invalidates continuation after an engine-side failure following replay.
- `tools/test_exact_greedy_decode_burst.py`
  - dependency-light state-machine and failure tests.
- `tools/test_model_runner_spec_verify.py`
  - source-level integration and configuration checks.
- `AGENT_HANDOFF_STATE.md`
  - terminal source/run/result handoff at true EOF.
- `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
  - prompt-to-artifact reconciliation at true EOF.

**Create**

- `tools/profile_exact_burst_continuation_epoch.py`
  - four-arm performance and correctness evidence producer.
- `tools/test_profile_exact_burst_continuation_epoch.py`
  - schema, inventory, ordering, and source-manifest tests.
- `tools/exact_burst_continuation_epoch_gate.py`
  - producer comparison, classification, and manifest writer.
- `tools/test_exact_burst_continuation_epoch_gate.py`
  - threshold, precedence, and tamper tests.
- `tools/exact_burst_continuation_epoch_verify.py`
  - independent reconstruction with no producer-gate imports.
- `tools/test_exact_burst_continuation_epoch_verify.py`
  - disagreement and corruption tests.
- `tools/run_exact_burst_continuation_epoch_remote.py`
  - source-bound clean-GPU controller.
- `tools/test_run_exact_burst_continuation_epoch_remote.py`
  - path, Kerberos, admission, lifecycle, and download tests.

---

### Task 1: Continuation Receipt and Decision

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Test: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- Produces:
  `ExactGreedyDecodeBurstContinuationReceipt`,
  `ExactGreedyDecodeBurstContinuationDecision`, and
  `decide_exact_greedy_decode_burst_continuation(...)`.
- Consumes: existing `ExactGreedyDecodeBurstLease`.

- [ ] **Step 1: Write failing receipt and exact-match tests**

Add tests that construct a receipt and lease with explicit values:

```python
receipt = ExactGreedyDecodeBurstContinuationReceipt(
    sequence_id=7,
    graph_generation=3,
    block_table_identity=((11, 4), (12, 1)),
    write_block_id=12,
    write_block_generation=1,
    next_input_token=99,
    next_position=260,
    next_context_length=261,
    next_physical_slot=12 * 256 + 4,
    history_cursor=4,
)
decision = decide_exact_greedy_decode_burst_continuation(
    enabled=True,
    receipt=receipt,
    lease=continuation_lease,
    initial_token=99,
    graph_generation=3,
    history_capacity=256,
    block_size=256,
)
assert decision.continue_from_resident_state is True
assert decision.history_start == 4
assert decision.miss_reason is None
```

Add one assertion for each mismatch:

```text
disabled
receipt_missing
sequence_identity_drift
graph_generation_drift
block_table_identity_drift
write_block_identity_drift
initial_token_drift
position_drift
context_length_drift
physical_slot_drift
physical_block_boundary_crossed
history_capacity_exceeded
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: FAIL because the receipt and decision interfaces do not exist.

- [ ] **Step 3: Implement immutable receipt and pure decision**

Add:

```python
@dataclass(frozen=True)
class ExactGreedyDecodeBurstContinuationReceipt:
    sequence_id: int
    graph_generation: int
    block_table_identity: tuple[tuple[int, int], ...]
    write_block_id: int
    write_block_generation: int
    next_input_token: int
    next_position: int
    next_context_length: int
    next_physical_slot: int
    history_cursor: int


@dataclass(frozen=True)
class ExactGreedyDecodeBurstContinuationDecision:
    continue_from_resident_state: bool
    history_start: int
    miss_reason: Optional[str]
```

Implement the pure decision in the exact order listed in Step 1. Require all
integers to be non-boolean and non-negative, require positive capacity and
block size, and require the entire authorized range to remain in one physical
block.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: `exact greedy decode burst tests passed`.

- [ ] **Step 5: Commit the pure policy**

```bash
git add -- tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): define exact burst continuation receipts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 2: Continuation Accounting and Invalidation

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Test: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- Extends `ExactGreedyDecodeBurstStats`.
- Produces:
  `record_continuation_attempt`,
  `record_continuation_hit`,
  `record_cold_bind`,
  `record_continuation_miss`,
  `record_continuation_invalidation`.

- [ ] **Step 1: Write failing accounting tests**

Assert exact summary values after one cold bind, two hits, one miss, and one
invalidation:

```python
assert summary["continuation_attempts"] == 3
assert summary["continuation_hits"] == 2
assert summary["cold_binds"] == 1
assert summary["continuation_miss_counts"] == {
    "position_drift": 1,
}
assert summary["continuation_invalidation_counts"] == {
    "engine_failure:RuntimeError": 1,
}
assert summary["continuation_tokens"] == 8
assert summary["continuation_bursts"] == 2
assert summary["skipped_static_reset_operations"] == 14
assert summary["skipped_scalar_bind_operations"] == 10
assert summary["skipped_block_table_constructions"] == 2
assert summary["skipped_block_table_copy_calls"] == 2
```

Reject empty reasons, boolean counts, negative counts, and invalidation after
an already terminal quarantine only if it would double-count.

- [ ] **Step 2: Run and verify RED**

```bash
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: FAIL on missing continuation accounting fields.

- [ ] **Step 3: Implement exact counters**

Count seven reset operations and five bind operations skipped per continuation
hit. Count skipped block-table bytes from the graph-owned block-table
`numel() * element_size()` supplied by the caller; do not infer bytes from a
model name or prompt length.

- [ ] **Step 4: Run and verify GREEN**

```bash
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -- tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): account for burst continuation reuse" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 3: Graph Cold-Bind and Continuation Replay

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Test: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- `ExactGreedyDecodeBurstGraph.replay(...)` adds:

```python
continuation_enabled: bool = False
block_table_factory: Optional[Callable[[], object]] = None
```

- `ExactGreedyDecodeBurstGraph.invalidate_continuation(reason: str) -> None`
  clears the private receipt and records one invalidation.

- [ ] **Step 1: Write failing cold-bind and hit tests**

Use the existing fake tensors and fake graph. The first replay must:

```text
invoke block_table_factory once
perform reset and scalar binds
read token_history[0:4]
install history_cursor=4
record one cold bind
```

The second contiguous replay must:

```text
not invoke block_table_factory
not add reset/fill/copy events
read token_history[4:8]
advance history_cursor to 8
record one continuation hit
```

Also prove that a position mismatch invokes the factory once and performs a
cold bind before replay.

- [ ] **Step 2: Run and verify RED**

```bash
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: FAIL because replay always resets and requires an eager block table.

- [ ] **Step 3: Implement minimal replay state machine**

At replay entry:

```python
decision = decide_exact_greedy_decode_burst_continuation(
    enabled=continuation_enabled,
    receipt=self._continuation_receipt,
    lease=lease,
    initial_token=initial_token,
    graph_generation=graph_generation,
    history_capacity=int(self.tensors["token_history"].shape[0]),
    block_size=self.block_size,
)
```

On a miss, call `block_table_factory()` exactly once, validate its shape,
perform the current reset/bind path, and use `history_start = 0`. On a hit,
perform none of those operations and use `decision.history_start`.

After replay and D2H succeed, construct the next receipt from the result:

```python
self._continuation_receipt = (
    ExactGreedyDecodeBurstContinuationReceipt(
        sequence_id=lease.sequence_id,
        graph_generation=graph_generation,
        block_table_identity=lease.block_table_identity,
        write_block_id=lease.write_block_id,
        write_block_generation=lease.write_block_generation,
        next_input_token=tokens[-1],
        next_position=lease.first_write_position + completed_replays,
        next_context_length=(
            lease.initial_sequence_length + completed_replays
        ),
        next_physical_slot=(
            lease.first_physical_slot + completed_replays
        ),
        history_cursor=history_start + completed_replays,
    )
)
```

Invalidate before propagating replay, D2H, sampled-logit, or result
construction exceptions.

- [ ] **Step 4: Add failure-boundary tests**

Prove:

- factory failure occurs before graph replay and returns a stable cold-bind
  fallback;
- replay failure invalidates and quarantines;
- final-token D2H failure invalidates and quarantines;
- an invalid receipt never skips setup;
- disabling continuation gives byte-for-byte equivalent fake-event order to
  the existing path.

- [ ] **Step 5: Run and verify GREEN**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_scheduler_prepared_postprocess.py
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add -- tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): reuse resident exact burst state" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 4: Config, ModelRunner, and Engine Integration

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Test: `tools/test_model_runner_spec_verify.py`
- Test: `tools/test_llm_engine_exact_greedy_decode_burst.py`

**Interfaces:**

- Adds `Config.exact_greedy_decode_burst_continuation: bool = False`.
- `ModelRunner._run_exact_greedy_decode_burst` supplies a lazy block-table
  factory.
- `ModelRunner.invalidate_exact_greedy_decode_burst_continuation(reason)`
  invalidates production and correctness graphs.

- [ ] **Step 1: Write failing configuration and source-integration tests**

Require:

```python
assert Config().exact_greedy_decode_burst_continuation is False
```

Require non-booleans to raise. AST/source checks must prove that
`prepare_block_tables_from_rows` appears only inside the lazy factory passed
to `graph.replay`, not before continuation eligibility is evaluated.

Require the engine exception path after a burst replay to call:

```python
self.model_runner.invalidate_exact_greedy_decode_burst_continuation(
    "engine_failure:" + type(error).__name__
)
```

- [ ] **Step 2: Run and verify RED**

```bash
python3 tools/test_model_runner_spec_verify.py
python3 -m pytest -q tools/test_llm_engine_exact_greedy_decode_burst.py
```

Expected: FAIL on the missing flag, lazy factory, and invalidation method.

- [ ] **Step 3: Implement the integration**

Use:

```python
def materialize_block_table():
    return self.prepare_block_tables_from_rows(
        padded_block_table,
        "exact_greedy_burst_block_table",
    )
```

Pass:

```python
continuation_enabled=(
    self.config.exact_greedy_decode_burst_continuation
),
block_table_factory=materialize_block_table,
```

Increase production and correctness `token_history` capacity from eight to
`self.block_size`. Keep graph capture device placement explicit.

- [ ] **Step 4: Run focused and adjacent tests**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_model_runner_spec_verify.py
python3 -m pytest -q tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_scheduler_prepared_postprocess.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_chunked_prefill.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -- tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): integrate exact burst continuation epochs" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 5: Epoch-Relative Correctness Sampling

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Test: `tools/test_exact_greedy_decode_burst.py`
- Test: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Correctness capture accepts epoch-relative sampled ordinals up to
  `block_size - 1`.
- A cold bind resets sampled-logit storage; a continuation hit preserves it.

- [ ] **Step 1: Write failing tests**

Capture ordinals `(0, 63, 126)`. Replay contiguous K4 leases and assert:

```text
ordinal 0 is recorded during the first burst
ordinal 63 is recorded during the sixteenth burst
ordinal 126 is recorded during the final clipped burst
sampled-logit rows are not cleared on continuation hits
sampled-logit rows are cleared on a cold bind
```

Reject duplicate, descending, negative, or `>= block_size` ordinals.

- [ ] **Step 2: Run and verify RED**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_model_runner_spec_verify.py
```

Expected: FAIL because ordinals are currently capped below eight and sampled
storage is reset per replay.

- [ ] **Step 3: Implement epoch-relative sampling**

Replace the fixed `<8` ordinal check with `<history_capacity`. Keep exactly
three sampled-logit rows. Reset sampled logits only during cold bind.

- [ ] **Step 4: Run and verify GREEN**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_model_runner_spec_verify.py
```

Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add -- tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): trace logits across burst continuation epochs" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 6: Four-Arm Profile and Evidence Schema

**Files:**

- Create: `tools/profile_exact_burst_continuation_epoch.py`
- Create: `tools/test_profile_exact_burst_continuation_epoch.py`

**Interfaces:**

- Policies:
  `host_greedy`, `decode_burst_k4`, `decode_burst_k4_continuation`,
  `decode_burst_k8`.
- Produces `case_rows.jsonl`, `correctness_rows.jsonl`,
  `source_manifest.json`, `workload_manifest.json`, and `summary.json`.

- [ ] **Step 1: Write failing schema and inventory tests**

Require:

```python
POLICIES == (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k4_continuation",
    "decode_burst_k8",
)
```

Require 60 performance identities and 48 correctness identities. Require
alternating policy order, unique `(bucket, repetition, policy)` rows, finite
metrics, exact continuation counters, capture costs, history bytes, and
host-visible gaps.

- [ ] **Step 2: Run and verify RED**

```bash
python3 tools/test_profile_exact_burst_continuation_epoch.py
```

Expected: FAIL because the profile module does not exist.

- [ ] **Step 3: Implement the profile**

Reuse shared validation ideas from
`tools/profile_exact_greedy_decode_burst.py`, but define an independent schema:

```text
exact-burst-continuation-epoch.case.v1
exact-burst-continuation-epoch.correctness.v1
exact-burst-continuation-epoch.summary.v1
exact-burst-continuation-epoch.source.v1
exact-burst-continuation-epoch.workload.v1
```

Set continuation only for the K4 continuation arm. Sample logits at
`prefill-final`, `decode-first`, `decode-middle`, and `decode-final`, mapping
decode points to epoch ordinals `0`, `63`, and `126`.

- [ ] **Step 4: Run and verify GREEN**

```bash
python3 tools/test_profile_exact_burst_continuation_epoch.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-continuation-profile-pycache \
  python3 -m py_compile \
  tools/profile_exact_burst_continuation_epoch.py \
  tools/test_profile_exact_burst_continuation_epoch.py
```

Expected: PASS and no compiler output.

- [ ] **Step 5: Commit**

```bash
git add -- tools/profile_exact_burst_continuation_epoch.py \
  tools/test_profile_exact_burst_continuation_epoch.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add burst continuation profile" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 7: Producer Gate and Independent Verifier

**Files:**

- Create: `tools/exact_burst_continuation_epoch_gate.py`
- Create: `tools/test_exact_burst_continuation_epoch_gate.py`
- Create: `tools/exact_burst_continuation_epoch_verify.py`
- Create: `tools/test_exact_burst_continuation_epoch_verify.py`

**Interfaces:**

- Producer writes `comparison.json`, `gate.json`, and `manifest.sha256`.
- Verifier writes `independent-verification.json`.
- Verifier imports no symbols from the producer gate.

- [ ] **Step 1: Write failing producer tests**

Build synthetic 60-row and 48-row fixtures and require:

```text
GO_EXACT_BURST_CONTINUATION_EPOCH
NO_GO_CORRECTNESS
NO_GO_CONTINUATION_COVERAGE
NO_GO_K4_MEDIAN
NO_GO_K4_P95
NO_GO_K8_PARITY
NO_GO_VISIBILITY_RATIO
NO_GO_BUCKET_REGRESSION
NO_GO_TTFT_E2E
NO_GO_THROUGHPUT
NO_GO_MEMORY
NO_GO_COST_INCOMPLETE
NO_GO_EVIDENCE_INCOMPLETE
```

Assert fixed failure precedence follows the order above.

- [ ] **Step 2: Run producer tests and verify RED**

```bash
python3 tools/test_exact_burst_continuation_epoch_gate.py
```

Expected: FAIL because the gate module does not exist.

- [ ] **Step 3: Implement producer classification**

Use exact thresholds from the design. Reconstruct percentiles from raw TPOT
samples. Select only `decode_burst_k4_continuation`; K8 is a paired reference,
not a selectable candidate. Bind every primary artifact and every float32
sidecar in `manifest.sha256`.

- [ ] **Step 4: Write failing independent-verifier tests**

Require the verifier to reject:

- altered raw TPOT samples with unchanged comparison;
- altered continuation counters;
- missing or extra sidecars;
- changed sidecar bytes;
- changed source or workload digests;
- producer/verifier classification disagreement;
- row-count or identity duplication.

- [ ] **Step 5: Implement independent reconstruction**

Duplicate constants intentionally. Parse JSON with non-finite values rejected.
Recompute summaries, percentiles, threshold booleans, classification, file
inventory, SHA-256 values, and source-file digests without importing the
producer.

- [ ] **Step 6: Run and verify GREEN**

```bash
python3 tools/test_exact_burst_continuation_epoch_gate.py
python3 tools/test_exact_burst_continuation_epoch_verify.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-continuation-gate-pycache \
  python3 -m py_compile \
  tools/exact_burst_continuation_epoch_gate.py \
  tools/exact_burst_continuation_epoch_verify.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add -- tools/exact_burst_continuation_epoch_gate.py \
  tools/test_exact_burst_continuation_epoch_gate.py \
  tools/exact_burst_continuation_epoch_verify.py \
  tools/test_exact_burst_continuation_epoch_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): gate exact burst continuation epochs" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 8: Source-Bound Remote Controller

**Files:**

- Create: `tools/run_exact_burst_continuation_epoch_remote.py`
- Create: `tools/test_run_exact_burst_continuation_epoch_remote.py`

**Interfaces:**

- Reuses strict-clean GPU and remote requirement helpers from
  `tools/run_staged_inference_benchmark_remote.py`.
- Uses a task root ending in `exact-burst-continuation-epoch`.

- [ ] **Step 1: Write failing controller tests**

Require:

- all remote paths remain below the approved mounted root;
- source archive contains only `tinyvllm` and `tools`;
- HEAD equals the pushed branch head before launch;
- Kerberos is checked with at least 5,400 seconds before remote setup and
  before upload, never refreshed, and never used as a post-launch lifetime
  threshold;
- polling tolerates two transport failures and fails on the third;
- the selected GPU UUID remains strict-clean immediately before launch;
- no generated command contains `kinit`, `pkill`, `killall`,
  `nvidia-smi --gpu-reset`, `os.kill(`, or `os.killpg(`;
- download rejects missing manifest files and preserves partial evidence;
- remote and local verifier receipts must match exactly.

- [ ] **Step 2: Run and verify RED**

```bash
python3 tools/test_run_exact_burst_continuation_epoch_remote.py
```

Expected: FAIL because the controller does not exist.

- [ ] **Step 3: Implement the controller**

Use:

```text
staging/<run-tag>
runs/<run-tag>
controller-verification/<run-tag>
```

under:

```text
/data00/home/sitian/tinyllmforge-workspaces/
command-timeline-20260818/exact-burst-continuation-epoch
```

The worker command must pin Qwen3-0.6B, TP1, prompt lengths
`256,2048,8192`, 128 generated tokens, two warmups, and five repetitions.
After launch, monitor the worker through the existing SSH transport and rely
on actual command return codes rather than reapplying the 5,400-second launch
threshold.

- [ ] **Step 4: Run and verify GREEN**

```bash
python3 tools/test_run_exact_burst_continuation_epoch_remote.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-continuation-controller-pycache \
  python3 -m py_compile \
  tools/run_exact_burst_continuation_epoch_remote.py \
  tools/test_run_exact_burst_continuation_epoch_remote.py
```

Expected: PASS and no compiler output.

- [ ] **Step 5: Commit**

```bash
git add -- tools/run_exact_burst_continuation_epoch_remote.py \
  tools/test_run_exact_burst_continuation_epoch_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add continuation remote gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 9: Full Local Verification and Source Commit

**Files:**

- Modify only files listed in Tasks 1-8 if verification reveals a scoped
  defect.

- [ ] **Step 1: Run the complete focused suite**

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_model_runner_spec_verify.py
python3 -m pytest -q tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_scheduler_prepared_postprocess.py
python3 tools/test_profile_exact_burst_continuation_epoch.py
python3 tools/test_exact_burst_continuation_epoch_gate.py
python3 tools/test_exact_burst_continuation_epoch_verify.py
python3 tools/test_run_exact_burst_continuation_epoch_remote.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_chunked_prefill.py
```

Expected: all commands exit zero.

- [ ] **Step 2: Run static verification**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-continuation-final-pycache \
  python3 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/profile_exact_burst_continuation_epoch.py \
  tools/exact_burst_continuation_epoch_gate.py \
  tools/exact_burst_continuation_epoch_verify.py \
  tools/run_exact_burst_continuation_epoch_remote.py
git diff --check -- \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/profile_exact_burst_continuation_epoch.py \
  tools/test_profile_exact_burst_continuation_epoch.py \
  tools/exact_burst_continuation_epoch_gate.py \
  tools/test_exact_burst_continuation_epoch_gate.py \
  tools/exact_burst_continuation_epoch_verify.py \
  tools/test_exact_burst_continuation_epoch_verify.py \
  tools/run_exact_burst_continuation_epoch_remote.py \
  tools/test_run_exact_burst_continuation_epoch_remote.py
```

Expected: no output and exit zero.

- [ ] **Step 3: Commit any final source corrections**

If and only if exact task files remain modified:

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/profile_exact_burst_continuation_epoch.py \
  tools/test_profile_exact_burst_continuation_epoch.py \
  tools/exact_burst_continuation_epoch_gate.py \
  tools/test_exact_burst_continuation_epoch_gate.py \
  tools/exact_burst_continuation_epoch_verify.py \
  tools/test_exact_burst_continuation_epoch_verify.py \
  tools/run_exact_burst_continuation_epoch_remote.py \
  tools/test_run_exact_burst_continuation_epoch_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "fix(perf): finalize burst continuation gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

Record the resulting full 40-character source commit. Do not include
artifacts in this commit.

---

### Task 10: Remote Gate, Independent Verification, and Reconciliation

**Files:**

- Create locally under:
  `artifacts/exact_burst_continuation_epoch/20260822-qwen3-06b-exact-burst-continuation-r1/`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Launch one immutable Stage-1 run**

Run:

```bash
continuation_tag="20260822-qwen3-06b-exact-burst-continuation-r1"
continuation_source_commit="$(git rev-parse HEAD)"
python3 tools/run_exact_burst_continuation_epoch_remote.py \
  --run-tag "$continuation_tag" \
  --model-tier qwen3-0.6b \
  --source-commit "$continuation_source_commit"
```

Before launch, verify the exact fixed tag is absent locally and in all three
remote task namespaces. If it exists, edit this plan to use the next `rN`
suffix and commit that plan change before launching. The controller must
monitor through completion; do not start a second worker.

- [ ] **Step 2: Verify immutable inventories**

Require:

```text
performance rows = 60
correctness rows = 48
producer gate present
independent verifier status = PASS
manifest contains every primary file and sidecar
controller completion worker_exitcode = 0
```

Re-run producer and verifier locally against a read-only checkout or archive
of the exact source commit, not against a later branch head.

- [ ] **Step 3: Apply the promotion boundary**

If and only if both verifiers return
`GO_EXACT_BURST_CONTINUATION_EPOCH`, continuation may be enabled for the
proven Qwen3-0.6B scope and a separate Qwen3-8B gate may be planned.

For every NO-GO:

```text
keep exact_greedy_decode_burst_continuation=false
do not run Qwen3-8B
record the first failing classification
retain all partial and terminal artifacts
make no performance claim
```

- [ ] **Step 4: Append canonical reconciliation at true EOF**

Append exact source commit, run tag, GPU UUID, row counts, producer and
verifier classifications, comparison and manifest digests, per-bucket
benefit, K8 parity, visibility ratio, memory cost, continuation coverage, and
promotion result to both canonical documents.

- [ ] **Step 5: Verify documentation and commit**

```bash
git diff --check -- \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
git add -- \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(perf): reconcile burst continuation evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

Expected: the remote branch contains the source, gate tooling, and terminal
reconciliation; unrelated artifacts remain untouched.
