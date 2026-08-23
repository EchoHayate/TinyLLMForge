# Split-Phase K8 Exact-Burst Superlease Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. The user
> has prohibited subagents and additional worktrees for this repository.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one exact-greedy K8 GPU burst while publishing two ordered
K4 token phases, preserving K8 efficiency and reducing host-visible cadence.

**Architecture:** A new dependency-light split-phase module owns immutable
publication tickets, mailbox transfer handles, transaction state, and
inventory validation. `ExactGreedyDecodeBurstGraph` enqueues four replays,
an event-ordered prefix D2H, four more replays, and a suffix D2H; the scheduler
commits prefix and suffix under one parent lease, while `LLMEngine.step()`
drains any pending suffix before the next scheduler decision.

**Tech Stack:** Python 3, dataclasses, PyTorch CUDA streams/events and pinned
memory, pytest-compatible dependency-light test scripts, JSON/JSONL evidence,
SHA-256 manifests, SSH ControlMaster.

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
- Keep `exact_greedy_decode_burst_split_phase=False` until the complete
  source-bound gate returns `GO_EXACT_BURST_SPLIT_PHASE`.
- Require `exact_greedy_decode_burst=True`,
  `exact_greedy_decode_burst_tokens=8`, and
  `exact_greedy_decode_burst_continuation=False` when split phase is enabled.
- Stage 1 supports TP1, rank 0, batch size 1, completion-only,
  `temperature == 0`, and `ignore_eos == true`.
- Do not launch Qwen3-8B unless Qwen3-0.6B is formally GO.
- Remote task data must stay below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Never write remote task data to `/`, `/tmp`, `/private/tmp`, or the retired
  checkout.
- Do not refresh Kerberos automatically.
- Do not terminate, reset, or interfere with unrelated GPU processes.
- GPU admission requires memory `<=1024 MiB`, utilization `<=5%`, and no
  compute process.
- Use a fresh immutable run tag for every remote attempt.
- Preserve the complete frozen matrix after an early threshold failure.
- Report benefit and cost together.

---

## File Structure

**Create**

- `tinyvllm/engine/exact_greedy_decode_burst_split_phase.py`
  - owns publication tickets, transfer/result records, mailbox generations,
    transaction transitions, and pure inventory validation.
- `tools/test_exact_greedy_decode_burst_split_phase.py`
  - dependency-light contract, state-machine, mailbox, and failure tests.
- `tools/profile_exact_burst_split_phase.py`
  - four-arm Qwen3-0.6B performance and correctness producer.
- `tools/test_profile_exact_burst_split_phase.py`
  - matrix, phase inventory, source binding, and sidecar tests.
- `tools/exact_burst_split_phase_gate.py`
  - producer comparison, frozen threshold evaluation, and manifest writer.
- `tools/test_exact_burst_split_phase_gate.py`
  - classification, boundary, and tamper tests.
- `tools/exact_burst_split_phase_verify.py`
  - independent reconstruction without importing the producer gate.
- `tools/test_exact_burst_split_phase_verify.py`
  - producer/verifier disagreement and corruption tests.
- `tools/run_exact_burst_split_phase_remote.py`
  - source-bound clean-GPU controller and artifact collector.
- `tools/test_run_exact_burst_split_phase_remote.py`
  - path, Kerberos, admission, lifecycle, and download tests.

**Modify**

- `tinyvllm/config.py`
  - owns the default-disabled strict split-phase flag and composition checks.
- `tinyvllm/engine/exact_greedy_decode_burst.py`
  - owns the graph replay boundary and split-result validation hook.
- `tinyvllm/engine/model_runner.py`
  - owns the CUDA stream/event/pinned-mailbox backend and split capability.
- `tinyvllm/engine/scheduler.py`
  - owns parent-lease phase state and two ordered publication transactions.
- `tinyvllm/engine/llm_engine.py`
  - owns pending suffix state and pre-schedule drain.
- `tools/test_exact_greedy_decode_burst.py`
  - graph replay ordering and no-regression tests.
- `tools/test_llm_engine_exact_greedy_decode_burst.py`
  - scheduler and engine integration tests.
- `AGENT_HANDOFF_STATE.md`
  - terminal source/run/result handoff at true EOF.
- `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
  - prompt-to-artifact reconciliation at true EOF.

---

### Task 1: Split-Phase Contracts, State Machine, and Configuration

**Files:**

- Create: `tinyvllm/engine/exact_greedy_decode_burst_split_phase.py`
- Create: `tools/test_exact_greedy_decode_burst_split_phase.py`
- Modify: `tinyvllm/config.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces:
  `ExactBurstPublicationTicket`,
  `ExactBurstPhaseTransfer`,
  `ExactGreedyDecodeBurstSplitResult`,
  `ExactBurstSplitPhaseTransaction`,
  `build_exact_burst_publication_tickets(...)`, and
  `validate_exact_burst_split_result(...)`.
- Consumes: parent lease scalar fields and SHA-256 identity; no PyTorch import
  is allowed in the contract module.

- [ ] **Step 1: Write failing ticket, result, transition, and config tests**

Add dependency-light tests with one valid parent:

```python
prefix, suffix = build_exact_burst_publication_tickets(
    parent_lease_identity_sha256="a" * 64,
    first_write_position=259,
    first_physical_slot=11 * 256 + 3,
    parent_token_count=8,
    prefix_token_count=4,
)
assert prefix.phase == "prefix"
assert prefix.phase_start_ordinal == 0
assert prefix.phase_token_count == 4
assert suffix.phase == "suffix"
assert suffix.phase_start_ordinal == 4
assert suffix.first_write_position == 263
assert suffix.first_physical_slot == 11 * 256 + 7
```

Construct a split result with replay count eight and two four-token transfer
records. Assert validation rejects:

```text
non-K8 parent
non-K4 prefix
overlap
gap
wrong parent digest
wrong phase order
wrong D2H byte count
wrong replay count
duplicate phase
```

Exercise transitions:

```python
transaction = ExactBurstSplitPhaseTransaction.create(
    parent_lease_identity_sha256="a" * 64,
    result=result,
)
transaction.mark_prefix_ready()
transaction.mark_prefix_committed()
transaction.mark_suffix_ready()
transaction.mark_suffix_committed()
assert transaction.state == "suffix_committed"
```

Assert duplicate or skipped transitions raise `ValueError`. Add configuration
tests for the strict boolean and invalid composition.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 tools/test_model_runner_spec_verify.py
```

Expected: FAIL because the module, interfaces, and config field do not exist.

- [ ] **Step 3: Implement immutable dependency-light contracts**

Implement the public records:

```python
@dataclass(frozen=True)
class ExactBurstPublicationTicket:
    parent_lease_identity_sha256: str
    phase: str
    phase_start_ordinal: int
    phase_token_count: int
    first_write_position: int
    last_write_position: int
    first_physical_slot: int
    last_physical_slot: int
    identity_sha256: str


@dataclass(frozen=True)
class ExactBurstPhaseTransfer:
    ticket: ExactBurstPublicationTicket
    mailbox_generation: int
    token_count: int
    byte_count: int
    completion: object
    mailbox: object


@dataclass(frozen=True)
class ExactGreedyDecodeBurstSplitResult:
    parent_lease_identity_sha256: str
    graph_identity_sha256: str
    replay_count: int
    prefix: ExactBurstPhaseTransfer
    suffix: ExactBurstPhaseTransfer
```

`build_exact_burst_publication_tickets(...)` must construct canonical JSON
with sorted keys and compact separators before SHA-256 hashing. Require an
eight-token parent and four-token prefix. `validate_exact_burst_split_result`
must validate ranges, digests, replay count, D2H bytes, and object identity
without touching CUDA.

Implement a mutable transaction with states:

```text
enqueued
prefix_ready
prefix_committed
suffix_ready
suffix_committed
pre_prefix_failed
post_prefix_failed
```

Every transition method checks its exact predecessor.

- [ ] **Step 4: Add strict configuration**

Add:

```python
exact_greedy_decode_burst_split_phase: bool = False
```

Validate:

```python
if not isinstance(self.exact_greedy_decode_burst_split_phase, bool):
    raise ValueError(
        "exact_greedy_decode_burst_split_phase must be a bool"
    )
if self.exact_greedy_decode_burst_split_phase:
    if not self.exact_greedy_decode_burst:
        raise ValueError(
            "split phase requires exact_greedy_decode_burst"
        )
    if self.exact_greedy_decode_burst_tokens != 8:
        raise ValueError("split phase requires K8")
    if self.exact_greedy_decode_burst_continuation:
        raise ValueError(
            "split phase cannot compose with continuation"
        )
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 tools/test_model_runner_spec_verify.py
python3 -m py_compile \
  tinyvllm/engine/exact_greedy_decode_burst_split_phase.py \
  tinyvllm/config.py
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/exact_greedy_decode_burst_split_phase.py \
  tinyvllm/config.py \
  tools/test_exact_greedy_decode_burst_split_phase.py \
  tools/test_model_runner_spec_verify.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): define split-phase burst contracts" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 2: Event-Ordered Pinned Mailboxes and Split Graph Replay

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst_split_phase.py`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_exact_greedy_decode_burst_split_phase.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**

- Consumes:
  `ExactBurstPublicationTicket` and
  `ExactGreedyDecodeBurstSplitResult`.
- Produces:
  `ExactBurstSplitPhaseMailboxBackend.enqueue_phase(...)`,
  `ExactBurstPhaseTransfer.wait_tokens()`, and
  `ExactGreedyDecodeBurstGraph.replay_split_phase(...)`.

- [ ] **Step 1: Write failing fake-stream and graph-order tests**

Use fake tensors, streams, and events to record operations. Require this exact
ordering:

```text
replay:0
replay:1
replay:2
replay:3
record:prefix_compute_done
copy_wait:prefix_compute_done
copy:history[0:4]->mailbox_a
record:prefix_copy_done
replay:4
replay:5
replay:6
replay:7
record:suffix_compute_done
copy_wait:suffix_compute_done
copy:history[4:8]->mailbox_b
record:suffix_copy_done
```

Assert:

- prefix `wait_tokens()` waits only on `prefix_copy_done`;
- suffix `wait_tokens()` waits only on `suffix_copy_done`;
- a second transaction cannot reuse either mailbox generation;
- releasing both transfers permits mailbox reuse;
- any event/copy exception after replay starts quarantines the graph;
- ordinary `replay(...)` behavior and output remain unchanged.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: FAIL because the mailbox backend and split replay do not exist.

- [ ] **Step 3: Implement the mailbox backend**

Add a backend initialized with injected CUDA operations:

```python
class ExactBurstSplitPhaseMailboxBackend:
    def __init__(
        self,
        *,
        copy_stream,
        prefix_mailbox,
        suffix_mailbox,
        event_factory,
        current_stream,
        stream_context,
    ):
        ...

    def enqueue_phase(
        self,
        *,
        ticket,
        token_slice,
    ) -> ExactBurstPhaseTransfer:
        producer = self._event_factory()
        producer.record(self._current_stream())
        self._copy_stream.wait_event(producer)
        with self._stream_context(self._copy_stream):
            mailbox.copy_(token_slice, non_blocking=True)
        completion = self._event_factory()
        completion.record(self._copy_stream)
        return ExactBurstPhaseTransfer(...)
```

Allocate exactly two reusable pinned CPU `torch.int64` tensors of length four
and one dedicated CUDA copy stream in `ModelRunner`. Track mailbox ownership
with monotonically increasing generations. `wait_tokens()` synchronizes only
its completion event, converts exactly four values, and marks that transfer
host-ready without releasing ownership.

- [ ] **Step 4: Implement split replay**

Add:

```python
def replay_split_phase(
    self,
    *,
    lease,
    initial_token,
    block_table_factory,
    mailbox_backend,
    graph_generation,
    rank,
    tensor_parallel_size,
    expected_graph_identity_sha256,
) -> ExactGreedyDecodeBurstSplitResult | ExactGreedyDecodeBurstFallback:
    ...
```

Reuse the ordinary pre-replay validation and cold-bind helpers. Require
exactly eight authorized tokens and continuation disabled. Enqueue four
replays, prefix transfer, four replays, and suffix transfer. Do not call
`.tolist()` or synchronize the whole device. On any exception after the first
replay, invalidate continuation, quarantine, release mailbox ownership only
after the backend reaches a safe point, and re-raise.

In `ModelRunner._run_exact_greedy_decode_burst`, select
`replay_split_phase(...)` only when the strict config flag is true.

- [ ] **Step 5: Run focused and adjacent tests**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_model_runner_spec_verify.py
python3 -m py_compile \
  tinyvllm/engine/exact_greedy_decode_burst_split_phase.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/exact_greedy_decode_burst_split_phase.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/model_runner.py \
  tools/test_exact_greedy_decode_burst_split_phase.py \
  tools/test_exact_greedy_decode_burst.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): enqueue split-phase K8 token transfers" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 3: Parent-Lease Prefix and Suffix Scheduler Transactions

**Files:**

- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst_split_phase.py`
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_llm_engine_exact_greedy_decode_burst.py`

**Interfaces:**

- Consumes:
  `ExactGreedyDecodeBurstSplitResult`,
  `ExactBurstPublicationTicket`, and four-token phase tuples.
- Produces:
  `Scheduler.prepare_exact_greedy_decode_burst_phase_commit(...)` and
  phase-aware `PreparedSchedulerPostprocess`.

- [ ] **Step 1: Write failing scheduler lifecycle tests**

Test one K8 parent lease:

```python
prefix_prepared = scheduler.prepare_exact_greedy_decode_burst_phase_commit(
    (sequence,),
    lease,
    split_result,
    phase="prefix",
    tokens=(10, 11, 12, 13),
    host_visible_gap_ns=12_000_000,
)
scheduler.commit_prepared_postprocess(prefix_prepared)
assert sequence.num_completion_tokens == initial_completion + 4
assert scheduler._exact_greedy_decode_burst_pending_lease == lease
assert scheduler._exact_greedy_decode_burst_split_phase == "prefix_committed"
```

Then commit suffix and assert:

```python
assert sequence.num_completion_tokens == initial_completion + 8
assert scheduler._exact_greedy_decode_burst_pending_lease is None
assert scheduler._exact_greedy_decode_burst_split_phase == "idle"
assert summary["pending_leases"] == 0
```

Add rejection tests for suffix-before-prefix, duplicate prefix, wrong ticket,
wrong token count, schedule-generation drift, block-generation drift, and
sequence state other than initial+4 at suffix prepare.

Add rollback tests proving prefix rollback returns to the parent initial host
state and suffix rollback preserves the committed prefix.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_exact_greedy_decode_burst.py
```

Expected: FAIL because phase-aware prepare/commit does not exist.

- [ ] **Step 3: Add phase fields and validation**

Extend `ScheduledOutputRow` and `PreparedSchedulerPostprocess`:

```python
exact_burst_phase: str | None = None
exact_burst_split_result: ExactGreedyDecodeBurstSplitResult | None = None
```

Store scheduler phase as one of:

```text
idle
enqueued
prefix_committed
```

`prepare_exact_greedy_decode_burst_phase_commit(...)` validates the parent
lease, split result, phase ticket, four tokens, sequence state, block
identities, and unchanged scheduler generation. It sets the materialized
boundary from the phase ticket, not from the parent final position.

- [ ] **Step 4: Implement phase-aware commit accounting**

Prefix commit appends four tokens and records:

```text
prefix_commits += 1
prefix_committed_tokens += 4
pending_leases unchanged
split phase = prefix_committed
```

Suffix commit appends four tokens and records:

```text
suffix_commits += 1
suffix_committed_tokens += 4
commits += 1
committed_tokens += 8
pending_leases -= 1
pending parent lease = None
split phase = idle
```

Add stable counters for tickets, D2H calls/bytes, phase waits, drains, and
failure classes. Keep ordinary one-phase accounting unchanged.

- [ ] **Step 5: Run focused and adjacent tests**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 -m py_compile \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/exact_greedy_decode_burst_split_phase.py \
  tinyvllm/engine/scheduler.py
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/exact_greedy_decode_burst_split_phase.py \
  tinyvllm/engine/scheduler.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): commit K8 bursts in ordered phases" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 4: Engine Prefix Publication and Pre-Schedule Suffix Drain

**Files:**

- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_llm_engine_exact_greedy_decode_burst.py`

**Interfaces:**

- Consumes:
  `ExactGreedyDecodeBurstSplitResult`,
  `ExactBurstSplitPhaseTransaction`, and scheduler phase commit.
- Produces:
  `LLMEngine._drain_exact_burst_split_phase_suffix(...)` and one pending
  transaction slot.

- [ ] **Step 1: Write failing engine-order and failure tests**

Use a fake scheduler with call recording. After a prefix commit, call the next
engine step and assert:

```python
assert calls == [
    "suffix.wait_tokens",
    "scheduler.prepare_suffix",
    "scheduler.commit_suffix",
]
assert "scheduler.schedule" not in calls
```

Assert the prefix path waits only for prefix completion, commits exactly four
tokens, stores the transaction, and returns without suffix synchronization.

Add tests for:

- no second split transaction while one is pending;
- suffix ticket mismatch is terminal;
- suffix wait failure never falls through to `schedule()`;
- suffix commit rollback preserves prefix;
- pre-prefix failure waits for a GPU-safe point and publishes zero tokens;
- cancellation after prefix performs a measured suffix drain first;
- ordinary one-phase K8 behavior remains unchanged.

- [ ] **Step 2: Run focused test and verify RED**

Run:

```bash
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
```

Expected: FAIL because the engine has no pending transaction or suffix drain.

- [ ] **Step 3: Add pending transaction ownership**

Initialize:

```python
self._exact_burst_split_phase_transaction = None
```

When model runner returns a split result:

1. validate it against the parent lease;
2. create transaction state `enqueued`;
3. wait only for prefix transfer;
4. mark prefix ready;
5. prepare and commit prefix;
6. mark prefix committed;
7. retain the transaction for the next engine call;
8. release neither mailbox nor parent lease.

- [ ] **Step 4: Drain suffix before scheduling**

At the start of `step()`, before `scheduler.schedule()`:

```python
pending = self._exact_burst_split_phase_transaction
if pending is not None:
    return self._drain_exact_burst_split_phase_suffix(
        pending,
        completion_only=completion_only,
    )
```

The drain waits for suffix completion, validates and commits suffix, releases
both mailbox generations through `ModelRunner`, clears the transaction, and
returns an observation row with:

```text
phase_published = suffix
phase_token_count = 4
pending_suffix = false
scheduler_schedule_calls = 0
```

Any post-enqueue failure invalidates continuation, quarantines the graph,
terminally fails the lease, reaches a GPU-safe point, and re-raises.

- [ ] **Step 5: Run focused and adjacent tests**

Run:

```bash
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 tools/test_llm_engine_speculative_dispatch.py
python3 -m py_compile \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/model_runner.py
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/model_runner.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): drain split burst suffixes before scheduling" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 5: Four-Arm Performance and Correctness Producer

**Files:**

- Create: `tools/profile_exact_burst_split_phase.py`
- Create: `tools/test_profile_exact_burst_split_phase.py`

**Interfaces:**

- Consumes: existing exact-burst profiler helpers and engine observation
  fields.
- Produces: 60 performance rows, complete correctness rows, float32 logit
  sidecars, workload manifest, and source manifest.

- [ ] **Step 1: Write failing frozen-matrix and inventory tests**

Assert the exact arms:

```python
ARMS = (
    "host_greedy",
    "decode_burst_k4",
    "decode_burst_k8",
    "decode_burst_k8_split_phase",
)
```

Assert three buckets, five repetitions, and 60 performance rows. For each
split request require:

```text
one parent lease
one prefix ticket
one suffix ticket
prefix phase count = 4
suffix phase count = 4
total replay count = 8
prefix D2H calls = 1
suffix D2H calls = 1
prefix D2H bytes = 32
suffix D2H bytes = 32
pending suffix at prefix row
no pending suffix at suffix row
zero unexpected scheduler calls during suffix drain
```

Assert correctness rows and logit sidecars are complete for all buckets and
arms.

- [ ] **Step 2: Run focused test and verify RED**

Run:

```bash
python3 tools/test_profile_exact_burst_split_phase.py
```

Expected: FAIL because the profiler does not exist.

- [ ] **Step 3: Implement the producer**

Clone only reusable workload/source-manifest helpers from
`profile_exact_greedy_decode_burst.py`; do not import the gate. Configure each
arm explicitly:

```python
"host_greedy": dict(exact=False, width=4, split=False)
"decode_burst_k4": dict(exact=True, width=4, split=False)
"decode_burst_k8": dict(exact=True, width=8, split=False)
"decode_burst_k8_split_phase": dict(
    exact=True,
    width=8,
    split=True,
)
```

Aggregate host-visible gaps from both prefix and suffix publication rows.
Write stable phase inventories, D2H/event/mailbox costs, TTFT, TPOT, E2E,
throughput, allocated memory, and reserved memory.

- [ ] **Step 4: Run focused and adjacent profiler tests**

Run:

```bash
python3 tools/test_profile_exact_burst_split_phase.py
python3 tools/test_profile_exact_greedy_decode_burst.py
python3 tools/test_profile_exact_burst_continuation_epoch.py
python3 -m py_compile tools/profile_exact_burst_split_phase.py
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tools/profile_exact_burst_split_phase.py \
  tools/test_profile_exact_burst_split_phase.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add split-phase K8 evidence producer" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 6: Producer Gate and Independent Verifier

**Files:**

- Create: `tools/exact_burst_split_phase_gate.py`
- Create: `tools/test_exact_burst_split_phase_gate.py`
- Create: `tools/exact_burst_split_phase_verify.py`
- Create: `tools/test_exact_burst_split_phase_verify.py`

**Interfaces:**

- Consumes: profiler artifacts only.
- Produces:
  `gate.json`, `comparison.json`, `summary.json`,
  `independent-verification.json`, and `manifest.sha256`.

- [ ] **Step 1: Write failing threshold and precedence tests**

Create fixtures for:

```text
GO_EXACT_BURST_SPLIT_PHASE
NO_GO_EXACT_BURST_SPLIT_PHASE_CORRECTNESS
NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE
INCOMPLETE_EXACT_BURST_SPLIT_PHASE_EVIDENCE
```

Require correctness failure to take precedence over performance. Test exact
boundaries:

```text
median TPOT regression <= 2%
throughput regression <= 2%
TTFT/E2E regression <= 3%
reserved memory regression <= 3%
max gap <= 60% of K8
median max gap <= 3% regression versus K4
bucket median/P95 TPOT regression <= 3%
```

Add tamper tests for rows, sidecars, source commit, dirty patch, phase
inventory, and producer/verifier disagreement.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 tools/test_exact_burst_split_phase_gate.py
python3 tools/test_exact_burst_split_phase_verify.py
```

Expected: FAIL because gate and verifier do not exist.

- [ ] **Step 3: Implement producer gate**

The producer gate validates manifests, reconstructs paired metrics, verifies
tokens/text/logits, checks every lifecycle inventory, applies frozen
thresholds, writes deterministic JSON, and hashes every evidence file.

Use nearest-rank P95, median of paired request metrics, and percent regression:

```python
regression_pct = 100.0 * (candidate - baseline) / baseline
improvement_pct = 100.0 * (baseline - candidate) / baseline
```

- [ ] **Step 4: Implement independent verifier**

The verifier must not import `exact_burst_split_phase_gate`. It independently
parses JSONL, float32 sidecars, and manifests; reconstructs all counts and
metrics; reapplies thresholds; and requires exact agreement on classification
and metric values within `1e-9`.

- [ ] **Step 5: Run focused and adjacent gate suites**

Run:

```bash
python3 tools/test_exact_burst_split_phase_gate.py
python3 tools/test_exact_burst_split_phase_verify.py
python3 tools/test_exact_greedy_decode_burst_gate.py
python3 tools/test_exact_greedy_decode_burst_verify.py
python3 tools/test_exact_burst_continuation_epoch_gate.py
python3 tools/test_exact_burst_continuation_epoch_verify.py
python3 -m py_compile \
  tools/exact_burst_split_phase_gate.py \
  tools/exact_burst_split_phase_verify.py
```

Expected: all commands exit zero.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tools/exact_burst_split_phase_gate.py \
  tools/test_exact_burst_split_phase_gate.py \
  tools/exact_burst_split_phase_verify.py \
  tools/test_exact_burst_split_phase_verify.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): gate split-phase K8 benefit and cost" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 7: Source-Bound Remote Controller

**Files:**

- Create: `tools/run_exact_burst_split_phase_remote.py`
- Create: `tools/test_run_exact_burst_split_phase_remote.py`

**Interfaces:**

- Consumes: source commit, immutable run tag, model tier, GPU admission
  thresholds, profiler, gate, and verifier.
- Produces: remote workspace, local artifact copy, controller preflight,
  runner log, completion record, and remote exit code.

- [ ] **Step 1: Write failing controller safety tests**

Require:

- source commit equals pushed `origin/feat/kv-sparse-attention` head;
- dirty patch hash is explicit;
- run tag cannot already exist locally or remotely;
- Kerberos TTL fails fast before upload;
- remote root is exactly under the approved `/data00/home/sitian` prefix;
- no `/`, `/tmp`, `/private/tmp`, or retired checkout path;
- GPU memory `<=1024 MiB`, utilization `<=5%`, and no compute process;
- controller waits and automatically launches after admission;
- no unrelated process is killed;
- worker exit, producer gate, verifier, manifest, and artifact copy are all
  required for controller success.

- [ ] **Step 2: Run focused test and verify RED**

Run:

```bash
python3 tools/test_run_exact_burst_split_phase_remote.py
```

Expected: FAIL because the controller does not exist.

- [ ] **Step 3: Implement by reusing the proven remote base**

Use helpers from `run_staged_inference_benchmark_remote.py` and the exact
burst controller. The worker command must run, in order:

```text
profile_exact_burst_split_phase.py
exact_burst_split_phase_gate.py
exact_burst_split_phase_verify.py
sha256 manifest verification
```

Write runtime, cache, temporary, and artifact paths only under:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

- [ ] **Step 4: Run focused and adjacent controller tests**

Run:

```bash
python3 tools/test_run_exact_burst_split_phase_remote.py
python3 tools/test_run_exact_greedy_decode_burst_remote.py
python3 tools/test_run_exact_burst_continuation_epoch_remote.py
python3 -m py_compile tools/run_exact_burst_split_phase_remote.py
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tools/run_exact_burst_split_phase_remote.py \
  tools/test_run_exact_burst_split_phase_remote.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): add split-phase remote hardware gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

---

### Task 8: Full Local Verification and Qwen3-0.6B Hardware Gate

**Files:**

- No source edits before verification.
- Runtime artifacts:
  `artifacts/exact_burst_split_phase/<fresh-run-tag>/`.

**Interfaces:**

- Consumes: pushed source commit from Tasks 1 through 7.
- Produces: source-bound producer/verifier result and formal Stage-1
  classification.

- [ ] **Step 1: Run the full focused local suite**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_profile_exact_burst_split_phase.py
python3 tools/test_exact_burst_split_phase_gate.py
python3 tools/test_exact_burst_split_phase_verify.py
python3 tools/test_run_exact_burst_split_phase_remote.py
python3 tools/test_exact_greedy_decode_burst_gate.py
python3 tools/test_exact_greedy_decode_burst_verify.py
python3 tools/test_exact_burst_continuation_epoch_gate.py
python3 tools/test_exact_burst_continuation_epoch_verify.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 2: Confirm source and authentication**

Run:

```bash
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
klist
```

Expected: local and remote SHA match; Kerberos has enough TTL for the complete
gate. Do not refresh credentials automatically.

- [ ] **Step 3: Launch one immutable Qwen3-0.6B run**

Use a fresh tag:

```bash
TINYLLMFORGE_SSH_CONTROL_PATH=/tmp/ssh-sitian-10.232.195.203 \
python3 tools/run_exact_burst_split_phase_remote.py \
  --run-tag <fresh-run-tag> \
  --model-tier qwen3-0.6b \
  --source-commit <exact-pushed-head> \
  --gpu-wait-timeout-seconds 28800 \
  --gpu-poll-interval-seconds 60
```

The local controller must monitor GPU admission and launch automatically.

- [ ] **Step 4: Verify complete evidence**

Require:

```text
60/60 performance rows
all correctness rows
all logit sidecars
gate.json
comparison.json
summary.json
independent-verification.json
manifest.sha256
producer/verifier agreement
remote_exitcode = 0
```

If credentials expire after the worker completes, preserve the remote run and
resume only artifact retrieval and verification after the user refreshes
Kerberos. Do not rerun the benchmark.

- [ ] **Step 5: Classify without threshold changes**

Use only:

```text
GO_EXACT_BURST_SPLIT_PHASE
NO_GO_EXACT_BURST_SPLIT_PHASE_CORRECTNESS
NO_GO_EXACT_BURST_SPLIT_PHASE_PERFORMANCE
INCOMPLETE_EXACT_BURST_SPLIT_PHASE_EVIDENCE
```

Report token/logit correctness, TPOT, throughput, TTFT, E2E, memory,
host-visible gap, two D2H costs, event count, pinned bytes, pending suffix,
and cancellation-drain cost.

---

### Task 9: Reconciliation, Final Verification, Commit, and Push

**Files:**

- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

**Interfaces:**

- Consumes: immutable hardware artifacts and exact source commit.
- Produces: terminal audit/handoff entries and final pushed documentation
  commit.

- [ ] **Step 1: Append the exact evidence boundary**

Record:

- source commit and run tag;
- model, GPU, workload, row counts, and artifact paths;
- producer and verifier classifications;
- benefit and cost metrics;
- default-disabled state;
- whether Qwen3-8B promotion is authorized;
- continuation r4 remains separate incomplete evidence until recovered.

- [ ] **Step 2: Run final verification**

Run:

```bash
python3 tools/test_exact_greedy_decode_burst_split_phase.py
python3 tools/test_exact_greedy_decode_burst.py
python3 tools/test_llm_engine_exact_greedy_decode_burst.py
python3 tools/test_profile_exact_burst_split_phase.py
python3 tools/test_exact_burst_split_phase_gate.py
python3 tools/test_exact_burst_split_phase_verify.py
python3 tools/test_run_exact_burst_split_phase_remote.py
git diff --check -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
```

Expected: all commands exit zero.

- [ ] **Step 3: Commit exact reconciliation paths**

```bash
git add -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "docs(perf): record split-phase K8 gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

- [ ] **Step 4: Verify remote branch and clean task paths**

Run:

```bash
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git status --short -- \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/exact_greedy_decode_burst_split_phase.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_exact_greedy_decode_burst_split_phase.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/profile_exact_burst_split_phase.py \
  tools/test_profile_exact_burst_split_phase.py \
  tools/exact_burst_split_phase_gate.py \
  tools/test_exact_burst_split_phase_gate.py \
  tools/exact_burst_split_phase_verify.py \
  tools/test_exact_burst_split_phase_verify.py \
  tools/run_exact_burst_split_phase_remote.py \
  tools/test_run_exact_burst_split_phase_remote.py \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
```

Expected: local and remote SHA match; task source/document paths are clean.
