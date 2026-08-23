# Exact-Burst Lease-Local Delta Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's active user constraint forbids subagents and additional worktrees, so execute this plan inline in the authoritative checkout with the executing-plans workflow.

**Goal:** Replace context-length-scaled generic scheduler journaling and full block-table publication scans with a bounded lease-local delta transaction for eligible non-terminal four-token split phases, then prove its correctness, transactional safety, host-side scaling benefit, and real-GPU cost/benefit.

**Architecture:** Keep every existing exact-burst lease, block-identity, split-result, publication-ticket, token, and output-budget check. After those checks, select a default-off `ExactBurstPhaseDeltaJournal` only for a single K8 split prefix or non-terminal suffix whose mutations are bounded to one sequence and one lease-owned write block; otherwise count a stable fallback reason and retain `SchedulerPostprocessJournal`. Build an immutable one-block publication plan during prepare, before any mutation, then apply it through a narrow `BlockManager.publish_lease_write_block()` API so the delta path computes at most one hash and retains rollback authority even if commit fails immediately after hash registration.

**Tech Stack:** Python 3, dataclasses, pytest, xxhash-backed prefix-cache metadata, TinyLLMForge scheduler and block manager, JSON/JSONL benchmark artifacts, SSH source-bound remote runner, Qwen3-0.6B on one clean A100.

## Global Constraints

- The only authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`, which resolves to `/Users/bytedance/dev/TinyLLMForge`.
- Stay on `feat/kv-sparse-attention`; do not create a worktree or use a subagent.
- Preserve every unrelated dirty or untracked file. Stage only the exact paths named by each commit step.
- Use `git -c core.hooksPath=/dev/null commit`; every commit has exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- The feature is default-disabled and requires exact burst, split phase, and K8. Ragged coalescing may be either enabled or disabled.
- The fast path is restricted to TP1/rank 0, batch size 1, completion-only greedy decode, `temperature == 0`, `ignore_eos == true`, one K8 parent lease, and a four-token prefix or non-terminal suffix.
- Preserve full pending-lease and complete block-table identity revalidation. The performance claim is only that journal capture and publication no longer scale with sequence context length.
- Prefix is eligible only when it remains non-terminal. Suffix is eligible only when `sequence.num_completion_tokens + 4 < sequence.max_tokens`.
- Terminal suffix, ordinary decode, one-phase exact burst, ragged K2-K4 commits, prefill, mixed batches, multi-sequence work, uncertain publication state, and completion release continue to use the generic journal.
- Delta rollback uses `del sequence.token_ids[original_length:]`; it never copies the complete original token list.
- Delta publication validates the block-table index, block ID, generation, predecessor authority, materialized boundary, and prior unpublished state. It computes at most one hash and registers at most one block.
- Preserve duplicate-hash and primary-hash indexes exactly across rollback.
- Preserve the existing commit exception boundary and terminal `SchedulerPostprocessRollbackError` behavior.
- Do not change graph replay count, target-model forward count, D2H calls or bytes, split mailbox ownership, CUDA events, or token output.
- Hardware gate inventory is fixed at 60 performance rows and 24 correctness rows with interleaved paired order and dual independent verification.
- GO requires exact outputs/logits, lifecycle counter authority, identical forward/replay/D2H inventories, at least 50% long-context phase-prepare median and P95 improvement, no more than 3% regression in shorter-context prepare, aggregate TPOT median/P95, TTFT, E2E, or throughput, and no more than 1% reserved-memory regression.
- Remote task data may be written only below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- GPU admission requires memory `<=1024 MiB`, utilization `<=5%`, and no compute process. Do not terminate or interfere with external GPU work.
- Do not run `kinit`; fail fast when Kerberos lifetime is insufficient.
- A failed, interrupted, or unstarted run tag is never reused.
- The source commit for a remote run must be the already-pushed branch HEAD.
- Every final performance statement reports both benefit and cost. A neutral TPOT result is reported as end-to-end neutral, not hidden.

---

## File Map

- Modify `tinyvllm/config.py`: declare and fail-closed validate the default-off feature flag.
- Modify `tinyvllm/engine/exact_greedy_decode_burst.py`: own lifecycle and fallback counters in `ExactGreedyDecodeBurstStats`.
- Modify `tinyvllm/engine/block_manager.py`: define the one-block publication result and exact lease-owned publication helper.
- Modify `tinyvllm/engine/scheduler.py`: define the delta journal, eligibility/fallback selection, journal-polymorphic commit/rollback, and one-block publication dispatch.
- Modify `tools/test_model_runner_spec_verify.py`: configuration contract tests.
- Modify `tools/test_scheduler_prepared_postprocess.py`: eligibility, publication, lifecycle, fault-injection, and rollback tests.
- Modify `tools/test_llm_engine_exact_greedy_decode_burst.py`: engine-level default-off and split lifecycle coverage with the feature enabled.
- Create `tools/profile_exact_burst_lease_local_delta_journal.py`: deterministic CPU context-scaling profiler and artifact producer.
- Create `tools/test_profile_exact_burst_lease_local_delta_journal.py`: profiler schema, production-binding, and complexity tests.
- Create `tools/exact_burst_lease_local_delta_journal_gate.py`: fixed 60/24 paired gate and classification logic.
- Create `tools/test_exact_burst_lease_local_delta_journal_gate.py`: row inventory, threshold, parity, and negative-classification tests.
- Create `tools/exact_burst_lease_local_delta_journal_verify.py`: independent artifact verifier.
- Create `tools/test_exact_burst_lease_local_delta_journal_verify.py`: tamper and incomplete-evidence tests.
- Create `tools/run_exact_burst_lease_local_delta_journal_remote.py`: source-bound local controller, remote worker launch, clean-GPU monitoring, artifact retrieval, and dual verification.
- Create `tools/test_run_exact_burst_lease_local_delta_journal_remote.py`: controller path, source, GPU, Kerberos, tag, and process-safety tests.
- Modify `AGENT_HANDOFF_STATE.md`: final source SHA, commands, artifact paths, measured benefit/cost, and remaining boundary.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`: prompt-to-artifact reconciliation and final classification.

### Task 1: Add the Default-Off Configuration Contract and Statistics Surface

**Files:**
- Modify: `tinyvllm/config.py:40-45`
- Modify: `tinyvllm/config.py:213-276`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py:827-1235`
- Test: `tools/test_model_runner_spec_verify.py:5836-6000`

**Interfaces:**
- Produces: `Config.exact_greedy_decode_burst_lease_local_delta_journal: bool`
- Produces: `ExactGreedyDecodeBurstStats.record_lease_local_delta_journal_attempt() -> None`
- Produces: `ExactGreedyDecodeBurstStats.record_lease_local_delta_journal_capture() -> None`
- Produces: `ExactGreedyDecodeBurstStats.record_lease_local_delta_journal_commit(*, published_blocks: int) -> None`
- Produces: `ExactGreedyDecodeBurstStats.record_lease_local_delta_journal_rollback() -> None`
- Produces: `ExactGreedyDecodeBurstStats.record_lease_local_delta_journal_fallback(reason: str) -> None`
- Produces summary keys: `lease_local_delta_journal_attempts`, `lease_local_delta_journal_captures`, `lease_local_delta_journal_commits`, `lease_local_delta_journal_rollbacks`, `lease_local_delta_journal_published_blocks`, and `lease_local_delta_journal_fallback_counts`

- [ ] **Step 1: Add RED configuration tests**

Append these assertions to `test_exact_greedy_decode_burst_config_is_strict_and_default_off()` and keep the existing exact-burst checks:

```python
assert (
    fields[
        "exact_greedy_decode_burst_lease_local_delta_journal"
    ].default
    is False
)
for invalid in (None, 0, 1, "true"):
    with pytest.raises(
        ValueError,
        match=(
            "^exact_greedy_decode_burst_lease_local_delta_journal "
            "must be a bool$"
        ),
    ):
        Config(
            model=model,
            exact_greedy_decode_burst_lease_local_delta_journal=invalid,
        )
for overrides, message in (
    (
        {
            "exact_greedy_decode_burst": False,
            "exact_greedy_decode_burst_split_phase": True,
            "exact_greedy_decode_burst_tokens": 8,
        },
        "lease-local delta journal requires exact_greedy_decode_burst",
    ),
    (
        {
            "exact_greedy_decode_burst": True,
            "exact_greedy_decode_burst_split_phase": False,
            "exact_greedy_decode_burst_tokens": 8,
        },
        "lease-local delta journal requires split phase",
    ),
    (
        {
            "exact_greedy_decode_burst": True,
            "exact_greedy_decode_burst_split_phase": True,
            "exact_greedy_decode_burst_tokens": 4,
        },
        "lease-local delta journal requires K8",
    ),
):
    with pytest.raises(ValueError, match=f"^{message}$"):
        Config(
            model=model,
            **overrides,
            exact_greedy_decode_burst_lease_local_delta_journal=True,
        )
for ragged in (False, True):
    enabled = Config(
        model=model,
        exact_greedy_decode_burst=True,
        exact_greedy_decode_burst_split_phase=True,
        exact_greedy_decode_burst_ragged_coalescing=ragged,
        exact_greedy_decode_burst_tokens=8,
        exact_greedy_decode_burst_lease_local_delta_journal=True,
    )
    assert (
        enabled.exact_greedy_decode_burst_lease_local_delta_journal
        is True
    )
```

- [ ] **Step 2: Run the focused configuration test and preserve RED evidence**

Run:

```bash
pytest -q tools/test_model_runner_spec_verify.py::test_exact_greedy_decode_burst_config_is_strict_and_default_off
```

Expected: FAIL because the dataclass field does not exist.

- [ ] **Step 3: Add the configuration field and validation**

Add beside the other exact-burst fields:

```python
exact_greedy_decode_burst_lease_local_delta_journal: bool = False
```

Add after validating `exact_greedy_decode_burst_ragged_coalescing`:

```python
if not isinstance(
    self.exact_greedy_decode_burst_lease_local_delta_journal,
    bool,
):
    raise ValueError(
        "exact_greedy_decode_burst_lease_local_delta_journal "
        "must be a bool"
    )
if self.exact_greedy_decode_burst_lease_local_delta_journal:
    if not self.exact_greedy_decode_burst:
        raise ValueError(
            "lease-local delta journal requires "
            "exact_greedy_decode_burst"
        )
    if not self.exact_greedy_decode_burst_split_phase:
        raise ValueError(
            "lease-local delta journal requires split phase"
        )
    if self.exact_greedy_decode_burst_tokens != 8:
        raise ValueError(
            "lease-local delta journal requires K8"
        )
```

- [ ] **Step 4: Add RED statistics tests**

In the existing stats test, exercise exactly one lifecycle:

```python
stats.record_lease_local_delta_journal_attempt()
stats.record_lease_local_delta_journal_capture()
stats.record_lease_local_delta_journal_commit(
    published_blocks=1,
)
stats.record_lease_local_delta_journal_rollback()
stats.record_lease_local_delta_journal_fallback(
    "terminal_suffix",
)
snapshot = stats.summary()
assert snapshot["lease_local_delta_journal_attempts"] == 1
assert snapshot["lease_local_delta_journal_captures"] == 1
assert snapshot["lease_local_delta_journal_commits"] == 1
assert snapshot["lease_local_delta_journal_rollbacks"] == 1
assert snapshot["lease_local_delta_journal_published_blocks"] == 1
assert snapshot["lease_local_delta_journal_fallback_counts"] == {
    "terminal_suffix": 1,
}
```

Expected focused result before implementation: FAIL with a missing method.

- [ ] **Step 5: Implement validated statistics fields and methods**

Add dataclass fields:

```python
lease_local_delta_journal_attempts: int = 0
lease_local_delta_journal_captures: int = 0
lease_local_delta_journal_commits: int = 0
lease_local_delta_journal_rollbacks: int = 0
lease_local_delta_journal_published_blocks: int = 0
lease_local_delta_journal_fallback_counts: dict[str, int] = field(
    default_factory=dict
)
```

Add methods:

```python
def record_lease_local_delta_journal_attempt(self) -> None:
    self.lease_local_delta_journal_attempts += 1

def record_lease_local_delta_journal_capture(self) -> None:
    self.lease_local_delta_journal_captures += 1

def record_lease_local_delta_journal_commit(
    self,
    *,
    published_blocks: int,
) -> None:
    _require_non_negative_int(
        published_blocks,
        "published_blocks",
    )
    if published_blocks > 1:
        raise ValueError(
            "published_blocks must be at most one"
        )
    self.lease_local_delta_journal_commits += 1
    self.lease_local_delta_journal_published_blocks += (
        published_blocks
    )

def record_lease_local_delta_journal_rollback(self) -> None:
    self.lease_local_delta_journal_rollbacks += 1

def record_lease_local_delta_journal_fallback(
    self,
    reason: str,
) -> None:
    reason = _require_reason(
        reason,
        "lease-local delta journal fallback reason",
    )
    counts = self.lease_local_delta_journal_fallback_counts
    counts[reason] = counts.get(reason, 0) + 1
```

Expose copies in `summary()`:

```python
"lease_local_delta_journal_attempts": (
    self.lease_local_delta_journal_attempts
),
"lease_local_delta_journal_captures": (
    self.lease_local_delta_journal_captures
),
"lease_local_delta_journal_commits": (
    self.lease_local_delta_journal_commits
),
"lease_local_delta_journal_rollbacks": (
    self.lease_local_delta_journal_rollbacks
),
"lease_local_delta_journal_published_blocks": (
    self.lease_local_delta_journal_published_blocks
),
"lease_local_delta_journal_fallback_counts": dict(
    sorted(
        self.lease_local_delta_journal_fallback_counts.items()
    )
),
```

- [ ] **Step 6: Run focused GREEN tests**

Run:

```bash
pytest -q \
  tools/test_model_runner_spec_verify.py::test_exact_greedy_decode_burst_config_is_strict_and_default_off \
  tools/test_exact_greedy_decode_burst.py
```

Expected: all selected tests PASS.

- [ ] **Step 7: Commit the configuration and statistics contract**

```bash
git add \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit -m "feat(runtime): add delta journal contract" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing only the three named paths.

### Task 2: Add Exact One-Block Prefix-Cache Publication

**Files:**
- Modify: `tinyvllm/engine/block_manager.py:249-365`
- Modify: `tinyvllm/engine/block_manager.py:1861-1880`
- Test: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**
- Produces: immutable `LeaseWriteBlockPublicationPlan`
- Produces: `BlockManager.plan_lease_write_block_publication(seq: Sequence, *, appended_tokens: tuple[int, ...], block_table_index: int, expected_block_id: int, expected_generation: int, materialized_tokens: int, predecessor_hash: int | None) -> LeaseWriteBlockPublicationPlan`
- Produces: `BlockManager.publish_lease_write_block(seq: Sequence, *, plan: LeaseWriteBlockPublicationPlan) -> bool`
- `LeaseWriteBlockPublicationPlan` fields: `will_publish: bool`, `block_table_index: int`, `block_id: int`, `block_generation: int`, `materialized_tokens: int`, `predecessor_block_id: int | None`, `predecessor_hash: int | None`, `planned_block_hash: int | None`, `planned_block_token_ids: tuple[int, ...]`, `prior_block_hash: int`, `prior_block_token_ids: tuple[int, ...]`, `prior_primary_block_id: int | None`, `prior_duplicate_block_ids: frozenset[int] | None`

- [ ] **Step 1: Add RED tests for partial, first-block, predecessor, and collision behavior**

Create focused tests using the existing scheduler/block-manager fixtures:

```python
def test_publish_lease_write_block_is_bounded_and_exact(
    monkeypatch,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        list(range(12)),
        max_tokens=32,
        ignore_eos=True,
    )
    manager = scheduler.block_manager
    block_index = len(sequence.block_table) - 1
    block_id = sequence.block_table[block_index]
    generation = manager.blocks[block_id].generation
    appended_tokens = (101, 102, 103, 104)
    partial = manager.plan_lease_write_block_publication(
        sequence,
        appended_tokens=appended_tokens,
        block_table_index=block_index,
        expected_block_id=block_id,
        expected_generation=generation,
        materialized_tokens=15,
        predecessor_hash=(
            manager.blocks[sequence.block_table[block_index - 1]].hash
            if block_index
            else None
        ),
    )
    assert partial.will_publish is False
    assert manager.publish_lease_write_block(
        sequence,
        plan=partial,
    ) is False
    assert manager.blocks[block_id].hash == -1

    full = manager.plan_lease_write_block_publication(
        sequence,
        appended_tokens=appended_tokens,
        block_table_index=block_index,
        expected_block_id=block_id,
        expected_generation=generation,
        materialized_tokens=16,
        predecessor_hash=(
            manager.blocks[sequence.block_table[block_index - 1]].hash
            if block_index
            else None
        ),
    )
    assert full.will_publish is True
    assert full.block_id == block_id
    assert full.prior_block_hash == -1
    for token_id in appended_tokens:
        sequence.append_token(token_id)
    assert manager.publish_lease_write_block(
        sequence,
        plan=full,
    ) is True
    assert (
        full.planned_block_hash
        == manager.blocks[block_id].hash
    )
```

Add negative assertions:

```python
with pytest.raises(RuntimeError, match="write block identity is stale"):
    manager.plan_lease_write_block_publication(
        sequence,
        appended_tokens=appended_tokens,
        block_table_index=block_index,
        expected_block_id=block_id,
        expected_generation=generation + 1,
        materialized_tokens=16,
        predecessor_hash=predecessor_hash,
    )
with pytest.raises(
    RuntimeError,
    match="predecessor hash authority is unavailable",
):
    manager.plan_lease_write_block_publication(
        sequence,
        appended_tokens=appended_tokens,
        block_table_index=block_index,
        expected_block_id=block_id,
        expected_generation=generation,
        materialized_tokens=16,
        predecessor_hash=None,
    )
```

Add a duplicate-hash fixture with a pre-existing primary and duplicate set, then assert the returned result captures both prior values and registration adds only the write block.

- [ ] **Step 2: Run the focused publication tests and preserve RED evidence**

Run:

```bash
pytest -q tools/test_scheduler_prepared_postprocess.py -k publish_lease_write_block
```

Expected: FAIL because `plan_lease_write_block_publication` is absent.

- [ ] **Step 3: Implement the immutable publication plan**

At module scope above `BlockManager`, add:

```python
@dataclass(frozen=True)
class LeaseWriteBlockPublicationPlan:
    will_publish: bool
    block_table_index: int
    block_id: int
    block_generation: int
    materialized_tokens: int
    predecessor_block_id: int | None
    predecessor_hash: int | None
    planned_block_hash: int | None
    planned_block_token_ids: tuple[int, ...]
    prior_block_hash: int
    prior_block_token_ids: tuple[int, ...]
    prior_primary_block_id: int | None
    prior_duplicate_block_ids: frozenset[int] | None
```

Import `dataclass` from `dataclasses`.

- [ ] **Step 4: Implement read-only planning before commit mutation**

Add this method without modifying `publish_full_blocks()`. It computes the possible future hash from the current partial block plus the exact four output tokens, and captures the affected hash-index entries before commit can mutate them:

```python
def plan_lease_write_block_publication(
    self,
    seq: Sequence,
    *,
    appended_tokens: tuple[int, ...],
    block_table_index: int,
    expected_block_id: int,
    expected_generation: int,
    materialized_tokens: int,
    predecessor_hash: int | None,
) -> LeaseWriteBlockPublicationPlan:
    if not isinstance(appended_tokens, tuple):
        raise ValueError("appended_tokens must be a tuple")
    if (
        isinstance(block_table_index, bool)
        or not isinstance(block_table_index, int)
        or block_table_index < 0
        or block_table_index >= len(seq.block_table)
    ):
        raise ValueError(
            "block_table_index is out of range"
        )
    if seq.block_table[block_table_index] != expected_block_id:
        raise RuntimeError("write block identity is stale")
    block = self.blocks[expected_block_id]
    if (
        block.generation != expected_generation
        or expected_block_id not in self.used_block_ids
        or block.ref_count <= 0
    ):
        raise RuntimeError("write block identity is stale")
    block_end = (block_table_index + 1) * self.block_size
    if materialized_tokens < block_end:
        return LeaseWriteBlockPublicationPlan(
            will_publish=False,
            block_table_index=block_table_index,
            block_id=expected_block_id,
            block_generation=expected_generation,
            materialized_tokens=materialized_tokens,
            predecessor_block_id=None,
            predecessor_hash=None,
            planned_block_hash=None,
            planned_block_token_ids=(),
            prior_block_hash=block.hash,
            prior_block_token_ids=tuple(block.token_ids),
            prior_primary_block_id=None,
            prior_duplicate_block_ids=None,
        )
    block_start = block_table_index * self.block_size
    existing_end = min(len(seq.token_ids), block_end)
    token_ids = (
        tuple(seq.token_ids[block_start:existing_end])
        + appended_tokens
    )
    materialized_in_block = max(
        0,
        min(materialized_tokens, block_end) - block_start,
    )
    token_ids = token_ids[:materialized_in_block]
    if len(token_ids) != self.block_size:
        return LeaseWriteBlockPublicationPlan(
            will_publish=False,
            block_table_index=block_table_index,
            block_id=expected_block_id,
            block_generation=expected_generation,
            materialized_tokens=materialized_tokens,
            predecessor_block_id=None,
            predecessor_hash=None,
            planned_block_hash=None,
            planned_block_token_ids=token_ids,
            prior_block_hash=block.hash,
            prior_block_token_ids=tuple(block.token_ids),
            prior_primary_block_id=None,
            prior_duplicate_block_ids=None,
        )
    if block.hash != -1:
        raise RuntimeError("write block is already published")
    prefix = -1
    predecessor_id = None
    if block_table_index > 0:
        predecessor_id = seq.block_table[block_table_index - 1]
        predecessor = self.blocks[predecessor_id]
        if (
            predecessor_hash is None
            or predecessor.hash != predecessor_hash
            or predecessor_hash == -1
            or (
                self.hash_to_block_id.get(predecessor_hash)
                != predecessor_id
                and predecessor_id not in self.hash_to_block_ids.get(
                    predecessor_hash,
                    (),
                )
            )
        ):
            raise RuntimeError(
                "predecessor hash authority is unavailable"
            )
        prefix = predecessor_hash
    block_hash = self.compute_hash(list(token_ids), prefix)
    prior_primary = self.hash_to_block_id.get(block_hash)
    prior_duplicates = self.hash_to_block_ids.get(block_hash)
    return LeaseWriteBlockPublicationPlan(
        will_publish=True,
        block_table_index=block_table_index,
        block_id=expected_block_id,
        block_generation=expected_generation,
        materialized_tokens=materialized_tokens,
        predecessor_block_id=predecessor_id,
        predecessor_hash=(
            predecessor_hash
            if predecessor_id is not None
            else None
        ),
        planned_block_hash=block_hash,
        planned_block_token_ids=token_ids,
        prior_block_hash=block.hash,
        prior_block_token_ids=tuple(block.token_ids),
        prior_primary_block_id=prior_primary,
        prior_duplicate_block_ids=(
            frozenset(prior_duplicates)
            if prior_duplicates is not None
            else None
        ),
    )
```

- [ ] **Step 5: Implement bounded publication from the immutable plan**

The apply helper performs no hash calculation. Because the plan and rollback metadata exist before commit begins, a fault injected immediately after `_register_cached_block()` remains recoverable:

```python
def publish_lease_write_block(
    self,
    seq: Sequence,
    *,
    plan: LeaseWriteBlockPublicationPlan,
) -> bool:
    if not isinstance(plan, LeaseWriteBlockPublicationPlan):
        raise ValueError(
            "plan must be LeaseWriteBlockPublicationPlan"
        )
    if (
        plan.block_table_index >= len(seq.block_table)
        or seq.block_table[plan.block_table_index]
        != plan.block_id
    ):
        raise RuntimeError("write block identity is stale")
    block = self.blocks[plan.block_id]
    if (
        block.generation != plan.block_generation
        or block.hash != plan.prior_block_hash
        or tuple(block.token_ids) != plan.prior_block_token_ids
    ):
        raise RuntimeError(
            "write block publication state drifted"
        )
    if not plan.will_publish:
        return False
    if plan.predecessor_block_id is not None:
        predecessor = self.blocks[plan.predecessor_block_id]
        if (
            predecessor.hash != plan.predecessor_hash
            or (
                self.hash_to_block_id.get(plan.predecessor_hash)
                != plan.predecessor_block_id
                and plan.predecessor_block_id
                not in self.hash_to_block_ids.get(
                    plan.predecessor_hash,
                    (),
                )
            )
        ):
            raise RuntimeError(
                "predecessor hash authority drifted"
            )
    current_primary = self.hash_to_block_id.get(
        plan.planned_block_hash
    )
    current_duplicates = self.hash_to_block_ids.get(
        plan.planned_block_hash
    )
    if (
        current_primary != plan.prior_primary_block_id
        or (
            frozenset(current_duplicates)
            if current_duplicates is not None
            else None
        )
        != plan.prior_duplicate_block_ids
    ):
        raise RuntimeError(
            "publication hash index authority drifted"
        )
    token_ids = tuple(seq.block(plan.block_table_index))
    if token_ids != plan.planned_block_token_ids:
        raise RuntimeError(
            "write block tokens do not match publication plan"
        )
    if plan.planned_block_hash is None:
        raise RuntimeError(
            "publishable plan is missing a block hash"
        )
    self._register_cached_block(
        plan.block_id,
        plan.planned_block_hash,
        list(plan.planned_block_token_ids),
    )
    return True
```

- [ ] **Step 6: Run publication tests and the adjacent block-manager suite**

Run:

```bash
pytest -q \
  tools/test_scheduler_prepared_postprocess.py -k "publish_lease_write_block or publish_full_blocks"
```

Expected: all selected tests PASS; existing generic publication behavior is unchanged.

- [ ] **Step 7: Commit the bounded block-manager API**

```bash
git add \
  tinyvllm/engine/block_manager.py \
  tools/test_scheduler_prepared_postprocess.py
git -c core.hooksPath=/dev/null commit -m "feat(cache): publish one lease write block" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 3: Define the Delta Journal and Fail-Closed Eligibility

**Files:**
- Modify: `tinyvllm/engine/scheduler.py:51-245`
- Modify: `tinyvllm/engine/scheduler.py:640-725`
- Modify: `tinyvllm/engine/scheduler.py:2323-2587`
- Test: `tools/test_scheduler_prepared_postprocess.py:1265-1946`

**Interfaces:**
- Produces: `ExactBurstPhaseDeltaJournal`
- Produces: `ExactBurstPhaseDeltaJournal.capture(scheduler, sequence, *, expected_block_table_identity: tuple[tuple[int, int], ...], publication_plan: LeaseWriteBlockPublicationPlan) -> ExactBurstPhaseDeltaJournal`
- Produces: `Scheduler._select_exact_burst_phase_journal(...) -> SchedulerPostprocessJournal | ExactBurstPhaseDeltaJournal`
- Stable fallback reasons: `terminal_suffix`, `write_block_position_mismatch`, `write_block_already_published`, `predecessor_hash_unavailable`, `unsupported_phase_shape`

- [ ] **Step 1: Add RED selection tests**

Parameterize three fast selections:

```python
def _make_delta_split_phase_fixture(
    monkeypatch,
    *,
    phase: str,
    prompt_length: int,
    max_tokens: int,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
                (
                    "exact_greedy_decode_burst_"
                    "lease_local_delta_journal"
                ): phase == "prefix",
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        list(range(1, prompt_length + 1)),
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
        split_phase_enabled=True,
    )
    split_result = _exact_burst_split_result(lease)
    if phase == "suffix":
        prefix = (
            scheduler.prepare_exact_greedy_decode_burst_phase_commit(
                (sequence,),
                lease,
                split_result,
                phase="prefix",
                tokens=split_result.prefix.wait_tokens(),
            )
        )
        scheduler.commit_prepared_postprocess(prefix)
        scheduler._exact_greedy_decode_burst_lease_local_delta_journal = (
            True
        )
    return scheduler, sequence, lease, split_result


@pytest.mark.parametrize(
    ("phase", "prompt_length", "max_tokens"),
    (
        ("prefix", 2, 16),
        ("suffix", 2, 16),
        ("suffix", 9, 16),
    ),
)
def test_delta_journal_selects_only_bounded_non_terminal_split_phases(
    monkeypatch,
    phase,
    prompt_length,
    max_tokens,
):
    scheduler, sequence, lease, split_result = (
        _make_delta_split_phase_fixture(
            monkeypatch,
            phase=phase,
            prompt_length=prompt_length,
            max_tokens=max_tokens,
        )
    )
    prepared = scheduler.prepare_exact_greedy_decode_burst_phase_commit(
        (sequence,),
        lease,
        split_result,
        phase=phase,
        tokens=getattr(split_result, phase).wait_tokens(),
    )
    assert isinstance(
        prepared.snapshot,
        scheduler_module.ExactBurstPhaseDeltaJournal,
    )
    assert (
        scheduler.exact_greedy_decode_burst_summary()[
            "lease_local_delta_journal_captures"
        ]
        == 1
    )
```

Parameterize generic fallback for disabled, terminal suffix, wrong phase shape, already-published write block, and missing predecessor authority. Assert the snapshot remains `SchedulerPostprocessJournal` and exactly one expected reason is counted when the feature is enabled.

- [ ] **Step 2: Run selection tests and preserve RED evidence**

Run:

```bash
pytest -q tools/test_scheduler_prepared_postprocess.py -k delta_journal_selects
```

Expected: FAIL because the delta journal type is absent.

- [ ] **Step 3: Add the delta journal state model**

Define:

```python
@dataclass
class ExactBurstPhaseDeltaJournal:
    sequence: Sequence
    token_list: list[int]
    original_length: int
    original_last_token: int
    original_num_tokens: int
    original_status: SequenceStatus
    expected_block_table_identity: tuple[tuple[int, int], ...]
    waiting_length: int
    prefilling_length: int
    running_length: int
    sequence_was_running: bool
    decode_progress_present: bool
    decode_progress_value: int | None
    last_slo_postprocess: dict
    adaptive_mixed_state: str
    adaptive_high_streak: int
    adaptive_low_streak: int
    adaptive_consecutive_mixed_steps: int
    consecutive_prefill_chunks: int
    slo_clock_invalid: bool
    slo_clock_invalid_reason: object
    last_slo_decision_now_ns: int | None
    publication_plan: LeaseWriteBlockPublicationPlan
    publication_applied: bool = False
    state: str = "active"

    @property
    def scheduled_sequences(self) -> tuple[Sequence, ...]:
        return (self.sequence,)

    @classmethod
    def capture(
        cls,
        scheduler,
        sequence: Sequence,
        *,
        expected_block_table_identity: tuple[
            tuple[int, int],
            ...,
        ],
        publication_plan: LeaseWriteBlockPublicationPlan,
    ) -> "ExactBurstPhaseDeltaJournal":
        present = (
            sequence.seq_id
            in scheduler.decode_progress_ns_by_seq_id
        )
        return cls(
            sequence=sequence,
            token_list=sequence.token_ids,
            original_length=len(sequence.token_ids),
            original_last_token=sequence.last_token,
            original_num_tokens=sequence.num_tokens,
            original_status=sequence.status,
            expected_block_table_identity=(
                expected_block_table_identity
            ),
            waiting_length=len(scheduler.waiting),
            prefilling_length=len(scheduler.prefilling),
            running_length=len(scheduler.running),
            sequence_was_running=(
                len(scheduler.running) == 1
                and scheduler.running[0] is sequence
            ),
            decode_progress_present=present,
            decode_progress_value=(
                scheduler.decode_progress_ns_by_seq_id.get(
                    sequence.seq_id
                )
            ),
            last_slo_postprocess=dict(
                scheduler._last_slo_postprocess
            ),
            adaptive_mixed_state=scheduler.adaptive_mixed_state,
            adaptive_high_streak=scheduler.adaptive_high_streak,
            adaptive_low_streak=scheduler.adaptive_low_streak,
            adaptive_consecutive_mixed_steps=(
                scheduler.adaptive_consecutive_mixed_steps
            ),
            consecutive_prefill_chunks=(
                scheduler._consecutive_prefill_chunks
            ),
            slo_clock_invalid=scheduler.slo_clock_invalid,
            slo_clock_invalid_reason=(
                scheduler.slo_clock_invalid_reason
            ),
            last_slo_decision_now_ns=(
                scheduler._last_slo_decision_now_ns
            ),
            publication_plan=publication_plan,
        )
```

Import `LeaseWriteBlockPublicationPlan` from `block_manager`.

- [ ] **Step 4: Store the normalized feature flag**

In `Scheduler.__init__`, beside the existing exact-burst flags:

```python
self._exact_greedy_decode_burst_lease_local_delta_journal = bool(
    getattr(
        config,
        "exact_greedy_decode_burst_lease_local_delta_journal",
        False,
    )
)
```

- [ ] **Step 5: Implement a fail-closed eligibility selector**

Add a helper that runs only after current row validation:

```python
def _select_exact_burst_phase_journal(
    self,
    seqs: tuple[Sequence, ...],
    rows: tuple[ScheduledOutputRow, ...],
    *,
    is_prefill: bool,
    do_sample: bool,
    batch_kind: str | None,
) -> SchedulerPostprocessJournal | ExactBurstPhaseDeltaJournal:
    row = rows[0] if len(rows) == 1 else None
    if (
        row is None
        or len(seqs) != 1
        or not row.exact_burst
        or row.exact_burst_phase not in ("prefix", "suffix")
    ):
        return SchedulerPostprocessJournal.capture(self, seqs)
    if not self._exact_greedy_decode_burst_lease_local_delta_journal:
        return SchedulerPostprocessJournal.capture(self, seqs)
    self._exact_greedy_decode_burst_stats.record_lease_local_delta_journal_attempt()
    sequence = seqs[0]
    lease = self._exact_greedy_decode_burst_pending_lease
    reason = None
    if (
        is_prefill
        or not do_sample
        or batch_kind is not None
        or lease is None
        or lease.authorized_token_count != 8
        or len(row.output_tokens) != 4
        or sequence.status != SequenceStatus.RUNNING
        or not sequence.ignore_eos
    ):
        reason = "unsupported_phase_shape"
    elif (
        row.exact_burst_phase == "suffix"
        and sequence.num_completion_tokens + 4
        >= sequence.max_tokens
    ):
        reason = "terminal_suffix"
    write_block_index = (
        lease.first_write_position
        // self.block_manager.block_size
        if reason is None
        else -1
    )
    if reason is None and (
        write_block_index >= len(sequence.block_table)
        or sequence.block_table[write_block_index]
        != lease.write_block_id
    ):
        reason = "write_block_position_mismatch"
    block = (
        self.block_manager.blocks[lease.write_block_id]
        if reason is None
        else None
    )
    if reason is None and block.hash != -1:
        reason = "write_block_already_published"
    predecessor_hash = None
    if reason is None and write_block_index > 0:
        predecessor_id = sequence.block_table[write_block_index - 1]
        predecessor = self.block_manager.blocks[predecessor_id]
        predecessor_hash = predecessor.hash
        if (
            predecessor_hash == -1
            or (
                self.block_manager.hash_to_block_id.get(
                    predecessor_hash
                )
                != predecessor_id
                and predecessor_id
                not in self.block_manager.hash_to_block_ids.get(
                    predecessor_hash,
                    (),
                )
            )
        ):
            reason = "predecessor_hash_unavailable"
    if reason is not None:
        self._exact_greedy_decode_burst_stats.record_lease_local_delta_journal_fallback(
            reason
        )
        return SchedulerPostprocessJournal.capture(self, seqs)
    materialized_tokens = (
        lease.first_write_position + 4
        if row.exact_burst_phase == "prefix"
        else lease.last_write_position + 1
    )
    publication_plan = (
        self.block_manager.plan_lease_write_block_publication(
            sequence,
            appended_tokens=row.output_tokens,
            block_table_index=write_block_index,
            expected_block_id=lease.write_block_id,
            expected_generation=lease.write_block_generation,
            materialized_tokens=materialized_tokens,
            predecessor_hash=predecessor_hash,
        )
    )
    journal = ExactBurstPhaseDeltaJournal.capture(
        self,
        sequence,
        expected_block_table_identity=(
            lease.block_table_identity
        ),
        publication_plan=publication_plan,
    )
    self._exact_greedy_decode_burst_stats.record_lease_local_delta_journal_capture()
    return journal
```

Use this selector instead of unconditional generic capture. Retain `capture_exact_burst_publication_hashes()` only when the selected journal is generic:

```python
journal = self._select_exact_burst_phase_journal(
    seqs,
    rows,
    is_prefill=is_prefill,
    do_sample=do_sample,
    batch_kind=batch_kind,
)
if exact_rows and isinstance(
    journal,
    SchedulerPostprocessJournal,
):
    journal.capture_exact_burst_publication_hashes(...)
```

- [ ] **Step 6: Run selection and existing prepare validation tests**

Run:

```bash
pytest -q tools/test_scheduler_prepared_postprocess.py -k \
  "delta_journal or split_phase or exact_burst_commit"
```

Expected: all selected tests PASS; disabled behavior remains generic.

- [ ] **Step 7: Commit journal capture and selection**

```bash
git add \
  tinyvllm/engine/scheduler.py \
  tools/test_scheduler_prepared_postprocess.py
git -c core.hooksPath=/dev/null commit -m "feat(scheduler): capture lease-local delta" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 4: Integrate Delta Commit, Publication, and Exact Rollback

**Files:**
- Modify: `tinyvllm/engine/scheduler.py:2589-3055`
- Test: `tools/test_scheduler_prepared_postprocess.py:1265-2205`
- Test: `tools/test_llm_engine_exact_greedy_decode_burst.py:850-1090`

**Interfaces:**
- `commit_prepared_postprocess()` accepts either journal type.
- `_apply_prepared_decode_row(..., publication_journal: ExactBurstPhaseDeltaJournal | None = None) -> None`
- `ExactBurstPhaseDeltaJournal.rollback(scheduler) -> None`
- `rollback_prepared_postprocess()` accepts either journal type.

- [ ] **Step 1: Add RED success and lifecycle tests**

Run the existing prefix/suffix test once with the feature disabled and once enabled. For the enabled arm assert:

```python
assert isinstance(
    prefix_prepared.snapshot,
    ExactBurstPhaseDeltaJournal,
)
scheduler.commit_prepared_postprocess(prefix_prepared)
assert prefix_prepared.snapshot.state == "committed"
assert scheduler._exact_greedy_decode_burst_split_phase == (
    "prefix_committed"
)
assert scheduler._exact_greedy_decode_burst_pending_lease is lease

assert isinstance(
    suffix_prepared.snapshot,
    ExactBurstPhaseDeltaJournal,
)
scheduler.commit_prepared_postprocess(suffix_prepared)
assert suffix_prepared.snapshot.state == "committed"
assert scheduler._exact_greedy_decode_burst_split_phase == "idle"
assert scheduler._exact_greedy_decode_burst_pending_lease is None
stats = scheduler.exact_greedy_decode_burst_summary()
assert stats["lease_local_delta_journal_commits"] == 2
assert stats["lease_local_delta_journal_rollbacks"] == 0
```

For a suffix that fills a block, patch `compute_hash` with a counter and assert one call. For prefix and partial suffix assert zero calls.

- [ ] **Step 2: Add RED fault-injection tests**

Parameterize injection points:

```python
(
    "after_first_token",
    "after_all_tokens",
    "before_publication",
    "after_hash_registration",
    "decode_progress",
    "slo_publication",
    "adaptive_reset",
)
```

Capture a logical snapshot containing token-list object identity and values, sequence metadata, block metadata, both hash maps, decode progress, SLO fields, adaptive fields, pending lease, split phase, and scheduler queues. Inject one `RuntimeError("injected delta commit failure")` at each point, then assert the post-rollback snapshot equals the pre-commit snapshot and:

```python
assert prepared.state == "commit_failed"
assert prepared.snapshot.state == "rolled_back"
assert stats["lease_local_delta_journal_rollbacks"] == 1
```

- [ ] **Step 3: Implement journal-polymorphic sequence extraction**

Replace the strict generic type check with:

```python
if isinstance(journal, SchedulerPostprocessJournal):
    seqs = tuple(
        sequence
        for sequence, _ in journal.sequence_states
    )
elif isinstance(journal, ExactBurstPhaseDeltaJournal):
    seqs = journal.scheduled_sequences
else:
    raise ValueError(
        "prepared Scheduler snapshot must be a supported "
        "postprocess journal"
    )
```

- [ ] **Step 4: Dispatch exact-burst publication by journal type**

Extend `_apply_prepared_decode_row()`:

```python
def _apply_prepared_decode_row(
    self,
    seq: Sequence,
    row: ScheduledOutputRow,
    *,
    step_end_ns: int | None,
    progress_updates: dict[int, int],
    finished_progress_entries_removed: list[int],
    requeue: bool,
    publication_journal: (
        ExactBurstPhaseDeltaJournal | None
    ) = None,
) -> None:
```

After appending tokens and revalidating the lease:

```python
if publication_journal is None:
    self.block_manager.publish_full_blocks(
        seq,
        materialized_tokens=materialized_tokens,
    )
else:
    publication_journal.publication_applied = (
        self.block_manager.publish_lease_write_block(
            seq,
            plan=publication_journal.publication_plan,
        )
    )
```

Pass `publication_journal=journal` only when `journal` is `ExactBurstPhaseDeltaJournal`; all other call sites pass the default.

- [ ] **Step 5: Implement exact delta rollback**

Add:

```python
def rollback(self, scheduler) -> None:
    if self.state != "active":
        raise RuntimeError(
            "delta journal is not active: "
            f"{self.state}"
        )
    try:
        sequence = self.sequence
        if sequence.token_ids is not self.token_list:
            raise RuntimeError(
                "delta journal token list identity changed"
            )
        if len(sequence.token_ids) < self.original_length:
            raise RuntimeError(
                "delta journal token list was truncated"
            )
        expected_block_ids = tuple(
            block_id
            for block_id, _ in self.expected_block_table_identity
        )
        if tuple(sequence.block_table) != expected_block_ids:
            raise RuntimeError(
                "delta journal block table changed"
            )
        scheduler.block_manager.validate_block_identities(
            self.expected_block_table_identity
        )
        if (
            len(scheduler.waiting) != self.waiting_length
            or len(scheduler.prefilling) != self.prefilling_length
            or len(scheduler.running) != self.running_length
            or (
                len(scheduler.running) == 1
                and scheduler.running[0] is sequence
            )
            != self.sequence_was_running
        ):
            raise RuntimeError(
                "delta journal scheduler queues changed"
            )
        publication = self.publication_plan
        block = scheduler.block_manager.blocks[
            publication.block_id
        ]
        publication_is_present = (
            publication.will_publish
            and block.generation
            == publication.block_generation
            and block.hash
            == publication.planned_block_hash
            and tuple(block.token_ids)
            == publication.planned_block_token_ids
        )
        if self.publication_applied or publication_is_present:
            block = scheduler.block_manager.blocks[
                publication.block_id
            ]
            if (
                block.generation
                != publication.block_generation
                or block.hash
                != publication.planned_block_hash
            ):
                raise RuntimeError(
                    "delta journal publication identity changed"
                )
            block.hash = publication.prior_block_hash
            block.token_ids = list(
                publication.prior_block_token_ids
            )
            block_hash = publication.planned_block_hash
            if publication.prior_duplicate_block_ids is None:
                scheduler.block_manager.hash_to_block_ids.pop(
                    block_hash,
                    None,
                )
            else:
                scheduler.block_manager.hash_to_block_ids[
                    block_hash
                ] = set(
                    publication.prior_duplicate_block_ids
                )
            if publication.prior_primary_block_id is None:
                scheduler.block_manager.hash_to_block_id.pop(
                    block_hash,
                    None,
                )
            else:
                scheduler.block_manager.hash_to_block_id[
                    block_hash
                ] = publication.prior_primary_block_id
        del sequence.token_ids[self.original_length:]
        sequence.last_token = self.original_last_token
        sequence.num_tokens = self.original_num_tokens
        sequence.status = self.original_status
        if self.decode_progress_present:
            scheduler.decode_progress_ns_by_seq_id[
                sequence.seq_id
            ] = (
                self.decode_progress_value
            )
        else:
            scheduler.decode_progress_ns_by_seq_id.pop(
                sequence.seq_id,
                None,
            )
        scheduler._last_slo_postprocess = dict(
            self.last_slo_postprocess
        )
        scheduler.adaptive_mixed_state = (
            self.adaptive_mixed_state
        )
        scheduler.adaptive_high_streak = (
            self.adaptive_high_streak
        )
        scheduler.adaptive_low_streak = self.adaptive_low_streak
        scheduler.adaptive_consecutive_mixed_steps = (
            self.adaptive_consecutive_mixed_steps
        )
        scheduler._consecutive_prefill_chunks = (
            self.consecutive_prefill_chunks
        )
        scheduler.slo_clock_invalid = self.slo_clock_invalid
        scheduler.slo_clock_invalid_reason = (
            self.slo_clock_invalid_reason
        )
        scheduler._last_slo_decision_now_ns = (
            self.last_slo_decision_now_ns
        )
    except BaseException:
        self.state = "rollback_failed"
        raise
    self.state = "rolled_back"
```

Do not restore queues or leases in this method; the eligible path cannot mutate them before the existing post-commit split transition, and tests make any such mutation fail visibly.

- [ ] **Step 6: Record lifecycle counters at terminal transitions**

On successful commit:

```python
if isinstance(journal, ExactBurstPhaseDeltaJournal):
    journal.state = "committed"
    self._exact_greedy_decode_burst_stats.record_lease_local_delta_journal_commit(
        published_blocks=int(
            journal.publication_applied
        )
    )
else:
    journal.state = "committed"
```

After successful automatic or explicit delta rollback:

```python
self._exact_greedy_decode_burst_stats.record_lease_local_delta_journal_rollback()
```

Ensure the counter is incremented once, after rollback reaches `rolled_back`, not when rollback starts.

- [ ] **Step 7: Preserve terminal rollback-failure behavior**

Add a test that replaces `journal.rollback` with a method raising `RuntimeError("injected delta rollback failure")`. Assert:

```python
with pytest.raises(
    SchedulerPostprocessRollbackError,
) as captured:
    scheduler.commit_prepared_postprocess(prepared)
assert str(captured.value.commit_error) == (
    "injected delta commit failure"
)
assert str(captured.value.rollback_error) == (
    "injected delta rollback failure"
)
assert prepared.state == "rollback_failed"
assert journal.state == "rollback_failed"
with pytest.raises(RuntimeError, match="not active"):
    scheduler.commit_prepared_postprocess(prepared)
with pytest.raises(RuntimeError, match="not active"):
    scheduler.rollback_prepared_postprocess(prepared)
```

- [ ] **Step 8: Run transaction and engine lifecycle suites**

Run:

```bash
pytest -q \
  tools/test_scheduler_prepared_postprocess.py -k \
    "split_phase or delta_journal or rollback" \
  tools/test_llm_engine_exact_greedy_decode_burst.py -k \
    "split_phase"
```

Expected: all selected tests PASS with both flag arms.

- [ ] **Step 9: Commit transactional integration**

```bash
git add \
  tinyvllm/engine/scheduler.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit -m "feat(scheduler): commit delta journal phases" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 5: Prove Context-Bounded Host Bookkeeping

**Files:**
- Create: `tools/profile_exact_burst_lease_local_delta_journal.py`
- Create: `tools/test_profile_exact_burst_lease_local_delta_journal.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**
- Produces: `build_profile_cases() -> tuple[dict, ...]`
- Produces: `run_profile_case(*, policy: str, sequence_length: int, repetitions: int) -> dict`
- Produces artifact schema `exact_burst_lease_local_delta_journal_cpu_profile_v1`
- Fixed contexts: 249, 2041, 8185 tokens
- Fixed policies: `generic`, `lease_local_delta`

- [ ] **Step 1: Add RED complexity tests**

Write tests that wrap the token list and block table with iteration-counting list subclasses after normal fixture construction. Assert delta capture does not call either full iterator and generic capture does. Patch `BlockManager.compute_hash` and assert:

```python
assert prefix_delta_compute_hash_calls == 0
assert partial_suffix_delta_compute_hash_calls == 0
assert full_suffix_delta_compute_hash_calls <= 1
```

Also assert:

```python
assert delta_journal.publication_applied is False
assert not hasattr(delta_journal, "sequence_states")
assert not hasattr(delta_journal, "blocks")
assert not hasattr(delta_journal, "hashes")
```

- [ ] **Step 2: Run complexity tests and preserve RED evidence**

Run:

```bash
pytest -q tools/test_scheduler_prepared_postprocess.py -k \
  "delta_journal_context_bounded or delta_journal_compute_hash"
```

Expected: FAIL until the implementation avoids all complete-container iteration beyond existing lease revalidation.

- [ ] **Step 3: Tighten the implementation until complexity tests pass**

Use direct indexed reads only in delta capture and publication. Do not call:

```python
tuple(sequence.token_ids)
tuple(sequence.block_table)
SchedulerPostprocessJournal.capture(...)
capture_exact_burst_publication_hashes(...)
publish_full_blocks(...)
```

from an eligible delta path.

- [ ] **Step 4: Add RED profiler contract tests**

Test:

```python
cases = build_profile_cases()
assert len(cases) == 6
assert {
    (row["policy"], row["sequence_length"])
    for row in cases
} == {
    (policy, length)
    for policy in ("generic", "lease_local_delta")
    for length in (249, 2041, 8185)
}
row = run_profile_case(
    policy="lease_local_delta",
    sequence_length=249,
    repetitions=3,
)
assert row["schema"] == (
    "exact_burst_lease_local_delta_journal_cpu_profile_v1"
)
assert row["sample_count"] == 3
assert row["prepare_median_us"] >= 0
assert row["prepare_p95_us"] >= row["prepare_median_us"]
assert row["positive_python_allocation_bytes"] >= 0
assert row["journal_touched_block_count"] <= 1
assert row["journal_hash_key_count"] <= 1
```

- [ ] **Step 5: Implement the CPU profiler**

The script must:

```python
PROFILE_SCHEMA = (
    "exact_burst_lease_local_delta_journal_cpu_profile_v1"
)
CONTEXT_LENGTHS = (249, 2041, 8185)
POLICIES = ("generic", "lease_local_delta")
DEFAULT_REPETITIONS = 100
```

For each case:

1. Build the same split-phase K8 scheduler and non-terminal phase fixture.
2. Warm up ten prepare-plus-explicit-rollback cycles.
3. Measure each prepare with `time.perf_counter_ns()`.
4. Roll back every prepared object so iterations start from identical logical state.
5. Use `tracemalloc` snapshots around the measured loop and sum only positive size differences.
6. Report median and nearest-rank P95 in microseconds.
7. Report attempts, captures, fallbacks, touched blocks, hash keys, and `compute_hash` calls.
8. Write one JSON summary and one JSONL row file when `--output-dir` is supplied.

The command-line entrypoint is:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
    )
    args = parser.parse_args(argv)
    rows = tuple(
        run_profile_case(
            policy=case["policy"],
            sequence_length=case["sequence_length"],
            repetitions=args.repetitions,
        )
        for case in build_profile_cases()
    )
    write_profile_artifacts(args.output_dir, rows)
    return 0
```

- [ ] **Step 6: Run profiler tests and a local profile**

Run:

```bash
pytest -q tools/test_profile_exact_burst_lease_local_delta_journal.py
python tools/profile_exact_burst_lease_local_delta_journal.py \
  --output-dir artifacts/exact_burst_lease_local_delta_journal/cpu-profile-local \
  --repetitions 100
```

Expected: tests PASS; six complete rows are written. The result is host evidence only and is not used as an end-to-end GO by itself.

- [ ] **Step 7: Commit the complexity proof and profiler**

```bash
git add \
  tinyvllm/engine/scheduler.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/profile_exact_burst_lease_local_delta_journal.py \
  tools/test_profile_exact_burst_lease_local_delta_journal.py
git -c core.hooksPath=/dev/null commit -m "test(runtime): profile delta journal scaling" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 6: Build the Fixed 60/24 Gate and Independent Verifier

**Files:**
- Create: `tools/exact_burst_lease_local_delta_journal_gate.py`
- Create: `tools/test_exact_burst_lease_local_delta_journal_gate.py`
- Create: `tools/exact_burst_lease_local_delta_journal_verify.py`
- Create: `tools/test_exact_burst_lease_local_delta_journal_verify.py`

**Interfaces:**
- Gate schema: `exact_burst_lease_local_delta_journal_gate_v1`
- Performance rows: `2 policies x 3 contexts x 10 repetitions = 60`
- Correctness rows: `2 policies x 3 contexts x 4 sampling points = 24`
- Classifications: `GO_EXACT_BURST_LEASE_LOCAL_DELTA_JOURNAL`, `NO_GO_PERFORMANCE`, `NO_GO_CORRECTNESS`, `NO_GO_TRANSACTIONAL_SAFETY`, `NO_GO_EVIDENCE_INCOMPLETE`
- Verifier: `verify_artifact_directory(path: Path) -> dict`

- [ ] **Step 1: Add RED manifest and order tests**

Assert:

```python
manifest = build_workload_manifest()
assert manifest["performance_row_count"] == 60
assert manifest["correctness_row_count"] == 24
assert manifest["policies"] == (
    "generic",
    "lease_local_delta",
)
assert manifest["contexts"] == (
    "short",
    "medium",
    "long",
)
assert manifest["performance_repetitions"] == 10
assert manifest["correctness_sampling_points"] == 4
```

For every context/repetition pair, assert both policies occur once and the first policy rotates and reverses so neither arm owns a fixed thermal position.

- [ ] **Step 2: Add RED classification tests**

Build a complete passing fixture and assert GO. Mutate one field at a time:

```python
("output_exact", False, "NO_GO_CORRECTNESS")
("sampled_logit_max_abs_diff", 1e-5, "NO_GO_CORRECTNESS")
("candidate_fallbacks", 1, "NO_GO_TRANSACTIONAL_SAFETY")
("candidate_rollbacks", 1, "NO_GO_TRANSACTIONAL_SAFETY")
("long_prepare_median_improvement_pct", 49.9, "NO_GO_PERFORMANCE")
("long_prepare_p95_improvement_pct", 49.9, "NO_GO_PERFORMANCE")
("aggregate_tpot_p95_regression_pct", 3.01, "NO_GO_PERFORMANCE")
("reserved_memory_regression_pct", 1.01, "NO_GO_PERFORMANCE")
("performance_row_count", 59, "NO_GO_EVIDENCE_INCOMPLETE")
("correctness_row_count", 23, "NO_GO_EVIDENCE_INCOMPLETE")
```

Also require exact equality for target forwards, graph replays, D2H calls, and D2H bytes.

- [ ] **Step 3: Implement the gate by adapting the split-phase producer**

Reuse the proven split-phase producer structure, but bind these policy overrides:

```python
POLICY_OVERRIDES = {
    "generic": {
        "exact_greedy_decode_burst": True,
        "exact_greedy_decode_burst_split_phase": True,
        "exact_greedy_decode_burst_tokens": 8,
        "exact_greedy_decode_burst_lease_local_delta_journal": False,
    },
    "lease_local_delta": {
        "exact_greedy_decode_burst": True,
        "exact_greedy_decode_burst_split_phase": True,
        "exact_greedy_decode_burst_tokens": 8,
        "exact_greedy_decode_burst_lease_local_delta_journal": True,
    },
}
```

Wrap the real scheduler `prepare_exact_greedy_decode_burst_phase_commit()` and `commit_prepared_postprocess()` methods only inside the benchmark process to record phase durations. Do not add production timing calls.

Write:

```text
workload_manifest.json
performance_rows.jsonl
correctness_rows.jsonl
phase_samples.jsonl
summary.json
source_manifest.json
runner_receipt.json
```

Every row carries policy, context, repetition/sample ordinal, order position, prompt digest, generated-token count, exact-burst counters, delta lifecycle counters, forwards, replays, D2H inventory, latency, throughput, and CUDA memory.

- [ ] **Step 4: Implement the independent verifier without importing gate classification**

The verifier separately:

1. Validates schema and exact file inventory.
2. Recomputes file SHA-256 values from `source_manifest.json`.
3. Validates pushed source SHA and patch hash.
4. Reconstructs the 60/24 key sets and rejects duplicates or missing rows.
5. Recomputes paired metrics and nearest-rank percentiles from raw rows.
6. Checks exact tokens and sampled logits.
7. Checks lifecycle counter authority per policy.
8. Checks forward/replay/D2H parity.
9. Recomputes all threshold booleans and final classification.
10. Rejects non-finite values, partial rows, unknown fallback reasons, and reused run tags.

Return:

```python
{
    "schema": "exact_burst_lease_local_delta_journal_verify_v1",
    "verified": True,
    "classification": recomputed_classification,
    "performance_row_count": 60,
    "correctness_row_count": 24,
}
```

- [ ] **Step 5: Add verifier tamper tests**

Starting from a complete fixture, independently verify rejection after:

```text
delete one performance row
duplicate one correctness key
change one output token
change one sampled logit
change one lifecycle counter
change one D2H byte count
change one source hash
change summary classification
insert NaN
insert an unknown fallback reason
```

- [ ] **Step 6: Run gate and verifier suites**

Run:

```bash
pytest -q \
  tools/test_exact_burst_lease_local_delta_journal_gate.py \
  tools/test_exact_burst_lease_local_delta_journal_verify.py
```

Expected: all tests PASS, including every negative classification and tamper case.

- [ ] **Step 7: Commit gate and verifier**

```bash
git add \
  tools/exact_burst_lease_local_delta_journal_gate.py \
  tools/test_exact_burst_lease_local_delta_journal_gate.py \
  tools/exact_burst_lease_local_delta_journal_verify.py \
  tools/test_exact_burst_lease_local_delta_journal_verify.py
git -c core.hooksPath=/dev/null commit -m "test(runtime): gate lease-local delta journal" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 7: Build the Source-Bound Remote Controller

**Files:**
- Create: `tools/run_exact_burst_lease_local_delta_journal_remote.py`
- Create: `tools/test_run_exact_burst_lease_local_delta_journal_remote.py`

**Interfaces:**
- Controller schema: `exact_burst_lease_local_delta_journal_remote_v1`
- Remote root: `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`
- Host: `sitian@10.232.195.203`
- Local ControlMaster: `/tmp/ssh-sitian-10.232.195.203`
- Kerberos cache: `FILE:/Users/bytedance/krb5cc_sitian`

- [ ] **Step 1: Add RED controller safety tests**

Assert the parser rejects:

```text
remote roots outside the approved mounted root
local or remote /tmp task roots
source SHA different from local HEAD
source SHA not equal to origin/feat/kv-sparse-attention
an existing run tag
Kerberos lifetime below the declared minimum
GPU memory above 1024 MiB
GPU utilization above 5 percent
any compute process
```

Patch subprocess calls and assert no command contains `kinit`, `kill`, `pkill`, `killall`, or a write redirection to `/`, `/tmp`, or `/private/tmp`.

- [ ] **Step 2: Run controller tests and preserve RED evidence**

Run:

```bash
pytest -q tools/test_run_exact_burst_lease_local_delta_journal_remote.py
```

Expected: FAIL because the controller module is absent.

- [ ] **Step 3: Implement source and credential preflight**

The controller must:

```python
REMOTE_TASK_ROOT = Path(
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
GPU_MEMORY_LIMIT_MIB = 1024
GPU_UTILIZATION_LIMIT_PERCENT = 5
PERFORMANCE_ROWS = 60
CORRECTNESS_ROWS = 24
```

Before upload:

1. Require a clean tracked diff for the exact implementation/tool write set; ignore unrelated untracked artifacts.
2. Read `git rev-parse HEAD`.
3. Read `git rev-parse origin/feat/kv-sparse-attention`.
4. Require equality.
5. Require the full 40-character SHA passed on the command line.
6. Run `klist -s` and parse remaining lifetime without refreshing.
7. Reject an existing local or remote run tag.
8. Build a source archive from the pushed commit, not the dirty worktree.

- [ ] **Step 4: Implement local monitoring and automatic launch**

Use a single long-lived local controller process. Each poll:

1. Reuses the SSH ControlMaster.
2. Reads all GPU inventory with `nvidia-smi`.
3. Selects one GPU satisfying all clean thresholds.
4. Does not alter external processes when no GPU is eligible.
5. Sleeps with a bounded poll interval and writes controller state locally.
6. Launches the remote worker immediately when a clean GPU is found.
7. Writes remote logs and artifacts only under the unique approved run directory.
8. Downloads the complete artifact directory after worker exit.
9. Runs the remote-produced independent verification receipt check.
10. Runs the local independent verifier from the downloaded raw artifacts.

The worker command exports `CUDA_VISIBLE_DEVICES=<selected-id>` and invokes:

```bash
python tools/exact_burst_lease_local_delta_journal_gate.py \
  --output-dir <approved-remote-run-dir>/primary \
  --source-sha <full-pushed-sha> \
  --run-tag <fresh-tag>
```

- [ ] **Step 5: Implement process-safe receipts**

Write local controller files:

```text
controller_manifest.json
controller_state.json
gpu_inventory.jsonl
launch_receipt.json
download_receipt.json
local_verify/verification.json
```

Write remote files:

```text
worker.pid
worker.pgid
worker.exitcode
worker.stdout.log
worker.stderr.log
primary/
independent-verify/verification.json
```

The controller may signal only the exact child PID/PGID it launched, after validating the receipt belongs to the current run tag. It never signals inventory processes or unrelated benchmark workers.

- [ ] **Step 6: Run controller safety tests**

Run:

```bash
pytest -q tools/test_run_exact_burst_lease_local_delta_journal_remote.py
```

Expected: all tests PASS, including path, credential, GPU, source, tag, and process ownership negatives.

- [ ] **Step 7: Commit and push all source needed by the hardware run**

```bash
git add \
  tools/run_exact_burst_lease_local_delta_journal_remote.py \
  tools/test_run_exact_burst_lease_local_delta_journal_remote.py
git -c core.hooksPath=/dev/null commit -m "test(runtime): run delta journal GPU gate" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin HEAD:feat/kv-sparse-attention
```

Expected: pushed branch HEAD equals local HEAD.

### Task 8: Run Full Local Regression and the Source-Bound 60/24 GPU Gate

**Files:**
- No production source changes expected
- Create local artifacts below `artifacts/exact_burst_lease_local_delta_journal/<fresh-run-tag>/`

**Interfaces:**
- Consumes the pushed source SHA from Task 7.
- Produces one complete primary artifact, one remote verifier receipt, and one local verifier receipt.

- [ ] **Step 1: Run focused and adjacent regressions**

Run:

```bash
pytest -q \
  tools/test_model_runner_spec_verify.py -k \
    "exact_greedy_decode_burst or split_phase or ragged" \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/test_profile_exact_burst_lease_local_delta_journal.py \
  tools/test_exact_burst_lease_local_delta_journal_gate.py \
  tools/test_exact_burst_lease_local_delta_journal_verify.py \
  tools/test_run_exact_burst_lease_local_delta_journal_remote.py
```

Expected: all selected tests PASS. Record exact counts and elapsed time.

- [ ] **Step 2: Run static and diff checks**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-delta-journal-pycache \
python -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/scheduler.py \
  tools/profile_exact_burst_lease_local_delta_journal.py \
  tools/exact_burst_lease_local_delta_journal_gate.py \
  tools/exact_burst_lease_local_delta_journal_verify.py \
  tools/run_exact_burst_lease_local_delta_journal_remote.py
git diff --check
git status --short --branch
```

Expected: compilation succeeds, `git diff --check` is silent, and no tracked task changes remain.

- [ ] **Step 3: Freeze the source identity**

Run:

```bash
SOURCE_SHA="$(git rev-parse HEAD)"
test "$SOURCE_SHA" = "$(git rev-parse origin/feat/kv-sparse-attention)"
printf '%s\n' "$SOURCE_SHA"
```

Expected: one full 40-character SHA and successful equality.

- [ ] **Step 4: Start the local controller with a fresh tag**

Use a never-before-used tag such as:

```bash
python tools/run_exact_burst_lease_local_delta_journal_remote.py \
  --host sitian@10.232.195.203 \
  --control-path /tmp/ssh-sitian-10.232.195.203 \
  --kerberos-cache FILE:/Users/bytedance/krb5cc_sitian \
  --remote-task-root \
    /data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818 \
  --source-sha "$SOURCE_SHA" \
  --run-tag 20260823-qwen3-06b-delta-journal-canonical-r1 \
  --local-output-dir \
    artifacts/exact_burst_lease_local_delta_journal/20260823-qwen3-06b-delta-journal-canonical-r1
```

If that tag exists or was attempted, increment the suffix. Never delete or reuse the prior tag.

- [ ] **Step 5: Reconcile only after all rows and receipts exist**

Require:

```text
worker.exitcode = 0
performance rows = 60
correctness rows = 24
phase sample inventory complete
remote verifier verified = true
local verifier verified = true
remote and local classifications identical
source SHA equals pushed frozen SHA
```

Do not classify from partial rows.

- [ ] **Step 6: Report benefit and cost from raw evidence**

Extract:

```text
long-context phase-prepare median and P95 improvement
short/medium phase-prepare median and P95 deltas
aggregate TTFT, TPOT median/P95/P99, E2E, throughput deltas
CUDA allocated and reserved-memory deltas
forward/replay/D2H parity
delta attempts/captures/commits/rollbacks/published blocks/fallbacks
CPU positive-allocation reduction
remaining full lease block-identity validation cost
```

If classification is GO, state the narrow Stage-1 authorization. If TPOT is neutral, explicitly say host bookkeeping improved while end-to-end decode remained neutral. For any NO_GO, retain all artifacts and name the failed threshold.

### Task 9: Prompt-to-Artifact Reconciliation, Final Audit, and Push

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`

**Interfaces:**
- Produces an exact final checklist mapping every requested deliverable to source, tests, commands, artifacts, verifier evidence, commits, and classification.

- [ ] **Step 1: Build the completion checklist from the actual repository and artifact state**

The checklist must include:

```text
original request: independently conceived inference optimization
design/spec file and pushed commit
implementation-plan file and pushed commit
default-off config and fail-closed validation
delta journal data model and lifecycle
bounded one-block publication
generic fallback matrix and stable reasons
fault injection and rollback-failure parity
CPU context-scaling and allocation profile
fixed 60/24 hardware workload
source-bound pushed SHA
strict-clean GPU admission
remote task-root compliance
remote independent verifier
local independent verifier
benefit metrics
cost metrics
final classification
remaining validation boundary
all implementation and evidence commits pushed
```

For every row, cite a concrete file path plus line, test name plus command/result, artifact path plus digest/count, or Git commit SHA. Mark uncertainty or missing evidence as incomplete and continue work rather than closing the goal.

- [ ] **Step 2: Update the handoff state**

Append a dated section containing:

```text
branch and exact pushed HEAD
design and implementation commits
focused and adjacent regression commands and counts
CPU profile artifact and metrics
canonical GPU run tag
primary, remote-verifier, and local-verifier paths
60/24 row counts
classification
benefit and cost numbers
default-off rollout boundary
remaining complete block-table revalidation cost
next safe optimization candidate
```

- [ ] **Step 3: Update the Phase 1 completion audit**

Add a complete reconciliation section and update the executive matrix and final classification. Distinguish:

```text
proven local transactional correctness
proven host context-scaling change
proven hardware benefit/cost
default-off production status
unproven broader applicability
```

Do not characterize the idea as academic novelty; use “a runtime-data-flow-specific original engineering design.”

- [ ] **Step 4: Verify documentation against artifacts**

Run:

```bash
rg -n \
  "lease-local delta|60 performance|24 correctness|classification|benefit|cost|default-off|block-table" \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git diff --check
git diff -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
```

Expected: every reported number matches raw artifacts and both verifiers; no whitespace errors.

- [ ] **Step 5: Commit and push final reconciliation**

```bash
git add \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git -c core.hooksPath=/dev/null commit -m "docs(runtime): reconcile delta journal gate" -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin HEAD:feat/kv-sparse-attention
```

- [ ] **Step 6: Perform the final completion audit**

Run:

```bash
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git log -1 --format='%H%n%B'
git status --short --branch
```

Confirm:

1. Local HEAD equals remote branch HEAD.
2. The final commit has exactly one required trailer.
3. No tracked task change remains.
4. Unrelated untracked artifacts are preserved.
5. Every checklist row is backed by actual evidence.
6. The final classification is taken only from complete dual-verified artifacts.

Only after all six checks pass is the optimization objective complete.
