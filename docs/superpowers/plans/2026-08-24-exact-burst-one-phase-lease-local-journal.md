# Exact-Burst One-Phase Lease-Local Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's active constraint forbids subagents and additional worktrees, so execute inline in the authoritative checkout with `executing-plans`.

**Goal:** Extend the existing lease-local scheduler transaction to eligible non-terminal one-phase exact-greedy K8 commits, then prove exact correctness, bounded rollback, host-side benefit, and real-GPU benefit/cost.

**Architecture:** Keep the one-phase CUDA Graph, forward count, token transfer, and eight-token append sequence unchanged. Generalize the existing split-phase delta journal and one-block publication plan so `prepare_postprocess()` can select them for a non-terminal one-phase K8 row; every uncertain or terminal case falls back to the generic journal before mutation. Add path-specific counters and a source-bound paired Qwen3-0.6B gate comparing the same one-phase K8 policy with the journal disabled and enabled.

**Tech Stack:** Python 3, dataclasses, pytest, TinyLLMForge scheduler/block manager, xxhash prefix-cache metadata, JSON/JSONL evidence, SSH remote controller, Qwen3-0.6B on one strict-clean A100.

## Global Constraints

- The only authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`, which resolves to `/Users/bytedance/dev/TinyLLMForge`.
- Stay on `feat/kv-sparse-attention`; do not create a worktree or use a subagent.
- Preserve every unrelated dirty or untracked file. Stage only exact task paths.
- Use `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- Use `python3 -m pytest`; `python` is not available in the local environment.
- The feature remains default-disabled.
- The journal flag requires exact burst and configured width 8; split-phase is optional.
- Stage-1 one-phase eligibility is TP1/rank 0, batch size 1, completion-only greedy decode, `temperature == 0`, `ignore_eos == true`, one exact-burst row, phase `None`, width 8, one lease write block, and a non-terminal result.
- Preserve complete lease, schedule-generation, block-table ID/generation, write-block ID/generation, and predecessor-hash validation.
- Keep the eight calls to `Sequence.append_token()` unchanged.
- Do not change CUDA Graph capture/replay, target forward count, logits, argmax, output token IDs, D2H calls/bytes, or KV write locations.
- Terminal K8, K2-K7 ragged, split rows outside existing eligibility, ordinary decode, prefill, mixed batches, speculative paths, TP greater than one, and uncertain publication state use the generic journal.
- Delta rollback truncates only the appended suffix and restores at most one block plus its primary/duplicate hash-index entries.
- A rollback failure remains terminal through `SchedulerPostprocessRollbackError`.
- Hardware data may be written only below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not run `kinit`.
- Do not terminate or interfere with external GPU processes.
- Strict-clean admission requires memory `<=1024 MiB`, utilization `<=5%`, and no compute process.
- Never reuse an attempted run tag.
- The remote source SHA must equal the already-pushed branch HEAD.
- Do not classify partial rows as a terminal gate.
- Report both benefit and cost for every conclusion.

---

## File Map

- Modify `tinyvllm/config.py`: allow the existing default-off flag with one-phase K8 while retaining exact-burst/K8 fail-closed validation.
- Modify `tinyvllm/engine/exact_greedy_decode_burst.py`: add one-phase-attributable journal counters and summary fields.
- Modify `tinyvllm/engine/scheduler.py`: generalize the journal name, selector, one-phase eligibility, rollback accounting, and publication dispatch.
- Modify `tools/test_model_runner_spec_verify.py`: configuration RED/GREEN coverage.
- Modify `tools/test_exact_greedy_decode_burst.py`: counter validation.
- Modify `tools/test_scheduler_prepared_postprocess.py`: one-phase selection, fallback, publication, bounded-capture, and rollback tests.
- Create `tools/profile_exact_burst_one_phase_lease_local_journal.py`: direct CPU profile at 256/2K/8K.
- Create `tools/test_profile_exact_burst_one_phase_lease_local_journal.py`: profile contract and production binding.
- Create `tools/exact_burst_one_phase_lease_local_journal_gate.py`: fixed 60 performance/24 correctness row gate.
- Create `tools/test_exact_burst_one_phase_lease_local_journal_gate.py`: gate inventory, threshold, and negative tests.
- Create `tools/exact_burst_one_phase_lease_local_journal_verify.py`: independent verifier.
- Create `tools/test_exact_burst_one_phase_lease_local_journal_verify.py`: tamper and incomplete-evidence tests.
- Create `tools/run_exact_burst_one_phase_lease_local_journal_remote.py`: source-bound controller and remote worker.
- Create `tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py`: path, source, GPU, Kerberos, and process-safety tests.
- Create `docs/superpowers/audits/2026-08-24-exact-burst-one-phase-lease-local-journal-audit.md`: final benefit/cost and classification.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`: reconciliation pointer and matrix.
- Modify `AGENT_HANDOFF_STATE.md`: source SHA, evidence, result, and next action.

### Task 1: Generalize the Configuration Contract and Statistics Surface

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Test: `tools/test_model_runner_spec_verify.py`
- Test: `tools/test_exact_greedy_decode_burst.py`

**Interfaces:**
- Consumes: existing `Config.exact_greedy_decode_burst_lease_local_delta_journal: bool`
- Produces: the same flag valid when exact burst is enabled, width is 8, and split phase is either false or true
- Produces: `record_lease_local_delta_journal_one_phase_attempt()`
- Produces: `record_lease_local_delta_journal_one_phase_capture()`
- Produces: `record_lease_local_delta_journal_one_phase_commit(*, published_blocks: int)`
- Produces: `record_lease_local_delta_journal_one_phase_rollback()`
- Produces: `record_lease_local_delta_journal_one_phase_fallback(reason: str)`

- [ ] **Step 1: Write the failing configuration test**

Replace the old “requires split phase” rejection with:

```python
enabled_one_phase = Config(
    model=model,
    exact_greedy_decode_burst=True,
    exact_greedy_decode_burst_tokens=8,
    exact_greedy_decode_burst_split_phase=False,
    exact_greedy_decode_burst_lease_local_delta_journal=True,
)
assert enabled_one_phase.exact_greedy_decode_burst_split_phase is False
assert (
    enabled_one_phase
    .exact_greedy_decode_burst_lease_local_delta_journal
    is True
)
```

Retain failures for disabled exact burst, non-boolean values, and width other
than 8.

- [ ] **Step 2: Run the configuration RED test**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py::test_exact_greedy_decode_burst_config_is_strict_and_default_off \
  -q
```

Expected: FAIL with `lease-local delta journal requires split phase`.

- [ ] **Step 3: Implement the minimal configuration change**

In `Config.__post_init__`, keep:

```python
if self.exact_greedy_decode_burst_lease_local_delta_journal:
    if not self.exact_greedy_decode_burst:
        raise ValueError(
            "lease-local delta journal requires "
            "exact_greedy_decode_burst"
        )
    if self.exact_greedy_decode_burst_tokens != 8:
        raise ValueError(
            "lease-local delta journal requires K8"
        )
```

Delete only the split-phase requirement.

- [ ] **Step 4: Write failing one-phase statistics tests**

Extend `test_stats_track_lease_local_delta_journal_lifecycle()`:

```python
stats.record_lease_local_delta_journal_one_phase_attempt()
stats.record_lease_local_delta_journal_one_phase_capture()
stats.record_lease_local_delta_journal_one_phase_commit(
    published_blocks=1,
)
stats.record_lease_local_delta_journal_one_phase_rollback()
stats.record_lease_local_delta_journal_one_phase_fallback(
    "terminal_one_phase",
)
summary = stats.summary()
assert summary[
    "lease_local_delta_journal_one_phase_attempts"
] == 1
assert summary[
    "lease_local_delta_journal_one_phase_captures"
] == 1
assert summary[
    "lease_local_delta_journal_one_phase_commits"
] == 1
assert summary[
    "lease_local_delta_journal_one_phase_rollbacks"
] == 1
assert summary[
    "lease_local_delta_journal_one_phase_published_blocks"
] == 1
assert summary[
    "lease_local_delta_journal_one_phase_fallback_counts"
] == {"terminal_one_phase": 1}
```

- [ ] **Step 5: Run the statistics RED test**

Run:

```bash
python3 -m pytest \
  tools/test_exact_greedy_decode_burst.py::test_stats_track_lease_local_delta_journal_lifecycle \
  -q
```

Expected: FAIL because the first one-phase method is absent.

- [ ] **Step 6: Implement path-specific counters**

Add dataclass fields:

```python
lease_local_delta_journal_one_phase_attempts: int = 0
lease_local_delta_journal_one_phase_captures: int = 0
lease_local_delta_journal_one_phase_commits: int = 0
lease_local_delta_journal_one_phase_rollbacks: int = 0
lease_local_delta_journal_one_phase_published_blocks: int = 0
lease_local_delta_journal_one_phase_fallback_counts: dict[
    str, int
] = field(default_factory=dict)
```

Implement the five methods with the same validation as aggregate journal
counters. `published_blocks` must be an integer in `[0, 1]`; fallback reasons
must be non-empty strings. Add all six fields to `summary()`.

- [ ] **Step 7: Run Task 1 GREEN tests**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py::test_exact_greedy_decode_burst_config_is_strict_and_default_off \
  tools/test_exact_greedy_decode_burst.py::test_stats_track_lease_local_delta_journal_lifecycle \
  -q
```

Expected: `2 passed`.

- [ ] **Step 8: Commit and push Task 1**

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_greedy_decode_burst.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): expose one-phase journal contract" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Select the Lease-Local Journal for Eligible One-Phase K8

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Test: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**
- Produces: `ExactBurstLeaseLocalDeltaJournal`
- Produces: `Scheduler._select_exact_burst_lease_local_journal(...)`
- Consumes: `BlockManager.plan_lease_write_block_publication(...)`
- Preserves: `PreparedSchedulerPostprocess.snapshot`

- [ ] **Step 1: Add a one-phase fixture**

Add a helper beside `_delta_split_phase_fixture`:

```python
def _delta_one_phase_fixture(
    monkeypatch,
    *,
    prompt_length: int,
    max_tokens: int = 32,
    enabled: bool = True,
):
    monkeypatch.setattr(Sequence, "block_size", 16)
    scheduler = Scheduler(
        SimpleNamespace(
            **{
                **vars(_config()),
                "kvcache_block_size": 16,
                "exact_greedy_decode_burst_split_phase": False,
                "exact_greedy_decode_burst_lease_local_delta_journal": (
                    enabled
                ),
            }
        )
    )
    sequence = _running_sequence(
        scheduler,
        list(range(prompt_length)),
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    lease = _prepare_exact_burst_lease(
        scheduler,
        sequence,
        configured_width=8,
        split_phase_enabled=False,
    )
    result = _exact_burst_result(
        lease,
        tokens=(11, 12, 13, 14, 15, 16, 17, 18),
    )
    row = ScheduledOutputRow(
        sequence_id=sequence.seq_id,
        output_tokens=result.tokens,
        speculative=False,
        exact_burst=True,
        exact_burst_phase=None,
    )
    return scheduler, sequence, lease, result, row
```

- [ ] **Step 2: Write RED selection tests**

Add:

```python
@pytest.mark.parametrize(
    ("prompt_length", "expected_publish"),
    ((1, False), (8, True)),
)
def test_one_phase_k8_selects_lease_local_journal(
    monkeypatch,
    prompt_length,
    expected_publish,
):
    scheduler, sequence, lease, result, row = (
        _delta_one_phase_fixture(
            monkeypatch,
            prompt_length=prompt_length,
        )
    )
    prepared = scheduler.prepare_postprocess(
        (sequence,),
        (row,),
        is_prefill=False,
        do_sample=True,
        batch_kind=None,
    )
    prepared.exact_burst_result = result
    assert isinstance(
        prepared.snapshot,
        scheduler_module.ExactBurstLeaseLocalDeltaJournal,
    )
    assert prepared.snapshot.publication_plan.will_publish is expected_publish
```

- [ ] **Step 3: Run the selection RED test**

Run:

```bash
python3 -m pytest \
  tools/test_scheduler_prepared_postprocess.py::test_one_phase_k8_selects_lease_local_journal \
  -q
```

Expected: FAIL because the generalized class is absent or the generic journal
is selected.

- [ ] **Step 4: Rename the internal journal and selector**

Rename:

```python
ExactBurstPhaseDeltaJournal
```

to:

```python
ExactBurstLeaseLocalDeltaJournal
```

and:

```python
_select_exact_burst_phase_journal
```

to:

```python
_select_exact_burst_lease_local_journal
```

Update internal type checks and focused tests. Do not retain a compatibility
alias because the class is private and un-serialized.

- [ ] **Step 5: Implement one-phase eligibility**

In the selector, classify the path before applying phase-specific checks:

```python
is_split = row.exact_burst_phase in ("prefix", "suffix")
is_one_phase = row.exact_burst_phase is None
is_one_phase_k8 = (
    is_one_phase
    and lease is not None
    and lease.authorized_token_count == 8
    and len(row.output_tokens) == 8
)
```

For one-phase, require:

```python
not is_prefill
do_sample
batch_kind is None
sequence.status == SequenceStatus.RUNNING
sequence.ignore_eos
float(sequence.temperature) == 0.0
sequence.num_completion_tokens + 8 < sequence.max_tokens
lease.first_write_position // block_size
    == lease.last_write_position // block_size
```

Compute:

```python
materialized_tokens = lease.last_write_position + 1
```

and call the existing publication planner with all eight output tokens.

- [ ] **Step 6: Add RED fallback tests**

Parameterize at least:

```python
(
    ("disabled", "generic", None),
    ("terminal", "generic", "terminal_one_phase"),
    ("ragged_width", "generic", "unsupported_burst_shape"),
    ("published_write_block", "generic", "write_block_already_published"),
    ("missing_predecessor", "generic", "predecessor_hash_unavailable"),
)
```

Assert both journal type and the exact one-phase fallback counter.

- [ ] **Step 7: Implement closed one-phase fallback accounting**

On every enabled one-phase exact row:

```python
stats.record_lease_local_delta_journal_attempt()
stats.record_lease_local_delta_journal_one_phase_attempt()
```

On fallback, increment both aggregate and one-phase reason maps. On capture,
increment aggregate and one-phase capture counters.

- [ ] **Step 8: Run Task 2 GREEN tests**

Run:

```bash
python3 -m pytest \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit and push Task 2**

```bash
git add -- \
  tinyvllm/engine/scheduler.py \
  tools/test_scheduler_prepared_postprocess.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(scheduler): prepare one-phase lease-local journal" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Prove Commit and Rollback Safety

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Test: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**
- Consumes: `ExactBurstLeaseLocalDeltaJournal.publication_plan`
- Produces: exact one-phase commit accounting
- Preserves: `SchedulerPostprocessRollbackError`

- [ ] **Step 1: Add RED success tests for partial and full blocks**

For prompt lengths that leave the write block partial and make it full:

```python
scheduler.commit_prepared_postprocess(prepared)
assert sequence.completion_token_ids == [
    11, 12, 13, 14, 15, 16, 17, 18
]
assert prepared.state == "committed"
assert prepared.snapshot.state == "committed"
assert scheduler._exact_greedy_decode_burst_pending_lease is None
summary = scheduler.exact_greedy_decode_burst_summary()
assert summary[
    "lease_local_delta_journal_one_phase_commits"
] == 1
assert summary[
    "lease_local_delta_journal_one_phase_published_blocks"
] == int(expected_publish)
```

Wrap `compute_hash()` and assert zero calls for a partial block and one call
for a newly full block.

- [ ] **Step 2: Run the success RED tests**

Run:

```bash
python3 -m pytest \
  tools/test_scheduler_prepared_postprocess.py::test_one_phase_k8_delta_journal_commits_partial_block \
  tools/test_scheduler_prepared_postprocess.py::test_one_phase_k8_delta_journal_commits_and_publishes_full_block \
  -q
```

Expected: FAIL because one-phase commit does not yet update path-specific
lifecycle counters.

- [ ] **Step 3: Implement one-phase commit accounting**

After a delta journal commits, inspect:

```python
is_one_phase = prepared.rows[0].exact_burst_phase is None
```

Record the existing aggregate commit and, when true:

```python
stats.record_lease_local_delta_journal_one_phase_commit(
    published_blocks=int(journal.publication_applied),
)
```

Do not change `_apply_prepared_decode_row()` token append order.

- [ ] **Step 4: Add RED rollback fault matrix**

Inject failures:

```text
after first append
after eighth append
after write-block publication
during decode-progress publication
during SLO publication
```

Before prepare, snapshot:

```python
before = _delta_transaction_snapshot(scheduler, sequence)
```

After each expected exception:

```python
assert _delta_transaction_snapshot(scheduler, sequence) == before
assert prepared.state == "commit_failed"
assert prepared.snapshot.state == "rolled_back"
assert summary[
    "lease_local_delta_journal_one_phase_rollbacks"
] == 1
```

- [ ] **Step 5: Implement one-phase rollback accounting**

When an `ExactBurstLeaseLocalDeltaJournal` rolls back and the prepared row is
one-phase, increment both aggregate and one-phase rollback counters. Preserve
the existing exception and rollback-failure paths.

- [ ] **Step 6: Add rollback-failure RED test**

Mutate token-list identity after prepare, then inject a commit failure:

```python
sequence.token_ids = list(sequence.token_ids)
with pytest.raises(
    scheduler_module.SchedulerPostprocessRollbackError
) as caught:
    scheduler.commit_prepared_postprocess(prepared)
assert caught.value.commit_error is not None
assert caught.value.rollback_error is not None
assert prepared.state == "rollback_failed"
assert prepared.snapshot.state == "rollback_failed"
```

- [ ] **Step 7: Add bounded-capture RED test**

Use iteration-counting token and block-table lists at contexts 249, 2041, and
8185. Assert:

```python
assert token_ids.iterations == 0
assert token_ids.slice_reads == 0
assert block_table.iterations == 0
assert block_table.slice_reads == 0
assert not hasattr(journal, "sequence_states")
assert not hasattr(journal, "blocks")
assert not hasattr(journal, "hashes")
```

The lease identity validation may iterate its already-frozen tuple; the
journal capture must not create another context-sized copy.

- [ ] **Step 8: Run Task 3 GREEN and adjacent tests**

Run:

```bash
python3 -m pytest \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit and push Task 3**

```bash
git add -- \
  tinyvllm/engine/scheduler.py \
  tools/test_scheduler_prepared_postprocess.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(scheduler): commit one-phase delta journal" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Add the Direct CPU Scaling Profile

**Files:**
- Create: `tools/profile_exact_burst_one_phase_lease_local_journal.py`
- Create: `tools/test_profile_exact_burst_one_phase_lease_local_journal.py`

**Interfaces:**
- Produces schema: `exact_burst_one_phase_lease_local_journal_cpu_profile_v1`
- Produces policies: `generic`, `lease_local_delta`
- Produces artifacts: `rows.jsonl`, `summary.json`

- [ ] **Step 1: Write RED profile contract tests**

Assert:

```python
assert module.CONTEXT_LENGTHS == (249, 2041, 8185)
assert module.POLICIES == ("generic", "lease_local_delta")
assert module.DEFAULT_REPETITIONS == 100
```

Run a three-repetition temporary profile and require six rows with:

```text
schema
policy
sequence_length
sample_count
prepare_median_us
prepare_p95_us
positive_python_allocation_bytes
compute_hash_calls
generic_journal_captures
one_phase_attempts
one_phase_captures
one_phase_rollbacks
one_phase_fallbacks
```

- [ ] **Step 2: Run the profile RED test**

Run:

```bash
python3 -m pytest \
  tools/test_profile_exact_burst_one_phase_lease_local_journal.py \
  -q
```

Expected: collection failure because the profile module is absent.

- [ ] **Step 3: Implement the profile from production entrypoints**

Reuse the dependency-light loader pattern from
`profile_exact_burst_lease_local_delta_journal.py`, but construct one-phase
rows:

```python
ScheduledOutputRow(
    sequence_id=sequence.seq_id,
    output_tokens=(11, 12, 13, 14, 15, 16, 17, 18),
    speculative=False,
    exact_burst=True,
    exact_burst_phase=None,
)
```

For every sample:

1. create a fresh scheduler/sequence/lease fixture;
2. time `prepare_postprocess()` with `time.perf_counter_ns()`;
3. call `prepared.snapshot.rollback(scheduler)`;
4. record stats and hash-call counts; and
5. never call the model or synthesize end-to-end TPOT.

- [ ] **Step 4: Run profile GREEN tests and produce local evidence**

Run:

```bash
python3 -m pytest \
  tools/test_profile_exact_burst_one_phase_lease_local_journal.py \
  -q
python3 tools/profile_exact_burst_one_phase_lease_local_journal.py \
  --output-dir \
  artifacts/exact_burst_one_phase_lease_local_journal/cpu-profile-local
```

Expected: tests pass; summary has six rows and no candidate fallback.

- [ ] **Step 5: Commit and push Task 4**

```bash
git add -- \
  tools/profile_exact_burst_one_phase_lease_local_journal.py \
  tools/test_profile_exact_burst_one_phase_lease_local_journal.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): profile one-phase journal scaling" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Do not commit generated local artifacts.

### Task 5: Add the Paired Gate and Independent Verifier

**Files:**
- Create: `tools/exact_burst_one_phase_lease_local_journal_gate.py`
- Create: `tools/test_exact_burst_one_phase_lease_local_journal_gate.py`
- Create: `tools/exact_burst_one_phase_lease_local_journal_verify.py`
- Create: `tools/test_exact_burst_one_phase_lease_local_journal_verify.py`

**Interfaces:**
- Produces performance schema: `exact_burst_one_phase_lease_local_journal_performance_v1`
- Produces correctness schema: `exact_burst_one_phase_lease_local_journal_correctness_v1`
- Produces gate schema: `exact_burst_one_phase_lease_local_journal_gate_v1`
- Produces GO: `GO_EXACT_BURST_ONE_PHASE_LEASE_LOCAL_JOURNAL`

- [ ] **Step 1: Write RED gate inventory tests**

Require:

```python
POLICIES = ("generic", "lease_local_delta")
CONTEXTS = ("2k", "4k", "8k")
PERFORMANCE_REPETITIONS = 10
PERFORMANCE_ROW_COUNT = 60
CORRECTNESS_ROW_COUNT = 24
```

Verify alternating/reversed policy order and rejection of duplicate, missing,
non-finite, wrong-schema, or mixed-run-tag rows.

- [ ] **Step 2: Write RED threshold tests**

Synthetic GO evidence must satisfy:

```text
exact outputs/logits
unchanged forwards/replays/D2H
candidate generic captures == 0
one-phase captures/commits == eligible bursts
one-phase fallbacks/rollbacks == 0
8K prepare median/P95 >= 50%
aggregate prepare median/P95 >= 35%
aggregate TPOT median/P95 >= 1%
TTFT/E2E/throughput/TPOT-P99 within 2%
allocated/reserved memory within 1%
```

Perturb one invariant at a time and assert the specified NO_GO.

- [ ] **Step 3: Implement the gate**

Reuse statistical helpers and manifest hashing patterns from
`exact_burst_lease_local_delta_journal_gate.py`. Change the workload to
one-phase K8 with split disabled in both arms. Wrap real scheduler prepare and
commit entrypoints to collect duration samples without production timing
instrumentation.

- [ ] **Step 4: Implement the independent verifier**

The verifier independently:

- checks source/workload/runner manifests and hashes;
- verifies exact row inventories;
- recomputes pair metrics and threshold booleans;
- rejects unknown fallback reasons;
- rejects partial evidence;
- rejects source SHA drift; and
- checks the reported classification equals the recomputed classification.

- [ ] **Step 5: Run gate/verifier GREEN tests**

Run:

```bash
python3 -m pytest \
  tools/test_exact_burst_one_phase_lease_local_journal_gate.py \
  tools/test_exact_burst_one_phase_lease_local_journal_verify.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit and push Task 5**

```bash
git add -- \
  tools/exact_burst_one_phase_lease_local_journal_gate.py \
  tools/test_exact_burst_one_phase_lease_local_journal_gate.py \
  tools/exact_burst_one_phase_lease_local_journal_verify.py \
  tools/test_exact_burst_one_phase_lease_local_journal_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): gate one-phase lease-local journal" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Add the Safe Remote Controller

**Files:**
- Create: `tools/run_exact_burst_one_phase_lease_local_journal_remote.py`
- Create: `tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py`

**Interfaces:**
- Consumes: pushed source SHA and a unique run tag
- Produces: controller preflight, runner log, remote exit code, mirrored primary bundle, remote verification, and local verification

- [ ] **Step 1: Write RED controller safety tests**

Require:

```text
remote paths remain below approved mounted root
runtime TMP/cache directories remain below the run staging directory
source SHA equals pushed branch HEAD
Kerberos lifetime >= 5400 seconds
GPU memory <= 1024 MiB
GPU utilization <= 5%
no compute process
selected GPU is rechecked immediately before launch
attempted tags are rejected
only the selected benchmark PID/process group is managed
no kinit command exists
```

- [ ] **Step 2: Run the controller RED tests**

Run:

```bash
python3 -m pytest \
  tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py \
  -q
```

Expected: collection failure because the controller module is absent.

- [ ] **Step 3: Implement the controller**

Adapt the existing delta-journal controller with:

```python
TASK_REMOTE_ROOT = (
    APPROVED_ROOT
    + "/exact-burst-one-phase-lease-local-journal"
)
```

Keep:

- immutable committed-source archive;
- empty source patch hash;
- remote runtime environment rooted under staging;
- Kerberos fail-fast;
- strict-clean polling;
- unique distributed port from run-tag hash;
- selected-GPU recheck;
- partial-preserving artifact download;
- remote verifier then local verifier; and
- no external-process termination.

- [ ] **Step 4: Run controller GREEN and full focused suite**

Run:

```bash
python3 -m pytest \
  tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py \
  tools/test_exact_burst_one_phase_lease_local_journal_gate.py \
  tools/test_exact_burst_one_phase_lease_local_journal_verify.py \
  tools/test_profile_exact_burst_one_phase_lease_local_journal.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit and push Task 6**

```bash
git add -- \
  tools/run_exact_burst_one_phase_lease_local_journal_remote.py \
  tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): run one-phase journal gate remotely" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 7: Execute the Source-Bound GPU Gate and Reconcile Evidence

**Files:**
- Create: `docs/superpowers/audits/2026-08-24-exact-burst-one-phase-lease-local-journal-audit.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Preserve untracked artifacts under `artifacts/exact_burst_one_phase_lease_local_journal/<run-tag>/`

**Interfaces:**
- Consumes: already-pushed branch HEAD
- Produces: one terminal classification backed by 60 performance rows, 24 correctness rows, and two verifiers

- [ ] **Step 1: Verify local source readiness**

Run:

```bash
git status --short -- \
  tinyvllm/config.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tools/profile_exact_burst_one_phase_lease_local_journal.py \
  tools/exact_burst_one_phase_lease_local_journal_gate.py \
  tools/exact_burst_one_phase_lease_local_journal_verify.py \
  tools/run_exact_burst_one_phase_lease_local_journal_remote.py
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
```

Expected: no task-path changes and equal SHAs.

- [ ] **Step 2: Run the final local focused suite**

Run the Task 6 suite plus adjacent split/ragged/continuation tests. Expected:
all selected tests pass.

- [ ] **Step 3: Launch with a fresh immutable tag**

Use a new tag matching:

```text
20260824-qwen3-06b-one-phase-lease-local-r<N>
```

Run the controller with the exact pushed SHA. Do not reuse `<N>` if preflight
or launch is attempted.

- [ ] **Step 4: Monitor while doing non-conflicting local audit preparation**

The controller must poll remote GPU state and launch immediately when one
strict-clean GPU is available. While it waits, draft only source-bound audit
structure and test-command inventory; do not write measured values before the
terminal bundle exists.

- [ ] **Step 5: Verify terminal evidence**

Require:

```text
remote_exitcode == 0
60 performance rows
24 correctness rows
source manifest SHA == pushed HEAD
remote verifier PASS
local verifier PASS
no candidate fallback or rollback
classification recomputes identically
```

If the bundle is partial, classify only `NO_GO_EVIDENCE_INCOMPLETE`; do not
infer performance.

- [ ] **Step 6: Write the audit with benefit and cost**

Record:

- exact classification;
- prepare median/P95 improvements by context and aggregate;
- TPOT median/P95/P99, TTFT, E2E, throughput;
- allocated/reserved memory delta;
- forwards/replays/D2H invariance;
- output/logit parity;
- lifecycle/fallback inventory;
- local Python allocation result;
- implementation/test surface cost; and
- remaining full lease block-identity validation cost.

- [ ] **Step 7: Run verification-before-completion**

Run:

```bash
python3 -m pytest \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_profile_exact_burst_one_phase_lease_local_journal.py \
  tools/test_exact_burst_one_phase_lease_local_journal_gate.py \
  tools/test_exact_burst_one_phase_lease_local_journal_verify.py \
  tools/test_run_exact_burst_one_phase_lease_local_journal_remote.py \
  -q
git diff --check -- \
  docs/superpowers/audits/2026-08-24-exact-burst-one-phase-lease-local-journal-audit.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
```

- [ ] **Step 8: Commit and push final reconciliation**

```bash
git add -- \
  docs/superpowers/audits/2026-08-24-exact-burst-one-phase-lease-local-journal-audit.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(bench): record one-phase journal result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Do not add generated benchmark artifacts unless a later explicit repository
policy requests tracked evidence bundles.
