# Engine Speculative Scheduler Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit selected multi-token speculative outputs and suppressed ordinary outputs through one rollback-safe Scheduler transaction, then install and execute a source-agnostic speculative runtime in `LLMEngine.step()`.

**Architecture:** Scheduler gains immutable output rows plus a prepared postprocess object that captures all host-side state before KV publication. Its commit path reuses the existing prefill, mixed, finish, release, and SLO semantics while restoring sequence, queue, allocator, block-manager, progress, and release-event state on any exception. `LLMEngine` validates an explicitly installed generic runtime, runs suppressed rows ordinarily, prepares selected rows through the existing batch runtime and fixed-Q ModelRunner callbacks, prebuilds KV and Scheduler plans, commits KV then Scheduler metadata, and synchronizes an optional draft lifecycle only after target state is authoritative.

**Tech Stack:** Python 3.9-compatible dataclasses and protocols, pytest dependency-light module loading, TinyLLMForge Scheduler/BlockManager/ModelRunner APIs.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, or run `git clean`.
- Generic runtime, Scheduler, verifier, and commit code must not contain model-name or proposal-source branches.
- Selected rows are greedy decode/sample rows only; nonzero temperature remains suppressed and guarded.
- Accepted KV is committed directly; rejected suffix blocks are released without KV copy or per-token rematerialization.
- Variable proposal lengths are grouped by distinct fixed query length; they are not padded.
- Stateful recurrent/convolution execution remains fail closed until non-KV state transactions exist.
- Default-off ordinary execution and return shape remain unchanged.
- This work does not support stochastic speculative decoding and must not claim end-to-end performance, GPU parity, TP1/TP4 correctness, long-context benefit, or promotion.
- Overall classification remains `NOT_PROMOTABLE`.

---

### Task 1: Immutable Scheduler Postprocess Preparation

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Create: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**
- Consumes: existing `Scheduler.postprocess(...)`, `Sequence`, `SequenceStatus`, and schedule metadata fields.
- Produces:

```python
@dataclass(frozen=True)
class ScheduledOutputRow:
    sequence_id: int
    output_tokens: tuple[int, ...]
    speculative: bool
    accepted_draft_tokens: tuple[int, ...] = ()


@dataclass
class PreparedSchedulerPostprocess:
    scheduled_sequence_ids: tuple[int, ...]
    rows: tuple[ScheduledOutputRow, ...]
    is_prefill: bool
    do_sample: bool
    batch_kind: str | None
    decision_now_ns: int | None
    step_end_ns: int | None
    snapshot: object
    state: str = "prepared"


Scheduler.prepare_postprocess(
    seqs: tuple[Sequence, ...],
    rows: tuple[ScheduledOutputRow, ...],
    is_prefill: bool = False,
    do_sample: bool = True,
    batch_kind: str | None = None,
    *,
    decision_now_ns: int | None = None,
    step_end_ns: int | None = None,
) -> PreparedSchedulerPostprocess
```

- [x] **Step 1: Write failing preparation tests**

Add tests that create real `Scheduler` and `Sequence` objects and assert:

```python
prepared = scheduler.prepare_postprocess(
    (selected, ordinary),
    (
        ScheduledOutputRow(
            sequence_id=selected.seq_id,
            output_tokens=(11, 12, 13),
            speculative=True,
            accepted_draft_tokens=(11, 12),
        ),
        ScheduledOutputRow(
            sequence_id=ordinary.seq_id,
            output_tokens=(21,),
            speculative=False,
        ),
    ),
)
assert prepared.state == "prepared"
assert selected.completion_token_ids == []
assert ordinary.completion_token_ids == []
assert tuple(scheduler.running) == (selected, ordinary)
```

Cover exact row order, duplicate/missing/extra IDs, non-tuple tokens, multi-token ordinary rows, selected prefill rows, selected non-sampling rows, selected non-greedy rows, output-budget overflow, and tokens after effective EOS.

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_scheduler_prepared_postprocess.py
```

Expected: collection succeeds and tests fail because `ScheduledOutputRow`, `PreparedSchedulerPostprocess`, and `Scheduler.prepare_postprocess` do not exist.

- [x] **Step 3: Implement immutable rows and non-mutating validation**

Add the dataclasses and validation helpers to `scheduler.py`. Preparation must:

```python
scheduled_ids = tuple(seq.seq_id for seq in seqs)
row_ids = tuple(row.sequence_id for row in rows)
if row_ids != scheduled_ids:
    raise ValueError(
        "postprocess rows must exactly match scheduled sequence order"
    )
```

For every row, validate integer tokens, uniqueness, remaining output budget, EOS truncation, and speculative eligibility from `is_prefill`, `do_sample`, `batch_kind`, `step_is_decode`, `step_do_sample`, and `temperature`. Capture state without mutating tokens, queues, statuses, block ownership, hybrid leases, progress maps, release events, or SLO observations.

- [x] **Step 4: Run preparation tests and verify GREEN**

Run:

```bash
python3 -m pytest -q tools/test_scheduler_prepared_postprocess.py
```

Expected: all preparation tests pass.

---

### Task 2: Atomic Scheduler Commit and Compatibility Wrapper

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`
- Modify: `tools/test_chunked_prefill.py`
- Modify: `tools/test_hybrid_state_scheduler.py`
- Modify: `tools/test_scheduler_prefill_commit_hook.py`

**Interfaces:**
- Consumes: `PreparedSchedulerPostprocess` from Task 1.
- Produces:

```python
Scheduler.commit_prepared_postprocess(
    prepared: PreparedSchedulerPostprocess,
) -> None

Scheduler.rollback_prepared_postprocess(
    prepared: PreparedSchedulerPostprocess,
) -> None
```

- Existing `Scheduler.postprocess(...)` remains a compatibility wrapper that converts the ordinary sampled token list into one output row per token-consuming sequence, then calls prepare and commit.

- [x] **Step 1: Write failing multi-token and rollback tests**

Add tests for:

```python
scheduler.commit_prepared_postprocess(prepared)
assert selected.completion_token_ids == [11, 12, 13]
assert ordinary.completion_token_ids == [21]
assert prepared.state == "committed"
```

Also cover EOS at an accepted token, output-budget finish, mixed intermediate prefill plus selected decode, final prefill plus selected decode, exactly-once commit, explicit rollback before commit, and default ordinary one-token compatibility.

Inject failures by temporarily replacing deterministic mutation helpers such as `Sequence.append_token`, `_release_request_storage`, `_record_decode_progress`, and deque `append` boundaries. After each failure assert restoration of:

```python
sequence.token_ids
sequence.last_token
sequence.num_tokens
sequence.status
sequence.block_table
sequence.num_cached_tokens
sequence.num_computed_tokens
sequence.prefill_chunk_start
sequence.prefill_chunk_end
sequence.prefill_chunk_final
sequence.step_is_decode
sequence.step_do_sample
scheduler.waiting
scheduler.prefilling
scheduler.running
scheduler.block_manager.free_block_ids
scheduler.block_manager.used_block_ids
scheduler.block_manager.blocks
scheduler.block_manager.hash_to_block_id
scheduler.block_manager.hash_to_block_ids
scheduler.hybrid_state_allocator
scheduler._hybrid_state_release_events
scheduler.decode_progress_ns_by_seq_id
scheduler._last_slo_postprocess
scheduler._prefill_commit_notified_request_ids
```

- [x] **Step 2: Run atomic commit tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_scheduler_prepared_postprocess.py
```

Expected: commit and rollback tests fail because the terminal operations do not exist.

- [x] **Step 3: Implement snapshot restore and tuple-token mutation**

Implement private frozen snapshots for sequences, queues, BlockManager blocks/indexes, hybrid allocator internals, release events, progress maps, prefill-hook bookkeeping, adaptive controller values, and SLO observation state.

Commit must:

```python
if prepared.state != "prepared":
    raise RuntimeError(
        f"prepared Scheduler postprocess is not active: {prepared.state}"
    )
try:
    self._apply_prepared_postprocess(prepared)
except BaseException:
    self._restore_postprocess_snapshot(prepared.snapshot)
    prepared.state = "commit_failed"
    raise
prepared.state = "committed"
```

For selected decode rows, append every token in `output_tokens` in order, stop only at the already validated terminal token, record decode progress once for the row, and release storage/remove queue membership once. Ordinary and prefill rows must retain existing behavior.

- [x] **Step 4: Convert legacy `postprocess` into prepare/commit**

Build ordinary rows by consuming the existing flat `token_ids` iterator only for rows whose current semantics sample a token. Use `speculative=False` and one-token tuples, then call the prepared API. Preserve existing positional/keyword signature and `None` handling.

- [x] **Step 5: Run Scheduler regressions**

Run:

```bash
python3 -m pytest -q \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_scheduler_speculative_selection.py \
  tools/test_hybrid_state_scheduler.py \
  tools/test_scheduler_prefill_commit_hook.py
python3 tools/test_chunked_prefill.py
```

Expected: all pytest cases pass and the chunked-prefill script prints its existing PASS summary.

---

### Task 3: Source-Agnostic Engine Runtime Installation and Lifecycle

**Files:**
- Create: `tinyvllm/engine/speculative_runtime.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Create: `tools/test_engine_speculative_runtime.py`
- Modify: `tools/test_speculative_public_api.py`

**Interfaces:**
- Consumes: `DraftAdapter`, adapter `DraftCapabilities`, Scheduler selection config, and optional lifecycle methods.
- Produces:

```python
class DraftLifecycle(Protocol):
    def register_sequence(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> None: ...

    def synchronize_verified_history(
        self,
        sequence_id: int,
        verified_token_ids: tuple[int, ...],
    ) -> int: ...

    def release_sequence(self, sequence_id: int) -> None: ...


@dataclass(frozen=True)
class EngineSpeculativeRuntime:
    draft_adapter: DraftAdapter
    lifecycle: DraftLifecycle | None = None


LLMEngine.install_speculative_runtime(
    runtime: EngineSpeculativeRuntime,
) -> None
```

- [x] **Step 1: Write failing installation and lifecycle tests**

Test validation of batch capability, positive proposal limit, Scheduler selection enabled, exact proposal-limit compatibility, required ModelRunner callback support, idempotence for the same runtime object, rejection of replacement runtime, and initial `poisoned=False`.

Test `add_request()` ordering with a fake lifecycle:

```python
engine.add_request([1, 2, 3], sampling_params)
assert lifecycle.events == [
    ("register", sequence_id, (1, 2, 3)),
]
assert scheduler.added_sequence.seq_id == sequence_id
```

If scheduler admission fails, release the just-registered lifecycle entry and re-raise.

- [x] **Step 2: Run runtime tests and verify RED**

Run:

```bash
python3 -m pytest -q tools/test_engine_speculative_runtime.py
```

Expected: tests fail because the runtime module and installation method do not exist.

- [x] **Step 3: Implement runtime contract and installation**

Validate only generic capabilities and callable interfaces. Do not branch on adapter type or source string. Initialize:

```python
self.speculative_runtime = None
self.speculative_runtime_poisoned = False
self.speculative_runtime_poison_reason = None
```

Keep the existing no-runtime selected-row guard. Register lifecycle history after `Sequence` construction and before Scheduler admission.

- [x] **Step 4: Run runtime tests and verify GREEN**

Run:

```bash
python3 -m pytest -q \
  tools/test_engine_speculative_runtime.py \
  tools/test_speculative_public_api.py
```

Expected: all tests pass.

---

### Task 4: Selected/Suppressed `LLMEngine.step()` Wiring

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tinyvllm/engine/speculative_execution.py`
- Modify: `tools/test_engine_speculative_execution.py`
- Modify: `tools/test_engine_speculative_runtime.py`
- Modify: `tools/test_llm_engine_speculative_selection_source.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**
- Consumes:
  - `prepare_native_speculative_batch(...)`
  - `rollback_prepared_native_speculative_batch(...)`
  - `BlockManager.prepare_speculative_kv_commit(...)`
  - `BlockManager.commit_speculative_kv_commit_batch(...)`
  - `build_engine_speculative_commit_rows(...)`
  - `Scheduler.prepare_postprocess(...)`
  - `Scheduler.commit_prepared_postprocess(...)`
- Produces a mixed engine transaction with selected speculative rows and suppressed ordinary rows in original schedule order.

- [x] **Step 1: Write failing engine transaction tests**

Using an `object.__new__`/`SimpleNamespace` engine shell with real Scheduler state and fake ModelRunner callbacks, prove:

1. no installed runtime raises before any ModelRunner call;
2. suppressed rows make exactly one ordinary `run` call with original row metadata;
3. selected rows make one first-target call and one tail call per distinct fixed query length;
4. pending release events are carried exactly once;
5. selected output rows are merged with suppressed rows in original schedule order;
6. prepare failure rolls back all active transactions and restores drained release events;
7. KV commit failure leaves Scheduler metadata unchanged;
8. Scheduler commit failure restores pre-step KV ownership and metadata;
9. selected tokens append exactly once;
10. ordinary default-off execution preserves return value and observation shape.

- [x] **Step 2: Run engine tests and verify RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_engine_speculative_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_llm_engine_speculative_selection_source.py
```

Expected: selected execution tests fail at the current guard or missing prepared wiring.

- [x] **Step 3: Add prepared-row conversion helpers**

Extend `speculative_execution.py` with a helper that converts `PreparedNativeSpeculativeBatch` into immutable commit rows without mutating live sequences. The helper must validate exact selected sequence order and use the pre-prepare completion counts.

- [x] **Step 4: Implement selected/suppressed execution**

In `LLMEngine.step()`:

```python
runtime = self.speculative_runtime
if partition.selected_sequences and runtime is None:
    raise RuntimeError(
        "speculative rows selected before engine runtime installation"
    )
if partition.selected_sequences and self.speculative_runtime_poisoned:
    raise RuntimeError(
        "speculative runtime is poisoned: "
        f"{self.speculative_runtime_poison_reason}"
    )
```

Run suppressed rows ordinarily. If no suppressed rows exist, flush pending releases before speculative callbacks. Prepare selected rows with `run_model_runner_first_targets` and `run_model_runner_tail_batch`. Build selected KV plans, merge Scheduler rows in original order, prepare Scheduler state before KV commit, commit KV plans, then commit Scheduler metadata. On any pre-KV failure roll back prepared speculative transactions. On Scheduler commit failure rely on its pre-KV snapshot to restore the full host transaction.

- [x] **Step 5: Synchronize and release lifecycle after target commit**

For every emitted row, call:

```python
lifecycle.synchronize_verified_history(
    seq.seq_id,
    tuple(seq.token_ids),
)
```

For finished rows, call `release_sequence` only after synchronization succeeds. On lifecycle failure, retain committed target state, set the runtime poison fields, and reject later selected execution before ModelRunner work.

- [x] **Step 6: Extend observations without changing returns**

Add:

```python
"speculative_output_token_counts"
"speculative_accepted_draft_token_counts"
"speculative_fixed_q_group_count"
"speculative_runtime_timing_ms"
```

Preserve `(outputs, num_tokens)` and the ordinary default-off observation fields. Decode `num_tokens` remains the existing negative scheduled-row accounting until a separately approved metric change.

- [x] **Step 7: Run engine and Scheduler regressions**

Run:

```bash
python3 -m pytest -q \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_llm_engine_speculative_selection_source.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_public_api.py
python3 tools/test_chunked_prefill.py
```

Expected: all pytest cases pass and chunked-prefill remains PASS.

---

### Task 5: Full Evidence and Documentation

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-generic-inference-optimization-goal-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-08-12-engine-speculative-scheduler-wiring.md`

**Interfaces:**
- Consumes: completed implementation and fresh command output.
- Produces: reproducible evidence, explicit limitations, and no unchecked plan tasks.

- [x] **Step 1: Run focused and full matrices**

Run the focused tests from Task 4, then the existing full speculative/engine/serialization matrix used for the prior `320 passed` baseline. Run:

```bash
python3 tools/test_native_verifier_attention.py
python3 tools/test_chunked_prefill.py
python3.9 -m py_compile \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/speculative_execution.py \
  tinyvllm/engine/llm_engine.py
python3.12 -m py_compile \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/speculative_execution.py \
  tinyvllm/engine/llm_engine.py
```

- [x] **Step 2: Run generic source and hygiene gates**

Run:

```bash
rg -n "qwen|sam|ngram" \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/speculative_runtime.py \
  tinyvllm/engine/speculative_execution.py
git diff --check
test -z "$(git diff --cached --name-only)"
```

Expected: no model/proposal-source branches in generic files, clean diff, and empty staged diff.

- [x] **Step 3: Update audit and handoff**

Record exact test counts and commands. State what the result proves:

- production selected/suppressed control-plane integration exists;
- accepted KV and Scheduler metadata are host-transactional under injected failures;
- lifecycle ordering and poisoning are tested;
- ordinary default-off behavior remains compatible.

State what it does not prove:

- GPU numerical parity;
- TP1/TP4 correctness;
- end-to-end throughput, TTFT, or TPOT improvement;
- long-context benefit;
- stochastic speculative decoding;
- recurrent/convolution non-KV transactional safety.

Keep classification `NOT_PROMOTABLE`.

- [x] **Step 4: Mark plan complete and rerun final gate**

Replace every task checkbox with `[x]`, then run:

```bash
if rg -n "^- \[ \]" \
  docs/superpowers/plans/2026-08-12-engine-speculative-scheduler-wiring.md
then
  exit 1
fi
git diff --check
test -z "$(git diff --cached --name-only)"
```

## Fresh Completion Evidence

```text
Scheduler prepared postprocess plus engine/runtime focused regression:
  116 passed
prior first-target/fixed-Q/speculative/engine/serialization matrix:
  322 passed
native verifier attention:
  passed; CUDA numerical cases deferred to remote gate
hybrid state Scheduler:
  passed
Scheduler prefill commit hook:
  passed (8 tests)
chunked prefill:
  passed
Python 3.9-compatible and 3.12 py_compile:
  passed
generic source scan and git diff hygiene:
  passed; staged diff empty
overall classification:
  NOT_PROMOTABLE
```

Expected: no unchecked tasks, clean diff, and empty staged diff.
