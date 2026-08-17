# Variable-Q Speculative CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Do not use
> subagents unless the user explicitly re-authorizes them. Steps use checkbox
> (`- [ ]`) syntax for tracking.

## 2026-08-15 Evidence Reconciliation

This plan is reconciled against the current exact-family implementation,
focused tests, the retained fail-closed preflight records, the durable
handoff, and the Phase 1 prompt-to-artifact audit.

```text
VARIABLE_Q_PLAN_TOTAL_STEPS=48
VARIABLE_Q_PLAN_CHECKED=44
VARIABLE_Q_PLAN_INTENTIONALLY_OPEN=4
VARIABLE_Q_FOCUSED_TESTS=93_PASSED
VARIABLE_Q_PRODUCER_VERIFIER_RUNNER_PYCOMPILE=PASS
```

The handoff and prompt-to-artifact audit steps are checked because both
documents now map the fixed TP1/no-offload/4K exact `(B,Q,W)` contract, local
test evidence, legacy artifact limitations, and the two blocked preflight
records. They do not establish CUDA correctness or performance.

The following remain open:

- Task 8 Step 5: no exact-family CUDA correctness PASS artifact;
- Task 9 Step 2: no CUDA performance artifact;
- Task 9 Step 3: no final full gate because both CUDA gates remain absent;
- Task 9 Step 6: the plan is not complete.

Strict boundary:

```text
VARIABLE_Q_EXACT_FAMILY_PASS_ARTIFACT=ABSENT
VARIABLE_Q_SOURCE_BOUND_ARCHIVED_VERIFIER=NOT_ESTABLISHED
VARIABLE_Q_TP4=NOT_ESTABLISHED
VARIABLE_Q_OFFLOAD=NOT_ESTABLISHED
VARIABLE_Q_LONG_CONTEXT=NOT_ESTABLISHED
VARIABLE_Q_PERFORMANCE=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

**Goal:** Add an opt-in TP1, no-KV-offload CUDA Graph path for batch-native
speculative tail verification, with one exact graph family per `(B, Q, W)` and
unchanged transactional KV finalization.

**Architecture:** Keep variable proposal lengths grouped by distinct exact
query length before `ModelRunner`. Add an independent spec-verify graph cache,
immutable live-transaction authorization snapshots, and ModelRunner-private
capture scratch blocks above the scheduler-visible KV range. Cold families run
eager and may capture after the live forward; ready exact identities replay
into the live transaction slots, while replay-started failures quarantine the
family and propagate to the existing batch-runtime rollback owner.

**Tech Stack:** Python 3.10+, dataclasses, PyTorch CUDA Graphs, paged
FlashAttention, SHA-256 identities, pytest, source/AST contract tests, CUDA
correctness and benchmark scripts.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, run `git clean`,
  or create a worktree.
- Do not use subagents unless the user explicitly authorizes them again.
- The first version supports TP1 only.
- The first version requires `kv_offload_mvp0=False`.
- Blockwise speculative verification remains eager.
- Preserve exact-Q grouping; never pad, round, or merge distinct Q values.
- Preserve one target forward per fixed-Q group.
- Preserve accepted-KV in-place commit and rejected-suffix rollback.
- Never replay, copy, or rematerialize accepted KV token by token.
- Keep proposal generation, logits, argmax, acceptance, commit, rollback,
  prefix refcounts, and scheduler metadata outside the graph.
- Non-zero temperature and non-transactional recurrent/convolution state fail
  closed.
- Before `graph.replay()` starts, an eager-safe mismatch may fall back to one
  eager forward.
- After `graph.replay()` starts, never retry eager for the same live
  transaction.
- A replay-started failure must quarantine the family, propagate the CUDA
  error, and let the existing batch-runtime owner roll back the live
  transaction.
- Capture may roll back only its private capture scratch lease; graph helpers
  must never finalize or roll back a live speculative transaction.
- Do not claim H2D or D2H savings because KV offload is disabled.
- Keep the repository-level classification `NOT_PROMOTABLE`.
- The approved configuration defaults are:

```text
spec_verify_cuda_graphs = False
spec_verify_cuda_graph_batch_allowlist = (1, 4)
spec_verify_cuda_graph_query_len_allowlist = ()
spec_verify_cuda_graph_min_observations = 2
spec_verify_cuda_graph_max_entries = 8
spec_verify_cuda_graph_max_static_bytes = 64 * 1024 * 1024
spec_verify_cuda_graph_max_reserved_bytes = 512 * 1024 * 1024
spec_verify_cuda_graph_max_total_capture_ns = 5_000_000_000
spec_verify_cuda_graph_max_single_capture_ns = 2_000_000_000
```

---

## File Map

### New Files

- `tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`
  - exact identity, cache configuration, entries, admission decisions,
    quarantine reasons, counters, LRU, and capture scratch leases.
- `tools/test_spec_verify_exact_cuda_graph_cache.py`
  - dependency-light identity/cache/lifecycle/budget tests.
- `tools/test_spec_verify_cuda_graph_config.py`
  - independent configuration normalization and fail-closed defaults.
- `tools/test_spec_verify_capture_transaction.py`
  - private scratch capacity, lease exclusivity, rollback, and publication
    ordering tests.
- `tools/test_model_runner_spec_verify_cuda_graph.py`
  - ModelRunner admission, dispatch, replay, observability, and failure tests.
- `tools/spec_verify_cuda_graph_smoke.py`
  - CUDA eager-versus-warmed-graph correctness and injected replay-failure
    gate.
- `tools/run_spec_verify_cuda_graph_gate_remote.py`
  - pinned remote correctness/performance evidence collector.
- `tools/verify_spec_verify_cuda_graph_gate.py`
  - fail-closed JSON evidence verifier.
- `tools/test_spec_verify_cuda_graph_gate.py`
  - dependency-light evidence-verifier schema and rejection tests.

### Modified Files

- `tinyvllm/config.py`
  - independent spec-verify graph fields and validator.
- `tinyvllm/engine/model_runner.py`
  - cache initialization, scratch capacity, exact identity, admission,
    capture, replay, dispatch events, and common postprocessing.
- `tinyvllm/engine/block_manager.py`
  - immutable live-transaction authorization snapshots.
- `tinyvllm/speculative/batch_runtime.py`
  - attach transaction authorization to each `TailBatchItem`.
- `tinyvllm/engine/speculative_model_runner.py`
  - preserve authorization through exact-Q grouping and RPC.
- `tools/test_speculative_kv_transaction.py`
  - authorization validity and stale-state tests.
- `tools/test_speculative_model_runner_callbacks.py`
  - authorization-preserving grouping/RPC tests.
- `tools/test_model_runner_spec_verify.py`
  - prepared metadata and ordinary eager-path regression tests.
- `tools/test_multi_sequence_cuda_graph_gate.py`
  - decode graph isolation regression.
- `AGENT_HANDOFF_STATE.md`
  - verified scope, commands, evidence, limitations, and next gate.

---

### Task 1: Independent Exact-Q Identity, Cache, and Configuration

**Files:**
- Create: `tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`
- Create: `tools/test_spec_verify_exact_cuda_graph_cache.py`
- Create: `tools/test_spec_verify_cuda_graph_config.py`
- Modify: `tinyvllm/config.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class SpecVerifyGraphIdentity:
    active_batch_size: int
    query_len: int
    total_query_tokens: int
    page_table_width: int
    flash_attn_num_splits: int
    attention_backend: str
    attention_backend_version: str
    input_dtype: str
    output_dtype: str
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_block_size: int
    device_compute_capability: tuple[int, int]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(
            _canonical_identity_payload(self)
        ).hexdigest()


@dataclass(frozen=True)
class SpecVerifyExactCudaGraphCacheConfig:
    enabled: bool
    batch_allowlist: tuple[int, ...]
    query_len_allowlist: tuple[int, ...]
    min_observations: int
    max_entries: int
    max_static_bytes: int
    max_reserved_bytes: int
    max_total_capture_ns: int
    max_single_capture_ns: int


@dataclass
class SpecVerifyExactCudaGraphEntry:
    identity: SpecVerifyGraphIdentity
    identity_sha256: str
    graph: object
    tensors: dict[str, object]
    static_bytes: int
    capture_duration_ns: int
    allocated_delta_bytes: int
    reserved_delta_bytes: int
    replay_count: int = 0
    last_replay_step: int | None = None
    last_use_step: int = 0
    in_flight_replays: int = 0
    state: str = "ready"
    terminal_reason: str | None = None


@dataclass(frozen=True)
class SpecVerifyGraphAdmissionDecision:
    should_capture: bool
    cache_state: str
    decision: str
    fallback_reason: str | None
    observation_count: int


class SpecVerifyExactCudaGraphCache:
    def ready_entry(
        self,
        identity: SpecVerifyGraphIdentity,
    ) -> SpecVerifyExactCudaGraphEntry | None:
        return self.ready_entries.get(identity.sha256)

    def observe_success(
        self,
        identity: SpecVerifyGraphIdentity,
        *,
        estimated_static_bytes: int,
        step_id: int,
    ) -> SpecVerifyGraphAdmissionDecision:
        """Record an eligible eager success and return capture admission."""

    def commit_capture(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
    ) -> None:
        """Publish a budget-valid captured entry as ready."""

    def quarantine(
        self,
        identity: SpecVerifyGraphIdentity,
        reason: str,
        *,
        retained_reserved_bytes: int = 0,
    ) -> None:
        """Install one stable process-lifetime terminal reason."""

    def begin_replay(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
        *,
        step_id: int,
    ) -> None:
        """Increment in-flight replay state after final preflight."""

    def finish_replay(
        self,
        entry: SpecVerifyExactCudaGraphEntry,
        *,
        step_id: int,
        succeeded: bool,
    ) -> None:
        """Clear in-flight state and update replay/LRU counters."""

    def summary(self) -> dict[str, object]:
        """Return a deterministic immutable-compatible cache snapshot."""
```

- Canonical reasons:

```python
SPEC_VERIFY_GRAPH_FALLBACK_REASONS = (
    "feature_disabled",
    "enforce_eager",
    "unsupported_mode",
    "tp_not_one",
    "kv_offload_enabled",
    "blockwise_enabled",
    "batch_not_allowlisted",
    "query_len_not_allowlisted",
    "non_greedy",
    "input_embeds_active",
    "hidden_state_return_active",
    "non_transactional_state",
    "transaction_unauthorized",
    "identity_invalid",
    "cold_identity",
    "entry_limit",
    "static_byte_budget",
    "reserved_byte_budget",
    "single_capture_budget",
    "total_capture_budget",
    "scratch_unavailable",
    "capture_failed",
    "identity_drift",
    "shape_drift",
    "cache_state_drift",
)

SPEC_VERIFY_GRAPH_QUARANTINE_REASONS = (
    "capture_failed",
    "capture_rollback_failed",
    "post_capture_budget",
    "identity_drift",
    "shape_drift",
    "replay_failed",
)
```

- `Config` gains the exact fields from Global Constraints.

- [x] **Step 1: Write identity and validator RED tests**

Create tests that instantiate the identity with:

```python
IDENTITY_KWARGS = {
    "active_batch_size": 4,
    "query_len": 3,
    "total_query_tokens": 12,
    "page_table_width": 17,
    "flash_attn_num_splits": 16,
    "attention_backend": "flash_attn",
    "attention_backend_version": "test-version",
    "input_dtype": "torch.int64",
    "output_dtype": "torch.bfloat16",
    "num_query_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "page_block_size": 256,
    "device_compute_capability": (8, 0),
}
```

Assert:

```text
same fields -> same SHA-256
change B, Q, W, dtype, backend version, or capability -> different SHA-256
total_query_tokens != B * Q -> ValueError
B <= 0, Q <= 0, W <= 0 -> ValueError
flash_attn_num_splits != 16 -> ValueError
bool in an integer field -> ValueError
```

Add configuration tests proving:

```text
batch allowlist (1, 4) is valid and canonical
batch allowlist accepts B == 1
Q allowlist () is valid and admits no family
duplicate or unsorted allowlists normalize to sorted unique tuples
zero, negative, boolean, string, or non-container values fail
decode graph fields and defaults are unchanged
```

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q \
  tools/test_spec_verify_exact_cuda_graph_cache.py \
  tools/test_spec_verify_cuda_graph_config.py
```

Expected: import and missing-field failures because the new cache module and
configuration fields do not exist.

- [x] **Step 3: Implement identity and configuration normalization**

Serialize identity fields in a fixed tuple order:

```python
def _canonical_identity_payload(
    identity: SpecVerifyGraphIdentity,
) -> bytes:
    values = (
        identity.active_batch_size,
        identity.query_len,
        identity.total_query_tokens,
        identity.page_table_width,
        identity.flash_attn_num_splits,
        identity.attention_backend,
        identity.attention_backend_version,
        identity.input_dtype,
        identity.output_dtype,
        identity.num_query_heads,
        identity.num_kv_heads,
        identity.head_dim,
        identity.page_block_size,
        identity.device_compute_capability,
    )
    return repr(values).encode("utf-8")
```

Use `hashlib.sha256(_canonical_identity_payload(self)).hexdigest()` and reject
all invariant violations in `__post_init__`.

Add a dedicated `Config.__post_init__` normalization block that does not call
or weaken the decode graph validator:

```python
self.spec_verify_cuda_graph_batch_allowlist = (
    _normalize_positive_int_tuple(
        self.spec_verify_cuda_graph_batch_allowlist,
        name="spec_verify_cuda_graph_batch_allowlist",
        allow_empty=False,
    )
)
self.spec_verify_cuda_graph_query_len_allowlist = (
    _normalize_positive_int_tuple(
        self.spec_verify_cuda_graph_query_len_allowlist,
        name="spec_verify_cuda_graph_query_len_allowlist",
        allow_empty=True,
    )
)
```

Validate `spec_verify_cuda_graphs` as an exact `bool` and every budget as a
positive non-boolean integer.

- [x] **Step 4: Implement cache lifecycle and budgets**

Implement:

```text
unseen eligible identity -> eager miss, observation 1
observation 2 -> capture requested
quarantined identity -> stable terminal reason, never recapture
ready exact identity -> hit
ready identity with in_flight_replays > 0 -> not evictable
capturing identity -> not evictable
quarantined identity -> not evictable
entry pressure -> evict least last_use_step among ready zero-in-flight entries
eviction -> drop Python references but retain conservative reserved accounting
pre-capture and post-capture budget failures -> stable reason and no publish
```

Every state transition must update deterministic counters exposed by
`summary()`:

```text
hits
misses
capture_attempts
captures
fallbacks
quarantines
evictions
```

- [x] **Step 5: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_spec_verify_exact_cuda_graph_cache.py \
  tools/test_spec_verify_cuda_graph_config.py
```

Expected: all tests pass.

---

### Task 2: Capture-Only Scratch Capacity and Reversible Lease

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`
- Create: `tools/test_spec_verify_capture_transaction.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: normalized batch/Q allowlists from Task 1.
- Produces:

```python
def required_spec_verify_capture_scratch_blocks(
    *,
    batch_allowlist: tuple[int, ...],
    query_len_allowlist: tuple[int, ...],
    block_size: int,
) -> int:
    if not query_len_allowlist:
        return 0
    blocks_per_row = (
        block_size - 1
        + max(query_len_allowlist)
        + block_size - 1
    ) // block_size
    return max(batch_allowlist) * blocks_per_row


@dataclass
class SpecVerifyCaptureScratchLease:
    lease_id: int
    block_ids: tuple[int, ...]
    block_generations: tuple[int, ...]
    row_block_counts: tuple[int, ...]
    state: str = "active"


class SpecVerifyCaptureScratchPool:
    def acquire(
        self,
        *,
        active_batch_size: int,
        query_len: int,
        row_offsets: tuple[int, ...],
    ) -> SpecVerifyCaptureScratchLease:
        """Reserve disjoint private blocks for one exact capture."""

    def rollback(
        self,
        lease: SpecVerifyCaptureScratchLease,
    ) -> None:
        """Release every leased block and advance private generations."""
```

- [x] **Step 1: Write scratch capacity and lease RED tests**

Cover:

```python
assert required_spec_verify_capture_scratch_blocks(
    batch_allowlist=(1, 4),
    query_len_allowlist=(),
    block_size=256,
) == 0

assert required_spec_verify_capture_scratch_blocks(
    batch_allowlist=(1, 4),
    query_len_allowlist=(1, 8),
    block_size=256,
) == 8

assert required_spec_verify_capture_scratch_blocks(
    batch_allowlist=(1, 4),
    query_len_allowlist=(256, 257),
    block_size=256,
) == 8

assert required_spec_verify_capture_scratch_blocks(
    batch_allowlist=(1, 4),
    query_len_allowlist=(257, 258),
    block_size=256,
) == 12
```

The per-row requirement is:

```python
blocks_per_row = (
    block_size - 1
    + max(query_len_allowlist)
    + block_size - 1
) // block_size
```

Capacity covers the worst valid live terminal-block offset
`block_size - 1`. A partially occupied terminal prefix block must be cloned
into the first private block before capture, and speculative writes may then
cross into following private blocks.

Test that:

```text
scratch block IDs are all >= config.num_kvcache_blocks
decode scratch IDs and spec-verify scratch IDs are disjoint
two active leases never overlap
rollback releases every block and increments its private generation
double rollback fails
unknown lease fails
pool exhaustion returns scratch_unavailable
no scheduler Block, hash index, refcount, or sequence metadata is touched
```

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q \
  tools/test_spec_verify_capture_transaction.py \
  tools/test_model_runner_spec_verify.py \
  -k 'scratch or capacity'
```

Expected: failures because the spec-verify scratch pool does not exist.

- [x] **Step 3: Partition physical KV capacity**

In `ModelRunner.allocate_kv_cache()` calculate:

```python
decode_scratch_blocks = (
    max(config.multi_sequence_cuda_graph_batch_allowlist)
    if config.multi_sequence_cuda_graphs
    else 0
)
spec_verify_scratch_blocks = (
    required_spec_verify_capture_scratch_blocks(
        batch_allowlist=(
            config.spec_verify_cuda_graph_batch_allowlist
        ),
        query_len_allowlist=(
            config.spec_verify_cuda_graph_query_len_allowlist
        ),
        block_size=self.block_size,
    )
    if config.spec_verify_cuda_graphs
    else 0
)
total_scratch_blocks = (
    decode_scratch_blocks + spec_verify_scratch_blocks
)
```

Call `resolve_exact_graph_kv_capacity()` once with `total_scratch_blocks`.
Partition the physical tail deterministically:

```text
[0, visible_blocks)                         scheduler visible
[visible_blocks, decode_scratch_end)        decode graph scratch
[decode_scratch_end, physical_blocks)       spec-verify capture scratch
```

Keep `config.num_kvcache_blocks == visible_blocks` so the scheduler's
`BlockManager` never sees either scratch range.

- [x] **Step 4: Implement private scratch leases**

The pool owns only ModelRunner-private physical block IDs. It tracks a private
generation per block, active lease IDs, and free IDs. `acquire()` consumes the
exact per-row terminal offsets, reserves
`ceil((row_offset + Q) / block_size)` whole blocks for each row, records those
row counts on the lease, and returns one lease. `rollback()` zeros or
invalidates graph-private scratch metadata, releases the IDs, increments
private generations, and marks the lease `rolled_back`.

Do not call:

```text
BlockManager.prepare_speculative_kv_commit
BlockManager.commit_speculative_kv_commit_batch
BlockManager.rollback_speculative_kv_transaction
prefix publication
scheduler mutation
```

- [x] **Step 5: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_spec_verify_capture_transaction.py \
  tools/test_model_runner_spec_verify.py \
  -k 'scratch or capacity'
```

Expected: all selected tests pass.

---

### Task 3: Live Speculative Transaction Authorization

**Files:**
- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tools/test_speculative_kv_transaction.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class SpeculativeKVTransactionAuthorization:
    sequence_id: int
    original_num_tokens: int
    proposed_token_count: int
    materialized_token_count: int
    state: str
    original_block_identities: tuple[tuple[int, int], ...]
    reserved_block_identities: tuple[tuple[int, int], ...]
    authorization_sha256: str


def BlockManager.authorize_speculative_kv_write(
    self,
    transaction: SpeculativeKVTransaction,
    seq: Sequence,
) -> SpeculativeKVTransactionAuthorization:
    """Validate live ownership and return an immutable authorization."""
```

- Extends:

```python
@dataclass(frozen=True)
class TailBatchItem:
    sequence_id: int
    plan: SpecVerifyPlan
    proxy_block_table: tuple[int, ...]
    original_block_identities: tuple[tuple[int, int], ...] = ()
    reserved_block_identities: tuple[tuple[int, int], ...] = ()
    transaction_authorization: (
        SpeculativeKVTransactionAuthorization | None
    ) = None
```

- [x] **Step 1: Write authorization RED tests**

Construct a reserved transaction and assert the authorization contains exact:

```text
sequence_id
original_num_tokens
proposed_token_count
materialized_token_count == 0
state == reserved
original block IDs and generations
reserved block IDs and generations
canonical SHA-256
```

Reject authorization when:

```text
transaction state is materialized, committed, or rolled_back
transaction materialized_token_count is non-zero
sequence owner differs
sequence snapshot differs
an original or reserved generation is stale
reserved ownership is stale
```

Extend callback tests so grouping `seq8/Q2, seq4/Q1, seq2/Q2` preserves the
exact authorization object by identity in every RPC item.

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_spec_verify.py \
  -k 'authorization or fixed_q or transaction'
```

Expected: missing authorization type/method/field failures.

- [x] **Step 3: Implement canonical authorization**

`authorize_speculative_kv_write()` must call the existing transaction structure,
sequence owner, original block, and reserved block validators before building
the frozen snapshot. Hash this fixed payload:

```python
payload = (
    sequence_id,
    original_num_tokens,
    proposed_token_count,
    materialized_token_count,
    state,
    original_block_identities,
    reserved_block_identities,
)
authorization_sha256 = hashlib.sha256(
    repr(payload).encode("utf-8")
).hexdigest()
```

Only `state == "reserved"` and `materialized_token_count == 0` are authorized.

- [x] **Step 4: Attach and preserve authorization**

In `prepare_native_speculative_batch()`, build each `TailBatchItem` with:

```python
transaction_authorization=(
    block_manager.authorize_speculative_kv_write(
        transaction,
        seq,
    )
)
```

Do not change exact-Q grouping. `FixedQTailBatch.items` must carry the frozen
objects unchanged through `run_model_runner_tail_batch()` and
`run_spec_verify_batch()`.

- [x] **Step 5: Validate authorization against prepared rows**

Before graph admission, `ModelRunner` validates:

```text
authorization sequence_id == metadata row sequence_id
authorization state == reserved
authorization materialized_token_count == 0
authorization proposed_token_count >= row.query_len + 1
authorization original identities == item.original_block_identities
authorization reserved identities == item.reserved_block_identities
proxy block table == original IDs + reserved IDs
logical slots map into the authorized proxy table
physical slots are exactly the prepared direct-write slots
SHA-256 recomputes exactly
```

An invalid or absent authorization is not an eager-safe graph mismatch. It
must raise before replay/capture because deterministic transaction ownership
cannot be established.

- [x] **Step 6: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_model_runner_spec_verify.py \
  -k 'authorization or fixed_q or transaction'
```

Expected: all selected tests pass.

---

### Task 4: ModelRunner Admission, Identity, and Observability

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Create: `tools/test_model_runner_spec_verify_cuda_graph.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Produces:

```python
def ModelRunner._spec_verify_graph_incompatible_reason(
    self,
    *,
    input_ids,
    input_embeds,
    return_hidden: bool,
    context,
    transaction_authorized: bool,
) -> tuple[str | None, bool]:
    """Return `(reason, eager_safe)` for exact graph admission."""


def ModelRunner._build_spec_verify_graph_identity(
    self,
    *,
    input_ids,
    outputs,
    context,
) -> SpecVerifyGraphIdentity:
    """Build and validate the canonical exact runtime identity."""


def ModelRunner._estimate_spec_verify_graph_static_bytes(
    self,
    identity: SpecVerifyGraphIdentity,
) -> int:
    """Return exact static input, metadata, and output tensor bytes."""


def ModelRunner._publish_spec_verify_graph_dispatch_event(
    self,
    *,
    identity: SpecVerifyGraphIdentity | None,
    dispatch: str,
    decision: str,
    fallback_reason: str | None,
    cache_state: str,
    observation_count: int,
    capture_attempted: bool,
    capture_entry: SpecVerifyExactCudaGraphEntry | None,
    transaction_authorized: bool,
) -> None:
    """Publish one fixed-schema spec-verify dispatch event."""


def ModelRunner.spec_verify_graph_dispatch_observation(
    self,
) -> dict[str, object] | None:
    event = self.last_spec_verify_cuda_graph_dispatch_event
    return None if event is None else dict(event)
```

- [x] **Step 1: Write admission RED tests**

Use dependency-light runner fixtures and assert:

```text
feature disabled -> eager-safe feature_disabled
enforce_eager -> eager-safe enforce_eager
mode != spec_verify -> eager-safe unsupported_mode
TP > 1 -> eager-safe tp_not_one
kv_offload_mvp0 -> eager-safe kv_offload_enabled
blockwise spec-verify -> eager-safe blockwise_enabled
B miss -> eager-safe batch_not_allowlisted
Q miss -> eager-safe query_len_not_allowlisted
input embeds -> eager-safe input_embeds_active
return_hidden -> eager-safe hidden_state_return_active
active recurrent/convolution bridge -> fail-closed non_transactional_state
invalid transaction -> fail-closed transaction_unauthorized
B == 1 and B == 4 exact allowlisted families -> eligible
```

Assert identity uses real tensor shapes:

```text
input_ids [B * Q]
slot_mapping [B * Q]
context_lens [B]
block_tables [B, W]
outputs [B * Q, hidden_size]
```

Reject any mismatch without rounding.

- [x] **Step 2: Write fixed event-schema RED test**

Require exactly:

```python
SPEC_VERIFY_DISPATCH_EVENT_FIELDS = (
    "step_id",
    "request_ids_hash",
    "mode",
    "active_batch_size",
    "query_len",
    "total_query_tokens",
    "page_table_width",
    "flash_attn_num_splits",
    "graph_identity_sha256",
    "feature_enabled",
    "dispatch",
    "decision",
    "fallback_reason",
    "cache_state",
    "observation_count",
    "capture_attempted",
    "capture_duration_ns",
    "capture_static_bytes",
    "capture_allocated_delta_bytes",
    "capture_reserved_delta_bytes",
    "cache_ready_entries",
    "cache_static_bytes",
    "cache_reserved_delta_bytes",
    "cache_total_capture_ns",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "cache_quarantines",
    "transaction_authorized",
    "source_sha256",
)
```

The event must be separate from `last_cuda_graph_dispatch_event`.

- [x] **Step 3: Run RED**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_model_runner_spec_verify.py \
  -k 'admission or identity or dispatch_event'
```

Expected: missing helper, cache, and event-schema failures.

- [x] **Step 4: Initialize independent state**

In `ModelRunner.__init__` create:

```python
self.spec_verify_exact_cuda_graph_cache = (
    SpecVerifyExactCudaGraphCache(
        SpecVerifyExactCudaGraphCacheConfig(
            enabled=config.spec_verify_cuda_graphs,
            batch_allowlist=(
                config.spec_verify_cuda_graph_batch_allowlist
            ),
            query_len_allowlist=(
                config.spec_verify_cuda_graph_query_len_allowlist
            ),
            min_observations=(
                config.spec_verify_cuda_graph_min_observations
            ),
            max_entries=(
                config.spec_verify_cuda_graph_max_entries
            ),
            max_static_bytes=(
                config.spec_verify_cuda_graph_max_static_bytes
            ),
            max_reserved_bytes=(
                config.spec_verify_cuda_graph_max_reserved_bytes
            ),
            max_total_capture_ns=(
                config.spec_verify_cuda_graph_max_total_capture_ns
            ),
            max_single_capture_ns=(
                config.spec_verify_cuda_graph_max_single_capture_ns
            ),
        )
    )
)
self.last_spec_verify_cuda_graph_dispatch_event = None
self._spec_verify_cuda_graph_step_id = 0
```

Do not alter decode cache initialization or decode event state.

- [x] **Step 5: Implement admission, identity, and event publication**

Build identity only from prepared exact tensors and active model/backend
metadata. `total_query_tokens` must equal both `B * Q` and
`input_ids.numel()`. Estimate static bytes by summing exact tensor
`numel() * element_size()` values, including `[B * Q, hidden_size]` output.

Publish one event for every spec-verify dispatch attempt, including disabled
and rejected cases. Enforce:

```python
if tuple(event) != SPEC_VERIFY_DISPATCH_EVENT_FIELDS:
    raise RuntimeError(
        "spec-verify CUDA Graph dispatch event schema drift"
    )
```

- [x] **Step 6: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_model_runner_spec_verify.py \
  -k 'admission or identity or dispatch_event'
```

Expected: all selected tests pass.

---

### Task 5: Cold Eager, Post-Step Capture, and Exact Replay

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`
- Modify: `tools/test_model_runner_spec_verify_cuda_graph.py`
- Modify: `tools/test_spec_verify_capture_transaction.py`

**Interfaces:**
- Produces:

```python
def ModelRunner._capture_spec_verify_graph(
    self,
    *,
    identity: SpecVerifyGraphIdentity,
    live_input_ids,
    live_positions,
    live_context,
) -> SpecVerifyExactCudaGraphEntry:
    """Capture one exact family using only a private scratch lease."""


def ModelRunner._attempt_post_step_spec_verify_capture(
    self,
    *,
    identity: SpecVerifyGraphIdentity,
    live_input_ids,
    live_positions,
    live_context,
) -> SpecVerifyExactCudaGraphEntry | None:
    """Capture, roll back scratch, budget-check, then publish."""


def ModelRunner._replay_spec_verify_graph(
    self,
    entry: SpecVerifyExactCudaGraphEntry,
    *,
    input_ids,
    positions,
    context,
) -> object:
    """Replay one ready exact family and return transformer outputs."""
```

- [x] **Step 1: Write cold/capture/replay RED tests**

Use a fake model and fake graph to prove:

```text
observation 1 -> one eager live forward, no capture
observation 2 -> one eager live forward, one private capture forward
threshold-triggering request returns the live eager output
capture uses private scratch slots, never live write slots
capture rollback happens before cache publication
capture outputs/logits are discarded
observation 3 exact hit -> zero eager forwards, one graph replay
exact Q mismatch -> one eager forward, no replay
exact W mismatch -> one eager forward, no replay
graph output proceeds through the same compute_logits/argmax path
```

Instrument call order:

```python
events = [
    "live_eager",
    "scratch_acquire",
    "capture",
    "scratch_rollback",
    "cache_publish",
]
```

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_spec_verify_capture_transaction.py \
  -k 'cold or capture or replay or publication'
```

Expected: dispatch remains unconditional eager and capture/replay helpers are
missing.

- [x] **Step 3: Build exact static tensors**

Allocate:

```text
input_ids:    int64 [B * Q]
positions:    int64 [B * Q]
slot_mapping: int32 [B * Q]
context_lens: int32 [B]
block_tables: int32 [B, W]
outputs:      output_dtype [B * Q, hidden_size]
```

For capture:

1. acquire a private scratch lease;
2. construct scratch block tables from read-only live prefix blocks plus
   private scratch blocks;
3. clone any partially occupied terminal prefix block into its private scratch
   replacement before capture;
4. map every speculative write position to private scratch slots;
5. install `mode="spec_verify"` context with exact `(B, Q, W)` and
   `SPEC_VERIFY_FLASH_ATTN_NUM_SPLITS`;
6. capture only the target transformer forward;
7. reset context;
8. roll back the private scratch lease on success and failure;
9. publish only after rollback and post-capture budgets pass.

The capture helper must not call `compute_logits()`, argmax, acceptance, or any
live transaction finalizer.

- [x] **Step 4: Implement exact replay**

Before replay:

```text
rebuild identity and SHA
verify entry state == ready
verify exact tensor shapes/dtypes/devices
revalidate live transaction authorization
copy live tensors into static tensors
install exact spec_verify context
mark entry in flight
```

Launch:

```python
entry.graph.replay()
```

After successful launch:

```text
reset context
clear in-flight
increment replay count
update last replay/use step
return static transformer outputs
```

Do not call `compute_logits()` inside `_replay_spec_verify_graph`; the common
caller performs it exactly once for eager and graph outputs.

- [x] **Step 5: Replace unconditional spec-verify eager dispatch**

In `run_model()`:

```text
admission fail-closed -> raise
admission eager-safe -> run eager once, publish fallback
eligible identity + ready exact entry -> replay
eligible identity + miss -> run eager once, observe success
capture requested -> capture after live eager result exists
return original live eager transformer output
```

Keep decode dispatch byte-for-byte behaviorally isolated.

- [x] **Step 6: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_spec_verify_capture_transaction.py \
  -k 'cold or capture or replay or publication'
```

Expected: all selected tests pass.

---

### Task 6: Replay-Started Failure and Existing Rollback Ownership

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py`
- Modify: `tools/test_model_runner_spec_verify_cuda_graph.py`
- Modify: `tools/test_speculative_kv_transaction.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`

**Interfaces:**
- Consumes: existing `prepare_native_speculative_batch()` exception rollback
  path and `_rollback_active_transactions()`.
- Produces:

```python
class SpecVerifyGraphReplayError(RuntimeError):
    def __init__(
        self,
        identity_sha256: str,
        cause: BaseException,
    ) -> None:
        self.identity_sha256 = identity_sha256
        self.cause = cause
        super().__init__(
            "spec-verify CUDA Graph replay failed: "
            f"{identity_sha256}"
        )
```

- [x] **Step 1: Write pre-replay and replay-started failure RED tests**

Assert:

```text
identity mismatch before replay -> quarantine identity_drift, eager once
shape mismatch before replay -> quarantine shape_drift, eager once
cache-state mismatch before replay -> eager once
copy failure before replay() -> eager once when live transaction remains valid
graph.replay() raises -> no eager retry
post-replay synchronization raises -> no eager retry
replay failure -> family quarantined with replay_failed
replay failure -> no target rows returned
replay failure -> batch-runtime owner rolls back every active live transaction
graph helper never calls live rollback/precommit/commit/seal
subsequent same identity -> remains quarantined and uses eager only when a new
live transaction is valid
```

The end-to-end rollback test must create real `Sequence`,
`BlockManager`, and `SpeculativeKVTransaction` objects, then inject a tail
callback that raises `SpecVerifyGraphReplayError`.

- [x] **Step 2: Run RED**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_model_runner_callbacks.py \
  -k 'replay_failure or no_eager_retry or rollback_owner'
```

Expected: replay exceptions do not yet have the required quarantine and
rollback evidence.

- [x] **Step 3: Separate pre-launch from post-launch state**

Use an explicit local flag:

```python
replay_started = False
try:
    copy_and_revalidate()
    cache.begin_replay(entry, step_id=step_id)
    replay_started = True
    entry.graph.replay()
    synchronize_if_required_by_test_gate()
except BaseException as error:
    if replay_started:
        cache.quarantine(
            entry.identity,
            "replay_failed",
        )
        raise SpecVerifyGraphReplayError(
            entry.identity_sha256,
            error,
        ) from error
    raise
finally:
    if entry.in_flight_replays:
        cache.finish_replay(
            entry,
            step_id=step_id,
            succeeded=not replay_started or error_is_absent,
        )
    reset_context()
```

Implement the same semantics without referencing an exception variable outside
its valid scope; use a boolean `replay_succeeded`.

- [x] **Step 4: Preserve existing rollback owner**

Do not add a ModelRunner RPC that commits or rolls back scheduler
transactions. Let the exception propagate through:

```text
run_spec_verify_batch
run_model_runner_tail_batch
prepare_native_speculative_batch
_rollback_active_transactions
```

Verify the existing owner transitions each transaction from `reserved` to
`rolled_back`, releases reserved blocks, leaves sequence tokens/block tables
unchanged, and returns no speculative result.

- [x] **Step 5: Run GREEN**

```bash
python3 -m pytest -q \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_speculative_model_runner_callbacks.py \
  -k 'replay_failure or no_eager_retry or rollback_owner'
```

Expected: all selected tests pass.

---

### Task 7: Public Verifier Semantics and Full CPU Regression

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_spec_verify_batch_contract.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Preserves:

```python
ModelRunner.run_spec_verify_batch(
    items: tuple[object, ...],
    residency_ticket_id: int | None = None,
) -> tuple[SpecVerifyBatchResultRow, ...] | None
```

- [x] **Step 1: Add public-path regression tests**

Prove:

```text
run_spec_verify_batch prepares exactly once
run_model executes exactly once
rank 0 computes logits/argmax exactly once
row splitter preserves input order
batch runtime marks the live transaction materialized exactly once after the
tail callback returns
graph capture never marks the live transaction materialized
feature disabled matches existing eager outputs
TP>1 remains eager
kv_offload_mvp0 remains eager
blockwise remains eager
decode exact-width graph tests remain unchanged
fixed-Q grouping still performs one RPC per distinct exact Q
```

- [x] **Step 2: Run focused RED/GREEN gate**

Run after adding tests; any failure is a regression to fix within Tasks 1-6:

```bash
python3 -m pytest -q \
  tools/test_spec_verify_exact_cuda_graph_cache.py \
  tools/test_spec_verify_cuda_graph_config.py \
  tools/test_spec_verify_capture_transaction.py \
  tools/test_model_runner_spec_verify_cuda_graph.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_spec_verify_batch_contract.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_kv_transaction.py \
  tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: all tests pass.

- [x] **Step 3: Run broader speculative/KV regression**

```bash
python3 -m pytest -q \
  tools/test_speculative_batch_runtime.py \
  tools/test_engine_speculative_execution.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_scheduler_speculative_selection.py \
  tools/test_speculative_residency.py \
  tools/test_prefix_kv_offload_integration.py \
  tools/test_chunked_prefill.py \
  tools/test_blockwise_speculative_verifier_gate.py
```

Expected: all listed existing files collect and pass.

- [x] **Step 4: Run source and compatibility checks**

```bash
python3.12 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py \
  tinyvllm/engine/block_manager.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/speculative/batch_runtime.py

rg -n \
  'Qwen|qwen|proposal_source|draft_model_name' \
  tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/speculative/batch_runtime.py

git diff --check
```

Expected:

```text
py_compile: exit 0
generic-source scan: no matches
git diff --check: exit 0
```

---

### Task 8: CUDA Correctness and Fail-Closed Gate

**Files:**
- Create: `tools/spec_verify_cuda_graph_smoke.py`
- Create: `tools/run_spec_verify_cuda_graph_gate_remote.py`
- Create: `tools/verify_spec_verify_cuda_graph_gate.py`
- Create: `tools/test_spec_verify_cuda_graph_gate.py`

**Interfaces:**
- Produces one JSON artifact containing:

```text
schema_version
source_sha256
model
checkpoint
device_name
device_compute_capability
torch_version
cuda_version
flash_attn_version
configuration
families
eager_baseline
warmed_graph
replay_failure_injection
transaction_results
classification
```

- [x] **Step 1: Write verifier RED tests**

The verifier must reject evidence when any required family lacks:

```text
exact eager-versus-graph logits parity
exact greedy target-token parity
accepted-length parity
final-token parity
accepted-prefix KV parity
rejected-suffix release evidence
one target forward on warmed hit
zero eager retries after injected replay failure
stable quarantine reason
source/config/device identity
```

Require:

```text
B in {1, 4}
every explicitly enabled Q
at least two exact W values
TP == 1
kv_offload_mvp0 == false
context length == 4096
classification == NOT_PROMOTABLE
```

- [x] **Step 2: Run verifier RED**

```bash
python3 -m pytest -q \
  tools/test_spec_verify_cuda_graph_gate.py
```

Expected: verifier module is missing.

- [x] **Step 3: Implement CUDA smoke**

For each configured exact family:

1. run eager baseline from identical sequence/transaction state;
2. run cold observations until capture;
3. run warmed exact hit;
4. compare logits with the repository's deterministic dtype-specific
   tolerance and require exact argmax parity;
5. execute acceptance/finalize outside the graph;
6. compare final sequence tokens and accepted KV;
7. verify rejected suffix blocks are released;
8. count target forwards and prove warmed replay adds no eager forward;
9. inject `graph.replay()` failure and prove no eager retry;
10. record the quarantined family and propagated error.

- [x] **Step 4: Implement fail-closed evidence verifier**

`tools/verify_spec_verify_cuda_graph_gate.py` exits non-zero for missing,
malformed, stale, partial, or contradictory evidence. It must explicitly
reject:

```text
TP > 1 claims
KV offload claims
H2D/D2H benefit claims
rounded B/Q/W identities
capture latency mixed into warmed-hit latency
missing failure injection
PROMOTABLE classification
```

- [ ] **Step 5: Run CUDA correctness gate**

Use the existing TinyLLMForge remote execution route and a pinned checkout:

```bash
python3 tools/run_spec_verify_cuda_graph_gate_remote.py \
  --context-length 4096 \
  --batch-sizes 1 4 \
  --output-json /tmp/spec_verify_cuda_graph_gate.json

python3 tools/verify_spec_verify_cuda_graph_gate.py \
  /tmp/spec_verify_cuda_graph_gate.json
```

Expected:

```text
all configured exact families pass
two or more exact W values pass
failure injection passes
classification=NOT_PROMOTABLE
```

Do not claim CUDA correctness if the remote environment, checkpoint,
FlashAttention backend, or required GPU is unavailable; record the exact
blocker instead.

2026-08-12 live preflight status: blocked before source upload. The
configured remote Python, repository, CUDA runtime, eight A100 80GB GPUs,
and both Qwen3-0.6B checkpoint candidates were present, but every GPU had
at least one active compute process. GPU 5 was the least occupied at
approximately 1.3 GiB and 0% sampled utilization, but it was still owned
by a root `python server.py` process that had been running for more than
four days. The fail-closed runner therefore did not share the GPU, kill
the process, upload source, or execute the CUDA gate.

The fail-closed preflight is also preserved as machine-readable evidence:

```text
experiments/spec_verify_cuda_graph/
  preflight-20260812-idle-gpu-blocked.json
status:                 BLOCKED
runner exit code:       2
source_upload_started:  false
cuda_gate_started:      false
```

---

### Task 9: Performance Evidence, Documentation, and Completion Audit

**Files:**
- Modify: `tools/run_spec_verify_cuda_graph_gate_remote.py`
- Modify: `tools/verify_spec_verify_cuda_graph_gate.py`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-variable-q-speculative-cuda-graph.md`

**Interfaces:**
- Produces separate metrics for:

```text
warmed exact-family graph-hit verifier latency
mixed-hit-rate end-to-end TPOT
TTFT
throughput
GPU allocated/reserved memory
capture duration
capture allocated/reserved deltas
cache hit/miss/eviction/quarantine counts
acceptance
```

- [x] **Step 1: Add controlled performance measurements**

Pin and record:

```text
source SHA
model/checkpoint
GPU and compute capability
PyTorch/CUDA/FlashAttention versions
all graph configuration fields
prompt lengths
proposal-length distribution
batch distribution
warmup count
measurement count
eager baseline
```

Report warmed hits separately from mixed-hit-rate end-to-end runs. Include
batch 1 and 4. Do not report H2D or D2H improvement because
`kv_offload_mvp0=False`.

2026-08-12 implementation evidence:

```text
exact families:                    8
warmup observations per family:   2 (cold eager + capture)
warmed measurements per family:   5
total warmed measurements:        40
mixed observations:               56 (40 hits + 16 misses)
verifier scope:                    exact B=(1,4), Q=(1,3), W=(1,2)
performance classification:       NOT_PROMOTABLE
```

The producer records source identity, model/checkpoint, device and software
versions, exact graph configuration, prompt lengths, proposal and batch
distributions, eager metrics, warmed exact-hit latency, mixed TPOT/TTFT and
throughput, memory, capture deltas, cache counters, and acceptance. The
verifier fail-closes on any non-MVP warmup count, total/per-family measurement
count, Q distribution, or batch distribution.

Fresh local validation:

```text
focused smoke/verifier/remote-runner tests: 93 passed in 0.22s
py_compile:                                PASS
scoped git diff --check:                   PASS
placeholder scan:                          PASS
```

This completes the measurement producer/schema implementation only. It is not
real CUDA performance evidence and does not complete Step 2.

- [ ] **Step 2: Run performance gate**

```bash
python3 tools/run_spec_verify_cuda_graph_gate_remote.py \
  --context-length 4096 \
  --batch-sizes 1 4 \
  --measure-performance \
  --output-json /tmp/spec_verify_cuda_graph_perf.json

python3 tools/verify_spec_verify_cuda_graph_gate.py \
  /tmp/spec_verify_cuda_graph_perf.json
```

Expected:

```text
warmed graph-hit latency is reported separately
mixed-hit-rate TPOT is reported
TTFT regression is reported
memory and capture stalls are reported
no H2D/D2H benefit field is present
```

Performance improvement is evidence, not a correctness substitute.

2026-08-12 refreshed remote preflight:

```text
artifact:
  experiments/spec_verify_cuda_graph/
    preflight-20260812-task9-refresh-idle-gpu-blocked.json
status:                 BLOCKED
error:                  no idle GPU is available
idle_gpu_indices:       []
source_upload_started:  false
cuda_gate_started:      false
runner exit code:       2
```

All eight A100 devices, the remote Python/repository, CUDA runtime, and both
checkpoint candidates remain visible. Step 2 remains open because the runner
correctly refused to share an occupied GPU.

- [ ] **Step 3: Run final full gate**

Run Tasks 7 and 8 commands again from a clean command invocation, then:

```bash
git diff --check

rg -n 'TODO|TBD|implement later|fill in details' \
  tinyvllm/engine/spec_verify_exact_cuda_graph_cache.py \
  tools/spec_verify_cuda_graph_smoke.py \
  tools/run_spec_verify_cuda_graph_gate_remote.py \
  tools/verify_spec_verify_cuda_graph_gate.py

git status --short
```

Expected:

```text
all CPU tests pass
CUDA evidence verifier passes
git diff --check passes
placeholder scan has no matches
no staged files
```

2026-08-12 partial audit status:

```text
focused CPU/schema tests:   PASS (93)
py_compile:                 PASS
scoped diff check:          PASS
placeholder scan:           PASS
global git diff --check:    BLOCKED by pre-existing trailing whitespace in
                            tinyvllm/engine/model_runner.py
CUDA correctness evidence: BLOCKED (no idle remote GPU)
CUDA performance evidence: BLOCKED (no idle remote GPU)
```

Do not mark this step complete until the global workspace issue and both CUDA
evidence gates are resolved.

- [x] **Step 4: Update handoff evidence**

Record in `AGENT_HANDOFF_STATE.md`:

```text
implemented exact configuration and cache
live transaction authorization contract
capture-only private scratch ownership
exact B/Q/W families tested
CPU commands and counts
CUDA command and artifact path
correctness parity result
replay-failure result
performance metrics and limitations
TP1/no-offload scope
NOT_PROMOTABLE classification
next missing gates: TP4, KV offload, 16K/32K, second model structure,
learned/MTP executor, real H2D/D2H counters
```

- [x] **Step 5: Complete the prompt-to-artifact audit**

Before marking this plan complete, map every approved requirement to evidence:

```text
TP1 only -> admission test + CUDA artifact
no KV offload -> admission test + artifact config
exact-Q/no padding -> grouping regression + identity tests
B 1/4 -> CPU dispatch tests + CUDA families
two W values -> CUDA families
transaction authorization -> block-manager tests
accepted KV in-place -> transaction/CUDA parity evidence
rejected suffix rollback -> transaction/CUDA evidence
cold eager -> dispatch tests
post-step private capture -> scratch tests
ready replay -> dispatch/CUDA tests
pre-replay fallback -> dispatch tests
post-replay no retry -> injected failure evidence
quarantine -> cache and injected failure evidence
common logits/argmax -> public-path tests
single materialization/finalize -> batch-runtime tests
decode isolation -> decode graph regression
observability schema -> fixed-field test
performance separation -> verified JSON
no H2D/D2H claim -> verifier rejection rule
NOT_PROMOTABLE -> artifact and handoff
```

Treat any missing or weak evidence as incomplete and keep the corresponding
checkbox open.

- [ ] **Step 6: Mark plan complete only after fresh verification**

Change this plan's checkboxes from `[ ]` to `[x]` only for steps backed by
fresh command output and artifacts. Do not mark CUDA or performance steps
complete from CPU-only tests.

---

## Execution Order and Review Checkpoints

Execute inline in this order:

```text
Checkpoint A: Task 1
  exact identity, configuration, cache lifecycle

Checkpoint B: Tasks 2-3
  private capture scratch and live transaction authorization

Checkpoint C: Tasks 4-6
  admission, dispatch, capture, replay, fail-closed errors

Checkpoint D: Task 7
  full CPU/source regression

Checkpoint E: Tasks 8-9
  CUDA correctness, performance evidence, handoff, completion audit
```

At each checkpoint, inspect the focused diff and run only the named focused
tests before broadening validation. Do not start the CUDA gate until the full
CPU/source regression is green.
