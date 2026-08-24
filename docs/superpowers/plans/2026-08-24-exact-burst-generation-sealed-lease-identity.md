# Exact-Burst Generation-Sealed Lease Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This repository's active constraint
> forbids subagents and additional worktrees, so execute inline in the
> authoritative checkout with `executing-plans`.

**Goal:** Replace repeated full block-table identity construction and
validation on stable one-phase exact-greedy K8 leases with a fail-closed,
constant-time generation seal.

**Architecture:** Make `Sequence.block_table` a list-compatible
mutation-tracked container, add a BlockManager ownership generation, and
capture an immutable identity seal only when either generation changes. Keep
the existing full-identity path as the default and as the fallback authority;
the candidate changes only lease identity bookkeeping and never changes model,
CUDA Graph, KV-slot, token, or D2H behavior.

**Tech Stack:** Python 3, dataclasses, pickle-compatible Sequence state,
pytest, TinyLLMForge scheduler/block manager, SHA256 receipts, JSON/JSONL
evidence, SSH remote controller, Qwen3-0.6B on one strict-clean A100.

## Global Constraints

- The only authoritative checkout is
  `/Users/bytedance/Desktop/TinyLLMForge`, resolving to
  `/Users/bytedance/dev/TinyLLMForge`.
- Stay on `feat/kv-sparse-attention`; do not create a worktree or use a
  subagent.
- Complete and reconcile the r10 one-phase journal gate before modifying any
  source file named in its source manifest.
- Preserve every unrelated dirty or untracked file. Stage only exact task
  paths.
- Use `git -c core.hooksPath=/dev/null commit`.
- Every commit has exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- Use `python3 -m pytest`; `python` is unavailable locally.
- The new feature remains default-disabled.
- The baseline arm keeps the current full `(block_id, generation)` identity.
- The candidate arm may use a hot seal only when both the sequence table
  revision and BlockManager ownership generation are unchanged.
- Generation drift rejects the optimized transaction before mutation; it
  never refreshes a stale lease.
- Preserve exact output tokens, sampled logits, argmax, target forwards, graph
  replays, D2H calls/bytes, KV slots, queue order, and rollback semantics.
- Hardware data may be written only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not run `kinit`, kill external processes, or reuse an attempted run tag.
- Strict-clean GPU admission is memory `<=1024 MiB`, utilization `<=5%`, and
  no compute process.
- A GPU gate source SHA must equal the already-pushed branch HEAD at launch.
- Partial rows never authorize a performance classification.
- Report both benefit and cost.

---

## File Map

- Modify `tinyvllm/engine/sequence.py`: add the list-compatible
  `VersionedBlockTable`, property wrapping, and backward-compatible
  serialization.
- Modify `tinyvllm/engine/block_manager.py`: add ownership generation and the
  immutable `BlockTableIdentitySeal` capture/validation API.
- Modify `tinyvllm/config.py`: add the default-off candidate flag and strict
  dependency validation.
- Modify `tinyvllm/engine/exact_greedy_decode_burst.py`: add optional sealed
  lease identity and lifecycle counters.
- Modify `tinyvllm/engine/scheduler.py`: select sealed capture/validation for
  eligible one-phase K8 leases.
- Create `tools/test_generation_sealed_block_table.py`: focused mutation,
  replacement, and serialization tests.
- Keep `tools/test_hybrid_state_sequence.py` as adjacent legacy-state
  regression coverage.
- Modify `tools/test_chunked_prefill.py`: BlockManager ownership-generation
  and allocation compatibility tests.
- Modify `tools/test_model_runner_spec_verify.py`: configuration and
  cross-rank lease serialization tests.
- Modify `tools/test_scheduler_prepared_postprocess.py`: stale-seal, commit,
  fallback, and rollback tests.
- Create
  `tools/profile_exact_burst_generation_sealed_lease_identity.py`: CPU
  lifecycle profile.
- Create
  `tools/test_profile_exact_burst_generation_sealed_lease_identity.py`.
- Create `tools/exact_burst_generation_sealed_lease_identity_gate.py`:
  source-bound paired GPU gate.
- Create
  `tools/test_exact_burst_generation_sealed_lease_identity_gate.py`.
- Create `tools/exact_burst_generation_sealed_lease_identity_verify.py`:
  independent verifier.
- Create
  `tools/test_exact_burst_generation_sealed_lease_identity_verify.py`.
- Create `tools/run_exact_burst_generation_sealed_lease_identity_remote.py`:
  safe remote controller.
- Create
  `tools/test_run_exact_burst_generation_sealed_lease_identity_remote.py`.
- Create
  `docs/superpowers/audits/2026-08-24-exact-burst-generation-sealed-lease-identity-audit.md`.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`.
- Modify `AGENT_HANDOFF_STATE.md`.

### Task 1: Add a Mutation-Tracked Sequence Block Table

**Files:**

- Modify: `tinyvllm/engine/sequence.py`
- Create: `tools/test_generation_sealed_block_table.py`
- Modify: `tools/test_chunked_prefill.py`

**Interfaces:**

- Produces:
  `VersionedBlockTable(values: Iterable[int] = (), *, revision: int = 0)`
- Produces: `VersionedBlockTable.revision: int`
- Preserves: ordinary list indexing, slicing, equality, iteration, and
  mutation APIs.
- Preserves: legacy Sequence pickle states.

- [ ] **Step 1: Write RED mutation tests**

Add tests that construct a `Sequence`, retain its `block_table` object, and
assert every supported mutation changes `revision` exactly once:

```python
table = sequence.block_table
start = table.revision
table.append(1)
assert table.revision == start + 1
table.extend([2, 3])
assert table.revision == start + 2
table[1] = 4
assert table.revision == start + 3
del table[1]
assert table.revision == start + 4
table += [5]
assert table.revision == start + 5
table.pop()
assert table.revision == start + 6
table.clear()
assert table.revision == start + 7
```

Add separate coverage for `insert`, `remove`, slice assignment, slice
deletion, `reverse`, `sort`, and `*=`.
Assert direct assignment to `table.revision` raises `AttributeError`.

Exercise augmented assignment through the property, not only through a local
table variable:

```python
table = sequence.block_table
start = table.revision
sequence.block_table += [6]
assert sequence.block_table is table
assert table.revision == start + 1
sequence.block_table *= 2
assert sequence.block_table is table
assert table.revision == start + 2
```

- [ ] **Step 2: Write RED replacement and serialization tests**

Require:

```python
sequence.block_table = [7, 8]
assert list(sequence.block_table) == [7, 8]
assert sequence.block_table.revision > prior_revision

restored = pickle.loads(pickle.dumps(sequence))
assert list(restored.block_table) == [7, 8]
assert restored.block_table.revision == sequence.block_table.revision
```

Pass existing 16-, 15-, 14-, 13-, 11-, and 5-field legacy states to
`__setstate__` and assert they restore a tracked table with revision zero.
Assert the new state's block-table field is a plain list rather than a
`VersionedBlockTable`.

- [ ] **Step 3: Run the Sequence RED tests**

Run:

```bash
python3 -m pytest \
  tools/test_generation_sealed_block_table.py \
  tools/test_chunked_prefill.py::test_sequence_pickle_preserves_mixed_step_metadata_for_tp_workers \
  -q
```

Expected: failures because `VersionedBlockTable` and revision-preserving state
do not exist.

- [ ] **Step 4: Implement `VersionedBlockTable`**

Implement a private `_bump()` method that raises on integer overflow and
override every list mutator used by Python's list API:

```python
class VersionedBlockTable(list):
    __slots__ = ("_revision",)

    def __init__(self, values=(), *, revision: int = 0):
        super().__init__(values)
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
        ):
            raise ValueError(
                "block-table revision must be a non-negative integer"
            )
        self._revision = revision

    @property
    def revision(self) -> int:
        return self._revision

    def _bump(self) -> None:
        if self.revision >= (1 << 63) - 1:
            raise OverflowError("block-table revision exhausted")
        self._revision += 1
```

Every mutator calls `_bump()` exactly once before delegating to the list
operation. This makes failure conservative: even if an iterable, comparison,
or other user-provided callback raises after a partial mutation, every prior
seal is already invalid. A failed mutation may therefore advance the revision
without changing the final list contents.

- [ ] **Step 5: Wrap `Sequence.block_table`**

Store `_block_table` internally and expose:

```python
@property
def block_table(self) -> VersionedBlockTable:
    return self._block_table

@block_table.setter
def block_table(self, values) -> None:
    prior = getattr(self, "_block_table", None)
    if values is prior:
        return
    revision = 0 if prior is None else prior.revision + 1
    self._block_table = VersionedBlockTable(
        values,
        revision=revision,
    )
```

Emit a new 17-field Sequence pickle state with `list(self.block_table)` at
index 3, the block-table revision at index 15, and the prompt token list or
decode last-token payload still at index 16. In `__setstate__`, accept the new
state length and all legacy lengths. For a 17-field state, validate the
revision and construct `_block_table` directly with
`VersionedBlockTable(block_ids, revision=revision)`; for legacy states, route
the IDs through the property so revision starts at zero. Do not add a public
revision setter. Existing code that treats `state[-1]` as the token payload
must remain valid.

- [ ] **Step 6: Run Task 1 GREEN and adjacent tests**

Run:

```bash
python3 -m pytest \
  tools/test_generation_sealed_block_table.py \
  tools/test_hybrid_state_sequence.py \
  tools/test_chunked_prefill.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_speculative_kv_transaction.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit and push Task 1**

```bash
git add -- \
  tinyvllm/engine/sequence.py \
  tools/test_generation_sealed_block_table.py \
  tools/test_chunked_prefill.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): version sequence block tables" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Add Block Ownership Generations and Identity Seals

**Files:**

- Modify: `tinyvllm/engine/block_manager.py`
- Modify: `tools/test_chunked_prefill.py`
- Modify: `tools/test_speculative_kv_transaction.py`

**Interfaces:**

- Produces:
  `BlockTableIdentitySeal`
- Produces:
  `BlockManager.capture_block_table_identity(sequence, write_block_index)`
- Produces:
  `BlockManager.validate_block_table_identity(sequence, seal)`
- Produces:
  `BlockManager.ownership_generation: int`

- [ ] **Step 1: Write RED ownership-generation tests**

For allocate, cached activation, extra decode allocation, reference release,
deallocation, speculative reservation, commit, and rollback restoration:

```python
before = manager.ownership_generation
manager.allocate(sequence)
assert manager.ownership_generation > before
```

Also assert `publish_full_blocks()` does not increment the generation when it
only publishes hash/token metadata.

- [ ] **Step 2: Write RED seal capture and validation tests**

Require:

```python
seal = manager.capture_block_table_identity(
    sequence,
    write_block_index=len(sequence.block_table) - 1,
)
same = manager.capture_block_table_identity(
    sequence,
    write_block_index=len(sequence.block_table) - 1,
)
assert same is seal
manager.validate_block_table_identity(sequence, seal)
```

Mutation cases must reject:

```text
block-table append
same-length block-table replacement
write-block replacement
block allocation or deallocation
block generation change
predecessor replacement
wrong sequence
malformed seal
```

- [ ] **Step 3: Run Task 2 RED**

Run:

```bash
python3 -m pytest \
  tools/test_chunked_prefill.py \
  tools/test_speculative_kv_transaction.py \
  -q
```

Expected: focused new tests fail because ownership generations and seals are
absent.

- [ ] **Step 4: Implement ownership generation**

Initialize:

```python
self.ownership_generation = 0
```

Add:

```python
def _advance_ownership_generation(self) -> None:
    if self.ownership_generation >= (1 << 63) - 1:
        raise OverflowError(
            "block ownership generation exhausted"
        )
    self.ownership_generation += 1
```

Call it at the lowest mutation-authority points, including
`_allocate_block()`, `_activate_cached_block()`, `_deallocate_block()`, direct
reference-count changes, reservation adoption/release, and rollback state
restoration. It is an invalidation epoch rather than an operation counter:
one logical transaction may advance it more than once. For rollback or
multi-step restoration, advance before the first mutation so even a partial
failure invalidates every older seal. Tests assert monotonicity and
invalidation, not an exact delta for multi-block transactions.

- [ ] **Step 5: Implement the immutable seal**

Add:

```python
@dataclass(frozen=True)
class BlockTableIdentitySeal:
    sequence_id: int
    table_revision: int
    ownership_generation: int
    block_count: int
    write_block_index: int
    write_block_id: int
    write_block_generation: int
    predecessor_block_id: int | None
    predecessor_block_generation: int | None
    identity_sha256: str
```

Cold capture uses the existing `block_identities()` result to derive a
canonical SHA256 and caches:

```text
tracked table object
table revision
ownership generation
write-block index
seal
complete identity rows
```

Hot capture returns the identical seal object only when all keys match.

- [ ] **Step 6: Implement constant-time validation**

`validate_block_table_identity()` checks the current revision, ownership
generation, block count, write block, and predecessor fields. It must not
iterate the block table or call `block_identities()` on a hot valid seal.

Expose a counter or test hook so focused tests can prove zero rows were
visited on hot capture and validation.

- [ ] **Step 7: Run Task 2 GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_chunked_prefill.py \
  tools/test_speculative_kv_transaction.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit and push Task 2**

```bash
git add -- \
  tinyvllm/engine/block_manager.py \
  tools/test_chunked_prefill.py \
  tools/test_speculative_kv_transaction.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(kv): seal stable block-table identities" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Bind Generation Seals to Exact-Burst Leases

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/exact_greedy_decode_burst.py`
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_exact_greedy_decode_burst.py`
- Modify: `tools/test_scheduler_prepared_postprocess.py`

**Interfaces:**

- Consumes: `BlockTableIdentitySeal`
- Produces:
  `Config.exact_greedy_decode_burst_generation_sealed_identity`
- Produces sealed/full identity mode in `ExactGreedyDecodeBurstLease`
- Produces the same sealed/full identity mode in
  `ExactGreedyDecodeBurstContinuationReceipt`
- Produces lifecycle counters:
  `identity_seal_cold_captures`, `identity_seal_hot_reuses`,
  `identity_seal_validations`, `identity_seal_fallback_counts`

- [ ] **Step 1: Write RED configuration tests**

Require a default of `False`. Enabling it requires:

```text
exact_greedy_decode_burst == true
exact_greedy_decode_burst_tokens == 8
exact_greedy_decode_burst_lease_local_delta_journal == true
```

Non-boolean values and missing prerequisites raise exact `ValueError`
messages.

- [ ] **Step 2: Write RED lease serialization tests**

Construct one baseline lease and one sealed lease. Require:

```python
assert baseline.block_table_identity
assert baseline.block_table_identity_seal is None
assert sealed.block_table_identity == ()
assert sealed.block_table_identity_seal == seal
assert pickle.loads(pickle.dumps(sealed)) == sealed
```

The lease SHA must change if any seal field changes and remain deterministic
across pickle round trips. Require the baseline canonical payload and digest
to remain identical to the pre-feature implementation.

Construct matching continuation receipts for both modes. Require:

```python
assert baseline_receipt.block_table_identity == baseline.block_table_identity
assert baseline_receipt.block_table_identity_seal is None
assert sealed_receipt.block_table_identity == ()
assert sealed_receipt.block_table_identity_seal == seal
```

Reject receipts that contain both representations or neither representation.

- [ ] **Step 3: Write RED scheduler hot-reuse tests**

For a stable 8K sequence:

1. grant one K8 lease and complete it;
2. grant a second lease without crossing a block boundary;
3. assert one cold capture and one hot reuse;
4. instrument `block_identities()` and assert it is called only for the first
   lease; and
5. assert the second replay is a continuation hit rather than a forced
   block-table reload; and
6. assert both commits produce the same tokens and state as the baseline.

- [ ] **Step 4: Write RED stale-seal fault matrix**

Between lease grant and prepare/commit, independently inject:

```text
block-table append
same-length index replacement
whole-table replacement
write-block generation change
unrelated BlockManager allocation
deallocation
ownership rollback restoration
```

Every case must reject before token mutation. For cases that can be known
before journal capture, require generic fallback; for post-capture drift,
require the existing terminal rollback error semantics.

- [ ] **Step 5: Run Task 3 RED**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py::test_exact_greedy_decode_burst_config_is_strict_and_default_off \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_scheduler_prepared_postprocess.py \
  -q
```

Expected: new tests fail because the sealed mode and counters are absent.

- [ ] **Step 6: Implement sealed lease construction**

When the flag is disabled, keep the current full identity path byte-for-byte.
When enabled, call:

```python
seal = self.block_manager.capture_block_table_identity(
    sequence,
    write_block_index=write_block_index,
)
```

Build the lease with `block_table_identity=()` and
`block_table_identity_seal=seal`. The canonical lease payload includes the
seal fields and digest, not the complete identity rows.

Extend `ExactGreedyDecodeBurstContinuationReceipt` with the same mutually
exclusive identity representations. Receipt construction copies the active
representation from the lease. Continuation matching compares the full tuple
for baseline leases and the immutable seal for sealed leases. If the modes
differ, or either representation drifts, record a continuation miss.

- [ ] **Step 7: Implement sealed validation**

In `_validate_pending_exact_greedy_decode_burst()`, dispatch by identity mode:

```python
if lease.block_table_identity_seal is not None:
    self.block_manager.validate_block_table_identity(
        sequence,
        lease.block_table_identity_seal,
    )
else:
    self.block_manager.validate_block_identities(
        lease.block_table_identity
    )
```

Update the lease-local journal to retain and validate the seal without
constructing a second block-table tuple.

- [ ] **Step 8: Add lifecycle accounting**

Record exactly one outcome per candidate capture:

```text
cold capture
hot reuse
fallback(reason)
```

Record every constant-time validation. Reject unknown fallback reasons in
summary validation.

- [ ] **Step 9: Run Task 3 GREEN and adjacent tests**

Run:

```bash
python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py \
  tools/test_exact_burst_continuation_epoch.py \
  tools/test_exact_burst_ragged_coalescing.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit and push Task 3**

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/exact_greedy_decode_burst.py \
  tinyvllm/engine/scheduler.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_scheduler_prepared_postprocess.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): reuse generation-sealed burst identity" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Add the CPU Lifecycle Profile

**Files:**

- Create:
  `tools/profile_exact_burst_generation_sealed_lease_identity.py`
- Create:
  `tools/test_profile_exact_burst_generation_sealed_lease_identity.py`

**Interfaces:**

- Produces schema:
  `exact_burst_generation_sealed_lease_identity_cpu_profile_v1`
- Produces policies: `full_identity`, `generation_sealed`
- Produces artifacts: `rows.jsonl`, `summary.json`

- [ ] **Step 1: Write RED profile contract tests**

Require:

```python
CONTEXT_LENGTHS = (249, 2041, 8185)
POLICIES = ("full_identity", "generation_sealed")
DEFAULT_REPETITIONS = 100
```

Each row contains:

```text
lease_grant_median_us
lease_grant_p95_us
lease_lifecycle_median_us
lease_lifecycle_p95_us
identity_rows_visited
identity_seal_cold_captures
identity_seal_hot_reuses
identity_seal_validations
positive_python_allocation_bytes
fallback_counts
```

- [ ] **Step 2: Run the profile RED test**

Run:

```bash
python3 -m pytest \
  tools/test_profile_exact_burst_generation_sealed_lease_identity.py \
  -q
```

Expected: collection failure because the module is absent.

- [ ] **Step 3: Implement the profile through production entrypoints**

For every sample:

1. clear the prior pending lease only through the production
   commit/rollback lifecycle;
2. time `prepare_exact_greedy_decode_burst()`;
3. construct an exact K8 result;
4. time prepare plus rollback or commit;
5. preserve the sequence layout between hot samples; and
6. force exactly one cold capture per fixture.

Do not replace production methods with no-op timing stubs.

- [ ] **Step 4: Run profile GREEN and emit local evidence**

Run:

```bash
python3 -m pytest \
  tools/test_profile_exact_burst_generation_sealed_lease_identity.py \
  -q
python3 tools/profile_exact_burst_generation_sealed_lease_identity.py \
  --output-dir \
  artifacts/exact_burst_generation_sealed_lease_identity/cpu-profile-local
```

Expected:

```text
8K lifecycle median improvement >= 30%
8K lifecycle P95 improvement >= 25%
aggregate lifecycle median improvement >= 20%
candidate hot-path identity rows visited == 0
candidate cold captures == 1 per fixture
candidate fallback/rollback failures == 0
```

- [ ] **Step 5: Commit and push Task 4**

```bash
git add -- \
  tools/profile_exact_burst_generation_sealed_lease_identity.py \
  tools/test_profile_exact_burst_generation_sealed_lease_identity.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): profile generation-sealed burst identity" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Do not commit generated profile artifacts.

### Task 5: Add the Paired GPU Gate and Independent Verifier

**Files:**

- Create: `tools/exact_burst_generation_sealed_lease_identity_gate.py`
- Create:
  `tools/test_exact_burst_generation_sealed_lease_identity_gate.py`
- Create:
  `tools/exact_burst_generation_sealed_lease_identity_verify.py`
- Create:
  `tools/test_exact_burst_generation_sealed_lease_identity_verify.py`

**Interfaces:**

- Produces performance schema:
  `exact_burst_generation_sealed_lease_identity_performance_v1`
- Produces correctness schema:
  `exact_burst_generation_sealed_lease_identity_correctness_v1`
- Produces gate schema:
  `exact_burst_generation_sealed_lease_identity_gate_v1`
- Produces GO:
  `GO_EXACT_BURST_GENERATION_SEALED_LEASE_IDENTITY`

- [ ] **Step 1: Write RED inventory and paired-order tests**

Require:

```python
POLICIES = ("full_identity", "generation_sealed")
CONTEXTS = ("2k", "4k", "8k")
PERFORMANCE_REPETITIONS = 10
PERFORMANCE_ROW_COUNT = 60
CORRECTNESS_ROW_COUNT = 24
```

Reject duplicate, missing, mixed-tag, mixed-source, non-finite, wrong-schema,
or wrong-policy rows.

- [ ] **Step 2: Write RED threshold tests**

Synthetic GO evidence must satisfy:

```text
exact output-token/logit/argmax parity
unchanged target forwards, graph replays, and D2H
candidate stale/fallback/rollback count == 0
hot reuses == eligible bursts minus cold captures
8K lifecycle median/P95 improvement >= 25%
aggregate lifecycle median/P95 improvement >= 15%
aggregate TPOT median/P95 improvement >= 0.5%
TTFT/E2E/TPOT-P99/throughput regression <= 2%
allocated/reserved memory regression <= 1%
```

Perturb one invariant per negative test and require the corresponding
`NO_GO_CORRECTNESS`, `NO_GO_TRANSACTIONAL_SAFETY`,
`NO_GO_PERFORMANCE`, or `NO_GO_EVIDENCE_INCOMPLETE`.

- [ ] **Step 3: Implement the paired gate**

Reuse the canonical one-phase K8 gate entrypoint. Both arms use:

```text
exact_greedy_decode_burst=true
exact_greedy_decode_burst_tokens=8
exact_greedy_decode_burst_split_phase=false
exact_greedy_decode_burst_lease_local_delta_journal=true
```

Only the generation-sealed identity flag differs.

- [ ] **Step 4: Implement the independent verifier**

Independently validate:

- exact source/workload/runner manifests and hashes;
- fixed row inventories and alternating policy order;
- source SHA and empty source patch;
- exact correctness and execution counters;
- lifecycle counters and generation-seal accounting;
- all benefit/cost thresholds; and
- producer/verifier classification equality.

- [ ] **Step 5: Run Task 5 GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_exact_burst_generation_sealed_lease_identity_gate.py \
  tools/test_exact_burst_generation_sealed_lease_identity_verify.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit and push Task 5**

```bash
git add -- \
  tools/exact_burst_generation_sealed_lease_identity_gate.py \
  tools/test_exact_burst_generation_sealed_lease_identity_gate.py \
  tools/exact_burst_generation_sealed_lease_identity_verify.py \
  tools/test_exact_burst_generation_sealed_lease_identity_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): gate generation-sealed burst identity" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 6: Add the Safe Remote Controller

**Files:**

- Create:
  `tools/run_exact_burst_generation_sealed_lease_identity_remote.py`
- Create:
  `tools/test_run_exact_burst_generation_sealed_lease_identity_remote.py`

**Interfaces:**

- Consumes an already-pushed source SHA and unique run tag.
- Produces source-bound preflight, worker receipt, remote verifier receipt,
  local verifier receipt, and preserved terminal bundle.

- [ ] **Step 1: Write RED controller safety tests**

Require:

```text
all remote paths below the approved mounted root
all TMP/cache paths below run staging
source SHA equals pushed branch HEAD
Kerberos lifetime >= 5400 seconds
strict-clean GPU admission and immediate recheck
attempted tag rejection
receipt-driven exactly-once worker launch
idempotent resume without relaunch
bounded retry only for read-only/idempotent operations
no kinit
no external-process termination
```

- [ ] **Step 2: Run controller RED**

Run:

```bash
python3 -m pytest \
  tools/test_run_exact_burst_generation_sealed_lease_identity_remote.py \
  -q
```

Expected: collection failure because the controller is absent.

- [ ] **Step 3: Implement controller and resume mode**

Use:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  exact-burst-generation-sealed-lease-identity/
```

Add explicit mutually exclusive modes:

```text
--launch
--resume-existing
```

`--resume-existing` requires existing local and remote launch receipts, checks
their run tag/source SHA/PID/PGID equality, polls only that worker, and never
executes the launch command.

- [ ] **Step 4: Run controller and full focused GREEN**

Run:

```bash
python3 -m pytest \
  tools/test_run_exact_burst_generation_sealed_lease_identity_remote.py \
  tools/test_exact_burst_generation_sealed_lease_identity_gate.py \
  tools/test_exact_burst_generation_sealed_lease_identity_verify.py \
  tools/test_profile_exact_burst_generation_sealed_lease_identity.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_generation_sealed_block_table.py \
  tools/test_hybrid_state_sequence.py \
  tools/test_chunked_prefill.py \
  tools/test_speculative_kv_transaction.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit and push Task 6**

```bash
git add -- \
  tools/run_exact_burst_generation_sealed_lease_identity_remote.py \
  tools/test_run_exact_burst_generation_sealed_lease_identity_remote.py
git -c core.hooksPath=/dev/null commit \
  -m "test(runtime): run sealed-identity gate remotely" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 7: Execute the GPU Gate and Reconcile Evidence

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-24-exact-burst-generation-sealed-lease-identity-audit.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Verify source readiness**

Require exact task-path cleanliness and:

```bash
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
```

Expected: equal SHAs.

- [ ] **Step 2: Run final local focused and adjacent suites**

Run the Task 6 suite plus the existing one-phase journal, continuation,
ragged-coalescing, and split-phase gates.

- [ ] **Step 3: Launch one fresh immutable GPU run**

Use a never-attempted tag and the exact pushed source SHA. Admit only one
strict-clean A100. The controller must launch immediately after admission and
must remain resumable without duplicate execution.

- [ ] **Step 4: Require terminal evidence**

Require:

```text
worker exit code 0
60/60 performance rows
24/24 correctness rows
remote verifier PASS
local verifier PASS
matching source/workload/artifact digests
matching producer/verifier classification
```

- [ ] **Step 5: Write benefit/cost audit**

Record CPU and GPU lifecycle benefit, TPOT median/P95/P99, TTFT, E2E,
throughput, allocated/reserved memory, forward/replay/D2H invariance,
correctness, cold/hot/fallback counts, implementation cost, and genericity
boundary.

- [ ] **Step 6: Run verification-before-completion**

Run:

```bash
python3 -m pytest \
  tools/test_run_exact_burst_generation_sealed_lease_identity_remote.py \
  tools/test_exact_burst_generation_sealed_lease_identity_gate.py \
  tools/test_exact_burst_generation_sealed_lease_identity_verify.py \
  tools/test_profile_exact_burst_generation_sealed_lease_identity.py \
  tools/test_scheduler_prepared_postprocess.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_generation_sealed_block_table.py \
  tools/test_hybrid_state_sequence.py \
  tools/test_chunked_prefill.py \
  tools/test_speculative_kv_transaction.py \
  -q
git diff --check -- \
  docs/superpowers/audits/2026-08-24-exact-burst-generation-sealed-lease-identity-audit.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
```

- [ ] **Step 7: Commit and push final reconciliation**

```bash
git add -- \
  docs/superpowers/audits/2026-08-24-exact-burst-generation-sealed-lease-identity-audit.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(bench): record sealed-identity result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Do not add generated benchmark artifacts unless repository policy is changed
explicitly.
