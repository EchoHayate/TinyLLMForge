# TP4 Dynamic Pool-Index Graph Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse one exact Qwen3.8 transactional decode CUDA Graph across valid lease rotations while preserving full ordered lease validation and all frozen TP4 qualification gates.

**Architecture:** Introduce a protocol-aware stable graph-program cache key that excludes only the concrete lease seal for `lease_pool_index_v1`. Before every capture or replay, build and validate a full ordered lease manifest; copy its physical slot IDs into graph-owned device storage, then use capture-safe tensor-indexed hybrid-state gather and commit operations.

**Tech Stack:** Python 3.12, PyTorch CUDA Graphs, torch.distributed/NCCL, pytest, Qwen3.8 hybrid-state runtime, JSON receipt and verifier pipeline.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`; the Desktop path is a symlink to this checkout.
- Do not use `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not create a worktree or dispatch subagents; execute inline.
- Push only to `origin/feat/kv-sparse-attention`.
- Use strict RED -> minimal implementation -> GREEN for every runtime change.
- Stage exact paths only; never use `git add -A`, `git reset`, `git clean`, or broad formatting.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit must contain exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Preserve unrelated tracked and untracked files.
- Keep `multi_sequence_cuda_graph_dynamic_pool_indices` default-disabled.
- Do not alter the ordinary `forward_v1` path, exact-prefill graphs, speculative verification graphs, Exact Greedy K8, KV offload, Quest, or graph-resident greedy tail.
- Preserve ordered `slot_id + generation + request_id` validation before graph launch.
- Preserve warmup/measured graph-cache reset; do not retain warmup graphs in measured evidence.
- Keep single-capture `2_000_000_000 ns`, total-capture-per-rank `5_000_000_000 ns`, replay coverage `0.80`, exact-output, memory, throughput, and latency gates unchanged.
- Shared-capacity evidence remains `DIAGNOSTIC_ONLY`.
- Remote task data, source, logs, cache, and temporary files must remain below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not execute `kinit` or `krenew`; the user refreshes credentials externally.
- Do not terminate, adopt, or clean foreign GPU processes. Cleanup may target only exact-tag owned processes.
- Do not mutate or reclassify r48 or r49.
- Do not copy large remote artifacts to the Mac.

---

## File and Responsibility Map

- `tinyvllm/config.py`
  - owns the default-false feature flag and validation.
- `tinyvllm/engine/flash_attn_split_policy.py`
  - owns full invocation identity and protocol-aware stable cache-key derivation.
- `tinyvllm/engine/exact_cuda_graph_cache.py`
  - owns cache admission, ready lookup, rejection, accounting, and new cross-lease counters by stable key.
- `tinyvllm/engine/exact_cuda_graph_lease_manifest.py`
  - new pure manifest dataclasses and canonical hashing/validation.
- `tinyvllm/engine/hybrid_state.py`
  - owns validation of current lease bindings and ordered slot extraction.
- `tinyvllm/engine/qwen35_layer_state.py`
  - owns capture-safe tensor-indexed gather and commit for one layer.
- `tinyvllm/engine/qwen35_state_transaction.py`
  - owns cross-layer tensor-indexed gather and commit.
- `tinyvllm/layers/qwen35_packed_layer_stack.py`
  - owns transactional layer execution from dynamically gathered state.
- `tinyvllm/models/qwen35_packed.py`
  - exposes the Qwen3.8 graph-manifest and pool-index execution hooks.
- `tinyvllm/engine/model_runner.py`
  - selects protocol v2, builds manifests, captures/replays with graph-owned slot tensors, and emits dispatch evidence.
- `tinyvllm/engine/exact_cuda_graph_capture_receipt.py`
  - records program-key, invocation, manifest, and validation phases.
- `tools/tp4_decode_replay_worker.py`
  - enables v2 for graph cases and retains the new dispatch evidence.
- `tools/assemble_tp4_decode_replay.py`
  - checks producer-side TP4 agreement and cross-lease mechanism coverage.
- `tools/verify_tp4_decode_replay.py`
  - independently reconstructs the new evidence and rejects disagreement/tampering.
- Existing focused test files
  - receive RED/GREEN coverage beside the owned component.
- `tools/test_exact_cuda_graph_lease_manifest.py`
  - new focused pure-Python manifest contract test.
- `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
  - receives immutable r50/r51 result reconciliation.
- `AGENT_HANDOFF_STATE.md`
  - receives the final resumable checkpoint.

---

### Task 1: Add the protocol flag and stable graph-program cache key

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/flash_attn_split_policy.py`
- Modify: `tinyvllm/engine/exact_cuda_graph_cache.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Produces: `Config.multi_sequence_cuda_graph_dynamic_pool_indices: bool`
- Produces: `FlashAttentionGraphIdentity.cache_key_sha256: str`
- Produces: `ExactCudaGraphEntry.cache_key_sha256: str`
- Preserves: `FlashAttentionGraphIdentity.sha256` as the full invocation hash

- [ ] **Step 1: Write failing identity and configuration tests**

Add tests that construct two identities with identical structural fields and
different lease seals:

```python
def test_pool_index_protocol_separates_invocation_and_program_identity():
    first = make_identity(
        execution_protocol="lease_pool_index_v1",
        lease_seal="1" * 64,
    )
    second = replace(first, lease_seal="2" * 64)
    assert first.sha256 != second.sha256
    assert first.cache_key_sha256 == second.cache_key_sha256


def test_legacy_protocol_cache_key_remains_full_identity():
    first = make_identity(
        execution_protocol="lease_transaction_v1",
        lease_seal="1" * 64,
    )
    second = replace(first, lease_seal="2" * 64)
    assert first.cache_key_sha256 == first.sha256
    assert second.cache_key_sha256 == second.sha256
    assert first.cache_key_sha256 != second.cache_key_sha256


def test_dynamic_pool_index_flag_defaults_false_and_requires_graphs():
    assert Config().multi_sequence_cuda_graph_dynamic_pool_indices is False
    with pytest.raises(ValueError, match="requires multi_sequence_cuda_graphs"):
        Config(
            multi_sequence_cuda_graphs=False,
            multi_sequence_cuda_graph_dynamic_pool_indices=True,
        )
```

Add cache tests proving that observation and ready lookup use
`cache_key_sha256` for v2 while v1 behavior remains unchanged.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  -k 'pool_index or cache_key or dynamic_pool_index_flag' -q
```

Expected: failures for the missing flag, property, and cache-key field.

- [ ] **Step 3: Implement the minimal configuration and identity changes**

Add the configuration field:

```python
multi_sequence_cuda_graph_dynamic_pool_indices: bool = False
```

Validate it as a strict boolean and reject it unless
`multi_sequence_cuda_graphs` is enabled.

Add canonical identity hashing:

```python
@property
def cache_key_sha256(self) -> str:
    if self.execution_protocol != "lease_pool_index_v1":
        return self.sha256
    payload = asdict(self)
    payload["lease_seal"] = ""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
```

Extend `ExactCudaGraphEntry`:

```python
cache_key_sha256: str
```

Change cache dictionaries and admission methods to use
`identity.cache_key_sha256`. Keep `entry.identity_sha256` equal to the
full capture invocation hash, and reject a committed entry unless:

```python
entry.cache_key_sha256 == entry.identity.cache_key_sha256
```

For v1 identities, this remains the same key as before.

- [ ] **Step 4: Run focused and adjacent tests**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Check and commit exact paths**

Run:

```bash
git diff --check -- \
  tinyvllm/config.py \
  tinyvllm/engine/flash_attn_split_policy.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/flash_attn_split_policy.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): separate graph program identity" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one commit containing only the five listed files.

---

### Task 2: Build and validate ordered lease manifests

**Files:**
- Create: `tinyvllm/engine/exact_cuda_graph_lease_manifest.py`
- Create: `tools/test_exact_cuda_graph_lease_manifest.py`
- Modify: `tinyvllm/engine/hybrid_state.py`
- Modify: `tinyvllm/models/qwen35_packed.py`
- Modify: `tools/test_hybrid_state.py`
- Modify: `tools/test_qwen35_prepared_model_step.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class ExactCudaGraphLeaseManifestRow:
    batch_index: int
    slot_id: int
    generation: int
    request_id: int


@dataclass(frozen=True)
class ExactCudaGraphLeaseManifest:
    rows: tuple[ExactCudaGraphLeaseManifestRow, ...]

    @property
    def sha256(self) -> str: ...

    @property
    def slot_ids(self) -> tuple[int, ...]: ...
```

- Produces:

```python
def build_exact_cuda_graph_lease_manifest(
    *,
    leases: tuple[HybridStateLease, ...],
    expected_request_ids: tuple[int, ...],
    validate_lease: Callable[[HybridStateLease], HybridStateLease],
) -> ExactCudaGraphLeaseManifest: ...
```

- Produces:

```python
Qwen35PackedForCausalLM.exact_cuda_graph_lease_manifest(
    leases: tuple[HybridStateLease, ...],
    expected_request_ids: tuple[int, ...],
) -> ExactCudaGraphLeaseManifest
```

- [ ] **Step 1: Write the pure manifest RED tests**

Cover canonical hashing and every rejection:

```python
def test_manifest_preserves_order_and_hashes_canonically():
    manifest = build_exact_cuda_graph_lease_manifest(
        leases=(
            HybridStateLease(4, 2, 101),
            HybridStateLease(1, 7, 202),
        ),
        expected_request_ids=(101, 202),
        validate_lease=lambda lease: lease,
    )
    assert manifest.slot_ids == (4, 1)
    assert [row.batch_index for row in manifest.rows] == [0, 1]
    assert len(manifest.sha256) == 64


@pytest.mark.parametrize(
    "leases,request_ids,message",
    [
        ((), (), "non-empty"),
        (
            (HybridStateLease(0, 1, 10),),
            (11,),
            "request order",
        ),
        (
            (
                HybridStateLease(0, 1, 10),
                HybridStateLease(0, 1, 11),
            ),
            (10, 11),
            "distinct slots",
        ),
    ],
)
def test_manifest_rejects_invalid_ownership_shape(
    leases,
    request_ids,
    message,
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        build_exact_cuda_graph_lease_manifest(
            leases=leases,
            expected_request_ids=request_ids,
            validate_lease=lambda lease: lease,
        )
```

Add a test whose validator raises for a stale generation and assert the
exception is propagated before a manifest is returned.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_exact_cuda_graph_lease_manifest.py -q
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement the immutable manifest module**

Use canonical JSON over ordered rows:

```python
def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
```

The builder must:

```python
if not leases:
    raise ValueError("exact CUDA Graph lease manifest must be non-empty")
if len(leases) != len(expected_request_ids):
    raise ValueError("lease and request row counts must match")

rows = []
seen_slots = set()
for batch_index, (lease, request_id) in enumerate(
    zip(leases, expected_request_ids)
):
    validated = validate_lease(lease)
    if validated != lease:
        raise RuntimeError("lease validator changed identity")
    if lease.request_id != request_id:
        raise RuntimeError("lease manifest request order mismatch")
    if lease.slot_id in seen_slots:
        raise RuntimeError("lease manifest requires distinct slots")
    seen_slots.add(lease.slot_id)
    rows.append(ExactCudaGraphLeaseManifestRow(
        batch_index=batch_index,
        slot_id=lease.slot_id,
        generation=lease.generation,
        request_id=lease.request_id,
    ))
return ExactCudaGraphLeaseManifest(tuple(rows))
```

- [ ] **Step 4: Wire pool and model validation**

Add to `HybridStateTensorPool`:

```python
def validate_leases(
    self,
    leases: tuple[HybridStateLease, ...],
) -> tuple[HybridStateLease, ...]:
    for lease in leases:
        self.validate(lease)
    return leases
```

Add to `Qwen35PackedForCausalLM`:

```python
def exact_cuda_graph_lease_manifest(
    self,
    leases,
    expected_request_ids,
):
    pool = self.layer_stack.state_transaction.pool
    return build_exact_cuda_graph_lease_manifest(
        leases=leases,
        expected_request_ids=expected_request_ids,
        validate_lease=pool.validate,
    )
```

- [ ] **Step 5: Run focused and adjacent tests**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_exact_cuda_graph_lease_manifest.py \
  tools/test_hybrid_state.py \
  tools/test_qwen35_prepared_model_step.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Check and commit exact paths**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/exact_cuda_graph_lease_manifest.py \
  tinyvllm/engine/hybrid_state.py \
  tinyvllm/models/qwen35_packed.py \
  tools/test_exact_cuda_graph_lease_manifest.py \
  tools/test_hybrid_state.py \
  tools/test_qwen35_prepared_model_step.py
git add -- \
  tinyvllm/engine/exact_cuda_graph_lease_manifest.py \
  tinyvllm/engine/hybrid_state.py \
  tinyvllm/models/qwen35_packed.py \
  tools/test_exact_cuda_graph_lease_manifest.py \
  tools/test_hybrid_state.py \
  tools/test_qwen35_prepared_model_step.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): validate dynamic lease manifests" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 3: Add capture-safe tensor-indexed hybrid-state transactions

**Files:**
- Modify: `tinyvllm/engine/qwen35_layer_state.py`
- Modify: `tinyvllm/engine/qwen35_state_transaction.py`
- Modify: `tinyvllm/layers/qwen35_packed_layer_stack.py`
- Modify: `tinyvllm/models/qwen35_packed.py`
- Modify: `tools/test_qwen35_layer_state_adapter.py`
- Modify: `tools/test_qwen35_cross_layer_state_transaction.py`
- Modify: `tools/test_qwen35_prepared_model_step.py`

**Interfaces:**
- Produces:

```python
Qwen35LayerStateAdapter.gather_batch_by_slot_tensor(
    slot_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]

Qwen35LayerStateAdapter.commit_batch_by_slot_tensor(
    slot_ids: torch.Tensor,
    convolution_states: torch.Tensor,
    recurrent_states: torch.Tensor,
) -> None

Qwen35CrossLayerStateTransaction.gather_by_slot_tensor(
    slot_ids: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]

Qwen35CrossLayerStateTransaction.commit_by_slot_tensor(
    slot_ids: torch.Tensor,
    candidates: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None
```

- Produces:

```python
Qwen35PackedForCausalLM.run_exact_cuda_graph_step_by_pool_index(
    state_slot_ids: torch.Tensor,
    token_counts: tuple[int, ...],
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor | None
```

- [ ] **Step 1: Write RED tests for dynamic gather and commit**

Use CPU tensors to prove that changing only the contents of one stable slot
tensor redirects access:

```python
def test_dynamic_slot_tensor_redirects_gather_and_commit():
    pool, adapter = make_pool_and_adapter(capacity=4)
    first = torch.tensor([0, 2], dtype=torch.int64)
    second = torch.tensor([1, 3], dtype=torch.int64)
    assert first.data_ptr() != second.data_ptr()

    gathered_first = adapter.gather_batch_by_slot_tensor(first)
    gathered_second = adapter.gather_batch_by_slot_tensor(second)
    assert not torch.equal(gathered_first[0], gathered_second[0])

    original_unselected = adapter.convolution[[0, 2]].clone()
    adapter.commit_batch_by_slot_tensor(
        second,
        torch.full_like(gathered_second[0], 17),
        torch.full_like(gathered_second[1], 23),
    )
    assert torch.equal(adapter.convolution[[0, 2]], original_unselected)
```

Also test rank, dtype, device, bounds, duplicate index, candidate shape, and
candidate dtype rejection.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_qwen35_layer_state_adapter.py \
  tools/test_qwen35_cross_layer_state_transaction.py \
  tools/test_qwen35_prepared_model_step.py \
  -k 'slot_tensor or pool_index' -q
```

Expected: missing-method failures.

- [ ] **Step 3: Implement tensor-index validation and layer access**

Add a shared private validator in `Qwen35LayerStateAdapter`:

```python
def _validate_slot_tensor(self, slot_ids: torch.Tensor) -> None:
    if not isinstance(slot_ids, torch.Tensor):
        raise ValueError("slot_ids must be a tensor")
    if slot_ids.ndim != 1 or slot_ids.numel() == 0:
        raise ValueError("slot_ids must be a non-empty rank-one tensor")
    if slot_ids.dtype != torch.int64:
        raise ValueError("slot_ids must use torch.int64")
    if slot_ids.device != self.convolution.device:
        raise ValueError("slot_ids must share the state-pool device")
```

Do not call `.item()` or convert device values to a Python list in these
graph-specific methods:

```python
def gather_batch_by_slot_tensor(self, slot_ids):
    self._validate_slot_tensor(slot_ids)
    return (
        torch.index_select(self.convolution, 0, slot_ids),
        torch.index_select(self.recurrent, 0, slot_ids),
    )

def commit_batch_by_slot_tensor(
    self,
    slot_ids,
    convolution_states,
    recurrent_states,
):
    self._validate_slot_tensor(slot_ids)
    self._validate_batch_candidate(
        convolution_states,
        batch_size=slot_ids.shape[0],
        reference=self.convolution[0],
        name="convolution_states",
    )
    self._validate_batch_candidate(
        recurrent_states,
        batch_size=slot_ids.shape[0],
        reference=self.recurrent[0],
        name="recurrent_states",
    )
    self.convolution.index_copy_(0, slot_ids, convolution_states)
    self.recurrent.index_copy_(0, slot_ids, recurrent_states)
```

Host-side manifest validation, not these capture-time methods, proves bounds
and uniqueness before graph launch.

- [ ] **Step 4: Implement cross-layer and model graph APIs**

Add transaction methods that call every adapter with the same slot tensor and
validate candidates without extracting slot values to the host.

Add `prepare_transactional_by_pool_index` to the packed layer stack. It must
reuse the existing layer loop and `_run_linear_layer`, but obtain the initial
state from `gather_by_slot_tensor` and return the same
`Qwen35PreparedLayerStack`.

Add the model method:

```python
def run_exact_cuda_graph_step_by_pool_index(
    self,
    state_slot_ids,
    token_counts,
    input_ids,
    position_ids,
):
    prepared = self.prepare_step_by_pool_index(
        state_slot_ids,
        token_counts,
        input_ids,
        position_ids,
    )
    self.layer_stack.state_transaction.commit_by_slot_tensor(
        state_slot_ids,
        prepared.final_candidates,
    )
    prepared.state = "committed"
    return prepared.logits
```

Keep `run_step`, `prepare_step`, and lease-based commit unchanged.

- [ ] **Step 5: Run focused and adjacent state tests**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_qwen35_layer_state_adapter.py \
  tools/test_qwen35_cross_layer_state_transaction.py \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_hybrid_state.py \
  tools/test_hybrid_state_runtime_bridge.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Check and commit exact paths**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/qwen35_layer_state.py \
  tinyvllm/engine/qwen35_state_transaction.py \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/models/qwen35_packed.py \
  tools/test_qwen35_layer_state_adapter.py \
  tools/test_qwen35_cross_layer_state_transaction.py \
  tools/test_qwen35_prepared_model_step.py
git add -- \
  tinyvllm/engine/qwen35_layer_state.py \
  tinyvllm/engine/qwen35_state_transaction.py \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/models/qwen35_packed.py \
  tools/test_qwen35_layer_state_adapter.py \
  tools/test_qwen35_cross_layer_state_transaction.py \
  tools/test_qwen35_prepared_model_step.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(qwen38): index graph state dynamically" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 4: Integrate the pool-index protocol into ModelRunner

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/exact_cuda_graph_cache.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `identity.cache_key_sha256`
- Consumes: `model.exact_cuda_graph_lease_manifest(...)`
- Consumes: `model.run_exact_cuda_graph_step_by_pool_index(...)`
- Produces: graph entry tensor `state_slot_ids`
- Produces dispatch fields:

```text
graph_program_key_sha256
graph_invocation_identity_sha256
lease_manifest_sha256
cross_lease_replay
```

- [ ] **Step 1: Write RED protocol-selection and reuse tests**

Add a runner test with two valid lease cohorts:

```python
capture_leases = (
    HybridStateLease(0, 1, 100),
    HybridStateLease(1, 1, 101),
)
replay_leases = (
    HybridStateLease(3, 4, 200),
    HybridStateLease(2, 6, 201),
)
```

Assert:

```python
assert capture_identity.sha256 != replay_identity.sha256
assert capture_identity.cache_key_sha256 == replay_identity.cache_key_sha256
assert graph.capture_count == 1
assert graph.replay_count == 1
assert entry.tensors["state_slot_ids"].tolist() == [3, 2]
assert event["cross_lease_replay"] is True
```

Add negative tests proving a stale generation, wrong request, duplicate slot,
or manifest order mismatch raises before `graph.replay()` and before
`state_slot_ids.copy_()`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  -k 'pool_index or cross_lease or lease_manifest' -q
```

Expected: failures because v2 selection, manifest admission, static slot
storage, and evidence fields are absent.

- [ ] **Step 3: Select v2 only under the default-false flag**

Extend `_exact_graph_execution_identity()` so the complete hook set includes:

```python
"exact_cuda_graph_lease_manifest"
"run_exact_cuda_graph_step_by_pool_index"
```

Return:

```python
protocol = (
    "lease_pool_index_v1"
    if self.config.multi_sequence_cuda_graph_dynamic_pool_indices
    else "lease_transaction_v1"
)
```

Always compute the full lease seal. The full invocation identity therefore
continues to change across ownership rotations.

- [ ] **Step 4: Store request-row identity and build manifests**

In `_run_model_step`, preserve request order:

```python
self._last_hybrid_state_request_ids = tuple(
    int(seq.seq_id) for seq in seqs
)
```

Add:

```python
def _exact_graph_lease_manifest(self):
    return self.model.exact_cuda_graph_lease_manifest(
        tuple(self._last_hybrid_state_leases),
        tuple(self._last_hybrid_state_request_ids),
    )
```

Call it before capture and before replay for v2. Do not catch ownership
validation errors as ordinary identity misses.

- [ ] **Step 5: Capture and replay through stable slot storage**

During v2 capture allocate:

```python
tensors["state_slot_ids"] = torch.empty(
    batch_size,
    dtype=torch.int64,
    device=device,
)
tensors["state_slot_ids"].copy_(
    torch.tensor(
        manifest.slot_ids,
        dtype=torch.int64,
        device=device,
    )
)
```

Inside `torch.cuda.graph(...)`, call:

```python
tensors["outputs"] = (
    self.model.run_exact_cuda_graph_step_by_pool_index(
        tensors["state_slot_ids"],
        token_counts,
        tensors["input_ids"],
        tensors["positions"],
    )
)
```

At replay, validate the current manifest before any dynamic copy, then copy
the new slot IDs and launch the existing graph.

Set `cross_lease_replay` when:

```python
entry.identity_sha256 != current_identity.sha256
```

The program-key comparison must still match.

- [ ] **Step 6: Keep capture rollback and phase reset intact**

Capture snapshots and restores the capture cohort through the existing
lease-based snapshot/restore hooks. `reset_exact_cuda_graph_cache()` must
continue releasing every ready graph and clearing `_exact_cuda_graph_pool`.

Add assertions that phase reset clears v2 entries, invocation tracking, and
cross-lease counters.

- [ ] **Step 7: Run focused and adjacent tests**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Check and commit exact paths**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py
git add -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): replay graphs across lease rotations" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 5: Extend receipts, worker output, assembler, and verifier

**Files:**
- Modify: `tinyvllm/engine/exact_cuda_graph_capture_receipt.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/tp4_decode_replay_worker.py`
- Modify: `tools/assemble_tp4_decode_replay.py`
- Modify: `tools/verify_tp4_decode_replay.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_tp4_decode_replay_worker.py`
- Modify: `tools/test_assemble_tp4_decode_replay.py`
- Modify: `tools/test_verify_tp4_decode_replay.py`

**Interfaces:**
- Produces receipt schema version 2 with:

```text
execution_protocol
program_key_sha256
invocation_identity_sha256
lease_manifest_sha256
ordered_slot_ids
cross_lease_replay
```

- Produces final mechanism fields:

```text
unique_program_key_count
unique_invocation_identity_count
cross_lease_replay_count
manifest_validation_count
```

- [ ] **Step 1: Write RED receipt and evidence-chain tests**

Add tests proving:

- capture and replay receipts reject malformed 64-character digests;
- receipt phases remain ordered;
- a graph dispatch may change invocation identity while retaining one
  program key;
- producer and independent verifier require all four ranks to agree on
  program key and manifest digest for each dispatch;
- mutating one rank's manifest digest makes both assembler and verifier fail;
- `cross_lease_replay_count == 0` yields
  `NO_GO_MECHANISM_NOT_EXERCISED`;
- capture cost is grouped per rank/program capture and is not multiplied by
  four replicated TP-wide MAX rows.

- [ ] **Step 2: Run evidence tests and verify RED**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py \
  -k 'program_key or manifest or cross_lease or capture_cost' -q
```

Expected: failures for missing schema fields and mechanism checks.

- [ ] **Step 3: Extend receipt constructors and payloads**

Change capture and replay receipt constructors to accept the four identity
fields explicitly. Add a replay phase between entry and input copy:

```text
entered_replay
lease_manifest_validated
static_inputs_copied
context_set
graph_replay_returned
logits_compute_returned
context_reset_completed
```

Write receipt schema version 2 atomically. Do not rewrite old artifacts.

- [ ] **Step 4: Extend dispatch and worker rows**

Append the four new dispatch fields to `DISPATCH_EVENT_FIELDS` and populate
them for eager, capture, same-lease replay, and cross-lease replay.

Set the graph arm configuration:

```python
"multi_sequence_cuda_graph_dynamic_pool_indices": arm == "graph",
```

Keep eager disabled and preserve every existing gate value.

Worker agreement must compare:

```python
(
    "graph_program_key_sha256",
    "graph_invocation_identity_sha256",
    "lease_manifest_sha256",
    "cross_lease_replay",
)
```

alongside the existing fields.

- [ ] **Step 5: Extend producer and independent verification**

Reconstruct mechanism coverage from immutable dispatch rows:

```python
cross_lease_rows = [
    row for row in measured_graph_rows
    if row["dispatch"] == "graph"
    and row["cross_lease_replay"] is True
]
```

Require:

- exactly one program key per structural case/rank;
- at least two invocation identities for a claimed rotated cohort;
- at least one cross-lease replay per rank;
- identical program and manifest digests across ranks for each step;
- one capture-cost row per rank/program capture;
- per-rank total capture accounting, never a sum of replicated TP-wide MAX
  rows.

Both assembler and independent verifier must derive these values rather than
trust producer summary fields.

- [ ] **Step 6: Run complete evidence-chain tests**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Check and commit exact paths**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/exact_cuda_graph_capture_receipt.py \
  tinyvllm/engine/model_runner.py \
  tools/tp4_decode_replay_worker.py \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py
git add -- \
  tinyvllm/engine/exact_cuda_graph_capture_receipt.py \
  tinyvllm/engine/model_runner.py \
  tools/tp4_decode_replay_worker.py \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): verify cross-lease graph replay" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

### Task 6: Run the complete local verification and review gate

**Files:**
- Modify only if a test exposes a defect in files already owned by Tasks 1–5.
- Do not change frozen thresholds or unrelated fixtures to obtain green.

**Interfaces:**
- Consumes all runtime and evidence interfaces from Tasks 1–5.
- Produces a recorded local verification checkpoint at a pushed source SHA.

- [ ] **Step 1: Run syntax and whitespace checks**

Run:

```bash
python3 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/flash_attn_split_policy.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tinyvllm/engine/exact_cuda_graph_lease_manifest.py \
  tinyvllm/engine/hybrid_state.py \
  tinyvllm/engine/qwen35_layer_state.py \
  tinyvllm/engine/qwen35_state_transaction.py \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/models/qwen35_packed.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/exact_cuda_graph_capture_receipt.py \
  tools/tp4_decode_replay_worker.py \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 2: Run the focused Python 3.12/Torch suite**

Use the verified local Python 3.12/Torch interpreter:

```bash
PYTHONPATH=. /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_exact_cuda_graph_lease_manifest.py \
  tools/test_hybrid_state.py \
  tools/test_hybrid_state_runtime_bridge.py \
  tools/test_qwen35_layer_state_adapter.py \
  tools/test_qwen35_cross_layer_state_transaction.py \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py \
  tools/test_run_tp4_decode_replay.py -q
```

Expected: all selected tests pass. Record
`/opt/homebrew/opt/python@3.12/bin/python3.12` and the observed Torch version
in the verification checkpoint.

- [ ] **Step 3: Run the adjacent system-Python suite**

Run:

```bash
PYTHONPATH=. python3 -m pytest \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_exact_cuda_graph_phase_reset_wiring.py \
  tools/test_tp4_decode_replay_contract.py \
  tools/test_tp4_decode_replay_worker.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py \
  tools/test_run_tp4_decode_replay.py -q
```

Expected: all tests that do not require unavailable local Torch pass. Record
environment/setup failures separately; do not report them as passing or as
runtime regressions.

- [ ] **Step 4: Run code review**

Use `bits-code-guard` on the exact commit range beginning at
`77176d9839cf0aba009410cf3fb6ecf541774e87`. Resolve every P0–P2 finding with
a new RED/GREEN cycle. Do not alter unrelated files.

- [ ] **Step 5: Push and verify source identity**

Run:

```bash
git push -u origin feat/kv-sparse-attention
LOCAL_SHA=$(git rev-parse HEAD)
TRACKING_SHA=$(git rev-parse origin/feat/kv-sparse-attention)
REMOTE_SHA=$(git ls-remote origin refs/heads/feat/kv-sparse-attention | awk '{print $1}')
test "$LOCAL_SHA" = "$TRACKING_SHA"
test "$LOCAL_SHA" = "$REMOTE_SHA"
printf 'source_sha=%s\n' "$LOCAL_SHA"
```

Expected: all three SHAs are identical.

---

### Task 7: Run fresh r50 smoke, conditionally run r51 full gate, and audit

**Files:**
- Modify: `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify runner/controller/verifier files only through a new RED/GREEN repair if the fresh run exposes a genuine defect.

**Interfaces:**
- Smoke tag:
  `20260906-qwen38-tp4-decode-replay-r50-pool-index-q1-smoke`
- Conditional full tag:
  `20260906-qwen38-tp4-decode-replay-r51-pool-index-full`
- Remote root:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/tp4-collective-stable-decode-replay`

- [ ] **Step 1: Revalidate external prerequisites**

Run locally:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist
ssh -o BatchMode=yes sitian@10.232.195.203 \
  'hostname; df -h /data00/home/sitian; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'
```

Expected:

- the ticket lifetime satisfies the controller's frozen TTL gate;
- the mounted `/data00/home/sitian` root is writable and has sufficient space;
- four GPUs satisfy the existing formal admission policy.

If credentials are insufficient, stop without creating r50 and ask the user
to refresh them. Do not run `kinit` or `krenew`.

- [ ] **Step 2: Freeze source and create the r50 smoke**

Use the pushed source SHA from Task 6. The smoke runner must execute only the
frozen pair:

```text
Q1__r0__eager
Q1__r0__graph
```

in separate worker processes, using:

```text
tensor_parallel_size=4
model_revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
multi_sequence_cuda_graphs=true for graph only
multi_sequence_cuda_graph_dynamic_pool_indices=true for graph only
admission_mode=strict_clean
```

The graph case must force at least one valid request/generation/physical-slot
rotation after the measured graph capture while preserving batch size and
page-table width. Store all remote files below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/tp4-collective-stable-decode-replay/20260906-qwen38-tp4-decode-replay-r50-pool-index-q1-smoke
```

Do not reuse the tag if any path with that name already exists.

- [ ] **Step 3: Apply the smoke stop rule**

Verify from raw rows and receipts:

```text
exact output equality
four-rank program-key agreement
four-rank manifest agreement per step
at least one cross-lease replay per rank
one measured capture per structural key per rank
no stale or unselected state mutation
single capture <= 2,000,000,000 ns
per-rank total capture <= 5,000,000,000 ns
replay coverage >= 0.80
memory within frozen limits
clean rank/process-group teardown
no exact-tag process remains
```

If any item fails, classify r50 terminally, skip r51, preserve all evidence,
append the negative audit/handoff, commit, and push.

- [ ] **Step 4: Run r51 only if every r50 gate passes**

Launch the existing full controller with a fresh immutable tag:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
PYTHONPATH=.:tools \
/usr/bin/python3 tools/run_tp4_decode_replay.py monitor-and-run \
  --run-tag 20260906-qwen38-tp4-decode-replay-r51-pool-index-full \
  --admission-mode strict_clean \
  --ssh-target sitian@10.232.195.203 \
  --remote-python /data00/home/sitian/tllm/env/bin/python
```

Do not lower TTL, GPU, timeout, correctness, capture, memory, replay, latency,
or throughput gates. Do not launch a duplicate controller while the exact tag
has a live owner.

- [ ] **Step 5: Complete producer and dual verification**

For a complete r51 run, require:

- 30 complete cases and 15 complete eager/graph pairs;
- producer classification;
- immutable final bundle and manifest;
- remote independent verification;
- local frozen-source verification;
- byte/hash agreement between producer and both verifiers;
- `report.md`;
- complete process and cleanup receipts.

Do not call r51 complete based only on worker exit, manifest existence, or one
verifier.

- [ ] **Step 6: Append the prompt-to-artifact audit**

Record:

- exact source commit and source-tree SHA;
- model revision and GPU IDs;
- Kerberos and storage admission;
- r50 and, if authorized, r51 tags;
- capture/program/invocation/manifest counts;
- cross-lease replay coverage;
- correctness and state-isolation evidence;
- capture, memory, throughput, TTFT, TPOT, and P99 values;
- cleanup state;
- producer and verifier classifications;
- explicit claim boundary.

The conclusion must report both benefit and cost. A mechanism-only result is
not a performance win.

- [ ] **Step 7: Commit and push audit/handoff only**

Run:

```bash
git diff --check -- \
  docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md \
  AGENT_HANDOFF_STATE.md
git add -- \
  docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(tp4): record dynamic pool-index result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push -u origin feat/kv-sparse-attention
```

Verify local, tracking, and remote SHAs are equal.

---

## Completion Audit

Before declaring this plan complete, build a prompt-to-artifact table with one
row for every requirement below:

| Requirement | Required evidence |
|---|---|
| Default-off protocol | config test and source |
| Legacy behavior preserved | v1 and `forward_v1` regression tests |
| Stable structural cache key | identity and cache tests |
| Full lease identity retained | invocation hash and manifest rows |
| Replay-time ownership validation | stale generation/request/order/duplicate-slot tests |
| Dynamic physical slot selection | gather/commit and model-runner tests |
| Cross-lease graph reuse | one capture plus replay under a changed manifest |
| No unselected state mutation | before/after state hashes or exact tensors |
| Warmup/measured isolation | reset test and measured-only capture rows |
| TP4 agreement | four-rank program and manifest digests |
| Capture limits | per-rank capture rows and reconstructed maxima |
| Replay coverage | raw eligible/replay counts |
| Correctness | exact token equality |
| Benefit and cost | throughput, TTFT, TPOT, P99, memory, capture |
| Lifecycle | rank exits, process-group destruction, zero owned children |
| Immutable evidence | manifest plus remote/local verifier agreement |
| Git delivery | commit trailer, push, local/tracking/remote SHA equality |

Any missing or uncertain row means the implementation is not complete.
