# Autoregressive Draft Proposal-KV Offload Runtime Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user forbids subagents and git commits in this workspace.

**Goal:** Connect the independent Qwen3 draft model to the generic Proposal-KV residency manager through an explicit default-off configuration path.

**Architecture:** Add learned-drafter-specific offload enable and backing-capacity fields, build either the existing direct allocator or the generic residency manager through one Qwen3 storage-aware builder, and make ModelRunner registration consume the allocator without changing executor or verifier semantics. Existing generic Proposal-KV lifecycle and lease APIs remain the only residency state machine.

**Tech Stack:** Python 3.12, PyTorch tensors, dataclass configuration, pytest, TinyLLMForge speculative runtime.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, push, stash, reset, or clean.
- Do not run GPU, remote, NCCL, loaded-checkpoint, or performance workloads.
- `autoregressive_draft_max_proposal_tokens` remains in `1..4`.
- Default-off mode allocates no CPU backing and creates no transfer backend.
- Do not change verifier selection, fallback indexing, accepted-prefix semantics, target-KV transactions, side-state, Scheduler, n-gram, SAM, or native MTP behavior.
- Preserve every unrelated modified or untracked file.

---

### Task 1: Learned-Drafter Offload Configuration Contract

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`

**Interfaces:**
- Produces:
  - `Config.autoregressive_draft_proposal_kv_offload_enabled: bool`
  - `Config.autoregressive_draft_logical_entry_capacity: int`
  - `Config.autoregressive_draft_cpu_backing_capacity: int`

- [x] **Step 1: Add failing default and valid-enabled tests**

Add assertions to the existing default-config test:

```python
assert (
    config.autoregressive_draft_proposal_kv_offload_enabled
    is False
)
assert config.autoregressive_draft_logical_entry_capacity == 0
assert config.autoregressive_draft_cpu_backing_capacity == 0
```

Add:

```python
def test_autoregressive_draft_proposal_kv_offload_config_accepts_valid_shape():
    config = Config(
        autoregressive_draft_enabled=True,
        autoregressive_draft_model="/tmp/draft",
        autoregressive_draft_gpu_slot_capacity=8,
        autoregressive_draft_proposal_kv_offload_enabled=True,
        autoregressive_draft_logical_entry_capacity=16,
        autoregressive_draft_cpu_backing_capacity=16,
    )
    assert config.autoregressive_draft_logical_entry_capacity == 16
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:$PWD \
  /opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_autoregressive_draft_model_runner_integration.py \
  -k 'config and (default or offload)'
```

Expected: fail because the new fields are absent.

- [x] **Step 3: Add failing invalid-combination cases**

Extend the existing invalid configuration parameterization with:

```python
{
    "autoregressive_draft_enabled": False,
    "autoregressive_draft_proposal_kv_offload_enabled": True,
    "autoregressive_draft_logical_entry_capacity": 16,
    "autoregressive_draft_gpu_slot_capacity": 8,
    "autoregressive_draft_cpu_backing_capacity": 16,
}
{
    "autoregressive_draft_enabled": True,
    "autoregressive_draft_model": "/tmp/draft",
    "autoregressive_draft_proposal_kv_offload_enabled": True,
    "autoregressive_draft_logical_entry_capacity": 16,
    "autoregressive_draft_gpu_slot_capacity": 8,
    "autoregressive_draft_cpu_backing_capacity": 15,
}
{
    "autoregressive_draft_enabled": True,
    "autoregressive_draft_model": "/tmp/draft",
    "autoregressive_draft_proposal_kv_offload_enabled": True,
    "autoregressive_draft_logical_entry_capacity": 8,
    "autoregressive_draft_gpu_slot_capacity": 8,
    "autoregressive_draft_cpu_backing_capacity": 8,
}
```

- [x] **Step 4: Implement the minimal configuration fields and validation**

Add dataclass defaults:

```python
autoregressive_draft_proposal_kv_offload_enabled: bool = False
autoregressive_draft_logical_entry_capacity: int = 0
autoregressive_draft_cpu_backing_capacity: int = 0
```

Validate the enable as a bool and capacities as nonnegative integers. When
enabled, require:

```python
self.autoregressive_draft_enabled
self.autoregressive_draft_logical_entry_capacity \
    == self.autoregressive_draft_cpu_backing_capacity
self.autoregressive_draft_logical_entry_capacity \
    > self.autoregressive_draft_gpu_slot_capacity > 0
```

- [x] **Step 5: Run the complete configuration file**

Run:

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:$PWD \
  /opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_autoregressive_draft_model_runner_integration.py \
  -k config
```

Expected: all selected tests pass.

---

### Task 2: Qwen3 Draft Storage-Aware Allocator Builder

**Files:**
- Modify: `tinyvllm/engine/qwen3_draft_proposal_kv.py`
- Modify: `tools/test_qwen3_draft_proposal_kv_storage.py`

**Interfaces:**
- Consumes:
  - `Qwen3DraftProposalKVStorage`
  - `Qwen3DraftPhysicalSlotStore`
  - `DirectProposalKVAllocator`
  - `ProposalKVResidencyManager`
- Produces:

```python
build_qwen3_draft_proposal_kv_allocator(
    model,
    *,
    offload_enabled,
    logical_entry_capacity,
    gpu_slot_capacity,
    cpu_backing_capacity,
    async_copy,
    batch_copy,
    dtype,
    device,
    _copy_backend=None,
)
```

- [x] **Step 1: Write failing direct-mode builder test**

```python
def test_builder_default_direct_mode_allocates_no_cpu_backing():
    allocator = build_qwen3_draft_proposal_kv_allocator(
        model,
        offload_enabled=False,
        logical_entry_capacity=8,
        gpu_slot_capacity=8,
        cpu_backing_capacity=8,
        async_copy=True,
        batch_copy=True,
        dtype=torch.float32,
        device="cpu",
    )
    assert isinstance(allocator, DirectProposalKVAllocator)
    assert allocator.physical_store.cpu_key_cache is None
    assert allocator.physical_store.cpu_value_cache is None
```

- [x] **Step 2: Run direct test and verify RED**

Run the exact test. Expected: import failure because the builder does not
exist.

- [x] **Step 3: Implement direct mode**

Validate booleans, require matching logical/GPU capacities, construct
`Qwen3DraftPhysicalSlotStore`, and return `DirectProposalKVAllocator`.
Do not instantiate a copy backend.

- [x] **Step 4: Run direct test and verify GREEN**

Run the exact test. Expected: pass.

- [x] **Step 5: Write failing residency-mode builder and copy test**

Use `SynchronousProposalKVCopyBackend` through `_copy_backend`, require a
`ProposalKVResidencyManager`, assert CPU backing exists, write distinct values
for every layer, perform a lease-driven D2H/eviction/H2D round trip, and assert
all layers are restored exactly.

- [x] **Step 6: Run residency test and verify RED**

Expected: fail because offload mode is not implemented.

- [x] **Step 7: Implement residency mode**

Construct `Qwen3DraftProposalKVStorage` with:

```python
logical_capacity=logical_entry_capacity
gpu_capacity=gpu_slot_capacity
allocate_cpu_backing=True
allocate_pinned_cpu=torch.device(device).type == "cuda"
```

Select:

```python
TorchProposalKVCopyBackend() if async_copy
else SynchronousProposalKVCopyBackend()
```

unless `_copy_backend` is supplied, then return
`ProposalKVResidencyManager(storage=..., copy_backend=..., batch_copy=...)`.

- [x] **Step 8: Run the complete storage test file**

Run:

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:$PWD \
  /opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_qwen3_draft_proposal_kv_storage.py
```

Expected: all tests pass.

---

### Task 3: ModelRunner Registration Wiring

**Files:**
- Modify: `tinyvllm/engine/autoregressive_draft_registration.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_autoregressive_draft_model_runner_integration.py`
- Modify: `tools/test_autoregressive_draft_proposal_kv_allocator_contract.py`

**Interfaces:**
- Replaces dependency pair:

```text
build_physical_store
build_proposal_kv_allocator
```

with one dependency:

```python
build_proposal_kv_allocator(
    model,
    *,
    offload_enabled,
    logical_entry_capacity,
    gpu_slot_capacity,
    cpu_backing_capacity,
    async_copy,
    batch_copy,
    dtype,
    device,
)
```

- [x] **Step 1: Write failing offload registration test**

Extend the fake dependency object to capture allocator-builder arguments.
Configure:

```text
offload_enabled=True
logical=16
gpu=8
cpu=16
async_copy=False
batch_copy=True
```

Return a fake allocator exposing `.storage`, then assert registration:

- passes the exact values;
- publishes that storage through
  `runner.autoregressive_draft_physical_store`;
- publishes the existing cache/backend/executor/descriptor exactly once.

- [x] **Step 2: Run the exact test and verify RED**

Expected: fail because ModelRunner still separately constructs a direct store.

- [x] **Step 3: Update registration dependencies**

Import `build_qwen3_draft_proposal_kv_allocator` and expose it as the dependency
builder. Remove production dependency construction of
`Qwen3DraftPhysicalSlotStore` and `DirectProposalKVAllocator`.

- [x] **Step 4: Update ModelRunner construction**

Resolve capacities:

```python
offload_enabled = bool(
    config.autoregressive_draft_proposal_kv_offload_enabled
)
logical_capacity = (
    config.autoregressive_draft_logical_entry_capacity
    if offload_enabled
    else config.autoregressive_draft_gpu_slot_capacity
)
cpu_capacity = (
    config.autoregressive_draft_cpu_backing_capacity
    if offload_enabled
    else config.autoregressive_draft_gpu_slot_capacity
)
```

Build the allocator, then resolve:

```python
physical_store = getattr(
    entry_allocator,
    "storage",
    getattr(entry_allocator, "physical_store", None),
)
```

Fail closed if neither exists.

- [x] **Step 5: Preserve failure atomicity**

Keep allocator/storage/cache/backend/executor objects candidate-local until
all-rank consensus succeeds. Do not publish any field before consensus.

- [x] **Step 6: Update source-contract assertions**

Require:

- `build_qwen3_draft_proposal_kv_allocator`;
- both `.storage` and `.physical_store` resolution;
- no direct-only allocator construction in production registration;
- no stale `ProposalKVCache.physical_store` access.

- [x] **Step 7: Run registration-focused tests**

Run:

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:$PWD \
  /opt/homebrew/bin/python3.12 -m pytest -q \
  tools/test_autoregressive_draft_proposal_kv_allocator_contract.py \
  tools/test_autoregressive_draft_model_runner_integration.py \
  tools/test_qwen3_draft_backend.py
```

Expected: all tests pass.

---

### Task 4: Regression Gate and Evidence Update

**Files:**
- Modify: `docs/superpowers/audits/2026-08-15-phase1-prompt-to-artifact-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Produces terminal classification:

```text
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_RUNTIME_WIRING=ESTABLISHED_LOCAL
AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_OFFLOAD_DEFAULT=DISABLED
REAL_AUTOREGRESSIVE_DRAFT_PROPOSAL_KV_MOVEMENT=NOT_ESTABLISHED
LEARNED_DRAFTER_LOADED_PARITY=NOT_ESTABLISHED
PERFORMANCE_IMPROVEMENT=NOT_ESTABLISHED
PHASE_1=NOT_ACHIEVED
PROMOTION=NOT_PROMOTABLE
```

- [x] **Step 1: Run learned-drafter focused matrix**

Run all learned-drafter allocator, storage, backend, executor, ModelRunner, and
TP contract tests already listed by the current handoff.

- [x] **Step 2: Run Proposal-KV Tasks 1-7 regression**

Run the existing nine-file dependency-light suite. Expected: 78 or more tests,
all passing.

- [x] **Step 3: Run generic speculative runtime regression**

Run:

```text
tools/test_engine_speculative_runtime.py
tools/test_speculative_runtime.py
tools/test_speculative_batch_runtime.py
tools/test_speculative_kv_transaction.py
tools/test_speculative_side_state.py
```

Expected: all tests pass.

- [x] **Step 4: Run static gates**

Run `py_compile` for every changed production and test file, scan the
default-off path for CPU backing/copy-backend construction, scan for stale
direct-only symbols, and run scoped `git diff --check`.

- [x] **Step 5: Update audit and handoff**

Record exact commands, counts, local behavior established, and explicit GPU/TP
and performance limitations. Do not claim real movement from synchronous CPU
tests.

- [x] **Step 6: Perform prompt-to-artifact completion audit**

Map every design requirement to source/test evidence. Treat missing GPU/TP,
loaded parity, real movement, and performance as not established. Keep
`PHASE_1=NOT_ACHIEVED`.

## 2026-08-15 Status Reconciliation

The checkboxes above were reconciled against the current source, the recorded
RED/GREEN history in `AGENT_HANDOFF_STATE.md`, and fresh isolated local
verification:

```text
configuration contract: 21 passed, 39 deselected
storage-aware allocator: 19 passed
registration-focused:    99 passed
learned-drafter matrix:  305 passed in 15.22s
focused py_compile:      PASS
TP4 gate CLI --help:     PASS
scoped diff check:       PASS
```

These results establish the local default-off runtime-wiring contract only.
They do not establish loaded-checkpoint parity, real CUDA H2D/D2H movement,
performance improvement, Phase 1 completion, or promotion.
