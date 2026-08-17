# Hybrid Request-State Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a model-independent, request-indexed fixed-state allocator and tensor pool, preserve its lease through `Sequence` serialization, and integrate its lifecycle transactionally with the existing KV scheduler.

**Architecture:** Full-attention KV remains owned by `BlockManager`. A new metadata-only `HybridStateSlotAllocator` gives each live request a generation-tagged slot lease, while `HybridStateTensorPool` owns stable per-component tensors indexed by those slots. `Scheduler` receives the allocator optionally and centralizes paired KV/state allocation and release so the existing path is unchanged when the allocator is absent.

**Tech Stack:** Python 3, dataclasses, PyTorch CPU tensors, SHA-256, existing dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify or reinterpret the Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not add Qwen3.5 model loading, Gated DeltaNet math, CUDA kernels, compression, or performance claims.
- Do not change default Qwen3 behavior when no hybrid allocator is installed.
- Keep recurrent/convolution state request-indexed; never encode it as KV blocks.
- Prefix-cache reuse with hybrid state must fail closed until aligned state snapshots exist.
- Preserve old `Sequence` pickle tuple compatibility.
- Use inline execution; do not dispatch subagents.
- Do not stage unrelated experiment artifacts or commit unless explicitly requested.

---

### Task 1: Hybrid State Layout, Lease Allocator, and Tensor Pool

**Files:**
- Create: `tinyvllm/engine/hybrid_state.py`
- Create: `tools/test_hybrid_state.py`

**Interfaces:**
- Produces: `HybridStateComponentSpec`, `HybridStateLayout`, `HybridStateLease`, `HybridStateSlotAllocator`, and `HybridStateTensorPool`.
- Consumes: only Python standard library and `torch`.

- [ ] **Step 1: Write failing layout and allocator tests**

Add script tests that construct the frozen 18-linear-layer Qwen3.5 fixture:

```python
components = tuple(
    component
    for layer_index in range(18)
    for component in (
        HybridStateComponentSpec(
            layer_index,
            "linear_convolution",
            (6144, 4),
            torch.bfloat16,
        ),
        HybridStateComponentSpec(
            layer_index,
            "linear_recurrent",
            (16, 128, 128),
            torch.bfloat16,
        ),
    )
)
layout = HybridStateLayout(components)
assert layout.bytes_per_slot == 10_321_920
```

Cover fingerprint stability, FP32 byte doubling, invalid specs, deterministic
allocation, exhaustion, duplicate request allocation, wrong-owner release,
stale generation, double release, and generation increment after slot reuse.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state.py
```

Expected: import failure because `tinyvllm.engine.hybrid_state` does not exist.

- [ ] **Step 3: Implement immutable layout and generation-tagged allocator**

Implement:

```python
HybridStateRole = Literal["linear_convolution", "linear_recurrent"]

@dataclass(frozen=True)
class HybridStateComponentSpec:
    layer_index: int
    role: HybridStateRole
    shape: tuple[int, ...]
    dtype: torch.dtype

@dataclass(frozen=True)
class HybridStateLease:
    slot_id: int
    generation: int
    request_id: int
```

`HybridStateLayout.__post_init__` validates and canonicalizes components by
`(layer_index, role)`. Its SHA-256 fingerprint serializes dtype names and
integer shapes. `HybridStateSlotAllocator` uses a FIFO `deque`, per-slot
generation counters, an owner map, and a request-to-lease map.

- [ ] **Step 4: Add failing tensor-pool tests**

Test CPU-backed pool allocation, stable component addresses, zero activation,
mutation then zero release, conflict rejection, stale lease rejection, batch
slot validation, logical bytes, and physical storage bytes.

- [ ] **Step 5: Implement `HybridStateTensorPool`**

Preallocate:

```python
self._tensors[key] = torch.zeros(
    (capacity, *spec.shape),
    dtype=spec.dtype,
    device=device,
)
```

Track bound `(request_id, generation)` per slot. `activate()` and `release()`
clear all component rows. `slot_ids()` validates all leases and returns a
`torch.int32` tensor on the pool device.

- [ ] **Step 6: Run focused tests and confirm GREEN**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state.py
```

Expected: all hybrid-state tests pass.

### Task 2: Sequence Lease Serialization

**Files:**
- Modify: `tinyvllm/engine/sequence.py`
- Create: `tools/test_hybrid_state_sequence.py`

**Interfaces:**
- Consumes: lease scalar fields from Task 1.
- Produces: `Sequence.hybrid_state_slot_id` and `Sequence.hybrid_state_generation` across process serialization.

- [ ] **Step 1: Write failing round-trip and legacy tests**

Test both a prompt-only sequence and a decoded sequence through
`pickle.dumps`/`pickle.loads`. Also call `__setstate__` with an old 11-field
tuple and assert:

```python
assert restored.hybrid_state_slot_id == -1
assert restored.hybrid_state_generation == 0
```

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
python3 tools/test_hybrid_state_sequence.py
```

Expected: missing hybrid-state attributes after round trip.

- [ ] **Step 3: Extend serialization compatibly**

Initialize the two scalar fields in `Sequence.__init__`. Append them before the
payload item in new state tuples, and parse tuple lengths explicitly so old
11-field and older legacy formats retain their previous meaning.

- [ ] **Step 4: Run serialization and existing chunked-prefill tests**

Run:

```bash
python3 tools/test_hybrid_state_sequence.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_chunked_prefill.py
```

Expected: both pass.

### Task 3: Transactional Scheduler Lifecycle

**Files:**
- Modify: `tinyvllm/engine/scheduler.py`
- Create: `tools/test_hybrid_state_scheduler.py`

**Interfaces:**
- Consumes: `HybridStateSlotAllocator` and scalar lease metadata.
- Produces: optional paired KV/state admission, allocation, preemption, completion, and observation accounting.

- [ ] **Step 1: Write failing disabled-path and lifecycle tests**

Build small scheduler configs and cover:

```python
scheduler = Scheduler(config, hybrid_state_allocator=allocator)
```

Required assertions:

- allocator-disabled scheduler behavior and observations remain unchanged;
- state-slot exhaustion leaves waiting requests unallocated and consumes no KV
  blocks;
- a first prefill gets one lease;
- chunked continuation and decode retain that lease;
- completion releases both resources;
- preemption releases both and readmission increments generation;
- a request with reusable prefix KV fails closed;
- injected KV allocation failure rolls back the new lease.

- [ ] **Step 2: Run and confirm RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_scheduler.py
```

Expected: `Scheduler` rejects the new keyword argument.

- [ ] **Step 3: Add shared lifecycle helpers**

Add:

```python
def _allocate_request_storage(
    self,
    seq: Sequence,
    *,
    publish_hashes: bool,
    max_cached_tokens: int,
) -> None:
    ...

def _release_request_storage(self, seq: Sequence) -> None:
    ...
```

The allocation helper acquires a lease before KV allocation and rolls it back
if KV allocation raises. It rejects hybrid prefix reuse before allocation.
The release helper deallocates KV, validates/releases the lease, and resets the
sequence sentinel fields.

- [ ] **Step 4: Route every scheduler lifecycle branch through helpers**

Replace direct `block_manager.allocate()` calls in legacy prefill, chunked
prefill, and mixed admission. Replace direct `block_manager.deallocate()` calls
in preemption, ordinary completion, chunked completion, and mixed completion.
State-slot capacity must participate in admission before either resource is
consumed.

- [ ] **Step 5: Extend observation snapshots**

Only when an allocator exists, add:

```python
"hybrid_state": self.hybrid_state_allocator.observation_snapshot()
```

Do not change disabled-path observation keys.

- [ ] **Step 6: Run focused scheduler tests and broad scheduler regression**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_scheduler.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_chunked_prefill.py
```

Expected: all pass.

### Task 4: Completion Audit and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Verify: all files from Tasks 1-3

**Interfaces:**
- Consumes: test output and current git state.
- Produces: exact evidence, scope boundary, and next native-model gate.

- [ ] **Step 1: Run syntax and focused regression suites**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/hybrid_state.py \
  tinyvllm/engine/sequence.py \
  tinyvllm/engine/scheduler.py \
  tools/test_hybrid_state.py \
  tools/test_hybrid_state_sequence.py \
  tools/test_hybrid_state_scheduler.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state.py
python3 tools/test_hybrid_state_sequence.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_scheduler.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_chunked_prefill.py
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 2: Audit prompt-to-artifact coverage**

Map every spec requirement to code and a passing test. Treat missing prefix
fail-closed, stale-generation, rollback, or lifecycle coverage as incomplete.

- [ ] **Step 3: Update handoff without performance claims**

Record:

- files added/modified;
- exact commands and pass results;
- `9.84375 MiB` as reference fixture accounting, not measured engine savings;
- no native Qwen3.5 support or speedup;
- next step is model/layout adapter plus reference-equivalent kernel gate;
- Exact CUDA Graph remains independently blocked on GPU0 admission.

- [ ] **Step 4: Inspect final status**

Run:

```bash
git status --short --branch
git diff --stat
```

Expected: only intentional tracked edits plus preserved pre-existing untracked
experiment directories.
