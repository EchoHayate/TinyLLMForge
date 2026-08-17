# Qwen3.5 Layout and Runtime Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict Qwen3.5 config-to-TP-local hybrid-state layout adapter and a generation-safe Scheduler→LLMEngine→all-ModelRunner-ranks release-before-activate bridge that is fully testable on CPU.

**Architecture:** The adapter converts only explicit Qwen3.5 config fields into the existing immutable `HybridStateLayout`; it never infers missing architecture. Scheduler publishes exact released `HybridStateLease` events, LLMEngine drains and broadcasts them with the next run, and each rank-local `HybridStateRuntimeBridge` releases old generations before activating the active batch. Production ModelRunner pool installation remains dormant until a native Qwen3.5 model loader exists.

**Tech Stack:** Python 3, dataclasses, PyTorch CPU tensors, existing shared-memory ModelRunner RPC, dependency-light script tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use inline execution; do not dispatch subagents.
- Do not modify or reinterpret the Qwen3.5 schema-v2 canonical `NO_GO`.
- Do not implement Qwen3.5 model math, weight loading, Gated DeltaNet kernels, recurrent updates, or hybrid CUDA Graphs.
- Do not start a local or remote GPU model process.
- Preserve current Qwen3 behavior when no hybrid runtime bridge is installed.
- Release an old generation on every rank before activating a reused slot generation.
- Do not infer release from active-batch absence.
- Keep hybrid prefix reuse fail-closed.
- Preserve all untracked `experiments/` artifacts.
- Do not commit or stage unrelated files.

---

### Task 1: Strict Qwen3.5 Config-to-Layout Adapter

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_state.py`
- Create: `tools/test_qwen35_hybrid_state_layout.py`

**Interfaces:**
- Consumes: `HybridStateComponentSpec`, `HybridStateLayout`, an HF-style config object, TP size, state dtype, and speculative token width.
- Produces: `build_qwen35_hybrid_state_layout(hf_config, *, tensor_parallel_size, dtype, speculative_tokens=1) -> HybridStateLayout`.

- [ ] **Step 1: Write canonical and TP-local failing tests**

Create a dependency-light script with a canonical config fixture:

```python
CANONICAL_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
    for index in range(24)
)

config = SimpleNamespace(
    num_hidden_layers=24,
    layer_types=CANONICAL_LAYER_TYPES,
    linear_num_key_heads=16,
    linear_num_value_heads=16,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
)
```

Assert TP=1 BF16 has 36 components, the exact linear-layer indices,
convolution shape `(6144, 4)`, recurrent shape `(16, 128, 128)`, and
`10_321_920` bytes. Assert TP=2 has `(3072, 4)` and `(8, 128, 128)` and half
the bytes. Assert FP32 doubles BF16 bytes and `text_config` wrapping is
accepted.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_qwen35_hybrid_state_layout.py
```

Expected: import failure because `tinyvllm.engine.qwen35_hybrid_state` does not
exist.

- [ ] **Step 3: Implement the minimal strict adapter**

Implement exact integer validation, explicit layer schedule normalization, TP
divisibility checks, and the frozen formulas:

```python
conv_channels = (
    key_head_dim * key_heads * 2
    + value_head_dim * value_heads
)
conv_shape = (
    conv_channels // tensor_parallel_size,
    conv_kernel_dim - 1 + speculative_tokens,
)
recurrent_shape = (
    value_heads // tensor_parallel_size,
    value_head_dim,
    key_head_dim,
)
```

Only linear-attention layers produce components.

- [ ] **Step 4: Add malformed-config failing tests**

Cover missing fields, booleans, non-positive dimensions, schedule length
mismatch, unsupported layer type, no linear layer, unsupported dtype, invalid
TP size, non-divisible key/value heads or convolution channels, and
`speculative_tokens < 1`.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_qwen35_hybrid_state_layout.py
```

Expected: `qwen35 hybrid state layout tests passed`.

### Task 2: Scheduler Release Queue and Rank-Local Runtime Bridge

**Files:**
- Modify: `tinyvllm/engine/hybrid_state.py`
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tools/test_hybrid_state.py`
- Modify: `tools/test_hybrid_state_scheduler.py`

**Interfaces:**
- Consumes: exact `HybridStateLease` values released by scheduler lifecycle paths.
- Produces:
  - `HybridStateRuntimeBridge.prepare_batch(released_leases, active_leases) -> torch.Tensor`;
  - `HybridStateRuntimeBridge.release(released_leases) -> None`;
  - `Scheduler.drain_hybrid_state_release_events() -> tuple[HybridStateLease, ...]`;
  - `Scheduler.restore_hybrid_state_release_events(leases) -> None`.

- [ ] **Step 1: Write failing runtime ordering tests**

Add tests that bind generation 1, mutate its state, then call:

```python
slot_ids = bridge.prepare_batch((generation_1,), (generation_2,))
```

Assert release happens before activation, the reused row is zero, and returned
slot ids match active row order. Also cover idempotent same-generation
activation preserving mutation, independent rows, stale release, duplicate
release, wrong-owner release, and conflicting activation.

- [ ] **Step 2: Run runtime tests and confirm RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state.py
```

Expected: missing `HybridStateRuntimeBridge`.

- [ ] **Step 3: Implement the minimal runtime bridge**

The implementation must be phase-ordered:

```python
def prepare_batch(self, released_leases, active_leases):
    self.release(released_leases)
    for lease in active_leases:
        self.pool.activate(lease)
    return self.pool.slot_ids(active_leases)
```

Do not catch or weaken tensor-pool ownership exceptions.

- [ ] **Step 4: Write failing scheduler event tests**

Extend scheduler tests to assert:

- finish and preemption each publish the exact old lease once;
- allocator-disabled paths publish no event;
- drain preserves FIFO order and clears the queue;
- invalid release publishes nothing;
- restore prepends older failed-dispatch events before events published later.

- [ ] **Step 5: Run scheduler tests and confirm RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_scheduler.py
```

Expected: scheduler has no release-event drain method.

- [ ] **Step 6: Implement queue publication, drain, and restore**

Initialize:

```python
self._hybrid_state_release_events: deque[HybridStateLease] = deque()
```

Append only after successful paired release and metadata clearing. Drain into
a tuple and clear. Restore validates tuple entries and prepends them while
preserving original FIFO order.

- [ ] **Step 7: Run focused runtime and scheduler tests**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_scheduler.py
```

Expected: both scripts pass.

### Task 3: LLMEngine Dispatch and Dormant ModelRunner Plumbing

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Create: `tools/test_hybrid_state_runtime_bridge.py`

**Interfaces:**
- Consumes: scheduler release events and sequence lease metadata.
- Produces:
  - `LLMEngine.step()` forwarding events with `run`;
  - dispatch-failure restoration;
  - final exit drain through `release_hybrid_state`;
  - `ModelRunner.run(..., released_hybrid_state_leases=())`;
  - `ModelRunner.release_hybrid_state(leases)`;
  - `ModelRunner._prepare_hybrid_state_batch(seqs, released_leases)`.

- [ ] **Step 1: Write failing LLMEngine dispatch tests**

Load `LLMEngine.step` with lightweight stubs. The fake scheduler returns one
release event and the fake runner records:

```python
("run", seqs, is_prefill, do_sample, batch_kind, released_leases)
```

Assert the exact tuple is forwarded. In a second test, make `run` raise and
assert the scheduler receives `restore_hybrid_state_release_events()` with the
drained tuple.

- [ ] **Step 2: Write failing dormant ModelRunner bridge tests**

Load the relevant ModelRunner methods using the existing stub pattern. Assert:

- no leases and no bridge returns `None`;
- active or released leases without a bridge raise a fail-closed error;
- an installed CPU runtime bridge releases generation 1 before activating
  generation 2;
- `release_hybrid_state()` forwards exact events to the bridge.

- [ ] **Step 3: Run the focused test and confirm RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_runtime_bridge.py
```

Expected: old `run` call signature and missing ModelRunner bridge methods.

- [ ] **Step 4: Implement LLMEngine event forwarding and restoration**

Drain immediately before `run`. Wrap only dispatch in:

```python
try:
    token_ids = self.model_runner.call(..., released_leases)
except BaseException:
    self.scheduler.restore_hybrid_state_release_events(released_leases)
    raise
```

At exit, drain and broadcast `release_hybrid_state` before `exit`.

- [ ] **Step 5: Implement dormant ModelRunner bridge methods**

Initialize `self.hybrid_state_runtime_bridge = None`. Extract active leases
from sequences whose slot id is non-negative, rejecting partially enabled
metadata. Call the bridge before input preparation. Store returned slot ids in
`self._last_hybrid_state_slot_ids` for future native model dispatch. Keep the
argument optional to preserve direct callers.

- [ ] **Step 6: Run focused bridge tests and confirm GREEN**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_runtime_bridge.py
```

Expected: `hybrid state runtime bridge tests passed`.

### Task 4: Regression Verification and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: all implementation and validation evidence.
- Produces: an exact continuation record with claim boundaries and next gate.

- [ ] **Step 1: Run focused hybrid-state verification**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_qwen35_hybrid_state_layout.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_sequence.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_scheduler.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_hybrid_state_runtime_bridge.py
```

Expected: all five scripts pass.

- [ ] **Step 2: Run existing engine regression gates**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_model_runner_spec_verify.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python tools/test_multi_sequence_cuda_graph_gate.py
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/python3.12 tools/test_arrival_load_gate.py
```

Load `tools/test_chunked_prefill.py` and invoke all zero-argument `test_*`
functions, skipping only the known Python-3.12 Config-AST parser case if it
still fails for the pre-existing multiplication-default limitation.

- [ ] **Step 3: Run syntax and diff hygiene checks**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python -m py_compile \
  tinyvllm/engine/hybrid_state.py \
  tinyvllm/engine/qwen35_hybrid_state.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tinyvllm/engine/model_runner.py
git diff --check
```

Expected: exit code 0.

- [ ] **Step 4: Update handoff without performance claims**

Append exact files, interfaces, RED/GREEN evidence, regression results,
limitations, and next work. State explicitly:

- no Qwen3.5 model math or kernel exists;
- no GPU process ran;
- canonical schema-v2 remains `NO_GO`;
- no speed, quality, compression, or GPU-memory benefit is proven;
- next gate is native Qwen3.5 model-math/kernel design and remote correctness
  only after GPU0 admission.
