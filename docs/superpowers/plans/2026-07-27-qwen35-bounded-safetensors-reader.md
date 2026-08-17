# Qwen3.5 Bounded Safetensors Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read exactly the requested Qwen3.5 tensors from temporary CPU safetensors shards under a hard byte budget, then call transactional assignment only after complete validation.

**Architecture:** Add a reader layer before the completed assignment executor. It validates required bytes and shard paths before opening files, loads only binding-plan source names inside context managers, publishes an immutable source mapping after all shards close, and optionally feeds the mapping to transactional assignment.

**Tech Stack:** Python 3.12, PyTorch CPU, safetensors 0.7.0, pathlib, dataclasses, MappingProxyType.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Use only temporary small safetensors files; do not read the real checkpoint.
- Do not start local/remote GPU work.
- Do not connect generic loader, ModelRunner, Engine, or Scheduler.
- Preserve production `Qwen3ForCausalLM` and Scheduler aligned-state guard.
- Do not modify schema-v2 canonical `NO_GO`.
- Do not claim performance/cache/memory/compression/quality benefit.

---

### Task 1: Write Bounded Reader RED Tests

**Files:**
- Create: `tools/test_qwen35_checkpoint_reader.py`

**Interfaces:**
- Consumes the existing assignment fixture/helpers.
- Produces requirements for:

```python
materialize_qwen35_checkpoint_sources(...)
load_and_assign_qwen35_checkpoint(...)
```

- [x] **Step 1: Create two temporary safetensors shards**

Rewrite the fixture tensor plan so requested sources are split across two safe
relative shard names. Save deterministic source tensors plus unrelated extras
with `safetensors.torch.save_file()`.

- [x] **Step 2: Assert exact positive materialization**

Assert 27 requested tensors, two shards, exact byte count, exact values, CPU
device, immutable mapping container, and no extra tensors.

- [x] **Step 3: Assert TP=1/2 load-and-assign**

For every rank, initialize destination sentinels, call the combined API, and
reuse independent assignment expectations to validate all destination values.

- [x] **Step 4: Add budget/path/metadata failure matrix**

Cover invalid/insufficient budget, missing directory/shard/source, corrupt
shape/dtype, and conflicting duplicate contract. Assert destinations remain
unchanged and no assignment loader runs.

- [x] **Step 5: Track file-handle cleanup**

Wrap module `safe_open` with a context manager that records enter/exit.
Assert every entered handle exits on success and failure.

- [x] **Step 6: Assert assignment starts after handles close**

Inject a late custom-loader failure and assert all tracking handles are closed
before the loader executes; then assert transactional rollback.

- [x] **Step 7: Run RED**

Expected missing module:

```text
tinyvllm.models.qwen35_checkpoint_reader
```

### Task 2: Implement Reader and Combined Boundary

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_reader.py`

**Interfaces:**
- Produces materialization/load result dataclasses and two public functions.

- [x] **Step 1: Validate plan, directory, budget, and required bytes**

Reject before file open whenever possible. Derive unique source contracts and
safe shard paths.

- [x] **Step 2: Materialize requested sources inside context managers**

Open each required shard once, verify requested keys, call `get_tensor()` only
for requested names, and validate CPU shape/dtype/bytes.

- [x] **Step 3: Publish immutable source mapping**

Return a `MappingProxyType` only after all shards close and all requested
sources validate.

- [x] **Step 4: Implement load-then-assign**

Call materialization first, then `assign_qwen35_checkpoint_tensors()`, and
return both frozen result records.

- [x] **Step 5: Run GREEN**

Expected:

```text
qwen35 checkpoint reader tests passed
```

### Task 3: Regression and Handoff

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-bounded-safetensors-reader.md`

- [x] **Step 1: Run reader/assignment/binding/metadata/factory/root regressions**

- [x] **Step 2: Run py_compile and production/static guards**

- [x] **Step 3: Complete plan and append unique EOF handoff**

Next gate:

```text
real 320-entry bounded CPU dry materialization feasibility or chunked/streamed
assignment design, still without GPU or production wiring
```
