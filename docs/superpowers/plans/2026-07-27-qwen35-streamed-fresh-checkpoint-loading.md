# Qwen3.5 Streamed Fresh Checkpoint Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream a Qwen3.5 checkpoint one CPU source tensor at a time into a fresh unpublished model candidate, discard failures without rollback snapshots, and publish only a completely loaded owner through a one-shot slot.

**Architecture:** Refactor the existing assignment executor around a shared per-source operation primitive while preserving its public rollback behavior. Add a streaming loader that constructs a candidate internally, validates every source/path/budget contract before opening files, assigns one source group at a time, releases it, and builds an owner only after success. Add an isolated publication slot that accepts only a completed loaded candidate and never replaces an occupied owner.

**Tech Stack:** Python 3.12, PyTorch CPU, safetensors 0.7.0, pathlib, dataclasses, context managers, weak references.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Inline execution only; do not use subagents.
- Do not stage, commit, merge, or delete experiment evidence.
- Use only temporary small safetensors files; do not read the real checkpoint.
- Do not start local or remote GPU work.
- Do not connect generic loader, ModelRunner, Engine, or Scheduler.
- Production `ModelRunner` must continue constructing `Qwen3ForCausalLM`.
- Preserve `RuntimeError("hybrid prefix reuse requires aligned state snapshot")`.
- State tensors remain owned only by the model/root owner and supplied pool.
- Do not create a state pool from config or copy state into Scheduler.
- Do not modify schema-v2 canonical `NO_GO`.
- Do not claim performance/cache/memory/compression/quality benefit.

---

### Task 1: Extract a Shared Per-Source Assignment Primitive

**Files:**
- Modify: `tinyvllm/models/qwen35_checkpoint_assignment.py`
- Modify: `tools/test_qwen35_checkpoint_assignment.py`

**Interfaces:**
- Consumes: `Qwen35CheckpointTensorBinding`, CPU source tensors, TP size/rank.
- Produces:

```python
def _assign_qwen35_checkpoint_source_bindings(
    bindings: tuple[Qwen35CheckpointTensorBinding, ...],
    source: torch.Tensor,
    *,
    tensor_parallel_size: int,
    tensor_parallel_rank: int,
) -> int:
    ...
```

- [x] **Step 1: Add RED coverage for one source group**

Select the two packed gate/up bindings that share one destination. Call the
new primitive separately for each source and independently verify both packed
slices. Add cases for mixed source names, empty bindings, wrong source
shape/dtype/device, invalid TP context, and a loader failure.

- [x] **Step 2: Run assignment RED**

Run:

```bash
python tools/test_qwen35_checkpoint_assignment.py
```

Expected: import or attribute failure for
`_assign_qwen35_checkpoint_source_bindings`.

- [x] **Step 3: Refactor preparation by source group**

Keep `_validate_source()`, `_transform_source()`, direct-buffer preparation,
loader-kind validation, and operation execution in the assignment module.
Require a non-empty exact-binding tuple whose entries all share one source
name and identical metadata.

- [x] **Step 4: Preserve transactional public assignment**

Make `assign_qwen35_checkpoint_tensors()` group operations by source only after
its existing exact-coverage and complete prevalidation. Preserve unique
destination snapshots, binding-plan execution order, contextual errors, and
full rollback.

- [x] **Step 5: Run assignment GREEN**

Run:

```bash
python tools/test_qwen35_checkpoint_assignment.py
```

Expected:

```text
qwen35 checkpoint assignment tests passed
```

### Task 2: Add Streamed Fresh-Candidate RED Tests

**Files:**
- Create: `tools/test_qwen35_streamed_fresh_checkpoint.py`

**Interfaces:**
- Consumes the existing assignment/reader fixture helpers.
- Produces requirements for:

```python
load_qwen35_fresh_checkpoint_candidate(...)
Qwen35StreamedCheckpointLoadStats
Qwen35LoadedCheckpointCandidate
```

- [x] **Step 1: Build a fresh candidate factory**

Wrap the two-layer fixture so every invocation creates a distinct packed model,
its supplied pool, and its binding plan. Retain diagnostic references in the
test only; initialize destinations to sentinels.

- [x] **Step 2: Assert TP=1/2 exact streamed success**

Create two temporary shards with extra unrequested tensors. For TP=1 rank 0
and TP=2 ranks 0/1, verify exact independent destination expectations, tied
embedding storage, owner coherence, one factory call, 27 bindings, 27 sources,
two shards, exact total bytes, and exact largest-source bytes.

- [x] **Step 3: Assert one-source retention**

Wrap the internal source-group assignment hook. Store only a weak reference to
the current source and force collection before the next invocation. Assert the
prior source is no longer live and that the returned candidate retains no
source mapping.

- [x] **Step 4: Add pre-open failure matrix**

Cover invalid factory/budget/result, non-CPU destination, conflicting source
contract, oversized source, unsafe path, and missing shard. Track `safe_open`
and assert zero entries for failures that can be detected before opening.

- [x] **Step 5: Add in-stream failure matrix**

Cover missing source, wrong shape/dtype, and a late custom-loader exception.
Assert balanced handle entry/exit and no loaded candidate.

- [x] **Step 6: Prove discard instead of rollback**

After the late loader fails, inspect the diagnostically retained private
candidate. Assert an earlier destination remains loaded rather than restored,
proving the streaming path did not allocate or execute full-model rollback.

- [x] **Step 7: Run streaming RED**

Run:

```bash
python tools/test_qwen35_streamed_fresh_checkpoint.py
```

Expected missing module:

```text
tinyvllm.models.qwen35_checkpoint_streaming
```

### Task 3: Implement Streamed Fresh-Candidate Loading

**Files:**
- Create: `tinyvllm/models/qwen35_checkpoint_streaming.py`

**Interfaces:**
- Produces the two frozen result dataclasses and
  `load_qwen35_fresh_checkpoint_candidate()`.
- Uses `_assign_qwen35_checkpoint_source_bindings()` from Task 1.

- [x] **Step 1: Validate factory, candidate, budget, and destination ownership**

Invoke the factory exactly once. Validate exact tuple/model/plan types, positive
non-boolean budget, CPU non-meta destinations, and that every destination
object appears in the candidate model's registered parameters or buffers.

- [x] **Step 2: Build immutable unique source contracts**

Reject conflicting duplicate source metadata/shards. Compute each exact source
byte count, total bytes, and maximum bytes. Reject any individual source over
`max_tensor_bytes` before opening a shard.

- [x] **Step 3: Validate directory and all shard paths before opening**

Reuse the bounded reader's safe relative-path rules without importing or
calling its all-source materialization API.

- [x] **Step 4: Stream one source group at a time**

Open each shard once in deterministic order. For each requested source, call
`get_tensor()`, validate CPU shape/dtype/bytes, assign its complete binding
group, then delete source/transformed references before the next source.

- [x] **Step 5: Build owner only after complete success**

After all handles close and coverage is exact, call
`build_qwen35_hybrid_model_owner(model)` and return the loaded candidate with
scalar stats. Wrap assignment exceptions with source/target context and never
return the private candidate on failure.

- [x] **Step 6: Run streaming GREEN**

Run:

```bash
python tools/test_qwen35_streamed_fresh_checkpoint.py
```

Expected:

```text
qwen35 streamed fresh checkpoint tests passed
```

### Task 4: Add One-Shot Owner Publication

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_model_publication.py`
- Create: `tools/test_qwen35_hybrid_model_publication.py`

**Interfaces:**
- Consumes: exact `Qwen35LoadedCheckpointCandidate`.
- Produces:

```python
class Qwen35HybridModelOwnerPublicationSlot:
    @property
    def owner(self) -> Qwen35HybridModelOwner | None:
        ...

    def publish(
        self,
        candidate: Qwen35LoadedCheckpointCandidate,
    ) -> Qwen35HybridModelOwner:
        ...
```

- [x] **Step 1: Write publication RED tests**

Assert an empty slot, invalid candidate rejection, exact successful owner
identity, occupied-slot rejection, first-owner preservation, and no clear or
replace API.

- [x] **Step 2: Add failed-load isolation test**

Create an occupied slot, inject a fresh-candidate stream failure, and assert
the slot still contains the original owner.

- [x] **Step 3: Run publication RED**

Run:

```bash
python tools/test_qwen35_hybrid_model_publication.py
```

Expected missing module:

```text
tinyvllm.engine.qwen35_hybrid_model_publication
```

- [x] **Step 4: Implement minimal one-shot slot**

Validate exact candidate and coherent owner graph before assignment. Reject an
occupied slot. Publish with one final `_owner = candidate.owner` reference
write and return that owner.

- [x] **Step 5: Run publication GREEN**

Run:

```bash
python tools/test_qwen35_hybrid_model_publication.py
```

Expected:

```text
qwen35 hybrid model publication tests passed
```

### Task 5: Regression, Static Guards, and Handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-qwen35-streamed-fresh-checkpoint-loading.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes all completed checkpoint, graph, owner, and production guard gates.
- Produces a unique canonical EOF handoff section.

- [x] **Step 1: Run focused checkpoint regressions**

Run streamed loader, publication, bounded reader, transactional assignment,
target binding, metadata, concrete factory, real binding, root assembly,
transactional root, and native owner tests.

- [x] **Step 2: Run focused compilation and static production guards**

Compile changed modules/tests with Python 3.12. Verify production ModelRunner
still constructs `Qwen3ForCausalLM`, Scheduler retains the aligned-state
RuntimeError, Engine has no streaming/publication references, and new modules
contain no CUDA or real-checkpoint path.

- [x] **Step 3: Run workspace hygiene checks**

Run:

```bash
git diff --check
git diff --cached --name-only
```

Expected: no whitespace errors and no staged files.

- [x] **Step 4: Complete the plan and append canonical handoff**

Record exact RED/GREEN evidence, test counts, source-retention proof,
failure-discard semantics, publication behavior, allowed conclusion, and
remaining limitations. Make the new `##` heading unique and the actual final
level-two heading in `AGENT_HANDOFF_STATE.md`.

- [x] **Step 5: Define the next gate conservatively**

The next gate may evaluate real-checkpoint CPU feasibility or sub-tensor
streaming for the 1.017 GB embedding, but must remain production-unwired and
must not claim runtime benefit before token/logit and performance gates.

