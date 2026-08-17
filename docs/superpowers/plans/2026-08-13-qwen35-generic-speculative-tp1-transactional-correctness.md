# Qwen3.5 Generic Speculative TP1 Transactional Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Subagents
> are prohibited for this workspace, so execute every task inline and stop at
> the explicit review checkpoints.

**Goal:** Establish a source-bound real-checkpoint TP1 authority proving that
the generic n-gram speculative runtime preserves exact greedy Qwen3.5 output
while committing only the accepted full-attention KV and recurrent-state
prefix.

**Architecture:** Add a draft-source-neutral optional side-state transaction
to the generic speculative batch lifecycle. Refactor the existing Qwen3.5
prepare/commit split so speculative callbacks retain private candidate state
and per-input-prefix checkpoints without mutating live leases, then join
reversible side-state apply/rollback/seal with the existing KV and Scheduler
publication boundary.

**Tech Stack:** Python 3, PyTorch, TinyLLMForge `LLMEngine`/`ModelRunner`,
Qwen3.5 hybrid state, pytest, JSON authority artifacts, SSH/rsync remote GPU
execution.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Use `apply_patch` for every file edit.
- Do not switch branches or create a worktree.
- Do not stage, commit, stash, reset, clean, or push.
- Do not use subagents.
- Every behavior change requires an observed focused RED before production
  code and a focused GREEN afterward.
- The remote target is only `sitian@10.232.195.203`.
- Set
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use SSH with `ControlMaster=no` and `ControlPath=none`.
- Run SSH and rsync serially with bounded retries.
- Reuse one unified exec session when a long-running process is required.
- Bind the real checkpoint manifest
  `3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0`.
- Do not enable Qwen3.5 KV offload in this TP1 gate.
- Do not count synthetic or simulated movement as KV-offload evidence.
- Do not use a second accepted-prefix model forward as the production
  recurrent-state solution.
- Do not claim TP4 second-model support, 16K/32K second-model support,
  performance, learned-drafter support, or Phase 1 completion.
- Keep existing Qwen3 generic speculative contracts and artifacts
  backward-compatible.

---

## File Structure

### New production files

- `tinyvllm/engine/speculative_side_state.py`
  - model-neutral side-state lifecycle types;
  - canonical committed-input mapping;
  - callback and receipt validation.
- `tinyvllm/engine/qwen35_speculative_state.py`
  - Qwen3.5 active transaction ownership;
  - first-target and verification candidate-state retention;
  - checkpoint selection;
  - reversible apply, seal, and rollback.

### Modified production files

- `tinyvllm/speculative/batch_runtime.py`
  - prepare/select/rollback the optional side-state transaction;
  - retain selected side-state handle in the prepared batch.
- `tinyvllm/layers/gated_delta.py`
  - deterministic token-prefix recurrent trace used only when requested.
- `tinyvllm/layers/qwen35_linear_attention.py`
  - optional convolution/recurrent prefix-state capture.
- `tinyvllm/layers/qwen35_packed_layer_stack.py`
  - run from supplied candidate state;
  - return per-sequence, per-prefix cross-layer checkpoints.
- `tinyvllm/models/qwen35_packed.py`
  - expose a prepared model step with commit suppressed;
  - preserve ordinary `run_step()` behavior.
- `tinyvllm/engine/model_runner.py`
  - install the Qwen3.5 side-state owner;
  - route speculative Qwen3.5 callbacks through prepared steps;
  - expose acknowledged lifecycle methods.
- `tinyvllm/engine/speculative_model_runner.py`
  - construct model-runner side-state callbacks without putting tensors in
    Engine-owned results.
- `tinyvllm/engine/llm_engine.py`
  - reversibly apply side state before KV/Scheduler visibility;
  - rollback before visibility;
  - seal after successful publication.

### New authority files

- `tools/qwen35_generic_speculative_tp1_gate.py`
- `tools/qwen35_generic_speculative_tp1_worker.py`
- `tools/verify_qwen35_generic_speculative_tp1_gate.py`
- `tools/run_qwen35_generic_speculative_tp1_gate_remote.sh`

### New focused tests

- `tools/test_speculative_side_state.py`
- `tools/test_qwen35_gated_delta_prefix_trace.py`
- `tools/test_qwen35_prepared_model_step.py`
- `tools/test_qwen35_speculative_state.py`
- `tools/test_qwen35_generic_speculative_tp1_gate.py`

### Modified focused tests

- `tools/test_speculative_batch_runtime.py`
- `tools/test_speculative_model_runner_callbacks.py`
- `tools/test_model_runner_spec_verify.py`
- `tools/test_ngram_speculative.py`

---

### Task 1: Canonical Side-State Contract and Consumed-Input Mapping

**Files:**
- Create: `tinyvllm/engine/speculative_side_state.py`
- Create: `tools/test_speculative_side_state.py`

**Interfaces:**
- Produces:
  - `SpeculativeSideStateSelectionRow`
  - `SpeculativeSideStateCallbacks`
  - `build_speculative_side_state_selection_rows(prepared_rows)`
  - `validate_speculative_side_state_receipt(receipt, ...)`
- Consumes only immutable prepared speculative row attributes; no model or
  tensor dependency.

- [ ] **Step 1: Write failing mapping and lifecycle tests**

Add tests that construct lightweight prepared rows with `SimpleNamespace`:

```python
from types import SimpleNamespace

from tinyvllm.engine.speculative_side_state import (
    SpeculativeSideStateCallbacks,
    build_speculative_side_state_selection_rows,
)


def _row(
    sequence_id,
    *,
    proposal_tokens,
    accepted_tokens,
    verify_input_count,
):
    plan = (
        None
        if verify_input_count == 0
        else SimpleNamespace(query_len=verify_input_count)
    )
    return SimpleNamespace(
        sequence_id=sequence_id,
        proposal=SimpleNamespace(token_ids=proposal_tokens),
        accepted_tokens=accepted_tokens,
        plan=plan,
    )


def test_selection_uses_consumed_inputs_not_emitted_outputs():
    rows = build_speculative_side_state_selection_rows((
        _row(
            7,
            proposal_tokens=(11, 12, 13, 14),
            accepted_tokens=(11, 12),
            verify_input_count=3,
        ),
    ))
    assert rows[0].sequence_id == 7
    assert rows[0].accepted_draft_count == 2
    assert rows[0].verify_input_count == 3
    assert rows[0].committed_tail_input_count == 2
    assert rows[0].committed_input_count == 3


def test_fully_accepted_proposal_leaves_last_output_unconsumed():
    rows = build_speculative_side_state_selection_rows((
        _row(
            8,
            proposal_tokens=(21, 22, 23, 24),
            accepted_tokens=(21, 22, 23, 24),
            verify_input_count=3,
        ),
    ))
    assert rows[0].committed_tail_input_count == 3
    assert rows[0].committed_input_count == 4


def test_callbacks_require_five_callable_phases():
    callbacks = SpeculativeSideStateCallbacks(
        prepare=lambda sequences: object(),
        select=lambda handle, rows: object(),
        apply=lambda handle: {},
        seal=lambda handle: {},
        rollback=lambda handle: {},
    )
    assert callable(callbacks.rollback)
```

Also test:

- zero accepted drafts maps to `committed_input_count=1`;
- one-token proposal has `verify_input_count=0`;
- duplicate sequence IDs fail;
- accepted tokens must be an exact proposal prefix;
- `verify_input_count` must equal `len(proposal)-1` when a proposal exists;
- non-callable lifecycle members fail construction; and
- receipt sequence inventory must exactly match the selection inventory.

- [ ] **Step 2: Run the test and observe RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_side_state.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'tinyvllm.engine.speculative_side_state'`.

- [ ] **Step 3: Implement the immutable contract**

Create:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SpeculativeSideStateSelectionRow:
    sequence_id: int
    proposal_token_count: int
    accepted_draft_count: int
    verify_input_count: int
    committed_tail_input_count: int
    committed_input_count: int


@dataclass(frozen=True)
class SpeculativeSideStateCallbacks:
    prepare: Callable[[tuple[object, ...]], object]
    select: Callable[
        [object, tuple[SpeculativeSideStateSelectionRow, ...]],
        object,
    ]
    apply: Callable[[object], object]
    seal: Callable[[object], object]
    rollback: Callable[[object], object]

    def __post_init__(self):
        for name in ("prepare", "select", "apply", "seal", "rollback"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")


def build_speculative_side_state_selection_rows(
    prepared_rows: tuple[object, ...],
) -> tuple[SpeculativeSideStateSelectionRow, ...]:
    if not isinstance(prepared_rows, tuple) or not prepared_rows:
        raise ValueError("prepared_rows must be a non-empty tuple")
    result = []
    seen = set()
    for row in prepared_rows:
        sequence_id = getattr(row, "sequence_id", None)
        if (
            isinstance(sequence_id, bool)
            or not isinstance(sequence_id, int)
            or sequence_id < 0
            or sequence_id in seen
        ):
            raise ValueError("side-state sequence IDs must be unique")
        proposal_tokens = getattr(
            getattr(row, "proposal", None),
            "token_ids",
            None,
        )
        accepted_tokens = getattr(row, "accepted_tokens", None)
        if not isinstance(proposal_tokens, tuple):
            raise ValueError("proposal token_ids must be a tuple")
        if not isinstance(accepted_tokens, tuple):
            raise ValueError("accepted_tokens must be a tuple")
        if (
            accepted_tokens
            != proposal_tokens[:len(accepted_tokens)]
        ):
            raise ValueError(
                "accepted tokens must be an exact proposal prefix"
            )
        plan = getattr(row, "plan", None)
        verify_input_count = (
            0 if plan is None else getattr(plan, "query_len", None)
        )
        expected_verify = max(0, len(proposal_tokens) - 1)
        if verify_input_count != expected_verify:
            raise ValueError(
                "verify input count must equal proposal length minus one"
            )
        committed_tail = min(
            len(accepted_tokens),
            verify_input_count,
        )
        result.append(SpeculativeSideStateSelectionRow(
            sequence_id=sequence_id,
            proposal_token_count=len(proposal_tokens),
            accepted_draft_count=len(accepted_tokens),
            verify_input_count=verify_input_count,
            committed_tail_input_count=committed_tail,
            committed_input_count=1 + committed_tail,
        ))
        seen.add(sequence_id)
    return tuple(result)
```

Implement receipt validation in the same file with exact fields:

```python
RECEIPT_FIELDS = {
    "operation",
    "status",
    "transaction_id",
    "sequence_ids",
}
```

Require operation/status pairs:

- `prepare/prepared`
- `select/selected`
- `apply/applied`
- `seal/sealed`
- `rollback/rolled_back`

- [ ] **Step 4: Run focused GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_side_state.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Run hygiene checkpoint without git mutation**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/speculative_side_state.py \
  tools/test_speculative_side_state.py
git diff --check -- \
  tinyvllm/engine/speculative_side_state.py \
  tools/test_speculative_side_state.py
```

Expected: both commands succeed. Do not stage or commit.

---

### Task 2: Bind Side-State Lifecycle to Prepared Speculative Batches

**Files:**
- Modify: `tinyvllm/speculative/batch_runtime.py`
- Modify: `tools/test_speculative_batch_runtime.py`

**Interfaces:**
- Consumes:
  - `SpeculativeSideStateCallbacks`
  - `build_speculative_side_state_selection_rows`
- Produces new `PreparedNativeSpeculativeBatch` fields:
  - `side_state_callbacks`
  - `side_state_handle`
  - `side_state_selection`
  - `side_state_state`
- Produces helpers:
  - `apply_prepared_speculative_side_state(prepared)`
  - `seal_prepared_speculative_side_state(prepared)`
  - `rollback_prepared_speculative_side_state(prepared)`

- [ ] **Step 1: Write failing lifecycle-order tests**

Add a recorder:

```python
class RecordingSideState:
    def __init__(self):
        self.events = []

    def callbacks(self):
        return SpeculativeSideStateCallbacks(
            prepare=self.prepare,
            select=self.select,
            apply=self.apply,
            seal=self.seal,
            rollback=self.rollback,
        )

    def prepare(self, sequences):
        self.events.append(("prepare", tuple(seq.seq_id for seq in sequences)))
        return {"transaction_id": "side-1"}

    def select(self, handle, rows):
        self.events.append(("select", rows))
        return {"transaction_id": handle["transaction_id"]}

    def apply(self, handle):
        self.events.append(("apply", handle["transaction_id"]))
        return {}

    def seal(self, handle):
        self.events.append(("seal", handle["transaction_id"]))
        return {}

    def rollback(self, handle):
        self.events.append(("rollback", handle["transaction_id"]))
        return {}
```

Assert:

- `prepare` precedes first-target callback;
- `select` occurs after acceptance;
- partial acceptance produces the canonical mapping from Task 1;
- callback failure invokes rollback once;
- KV reservation failure invokes rollback once;
- side-state select failure rolls back active KV transactions;
- `apply`, `seal`, and `rollback` enforce legal state transitions; and
- the no-provider path preserves existing results byte-for-byte.

- [ ] **Step 2: Run focused RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_batch_runtime.py \
  -k 'side_state' -q
```

Expected: tests fail because `prepare_native_speculative_batch()` does not
accept `side_state_callbacks`.

- [ ] **Step 3: Add optional callbacks to prepare**

Extend the signature:

```python
def prepare_native_speculative_batch(
    *,
    block_manager,
    seqs,
    eos_token,
    run_tail_batch,
    draft_adapter=None,
    run_first_targets=None,
    run_first_targets_and_proposals=None,
    side_state_callbacks=None,
) -> PreparedNativeSpeculativeBatch:
```

Begin before first-target execution:

```python
side_state_handle = None
if side_state_callbacks is not None:
    side_state_handle = side_state_callbacks.prepare(seqs)
```

After `prepared_rows` are built:

```python
side_state_selection = ()
if side_state_callbacks is not None:
    side_state_selection = (
        build_speculative_side_state_selection_rows(
            tuple(
                prepared_rows[sequence_id]
                for sequence_id in sequence_ids
            )
        )
    )
    side_state_callbacks.select(
        side_state_handle,
        side_state_selection,
    )
```

On every exception after `prepare`, call side-state rollback and include a
rollback failure in `NativeSpeculativeBatchError.rollback_errors` under key
`"side_state"` without discarding KV rollback errors.

- [ ] **Step 4: Add explicit apply/seal/rollback helpers**

Add:

```python
def apply_prepared_speculative_side_state(prepared):
    _require_prepared_batch(prepared)
    if prepared.side_state_callbacks is None:
        return None
    if prepared.side_state_state != "selected":
        raise RuntimeError("side state must be selected before apply")
    receipt = prepared.side_state_callbacks.apply(
        prepared.side_state_handle
    )
    prepared.side_state_state = "applied"
    return receipt


def seal_prepared_speculative_side_state(prepared):
    if prepared.side_state_callbacks is None:
        return None
    if prepared.side_state_state != "applied":
        raise RuntimeError("side state must be applied before seal")
    receipt = prepared.side_state_callbacks.seal(
        prepared.side_state_handle
    )
    prepared.side_state_state = "sealed"
    return receipt


def rollback_prepared_speculative_side_state(prepared):
    if prepared.side_state_callbacks is None:
        return None
    if prepared.side_state_state == "sealed":
        raise RuntimeError("sealed side state cannot be rolled back")
    receipt = prepared.side_state_callbacks.rollback(
        prepared.side_state_handle
    )
    prepared.side_state_state = "rolled_back"
    return receipt
```

Make legacy `commit_prepared_native_speculative_batch()` apply side state
before KV commit, roll it back if KV commit fails, and seal only after KV
metadata commit succeeds.

- [ ] **Step 5: Run focused and compatibility GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_batch_runtime.py \
  tools/test_ngram_speculative.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Review checkpoint**

Run:

```bash
git diff --check -- \
  tinyvllm/speculative/batch_runtime.py \
  tools/test_speculative_batch_runtime.py
```

Expected: success. Do not stage or commit.

---

### Task 3: Expose a Prepared Qwen3.5 Model Step

**Files:**
- Modify: `tinyvllm/models/qwen35_packed.py`
- Create: `tools/test_qwen35_prepared_model_step.py`

**Interfaces:**
- Produces:
  - `Qwen35PreparedModelStep`
  - `Qwen35PackedForCausalLM.prepare_step(...)`
  - `Qwen35PackedForCausalLM.commit_prepared_step(...)`
- Preserves the existing `run_step(...) -> (normalized, logits)` interface.

- [ ] **Step 1: Write failing no-mutation tests**

Use the repository's existing small packed-model fixtures. Assert:

```python
prepared = model.prepare_step(
    leases,
    token_counts,
    input_ids,
    position_ids,
)
assert torch.equal(pool_state_after_prepare, pool_state_before)
assert prepared.normalized.shape[0] == input_ids.shape[0]
assert prepared.logits.shape[0] > 0

model.commit_prepared_step(leases, prepared)
assert not torch.equal(pool_state_after_commit, pool_state_before)
```

Also assert:

- committing twice fails;
- committing with different leases fails;
- ordinary `run_step()` output and final pool state equal
  `prepare_step()+commit_prepared_step()`; and
- a failed final norm or LM head leaves live state unchanged.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_prepared_model_step.py -q
```

Expected: failure because `prepare_step` does not exist.

- [ ] **Step 3: Add the prepared result and split existing run_step**

Add:

```python
@dataclass
class Qwen35PreparedModelStep:
    leases: tuple[HybridStateLease, ...]
    token_counts: tuple[int, ...]
    normalized: torch.Tensor
    logits: torch.Tensor | None
    final_candidates: tuple[
        tuple[torch.Tensor, torch.Tensor],
        ...,
    ]
    prefix_candidates: object | None = None
    state: str = "prepared"
```

Implement:

```python
def prepare_step(
    self,
    leases,
    token_counts,
    input_ids,
    position_ids,
    input_embeds=None,
):
    self._validate_public_inputs(
        leases,
        token_counts,
        input_ids,
        input_embeds,
    )
    hidden_states = (
        self.embed_tokens(input_ids)
        if input_embeds is None
        else input_embeds
    )
    hidden_states, final_candidates = self.layer_stack.prepare(
        leases,
        token_counts,
        position_ids,
        hidden_states,
    )
    normalized = self.final_norm(hidden_states)
    logits = self.lm_head(normalized)
    return Qwen35PreparedModelStep(
        leases=leases,
        token_counts=token_counts,
        normalized=normalized,
        logits=logits,
        final_candidates=final_candidates,
    )
```

`commit_prepared_step()` validates exact lease identity and calls the existing
cross-layer commit once.

Rewrite `run_step()` as prepare followed by commit while preserving its
return type.

Task 5 later extends `prepare_step()` with `initial_candidates` and
`capture_prefix_states`; Task 3 must use only the already-existing
`layer_stack.prepare()` API so this task can independently reach GREEN.

- [ ] **Step 4: Run GREEN and existing model tests**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_qwen35_root_model_assembly_factory.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Review checkpoint**

Run:

```bash
git diff --check -- \
  tinyvllm/models/qwen35_packed.py \
  tools/test_qwen35_prepared_model_step.py
```

Expected: success.

---

### Task 4: Deterministic Token-Prefix Convolution and Recurrent Traces

**Files:**
- Modify: `tinyvllm/layers/gated_delta.py`
- Create: `tools/test_qwen35_gated_delta_prefix_trace.py`

**Interfaces:**
- Produces:
  - `qwen35_causal_depthwise_conv_prefix_trace(...)`
  - `qwen35_gated_delta_prefix_trace(...)`
- The trace APIs are opt-in and do not change ordinary chunk execution.

- [ ] **Step 1: Write failing sequential-oracle tests**

For two and four input tokens:

```python
output, states = qwen35_gated_delta_prefix_trace(
    query,
    key,
    value,
    a,
    b,
    A_log,
    dt_bias,
    initial_state,
)
assert states.shape == (
    token_count,
    heads,
    value_dim,
    key_dim,
)
```

Construct the oracle by repeatedly calling
`qwen35_gated_delta_recurrent()` on one token and assert every output row and
state checkpoint matches the corresponding oracle row with the same explicit
tolerance already used by gated-delta reference tests.

For convolution, repeatedly append one projected row and assert every trace
checkpoint matches the sequential sliding-window state.

Test empty input rejection, dtype/device preservation, and no mutation of the
initial state.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_gated_delta_prefix_trace.py -q
```

Expected: import failure for the new trace functions.

- [ ] **Step 3: Implement the trace functions**

The recurrent trace deliberately uses the existing one-token reference kernel
inside one outer model callback:

```python
def qwen35_gated_delta_prefix_trace(
    query,
    key,
    value,
    a,
    b,
    A_log,
    dt_bias,
    recurrent_state_v_k,
):
    if query.shape[0] <= 0:
        raise ValueError("prefix trace requires at least one token")
    state = recurrent_state_v_k.clone()
    outputs = []
    states = []
    for index in range(query.shape[0]):
        output, state = qwen35_gated_delta_recurrent(
            query[index:index + 1],
            key[index:index + 1],
            value[index:index + 1],
            a[index:index + 1],
            b[index:index + 1],
            A_log,
            dt_bias,
            state,
        )
        outputs.append(output)
        states.append(state)
    return torch.cat(outputs, dim=0), torch.stack(states)
```

The convolution trace builds a checkpoint after each input row while reusing
the existing validation and activation semantics.

This is a correctness path, not a performance claim. It is not an
accepted-prefix replay because it executes once before acceptance and retains
all candidate prefixes.

- [ ] **Step 4: Run focused and primitive GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_gated_delta_prefix_trace.py \
  tools/test_qwen35_gated_delta_reference.py \
  tools/test_qwen35_gated_delta_chunk.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Review checkpoint**

Run:

```bash
git diff --check -- \
  tinyvllm/layers/gated_delta.py \
  tools/test_qwen35_gated_delta_prefix_trace.py
```

Expected: success.

---

### Task 5: Capture Cross-Layer Prefix Candidates in One Prepared Step

**Files:**
- Modify: `tinyvllm/layers/qwen35_linear_attention.py`
- Modify: `tinyvllm/layers/qwen35_packed_layer_stack.py`
- Modify: `tinyvllm/models/qwen35_packed.py`
- Modify: `tools/test_qwen35_packed_layer_stack.py`
- Modify: `tools/test_qwen35_linear_attention_shell.py`
- Modify: `tools/test_qwen35_prepared_model_step.py`

**Interfaces:**
- Produces:
  - `Qwen35PreparedLayerStack`
  - `Qwen35PackedHeterogeneousLayerStack.prepare_transactional(...)`
- Consumes the trace functions from Task 4.

- [ ] **Step 1: Write failing linear-attention trace tests**

Add:

```python
output, final_conv, final_recurrent, trace = layer.forward_with_state_trace(
    hidden_states,
    convolution_state,
    recurrent_state,
)
assert trace.convolution.shape[0] == hidden_states.shape[0]
assert trace.recurrent.shape[0] == hidden_states.shape[0]
assert torch.equal(trace.convolution[-1], final_conv)
assert torch.equal(trace.recurrent[-1], final_recurrent)
```

Compare each trace prefix with independent sequential one-token execution.
Require the ordinary `forward()` output and state to remain unchanged.

- [ ] **Step 2: Write failing layer-stack transaction tests**

For batch 1 and batch 4 with fixed per-sequence query length:

- supplied `initial_candidates` replace live gathered state;
- live pool tensors remain unchanged;
- `prefix_candidates[sequence_index][prefix_index]` contains every linear
  layer exactly once;
- the last prefix checkpoint equals `final_candidates`;
- mismatched layer, batch, or prefix inventory fails; and
- `capture_prefix_states=False` returns no trace and preserves the existing
  fast path.

- [ ] **Step 3: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_linear_attention_shell.py \
  tools/test_qwen35_packed_layer_stack.py \
  -k 'prefix or transactional' -q
```

Expected: failures for missing trace and transactional APIs.

- [ ] **Step 4: Add opt-in trace execution to linear attention**

Add a frozen trace container:

```python
@dataclass(frozen=True)
class Qwen35LinearStateTrace:
    convolution: torch.Tensor
    recurrent: torch.Tensor
```

Refactor projection setup into a shared private method. Ordinary `forward()`
continues to select `qwen35_gated_delta_chunk` for multi-token execution.
`forward_with_state_trace()` uses the Task 4 trace functions and returns:

```python
(
    output,
    convolution_trace[-1],
    recurrent_trace[-1],
    Qwen35LinearStateTrace(
        convolution=convolution_trace,
        recurrent=recurrent_trace,
    ),
)
```

- [ ] **Step 5: Add transactional layer-stack preparation**

Add:

```python
@dataclass(frozen=True)
class Qwen35PreparedLayerStack:
    hidden_states: torch.Tensor
    final_candidates: tuple[
        tuple[torch.Tensor, torch.Tensor],
        ...,
    ]
    prefix_candidates: tuple[
        tuple[
            tuple[
                tuple[torch.Tensor, torch.Tensor],
                ...,
            ],
            ...,
        ],
        ...,
    ] | None
```

The nesting is:

```text
sequence -> consumed tail prefix -> linear layer -> (conv, recurrent)
```

`prepare_transactional()`:

- gathers live state when `initial_candidates is None`;
- otherwise validates and uses supplied candidate state;
- runs full-attention layers unchanged;
- runs linear layers with trace only when requested;
- combines layer-local traces into per-sequence cross-layer checkpoints; and
- never calls `state_transaction.commit()`.

Keep `prepare()` and `forward()` as compatibility wrappers.

- [ ] **Step 6: Extend the prepared model step with candidate inputs**

Extend `Qwen35PackedForCausalLM.prepare_step()` with:

```python
*,
initial_candidates=None,
capture_prefix_states=False,
```

Call `layer_stack.prepare_transactional()` and copy its
`final_candidates` and `prefix_candidates` into
`Qwen35PreparedModelStep`. Preserve the Task 3 behavior when both options use
their defaults.

- [ ] **Step 7: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_linear_attention_shell.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_qwen35_prepared_model_step.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Review checkpoint**

Run:

```bash
git diff --check -- \
  tinyvllm/layers/qwen35_linear_attention.py \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/models/qwen35_packed.py \
  tools/test_qwen35_linear_attention_shell.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_qwen35_prepared_model_step.py
```

Expected: success.

---

### Task 6: Implement the Qwen3.5 Reversible Side-State Owner

**Files:**
- Create: `tinyvllm/engine/qwen35_speculative_state.py`
- Create: `tools/test_qwen35_speculative_state.py`

**Interfaces:**
- Produces `Qwen35SpeculativeStateOwner` with:
  - `prepare(sequences, leases)`
  - `record_first_target(prepared_step)`
  - `initial_tail_candidates(sequence_ids)`
  - `record_tail(prepared_step, sequence_ids)`
  - `select(handle, selection_rows)`
  - `apply(handle)`
  - `seal(handle)`
  - `rollback(handle)`
- Consumes `Qwen35CrossLayerStateTransaction` and prepared-step candidates.

- [ ] **Step 1: Write failing lifecycle and partial-acceptance tests**

Use a two-layer deterministic state transaction fixture. Assert:

- `prepare` snapshots the exact live state and rejects a second active batch;
- `record_first_target` stores consumed-input checkpoint `1`;
- a three-row tail stores checkpoints `2`, `3`, `4`;
- selecting `committed_input_count=3` chooses the second tail checkpoint;
- `apply` changes live state to only the selected checkpoint;
- `rollback` after apply restores exact original tensors;
- `seal` discards originals and prevents rollback;
- selection of a missing checkpoint fails before apply;
- sequence/lease identity drift fails;
- batch-4 rows remain independent; and
- transaction results contain no tensors.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_speculative_state.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement explicit transaction records**

Define private transaction state:

```python
@dataclass
class _Qwen35SpeculativeStateBatch:
    transaction_id: str
    sequence_ids: tuple[int, ...]
    leases: tuple[HybridStateLease, ...]
    original_candidates: tuple[object, ...]
    checkpoints: dict[int, dict[int, object]]
    selected: dict[int, object]
    phase: str = "prepared"
```

Use monotonically increasing opaque transaction IDs. Do not encode dates or
ordering claims in artifact validation.

At `apply`, gather and retain the current live state, validate that it still
equals the original transaction state, then call the existing cross-layer
transaction once with selected candidates.

At rollback after apply, call the same cross-layer transaction with the
retained originals. At seal, clear all candidate and original tensor
references and clear the active batch.

- [ ] **Step 4: Return tensor-free receipts**

Every lifecycle method returns:

```python
{
    "operation": "apply",
    "status": "applied",
    "transaction_id": transaction_id,
    "sequence_ids": list(sequence_ids),
}
```

Selection receipts additionally contain integer-only rows with
`committed_input_count` and `checkpoint_index`.

- [ ] **Step 5: Run GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_speculative_state.py \
  tools/test_qwen35_cross_layer_state_transaction.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Review checkpoint**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/qwen35_speculative_state.py
git diff --check -- \
  tinyvllm/engine/qwen35_speculative_state.py \
  tools/test_qwen35_speculative_state.py
```

Expected: success.

---

### Task 7: Route Qwen3.5 Speculative Callbacks Through Candidate State

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_qwen35_model_runner_native_entry.py`

**Interfaces:**
- Consumes `Qwen35SpeculativeStateOwner`.
- Produces ModelRunner methods:
  - `prepare_speculative_side_state_batch(seqs)`
  - `select_speculative_side_state_batch(rows)`
  - `apply_speculative_side_state_batch()`
  - `seal_speculative_side_state_batch()`
  - `rollback_speculative_side_state_batch()`
- Keeps standard Qwen3 behavior unchanged.

- [ ] **Step 1: Write failing first-target no-live-mutation test**

For a Qwen3.5 ModelRunner fixture:

1. prepare side state;
2. snapshot live pool;
3. call `run_spec_first_target_batch`;
4. assert logits/tokens are returned;
5. assert live pool is unchanged; and
6. assert checkpoint `1` exists in the owner.

Also assert a standard Qwen3 fixture returns a disabled/no-op side-state
descriptor and uses its existing callback path.

- [ ] **Step 2: Write failing tail checkpoint test**

Prepare a fixed-q tail group and assert:

- tail execution starts from first-target candidates;
- live state remains unchanged;
- checkpoints exist for every consumed input count;
- result token rows retain the existing schema; and
- no tensor enters `TailBatchResult.auxiliary`.

- [ ] **Step 3: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_model_runner_native_entry.py \
  -k 'side_state or candidate_state' -q
```

Expected: failures for missing lifecycle methods and live-state mutation.

- [ ] **Step 4: Install the owner when Qwen3.5 model ownership binds**

Initialize:

```python
self.qwen35_speculative_state_owner = None
```

Inside `bind_qwen35_hybrid_model_owner()`:

```python
self.qwen35_speculative_state_owner = (
    Qwen35SpeculativeStateOwner(
        owner.state_transaction,
    )
)
```

Reject owner replacement while a speculative state transaction is active.

- [ ] **Step 5: Add a Qwen3.5 prepared eager path**

Extend `_run_model_runner_eager()` with keyword-only:

```python
prepare_qwen35_state: bool = False
initial_qwen35_candidates=None
capture_qwen35_prefix_states: bool = False
```

When the model exposes `prepare_step` and `prepare_qwen35_state` is true,
return the prepared step instead of committing it. All ordinary callers keep
the current `run_step()` behavior.

In first-target callback:

- use prepared mode only when an active Qwen3.5 side-state transaction exists;
- record the prepared step;
- return its logits; and
- do not commit.

In tail callback:

- obtain first-target candidates from the owner;
- run prepared mode with prefix capture;
- record every checkpoint; and
- return target token rows unchanged.

- [ ] **Step 6: Expose tensor-free lifecycle command methods**

Lifecycle methods validate the active transaction and delegate to the owner.
They return only validated receipts.

Register them in the existing ModelRunner acknowledged command allowlist so
the later TP4 extension can dispatch the same contract to all ranks.

- [ ] **Step 7: Run focused GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_model_runner_native_entry.py \
  tools/test_qwen35_speculative_state.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Run Qwen3 callback regression**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_ngram_speculative.py -q
```

Expected: all tests pass with no changed Qwen3 result schema.

- [ ] **Step 9: Review checkpoint**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_model_runner_native_entry.py
```

Expected: success.

---

### Task 8: Build ModelRunner Side-State Callbacks and Unified Engine Publication

**Files:**
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`
- Create: `tools/test_speculative_side_state_engine_publication.py`

**Interfaces:**
- Produces:
  - `build_model_runner_side_state_callbacks(model_runner)`
- Consumes Task 2 apply/seal/rollback helpers.

- [ ] **Step 1: Write failing callback builder tests**

Assert:

- builder returns `None` when ModelRunner reports no side-state owner;
- Qwen3.5 builder delegates prepare/select/apply/seal/rollback exactly once;
- selection rows are serialized as immutable tensor-free tuples;
- duplicate transaction operations fail; and
- ModelRunner exceptions propagate without conversion to success receipts.

- [ ] **Step 2: Write failing Engine publication-order tests**

Use fake KV, Scheduler, proposal-finalization, and side-state participants.
Require success order:

```text
side apply
KV commit
Scheduler commit
proposal finalize commit
side seal
```

Require rollback order for failures before Scheduler visibility:

```text
proposal finalize rollback, if prepared
KV transaction state restored to materialized
side rollback
```

Test failures at:

- side apply;
- KV commit;
- Scheduler commit;
- proposal-finalize prepare;
- proposal-finalize commit; and
- side seal.

Before Scheduler visibility, live side state must equal its original value.
After an irreversible visibility failure, runtime must be poisoned with an
exact reason and no success classification.

- [ ] **Step 3: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_side_state_engine_publication.py \
  -k 'side_state' -q
```

Expected: missing builder and publication hooks.

- [ ] **Step 4: Build callbacks without crossing tensor ownership**

Implement:

```python
def build_model_runner_side_state_callbacks(model_runner):
    available = getattr(
        model_runner,
        "speculative_side_state_available",
        None,
    )
    if not callable(available) or not available():
        return None
    return SpeculativeSideStateCallbacks(
        prepare=lambda seqs: model_runner.call(
            "prepare_speculative_side_state_batch",
            seqs,
        ),
        select=lambda handle, rows: model_runner.call(
            "select_speculative_side_state_batch",
            rows,
        ),
        apply=lambda handle: model_runner.call(
            "apply_speculative_side_state_batch",
        ),
        seal=lambda handle: model_runner.call(
            "seal_speculative_side_state_batch",
        ),
        rollback=lambda handle: model_runner.call(
            "rollback_speculative_side_state_batch",
        ),
    )
```

The handle is an identity receipt only. Candidate tensors stay inside each
ModelRunner process.

- [ ] **Step 5: Pass callbacks into batch preparation**

In `LLMEngine.step()`:

```python
side_state_callbacks = build_model_runner_side_state_callbacks(
    self.model_runner
)
prepared_runtime = prepare_native_speculative_batch(
    ...,
    side_state_callbacks=side_state_callbacks,
)
```

Do not enable this for a model runner without a bound side-state owner.

- [ ] **Step 6: Extend publication with reversible side-state apply**

Update `_commit_prepared_speculative_publication()`:

```python
side_applied = False
try:
    apply_prepared_speculative_side_state(prepared_runtime)
    side_applied = (
        prepared_runtime.side_state_state == "applied"
    )
    if kv_plans:
        block_manager.commit_speculative_kv_commit_batch(kv_plans)
    scheduler.commit_prepared_postprocess(prepared_scheduler)
except BaseException as error:
    if side_applied:
        rollback_prepared_speculative_side_state(prepared_runtime)
    rollback_finalize_if_prepared(...)
    raise
```

After proposal finalization succeeds:

```python
seal_prepared_speculative_side_state(prepared_runtime)
prepared_runtime.state = "committed"
```

If proposal finalization or seal fails after Scheduler visibility, set
`speculative_runtime_poisoned=True`, preserve the exact reason, and raise.

- [ ] **Step 7: Extend outer exception rollback**

When `prepared_runtime.state == "prepared"` or side state is selected/applied,
the existing Engine exception handler must roll back both side state and KV.
Aggregate rollback failures and poison the runtime rather than hiding either
failure.

- [ ] **Step 8: Run focused GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_side_state_engine_publication.py \
  tools/test_speculative_batch_runtime.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Run existing Engine speculative regression**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_ngram_speculative.py \
  tools/test_speculative_tp1_parity_gate.py \
  tools/test_generic_speculative_tp4_gate.py -q
```

Expected: all tests pass; Qwen3 artifacts retain their existing schemas.

- [ ] **Step 10: Review checkpoint**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_side_state_engine_publication.py
```

Expected: success.

---

### Task 9: Add the Qwen3.5 TP1 Authority Contract, Worker, and Verifier

**Files:**
- Create: `tools/qwen35_generic_speculative_tp1_gate.py`
- Create: `tools/qwen35_generic_speculative_tp1_worker.py`
- Create: `tools/verify_qwen35_generic_speculative_tp1_gate.py`
- Create: `tools/test_qwen35_generic_speculative_tp1_gate.py`

**Interfaces:**
- Produces schema:
  `qwen35.generic-speculative-tp1-transactional-correctness.v1`
- Final classification:
  `SECOND_MODEL_TP1_ESTABLISHED`

- [ ] **Step 1: Write failing schema and verifier tests**

Freeze:

```python
SCHEMA_VERSION = (
    "qwen35.generic-speculative-tp1-"
    "transactional-correctness.v1"
)
POLICIES = ("baseline", "ngram")
BATCH_SIZES = (1, 4)
WORLD_SIZE = 1
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe"
    "69982efedaf4a3e631ad0b14aad7dd0"
)
```

Tests must reject:

- wrong model manifest;
- a non-`qwen3_5` architecture;
- missing linear/full layer inventory;
- missing baseline or candidate cells;
- output mismatch;
- zero proposed, accepted, or rejected tokens;
- missing first-target or verify callbacks;
- missing side-state prepare/select/apply/seal receipts;
- a declared failure-path rollback without a matching rollback receipt;
- `committed_input_count` mismatch;
- any accepted-prefix replay counter above zero;
- leaked leases;
- poisoned runtime;
- incomplete cleanup;
- source-tree mismatch; and
- opaque run-ID ordering assumptions.

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp1_gate.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement the contract and independent verifier**

The verifier imports no worker objects. It recomputes:

- source inventory hashes;
- model/config identity;
- prompt and output digests;
- baseline/candidate exact parity;
- proposal/accept/reject totals;
- callback inventory;
- consumed-input mappings;
- lifecycle receipt transitions;
- no-replay counters;
- lease inventory; and
- cleanup.

The verifier returns:

```python
{
    "classification": "SECOND_MODEL_TP1_ESTABLISHED",
    "failures": [],
}
```

only when every invariant passes.

- [ ] **Step 4: Implement the real Engine worker**

The worker constructs one fresh Engine per cell with:

```python
LLM(
    model_path,
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=4096,
    max_num_batched_tokens=8192,
    max_num_seqs=batch_size,
    kv_offload_mvp0=False,
)
```

For candidate cells install:

```python
EngineSpeculativeRuntime(
    NGramDraftAdapter(
        ngram_size=3,
        max_proposal_tokens=4,
    )
)
```

The worker records:

- exact prompt token rows and output rows;
- `last_step_observation` counters;
- callback profile rows;
- side-state lifecycle receipts;
- consumed-input mappings;
- accepted-prefix replay count;
- runtime poison state;
- hybrid-state lease inventory before and after;
- Engine cleanup receipt; and
- model architecture identity.

Use one acceptance-rich repeated pattern and one rejection-rich divergence
pattern. Batch 4 contains four distinct prompt rows. Keep exact prompt length
below 4096 by the full output budget.

- [ ] **Step 5: Implement atomic campaign output**

Run four fresh subprocess cells:

```text
baseline batch 1
ngram batch 1
baseline batch 4
ngram batch 4
```

Write to a temporary run directory, independently verify it, then rename to
`authority`. On failure rename the temporary directory to
`authority.failed` and include its path in the raised error.

- [ ] **Step 6: Run focused GREEN**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp1_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Run gate plus generic regression**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp1_gate.py \
  tools/test_generic_speculative_tp4_gate.py \
  tools/test_model_runner_spec_verify.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Review checkpoint**

Run:

```bash
python3 -m py_compile \
  tools/qwen35_generic_speculative_tp1_gate.py \
  tools/qwen35_generic_speculative_tp1_worker.py \
  tools/verify_qwen35_generic_speculative_tp1_gate.py
git diff --check -- \
  tools/qwen35_generic_speculative_tp1_gate.py \
  tools/qwen35_generic_speculative_tp1_worker.py \
  tools/verify_qwen35_generic_speculative_tp1_gate.py \
  tools/test_qwen35_generic_speculative_tp1_gate.py
```

Expected: success.

---

### Task 10: Add the Serial Remote Runner and Produce Real Authority

**Files:**
- Create: `tools/run_qwen35_generic_speculative_tp1_gate_remote.sh`
- Modify: `tools/test_qwen35_generic_speculative_tp1_gate.py`

**Interfaces:**
- Consumes the Task 9 campaign CLI.
- Produces:
  - `artifacts/qwen35_generic_speculative_tp1/<opaque-id>/authority`
  - `artifacts/qwen35_generic_speculative_tp1/last_completed_run_path.txt`

- [x] **Step 1: Write failing runner contract tests**

Parse the shell source and require:

- exact remote user/host;
- exact Kerberos cache;
- `ControlMaster=no`;
- `ControlPath=none`;
- serial SSH and rsync operations;
- bounded retry counts;
- real GPU free-memory preflight;
- approved checkpoint path and manifest;
- no `kv_offload_mvp0=True`;
- source closure upload;
- remote independent verification;
- local copied-back verification;
- failed artifact retention; and
- no date parsing from opaque run IDs.

- [x] **Step 2: Run RED**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp1_gate.py \
  -k 'remote_runner' -q
```

Expected: failure because the runner is absent.

- [x] **Step 3: Implement the bounded serial runner**

The runner:

1. exports the required Kerberos cache;
2. verifies the checkpoint manifest and config;
3. selects one GPU with sufficient free bytes;
4. creates a source tar from the exact source inventory;
5. uploads source and inputs serially;
6. runs the remote campaign with the remote Python:

```text
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
```

7. runs the remote independent verifier;
8. copies the artifact back;
9. runs the local independent verifier against the copied source identity; and
10. updates `last_completed_run_path.txt` only after all verification passes.

- [x] **Step 4: Run local shell and contract GREEN**

Run:

```bash
bash -n tools/run_qwen35_generic_speculative_tp1_gate_remote.sh
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_qwen35_generic_speculative_tp1_gate.py -q
```

Expected: syntax pass and all tests pass.

- [x] **Step 5: Run the real remote campaign**

Run:

```bash
tools/run_qwen35_generic_speculative_tp1_gate_remote.sh
```

Expected:

- four cells complete;
- independent remote verifier returns
  `SECOND_MODEL_TP1_ESTABLISHED`;
- copied-back local verifier returns the same classification;
- baseline/candidate outputs match for batch 1 and 4;
- proposed, accepted, and rejected totals are positive;
- no accepted-prefix replay is recorded;
- no hybrid-state lease remains;
- runtime is not poisoned; and
- cleanup passes.

If the first real run exposes a correctness failure, preserve the failed
artifact, return to the smallest responsible task, add a focused RED, and do
not weaken the verifier.

- [x] **Step 6: Re-run the authoritative verifier explicitly**

Run:

```bash
run_path="$(
  cat artifacts/qwen35_generic_speculative_tp1/last_completed_run_path.txt
)"
PYTHONPATH=$PWD python3 \
  tools/verify_qwen35_generic_speculative_tp1_gate.py \
  "$run_path"
```

Expected:

```text
classification=SECOND_MODEL_TP1_ESTABLISHED
```

- [x] **Step 7: Review checkpoint**

Run:

```bash
git diff --check -- \
  tools/run_qwen35_generic_speculative_tp1_gate_remote.sh \
  tools/test_qwen35_generic_speculative_tp1_gate.py
```

Expected: success.

---

### Task 11: Final Regression, Audit, and Handoff Update

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes the independently verified Task 10 authority.
- Produces an honest objective status; it must remain not promotable until
  TP4 and long-context/performance gates exist for the second architecture.

- [x] **Step 1: Run the focused full regression**

Run:

```bash
PYTHONPATH=$PWD python3 -m pytest \
  tools/test_speculative_side_state.py \
  tools/test_speculative_batch_runtime.py \
  tools/test_qwen35_gated_delta_prefix_trace.py \
  tools/test_qwen35_linear_attention_shell.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_qwen35_prepared_model_step.py \
  tools/test_qwen35_speculative_state.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_speculative_side_state_engine_publication.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_qwen35_generic_speculative_tp1_gate.py \
  tools/test_generic_speculative_tp4_gate.py \
  tools/test_ngram_speculative.py -q
```

Expected: all tests pass.

- [x] **Step 2: Compile every changed Python file**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/speculative_side_state.py \
  tinyvllm/engine/qwen35_speculative_state.py \
  tinyvllm/speculative/batch_runtime.py \
  tinyvllm/layers/gated_delta.py \
  tinyvllm/layers/qwen35_linear_attention.py \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/models/qwen35_packed.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_generic_speculative_tp1_gate.py \
  tools/qwen35_generic_speculative_tp1_worker.py \
  tools/verify_qwen35_generic_speculative_tp1_gate.py
```

Expected: success.

- [x] **Step 3: Audit the real artifact against the design**

Create a checklist in the audit document mapping:

- real Qwen3.5 architecture identity;
- TP1 baseline/ngram cells;
- batch 1/4;
- 4K-class exact token counts;
- exact greedy parity;
- positive proposal/accept/reject coverage;
- first-target and verify callbacks;
- direct accepted KV commit;
- rejected KV rollback;
- recurrent-state consumed-input selection;
- reversible apply/seal receipts;
- no accepted-prefix replay;
- lease cleanup;
- Engine cleanup;
- source/model identity; and
- independent verification.

Mark only this narrow second-model TP1 row `ESTABLISHED`.

Keep:

```text
second-model TP4: MISSING
second-model 16K/32K: MISSING
second-model performance: MISSING
Phase 1: NOT_PROMOTABLE
```

- [x] **Step 4: Update handoff with exact evidence**

Record:

- authority path;
- result SHA-256;
- source-tree SHA-256;
- model manifest SHA-256;
- selected GPU;
- exact output parity;
- proposal/accept/reject counts;
- side-state checkpoint counts;
- no-replay evidence;
- cleanup status;
- test count; and
- the next ordered gate: Qwen3.5 TP4 speculative transaction authority.

- [x] **Step 5: Run final hygiene**

Run:

```bash
git diff --check
bash -n tools/run_qwen35_generic_speculative_tp1_gate_remote.sh
```

Expected: success.

- [x] **Step 6: Completion boundary**

Report:

```text
Qwen3.5 generic speculative TP1 transactional correctness:
  ESTABLISHED only if the real artifact independently verifies

second-model TP4:
  NOT ESTABLISHED

Phase 1:
  NOT_PROMOTABLE
```

Do not stage, commit, or push.

---

## Plan Self-Review Checklist

- [x] Every behavior task starts with a focused failing test.
- [x] Every production change has a focused GREEN command.
- [x] Standard Qwen3 no-provider behavior is explicitly regressed.
- [x] `committed_input_count` uses consumed inputs, not emitted outputs.
- [x] Qwen3.5 first-target state is included in the transaction.
- [x] Tail verification starts from first-target candidate state.
- [x] Prefix checkpoints are produced before acceptance, not replayed after
  acceptance.
- [x] Live hybrid state is unchanged until reversible apply.
- [x] Applied state can roll back until seal.
- [x] No tensor crosses Engine-owned callback receipts.
- [x] Existing KV transaction remains the full-attention KV authority.
- [x] The real gate uses the approved checkpoint and real `LLMEngine.step()`.
- [x] Candidate evidence includes both acceptance and rejection.
- [x] Failed remote artifacts are retained.
- [x] The final classification remains narrow and non-promotable.
