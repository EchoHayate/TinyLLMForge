# Qwen3.5 MTP Exact-Q CUDA Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. The user
> requires inline execution; do not dispatch subagents. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Install a real TP1, no-KV-offload CUDA Graph backend for Qwen3.5
native MTP exact-Q proposal families Q2/Q3/Q4 and exact batches 1/4 while
preserving transactional proposal-KV finalization.

**Architecture:** Retain `Qwen35MTPExactGraphRunner` as the admission,
observation, budget, and quarantine policy. Add focused production backend and
scratch-owner modules that capture batched GPU argmax chains over static
tensors, prepare live proposal transactions before replay, and return ordinary
`DraftProposal` records for executor-owned registration and finalization.

**Tech Stack:** Python 3.10+, PyTorch CUDA Graphs, FlashAttention KV-cache
decode, dataclasses, SHA-256 graph identities, pytest, source-contract tests,
and remote A100 real-checkpoint gates.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, run `git clean`,
  or create a worktree.
- Do not use subagents.
- Use `apply_patch` for every file edit.
- Follow strict RED -> GREEN -> regression for each behavior.
- The first version supports TP1 only.
- The first version requires KV offload disabled.
- The first version is greedy only and supports one MTP layer.
- Preserve shared embedding and LM-head identity.
- Real graph families cover only exact Q `(2, 3, 4)`.
- Real graph batches cover only exact batch sizes `(1, 4)`.
- Q1 remains an eager no-MTP-forward passthrough.
- Never pad or merge distinct Q or batch families.
- Accepted proposal KV commits in place.
- Rejected proposal suffix rolls back in place.
- Never replay, copy, or rematerialize accepted proposal KV.
- Pre-replay graph preparation failures may fall back to one eager execution
  only after all graph-created live transactions are aborted.
- After `graph.replay()` starts, never retry eager.
- Generic engine, scheduler, verifier, residency, target-KV, and generic
  speculative runtime code must not gain Qwen/MTP source dispatch.
- Keep the repository classification `NOT_PROMOTABLE`.
- Treat future-looking remote run tokens as opaque IDs, not timestamps.

---

## File Map

### New Files

- `tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py`
  - static entry tensors, byte estimation, capture, replay preflight, batched
    GPU argmax chaining, live transaction cleanup, and proposal assembly.
- `tinyvllm/engine/qwen35_mtp_graph_scratch.py`
  - capture-only transaction leases and rollback.
- `tools/test_qwen35_mtp_cuda_graph_backend.py`
  - dependency-light backend shape, preflight, replay, and source-contract
    tests.
- `tools/test_qwen35_mtp_graph_scratch.py`
  - scratch isolation and rollback tests.
- `tools/qwen35_mtp_cuda_graph_smoke.py`
  - CUDA synthetic eager-versus-graph and injected replay-failure smoke.

### Modified Files

- `tinyvllm/engine/qwen35_mtp_graph.py`
  - explicit pre-replay fallback exception handling while preserving
    replay-started hard failure.
- `tinyvllm/engine/qwen35_mtp_executor.py`
  - group-level proposal transaction registration shared by eager and graph.
- `tinyvllm/engine/model_runner.py`
  - construct the production backend, scratch owner, and graph runner.
- `tools/test_qwen35_mtp_graph.py`
  - policy contract for pre-replay fallback versus replay hard failure.
- `tools/test_qwen35_mtp_executor.py`
  - graph-result registration and Q1 passthrough tests.
- `tools/test_qwen35_mtp_model_runner_integration.py`
  - production builder installation and disabled-path tests.
- `tools/qwen35_mtp_real_checkpoint_gate.py`
  - graph capture/replay/eager parity and failure-boundary artifact fields.
- `tools/test_qwen35_mtp_real_checkpoint_gate.py`
  - fail-closed graph artifact contract.
- `tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh`
  - enable exact-Q graph families `(2,3,4)` and batches `(1,4)` on GPU7.
- `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
  - replace the graph blocker only when real evidence passes.
- `AGENT_HANDOFF_STATE.md`
  - commands, evidence, limitations, and next blocker.

---

### Task 1: Pre-Replay Fallback Contract

**Files:**
- Modify: `tools/test_qwen35_mtp_graph.py`
- Modify: `tinyvllm/engine/qwen35_mtp_graph.py`

**Interfaces:**
- Produces:

```python
class Qwen35MTPGraphPreReplayError(RuntimeError):
    """Replay was not launched and backend-created live state is clean."""
```

- `capture_backend.replay(entry, rows)` may raise
  `Qwen35MTPGraphPreReplayError` only before calling `graph.replay()`.

- [ ] **Step 1: Write the failing policy tests**

Add tests with a ready fake entry whose backend:

```python
def replay(self, entry, rows):
    raise Qwen35MTPGraphPreReplayError("static input mismatch")
```

Assert:

```python
result == eager_result
eager_calls == [(2, rows)]
runner.quarantine_reason(identity) is None
runner.counters["replays"] == 0
```

Keep the existing generic replay exception test and assert:

```python
with pytest.raises(Qwen35MTPGraphReplayError):
    runner.run(...)
assert eager_calls == []
assert runner.quarantine_reason(identity) == "replay_failed"
```

- [ ] **Step 2: Run RED**

Run:

```bash
python -m pytest tools/test_qwen35_mtp_graph.py -q
```

Expected: the new import or fallback assertion fails because no explicit
pre-replay exception exists.

- [ ] **Step 3: Implement the minimal policy**

Add the exception and narrow handling:

```python
try:
    result = self.capture_backend.replay(entry, rows)
except Qwen35MTPGraphPreReplayError:
    self.counters["fallback_pre_replay"] += 1
    return eager(exact_q, rows)
except BaseException as error:
    self._quarantine(identity, "replay_failed")
    raise Qwen35MTPGraphReplayError(identity, error) from error
```

- [ ] **Step 4: Run GREEN**

Run the same command. Expected: all graph policy tests pass.

---

### Task 2: Capture Scratch Isolation

**Files:**
- Create: `tools/test_qwen35_mtp_graph_scratch.py`
- Create: `tinyvllm/engine/qwen35_mtp_graph_scratch.py`

**Interfaces:**
- Consumes:

```python
Qwen35MTPGraphIdentity
ProposalKVCache.begin(sequence_id, sequence_epoch, staged_entry_count)
ProposalKVCache.abort(transaction_id)
```

- Produces:

```python
@dataclass
class Qwen35MTPGraphScratchRow:
    input_row: ModelRunnerProposalInput
    bootstrap: object
    transaction: object


@dataclass
class Qwen35MTPGraphScratchLease:
    identity: Qwen35MTPGraphIdentity
    rows: tuple[Qwen35MTPGraphScratchRow, ...]
    rolled_back: bool = False


class Qwen35MTPGraphScratchOwner:
    def __init__(
        self,
        *,
        live_cache: ProposalKVCache,
        scratch_cache: ProposalKVCache,
    ):
        ...

    def acquire(
        self,
        identity: Qwen35MTPGraphIdentity,
        rows: tuple,
    ) -> Qwen35MTPGraphScratchLease:
        ...

    def rollback(self, lease: Qwen35MTPGraphScratchLease) -> None:
        ...
```

- [ ] **Step 1: Write RED tests**

Test:

1. Q3/B4 acquires four transactions with two staged entries each from a
   dedicated scratch cache.
2. acquisition failure on row three aborts rows one and two before raising.
3. rollback aborts all reserved/materialized transactions exactly once.
4. double rollback raises.
5. lease rows retain source row ordering.
6. scratch owner has no reference to executor `_proposal_transactions`.
7. an active live transaction on every source sequence does not block scratch
   acquisition.
8. synthetic scratch sequence IDs are positive, private, and released after
   rollback.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tools/test_qwen35_mtp_graph_scratch.py -q
```

Expected: collection fails because the scratch module does not exist.

- [ ] **Step 3: Implement minimal scratch ownership**

Construct `scratch_cache = ProposalKVCache(live_cache.physical_store)` once;
do not reuse `live_cache` as the scratch transaction namespace. Use
`scratch_cache.begin()` with owner-private synthetic sequence IDs for each row
and abort already-created transactions in reverse order on partial failure.
Read source committed slots from `live_cache` without mutation. Validate:

```python
identity.exact_q >= 2
len(rows) == identity.exact_batch_size
staged_entry_count == identity.exact_q - 1
```

Rollback must inspect each transaction state and abort only
`"reserved"`/`"materialized"` entries.

- [ ] **Step 4: Run GREEN**

Run the focused test. Expected: all scratch tests pass.

---

### Task 3: Static Layout and Live Replay Preparation

**Files:**
- Create: `tools/test_qwen35_mtp_cuda_graph_backend.py`
- Create: `tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py`

**Interfaces:**
- Consumes:

```python
Qwen35MTPGraphIdentity
Qwen35MTPGraphEntry
Qwen35MTPGraphPreReplayError
ProposalKVCache
DraftProposal
temporary_context
```

- Produces:

```python
@dataclass
class Qwen35MTPCudaGraphTensors:
    first_tokens: torch.Tensor
    current_tokens: torch.Tensor
    positions: torch.Tensor
    initial_hidden: torch.Tensor
    current_hidden: torch.Tensor
    next_hidden: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    proposal_tokens: torch.Tensor


@dataclass
class Qwen35MTPCudaGraphPayload:
    graph: object
    tensors: Qwen35MTPCudaGraphTensors


class Qwen35MTPCudaGraphBackend:
    def estimate_static_bytes(self, identity, rows) -> int:
        ...

    def capture(
        self,
        identity,
        rows,
        eager,
        scratch_lease,
    ) -> Qwen35MTPGraphEntry:
        ...

    def replay(self, entry, rows) -> tuple[DraftProposal, ...]:
        ...
```

- [ ] **Step 1: Write static-layout RED tests**

For every `(B, Q)` in:

```python
((1, 2), (1, 3), (1, 4), (4, 2), (4, 3), (4, 4))
```

assert exact shapes and dtypes from the design. Assert estimated bytes equal
the sum of tensor element counts times element sizes.

- [ ] **Step 2: Write preparation RED tests**

Using CPU fake tensors and a fake proposal cache, assert:

- exact B/Q/device/dtype/hidden mismatches raise
  `Qwen35MTPGraphPreReplayError`;
- each row receives exactly `Q - 1` staged slots;
- step `s` block tables contain committed slots plus staged slots through
  `s`;
- unused block-table columns are zero;
- visible-table overflow aborts every begun transaction;
- copy failure aborts every begun transaction;
- no transaction ID is published to the executor.

- [ ] **Step 3: Run RED**

```bash
python -m pytest tools/test_qwen35_mtp_cuda_graph_backend.py -q
```

Expected: collection fails because the backend module does not exist.

- [ ] **Step 4: Implement layout and preparation only**

Add private methods:

```python
def _allocate_tensors(self, identity) -> Qwen35MTPCudaGraphTensors:
    ...

def _prepare_live_replay(
    self,
    identity,
    tensors,
    rows,
) -> tuple[object, ...]:
    ...

def _abort_transactions(self, transactions) -> None:
    ...
```

Do not implement CUDA capture yet. Keep replay raising an explicit
`NotImplementedError` after successful preparation so tests cannot
accidentally claim graph execution.

- [ ] **Step 5: Run GREEN for layout/preparation**

Run the focused tests selected by `-k "layout or prepare or overflow"`.
Expected: those tests pass and the deliberate replay test remains RED.

---

### Task 4: Batched GPU Argmax Capture

**Files:**
- Modify: `tools/test_qwen35_mtp_cuda_graph_backend.py`
- Modify: `tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py`
- Create: `tools/qwen35_mtp_cuda_graph_smoke.py`

**Interfaces:**
- Produces graph payloads whose recorded body performs:

```python
for step in range(identity.exact_q - 1):
    with temporary_context(...static step tensors...):
        next_hidden, logits = module.forward_step(
            tensors.current_tokens,
            tensors.positions[step],
            tensors.current_hidden,
        )
    next_tokens = torch.argmax(logits, dim=-1)
    tensors.proposal_tokens[:, step + 1].copy_(next_tokens)
    tensors.current_tokens.copy_(next_tokens)
    tensors.current_hidden.copy_(next_hidden)
```

- [ ] **Step 1: Write source and fake-capture RED tests**

Assert:

- backend capture/replay source contains no `.item(`;
- exactly `Q - 1` module forwards occur during fake capture;
- every step uses batched `[B]` tokens and `[B, H]` hidden states;
- graph output column zero is copied from first target tokens;
- generic attention backend is selected only inside the scoped graph call and
  restored on failure.

- [ ] **Step 2: Write CUDA smoke RED**

The smoke script must:

1. skip with a clear non-evidence message when CUDA is unavailable;
2. construct a minimal synthetic module using CUDA tensor operations;
3. warm/capture Q2/Q3/Q4 for B1 and B4;
4. compare graph proposal tokens with eager batched GPU execution;
5. inject `graph.replay()` failure and record zero eager retries.

- [ ] **Step 3: Run RED**

```bash
python -m pytest tools/test_qwen35_mtp_cuda_graph_backend.py -q
python tools/qwen35_mtp_cuda_graph_smoke.py
```

Expected: capture tests fail because capture is not implemented. Local smoke
may skip if CUDA is absent.

- [ ] **Step 4: Implement minimal capture**

Use:

```python
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    self._run_static_chain(identity, tensors)
```

Measure:

```python
capture_duration_ns
allocated_delta_bytes
reserved_delta_bytes
static_bytes
```

Store `Qwen35MTPCudaGraphPayload` in `Qwen35MTPGraphEntry.graph`.

- [ ] **Step 5: Run GREEN**

Run the backend tests and smoke. Expected: dependency-light tests pass; CUDA
smoke passes only on a CUDA host.

---

### Task 5: Replay, Materialization, and Proposal Assembly

**Files:**
- Modify: `tools/test_qwen35_mtp_cuda_graph_backend.py`
- Modify: `tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py`

**Interfaces:**
- `replay()` returns:

```python
DraftProposal(
    sequence_id=input_row.sequence_id,
    token_ids=tuple(int(token) for token in output_row),
    source_type="native_model_runner",
    metadata={
        "exact_q": identity.exact_q,
        "staged_entry_count": identity.exact_q - 1,
        "execution_mode": "cuda_graph",
    },
    proposal_transaction_id=transaction.transaction_id,
)
```

- [ ] **Step 1: Write replay RED tests**

Assert:

1. preflight failure calls no graph replay and falls back through the policy
   test from Task 1;
2. successful replay calls `mark_materialized(transaction, Q - 1)` for every
   row;
3. proposals preserve row order, sequence IDs, exact Q, and transaction IDs;
4. replay exception aborts still-abortable transactions and re-raises;
5. output validation failure after replay is a hard failure;
6. no path invokes the eager callback after replay starts.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tools/test_qwen35_mtp_cuda_graph_backend.py \
  tools/test_qwen35_mtp_graph.py -q
```

Expected: replay-result tests fail.

- [ ] **Step 3: Implement minimal replay**

Set a local `replay_started = False`, call `_prepare_live_replay()`, then:

```python
replay_started = True
payload.graph.replay()
```

Only errors raised while `replay_started is False` may be converted to
`Qwen35MTPGraphPreReplayError`. Validate output after replay, materialize all
transactions, and assemble proposals.

- [ ] **Step 4: Run GREEN**

Run the focused command. Expected: all backend and graph-policy tests pass.

---

### Task 6: Executor Group Registration

**Files:**
- Modify: `tools/test_qwen35_mtp_executor.py`
- Modify: `tinyvllm/engine/qwen35_mtp_executor.py`

**Interfaces:**
- Produces:

```python
def _register_group_proposals(
    self,
    proposals,
    rows,
) -> tuple[DraftProposal, ...]:
    ...
```

- [ ] **Step 1: Write RED tests**

Assert:

- graph-produced transaction IDs become finalizable;
- eager-produced transaction IDs remain finalizable;
- duplicate transaction ID across rows fails;
- sequence or epoch mismatch fails;
- partial group registration publishes nothing;
- Q1 does not call the graph runner;
- graph group failure leaves no executor transaction registration.

- [ ] **Step 2: Run RED**

```bash
python -m pytest tools/test_qwen35_mtp_executor.py -q
```

Expected: graph-produced IDs are not registered because `_run_proposal()`
currently owns eager-only registration.

- [ ] **Step 3: Implement group registration**

Remove direct `_proposal_transactions` mutation from `_run_proposal()`.
Validate every proposal first, build a temporary registration dictionary,
then update `_proposal_transactions` once for the whole group.

Call the helper after either eager or graph execution and before scattering
proposals back to original input order.

- [ ] **Step 4: Run GREEN**

Run the focused executor test. Expected: all tests pass.

---

### Task 7: Production Builder Installation

**Files:**
- Modify: `tools/test_qwen35_mtp_model_runner_integration.py`
- Modify: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Consumes:

```python
Qwen35MTPGraphScratchOwner
Qwen35MTPCudaGraphBackend
Qwen35MTPExactGraphRunner
```

- [ ] **Step 1: Write builder RED tests**

Assert:

- disabled config returns `None`;
- enabled config returns `Qwen35MTPExactGraphRunner`;
- runner owns production backend and scratch owner classes;
- builder passes exact config budgets and identity fields;
- TP != 1 fails registration;
- KV offload enabled fails registration;
- MTP layer count != 1 fails registration;
- model-runner source contains no capture implementation beyond construction.

- [ ] **Step 2: Run RED**

```bash
python -m pytest \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

Expected: enabled registration fails with the current “backend is not
installed” exception.

- [ ] **Step 3: Implement construction**

Replace the placeholder exception with construction using:

```python
block_table_width=int(config.max_model_len)
device_index=module.fc.weight.device.index
compute_dtype=str(module.fc.weight.dtype)
hidden_size=int(config.hf_config.hidden_size)
mtp_layer_count=int(config.hf_config.mtp_num_hidden_layers)
```

Pass config allowlists and budgets verbatim. Do not alter unrelated whitespace
in `model_runner.py`.

- [ ] **Step 4: Run GREEN**

Run the focused integration command. Expected: production graph runner
installation tests pass.

---

### Task 8: Local Regression and Static Safety

**Files:**
- Modify only files required to fix failures caused by Tasks 1-7.

- [ ] **Step 1: Run focused graph stack**

```bash
python -m pytest \
  tools/test_qwen35_mtp_graph.py \
  tools/test_qwen35_mtp_graph_scratch.py \
  tools/test_qwen35_mtp_cuda_graph_backend.py \
  tools/test_qwen35_mtp_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py -q
```

- [ ] **Step 2: Run MTP transactional regressions**

```bash
python -m pytest \
  tools/test_qwen35_mtp.py \
  tools/test_qwen35_mtp_physical_kv.py \
  tools/test_qwen35_mtp_real_transaction_probe.py \
  tools/test_qwen35_mtp_real_eager_reference_probe.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

- [ ] **Step 3: Run generic source-neutral regressions**

```bash
python -m pytest \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_runtime.py \
  tools/test_speculative_kv_transaction.py -q
```

- [ ] **Step 4: Run syntax and source checks**

```bash
python -m py_compile \
  tinyvllm/engine/qwen35_mtp_graph.py \
  tinyvllm/engine/qwen35_mtp_graph_scratch.py \
  tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tinyvllm/engine/model_runner.py \
  tools/qwen35_mtp_cuda_graph_smoke.py
bash -n tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh
git diff --check -- \
  tinyvllm/engine/qwen35_mtp_graph.py \
  tinyvllm/engine/qwen35_mtp_graph_scratch.py \
  tinyvllm/engine/qwen35_mtp_cuda_graph_backend.py \
  tinyvllm/engine/qwen35_mtp_executor.py \
  tools/
```

Expected: all commands pass. Existing unrelated whitespace outside the scoped
paths is not modified.

---

### Task 9: Remote Real-Checkpoint Graph Gate

**Files:**
- Modify: `tools/qwen35_mtp_real_checkpoint_gate.py`
- Modify: `tools/test_qwen35_mtp_real_checkpoint_gate.py`
- Modify: `tools/run_qwen35_mtp_real_checkpoint_gate_remote.sh`

- [x] **Step 1: Write artifact-contract RED tests**

Require:

```text
graph_backend_installed=true
graph_capture_count > 0
graph_replay_count > 0
graph_eager_argmax_equal=true
graph_eager_proposal_tokens_equal=true
graph_transaction_commit=true
graph_transaction_rollback=true
replay_failure_quarantined=true
replay_failure_eager_retry_count=0
backend_failures excludes graph_eager
```

Corrupt each critical field and assert `FAIL / NOT_PROMOTABLE`.

- [x] **Step 2: Run RED**

```bash
python -m pytest tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

Expected: tests fail because graph evidence is not yet produced.

- [x] **Step 3: Extend the gate**

Exercise Q2/Q3/Q4 B1 and at least one B4 family with fresh sequence IDs.
Compare graph proposal tokens and argmax with fresh-sequence eager execution.
Inject one post-replay failure and verify zero eager retries.

- [x] **Step 4: Run local contract GREEN**

Run the focused gate tests. Expected: schema and corruption tests pass.

- [x] **Step 5: Run remote GPU7 gate serially**

Use:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh -o ControlMaster=no -o ControlPath=none \
  sitian@10.232.195.203 \
  'CUDA_VISIBLE_DEVICES=7 ...'
```

Do not create parallel SSH sessions. Poll the same foreground command or
existing session until it exits.

- [x] **Step 6: Download and verify artifact**

Run the local verifier/tests against the downloaded JSON. Expected:

```text
backend_failures == []
status reflects the broader gate truth
promotion_classification remains NOT_PROMOTABLE unless every independent
phase-one criterion is satisfied
```

Do not use the run-token date as ordering evidence.

---

### Task 10: Audit and Handoff

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Record exact evidence**

Write:

- local test counts and commands;
- remote device, PyTorch, CUDA, checkpoint SHA-256, and opaque run ID;
- graph capture/replay counts;
- eager/graph token parity;
- transaction commit/rollback results;
- injected replay failure and zero-retry result;
- remaining unsupported scope.

- [x] **Step 2: Preserve claim boundaries**

State explicitly that evidence does not establish TP4, KV offload, arbitrary
Q/B, second-model support, long-context behavior, or performance gains.

- [x] **Step 3: Final verification**

Run the focused and regression commands from Task 8 plus artifact contract
tests. Only then mark the relevant plan checkboxes complete.

Completion note: the local host has no `torch` installation, so the five
Task 8 files that import `torch` cannot collect locally. The dependency-light
eight-file subset passed with `211 passed`; the downloaded real-checkpoint
GPU artifact passed `validate_gate_report(...)`; syntax, wrapper, and scoped
diff checks passed. Torch/CUDA execution evidence is supplied by the
authoritative remote GPU7 artifact rather than a local rerun.

