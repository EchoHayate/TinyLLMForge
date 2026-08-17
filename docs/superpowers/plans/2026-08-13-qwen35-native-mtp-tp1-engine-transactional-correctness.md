# Qwen3.5 Native MTP TP1 Engine Transactional Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session. Subagents are prohibited. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Prove that the real Qwen3.5 native MTP checkpoint executor runs
through the production TP1 `LLMEngine.step()` generic speculative pipeline,
publishes target KV, recurrent side state, and MTP proposal KV transactionally,
and releases all learned-executor request state.

**Architecture:** Keep runtime activation explicit. Add one generic
ModelRunner-executor sequence-release lifecycle operation, then build an
independent baseline/native-MTP TP1/4K authority with exact greedy parity,
real transaction receipts, source/model binding, remote execution, and an
independent verifier.

**Tech Stack:** Python 3, PyTorch, CUDA, dataclasses, existing TinyLLMForge
Engine/ModelRunner/speculative runtime, JSON authority artifacts, pytest,
Bash, SSH/rsync.

## Global Constraints

- Modify files only under
  `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, push, switch branches/worktrees, stash, reset, or
  clean.
- Do not use subagents.
- Preserve unrelated modified and untracked files.
- Follow strict RED -> minimal GREEN -> focused regression for every behavior
  change.
- Use the real Qwen3.5 checkpoint and real CUDA Engine for authority.
- Do not modify or parameterize frozen TP1/TP4/4K/16K/32K or performance
  authorities.
- First authority is TP1, 4K, batch 1/4, greedy, eager native MTP, and target
  KV offload disabled.
- Runtime activation remains explicit.
- No accepted-prefix target replay, copy, or rematerialization.
- The new release lifecycle is source-neutral.
- Remote host is `sitian@10.232.195.203`.
- Kerberos cache is
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- SSH uses `ControlMaster=no`, `ControlPath=none`, and finite serial retries.
- Never kill unrelated GPU processes.
- A PASS does not establish TP4, KV offload, 16K/32K native-MTP behavior,
  performance, second learned architecture, KV8/KV4, production readiness, or
  Phase 1 completion.

---

## File Structure

### Production modifications

- `tinyvllm/engine/speculative_proposal_executor.py`
  - Adds generic executor/registry sequence release.
- `tinyvllm/engine/speculative_model_runner.py`
  - Adds tensor-free ModelRunner release bridge.
- `tinyvllm/engine/model_runner.py`
  - Exposes the release command and cleanup snapshot fields.
- `tinyvllm/engine/llm_engine.py`
  - Releases finished ModelRunner-executor sequences after successful
    publication.

### New gate files

- `tools/qwen35_native_mtp_tp1_4k_engine_gate.py`
  - Owns schema, validators, aggregation, source/model binding, and atomic
    authority publication.
- `tools/qwen35_native_mtp_tp1_4k_engine_worker.py`
  - Runs one baseline or native-MTP Engine cell.
- `tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py`
  - Independently recomputes the authority.
- `tools/run_qwen35_native_mtp_tp1_4k_engine_gate_remote.sh`
  - Runs the isolated real-checkpoint campaign.

### Tests

- `tools/test_model_runner_proposal_executor.py`
- `tools/test_speculative_model_runner_callbacks.py`
- `tools/test_engine_speculative_runtime.py`
- `tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py`

### Documentation

- `AGENT_HANDOFF_STATE.md`

---

### Task 1: Add Generic ModelRunner Executor Sequence Release

**Files:**
- Modify: `tinyvllm/engine/speculative_proposal_executor.py`
- Modify: `tools/test_model_runner_proposal_executor.py`

**Interfaces:**
- Consumes:
  `ProposalExecutor.release_sequence(sequence_id, *, sequence_epoch)`.
- Produces:
  `ModelRunnerProposalExecutorRegistry.release_sequence(...)`.

- [x] **Step 1: Write the failing registry tests**

Add a lifecycle executor fixture whose release method records:

```python
("release", sequence_id, sequence_epoch)
```

Cover:

```python
registry.release_sequence(
    "native",
    7,
    3,
    capabilities,
)
assert executor.events[-1] == ("release", 7, 3)
```

Also require rejection of:

- empty executor ID;
- unknown executor;
- negative or boolean sequence ID;
- negative or boolean sequence epoch;
- mismatched capabilities;
- non-lifecycle executors;
- missing/non-callable executor release method; and
- non-`None` release acknowledgement.

- [ ] **Step 2: Run RED**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_model_runner_proposal_executor.py \
  -k release_sequence
```

Expected: FAIL because the registry method is absent.

- [x] **Step 3: Implement the minimal registry operation**

Extend the protocol:

```python
def release_sequence(
    self,
    sequence_id: int,
    *,
    sequence_epoch: int,
) -> None:
    ...
```

Implement a registry method that validates identity, lifecycle capability,
integers, callable presence, and `None` acknowledgement before returning.

- [x] **Step 4: Run GREEN and focused regression**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_model_runner_proposal_executor.py
python3 -m py_compile \
  tinyvllm/engine/speculative_proposal_executor.py
git diff --check -- \
  tinyvllm/engine/speculative_proposal_executor.py \
  tools/test_model_runner_proposal_executor.py
```

Expected: all commands exit zero.

---

### Task 2: Bridge Release Through ModelRunner and Engine

**Files:**
- Modify: `tinyvllm/engine/speculative_model_runner.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_speculative_model_runner_callbacks.py`
- Modify: `tools/test_engine_speculative_runtime.py`

**Interfaces:**
- Produces:
  `release_model_runner_proposal_sequence(model_runner, descriptor,
  sequence_id, sequence_epoch)`.
- Produces:
  `ModelRunner.release_speculative_proposal_sequence(...)`.
- Engine consumes the bridge only after successful prepared publication.

- [x] **Step 1: Write the failing bridge tests**

Require the bridge to dispatch:

```python
model_runner.call(
    "release_speculative_proposal_sequence",
    "native_checkpoint_proposal",
    7,
    0,
)
```

Require tensor-free inputs and a `None` acknowledgement. Cover invalid
descriptor, IDs, epochs, and non-`None` acknowledgements.

- [x] **Step 2: Write the failing Engine ordering tests**

Build a prepared native-MTP runtime fixture with one finished and one active
sequence. Record:

```text
prepare finalize
side apply
KV commit
Scheduler commit
finalize commit
side seal
release finished sequence
```

Assert:

- only finished rows are released;
- release uses the row's `sequence_epoch`, defaulting to zero;
- release is after finalize commit and side-state seal;
- release failure poisons the runtime and propagates;
- baseline/host adapters do not call the ModelRunner release bridge.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py \
  -k "release and proposal"
```

Expected: failures because the bridge, command, and Engine wiring are absent.

- [x] **Step 4: Implement the minimal bridge and command**

Add the source-neutral bridge to
`tinyvllm/engine/speculative_model_runner.py`.

Add to `ModelRunner`:

```python
def release_speculative_proposal_sequence(
    self,
    executor_id: str,
    sequence_id: int,
    sequence_epoch: int,
) -> None:
    capabilities = (
        self.speculative_proposal_executors
        .capabilities_for(executor_id)
    )
    self.speculative_proposal_executors.release_sequence(
        executor_id,
        sequence_id,
        sequence_epoch,
        capabilities,
    )
```

- [x] **Step 5: Wire Engine release after successful publication**

For a runtime with `model_runner_executor`, release every finished sequence
after proposal finalize commit and side-state seal. Use:

```python
int(getattr(seq, "sequence_epoch", 0))
```

If release raises, set:

```python
engine.speculative_runtime_poisoned = True
engine.speculative_runtime_poison_reason = (
    "proposal executor sequence release failed: ..."
)
```

and propagate.

- [x] **Step 6: Run GREEN and broad runtime regression**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py \
  tools/test_qwen35_mtp_executor.py
python3 -m py_compile \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py
git diff --check -- \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_speculative_model_runner_callbacks.py \
  tools/test_engine_speculative_runtime.py
```

Expected: all commands exit zero.

---

### Task 3: Define the Independent Authority Schema

**Files:**
- Create: `tools/qwen35_native_mtp_tp1_4k_engine_gate.py`
- Create: `tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py`

**Interfaces:**
- Produces schema:
  `qwen35.native-mtp-tp1-4k-engine-transactional-correctness.v1`.
- Produces classification:
  `QWEN35_NATIVE_MTP_TP1_4K_ENGINE_ESTABLISHED`.
- Produces cell validators and `assemble_authority(...)`.

- [x] **Step 1: Write failing schema and validator tests**

Require exactly:

```python
POLICIES = ("baseline", "native_mtp")
BATCH_SIZES = (1, 4)
PROMPT_TOKENS = 4096
MAX_OUTPUT_TOKENS = 32
MAX_PROPOSAL_TOKENS = 4
WORLD_SIZE = 1
```

Test rejection for:

- missing/extra cells;
- wrong policy or batch;
- prompt/output count drift;
- invalid model/checkpoint identity;
- baseline speculative activity;
- missing native descriptor/module/store;
- absent proposals, accepts, rejects, target callbacks, finalize receipts, or
  side-state receipts;
- accepted-prefix replay;
- incomplete lifecycle ordering;
- leaked transactions/tickets/sequences/slots;
- runtime poison;
- incomplete cleanup;
- output parity mismatch;
- source/model digest mismatch; and
- forbidden promotion claims.

- [ ] **Step 2: Run RED**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
```

Expected: import failure because the gate module does not exist.

- [x] **Step 3: Implement validators and atomic publication**

Implement:

- deterministic JSON SHA-256;
- source-file tree digest;
- checkpoint manifest binding;
- exact token-row parity;
- ordered receipt validation;
- aggregate learned-token validation;
- cleanup validation;
- atomic `result.json`, `source_manifest.json`, and `status.json`
  publication.

- [x] **Step 4: Run GREEN**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
python3 -m py_compile \
  tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
git diff --check -- \
  tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
```

Expected: all commands exit zero.

---

### Task 4: Implement the Real Engine Cell Worker

**Files:**
- Create: `tools/qwen35_native_mtp_tp1_4k_engine_worker.py`
- Modify: `tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py`

**Interfaces:**
- Consumes one model path, GPU index, policy, and batch size.
- Produces one validated cell JSON.

- [x] **Step 1: Write failing worker contract tests**

Test that the native cell:

- passes `qwen35_mtp_enabled=True`;
- passes `qwen35_mtp_cuda_graphs=False`;
- activates `EngineSpeculativeRuntime(model_runner_executor=descriptor)`;
- rejects missing registration/error/identity;
- records all proposal lifecycle calls;
- records side-state calls;
- checks exact output length;
- snapshots executor/store state before exit; and
- always calls `engine.exit()` in `finally`.

Test that baseline sets `qwen35_mtp_enabled=False` and does not activate a
speculative runtime.

- [ ] **Step 2: Run RED**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py \
  -k worker
```

Expected: FAIL because the worker module is absent.

- [x] **Step 3: Implement deterministic prompt construction**

Build exact 4,096-token rows from tokenizer-valid deterministic token
patterns. Store:

- prompt index;
- exact token IDs;
- token count; and
- SHA-256.

Each batch row must differ while remaining reproducible.

- [x] **Step 4: Implement cell execution**

Construct:

```python
engine = LLM(
    model_path,
    tensor_parallel_size=1,
    enforce_eager=True,
    max_model_len=8192,
    max_num_batched_tokens=16384,
    max_num_prefill_tokens_per_step=1024,
    max_num_seqs=batch_size,
    kv_offload_mvp0=False,
    qwen35_mtp_enabled=(policy == "native_mtp"),
    qwen35_mtp_cuda_graphs=False,
    qwen35_mtp_max_proposal_tokens=4,
)
```

For native MTP:

```python
descriptor = (
    engine.model_runner.qwen35_mtp_executor_descriptor
)
engine.activate_speculative_runtime(
    EngineSpeculativeRuntime(
        model_runner_executor=descriptor,
    )
)
```

Capture ModelRunner call receipts for:

```text
observe_speculative_target_prefill_batch
prepare_speculative_proposal_finalize_batch
commit_speculative_proposal_finalize_batch
rollback_speculative_proposal_finalize_batch
release_speculative_proposal_sequence
prepare/select/apply/seal/rollback speculative side state
```

Accumulate Engine step observations until all requests finish.

- [x] **Step 5: Snapshot zero-leak state**

Before Engine exit, record:

```python
{
    "pending_prefix_count": len(executor._pending_prefixes),
    "bootstrapped_sequence_count": len(executor._bootstrapped),
    "proposal_transaction_count": len(
        executor._proposal_transactions
    ),
    "batch_ticket_count": len(executor._batch_tickets),
    "batch_ticket_transaction_count": len(
        executor._batch_ticket_transactions
    ),
    "allocated_physical_slot_count": len(
        store._allocated_slot_ids
    ),
}
```

Every count must be zero in a successful native-MTP cell.

- [x] **Step 6: Run GREEN**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
python3 -m py_compile \
  tools/qwen35_native_mtp_tp1_4k_engine_worker.py
git diff --check -- \
  tools/qwen35_native_mtp_tp1_4k_engine_worker.py \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
```

Expected: all commands exit zero.

---

### Task 5: Add Independent Verifier and Remote Runner

**Files:**
- Create: `tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py`
- Create: `tools/run_qwen35_native_mtp_tp1_4k_engine_gate_remote.sh`
- Modify: `tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py`

**Interfaces:**
- Verifier consumes authority directory and source root.
- Runner produces isolated remote and local authority directories.

- [x] **Step 1: Write failing verifier/runner contract tests**

Require the verifier to reject:

- mutated result JSON;
- changed source files;
- changed checkpoint/model digest;
- missing cells;
- parity drift;
- lifecycle drift;
- leak-state drift; and
- digest mismatch.

Require the runner source to contain:

```text
sitian@10.232.195.203
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
finite retry loops
unique remote root
pre/post GPU inventory
remote verifier
local authority download
```

- [ ] **Step 2: Run RED**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py \
  -k "verifier or runner"
```

Expected: failures because verifier and runner are absent.

- [x] **Step 3: Implement independent verifier**

The verifier must import validation logic but independently read and recompute
all digests and comparisons from disk. It prints only:

```json
{"classification":"PASS","failures":[]}
```

on success.

- [x] **Step 4: Implement remote runner**

Run cells serially, with fresh Engine processes:

```text
baseline:b1
native_mtp:b1
baseline:b4
native_mtp:b4
```

Reject any pre/post admission change or unexpected GPU process. Preserve
failed run roots. Never kill unrelated processes.

- [x] **Step 5: Run GREEN and static validation**

Run:

```bash
uv run --offline --with pytest pytest -q \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
python3 -m py_compile \
  tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/qwen35_native_mtp_tp1_4k_engine_worker.py \
  tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py
bash -n \
  tools/run_qwen35_native_mtp_tp1_4k_engine_gate_remote.sh
git diff --check -- \
  tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/qwen35_native_mtp_tp1_4k_engine_worker.py \
  tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/run_qwen35_native_mtp_tp1_4k_engine_gate_remote.sh \
  tools/test_qwen35_native_mtp_tp1_4k_engine_gate.py
```

Expected: all commands exit zero.

---

### Task 6: Run Real Authority and Update Handoff

**Files:**
- Generate:
  `artifacts/qwen35_native_mtp_tp1_4k_engine/<opaque-run-id>/...`
- Modify: `AGENT_HANDOFF_STATE.md`

- [x] **Step 1: Run complete local regression**

Run the focused release, runtime, MTP executor, and gate tests in one fresh
process. Record the exact pass count.

- [x] **Step 2: Run the remote campaign**

Execute:

```bash
bash tools/run_qwen35_native_mtp_tp1_4k_engine_gate_remote.sh
```

Poll the owned campaign only. If GPU admission changes, preserve the failed
run and restart on a clean allowed GPU; do not terminate unrelated work.

- [x] **Step 3: Run independent verification**

Run the verifier remotely and locally against the downloaded authority.
Require `PASS` from both.

- [x] **Step 4: Audit objective coverage**

Confirm the authority proves:

- real learned `mtp.*` checkpoint source;
- real `LLMEngine.step()` activation;
- batch 1 and 4;
- 4K prompts;
- exact greedy parity;
- accepts and rejects;
- target first/tail callbacks;
- target KV and recurrent side-state publication;
- MTP finalize commit;
- no accepted replay;
- finished-sequence release;
- zero leaked MTP state/slots;
- complete Engine cleanup; and
- source/model binding.

- [x] **Step 5: Update handoff**

Append:

- release-lifecycle root cause and implementation;
- failed runs and their exact reasons;
- successful authority path and digests;
- proposal/acceptance/rollback counts;
- parity and cleanup evidence;
- remote/local verifier results; and
- explicit `NOT_PROMOTABLE` limitations.

- [x] **Step 6: Final fresh checks**

Run:

```bash
git diff --check
python3 -m py_compile \
  tinyvllm/engine/speculative_proposal_executor.py \
  tinyvllm/engine/speculative_model_runner.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_native_mtp_tp1_4k_engine_gate.py \
  tools/qwen35_native_mtp_tp1_4k_engine_worker.py \
  tools/verify_qwen35_native_mtp_tp1_4k_engine_gate.py
bash -n \
  tools/run_qwen35_native_mtp_tp1_4k_engine_gate_remote.sh
```

Expected: all commands exit zero.

Do not mark the three-direction objective or Phase 1 complete after this gate.
