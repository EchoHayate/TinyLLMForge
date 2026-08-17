# Loaded-GPU ModelRunner Proposal Ownership Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. This
> work must execute inline; subagents are prohibited for this worktree.

**Goal:** Produce a fail-closed real-checkpoint GPU artifact proving that
target hidden/logit CUDA tensors remain inside the production ModelRunner
fused proposal path while only tensor-free speculative rows cross its public
command boundary.

**Architecture:** Add a standalone gate rather than changing the existing
native-MTP graph artifact. The gate loads the production TP1 Qwen3.5
ModelRunner, creates fresh target KV/hybrid-state ownership per scenario,
temporarily observes the real target forward and real MTP executor without
retaining tensors, invokes
`ModelRunner.call("run_spec_first_target_and_proposal_batch", ...)`, and
compares fresh graph/eager fused results. A separate schema verifier and
remote GPU7 wrapper provide fail-closed authority.

**Tech Stack:** Python 3, PyTorch/CUDA, pytest, TinyLLMForge ModelRunner,
Qwen3.5 native MTP executor, JSON artifact validation, Bash/SSH.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not switch branches, stage, commit, stash, reset, push, clean, or create
  another worktree.
- Use `apply_patch` for every edit.
- Follow strict RED then GREEN for every behavior.
- No subagents.
- Current date is 2026-08-12.
- First version is TP1, KV offload disabled, greedy, one MTP layer,
  Q `(1,2,3,4)`, batch `(1,4)`.
- Q1 is eager passthrough; Q2/Q3/Q4 use exact graph families.
- The target forward must be real. Do not patch or replace `run_model`.
- Observers may record scalar metadata only and must not retain, clone,
  detach, hash, serialize, or copy hidden/logit tensors.
- Public fused results must contain zero tensors before and after a pickle
  round-trip.
- Graph and eager scenarios must use fresh sequence IDs, target KV blocks,
  hybrid-state leases, proposal KV state, and transactions.
- Preserve pre-replay eager fallback and post-replay no-eager-retry
  quarantine semantics.
- Remote execution uses only `sitian@10.232.195.203`,
  `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`,
  `ControlMaster=no`, `ControlPath=none`, GPU 7, and serial SSH.
- A PASS remains `NOT_PROMOTABLE`.

## File Map

- Create:
  `tools/qwen35_mtp_model_runner_ownership_gate.py` — schema, verifier,
  ownership observers, real scenario owner, fused graph/eager probe, artifact
  builder, CLI.
- Create:
  `tools/test_qwen35_mtp_model_runner_ownership_gate.py` — dependency-light
  RED/GREEN contract and corruption tests.
- Create:
  `tools/run_qwen35_mtp_model_runner_ownership_gate_remote.sh` — isolated
  source sync, serial GPU7 execution, artifact download.
- Modify:
  `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md` — append
  authoritative ownership result only after remote verification.
- Modify:
  `AGENT_HANDOFF_STATE.md` — append continuation evidence and next action.

---

### Task 1: Freeze the Artifact Schema and Fail-Closed Verifier

**Files:**
- Create: `tools/qwen35_mtp_model_runner_ownership_gate.py`
- Create: `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces `SCHEMA_VERSION = 1`.
- Produces `REQUIRED_Q_VALUES = (1, 2, 3, 4)`.
- Produces `REQUIRED_BATCH_SIZES = (1, 4)`.
- Produces `REQUIRED_REPORT_FIELDS`.
- Produces
  `validate_ownership_gate_report(report, *, required_q_values,
  required_batch_sizes) -> None`.

- [ ] **Step 1: Write schema RED tests**

Create tests importing the missing module and requiring:

```python
def test_valid_report_passes():
    report = _valid_report()
    validate_ownership_gate_report(
        report,
        required_q_values=REQUIRED_Q_VALUES,
        required_batch_sizes=REQUIRED_BATCH_SIZES,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fused_model_runner_path_exercised", False),
        ("target_forward_real", False),
        ("target_logits_cuda", False),
        ("target_hidden_cuda", False),
        ("target_hidden_consumed_by_real_executor", False),
        ("target_logits_not_passed_to_mtp_executor", False),
        ("public_result_tensor_count", 1),
        ("public_result_pickle_roundtrip", False),
        ("public_result_tensor_free", False),
        ("executor_identity_preserved", False),
        ("sequence_order_preserved", False),
        ("graph_eager_first_target_tokens_equal", False),
        ("graph_eager_proposal_tokens_equal", False),
        ("cleanup_passed", False),
        ("backend_failures", ["ownership"]),
        ("status", "FAIL"),
    ),
)
def test_critical_field_corruption_fails(field, value):
    report = _valid_report()
    report[field] = value
    with pytest.raises(ValueError):
        validate_ownership_gate_report(
            report,
            required_q_values=REQUIRED_Q_VALUES,
            required_batch_sizes=REQUIRED_BATCH_SIZES,
        )
```

Also require exact Q/batch domains, non-negative capture/replay counts,
`promotion_classification == "NOT_PROMOTABLE"`, explicit false coverage for
TP4/KV offload/long context/second model/performance, and a non-empty
limitations list.

- [ ] **Step 2: Run RED**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_mtp_model_runner_ownership_gate.py -q
```

Expected: collection fails because the gate module does not exist.

- [ ] **Step 3: Implement the minimal schema and verifier**

The module begins with:

```python
SCHEMA_VERSION = 1
REQUIRED_Q_VALUES = (1, 2, 3, 4)
REQUIRED_BATCH_SIZES = (1, 4)
REQUIRED_REPORT_FIELDS = (
    "schema_version",
    "checkpoint_path",
    "checkpoint_manifest_sha256",
    "device_name",
    "torch_version",
    "cuda_version",
    "q_values",
    "batch_sizes",
    "loader_passed",
    "fused_model_runner_path_exercised",
    "target_forward_real",
    "target_logits_cuda",
    "target_hidden_cuda",
    "target_hidden_consumed_by_real_executor",
    "target_logits_not_passed_to_mtp_executor",
    "public_result_tensor_count",
    "public_result_pickle_roundtrip",
    "public_result_tensor_free",
    "executor_identity_preserved",
    "sequence_order_preserved",
    "graph_eager_first_target_tokens_equal",
    "graph_eager_proposal_tokens_equal",
    "graph_capture_count",
    "graph_replay_count",
    "cleanup_passed",
    "backend_failures",
    "status",
    "promotion_classification",
    "coverage",
    "limitations",
)
```

The verifier rejects missing fields, domain drift, wrong types, any failed
critical boolean, nonzero public tensor count, backend failures, non-PASS
status, or a promotion classification other than `NOT_PROMOTABLE`.

- [ ] **Step 4: Run GREEN**

Run the focused test file. Expected: all schema/corruption tests pass.

---

### Task 2: Add Tensor-Free and Pickle Boundary Helpers

**Files:**
- Modify: `tools/qwen35_mtp_model_runner_ownership_gate.py`
- Modify: `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces `_count_tensors(value) -> int`.
- Produces `_canonical_public_result(rows) -> tuple[dict, ...]`.
- Produces `_validate_public_result(rows, sequence_ids) -> dict`.

- [ ] **Step 1: Write helper RED tests**

Require:

```python
def test_public_result_rejects_nested_tensor():
    Tensor = type("Tensor", (), {"__module__": "torch"})
    rows = (_row(1, metadata={"payload": Tensor()}),)
    with pytest.raises(ValueError, match="tensor"):
        _validate_public_result(rows, (1,))


def test_public_result_pickle_roundtrip_is_canonical():
    rows = (_row(3, proposal_tokens=(7, 8)),)
    observation = _validate_public_result(rows, (3,))
    assert observation == {
        "tensor_count": 0,
        "tensor_free": True,
        "pickle_roundtrip": True,
        "sequence_order_preserved": True,
        "canonical_rows": (
            {
                "sequence_id": 3,
                "target_token": 11,
                "proposal_token_ids": (7, 8),
                "source_type": "native_model_runner",
            },
        ),
    }
```

Require rejection of result rows exposing `target_hidden` or
`target_logits`, non-integer tokens, order drift, callable/module metadata,
and a tensor copied to CPU.

- [ ] **Step 2: Run RED**

Expected: tests fail because the helpers are absent.

- [ ] **Step 3: Implement minimal recursive validation**

Use `dataclasses.fields`, mappings, tuples/lists/sets, and object
`__module__`/class names to reject tensors, storages, CUDA graphs, streams,
events, modules, and callables. Call the production
`assert_tensor_free(...)`, then perform an independent recursive count and a
pickle round-trip. Canonicalize only sequence ID, target token, proposal
tokens, and source type.

- [ ] **Step 4: Run GREEN**

Run the focused file. Expected: helper tests pass.

---

### Task 3: Add Identity-Preserving Ownership Observers

**Files:**
- Modify: `tools/qwen35_mtp_model_runner_ownership_gate.py`
- Modify: `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces `_observe_loaded_fused_call(runner, executor)`.
- Produces an observation object with:
  `forward_rows`, `executor_rows`, `executor_identity_preserved`, and
  `restored`.

- [ ] **Step 1: Write observer RED tests**

Use dependency-light fake tensor objects with `is_cuda`, `device`, `dtype`,
and `shape`. Require:

```python
def test_observer_delegates_and_restores_original_identities():
    original_run_model = runner.run_model
    original_propose_batch = executor.propose_batch
    with _observe_loaded_fused_call(runner, executor) as observation:
        runner.run_model(..., return_hidden=True, execution_mode="decode")
        executor.propose_batch((_input(target_hidden=hidden),))
    assert runner.run_model is original_run_model
    assert executor.propose_batch is original_propose_batch
    assert observation.restored is True
    assert observation.executor_identity_preserved is True
```

Require rejection if `run_model` is not called with real decode/hidden
arguments, if hidden/logits are not CUDA, if devices differ, if MTP receives
target logits, or if wrappers retain a tensor reference in durable
observation state.

- [ ] **Step 2: Run RED**

Expected: missing observer failure.

- [ ] **Step 3: Implement wrappers**

Bind wrappers with `types.MethodType`, delegate exactly once, record only
plain strings/integers/booleans/shape tuples, and restore both original bound
methods in `finally`. Compare `runner.model` and executor identities before
and after; do not change registry membership.

- [ ] **Step 4: Run GREEN**

Run the focused file. Expected: observer tests pass.

---

### Task 4: Add Fresh Target KV and Hybrid-State Scenario Ownership

**Files:**
- Modify: `tools/qwen35_mtp_model_runner_ownership_gate.py`
- Modify: `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces `_LoadedScenarioOwner`.
- Produces
  `owner.build(q, batch_size, *, sequence_id_base) -> tuple[Sequence, ...]`.
- Produces `owner.cleanup() -> dict`.

- [ ] **Step 1: Write scenario-owner RED tests**

Use fake KV tensors, a real dependency-light
`HybridStateSlotAllocator`, and a fake runtime bridge. Require:

```python
def test_scenario_owner_allocates_distinct_blocks_and_leases():
    seqs = owner.build(4, 4, sequence_id_base=1000)
    assert tuple(seq.seq_id for seq in seqs) == (1000, 1001, 1002, 1003)
    assert len({seq.block_table[-1] for seq in seqs}) == 4
    assert len({seq.hybrid_state_slot_id for seq in seqs}) == 4
    assert all(seq.max_tokens - seq.num_completion_tokens == 4 for seq in seqs)


def test_scenario_owner_cleanup_releases_and_zeros_everything():
    owner.build(2, 1, sequence_id_base=2000)
    result = owner.cleanup()
    assert result["cleanup_passed"] is True
    assert result["active_leases"] == 0
    assert result["nonzero_target_kv_rows"] == 0
```

Require capacity failure before mutation, unique ownership, reverse-order
cleanup, idempotent cleanup, and fail-closed reporting on release failure.

- [ ] **Step 2: Run RED**

Expected: missing owner failure.

- [ ] **Step 3: Implement scenario ownership**

Use:

```python
HybridStateSlotAllocator(
    runner.qwen35_hybrid_model_owner.pool.capacity
)
```

For each sequence, allocate one lease, assign its slot/generation, reserve a
distinct target KV block within `runner.kv_cache.shape[2]`, and zero that
block before use. Build deterministic one-token decode histories with
`SamplingParams(temperature=0.0, max_tokens=q)`. Cleanup releases active
runtime-bridge pool bindings, allocator leases, zeros reserved target KV
blocks, calls `executor.release_sequence(...)`, and resets context.

- [ ] **Step 4: Run GREEN**

Run the focused file. Expected: scenario tests pass.

---

### Task 5: Add the Fused Graph/Eager Ownership Probe

**Files:**
- Modify: `tools/qwen35_mtp_model_runner_ownership_gate.py`
- Modify: `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces
  `_build_fused_ownership_probe(runner, descriptor, executor)`.
- Probe signature:
  `probe(q: int, batch_size: int) -> dict`.

- [ ] **Step 1: Write probe RED tests**

Use a fake runner whose `call(...)` records the exact method name and returns
tensor-free rows. Require:

```python
def test_probe_uses_model_runner_call_and_fresh_graph_eager_state():
    result = probe(4, 4)
    assert runner.calls == [
        ("run_spec_first_target_and_proposal_batch", graph_ids),
        ("run_spec_first_target_and_proposal_batch", eager_ids),
    ]
    assert set(graph_ids).isdisjoint(eager_ids)
    assert result["first_target_tokens_equal"] is True
    assert result["proposal_tokens_equal"] is True
    assert result["public_result_tensor_count"] == 0
```

Require Q/batch validation, graph runner disabled only for the eager side,
restoration in `finally`, exact ordered token comparison, observer success on
both sides, and cleanup success even when the fused call raises.

- [ ] **Step 2: Run RED**

Expected: probe tests fail because the builder is absent.

- [ ] **Step 3: Implement minimal probe**

For each side:

1. create a fresh `_LoadedScenarioOwner`;
2. install `_observe_loaded_fused_call`;
3. call
   `runner.call("run_spec_first_target_and_proposal_batch", seqs,
   descriptor, ())`;
4. validate the public result;
5. clean up in `finally`.

Temporarily set `executor.graph_runner = None` only for the eager side and
restore the exact original object. Return only canonical rows and scalar
observations.

- [ ] **Step 4: Run GREEN**

Run the focused file. Expected: probe tests pass.

---

### Task 6: Build the Report and CLI

**Files:**
- Modify: `tools/qwen35_mtp_model_runner_ownership_gate.py`
- Modify: `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces `run_gate(checkpoint_path, backend=None) -> dict`.
- Produces CLI:
  `python3 tools/qwen35_mtp_model_runner_ownership_gate.py
  --checkpoint PATH --output FILE`.

- [ ] **Step 1: Write orchestration RED tests**

Create a recording backend and require all eight Q/batch pairs, aggregate
booleans, exact graph counters, `backend_failures=[]`, checkpoint hash
preservation, `PASS`, and `NOT_PROMOTABLE`. Inject one failure in every
backend phase and require a written FAIL artifact that the success verifier
rejects.

- [ ] **Step 2: Run RED**

Expected: missing `run_gate`.

- [ ] **Step 3: Implement report aggregation**

Aggregate:

```python
cases = [
    backend.compare_fused_graph_eager(q, batch_size)
    for batch_size in REQUIRED_BATCH_SIZES
    for q in REQUIRED_Q_VALUES
]
```

Require eight cases, Q1 no capture, six exact graph family captures for
Q2/Q3/Q4 and batch 1/4, positive replay count, zero public tensors,
successful cleanup, and no backend failures. Always write a JSON report,
then call the success verifier only when `status == "PASS"`.

- [ ] **Step 4: Run GREEN**

Run the focused file. Expected: orchestration and corruption tests pass.

---

### Task 7: Load the Real ModelRunner and Install the Probe

**Files:**
- Modify: `tools/qwen35_mtp_model_runner_ownership_gate.py`
- Modify: `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces `RealLoadedModelRunnerOwnershipBackend`.
- Reuses
  `checkpoint_manifest_sha256` and the real Qwen3.5 runtime construction from
  `tools/qwen35_mtp_real_checkpoint_gate.py`.

- [ ] **Step 1: Write loader/source RED tests**

Require source to:

- construct TP1, eager target execution, native MTP, exact graph Q2/Q3/Q4,
  batch 1/4, and KV offload disabled;
- obtain exact `runner`, descriptor, and executor identities;
- reject missing target model, owner, descriptor, executor, physical store,
  or graph runner;
- never assign or patch `runner.run_model` outside the temporary observer;
- expose one callable fused ownership probe;
- preserve checkpoint manifest before/after.

- [ ] **Step 2: Run RED**

Expected: backend source/loader tests fail.

- [ ] **Step 3: Implement the real backend**

Use the existing real runtime loader to avoid loading a second copy of the
checkpoint. Build the fused probe from:

```python
runner = runtime["runner"]
descriptor = runner.qwen35_mtp_executor_descriptor
executor = runner.qwen35_mtp_executor
```

Record blockers under `load`, `ownership`, `graph_eager`, or `cleanup`.
Return metadata from the current CUDA device and fail closed on any missing
identity.

- [ ] **Step 4: Run GREEN**

Run the focused file. Expected: loader/source tests pass without requiring
local torch.

---

### Task 8: Add the Serial Remote GPU7 Wrapper

**Files:**
- Create:
  `tools/run_qwen35_mtp_model_runner_ownership_gate_remote.sh`
- Modify:
  `tools/test_qwen35_mtp_model_runner_ownership_gate.py`

**Interfaces:**
- Produces a local artifact under
  `artifacts/qwen35-mtp-runs/<opaque-run-id>/`.

- [ ] **Step 1: Write wrapper RED tests**

Require:

- `sitian@10.232.195.203`;
- `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`;
- `ControlMaster=no`;
- `ControlPath=none`;
- `CUDA_VISIBLE_DEVICES=7`;
- one foreground SSH execution path;
- a fresh high `TINYVLLM_DIST_PORT`;
- sync of the new gate plus every modified runtime dependency used by the
  existing MTP graph gate;
- artifact download and local verifier execution;
- no parallel `ssh`, `scp`, `&`, `xargs -P`, or background subshell.

- [ ] **Step 2: Run RED**

Expected: wrapper contract fails because the file is absent.

- [ ] **Step 3: Implement wrapper**

Follow the existing native-MTP remote wrapper layout, create a fresh isolated
remote run directory, upload source serially, execute the gate once on GPU7,
download the JSON, and run the local success verifier.

- [ ] **Step 4: Run GREEN and syntax checks**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_mtp_model_runner_ownership_gate.py -q
bash -n \
  tools/run_qwen35_mtp_model_runner_ownership_gate_remote.sh
```

Expected: both pass.

---

### Task 9: Run the Real GPU Gate and Verify the Artifact

**Files:**
- Create:
  `artifacts/qwen35-mtp-runs/<opaque-run-id>/qwen35_mtp_model_runner_ownership_gate.json`

- [x] **Step 1: Run local focused regression**

Run the new focused tests plus:

```bash
python3 -m pytest \
  tools/test_model_runner_proposal_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_cuda_graph_backend.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
```

Expected: all dependency-light tests pass.

Result:

```text
ownership focused:          71 passed
runtime regression group 1: 126 passed
runtime regression group 2: 83 passed
runtime regression total:   209 passed
```

- [x] **Step 2: Run remote wrapper serially**

Execute the new wrapper once. If SSH closes with unknown port 65535, retry
the same serial command a bounded number of times. Do not start parallel SSH
sessions.

- [x] **Step 3: Verify exact artifact fields**

Require:

```text
status=PASS
promotion_classification=NOT_PROMOTABLE
backend_failures=[]
fused_model_runner_path_exercised=true
target_forward_real=true
target_logits_cuda=true
target_hidden_cuda=true
target_hidden_consumed_by_real_executor=true
target_logits_not_passed_to_mtp_executor=true
public_result_tensor_count=0
public_result_pickle_roundtrip=true
public_result_tensor_free=true
executor_identity_preserved=true
sequence_order_preserved=true
graph_eager_first_target_tokens_equal=true
graph_eager_proposal_tokens_equal=true
cleanup_passed=true
```

Also record checkpoint SHA-256, device, PyTorch, CUDA, capture/replay counts,
and exact Q/batch domains.

Result:

```text
opaque run ID:
  qwen35-mtp-ownership-26023-99638
status / classification: PASS / NOT_PROMOTABLE
checkpoint SHA-256:
  9a975bdcf0383774183cae560594dd60b522b83fe9c4cd595c47c12e2403702b
device:                   NVIDIA A100 80GB PCIe
PyTorch / CUDA:           2.4.1+cu121 / 12.1
Q / batch domain:         (1,2,3,4) / (1,4)
capture / replay count:   6 / 6
cleanup:                  PASS for all 8 cases
```

---

### Task 10: Final Regression, Audit, and Handoff

**Files:**
- Modify:
  `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify:
  `docs/superpowers/plans/2026-08-12-loaded-gpu-model-runner-proposal-ownership-gate.md`

- [x] **Step 1: Run final verification**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_mtp_model_runner_ownership_gate.py \
  tools/test_model_runner_spec_verify.py \
  tools/test_model_runner_proposal_executor.py \
  tools/test_qwen35_mtp_model_runner_integration.py \
  tools/test_qwen35_mtp_cuda_graph_backend.py \
  tools/test_qwen35_mtp_real_checkpoint_gate.py -q
python3 -m py_compile \
  tools/qwen35_mtp_model_runner_ownership_gate.py
bash -n \
  tools/run_qwen35_mtp_model_runner_ownership_gate_remote.sh
git diff --check -- \
  tools/qwen35_mtp_model_runner_ownership_gate.py \
  tools/test_qwen35_mtp_model_runner_ownership_gate.py \
  tools/run_qwen35_mtp_model_runner_ownership_gate_remote.sh \
  docs/superpowers/specs/2026-08-12-loaded-gpu-model-runner-proposal-ownership-gate-design.md \
  docs/superpowers/plans/2026-08-12-loaded-gpu-model-runner-proposal-ownership-gate.md \
  docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md \
  AGENT_HANDOFF_STATE.md
```

Result:

```text
combined relevant pytest matrix: 280 passed
Python syntax compilation:       PASS
remote wrapper bash syntax:      PASS
downloaded artifact verifier:    PASS
scoped git diff check:           PASS
```

- [x] **Step 2: Update the audit**

Append the exact local counts, opaque run ID, checkpoint hash, device,
PyTorch/CUDA versions, ownership fields, graph/eager parity, cleanup result,
and unsupported scope. Do not rewrite historical evidence.

- [x] **Step 3: Update handoff**

Append a continuation that marks only the loaded-GPU TP1 ownership gate
closed. Preserve `NOT_PROMOTABLE` and set the next action to the pending TP1
16K/32K blockwise campaign.

- [x] **Step 4: Re-run document checks**

Run the artifact verifier and scoped `git diff --check` after the final
document patch. Mark plan checkboxes complete only after fresh evidence.
