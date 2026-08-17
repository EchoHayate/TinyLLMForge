# Qwen3.5 Generic Speculative TP4 Transactional Correctness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Execute inline in the current session; the user explicitly forbids subagents.

**Goal:** Establish a real, source-bound Qwen3.5 TP4/4K generic speculative authority with exact greedy parity, all-rank transactional KV and recurrent side-state evidence, no accepted-prefix replay, production movement provenance, and complete cleanup.

**Architecture:** Add an independent Qwen3.5 TP4 gate, worker, verifier, remote runner, and focused test module. Compose the already-established Qwen3.5 TP1 side-state semantics with the standard Qwen3 TP4 profile/residency/movement evidence shape and the existing Qwen3.5 TP4 Engine construction; do not create another speculative runtime or alter an existing authority schema.

**Tech Stack:** Python 3, pytest, TinyLLMForge `LLMEngine`, PyTorch distributed/NCCL, JSON authority artifacts, Bash, SSH/rsync, SHA-256 source binding.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not stage, commit, push, switch branches, create worktrees, stash, reset, or clean.
- Do not use subagents.
- Use `apply_patch` for every file edit.
- Follow strict TDD for every behavior change: valid RED, minimal implementation, GREEN.
- Preserve the existing Qwen3 generic TP4 and Qwen3.5 generic TP1 schemas and behavior.
- Use only `sitian@10.232.195.203` for remote execution.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use SSH with `ControlMaster=no` and `ControlPath=none`.
- Keep SSH, rsync, and polling serial, bounded, and finite-retry.
- Use the approved Qwen3.5 checkpoint and manifest SHA-256 `3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0`.
- Use real four-GPU/checkpoint evidence; controlled fakes prove only orchestration contracts.
- Do not accept a second full model forward replay of the accepted prefix.
- Do not treat synthetic tensor copies as production KV movement.
- A passing gate may claim only `SECOND_MODEL_TP4_4K_ESTABLISHED`.
- Keep Phase 1 `NOT_PROMOTABLE`; do not claim 16K/32K, performance, learned drafter, KV8/KV4, or Phase 1 completion.
- Do not modify production runtime unless a focused test or real gate produces a reproducible RED proving that authority cannot be assembled from existing behavior.

## File Structure

- Create `tools/qwen35_generic_speculative_tp4_gate.py`: frozen schema, validators, source/model hashing, campaign orchestration, and atomic artifact publication.
- Create `tools/qwen35_generic_speculative_tp4_worker.py`: real TP4 Engine cell execution and rank-local evidence normalization.
- Create `tools/verify_qwen35_generic_speculative_tp4_gate.py`: fresh-process independent artifact and source verifier.
- Create `tools/run_qwen35_generic_speculative_tp4_gate_remote.sh`: serial bounded remote campaign runner with a non-replayable status machine.
- Create `tools/test_qwen35_generic_speculative_tp4_gate.py`: contract, worker, campaign, verifier, and runner tests.
- Modify `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`: add the TP4/4K result only after real verification.
- Modify `AGENT_HANDOFF_STATE.md`: record exact evidence, hashes, limitations, and next gate only after real verification.
- Modify production files only if Task 8 exposes a focused reproducible runtime defect; such a change must have its own RED/GREEN cycle and remain inside the existing generic runtime.

---

### Task 1: Freeze the TP4 Authority Identity and Cell Inventory

**Files:**
- Create: `tools/test_qwen35_generic_speculative_tp4_gate.py`
- Create: `tools/qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces: `SCHEMA_VERSION`, `CLASSIFICATION`, `CLAIM_SCOPE`, `LIMITATIONS`, `POLICIES`, `BATCH_SIZES`, `WORLD_SIZE`, `CONTEXT_TOKENS`, `NGRAM_SIZE`, `MAX_PROPOSAL_TOKENS`, `MAX_OUTPUT_TOKENS`, `MODEL_MANIFEST_SHA256`, `DEFAULT_SOURCE_FILES`, and `cell_key(policy: str, batch_size: int) -> str`.
- Consumed by: every later task.

- [ ] **Step 1: Write the failing frozen-contract test**

```python
def test_contract_constants_are_frozen():
    gate = _load_module(
        "qwen35_generic_speculative_tp4_gate",
        TOOLS / "qwen35_generic_speculative_tp4_gate.py",
    )
    assert gate.SCHEMA_VERSION == (
        "qwen35.generic-speculative-tp4-"
        "transactional-correctness.v1"
    )
    assert gate.CLASSIFICATION == "SECOND_MODEL_TP4_4K_ESTABLISHED"
    assert gate.CLAIM_SCOPE == "second_model_tp4_4k_only"
    assert gate.WORLD_SIZE == 4
    assert gate.BATCH_SIZES == (1, 4)
    assert gate.POLICIES == ("baseline", "ngram")
    assert gate.CONTEXT_TOKENS == 4096
    assert gate.NGRAM_SIZE == 3
    assert gate.MAX_PROPOSAL_TOKENS == 4
    assert gate.MAX_OUTPUT_TOKENS == 8
    assert gate.MODEL_MANIFEST_SHA256 == (
        "3e650a908234771c3cf1ac4e20c4d38fe"
        "69982efedaf4a3e631ad0b14aad7dd0"
    )
    assert "phase1_not_promotable" in gate.LIMITATIONS
    assert gate.cell_key("baseline", 1) == "baseline:b1"
    assert gate.cell_key("ngram", 4) == "ngram:b4"
```

- [ ] **Step 2: Run the test and verify the import is RED**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_generic_speculative_tp4_gate.py::test_contract_constants_are_frozen
```

Expected: FAIL because `tools/qwen35_generic_speculative_tp4_gate.py` does not exist.

- [ ] **Step 3: Add the minimal frozen constants and `cell_key`**

```python
SCHEMA_VERSION = (
    "qwen35.generic-speculative-tp4-"
    "transactional-correctness.v1"
)
CLASSIFICATION = "SECOND_MODEL_TP4_4K_ESTABLISHED"
CLAIM_SCOPE = "second_model_tp4_4k_only"
LIMITATIONS = (
    "phase1_not_promotable",
    "context_16k_not_established",
    "context_32k_not_established",
    "performance_not_established",
    "learned_drafter_not_established",
    "kv_quantization_not_established",
)
POLICIES = ("baseline", "ngram")
BATCH_SIZES = (1, 4)
WORLD_SIZE = 4
CONTEXT_TOKENS = 4096
NGRAM_SIZE = 3
MAX_PROPOSAL_TOKENS = 4
MAX_OUTPUT_TOKENS = 8
MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe"
    "69982efedaf4a3e631ad0b14aad7dd0"
)


def cell_key(policy: str, batch_size: int) -> str:
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    if batch_size not in BATCH_SIZES:
        raise ValueError("unsupported batch size")
    return f"{policy}:b{batch_size}"
```

Set `DEFAULT_SOURCE_FILES` to the production speculative/side-state sources
from the TP1 gate plus the new TP4 gate, worker, and verifier.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run the Step 2 command.

Expected: `1 passed`.

---

### Task 2: Validate Qwen3.5 Identity, Tokens, and Canonical Consumed Inputs

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`
- Modify: `tools/qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces: `_integer`, `_sha256`, `_json_sha256`, `_validate_token_rows`, `_validate_model_identity`, `_validate_mapping`.
- Consumed by: Tasks 3–7.

- [ ] **Step 1: Add failing identity and mapping tests**

```python
def test_model_identity_requires_real_qwen35_hybrid_shape():
    assert gate._validate_model_identity({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_layer_count": 24,
        "linear_layer_count": 18,
        "full_attention_layer_count": 6,
    })["linear_layer_count"] == 18


def test_consumed_input_mapping_is_canonical():
    row = gate._validate_mapping({
        "sequence_id": 9,
        "proposal_token_count": 4,
        "accepted_draft_count": 2,
        "verify_input_count": 3,
        "committed_tail_input_count": 2,
        "committed_input_count": 3,
    })
    assert row["committed_input_count"] == 3


def test_consumed_input_mapping_rejects_output_length_inference():
    with pytest.raises(ValueError, match="committed input count mismatch"):
        gate._validate_mapping({
            "sequence_id": 9,
            "proposal_token_count": 4,
            "accepted_draft_count": 2,
            "verify_input_count": 3,
            "committed_tail_input_count": 2,
            "committed_input_count": 4,
        })
```

- [ ] **Step 2: Run the three tests and verify RED**

Expected: FAIL because the validators are not defined.

- [ ] **Step 3: Implement the validators using the frozen TP1 semantics**

The mapping implementation must compute:

```python
verify_count = max(0, proposal_count - 1)
committed_tail = min(accepted_count, verify_count)
committed_input = 1 + committed_tail
```

The model validator must require exactly 24 text layers, 18 linear-attention
layers, 6 full-attention layers, and architecture
`Qwen3_5ForConditionalGeneration`.

- [ ] **Step 4: Run the tests and verify GREEN**

Expected: all three tests pass.

---

### Task 3: Freeze All-Rank Transaction and Side-State Evidence

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`
- Modify: `tools/qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces:
  - `_transaction_semantic_digest(row: dict) -> str`
  - `_validate_sequence_transaction(row: object, *, rank: int) -> dict`
  - `_validate_side_state(receipts: object, failure_rollbacks: object, *, rank: int) -> list[dict]`
  - `_validate_rank_transaction_evidence(value: object, *, rank: int, policy: str) -> dict`
- Semantic digest includes sequence/cell/proposal/acceptance/committed-input/KV-decision/selected-checkpoint semantics and excludes physical slot IDs.

- [ ] **Step 1: Add failing complete-lifecycle and cross-rank tests**

Create helpers:

```python
def _side_state_receipts(rank, handle_id, sequence_id):
    return [
        {
            "rank": rank,
            "handle_id": handle_id,
            "sequence_id": sequence_id,
            "operation": operation,
            "state": state,
        }
        for operation, state in (
            ("prepare", "prepared"),
            ("select", "selected"),
            ("apply", "applied"),
            ("seal", "sealed"),
        )
    ]
```

Add tests that:

- accept exactly `prepare, select, apply, seal` for each
  `(rank, handle_id, sequence_id)`;
- reject aggregation that omits rank;
- reject missing `select`;
- reject duplicate `seal`;
- reject a semantic digest mismatch between ranks; and
- prove different physical slot IDs do not change the semantic digest.

- [ ] **Step 2: Run the focused lifecycle tests and verify RED**

Expected: FAIL because the all-rank validators do not exist.

- [ ] **Step 3: Implement minimal rank-aware validation**

Use:

```python
by_lifecycle: dict[tuple[int, str, int], list[str]] = {}
```

Successful lifecycle operations must equal:

```python
["prepare", "select", "apply", "seal"]
```

Build the digest from a normalized mapping and hash with compact sorted JSON:

```python
hashlib.sha256(
    json.dumps(
        semantic_row,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
).hexdigest()
```

- [ ] **Step 4: Run the lifecycle tests and verify GREEN**

Expected: all lifecycle and digest tests pass.

---

### Task 4: Reuse TP4 Profile, Residency, Movement, and Cleanup Contracts

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`
- Modify: `tools/qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces:
  - `validate_rank_profile(value: object, *, policy: str) -> dict`
  - `validate_residency_phases(value: object) -> list[dict]`
  - `_validate_kv_rank_deltas(value: object) -> list[dict]`
  - `validate_cleanup_receipt(value: object) -> dict`
- Contract is behaviorally aligned with `tools/generic_speculative_tp4_gate.py` but remains in the independent Qwen3.5 schema.

- [ ] **Step 1: Add failing tests for four-rank distributed evidence**

Add tests that reject:

- rank inventory other than `{0, 1, 2, 3}`;
- missing speculative callback on one candidate rank;
- speculative callback in a baseline profile;
- missing or mismatched collective identity;
- incomplete residency operation order;
- movement rows without production counter provenance;
- a nonzero rank exit;
- a rank with process group still initialized;
- a rank with live leases or prepared transactions; and
- an Engine whose `exit()` receipt is missing.

Movement rows must include a provenance field equal to:

```text
engine.kv_offload_summaries
```

- [ ] **Step 2: Run the profile/residency/movement/cleanup tests and verify RED**

Expected: FAIL because the validators are absent.

- [ ] **Step 3: Implement only the required Qwen3.5 TP4 validators**

Port the proven normalization rules from
`tools/generic_speculative_tp4_gate.py`, adding Qwen3.5 cleanup fields:

```python
{
    "rank": rank,
    "worker_exit_code": 0,
    "process_group_initialized": False,
    "engine_exit_called": True,
    "live_lease_count": 0,
    "prepared_transaction_count": 0,
    "runtime_poisoned": False,
}
```

Do not import the existing gate module at runtime; independent schema
validation must not depend on another authority implementation.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Expected: all distributed-evidence tests pass.

---

### Task 5: Validate Complete Cells and the Final Authority Result

**Files:**
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`
- Modify: `tools/qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces:
  - `_validate_runtime(value: object, *, policy: str, batch_size: int) -> dict`
  - `validate_cell_result(value: object) -> dict`
  - `validate_result(value: object) -> dict`

- [ ] **Step 1: Build a valid four-rank fixture and failing mutation tests**

Create `_valid_cell(policy="ngram", batch_size=1)` and `_valid_result()` with
all four rank records.

Add mutation tests proving rejection of:

- incomplete cell inventory;
- world size not equal to four;
- context length not equal to 4096;
- prompt or output mismatch between baseline and candidate;
- zero accepted draft tokens;
- zero rejected draft tokens;
- nonzero accepted-prefix replay count;
- missing sequence transaction on any rank;
- cross-rank semantic digest mismatch;
- absent callback/collective/residency/movement evidence;
- incomplete cleanup;
- missing `phase1_not_promotable`; and
- a classification broader than `SECOND_MODEL_TP4_4K_ESTABLISHED`.

- [ ] **Step 2: Run the cell/result tests and verify RED**

Expected: FAIL because `validate_cell_result` and `validate_result` are absent.

- [ ] **Step 3: Implement minimal complete-result validation**

The expected cell set is:

```python
{
    cell_key(policy, batch_size)
    for batch_size in BATCH_SIZES
    for policy in POLICIES
}
```

For each batch size, require exact equality of normalized prompt and output
rows between baseline and n-gram cells.

For each candidate sequence, require one semantic digest shared by all four
ranks.

- [ ] **Step 4: Run the cell/result tests and verify GREEN**

Expected: all contract mutation tests pass.

---

### Task 6: Build the Rank-Aware Worker with Controlled Fakes

**Files:**
- Create: `tools/qwen35_generic_speculative_tp4_worker.py`
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces:
  - `_integer_mapping(value: object, name: str) -> dict[int, int]`
  - `normalize_side_state_receipts(receipts: list[dict], *, rank: int) -> list[dict]`
  - `summarize_step_observations(observations: list[dict], *, rank_receipts: dict[int, list[dict]]) -> dict`
  - `capture_rank_side_state_receipts(engine) -> dict[int, list[dict]]`
  - `build_prompt_rows(tokenizer, batch_size: int) -> list[dict]`
  - `run_generation(...) -> dict`
  - `run_policy_cell(...) -> dict`

- [ ] **Step 1: Add failing worker normalization tests**

Tests must prove:

- batch receipts expand per sequence and retain rank;
- consumed-input mappings are recomputed from proposal/acceptance counts;
- rank 0 cannot synthesize missing receipts for ranks 1–3;
- accepted-prefix replay remains zero;
- baseline returns empty speculative/side-state evidence;
- candidate records both accepted and rejected tokens; and
- model identity is read from the nested Qwen3.5 text config.

- [ ] **Step 2: Run the worker tests and verify RED**

Expected: FAIL because the worker module does not exist.

- [ ] **Step 3: Implement normalization and fake-compatible orchestration**

Load the gate with `importlib.util` as the existing workers do.

Normalize receipts to:

```python
{
    "rank": rank,
    "sequence_id": sequence_id,
    "handle_id": transaction_id,
    "operation": operation,
    "state": status,
}
```

Do not accept a receipt lacking an explicit source rank.

- [ ] **Step 4: Add a fake TP4 Engine test for a complete cell**

The fake must expose the same public methods used by the real path:

```python
activate_speculative_runtime(...)
configure_decode_internal_profile(...)
step()
clear_reusable_prefix_cache()
kv_offload_summaries(...)
finalize_decode_internal_profile(...)
exit()
```

The test must assert that the worker:

- builds TP size four;
- configures the existing generic runtime only for `ngram`;
- captures four-rank side-state/profile/movement/cleanup evidence; and
- returns `gate.validate_cell_result(cell)`.

- [ ] **Step 5: Run all worker tests and verify GREEN**

Expected: all worker tests pass with controlled fakes.

---

### Task 7: Wire the Real Qwen3.5 TP4 Engine Cell

**Files:**
- Modify: `tools/qwen35_generic_speculative_tp4_worker.py`
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Consumes:
  - Qwen3.5 Engine construction pattern from `tools/qwen35_tp4_engine_backend_session.py`
  - `EngineSpeculativeRuntime`
  - `NGramDraftAdapter`
  - `SamplingParams`
  - existing Engine profile, residency, movement, side-state, and cleanup APIs.
- Produces: a real cell artifact accepted by `gate.validate_cell_result`.

- [ ] **Step 1: Add a source-contract RED test for real dependencies**

The test inspects the worker source and requires:

```python
from tinyvllm import LLM
from tinyvllm.engine.speculative_runtime import EngineSpeculativeRuntime
from tinyvllm.speculative.ngram_adapter import NGramDraftAdapter
from tinyvllm.sampling_params import SamplingParams
```

It also requires `tensor_parallel_size=gate.WORLD_SIZE`,
`enforce_eager=True`, and no replay helper or second accepted-prefix
`engine.step()` loop.

- [ ] **Step 2: Run the source-contract test and verify RED**

Expected: FAIL until the real dependency factory and cell configuration exist.

- [ ] **Step 3: Implement the real dependency factory and cell execution**

Construct the Engine with the established TP4/Qwen3.5-safe configuration:

```python
engine = engine_factory(
    model_path,
    tensor_parallel_size=gate.WORLD_SIZE,
    enforce_eager=True,
    max_model_len=4352,
    max_num_batched_tokens=16384,
    max_num_seqs=batch_size,
    max_num_prefill_tokens_per_step=1024,
    chunked_prefill_decode_first=False,
    chunked_prefill_mixed_batch=False,
    kv_offload_mvp0=True,
    kv_offload_gpu_blocks=68,
    kv_offload_logical_blocks=640,
    kv_offload_blockwise_decode=True,
    kv_offload_blockwise_prefill=True,
    kv_offload_blockwise_blocks=8,
)
```

For `ngram`, activate exactly:

```python
EngineSpeculativeRuntime(
    NGramDraftAdapter(
        ngram_size=gate.NGRAM_SIZE,
        max_proposal_tokens=gate.MAX_PROPOSAL_TOKENS,
    )
)
```

Use the existing profile/residency/movement APIs. Read side-state receipts
from the production ModelRunner evidence source already used by the TP1
worker, extended only to preserve explicit rank identity.

- [ ] **Step 4: Run worker and existing authority regressions**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_generic_speculative_tp4_gate.py \
  tools/test_generic_speculative_tp4_gate.py \
  tools/test_qwen35_generic_speculative_tp1_gate.py
```

Expected: all tests pass and neither existing schema changes.

---

### Task 8: Add Campaign Orchestration and Independent Verification

**Files:**
- Modify: `tools/qwen35_generic_speculative_tp4_gate.py`
- Create: `tools/verify_qwen35_generic_speculative_tp4_gate.py`
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces:
  - `atomic_write_json(path: Path, value: object) -> None`
  - `sha256_file(path: Path) -> str`
  - `hash_source_files(root: Path, source_files: tuple[str, ...]) -> dict[str, str]`
  - `source_tree_sha256(root: Path, source_files: tuple[str, ...]) -> str`
  - `model_manifest_sha256(model_path: str) -> str`
  - `run_campaign(...) -> dict`
  - verifier `verify_run(run_dir: Path, source_root: Path | None = None) -> dict`

- [ ] **Step 1: Add failing campaign tests**

Tests require:

- four fresh subprocess cells in order
  `baseline:b1`, `ngram:b1`, `baseline:b4`, `ngram:b4`;
- fresh distributed and master ports per cell;
- no output-directory reuse;
- atomic publication only after verification;
- `.failed` artifact retention for worker or verifier failure;
- source-tree and model-manifest binding; and
- no reliance on a prior `verify.json`.

- [ ] **Step 2: Add failing verifier tamper tests**

Write a complete fake run and prove rejection after separately tampering:

- a rank semantic digest;
- a side-state receipt;
- a movement provenance row;
- a cleanup receipt;
- `result.json` without updating its manifest hash;
- a bound source file; and
- the approved model manifest hash.

- [ ] **Step 3: Run campaign/verifier tests and verify RED**

Expected: FAIL because campaign and verifier implementations are incomplete.

- [ ] **Step 4: Implement campaign assembly**

Follow the existing campaign shape:

```python
for batch_size in BATCH_SIZES:
    for policy in POLICIES:
        subprocess.run(
            [
                python_executable,
                str(worker_script),
                "--model", model_path,
                "--gpu-indices", ",".join(map(str, gpu_indices)),
                "--policy", policy,
                "--batch-size", str(batch_size),
                "--dist-port", str(dist_port_base + cell_index),
                "--master-port", str(master_port_base + cell_index),
                "--out", str(cell_path),
            ],
            cwd=repo_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
```

Build and validate `result.json`, then bind it in `source_manifest.json`, run
the fresh verifier, and only then atomically rename the temporary directory.

- [ ] **Step 5: Implement the independent verifier**

The verifier must reload the new gate module, re-run `validate_result`, verify
the artifact hash, recompute every bound source hash when `source_root` is
provided, and return:

```python
{
    "classification": "PASS" if not failures else "FAIL",
    "failures": failures,
}
```

- [ ] **Step 6: Run campaign/verifier tests and verify GREEN**

Expected: all campaign and tamper tests pass.

---

### Task 9: Add the Serial Non-Replayable Remote Runner

**Files:**
- Create: `tools/run_qwen35_generic_speculative_tp4_gate_remote.sh`
- Modify: `tools/test_qwen35_generic_speculative_tp4_gate.py`

**Interfaces:**
- Produces: one local `authority` directory after successful fresh
  verification or one preserved `authority.failed` directory after failure.

- [ ] **Step 1: Add a failing runner source-contract test**

The test must require:

- default host `sitian@10.232.195.203`;
- remote Python
  `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`;
- approved Qwen3.5 checkpoint;
- local Kerberos cache
  `FILE:/Users/bytedance/krb5cc_sitian`;
- `ControlMaster=no`;
- `ControlPath=none`;
- finite retry counts;
- serial transfer and polling;
- `campaign.status`, `campaign.pid`, and `campaign.exit_code`;
- refusal to replay an existing campaign;
- four GPU indices;
- fresh nonzero distributed/master port bases; and
- local independent verification after artifact retrieval.

- [ ] **Step 2: Run the runner test and verify RED**

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the minimal runner**

Adapt the proven status machine from
`tools/run_qwen35_generic_speculative_tp1_gate_remote.sh` and the four-GPU
launch parameters from `tools/run_generic_speculative_tp4_gate_remote.sh`.

Use a single SSH option array containing:

```bash
-o ControlMaster=no
-o ControlPath=none
-o BatchMode=yes
-o ConnectTimeout=20
```

All retries must be bounded by numeric environment variables with finite
defaults.

- [ ] **Step 4: Run shell and runner validation**

Run:

```bash
bash -n tools/run_qwen35_generic_speculative_tp4_gate_remote.sh
python -m pytest -q \
  tools/test_qwen35_generic_speculative_tp4_gate.py \
  -k remote_runner
```

Expected: shell syntax passes and runner tests pass.

---

### Task 10: Execute the Complete Local Preflight

**Files:**
- No edit unless a valid focused RED is found.

**Interfaces:**
- Produces: permission to launch the real campaign only if every preflight is
  GREEN.

- [ ] **Step 1: Compile the new Python tools**

Run:

```bash
python -m py_compile \
  tools/qwen35_generic_speculative_tp4_gate.py \
  tools/qwen35_generic_speculative_tp4_worker.py \
  tools/verify_qwen35_generic_speculative_tp4_gate.py
```

Expected: exit code 0.

- [ ] **Step 2: Run the focused new test module**

Run:

```bash
python -m pytest -q \
  tools/test_qwen35_generic_speculative_tp4_gate.py
```

Expected: all tests pass.

- [ ] **Step 3: Run authority regression tests**

Run:

```bash
python -m pytest -q \
  tools/test_generic_speculative_tp4_gate.py \
  tools/test_qwen35_generic_speculative_tp1_gate.py
```

Expected: all tests pass without schema changes.

- [ ] **Step 4: Run shell and diff checks**

Run:

```bash
bash -n tools/run_qwen35_generic_speculative_tp4_gate_remote.sh
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 5: Stop on any production-runtime RED**

If a focused test reveals a production-runtime defect:

1. invoke `systematic-debugging`;
2. record the exact failing test and root cause;
3. add the smallest focused regression in the existing relevant test module;
4. verify the regression fails for the intended reason;
5. patch only the existing generic runtime with `apply_patch`;
6. run the focused GREEN test; and
7. rerun Steps 1–4.

Do not weaken the new gate to accommodate missing evidence.

---

### Task 11: Run the Real Qwen3.5 TP4/4K Authority

**Files:**
- Generated under: `artifacts/qwen35_generic_speculative_tp4/*/artifacts/authority`
- Generated on failure: corresponding `authority.failed`

**Interfaces:**
- Produces: real GPU/checkpoint authority or preserved failure evidence.

- [ ] **Step 1: Verify Kerberos without renewing credentials**

Run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian klist -s
```

Expected: exit code 0. If it fails, stop with a credential blocker; do not use
another identity.

- [ ] **Step 2: Launch the serial remote campaign**

Run:

```bash
bash tools/run_qwen35_generic_speculative_tp4_gate_remote.sh
```

Expected: one new opaque run directory. The command must not reuse an existing
campaign identity.

- [ ] **Step 3: Inspect raw evidence before trusting verification**

Confirm from `result.json` and rank records:

- exact rank set `{0,1,2,3}`;
- baseline/candidate cells for batch 1 and 4;
- 4096 context tokens;
- exact per-sequence greedy token parity;
- accepted and rejected drafts in each candidate cell;
- canonical committed-input mappings;
- complete side-state lifecycle on every rank and sequence;
- identical semantic digests across ranks;
- callback and collective evidence on every rank;
- complete residency phases;
- production KV movement provenance;
- zero accepted-prefix replay;
- no leases, poison, prepared transactions, live children, or initialized
  process groups after cleanup.

Expected: every item is present in raw evidence. Do not rely only on
`verify.json`.

- [ ] **Step 4: Run a fresh independent verifier**

Run:

```bash
AUTHORITY_DIR="$(
  find artifacts/qwen35_generic_speculative_tp4 \
    -type d \
    -path '*/artifacts/authority' \
    -print \
  | sort \
  | tail -n 1
)"
test -n "${AUTHORITY_DIR}"
python tools/verify_qwen35_generic_speculative_tp4_gate.py \
  "${AUTHORITY_DIR}" \
  --source-root .
```

Expected:

```text
"classification":"PASS"
```

and the validated result classification is
`SECOND_MODEL_TP4_4K_ESTABLISHED`.

- [ ] **Step 5: Preserve semantic failures**

If execution or verification fails, retain `authority.failed`, identify the
first semantic failure, invoke `systematic-debugging`, and add a focused RED
before changing code. Never overwrite or relabel failed evidence.

---

### Task 12: Record the Established Boundary and Re-Audit Phase 1

**Files:**
- Modify: `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: the real authority path, source/result hashes, raw counts, and
  independent verifier output from Task 11.
- Produces: an accurate handoff and objective coverage matrix.

- [ ] **Step 1: Add the TP4/4K audit row only after PASS**

Record:

```text
Qwen3.5 | generic n-gram speculative | TP4 | 4K | batch 1/4 |
SECOND_MODEL_TP4_4K_ESTABLISHED
```

Include exact authority path, source-tree hash, result hash, manifest hash,
accepted/rejected counts, rank inventory, and cleanup result.

- [ ] **Step 2: Keep uncovered requirements explicitly open**

The audit must continue to mark as missing:

- Qwen3.5 16K;
- Qwen3.5 32K;
- controlled TPOT/TTFT/throughput/memory/KV-H2D/acceptance comparison;
- learned drafter or native MTP through the unified runtime;
- KV8/KV4;
- real offload benefit; and
- Phase 1 promotion.

- [ ] **Step 3: Update the handoff**

Record:

- what the TP4 gate proves;
- what it does not prove;
- all validation commands and counts;
- any failed authorities and root-cause fixes;
- no-replay and production-movement evidence;
- the next concrete gate: Qwen3.5 16K/32K correctness before performance
  promotion.

- [ ] **Step 4: Run final document and repository checks**

Run:

```bash
git diff --check
rg -n \
  'SECOND_MODEL_TP4_4K_ESTABLISHED|NOT_PROMOTABLE|16K|32K|performance' \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md
```

Expected: diff check passes, the established TP4/4K row is present, and all
unestablished boundaries remain explicit.

---

## Final Completion Audit

Before reporting this plan complete, map each spec requirement to evidence:

| Requirement | Required evidence |
|---|---|
| Real Qwen3.5 checkpoint | approved manifest hash plus all-rank load receipts |
| TP4 | exact rank set `{0,1,2,3}` and process-group evidence |
| 4K | fixed cell context value `4096` |
| Batch 1/4 and multi-sequence | four cells and per-sequence rows |
| Exact greedy parity | identical token IDs, not text-only equality |
| Accepted and rejected drafts | positive counts in each candidate cell |
| Transactional KV | prepared commit and rejected-suffix rollback receipts |
| Recurrent side state | complete rank/handle/sequence lifecycle and selected checkpoint |
| No rematerialization | accepted-prefix replay count exactly zero |
| All-rank agreement | identical semantic transaction digests |
| Real movement | production `engine.kv_offload_summaries` provenance |
| Distributed execution | callback and collective profiles on all ranks |
| Cleanup | zero leaks/poison/prepared transactions/children and closed process groups |
| Source binding | source manifest and fresh source-tree recomputation |
| Independent authority | fresh verifier PASS |
| Narrow claim | only `SECOND_MODEL_TP4_4K_ESTABLISHED` |
| Phase 1 boundary | explicit `NOT_PROMOTABLE` and missing 16K/32K/performance rows |

Any missing, ambiguous, inherited, or rank-0-only evidence means the gate is
not established and work continues.
