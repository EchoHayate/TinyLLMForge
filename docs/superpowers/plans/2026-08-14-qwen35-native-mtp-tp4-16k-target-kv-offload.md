# Qwen3.5 Native MTP TP4 16K Target-KV Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: execute this plan inline with strict test-driven development. Subagents, worktrees, commits, staging, pushes, stashes, resets, and cleans are forbidden by the active user constraints.

**Goal:** Build and run an independent source-bound Qwen3.5 native-MTP TP4/16K correctness authority that uses real production target-KV offload and blockwise attention under a fixed 68-slot GPU staging budget.

**Architecture:** Add one small production instrumentation field for peak target-KV residency, then create a native-MTP 16K authority that reuses only side-effect-free validators and execution helpers from the frozen TP4/4K and generic TP4 authority code. The new worker uses the production Engine, native-MTP executor, speculative residency participant, and `KVOffloadMVP0`; it does not bind to the TP1/4K authority and does not offload proposal MTP KV.

**Tech Stack:** Python 3, PyTorch distributed/NCCL, TinyLLMForge Engine, `KVOffloadMVP0`, native Qwen3.5 MTP, pytest-compatible direct tests, Bash, SSH/rsync, JSON/SHA-256 authority artifacts.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify or reinterpret the established TP4/4K authority or its artifacts.
- Do not stage, commit, push, switch branches/worktrees, stash, reset, or clean.
- Do not use subagents.
- Every behavior change must follow observed RED, minimal implementation, observed GREEN, then focused regression.
- Use only `sitian@10.232.195.203` for remote GPU execution.
- Set `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use SSH options `ControlMaster=no` and `ControlPath=none`.
- Keep SSH, rsync, launch, and polling serial with finite retries.
- Never terminate unrelated GPU processes.
- Freeze schema `qwen35.native-mtp-tp4-16k-target-kv-offload.v1`.
- Freeze classification `QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED`.
- Freeze promotion classification `NOT_PROMOTABLE`.
- Freeze TP4, policies `(baseline, native_mtp)`, batch sizes `(1, 4)`, prompt `16384`, output `8`, and proposal maximum `4`.
- Freeze eager target and proposal execution; keep target and MTP CUDA Graphs disabled.
- Freeze `max_model_len=33024`, `max_num_batched_tokens=132096`, and prefill chunk `1024`.
- Freeze `kvcache_block_size=256`, `kv_offload_mvp0=True`, GPU blocks `68`, logical blocks `640`, blockwise prefill/decode enabled, and blockwise window `8`.
- Require exact greedy baseline/native parity.
- Require native proposal, acceptance, rejection, direct target-KV commit, rejected-suffix rollback, and zero accepted-prefix target replay.
- Require positive production target-KV D2H and H2D copies and bytes in native batch 4.
- Proposal MTP KV remains GPU-resident and is excluded from target-KV movement claims.
- Do not claim TP1/16K parity, 32K, performance improvement, KV8/KV4, proposal-KV offload, a second learned model structure, production readiness, or Phase 1 completion.

---

### Task 1: Add Peak Target-KV Residency Evidence

**Files:**
- Modify: `tools/test_kv_offload.py`
- Modify: `tinyvllm/engine/model_runner.py`

**Interfaces:**
- Consumes: `KVOffloadMVP0.logical_to_slot`, `KVOffloadMVP0.gpu_blocks`, and `KVOffloadMVP0.summary()`.
- Produces: monotonic integer `summary()["peak_resident_blocks"]` satisfying `resident_blocks <= peak_resident_blocks <= gpu_blocks`.

- [x] **Step 1: Write the failing CPU-only peak-residency test**

Add a test using `_NoopKVOffload` so the contract does not require CUDA:

```python
def test_summary_tracks_peak_resident_blocks_without_exceeding_capacity():
    manager = _NoopKVOffload()
    manager.stats["peak_resident_blocks"] = 0

    manager.logical_to_slot = {0: 0}
    manager._record_peak_resident_blocks()
    manager.logical_to_slot = {0: 0, 1: 1}
    manager._record_peak_resident_blocks()
    manager.logical_to_slot = {1: 1}
    manager._record_peak_resident_blocks()

    summary = manager.summary()
    assert summary["resident_blocks"] == 1
    assert summary["peak_resident_blocks"] == 2
    assert summary["gpu_blocks"] == 2
```

Add an invariant test:

```python
def test_peak_resident_blocks_rejects_mapping_over_capacity():
    manager = _NoopKVOffload()
    manager.stats["peak_resident_blocks"] = 0
    manager.logical_to_slot = {0: 0, 1: 1, 2: 2}

    _assert_raises(
        RuntimeError,
        "KV offload resident block count exceeds GPU capacity",
        manager._record_peak_resident_blocks,
    )
```

- [ ] **Step 2: Run the focused test on the remote Torch environment and observe RED**

Copy only the current source tree through the later runner staging convention,
then run:

```bash
KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian \
ssh -o ControlMaster=no -o ControlPath=none \
  sitian@10.232.195.203 \
  'cd /data00/home/sitian/sitian-workspace01/tllm/TinyLLMForge-adaptive-ngram && \
   PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
   tools/test_kv_offload.py'
```

Expected: FAIL because `_record_peak_resident_blocks` does not exist.

- [x] **Step 3: Implement the minimal monotonic counter**

Initialize the counter with the existing movement statistics:

```python
self.stats = {
    ...
    "peak_resident_blocks": 0,
    ...
}
```

Add:

```python
def _record_peak_resident_blocks(self) -> None:
    resident_blocks = len(self.logical_to_slot)
    if resident_blocks > self.gpu_blocks:
        raise RuntimeError(
            "KV offload resident block count exceeds GPU capacity"
        )
    self.stats["peak_resident_blocks"] = max(
        int(self.stats.get("peak_resident_blocks", 0)),
        resident_blocks,
    )
```

Call `_record_peak_resident_blocks()` immediately after every code path that
adds or remaps entries in `logical_to_slot`, before returning the mapping.
Do not decrement the counter during eviction, rollback, or cleanup.

Keep `summary()` unchanged except that `**self.stats` now exposes the new
field.

- [x] **Step 4: Run the focused test and observe GREEN**

Run the command from Step 2.

Expected: all direct `tools/test_kv_offload.py` tests pass or explicitly print
their existing CUDA skip messages; both new CPU-only tests execute and pass.

- [x] **Step 5: Run static verification without committing**

Run:

```bash
python3 -m py_compile tinyvllm/engine/model_runner.py tools/test_kv_offload.py
git diff --check -- tinyvllm/engine/model_runner.py tools/test_kv_offload.py
```

Expected: both commands succeed with no `git diff --check` output.

### Task 2: Freeze an Independent 16K Native-MTP Gate Contract

**Files:**
- Create: `tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`
- Create: `tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`

**Interfaces:**
- Consumes: side-effect-free helpers from
  `tools/qwen35_native_mtp_tp4_4k_engine_gate.py`.
- Produces:

```python
def cell_key(policy: str, batch_size: int) -> str: ...
def validate_engine_config(value: object) -> dict: ...
def validate_kv_rank_deltas(value: object) -> list[dict]: ...
def validate_kv_capacity_rows(value: object) -> list[dict]: ...
def validate_cell_result(value: object) -> dict: ...
def validate_result(value: object) -> dict: ...
```

- [x] **Step 1: Write failing frozen-constant and source-isolation tests**

Assert:

```python
assert gate.SCHEMA_VERSION == (
    "qwen35.native-mtp-tp4-16k-target-kv-offload.v1"
)
assert gate.CLASSIFICATION == (
    "QWEN35_NATIVE_MTP_TP4_16K_TARGET_KV_OFFLOAD_ESTABLISHED"
)
assert gate.PROMOTION_CLASSIFICATION == "NOT_PROMOTABLE"
assert gate.POLICIES == ("baseline", "native_mtp")
assert gate.BATCH_SIZES == (1, 4)
assert gate.PROMPT_TOKENS == 16384
assert gate.MAX_OUTPUT_TOKENS == 8
assert gate.MAX_PROPOSAL_TOKENS == 4
assert gate.WORLD_SIZE == 4
assert gate.KV_OFFLOAD_GPU_BLOCKS == 68
assert gate.KV_OFFLOAD_LOGICAL_BLOCKS == 640
assert gate.KV_OFFLOAD_BLOCKWISE_BLOCKS == 8
assert gate.BLOCK_SIZE == 256
assert "tp1_authority_sha256" not in gate.RESULT_FIELDS
assert "tp1_output_rows" not in gate.CELL_FIELDS
```

Hash the frozen TP4/4K gate before and after importing the new gate and assert
the digest is unchanged.

- [ ] **Step 2: Run the focused test and observe RED**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  -k 'constants or source_isolation' -q
```

Expected: collection/import failure because the new gate does not exist.

- [x] **Step 3: Implement the independent constants and helper loading**

Load the frozen gate under a private module name:

```python
def _load_frozen_gate():
    path = (
        Path(__file__).resolve().parent
        / "qwen35_native_mtp_tp4_4k_engine_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_qwen35_native_mtp_tp4_4k_frozen_gate",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Define new constants locally. Do not mutate the frozen module's constants and
do not re-export its `validate_cell_result`, `validate_result`, campaign CLI,
or TP1 authority constants.

Reuse only these private helpers by explicit assignment:

```python
_exact_keys = _frozen_gate._exact_keys
_sha256 = _frozen_gate._sha256
_json_sha256 = _frozen_gate._json_sha256
_validate_model_identity = _frozen_gate._validate_model_identity
_validate_token_rows = _frozen_gate._validate_token_rows
_validate_receipts = _frozen_gate._validate_receipts
_validate_cleanup = _frozen_gate._validate_cleanup
atomic_write_json = _frozen_gate.atomic_write_json
```

Define local versions of the frozen gate's
`_validate_selected_tokens`, `_validate_cache_snapshot`,
`_validate_native_rank`, `_validate_baseline_rank`, and
`_validate_rank_snapshots`. Preserve their transaction-ID, committed-length,
rank-agreement, and cleanup checks, but read the new local
`PROMPT_TOKENS=16384` and `MAX_OUTPUT_TOKENS=8` constants. Do not copy the
frozen cell/result validators because those require TP1/4K evidence.

- [x] **Step 4: Define the exact new cell and result shapes**

Use:

```python
CELL_FIELDS = {
    "schema_version",
    "policy",
    "batch_size",
    "world_size",
    "rank_inventory",
    "gpu_indices",
    "prompt_token_count",
    "max_output_tokens",
    "max_proposal_tokens",
    "model_identity",
    "engine_config",
    "prompt_rows",
    "output_rows",
    "rank_snapshots",
    "side_state_receipts",
    "target_kv_receipts",
    "residency_phases",
    "kv_rank_deltas",
    "kv_capacity_rows",
    "runtime_poisoned",
    "cleanup",
}

RESULT_FIELDS = {
    "schema_version",
    "classification",
    "promotion_classification",
    "target_model_manifest_sha256",
    "mtp_checkpoint_manifest_sha256",
    "source_tree_sha256",
    "world_size",
    "rank_inventory",
    "gpu_indices",
    "gpu_process_inventory_before",
    "gpu_process_inventory_after",
    "cells",
    "parity",
    "limitations",
}
```

The result parity object is exactly:

```python
{
    "baseline_native": {
        "b1": True,
        "b4": True,
    },
}
```

There is no TP1 field, digest, output row, or parity claim.

- [x] **Step 5: Run the focused tests and observe GREEN**

Run the command from Step 2.

Expected: selected tests pass.

### Task 3: Enforce Real Movement, Capacity, and Transaction Evidence

**Files:**
- Modify: `tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`
- Modify: `tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`

**Interfaces:**
- Consumes: result fields defined by Task 2.
- Produces fail-closed validation for production movement provenance,
  fixed-capacity target KV, native-MTP proposal history, target transactions,
  residency phases, exact parity, and cleanup.

- [x] **Step 1: Write failing movement and capacity tests**

Create a valid four-cell fixture and independently mutate native batch 4:

```python
cell["kv_rank_deltas"][0]["provenance"] = "synthetic_copy"
cell["kv_rank_deltas"][0]["d2h_copies"] = 0
cell["kv_rank_deltas"][0]["d2h_bytes"] = 0
cell["kv_rank_deltas"][0]["h2d_copies"] = 0
cell["kv_rank_deltas"][0]["h2d_bytes"] = 0
cell["kv_capacity_rows"][0]["gpu_blocks"] = 69
cell["kv_capacity_rows"][0]["logical_blocks"] = 639
cell["kv_capacity_rows"][0]["peak_resident_blocks"] = 69
cell["kv_capacity_rows"][0]["resident_blocks"] = 69
```

Assert specific failures:

```text
KV movement provenance is invalid
native batch-4 requires real target-KV D2H copies
native batch-4 requires real target-KV D2H bytes
native batch-4 requires real target-KV H2D copies
native batch-4 requires real target-KV H2D bytes
target-KV GPU block capacity mismatch
target-KV logical block capacity mismatch
target-KV peak residency exceeds GPU capacity
target-KV resident blocks exceed GPU capacity
```

The positive rule aggregates all four ranks, but every row must have production
provenance and valid non-negative counters.

- [x] **Step 2: Write failing no-false-TP1 and parity tests**

Assert the validator rejects:

- any added `tp1_authority_sha256` result field;
- any added `tp1_output_rows` cell field;
- baseline/native prompt mismatch;
- baseline/native output mismatch; and
- parity objects containing `tp1_tp4_native`.

- [x] **Step 3: Write failing transactional-history tests**

Adapt the realistic TP4/4K bootstrap/proposal fixture to `PROMPT_TOKENS=16384`
and `MAX_OUTPUT_TOKENS=8`.

Require per native sequence:

- one bootstrap proposal-KV transaction with 16,384 staged and materialized
  entries;
- proposal transactions distinguished by transaction ID;
- committed length advancing only by accepted proposal prefixes;
- positive accepted and rejected draft totals;
- target KV operations `prepare`, then `commit`;
- side-state operations `prepare`, `select`, `apply`, then `seal`;
- residency operations `prepare`, `precommit`, then `seal`;
- zero accepted-prefix target replay; and
- zero active transactions, tickets, sequences, and slots after cleanup.

Add one rejection test for each missing or reordered lifecycle.

- [ ] **Step 4: Run the focused tests and observe RED**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  -k 'movement or capacity or tp1 or parity or transaction or residency' -q
```

Expected: failures because the validators are incomplete.

- [x] **Step 5: Implement exact validators**

Use movement keys:

```python
MOVEMENT_KEYS = (
    "h2d_copies",
    "h2d_bytes",
    "d2h_copies",
    "d2h_bytes",
    "copy_waits",
    "evictions",
    "evict_clean",
    "speculative_residency_committed_blocks",
    "speculative_residency_rejected_blocks",
    "speculative_residency_rejected_d2h_copies",
)
```

Normalize each capacity row to:

```python
{
    "rank": rank,
    "provenance": "engine.kv_offload_summaries",
    "gpu_blocks": 68,
    "logical_blocks": 640,
    "resident_blocks": resident_blocks,
    "peak_resident_blocks": peak_resident_blocks,
}
```

Require:

```python
0 <= resident_blocks <= peak_resident_blocks <= 68
```

Require native batch-4 aggregate D2H/H2D copies and bytes to be positive.
Require
`speculative_residency_rejected_d2h_copies == 0` on every rank.

Return normalized cells without inserting TP1 placeholders.

- [x] **Step 6: Run focused and complete gate tests**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py -q
```

Expected: all tests pass.

### Task 4: Add the Real 16K Native-MTP Worker

**Files:**
- Modify: `tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`
- Create: `tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py`

**Interfaces:**
- Consumes: production `LLM`, `EngineSpeculativeRuntime`, native-MTP
  registration, `engine.kv_offload_summaries()`, and Task 3 validators.
- Produces:

```python
def run_policy_cell(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    policy: str,
    batch_size: int,
    dist_port: int,
    master_port: int,
    engine_factory,
    sampling_params_type,
    runtime_type,
    synchronize,
) -> dict: ...
```

- [x] **Step 1: Write the failing Engine-configuration test**

Inject a fake Engine factory and assert these exact arguments:

```python
assert kwargs["tensor_parallel_size"] == 4
assert kwargs["enforce_eager"] is True
assert kwargs["max_model_len"] == 33024
assert kwargs["max_num_batched_tokens"] == 132096
assert kwargs["max_num_prefill_tokens_per_step"] == 1024
assert kwargs["max_num_seqs"] == batch_size
assert kwargs["kvcache_block_size"] == 256
assert kwargs["chunked_prefill_decode_first"] is False
assert kwargs["chunked_prefill_mixed_batch"] is False
assert kwargs["kv_offload_mvp0"] is True
assert kwargs["kv_offload_gpu_blocks"] == 68
assert kwargs["kv_offload_logical_blocks"] == 640
assert kwargs["kv_offload_blockwise_prefill"] is True
assert kwargs["kv_offload_blockwise_decode"] is True
assert kwargs["kv_offload_blockwise_blocks"] == 8
assert kwargs["qwen35_mtp_enabled"] is native
assert kwargs["qwen35_mtp_cuda_graphs"] is False
assert kwargs["qwen35_mtp_max_proposal_tokens"] == 4
```

Assert prompt construction produces exactly 16,384 token IDs per request and
the worker parser has no `--tp1-result` argument.

- [x] **Step 2: Write failing production-evidence tests**

Use fake rank summaries before and after generation. Assert the worker emits:

```python
cell["kv_rank_deltas"][rank]["provenance"] == (
    "engine.kv_offload_summaries"
)
cell["kv_capacity_rows"][rank] == {
    "rank": rank,
    "provenance": "engine.kv_offload_summaries",
    "gpu_blocks": 68,
    "logical_blocks": 640,
    "resident_blocks": 64,
    "peak_resident_blocks": 68,
}
```

Assert `tp1_output_rows` is absent and no TP1 result file is opened.

- [ ] **Step 3: Run the worker tests and observe RED**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  -k 'worker' -q
```

Expected: import failure because the worker does not exist.

- [x] **Step 4: Implement the worker without a TP1 dependency**

Create the worker with the TP4/4K worker's explicit local helpers for
distributed environment setup, checkpoint-manifest resolution, native
registration, rank snapshots, receipt compaction, generation, cleanup, and
CLI parsing. Do not import or invoke its `run_policy_cell` or `main`, because
those require TP1/4K evidence. Omit:

- `load_tp1_output_rows`;
- `tp1_result_path`;
- the `--tp1-result` CLI option;
- all TP1 digest checks; and
- `tp1_output_rows` result fields.

Before generation:

```python
before_rows = engine.kv_offload_summaries(timeout_s=60.0)
```

After generation and before Engine exit:

```python
after_rows = engine.kv_offload_summaries(timeout_s=60.0)
```

Compute counter deltas from the exact Task 3 movement keys. Project capacity
rows from `after_rows`, preserving rank order and production provenance.

Define the residency capture locally:

```python
@contextmanager
def capture_residency_phases(engine):
    captured = []
    original = engine._call_speculative_residency_phase

    def recorded(method_name, ticket_id, *args, **kwargs):
        rows = original(method_name, ticket_id, *args, **kwargs)
        captured.append({
            "ticket_id": ticket_id,
            "operation": kwargs["expected_operation"],
            "status": kwargs["expected_status"],
            "rows": [dict(row) for row in rows],
        })
        return rows

    engine._call_speculative_residency_phase = recorded
    try:
        yield captured
    finally:
        engine._call_speculative_residency_phase = original
```

Continue using the native TP4 worker's real rank snapshots, proposal
lifecycle receipts, target callbacks, and cleanup collection.

- [x] **Step 5: Run worker tests and observe GREEN**

Run the command from Step 3.

Expected: selected tests pass.

- [x] **Step 6: Run frozen worker regressions**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_4k_engine_gate.py \
  tools/test_qwen35_mtp_executor.py -q
```

Expected: the established TP4/4K gate tests remain green.

### Task 5: Add the Campaign Orchestrator and Independent Verifier

**Files:**
- Modify: `tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`
- Modify: `tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`
- Create: `tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`

**Interfaces:**
- Produces:

```python
def run_campaign(
    *,
    model_path: str,
    gpu_indices: tuple[int, ...],
    output_dir: Path,
    dist_port_base: int,
    master_port_base: int,
    repo_root: Path | None = None,
    worker_script: Path | None = None,
    python_executable: str = sys.executable,
    source_files: tuple[str, ...] | None = None,
    verifier=None,
) -> dict: ...

def verify_run(
    run_dir: Path,
    source_root: Path | None = None,
) -> dict: ...
```

- [x] **Step 1: Write failing campaign tests**

Assert the campaign:

- dispatches exactly `baseline:b1`, `baseline:b4`, `native_mtp:b1`, and
  `native_mtp:b4`;
- passes no TP1 argument to the worker;
- binds the target and MTP manifests;
- inventories only the selected four GPUs;
- rejects a changed selected-GPU process inventory;
- ignores unrelated processes on unselected GPUs;
- writes the source manifest before executing cells;
- rejects replay into an existing run directory; and
- writes no `authority` before independent verification succeeds.

- [x] **Step 2: Write failing verifier tamper tests**

Create one valid run fixture, then independently tamper with:

- `result.json`;
- target manifest;
- MTP manifest;
- source-tree digest;
- one bound source file;
- schema or classification;
- engine configuration;
- baseline/native output parity;
- movement provenance;
- D2H/H2D counters;
- peak residency;
- proposal transaction history;
- residency lifecycle; and
- cleanup evidence.

Every tampered case must return:

```python
{
    "classification": "FAIL",
    "failures": [...],
}
```

The untampered fixture returns:

```python
{
    "classification": "PASS",
    "failures": [],
}
```

- [ ] **Step 3: Run campaign and verifier tests and observe RED**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  -k 'campaign or verifier or inventory or tamper' -q
```

Expected: failures because campaign and verifier entry points are missing.

- [x] **Step 4: Implement campaign source binding**

Set `DEFAULT_SOURCE_FILES` to include all production and authority sources
that can affect the result, including:

```text
tinyvllm/config.py
tinyvllm/engine/llm_engine.py
tinyvllm/engine/model_runner.py
tinyvllm/engine/speculative_execution.py
tinyvllm/engine/speculative_residency.py
tinyvllm/engine/qwen35_mtp_executor.py
tinyvllm/engine/proposal_kv_cache.py
tools/qwen35_native_mtp_tp4_4k_engine_gate.py
tools/qwen35_native_mtp_tp4_4k_engine_worker.py
tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py
tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
```

Implement the campaign loop directly. For each batch size in `(1, 4)` and
policy in `("baseline", "native_mtp")`, run:

```python
command = [
    python_executable,
    str(worker_script),
    "--model",
    model_path,
    "--gpu-indices",
    ",".join(str(index) for index in gpu_indices),
    "--policy",
    policy,
    "--batch-size",
    str(batch_size),
    "--dist-port",
    str(dist_port_base + ordinal),
    "--master-port",
    str(master_port_base + ordinal),
    "--out",
    str(cell_path),
]
```

Capture selected-GPU inventory before the first cell and after the last cell
with:

```python
[
    "nvidia-smi",
    "-i",
    ",".join(str(index) for index in gpu_indices),
    "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
    "--format=csv,noheader,nounits",
]
```

Require unchanged source hashes during the campaign, assemble the result with
only target/MTP/source digests, publish into a temporary directory, invoke
the independent verifier, and atomically rename only after verifier PASS.
There is no TP1 path, command-line argument, digest, or result field.

- [x] **Step 5: Implement the independent verifier**

The verifier loads only the new 16K gate module, recomputes result and source
digests, optionally compares the live source root, invokes
`validate_result`, and aggregates failures without accepting partial evidence.

It writes no files itself. The campaign writes `authority` only for verifier
PASS and `authority.failed` otherwise.

- [x] **Step 6: Run the focused and full tests**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py -q
```

Expected: all tests pass.

### Task 6: Add the Bounded Remote Runner

**Files:**
- Modify: `tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py`
- Create: `tools/run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh`

**Interfaces:**
- Consumes: approved checkpoint, remote Python, new gate/worker/verifier, and
  selected idle GPUs.
- Produces: one fresh local artifact directory containing terminal
  `authority` or `authority.failed`.

- [x] **Step 1: Write the failing runner source-contract test**

Assert the script contains:

```text
sitian@10.232.195.203
FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py
verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
qwen35_native_mtp_tp4_16k_target_kv_offload
campaign.status
campaign.pid
campaign.exit_code
authority.failed
REMOTE_COMMAND_RETRY_ATTEMPTS
REMOTE_RSYNC_RETRY_ATTEMPTS
POLL_INTERVAL_SECONDS
```

Assert the script never invokes `pkill`, `killall`, `nvidia-smi --gpu-reset`,
`git clean`, or an unbounded polling loop.

- [ ] **Step 2: Run the runner test and observe RED**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  -k 'remote_runner' -q
```

Expected: failure because the runner does not exist.

- [x] **Step 3: Implement the runner**

The script must:

1. validate the local checkout path;
2. obtain or refresh the authorized Kerberos ticket without printing secrets;
3. select four idle GPUs without killing existing processes;
4. create a fresh opaque run ID;
5. copy the exact source-bound tree to a fresh remote directory;
6. launch one bounded background campaign with recorded PID and status;
7. poll at a fixed interval with a maximum poll count;
8. copy back result, logs, source manifest, and terminal marker;
9. run the independent verifier locally against the copied source tree; and
10. return non-zero unless `authority` exists and contains verifier PASS.

Use serial retries for SSH and rsync. Reuse no ControlMaster socket.

- [x] **Step 4: Run source and shell validation**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  -k 'remote_runner' -q
bash -n \
  tools/run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh
```

Expected: tests pass and `bash -n` exits zero.

### Task 7: Run Local and Remote Regression Gates

**Files:**
- Modify only if a test exposes a scoped defect.

**Interfaces:**
- Consumes all files from Tasks 1–6.
- Produces local pure-Python GREEN evidence and remote Torch GREEN evidence.

- [x] **Step 1: Run pure-Python authority tests locally**

Run:

```bash
python3 -m pytest \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/test_qwen35_native_mtp_tp4_4k_engine_gate.py -q
```

Expected: all collected tests pass.

- [x] **Step 2: Run compilation and formatting checks**

Run:

```bash
python3 -m py_compile \
  tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py \
  tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tinyvllm/engine/model_runner.py
bash -n \
  tools/run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh
git diff --check -- \
  docs/superpowers/specs/2026-08-14-qwen35-native-mtp-tp4-16k-target-kv-offload-design.md \
  docs/superpowers/plans/2026-08-14-qwen35-native-mtp-tp4-16k-target-kv-offload.md \
  tinyvllm/engine/model_runner.py \
  tools/test_kv_offload.py \
  tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py \
  tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
```

Expected: all commands exit zero with no whitespace errors.

- [x] **Step 3: Run focused remote Torch regressions**

Through the bounded source-copy path, run:

```bash
PYTHONPATH=$PWD /data00/home/sitian/sitian-workspace01/tllm/env/bin/python \
  tools/test_kv_offload.py
```

Also run the native-MTP executor tests using the copied pure-Python pytest
runtime if remote `pytest` remains unavailable.

Expected: peak-residency tests, KV-offload regressions, and native-MTP
executor tests pass.

### Task 8: Establish the Fresh TP4/16K Authority

**Files:**
- Create under:
  `artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/<opaque-run-id>/`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes the completed runner and approved checkpoint.
- Produces a source-bound authority artifact and an explicit promotion-boundary handoff.

- [x] **Step 1: Launch one fresh campaign**

Run:

```bash
bash \
  tools/run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh
```

Expected: the runner prints the fresh opaque run directory and eventually
creates exactly one terminal marker.

- [x] **Step 2: Poll the existing campaign instead of launching another**

Reuse the runner's recorded PID/status paths. Do not create a second campaign
while the first is running.

Expected terminal success files:

```text
result.json
source_manifest.json
source_tree/
artifacts/authority
campaign.exit_code
```

- [x] **Step 3: Run independent verification explicitly**

Run:

```bash
python3 \
  tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  --run-dir \
  artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/<opaque-run-id> \
  --source-root \
  artifacts/qwen35_native_mtp_tp4_16k_target_kv_offload/<opaque-run-id>/source_tree
```

Expected:

```json
{"classification":"PASS","failures":[]}
```

- [x] **Step 4: Audit the authority against the design**

Confirm from `result.json` and rank evidence:

- four real TP ranks in every cell;
- baseline/native exact output parity for batch 1 and 4;
- no TP1/4K digest or parity field;
- 16,384-token prompts;
- native proposal, acceptance, and rejection;
- zero accepted-prefix target replay;
- target KV and recurrent-state transactional lifecycle;
- residency prepare/precommit/seal lifecycle;
- production movement provenance on every rank;
- positive native batch-4 D2H/H2D copies and bytes;
- GPU capacity exactly 68 and logical capacity exactly 640;
- `peak_resident_blocks <= 68` on every rank;
- proposal bootstrap and proposal transaction history;
- zero runtime poison;
- complete rank/process/shared-memory cleanup; and
- selected-GPU process inventory unchanged.

Any missing evidence means the authority is not established.

- [x] **Step 5: Update the handoff without overstating scope**

Append to `AGENT_HANDOFF_STATE.md`:

- opaque run ID and authority path;
- exact verifier output;
- local and remote test commands/results;
- target and MTP manifest digests;
- source-tree digest;
- baseline/native parity result;
- proposal acceptance/rejection totals;
- target-KV D2H/H2D copies and bytes;
- per-rank capacity and peak residency;
- cleanup result;
- defects fixed during RED/GREEN work; and
- explicit non-claims: proposal-KV offload, TP1/16K, 32K, performance,
  KV8/KV4, second learned structure, production readiness, and Phase 1.

- [x] **Step 6: Run final scoped verification**

Run:

```bash
git diff --check -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/specs/2026-08-14-qwen35-native-mtp-tp4-16k-target-kv-offload-design.md \
  docs/superpowers/plans/2026-08-14-qwen35-native-mtp-tp4-16k-target-kv-offload.md \
  tinyvllm/engine/model_runner.py \
  tools/test_kv_offload.py \
  tools/qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/qwen35_native_mtp_tp4_16k_target_kv_offload_worker.py \
  tools/verify_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py \
  tools/run_qwen35_native_mtp_tp4_16k_target_kv_offload_remote.sh \
  tools/test_qwen35_native_mtp_tp4_16k_target_kv_offload_gate.py
```

Expected: no output.

Do not stage or commit any file.
