# Qwen3.5 Recurrent Full-Fidelity Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a default-disabled production path that captures one real FP32 recurrent snapshot for every TP rank, canonical workload, and all 18 Qwen3.5 linear recurrent layers, closes rank-owned staging artifacts, and independently publishes the existing immutable `qwen35.recurrent-full-fidelity-bundle.v1`.

**Architecture:** Runtime workers write one tensor at a time into rank-owned staging directories immediately after the source request's final prefill state has committed and before final-prefill dtype rounding, cache encoding, or lease release. Capture runs only for the canonical `exact_restore / correctness / repetition=0` case, arms only the first source request for each workload, and never changes ordinary schema-v1 execution when the capture flag is absent. After all workload workers finish, a CPU-only rank closer validates complete Cartesian coverage and publishes immutable rank manifests; a separate assembler then revalidates bytes and identities and atomically publishes the calibration source bundle.

**Tech Stack:** Python 3.12, PyTorch tensor serialization, SHA-256, canonical JSON, `pathlib`, atomic same-filesystem rename, pytest, existing ModelRunner command acknowledgements.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Do not stage, commit, branch, stash, reset, or clean. The commit steps normally required by `writing-plans` are replaced by scoped diff review checkpoints.
- Do not execute SSH, create remote directories, initialize CUDA, or run GPU workloads while implementing or validating this plan.
- Use `/opt/homebrew/bin/python3.12` for Python and Torch.
- Run pytest with `PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not inject the complete Python 3.9 `site-packages` directory into Python 3.12.
- Capture is absent by default and must leave ordinary schema-v1 behavior and artifacts unchanged.
- Capture eligibility is exact and fail-closed: `policy=exact_restore`, `phase=correctness`, `repetition=0`.
- Within an eligible workload, capture only the first source request and capture its state after the final prefill commit, before `_round_qwen35_final_prefill_recurrent_states()`, cache encoding, eviction, lease release, or generated-token decode.
- Persist each tensor using `detach -> float32 -> contiguous -> CPU -> temporary file -> fsync -> atomic rename`; retain only immutable metadata after publication.
- Never accumulate the full recurrent capture set in GPU memory, CPU memory, or a deferred tensor queue.
- Tensor IDs are exactly `rank{rank}:{workload_id}:layer{layer_index}:linear_recurrent`.
- Persisted tensors are FP32, rank-3, and cover all declared ranks, workload IDs, and exactly the 18 declared linear recurrent layer indices.
- Failed or partial capture roots are not resumable as successful evidence. Final paths and manifests are never overwritten.
- The five canonical workload workers may open the same rank staging root sequentially only when `capture_identity.json` is byte-identical and the requested workload has no tensor or completion artifact yet. A completed workload, any pre-existing tensor for the requested workload, or any leftover temporary file makes that worker fail closed; recovery uses a new capture root.
- The existing strict-P1 benchmark assembler must not publish or consume the recurrent calibration bundle.
- CPU-only tests and synthetic fixtures do not establish real accuracy, speed, cache, VRAM, or capacity improvements.
- A real run remains separately gated by correctness `PASS`, Gate-1 `PASS`, fresh v2 preflight `READY`, strict-P1 resource availability on GPUs `2,4,5,6`, explicit execution approval, calibration `PASS`, and canonical verifier `GO`.

---

## File Structure

### New production files

- `tinyvllm/engine/qwen35_recurrent_capture_contract.py`
  - Defines immutable run identity, tensor-record, and rank-manifest schemas.
  - Performs exact-field, identity, path, hash, shape, dtype, and coverage validation without importing runtime benchmark code.
- `tinyvllm/engine/qwen35_recurrent_capture.py`
  - Owns rank-local staging initialization, one-tensor-at-a-time atomic persistence, duplicate refusal, and workload completion receipts.
- `tools/qwen35_recurrent_full_fidelity_capture_closer.py`
  - CPU-only closer that scans one rank staging directory, validates all expected workload/layer tensors, and atomically publishes `rank_capture_manifest.json`.
- `tools/qwen35_recurrent_full_fidelity_bundle_assembler.py`
  - CPU-only independent assembler that consumes only closed rank manifests and publishes `qwen35.recurrent-full-fidelity-bundle.v1`.

### Modified production files

- `tinyvllm/engine/model_runner.py`
  - Adds a default-`None` capture session, acknowledged configure/arm/finish methods, and one narrow observation call after recurrent-state commit.
- `tinyvllm/engine/llm_engine.py`
  - Adds rank-acknowledged capture configuration, arming, and workload-finish transport.
- `tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`
  - Arms only the first source request and closes only the current workload capture.
- `tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py`
  - Adds `--recurrent-calibration-capture-dir`, exact eligibility checks, and capture configuration forwarding while preserving the no-flag path.

### New test files

- `tools/test_qwen35_recurrent_capture_contract.py`
- `tools/test_qwen35_recurrent_capture.py`
- `tools/test_qwen35_recurrent_full_fidelity_capture_closer.py`
- `tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py`
- `tools/test_model_runner_recurrent_capture.py`

### Modified test files

- `tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`
- `tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py`

---

### Task 1: Freeze the Capture Contract

**Files:**
- Create: `tinyvllm/engine/qwen35_recurrent_capture_contract.py`
- Create: `tools/test_qwen35_recurrent_capture_contract.py`

**Interfaces:**
- Produces:
  - `CAPTURE_IDENTITY_SCHEMA_VERSION = "qwen35.recurrent-capture-identity.v1"`
  - `RANK_CAPTURE_MANIFEST_SCHEMA_VERSION = "qwen35.recurrent-rank-capture.v1"`
  - `CaptureRunIdentity`
  - `CapturedTensorRecord`
  - `RankCaptureManifest`
  - `canonical_json_bytes(value) -> bytes`
  - `canonical_json_sha256(value) -> str`
  - `validate_run_identity(value) -> CaptureRunIdentity`
  - `validate_tensor_record(value, *, identity, expected_rank=None) -> CapturedTensorRecord`
  - `validate_rank_manifest(value, *, expected_identity=None) -> RankCaptureManifest`
  - `expected_tensor_ids(*, world_size, workload_ids, linear_layer_indices) -> tuple[str, ...]`

- [x] **Step 1: Write failing exact-schema tests**

```python
def test_run_identity_and_expected_tensor_ids_are_exact():
    identity = contract.validate_run_identity({
        "schema_version": contract.CAPTURE_IDENTITY_SCHEMA_VERSION,
        "model_manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
        "workload_manifest_sha256": "c" * 64,
        "world_size": 2,
        "workload_ids": ["w0", "w1"],
        "linear_layer_indices": [0, 2],
    })
    assert identity.world_size == 2
    assert contract.expected_tensor_ids(
        world_size=2,
        workload_ids=("w0", "w1"),
        linear_layer_indices=(0, 2),
    ) == (
        "rank0:w0:layer0:linear_recurrent",
        "rank0:w0:layer2:linear_recurrent",
        "rank0:w1:layer0:linear_recurrent",
        "rank0:w1:layer2:linear_recurrent",
        "rank1:w0:layer0:linear_recurrent",
        "rank1:w0:layer2:linear_recurrent",
        "rank1:w1:layer0:linear_recurrent",
        "rank1:w1:layer2:linear_recurrent",
    )
```

Add tests that reject:

```python
@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra=True),
    lambda value: value.update(world_size=True),
    lambda value: value.update(workload_ids=["w0", "w0"]),
    lambda value: value.update(linear_layer_indices=[2, 0]),
    lambda value: value.update(source_tree_sha256="not-a-hash"),
])
def test_run_identity_rejects_unknown_or_noncanonical_fields(mutation):
    value = valid_identity_dict()
    mutation(value)
    with pytest.raises(ValueError):
        contract.validate_run_identity(value)
```

Also assert tensor records require:

```python
{
    "tensor_id": "rank0:w0:layer0:linear_recurrent",
    "rank": 0,
    "workload_id": "w0",
    "layer_index": 0,
    "relative_path": "rank0/tensors/w0/layer0.pt",
    "sha256": "d" * 64,
    "shape": [2, 3, 8],
    "dtype": "float32",
    "logical_bytes": 192,
}
```

- [x] **Step 2: Run the contract tests and verify RED**

Run:

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_recurrent_capture_contract.py -q
```

Expected: collection or import failure because `qwen35_recurrent_capture_contract.py` does not exist.

- [x] **Step 3: Implement immutable records and exact validators**

Use frozen dataclasses and exact dictionaries:

```python
@dataclass(frozen=True)
class CaptureRunIdentity:
    model_manifest_sha256: str
    source_tree_sha256: str
    workload_manifest_sha256: str
    world_size: int
    workload_ids: tuple[str, ...]
    linear_layer_indices: tuple[int, ...]

    def payload(self) -> dict:
        return {
            "schema_version": CAPTURE_IDENTITY_SCHEMA_VERSION,
            "model_manifest_sha256": self.model_manifest_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "workload_manifest_sha256": self.workload_manifest_sha256,
            "world_size": self.world_size,
            "workload_ids": list(self.workload_ids),
            "linear_layer_indices": list(self.linear_layer_indices),
        }
```

Implement `CapturedTensorRecord.payload()` and `RankCaptureManifest.payload()` in the same style. Require sorted unique workload IDs and sorted unique layer indices so canonical ordering is unambiguous. Validate relative paths with `PurePosixPath`; reject absolute paths, `.`/`..`, empty components, backslashes, and any path whose first component is not `rank{rank}`.

- [x] **Step 4: Run the contract tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass.

- [x] **Step 5: Review the scoped diff**

Run:

```bash
git diff --check -- \
  tinyvllm/engine/qwen35_recurrent_capture_contract.py \
  tools/test_qwen35_recurrent_capture_contract.py
```

Expected: no output. Do not stage or commit.

---

### Task 2: Implement One-Tensor-at-a-Time Rank Staging

**Files:**
- Create: `tinyvllm/engine/qwen35_recurrent_capture.py`
- Create: `tools/test_qwen35_recurrent_capture.py`

**Interfaces:**
- Consumes:
  - `CaptureRunIdentity`
  - `CapturedTensorRecord`
  - `validate_run_identity`
- Produces:
  - `Qwen35RecurrentCaptureSession`
  - `capture_recurrent_state(*, run_identity, rank, workload_id, layer_index, tensor, staging_dir=None) -> CapturedTensorRecord`
  - `Qwen35RecurrentCaptureSession.capture_layer(*, workload_id, layer_index, tensor) -> CapturedTensorRecord`
  - `Qwen35RecurrentCaptureSession.finish_workload(workload_id) -> dict`

- [x] **Step 1: Write failing writer tests**

Cover the exact conversion and atomicity sequence:

```python
def test_capture_persists_fp32_contiguous_cpu_tensor_and_metadata(tmp_path):
    identity = capture_identity(world_size=1, workloads=("w0",), layers=(0,))
    session = capture.Qwen35RecurrentCaptureSession(
        run_identity=identity,
        rank=0,
        staging_dir=tmp_path,
    )
    source = torch.arange(
        24, dtype=torch.float16
    ).reshape(2, 3, 4).transpose(1, 2)
    record = session.capture_layer(
        workload_id="w0",
        layer_index=0,
        tensor=source,
    )
    persisted = torch.load(tmp_path / record.relative_path)
    assert persisted.dtype == torch.float32
    assert persisted.device.type == "cpu"
    assert persisted.is_contiguous()
    assert list(persisted.shape) == record.shape
    assert record.logical_bytes == persisted.numel() * 4
```

Add tests that:

- reject duplicate tensor IDs before invoking the serializer;
- allow a new process-local session with a byte-identical identity to capture a different untouched workload in the same rank root;
- reject a second session with a mismatched identity in the same rank root;
- reject reopening a completed workload, a workload with any pre-existing tensor, or a rank root containing leftover temporary files;
- leave no final tensor and no record when serialization, fsync, or rename fails;
- reject symlinked rank roots and pre-existing final paths;
- keep only metadata records, verified by weak-reference collection of the source and converted tensors;
- write `capture_identity.json` atomically and idempotently only for an identical identity;
- publish `workloads/<workload_id>.complete.json` only after every declared layer for that workload exists.

- [x] **Step 2: Run writer tests and verify RED**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_recurrent_capture.py -q
```

Expected: import failure because the writer module does not exist.

- [x] **Step 3: Implement atomic persistence**

The session constructor must normalize the rank root:

```python
self.rank_root = Path(staging_dir) / f"rank{rank}"
self.tensor_root = self.rank_root / "tensors"
self.identity_path = self.rank_root / "capture_identity.json"
```

On construction, scan the existing rank root without loading tensor payloads. Accept only:

- the byte-identical `capture_identity.json`;
- final tensor files belonging exclusively to other completed workloads; and
- their exact `workloads/<workload_id>.complete.json` receipts.

Build the duplicate inventory from existing relative paths and receipt metadata so a fresh worker process cannot overwrite evidence written by an earlier workload worker. If any workload is partial, any temporary file remains, or any path is untracked, fail before accepting a tensor.

The hot path must be structurally equivalent to:

```python
cpu_tensor = tensor.detach().to(
    dtype=torch.float32,
    device="cpu",
).contiguous()
try:
    temporary_path = _new_temporary_path(final_path)
    save_tensor(cpu_tensor, temporary_path)
    _fsync_regular_file(temporary_path)
    payload_sha256 = sha256_file(temporary_path)
    temporary_path.replace(final_path)
    record = CapturedTensorRecord(
        tensor_id=tensor_id,
        rank=self.rank,
        workload_id=workload_id,
        layer_index=layer_index,
        relative_path=final_path.relative_to(self.capture_root).as_posix(),
        sha256=payload_sha256,
        shape=tuple(cpu_tensor.shape),
        dtype="float32",
        logical_bytes=cpu_tensor.numel() * cpu_tensor.element_size(),
    )
finally:
    del cpu_tensor
```

Do not store `tensor`, `cpu_tensor`, or loaded tensor objects on the session. Store only `dict[tensor_id, CapturedTensorRecord]`.

- [x] **Step 4: Run writer tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass.

- [x] **Step 5: Run contract plus writer tests**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_recurrent_capture_contract.py \
  tools/test_qwen35_recurrent_capture.py -q
```

Expected: all tests pass.

- [x] **Step 6: Review the scoped diff**

```bash
git diff --check -- \
  tinyvllm/engine/qwen35_recurrent_capture.py \
  tools/test_qwen35_recurrent_capture.py
```

Expected: no output. Do not stage or commit.

---

### Task 3: Close Rank-Owned Capture Artifacts

**Files:**
- Create: `tools/qwen35_recurrent_full_fidelity_capture_closer.py`
- Create: `tools/test_qwen35_recurrent_full_fidelity_capture_closer.py`

**Interfaces:**
- Consumes:
  - `CaptureRunIdentity`
  - `CapturedTensorRecord`
  - the rank staging layout from Task 2
- Produces:
  - `close_rank_capture(*, staging_dir, expected_workload_ids, expected_linear_layer_indices, load_tensor=torch.load) -> RankCaptureManifest`
  - CLI:

```text
python tools/qwen35_recurrent_full_fidelity_capture_closer.py \
  --capture-root <root> \
  --rank <rank> \
  --expected-workload-id <id> ... \
  --expected-linear-layer-index <index> ...
```

- [x] **Step 1: Write failing rank-closure tests**

Create two workload-complete receipts and tensor payloads, then assert:

```python
manifest = closer.close_rank_capture(
    staging_dir=tmp_path / "rank0",
    expected_workload_ids=("w0", "w1"),
    expected_linear_layer_indices=(0, 2),
)
assert manifest.rank == 0
assert tuple(row.tensor_id for row in manifest.tensors) == (
    "rank0:w0:layer0:linear_recurrent",
    "rank0:w0:layer2:linear_recurrent",
    "rank0:w1:layer0:linear_recurrent",
    "rank0:w1:layer2:linear_recurrent",
)
assert (tmp_path / "rank0/rank_capture_manifest.json").is_file()
```

Add rejection tests for:

- missing or extra workload-complete receipts;
- missing, duplicate, extra, or untracked tensor payloads;
- tampered bytes after capture;
- dtype other than `torch.float32`;
- rank other than three;
- symlinks anywhere below the rank root;
- mismatch between expected workloads/layers and capture identity;
- an existing final rank manifest;
- temporary files being treated as evidence.

- [x] **Step 2: Run rank-closure tests and verify RED**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_recurrent_full_fidelity_capture_closer.py -q
```

Expected: import failure because the closer does not exist.

- [x] **Step 3: Implement byte-authoritative closure**

For every expected tensor:

```python
with payload_path.open("rb") as handle:
    payload = handle.read()
observed_sha256 = hashlib.sha256(payload).hexdigest()
tensor = load_tensor(io.BytesIO(payload), map_location="cpu")
```

Validate the hash and tensor from the same byte snapshot. Do not hash, close, and reopen the path for parsing. Publish `rank_capture_manifest.json` through a temporary sibling and atomic rename only after the entire rank passes.

- [x] **Step 4: Run rank-closure tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass.

- [x] **Step 5: Exercise the CLI with a CPU fixture**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 \
  tools/qwen35_recurrent_full_fidelity_capture_closer.py \
  --help
```

Expected: exit 0 and display all required arguments.

- [x] **Step 6: Review the scoped diff**

```bash
git diff --check -- \
  tools/qwen35_recurrent_full_fidelity_capture_closer.py \
  tools/test_qwen35_recurrent_full_fidelity_capture_closer.py
```

Expected: no output. Do not stage or commit.

---

### Task 4: Assemble the Immutable Full-Fidelity Bundle

**Files:**
- Create: `tools/qwen35_recurrent_full_fidelity_bundle_assembler.py`
- Create: `tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py`

**Interfaces:**
- Consumes:
  - closed `rank_capture_manifest.json` files from Task 3;
  - `SOURCE_BUNDLE_SCHEMA_VERSION` and `validate_source_bundle_manifest()` from `tools/qwen35_recurrent_int8_calibration_contract.py`.
- Produces:
  - `assemble_full_fidelity_bundle(*, capture_root, output_dir, model_manifest_sha256, source_tree_sha256, workload_manifest_sha256, world_size) -> dict`
  - CLI with the same keyword inputs.

- [x] **Step 1: Write failing assembler tests**

Build two closed rank fixtures and assert:

```python
result = assembler.assemble_full_fidelity_bundle(
    capture_root=capture_root,
    output_dir=output_dir,
    model_manifest_sha256="a" * 64,
    source_tree_sha256="b" * 64,
    workload_manifest_sha256="c" * 64,
    world_size=2,
)
manifest = json.loads(
    (output_dir / "source_bundle_manifest.json").read_text()
)
assert manifest["schema_version"] == (
    "qwen35.recurrent-full-fidelity-bundle.v1"
)
calibration_contract.validate_source_bundle_manifest(manifest)
assert result["tensor_count"] == 8
```

Add tests that reject:

- an unclosed rank;
- missing, duplicate, or out-of-range ranks;
- cross-rank identity mismatch;
- false-root or absolute-path rebinding;
- payload tampering after rank closure;
- duplicate tensor IDs or output relative paths;
- an extra file under a closed rank root;
- pre-existing or non-empty output directories;
- copy failure without publishing the final output.

- [x] **Step 2: Run assembler tests and verify RED**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py -q
```

Expected: import failure because the assembler does not exist.

- [x] **Step 3: Implement independent validation and atomic publication**

The assembler must:

```python
expected_ranks = tuple(range(world_size))
rank_manifests = tuple(
    load_and_validate_rank_manifest(
        capture_root / f"rank{rank}" / "rank_capture_manifest.json",
        expected_rank=rank,
    )
    for rank in expected_ranks
)
```

For each payload, read one byte snapshot, validate SHA-256 and tensor metadata from those bytes, then write those exact bytes beneath:

```text
source/rank{rank}/{workload_id}/layer{layer_index}.pt
```

Create the source manifest with only the existing required fields:

```python
manifest = {
    "schema_version": calibration_contract.SOURCE_BUNDLE_SCHEMA_VERSION,
    "model_manifest_sha256": identity.model_manifest_sha256,
    "source_tree_sha256": identity.source_tree_sha256,
    "workload_manifest_sha256": identity.workload_manifest_sha256,
    "world_size": identity.world_size,
    "linear_layer_indices": list(identity.linear_layer_indices),
    "workload_ids": list(identity.workload_ids),
    "tensors": tensor_rows,
}
```

Validate it with `validate_source_bundle_manifest()` before atomically renaming the temporary output directory.

- [x] **Step 4: Run assembler tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass.

- [x] **Step 5: Prove compatibility with the existing calibration producer**

Add an integration test that passes the assembled bundle to:

```python
calibration.run_calibration(
    output_dir,
    calibration_dir,
    thresholds_path=thresholds_path,
    load_tensor=torch.load,
    save_tensor=torch.save,
)
```

Then assert:

```python
verification = verifier.verify_calibration(calibration_dir)
assert verification["classification"] in {"PASS", "NO_GO"}
```

The assertion deliberately excludes `INVALID`; synthetic values do not guarantee `PASS`.

- [x] **Step 6: Run assembler plus existing calibration suites**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/test_qwen35_recurrent_int8_calibration_contract.py \
  tools/test_qwen35_recurrent_int8_calibration.py \
  tools/test_verify_qwen35_recurrent_int8_calibration.py -q
```

Expected: all tests pass.

- [x] **Step 7: Review the scoped diff**

```bash
git diff --check -- \
  tools/qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py
```

Expected: no output. Do not stage or commit.

---

### Task 5: Add the Default-Off ModelRunner Capture Hook

**Files:**
- Modify: `tinyvllm/engine/model_runner.py`
- Create: `tools/test_model_runner_recurrent_capture.py`

**Interfaces:**
- Consumes:
  - `Qwen35RecurrentCaptureSession`
  - `Qwen35HybridModelOwner.state_transaction.adapters`
  - `_last_hybrid_state_leases`
- Produces ModelRunner methods:
  - `configure_qwen35_recurrent_capture(configuration) -> dict`
  - `arm_qwen35_recurrent_capture(workload_id) -> dict`
  - `finish_qwen35_recurrent_capture_workload(workload_id) -> dict`
  - `_capture_qwen35_recurrent_source_state(seqs, *, is_prefill, batch_kind) -> None`

- [x] **Step 1: Write failing unit tests against an uninitialized ModelRunner shell**

Construct with `runner = object.__new__(ModelRunner)` and inject only:

```python
runner.rank = 0
runner.qwen35_hybrid_model_owner = fake_owner
runner.qwen35_recurrent_capture_session = fake_session
runner.qwen35_recurrent_capture_workload_id = "w0"
runner.qwen35_recurrent_capture_armed = True
runner._last_hybrid_state_leases = (lease,)
```

Assert the hook:

- does nothing when the session is `None`;
- does nothing when not armed;
- rejects non-prefill, mixed batches, multiple active leases, or non-final prefill;
- reads exactly `adapter.recurrent[slot_id]` for each declared adapter;
- calls `capture_layer(workload_id="w0", layer_index=..., tensor=...)` once per layer;
- observes the unrounded committed tensor even when `_round_qwen35_final_prefill_recurrent_states()` would subsequently change its values;
- disarms only after every layer succeeds;
- leaves the session armed when any layer write fails;
- never replaces or mutates the recurrent tensor.

Use:

```python
seq = SimpleNamespace(
    hybrid_state_slot_id=3,
    hybrid_state_generation=1,
    prefill_chunk_final=True,
)
```

- [x] **Step 2: Run ModelRunner capture tests and verify RED**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_model_runner_recurrent_capture.py -q
```

Expected: failure because the capture methods are absent.

- [x] **Step 3: Add default-off state and acknowledged lifecycle methods**

Initialize:

```python
self.qwen35_recurrent_capture_session = None
self.qwen35_recurrent_capture_workload_id = None
self.qwen35_recurrent_capture_armed = False
```

`configure_qwen35_recurrent_capture()` must reject reconfiguration and create a rank-local session only after validating:

```python
configuration == {
    "capture_root": str,
    "model_manifest_sha256": str,
    "source_tree_sha256": str,
    "workload_manifest_sha256": str,
    "world_size": int,
    "workload_ids": list[str],
}
```

The ModelRunner must derive:

```python
linear_layer_indices = tuple(
    self.qwen35_hybrid_model_owner.layer_stack.linear_indices
)
```

and require exactly 18 unique sorted indices. It then constructs and validates the complete `CaptureRunIdentity` locally. This prevents a worker-side manifest or CLI argument from declaring a layer inventory different from the model actually bound on that rank.

`arm_qwen35_recurrent_capture(workload_id)` must require a configured session, a declared workload, and no current arm. It returns:

```python
{"rank": self.rank, "workload_id": workload_id, "armed": True}
```

- [x] **Step 4: Insert the observation hook at the exact state boundary**

In `ModelRunner.run()`, call:

```python
self._capture_qwen35_recurrent_source_state(
    seqs,
    is_prefill=is_prefill,
    batch_kind=batch_kind,
)
```

immediately after:

```python
logits = self.run_model(input_ids, positions, is_prefill)
```

and before:

```python
_round_qwen35_final_prefill_recurrent_states(...)
self._kv_offload_after_forward()
```

The hook must execute after `self.model(...)` has committed recurrent candidates to the owner pool and before the existing final-prefill conversion:

```python
recurrent[slot_id].copy_(
    recurrent[slot_id]
    .to(target_dtype)
    .to(recurrent.dtype)
)
```

This conversion may round values even though the pool dtype is FP32, so a post-rounding capture is not full fidelity. The hook must not be added inside `qwen35_linear_attention.py`, the state transaction, or the snapshot cache.

- [x] **Step 5: Run focused ModelRunner tests**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_model_runner_recurrent_capture.py \
  tools/test_qwen35_model_runner_native_entry.py \
  tools/test_qwen35_cross_layer_state_transaction.py \
  tools/test_qwen35_layer_state_adapter.py -q
```

Expected: all tests pass.

- [x] **Step 6: Prove the ordinary run path is unchanged when disabled**

Add a source-inspection or fake-run assertion:

```python
runner.qwen35_recurrent_capture_session = None
runner._capture_qwen35_recurrent_source_state(
    [seq],
    is_prefill=True,
    batch_kind="prefill",
)
assert fake_owner.access_count == 0
```

Run the Step 5 command again.

- [x] **Step 7: Review the scoped diff**

```bash
git diff --check -- \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_recurrent_capture.py
```

Expected: no output. Do not stage or commit.

---

### Task 6: Add Rank-Acknowledged Engine Transport

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_model_runner_recurrent_capture.py`

**Interfaces:**
- Consumes ModelRunner methods from Task 5.
- Produces LLMEngine methods:
  - `configure_qwen35_recurrent_capture(*, capture_root, model_manifest_sha256, source_tree_sha256, workload_manifest_sha256, world_size, workload_ids, timeout_s) -> tuple[dict, ...]`
  - `arm_qwen35_recurrent_capture(workload_id, *, timeout_s) -> tuple[dict, ...]`
  - `finish_qwen35_recurrent_capture_workload(workload_id, *, timeout_s) -> tuple[dict, ...]`

- [x] **Step 1: Write failing transport tests**

Use a fake `call_model_runner_acknowledged()` that returns rank 0 plus worker acknowledgements. Assert each public method:

```python
rows = engine.arm_qwen35_recurrent_capture(
    "w0",
    timeout_s=120.0,
)
assert rows == (
    {"rank": 0, "workload_id": "w0", "armed": True},
    {"rank": 1, "workload_id": "w0", "armed": True},
)
```

Reject:

- missing ranks;
- duplicate ranks;
- wrong workload IDs;
- mixed `armed` or `complete` status;
- malformed result fields;
- a timeout or poisoned acknowledgement collector.

- [x] **Step 2: Run the transport tests and verify RED**

Run:

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_model_runner_recurrent_capture.py -q
```

Expected: failures because LLMEngine transport methods are absent.

- [x] **Step 3: Implement one shared rank-result validator**

Add:

```python
def _collect_qwen35_recurrent_capture_rows(
    self,
    method_name,
    expected_fields,
    *args,
    timeout_s,
):
    local_result, worker_acks = self.call_model_runner_acknowledged(
        method_name,
        *args,
        timeout_s=timeout_s,
    )
    ranked = [(0, local_result)]
    ranked.extend((ack.rank, ack.result) for ack in worker_acks)
    ...
```

Require rank inventory exactly `0..world_size-1`, exact result fields, and equal non-rank values across ranks.

- [x] **Step 4: Run transport and existing acknowledgement tests**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_model_runner_recurrent_capture.py \
  tools/test_model_runner_command_ack.py \
  tools/test_model_runner_live_ack_wiring.py -q
```

Expected: all tests pass.

- [x] **Step 5: Review the scoped diff**

```bash
git diff --check -- \
  tinyvllm/engine/llm_engine.py \
  tools/test_model_runner_recurrent_capture.py
```

Expected: no output. Do not stage or commit.

---

### Task 7: Arm Only the First Source Request in the Engine Adapter

**Files:**
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`
- Modify: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`

**Interfaces:**
- Consumes:
  - optional `configuration["recurrent_calibration_capture"]`;
  - LLMEngine capture transport from Task 6.
- Produces:
  - `_run_source(workload_spec, *, capture_workload_id=None)`
  - exact one-source capture behavior per workload.

- [x] **Step 1: Write failing adapter tests**

Extend `FakeEngine` with:

```python
def configure_qwen35_recurrent_capture(self, **kwargs):
    self.capture_configure_calls.append(kwargs)

def arm_qwen35_recurrent_capture(self, workload_id, *, timeout_s):
    self.capture_arm_calls.append((workload_id, timeout_s))
    return tuple(
        {"rank": rank, "workload_id": workload_id, "armed": True}
        for rank in range(4)
    )

def finish_qwen35_recurrent_capture_workload(
    self, workload_id, *, timeout_s
):
    self.capture_finish_calls.append((workload_id, timeout_s))
    return tuple(
        {"rank": rank, "workload_id": workload_id, "complete": True}
        for rank in range(4)
    )
```

Assert:

- no capture methods are called when the optional configuration is absent;
- capture configuration occurs once during adapter construction when present;
- `w0` through `w3` arm exactly once before `_run_source`;
- `w4_miss_invalidation` still runs three source requests but arms only `request_index == 0`;
- `finish_qwen35_recurrent_capture_workload()` runs only after the captured source request succeeds;
- a source failure does not produce a complete workload receipt;
- continuation requests are never armed.

- [x] **Step 2: Run adapter tests and verify RED**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py -q
```

Expected: new tests fail because capture support is absent.

- [x] **Step 3: Implement optional capture configuration**

In `__init__`:

```python
self.recurrent_calibration_capture = configuration.get(
    "recurrent_calibration_capture"
)
if self.recurrent_calibration_capture is not None:
    self.engine.configure_qwen35_recurrent_capture(
        capture_root=self.recurrent_calibration_capture["capture_root"],
        model_manifest_sha256=(
            self.recurrent_calibration_capture[
                "model_manifest_sha256"
            ]
        ),
        source_tree_sha256=(
            self.recurrent_calibration_capture["source_tree_sha256"]
        ),
        workload_manifest_sha256=(
            self.recurrent_calibration_capture[
                "workload_manifest_sha256"
            ]
        ),
        world_size=self.recurrent_calibration_capture["world_size"],
        workload_ids=self.recurrent_calibration_capture[
            "workload_ids"
        ],
        timeout_s=self.timeout_s,
    )
```

Do not add a default empty dictionary; `None` must preserve the no-capture path.

- [x] **Step 4: Implement exact source arming**

Use:

```python
def _run_source(self, workload_spec, *, capture_workload_id=None):
    if capture_workload_id is not None:
        self.engine.arm_qwen35_recurrent_capture(
            capture_workload_id,
            timeout_s=self.timeout_s,
        )
    self._run_requests([...], record_logits=False)
    if capture_workload_id is not None:
        self.engine.finish_qwen35_recurrent_capture_workload(
            capture_workload_id,
            timeout_s=self.timeout_s,
        )
```

For `w4_miss_invalidation`, pass `capture_workload_id=workload` only when `request_index == 0`.

- [x] **Step 5: Run adapter tests and verify GREEN**

Run the Step 2 command.

Expected: all tests pass.

- [x] **Step 6: Review the scoped diff**

```bash
git diff --check -- \
  tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py
```

Expected: no output. Do not stage or commit.

---

### Task 8: Add the Worker CLI Gate Without Changing Schema-v1

**Files:**
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py`
- Modify: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py`

**Interfaces:**
- Consumes the capture configuration expected by Task 7.
- Produces:
  - CLI `--recurrent-calibration-capture-dir PATH`
  - optional keyword `recurrent_calibration_capture_dir=None` in `run_benchmark_case()`
  - exact capture eligibility validation.

- [x] **Step 1: Write failing worker tests**

Add tests:

```python
def test_capture_flag_adds_only_optional_capture_configuration(tmp_path):
    configuration = worker.build_engine_configuration(
        "exact_restore",
        canonical_correctness_case(),
        recurrent_calibration_capture_dir=tmp_path,
        capture_identity_fields=valid_identity_fields(),
    )
    assert configuration["recurrent_calibration_capture"] == {
        "capture_root": str(tmp_path),
        **valid_identity_fields(),
    }
```

And:

```python
def test_no_capture_flag_preserves_existing_configuration():
    assert worker.build_engine_configuration(
        "exact_restore",
        canonical_correctness_case(),
    ) == existing_expected_configuration()
```

Reject the capture flag for:

- `policy != "exact_restore"`;
- `phase != "correctness"`;
- `repetition != 0`;
- a capture root that exists as a file;
- capture identity fields whose model/source/workload SHA or world size differ from authorized inputs;
- a workload list different from the canonical benchmark `WORKLOADS`.

- [x] **Step 2: Run worker tests and verify RED**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py -q
```

Expected: new tests fail because the flag and optional parameters are absent.

- [x] **Step 3: Add the exact CLI option and identity construction**

Add:

```python
parser.add_argument(
    "--recurrent-calibration-capture-dir",
    type=Path,
)
```

When present, construct the externally bound identity fields:

```python
capture_identity_fields = {
    "model_manifest_sha256": authorized["model_manifest_sha256"],
    "source_tree_sha256": args.source_tree_sha256,
    "workload_manifest_sha256": authorized[
        "workload_manifest_sha256"
    ],
    "world_size": contract.WORLD_SIZE,
    "workload_ids": list(contract.WORKLOADS),
}
```

Do not add a guessed `linear_layer_indices` field to `validate_runtime_artifacts()` or trust a new CLI value. Each rank derives and validates the actual 18-layer inventory from its bound runtime owner in Task 5.

- [x] **Step 4: Keep configuration byte-equivalent when the flag is absent**

Change `build_engine_configuration()` only through:

```python
if recurrent_calibration_capture_dir is not None:
    configuration["recurrent_calibration_capture"] = {
        "capture_root": str(
            Path(recurrent_calibration_capture_dir).resolve()
        ),
        **capture_identity_fields,
    }
```

No capture key may exist otherwise.

- [x] **Step 5: Run worker, adapter, and benchmark contract tests**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_assembler.py -q
```

Expected: all tests pass and existing assembler tests require no recurrent bundle fields.

- [x] **Step 6: Verify CLI help without importing CUDA**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 \
  tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py --help \
  | rg -- '--recurrent-calibration-capture-dir'
```

Expected: one matching option line and exit 0.

- [x] **Step 7: Review the scoped diff**

```bash
git diff --check -- \
  tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py
```

Expected: no output. Do not stage or commit.

---

### Task 9: Run the CPU-Only Integration and Completion Audit

**Files:**
- Modify only if tests expose a defect in Tasks 1-8.
- Update after implementation validation: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes every component from Tasks 1-8.
- Produces:
  - a CPU-only synthetic end-to-end proof;
  - a prompt-to-artifact completion checklist;
  - an explicit statement that real capture and performance claims remain unexecuted.

- [x] **Step 1: Add a synthetic end-to-end integration test**

The test must:

1. create one capture identity with `world_size=2`, two workloads, and two layers;
2. use a fresh `Qwen35RecurrentCaptureSession` per workload, reopening each rank root with the same identity, and write each tensor one at a time;
3. finish each workload;
4. close rank 0 and rank 1 independently;
5. assemble the full-fidelity bundle;
6. validate the manifest with the existing calibration contract;
7. run calibration;
8. run the independent calibration verifier; and
9. assert no temporary files, symlinks, or untracked payloads remain.

Use small CPU tensors:

```python
torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
```

- [x] **Step 2: Run the complete focused CPU suite**

```bash
PYTHONPATH=/tmp/tinyllmforge-pytest312-shim:/Users/bytedance/dev/TinyLLMForge-adaptive-ngram \
  /opt/homebrew/bin/python3.12 -m pytest \
  tools/test_qwen35_recurrent_capture_contract.py \
  tools/test_qwen35_recurrent_capture.py \
  tools/test_qwen35_recurrent_full_fidelity_capture_closer.py \
  tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/test_model_runner_recurrent_capture.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_qwen35_recurrent_int8_calibration_contract.py \
  tools/test_qwen35_recurrent_int8_calibration.py \
  tools/test_verify_qwen35_recurrent_int8_calibration.py -q
```

Expected: all tests pass.

- [x] **Step 3: Run syntax and whitespace validation**

```bash
/opt/homebrew/bin/python3.12 -m py_compile \
  tinyvllm/engine/qwen35_recurrent_capture_contract.py \
  tinyvllm/engine/qwen35_recurrent_capture.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_recurrent_full_fidelity_capture_closer.py \
  tools/qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py

git diff --check -- \
  tinyvllm/engine/qwen35_recurrent_capture_contract.py \
  tinyvllm/engine/qwen35_recurrent_capture.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_recurrent_full_fidelity_capture_closer.py \
  tools/qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_qwen35_recurrent_capture_contract.py \
  tools/test_qwen35_recurrent_capture.py \
  tools/test_qwen35_recurrent_full_fidelity_capture_closer.py \
  tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/test_model_runner_recurrent_capture.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py
```

Expected: both commands exit 0.

- [x] **Step 4: Build the prompt-to-artifact checklist**

Record concrete evidence for:

```text
[ ] default-disabled path has no capture key, calls, files, or schema changes
[ ] eligibility is exact_restore/correctness/repetition=0 only
[ ] first source request only, including one of three w4 source requests
[ ] hook runs after recurrent commit and before final-prefill rounding/cache/offload/release
[ ] detach -> FP32 -> contiguous -> CPU -> atomic file -> release
[ ] rank/workload/18-layer Cartesian coverage
[ ] duplicate, tamper, symlink, traversal, false-root, and partial-output rejection
[ ] rank closure precedes bundle assembly
[ ] assembler output validates as qwen35.recurrent-full-fidelity-bundle.v1
[ ] existing calibration producer accepts the bundle
[ ] independent calibration verifier rejects coordinated or post-publication tamper
[ ] ordinary schema-v1 worker and assembler tests remain green
[ ] no SSH, remote path, CUDA, or GPU operation occurred
[ ] no real accuracy/performance/cache/VRAM claim is made
```

- [x] **Step 5: Update the handoff without overclaiming**

Append a concise section to `AGENT_HANDOFF_STATE.md` containing:

- implemented file list;
- exact test commands and pass counts;
- CPU-only validation boundary;
- remaining real execution gates;
- current strict-P1 resource status if it has not been refreshed, explicitly labeled as the prior snapshot rather than current fact;
- the exact next command sequence for rank closure and bundle assembly, using placeholders for future approved real paths.

Do not write that calibration passed on real data or that speed/cache/VRAM improved.

- [x] **Step 6: Review all scoped changes**

```bash
git status --short
git diff --stat -- \
  tinyvllm/engine/qwen35_recurrent_capture_contract.py \
  tinyvllm/engine/qwen35_recurrent_capture.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/qwen35_recurrent_full_fidelity_capture_closer.py \
  tools/qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  tools/test_qwen35_recurrent_capture_contract.py \
  tools/test_qwen35_recurrent_capture.py \
  tools/test_qwen35_recurrent_full_fidelity_capture_closer.py \
  tools/test_qwen35_recurrent_full_fidelity_bundle_assembler.py \
  tools/test_model_runner_recurrent_capture.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_worker.py \
  AGENT_HANDOFF_STATE.md
```

Expected: only intended files are reviewed; unrelated existing worktree changes remain untouched. Do not stage or commit.

---

## Post-Implementation Real-Execution Gate

Completing Tasks 1-9 does not authorize or complete real validation. A separate approved execution phase must:

1. confirm correctness prerequisite `PASS`;
2. confirm Gate-1 `PASS`;
3. generate a fresh v2 preflight and require `READY`;
4. verify GPUs `2,4,5,6` are available without killing unrelated processes;
5. obtain explicit approval for SSH, remote directory creation, and GPU execution;
6. execute the five canonical capture workers with the shared capture root;
7. close all four rank staging directories;
8. assemble the real source bundle;
9. run the existing CPU calibration and require classification `PASS`;
10. disable capture and execute canonical P2;
11. require the canonical independent verifier classification `GO`; and
12. only then report accuracy, speed, cache, VRAM, or capacity conclusions.

If preflight returns `BLOCKED_RESOURCES`, stop without creating a remote path or launching a worker.
