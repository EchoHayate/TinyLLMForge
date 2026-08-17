# Qwen3.5 Recurrent INT8 Codec Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a default-off, dependency-light Qwen3.5 recurrent-state INT8 reference codec plus a closed-schema offline calibration protocol and independent verifier, without changing hybrid-prefix runtime behavior.

**Architecture:** A focused codec module owns deterministic per-row symmetric INT8 encode/decode and exact byte accounting. A standard-library calibration contract freezes source-bundle identity, row schemas, thresholds, artifact inventory, and classifications. A producer evaluates immutable full-fidelity recurrent tensors through injected tensor I/O, while an independent verifier reconstructs hashes, errors, byte ratios, and the final classification without trusting producer summaries. Real snapshot capture, remote canonical execution, cache integration, and the P2 runtime benchmark remain separate later plans gated on the current fresh source-bound correctness campaign and the P1 exact-restore authority; the active correctness configuration is attempt19 because attempt18 was superseded by a source-tree change.

**Tech Stack:** Python 3, PyTorch for codec/producer tensor operations, standard library for contracts and artifact verification, JSON/JSONL, SHA-256.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Do not stage, commit, merge, create a branch, or create a PR.
- Do not modify `Qwen35HybridPrefixSnapshotCache`, Scheduler, Engine, ModelRunner, publication, restore, or live state-pool behavior in this plan.
- Do not add a Config flag or enable the codec in production.
- Do not launch a GPU model worker or remote canonical calibration in this plan.
- Preserve exact BF16 convolution state; only FP32 recurrent tensors are codec inputs.
- Use codec identity `qwen35_recurrent_symmetric_int8_per_row_v1`.
- Quantized values use symmetric range `[-127, 127]`; `-128` is forbidden.
- Use one FP32 scale for each `recurrent[head, value_row, :]` row.
- Decode all candidate layers before any future live-pool transaction; this plan does not perform a live-pool transaction.
- Distinguish full-fidelity logical bytes, INT8 payload bytes, FP32 scale bytes, encoded physical bytes, and temporary workspace bytes.
- Treat malformed values, NaN, infinity, shape drift, dtype drift, undeclared saturation, missing rows, and accounting mismatch as fail-closed.
- Do not claim production cache reduction, GPU-memory savings, quality retention, latency, throughput, or accuracy improvement from this plan.
- The static TP4 estimate remains reporting-only: `4,939,776 -> 1,437,696` bytes per rank, or `3.4358974359x`.
- Real canonical calibration remains blocked until attempt19 correctness and the strict P1 exact-restore benchmark are authoritative.

---

## File Map

- Create `tinyvllm/engine/qwen35_recurrent_int8_codec.py`: immutable encoded-tensor type, deterministic reference encode/decode, validation, metrics, and byte accounting.
- Create `tools/test_qwen35_recurrent_int8_codec.py`: dependency-light CPU codec RED/GREEN suite and malformed-input coverage.
- Create `tools/qwen35_recurrent_int8_calibration_contract.py`: pure-standard-library schema, source-bundle manifest, threshold profile, artifact inventory, row validation, and classification.
- Create `tools/test_qwen35_recurrent_int8_calibration_contract.py`: exact constants, schema closure, threshold, matrix, and classification tests.
- Create `tools/qwen35_recurrent_int8_calibration.py`: offline producer over immutable tensor files, injected tensor I/O, raw row generation, and atomic artifact writing.
- Create `tools/test_qwen35_recurrent_int8_calibration.py`: synthetic tensor-bundle producer tests with no CUDA initialization.
- Create `tools/verify_qwen35_recurrent_int8_calibration.py`: independent closed-schema artifact/hash/metric verifier and Markdown report writer.
- Create `tools/test_verify_qwen35_recurrent_int8_calibration.py`: complete valid fixture plus tamper and threshold rejection.
- Modify `AGENT_HANDOFF_STATE.md`: implementation status, commands, results, blocked canonical boundary, and next plan.

## Public Interfaces

Task implementations must use these exact names and signatures.

```python
# tinyvllm/engine/qwen35_recurrent_int8_codec.py

QWEN35_RECURRENT_INT8_CODEC = (
    "qwen35_recurrent_symmetric_int8_per_row_v1"
)

@dataclass(frozen=True)
class Qwen35EncodedRecurrentInt8:
    codec: str
    values: torch.Tensor
    scales: torch.Tensor
    source_shape: tuple[int, int, int]
    source_dtype: torch.dtype
    logical_bytes: int
    payload_bytes: int
    scale_bytes: int
    encoded_bytes: int

def encode_qwen35_recurrent_int8_per_row(
    recurrent: torch.Tensor,
) -> Qwen35EncodedRecurrentInt8

def decode_qwen35_recurrent_int8_per_row(
    encoded: Qwen35EncodedRecurrentInt8,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor

def qwen35_recurrent_int8_error_metrics(
    source: torch.Tensor,
    decoded: torch.Tensor,
) -> dict[str, int | float]
```

```python
# tools/qwen35_recurrent_int8_calibration_contract.py

SCHEMA_VERSION = "qwen35.recurrent-int8-calibration.v1"
SOURCE_BUNDLE_SCHEMA_VERSION = (
    "qwen35.recurrent-full-fidelity-bundle.v1"
)
CODEC_ID = "qwen35_recurrent_symmetric_int8_per_row_v1"

@dataclass(frozen=True)
class CalibrationThresholds:
    max_abs_error: float
    relative_l2_error: float
    cosine_similarity: float
    minimum_compression_ratio: float

def build_expected_tensor_ids(
    *,
    world_size: int,
    workload_ids: tuple[str, ...],
    linear_layer_indices: tuple[int, ...],
) -> tuple[str, ...]

def validate_source_bundle_manifest(
    manifest: Mapping[str, object],
) -> tuple[str, ...]

def validate_calibration_row(row: Mapping[str, object]) -> tuple[str, ...]

def classify_calibration(
    rows: tuple[Mapping[str, object], ...],
    *,
    expected_tensor_ids: tuple[str, ...],
    thresholds: CalibrationThresholds,
) -> tuple[str, tuple[str, ...]]
```

```python
# tools/qwen35_recurrent_int8_calibration.py

def run_calibration(
    source_bundle_dir: Path,
    output_dir: Path,
    *,
    thresholds_path: Path,
    load_tensor: Callable[[Path], torch.Tensor],
    save_tensor: Callable[[torch.Tensor, Path], None],
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, object]
```

```python
# tools/verify_qwen35_recurrent_int8_calibration.py

def verify_calibration(
    run_dir: Path,
) -> dict[str, object]
```

## Frozen Calibration Row

Every raw tensor row has exactly these fields:

```text
tensor_id
rank
workload_id
layer_index
source_path
source_sha256
source_shape
source_dtype
codec
encoded_values_path
encoded_values_sha256
encoded_values_shape
encoded_values_dtype
scales_path
scales_sha256
scales_shape
scales_dtype
decoded_path
decoded_sha256
decoded_shape
decoded_dtype
logical_bytes
payload_bytes
scale_bytes
encoded_bytes
compression_ratio
zero_row_count
saturation_count
max_abs_error
mean_abs_error
rmse
relative_l2_error
cosine_similarity
encode_ns
decode_ns
finite_source
finite_scales
finite_decoded
```

The first implementation accepts rank-three FP32 recurrent tensors only.
Canonical Qwen3.5 TP4 tensors have shape `[4, 128, 128]`, but the reference
codec tests may use smaller positive dimensions.

---

### Task 1: Implement the Deterministic Reference Codec

**Files:**
- Create: `tinyvllm/engine/qwen35_recurrent_int8_codec.py`
- Create: `tools/test_qwen35_recurrent_int8_codec.py`

**Interfaces:**
- Produces: `Qwen35EncodedRecurrentInt8`, `encode_qwen35_recurrent_int8_per_row()`, `decode_qwen35_recurrent_int8_per_row()`, and `qwen35_recurrent_int8_error_metrics()`.
- Consumed by: Tasks 3 and 4.

- [ ] **Step 1: Write the import and exact-metadata RED test**

Create the same dependency-light package loading pattern used by
`tools/test_qwen35_hybrid_prefix_cache.py`, then add:

```python
def test_encode_returns_exact_per_row_int8_metadata():
    source = torch.tensor(
        [[[
            -2.0, -1.0, 0.0, 2.0,
        ], [
            0.0, 0.0, 0.0, 0.0,
        ]]],
        dtype=torch.float32,
    )
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    assert encoded.codec == QWEN35_RECURRENT_INT8_CODEC
    assert encoded.values.shape == source.shape
    assert encoded.values.dtype == torch.int8
    assert encoded.scales.shape == source.shape[:-1]
    assert encoded.scales.dtype == torch.float32
    assert encoded.source_shape == tuple(source.shape)
    assert encoded.source_dtype == torch.float32
    assert encoded.logical_bytes == source.numel() * 4
    assert encoded.payload_bytes == source.numel()
    assert encoded.scale_bytes == source.shape[0] * source.shape[1] * 4
    assert encoded.encoded_bytes == (
        encoded.payload_bytes + encoded.scale_bytes
    )
    assert encoded.values.min().item() >= -127
    assert encoded.values.max().item() <= 127
    assert encoded.scales[0, 1].item() == 1.0
    assert torch.count_nonzero(encoded.values[0, 1]).item() == 0
```

- [ ] **Step 2: Run the codec test and confirm RED**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_codec.py
```

Expected: import failure because
`tinyvllm/engine/qwen35_recurrent_int8_codec.py` does not exist.

- [ ] **Step 3: Implement immutable validation and encoding**

Implement:

```python
from dataclasses import dataclass
import math
import torch

QWEN35_RECURRENT_INT8_CODEC = (
    "qwen35_recurrent_symmetric_int8_per_row_v1"
)

@dataclass(frozen=True)
class Qwen35EncodedRecurrentInt8:
    codec: str
    values: torch.Tensor
    scales: torch.Tensor
    source_shape: tuple[int, int, int]
    source_dtype: torch.dtype
    logical_bytes: int
    payload_bytes: int
    scale_bytes: int
    encoded_bytes: int

def _validate_source(recurrent):
    if not isinstance(recurrent, torch.Tensor):
        raise ValueError("recurrent must be a tensor")
    if recurrent.ndim != 3:
        raise ValueError("recurrent must be rank three")
    if any(dimension <= 0 for dimension in recurrent.shape):
        raise ValueError("recurrent dimensions must be positive")
    if recurrent.dtype != torch.float32:
        raise ValueError("recurrent must use torch.float32")
    if not torch.isfinite(recurrent).all().item():
        raise ValueError("recurrent must contain only finite values")

def encode_qwen35_recurrent_int8_per_row(recurrent):
    _validate_source(recurrent)
    source = recurrent.detach().clone().contiguous()
    amax = source.abs().amax(dim=-1)
    scales = torch.where(
        amax == 0,
        torch.ones_like(amax, dtype=torch.float32),
        amax / 127.0,
    ).contiguous()
    values = torch.round(source / scales.unsqueeze(-1))
    values = values.clamp(-127, 127).to(torch.int8).contiguous()
    if torch.any(values == -128).item():
        raise RuntimeError("encoded recurrent contains forbidden -128")
    logical_bytes = source.numel() * source.element_size()
    payload_bytes = values.untyped_storage().nbytes()
    scale_bytes = scales.untyped_storage().nbytes()
    return Qwen35EncodedRecurrentInt8(
        codec=QWEN35_RECURRENT_INT8_CODEC,
        values=values,
        scales=scales,
        source_shape=tuple(source.shape),
        source_dtype=source.dtype,
        logical_bytes=logical_bytes,
        payload_bytes=payload_bytes,
        scale_bytes=scale_bytes,
        encoded_bytes=payload_bytes + scale_bytes,
    )
```

The returned tensors must not alias the source tensor.

- [ ] **Step 4: Add decode RED tests**

Add:

```python
def test_decode_returns_private_finite_fp32_tensor():
    source = torch.arange(
        2 * 3 * 8,
        dtype=torch.float32,
    ).reshape(2, 3, 8) - 17.0
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    decoded = decode_qwen35_recurrent_int8_per_row(encoded)
    assert decoded.shape == source.shape
    assert decoded.dtype == torch.float32
    assert decoded.device == source.device
    assert torch.isfinite(decoded).all().item()
    assert decoded.data_ptr() != source.data_ptr()
    assert decoded.data_ptr() != encoded.values.data_ptr()

def test_decode_can_target_an_explicit_device():
    source = torch.ones((1, 2, 4), dtype=torch.float32)
    encoded = encode_qwen35_recurrent_int8_per_row(source)
    decoded = decode_qwen35_recurrent_int8_per_row(
        encoded,
        device="cpu",
    )
    assert decoded.device.type == "cpu"
```

Run the focused test and expect failure because the decoder is absent.

- [ ] **Step 5: Implement fail-closed decoding**

Implement exact validation:

```python
def _validate_encoded(encoded):
    if type(encoded) is not Qwen35EncodedRecurrentInt8:
        raise ValueError(
            "encoded must be an exact Qwen35EncodedRecurrentInt8"
        )
    if encoded.codec != QWEN35_RECURRENT_INT8_CODEC:
        raise ValueError("encoded codec identity mismatch")
    if encoded.values.dtype != torch.int8:
        raise ValueError("encoded values must use torch.int8")
    if tuple(encoded.values.shape) != encoded.source_shape:
        raise ValueError("encoded values shape mismatch")
    if encoded.scales.dtype != torch.float32:
        raise ValueError("encoded scales must use torch.float32")
    if tuple(encoded.scales.shape) != encoded.source_shape[:-1]:
        raise ValueError("encoded scales shape mismatch")
    if encoded.source_dtype != torch.float32:
        raise ValueError("encoded source dtype must be torch.float32")
    if not torch.isfinite(encoded.scales).all().item():
        raise ValueError("encoded scales must be finite")
    if not torch.all(encoded.scales > 0).item():
        raise ValueError("encoded scales must be positive")
    if torch.any(encoded.values == -128).item():
        raise ValueError("encoded values contain forbidden -128")
    payload_bytes = encoded.values.untyped_storage().nbytes()
    scale_bytes = encoded.scales.untyped_storage().nbytes()
    if encoded.payload_bytes != payload_bytes:
        raise ValueError("encoded payload byte accounting mismatch")
    if encoded.scale_bytes != scale_bytes:
        raise ValueError("encoded scale byte accounting mismatch")
    if encoded.encoded_bytes != payload_bytes + scale_bytes:
        raise ValueError("encoded total byte accounting mismatch")
    logical_bytes = math.prod(encoded.source_shape) * 4
    if encoded.logical_bytes != logical_bytes:
        raise ValueError("encoded logical byte accounting mismatch")

def decode_qwen35_recurrent_int8_per_row(encoded, *, device=None):
    _validate_encoded(encoded)
    target = encoded.values.device if device is None else torch.device(device)
    values = encoded.values.to(device=target, dtype=torch.float32)
    scales = encoded.scales.to(device=target, dtype=torch.float32)
    decoded = (values * scales.unsqueeze(-1)).contiguous()
    if tuple(decoded.shape) != encoded.source_shape:
        raise RuntimeError("decoded recurrent shape mismatch")
    if not torch.isfinite(decoded).all().item():
        raise RuntimeError("decoded recurrent must be finite")
    return decoded
```

- [ ] **Step 6: Add malformed input and clone-isolation tests**

Cover exact errors for:

- Python non-tensor input;
- rank-two and rank-four input;
- FP16 and BF16 input;
- NaN and positive/negative infinity;
- wrong codec string;
- non-int8 values;
- wrong values shape;
- non-FP32 scales;
- wrong scale shape;
- zero, negative, NaN, or infinite scale;
- explicit `-128`;
- logical, payload, scale, or encoded byte mismatch;
- source mutation after encode;
- encoded mutation after decode does not mutate an already returned decoded
  tensor.

Use `dataclasses.replace()` to build malformed immutable records.

- [ ] **Step 7: Add deterministic error metrics**

Add tests:

```python
def test_error_metrics_are_recomputed_in_float64():
    source = torch.tensor(
        [[[0.0, 1.0, -2.0, 3.0]]],
        dtype=torch.float32,
    )
    decoded = source + torch.tensor(
        [[[0.0, 0.25, -0.5, 0.75]]],
        dtype=torch.float32,
    )
    metrics = qwen35_recurrent_int8_error_metrics(source, decoded)
    assert metrics["element_count"] == 4
    assert metrics["finite_source"] is True
    assert metrics["finite_decoded"] is True
    assert metrics["max_abs_error"] == 0.75
    assert metrics["mean_abs_error"] == 0.375
    assert metrics["rmse"] > 0
    assert metrics["relative_l2_error"] > 0
    assert -1.0 <= metrics["cosine_similarity"] <= 1.0
```

Implement calculations after converting both tensors to CPU float64. Require
identical shape, FP32 dtype, and finite values. Define zero-norm behavior:

```text
both source and decoded norm zero -> cosine 1.0, relative L2 0.0
source norm zero and decoded norm nonzero -> relative L2 infinity and reject
```

The public function must reject the second condition rather than emit a
non-finite metric.

- [ ] **Step 8: Run the complete codec suite**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_codec.py
python3 -m py_compile \
  tinyvllm/engine/qwen35_recurrent_int8_codec.py \
  tools/test_qwen35_recurrent_int8_codec.py
```

Expected:

```text
qwen35 recurrent int8 codec tests passed
```

No CUDA initialization is allowed in the CPU test process.

---

### Task 2: Freeze the Calibration Contract and Classification

**Files:**
- Create: `tools/qwen35_recurrent_int8_calibration_contract.py`
- Create: `tools/test_qwen35_recurrent_int8_calibration_contract.py`

**Interfaces:**
- Consumes: codec identity and the frozen row fields in this plan.
- Produces: schema constants, expected tensor IDs, source-bundle validation,
  row validation, thresholds, artifact inventory, and classification.
- Consumed by: Tasks 3 and 4.

- [ ] **Step 1: Write exact constant and artifact-inventory RED tests**

Require:

```python
SCHEMA_VERSION = "qwen35.recurrent-int8-calibration.v1"
SOURCE_BUNDLE_SCHEMA_VERSION = (
    "qwen35.recurrent-full-fidelity-bundle.v1"
)
CODEC_ID = "qwen35_recurrent_symmetric_int8_per_row_v1"
TOP_LEVEL_ARTIFACTS = (
    "source_bundle_manifest.json",
    "thresholds.json",
    "commands.json",
    "calibration_rows.jsonl",
    "summary.json",
    "artifact_manifest.json",
    "independent_verification.json",
    "report.md",
)
NESTED_ARTIFACT_DIRECTORIES = (
    "source",
    "encoded_values",
    "scales",
    "decoded",
)
```

The artifact manifest hashes all producer inputs and outputs except itself,
`independent_verification.json`, and `report.md`.

- [ ] **Step 2: Run contract tests and confirm RED**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_calibration_contract.py
```

Expected: import failure because the contract does not exist.

- [ ] **Step 3: Implement pure canonical JSON and hash helpers**

Implement with standard library only:

```python
def canonical_json_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")

def canonical_json_sha256(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

No Torch or TinyLLMForge import is permitted in this module.

- [ ] **Step 4: Freeze source-bundle schema**

Require exact top-level fields:

```text
schema_version
model_manifest_sha256
source_tree_sha256
workload_manifest_sha256
world_size
linear_layer_indices
workload_ids
tensors
```

Every tensor row contains exactly:

```text
tensor_id
rank
workload_id
layer_index
relative_path
sha256
shape
dtype
logical_bytes
```

Validation rules:

- schema equals `SOURCE_BUNDLE_SCHEMA_VERSION`;
- `world_size` is a positive integer;
- ranks are exactly contiguous `0..world_size-1`;
- `linear_layer_indices` and `workload_ids` are non-empty and unique;
- tensor IDs are exactly
  `rank{rank}:{workload_id}:layer{layer_index}:linear_recurrent`;
- paths are normalized relative POSIX paths below `source/`;
- paths contain no `..`, absolute prefix, or duplicate;
- hashes are lowercase 64-character hexadecimal;
- shape is exactly three positive integers;
- dtype is exactly `float32`;
- logical bytes equal `product(shape) * 4`;
- the tensor set exactly equals `build_expected_tensor_ids(...)`;
- unknown fields are rejected.

- [ ] **Step 5: Freeze threshold schema**

`thresholds.json` has exactly:

```text
schema_version
codec
pilot_source_bundle_sha256
max_abs_error
relative_l2_error
cosine_similarity
minimum_compression_ratio
```

Use a distinct schema:

```python
THRESHOLD_SCHEMA_VERSION = (
    "qwen35.recurrent-int8-calibration-thresholds.v1"
)
```

All numeric thresholds must be finite. Require:

```text
max_abs_error > 0
relative_l2_error > 0
-1 <= cosine_similarity <= 1
minimum_compression_ratio > 1
```

This task defines validation only. It does not choose canonical numerical
values before a read-only pilot exists.

- [ ] **Step 6: Implement closed calibration-row validation**

Use the exact field list under “Frozen Calibration Row”. Validate:

- exact tensor identity fields;
- normalized relative paths;
- lower-case SHA-256 strings;
- source/values/decoded rank-three shapes match;
- scales shape equals source shape without the final dimension;
- source and decoded dtype are `float32`;
- encoded values dtype is `int8`;
- scales dtype is `float32`;
- codec identity matches;
- all byte values are non-negative integers;
- `logical_bytes == product(source_shape) * 4`;
- `payload_bytes == product(source_shape)`;
- `scale_bytes == product(scales_shape) * 4`;
- `encoded_bytes == payload_bytes + scale_bytes`;
- `compression_ratio == logical_bytes / encoded_bytes` within `1e-12`;
- `zero_row_count` and `saturation_count` are bounded by row/element counts;
- error and timing metrics are finite and non-negative;
- cosine is within `[-1, 1]`;
- all three finite flags are exact booleans.

- [ ] **Step 7: Add classifier RED/GREEN tests**

Use exact classifications:

```text
PASS
NO_GO
INVALID
```

Classification rules:

- `INVALID` for schema, identity, completeness, finite-value, hash,
  accounting, duplicate, missing, or unexpected-row failure;
- `NO_GO` when structurally valid rows exceed `max_abs_error`, exceed
  `relative_l2_error`, fall below `cosine_similarity`, or fall below
  `minimum_compression_ratio`;
- `PASS` only when every expected tensor row is present exactly once and all
  thresholds pass.

`saturation_count` counts values equal to `-127` or `127`; it is reported and
valid because max-absolute scaling normally maps at least one non-zero source
element per non-zero row to an endpoint. `-128` is not representable in the
row schema and is rejected by codec/artifact verification.

- [ ] **Step 8: Run the complete contract suite**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_calibration_contract.py
python3 -m py_compile \
  tools/qwen35_recurrent_int8_calibration_contract.py \
  tools/test_qwen35_recurrent_int8_calibration_contract.py
```

Expected:

```text
qwen35 recurrent int8 calibration contract tests passed
```

---

### Task 3: Build the Offline Calibration Producer

**Files:**
- Create: `tools/qwen35_recurrent_int8_calibration.py`
- Create: `tools/test_qwen35_recurrent_int8_calibration.py`

**Interfaces:**
- Consumes: Task 1 codec and Task 2 contract.
- Produces: immutable copied source tensors, encoded values, scales, decoded
  tensors, raw calibration rows, summary, commands, and artifact manifest.
- Does not produce: independent verification or report.

- [ ] **Step 1: Build a synthetic immutable source bundle fixture**

In the test, create:

```text
source_bundle/
  source_bundle_manifest.json
  source/
    rank0/
      w1/
        layer0.pt
        layer1.pt
    rank1/
      w1/
        layer0.pt
        layer1.pt
```

Use tensors shaped `[2, 3, 8]`, dtype FP32, with:

- one all-zero row;
- positive and negative extrema;
- distinct rank/layer values;
- deterministic SHA-256;
- exact manifest logical bytes.

Create a valid test-only `thresholds.json` with permissive finite values and
the SHA-256 of the source manifest as `pilot_source_bundle_sha256`.

- [ ] **Step 2: Write the producer RED test**

Require:

```python
result = run_calibration(
    source_bundle,
    output_dir,
    thresholds_path=thresholds,
    load_tensor=torch.load,
    save_tensor=lambda tensor, path: torch.save(tensor, path),
    clock_ns=deterministic_clock,
)
assert result["classification"] in {"PASS", "NO_GO"}
assert result["row_count"] == 4
assert not (
    output_dir / "independent_verification.json"
).exists()
assert not (output_dir / "report.md").exists()
```

Require exact copied source, encoded, scale, decoded, row, summary, command,
and artifact-manifest inventories.

- [ ] **Step 3: Run producer tests and confirm RED**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_calibration.py
```

Expected: import failure because the producer does not exist.

- [ ] **Step 4: Implement source-bundle preflight before output creation**

`run_calibration()` must:

1. resolve all input paths;
2. reject output equal to or nested inside the source bundle;
3. reject an existing non-empty output directory;
4. load and closed-schema validate the source manifest;
5. hash every declared source tensor;
6. reject missing, extra, symlink, non-regular, or hash-mismatched source
   files;
7. validate thresholds and bind them to the source manifest hash;
8. only then create the output directory.

An input preflight failure must leave no output directory.

- [ ] **Step 5: Implement one-row production**

For each tensor in sorted `tensor_id` order:

```python
source = load_tensor(source_path)
encoded = encode_qwen35_recurrent_int8_per_row(source)
decoded = decode_qwen35_recurrent_int8_per_row(encoded, device="cpu")
metrics = qwen35_recurrent_int8_error_metrics(source, decoded)
```

Measure encode and decode using separate injected `clock_ns()` calls. Save
CPU-contiguous tensors only:

```text
source/<rank>/<workload>/<layer>.pt
encoded_values/<rank>/<workload>/<layer>.pt
scales/<rank>/<workload>/<layer>.pt
decoded/<rank>/<workload>/<layer>.pt
```

Copy the source tensor through `save_tensor()` rather than linking it.

Compute:

```python
zero_row_count = int(
    torch.all(source == 0, dim=-1).sum().item()
)
saturation_count = int(
    torch.logical_or(
        encoded.values == -127,
        encoded.values == 127,
    ).sum().item()
)
compression_ratio = (
    encoded.logical_bytes / encoded.encoded_bytes
)
```

- [ ] **Step 6: Implement atomic producer artifacts**

Write files to sibling temporary paths and replace atomically:

- `source_bundle_manifest.json`: exact copy of the validated input manifest;
- `thresholds.json`: exact canonical validated thresholds;
- `commands.json`: producer argv, Python version, Torch version, codec,
  start/end timestamps, and no user secrets;
- `calibration_rows.jsonl`: canonical JSON rows in sorted tensor order;
- `summary.json`: producer aggregation and preliminary classification;
- `artifact_manifest.json`: relative path, byte size, and SHA-256 for every
  hashed producer/input artifact.

Do not include absolute local usernames or SSH credentials in artifacts.

- [ ] **Step 7: Add injected-failure and no-partial-success tests**

Cover:

- source hash mismatch before output creation;
- unknown source file;
- source symlink;
- existing non-empty output;
- loader exception on the second tensor;
- encoder exception on the second tensor;
- save exception after values but before scales;
- clock returning decreasing timestamps;
- malformed thresholds;
- threshold manifest binding mismatch.

On mid-run failure, preserve the run directory and write:

```text
failure.json
```

with exact completed tensor IDs and the exception type/message. Do not write
`summary.json` or `artifact_manifest.json`, and do not classify partial output
as `PASS` or `NO_GO`.

- [ ] **Step 8: Add producer non-authority tests**

Assert:

- producer summary classification is not accepted as independent authority;
- no `independent_verification.json` or `report.md` exists;
- no CUDA API is called for CPU fixtures;
- the original source bundle remains byte-identical;
- all saved decoded tensors are FP32 and finite;
- all saved source copies equal the originals exactly;
- repeated production with the same tensor I/O and deterministic clock yields
  identical rows except explicitly normalized timing fields.

- [ ] **Step 9: Run the producer suite**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_calibration.py
python3 -m py_compile \
  tools/qwen35_recurrent_int8_calibration.py \
  tools/test_qwen35_recurrent_int8_calibration.py
```

Expected:

```text
qwen35 recurrent int8 calibration producer tests passed
```

---

### Task 4: Build the Independent Calibration Verifier

**Files:**
- Create: `tools/verify_qwen35_recurrent_int8_calibration.py`
- Create: `tools/test_verify_qwen35_recurrent_int8_calibration.py`

**Interfaces:**
- Consumes: Task 2 immutable schemas plus on-disk artifacts.
- May import from Task 1 only the immutable codec identity and decoder type;
  it must independently recompute row metrics and byte equations.
- Produces: `independent_verification.json` and `report.md`.

- [ ] **Step 1: Create a complete valid fixture through the real producer**

Reuse the Task 3 synthetic source bundle and call `run_calibration()`. The
verifier test must then operate on disk artifacts, not producer Python return
values.

- [ ] **Step 2: Write independent reconstruction RED tests**

Require `verify_calibration(run_dir)` to:

- validate exact top-level and nested inventory;
- reject all symlinks and non-regular files;
- verify `artifact_manifest.json` path, size, and SHA-256 coverage;
- validate source bundle and threshold schemas independently;
- reconstruct expected tensor IDs;
- load source, encoded values, scales, and decoded tensors;
- validate shape, dtype, finite values, and forbidden `-128`;
- independently decode INT8 values and scales;
- require saved decoded tensor equality to the independently decoded tensor;
- recompute source/encoded/decoded hashes;
- recompute all byte equations;
- recompute zero rows and saturation;
- recompute max absolute error, mean absolute error, RMSE, relative L2, and
  cosine similarity in CPU float64;
- require raw-row metrics to match within frozen tolerances;
- independently classify `PASS | NO_GO | INVALID`;
- write its own verification and Markdown report.

- [ ] **Step 3: Run verifier tests and confirm RED**

Run:

```bash
python3 tools/test_verify_qwen35_recurrent_int8_calibration.py
```

Expected: import failure because the verifier does not exist.

- [ ] **Step 4: Implement closed inventory verification**

Permit exactly:

```text
source_bundle_manifest.json
thresholds.json
commands.json
calibration_rows.jsonl
summary.json
artifact_manifest.json
source/**
encoded_values/**
scales/**
decoded/**
```

before verification. Reject pre-existing verifier outputs unless
`overwrite_verification=True` is explicitly passed to a private CLI helper;
the public `verify_calibration()` must be single-use and fail closed.

- [ ] **Step 5: Implement independent tensor and metric verification**

Do not call `qwen35_recurrent_int8_error_metrics()`. Implement a private
float64 recomputation in the verifier so producer and verifier cannot share
the same aggregation bug.

Use exact comparison for:

- identities;
- paths;
- hashes;
- shapes;
- dtypes;
- integer counters;
- byte counts;
- finite flags;
- saved decoded tensor versus independently decoded tensor.

Use `math.isclose(..., rel_tol=1e-12, abs_tol=1e-12)` for compression ratio
and `rel_tol=1e-9, abs_tol=1e-12` for recorded error metrics.

- [ ] **Step 6: Add tamper rejection**

Create one isolated fixture per mutation:

- extra top-level file;
- extra nested tensor file;
- missing tensor file;
- symlink;
- artifact size/hash mismatch;
- reordered or duplicate tensor row;
- changed source tensor bytes;
- changed INT8 value;
- inserted `-128`;
- changed scale;
- zero/negative/NaN/infinite scale;
- changed decoded tensor;
- changed source/decoded dtype;
- changed shape;
- changed logical/payload/scale/encoded bytes;
- changed compression ratio;
- changed zero-row or saturation count;
- changed error metric;
- changed timing to negative/non-finite;
- changed finite flag;
- missing expected rank/layer/workload;
- changed codec;
- changed thresholds after producer execution;
- producer summary claiming `PASS` while thresholds require `NO_GO`.

Every structural or integrity mutation must classify `INVALID` or raise the
documented verification error. The verifier must never trust the producer
summary classification.

- [ ] **Step 7: Freeze report contents**

`report.md` must include:

- source bundle, model, source tree, workload, threshold, and codec hashes;
- exact tensor count and rank/workload/layer coverage;
- aggregate logical, payload, scale, and encoded bytes;
- measured aggregate compression ratio;
- worst layer/tensor for max absolute and relative L2 error;
- minimum cosine similarity;
- zero-row and saturation totals;
- encode/decode timing summaries labeled producer-observed;
- independent classification and every reason;
- claim boundary stating no runtime integration, GPU-memory, speed, or quality
  authority.

- [ ] **Step 8: Run the verifier suite**

Run:

```bash
python3 tools/test_verify_qwen35_recurrent_int8_calibration.py
python3 -m py_compile \
  tools/verify_qwen35_recurrent_int8_calibration.py \
  tools/test_verify_qwen35_recurrent_int8_calibration.py
```

Expected:

```text
qwen35 recurrent int8 calibration verifier tests passed
```

---

### Task 5: Regression, Documentation, and Completion Audit

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Verify only: all files from Tasks 1-4

**Interfaces:**
- Consumes: all completed tasks.
- Produces: a verified CPU-safe codec/calibration foundation and exact next
  prerequisites.

- [ ] **Step 1: Run all new focused suites together**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_codec.py
python3 tools/test_qwen35_recurrent_int8_calibration_contract.py
python3 tools/test_qwen35_recurrent_int8_calibration.py
python3 tools/test_verify_qwen35_recurrent_int8_calibration.py
```

Expected: all four suites pass in separate fresh processes.

- [ ] **Step 2: Run adjacent hybrid-state regressions**

Run:

```bash
python3 tools/test_hybrid_state.py
python3 tools/test_qwen35_hybrid_state_layout.py
python3 tools/test_qwen35_layer_state_adapter.py
python3 tools/test_qwen35_cross_layer_state_transaction.py
python3 tools/test_qwen35_hybrid_prefix_cache.py
```

Expected: all existing suites pass unchanged. If an exact filename differs in
the current checkout, resolve it with:

```bash
rg --files tools | rg \
  'test_(hybrid_state|qwen35_(hybrid_state_layout|layer_state|cross_layer_state_transaction|hybrid_prefix_cache))\\.py$'
```

Run only the matching existing files; do not create aliases or modify adjacent
tests to hide a failure.

- [ ] **Step 3: Verify import and CUDA non-initialization policy**

Run:

```bash
python3 - <<'PY'
import torch
before = torch.cuda.is_initialized()
from tinyvllm.engine.qwen35_recurrent_int8_codec import (
    encode_qwen35_recurrent_int8_per_row,
)
after = torch.cuda.is_initialized()
assert before is False
assert after is False
print("QWEN35_RECURRENT_INT8_IMPORT_CUDA_FALSE")
PY
```

Expected:

```text
QWEN35_RECURRENT_INT8_IMPORT_CUDA_FALSE
```

- [ ] **Step 4: Run syntax and whitespace validation**

Run:

```bash
python3 -m py_compile \
  tinyvllm/engine/qwen35_recurrent_int8_codec.py \
  tools/test_qwen35_recurrent_int8_codec.py \
  tools/qwen35_recurrent_int8_calibration_contract.py \
  tools/test_qwen35_recurrent_int8_calibration_contract.py \
  tools/qwen35_recurrent_int8_calibration.py \
  tools/test_qwen35_recurrent_int8_calibration.py \
  tools/verify_qwen35_recurrent_int8_calibration.py \
  tools/test_verify_qwen35_recurrent_int8_calibration.py
git diff --check
git diff --cached --quiet
```

Expected: compilation and whitespace pass, with no staged files.

- [ ] **Step 5: Perform a prompt-to-artifact completion audit**

Record a table in `AGENT_HANDOFF_STATE.md` mapping:

| Requirement | Required evidence |
|---|---|
| deterministic per-row INT8 | codec tests and source |
| zero-row policy | exact scale/value test |
| finite/malformed rejection | codec and verifier tests |
| exact byte accounting | codec, row, producer, verifier equations |
| source immutability | producer test and hashes |
| independent metrics | verifier private float64 recomputation |
| closed artifact schema | inventory and tamper tests |
| default-off runtime | no runtime/cache/Engine/Scheduler modifications |
| no CUDA import side effect | fresh-process import assertion |
| no production claim | handoff claim boundary |
| real calibration | explicitly pending |
| P2 runtime authority | explicitly pending |

Do not mark real calibration or P2 runtime authority complete.

- [ ] **Step 6: Update the handoff**

Append:

- files added;
- exact test counts and commands;
- RED failures observed before each implementation;
- synthetic fixture dimensions and measured static ratio;
- source/encoded/decoded hash semantics;
- any negative/tamper branches;
- no staged files;
- no runtime integration;
- no remote/GPU run;
- attempt19 and P1 prerequisites still pending;
- next plan is real full-fidelity snapshot capture and canonical calibration,
  but only after P1 is authoritative.

## Completion Boundary

This plan is complete when the CPU reference codec, immutable calibration
contract, producer, independent verifier, tamper suite, adjacent regressions,
and handoff audit all pass.

Completion permits only this claim:

> TinyLLMForge has a locally tested, default-off Qwen3.5 recurrent-state
> per-row INT8 reference codec and independently verifiable offline
> calibration protocol.

It does not permit claims about:

- a real Qwen3.5 snapshot calibration result;
- the static `3.4358974359x` estimate being achieved in production;
- quality or accuracy preservation;
- physical GPU cache or allocator reduction;
- TTFT, decode latency, throughput, or end-to-end speed;
- INT4, low-rank, sparse, Gist, or layer-sharing benefit;
- completion of the active high-performance-engine objective.

