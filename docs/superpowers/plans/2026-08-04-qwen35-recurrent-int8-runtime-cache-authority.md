# Qwen3.5 Recurrent INT8 Runtime Cache and Authority Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the approved recurrent-only per-row INT8 snapshot representation as an explicit default-off Qwen3.5 hybrid-prefix runtime mode, then add a closed schema-v2 P0/P1/P2 canonical benchmark whose independent verifier alone determines the result.

**Architecture:** Keep the existing P1 `Qwen35HybridPrefixSnapshotCache` and its `storage_bytes`/counter semantics unchanged. Add a representation contract plus a separate P2 cache that stores exact BF16 convolution tensors, immutable INT8 recurrent payloads, and FP32 row scales; publication remains an 18-layer distributed transaction, while restore acquires an immutable reader lease, decodes all recurrent layers into private FP32 staging, and calls the existing cross-layer transaction exactly once. Only after the default-off runtime gate passes may the benchmark stack advance to schema-v2, where the producer records raw P0/P1/P2 evidence and an independent verifier recomputes correctness, physical-cache, capacity, performance, memory, safety, and final classification without trusting producer summaries.

**Tech Stack:** Python 3, PyTorch, TinyLLMForge Engine/ModelRunner distributed command transport, JSON/JSONL, SHA-256, pytest-style dependency-light test scripts, SSH/Kerberos remote execution on the fixed TP4 host.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Never modify `/Users/bytedance/dev/TinyLLMForge`.
- Do not clean, overwrite, stage, or commit unrelated modified/untracked files.
- Do not create a git commit unless the user explicitly requests one.
- P1 `exact_restore` remains the default representation; its runtime behavior, namespace, `storage_bytes` meaning, counters, benchmark schema-v1, and authority contract remain unchanged.
- P2 representation ID is exactly `recurrent_int8_per_row`.
- P2 codec ID is exactly `qwen35_recurrent_symmetric_int8_per_row_v1`.
- P2 is default-off and requires an explicit Config/CLI/benchmark selection; ordinary KV quantization flags must not select it.
- Compress only FP32 recurrent state. Convolution remains an exact detached contiguous BF16 clone. KV cache behavior is unchanged.
- Quantized recurrent values are INT8 in `[-127, 127]`; `-128` is forbidden. Scales are FP32, one per `[head, value_row, :]`.
- Decode target is FP32.
- The approved Qwen3.5 inventory contains exactly 18 linear-attention layers; publication and restore reject missing, duplicate, mixed, or extra layers.
- P1 and P2 use representation-specific cache namespaces. An exact entry never satisfies an INT8 lookup, and an INT8 entry never satisfies an exact lookup.
- P2 publication becomes visible only after all 18 layers, identities, encoded tensors, accounting, and intern references validate. Failure preserves any previous valid entry.
- P2 restore decodes all 18 recurrent layers into private FP32 staging before one call to `Qwen35CrossLayerStateTransaction.commit()`.
- Corruption or decode failure before commit quarantines the P2 entry, releases the reader lease, records an accounted miss, and recomputes; it never silently retries through P1.
- A cross-layer commit failure must roll back every destination layer and record a failed rollback if rollback itself does not complete.
- Do not add a fused decode/restore kernel in this implementation. Measure private FP32 staging allocated/reserved CUDA workspace on both success and failure.
- Continuation tokens must be exactly identical to P1.
- Final-logit tolerance is frozen at `atol=2e-5, rtol=0`.
- P2 measured unique physical snapshot-byte ratio versus P1 must be `<= 0.40`.
- P2 same-budget entry-capacity ratio versus P1 must be `>= 2.5`.
- Relative to P1: W1/W2 median TTFT ratios must be `<= 1.03`; every W1/W2 repetition must be `<= 1.05`; W3 throughput ratio must be `>= 0.98`; steady-state peak CUDA-reserved ratio must be `<= 1.05`.
- Relative to P0 recompute: W1 median TTFT ratio must be `<= 0.85`; W2 median TTFT ratio must be `<= 0.75`; W3 throughput ratio must be `>= 1.15`; decode-latency ratio must be `<= 1.02`.
- Runtime safety requires zero OOM, undeclared eviction, corruption, fallback, partial restore, mixed representation, missing layer, and failed rollback events.
- Closed result vocabulary is `GO`, `NO_GO_CORRECTNESS`, `NO_GO_RUNTIME_SAFETY`, `NO_GO_CACHE`, `NO_GO_PERFORMANCE`, `BLOCKED_RESOURCES`, and `INVALID_ARTIFACT`.
- Primary failure precedence is correctness, runtime safety, cache, then performance.
- Producer code never decides `GO`; only the independent verifier may classify a complete canonical artifact.
- The static `4,939,776 -> 1,437,696` bytes/rank, `3.4358974359x`, and `70.8955223881%` saving remain design estimates, not acceptance evidence.
- Do not run P2 canonical authority unless the strict P1 artifact is independently authoritative, the approved calibration artifact passes, fresh preflight is `READY`, and one-time execution authorization exists.
- Use only fixed remote target `sitian@10.232.195.203`, GPUs `2,4,5,6`, and minimum free bytes/card `25769803776`. Do not kill another user's process or weaken the resource gate.
- Use `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` and SSH options `-o BatchMode=yes -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=20`.
- A `BLOCKED_RESOURCES` preflight creates no remote run directory and starts or kills no process.
- No runtime, cache, memory, correctness, or performance benefit claim is allowed before a canonical artifact is accepted as `GO`.

---

## Mandatory Acceptance Gates

### Gate 1: Default-Off Runtime Integration

Gate 1 passes only when the representation/config contract, P2 cache, atomic publication, immutable reader lease, private all-layer decode staging, quarantine/miss behavior, rollback, Engine/ModelRunner wiring, runtime event identity, staging-memory accounting, and exact-off regressions all pass locally. Gate 1 must not alter schema-v1 benchmark files except for tests that prove they still expose exactly P0/P1.

### Gate 2: Schema-v2 Independent Authority

Gate 2 work starts only after Gate 1 passes. It adds schema-v2 P0/P1/P2 producer and verifier files while retaining schema-v1 files unchanged. A remote canonical run starts only after fresh source-bound prerequisites, calibration, P1 authority, resource preflight, one-time authorization, and execution-plan verification pass.

---

## File Map

### Gate 1 files

- Create `tinyvllm/engine/qwen35_hybrid_prefix_representation.py`: closed representation IDs, codec binding, versioned namespace identity, and default selection.
- Create `tools/test_qwen35_hybrid_prefix_representation.py`: representation/config closure and namespace tests.
- Modify `tinyvllm/config.py`: explicit default-off `qwen35_hybrid_prefix_representation` field and validation.
- Create `tinyvllm/engine/qwen35_hybrid_prefix_int8_cache.py`: P2 snapshot, prepared-publication state, encoded interning, immutable reader lease, quarantine, byte/workspace accounting, and all-layer decode-before-commit.
- Create `tools/test_qwen35_hybrid_prefix_int8_cache.py`: CPU atomicity, interning, corruption, lease, quarantine, accounting, and rollback tests.
- Modify `tinyvllm/engine/qwen35_hybrid_prefix_owner.py`: construct either the unchanged P1 cache or the new P2 cache from an explicit representation.
- Modify `tinyvllm/engine/model_runner.py`: representation-aware owner configuration and complete per-rank P2 observations.
- Modify `tinyvllm/engine/llm_engine.py`: representation-aware distributed configuration identity.
- Create `tools/test_qwen35_hybrid_prefix_int8_runtime.py`: ModelRunner/Engine default-off wiring, event identity, recompute-on-quarantine, no silent fallback, and exact regression tests.
- Modify `tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`: accept and forward explicit representation without changing schema-v1 policy behavior.
- Modify `tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`: prove exact schema-v1 behavior is unchanged and P2 must be explicitly selected outside v1.

### Gate 2 files

- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py`: closed schema-v2 matrix, thresholds, raw row schemas, inventory, and classification vocabulary.
- Create `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py`: exact matrix/threshold/schema/classification tests.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py`: P0/P1/P2 worker, raw token/logit/cache/workspace/memory/safety evidence.
- Create `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py`: configuration, matrix, aggregation, failure-atomicity, and raw-evidence tests.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py`: schema-v2 adapter selecting P0/P1/P2 explicitly.
- Create `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py`: P2 configuration and observation tests.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py`: atomic closed-inventory artifact assembly.
- Create `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py`: completeness, provenance, and no-partial-publication tests.
- Create `tools/verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py`: independent raw-evidence verifier and classifier.
- Create `tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py`: valid reconstruction plus manifest/receipt/inventory/raw/summary/threshold tamper rejection.
- Create `tools/run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py`: fixed-host preflight and source-bound launch-plan builder.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_plan.py`: command-only deterministic plan.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_authorization.py`: one-time nonce authorization.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_executor.py`: bounded step executor with failure evidence.
- Create `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt.py`: plan/authorization/resource/artifact receipt binding.
- Create corresponding `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_*.py` and `tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py`.
- Modify `AGENT_HANDOFF_STATE.md`: Gate 1/Gate 2 commands, evidence, resource state, claim boundary, and exact next action.

## Public Interfaces

Implementers must preserve these names and signatures across tasks.

```python
# tinyvllm/engine/qwen35_hybrid_prefix_representation.py

QWEN35_HYBRID_PREFIX_EXACT = "exact_restore"
QWEN35_HYBRID_PREFIX_RECURRENT_INT8 = "recurrent_int8_per_row"
QWEN35_HYBRID_PREFIX_DEFAULT = QWEN35_HYBRID_PREFIX_EXACT
QWEN35_HYBRID_PREFIX_REPRESENTATIONS = (
    QWEN35_HYBRID_PREFIX_EXACT,
    QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
)
QWEN35_HYBRID_PREFIX_EXACT_VERSION = "qwen35_hybrid_prefix_exact_v1"
QWEN35_HYBRID_PREFIX_INT8_VERSION = "qwen35_hybrid_prefix_recurrent_int8_v1"

@dataclass(frozen=True)
class Qwen35HybridPrefixRepresentation:
    name: str
    version: str
    codec: str | None

def resolve_qwen35_hybrid_prefix_representation(
    value: str,
) -> Qwen35HybridPrefixRepresentation
```

```python
# tinyvllm/engine/qwen35_hybrid_prefix_int8_cache.py

@dataclass(frozen=True)
class Qwen35HybridPrefixInt8Key:
    exact_key: Qwen35HybridPrefixKey
    representation: str
    representation_version: str
    codec: str

@dataclass(frozen=True)
class Qwen35HybridPrefixInt8Layer:
    layer_index: int
    convolution_state: torch.Tensor
    recurrent_values: torch.Tensor
    recurrent_scales: torch.Tensor
    source_shape: tuple[int, int, int]

@dataclass(frozen=True)
class Qwen35HybridPrefixInt8Accounting:
    full_fidelity_logical_bytes: int
    encoded_physical_bytes: int
    codec_metadata_bytes: int
    temporary_encode_workspace_bytes: int
    temporary_decode_workspace_bytes: int

@dataclass(frozen=True)
class Qwen35HybridPrefixInt8Snapshot:
    key: Qwen35HybridPrefixInt8Key
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    layers: tuple[Qwen35HybridPrefixInt8Layer, ...]
    accounting: Qwen35HybridPrefixInt8Accounting

@dataclass(frozen=True)
class Qwen35HybridPrefixInt8PreparedPublication:
    publication_id: int
    key: Qwen35HybridPrefixInt8Key
    token_ids: tuple[int, ...]
    block_identities: tuple[tuple[int, int, int], ...]
    accounting: Qwen35HybridPrefixInt8Accounting

class Qwen35HybridPrefixInt8ReaderLease:
    @property
    def snapshot(self) -> Qwen35HybridPrefixInt8Snapshot
    def release(self) -> None
    def __enter__(self) -> "Qwen35HybridPrefixInt8ReaderLease"
    def __exit__(self, exc_type, exc, traceback) -> None

class Qwen35HybridPrefixInt8SnapshotCache:
    def __init__(
        self,
        state_transaction: Qwen35CrossLayerStateTransaction,
        *,
        max_entries: int,
        max_bytes: int,
    )

    def prepare_publication(
        self,
        key: Qwen35HybridPrefixInt8Key,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        lease: HybridStateLease,
    ) -> Qwen35HybridPrefixInt8PreparedPublication | None

    def precommit_publication(
        self,
        prepared: Qwen35HybridPrefixInt8PreparedPublication,
    ) -> None

    def finalize_publication(
        self,
        prepared: Qwen35HybridPrefixInt8PreparedPublication,
    ) -> bool

    def seal_publication(
        self,
        prepared: Qwen35HybridPrefixInt8PreparedPublication,
    ) -> None

    def rollback_publication(
        self,
        prepared: Qwen35HybridPrefixInt8PreparedPublication,
    ) -> None

    def publish(
        self,
        key: Qwen35HybridPrefixInt8Key,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        lease: HybridStateLease,
    ) -> bool

    def acquire_reader(
        self,
        key: Qwen35HybridPrefixInt8Key,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
    ) -> Qwen35HybridPrefixInt8ReaderLease | None

    def acquire(
        self,
        key: Qwen35HybridPrefixInt8Key,
        token_ids: tuple[int, ...],
        block_identities: tuple[tuple[int, int, int], ...],
        leases: tuple[HybridStateLease, ...],
    ) -> bool

    def quarantine(
        self,
        key: Qwen35HybridPrefixInt8Key,
        token_ids: tuple[int, ...],
        *,
        reason: str,
    ) -> bool

    def invalidate_blocks(
        self,
        block_identities: tuple[tuple[int, int, int], ...],
    ) -> int

    def clear(self) -> int
    def observation_snapshot(self) -> dict[str, int | str]
```

```python
# tinyvllm/engine/qwen35_hybrid_prefix_owner.py

def build_qwen35_hybrid_prefix_restore_owner(
    pool,
    *,
    participant_id,
    max_entries,
    max_bytes,
    representation="exact_restore",
)
```

```python
# tinyvllm/engine/model_runner.py

def configure_qwen35_hybrid_prefix_restore_owner(
    self,
    max_entries,
    max_bytes,
    representation="exact_restore",
)
```

```python
# tinyvllm/engine/llm_engine.py

def configure_qwen35_hybrid_prefix_restore(
    self,
    *,
    max_entries,
    max_bytes,
    timeout_s,
    representation="exact_restore",
)

def configure_qwen35_hybrid_prefix_publication_runtime(
    self,
    *,
    model_fingerprint,
    max_entries,
    max_bytes,
    timeout_s,
    representation="exact_restore",
)
```

```python
# tools/qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py

SCHEMA_VERSION = "qwen35.tp4-hybrid-prefix-performance-cache.v2"
PROFILES = (
    "recompute",
    "exact_restore",
    "recurrent_int8_per_row",
)
CODEC_ID = "qwen35_recurrent_symmetric_int8_per_row_v1"
RESULTS = (
    "GO",
    "NO_GO_CORRECTNESS",
    "NO_GO_RUNTIME_SAFETY",
    "NO_GO_CACHE",
    "NO_GO_PERFORMANCE",
    "BLOCKED_RESOURCES",
    "INVALID_ARTIFACT",
)

def classify_run(metrics: Mapping[str, object]) -> str
```

```python
# tools/verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py

def verify_run(run_dir: Path) -> dict[str, object]
```

---

## Gate 1 — Default-Off Runtime Integration

### Task 1: Freeze the Representation and Config Contract

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_prefix_representation.py`
- Create: `tools/test_qwen35_hybrid_prefix_representation.py`
- Modify: `tinyvllm/config.py`

**Interfaces:**
- Produces: all representation constants, `Qwen35HybridPrefixRepresentation`, and `resolve_qwen35_hybrid_prefix_representation()`.
- Consumed by: Tasks 2-5 and every schema-v2 runtime adapter.

- [ ] **Step 1: Write the RED representation/config tests**

```python
def test_default_representation_is_exact_and_int8_is_explicit():
    assert QWEN35_HYBRID_PREFIX_DEFAULT == "exact_restore"
    assert resolve_qwen35_hybrid_prefix_representation(
        "exact_restore"
    ).codec is None
    int8 = resolve_qwen35_hybrid_prefix_representation(
        "recurrent_int8_per_row"
    )
    assert int8.version == "qwen35_hybrid_prefix_recurrent_int8_v1"
    assert int8.codec == "qwen35_recurrent_symmetric_int8_per_row_v1"


def test_unknown_or_kv_quantization_names_are_rejected():
    for value in ("int8", "kv8", "", None):
        with raises((TypeError, ValueError)):
            resolve_qwen35_hybrid_prefix_representation(value)
```

Construct `Config` with a temporary model directory and assert its default field is `exact_restore`, explicit `recurrent_int8_per_row` is accepted, and any other value fails `__post_init__`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_representation.py
```

Expected: FAIL because the representation module and Config field do not exist.

- [ ] **Step 3: Implement the closed representation resolver**

```python
from dataclasses import dataclass

from tinyvllm.engine.qwen35_recurrent_int8_codec import (
    QWEN35_RECURRENT_INT8_CODEC,
)

QWEN35_HYBRID_PREFIX_EXACT = "exact_restore"
QWEN35_HYBRID_PREFIX_RECURRENT_INT8 = "recurrent_int8_per_row"
QWEN35_HYBRID_PREFIX_DEFAULT = QWEN35_HYBRID_PREFIX_EXACT
QWEN35_HYBRID_PREFIX_REPRESENTATIONS = (
    QWEN35_HYBRID_PREFIX_EXACT,
    QWEN35_HYBRID_PREFIX_RECURRENT_INT8,
)
QWEN35_HYBRID_PREFIX_EXACT_VERSION = "qwen35_hybrid_prefix_exact_v1"
QWEN35_HYBRID_PREFIX_INT8_VERSION = "qwen35_hybrid_prefix_recurrent_int8_v1"


@dataclass(frozen=True)
class Qwen35HybridPrefixRepresentation:
    name: str
    version: str
    codec: str | None


def resolve_qwen35_hybrid_prefix_representation(value):
    if value == QWEN35_HYBRID_PREFIX_EXACT:
        return Qwen35HybridPrefixRepresentation(
            value,
            QWEN35_HYBRID_PREFIX_EXACT_VERSION,
            None,
        )
    if value == QWEN35_HYBRID_PREFIX_RECURRENT_INT8:
        return Qwen35HybridPrefixRepresentation(
            value,
            QWEN35_HYBRID_PREFIX_INT8_VERSION,
            QWEN35_RECURRENT_INT8_CODEC,
        )
    raise ValueError(f"unsupported Qwen3.5 hybrid prefix representation: {value}")
```

Add to `Config`:

```python
qwen35_hybrid_prefix_representation: str = "exact_restore"
```

and in `__post_init__`:

```python
assert self.qwen35_hybrid_prefix_representation in (
    "exact_restore",
    "recurrent_int8_per_row",
)
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_representation.py
python3 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/qwen35_hybrid_prefix_representation.py
```

Expected: representation tests pass and both modules compile.

### Task 2: Add the P2 Encoded Snapshot and Atomic Publication Cache

**Files:**
- Create: `tinyvllm/engine/qwen35_hybrid_prefix_int8_cache.py`
- Create: `tools/test_qwen35_hybrid_prefix_int8_cache.py`

**Interfaces:**
- Consumes: `Qwen35HybridPrefixKey`, `Qwen35CrossLayerStateTransaction`, `Qwen35EncodedRecurrentInt8`, `encode_qwen35_recurrent_int8_per_row()`, and Task 1 representation constants.
- Produces: all P2 cache dataclasses plus `Qwen35HybridPrefixInt8SnapshotCache`.
- Consumed by: Tasks 3-5.

- [ ] **Step 1: Build an 18-layer CPU fixture and write RED publication tests**

Use 18 synthetic layer indices, exact BF16 convolution shapes, smaller positive FP32 recurrent shapes, and the existing `HybridStateTensorPool`/adapter/transaction fixture style. Add tests that assert:

```python
def test_all_18_layers_publish_atomically_and_charge_encoded_storage():
    cache, source_lease = int8_fixture()
    assert cache.publish(int8_key(), tokens(), blocks(), source_lease)
    snapshot = only_snapshot(cache)
    assert tuple(layer.layer_index for layer in snapshot.layers) == tuple(range(18))
    assert all(layer.convolution_state.dtype == torch.bfloat16 for layer in snapshot.layers)
    assert all(layer.recurrent_values.dtype == torch.int8 for layer in snapshot.layers)
    assert all(layer.recurrent_scales.dtype == torch.float32 for layer in snapshot.layers)
    observed = cache.observation_snapshot()
    assert observed["representation"] == "recurrent_int8_per_row"
    assert observed["codec"] == QWEN35_RECURRENT_INT8_CODEC
    assert observed["current_bytes"] == observed["current_encoded_physical_bytes"]


def test_late_encode_failure_preserves_previous_entry_and_releases_intern_refs():
    cache, source_lease = int8_fixture()
    assert cache.publish(int8_key(), tokens(), blocks(), source_lease)
    previous = only_snapshot(cache)
    inject_non_finite_source(layer_index=17)
    with raises(ValueError):
        cache.publish(int8_key(), tokens(), blocks(), source_lease)
    assert only_snapshot(cache) is previous
    assert cache.observation_snapshot()["current_prepared_publications"] == 0
```

Also add exact-byte-equality interning, digest-collision safety, mixed codec rejection, partial-layer rejection, oversize rejection, replacement rollback, LRU, invalidation, clear, and distinct exact/P2 key tests.

- [ ] **Step 2: Run the publication tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_int8_cache.py
```

Expected: FAIL because the P2 cache module does not exist.

- [ ] **Step 3: Implement immutable P2 layer/snapshot/accounting types and validation**

The cache must validate:

```python
EXPECTED_LINEAR_LAYER_COUNT = 18


def _validate_layers(layers):
    if len(layers) != EXPECTED_LINEAR_LAYER_COUNT:
        raise ValueError("INT8 snapshot requires exactly 18 layers")
    indices = tuple(layer.layer_index for layer in layers)
    if len(set(indices)) != EXPECTED_LINEAR_LAYER_COUNT:
        raise ValueError("INT8 snapshot layer identities are not unique")
    if indices != tuple(sorted(indices)):
        raise ValueError("INT8 snapshot layers are not ordered")
    for layer in layers:
        if layer.convolution_state.dtype != torch.bfloat16:
            raise ValueError("INT8 snapshot convolution must remain BF16")
        if layer.recurrent_values.dtype != torch.int8:
            raise ValueError("INT8 recurrent payload dtype mismatch")
        if layer.recurrent_scales.dtype != torch.float32:
            raise ValueError("INT8 recurrent scale dtype mismatch")
        if torch.any(layer.recurrent_values == -128):
            raise ValueError("INT8 recurrent payload contains forbidden -128")
```

Clone convolution with `detach().contiguous().clone()`, encode recurrent tensors through the approved codec, and store no decoded FP32 recurrent tensor in the resident snapshot.

- [ ] **Step 4: Implement prepare/precommit/finalize/seal/rollback**

Mirror the P1 phase machine while keeping a separate class and counters. `prepare_publication()` must gather all layers, clone/encode into private candidates, compute:

```python
convolution_bytes = sum(
    layer.convolution_state.numel()
    * layer.convolution_state.element_size()
    for layer in private_layers
)
recurrent_logical_bytes = sum(
    math.prod(layer.source_shape)
    * torch.tensor([], dtype=torch.float32).element_size()
    for layer in private_layers
)
recurrent_payload_bytes = sum(
    layer.recurrent_values.numel()
    * layer.recurrent_values.element_size()
    for layer in private_layers
)
scale_bytes = sum(
    layer.recurrent_scales.numel()
    * layer.recurrent_scales.element_size()
    for layer in private_layers
)
accounting = Qwen35HybridPrefixInt8Accounting(
    full_fidelity_logical_bytes=(
        convolution_bytes + recurrent_logical_bytes
    ),
    encoded_physical_bytes=(
        convolution_bytes + recurrent_payload_bytes + scale_bytes
    ),
    codec_metadata_bytes=encoded_metadata_bytes(private_layers),
    temporary_encode_workspace_bytes=encode_workspace_peak_bytes,
    temporary_decode_workspace_bytes=0,
)
```

`encoded_metadata_bytes()` must count representation-owned scalar and shape
metadata using canonical JSON bytes. `encode_workspace_peak_bytes` is the
maximum observed CUDA allocated delta while encoding all private layers; it is
reported separately and is not charged to resident `current_bytes`.

`precommit_publication()` interns convolution, values, and scales using dtype/shape/device/codec/digest plus byte equality. `finalize_publication()` is the first visible mutation; `rollback_publication()` restores the exact previous entry/LRU/counters and releases all private references.

- [ ] **Step 5: Run publication/cache tests and verify GREEN**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_int8_cache.py
python3 tools/test_qwen35_hybrid_prefix_cache.py
```

Expected: all P2 tests pass and the unchanged P1 cache suite still passes.

### Task 3: Implement Immutable Reader Leases, Private Decode Staging, and Quarantine

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_int8_cache.py`
- Modify: `tools/test_qwen35_hybrid_prefix_int8_cache.py`

**Interfaces:**
- Produces: `acquire_reader()`, `Qwen35HybridPrefixInt8ReaderLease`, quarantine semantics, and `acquire()` all-layer decode-before-commit.
- Consumed by: Tasks 4-5.

- [ ] **Step 1: Write RED restore atomicity and lease tests**

```python
def test_reader_lease_keeps_snapshot_alive_across_concurrent_eviction():
    cache, source_lease = int8_fixture(max_entries=1)
    assert cache.publish(int8_key(1), tokens(1), blocks(1), source_lease)
    reader = cache.acquire_reader(int8_key(1), tokens(1), blocks(1))
    assert reader is not None
    assert cache.publish(int8_key(2), tokens(2), blocks(2), source_lease)
    assert reader.snapshot.key == int8_key(1)
    reader.release()
    assert cache.observation_snapshot()["current_reader_leases"] == 0


def test_late_decode_failure_never_mutates_live_rows_and_quarantines_entry():
    cache, source_lease, destination_leases, adapters = int8_fixture()
    assert cache.publish(int8_key(), tokens(), blocks(), source_lease)
    before = destination_rows(adapters, destination_leases)
    corrupt_scale(cache, layer_index=17, value=float("nan"))
    assert cache.acquire(
        int8_key(), tokens(), blocks(), destination_leases
    ) is False
    assert destination_rows(adapters, destination_leases) == before
    observed = cache.observation_snapshot()
    assert observed["quarantines"] == 1
    assert observed["decode_failures"] == 1
    assert observed["misses"] == 1
    assert observed["hits"] == 0
```

Add tests for successful FP32 decode, exact BF16 convolution restoration, one commit call after all 18 decodes, commit failure rollback of every layer, failed rollback accounting, workspace release on success/failure, and no LRU refresh on failed restore.

- [ ] **Step 2: Run focused restore tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_int8_cache.py
```

Expected: restore/lease/quarantine tests fail because the methods are incomplete.

- [ ] **Step 3: Implement reader lease pinning and deferred intern release**

Each visible snapshot has a reader count. Eviction, invalidation, replacement, clear, or quarantine detaches the entry immediately but releases its intern references only after the last reader releases:

```python
class Qwen35HybridPrefixInt8ReaderLease:
    def release(self):
        if self._released:
            return
        self._released = True
        self._cache._release_reader(self._snapshot)
```

Reject use-after-release and double ownership transfer. Report `current_reader_leases`, `peak_reader_leases`, and `deferred_snapshot_releases`.

- [ ] **Step 4: Implement decode-all-then-commit**

`acquire()` must:

```python
reader = self.acquire_reader(key, token_ids, block_identities)
if reader is None:
    return False
with reader:
    decoded_layers = []
    try:
        for layer, adapter in zip(
            reader.snapshot.layers,
            self.state_transaction.adapters,
        ):
            recurrent = decode_qwen35_recurrent_int8_per_row(
                encoded_from_layer(layer),
                device=adapter.recurrent.device,
            )
            validate_decoded_layer(layer, recurrent, adapter)
            decoded_layers.append((layer.convolution_state, recurrent))
        candidates = expand_candidates(decoded_layers, leases)
        self.state_transaction.commit(leases, candidates)
    except (ValueError, RuntimeError) as error:
        if commit_started(error):
            self._record_commit_failure(error)
            raise
        self.quarantine(key, token_ids, reason=str(error))
        self._counters["misses"] += 1
        return False
```

No live-pool tensor is passed to the decoder. Record measured staging allocated/reserved CUDA deltas around private decode construction and release.

- [ ] **Step 5: Run P2 and transaction regressions**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_int8_cache.py
python3 tools/test_qwen35_hybrid_prefix_cache.py
python3 tools/test_qwen35_hybrid_prefix_restore_ticket.py
python3 tools/test_qwen35_hybrid_prefix_acquisition.py
```

Expected: all tests pass; P2 corruption is an accounted miss, while commit failure remains an explicit runtime failure with transaction rollback.

### Task 4: Wire Owner, ModelRunner, Engine, and Default-Off Runtime Identity

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_owner.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Create: `tools/test_qwen35_hybrid_prefix_int8_runtime.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: representation-aware owner/configuration and complete rank-local observations.
- Consumed by: Task 5 and Gate 2 adapters.

- [ ] **Step 1: Write RED owner and distributed configuration tests**

```python
def test_owner_defaults_to_unchanged_exact_cache():
    owner = build_qwen35_hybrid_prefix_restore_owner(
        pool,
        participant_id=0,
        max_entries=4,
        max_bytes=1 << 20,
    )
    assert isinstance(owner.snapshot_cache, Qwen35HybridPrefixSnapshotCache)
    assert owner.representation == "exact_restore"


def test_owner_constructs_int8_cache_only_when_explicit():
    owner = build_qwen35_hybrid_prefix_restore_owner(
        pool,
        participant_id=0,
        max_entries=4,
        max_bytes=1 << 20,
        representation="recurrent_int8_per_row",
    )
    assert isinstance(
        owner.snapshot_cache,
        Qwen35HybridPrefixInt8SnapshotCache,
    )
    assert owner.codec == QWEN35_RECURRENT_INT8_CODEC
```

Add ModelRunner and Engine fake-rank tests proving representation/version/codec must match on every rank, reconfiguration across representations is rejected, and omitted arguments preserve the previous P1 tuple and result fields.

- [ ] **Step 2: Run the runtime tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_int8_runtime.py
```

Expected: FAIL because owner/ModelRunner/Engine do not accept representation.

- [ ] **Step 3: Extend owner and ModelRunner without changing default behavior**

Add fields to `Qwen35HybridPrefixRestoreOwner`:

```python
representation: str
representation_version: str
codec: str | None
```

Resolve representation once in the owner builder. Construct the existing P1 cache for `exact_restore` and the P2 cache only for `recurrent_int8_per_row`.

Extend ModelRunner's configuration identity to `(max_entries, max_bytes, representation)` and return:

```python
{
    "participant_id": int(self.rank),
    "capacity": int(owner.pool.capacity),
    "layout_fingerprint": owner.pool.layout.fingerprint,
    "bytes_per_slot": int(owner.pool.layout.bytes_per_slot),
    "max_entries": int(owner.max_entries),
    "max_bytes": int(owner.max_bytes),
    "representation": owner.representation,
    "representation_version": owner.representation_version,
    "codec": owner.codec,
}
```

For `qwen35_hybrid_prefix_cache_snapshot()`, retain every old field and add P2-only accounting/safety fields with explicit zero values on P1 so callers never infer representation from missing fields.

- [ ] **Step 4: Extend Engine distributed identity**

Include representation in restore/publication configuration tuples and acknowledged calls:

```python
self.call_model_runner_acknowledged(
    "configure_qwen35_hybrid_prefix_restore_owner",
    max_entries,
    max_bytes,
    representation,
    timeout_s=timeout_s,
)
```

Require all ranks to agree on representation/version/codec. Preserve the default `representation="exact_restore"` at every public entry point.

- [ ] **Step 5: Run runtime and exact-off regressions**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_int8_runtime.py
python3 tools/test_qwen35_hybrid_prefix_cache.py
python3 tools/test_qwen35_hybrid_prefix_engine_restore.py
python3 tools/test_qwen35_hybrid_prefix_engine_publication.py
python3 tools/test_qwen35_hybrid_prefix_publication_coordinator.py
python3 -m py_compile \
  tinyvllm/engine/qwen35_hybrid_prefix_owner.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py
```

Expected: all pass; an omitted representation still builds P1 and produces unchanged P1 cache values.

### Task 5: Complete Runtime Events, Recompute-on-Quarantine, and Staging-Memory Evidence

**Files:**
- Modify: `tinyvllm/engine/qwen35_hybrid_prefix_int8_cache.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_qwen35_hybrid_prefix_int8_runtime.py`
- Modify: `tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`
- Modify: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py`

**Interfaces:**
- Produces: complete runtime observation schema and explicit P2 adapter selection.
- Gate: finishing this task and its audit completes Gate 1.

- [ ] **Step 1: Write RED event and fallback tests**

Assert every publish/restore/miss/quarantine/rollback observation includes:

```text
representation
representation_version
codec
publishes
hits
misses
evictions
validation_failures
quarantines
decode_failures
commit_failures
rollback_failures
fallbacks
partial_restore_attempts
mixed_representation_rejections
missing_layer_rejections
current_full_fidelity_logical_bytes
current_encoded_physical_bytes
current_codec_metadata_bytes
peak_temporary_encode_workspace_bytes
peak_temporary_decode_workspace_bytes
peak_temporary_decode_cuda_allocated_bytes
peak_temporary_decode_cuda_reserved_bytes
```

Add an Engine request test where an INT8 entry is corrupted before commit: restore returns miss, the normal prefill path recomputes, `fallbacks == 0`, `quarantines == 1`, and no P1 lookup occurs.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_hybrid_prefix_int8_runtime.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py
```

Expected: FAIL on missing observation fields and explicit adapter selection.

- [ ] **Step 3: Complete observations and benchmark adapter forwarding**

The schema-v1 adapter must still accept only `recompute` and `exact_restore`. Add an internal adapter argument:

```python
def configure_qwen35_hybrid_prefix_publication_runtime(
    self,
    *,
    model_fingerprint,
    max_entries,
    max_bytes,
    timeout_s,
    representation="exact_restore",
):
    return self.engine.configure_qwen35_hybrid_prefix_publication_runtime(
        model_fingerprint=model_fingerprint,
        max_entries=max_entries,
        max_bytes=max_bytes,
        timeout_s=timeout_s,
        representation=representation,
    )
```

Its existing schema-v1 `exact_restore` branch passes `representation="exact_restore"`. It must reject `recurrent_int8_per_row` when invoked through the v1 policy contract, proving P2 cannot leak into P1 authority.

- [x] **Step 4: Run the complete Gate 1 suite**

Run:

```bash
python3 tools/test_qwen35_recurrent_int8_codec.py
python3 tools/test_qwen35_hybrid_prefix_representation.py
python3 tools/test_qwen35_hybrid_prefix_int8_cache.py
python3 tools/test_qwen35_hybrid_prefix_int8_runtime.py
python3 tools/test_qwen35_hybrid_prefix_cache.py
python3 tools/test_qwen35_hybrid_prefix_acquisition.py
python3 tools/test_qwen35_hybrid_prefix_publication_candidate.py
python3 tools/test_qwen35_hybrid_prefix_publication_coordinator.py
python3 tools/test_qwen35_hybrid_prefix_publication_ticket.py
python3 tools/test_qwen35_hybrid_prefix_restore_ticket.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
git diff --check -- \
  tinyvllm/config.py \
  tinyvllm/engine/qwen35_hybrid_prefix_representation.py \
  tinyvllm/engine/qwen35_hybrid_prefix_int8_cache.py \
  tinyvllm/engine/qwen35_hybrid_prefix_owner.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_qwen35_hybrid_prefix_representation.py \
  tools/test_qwen35_hybrid_prefix_int8_cache.py \
  tools/test_qwen35_hybrid_prefix_int8_runtime.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py \
  tools/test_qwen35_tp4_hybrid_prefix_benchmark_engine_adapter.py
```

Expected: all tests pass and `git diff --check` is silent.

- [x] **Step 5: Record the Gate 1 decision**

Append to `AGENT_HANDOFF_STATE.md`:

```markdown
### P2 Gate 1 — default-off runtime integration

- Representation: `recurrent_int8_per_row`
- Default remains: `exact_restore`
- Gate 1 tests: PASS/FAIL with exact commands and counts
- Exact-off regression: PASS/FAIL
- Canonical P2 authority: NOT RUN
- Claim boundary: no cache, memory, correctness, or performance benefit claim
```

If any Gate 1 test fails, stop before creating or running schema-v2 authority code.

---

## Gate 2 — Schema-v2 Canonical Independent Authority

### Task 6: Freeze the Schema-v2 P0/P1/P2 Contract

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py`

**Interfaces:**
- Produces: exact profiles, matrix, thresholds, closed schemas, artifact inventory, and `classify_run()`.
- Consumed by: Tasks 7-10.

- [ ] **Step 1: Write RED contract tests**

```python
def test_v2_profiles_thresholds_and_result_vocabulary_are_frozen():
    assert contract.PROFILES == (
        "recompute",
        "exact_restore",
        "recurrent_int8_per_row",
    )
    assert contract.LOGIT_TOLERANCE == {"atol": 2e-5, "rtol": 0.0}
    assert contract.THRESHOLDS == {
        "int8_to_exact_unique_physical_bytes_max_ratio": 0.40,
        "int8_to_exact_same_budget_capacity_min_ratio": 2.5,
        "w1_int8_to_exact_median_ttft_max_ratio": 1.03,
        "w1_int8_to_exact_every_ttft_max_ratio": 1.05,
        "w2_int8_to_exact_median_ttft_max_ratio": 1.03,
        "w2_int8_to_exact_every_ttft_max_ratio": 1.05,
        "w3_int8_to_exact_throughput_min_ratio": 0.98,
        "int8_to_exact_peak_cuda_reserved_max_ratio": 1.05,
        "w1_int8_to_recompute_median_ttft_max_ratio": 0.85,
        "w2_int8_to_recompute_median_ttft_max_ratio": 0.75,
        "w3_int8_to_recompute_throughput_min_ratio": 1.15,
        "int8_to_recompute_decode_latency_max_ratio": 1.02,
    }
    assert contract.RESULTS == (
        "GO",
        "NO_GO_CORRECTNESS",
        "NO_GO_RUNTIME_SAFETY",
        "NO_GO_CACHE",
        "NO_GO_PERFORMANCE",
        "BLOCKED_RESOURCES",
        "INVALID_ARTIFACT",
    )
```

Test deterministic triple-profile ordering, exact case counts, strict serial correctness cases, closed raw schemas, and classification precedence.

- [ ] **Step 2: Run contract tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py
```

Expected: FAIL because schema-v2 contract does not exist.

- [ ] **Step 3: Implement the contract by copying frozen v1 workload identities**

Import no mutable v1 runtime state. Copy/freeze W0-W4 workload payloads and prerequisite validators, then add P2 codec/representation/calibration/P1-authority bindings. Define raw case rows with token/logit paths and raw process rows with per-rank cache, workspace, CUDA, safety, and capacity fields. Unknown fields fail validation.

`classify_run()` must execute:

```python
if metrics["artifact_invalid"]:
    return "INVALID_ARTIFACT"
if metrics["resources_blocked"]:
    return "BLOCKED_RESOURCES"
if not metrics["correctness_pass"]:
    return "NO_GO_CORRECTNESS"
if not metrics["runtime_safety_pass"]:
    return "NO_GO_RUNTIME_SAFETY"
if not metrics["cache_pass"]:
    return "NO_GO_CACHE"
if not metrics["performance_pass"]:
    return "NO_GO_PERFORMANCE"
return "GO"
```

- [ ] **Step 4: Run v2 and v1 contract tests**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_contract.py
```

Expected: both pass; v1 still exposes exactly two policies.

### Task 7: Implement the Schema-v2 Adapter and Raw Worker

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py`
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py`

**Interfaces:**
- Consumes: Gate 1 runtime and Task 6 contract.
- Produces: one authorized case execution and raw evidence only.
- Consumed by: Tasks 8-10.

- [ ] **Step 1: Write RED profile-configuration tests**

Assert the only configuration differences are:

```python
recompute = {
    "hybrid_prefix_enabled": False,
    "representation": None,
}
exact = {
    "hybrid_prefix_enabled": True,
    "representation": "exact_restore",
}
int8 = {
    "hybrid_prefix_enabled": True,
    "representation": "recurrent_int8_per_row",
}
```

All model, tokenizer, TP, sampling, prompt, concurrency, KV capacity, repetitions, source SHA, and GPU assignments must match.

- [ ] **Step 2: Write RED raw-evidence tests**

Require raw rows to include continuation token IDs, final-logit binary files plus hashes/shapes/dtypes, per-rank unique physical and logical bytes, metadata bytes, encode/decode workspace peaks, CUDA allocated/reserved peaks, same-budget capacity, all cache/safety counters, process identity, ports, nonce, and prerequisite hashes. The worker result contains no `GO` field.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py
```

Expected: FAIL because v2 adapter/worker do not exist.

- [ ] **Step 4: Implement explicit profile wiring and raw aggregation**

The P2 adapter calls:

```python
engine.configure_qwen35_hybrid_prefix_publication_runtime(
    model_fingerprint=model_fingerprint,
    max_entries=max_entries,
    max_bytes=max_bytes,
    timeout_s=timeout_s,
    representation="recurrent_int8_per_row",
)
```

Aggregate per-rank bytes by summation, require counter parity where counters are distributed transaction counts, retain rank-local allocator/workspace observations, and reject missing ranks. Correctness runs remain serial; W3 uses the frozen batched fanout only.

- [ ] **Step 5: Run adapter/worker tests**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py
```

Expected: pass with no CUDA initialization in synthetic tests.

### Task 8: Assemble Closed Artifacts and Independently Verify Every Gate

**Files:**
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py`
- Create: `tools/verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py`
- Create: `tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py`

**Interfaces:**
- Consumes: Tasks 6-7 raw evidence.
- Produces: atomically assembled artifact and authoritative independent classification.

- [x] **Step 1: Write RED assembler completeness tests**

Test that missing profile/case/rank/logit/token/cache/workspace/case-local raw
worker receipt evidence prevents publication of the final run directory.
Reject traceback logs, unknown files, symlinks, provenance drift, duplicate
rows, and a producer-authored classification. The global Task 9 execution
receipt is detached authority and must not appear in `run_dir`.

- [x] **Step 2: Write RED verifier recomputation and tamper tests**

Build a complete synthetic fixture whose producer summary lies. Assert `verify_run()` recomputes the correct result from raw rows. Add one test each for tampering with:

```text
artifact manifest
source manifest
prerequisite hash
calibration hash
P1 authority hash
authorization
case-local raw worker receipt
tensor inventory
token IDs
logit bytes
raw timing
raw cache bytes
raw capacity
raw CUDA memory
safety counters
producer summary
thresholds
```

Each integrity or semantic tamper must reject as `INVALID_ARTIFACT`. A
structurally valid artifact that misses a frozen correctness, runtime-safety,
cache/capacity, or performance/memory gate must instead produce the
corresponding legitimate `NO_GO_*` classification.

- [x] **Step 3: Run tests and verify RED**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py
python3 tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py
```

Expected: FAIL because assembler/verifier do not exist.

- [x] **Step 4: Implement independent recomputation**

The verifier must load raw logits and use:

```python
torch.testing.assert_close(
    int8_logits,
    exact_logits,
    atol=2e-5,
    rtol=0.0,
)
```

It must recompute token equality, unique physical byte totals/ratios,
same-budget capacities/ratio, median and every-repetition TTFT ratios, W3
throughput ratios, decode ratio, steady-state peak reserved ratio, and all
forbidden-event counts. It writes `independent_verification.json` and
`report.md` only after all inventory/hash/schema checks pass, and may
legitimately publish `GO`, `NO_GO_CORRECTNESS`,
`NO_GO_RUNTIME_SAFETY`, `NO_GO_CACHE`, or `NO_GO_PERFORMANCE`.

- [x] **Step 5: Run assembler/verifier and v1 regressions**

Run:

```bash
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py
python3 tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_assembler.py
python3 tools/test_verify_qwen35_tp4_hybrid_prefix_benchmark.py
```

Current evidence: schema-v2 assembler `6 passed`, schema-v2 verifier
`16 passed`, and schema-v2 contract `578 passed`. The schema-v1 combined
regression has one pre-existing fixture-only numerical expectation mismatch
(`0.34285714285714286` versus `1/3`); it was not changed because it is
unrelated to schema-v2 authority.

### Task 9: Add Source-Bound Remote Preflight, One-Time Authorization, Executor, and Receipt

**Files:**
- Create: `tools/run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py`
- Create: `tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py`
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_plan.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_plan.py`
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_authorization.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_authorization.py`
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_executor.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_executor.py`
- Create: `tools/qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt.py`
- Create: `tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt.py`

**Interfaces:**
- Consumes: Tasks 6-8.
- Produces: non-destructive preflight, deterministic command plan, one-time authorization, bounded execution evidence, and cryptographic receipt.

- [x] **Step 1: Write RED fixed-resource and prerequisite tests**

Freeze:

```python
SSH_TARGET = "sitian@10.232.195.203"
REQUIRED_GPU_INDICES = (2, 4, 5, 6)
MIN_GPU_FREE_BYTES = 25769803776
KRB5CCNAME = "FILE:/Users/bytedance/krb5cc_sitian"
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
    "-o", "ConnectTimeout=20",
)
```

Assert preflight rejects missing/non-PASS P1 authority, calibration, Gate 1 audit, model/workload/source hashes, or any GPU below the threshold. A blocked preflight must call no stage/launch/kill function and create no remote path.

- [x] **Step 2: Write RED plan/authorization/receipt tests**

Require fresh unique port pairs for all 105 cases, a fresh nonce, one-time
atomic authorization consumption, bounded logs, exact command order,
before/after resource guards, source tar inventory/hash binding, local and
remote independent verification, package hash, and final inventory.

The inventory semantics are frozen as follows:

```text
package_inventory = every regular producer file in
                    ARTIFACT_MANIFEST_HASH_DOMAIN
final_inventory   = every regular producer file in PRODUCER_TRUST_DOMAIN
                    = package_inventory + artifact_manifest.json
verifier outputs  = VERIFIER_TRUST_DOMAIN, outside both producer inventories
execution receipt = detached from run_dir, package contents,
                    final producer inventory, artifact-manifest hash domain,
                    and verifier trust domain
```

The detached receipt binds the package/final inventory hashes but is not
hashed by either inventory. This prevents an artifact-manifest/receipt
self-reference cycle. `package_inventory` and `final_inventory` are closed
regular-file inventories for the declared producer domains, not a partial
top-level sample and not a recursive inventory containing the receipt.

- [x] **Step 3: Run tests and verify RED**

Run:

```bash
python3 tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_plan.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_authorization.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_executor.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt.py
```

Observed: all five test files pass `python3 -m py_compile`; each direct test
entrypoint exits `1` only because its corresponding v2 production module does
not exist.

- [x] **Step 4: Implement preflight and deterministic planning**

Preflight may run only read-only local validation plus a bounded GPU query. It returns either:

```python
{
    "classification": "BLOCKED_RESOURCES",
    "run_tag": run_tag,
    "source_tree_sha256": source_tree_sha256,
    "required_gpu_indices": [2, 4, 5, 6],
    "minimum_free_bytes_per_gpu": 25769803776,
    "gpu_query": gpu_query_rows,
    "blocking_reasons": blocking_reasons,
    "worker_authorization": None,
    "remote_path_created": False,
    "process_started": False,
    "process_killed": False,
}
```

or:

```python
{
    "classification": "READY",
    "run_tag": run_tag,
    "source_bundle": {
        "tar_path": str(source_tar_path),
        "tar_sha256": source_tar_sha256,
        "source_tree_sha256": source_tree_sha256,
    },
    "worker_authorization": {
        "source_tree_sha256": source_tree_sha256,
        "model_manifest_sha256": model_manifest_sha256,
        "workload_manifest_sha256": workload_manifest_sha256,
        "calibration_sha256": calibration_sha256,
        "p1_authority_sha256": p1_authority_sha256,
        "gate1_audit_sha256": gate1_audit_sha256,
        "gpu_indices": [2, 4, 5, 6],
    },
    "gpu_query": gpu_query_rows,
    "minimum_free_bytes_per_gpu": 25769803776,
    "remote_path_created": False,
    "process_started": False,
    "process_killed": False,
}
```

The plan is command-only and performs no SSH. Authorization is a separate one-time file bound to plan SHA, nonce, run tag, ports, GPUs, and all prerequisite hashes.

- [x] **Step 5: Implement executor and receipt**

The executor consumes authorization atomically, executes bounded commands in
declared order, and writes either a complete detached receipt or structured
failure evidence. It never invokes `kill`, `pkill`, or `killall`. Receipt
verification checks both resource guards, command hashes, outputs,
local/remote verifier identities, package inventory, final producer
inventory, and the detached-domain exclusions above. The receipt output path
must resolve outside `run_dir`; it is never copied into the remote package,
downloaded package, extracted final run, artifact manifest, or verifier
output domain.

- [x] **Step 6: Run remote-stack tests without launching remote work**

Run:

```bash
python3 tools/test_run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_plan.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_authorization.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_executor.py
python3 tools/test_qwen35_tp4_hybrid_prefix_benchmark_v2_remote_execution_receipt.py
```

Expected: all synthetic tests pass; no SSH command is executed.

Observed closure evidence (2026-08-05):

- Direct Task 9 suite: `49 passed in 0.61s`.
- Schema-v2 contract regression: `578 passed in 27.82s`.
- `python3 -m py_compile` and scoped `git diff --check`: PASS.
- Final full-file review: 0 P0-P2, Spec PASS, Quality APPROVED.
- Review artifacts:
  `/tmp/TinyLLMForge-adaptive-ngram_task9_final_review_clean_1785938111/report.html`
  and
  `/tmp/TinyLLMForge-adaptive-ngram_task9_final_review_clean_1785938111/report.md`.

The final review cycle additionally froze two filesystem authority properties:

1. physical authority/artifact root identity is the canonical hash of resolved
   path plus directory `st_dev`/`st_ino`, so same-path directory replacement
   is rejected;
2. authorization opens the plan-bound root once and performs all active,
   claim, consumed, and tombstone operations through rooted directory FDs with
   no-follow traversal, so intermediate symlinks and
   identity-check-then-reopen TOCTOU cannot escape or rebind the authority
   domain.

No SSH, GPU, CUDA, remote directory creation, package transfer, extraction, or
runtime authority was executed during Task 9. Task 10 remains pending.

### Task 10: Run the Completion Audit, Then Conditionally Execute Canonical Authority

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Runtime artifacts: a new source-bound output directory under the existing Qwen3.5 benchmark artifact root; never overwrite P1 or an earlier P2 run.

**Interfaces:**
- Consumes: all previous tasks plus authoritative calibration and P1 artifacts.
- Produces: either a verified `BLOCKED_RESOURCES` record or a complete independently classified P2 artifact.

- [ ] **Step 1: Run all local Gate 1 and Gate 2 tests**

Run every command from Tasks 5-9, followed by:

```bash
python3 -m py_compile \
  tinyvllm/engine/qwen35_hybrid_prefix_representation.py \
  tinyvllm/engine/qwen35_hybrid_prefix_int8_cache.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_v2_contract.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_v2_worker.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_v2_engine_adapter.py \
  tools/qwen35_tp4_hybrid_prefix_benchmark_v2_assembler.py \
  tools/verify_qwen35_tp4_hybrid_prefix_benchmark_v2.py \
  tools/run_qwen35_tp4_hybrid_prefix_benchmark_v2_remote.py
git diff --check
```

Expected: all tests pass, modules compile, and `git diff --check` is silent. Do not repair unrelated failures.

- [ ] **Step 2: Verify canonical prerequisites before any SSH**

Run the independent calibration verifier, independent strict-P1 verifier, Gate 1 audit verifier, and v2 preflight in that order. Expected prerequisite state:

```text
calibration: PASS
strict P1 independent authority: GO
Gate 1 audit: PASS
v2 preflight: READY or BLOCKED_RESOURCES
```

If calibration or P1 is absent/non-authoritative, record `INVALID_ARTIFACT` for the attempted P2 authority setup and stop. If resources are insufficient, record `BLOCKED_RESOURCES` and stop without creating a remote directory.

- [ ] **Step 3: If and only if preflight is READY, build and inspect the command plan**

Use a fresh run tag, fresh source bundle, fresh ports, and fresh nonce. Build the deterministic plan, verify it locally, and inspect that it contains only GPUs `2,4,5,6`, the fixed SSH options, unique artifact paths, and no process-kill command.

- [ ] **Step 4: Obtain and consume one-time authorization**

Create authorization only after the plan SHA and all bindings are final. Verify authorization, then pass it to the executor exactly once. Reuse must fail.

- [ ] **Step 5: Execute the canonical matrix and verify twice**

The executor runs P0/P1/P2 W0-W4 with frozen repetitions and matched configuration. Remote verification uses the staged source verifier. After download, local verification extracts a fresh verifier source bundle and independently re-verifies the artifact. Both verification outputs and their source hashes must appear in the execution receipt.

- [ ] **Step 6: Interpret the result without overclaiming**

Report exactly one closed classification. For `GO`, state only the bounded claim from the approved spec and include the bound model, hardware, source, workloads, configuration, thresholds, and artifact hashes. For every non-GO result, state which raw gate failed. `BLOCKED_RESOURCES` and `INVALID_ARTIFACT` form no runtime conclusion.

- [ ] **Step 7: Update the handoff**

Append:

```markdown
## Qwen3.5 recurrent INT8 runtime/cache authority

- Gate 1 status and exact local commands
- Source SHA and dirty-tree policy
- Codec/representation/version
- Calibration artifact and independent result
- P1 authority artifact and independent result
- P2 preflight/run tag/GPU inventory
- Authorization and execution receipt hashes
- Remote and local verifier hashes/results
- Final classification
- Correctness/cache/capacity/performance/memory/safety metrics
- Claim boundary and limitations
- Exact next action
```

Do not write “faster”, “more cache efficient”, “memory saving”, “accuracy preserved”, or an equivalent benefit statement unless the final independently verified classification is `GO`.

---

## Final Implementation Audit

- [ ] P1 exact remains the default when every new argument is omitted.
- [ ] P1 snapshot class, `storage_bytes`, namespace, counters, schema-v1 contract, and schema-v1 verifier remain unchanged.
- [ ] Config has a dedicated Qwen3.5 hybrid-prefix representation field; KV quantization cannot enable P2.
- [ ] P2 resident cache contains exact BF16 convolution, INT8 recurrent values, and FP32 scales only.
- [ ] P2 key binds representation, representation version, codec, model/layout/prefix/TP identity, and dtype.
- [ ] Exactly 18 ordered unique layers are required for publication and restore.
- [ ] Publication is invisible until finalize and rollback preserves the previous valid entry.
- [ ] Encoded interning requires metadata plus byte equality; digest collision cannot alias tensors.
- [ ] Reader lease pins detached/quarantined snapshots until release.
- [ ] All recurrent layers decode into private FP32 staging before one transaction commit.
- [ ] Decode/corruption failure quarantines and recomputes without P1 fallback.
- [ ] Commit failure rolls back all live rows; rollback failure is explicit.
- [ ] Logical, encoded physical, metadata, encode workspace, decode workspace, allocated, and reserved bytes are separate.
- [ ] Runtime events always expose representation and codec identity plus all forbidden counters.
- [ ] Gate 1 CPU/runtime tests and P1 regressions pass before schema-v2 work.
- [ ] Schema-v2 contains exactly P0/P1/P2 and freezes all approved thresholds.
- [ ] Producer stores raw evidence and never decides `GO`.
- [ ] Independent verifier reconstructs token/logit/cache/capacity/performance/memory/safety gates.
- [ ] Unknown/missing/tampered evidence fails closed.
- [ ] Remote preflight is source-bound, fixed-GPU, non-destructive, and resource-blocked runs create no remote path.
- [ ] One-time authorization, fresh ports/nonce/path, and execution receipt are verified.
- [ ] Canonical execution occurs only after authoritative calibration and P1 plus fresh `READY` resources.
- [ ] `AGENT_HANDOFF_STATE.md` records commands, results, hashes, limitations, and next action.
- [ ] No unrelated files were cleaned, staged, committed, or overwritten.
