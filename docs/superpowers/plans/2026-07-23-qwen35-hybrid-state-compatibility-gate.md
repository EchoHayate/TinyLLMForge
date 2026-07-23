# Qwen3.5 Hybrid-State Compatibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's approved execution mode is inline; do not dispatch subagents.

**Goal:** Build and run a source-bound reference gate that determines whether Qwen3.5-2B hybrid recurrent, convolution, and full-attention state can be represented by an explicit heterogeneous TinyLLMForge request-state contract with correct continuation, request isolation, lifecycle, and physical-memory accounting.

**Architecture:** Keep the production engine unchanged. Add a dependency-light frozen contract, an isolated Transformers reference probe, a non-destructive remote orchestrator, and an independent local verifier; the probe emits raw tensor/lifecycle/correctness evidence while the verifier reconstructs the required domain, storage ledger, invariants, and final `GO | NO_GO | INCOMPLETE` classification without trusting worker aggregates.

**Tech Stack:** Python 3, dataclasses, PyTorch, Transformers, JSON/JSONL, SHA-256, Qwen3.5-2B official reference implementation, CUDA memory APIs, SSH ControlMaster, dependency-light script tests.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; never modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- Preserve the existing modified `AGENT_HANDOFF_STATE.md` and all unrelated untracked `experiments/` directories until the final evidence task explicitly updates the handoff.
- Stage exact paths only; never use `git add -A`.
- The approved design is `docs/superpowers/specs/2026-07-23-qwen35-hybrid-state-compatibility-gate-design.md`.
- Do not modify `tinyvllm/models/`, `tinyvllm/engine/model_runner.py`, the scheduler, block tables, production cache allocation, or any production inference path in this implementation.
- Do not combine this gate with Light Doc Cache, Attention Matching, KV quantization, KV offload, speculative decoding, CUDA Graphs, token sparsity, low rank, recurrent-state compression, or sparse-kernel benchmarking.
- Do not update `README.md` or claim TinyLLMForge Qwen3.5 support, compression, quality retention, speedup, throughput, latency, or physical-memory reduction from this gate.
- Compatibility `GO` means only that the official reference state is fully characterized and representable by the exported semantic contract.
- Use only the official `Qwen/Qwen3.5-2B` model source. Resolve a commit SHA before acquisition; mutable aliases are never canonical evidence.
- The acquisition method is `huggingface_hub.snapshot_download()` with the runner's validated `resolved_revision`, `local_dir=f"{remote_run_dir}/model"`, `local_dir_use_symlinks=False`, and an allow-list containing model config, tokenizer/generation metadata, index files, safetensors shards, and required reference Python files.
- Never use another user's cache as the canonical path. A complete existing snapshot may be copied only when its immutable revision and every required file hash are independently verified; otherwise use the unique run-local model directory.
- Freeze acquisition peak bytes as `declared_model_file_bytes + declared_model_file_bytes + 512 MiB artifact allowance + 2 GiB safety reserve`. The second model-size term is temporary-download overhead. If free bytes on the exact target filesystem are below that value, classify `INCOMPLETE_RESOURCE_BLOCKED` and stop before download or GPU work.
- Re-measure free bytes immediately before acquisition. The previous approximately `9.41 GiB` observation is informational only and must not be reused.
- Never delete model caches, package caches, shared `/tmp`, remote files from other runs, or remote checkout content.
- GPU/model work runs only on `sitian@10.232.195.203` as user `sitian`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use only `CUDA_VISIBLE_DEVICES=0`.
- Every model process receives fresh, mutually distinct `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Retry only when stderr contains the exact string `EADDRINUSE`; allow at most three attempts and allocate a fresh port pair for each attempt.
- Do not use `rsync`, modify the remote checkout, run `kill` or `pkill`, switch GPUs, or clean shared paths.
- Use `remote_run_dir = f"/data00/home/sitian/sitian-workspace01/tllm/qwen35-hybrid-state-runs/{run_tag}"` after validating `run_tag` against `[A-Za-z0-9_-]+`.
- Source staging must come from one clean approved local commit, use a tar stream, verify local and remote SHA-256, and reject untracked remote leftovers as dependencies.
- The reference may use the official PyTorch fallback. Missing `fla`, `causal_conv1d`, Triton, or FlashAttention must be recorded and prevents performance interpretation, but does not itself block compatibility if the fallback works.
- Unsupported model/runtime semantics, absent artifacts, insufficient disk, dependency/acquisition failure, worker failure, timeout, missing rows, unexplained state, or verifier failure classify `INCOMPLETE`, not `NO_GO`.
- Exact decoded-token equality is mandatory for every correctness comparison.
- Logit tolerance is not guessed. The canonical tolerance is derived from the same-path repeatability phase as `atol = max(1e-6, 4 * observed_max_abs_diff)` and `rtol = max(1e-6, 4 * observed_max_rel_diff)`, capped at `atol <= 1e-3` and `rtol <= 1e-4`; exceeding either cap classifies `INCOMPLETE_NUMERICAL_INSTABILITY`.
- The verifier may trust bounded tensor and logit hashes emitted by the worker, but never worker-computed counts, byte totals, invariant booleans, or final classification.

---

## File Map

- Create `tools/qwen35_hybrid_state_contract.py`: frozen model identity rules, prompt/chunk matrix, schemas, state roles, dtype sizes, expected row generation, tolerance derivation, canonical JSON, and fail-closed classification helpers.
- Create `tools/qwen35_hybrid_state_probe.py`: isolated official-reference loader, architecture inspection, deterministic execution modes, recursive state discovery, normalized tensor snapshots, state export/import, CUDA snapshots, and artifact emission.
- Create `tools/verify_qwen35_hybrid_state_gate.py`: independent source/model/environment/artifact/domain/correctness/lifecycle/storage-ledger reconstruction and authoritative classification.
- Create `tools/run_qwen35_hybrid_state_gate_remote.py`: clean-source staging, read-only preflight, immutable model acquisition, dynamic ports, exact retry policy, remote worker launch, chunked-safe download, and local verifier invocation.
- Create `tools/test_qwen35_hybrid_state_contract.py`: dependency-light contract, matrix, schema, tolerance, byte-accounting, and classification tests.
- Create `tools/test_qwen35_hybrid_state_probe.py`: CPU/synthetic tests for recursive tensor normalization, alias accounting inputs, mutation transitions, request generations, export/import serialization, and artifact schema.
- Create `tools/test_verify_qwen35_hybrid_state_gate.py`: complete synthetic run builder plus tamper tests for every authoritative verifier guard.
- Create `tools/test_run_qwen35_hybrid_state_gate_remote.py`: static and unit tests for host/runtime binding, clean staging, acquisition safety, dynamic ports, retry limits, remote prohibitions, partial-artifact preservation, and verifier invocation.
- Create canonical raw runs under `experiments/qwen35_hybrid_state/{run_id}/`; never stage raw run directories.
- Create `docs/qwen35_hybrid_state_evidence_registry.json` only after canonical evidence exists: tracked closed-schema index of the approved spec, implementation plan, source commit, raw artifact path/hash, verifier path/hash, classification, and claim boundary.
- Modify `AGENT_HANDOFF_STATE.md` only after canonical verification: append exact implementation commits, commands, evidence path, classification, what the result proves, what it does not prove, and the next authorized gate.

## Shared Interfaces

Use these exact frozen constants in `tools/qwen35_hybrid_state_contract.py`:

```python
SCHEMA_VERSION = 1
MODEL_REPOSITORY = "Qwen/Qwen3.5-2B"
EXPECTED_NUM_HIDDEN_LAYERS = 24
EXPECTED_LINEAR_LAYERS = 18
EXPECTED_FULL_ATTENTION_LAYERS = 6
EXPECTED_FULL_ATTENTION_INTERVAL = 4
EXPECTED_LINEAR_NUM_KEY_HEADS = 16
EXPECTED_LINEAR_NUM_VALUE_HEADS = 16
EXPECTED_LINEAR_KEY_HEAD_DIM = 128
EXPECTED_LINEAR_VALUE_HEAD_DIM = 128
EXPECTED_LINEAR_CONV_KERNEL_DIM = 4
EXPECTED_MAMBA_SSM_DTYPE = "float32"

PROMPT_LENGTHS = (17, 65, 257, 1025)
DECODE_STEPS = 8
SAME_PATH_REPEATS = 2
CHUNK_TEMPLATES = (
    (1,),
    (3, 5),
    (31, 34),
    (64,),
)
MULTI_REQUEST_LENGTHS = (17, 65, 257)
SLOT_REUSE_PROMPT_LENGTH = 33
STATE_ROLES = (
    "full_attention_key",
    "full_attention_value",
    "linear_recurrent_state",
    "linear_convolution_state",
    "position_or_sequence_metadata",
    "other_persistent_state",
)
UPDATE_KINDS = (
    "created",
    "unchanged",
    "replaced",
    "grown",
    "mutated_in_place",
    "released",
)
FINAL_CLASSIFICATIONS = ("GO", "NO_GO", "INCOMPLETE")
```

Generate deterministic prompt IDs without depending on tokenizer text:

```python
def deterministic_token_ids(
    *,
    length: int,
    vocab_size: int,
    seed: int,
    forbidden_ids: set[int],
) -> tuple[int, ...]:
    modulus = max(1, vocab_size - 256)
    values = []
    cursor = seed
    while len(values) < length:
        candidate = 128 + ((cursor * 1103515245 + 12345) % modulus)
        cursor += 1
        if candidate < vocab_size and candidate not in forbidden_ids:
            values.append(candidate)
    return tuple(values)
```

Use these exact normalized records:

```python
@dataclass(frozen=True)
class GateCase:
    phase: str
    case_id: str
    execution_mode: str
    prompt_length: int
    chunk_schedule: tuple[int, ...]
    request_count: int
    decode_steps: int
    repeat_index: int
    expected_state_snapshots: int


@dataclass(frozen=True)
class StateComponent:
    request_id: str
    request_generation: int
    layer_index: int
    declared_layer_type: str
    state_role: str
    tensor_path: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool
    logical_numel: int
    logical_bytes: int
    storage_data_ptr: int
    storage_offset: int
    storage_nbytes: int
    storage_identity: str
    lifetime_epoch: int
    sequence_length: int
    update_kind: str
    content_sha256: str
```

Use these exact public helpers:

```python
def build_chunk_schedule(
    prompt_length: int,
    prefix_chunks: tuple[int, ...],
) -> tuple[int, ...]


def build_case_matrix() -> tuple[GateCase, ...]


def derive_logit_tolerance(
    repeatability_rows: list[dict],
) -> dict[str, float]


def logical_bytes(shape: tuple[int, ...], dtype: str) -> int


def unique_storage_bytes(components: list[dict]) -> int


def classify_evidence(guards: dict[str, bool], failure_kind: str | None) -> str
```

`build_case_matrix()` must emit these phases before execution:

```text
environment_preflight
architecture_verification
same_path_repeatability
one_shot_vs_cached
one_shot_vs_chunked
state_export_import
interleaved_multi_request
completion_release_slot_reuse
state_memory_ledger
post_run_audit
```

Single-request correctness rows cover all four prompt lengths. Chunked rows cover the 65-, 257-, and 1025-token prompts with every valid schedule produced from the four chunk templates. The multi-request row uses three initial request lengths, finishes the shortest request after two decode steps, increments that slot's `request_generation`, inserts the 33-token replacement request, and continues all live requests until each has at least eight continuation logits.

Every `case_rows.jsonl` row uses these exact keys:

```python
CASE_ROW_FIELDS = (
    "row_id",
    "case_id",
    "phase",
    "execution_mode",
    "prompt_length",
    "chunk_schedule",
    "request_count",
    "decode_steps",
    "repeat_index",
    "request_ids",
    "request_generations",
    "decoded_token_ids",
    "logit_records",
    "state_snapshot_ids",
    "memory_snapshot_ids",
    "complete",
    "failure_kind",
    "failure_detail",
)
```

Every logit record contains:

```python
LOGIT_RECORD_FIELDS = (
    "request_id",
    "request_generation",
    "step_index",
    "full_logit_sha256",
    "topk_token_ids",
    "topk_logits",
    "max_abs_diff",
    "mean_abs_diff",
    "max_rel_diff",
    "mean_rel_diff",
    "sequence_length",
    "position_metadata",
)
```

The probe CLI is:

```text
python3 tools/qwen35_hybrid_state_probe.py
  inspect-model|run-canonical
  --model-dir MODEL_DIR
  --run-dir RUN_DIR
  --contract-sha256 SHA256
```

The remote runner CLI is:

```text
python3 tools/run_qwen35_hybrid_state_gate_remote.py
  preflight|acquire|smoke|canonical|download-only|verify-only
  --run-tag RUN_TAG
  [--resolved-revision 40_HEX_SHA]
```

`preflight` is read-only and never downloads. `acquire` performs only the already-approved immutable acquisition after repeating the disk check. `smoke` runs architecture inspection plus the 17-token cached/export-import case. `canonical` requires a successful smoke artifact bound to the same source and model hashes.

The independent verifier CLI is:

```text
python3 tools/verify_qwen35_hybrid_state_gate.py
  --run-dir RUN_DIR
  --write-report
```

The canonical artifact layout is:

```text
experiments/qwen35_hybrid_state/{run_id}/
  manifest.json
  source_manifest.json
  model_manifest.json
  environment.json
  case_rows.jsonl
  state_snapshots.jsonl
  state_components.jsonl
  memory_snapshots.jsonl
  processes.json
  ports.json
  stdout/
  stderr/
  summary.json
  independent_verification.json
  report.md
```

---

### Task 1: Freeze the Contract, Domain, and Fail-Closed Classification

**Files:**
- Create: `tools/qwen35_hybrid_state_contract.py`
- Create: `tools/test_qwen35_hybrid_state_contract.py`

**Interfaces:**
- Consumes: only the approved design and Python standard library.
- Produces: all constants, dataclasses, schemas, deterministic prompt generation, case-matrix generation, dtype-byte calculation, storage deduplication, tolerance derivation, canonical JSON hashing, and `classify_evidence()` used by every later task.

- [ ] **Step 1: Write failing matrix and chunk-schedule tests**

```python
def test_chunk_schedules_omit_only_zero_remainders():
    assert contract.build_chunk_schedule(65, (64,)) == (64, 1)
    assert contract.build_chunk_schedule(65, (31, 34)) == (31, 34)
    assert contract.build_chunk_schedule(257, (3, 5)) == (3, 5, 249)


def test_case_matrix_is_closed_unique_and_covers_every_phase():
    matrix = contract.build_case_matrix()
    assert len({case.case_id for case in matrix}) == len(matrix)
    assert {case.phase for case in matrix} == set(contract.REQUIRED_PHASES)
    chunked = [case for case in matrix if case.phase == "one_shot_vs_chunked"]
    assert {case.prompt_length for case in chunked} == {65, 257, 1025}
    assert all(case.chunk_schedule for case in chunked)
```

- [ ] **Step 2: Run the focused tests and verify the contract is absent**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_contract.py
```

Expected: FAIL because `qwen35_hybrid_state_contract.py` does not exist.

- [ ] **Step 3: Implement constants, dataclasses, deterministic IDs, and the exact case matrix**

Implement `GateCase`, `StateComponent`, `build_chunk_schedule()`, `deterministic_token_ids()`, and `build_case_matrix()` exactly as defined in Shared Interfaces. Reject non-positive prompt lengths, duplicate case IDs, negative decode steps, chunk sums that do not equal the prompt length, and any phase outside `REQUIRED_PHASES`.

- [ ] **Step 4: Write failing byte-ledger, tolerance, and classification tests**

```python
def test_unique_storage_bytes_deduplicates_aliases_by_device_and_identity():
    rows = [
        {"device": "cuda:0", "storage_identity": "a", "storage_nbytes": 64},
        {"device": "cuda:0", "storage_identity": "a", "storage_nbytes": 64},
        {"device": "cuda:0", "storage_identity": "b", "storage_nbytes": 32},
    ]
    assert contract.unique_storage_bytes(rows) == 96


def test_repeatability_tolerance_is_four_x_observed_and_capped():
    rows = [{"max_abs_diff": 2e-5, "max_rel_diff": 3e-6}]
    assert contract.derive_logit_tolerance(rows) == {
        "atol": 8e-5,
        "rtol": 1.2e-5,
    }
    try:
        contract.derive_logit_tolerance([{
            "max_abs_diff": 1e-3,
            "max_rel_diff": 1e-4,
        }])
    except ValueError as exc:
        assert "INCOMPLETE_NUMERICAL_INSTABILITY" in str(exc)
    else:
        raise AssertionError("unstable repeatability was accepted")


def test_classification_separates_incomplete_from_semantic_no_go():
    passing = {name: True for name in contract.GO_GUARDS}
    assert contract.classify_evidence(passing, None) == "GO"
    assert contract.classify_evidence(
        {**passing, "slot_reuse_pass": False},
        "semantic_failure",
    ) == "NO_GO"
    assert contract.classify_evidence(
        passing,
        "INCOMPLETE_RESOURCE_BLOCKED",
    ) == "INCOMPLETE"
```

- [ ] **Step 5: Implement schemas, dtype bytes, storage deduplication, tolerance derivation, and classification**

Use an explicit dtype-size map for `bool`, `uint8`, `int8`, `int16`, `int32`, `int64`, `float8_e4m3fn`, `float8_e5m2`, `float16`, `bfloat16`, `float32`, and `float64`. Reject unknown dtypes. `classify_evidence()` returns `INCOMPLETE` for any failure kind beginning with `INCOMPLETE_`; returns `NO_GO` only for a complete canonical domain with a false semantic guard; otherwise all `GO_GUARDS` must be exactly `True`.

- [ ] **Step 6: Run contract tests**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_contract.py
```

Expected: PASS and print `qwen35 hybrid-state contract tests passed`.

- [ ] **Step 7: Commit the frozen contract**

```bash
git add tools/qwen35_hybrid_state_contract.py tools/test_qwen35_hybrid_state_contract.py
git commit -m "test: freeze qwen35 hybrid state gate contract"
```

---

### Task 2: Normalize Reference State and Reconstruct Tensor Lifecycles

**Files:**
- Create: `tools/qwen35_hybrid_state_probe.py`
- Create: `tools/test_qwen35_hybrid_state_probe.py`

**Interfaces:**
- Consumes: `StateComponent`, `STATE_ROLES`, `UPDATE_KINDS`, `logical_bytes()`, and canonical JSON helpers from Task 1.
- Produces: `walk_tensor_leaves()`, `classify_state_role()`, `normalize_state_components()`, `compare_state_snapshots()`, `export_normalized_state()`, `import_normalized_state()`, and atomic JSON/JSONL artifact writers used by the real reference execution in Task 3.

- [ ] **Step 1: Write failing recursive-discovery and alias tests**

```python
def test_walk_tensor_leaves_preserves_explicit_paths_and_aliases():
    storage = torch.arange(8, dtype=torch.float32)
    state = {"layers": [{"key": storage[:4], "value": storage[4:]}]}
    leaves = list(probe.walk_tensor_leaves(state))
    assert [path for path, _ in leaves] == [
        "layers[0].key",
        "layers[0].value",
    ]
    assert leaves[0][1].untyped_storage().data_ptr() == (
        leaves[1][1].untyped_storage().data_ptr()
    )


def test_normalization_assigns_request_generation_and_storage_identity():
    tensor = torch.zeros((1, 2, 3), dtype=torch.float32)
    rows = probe.normalize_state_components(
        state={"recurrent_state": tensor},
        request_id="request-a",
        request_generation=2,
        sequence_length=17,
        lifetime_epoch=3,
        layer_schedule={0: "linear_attention"},
    )
    assert rows[0]["request_generation"] == 2
    assert rows[0]["logical_bytes"] == tensor.numel() * tensor.element_size()
    assert rows[0]["storage_identity"]
```

- [ ] **Step 2: Run probe tests and verify missing helpers fail**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_probe.py
```

Expected: FAIL with missing `walk_tensor_leaves` or `normalize_state_components`.

- [ ] **Step 3: Implement recursive tensor discovery and normalized component records**

Walk mappings in sorted key order, sequences by numeric index, dataclasses by field order, named tuples by field name, and object attributes only from an explicit adapter registry. Never walk arbitrary `__dict__` content. Compute `storage_identity` as SHA-256 of canonical `{device, storage_data_ptr, storage_nbytes}` and compute `content_sha256` from a detached contiguous CPU byte serialization.

`classify_state_role()` uses explicit adapter-supplied component names and layer type. Unknown persistent leaves become `other_persistent_state`; they are emitted, never dropped.

- [ ] **Step 4: Write failing transition and serialization tests**

```python
def test_snapshot_comparison_distinguishes_growth_replacement_and_in_place():
    previous = synthetic_components(sequence_length=17)
    current = mutate_synthetic_components(previous)
    transitions = probe.compare_state_snapshots(previous, current)
    assert transitions["full_attention_key"] == "grown"
    assert transitions["linear_recurrent_state"] == "mutated_in_place"
    assert transitions["linear_convolution_state"] == "replaced"


def test_export_import_round_trip_is_ordered_by_request_layer_and_role():
    components = shuffled_synthetic_components()
    payload = probe.export_normalized_state(components)
    restored = probe.import_normalized_state(payload)
    assert [item["layer_index"] for item in restored] == sorted(
        item["layer_index"] for item in restored
    )
    assert contract.canonical_json_sha256(restored) == (
        contract.canonical_json_sha256(
            probe.export_normalized_state(restored)["components"]
        )
    )
```

- [ ] **Step 5: Implement transition detection and framework-neutral export/import**

Compare consecutive records by request generation, layer, role, tensor path, shape, storage identity, storage offset, storage bytes, and content hash. Emit `released` rows for disappeared state and reject a component mapped to two request generations. Export order is `(request_id, request_generation, layer_index, state_role, tensor_path)`.

- [ ] **Step 6: Add exact schema and atomic-write tests**

Create a temporary synthetic run, write `state_snapshots.jsonl`, `state_components.jsonl`, and `memory_snapshots.jsonl`, then assert no `.partial` files remain, every JSONL line parses independently, and every row has exactly the contract fields.

- [ ] **Step 7: Run probe tests**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_probe.py
```

Expected: PASS and print `qwen35 hybrid-state probe unit tests passed`.

- [ ] **Step 8: Commit state normalization**

```bash
git add tools/qwen35_hybrid_state_probe.py tools/test_qwen35_hybrid_state_probe.py
git commit -m "feat: normalize qwen35 reference state"
```

---

### Task 3: Implement Architecture Inspection and Reference Correctness Modes

**Files:**
- Modify: `tools/qwen35_hybrid_state_probe.py`
- Modify: `tools/test_qwen35_hybrid_state_probe.py`

**Interfaces:**
- Consumes: Task 2 normalization/export helpers and Task 1 matrix/tolerance rules.
- Produces: `inspect_model()`, `ReferenceStateAdapter`, `run_one_shot_oracle()`, `run_cached_decode()`, `run_chunked_prefill_decode()`, `run_export_import_continuation()`, `run_interleaved_requests()`, CUDA memory snapshots, and the complete raw canonical artifact set.

- [ ] **Step 1: Write failing architecture-inspection tests with a fake config/model**

```python
def test_inspect_model_reconstructs_exact_hybrid_schedule():
    result = probe.inspect_model(
        model=FakeQwen35Model(),
        config=FakeQwen35Config(),
        tokenizer=FakeTokenizer(),
    )
    assert result["num_hidden_layers"] == 24
    assert result["linear_attention_layers"] == 18
    assert result["full_attention_layers"] == 6
    assert result["full_attention_interval"] == 4
    assert len(result["layer_schedule"]) == 24


def test_architecture_mismatch_fails_before_correctness_execution():
    try:
        probe.require_canonical_architecture({
            "num_hidden_layers": 23,
            "layer_schedule": [],
        })
    except probe.IncompleteRun as exc:
        assert exc.failure_kind == "INCOMPLETE_ARCHITECTURE"
    else:
        raise AssertionError("architecture mismatch was accepted")
```

- [ ] **Step 2: Implement official model/tokenizer loading and exact architecture verification**

Load from the run-local immutable model path using:

```python
config = AutoConfig.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    local_files_only=True,
    trust_remote_code=False,
    torch_dtype="auto",
    device_map={"": "cuda:0"},
)
```

Record selected classes and actual parameter dtypes. Reconstruct layer types from the loaded config/model rather than assuming interval arithmetic. Require every expected Qwen3.5 field and exact schedule counts; mismatch raises `IncompleteRun("INCOMPLETE_ARCHITECTURE", detail)`.

- [ ] **Step 3: Write failing same-path, cached, chunked, and export/import tests using a deterministic fake reference**

```python
def test_reference_modes_emit_comparable_step_records():
    adapter = FakeReferenceStateAdapter()
    oracle = probe.run_one_shot_oracle(adapter, token_ids=(1, 2, 3), decode_steps=2)
    cached = probe.run_cached_decode(adapter, token_ids=(1, 2, 3), decode_steps=2)
    chunked = probe.run_chunked_prefill_decode(
        adapter,
        token_ids=(1, 2, 3),
        chunk_schedule=(1, 2),
        decode_steps=2,
    )
    assert cached["decoded_token_ids"] == oracle["decoded_token_ids"]
    assert chunked["decoded_token_ids"] == oracle["decoded_token_ids"]
    assert len(cached["state_snapshot_ids"]) == 3


def test_export_import_preserves_next_step_logits():
    adapter = FakeReferenceStateAdapter()
    result = probe.run_export_import_continuation(
        adapter,
        token_ids=(1, 2, 3),
    )
    assert result["decoded_token_ids_equal"] is True
    assert result["full_logit_sha256_equal"] is True
```

- [ ] **Step 4: Implement `ReferenceStateAdapter` against the loaded Transformers cache API**

The adapter must make cache conversion explicit:

```python
class ReferenceStateAdapter:
    def prefill(
        self,
        input_ids: torch.Tensor,
        state: object | None,
    ) -> tuple[torch.Tensor, object]

    def decode_one(
        self,
        token_id: int,
        state: object,
        sequence_length: int,
    ) -> tuple[torch.Tensor, object]

    def export_state(
        self,
        state: object,
        request_id: str,
        request_generation: int,
        sequence_length: int,
    ) -> dict

    def import_state(self, exported: dict) -> object
```

Use feature detection for official cache methods, but fail closed with `INCOMPLETE_REFERENCE_SEMANTICS` when explicit export/import cannot be implemented. Do not infer layer identity from list position alone: bind every state component to the inspected layer schedule.

- [ ] **Step 5: Implement deterministic correctness evidence**

For each step, detach logits to contiguous CPU `float32`, hash the entire serialized tensor, record top-20 IDs/logits, and compute absolute/relative differences against the oracle. Use greedy next-token selection so every execution mode follows the same accepted-token path. Reset `torch.cuda` peak stats before each case.

- [ ] **Step 6: Write failing interleaving, release, and slot-reuse tests**

```python
def test_interleaved_decode_does_not_mutate_inactive_requests():
    result = probe.run_interleaved_requests(FakeReferenceStateAdapter())
    assert result["inactive_request_hash_changes"] == []


def test_slot_reuse_increments_generation_and_releases_old_state():
    result = probe.run_interleaved_requests(FakeReferenceStateAdapter())
    assert result["slot_generations"]["slot-0"] == [0, 1]
    assert result["released_generations"] == [["slot-0", 0]]
    assert result["stale_state_reads"] == []
```

- [ ] **Step 7: Implement three-request interleaving and explicit release**

Keep one independent state object per request. Before decoding the active request, hash all inactive state; hash them again afterward and emit any mutation. After request 0 completes, emit `released` transitions, drop every strong reference to generation 0, call `gc.collect()`, synchronize CUDA for the memory snapshot, increment the generation, initialize the replacement request from `None`, and prove no old storage identity or content hash is read.

- [ ] **Step 8: Implement memory snapshots and canonical artifact emission**

Record CUDA allocated/reserved bytes at:

```text
before_model_load
after_model_load
before_prefill
after_prefill
after_each_decode_step
after_request_release
after_slot_reuse
after_model_release
```

Also record `max_memory_allocated`, `max_memory_reserved`, parameter bytes, state logical bytes, state unique storage bytes, and non-state peak as a labeled allocator observation only. Never call allocator deltas exact state bytes.

- [ ] **Step 9: Run all probe tests**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_probe.py
```

Expected: PASS and print `qwen35 hybrid-state probe unit tests passed`.

- [ ] **Step 10: Commit reference execution**

```bash
git add tools/qwen35_hybrid_state_probe.py tools/test_qwen35_hybrid_state_probe.py
git commit -m "feat: add qwen35 reference compatibility probe"
```

---

### Task 4: Build the Independent Verifier and Tamper Matrix

**Files:**
- Create: `tools/verify_qwen35_hybrid_state_gate.py`
- Create: `tools/test_verify_qwen35_hybrid_state_gate.py`

**Interfaces:**
- Consumes: the frozen contract only; it imports no worker classification or aggregate helper.
- Produces: `verify_run()`, authoritative `independent_verification.json`, `report.md`, and a complete synthetic/tamper test matrix.

- [ ] **Step 1: Create a complete synthetic artifact builder and failing happy-path test**

```python
def test_complete_synthetic_domain_verifies_go():
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = fixtures.write_complete_run(Path(tmp))
        result = verifier.verify_run(run_dir, write_report=True)
        assert result["classification"] == "GO"
        assert result["expected_case_count"] == len(contract.build_case_matrix())
        assert (run_dir / "independent_verification.json").is_file()
        assert (run_dir / "report.md").is_file()
```

The fixture must generate every expected case, state snapshot, component, memory snapshot, process, port, and manifest entry; a tiny subset is not an acceptable proxy.

- [ ] **Step 2: Implement artifact inventory and provenance verification**

Require exactly the canonical files/directories. Recompute each listed file's size and SHA-256, reject unlisted classification inputs, verify source commit/branch/clean state, local/remote source hashes, official model repository/resolved revision/file hashes, tokenizer/config identities, host/user/GPU/runtime fields, `CUDA_VISIBLE_DEVICES=0`, distinct process ports, and process exit codes.

- [ ] **Step 3: Add tamper tests for domain closure and provenance**

```python
def test_provenance_and_domain_tampering_is_incomplete():
    cases = (
        (remove_case_row, "missing canonical case"),
        (duplicate_case_row, "duplicate canonical case"),
        (add_unknown_case, "unknown canonical case"),
        (alter_source_hash, "source hash mismatch"),
        (alter_model_revision, "model revision mismatch"),
        (reuse_port, "port reuse"),
        (add_unlisted_input, "unlisted artifact"),
    )
    for mutator, message in cases:
        expect_incomplete(mutator, message)
```

- [ ] **Step 4: Implement independent correctness and tolerance reconstruction**

Reconstruct the same-path tolerance from raw repeatability rows, enforce the caps, then check exact token equality plus every per-step absolute/relative statistic for cached, chunked, and export/import comparisons. Recompute expected sequence lengths and reject silently skipped invalid schedules.

- [ ] **Step 5: Add correctness tamper tests**

Test wrong token, missing step, changed full-logit hash, tolerance above cap, chunk schedule mismatch, export/import mismatch, and non-finite top-k values. Wrong semantics in a complete domain must produce `NO_GO`; unavailable/corrupt evidence must produce `INCOMPLETE`.

- [ ] **Step 6: Implement state-role, lifecycle, isolation, and slot-reuse reconstruction**

Require exactly one request generation per component row; explicit request/layer ordering; explained roles; bounded recurrent/convolution shapes across prompt lengths; full-attention K/V growth with sequence length; inactive-request hash stability; released old generation before reuse; new generation initialized without old storage/content; and fail closed on `other_persistent_state`.

- [ ] **Step 7: Add lifecycle tamper tests**

Cover unexplained role, recurrent growth, full-KV non-growth, cross-request mutation, missing release, unchanged generation on reuse, stale storage identity reuse, stale content reuse, ambiguous layer identity, and unsupported update kind.

- [ ] **Step 8: Implement independent logical and physical storage ledgers**

Recompute logical bytes from shape and dtype. Deduplicate physical bytes by `(device, storage_identity)` and require consistent `storage_nbytes` for every alias. Compare reconstructed totals by role/layer/request/epoch to raw components; never trust worker totals. Validate allocator snapshots independently and ensure all required epochs exist.

- [ ] **Step 9: Add ledger tamper tests**

Cover incorrect logical bytes, conflicting alias storage sizes, double-counted storage, missing memory epoch, negative allocator bytes, parameter-byte mismatch, and worker aggregate disagreement.

- [ ] **Step 10: Implement authoritative classification and report**

`GO` requires every design guard. `NO_GO` is limited to complete-domain semantic incompatibility. Every provenance, resource, execution, schema, row, artifact, or reconstruction defect is `INCOMPLETE`. Include machine-readable reasons and a report section titled `Claim Boundary` that explicitly prohibits support/compression/performance claims.

- [ ] **Step 11: Run verifier tests**

Run:

```bash
python3 tools/test_verify_qwen35_hybrid_state_gate.py
```

Expected: PASS and print `qwen35 hybrid-state verifier tests passed`.

- [ ] **Step 12: Commit the independent verifier**

```bash
git add tools/verify_qwen35_hybrid_state_gate.py tools/test_verify_qwen35_hybrid_state_gate.py
git commit -m "test: add qwen35 hybrid state independent verifier"
```

---

### Task 5: Implement the Non-Destructive Remote Runner

**Files:**
- Create: `tools/run_qwen35_hybrid_state_gate_remote.py`
- Create: `tools/test_run_qwen35_hybrid_state_gate_remote.py`

**Interfaces:**
- Consumes: Tasks 1-4 source files and CLIs.
- Produces: immutable source staging, read-only preflight, approved immutable acquisition, smoke/canonical execution, artifact preservation/download, and verifier invocation.

- [ ] **Step 1: Write failing static host, mode, and prohibition tests**

```python
def test_runner_binds_exact_remote_identity_and_modes():
    source = RUNNER_PATH.read_text()
    for required in (
        "sitian@10.232.195.203",
        "/tmp/ssh-sitian-10.232.195.203",
        "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python",
        "CUDA_VISIBLE_DEVICES",
        "qwen35-hybrid-state-runs",
    ):
        assert required in source
    for mode in ("preflight", "acquire", "smoke", "canonical", "download-only", "verify-only"):
        assert f'"{mode}"' in source


def test_runner_forbids_remote_mutation_and_cleanup():
    source = RUNNER_PATH.read_text()
    for forbidden in ("rsync ", "pkill", "killall", "rm -rf", "git checkout", "git reset", "git clean"):
        assert forbidden not in source
```

- [ ] **Step 2: Implement CLI constants, safe run tags, SSH commands, and exact source ownership**

Stage only:

```python
OWNED_SOURCE_FILES = (
    "tools/qwen35_hybrid_state_contract.py",
    "tools/qwen35_hybrid_state_probe.py",
    "tools/verify_qwen35_hybrid_state_gate.py",
    "tools/run_qwen35_hybrid_state_gate_remote.py",
    "tools/test_qwen35_hybrid_state_contract.py",
    "tools/test_qwen35_hybrid_state_probe.py",
    "tools/test_verify_qwen35_hybrid_state_gate.py",
    "tools/test_run_qwen35_hybrid_state_gate_remote.py",
)
```

Require `git status --porcelain -- "${OWNED_SOURCE_FILES[@]}"` to be empty and record the approved commit. Create a local tar from exact files, stream it over SSH, and verify every remote staged file hash before preflight.

- [ ] **Step 3: Write failing preflight/acquisition policy tests**

```python
def test_preflight_is_read_only_and_computes_frozen_peak_bytes():
    result = runner.evaluate_disk_preflight(
        declared_model_file_bytes=4 * GIB,
        free_bytes=11 * GIB,
    )
    assert result["required_bytes"] == (4 * GIB * 2) + (512 * MIB) + (2 * GIB)
    assert result["can_acquire"] is True


def test_insufficient_disk_stops_before_download_or_gpu():
    result = runner.evaluate_disk_preflight(
        declared_model_file_bytes=4 * GIB,
        free_bytes=8 * GIB,
    )
    assert result["classification_detail"] == "INCOMPLETE_RESOURCE_BLOCKED"
```

- [ ] **Step 4: Implement read-only preflight and immutable revision resolution**

Preflight records checked cache roots, candidate snapshots, free bytes on the exact run filesystem, GPU processes, package versions, and dependency availability. Resolve the official model revision using Hugging Face metadata without downloading weights; require a 40-hex commit. Query the immutable revision's file metadata to sum declared sizes and construct the allow-list. Do not acquire in `preflight`.

- [ ] **Step 5: Implement approved acquisition with repeated disk check**

`acquire` requires `--resolved-revision`, repeats metadata and free-space checks, verifies the revision matches current official metadata, then calls a small remote Python acquisition script using the exact `snapshot_download()` arguments from Global Constraints. Hash every acquired file and reject incomplete shard/index sets.

- [ ] **Step 6: Write failing port allocation and retry tests**

```python
def test_port_pairs_are_globally_unique_and_distinct():
    pairs = runner.allocate_unique_port_pairs(3, allocator=fake_allocator())
    assert len({port for pair in pairs for port in pair}) == 6
    assert all(dist != master for dist, master in pairs)


def test_only_exact_eaddrinuse_is_retryable_and_attempts_are_capped():
    assert runner.is_retryable_port_collision(1, "EADDRINUSE") is True
    assert runner.is_retryable_port_collision(1, "Address already in use") is False
    assert runner.MAX_PORT_ATTEMPTS == 3
```

- [ ] **Step 7: Implement smoke/canonical worker launch and partial preservation**

Run source tests remotely before GPU launch. Allocate a fresh port pair for each process/attempt, set environment explicitly, capture exact command/stdout/stderr/exit code, retry only exact `EADDRINUSE`, and atomically publish artifacts. On any failure, preserve and download all available files and synthesize an `INCOMPLETE` manifest rather than deleting the run.

- [ ] **Step 8: Implement chunked-safe artifact download and local verification**

List remote artifact files, reject unsafe paths, download in fixed-size blocks with byte-count verification and retries, support zero-byte files, then run:

```bash
python3 tools/verify_qwen35_hybrid_state_gate.py \
  --run-dir "experiments/qwen35_hybrid_state/${RUN_TAG}" \
  --write-report
```

Record verifier stdout, stderr, exit code, and hash. `verify-only` must perform no SSH or source staging; `download-only` must perform no source upload or process launch.

- [ ] **Step 9: Run remote-runner tests**

Run:

```bash
python3 tools/test_run_qwen35_hybrid_state_gate_remote.py
```

Expected: PASS and print `qwen35 hybrid-state remote runner tests passed`.

- [ ] **Step 10: Commit the remote runner**

```bash
git add tools/run_qwen35_hybrid_state_gate_remote.py tools/test_run_qwen35_hybrid_state_gate_remote.py
git commit -m "feat: add qwen35 hybrid state remote gate runner"
```

---

### Task 6: Run Local Verification and Source-Bound Remote Preflight

**Files:**
- No production code changes.
- Raw output: `experiments/qwen35_hybrid_state/${PREFLIGHT_RUN_TAG}/`

**Interfaces:**
- Consumes: committed Tasks 1-5.
- Produces: proof that the local implementation is green plus a current read-only remote resource/model/runtime decision.

- [ ] **Step 1: Run the complete local gate test suite**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_contract.py
python3 tools/test_qwen35_hybrid_state_probe.py
python3 tools/test_verify_qwen35_hybrid_state_gate.py
python3 tools/test_run_qwen35_hybrid_state_gate_remote.py
git diff --check
```

Expected: all tests PASS; `git diff --check` emits no output.

- [ ] **Step 2: Confirm the owned source is committed and unrelated work is preserved**

Run:

```bash
git status --short
git log --oneline -6
```

Expected: no changes under the eight owned `tools/` files or this plan; the pre-existing `AGENT_HANDOFF_STATE.md` modification and unrelated experiment directories remain untouched.

- [ ] **Step 3: Run read-only remote preflight**

Run:

```bash
python3 tools/run_qwen35_hybrid_state_gate_remote.py preflight \
  --run-tag qwen35-2b-hybrid-preflight-$(date +%Y%m%d-%H%M%S)
```

Expected: a local preflight artifact with current free bytes, immutable revision candidate, model declared bytes, required acquisition peak bytes, cache candidates, package/runtime identity, GPU process inventory, and one of:

```text
READY_EXISTING_SNAPSHOT
READY_TO_ACQUIRE
INCOMPLETE_RESOURCE_BLOCKED
INCOMPLETE_MODEL_METADATA
INCOMPLETE_RUNTIME
```

- [ ] **Step 4: Independently inspect the preflight artifact**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

runs = sorted(Path("experiments/qwen35_hybrid_state").glob("*preflight*"))
assert runs, "missing preflight run"
run = runs[-1]
payload = json.loads((run / "summary.json").read_text())
required = {
    "status",
    "resolved_revision",
    "declared_model_file_bytes",
    "free_bytes",
    "required_acquisition_peak_bytes",
    "runtime",
    "gpu_processes",
}
assert required <= payload.keys(), sorted(required - payload.keys())
print(run)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
```

Expected: all required fields exist. Do not proceed to acquisition when status begins with `INCOMPLETE_`.

- [ ] **Step 5: Record the preflight decision in a dedicated commit only if tracked source changed**

No raw preflight artifacts are staged. If no tracked source changed, do not create an empty commit.

---

### Task 7: Acquire the Immutable Snapshot, Run Smoke, Then Canonical

**Files:**
- Raw output: `experiments/qwen35_hybrid_state/${SMOKE_RUN_TAG}/`
- Raw output: `experiments/qwen35_hybrid_state/${CANONICAL_RUN_TAG}/`

**Interfaces:**
- Consumes: a Task 6 `READY_EXISTING_SNAPSHOT` or `READY_TO_ACQUIRE` decision.
- Produces: source/model/environment-bound smoke and canonical evidence. This task stops at the first real blocker and preserves `INCOMPLETE` evidence.

- [ ] **Step 1: Acquire only when preflight authorized it**

For `READY_TO_ACQUIRE`, run with the exact 40-hex revision from preflight:

```bash
RESOLVED_REVISION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_revision"])' \
  "experiments/qwen35_hybrid_state/${PREFLIGHT_RUN_TAG}/summary.json")"
python3 tools/run_qwen35_hybrid_state_gate_remote.py acquire \
  --run-tag "qwen35-2b-hybrid-acquire-$(date +%Y%m%d-%H%M%S)" \
  --resolved-revision "${RESOLVED_REVISION}"
```

Expected: repeated disk preflight passes, complete run-local snapshot is acquired, every file is hashed, and `model_manifest.json` binds the immutable revision. If disk or acquisition fails, preserve `INCOMPLETE` and stop.

- [ ] **Step 2: Run the smoke domain**

Run:

```bash
SMOKE_RUN_TAG="qwen35-2b-hybrid-smoke-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_qwen35_hybrid_state_gate_remote.py smoke \
  --run-tag "${SMOKE_RUN_TAG}" \
  --resolved-revision "${RESOLVED_REVISION}"
```

Expected: architecture verification, 17-token cached decode, export/import continuation, state normalization, allocator snapshots, and independent verification complete. The smoke may classify only `SMOKE_PASS` or `INCOMPLETE`; it cannot issue canonical `GO`.

- [ ] **Step 3: Audit smoke coverage before canonical**

Run:

```bash
python3 tools/verify_qwen35_hybrid_state_gate.py \
  --run-dir "experiments/qwen35_hybrid_state/${SMOKE_RUN_TAG}" \
  --write-report
```

Expected: verifier confirms smoke source/model/environment bindings and smoke-specific domain. Any unexplained state, unsupported export/import, or runtime failure stops canonical execution as `INCOMPLETE`.

- [ ] **Step 4: Run the complete canonical matrix**

Run:

```bash
CANONICAL_RUN_TAG="qwen35-2b-hybrid-canonical-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_qwen35_hybrid_state_gate_remote.py canonical \
  --run-tag "${CANONICAL_RUN_TAG}" \
  --resolved-revision "${RESOLVED_REVISION}" \
  --smoke-run-tag "${SMOKE_RUN_TAG}"
```

Expected: every matrix row exists once, the worker exits cleanly or preserves a classified partial run, and the local independent verifier emits one authoritative `GO`, `NO_GO`, or `INCOMPLETE`.

- [ ] **Step 5: Re-run independent verification from downloaded raw artifacts**

Run:

```bash
python3 tools/verify_qwen35_hybrid_state_gate.py \
  --run-dir "experiments/qwen35_hybrid_state/${CANONICAL_RUN_TAG}" \
  --write-report
```

Expected: deterministic agreement with the first verifier result and no source/model/artifact drift.

- [ ] **Step 6: Inspect the evidence, not only the final label**

Run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

import os

run = Path("experiments/qwen35_hybrid_state") / os.environ["CANONICAL_RUN_TAG"]
verification = json.loads((run / "independent_verification.json").read_text())
required = {
    "classification",
    "reasons",
    "expected_case_count",
    "observed_case_count",
    "architecture",
    "correctness",
    "state_lifecycle",
    "logical_bytes_by_role",
    "unique_storage_bytes_by_role",
    "allocator_snapshots",
    "claim_boundary",
}
assert required <= verification.keys(), sorted(required - verification.keys())
assert verification["expected_case_count"] == verification["observed_case_count"]
print(json.dumps(verification, indent=2, sort_keys=True))
PY
```

Expected: complete coverage and independently reconstructed ledgers. Treat any absent field or count mismatch as `INCOMPLETE` even if a summary says green.

---

### Task 8: Completion Audit, Evidence Registry, and Handoff

**Files:**
- Create: `docs/qwen35_hybrid_state_evidence_registry.json`
- Modify: `AGENT_HANDOFF_STATE.md`
- Never modify: `README.md`

**Interfaces:**
- Consumes: the canonical raw run and independent verifier from Task 7.
- Produces: durable project evidence, an explicit prompt-to-artifact audit, exact claim boundaries, and the next authorized direction.

- [ ] **Step 1: Build the prompt-to-artifact completion checklist**

Append a checklist to `AGENT_HANDOFF_STATE.md` mapping every approved requirement to evidence:

```text
1. immutable model/tokenizer/source/runtime binding
2. exact 24-layer hybrid schedule
3. every persistent tensor discovered and normalized
4. fixed recurrent/conv versus growing full-KV state
5. logical and unique physical storage ledgers
6. one-shot/cached/chunked continuation equivalence
7. three-request interleaving isolation
8. completion/release/generation increment/slot reuse
9. framework-neutral export/import contract
10. CUDA allocated/reserved snapshots
11. complete canonical matrix
12. independent verifier reconstruction
13. remote safety and process/port audit
14. compression/kernel/performance claims remain out of scope
```

For each item, record the exact artifact filename and verifier field. Mark uncertainty or missing evidence as not complete.

- [ ] **Step 2: Create the closed evidence registry**

Use this exact schema:

```json
{
  "schema_version": 1,
  "gate": "qwen35_hybrid_state_compatibility",
  "spec": {
    "path": "docs/superpowers/specs/2026-07-23-qwen35-hybrid-state-compatibility-gate-design.md",
    "commit": "44b337a",
    "sha256": "generated with sha256sum"
  },
  "plan": {
    "path": "docs/superpowers/plans/2026-07-23-qwen35-hybrid-state-compatibility-gate.md",
    "commit": "resolved with git log -1 --format=%H -- docs/superpowers/plans/2026-07-23-qwen35-hybrid-state-compatibility-gate.md",
    "sha256": "generated with sha256sum"
  },
  "implementation_commit": "resolved from the committed owned source files",
  "canonical_run": {
    "path": "experiments/qwen35_hybrid_state/${CANONICAL_RUN_TAG}",
    "manifest_sha256": "generated with sha256sum",
    "independent_verification_sha256": "generated with sha256sum"
  },
  "verifier": {
    "path": "tools/verify_qwen35_hybrid_state_gate.py",
    "sha256": "generated with sha256sum"
  },
  "classification": "GO|NO_GO|INCOMPLETE",
  "claim_boundary": "Compatibility only; no native support, compression, quality, latency, throughput, or memory-reduction claim."
}
```

- [ ] **Step 3: Record the exact conclusion and next gate**

If classification is `GO`, the next permitted work is a separate **native hybrid-state integration design**; do not start it in this task. If `NO_GO`, record the exact complete-domain semantic incompatibility. If `INCOMPLETE`, record the blocker and the smallest safe recovery action without converting it to a negative architecture result.

- [ ] **Step 4: Run the final completion audit**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_contract.py
python3 tools/test_qwen35_hybrid_state_probe.py
python3 tools/test_verify_qwen35_hybrid_state_gate.py
python3 tools/test_run_qwen35_hybrid_state_gate_remote.py
python3 tools/verify_qwen35_hybrid_state_gate.py \
  --run-dir "experiments/qwen35_hybrid_state/${CANONICAL_RUN_TAG}" \
  --write-report
python3 -m json.tool docs/qwen35_hybrid_state_evidence_registry.json >/dev/null
git diff --check
git status --short
```

Expected: tests PASS; verifier deterministically reproduces the authoritative classification; registry parses; diff check is clean; only intended tracked files plus preserved unrelated pre-existing work appear.

- [ ] **Step 5: Precisely stage durable evidence**

```bash
git add \
  docs/qwen35_hybrid_state_evidence_registry.json \
  AGENT_HANDOFF_STATE.md
git diff --cached --stat
git diff --cached --check
```

Expected: no raw experiment directory and no unrelated file is staged.

- [ ] **Step 6: Commit the evidence conclusion**

```bash
git commit -m "docs: record qwen35 hybrid state compatibility result"
```

- [ ] **Step 7: Report only evidence-supported claims**

Report:

```text
classification
what correctness/lifecycle evidence passed
logical bytes by state role
unique physical storage bytes by state role
allocator observations
what the result proves
what it does not prove
next separately gated step
```

Never report a TinyLLMForge speedup, Qwen3.5 production support, `2.57x` compression, `<0.1%` quality loss, or sparse-linear-attention speedup from this gate.
