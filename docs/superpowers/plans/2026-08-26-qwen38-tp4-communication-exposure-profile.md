# Qwen3.8-27B TP4 Communication-Exposure Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository must use inline execution; do not create a worktree or dispatch subagents.

**Goal:** Establish a source-bound, text-only Qwen3.8-27B TP1/TP4 correctness authority and a strict-clean TP4 profiling campaign that measures per-layer compute, exposed NCCL time, GPU idle time, and online latency before deciding whether communication/computation overlap is justified.

**Architecture:** Reuse the Qwen3.5 hybrid runtime only through a fail-closed Qwen3.8 text-only adopter that validates checkpoint identity, nested text topology, supported tensor contracts, and the absence of multimodal input before GPU mutation. Extend the existing rank-local profiler with model-agnostic layer/operation events, parse Nsight SQLite intervals with interval-union arithmetic, and make producer and independent verifier recompute the terminal classification from immutable artifacts. Keep all baseline collectives synchronous; a verified `GO_COMMUNICATION_OVERLAP` authorizes a separate design and does not itself modify collective execution.

**Tech Stack:** Python 3, pytest, PyTorch BF16 and `torch.distributed`, CUDA Events, NVTX, NVIDIA Nsight Systems SQLite export, Hugging Face Hub/Transformers metadata, JSON/JSONL, SHA-256 manifests, SSH.

## Global Constraints

- The only authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`, which currently resolves to `/Users/bytedance/dev/TinyLLMForge`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Stay on `feat/kv-sparse-attention`; push only to `origin/feat/kv-sparse-attention`.
- Preserve unrelated dirty and untracked files. Stage only the exact files named by each task.
- Do not create a worktree or use subagents.
- Use RED, minimal implementation, GREEN, adjacent regression tests, then an exact-path commit for every implementation unit.
- Every commit uses `git -c core.hooksPath=/dev/null commit` and contains exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- The checkpoint repository is exactly `Qwen/Qwen3.8-27B` at an immutable resolved revision; `main` is not an execution identity.
- The first adopter is text-only: read topology from `text_config`, reject image/video tokens, and never construct or load a vision encoder.
- Preserve official vocabulary, embeddings, output head, layer cadence, RMSNorm, rotary, DeltaNet, and full-attention semantics.
- Keep checkpoint-specific names, shapes, and transforms outside scheduler, generic TP, interval, and profiler modules.
- Baseline execution remains synchronous. Adding async/stream fields to observations must not enable asynchronous collectives.
- Do not infer Qwen3.8-27B performance from Qwen3.5-2B evidence.
- Do not infer exposed communication by subtracting summed durations or by kernel-name guesses; use validated interval unions and trace correlation.
- At controller entry and worker entry select exactly four distinct GPU UUIDs with memory used `<= 1024 MiB`, utilization `<= 5%`, and no compute process.
- After launch, only current-attempt PIDs may use selected GPUs. Never kill, pause, signal, or adopt unrelated processes.
- All remote task data must stay below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not write remote task data under `/`, `/tmp`, `/private/tmp`, an old checkout, or an adaptive-ngram checkout.
- Do not run Kerberos initialization commands.
- Frozen workloads are P0 `256/128/c1`, P1 `2048/128/c1`, Q0 `256/128/c4`, Q1 `256/128/c8`, and Q2 `2048/128/c4`, each with two warmups and five measured repetitions.
- All performance rows use BF16, TP4, greedy decoding, temperature zero, fixed output length, and identical scheduler/CUDA Graph policy.
- Only a terminal independently verified `GO_COMMUNICATION_OVERLAP` may create the later overlap implementation design.

## File and Responsibility Map

| File | Responsibility |
| --- | --- |
| `tinyvllm/models/qwen38_text_adopter.py` | Validate Qwen3.8 identity/topology and expose the existing Qwen3.5 text runtime configuration without multimodal leakage. |
| `tinyvllm/engine/model_runner.py` | Invoke the adopter before model construction/loading and reject multimodal request metadata. |
| `tinyvllm/models/qwen35_checkpoint.py` | Accept only validated Qwen3.8 text tensor inventory through existing language-model weight planning. |
| `tools/qwen38_model_manifest.py` | Build and verify immutable source/model/tokenizer/config inventory. |
| `tools/qwen38_tp_correctness.py` | Validate TP1 official reference, TinyLLMForge TP1, and real TP4 token/logit/rank cleanup evidence. |
| `tinyvllm/engine/decode_internal_profiler.py` | Record generic layer/operation/collective CUDA-event metadata without synchronizing the hot path. |
| `tinyvllm/layers/linear.py` and Qwen3.5 layer stack files | Attach generic layer role/index and operation identity to existing synchronous operations. |
| `tools/qwen38_nsys_intervals.py` | Read Nsight SQLite exports, correlate NVTX/NCCL/kernel rows, and compute interval unions/subtractions. |
| `tools/qwen38_communication_exposure.py` | Validate structured rows, aggregate five repetitions, calculate headroom, and classify the gate. |
| `tools/run_qwen38_tp4_communication_profile.py` | Build immutable attempts, select strict-clean GPUs, run correctness/workloads/Nsight, download artifacts, and clean only owned children. |
| `tools/verify_qwen38_tp4_communication_profile.py` | Independently recompute inventories, hashes, interval metrics, correctness, overhead, and terminal classification. |
| `tools/test_qwen38_*.py` | CPU unit/contract tests for every new boundary. |
| `docs/superpowers/audits/2026-08-26-qwen38-tp4-communication-exposure-audit.md` | Final prompt-to-artifact audit and claim boundary. |
| `AGENT_HANDOFF_STATE.md` | Exact source, run tag, classification, artifacts, and next authorized action. |

---

## Phase A: Qwen3.8-27B Text-Only and Correctness Authority

### Task 1: Fail-Closed Qwen3.8 Text Adopter

**Files:**
- Create: `tinyvllm/models/qwen38_text_adopter.py`
- Create: `tools/test_qwen38_text_adopter.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Test: `tools/test_qwen35_model_runner_native_entry.py`
- Test: `tools/test_qwen35_concrete_component_factory.py`

**Interfaces:**
- Produces: `Qwen38TextRuntimeProfile` frozen dataclass.
- Produces: `adopt_qwen38_text_config(hf_config) -> Qwen38TextRuntimeProfile`.
- Produces: `reject_qwen38_multimodal_inputs(*, input_ids, multimodal_inputs=None) -> None`.
- Consumes: top-level Hugging Face config and nested `text_config`.
- Guarantees: accepted configurations may enter `_load_qwen35_model_runner_model`; rejected configurations fail before model construction or CUDA allocation.

- [ ] **Step 1: Write RED identity and topology tests**

Create a synthetic official-shaped config and assert:

```python
profile = adopt_qwen38_text_config(_official_config())
assert profile.repository == "Qwen/Qwen3.8-27B"
assert profile.architecture == "Qwen3_5ForConditionalGeneration"
assert profile.text_model_type == "qwen3_5_text"
assert profile.num_hidden_layers == 64
assert profile.hidden_size == 5120
assert profile.intermediate_size == 17408
assert profile.layer_types[3] == "full_attention"
assert profile.layer_types[0] == "linear_attention"
assert profile.dtype == "bfloat16"
assert profile.language_model_only is False
```

Add parameterized failures for wrong repository, floating revision, wrong architecture, absent `text_config`, wrong text model type, layer-count mismatch, cadence mismatch, unsupported dtype, unsupported tied-head semantics, and missing required dimensions.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_qwen38_text_adopter.py
```

Expected: collection fails because `tinyvllm.models.qwen38_text_adopter` does not exist.

- [ ] **Step 3: Implement the immutable adopter profile**

Implement:

```python
@dataclass(frozen=True)
class Qwen38TextRuntimeProfile:
    repository: str
    revision: str
    architecture: str
    text_model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    layer_types: tuple[str, ...]
    dtype: str
    vocab_size: int
    language_model_only: bool


def adopt_qwen38_text_config(hf_config):
    text = getattr(hf_config, "text_config", None)
    if text is None:
        raise ValueError("Qwen3.8 requires nested text_config")
    profile = Qwen38TextRuntimeProfile(
        repository=_required(hf_config, "_name_or_path"),
        revision=_immutable_revision(hf_config),
        architecture=_single_architecture(hf_config),
        text_model_type=_required(text, "model_type"),
        num_hidden_layers=_positive_int(text, "num_hidden_layers"),
        hidden_size=_positive_int(text, "hidden_size"),
        intermediate_size=_positive_int(text, "intermediate_size"),
        layer_types=tuple(_required(text, "layer_types")),
        dtype=_normalized_dtype(text),
        vocab_size=_positive_int(text, "vocab_size"),
        language_model_only=bool(
            getattr(hf_config, "language_model_only", False)
        ),
    )
    _validate_qwen38_profile(profile)
    return profile
```

Validation must compare exact identity fields and derive the expected 64-layer cadence rather than accepting any `qwen3_5` model.

- [ ] **Step 4: Add RED multimodal rejection tests**

Assert rejection when `multimodal_inputs` is non-empty or when token IDs contain any configured image/video special token. Assert ordinary text token IDs pass and the function does not import or instantiate any visual class.

- [ ] **Step 5: Implement text-only request rejection**

Implement a pure preflight that converts token IDs to Python integers without calling `.item()` in a measured path, intersects them with configured multimodal token IDs, and raises:

```text
Qwen3.8 first adopter is text-only; image/video tokens are unsupported
```

- [ ] **Step 6: Wire the adopter into model initialization**

In `_initialize_model_runner_model`, distinguish the official Qwen3.8 architecture from other `qwen3_5` checkpoints:

```python
if is_qwen38_checkpoint(config.hf_config):
    profile = adopt_qwen38_text_config(config.hf_config)
    model, owner = load_qwen35_model(config, rank)
    owner.bind_qwen38_text_profile(profile)
    return model, owner
```

Do not add Qwen3.8 branches to scheduler code or duplicate the Qwen3.5 model implementation.

- [ ] **Step 7: Run focused and adjacent tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_qwen38_text_adopter.py \
  tools/test_qwen35_model_runner_native_entry.py \
  tools/test_qwen35_concrete_component_factory.py
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit exact files**

```bash
git add \
  tinyvllm/models/qwen38_text_adopter.py \
  tinyvllm/engine/model_runner.py \
  tools/test_qwen38_text_adopter.py \
  tools/test_qwen35_model_runner_native_entry.py
git -c core.hooksPath=/dev/null commit -m "feat(runtime): adopt Qwen3.8 text model

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Before committing, use `git diff --cached --name-only` and reject any path not listed above.

### Task 2: Immutable Model Manifest and Checkpoint Contract

**Files:**
- Create: `tools/qwen38_model_manifest.py`
- Create: `tools/test_qwen38_model_manifest.py`
- Modify: `tinyvllm/models/qwen35_checkpoint.py`
- Modify: `tools/test_qwen35_checkpoint_weight_name_contract.py`

**Interfaces:**
- Produces: `build_model_manifest(model_root: Path, *, repository: str, revision: str) -> dict`.
- Produces: `verify_model_manifest(path: Path) -> dict`.
- Produces: `validate_qwen38_checkpoint_inventory(hf_config, weight_index) -> dict`.
- Consumes: local immutable checkpoint snapshot and existing Qwen3.5 language tensor planner.

- [ ] **Step 1: Write RED manifest tests**

Build a temporary checkpoint tree containing `config.json`, tokenizer files, index JSON, and shard fixtures. Assert every regular file has relative path, byte size, and SHA-256; configs and tokenizer inventory have dedicated hashes; repository/revision are exact; sorted canonical JSON produces a stable manifest digest.

Add rejection tests for symlinks escaping the model root, duplicate paths, missing tokenizer, missing shard, extra unlisted shard, floating revision, changed byte size/hash, and absolute paths.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_qwen38_model_manifest.py
```

Expected: import failure for the missing module.

- [ ] **Step 3: Implement canonical manifest construction and verification**

Use:

```python
MODEL_SCHEMA = "tinyllmforge.qwen38-model-manifest.v1"
REPOSITORY = "Qwen/Qwen3.8-27B"

def canonical_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )
```

Hash files in binary streaming chunks, require the resolved revision to match `^[0-9a-f]{40}$`, and re-read every file during verification.

- [ ] **Step 4: Write RED checkpoint inventory tests**

Extend the checkpoint contract fixture to represent 64 layers, the official cadence, and `model.language_model.*` names. Assert all expected text tensors are planned, all vision tensors are explicitly skipped, no text tensor is silently skipped, and shape mismatch names the source tensor and expected/observed shape.

- [ ] **Step 5: Reuse the existing language planner behind validated input**

Add a `qwen38_text_profile` optional argument to the Qwen3.5 checkpoint planner. When present, verify it agrees with `text_config`, allow explicit `model.visual.*` skips plus the exact adapter-declared 15-tensor `mtp.*` auxiliary inventory that the official base `Qwen3_5ForConditionalGeneration.forward` does not consume, and preserve the existing TP shard rules. Missing, extra, duplicated, or undeclared MTP tensors fail closed. This base-decode exclusion does not claim Qwen3.8 speculative-MTP support. No generic checkpoint module may contain the repository string.

- [ ] **Step 6: Run focused and adjacent tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_qwen38_model_manifest.py \
  tools/test_qwen35_checkpoint_weight_name_contract.py \
  tools/test_qwen35_checkpoint_metadata.py \
  tools/test_qwen35_checkpoint_assignment.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit exact files**

```bash
git add \
  tools/qwen38_model_manifest.py \
  tools/test_qwen38_model_manifest.py \
  tinyvllm/models/qwen35_checkpoint.py \
  tools/test_qwen35_checkpoint_weight_name_contract.py
git -c core.hooksPath=/dev/null commit -m "feat(runtime): bind Qwen3.8 checkpoint identity

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 3: TP1 Reference and TP4 Correctness Contract

**Files:**
- Create: `tools/qwen38_tp_correctness.py`
- Create: `tools/test_qwen38_tp_correctness.py`
- Create: `tools/run_qwen38_tp_correctness.py`
- Create: `tools/test_run_qwen38_tp_correctness.py`

**Interfaces:**
- Produces: `validate_correctness_bundle(root: Path) -> dict`.
- Produces: `compare_decode_rows(reference: dict, tp1: dict, tp4: dict) -> dict`.
- Produces: CLI that runs official TP1, TinyLLMForge TP1, and TinyLLMForge TP4 under one model/workload/source identity.
- Consumes: `model_manifest.json`, prompt token IDs, per-position top-k logits, generated IDs, rank load receipts, and cleanup receipts.

- [ ] **Step 1: Write RED semantic-verifier tests**

Create three fixture runs and assert hard requirements:

```python
result = validate_correctness_bundle(bundle)
assert result["classification"] == "PASS"
assert result["exact_prompt_tokens"] is True
assert result["exact_generated_tokens"] is True
assert result["exact_argmax_positions"] is True
assert result["finite_logits_all_ranks"] is True
assert result["rank_inventory"] == [0, 1, 2, 3]
assert result["distinct_expected_shards"] is True
assert result["process_groups_destroyed"] is True
assert result["owned_children_remaining"] == []
```

Reject one fixture per violated invariant, including a numerically close but different argmax.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_qwen38_tp_correctness.py
```

Expected: import failure for the missing verifier.

- [ ] **Step 3: Implement the pure correctness verifier**

Record max absolute and relative logit error, top-k identity, and configured numeric tolerance, but classify token or argmax mismatch as failure regardless of numeric tolerance. Bind every row to source-tree SHA, model-manifest SHA, prompt SHA, mode, dtype, and TP size.

- [ ] **Step 4: Write RED runner-plan tests**

Assert the runner:

- emits official TP1, TinyLLMForge TP1, and four-rank TinyLLMForge TP4 commands;
- uses text-only fixed token IDs and greedy fixed-length decode;
- enables no profiler;
- writes only below an attempt path supplied beneath the approved remote root;
- uses fresh rendezvous ports and process groups;
- records worker PIDs/PGIDs and destroys process groups;
- never emits `kill`, `pkill`, `killall`, Kerberos commands, `/tmp`, or adaptive-ngram paths.

- [ ] **Step 5: Implement the correctness runner**

Build commands as argv arrays, not interpolated shell strings. Use attempt-scoped environment variables and paths. Return a receipt even on failure; the receipt must distinguish model load, reference, TP1, TP4, timeout, cleanup, and verification stages.

- [ ] **Step 6: Verify focused and adjacent GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_qwen38_tp_correctness.py \
  tools/test_run_qwen38_tp_correctness.py \
  tools/test_qwen35_tp1_real_root_logit_correctness_contract.py \
  tools/test_qwen35_tp4_engine_correctness_contract.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit exact files**

```bash
git add \
  tools/qwen38_tp_correctness.py \
  tools/test_qwen38_tp_correctness.py \
  tools/run_qwen38_tp_correctness.py \
  tools/test_run_qwen38_tp_correctness.py
git -c core.hooksPath=/dev/null commit -m "feat(runtime): add Qwen3.8 TP correctness gate

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

## Phase B: Generic Communication-Exposure Profiler and Gate

### Task 4: Generic Layer and Operation Event Schema

**Files:**
- Modify: `tinyvllm/engine/decode_internal_profiler.py`
- Modify: `tinyvllm/layers/linear.py`
- Modify: `tinyvllm/layers/qwen35_packed_layer_stack.py`
- Modify: `tinyvllm/layers/qwen35_packed_full_decoder_layer.py`
- Modify: `tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py`
- Create: `tools/test_communication_exposure_event_schema.py`
- Modify: `tools/test_decode_internal_profiler.py`
- Modify: `tools/test_decode_internal_profile_wiring.py`

**Interfaces:**
- Produces: `profiler.layer(layer_index: int, layer_role: str)`.
- Produces: `profiler.operation(operation_class: str, operation_name: str, *, tensor=None)`.
- Extends: `profile_collective(..., collective_kind, process_group, async_mode, source_stream, completion_stream)`.
- Emits keys `(attempt, workload, repetition, request_set_sha256, decode_ordinal, rank, layer_index, layer_role, operation_ordinal)`.

- [ ] **Step 1: Write RED schema and lifecycle tests**

With fake clock/events/streams, assert nested layer and operation contexts emit CPU enqueue bounds plus unresolved CUDA event pairs, monotonically increasing operation ordinals, and the exact generic role/class enums. Assert finalization synchronizes once and resolves all events.

Reject missing active step, nested mismatched layer exits, invalid role/class, duplicate identity, finalization with open scopes, per-operation synchronization, and events after finalization.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_communication_exposure_event_schema.py \
  tools/test_decode_internal_profiler.py
```

Expected: new schema tests fail because layer/operation scopes are absent.

- [ ] **Step 3: Implement generic scopes without changing execution**

Add frozen enums:

```python
LAYER_ROLES = {
    "linear_attention", "full_attention", "mlp", "normalization",
    "residual", "embedding", "output_head",
}
OPERATION_CLASSES = {
    "gemm", "attention", "recurrent", "collective",
    "memory", "other_compute",
}
```

Record stream identity using injected/testable stream resolvers. Preserve existing synchronous calls and only observe `async_mode=False` in this phase.

- [ ] **Step 4: Instrument layer and collective call sites**

Wrap existing layer phases with generic labels and pass metadata to the existing collective wrapper. Do not add Qwen3.8 names to these files. Ensure an unprofiled run follows the same call path except for no-op contexts.

- [ ] **Step 5: Verify focused and adjacent GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_communication_exposure_event_schema.py \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py \
  tools/test_qwen35_packed_layer_stack.py \
  tools/test_qwen35_packed_full_decoder_layer.py \
  tools/test_qwen35_packed_stateful_linear_decoder_layer.py \
  tools/test_replicated_weight_row_parallel_linear.py
```

Expected: all selected tests pass and existing synchronous collective assertions remain unchanged.

- [ ] **Step 6: Commit exact files**

```bash
git add \
  tinyvllm/engine/decode_internal_profiler.py \
  tinyvllm/layers/linear.py \
  tinyvllm/layers/qwen35_packed_layer_stack.py \
  tinyvllm/layers/qwen35_packed_full_decoder_layer.py \
  tinyvllm/layers/qwen35_packed_stateful_decoder_layer.py \
  tools/test_communication_exposure_event_schema.py \
  tools/test_decode_internal_profiler.py \
  tools/test_decode_internal_profile_wiring.py
git -c core.hooksPath=/dev/null commit -m "feat(runtime): trace generic TP layer operations

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 5: Nsight Interval Correlation and Union Arithmetic

**Files:**
- Create: `tools/qwen38_nsys_intervals.py`
- Create: `tools/test_qwen38_nsys_intervals.py`

**Interfaces:**
- Produces: `Interval(start_ns: int, end_ns: int)`.
- Produces: `union_duration(intervals: Iterable[Interval]) -> int`.
- Produces: `subtract_intervals(base, covered) -> tuple[Interval, ...]`.
- Produces: `parse_nsys_sqlite(path: Path, structured_rows: list[dict]) -> dict`.
- Produces: per-step/per-rank/per-layer GEMM, collective, compute, exposed collective, idle, bytes, and critical-path rows.
- Computes: `gpu_idle = step_critical_interval - union(all required GPU work)`.

- [ ] **Step 1: Write RED interval arithmetic tests**

Cover disjoint, adjacent, nested, partially overlapping, identical, and zero-length intervals. Assert:

```python
exposed = subtract_intervals(collective_union, compute_union)
idle = subtract_intervals((critical_interval,), required_work_union)
```

never double-counts concurrent kernels and rejects negative/non-monotonic intervals.

- [ ] **Step 2: Verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_qwen38_nsys_intervals.py -k interval
```

Expected: import failure for the missing module.

- [ ] **Step 3: Implement pure interval operations**

Sort by `(start_ns, end_ns)`, merge overlap and adjacency, and perform subtraction with a two-pointer scan. Keep arithmetic integer nanoseconds throughout.

- [ ] **Step 4: Write RED synthetic SQLite correlation tests**

Create a minimal SQLite fixture with:

- four ranks and aligned NVTX step/layer/operation ranges;
- GEMM and required-compute kernels;
- NCCL kernels correlated to structured collective ordinal/kind/bytes;
- overlapping compute/NCCL intervals;
- explicit stream and CPU-thread identity.

Assert the critical rank is the rank whose final required event ends last, and assert exact union, exposed, overlap, idle, and critical-path durations. Add failures for missing NCCL correlation, missing rank, duplicate operation mapping, range mismatch, cross-step kernel, and unsupported schema.

- [ ] **Step 5: Implement schema discovery and fail-closed correlation**

Read only known Nsight tables/columns discovered from `sqlite_master` and `PRAGMA table_info`. Use NVTX payload identity rather than kernel-name-only matching. Return `INCONCLUSIVE_TRACE_COVERAGE` when required NCCL kernels cannot be correlated.

- [ ] **Step 6: Verify GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_qwen38_nsys_intervals.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit exact files**

```bash
git add \
  tools/qwen38_nsys_intervals.py \
  tools/test_qwen38_nsys_intervals.py
git -c core.hooksPath=/dev/null commit -m "feat(profiling): compute exposed NCCL intervals

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 6: Five-Workload Aggregator and Terminal Classifier

**Files:**
- Create: `tools/qwen38_communication_exposure.py`
- Create: `tools/test_qwen38_communication_exposure.py`

**Interfaces:**
- Produces: `validate_profile_rows(rows: list[dict]) -> dict`.
- Produces: `aggregate_profile_bundle(root: Path) -> dict`.
- Produces: `classify_communication_exposure(summary: dict) -> str`.
- Produces: `select_representative_repetition(rows: list[dict]) -> int`.

- [ ] **Step 1: Write RED workload and alignment tests**

Require exactly P0/P1/Q0/Q1/Q2, two warmups plus repetitions `0..4`, four distinct ranks per measured repetition, aligned request digest/decode ordinal/layer/operation inventory, correct prompt/output/concurrency, and no reuse after failed finalization.

- [ ] **Step 2: Write RED metric tests**

Assert:

```python
ratio = exposed_collective_ns / step_critical_interval_ns
headroom = min(exposed_collective_ns, independent_compute_ns) / (
    step_critical_interval_ns
)
```

and verify per-layer plus end-to-end aggregates for QPS, output tokens/s, TTFT/TPOT/E2E P50/P95/P99, peak allocated/reserved memory per rank, utilization, power, tokens, and argmax.

- [ ] **Step 3: Write RED classification precedence tests**

Cover every terminal classification in exact precedence:

```text
INVALID_CORRECTNESS
INVALID_RESOURCE_IDENTITY
INCONCLUSIVE_TRACE_COVERAGE
INCONCLUSIVE_VARIANCE
GO_COMMUNICATION_OVERLAP
NO_GO_ALREADY_HIDDEN
INCONCLUSIVE_LOW_HEADROOM
```

For `GO`, require one causal and one online workload at median exposed ratio `>= 0.10`, qualifying headroom `>= 0.05`, four-of-five direction agreement, complete alignment, and overhead `<= 0.03`. For `NO_GO`, require every workload ratio `< 0.05` and headroom `< 0.02`.

- [ ] **Step 4: Verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_qwen38_communication_exposure.py
```

Expected: import failure for the missing module.

- [ ] **Step 5: Implement validation, aggregation, and classifier**

Use deterministic nearest-to-median selection after all five structured repetitions complete. Paired profiler overhead compares matched profiled/unprofiled controls with identical source/model/workload/rank/GPU identity.

- [ ] **Step 6: Verify GREEN**

Run Step 4. Expected: all tests pass.

- [ ] **Step 7: Commit exact files**

```bash
git add \
  tools/qwen38_communication_exposure.py \
  tools/test_qwen38_communication_exposure.py
git -c core.hooksPath=/dev/null commit -m "feat(profiling): classify TP4 communication exposure

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 7: Strict-Clean Remote Campaign Controller

**Files:**
- Create: `tools/run_qwen38_tp4_communication_profile.py`
- Create: `tools/test_run_qwen38_tp4_communication_profile.py`

**Interfaces:**
- Produces: `select_strict_clean_gpus(inventory: list[dict]) -> tuple[dict, ...]`.
- Produces: `build_workload_cases() -> tuple[ProfileCase, ...]`.
- Produces: `build_attempt_plan(...) -> dict`.
- Produces: `run_attempt(...) -> dict`.
- Produces: CLI with `--ssh-target`, `--remote-root`, `--model-root`, `--attempt-tag`, and bounded timeout/retry options.

- [ ] **Step 1: Write RED strict-clean selection tests**

Assert selection requires four unique UUIDs, each with used memory at most `1024 MiB`, utilization at most `5`, and empty compute process list. Test boundary equality, insufficient GPUs, duplicate UUID, malformed telemetry, and unrelated process appearance after launch.

- [ ] **Step 2: Write RED workload-plan tests**

Assert 10 warmups, 25 unprofiled measured structured cases, and 25 measured
Nsight replays. Select one representative replay per workload only after the
unprofiled runs complete, but retain all five per-workload Nsight replays for
the four-of-five exposure-direction gate. Require 25 paired
profiled/unprofiled overhead controls. Freeze P0/P1/Q0/Q1/Q2 exactly and
preserve deterministic measured order.

- [ ] **Step 3: Write RED remote safety tests**

Assert all generated paths resolve below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818
```

Reject `/`, `/tmp`, `/private/tmp`, old checkout, adaptive-ngram, traversal, symlink escape, reused attempt tag, shell interpolation, Kerberos initialization, and commands that signal unknown PIDs.

- [ ] **Step 4: Verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_run_qwen38_tp4_communication_profile.py
```

Expected: import failure for the missing controller.

- [ ] **Step 5: Implement plan-only and dry-run paths**

The controller must first produce source/model/environment/workload/topology manifests and a command plan without launching GPU work. Use a fresh attempt directory, temporary files inside that directory, and atomic rename.

- [ ] **Step 6: Implement guarded execution and owned cleanup**

At controller and worker entry, sample inventory and bind UUID-to-rank mapping. During execution, sample resource rows and compare observed PIDs to the owned PID set. On failure, stop launching new children, wait/reap owned children, destroy owned process groups through their normal control path, preserve failure evidence, and never signal unrelated PIDs.

- [ ] **Step 7: Implement structured and Nsight phases**

Run correctness first. Only after it passes, run two warmups and five
unprofiled measured repetitions per workload. Select the nearest-median
repetition for each workload as its report representative, then replay all five
measured repetitions per workload under Nsight with CUDA, NVTX, OS runtime,
and context-switch traces when supported. Export all 25 SQLite databases below
`attempt/nsys/`; mark five as representative without dropping or duplicating
their contribution to the four-of-five gate.

- [ ] **Step 8: Verify focused and adjacent GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_run_qwen38_tp4_communication_profile.py \
  tools/test_qwen35_tp4_correctness_resource_policy.py \
  tools/test_run_qwen35_tp4_decode_internal_profile.py \
  tools/test_qwen35_tp4_controlmaster_transport.py
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit exact files**

```bash
git add \
  tools/run_qwen38_tp4_communication_profile.py \
  tools/test_run_qwen38_tp4_communication_profile.py
git -c core.hooksPath=/dev/null commit -m "feat(profiling): orchestrate strict Qwen3.8 TP4 gate

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 8: Independent Artifact Verifier

**Files:**
- Create: `tools/verify_qwen38_tp4_communication_profile.py`
- Create: `tools/test_verify_qwen38_tp4_communication_profile.py`

**Interfaces:**
- Produces: `verify_bundle(root: Path) -> dict`.
- Produces: terminal `independent_verification.json`.
- Consumes every required artifact and recomputes all semantic gates without trusting producer summaries.

- [ ] **Step 1: Write RED complete-bundle test**

Create a synthetic bundle containing:

```text
source_manifest.json
model_manifest.json
environment.json
gpu_topology.json
workload_manifest.json
correctness_rows.jsonl
profile_rows.jsonl
layer_summary.json
communication_exposure_summary.json
online_metrics.json
memory_summary.json
resource_samples.jsonl
nsys/
manifest.sha256
report.md
```

Assert the verifier recomputes hashes, row inventory, rank alignment, interval arithmetic, ratios, overhead, correctness, cleanup, and terminal classification before writing `independent_verification.json`.

- [ ] **Step 2: Write RED tamper and semantic-gap tests**

Reject missing/extra artifact, hash mismatch, summary changed without raw-row change, producer/verifier classification mismatch, wrong revision, workload drift, rank drift, GPU UUID drift, incomplete Nsight correlation, invalid correctness, overhead above 3%, and report text inconsistent with the machine result.

- [ ] **Step 3: Verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q tools/test_verify_qwen38_tp4_communication_profile.py
```

Expected: import failure for the missing verifier.

- [ ] **Step 4: Implement independent verification**

Do not import producer summary-writing functions. It may share only low-level canonical JSON and interval primitives whose outputs are independently asserted by verifier tests. Write the verification result atomically and then regenerate `manifest.sha256` including the verification file.

- [ ] **Step 5: Verify GREEN and run the complete local suite**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_qwen38_text_adopter.py \
  tools/test_qwen38_model_manifest.py \
  tools/test_qwen38_tp_correctness.py \
  tools/test_run_qwen38_tp_correctness.py \
  tools/test_communication_exposure_event_schema.py \
  tools/test_qwen38_nsys_intervals.py \
  tools/test_qwen38_communication_exposure.py \
  tools/test_run_qwen38_tp4_communication_profile.py \
  tools/test_verify_qwen38_tp4_communication_profile.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit exact files**

```bash
git add \
  tools/verify_qwen38_tp4_communication_profile.py \
  tools/test_verify_qwen38_tp4_communication_profile.py
git -c core.hooksPath=/dev/null commit -m "feat(profiling): independently verify TP4 gate

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

---

## Phase C: Real Four-GPU Evidence and Terminal Audit

### Task 9: Local Preflight, SSH Recovery, and Dry Run

**Files:**
- Runtime output only under a fresh local `artifacts/qwen38_tp4_communication_profile/<attempt-tag>/`.
- Remote output only under the approved mounted root.

**Interfaces:**
- Consumes committed source and a local/remote immutable model manifest.
- Produces plan, preflight, SSH, storage, and strict-clean admission receipts.

- [x] **Step 1: Verify branch/source identity and push implementation**

Run:

```bash
git status --short --branch
git log -1 --format='%H%n%B'
git push origin feat/kv-sparse-attention
```

Expected: branch tracks the same remote commit; intended implementation commits each have one required trailer.

Result (2026-08-26): local `HEAD`, the tracking branch, and GitHub all
resolved to `116625a225d574c5561382df3d6e50f47eac27fa`. The commit contains
exactly one required trailer.

- [x] **Step 2: Rebuild or reuse SSH ControlMaster without Kerberos initialization**

Use the existing authenticated SSH configuration and a socket stored beneath the approved local workspace/session area. Verify with a read-only command that prints remote hostname, current user, mount information for `/data00/home/sitian`, and no task output elsewhere.

Expected: successful read-only connection. If authentication closes the connection, preserve the receipt and continue local verification; do not classify GPU/model state from that error.

Result (2026-08-26): reused the already-authenticated
`/tmp/ssh-sitian-10.232.195.203` master because no Kerberos credential was
available to establish a replacement and a prior workspace-derived socket
name exceeded the Unix-domain path limit. The read-only probe reached
`n232-195-203` as `sitian`, recorded both `/data00/home/sitian` mount layers,
and performed no write probe. The legacy socket location is a documented
local-only variance; no task data was written there.

- [x] **Step 3: Run controller dry-run**

Run the new controller with `--dry-run` and a fresh tag. Inspect every emitted local and remote path and every argv element.

Expected: no GPU worker starts; all remote paths are below the approved root; source/model/workload identities are immutable.

Result (2026-08-26): `--plan-only` passed the complete path/argv audit for
the immutable source and declared model revisions. The real `--dry-run`
then exited `2` with `BLOCKED_KERBEROS_TTL` before any remote write or worker
launch. This is the required fail-fast result; it is not `DRY_RUN_READY`.

- [x] **Step 4: Run strict-clean admission**

Query current NVML inventory and select four qualifying GPUs by current state rather than historical indices.

Expected: exactly four unique UUIDs satisfy all three thresholds. If fewer are available, start the local controller's bounded monitor so the same committed attempt launches immediately when the clean window appears; do not reserve GPUs or affect external work.

Result (2026-08-26): current NVML admission selected GPU indices
`0,1,2,3`; all four had `0 MiB`, `0%` utilization, and no compute process.
No monitor was needed, and no GPU was reserved or modified.

### Task 10: Qwen3.8 Correctness Campaign

**Files:**
- Runtime artifacts in the fresh attempt.

- [ ] **Step 1: Acquire and verify immutable model identity**

Verify every checkpoint/config/tokenizer file against `model_manifest.json`. Record Transformers, PyTorch, CUDA, driver, NCCL, GPU UUID, topology, and source-tree identities.

- [ ] **Step 2: Execute official TP1 reference**

Run fixed text-only prompts, greedy decoding, and fixed output length. Persist prompt tokens, generated tokens, per-position argmax/top-k logits, finite checks, and numeric logit rows.

- [ ] **Step 3: Execute TinyLLMForge TP1**

Use the same prompt tokens, decode length, dtype, and checkpoint revision. Verify exact generated-token and argmax parity against the official reference.

- [ ] **Step 4: Execute TinyLLMForge TP4**

Bind ranks to admitted UUIDs, record distinct expected shard receipts, run the same prompts, and verify exact generated-token and argmax parity. Confirm all four ranks exit, process groups are destroyed, and no owned child remains.

- [ ] **Step 5: Run local independent correctness verification**

Download the correctness artifacts and run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
python tools/qwen38_tp_correctness.py \
  --bundle artifacts/qwen38_tp4_communication_profile/<attempt-tag>/correctness
```

Expected: `classification=PASS`. Any other result stops the performance campaign.

### Task 11: Structured Profiles, Nsight Replays, and Terminal Bundle

**Files:**
- Runtime artifacts matching the Artifact Contract.

- [ ] **Step 1: Execute structured workload matrix**

Run P0/P1/Q0/Q1/Q2 with two warmups and five measured repetitions. Record all four ranks, event finalization, online request timing, memory, utilization, power, and resource identity.

- [ ] **Step 2: Run paired profiler-overhead controls**

For each workload, run matched profiled/unprofiled controls on the same admitted GPU UUIDs and compute paired overhead.

Expected: overhead `<= 3%` for a `GO`; higher overhead prevents promotion.

- [ ] **Step 3: Capture all measured Nsight replays and mark representatives**

After all structured rows complete, choose the repetition nearest median decode
time for each workload as its representative. Replay all five measured
repetitions for every workload under Nsight, export 25 SQLite databases, and
correlate every required collective with its structured operation identity.
Use all five exposure observations in the direction gate; use the representative
label only to select the detailed trace shown in the report.

- [ ] **Step 4: Produce all summaries**

Generate `layer_summary.json`, `communication_exposure_summary.json`, `online_metrics.json`, and `memory_summary.json` from raw rows and validated Nsight intervals.

- [ ] **Step 5: Run independent verification**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
python tools/verify_qwen38_tp4_communication_profile.py \
  --bundle artifacts/qwen38_tp4_communication_profile/<attempt-tag>
```

Expected: producer and verifier agree on one terminal classification and all artifact hashes validate.

### Task 12: Audit, Handoff, Commit, and Conditional Next Design

**Files:**
- Create: `docs/superpowers/audits/2026-08-26-qwen38-tp4-communication-exposure-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Write the prompt-to-artifact audit**

Map every user/spec requirement to exact artifact path, hash, command, and verdict. Include:

- official immutable Qwen3.8 identity;
- strict-clean four-GPU entry/worker evidence;
- TP1 reference, TinyLLMForge TP1, and TP4 correctness;
- per-layer GEMM/NCCL/exposed/idle/critical-path evidence;
- P0/P1/Q0/Q1/Q2 inventory and five repetitions;
- QPS, output tokens/s, TTFT, TPOT, E2E, memory, utilization, and power;
- profiler overhead;
- producer/verifier agreement;
- benefit and cost numbers;
- exact terminal classification and claim boundary.

- [ ] **Step 2: Update handoff state**

Append the source commit, attempt tag, remote approved root, model revision, GPU UUID/rank map, artifact root/hash, exact test results, terminal classification, and immediate next authorized action.

- [ ] **Step 3: Run final verification**

Run:

```bash
git diff --check
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
pytest -q \
  tools/test_qwen38_text_adopter.py \
  tools/test_qwen38_model_manifest.py \
  tools/test_qwen38_tp_correctness.py \
  tools/test_run_qwen38_tp_correctness.py \
  tools/test_communication_exposure_event_schema.py \
  tools/test_qwen38_nsys_intervals.py \
  tools/test_qwen38_communication_exposure.py \
  tools/test_run_qwen38_tp4_communication_profile.py \
  tools/test_verify_qwen38_tp4_communication_profile.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-qwen38-pycache \
python tools/verify_qwen38_tp4_communication_profile.py \
  --bundle artifacts/qwen38_tp4_communication_profile/<attempt-tag>
```

Expected: diff check passes, all focused tests pass, and independent verification returns the same terminal classification recorded in the audit.

- [ ] **Step 4: Commit and push exact documentation**

```bash
git add \
  docs/superpowers/audits/2026-08-26-qwen38-tp4-communication-exposure-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit -m "docs(profiling): audit Qwen3.8 TP4 exposure

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

- [ ] **Step 5: Enforce the conditional continuation**

If and only if both producer and independent verifier return `GO_COMMUNICATION_OVERLAP`, create:

```text
docs/superpowers/specs/2026-08-26-qwen38-tp4-communication-overlap-design.md
```

That separate design must choose one ownership transformation and specify chunk ownership/order, producer/consumer streams, CUDA Event wait edges, buffer lifetime, residual/normalization ownership, failure cleanup, deterministic synchronous fallback, CUDA Graph compatibility, correctness tolerances, and before/after performance gates.

For `NO_GO_ALREADY_HIDDEN` or any `INCONCLUSIVE_*`/`INVALID_*` result, do not implement async collectives, chunked ReduceScatter/AllGather, dual-stream execution, or event dependencies.

## Plan Completion Check

This implementation plan is complete when:

1. Phase A independently establishes the official-reference TP1, TinyLLMForge TP1, and real four-rank TP4 correctness authority for the immutable Qwen3.8 checkpoint.
2. Phase B supplies model-agnostic event instrumentation, interval-union Nsight analysis, terminal classification, a strict-clean controller, and an independent verifier.
3. Phase C produces every artifact named by the design, validates it locally, records benefit and cost, and commits/pushes the audit and handoff.
4. No communication-overlap implementation begins unless the terminal verified result is exactly `GO_COMMUNICATION_OVERLAP`.
