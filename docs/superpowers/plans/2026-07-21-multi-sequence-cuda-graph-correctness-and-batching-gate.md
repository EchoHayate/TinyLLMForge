# Multi-Sequence CUDA Graph Correctness and Production Batching Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose historical multi-sequence CUDA Graph corruption, independently prove whether exact-size replay is correct, and only then evaluate a default-off exact-bucket production path under a source-bound batching gate.

**Architecture:** A dependency-light contract module owns the frozen diagnostic matrix, tensor-shard metadata, classifications, and production thresholds. A GPU diagnostic process constructs identical prompt/KV state for eager, exact-size, and strictly-larger rounded graphs; every mode runs in a fresh process, follows eager teacher-forced tokens, and writes full logits/layer/KV evidence. A separate independent verifier reconstructs every comparison from raw artifacts. Only an independently verified `EXACT_REPLAY_CORRECT` diagnostic unlocks a minimal exact-key production dispatch; a second arrival-load gate then compares default eager multi-sequence decode with the default-off exact-graph candidate.

**Tech Stack:** Python 3, PyTorch CUDA Graphs, FlashAttention KV-cache decode, Qwen3-0.6B BF16, JSON/JSONL, `torch.save` tensor shards, SHA256, TinyLLMForge `ModelRunner`/`LLMEngine`, source-bound SSH execution on `sitian@10.232.195.203`.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; never modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- Preserve the current production multi-sequence eager guard through the complete diagnostic stage.
- Do not write the production dispatch patch before an independent canonical diagnostic returns `EXACT_REPLAY_CORRECT`.
- Keep `multi_sequence_cuda_graph_exact=False` by default.
- Production replay may use only `input_ids.size(0) in self.graphs`; never select a larger graph key.
- Non-exact batches such as `3`, `5`, and `9` remain eager in the first production candidate.
- Diagnostic scope is Qwen3-0.6B, BF16, TP=1, greedy, baseline full attention only.
- Exclude mixed prefill, spec verify, Quest, Attention Matching, KV quantization, weight/activation quantization, CPU/KV offload, KV cartridge, input embeddings, hidden-state return, and non-greedy sampling.
- The diagnostic batch shapes are exactly `2, 3, 4, 5, 8, 9, 16`.
- Diagnostic graph mappings are exact `N -> N` and rounded `2->4, 3->4, 4->8, 5->8, 8->16, 9->16, 16->32`.
- Diagnostic trajectories are exactly `uniform-short`, `ragged-context`, and `duplicate-and-distinct`.
- Run exactly `2` warmup decode steps, `16` measured decode steps, and `3` repetitions.
- The canonical diagnostic is exactly `7 × 3 × 3 × 3 = 189` isolated model processes.
- Eager produces the reference token array first; graph modes record their own argmax but feed eager's reference token into the next step.
- Logit comparison is frozen at `torch.testing.assert_close(rtol=1e-3, atol=1e-2)` plus exact argmax equality.
- Full logit and layer tensors must be independently loaded from hashed `.pt` shards; producer summary fields are not trusted.
- Every required Qwen3 decoder layer must be observed; missing layer evidence makes the diagnostic `INCOMPLETE`.
- KV evidence must cover every active write slot, slot zero, inactive declared slots, and deterministic untouched sentinels.
- Top-level diagnostic classifications are exactly `EXACT_REPLAY_CORRECT`, `EXACT_REPLAY_CORRUPT`, and `INCOMPLETE`; rounded classification is separate.
- GPU/model work runs only on `sitian@10.232.195.203`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Give every model process distinct dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Do not modify the remote checkout, use rsync, kill unrelated processes, clear shared `/tmp`, or reuse fixed ports.
- Only `EADDRINUSE` permits retry with a fresh port pair.
- Preserve unrelated untracked `experiments/` directories; stage exact paths only and never use `git add -A`.
- Do not update README before an independently verified production `GO`.
- If diagnostic is not `EXACT_REPLAY_CORRECT`, stop before Tasks 6-8 and record the result in `AGENT_HANDOFF_STATE.md`.
- Production `GO` thresholds are frozen: aggregate decode throughput `>=1.15x`, stable-exact decode throughput `>=1.25x`, every request-throughput ratio `>=0.95x`, every p95 ITL ratio `<=1.05x`, every p99 ITL ratio `<=1.10x`, peak reserved memory ratio `<=1.02x`, initialization ratio `<=1.05x`, stable-exact graph hit rate `>=0.60`.

---

## File Map

- Create `tools/multi_sequence_cuda_graph_contract.py`: frozen diagnostic/production constants, matrix builders, canonical JSON/hash helpers, tensor-shard metadata validation, comparison helpers, and classifications.
- Create `tools/test_multi_sequence_cuda_graph_gate.py`: dependency-light TDD for matrix, artifacts, comparison, classification, production thresholds, and source/remote contracts.
- Create `tools/diagnose_multi_sequence_cuda_graph.py`: one isolated GPU diagnostic process; prompt/KV construction, exact/rounded graph capture, teacher forcing, layer observation, tensor shards, and KV evidence.
- Create `tools/verify_multi_sequence_cuda_graph_diagnostic.py`: independent diagnostic and production artifact verifier; never imports producer aggregation functions.
- Create `tools/run_multi_sequence_cuda_graph_diagnostic_remote.py`: source snapshot upload, capability preflight, dynamic-port process orchestration, resumable download, and local independent verification.
- Create `tools/multi_sequence_cuda_graph_batching_gate.py`: fixed arrival manifests, two-policy batching matrix, case execution, aggregation, report, and artifact finalization.
- Modify `tinyvllm/config.py`: add default-off `multi_sequence_cuda_graph_exact` and fail-closed incompatibility validation.
- Modify `tinyvllm/engine/model_runner.py`: extract exact graph-key selection/replay helpers, preserve current guard by default, and publish execution-path evidence.
- Modify `tinyvllm/engine/llm_engine.py`: include model-runner execution-path evidence in `last_step_observation`.
- Modify `tools/arrival_load_driver.py`: validate and persist exact-graph execution evidence when the production candidate is active.
- Modify `tools/test_model_runner_spec_verify.py`: dependency-light exact-key/eager fallback dispatch tests.
- Modify `tools/test_arrival_load_driver.py`: execution-path evidence tests.
- Modify `AGENT_HANDOFF_STATE.md`: record canonical diagnostic and, if reached, production gate result and boundaries.
- Modify `README.md` only after independent production `GO`.
- Create untracked raw artifacts under `experiments/cuda_graph/` using the
  runner's timestamped run tag; never stage them unless the user later
  explicitly requests artifact promotion.

## Shared Interfaces

Use these exact interfaces across tasks:

```python
DIAGNOSTIC_BATCH_SIZES = (2, 3, 4, 5, 8, 9, 16)
DIAGNOSTIC_TRAJECTORIES = (
    "uniform-short",
    "ragged-context",
    "duplicate-and-distinct",
)
DIAGNOSTIC_MODES = ("eager", "exact_graph", "rounded_graph")
ROUNDED_GRAPH_SIZE = {2: 4, 3: 4, 4: 8, 5: 8, 8: 16, 9: 16, 16: 32}
DIAGNOSTIC_REPETITIONS = 3
WARMUP_STEPS = 2
MEASURED_STEPS = 16
LOGIT_RTOL = 1e-3
LOGIT_ATOL = 1e-2
```

```python
@dataclass(frozen=True)
class DiagnosticCase:
    batch_size: int
    trajectory: str
    mode: str
    repetition: int
    graph_size: int

    @property
    def case_id(self) -> str:
        return (
            f"b{self.batch_size}__{self.trajectory}__"
            f"{self.mode}__r{self.repetition}"
        )
```

```python
def build_diagnostic_matrix() -> tuple[DiagnosticCase, ...]:
    """Return the exact 189-case diagnostic matrix."""


def canonical_json_bytes(value: object) -> bytes:
    """Return stable UTF-8 JSON bytes with NaN rejected."""


def canonical_json_sha256(value: object) -> str:
    """Hash canonical JSON bytes."""


def sha256_file(path: Path) -> str:
    """Hash one closed artifact file."""


def diagnostic_graph_size(batch_size: int, mode: str) -> int:
    """Return exact or strictly larger graph size from the frozen contract."""


def tensor_metadata(tensor: torch.Tensor) -> dict:
    """Return dtype, shape, finite status, and contiguous-byte SHA256."""


def compare_tensor_pair(
    eager: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float = LOGIT_RTOL,
    atol: float = LOGIT_ATOL,
) -> dict:
    """Independently compare shape, dtype, finite values, argmax, and closeness."""


def classify_diagnostic(
    *,
    matrix_rows: list[dict],
    logit_results: list[dict],
    layer_results: list[dict],
    kv_results: list[dict],
) -> dict:
    """Return exact and rounded classifications from complete raw evidence."""


def classify_production_gate(case_rows: list[dict]) -> dict:
    """Apply frozen correctness and performance thresholds."""
```

Diagnostic process CLI:

```text
python tools/diagnose_multi_sequence_cuda_graph.py
  --model MODEL
  --case-spec CASE_JSON
  --reference-tokens REFERENCE_JSON
  --output-dir OUTPUT_DIR
```

Remote orchestrator CLI:

```text
python tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
  preflight|diagnostic-smoke|diagnostic-canonical|production-smoke|production-canonical|download-only|verify-only
  --run-tag RUN_TAG
  [--diagnostic-run-tag DIAGNOSTIC_RUN_TAG]
```

Independent verifier CLI:

```text
python tools/verify_multi_sequence_cuda_graph_diagnostic.py
  --run-dir RUN_DIR
  --kind diagnostic|production
```

Model-runner execution evidence:

```python
{
    "execution_path": "eager" | "graph_single" | "graph_exact",
    "active_batch_size": int,
    "graph_batch_size": int | None,
}
```

---

### Task 1: Freeze the Diagnostic and Production Contracts

**Files:**
- Create: `tools/multi_sequence_cuda_graph_contract.py`
- Create: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: the approved design constants.
- Produces: `DiagnosticCase`, matrix builders, canonical hashing, tensor comparison, diagnostic classification, and production classification.

- [ ] **Step 1: Write failing matrix and graph-size tests**

Create `tools/test_multi_sequence_cuda_graph_gate.py` with:

```python
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "multi_sequence_cuda_graph_contract.py"


def load_contract():
    spec = importlib.util.spec_from_file_location("cuda_graph_contract", CONTRACT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


contract = load_contract()


def test_diagnostic_matrix_is_exact_and_unique():
    matrix = contract.build_diagnostic_matrix()
    assert len(matrix) == 189
    assert len({case.case_id for case in matrix}) == 189
    assert {case.batch_size for case in matrix} == {2, 3, 4, 5, 8, 9, 16}
    assert {case.trajectory for case in matrix} == {
        "uniform-short", "ragged-context", "duplicate-and-distinct",
    }
    assert {case.mode for case in matrix} == {
        "eager", "exact_graph", "rounded_graph",
    }
    assert {case.repetition for case in matrix} == {0, 1, 2}


def test_exact_and_rounded_graph_sizes_are_frozen():
    for batch_size in (2, 3, 4, 5, 8, 9, 16):
        assert contract.diagnostic_graph_size(batch_size, "eager") == batch_size
        assert contract.diagnostic_graph_size(batch_size, "exact_graph") == batch_size
    assert contract.ROUNDED_GRAPH_SIZE == {
        2: 4, 3: 4, 4: 8, 5: 8, 8: 16, 9: 16, 16: 32,
    }
```

- [ ] **Step 2: Write failing tensor-comparison tests**

Add:

```python
def test_tensor_comparison_requires_finite_close_and_equal_argmax():
    import torch

    eager = torch.tensor([[1.0, 2.0], [3.0, 1.0]])
    close = eager + torch.tensor([[0.001, -0.001], [0.001, 0.0]])
    result = contract.compare_tensor_pair(eager, close)
    assert result["finite"] is True
    assert result["argmax_equal"] is True
    assert result["close"] is True

    wrong_argmax = torch.tensor([[2.1, 2.0], [3.0, 1.0]])
    assert contract.compare_tensor_pair(eager, wrong_argmax)["argmax_equal"] is False

    nonfinite = eager.clone()
    nonfinite[0, 0] = float("nan")
    assert contract.compare_tensor_pair(eager, nonfinite)["finite"] is False
```

- [ ] **Step 3: Write failing diagnostic-classification tests**

Add this complete fixture helper and then tamper one exact row, one rounded
row, and one matrix row:

```python
def make_complete_diagnostic_evidence():
    matrix_rows = []
    logit_results = []
    layer_results = []
    kv_results = []
    for case in contract.build_diagnostic_matrix():
        matrix_rows.append({
            "case_id": case.case_id,
            "batch_size": case.batch_size,
            "trajectory": case.trajectory,
            "mode": case.mode,
            "repetition": case.repetition,
            "graph_size": case.graph_size,
            "status": "PASS",
        })
        if case.mode == "eager":
            continue
        common = {
            "case_id": case.case_id,
            "mode": case.mode,
            "batch_size": case.batch_size,
            "graph_size": case.graph_size,
        }
        logit_results.append({
            **common,
            "finite": True,
            "argmax_equal": True,
            "close": True,
        })
        layer_results.append({
            **common,
            "required_layer_count": 4,
            "observed_layer_count": 4,
            "finite": True,
            "close": True,
        })
        kv_results.append({
            **common,
            "active_slots_equal": True,
            "unexpected_slot_mutations": [],
        })
    return {
        "matrix_rows": matrix_rows,
        "logit_results": logit_results,
        "layer_results": layer_results,
        "kv_results": kv_results,
    }


def test_diagnostic_classification_separates_exact_and_rounded():
    complete = make_complete_diagnostic_evidence()
    result = contract.classify_diagnostic(**complete)
    assert result["classification"] == "EXACT_REPLAY_CORRECT"
    assert result["rounded_classification"] == "ROUNDED_REPLAY_CORRECT"

    rounded_bad = make_complete_diagnostic_evidence()
    rounded_bad["kv_results"][0]["unexpected_slot_mutations"] = [0]
    rounded_bad["kv_results"][0]["mode"] = "rounded_graph"
    result = contract.classify_diagnostic(**rounded_bad)
    assert result["classification"] == "EXACT_REPLAY_CORRECT"
    assert result["rounded_classification"] == "ROUNDED_REPLAY_CORRUPT"

    exact_bad = make_complete_diagnostic_evidence()
    exact_bad["logit_results"][0]["mode"] = "exact_graph"
    exact_bad["logit_results"][0]["close"] = False
    assert contract.classify_diagnostic(**exact_bad)["classification"] == (
        "EXACT_REPLAY_CORRUPT"
    )

    incomplete = make_complete_diagnostic_evidence()
    incomplete["matrix_rows"].pop()
    assert contract.classify_diagnostic(**incomplete)["classification"] == "INCOMPLETE"
```

- [ ] **Step 4: Write failing production-threshold boundary tests**

```python
def make_complete_production_rows(**overrides):
    values = {
        "aggregate_decode_ratio": 1.15,
        "stable_decode_ratio": 1.25,
        "minimum_request_ratio": 0.95,
        "maximum_p95_itl_ratio": 1.05,
        "maximum_p99_itl_ratio": 1.10,
        "peak_reserved_ratio": 1.02,
        "initialization_ratio": 1.05,
        "stable_graph_hit_rate": 0.60,
    }
    values.update(overrides)
    return [{
        **values,
        "structural_failures": [],
        "correctness_failures": [],
        "measured_repetitions_complete": True,
    }]


def test_production_gate_frozen_boundaries():
    rows = make_complete_production_rows(
        aggregate_decode_ratio=1.15,
        stable_decode_ratio=1.25,
        minimum_request_ratio=0.95,
        maximum_p95_itl_ratio=1.05,
        maximum_p99_itl_ratio=1.10,
        peak_reserved_ratio=1.02,
        initialization_ratio=1.05,
        stable_graph_hit_rate=0.60,
    )
    assert contract.classify_production_gate(rows)["classification"] == "GO"
    rows[0]["p99_itl_ratio"] = 1.100001
    assert contract.classify_production_gate(rows)["classification"] == "NO_GO"
```

- [ ] **Step 5: Run tests and verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: import failure because `multi_sequence_cuda_graph_contract.py` does not exist.

- [ ] **Step 6: Implement the immutable contract**

Create `tools/multi_sequence_cuda_graph_contract.py` with:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

DIAGNOSTIC_BATCH_SIZES = (2, 3, 4, 5, 8, 9, 16)
DIAGNOSTIC_TRAJECTORIES = (
    "uniform-short",
    "ragged-context",
    "duplicate-and-distinct",
)
DIAGNOSTIC_MODES = ("eager", "exact_graph", "rounded_graph")
ROUNDED_GRAPH_SIZE = {2: 4, 3: 4, 4: 8, 5: 8, 8: 16, 9: 16, 16: 32}
DIAGNOSTIC_REPETITIONS = 3
WARMUP_STEPS = 2
MEASURED_STEPS = 16
LOGIT_RTOL = 1e-3
LOGIT_ATOL = 1e-2

PRODUCTION_THRESHOLDS = {
    "aggregate_decode_ratio": 1.15,
    "stable_decode_ratio": 1.25,
    "minimum_request_ratio": 0.95,
    "maximum_p95_itl_ratio": 1.05,
    "maximum_p99_itl_ratio": 1.10,
    "peak_reserved_ratio": 1.02,
    "initialization_ratio": 1.05,
    "stable_graph_hit_rate": 0.60,
}


@dataclass(frozen=True)
class DiagnosticCase:
    batch_size: int
    trajectory: str
    mode: str
    repetition: int
    graph_size: int

    @property
    def case_id(self) -> str:
        return (
            f"b{self.batch_size}__{self.trajectory}__"
            f"{self.mode}__r{self.repetition}"
        )


def diagnostic_graph_size(batch_size: int, mode: str) -> int:
    if batch_size not in DIAGNOSTIC_BATCH_SIZES:
        raise ValueError(f"unsupported batch size: {batch_size}")
    if mode not in DIAGNOSTIC_MODES:
        raise ValueError(f"unsupported mode: {mode}")
    return ROUNDED_GRAPH_SIZE[batch_size] if mode == "rounded_graph" else batch_size


def build_diagnostic_matrix() -> tuple[DiagnosticCase, ...]:
    return tuple(
        DiagnosticCase(
            batch_size=batch_size,
            trajectory=trajectory,
            mode=mode,
            repetition=repetition,
            graph_size=diagnostic_graph_size(batch_size, mode),
        )
        for repetition in range(DIAGNOSTIC_REPETITIONS)
        for trajectory in DIAGNOSTIC_TRAJECTORIES
        for batch_size in DIAGNOSTIC_BATCH_SIZES
        for mode in DIAGNOSTIC_MODES
    )


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
```

Implement `compare_tensor_pair()` using CPU float32 difference math, explicit finite checks, exact shape/dtype checks, `torch.argmax(..., dim=-1)`, and `torch.testing.assert_close`. Implement classifiers by reconstructing the expected case IDs from `build_diagnostic_matrix()` and rejecting missing/duplicate keys before evaluating correctness.

- [ ] **Step 7: Run focused tests**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 -m py_compile \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "test: freeze multi-sequence cuda graph gate"
```

---

### Task 2: Implement One Isolated GPU Diagnostic Process

**Files:**
- Create: `tools/diagnose_multi_sequence_cuda_graph.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: one `DiagnosticCase`, one frozen prompt manifest, and eager reference tokens.
- Produces: `case_result.json`, `raw_rows.jsonl`, `layer_observations.jsonl`,
  `kv_observations.jsonl`, `tensors/logits/{case_id}.pt`,
  `tensors/layers/{case_id}.pt`, `tensors/kv/{case_id}.pt`,
  `process_environment.json`, and `exitcode`.

- [ ] **Step 1: Write failing pure prompt/slot-plan tests**

Add tests for:

```python
def test_prompt_plan_is_deterministic_and_covers_required_trajectories():
    diagnostic = load_diagnostic_module_without_gpu()
    first = diagnostic.build_prompt_plan(batch_size=16)
    second = diagnostic.build_prompt_plan(batch_size=16)
    assert first == second
    assert set(first) == {
        "uniform-short", "ragged-context", "duplicate-and-distinct",
    }
    ragged = first["ragged-context"]
    assert min(len(row) for row in ragged) < 256
    assert max(len(row) for row in ragged) > 256
    duplicate = first["duplicate-and-distinct"]
    assert duplicate[0] == duplicate[1]
    assert duplicate[0] != duplicate[2]


def test_kv_observation_plan_covers_active_zero_inactive_and_sentinels():
    plan = diagnostic.build_kv_observation_plan(
        active_slots=(300, 557, 814),
        graph_size=4,
        inactive_slots=(0,),
        total_slots=4096,
    )
    assert set(plan["active_write_slots"]) == {300, 557, 814}
    assert plan["slot_zero"] == 0
    assert plan["inactive_declared_slots"] == [0]
    assert set(plan["sentinel_slots"]).isdisjoint({0, 300, 557, 814})
```

- [ ] **Step 2: Write failing teacher-forcing and shard-schema tests**

```python
def test_teacher_forcing_records_observed_and_reference_tokens_separately():
    row = diagnostic.build_step_row(
        observed_argmax_token_ids=[7, 8],
        reference_next_input_token_ids=[7, 9],
    )
    assert row["observed_argmax_token_ids"] == [7, 8]
    assert row["reference_next_input_token_ids"] == [7, 9]


def test_tensor_shard_schema_rejects_missing_order_fields(tmp_path):
    import torch

    shard = {
        "schema_version": 1,
        "case_id": "b2__uniform-short__eager__r0",
        "tensor": torch.zeros(1, 2),
    }
    try:
        diagnostic.validate_tensor_shard(shard)
    except ValueError as exc:
        assert "step_ids" in str(exc)
    else:
        raise AssertionError("missing step_ids accepted")
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: failure because diagnostic helpers do not exist.

- [ ] **Step 4: Implement deterministic prompt and sequence preparation**

Implement:

```python
def build_prompt_plan(tokenizer, batch_size: int) -> dict[str, list[list[int]]]:
    short = tokenizer.encode("CUDA graph row isolation test.")
    long_seed = tokenizer.encode(
        "Explain why deterministic batching metadata matters for KV-cache decode. "
    )
    ragged_targets = [32, 64, 96, 128, 192, 224, 255, 257, 288, 320, 384, 448, 512, 576, 640, 704]
    ragged = [
        repeat_to_exact_token_count(long_seed, ragged_targets[index])
        for index in range(batch_size)
    ]
    distinct = [
        tokenizer.encode(f"Distinct CUDA graph row {index}.")
        for index in range(batch_size)
    ]
    if batch_size >= 2:
        distinct[1] = list(distinct[0])
    return {
        "uniform-short": [list(short) for _ in range(batch_size)],
        "ragged-context": ragged,
        "duplicate-and-distinct": distinct,
    }
```

Create an `LLM` with:

```python
LLM(
    model,
    enforce_eager=True,
    tensor_parallel_size=1,
    max_num_seqs=32,
    max_num_batched_tokens=32768,
    max_model_len=1024,
    gpu_memory_utilization=0.55,
)
```

Use the scheduler/block manager to allocate and prefill each sequence through the normal engine path. After prefill, retain the live `Sequence` objects and their real block tables, then bypass scheduler selection only for controlled decode calls.

- [ ] **Step 5: Implement tool-owned graph capture**

Implement a focused `CapturedDecodeGraph`:

```python
@dataclass
class CapturedDecodeGraph:
    graph_size: int
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    outputs: torch.Tensor
    layer_outputs: torch.Tensor
```

Capture exactly one graph per process. Allocate `layer_outputs` with shape:

```text
[2, num_hidden_layers, graph_size, hidden_size]
```

Register one forward hook per `llm.model_runner.model.model.layers[index]`. Each hook performs only:

```python
layer_outputs[0, index, :graph_size].copy_(output[0][:graph_size])
layer_outputs[1, index, :graph_size].copy_(output[1][:graph_size])
```

The hook and `copy_` must be active during warmup and inside `torch.cuda.graph(...)`; no Python list append, CPU copy, hash, allocation, or synchronization is allowed inside replay.

Dimension zero is fixed as:

```text
0 = hidden_states
1 = residual
```

The independent verifier compares both components for every layer, step, and
active row.

For rounded mode, reproduce the historical contract exactly: zero all static buffers, copy only active rows, leave inactive rows zero, replay, and slice active outputs. Do not introduce scratch rows in this task.

- [ ] **Step 6: Implement step evidence and immutable tensor shards**

For each step:

1. call `prepare_decode(seqs)` to derive real active metadata;
2. copy active metadata into static graph buffers or execute eager;
3. record logits and layer buffers after synchronization;
4. snapshot KV slots before and after with `snapshot_kv_slots()`;
5. append observed argmax and eager reference next-input token IDs;
6. append the eager reference tokens to each sequence for the next step.

Write shards with:

```python
torch.save({
    "schema_version": 1,
    "case_id": case.case_id,
    "dtype": str(tensor.dtype),
    "shape": list(tensor.shape),
    "step_ids": list(range(MEASURED_STEPS)),
    "row_ids": list(range(case.batch_size)),
    "tensor": tensor.cpu().contiguous(),
}, path)
```

Write KV shards containing ordered slot IDs plus full before/after K/V tensors. Hash every artifact after close; use atomic JSON writes and append-only JSONL with final newlines.

- [ ] **Step 7: Add fail-closed runtime validation**

Reject before model construction unless:

- case is in `build_diagnostic_matrix()`;
- model config is Qwen3;
- actual dtype is BF16;
- TP is 1;
- all excluded features are disabled;
- `max_num_seqs >= 32`;
- exact/rounded graph size matches contract;
- eager reference token array exists for graph modes and has shape `[18, batch_size]` including warmup plus measured steps.

On exception, write `case_result.json`, traceback tail, and nonzero `exitcode`.

- [ ] **Step 8: Run local dependency-light tests**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 -m py_compile tools/diagnose_multi_sequence_cuda_graph.py
git diff --check
```

Expected: all PASS. Do not run the GPU diagnostic locally.

- [ ] **Step 9: Commit Task 2**

```bash
git add \
  tools/diagnose_multi_sequence_cuda_graph.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: add isolated cuda graph diagnostic"
```

---

### Task 3: Build the Independent Diagnostic Verifier

**Files:**
- Create: `tools/verify_multi_sequence_cuda_graph_diagnostic.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: canonical diagnostic raw artifacts and tensor shards.
- Produces: `independent-verification/summary.json`, `independent-verification/report.md`, and `independent-verification/verify.exitcode`.

- [ ] **Step 1: Write a complete synthetic artifact fixture**

The fixture must write all 189 process records, full `.pt` shards for logits/layers/KV, source/environment evidence, manifest, raw rows, and SHA256 map. Tensor sizes may be tiny, but schemas and keys must match canonical production schemas.

- [ ] **Step 2: Write failing independent reconstruction tests**

```python
def test_verifier_reconstructs_complete_diagnostic(tmp_path):
    run_dir = write_complete_diagnostic_fixture(tmp_path)
    summary = verifier.verify_diagnostic(run_dir)
    assert summary["classification"] == "EXACT_REPLAY_CORRECT"
    assert summary["rounded_classification"] == "ROUNDED_REPLAY_CORRECT"
    assert summary["case_count"] == 189
```

- [ ] **Step 3: Write one tamper test per evidence class**

Cover:

- missing/duplicate matrix key;
- source-tree drift;
- environment drift;
- duplicate/reused port;
- graph-size mismatch;
- changed prompt/reference-token hash;
- truncated JSONL;
- rehashed logit tensor mutation;
- non-finite logit;
- exact argmax mismatch;
- close-threshold failure;
- missing layer index;
- rehashed layer tensor mutation;
- active KV mismatch;
- unexpected slot-zero mutation;
- unexpected sentinel mutation;
- producer classification tamper;
- missing artifact hash.

Each test must rehash the tampered file and still expect verifier failure, proving the verifier checks semantics rather than hashes alone.

- [ ] **Step 4: Verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: verifier import or semantic tests fail.

- [ ] **Step 5: Implement verifier without producer aggregation imports**

The verifier may import constants and pure `compare_tensor_pair()` from `multi_sequence_cuda_graph_contract.py`. It must not import `diagnose_multi_sequence_cuda_graph.py`.

Implement:

```python
def verify_diagnostic(run_dir: Path) -> dict:
    verify_source_and_environment(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    verify_frozen_manifest(manifest)
    process_rows = read_jsonl(run_dir / "process_rows.jsonl")
    raw_rows = read_jsonl(run_dir / "raw_rows.jsonl")
    layer_rows = read_jsonl(run_dir / "layer_observations.jsonl")
    kv_rows = read_jsonl(run_dir / "kv_observations.jsonl")
    verify_exact_matrix(process_rows)
    logit_results = independently_compare_logit_shards(run_dir, manifest)
    layer_results = independently_compare_layer_shards(run_dir, manifest)
    kv_results = independently_compare_kv_shards(run_dir, manifest)
    result = classify_diagnostic(
        matrix_rows=process_rows,
        logit_results=logit_results,
        layer_results=layer_results,
        kv_results=kv_results,
    )
    verify_producer_summary(run_dir, result)
    return result
```

Derive first divergent step/row/layer/slot by scanning ordered raw tensor evidence, not producer labels.

- [ ] **Step 6: Run focused tests**

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 -m py_compile \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/multi_sequence_cuda_graph_contract.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: verify cuda graph diagnostic evidence"
```

---

### Task 4: Add Source-Bound Remote Diagnostic Orchestration

**Files:**
- Create: `tools/run_multi_sequence_cuda_graph_diagnostic_remote.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: clean or explicitly snapshotted local source plus the fixed diagnostic matrix.
- Produces: source/environment evidence, per-process artifacts, merged canonical artifacts, and a local independent verification.

- [ ] **Step 1: Write failing command/transport contract tests**

Assert the runner source contains and uses:

```text
sitian@10.232.195.203
/tmp/ssh-sitian-10.232.195.203
/data00/home/sitian/sitian-workspace01/tllm/env/bin/python
/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B
TINYVLLM_DIST_PORT
MASTER_PORT
EADDRINUSE
source_audit.build_source_evidence
source_audit.validate_source_snapshot
```

Assert it does not contain:

```text
rsync
pkill
killall
rm -rf /tmp
git checkout
git reset
git clean
git add -A
```

- [ ] **Step 2: Write failing resume and port tests**

Mock `subprocess.run` and test:

- two ports are distinct;
- every process gets a globally unique pair;
- only `EADDRINUSE` retries;
- completed case resume requires matching source/environment/case identity and artifact hashes;
- graph modes cannot start before the matching eager reference-token artifact exists;
- failed processes preserve stdout/stderr/case artifacts.

- [ ] **Step 3: Verify RED**

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: remote-runner tests fail because the file does not exist.

- [ ] **Step 4: Implement immutable source snapshot and capability preflight**

Use `tools/source_audit.py` to create:

```text
source_evidence.json
source.patch
source_snapshot.tar.gz
```

Upload with:

```text
tar -C STAGING -cf - . | ssh -S /tmp/ssh-sitian-10.232.195.203 ...
```

Never use rsync. Validate bytes remotely before running tests. Record:

- Python;
- PyTorch;
- CUDA runtime;
- NVIDIA driver;
- GPU name;
- FlashAttention;
- Transformers;
- Qwen3 model identity;
- BF16 support;
- source tree SHA256.

Run remote preflight:

```bash
REMOTE_PYTHON tools/test_multi_sequence_cuda_graph_gate.py
REMOTE_PYTHON tools/test_model_runner_spec_verify.py
REMOTE_PYTHON -m py_compile \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/diagnose_multi_sequence_cuda_graph.py \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py
```

- [ ] **Step 5: Implement eager-first orchestration**

For each `(batch_size, trajectory, repetition)`:

1. run eager;
2. validate eager process exit and reference token artifact;
3. launch exact graph;
4. launch rounded graph.

The canonical order may interleave tuples, but graph cases must never run before their eager dependency. Use at most one GPU model process at a time.

Create non-authoritative smoke with:

```text
batch sizes       2, 3, 4
trajectories      uniform-short, ragged-context
repetitions       1
modes             eager, exact_graph, rounded_graph
```

Smoke cannot write canonical classification.

- [ ] **Step 6: Implement artifact merge/download and local verification**

Download every file in bounded blocks with remote size checks and atomic `.partial` rename. Build canonical:

```text
manifest.json
environment.json
process_rows.jsonl
raw_rows.jsonl
layer_observations.jsonl
kv_observations.jsonl
tensors/logits/*.pt
tensors/layers/*.pt
tensors/kv/*.pt
summary.json
report.md
sha256sums.txt
```

Run the local independent verifier after download. Remote success alone is insufficient.

- [ ] **Step 7: Run local tests**

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 -m py_compile tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: orchestrate remote cuda graph diagnostic"
```

---

### Task 5: Run the Diagnostic Smoke and Canonical Checkpoint

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Create untracked: timestamped smoke and canonical directories under
  `experiments/cuda_graph/`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: a fresh independently verified diagnostic classification.

- [ ] **Step 1: Run local preflight**

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_model_runner_spec_verify.py
python3 -m py_compile \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/diagnose_multi_sequence_cuda_graph.py \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 2: Verify remote connectivity without changing state**

```bash
ssh -n \
  -o BatchMode=yes \
  -o ControlMaster=auto \
  -o ControlPersist=600 \
  -S /tmp/ssh-sitian-10.232.195.203 \
  sitian@10.232.195.203 \
  'printf "REMOTE_OK\n"'
```

Expected: `REMOTE_OK`.

- [ ] **Step 3: Run fresh remote preflight**

```bash
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py preflight
```

Expected: source validation and remote dependency-light tests PASS.

- [ ] **Step 4: Run diagnostic smoke**

```bash
SMOKE_RUN_TAG="qwen3-06b-cuda-graph-smoke-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  diagnostic-smoke \
  --run-tag "${SMOKE_RUN_TAG}"
```

Expected: all smoke processes complete, local independent verifier exits `0`, and smoke is explicitly marked non-authoritative.

- [ ] **Step 5: Review smoke evidence before canonical**

Inspect:

```bash
export SMOKE_RUN_TAG
python3 - <<'PY'
import json
import os
from pathlib import Path
run = Path("experiments/cuda_graph") / os.environ["SMOKE_RUN_TAG"]
for name in ("manifest.json", "summary.json", "independent-verification/summary.json"):
    print(name)
    print(json.dumps(json.loads((run / name).read_text()), indent=2)[:6000])
PY
```

Do not alter matrix, prompts, thresholds, or comparison tolerances based on smoke values. Fix only implementation/evidence defects, using RED→GREEN tests, and rerun a fresh smoke if source changes.

- [ ] **Step 6: Run canonical diagnostic**

```bash
DIAGNOSTIC_RUN_TAG="qwen3-06b-cuda-graph-canonical-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  diagnostic-canonical \
  --run-tag "${DIAGNOSTIC_RUN_TAG}"
```

Expected: 189 complete isolated processes, unique ports, source/environment identity match, and local independent verifier exit `0`.

- [ ] **Step 7: Enforce the hard checkpoint**

Read:

```bash
python3 - "${DIAGNOSTIC_RUN_TAG}" <<'PY'
import json
import sys
from pathlib import Path
path = (
    Path("experiments/cuda_graph")
    / sys.argv[1]
    / "independent-verification/summary.json"
)
print(json.dumps(json.loads(path.read_text()), indent=2))
PY
```

Branch:

- `EXACT_REPLAY_CORRECT`: continue to Task 6.
- `EXACT_REPLAY_CORRUPT`: stop; do not modify `Config` or production dispatch.
- `INCOMPLETE`: repair evidence/runner defects only, then run a fresh source-bound canonical.

Rounded classification does not control Task 6. Even `ROUNDED_REPLAY_CORRECT` remains non-production.

- [ ] **Step 8: Record diagnostic evidence**

Append to `AGENT_HANDOFF_STATE.md`:

- source commit/tree SHA;
- remote environment identity;
- run ID;
- exact and rounded classifications;
- first divergence evidence if any;
- what the diagnostic proves and does not prove;
- whether Tasks 6-8 are admitted.

- [ ] **Step 9: Commit only the handoff update**

```bash
git add AGENT_HANDOFF_STATE.md
git commit -m "docs: record multi-sequence graph diagnostic"
```

Do not stage `experiments/cuda_graph/`.

---

### Task 6: Add the Default-Off Exact-Key Production Dispatch

**Prerequisite:** Task 5 independent classification is
`EXACT_REPLAY_CORRECT`, and `DIAGNOSTIC_RUN_TAG` remains set to that canonical
run tag for Tasks 6-8.

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: independently verified exact replay.
- Produces: default-off `multi_sequence_cuda_graph_exact` and exact-key-only graph dispatch.

- [ ] **Step 1: Write failing config tests**

Extend the real-config test fixture:

```python
def test_multi_sequence_cuda_graph_exact_defaults_off_and_rejects_incompatible_modes():
    config = make_real_config()
    assert config.multi_sequence_cuda_graph_exact is False
    make_real_config(multi_sequence_cuda_graph_exact=True)
    for incompatible in (
        {"enforce_eager": True},
        {"quest_top_k_blocks": 4},
        {"am_compact_blocks": 4},
        {"kv_quant_bits": 4},
        {"cpu_offload": True},
        {"kv_offload_mvp0": True},
        {"tensor_parallel_size": 2},
    ):
        try:
            make_real_config(
                multi_sequence_cuda_graph_exact=True,
                **incompatible,
            )
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"incompatible exact graph config accepted: {incompatible}"
            )
```

- [ ] **Step 2: Replace the old dispatch tests with exact-key tests**

Add:

```python
def test_exact_multi_sequence_decode_replays_same_size_graph():
    runner = make_runner(multi_sequence_cuda_graph_exact=True)
    runner.graphs = {2: RecordingGraph(2), 4: ForbiddenGraph()}
    runner.graph_bs = [1, 2, 4]
    install_graph_buffers(runner, max_bs=4)
    logits = run_decode(runner, batch_size=2)
    assert runner.graphs[2].replay_calls == 1
    assert logits.values == expected_first_two_rows()


def test_non_exact_multi_sequence_decode_stays_eager():
    runner = make_runner(multi_sequence_cuda_graph_exact=True)
    runner.graphs = {4: ForbiddenGraph()}
    runner.graph_bs = [1, 2, 4]
    logits = run_decode(runner, batch_size=3)
    assert logits.values == eager_rows(3)


def test_missing_exact_graph_key_fails_closed_to_eager():
    runner = make_runner(multi_sequence_cuda_graph_exact=True)
    runner.graphs = {}
    assert run_decode(runner, batch_size=2).values == eager_rows(2)
```

Keep the existing batch-one replay test and unsupported-feature eager tests.

- [ ] **Step 3: Verify RED**

```bash
python3 tools/test_model_runner_spec_verify.py
```

Expected: failure because config and exact-key dispatch do not exist.

- [ ] **Step 4: Add config field and validation**

Add:

```python
multi_sequence_cuda_graph_exact: bool = False
```

When true, require:

```python
assert not self.enforce_eager
assert self.tensor_parallel_size == 1
assert self.quest_top_k_blocks <= 0
assert self.am_compact_blocks == 0
assert self.kv_quant_bits == 0
assert not self.cpu_offload
assert not self.kv_offload_mvp0
```

- [ ] **Step 5: Extract exact-key replay helper**

Add:

```python
def _decode_graph_key(self, *, mode: str, batch_size: int) -> int | None:
    if mode != "decode":
        return None
    if batch_size == 1:
        return 1 if 1 in self.graphs else None
    if not self.config.multi_sequence_cuda_graph_exact:
        return None
    return batch_size if batch_size in self.graphs else None
```

Change replay to index `self.graphs[graph_key]` directly. Never use:

```python
next(x for x in self.graph_bs if x >= bs)
```

for production dispatch.

Set per-call evidence:

```python
self.last_execution_observation = {
    "execution_path": "graph_single" if bs == 1 else "graph_exact",
    "active_batch_size": bs,
    "graph_batch_size": graph_key,
}
```

The eager branch publishes `execution_path="eager"` and `graph_batch_size=None`.

- [ ] **Step 6: Run focused and regression tests**

```bash
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_chunked_prefill.py
python3 tools/test_ngram_speculative.py
python3 -m py_compile tinyvllm/config.py tinyvllm/engine/model_runner.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 7: Commit Task 6**

```bash
git add \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py
git commit -m "feat: enable exact-bucket multi-sequence graphs"
```

---

### Task 7: Build Production Execution Evidence and Batching Gate

**Prerequisite:** Task 6 exists and remains default-off.

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/arrival_load_driver.py`
- Modify: `tools/test_arrival_load_driver.py`
- Create: `tools/multi_sequence_cuda_graph_batching_gate.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`
- Modify: `tools/verify_multi_sequence_cuda_graph_diagnostic.py`
- Modify: `tools/run_multi_sequence_cuda_graph_diagnostic_remote.py`

**Interfaces:**
- Consumes: exact-key production dispatch.
- Produces: two-policy source-bound production batching artifacts and independent `GO/NO_GO/INCOMPLETE`.

- [ ] **Step 1: Write failing engine/driver evidence tests**

Assert `LLMEngine.last_step_observation` includes:

```python
"execution_path"
"active_batch_size"
"graph_batch_size"
"model_step_start_ns"
"model_step_end_ns"
"model_step_duration_ns"
```

Driver validation must reject:

- `graph_exact` with null/different graph size;
- `graph_exact` when candidate flag is false;
- `graph_exact` for batch 1;
- `eager` with non-null graph size;
- rounded graph size;
- boolean values where integers are required.

- [ ] **Step 2: Write failing workload/matrix tests**

Freeze:

```python
PRODUCTION_WORKLOADS = ("stable-exact", "ragged-natural", "churn")
PRODUCTION_POLICIES = ("EAGER_BASELINE", "EXACT_GRAPH_CANDIDATE")
PRODUCTION_WARMUP_REPETITIONS = 1
PRODUCTION_MEASURED_REPETITIONS = 5
```

Tests must assert:

- policy order alternates each repetition;
- every process has a unique port pair;
- stable-exact produces sustained active sizes `2,4,8,16`;
- churn includes observed `3,5,9`;
- eager policy sets `multi_sequence_cuda_graph_exact=False`;
- candidate sets it true;
- all other engine config is byte-identical.

- [ ] **Step 3: Verify RED**

```bash
python3 tools/test_arrival_load_driver.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: missing evidence/gate failures.

- [ ] **Step 4: Publish model-step execution evidence**

In `LLMEngine.step()` bracket only the synchronous call:

```python
model_step_start_ns = self._clock_ns()
token_ids = self.model_runner.call(
    "run", seqs, is_prefill, do_sample, batch_kind
)
model_step_end_ns = self._clock_ns()
```

Merge `self.model_runner.last_execution_observation` into `last_step_observation`, with prefill/mixed paths explicitly marked eager/non-graph.

- [ ] **Step 5: Validate evidence in arrival driver**

Add `_validate_cuda_graph_execution_observation()` and invoke it when the case spec contains:

```python
"cuda_graph_gate_policy": "EAGER_BASELINE" | "EXACT_GRAPH_CANDIDATE"
```

Persist evidence unchanged into `scheduler_trace.jsonl`; do not substitute driver timestamps for model-step timestamps.

- [ ] **Step 6: Implement fixed production workloads**

Create deterministic tokenized manifests:

- `stable-exact`: four waves that hold active decode batches at `2`, `4`, `8`, and `16` for at least 16 measured decode steps each;
- `ragged-natural`: staggered arrivals and output lengths producing mixed exact/non-exact shapes;
- `churn`: output lengths chosen so active sizes cross `2,3,4,5,8,9,16`, with verifier-required observed fallback events for `3,5,9`.

Each request uses greedy `temperature=0`, `ignore_eos=True`, and frozen output length. Record workload SHA256 before execution.

- [ ] **Step 7: Implement two-policy case execution and aggregation**

Reuse `arrival_load_driver.py` as a subprocess. Record:

```text
request_timeline.jsonl
scheduler_trace.jsonl
memory_trace.jsonl
case_rows.jsonl
```

Aggregate:

- request throughput;
- decode token throughput;
- median/p95/p99 ITL;
- median/p95/max model-step duration;
- exact graph hit rate;
- non-exact eager fallback rate;
- peak allocated/reserved memory;
- initialization duration;
- exact output token SHA.

Run one warmup plus five measured repetitions for every policy/workload pair.

- [ ] **Step 8: Extend independent verifier for production**

`verify_production()` must:

- reconstruct all case metrics from raw request/scheduler/memory rows;
- compare exact output arrays between paired policies;
- require candidate `graph_exact` only when active batch equals graph batch;
- require observed eager fallback for `3`, `5`, and `9` in churn;
- reject any rounded replay;
- validate process/source/environment/port identity;
- call `classify_production_gate()` only after structural correctness passes.

- [ ] **Step 9: Extend remote runner modes**

Add:

```text
production-smoke
production-canonical
```

Production modes require an explicit diagnostic predecessor run ID whose independent summary is `EXACT_REPLAY_CORRECT` and whose source/environment identity matches current source. If source changed for the production patch, bind the predecessor as design admission evidence but run a fresh production preflight and smoke from the new source.

- [ ] **Step 10: Run local suites**

```bash
python3 tools/test_arrival_load_driver.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_chunked_prefill.py
python3 -m py_compile \
  tinyvllm/engine/llm_engine.py \
  tools/arrival_load_driver.py \
  tools/multi_sequence_cuda_graph_batching_gate.py \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 11: Commit Task 7**

```bash
git add \
  tinyvllm/engine/llm_engine.py \
  tools/arrival_load_driver.py \
  tools/test_arrival_load_driver.py \
  tools/multi_sequence_cuda_graph_batching_gate.py \
  tools/test_multi_sequence_cuda_graph_gate.py \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
git commit -m "feat: gate exact multi-sequence graph batching"
```

---

### Task 8: Run the Production Gate, Audit Claims, and Record the Result

**Prerequisite:** Tasks 6-7 complete and local suites PASS.

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify conditionally: `README.md`
- Create untracked: a timestamped production run under
  `experiments/cuda_graph/`

**Interfaces:**
- Consumes: production candidate and diagnostic predecessor.
- Produces: independent production classification and final documentation.

- [ ] **Step 1: Run full local validation**

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_arrival_load_driver.py
python3 tools/test_chunked_prefill.py
python3 tools/test_ngram_speculative.py
python3 tools/test_arrival_load_gate.py
python3 tools/test_arrival_load_verify.py
python3 -m py_compile \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/llm_engine.py \
  tools/arrival_load_driver.py \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/diagnose_multi_sequence_cuda_graph.py \
  tools/multi_sequence_cuda_graph_batching_gate.py \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py
git diff --check
```

Expected: all PASS.

- [ ] **Step 2: Run fresh production preflight**

```bash
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  preflight \
  --run-tag "qwen3-06b-cuda-graph-production-preflight-$(date +%Y%m%d-%H%M%S)" \
  --diagnostic-run-tag "${DIAGNOSTIC_RUN_TAG}"
```

- [ ] **Step 3: Run production smoke**

```bash
PRODUCTION_SMOKE_RUN_TAG="qwen3-06b-cuda-graph-production-smoke-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  production-smoke \
  --run-tag "${PRODUCTION_SMOKE_RUN_TAG}" \
  --diagnostic-run-tag "${DIAGNOSTIC_RUN_TAG}"
```

Smoke must prove:

- exact outputs;
- at least one `graph_exact` event;
- at least one non-exact eager fallback;
- no rounded graph event;
- independent verifier exit `0`.

Smoke is not performance evidence.

- [ ] **Step 4: Run production canonical**

```bash
PRODUCTION_RUN_TAG="qwen3-06b-cuda-graph-production-canonical-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  production-canonical \
  --run-tag "${PRODUCTION_RUN_TAG}" \
  --diagnostic-run-tag "${DIAGNOSTIC_RUN_TAG}"
```

Expected: all policy/workload/repetition cases complete and local independent verifier exit `0`.

- [ ] **Step 5: Perform prompt-to-artifact completion audit**

Build a checklist mapping every approved design requirement to evidence:

1. exact 189-process diagnostic matrix;
2. exact/rounded separation;
3. teacher-forced reference arrays;
4. full logit shards;
5. all decoder layer shards;
6. KV active/zero/inactive/sentinel evidence;
7. unique ports and isolated processes;
8. source/environment identity;
9. default-off config;
10. exact-key-only dispatch;
11. non-exact eager fallback;
12. three production workloads;
13. one warmup plus five measured repetitions;
14. exact output/lifecycle equality;
15. throughput/latency/memory/init thresholds;
16. independent verifier reconstruction;
17. claim boundaries.

For each item, cite the concrete artifact filename and recomputed field. Treat missing evidence as `INCOMPLETE`; do not rely on producer `GO`.

- [ ] **Step 6: Record handoff result**

Append to `AGENT_HANDOFF_STATE.md`:

- production source commit/tree SHA;
- diagnostic predecessor;
- production run ID;
- independent classification;
- all gate ratios;
- graph hit/fallback shape histogram;
- correctness result;
- exact claim boundary;
- next direction under `NO_GO` or `INCOMPLETE`.

- [ ] **Step 7: Update README only for independent GO**

If and only if independent classification is `GO`, add a narrow bullet:

```text
Exact-bucket multi-sequence CUDA Graph decode: in the source-bound
Qwen3-0.6B BF16 TP=1 greedy production batching gate, exact-size graph replay
preserved exact output and improved aggregate decode throughput by the
independently verified aggregate ratio recorded in the canonical report, while
satisfying frozen latency, memory, and initialization gates.
Non-exact batches remain eager.
```

Before committing, replace the phrase `the independently verified aggregate
ratio recorded in the canonical report` with the exact numeric ratio from
`independent-verification/summary.json`. Do not claim padded replay, arbitrary
batches, other models, TP, quantization, or mixed prefill.

For `NO_GO` or `INCOMPLETE`, do not modify README.

- [ ] **Step 8: Run final documentation and state validation**

```bash
git diff --check
git status --short
git diff -- README.md AGENT_HANDOFF_STATE.md
```

Confirm no `experiments/cuda_graph/` path is staged.

- [ ] **Step 9: Commit documentation selectively**

For `GO`:

```bash
git add README.md AGENT_HANDOFF_STATE.md
git commit -m "docs: record exact graph batching result"
```

For `NO_GO` or `INCOMPLETE`:

```bash
git add AGENT_HANDOFF_STATE.md
git commit -m "docs: record exact graph batching result"
```

---

## Execution Stop Rules

Stop immediately and report rather than continuing when:

1. diagnostic independent verification is not `EXACT_REPLAY_CORRECT`;
2. any task requires weakening `rtol`, `atol`, exact token equality, layer coverage, or KV integrity;
3. exact production dispatch needs a rounded graph key;
4. remote execution would require modifying the remote checkout or shared `/tmp`;
5. source/environment identity cannot be reconstructed;
6. production correctness fails;
7. performance `GO` would require changing frozen workloads or thresholds.

If diagnostic exact replay is corrupt, the next written design should target the first divergent layer/step and distinguish FlashAttention capture from metadata/KV-store behavior.

If diagnostic exact replay is correct but production is `NO_GO`, retain the default-off implementation only if the user explicitly wants it; otherwise revert the production patch while preserving diagnostic/gate infrastructure. The next written design must choose scratch-row padded replay, bounded demand-driven exact capture, or another kernel/quantization bottleneck.
