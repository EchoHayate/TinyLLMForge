# Fixed-Split Multi-Sequence CUDA Graph Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover a default-off exact-key multi-sequence CUDA Graph candidate by making FlashAttention split policy explicit, independently proving fixed-16 replay correctness and legacy auto-split compatibility, and only then running the frozen production performance gate.

**Architecture:** Extend the existing CUDA Graph diagnostic rather than replacing it. A dependency-light contract owns fixed split identity, the 189-process same-policy matrix, the 63-pair/126-process compatibility matrix, and all classifications. An exception-safe execution-context scope installs fixed split 16 only around candidate eager, graph capture, and graph replay; ordinary eager remains auto split 0. The independent verifier reconstructs Gate A and Gate B from hashed raw tensors before any production dispatch code is allowed. If both gates pass, a separate default-off exact-key dispatch and arrival-load gate compare fixed-16 graph execution with the unchanged auto-split eager baseline.

**Tech Stack:** Python 3, PyTorch inference mode and CUDA Graphs, FlashAttention 2.6.3 KV-cache decode, TinyLLMForge `Context`/`ModelRunner`/`LLMEngine`, Qwen3-0.6B BF16 TP=1, JSON/JSONL, `torch.save`, SHA256, SSH source snapshots, dynamic distributed ports.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; never modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not spawn subagents.
- Preserve commit `d51f4ab` and the approved design at `docs/superpowers/specs/2026-07-21-fixed-split-multi-sequence-cuda-graph-recovery-design.md`.
- Preserve the historical canonical `EXACT_REPLAY_CORRUPT / ROUNDED_REPLAY_CORRUPT` artifacts and conclusion; do not relabel or overwrite them.
- Preserve the production batch-greater-than-one eager guard until a fresh independent Gate A returns `EXACT_REPLAY_CORRECT` and Gate B returns `LEGACY_COMPATIBLE`.
- Candidate graph warmup, capture, replay, and candidate eager comparator use exactly `flash_attn_num_splits=16`.
- Ordinary eager baseline and fallback use exactly `flash_attn_num_splits=0`.
- Do not expose arbitrary split tuning as a public configuration option.
- Gate A is exactly `7 batches × 3 trajectories × 3 modes × 3 repetitions = 189` isolated processes.
- Gate A modes are exactly `candidate_eager`, `exact_graph_fixed16`, and `rounded_graph_fixed16`.
- Gate B is exactly `7 batches × 3 trajectories × 3 repetitions = 63` logical pairs and `126` isolated processes.
- Gate B policies are exactly `legacy_eager_auto` and `candidate_eager_fixed16`.
- Batch sizes remain exactly `2, 3, 4, 5, 8, 9, 16`.
- Trajectories remain exactly `uniform-short`, `ragged-context`, and `duplicate-and-distinct`.
- Run exactly `2` warmup decode steps and `16` measured decode steps.
- Preserve teacher-forced continuation, full logits/layer/KV tensor shards, and exact greedy token comparison.
- Preserve `torch.testing.assert_close(rtol=1e-3, atol=1e-2)`; do not widen tolerance after observing results.
- Rounded replay remains diagnostic-only and can never unlock production rounded replay.
- Production candidate remains `multi_sequence_cuda_graph_exact=False` by default.
- Production candidate may replay only an exact `input_ids.size(0)` graph key with fixed-16 identity.
- Batch one behavior remains unchanged.
- All unsupported features remain fail-closed to eager.
- GPU/model work runs only on `sitian@10.232.195.203` as user `sitian`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Every model process uses unique dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Do not modify the remote checkout, use rsync, kill unrelated processes, or clear shared `/tmp`.
- Retry only recognized `EADDRINUSE` failures with a fresh port pair.
- Keep raw `experiments/` artifacts untracked; use exact-path staging and never `git add -A`.
- Do not update README before independent production `GO`.
- Every code task uses RED → GREEN TDD with focused tests before broader tests.
- When unit-test work begins, invoke `bits-unit-test-gen` and follow its workflow exactly, including start/end telemetry and mandatory `utree flush`.

---

## File Map

### Diagnostic and Contracts

- Modify `tools/multi_sequence_cuda_graph_contract.py`
  - fixed split constants and policy identities;
  - 189-case Gate A matrix;
  - 126-process Gate B matrix;
  - policy-aware case IDs and validation;
  - `classify_legacy_compatibility()`.
- Modify `tools/test_multi_sequence_cuda_graph_gate.py`
  - dependency-light RED/GREEN coverage for every new contract, artifact field, verifier branch, resume rule, and frozen threshold.
- Modify `tools/diagnose_multi_sequence_cuda_graph.py`
  - execute explicit split policy;
  - emit split identity in every artifact;
  - support both Gate A and Gate B cases.
- Modify `tools/verify_multi_sequence_cuda_graph_diagnostic.py`
  - independently verify policy identity;
  - compare fixed graph with fixed eager for Gate A;
  - compare fixed eager with auto eager for Gate B;
  - emit separate classifications.
- Modify `tools/run_multi_sequence_cuda_graph_diagnostic_remote.py`
  - orchestrate same-policy and compatibility smoke/canonical runs;
  - preserve eager/reference ordering;
  - merge both matrices and independently verify each gate.

### Execution Context

- Modify `tinyvllm/utils/context.py`
  - add exception-safe `temporary_flash_attn_num_splits()`.
- Modify `tools/test_context_modes.py`
  - prove nested restoration and exception restoration.

### Conditional Production Candidate

These files must not be modified until Task 7 verifies both canonical gates:

- Modify `tinyvllm/config.py`
  - add default-off `multi_sequence_cuda_graph_exact`.
- Modify `tinyvllm/engine/model_runner.py`
  - capture multi-sequence candidate graphs with fixed split 16;
  - store graph split identity;
  - dispatch exact keys only;
  - emit execution evidence.
- Modify `tools/test_model_runner_spec_verify.py`
  - dependency-light dispatch, identity, fallback, and policy-scope tests.
- Modify `tinyvllm/engine/llm_engine.py`
  - include model-runner execution evidence in `last_step_observation`.
- Modify `tools/arrival_load_driver.py`
  - persist and validate exact graph/fallback evidence.
- Modify `tools/test_arrival_load_driver.py`
  - test execution evidence propagation.
- Create `tools/multi_sequence_cuda_graph_batching_gate.py`
  - fixed workloads, paired policies, aggregation, frozen production classification.

### Results

- Modify `AGENT_HANDOFF_STATE.md`
  - record Gate A/B source-bound result;
  - record production result only if reached.
- Modify `README.md` only after independent production `GO`.
- Create timestamped untracked artifacts under `experiments/cuda_graph/`.

## Shared Interfaces

Use these exact names throughout implementation:

```python
AUTO_FLASH_ATTN_NUM_SPLITS = 0
MULTI_SEQUENCE_CUDA_GRAPH_FLASH_ATTN_NUM_SPLITS = 16

SAME_POLICY_MODES = (
    "candidate_eager",
    "exact_graph_fixed16",
    "rounded_graph_fixed16",
)

LEGACY_COMPATIBILITY_POLICIES = (
    "legacy_eager_auto",
    "candidate_eager_fixed16",
)

SPLIT_POLICIES = {
    "legacy_eager_auto": {
        "split_policy_name": "auto",
        "flash_attn_num_splits": 0,
    },
    "candidate_eager_fixed16": {
        "split_policy_name": "fixed16",
        "flash_attn_num_splits": 16,
    },
    "candidate_eager": {
        "split_policy_name": "fixed16",
        "flash_attn_num_splits": 16,
    },
    "exact_graph_fixed16": {
        "split_policy_name": "fixed16",
        "flash_attn_num_splits": 16,
    },
    "rounded_graph_fixed16": {
        "split_policy_name": "fixed16",
        "flash_attn_num_splits": 16,
    },
}
```

```python
@dataclass(frozen=True)
class DiagnosticCase:
    batch_size: int
    trajectory: str
    mode: str
    repetition: int
    graph_size: int
    split_policy_name: str
    flash_attn_num_splits: int

    @property
    def case_id(self) -> str:
        return (
            f"b{self.batch_size}__{self.trajectory}__"
            f"{self.mode}__{self.split_policy_name}"
            f"-s{self.flash_attn_num_splits}__r{self.repetition}"
        )
```

```python
@dataclass(frozen=True)
class LegacyCompatibilityCase:
    batch_size: int
    trajectory: str
    policy: str
    repetition: int
    split_policy_name: str
    flash_attn_num_splits: int

    @property
    def pair_id(self) -> str:
        return (
            f"b{self.batch_size}__{self.trajectory}"
            f"__compat__r{self.repetition}"
        )

    @property
    def case_id(self) -> str:
        return (
            f"{self.pair_id}__{self.policy}"
            f"__{self.split_policy_name}-s{self.flash_attn_num_splits}"
        )
```

```python
def build_diagnostic_matrix() -> tuple[DiagnosticCase, ...]:
    """Return the exact 189-process Gate A matrix."""


def build_legacy_compatibility_matrix(
) -> tuple[LegacyCompatibilityCase, ...]:
    """Return the exact 126-process Gate B matrix."""


def classify_legacy_compatibility(
    *,
    process_rows: list[dict],
    logit_results: list[dict],
    kv_results: list[dict],
    token_results: list[dict],
) -> dict:
    """Return LEGACY_COMPATIBLE, LEGACY_INCOMPATIBLE, or INCOMPLETE."""
```

```python
@contextmanager
def temporary_flash_attn_num_splits(num_splits: int):
    """Temporarily replace only Context.flash_attn_num_splits."""
```

Every process and tensor artifact must include:

```python
{
    "flash_attn_version": str,
    "split_policy_name": "auto" | "fixed16",
    "flash_attn_num_splits": 0 | 16,
    "comparison_policy_name": (
        "same_policy_fixed16"
        | "legacy_auto_vs_fixed16"
    ),
}
```

Production graph metadata uses:

```python
{
    "graph_batch_size": int,
    "split_policy_name": "auto" | "fixed16",
    "flash_attn_num_splits": int,
}
```

Production execution evidence uses:

```python
{
    "execution_path": "eager" | "graph_single" | "graph_exact_fixed16",
    "active_batch_size": int,
    "graph_batch_size": int | None,
    "split_policy_name": "auto" | "fixed16",
    "flash_attn_num_splits": 0 | 16,
}
```

---

### Task 1: Freeze Policy-Aware Gate A and Gate B Contracts

**Files:**
- Modify: `tools/multi_sequence_cuda_graph_contract.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: approved fixed-split design constants.
- Produces: policy-aware case classes, exact matrices, policy lookup, and compatibility classification.

- [ ] **Step 1: Start the mandatory unit-test workflow**

Invoke `bits-unit-test-gen` before editing tests. Follow its Python bootstrap,
target analysis, lite/pipeline routing, telemetry, and artifact rules exactly.
Targets for this task are:

```text
tools/multi_sequence_cuda_graph_contract.py
tools/test_multi_sequence_cuda_graph_gate.py
```

- [ ] **Step 2: Write failing policy and matrix tests**

Add focused tests:

```python
def test_same_policy_matrix_is_exact_policy_aware_and_unique():
    matrix = contract.build_diagnostic_matrix()
    assert len(matrix) == 189
    assert len({case.case_id for case in matrix}) == 189
    assert {case.mode for case in matrix} == {
        "candidate_eager",
        "exact_graph_fixed16",
        "rounded_graph_fixed16",
    }
    assert {
        (case.split_policy_name, case.flash_attn_num_splits)
        for case in matrix
    } == {("fixed16", 16)}


def test_legacy_compatibility_matrix_is_63_pairs_126_processes():
    matrix = contract.build_legacy_compatibility_matrix()
    assert len(matrix) == 126
    assert len({case.case_id for case in matrix}) == 126
    pair_counts = collections.Counter(case.pair_id for case in matrix)
    assert len(pair_counts) == 63
    assert set(pair_counts.values()) == {2}
    assert {
        (case.policy, case.split_policy_name, case.flash_attn_num_splits)
        for case in matrix
    } == {
        ("legacy_eager_auto", "auto", 0),
        ("candidate_eager_fixed16", "fixed16", 16),
    }


def test_case_ids_bind_split_policy_identity():
    case = contract.build_diagnostic_matrix()[0]
    assert "fixed16-s16" in case.case_id
    compatibility = contract.build_legacy_compatibility_matrix()
    assert any("auto-s0" in case.case_id for case in compatibility)
    assert any("fixed16-s16" in case.case_id for case in compatibility)
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because policy constants, `LegacyCompatibilityCase`, and
`build_legacy_compatibility_matrix()` do not exist.

- [ ] **Step 4: Implement the minimal policy-aware contracts**

Replace the old mode constants and extend the dataclasses:

```python
AUTO_FLASH_ATTN_NUM_SPLITS = 0
MULTI_SEQUENCE_CUDA_GRAPH_FLASH_ATTN_NUM_SPLITS = 16
SAME_POLICY_MODES = (
    "candidate_eager",
    "exact_graph_fixed16",
    "rounded_graph_fixed16",
)
LEGACY_COMPATIBILITY_POLICIES = (
    "legacy_eager_auto",
    "candidate_eager_fixed16",
)


def split_policy_for(execution_name: str) -> tuple[str, int]:
    try:
        policy = SPLIT_POLICIES[execution_name]
    except KeyError as exc:
        raise ValueError(
            f"unsupported split execution policy: {execution_name}"
        ) from exc
    return (
        str(policy["split_policy_name"]),
        int(policy["flash_attn_num_splits"]),
    )
```

Implement both matrix builders using the frozen loop order:

```python
def build_legacy_compatibility_matrix():
    return tuple(
        LegacyCompatibilityCase(
            batch_size=batch_size,
            trajectory=trajectory,
            policy=policy,
            repetition=repetition,
            split_policy_name=split_policy_for(policy)[0],
            flash_attn_num_splits=split_policy_for(policy)[1],
        )
        for repetition in range(DIAGNOSTIC_REPETITIONS)
        for trajectory in DIAGNOSTIC_TRAJECTORIES
        for batch_size in DIAGNOSTIC_BATCH_SIZES
        for policy in LEGACY_COMPATIBILITY_POLICIES
    )
```

- [ ] **Step 5: Write failing compatibility classification tests**

Add fixture helpers and tests that require:

```python
def test_legacy_compatibility_requires_tokens_close_logits_and_kv_ownership():
    complete = make_complete_legacy_compatibility_evidence()
    assert contract.classify_legacy_compatibility(
        **complete
    )["classification"] == "LEGACY_COMPATIBLE"

    token_bad = copy.deepcopy(complete)
    token_bad["token_results"][0]["tokens_equal"] = False
    assert contract.classify_legacy_compatibility(
        **token_bad
    )["classification"] == "LEGACY_INCOMPATIBLE"

    close_bad = copy.deepcopy(complete)
    close_bad["logit_results"][0]["close"] = False
    assert contract.classify_legacy_compatibility(
        **close_bad
    )["classification"] == "LEGACY_INCOMPATIBLE"

    kv_bad = copy.deepcopy(complete)
    kv_bad["kv_results"][0]["touched_slot_sets_equal"] = False
    assert contract.classify_legacy_compatibility(
        **kv_bad
    )["classification"] == "LEGACY_INCOMPATIBLE"


def test_legacy_compatibility_missing_or_mixed_policy_is_incomplete():
    complete = make_complete_legacy_compatibility_evidence()
    complete["process_rows"].pop()
    result = contract.classify_legacy_compatibility(**complete)
    assert result["classification"] == "INCOMPLETE"

    mixed = make_complete_legacy_compatibility_evidence()
    mixed["process_rows"][0]["flash_attn_num_splits"] = 16
    result = contract.classify_legacy_compatibility(**mixed)
    assert result["classification"] == "INCOMPLETE"
```

- [ ] **Step 6: Run classification tests and verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because `classify_legacy_compatibility()` is absent.

- [ ] **Step 7: Implement fail-closed compatibility classification**

Implement exact evidence-set checks. A logical pair is compatible only when:

```python
def _legacy_pair_correct(logit_row, kv_row, token_row):
    return all(
        (
            logit_row.get("finite") is True,
            logit_row.get("argmax_equal") is True,
            logit_row.get("close") is True,
            token_row.get("tokens_equal") is True,
            kv_row.get("touched_slot_sets_equal") is True,
            kv_row.get("unexpected_slot_mutations") == [],
        )
    )
```

Structural/policy omissions return `INCOMPLETE`; complete semantic failures
return `LEGACY_INCOMPATIBLE`.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: PASS with the existing suite plus new contract tests.

- [ ] **Step 9: Commit the contract**

```bash
git add -- \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "test: freeze fixed-split cuda graph contracts"
```

---

### Task 2: Add Exception-Safe Attention Split Scoping

**Files:**
- Modify: `tinyvllm/utils/context.py`
- Modify: `tools/test_context_modes.py`

**Interfaces:**
- Consumes: existing global `Context`.
- Produces: `temporary_flash_attn_num_splits(num_splits)`.

- [ ] **Step 1: Write failing restoration tests**

Add:

```python
def test_temporary_flash_attn_split_restores_previous_context():
    context.set_context(mode="decode", flash_attn_num_splits=0)
    original = context.get_context()
    with context.temporary_flash_attn_num_splits(16):
        assert context.get_context() is not original
        assert context.get_context().flash_attn_num_splits == 16
    assert context.get_context() is original
    assert context.get_context().flash_attn_num_splits == 0


def test_temporary_flash_attn_split_restores_after_exception():
    context.set_context(mode="decode", flash_attn_num_splits=0)
    original = context.get_context()
    try:
        with context.temporary_flash_attn_num_splits(16):
            raise RuntimeError("capture failed")
    except RuntimeError:
        pass
    assert context.get_context() is original
    assert context.get_context().flash_attn_num_splits == 0


def test_temporary_flash_attn_split_supports_nested_scopes():
    context.set_context(mode="decode", flash_attn_num_splits=0)
    with context.temporary_flash_attn_num_splits(16):
        with context.temporary_flash_attn_num_splits(1):
            assert context.get_context().flash_attn_num_splits == 1
        assert context.get_context().flash_attn_num_splits == 16
    assert context.get_context().flash_attn_num_splits == 0
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python3 tools/test_context_modes.py
```

Expected: FAIL because the context manager is absent.

- [ ] **Step 3: Implement the minimal context manager**

Use dataclass replacement and restore the exact previous object:

```python
from contextlib import contextmanager
from dataclasses import dataclass, replace


@contextmanager
def temporary_flash_attn_num_splits(num_splits: int):
    global _CONTEXT
    num_splits = int(num_splits)
    if num_splits < 0:
        raise ValueError("flash_attn_num_splits must be non-negative")
    previous = _CONTEXT
    _CONTEXT = replace(
        previous,
        flash_attn_num_splits=num_splits,
    )
    try:
        yield _CONTEXT
    finally:
        _CONTEXT = previous
```

- [ ] **Step 4: Run focused and neighboring tests**

Run:

```bash
python3 tools/test_context_modes.py
python3 tools/test_model_runner_spec_verify.py
```

Expected: both PASS.

- [ ] **Step 5: Commit the scope**

```bash
git add -- tinyvllm/utils/context.py tools/test_context_modes.py
git commit -m "feat: scope flash attention split policy"
```

---

### Task 3: Make the GPU Diagnostic Execute Explicit Policies

**Files:**
- Modify: `tools/diagnose_multi_sequence_cuda_graph.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: policy-aware cases and `temporary_flash_attn_num_splits()`.
- Produces: policy-bound process artifacts for Gate A and Gate B.

- [ ] **Step 1: Write failing case parsing and execution-policy tests**

Add tests using the dependency-light module loader:

```python
def test_diagnostic_case_parser_rejects_split_policy_drift():
    case = dataclasses.asdict(contract.build_diagnostic_matrix()[0])
    case["flash_attn_num_splits"] = 0
    with assert_raises(ValueError, "outside frozen diagnostic matrix"):
        diagnostic._parse_case(case)


def test_execution_policy_maps_gate_cases_to_expected_split():
    for case in contract.build_diagnostic_matrix():
        assert diagnostic.execution_split_count(case) == 16
    for case in contract.build_legacy_compatibility_matrix():
        expected = 0 if case.policy == "legacy_eager_auto" else 16
        assert diagnostic.execution_split_count(case) == expected
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because compatibility parsing and `execution_split_count()` are
absent.

- [ ] **Step 3: Generalize case parsing without weakening validation**

Implement:

```python
def _all_frozen_cases():
    return (
        tuple(contract.build_diagnostic_matrix())
        + tuple(contract.build_legacy_compatibility_matrix())
    )


def _parse_case(case_spec: dict):
    case_id = str(case_spec.get("case_id", ""))
    expected = {case.case_id: case for case in _all_frozen_cases()}
    if case_id not in expected:
        raise ValueError(f"case is outside frozen matrices: {case_id}")
    case = expected[case_id]
    if case_spec != {"case_id": case.case_id, **asdict(case)}:
        raise ValueError(f"case identity drift: {case.case_id}")
    return case


def execution_split_count(case) -> int:
    return int(case.flash_attn_num_splits)
```

The remote runner must write both `case_id` and the complete dataclass fields.

- [ ] **Step 4: Write failing context-observation tests**

Test helper boundaries rather than requiring CUDA:

```python
def test_candidate_eager_forward_observes_fixed16_and_restores_auto():
    seen = []
    result = diagnostic._run_with_split_policy(
        16,
        lambda: seen.append(context.get_context().flash_attn_num_splits),
    )
    assert result is None
    assert seen == [16]
    assert context.get_context().flash_attn_num_splits == 0


def test_legacy_eager_forward_observes_auto():
    seen = []
    diagnostic._run_with_split_policy(
        0,
        lambda: seen.append(context.get_context().flash_attn_num_splits),
    )
    assert seen == [0]
```

- [ ] **Step 5: Implement policy-scoped capture/eager/replay**

Add:

```python
def _run_with_split_policy(num_splits: int, operation):
    from tinyvllm.utils.context import temporary_flash_attn_num_splits
    with temporary_flash_attn_num_splits(num_splits):
        return operation()
```

Update `_run_eager_step()`:

```python
return _run_with_split_policy(
    execution_split_count(case),
    lambda: _forward_and_logits_with_layer_hooks(runner, dynamic),
)
```

Update `_capture_decode_graph()` so `set_context(...)` runs first and both
warmup and capture execute inside fixed-16 scope.

Update `_run_graph_step()` so replay executes inside fixed-16 scope and the
scope is restored before returning.

Legacy compatibility cases never capture graphs.

- [ ] **Step 6: Add policy identity to every artifact**

Extend:

```python
def policy_evidence(case, flash_attn_version: str) -> dict:
    return {
        "flash_attn_version": flash_attn_version,
        "split_policy_name": case.split_policy_name,
        "flash_attn_num_splits": case.flash_attn_num_splits,
        "comparison_policy_name": (
            "same_policy_fixed16"
            if isinstance(case, contract.DiagnosticCase)
            else "legacy_auto_vs_fixed16"
        ),
    }
```

Write it into:

```text
process_environment.json
case_result.json
raw_rows.jsonl
layer_observations.jsonl
kv_observations.jsonl
logit shard metadata
layer shard metadata
KV shard metadata
```

- [ ] **Step 7: Run focused tests and syntax checks**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 -m py_compile \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/diagnose_multi_sequence_cuda_graph.py
```

Expected: PASS.

- [ ] **Step 8: Commit diagnostic execution**

```bash
git add -- \
  tools/diagnose_multi_sequence_cuda_graph.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: bind cuda graph diagnostic split policy"
```

---

### Task 4: Extend the Independent Verifier for Gate A and Gate B

**Files:**
- Modify: `tools/verify_multi_sequence_cuda_graph_diagnostic.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: merged policy-aware artifacts.
- Produces: independent same-policy and legacy-compatibility classifications.

- [ ] **Step 1: Write failing verifier policy-integrity tests**

Add tests that mutate a complete fixture:

```python
def test_verifier_rejects_missing_split_identity():
    run_dir = write_complete_fixed_split_fixture()
    rows = _read_jsonl(run_dir / "process_rows.jsonl")
    rows[0].pop("flash_attn_num_splits")
    _rewrite_jsonl(run_dir / "process_rows.jsonl", rows)
    _refresh_sha256sums(run_dir)
    summary = verifier.verify_diagnostic(run_dir)
    assert summary["classification"] == "INCOMPLETE"


def test_verifier_rejects_auto_graph_as_fixed16_evidence():
    run_dir = write_complete_fixed_split_fixture()
    rows = _read_jsonl(run_dir / "process_rows.jsonl")
    graph = next(row for row in rows if "graph_fixed16" in row["mode"])
    graph["split_policy_name"] = "auto"
    graph["flash_attn_num_splits"] = 0
    _rewrite_jsonl(run_dir / "process_rows.jsonl", rows)
    _refresh_sha256sums(run_dir)
    summary = verifier.verify_diagnostic(run_dir)
    assert summary["classification"] == "INCOMPLETE"
```

- [ ] **Step 2: Write failing independent compatibility tests**

Add:

```python
def test_verifier_reconstructs_legacy_compatibility():
    run_dir = write_complete_fixed_split_fixture()
    summary = verifier.verify_diagnostic(run_dir)
    assert summary["classification"] == "EXACT_REPLAY_CORRECT"
    assert summary["legacy_compatibility"] == "LEGACY_COMPATIBLE"
    assert summary["same_policy_case_count"] == 189
    assert summary["compatibility_process_count"] == 126
    assert summary["compatibility_pair_count"] == 63


def test_verifier_reports_fixed_vs_auto_token_mismatch_as_legacy_incompatible():
    run_dir = write_complete_fixed_split_fixture()
    _mutate_compatibility_candidate_argmax(run_dir)
    _refresh_sha256sums(run_dir)
    summary = verifier.verify_diagnostic(run_dir)
    assert summary["classification"] == "EXACT_REPLAY_CORRECT"
    assert summary["legacy_compatibility"] == "LEGACY_INCOMPATIBLE"
```

- [ ] **Step 3: Run and verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the verifier assumes one 189-case matrix.

- [ ] **Step 4: Split verifier indexing by gate**

Add explicit builders:

```python
def _expected_same_policy_cases():
    return {
        case.case_id: case for case in contract.build_diagnostic_matrix()
    }


def _expected_compatibility_cases():
    return {
        case.case_id: case
        for case in contract.build_legacy_compatibility_matrix()
    }
```

Validate all policy fields against the frozen case object before loading tensor
shards. Missing or mixed policy is structural `INCOMPLETE`.

- [ ] **Step 5: Pair the correct references**

Gate A pairing key:

```python
(batch_size, trajectory, repetition)
candidate_eager -> exact_graph_fixed16 / rounded_graph_fixed16
```

Gate B pairing key:

```python
pair_id
legacy_eager_auto -> candidate_eager_fixed16
```

Reuse `compare_tensor_pair()` for logits. Require exact ordered token arrays and
equal touched-slot ownership sets for Gate B. Do not require bitwise logit or KV
value equality across auto and fixed reduction policies.

- [ ] **Step 6: Emit separate classifications and exit semantics**

Summary fields:

```python
{
    "classification": "EXACT_REPLAY_CORRECT" | "EXACT_REPLAY_CORRUPT" | "INCOMPLETE",
    "rounded_classification": str,
    "legacy_compatibility": "LEGACY_COMPATIBLE" | "LEGACY_INCOMPATIBLE" | "INCOMPLETE",
    "same_policy_case_count": 189,
    "compatibility_process_count": 126,
    "compatibility_pair_count": 63,
}
```

Verifier exit code is `0` only when:

```python
summary["classification"] == "EXACT_REPLAY_CORRECT"
and summary["legacy_compatibility"] == "LEGACY_COMPATIBLE"
```

Rounded classification does not affect exit zero.

- [ ] **Step 7: Run the full dependency-light suite**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 -m py_compile tools/verify_multi_sequence_cuda_graph_diagnostic.py
```

Expected: PASS.

- [ ] **Step 8: Commit verifier support**

```bash
git add -- \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: verify fixed-split replay and compatibility"
```

---

### Task 5: Extend Source-Bound Remote Orchestration

**Files:**
- Modify: `tools/run_multi_sequence_cuda_graph_diagnostic_remote.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: both frozen matrices and policy-aware diagnostic CLI.
- Produces: resumable smoke/canonical artifacts for Gate A and Gate B.

- [ ] **Step 1: Write failing runner matrix and ordering tests**

Add:

```python
def test_remote_runner_builds_fixed_split_smoke_for_both_gates():
    same_policy, compatibility = runner.build_smoke_cases()
    assert same_policy
    assert compatibility
    assert all(case.repetition == 0 for case in same_policy)
    assert all(case.repetition == 0 for case in compatibility)


def test_runner_orders_each_reference_before_candidates():
    ordered = runner.ordered_canonical_cases()
    positions = {case.case_id: index for index, case in enumerate(ordered)}
    for case in contract.build_diagnostic_matrix():
        if case.mode == "candidate_eager":
            continue
        reference = runner.same_policy_reference_case(case)
        assert positions[reference.case_id] < positions[case.case_id]
    for case in contract.build_legacy_compatibility_matrix():
        if case.policy == "legacy_eager_auto":
            continue
        reference = runner.compatibility_reference_case(case)
        assert positions[reference.case_id] < positions[case.case_id]
```

- [ ] **Step 2: Write failing resume-identity tests**

Require resume rejection when any of these differ:

```text
split_policy_name
flash_attn_num_splits
comparison_policy_name
flash_attn_version
source hash
environment hash
artifact hash
```

- [ ] **Step 3: Run and verify RED**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the runner supports only the old diagnostic modes.

- [ ] **Step 4: Add explicit CLI modes**

Use:

```text
preflight
fixed-split-smoke
fixed-split-canonical
download-only
verify-only
```

`fixed-split-smoke` runs a non-authoritative subset from both gates.
`fixed-split-canonical` runs all `315` isolated processes:

```text
189 Gate A + 126 Gate B = 315
```

- [ ] **Step 5: Generalize reference token handling**

Reference producers:

```text
Gate A: candidate_eager
Gate B: legacy_eager_auto
```

Candidate cases copy their paired reference token file before launch. Resume
must validate the reference artifact SHA256.

- [ ] **Step 6: Merge policy-aware artifacts**

Manifest includes:

```python
{
    "kind": "fixed_split_recovery",
    "same_policy_case_ids": [...189 ids...],
    "legacy_compatibility_case_ids": [...126 ids...],
    "same_policy_process_count": 189,
    "compatibility_process_count": 126,
    "compatibility_pair_count": 63,
    "flash_attn_version": environment["flash_attention"],
    "fixed_split_count": 16,
    "auto_split_count": 0,
}
```

Producer summaries remain non-authoritative. The local independent verifier is
always run after a canonical download.

- [ ] **Step 7: Preserve transport safety**

Keep the existing:

```text
source snapshot hash validation
dynamic globally unique port allocation
EADDRINUSE-only retry
single remote shell argument
safe tar extraction
isolated /tmp/tllm-cuda-graph-* root
download-before-delete behavior
no remote checkout mutation
```

- [ ] **Step 8: Run the complete dependency-light suite**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_model_runner_spec_verify.py
python3 -m py_compile \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  tools/diagnose_multi_sequence_cuda_graph.py \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py
```

Expected: all PASS.

- [ ] **Step 9: Complete the mandatory unit-test workflow**

Run the workflow-required `utree flush` using the `TMP_ROOT` created at Task 1,
perform its artifact check, and run the mandatory end telemetry command with
status `success`. Do not proceed if the workflow reports missing artifacts or
failing tests.

- [ ] **Step 10: Commit orchestration**

```bash
git add -- \
  tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: orchestrate fixed-split recovery gates"
```

---

### Task 6: Run Fresh Remote Smoke and Audit Coverage

**Files:**
- Produce untracked artifacts: `experiments/cuda_graph/<run-tag>/`
- Modify only on result: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: non-authoritative smoke evidence proving the runner and verifier can exercise both policies.

- [ ] **Step 1: Verify the local source boundary**

Run:

```bash
git status --short
git log -1 --oneline
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_context_modes.py
python3 tools/test_model_runner_spec_verify.py
```

Expected:

- tests PASS;
- tracked worktree clean;
- unrelated `experiments/` remain untracked.

- [ ] **Step 2: Run source-bound remote preflight**

Run:

```bash
RUN_TAG="qwen3-06b-fixed-split-preflight-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  preflight \
  --run-tag "$RUN_TAG" \
  --verifier-python /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

Expected: remote tests and `py_compile` pass from the uploaded immutable source.

- [ ] **Step 3: Run a fresh mixed-gate smoke**

Run:

```bash
RUN_TAG="qwen3-06b-fixed-split-smoke-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  fixed-split-smoke \
  --run-tag "$RUN_TAG" \
  --verifier-python /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

Expected:

- candidate eager, exact graph, rounded graph, legacy auto eager, and fixed
  eager all execute;
- every process has unique dynamic ports;
- smoke writes `NON_AUTHORITATIVE_SMOKE`;
- no canonical `EXACT_REPLAY_CORRECT` or `LEGACY_COMPATIBLE` claim is emitted.

- [ ] **Step 4: Audit prompt-to-artifact smoke coverage**

Inspect:

```bash
jq '.' "experiments/cuda_graph/$RUN_TAG/manifest.json"
jq '.' "experiments/cuda_graph/$RUN_TAG/independent-verification-smoke.json"
python3 - <<'PY'
import json
from pathlib import Path
root = Path("experiments/cuda_graph") / Path(__import__("os").environ["RUN_TAG"])
rows = [
    json.loads(line)
    for line in (root / "process_rows.jsonl").read_text().splitlines()
]
assert rows
assert {row["flash_attn_num_splits"] for row in rows} == {0, 16}
assert {row["split_policy_name"] for row in rows} == {"auto", "fixed16"}
ports = [
    port
    for row in rows
    for port in (row["tinyvllm_dist_port"], row["master_port"])
]
assert len(ports) == len(set(ports))
print("FIXED_SPLIT_SMOKE_AUDIT_OK", len(rows))
PY
```

- [ ] **Step 5: Fix only implementation/evidence defects**

If smoke fails, use `systematic-debugging`; do not change matrix, prompts,
split counts, repetitions, tolerance, or thresholds. Any source change requires
a fresh run tag and a fresh smoke.

- [ ] **Step 6: Commit smoke-only fixes if any**

Use exact-path staging and a focused commit. Do not stage artifacts.

---

### Task 7: Run Gate A and Gate B Canonical Checkpoint

**Files:**
- Produce untracked artifacts: `experiments/cuda_graph/<canonical-run-tag>/`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: a clean source that passed Task 6.
- Produces: authoritative independent Gate A/B classifications.

- [ ] **Step 1: Freeze source and launch all 315 processes**

Run:

```bash
RUN_TAG="qwen3-06b-fixed-split-canonical-$(date +%Y%m%d-%H%M%S)"
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  fixed-split-canonical \
  --run-tag "$RUN_TAG" \
  --verifier-python /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

If transport is interrupted, resume only the same source-bound run:

```bash
python3 tools/run_multi_sequence_cuda_graph_diagnostic_remote.py \
  fixed-split-canonical \
  --run-tag "$RUN_TAG" \
  --resume \
  --verifier-python /Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python
```

- [ ] **Step 2: Independently verify after download**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/verify_multi_sequence_cuda_graph_diagnostic.py \
  --run-dir "experiments/cuda_graph/$RUN_TAG"
```

Expected success checkpoint:

```text
classification=EXACT_REPLAY_CORRECT
legacy_compatibility=LEGACY_COMPATIBLE
same_policy_case_count=189
compatibility_process_count=126
compatibility_pair_count=63
```

- [ ] **Step 3: Perform the canonical completeness audit**

Verify:

```text
315 unique case IDs
630 unique dynamic ports
189 same-policy rows
126 compatibility process rows
63 complete compatibility pairs
7 batch sizes
3 trajectories
3 repetitions
all required layer shards
all required KV shards
all raw JSONL rows
all source/environment/prompt hashes
no mixed or missing split identities
fixed graph compared only with fixed eager
fixed eager compatibility compared only with auto eager
```

- [ ] **Step 4: Record the result honestly**

Append to `AGENT_HANDOFF_STATE.md`:

- source commit/tree SHA256;
- remote environment and FlashAttention version;
- exact commands and run tag;
- Gate A, rounded, and Gate B classifications;
- first divergence if any;
- what the result proves and does not prove;
- whether production Tasks 8-10 are authorized.

- [ ] **Step 5: Commit only the handoff**

```bash
git add -- AGENT_HANDOFF_STATE.md
git commit -m "docs: record fixed-split cuda graph gates"
```

### Hard Checkpoint

If either condition is false:

```text
Gate A == EXACT_REPLAY_CORRECT
Gate B == LEGACY_COMPATIBLE
```

stop here. Do not modify `tinyvllm/config.py`, production graph capture,
production dispatch, arrival-load evidence, or README.

---

### Task 8: Add the Default-Off Exact-Key Production Candidate

**Conditional:** Execute only after Task 7 authorizes production work.

**Files:**
- Modify: `tinyvllm/config.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: independently admitted fixed-16 exact replay.
- Produces: default-off exact-key production dispatch with auto eager fallback.

- [ ] **Step 1: Start a fresh mandatory unit-test workflow**

Invoke `bits-unit-test-gen` for:

```text
tinyvllm/config.py
tinyvllm/engine/model_runner.py
tools/test_model_runner_spec_verify.py
```

Follow all telemetry/bootstrap/`utree flush` requirements.

- [ ] **Step 2: Write failing configuration tests**

Add:

```python
def test_multi_sequence_cuda_graph_exact_defaults_off():
    config = make_config()
    assert config.multi_sequence_cuda_graph_exact is False


def test_multi_sequence_cuda_graph_exact_rejects_incompatible_modes():
    incompatible = (
        {"kv_quant_bits": 4},
        {"cpu_offload": True},
        {"kv_offload_mvp0": True},
        {"quest_top_k_blocks": 8},
        {"am_compact_blocks": 8},
    )
    for overrides in incompatible:
        with assert_raises(AssertionError):
            make_config(
                multi_sequence_cuda_graph_exact=True,
                **overrides,
            )
```

- [ ] **Step 3: Implement the default-off config**

Add:

```python
multi_sequence_cuda_graph_exact: bool = False
```

Validate incompatible combinations in `__post_init__`.

- [ ] **Step 4: Write failing graph identity and dispatch tests**

Add tests proving:

```python
def test_exact_multi_sequence_graph_requires_fixed16_identity():
    runner = fake_runner(
        enabled=True,
        graphs={2: FakeGraph()},
        metadata={
            2: {
                "split_policy_name": "fixed16",
                "flash_attn_num_splits": 16,
            }
        },
    )
    assert runner._decode_graph_key(mode="decode", batch_size=2) == 2


def test_auto_or_missing_identity_fails_closed_to_eager():
    for metadata in ({}, {2: {"split_policy_name": "auto",
                              "flash_attn_num_splits": 0}}):
        runner = fake_runner(enabled=True, graphs={2: FakeGraph()},
                             metadata=metadata)
        assert runner._decode_graph_key(
            mode="decode",
            batch_size=2,
        ) is None


def test_non_exact_batch_never_rounds_up():
    runner = fake_runner(enabled=True, graphs={4: FakeGraph()})
    assert runner._decode_graph_key(mode="decode", batch_size=3) is None
```

- [ ] **Step 5: Implement graph metadata and exact-key selection**

Add:

```python
def _fixed16_graph_identity(self, batch_size: int) -> bool:
    metadata = self.graph_metadata.get(batch_size)
    return metadata == {
        "graph_batch_size": batch_size,
        "split_policy_name": "fixed16",
        "flash_attn_num_splits": 16,
    }


def _decode_graph_key(self, *, mode: str, batch_size: int) -> int | None:
    if mode != "decode":
        return None
    if batch_size == 1:
        return 1 if 1 in self.graphs else None
    if not self.config.multi_sequence_cuda_graph_exact:
        return None
    if batch_size not in self.graphs:
        return None
    if not self._fixed16_graph_identity(batch_size):
        return None
    return batch_size
```

- [ ] **Step 6: Capture batch one unchanged and multi-sequence fixed16**

During `capture_cudagraph()`:

```python
split_count = 0 if bs == 1 else 16
with temporary_flash_attn_num_splits(split_count):
    # warmup and capture
```

Record `self.graph_metadata[bs]`. Do not capture multi-sequence candidate graphs
when the config is false.

- [ ] **Step 7: Refine dispatch without removing fail-closed guards**

Compute `graph_key` only after all existing incompatibility checks. Eager
fallback remains auto split 0. Exact graph replay uses fixed-16 scope and emits:

```python
self.last_execution_observation = {
    "execution_path": "graph_exact_fixed16",
    "active_batch_size": bs,
    "graph_batch_size": graph_key,
    "split_policy_name": "fixed16",
    "flash_attn_num_splits": 16,
}
```

Eager emits auto/0; batch-one graph emits its unchanged identity.

- [ ] **Step 8: Run focused tests and mandatory flush**

Run:

```bash
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_context_modes.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 -m py_compile tinyvllm/config.py tinyvllm/engine/model_runner.py
```

Complete `utree flush`, artifact check, and end telemetry.

- [ ] **Step 9: Commit the candidate**

```bash
git add -- \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tools/test_model_runner_spec_verify.py
git commit -m "feat: add exact fixed-split cuda graph candidate"
```

---

### Task 9: Add Production Execution Evidence and Frozen Batching Gate

**Conditional:** Execute only after Task 8.

**Files:**
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/arrival_load_driver.py`
- Modify: `tools/test_arrival_load_driver.py`
- Create: `tools/multi_sequence_cuda_graph_batching_gate.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: `ModelRunner.last_execution_observation`.
- Produces: source-bound paired production rows and `GO/NO_GO/INCOMPLETE`.

- [ ] **Step 1: Write failing engine evidence propagation tests**

Require `last_step_observation` to include:

```text
execution_path
active_batch_size
graph_batch_size
split_policy_name
flash_attn_num_splits
```

- [ ] **Step 2: Propagate model-runner evidence**

After `model_runner.call("run", ...)`, fetch a copy of the runner observation
and merge it into `LLMEngine.last_step_observation`. Missing evidence is a
driver error for the production candidate.

- [ ] **Step 3: Write failing fixed workload and threshold tests**

The gate module must build exactly:

```text
policies: EAGER_BASELINE, EXACT_GRAPH_FIXED16_CANDIDATE
workloads: stable-exact, ragged-natural, churn
warmup repetitions: 1
measured repetitions: 5
```

Reuse `contract.classify_production_gate()` and preserve all frozen thresholds.

- [ ] **Step 4: Implement execution validation**

Candidate rules:

```text
graph_exact_fixed16 only when active_batch_size == graph_batch_size > 1
graph identity must be fixed16/16
batch 3, 5, 9 events in churn must be eager auto/0
no rounded graph event
baseline is always eager auto/0 for batch >1
```

- [ ] **Step 5: Implement paired output and lifecycle correctness**

Each candidate run must match paired baseline:

```text
ordered generated token arrays
request count
finished request IDs
terminal lifecycle states
no non-finite/runtime errors
no allocator/block invariant failure
```

- [ ] **Step 6: Run dependency-light tests**

Run:

```bash
python3 tools/test_arrival_load_driver.py
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_model_runner_spec_verify.py
python3 -m py_compile \
  tinyvllm/engine/llm_engine.py \
  tools/arrival_load_driver.py \
  tools/multi_sequence_cuda_graph_batching_gate.py
```

Expected: PASS.

- [ ] **Step 7: Commit the production gate**

```bash
git add -- \
  tinyvllm/engine/llm_engine.py \
  tools/arrival_load_driver.py \
  tools/test_arrival_load_driver.py \
  tools/multi_sequence_cuda_graph_batching_gate.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: gate fixed-split cuda graph batching"
```

---

### Task 10: Run Production Smoke, Canonical Gate, and Claim Audit

**Conditional:** Execute only after Tasks 8-9.

**Files:**
- Produce untracked artifacts: `experiments/cuda_graph/<production-run-tag>/`
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `README.md` only on independent `GO`.

**Interfaces:**
- Consumes: source-bound default-off candidate and production gate.
- Produces: final `GO`, `NO_GO`, or `INCOMPLETE`.

- [ ] **Step 1: Run a fresh production smoke**

Use a timestamped run tag and the fixed model/host/ports contract. Smoke must
exercise:

```text
stable exact graph hits
natural exact and non-exact transitions
churn batches 3, 5, 9 using eager auto/0
exact batches using graph fixed16/16
```

Smoke is not performance evidence.

- [ ] **Step 2: Fix only implementation/evidence defects**

Do not change workload, policy order, repetition count, thresholds, or
correctness rules after observing smoke.

- [ ] **Step 3: Run fresh production canonical**

Run all policy/workload pairs with one warmup and five measured repetitions,
alternating policy order and using fresh processes/dynamic ports.

- [ ] **Step 4: Independently reconstruct the decision**

Require all correctness gates and:

```text
aggregate decode throughput ratio >= 1.15
stable-exact decode throughput ratio >= 1.25
each request-throughput ratio >= 0.95
each p95 ITL ratio <= 1.05
each p99 ITL ratio <= 1.10
peak reserved memory ratio <= 1.02
initialization ratio <= 1.05
stable exact graph hit rate >= 0.60
all measured repetitions complete
```

- [ ] **Step 5: Perform prompt-to-artifact completion audit**

Map every approved spec requirement to:

```text
source commit/tree hash
manifest field
raw row/tensor artifact
independent verifier rule
observed classification
handoff/README claim
```

Treat any missing evidence as `INCOMPLETE`.

- [ ] **Step 6: Record result boundaries**

Always update `AGENT_HANDOFF_STATE.md`.

Update `README.md` only if independent classification is `GO`, and limit the
claim to the recorded Qwen3-0.6B BF16 TP=1 greedy FlashAttention 2.6.3 exact-key
workload.

- [ ] **Step 7: Run final verification**

Run:

```bash
python3 tools/test_multi_sequence_cuda_graph_gate.py
python3 tools/test_context_modes.py
python3 tools/test_model_runner_spec_verify.py
python3 tools/test_arrival_load_driver.py
git diff --check
git status --short
```

Expected: all tests PASS; only intended tracked docs remain for commit; all
experiment artifacts remain untracked.

- [ ] **Step 8: Commit result documentation selectively**

For `GO`:

```bash
git add -- AGENT_HANDOFF_STATE.md README.md
git commit -m "docs: record fixed-split cuda graph go"
```

For `NO_GO` or `INCOMPLETE`:

```bash
git add -- AGENT_HANDOFF_STATE.md
git commit -m "docs: record fixed-split cuda graph result"
```

Do not push unless explicitly requested.

---

## Execution Stop Rules

Stop immediately and report the exact evidence if:

1. fixed-16 graph differs from fixed-16 eager in any Gate A exact case;
2. fixed-16 eager violates Gate B exact-token, logit-tolerance, or KV-ownership
   compatibility with auto eager;
3. a split identity is missing, mixed, or inconsistent;
4. the canonical matrix is not exactly 189 + 126 processes;
5. a required layer/KV/logit shard is absent or unhashed;
6. source, prompt, model, FlashAttention, GPU, driver, or threshold identity
   drifts;
7. any task would require weakening matrix, prompts, repetitions, tolerance,
   exact token equality, KV rules, or production thresholds;
8. remote execution would require modifying the remote checkout, killing
   unrelated processes, clearing shared `/tmp`, or reusing fixed ports;
9. `bits-unit-test-gen`, focused tests, `utree flush`, or telemetry cannot
   complete;
10. the user says to stop or pause.

## Plan Self-Review

- Spec coverage: policy identity, same-policy correctness, legacy
  compatibility, exact-key production dispatch, frozen performance gate,
  remote safety, TDD, artifacts, and claim boundaries all map to explicit
  tasks.
- Scope: Light Doc Cache/M8/token-sparse/low-rank work is intentionally
  excluded from this plan and requires a separate design/plan.
- Placeholder scan: no `TBD`, `TODO`, “implement later”, or unspecified test
  step remains.
- Type consistency: `DiagnosticCase`, `LegacyCompatibilityCase`,
  `temporary_flash_attn_num_splits()`, policy names, artifact fields, and
  classifications are consistent across tasks.
- Admission safety: production files are explicitly conditional on both Gate A
  and Gate B passing independently.
