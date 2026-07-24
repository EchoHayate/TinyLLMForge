# Qwen3.5 Hybrid-State Dtype-Aware Equivalence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This repository's approved execution mode is inline; do not dispatch subagents.

**Goal:** Replace the Qwen3.5 hybrid-state gate's BF16 full-vocabulary tolerance rejection with a schema-v2, output-behavior-first, dtype-aware equivalence gate while preserving exact repeatability, strict FP32 control, all state/lifecycle/accounting guards, and the existing claim boundary.

**Architecture:** Extend the frozen contract with schema-v2 dtype and decision records, make the reference probe emit independently derived actual/oracle top-k and FP32 comparison metrics, and make the independent verifier enforce deterministic repeats, greedy decision preservation, positive winner margins, strict FP32 control, and unchanged state semantics. Keep the production engine untouched, preserve schema-v1 evidence as `INCOMPLETE`, and run a new source-bound remote smoke before allowing canonical execution.

**Tech Stack:** Python 3, dataclasses, PyTorch, Transformers, JSON/JSONL, SHA-256, dependency-light script tests, SSH ControlMaster, NVIDIA A100 remote execution.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`; never modify `/Users/bytedance/dev/TinyLLMForge`.
- Execute inline in the current session; do not dispatch subagents.
- The approved design is `docs/superpowers/specs/2026-07-24-qwen35-hybrid-state-dtype-aware-equivalence-gate-design.md`.
- Preserve the original compatibility design and implementation plan as historical schema-v1 documents.
- Preserve the modified `AGENT_HANDOFF_STATE.md` and every unrelated untracked `experiments/` directory.
- Preserve `experiments/qwen35_hybrid_state/qwen35-2b-hybrid-smoke-20260723-225302/` as immutable schema-v1 evidence classified `INCOMPLETE`.
- Stage exact paths only; never use `git add -A`.
- Do not modify `tinyvllm/`, `README.md`, production model loading, scheduler code, cache allocation, block tables, or production inference behavior.
- Do not combine this work with compression, token sparsity, low rank, Light Doc Cache, Attention Matching, KV quantization, KV offload, speculative decoding, CUDA Graphs, or sparse-kernel benchmarking.
- Do not claim native Qwen3.5 support, quality retention, compression, latency, throughput, speedup, or physical GPU-memory reduction.
- Use only `Qwen/Qwen3.5-2B` at immutable revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Schema-v2 BF16 acceptance is limited to deterministic greedy continuation in the frozen matrix.
- Same-path repeated cached logits must remain bitwise identical.
- Every cross-path comparison must preserve the greedy token and strictly positive actual and oracle winner margins.
- Use `FP32_ATOL = 2e-5`, `FP32_RTOL = 1e-5`, and `FP32_MEAN_ABS_CAP = 3e-6`; do not tune these values against the new smoke.
- BF16 full-vocabulary drift statistics are mandatory evidence but are not acceptance caps.
- A schema-v1 bundle or a schema-v2 bundle missing decision metrics or dtype identity is `INCOMPLETE`.
- A complete canonical semantic mismatch may be `NO_GO`; the same mismatch in smoke remains `INCOMPLETE`.
- Canonical execution is forbidden until a schema-v2 smoke independently classifies `SMOKE_PASS`.
- GPU/model work runs only on `sitian@10.232.195.203` as user `sitian`.
- Use SSH ControlMaster `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use only `CUDA_VISIBLE_DEVICES=0`.
- Give every model process fresh, mutually distinct `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Retry only when stderr contains exact `EADDRINUSE`, at most three attempts, with a fresh port pair per attempt.
- Do not use `rsync`, modify the remote checkout, run `kill` or `pkill`, switch GPU, clean shared caches or `/tmp`, or delete evidence.
- Source staging must use the existing tar-over-SSH clean-commit flow and verify local and remote hashes.

---

## File Map

- Modify `tools/qwen35_hybrid_state_contract.py`: schema-v2 constants, exact row fields, dtype/comparison policies, FP32 control case, top-k and elementwise helper functions, and schema-version classification support.
- Modify `tools/test_qwen35_hybrid_state_contract.py`: frozen schema, helper boundaries, FP32 control-domain, and schema-v1 fail-closed tests.
- Modify `tools/qwen35_hybrid_state_probe.py`: actual/oracle decision records, percentile/cosine/allclose metrics, dtype profiles, FP32 control execution, and schema-v2 artifact output.
- Modify `tools/test_qwen35_hybrid_state_probe.py`: synthetic decision metrics, tie handling, dtype reconstruction, and schema-v2 output tests.
- Modify `tools/verify_qwen35_hybrid_state_gate.py`: schema-version gate, exact metric validation, same-path determinism, BF16 decision checks, FP32 strict checks, smoke/canonical classification, and report output.
- Modify `tools/test_verify_qwen35_hybrid_state_gate.py`: complete schema-v2 fixture and tamper matrix, including preserved lifecycle/accounting coverage.
- Modify `tools/run_qwen35_hybrid_state_gate_remote.py`: schema-v2 smoke requirements, FP32 control invocation, immutable smoke admission record, and canonical admission check.
- Modify `tools/test_run_qwen35_hybrid_state_gate_remote.py`: source inventory, smoke command, admission identity, retry, and canonical-blocking tests.
- Create new raw evidence under `experiments/qwen35_hybrid_state/{new-run-id}/`; never stage raw run directories.
- Modify `docs/qwen35_hybrid_state_evidence_registry.json` only if a new canonical run exists and independently verifies.
- Modify `AGENT_HANDOFF_STATE.md` only after the final authoritative result is known.

## Shared Interfaces

Freeze these constants in `tools/qwen35_hybrid_state_contract.py`:

```python
SCHEMA_VERSION = 2
DECISION_TOPK = 20
FP32_ATOL = 2e-5
FP32_RTOL = 1e-5
FP32_MEAN_ABS_CAP = 3e-6
EXECUTION_DTYPES = ("bfloat16", "float32", "metadata_only")
COMPARISON_POLICIES = (
    "bf16_decision_preserving",
    "fp32_elementwise",
    "none",
)
ABS_DIFF_PERCENTILE_FIELDS = ("p50", "p95", "p99", "p99_9")
FP32_CONTROL_CASE_ID = (
    "fp32_path_control__cached_vs_one_shot__p17__r0__c17"
)
```

Add these exact helpers:

```python
def validate_ranked_topk(
    token_ids: list[int],
    logits: list[float],
    *,
    expected_count: int = DECISION_TOPK,
) -> None:


def winner_margin(
    token_ids: list[int],
    logits: list[float],
) -> dict[str, int | float]:
```

Every schema-v2 case row contains the original fields plus:

```python
"execution_dtype"
"comparison_policy"
```

Every schema-v2 logit record contains the original fields plus:

```python
"actual_topk_token_ids"
"actual_topk_logits"
"oracle_topk_token_ids"
"oracle_topk_logits"
"topk_intersection_size"
"oracle_topk_recall"
"actual_winner_token_id"
"oracle_winner_token_id"
"actual_runner_up_token_id"
"oracle_runner_up_token_id"
"actual_winner_logit"
"oracle_winner_logit"
"actual_runner_up_logit"
"oracle_runner_up_logit"
"actual_winner_margin"
"oracle_winner_margin"
"winner_logit_abs_diff"
"runner_up_logit_abs_diff"
"winner_margin_abs_diff"
"abs_diff_percentiles"
"cosine_similarity"
"allclose_violation_count"
"max_allclose_scaled_error"
```

The probe's internal comparison helper becomes:

```python
def _logit_record(
    value,
    *,
    oracle_logits,
    request_id,
    request_generation,
    step_index,
    sequence_length,
    position_metadata,
    comparison_policy,
):
    ...
```

The verifier exposes:

```python
def _verify_decision_record(row, record) -> None:
    ...


def _verify_same_path_repeatability(rows_by_phase) -> None:
    ...


def _verify_fp32_control(rows_by_phase) -> dict[str, float | int]:
    ...


def _verify_bf16_behavior(rows_by_phase) -> dict[str, float | int]:
    ...
```

---

### Task 1: Freeze Schema V2 and Numerical Helper Contracts

**Files:**
- Modify: `tools/qwen35_hybrid_state_contract.py`
- Modify: `tools/test_qwen35_hybrid_state_contract.py`

**Interfaces:**
- Consumes: the approved schema-v2 design and existing `GateCase`, `build_case_matrix()`, canonical JSON, byte-accounting, and classification helpers.
- Produces: schema-v2 constants, exact field tuples, FP32 control case, `validate_ranked_topk()`, `winner_margin()`, and FP32 limit constants used by the probe and verifier.

- [ ] **Step 1: Add failing schema-v2 constant and field tests**

Append these exact tests:

```python
def test_schema_v2_freezes_dtype_and_decision_fields():
    assert contract.SCHEMA_VERSION == 2
    assert contract.DECISION_TOPK == 20
    assert contract.FP32_ATOL == 2e-5
    assert contract.FP32_RTOL == 1e-5
    assert contract.FP32_MEAN_ABS_CAP == 3e-6
    assert contract.EXECUTION_DTYPES == (
        "bfloat16",
        "float32",
        "metadata_only",
    )
    assert "execution_dtype" in contract.CASE_ROW_FIELDS
    assert "comparison_policy" in contract.CASE_ROW_FIELDS
    for field in (
        "actual_topk_token_ids",
        "oracle_topk_token_ids",
        "actual_winner_margin",
        "oracle_winner_margin",
        "abs_diff_percentiles",
        "allclose_violation_count",
    ):
        assert field in contract.LOGIT_RECORD_FIELDS


def test_fp32_control_case_is_frozen():
    cases = contract.build_case_matrix()
    control = [
        case
        for case in cases
        if case.case_id == contract.FP32_CONTROL_CASE_ID
    ]
    assert len(control) == 1
    assert control[0].phase == "fp32_path_control"
    assert control[0].execution_mode == "cached_vs_one_shot"
    assert control[0].prompt_length == 17
    assert control[0].execution_dtype == "float32"
    assert control[0].comparison_policy == "fp32_elementwise"
```

- [ ] **Step 2: Run the contract test and verify the new assertions fail**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_contract.py
```

Expected: non-zero exit with the first missing schema-v2 constant or field.

- [ ] **Step 3: Add schema-v2 constants and extend `GateCase`**

Implement:

```python
SCHEMA_VERSION = 2
DECISION_TOPK = 20
FP32_ATOL = 2e-5
FP32_RTOL = 1e-5
FP32_MEAN_ABS_CAP = 3e-6
EXECUTION_DTYPES = ("bfloat16", "float32", "metadata_only")
COMPARISON_POLICIES = (
    "bf16_decision_preserving",
    "fp32_elementwise",
    "none",
)
ABS_DIFF_PERCENTILE_FIELDS = ("p50", "p95", "p99", "p99_9")
FP32_CONTROL_CASE_ID = (
    "fp32_path_control__cached_vs_one_shot__p17__r0__c17"
)
```

Add `execution_dtype: str` and `comparison_policy: str` to `GateCase`, validate
them in `__post_init__()`, assign `metadata_only/none` to non-model rows,
`bfloat16/bf16_decision_preserving` to the existing model matrix, and append
the frozen FP32 control case.

- [ ] **Step 4: Add failing top-k and margin helper tests**

Add:

```python
def test_ranked_topk_and_winner_margin():
    token_ids = list(range(20))
    logits = [float(20 - index) for index in range(20)]
    contract.validate_ranked_topk(token_ids, logits)
    result = contract.winner_margin(token_ids, logits)
    assert result == {
        "winner_token_id": 0,
        "runner_up_token_id": 1,
        "winner_logit": 20.0,
        "runner_up_logit": 19.0,
        "winner_margin": 1.0,
    }


def test_ranked_topk_rejects_duplicates_unsorted_and_ties():
    token_ids = list(range(20))
    logits = [float(20 - index) for index in range(20)]
    for bad_ids, bad_logits in (
        ([0] + token_ids[:-1], logits),
        (token_ids, [19.0, 20.0] + logits[2:]),
        (token_ids, [20.0, 20.0] + logits[2:]),
    ):
        try:
            contract.validate_ranked_topk(bad_ids, bad_logits)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid top-k was accepted")
```

- [ ] **Step 5: Implement exact top-k and margin validation**

Implement `validate_ranked_topk()` to require:

- exact `expected_count`;
- integer, non-boolean token IDs;
- unique token IDs;
- finite numeric logits;
- non-increasing logits;
- strict `logits[0] > logits[1]`.

Implement `winner_margin()` by first calling `validate_ranked_topk()` and then
returning the exact dictionary asserted above.

- [ ] **Step 6: Replace old tolerance-derivation tests with frozen FP32 limits**

Remove tests that expect BF16 acceptance to be derived from
`derive_logit_tolerance()`. Keep the helper only if schema-v1 reading needs it,
but ensure schema-v2 code cannot call it. Add:

```python
def test_fp32_limits_are_not_derived_from_bf16_rows():
    assert contract.FP32_ATOL == 2e-5
    assert contract.FP32_RTOL == 1e-5
    assert contract.FP32_MEAN_ABS_CAP == 3e-6
    assert not hasattr(contract, "BF16_MAX_LOGIT_ATOL")
    assert not hasattr(contract, "BF16_MAX_LOGIT_RTOL")
```

- [ ] **Step 7: Run contract tests**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_contract.py
python3 -m py_compile \
  tools/qwen35_hybrid_state_contract.py \
  tools/test_qwen35_hybrid_state_contract.py
```

Expected: both commands exit 0 and print the existing contract success line.

- [ ] **Step 8: Commit Task 1**

Run:

```bash
git add \
  tools/qwen35_hybrid_state_contract.py \
  tools/test_qwen35_hybrid_state_contract.py
git diff --cached --check
git commit -m "feat: freeze qwen35 dtype-aware gate contract"
```

Expected: one commit containing only the two exact files.

---

### Task 2: Emit Decision-Preserving and FP32 Control Evidence

**Files:**
- Modify: `tools/qwen35_hybrid_state_probe.py`
- Modify: `tools/test_qwen35_hybrid_state_probe.py`

**Interfaces:**
- Consumes: Task 1 schema constants, `validate_ranked_topk()`, `winner_margin()`, FP32 constants, and existing reference execution/state snapshot helpers.
- Produces: schema-v2 case rows, dual-path top-k records, decision metrics, BF16 drift summaries, FP32 allclose summaries, dtype profiles, and the frozen FP32 control case.

- [ ] **Step 1: Add failing synthetic dual-path metric tests**

Add a test using CPU tensors:

```python
def test_logit_record_contains_independent_decision_evidence():
    actual = torch.linspace(-2.0, 2.0, 32, dtype=torch.float32)
    oracle = actual.clone()
    actual[7] = 4.0
    actual[9] = 3.0
    oracle[7] = 3.75
    oracle[9] = 2.75
    record = probe._logit_record(
        actual,
        oracle_logits=oracle,
        request_id="request-0",
        request_generation=0,
        step_index=1,
        sequence_length=18,
        position_metadata={},
        comparison_policy="bf16_decision_preserving",
    )
    assert record["actual_winner_token_id"] == 7
    assert record["oracle_winner_token_id"] == 7
    assert record["actual_runner_up_token_id"] == 9
    assert record["oracle_runner_up_token_id"] == 9
    assert record["actual_winner_margin"] == 1.0
    assert record["oracle_winner_margin"] == 1.0
    assert record["topk_token_ids"] == record["actual_topk_token_ids"]
    assert record["topk_logits"] == record["actual_topk_logits"]
    assert set(record["abs_diff_percentiles"]) == {
        "p50",
        "p95",
        "p99",
        "p99_9",
    }
```

- [ ] **Step 2: Run the focused probe test and verify failure**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_probe.py
```

Expected: non-zero exit because `_logit_record()` does not accept
`comparison_policy` or does not emit schema-v2 metrics.

- [ ] **Step 3: Implement detached FP32 comparison summaries**

Refactor `_logit_record()` so it:

```python
actual_fp32 = value.detach().float().reshape(-1)
oracle_fp32 = oracle_logits.detach().float().reshape(-1)
absolute = (actual_fp32 - oracle_fp32).abs()
denominator = oracle_fp32.abs().clamp_min(torch.finfo(torch.float32).tiny)
relative = absolute / denominator
```

Compute actual and oracle top-20 independently with stable descending
`torch.topk()`. Compute:

```python
threshold = contract.FP32_ATOL + contract.FP32_RTOL * oracle_fp32.abs()
scaled = absolute / threshold.clamp_min(torch.finfo(torch.float32).tiny)
allclose_violation_count = int((absolute > threshold).sum().item())
max_allclose_scaled_error = float(scaled.max().item())
```

Use `torch.quantile()` for p50, p95, p99, and p99.9, and
`torch.nn.functional.cosine_similarity()` for cosine similarity. Convert every
persisted metric to a finite Python scalar.

- [ ] **Step 4: Add failing tie, percentile, and FP32 boundary tests**

Add tests that:

```python
def test_logit_record_rejects_winner_tie():
    actual = torch.zeros(32)
    oracle = torch.zeros(32)
    actual[3] = actual[4] = 2.0
    oracle[3] = 2.0
    oracle[4] = 1.0
    with expect_value_error("winner"):
        probe._logit_record(
            actual,
            oracle_logits=oracle,
            request_id="request-0",
            request_generation=0,
            step_index=0,
            sequence_length=17,
            position_metadata={},
            comparison_policy="bf16_decision_preserving",
        )


def test_fp32_summary_counts_only_values_outside_frozen_allclose():
    oracle = torch.tensor([1.0, 0.0], dtype=torch.float32)
    inside = torch.tensor(
        [1.0 + contract.FP32_ATOL, contract.FP32_ATOL],
        dtype=torch.float32,
    )
    outside = inside.clone()
    outside[1] += 1e-6
    inside_record = make_fp32_record(inside, oracle)
    outside_record = make_fp32_record(outside, oracle)
    assert inside_record["allclose_violation_count"] == 0
    assert outside_record["allclose_violation_count"] == 1
```

Use the test file's existing exception helper style; do not introduce pytest.

- [ ] **Step 5: Add dtype profile collection**

Create:

```python
def _dtype_profile(model, state_components, logits):
    parameter_dtypes = [
        _normalized_dtype(parameter.dtype)
        for parameter in model.parameters()
    ]
    return {
        "requested_model_dtype": _normalized_dtype(model.dtype),
        "dominant_parameter_dtype": _mode(parameter_dtypes),
        "logit_dtype_before_comparison": _normalized_dtype(logits.dtype),
        "comparison_accumulator_dtype": "float32",
        "recurrent_state_dtypes": sorted({
            row["dtype"]
            for row in state_components
            if row["state_role"] in {
                "linear_recurrent_state",
                "linear_convolution_state",
            }
        }),
        "kv_state_dtypes": sorted({
            row["dtype"]
            for row in state_components
            if row["state_role"] in {
                "full_attention_key",
                "full_attention_value",
            }
        }),
    }
```

Write the same normalized profile to `model_manifest.json` and
`environment.json`. Reject an empty parameter inventory or missing recurrent
or KV dtype set.

- [ ] **Step 6: Add the FP32 control execution case**

Add a case dispatcher for phase `fp32_path_control` that:

- unloads the BF16 model and calls `torch.cuda.empty_cache()` only for
  process-local tensors;
- loads the same immutable model snapshot with requested dtype `torch.float32`;
- runs the frozen 17-token one-shot and cached comparison for eight steps;
- emits `execution_dtype="float32"` and
  `comparison_policy="fp32_elementwise"`;
- records state and memory snapshots using the existing lifecycle machinery;
- never mutates or deletes model artifacts;
- exits non-zero on worker failure without assigning the authoritative
  classification.

- [ ] **Step 7: Update every case row to exact schema-v2 fields**

Ensure metadata-only rows use:

```python
"execution_dtype": "metadata_only",
"comparison_policy": "none",
```

Ensure BF16 model rows use:

```python
"execution_dtype": "bfloat16",
"comparison_policy": "bf16_decision_preserving",
```

Ensure every row satisfies `set(row) == set(contract.CASE_ROW_FIELDS)` and
every record satisfies `set(record) == set(contract.LOGIT_RECORD_FIELDS)`.

- [ ] **Step 8: Run probe tests**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_probe.py
python3 -m py_compile \
  tools/qwen35_hybrid_state_probe.py \
  tools/test_qwen35_hybrid_state_probe.py
```

Expected: both commands exit 0 and the probe test prints its existing success
line.

- [ ] **Step 9: Commit Task 2**

Run:

```bash
git add \
  tools/qwen35_hybrid_state_probe.py \
  tools/test_qwen35_hybrid_state_probe.py
git diff --cached --check
git commit -m "feat: emit qwen35 decision equivalence evidence"
```

Expected: one commit containing only the probe and probe-test files.

---

### Task 3: Enforce Dtype-Aware Semantics in the Independent Verifier

**Files:**
- Modify: `tools/verify_qwen35_hybrid_state_gate.py`
- Modify: `tools/test_verify_qwen35_hybrid_state_gate.py`

**Interfaces:**
- Consumes: schema-v2 rows and records from Tasks 1-2, existing provenance/domain/state/lifecycle/storage verifiers, and smoke/canonical domain selection.
- Produces: authoritative schema-v2 `SMOKE_PASS | GO | NO_GO | INCOMPLETE`, numerical summary fields, tamper-resistant reports, and fail-closed schema-v1 handling.

- [ ] **Step 1: Convert the complete synthetic fixture to schema v2**

Update `_logit_record()` in the verifier test fixture to emit:

```python
actual_ids = list(range(20))
actual_logits = [float(20 - index) for index in range(20)]
oracle_ids = list(actual_ids)
oracle_logits = list(actual_logits)
```

Populate all schema-v2 fields with:

```python
"actual_winner_token_id": 0,
"oracle_winner_token_id": 0,
"actual_runner_up_token_id": 1,
"oracle_runner_up_token_id": 1,
"actual_winner_logit": 20.0,
"oracle_winner_logit": 20.0,
"actual_runner_up_logit": 19.0,
"oracle_runner_up_logit": 19.0,
"actual_winner_margin": 1.0,
"oracle_winner_margin": 1.0,
"winner_logit_abs_diff": 0.0,
"runner_up_logit_abs_diff": 0.0,
"winner_margin_abs_diff": 0.0,
"abs_diff_percentiles": {
    "p50": 0.0,
    "p95": 0.0,
    "p99": 0.0,
    "p99_9": 0.0,
},
"cosine_similarity": 1.0,
"allclose_violation_count": 0,
"max_allclose_scaled_error": 0.0,
```

Add the FP32 control row and dtype profiles to the fixture.

- [ ] **Step 2: Add failing schema-version and decision-tamper tests**

Add mutators for:

```python
def _downgrade_schema_version(run_dir):
    mutate_manifest_and_summary_schema(run_dir, 1)


def _remove_oracle_winner_from_actual_topk(run_dir):
    mutate_first_bf16_record(
        run_dir,
        lambda record: record["actual_topk_token_ids"].remove(
            record["oracle_winner_token_id"]
        ),
    )


def _tie_actual_winner(run_dir):
    def mutate(record):
        record["actual_topk_logits"][1] = record["actual_topk_logits"][0]
        record["topk_logits"][1] = record["topk_logits"][0]
        record["actual_winner_margin"] = 0.0
    mutate_first_bf16_record(run_dir, mutate)


def _break_fp32_allclose(run_dir):
    mutate_fp32_record(
        run_dir,
        lambda record: record.__setitem__(
            "allclose_violation_count",
            1,
        ),
    )
```

Assert schema downgrade, missing cross-top-k winner, and FP32 control failure
are `INCOMPLETE`. Assert a winner tie is `NO_GO` in canonical and
`INCOMPLETE` in smoke.

- [ ] **Step 3: Run verifier tests and verify the new cases fail**

Run:

```bash
python3 tools/test_verify_qwen35_hybrid_state_gate.py
```

Expected: non-zero exit at the first new schema-v2 or classification
assertion.

- [ ] **Step 4: Add exact schema and dtype-profile verification**

Before domain verification:

```python
if manifest.get("schema_version") != contract.SCHEMA_VERSION:
    _fail("schema-v2 evidence is required")
if summary.get("schema_version") != contract.SCHEMA_VERSION:
    _fail("summary schema version mismatch")
```

Require exact case and logit field sets. Validate case
`execution_dtype/comparison_policy` against the frozen `GateCase`. Reconstruct
the recurrent and KV dtype sets from state components and compare them with the
identical profiles in `model_manifest.json` and `environment.json`.

- [ ] **Step 5: Implement `_verify_decision_record()`**

The function must:

1. call `contract.validate_ranked_topk()` on actual and oracle lists;
2. require old top-k aliases to exactly equal actual top-k;
3. require actual and oracle winners to be each list's first token;
4. require runners-up to be each list's second token;
5. independently recompute each margin from persisted top-k logits;
6. require both margins to be finite and strictly positive;
7. require the oracle winner in actual top-20 and the actual winner in oracle
   top-20;
8. recompute intersection size and oracle recall;
9. require decoded token, actual greedy token, oracle greedy token, actual
   winner, and oracle winner to be identical;
10. validate exact percentile keys, finite metrics, and monotonic
    `p50 <= p95 <= p99 <= p99_9`;
11. validate non-negative absolute drift fields and
    `-1.0 <= cosine_similarity <= 1.0`.

Use `_semantic_fail()` only for a proved output decision violation. Use
`_fail()` for malformed, missing, inconsistent, or non-finite evidence.

- [ ] **Step 6: Split repeatability, BF16, and FP32 verification**

Implement:

```python
def _verify_same_path_repeatability(rows_by_phase):
    ...


def _verify_bf16_behavior(rows_by_phase):
    ...


def _verify_fp32_control(rows_by_phase):
    ...
```

Repeatability compares actual full-logit hashes, decoded tokens, request
identity, sequence/position metadata, state snapshot IDs, and dtype/policy
across repeats.

BF16 behavior calls `_verify_decision_record()` and aggregates drift only for
reporting. It must not compare `max_abs_diff` or `max_rel_diff` with a cap.

FP32 control additionally requires:

```python
record["allclose_violation_count"] == 0
record["mean_abs_diff"] <= contract.FP32_MEAN_ABS_CAP
record["max_allclose_scaled_error"] <= 1.0
```

Any FP32 control failure is `_fail()`, producing `INCOMPLETE`.

- [ ] **Step 7: Preserve smoke versus canonical classification**

In `verify_run()`:

- map `SemanticFailure` to `NO_GO` only when `domain == "canonical"` and the
  complete canonical domain was already reconstructed;
- map `SemanticFailure` to `INCOMPLETE` for `domain == "smoke"`;
- map every provenance, schema, dtype, repeatability, FP32, lifecycle,
  isolation, ledger, or audit failure to `INCOMPLETE`;
- emit `SMOKE_PASS` only for a fully verified smoke;
- emit `GO` only for a fully verified canonical bundle.

- [ ] **Step 8: Prove BF16 max-abs drift is diagnostic-only**

Add a tamper that changes a BF16 record to:

```python
record["max_abs_diff"] = 1000.0
record["mean_abs_diff"] = 10.0
record["max_rel_diff"] = 5000.0
record["mean_rel_diff"] = 50.0
record["abs_diff_percentiles"] = {
    "p50": 1.0,
    "p95": 5.0,
    "p99": 9.0,
    "p99_9": 10.0,
}
```

Keep all decision fields consistent. Assert the complete synthetic canonical
fixture still classifies `GO`. Add a separate non-finite metric tamper and
assert `INCOMPLETE`.

- [ ] **Step 9: Run verifier tests and compilation**

Run:

```bash
python3 tools/test_verify_qwen35_hybrid_state_gate.py
python3 -m py_compile \
  tools/verify_qwen35_hybrid_state_gate.py \
  tools/test_verify_qwen35_hybrid_state_gate.py
```

Expected: both commands exit 0. Existing provenance, domain, lifecycle,
request-isolation, and storage-ledger tamper tests remain green.

- [ ] **Step 10: Verify old smoke remains fail-closed**

Run:

```bash
python3 tools/verify_qwen35_hybrid_state_gate.py \
  --run-dir \
  experiments/qwen35_hybrid_state/qwen35-2b-hybrid-smoke-20260723-225302
```

Expected: classification `INCOMPLETE` with a reason containing
`schema-v2 evidence is required`; no file in the old smoke directory changes.

- [ ] **Step 11: Commit Task 3**

Run:

```bash
git add \
  tools/verify_qwen35_hybrid_state_gate.py \
  tools/test_verify_qwen35_hybrid_state_gate.py
git diff --cached --check
git commit -m "feat: verify qwen35 dtype-aware equivalence"
```

Expected: one commit containing only verifier and verifier-test changes.

---

### Task 4: Gate Remote Canonical Execution on Schema-V2 Smoke

**Files:**
- Modify: `tools/run_qwen35_hybrid_state_gate_remote.py`
- Modify: `tools/test_run_qwen35_hybrid_state_gate_remote.py`

**Interfaces:**
- Consumes: schema-v2 source files, probe CLI, verifier CLI, existing clean tar staging, process retry policy, artifact download, and smoke admission flow.
- Produces: a new schema-v2 smoke command, FP32 control inclusion, immutable admission identity, fail-closed canonical command, and preserved partial artifacts.

- [ ] **Step 1: Add failing source inventory and admission tests**

Extend static tests to require the staged source inventory to include all four
schema-v2 implementation files and tests. Add:

```python
def test_canonical_requires_schema_v2_smoke_pass():
    admission = {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": "SMOKE_PASS",
        "source_commit": "a" * 40,
        "contract_sha256": "b" * 64,
        "model_revision": (
            "15852e8c16360a2fea060d615a32b45270f8a8fc"
        ),
        "model_file_sha256": {"config.json": "c" * 64},
        "environment_identity_sha256": "d" * 64,
    }
    runner._validate_smoke_admission(
        admission,
        expected_source_commit="a" * 40,
        expected_contract_sha256="b" * 64,
        expected_model_revision=(
            "15852e8c16360a2fea060d615a32b45270f8a8fc"
        ),
        expected_model_file_sha256={"config.json": "c" * 64},
        expected_environment_identity_sha256="d" * 64,
    )
```

Add negative copies for schema 1, `INCOMPLETE`, source mismatch, contract
mismatch, model mismatch, file-hash mismatch, and environment mismatch.

- [ ] **Step 2: Run runner tests and verify failure**

Run:

```bash
python3 tools/test_run_qwen35_hybrid_state_gate_remote.py
```

Expected: non-zero exit because schema-v2 admission validation is absent.

- [ ] **Step 3: Implement immutable smoke admission validation**

Add:

```python
def _validate_smoke_admission(
    admission,
    *,
    expected_source_commit,
    expected_contract_sha256,
    expected_model_revision,
    expected_model_file_sha256,
    expected_environment_identity_sha256,
):
    ...
```

Require exact schema version, classification `SMOKE_PASS`, and exact identity
matches. Reject missing or additional model hash entries. The runner must read
the independent verifier result, not the worker summary, when creating the
admission record.

- [ ] **Step 4: Update smoke execution and expected row domain**

Ensure `smoke` launches one source-bound worker process that emits:

- environment preflight;
- architecture verification;
- two BF16 same-path repeats;
- BF16 one-shot versus cached;
- BF16 state export/import;
- FP32 path control;
- post-run audit.

Update smoke expected case count from 7 to 8 and make the runner fail closed if
the FP32 case or any schema-v2 field is absent.

- [ ] **Step 5: Preserve exact process safety rules**

Keep and test:

```text
host = sitian@10.232.195.203
python = /data00/home/sitian/sitian-workspace01/tllm/env/bin/python
CUDA_VISIBLE_DEVICES = 0
TINYVLLM_DIST_PORT != MASTER_PORT
retry_count <= 3
retry_condition == exact EADDRINUSE in stderr
```

Tests must continue to reject `rsync`, `kill`, `pkill`, GPU changes, remote
checkout edits, shared cleanup, and wildcard staging.

- [ ] **Step 6: Run runner tests and compilation**

Run:

```bash
python3 tools/test_run_qwen35_hybrid_state_gate_remote.py
python3 -m py_compile \
  tools/run_qwen35_hybrid_state_gate_remote.py \
  tools/test_run_qwen35_hybrid_state_gate_remote.py
```

Expected: both commands exit 0 and the runner test prints its success line.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add \
  tools/run_qwen35_hybrid_state_gate_remote.py \
  tools/test_run_qwen35_hybrid_state_gate_remote.py
git diff --cached --check
git commit -m "feat: gate qwen35 canonical on dtype-aware smoke"
```

Expected: one commit containing only runner and runner-test changes.

---

### Task 5: Run the Complete Local Regression and Source Audit

**Files:**
- Inspect: all eight modified `tools/` files
- Inspect: `docs/superpowers/specs/2026-07-24-qwen35-hybrid-state-dtype-aware-equivalence-gate-design.md`
- Inspect: `docs/superpowers/plans/2026-07-24-qwen35-hybrid-state-dtype-aware-equivalence-gate.md`
- Inspect: `experiments/qwen35_hybrid_state/qwen35-2b-hybrid-smoke-20260723-225302/`

**Interfaces:**
- Consumes: Tasks 1-4 commits.
- Produces: a clean, source-bound implementation commit range ready for remote staging and evidence that schema-v1 artifacts were not modified.

- [ ] **Step 1: Run all four dependency-light suites**

Run:

```bash
python3 tools/test_qwen35_hybrid_state_contract.py
python3 tools/test_qwen35_hybrid_state_probe.py
python3 tools/test_verify_qwen35_hybrid_state_gate.py
python3 tools/test_run_qwen35_hybrid_state_gate_remote.py
```

Expected: all four commands exit 0 and print their success lines.

- [ ] **Step 2: Compile all implementation and test files**

Run:

```bash
python3 -m py_compile \
  tools/qwen35_hybrid_state_contract.py \
  tools/qwen35_hybrid_state_probe.py \
  tools/verify_qwen35_hybrid_state_gate.py \
  tools/run_qwen35_hybrid_state_gate_remote.py \
  tools/test_qwen35_hybrid_state_contract.py \
  tools/test_qwen35_hybrid_state_probe.py \
  tools/test_verify_qwen35_hybrid_state_gate.py \
  tools/test_run_qwen35_hybrid_state_gate_remote.py
```

Expected: exit 0 with no output.

- [ ] **Step 3: Audit forbidden production changes**

Run:

```bash
git diff 3c98836..HEAD -- \
  tinyvllm \
  README.md
```

Expected: no output.

- [ ] **Step 4: Audit schema-v1 evidence immutability**

Run:

```bash
git status --short -- \
  experiments/qwen35_hybrid_state/qwen35-2b-hybrid-smoke-20260723-225302
find \
  experiments/qwen35_hybrid_state/qwen35-2b-hybrid-smoke-20260723-225302 \
  -type f -print0 | sort -z | xargs -0 shasum -a 256 \
  > /tmp/qwen35-schema-v1-post-implementation.sha256
```

Expected: status remains untracked as before; no file is deleted or rewritten
by the new verifier command. Compare the manifest-listed hashes against the
files and require every listed hash to match.

- [ ] **Step 5: Run formatting and repository-state checks**

Run:

```bash
git diff --check
git status --short
git log --oneline -6
```

Expected: no whitespace errors; only the pre-existing modified handoff and
untracked evidence directories remain outside committed implementation files.

- [ ] **Step 6: Record the exact source commit**

Run:

```bash
git rev-parse HEAD
git status --porcelain --untracked-files=no
```

Expected: the tracked tree is dirty only because of the pre-existing
`AGENT_HANDOFF_STATE.md`. The remote runner's existing exact-source staging
must stage committed files from `HEAD`, not the modified handoff.

---

### Task 6: Run Schema-V2 Remote Smoke, Then Canonical Only on `SMOKE_PASS`

**Files:**
- Create: `experiments/qwen35_hybrid_state/{schema-v2-smoke-run-id}/`
- Conditionally create: `experiments/qwen35_hybrid_state/{schema-v2-canonical-run-id}/`
- Conditionally modify: `docs/qwen35_hybrid_state_evidence_registry.json`
- Modify: `AGENT_HANDOFF_STATE.md`

**Interfaces:**
- Consumes: the exact clean implementation source commit from Task 5, immutable model revision, existing remote model snapshot, runner safety policy, and independent verifier.
- Produces: authoritative schema-v2 smoke evidence; canonical compatibility evidence only if admitted; final handoff with claim boundary and next gate.

- [ ] **Step 1: Verify SSH ControlMaster and remote identity**

Run:

```bash
ssh -S /tmp/ssh-sitian-10.232.195.203 \
  sitian@10.232.195.203 \
  'id -un && hostname && test -x /data00/home/sitian/sitian-workspace01/tllm/env/bin/python'
```

Expected:

```text
sitian
```

followed by one non-empty hostname line, and exit 0. If the socket is
unavailable, stop with `INCOMPLETE`; do not switch user or route.

- [ ] **Step 2: Run the runner's read-only preflight**

Create and persist a fresh tag, then run:

```bash
PREFLIGHT_TAG="qwen35-2b-hybrid-v2-preflight-$(date +%Y%m%d-%H%M%S)"
printf '%s\n' "$PREFLIGHT_TAG" \
  > /tmp/qwen35-hybrid-v2-preflight-tag
python3 tools/run_qwen35_hybrid_state_gate_remote.py preflight \
  --run-tag "$PREFLIGHT_TAG" \
  --resolved-revision 15852e8c16360a2fea060d615a32b45270f8a8fc
```

Expected: read-only success with the approved model revision, remote Python,
GPU 0 binding, disk observation, model-file identity, and no download or
remote checkout modification.

- [ ] **Step 3: Run remote source tests**

Use the runner's clean tar staging path to execute the four dependency-light
tests with the remote Python before GPU work.

Expected: local and remote SHA-256 values match for every staged source file,
all four remote tests exit 0, and the result is recorded in
`source_tests.json`.

- [ ] **Step 4: Run a new schema-v2 smoke**

Run:

```bash
SMOKE_TAG="qwen35-2b-hybrid-v2-smoke-$(date +%Y%m%d-%H%M%S)"
printf '%s\n' "$SMOKE_TAG" > /tmp/qwen35-hybrid-v2-smoke-tag
python3 tools/run_qwen35_hybrid_state_gate_remote.py smoke \
  --run-tag "$SMOKE_TAG" \
  --resolved-revision 15852e8c16360a2fea060d615a32b45270f8a8fc
```

Expected:

- exactly eight complete smoke rows;
- two BF16 cached repeats with identical actual full-logit hashes;
- BF16 greedy token and positive-margin agreement for every step;
- one complete FP32 control row with zero allclose violations and
  `mean_abs_diff <= 3e-6`;
- worker exit 0;
- fresh distinct ports;
- independent classification `SMOKE_PASS` or a preserved `INCOMPLETE`
  reason.

- [ ] **Step 5: Independently verify downloaded smoke evidence**

Run:

```bash
SMOKE_TAG="$(cat /tmp/qwen35-hybrid-v2-smoke-tag)"
python3 tools/verify_qwen35_hybrid_state_gate.py \
  --run-dir "experiments/qwen35_hybrid_state/$SMOKE_TAG" \
  --domain smoke \
  --write-report
```

Expected: the local independent result exactly matches the runner-recorded
independent result. Do not trust `summary.json` classification.

- [ ] **Step 6: Branch strictly on smoke classification**

If classification is `INCOMPLETE`:

- preserve every local and remote artifact;
- do not run canonical;
- append the exact reason, commands, source/model hashes, and claim boundary to
  `AGENT_HANDOFF_STATE.md`;
- do not create or modify the canonical evidence registry.

If classification is `SMOKE_PASS`, continue to Step 7.

- [ ] **Step 7: Run canonical from the admitted smoke identity**

Run:

```bash
SMOKE_TAG="$(cat /tmp/qwen35-hybrid-v2-smoke-tag)"
CANONICAL_TAG="qwen35-2b-hybrid-v2-canonical-$(date +%Y%m%d-%H%M%S)"
printf '%s\n' "$CANONICAL_TAG" \
  > /tmp/qwen35-hybrid-v2-canonical-tag
python3 tools/run_qwen35_hybrid_state_gate_remote.py canonical \
  --run-tag "$CANONICAL_TAG" \
  --resolved-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --smoke-run-tag "$SMOKE_TAG"
```

Expected: the full frozen case matrix completes or produces a preserved
`INCOMPLETE` bundle. Retry only exact `EADDRINUSE`, at most three attempts.

- [ ] **Step 8: Independently verify canonical evidence**

Run:

```bash
CANONICAL_TAG="$(cat /tmp/qwen35-hybrid-v2-canonical-tag)"
python3 tools/verify_qwen35_hybrid_state_gate.py \
  --run-dir "experiments/qwen35_hybrid_state/$CANONICAL_TAG" \
  --write-report
```

Expected: authoritative `GO`, `NO_GO`, or `INCOMPLETE`, with observed and
expected case counts, numerical summaries, state counts, logical bytes, unique
storage bytes, CUDA snapshots, and claim boundary.

- [ ] **Step 9: Update canonical registry only when canonical evidence exists**

If canonical ran, create or update
`docs/qwen35_hybrid_state_evidence_registry.json`. Construct every dynamic
value from the actual repository and artifacts:

```python
plan_commit = subprocess.check_output(
    [
        "git",
        "log",
        "-1",
        "--format=%H",
        "--",
        (
            "docs/superpowers/plans/"
            "2026-07-24-qwen35-hybrid-state-"
            "dtype-aware-equivalence-gate.md"
        ),
    ],
    text=True,
).strip()
source_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()
smoke_tag = Path("/tmp/qwen35-hybrid-v2-smoke-tag").read_text().strip()
canonical_tag = Path(
    "/tmp/qwen35-hybrid-v2-canonical-tag"
).read_text().strip()
smoke_report = json.loads(
    (
        Path("experiments/qwen35_hybrid_state")
        / smoke_tag
        / "independent_verification.json"
    ).read_text()
)
canonical_report_path = (
    Path("experiments/qwen35_hybrid_state")
    / canonical_tag
    / "independent_verification.json"
)
canonical_report = json.loads(canonical_report_path.read_text())
```

Write a closed object with:

- `schema_version=2`;
- the exact approved claim boundary;
- design path and commit `3c98836`;
- implementation-plan path and `plan_commit`;
- `source_commit`;
- fixed model repository and revision;
- smoke run path, required classification `SMOKE_PASS`, and SHA-256 of its
  `independent_verification.json`;
- canonical run path, observed classification from `canonical_report`, and
  SHA-256 of `canonical_report_path`.

Reject registry creation if any source value is unavailable, the smoke report
is not `SMOKE_PASS`, the canonical classification is outside
`GO | NO_GO | INCOMPLETE`, or any hash is not 64 lowercase hex characters.
Raw run directories remain untracked.

- [ ] **Step 10: Append the final handoff**

Append, without overwriting existing KV-residency content:

- design and plan commits;
- implementation commit range;
- exact local and remote commands;
- model revision and source hashes;
- smoke and canonical evidence paths;
- authoritative classifications and reasons;
- what the result proves;
- what it does not prove;
- whether production engine files changed;
- the next separately authorized gate.

- [ ] **Step 11: Validate and commit only tracked evidence metadata**

Run:

```bash
python3 -m json.tool docs/qwen35_hybrid_state_evidence_registry.json \
  >/dev/null 2>&1 || test ! -e docs/qwen35_hybrid_state_evidence_registry.json
git diff --check
git status --short
```

If a canonical registry exists:

```bash
git add \
  AGENT_HANDOFF_STATE.md \
  docs/qwen35_hybrid_state_evidence_registry.json
```

If canonical did not run:

```bash
git add AGENT_HANDOFF_STATE.md
```

Then run:

```bash
git diff --cached --check
git commit -m "docs: record qwen35 dtype-aware gate evidence"
```

Expected: no raw run directory is staged and no production file is changed.

---

## Completion Audit

Before claiming completion, map each requirement to evidence:

| Requirement | Required evidence |
|---|---|
| Schema-v2 dtype-aware contract | Contract source, contract tests, exact commit |
| Bitwise cached repeatability | Two-repeat hashes in new smoke and verifier guard |
| Greedy decision preservation | Actual/oracle winner records across every executed case |
| Positive winner margins | Independently recomputed margins and tamper tests |
| Strict FP32 control | FP32 control row, zero violations, mean-abs cap, verifier test |
| BF16 drift remains diagnostic | Large-finite-drift verifier test still classifies `GO` |
| State/lifecycle/isolation/accounting preserved | Existing and schema-v2 tamper suites plus canonical guards |
| Schema-v1 remains `INCOMPLETE` | Old bundle verification and unchanged artifact hashes |
| Canonical blocked before smoke pass | Runner admission tests and actual command ordering |
| Remote safety constraints | Process/port/source manifests and runner tests |
| Production engine unchanged | Empty `git diff 3c98836..HEAD -- tinyvllm README.md` |
| No performance/compression/quality claim | Spec, reports, registry, and handoff claim boundary |
| Final classification authoritative | Local independent verifier report and SHA-256 |

Completion requires inspecting the actual artifacts for every row above. Passing
unit tests alone is insufficient. If smoke is `INCOMPLETE`, the implementation
and smoke investigation may be complete, but Qwen3.5 compatibility remains
unproven and canonical execution remains blocked.
