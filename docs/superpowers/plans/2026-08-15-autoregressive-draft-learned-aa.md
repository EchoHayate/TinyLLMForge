# Autoregressive Draft Learned A/A Process-Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are prohibited for this worktree.

**Goal:** Build and run a source-bound learned/learned A/A control that compares two independent primed learned-policy process epochs in one TP4 batch-four bundle without changing runtime execution semantics.

**Architecture:** Add a dedicated learned A/A diagnostic, independent filesystem-bound verifier, and remote runner rather than extending the target/learned authority chain. The diagnostic consumes two distinct learned worker artifacts plus repeat-local GPU and host telemetry, enforces exact workload/output identity, computes per-epoch stationarity and A/B summaries, and emits only stable, candidate-effect, or inconclusive single-bundle classifications. The runner packages one source snapshot, launches `learned_a` and `learned_b` as isolated prime and measured processes, verifies the canonical artifact remotely and locally, and preserves every raw input in a manifest.

**Tech Stack:** Python 3 standard library, pytest, Bash, SSH/rsync, PyTorch runtime worker, `nvidia-smi`, `/proc`, `vmstat`, `mpstat`, and `pidstat`.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify runtime execution semantics for this control.
- Do not use subagents.
- Do not create or switch branches or worktrees.
- Do not stage, commit, push, stash, reset, or clean.
- Use `sitian@10.232.195.203`.
- Use `/data00/home/sitian/miniconda3/envs/py311/bin/python` only as the remote Python executable.
- Write new remote artifacts only below `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815`.
- Do not write experiment artifacts below `/data00`.
- Preserve `MAX_PROPOSAL_TOKENS=4`, temperature zero, accepted-prefix semantics, and exact greedy parity.
- Preserve the workload-derived exact Proposal-KV capacity.
- Do not introduce `torch.cuda.synchronize()` in the measured request path.
- Do not terminate unrelated GPU processes; preserve GPU-7 PID `703088`.
- Do not represent synthetic or fake KV copies as real KV movement.
- Use bounded foreground commands; do not start background watchers.
- Reuse existing execution sessions where practical because the unified-exec process count is already near its limit.
- A single bundle may set `candidate_process_boundary_effect=true`, but must always set `process_boundary_effect_established=false`.
- This slice does not implement a cross-bundle promotion artifact. A candidate result stops at a follow-up design gate.

## File Map

- Create `tools/autoregressive_draft_learned_aa_diagnostic.py`: canonical schema, raw-input validation, repeat alignment, stationarity, comparison, classification, CLI assembly, and deterministic JSON output.
- Create `tools/verify_autoregressive_draft_learned_aa_diagnostic.py`: safe relative-path resolution, digest verification, full recomputation, structural-equality check, and receipt output.
- Create `tools/test_autoregressive_draft_learned_aa_diagnostic.py`: fixtures and focused tests for schema, parity, workload identity, coverage, stationarity, classification, path safety, and tamper rejection.
- Create `tools/run_autoregressive_draft_learned_aa_remote.sh`: source packaging, preflight, isolated A/B prime and measured epochs, sampler ownership, remote/local verification, transfer, and manifest.
- Modify `tools/test_autoregressive_draft_instability_telemetry.py`: source-contract assertions for the dedicated runner.
- Create one discovery experiment directory below `experiments/autoregressive_draft/` only after all local gates pass.
- Modify `AGENT_HANDOFF_STATE.md` only after a verified discovery bundle exists.

---

### Task 1: Canonical Schema and Learned-Worker Invariants

**Files:**
- Create: `tools/test_autoregressive_draft_learned_aa_diagnostic.py`
- Create: `tools/autoregressive_draft_learned_aa_diagnostic.py`

**Interfaces:**
- Consumes: `validate_worker_result(worker, expected_warmup_runs, expected_measured_runs)` from `tools/autoregressive_draft_performance_gate.py`.
- Produces: `validate_prime_worker(worker: object, *, artifact_identity: str) -> dict`.
- Produces: `validate_measured_worker(worker: object, *, artifact_identity: str) -> dict`.
- Produces: `validate_workload_identity(learned_a: dict, learned_b: dict) -> dict`.
- Produces: constants `EPOCH_ORDER`, `PRIMARY_METRICS`, `RANGE_OVER_MEDIAN_LIMIT`, `HALF_DRIFT_FRACTION_LIMIT`, and `E2E_EFFECT_THRESHOLD`.

- [ ] **Step 1: Write fixtures that create valid prime and measured learned workers**

Add fixture builders that start from the established performance-worker schema rather than inventing a second worker schema:

```python
EPOCHS = ("learned_a", "learned_b")


def make_worker(
    *,
    measured_runs: int,
    e2e_values: list[float],
    tpot_values: list[float],
    proposal_forward_values: list[float],
) -> dict:
    worker = make_valid_worker_result(
        policy="learned",
        batch_size=4,
        warmup_runs=2,
        measured_runs=measured_runs,
    )
    for repeat, run in enumerate(worker["measured_runs"]):
        for request in run["timing"]["per_request"]:
            request["completion_latency_s"] = e2e_values[repeat]
            request["tpot_s"] = tpot_values[repeat]
        run["runtime"]["draft_executor_timing"]["max_rank_ms"][
            "proposal_forward"
        ] = proposal_forward_values[repeat]
    return worker
```

The fixture must preserve the real fields used by `validate_worker_result`, including prompt rows, model identifiers, outputs, Proposal-KV allocator metadata, runtime counters, and request timing rows.

- [ ] **Step 2: Write failing invariant tests**

Cover both valid epochs and each rejection independently:

```python
@pytest.mark.parametrize("artifact_identity", EPOCHS)
def test_measured_worker_requires_learned_policy(artifact_identity):
    worker = make_measured_worker()
    worker["policy"] = "target"
    with pytest.raises(ValueError, match="policy must be learned"):
        diagnostic.validate_measured_worker(
            worker,
            artifact_identity=artifact_identity,
        )


def test_prime_worker_requires_two_warmups_and_one_repeat():
    worker = make_prime_worker()
    worker["measured_runs"].append(copy.deepcopy(worker["measured_runs"][0]))
    with pytest.raises(ValueError, match="prime worker"):
        diagnostic.validate_prime_worker(
            worker,
            artifact_identity="learned_a",
        )


def test_measured_worker_requires_batch_four_temperature_zero():
    worker = make_measured_worker()
    worker["batch_size"] = 1
    with pytest.raises(ValueError, match="batch size must be four"):
        diagnostic.validate_measured_worker(
            worker,
            artifact_identity="learned_a",
        )


def test_workload_identity_requires_exact_proposal_kv_capacity_inputs():
    learned_a = make_measured_worker()
    learned_b = copy.deepcopy(learned_a)
    learned_b["proposal_kv_capacity"]["max_proposal_tokens"] = 8
    with pytest.raises(ValueError, match="Proposal-KV capacity"):
        diagnostic.validate_workload_identity(learned_a, learned_b)
```

Also reject invalid artifact identities, nonzero temperature, a non-four `max_proposal_tokens`, different TP world size or GPU index set, different target/draft model identities, prompt token IDs, requested output lengths, and fixed or oversized Proposal-KV capacity derivation.

- [ ] **Step 3: Run the invariant tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py \
  -k 'worker or workload_identity'
```

Expected: collection fails because `autoregressive_draft_learned_aa_diagnostic` does not exist.

- [ ] **Step 4: Implement constants and worker validation**

Start the module with:

```python
SCHEMA_VERSION = 1
EPOCH_ORDER = ("learned_a", "learned_b")
WORKER_POLICY = "learned"
PRIME_WARMUP_RUNS = 2
PRIME_MEASURED_RUNS = 1
MEASURED_WARMUP_RUNS = 2
MEASURED_RUNS = 8
BATCH_SIZE = 4
MAX_PROPOSAL_TOKENS = 4
RANGE_OVER_MEDIAN_LIMIT = 0.25
HALF_DRIFT_FRACTION_LIMIT = 0.20
E2E_EFFECT_THRESHOLD = 0.10
PRIMARY_METRICS = (
    "e2e_s",
    "tpot_s",
    "executor_proposal_forward_ms",
)
GPU_INDICES = (3, 4, 6, 7)
```

Implement one shared validator and two explicit entry points:

```python
def _validate_worker(
    worker: object,
    *,
    artifact_identity: str,
    expected_measured_runs: int,
    kind: str,
) -> dict:
    if artifact_identity not in EPOCH_ORDER:
        raise ValueError("invalid learned A/A artifact identity")
    normalized = validate_worker_result(
        worker,
        expected_warmup_runs=2,
        expected_measured_runs=expected_measured_runs,
    )
    if normalized["policy"] != WORKER_POLICY:
        raise ValueError(f"{kind} policy must be learned")
    if normalized["batch_size"] != BATCH_SIZE:
        raise ValueError(f"{kind} batch size must be four")
    if normalized.get("temperature", 0.0) != 0.0:
        raise ValueError(f"{kind} temperature must be zero")
    return normalized


def validate_prime_worker(worker: object, *, artifact_identity: str) -> dict:
    return _validate_worker(
        worker,
        artifact_identity=artifact_identity,
        expected_measured_runs=PRIME_MEASURED_RUNS,
        kind="prime worker",
    )


def validate_measured_worker(
    worker: object,
    *,
    artifact_identity: str,
) -> dict:
    return _validate_worker(
        worker,
        artifact_identity=artifact_identity,
        expected_measured_runs=MEASURED_RUNS,
        kind="measured worker",
    )
```

`validate_workload_identity` must return a normalized mapping only after checking exact equality of target checkpoint identity, draft checkpoint identity, tokenizer identity, prompt rows/token IDs, output length, batch size, temperature, `MAX_PROPOSAL_TOKENS`, Proposal-KV capacity derivation inputs, TP world size, and GPU index set.

- [ ] **Step 5: Run the invariant tests and confirm GREEN**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py \
  -k 'worker or workload_identity'
```

Expected: all selected tests pass.

---

### Task 2: Repeat Alignment, Stationarity, Comparison, and Classification

**Files:**
- Modify: `tools/test_autoregressive_draft_learned_aa_diagnostic.py`
- Modify: `tools/autoregressive_draft_learned_aa_diagnostic.py`

**Interfaces:**
- Consumes: `parse_gpu_telemetry`, `summarize_gpu_telemetry`, and `validate_campaign_intervals` from `tools/autoregressive_draft_instability_telemetry.py`.
- Consumes: `parse_host_jsonl`, `align_repeat_samples`, and `derive_repeat_metrics` from `tools/autoregressive_draft_host_semantic_diagnostic.py`.
- Produces: `build_epoch_summary(...) -> dict`.
- Produces: `compare_epochs(learned_a: dict, learned_b: dict) -> dict`.
- Produces: `classify_learned_aa(*, epochs: dict, comparison: dict) -> tuple[str, list[str], dict]`.
- Produces: `build_learned_aa_artifact(...) -> dict`.
- Produces: `validate_learned_aa_artifact(artifact: object) -> dict`.

- [ ] **Step 1: Write failing exact-parity and coverage tests**

Add tests that prove failures are verifier errors rather than inconclusive classifications:

```python
def test_build_artifact_requires_exact_output_parity_per_repeat():
    inputs = make_valid_inputs()
    inputs["workers"]["learned_b"]["measured_runs"][3]["outputs"][0][0] += 1
    with pytest.raises(ValueError, match="exact parity failed at repeat 3"):
        diagnostic.build_learned_aa_artifact(**inputs)


def test_build_artifact_requires_five_samples_per_gpu_per_repeat():
    inputs = make_valid_inputs()
    inputs["gpu_rows"]["learned_b"] = remove_repeat_gpu_samples(
        inputs["gpu_rows"]["learned_b"],
        repeat=5,
        gpu_index=7,
        keep=4,
    )
    with pytest.raises(ValueError, match="GPU coverage"):
        diagnostic.build_learned_aa_artifact(**inputs)


def test_build_artifact_requires_host_repeat_local_gap_below_limit():
    inputs = make_valid_inputs()
    inputs["host_rows"]["learned_a"] = create_internal_gap(
        inputs["host_rows"]["learned_a"],
        repeat=2,
        gap_seconds=0.61,
    )
    with pytest.raises(ValueError, match="sample gap"):
        diagnostic.build_learned_aa_artifact(**inputs)
```

Host fixtures must use cadence `0.2`, maximum repeat-local gap `0.6`, and boundary allowance `0.4`. GPU fixtures must cover every repeat on GPUs `3,4,6,7` with at least five samples after the `0.6` boundary-nearest allowance.

- [ ] **Step 2: Write failing classification tests**

Use deterministic eight-repeat series:

```python
@pytest.mark.parametrize(
    ("a_e2e", "b_e2e", "a_tpot", "b_tpot", "a_forward", "b_forward",
     "expected"),
    [
        (10.5, 10.0, 5.2, 5.0, 102.0, 100.0, "LEARNED_AA_STABLE"),
        (12.0, 10.0, 6.0, 5.0, 120.0, 100.0,
         "LEARNED_AA_PROCESS_BOUNDARY_EFFECT"),
        (12.0, 10.0, 4.0, 5.0, 120.0, 100.0,
         "LEARNED_AA_INCONCLUSIVE"),
        (12.0, 10.0, 6.0, 5.0, 100.0, 100.0,
         "LEARNED_AA_INCONCLUSIVE"),
    ],
)
def test_classification_thresholds_and_direction(
    a_e2e,
    b_e2e,
    a_tpot,
    b_tpot,
    a_forward,
    b_forward,
    expected,
):
    artifact = diagnostic.build_learned_aa_artifact(
        **make_valid_inputs(
            learned_a_metrics=(a_e2e, a_tpot, a_forward),
            learned_b_metrics=(b_e2e, b_tpot, b_forward),
        )
    )
    assert artifact["classification"] == expected
```

Add explicit boundary cases for `0.099999` as stable and `0.10` as candidate when all directions agree. Add a nonstationary primary metric case and assert inconclusive with the failed threshold in `classification_reasons`.

- [ ] **Step 3: Run the alignment/classification tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py \
  -k 'parity or coverage or stationarity or classification'
```

Expected: failures identify missing epoch-summary, comparison, and classification functions.

- [ ] **Step 4: Implement metric extraction and stationarity**

For each epoch and measured repeat, preserve raw outputs and the worker timing/runtime row, then attach repeat-local GPU and host summaries:

```python
def _stationarity(metric: str, values: list[float]) -> dict:
    if len(values) != MEASURED_RUNS:
        raise ValueError("stationarity requires eight measured values")
    median = statistics.median(values)
    first = statistics.median(values[:4])
    second = statistics.median(values[4:])
    minimum = min(values)
    maximum = max(values)
    range_over_median = (
        0.0 if median == 0.0 and maximum == minimum
        else None if median == 0.0
        else (maximum - minimum) / abs(median)
    )
    half_drift_fraction = (
        0.0 if median == 0.0 and first == second
        else None if median == 0.0
        else abs(second - first) / abs(median)
    )
    stable = (
        range_over_median is not None
        and half_drift_fraction is not None
        and range_over_median <= RANGE_OVER_MEDIAN_LIMIT
        and half_drift_fraction <= HALF_DRIFT_FRACTION_LIMIT
    )
    return {
        "metric": metric,
        "values": values,
        "median": median,
        "minimum": minimum,
        "maximum": maximum,
        "range_over_median": range_over_median,
        "first_half_median": first,
        "second_half_median": second,
        "half_drift_fraction": half_drift_fraction,
        "stable": stable,
    }
```

`build_epoch_summary` must:

1. validate the measured worker;
2. validate nonoverlapping worker intervals;
3. align GPU samples with `edge_allowance_ns=600_000_000`;
4. align host samples with cadence `0.2`, maximum internal gap `0.6`, and `boundary_allowance_ns=400_000_000`;
5. require five samples per repeat for each GPU `3,4,6,7`;
6. build eight measured-repeat rows;
7. compute stationarity independently for E2E, TPOT, and proposal-forward;
8. preserve TTFT, throughput, acceptance, GPU memory, real KV H2D/D2H, Proposal-KV movement, and primary host metrics when present;
9. represent absent secondary counters as missing evidence rather than zero.

- [ ] **Step 5: Implement comparison and single-bundle classification**

Use one comparison helper for every numeric metric:

```python
def _comparison_row(metric: str, a_values: list[float], b_values: list[float]):
    median_a = statistics.median(a_values)
    median_b = statistics.median(b_values)
    delta = median_a - median_b
    relative_delta = None if median_b == 0.0 else delta / median_b
    sign = 0 if delta == 0.0 else 1 if delta > 0.0 else -1
    return {
        "metric": metric,
        "learned_a_values": a_values,
        "learned_b_values": b_values,
        "learned_a_median": median_a,
        "learned_b_median": median_b,
        "absolute_difference": abs(delta),
        "relative_delta": relative_delta,
        "absolute_relative_delta": (
            None if relative_delta is None else abs(relative_delta)
        ),
        "sign": sign,
    }
```

Classification must implement exactly:

```python
if not all_primary_stationary:
    classification = "LEARNED_AA_INCONCLUSIVE"
elif e2e["absolute_relative_delta"] < E2E_EFFECT_THRESHOLD:
    classification = "LEARNED_AA_STABLE"
elif (
    e2e["sign"] != 0
    and tpot["sign"] == e2e["sign"]
    and proposal_forward["sign"] == e2e["sign"]
):
    classification = "LEARNED_AA_PROCESS_BOUNDARY_EFFECT"
else:
    classification = "LEARNED_AA_INCONCLUSIVE"
```

Every artifact must include:

```python
"claim_state": {
    "candidate_process_boundary_effect": (
        classification == "LEARNED_AA_PROCESS_BOUNDARY_EFFECT"
    ),
    "process_boundary_effect_established": False,
}
```

`validate_learned_aa_artifact` must rebuild the artifact from its embedded normalized data and reject structural differences using `_equivalent`.

- [ ] **Step 6: Run the complete diagnostic unit suite**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py \
  -k 'not verifier and not runner'
```

Expected: all selected tests pass, including exact parity, coverage, stationarity, threshold boundaries, conflicting directions, and claim-state assertions.

---

### Task 3: Hash-Bound CLI and Independent Verifier

**Files:**
- Modify: `tools/test_autoregressive_draft_learned_aa_diagnostic.py`
- Modify: `tools/autoregressive_draft_learned_aa_diagnostic.py`
- Create: `tools/verify_autoregressive_draft_learned_aa_diagnostic.py`

**Interfaces:**
- Produces diagnostic CLI arguments for two prime workers, two measured workers, two GPU CSV files, two host JSONL files, epoch-order and prime-control files, repository root, bundle role, and output path.
- Produces: `verify_learned_aa_diagnostic(artifact_path: Path, repo_root: Path) -> dict`.
- Produces verifier CLI arguments `--artifact`, `--repo-root`, and optional `--receipt`.

- [ ] **Step 1: Write failing digest, path-safety, and recomputation tests**

Add tests for all ten minimum bound input files and all required source files:

```python
def test_verifier_recomputes_from_hash_bound_raw_inputs(tmp_path):
    artifact_path = write_bound_campaign(tmp_path)
    receipt = verifier.verify_learned_aa_diagnostic(
        artifact_path,
        ROOT,
    )
    assert receipt["status"] == "PASS"
    assert receipt["input_files_verified"] >= 10
    assert receipt["source_files_verified"] >= 6
    assert receipt["process_boundary_effect_established"] is False


@pytest.mark.parametrize("path", ["/tmp/worker.json", "../worker.json"])
def test_verifier_rejects_unsafe_input_paths(tmp_path, path):
    artifact_path = write_bound_campaign(tmp_path)
    artifact = json.loads(artifact_path.read_text())
    artifact["input_files"]["learned_a_worker"]["path"] = path
    artifact_path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="relative path"):
        verifier.verify_learned_aa_diagnostic(artifact_path, ROOT)


def test_verifier_rejects_raw_input_tampering(tmp_path):
    artifact_path = write_bound_campaign(tmp_path)
    worker_path = artifact_path.parent / "workers/learned-a-b4.json"
    worker_path.write_text(worker_path.read_text() + "\n")
    with pytest.raises(ValueError, match="input hash mismatch"):
        verifier.verify_learned_aa_diagnostic(artifact_path, ROOT)
```

Also tamper each prime worker, measured worker, GPU CSV, host JSONL, `epoch-order.txt`, `prime-each-epoch.txt`, canonical artifact field, and source file digest. Require distinct A/B digests for measured worker and raw telemetry files.

- [ ] **Step 2: Run verifier tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py \
  -k 'verifier or tamper or unsafe or digest'
```

Expected: failures identify the missing verifier and CLI assembly.

- [ ] **Step 3: Implement deterministic CLI assembly**

The diagnostic CLI must require:

```text
--learned-a-prime-worker
--learned-b-prime-worker
--learned-a-worker
--learned-b-worker
--learned-a-gpu-csv
--learned-b-gpu-csv
--learned-a-host-jsonl
--learned-b-host-jsonl
--epoch-order-file
--prime-each-epoch-file
--bundle-role
--repo-root
--out
```

Only `discovery` is accepted in this slice. The CLI must:

1. load all JSON/CSV/JSONL/text inputs;
2. require `epoch-order.txt` to contain `learned_a,learned_b`;
3. require `prime-each-epoch.txt` to contain `1`;
4. calculate relative paths and SHA-256 digests;
5. hash-bind at least the worker, sampler, diagnostic, verifier, runner, performance gate, performance worker, GPU telemetry helper, and host-semantic helper sources;
6. build the canonical artifact;
7. write with `write_json_atomic`.

- [ ] **Step 4: Implement safe path resolution and full recomputation**

The verifier must reject absolute and parent-traversing paths before touching the filesystem:

```python
def _resolve_bound_path(base: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("bound path must be a non-empty relative path")
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("bound path must be a safe relative path")
    resolved = base / candidate
    if not resolved.is_file():
        raise ValueError(f"bound file is missing: {relative_path}")
    return resolved
```

`verify_learned_aa_diagnostic` must:

1. load `learned-aa.json`;
2. validate schema and fixed claim boundary;
3. verify every `input_files` digest relative to the artifact directory;
4. verify every `source_files` digest relative to `repo_root`;
5. reload all raw inputs;
6. call the same public `build_learned_aa_artifact` function;
7. require exact structural equality;
8. return a receipt containing classification, exact parity, epoch order, measured repeats, coverage, source/input counts, candidate flag, and `process_boundary_effect_established=false`.

- [ ] **Step 5: Run focused tests and compilation**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py

python3 -m py_compile \
  tools/autoregressive_draft_learned_aa_diagnostic.py \
  tools/verify_autoregressive_draft_learned_aa_diagnostic.py
```

Expected: all tests pass and compilation exits zero.

---

### Task 4: Dedicated Remote Runner and Source Contract

**Files:**
- Create: `tools/run_autoregressive_draft_learned_aa_remote.sh`
- Modify: `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Consumes: existing performance worker and host sampler CLIs.
- Consumes: diagnostic and verifier CLIs from Task 3.
- Produces fixed epoch order `learned_a,learned_b`.
- Produces distinct prime, measured worker, log, GPU, host-semantic, `vmstat`, `mpstat`, and `pidstat` paths for each epoch.
- Produces `learned-aa.json`, remote/local receipts, exit-code receipts, and `manifest.sha256`.

- [ ] **Step 1: Write failing runner source-contract tests**

Add a second runner path constant and assertions:

```python
LEARNED_AA_RUNNER_PATH = (
    ROOT / "tools/run_autoregressive_draft_learned_aa_remote.sh"
)


def test_learned_aa_runner_owns_isolated_epoch_contract():
    script = LEARNED_AA_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        'EPOCH_ORDER="learned_a,learned_b"',
        'for epoch in learned_a learned_b; do',
        '--policy learned',
        '--batch-size 4',
        '--warmup-runs 2',
        '--measured-runs 1',
        '--measured-runs 8',
        'prime_epoch "${epoch}"',
        'run_epoch "${epoch}"',
        'start_samplers "${epoch}"',
        'stop_samplers',
        'workers/${epoch//_/-}-b4.json',
        'prime-workers/${epoch//_/-}-prime-b4.json',
        'telemetry/${epoch//_/-}-gpu.csv',
        'host-semantic/${epoch//_/-}-host.jsonl',
        'tools/autoregressive_draft_learned_aa_diagnostic.py',
        'tools/verify_autoregressive_draft_learned_aa_diagnostic.py',
        'verify.learned-aa.remote.json',
        'verify.learned-aa.local.json',
        'manifest.sha256',
    ):
        assert expected in script
    assert "torch.cuda.synchronize" not in script
```

Add ordering assertions that each epoch's prime call precedes sampler startup and measured worker launch. Assert that the script contains no `--policy learned_a`, no `--policy learned_b`, no `killall`, no `pkill`, and no command targeting PID `703088`.

- [ ] **Step 2: Run runner-contract tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k learned_aa_runner
```

Expected: failure reports the missing dedicated runner.

- [ ] **Step 3: Implement argument parsing and preflight**

Copy only generic transport/safety patterns from the existing r8 runner. Support:

```text
--run-tag <tag>
--local-run <path>
--remote-run <path>
--ssh-control-path <existing-socket>
--bundle-role discovery
```

Set defaults:

```bash
REMOTE_HOST="${REMOTE_HOST:-sitian@10.232.195.203}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/data00/home/sitian/miniconda3/envs/py311/bin/python}"
REMOTE_BASE="${REMOTE_BASE:-/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815}"
TARGET_MODEL="${TARGET_MODEL:-${REMOTE_BASE}/target-qwen3-1.7b}"
DRAFT_MODEL="${DRAFT_MODEL:-${REMOTE_BASE}/draft}"
GPU_INDICES="${GPU_INDICES:-3,4,6,7}"
EPOCH_ORDER="learned_a,learned_b"
PRIME_EACH_EPOCH=1
LOCAL_RUN="${LOCAL_RUN:-}"
REMOTE_RUN="${REMOTE_RUN:-}"
```

Derive `LOCAL_RUN` and `REMOTE_RUN` only after parsing. Refuse an existing local path before SSH and an existing remote path before workload execution. Record all effective values in `command.txt`.

Package only the source files required by the worker, sampler, diagnostic, verifier, and their imported runtime modules. Run remote `py_compile`, focused pytest, and `bash -n` from the packaged source before loading models.

- [ ] **Step 4: Implement isolated prime and measured epoch functions**

Use artifact identity only for paths:

```bash
epoch_slug() {
  printf '%s\n' "${1//_/-}"
}

prime_epoch() {
  local epoch="$1"
  local slug
  slug="$(epoch_slug "${epoch}")"
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy learned \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 1 \
      --out "${artifacts}/prime-workers/${slug}-prime-b4.json" \
      >"${artifacts}/prime-logs/${slug}-prime-b4.log" 2>&1
}

run_epoch() {
  local epoch="$1"
  local slug
  slug="$(epoch_slug "${epoch}")"
  start_samplers "${epoch}"
  set +e
  "${python_executable}" \
    tools/autoregressive_draft_performance_worker.py \
      --target-model "${target_model}" \
      --draft-model "${draft_model}" \
      --policy learned \
      --batch-size 4 \
      --warmup-runs 2 \
      --measured-runs 8 \
      --out "${artifacts}/workers/${slug}-b4.json" \
      >"${artifacts}/logs/${slug}-b4.log" 2>&1
  local worker_status=$?
  set -e
  stop_samplers
  return "${worker_status}"
}

for epoch in learned_a learned_b; do
  prime_epoch "${epoch}"
  run_epoch "${epoch}"
done
```

Each `start_samplers` call must create a fresh PID list and unique A/B output paths. `stop_samplers` may signal and reap only PIDs stored by the current script. No worker process, Python interpreter, CUDA context, model object, or sampler process may span both epochs.

- [ ] **Step 5: Implement assembly, verification, transfer, and manifest**

Write:

```text
epoch-order.txt        learned_a,learned_b
prime-each-epoch.txt   1
bundle-role.txt        discovery
```

Invoke the diagnostic with every raw path, then invoke the independent verifier remotely. Record preflight, prime A, measured A, prime B, measured B, diagnostic, and remote-verifier statuses in distinct receipt files even when a later stage fails.

Always attempt to transfer the remote artifact directory after remote command completion. If the canonical artifact exists and the remote verifier passed, run the local verifier against the current local source. Build the manifest with:

```bash
(
  cd "${LOCAL_RUN}"
  find . -type f ! -name manifest.sha256 -print0 |
    sort -z |
    xargs -0 shasum -a 256 > manifest.sha256
  shasum -a 256 -c manifest.sha256
)
```

Preserve partial artifacts and original nonzero receipts. Do not rerun only one measured epoch in an existing bundle.

- [ ] **Step 6: Run shell and source-contract validation**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
bash -n tools/run_autoregressive_draft_learned_aa_remote.sh

python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k learned_aa_runner

rg -n 'torch\\.cuda\\.synchronize|--policy learned_[ab]|killall|pkill|703088' \
  tools/run_autoregressive_draft_learned_aa_remote.sh
```

Expected: shell syntax and tests pass; `rg` returns no matches.

---

### Task 5: Full Local Gate and Spec-Coverage Audit

**Files:**
- Modify only files already listed if a gate exposes a defect.

**Interfaces:**
- Consumes all implementation artifacts from Tasks 1-4.
- Produces a local readiness decision; it does not produce performance authority.

- [ ] **Step 1: Run the exact focused regression gate**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_executor.py \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run compilation and shell validation**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m py_compile \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_learned_aa_diagnostic.py \
  tools/verify_autoregressive_draft_learned_aa_diagnostic.py

bash -n tools/run_autoregressive_draft_learned_aa_remote.sh
```

Expected: both commands exit zero.

- [ ] **Step 3: Audit safety and source scope**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
git diff -- \
  tools/autoregressive_draft_learned_aa_diagnostic.py \
  tools/verify_autoregressive_draft_learned_aa_diagnostic.py \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py \
  tools/run_autoregressive_draft_learned_aa_remote.sh \
  tools/test_autoregressive_draft_instability_telemetry.py

rg -n 'torch\\.cuda\\.synchronize' \
  tools/autoregressive_draft_performance_worker.py \
  tools/run_autoregressive_draft_learned_aa_remote.sh

git status --short
```

Expected:

- no measured-path synchronization was added;
- no runtime module was modified for the control;
- only the planned files are attributable to this slice;
- pre-existing unrelated changes remain untouched;
- no files are staged.

- [ ] **Step 4: Map every approved design requirement to a test or runner assertion**

Create a review note in the execution transcript, not a new repository file, with one evidence line for each:

```text
fixed learned_a,learned_b order
both worker policies equal learned
isolated prime and measured processes
distinct A/B artifact paths
2 warmups + 1 discarded prime repeat
2 warmups + 8 measured repeats
batch 4 and temperature 0
MAX_PROPOSAL_TOKENS=4
workload-derived Proposal-KV capacity
exact per-repeat output parity
host cadence/gap/boundary coverage
GPU per-repeat/per-device five-sample coverage
0.25 range-over-median threshold
0.20 half-drift threshold
stable less-than 0.10 boundary
candidate greater-than-or-equal 0.10 boundary
direction agreement
inconclusive cases
safe relative paths
raw and source tamper rejection
single-bundle established flag always false
no measured-path torch.cuda.synchronize
remote/local verifier and manifest chain
```

Any uncovered line blocks remote execution.

---

### Task 6: Discovery Bundle and Result-Dependent Handoff

**Files:**
- Create: `experiments/autoregressive_draft/<run-tag>/` through runner download.
- Modify: `AGENT_HANDOFF_STATE.md` only after remote and local verification pass.

**Interfaces:**
- Consumes the validated runner from Task 5.
- Produces one source-bound discovery bundle.
- Produces no established process-boundary claim.

- [x] **Step 1: Preflight remote filesystem and protected GPU process**

Run one bounded SSH command:

```bash
ssh sitian@10.232.195.203 '
set -e
test -x /data00/home/sitian/miniconda3/envs/py311/bin/python
test -w /dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815
df -h /dev/shm /data00
nvidia-smi
ps -p 703088 -o pid=,user=,cmd=
'
```

Expected: Python is executable, the `/dev/shm` base is writable, GPU state is recorded, and PID `703088` remains present. A missing protected PID or an occupied required GPU blocks launch and requires diagnosis rather than termination.

- [x] **Step 2: Launch one foreground discovery bundle**

Choose a fresh UTC run tag and run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
RUN_TAG="tp4-qwen3-b4-learned-aa-discovery-$(date -u +%Y%m%dT%H%M%SZ)"
tools/run_autoregressive_draft_learned_aa_remote.sh \
  --run-tag "${RUN_TAG}" \
  --bundle-role discovery
```

If a valid existing SSH control socket is available, add:

```text
--ssh-control-path <existing-socket>
```

Expected: the command remains foreground and exits zero. Do not launch a watcher or a second bundle concurrently.

- [x] **Step 3: Verify authority receipts and manifest**

Run:

```bash
cd "/Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/autoregressive_draft/${RUN_TAG}"
cat remote-status.txt
cat preflight-exit-code.txt
cat learned-a-prime-exit-code.txt
cat learned-a-worker-exit-code.txt
cat learned-b-prime-exit-code.txt
cat learned-b-worker-exit-code.txt
cat diagnostic-exit-code.txt
cat verify-learned-aa-remote-exit-code.txt
shasum -a 256 -c manifest.sha256
python3 -m json.tool verify.learned-aa.remote.json >/dev/null
python3 -m json.tool verify.learned-aa.local.json >/dev/null
python3 -m json.tool learned-aa.json >/dev/null
```

Expected: all exit receipts are zero, manifest verification passes, and all JSON files parse.

- [x] **Step 4: Inspect the canonical result without overclaiming**

Run:

```bash
python3 - <<'PY' "/Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/autoregressive_draft/${RUN_TAG}/learned-aa.json"
import json
import sys

artifact = json.load(open(sys.argv[1], encoding="utf-8"))
summary = {
    "status": artifact["status"],
    "classification": artifact["classification"],
    "classification_reasons": artifact["classification_reasons"],
    "claim_state": artifact["claim_state"],
    "epoch_order": artifact["epoch_order"],
    "exact_parity": artifact["exact_parity"],
    "e2e": artifact["comparison"]["primary"]["e2e_s"],
    "tpot": artifact["comparison"]["primary"]["tpot_s"],
    "proposal_forward": artifact["comparison"]["primary"][
        "executor_proposal_forward_ms"
    ],
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
```

Require:

```text
status: PASS
epoch_order: learned_a, learned_b
exact_parity: true
measured repeats per epoch: 8
host/GPU repeat coverage: PASS
source identity: PASS
process_boundary_effect_established: false
```

- [x] **Step 5: Follow the result-dependent boundary**

Apply exactly one branch:

```text
LEARNED_AA_STABLE
  Record that the r7/r8 learned reversal was not reproduced by the
  same-policy process boundary. Return to a separately designed controlled
  runtime/host attribution experiment.

LEARNED_AA_PROCESS_BOUNDARY_EFFECT
  Record only a candidate effect. Stop before a second bundle and write a
  follow-up cross-bundle replication design with its own artifact and
  verifier. Do not set process_boundary_effect_established=true.

LEARNED_AA_INCONCLUSIVE
  Record the exact stationarity or metric-direction failures. Repair the
  evidence protocol before any causal or optimization experiment.
```

- [x] **Step 6: Update the handoff with evidence and remaining limits**

Append the run tag, local and remote bundle paths, source hashes, verifier and manifest status, classification, primary medians/deltas, stationarity rows, coverage summary, and next action to `AGENT_HANDOFF_STATE.md`.

State explicitly that this bundle does not establish:

- a host or GPU root cause;
- a stable 4K/16K/32K baseline;
- Proposal-KV offload benefit;
- real KV H2D reduction;
- a second model structure;
- TP1/TP4 promotion beyond the tested workload;
- Phase-1 completion; or
- Generic MTP/Speculative Runtime + Transactional KV Cache promotion.

Do not stage or commit the handoff update.

## Execution Boundary

This plan ends after one verified discovery bundle and result-dependent handoff. It does not authorize an automatic replication bundle, a cross-bundle established-effect artifact, runtime optimization, or Phase-1 promotion.
