# Staged Inference Benchmark Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents and new worktrees are intentionally excluded by the approved project constraints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce source-bound, independently verified Qwen3-0.6B Prefix Cache and Chunked Prefill performance gates, then promote exactly one verified Stage-1 winner to a fresh Qwen3-8B gate.

**Architecture:** Add one dependency-light contract module for frozen workloads, thresholds, classifications, and deterministic winner selection. Extend the existing Prefix Cache profiler and reuse the existing arrival-load driver for raw execution, while a new staged orchestrator owns immutable source/environment snapshots, isolated processes, remote admission, manifests, finalization, and downloads; a separate verifier reconstructs every result from raw artifacts without importing producer aggregation code.

**Tech Stack:** Python 3, TinyLLMForge `LLMEngine`, JSON/JSONL, SHA-256 manifests, `nvidia-smi`, SSH ControlMaster, dependency-light assertion scripts, Qwen3-0.6B, Qwen3-8B.

## Global Constraints

- The authoritative checkout is `/Users/bytedance/Desktop/TinyLLMForge`.
- Work only on `feat/kv-sparse-attention`; bind every remote source snapshot to the exact pushed `origin/feat/kv-sparse-attention` commit.
- Do not create a worktree or dispatch subagents.
- Stage exact paths only; never run `git add -A`.
- Commit with `git -c core.hooksPath=/dev/null commit`.
- Every commit must contain exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Push only to `origin/feat/kv-sparse-attention`.
- Remote execution host is `sitian@10.232.195.203`.
- Remote Python is `/data00/home/sitian/tllm/env/bin/python`.
- Qwen3-0.6B is `/data00/home/sitian/.ms_cache/Qwen/Qwen3-0___6B`.
- Qwen3-8B is `/data00/home/sitian/.ms_cache/Qwen/Qwen3-8B`.
- Every remote run, source snapshot, temporary directory, cache, log, manifest, and verification artifact must be below `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
- Do not write remote artifacts under `/`, `/tmp`, `/private/tmp`, or `/data00/home/sitian/tllm/TinyLLMForge`.
- The local SSH ControlMaster path may remain `/tmp/ssh-sitian-10.232.195.203`.
- Use the existing Kerberos cache only; do not refresh Kerberos automatically.
- Do not terminate or interfere with unrelated GPU processes.
- A selected GPU is admissible only when memory used is `<=1024 MiB`, utilization is `<=5%`, and it has no compute process.
- Every run tag is immutable. Preserve failed and partial tags; never resume by overwriting a tag.
- Do not add measured-path synchronization, `.item()`, logging, profiling, acknowledgements, fences, or GC controls solely for the gate.
- Prefix Stage 1 uses shared prefixes `256/1024/2048`, suffix `64`, batch `8`, two warmups, seven measured repetitions, and eager execution.
- Chunked Stage 1 compares only `OFF` against `FAIR_CHUNKED`, with eight warmup requests, 96 measured requests, and five paired repetitions.
- `OFF` sets `max_num_prefill_tokens_per_step=0`.
- `FAIR_CHUNKED` sets `max_num_prefill_tokens_per_step=128`, `chunked_prefill_decode_first=False`, `chunked_prefill_max_consecutive_chunks=2`, and all mixed/adaptive/SLO modes to `False`.
- Chunked prompt counts are exactly 58 short `64`-token, 24 medium `512`-token, and 14 long `4096`-token measured requests; outputs are balanced between `16` and `64` tokens within each prompt class.
- Chunked engine limits are `max_model_len=4352`, `max_num_batched_tokens=16384`, and `max_num_seqs=512`.
- Prefix `GO` and Chunked `GO` thresholds must be copied verbatim from `docs/superpowers/specs/2026-08-21-staged-inference-benchmark-evidence-design.md`.
- Every published result reports both benefit and cost.
- Do not call the current cache a radix tree, RadixAttention, or partial-block cache.
- Qwen3-0.6B results do not support Qwen3-8B claims. Stage 2 requires a fresh 8B gate.
- If neither Stage-1 feature is `GO`, do not run Stage 2.

---

## File Structure

- Create `tools/staged_inference_benchmark_contract.py`: frozen schemas, policies, deterministic workloads, metrics, gate classifications, and Stage-2 winner selection.
- Create `tools/test_staged_inference_benchmark_contract.py`: dependency-light RED/GREEN tests for every fixed shape, threshold, failure class, and tie-break.
- Modify `tools/profile_prefix_cache.py`: append-only raw Prefix evidence, memory/cache cost observations, p95 summaries, explicit incomplete/correct/no-go classifications, and contract-compatible report data.
- Modify `tools/test_profile_prefix_cache.py`: Prefix accounting, cost, classification, and artifact-tamper tests.
- Create `tools/staged_inference_benchmark_worker.py`: isolated Prefix or Chunked worker entrypoint using the existing engine/profiler/arrival driver.
- Create `tools/test_staged_inference_benchmark_worker.py`: fake-engine lifecycle, policy propagation, warmup exclusion, exact-output, and fail-closed worker tests.
- Create `tools/staged_inference_benchmark_gate.py`: source snapshot, environment capture, case launch, finalization, manifest hashing, promotion binding, and primary reports.
- Create `tools/test_staged_inference_benchmark_gate.py`: immutable identity, matrix, launch, finalization, and promotion tests.
- Create `tools/staged_inference_benchmark_verify.py`: independent reconstruction from raw artifacts without importing producer aggregation.
- Create `tools/test_staged_inference_benchmark_verify.py`: complete synthetic bundles plus source, workload, output, threshold, hash, and promotion tamper tests.
- Create `tools/run_staged_inference_benchmark_remote.py`: strict GPU admission, approved remote-root staging, detached execution, safe chunked downloads, controller verification, and model-tier commands.
- Create `tools/test_run_staged_inference_benchmark_remote.py`: remote path, GPU ownership, Kerberos, immutability, transport, and prohibited-operation contracts.
- Modify `AGENT_HANDOFF_STATE.md`: exact Stage-1/Stage-2 state and claim boundary.
- Modify `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`: evidence inventory and final classification.
- Modify `README.md` only if an independently verified `GO` supports exact benefit-and-cost wording.

### Shared interfaces

- `build_prefix_case_matrix(*, model_tier: str) -> list[dict]`
- `build_chunked_workload(*, seed: int = 20260821) -> list[dict]`
- `build_chunked_case_matrix(*, model_tier: str) -> list[dict]`
- `classify_prefix_bundle(raw: dict) -> dict`
- `classify_chunked_bundle(raw: dict) -> dict`
- `select_stage2_winner(prefix: dict, chunked: dict) -> dict`
- `run_worker(spec: dict, output_dir: Path) -> dict`
- `snapshot_source(repo_root: Path, output_dir: Path) -> dict`
- `initialize_run(run_dir: Path, gate_name: str, model_tier: str, run_tag: str, source_evidence: dict, environment_evidence: dict) -> dict`
- `run_cases(run_dir: Path, python_bin: str, model_path: str) -> dict`
- `finalize_run(run_dir: Path) -> dict`
- `verify_run(run_dir: Path, controller_dir: Path) -> dict`

## Task 1: Freeze the Shared Contract and Workloads

**Files:**
- Create: `tools/staged_inference_benchmark_contract.py`
- Create: `tools/test_staged_inference_benchmark_contract.py`

**Interfaces:**
- Produces: all shared interfaces listed above through `select_stage2_winner`.
- Consumes: only Python standard library values and raw dictionaries; no engine imports.

- [ ] **Step 1: Write failing workload-shape tests**

```python
def test_chunked_workload_has_exact_frozen_shape():
    rows = contract.build_chunked_workload()
    warmup = [row for row in rows if row["warmup"]]
    measured = [row for row in rows if not row["warmup"]]
    assert len(warmup) == 8
    assert len(measured) == 96
    assert Counter(row["prompt_tokens"] for row in measured) == {
        64: 58,
        512: 24,
        4096: 14,
    }
    for prompt_tokens in (64, 512, 4096):
        outputs = Counter(
            row["requested_output_tokens"]
            for row in measured
            if row["prompt_tokens"] == prompt_tokens
        )
        assert max(outputs.values()) - min(outputs.values()) <= 1
    assert [row["arrival_offset_ns"] for row in rows] == sorted(
        row["arrival_offset_ns"] for row in rows
    )
    assert len({row["request_id"] for row in rows}) == 104
```

```python
def test_case_matrices_are_exact_and_paired():
    prefix = contract.build_prefix_case_matrix(model_tier="qwen3-0.6b")
    assert len(prefix) == 15
    assert {
        (row["shape"], row["state"])
        for row in prefix
    } == {
        *{(f"single-{n}", state) for n in (256, 1024, 2048)
          for state in ("cold", "warm", "cache_cleared")},
        *{(f"batch8-{n}", state) for n in (1024, 2048)
          for state in ("cold", "warm", "cache_cleared")},
    }
    chunked = contract.build_chunked_case_matrix(model_tier="qwen3-0.6b")
    assert len(chunked) == 10
    assert Counter(row["policy"] for row in chunked) == {
        "OFF": 5,
        "FAIR_CHUNKED": 5,
    }
    assert {
        row["repetition"] for row in chunked
    } == set(range(5))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-contract-pycache \
python3 tools/test_staged_inference_benchmark_contract.py
```

Expected: import failure because `staged_inference_benchmark_contract.py` does not exist.

- [ ] **Step 3: Implement immutable constants and deterministic builders**

```python
PREFIX_POLICY = {
    "prefix_tokens": (256, 1024, 2048),
    "batch_prefix_tokens": (1024, 2048),
    "suffix_tokens": 64,
    "batch_size": 8,
    "warmup_repetitions": 2,
    "measured_repetitions": 7,
    "enforce_eager": True,
}

CHUNKED_POLICIES = {
    "OFF": {
        "max_num_prefill_tokens_per_step": 0,
        "chunked_prefill_decode_first": True,
        "chunked_prefill_max_consecutive_chunks": 0,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
    },
    "FAIR_CHUNKED": {
        "max_num_prefill_tokens_per_step": 128,
        "chunked_prefill_decode_first": False,
        "chunked_prefill_max_consecutive_chunks": 2,
        "chunked_prefill_mixed_batch": False,
        "chunked_prefill_adaptive_mixed": False,
        "chunked_prefill_slo_mixed": False,
    },
}

CHUNKED_ENGINE_CONFIG = {
    "max_model_len": 4352,
    "max_num_batched_tokens": 16384,
    "max_num_seqs": 512,
}

def build_chunked_workload(*, seed: int = 20260821) -> list[dict]:
    rng = random.Random(seed)
    measured_shapes = [64] * 58 + [512] * 24 + [4096] * 14
    rng.shuffle(measured_shapes)
    rows = []
    offset_ns = 0
    for index in range(104):
        warmup = index < 8
        prompt_tokens = (64, 512, 4096)[index % 3] if warmup else measured_shapes[index - 8]
        output_tokens = 16 if index % 2 == 0 else 64
        phase = "steady" if index < 40 else "burst" if index < 72 else "long_injection"
        if phase == "burst":
            offset_ns += 5_000_000
        elif phase == "long_injection":
            offset_ns += 40_000_000 if prompt_tokens == 4096 else 15_000_000
        else:
            offset_ns += 25_000_000
        rows.append({
            "request_id": f"{'warmup' if warmup else 'measured'}-{index:03d}",
            "warmup": warmup,
            "phase": phase,
            "arrival_offset_ns": offset_ns,
            "prompt_tokens": prompt_tokens,
            "requested_output_tokens": output_tokens,
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "max_tokens": output_tokens,
            },
            "starvation_deadline_ns": 30_000_000_000,
            "drain_timeout_ns": 180_000_000_000,
        })
    return rows
```

The final implementation must add canonical JSON hashing and validate the exact counts, monotonic arrivals, unique IDs, finite non-negative offsets, supported output lengths, and `prompt + output <= 4352`.

- [ ] **Step 4: Write failing classification and promotion tests**

```python
def test_prefix_classification_separates_invalid_no_go_and_go():
    raw = complete_prefix_fixture()
    assert contract.classify_prefix_bundle(raw)["classification"] == "PREFIX_CACHE_GO"
    raw["single"]["1024"]["warm"]["median_ttft_ms"] = 90.1
    assert contract.classify_prefix_bundle(raw)["classification"] == "PREFIX_CACHE_NO_GO"
    raw["single"]["1024"]["warm"]["exact_outputs"] = False
    assert contract.classify_prefix_bundle(raw)["classification"] == (
        "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    )

def test_chunked_classification_requires_four_of_five_directions():
    raw = complete_chunked_fixture(favorable_repetitions=4)
    assert contract.classify_chunked_bundle(raw)["classification"] == "FAIR_CHUNKED_GO"
    raw = complete_chunked_fixture(favorable_repetitions=3)
    assert contract.classify_chunked_bundle(raw)["classification"] == "FAIR_CHUNKED_NO_GO"

def test_stage2_winner_uses_frozen_tie_breaks():
    prefix = prefix_go(primary_benefit=0.22, worst_regression=1.02, memory=1.01)
    chunked = chunked_go(primary_benefit=0.22, worst_regression=1.02, memory=1.01)
    assert contract.select_stage2_winner(prefix, chunked)["winner"] == "prefix"
    assert contract.select_stage2_winner(
        prefix_no_go(), chunked_no_go()
    )["winner"] is None
```

- [ ] **Step 5: Implement classifications exactly as frozen**

Implement:

```python
def classify_prefix_bundle(raw: dict) -> dict:
    structural, correctness, performance = _prefix_failures(raw)
    if structural or correctness:
        classification = "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
    elif performance:
        classification = "PREFIX_CACHE_NO_GO"
    else:
        classification = "PREFIX_CACHE_GO"
    return {
        "classification": classification,
        "structural_failures": structural,
        "correctness_failures": correctness,
        "performance_failures": performance,
        "benefit": _prefix_benefit(raw),
        "cost": _prefix_cost(raw),
    }

def classify_chunked_bundle(raw: dict) -> dict:
    structural, correctness, performance = _chunked_failures(raw)
    if structural:
        classification = "FAIR_CHUNKED_INCOMPLETE"
    elif correctness:
        classification = "FAIR_CHUNKED_INCOMPLETE"
    elif performance:
        classification = "FAIR_CHUNKED_NO_GO"
    else:
        classification = "FAIR_CHUNKED_GO"
    return {
        "classification": classification,
        "structural_failures": structural,
        "correctness_failures": correctness,
        "performance_failures": performance,
        "benefit": _chunked_benefit(raw),
        "cost": _chunked_cost(raw),
    }
```

The exact checks are:

- Prefix exact output IDs/text, argmax equality, `max_abs <= 0.25`, `mean_abs <= 0.05`, exact token accounting, reusable ratio `1.0`, 1024/2048 TTFT improvement `>=20%`, both batch improvements `>=15%`, one warm model batch, cache-cleared regression `<=5%`, CUDA reserved regression `<=5%`, complete artifacts.
- Chunked exact outputs/lifecycle, zero dropped/rejected/truncated/unfinished/starved, short p99 TTFT improvement `>=10%`, short p99 ITL regression `<=5%`, maximum decode-gap regression `<=10%`, every service-class p95 completion regression `<=10%`, long p95 completion regression `<=10%`, both throughput regressions `<=3%`, CUDA reserved regression `<=5%`, favorable direction in at least four paired repetitions, complete artifacts.

- [ ] **Step 6: Run contract tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-contract-pycache \
python3 tools/test_staged_inference_benchmark_contract.py
```

Expected: `staged inference benchmark contract tests passed`.

- [ ] **Step 7: Commit the contract**

```bash
git add \
  tools/staged_inference_benchmark_contract.py \
  tools/test_staged_inference_benchmark_contract.py
git -c core.hooksPath=/dev/null commit -m "test(perf): freeze staged benchmark contract" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 2: Complete Prefix Raw Evidence and Cost Accounting

**Files:**
- Modify: `tools/profile_prefix_cache.py`
- Modify: `tools/test_profile_prefix_cache.py`

**Interfaces:**
- Consumes: `PREFIX_POLICY`, `classify_prefix_bundle`.
- Produces: `prefix_correctness_rows.jsonl`, `prefix_performance_rows.jsonl`, `prefix_cache_rows.jsonl`, `prefix_memory_rows.jsonl`, and `prefix_primary_summary.json`.

- [ ] **Step 1: Write failing p95, cache-cost, and incomplete tests**

```python
def test_prefix_summary_reports_p95_and_costs():
    rows = [
        prefix_row(ttft_ms=value, retained_blocks=4, retained_bytes=8192,
                   cuda_allocated=100, cuda_reserved=200, clear_host_ms=0.2)
        for value in (10, 11, 12, 13, 20, 21, 22)
    ]
    summary = summarize_case_rows(rows)
    assert summary["median_ttft_ms"] == 13
    assert summary["p95_ttft_ms"] == 22
    assert summary["peak_retained_reusable_blocks"] == 4
    assert summary["peak_retained_logical_kv_bytes"] == 8192
    assert summary["peak_cuda_reserved_bytes"] == 200
    assert summary["median_cache_clear_host_ms"] == 0.2

def test_prefix_missing_memory_row_is_incomplete():
    bundle = complete_prefix_bundle()
    bundle["memory_rows"].pop()
    result = decide_gate_from_bundle(bundle)
    assert result["classification"] == "PREFIX_CACHE_INCOMPLETE_OR_INCORRECT"
```

- [ ] **Step 2: Run focused tests RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-prefix-pycache \
python3 tools/test_profile_prefix_cache.py
```

Expected: failure because p95/cost fields and bundle classification are absent.

- [ ] **Step 3: Add observation helpers outside timed intervals**

```python
def cache_cost_observation(llm) -> dict:
    manager = llm.scheduler.block_manager
    retained_blocks = int(manager.num_cached_blocks)
    block_bytes = int(llm.model_runner.kv_cache.block_bytes)
    return {
        "retained_reusable_blocks": retained_blocks,
        "retained_logical_kv_bytes": retained_blocks * block_bytes,
    }

def cuda_memory_observation() -> dict:
    import torch
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

def clear_cache_observation(block_manager) -> dict:
    started_ns = time.perf_counter_ns()
    released_blocks = int(block_manager.clear_reusable_cache())
    elapsed_ns = time.perf_counter_ns() - started_ns
    return {
        "released_reusable_blocks": released_blocks,
        "cache_clear_host_ns": elapsed_ns,
    }
```

Call these only before or after `schedule_and_run_prefill*` timed regions. Do not insert extra synchronization or scalar extraction in the timed model path.

- [ ] **Step 4: Write append-only JSONL evidence and contract summary**

```python
def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                row, sort_keys=True, separators=(",", ":"), allow_nan=False
            ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
```

Each measured row must bind:

```python
{
    "schema_version": 2,
    "case_id": "single-1024__warm__r3",
    "shape": "single-1024",
    "state": "warm",
    "repetition": 3,
    "warmup": False,
    "prompt_token_ids_sha256": "0" * 64,
    "output_token_ids": [1, 2],
    "decoded_text": "sample",
    "ttft_ns": 123,
    "model_batches": 1,
    "cached_prompt_tokens": 1024,
    "executed_query_tokens": 64,
    "logit": {"argmax_match": True, "max_abs": 0.0, "mean_abs": 0.0},
    "retained_reusable_blocks": 4,
    "retained_logical_kv_bytes": 8192,
    "cuda_peak_allocated_bytes": 100,
    "cuda_peak_reserved_bytes": 200,
    "cache_clear_host_ns": 1000,
}
```

- [ ] **Step 5: Run Prefix tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-prefix-pycache \
python3 tools/test_profile_prefix_cache.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-prefix-pycache \
python3 tools/test_chunked_prefill.py
```

Expected: both scripts pass.

- [ ] **Step 6: Commit Prefix evidence changes**

```bash
git add tools/profile_prefix_cache.py tools/test_profile_prefix_cache.py
git -c core.hooksPath=/dev/null commit -m "feat(perf): record prefix gate costs" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 3: Add Isolated Stage Workers

**Files:**
- Create: `tools/staged_inference_benchmark_worker.py`
- Create: `tools/test_staged_inference_benchmark_worker.py`
- Modify: `tools/arrival_load_driver.py`
- Modify: `tools/test_arrival_load_driver.py`

**Interfaces:**
- Consumes: one immutable case spec and frozen workload JSONL.
- Produces: process-local append-only request, scheduler, memory, and result artifacts.

- [ ] **Step 1: Write failing policy and warmup tests**

```python
def test_chunked_worker_passes_only_frozen_policy_fields(tmp_path):
    seen = {}
    def engine_factory(spec):
        seen.update(spec["engine_config"])
        return FakeEngine(spec)
    run_worker(chunked_spec("FAIR_CHUNKED"), tmp_path, engine_factory=engine_factory)
    assert seen["max_num_prefill_tokens_per_step"] == 128
    assert seen["chunked_prefill_decode_first"] is False
    assert seen["chunked_prefill_max_consecutive_chunks"] == 2
    assert seen["chunked_prefill_mixed_batch"] is False
    assert seen["chunked_prefill_adaptive_mixed"] is False
    assert seen["chunked_prefill_slo_mixed"] is False

def test_worker_keeps_warmup_lifecycle_but_excludes_warmup_metrics(tmp_path):
    result = run_worker(chunked_spec("OFF"), tmp_path, engine_factory=FakeEngine)
    assert result["lifecycle_requests"] == 104
    assert result["measured_requests"] == 96
```

- [ ] **Step 2: Run worker tests RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-worker-pycache \
python3 tools/test_staged_inference_benchmark_worker.py
```

Expected: import failure because the worker module is absent.

- [ ] **Step 3: Implement worker dispatch**

```python
def run_worker(
    spec: dict,
    output_dir: Path,
    *,
    engine_factory=None,
) -> dict:
    validate_case_spec(spec)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if spec["gate"] == "prefix":
        return _run_prefix(spec, output_dir, engine_factory=engine_factory)
    if spec["gate"] == "chunked":
        return _run_chunked(spec, output_dir, engine_factory=engine_factory)
    raise ValueError(f"unsupported staged gate: {spec['gate']!r}")
```

For Chunked, adapt `arrival_load_driver.run_case` so the caller may supply a prevalidated workload list and exact engine config without adding producer aggregation. Preserve its existing append-only evidence, lifecycle validation, memory rows, and watchdog behavior.

- [ ] **Step 4: Add fail-closed worker cases**

Test exact failures for duplicate request IDs, wrong prompt counts, output mismatch, missing token timestamps, starvation, unfinished requests, unexpected policy fields, and non-finite memory/timing values.

- [ ] **Step 5: Run worker and existing driver tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-worker-pycache \
python3 tools/test_staged_inference_benchmark_worker.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-worker-pycache \
python3 tools/test_arrival_load_driver.py
```

Expected: both scripts pass.

- [ ] **Step 6: Commit worker support**

```bash
git add \
  tools/staged_inference_benchmark_worker.py \
  tools/test_staged_inference_benchmark_worker.py \
  tools/arrival_load_driver.py \
  tools/test_arrival_load_driver.py
git -c core.hooksPath=/dev/null commit -m "feat(perf): add staged benchmark workers" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 4: Build the Source-Bound Primary Orchestrator

**Files:**
- Create: `tools/staged_inference_benchmark_gate.py`
- Create: `tools/test_staged_inference_benchmark_gate.py`

**Interfaces:**
- Consumes: contract matrices, worker CLI, `tools/source_audit.py`.
- Produces: immutable primary bundles with `run_manifest.json`, raw rows, summary, report, hashes, and process receipts.

- [ ] **Step 1: Write failing identity and immutability tests**

```python
def test_initialize_binds_source_environment_workload_and_policy(tmp_path):
    manifest = gate.initialize_run(
        run_dir=tmp_path / "run",
        run_tag="qwen3-06b-prefix-r1",
        gate_name="prefix",
        model_tier="qwen3-0.6b",
        source_evidence=source_evidence(),
        environment_evidence=environment_evidence(),
    )
    assert len(manifest["source_tree_sha256"]) == 64
    assert len(manifest["environment_sha256"]) == 64
    assert len(manifest["workload_sha256"]) == 64
    assert len(manifest["policy_sha256"]) == 64

def test_existing_run_tag_is_never_overwritten(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    try:
        gate.initialize_run(
            run_dir=run_dir,
            run_tag="qwen3-06b-prefix-r1",
            gate_name="prefix",
            model_tier="qwen3-0.6b",
            source_evidence=source_evidence(),
            environment_evidence=environment_evidence(),
        )
    except ValueError as error:
        assert "already exists" in str(error)
    else:
        raise AssertionError("existing run directory must fail closed")
```

- [ ] **Step 2: Run gate tests RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-gate-pycache \
python3 tools/test_staged_inference_benchmark_gate.py
```

Expected: import failure because the gate module is absent.

- [ ] **Step 3: Implement deterministic source and environment identity**

Reuse `tools/source_audit.py` with an exact owned-root tuple containing the staged modules, Prefix profiler, arrival driver, scheduler/config/model files, and their tests. Reject dirty owned paths and require:

```python
local_head = git_output(repo_root, "rev-parse", "HEAD")
tracking_head = git_output(
    repo_root, "rev-parse", "origin/feat/kv-sparse-attention"
)
if local_head != tracking_head:
    raise ValueError("local HEAD must equal origin/feat/kv-sparse-attention")
```

Environment evidence must include Python, torch, CUDA, GPU UUID/name, model config hash, checkpoint identifier, engine limits, and the exact selected GPU inventory.

- [ ] **Step 4: Implement isolated case launch and atomic receipts**

Each case receives unique distributed ports and runs:

```python
command = [
    python_bin,
    "tools/staged_inference_benchmark_worker.py",
    "--case-spec", str(case_spec_path),
    "--workload", str(workload_path),
    "--output-dir", str(case_output),
]
```

Write `process.json`, `stdout.log`, `stderr.log`, and `exitcode` via temporary files followed by `Path.replace`. Never reuse a case directory. Alternate Chunked policy order by repetition:

```python
CHUNKED_POLICY_ORDER = {
    0: ("OFF", "FAIR_CHUNKED"),
    1: ("FAIR_CHUNKED", "OFF"),
    2: ("OFF", "FAIR_CHUNKED"),
    3: ("FAIR_CHUNKED", "OFF"),
    4: ("OFF", "FAIR_CHUNKED"),
}
```

- [ ] **Step 5: Implement primary finalization**

Merge process-local JSONL in manifest order, compare exact outputs within each paired repetition, reconstruct request/service-class metrics, call the shared contract classifier, and write:

```text
run_manifest.json
resolved_config.json
workload_manifest.jsonl
request_timeline.jsonl
scheduler_trace.jsonl
cache_trace.jsonl
memory_trace.jsonl
case_rows.jsonl
summary.json
report.md
primary_verification_receipt.json
artifact_hashes.json
manifest.sha256
```

The report must contain:

```markdown
| Benefit | Cost |
| --- | --- |
| <primary benefit and absolute/relative value> | <worst protected metric, memory, retained-KV, or fairness cost> |
```

- [ ] **Step 6: Implement Stage-2 promotion binding**

The 8B initializer requires two independently verified Stage-1 summaries and records:

```python
{
    "promotion": {
        "winner": "prefix" or "chunked",
        "prefix_summary_sha256": "a" * 64,
        "chunked_summary_sha256": "b" * 64,
        "selection_rule": {
            "primary_benefit": "larger normalized primary benefit",
            "worst_protected_regression": "smaller ratio wins",
            "peak_cuda_reserved_regression": "smaller ratio wins",
            "exact_tie": "prefix",
        },
    }
}
```

Reject promotion when neither summary is `GO`, when the selected policy differs from `select_stage2_winner`, or when the model tier is not `qwen3-8b`.

- [ ] **Step 7: Run gate tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-gate-pycache \
python3 tools/test_staged_inference_benchmark_gate.py
```

Expected: `staged inference benchmark gate tests passed`.

- [ ] **Step 8: Commit the primary orchestrator**

```bash
git add \
  tools/staged_inference_benchmark_gate.py \
  tools/test_staged_inference_benchmark_gate.py
git -c core.hooksPath=/dev/null commit -m "feat(perf): orchestrate staged evidence gates" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 5: Add the Independent Verifier

**Files:**
- Create: `tools/staged_inference_benchmark_verify.py`
- Create: `tools/test_staged_inference_benchmark_verify.py`

**Interfaces:**
- Consumes: only immutable artifacts and Python standard library.
- Produces: controller-side independently rebuilt summary/report/receipt.

- [ ] **Step 1: Write a complete synthetic bundle and tamper tests**

```python
def test_verifier_rebuilds_complete_prefix_bundle(tmp_path):
    run_dir = write_complete_prefix_bundle(tmp_path)
    result = verifier.verify_run(run_dir, tmp_path / "controller")
    assert result["classification"] == "PREFIX_CACHE_GO"

def test_verifier_fails_closed_on_rehashed_tamper():
    mutations = (
        tamper_source_hash,
        tamper_workload,
        tamper_output_token,
        tamper_cached_tokens,
        tamper_ttft_summary,
        tamper_cuda_reserved,
        truncate_jsonl,
        duplicate_case,
    )
    for mutation in mutations:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = write_complete_chunked_bundle(root)
            mutation(run_dir)
            refresh_manifest_hashes(run_dir)
            try:
                verifier.verify_run(run_dir, root / "controller")
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f"verifier accepted tamper: {mutation.__name__}"
                )
```

- [ ] **Step 2: Run verifier tests RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-verify-pycache \
python3 tools/test_staged_inference_benchmark_verify.py
```

Expected: import failure because the verifier module is absent.

- [ ] **Step 3: Implement independent parsing and recomputation**

The verifier must not import:

```python
FORBIDDEN_IMPORTS = {
    "tools.staged_inference_benchmark_gate",
    "tools.profile_prefix_cache",
    "tools.arrival_load_gate",
}
```

It independently:

1. verifies `manifest.sha256` and every file hash;
2. safely extracts and hashes the source snapshot;
3. checks source commit, tree hash, environment, model, GPU, config, workload, ports, and case matrix;
4. validates JSONL final newlines, schemas, uniqueness, finite values, and lifecycle ordering;
5. recomputes Prefix and Chunked metrics directly from raw rows;
6. duplicates the frozen thresholds from the approved spec;
7. compares its summary and report byte-for-byte with the primary outputs; and
8. writes controller artifacts only after agreement.

- [ ] **Step 4: Publish controller receipt atomically**

```python
receipt = {
    "status": "PASS",
    "run_manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
    "primary_summary_sha256": sha256_file(run_dir / "summary.json"),
    "controller_summary_sha256": sha256_file(controller / "summary.json"),
    "classification": computed["classification"],
}
atomic_write_json(controller / "verification_receipt.json", receipt)
atomic_write_text(controller / "verify.exitcode", "0\n")
```

- [ ] **Step 5: Run verifier tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-verify-pycache \
python3 tools/test_staged_inference_benchmark_verify.py
```

Expected: `staged inference benchmark verifier tests passed`.

- [ ] **Step 6: Commit the independent verifier**

```bash
git add \
  tools/staged_inference_benchmark_verify.py \
  tools/test_staged_inference_benchmark_verify.py
git -c core.hooksPath=/dev/null commit -m "feat(perf): verify staged evidence independently" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 6: Add the Safe Remote Runner

**Files:**
- Create: `tools/run_staged_inference_benchmark_remote.py`
- Create: `tools/test_run_staged_inference_benchmark_remote.py`

**Interfaces:**
- Consumes: pushed source commit, model tier, gate, immutable run tag.
- Produces: remote primary/controller bundles and local downloaded copies.

- [ ] **Step 1: Write failing remote policy tests**

```python
def test_remote_paths_are_all_below_approved_root():
    source = RUNNER_PATH.read_text()
    assert "/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818" in source
    for forbidden in (
        "sitian-workspace01",
        "/data00/home/sitian/tllm/TinyLLMForge",
        "TMPDIR=/tmp",
        "TMPDIR=/private/tmp",
    ):
        assert forbidden not in source

def test_gpu_admission_is_strict_and_non_destructive():
    source = RUNNER_PATH.read_text()
    assert "memory_used_mib <= 1024" in source
    assert "utilization_percent <= 5" in source
    assert "not compute_processes" in source
    for forbidden in ("pkill", "killall", "nvidia-smi --gpu-reset"):
        assert forbidden not in source
```

- [ ] **Step 2: Run runner tests RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-remote-pycache \
python3 tools/test_run_staged_inference_benchmark_remote.py
```

Expected: import/file failure because the runner is absent.

- [ ] **Step 3: Implement commands and preflight**

CLI:

```text
preflight
execute
download-only
verify-local
```

Required arguments:

```text
--gate prefix|chunked
--model-tier qwen3-0.6b|qwen3-8b
--run-tag <immutable-tag>
--promotion-prefix-run <path>   # 8B only
--promotion-chunked-run <path>  # 8B only
```

`preflight` must:

- validate `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian` without refreshing it;
- require local `HEAD == origin/feat/kv-sparse-attention`;
- reject existing remote primary/controller paths;
- select exactly one strict-clean GPU for Stage 1;
- select the model-specific required GPU count only after an explicit capacity preflight for Stage 2;
- record all GPU rows and reject ownership changes before and after every process;
- verify remote Python/model paths and approved-root free space.

- [ ] **Step 4: Implement immutable staging and detached execution**

Upload a deterministic source snapshot to:

```text
<approved-root>/staged-benchmark/staging/<run-tag>
```

Execute under:

```text
<approved-root>/staged-benchmark/runs/<run-tag>
<approved-root>/staged-benchmark/controller-verification/<run-tag>
```

Set all of:

```bash
TMPDIR=<run-root>/tmp
TEMP=<run-root>/tmp
TMP=<run-root>/tmp
PYTHONPYCACHEPREFIX=<run-root>/pycache
HF_HOME=<run-root>/hf-home
TORCH_EXTENSIONS_DIR=<run-root>/torch-extensions
```

Do not use `rsync`; stream deterministic tar input and chunk-download files with explicit sizes, retries, path validation, and SHA-256 verification.

- [ ] **Step 5: Add source-bound remote and local verification**

Remote completion order:

```text
worker cases
→ primary finalization
→ remote independent verifier
→ manifest freeze
→ controller copy
→ controller verifier
→ receipt comparison
→ local download
→ local independent verifier
```

Any disagreement is `INCOMPLETE`; preserve all available artifacts.

- [ ] **Step 6: Run runner tests GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-remote-pycache \
python3 tools/test_run_staged_inference_benchmark_remote.py
```

Expected: `staged inference benchmark remote runner tests passed`.

- [ ] **Step 7: Commit the remote runner**

```bash
git add \
  tools/run_staged_inference_benchmark_remote.py \
  tools/test_run_staged_inference_benchmark_remote.py
git -c core.hooksPath=/dev/null commit -m "feat(perf): run staged gates remotely" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Task 7: Run the Full Local Verification Matrix

**Files:**
- No production edits unless a test exposes a defect.

- [ ] **Step 1: Run all focused staged tests**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_staged_inference_benchmark_contract.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_profile_prefix_cache.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_staged_inference_benchmark_worker.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_staged_inference_benchmark_gate.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_staged_inference_benchmark_verify.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_run_staged_inference_benchmark_remote.py
```

Expected: every script passes.

- [ ] **Step 2: Run reused subsystem regressions**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_chunked_prefill.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_arrival_load_driver.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_arrival_load_gate.py
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 tools/test_arrival_load_verify.py
```

Expected: every script passes with no existing policy behavior changes.

- [ ] **Step 3: Run syntax and whitespace verification**

```bash
PYTHONPYCACHEPREFIX=/tmp/tinyllmforge-staged-all-pycache \
python3 -m py_compile \
  tools/staged_inference_benchmark_contract.py \
  tools/profile_prefix_cache.py \
  tools/staged_inference_benchmark_worker.py \
  tools/staged_inference_benchmark_gate.py \
  tools/staged_inference_benchmark_verify.py \
  tools/run_staged_inference_benchmark_remote.py
git diff --check
```

Expected: no output and exit code 0.

## Task 8: Execute Qwen3-0.6B Prefix and Chunked Gates

**Files:**
- Remote and local run artifacts only; do not stage experiment bundles.

- [ ] **Step 1: Run Prefix preflight with a fresh immutable tag**

```bash
export KRB5CCNAME='FILE:/Users/bytedance/krb5cc_sitian'
python3 tools/run_staged_inference_benchmark_remote.py preflight \
  --gate prefix \
  --model-tier qwen3-0.6b \
  --run-tag 20260821-qwen3-06b-prefix-r1
```

Expected: `READY` with one strict-clean GPU, exact pushed source SHA, absent primary/controller paths, and approved remote roots.

- [ ] **Step 2: Execute and verify Prefix**

```bash
python3 tools/run_staged_inference_benchmark_remote.py execute \
  --gate prefix \
  --model-tier qwen3-0.6b \
  --run-tag 20260821-qwen3-06b-prefix-r1
```

Expected: primary, remote verifier, controller verifier, and local verifier agree on one of `PREFIX_CACHE_GO`, `PREFIX_CACHE_NO_GO`, or `PREFIX_CACHE_INCOMPLETE_OR_INCORRECT`.

- [ ] **Step 3: Run Chunked preflight with a new tag**

```bash
python3 tools/run_staged_inference_benchmark_remote.py preflight \
  --gate chunked \
  --model-tier qwen3-0.6b \
  --run-tag 20260821-qwen3-06b-fair-chunked-r1
```

Expected: `READY` under the same strict admission and source-bound rules.

- [ ] **Step 4: Execute and verify Chunked**

```bash
python3 tools/run_staged_inference_benchmark_remote.py execute \
  --gate chunked \
  --model-tier qwen3-0.6b \
  --run-tag 20260821-qwen3-06b-fair-chunked-r1
```

Expected: all four verification layers agree on `FAIR_CHUNKED_GO`, `FAIR_CHUNKED_NO_GO`, or `FAIR_CHUNKED_INCOMPLETE`.

- [ ] **Step 5: Preserve failures and rerun only with a fresh tag**

If either run is incomplete due to environmental interruption, diagnose from preserved artifacts and use `r2`, `r3`, and so on. Never delete or overwrite `r1`.

## Task 9: Promote Exactly One Stage-1 Winner to Qwen3-8B

**Files:**
- Remote and local Stage-2 artifacts only.

- [ ] **Step 1: Compute eligibility before GPU admission**

```bash
python3 tools/staged_inference_benchmark_gate.py select-winner \
  --prefix-run experiments/staged_inference/20260821-qwen3-06b-prefix-r1 \
  --chunked-run experiments/staged_inference/20260821-qwen3-06b-fair-chunked-r1
```

Expected:

- one deterministic winner when at least one Stage-1 gate is `GO`; or
- `winner: null` and no Stage-2 launch when neither is `GO`.

- [ ] **Step 2: Run 8B preflight for only the winner**

For Prefix:

```bash
python3 tools/run_staged_inference_benchmark_remote.py preflight \
  --gate prefix \
  --model-tier qwen3-8b \
  --run-tag 20260821-qwen3-8b-prefix-r1 \
  --promotion-prefix-run experiments/staged_inference/20260821-qwen3-06b-prefix-r1 \
  --promotion-chunked-run experiments/staged_inference/20260821-qwen3-06b-fair-chunked-r1
```

For Chunked, substitute `--gate chunked` and `20260821-qwen3-8b-fair-chunked-r1`.

Expected: preflight freezes any required capacity-driven shape scaling before model results are visible.

- [ ] **Step 3: Execute the 8B gate**

Use the same command with `execute`. Require at least one warmup and five measured repetitions, unchanged metrics/thresholds, a new source/environment/workload manifest, and independent verification.

- [ ] **Step 4: Enforce claim boundary**

Only a completed 8B `GO` supports an 8B performance sentence. An 8B `NO_GO` or incomplete run is documented as implementation/correctness evidence with no speedup claim.

## Task 10: Update Audit, Handoff, and Final Claim

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify: `docs/superpowers/audits/2026-08-16-phase1-completion-audit.md`
- Modify only after verified `GO`: `README.md`

- [ ] **Step 1: Record exact evidence levels**

Document separately:

- local unit/contract tests;
- source-bound remote execution;
- primary finalizer;
- remote independent verifier;
- controller verifier;
- local verifier;
- Stage-1 Prefix classification;
- Stage-1 Chunked classification;
- Stage-2 eligibility and result;
- active TP4 schedstat diagnostic status.

- [ ] **Step 2: Write benefit-plus-cost wording**

For Prefix `GO`:

```text
On the frozen Qwen3-0.6B workload, hash-based full-block prefix reuse
reduced executed prefill tokens by X and median TTFT by Y, while
cache-cleared TTFT changed by Z and retained K logical KV bytes.
```

For Chunked `GO`:

```text
On the frozen Qwen3-0.6B mixed-arrival workload, bounded fair chunking
changed short-request p99 TTFT by X, with throughput change Y and
long-request p95 completion change Z.
```

Use the equivalent 8B sentence only after the fresh 8B gate passes.

- [ ] **Step 3: Run final documentation checks**

```bash
rg -n "Radix|RadixAttention|speedup|improved|optimization" \
  README.md AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
git diff --check
```

Expected: every performance word is supported by a verified gate and every Prefix description says hash-based full-block reuse.

- [ ] **Step 4: Commit and push exact documentation paths**

```bash
git add \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/audits/2026-08-16-phase1-completion-audit.md
if git diff --quiet -- README.md; then
  :
else
  git add README.md
fi
git -c core.hooksPath=/dev/null commit -m "docs(perf): record staged benchmark evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

## Plan Self-Review

- Spec coverage: Tasks 1-10 cover every frozen Prefix shape, Chunked policy/workload/threshold, benefit-plus-cost report, source/environment binding, dual verification, manifest, immutable tag, and deterministic 8B promotion requirement.
- Scope separation: Prefix and Chunked workers are independently testable; shared code is limited to immutable contracts and orchestration primitives. Historical P0/P4/P5 classification is not reused.
- Placeholder scan: the plan contains no `TBD`, `TODO`, “implement later”, or unspecified error-handling steps.
- Type consistency: contract builders return `list[dict]`; classifiers and winner selection return `dict`; workers and orchestrators consume those exact shapes.
- Claim boundary: no Qwen3-0.6B result is transferred to Qwen3-8B, and no `NO_GO` or incomplete result is called an optimization.
- Remote storage: every remote artifact/caching path is rooted below the approved `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818`.
