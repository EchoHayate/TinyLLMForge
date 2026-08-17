# Qwen3.5 Generic Speculative TP4/16K Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` and execute inline in the current checkout.
> Subagents, worktrees, staging, commits, pushes, stashes, resets, and cleans
> are prohibited for this plan.

**Goal:** Build and execute an independent, source-bound Qwen3.5 TP4/16K
performance gate comparing baseline greedy decoding with the generic n-gram
speculative runtime.

**Architecture:** A pure gate module owns constants, schema validation,
aggregation, direction/ratio derivation, subprocess orchestration, source
hashing, and failure artifacts. A loaded TP4 worker owns one policy/batch
Engine, repeated synchronized runs, real rank-wise KV movement, peak-memory
evidence, and cleanup. A bounded remote runner selects a fixed four-GPU set,
records capacity stability, uploads an isolated source archive, runs the gate,
downloads artifacts, and invokes an independent verifier.

**Tech Stack:** Python 3.11, PyTorch/CUDA, TinyLLMForge `LLMEngine`, TP4
`torch.distributed`/NCCL, `KVOffloadMVP0`, Bash, SSH/rsync, JSON/SHA-256,
pytest.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify or parameterize the established 4K/16K/32K correctness gates.
- Do not modify the existing TP1 performance gate or its authority.
- Use only `sitian@10.232.195.203` for remote GPU execution.
- Export `KRB5CCNAME=FILE:/Users/bytedance/krb5cc_sitian`.
- Use `ControlMaster=no`, `ControlPath=none`, serial SSH, finite retries, and
  bounded polling.
- Use real loaded-model KV-offload summaries; never substitute synthetic
  tensor copies or helper-only microbenchmarks.
- Do not replay accepted prefixes through a second full-model forward.
- Preserve exact greedy output parity before interpreting performance.
- Every behavior change follows RED -> minimal GREEN -> regression.
- Use `apply_patch` for every file edit.
- Keep Phase 1 classified `NOT_PROMOTABLE`.
- Do not claim 32K performance, learned-drafter/MTP performance, KV4/KV8,
  statistical significance, or production readiness.

---

### Task 1: Freeze Pure Campaign and Schema Contracts

**Files:**
- Create:
  `tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py`
- Create:
  `tools/qwen35_generic_speculative_tp4_16k_performance_gate.py`

**Interfaces:**
- Produces constants:
  `SCHEMA_VERSION`, `CLASSIFICATION`, `WORLD_SIZE`, `PROMPT_TOKENS`,
  `MAX_OUTPUT_TOKENS`, `BATCH_SIZES`, `POLICIES`, `WARMUP_RUNS`,
  `PARITY_RUNS`, `MEASURED_RUNS`, `NGRAM_SIZE`,
  `MAX_PROPOSAL_TOKENS`, `REAL_MOVEMENT_KEYS`, `DEFAULT_SOURCE_FILES`.
- Produces pure helpers:
  `cell_key(policy: str, batch_size: int) -> str`,
  `subtract_counter_summaries(...) -> dict[str, int]`,
  `build_run_metrics(...) -> dict`,
  `aggregate_measurements(values: list[float]) -> dict`,
  `classify_batch_direction(baseline: dict, candidate: dict) -> str`.

- [ ] **Step 1: Write failing constant and helper tests**

Add tests that require:

```python
assert gate.SCHEMA_VERSION == (
    "qwen35.generic-speculative-tp4-16k-performance.v1"
)
assert gate.CLASSIFICATION == (
    "SECOND_MODEL_TP4_16K_PERFORMANCE_MEASURED"
)
assert gate.WORLD_SIZE == 4
assert gate.PROMPT_TOKENS == 16384
assert gate.MAX_OUTPUT_TOKENS == 64
assert gate.BATCH_SIZES == (1, 4)
assert gate.POLICIES == ("baseline", "ngram")
assert gate.WARMUP_RUNS == 1
assert gate.PARITY_RUNS == 1
assert gate.MEASURED_RUNS == 5
```

Add focused tests copied semantically from the frozen TP1 gate for:

```python
subtract_counter_summaries()
build_run_metrics()
aggregate_measurements()
classify_batch_direction()
```

The tests must import only the new module.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m pytest -q \
  tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py
```

Expected: collection fails because
`qwen35_generic_speculative_tp4_16k_performance_gate.py` does not exist.

- [ ] **Step 3: Implement the minimal pure module**

Load the frozen TP1 performance gate under a private module name and re-export
only its pure helpers:

```python
_frozen = _load_module(
    "_qwen35_tp4_16k_frozen_performance_helpers",
    tools / "speculative_runtime_performance_gate.py",
)

subtract_counter_summaries = _frozen.subtract_counter_summaries
build_run_metrics = _frozen.build_run_metrics
aggregate_measurements = _frozen.aggregate_measurements
classify_batch_direction = _frozen.classify_batch_direction
```

Define independent constants and source files. Do not mutate the frozen
module's globals.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2.

Expected: all Task 1 tests pass.

---

### Task 2: Validate Rank-Wise Movement, Memory, Runtime, and Run Rows

**Files:**
- Modify:
  `tools/qwen35_generic_speculative_tp4_16k_performance_gate.py`
- Modify:
  `tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py`

**Interfaces:**
- Produces:
  `validate_movement(value: object) -> dict`,
  `validate_memory(value: object) -> dict`,
  `validate_runtime(value: object, *, policy: str) -> dict`,
  `validate_run(value: object, *, policy: str, batch_size: int) -> dict`.

- [ ] **Step 1: Write failing rank-inventory tests**

Create a four-rank movement fixture with exact totals and assert:

```python
normalized = gate.validate_movement(fixture)
assert [row["rank"] for row in normalized["ranks"]] == [0, 1, 2, 3]
assert normalized["totals"]["h2d_bytes"] == expected_h2d
```

Add rejection tests for:

```text
three or five ranks
duplicate/missing rank
negative counter
totals that differ from rank sums
nonzero rejected speculative D2H copies
```

Create four-rank reset/final memory fixtures and require maximum peak and
peak-minus-reset calculations. Add runtime fixtures that require positive
proposal/accepted/callback evidence for candidate runs and all-zero
speculative evidence for baseline runs.

Create a complete run fixture and reject:

```text
wrong batch output count
not exactly 64 tokens per output
missing timing evidence
incomplete four-rank movement
incomplete four-rank memory
```

- [ ] **Step 2: Run focused tests and verify RED**

Run only the new validator tests.

Expected: fail because the validators are absent.

- [ ] **Step 3: Implement minimal validators**

Reuse frozen pure scalar validation where useful, but enforce `WORLD_SIZE=4`
in the new module. Recompute every aggregate from raw rank rows and reject
stored totals that disagree.

`validate_runtime()` must calculate acceptance as:

```python
accepted_draft_tokens / proposed_tokens
```

and compare it with any stored acceptance rate.

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: all Task 2 tests pass.

---

### Task 3: Aggregate Five Runs and Derive Honest Direction Ratios

**Files:**
- Modify:
  `tools/qwen35_generic_speculative_tp4_16k_performance_gate.py`
- Modify:
  `tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py`

**Interfaces:**
- Produces:
  `validate_worker_result(value: object) -> dict`,
  `aggregate_worker(worker: dict) -> dict`,
  `derive_comparison(cells: dict, batch_size: int) -> dict`,
  `derive_artifact(worker_results: list[dict]) -> dict`.

- [ ] **Step 1: Write failing worker/aggregation tests**

Build deterministic baseline and candidate worker fixtures with exactly:

```text
1 warmup run
1 parity run
5 measured runs
```

Require raw distributions for:

```text
TTFT
TPOT
completion latency
batch token throughput
request throughput
peak allocated/reserved bytes
movement totals
runtime totals and acceptance
```

Require per-batch comparison fields:

```python
{
    "direction": "IMPROVED",
    "tpot_ratio": candidate_tpot / baseline_tpot,
    "tpot_percent_delta": (ratio - 1.0) * 100.0,
    "throughput_ratio": candidate_throughput / baseline_throughput,
    "ttft_ratio": ...,
    "peak_allocated_ratio": ...,
    "h2d_bytes_ratio": ...,
    "d2h_bytes_ratio": ...,
}
```

Add rejection tests for:

```text
duplicate/missing policy/batch cells
wrong run counts
prompt mismatch
any parity/measured output mismatch
candidate with zero proposals or accepted tokens
batch-4 baseline/candidate without positive H2D and D2H
```

- [ ] **Step 2: Run focused tests and verify RED**

Expected: fail because aggregation and comparison helpers are absent.

- [ ] **Step 3: Implement minimal aggregation**

Aggregate the five measured runs exactly as the design specifies. Derive
campaign direction:

```python
POSITIVE if both batches are IMPROVED
NEGATIVE if either batch is REGRESSED
MIXED otherwise
```

Do not convert `POSITIVE` into Phase 1 promotion.

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: all Task 3 tests pass.

---

### Task 4: Implement the Loaded TP4 Repeated-Run Worker

**Files:**
- Create:
  `tools/qwen35_generic_speculative_tp4_16k_performance_worker.py`
- Modify:
  `tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py`

**Interfaces:**
- Produces:
  `movement_delta(before_rows, after_rows) -> dict`,
  `memory_result(reset_rows, final_rows) -> dict`,
  `run_request_batch(...) -> dict`,
  `run_policy_campaign(...) -> dict`.
- Consumes deterministic prompts from:
  `qwen35_generic_speculative_tp4_16k_gate.build_prompt_rows`.
- Consumes distributed cleanup helpers from:
  `qwen35_generic_speculative_tp4_worker.py`.

- [ ] **Step 1: Write failing worker unit tests**

Use a fake rank-aware Engine that records this exact run order:

```text
idle check
clear reusable prefix cache
before movement summaries
peak reset
synchronize
add all requests
step/synchronize until finished
after movement summaries
final memory snapshots
```

Assert the worker:

- never calls a manual eviction or upload helper;
- computes token-event TTFT/TPOT from synchronized step boundaries;
- returns four-rank movement and memory evidence;
- executes exactly 1/1/5 run groups;
- activates `EngineSpeculativeRuntime(NGramDraftAdapter)` only for `ngram`;
- constructs the Engine with TP4, 16K, blockwise offload, and 48 physical
  blocks;
- calls `engine.exit()` in `finally`.

- [ ] **Step 2: Run worker tests and verify RED**

Expected: fail because the worker module is absent.

- [ ] **Step 3: Implement minimal worker**

Load the 16K correctness gate and frozen correctness worker under private
module names. Use its distributed environment and cleanup collection around
one Engine per cell.

Implement:

```python
run_request_batch(
    engine,
    prompt_rows,
    sampling_params,
    synchronize,
    clock_ns,
)
```

without forced eviction. Build token events from
`last_step_observation["new_completion_tokens_by_seq"]`.

Implement `run_policy_campaign()` with:

```python
warmup_runs = [run_once()]
parity_runs = [run_once()]
measured_runs = [run_once() for _ in range(5)]
```

Return cleanup receipts even when the Engine raises.

- [ ] **Step 4: Run worker tests and verify GREEN**

Expected: all Task 4 tests pass.

---

### Task 5: Implement Source-Bound Artifact Assembly and Failure Retention

**Files:**
- Modify:
  `tools/qwen35_generic_speculative_tp4_16k_performance_gate.py`
- Modify:
  `tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py`

**Interfaces:**
- Produces:
  `hash_source_files(repo_root, source_files) -> dict[str, str]`,
  `source_tree_sha256(repo_root, source_files) -> str`,
  `build_performance_artifact(...) -> dict`,
  `validate_performance_artifact(value: object) -> dict`,
  `run_campaign(...) -> dict`.

- [ ] **Step 1: Write failing artifact and campaign tests**

Require the artifact to contain:

```text
schema/status/classification/claim scope/limitations
model manifest and source-tree digests
world size and physical GPU indices
fixed campaign and engine configuration
raw four-cell worker results
five-run aggregates
batch comparisons and campaign direction
environment and GPU inventory snapshots
cleanup receipts
source file hashes
```

Require `validate_performance_artifact()` to recompute aggregates, ratios,
directions, rank totals, parity, and source identity from raw rows.

Use a fake worker runner to assert command order:

```text
baseline:b1
ngram:b1
baseline:b4
ngram:b4
```

Require atomic success publication and atomic rename to
`authority.failed` on any worker or validation failure.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: fail because artifact assembly/campaign functions are absent.

- [ ] **Step 3: Implement minimal campaign**

Use fresh distributed/master ports per cell. Store worker logs beside worker
JSON. Validate each worker immediately, then assemble the final artifact only
after all four cells pass.

Bind these source files:

```text
tinyvllm/engine/llm_engine.py
tinyvllm/engine/model_runner.py
tinyvllm/engine/speculative_runtime.py
tinyvllm/engine/speculative_side_state.py
tinyvllm/speculative/batch_runtime.py
tinyvllm/speculative/ngram_adapter.py
tinyvllm/layers/qwen35_full_attention.py
tools/speculative_runtime_performance_gate.py
tools/qwen35_generic_speculative_tp4_gate.py
tools/qwen35_generic_speculative_tp4_worker.py
tools/qwen35_generic_speculative_tp4_16k_gate.py
tools/qwen35_generic_speculative_tp4_16k_performance_gate.py
tools/qwen35_generic_speculative_tp4_16k_performance_worker.py
tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Expected: all Task 5 tests pass.

---

### Task 6: Add the Independent Verifier

**Files:**
- Create:
  `tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py`
- Modify:
  `tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py`

**Interfaces:**
- Produces:
  `verify_run(authority_path: Path, source_root: Path) -> dict`.

- [ ] **Step 1: Write failing verifier mutation tests**

Create a valid temporary authority and require PASS. Mutate one field at a
time and require a named failure for:

```text
source hash
source tree digest
model manifest
raw timing
aggregate median
ratio
batch direction
campaign direction
prompt/output parity
rank movement total
rejected speculative D2H
cleanup receipt
```

- [ ] **Step 2: Run verifier tests and verify RED**

Expected: fail because the verifier is absent.

- [ ] **Step 3: Implement minimal verifier**

Load the gate module by path, validate the artifact, recompute source hashes
from `source_root`, compare the approved model digest, and return:

```json
{"classification":"PASS","failures":[]}
```

or a deterministic failure list.

- [ ] **Step 4: Run verifier tests and verify GREEN**

Expected: all Task 6 tests pass.

---

### Task 7: Add the Bounded Remote Runner

**Files:**
- Create:
  `tools/run_qwen35_generic_speculative_tp4_16k_performance_gate_remote.sh`
- Modify:
  `tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py`

**Interfaces:**
- Produces an opaque local run directory:
  `artifacts/qwen35_generic_speculative_tp4_16k_performance/opaque-*/`.

- [ ] **Step 1: Write failing runner contract tests**

Require the shell source to contain:

```text
sitian@10.232.195.203
FILE:/Users/bytedance/krb5cc_sitian
ControlMaster=no
ControlPath=none
fixed four-GPU selection
48 GiB pre-cell minimum
10 percent utilization ceiling
4 GiB post-cell drift ceiling
finite SSH/rsync retries
opaque run ID
campaign.status / campaign.pid / campaign.exit_code
authority.failed retention
remote and local independent verifier calls
```

Reject persistent control sockets and unbounded retry loops.

- [ ] **Step 2: Run runner tests and verify RED**

Expected: fail because the runner is absent.

- [ ] **Step 3: Implement minimal bounded runner**

Archive the complete `tinyvllm` tree and explicit gate/worker/verifier source
set into an isolated remote source directory. Select four GPUs once, allocate
fresh non-ephemeral distributed/master port ranges, launch one background
campaign, poll terminal status, copy the complete run directory, and run the
local verifier.

Do not mutate a shared remote checkout.

- [ ] **Step 4: Run runner tests and verify GREEN**

Run:

```bash
bash -n \
  tools/run_qwen35_generic_speculative_tp4_16k_performance_gate_remote.sh

python3 -m pytest -q \
  tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py
```

Expected: shell syntax passes and all performance-gate tests pass.

---

### Task 8: Run Local Regression and Real TP4/16K Campaign

**Files:**
- Generated:
  `artifacts/qwen35_generic_speculative_tp4_16k_performance/opaque-*/`
- Modify after successful audit:
  `docs/superpowers/audits/2026-08-12-phase1-objective-coverage.md`
  `AGENT_HANDOFF_STATE.md`

- [ ] **Step 1: Run local static and contract validation**

Run:

```bash
python3 -m py_compile \
  tools/qwen35_generic_speculative_tp4_16k_performance_gate.py \
  tools/qwen35_generic_speculative_tp4_16k_performance_worker.py \
  tools/verify_qwen35_generic_speculative_tp4_16k_performance_gate.py

bash -n \
  tools/run_qwen35_generic_speculative_tp4_16k_performance_gate_remote.sh

python3 -m pytest -q \
  tools/test_qwen35_generic_speculative_tp4_16k_performance_gate.py \
  tools/test_qwen35_generic_speculative_tp4_16k_gate.py \
  tools/test_qwen35_generic_speculative_tp4_gate.py

git diff --check
```

Expected: all commands pass.

- [ ] **Step 2: Run the bounded remote campaign**

Run:

```bash
bash \
  tools/run_qwen35_generic_speculative_tp4_16k_performance_gate_remote.sh
```

Poll the same unified exec session until terminal status. Do not open
additional persistent PTYs.

- [ ] **Step 3: Audit the raw authority**

Verify from raw rows:

```text
four cells present
1/1/5 run counts per cell
exact 64-token parity for all recorded runs
all four ranks represented
positive candidate proposals and accepted tokens
positive batch-4 H2D and D2H for both policies
zero rejected speculative D2H
rank-wise peak memory complete
cleanup complete
source/model hashes exact
derived ratios and directions recompute
remote and local verifier PASS
```

- [ ] **Step 4: Update audit and handoff honestly**

Record:

- authoritative path and opaque run ID;
- exact campaign constants;
- median/min/max/pstdev metrics;
- acceptance and callback counts;
- real KV movement;
- peak memory;
- direction per batch and overall;
- what the result proves and does not prove;
- retained failed artifacts and root causes;
- next gate: independent TP4/32K performance.

Keep Phase 1 `NOT_PROMOTABLE`. Do not claim improvement unless the direction
is independently verified as `POSITIVE`.

- [ ] **Step 5: Run final fresh verification**

Repeat the independent verifier, focused tests, compilation, shell syntax,
and `git diff --check` after documentation updates.
