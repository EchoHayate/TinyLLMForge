# Exact CUDA Graph Budget Fallback Fault-Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independently verified, source-bound remote Qwen3-0.6B correctness evidence for all eight terminal exact CUDA Graph budget/fault fallbacks without adding a production fault switch or contaminating performance metrics.

**Architecture:** Freeze an eight-reason contract and a `budget_fallback_rows.jsonl` binding artifact, then extend the existing production gate with one isolated remote worker process per reason. Faults are injected only by harness-scoped mutations on a constructed `ModelRunner`; the independent verifier reconstructs identities, lifecycle, output/logit/KV correctness, terminal rejection, no replay, no recapture, and performance exclusion from raw evidence.

**Tech Stack:** Python 3, PyTorch/CUDA, TinyLLMForge `LLMEngine`/`ModelRunner`, JSON/JSONL immutable artifacts, SSH ControlMaster, Qwen3-0.6B on remote A100, dependency-light executable tests.

## Global Constraints

- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Execute inline in the current linked worktree on branch `feat/adaptive-ngram-speculation`; do not spawn subagents.
- Remote GPU/model work runs only as `sitian@10.232.195.203`.
- Use SSH ControlMaster path `/tmp/ssh-sitian-10.232.195.203`.
- Use remote Python `/data00/home/sitian/sitian-workspace01/tllm/env/bin/python`.
- Use model `/data00/home/sitian/sitian-workspace01/.ms_cache/Qwen/Qwen3-0___6B`.
- Use `CUDA_VISIBLE_DEVICES=0`.
- Every model process receives fresh distinct dynamic `TINYVLLM_DIST_PORT` and `MASTER_PORT`.
- Do not use `rsync`, modify the remote checkout, kill unrelated processes, delete shared `/tmp`, or switch GPU.
- Retry only `EADDRINUSE`.
- Preserve untracked `experiments/`; stage exact tracked paths only and never use `git add -A`.
- Preserve default-off behavior, batch-one CUDA Graph behavior, exact identity matching, no batch/width rounding, and fail-closed eager fallback.
- Fault injection exists only in the test/gate harness; do not add a production config field, environment variable, CLI switch, or `ModelRunner` fault branch.
- Fault-worker metrics must never contribute to throughput, latency, memory, initialization, or graph-hit performance ratios.
- Do not change `README.md` or claim a performance improvement before independently verified canonical correctness and canonical arrival-load both return `GO`.

---

### Task 1: Freeze the Eight-Reason Evidence Contract

**Files:**
- Modify: `tools/multi_sequence_cuda_graph_contract.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: existing `FALLBACK_REASONS`, `PRODUCTION_ARTIFACT_FILES`, `PRODUCTION_MANIFEST_FIELDS`, and canonical JSON/hash helpers.
- Produces: `BUDGET_FALLBACK_REASONS`, `BUDGET_FALLBACK_ROW_FIELDS`, `budget_fallback_rows.jsonl` in the production artifact domain, and manifest binding field `budget_fallback_sha256`.

- [ ] **Step 1: Write a failing closed-domain contract test**

Add:

```python
def test_budget_fallback_contract_is_closed_ordered_and_artifact_bound():
    assert contract.BUDGET_FALLBACK_REASONS == (
        "entry_limit",
        "static_byte_budget",
        "reserved_byte_budget",
        "single_capture_budget",
        "total_capture_budget",
        "scratch_unavailable",
        "capture_failed",
        "identity_drift",
    )
    assert set(contract.BUDGET_FALLBACK_REASONS) <= set(
        contract.FALLBACK_REASONS
    )
    assert "budget_fallback_rows.jsonl" in (
        contract.PRODUCTION_ARTIFACT_FILES
    )
    assert "budget_fallback_sha256" in (
        contract.PRODUCTION_MANIFEST_FIELDS
    )
```

- [ ] **Step 2: Write a failing exact row-schema test**

Add:

```python
def test_budget_fallback_row_schema_is_exact():
    assert contract.BUDGET_FALLBACK_ROW_FIELDS == (
        "row_id",
        "case_id",
        "reason",
        "source_sha256",
        "worker_pid",
        "tinyvllm_dist_port",
        "master_port",
        "gpu",
        "injection_class",
        "injection_installed",
        "injection_restored",
        "effective_cache_config",
        "pre_target_cache_summary",
        "target_identity_fields",
        "target_identity_sha256",
        "observation_dispatch_row_ids",
        "terminal_dispatch_row_ids",
        "capture_row_ids",
        "eager_output_token_ids",
        "candidate_output_token_ids",
        "logits_allclose",
        "logits_max_abs_diff",
        "eager_live_kv_sha256",
        "candidate_live_kv_sha256",
        "terminal_rejection_reason",
        "target_graph_replay_count",
        "target_capture_attempt_count",
        "post_rejection_capture_attempt_count",
        "complete",
    )
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the new constants and artifact binding do not exist.

- [ ] **Step 4: Implement the minimal frozen contract**

Add the exact constants from Steps 1-2, append
`budget_fallback_rows.jsonl` to `PRODUCTION_ARTIFACT_FILES`, and append
`budget_fallback_sha256` to `PRODUCTION_MANIFEST_FIELDS`. Do not change
`FALLBACK_REASONS`.

- [ ] **Step 5: Run the focused tests and verify GREEN**

Run the command from Step 3.

Expected: PASS with zero failed tests.

- [ ] **Step 6: Commit**

```bash
git add \
  tools/multi_sequence_cuda_graph_contract.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "test: freeze cuda graph fallback fault evidence"
```

---

### Task 2: Independently Verify Eight Terminal Fallback Rows

**Files:**
- Modify: `tools/verify_multi_sequence_cuda_graph_production.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: Task 1 constants, raw `dispatch_events.jsonl`,
  `capture_events.jsonl`, `correctness_rows.jsonl`, source manifest, process
  rows, and exact identity reconstruction.
- Produces:

```python
def _validate_budget_fallback_rows(
    *,
    mode: str,
    manifest: dict,
    rows: list[dict],
    dispatch_rows: list[dict],
    capture_rows: list[dict],
    correctness_rows: list[dict],
) -> tuple[list[str], dict]:
    ...
```

The summary contains:

```python
{
    "budget_fallback_required": 8,
    "budget_fallback_verified": int,
    "budget_fallback_reasons": list[str],
}
```

- [ ] **Step 1: Extend the production fixture with valid 8/8 raw evidence**

Update the existing fixture builder so each correctness run includes:

- one row per contractual reason in exact order;
- two cold observation dispatch rows;
- one target eager terminal row;
- two later eager terminal rows;
- no graph row for the target SHA;
- matching source and exact identity;
- matching eager/candidate tokens and live-KV hashes;
- `logits_allclose=True`;
- positive distinct worker ports;
- installation/restoration true only for runtime faults;
- raw capture rows consistent with the declared reason;
- `budget_fallback_sha256` equal to the artifact hash.

Arrival fixtures bind the valid canonical correctness summary rather than
creating fault-worker performance cases.

- [ ] **Step 2: Write failing happy-path verification assertions**

Assert:

```python
result = verifier.verify_run(run_dir, write_report=True)
assert result["classification"] == "GO"
assert result["budget_fallback_required"] == 8
assert result["budget_fallback_verified"] == 8
assert result["budget_fallback_reasons"] == list(
    contract.BUDGET_FALLBACK_REASONS
)
```

- [ ] **Step 3: Write failing tamper tests**

Create isolated fixture mutations for:

```text
missing reason
duplicate reason
unknown reason
declared/raw reason mismatch
missing cold observation
graph replay after rejection
capture after rejection
token mismatch
logits_allclose false
live KV hash mismatch
source SHA mismatch
identity field/SHA mismatch
identical worker ports
reused worker port
runtime injection not restored
impossible budget configuration
fault worker added to case_summaries performance matrix
```

Each mutation must make `verify_run()` return `NO_GO` with a specific failure
substring.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the verifier does not read or reconstruct the new rows.

- [ ] **Step 5: Add artifact loading and hash validation**

Add `budget_fallback_rows.jsonl` to `HASHED_PRODUCTION_FILES`, load it in
`verify_run()`, include it in source-row checks, verify
`manifest["budget_fallback_sha256"]`, and require an empty row list only for
non-correctness smoke producers that are not correctness bindings.

- [ ] **Step 6: Implement exact schema, identity, and lifecycle validation**

For each row:

```python
if tuple(row) != contract.BUDGET_FALLBACK_ROW_FIELDS:
    failures.append(f"{row_id}: budget fallback fields mismatch")
```

Then independently:

- recompute identity from `target_identity_fields`;
- index all referenced row IDs uniquely;
- require two cold eager observations before the terminal row;
- require terminal and later rows to be eager with the exact reason;
- reject graph rows or capture attempts after terminal rejection;
- compare token arrays, logits flag/tolerance, and live-KV hashes;
- validate injection class/install/restore fields;
- validate port uniqueness across all manifest processes and fault rows;
- validate each budget reason against `effective_cache_config` and
  `pre_target_cache_summary`.

- [ ] **Step 7: Exclude fault workers from performance aggregation**

Reject any `case_summaries.json` row or production matrix process row whose
worker kind is `budget-fallback`. Only the binding artifact and raw correctness
rows may contain fault-worker evidence.

- [ ] **Step 8: Bind arrival canonical to correctness 8/8**

When verifying `arrival-canonical`, require its correctness binding to contain:

```python
{
    "classification": "GO",
    "budget_fallback_required": 8,
    "budget_fallback_verified": 8,
}
```

Do not rerun the workers from arrival mode.

- [ ] **Step 9: Run the focused tests and verify GREEN**

Run the command from Step 4.

Expected: PASS with all tamper cases rejected.

- [ ] **Step 10: Commit**

```bash
git add \
  tools/verify_multi_sequence_cuda_graph_production.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: verify cuda graph fallback fault evidence"
```

---

### Task 3: Build Isolated Worker Commands and Aggregate Results

**Files:**
- Modify: `tools/run_multi_sequence_cuda_graph_production_gate_remote.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: Task 1 contract, existing `build_worker_command()`,
  `allocate_port_pair()`, source snapshot, occupancy checks, and atomic JSON
  helpers.
- Produces:

```python
def build_budget_fallback_worker_command(
    *,
    remote_source: str,
    output_dir: str,
    source_sha256: str,
    dist_port: int,
    master_port: int,
    reason: str,
) -> dict:
    ...

def build_budget_fallback_plan() -> list[dict]:
    ...
```

- [ ] **Step 1: Write failing CLI-domain tests**

Require:

```python
assert parser accepts:
    --worker-kind budget-fallback
    --budget-fallback-reason entry_limit
assert parser rejects:
    --worker-kind budget-fallback without a reason
    unknown reason
    a reason with worker-kind arrival/correctness/capacity
```

- [ ] **Step 2: Write failing command-isolation tests**

For all eight commands assert:

```python
assert command["env"]["CUDA_VISIBLE_DEVICES"] == "0"
assert command["env"]["TINYVLLM_SOURCE_SHA256"] == source_sha256
assert command["env"]["TINYVLLM_DIST_PORT"] != (
    command["env"]["MASTER_PORT"]
)
assert "--worker-kind" followed by "budget-fallback"
assert "--budget-fallback-reason" followed by the exact reason
assert command uses the fixed remote Python and source directory
assert all 16 ports are positive and unique
```

Also retain prohibited-operation assertions for `rsync`, `kill`, `pkill`,
shared `/tmp` deletion, remote checkout writes, and fixed ports.

- [ ] **Step 3: Write failing orchestration tests**

For correctness smoke and canonical planning, assert exactly eight isolated
fault workers in contractual order. Assert arrival planning creates zero such
workers and requires a correctness binding.

- [ ] **Step 4: Write failing aggregation tests**

Given eight valid worker directories, require:

```python
aggregate["budget_fallback_rows"] has eight ordered rows
aggregate dispatch/capture/correctness rows retain unique row IDs
fault workers do not appear in case_summaries
manifest["budget_fallback_sha256"] hashes the aggregate artifact
```

Missing/interrupted workers classify the run `INCOMPLETE`.

- [ ] **Step 5: Run focused tests and verify RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the worker kind, plan, and aggregation do not exist.

- [ ] **Step 6: Implement parser and command construction**

Extend `--worker-kind` choices with `budget-fallback`, add the optional reason
argument, validate the legal argument combinations, and build one source-bound
command per reason with fresh ports.

- [ ] **Step 7: Integrate correctness-only orchestration**

After capacity calibration and before production aggregation:

- launch one worker at a time;
- record before/after GPU occupancy;
- retry only `EADDRINUSE` with newly allocated ports;
- propagate unrelated occupancy as structured `INCOMPLETE`;
- never launch fault workers in arrival mode.

- [ ] **Step 8: Aggregate and bind the new artifact**

Write `budget_fallback_rows.jsonl` atomically, merge referenced raw rows into
the existing raw artifacts, exclude workers from `case_summaries`, add the
artifact hash to the manifest, and include it in source artifact hashes.

- [ ] **Step 9: Run focused tests and verify GREEN**

Run the command from Step 5.

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add \
  tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: orchestrate cuda graph fallback workers"
```

---

### Task 4: Implement Harness-Scoped Fault Preconditions and Worker Evidence

**Files:**
- Modify: `tools/run_multi_sequence_cuda_graph_production_gate_remote.py`
- Modify: `tools/test_multi_sequence_cuda_graph_gate.py`
- Test: `tools/test_model_runner_spec_verify.py`

**Interfaces:**
- Consumes: real `LLMEngine`, `ModelRunner`, `ExactCudaGraphCacheConfig`,
  `_ExactGraphCaptureError`, exact identity builder, dispatch observation,
  existing correctness workload helpers, and Task 3 worker CLI.
- Produces:

```python
def _run_budget_fallback_worker(args) -> dict:
    ...

def _install_budget_fault(model_runner, reason: str):
    """Return a restoration callback and injection metadata."""
```

No symbol is added to `tinyvllm/config.py`,
`tinyvllm/engine/model_runner.py`, or
`tinyvllm/engine/exact_cuda_graph_cache.py`.

- [ ] **Step 1: Write failing source-boundary tests**

Assert the production source files do not contain:

```text
budget_fallback_reason
fault_injection
TINYVLLM_CUDA_GRAPH_FAULT
inject_exact_cuda_graph
```

The only injection implementation is in the remote gate script.

- [ ] **Step 2: Write failing per-reason precondition tests**

Use lightweight fake runners/caches to assert:

- `entry_limit` creates one ready seed entry and a distinct target identity;
- `static_byte_budget` makes target estimate exceed the ceiling;
- `reserved_byte_budget` is post-capture binding;
- `single_capture_budget` binds only the single ceiling;
- `total_capture_budget` binds only the cumulative ceiling;
- `scratch_unavailable` wraps and restores scratch restoration;
- `capture_failed` raises production `_ExactGraphCaptureError`;
- `identity_drift` changes only the post-capture rebuild identity.

- [ ] **Step 3: Write failing worker lifecycle tests**

For every reason, the worker result must contain:

```python
assert result["reason"] == reason
assert result["observation_count"] >= 3
assert result["target_dispatch"] == "eager"
assert result["terminal_rejection_reason"] == reason
assert result["target_graph_replay_count"] == 0
assert result["post_rejection_capture_attempt_count"] == 0
assert result["eager_output_token_ids"] == (
    result["candidate_output_token_ids"]
)
assert result["logits_allclose"] is True
assert result["eager_live_kv_sha256"] == (
    result["candidate_live_kv_sha256"]
)
assert result["complete"] is True
```

Runtime faults additionally require install and restore true.

- [ ] **Step 4: Write failing exception-restoration tests**

Force comparison, artifact writing, and terminal-step exceptions separately;
require the harness wrapper to restore in `finally` and the worker to emit a
structured incomplete artifact rather than leave a mutated runner.

- [ ] **Step 5: Run focused tests and verify RED**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
```

Expected: FAIL because the injection and worker functions do not exist.

- [ ] **Step 6: Implement config/state preconditions**

Construct cache configurations and seed/target trajectories so exactly one
ceiling is binding. Record the effective config and pre-target summary before
the target's third observation.

- [ ] **Step 7: Implement runtime wrappers with unconditional restoration**

Wrap only the worker's constructed `ModelRunner` instance. Store original
bound methods, install immediately before the target third observation, and
restore in `finally`. Do not patch module globals or production source.

- [ ] **Step 8: Execute real eager/candidate correctness comparison**

Use equivalent target inputs and record:

- output token arrays;
- logits allclose and maximum absolute difference;
- live-slot KV hashes;
- exact identity fields/SHA;
- raw dispatch and capture rows;
- two later terminal steps.

Require the target candidate step to remain eager. A graph dispatch, wrong
reason, changed output/KV, recapture, or non-terminal later state makes the
worker complete-but-failed rather than silently incomplete.

- [ ] **Step 9: Write the atomic worker artifacts**

Write:

```text
budget_fallback_row.json
dispatch_events.jsonl
capture_events.jsonl
correctness_rows.jsonl
memory_trace.jsonl
environment.json
```

Every row carries the source SHA and unique case/row IDs.

- [ ] **Step 10: Run focused suites and verify GREEN**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_model_runner_spec_verify.py
```

Expected: both commands PASS with zero failures.

- [ ] **Step 11: Commit**

```bash
git add \
  tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  tools/test_multi_sequence_cuda_graph_gate.py
git commit -m "feat: inject cuda graph fallback faults in gate"
```

---

### Task 5: Validate Source Binding and Remote Command Discipline

**Files:**
- Modify only if a demonstrated defect exists:
  `tools/run_multi_sequence_cuda_graph_production_gate_remote.py`
- Modify only if a demonstrated defect exists:
  `tools/test_multi_sequence_cuda_graph_gate.py`

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: fresh local-suite output and a source-bound remote preflight
  artifact for the current commit/tree SHA.

- [ ] **Step 1: Run both complete local exact CUDA Graph suites**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_model_runner_spec_verify.py
```

Expected: both PASS with zero failures.

- [ ] **Step 2: Verify no production injection surface**

Run:

```bash
rg -n \
  'budget_fallback_reason|fault_injection|TINYVLLM_CUDA_GRAPH_FAULT|inject_exact_cuda_graph' \
  tinyvllm/config.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/exact_cuda_graph_cache.py
```

Expected: no matches.

- [ ] **Step 3: Verify worktree and exact staging boundary**

Run:

```bash
git status --short --branch
git diff --check
```

Expected: only intentional tracked edits, if any, plus untouched untracked
`experiments/`.

- [ ] **Step 4: Run source-bound remote preflight**

Run:

```bash
RUN_TAG=qwen3-06b-exact-cuda-graph-fallback-preflight-$(date +%Y%m%d-%H%M%S)
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  preflight \
  --run-tag "$RUN_TAG"
```

Expected: remote source audit passes, fixed host/Python/model/GPU are recorded,
dynamic-port discipline passes, and the downloaded artifact binds the current
source tree SHA.

- [ ] **Step 5: Fix only demonstrated harness defects with TDD**

If a failure is not environmental:

1. add a regression test;
2. run it and observe the intended failure;
3. apply the minimal harness/verifier fix;
4. rerun both local suites;
5. rerun preflight with a new run tag.

Do not alter production dispatch semantics to make the gate pass.

- [ ] **Step 6: Commit any validated fix**

Stage only exact changed tracked paths. If no defect was found, do not create
an empty commit.

---

### Task 6: Run Remote Smoke Correctness With 8/8 Fault Evidence

**Files:**
- Produce untracked artifacts under: `experiments/cuda_graph/`

**Interfaces:**
- Consumes: passing Task 5 evidence and exclusive remote GPU 0.
- Produces: independently verified non-authoritative smoke correctness with
eight terminal fallback rows.

- [ ] **Step 1: Check GPU 0 occupancy without modifying it**

Run:

```bash
ssh \
  -o ControlMaster=auto \
  -o ControlPath=/tmp/ssh-sitian-10.232.195.203 \
  -o ControlPersist=600 \
  sitian@10.232.195.203 \
  "bash -lc 'CUDA_VISIBLE_DEVICES=0 nvidia-smi \
    --query-compute-apps=pid,used_memory,process_name \
    --format=csv,noheader,nounits'"
```

Expected: no unrelated process on GPU 0. If occupied, write/retain structured
`INCOMPLETE`, do not kill processes or switch GPU, and stop this task.

- [ ] **Step 2: Run correctness smoke**

Run:

```bash
RUN_TAG=qwen3-06b-exact-cuda-graph-fallback-correctness-smoke-$(date +%Y%m%d-%H%M%S)
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  correctness-smoke \
  --run-tag "$RUN_TAG"
```

Expected:

```text
classification = NON_AUTHORITATIVE_SMOKE
budget_fallback_required = 8
budget_fallback_verified = 8
```

Normal allowlisted hit, non-allowlisted eager fallback, output/logit/KV
correctness, and exact lifecycle checks also pass.

- [ ] **Step 3: Independently rerun local verification**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/verify_multi_sequence_cuda_graph_production.py \
  --run-dir "experiments/cuda_graph/$RUN_TAG" \
  --write-report
```

Expected: identical smoke classification and 8/8 reconstruction.

- [ ] **Step 4: Stop on any incorrect or incomplete result**

If the worker is complete-but-wrong, use systematic debugging and TDD before
retrying. If blocked by occupancy or transport, retain the exact structured
artifact and do not claim correctness.

---

### Task 7: Run Canonical Correctness and Arrival Gates

**Files:**
- Produce untracked artifacts under: `experiments/cuda_graph/`

**Interfaces:**
- Consumes: passing smoke evidence and exclusive remote GPU 0.
- Produces: independently verified canonical correctness and paired
  arrival-load `GO | NO_GO | INCOMPLETE`.

- [ ] **Step 1: Run canonical correctness**

Run:

```bash
CORRECTNESS_TAG=qwen3-06b-exact-cuda-graph-fallback-correctness-canonical-$(date +%Y%m%d-%H%M%S)
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  correctness-canonical \
  --run-tag "$CORRECTNESS_TAG"
```

Expected:

- all 315 diagnostic cases complete;
- `EXACT_REPLAY_CORRECT`;
- `ROUNDED_REPLAY_CORRUPT`;
- `LEGACY_COMPATIBLE`;
- `POLICY_EXACT`;
- actual `ModelRunner` output/logit/live-KV correctness passes;
- `budget_fallback_required=8`;
- `budget_fallback_verified=8`;
- each reason is terminal with no replay or recapture.

- [ ] **Step 2: Stop unless correctness independently returns `GO`**

Do not launch canonical arrival-load from producer status alone. Rerun:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/verify_multi_sequence_cuda_graph_production.py \
  --run-dir "experiments/cuda_graph/$CORRECTNESS_TAG" \
  --write-report
```

Proceed only if the independent result is canonical `GO` with 8/8 evidence.

- [ ] **Step 3: Run canonical paired arrival-load**

Run:

```bash
ARRIVAL_TAG=qwen3-06b-exact-cuda-graph-arrival-canonical-$(date +%Y%m%d-%H%M%S)
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/run_multi_sequence_cuda_graph_production_gate_remote.py \
  arrival-canonical \
  --run-tag "$ARRIVAL_TAG" \
  --correctness-run-tag "$CORRECTNESS_TAG"
```

Expected: the arrival verifier binds the correctness source SHA and 8/8 result,
uses no fault-worker performance samples, and independently applies:

```text
aggregate decode throughput ratio       >= 1.15
stable-exact decode throughput ratio    >= 1.25
minimum per-case request throughput     >= 0.95
maximum per-case p95 ITL ratio          <= 1.05
maximum per-case p99 ITL ratio          <= 1.10
peak reserved-memory ratio              <= 1.02
initialization-duration ratio           <= 1.05
stable-exact graph hit rate             >= 0.60
```

- [ ] **Step 4: Independently rerun arrival verification**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/verify_multi_sequence_cuda_graph_production.py \
  --run-dir "experiments/cuda_graph/$ARRIVAL_TAG" \
  --write-report
```

Record the exact `GO | NO_GO | INCOMPLETE` classification and all recomputed
ratios. Do not convert a threshold loss into an implementation success claim.

---

### Task 8: Completion Audit and Durable Result Recording

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Modify only after canonical arrival `GO`: `README.md`

**Interfaces:**
- Consumes: current git state, all local commands, source-bound remote
  artifacts, independent verification, and the original production plan.
- Produces: a prompt-to-artifact checklist and durable evidence record without
  overclaiming.

- [ ] **Step 1: Restate the concrete completion criteria**

Record:

1. default-off exact cache;
2. exact identity and no rounding;
3. three observations before capture;
4. scratch/capacity isolation;
5. eight independently verified terminal fallback reasons;
6. no production injection surface;
7. 315-case diagnostic correctness;
8. actual remote `ModelRunner` correctness;
9. canonical arrival thresholds;
10. fault-worker performance exclusion;
11. source/provenance binding;
12. README unchanged unless canonical arrival is `GO`.

- [ ] **Step 2: Build and inspect a prompt-to-artifact checklist**

For each criterion identify:

- exact source file and test;
- exact command and exit status;
- exact run directory;
- exact artifact and row IDs;
- independent verifier field;
- any limitation or missing evidence.

Treat uncertainty, missing rows, producer-only status, smoke-only status, or
7/8 coverage as incomplete.

- [ ] **Step 3: Run fresh final local verification**

Run:

```bash
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_multi_sequence_cuda_graph_gate.py
/Users/bytedance/Desktop/RL_local_mirror/.venv/bin/python \
  tools/test_model_runner_spec_verify.py
git diff --check
git status --short --branch
```

Expected: both suites PASS, no whitespace errors, and only intentional tracked
documentation plus untouched untracked `experiments/`.

- [ ] **Step 4: Append exact evidence to the handoff**

Record current commit/tree SHA, all run tags, commands, classifications,
8/8 reason list, threshold ratios, capacity parity, source binding, GPU
occupancy state, artifact paths, limitations, and next step.

For `NO_GO` or `INCOMPLETE`, state explicitly that no production performance
improvement is proven.

- [ ] **Step 5: Update README only after canonical arrival `GO`**

If and only if independent canonical correctness and arrival both return `GO`,
document:

- default-off opt-in configuration;
- measured Qwen3-0.6B A100 evidence;
- exact source/run tags;
- thresholds and observed ratios;
- 8/8 fallback safety result;
- limitations and non-generalization boundary.

Otherwise preserve a zero README diff.

- [ ] **Step 6: Selectively commit tracked documentation**

For `NO_GO` or `INCOMPLETE`:

```bash
git add AGENT_HANDOFF_STATE.md
git commit -m "docs: record cuda graph fallback gate"
```

For canonical `GO`:

```bash
git add AGENT_HANDOFF_STATE.md README.md
git commit -m "docs: publish exact cuda graph production evidence"
```

Never stage `experiments/`.

---

## Plan Self-Review Checklist

1. Every one of the eight reasons has a frozen contract, a worker, raw
   evidence, independent reconstruction, and a tamper test.
2. Correctness smoke/canonical run workers; arrival modes bind correctness and
   do not rerun or measure workers.
3. No task modifies production config/cache/runner to add fault switches.
4. Every implementation task follows RED, minimal GREEN, verification, then
   exact-path commit.
5. Remote constraints use the exact user, host, Python, model, GPU, ControlPath,
   and dynamic ports.
6. `experiments/` remains untracked and unstaged.
7. Performance claims remain blocked until independent canonical correctness
   and arrival `GO`.
