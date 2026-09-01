# TP4 Collective-Stable Decode Replay Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking. This repository's approved execution
> mode is inline; do not dispatch subagents.

**Goal:** Produce a source-bound, independently verified Qwen3.8-27B BF16 TP4
eager-versus-existing-exact-CUDA-Graph gate for Q0/Q1/Q2 that reports
correctness, all-rank collective stability, steady-state benefit, and capture,
memory, startup, and teardown cost.

**Architecture:** Reuse the existing default-off exact multi-sequence CUDA
Graph mechanism and add one bounded generic transactional-model protocol
required by the r5 hardware RED. Conventional models retain the existing
forward/hidden-output path; transactional models provide opaque schema,
lease-seal, snapshot/restore, and full-step hooks so capture can roll back
state and replay can commit exactly once to the sealed leases. The existing
contract, worker, controller, assembler, and two independent verifier
executions then reconstruct the terminal classification from hash-bound
evidence.

**Tech Stack:** Python 3, PyTorch, TinyLLMForge `LLMEngine`, CUDA Graphs, NCCL
TP4, JSON/JSONL, SHA-256, `unittest`/dependency-light script tests, SSH
ControlMaster, Qwen3.8-27B BF16, four A100 80 GB PCIe GPUs.

## Execution Status

- Tasks 1–4 are implemented and pushed through
  `c66f2dbfe12ba31ed010c6d733b569ae83fc7aa1`.
- The five new gate suites pass with counts `12 + 7 + 6 + 6 + 16`.
- `test_model_runner_spec_verify.py` passes after correcting its stale
  reference to the renamed
  `test_model_runner_invalidates_all_distinct_burst_graphs`.
- Task 5 is not fully green because the local environment lacks `torch` for
  `test_multi_sequence_cuda_graph_gate.py`.
- Attempt `20260831-qwen38-tp4-decode-replay-r1` stopped before SSH or GPU
  admission with an evidence-backed `INCOMPLETE` credential preflight.
- The r1 audit is
  `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`.
- Attempts r2-r5 are also consumed. r5 reached strict-clean real hardware and
  exposed a uniform all-rank capture failure:
  `Qwen35PackedForCausalLM` has no ordinary `forward()`.
- The design now selects lease-sealed transactional full-step replay. The
  lease-independent two-bank alternative is rejected because the real TP4
  state layout requires about 297.4 MiB at batch 4 and 594.8 MiB at batch 8
  before graph workspace; Q1 exceeds the frozen 512 MiB cost gate.
- No performance classification exists. Stage 1 remains prohibited.
- The next real retry must use fresh tag
  `20260831-qwen38-tp4-decode-replay-r6`.

## Global Constraints

- Work only in `/Users/bytedance/dev/TinyLLMForge`; Desktop is a symlink.
- Do not use `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Execute inline; do not create a worktree or dispatch subagents.
- Push only to `origin/feat/kv-sparse-attention`.
- Use strict RED -> minimal implementation -> GREEN for every code task.
- Keep `multi_sequence_cuda_graphs=False` by default.
- The r5 RED proves a bounded `tinyvllm/` correction is required. Limit it to
  generic transactional graph hooks, identity sealing, capture rollback, and
  first-adopter wiring described in the amended design.
- Baseline is `enforce_eager=True`.
- Candidate is `enforce_eager=False`,
  `multi_sequence_cuda_graphs=True`, with batch allowlist `(2, 4, 8)`.
- Use Qwen3.8-27B revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Use Q0 `(256, 128, 4)`, Q1 `(256, 128, 8)`, and Q2 `(2048, 128, 4)`.
- Use TP4, BF16, greedy decoding, one process per GPU, and five measured
  paired repetitions after unmeasured warmup.
- Require exact token IDs, output length, stop reason, request identity, and
  all-rank graph/collective/lifecycle agreement.
- Do not silently rerun eager after an authoritative graph replay begins.
- Separate cold capture cost from steady-state performance.
- Report output tokens/s, QPS, TPOT, P95/P99 TPOT, P99 E2E, TTFT, capture
  cost, peak allocated/reserved memory, initialization, replay coverage, and
  teardown.
- All remote task data, caches, logs, source snapshots, and temporary files
  must be below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Do not write task data to remote `/`, `/tmp`, model caches, or old checkouts.
- Do not run `kinit` or `krenew`.
- Do not terminate, take over, or clean external GPU processes.
- The local controller waits for four strict-clean GPUs and launches
  immediately when admitted.
- Do not launch a duplicate worker for an existing active run tag.
- Keep large traces remote; download only compact verifier-required evidence.
- Stage exact paths only; never use broad `git add`, `git reset`, `git clean`,
  or unrelated formatting.
- Every commit uses `git -c core.hooksPath=/dev/null commit` and exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

## File Map

- Create `tools/tp4_decode_replay_contract.py`
  - frozen profile, schemas, validation helpers, paired aggregation, and final
    classification.
- Create `tools/test_tp4_decode_replay_contract.py`
  - RED/GREEN tests for profile identity, strict rows, cross-rank agreement,
    benefit/cost gates, precedence, and non-finite rejection.
- Create `tools/tp4_decode_replay_worker.py`
  - canonical `LLMEngine` baseline/candidate execution, per-step all-rank
    graph observations, internal collective profile, memory, request metrics,
    capture cost, and cleanup.
- Create `tools/test_tp4_decode_replay_worker.py`
  - dependency-injected worker tests; no local model or GPU required.
- Create `tools/assemble_tp4_decode_replay.py`
  - strict raw-attempt validation, compact immutable bundle assembly,
    producer classification, and manifest.
- Create `tools/verify_tp4_decode_replay.py`
  - independent reconstruction from bundle rows and manifest hashes.
- Create `tools/test_assemble_tp4_decode_replay.py`
  - assembler tamper and completeness tests.
- Create `tools/test_verify_tp4_decode_replay.py`
  - verifier mutation matrix.
- Create `tools/run_tp4_decode_replay.py`
  - plan-only, monitor, launch, download, remote verification, local
    verification, and cleanup orchestration.
- Create `tools/test_run_tp4_decode_replay.py`
  - strict-clean admission, remote-root, no-duplicate, ordering, and failure
    cleanup tests.
- Create
  `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
  - terminal evidence, benefit/cost table, limitations, and
    prompt-to-artifact checklist.
- Modify
  `docs/superpowers/specs/2026-08-31-tp4-collective-stable-decode-replay-qualification-design.md`
  - terminal classification and evidence links only after the gate.
- Modify this plan
  - task status and terminal reconciliation only after evidence exists.
- Modify `tinyvllm/engine/flash_attn_split_policy.py`
  - bind exact graph identity to execution protocol, state schema, and opaque
    lease seal.
- Modify `tinyvllm/engine/model_runner.py`
  - dispatch conventional versus transactional capture/replay, restore
    capture-time state, and reject lease drift before replay.
- Modify `tinyvllm/models/qwen35_packed.py`
  - provide the first-adopter implementation of the generic transactional
    graph hooks.
- Modify `tinyvllm/engine/exact_cuda_graph_cache.py`
  - retain transactional output protocol metadata without model-specific
    fields.
- Modify `tools/test_model_runner_spec_verify.py`
  - cover capture rollback, exact-once replay state advancement, lease drift,
    output semantics, and conventional-path compatibility.
- Modify `tools/test_qwen35_prepared_model_step.py`
  - cover the Qwen first-adopter hook contract independently.

## Shared Interfaces

`tools/tp4_decode_replay_contract.py` must expose:

```text
WORKLOADS = {
    "Q0": {"prompt_tokens": 256, "output_tokens": 128, "concurrency": 4},
    "Q1": {"prompt_tokens": 256, "output_tokens": 128, "concurrency": 8},
    "Q2": {"prompt_tokens": 2048, "output_tokens": 128, "concurrency": 4},
}
RANKS = (0, 1, 2, 3)
ARMS = ("eager", "graph")
MEASURED_REPETITIONS = 5
CLASSIFICATIONS = (
    "GO_STAGE1_JUSTIFIED",
    "NO_GO_PERFORMANCE",
    "NO_GO_CORRECTNESS_OR_LIFECYCLE",
    "NO_GO_MECHANISM_NOT_EXERCISED",
    "INCOMPLETE",
)
THRESHOLDS = {
    "aggregate_output_throughput_ratio": 1.05,
    "aggregate_median_tpot_ratio": 0.95,
    "minimum_workload_output_throughput_ratio": 0.97,
    "maximum_workload_median_tpot_ratio": 1.03,
    "maximum_workload_p99_e2e_ratio": 1.03,
    "maximum_workload_ttft_ratio": 1.03,
    "minimum_replay_coverage": 0.80,
    "maximum_added_peak_allocated_bytes_per_rank": 512 * 1024 * 1024,
    "maximum_added_peak_reserved_bytes_per_rank": 512 * 1024 * 1024,
}

canonical_json_bytes(value: object) -> bytes
canonical_json_sha256(value: object) -> str
build_case_matrix() -> tuple[dict, ...]
validate_rank_dispatch_rows(rows: list[dict]) -> dict
validate_correctness_rows(rows: list[dict]) -> dict
def classify(
    *,
    performance_rows: list[dict],
    correctness_rows: list[dict],
    rank_dispatch_rows: list[dict],
    rank_collective_rows: list[dict],
    rank_lifecycle_rows: list[dict],
    memory_rows: list[dict],
    capture_cost_rows: list[dict],
) -> dict
```

`tools/tp4_decode_replay_worker.py` must expose:

```text
build_engine_config(*, arm: str, workload: str) -> dict
def collect_rank_graph_observations(
    engine,
    *,
    case_id: str,
    phase: str,
    step_index: int,
    timeout_s: float,
) -> list[dict]
def run_arm(
    *,
    model_root: Path,
    case: dict,
    output_dir: Path,
    engine_factory=default_engine_factory,
    sampling_params_factory=default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
) -> dict
run_pair(
    *,
    model_root: Path,
    pair_cases: tuple[dict, dict],
    output_dir: Path,
    engine_factory=default_engine_factory,
    sampling_params_factory=default_sampling_params_factory,
    clock_ns=time.monotonic_ns,
) -> dict
```

`tools/assemble_tp4_decode_replay.py` must expose:

```text
def assemble_bundle(
    *,
    raw_root: Path,
    output_root: Path,
    source_identity: dict,
    launch_admission: dict,
    cleanup: dict,
) -> dict
```

`tools/verify_tp4_decode_replay.py` must expose:

```text
verify_bundle(root: Path) -> dict
```

`tools/run_tp4_decode_replay.py` must expose:

```text
REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818/"
    "tp4-collective-stable-decode-replay"
)

build_plan(
    *,
    run_tag: str,
    source_identity: dict,
    selected_gpus: list[dict],
) -> dict
run_attempt(*, plan: dict, adapter: object) -> dict
monitor_and_run(
    *,
    run_tag: str,
    gpu_monitor: object,
    adapter: object,
) -> dict
```

---

### Task 1: Freeze the Qualification Contract and Classifier

**Files:**

- Create: `tools/tp4_decode_replay_contract.py`
- Create: `tools/test_tp4_decode_replay_contract.py`

**Interfaces:**

- Consumes: the shared constants and signatures above.
- Produces: the only authoritative workload matrix, row validation, paired
  ratios, and classification precedence used by producer and verifier.

- [ ] **Step 1: Write failing profile and matrix tests**

```python
def test_case_matrix_is_paired_and_frozen():
    rows = contract.build_case_matrix()
    assert len(rows) == 3 * 5 * 2
    assert {row["workload"] for row in rows} == {"Q0", "Q1", "Q2"}
    for workload, expected in contract.WORKLOADS.items():
        matching = [row for row in rows if row["workload"] == workload]
        assert len(matching) == 10
        assert all(row["profile"] == expected for row in matching)
        assert {row["arm"] for row in matching} == {"eager", "graph"}
```

- [ ] **Step 2: Run RED**

Run:

```bash
python tools/test_tp4_decode_replay_contract.py
```

Expected: non-zero exit because `tools.tp4_decode_replay_contract` does not
exist.

- [ ] **Step 3: Implement canonical constants, hashes, and case matrix**

Implement the exact shared constants above. Alternate paired order with:

```python
order = ("eager", "graph") if repetition % 2 == 0 else ("graph", "eager")
```

Every case row contains:

```python
{
    "case_id": f"{workload}__r{repetition}__{arm}",
    "pair_id": f"{workload}__r{repetition}",
    "workload": workload,
    "repetition": repetition,
    "arm": arm,
    "order_index": order.index(arm),
    "profile": dict(WORKLOADS[workload]),
}
```

- [ ] **Step 4: Add RED tests for strict completeness and precedence**

Construct one synthetic passing bundle, then mutate independently:

```python
mutations = {
    "missing_rank": "INCOMPLETE",
    "token_mismatch": "NO_GO_CORRECTNESS_OR_LIFECYCLE",
    "mixed_graph_eager_dispatch": "NO_GO_CORRECTNESS_OR_LIFECYCLE",
    "collective_order_mismatch": "NO_GO_CORRECTNESS_OR_LIFECYCLE",
    "cleanup_failure": "NO_GO_CORRECTNESS_OR_LIFECYCLE",
    "low_replay_coverage": "NO_GO_MECHANISM_NOT_EXERCISED",
    "throughput_below_gate": "NO_GO_PERFORMANCE",
    "tpot_below_gate": "NO_GO_PERFORMANCE",
    "ttft_regression": "NO_GO_PERFORMANCE",
    "memory_regression": "NO_GO_PERFORMANCE",
}
```

Also reject `NaN`, infinity, duplicate row IDs, unknown workload/arm/rank,
missing repetitions, and non-canonical case IDs.

- [ ] **Step 5: Run RED**

Run:

```bash
python tools/test_tp4_decode_replay_contract.py
```

Expected: the new classifier tests fail because validation and classification
are absent.

- [ ] **Step 6: Implement the minimal classifier**

Implement in this order:

```python
if mandatory_evidence_missing:
    classification = "INCOMPLETE"
elif correctness_or_lifecycle_failed:
    classification = "NO_GO_CORRECTNESS_OR_LIFECYCLE"
elif replay_coverage < THRESHOLDS["minimum_replay_coverage"]:
    classification = "NO_GO_MECHANISM_NOT_EXERCISED"
elif any_performance_or_cost_gate_failed:
    classification = "NO_GO_PERFORMANCE"
else:
    classification = "GO_STAGE1_JUSTIFIED"
```

Return the classification, failed gates, per-workload ratios, aggregate
ratios, maximum per-rank memory deltas, replay coverage, capture cost, and
capture-amortization tokens.

- [ ] **Step 7: Run GREEN**

Run:

```bash
python tools/test_tp4_decode_replay_contract.py
python -m py_compile \
  tools/tp4_decode_replay_contract.py \
  tools/test_tp4_decode_replay_contract.py
```

Expected: all contract tests pass and compilation exits zero.

- [ ] **Step 8: Commit Task 1**

```bash
git add -- \
  tools/tp4_decode_replay_contract.py \
  tools/test_tp4_decode_replay_contract.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "test(perf): freeze TP4 decode replay gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 2: Build the Canonical TP4 Worker

**Files:**

- Create: `tools/tp4_decode_replay_worker.py`
- Create: `tools/test_tp4_decode_replay_worker.py`

**Interfaces:**

- Consumes: `build_case_matrix()`, canonical `LLMEngine`,
  `LLMEngine.call_model_runner_acknowledged()`,
  `configure_decode_internal_profile()`,
  `finalize_decode_internal_profile()`, `memory_snapshots()`, and
  `reset_peak_memory_stats()`.
- Produces: one complete raw attempt directory containing per-rank graph,
  collective, lifecycle, request, performance, memory, correctness, and
  capture-cost rows.

- [ ] **Step 1: Write RED tests for engine configuration**

```python
def test_engine_config_differs_only_by_graph_policy():
    eager = worker.build_engine_config(arm="eager", workload="Q1")
    graph = worker.build_engine_config(arm="graph", workload="Q1")
    assert eager | {
        "enforce_eager": False,
        "multi_sequence_cuda_graphs": True,
    } == graph
    assert eager["tensor_parallel_size"] == 4
    assert eager["enforce_eager"] is True
    assert graph["multi_sequence_cuda_graph_batch_allowlist"] == (2, 4, 8)
```

Also assert BF16, greedy sampling, max lengths, and Q0/Q1/Q2 values.

- [ ] **Step 2: Run RED**

Run:

```bash
python tools/test_tp4_decode_replay_worker.py
```

Expected: import failure because the worker does not exist.

- [ ] **Step 3: Implement request and engine setup**

Use:

```python
LLMEngine(
    str(model_root),
    tensor_parallel_size=4,
    enforce_eager=(arm == "eager"),
    multi_sequence_cuda_graphs=(arm == "graph"),
    multi_sequence_cuda_graph_batch_allowlist=(2, 4, 8),
    max_num_seqs=max(8, concurrency),
    max_model_len=prompt_tokens + output_tokens,
    max_num_batched_tokens=prompt_tokens * concurrency,
)
SamplingParams(temperature=0.0, max_tokens=128)
```

Generate deterministic token-ID prompts with request-specific offsets; do not
depend on tokenizer text generation.

- [ ] **Step 4: Write RED tests for all-rank graph observations**

Use a fake engine whose acknowledged call returns rank 0 plus ranks 1–3.
Require `collect_rank_graph_observations()` to:

- return ranks in order 0–3;
- attach `case_id`, phase, and step index;
- reject missing, duplicate, or mismatched ranks;
- reject rank disagreement in dispatch, graph identity, cache state, capture
  attempt, or fallback reason;
- allow eager baseline rows with `feature_enabled=False`;
- require graph candidate measured rows to expose replay.

- [ ] **Step 5: Implement all-rank observation collection**

Call:

```python
local, acknowledgements = engine.call_model_runner_acknowledged(
    "cuda_graph_dispatch_observation",
    timeout_s=timeout_s,
)
```

Normalize:

```python
ranked = [(0, local)] + [
    (ack.rank, ack.result) for ack in acknowledgements
]
```

Write one row per rank per observed decode step. Do not infer worker state from
rank 0.

- [ ] **Step 6: Write RED tests for paired arm execution**

Fake engine events must prove:

- warmup rows are excluded from steady-state metrics;
- each decode `engine.step()` is followed by acknowledged rank observation;
- request outputs retain token IDs, lengths, stop reasons, TTFT, TPOT, and E2E;
- decode internal profile yields four rank rows and collective-order digests;
- memory snapshots contain four ranks;
- cleanup requires four zero exit codes and four rank receipts;
- exceptions still call `engine.exit()` exactly once.

- [ ] **Step 7: Implement minimal worker execution**

For each arm:

1. create a fresh engine;
2. reset peak memory;
3. configure decode internal profile;
4. execute unmeasured warmup requests;
5. reset the profile;
6. execute measured requests;
7. collect graph observations after every decode step;
8. finalize the profile;
9. snapshot memory;
10. exit and validate cleanup.

Write each raw file atomically with `allow_nan=False`. Keep full internal
profile traces remote; emit compact collective-order digests and per-step
timing rows for the final bundle.

- [ ] **Step 8: Run GREEN and adjacent regressions**

Run:

```bash
python tools/test_tp4_decode_replay_worker.py
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_model_runner_spec_verify.py
python -m py_compile \
  tools/tp4_decode_replay_worker.py \
  tools/test_tp4_decode_replay_worker.py
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add -- \
  tools/tp4_decode_replay_worker.py \
  tools/test_tp4_decode_replay_worker.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): add TP4 decode replay worker" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 3: Assemble and Independently Verify Immutable Evidence

**Files:**

- Create: `tools/assemble_tp4_decode_replay.py`
- Create: `tools/verify_tp4_decode_replay.py`
- Create: `tools/test_assemble_tp4_decode_replay.py`
- Create: `tools/test_verify_tp4_decode_replay.py`

**Interfaces:**

- Consumes: raw worker rows, source identity, launch admission, cleanup, and
  `contract.classify()`.
- Produces: a compact `final_bundle/`, producer classification, manifest, and
  independently reconstructed verification.

- [ ] **Step 1: Write RED assembler completeness tests**

Build a complete synthetic raw attempt and assert these required files:

```python
REQUIRED_INPUTS = (
    "source_manifest.json",
    "source.patch",
    "environment.json",
    "gpu_inventory.json",
    "workload_profile.json",
    "process_receipts.json",
    "rank_environment.jsonl",
    "rank_dispatch_events.jsonl",
    "rank_collective_events.jsonl",
    "rank_lifecycle_rows.jsonl",
    "request_rows.jsonl",
    "performance_rows.jsonl",
    "memory_rows.jsonl",
    "correctness_rows.jsonl",
    "capture_cost_rows.jsonl",
)
```

Delete or truncate each file in turn and require assembly to fail.

- [ ] **Step 2: Run RED**

Run:

```bash
python tools/test_assemble_tp4_decode_replay.py
```

Expected: import failure because the assembler does not exist.

- [ ] **Step 3: Implement strict assembly**

The assembler must:

- parse JSONL only when the file ends in newline;
- reject duplicates and non-finite values;
- validate source, workload, rank, process, and cleanup identity;
- call `contract.classify()` from raw rows;
- write `summary.json` and `producer_classification.json`;
- hash every verifier input into `manifest.json`;
- use atomic replace for every final file.

- [ ] **Step 4: Write RED verifier mutation tests**

Start from one assembled synthetic `GO_STAGE1_JUSTIFIED` bundle and mutate:

- one manifest SHA;
- source tree SHA;
- model revision;
- workload parameters;
- one output token;
- one rank dispatch kind;
- one graph identity;
- one collective-order digest;
- one rank cleanup;
- replay coverage;
- throughput, TPOT, TTFT, P99 E2E, allocated, and reserved gates;
- producer classification.

The verifier must return the independently reconstructed classification or
`INCOMPLETE`; it must never trust the producer summary.

- [ ] **Step 5: Implement the independent verifier**

`verify_bundle()`:

1. validates manifest inventory and hashes;
2. parses every raw row independently;
3. reconstructs case completeness;
4. reconstructs all-rank agreement;
5. calls the contract classifier on independently loaded rows;
6. compares, but does not trust, producer classification;
7. returns `classification`, `failed_gates`, `verified_hashes`, and compact
   reconstructed metrics.

- [ ] **Step 6: Run GREEN**

Run:

```bash
python tools/test_assemble_tp4_decode_replay.py
python tools/test_verify_tp4_decode_replay.py
python -m py_compile \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add -- \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py \
  tools/test_assemble_tp4_decode_replay.py \
  tools/test_verify_tp4_decode_replay.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): verify TP4 decode replay evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 4: Build the Local Controller and Remote Safety Boundary

**Files:**

- Create: `tools/run_tp4_decode_replay.py`
- Create: `tools/test_run_tp4_decode_replay.py`

**Interfaces:**

- Consumes:
  `tools.qwen38_tp4_communication_profile.select_strict_clean_gpus`,
  established SSH/source-freezing helpers, the worker, assembler, and
  verifier.
- Produces: plan-only receipt, strict-clean admission, one remote launch,
  compact download, remote verification, local verification, and cleanup.

- [ ] **Step 1: Write RED plan and path tests**

Assert:

```python
assert plan["remote_root"].startswith(
    "/data00/home/sitian/tinyllmforge-workspaces/"
)
assert "/tmp" not in plan["remote_root"]
assert plan["selected_gpu_indices"] == [0, 1, 2, 3]
assert plan["world_size"] == 4
assert plan["model_revision"] == (
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
)
```

Reject fewer than four strict-clean GPUs, duplicate UUIDs, source drift,
existing active run tags, and any remote path outside the approved root.

- [ ] **Step 2: Run RED**

Run:

```bash
python tools/test_run_tp4_decode_replay.py
```

Expected: import failure because the controller does not exist.

- [ ] **Step 3: Implement plan-only and strict-clean monitoring**

Reuse the current monitor semantics:

- no external compute process;
- memory and utilization below the frozen thresholds;
- exact UUID/index inventory;
- repeated poll until timeout;
- local process remains responsible for detecting readiness and launching.

Plan-only writes no remote state and does not query GPUs.

- [ ] **Step 4: Write RED orchestration-order tests**

Dependency-inject operations and require:

```text
source freeze
-> SSH/storage preflight
-> strict-clean admission
-> launch
-> wait
-> download compact evidence
-> assemble
-> remote verifier
-> remote post-verification manifest
-> local frozen-source verifier
-> cleanup validation
```

On any failure, cleanup validation still runs. Cleanup failure overrides an
otherwise successful result but does not mask the original operation error.

- [ ] **Step 5: Implement remote launch and verification**

Use one fresh run tag and fresh dynamic ports. Launch exactly one worker
process group. Pass paths only below:

```text
/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/
  tp4-collective-stable-decode-replay/<run-tag>/
```

The remote verifier runs from the frozen source snapshot. After verification,
hash the final remote verifier output into
`remote_post_verification_manifest.json`. Download only the final compact
bundle and required controller receipts.

- [ ] **Step 6: Run GREEN**

Run:

```bash
python tools/test_run_tp4_decode_replay.py
python -m py_compile \
  tools/run_tp4_decode_replay.py \
  tools/test_run_tp4_decode_replay.py
python tools/run_tp4_decode_replay.py plan-only \
  --run-tag 20260831-qwen38-tp4-decode-replay-r1
```

Expected: tests and compilation pass; plan-only records no GPU query, SSH
mutation, or remote launch.

- [ ] **Step 7: Commit Task 4**

```bash
git add -- \
  tools/run_tp4_decode_replay.py \
  tools/test_run_tp4_decode_replay.py
git diff --cached --check
git -c core.hooksPath=/dev/null commit \
  -m "feat(perf): orchestrate TP4 decode replay gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

---

### Task 5: Run the Complete Local Qualification Regression

**Files:**

- Modify only if a test exposes a defect in Task 1–4 files.

**Interfaces:**

- Consumes: all new gate files plus adjacent existing graph and Qwen3.8
  controller tests.
- Produces: fresh local contract evidence before remote execution.

- [ ] **Step 1: Run focused new tests**

```bash
python tools/test_tp4_decode_replay_contract.py
python tools/test_tp4_decode_replay_worker.py
python tools/test_assemble_tp4_decode_replay.py
python tools/test_verify_tp4_decode_replay.py
python tools/test_run_tp4_decode_replay.py
```

- [ ] **Step 2: Run adjacent graph regressions**

```bash
python tools/test_multi_sequence_cuda_graph_gate.py
python tools/test_model_runner_spec_verify.py
```

- [ ] **Step 3: Run adjacent TP4 controller regressions**

```bash
python tools/test_run_qwen38_tp4_collective_reduction.py
python tools/test_qwen38_tp4_collective_reduction_supervisor.py
python tools/test_run_cross_request_wavefront_microgate.py
```

- [ ] **Step 4: Run compilation and diff checks**

```bash
python -m py_compile \
  tools/tp4_decode_replay_contract.py \
  tools/tp4_decode_replay_worker.py \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py \
  tools/run_tp4_decode_replay.py
git diff --check
```

Expected: every command exits zero. Record exact pass counts and elapsed time
for the final audit.

---

### Task 5A: Add Lease-Sealed Transactional Decode Replay

**Files:**

- Modify: `tinyvllm/engine/flash_attn_split_policy.py`
- Modify: `tinyvllm/engine/exact_cuda_graph_cache.py`
- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tinyvllm/models/qwen35_packed.py`
- Modify: `tools/test_model_runner_spec_verify.py`
- Modify: `tools/test_qwen35_prepared_model_step.py`

**Interfaces:**

- Consumes: current ordered `_last_hybrid_state_leases`,
  `_last_hybrid_state_token_counts`, and the existing exact graph identity.
- Produces: a conventional `forward_v1` protocol or opaque
  `lease_transaction_v1` protocol, with capture rollback and exact-once
  replay state transition.

- [ ] **Step 1: RED — identity and first-adopter hook contract**

Add tests proving:

- conventional models retain `execution_protocol == "forward_v1"` with empty
  state schema and lease seal;
- transactional models produce `execution_protocol ==
  "lease_transaction_v1"`;
- changing slot, generation, request, order, or state schema changes the graph
  identity SHA;
- Qwen hook snapshot/restore round-trips the complete active state;
- Qwen full-step hook matches `run_step()` output and state.

Run:

```bash
PYTHONPATH=. python3 tools/test_qwen35_prepared_model_step.py
PYTHONPATH=. python3 tools/test_model_runner_spec_verify.py
```

Expected: fail because the hooks and identity fields do not exist.

- [ ] **Step 2: GREEN — minimal opaque model hooks and identity**

Add exact hook names:

```text
exact_cuda_graph_state_schema_sha256() -> str
exact_cuda_graph_lease_seal(leases) -> str
snapshot_exact_cuda_graph_state(leases) -> object
restore_exact_cuda_graph_state(leases, snapshot) -> None
run_exact_cuda_graph_step(leases, token_counts, input_ids, positions)
  -> logits or None
```

`ModelRunner` may call these hooks only through capability detection. It must
not inspect Qwen classes, layer stacks, component roles, or checkpoint fields.

- [ ] **Step 3: RED — capture rollback and replay lifecycle**

Add tests proving:

- warmup plus capture may mutate scratch KV and lease state internally, but
  both are restored before capture returns;
- successful replay invokes the captured state transition exactly once and
  returns captured logits directly;
- lease/schema drift disables the entry before `graph.replay()`;
- replay or state-restore failure is terminal and never retries eager;
- the existing conventional model capture/replay path remains unchanged.

Run:

```bash
PYTHONPATH=. python3 tools/test_model_runner_spec_verify.py
```

Expected: fail on missing transactional capture/replay behavior.

- [ ] **Step 4: GREEN — minimal transactional capture/replay**

Capture `run_exact_cuda_graph_step(...)` with the exact sealed leases and token
counts. Snapshot active transactional state and scratch KV before warmup,
restore both in `finally`, and preserve both errors if either capture or
restore fails. Retain captured logits and protocol metadata in
`ExactCudaGraphEntry`.

Before replay, rebuild the complete identity from current runtime state. Only
after exact equality may `graph.replay()` begin. Once replay begins, any
failure raises terminally and no eager fallback is allowed.

- [ ] **Step 5: GREEN — focused and compatibility regressions**

Run:

```bash
PYTHONPATH=. python3 tools/test_qwen35_prepared_model_step.py
PYTHONPATH=. python3 tools/test_model_runner_spec_verify.py
PYTHONPATH=. python3 tools/test_multi_sequence_cuda_graph_gate.py
PYTHONPATH=. python3 tools/test_tp4_decode_replay_worker.py
PYTHONPATH=. python3 tools/test_run_tp4_decode_replay.py
python3 -m py_compile \
  tinyvllm/engine/flash_attn_split_policy.py \
  tinyvllm/engine/exact_cuda_graph_cache.py \
  tinyvllm/engine/model_runner.py \
  tinyvllm/models/qwen35_packed.py
git diff --check
```

Expected: all exit zero.

- [ ] **Step 6: Exact commit and push**

Stage only the six files above plus the amended spec and plan. Commit with
exactly one required co-author trailer, push only
`origin/feat/kv-sparse-attention`, and verify local/tracking/remote SHA
equality before r6.

---

### Task 6: Execute the Real Qwen3.8-27B TP4 Gate

**Files:**

- Create locally under:
  `artifacts/tp4_decode_replay/<run-tag>/controller/`
- Create remotely under the approved `/data00/home/sitian/` run root.

**Interfaces:**

- Consumes: committed source SHA, clean GPU admission, frozen model revision,
  Q0/Q1/Q2 case matrix, and all Task 1–5 tools.
- Produces: a complete immutable terminal bundle or an evidence-backed
  `INCOMPLETE` result.

- [x] **Step 1: Confirm source and credential preconditions**

```bash
git status --short -- \
  tools/tp4_decode_replay_contract.py \
  tools/tp4_decode_replay_worker.py \
  tools/assemble_tp4_decode_replay.py \
  tools/verify_tp4_decode_replay.py \
  tools/run_tp4_decode_replay.py
git rev-parse HEAD
klist
```

Do not renew credentials. If SSH fails, preserve controller evidence and stop
the attempt as `INCOMPLETE`; do not create a second run with the same tag.

- [x] **Step 2: Launch the local monitor/controller**

```bash
python tools/run_tp4_decode_replay.py monitor-and-run \
  --run-tag 20260831-qwen38-tp4-decode-replay-r6
```

The controller waits locally, launches immediately after four strict-clean
GPUs are observed, and does not terminate external work.

- [ ] **Step 3: Validate terminal remote evidence**

Require:

- worker exit receipt;
- all four rank lifecycle receipts;
- all 30 measured arm rows;
- no partial JSONL line;
- remote independent verification;
- post-verification manifest;
- cleanup with no owned child remaining.

- [ ] **Step 4: Run local frozen-source verification**

```bash
python tools/verify_tp4_decode_replay.py \
  --bundle artifacts/tp4_decode_replay/\
20260831-qwen38-tp4-decode-replay-r6/final_bundle \
  --write-result artifacts/tp4_decode_replay/\
20260831-qwen38-tp4-decode-replay-r6/controller/\
local_frozen_source_verification.json
```

The local classification must equal the producer and remote verifier
classification. Any disagreement is `INCOMPLETE`.

---

### Task 7: Write the Terminal Audit and Reconcile Design/Plan

**Files:**

- Create:
  `docs/superpowers/audits/2026-08-31-tp4-collective-stable-decode-replay-audit.md`
- Modify:
  `docs/superpowers/specs/2026-08-31-tp4-collective-stable-decode-replay-qualification-design.md`
- Modify:
  `docs/superpowers/plans/2026-08-31-tp4-collective-stable-decode-replay-qualification.md`

**Interfaces:**

- Consumes: immutable bundle, all three classifications, local test logs, and
  remote/local SHA evidence.
- Produces: one exact terminal statement and a prompt-to-artifact audit.

- [x] **Step 1: Write the benefit/cost result table**

For Q0/Q1/Q2 and aggregate, report:

- output throughput ratio;
- median TPOT ratio;
- P99 TPOT ratio;
- P99 E2E ratio;
- TTFT ratio;
- replay coverage;
- capture duration;
- amortization tokens;
- per-rank allocated and reserved deltas.

- [x] **Step 2: State the exact evidence boundary**

If `GO_STAGE1_JUSTIFIED`, state only that Stage 1 is justified. Do not claim
production readiness.

If any `NO_GO`, stop the direction and name the failed thresholds and costs.

If `INCOMPLETE`, name each missing artifact or unverifiable field and make no
performance claim.

- [x] **Step 3: Build the prompt-to-artifact checklist**

Map every requirement in the design to:

- exact file;
- exact row/field;
- verifier check;
- result;
- limitation.

Include source, model revision, topology, strict-clean admission, remote root,
Q0/Q1/Q2 matrix, exact output parity, all-rank dispatch, collective order,
replay coverage, benefit, cost, cleanup, manifest, dual verifier, tests,
commit, push, and remote SHA.

- [x] **Step 4: Reconcile spec and plan**

Add terminal classification, immutable bundle path, verifier result paths,
measured summary, and Stage-1 authorization/prohibition. Mark completed plan
checkboxes only when their named evidence exists.

---

### Task 8: Final Verification, Exact Commit, and Push

**Files:**

- Stage only the new gate files, the compact terminal artifact, the audit,
  and the two reconciled docs.

**Interfaces:**

- Consumes: all terminal deliverables.
- Produces: one pushed evidence commit with local/tracking/remote SHA equality.

- [ ] **Step 1: Re-run the complete local regression**

Run every command from Task 5 and the local frozen-source verifier from
Task 6. All must use the final source.

- [ ] **Step 2: Audit artifact inventory**

Programmatically assert every required compact artifact exists, every
manifest SHA matches, and no raw large trace or unrelated historical artifact
is staged.

- [ ] **Step 3: Inspect exact staged paths**

```bash
git status --short
git diff --check
git diff --stat
```

Stage only an explicit path list. Do not use a wildcard that could include
historical untracked artifacts.

- [ ] **Step 4: Commit**

```bash
git -c core.hooksPath=/dev/null commit \
  -m "perf(runtime): qualify TP4 decode replay" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Verify exactly one matching trailer:

```bash
git show -s --format=%B HEAD | \
  grep -c '^Co-authored-by: TRAE CLI <noreply@bytedance.com>$'
```

Expected: `1`.

- [ ] **Step 5: Push and verify three-way SHA equality**

```bash
git push origin feat/kv-sparse-attention
git rev-parse HEAD
git rev-parse @{u}
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: local HEAD, tracking SHA, and remote SHA are identical.

---

## r25 Terminal Reconciliation and r26 Checkpoint

Attempt `20260901-qwen38-tp4-decode-replay-r25-full` is consumed as
`INCOMPLETE_EXTERNAL_PREEMPTION`.

- [x] Frozen source `1e18c30e5cf134943b39f984100583b2b1a3f55d`
- [x] Strict-clean admission on GPUs `0,3,4,6`
- [x] Local continuous ownership guard
- [x] Safe response to external PID `3877390`
- [x] Exact-tag-owned process cleanup with final environment match set `[]`
- [x] Thirteen atomic case files and six complete pairs preserved remotely
- [ ] Complete 30-case / 15-pair matrix
- [ ] Immutable final bundle and manifest
- [ ] Remote independent verifier
- [ ] Local frozen-source verifier
- [ ] Terminal producer/verifier classification

TDD corrections before r26:

- [x] cmdline ownership requires the full attempt root rather than any mention
  of the short run tag;
- [x] measured dispatch rows, not warmup rows, source
  `capture_cost_rows`;
- [x] focused RED failures observed for both defects;
- [x] focused GREEN and complete worker/controller script suites pass.

The next executable run is:

```text
tag:
  20260901-qwen38-tp4-decode-replay-r26-full
command timeout:
  default 21,600 seconds
minimum Kerberos lifetime:
  22,500 seconds
```

Do not launch until the external ticket satisfies the full window. Do not run
`kinit` or `krenew`. Once the prerequisite is available, execute the complete
unchanged 30-case matrix from the newly committed source and continue with
Tasks 6–8.

## Plan Self-Review

- Spec coverage: every objective, gate, artifact, safety rule, and evidence
  boundary maps to Tasks 1–8.
- Scope: one Stage-0 qualification plus the r5-authorized generic
  lease-sealed transactional repair; no distributed admission implementation,
  dynamic pool-index protocol, or production-default change.
- Type consistency: shared function names and constants are defined once in
  this plan and reused unchanged.
- TDD: each code-producing task has explicit RED, minimal implementation,
  GREEN, and commit steps.
- Placeholder scan: every code-producing step names its implementation and
  expected test evidence.
- Evidence discipline: producer, remote verifier, local frozen-source
  verifier, manifest, cleanup, and prompt-to-artifact audit are independent
  completion requirements.
