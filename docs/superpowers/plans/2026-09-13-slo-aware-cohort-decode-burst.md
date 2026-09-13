# SLO-Aware Cohort Decode Burst Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and qualify a default-disabled multi-request exact-greedy
cohort decode burst that improves output throughput by at least 10% while
protecting P99 ITL, TTFT, E2E latency, fairness, correctness, and memory.

**Architecture:** The scheduler continues to choose the ordered decode cohort.
A pure SLO policy chooses `K` from `1, 2, 4, 8` using a frozen P99 cost table
and the minimum slack of every request the burst can block. A separate
multi-row lease/result contract binds graph execution to scheduler-owned KV
and publication state, and the scheduler commits all per-request prefixes
atomically.

**Tech Stack:** Python 3, dataclasses, PyTorch, CUDA Graphs, pytest, JSON/JSONL,
SHA-256 manifests, TinyLLMForge scheduler and ModelRunner.

## Global Constraints

- Authoritative checkout:
  `/Users/bytedance/Desktop/TinyLLMForge`.
- The feature is default-disabled.
- Stage 0 and Stage 1 use Qwen3-0.6B, TP1, one NVIDIA A100 80GB PCIe.
- Supported burst widths are exactly `1, 2, 4, 8`.
- The controller never changes scheduler cohort membership or request order.
- The controller computes slack over the selected cohort, omitted runnable
  decode requests, waiting requests, and incomplete-prefill requests.
- The canonical candidate consumes a cost table frozen before candidate
  execution and never retunes it from candidate results.
- A graph replay remains one exact target-model step. K8 means eight ordered
  replays, not one forward producing eight unknown tokens.
- Before replay, a recognized failure may close the lease and fall back to
  ordinary K1. After replay begins, same-step eager/K1 retry is forbidden.
- EOS commits only the prefix through the first EOS and records all post-EOS
  device work as waste.
- Cohort publication is atomic across all rows.
- New task data on the remote host is written only below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Large traces and profiler databases remain remote; the local checkout
  receives only the compact final bundle and reconstruction evidence.
- Existing unrelated staged files must remain untouched. Every commit uses
  `git commit --only` with the task's exact paths.
- Stage 0 stops the project as `NO_GO_CEILING` if both medium and high load
  have less than 12% optimistic throughput headroom.
- Final GO requires:

```text
aggregate output-throughput improvement >= 10%
medium-load throughput improvement      >= 10%
high-load throughput improvement        >= 10%
any workload throughput regression      <= 2%
any workload P99 ITL regression         <= 3%
any workload P99 TTFT regression        <= 5%
any workload P99 E2E regression         <= 5%
maximum host-visible gap                <= 40 ms
starved requests                         = 0
post-EOS wasted-forward fraction        <= 10%
peak CUDA reserved-memory regression    <= 5%
all transaction inventories             closed
remote and local verifier               agree
```

---

## File Map

### New runtime files

- `tinyvllm/engine/slo_cohort_burst.py`: immutable cost-table schema, request
  SLO state, slack calculation, deterministic width selection, suppression
  reasons, and decision telemetry.
- `tinyvllm/engine/exact_greedy_cohort_burst.py`: cohort row authority,
  lease/result identities, EOS-prefix validation, transaction state, graph
  health, and execution statistics.

### Existing runtime files

- `tinyvllm/config.py`: default-disabled configuration and strict validation.
- `tinyvllm/engine/scheduler.py`: request timing hooks, protected-request
  snapshots, cohort lease ownership, atomic prepare/commit/abort, and
  lifecycle summary.
- `tinyvllm/engine/model_runner.py`: TP1 multi-row graph cache, capture,
  binding, replay, one final D2H, and quarantine.
- `tinyvllm/engine/llm_engine.py`: orchestration only; no policy logic.

### New test files

- `tools/test_slo_cohort_burst.py`: pure policy and cost-table tests.
- `tools/test_exact_greedy_cohort_burst.py`: lease/result/state-machine tests.
- `tools/test_scheduler_slo_cohort_burst.py`: fake-clock scheduler tests.
- `tools/test_llm_engine_slo_cohort_burst.py`: engine orchestration and failure
  boundary tests.
- `tools/test_slo_cohort_burst_ceiling.py`: Stage-0 reconstruction tests.
- `tools/test_slo_cohort_burst_gate.py`: Stage-1/2 artifact and classification
  tests.
- `tools/test_slo_cohort_burst_verify.py`: independent verifier mutation tests.
- `tools/test_run_slo_cohort_burst_remote.py`: remote path, snapshot, resume,
  and bundle tests.

### New qualification tools

- `tools/profile_slo_cohort_burst_ceiling.py`: baseline attribution and frozen
  P99 cost-table producer.
- `tools/slo_cohort_burst_ceiling.py`: deterministic Stage-0 reconstruction
  and `NO_GO_CEILING` classification.
- `tools/slo_cohort_burst_gate.py`: workload generation, correctness matrix,
  canonical aggregation, and formal classification.
- `tools/slo_cohort_burst_verify.py`: local independent reconstruction.
- `tools/run_slo_cohort_burst_remote.py`: source snapshot, remote execution,
  resume, compact bundle, and remote verifier.

---

### Task 1: Freeze the Stage-0 Ceiling Contract

**Files:**

- Create: `tools/slo_cohort_burst_ceiling.py`
- Create: `tools/test_slo_cohort_burst_ceiling.py`

**Interfaces:**

- Produces:
  `SLOCohortCostKey(batch_size, context_bucket, burst_width)`,
  `build_frozen_cost_table(rows, source_identity) -> dict`,
  `classify_ceiling(summary) -> str`, and
  `verify_ceiling_artifact(artifact) -> dict`.
- Consumes: raw baseline timing rows only; no candidate runtime data.

- [ ] **Step 1: Write failing schema and nearest-rank tests**

```python
def test_cost_table_is_source_bound_and_uses_nearest_rank_p99():
    rows = [
        {
            "batch_size": 4,
            "context_bucket": 2048,
            "burst_width": 4,
            "duration_ns": value,
        }
        for value in (10, 20, 30, 40, 50)
    ]
    table = build_frozen_cost_table(rows, SOURCE_IDENTITY)
    key = "b4-c2048-k4"
    assert table["entries"][key]["p99_ns"] == 50
    assert table["source_identity"] == SOURCE_IDENTITY
    assert len(table["table_sha256"]) == 64


def test_ceiling_stops_only_when_medium_and_high_are_below_twelve_percent():
    assert classify_ceiling({
        "medium_headroom_ratio": 0.119,
        "high_headroom_ratio": 0.118,
    }) == "NO_GO_CEILING"
    assert classify_ceiling({
        "medium_headroom_ratio": 0.120,
        "high_headroom_ratio": 0.050,
    }) == "CONTINUE_RUNTIME"
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
pytest -q tools/test_slo_cohort_burst_ceiling.py
```

Expected: collection fails because `tools.slo_cohort_burst_ceiling` does not
exist.

- [ ] **Step 3: Implement canonical schemas and classification**

```python
@dataclass(frozen=True, order=True)
class SLOCohortCostKey:
    batch_size: int
    context_bucket: int
    burst_width: int


def nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def classify_ceiling(summary: Mapping[str, float]) -> str:
    medium = float(summary["medium_headroom_ratio"])
    high = float(summary["high_headroom_ratio"])
    if medium < 0.12 and high < 0.12:
        return "NO_GO_CEILING"
    return "CONTINUE_RUNTIME"
```

`build_frozen_cost_table` must sort keys, include raw sample digests,
P50/P95/P99/sample count, canonical JSON encoding, and a SHA-256 excluding
only the `table_sha256` field itself. `verify_ceiling_artifact` recomputes
every aggregate and rejects duplicate keys, missing load levels, non-finite
values, source drift, and hash drift.

- [ ] **Step 4: Run focused tests and confirm GREEN**

Run:

```bash
pytest -q tools/test_slo_cohort_burst_ceiling.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add -N -- \
  tools/slo_cohort_burst_ceiling.py \
  tools/test_slo_cohort_burst_ceiling.py
git commit --only -m "feat(inference): define cohort burst ceiling contract" -- \
  tools/slo_cohort_burst_ceiling.py \
  tools/test_slo_cohort_burst_ceiling.py
```

### Task 2: Measure Stage-0 Baseline Headroom

**Files:**

- Create: `tools/profile_slo_cohort_burst_ceiling.py`
- Create: `tools/run_slo_cohort_burst_remote.py`
- Create: `tools/test_run_slo_cohort_burst_remote.py`
- Modify: `tools/test_slo_cohort_burst_ceiling.py`

**Interfaces:**

- Consumes: `build_frozen_cost_table` and `verify_ceiling_artifact`.
- Produces:
  `profile_baseline_case(engine, case, clock) -> dict`,
  `build_ceiling_summary(rows) -> dict`, and the immutable remote files
  `raw_rows.jsonl`, `cost_table.json`, `ceiling_summary.json`,
  `source_manifest.json`, and `remote_verify.json`.

- [ ] **Step 1: Write failing profiler and remote-path tests**

```python
def test_profiler_attributes_complete_step_time_without_double_counting():
    row = profile_baseline_case(FakeEngine(), CASE, FakeClock())
    components = row["component_ns"]
    assert sum(components.values()) == row["wall_ns"]
    assert set(components) == {
        "target_cuda",
        "graph_launch_gap",
        "scheduler",
        "token_d2h_publication",
        "batch_binding",
        "unattributed",
    }


def test_remote_runner_uses_only_large_mount():
    args = build_remote_paths("20260913-stage0-r1")
    root = "/data00/home/sitian/tinyllmforge-workspaces/" \
        "command-timeline-20260818/"
    assert all(str(path).startswith(root) for path in args.values())
```

- [ ] **Step 2: Run the tests and confirm RED**

Run:

```bash
pytest -q \
  tools/test_slo_cohort_burst_ceiling.py \
  tools/test_run_slo_cohort_burst_remote.py
```

Expected: imports or missing profiler/runner symbols fail.

- [ ] **Step 3: Implement the baseline profiler**

The profiler must run ordinary multi-request decode at batch sizes
`1, 2, 4, 8`, preserve the offered arrival window, and record:

```python
{
    "schema": "slo_cohort_burst_ceiling_row_v1",
    "case_id": case.case_id,
    "load": case.load,
    "batch_size": case.batch_size,
    "context_bucket": case.context_bucket,
    "burst_width": case.burst_width,
    "component_ns": component_ns,
    "wall_ns": wall_ns,
    "committed_tokens": committed_tokens,
    "cuda_reserved_bytes": cuda_reserved_bytes,
}
```

The optimistic headroom removes only host costs that the proposed cohort
burst can amortize. Target-model CUDA time and irreducible scheduler work
remain in the denominator.

- [ ] **Step 4: Implement source-exact remote execution**

The runner must:

1. reject a Kerberos ticket with less than the configured guard;
2. snapshot the committed source and record its SHA;
3. create a fresh tag under the approved remote root;
4. refuse `/`, `/tmp`, and retired checkout paths;
5. run calibration and remote verification;
6. retain raw traces remotely;
7. copy back only compact JSON/JSONL, manifests, hashes, and logs;
8. support immutable resume without silently replacing completed evidence.

- [ ] **Step 5: Run local contract tests**

Run:

```bash
pytest -q \
  tools/test_slo_cohort_burst_ceiling.py \
  tools/test_run_slo_cohort_burst_remote.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit only Stage-0 tooling**

```bash
git add -N -- \
  tools/profile_slo_cohort_burst_ceiling.py \
  tools/run_slo_cohort_burst_remote.py \
  tools/test_run_slo_cohort_burst_remote.py
git commit --only -m "feat(inference): add cohort burst ceiling profiler" -- \
  tools/profile_slo_cohort_burst_ceiling.py \
  tools/slo_cohort_burst_ceiling.py \
  tools/test_slo_cohort_burst_ceiling.py \
  tools/run_slo_cohort_burst_remote.py \
  tools/test_run_slo_cohort_burst_remote.py
```

- [ ] **Step 7: Run the Stage-0 remote gate**

Run with a fresh tag:

```bash
python tools/run_slo_cohort_burst_remote.py \
  --stage ceiling \
  --tag 20260913-slo-cohort-ceiling-r1
```

Expected terminal artifact:

```text
classification = CONTINUE_RUNTIME
```

If the terminal artifact instead reports `NO_GO_CEILING`, verify it locally,
commit the compact negative bundle and retrospective, push, and stop Tasks
3-10. Do not implement the cohort runtime.

### Task 3: Add Strict Default-Off Configuration

**Files:**

- Modify: `tinyvllm/config.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Produces these `Config` fields:
  `exact_greedy_cohort_burst`,
  `exact_greedy_cohort_burst_widths`,
  `exact_greedy_cohort_burst_max_batch_size`,
  `exact_greedy_cohort_burst_target_itl_ns`,
  `exact_greedy_cohort_burst_target_ttft_ns`,
  `exact_greedy_cohort_burst_reserve_ns`, and
  `exact_greedy_cohort_burst_cost_table_path`.
- Requires the existing `exact_greedy_decode_burst=True` capability.

- [ ] **Step 1: Add failing config tests**

```python
def test_slo_cohort_burst_config_is_strict_and_default_off():
    fields_by_name = {field.name: field for field in fields(Config)}
    assert fields_by_name["exact_greedy_cohort_burst"].default is False
    assert fields_by_name[
        "exact_greedy_cohort_burst_widths"
    ].default == (1, 2, 4, 8)
    with pytest.raises(ValueError, match="requires exact_greedy_decode_burst"):
        Config(model=MODEL_DIR, exact_greedy_cohort_burst=True)
```

Add parametrized failures for booleans used as integers, unsorted/duplicate
widths, widths outside `1,2,4,8`, reserve greater than or equal to either SLO,
missing cost-table path, TP greater than one, and max batch outside `1..8`.

- [ ] **Step 2: Run the exact config test and confirm RED**

Run:

```bash
pytest -q \
  tools/test_model_runner_spec_verify.py \
  -k "slo_cohort_burst_config"
```

Expected: missing dataclass fields fail.

- [ ] **Step 3: Implement the fields and validation**

```python
exact_greedy_cohort_burst: bool = False
exact_greedy_cohort_burst_widths: tuple = (1, 2, 4, 8)
exact_greedy_cohort_burst_max_batch_size: int = 8
exact_greedy_cohort_burst_target_itl_ns: int = 0
exact_greedy_cohort_burst_target_ttft_ns: int = 0
exact_greedy_cohort_burst_reserve_ns: int = 0
exact_greedy_cohort_burst_cost_table_path: str | None = None
```

Normalize widths with `_normalize_positive_int_tuple`, then reject any tuple
other than a prefix-compatible subset containing one and drawn from
`(1, 2, 4, 8)`. Enabled mode requires positive TTFT/ITL targets, reserve below
both targets, TP1, a non-empty cost-table path, and the base exact burst.

- [ ] **Step 4: Run focused config tests and confirm GREEN**

Run:

```bash
pytest -q \
  tools/test_model_runner_spec_verify.py \
  -k "exact_greedy_decode_burst_config or slo_cohort_burst_config"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit exact config paths**

```bash
git commit --only -m "feat(inference): configure SLO cohort burst" -- \
  tinyvllm/config.py \
  tools/test_model_runner_spec_verify.py
```

### Task 4: Implement the Pure SLO Width Controller

**Files:**

- Create: `tinyvllm/engine/slo_cohort_burst.py`
- Create: `tools/test_slo_cohort_burst.py`

**Interfaces:**

- Produces:
  `RequestSLOState`,
  `ProtectedRequestSnapshot`,
  `SLOCohortCostTable.load(path)`,
  `SLOCohortBurstObservation`,
  `SLOCohortBurstDecision`, and
  `select_slo_cohort_burst_width(observation, cost_table)`.
- Does not import `Scheduler`, `ModelRunner`, or mutable `Sequence`.

- [ ] **Step 1: Write failing selector tests**

```python
def test_selector_uses_minimum_slack_across_all_protected_requests():
    decision = select_slo_cohort_burst_width(
        observation=make_observation(
            cohort_slacks=(90_000_000, 80_000_000),
            omitted_decode_slacks=(19_000_000,),
            waiting_slacks=(50_000_000,),
        ),
        cost_table=make_table(k8=30_000_000, k4=18_000_000, k2=9_000_000),
    )
    assert decision.global_slack_ns == 19_000_000
    assert decision.selected_width == 4


def test_selector_fallback_precedence_is_stable():
    decision = select_slo_cohort_burst_width(
        observation=make_observation(
            enabled=False,
            clock_valid=False,
            missing_slo_state=True,
        ),
        cost_table=make_invalid_table(),
    )
    assert decision.selected_width == 1
    assert decision.reason == "disabled"
```

- [ ] **Step 2: Run the policy tests and confirm RED**

Run:

```bash
pytest -q tools/test_slo_cohort_burst.py
```

Expected: module import fails.

- [ ] **Step 3: Implement immutable policy types**

```python
@dataclass(frozen=True)
class RequestSLOState:
    sequence_id: int
    arrival_ns: int
    first_token_visible_ns: int | None
    last_token_visible_ns: int | None
    service_class: str


@dataclass(frozen=True)
class SLOCohortBurstDecision:
    selected_width: int
    reason: str
    global_slack_ns: int
    predicted_cost_ns_by_width: tuple[tuple[int, int], ...]
    protected_sequence_ids: tuple[int, ...]
```

Implement the exact fallback precedence from the design. For emitted requests
use ITL slack; for requests without a first token use TTFT slack. Reject
missing/future/decreasing timestamps and missing cost keys. Iterate widths in
`(8, 4, 2)` order and return the first structurally eligible cost not greater
than global slack; otherwise return K1.

- [ ] **Step 4: Run pure policy tests and confirm GREEN**

Run:

```bash
pytest -q tools/test_slo_cohort_burst.py
```

Expected: all tests pass without CUDA or model imports.

- [ ] **Step 5: Commit policy files**

```bash
git add -N -- \
  tinyvllm/engine/slo_cohort_burst.py \
  tools/test_slo_cohort_burst.py
git commit --only -m "feat(inference): add SLO cohort width controller" -- \
  tinyvllm/engine/slo_cohort_burst.py \
  tools/test_slo_cohort_burst.py
```

### Task 5: Add Scheduler-Owned SLO State

**Files:**

- Modify: `tinyvllm/engine/scheduler.py`
- Create: `tools/test_scheduler_slo_cohort_burst.py`

**Interfaces:**

- Consumes: `RequestSLOState`, `ProtectedRequestSnapshot`, and the pure
  selector from Task 4.
- Produces:
  `register_slo_request(seq, arrival_ns, service_class)`,
  `record_slo_publication(seq_id, visible_ns)`,
  `remove_slo_request(seq_id)`,
  `build_slo_cohort_observation(seqs, decision_now_ns, ...)`, and
  `select_slo_cohort_burst(seqs, decision_now_ns, ...)`.

- [ ] **Step 1: Write failing fake-clock lifecycle tests**

```python
def test_waiting_request_contracts_width_without_reordering_cohort():
    scheduler, cohort = make_scheduler_with_fake_clock()
    scheduler.register_slo_request(cohort[0], 0, "default")
    waiting = add_waiting_request(scheduler, arrival_ns=95)
    decision = scheduler.select_slo_cohort_burst(
        tuple(cohort),
        decision_now_ns=100,
        graph_capability=CAPABILITY,
    )
    assert decision.ordered_cohort_ids == tuple(
        seq.seq_id for seq in cohort
    )
    assert waiting.seq_id in decision.protected_sequence_ids


def test_completion_and_cancellation_remove_slo_state():
    scheduler, seq = make_running_sequence()
    scheduler.register_slo_request(seq, 10, "default")
    scheduler.remove_slo_request(seq.seq_id)
    assert seq.seq_id not in scheduler.slo_request_state_by_seq_id
```

- [ ] **Step 2: Run scheduler tests and confirm RED**

Run:

```bash
pytest -q tools/test_scheduler_slo_cohort_burst.py
```

Expected: scheduler methods are missing.

- [ ] **Step 3: Implement independent scheduler state**

Initialize:

```python
self.slo_request_state_by_seq_id: dict[int, RequestSLOState] = {}
self._last_slo_cohort_decision: SLOCohortBurstDecision | None = None
```

Use immutable replacement on publication. Do not add fields to
`Sequence.__getstate__` or `Sequence.__setstate__`. Snapshot waiting,
prefilling, running, and selected cohort IDs from existing scheduler queues;
deduplicate by sequence ID while preserving deterministic category order.

- [ ] **Step 4: Wire admission, publication, preemption, and terminal hooks**

Admission records the monotonic arrival timestamp. First publication sets both
`first_token_visible_ns` and `last_token_visible_ns`; later publication changes
only the latter. Preemption retains state. Completion, cancellation, and
terminal failure remove state. A clock rollback sets a sticky invalid reason
and forces K1.

- [ ] **Step 5: Run scheduler and serialization regression tests**

Run:

```bash
pytest -q \
  tools/test_scheduler_slo_cohort_burst.py \
  tools/test_hybrid_state_sequence.py
```

Expected: fake-clock tests pass and the existing Sequence serialization shape
is unchanged.

- [ ] **Step 6: Commit scheduler state**

```bash
git add -N -- tools/test_scheduler_slo_cohort_burst.py
git commit --only -m "feat(inference): track cohort burst SLO state" -- \
  tinyvllm/engine/scheduler.py \
  tools/test_scheduler_slo_cohort_burst.py
```

### Task 6: Implement the Cohort Lease and Atomic Result Contract

**Files:**

- Create: `tinyvllm/engine/exact_greedy_cohort_burst.py`
- Create: `tools/test_exact_greedy_cohort_burst.py`

**Interfaces:**

- Produces:
  `CohortWriteAuthority`,
  `ExactGreedyCohortBurstLease`,
  `ExactGreedyCohortBurstRowResult`,
  `ExactGreedyCohortBurstResult`,
  `ExactGreedyCohortBurstFallback`,
  `ExactGreedyCohortBurstTransaction`,
  `build_exact_greedy_cohort_burst_lease(...)`, and
  `validate_exact_greedy_cohort_burst_result(...)`.

- [ ] **Step 1: Write failing identity and atomicity tests**

```python
def test_cohort_identity_binds_order_and_every_write_authority():
    lease = build_lease(rows=(row(7), row(9)), width=4)
    reversed_result = build_result(
        lease,
        rows=(result_row(9), result_row(7)),
    )
    with pytest.raises(ValueError, match="ordered sequence IDs"):
        validate_exact_greedy_cohort_burst_result(
            lease,
            reversed_result,
            eos_token_id=2,
        )


def test_eos_prefix_is_committed_and_suffix_is_counted_as_waste():
    lease = build_lease(rows=(row(7), row(9)), width=4)
    result = build_result(
        lease,
        tokens=((11, 2, 91, 92), (21, 22, 23, 24)),
    )
    validated = validate_exact_greedy_cohort_burst_result(
        lease,
        result,
        eos_token_id=2,
    )
    assert validated.commit_tokens == ((11, 2), (21, 22, 23, 24))
    assert validated.wasted_post_eos_tokens == 2
```

- [ ] **Step 2: Run contract tests and confirm RED**

Run:

```bash
pytest -q tools/test_exact_greedy_cohort_burst.py
```

Expected: module import fails.

- [ ] **Step 3: Implement canonical lease identity**

Each `CohortWriteAuthority` binds sequence generation, block-table identity,
writable block IDs and generations, logical position range, physical slot
range, initial completion count, and remaining output budget. Canonically
encode ordered rows and scalar lease fields with sorted JSON separators, then
SHA-256 the bytes.

```python
@dataclass(frozen=True)
class ExactGreedyCohortBurstLease:
    schedule_generation: int
    graph_generation: int
    ordered_sequence_ids: tuple[int, ...]
    requested_width: int
    authorized_width: int
    decision_now_ns: int
    cost_table_sha256: str
    predicted_duration_ns: int
    global_slack_ns: int
    rows: tuple[CohortWriteAuthority, ...]
    identity_sha256: str
```

- [ ] **Step 4: Implement fail-closed result validation**

Validate lease, graph, row order, row count, replay count, token count,
positions, context lengths, slots, D2H counts, finite sampled logits, and
argmax equality. Return a validated publication object only after all rows
pass. The transaction states are exactly:

```text
reserved -> dispatched -> validated -> committed
reserved -> cancelled
dispatched -> quarantined -> failed
```

Illegal transitions raise before mutation. A fallback object requires zero
replays.

- [ ] **Step 5: Run property and state-machine tests**

Run:

```bash
pytest -q tools/test_exact_greedy_cohort_burst.py
```

Expected: all tests pass, including duplicate commit, stale identity,
overlapping unauthorized slots, post-replay fallback, and pending-inventory
cases.

- [ ] **Step 6: Commit contract files**

```bash
git add -N -- \
  tinyvllm/engine/exact_greedy_cohort_burst.py \
  tools/test_exact_greedy_cohort_burst.py
git commit --only -m "feat(inference): add exact cohort burst transaction" -- \
  tinyvllm/engine/exact_greedy_cohort_burst.py \
  tools/test_exact_greedy_cohort_burst.py
```

### Task 7: Add the TP1 Multi-Row Cohort Graph Runtime

**Files:**

- Modify: `tinyvllm/engine/model_runner.py`
- Modify: `tools/test_exact_greedy_cohort_burst.py`
- Modify: `tools/test_model_runner_spec_verify.py`

**Interfaces:**

- Consumes: `ExactGreedyCohortBurstLease`.
- Produces:
  `exact_greedy_cohort_burst_capability(...)`,
  `capture_exact_greedy_cohort_burst_graph(...)`,
  `run_exact_greedy_cohort_burst(lease, seqs, correctness_trace=False)`, and
  `quarantine_exact_greedy_cohort_burst_graph(identity, reason)`.

- [ ] **Step 1: Write failing capture-isolation and replay tests**

```python
def test_cohort_capture_uses_private_scratch_and_row_indexed_tensors():
    graph = capture_with_fake_runner(batch_size=4)
    assert graph.static_input_tokens.shape == (4,)
    assert graph.static_context_lengths.shape == (4,)
    assert graph.static_token_history.shape == (4, 8)
    assert graph.capture_live_kv_mutations == ()


def test_cohort_replay_runs_k_steps_then_one_token_history_d2h():
    graph = make_fake_cohort_graph(batch_size=4)
    result = graph.replay(build_lease(batch_size=4, width=8))
    assert graph.logical_replays == 8
    assert result.token_d2h_calls == 1
    assert tuple(len(row.tokens) for row in result.rows) == (8, 8, 8, 8)
```

- [ ] **Step 2: Run ModelRunner tests and confirm RED**

Run:

```bash
pytest -q \
  tools/test_exact_greedy_cohort_burst.py \
  tools/test_model_runner_spec_verify.py \
  -k "cohort_burst"
```

Expected: graph/runtime symbols are missing.

- [ ] **Step 3: Implement graph identity and static storage**

Key cache entries by:

```python
(
    batch_size,
    block_table_width,
    str(dtype),
    device_identity,
    tensor_parallel_size,
    correctness_trace,
)
```

Do not key on burst width. Allocate row-indexed input tokens, positions,
context lengths, slot mappings, padded block tables, active masks,
`[batch_size, 8]` token histories, history indices, and EOS observations.
Capture only against scheduler-inaccessible scratch KV blocks.

- [ ] **Step 4: Implement bind, replay, D2H, and quarantine**

Before replay, validate all identities and copy every row's static inputs.
Replay the complete-step graph `authorized_width` times. Keep feedback and
history on device, then copy token histories, EOS observations, and sampled
correctness logits once. Any failure after the first launch records completed
replays, quarantines the exact graph identity, and raises a typed terminal
error; it never returns a fallback.

- [ ] **Step 5: Run focused runtime regressions**

Run:

```bash
pytest -q \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_exact_greedy_cohort_burst.py \
  tools/test_model_runner_spec_verify.py \
  -k "exact_greedy_decode_burst or cohort_burst"
```

Expected: batch-one behavior remains green and cohort tests pass.

- [ ] **Step 6: Commit ModelRunner runtime**

```bash
git commit --only -m "feat(inference): execute TP1 exact cohort bursts" -- \
  tinyvllm/engine/model_runner.py \
  tinyvllm/engine/exact_greedy_cohort_burst.py \
  tools/test_exact_greedy_cohort_burst.py \
  tools/test_model_runner_spec_verify.py
```

### Task 8: Integrate Scheduler and Engine Atomic Publication

**Files:**

- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tinyvllm/engine/llm_engine.py`
- Modify: `tools/test_scheduler_slo_cohort_burst.py`
- Create: `tools/test_llm_engine_slo_cohort_burst.py`

**Interfaces:**

- Scheduler produces:
  `prepare_exact_greedy_cohort_burst(...)`,
  `cancel_exact_greedy_cohort_burst(lease, reason)`,
  `prepare_exact_greedy_cohort_burst_commit(...)`, and
  `fail_exact_greedy_cohort_burst(lease, terminal=True)`.
- Engine calls the controller only in the non-speculative decode branch after
  the scheduler has frozen the exact ordered cohort.

- [ ] **Step 1: Write failing engine orchestration tests**

```python
def test_engine_commits_all_cohort_prefixes_once_in_scheduler_order():
    engine = make_engine(width=4, eos_rows={1: 2})
    outputs = engine.step()
    assert [row.sequence_id for row in outputs] == [7, 9, 11, 13]
    assert engine.scheduler.commit_count == 1
    assert engine.model_runner.replay_count == 4


def test_post_replay_failure_is_terminal_without_k1_retry():
    engine = make_engine(fail_after_replay=2)
    with pytest.raises(RuntimeError, match="cohort burst"):
        engine.step()
    assert engine.model_runner.ordinary_forward_count == 0
    assert engine.scheduler.pending_cohort_lease is None
```

- [ ] **Step 2: Run engine tests and confirm RED**

Run:

```bash
pytest -q \
  tools/test_scheduler_slo_cohort_burst.py \
  tools/test_llm_engine_slo_cohort_burst.py
```

Expected: scheduler and engine integration methods are missing.

- [ ] **Step 3: Implement scheduler lease preparation**

Use the selected cohort without filtering or reordering. Clip width by the
minimum output budget and writable capacity. Bind every block identity and
physical slot range. Store exactly one pending cohort transaction. A
pre-replay error closes it and returns ordinary K1 eligibility.

- [ ] **Step 4: Implement atomic prepared postprocess**

Construct one `PreparedSchedulerPostprocess` containing one
`ScheduledOutputRow` per sequence and each EOS-truncated prefix. Validate all
rows before calling `commit_prepared_postprocess`. On any validation error,
publish zero rows and follow the terminal path.

- [ ] **Step 5: Wire engine execution without embedding policy**

The engine flow is:

```python
decision = scheduler.select_slo_cohort_burst(...)
if decision.selected_width == 1:
    return ordinary_decode()
lease = scheduler.prepare_exact_greedy_cohort_burst(...)
try:
    result = model_runner.run_exact_greedy_cohort_burst(lease, seqs)
except BaseException:
    scheduler.fail_exact_greedy_cohort_burst(lease, terminal=True)
    raise
if isinstance(result, ExactGreedyCohortBurstFallback):
    scheduler.cancel_exact_greedy_cohort_burst(
        lease,
        result.fallback_reason,
    )
    return ordinary_decode()
prepared = scheduler.prepare_exact_greedy_cohort_burst_commit(
    seqs,
    lease,
    result,
)
scheduler.commit_prepared_postprocess(prepared)
```

Fallback is a returned typed value matching the existing batch-one
convention; it is never raised as an exception.

- [ ] **Step 6: Run focused scheduler/engine tests**

Run:

```bash
pytest -q \
  tools/test_scheduler_slo_cohort_burst.py \
  tools/test_llm_engine_slo_cohort_burst.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit integration**

```bash
git add -N -- tools/test_llm_engine_slo_cohort_burst.py
git commit --only -m "feat(inference): publish exact cohort bursts atomically" -- \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/test_scheduler_slo_cohort_burst.py \
  tools/test_llm_engine_slo_cohort_burst.py
```

### Task 9: Add Closed Telemetry and Formal Gate

**Files:**

- Create: `tools/slo_cohort_burst_gate.py`
- Create: `tools/test_slo_cohort_burst_gate.py`
- Modify: `tinyvllm/engine/slo_cohort_burst.py`
- Modify: `tinyvllm/engine/exact_greedy_cohort_burst.py`
- Modify: `tinyvllm/engine/scheduler.py`
- Modify: `tinyvllm/engine/llm_engine.py`

**Interfaces:**

- Produces immutable decision, execution, and request JSONL rows.
- Produces `classify_slo_cohort_burst(bundle) -> str`.
- Consumes the frozen arrival traces and cost-table SHA.

- [ ] **Step 1: Write failing telemetry and boundary tests**

```python
def test_formal_gate_passes_exact_boundaries():
    summary = complete_summary(
        aggregate_throughput_improvement=0.10,
        medium_throughput_improvement=0.10,
        high_throughput_improvement=0.10,
        worst_throughput_regression=0.02,
        worst_p99_itl_regression=0.03,
        worst_p99_ttft_regression=0.05,
        worst_p99_e2e_regression=0.05,
        maximum_host_visible_gap_ns=40_000_000,
        starved_requests=0,
        post_eos_wasted_forward_fraction=0.10,
        peak_reserved_memory_regression=0.05,
    )
    assert classify_slo_cohort_burst(summary) == (
        "GO_SLO_AWARE_COHORT_DECODE_BURST"
    )


def test_tail_failure_precedes_throughput_failure():
    summary = complete_summary(
        worst_p99_itl_regression=0.031,
        aggregate_throughput_improvement=0.01,
    )
    assert classify_slo_cohort_burst(summary) == "NO_GO_TAIL_LATENCY"
```

- [ ] **Step 2: Run gate tests and confirm RED**

Run:

```bash
pytest -q tools/test_slo_cohort_burst_gate.py
```

Expected: gate module is missing.

- [ ] **Step 3: Implement bounded telemetry**

Decision rows contain all protected request ages/slacks, costs for K8/K4/K2,
structural eligibility, selected width, reason, ordered cohort, queue depths,
and cost-table SHA. Execution rows contain identities, replay/D2H counts,
actual duration, publication gap, generated/committed/EOS-discarded counts,
quarantine state, and terminal inventory. Request rows contain every
host-visible token timestamp and output identity.

- [ ] **Step 4: Implement formal failure precedence**

```python
FAILURE_PRECEDENCE = (
    "INVALID_SOURCE_OR_EVIDENCE",
    "NO_GO_CORRECTNESS",
    "NO_GO_LIFECYCLE",
    "NO_GO_STARVATION",
    "NO_GO_TAIL_LATENCY",
    "NO_GO_MEMORY",
    "NO_GO_EOS_WASTE",
    "NO_GO_THROUGHPUT",
    "GO_SLO_AWARE_COHORT_DECODE_BURST",
)
```

Reconstruct nearest-rank percentiles from raw request timestamps. Tokens
published together share a timestamp; do not replace ITL with amortized TPOT.
Output throughput uses committed tokens divided by the interval from first
frozen arrival to last completion.

- [ ] **Step 5: Run telemetry and gate tests**

Run:

```bash
pytest -q \
  tools/test_slo_cohort_burst.py \
  tools/test_exact_greedy_cohort_burst.py \
  tools/test_scheduler_slo_cohort_burst.py \
  tools/test_llm_engine_slo_cohort_burst.py \
  tools/test_slo_cohort_burst_gate.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit telemetry and classification**

```bash
git add -N -- \
  tools/slo_cohort_burst_gate.py \
  tools/test_slo_cohort_burst_gate.py
git commit --only -m "feat(inference): gate SLO-aware cohort bursts" -- \
  tinyvllm/engine/slo_cohort_burst.py \
  tinyvllm/engine/exact_greedy_cohort_burst.py \
  tinyvllm/engine/scheduler.py \
  tinyvllm/engine/llm_engine.py \
  tools/slo_cohort_burst_gate.py \
  tools/test_slo_cohort_burst_gate.py
```

### Task 10: Add Independent Verification and Run the Canonical Gate

**Files:**

- Create: `tools/slo_cohort_burst_verify.py`
- Create: `tools/test_slo_cohort_burst_verify.py`
- Modify: `tools/run_slo_cohort_burst_remote.py`
- Modify: `tools/test_run_slo_cohort_burst_remote.py`
- Modify: `AGENT_HANDOFF_STATE.md`
- Create after execution:
  `artifacts/slo_cohort_burst/<tag>/final_bundle/summary.json`
- Create after execution:
  `artifacts/slo_cohort_burst/<tag>/final_bundle/manifest.json`
- Create after execution:
  `artifacts/slo_cohort_burst/<tag>/final_bundle/remote_verify.json`
- Create after execution:
  `artifacts/slo_cohort_burst/<tag>/final_bundle/local_verify.json`
- Create after execution:
  `artifacts/slo_cohort_burst/<tag>/final_bundle/report.md`

**Interfaces:**

- Consumes only immutable raw rows, source manifest, frozen arrival traces,
  cost table, and artifact hashes.
- Produces an independent classification and a closed manifest.

- [ ] **Step 1: Write failing verifier mutation tests**

```python
@pytest.mark.parametrize(
    "mutation",
    (
        "decision_width",
        "protected_request_slack",
        "lease_identity",
        "row_order",
        "eos_prefix",
        "request_timestamp",
        "reserved_memory",
        "source_sha",
        "artifact_hash",
    ),
)
def test_verifier_rejects_authoritative_mutation(mutation):
    bundle = complete_synthetic_bundle()
    mutate(bundle, mutation)
    with pytest.raises(ValueError):
        verify_slo_cohort_burst_bundle(bundle)
```

- [ ] **Step 2: Run verifier tests and confirm RED**

Run:

```bash
pytest -q tools/test_slo_cohort_burst_verify.py
```

Expected: verifier module is missing.

- [ ] **Step 3: Implement independent reconstruction**

The verifier must not import the gate's classification function. It
independently reconstructs:

- source and environment identity;
- arrival-trace equality;
- cost-table identity and every width decision;
- every lease/result/graph identity;
- per-row EOS prefixes and exact output equality;
- replay, D2H, publication, and pending-lease inventories;
- TTFT, ITL, E2E, throughput, starvation, EOS waste, and memory;
- fixed failure precedence and final classification;
- SHA-256 for every authoritative artifact.

- [ ] **Step 4: Run the complete local contract suite**

Run:

```bash
pytest -q \
  tools/test_slo_cohort_burst_ceiling.py \
  tools/test_run_slo_cohort_burst_remote.py \
  tools/test_slo_cohort_burst.py \
  tools/test_exact_greedy_cohort_burst.py \
  tools/test_scheduler_slo_cohort_burst.py \
  tools/test_llm_engine_slo_cohort_burst.py \
  tools/test_slo_cohort_burst_gate.py \
  tools/test_slo_cohort_burst_verify.py \
  tools/test_exact_greedy_decode_burst.py \
  tools/test_llm_engine_exact_greedy_decode_burst.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit verifier and runner**

```bash
git add -N -- \
  tools/slo_cohort_burst_verify.py \
  tools/test_slo_cohort_burst_verify.py
git commit --only -m "feat(inference): independently verify cohort burst" -- \
  tools/slo_cohort_burst_verify.py \
  tools/test_slo_cohort_burst_verify.py \
  tools/run_slo_cohort_burst_remote.py \
  tools/test_run_slo_cohort_burst_remote.py
```

- [ ] **Step 6: Run Stage-1 correctness and lifecycle qualification**

Run:

```bash
python tools/run_slo_cohort_burst_remote.py \
  --stage correctness \
  --tag 20260913-slo-cohort-correctness-r1
```

Expected: exact tokens/text/logits/argmax, zero duplicate commits, zero
unauthorized KV publication, and zero pending leases for all `B x K` cases.
Any correctness or lifecycle failure stops before performance.

- [ ] **Step 7: Run frozen Stage-2 open-loop qualification**

Run:

```bash
python tools/run_slo_cohort_burst_remote.py \
  --stage canonical \
  --tag 20260913-slo-cohort-canonical-r1
```

The runner freezes low/medium/high arrival traces and the complete paired arm
order before launching either arm. It executes at least 128 measured requests
per workload and arm and at least five repetitions. The frozen workload
definitions are:

```text
decode-heavy steady:
  prompt tokens = 256
  maximum output tokens = 128

mixed short and long:
  70% = 256 prompt / 64 output
  20% = 2048 prompt / 128 output
  10% = 8192 prompt / 128 output

bursty EOS-sensitive:
  frozen arrival waves
  ignore_eos = false
  heterogeneous natural termination lengths
```

The load points are 40%, 70%, and 90% of separately calibrated baseline
saturation. Baseline and candidate use identical request IDs, prompts,
arrival timestamps, output budgets, and EOS settings. The frozen paired order
includes `baseline -> candidate -> candidate -> baseline`; closed-loop request
generation is prohibited.

- [ ] **Step 8: Run the local verifier on the compact bundle**

Run:

```bash
python tools/slo_cohort_burst_verify.py \
  artifacts/slo_cohort_burst/20260913-slo-cohort-canonical-r1/final_bundle
```

Expected: local and remote verifier classifications and all reconstructed
metrics agree exactly.

- [ ] **Step 9: Record the result without overstating it**

Append a dated section to `AGENT_HANDOFF_STATE.md` containing:

- source commit and dirty-state boundary;
- remote tag and approved storage root;
- Stage-0 ceiling result;
- correctness/lifecycle result;
- benefit and cost metrics for every workload;
- remote/local verifier result;
- terminal classification;
- explicit unsupported scopes.

Do not generalize a Qwen3-0.6B/TP1 result to TP2, Qwen3-8B/27B, sampling,
speculative decoding, joint chunked prefill, or production default.

- [ ] **Step 10: Commit only the compact result bundle and handoff**

```bash
git add -N -- \
  artifacts/slo_cohort_burst/20260913-slo-cohort-canonical-r1/final_bundle \
  AGENT_HANDOFF_STATE.md
git commit --only -m "docs(inference): record cohort burst qualification" -- \
  artifacts/slo_cohort_burst/20260913-slo-cohort-canonical-r1/final_bundle \
  AGENT_HANDOFF_STATE.md
```

- [ ] **Step 11: Push and verify the exact remote SHA**

```bash
branch=$(git branch --show-current)
git push origin "HEAD:$branch"
test "$(git rev-parse HEAD)" = "$(
  git ls-remote --heads origin "refs/heads/$branch" | awk '{print $1}'
)"
```

Expected: the local and remote branch SHAs are identical.

---

## Final Review Checklist

- [ ] Stage 0 is source-bound and ran before runtime implementation.
- [ ] A `NO_GO_CEILING` result stopped Tasks 3-10.
- [ ] Disabled mode preserves the previous runtime path.
- [ ] Scheduler cohort membership and order are unchanged.
- [ ] Every protected request contributes to global slack.
- [ ] Cost-table identity is frozen and verified.
- [ ] K1/K2/K4/K8 selection is deterministic.
- [ ] Every row has explicit KV write authority.
- [ ] EOS commits only an exact prefix and reports wasted work.
- [ ] Pre-replay fallback and post-replay terminal failure are disjoint.
- [ ] Every terminal path closes the pending lease inventory.
- [ ] Raw request timestamps reconstruct TTFT, ITL, and E2E.
- [ ] Throughput, tail latency, starvation, memory, and EOS waste are reported.
- [ ] Both verifiers independently reproduce the classification.
- [ ] Only compact evidence is stored locally.
- [ ] Every commit contains only named task paths.
