# Lease-Sealed State-Commit / AllReduce Overlap Stage-0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and independently qualify a model-neutral four-GPU primitive
that overlaps an NCCL AllReduce with an invisible shadow-state copy, and stop
before Qwen3.8 integration unless the frozen Stage-0 gate returns
`GO_LEASE_SEALED_OVERLAP_MICROGATE`.

**Architecture:** Add a small runtime primitive with explicit stream, event,
ticket, join, seal, publish, and abort ownership. Exercise it through a
real-shape TP4 worker using active-token counts 1, 4, and 8, then assemble and
independently verify a compact immutable evidence bundle. Qwen layer,
`RowParallelLinear`, and model transaction integration are intentionally
excluded from this plan.

**Tech Stack:** Python 3.12, PyTorch distributed/NCCL, CUDA streams and events,
pytest, JSON/JSONL, SHA-256 manifests, SSH, four NVIDIA A100 GPUs.

## Global Constraints

- Work only in `/Users/bytedance/Desktop/TinyLLMForge`, which resolves to the
  authoritative checkout.
- Stay on `feat/kv-sparse-attention` and push only to
  `origin/feat/kv-sparse-attention`.
- Do not create a worktree or use subagents; execute this plan inline.
- Use strict RED, minimal implementation, and GREEN evidence for every code
  task.
- Stage exact paths only. Never use `git add -A`, `git reset`, `git clean`, or
  broad formatting.
- Commit with `git -c core.hooksPath=/dev/null commit` and exactly one
  `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.
- Do not modify `tinyvllm/layers/linear.py`, any Qwen layer, or any Qwen model
  transaction in Stage 0.
- The generic runtime contract may contain only `local_result`,
  `side_effect_payload`, readiness events, and an opaque `commit_identity`.
- Keep the feature default-disabled and unreachable from production model
  execution in Stage 0.
- Preserve NCCL, BF16 output, FP32 accumulation where already used, and exact
  tensor bytes. Do not add approximate math.
- Do not use `.item()`, host event polling, a Python busy loop, or
  `torch.cuda.synchronize()` in the timed candidate path.
- Preallocate streams, events, output tensors, and shadow tensors before
  warmup. No timed-path allocation is allowed.
- Require exactly four strict-clean GPUs immediately before launch: each at or
  below `1,024 MiB`, at or below `5%` utilization, and with no compute
  process.
- Do not run `kinit` or `krenew`.
- Do not kill, pause, adopt, or clean foreign GPU processes. Cleanup is
  restricted to exact-tag-owned descendants.
- Put every remote task file, cache, log, artifact, and temporary file below
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/`.
- Keep model weights, caches, large traces, and scratch tensors off the Mac.
  Download only the compact final bundle and verifier receipts.
- Every attempt uses a fresh immutable tag and source revision. Never mutate
  or reclassify an old attempt.
- Report benefit and cost together. Stage 0 proves only mechanism viability,
  never Qwen3.8 end-to-end performance.
- Stage 1 is forbidden unless producer plus remote and local independent
  verifiers agree on `GO_LEASE_SEALED_OVERLAP_MICROGATE`.

## File map

| File | Responsibility |
|---|---|
| `tinyvllm/engine/collective_side_effect_overlap.py` | Generic stream/event lifecycle and opaque transaction ticket |
| `tools/lease_sealed_state_commit_overlap.py` | Frozen Stage-0 constants, interval math, row validation, classifier |
| `tools/lease_sealed_state_commit_overlap_worker.py` | Four-rank CUDA worker, preallocated buffers, paired measurements |
| `tools/assemble_lease_sealed_state_commit_overlap.py` | Strict producer bundle assembly and manifest generation |
| `tools/verify_lease_sealed_state_commit_overlap.py` | Independent reconstruction without importing the assembler |
| `tools/run_lease_sealed_state_commit_overlap.py` | Safe remote planning, admission, source staging, supervision, dual verification |
| `tools/test_collective_side_effect_overlap.py` | Generic lifecycle and ownership RED/GREEN tests |
| `tools/test_lease_sealed_state_commit_overlap.py` | Pure classifier and interval-math tests |
| `tools/test_lease_sealed_state_commit_overlap_worker.py` | Schedule, schema, preallocation, and timed-path source tests |
| `tools/test_assemble_lease_sealed_state_commit_overlap.py` | Bundle, identity, finite-value, and manifest tests |
| `tools/test_verify_lease_sealed_state_commit_overlap.py` | Independent-verifier mutation and disagreement tests |
| `tools/test_run_lease_sealed_state_commit_overlap.py` | Remote-path, admission, auth, launch, and cleanup tests |
| `docs/superpowers/audits/2026-09-07-lease-sealed-state-commit-overlap-stage0-audit.md` | Terminal Stage-0 result and claim boundary |
| `AGENT_HANDOFF_STATE.md` | Append-only final checkpoint and authorized next action |

---

### Task 1: Freeze the Stage-0 evidence contract and classifier

**Files:**

- Create: `tools/lease_sealed_state_commit_overlap.py`
- Create: `tools/test_lease_sealed_state_commit_overlap.py`

**Interfaces:**

- Produces:
  `interval_intersection_ns(left: tuple[int, int], right: tuple[int, int]) -> int`.
- Produces:
  `validate_measurement_row(row: dict) -> dict`.
- Produces:
  `classify_stage0(rows: list[dict], memory: dict, cleanup: dict) -> dict`.
- The worker, assembler, verifier, and controller import the frozen constants
  from this module.

- [ ] **Step 1: Write the failing classifier tests**

Create `tools/test_lease_sealed_state_commit_overlap.py` with fixtures using
exactly 180 rows: 3 active-token shapes, 15 pairs, and 4 ranks.

```python
from __future__ import annotations

import copy

import pytest

from tools.lease_sealed_state_commit_overlap import (
    ACTIVE_TOKEN_GROUPS,
    MEASURED_PAIR_COUNT,
    WORLD_SIZE,
    classify_stage0,
    interval_intersection_ns,
    validate_measurement_row,
)


def passing_rows():
    return [
        {
            "attempt": "20260907-lease-sealed-stage0-r1",
            "source_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "active_tokens": active_tokens,
            "pair_index": pair_index,
            "rank": rank,
            "arm_order": (
                ["baseline", "candidate"]
                if pair_index % 2 == 0
                else ["candidate", "baseline"]
            ),
            "baseline_critical_ns": 100_000,
            "candidate_critical_ns": 90_000,
            "baseline_host_submission_ns": 20_000,
            "candidate_host_submission_ns": 20_400,
            "allreduce_interval_ns": [10_000, 60_000],
            "state_copy_interval_ns": [35_000, 75_000],
            "overlap_intersection_ns": 25_000,
            "reduced_output_exact": True,
            "final_output_exact": True,
            "shadow_payload_exact": True,
            "active_state_preserved_before_publish": True,
            "published_state_exact": True,
            "abort_preserved_old_state": True,
            "commit_identity_match": True,
            "finite_output": True,
            "timed_path_allocation_count": 0,
            "timed_out": False,
        }
        for active_tokens in ACTIVE_TOKEN_GROUPS
        for pair_index in range(MEASURED_PAIR_COUNT)
        for rank in range(WORLD_SIZE)
    ]


def passing_memory():
    return {
        "rank_rows": [
            {
                "rank": rank,
                "maximum_allocated_delta_bytes": 100_000_000,
                "maximum_reserved_delta_bytes": 120_000_000,
                "maximum_theoretical_shadow_bytes": 104_202_240,
            }
            for rank in range(WORLD_SIZE)
        ]
    }


def test_frozen_inventory_and_interval_math():
    assert ACTIVE_TOKEN_GROUPS == (1, 4, 8)
    assert MEASURED_PAIR_COUNT == 15
    assert WORLD_SIZE == 4
    assert interval_intersection_ns((10, 60), (35, 75)) == 25


def test_classifier_accepts_complete_profitable_evidence():
    result = classify_stage0(
        passing_rows(),
        passing_memory(),
        {"classification": "CLEAN"},
    )

    assert result["classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert result["stage1_authorized"] is True
    assert result["measurement_row_count"] == 180


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("coverage", "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"),
        ("correctness", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("final_output", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("identity", "NO_GO_CORRECTNESS_OR_LIFECYCLE"),
        ("allocation", "NO_GO_MEMORY_OR_ALLOCATION"),
        ("memory", "NO_GO_MEMORY_OR_ALLOCATION"),
        ("overlap", "NO_GO_INSUFFICIENT_OVERLAP"),
        ("median", "NO_GO_PERFORMANCE"),
        ("tail", "NO_GO_PERFORMANCE"),
        ("host", "NO_GO_PERFORMANCE"),
        ("direction", "NO_GO_PERFORMANCE"),
        ("cleanup", "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT"),
    ),
)
def test_classifier_fails_closed(mutation, expected):
    rows = passing_rows()
    memory = passing_memory()
    cleanup = {"classification": "CLEAN"}
    if mutation == "coverage":
        rows.pop()
    elif mutation == "correctness":
        rows[0]["shadow_payload_exact"] = False
    elif mutation == "final_output":
        rows[0]["final_output_exact"] = False
    elif mutation == "identity":
        rows[0]["commit_identity_match"] = False
    elif mutation == "allocation":
        rows[0]["timed_path_allocation_count"] = 1
    elif mutation == "memory":
        memory["rank_rows"][0]["maximum_reserved_delta_bytes"] = (
            memory["rank_rows"][0]["maximum_theoretical_shadow_bytes"]
            + 64 * 1024 * 1024
            + 1
        )
    elif mutation == "overlap":
        for row in rows:
            if row["active_tokens"] == 4:
                row["state_copy_interval_ns"] = [58_000, 75_000]
                row["overlap_intersection_ns"] = 2_000
    elif mutation == "median":
        for row in rows:
            if row["active_tokens"] in (4, 8):
                row["candidate_critical_ns"] = 98_000
    elif mutation == "tail":
        for row in rows:
            if row["pair_index"] == 14:
                row["candidate_critical_ns"] = 110_000
    elif mutation == "host":
        for row in rows:
            row["candidate_host_submission_ns"] = 21_000
    elif mutation == "direction":
        for row in rows:
            if row["active_tokens"] == 8 and row["pair_index"] >= 10:
                row["candidate_critical_ns"] = 101_000
    elif mutation == "cleanup":
        cleanup["classification"] = "DIRTY"

    assert classify_stage0(rows, memory, cleanup)["classification"] == expected


def test_measurement_row_rejects_nonfinite_duplicate_or_wrong_order():
    row = passing_rows()[0]
    assert validate_measurement_row(row) == row

    broken = copy.deepcopy(row)
    broken["candidate_critical_ns"] = float("nan")
    with pytest.raises(ValueError, match="candidate_critical_ns"):
        validate_measurement_row(broken)

    broken = copy.deepcopy(row)
    broken["arm_order"] = ["candidate", "baseline"]
    with pytest.raises(ValueError, match="arm_order"):
        validate_measurement_row(broken)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_lease_sealed_state_commit_overlap.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'tools.lease_sealed_state_commit_overlap'`.

- [ ] **Step 3: Implement the frozen contract**

Create `tools/lease_sealed_state_commit_overlap.py` with these public values
and exact classifier precedence:

```python
from __future__ import annotations

import math
from statistics import median


WORLD_SIZE = 4
ACTIVE_TOKEN_GROUPS = (1, 4, 8)
WARMUP_PAIR_COUNT = 2
MEASURED_PAIR_COUNT = 15
HIDDEN_SIZE = 5120
STATE_BYTES_PER_TOKEN_PER_LAYER = 271_360
LINEAR_LAYER_COUNT = 48
MAX_RESERVED_SLACK_BYTES = 64 * 1024 * 1024
MIN_OVERLAP_RATIO = 0.20
MIN_AGGREGATE_SPEEDUP = 0.05
MAX_SINGLE_TOKEN_MEDIAN_REGRESSION = 0.01
MAX_P99_REGRESSION = 0.03
MAX_HOST_SUBMISSION_REGRESSION = 0.03
MIN_DIRECTIONAL_PAIR_COUNT = 11


def interval_intersection_ns(left, right):
    if (
        not isinstance(left, (tuple, list))
        or not isinstance(right, (tuple, list))
        or len(left) != 2
        or len(right) != 2
    ):
        raise ValueError("interval must contain two endpoints")
    left_start, left_end = left
    right_start, right_end = right
    values = (left_start, left_end, right_start, right_end)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError("interval endpoints are invalid")
    if left_end < left_start or right_end < right_start:
        raise ValueError("interval endpoints are invalid")
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def _nearest_rank_percentile(values, percentile):
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1),
    )
    return ordered[index]


def _finite_nonnegative(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def validate_measurement_row(row):
    if not isinstance(row, dict):
        raise ValueError("measurement row must be an object")
    numeric = (
        "baseline_critical_ns",
        "candidate_critical_ns",
        "baseline_host_submission_ns",
        "candidate_host_submission_ns",
        "overlap_intersection_ns",
    )
    for name in numeric:
        if not _finite_nonnegative(row.get(name)):
            raise ValueError(f"{name} is invalid")
    for name in ("allreduce_interval_ns", "state_copy_interval_ns"):
        value = row.get(name)
        interval_intersection_ns(value, value)
    if row["overlap_intersection_ns"] != interval_intersection_ns(
        row["allreduce_interval_ns"],
        row["state_copy_interval_ns"],
    ):
        raise ValueError("overlap_intersection_ns is invalid")
    active_tokens = row.get("active_tokens")
    pair_index = row.get("pair_index")
    rank = row.get("rank")
    if active_tokens not in ACTIVE_TOKEN_GROUPS:
        raise ValueError("active_tokens is invalid")
    if type(pair_index) is not int or pair_index not in range(
        MEASURED_PAIR_COUNT
    ):
        raise ValueError("pair_index is invalid")
    if type(rank) is not int or rank not in range(WORLD_SIZE):
        raise ValueError("rank is invalid")
    expected_order = (
        ["baseline", "candidate"]
        if pair_index % 2 == 0
        else ["candidate", "baseline"]
    )
    if row.get("arm_order") != expected_order:
        raise ValueError("arm_order is invalid")
    booleans = (
        "reduced_output_exact",
        "final_output_exact",
        "shadow_payload_exact",
        "active_state_preserved_before_publish",
        "published_state_exact",
        "abort_preserved_old_state",
        "commit_identity_match",
        "finite_output",
        "timed_out",
    )
    if any(type(row.get(name)) is not bool for name in booleans):
        raise ValueError("correctness or lifecycle flag is invalid")
    if (
        type(row.get("timed_path_allocation_count")) is not int
        or row["timed_path_allocation_count"] < 0
    ):
        raise ValueError("timed_path_allocation_count is invalid")
    return dict(row)
```

Complete `classify_stage0` with this exact decision order:

```python
def _result(classification, summaries, row_count):
    return {
        "classification": classification,
        "stage1_authorized": (
            classification == "GO_LEASE_SEALED_OVERLAP_MICROGATE"
        ),
        "measurement_row_count": row_count,
        "shape_summaries": summaries,
    }


def classify_stage0(rows, memory, cleanup):
    expected = {
        (shape, pair, rank)
        for shape in ACTIVE_TOKEN_GROUPS
        for pair in range(MEASURED_PAIR_COUNT)
        for rank in range(WORLD_SIZE)
    }
    validated = []
    seen = set()
    incomplete = False
    correctness_failed = False
    allocation_failed = False
    for raw in rows if isinstance(rows, (list, tuple)) else ():
        try:
            row = validate_measurement_row(raw)
        except ValueError:
            incomplete = True
            continue
        identity = (
            row["active_tokens"],
            row["pair_index"],
            row["rank"],
        )
        if identity in seen:
            incomplete = True
            continue
        seen.add(identity)
        validated.append(row)
        correctness_failed |= not all(
            row[name]
            for name in (
                "reduced_output_exact",
                "final_output_exact",
                "shadow_payload_exact",
                "active_state_preserved_before_publish",
                "published_state_exact",
                "abort_preserved_old_state",
                "commit_identity_match",
                "finite_output",
            )
        )
        correctness_failed |= row["timed_out"]
        allocation_failed |= row["timed_path_allocation_count"] != 0
    incomplete |= seen != expected
    if correctness_failed:
        return _result(
            "NO_GO_CORRECTNESS_OR_LIFECYCLE",
            [],
            len(validated),
        )

    memory_rows = memory.get("rank_rows") if isinstance(memory, dict) else None
    if not isinstance(memory_rows, list) or len(memory_rows) != WORLD_SIZE:
        incomplete = True
    else:
        for row in memory_rows:
            required = (
                row.get("maximum_reserved_delta_bytes"),
                row.get("maximum_theoretical_shadow_bytes"),
            )
            if any(not _finite_nonnegative(value) for value in required):
                incomplete = True
                continue
            allocation_failed |= (
                required[0] > required[1] + MAX_RESERVED_SLACK_BYTES
            )
    if allocation_failed:
        return _result(
            "NO_GO_MEMORY_OR_ALLOCATION",
            [],
            len(validated),
        )
    if (
        not isinstance(cleanup, dict)
        or cleanup.get("classification") != "CLEAN"
    ):
        incomplete = True
    if incomplete:
        return _result(
            "INCONCLUSIVE_ENVIRONMENT_OR_MEASUREMENT",
            [],
            len(validated),
        )

    summaries = []
    for shape in ACTIVE_TOKEN_GROUPS:
        pair_rows = []
        for pair in range(MEASURED_PAIR_COUNT):
            ranks = [
                row
                for row in validated
                if row["active_tokens"] == shape
                and row["pair_index"] == pair
            ]
            pair_rows.append({
                "baseline": max(row["baseline_critical_ns"] for row in ranks),
                "candidate": max(row["candidate_critical_ns"] for row in ranks),
                "baseline_host": max(
                    row["baseline_host_submission_ns"] for row in ranks
                ),
                "candidate_host": max(
                    row["candidate_host_submission_ns"] for row in ranks
                ),
                "overlap_ratio": min(
                    row["overlap_intersection_ns"]
                    / max(
                        1,
                        min(
                            row["allreduce_interval_ns"][1]
                            - row["allreduce_interval_ns"][0],
                            row["state_copy_interval_ns"][1]
                            - row["state_copy_interval_ns"][0],
                        ),
                    )
                    for row in ranks
                ),
            })
        baseline = [row["baseline"] for row in pair_rows]
        candidate = [row["candidate"] for row in pair_rows]
        baseline_median = median(baseline)
        candidate_median = median(candidate)
        baseline_p99 = _nearest_rank_percentile(baseline, 0.99)
        candidate_p99 = _nearest_rank_percentile(candidate, 0.99)
        baseline_host = median(row["baseline_host"] for row in pair_rows)
        candidate_host = median(row["candidate_host"] for row in pair_rows)
        summaries.append({
            "active_tokens": shape,
            "median_speedup_ratio": 1 - candidate_median / baseline_median,
            "p99_regression_ratio": candidate_p99 / baseline_p99 - 1,
            "host_submission_regression_ratio": (
                candidate_host / baseline_host - 1
            ),
            "median_realized_overlap_ratio": median(
                row["overlap_ratio"] for row in pair_rows
            ),
            "improving_pair_count": sum(
                row["candidate"] < row["baseline"] for row in pair_rows
            ),
        })

    by_shape = {row["active_tokens"]: row for row in summaries}
    if any(
        by_shape[shape]["median_realized_overlap_ratio"]
        < MIN_OVERLAP_RATIO
        for shape in (4, 8)
    ):
        return _result(
            "NO_GO_INSUFFICIENT_OVERLAP",
            summaries,
            len(validated),
        )
    aggregate_speedup = 1 - math.sqrt(
        (1 - by_shape[4]["median_speedup_ratio"])
        * (1 - by_shape[8]["median_speedup_ratio"])
    )
    performance_failed = (
        aggregate_speedup < MIN_AGGREGATE_SPEEDUP
        or any(
            by_shape[shape]["median_speedup_ratio"] < 0
            for shape in (4, 8)
        )
        or by_shape[1]["median_speedup_ratio"]
        < -MAX_SINGLE_TOKEN_MEDIAN_REGRESSION
        or any(
            row["p99_regression_ratio"] > MAX_P99_REGRESSION
            or row["host_submission_regression_ratio"]
            > MAX_HOST_SUBMISSION_REGRESSION
            for row in summaries
        )
        or any(
            by_shape[shape]["improving_pair_count"]
            < MIN_DIRECTIONAL_PAIR_COUNT
            for shape in (4, 8)
        )
    )
    return _result(
        "NO_GO_PERFORMANCE"
        if performance_failed
        else "GO_LEASE_SEALED_OVERLAP_MICROGATE",
        summaries,
        len(validated),
    )
```

- [ ] **Step 4: Run GREEN and static checks**

Run:

```bash
python3 -m pytest \
  tools/test_lease_sealed_state_commit_overlap.py -q
python3 -m py_compile \
  tools/lease_sealed_state_commit_overlap.py
git diff --check -- \
  tools/lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py
```

Expected: all tests pass, compilation exits zero, and diff check is empty.

- [ ] **Step 5: Commit and push**

```bash
git add -- \
  tools/lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py
git -c core.hooksPath=/dev/null commit \
  -m "test(tp4): freeze state commit overlap gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 2: Add the generic overlap lifecycle primitive

**Files:**

- Create: `tinyvllm/engine/collective_side_effect_overlap.py`
- Create: `tools/test_collective_side_effect_overlap.py`

**Interfaces:**

- Produces `OverlapResources`.
- Produces `LeaseSealedOverlapTicket`.
- Produces
  `LeaseSealedCollectiveSideEffect.launch(...) -> LeaseSealedOverlapTicket`.
- Produces `join`, `seal`, `publish`, and `abort` lifecycle methods.
- Consumes an injected collective function and side-effect materializer so
  CPU fake tests do not require CUDA.

- [ ] **Step 1: Write failing lifecycle tests**

Create `tools/test_collective_side_effect_overlap.py` with deterministic fake
streams, events, work handles, and callbacks:

```python
from __future__ import annotations

from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package_name in ("tinyvllm", "tinyvllm.engine"):
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / package_name.replace(".", "/"))]
    sys.modules.setdefault(package_name, package)


from tinyvllm.engine.collective_side_effect_overlap import (
    LeaseSealedCollectiveSideEffect,
    OverlapResources,
)


class FakeStream:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait_event(self, event):
        self.events.append((self.name, "wait", event.name))


class FakeEvent:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def record(self, stream):
        self.events.append((stream.name, "record", self.name))

    def synchronize(self):
        self.events.append(("host", "synchronize", self.name))


class FakeWork:
    def __init__(self, events):
        self.events = events

    def wait(self):
        self.events.append(("host", "wait", "collective"))


class FakeContext:
    def __init__(self, stream):
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, exc_type, exc, traceback):
        return False


def resources(events):
    return OverlapResources(
        communication_stream=FakeStream("communication", events),
        side_effect_stream=FakeStream("side_effect", events),
        producer_ready_event=FakeEvent("producer_ready", events),
        consumer_ready_event=FakeEvent("consumer_ready", events),
        side_effect_ready_event=FakeEvent("side_effect_ready", events),
    )


def executor(events):
    current = FakeStream("current", events)
    return LeaseSealedCollectiveSideEffect(
        resources=resources(events),
        current_stream=lambda _tensor: current,
        stream_context=lambda stream: FakeContext(stream),
        collective=lambda tensor: events.append(
            ("communication", "collective", tensor)
        ) or FakeWork(events),
    )


def test_launch_forks_collective_and_side_effect_then_join_waits_both():
    events = []
    runtime = executor(events)
    shadow = {}

    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda payload: shadow.update(value=payload),
        commit_identity="identity-a",
    )
    result = runtime.join(ticket)

    assert result == "local"
    assert shadow == {"value": "candidate"}
    assert ticket.state == "joined"
    assert ("current", "wait", "consumer_ready") in events
    assert ("current", "wait", "side_effect_ready") in events


def test_publish_requires_join_seal_and_matching_identity():
    events = []
    runtime = executor(events)
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )

    with pytest.raises(RuntimeError, match="joined"):
        runtime.seal(ticket, "identity-a")
    runtime.join(ticket)
    with pytest.raises(RuntimeError, match="identity"):
        runtime.seal(ticket, "identity-b")
    runtime.seal(ticket, "identity-a")
    published = []
    runtime.publish(
        ticket,
        "identity-a",
        lambda: published.append("published"),
    )
    assert published == ["published"]
    assert ticket.state == "published"


def test_abort_is_terminal_and_never_publishes():
    events = []
    runtime = executor(events)
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )
    aborted = []
    runtime.abort(
        ticket,
        lambda: (
            events.append(("host", "abort", "callback")),
            aborted.append("aborted"),
        ),
    )

    assert aborted == ["aborted"]
    assert ticket.state == "aborted"
    assert events[-3:] == [
        ("host", "wait", "collective"),
        ("host", "synchronize", "side_effect_ready"),
        ("host", "abort", "callback"),
    ]
    with pytest.raises(RuntimeError, match="aborted"):
        runtime.publish(ticket, "identity-a", lambda: None)


def test_launch_rejects_empty_identity_and_reuse_of_active_resources():
    events = []
    runtime = executor(events)
    with pytest.raises(ValueError, match="commit_identity"):
        runtime.launch(
            local_result="local",
            side_effect_payload="candidate",
            materialize_side_effect=lambda _payload: None,
            commit_identity="",
        )
    runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )
    with pytest.raises(RuntimeError, match="active"):
        runtime.launch(
            local_result="local-2",
            side_effect_payload="candidate-2",
            materialize_side_effect=lambda _payload: None,
            commit_identity="identity-b",
        )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python3 -m pytest \
  tools/test_collective_side_effect_overlap.py -q
```

Expected: collection fails because
`tinyvllm.engine.collective_side_effect_overlap` does not exist.

- [ ] **Step 3: Implement the minimal generic lifecycle**

Create `tinyvllm/engine/collective_side_effect_overlap.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal


TicketState = Literal[
    "launched",
    "joined",
    "sealed",
    "published",
    "aborted",
]


@dataclass(frozen=True)
class OverlapResources:
    communication_stream: object
    side_effect_stream: object
    producer_ready_event: object
    consumer_ready_event: object
    side_effect_ready_event: object


@dataclass
class LeaseSealedOverlapTicket:
    commit_identity: str
    local_result: object
    collective_work: object
    consumer_ready_event: object
    side_effect_ready_event: object
    state: TicketState = "launched"


class LeaseSealedCollectiveSideEffect:
    def __init__(
        self,
        *,
        resources: OverlapResources,
        current_stream: Callable[[object], object],
        stream_context: Callable[[object], object],
        collective: Callable[[object], object],
    ):
        self.resources = resources
        self.current_stream = current_stream
        self.stream_context = stream_context
        self.collective = collective
        self._active_ticket = None

    def launch(
        self,
        *,
        local_result,
        side_effect_payload,
        materialize_side_effect,
        commit_identity: str,
    ) -> LeaseSealedOverlapTicket:
        if not isinstance(commit_identity, str) or not commit_identity:
            raise ValueError("commit_identity must be a non-empty string")
        if self._active_ticket is not None:
            raise RuntimeError("overlap resources already have an active ticket")
        current = self.current_stream(local_result)
        resource = self.resources
        resource.producer_ready_event.record(current)
        with self.stream_context(resource.communication_stream):
            resource.communication_stream.wait_event(
                resource.producer_ready_event
            )
            work = self.collective(local_result)
            resource.consumer_ready_event.record(
                resource.communication_stream
            )
        with self.stream_context(resource.side_effect_stream):
            resource.side_effect_stream.wait_event(
                resource.producer_ready_event
            )
            materialize_side_effect(side_effect_payload)
            resource.side_effect_ready_event.record(
                resource.side_effect_stream
            )
        ticket = LeaseSealedOverlapTicket(
            commit_identity=commit_identity,
            local_result=local_result,
            collective_work=work,
            consumer_ready_event=resource.consumer_ready_event,
            side_effect_ready_event=resource.side_effect_ready_event,
        )
        self._active_ticket = ticket
        return ticket

    def join(self, ticket: LeaseSealedOverlapTicket):
        self._require_active(ticket, "launched")
        current = self.current_stream(ticket.local_result)
        current.wait_event(ticket.consumer_ready_event)
        current.wait_event(ticket.side_effect_ready_event)
        ticket.state = "joined"
        return ticket.local_result

    def seal(self, ticket, observed_identity: str) -> None:
        self._require_active(ticket, "joined")
        if observed_identity != ticket.commit_identity:
            raise RuntimeError("overlap commit identity mismatch")
        ticket.state = "sealed"

    def publish(self, ticket, observed_identity: str, publisher) -> None:
        self._require_active(ticket, "sealed")
        if observed_identity != ticket.commit_identity:
            raise RuntimeError("overlap commit identity mismatch")
        publisher()
        ticket.state = "published"
        self._active_ticket = None

    def abort(self, ticket, aborter) -> None:
        if self._active_ticket is not ticket:
            raise RuntimeError("overlap ticket is not active")
        if ticket.state in ("published", "aborted"):
            raise RuntimeError(f"overlap ticket is already {ticket.state}")
        ticket.collective_work.wait()
        ticket.side_effect_ready_event.synchronize()
        aborter()
        ticket.state = "aborted"
        self._active_ticket = None

    def _require_active(self, ticket, expected_state: str) -> None:
        if self._active_ticket is not ticket:
            raise RuntimeError("overlap ticket is not active")
        if ticket.state != expected_state:
            raise RuntimeError(
                f"overlap ticket must be {expected_state}, "
                f"received {ticket.state}"
            )
```

The CUDA adapter passed to `collective` must call
`dist.all_reduce(tensor, group=process_group, async_op=True)`. The generic
module must not import Qwen-specific code or create CUDA resources internally.

- [ ] **Step 4: Add failure-transition tests**

Extend the test file to cover:

```python
def test_publish_failure_leaves_ticket_sealed_for_explicit_abort():
    events = []
    runtime = executor(events)
    ticket = runtime.launch(
        local_result="local",
        side_effect_payload="candidate",
        materialize_side_effect=lambda _payload: None,
        commit_identity="identity-a",
    )
    runtime.join(ticket)
    runtime.seal(ticket, "identity-a")

    with pytest.raises(RuntimeError, match="publish failed"):
        runtime.publish(
            ticket,
            "identity-a",
            lambda: (_ for _ in ()).throw(RuntimeError("publish failed")),
        )
    assert ticket.state == "sealed"
    runtime.abort(ticket, lambda: None)
    assert ticket.state == "aborted"
```

The implementation must set `published` only after the callback returns.

- [ ] **Step 5: Run GREEN and static checks**

```bash
python3 -m pytest \
  tools/test_collective_side_effect_overlap.py -q
python3 -m py_compile \
  tinyvllm/engine/collective_side_effect_overlap.py
git diff --check -- \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/test_collective_side_effect_overlap.py
```

Expected: all tests pass and both static commands exit zero.

- [ ] **Step 6: Commit and push**

```bash
git add -- \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/test_collective_side_effect_overlap.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(runtime): add lease sealed overlap primitive" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 3: Build the four-rank Stage-0 worker

**Files:**

- Create: `tools/lease_sealed_state_commit_overlap_worker.py`
- Create: `tools/test_lease_sealed_state_commit_overlap_worker.py`

**Interfaces:**

- Produces `OverlapBuffers.create(torch, device, active_tokens)`.
- Produces `build_overlap_runtime(buffers, torch, dist)`.
- Produces `build_workload_schedule() -> tuple[dict, ...]`.
- Produces one validated row per rank, shape, and measured pair.
- Writes `measurement_rows.rank-<rank>.jsonl`,
  `memory.rank-<rank>.json`, `lifecycle.rank-<rank>.json`, and
  `cleanup.rank-<rank>.json`.

- [ ] **Step 1: Write failing schedule and preallocation tests**

Create `tools/test_lease_sealed_state_commit_overlap_worker.py`:

```python
from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import pytest

from tools.lease_sealed_state_commit_overlap_worker import (
    OverlapBuffers,
    _run_candidate,
    build_argument_parser,
    build_workload_schedule,
)


class FakeCuda:
    def __init__(self):
        self.streams = []
        self.events = []

    def Stream(self, *, device):
        value = ("stream", device, len(self.streams))
        self.streams.append(value)
        return value

    def Event(self, *, enable_timing):
        value = ("event", enable_timing, len(self.events))
        self.events.append(value)
        return value


class FakeTorch:
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self):
        self.cuda = FakeCuda()
        self.allocations = []

    def empty(self, shape, *, dtype, device):
        value = {"shape": shape, "dtype": dtype, "device": device}
        self.allocations.append(value)
        return value


def test_schedule_freezes_shapes_warmups_pairs_and_abba_order():
    schedule = build_workload_schedule()

    assert [row["active_tokens"] for row in schedule] == [1, 4, 8]
    assert all(len(row["warmups"]) == 2 for row in schedule)
    assert all(len(row["measurements"]) == 15 for row in schedule)
    assert schedule[0]["measurements"][0]["arm_order"] == (
        "baseline",
        "candidate",
    )
    assert schedule[0]["measurements"][1]["arm_order"] == (
        "candidate",
        "baseline",
    )


def test_buffers_preallocate_two_streams_five_events_and_real_shapes():
    torch = FakeTorch()
    buffers = OverlapBuffers.create(torch, "cuda:0", active_tokens=8)

    assert len(torch.cuda.streams) == 2
    assert len(torch.cuda.events) >= 5
    assert buffers.local_result["shape"] == (8, 5120)
    assert buffers.shadow_payload["shape"] == (8, 271360 // 2)
    assert [
        row["dtype"] for row in torch.allocations[:3]
    ] == ["float32", "float32", "float32"]
    assert [
        row["dtype"] for row in torch.allocations[3:]
    ] == ["bfloat16", "bfloat16", "bfloat16", "bfloat16", "bfloat16"]


def test_candidate_timed_path_has_no_sync_item_or_allocation():
    source = inspect.getsource(_run_candidate)

    for forbidden in (
        "torch.cuda.synchronize",
        ".item(",
        "torch.empty",
        "torch.zeros",
        "torch.cuda.Stream",
        "torch.cuda.Event",
        "LeaseSealedCollectiveSideEffect(",
    ):
        assert forbidden not in source


def test_cli_requires_attempt_source_rank_and_output_identity():
    parser = build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])

    script = Path(__file__).with_name(
        "lease_sealed_state_commit_overlap_worker.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python3 -m pytest \
  tools/test_lease_sealed_state_commit_overlap_worker.py -q
```

Expected: collection fails because the worker module does not exist.

- [ ] **Step 3: Implement immutable schedule and preallocated buffers**

The worker imports all dimensions and repetition counts from
`tools.lease_sealed_state_commit_overlap`.

```python
@dataclass
class OverlapBuffers:
    communication_stream: object
    side_effect_stream: object
    producer_ready_event: object
    consumer_ready_event: object
    side_effect_ready_event: object
    baseline_started: object
    baseline_completed: object
    candidate_started: object
    candidate_completed: object
    allreduce_started: object
    allreduce_completed: object
    state_copy_started: object
    state_copy_completed: object
    local_result: object
    baseline_result: object
    candidate_result: object
    baseline_output: object
    candidate_output: object
    side_effect_payload: object
    baseline_shadow: object
    candidate_shadow: object

    @classmethod
    def create(cls, torch, device, active_tokens):
        def event():
            return torch.cuda.Event(enable_timing=True)

        state_elements = STATE_BYTES_PER_TOKEN_PER_LAYER // 2
        return cls(
            communication_stream=torch.cuda.Stream(device=device),
            side_effect_stream=torch.cuda.Stream(device=device),
            producer_ready_event=event(),
            consumer_ready_event=event(),
            side_effect_ready_event=event(),
            baseline_started=event(),
            baseline_completed=event(),
            candidate_started=event(),
            candidate_completed=event(),
            allreduce_started=event(),
            allreduce_completed=event(),
            state_copy_started=event(),
            state_copy_completed=event(),
            local_result=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            baseline_result=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            candidate_result=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.float32,
                device=device,
            ),
            baseline_output=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            candidate_output=torch.empty(
                (active_tokens, HIDDEN_SIZE),
                dtype=torch.bfloat16,
                device=device,
            ),
            side_effect_payload=torch.empty(
                (active_tokens, state_elements),
                dtype=torch.bfloat16,
                device=device,
            ),
            baseline_shadow=torch.empty(
                (active_tokens, state_elements),
                dtype=torch.bfloat16,
                device=device,
            ),
            candidate_shadow=torch.empty(
                (active_tokens, state_elements),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
```

Use fixed seeds `2026090701`, `2026090704`, and `2026090708`. Build two
warmup pairs and fifteen measured pairs per shape with even pairs baseline
first and odd pairs candidate first.

- [ ] **Step 4: Implement baseline and candidate device paths**

The baseline:

```python
def _run_baseline(*, buffers, torch, dist):
    stream = torch.cuda.current_stream(buffers.local_result.device)
    submitted = time.perf_counter_ns()
    buffers.baseline_started.record(stream)
    buffers.baseline_result.copy_(buffers.local_result)
    dist.all_reduce(buffers.baseline_result)
    buffers.baseline_output.copy_(buffers.baseline_result)
    buffers.baseline_shadow.copy_(buffers.side_effect_payload)
    buffers.baseline_completed.record(stream)
    return {
        "reduced_result": buffers.baseline_result,
        "final_output": buffers.baseline_output,
        "shadow": buffers.baseline_shadow,
        "started": buffers.baseline_started,
        "completed": buffers.baseline_completed,
        "host_submission_ns": time.perf_counter_ns() - submitted,
    }
```

Construct one `LeaseSealedCollectiveSideEffect` per preallocated buffer set
before warmup. Its injected collective uses
`dist.all_reduce(tensor, async_op=True)` and records component intervals in
the owning streams:

```python
def build_overlap_runtime(*, buffers, torch, dist):
    return LeaseSealedCollectiveSideEffect(
        resources=OverlapResources(
            communication_stream=buffers.communication_stream,
            side_effect_stream=buffers.side_effect_stream,
            producer_ready_event=buffers.producer_ready_event,
            consumer_ready_event=buffers.consumer_ready_event,
            side_effect_ready_event=buffers.side_effect_ready_event,
        ),
        current_stream=lambda tensor: torch.cuda.current_stream(tensor.device),
        stream_context=torch.cuda.stream,
        collective=lambda tensor: _timed_allreduce(
            tensor,
            buffers,
            dist,
        ),
    )


def _run_candidate(*, buffers, runtime, torch, commit_identity):
    stream = torch.cuda.current_stream(buffers.local_result.device)
    submitted = time.perf_counter_ns()
    buffers.candidate_started.record(stream)
    buffers.candidate_result.copy_(buffers.local_result)
    ticket = runtime.launch(
        local_result=buffers.candidate_result,
        side_effect_payload=buffers.side_effect_payload,
        materialize_side_effect=lambda payload: _timed_state_copy(
            payload,
            buffers,
        ),
        commit_identity=commit_identity,
    )
    result = runtime.join(ticket)
    buffers.candidate_output.copy_(result)
    runtime.seal(ticket, commit_identity)
    runtime.publish(ticket, commit_identity, lambda: None)
    buffers.candidate_completed.record(stream)
    return {
        "reduced_result": result,
        "final_output": buffers.candidate_output,
        "shadow": buffers.candidate_shadow,
        "started": buffers.candidate_started,
        "completed": buffers.candidate_completed,
        "host_submission_ns": time.perf_counter_ns() - submitted,
    }
```

The worker creates `runtime = build_overlap_runtime(...)` once per shape
immediately after `OverlapBuffers.create(...)`, before any warmup or measured
pair. `_timed_allreduce` records `allreduce_started` and
`allreduce_completed` on the communication stream around `async_op=True`.
`_timed_state_copy` records
`state_copy_started` and `state_copy_completed` on the side-effect stream
around `candidate_shadow.copy_(payload)`.

Synchronize only after a complete warmup or measured pair, outside
`_run_baseline` and `_run_candidate`, before reading event elapsed times and
correctness data.

- [ ] **Step 5: Implement correctness, lifecycle, memory, and cleanup rows**

For every measured pair and rank:

- compare candidate FP32 reduced output byte-for-byte with baseline;
- compare candidate BF16 final output byte-for-byte with baseline;
- compare candidate shadow byte-for-byte with baseline shadow;
- verify active state remains unchanged before publish;
- run untimed lifecycle probes for publish and abort;
- all-gather output and commit-identity digests across ranks;
- store absolute event timestamps relative to a shared per-rank origin;
- compute `overlap_intersection_ns` using the pure interval helper;
- record peak allocated and reserved memory before and after each shape;
- record allocator counters before and after the timed loop;
- destroy the process group and release Python references in `finally`;
- rank zero atomically merges rank artifacts only after all four rank files
  exist and validate.

The worker must reject:

- world size other than four;
- rank outside `0..3`;
- non-fresh output directories;
- source or attempt identity drift;
- a shape or pair count different from the frozen schedule.

- [ ] **Step 6: Run GREEN and adjacent tests**

```bash
python3 -m pytest \
  tools/test_collective_side_effect_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap_worker.py -q
python3 -m py_compile \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/lease_sealed_state_commit_overlap.py \
  tools/lease_sealed_state_commit_overlap_worker.py
git diff --check -- \
  tinyvllm/engine/collective_side_effect_overlap.py \
  tools/lease_sealed_state_commit_overlap.py \
  tools/lease_sealed_state_commit_overlap_worker.py \
  tools/test_collective_side_effect_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap_worker.py
```

Expected: all focused tests pass and both static checks exit zero.

- [ ] **Step 7: Commit and push**

```bash
git add -- \
  tools/lease_sealed_state_commit_overlap_worker.py \
  tools/test_lease_sealed_state_commit_overlap_worker.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): add state commit overlap worker" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 4: Assemble and independently verify the evidence bundle

**Files:**

- Create: `tools/assemble_lease_sealed_state_commit_overlap.py`
- Create: `tools/verify_lease_sealed_state_commit_overlap.py`
- Create: `tools/test_assemble_lease_sealed_state_commit_overlap.py`
- Create: `tools/test_verify_lease_sealed_state_commit_overlap.py`

**Interfaces:**

- Produces
  `assemble_bundle(output_root: Path, source_identity: dict, rows: list[dict],
  memory: dict, lifecycle: dict, cleanup: dict) -> dict`.
- Produces `verify_bundle(root: Path) -> dict`.
- The verifier imports only the pure contract/classifier module, never the
  producer assembler.

- [ ] **Step 1: Write failing assembler tests**

Create `tools/test_assemble_lease_sealed_state_commit_overlap.py` with a
passing 180-row fixture from Task 1 and these assertions:

```python
def test_assembler_writes_complete_manifested_go_bundle(tmp_path):
    result = assemble_bundle(output_root=tmp_path, **passing_inputs())

    assert result["classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert {path.name for path in tmp_path.iterdir()} == {
        "source_manifest.json",
        "environment_manifest.json",
        "gpu_rank_manifest.json",
        "workload_manifest.json",
        "admission.json",
        "paired_rows.jsonl",
        "correctness_rows.jsonl",
        "lifecycle_rows.jsonl",
        "memory_rows.jsonl",
        "overlap_rows.jsonl",
        "cleanup.json",
        "producer_result.json",
        "report.md",
        "manifest.sha256",
    }


def test_assembler_rejects_nonempty_output_identity_drift_and_nan(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing").write_text("occupied")
    with pytest.raises(ValueError, match="must be empty"):
        assemble_bundle(output_root=occupied, **passing_inputs())

    inputs = passing_inputs()
    inputs["rows"][0]["attempt"] = "different-attempt"
    with pytest.raises(ValueError, match="identity"):
        assemble_bundle(output_root=tmp_path / "identity", **inputs)

    inputs = passing_inputs()
    inputs["memory"]["rank_rows"][0][
        "maximum_reserved_delta_bytes"
    ] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        assemble_bundle(output_root=tmp_path / "nan", **inputs)
```

- [ ] **Step 2: Write failing independent-verifier tests**

Create `tools/test_verify_lease_sealed_state_commit_overlap.py`:

```python
from __future__ import annotations

import hashlib
import inspect
import json

import pytest

from tools.assemble_lease_sealed_state_commit_overlap import assemble_bundle
from tools.test_assemble_lease_sealed_state_commit_overlap import (
    passing_inputs,
)
from tools.verify_lease_sealed_state_commit_overlap import verify_bundle
import tools.verify_lease_sealed_state_commit_overlap as verifier_module


def test_verifier_reconstructs_go_without_importing_assembler(tmp_path):
    assemble_bundle(output_root=tmp_path, **passing_inputs())
    result = verify_bundle(tmp_path)

    assert result["status"] == "PASS"
    assert result["producer_classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert result["reconstructed_classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    source = inspect.getsource(verifier_module)
    assert "assemble_lease_sealed_state_commit_overlap" not in source


def test_verifier_rejects_hash_mutation_extra_file_and_missing_row(tmp_path):
    assemble_bundle(output_root=tmp_path, **passing_inputs())
    (tmp_path / "memory_rows.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest artifact hash"):
        verify_bundle(tmp_path)

    clean = tmp_path / "extra"
    assemble_bundle(output_root=clean, **passing_inputs())
    (clean / "unexpected.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="artifact inventory"):
        verify_bundle(clean)

    missing = tmp_path / "missing"
    assemble_bundle(output_root=missing, **passing_inputs())
    rows_path = missing / "paired_rows.jsonl"
    rows = rows_path.read_text().splitlines()
    rows_path.write_text("\n".join(rows[:-1]) + "\n")
    rewrite_manifest(missing)
    with pytest.raises(ValueError, match="producer classification"):
        verify_bundle(missing)
```

`rewrite_manifest` in the test must recompute SHA-256 for all expected producer
files so the missing-row test reaches classifier reconstruction rather than
failing at the hash layer.

- [ ] **Step 3: Run both test files and verify RED**

```bash
python3 -m pytest \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py -q
```

Expected: collection fails because both modules are absent.

- [ ] **Step 4: Implement strict producer assembly**

Use atomic same-directory temporary files plus `os.fsync` and `Path.replace`.
Reject duplicate JSON keys, non-finite values, nonempty output directories,
identity drift, incomplete rank/pair coverage, and unexpected artifacts.

`producer_result.json` must contain:

```python
{
    "schema_version": (
        "lease-sealed-state-commit-overlap-producer-result.v1"
    ),
    "classification": classification["classification"],
    "stage1_authorized": classification["stage1_authorized"],
    "attempt": source_identity["attempt"],
    "source_revision": source_identity["source_revision"],
    "source_tree_sha256": source_identity["source_tree_sha256"],
    "measurement_row_count": classification["measurement_row_count"],
    "shape_summaries": classification["shape_summaries"],
}
```

`manifest.sha256` must enumerate every producer file except itself. `report.md`
must print benefit and cost side by side for all three shapes and state the
claim boundary.

- [ ] **Step 5: Implement the independent verifier**

`verify_bundle` must:

1. require the exact producer artifact inventory;
2. verify every manifest digest before parsing evidence;
3. parse strict JSON/JSONL with duplicate-key and non-finite rejection;
4. independently validate source, workload, rank, admission, lifecycle,
   memory, and cleanup identities;
5. call only `classify_stage0` from the pure contract module;
6. compare reconstructed summaries and classification with
   `producer_result.json`;
7. atomically write `independent_verification.json`;
8. rewrite the manifest to include the verifier receipt.

The receipt contains:

```python
{
    "schema_version": (
        "lease-sealed-state-commit-overlap-independent-verification.v1"
    ),
    "status": "PASS",
    "producer_classification": producer["classification"],
    "reconstructed_classification": reconstructed["classification"],
    "artifact_hashes_verified": True,
    "measurement_row_count": reconstructed["measurement_row_count"],
}
```

- [ ] **Step 6: Run GREEN, mutation tests, and static checks**

```bash
python3 -m pytest \
  tools/test_lease_sealed_state_commit_overlap.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py -q
python3 -m py_compile \
  tools/assemble_lease_sealed_state_commit_overlap.py \
  tools/verify_lease_sealed_state_commit_overlap.py
git diff --check -- \
  tools/assemble_lease_sealed_state_commit_overlap.py \
  tools/verify_lease_sealed_state_commit_overlap.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py
```

Expected: all tests pass and static checks exit zero.

- [ ] **Step 7: Commit and push**

```bash
git add -- \
  tools/assemble_lease_sealed_state_commit_overlap.py \
  tools/verify_lease_sealed_state_commit_overlap.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): verify state commit overlap evidence" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

### Task 5: Add the safe remote controller and automatic launch

**Files:**

- Create: `tools/run_lease_sealed_state_commit_overlap.py`
- Create: `tools/test_run_lease_sealed_state_commit_overlap.py`

**Interfaces:**

- Produces
  `build_attempt_plan(...) -> dict`.
- Produces `build_remote_worker_commands(plan, ...) -> tuple[list[str], ...]`.
- Produces dependency-injected
  `run_attempt(plan, kerberos_probe, gpu_probe, remote_writer, worker_runner,
  assembler, remote_verifier, downloader, local_verifier) -> dict`.
- Reuses strict-clean parsing and Kerberos inspection from
  `tools/run_qwen38_tp4_communication_profile.py`.

- [ ] **Step 1: Write failing controller safety tests**

Create `tools/test_run_lease_sealed_state_commit_overlap.py`:

```python
from __future__ import annotations

from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from tools.run_lease_sealed_state_commit_overlap import (
    APPROVED_REMOTE_ROOT,
    build_attempt_plan,
    build_remote_worker_commands,
    run_attempt,
    run_ssh_with_retry,
)


def gpu(index, memory=0, utilization=0, processes=()):
    return {
        "gpu_index": index,
        "gpu_uuid": f"GPU-{index}",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
        "compute_processes": list(processes),
    }


def plan(**overrides):
    values = {
        "attempt_tag": (
            "20260907-lease-sealed-state-commit-overlap-stage0-r1"
        ),
        "source_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "selected_gpus": [gpu(index) for index in range(4)],
        "remote_path_state": {
            "attempt_exists": False,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        },
    }
    values.update(overrides)
    return build_attempt_plan(**values)


def test_every_remote_path_is_below_approved_mount():
    root = PurePosixPath(APPROVED_REMOTE_ROOT)
    candidate = plan()

    for name in (
        "attempt_root",
        "source_root",
        "raw_root",
        "bundle_root",
        "controller_root",
    ):
        assert PurePosixPath(candidate[name]).is_relative_to(root)
    for path in candidate["environment"].values():
        assert PurePosixPath(path).is_relative_to(
            PurePosixPath(candidate["attempt_root"])
        )


def test_plan_requires_fresh_path_and_four_strict_clean_gpus():
    with pytest.raises(ValueError, match="fresh"):
        plan(remote_path_state={
            "attempt_exists": True,
            "attempt_parent_is_symlink": False,
            "remote_root_is_symlink": False,
        })
    with pytest.raises(ValueError, match="four strict-clean"):
        plan(selected_gpus=[
            gpu(0),
            gpu(1),
            gpu(2),
            gpu(3, memory=1025),
        ])


def test_run_attempt_checks_auth_then_gpu_twice_and_both_verifiers():
    events = []
    clean = [gpu(index) for index in range(4)]
    result = run_attempt(
        plan(),
        kerberos_probe=lambda: events.append("kerberos")
        or {"classification": "PASS"},
        gpu_probe=lambda: events.append("gpu") or clean,
        remote_writer=lambda _plan: events.append("write")
        or {"classification": "PASS"},
        worker_runner=lambda _plan: events.append("worker")
        or {"classification": "PASS"},
        assembler=lambda _plan: events.append("assemble")
        or {"classification": "GO_LEASE_SEALED_OVERLAP_MICROGATE"},
        remote_verifier=lambda _plan: events.append("remote_verify")
        or {
            "status": "PASS",
            "reconstructed_classification": (
                "GO_LEASE_SEALED_OVERLAP_MICROGATE"
            ),
        },
        downloader=lambda _plan: events.append("download")
        or {"classification": "PASS"},
        local_verifier=lambda _plan: events.append("local_verify")
        or {
            "status": "PASS",
            "reconstructed_classification": (
                "GO_LEASE_SEALED_OVERLAP_MICROGATE"
            ),
        },
    )

    assert result["classification"] == (
        "GO_LEASE_SEALED_OVERLAP_MICROGATE"
    )
    assert events == [
        "kerberos",
        "gpu",
        "write",
        "gpu",
        "worker",
        "assemble",
        "remote_verify",
        "download",
        "local_verify",
    ]


def test_expired_auth_stops_before_remote_or_gpu_access():
    events = []
    result = run_attempt(
        plan(),
        kerberos_probe=lambda: events.append("kerberos")
        or {"classification": "BLOCKED"},
        gpu_probe=lambda: events.append("gpu") or [],
        remote_writer=lambda _plan: events.append("write"),
        worker_runner=lambda _plan: events.append("worker"),
    )
    assert result["classification"] == "BLOCKED_KERBEROS"
    assert result["worker_started"] is False
    assert events == ["kerberos"]


def test_ssh_255_retries_only_within_fixed_budget():
    returncodes = iter((255, 255, 0))
    calls = []
    result = run_ssh_with_retry(
        ["ssh", "host", "true"],
        retry_count=2,
        runner=lambda argv, **kwargs: (
            calls.append(list(argv))
            or SimpleNamespace(
                returncode=next(returncodes),
                stdout="",
                stderr="",
            )
        ),
    )
    assert result.returncode == 0
    assert len(calls) == 3
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python3 -m pytest \
  tools/test_run_lease_sealed_state_commit_overlap.py -q
```

Expected: collection fails because the controller module does not exist.

- [ ] **Step 3: Implement path, source, and launch planning**

Freeze:

```python
APPROVED_REMOTE_ROOT = (
    "/data00/home/sitian/tinyllmforge-workspaces/"
    "command-timeline-20260818"
)
DEFAULT_REMOTE_PYTHON = "/data00/home/sitian/tllm/env/bin/python"
DEFAULT_PROXY_HOST = "jump-proxy-lf"
DEFAULT_GPU_WAIT_TIMEOUT_S = 21_600
DEFAULT_GPU_POLL_INTERVAL_S = 15
DEFAULT_COMMAND_TIMEOUT_S = 60
DEFAULT_RETRY_COUNT = 3
DEFAULT_DIST_PORT = 29741
```

`build_attempt_plan` must:

- reject an existing or symlinked attempt path;
- reject any path outside `APPROVED_REMOTE_ROOT`;
- require exactly four strict-clean GPU inventory rows;
- place `TMPDIR`, `XDG_CACHE_HOME`, `TORCH_EXTENSIONS_DIR`, and
  `CUDA_CACHE_PATH` below the attempt root;
- contain no `kinit`, `krenew`, `kill`, `pkill`, or `killall` command;
- launch ranks `0..3` with one fixed world size and port.

`capture_source_identity` must require:

```text
git rev-parse HEAD == requested source revision
git status --porcelain=v1 --untracked-files=no -- tinyvllm tools == empty
```

Archive only committed `tinyvllm` and `tools` paths with `git archive`; do not
copy the full checkout or any local artifact directory.

- [ ] **Step 4: Implement monitoring, assembly, and dual verification**

Reuse:

```python
from tools.run_qwen38_tp4_communication_profile import (
    parse_nvidia_smi_inventory,
    query_local_kerberos,
    select_strict_clean_gpus,
    validate_selected_gpu_processes,
    wait_for_strict_clean_gpus,
    write_json_atomic,
)
```

`run_attempt` must execute exactly:

1. Kerberos TTL fail-fast check;
2. wait for four strict-clean GPUs;
3. create the fresh remote attempt directories;
4. upload plan and source identity;
5. stage the committed source archive;
6. repeat strict-clean admission immediately before launch;
7. launch four rank workers;
8. monitor selected GPUs and exact-tag-owned descendants;
9. assemble the producer bundle remotely;
10. run the independent verifier remotely;
11. download only the compact final bundle and remote verifier receipt;
12. run the same independent verifier locally;
13. require producer and both verifiers to agree;
14. write the controller terminal receipt.

SSH return code 255 may be retried only within the fixed retry budget.
Non-255 failures are terminal. The controller must never invoke authentication
commands or send signals to an unowned PID.

- [ ] **Step 5: Run GREEN and source-safety checks**

```bash
python3 -m pytest \
  tools/test_run_lease_sealed_state_commit_overlap.py -q
python3 -m py_compile \
  tools/run_lease_sealed_state_commit_overlap.py
rg -n \
  '(^|[^A-Za-z])(kinit|krenew|pkill|killall)([^A-Za-z]|$)|/tmp/' \
  tools/run_lease_sealed_state_commit_overlap.py
git diff --check -- \
  tools/run_lease_sealed_state_commit_overlap.py \
  tools/test_run_lease_sealed_state_commit_overlap.py
```

Expected: tests and compilation pass; `rg` has no matches; diff check is
empty.

- [ ] **Step 6: Run the complete Stage-0 local suite**

```bash
python3 -m pytest \
  tools/test_collective_side_effect_overlap.py \
  tools/test_lease_sealed_state_commit_overlap.py \
  tools/test_lease_sealed_state_commit_overlap_worker.py \
  tools/test_assemble_lease_sealed_state_commit_overlap.py \
  tools/test_verify_lease_sealed_state_commit_overlap.py \
  tools/test_run_lease_sealed_state_commit_overlap.py -q
```

Expected: zero failures.

- [ ] **Step 7: Commit and push**

```bash
git add -- \
  tools/run_lease_sealed_state_commit_overlap.py \
  tools/test_run_lease_sealed_state_commit_overlap.py
git -c core.hooksPath=/dev/null commit \
  -m "feat(tp4): orchestrate state commit overlap gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
```

Verify local, tracking, and remote SHA equality:

```bash
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

### Task 6: Run the fresh Stage-0 campaign

**Files:**

- Create locally after download:
  `artifacts/lease_sealed_state_commit_overlap/20260907-lease-sealed-state-commit-overlap-stage0-r1/`
- Create remotely:
  `/data00/home/sitian/tinyllmforge-workspaces/command-timeline-20260818/attempts/20260907-lease-sealed-state-commit-overlap-stage0-r1/`

**Interfaces:**

- Consumes the committed source revision produced by Tasks 1-5.
- Produces a terminal producer classification, remote verifier receipt, local
  verifier receipt, manifest, and cleanup record.
- Produces no model-level or production claim.

- [ ] **Step 1: Verify launch preconditions**

Run locally:

```bash
git status --porcelain=v1 --untracked-files=no -- tinyvllm tools
git rev-parse HEAD
KRB5CCNAME=/Users/bytedance/krb5cc_sitian klist -s
```

Expected:

- tracked source scope is clean;
- HEAD is a pushed commit;
- `klist -s` exits zero.

Do not run `kinit` or `krenew`.

- [ ] **Step 2: Start the controller with automatic strict-clean admission**

Run the controller in a reusable background shell so it waits and launches
immediately when four strict-clean GPUs are simultaneously available:

```bash
KRB5CCNAME=/Users/bytedance/krb5cc_sitian \
python3 \
  tools/run_lease_sealed_state_commit_overlap.py \
  --attempt-tag \
  20260907-lease-sealed-state-commit-overlap-stage0-r1 \
  --ssh-target 10.232.195.203 \
  --proxy-host jump-proxy-lf \
  --gpu-wait-timeout-s 21600 \
  --gpu-poll-interval-s 15
```

The controller, not the user, performs repeated remote GPU checks and launches
the workers as soon as admission passes. Do not start a duplicate controller
while this one is alive.

- [ ] **Step 3: Monitor transitions without inferring success from silence**

Inspect the existing background session and controller receipt. Report only:

- authentication failure;
- strict-clean admission;
- worker launch;
- resource-identity violation;
- producer completion;
- verifier disagreement;
- cleanup failure;
- terminal classification.

Do not report a running shell or elapsed time as experimental evidence.

- [ ] **Step 4: Verify the compact final bundle**

Require these exact artifacts:

```text
source_manifest.json
environment_manifest.json
gpu_rank_manifest.json
workload_manifest.json
admission.json
paired_rows.jsonl
correctness_rows.jsonl
lifecycle_rows.jsonl
memory_rows.jsonl
overlap_rows.jsonl
cleanup.json
producer_result.json
remote_independent_verification.json
local_streaming_independent_verification.json
report.md
manifest.json
manifest.sha256
```

Run:

```bash
python3 \
  tools/verify_lease_sealed_state_commit_overlap.py \
  artifacts/lease_sealed_state_commit_overlap/20260907-lease-sealed-state-commit-overlap-stage0-r1/final_bundle
```

Expected: verifier status `PASS`. Its reconstructed classification may be GO,
NO_GO, or INCONCLUSIVE; do not prejudge it.

- [ ] **Step 5: Enforce the Stage-1 stop rule**

If and only if all three classifications equal
`GO_LEASE_SEALED_OVERLAP_MICROGATE`:

```text
producer_result.json
remote_independent_verification.json
local_streaming_independent_verification.json
```

then Stage 1 becomes eligible for a separate design/implementation plan.

For every other classification:

- do not modify Qwen integration files;
- do not lower thresholds;
- do not tune from measured rows and rerun under the same evidence contract;
- preserve the terminal attempt;
- publish the reason and measured benefit/cost.

### Task 7: Publish the Stage-0 terminal audit and handoff

**Files:**

- Create:
  `docs/superpowers/audits/2026-09-07-lease-sealed-state-commit-overlap-stage0-audit.md`
- Modify: `AGENT_HANDOFF_STATE.md` by appending a new EOF section only

**Interfaces:**

- Consumes the verified immutable Stage-0 bundle.
- Produces the terminal claim boundary and exact next authorized action.

- [ ] **Step 1: Write the audit from verified artifacts**

The audit must contain:

- exact attempt tag;
- source commit and tree hash;
- GPU UUID/rank mapping;
- admission and cleanup results;
- all three classifications;
- one row per active-token shape with baseline/candidate median and P99,
  realized overlap, host-submission delta, allocated/reserved delta, and
  directional pair count;
- correctness and lifecycle results;
- measured benefit and cost;
- explicit Stage-1 authorization state;
- explicit statement that Stage 0 is not Qwen3.8 end-to-end evidence.

- [ ] **Step 2: Append the handoff checkpoint**

Append a section titled
`2026-09-07 Lease-Sealed State-Commit / AllReduce Overlap Stage-0` without
editing earlier history. Populate its fields using this exact mapping:

| Handoff field | Source |
|---|---|
| `Attempt` | `producer_result.json["attempt"]` |
| `Source` | `source_manifest.json["source_revision"]` |
| `Producer classification` | `producer_result.json["classification"]` |
| `Remote verifier` | remote receipt `status` plus `reconstructed_classification` |
| `Local verifier` | local receipt `status` plus `reconstructed_classification` |
| `Cleanup` | `cleanup.json["classification"]` |
| `Stage-1 authorized` | `true` only when all three classifications equal `GO_LEASE_SEALED_OVERLAP_MICROGATE`; otherwise `false` |
| `Evidence` | repository-relative path of the immutable compact final bundle |
| `Next action` | `write Stage-1 integration plan` only for a verified GO; otherwise `stop mechanism and preserve terminal evidence` |

Use `apply_patch` only after resolving every field to a literal value from the
verified artifacts. No template marker or unresolved field may be staged.

- [ ] **Step 3: Audit the publication**

```bash
rg -n 'T(BD)|TO(DO)|FIX(ME)|<ex(act)|<tr(ue)|<repo(sitory)' \
  docs/superpowers/audits/2026-09-07-lease-sealed-state-commit-overlap-stage0-audit.md \
  AGENT_HANDOFF_STATE.md
git diff --check -- \
  docs/superpowers/audits/2026-09-07-lease-sealed-state-commit-overlap-stage0-audit.md \
  AGENT_HANDOFF_STATE.md
```

Expected: `rg` has no matches in the newly added audit/handoff section and
diff check is empty. If older handoff content contains a match, restrict the
check to the appended line range and record that range.

- [ ] **Step 4: Commit, push, and verify SHA equality**

```bash
git add -- \
  docs/superpowers/audits/2026-09-07-lease-sealed-state-commit-overlap-stage0-audit.md \
  AGENT_HANDOFF_STATE.md
git -c core.hooksPath=/dev/null commit \
  -m "docs(tp4): record state commit overlap result" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
git push origin feat/kv-sparse-attention
git rev-parse HEAD
git rev-parse origin/feat/kv-sparse-attention
git ls-remote origin refs/heads/feat/kv-sparse-attention
```

Expected: local HEAD, tracking SHA, and remote SHA are identical.

## Completion audit

Before calling Stage 0 complete, build this prompt-to-artifact checklist from
the actual final state:

| Requirement | Required evidence |
|---|---|
| Generic mechanism only | no changes to Qwen or `linear.py`; runtime primitive exists |
| RED then GREEN | captured failing and passing focused pytest output per task |
| Three frozen shapes | 180 unique rank/pair/shape rows |
| Exact correctness | reduced output, shadow, publish, and abort checks all pass |
| Real overlap | event intervals and intersection for every candidate row |
| Benefit and cost | latency, overlap, host overhead, memory, allocation rows |
| Strict-clean TP4 | admission plus four GPU UUID/rank rows |
| Safe storage | every remote path under the approved `/data00/home/sitian` root |
| Immutable source | pushed source revision and tree hash in every artifact |
| Complete producer bundle | exact artifact inventory and valid manifest hashes |
| Independent evidence | remote and local verifier PASS with same reconstruction |
| Clean lifecycle | process group, streams, events, children, and exact-tag scans clean |
| Claim boundary | audit says mechanism-only, not Qwen E2E |
| Stage-1 authorization | true only for three-way verified GO |
| Repository publication | audit/handoff commit pushed and SHA-equal |

Any missing, weakly covered, or ambiguous row means Stage 0 is not complete.
