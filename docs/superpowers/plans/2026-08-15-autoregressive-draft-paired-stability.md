# Autoregressive Draft Paired Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are prohibited for this worktree.

**Goal:** Implement a source-bound, fixed-schedule paired-stability diagnostic, independent verifier, and safe remote runner that can classify one learned/learned bundle as `PAIRED_PROTOCOL_UNSTABLE`, `NO_REPRODUCIBLE_PROCESS_EFFECT`, or `CANDIDATE_PROCESS_BOUNDARY_EFFECT` without changing runtime semantics or launching the workload automatically.

**Architecture:** Add a new schema instead of extending or reinterpreting the completed learned A/A discovery schema. The diagnostic consumes eight uniquely identified prime/worker/telemetry/invariant epoch directories under the precommitted `AB, BA, BA, AB` schedule, validates all-or-nothing epoch admission, computes log-scale position and label effects, and emits a deterministic canonical artifact. A separate filesystem-bound verifier rehashes every source and raw input and independently recomputes the artifact; a dedicated runner owns packaging, preflight, isolated process execution, partial-evidence preservation, remote/local verification, and manifest construction.

**Tech Stack:** Python 3 standard library, pytest, Bash, SSH/rsync, existing TinyLLMForge learned-policy performance worker, `nvidia-smi`, `/proc`, `vmstat`, `mpstat`, and `pidstat`.

## Global Constraints

- The authoritative date for this plan is Saturday, August 15, 2026.
- Modify only `/Users/bytedance/dev/TinyLLMForge-adaptive-ngram`.
- Do not modify `/Users/bytedance/dev/TinyLLMForge`.
- Do not use subagents.
- Do not create or switch branches or worktrees.
- Do not stage, commit, push, stash, reset, or clean.
- Keep the current branch `feat/adaptive-ngram-speculation`.
- Use bounded foreground commands and avoid persistent PTY sessions because the unified-exec process count is already near its limit.
- Keep the existing learned A/A artifact, diagnostic, verifier, runner, and classification semantics unchanged.
- Use the fixed schedule `AB, BA, BA, AB`; do not accept a schedule override from CLI arguments or environment variables.
- Use exactly four blocks, eight prime processes, eight measured worker processes, five measured repeats per epoch, and forty measured repeats total.
- Each prime process runs exactly two warmups and one measured run; prime timing is recorded but excluded from stationarity, effect, and classification calculations.
- Each measured worker runs exactly two warmups and five measured repeats.
- Use `sitian@10.232.195.203`.
- Use `/data00/home/sitian/miniconda3/envs/py311/bin/python` only as the remote Python executable.
- Write source archives, temporary files, logs, artifacts, and experiment output only below `/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815`.
- Do not write new experiment data below `/data00`; it has only about `5.4G` free.
- Preserve target model `${REMOTE_BASE}/target-qwen3-1.7b` and draft model `${REMOTE_BASE}/draft`.
- Preserve GPU indices `3,4,6,7`.
- Preserve `MAX_PROPOSAL_TOKENS=4`, batch size four, temperature zero, accepted-prefix semantics, and exact greedy parity.
- Preserve the workload-derived exact Proposal-KV capacity.
- Do not add `torch.cuda.synchronize()` to the measured request path.
- Do not terminate unrelated GPU processes; preserve GPU-7 PID `703088`.
- Stop and reap only runner-owned worker and sampler processes.
- Do not represent synthetic or fake KV copies as real KV movement.
- Admission thresholds are inclusive: `MAD / median <= 0.10` and `half_drift <= 0.15`.
- The five-repeat half-drift halves are repeats `0,1` versus `3,4`; repeat `2` contributes only to the epoch median and MAD.
- The candidate E2E magnitude threshold is inclusive at `10%`.
- Any source, identity, parity, coverage, invariant, repeat-count, or stationarity failure forces `PAIRED_PROTOCOL_UNSTABLE`.
- Every artifact and receipt must force `process_boundary_effect_established=false`.
- One bundle can produce only a candidate; it cannot establish a process-boundary cause, a performance improvement, Phase-1 completion, or promotion readiness.
- This implementation plan authorizes only source changes and local validation. It does not authorize a remote workload or a replication bundle.

## File Map

- Create `tools/autoregressive_draft_paired_stability_diagnostic.py`: fixed schedule constants, epoch identity model, raw-input schema validation, coverage and invariant checks, stationarity admission, paired effects, classification, canonical artifact assembly, and CLI.
- Create `tools/verify_autoregressive_draft_paired_stability_diagnostic.py`: safe path resolution, hash verification, complete raw-input reload, independent recomputation, exact structural comparison, receipt construction, and CLI.
- Create `tools/test_autoregressive_draft_paired_stability_diagnostic.py`: fixtures and focused tests for schedule identity, admission, calculations, classification precedence, tamper rejection, verifier equivalence, and manifest inventory.
- Create `tools/run_autoregressive_draft_paired_stability_remote.sh`: fixed schedule materialization, source packaging, remote preflight, isolated primes/workers, sampler ownership, invariant snapshots, partial-evidence preservation, canonical assembly, remote/local verification, and manifest construction.
- Modify `tools/test_autoregressive_draft_instability_telemetry.py`: add executable-mode, fixed-schedule, dependency-closure, safety, storage, process-ownership, and manifest contracts for the new runner.
- Modify `AGENT_HANDOFF_STATE.md`: record implementation and local-validation status only after all local gates pass; explicitly retain the remote-execution block.

---

### Task 1: Fixed Schedule, Epoch Identity, and Worker Semantics

**Files:**
- Create: `tools/test_autoregressive_draft_paired_stability_diagnostic.py`
- Create: `tools/autoregressive_draft_paired_stability_diagnostic.py`

**Interfaces:**
- Consumes: `validate_worker_result(worker, expected_warmup_runs, expected_measured_runs) -> dict` and `proposal_slot_capacity_for_batch(batch_size, max_proposal_tokens) -> int` from `tools/autoregressive_draft_performance_gate.py`.
- Produces: `EpochIdentity(block_index: int, order: str, label: str, position: str, epoch_index: int)`.
- Produces: `expected_epoch_identities() -> tuple[EpochIdentity, ...]`.
- Produces: `validate_prime_worker(worker: object, *, identity: EpochIdentity) -> dict`.
- Produces: `validate_measured_worker(worker: object, *, identity: EpochIdentity) -> dict`.
- Produces: `validate_epoch_workload_identity(epochs: dict[str, dict]) -> dict`.

- [ ] **Step 1: Write schedule and epoch-identity tests**

Add imports and tests that make the exact schedule and all eight identities executable requirements:

```python
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import statistics
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_PATH = (
    ROOT / "tools" / "autoregressive_draft_paired_stability_diagnostic.py"
)
VERIFIER_PATH = (
    ROOT
    / "tools"
    / "verify_autoregressive_draft_paired_stability_diagnostic.py"
)
RUNNER_PATH = (
    ROOT / "tools" / "run_autoregressive_draft_paired_stability_remote.sh"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_fixed_schedule_and_digest():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_schedule")
    assert diagnostic.BLOCK_SCHEDULE == (
        ("A", "B"),
        ("B", "A"),
        ("B", "A"),
        ("A", "B"),
    )
    assert diagnostic.SCHEDULE_TEXT == "AB\nBA\nBA\nAB\n"
    assert diagnostic.SCHEDULE_SHA256 == hashlib.sha256(
        diagnostic.SCHEDULE_TEXT.encode("utf-8")
    ).hexdigest()


def test_expected_epoch_identities_are_unique_and_ordered():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_identities")
    identities = diagnostic.expected_epoch_identities()
    assert len(identities) == 8
    assert len({identity.key for identity in identities}) == 8
    assert [
        (
            identity.block_index,
            identity.order,
            identity.label,
            identity.position,
            identity.epoch_index,
        )
        for identity in identities
    ] == [
        (0, "AB", "A", "first", 0),
        (0, "AB", "B", "second", 1),
        (1, "BA", "B", "first", 2),
        (1, "BA", "A", "second", 3),
        (2, "BA", "B", "first", 4),
        (2, "BA", "A", "second", 5),
        (3, "AB", "A", "first", 6),
        (3, "AB", "B", "second", 7),
    ]
```

- [ ] **Step 2: Write worker-semantic rejection tests**

Build prime and measured fixtures from the existing performance-gate fixture helpers, preserving model identity, prompt rows, exact Proposal-KV capacity derivation, accepted outputs, runtime counters, and timing intervals. Add independent rejection tests:

```python
def test_prime_is_two_warmups_and_one_measured_run(valid_prime, identity):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_prime")
    invalid = copy.deepcopy(valid_prime)
    invalid["measured_runs"].append(copy.deepcopy(invalid["measured_runs"][0]))
    with pytest.raises(ValueError, match="prime worker"):
        diagnostic.validate_prime_worker(invalid, identity=identity)


def test_measured_worker_is_two_warmups_and_five_repeats(
    valid_worker,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_worker_count")
    invalid = copy.deepcopy(valid_worker)
    invalid["measured_runs"].pop()
    with pytest.raises(ValueError, match="five measured repeats"):
        diagnostic.validate_measured_worker(invalid, identity=identity)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy", "target", "policy must be learned"),
        ("batch_size", 1, "batch size must be four"),
        ("temperature", 0.5, "temperature must be zero"),
    ],
)
def test_worker_fixed_runtime_fields(
    valid_worker,
    identity,
    field,
    value,
    message,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_worker_fields")
    invalid = copy.deepcopy(valid_worker)
    invalid[field] = value
    with pytest.raises(ValueError, match=message):
        diagnostic.validate_measured_worker(invalid, identity=identity)


def test_epoch_identity_rejects_wrong_schedule_position(
    valid_worker,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_wrong_identity")
    wrong = diagnostic.EpochIdentity(
        block_index=identity.block_index,
        order=identity.order,
        label=identity.label,
        position="second" if identity.position == "first" else "first",
        epoch_index=identity.epoch_index,
    )
    with pytest.raises(ValueError, match="epoch identity"):
        diagnostic.validate_measured_worker(valid_worker, identity=wrong)
```

Add exact-equality tests for target checkpoint identity, draft checkpoint identity, tokenizer identity, prompts and prompt token IDs, requested output lengths, request order, TP world size, GPU indices, proposal counts, proposal lengths, total verified-token counts, accepted token IDs, output token IDs, accepted-prefix metadata, `MAX_PROPOSAL_TOKENS=4`, and `proposal_slot_capacity_for_batch(4, 4)`.

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  -k 'schedule or identity or prime or worker or workload'
```

Expected: collection fails because `autoregressive_draft_paired_stability_diagnostic.py` does not exist.

- [ ] **Step 4: Implement fixed constants and identity validation**

Create the module with these definitions:

```python
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_host_semantic_diagnostic import (
    align_repeat_samples,
    derive_repeat_metrics,
    parse_host_jsonl,
)
from autoregressive_draft_instability_telemetry import (
    parse_gpu_telemetry,
    summarize_gpu_telemetry,
    validate_campaign_intervals,
)
from autoregressive_draft_performance_gate import (
    proposal_slot_capacity_for_batch,
    validate_worker_result,
)


SCHEMA_VERSION = 1
BLOCK_SCHEDULE = (
    ("A", "B"),
    ("B", "A"),
    ("B", "A"),
    ("A", "B"),
)
SCHEDULE_TEXT = "".join("".join(block) + "\n" for block in BLOCK_SCHEDULE)
SCHEDULE_SHA256 = hashlib.sha256(SCHEDULE_TEXT.encode("utf-8")).hexdigest()
PRIMARY_METRICS = (
    "e2e_s",
    "tpot_s",
    "executor_proposal_forward_ms",
)
DIAGNOSTIC_METRICS = ("executor_backend_submit_ms",)
WORKER_POLICY = "learned"
PRIME_WARMUP_RUNS = 2
PRIME_MEASURED_RUNS = 1
MEASURED_WARMUP_RUNS = 2
MEASURED_RUNS_PER_EPOCH = 5
MEASURED_RUNS_TOTAL = 40
BATCH_SIZE = 4
MAX_PROPOSAL_TOKENS = 4
GPU_INDICES = (3, 4, 6, 7)
PROTECTED_GPU7_PID = 703088
TEMPERATURE = 0.0
ROBUST_DISPERSION_LIMIT = 0.10
HALF_DRIFT_LIMIT = 0.15
EFFECT_MAGNITUDE_THRESHOLD = 0.10
CLASSIFICATIONS = (
    "PAIRED_PROTOCOL_UNSTABLE",
    "NO_REPRODUCIBLE_PROCESS_EFFECT",
    "CANDIDATE_PROCESS_BOUNDARY_EFFECT",
)


@dataclass(frozen=True)
class EpochIdentity:
    block_index: int
    order: str
    label: str
    position: str
    epoch_index: int

    @property
    def key(self) -> str:
        return (
            f"block-{self.block_index}-{self.order.lower()}/"
            f"{self.label.lower()}-{self.position}"
        )


def expected_epoch_identities() -> tuple[EpochIdentity, ...]:
    identities = []
    epoch_index = 0
    for block_index, labels in enumerate(BLOCK_SCHEDULE):
        order = "".join(labels)
        for position, label in zip(("first", "second"), labels):
            identities.append(
                EpochIdentity(
                    block_index=block_index,
                    order=order,
                    label=label,
                    position=position,
                    epoch_index=epoch_index,
                )
            )
            epoch_index += 1
    return tuple(identities)


def _require_expected_identity(identity: EpochIdentity) -> None:
    if identity not in expected_epoch_identities():
        raise ValueError("epoch identity is not in the fixed schedule")


def _validate_worker(
    worker: object,
    *,
    identity: EpochIdentity,
    expected_measured_runs: int,
    kind: str,
) -> dict:
    _require_expected_identity(identity)
    if not isinstance(worker, dict):
        raise ValueError(f"{kind} must be a mapping")
    if worker.get("policy") != WORKER_POLICY:
        raise ValueError(f"{kind} policy must be learned")
    if worker.get("batch_size") != BATCH_SIZE:
        raise ValueError(f"{kind} batch size must be four")
    if float(worker.get("temperature", 0.0)) != TEMPERATURE:
        raise ValueError(f"{kind} temperature must be zero")
    normalized = validate_worker_result(
        worker,
        expected_warmup_runs=MEASURED_WARMUP_RUNS,
        expected_measured_runs=expected_measured_runs,
    )
    if len(normalized["measured_runs"]) != expected_measured_runs:
        raise ValueError(f"{kind} measured repeat count is invalid")
    return normalized


def validate_prime_worker(
    worker: object,
    *,
    identity: EpochIdentity,
) -> dict:
    return _validate_worker(
        worker,
        identity=identity,
        expected_measured_runs=PRIME_MEASURED_RUNS,
        kind="prime worker",
    )


def validate_measured_worker(
    worker: object,
    *,
    identity: EpochIdentity,
) -> dict:
    normalized = _validate_worker(
        worker,
        identity=identity,
        expected_measured_runs=MEASURED_RUNS_PER_EPOCH,
        kind="measured worker",
    )
    if len(normalized["measured_runs"]) != 5:
        raise ValueError("measured worker must contain five measured repeats")
    return normalized
```

Implement `validate_epoch_workload_identity` as exact comparisons against the first epoch after all eight identities are present. Return a normalized identity map only after verifying the runtime fields listed in Step 2 and:

```python
expected_capacity = proposal_slot_capacity_for_batch(
    BATCH_SIZE,
    MAX_PROPOSAL_TOKENS,
)
if epoch["proposal_kv_capacity"]["slots"] != expected_capacity:
    raise ValueError("Proposal-KV capacity is not the workload-derived bound")
```

- [ ] **Step 5: Run the focused tests and confirm GREEN**

Run the Step 3 command.

Expected: all selected tests pass.

- [ ] **Step 6: Record the task checkpoint without staging**

Run:

```bash
git diff --check -- \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
git status --short -- \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: `git diff --check` is silent and both files remain unstaged.

---

### Task 2: Epoch Raw Inputs, Coverage, Invariants, and Stationarity Admission

**Files:**
- Modify: `tools/test_autoregressive_draft_paired_stability_diagnostic.py`
- Modify: `tools/autoregressive_draft_paired_stability_diagnostic.py`

**Interfaces:**
- Consumes: `EpochIdentity`, `validate_prime_worker`, and `validate_measured_worker` from Task 1.
- Consumes: `parse_gpu_telemetry`, `summarize_gpu_telemetry`, `validate_campaign_intervals`, `parse_host_jsonl`, `align_repeat_samples`, and `derive_repeat_metrics`.
- Produces: `AdmissionFailure(code: str, identity: EpochIdentity | None, metric: str | None, observed: object, expected: str, source_path: str)`.
- Produces: `stationarity_for_values(metric: str, values: list[float]) -> dict`.
- Produces: `build_epoch_admission(identity: EpochIdentity, raw: dict) -> dict`.
- Produces: `build_bundle_admission(epochs: dict[str, dict]) -> dict`.

- [ ] **Step 1: Write stationarity boundary tests**

Use explicit five-value inputs to prove the center repeat is excluded from half medians:

```python
def test_stationarity_uses_all_values_for_median_and_mad():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_stationarity_all")
    row = diagnostic.stationarity_for_values(
        "e2e_s",
        [9.0, 10.0, 1000.0, 11.0, 12.0],
    )
    assert row["epoch_median"] == 11.0
    assert row["epoch_mad"] == 1.0


def test_half_drift_excludes_center_repeat():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_half_drift")
    row = diagnostic.stationarity_for_values(
        "e2e_s",
        [10.0, 10.0, 500.0, 11.5, 11.5],
    )
    assert row["first_half_values"] == [10.0, 10.0]
    assert row["center_value"] == 500.0
    assert row["second_half_values"] == [11.5, 11.5]
    assert row["half_drift"] == pytest.approx(
        abs(11.5 - 10.0) / row["epoch_median"]
    )


@pytest.mark.parametrize(
    ("values", "stable"),
    [
        ([9.0, 9.0, 10.0, 11.0, 11.0], True),
        ([8.9, 8.9, 10.0, 11.1, 11.1], False),
    ],
)
def test_robust_dispersion_equality_passes(values, stable):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_mad_boundary")
    assert (
        diagnostic.stationarity_for_values("e2e_s", values)[
            "robust_dispersion"
        ]
        <= diagnostic.ROBUST_DISPERSION_LIMIT
    ) is stable


@pytest.mark.parametrize(
    ("values", "stable"),
    [
        ([10.0, 10.0, 10.0, 11.5, 11.5], True),
        ([10.0, 10.0, 10.0, 11.501, 11.501], False),
    ],
)
def test_half_drift_equality_passes(values, stable):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_drift_boundary")
    assert (
        diagnostic.stationarity_for_values("e2e_s", values)["stable"]
        is stable
    )
```

Also reject zero, negative, non-finite, non-list, and non-five-value inputs.

- [ ] **Step 2: Write coverage and invariant rejection tests**

Construct one valid raw epoch fixture with all required paths and add one mutation per test:

```python
def test_epoch_rejects_duplicate_repeat_index(valid_raw_epoch, identity):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_duplicate_repeat")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["worker"]["measured_runs"][4]["repeat"] = 3
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "DUPLICATE_REPEAT_INDEX" in {
        failure["code"] for failure in admission["failures"]
    }


def test_epoch_rejects_missing_gpu_coverage(valid_raw_epoch, identity):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_gpu_coverage")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["gpu_rows"] = invalid["gpu_rows"][:1]
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "GPU_TELEMETRY_COVERAGE" in {
        failure["code"] for failure in admission["failures"]
    }


def test_epoch_rejects_protected_process_disappearance(
    valid_raw_epoch,
    identity,
):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_protected_pid")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["process_after"]["protected_gpu7_pid_present"] = False
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "PROTECTED_PROCESS_MISSING" in {
        failure["code"] for failure in admission["failures"]
    }


def test_epoch_rejects_script_owned_process_leak(valid_raw_epoch, identity):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_process_leak")
    invalid = copy.deepcopy(valid_raw_epoch)
    invalid["process_after"]["runner_owned_pids_remaining"] = [999999]
    admission = diagnostic.build_epoch_admission(identity, invalid)
    assert admission["passed"] is False
    assert "RUNNER_PROCESS_LEAK" in {
        failure["code"] for failure in admission["failures"]
    }
```

Add tests for missing/extra repeat records, non-monotonic timestamps, proposal-forward coverage, backend-submit coverage, host telemetry gaps, GPU UUID changes, undeclared GPU usage, Xid, reset, throttle validity failure, unavailable telemetry, clock/P-state invalidity, changed unrelated-process inventory, exact parity failure, accepted-prefix mismatch, proposal-count mismatch, proposal-length mismatch, total verified-token mismatch, and prime data entering measured statistics.

- [ ] **Step 3: Run the admission tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  -k 'stationarity or coverage or invariant or parity or accepted_prefix'
```

Expected: failures report missing admission and stationarity functions.

- [ ] **Step 4: Implement stable failure records and stationarity**

Add:

```python
@dataclass(frozen=True)
class AdmissionFailure:
    code: str
    identity: EpochIdentity | None
    metric: str | None
    observed: object
    expected: str
    source_path: str

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "block": (
                None if self.identity is None else self.identity.block_index
            ),
            "label": None if self.identity is None else self.identity.label,
            "position": (
                None if self.identity is None else self.identity.position
            ),
            "epoch": (
                None if self.identity is None else self.identity.epoch_index
            ),
            "metric": self.metric,
            "observed": self.observed,
            "expected": self.expected,
            "source_path": self.source_path,
        }


def stationarity_for_values(metric: str, values: list[float]) -> dict:
    if metric not in PRIMARY_METRICS:
        raise ValueError("stationarity metric is invalid")
    if not isinstance(values, list) or len(values) != 5:
        raise ValueError("stationarity requires exactly five values")
    normalized = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError("stationarity values must be finite and positive")
        normalized.append(float(value))
    epoch_median = statistics.median(normalized)
    absolute_deviations = [
        abs(value - epoch_median) for value in normalized
    ]
    epoch_mad = statistics.median(absolute_deviations)
    robust_dispersion = epoch_mad / epoch_median
    first_half_values = normalized[0:2]
    center_value = normalized[2]
    second_half_values = normalized[3:5]
    first_half_median = statistics.median(first_half_values)
    second_half_median = statistics.median(second_half_values)
    half_drift = (
        abs(second_half_median - first_half_median) / epoch_median
    )
    return {
        "metric": metric,
        "values": normalized,
        "epoch_median": epoch_median,
        "epoch_mad": epoch_mad,
        "robust_dispersion": robust_dispersion,
        "first_half_values": first_half_values,
        "center_value": center_value,
        "second_half_values": second_half_values,
        "first_half_median": first_half_median,
        "second_half_median": second_half_median,
        "half_drift": half_drift,
        "robust_dispersion_limit": ROBUST_DISPERSION_LIMIT,
        "half_drift_limit": HALF_DRIFT_LIMIT,
        "stable": (
            robust_dispersion <= ROBUST_DISPERSION_LIMIT
            and half_drift <= HALF_DRIFT_LIMIT
        ),
    }
```

- [ ] **Step 5: Implement epoch and bundle admission**

`build_epoch_admission` must catch each check independently and append stable `AdmissionFailure` rows rather than stopping after the first failure. Its return shape is:

```python
{
    "identity": asdict(identity) | {"key": identity.key},
    "passed": not failures,
    "failures": [failure.to_dict() for failure in failures],
    "prime": {
        "recorded": True,
        "excluded_from_measured_statistics": True,
    },
    "repeat_count": len(measured_worker["measured_runs"]),
    "metrics": metric_values,
    "stationarity": stationarity,
    "coverage": coverage,
    "gpu_invariants": gpu_invariants,
    "process_invariants": process_invariants,
    "exact_parity": exact_parity,
    "accepted_prefix_semantics": accepted_prefix_semantics,
    "telemetry": {
        "gpu": gpu_summary,
        "host": host_summary,
    },
}
```

`build_bundle_admission` must require exactly the keys from `expected_epoch_identities()` and return:

```python
def build_bundle_admission(epochs: dict[str, dict]) -> dict:
    expected_keys = tuple(
        identity.key for identity in expected_epoch_identities()
    )
    if tuple(epochs) != expected_keys:
        raise ValueError("epoch inventory or order is invalid")
    failures = [
        failure
        for key in expected_keys
        for failure in epochs[key]["failures"]
    ]
    return {
        "passed": not failures,
        "epoch_count": len(expected_keys),
        "measured_repeat_count_total": sum(
            epochs[key]["repeat_count"] for key in expected_keys
        ),
        "failed_epoch_keys": [
            key for key in expected_keys if not epochs[key]["passed"]
        ],
        "failures": failures,
    }
```

Do not delete, replace, or rerun an epoch in diagnostic code. If any epoch fails, retain every raw and derived row and mark the complete bundle inadmissible.

- [ ] **Step 6: Run the admission tests and confirm GREEN**

Run the Step 3 command.

Expected: all selected tests pass.

- [ ] **Step 7: Record the task checkpoint without staging**

Run:

```bash
git diff --check -- \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: no output.

---

### Task 3: Paired Effects, Order Check, and Classification Precedence

**Files:**
- Modify: `tools/test_autoregressive_draft_paired_stability_diagnostic.py`
- Modify: `tools/autoregressive_draft_paired_stability_diagnostic.py`

**Interfaces:**
- Consumes: admitted epoch summaries from Task 2.
- Produces: `compute_paired_effects(epochs: dict[str, dict]) -> dict`.
- Produces: `order_effect_check(effects: dict) -> dict`.
- Produces: `classify_paired_stability(bundle_admission: dict, effects: dict, order_check: dict) -> tuple[str, bool, list[str]]`.

- [ ] **Step 1: Write exact effect-calculation tests**

Use medians that make the log ratios transparent:

```python
def test_ab_block_position_and_label_effects():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_ab_effects")
    epochs = make_admitted_epoch_summaries(
        block_medians={
            0: {"A": 10.0, "B": 12.0},
            1: {"B": 10.0, "A": 12.0},
            2: {"B": 10.0, "A": 12.0},
            3: {"A": 10.0, "B": 12.0},
        }
    )
    effects = diagnostic.compute_paired_effects(epochs)
    block0 = effects["block_local"]["e2e_s"][0]
    assert block0["position_effect"] == pytest.approx(math.log(12.0 / 10.0))
    assert block0["position_relative"] == pytest.approx(0.20)
    assert block0["label_effect"] == pytest.approx(math.log(10.0 / 12.0))
    assert block0["label_relative"] == pytest.approx(-1.0 / 6.0)


def test_ba_block_position_and_label_effects():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_ba_effects")
    epochs = make_admitted_epoch_summaries(
        block_medians={
            0: {"A": 10.0, "B": 12.0},
            1: {"B": 10.0, "A": 12.0},
            2: {"B": 10.0, "A": 12.0},
            3: {"A": 10.0, "B": 12.0},
        }
    )
    effects = diagnostic.compute_paired_effects(epochs)
    block1 = effects["block_local"]["e2e_s"][1]
    assert block1["position_effect"] == pytest.approx(math.log(12.0 / 10.0))
    assert block1["label_effect"] == pytest.approx(math.log(12.0 / 10.0))


def test_aggregate_and_sequence_interaction_are_log_medians():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_aggregate_effects")
    effects = diagnostic.compute_paired_effects(
        make_effect_fixture(
            e2e_position_relatives=[0.10, 0.20, 0.30, 0.40],
            e2e_label_relatives=[0.01, -0.01, 0.02, -0.02],
        )
    )
    expected = statistics.median(
        [math.log1p(value) for value in [0.10, 0.20, 0.30, 0.40]]
    )
    assert effects["aggregate_position_effects"]["e2e_s"][
        "log_effect"
    ] == pytest.approx(expected)
    assert effects["sequence_interactions"]["e2e_s"] == pytest.approx(
        effects["ab_position_effects"]["e2e_s"]["log_effect"]
        - effects["ba_position_effects"]["e2e_s"]["log_effect"]
    )
```

Add tests for backend-submit diagnostic effects, raw repeat ratios, chronological block trend, acceptance summaries, proposal-length summaries, and verified-token summaries.

- [ ] **Step 2: Write classification and boundary tests**

```python
def test_any_admission_failure_has_highest_precedence():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_unstable_precedence")
    admission = {"passed": False, "failures": [{"code": "HALF_DRIFT"}]}
    effects = make_candidate_effects()
    order_check = {"passed": True, "reasons": []}
    classification, candidate, reasons = (
        diagnostic.classify_paired_stability(
            admission,
            effects,
            order_check,
        )
    )
    assert classification == "PAIRED_PROTOCOL_UNSTABLE"
    assert candidate is False
    assert reasons == ["bundle admission failed"]


def test_candidate_accepts_exact_ten_percent_aggregate_boundary():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_ten_percent")
    effects = make_candidate_effects(
        e2e_block_relatives=[0.10, 0.10, 0.10, -0.01],
        e2e_aggregate_relative=0.10,
    )
    classification, candidate, _ = diagnostic.classify_paired_stability(
        {"passed": True, "failures": []},
        effects,
        {"passed": True, "reasons": []},
    )
    assert classification == "CANDIDATE_PROCESS_BOUNDARY_EFFECT"
    assert candidate is True


def test_candidate_requires_three_of_four_common_direction():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_three_of_four")
    effects = make_candidate_effects(
        e2e_block_relatives=[0.20, 0.20, -0.20, -0.20]
    )
    classification, candidate, reasons = (
        diagnostic.classify_paired_stability(
            {"passed": True, "failures": []},
            effects,
            {"passed": True, "reasons": []},
        )
    )
    assert classification == "NO_REPRODUCIBLE_PROCESS_EFFECT"
    assert candidate is False
    assert "fewer than three E2E blocks share a direction" in reasons


@pytest.mark.parametrize(
    "metric",
    ["tpot_s", "executor_proposal_forward_ms"],
)
def test_candidate_rejects_primary_direction_disagreement(metric):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_direction_disagreement")
    effects = make_candidate_effects()
    effects["aggregate_position_effects"][metric]["log_effect"] *= -1.0
    effects["aggregate_position_effects"][metric]["relative_effect"] = (
        math.exp(
            effects["aggregate_position_effects"][metric]["log_effect"]
        )
        - 1.0
    )
    classification, candidate, _ = diagnostic.classify_paired_stability(
        {"passed": True, "failures": []},
        effects,
        {"passed": True, "reasons": []},
    )
    assert classification == "NO_REPRODUCIBLE_PROCESS_EFFECT"
    assert candidate is False
```

Add order-check tests for opposite AB/BA signs, no qualifying AB block, no qualifying BA block, aggregate label magnitude exactly `10%` failing because it must be below `10%`, and four label effects independently satisfying the direction-and-magnitude rule.

- [ ] **Step 3: Run effect and classification tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  -k 'effect or aggregate or sequence or classification or candidate or order'
```

Expected: failures report missing effect and classification functions.

- [ ] **Step 4: Implement block-local and aggregate effects**

Add:

```python
def _effect_row(first: float, second: float) -> dict:
    if first <= 0.0 or second <= 0.0:
        raise ValueError("effect medians must be positive")
    log_effect = math.log(second) - math.log(first)
    return {
        "log_effect": log_effect,
        "relative_effect": math.exp(log_effect) - 1.0,
    }


def _aggregate_effect(rows: list[dict], key: str) -> dict:
    log_effect = statistics.median(row[key] for row in rows)
    return {
        "log_effect": log_effect,
        "relative_effect": math.exp(log_effect) - 1.0,
    }


def compute_paired_effects(epochs: dict[str, dict]) -> dict:
    block_local = {
        metric: [] for metric in PRIMARY_METRICS + DIAGNOSTIC_METRICS
    }
    for block_index, labels in enumerate(BLOCK_SCHEDULE):
        first_identity, second_identity = [
            identity
            for identity in expected_epoch_identities()
            if identity.block_index == block_index
        ]
        first_epoch = epochs[first_identity.key]
        second_epoch = epochs[second_identity.key]
        by_label = {
            first_identity.label: first_epoch,
            second_identity.label: second_epoch,
        }
        for metric in block_local:
            first_median = first_epoch["metric_medians"][metric]
            second_median = second_epoch["metric_medians"][metric]
            label_a_median = by_label["A"]["metric_medians"][metric]
            label_b_median = by_label["B"]["metric_medians"][metric]
            position = _effect_row(first_median, second_median)
            label = _effect_row(label_b_median, label_a_median)
            block_local[metric].append(
                {
                    "block_index": block_index,
                    "order": "".join(labels),
                    "first_epoch_key": first_identity.key,
                    "second_epoch_key": second_identity.key,
                    "position_effect": position["log_effect"],
                    "position_relative": position["relative_effect"],
                    "label_effect": label["log_effect"],
                    "label_relative": label["relative_effect"],
                }
            )
    aggregate_position = {}
    aggregate_label = {}
    ab_position = {}
    ba_position = {}
    sequence_interactions = {}
    for metric, rows in block_local.items():
        aggregate_position[metric] = _aggregate_effect(
            rows,
            "position_effect",
        )
        aggregate_label[metric] = _aggregate_effect(rows, "label_effect")
        ab_rows = [row for row in rows if row["order"] == "AB"]
        ba_rows = [row for row in rows if row["order"] == "BA"]
        ab_position[metric] = _aggregate_effect(
            ab_rows,
            "position_effect",
        )
        ba_position[metric] = _aggregate_effect(
            ba_rows,
            "position_effect",
        )
        sequence_interactions[metric] = (
            ab_position[metric]["log_effect"]
            - ba_position[metric]["log_effect"]
        )
    return {
        "block_local": block_local,
        "aggregate_position_effects": aggregate_position,
        "aggregate_label_effects": aggregate_label,
        "ab_position_effects": ab_position,
        "ba_position_effects": ba_position,
        "sequence_interactions": sequence_interactions,
        "diagnostic_effects": build_diagnostic_effects(epochs, block_local),
    }
```

Implement `build_diagnostic_effects` to retain backend-submit rows, repeat-local ratios, block-index trends, GPU and host summaries, acceptance, proposal-length, and verified-token summaries without using them as candidate criteria.

- [ ] **Step 5: Implement order-effect and classification logic**

Add:

```python
def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def order_effect_check(effects: dict) -> dict:
    metric = "e2e_s"
    aggregate = effects["aggregate_position_effects"][metric]
    aggregate_sign = _sign(aggregate["log_effect"])
    ab = effects["ab_position_effects"][metric]
    ba = effects["ba_position_effects"][metric]
    blocks = effects["block_local"][metric]
    label_aggregate = effects["aggregate_label_effects"][metric]
    label_directions = [
        _sign(row["label_effect"]) for row in blocks
    ]
    positive_label_count = label_directions.count(1)
    negative_label_count = label_directions.count(-1)
    label_common_sign = (
        1
        if positive_label_count >= 3
        else -1
        if negative_label_count >= 3
        else 0
    )
    label_candidate = (
        label_common_sign != 0
        and sum(
            _sign(row["label_effect"]) == label_common_sign
            and abs(row["label_relative"])
            >= EFFECT_MAGNITUDE_THRESHOLD
            for row in blocks
        )
        >= 3
    )
    checks = {
        "ab_direction_matches": _sign(ab["log_effect"]) == aggregate_sign != 0,
        "ba_direction_matches": _sign(ba["log_effect"]) == aggregate_sign != 0,
        "ab_has_qualifying_block": any(
            row["order"] == "AB"
            and _sign(row["position_effect"]) == aggregate_sign
            and abs(row["position_relative"])
            >= EFFECT_MAGNITUDE_THRESHOLD
            for row in blocks
        ),
        "ba_has_qualifying_block": any(
            row["order"] == "BA"
            and _sign(row["position_effect"]) == aggregate_sign
            and abs(row["position_relative"])
            >= EFFECT_MAGNITUDE_THRESHOLD
            for row in blocks
        ),
        "aggregate_label_below_threshold": (
            abs(label_aggregate["relative_effect"])
            < EFFECT_MAGNITUDE_THRESHOLD
        ),
        "label_does_not_form_candidate": not label_candidate,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "sequence_interaction": effects["sequence_interactions"][metric],
        "reasons": [name for name, passed in checks.items() if not passed],
    }


def classify_paired_stability(
    bundle_admission: dict,
    effects: dict,
    order_check: dict,
) -> tuple[str, bool, list[str]]:
    if not bundle_admission["passed"]:
        return (
            "PAIRED_PROTOCOL_UNSTABLE",
            False,
            ["bundle admission failed"],
        )
    e2e_rows = effects["block_local"]["e2e_s"]
    aggregate = effects["aggregate_position_effects"]["e2e_s"]
    aggregate_sign = _sign(aggregate["log_effect"])
    same_direction_count = sum(
        _sign(row["position_effect"]) == aggregate_sign
        for row in e2e_rows
    )
    reasons = []
    if aggregate_sign == 0 or same_direction_count < 3:
        reasons.append("fewer than three E2E blocks share a direction")
    if abs(aggregate["relative_effect"]) < EFFECT_MAGNITUDE_THRESHOLD:
        reasons.append("aggregate E2E magnitude is below ten percent")
    for metric in ("tpot_s", "executor_proposal_forward_ms"):
        if (
            _sign(
                effects["aggregate_position_effects"][metric][
                    "log_effect"
                ]
            )
            != aggregate_sign
        ):
            reasons.append(f"{metric} aggregate direction disagrees")
    if not order_check["passed"]:
        reasons.append("E2E order-effect check failed")
    if reasons:
        return "NO_REPRODUCIBLE_PROCESS_EFFECT", False, reasons
    return "CANDIDATE_PROCESS_BOUNDARY_EFFECT", True, []
```

- [ ] **Step 6: Run effect and classification tests and confirm GREEN**

Run the Step 3 command.

Expected: all selected tests pass.

- [ ] **Step 7: Record the task checkpoint without staging**

Run:

```bash
git diff --check -- \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: no output.

---

### Task 4: Canonical Artifact and Deterministic CLI Assembly

**Files:**
- Modify: `tools/test_autoregressive_draft_paired_stability_diagnostic.py`
- Modify: `tools/autoregressive_draft_paired_stability_diagnostic.py`

**Interfaces:**
- Consumes: all Task 1-3 functions.
- Produces: `build_paired_stability_artifact(*, metadata: dict, epoch_raw_inputs: dict[str, dict], input_files: dict[str, dict], source_files: dict[str, str]) -> dict`.
- Produces: `validate_paired_stability_artifact(artifact: object) -> dict`.
- Produces CLI arguments `--bundle-root`, `--repo-root`, and `--out`.

- [ ] **Step 1: Write canonical artifact tests**

```python
def test_canonical_artifact_is_self_describing(valid_bundle_inputs):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_artifact")
    artifact = diagnostic.build_paired_stability_artifact(
        **valid_bundle_inputs
    )
    required = {
        "schema_version",
        "classification",
        "candidate_process_boundary_effect",
        "process_boundary_effect_established",
        "claim_boundary",
        "run_tag",
        "bundle_start_utc",
        "bundle_finish_utc",
        "remote_host",
        "remote_base",
        "schedule",
        "schedule_sha256",
        "configuration",
        "source_files",
        "source_sha256",
        "model_identity",
        "prompt_identity",
        "command_identity",
        "blocks",
        "epochs",
        "measured_repeat_count_total",
        "epoch_admission",
        "bundle_admission",
        "primary_stationarity",
        "coverage",
        "gpu_invariants",
        "process_invariants",
        "exact_parity",
        "block_local_position_effects",
        "block_local_label_effects",
        "aggregate_position_effects",
        "aggregate_label_effects",
        "ab_position_effects",
        "ba_position_effects",
        "sequence_interactions",
        "diagnostic_effects",
        "raw_input_files",
        "raw_input_sha256",
    }
    assert required <= set(artifact)
    assert artifact["process_boundary_effect_established"] is False
    assert artifact["schedule"] == ["AB", "BA", "BA", "AB"]
    assert artifact["schedule_sha256"] == diagnostic.SCHEDULE_SHA256
    assert len(artifact["blocks"]) == 4
    assert len(artifact["epochs"]) == 8
    assert artifact["measured_repeat_count_total"] == 40


def test_unstable_artifact_retains_all_raw_inputs(valid_bundle_inputs):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_partial_evidence")
    invalid = copy.deepcopy(valid_bundle_inputs)
    first_key = diagnostic.expected_epoch_identities()[0].key
    invalid["epoch_raw_inputs"][first_key]["worker"][
        "measured_runs"
    ].pop()
    artifact = diagnostic.build_paired_stability_artifact(**invalid)
    assert artifact["classification"] == "PAIRED_PROTOCOL_UNSTABLE"
    assert len(artifact["epochs"]) == 8
    assert first_key in artifact["raw_input_files"]


def test_artifact_forces_established_false(valid_bundle_inputs):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_claim_boundary")
    artifact = diagnostic.build_paired_stability_artifact(
        **valid_bundle_inputs
    )
    artifact["process_boundary_effect_established"] = True
    with pytest.raises(ValueError, match="must remain false"):
        diagnostic.validate_paired_stability_artifact(artifact)
```

Add tests that schedule text and digest are bound, source hashes cover all declared files, raw inputs have safe relative paths and SHA-256 digests, admission failure rows carry every required field, block/epoch/repeat counts are exact, and classification/boolean pairs are valid.

- [ ] **Step 2: Write CLI path and no-overwrite tests**

```python
def test_cli_requires_bundle_root_repo_root_and_out():
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_cli")
    with pytest.raises(SystemExit):
        diagnostic.parse_args([])
    args = diagnostic.parse_args(
        [
            "--bundle-root",
            "/tmp/bundle",
            "--repo-root",
            str(ROOT),
            "--out",
            "/tmp/paired-stability.json",
        ]
    )
    assert args.bundle_root == "/tmp/bundle"
    assert args.repo_root == str(ROOT)
    assert args.out == "/tmp/paired-stability.json"


def test_cli_refuses_existing_output(valid_bundle_directory, tmp_path):
    diagnostic = load_module(DIAGNOSTIC_PATH, "paired_cli_overwrite")
    out = tmp_path / "paired-stability.json"
    out.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.build_from_paths(
            bundle_root=valid_bundle_directory,
            repo_root=ROOT,
            out=out,
        )
```

The CLI fixture must use the exact intended directory layout:

```text
schedule.txt
command.txt
source.sha256
blocks/block-0-ab/a-first/
blocks/block-0-ab/b-second/
blocks/block-1-ba/b-first/
blocks/block-1-ba/a-second/
blocks/block-2-ba/b-first/
blocks/block-2-ba/a-second/
blocks/block-3-ab/a-first/
blocks/block-3-ab/b-second/
```

- [ ] **Step 3: Run artifact and CLI tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  -k 'artifact or canonical or claim or cli or overwrite or raw_input'
```

Expected: failures report missing artifact and CLI functions.

- [ ] **Step 4: Implement canonical artifact validation**

Use deterministic dictionaries and lists in fixed schedule order. The top-level claim fields must be assigned, not copied from inputs:

```python
CLAIM_BOUNDARY = (
    "one source-bound bundle can establish only internal admission and "
    "a balanced paired candidate or no-candidate result; it cannot "
    "establish a host or GPU cause, a production regression, a performance "
    "improvement, generalization, Phase-1 completion, or promotion readiness"
)


def build_paired_stability_artifact(
    *,
    metadata: dict,
    epoch_raw_inputs: dict[str, dict],
    input_files: dict[str, dict],
    source_files: dict[str, str],
) -> dict:
    expected_keys = tuple(
        identity.key for identity in expected_epoch_identities()
    )
    if tuple(epoch_raw_inputs) != expected_keys:
        raise ValueError("raw epoch inventory or order is invalid")
    epochs = {}
    for identity in expected_epoch_identities():
        epochs[identity.key] = build_epoch_admission(
            identity,
            epoch_raw_inputs[identity.key],
        )
    validate_epoch_workload_identity(
        {
            key: epoch_raw_inputs[key]["worker"]
            for key in expected_keys
        }
    )
    bundle_admission = build_bundle_admission(epochs)
    effects = (
        compute_paired_effects(epochs)
        if bundle_admission["passed"]
        else empty_effects()
    )
    order_check = (
        order_effect_check(effects)
        if bundle_admission["passed"]
        else {"passed": False, "checks": {}, "reasons": ["not admitted"]}
    )
    classification, candidate, reasons = classify_paired_stability(
        bundle_admission,
        effects,
        order_check,
    )
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "classification": classification,
        "classification_reasons": reasons,
        "candidate_process_boundary_effect": candidate,
        "process_boundary_effect_established": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "run_tag": metadata["run_tag"],
        "bundle_start_utc": metadata["bundle_start_utc"],
        "bundle_finish_utc": metadata["bundle_finish_utc"],
        "remote_host": metadata["remote_host"],
        "remote_base": metadata["remote_base"],
        "schedule": ["".join(block) for block in BLOCK_SCHEDULE],
        "schedule_sha256": SCHEDULE_SHA256,
        "configuration": copy.deepcopy(metadata["configuration"]),
        "source_files": copy.deepcopy(source_files),
        "source_sha256": digest_mapping(source_files),
        "model_identity": copy.deepcopy(metadata["model_identity"]),
        "prompt_identity": copy.deepcopy(metadata["prompt_identity"]),
        "command_identity": copy.deepcopy(metadata["command_identity"]),
        "blocks": build_block_view(epochs),
        "epochs": epochs,
        "measured_repeat_count_total": bundle_admission[
            "measured_repeat_count_total"
        ],
        "epoch_admission": {
            key: {
                "passed": epochs[key]["passed"],
                "failures": epochs[key]["failures"],
            }
            for key in expected_keys
        },
        "bundle_admission": bundle_admission,
        "primary_stationarity": collect_stationarity(epochs),
        "coverage": collect_field(epochs, "coverage"),
        "gpu_invariants": collect_field(epochs, "gpu_invariants"),
        "process_invariants": collect_field(
            epochs,
            "process_invariants",
        ),
        "exact_parity": collect_field(epochs, "exact_parity"),
        "block_local_position_effects": position_effect_view(effects),
        "block_local_label_effects": label_effect_view(effects),
        "aggregate_position_effects": effects[
            "aggregate_position_effects"
        ],
        "aggregate_label_effects": effects["aggregate_label_effects"],
        "ab_position_effects": effects["ab_position_effects"],
        "ba_position_effects": effects["ba_position_effects"],
        "sequence_interactions": effects["sequence_interactions"],
        "diagnostic_effects": effects["diagnostic_effects"],
        "order_effect_check": order_check,
        "raw_input_files": copy.deepcopy(input_files),
        "raw_input_sha256": digest_input_inventory(input_files),
    }
    return validate_paired_stability_artifact(artifact)
```

Implement helper functions named above as deterministic transformations. `empty_effects()` must emit the full effect schema with empty rows and must not invent numeric effects for an inadmissible bundle.

- [ ] **Step 5: Implement safe filesystem assembly and CLI**

Use safe relative-path validation copied in behavior, not imported from, the learned A/A verifier:

```python
def _safe_relative_path(root: Path, path: Path, *, name: str) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{name} must be below the bundle root") from error
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} must be a safe relative path")
    return pure.as_posix()


def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    artifact = build_from_paths(
        bundle_root=Path(args.bundle_root),
        repo_root=Path(args.repo_root),
        out=Path(args.out),
    )
    print(
        json.dumps(
            {
                "classification": artifact["classification"],
                "candidate_process_boundary_effect": artifact[
                    "candidate_process_boundary_effect"
                ],
                "process_boundary_effect_established": False,
            },
            sort_keys=True,
        )
    )
    return 0
```

`build_from_paths` must load all eight epoch directories, verify `schedule.txt` byte-for-byte against `SCHEDULE_TEXT`, load command/source/model/prompt metadata, hash every authoritative raw input, call `build_paired_stability_artifact`, and use `_write_json_exclusive`.

- [ ] **Step 6: Run artifact and CLI tests and confirm GREEN**

Run the Step 3 command.

Expected: all selected tests pass.

- [ ] **Step 7: Record the task checkpoint without staging**

Run:

```bash
python3 -m py_compile \
  tools/autoregressive_draft_paired_stability_diagnostic.py
git diff --check -- \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: both commands succeed without output.

---

### Task 5: Independent Filesystem-Bound Verifier

**Files:**
- Create: `tools/verify_autoregressive_draft_paired_stability_diagnostic.py`
- Modify: `tools/test_autoregressive_draft_paired_stability_diagnostic.py`

**Interfaces:**
- Consumes: canonical artifact and hash-bound raw files.
- Produces: `verify_paired_stability_diagnostic(*, artifact_path: Path, repo_root: Path, manifest_path: Path | None = None) -> dict`.
- Produces CLI arguments `--artifact`, `--repo-root`, `--manifest`, and `--receipt`.

- [ ] **Step 1: Write tamper and recomputation tests**

```python
def test_verifier_recomputes_artifact_from_raw_inputs(
    valid_bundle_directory,
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_verifier")
    receipt = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
    )
    assert receipt["verified"] is True
    assert receipt["classification"] in {
        "PAIRED_PROTOCOL_UNSTABLE",
        "NO_REPRODUCIBLE_PROCESS_EFFECT",
        "CANDIDATE_PROCESS_BOUNDARY_EFFECT",
    }
    assert receipt["process_boundary_effect_established"] is False


def test_verifier_rejects_schedule_tamper(
    valid_bundle_directory,
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_schedule_tamper")
    schedule = valid_bundle_directory / "schedule.txt"
    schedule.write_text("AB\nBA\nAB\nBA\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schedule"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_verifier_rejects_raw_input_tamper(
    valid_bundle_directory,
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_raw_tamper")
    worker = (
        valid_bundle_directory
        / "blocks"
        / "block-0-ab"
        / "a-first"
        / "worker.json"
    )
    worker.write_text(worker.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="raw input hash mismatch"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )


def test_verifier_rejects_derived_classification_tamper(
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_derived_tamper")
    artifact = json.loads(canonical_artifact_path.read_text())
    artifact["classification"] = "CANDIDATE_PROCESS_BOUNDARY_EFFECT"
    canonical_artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="canonical artifact mismatch"):
        verifier.verify_paired_stability_diagnostic(
            artifact_path=canonical_artifact_path,
            repo_root=ROOT,
        )
```

Add rejection tests for source hash tamper, unsafe `..` paths, absolute paths, missing raw files, duplicated/reordered identities, fewer/more than four blocks, fewer/more than eight epochs, fewer/more than forty repeats, non-finite values, incorrect median/MAD/drift, incorrect log ratios, incorrect aggregate effects, incorrect sequence interaction, wrong threshold equality, wrong classification precedence, and `process_boundary_effect_established=true`.

- [ ] **Step 2: Write receipt-equivalence and manifest tests**

```python
def test_remote_and_local_receipts_differ_only_in_permitted_metadata(
    canonical_artifact_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_receipt_equivalence")
    remote = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
    )
    local = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
    )
    for receipt in (remote, local):
        receipt.pop("verified_at_utc", None)
        receipt.pop("verification_location", None)
    assert remote == local


def test_manifest_must_cover_every_bundle_file(
    canonical_artifact_path,
    valid_manifest_path,
):
    verifier = load_module(VERIFIER_PATH, "paired_manifest")
    receipt = verifier.verify_paired_stability_diagnostic(
        artifact_path=canonical_artifact_path,
        repo_root=ROOT,
        manifest_path=valid_manifest_path,
    )
    assert receipt["manifest_verified"] is True
    assert receipt["manifest_sha256"]
```

Add a negative test that creates one unlisted authoritative input and requires
manifest verification to fail. Verifier receipts and their stdout/stderr logs
are detached attestations: they are the only permitted files outside the
manifest because each receipt binds the manifest hash and therefore cannot be
included in the manifest it signs.

- [ ] **Step 3: Run verifier tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  -k 'verifier or tamper or receipt or manifest'
```

Expected: collection fails because the verifier file does not exist.

- [ ] **Step 4: Implement safe bindings and full recomputation**

Create:

```python
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_paired_stability_diagnostic import (
    build_paired_stability_artifact,
    load_bound_bundle_inputs,
    validate_paired_stability_artifact,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_bound_path(
    root: Path,
    relative_path: object,
    *,
    name: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{name} path must be a relative path")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} path must be a safe relative path")
    path = root / Path(*pure.parts)
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{name} path escapes the artifact root") from error
    if not path.is_file():
        raise ValueError(f"bound file is missing: {name}")
    return path


def verify_paired_stability_diagnostic(
    *,
    artifact_path: Path,
    repo_root: Path,
    manifest_path: Path | None = None,
) -> dict:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    validate_paired_stability_artifact(artifact)
    artifact_root = artifact_path.parent
    verified_inputs = verify_raw_input_bindings(artifact, artifact_root)
    verified_sources = verify_source_bindings(artifact, repo_root)
    rebuilt_inputs = load_bound_bundle_inputs(
        artifact_root=artifact_root,
        artifact=artifact,
        verified_inputs=verified_inputs,
    )
    rebuilt = build_paired_stability_artifact(**rebuilt_inputs)
    if rebuilt != artifact:
        raise ValueError("canonical artifact mismatch after recomputation")
    manifest = (
        verify_manifest(manifest_path, artifact_root)
        if manifest_path is not None
        else {
            "verified": False,
            "sha256": None,
            "file_count": 0,
        }
    )
    return {
        "schema_version": 1,
        "verified": True,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_location": "unspecified",
        "artifact_path": artifact_path.name,
        "artifact_sha256": _sha256(artifact_path),
        "classification": artifact["classification"],
        "candidate_process_boundary_effect": artifact[
            "candidate_process_boundary_effect"
        ],
        "process_boundary_effect_established": False,
        "source_file_count": verified_sources,
        "source_inventory_sha256": _canonical_json_sha256(
            artifact["source_files"]
        ),
        "raw_input_file_count": len(verified_inputs),
        "raw_input_inventory_sha256": _canonical_json_sha256(
            artifact["raw_input_sha256"]
        ),
        "manifest_verified": manifest["verified"],
        "manifest_sha256": manifest["sha256"],
        "manifest_file_count": manifest["file_count"],
        "verifier_source_sha256": _sha256(Path(__file__)),
    }
```

Implement `verify_raw_input_bindings`, `verify_source_bindings`, and
`verify_manifest` without trusting any derived field. `verify_manifest` must
parse every line as `<sha256><two spaces><relative path>`, reject unsafe or
duplicate paths, verify every digest, and require the inventory to equal all
authoritative files below the bundle root. The only exclusions are:

```python
DETACHED_ATTESTATION_PATHS = {
    "manifest.sha256",
    "verify.paired-stability.remote.json",
    "verify.paired-stability.remote.log",
    "verify.paired-stability.local.json",
    "verify.paired-stability.local.log",
}
```

Any other unlisted file fails manifest completeness.

- [ ] **Step 5: Implement exclusive receipt output and CLI**

```python
def _write_json_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--receipt")
    parser.add_argument(
        "--verification-location",
        choices=("remote", "local"),
        default="local",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    receipt = verify_paired_stability_diagnostic(
        artifact_path=Path(args.artifact),
        repo_root=Path(args.repo_root),
        manifest_path=(
            None if args.manifest is None else Path(args.manifest)
        ),
    )
    receipt["verification_location"] = args.verification_location
    if args.receipt:
        _write_json_exclusive(Path(args.receipt), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0
```

- [ ] **Step 6: Run verifier tests and confirm GREEN**

Run the Step 3 command.

Expected: all selected tests pass.

- [ ] **Step 7: Record the task checkpoint without staging**

Run:

```bash
python3 -m py_compile \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py
git diff --check -- \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: both commands succeed without output.

---

### Task 6: Runner Source Contract and Safety Tests

**Files:**
- Create: `tools/run_autoregressive_draft_paired_stability_remote.sh`
- Modify: `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Consumes: diagnostic and verifier CLIs from Tasks 4-5.
- Produces: executable remote runner with no remote execution during this task.

- [ ] **Step 1: Add runner path and executable contract**

Add:

```python
PAIRED_STABILITY_RUNNER_PATH = (
    ROOT
    / "tools"
    / "run_autoregressive_draft_paired_stability_remote.sh"
)


def test_paired_stability_runner_is_executable():
    assert PAIRED_STABILITY_RUNNER_PATH.exists()
    assert os.access(PAIRED_STABILITY_RUNNER_PATH, os.X_OK)
```

- [ ] **Step 2: Add fixed-schedule, storage, and safety contracts**

```python
def test_paired_stability_runner_owns_fixed_protocol():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        'SCHEDULE_TEXT=$\\'AB\\nBA\\nBA\\nAB\\n\\'',
        "EXPECTED_BLOCKS=4",
        "EXPECTED_EPOCHS=8",
        "MEASURED_RUNS_PER_EPOCH=5",
        "MEASURED_RUNS_TOTAL=40",
        "--warmup-runs 2",
        "--measured-runs 1",
        "--measured-runs 5",
        "sitian@10.232.195.203",
        "/data00/home/sitian/miniconda3/envs/py311/bin/python",
        "/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815",
        "target-qwen3-1.7b",
        "GPU_INDICES=3,4,6,7",
        "PROTECTED_GPU7_PID=703088",
        "paired-stability.json",
        "verify.paired-stability.remote.json",
        "verify.paired-stability.local.json",
        "manifest.sha256",
    ):
        assert expected in script
    for forbidden in (
        "--schedule",
        "SCHEDULE_TEXT=${",
        "torch.cuda.synchronize",
        "kill 703088",
        "pkill",
        "/data00/run",
        "/data00/tmp",
        "/data00/experiments",
    ):
        assert forbidden not in script


def test_paired_stability_runner_refuses_overwrite_and_replication():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    assert "refusing to overwrite local run" in script
    assert "refusing to overwrite remote run" in script
    assert "--replicate" not in script
    assert "replication bundle" not in script
```

- [ ] **Step 3: Add dependency-closure and ownership contracts**

Require the source archive to include:

```python
def test_paired_stability_runner_packages_complete_dependency_closure():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "tinyvllm",
        "tools/autoregressive_draft_performance_worker.py",
        "tools/autoregressive_draft_performance_gate.py",
        "tools/autoregressive_draft_host_sampler.py",
        "tools/autoregressive_draft_host_semantic_diagnostic.py",
        "tools/autoregressive_draft_instability_telemetry.py",
        "tools/autoregressive_draft_paired_stability_diagnostic.py",
        "tools/verify_autoregressive_draft_paired_stability_diagnostic.py",
        "tools/test_autoregressive_draft_executor.py",
        "tools/test_autoregressive_draft_performance_gate.py",
        "tools/test_autoregressive_draft_host_sampler.py",
        "tools/test_autoregressive_draft_host_semantic_diagnostic.py",
        "tools/test_autoregressive_draft_instability_telemetry.py",
        "tools/test_autoregressive_draft_paired_stability_diagnostic.py",
        "tools/run_autoregressive_draft_paired_stability_remote.sh",
    ):
        assert expected in script


def test_paired_stability_runner_reaps_only_owned_processes():
    script = PAIRED_STABILITY_RUNNER_PATH.read_text(encoding="utf-8")
    for expected in (
        "worker_pid",
        "sampler_pids",
        "stop_owned_processes",
        "wait \"${worker_pid}\"",
        "trap stop_owned_processes EXIT TERM INT",
        "runner_owned_pids_remaining",
    ):
        assert expected in script
```

- [ ] **Step 4: Run runner contract tests and confirm RED**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k 'paired_stability'
```

Expected: failures report the missing runner or missing contract strings.

- [ ] **Step 5: Create a non-executing runner skeleton**

Create the file with argument parsing before path derivation and fixed constants:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="sitian@10.232.195.203"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-}"
REMOTE_PYTHON="/data00/home/sitian/miniconda3/envs/py311/bin/python"
REMOTE_BASE="/dev/shm/sitian/tllm-qwen35-target-qwen3-draft-20260815"
REMOTE_PACKAGE_ROOT="${REMOTE_BASE}/run_packages"
TARGET_MODEL="${REMOTE_BASE}/target-qwen3-1.7b"
DRAFT_MODEL="${REMOTE_BASE}/draft"
GPU_INDICES=3,4,6,7
PROTECTED_GPU7_PID=703088
DIST_PORT="${DIST_PORT:-29671}"
MASTER_PORT="${MASTER_PORT:-29771}"
HARD_TIMEOUT_SECONDS="${HARD_TIMEOUT_SECONDS:-14400}"
SCHEDULE_TEXT=$'AB\nBA\nBA\nAB\n'
EXPECTED_BLOCKS=4
EXPECTED_EPOCHS=8
MEASURED_RUNS_PER_EPOCH=5
MEASURED_RUNS_TOTAL=40
RUN_TAG="${RUN_TAG:-tp4-qwen3-b4-paired-stability-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_RUN="${LOCAL_RUN:-}"
REMOTE_RUN="${REMOTE_RUN:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote-host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --ssh-control-path)
      SSH_CONTROL_PATH="$2"
      shift 2
      ;;
    --dist-port)
      DIST_PORT="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --hard-timeout-seconds)
      HARD_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --local-run)
      LOCAL_RUN="$2"
      shift 2
      ;;
    --remote-run)
      REMOTE_RUN="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

LOCAL_RUN="${LOCAL_RUN:-${REPO_ROOT}/experiments/autoregressive_draft/${RUN_TAG}}"
REMOTE_RUN="${REMOTE_RUN:-${REMOTE_BASE}/run/${RUN_TAG}}"
```

Do not add CLI overrides for the fixed Python, remote base, target/draft models, GPU indices, protected PID, schedule, warmups, or repeat counts.

- [ ] **Step 6: Mark the runner executable and rerun the contract tests**

Run:

```bash
chmod +x tools/run_autoregressive_draft_paired_stability_remote.sh
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k 'paired_stability'
```

Expected: executable-mode test passes; content tests remain RED until Task 7.

- [ ] **Step 7: Record the task checkpoint without staging**

Run:

```bash
bash -n tools/run_autoregressive_draft_paired_stability_remote.sh
git diff --check -- \
  tools/run_autoregressive_draft_paired_stability_remote.sh \
  tools/test_autoregressive_draft_instability_telemetry.py
```

Expected: both commands succeed without output.

---

### Task 7: Complete Safe Runner Orchestration and Partial-Evidence Preservation

**Files:**
- Modify: `tools/run_autoregressive_draft_paired_stability_remote.sh`
- Modify: `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Consumes: the fixed protocol and CLIs from Tasks 1-6.
- Produces: a runner that is locally syntax- and source-contract-valid but is not executed remotely by this plan.

- [ ] **Step 1: Add local/remote no-overwrite and source packaging**

After path derivation, add:

```bash
if [[ -e "${LOCAL_RUN}" ]]; then
  printf 'refusing to overwrite local run: %s\n' "${LOCAL_RUN}" >&2
  exit 2
fi
mkdir -p "${LOCAL_RUN}"

SOURCE_PATHS=(
  tinyvllm
  tools/autoregressive_draft_performance_worker.py
  tools/autoregressive_draft_performance_gate.py
  tools/autoregressive_draft_host_sampler.py
  tools/autoregressive_draft_host_semantic_diagnostic.py
  tools/autoregressive_draft_instability_telemetry.py
  tools/autoregressive_draft_paired_stability_diagnostic.py
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py
  tools/test_autoregressive_draft_executor.py
  tools/test_autoregressive_draft_performance_gate.py
  tools/test_autoregressive_draft_host_sampler.py
  tools/test_autoregressive_draft_host_semantic_diagnostic.py
  tools/test_autoregressive_draft_instability_telemetry.py
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
  tools/run_autoregressive_draft_paired_stability_remote.sh
)

for source_path in "${SOURCE_PATHS[@]}"; do
  if [[ ! -e "${REPO_ROOT}/${source_path}" ]]; then
    printf 'missing source path: %s\n' "${source_path}" >&2
    exit 2
  fi
done

printf '%s' "${SCHEDULE_TEXT}" >"${LOCAL_RUN}/schedule.txt"
(
  cd "${REPO_ROOT}"
  COPYFILE_DISABLE=1 tar --no-xattrs -cf "${LOCAL_RUN}/source.tar" \
    "${SOURCE_PATHS[@]}"
)
```

Create SSH options with `BatchMode=yes`, `ConnectTimeout=20`, `ControlMaster=no`, `GSSAPIAuthentication=yes`, and optional existing `ControlPath`. The remote mkdir command must first reject an existing `REMOTE_RUN`, then create only paths below it.

- [ ] **Step 2: Add preflight with explicit failure receipts**

Initialize every status file to `125`, including eight prime exit codes, eight worker exit codes, diagnostic, remote verifier, campaign, and safety-stop receipts. Run:

```bash
PYTHONPATH="${remote_package_root}:${remote_source}" \
"${remote_python}" -m py_compile \
  tools/autoregressive_draft_performance_worker.py \
  tools/autoregressive_draft_host_sampler.py \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py

bash -n tools/run_autoregressive_draft_paired_stability_remote.sh

PYTHONPATH="${remote_package_root}:${remote_source}" \
"${remote_python}" -m pytest -q \
  tools/test_autoregressive_draft_executor.py \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Record the exact preflight exit code. Missing dependencies are failures and must not be rewritten as success.

- [ ] **Step 3: Add fixed epoch directory construction**

Materialize:

```bash
epoch_rows=(
  "0 AB A first 0 block-0-ab/a-first"
  "0 AB B second 1 block-0-ab/b-second"
  "1 BA B first 2 block-1-ba/b-first"
  "1 BA A second 3 block-1-ba/a-second"
  "2 BA B first 4 block-2-ba/b-first"
  "2 BA A second 5 block-2-ba/a-second"
  "3 AB A first 6 block-3-ab/a-first"
  "3 AB B second 7 block-3-ab/b-second"
)

for row in "${epoch_rows[@]}"; do
  read -r block_index order label position epoch_index relative <<<"${row}"
  epoch_dir="${remote_artifacts}/blocks/${relative}"
  mkdir -p "${epoch_dir}"
  cat >"${epoch_dir}/identity.json" <<EOF
{
  "block_index": ${block_index},
  "order": "${order}",
  "label": "${label}",
  "position": "${position}",
  "epoch_index": ${epoch_index}
}
EOF
done
```

The shell must not construct this list from an external schedule argument.

- [ ] **Step 4: Add invariant snapshots and safety-stop checks**

Before and after every measured epoch, record:

```bash
nvidia-smi -L >"${epoch_dir}/gpu.before.txt"
nvidia-smi >"${epoch_dir}/gpu.full.before.txt"
ps -eo pid,ppid,user,lstart,args >"${epoch_dir}/process.before.txt"
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory \
  --format=csv,noheader >"${epoch_dir}/gpu-process.before.csv"
```

Repeat after the worker. Generate machine-readable `gpu-invariants.json` and `process-invariants.json` using the remote Python. A safety stop is allowed only when expected GPUs disappear, PID `703088` disappears, source hashes change, or available `/dev/shm` storage falls below the declared bounded requirement. The safety-stop record must include executed and unexecuted epoch identities:

```json
{
  "stopped": true,
  "reason_code": "PROTECTED_PROCESS_MISSING",
  "executed_epoch_keys": ["block-0-ab/a-first"],
  "unexecuted_epoch_keys": [
    "block-0-ab/b-second",
    "block-1-ba/b-first",
    "block-1-ba/a-second",
    "block-2-ba/b-first",
    "block-2-ba/a-second",
    "block-3-ab/a-first",
    "block-3-ab/b-second"
  ]
}
```

Do not kill or restart PID `703088` or any unrelated process.

- [ ] **Step 5: Add owned sampler and worker lifecycle**

Use arrays that contain only PIDs created by the current shell:

```bash
sampler_pids=()
worker_pid=""

stop_owned_processes() {
  local pid
  if [[ -n "${worker_pid}" ]] && kill -0 "${worker_pid}" 2>/dev/null; then
    kill "${worker_pid}" 2>/dev/null || true
  fi
  for pid in "${sampler_pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  if [[ -n "${worker_pid}" ]]; then
    wait "${worker_pid}" 2>/dev/null || true
  fi
  for pid in "${sampler_pids[@]:-}"; do
    wait "${pid}" 2>/dev/null || true
  done
  worker_pid=""
  sampler_pids=()
}

trap stop_owned_processes EXIT TERM INT
```

Start GPU, host-semantic, `vmstat`, `mpstat`, and `pidstat` samplers only for the measured worker. Record their PIDs in `sampler-pids.txt`. After reaping, write `runner_owned_pids_remaining` into `process-invariants.json`; any remaining PID is an admission failure.

- [ ] **Step 6: Add prime and measured worker execution**

For each epoch:

```bash
"${python_executable}" \
  tools/autoregressive_draft_performance_worker.py \
    --target-model "${target_model}" \
    --draft-model "${draft_model}" \
    --policy learned \
    --batch-size 4 \
    --warmup-runs 2 \
    --measured-runs 1 \
    --out "${epoch_dir}/prime-worker.json" \
  >"${epoch_dir}/prime.log" 2>&1
```

Exit and reap the prime before snapshots and samplers. Then launch the measured worker:

```bash
"${python_executable}" \
  tools/autoregressive_draft_performance_worker.py \
    --target-model "${target_model}" \
    --draft-model "${draft_model}" \
    --policy learned \
    --batch-size 4 \
    --warmup-runs 2 \
    --measured-runs 5 \
    --out "${epoch_dir}/worker.json" \
  >"${epoch_dir}/worker.log" 2>&1 &
worker_pid="$!"
wait "${worker_pid}"
worker_status=$?
worker_pid=""
```

Preserve the actual exit code. Stop owned samplers after the worker exits. Continue through all precommitted epochs after an ordinary worker/sampler/admission failure; stop only for a recorded safety condition.

- [ ] **Step 7: Add canonical assembly and pre-manifest remote verification**

After the attempt ends, always invoke the diagnostic if the source and bundle layout remain readable:

```bash
"${remote_python}" \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
    --bundle-root "${remote_artifacts}" \
    --repo-root "${remote_source}" \
    --out "${remote_artifacts}/paired-stability.json"
```

Run the remote verifier without a receipt or manifest first. This proves the
canonical artifact is recomputable before the authoritative manifest is
sealed:

```bash
"${remote_python}" \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
    --artifact "${remote_artifacts}/paired-stability.json" \
    --repo-root "${remote_source}" \
    --verification-location remote \
  >"${remote_artifacts}/verify.paired-stability.pre-manifest.log" 2>&1
```

- [ ] **Step 8: Seal the authoritative manifest and create detached receipts**

Build the authoritative manifest remotely after the pre-manifest verification.
Exclude only the manifest itself and detached verifier receipts/logs:

```bash
(
  cd "${remote_artifacts}"
  find . -type f \
    ! -name manifest.sha256 \
    ! -name 'verify.paired-stability.remote.json' \
    ! -name 'verify.paired-stability.remote.log' \
    ! -name 'verify.paired-stability.local.json' \
    ! -name 'verify.paired-stability.local.log' \
    -print0 |
    sort -z |
    xargs -0 shasum -a 256 >manifest.sha256
  shasum -a 256 -c manifest.sha256
)
```

Run the remote verifier again with the sealed manifest and write the detached
remote receipt:

```bash
"${remote_python}" \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
    --artifact "${remote_artifacts}/paired-stability.json" \
    --repo-root "${remote_source}" \
    --manifest "${remote_artifacts}/manifest.sha256" \
    --verification-location remote \
    --receipt "${remote_artifacts}/verify.paired-stability.remote.json" \
  >"${remote_artifacts}/verify.paired-stability.remote.log" 2>&1
```

Download the complete artifact directory. Run the local verifier with the same
manifest and write `verify.paired-stability.local.json`. Normalize away only
`verified_at_utc` and `verification_location`; require the remaining remote and
local receipt structures to be byte-equivalent. Both receipts must include the
canonical artifact hash, manifest hash, verifier source hash, and authoritative
raw-input inventory digest.

The runner exits nonzero if preflight, safety, transfer, diagnostic,
pre-manifest verification, manifest construction, remote verifier,
receipt-equivalence, or local verifier fails. An admissible
`NO_REPRODUCIBLE_PROCESS_EFFECT` or
`CANDIDATE_PROCESS_BOUNDARY_EFFECT` classification is not a shell failure.

- [ ] **Step 9: Run runner contracts and shell syntax locally**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
bash -n tools/run_autoregressive_draft_paired_stability_remote.sh
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py \
  -k 'paired_stability'
```

Expected: shell syntax succeeds and all paired-stability runner contracts pass.

- [ ] **Step 10: Prove the runner was not executed**

Run:

```bash
find experiments/autoregressive_draft \
  -maxdepth 1 \
  -type d \
  -name 'tp4-qwen3-b4-paired-stability-*' \
  -print
```

Expected: no new paired-stability run directory created by this implementation plan.

- [ ] **Step 11: Record the task checkpoint without staging**

Run:

```bash
git diff --check -- \
  tools/run_autoregressive_draft_paired_stability_remote.sh \
  tools/test_autoregressive_draft_instability_telemetry.py
git status --short -- \
  tools/run_autoregressive_draft_paired_stability_remote.sh \
  tools/test_autoregressive_draft_instability_telemetry.py
```

Expected: no whitespace errors and both paths remain unstaged.

---

### Task 8: Full Local Validation and Handoff Update

**Files:**
- Modify: `AGENT_HANDOFF_STATE.md`
- Validate: `tools/autoregressive_draft_paired_stability_diagnostic.py`
- Validate: `tools/verify_autoregressive_draft_paired_stability_diagnostic.py`
- Validate: `tools/test_autoregressive_draft_paired_stability_diagnostic.py`
- Validate: `tools/run_autoregressive_draft_paired_stability_remote.sh`
- Validate: `tools/test_autoregressive_draft_instability_telemetry.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: local implementation receipts and a handoff that still blocks remote execution.

- [ ] **Step 1: Run compilation and shell syntax**

Run:

```bash
cd /Users/bytedance/dev/TinyLLMForge-adaptive-ngram
python3 -m py_compile \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py
bash -n tools/run_autoregressive_draft_paired_stability_remote.sh
```

Expected: all commands exit zero.

- [ ] **Step 2: Run the focused diagnostic and verifier suite**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: all paired-stability diagnostic and verifier tests pass.

- [ ] **Step 3: Run shared runner and telemetry contracts**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_instability_telemetry.py
```

Expected: the full shared telemetry/runner contract file passes, including the existing learned A/A contracts.

- [ ] **Step 4: Run the complete source dependency preflight locally**

Run:

```bash
python3 -m pytest -q \
  tools/test_autoregressive_draft_executor.py \
  tools/test_autoregressive_draft_performance_gate.py \
  tools/test_autoregressive_draft_host_sampler.py \
  tools/test_autoregressive_draft_host_semantic_diagnostic.py \
  tools/test_autoregressive_draft_instability_telemetry.py \
  tools/test_autoregressive_draft_learned_aa_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: all tests pass. Any missing dependency, collection error, or failure is recorded as a local validation failure, not a green result.

- [ ] **Step 5: Run static protocol guards**

Run:

```bash
rg -n \
  'torch\\.cuda\\.synchronize|kill 703088|pkill|/data00/(run|tmp|experiments)' \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
  tools/run_autoregressive_draft_paired_stability_remote.sh
```

Expected: no matches.

Run:

```bash
rg -n \
  'process_boundary_effect_established' \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py
```

Expected: every production assignment is `False`; tests include rejection of `True`.

- [ ] **Step 6: Verify no existing learned A/A semantics changed**

Run:

```bash
git diff --exit-code -- \
  tools/autoregressive_draft_learned_aa_diagnostic.py \
  tools/verify_autoregressive_draft_learned_aa_diagnostic.py \
  tools/run_autoregressive_draft_learned_aa_remote.sh
```

Expected: no diff for the three existing learned A/A files.

- [ ] **Step 7: Verify no paired-stability workload exists**

Run:

```bash
find experiments/autoregressive_draft \
  -maxdepth 1 \
  -type d \
  -name 'tp4-qwen3-b4-paired-stability-*' \
  -print
```

Expected: no new output.

- [ ] **Step 8: Update the handoff with local-only status**

Append a dated section to `AGENT_HANDOFF_STATE.md` with this exact claim boundary:

```markdown
## 2026-08-15 Paired-Stability Implementation Status

- Written design:
  `docs/superpowers/specs/2026-08-15-autoregressive-draft-paired-stability-design.md`
- Implementation plan:
  `docs/superpowers/plans/2026-08-15-autoregressive-draft-paired-stability.md`
- Local implementation files:
  `tools/autoregressive_draft_paired_stability_diagnostic.py`,
  `tools/verify_autoregressive_draft_paired_stability_diagnostic.py`,
  `tools/run_autoregressive_draft_paired_stability_remote.sh`.
- Local validation: record the exact commands and pass counts from Tasks 8.1-8.6.
- Remote paired-stability workload: NOT RUN.
- Replication bundle: NOT AUTHORIZED.
- `candidate_process_boundary_effect`: unavailable until a separately authorized bundle completes.
- `process_boundary_effect_established=false`.
- Phase 1 remains incomplete.
- Next gate: explicit user authorization is required before running
  `tools/run_autoregressive_draft_paired_stability_remote.sh`.
```

Replace the validation sentence with actual command outputs and pass counts. Do not claim remote evidence.

- [ ] **Step 9: Run final formatting and status checks**

Run:

```bash
git diff --check -- \
  AGENT_HANDOFF_STATE.md \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  tools/run_autoregressive_draft_paired_stability_remote.sh \
  tools/test_autoregressive_draft_instability_telemetry.py

git status --short -- \
  AGENT_HANDOFF_STATE.md \
  docs/superpowers/specs/2026-08-15-autoregressive-draft-paired-stability-design.md \
  docs/superpowers/plans/2026-08-15-autoregressive-draft-paired-stability.md \
  tools/autoregressive_draft_paired_stability_diagnostic.py \
  tools/verify_autoregressive_draft_paired_stability_diagnostic.py \
  tools/test_autoregressive_draft_paired_stability_diagnostic.py \
  tools/run_autoregressive_draft_paired_stability_remote.sh \
  tools/test_autoregressive_draft_instability_telemetry.py
```

Expected: no whitespace errors, no staged paths, and only the intended local changes are reported.

## Execution Stop Gate

After Task 8, stop. Do not invoke the remote runner.

The next permitted action requires a new explicit user authorization and must be limited to one source-bound paired-stability bundle. That later authorization must not be interpreted as permission to launch a replication bundle, terminate unrelated GPU processes, write experiment data below `/data00`, or claim Phase-1 completion.
