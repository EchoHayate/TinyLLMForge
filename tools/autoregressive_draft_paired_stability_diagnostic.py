from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_performance_gate import (
    proposal_slot_capacity_for_batch,
    validate_worker_result,
)
from autoregressive_draft_host_semantic_diagnostic import (
    align_repeat_samples,
    derive_repeat_metrics,
)
from autoregressive_draft_instability_telemetry import (
    summarize_gpu_telemetry,
    validate_campaign_intervals,
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
CLAIM_BOUNDARY = (
    "one source-bound bundle can establish only internal admission and "
    "a balanced paired candidate or no-candidate result; it cannot "
    "establish a host or GPU cause, a production regression, a performance "
    "improvement, generalization, Phase-1 completion, or promotion readiness"
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
            "label": (
                None if self.identity is None else self.identity.label
            ),
            "position": (
                None if self.identity is None else self.identity.position
            ),
            "epoch": (
                None if self.identity is None else self.identity.epoch_index
            ),
            "metric": self.metric,
            "observed": copy.deepcopy(self.observed),
            "expected": self.expected,
            "source_path": self.source_path,
        }


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
    fields = (
        "block_index",
        "order",
        "label",
        "position",
        "epoch_index",
    )
    try:
        identity_value = tuple(getattr(identity, field) for field in fields)
    except AttributeError as error:
        raise ValueError(
            "epoch identity is not in the fixed schedule"
        ) from error
    expected_values = {
        tuple(getattr(expected, field) for field in fields)
        for expected in expected_epoch_identities()
    }
    if identity_value not in expected_values:
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
    try:
        return validate_worker_result(
            worker,
            expected_warmup_runs=MEASURED_WARMUP_RUNS,
            expected_measured_runs=expected_measured_runs,
        )
    except ValueError as error:
        raise ValueError(f"{kind} is invalid: {error}") from error


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
    try:
        return _validate_worker(
            worker,
            identity=identity,
            expected_measured_runs=MEASURED_RUNS_PER_EPOCH,
            kind="measured worker",
        )
    except ValueError as error:
        if "measured run count" in str(error):
            raise ValueError(
                "measured worker must contain five measured repeats"
            ) from error
        raise


def _require_mapping(value: object, *, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _identity_from_epoch(epoch: dict) -> EpochIdentity:
    identity = _require_mapping(
        epoch.get("identity"),
        name="epoch identity",
    )
    try:
        return EpochIdentity(
            block_index=identity["block_index"],
            order=identity["order"],
            label=identity["label"],
            position=identity["position"],
            epoch_index=identity["epoch_index"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("epoch identity is invalid") from error


def _requested_output_tokens(worker: dict) -> int:
    values = {
        request["output_tokens"]
        for run in worker["measured_runs"]
        for request in run["timing"]["per_request"]
    }
    if len(values) != 1:
        raise ValueError("requested output lengths are inconsistent")
    return values.pop()


def _run_semantics(worker: dict) -> dict:
    return {
        "outputs": [
            copy.deepcopy(run["outputs"])
            for run in worker["measured_runs"]
        ],
        "proposed_tokens": [
            run["runtime"]["proposed_tokens"]
            for run in worker["measured_runs"]
        ],
        "accepted_draft_tokens": [
            run["runtime"]["accepted_draft_tokens"]
            for run in worker["measured_runs"]
        ],
    }


def _require_equal(
    actual: object,
    expected: object,
    *,
    message: str,
) -> None:
    if actual != expected:
        raise ValueError(message)


def validate_epoch_workload_identity(
    epochs: dict[str, dict],
) -> dict:
    if not isinstance(epochs, dict):
        raise ValueError("epochs must be a mapping")
    identities = expected_epoch_identities()
    expected_keys = [identity.key for identity in identities]
    if list(epochs) != expected_keys:
        raise ValueError("epoch identity inventory or order mismatch")

    normalized_epochs = []
    for identity in identities:
        epoch = _require_mapping(
            epochs.get(identity.key),
            name=f"epoch {identity.key}",
        )
        _require_equal(
            _identity_from_epoch(epoch),
            identity,
            message="epoch identity does not match its fixed schedule key",
        )
        if epoch.get("temperature") != TEMPERATURE:
            raise ValueError("temperature must be zero")
        if epoch.get("max_proposal_tokens") != MAX_PROPOSAL_TOKENS:
            raise ValueError("MAX_PROPOSAL_TOKENS must be four")
        if epoch.get("gpu_indices") != list(GPU_INDICES):
            raise ValueError("GPU indices must be 3,4,6,7")
        if epoch.get("request_order") != list(range(BATCH_SIZE)):
            raise ValueError("request order must be fixed")
        if epoch.get("accepted_prefix_semantics") is not True:
            raise ValueError("accepted-prefix semantics must be enabled")
        capacity = _require_mapping(
            epoch.get("proposal_kv_capacity"),
            name="Proposal-KV capacity",
        )
        expected_capacity = proposal_slot_capacity_for_batch(BATCH_SIZE)
        if capacity.get("allocator") != "direct":
            raise ValueError("Proposal-KV allocator must be direct")
        if capacity.get("slots") != expected_capacity:
            raise ValueError(
                "Proposal-KV capacity is not the workload-derived bound"
            )
        worker = validate_measured_worker(
            epoch.get("worker"),
            identity=identity,
        )
        normalized_epochs.append((epoch, worker))

    first_epoch, first_worker = normalized_epochs[0]
    field_messages = (
        ("target_checkpoint_identifier", "target checkpoint mismatch"),
        ("draft_checkpoint_identifier", "draft checkpoint mismatch"),
        ("tokenizer_identifier", "tokenizer mismatch"),
        ("dtype", "dtype mismatch"),
        ("prompt_rows", "prompt rows mismatch"),
        ("batch_size", "batch size mismatch"),
        ("tensor_parallel_size", "tensor parallel mismatch"),
        ("proposal_kv_allocator", "Proposal-KV allocator mismatch"),
        ("proposal_slot_capacity", "Proposal-KV capacity mismatch"),
    )
    first_semantics = _run_semantics(first_worker)
    requested_output_tokens = _requested_output_tokens(first_worker)
    for epoch, worker in normalized_epochs[1:]:
        for field, message in field_messages:
            _require_equal(
                worker.get(field),
                first_worker.get(field),
                message=message,
            )
        _require_equal(
            epoch["temperature"],
            first_epoch["temperature"],
            message="temperature mismatch",
        )
        _require_equal(
            epoch["request_order"],
            first_epoch["request_order"],
            message="request order mismatch",
        )
        _require_equal(
            epoch["accepted_prefix_semantics"],
            first_epoch["accepted_prefix_semantics"],
            message="accepted-prefix semantics mismatch",
        )
        if _requested_output_tokens(worker) != requested_output_tokens:
            raise ValueError("requested output lengths mismatch")
        semantics = _run_semantics(worker)
        _require_equal(
            semantics["outputs"],
            first_semantics["outputs"],
            message="output token IDs mismatch",
        )
        _require_equal(
            semantics["proposed_tokens"],
            first_semantics["proposed_tokens"],
            message="proposal counts mismatch",
        )
        _require_equal(
            semantics["accepted_draft_tokens"],
            first_semantics["accepted_draft_tokens"],
            message="accepted token counts mismatch",
        )

    return {
        "target_checkpoint_identifier": first_worker[
            "target_checkpoint_identifier"
        ],
        "draft_checkpoint_identifier": first_worker[
            "draft_checkpoint_identifier"
        ],
        "tokenizer_identifier": first_worker["tokenizer_identifier"],
        "dtype": first_worker["dtype"],
        "prompt_rows": copy.deepcopy(first_worker["prompt_rows"]),
        "requested_output_tokens": requested_output_tokens,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "proposal_kv_allocator": first_worker[
            "proposal_kv_allocator"
        ],
        "proposal_slot_capacity": proposal_slot_capacity_for_batch(
            BATCH_SIZE
        ),
        "tensor_parallel_size": first_worker["tensor_parallel_size"],
        "gpu_indices": list(GPU_INDICES),
        "request_order": list(range(BATCH_SIZE)),
        "accepted_prefix_semantics": True,
        "epoch_keys": expected_keys,
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
            raise ValueError(
                "stationarity values must be finite and positive"
            )
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


def _metric_values(worker: dict) -> dict[str, list[float]]:
    values = {
        "e2e_s": [],
        "tpot_s": [],
        "executor_proposal_forward_ms": [],
        "executor_backend_submit_ms": [],
    }
    for run in worker.get("measured_runs", []):
        per_request = run.get("timing", {}).get("per_request", [])
        if per_request:
            values["e2e_s"].append(statistics.median(
                float(row["completion_latency_s"])
                for row in per_request
            ))
            values["tpot_s"].append(statistics.median(
                float(row["tpot_s"]) for row in per_request
            ))
        runtime = run.get("runtime", {})
        timing = runtime.get("draft_executor_timing", {})
        max_rank_ms = timing.get("max_rank_ms", {})
        if "proposal_forward" in max_rank_ms:
            values["executor_proposal_forward_ms"].append(
                float(max_rank_ms["proposal_forward"])
            )
        detail = runtime.get(
            "draft_executor_proposal_detail",
            {},
        )
        detail_max = detail.get("max_rank_ms", {})
        if "backend_submit" in detail_max:
            values["executor_backend_submit_ms"].append(
                float(detail_max["backend_submit"])
            )
    return values


def _source_path(raw: dict, name: str) -> str:
    paths = raw.get("source_paths")
    if isinstance(paths, dict):
        value = paths.get(name)
        if isinstance(value, str):
            return value
    return name


def _append_failure(
    failures: list[AdmissionFailure],
    *,
    code: str,
    identity: EpochIdentity,
    observed: object,
    expected: str,
    source_path: str,
    metric: str | None = None,
) -> None:
    failures.append(AdmissionFailure(
        code=code,
        identity=identity,
        metric=metric,
        observed=observed,
        expected=expected,
        source_path=source_path,
    ))


def _check_boolean_invariant(
    failures: list[AdmissionFailure],
    *,
    raw: dict,
    identity: EpochIdentity,
    container: str,
    field: str,
    code: str,
    expected: bool,
) -> None:
    row = raw.get(container)
    observed = row.get(field) if isinstance(row, dict) else None
    if observed is not expected:
        _append_failure(
            failures,
            code=code,
            identity=identity,
            observed=observed,
            expected=str(expected).lower(),
            source_path=_source_path(raw, container),
        )


def build_epoch_admission(
    identity: EpochIdentity,
    raw: dict,
) -> dict:
    _require_expected_identity(identity)
    if not isinstance(raw, dict):
        raise ValueError("raw epoch must be a mapping")
    failures: list[AdmissionFailure] = []
    prime_worker = raw.get("prime_worker")
    worker = raw.get("worker")
    normalized_worker = None

    try:
        validate_prime_worker(prime_worker, identity=identity)
    except ValueError as error:
        _append_failure(
            failures,
            code="PRIME_WORKER_INVALID",
            identity=identity,
            observed=str(error),
            expected="two warmups and one measured prime run",
            source_path=_source_path(raw, "prime_worker"),
        )

    measured_runs = (
        worker.get("measured_runs", [])
        if isinstance(worker, dict)
        else []
    )
    if len(measured_runs) != MEASURED_RUNS_PER_EPOCH:
        _append_failure(
            failures,
            code="MEASURED_REPEAT_COUNT",
            identity=identity,
            observed=len(measured_runs),
            expected="exactly five measured repeats",
            source_path=_source_path(raw, "worker"),
        )
    repeat_indices = [
        run.get("repeat") for run in measured_runs
        if isinstance(run, dict)
    ]
    if (
        len(repeat_indices) != len(set(repeat_indices))
        or repeat_indices != list(range(len(repeat_indices)))
    ):
        _append_failure(
            failures,
            code="DUPLICATE_REPEAT_INDEX",
            identity=identity,
            observed=repeat_indices,
            expected="unique contiguous repeat indices 0 through 4",
            source_path=_source_path(raw, "worker"),
        )
    try:
        normalized_worker = validate_measured_worker(
            worker,
            identity=identity,
        )
    except ValueError as error:
        _append_failure(
            failures,
            code="MEASURED_WORKER_INVALID",
            identity=identity,
            observed=str(error),
            expected="valid learned measured worker",
            source_path=_source_path(raw, "worker"),
        )

    try:
        validate_campaign_intervals(worker)
    except (TypeError, ValueError) as error:
        _append_failure(
            failures,
            code="NON_MONOTONIC_INTERVALS",
            identity=identity,
            observed=str(error),
            expected="strictly increasing non-overlapping intervals",
            source_path=_source_path(raw, "worker"),
        )

    gpu_summary = None
    try:
        gpu_summary = summarize_gpu_telemetry(
            worker,
            raw.get("gpu_rows"),
            expected_gpu_indices=GPU_INDICES,
            minimum_samples=5,
        )
    except (TypeError, ValueError) as error:
        code = (
            "GPU_UUID_CHANGED"
            if "UUID changed" in str(error)
            else "GPU_TELEMETRY_COVERAGE"
        )
        _append_failure(
            failures,
            code=code,
            identity=identity,
            observed=str(error),
            expected="five valid samples per declared GPU per repeat",
            source_path=_source_path(raw, "gpu_rows"),
        )

    host_summary = None
    try:
        host_alignments = align_repeat_samples(
            worker,
            raw.get("host_rows"),
        )
        host_summary = [
            {
                "repeat": alignment["repeat"],
                "metrics": derive_repeat_metrics(alignment),
            }
            for alignment in host_alignments
        ]
    except (TypeError, ValueError, KeyError) as error:
        _append_failure(
            failures,
            code="HOST_TELEMETRY_COVERAGE",
            identity=identity,
            observed=str(error),
            expected="ordered host telemetry covering every repeat",
            source_path=_source_path(raw, "host_rows"),
        )

    gpu_invariants = raw.get("gpu_invariants")
    if not isinstance(gpu_invariants, dict):
        gpu_invariants = {}
    _check_boolean_invariant(
        failures,
        raw=raw,
        identity=identity,
        container="gpu_invariants",
        field="telemetry_available",
        code="GPU_TELEMETRY_UNAVAILABLE",
        expected=True,
    )
    _check_boolean_invariant(
        failures,
        raw=raw,
        identity=identity,
        container="gpu_invariants",
        field="throttle_valid",
        code="GPU_THROTTLE_INVALID",
        expected=True,
    )
    _check_boolean_invariant(
        failures,
        raw=raw,
        identity=identity,
        container="gpu_invariants",
        field="clocks_pstate_valid",
        code="GPU_CLOCK_PSTATE_INVALID",
        expected=True,
    )
    for field, code in (
        ("undeclared_gpu_indices", "UNDECLARED_GPU_USAGE"),
        ("xid_events", "GPU_XID"),
        ("reset_events", "GPU_RESET"),
    ):
        observed = gpu_invariants.get(field)
        if observed != []:
            _append_failure(
                failures,
                code=code,
                identity=identity,
                observed=observed,
                expected="empty list",
                source_path=_source_path(raw, "gpu_invariants"),
            )

    process_before = raw.get("process_before")
    process_after = raw.get("process_after")
    if not isinstance(process_before, dict):
        process_before = {}
    if not isinstance(process_after, dict):
        process_after = {}
    if process_after.get("protected_gpu7_pid_present") is not True:
        _append_failure(
            failures,
            code="PROTECTED_PROCESS_MISSING",
            identity=identity,
            observed=process_after.get("protected_gpu7_pid_present"),
            expected="protected GPU-7 PID remains present",
            source_path=_source_path(raw, "process_after"),
        )
    if process_after.get("runner_owned_pids_remaining") != []:
        _append_failure(
            failures,
            code="RUNNER_PROCESS_LEAK",
            identity=identity,
            observed=process_after.get("runner_owned_pids_remaining"),
            expected="no runner-owned process remains",
            source_path=_source_path(raw, "process_after"),
        )
    if (
        process_before.get("unrelated_process_inventory")
        != process_after.get("unrelated_process_inventory")
    ):
        _append_failure(
            failures,
            code="UNRELATED_PROCESS_INVENTORY_CHANGED",
            identity=identity,
            observed={
                "before": process_before.get(
                    "unrelated_process_inventory"
                ),
                "after": process_after.get(
                    "unrelated_process_inventory"
                ),
            },
            expected="unrelated process inventory is unchanged",
            source_path=_source_path(raw, "process_after"),
        )

    for field, code, expected in (
        ("exact_parity", "EXACT_PARITY_FAILED", True),
        (
            "accepted_prefix_semantics",
            "ACCEPTED_PREFIX_MISMATCH",
            True,
        ),
        (
            "prime_excluded_from_measured_statistics",
            "PRIME_ENTERED_MEASURED_STATISTICS",
            True,
        ),
    ):
        if raw.get(field) is not expected:
            _append_failure(
                failures,
                code=code,
                identity=identity,
                observed=raw.get(field),
                expected=str(expected).lower(),
                source_path=_source_path(raw, "worker"),
            )

    expected_proposals = [
        run.get("runtime", {}).get("proposed_tokens")
        for run in measured_runs
        if isinstance(run, dict)
    ]
    if raw.get("proposal_counts") != expected_proposals:
        _append_failure(
            failures,
            code="PROPOSAL_COUNT_MISMATCH",
            identity=identity,
            observed=raw.get("proposal_counts"),
            expected="exact worker proposal counts",
            source_path=_source_path(raw, "worker"),
        )
    proposal_lengths = raw.get("proposal_lengths")
    if (
        not isinstance(proposal_lengths, list)
        or len(proposal_lengths) != len(measured_runs)
        or any(
            value != MAX_PROPOSAL_TOKENS
            for value in proposal_lengths
        )
    ):
        _append_failure(
            failures,
            code="PROPOSAL_LENGTH_MISMATCH",
            identity=identity,
            observed=proposal_lengths,
            expected="one proposal length of four per repeat",
            source_path=_source_path(raw, "worker"),
        )
    if raw.get("total_verified_tokens") != sum(
        value for value in expected_proposals
        if isinstance(value, int)
    ):
        _append_failure(
            failures,
            code="VERIFIED_TOKEN_COUNT_MISMATCH",
            identity=identity,
            observed=raw.get("total_verified_tokens"),
            expected="sum of exact proposal counts",
            source_path=_source_path(raw, "worker"),
        )

    metric_values = _metric_values(
        normalized_worker
        if normalized_worker is not None
        else worker if isinstance(worker, dict) else {}
    )
    metric_medians = {
        metric: statistics.median(values)
        for metric, values in metric_values.items()
        if values
    }
    stationarity = {}
    for metric in PRIMARY_METRICS:
        try:
            row = stationarity_for_values(
                metric,
                metric_values.get(metric),
            )
            stationarity[metric] = row
            if not row["stable"]:
                _append_failure(
                    failures,
                    code="STATIONARITY_FAILED",
                    identity=identity,
                    metric=metric,
                    observed={
                        "robust_dispersion": row[
                            "robust_dispersion"
                        ],
                        "half_drift": row["half_drift"],
                    },
                    expected=(
                        "MAD/median <= 0.10 and half drift <= 0.15"
                    ),
                    source_path=_source_path(raw, "worker"),
                )
        except (TypeError, ValueError) as error:
            _append_failure(
                failures,
                code="METRIC_COVERAGE",
                identity=identity,
                metric=metric,
                observed=str(error),
                expected="exactly five finite positive values",
                source_path=_source_path(raw, "worker"),
            )

    return {
        "identity": {
            "block_index": identity.block_index,
            "order": identity.order,
            "label": identity.label,
            "position": identity.position,
            "epoch_index": identity.epoch_index,
            "key": identity.key,
        },
        "passed": not failures,
        "failures": [failure.to_dict() for failure in failures],
        "prime": {
            "recorded": prime_worker is not None,
            "excluded_from_measured_statistics": (
                raw.get("prime_excluded_from_measured_statistics")
                is True
            ),
        },
        "repeat_count": len(measured_runs),
        "metrics": metric_values,
        "metric_medians": metric_medians,
        "stationarity": stationarity,
        "coverage": {
            "gpu": gpu_summary is not None,
            "host": host_summary is not None,
            "proposal_forward": (
                len(metric_values["executor_proposal_forward_ms"])
                == MEASURED_RUNS_PER_EPOCH
            ),
            "backend_submit": (
                len(metric_values["executor_backend_submit_ms"])
                == MEASURED_RUNS_PER_EPOCH
            ),
        },
        "gpu_invariants": copy.deepcopy(gpu_invariants),
        "process_invariants": {
            "before": copy.deepcopy(process_before),
            "after": copy.deepcopy(process_after),
        },
        "exact_parity": raw.get("exact_parity") is True,
        "accepted_prefix_semantics": (
            raw.get("accepted_prefix_semantics") is True
        ),
        "acceptance_rate": (
            statistics.median([
                float(run["runtime"]["acceptance_rate"])
                for run in measured_runs
                if isinstance(run, dict)
                and isinstance(run.get("runtime"), dict)
                and "acceptance_rate" in run["runtime"]
            ])
            if measured_runs
            else None
        ),
        "proposal_lengths": copy.deepcopy(
            raw.get("proposal_lengths")
        ),
        "total_verified_tokens": raw.get("total_verified_tokens"),
        "telemetry": {
            "gpu": gpu_summary,
            "host": host_summary,
        },
    }


def build_bundle_admission(epochs: dict[str, dict]) -> dict:
    expected_keys = tuple(
        identity.key for identity in expected_epoch_identities()
    )
    if not isinstance(epochs, dict) or tuple(epochs) != expected_keys:
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


def _effect_row(first: float, second: float) -> dict:
    if first <= 0.0 or second <= 0.0:
        raise ValueError("effect medians must be positive")
    log_effect = math.log(second) - math.log(first)
    return {
        "log_effect": log_effect,
        "relative_effect": math.exp(log_effect) - 1.0,
    }


def _aggregate_effect(rows: list[dict], key: str) -> dict:
    if not rows:
        raise ValueError("aggregate effect requires rows")
    log_effect = statistics.median(row[key] for row in rows)
    return {
        "log_effect": log_effect,
        "relative_effect": math.exp(log_effect) - 1.0,
    }


def build_diagnostic_effects(
    epochs: dict[str, dict],
    block_local: dict[str, list[dict]],
) -> dict:
    raw_repeat_ratios = {
        metric: [] for metric in PRIMARY_METRICS + DIAGNOSTIC_METRICS
    }
    chronological_block_trend = {
        metric: [] for metric in PRIMARY_METRICS + DIAGNOSTIC_METRICS
    }
    for block_index in range(len(BLOCK_SCHEDULE)):
        identities = [
            identity
            for identity in expected_epoch_identities()
            if identity.block_index == block_index
        ]
        first_epoch = epochs[identities[0].key]
        second_epoch = epochs[identities[1].key]
        for metric in raw_repeat_ratios:
            first_values = first_epoch["metrics"][metric]
            second_values = second_epoch["metrics"][metric]
            raw_repeat_ratios[metric].append({
                "block_index": block_index,
                "order": identities[0].order,
                "ratios": [
                    second / first
                    for first, second in zip(
                        first_values,
                        second_values,
                    )
                ],
            })
            chronological_block_trend[metric].append({
                "block_index": block_index,
                "position_relative": block_local[metric][block_index][
                    "position_relative"
                ],
            })
    epoch_rows = [
        epochs[identity.key]
        for identity in expected_epoch_identities()
    ]
    return {
        "backend_submit_effects": copy.deepcopy(
            block_local["executor_backend_submit_ms"]
        ),
        "raw_repeat_ratios": raw_repeat_ratios,
        "chronological_block_trend": chronological_block_trend,
        "gpu_summaries": [
            copy.deepcopy(epoch.get("telemetry", {}).get("gpu"))
            for epoch in epoch_rows
        ],
        "host_summaries": [
            copy.deepcopy(epoch.get("telemetry", {}).get("host"))
            for epoch in epoch_rows
        ],
        "acceptance_summaries": [
            epoch.get("acceptance_rate") for epoch in epoch_rows
        ],
        "proposal_length_summaries": [
            copy.deepcopy(epoch.get("proposal_lengths"))
            for epoch in epoch_rows
        ],
        "verified_token_summaries": [
            epoch.get("total_verified_tokens") for epoch in epoch_rows
        ],
    }


def compute_paired_effects(epochs: dict[str, dict]) -> dict:
    expected_keys = tuple(
        identity.key for identity in expected_epoch_identities()
    )
    if not isinstance(epochs, dict) or tuple(epochs) != expected_keys:
        raise ValueError("effect epoch inventory or order is invalid")
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
            block_local[metric].append({
                "block_index": block_index,
                "order": "".join(labels),
                "first_epoch_key": first_identity.key,
                "second_epoch_key": second_identity.key,
                "position_effect": position["log_effect"],
                "position_relative": position["relative_effect"],
                "label_effect": label["log_effect"],
                "label_relative": label["relative_effect"],
            })
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
        aggregate_label[metric] = _aggregate_effect(
            rows,
            "label_effect",
        )
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
        "diagnostic_effects": build_diagnostic_effects(
            epochs,
            block_local,
        ),
    }


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
        "ab_direction_matches": (
            _sign(ab["log_effect"]) == aggregate_sign != 0
        ),
        "ba_direction_matches": (
            _sign(ba["log_effect"]) == aggregate_sign != 0
        ),
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
        "reasons": [
            name for name, passed in checks.items() if not passed
        ],
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
        reasons.append(
            "fewer than three E2E blocks share a direction"
        )
    if (
        abs(aggregate["relative_effect"])
        < EFFECT_MAGNITUDE_THRESHOLD
    ):
        reasons.append(
            "aggregate E2E magnitude is below ten percent"
        )
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


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def digest_mapping(mapping: dict[str, str]) -> str:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("source files must be a non-empty mapping")
    normalized = {}
    for path, digest in mapping.items():
        _validate_safe_relative_text(path, name="source path")
        _validate_sha256(digest, name=f"source hash {path}")
        normalized[path] = digest
    return _canonical_digest(normalized)


def digest_input_inventory(input_files: dict[str, dict]) -> str:
    normalized = _validate_input_inventory(input_files)
    return _canonical_digest(normalized)


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _validate_safe_relative_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a safe relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{name} must be a safe relative path")
    return pure.as_posix()


def _validate_input_inventory(
    input_files: object,
) -> dict[str, dict]:
    if not isinstance(input_files, dict) or not input_files:
        raise ValueError("raw input inventory must be non-empty")
    normalized = {}
    for key, row in input_files.items():
        if not isinstance(key, str) or not key:
            raise ValueError("raw input key is invalid")
        if not isinstance(row, dict):
            raise ValueError("raw input row must be a mapping")
        normalized[key] = {
            "path": _validate_safe_relative_text(
                row.get("path"),
                name="raw input path",
            ),
            "sha256": _validate_sha256(
                row.get("sha256"),
                name=f"raw input hash {key}",
            ),
        }
    return normalized


def empty_effects() -> dict:
    metrics = PRIMARY_METRICS + DIAGNOSTIC_METRICS
    return {
        "block_local": {metric: [] for metric in metrics},
        "aggregate_position_effects": {},
        "aggregate_label_effects": {},
        "ab_position_effects": {},
        "ba_position_effects": {},
        "sequence_interactions": {},
        "diagnostic_effects": {
            "backend_submit_effects": [],
            "raw_repeat_ratios": {metric: [] for metric in metrics},
            "chronological_block_trend": {
                metric: [] for metric in metrics
            },
            "gpu_summaries": [],
            "host_summaries": [],
            "acceptance_summaries": [],
            "proposal_length_summaries": [],
            "verified_token_summaries": [],
        },
    }


def build_block_view(epochs: dict[str, dict]) -> list[dict]:
    blocks = []
    for block_index, labels in enumerate(BLOCK_SCHEDULE):
        keys = [
            identity.key
            for identity in expected_epoch_identities()
            if identity.block_index == block_index
        ]
        blocks.append({
            "block_index": block_index,
            "order": "".join(labels),
            "epoch_keys": keys,
            "passed": all(epochs[key]["passed"] for key in keys),
        })
    return blocks


def collect_stationarity(epochs: dict[str, dict]) -> dict:
    return {
        key: copy.deepcopy(epoch["stationarity"])
        for key, epoch in epochs.items()
    }


def collect_field(epochs: dict[str, dict], field: str) -> dict:
    return {
        key: copy.deepcopy(epoch[field])
        for key, epoch in epochs.items()
    }


def position_effect_view(effects: dict) -> dict:
    return {
        metric: [
            {
                key: copy.deepcopy(row[key])
                for key in (
                    "block_index",
                    "order",
                    "first_epoch_key",
                    "second_epoch_key",
                    "position_effect",
                    "position_relative",
                )
            }
            for row in rows
        ]
        for metric, rows in effects["block_local"].items()
    }


def label_effect_view(effects: dict) -> dict:
    return {
        metric: [
            {
                key: copy.deepcopy(row[key])
                for key in (
                    "block_index",
                    "order",
                    "label_effect",
                    "label_relative",
                )
            }
            for row in rows
        ]
        for metric, rows in effects["block_local"].items()
    }


def _workload_envelopes(
    epoch_raw_inputs: dict[str, dict],
) -> dict[str, dict]:
    envelopes = {}
    for identity in expected_epoch_identities():
        raw = epoch_raw_inputs[identity.key]
        worker = raw["worker"]
        envelopes[identity.key] = {
            "identity": {
                "block_index": identity.block_index,
                "order": identity.order,
                "label": identity.label,
                "position": identity.position,
                "epoch_index": identity.epoch_index,
            },
            "temperature": TEMPERATURE,
            "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
            "gpu_indices": list(GPU_INDICES),
            "request_order": list(range(BATCH_SIZE)),
            "accepted_prefix_semantics": raw.get(
                "accepted_prefix_semantics"
            ),
            "proposal_kv_capacity": {
                "allocator": worker.get("proposal_kv_allocator"),
                "slots": worker.get("proposal_slot_capacity"),
            },
            "worker": worker,
        }
    return envelopes


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
    if (
        not isinstance(epoch_raw_inputs, dict)
        or tuple(epoch_raw_inputs) != expected_keys
    ):
        raise ValueError("raw epoch inventory or order is invalid")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    normalized_inputs = _validate_input_inventory(input_files)
    epochs = {}
    for identity in expected_epoch_identities():
        epochs[identity.key] = build_epoch_admission(
            identity,
            epoch_raw_inputs[identity.key],
        )
    bundle_admission = build_bundle_admission(epochs)
    if bundle_admission["passed"]:
        validate_epoch_workload_identity(
            _workload_envelopes(epoch_raw_inputs)
        )
    effects = (
        compute_paired_effects(epochs)
        if bundle_admission["passed"]
        else empty_effects()
    )
    order_check = (
        order_effect_check(effects)
        if bundle_admission["passed"]
        else {
            "passed": False,
            "checks": {},
            "sequence_interaction": None,
            "reasons": ["not admitted"],
        }
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
        "schedule_text": SCHEDULE_TEXT,
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
                "failures": copy.deepcopy(epochs[key]["failures"]),
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
        "aggregate_label_effects": effects[
            "aggregate_label_effects"
        ],
        "ab_position_effects": effects["ab_position_effects"],
        "ba_position_effects": effects["ba_position_effects"],
        "sequence_interactions": effects["sequence_interactions"],
        "diagnostic_effects": effects["diagnostic_effects"],
        "order_effect_check": order_check,
        "raw_input_files": normalized_inputs,
        "raw_input_sha256": digest_input_inventory(
            normalized_inputs
        ),
    }
    return validate_paired_stability_artifact(artifact)


def validate_paired_stability_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("paired stability artifact must be a mapping")
    required = {
        "schema_version",
        "classification",
        "candidate_process_boundary_effect",
        "process_boundary_effect_established",
        "claim_boundary",
        "schedule",
        "schedule_text",
        "schedule_sha256",
        "source_files",
        "source_sha256",
        "blocks",
        "epochs",
        "measured_repeat_count_total",
        "bundle_admission",
        "raw_input_files",
        "raw_input_sha256",
    }
    missing = sorted(required - set(artifact))
    if missing:
        raise ValueError(f"artifact fields are missing: {missing}")
    if artifact["schema_version"] != SCHEMA_VERSION:
        raise ValueError("artifact schema version is invalid")
    if artifact["process_boundary_effect_established"] is not False:
        raise ValueError(
            "process_boundary_effect_established must remain false"
        )
    classification = artifact["classification"]
    candidate = artifact["candidate_process_boundary_effect"]
    valid_pairs = {
        ("PAIRED_PROTOCOL_UNSTABLE", False),
        ("NO_REPRODUCIBLE_PROCESS_EFFECT", False),
        ("CANDIDATE_PROCESS_BOUNDARY_EFFECT", True),
    }
    if (classification, candidate) not in valid_pairs:
        raise ValueError("classification and candidate boolean mismatch")
    if artifact["schedule"] != [
        "".join(block) for block in BLOCK_SCHEDULE
    ]:
        raise ValueError("artifact schedule is invalid")
    if artifact["schedule_text"] != SCHEDULE_TEXT:
        raise ValueError("artifact schedule text is invalid")
    if artifact["schedule_sha256"] != SCHEDULE_SHA256:
        raise ValueError("artifact schedule digest is invalid")
    if artifact["source_sha256"] != digest_mapping(
        artifact["source_files"]
    ):
        raise ValueError("artifact source digest mismatch")
    if artifact["raw_input_sha256"] != digest_input_inventory(
        artifact["raw_input_files"]
    ):
        raise ValueError("artifact raw input digest mismatch")
    expected_keys = [
        identity.key for identity in expected_epoch_identities()
    ]
    if set(artifact["epochs"]) != set(expected_keys):
        raise ValueError("artifact epoch inventory or order is invalid")
    for identity in expected_epoch_identities():
        row = artifact["epochs"][identity.key]
        embedded = row.get("identity") if isinstance(row, dict) else None
        if not isinstance(embedded, dict) or (
            embedded.get("key") != identity.key
            or embedded.get("block_index") != identity.block_index
            or embedded.get("order") != identity.order
            or embedded.get("label") != identity.label
            or embedded.get("position") != identity.position
            or embedded.get("epoch_index") != identity.epoch_index
        ):
            raise ValueError("artifact epoch identity or order is invalid")
    if len(artifact["blocks"]) != 4:
        raise ValueError("artifact block count must be four")
    if artifact["measured_repeat_count_total"] != sum(
        artifact["epochs"][key]["repeat_count"]
        for key in expected_keys
    ):
        raise ValueError("artifact measured repeat total mismatch")
    if artifact["bundle_admission"]["passed"]:
        if artifact["measured_repeat_count_total"] != MEASURED_RUNS_TOTAL:
            raise ValueError("admitted artifact must contain 40 repeats")
    required_failure_fields = {
        "code",
        "block",
        "label",
        "position",
        "epoch",
        "metric",
        "observed",
        "expected",
        "source_path",
    }
    for key in expected_keys:
        for failure in artifact["epochs"][key]["failures"]:
            if not isinstance(failure, dict) or (
                set(failure) != required_failure_fields
            ):
                raise ValueError("admission failure row is invalid")
    return copy.deepcopy(artifact)


def _safe_relative_path(
    root: Path,
    path: Path,
    *,
    name: str,
) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(
            f"{name} must be below the bundle root"
        ) from error
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} must be a safe relative path")
    return pure.as_posix()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is invalid") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _write_json_exclusive(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_from_paths(
    *,
    bundle_root: Path,
    repo_root: Path,
    out: Path,
) -> dict:
    bundle_root = Path(bundle_root)
    repo_root = Path(repo_root)
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {out}")
    if not bundle_root.is_dir():
        raise ValueError("bundle root must be a directory")
    if not repo_root.is_dir():
        raise ValueError("repo root must be a directory")
    schedule_path = bundle_root / "schedule.txt"
    if schedule_path.read_text(encoding="utf-8") != SCHEDULE_TEXT:
        raise ValueError("schedule.txt does not match the fixed schedule")
    metadata = _load_json(
        bundle_root / "metadata.json",
        name="metadata.json",
    )
    source_files = _load_json(
        bundle_root / "source-files.json",
        name="source-files.json",
    )
    command_path = bundle_root / "command.txt"
    if not command_path.is_file():
        raise ValueError("command.txt is missing")
    epoch_raw_inputs = {}
    input_files = {
        "schedule": {
            "path": _safe_relative_path(
                bundle_root,
                schedule_path,
                name="schedule.txt",
            ),
            "sha256": _sha256_path(schedule_path),
        },
        "command": {
            "path": _safe_relative_path(
                bundle_root,
                command_path,
                name="command.txt",
            ),
            "sha256": _sha256_path(command_path),
        },
        "metadata": {
            "path": "metadata.json",
            "sha256": _sha256_path(bundle_root / "metadata.json"),
        },
        "source_files": {
            "path": "source-files.json",
            "sha256": _sha256_path(bundle_root / "source-files.json"),
        },
    }
    for identity in expected_epoch_identities():
        raw_path = bundle_root / identity.key / "raw.json"
        epoch_raw_inputs[identity.key] = _load_json(
            raw_path,
            name=f"{identity.key}/raw.json",
        )
        input_files[identity.key] = {
            "path": _safe_relative_path(
                bundle_root,
                raw_path,
                name=f"{identity.key} raw input",
            ),
            "sha256": _sha256_path(raw_path),
        }
    artifact = build_paired_stability_artifact(
        metadata=metadata,
        epoch_raw_inputs=epoch_raw_inputs,
        input_files=input_files,
        source_files=source_files,
    )
    _write_json_exclusive(out, artifact)
    return artifact


def load_bound_bundle_inputs(
    *,
    artifact_root: Path,
    artifact: dict,
    verified_inputs: dict[str, Path],
) -> dict:
    del artifact_root
    expected_keys = [
        identity.key for identity in expected_epoch_identities()
    ]
    required = {
        "metadata",
        "source_files",
        "schedule",
        "command",
        *expected_keys,
    }
    if set(verified_inputs) != required:
        raise ValueError("verified raw input inventory is incomplete")
    if (
        verified_inputs["schedule"].read_text(encoding="utf-8")
        != SCHEDULE_TEXT
    ):
        raise ValueError("bound schedule does not match fixed schedule")
    metadata = _load_json(
        verified_inputs["metadata"],
        name="bound metadata",
    )
    source_files = _load_json(
        verified_inputs["source_files"],
        name="bound source files",
    )
    if source_files != artifact["source_files"]:
        raise ValueError("bound source inventory mismatch")
    epoch_raw_inputs = {
        key: _load_json(
            verified_inputs[key],
            name=f"bound epoch {key}",
        )
        for key in expected_keys
    }
    return {
        "metadata": metadata,
        "epoch_raw_inputs": epoch_raw_inputs,
        "input_files": copy.deepcopy(artifact["raw_input_files"]),
        "source_files": source_files,
    }


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
    print(json.dumps(
        {
            "classification": artifact["classification"],
            "candidate_process_boundary_effect": artifact[
                "candidate_process_boundary_effect"
            ],
            "process_boundary_effect_established": False,
        },
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
