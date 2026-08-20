#!/usr/bin/env python3
"""Pure source-version paired gate for autoregressive-draft artifacts."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from typing import Iterable


BASELINE_REVISION = "596e724ea87966b2ab3b47cccda08c106f9084bb"

MEASURED_REPEATS_PER_EPOCH = 5
EPOCH_COUNT_PER_SOURCE = 8
REQUESTS_PER_REPEAT = 4
MEASURED_REPEATS_PER_SOURCE = (
    MEASURED_REPEATS_PER_EPOCH * EPOCH_COUNT_PER_SOURCE
)
REQUEST_SAMPLES_PER_SOURCE = (
    MEASURED_REPEATS_PER_SOURCE * REQUESTS_PER_REPEAT
)

CLASSIFICATION_PRECEDENCE = (
    "INCONCLUSIVE_ARTIFACT",
    "NO_GO_CORRECTNESS",
    "INCONCLUSIVE_STATIONARITY",
    "NO_GO_TPOT_P95",
    "NO_GO_TPOT_MEDIAN",
    "NO_GO_TTFT_REGRESSION",
    "NO_GO_THROUGHPUT_REGRESSION",
    "GO_TPOT_TAIL_OPTIMIZATION",
)

TPOT_P95_LIMIT_NS = 105_870_000
TPOT_MEDIAN_LIMIT_NS = 85_660_000
TTFT_REGRESSION_LIMIT = 0.03
THROUGHPUT_REGRESSION_LIMIT = 0.03
RATIO_ROBUST_DISPERSION_LIMIT = 0.10
RATIO_HALF_DRIFT_LIMIT = 0.15

_RECEIPT_LOCATIONS = ("remote", "local")
_VOLATILE_RECEIPT_FIELDS = {
    "artifact_path",
    "verification_location",
    "verified_at_utc",
}
_CORRECTNESS_FIELDS = (
    "proposal_token_rows",
    "proposal_row_lengths",
    "accepted_prefix_counts",
    "accepted_token_rows",
    "transaction_digest",
    "active_transaction_count",
    "acceptance",
)


@dataclass(frozen=True)
class SourcePairIdentity:
    pair_index: int
    cuda_mode: str
    first_source: str
    second_source: str


_SOURCE_PAIR_SCHEDULE = (
    SourcePairIdentity(0, "eager", "baseline", "candidate"),
    SourcePairIdentity(1, "graph", "candidate", "baseline"),
    SourcePairIdentity(2, "graph", "baseline", "candidate"),
    SourcePairIdentity(3, "eager", "candidate", "baseline"),
    SourcePairIdentity(4, "graph", "baseline", "candidate"),
    SourcePairIdentity(5, "eager", "baseline", "candidate"),
    SourcePairIdentity(6, "eager", "candidate", "baseline"),
    SourcePairIdentity(7, "graph", "candidate", "baseline"),
)


def expected_source_pair_schedule() -> tuple[SourcePairIdentity, ...]:
    return _SOURCE_PAIR_SCHEDULE


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _mapping(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _list(value: object, name: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _revision(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git revision")
    return value


def _finite_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    normalized = float(value)
    if positive and normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nearest_rank(values: Iterable[float], percentile: float) -> float:
    normalized = sorted(
        _finite_number(value, "percentile sample")
        for value in values
    )
    if not normalized:
        raise ValueError("percentile requires samples")
    if not math.isfinite(percentile) or not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be finite and in (0, 1]")
    return normalized[math.ceil(len(normalized) * percentile) - 1]


def _median(values: Iterable[float], name: str) -> float:
    normalized = [
        _finite_number(value, f"{name} sample") for value in values
    ]
    if not normalized:
        raise ValueError(f"{name} requires samples")
    return float(statistics.median(normalized))


def classify_source_pair(
    *,
    artifact_complete: bool,
    correctness_passed: bool,
    stationarity_passed: bool,
    candidate_tpot_p95_ns: float,
    candidate_tpot_median_ns: float,
    ttft_regression: float,
    throughput_regression: float,
) -> str:
    if not artifact_complete:
        return "INCONCLUSIVE_ARTIFACT"
    if not correctness_passed:
        return "NO_GO_CORRECTNESS"
    if not stationarity_passed:
        return "INCONCLUSIVE_STATIONARITY"
    if (
        _finite_number(candidate_tpot_p95_ns, "candidate TPOT p95")
        > TPOT_P95_LIMIT_NS
    ):
        return "NO_GO_TPOT_P95"
    if (
        _finite_number(
            candidate_tpot_median_ns,
            "candidate TPOT median",
        )
        > TPOT_MEDIAN_LIMIT_NS
    ):
        return "NO_GO_TPOT_MEDIAN"
    if (
        _finite_number(ttft_regression, "TTFT regression")
        > TTFT_REGRESSION_LIMIT
    ):
        return "NO_GO_TTFT_REGRESSION"
    if (
        _finite_number(
            throughput_regression,
            "throughput regression",
        )
        > THROUGHPUT_REGRESSION_LIMIT
    ):
        return "NO_GO_THROUGHPUT_REGRESSION"
    return "GO_TPOT_TAIL_OPTIMIZATION"


def _normalize_receipts(
    receipts: object,
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    source: str,
) -> tuple[dict, str]:
    rows = _mapping(receipts, f"{source} verifier receipts")
    if tuple(rows) != _RECEIPT_LOCATIONS:
        raise ValueError(
            f"{source} verifier receipts require remote and local entries"
        )
    normalized = []
    for location in _RECEIPT_LOCATIONS:
        receipt = copy.deepcopy(
            _mapping(rows[location], f"{source} {location} receipt")
        )
        if receipt.get("verified") is not True:
            raise ValueError(f"{source} verifier receipt is not verified")
        if receipt.get("manifest_verified") is not True:
            raise ValueError(f"{source} manifest was not verified")
        if receipt.get("artifact_sha256") != artifact_sha256:
            raise ValueError(f"{source} artifact receipt hash mismatch")
        if receipt.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{source} manifest receipt hash mismatch")
        if receipt.get("verification_location") != location:
            raise ValueError(
                f"{source} verifier receipt location mismatch"
            )
        normalized.append({
            key: value
            for key, value in receipt.items()
            if key not in _VOLATILE_RECEIPT_FIELDS
        })
    if canonical_json_bytes(normalized[0]) != canonical_json_bytes(
        normalized[1]
    ):
        raise ValueError(f"{source} verifier receipts disagree")
    return normalized[0], canonical_json_sha256(normalized[0])


def _source_epoch_rows(
    artifact: object,
    *,
    source: str,
) -> tuple[list[dict], dict]:
    row = _mapping(artifact, f"{source} artifact")
    if row.get("schema_version") != 1:
        raise ValueError(f"{source} artifact schema is invalid")
    epochs = _mapping(row.get("epochs"), f"{source} epochs")
    if len(epochs) != EPOCH_COUNT_PER_SOURCE:
        raise ValueError(f"{source} requires exactly eight epochs")
    expected = expected_source_pair_schedule()
    normalized_epochs = []
    source_commit = None
    source_tree_sha256 = None
    gpu_uuids = None
    all_epoch_stationarity = True
    all_identity_correctness = True
    all_timeline_conservation = True
    indexed_epochs = {}
    for epoch_key, raw_epoch in epochs.items():
        epoch = _mapping(raw_epoch, f"{source} epoch {epoch_key}")
        identity = _mapping(
            epoch.get("identity"),
            f"{source} epoch identity",
        )
        epoch_index = identity.get("epoch_index")
        if (
            isinstance(epoch_index, bool)
            or not isinstance(epoch_index, int)
            or epoch_index < 0
            or epoch_index >= EPOCH_COUNT_PER_SOURCE
        ):
            raise ValueError(f"{source} epoch index is invalid")
        if epoch_index in indexed_epochs:
            raise ValueError(f"{source} epoch index is duplicated")
        indexed_epochs[epoch_index] = (epoch_key, epoch, identity)
    expected_indices = {
        expected_row.pair_index for expected_row in expected
    }
    if set(indexed_epochs) != expected_indices:
        raise ValueError(f"{source} epoch indices are incomplete")
    for expected_row in expected:
        epoch_key, epoch, identity = indexed_epochs[
            expected_row.pair_index
        ]
        if (
            identity.get("epoch_index") != expected_row.pair_index
            or identity.get("label") != expected_row.cuda_mode
        ):
            raise ValueError(f"{source} epoch schedule is invalid")
        worker = _mapping(
            epoch.get("worker"),
            f"{source} epoch worker",
        )
        if worker.get("cuda_graph_mode") != expected_row.cuda_mode:
            raise ValueError(f"{source} CUDA mode is invalid")
        commit = _revision(
            worker.get("source_commit"),
            f"{source} source revision",
        )
        tree = _sha256(
            worker.get("source_tree_sha256"),
            f"{source} source tree",
        )
        epoch_gpu_uuids = _list(
            worker.get("gpu_uuids"),
            f"{source} GPU UUIDs",
        )
        if (
            len(epoch_gpu_uuids) != 4
            or len(set(epoch_gpu_uuids)) != 4
            or not all(
                isinstance(value, str) and value
                for value in epoch_gpu_uuids
            )
        ):
            raise ValueError(
                f"{source} epoch must contain exactly four GPU UUIDs"
            )
        if source_commit is None:
            source_commit = commit
            source_tree_sha256 = tree
            gpu_uuids = list(epoch_gpu_uuids)
        elif (
            commit != source_commit
            or tree != source_tree_sha256
            or epoch_gpu_uuids != gpu_uuids
        ):
            raise ValueError(f"{source} identity drifted across epochs")
        measured_runs = _list(
            worker.get("measured_runs"),
            f"{source} measured runs",
        )
        if len(measured_runs) != MEASURED_REPEATS_PER_EPOCH:
            raise ValueError(
                f"{source} epoch requires exactly five measured repeats"
            )
        normalized_epochs.append({
            "key": epoch_key,
            "mode": expected_row.cuda_mode,
            "worker": worker,
            "measured_runs": measured_runs,
        })
        all_epoch_stationarity = (
            all_epoch_stationarity
            and epoch.get("stationarity_passed") is True
        )
        all_identity_correctness = (
            all_identity_correctness
            and epoch.get("identity_correctness_passed") is True
        )
        all_timeline_conservation = (
            all_timeline_conservation
            and epoch.get("timeline_conservation_passed") is True
        )
    assert source_commit is not None
    assert source_tree_sha256 is not None
    assert gpu_uuids is not None
    return normalized_epochs, {
        "commit": source_commit,
        "tree_sha256": source_tree_sha256,
        "gpu_uuids": gpu_uuids,
        "all_epoch_stationarity": all_epoch_stationarity,
        "all_identity_correctness": all_identity_correctness,
        "all_timeline_conservation": all_timeline_conservation,
    }


def _timing_inventory(epochs: list[dict], *, source: str) -> dict:
    tpot_values = []
    ttft_values = []
    throughput_values = []
    epoch_metrics = []
    for epoch_index, epoch in enumerate(epochs):
        epoch_tpot = []
        epoch_throughput = []
        for repeat_index, raw_run in enumerate(epoch["measured_runs"]):
            run = _mapping(
                raw_run,
                f"{source} epoch {epoch_index} repeat {repeat_index}",
            )
            timing = _mapping(
                run.get("timing"),
                f"{source} repeat timing",
            )
            if timing.get("request_count") != REQUESTS_PER_REPEAT:
                raise ValueError(f"{source} request count is invalid")
            total_output_tokens = _positive_integer(
                timing.get("total_output_tokens"),
                f"{source} total output tokens",
            )
            if total_output_tokens != 64:
                raise ValueError(f"{source} output token total is invalid")
            batch_elapsed_ns = _finite_number(
                timing.get("batch_elapsed_ns"),
                f"{source} batch elapsed time",
                positive=True,
            )
            throughput = total_output_tokens * 1_000_000_000 / (
                batch_elapsed_ns
            )
            throughput_values.append(throughput)
            epoch_throughput.append(throughput)
            per_request = _list(
                timing.get("per_request"),
                f"{source} per-request timing",
            )
            if len(per_request) != REQUESTS_PER_REPEAT:
                raise ValueError(f"{source} request timing count is invalid")
            for sequence_id, raw_request in enumerate(per_request):
                request = _mapping(
                    raw_request,
                    f"{source} request timing",
                )
                if request.get("sequence_id") != sequence_id:
                    raise ValueError(f"{source} request order is invalid")
                tpot = _finite_number(
                    request.get("tpot_ns"),
                    f"{source} TPOT",
                    positive=True,
                )
                ttft = _finite_number(
                    request.get("ttft_ns"),
                    f"{source} TTFT",
                    positive=True,
                )
                tpot_values.append(tpot)
                ttft_values.append(ttft)
                epoch_tpot.append(tpot)
        epoch_metrics.append({
            "pair_index": epoch_index,
            "mode": epoch["mode"],
            "tpot_median_ns": _median(
                epoch_tpot,
                f"{source} epoch TPOT",
            ),
            "median_batch_throughput_tokens_per_s": _median(
                epoch_throughput,
                f"{source} epoch throughput",
            ),
        })
    if len(tpot_values) != REQUEST_SAMPLES_PER_SOURCE:
        raise ValueError(f"{source} request sample count is invalid")
    if len(throughput_values) != MEASURED_REPEATS_PER_SOURCE:
        raise ValueError(f"{source} repeat sample count is invalid")
    return {
        "aggregate": {
            "tpot_median_ns": _median(tpot_values, f"{source} TPOT"),
            "tpot_p95_ns": nearest_rank(tpot_values, 0.95),
            "ttft_p95_ns": nearest_rank(ttft_values, 0.95),
            "median_batch_throughput_tokens_per_s": _median(
                throughput_values,
                f"{source} throughput",
            ),
        },
        "epochs": epoch_metrics,
    }


def _semantic_view(epoch: dict, run: dict) -> dict:
    worker = epoch["worker"]
    correctness = _mapping(
        run.get("correctness"),
        "run correctness",
    )
    return {
        "prompt_sha256": worker.get("prompt_sha256"),
        "prompt_rows": worker.get("prompt_rows"),
        "request_order": worker.get("request_order"),
        "outputs": run.get("outputs"),
        **{
            key: correctness.get(key)
            for key in _CORRECTNESS_FIELDS
        },
    }


def _correctness_summary(
    baseline_epochs: list[dict],
    candidate_epochs: list[dict],
    baseline_identity: dict,
    candidate_identity: dict,
) -> dict:
    mismatches = []
    underlying_passed = all(
        (
            baseline_identity["all_identity_correctness"],
            baseline_identity["all_timeline_conservation"],
            candidate_identity["all_identity_correctness"],
            candidate_identity["all_timeline_conservation"],
        )
    )
    for pair_index, (baseline_epoch, candidate_epoch) in enumerate(
        zip(baseline_epochs, candidate_epochs)
    ):
        for repeat_index, (baseline_run, candidate_run) in enumerate(
            zip(
                baseline_epoch["measured_runs"],
                candidate_epoch["measured_runs"],
            )
        ):
            baseline_view = _semantic_view(
                baseline_epoch,
                baseline_run,
            )
            candidate_view = _semantic_view(
                candidate_epoch,
                candidate_run,
            )
            for source, view in (
                ("baseline", baseline_view),
                ("candidate", candidate_view),
            ):
                if view["active_transaction_count"] != 0:
                    mismatches.append({
                        "pair_index": pair_index,
                        "repeat_index": repeat_index,
                        "field": (
                            f"{source}.active_transaction_count"
                        ),
                    })
            if canonical_json_bytes(
                baseline_view
            ) != canonical_json_bytes(candidate_view):
                mismatches.append({
                    "pair_index": pair_index,
                    "repeat_index": repeat_index,
                    "field": "semantic_parity",
                })
            for source, run in (
                ("baseline", baseline_run),
                ("candidate", candidate_run),
            ):
                correctness = _mapping(
                    run.get("correctness"),
                    f"{source} correctness",
                )
                rank_rows = _list(
                    correctness.get("rank_graph_identities"),
                    f"{source} rank identities",
                )
                if [row.get("rank") for row in rank_rows] != [0, 1, 2, 3]:
                    mismatches.append({
                        "pair_index": pair_index,
                        "repeat_index": repeat_index,
                        "field": f"{source}.four_rank_identity",
                    })
    return {
        "underlying_command_timeline_passed": underlying_passed,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": underlying_passed and not mismatches,
    }


def _ratio_stationarity(values: list[float], *, name: str) -> dict:
    normalized = [
        _finite_number(value, f"{name} ratio", positive=True)
        for value in values
    ]
    if len(normalized) != 4:
        raise ValueError(f"{name} stationarity requires four ratios")
    median = _median(normalized, f"{name} ratio")
    if median <= 0:
        raise ValueError(f"{name} stationarity denominator is invalid")
    mad = _median(
        [abs(value - median) for value in normalized],
        f"{name} absolute deviation",
    )
    robust_dispersion = mad / median
    first_half_median = _median(
        normalized[:2],
        f"{name} first half",
    )
    second_half_median = _median(
        normalized[2:],
        f"{name} second half",
    )
    half_drift = abs(second_half_median - first_half_median) / median
    return {
        "values": normalized,
        "median": median,
        "mad": mad,
        "robust_dispersion": robust_dispersion,
        "robust_dispersion_limit": RATIO_ROBUST_DISPERSION_LIMIT,
        "half_drift": half_drift,
        "half_drift_limit": RATIO_HALF_DRIFT_LIMIT,
        "passed": (
            robust_dispersion <= RATIO_ROBUST_DISPERSION_LIMIT
            and half_drift <= RATIO_HALF_DRIFT_LIMIT
        ),
    }


def _stationarity_summary(
    baseline_timing: dict,
    candidate_timing: dict,
    baseline_identity: dict,
    candidate_identity: dict,
) -> dict:
    mode_rows = {}
    for mode in ("eager", "graph"):
        baseline_epochs = [
            row
            for row in baseline_timing["epochs"]
            if row["mode"] == mode
        ]
        candidate_epochs = [
            row
            for row in candidate_timing["epochs"]
            if row["mode"] == mode
        ]
        tpot_ratios = []
        throughput_ratios = []
        for baseline, candidate in zip(
            baseline_epochs,
            candidate_epochs,
        ):
            baseline_tpot = _finite_number(
                baseline["tpot_median_ns"],
                f"{mode} baseline TPOT",
                positive=True,
            )
            baseline_throughput = _finite_number(
                baseline["median_batch_throughput_tokens_per_s"],
                f"{mode} baseline throughput",
                positive=True,
            )
            tpot_ratios.append(
                candidate["tpot_median_ns"] / baseline_tpot
            )
            throughput_ratios.append(
                candidate[
                    "median_batch_throughput_tokens_per_s"
                ]
                / baseline_throughput
            )
        mode_rows[mode] = {
            "tpot_ratio": _ratio_stationarity(
                tpot_ratios,
                name=f"{mode} TPOT",
            ),
            "throughput_ratio": _ratio_stationarity(
                throughput_ratios,
                name=f"{mode} throughput",
            ),
        }
        mode_rows[mode]["passed"] = all(
            row["passed"]
            for key, row in mode_rows[mode].items()
            if key.endswith("_ratio")
        )
    child_epochs_passed = (
        baseline_identity["all_epoch_stationarity"]
        and candidate_identity["all_epoch_stationarity"]
    )
    return {
        "all_sixteen_source_epochs_passed": child_epochs_passed,
        **mode_rows,
        "passed": (
            child_epochs_passed
            and mode_rows["eager"]["passed"]
            and mode_rows["graph"]["passed"]
        ),
    }


def _schedule_payload() -> list[dict]:
    return [
        {
            "pair_index": row.pair_index,
            "cuda_mode": row.cuda_mode,
            "first_source": row.first_source,
            "second_source": row.second_source,
        }
        for row in expected_source_pair_schedule()
    ]


def build_source_pair_artifact(
    *,
    run_tag: str,
    baseline_artifact: dict,
    candidate_artifact: dict,
    baseline_manifest_sha256: str,
    candidate_manifest_sha256: str,
    baseline_verifier_receipts: dict,
    candidate_verifier_receipts: dict,
) -> dict:
    if not isinstance(run_tag, str) or not run_tag:
        raise ValueError("run tag must be non-empty")
    baseline_manifest_sha256 = _sha256(
        baseline_manifest_sha256,
        "baseline manifest digest",
    )
    candidate_manifest_sha256 = _sha256(
        candidate_manifest_sha256,
        "candidate manifest digest",
    )
    baseline_epochs, baseline_identity = _source_epoch_rows(
        baseline_artifact,
        source="baseline",
    )
    candidate_epochs, candidate_identity = _source_epoch_rows(
        candidate_artifact,
        source="candidate",
    )
    if baseline_identity["commit"] != BASELINE_REVISION:
        raise ValueError("baseline revision does not match authority")
    if baseline_identity["gpu_uuids"] != candidate_identity["gpu_uuids"]:
        raise ValueError("source GPU inventories disagree")
    baseline_timing = _timing_inventory(
        baseline_epochs,
        source="baseline",
    )
    candidate_timing = _timing_inventory(
        candidate_epochs,
        source="candidate",
    )
    baseline_artifact_sha256 = canonical_json_sha256(
        baseline_artifact
    )
    candidate_artifact_sha256 = canonical_json_sha256(
        candidate_artifact
    )
    _, baseline_receipt_sha256 = _normalize_receipts(
        baseline_verifier_receipts,
        artifact_sha256=baseline_artifact_sha256,
        manifest_sha256=baseline_manifest_sha256,
        source="baseline",
    )
    _, candidate_receipt_sha256 = _normalize_receipts(
        candidate_verifier_receipts,
        artifact_sha256=candidate_artifact_sha256,
        manifest_sha256=candidate_manifest_sha256,
        source="candidate",
    )
    correctness = _correctness_summary(
        baseline_epochs,
        candidate_epochs,
        baseline_identity,
        candidate_identity,
    )
    stationarity = _stationarity_summary(
        baseline_timing,
        candidate_timing,
        baseline_identity,
        candidate_identity,
    )
    baseline_metrics = baseline_timing["aggregate"]
    candidate_metrics = candidate_timing["aggregate"]
    baseline_ttft = _finite_number(
        baseline_metrics["ttft_p95_ns"],
        "baseline TTFT p95",
        positive=True,
    )
    baseline_throughput = _finite_number(
        baseline_metrics["median_batch_throughput_tokens_per_s"],
        "baseline throughput",
        positive=True,
    )
    regressions = {
        "ttft_p95": (
            candidate_metrics["ttft_p95_ns"] / baseline_ttft - 1.0
        ),
        "throughput": (
            1.0
            - candidate_metrics[
                "median_batch_throughput_tokens_per_s"
            ]
            / baseline_throughput
        ),
    }
    classification = classify_source_pair(
        artifact_complete=True,
        correctness_passed=correctness["passed"],
        stationarity_passed=stationarity["passed"],
        candidate_tpot_p95_ns=candidate_metrics["tpot_p95_ns"],
        candidate_tpot_median_ns=candidate_metrics["tpot_median_ns"],
        ttft_regression=regressions["ttft_p95"],
        throughput_regression=regressions["throughput"],
    )
    schedule = _schedule_payload()
    return {
        "schema_version": 1,
        "run_tag": run_tag,
        "schedule": schedule,
        "schedule_sha256": canonical_json_sha256(schedule),
        "sources": {
            "baseline": {
                "commit": baseline_identity["commit"],
                "tree_sha256": baseline_identity["tree_sha256"],
                "artifact_sha256": baseline_artifact_sha256,
                "manifest_sha256": baseline_manifest_sha256,
                "normalized_receipt_sha256": (
                    baseline_receipt_sha256
                ),
            },
            "candidate": {
                "commit": candidate_identity["commit"],
                "tree_sha256": candidate_identity["tree_sha256"],
                "artifact_sha256": candidate_artifact_sha256,
                "manifest_sha256": candidate_manifest_sha256,
                "normalized_receipt_sha256": (
                    candidate_receipt_sha256
                ),
            },
        },
        "gpu_uuids": baseline_identity["gpu_uuids"],
        "sample_counts": {
            "epochs_per_source": EPOCH_COUNT_PER_SOURCE,
            "measured_repeats_per_source": (
                MEASURED_REPEATS_PER_SOURCE
            ),
            "request_samples_per_source": REQUEST_SAMPLES_PER_SOURCE,
        },
        "correctness": correctness,
        "metrics": {
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        },
        "regressions": regressions,
        "stationarity": stationarity,
        "thresholds": {
            "tpot_p95_limit_ns": TPOT_P95_LIMIT_NS,
            "tpot_median_limit_ns": TPOT_MEDIAN_LIMIT_NS,
            "ttft_regression_limit": TTFT_REGRESSION_LIMIT,
            "throughput_regression_limit": (
                THROUGHPUT_REGRESSION_LIMIT
            ),
            "ratio_robust_dispersion_limit": (
                RATIO_ROBUST_DISPERSION_LIMIT
            ),
            "ratio_half_drift_limit": RATIO_HALF_DRIFT_LIMIT,
        },
        "classification": classification,
        "performance_improvement_established": (
            classification == "GO_TPOT_TAIL_OPTIMIZATION"
        ),
    }


def validate_source_pair_artifact(
    artifact: object,
    *,
    baseline_artifact: dict,
    candidate_artifact: dict,
    baseline_verifier_receipts: dict,
    candidate_verifier_receipts: dict,
) -> dict:
    row = copy.deepcopy(_mapping(artifact, "source-pair artifact"))
    sources = _mapping(row.get("sources"), "source-pair sources")
    baseline = _mapping(
        sources.get("baseline"),
        "baseline source binding",
    )
    candidate = _mapping(
        sources.get("candidate"),
        "candidate source binding",
    )
    expected = build_source_pair_artifact(
        run_tag=row.get("run_tag"),
        baseline_artifact=baseline_artifact,
        candidate_artifact=candidate_artifact,
        baseline_manifest_sha256=baseline.get("manifest_sha256"),
        candidate_manifest_sha256=candidate.get("manifest_sha256"),
        baseline_verifier_receipts=baseline_verifier_receipts,
        candidate_verifier_receipts=candidate_verifier_receipts,
    )
    if canonical_json_bytes(row) != canonical_json_bytes(expected):
        raise ValueError(
            "source-pair artifact derived-field recomputation mismatch"
        )
    return expected
