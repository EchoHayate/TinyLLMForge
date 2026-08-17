from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
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
    parse_gpu_telemetry,
    summarize_gpu_telemetry,
)
from autoregressive_draft_host_semantic_diagnostic import (
    parse_host_jsonl,
)


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
TEMPERATURE = 0.0
LIMITATIONS = (
    "one bundle can identify only a candidate process-boundary effect",
    "this control does not identify a host or GPU root cause",
    "256-token prompts do not establish 4K, 16K, or 32K performance",
    "direct Proposal-KV allocation does not establish offload benefit",
    "one learned model pair does not establish two model structures",
    "this control does not establish Phase-1 promotion",
)
SOURCE_FILE_PATHS = (
    "tools/autoregressive_draft_performance_gate.py",
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_host_sampler.py",
    "tools/autoregressive_draft_host_semantic_diagnostic.py",
    "tools/autoregressive_draft_instability_telemetry.py",
    "tools/autoregressive_draft_learned_aa_diagnostic.py",
    "tools/verify_autoregressive_draft_learned_aa_diagnostic.py",
    "tools/run_autoregressive_draft_learned_aa_remote.sh",
)
INPUT_FILE_ARGUMENTS = {
    "learned_a_prime_worker": "learned_a_prime_worker",
    "learned_b_prime_worker": "learned_b_prime_worker",
    "learned_a_worker": "learned_a_worker",
    "learned_b_worker": "learned_b_worker",
    "learned_a_gpu_csv": "learned_a_gpu_csv",
    "learned_b_gpu_csv": "learned_b_gpu_csv",
    "learned_a_host_jsonl": "learned_a_host_jsonl",
    "learned_b_host_jsonl": "learned_b_host_jsonl",
    "epoch_order": "epoch_order_file",
    "prime_each_epoch": "prime_each_epoch_file",
}


def _validate_worker(
    worker: object,
    *,
    artifact_identity: str,
    expected_measured_runs: int,
    kind: str,
) -> dict:
    if artifact_identity not in EPOCH_ORDER:
        raise ValueError("invalid learned A/A artifact identity")
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
    artifact_identity: str,
) -> dict:
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


def _require_equal(
    learned_a: dict,
    learned_b: dict,
    *,
    key: str,
    name: str,
) -> None:
    if learned_a.get(key) != learned_b.get(key):
        raise ValueError(f"learned A/A {name} mismatch")


def _output_tokens(worker: dict) -> int:
    values = {
        request["output_tokens"]
        for run in worker["measured_runs"]
        for request in run["timing"]["per_request"]
    }
    if len(values) != 1:
        raise ValueError("learned A/A requested output length is invalid")
    return values.pop()


def validate_workload_identity(
    learned_a: dict,
    learned_b: dict,
) -> dict:
    for key, name in (
        ("target_checkpoint_identifier", "target checkpoint"),
        ("draft_checkpoint_identifier", "draft checkpoint"),
        ("tokenizer_identifier", "tokenizer"),
        ("dtype", "dtype"),
        ("prompt_rows", "prompt rows"),
        ("batch_size", "batch size"),
        ("tensor_parallel_size", "tensor parallel size"),
        ("proposal_kv_allocator", "Proposal-KV allocator"),
        ("proposal_slot_capacity", "Proposal-KV capacity"),
    ):
        _require_equal(
            learned_a,
            learned_b,
            key=key,
            name=name,
        )

    normalized_a = validate_measured_worker(
        learned_a,
        artifact_identity="learned_a",
    )
    normalized_b = validate_measured_worker(
        learned_b,
        artifact_identity="learned_b",
    )
    output_tokens_a = _output_tokens(normalized_a)
    output_tokens_b = _output_tokens(normalized_b)
    if output_tokens_a != output_tokens_b:
        raise ValueError("learned A/A requested output length mismatch")

    expected_capacity = proposal_slot_capacity_for_batch(BATCH_SIZE)
    if normalized_a["proposal_slot_capacity"] != expected_capacity:
        raise ValueError(
            "learned A/A Proposal-KV capacity must match the "
            "workload-derived bound"
        )

    return {
        "target_checkpoint_identifier": normalized_a[
            "target_checkpoint_identifier"
        ],
        "draft_checkpoint_identifier": normalized_a[
            "draft_checkpoint_identifier"
        ],
        "tokenizer_identifier": normalized_a["tokenizer_identifier"],
        "dtype": normalized_a["dtype"],
        "prompt_rows": copy.deepcopy(normalized_a["prompt_rows"]),
        "requested_output_tokens": output_tokens_a,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "max_proposal_tokens": MAX_PROPOSAL_TOKENS,
        "proposal_kv_allocator": normalized_a[
            "proposal_kv_allocator"
        ],
        "proposal_slot_capacity": expected_capacity,
        "tensor_parallel_size": normalized_a[
            "tensor_parallel_size"
        ],
        "gpu_indices": list(GPU_INDICES),
    }


def _metric_values(worker: dict) -> dict[str, list[float]]:
    values = {
        "e2e_s": [],
        "tpot_s": [],
        "executor_proposal_forward_ms": [],
        "ttft_s": [],
        "output_throughput_tps": [],
        "acceptance_rate": [],
        "peak_allocated_bytes": [],
        "peak_reserved_bytes": [],
        "proposal_kv_h2d_bytes": [],
        "proposal_kv_d2h_bytes": [],
    }
    for run in worker["measured_runs"]:
        per_request = run["timing"]["per_request"]
        values["e2e_s"].append(statistics.median(
            row["completion_latency_s"] for row in per_request
        ))
        values["tpot_s"].append(statistics.median(
            row["tpot_s"] for row in per_request
        ))
        values["ttft_s"].append(statistics.median(
            row["ttft_s"] for row in per_request
        ))
        values["output_throughput_tps"].append(float(
            run["timing"]["batch_token_throughput_tps"]
        ))
        values["executor_proposal_forward_ms"].append(float(
            run["runtime"]["draft_executor_timing"]["max_rank_ms"][
                "proposal_forward"
            ]
        ))
        values["acceptance_rate"].append(float(
            run["runtime"]["acceptance_rate"]
        ))
        values["peak_allocated_bytes"].append(float(
            run["memory"]["peak_allocated_bytes"]
        ))
        values["peak_reserved_bytes"].append(float(
            run["memory"]["peak_reserved_bytes"]
        ))
        values["proposal_kv_h2d_bytes"].append(float(
            run["proposal_kv"]["totals"]["h2d_bytes"]
        ))
        values["proposal_kv_d2h_bytes"].append(float(
            run["proposal_kv"]["totals"]["d2h_bytes"]
        ))
    return values


def _stationarity(metric: str, values: list[float]) -> dict:
    if (
        len(values) != MEASURED_RUNS
        or any(not math.isfinite(value) for value in values)
    ):
        raise ValueError("stationarity requires eight finite values")
    median = statistics.median(values)
    first_half_median = statistics.median(values[:4])
    second_half_median = statistics.median(values[4:])
    minimum = min(values)
    maximum = max(values)
    if abs(median) <= 1e-12:
        range_over_median = (
            0.0 if abs(maximum - minimum) <= 1e-12 else None
        )
        half_drift_fraction = (
            0.0
            if abs(second_half_median - first_half_median) <= 1e-12
            else None
        )
    else:
        range_over_median = (maximum - minimum) / abs(median)
        half_drift_fraction = (
            abs(second_half_median - first_half_median)
            / abs(median)
        )
    stable = (
        range_over_median is not None
        and half_drift_fraction is not None
        and range_over_median <= RANGE_OVER_MEDIAN_LIMIT
        and half_drift_fraction <= HALF_DRIFT_FRACTION_LIMIT
    )
    return {
        "metric": metric,
        "count": len(values),
        "values": list(values),
        "median": median,
        "minimum": minimum,
        "maximum": maximum,
        "range_over_median": range_over_median,
        "first_half_median": first_half_median,
        "second_half_median": second_half_median,
        "half_drift_fraction": half_drift_fraction,
        "stable": stable,
    }


def _bound_digest(
    input_files: dict,
    *,
    epoch: str,
    suffix: str,
) -> str | None:
    for key in (
        f"{epoch}_{suffix}",
        f"{epoch.replace('_', '-')}_{suffix.replace('_', '-')}",
    ):
        row = input_files.get(key)
        if isinstance(row, dict):
            digest = row.get("sha256")
            if isinstance(digest, str):
                return digest
    return None


def build_epoch_summary(
    *,
    epoch: str,
    worker: dict,
    gpu_samples: list[dict],
    host_samples: list[dict],
    input_files: dict,
) -> dict:
    normalized = validate_measured_worker(
        worker,
        artifact_identity=epoch,
    )
    try:
        gpu_coverage = summarize_gpu_telemetry(
            normalized,
            gpu_samples,
            expected_gpu_indices=GPU_INDICES,
            minimum_samples=5,
        )
    except ValueError as error:
        raise ValueError(
            f"{epoch} GPU telemetry coverage failed: {error}"
        ) from error
    host_alignments = align_repeat_samples(
        normalized,
        host_samples,
    )
    host_by_repeat = {
        row["repeat"]: {
            "alignment": row,
            "metrics": derive_repeat_metrics(row),
        }
        for row in host_alignments
    }
    gpu_by_repeat = {
        row["repeat"]: row
        for row in gpu_coverage["measured_runs"]
    }
    metric_values = _metric_values(normalized)
    stationarity = {
        metric: _stationarity(metric, metric_values[metric])
        for metric in PRIMARY_METRICS
    }
    measured_runs = []
    for run in normalized["measured_runs"]:
        repeat = run["repeat"]
        measured_runs.append({
            "repeat": repeat,
            "outputs": copy.deepcopy(run["outputs"]),
            "timing": copy.deepcopy(run["timing"]),
            "runtime": copy.deepcopy(run["runtime"]),
            "memory": copy.deepcopy(run["memory"]),
            "proposal_kv": copy.deepcopy(run["proposal_kv"]),
            "gpu_summary": copy.deepcopy(gpu_by_repeat[repeat]),
            "host_metrics": copy.deepcopy(
                host_by_repeat[repeat]["metrics"]
            ),
            "coverage": {
                "gpu": True,
                "host": True,
            },
        })
    return {
        "artifact_identity": epoch,
        "worker_policy": WORKER_POLICY,
        "worker_sha256": _bound_digest(
            input_files,
            epoch=epoch,
            suffix="worker",
        ),
        "gpu_csv_sha256": _bound_digest(
            input_files,
            epoch=epoch,
            suffix="gpu_csv",
        ),
        "host_jsonl_sha256": _bound_digest(
            input_files,
            epoch=epoch,
            suffix="host_jsonl",
        ),
        "measured_runs": measured_runs,
        "stationarity": stationarity,
        "coverage": {
            "status": "PASS",
            "gpu": gpu_coverage,
            "host_repeat_count": len(host_alignments),
        },
        "metrics": metric_values,
    }


def _comparison_row(
    metric: str,
    learned_a_values: list[float],
    learned_b_values: list[float],
) -> dict:
    median_a = statistics.median(learned_a_values)
    median_b = statistics.median(learned_b_values)
    difference = median_a - median_b
    relative_delta = (
        None if abs(median_b) <= 1e-12 else difference / median_b
    )
    sign = 0 if abs(difference) <= 1e-12 else (
        1 if difference > 0.0 else -1
    )
    return {
        "metric": metric,
        "learned_a_values": list(learned_a_values),
        "learned_b_values": list(learned_b_values),
        "learned_a_median": median_a,
        "learned_b_median": median_b,
        "absolute_difference": abs(difference),
        "relative_delta": relative_delta,
        "absolute_relative_delta": (
            None if relative_delta is None else abs(relative_delta)
        ),
        "sign": sign,
    }


def compare_epochs(
    learned_a: dict,
    learned_b: dict,
) -> dict:
    metrics_a = learned_a["metrics"]
    metrics_b = learned_b["metrics"]
    primary = {
        metric: _comparison_row(
            metric,
            metrics_a[metric],
            metrics_b[metric],
        )
        for metric in PRIMARY_METRICS
    }
    secondary = {
        metric: _comparison_row(
            metric,
            metrics_a[metric],
            metrics_b[metric],
        )
        for metric in metrics_a
        if metric not in PRIMARY_METRICS
    }
    host_metric_names = sorted(
        learned_a["measured_runs"][0]["host_metrics"]
    )
    host = {
        metric: _comparison_row(
            metric,
            [
                row["host_metrics"][metric]
                for row in learned_a["measured_runs"]
            ],
            [
                row["host_metrics"][metric]
                for row in learned_b["measured_runs"]
            ],
        )
        for metric in host_metric_names
        if all(
            row["host_metrics"][metric] is not None
            for row in (
                learned_a["measured_runs"]
                + learned_b["measured_runs"]
            )
        )
    }
    return {
        "primary": primary,
        "secondary": secondary,
        "host": host,
    }


def classify_learned_aa(
    *,
    epochs: dict,
    comparison: dict,
) -> tuple[str, list[str], dict]:
    reasons = []
    for epoch in EPOCH_ORDER:
        for metric in PRIMARY_METRICS:
            if not epochs[epoch]["stationarity"][metric]["stable"]:
                reasons.append(
                    f"{epoch} {metric} stationarity failed"
                )
    primary = comparison["primary"]
    e2e = primary["e2e_s"]
    tpot = primary["tpot_s"]
    proposal_forward = primary["executor_proposal_forward_ms"]
    if reasons:
        classification = "LEARNED_AA_INCONCLUSIVE"
    elif (
        e2e["absolute_relative_delta"] is not None
        and e2e["absolute_relative_delta"] < E2E_EFFECT_THRESHOLD
    ):
        classification = "LEARNED_AA_STABLE"
        reasons.append(
            "absolute E2E relative delta is below 0.10"
        )
    elif (
        e2e["absolute_relative_delta"] is not None
        and e2e["absolute_relative_delta"] >= E2E_EFFECT_THRESHOLD
        and e2e["sign"] != 0
        and tpot["sign"] == e2e["sign"]
        and proposal_forward["sign"] == e2e["sign"]
    ):
        classification = "LEARNED_AA_PROCESS_BOUNDARY_EFFECT"
        reasons.append(
            "E2E, TPOT, and proposal-forward show a same-sign "
            "candidate effect"
        )
    else:
        classification = "LEARNED_AA_INCONCLUSIVE"
        reasons.append(
            "E2E threshold or primary metric direction gate failed"
        )
    claim_state = {
        "candidate_process_boundary_effect": (
            classification
            == "LEARNED_AA_PROCESS_BOUNDARY_EFFECT"
        ),
        "process_boundary_effect_established": False,
    }
    return classification, reasons, claim_state


def _validate_binding_map(
    value: object,
    *,
    name: str,
) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    return copy.deepcopy(value)


def build_learned_aa_artifact(
    *,
    prime_workers: dict,
    workers: dict,
    gpu_samples: dict,
    host_samples: dict,
    epoch_order: list[str],
    prime_each_epoch: bool,
    bundle_role: str,
    input_files: dict,
    source_files: dict,
) -> dict:
    if epoch_order != list(EPOCH_ORDER):
        raise ValueError("learned A/A epoch order mismatch")
    if prime_each_epoch is not True:
        raise ValueError("learned A/A requires priming each epoch")
    if bundle_role != "discovery":
        raise ValueError("learned A/A bundle role must be discovery")
    for mapping, name in (
        (prime_workers, "prime workers"),
        (workers, "workers"),
        (gpu_samples, "GPU samples"),
        (host_samples, "host samples"),
    ):
        if (
            not isinstance(mapping, dict)
            or set(mapping) != set(EPOCH_ORDER)
        ):
            raise ValueError(
                f"learned A/A {name} inventory mismatch"
            )
    normalized_inputs = _validate_binding_map(
        input_files,
        name="input files",
    )
    normalized_sources = _validate_binding_map(
        source_files,
        name="source files",
    )
    normalized_workers = {
        epoch: validate_measured_worker(
            workers[epoch],
            artifact_identity=epoch,
        )
        for epoch in EPOCH_ORDER
    }
    for epoch in EPOCH_ORDER:
        validate_prime_worker(
            prime_workers[epoch],
            artifact_identity=epoch,
        )
    workload_identity = validate_workload_identity(
        normalized_workers["learned_a"],
        normalized_workers["learned_b"],
    )
    for repeat in range(MEASURED_RUNS):
        if (
            normalized_workers["learned_a"]["measured_runs"][
                repeat
            ]["outputs"]
            != normalized_workers["learned_b"]["measured_runs"][
                repeat
            ]["outputs"]
        ):
            raise ValueError(
                f"learned A/A exact parity failed at repeat {repeat}"
            )
    epochs = {
        epoch: build_epoch_summary(
            epoch=epoch,
            worker=normalized_workers[epoch],
            gpu_samples=gpu_samples[epoch],
            host_samples=host_samples[epoch],
            input_files=normalized_inputs,
        )
        for epoch in EPOCH_ORDER
    }
    comparison = compare_epochs(
        epochs["learned_a"],
        epochs["learned_b"],
    )
    classification, reasons, claim_state = classify_learned_aa(
        epochs=epochs,
        comparison=comparison,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": classification,
        "classification_reasons": reasons,
        "claim_state": claim_state,
        "bundle_role": bundle_role,
        "epoch_order": list(EPOCH_ORDER),
        "prime_each_epoch": True,
        "exact_parity": True,
        "workload_identity": workload_identity,
        "input_files": normalized_inputs,
        "source_files": normalized_sources,
        "epochs": epochs,
        "comparison": comparison,
        "thresholds": {
            "range_over_median_limit": RANGE_OVER_MEDIAN_LIMIT,
            "half_drift_fraction_limit": (
                HALF_DRIFT_FRACTION_LIMIT
            ),
            "e2e_effect_threshold": E2E_EFFECT_THRESHOLD,
            "host_sample_cadence_seconds": 0.2,
            "host_maximum_repeat_local_gap_seconds": 0.6,
            "host_boundary_allowance_seconds": 0.4,
            "gpu_boundary_allowance_seconds": 0.6,
            "minimum_gpu_samples_per_repeat": 5,
            "primary_metrics": list(PRIMARY_METRICS),
        },
        "limitations": list(LIMITATIONS),
    }


def validate_learned_aa_artifact(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("learned A/A artifact must be a mapping")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("learned A/A schema version mismatch")
    if artifact.get("status") != "PASS":
        raise ValueError("learned A/A status must be PASS")
    if artifact.get("epoch_order") != list(EPOCH_ORDER):
        raise ValueError("learned A/A epoch order mismatch")
    if artifact.get("prime_each_epoch") is not True:
        raise ValueError("learned A/A prime contract mismatch")
    claim_state = artifact.get("claim_state")
    if (
        not isinstance(claim_state, dict)
        or claim_state.get("process_boundary_effect_established")
        is not False
    ):
        raise ValueError(
            "single-bundle process-boundary claim is invalid"
        )
    if artifact.get("exact_parity") is not True:
        raise ValueError("learned A/A exact parity is invalid")
    return {
        "status": "PASS",
        "classification": artifact.get("classification"),
        "exact_parity": True,
        "process_boundary_effect_established": False,
    }


def _load_json(path: Path, *, name: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(
    path: Path,
    *,
    base: Path,
    name: str,
) -> str:
    try:
        relative = path.resolve().relative_to(base.resolve())
    except ValueError as error:
        raise ValueError(
            f"{name} path must be below output directory"
        ) from error
    pure = PurePosixPath(relative.as_posix())
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"{name} path must be a safe relative path")
    return pure.as_posix()


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _input_paths_from_args(args) -> dict[str, Path]:
    return {
        name: Path(getattr(args, argument))
        for name, argument in INPUT_FILE_ARGUMENTS.items()
    }


def _validate_distinct_epoch_digests(input_files: dict) -> None:
    for suffix in (
        "prime_worker",
        "worker",
        "gpu_csv",
        "host_jsonl",
    ):
        if (
            input_files[f"learned_a_{suffix}"]["sha256"]
            == input_files[f"learned_b_{suffix}"]["sha256"]
        ):
            raise ValueError(
                f"learned A/A {suffix} digests must be distinct"
            )


def _build_from_args(args) -> dict:
    output_path = Path(args.out)
    output_directory = output_path.parent
    input_paths = _input_paths_from_args(args)
    input_files = {
        name: {
            "path": _relative_path(
                path,
                base=output_directory,
                name=name,
            ),
            "sha256": _sha256_path(path),
        }
        for name, path in input_paths.items()
    }
    _validate_distinct_epoch_digests(input_files)
    epoch_order = (
        input_paths["epoch_order"]
        .read_text(encoding="utf-8")
        .strip()
        .split(",")
    )
    prime_each_epoch_text = (
        input_paths["prime_each_epoch"]
        .read_text(encoding="utf-8")
        .strip()
    )
    if prime_each_epoch_text != "1":
        raise ValueError("prime-each-epoch file must contain 1")
    repo_root = Path(args.repo_root)
    source_files = {}
    for relative_path in SOURCE_FILE_PATHS:
        source_path = repo_root / relative_path
        if not source_path.is_file():
            raise ValueError(
                f"source file is missing: {relative_path}"
            )
        source_files[relative_path] = _sha256_path(source_path)
    return build_learned_aa_artifact(
        prime_workers={
            "learned_a": _load_json(
                input_paths["learned_a_prime_worker"],
                name="learned A prime worker",
            ),
            "learned_b": _load_json(
                input_paths["learned_b_prime_worker"],
                name="learned B prime worker",
            ),
        },
        workers={
            "learned_a": _load_json(
                input_paths["learned_a_worker"],
                name="learned A worker",
            ),
            "learned_b": _load_json(
                input_paths["learned_b_worker"],
                name="learned B worker",
            ),
        },
        gpu_samples={
            "learned_a": parse_gpu_telemetry(
                input_paths["learned_a_gpu_csv"].read_text(
                    encoding="utf-8"
                )
            ),
            "learned_b": parse_gpu_telemetry(
                input_paths["learned_b_gpu_csv"].read_text(
                    encoding="utf-8"
                )
            ),
        },
        host_samples={
            "learned_a": parse_host_jsonl(
                input_paths["learned_a_host_jsonl"].read_text(
                    encoding="utf-8"
                )
            ),
            "learned_b": parse_host_jsonl(
                input_paths["learned_b_host_jsonl"].read_text(
                    encoding="utf-8"
                )
            ),
        },
        epoch_order=epoch_order,
        prime_each_epoch=True,
        bundle_role=args.bundle_role,
        input_files=input_files,
        source_files=source_files,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--learned-a-prime-worker", required=True)
    parser.add_argument("--learned-b-prime-worker", required=True)
    parser.add_argument("--learned-a-worker", required=True)
    parser.add_argument("--learned-b-worker", required=True)
    parser.add_argument("--learned-a-gpu-csv", required=True)
    parser.add_argument("--learned-b-gpu-csv", required=True)
    parser.add_argument("--learned-a-host-jsonl", required=True)
    parser.add_argument("--learned-b-host-jsonl", required=True)
    parser.add_argument("--epoch-order-file", required=True)
    parser.add_argument("--prime-each-epoch-file", required=True)
    parser.add_argument(
        "--bundle-role",
        choices=("discovery",),
        required=True,
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    artifact = _build_from_args(args)
    _write_json_atomic(Path(args.out), artifact)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
