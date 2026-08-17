from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import platform
import statistics
import sys


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from autoregressive_draft_performance_gate import (
    PROPOSAL_FORWARD_DETAIL_KEYS,
    _equivalent,
    hash_source_files,
    validate_worker_result,
    write_json_atomic,
)


SCHEMA_VERSION = 1
WARMUP_RUNS = 2
MEASURED_RUNS = 8
BATCH_SIZE = 4
RANGE_OVER_MEDIAN_LIMIT = 0.25
HALF_DRIFT_FRACTION_LIMIT = 0.20
PRIMARY_STATIONARITY_METRICS = (
    "e2e_s",
    "tpot_s",
    "executor_proposal_forward_ms",
)
DEFAULT_SOURCE_FILES = (
    "tinyvllm/engine/autoregressive_draft_executor.py",
    "tinyvllm/engine/qwen3_draft_backend.py",
    "tools/autoregressive_draft_performance_gate.py",
    "tools/autoregressive_draft_performance_worker.py",
    "tools/autoregressive_draft_b4_timing_diagnostic.py",
    "tools/verify_autoregressive_draft_b4_timing_diagnostic.py",
)
LIMITATIONS = (
    "wall-clock substages do not establish GPU kernel duration",
    "classification is diagnostic stationarity, not performance promotion",
    "256-token prompts do not establish 4K, 16K, or 32K performance",
    "direct Proposal-KV allocation does not establish offload benefit",
    "one independent-draft model pair does not establish two structures",
)


def _validate_diagnostic_worker(
    worker: object,
    *,
    policy: str,
) -> dict:
    normalized = validate_worker_result(
        worker,
        expected_warmup_runs=WARMUP_RUNS,
        expected_measured_runs=MEASURED_RUNS,
    )
    if normalized["policy"] != policy:
        raise ValueError(f"diagnostic {policy} worker policy mismatch")
    if normalized["batch_size"] != BATCH_SIZE:
        raise ValueError("diagnostic worker batch size must be four")
    return normalized


def _run_metric_values(worker: dict) -> dict[str, list[float]]:
    values = {
        "ttft_s": [],
        "tpot_s": [],
        "e2e_s": [],
        "output_throughput_tps": [],
        "executor_proposal_forward_ms": [],
        **{
            f"executor_detail_{key}_ms": []
            for key in PROPOSAL_FORWARD_DETAIL_KEYS
        },
        "executor_detail_sum_ms": [],
        "executor_detail_residual_ms": [],
    }
    for run in worker["measured_runs"]:
        per_request = run["timing"]["per_request"]
        values["ttft_s"].append(statistics.median(
            row["ttft_s"] for row in per_request
        ))
        values["tpot_s"].append(statistics.median(
            row["tpot_s"] for row in per_request
        ))
        values["e2e_s"].append(statistics.median(
            row["completion_latency_s"] for row in per_request
        ))
        values["output_throughput_tps"].append(float(
            run["timing"]["batch_token_throughput_tps"]
        ))
        runtime = run["runtime"]
        values["executor_proposal_forward_ms"].append(float(
            runtime["draft_executor_timing"]["max_rank_ms"][
                "proposal_forward"
            ]
        ))
        detail = runtime["draft_executor_proposal_detail"]
        for key in PROPOSAL_FORWARD_DETAIL_KEYS:
            values[f"executor_detail_{key}_ms"].append(float(
                detail["critical_rank_ms"][key]
            ))
        values["executor_detail_sum_ms"].append(float(
            detail["detail_sum_ms"]
        ))
        values["executor_detail_residual_ms"].append(float(
            detail["residual_ms"]
        ))
    return values


def _stationarity_row(
    *,
    policy: str,
    metric: str,
    values: list[float],
) -> dict:
    if len(values) != MEASURED_RUNS or any(
        not math.isfinite(value) for value in values
    ):
        raise ValueError("stationarity values are invalid")
    midpoint = len(values) // 2
    median = statistics.median(values)
    first_half_median = statistics.median(values[:midpoint])
    second_half_median = statistics.median(values[midpoint:])
    minimum = min(values)
    maximum = max(values)
    if abs(median) <= 1e-12:
        range_over_median = 0.0 if maximum - minimum <= 1e-12 else None
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
        "policy": policy,
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


def _stationarity(target: dict, learned: dict) -> dict:
    rows = []
    for policy, worker in (
        ("target", target),
        ("learned", learned),
    ):
        for metric, values in _run_metric_values(worker).items():
            rows.append(_stationarity_row(
                policy=policy,
                metric=metric,
                values=values,
            ))
    learned_primary = {
        row["metric"]: row
        for row in rows
        if row["policy"] == "learned"
        and row["metric"] in PRIMARY_STATIONARITY_METRICS
    }
    stable = all(
        learned_primary[metric]["stable"]
        for metric in PRIMARY_STATIONARITY_METRICS
    )
    return {
        "thresholds": {
            "range_over_median_limit": RANGE_OVER_MEDIAN_LIMIT,
            "half_drift_fraction_limit": HALF_DRIFT_FRACTION_LIMIT,
            "primary_metrics": list(PRIMARY_STATIONARITY_METRICS),
        },
        "rows": rows,
        "stable": stable,
    }


def _validate_source_files(source_files: object) -> dict[str, str]:
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("diagnostic source files must be non-empty")
    normalized = {}
    for path, digest in source_files.items():
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in digest
            )
        ):
            raise ValueError("diagnostic source binding is invalid")
        normalized[path] = digest
    return normalized


def build_b4_timing_diagnostic(
    *,
    target_worker: dict,
    learned_worker: dict,
    environment: dict,
    source_files: dict[str, str],
) -> dict:
    target = _validate_diagnostic_worker(
        target_worker,
        policy="target",
    )
    learned = _validate_diagnostic_worker(
        learned_worker,
        policy="learned",
    )
    if not isinstance(environment, dict):
        raise ValueError("diagnostic environment must be a mapping")
    for key in (
        "target_checkpoint_identifier",
        "tokenizer_identifier",
        "dtype",
    ):
        if target[key] != learned[key]:
            raise ValueError("diagnostic worker identity mismatch")
    if target["prompt_rows"] != learned["prompt_rows"]:
        raise ValueError("diagnostic prompt parity failed")
    for repeat in range(MEASURED_RUNS):
        if (
            target["measured_runs"][repeat]["outputs"]
            != learned["measured_runs"][repeat]["outputs"]
        ):
            raise ValueError(
                f"diagnostic exact parity failed at repeat {repeat}"
            )
    stationarity = _stationarity(target, learned)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "classification": (
            "STABLE" if stationarity["stable"] else "UNSTABLE"
        ),
        "exact_parity": True,
        "campaign": {
            "tensor_parallel_size": 4,
            "batch_size": BATCH_SIZE,
            "prompt_tokens": 256,
            "max_output_tokens": 16,
            "temperature": 0.0,
            "max_proposal_tokens": 4,
            "warmup_runs": WARMUP_RUNS,
            "measured_runs": MEASURED_RUNS,
        },
        "environment": copy.deepcopy(environment),
        "workers": {
            "target": target,
            "learned": learned,
        },
        "stationarity": stationarity,
        "source_files": _validate_source_files(source_files),
        "limitations": list(LIMITATIONS),
    }


def validate_b4_timing_diagnostic(artifact: object) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("diagnostic artifact must be a mapping")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("diagnostic schema version mismatch")
    workers = artifact.get("workers")
    if not isinstance(workers, dict):
        raise ValueError("diagnostic workers must be a mapping")
    expected = build_b4_timing_diagnostic(
        target_worker=workers.get("target"),
        learned_worker=workers.get("learned"),
        environment=artifact.get("environment"),
        source_files=artifact.get("source_files"),
    )
    if not _equivalent(artifact, expected):
        raise ValueError("diagnostic artifact recomputation mismatch")
    return {
        "status": "PASS",
        "classification": expected["classification"],
        "exact_parity": True,
        "measured_runs": MEASURED_RUNS,
    }


def _default_environment(command: list[str]) -> dict:
    try:
        import torch

        torch_version = str(torch.__version__)
        device_names = (
            [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else []
        )
    except Exception:
        torch_version = "unavailable"
        device_names = []
    return {
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "device_names": device_names,
        "command": list(command),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-worker", required=True)
    parser.add_argument("--learned-worker", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    target_worker = json.loads(
        Path(args.target_worker).read_text(encoding="utf-8")
    )
    learned_worker = json.loads(
        Path(args.learned_worker).read_text(encoding="utf-8")
    )
    repo_root = Path(args.repo_root)
    artifact = build_b4_timing_diagnostic(
        target_worker=target_worker,
        learned_worker=learned_worker,
        environment=_default_environment(sys.argv),
        source_files=hash_source_files(
            repo_root=repo_root,
            source_files=DEFAULT_SOURCE_FILES,
        ),
    )
    validate_b4_timing_diagnostic(artifact)
    write_json_atomic(Path(args.out), artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
