"""Independent verifier for production arrival-load gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tarfile
import tempfile
from pathlib import Path


REQUIRED_FILES = (
    "run_manifest.json",
    "calibration_manifest.jsonl",
    "calibration_rows.jsonl",
    "workload_manifest.jsonl",
    "request_timeline.jsonl",
    "scheduler_trace.jsonl",
    "memory_trace.jsonl",
    "case_rows.jsonl",
    "summary.json",
    "report.md",
    "source_evidence.json",
    "source.patch",
    "source_snapshot.tar.gz",
    "artifact_hashes.json",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file: {path.name}") from exc


def _read_jsonl(path: Path) -> list[dict]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"missing JSONL file: {path.name}") from exc
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"JSONL missing final newline: {path.name}")
    rows = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line:
            raise ValueError(
                f"blank JSONL record: {path.name}:{line_number}"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"malformed JSONL: {path.name}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"JSONL record must be object: {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _finite(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be finite")
    return normalized


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    normalized = [_finite(value, "percentile sample") for value in values]
    normalized.sort()
    return normalized[math.ceil(len(normalized) * percentile) - 1]


def _verify_hashes(run_dir: Path) -> None:
    for name in REQUIRED_FILES:
        if not (run_dir / name).is_file():
            raise ValueError(f"missing artifact: {name}")
    hashes = _read_json(run_dir / "artifact_hashes.json")
    expected_names = set(REQUIRED_FILES) - {"artifact_hashes.json"}
    if set(hashes) != expected_names:
        raise ValueError("artifact hash manifest path set mismatch")
    for name in sorted(expected_names):
        if hashes[name] != _sha256_file(run_dir / name):
            raise ValueError(f"artifact hash mismatch: {name}")


def _safe_extract_snapshot(archive_path: Path, output_dir: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("unsafe source snapshot member")
        archive.extractall(output_dir)
    source_root = output_dir / "source"
    if not source_root.is_dir():
        raise ValueError("source snapshot is missing source root")
    return source_root


def _verify_source(run_dir: Path, manifest: dict) -> None:
    evidence = _read_json(run_dir / "source_evidence.json")
    if evidence.get("schema_version") != 1:
        raise ValueError("unsupported source evidence schema")
    patch_payload = (run_dir / "source.patch").read_bytes()
    if evidence.get("patch_size_bytes") != len(patch_payload):
        raise ValueError("source patch size mismatch")
    if (
        evidence.get("patch_sha256")
        != hashlib.sha256(patch_payload).hexdigest()
    ):
        raise ValueError("source patch hash mismatch")
    expected_files = evidence.get("files")
    if not isinstance(expected_files, list):
        raise ValueError("invalid source evidence files")
    with tempfile.TemporaryDirectory() as temporary:
        source_root = _safe_extract_snapshot(
            run_dir / "source_snapshot.tar.gz",
            Path(temporary),
        )
        actual_files = []
        for path in sorted(source_root.rglob("*")):
            if path.is_symlink():
                raise ValueError("source snapshot contains symlink")
            if path.is_file():
                actual_files.append({
                    "path": path.relative_to(source_root).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                })
        if actual_files != expected_files:
            raise ValueError("source snapshot file identity mismatch")
        tree_sha256 = hashlib.sha256(
            _canonical_bytes(actual_files)
        ).hexdigest()
        if tree_sha256 != evidence.get("tree_sha256"):
            raise ValueError("source tree hash mismatch")
        if tree_sha256 != manifest.get("source_tree_sha256"):
            raise ValueError("manifest source tree hash mismatch")


def _verify_ports(manifest: dict) -> None:
    rows = manifest.get("process_port_pairs")
    if not isinstance(rows, list):
        raise ValueError("invalid process port records")
    pairs = []
    case_ids = []
    for row in rows:
        case_id = row.get("case_id")
        dist_port = row.get("tinyvllm_dist_port")
        master_port = row.get("master_port")
        if (
            not isinstance(case_id, str)
            or not isinstance(dist_port, int)
            or not isinstance(master_port, int)
            or dist_port <= 0
            or master_port <= 0
            or dist_port == master_port
        ):
            raise ValueError("invalid process port record")
        case_ids.append(case_id)
        pairs.append((dist_port, master_port))
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate process port pair")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate process case id")
    if set(case_ids) != set(manifest.get("expected_case_ids", [])):
        raise ValueError("process case matrix mismatch")


def _request_metrics(workload: dict, timeline: dict) -> dict:
    names = (
        "scheduled_arrival_ns",
        "actual_arrival_ns",
        "first_scheduled_ns",
        "first_token_ns",
        "completion_ns",
    )
    times = {
        name: _finite(timeline.get(name), name)
        for name in names
    }
    if not (
        times["scheduled_arrival_ns"]
        <= times["actual_arrival_ns"]
        <= times["first_scheduled_ns"]
        <= times["first_token_ns"]
        <= times["completion_ns"]
    ):
        raise ValueError("impossible timestamp ordering")
    token_times = [
        _finite(value, "token timestamp")
        for value in timeline.get("token_timestamps_ns", [])
    ]
    output_ids = timeline.get("output_token_ids")
    if (
        not isinstance(output_ids, list)
        or len(output_ids) != workload.get("requested_output_tokens")
        or len(token_times) != len(output_ids)
        or not token_times
    ):
        raise ValueError("token timestamp or output count mismatch")
    if token_times[0] != times["first_token_ns"]:
        raise ValueError("first token timestamp mismatch")
    if token_times[-1] > times["completion_ns"]:
        raise ValueError("token timestamp exceeds completion")
    if any(
        current < previous
        for previous, current in zip(token_times, token_times[1:])
    ):
        raise ValueError("non-monotonic token timestamps")
    if timeline.get("finish_reason") != "length":
        raise ValueError("unexpected finish reason")
    if timeline.get("error") is not None:
        raise ValueError("request error")
    itl = [
        current - previous
        for previous, current in zip(token_times, token_times[1:])
    ]
    return {
        **workload,
        "output_token_ids": list(output_ids),
        "injection_lag_ns": (
            times["actual_arrival_ns"]
            - times["scheduled_arrival_ns"]
        ),
        "queue_delay_ns": (
            times["first_scheduled_ns"]
            - times["actual_arrival_ns"]
        ),
        "ttft_ns": (
            times["first_token_ns"]
            - times["scheduled_arrival_ns"]
        ),
        "e2e_ns": (
            times["completion_ns"]
            - times["scheduled_arrival_ns"]
        ),
        "itl_ns": itl,
        "maximum_decode_gap_ns": max(itl) if itl else None,
        "scheduled_arrival_ns": times["scheduled_arrival_ns"],
        "completion_ns": times["completion_ns"],
    }


def _percentiles(values: list[float], prefix: str) -> dict:
    return {
        f"p50_{prefix}": _nearest_rank(values, 0.50),
        f"p95_{prefix}": _nearest_rank(values, 0.95),
        f"p99_{prefix}": _nearest_rank(values, 0.99),
    }


def _jain_index(values: list[float]) -> float:
    if not values or any(value < 0.0 for value in values):
        raise ValueError("invalid Jain index samples")
    denominator = len(values) * sum(value * value for value in values)
    if denominator == 0.0:
        return 0.0
    return (sum(values) ** 2) / denominator


def _recompute_case(
    case_id: str,
    timeline_rows: list[dict],
    scheduler_rows: list[dict],
    memory_rows: list[dict],
    workload_by_id: dict[str, dict],
) -> dict:
    case_timeline = [
        row for row in timeline_rows if row.get("case_id") == case_id
    ]
    if not case_timeline:
        raise ValueError(f"missing request timeline: {case_id}")
    if len({
        row.get("request_id") for row in case_timeline
    }) != len(case_timeline):
        raise ValueError(f"duplicate request timeline: {case_id}")
    if len({row.get("seq_id") for row in case_timeline}) != len(
        case_timeline
    ):
        raise ValueError(f"duplicate request binding: {case_id}")
    case_scheduler = [
        row for row in scheduler_rows if row.get("case_id") == case_id
    ]
    if not case_scheduler:
        raise ValueError(f"missing scheduler trace: {case_id}")
    step_indices = [row.get("step_index") for row in case_scheduler]
    if step_indices != list(range(len(step_indices))):
        raise ValueError(f"invalid scheduler step sequence: {case_id}")
    case_memory = [
        row for row in memory_rows if row.get("case_id") == case_id
    ]
    if not case_memory:
        raise ValueError(f"missing memory trace: {case_id}")

    request_rows = []
    for timeline in case_timeline:
        request_id = timeline.get("request_id")
        if request_id not in workload_by_id:
            raise ValueError(f"unexpected request id: {request_id}")
        request_rows.append(
            _request_metrics(workload_by_id[request_id], timeline)
        )
    start_ns = min(row["scheduled_arrival_ns"] for row in request_rows)
    end_ns = max(row["completion_ns"] for row in request_rows)
    duration_s = (end_ns - start_ns) / 1_000_000_000.0
    if duration_s <= 0.0:
        raise ValueError(f"invalid measurement duration: {case_id}")
    itl_values = [
        value
        for row in request_rows
        for value in row["itl_ns"]
    ]
    if not itl_values:
        raise ValueError(f"case has no ITL samples: {case_id}")
    metrics = {
        "request_throughput_rps": len(request_rows) / duration_s,
        "output_token_throughput_tps": sum(
            len(row["output_token_ids"]) for row in request_rows
        ) / duration_s,
        "maximum_injection_lag_ns": max(
            row["injection_lag_ns"] for row in request_rows
        ),
        **_percentiles(
            [row["injection_lag_ns"] for row in request_rows],
            "injection_lag_ns",
        ),
        **_percentiles(
            [row["queue_delay_ns"] for row in request_rows],
            "queue_delay_ns",
        ),
        **_percentiles(
            [row["ttft_ns"] for row in request_rows],
            "ttft_ns",
        ),
        **_percentiles(itl_values, "itl_ns"),
        **_percentiles(
            [row["e2e_ns"] for row in request_rows],
            "e2e_ns",
        ),
        "maximum_decode_gap_ns": max(
            row["maximum_decode_gap_ns"]
            for row in request_rows
            if row["maximum_decode_gap_ns"] is not None
        ),
        "peak_cuda_allocated_bytes": int(max(
            _finite(row.get("cuda_allocated_bytes"), "allocated memory")
            for row in case_memory
        )),
        "peak_cuda_reserved_bytes": int(max(
            _finite(row.get("cuda_reserved_bytes"), "reserved memory")
            for row in case_memory
        )),
        "peak_used_kv_blocks": int(max(
            _finite(row.get("used_kv_blocks"), "used KV blocks")
            for row in case_memory
        )),
        "peak_kv_bytes": int(max(
            _finite(row.get("used_kv_blocks"), "used KV blocks")
            * _finite(row.get("kv_block_bytes"), "KV block bytes")
            for row in case_memory
        )),
    }
    service_buckets = {}
    service_rates = []
    for bucket in sorted({
        row["service_time_bucket"] for row in request_rows
    }):
        bucket_rows = [
            row for row in request_rows
            if row["service_time_bucket"] == bucket
        ]
        bucket_metrics = {
            "completed_requests": len(bucket_rows),
            "request_throughput_rps": len(bucket_rows) / duration_s,
            "worst_e2e_ns": max(
                row["e2e_ns"] for row in bucket_rows
            ),
            **_percentiles(
                [row["e2e_ns"] for row in bucket_rows],
                "e2e_ns",
            ),
        }
        service_buckets[bucket] = bucket_metrics
        service_rates.append(
            bucket_metrics["request_throughput_rps"]
        )
    metrics["service_buckets"] = service_buckets
    metrics["jain_service_rate_index"] = _jain_index(
        service_rates
    )
    first = case_timeline[0]
    return {
        "case_id": case_id,
        "policy": first["policy"],
        "scenario": first["scenario"],
        "repetition": first["repetition"],
        "status": "PASS",
        "correctness": {
            "exact_outputs": True,
            "complete_requests": True,
            "no_starvation": True,
            "valid_lifecycle": True,
            "stable_p0_outputs": True,
        },
        "metrics": metrics,
    }


def _ratio(candidate: dict, baseline: dict, metric: str) -> float:
    baseline_value = _finite(
        baseline["metrics"].get(metric),
        f"baseline {metric}",
    )
    candidate_value = _finite(
        candidate["metrics"].get(metric),
        f"candidate {metric}",
    )
    if baseline_value <= 0.0:
        raise ValueError(f"baseline {metric} must be positive")
    return candidate_value / baseline_value


def _candidate_result(
    policy: str,
    paired: list[tuple[dict, dict]],
) -> dict:
    metric_names = (
        "request_throughput_rps",
        "p95_ttft_ns",
        "p95_itl_ns",
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
        "peak_cuda_reserved_bytes",
        "peak_kv_bytes",
    )
    ratios = {
        metric: [
            _ratio(candidate, baseline, metric)
            for baseline, candidate in paired
        ]
        for metric in metric_names
    }
    bucket_ratios = []
    for baseline, candidate in paired:
        baseline_buckets = baseline["metrics"]["service_buckets"]
        candidate_buckets = candidate["metrics"]["service_buckets"]
        if set(baseline_buckets) != set(candidate_buckets):
            raise ValueError("service bucket set mismatch")
        for bucket in sorted(baseline_buckets):
            baseline_value = _finite(
                baseline_buckets[bucket]["p95_e2e_ns"],
                "baseline bucket p95",
            )
            candidate_value = _finite(
                candidate_buckets[bucket]["p95_e2e_ns"],
                "candidate bucket p95",
            )
            bucket_ratios.append(candidate_value / baseline_value)
    median = {
        metric: statistics.median(values)
        for metric, values in ratios.items()
    }
    worst = {
        "request_throughput_rps": min(
            ratios["request_throughput_rps"]
        ),
        **{
            metric: max(values)
            for metric, values in ratios.items()
            if metric != "request_throughput_rps"
        },
    }
    guard_failures = []
    for metric in (
        "p99_ttft_ns",
        "p99_itl_ns",
        "p99_e2e_ns",
        "maximum_decode_gap_ns",
    ):
        if worst[metric] > 1.10:
            guard_failures.append(f"{metric} regression exceeds 10%")
    if bucket_ratios and max(bucket_ratios) > 1.10:
        guard_failures.append(
            "service bucket p95 E2E regression exceeds 10%"
        )
    median_paths = {
        "throughput": (
            median["request_throughput_rps"] >= 1.05
            and median["p95_ttft_ns"] <= 1.05
            and median["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                median["p95_ttft_ns"] <= 0.90
                and median["p95_itl_ns"] <= 1.05
            )
            or (
                median["p95_itl_ns"] <= 0.90
                and median["p95_ttft_ns"] <= 1.05
            )
        ) and median["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                median["peak_cuda_reserved_bytes"],
                median["peak_kv_bytes"],
            ) <= 0.95
            and median["request_throughput_rps"] >= 0.98
            and median["p95_ttft_ns"] <= 1.02
            and median["p95_itl_ns"] <= 1.02
        ),
    }
    worst_paths = {
        "throughput": (
            worst["request_throughput_rps"] >= 1.05
            and worst["p95_ttft_ns"] <= 1.05
            and worst["p95_itl_ns"] <= 1.05
        ),
        "latency": (
            (
                worst["p95_ttft_ns"] <= 0.90
                and worst["p95_itl_ns"] <= 1.05
            )
            or (
                worst["p95_itl_ns"] <= 0.90
                and worst["p95_ttft_ns"] <= 1.05
            )
        ) and worst["request_throughput_rps"] >= 0.98,
        "memory": (
            min(
                worst["peak_cuda_reserved_bytes"],
                worst["peak_kv_bytes"],
            ) <= 0.95
            and worst["request_throughput_rps"] >= 0.98
            and worst["p95_ttft_ns"] <= 1.02
            and worst["p95_itl_ns"] <= 1.02
        ),
    }
    benefit_path = next(
        (
            path
            for path in ("throughput", "latency", "memory")
            if median_paths[path] and worst_paths[path]
        ),
        None,
    )
    favorable = (
        median["request_throughput_rps"] > 1.0
        or median["p95_ttft_ns"] < 1.0
        or median["p95_itl_ns"] < 1.0
        or median["peak_cuda_reserved_bytes"] < 1.0
        or median["peak_kv_bytes"] < 1.0
    )
    if guard_failures:
        classification = "NO_GO"
    elif benefit_path is not None:
        classification = "GO"
    elif favorable:
        classification = "PROMISING_NOT_PROVEN"
    else:
        classification = "NO_GO"
    return {
        "policy": policy,
        "classification": classification,
        "benefit_path": benefit_path,
        "median_ratios": median,
        "worst_repetition_ratios": worst,
        "guard_failures": guard_failures,
    }


def _classify(manifest: dict, rows: list[dict]) -> dict:
    required_scenarios = manifest.get("required_scenarios")
    repetitions = manifest.get("measured_repetitions")
    aliases = manifest.get("canonical_policy_by_name")
    identities = manifest.get("policy_identity_by_name")
    if (
        not isinstance(required_scenarios, list)
        or not required_scenarios
        or not isinstance(repetitions, int)
        or repetitions < 3
        or set(aliases or {}) != {"P0", "P1", "P2", "P3"}
        or set(identities or {}) != {"P0", "P1", "P2", "P3"}
    ):
        raise ValueError("invalid policy or case manifest")
    if aliases["P1"] == "P0" and identities["P1"] != identities["P0"]:
        raise ValueError("P1 alias identity mismatch")
    if identities["P2"] in {identities["P0"], identities["P3"]}:
        raise ValueError("unexpected P2 identity collision")
    if identities["P3"] == identities["P0"]:
        raise ValueError("unexpected P3 identity collision")
    canonical_policies = [
        name for name in ("P0", "P1", "P2", "P3")
        if aliases[name] == name
    ]
    expected = {
        (policy, scenario, repetition)
        for policy in canonical_policies
        for scenario in required_scenarios
        for repetition in range(repetitions)
    }
    by_key = {}
    for row in rows:
        key = (
            row.get("policy"),
            row.get("scenario"),
            row.get("repetition"),
        )
        if key in by_key:
            raise ValueError("duplicate case rows")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("missing or unexpected case rows")
    candidate_results = {}
    for policy in canonical_policies:
        if policy == "P0":
            continue
        paired = []
        for scenario in required_scenarios:
            for repetition in range(repetitions):
                paired.append((
                    by_key[("P0", scenario, repetition)],
                    by_key[(policy, scenario, repetition)],
                ))
        candidate_results[policy] = _candidate_result(policy, paired)
    classifications = {
        row["classification"] for row in candidate_results.values()
    }
    if "GO" in classifications:
        classification = "GO"
    elif "PROMISING_NOT_PROVEN" in classifications:
        classification = "PROMISING_NOT_PROVEN"
    else:
        classification = "NO_GO"
    return {
        "classification": classification,
        "structural_failures": [],
        "correctness_failures": [],
        "candidate_results": candidate_results,
    }


def _render_report(summary: dict) -> str:
    return (
        "# Production Arrival-Load Gate\n\n"
        f"Classification: `{summary['classification']}`\n"
    )


def _verify_output_equality(
    timeline_rows: list[dict],
    manifest: dict,
) -> None:
    by_case = {}
    for row in timeline_rows:
        key = (
            row.get("policy"),
            row.get("scenario"),
            row.get("repetition"),
        )
        requests = by_case.setdefault(key, {})
        request_id = row.get("request_id")
        if request_id in requests:
            raise ValueError("duplicate request timeline")
        requests[request_id] = row
    for scenario in manifest["required_scenarios"]:
        for repetition in range(manifest["measured_repetitions"]):
            baseline_rows = by_case.get(
                ("P0", scenario, repetition),
                {},
            )
            if not baseline_rows:
                raise ValueError("missing baseline request timeline")
            for policy in ("P2", "P3"):
                candidate_rows = by_case.get(
                    (policy, scenario, repetition),
                    {},
                )
                if set(candidate_rows) != set(baseline_rows):
                    raise ValueError("request-set mismatch")
                for request_id in baseline_rows:
                    if (
                        candidate_rows[request_id]["output_token_ids"]
                        != baseline_rows[request_id]["output_token_ids"]
                    ):
                        raise ValueError("output token mismatch")


def verify_run(
    run_dir: Path,
    *,
    write_output: bool = True,
) -> dict:
    run_dir = Path(run_dir)
    _verify_hashes(run_dir)
    manifest = _read_json(run_dir / "run_manifest.json")
    _verify_source(run_dir, manifest)
    _verify_ports(manifest)
    _read_jsonl(run_dir / "calibration_manifest.jsonl")
    _read_jsonl(run_dir / "calibration_rows.jsonl")
    workload_rows = _read_jsonl(
        run_dir / "workload_manifest.jsonl"
    )
    workload_by_id = {}
    for row in workload_rows:
        request_id = row.get("request_id")
        if request_id in workload_by_id:
            raise ValueError("duplicate workload request")
        workload_by_id[request_id] = row
    timeline_rows = _read_jsonl(
        run_dir / "request_timeline.jsonl"
    )
    scheduler_rows = _read_jsonl(
        run_dir / "scheduler_trace.jsonl"
    )
    memory_rows = _read_jsonl(run_dir / "memory_trace.jsonl")
    recorded_case_rows = _read_jsonl(run_dir / "case_rows.jsonl")
    _verify_output_equality(timeline_rows, manifest)
    recomputed_case_rows = [
        _recompute_case(
            case_id,
            timeline_rows,
            scheduler_rows,
            memory_rows,
            workload_by_id,
        )
        for case_id in manifest["expected_case_ids"]
    ]
    if recorded_case_rows != recomputed_case_rows:
        raise ValueError("case row disagreement")
    computed = _classify(manifest, recomputed_case_rows)
    recorded = _read_json(run_dir / "summary.json")
    if recorded != computed:
        raise ValueError("classification disagreement")
    report = _render_report(computed)
    if (run_dir / "report.md").read_text() != report:
        raise ValueError("report disagreement")
    if write_output:
        output_dir = run_dir / "independent-verify"
        output_dir.mkdir(exist_ok=True)
        _write_json(output_dir / "summary.json", computed)
        (output_dir / "report.md").write_text(report)
        (output_dir / "verify.stdout").write_text(
            json.dumps(computed, sort_keys=True) + "\n"
        )
        (output_dir / "verify.stderr").write_text("")
        (output_dir / "verify.exitcode").write_text("0\n")
    return computed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = args.run_dir / "independent-verify"
    output_dir.mkdir(exist_ok=True)
    try:
        result = verify_run(args.run_dir, write_output=True)
    except Exception as exc:
        (output_dir / "verify.stdout").write_text("")
        (output_dir / "verify.stderr").write_text(f"{exc}\n")
        (output_dir / "verify.exitcode").write_text("1\n")
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
