from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

import qwen35_tp4_hybrid_prefix_benchmark_contract as contract


class VerificationError(RuntimeError):
    pass


def _fail(message):
    raise VerificationError(message)


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _fail(f"failed to hash artifact: {error}")
    return digest.hexdigest()


def _load_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"failed to load JSON {Path(path).name}: {error}")
    _require_finite(value, Path(path).name)
    return value


def _load_jsonl(path):
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        _fail(f"failed to load JSONL {Path(path).name}: {error}")
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            _fail(
                f"invalid JSONL row {Path(path).name}:"
                f"{line_number}: {error}"
            )
        _require_finite(row, f"{Path(path).name}:{line_number}")
        rows.append(row)
    return rows


def _require_finite(value, label):
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"non-finite number in {label}")
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child, label)
    elif isinstance(value, list):
        for child in value:
            _require_finite(child, label)


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_json(path, value):
    _atomic_write(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8") + b"\n",
    )


def _safe_relative_file(run_dir, relative, label):
    if not isinstance(relative, str) or not relative:
        _fail(f"{label} path is invalid")
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        _fail(f"{label} path is unsafe")
    path = run_dir / value
    if not path.is_file() or path.is_symlink():
        _fail(f"{label} file is missing")
    return path


def _verify_final_inventory(run_dir):
    expected_files = set(contract.TOP_LEVEL_ARTIFACTS)
    optional_before_verification = {
        "independent_verification.json",
        "report.md",
    }
    expected_directories = set(contract.NESTED_ARTIFACT_DIRECTORIES)
    for entry in run_dir.iterdir():
        if entry.is_symlink():
            _fail(f"unexpected artifact symlink: {entry.name}")
        if entry.is_dir():
            if entry.name not in expected_directories:
                _fail(f"unexpected artifact directory: {entry.name}")
            continue
        if not entry.is_file():
            _fail(f"unexpected artifact type: {entry.name}")
        if entry.name not in expected_files:
            _fail(f"unexpected artifact: {entry.name}")
    present_files = {
        entry.name
        for entry in run_dir.iterdir()
        if entry.is_file()
    }
    required_before_verification = (
        expected_files - optional_before_verification
    )
    missing = sorted(required_before_verification - present_files)
    if missing:
        _fail(f"missing required artifact: {missing}")
    present_directories = {
        entry.name
        for entry in run_dir.iterdir()
        if entry.is_dir()
    }
    if present_directories != expected_directories:
        _fail(
            "nested artifact directory mismatch: "
            f"{sorted(present_directories)}"
        )


def _manifest_actual_domain(run_dir):
    paths = {
        relative: run_dir / relative
        for relative in contract.ARTIFACT_MANIFEST_HASH_DOMAIN
    }
    for directory in contract.NESTED_ARTIFACT_DIRECTORIES:
        root = run_dir / directory
        for path in root.rglob("*"):
            if path.is_symlink():
                _fail(
                    "unexpected artifact symlink: "
                    f"{path.relative_to(run_dir).as_posix()}"
                )
            if path.is_dir():
                continue
            if not path.is_file():
                _fail(
                    "unexpected nested artifact type: "
                    f"{path.relative_to(run_dir).as_posix()}"
                )
            paths[path.relative_to(run_dir).as_posix()] = path
    return paths


def _verify_artifact_manifest(run_dir):
    manifest = _load_json(run_dir / "artifact_manifest.json")
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "files"}
        or manifest["schema_version"] != contract.SCHEMA_VERSION
        or not isinstance(manifest["files"], dict)
    ):
        _fail("artifact manifest schema mismatch")
    actual = _manifest_actual_domain(run_dir)
    if set(manifest["files"]) != set(actual):
        _fail("artifact manifest domain mismatch")
    for relative, path in actual.items():
        row = manifest["files"][relative]
        if (
            not isinstance(row, dict)
            or set(row) != {"sha256", "size"}
            or row["size"] != path.stat().st_size
        ):
            _fail(f"artifact manifest metadata mismatch: {relative}")
        if _sha256(path) != row["sha256"]:
            _fail(f"artifact hash mismatch: {relative}")


def _verify_nested_inventory(run_dir):
    prerequisite_payload = _load_json(
        run_dir / "correctness_prerequisites.json"
    )
    expected_prerequisites = set()
    for name in (
        "tp4_root_logit",
        "cached_continuation",
        "engine_correctness",
    ):
        row = prerequisite_payload.get(name, {})
        expected_prerequisites.update({
            row.get("artifact_path"),
            row.get("independent_verification_path"),
            row.get("provenance_path"),
        })
        provenance_path = row.get("provenance_path")
        if isinstance(provenance_path, str):
            provenance = _load_json(run_dir / provenance_path)
            provenance_parent = Path(provenance_path).parent
            expected_prerequisites.update({
                (
                    provenance_parent
                    / provenance.get(field, "")
                ).as_posix()
                for field in (
                    "plan_path",
                    "authorization_path",
                    "receipt_path",
                )
            })
    logits_payload = _load_json(run_dir / "logits_manifest.json")
    log_payload = _load_json(run_dir / "worker_logs_manifest.json")
    expected_by_directory = {
        "prerequisites": expected_prerequisites,
        "logits": {
            row.get("path")
            for row in logits_payload.get("files", [])
            if isinstance(row, dict)
        },
        "logs": {
            row.get("path")
            for row in log_payload.get("files", [])
            if isinstance(row, dict)
        },
    }
    for directory, expected in expected_by_directory.items():
        if (
            None in expected
            or any(
                not isinstance(relative, str)
                or not relative.startswith(directory + "/")
                for relative in expected
            )
        ):
            _fail("nested artifact inventory reference mismatch")
        actual = {
            path.relative_to(run_dir).as_posix()
            for path in (run_dir / directory).rglob("*")
            if path.is_file()
        }
        if actual != expected:
            _fail(
                "nested artifact inventory mismatch: "
                f"{directory}"
            )


def _verify_prerequisites(run_dir):
    status = contract.validate_prerequisites(
        run_dir / "correctness_prerequisites.json"
    )
    if not status.authorized:
        _fail(
            "correctness prerequisites are not authorized: "
            + "; ".join(status.reasons)
        )


def _verify_workload_manifest(run_dir):
    path = run_dir / "workload_manifest.json"
    payload = _load_json(path)
    if payload != contract.workload_manifest_payload():
        _fail("workload manifest mismatch")
    return _sha256(path)


def _verify_source_manifest(run_dir):
    payload = _load_json(run_dir / "source_manifest.json")
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "schema_version",
            "source_tree_sha256",
            "model_manifest_sha256",
        }
        or payload["schema_version"] != contract.SCHEMA_VERSION
        or payload["model_manifest_sha256"]
        != contract.MODEL_MANIFEST_SHA256
        or not isinstance(payload["source_tree_sha256"], str)
        or len(payload["source_tree_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in payload["source_tree_sha256"]
        )
    ):
        _fail("source manifest mismatch")
    return payload["source_tree_sha256"]


def _verify_gpu_assignments(run_dir):
    payload = _load_json(run_dir / "gpu_assignments.json")
    rows = payload.get("assignments") if isinstance(payload, dict) else None
    resource_policy = payload.get(
        "resource_policy",
        "strict-exclusive",
    )
    maximum_utilization = payload.get(
        "maximum_gpu_utilization_percent",
    )
    if (
        payload.get("schema_version") != contract.SCHEMA_VERSION
        or not isinstance(rows, list)
        or len(rows) != contract.WORLD_SIZE
        or resource_policy not in {
            "strict-exclusive",
            "shared-low-utilization",
        }
        or (
            resource_policy == "strict-exclusive"
            and maximum_utilization is not None
        )
        or (
            resource_policy == "shared-low-utilization"
            and (
                isinstance(maximum_utilization, bool)
                or not isinstance(maximum_utilization, int)
                or not 0 <= maximum_utilization <= 100
            )
        )
    ):
        _fail("GPU assignment schema mismatch")
    ranks = []
    indices = []
    uuids = []
    for row in rows:
        if not isinstance(row, dict):
            _fail("GPU assignment row is invalid")
        ranks.append(row.get("rank"))
        indices.append(row.get("gpu_index"))
        uuids.append(row.get("gpu_uuid"))
        compute_processes = row.get("compute_processes")
        utilization = row.get("utilization_percent")
        if (
            row.get("free_bytes", -1) < contract.MIN_GPU_FREE_BYTES
            or not isinstance(compute_processes, list)
            or (
                resource_policy == "strict-exclusive"
                and compute_processes != []
            )
            or (
                resource_policy == "shared-low-utilization"
                and (
                    isinstance(utilization, bool)
                    or not isinstance(utilization, int)
                    or not 0 <= utilization <= maximum_utilization
                )
            )
        ):
            _fail("GPU assignment is not eligible")
    if (
        sorted(ranks) != list(range(contract.WORLD_SIZE))
        or len(set(indices)) != contract.WORLD_SIZE
        or len(set(uuids)) != contract.WORLD_SIZE
    ):
        _fail("GPU assignment identity mismatch")


def _verify_commands(run_dir, matrix):
    payload = _load_json(run_dir / "commands.json")
    commands = payload.get("commands") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != contract.SCHEMA_VERSION
        or not isinstance(commands, list)
        or len(commands) != len(matrix)
    ):
        _fail("command matrix mismatch")
    used_ports = set()
    for command, case in zip(commands, matrix):
        expected = {
            "case_id": case.case_id,
            "policy": case.policy,
            "workload": case.workload,
            "phase": case.phase,
            "repetition": case.repetition,
        }
        if any(command.get(name) != value for name, value in expected.items()):
            _fail("command order or identity mismatch")
        ports = (command.get("dist_port"), command.get("master_port"))
        if (
            any(
                isinstance(port, bool)
                or not isinstance(port, int)
                or port <= 0
                for port in ports
            )
            or ports[0] == ports[1]
            or any(port in used_ports for port in ports)
        ):
            _fail("command port identity mismatch")
        used_ports.update(ports)


def _verify_row_schema(rows, fields, label):
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(fields):
            _fail(f"{label} row schema mismatch")
        row_id = row.get("row_id", row.get("case_id"))
        if row_id in seen:
            _fail(f"duplicate {label} row ID")
        seen.add(row_id)


def _verify_process_matrix(process_rows, matrix):
    _verify_row_schema(
        process_rows,
        contract.PROCESS_ROW_FIELDS,
        "process",
    )
    expected_ids = [case.case_id for case in matrix]
    actual_ids = [row["case_id"] for row in process_rows]
    if actual_ids != expected_ids:
        _fail("process matrix mismatch")


def _verify_case_matrix(case_rows, matrix):
    _verify_row_schema(case_rows, contract.CASE_ROW_FIELDS, "case")
    expected = []
    for case in matrix:
        continuations = contract.WORKLOAD_SPECS[
            case.workload
        ]["continuations"]
        expected.extend(
            f"{case.case_id}__request-{index}"
            for index in range(continuations)
        )
    if [row["row_id"] for row in case_rows] != expected:
        _fail("case matrix mismatch")


def _verify_row_provenance(
    case_rows,
    *,
    source_tree_sha256,
    workload_sha,
    prerequisite_sha,
):
    for row in case_rows:
        if (
            row["source_tree_sha256"]
            != source_tree_sha256
            or row["model_manifest_sha256"]
            != contract.MODEL_MANIFEST_SHA256
            or row["workload_manifest_sha256"] != workload_sha
            or row["correctness_prerequisites_sha256"]
            != prerequisite_sha
        ):
            _fail("case row provenance mismatch")
        if (
            contract.canonical_json_sha256(
                row["output_token_ids"]
            )
            != row["output_token_ids_sha256"]
        ):
            _fail("output token hash mismatch")


def _load_logits(run_dir, row):
    relative = row["final_logits_path"]
    digest = row["final_logits_sha256"]
    if relative is None or digest is None:
        _fail("correctness logits missing")
    path = _safe_relative_file(run_dir, relative, "correctness logits")
    if _sha256(path) != digest:
        _fail("correctness logits hash mismatch")
    if path.suffix == ".json":
        values = _load_json(path)
        if not isinstance(values, list):
            _fail("correctness logits JSON is invalid")
        return [float(value) for value in values]
    try:
        import torch
        tensor = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        _fail(f"failed to load correctness logits: {error}")
    if not isinstance(tensor, torch.Tensor):
        _fail("correctness logits tensor is invalid")
    if not bool(torch.isfinite(tensor).all()):
        _fail("correctness logits are non-finite")
    return tensor.detach().float().reshape(-1).tolist()


def _verify_outputs_and_logits(run_dir, case_rows):
    pairs = defaultdict(dict)
    for row in case_rows:
        key = (
            row["workload"],
            row["phase"],
            row["repetition"],
            row["request_id"],
        )
        pairs[key][row["policy"]] = row
    for key, policies in pairs.items():
        if set(policies) != set(contract.POLICIES):
            _fail(f"policy pair missing for {key}")
        baseline = policies["recompute"]
        candidate = policies["exact_restore"]
        if baseline["output_token_ids"] != candidate["output_token_ids"]:
            _fail(f"output token mismatch for {key}")
        if key[1] == "correctness":
            baseline_logits = _load_logits(run_dir, baseline)
            candidate_logits = _load_logits(run_dir, candidate)
            if len(baseline_logits) != len(candidate_logits):
                _fail(f"correctness logits shape mismatch for {key}")
            for reference, observed in zip(
                baseline_logits,
                candidate_logits,
            ):
                tolerance = 2e-5 + 1e-5 * abs(reference)
                if abs(observed - reference) > tolerance:
                    _fail(f"correctness logits mismatch for {key}")
        elif any(
            row["final_logits_path"] is not None
            or row["final_logits_sha256"] is not None
            for row in (baseline, candidate)
        ):
            _fail("non-correctness row unexpectedly contains logits")


def _verify_restore_semantics(case_rows):
    for row in case_rows:
        if row["policy"] == "recompute":
            if (
                row["restored_hybrid_state"] is not False
                or row["reused_kv_tokens"] != 0
            ):
                _fail("recompute row restored hybrid state")
            continue
        expected_restore = row["workload"] in {
            "w1_medium_reuse",
            "w2_long_reuse",
            "w3_batched_fanout",
        }
        if bool(row["restored_hybrid_state"]) != expected_restore:
            _fail("exact-restore hit/miss semantics mismatch")
        expected_reused = (
            contract.WORKLOAD_SPECS[row["workload"]][
                "shared_prefix_tokens"
            ]
            if expected_restore
            else 0
        )
        if row["reused_kv_tokens"] != expected_reused:
            _fail("reused token accounting mismatch")


def _verify_request_accounting(case_rows):
    request_ids = defaultdict(set)
    for row in case_rows:
        spec = contract.WORKLOAD_SPECS[row["workload"]]
        key = (
            row["case_id"],
            row["policy"],
            row["phase"],
            row["repetition"],
        )
        request_id = row["request_id"]
        if (
            not isinstance(request_id, str)
            or not request_id
            or request_id in request_ids[key]
        ):
            _fail("request identity mismatch")
        request_ids[key].add(request_id)
        expected_prompt = (
            spec["shared_prefix_tokens"] + spec["suffix_tokens"]
        )
        if row["prompt_tokens"] != expected_prompt:
            _fail("prompt token accounting mismatch")
        restored = bool(row["restored_hybrid_state"])
        expected_prefill = (
            spec["suffix_tokens"] if restored else expected_prompt
        )
        if row["executed_prefill_tokens"] != expected_prefill:
            _fail("executed prefill accounting mismatch")
        if row["generated_tokens"] != spec["generated_tokens"]:
            _fail("generated token accounting mismatch")
        output_ids = row["output_token_ids"]
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != spec["generated_tokens"]
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                for token_id in output_ids
            )
        ):
            _fail("output token shape mismatch")
        decode_steps = row["decode_step_ns"]
        if (
            not isinstance(decode_steps, list)
            or len(decode_steps) != max(
                spec["generated_tokens"] - 1,
                0,
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in decode_steps
            )
        ):
            _fail("decode timing shape mismatch")
        for name in ("ttft_ns", "e2e_ns"):
            value = row[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                _fail(f"{name} is invalid")
        if row["e2e_ns"] < row["ttft_ns"]:
            _fail("request timing order mismatch")


def _verify_cache_accounting(process_rows):
    for row in process_rows:
        cache_fields = {
            name: row[name]
            for name in contract.PROCESS_ROW_FIELDS
            if name.startswith("hybrid_cache_")
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in cache_fields.values()
        ):
            _fail("cache counter is invalid")
        if row["policy"] == "recompute":
            if any(cache_fields.values()):
                _fail("recompute cache counters are nonzero")
            continue
        if (
            row["hybrid_cache_current_bytes"]
            > row["hybrid_cache_current_logical_bytes"]
            or row["hybrid_cache_deduplicated_bytes"]
            != (
                row["hybrid_cache_current_logical_bytes"]
                - row["hybrid_cache_current_bytes"]
            )
        ):
            _fail("cache accounting mismatch")
        if (
            row["hybrid_cache_current_bytes"]
            > 2 * 1024**3
            or row["hybrid_cache_peak_entries"] > 16
        ):
            _fail("cache limit exceeded")
        if (
            row["workload"]
            in {
                "w1_medium_reuse",
                "w2_long_reuse",
                "w3_batched_fanout",
            }
            and row["hybrid_cache_evictions"] != 0
        ):
            _fail("required workload cache eviction observed")


def _verify_capacity_parity(process_rows):
    by_case = defaultdict(dict)
    for row in process_rows:
        key = (row["workload"], row["phase"], row["repetition"])
        by_case[key][row["policy"]] = row
    for key, policies in by_case.items():
        if set(policies) != set(contract.POLICIES):
            _fail(f"process policy pair missing for {key}")
        baseline = policies["recompute"]
        candidate = policies["exact_restore"]
        if (
            baseline["scheduler_visible_kv_blocks"]
            != candidate["scheduler_visible_kv_blocks"]
            or baseline["kv_capacity_bytes"]
            != candidate["kv_capacity_bytes"]
        ):
            _fail("KV capacity mismatch")


def _verify_log_manifest(run_dir):
    payload = _load_json(run_dir / "worker_logs_manifest.json")
    rows = payload.get("files") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != contract.SCHEMA_VERSION
        or not isinstance(rows, list)
        or not rows
    ):
        _fail("worker log manifest mismatch")
    for row in rows:
        path = _safe_relative_file(
            run_dir,
            row.get("path"),
            "worker log",
        )
        if _sha256(path) != row.get("sha256"):
            _fail("worker log hash mismatch")
        text = path.read_text(encoding="utf-8", errors="replace")
        if "Traceback (most recent call last)" in text:
            _fail("worker log contains traceback")
        if "QWEN35_TP4_BENCHMARK_WORKER_COMPLETE" not in text:
            _fail("worker log completion marker missing")


def _verify_logits_manifest(run_dir, case_rows):
    payload = _load_json(run_dir / "logits_manifest.json")
    rows = payload.get("files") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != contract.SCHEMA_VERSION
        or not isinstance(rows, list)
    ):
        _fail("logits manifest mismatch")
    expected = {
        row["final_logits_path"]: row["final_logits_sha256"]
        for row in case_rows
        if row["phase"] == "correctness"
    }
    actual = {}
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "sha256"}
            or row["path"] in actual
        ):
            _fail("logits manifest row mismatch")
        actual[row["path"]] = row["sha256"]
    if actual != expected:
        _fail("logits manifest inventory mismatch")


def _median(values, label):
    if not values:
        _fail(f"missing samples for {label}")
    return float(statistics.median(values))


def _aggregate_metrics(case_rows, process_rows):
    measured_cases = [
        row for row in case_rows if row["phase"] == "measured"
    ]
    measured_processes = [
        row for row in process_rows if row["phase"] == "measured"
    ]
    reuse_workloads = {
        "w1_medium_reuse",
        "w2_long_reuse",
        "w3_batched_fanout",
    }
    workload_metrics = {}
    for workload in contract.WORKLOADS:
        by_policy = {
            policy: [
                row
                for row in measured_cases
                if row["workload"] == workload
                and row["policy"] == policy
            ]
            for policy in contract.POLICIES
        }
        baseline = by_policy["recompute"]
        candidate = by_policy["exact_restore"]
        baseline_ttft = _median(
            [row["ttft_ns"] for row in baseline],
            f"{workload} baseline TTFT",
        )
        candidate_ttft = _median(
            [row["ttft_ns"] for row in candidate],
            f"{workload} candidate TTFT",
        )
        baseline_e2e = _median(
            [row["e2e_ns"] for row in baseline],
            f"{workload} baseline E2E",
        )
        candidate_e2e = _median(
            [row["e2e_ns"] for row in candidate],
            f"{workload} candidate E2E",
        )
        baseline_decode = _median(
            [
                value
                for row in baseline
                for value in row["decode_step_ns"]
            ],
            f"{workload} baseline decode",
        )
        candidate_decode = _median(
            [
                value
                for row in candidate
                for value in row["decode_step_ns"]
            ],
            f"{workload} candidate decode",
        )
        ratios_by_repetition = []
        for repetition in range(contract.MEASURED_REPETITIONS):
            baseline_repetition = [
                row["ttft_ns"]
                for row in baseline
                if row["repetition"] == repetition
            ]
            candidate_repetition = [
                row["ttft_ns"]
                for row in candidate
                if row["repetition"] == repetition
            ]
            ratios_by_repetition.append(
                _median(
                    candidate_repetition,
                    f"{workload} candidate repetition",
                )
                / _median(
                    baseline_repetition,
                    f"{workload} baseline repetition",
                )
            )
        workload_metrics[workload] = {
            "median_ttft_ratio": candidate_ttft / baseline_ttft,
            "max_repetition_ttft_ratio": max(
                ratios_by_repetition
            ),
            "throughput_ratio": baseline_e2e / candidate_e2e,
            "median_e2e_ratio": candidate_e2e / baseline_e2e,
            "median_decode_ratio": (
                candidate_decode / baseline_decode
            ),
        }

    initializations = defaultdict(list)
    reserved = defaultdict(list)
    for row in measured_processes:
        initializations[row["policy"]].append(
            row["initialization_ns"]
        )
        reserved[row["policy"]].append(
            row["cuda_peak_reserved_bytes"]
        )
    measured_restore_rows = [
        row
        for row in measured_cases
        if row["policy"] == "exact_restore"
        and row["workload"] in reuse_workloads
    ]
    reused_tokens = sum(
        row["reused_kv_tokens"]
        for row in measured_restore_rows
    )
    saved_prefill_tokens = sum(
        row["prompt_tokens"] - row["executed_prefill_tokens"]
        for row in measured_restore_rows
    )
    restore_processes = [
        row
        for row in measured_processes
        if row["policy"] == "exact_restore"
        and row["workload"] in reuse_workloads
    ]
    physical_snapshot_bytes = sum(
        row["hybrid_cache_current_bytes"]
        for row in restore_processes
    )
    logical_snapshot_bytes = sum(
        row["hybrid_cache_current_logical_bytes"]
        for row in restore_processes
    )
    process_pairs = defaultdict(dict)
    for row in measured_processes:
        if row["workload"] not in reuse_workloads:
            continue
        key = (row["workload"], row["repetition"])
        process_pairs[key][row["policy"]] = row
    added_cuda_bytes = sum(
        max(
            0,
            pair["exact_restore"]["cuda_peak_reserved_bytes"]
            - pair["recompute"]["cuda_peak_reserved_bytes"],
        )
        for pair in process_pairs.values()
    )
    if (
        reused_tokens <= 0
        or saved_prefill_tokens <= 0
        or physical_snapshot_bytes <= 0
        or logical_snapshot_bytes < physical_snapshot_bytes
        or any(
            set(pair) != set(contract.POLICIES)
            for pair in process_pairs.values()
        )
    ):
        _fail("cache efficiency denominator is invalid")
    cache_efficiency = {
        "logical_to_physical_snapshot_ratio": (
            logical_snapshot_bytes / physical_snapshot_bytes
        ),
        "physical_snapshot_bytes_per_reused_token": (
            physical_snapshot_bytes / reused_tokens
        ),
        "added_cuda_bytes_per_reused_token": (
            added_cuda_bytes / reused_tokens
        ),
        "saved_prefill_tokens_per_physical_snapshot_byte": (
            saved_prefill_tokens / physical_snapshot_bytes
        ),
    }
    return {
        "prerequisites_pass": True,
        "eligible_gpu_count": contract.WORLD_SIZE,
        "evidence_complete": True,
        "measured_matrix_complete": True,
        "correctness_pass": True,
        "workloads": workload_metrics,
        "cache_efficiency": cache_efficiency,
        "initialization_ratio": (
            _median(
                initializations["exact_restore"],
                "candidate initialization",
            )
            / _median(
                initializations["recompute"],
                "baseline initialization",
            )
        ),
        "peak_cuda_reserved_ratio": (
            max(reserved["exact_restore"])
            / max(reserved["recompute"])
        ),
        "scheduler_visible_kv_capacity_equal": True,
        "kv_capacity_bytes_equal": True,
        "cache_accounting_valid": True,
        "cache_within_limits": True,
        "no_required_workload_evictions": True,
    }


def _render_report(result):
    lines = [
        "# Qwen3.5 TP4 Hybrid-Prefix Benchmark Verification",
        "",
        f"- Classification: `{result['classification']}`",
        f"- Case rows: `{result['case_rows']}`",
        f"- Process rows: `{result['process_rows']}`",
        "",
        "## Workloads",
        "",
    ]
    for workload in contract.WORKLOADS:
        row = result["workloads"][workload]
        lines.extend([
            f"### {workload}",
            "",
            f"- Median TTFT ratio: `{row['median_ttft_ratio']:.6f}`",
            f"- Throughput ratio: `{row['throughput_ratio']:.6f}`",
            f"- Median decode ratio: `{row['median_decode_ratio']:.6f}`",
            "",
        ])
    lines.extend([
        "## Cache Efficiency",
        "",
        (
            "- Logical/physical snapshot ratio: "
            f"`{result['cache_efficiency']['logical_to_physical_snapshot_ratio']:.6f}`"
        ),
        (
            "- Physical snapshot bytes/reused token: "
            f"`{result['cache_efficiency']['physical_snapshot_bytes_per_reused_token']:.6f}`"
        ),
        (
            "- Added CUDA bytes/reused token: "
            f"`{result['cache_efficiency']['added_cuda_bytes_per_reused_token']:.6f}`"
        ),
        (
            "- Saved prefill tokens/physical snapshot byte: "
            f"`{result['cache_efficiency']['saved_prefill_tokens_per_physical_snapshot_byte']:.6f}`"
        ),
        "",
        "## Claim Boundary",
        "",
        "This report verifies a synthetic artifact fixture only unless the "
        "source and remote evidence identify a real canonical run.",
        "",
    ])
    return "\n".join(lines)


def verify_run(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        _fail("run directory is missing")
    _verify_final_inventory(run_dir)
    _verify_artifact_manifest(run_dir)
    _verify_nested_inventory(run_dir)
    _verify_prerequisites(run_dir)
    workload_sha = _verify_workload_manifest(run_dir)
    source_tree_sha256 = _verify_source_manifest(run_dir)
    _verify_gpu_assignments(run_dir)
    matrix = contract.build_case_matrix()
    _verify_commands(run_dir, matrix)
    case_rows = _load_jsonl(run_dir / "case_rows.jsonl")
    process_rows = _load_jsonl(run_dir / "process_rows.jsonl")
    _verify_case_matrix(case_rows, matrix)
    _verify_process_matrix(process_rows, matrix)
    prerequisite_sha = _sha256(
        run_dir / "correctness_prerequisites.json"
    )
    _verify_row_provenance(
        case_rows,
        source_tree_sha256=source_tree_sha256,
        workload_sha=workload_sha,
        prerequisite_sha=prerequisite_sha,
    )
    _verify_outputs_and_logits(run_dir, case_rows)
    _verify_restore_semantics(case_rows)
    _verify_request_accounting(case_rows)
    _verify_cache_accounting(process_rows)
    _verify_capacity_parity(process_rows)
    _verify_logits_manifest(run_dir, case_rows)
    _verify_log_manifest(run_dir)
    metrics = _aggregate_metrics(case_rows, process_rows)
    classification = contract.classify_run(metrics)
    result = {
        "schema_version": contract.SCHEMA_VERSION,
        "classification": classification,
        "case_rows": len(case_rows),
        "process_rows": len(process_rows),
        "workloads": metrics["workloads"],
        "cache_efficiency": metrics["cache_efficiency"],
        "initialization_ratio": metrics["initialization_ratio"],
        "peak_cuda_reserved_ratio": (
            metrics["peak_cuda_reserved_ratio"]
        ),
    }
    _atomic_write_json(
        run_dir / "independent_verification.json",
        result,
    )
    _atomic_write(
        run_dir / "report.md",
        _render_report(result).encode("utf-8"),
    )
    return result
