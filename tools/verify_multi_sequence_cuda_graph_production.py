#!/usr/bin/env python3
"""Independently verify exact multi-sequence CUDA Graph production evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "tools" / "multi_sequence_cuda_graph_contract.py"
SPLIT_POLICY_PATH = (
    ROOT / "tinyvllm" / "engine" / "flash_attn_split_policy.py"
)

HASHED_PRODUCTION_FILES = (
    "environment.json",
    "diagnostic_binding.json",
    "dispatch_events.jsonl",
    "capture_events.jsonl",
    "request_metrics.jsonl",
    "model_step_metrics.jsonl",
    "memory_trace.jsonl",
    "correctness_rows.jsonl",
    "budget_fallback_rows.jsonl",
    "case_summaries.json",
    "summary.json",
)
EXPECTED_COMMAND = (
    "python",
    "tools/run_multi_sequence_cuda_graph_production_gate_remote.py",
    "local-contracts",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "multi_sequence_cuda_graph_production_contract",
    CONTRACT_PATH,
)
split_policy = _load_module(
    "multi_sequence_cuda_graph_production_split_policy",
    SPLIT_POLICY_PATH,
)


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    payload = path.read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"{path.name} lacks final newline")
    rows = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name} line {line_number} is invalid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(
                f"{path.name} line {line_number} is not an object"
            )
        rows.append(row)
    return rows


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_bytes(
        path,
        contract.canonical_json_bytes(value) + b"\n",
    )


def _index_unique(
    rows: list[dict],
    *,
    evidence_name: str,
) -> tuple[dict[str, dict], list[str]]:
    indexed = {}
    failures = []
    for row_index, row in enumerate(rows):
        row_id = row.get("row_id")
        if not isinstance(row_id, str) or not row_id:
            failures.append(
                f"{evidence_name}: row {row_index} missing row_id"
            )
            continue
        if row_id in indexed:
            failures.append(
                f"{evidence_name}: duplicate row_id {row_id}"
            )
            continue
        indexed[row_id] = row
    return indexed, failures


def _finite_positive(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires samples")
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _workload_identity(matrix) -> list[dict]:
    return [
        {
            "case_id": case.case_id,
            "workload": case.workload,
            "policy": case.policy,
            "repetition": case.repetition,
            "warmup": case.warmup,
            "policy_order": case.policy_order,
            "paired_order": list(case.paired_order),
        }
        for case in matrix
    ]


def _arrival_identity(request_rows: list[dict]) -> list[dict]:
    return [
        {
            "row_id": row["row_id"],
            "case_id": row["case_id"],
            "request_id": row["request_id"],
            "scheduled_arrival_ns": row["scheduled_arrival_ns"],
        }
        for row in request_rows
    ]


def _identity_from_fields(fields: dict):
    required = {
        "graph_batch_size",
        "active_batch_size",
        "page_table_width",
        "effective_num_splits",
        "flash_attn_version",
        "multi_processor_count",
        "num_query_heads",
        "num_kv_heads",
        "head_dim",
        "page_block_size",
        "max_seqlen_q",
    }
    if not isinstance(fields, dict) or set(fields) != required:
        raise ValueError("identity fields are incomplete")
    inputs = split_policy.FlashAttentionSplitInputs(
        batch_size=int(fields["active_batch_size"]),
        num_query_heads=int(fields["num_query_heads"]),
        num_kv_heads=int(fields["num_kv_heads"]),
        head_dim=int(fields["head_dim"]),
        page_block_size=int(fields["page_block_size"]),
        page_table_width=int(fields["page_table_width"]),
        max_seqlen_q=int(fields["max_seqlen_q"]),
        multi_processor_count=int(fields["multi_processor_count"]),
    )
    identity = split_policy.build_flash_attn_263_graph_identity(
        graph_batch_size=int(fields["graph_batch_size"]),
        inputs=inputs,
        flash_attn_version=str(fields["flash_attn_version"]),
        require_exact_batch=True,
    )
    if identity.effective_num_splits != int(
        fields["effective_num_splits"]
    ):
        raise ValueError("identity effective split mismatch")
    return identity


def _validate_required_files(run_dir: Path) -> list[str]:
    return [
        f"missing required file: {name}"
        for name in contract.PRODUCTION_ARTIFACT_FILES
        if not (run_dir / name).is_file()
    ]


def _validate_manifest(
    run_dir: Path,
    manifest: dict,
    environment: dict,
    source_manifest: dict,
    request_rows: list[dict],
    diagnostic_binding: dict,
) -> list[str]:
    failures = []
    if set(manifest) != set(contract.PRODUCTION_MANIFEST_FIELDS):
        failures.append("manifest fields disagree with frozen contract")
    mode = manifest.get("mode")
    if mode in {"correctness-smoke", "arrival-smoke"}:
        matrix = contract.build_production_smoke_matrix()
    else:
        matrix = contract.build_production_matrix()
    expected_case_ids = [case.case_id for case in matrix]
    if manifest.get("case_ids") != expected_case_ids:
        failures.append("manifest case_ids order or domain mismatch")
    if manifest.get("schema_version") != 1:
        failures.append("manifest schema_version mismatch")
    if mode not in {
        "correctness-smoke",
        "arrival-smoke",
        "correctness-canonical",
        "arrival-canonical",
    }:
        failures.append("manifest mode is unsupported")
    if manifest.get("thresholds") != contract.PRODUCTION_THRESHOLDS:
        failures.append("manifest thresholds mismatch")
    if manifest.get("commands") != list(EXPECTED_COMMAND):
        failures.append("manifest command mismatch")
    if (
        manifest.get("source_tree_sha256")
        != source_manifest.get("source_tree_sha256")
        or environment.get("source_tree_sha256")
        != manifest.get("source_tree_sha256")
    ):
        failures.append("source tree identity mismatch")
    if manifest.get("copied_file_sha256") != source_manifest.get("files"):
        failures.append("copied file hashes disagree")
    if manifest.get("diagnostic_binding_sha256") != contract.sha256_file(
        run_dir / "diagnostic_binding.json"
    ):
        failures.append("diagnostic binding hash mismatch")
    if manifest.get("budget_fallback_sha256") != contract.sha256_file(
        run_dir / "budget_fallback_rows.jsonl"
    ):
        failures.append("budget fallback hash mismatch")
    expected_diagnostic = {
        "classification": "EXACT_REPLAY_CORRECT",
        "rounded_classification": "ROUNDED_REPLAY_CORRUPT",
        "legacy_compatibility": "LEGACY_COMPATIBLE",
        "policy_integrity": "POLICY_EXACT",
    }
    canonical = mode in {
        "correctness-canonical",
        "arrival-canonical",
    }
    if diagnostic_binding.get("required") is not canonical:
        failures.append("diagnostic binding requirement mismatch")
    if canonical and diagnostic_binding.get("case_count") != 315:
        failures.append("canonical diagnostic case count mismatch")
    if canonical and diagnostic_binding.get(
        "classifications"
    ) != expected_diagnostic:
        failures.append("canonical diagnostic classifications mismatch")
    if not canonical and (
        diagnostic_binding.get("case_count") != 0
        or diagnostic_binding.get("classifications") is not None
    ):
        failures.append("smoke diagnostic binding must be empty")
    if diagnostic_binding.get("source_tree_sha256") != manifest.get(
        "source_tree_sha256"
    ):
        failures.append("canonical diagnostic source mismatch")
    if manifest.get("model_sha256") != contract.canonical_json_sha256(
        environment.get("model")
    ):
        failures.append("model hash mismatch")
    if manifest.get("config_sha256") != contract.canonical_json_sha256(
        manifest.get("policy_configs")
    ):
        failures.append("config hash mismatch")
    if manifest.get("workload_sha256") != contract.canonical_json_sha256(
        _workload_identity(matrix)
    ):
        failures.append("workload hash mismatch")
    try:
        arrival_identity = _arrival_identity(request_rows)
    except KeyError:
        failures.append("arrival identity fields missing")
    else:
        if (
            manifest.get("arrival_sha256")
            != contract.canonical_json_sha256(arrival_identity)
        ):
            failures.append("arrival hash mismatch")

    expected_orders = [
        {
            "workload": case.workload,
            "repetition": case.repetition,
            "paired_order": list(case.paired_order),
        }
        for case in matrix
        if case.policy_order == 0
    ]
    if manifest.get("paired_policy_order") != expected_orders:
        failures.append("paired policy order mismatch")

    process_rows = manifest.get("processes")
    if not isinstance(process_rows, list):
        failures.append("manifest processes missing")
        process_rows = []
    if len(process_rows) != len(matrix):
        failures.append("manifest process count mismatch")
    process_case_ids = [
        row.get("case_id")
        for row in process_rows
        if isinstance(row, dict)
    ]
    if process_case_ids != expected_case_ids:
        failures.append("manifest process order mismatch")
    used_ports = set()
    for process in process_rows:
        if not isinstance(process, dict):
            failures.append("manifest process row invalid")
            continue
        command = process.get("command")
        case_id = process.get("case_id")
        if command != [
            *EXPECTED_COMMAND,
            "--case-id",
            case_id,
        ]:
            failures.append(f"{case_id}: process command mismatch")
        if process.get("source_sha256") != manifest.get(
            "source_tree_sha256"
        ):
            failures.append(f"{case_id}: process source mismatch")
        for field in ("tinyvllm_dist_port", "master_port"):
            port = process.get(field)
            if not isinstance(port, int) or port <= 0:
                failures.append(f"{case_id}: invalid {field}")
            elif port in used_ports:
                failures.append(f"duplicate process port {port}")
            else:
                used_ports.add(port)
        if process.get("tinyvllm_dist_port") == process.get("master_port"):
            failures.append(f"{case_id}: identical process ports")
    if not isinstance(manifest.get("ports"), dict):
        failures.append("manifest ports missing")

    capacity = manifest.get("capacity")
    if not isinstance(capacity, dict):
        failures.append("capacity contract missing")
    else:
        baseline = capacity.get("baseline", {})
        candidate = capacity.get("candidate", {})
        if (
            baseline.get("scheduler_visible_blocks")
            != candidate.get("scheduler_visible_blocks")
        ):
            failures.append("scheduler-visible capacity mismatch")
        if candidate.get("physical_blocks") != (
            candidate.get("scheduler_visible_blocks", -1)
            + candidate.get("scratch_blocks", -1)
        ):
            failures.append("candidate physical capacity mismatch")
    return failures


def _validate_file_hashes(
    run_dir: Path,
    source_manifest: dict,
) -> list[str]:
    failures = []
    recorded = source_manifest.get("artifact_sha256")
    if not isinstance(recorded, dict):
        return ["source manifest artifact hashes missing"]
    if set(recorded) != set(HASHED_PRODUCTION_FILES):
        failures.append("source manifest artifact hash domain mismatch")
    for name in HASHED_PRODUCTION_FILES:
        path = run_dir / name
        if not path.is_file():
            continue
        expected = recorded.get(name)
        actual = contract.sha256_file(path)
        if expected != actual:
            failures.append(f"{name}: artifact hash mismatch")
    return failures


def _validate_source_rows(
    row_sets: tuple[tuple[str, list[dict]], ...],
    source_sha256: str,
) -> list[str]:
    failures = []
    for evidence_name, rows in row_sets:
        for row in rows:
            if row.get("source_sha256") != source_sha256:
                failures.append(
                    f"{evidence_name}: mixed source SHA"
                )
                break
    return failures


def _validate_identity_lifecycle(
    dispatch_rows: list[dict],
    capture_rows: list[dict],
) -> list[str]:
    failures = []
    identity_fields_by_sha = {}
    captures_by_case_sha = {}
    for row in capture_rows:
        try:
            identity = _identity_from_fields(row.get("identity_fields"))
        except (TypeError, ValueError) as exc:
            failures.append(
                f"{row.get('row_id')}: invalid capture identity: {exc}"
            )
            continue
        identity_sha = row.get("graph_identity_sha256")
        if identity_sha != identity.sha256:
            failures.append(
                f"{row.get('row_id')}: capture identity SHA mismatch"
            )
        canonical_fields = contract.canonical_json_sha256(
            row["identity_fields"]
        )
        existing = identity_fields_by_sha.setdefault(
            identity_sha,
            canonical_fields,
        )
        if existing != canonical_fields:
            failures.append("identity SHA shared by incompatible tensors")
        if row.get("observation_count") != 3:
            failures.append(f"{row.get('row_id')}: capture count mismatch")
        status = row.get("status")
        if status == "ready":
            key = (row.get("case_id"), identity_sha)
            if key in captures_by_case_sha:
                failures.append(f"duplicate capture lifecycle {key}")
            captures_by_case_sha[key] = row
            if row.get("budget_overshoot") is not False:
                failures.append(
                    f"{row.get('row_id')}: capture succeeded after overshoot"
                )
            if row.get("fallback_reason") is not None:
                failures.append(
                    f"{row.get('row_id')}: ready capture has fallback"
                )
        elif status == "rejected":
            if row.get("fallback_reason") not in (
                contract.BUDGET_FALLBACK_REASONS
            ):
                failures.append(
                    f"{row.get('row_id')}: rejected capture reason invalid"
                )
        else:
            failures.append(f"{row.get('row_id')}: capture status invalid")
        for field in (
            "capture_duration_ns",
            "static_bytes",
            "reserved_delta_bytes",
        ):
            try:
                _finite_positive(row.get(field), field)
            except ValueError as exc:
                failures.append(f"{row.get('row_id')}: {exc}")

    for row in dispatch_rows:
        fallback_reason = row.get("fallback_reason")
        if (
            fallback_reason is not None
            and fallback_reason not in contract.FALLBACK_REASONS
        ):
            failures.append(f"{row.get('row_id')}: unknown fallback reason")
        try:
            identity = _identity_from_fields(row.get("identity_fields"))
        except (TypeError, ValueError) as exc:
            failures.append(
                f"{row.get('row_id')}: invalid dispatch identity: {exc}"
            )
            continue
        identity_sha = row.get("graph_identity_sha256")
        if identity_sha != identity.sha256:
            failures.append(
                f"{row.get('row_id')}: dispatch identity SHA mismatch"
            )
        canonical_fields = contract.canonical_json_sha256(
            row["identity_fields"]
        )
        existing = identity_fields_by_sha.setdefault(
            identity_sha,
            canonical_fields,
        )
        if existing != canonical_fields:
            failures.append("identity SHA shared by incompatible tensors")
        if row.get("dispatch") == "graph":
            if row.get("cache_state") != "ready":
                failures.append(
                    f"{row.get('row_id')}: replay after rejection"
                )
            capture = captures_by_case_sha.get(
                (row.get("case_id"), identity_sha)
            )
            if capture is None:
                failures.append(
                    f"{row.get('row_id')}: graph replay lacks capture"
                )
            else:
                if int(capture.get("step_id", -1)) >= int(
                    row.get("step_id", -1)
                ):
                    failures.append(
                        f"{row.get('row_id')}: replay precedes capture"
                    )
                if row.get("observation_count") < capture.get(
                    "observation_count",
                    0,
                ):
                    failures.append(
                        f"{row.get('row_id')}: observation count regressed"
                    )
            if identity.graph_batch_size != identity.active_batch_size:
                failures.append(
                    f"{row.get('row_id')}: rounded replay"
                )
    return failures


def _validate_budget_fallback_rows(
    *,
    mode: str,
    manifest: dict,
    rows: list[dict],
    dispatch_rows: list[dict],
    capture_rows: list[dict],
    correctness_rows: list[dict],
) -> tuple[list[str], dict]:
    del mode, correctness_rows
    failures = []
    required_reasons = list(contract.BUDGET_FALLBACK_REASONS)
    observed_reasons = [
        row.get("reason") for row in rows
        if isinstance(row, dict)
    ]
    if observed_reasons != required_reasons:
        failures.append("budget fallback reason domain mismatch")

    dispatch_by_id, dispatch_index_failures = _index_unique(
        dispatch_rows,
        evidence_name="dispatch_events",
    )
    capture_by_id, capture_index_failures = _index_unique(
        capture_rows,
        evidence_name="capture_events",
    )
    failures.extend(dispatch_index_failures)
    failures.extend(capture_index_failures)
    manifest_source = manifest.get("source_tree_sha256")
    used_ports = set()
    for process in manifest.get("processes", []):
        if not isinstance(process, dict):
            continue
        for field in ("tinyvllm_dist_port", "master_port"):
            value = process.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                used_ports.add(value)

    verified_reasons = []
    runtime_reasons = {
        "scratch_unavailable",
        "capture_failed",
        "identity_drift",
    }
    capture_stage_reasons = {
        "reserved_byte_budget",
        "single_capture_budget",
        "total_capture_budget",
        *runtime_reasons,
    }
    for row_index, row in enumerate(rows):
        row_id = row.get("row_id") if isinstance(row, dict) else None
        row_id = row_id or f"budget-fallback-row-{row_index}"
        row_failures_before = len(failures)
        if not isinstance(row, dict):
            failures.append(f"{row_id}: budget fallback row invalid")
            continue
        if set(row) != set(contract.BUDGET_FALLBACK_ROW_FIELDS):
            failures.append(f"{row_id}: budget fallback fields mismatch")
        reason = row.get("reason")
        expected_case_id = f"budget-fallback:{reason}"
        if row.get("case_id") != expected_case_id:
            failures.append(f"{row_id}: budget fallback case mismatch")
        if row.get("source_sha256") != manifest_source:
            failures.append("budget_fallback_rows: mixed source SHA")
        if row.get("gpu") != 0:
            failures.append(f"{row_id}: budget fallback GPU mismatch")
        for port_field in ("tinyvllm_dist_port", "master_port"):
            port = row.get(port_field)
            if (
                isinstance(port, bool)
                or not isinstance(port, int)
                or port <= 0
            ):
                failures.append(f"{row_id}: invalid worker port")
            elif port in used_ports:
                failures.append(f"{row_id}: reused worker port")
            else:
                used_ports.add(port)
        if row.get("tinyvllm_dist_port") == row.get("master_port"):
            failures.append(f"{row_id}: identical worker ports")

        try:
            identity = _identity_from_fields(
                row.get("target_identity_fields")
            )
        except (TypeError, ValueError) as exc:
            failures.append(f"{row_id}: invalid target identity: {exc}")
            identity = None
        if (
            identity is not None
            and row.get("target_identity_sha256") != identity.sha256
        ):
            failures.append(f"{row_id}: target identity SHA mismatch")
        identity_sha = row.get("target_identity_sha256")
        case_id = row.get("case_id")

        def referenced_rows(field, index, evidence_name):
            row_ids = row.get(field)
            if not isinstance(row_ids, list):
                failures.append(f"{row_id}: {field} is not a list")
                return []
            referenced = []
            for referenced_id in row_ids:
                target = index.get(referenced_id)
                if target is None:
                    failures.append(
                        f"{row_id}: missing referenced {evidence_name} row"
                    )
                    continue
                if (
                    target.get("case_id") != case_id
                    or target.get("source_sha256") != manifest_source
                    or target.get("graph_identity_sha256") != identity_sha
                ):
                    failures.append(
                        f"{row_id}: referenced {evidence_name} row mismatch"
                    )
                referenced.append(target)
            return referenced

        observations = referenced_rows(
            "observation_dispatch_row_ids",
            dispatch_by_id,
            "dispatch",
        )
        terminals = referenced_rows(
            "terminal_dispatch_row_ids",
            dispatch_by_id,
            "dispatch",
        )
        captures = referenced_rows(
            "capture_row_ids",
            capture_by_id,
            "capture",
        )
        if len(observations) != 2:
            failures.append(f"{row_id}: cold observation count mismatch")
        observation_steps = []
        for observation in observations:
            observation_steps.append(observation.get("step_id"))
            if (
                observation.get("dispatch") != "eager"
                or observation.get("cache_state") != "observing"
                or observation.get("fallback_reason") != "cold_identity"
            ):
                failures.append(
                    f"{row_id}: cold observation lifecycle mismatch"
                )
        if len(terminals) < 3:
            failures.append(f"{row_id}: terminal evidence count mismatch")
        terminal_steps = []
        for terminal in terminals:
            terminal_steps.append(terminal.get("step_id"))
            if terminal.get("dispatch") == "graph":
                failures.append(f"{row_id}: graph replay after rejection")
            if terminal.get("fallback_reason") != reason:
                failures.append(
                    f"{row_id}: terminal dispatch reason mismatch"
                )
            if terminal.get("cache_state") != "rejected":
                failures.append(f"{row_id}: terminal cache state mismatch")
        if (
            observation_steps
            and terminal_steps
            and max(observation_steps) >= min(terminal_steps)
        ):
            failures.append(f"{row_id}: terminal ordering mismatch")

        first_terminal_step = (
            min(terminal_steps)
            if terminal_steps
            and all(isinstance(value, int) for value in terminal_steps)
            else None
        )
        case_target_dispatch = [
            item for item in dispatch_rows
            if item.get("case_id") == case_id
            and item.get("graph_identity_sha256") == identity_sha
        ]
        if any(
            item.get("dispatch") == "graph"
            and (
                first_terminal_step is None
                or item.get("step_id", -1) >= first_terminal_step
            )
            for item in case_target_dispatch
        ):
            failures.append(f"{row_id}: graph replay after rejection")
        if any(
            item.get("capture_attempted") is True
            and first_terminal_step is not None
            and item.get("step_id", -1) > first_terminal_step
            for item in case_target_dispatch
        ):
            failures.append(f"{row_id}: capture after rejection")

        target_capture_attempts = sum(
            item.get("capture_attempted") is True
            and item.get("step_id") == first_terminal_step
            for item in case_target_dispatch
        )
        post_rejection_attempts = sum(
            item.get("capture_attempted") is True
            and first_terminal_step is not None
            and item.get("step_id", -1) > first_terminal_step
            for item in case_target_dispatch
        )
        graph_replays = sum(
            item.get("dispatch") == "graph"
            for item in case_target_dispatch
        )
        if row.get("target_capture_attempt_count") != target_capture_attempts:
            failures.append(f"{row_id}: target capture count mismatch")
        if (
            row.get("post_rejection_capture_attempt_count")
            != post_rejection_attempts
        ):
            failures.append(
                f"{row_id}: post-rejection capture count mismatch"
            )
        if row.get("target_graph_replay_count") != graph_replays:
            failures.append(f"{row_id}: target graph replay count mismatch")
        if row.get("terminal_rejection_reason") != reason:
            failures.append(f"{row_id}: terminal rejection reason mismatch")
        if (
            row.get("eager_output_token_ids")
            != row.get("candidate_output_token_ids")
        ):
            failures.append(f"{row_id}: output token mismatch")
        if row.get("logits_allclose") is not True:
            failures.append(f"{row_id}: logits mismatch")
        logits_max_abs_diff = row.get("logits_max_abs_diff")
        if (
            isinstance(logits_max_abs_diff, bool)
            or not isinstance(logits_max_abs_diff, (int, float))
            or not math.isfinite(float(logits_max_abs_diff))
            or float(logits_max_abs_diff) < 0.0
            or float(logits_max_abs_diff) > contract.LOGIT_ATOL
        ):
            failures.append(f"{row_id}: logits maximum difference invalid")
        if (
            row.get("eager_live_kv_sha256")
            != row.get("candidate_live_kv_sha256")
        ):
            failures.append(f"{row_id}: live KV mismatch")
        if row.get("complete") is not True:
            failures.append(f"{row_id}: budget fallback incomplete")

        runtime_fault = reason in runtime_reasons
        expected_injection_class = (
            "runtime_fault" if runtime_fault else "budget_precondition"
        )
        if row.get("injection_class") != expected_injection_class:
            failures.append(f"{row_id}: injection class mismatch")
        if runtime_fault and (
            row.get("injection_installed") is not True
            or row.get("injection_restored") is not True
        ):
            failures.append(f"{row_id}: runtime injection not restored")
        if not runtime_fault and (
            row.get("injection_installed") is not False
            or row.get("injection_restored") is not False
        ):
            failures.append(f"{row_id}: unexpected runtime injection")

        config = row.get("effective_cache_config")
        summary = row.get("pre_target_cache_summary")
        if not isinstance(config, dict) or not isinstance(summary, dict):
            failures.append(f"{row_id}: budget precondition missing")
            config = {}
            summary = {}
        try:
            if reason == "entry_limit" and not (
                len(summary.get("ready_entries", []))
                >= int(config["max_entries"])
            ):
                failures.append(f"{row_id}: entry limit is not binding")
            elif reason == "static_byte_budget" and not (
                int(summary["static_bytes"])
                + int(config["target_estimated_static_bytes"])
                > int(config["max_static_bytes"])
            ):
                failures.append(
                    f"{row_id}: static byte budget is not binding"
                )
            elif reason == "reserved_byte_budget" and not (
                int(summary["reserved_delta_bytes"])
                + int(config["target_reserved_delta_bytes"])
                > int(config["max_reserved_bytes"])
            ):
                failures.append(
                    f"{row_id}: reserved byte budget is not binding"
                )
            elif reason == "single_capture_budget" and not (
                int(config["target_capture_duration_ns"])
                > int(config["max_single_capture_ns"])
            ):
                failures.append(
                    f"{row_id}: single capture budget is not binding"
                )
            elif reason == "total_capture_budget" and not (
                int(summary["total_capture_ns"])
                + int(config["target_capture_duration_ns"])
                > int(config["max_total_capture_ns"])
            ):
                failures.append(
                    f"{row_id}: total capture budget is not binding"
                )
        except (KeyError, TypeError, ValueError):
            failures.append(f"{row_id}: budget precondition invalid")

        expected_capture_count = 1 if reason in capture_stage_reasons else 0
        if len(captures) != expected_capture_count:
            failures.append(f"{row_id}: capture evidence count mismatch")
        for capture in captures:
            if (
                capture.get("status") != "rejected"
                or capture.get("fallback_reason") != reason
                or capture.get("step_id") != first_terminal_step
            ):
                failures.append(f"{row_id}: rejected capture mismatch")
        if len(failures) == row_failures_before:
            verified_reasons.append(reason)

    return failures, {
        "budget_fallback_required": len(required_reasons),
        "budget_fallback_verified": len(verified_reasons),
        "budget_fallback_reasons": verified_reasons,
    }


def _reconstruct_case_rows(
    matrix,
    manifest: dict,
    dispatch_rows: list[dict],
    request_rows: list[dict],
    model_step_rows: list[dict],
    memory_rows: list[dict],
    correctness_rows: list[dict],
    producer_case_rows: list[dict],
) -> tuple[list[dict], list[str]]:
    failures = []
    producer_by_id = {}
    for row in producer_case_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in producer_by_id:
            failures.append("case_summaries case_id invalid or duplicate")
            continue
        producer_by_id[case_id] = row
    expected_ids = [case.case_id for case in matrix]
    if list(producer_by_id) != expected_ids:
        failures.append("case_summaries order or domain mismatch")

    def rows_for(rows, case_id):
        return [row for row in rows if row.get("case_id") == case_id]

    reconstructed = []
    capacity = manifest["capacity"]
    for case in matrix:
        case_id = case.case_id
        requests = rows_for(request_rows, case_id)
        model_steps = rows_for(model_step_rows, case_id)
        memories = rows_for(memory_rows, case_id)
        correctness = rows_for(correctness_rows, case_id)
        dispatch = rows_for(dispatch_rows, case_id)
        if (
            not requests
            or len(model_steps) != 1
            or not memories
            or len(correctness) != 1
        ):
            failures.append(f"{case_id}: raw evidence cardinality mismatch")
            continue
        model_step = model_steps[0]
        correctness_row = correctness[0]
        try:
            duration_ns = _finite_positive(
                model_step.get("measurement_duration_ns"),
                "measurement_duration_ns",
            )
            decode_duration_ns = _finite_positive(
                model_step.get("decode_duration_ns"),
                "decode_duration_ns",
            )
            decoded_tokens = _finite_positive(
                model_step.get("decoded_tokens"),
                "decoded_tokens",
            )
            initialization_ns = _finite_positive(
                model_step.get("initialization_duration_ns"),
                "initialization_duration_ns",
            )
            reserved = max(
                _finite_positive(
                    row.get("reserved_bytes"),
                    "reserved_bytes",
                )
                for row in memories
            )
            itl_values = [
                _finite_positive(value, "itl_ns")
                for row in requests
                for value in row.get("itl_ns", [])
            ]
            request_throughput = (
                len(requests) / (duration_ns / 1_000_000_000.0)
            )
            decode_throughput = (
                decoded_tokens
                / (decode_duration_ns / 1_000_000_000.0)
            )
            p95_itl = _nearest_rank(itl_values, 0.95)
            p99_itl = _nearest_rank(itl_values, 0.99)
        except (TypeError, ValueError) as exc:
            failures.append(f"{case_id}: invalid metric evidence: {exc}")
            continue
        output_match = (
            correctness_row.get("output_token_ids")
            == correctness_row.get("reference_token_ids")
            and correctness_row.get("logits_close") is True
            and correctness_row.get("live_slot_kv_sha256")
            == correctness_row.get(
                "reference_live_slot_kv_sha256"
            )
        )
        graph_events = [
            row for row in dispatch
            if row.get("dispatch") == "graph"
        ]
        fallback_events = [
            row for row in dispatch
            if row.get("dispatch") == "eager"
            and row.get("fallback_reason") is not None
        ]
        dispatch_contract_rows = []
        for row in graph_events + fallback_events:
            try:
                rebuilt_identity_sha256 = _identity_from_fields(
                    row.get("identity_fields")
                ).sha256
            except (TypeError, ValueError) as exc:
                failures.append(
                    f"{row.get('row_id')}: cannot rebuild identity: {exc}"
                )
                rebuilt_identity_sha256 = None
            dispatch_contract_rows.append({
                "dispatch": row.get("dispatch"),
                "graph_identity_sha256": row.get(
                    "graph_identity_sha256"
                ),
                "rebuilt_identity_sha256": rebuilt_identity_sha256,
                "page_table_width": row.get("page_table_width"),
                "active_batch_size": row.get("active_batch_size"),
                "fallback_reason": row.get("fallback_reason"),
                "cache_state": row.get("cache_state"),
            })
        reconstructed_row = {
            "case_id": case_id,
            "workload": case.workload,
            "policy": case.policy,
            "repetition": case.repetition,
            "warmup": case.warmup,
            "policy_order": case.policy_order,
            "paired_order": list(case.paired_order),
            "status": "PASS",
            "output_match": output_match,
            "capacity_snapshot": {
                "scheduler_visible_blocks": capacity[case.policy][
                    "scheduler_visible_blocks"
                ],
            },
            "request_throughput_rps": request_throughput,
            "decode_throughput_tps": decode_throughput,
            "p95_itl_ns": p95_itl,
            "p99_itl_ns": p99_itl,
            "peak_reserved_bytes": reserved,
            "initialization_duration_ns": initialization_ns,
            "dispatch_events": dispatch_contract_rows,
            "capture_events": [],
            "replay_after_rejection": any(
                row.get("dispatch") == "graph"
                and row.get("cache_state") != "ready"
                for row in dispatch
            ),
            "graph_hits": len(graph_events),
            "graph_eligible_steps": int(
                model_step.get("graph_eligible_steps", 0)
            ),
        }
        reconstructed.append(reconstructed_row)

        producer = producer_by_id.get(case_id)
        if producer is None:
            continue
        comparisons = {
            "request_throughput_rps": request_throughput,
            "decode_throughput_tps": decode_throughput,
            "p95_itl_ns": p95_itl,
            "p99_itl_ns": p99_itl,
            "peak_reserved_bytes": reserved,
            "initialization_duration_ns": initialization_ns,
        }
        for field, expected in comparisons.items():
            value = producer.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not math.isclose(
                    float(value),
                    float(expected),
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            ):
                failures.append(
                    f"{case_id}: producer {field} mismatch"
                )
        if producer.get("output_match") is not output_match:
            failures.append(f"{case_id}: producer correctness mismatch")
    return reconstructed, failures


def _validate_smoke_requirements(
    case_summaries: list[dict],
) -> list[str]:
    failures = []
    if any(row.get("output_match") is not True for row in case_summaries):
        failures.append("smoke correctness mismatch")
    if not any(
        int(row.get("graph_hits", 0)) > 0
        for row in case_summaries
        if row.get("policy") == "candidate"
    ):
        failures.append("missing exact graph hit")
    if not any(
        event.get("dispatch") == "eager"
        and event.get("fallback_reason") == "batch_not_allowlisted"
        for row in case_summaries
        if row.get("policy") == "candidate"
        for event in row.get("dispatch_events", [])
    ):
        failures.append("missing non-allowlisted eager fallback")
    return failures


def _render_report(result: dict) -> str:
    lines = [
        "# Exact CUDA Graph Production Verification",
        "",
        f"- Classification: `{result['classification']}`",
        f"- Failures: {len(result.get('failures', []))}",
        "",
        "## Metrics",
        "",
    ]
    for name, value in sorted(result.get("metrics", {}).items()):
        lines.append(f"- `{name}`: {value}")
    if result.get("failures"):
        lines.extend(["", "## Failures", ""])
        lines.extend(
            f"- {failure}" for failure in result["failures"]
        )
    return "\n".join(lines) + "\n"


def verify_run(run_dir: Path, *, write_report: bool = True) -> dict:
    run_dir = Path(run_dir)
    failures = _validate_required_files(run_dir)
    if failures:
        result = {
            "classification": "INCOMPLETE",
            "failures": failures,
            "metrics": {},
            "thresholds": dict(contract.PRODUCTION_THRESHOLDS),
        }
    else:
        try:
            manifest = _read_json(run_dir / "manifest.json")
            environment = _read_json(run_dir / "environment.json")
            source_manifest = _read_json(
                run_dir / "source_manifest.json"
            )
            diagnostic_binding = _read_json(
                run_dir / "diagnostic_binding.json"
            )
            producer_summary = _read_json(run_dir / "summary.json")
            producer_case_rows = _read_json(
                run_dir / "case_summaries.json"
            )
            dispatch_rows = _read_jsonl(
                run_dir / "dispatch_events.jsonl"
            )
            capture_rows = _read_jsonl(
                run_dir / "capture_events.jsonl"
            )
            request_rows = _read_jsonl(
                run_dir / "request_metrics.jsonl"
            )
            model_step_rows = _read_jsonl(
                run_dir / "model_step_metrics.jsonl"
            )
            memory_rows = _read_jsonl(
                run_dir / "memory_trace.jsonl"
            )
            correctness_rows = _read_jsonl(
                run_dir / "correctness_rows.jsonl"
            )
            budget_fallback_rows = _read_jsonl(
                run_dir / "budget_fallback_rows.jsonl"
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result = {
                "classification": "INCOMPLETE",
                "failures": [f"artifact parse failure: {exc}"],
                "metrics": {},
                "thresholds": dict(contract.PRODUCTION_THRESHOLDS),
            }
        else:
            failures.extend(
                _validate_file_hashes(run_dir, source_manifest)
            )
            failures.extend(
                _validate_manifest(
                    run_dir,
                    manifest,
                    environment,
                    source_manifest,
                    request_rows,
                    diagnostic_binding,
                )
            )
            row_sets = (
                ("dispatch_events", dispatch_rows),
                ("capture_events", capture_rows),
                ("request_metrics", request_rows),
                ("model_step_metrics", model_step_rows),
                ("memory_trace", memory_rows),
                ("correctness_rows", correctness_rows),
                ("budget_fallback_rows", budget_fallback_rows),
            )
            for evidence_name, rows in row_sets:
                _, row_failures = _index_unique(
                    rows,
                    evidence_name=evidence_name,
                )
                failures.extend(row_failures)
            failures.extend(
                _validate_source_rows(
                    row_sets,
                    manifest.get("source_tree_sha256"),
                )
            )
            failures.extend(
                _validate_identity_lifecycle(
                    dispatch_rows,
                    capture_rows,
                )
            )
            mode = manifest.get("mode")
            budget_failures, budget_summary = (
                _validate_budget_fallback_rows(
                    mode=mode,
                    manifest=manifest,
                    rows=budget_fallback_rows,
                    dispatch_rows=dispatch_rows,
                    capture_rows=capture_rows,
                    correctness_rows=correctness_rows,
                )
            )
            failures.extend(budget_failures)
            matrix = (
                contract.build_production_smoke_matrix()
                if mode in {"correctness-smoke", "arrival-smoke"}
                else contract.build_production_matrix()
            )
            reconstructed, reconstruction_failures = (
                _reconstruct_case_rows(
                    matrix,
                    manifest,
                    dispatch_rows,
                    request_rows,
                    model_step_rows,
                    memory_rows,
                    correctness_rows,
                    producer_case_rows,
                )
            )
            failures.extend(reconstruction_failures)
            if mode in {"correctness-smoke", "arrival-smoke"}:
                failures.extend(
                    _validate_smoke_requirements(reconstructed)
                )
            if failures:
                result = {
                    "classification": "NO_GO",
                    "failures": failures,
                    "metrics": {},
                    "thresholds": dict(
                        contract.PRODUCTION_THRESHOLDS
                    ),
                    **budget_summary,
                }
            elif mode in {"correctness-smoke", "arrival-smoke"}:
                result = {
                    "classification": "NON_AUTHORITATIVE_SMOKE",
                    "failures": [],
                    "metrics": {},
                    "thresholds": dict(
                        contract.PRODUCTION_THRESHOLDS
                    ),
                    **budget_summary,
                }
            else:
                independent = contract.classify_production_gate(
                    reconstructed,
                    producer_summary=producer_summary,
                    independent_summary={"classification": "GO"},
                )
                result = {
                    "classification": independent["classification"],
                    "failures": independent["failures"],
                    "metrics": independent["metrics"],
                    "thresholds": independent["thresholds"],
                    **budget_summary,
                }

    if write_report:
        _atomic_write_json(
            run_dir / "independent_verification.json",
            result,
        )
        _atomic_write_bytes(
            run_dir / "report.md",
            _render_report(result).encode("utf-8"),
        )
    return result


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = verify_run(
        args.run_dir,
        write_report=args.write_report,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["classification"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
