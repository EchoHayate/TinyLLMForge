from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.assemble_qwen38_topology_local_tp2_whole_model import (
    assemble_attempt,
)
from tools.test_assemble_qwen38_topology_local_tp2_whole_model import (
    passing_attempt,
)
from tools.verify_qwen38_topology_local_tp2_whole_model import (
    verify_bundle,
)
import tools.verify_qwen38_topology_local_tp2_whole_model as verifier


def _read_json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _write_jsonl(path, rows):
    Path(path).write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ))


def _rewrite_manifest(root):
    manifest_path = root / "manifest.json"
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
        and path.name not in {"manifest.json", "manifest.sha256"}
    }
    manifest = _read_json(manifest_path)
    manifest["files"] = files
    _write_json(manifest_path, manifest)
    (root / "manifest.sha256").write_text(
        hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n"
    )


def _bundle(tmp_path):
    attempt = passing_attempt(tmp_path / "attempt")
    output = tmp_path / "bundle"
    assemble_attempt(attempt, output)
    return output


def test_verifier_is_standard_library_only_and_reconstructs_go(tmp_path):
    root = _bundle(tmp_path)
    source = Path(verifier.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    result = verify_bundle(root)

    assert result["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"
    )
    assert not any(
        name == "torch"
        or name == "tinyvllm"
        or (name or "").startswith("tinyvllm.")
        or (name or "").startswith(
            "tools.assemble_qwen38_topology_local_tp2_whole_model"
        )
        or (name or "").startswith(
            "tools.qwen38_topology_local_tp2_whole_model_worker"
        )
        for name in imported
    )


def test_verifier_preserves_non_authoritative_service_parity_result(
    tmp_path,
):
    attempt = passing_attempt(tmp_path / "attempt")
    path = attempt / "raw" / "service_control_rows.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["replicas"][0]["requests"][0]["output_token_ids"][0] = 999
    rows[0]["requests"][0]["output_token_ids"][0] = 999
    rows[0]["global_tp4_baseline_output_parity"] = False
    _write_jsonl(path, rows)
    bundle = tmp_path / "bundle"
    assemble_attempt(attempt, bundle)

    receipt = verify_bundle(bundle)

    assert receipt["classification"] == (
        "GO_TOPOLOGY_LOCAL_TP2_WHOLE_MODEL_GATE"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("source_sha", "source"),
        ("model_revision", "model"),
        ("static_manifest_identity", "identity"),
        ("workload_inventory", "workload"),
        ("epoch_arm", "epoch"),
        ("epoch_identity", "epoch"),
        ("epoch_workload_order", "epoch"),
        ("environment_incomplete", "environment"),
        ("feature_contract", "feature"),
        ("topology_matching", "topology"),
        ("rank_mapping_extra", "rank mapping"),
        ("rank_mapping_order", "rank mapping"),
        ("source_revision_charset", "source"),
        ("source_tree_charset", "source"),
        ("request_token", "token"),
        ("correctness_proof", "correctness"),
        ("correctness_top_logit", "correctness"),
        ("correctness_checkpoint", "correctness"),
        ("correctness_boolean_commit", "correctness"),
        ("correctness_source", "correctness"),
        ("correctness_layer_inventory", "correctness"),
        ("correctness_baseline_state", "correctness"),
        ("correctness_cohort", "correctness"),
        ("timing_correctness_proof", "timing correctness"),
        ("decoded_text", "decoded text"),
        ("request_concurrency", "concurrency"),
        ("request_timing_derivation", "timing"),
        ("request_queueing", "queueing"),
        ("scheduler_step_duration", "scheduler timing"),
        ("scheduler_host_submission", "scheduler timing"),
        ("scheduler_startup_load", "scheduler timing"),
        ("raw_gap", "classification"),
        ("candidate_hit", "candidate"),
        ("candidate_source", "candidate"),
        ("short_chunk", "short"),
        ("collective_bytes", "collective"),
        ("publication", "publication"),
        ("migration_break_even", "migration"),
        ("migration_identity", "migration"),
        ("memory_peak", "classification"),
        ("memory_identity", "memory"),
        ("cleanup", "cleanup"),
        ("cleanup_raw_receipt", "cleanup"),
        ("cleanup_raw_inventory", "cleanup"),
        ("service_authority", "service"),
        ("service_identity", "service"),
        ("service_metric", "service"),
        ("service_output_parity", "service"),
        ("service_memory", "service"),
        ("resource_scope", "resource"),
        ("resource_runtime_telemetry", "resource"),
        ("resource_source", "resource"),
        ("resource_raw_identity", "resource"),
        ("resource_power", "resource"),
        ("cleanup_duration", "cleanup"),
        ("report_claim_boundary", "report"),
        ("report_telemetry", "report"),
        ("report_primary_metric", "report"),
        ("report_service_metric", "report"),
        ("classification", "classification"),
    ),
)
def test_verifier_rejects_semantic_mutations(
    tmp_path,
    mutation,
    message,
):
    root = _bundle(tmp_path)
    if mutation == "source_sha":
        payload = _read_json(root / "source_manifest.json")
        payload["source_tree_sha256"] = "c" * 64
        _write_json(root / "source_manifest.json", payload)
    elif mutation == "model_revision":
        payload = _read_json(root / "model_manifest.json")
        payload["model_revision"] = "f" * 40
        _write_json(root / "model_manifest.json", payload)
    elif mutation == "static_manifest_identity":
        payload = _read_json(root / "environment_manifest.json")
        payload["source_revision"] = "f" * 40
        _write_json(root / "environment_manifest.json", payload)
    elif mutation == "workload_inventory":
        payload = _read_json(root / "workload_manifest.json")
        payload["workloads"] = list(payload["workloads"][:-1])
        _write_json(root / "workload_manifest.json", payload)
    elif mutation == "epoch_arm":
        payload = _read_json(root / "campaign_epoch_manifest.json")
        payload["epochs"][1]["arm"] = "baseline"
        _write_json(root / "campaign_epoch_manifest.json", payload)
    elif mutation == "epoch_identity":
        payload = _read_json(root / "campaign_epoch_manifest.json")
        payload["epochs"][1]["epoch"] = 0
        _write_json(root / "campaign_epoch_manifest.json", payload)
    elif mutation == "epoch_workload_order":
        payload = _read_json(root / "campaign_epoch_manifest.json")
        payload["epochs"][1]["workload_order"] = list(
            verifier.WORKLOADS
        )
        _write_json(root / "campaign_epoch_manifest.json", payload)
    elif mutation == "environment_incomplete":
        payload = _read_json(root / "environment_manifest.json")
        payload["environment_complete"] = False
        _write_json(root / "environment_manifest.json", payload)
    elif mutation == "feature_contract":
        payload = _read_json(root / "feature_contract.json")
        payload["default_off"] = False
        _write_json(root / "feature_contract.json", payload)
    elif mutation == "topology_matching":
        payload = _read_json(root / "gpu_topology.json")
        for row in payload["rows"]:
            pair = tuple(sorted((row["left_rank"], row["right_rank"])))
            row["link"] = (
                "PIX" if pair in {(0, 2), (1, 3)} else "SYS"
            )
        _write_json(root / "gpu_topology.json", payload)
    elif mutation == "rank_mapping_extra":
        payload = _read_json(root / "gpu_rank_manifest.json")
        payload["mapping"].append({})
        _write_json(root / "gpu_rank_manifest.json", payload)
    elif mutation == "rank_mapping_order":
        payload = _read_json(root / "gpu_rank_manifest.json")
        payload["mapping"].reverse()
        _write_json(root / "gpu_rank_manifest.json", payload)
    elif mutation == "source_revision_charset":
        payload = _read_json(root / "source_manifest.json")
        payload["source_revision"] = "z" * 40
        _write_json(root / "source_manifest.json", payload)
    elif mutation == "source_tree_charset":
        payload = _read_json(root / "source_manifest.json")
        payload["source_tree_sha256"] = "z" * 64
        _write_json(root / "source_manifest.json", payload)
    elif mutation == "request_token":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["requests"][0]["output_token_ids"][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "correctness_proof":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_step_proofs"][10][3]["token_ids"][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "correctness_top_logit":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_step_proofs"][10][3][
            "top_logit_values"
        ][0] += 1.0
        _write_jsonl(path, rows)
    elif mutation == "correctness_checkpoint":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_state_checkpoints"]["token_32"][2][
            "runtime_snapshot"
        ]["state"]["commit_count"] += 1
        _write_jsonl(path, rows)
    elif mutation == "correctness_boolean_commit":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_state_checkpoints"]["post_migration"][0][
            "runtime_snapshot"
        ]["state"]["commit_count"] = True
        _write_jsonl(path, rows)
    elif mutation == "correctness_source":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_revision"] = "f" * 40
        _write_jsonl(path, rows)
    elif mutation == "correctness_layer_inventory":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for rank_row in rows[0]["candidate_state_checkpoints"]["token_32"]:
            rank_row["output_digests"][0]["layer_index"] = 999
            rank_row["state_digests"][0]["layer_index"] = 999
        _write_jsonl(path, rows)
    elif mutation == "correctness_baseline_state":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["baseline_state_checkpoints"]["token_32"][2][
            "canonical_state_components"
        ][0]["recurrent_sha256"] = "f" * 64
        _write_jsonl(path, rows)
    elif mutation == "correctness_cohort":
        path = root / "correctness_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["candidate_state_checkpoints"]["token_32"][0][
            "cohort"
        ][0]["generation"] += 1
        _write_jsonl(path, rows)
    elif mutation == "timing_correctness_proof":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["timing_correctness_replay"]["step_proofs"][10][3][
            "token_ids"
        ][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "decoded_text":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["requests"][0]["decoded_text"] += "tampered"
        _write_jsonl(path, rows)
    elif mutation == "request_concurrency":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["workload_id"] == "Q1" and row["repetition"] == 0:
                row["requests"].pop()
                replay = row["timing_correctness_replay"]
                replay["requests"].pop()
                for step_rows in replay["step_proofs"]:
                    for rank_row in step_rows:
                        rank_row["sequence_ids"].pop()
                        rank_row["token_ids"].pop()
                        rank_row["top_logit_values"].pop()
        _write_jsonl(path, rows)
    elif mutation == "request_timing_derivation":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["requests"][0]["tpot_ns"] += 1
        _write_jsonl(path, rows)
    elif mutation == "request_queueing":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["requests"][0]["queueing_ns"] += 1
        _write_jsonl(path, rows)
    elif mutation == "scheduler_step_duration":
        path = root / "scheduler_step_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["steps"][0]["step_duration_ns"] += 1
        _write_jsonl(path, rows)
    elif mutation == "scheduler_host_submission":
        path = root / "scheduler_step_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["steps"][0]["host_submission_ns"] = 81
        _write_jsonl(path, rows)
    elif mutation == "scheduler_startup_load":
        path = root / "scheduler_step_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["startup_model_load_duration_ns"] += 1
        _write_jsonl(path, rows)
    elif mutation == "raw_gap":
        path = root / "request_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["arm"] == "candidate":
                request = row["requests"][0]
                request["token_timestamps_ns"] = [
                    request["admitted_ns"] + 1_000 + step * 20_000
                    for step in range(128)
                ]
                request["token_gaps_ns"] = [20_000] * 127
                request["tpot_ns"] = 20_000.0
                request["completion_ns"] = (
                    request["token_timestamps_ns"][-1]
                )
                request["e2e_ns"] = (
                    request["completion_ns"] - request["admitted_ns"]
                )
                row["cohort_makespan_ns"] = max(
                    item["completion_ns"] for item in row["requests"]
                ) - min(
                    item["admitted_ns"] for item in row["requests"]
                )
        _write_jsonl(path, rows)
        request_rows = {
            (
                row["epoch"],
                row["arm"],
                row["workload_id"],
                row["repetition"],
            ): row
            for row in rows
        }
        migration_path = root / "migration_rows.jsonl"
        migration_rows = [
            json.loads(line)
            for line in migration_path.read_text().splitlines()
        ]
        baseline_for_candidate = {1: 3, 2: 0}
        for row in migration_rows:
            identity = (row["workload_id"], row["repetition"])
            candidate = request_rows[
                (row["epoch"], "candidate", *identity)
            ]
            baseline = request_rows[
                (
                    baseline_for_candidate[row["epoch"]],
                    "baseline",
                    *identity,
                )
            ]
            candidate_tpot = sum(
                request["tpot_ns"] for request in candidate["requests"]
            ) / len(candidate["requests"])
            baseline_tpot = sum(
                request["tpot_ns"] for request in baseline["requests"]
            ) / len(baseline["requests"])
            savings = baseline_tpot - candidate_tpot
            row["break_even_output_tokens"] = (
                row["latency_ns"] / savings
                if savings > 0
                else 1e30
            )
        _write_jsonl(migration_path, migration_rows)
    elif mutation == "candidate_hit":
        path = root / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["tp2_decode_calls"] -= 1
        _write_jsonl(path, rows)
    elif mutation == "candidate_source":
        path = root / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_revision"] = "f" * 40
        _write_jsonl(path, rows)
    elif mutation == "short_chunk":
        path = root / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["short_chunk_calls"] = 1
        _write_jsonl(path, rows)
    elif mutation == "collective_bytes":
        path = root / "collective_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["pair_local_bytes"] -= 1
        _write_jsonl(path, rows)
    elif mutation == "publication":
        path = root / "candidate_hit_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["migration_publications"] -= 1
        _write_jsonl(path, rows)
    elif mutation == "migration_break_even":
        path = root / "migration_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["break_even_output_tokens"] = 0.0
        _write_jsonl(path, rows)
    elif mutation == "migration_identity":
        path = root / "migration_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["epoch"] = rows[0]["epoch"]
        rows[1]["workload_id"] = rows[0]["workload_id"]
        rows[1]["repetition"] = rows[0]["repetition"]
        _write_jsonl(path, rows)
    elif mutation == "memory_peak":
        path = root / "memory_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["peak_allocated_bytes"] = 79 * 1024**3
        rows[0]["peak_reserved_bytes"] = 79 * 1024**3
        _write_jsonl(path, rows)
    elif mutation == "memory_identity":
        path = root / "memory_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["epoch"] = rows[0]["epoch"]
        rows[1]["arm"] = rows[0]["arm"]
        rows[1]["rank"] = rows[0]["rank"]
        _write_jsonl(path, rows)
    elif mutation == "cleanup":
        payload = _read_json(root / "cleanup.json")
        payload["retained_process_groups"] = 1
        _write_json(root / "cleanup.json", payload)
    elif mutation == "cleanup_raw_receipt":
        payload = _read_json(root / "cleanup.json")
        payload["worker_cleanup_receipts"][1]["receipt"][
            "rank_cleanup_receipts"
        ][0]["qwen38_topology_local_tp2_cleanup"][
            "candidate_state_released"
        ] = False
        _write_json(root / "cleanup.json", payload)
    elif mutation == "cleanup_raw_inventory":
        payload = _read_json(root / "cleanup.json")
        payload["worker_cleanup_receipts"].pop()
        _write_json(root / "cleanup.json", payload)
    elif mutation == "service_authority":
        path = root / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["classification_authority"] = True
        _write_jsonl(path, rows)
    elif mutation == "service_identity":
        path = root / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows.append(json.loads(json.dumps(rows[0])))
        _write_jsonl(path, rows)
    elif mutation == "service_metric":
        path = root / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["request_qps"] += 1
        _write_jsonl(path, rows)
    elif mutation == "service_output_parity":
        path = root / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["replicas"][0]["requests"][0]["output_token_ids"][0] = 999
        _write_jsonl(path, rows)
    elif mutation == "service_memory":
        path = root / "service_control_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        memory = rows[0]["replicas"][0]["peak_memory_by_gpu"][0]
        memory["peak_reserved_bytes"] = (
            memory["physical_memory_bytes"] + 1
        )
        _write_jsonl(path, rows)
    elif mutation == "resource_scope":
        path = root / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        runtime = next(
            row for row in rows
            if row["measurement_scope"] == "runtime"
        )
        runtime["measurement_scope"] = "boundary"
        _write_jsonl(path, rows)
    elif mutation == "resource_runtime_telemetry":
        path = root / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        runtime = next(
            row for row in rows
            if row["measurement_scope"] == "runtime"
        )
        runtime["gpu_inventory"][0]["power_watts"] = -1
        runtime["strict_clean"] = True
        _write_jsonl(path, rows)
    elif mutation == "resource_source":
        path = root / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["source_revision"] = "f" * 40
        _write_jsonl(path, rows)
    elif mutation == "resource_raw_identity":
        path = root / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["gpu_inventory"][0]["gpu_uuid"] = "GPU-drift"
        _write_jsonl(path, rows)
    elif mutation == "resource_power":
        path = root / "resource_rows.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["gpu_inventory"][0]["power_watts"] = -1
        _write_jsonl(path, rows)
    elif mutation == "cleanup_duration":
        payload = _read_json(root / "cleanup.json")
        payload["worker_cleanup_receipts"][0]["receipt"][
            "cleanup_duration_ns"
        ] += 1
        _write_json(root / "cleanup.json", payload)
    elif mutation == "report_claim_boundary":
        path = root / "report.md"
        path.write_text(
            path.read_text().replace(
                "Production-default enablement: prohibited.",
                "Production-default enablement: allowed.",
            )
        )
    elif mutation == "report_telemetry":
        path = root / "report.md"
        path.write_text(
            path.read_text().replace(
                "Runtime GPU power P50/P95/P99 (W):",
                "GPU power omitted:",
            )
        )
    elif mutation == "report_primary_metric":
        path = root / "report.md"
        lines = path.read_text().splitlines()
        index = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("| P0 |")
        )
        lines[index] = "| P0 | fabricated primary metrics |"
        path.write_text("\n".join(lines) + "\n")
    elif mutation == "report_service_metric":
        path = root / "report.md"
        lines = path.read_text().splitlines()
        section = lines.index(
            "## TP2 x2 service control (non-authoritative)"
        )
        index = next(
            index
            for index in range(section + 1, len(lines))
            if lines[index].startswith("| Q0 |")
        )
        lines[index] = "| Q0 | fabricated service metrics |"
        path.write_text("\n".join(lines) + "\n")
    elif mutation == "classification":
        payload = _read_json(root / "classification.json")
        payload["classification"] = "NO_GO_PERFORMANCE"
        _write_json(root / "classification.json", payload)
    _rewrite_manifest(root)

    with pytest.raises((ValueError, RuntimeError), match=message):
        verify_bundle(root)


def test_verifier_rejects_manifest_digest_mutation(tmp_path):
    root = _bundle(tmp_path)
    (root / "manifest.sha256").write_text("0" * 64 + "\n")

    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(root)


def test_remote_and_local_receipts_are_byte_identical(tmp_path):
    root = _bundle(tmp_path)
    remote_path = tmp_path / "remote.json"
    local_path = tmp_path / "local.json"

    remote = verify_bundle(root, output_path=remote_path)
    local = verify_bundle(root, output_path=local_path)

    assert remote == local
    assert remote_path.read_bytes() == local_path.read_bytes()
