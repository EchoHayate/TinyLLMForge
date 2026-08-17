from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


RESULT_NAME = "tp4_live_concurrent_candidate_ownership.json"
MANIFEST_NAME = "source_manifest.json"
RESULT_SCHEMA = (
    "qwen35.tp4-live-concurrent-candidate-ownership.v1"
)
READY_SCHEMA = (
    "qwen35.tp4-live-concurrent-candidate-ready-rank.v1"
)
RELEASED_SCHEMA = (
    "qwen35.tp4-live-concurrent-candidate-released-rank.v1"
)
SNAPSHOT_SCHEMA = (
    "qwen35.tp4-live-concurrent-candidate-snapshot.v1"
)
PROVENANCE = (
    "real-checkpoint-derived-live-concurrent-tp4-ownership"
)
CLAIM_BOUNDARY = "not-constructed-engine-runtime-binding"
PREREQUISITE_ORACLE_SHA256 = (
    "d750d664219378c234a2127b708ec191feb9b2c9f1f2902c47d0ad5dc152d3ef"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "42dddc0eac0a6db6041d5abb71df34db4d5e7c99d3b74d69f94598a2f24eb137"
)
APPROVED_MODEL_MANIFEST_SHA256 = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
PREFLIGHT_SOURCE = (
    "tools/qwen35_tp4_live_concurrent_candidate_ownership_preflight.py"
)
PRIVATE_OBJECT_NAMES = (
    "runner",
    "production_slot",
    "request",
    "candidate",
    "owner",
    "runtime_bridge",
    "runtime_identity",
    "model",
    "pool",
    "target",
)
MEMORY_CEILINGS_KIB = {
    "per_worker_total_vmhwm_increment": 3145728,
    "aggregate_worker_vmhwm_increment": 12582912,
    "aggregate_ready_vmrss": 8388608,
    "host_mem_available_decrease": 12582912,
    "minimum_host_mem_available": 16777216,
}


class VerificationError(ValueError):
    pass


class Checker:
    def __init__(self):
        self.count = 0

    def require(self, condition, detail):
        self.count += 1
        if not condition:
            raise VerificationError(detail)


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(
            f"invalid JSON: {Path(path).name}"
        ) from error
    if not isinstance(value, dict):
        raise VerificationError(
            f"{Path(path).name} must be an object"
        )
    return value


def _is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _static_safety(source_root, checker):
    path = Path(source_root) / PREFLIGHT_SOURCE
    checker.require(path.is_file(), "missing preflight source")
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=PREFLIGHT_SOURCE)
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    checker.require(
        "tinyvllm.engine.llm_engine" not in modules,
        "production Engine import detected",
    )
    checker.require(
        "tinyvllm.engine.model_runner" not in modules,
        "production ModelRunner import detected",
    )
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]
    for name in (
        "LLMEngine",
        "ModelRunner",
        "Scheduler",
        "schedule",
        "step",
        "forward",
        "inference",
    ):
        count = sum(
            (
                isinstance(node.func, ast.Name)
                and node.func.id == name
            )
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == name
            )
            for node in calls
        )
        checker.require(count == 0, f"forbidden call detected: {name}")


def _verify_sources(record, manifest, source_root, checker):
    hashes = record.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and hashes,
        "source file hashes are invalid",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "source file ordering is invalid",
    )
    checker.require(
        manifest.get("source_file_sha256") == hashes,
        "manifest source hashes mismatch",
    )
    for name, expected in hashes.items():
        checker.require(
            isinstance(name, str) and name,
            "source filename is invalid",
        )
        checker.require(_is_sha(expected), f"source SHA invalid: {name}")
        path = Path(source_root) / name
        checker.require(path.is_file(), f"missing source: {name}")
        checker.require(
            _sha256(path.read_bytes()) == expected,
            f"source hash mismatch: {name}",
        )
    tree = _sha256(_canonical(hashes))
    checker.require(
        record.get("source_tree_sha256") == tree,
        "result source tree mismatch",
    )
    checker.require(
        manifest.get("source_tree_sha256") == tree,
        "manifest source tree mismatch",
    )
    return hashes


def _read_pristine_oracle(path, checker):
    payload = Path(path).read_bytes()
    checker.require(
        _sha256(payload) == PREREQUISITE_ORACLE_SHA256,
        "pristine oracle SHA mismatch",
    )
    oracle = json.loads(payload)
    checker.require(
        isinstance(oracle, dict)
        and oracle.get("schema_version")
        == "qwen35.tp4-real-candidate-provenance-oracle.v1"
        and oracle.get("status") == "PASS",
        "pristine oracle schema mismatch",
    )
    checker.require(
        oracle.get("source_tree_sha256")
        == PREREQUISITE_SOURCE_TREE_SHA256,
        "pristine oracle source tree mismatch",
    )
    checker.require(
        oracle.get("model_manifest_sha256")
        == APPROVED_MODEL_MANIFEST_SHA256,
        "pristine oracle model mismatch",
    )
    rows = oracle.get("producer_rows")
    checker.require(
        isinstance(rows, list)
        and [row.get("tp_rank") for row in rows] == [0, 1, 2, 3],
        "pristine oracle rows are incomplete",
    )
    for rank, row in enumerate(rows):
        checker.require(
            row.get("binding_hash_count") == 320
            and isinstance(row.get("binding_destination_sha256"), list)
            and len(row["binding_destination_sha256"]) == 320,
            f"pristine binding payload invalid: rank={rank}",
        )
        checker.require(
            row.get("phase_hash_count") == 26
            and isinstance(row.get("phase_destination_sha256"), dict)
            and len(row["phase_destination_sha256"]) == 26,
            f"pristine phase payload invalid: rank={rank}",
        )
        checker.require(
            _is_sha(row.get("aggregate_destination_sha256")),
            f"pristine aggregate payload invalid: rank={rank}",
        )
    return oracle


def _verify_ready(row, rank, pristine_row, checker):
    exact = {
        "schema_version": READY_SCHEMA,
        "status": "READY",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
        "all_private_objects_retained": True,
        "retained_private_objects": {
            name: True for name in PRIVATE_OBJECT_NAMES
        },
    }
    for name, expected in exact.items():
        checker.require(
            row.get(name) == expected,
            f"ready field mismatch: rank={rank}, field={name}",
        )
    checker.require(
        isinstance(row.get("process_id"), int)
        and not isinstance(row.get("process_id"), bool)
        and row["process_id"] > 0,
        f"ready process invalid: rank={rank}",
    )
    method = row.get("method_row")
    checker.require(
        isinstance(method, dict),
        f"ready method row invalid: rank={rank}",
    )
    for name, expected in {
        "participant_id": rank,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "bound",
        "dtype": "bfloat16",
        "detail": "",
    }.items():
        checker.require(
            method.get(name) == expected,
            f"ready method mismatch: rank={rank}, field={name}",
        )
    for name in ("model_fingerprint", "layout_fingerprint"):
        checker.require(
            _is_sha(method.get(name)),
            f"ready method SHA invalid: rank={rank}, field={name}",
        )
    checker.require(
        _is_sha(row.get("aggregate_destination_sha256")),
        f"ready aggregate invalid: rank={rank}",
    )
    binding_hashes = row.get("binding_destination_sha256")
    checker.require(
        isinstance(binding_hashes, list)
        and len(binding_hashes) == 320
        and all(_is_sha(value) for value in binding_hashes),
        f"ready binding payload invalid: rank={rank}",
    )
    checker.require(
        binding_hashes
        == pristine_row.get("binding_destination_sha256"),
        f"ready binding payload mismatch: rank={rank}",
    )
    phase_hashes = row.get("phase_destination_sha256")
    checker.require(
        isinstance(phase_hashes, dict)
        and len(phase_hashes) == 26
        and all(
            isinstance(name, str)
            and name
            and _is_sha(value)
            for name, value in phase_hashes.items()
        ),
        f"ready phase payload invalid: rank={rank}",
    )
    checker.require(
        phase_hashes == pristine_row.get("phase_destination_sha256"),
        f"ready phase payload mismatch: rank={rank}",
    )
    checker.require(
        row.get("aggregate_destination_sha256")
        == pristine_row.get("aggregate_destination_sha256"),
        f"ready aggregate payload mismatch: rank={rank}",
    )
    checker.require(
        row.get("alias_groups") == pristine_row.get("alias_groups"),
        f"ready alias payload mismatch: rank={rank}",
    )
    checker.require(
        row.get("loader_stats")
        == {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
        f"ready loader stats invalid: rank={rank}",
    )
    checker.require(
        row.get("loader_stats") == pristine_row.get("loader_stats"),
        f"ready loader stats mismatch: rank={rank}",
    )
    checker.require(
        method.get("model_fingerprint")
        == pristine_row.get("model_manifest_sha256")
        and method.get("layout_fingerprint")
        == pristine_row.get("layout_fingerprint")
        and method.get("dtype") == pristine_row.get("dtype"),
        f"ready runtime identity mismatch: rank={rank}",
    )
    memory = row.get("memory")
    checker.require(
        isinstance(memory, dict)
        and set(memory) == {"before", "ready"},
        f"ready memory invalid: rank={rank}",
    )
    for point in ("before", "ready"):
        checker.require(
            isinstance(memory[point], dict)
            and set(memory[point]) == {"vmrss_kib", "vmhwm_kib"},
            f"ready memory point invalid: rank={rank}, point={point}",
        )
        for name in ("vmrss_kib", "vmhwm_kib"):
            checker.require(
                isinstance(memory[point][name], int)
                and not isinstance(memory[point][name], bool)
                and memory[point][name] >= 0,
                f"ready memory value invalid: rank={rank}, point={point}",
            )
    checker.require(
        row.get("ready_memory") == memory["ready"],
        f"ready memory duplicate mismatch: rank={rank}",
    )


def _verify_released(row, rank, checker):
    exact = {
        "schema_version": RELEASED_SCHEMA,
        "status": "RELEASED",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "collected_private_objects": {
            name: True for name in PRIVATE_OBJECT_NAMES
        },
        "all_private_objects_collected": True,
    }
    checker.require(
        row == exact,
        f"released row mismatch: rank={rank}",
    )


def _verify_memory(record, ready_rows, snapshot, checker):
    before_host = record.get("host_memory_before")
    ready_host = record.get("host_memory_ready")
    for label, value in (
        ("before", before_host),
        ("ready", ready_host),
    ):
        checker.require(
            isinstance(value, dict)
            and set(value)
            == {"mem_available_kib", "swap_total_kib", "swap_free_kib"},
            f"host memory invalid: {label}",
        )
        checker.require(
            all(
                isinstance(item, int)
                and not isinstance(item, bool)
                and item >= 0
                for item in value.values()
            ),
            f"host memory values invalid: {label}",
        )
    checker.require(
        before_host["mem_available_kib"]
        >= MEMORY_CEILINGS_KIB["minimum_host_mem_available"],
        "host memory below 16777216 KiB",
    )
    checker.require(
        (
            before_host["swap_total_kib"],
            before_host["swap_free_kib"],
        )
        == (
            ready_host["swap_total_kib"],
            ready_host["swap_free_kib"],
        ),
        "swap changed during residency",
    )
    status_by_rank = {
        row.get("rank"): row
        for row in snapshot.get("process_status", [])
        if isinstance(row, dict)
    }
    checker.require(
        set(status_by_rank) == set(range(4)),
        "snapshot process ranks invalid",
    )
    increments = []
    ready_rss = []
    for rank, row in enumerate(ready_rows):
        memory = row["memory"]
        increment = (
            memory["ready"]["vmhwm_kib"]
            - memory["before"]["vmhwm_kib"]
        )
        checker.require(
            increment >= 0,
            f"negative worker memory delta: rank={rank}",
        )
        checker.require(
            increment
            <= MEMORY_CEILINGS_KIB[
                "per_worker_total_vmhwm_increment"
            ],
            (
                f"worker memory ceiling exceeded: rank={rank}, "
                f"allowed={MEMORY_CEILINGS_KIB['per_worker_total_vmhwm_increment']}"
            ),
        )
        status = status_by_rank[rank]
        checker.require(
            status.get("process_id") == row["process_id"],
            f"snapshot PID mismatch: rank={rank}",
        )
        checker.require(
            status.get("vmrss_kib")
            == memory["ready"]["vmrss_kib"],
            f"snapshot VmRSS mismatch: rank={rank}",
        )
        checker.require(
            status.get("vmhwm_kib")
            == memory["ready"]["vmhwm_kib"],
            f"snapshot VmHWM mismatch: rank={rank}",
        )
        increments.append(increment)
        ready_rss.append(memory["ready"]["vmrss_kib"])
    aggregate_increment = sum(increments)
    aggregate_rss = sum(ready_rss)
    available_decrease = (
        before_host["mem_available_kib"]
        - ready_host["mem_available_kib"]
    )
    checker.require(
        aggregate_increment
        <= MEMORY_CEILINGS_KIB["aggregate_worker_vmhwm_increment"],
        "aggregate worker memory ceiling exceeded",
    )
    checker.require(
        aggregate_rss
        <= MEMORY_CEILINGS_KIB["aggregate_ready_vmrss"],
        "aggregate ready VmRSS ceiling exceeded",
    )
    checker.require(
        0 <= available_decrease
        <= MEMORY_CEILINGS_KIB["host_mem_available_decrease"],
        "host MemAvailable decrease ceiling exceeded",
    )
    expected = {
        "per_worker_total_vmhwm_increment_kib": increments,
        "aggregate_worker_vmhwm_increment_kib": aggregate_increment,
        "aggregate_ready_vmrss_kib": aggregate_rss,
        "host_mem_available_decrease_kib": available_decrease,
        "memory_contract_passed": True,
    }
    checker.require(
        record.get("memory_summary") == expected,
        "memory summary mismatch",
    )


def _verify(run_dir, source_root, prerequisite_oracle, checker):
    run_dir = Path(run_dir)
    checker.require(
        sorted(path.name for path in run_dir.iterdir())
        == [MANIFEST_NAME, RESULT_NAME],
        "run inventory mismatch",
    )
    result_path = run_dir / RESULT_NAME
    manifest_path = run_dir / MANIFEST_NAME
    record = _read(result_path)
    manifest = _read(manifest_path)
    oracle = _read_pristine_oracle(prerequisite_oracle, checker)
    _static_safety(source_root, checker)
    checker.require(
        record.get("schema_version") == RESULT_SCHEMA,
        "result schema mismatch",
    )
    checker.require(record.get("status") == "PASS", "result status")
    checker.require(
        record.get("provenance") == PROVENANCE,
        "result provenance mismatch",
    )
    checker.require(
        record.get("claim_boundary") == CLAIM_BOUNDARY,
        "result claim boundary mismatch",
    )
    checker.require(
        record.get("start_order") == [0, 1, 2, 3],
        "start order mismatch",
    )
    checker.require(
        record.get("ready_order") == [0, 1, 2, 3],
        "ready order mismatch",
    )
    checker.require(
        record.get("release_order") == [3, 2, 1, 0],
        "release order mismatch",
    )
    checker.require(
        record.get("released_order") == [3, 2, 1, 0],
        "released order mismatch",
    )
    checker.require(
        record.get("all_workers_exited") is True,
        "worker exit proof missing",
    )
    checker.require(
        record.get("prerequisite_oracle_sha256")
        == PREREQUISITE_ORACLE_SHA256,
        "prerequisite oracle SHA mismatch",
    )
    checker.require(
        manifest.get("prerequisite_oracle_sha256")
        == record["prerequisite_oracle_sha256"],
        "manifest prerequisite mismatch",
    )
    hashes = _verify_sources(
        record,
        manifest,
        source_root,
        checker,
    )
    ready_rows = record.get("ready_rows")
    released_rows = record.get("released_rows")
    checker.require(
        isinstance(ready_rows, list) and len(ready_rows) == 4,
        "ready rows incomplete",
    )
    checker.require(
        isinstance(released_rows, list) and len(released_rows) == 4,
        "released rows incomplete",
    )
    for rank, row in enumerate(ready_rows):
        _verify_ready(row, rank, oracle["producer_rows"][rank], checker)
    for row, rank in zip(released_rows, (3, 2, 1, 0)):
        _verify_released(row, rank, checker)
    checker.require(
        record.get("ready_rows_sha256")
        == _sha256(_canonical(ready_rows)),
        "ready rows hash mismatch",
    )
    checker.require(
        record.get("released_rows_sha256")
        == _sha256(_canonical(released_rows)),
        "released rows hash mismatch",
    )
    snapshot = record.get("concurrent_snapshot")
    checker.require(
        isinstance(snapshot, dict)
        and snapshot.get("schema_version") == SNAPSHOT_SCHEMA,
        "snapshot schema mismatch",
    )
    checker.require(
        snapshot.get("start_order") == [0, 1, 2, 3],
        "snapshot start order mismatch",
    )
    checker.require(
        snapshot.get("ready_order") == [0, 1, 2, 3],
        "snapshot ready order mismatch",
    )
    checker.require(
        isinstance(snapshot.get("snapshot_unix_time_ns"), int)
        and not isinstance(snapshot.get("snapshot_unix_time_ns"), bool)
        and snapshot["snapshot_unix_time_ns"] > 0,
        "snapshot timestamp invalid",
    )
    process_ids = [row["process_id"] for row in ready_rows]
    checker.require(
        len(set(process_ids)) == 4,
        "ready process IDs overlap",
    )
    checker.require(
        snapshot.get("live_process_ids") == process_ids,
        "snapshot live process IDs mismatch",
    )
    checker.require(
        snapshot.get("ready_row_sha256")
        == [_sha256(_canonical(row)) for row in ready_rows],
        "snapshot ready row hashes mismatch",
    )
    checker.require(
        snapshot.get("release_acknowledgement_count") == 0,
        "premature release acknowledgement",
    )
    checker.require(
        snapshot.get("all_workers_live_concurrently") is True,
        "concurrent residency proof missing",
    )
    checker.require(
        record.get("worker_process_ids") == process_ids,
        "worker process inventory mismatch",
    )
    checker.require(
        record.get("residual_worker_process_ids") == [],
        "residual worker process detected",
    )
    checker.require(
        record.get("all_worker_process_ids_absent") is True,
        "worker process absence proof missing",
    )
    _verify_memory(record, ready_rows, snapshot, checker)
    checker.require(
        manifest.get("schema_version") == RESULT_SCHEMA,
        "manifest schema mismatch",
    )
    checker.require(
        manifest.get("remote_target") == REMOTE_TARGET,
        "manifest remote target mismatch",
    )
    checker.require(
        manifest.get("remote_python") == REMOTE_PYTHON,
        "manifest remote Python mismatch",
    )
    checker.require(
        manifest.get("source_tree_sha256")
        == record.get("source_tree_sha256"),
        "manifest source tree mismatch",
    )
    checker.require(
        manifest.get("result_sha256")
        == _sha256(result_path.read_bytes()),
        "manifest result SHA mismatch",
    )
    return {
        "ready_count": len(ready_rows),
        "released_count": len(released_rows),
        "unique_process_count": len(set(process_ids)),
        "source_file_count": len(hashes),
        "source_tree_sha256": record["source_tree_sha256"],
        "result_sha256": _sha256(result_path.read_bytes()),
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
    }


def verify_run(run_dir, *, source_root, prerequisite_oracle):
    checker = Checker()
    try:
        summary = _verify(
            run_dir,
            source_root,
            prerequisite_oracle,
            checker,
        )
    except (
        VerificationError,
        OSError,
        SyntaxError,
        KeyError,
        TypeError,
        IndexError,
    ) as error:
        return {
            "status": "FAIL",
            "checks": checker.count,
            "detail": str(error),
        }
    return {"status": "PASS", "checks": checker.count, **summary}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--prerequisite-oracle", required=True)
    arguments = parser.parse_args(argv)
    result = verify_run(
        arguments.run_dir,
        source_root=arguments.source_root,
        prerequisite_oracle=arguments.prerequisite_oracle,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
