from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ARTIFACT_NAME = "tp4_shared_memory_fanout_preflight.json"
MANIFEST_NAME = "source_manifest.json"
SCHEMA_VERSION = "qwen35.tp4-shared-memory-fanout.v1"
ROW_SCHEMA_VERSION = "qwen35.tp4-shared-memory-fanout-rank.v1"
PREREQUISITE_SHA256 = (
    "11f2decd379de668b575cb7f4a0c55874fbefb740d2b4841fb4db3b72ca39c57"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "6cc9672dbd80c211ccd64371573fd8de463b773fc5cc3ae7286ad21c9c664572"
)
LLM_ENGINE_SOURCE = "tinyvllm/engine/llm_engine.py"
LLM_ENGINE_FILE_SHA256 = (
    "6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae"
)
MODEL_RUNNER_SOURCE = "tinyvllm/engine/model_runner.py"
MODEL_RUNNER_FILE_SHA256 = (
    "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
)
ACK_SOURCE = "tinyvllm/engine/model_runner_command_ack.py"
ACK_FILE_SHA256 = (
    "ca28babca5cc725d8c9bf0e3e057fa4b0cabfd847bf0c052c40876fbc148c61b"
)
PREFLIGHT_SOURCE = "tools/qwen35_tp4_shared_memory_fanout_preflight.py"
METHOD_SPECS = {
    "write_shm": (
        MODEL_RUNNER_SOURCE,
        "ModelRunner",
        ("self", "envelope"),
        None,
        (),
        "f9a377bf748d5be91a3c3722850e5e486f8e7dd8157e87d3dc6d692a60be6d76",
    ),
    "read_shm": (
        MODEL_RUNNER_SOURCE,
        "ModelRunner",
        ("self",),
        None,
        (),
        "1266b5d20b2978b655716f9ec8b58ce0a5644b9709164a23c18b85346170054a",
    ),
    "loop": (
        MODEL_RUNNER_SOURCE,
        "ModelRunner",
        ("self",),
        None,
        (),
        "342bac6d01606e4834e7ed77ef3e76d59b2fc3ea617afebe2c195912159dd2bb",
    ),
    "dispatch_command": (
        MODEL_RUNNER_SOURCE,
        "ModelRunner",
        ("self", "method_name"),
        "args",
        ("requires_ack",),
        "9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342",
    ),
    "call_model_runner_acknowledged": (
        LLM_ENGINE_SOURCE,
        "LLMEngine",
        ("self", "method_name"),
        "args",
        ("timeout_s",),
        "6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d",
    ),
}
ATTEMPT_MODES = (
    "tp4_fanout_success_reverse_completion",
    "tp4_fanout_rank2_inner_error",
    "tp4_fanout_rank2_ack_exception",
    "tp4_fanout_rank2_exit_without_ack",
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
SHARED_MEMORY_CAPACITY = 2**20
WORKER_RANKS = (1, 2, 3)


class VerificationError(ValueError):
    pass


class Checker:
    def __init__(self):
        self.count = 0

    def require(self, condition, detail):
        self.count += 1
        if not condition:
            raise VerificationError(detail)


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(
            f"invalid JSON artifact {Path(path).name}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise VerificationError(
            f"JSON artifact {Path(path).name} must be an object"
        )
    return value


def _source_tree_sha256(hashes):
    return _sha256_bytes(json.dumps(
        dict(hashes),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _method_hashes(source_root, checker):
    root = Path(source_root)
    cache = {}
    observed = {}
    for name, (
        filename,
        class_name,
        positional,
        vararg,
        keyword_only,
        expected_hash,
    ) in METHOD_SPECS.items():
        source = cache.setdefault(
            filename,
            (root / filename).read_text(encoding="utf-8"),
        )
        tree = ast.parse(source, filename=filename)
        classes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == class_name
        ]
        checker.require(len(classes) == 1, f"class count mismatch: {name}")
        methods = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef)
            and node.name == name
        ]
        checker.require(len(methods) == 1, f"method count mismatch: {name}")
        node = methods[0]
        checker.require(
            tuple(
                argument.arg
                for argument in (*node.args.posonlyargs, *node.args.args)
            ) == positional,
            f"method positional arguments mismatch: {name}",
        )
        checker.require(
            (
                node.args.vararg.arg
                if node.args.vararg is not None
                else None
            ) == vararg,
            f"method vararg mismatch: {name}",
        )
        checker.require(
            tuple(argument.arg for argument in node.args.kwonlyargs)
            == keyword_only,
            f"method keyword-only arguments mismatch: {name}",
        )
        segment = ast.get_source_segment(source, node)
        checker.require(
            isinstance(segment, str),
            f"method source segment missing: {name}",
        )
        observed[name] = _sha256_bytes(segment.encode("utf-8"))
        checker.require(
            observed[name] == expected_hash,
            f"method SHA256 mismatch: {name}",
        )
    return observed


def _verify_prerequisite(record, checker):
    checker.require(
        record.get("schema_version")
        == "qwen35.live-shared-memory-engine-ack-dispatch.v1",
        "prerequisite schema mismatch",
    )
    checker.require(
        record.get("status") == "PASS",
        "prerequisite status mismatch",
    )
    checker.require(
        record.get("source_tree_sha256")
        == PREREQUISITE_SOURCE_TREE_SHA256,
        "prerequisite source tree mismatch",
    )
    hashes = record.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and len(hashes) == 55,
        "prerequisite source count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "prerequisite source ordering mismatch",
    )
    rows = record.get("rows")
    checker.require(
        isinstance(rows, list) and len(rows) == 4,
        "prerequisite row count mismatch",
    )
    checker.require(
        tuple(row.get("mode") for row in rows)
        == (
            "tp2_shm_success",
            "tp2_shm_worker_binding_error",
            "tp2_shm_worker_ack_exception",
            "tp2_shm_worker_exit_without_ack",
        ),
        "prerequisite row ordering mismatch",
    )
    return hashes


def _verify_source(
    record,
    manifest,
    prerequisite_hashes,
    source_root,
    method_hashes,
    checker,
):
    hashes = record.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and len(hashes) == 56,
        "source file count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "source file ordering mismatch",
    )
    checker.require(
        set(hashes) - set(prerequisite_hashes) == {PREFLIGHT_SOURCE},
        "source closure addition mismatch",
    )
    checker.require(
        {name: hashes.get(name) for name in prerequisite_hashes}
        == prerequisite_hashes,
        "inherited source closure mismatch",
    )
    checker.require(
        manifest.get("local_file_sha256") == hashes,
        "manifest local source hashes mismatch",
    )
    checker.require(
        manifest.get("remote_file_sha256") == hashes,
        "manifest remote source hashes mismatch",
    )
    for name, expected in hashes.items():
        path = Path(source_root) / name
        checker.require(path.is_file(), f"missing local source file: {name}")
        checker.require(_is_sha256(expected), f"invalid SHA256: {name}")
        checker.require(
            _sha256_bytes(path.read_bytes()) == expected,
            f"local source SHA256 mismatch: {name}",
        )
    tree_hash = _source_tree_sha256(hashes)
    checker.require(
        record.get("source_tree_sha256") == tree_hash,
        "record source tree mismatch",
    )
    checker.require(
        manifest.get("source_tree_sha256") == tree_hash,
        "manifest source tree mismatch",
    )
    checker.require(
        hashes.get(LLM_ENGINE_SOURCE) == LLM_ENGINE_FILE_SHA256,
        "LLMEngine source hash mismatch",
    )
    checker.require(
        hashes.get(MODEL_RUNNER_SOURCE) == MODEL_RUNNER_FILE_SHA256,
        "ModelRunner source hash mismatch",
    )
    checker.require(
        hashes.get(ACK_SOURCE) == ACK_FILE_SHA256,
        "ack source hash mismatch",
    )
    checker.require(
        record.get("method_source_sha256") == method_hashes,
        "record method hashes mismatch",
    )
    checker.require(
        manifest.get("method_source_sha256") == method_hashes,
        "manifest method hashes mismatch",
    )
    return tree_hash


def _verify_identity_row(row, rank, nonce, expected_status, checker):
    checker.require(
        set(row) == {
            "participant_id",
            "operation",
            "status",
            "attempt_nonce",
            "detail",
        },
        f"identity row fields mismatch: rank={rank}",
    )
    checker.require(
        row.get("participant_id") == rank,
        f"identity participant mismatch: rank={rank}",
    )
    checker.require(
        row.get("operation")
        == "report_tp4_shared_memory_fanout_identity",
        f"identity operation mismatch: rank={rank}",
    )
    checker.require(
        row.get("attempt_nonce") == nonce,
        f"identity nonce mismatch: rank={rank}",
    )
    checker.require(
        row.get("status") == expected_status,
        f"identity status mismatch: rank={rank}",
    )
    expected_detail = (
        "injected rank2 inner error"
        if expected_status == "error"
        else ""
    )
    checker.require(
        row.get("detail") == expected_detail,
        f"identity detail mismatch: rank={rank}",
    )


def _verify_row(row, mode, method_hashes, checker):
    success = mode == "tp4_fanout_success_reverse_completion"
    death = mode == "tp4_fanout_rank2_exit_without_ack"
    exact = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "PASS",
        "mode": mode,
        "observed_user": "sitian",
        "prerequisite_artifact_sha256": PREREQUISITE_SHA256,
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": method_hashes,
        "worker_ranks": [1, 2, 3],
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "segment_unlinked": True,
        "post_unlink_attach_failed": True,
        "dispatch_count": 2,
        "write_count": 2,
        "collector_call_count": 1,
        "exit_envelope_sent": True,
        "child_collected_by_rank": {
            "1": True, "2": True, "3": True,
        },
        "child_exitcodes": {
            "1": 0, "2": 9 if death else 0, "3": 0,
        },
        "collector_poisoned": mode in {
            "tp4_fanout_rank2_ack_exception",
            "tp4_fanout_rank2_exit_without_ack",
        },
        "fanout_validated": success,
    }
    for name, expected in exact.items():
        checker.require(
            row.get(name) == expected,
            f"row field mismatch: {mode} {name}",
        )
    checker.require(
        isinstance(row.get("process_id"), int)
        and not isinstance(row["process_id"], bool)
        and row["process_id"] > 0,
        f"outer PID invalid: {mode}",
    )
    child_ids = row.get("child_process_ids")
    checker.require(
        isinstance(child_ids, dict) and tuple(child_ids) == ("1", "2", "3"),
        f"child PID keys mismatch: {mode}",
    )
    checker.require(
        all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0
            for value in child_ids.values()
        ),
        f"child PID invalid: {mode}",
    )
    checker.require(
        len(set(child_ids.values())) == 3,
        f"child PID uniqueness mismatch: {mode}",
    )
    checker.require(
        row["process_id"] not in set(child_ids.values()),
        f"outer/child PID alias: {mode}",
    )
    checker.require(
        isinstance(row.get("observed_hostname"), str)
        and bool(row["observed_hostname"]),
        f"observed hostname invalid: {mode}",
    )
    name = row.get("shared_memory_name")
    checker.require(
        isinstance(name, str)
        and name.startswith("qwen35-")
        and name != "tinyvllm"
        and len(name) <= 30,
        f"shared_memory_name mismatch: {mode}",
    )
    nonce = row.get("attempt_nonce")
    checker.require(
        isinstance(nonce, str)
        and len(nonce) == 32
        and all(character in "0123456789abcdef" for character in nonce),
        f"attempt_nonce mismatch: {mode}",
    )
    checker.require(
        row.get("envelopes") == [
            {
                "command_id": 0,
                "method_name": "report_qwen35_tp4_fanout_identity",
                "args": [nonce],
                "requires_ack": True,
            },
            {
                "command_id": 1,
                "method_name": "exit",
                "args": [],
                "requires_ack": False,
            },
        ],
        f"envelopes mismatch: {mode}",
    )
    checker.require(
        row.get("write_payload_bytes") == [219, 154],
        f"write_payload_bytes mismatch: {mode}",
    )
    normal_counts = {"1": 2, "2": 2, "3": 2}
    death_counts = {"1": 2, "2": 1, "3": 2}
    for field in (
        "read_count_by_rank",
        "executor_count_by_rank",
        "event_wait_count_by_rank",
        "event_clear_count_by_rank",
    ):
        checker.require(
            row.get(field) == (death_counts if death else normal_counts),
            f"{field} mismatch: {mode}",
        )
    checker.require(
        row.get("event_set_count_by_rank") == normal_counts,
        f"event_set_count_by_rank mismatch: {mode}",
    )
    expected_status = {
        "tp4_fanout_success_reverse_completion": {
            "1": "ok", "2": "ok", "3": "ok",
        },
        "tp4_fanout_rank2_inner_error": {
            "1": "ok", "2": "ok", "3": "ok",
        },
        "tp4_fanout_rank2_ack_exception": {
            "1": "ok", "2": "error", "3": "ok",
        },
        "tp4_fanout_rank2_exit_without_ack": {
            "1": "ok", "2": "absent", "3": "ok",
        },
    }[mode]
    checker.require(
        row.get("ack_status_by_rank") == expected_status,
        f"ack_status_by_rank mismatch: {mode}",
    )
    expected_send_order = [3, 1] if death else [3, 2, 1]
    checker.require(
        row.get("ack_send_order") == expected_send_order,
        f"ack_send_order mismatch: {mode}",
    )
    ordered_results = mode in {
        "tp4_fanout_success_reverse_completion",
        "tp4_fanout_rank2_inner_error",
    }
    checker.require(
        row.get("collector_return_order")
        == ([1, 2, 3] if ordered_results else []),
        f"collector_return_order mismatch: {mode}",
    )
    checker.require(
        row.get("collector_result_participants")
        == ([1, 2, 3] if ordered_results else []),
        f"collector_result_participants mismatch: {mode}",
    )
    results = row.get("ack_result_by_rank")
    errors = row.get("ack_error_by_rank")
    checker.require(
        isinstance(results, dict) and tuple(results) == ("1", "2", "3"),
        f"ack_result_by_rank mismatch: {mode}",
    )
    checker.require(
        isinstance(errors, dict) and tuple(errors) == ("1", "2", "3"),
        f"ack_error_by_rank mismatch: {mode}",
    )
    for rank in WORKER_RANKS:
        key = str(rank)
        status = expected_status[key]
        if status == "ok":
            expected_row_status = (
                "error"
                if mode == "tp4_fanout_rank2_inner_error" and rank == 2
                else "ok"
            )
            _verify_identity_row(
                results[key],
                rank,
                nonce,
                expected_row_status,
                checker,
            )
            checker.require(
                errors[key] == {"error_type": "", "error_detail": ""},
                f"ack error mismatch: {mode} rank={rank}",
            )
        elif status == "error":
            checker.require(
                results[key] is None,
                f"error ack result mismatch: {mode} rank={rank}",
            )
            checker.require(
                errors[key] == {
                    "error_type": "RuntimeError",
                    "error_detail": (
                        "injected rank2 acknowledgement exception"
                    ),
                },
                f"error ack detail mismatch: {mode} rank={rank}",
            )
        else:
            checker.require(
                results[key] is None,
                f"absent ack result mismatch: {mode} rank={rank}",
            )
            checker.require(
                errors[key] == {"error_type": "", "error_detail": ""},
                f"absent ack detail mismatch: {mode} rank={rank}",
            )
    if success:
        rows = row.get("fanout_rows")
        checker.require(
            isinstance(rows, list) and len(rows) == 4,
            "success fanout rows mismatch",
        )
        for rank, identity in enumerate(rows):
            _verify_identity_row(identity, rank, nonce, "ok", checker)
        checker.require(
            row.get("error_detail") == "",
            "success error detail mismatch",
        )
    else:
        checker.require(
            row.get("fanout_rows") is None
            if mode not in {"tp4_fanout_rank2_inner_error"}
            else isinstance(row.get("fanout_rows"), list),
            f"failure fanout rows mismatch: {mode}",
        )
        expected_error = {
            "tp4_fanout_rank2_inner_error": (
                "RuntimeError: TP4 shared-memory fan-out identity failed: "
                "rank=2, detail=injected rank2 inner error"
            ),
            "tp4_fanout_rank2_ack_exception": (
                "RuntimeError: worker command failed: rank=2, "
                "type=RuntimeError, detail=injected rank2 "
                "acknowledgement exception"
            ),
            "tp4_fanout_rank2_exit_without_ack": (
                "RuntimeError: worker rank is not alive while waiting "
                "for acknowledgement: 2"
            ),
        }[mode]
        checker.require(
            row.get("error_detail") == expected_error,
            f"error_detail mismatch: {mode}",
        )


def _verify_static_safety(source_root, checker):
    path = Path(source_root) / PREFLIGHT_SOURCE
    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=PREFLIGHT_SOURCE,
    )
    calls = [
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ]

    def named(name):
        return sum(
            isinstance(node.func, ast.Name) and node.func.id == name
            for node in calls
        )

    def attribute(name):
        return sum(
            isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            for node in calls
        )

    checker.require(
        named("LLMEngine") + attribute("LLMEngine") == 0,
        "LLMEngine construction detected",
    )
    checker.require(
        named("ModelRunner") + attribute("ModelRunner") == 0,
        "ModelRunner construction detected",
    )
    for name in (
        "read_qwen35_checkpoint_metadata",
        "build_qwen35_checkpoint_tensor_plan",
        "prepare_qwen35_checkpoint_candidate_target",
        "build_qwen35_authorized_checkpoint_candidate_loader",
        "load_qwen35_fresh_checkpoint_candidate",
        "Scheduler",
        "schedule",
        "step",
        "cuda",
        "forward",
        "inference",
    ):
        checker.require(
            named(name) + attribute(name) == 0,
            f"forbidden call detected: {name}",
        )
    shared_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "SharedMemory"
    ]
    checker.require(
        sum(
            any(
                keyword.arg == "create"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            for node in shared_calls
        ) == 1,
        "SharedMemory create-site count mismatch",
    )
    checker.require(
        len(shared_calls) == 3,
        "SharedMemory attach/probe-site count mismatch",
    )
    checker.require(
        not any(
            any(
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == "tinyvllm"
                for keyword in node.keywords
            )
            for node in shared_calls
        ),
        "fixed tinyvllm shared-memory name detected",
    )


def _verify(
    run_dir,
    *,
    source_root,
    prerequisite_artifact,
    checker,
):
    run_dir = Path(run_dir)
    artifact_path = run_dir / ARTIFACT_NAME
    manifest_path = run_dir / MANIFEST_NAME
    prerequisite_path = Path(prerequisite_artifact)
    checker.require(artifact_path.is_file(), "missing TP4 artifact")
    checker.require(manifest_path.is_file(), "missing source manifest")
    checker.require(
        prerequisite_path.is_file(),
        "missing prerequisite artifact",
    )
    checker.require(
        _sha256_bytes(prerequisite_path.read_bytes())
        == PREREQUISITE_SHA256,
        "prerequisite artifact SHA256 mismatch",
    )
    record = _read_json(artifact_path)
    manifest = _read_json(manifest_path)
    prerequisite = _read_json(prerequisite_path)
    prerequisite_hashes = _verify_prerequisite(prerequisite, checker)
    method_hashes = _method_hashes(source_root, checker)
    exact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "prerequisite_artifact_sha256": PREREQUISITE_SHA256,
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": method_hashes,
        "fresh_process_per_attempt": True,
    }
    for name, expected in exact.items():
        checker.require(
            record.get(name) == expected,
            f"record field mismatch: {name}",
        )
    for name, expected in {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_dir.name,
        "remote_target": REMOTE_TARGET,
        "prerequisite_artifact_sha256": PREREQUISITE_SHA256,
        "method_source_sha256": method_hashes,
    }.items():
        checker.require(
            manifest.get(name) == expected,
            f"manifest field mismatch: {name}",
        )
    tree_hash = _verify_source(
        record,
        manifest,
        prerequisite_hashes,
        source_root,
        method_hashes,
        checker,
    )
    rows = record.get("rows")
    checker.require(
        isinstance(rows, list) and len(rows) == 4,
        "row count mismatch",
    )
    checker.require(
        tuple(row.get("mode") for row in rows) == ATTEMPT_MODES,
        "row ordering mismatch",
    )
    for row, mode in zip(rows, ATTEMPT_MODES):
        _verify_row(row, mode, method_hashes, checker)
    outer_ids = {row["process_id"] for row in rows}
    child_ids = {
        value
        for row in rows
        for value in row["child_process_ids"].values()
    }
    names = {row["shared_memory_name"] for row in rows}
    nonces = {row["attempt_nonce"] for row in rows}
    checker.require(len(outer_ids) == 4, "outer PID uniqueness mismatch")
    checker.require(len(child_ids) == 12, "child PID uniqueness mismatch")
    checker.require(
        outer_ids.isdisjoint(child_ids),
        "outer/child PID sets overlap",
    )
    checker.require(
        len(names) == 4,
        "shared-memory name uniqueness mismatch",
    )
    checker.require(len(nonces) == 4, "attempt nonce uniqueness mismatch")
    _verify_static_safety(source_root, checker)
    return {
        "row_count": len(rows),
        "unique_process_count": len(outer_ids),
        "unique_child_process_count": len(child_ids),
        "unique_shared_memory_count": len(names),
        "source_file_count": len(record["source_file_sha256"]),
        "source_tree_sha256": tree_hash,
    }


def verify_run(run_dir, *, source_root, prerequisite_artifact):
    checker = Checker()
    try:
        summary = _verify(
            run_dir,
            source_root=source_root,
            prerequisite_artifact=prerequisite_artifact,
            checker=checker,
        )
    except (VerificationError, OSError, SyntaxError, KeyError, TypeError) as error:
        return {
            "status": "FAIL",
            "checks": checker.count,
            "detail": str(error),
        }
    return {
        "status": "PASS",
        "checks": checker.count,
        **summary,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--prerequisite-artifact", required=True)
    arguments = parser.parse_args(argv)
    result = verify_run(
        arguments.run_dir,
        source_root=arguments.source_root,
        prerequisite_artifact=arguments.prerequisite_artifact,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
