from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ARTIFACT_NAME = "engine_ack_transport_preflight.json"
MANIFEST_NAME = "source_manifest.json"
SCHEMA_VERSION = "qwen35.real-binding-engine-ack.v1"
ROW_SCHEMA_VERSION = "qwen35.real-binding-engine-ack-rank.v1"
PREREQUISITE_SHA256 = (
    "79e140190376a01fb7c07cf80202432dd85791dc6112376a334e13ac9a81048a"
)
PREREQUISITE_SOURCE_TREE_SHA256 = (
    "0d69c3cb59a0bab1a3b19c2846bf2326afff71ca0908e53f7ff7a45c36335785"
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
PREFLIGHT_SOURCE = (
    "tools/qwen35_real_binding_engine_ack_transport_preflight.py"
)
METHOD_SPECS = {
    "call_model_runner_acknowledged": (
        LLM_ENGINE_SOURCE,
        "LLMEngine",
        ("self", "method_name"),
        "args",
        ("timeout_s",),
        "6eed126b80c9c823ceff37cc51273735d656c2b1be963bbea2bbd4ad9da9f14d",
    ),
    "bind_qwen35_loaded_checkpoint_candidates": (
        LLM_ENGINE_SOURCE,
        "LLMEngine",
        ("self",),
        None,
        ("timeout_s",),
        "82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c",
    ),
    "dispatch_command": (
        MODEL_RUNNER_SOURCE,
        "ModelRunner",
        ("self", "method_name"),
        "args",
        ("requires_ack",),
        "9a63e40ef7d16b6300e70d41c2f05575a50adbf6dc04942677034d6bee363342",
    ),
}
WORKER_MODES = (
    "tp1_success",
    "tp1_local_binding_error",
    "tp2_success",
    "tp2_worker_binding_error",
    "tp2_worker_ack_exception",
    "tp2_worker_exit_without_ack",
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
EXPECTED_ENVELOPE = {
    "command_id": 0,
    "method_name": (
        "bind_published_qwen35_loaded_checkpoint_candidate"
    ),
    "args": [],
    "requires_ack": True,
}
MODEL_FINGERPRINT = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)


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
    payload = json.dumps(
        dict(hashes),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _is_sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _method_node(source, filename, class_name, method_name, checker):
    tree = ast.parse(source, filename=filename)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == class_name
    ]
    checker.require(
        len(classes) == 1,
        f"class count mismatch: {class_name}",
    )
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    ]
    checker.require(
        len(methods) == 1,
        f"method count mismatch: {method_name}",
    )
    return methods[0]


def _method_hashes(source_root, checker):
    root = Path(source_root)
    source_cache = {}
    observed = {}
    for method_name, (
        filename,
        class_name,
        expected_positional,
        expected_vararg,
        expected_keyword_only,
        expected_hash,
    ) in METHOD_SPECS.items():
        if filename not in source_cache:
            source_cache[filename] = (
                root / filename
            ).read_text(encoding="utf-8")
        source = source_cache[filename]
        node = _method_node(
            source,
            filename,
            class_name,
            method_name,
            checker,
        )
        positional = tuple(
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
            )
        )
        vararg = (
            node.args.vararg.arg
            if node.args.vararg is not None
            else None
        )
        keyword_only = tuple(
            argument.arg for argument in node.args.kwonlyargs
        )
        checker.require(
            positional == expected_positional,
            f"method positional arguments mismatch: {method_name}",
        )
        checker.require(
            vararg == expected_vararg,
            f"method vararg mismatch: {method_name}",
        )
        checker.require(
            keyword_only == expected_keyword_only,
            f"method keyword-only arguments mismatch: {method_name}",
        )
        segment = ast.get_source_segment(source, node)
        checker.require(
            isinstance(segment, str),
            f"method source segment missing: {method_name}",
        )
        observed[method_name] = _sha256_bytes(
            segment.encode("utf-8")
        )
        checker.require(
            observed[method_name] == expected_hash,
            f"method SHA256 mismatch: {method_name}",
        )
    return observed


def _prerequisite_rows(prerequisite, checker):
    checker.require(
        prerequisite.get("schema_version")
        == "qwen35.real-checkpoint-model-runner-published-binding.v1",
        "prerequisite schema mismatch",
    )
    checker.require(
        prerequisite.get("status") == "PASS",
        "prerequisite status mismatch",
    )
    checker.require(
        prerequisite.get("source_tree_sha256")
        == PREREQUISITE_SOURCE_TREE_SHA256,
        "prerequisite source tree mismatch",
    )
    hashes = prerequisite.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and len(hashes) == 51,
        "prerequisite source file count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "prerequisite source ordering mismatch",
    )
    rows = prerequisite.get("rows")
    checker.require(
        isinstance(rows, list) and len(rows) == 6,
        "prerequisite row count mismatch",
    )
    contexts = tuple(
        (
            row.get("tp_size"),
            row.get("tp_rank"),
            row.get("mode"),
        )
        for row in rows
    )
    checker.require(
        contexts
        == (
            (1, 0, "success"),
            (1, 0, "injected_bridge_conflict"),
            (2, 0, "success"),
            (2, 0, "injected_bridge_conflict"),
            (2, 1, "success"),
            (2, 1, "injected_bridge_conflict"),
        ),
        "prerequisite row ordering mismatch",
    )
    process_ids = [row.get("process_id") for row in rows]
    checker.require(
        all(
            isinstance(process_id, int)
            and not isinstance(process_id, bool)
            and process_id > 0
            for process_id in process_ids
        ),
        "prerequisite process ID invalid",
    )
    checker.require(
        len(set(process_ids)) == 6,
        "prerequisite process IDs not unique",
    )
    result = {}
    for context, row in zip(contexts, rows):
        method_row = row.get("method_row")
        checker.require(
            isinstance(method_row, dict),
            f"prerequisite method row invalid: {context}",
        )
        result[context] = method_row
    return hashes, result


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
        isinstance(hashes, dict) and len(hashes) == 54,
        "source file count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "source file ordering mismatch",
    )
    checker.require(
        set(hashes) - set(prerequisite_hashes)
        == {
            LLM_ENGINE_SOURCE,
            ACK_SOURCE,
            PREFLIGHT_SOURCE,
        },
        "source closure additions mismatch",
    )
    checker.require(
        {
            name: hashes.get(name)
            for name in prerequisite_hashes
        }
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
        checker.require(
            path.is_file(),
            f"missing local source file: {name}",
        )
        checker.require(
            _is_sha256(expected),
            f"invalid source SHA256: {name}",
        )
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
        "record LLMEngine source hash mismatch",
    )
    checker.require(
        hashes.get(MODEL_RUNNER_SOURCE) == MODEL_RUNNER_FILE_SHA256,
        "record ModelRunner source hash mismatch",
    )
    checker.require(
        hashes.get(ACK_SOURCE) == ACK_FILE_SHA256,
        "record acknowledgement source hash mismatch",
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


def _expected_binding_rows(mode, prerequisite_rows):
    if mode == "tp1_success":
        return [prerequisite_rows[(1, 0, "success")]]
    if mode == "tp2_success":
        return [
            prerequisite_rows[(2, 0, "success")],
            prerequisite_rows[(2, 1, "success")],
        ]
    return None


def _verify_binding_method_row(row, expected, context, checker):
    checker.require(
        set(row)
        == {
            "participant_id",
            "operation",
            "status",
            "model_fingerprint",
            "layout_fingerprint",
            "dtype",
            "detail",
        },
        f"binding method row fields mismatch: {context}",
    )
    for name, expected_value in expected.items():
        checker.require(
            row.get(name) == expected_value,
            f"binding method row mismatch: {context} {name}",
        )


def _verify_row(
    row,
    mode,
    prerequisite_rows,
    method_hashes,
    checker,
):
    tp2 = mode.startswith("tp2")
    success = mode in {"tp1_success", "tp2_success"}
    expected_ack = {
        "tp1_success": "absent",
        "tp1_local_binding_error": "absent",
        "tp2_success": "ok",
        "tp2_worker_binding_error": "ok",
        "tp2_worker_ack_exception": "error",
        "tp2_worker_exit_without_ack": "absent",
    }[mode]
    exact = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "PASS",
        "mode": mode,
        "observed_user": "sitian",
        "timeout_s": 2.0,
        "prerequisite_artifact_sha256": PREREQUISITE_SHA256,
        "llm_engine_file_sha256": LLM_ENGINE_FILE_SHA256,
        "model_runner_file_sha256": MODEL_RUNNER_FILE_SHA256,
        "ack_file_sha256": ACK_FILE_SHA256,
        "method_source_sha256": method_hashes,
        "dispatch_count": 1 if tp2 else 0,
        "collector_call_count": 1 if tp2 else 0,
        "command_send_count": 1 if tp2 else 0,
        "child_receive_count": 1 if tp2 else 0,
        "acknowledgement_status": expected_ack,
        "completion_committed": success,
        "repeat_zero_dispatch": success,
        "collector_poisoned": mode
        in {
            "tp2_worker_ack_exception",
            "tp2_worker_exit_without_ack",
        },
        "child_collected": True,
    }
    for name, expected in exact.items():
        checker.require(
            row.get(name) == expected,
            f"row field mismatch: {mode} {name}",
        )
    checker.require(
        isinstance(row.get("observed_hostname"), str)
        and bool(row["observed_hostname"]),
        f"observed hostname invalid: {mode}",
    )
    checker.require(
        isinstance(row.get("process_id"), int)
        and not isinstance(row.get("process_id"), bool)
        and row["process_id"] > 0,
        f"process ID invalid: {mode}",
    )
    if tp2:
        checker.require(
            isinstance(row.get("child_process_id"), int)
            and not isinstance(row.get("child_process_id"), bool)
            and row["child_process_id"] > 0,
            f"child process ID invalid: {mode}",
        )
        checker.require(
            row["child_process_id"] != row["process_id"],
            f"child process aliases outer process: {mode}",
        )
        checker.require(
            row.get("child_exitcode") == 0,
            f"child exit code mismatch: {mode}",
        )
        checker.require(
            row.get("envelope") == EXPECTED_ENVELOPE,
            f"envelope mismatch: {mode}",
        )
    else:
        for name in (
            "child_process_id",
            "child_exitcode",
            "envelope",
        ):
            checker.require(
                row.get(name) is None,
                f"TP1 child field mismatch: {mode} {name}",
            )
    expected_rows = _expected_binding_rows(
        mode,
        prerequisite_rows,
    )
    checker.require(
        row.get("binding_rows") == expected_rows,
        f"binding rows mismatch: {mode}",
    )
    if expected_rows is not None:
        for index, (observed, expected) in enumerate(
            zip(row["binding_rows"], expected_rows)
        ):
            _verify_binding_method_row(
                observed,
                expected,
                (mode, index),
                checker,
            )
        local = expected_rows[0]
        expected_configuration = [
            MODEL_FINGERPRINT,
            local["layout_fingerprint"],
            "bfloat16",
            2.0,
        ]
        checker.require(
            row.get("completion_configuration")
            == expected_configuration,
            f"completion configuration mismatch: {mode}",
        )
        checker.require(
            row.get("error_detail") == "",
            f"success error detail mismatch: {mode}",
        )
    else:
        checker.require(
            row.get("completion_configuration") is None,
            f"failure completion configuration mismatch: {mode}",
        )
    expected_errors = {
        "tp1_local_binding_error": (
            "RuntimeError: loaded checkpoint candidate binding failed: "
            "rank=0, detail=RuntimeError: a different hybrid state "
            "runtime bridge is already installed"
        ),
        "tp2_worker_binding_error": (
            "RuntimeError: loaded checkpoint candidate binding failed: "
            "rank=1, detail=RuntimeError: a different hybrid state "
            "runtime bridge is already installed"
        ),
        "tp2_worker_ack_exception": (
            "RuntimeError: worker command failed: rank=1, "
            "type=RuntimeError, detail=injected worker "
            "acknowledgement exception"
        ),
        "tp2_worker_exit_without_ack": (
            "RuntimeError: rank 1 acknowledgement receive failed: "
        ),
    }
    if not success:
        checker.require(
            row.get("error_detail") == expected_errors[mode],
            f"error detail mismatch: {mode}",
        )
    expected_ack_error = {
        "tp2_worker_ack_exception": (
            "RuntimeError",
            "injected worker acknowledgement exception",
        ),
    }.get(mode, ("", ""))
    checker.require(
        row.get("acknowledgement_error_type")
        == expected_ack_error[0],
        f"acknowledgement_error_type mismatch: {mode}",
    )
    checker.require(
        row.get("acknowledgement_error_detail")
        == expected_ack_error[1],
        f"acknowledgement_error_detail mismatch: {mode}",
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
            isinstance(node.func, ast.Name)
            and node.func.id == name
            for node in calls
        )

    def attribute(name):
        return sum(
            isinstance(node.func, ast.Attribute)
            and node.func.attr == name
            for node in calls
        )

    imports = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "__import__"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    ]
    checker.require(
        not any(
            node.args[0].value
            in {
                "tinyvllm.engine.llm_engine",
                "tinyvllm.engine.model_runner",
            }
            for node in imports
        ),
        "production Engine or ModelRunner import detected",
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
    checker.require(
        attribute("ModelRunnerCommandAckCollector") == 1,
        "ack collector constructor count mismatch",
    )
    checker.require(
        attribute("execute_acknowledged_command") == 1,
        "ack executor call count mismatch",
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
    checker.require(
        artifact_path.is_file(),
        "missing Engine acknowledgement transport artifact",
    )
    checker.require(
        manifest_path.is_file(),
        "missing source manifest",
    )
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
    prerequisite_hashes, prerequisite_rows = _prerequisite_rows(
        prerequisite,
        checker,
    )
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
    run_tag = run_dir.name
    manifest_exact = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "remote_target": REMOTE_TARGET,
        "prerequisite_artifact_sha256": PREREQUISITE_SHA256,
        "method_source_sha256": method_hashes,
    }
    for name, expected in manifest_exact.items():
        checker.require(
            manifest.get(name) == expected,
            f"manifest field mismatch: {name}",
        )
    checker.require(
        manifest.get("remote_source_dir")
        == (
            "/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-real-binding-engine-ack-runs/"
            f"{run_tag}/source"
        ),
        "manifest remote source directory mismatch",
    )
    checker.require(
        manifest.get("remote_prerequisite_artifact")
        == (
            "/data00/home/sitian/sitian-workspace01/tllm/"
            "qwen35-real-binding-engine-ack-runs/"
            f"{run_tag}/"
            "model_runner_published_candidate_binding_preflight.json"
        ),
        "manifest remote prerequisite path mismatch",
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
        isinstance(rows, list) and len(rows) == 6,
        "transport row count mismatch",
    )
    checker.require(
        tuple(row.get("mode") for row in rows) == WORKER_MODES,
        "transport row ordering mismatch",
    )
    for row, mode in zip(rows, WORKER_MODES):
        _verify_row(
            row,
            mode,
            prerequisite_rows,
            method_hashes,
            checker,
        )
    outer_ids = [row["process_id"] for row in rows]
    child_ids = [
        row["child_process_id"]
        for row in rows
        if row["mode"].startswith("tp2")
    ]
    checker.require(
        len(set(outer_ids)) == 6,
        "outer process IDs not unique",
    )
    checker.require(
        len(child_ids) == 4 and len(set(child_ids)) == 4,
        "child process IDs not unique",
    )
    checker.require(
        set(outer_ids).isdisjoint(child_ids),
        "outer and child process IDs overlap",
    )
    _verify_static_safety(source_root, checker)
    return {
        "status": "PASS",
        "checks": checker.count,
        "row_count": len(rows),
        "unique_process_count": len(set(outer_ids)),
        "unique_child_process_count": len(set(child_ids)),
        "source_file_count": len(record["source_file_sha256"]),
        "source_tree_sha256": tree_hash,
        "artifact_sha256": _sha256_bytes(artifact_path.read_bytes()),
        "source_manifest_sha256": _sha256_bytes(
            manifest_path.read_bytes()
        ),
    }


def verify_run(
    run_dir,
    *,
    source_root,
    prerequisite_artifact,
):
    checker = Checker()
    try:
        return _verify(
            run_dir,
            source_root=source_root,
            prerequisite_artifact=prerequisite_artifact,
            checker=checker,
        )
    except (
        VerificationError,
        OSError,
        UnicodeDecodeError,
        SyntaxError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        return {
            "status": "FAIL",
            "checks": checker.count,
            "detail": str(error),
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--prerequisite-artifact",
        required=True,
    )
    arguments = parser.parse_args(argv)
    result = verify_run(
        arguments.run_dir,
        source_root=arguments.source_root,
        prerequisite_artifact=arguments.prerequisite_artifact,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
