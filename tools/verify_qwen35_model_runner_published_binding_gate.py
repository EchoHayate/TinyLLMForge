from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ARTIFACT_NAME = (
    "model_runner_published_candidate_binding_preflight.json"
)
MANIFEST_NAME = "source_manifest.json"
SCHEMA_VERSION = (
    "qwen35.real-checkpoint-model-runner-published-binding.v1"
)
ROW_SCHEMA_VERSION = (
    "qwen35.real-checkpoint-model-runner-published-binding-rank.v1"
)
PREREQUISITE_SHA256 = (
    "d5e6de1ec4a308945897c125eaf7ecff57c44710600ce607db4fd0ae7fb90e18"
)
MODEL_RUNNER_SOURCE = "tinyvllm/engine/model_runner.py"
MODEL_RUNNER_FILE_SHA256 = (
    "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
)
METHOD_SOURCE_SHA256 = {
    "publish_qwen35_loaded_checkpoint_candidate": (
        "37f95954c287d5dd0e8883f299d7049e66dcb3c79806624eb6da3ca7d51a6d4f"
    ),
    "bind_qwen35_hybrid_model_owner": (
        "462e2fefe22e90e60b85c786de6a95e7eaaae31bd9b257025088cd767555ee25"
    ),
    "bind_qwen35_loaded_checkpoint_candidate": (
        "a14e6856ad74eb935116075ee6fe81516c8e212f89914e0fcd55bb39e86d63e0"
    ),
    "bind_published_qwen35_loaded_checkpoint_candidate": (
        "aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd"
    ),
}
WORKER_CONTEXTS = (
    (1, 0, "success"),
    (1, 0, "injected_bridge_conflict"),
    (2, 0, "success"),
    (2, 0, "injected_bridge_conflict"),
    (2, 1, "success"),
    (2, 1, "injected_bridge_conflict"),
)
MEMORY_CEILINGS_KIB = {
    1: {
        "total": 10485760,
        "post_torch": 10223616,
        "post_metadata": 9961472,
    },
    2: {
        "total": 7340032,
        "post_torch": 7077888,
        "post_metadata": 6815744,
    },
}
CLAIM_BOUNDARY = (
    "Production binding-method correctness only; no Engine, CUDA, forward, "
    "inference, latency, throughput, cache, memory reduction, compression, "
    "or quality claim."
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


def _method_hashes(source_root, checker):
    path = Path(source_root) / MODEL_RUNNER_SOURCE
    payload = path.read_bytes()
    checker.require(
        _sha256_bytes(payload) == MODEL_RUNNER_FILE_SHA256,
        "ModelRunner file SHA256 mismatch",
    )
    source = payload.decode("utf-8")
    tree = ast.parse(source, filename=MODEL_RUNNER_SOURCE)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "ModelRunner"
    ]
    checker.require(
        len(classes) == 1,
        "ModelRunner class count mismatch",
    )
    observed = {}
    for name, expected in METHOD_SOURCE_SHA256.items():
        methods = [
            node
            for node in classes[0].body
            if isinstance(node, ast.FunctionDef)
            and node.name == name
        ]
        checker.require(
            len(methods) == 1,
            f"ModelRunner method count mismatch: {name}",
        )
        segment = ast.get_source_segment(source, methods[0])
        checker.require(
            isinstance(segment, str),
            f"ModelRunner method segment missing: {name}",
        )
        observed[name] = _sha256_bytes(segment.encode("utf-8"))
        checker.require(
            observed[name] == expected,
            f"ModelRunner method SHA256 mismatch: {name}",
        )
    return observed


def _prerequisite_rows(prerequisite, checker):
    checker.require(
        prerequisite.get("status") == "PASS",
        "prerequisite status mismatch",
    )
    rows = prerequisite.get("rows")
    checker.require(
        isinstance(rows, list) and len(rows) == 6,
        "prerequisite row count mismatch",
    )
    success = {}
    for row in rows:
        if row.get("mode") != "success":
            continue
        key = (row.get("tp_size"), row.get("tp_rank"))
        checker.require(
            key not in success,
            "duplicate prerequisite success row",
        )
        success[key] = row
    checker.require(
        tuple(success) == ((1, 0), (2, 0), (2, 1)),
        "prerequisite success rows mismatch",
    )
    return success


def _verify_source(
    record,
    manifest,
    source_root,
    method_hashes,
    checker,
):
    hashes = record.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and len(hashes) == 51,
        "source file count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "source file ordering mismatch",
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
            _sha256_bytes(path.read_bytes()) == expected,
            f"local source SHA256 mismatch: {name}",
        )
        checker.require(
            _is_sha256(expected),
            f"invalid source SHA256: {name}",
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
        hashes.get(MODEL_RUNNER_SOURCE)
        == MODEL_RUNNER_FILE_SHA256,
        "record ModelRunner source hash mismatch",
    )
    checker.require(
        record.get("model_runner_method_sha256") == method_hashes,
        "record method hashes mismatch",
    )
    checker.require(
        manifest.get("model_runner_method_sha256")
        == method_hashes,
        "manifest method hashes mismatch",
    )
    return tree_hash


def _verify_memory(row, checker):
    memory = row.get("memory")
    names = (
        "before",
        "after_torch",
        "after_metadata",
        "after_pool",
        "after_target",
        "after_binding_clear",
    )
    checker.require(
        isinstance(memory, dict) and set(memory) == set(names),
        "memory point set mismatch",
    )
    for name in names:
        point = memory[name]
        checker.require(
            isinstance(point, dict)
            and isinstance(point.get("vmhwm_kib"), int)
            and isinstance(point.get("vmrss_kib"), int),
            f"memory point invalid: {name}",
        )
    observed = (
        row.get("total_vmhwm_increment_kib"),
        row.get("post_torch_vmhwm_increment_kib"),
        row.get("post_metadata_vmhwm_increment_kib"),
    )
    recomputed = (
        memory["after_binding_clear"]["vmhwm_kib"]
        - memory["before"]["vmhwm_kib"],
        memory["after_binding_clear"]["vmhwm_kib"]
        - memory["after_torch"]["vmhwm_kib"],
        memory["after_binding_clear"]["vmhwm_kib"]
        - memory["after_metadata"]["vmhwm_kib"],
    )
    checker.require(
        observed == recomputed,
        "memory delta mismatch",
    )
    ceilings = MEMORY_CEILINGS_KIB[row["tp_size"]]
    for value, label in zip(
        observed,
        ("total", "post_torch", "post_metadata"),
    ):
        checker.require(
            value <= ceilings[label],
            f"memory ceiling exceeded: {label}",
        )


def _verify_row(row, prerequisite, context, checker):
    tp_size, tp_rank, mode = context
    exact = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "PASS",
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "mode": mode,
        "observed_user": "sitian",
        "publication_method_call_count": 1,
        "outer_binding_method_call_count": 1,
        "candidate_binding_method_call_count": 1,
        "owner_binding_method_call_count": 1,
        "adapter_call_count": 1,
        "provider_call_count": 1,
        "production_publish_call_count": 1,
        "production_slot_visibility_verified": True,
        "published_candidate_identity_verified": True,
        "candidate_installed": mode == "success",
        "owner_binding_visible": mode == "success",
        "runtime_bridge_binding_visible": mode == "success",
        "runtime_identity_binding_visible": mode == "success",
        "runtime_identity_owner_visible": mode == "success",
        "injected_bridge_preserved": (
            mode == "injected_bridge_conflict"
        ),
        "binding_state_pristine": (
            mode == "injected_bridge_conflict"
        ),
        "dtype": "bfloat16",
        "loaded_state_verified": True,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "aggregate_hash_verified": True,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "target_consumed_before": False,
        "target_consumed_after": True,
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "all_private_binding_objects_collected": True,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
    }
    for name, expected in exact.items():
        checker.require(
            row.get(name) == expected,
            f"row field mismatch: {context} {name}",
        )
    checker.require(
        isinstance(row.get("process_id"), int)
        and not isinstance(row.get("process_id"), bool)
        and row["process_id"] > 0,
        f"invalid process ID: {context}",
    )
    checker.require(
        _is_sha256(row.get("layout_fingerprint")),
        f"invalid layout fingerprint: {context}",
    )
    expected_method_row = {
        "participant_id": tp_rank,
        "operation": "bind_loaded_checkpoint_candidate",
        "status": "bound" if mode == "success" else "error",
        "model_fingerprint": (
            record_model_fingerprint(row)
            if mode == "success"
            else ""
        ),
        "layout_fingerprint": (
            row["layout_fingerprint"] if mode == "success" else ""
        ),
        "dtype": "bfloat16" if mode == "success" else "",
        "detail": (
            ""
            if mode == "success"
            else (
                "RuntimeError: a different hybrid state runtime "
                "bridge is already installed"
            )
        ),
    }
    checker.require(
        row.get("method_row") == expected_method_row,
        f"method row mismatch: {context}",
    )
    expected_collected = {
        "runner": True,
        "production_slot": True,
        "candidate": True,
        "owner": True,
        "runtime_bridge": True,
        "model": True,
        "pool": True,
        "target": True,
        (
            "runtime_identity"
            if mode == "success"
            else "injected_bridge"
        ): True,
    }
    checker.require(
        row.get("collected_private_objects") == expected_collected,
        f"collected object mismatch: {context}",
    )
    current_binding = row.get("binding_destination_sha256")
    expected_binding = prerequisite.get(
        "binding_destination_sha256"
    )
    checker.require(
        isinstance(current_binding, list)
        and len(current_binding) == 320,
        f"binding hash count mismatch: {context}",
    )
    checker.require(
        isinstance(expected_binding, list)
        and len(expected_binding) == 320,
        f"prerequisite binding hash count mismatch: {context}",
    )
    for index, (current, expected) in enumerate(
        zip(current_binding, expected_binding)
    ):
        checker.require(
            current == expected,
            f"binding prerequisite value mismatch: {context} {index}",
        )
    checker.require(
        row.get("phase_destination_sha256")
        == prerequisite.get("phase_destination_sha256"),
        f"phase prerequisite value mismatch: {context}",
    )
    checker.require(
        row.get("aggregate_destination_sha256")
        == prerequisite.get("aggregate_destination_sha256"),
        f"aggregate prerequisite value mismatch: {context}",
    )
    checker.require(
        row.get("loader_stats") == prerequisite.get("loader_stats"),
        f"loader stats mismatch: {context}",
    )
    _verify_memory(row, checker)


def record_model_fingerprint(row):
    method_row = row.get("method_row")
    if isinstance(method_row, dict):
        value = method_row.get("model_fingerprint")
        if _is_sha256(value):
            return value
    return (
        "3e650a908234771c3cf1ac4e20c4d38f"
        "e69982efedaf4a3e631ad0b14aad7dd0"
    )


def _verify(
    run_dir,
    source_root,
    prerequisite_artifact,
):
    checker = Checker()
    run_dir = Path(run_dir)
    artifact_path = run_dir / ARTIFACT_NAME
    manifest_path = run_dir / MANIFEST_NAME
    checker.require(run_dir.is_dir(), "run directory does not exist")
    checker.require(artifact_path.is_file(), "result artifact missing")
    checker.require(manifest_path.is_file(), "source manifest missing")
    prerequisite_path = Path(prerequisite_artifact)
    checker.require(
        prerequisite_path.is_file(),
        "prerequisite artifact missing",
    )
    checker.require(
        _sha256_bytes(prerequisite_path.read_bytes())
        == PREREQUISITE_SHA256,
        "prerequisite artifact SHA256 mismatch",
    )
    record = _read_json(artifact_path)
    manifest = _read_json(manifest_path)
    prerequisite = _read_json(prerequisite_path)
    checker.require(
        record.get("schema_version") == SCHEMA_VERSION,
        "record schema mismatch",
    )
    checker.require(
        manifest.get("schema_version") == SCHEMA_VERSION,
        "manifest schema mismatch",
    )
    checker.require(record.get("status") == "PASS", "record not PASS")
    checker.require(
        record.get("prerequisite_artifact_sha256")
        == PREREQUISITE_SHA256,
        "record prerequisite SHA256 mismatch",
    )
    checker.require(
        manifest.get("prerequisite_artifact_sha256")
        == PREREQUISITE_SHA256,
        "manifest prerequisite SHA256 mismatch",
    )
    methods = _method_hashes(source_root, checker)
    tree_hash = _verify_source(
        record,
        manifest,
        source_root,
        methods,
        checker,
    )
    prerequisite_rows = _prerequisite_rows(
        prerequisite,
        checker,
    )
    rows = record.get("rows")
    checker.require(
        isinstance(rows, list) and len(rows) == 6,
        "result row count mismatch",
    )
    observed_contexts = [
        (row.get("tp_size"), row.get("tp_rank"), row.get("mode"))
        for row in rows
    ]
    checker.require(
        observed_contexts == list(WORKER_CONTEXTS),
        "worker context ordering mismatch",
    )
    for row, context in zip(rows, WORKER_CONTEXTS):
        _verify_row(
            row,
            prerequisite_rows[context[:2]],
            context,
            checker,
        )
    process_ids = [row["process_id"] for row in rows]
    checker.require(
        len(set(process_ids)) == 6,
        "worker process IDs are not unique",
    )
    return {
        "status": "PASS",
        "checks": checker.count,
        "row_count": len(rows),
        "unique_process_count": len(set(process_ids)),
        "source_file_count": len(record["source_file_sha256"]),
        "source_tree_sha256": tree_hash,
        "model_runner_method_sha256": methods,
        "artifact_sha256": _sha256_bytes(artifact_path.read_bytes()),
        "source_manifest_sha256": _sha256_bytes(
            manifest_path.read_bytes()
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def verify_run(
    run_dir,
    *,
    source_root,
    prerequisite_artifact,
):
    try:
        return _verify(
            run_dir,
            source_root,
            prerequisite_artifact,
        )
    except (OSError, UnicodeDecodeError, VerificationError) as error:
        return {
            "status": "FAIL",
            "checks": 0,
            "detail": str(error),
            "claim_boundary": CLAIM_BOUNDARY,
        }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--prerequisite-artifact", required=True)
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
