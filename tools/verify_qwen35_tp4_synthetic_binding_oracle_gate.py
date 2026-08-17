from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ARTIFACT_NAME = "tp4_synthetic_binding_oracle_preflight.json"
MANIFEST_NAME = "source_manifest.json"
SCHEMA_VERSION = "qwen35.tp4-synthetic-binding-oracle-gate.v1"
ROW_SCHEMA_VERSION = "qwen35.tp4-synthetic-binding-oracle-rank.v1"
TP4_SHA256 = (
    "ec9c07ba903859dbc616dc6c799db4f977284539f9b09cdd85cc57da1a334f8a"
)
TP4_SOURCE_TREE_SHA256 = (
    "ec7b0dee43a06c47b72f8ac14ab26518845f57f070e6c27d394bb4c328644403"
)
ORACLE_SHA256 = (
    "1fdc3d64178308d6d26242805433ee31012890f6824a13826075db1e6937431e"
)
MODEL_FINGERPRINT = (
    "b48e29c9ea266197c56fe3e9133c65d378c682e9437d9e1299c4fa93d0241bd9"
)
LAYOUT_FINGERPRINT = (
    "fe2db881dc909cbd05894a4e194273f4692caed1c4df967e810330324248e2c6"
)
ALT_MODEL = (
    "bf1a56f2bf513a7a8946c4245a5cf3ab53d1853d4ecba9457e88a1636109a206"
)
ALT_LAYOUT = (
    "9ccdb4f0c9c2ad4003d1539bf7ba76188ce59686cd302216371b301e031dc1a9"
)
LLM_ENGINE_SOURCE = "tinyvllm/engine/llm_engine.py"
MODEL_RUNNER_SOURCE = "tinyvllm/engine/model_runner.py"
ACK_SOURCE = "tinyvllm/engine/model_runner_command_ack.py"
PREFLIGHT_SOURCE = (
    "tools/qwen35_tp4_synthetic_binding_oracle_preflight.py"
)
FILE_HASHES = {
    LLM_ENGINE_SOURCE: (
        "6cf68dc76641bf772c01d31fd60ee42cbab82e3c62a0ee8aa154dbe802c727ae"
    ),
    MODEL_RUNNER_SOURCE: (
        "0cba7e97b2d3425186722f53a0c14e7b01e2c90e190e6cdf2c9472517bcda849"
    ),
    ACK_SOURCE: (
        "ca28babca5cc725d8c9bf0e3e057fa4b0cabfd847bf0c052c40876fbc148c61b"
    ),
}
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
    "bind_qwen35_loaded_checkpoint_candidates": (
        LLM_ENGINE_SOURCE,
        "LLMEngine",
        ("self",),
        None,
        ("timeout_s",),
        "82c0528d6b06ae8d67812d1a8802e8163aadb4886afc3894bf28a0cf35c3c84c",
    ),
}
ATTEMPT_MODES = (
    "tp4_synthetic_binding_success",
    "tp4_synthetic_rank2_model_mismatch",
    "tp4_synthetic_rank2_layout_mismatch",
    "tp4_synthetic_rank2_dtype_mismatch",
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
SHARED_MEMORY_CAPACITY = 2**20


class VerificationError(ValueError):
    pass


class Checker:
    def __init__(self):
        self.count = 0

    def require(self, condition, detail):
        self.count += 1
        if not condition:
            raise VerificationError(detail)


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON {Path(path).name}: {error}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{Path(path).name} must be an object")
    return value


def _tree_hash(hashes):
    return _sha(_canonical(dict(hashes)))


def _is_sha(value):
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
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        checker.require(len(classes) == 1, f"class count mismatch: {name}")
        methods = [
            node for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        checker.require(len(methods) == 1, f"method count mismatch: {name}")
        node = methods[0]
        checker.require(
            tuple(
                item.arg
                for item in (*node.args.posonlyargs, *node.args.args)
            ) == positional,
            f"method positional mismatch: {name}",
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
            tuple(item.arg for item in node.args.kwonlyargs)
            == keyword_only,
            f"method keyword mismatch: {name}",
        )
        segment = ast.get_source_segment(source, node)
        checker.require(isinstance(segment, str), f"method source missing: {name}")
        observed[name] = _sha(segment.encode("utf-8"))
        checker.require(
            observed[name] == expected_hash,
            f"method hash mismatch: {name}",
        )
    return observed


def _verify_tp4(record, checker):
    checker.require(
        record.get("schema_version") == "qwen35.tp4-shared-memory-fanout.v1",
        "TP4 schema mismatch",
    )
    checker.require(record.get("status") == "PASS", "TP4 status mismatch")
    checker.require(
        record.get("source_tree_sha256") == TP4_SOURCE_TREE_SHA256,
        "TP4 source tree mismatch",
    )
    hashes = record.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and len(hashes) == 56,
        "TP4 source count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "TP4 source ordering mismatch",
    )
    checker.require(
        tuple(row.get("mode") for row in record.get("rows", ()))
        == (
            "tp4_fanout_success_reverse_completion",
            "tp4_fanout_rank2_inner_error",
            "tp4_fanout_rank2_ack_exception",
            "tp4_fanout_rank2_exit_without_ack",
        ),
        "TP4 rows mismatch",
    )
    return hashes


def _verify_oracle(record, checker):
    exact = {
        "schema_version": "qwen35.tp4-synthetic-binding-oracle.v1",
        "provenance": "synthetic-construction-free-oracle",
        "claim_boundary": "not-real-checkpoint-binding",
        "tensor_payload": "absent",
        "canonical_json": "sort_keys=true,separators=(comma,colon),utf8",
        "model_fingerprint": MODEL_FINGERPRINT,
        "layout_fingerprint": LAYOUT_FINGERPRINT,
        "alternate_model_fingerprint": ALT_MODEL,
        "alternate_layout_fingerprint": ALT_LAYOUT,
        "dtype": "bfloat16",
    }
    for name, expected in exact.items():
        checker.require(
            record.get(name) == expected,
            f"oracle {name} mismatch",
        )
    checker.require(
        _sha(_canonical(record.get("model_descriptor")))
        == MODEL_FINGERPRINT,
        "oracle model descriptor mismatch",
    )
    checker.require(
        _sha(_canonical(record.get("layout_descriptor")))
        == LAYOUT_FINGERPRINT,
        "oracle layout descriptor mismatch",
    )
    checker.require(
        _sha(_canonical({
            **record["model_descriptor"],
            "revision": "mismatch-v1",
        })) == ALT_MODEL,
        "oracle alternate model mismatch",
    )
    checker.require(
        _sha(_canonical({
            **record["layout_descriptor"],
            "revision": "mismatch-v1",
        })) == ALT_LAYOUT,
        "oracle alternate layout mismatch",
    )
    cases = record.get("cases")
    checker.require(
        isinstance(cases, list) and len(cases) == 4,
        "oracle case count mismatch",
    )
    checker.require(
        tuple(case.get("mode") for case in cases) == ATTEMPT_MODES,
        "oracle case ordering mismatch",
    )
    case_map = {case["mode"]: case["rows"] for case in cases}
    baseline = case_map[ATTEMPT_MODES[0]]
    changed = {
        ATTEMPT_MODES[0]: None,
        ATTEMPT_MODES[1]: "model_fingerprint",
        ATTEMPT_MODES[2]: "layout_fingerprint",
        ATTEMPT_MODES[3]: "dtype",
    }
    for mode in ATTEMPT_MODES:
        rows = case_map[mode]
        checker.require(
            isinstance(rows, list) and len(rows) == 4,
            f"oracle rows mismatch: {mode}",
        )
        for rank, row in enumerate(rows):
            _verify_binding_row(row, rank, checker, f"oracle {mode}")
        differences = [
            (rank, name)
            for rank in range(4)
            for name in ("model_fingerprint", "layout_fingerprint", "dtype")
            if rows[rank][name] != baseline[rank][name]
        ]
        expected = changed[mode]
        checker.require(
            differences == ([] if expected is None else [(2, expected)]),
            f"oracle mismatch scope invalid: {mode}",
        )
    return case_map


def _verify_binding_row(row, rank, checker, context):
    checker.require(
        isinstance(row, dict)
        and set(row) == {
            "participant_id",
            "operation",
            "status",
            "model_fingerprint",
            "layout_fingerprint",
            "dtype",
            "detail",
        },
        f"binding fields mismatch: {context} rank={rank}",
    )
    checker.require(
        row.get("participant_id") == rank,
        f"participant mismatch: {context} rank={rank}",
    )
    checker.require(
        row.get("operation") == "bind_loaded_checkpoint_candidate",
        f"operation mismatch: {context} rank={rank}",
    )
    checker.require(
        row.get("status") == "bound",
        f"status mismatch: {context} rank={rank}",
    )
    checker.require(
        _is_sha(row.get("model_fingerprint")),
        f"model fingerprint mismatch: {context} rank={rank}",
    )
    checker.require(
        _is_sha(row.get("layout_fingerprint")),
        f"layout fingerprint mismatch: {context} rank={rank}",
    )
    checker.require(
        row.get("dtype") in {"float16", "bfloat16", "float32"},
        f"dtype mismatch: {context} rank={rank}",
    )
    checker.require(
        row.get("detail") == "",
        f"detail mismatch: {context} rank={rank}",
    )


def _verify_sources(
    record,
    manifest,
    inherited,
    source_root,
    methods,
    checker,
):
    hashes = record.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and len(hashes) == 57,
        "source count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "source ordering mismatch",
    )
    checker.require(
        set(hashes) - set(inherited) == {PREFLIGHT_SOURCE},
        "source closure addition mismatch",
    )
    checker.require(
        {name: hashes.get(name) for name in inherited} == inherited,
        "inherited source closure mismatch",
    )
    checker.require(
        manifest.get("local_file_sha256") == hashes,
        "manifest local hashes mismatch",
    )
    checker.require(
        manifest.get("remote_file_sha256") == hashes,
        "manifest remote hashes mismatch",
    )
    for name, expected in hashes.items():
        path = Path(source_root) / name
        checker.require(path.is_file(), f"missing source: {name}")
        checker.require(_is_sha(expected), f"invalid source hash: {name}")
        checker.require(
            _sha(path.read_bytes()) == expected,
            f"source hash mismatch: {name}",
        )
    tree = _tree_hash(hashes)
    checker.require(
        record.get("source_tree_sha256") == tree,
        "record source tree mismatch",
    )
    checker.require(
        manifest.get("source_tree_sha256") == tree,
        "manifest source tree mismatch",
    )
    for name, expected in FILE_HASHES.items():
        checker.require(
            hashes.get(name) == expected,
            f"frozen file hash mismatch: {name}",
        )
    checker.require(
        record.get("method_source_sha256") == methods,
        "record method hashes mismatch",
    )
    checker.require(
        manifest.get("method_source_sha256") == methods,
        "manifest method hashes mismatch",
    )
    return tree


def _verify_row(row, mode, oracle_rows, methods, checker):
    success = mode == ATTEMPT_MODES[0]
    changed = {
        ATTEMPT_MODES[0]: None,
        ATTEMPT_MODES[1]: "model_fingerprint",
        ATTEMPT_MODES[2]: "layout_fingerprint",
        ATTEMPT_MODES[3]: "dtype",
    }[mode]
    exact = {
        "schema_version": ROW_SCHEMA_VERSION,
        "status": "PASS",
        "mode": mode,
        "observed_user": "sitian",
        "tp4_artifact_sha256": TP4_SHA256,
        "oracle_artifact_sha256": ORACLE_SHA256,
        "oracle_provenance": "synthetic-construction-free-oracle",
        "oracle_claim_boundary": "not-real-checkpoint-binding",
        "oracle_tensor_payload": "absent",
        "method_source_sha256": methods,
        "shared_memory_capacity": SHARED_MEMORY_CAPACITY,
        "dispatch_count": 2,
        "binding_dispatch_count": 1,
        "write_count": 2,
        "write_payload_bytes": [199, 154],
        "ack_send_order": [3, 2, 1],
        "collector_return_order": [1, 2, 3],
        "ack_status_by_rank": {"1": "ok", "2": "ok", "3": "ok"},
        "collector_poisoned": False,
        "authorized_changed_field": changed,
        "completion_committed": success,
        "repeat_zero_binding_dispatch": success,
        "child_exitcodes": {"1": 0, "2": 0, "3": 0},
        "child_collected_by_rank": {"1": True, "2": True, "3": True},
        "segment_unlinked": True,
        "post_unlink_attach_failed": True,
    }
    for name, expected in exact.items():
        checker.require(
            row.get(name) == expected,
            f"row field mismatch: {mode} {name}",
        )
    for field in (
        "read_count_by_rank",
        "executor_count_by_rank",
        "event_set_count_by_rank",
        "event_wait_count_by_rank",
        "event_clear_count_by_rank",
    ):
        checker.require(
            row.get(field) == {"1": 2, "2": 2, "3": 2},
            f"row count mismatch: {mode} {field}",
        )
    checker.require(
        row.get("envelopes") == [
            {
                "command_id": 0,
                "method_name": (
                    "bind_published_qwen35_loaded_checkpoint_candidate"
                ),
                "args": [],
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
        row.get("oracle_rows") == oracle_rows,
        f"oracle_rows mismatch: {mode}",
    )
    for rank, binding in enumerate(row["oracle_rows"]):
        _verify_binding_row(binding, rank, checker, f"row {mode}")
    checker.require(
        isinstance(row.get("process_id"), int)
        and not isinstance(row["process_id"], bool)
        and row["process_id"] > 0,
        f"outer PID invalid: {mode}",
    )
    children = row.get("child_process_ids")
    checker.require(
        isinstance(children, dict) and tuple(children) == ("1", "2", "3"),
        f"child PID keys invalid: {mode}",
    )
    checker.require(
        len(set(children.values())) == 3
        and row["process_id"] not in set(children.values()),
        f"child PID uniqueness invalid: {mode}",
    )
    checker.require(
        all(
            isinstance(pid, int) and not isinstance(pid, bool) and pid > 0
            for pid in children.values()
        ),
        f"child PID invalid: {mode}",
    )
    checker.require(
        isinstance(row.get("shared_memory_name"), str)
        and row["shared_memory_name"] != "tinyvllm"
        and len(row["shared_memory_name"]) <= 30,
        f"shared-memory name invalid: {mode}",
    )
    if success:
        checker.require(
            row.get("binding_rows") == oracle_rows,
            "success binding rows mismatch",
        )
        checker.require(
            row.get("completion_configuration") == [
                MODEL_FINGERPRINT,
                LAYOUT_FINGERPRINT,
                "bfloat16",
                4.0,
            ],
            "success completion mismatch",
        )
        checker.require(row.get("error_detail") == "", "success error mismatch")
    else:
        checker.require(
            row.get("binding_rows") is None,
            f"mismatch binding rows invalid: {mode}",
        )
        checker.require(
            row.get("completion_configuration") is None,
            f"mismatch completion invalid: {mode}",
        )
        checker.require(
            row.get("error_detail")
            == f"RuntimeError: loaded checkpoint candidate binding mismatch: {changed}",
            f"mismatch error invalid: {mode}",
        )


def _static_safety(source_root, checker):
    source = (
        Path(source_root) / PREFLIGHT_SOURCE
    ).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=PREFLIGHT_SOURCE)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]

    def named(name):
        return sum(
            isinstance(node.func, ast.Name) and node.func.id == name
            for node in calls
        )

    def attr(name):
        return sum(
            isinstance(node.func, ast.Attribute) and node.func.attr == name
            for node in calls
        )

    imported_modules = {
        alias.name
        for node in imports
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from_modules = {
        node.module
        for node in imports
        if isinstance(node, ast.ImportFrom)
    }
    checker.require(
        "tinyvllm.engine.llm_engine" not in imported_modules
        and "tinyvllm.engine.llm_engine" not in imported_from_modules,
        "production Engine import detected",
    )
    checker.require(
        "tinyvllm.engine.model_runner" not in imported_modules
        and "tinyvllm.engine.model_runner" not in imported_from_modules,
        "production ModelRunner import detected",
    )
    checker.require(named("LLMEngine") + attr("LLMEngine") == 0, "Engine constructed")
    checker.require(named("ModelRunner") + attr("ModelRunner") == 0, "Runner constructed")
    for name in (
        "read_qwen35_checkpoint_metadata",
        "load_qwen35_fresh_checkpoint_candidate",
        "prepare_qwen35_checkpoint_candidate_target",
        "build_qwen35_authorized_checkpoint_candidate_loader",
        "Scheduler",
        "schedule",
        "step",
        "cuda",
        "forward",
        "inference",
    ):
        checker.require(
            named(name) + attr(name) == 0,
            f"forbidden call detected: {name}",
        )


def _verify(
    run_dir,
    *,
    source_root,
    tp4_artifact,
    oracle_artifact,
    checker,
):
    run_dir = Path(run_dir)
    artifact = run_dir / ARTIFACT_NAME
    manifest_path = run_dir / MANIFEST_NAME
    checker.require(artifact.is_file(), "missing result artifact")
    checker.require(manifest_path.is_file(), "missing source manifest")
    checker.require(Path(tp4_artifact).is_file(), "missing TP4 artifact")
    checker.require(Path(oracle_artifact).is_file(), "missing oracle artifact")
    checker.require(
        _sha(Path(tp4_artifact).read_bytes()) == TP4_SHA256,
        "TP4 artifact hash mismatch",
    )
    checker.require(
        _sha(Path(oracle_artifact).read_bytes()) == ORACLE_SHA256,
        "oracle artifact hash mismatch",
    )
    record = _read(artifact)
    manifest = _read(manifest_path)
    tp4 = _read(tp4_artifact)
    oracle = _read(oracle_artifact)
    inherited = _verify_tp4(tp4, checker)
    oracle_cases = _verify_oracle(oracle, checker)
    methods = _method_hashes(source_root, checker)
    exact = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "tp4_artifact_sha256": TP4_SHA256,
        "tp4_source_tree_sha256": TP4_SOURCE_TREE_SHA256,
        "oracle_artifact_sha256": ORACLE_SHA256,
        "oracle_provenance": "synthetic-construction-free-oracle",
        "oracle_claim_boundary": "not-real-checkpoint-binding",
        "oracle_tensor_payload": "absent",
        "method_source_sha256": methods,
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
        "tp4_artifact_sha256": TP4_SHA256,
        "oracle_artifact_sha256": ORACLE_SHA256,
        "method_source_sha256": methods,
    }.items():
        checker.require(
            manifest.get(name) == expected,
            f"manifest field mismatch: {name}",
        )
    tree = _verify_sources(
        record,
        manifest,
        inherited,
        source_root,
        methods,
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
        _verify_row(row, mode, oracle_cases[mode], methods, checker)
    outer = {row["process_id"] for row in rows}
    children = {
        pid for row in rows for pid in row["child_process_ids"].values()
    }
    names = {row["shared_memory_name"] for row in rows}
    checker.require(len(outer) == 4, "outer PID uniqueness mismatch")
    checker.require(len(children) == 12, "child PID uniqueness mismatch")
    checker.require(outer.isdisjoint(children), "outer/child PID overlap")
    checker.require(len(names) == 4, "shared-memory uniqueness mismatch")
    _static_safety(source_root, checker)
    return {
        "row_count": 4,
        "unique_process_count": 4,
        "unique_child_process_count": 12,
        "unique_shared_memory_count": 4,
        "source_file_count": 57,
        "source_tree_sha256": tree,
    }


def verify_run(
    run_dir,
    *,
    source_root,
    tp4_artifact,
    oracle_artifact,
):
    checker = Checker()
    try:
        summary = _verify(
            run_dir,
            source_root=source_root,
            tp4_artifact=tp4_artifact,
            oracle_artifact=oracle_artifact,
            checker=checker,
        )
    except (
        VerificationError,
        OSError,
        SyntaxError,
        KeyError,
        TypeError,
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
    parser.add_argument("--tp4-artifact", required=True)
    parser.add_argument("--oracle-artifact", required=True)
    args = parser.parse_args(argv)
    result = verify_run(
        args.run_dir,
        source_root=args.source_root,
        tp4_artifact=args.tp4_artifact,
        oracle_artifact=args.oracle_artifact,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
