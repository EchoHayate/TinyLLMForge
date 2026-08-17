from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ORACLE_NAME = "tp4_real_candidate_provenance_oracle.json"
RESULT_NAME = "tp4_real_candidate_provenance_replay_preflight.json"
MANIFEST_NAME = "source_manifest.json"
ORACLE_SCHEMA = "qwen35.tp4-real-candidate-provenance-oracle.v1"
RESULT_SCHEMA = "qwen35.tp4-real-candidate-provenance-replay.v1"
PRODUCER_SCHEMA = "qwen35.tp4-real-candidate-producer-rank.v1"
REPLAY_ROW_SCHEMA = (
    "qwen35.tp4-real-candidate-provenance-replay-rank.v1"
)
PROVENANCE = "real-checkpoint-derived-serial-rank-replay"
CLAIM_BOUNDARY = "not-live-concurrent-tp4-candidate-binding"
MODEL_MANIFEST = (
    "3e650a908234771c3cf1ac4e20c4d38fe69982efedaf4a3e631ad0b14aad7dd0"
)
CONFIG_SHA = (
    "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4"
)
INDEX_SHA = (
    "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9"
)
COMPOSITE_SHA = (
    "27da983f5ef3e38480d8b5d5976e5c434fc4b5d0c70d09511c35154beecd8db9"
)
AUTHORIZATION_SHA = (
    "10a39d6eb918cb5e8d1ccf52a723cdca4db4dffb9fd4ded62b1b766474d4fde4"
)
MODEL_DIR = (
    "/data00/home/sitian/sitian-workspace01/tllm/"
    "qwen35-hybrid-state-runs/qwen35-2b-hybrid-acquire-20260723-222004/model"
)
REMOTE_TARGET = "sitian@10.232.195.203"
REMOTE_PYTHON = (
    "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
)
SOURCE_TREE = (
    "42dddc0eac0a6db6041d5abb71df34db4d5e7c99d3b74d69f94598a2f24eb137"
)
PREFLIGHT_SOURCE = (
    "tools/qwen35_tp4_real_candidate_provenance_replay_preflight.py"
)
MODEL_RUNNER_SOURCE = "tinyvllm/engine/model_runner.py"
LLM_ENGINE_SOURCE = "tinyvllm/engine/llm_engine.py"
METHOD_SPECS = {
    "load_and_publish_qwen35_checkpoint_candidate": (
        MODEL_RUNNER_SOURCE,
        "ModelRunner",
        ("self", "request"),
        None,
        (),
        "9134c5bad8c4127714e07ffd8af56209c247a746e9f0d0ceceb60227c1358612",
    ),
    "bind_published_qwen35_loaded_checkpoint_candidate": (
        MODEL_RUNNER_SOURCE,
        "ModelRunner",
        ("self",),
        None,
        (),
        "aa178f886d314893593039c5e890239fb954740f059b2d12fc697bd25790fbcd",
    ),
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
REPLAY_MODES = (
    "tp4_real_replay_success",
    "tp4_real_replay_rank2_model_mismatch",
    "tp4_real_replay_rank2_layout_mismatch",
    "tp4_real_replay_rank2_dtype_mismatch",
)
MEMORY_CEILINGS = {
    "total": 6291456,
    "post_torch": 6029312,
    "post_metadata": 5767168,
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
        raise VerificationError(f"invalid JSON {Path(path).name}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"{Path(path).name} must be an object")
    return value


def _is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _tree_hash(hashes):
    return _sha(_canonical(dict(hashes)))


def _method_hashes(source_root, checker):
    root = Path(source_root)
    sources = {}
    observed = {}
    for name, (
        filename,
        class_name,
        positional,
        vararg,
        keyword_only,
        expected,
    ) in METHOD_SPECS.items():
        source = sources.setdefault(
            filename,
            (root / filename).read_text(encoding="utf-8"),
        )
        tree = ast.parse(source, filename=filename)
        classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        checker.require(len(classes) == 1, f"class mismatch: {name}")
        methods = [
            node for node in classes[0].body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        checker.require(len(methods) == 1, f"method mismatch: {name}")
        node = methods[0]
        checker.require(
            tuple(
                argument.arg
                for argument in (*node.args.posonlyargs, *node.args.args)
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
            tuple(argument.arg for argument in node.args.kwonlyargs)
            == keyword_only,
            f"method keyword mismatch: {name}",
        )
        segment = ast.get_source_segment(source, node)
        checker.require(isinstance(segment, str), f"method source: {name}")
        observed[name] = _sha(segment.encode("utf-8"))
        checker.require(
            observed[name] == expected,
            f"method hash mismatch: {name}",
        )
    return observed


def _verify_binding_row(row, rank, model, layout, dtype, checker, context):
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
        f"binding fields mismatch: {context}",
    )
    checker.require(
        row.get("participant_id") == rank,
        f"participant mismatch: {context}",
    )
    checker.require(
        row.get("operation") == "bind_loaded_checkpoint_candidate",
        f"operation mismatch: {context}",
    )
    checker.require(row.get("status") == "bound", f"status: {context}")
    checker.require(
        row.get("model_fingerprint") == model,
        f"model mismatch: {context}",
    )
    checker.require(
        row.get("layout_fingerprint") == layout,
        f"layout mismatch: {context}",
    )
    checker.require(row.get("dtype") == dtype, f"dtype mismatch: {context}")
    checker.require(row.get("detail") == "", f"detail mismatch: {context}")


def _verify_sources(oracle, result, manifest, source_root, checker):
    hashes = oracle.get("source_file_sha256")
    checker.require(
        isinstance(hashes, dict) and len(hashes) == 58,
        "source file count mismatch",
    )
    checker.require(
        tuple(hashes) == tuple(sorted(hashes)),
        "source ordering mismatch",
    )
    checker.require(
        result.get("source_file_sha256") == hashes,
        "result source hashes mismatch",
    )
    checker.require(
        manifest.get("local_file_sha256") == hashes,
        "manifest source hashes mismatch",
    )
    root = Path(source_root)
    for name, expected in hashes.items():
        path = root / name
        checker.require(path.is_file(), f"missing source: {name}")
        checker.require(_is_sha(expected), f"invalid source SHA: {name}")
        checker.require(
            _sha(path.read_bytes()) == expected,
            f"source hash mismatch: {name}",
        )
    tree = _tree_hash(hashes)
    checker.require(tree == SOURCE_TREE, "frozen source tree mismatch")
    checker.require(
        oracle.get("source_tree_sha256") == tree,
        "oracle source tree mismatch",
    )
    checker.require(
        result.get("source_tree_sha256") == tree,
        "result source tree mismatch",
    )
    checker.require(
        manifest.get("source_tree_sha256") == tree,
        "manifest source tree mismatch",
    )
    return hashes


def _verify_producer(row, rank, methods, checker):
    exact = {
        "schema_version": PRODUCER_SCHEMA,
        "status": "PASS",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "tp_size": 4,
        "tp_rank": rank,
        "observed_user": "sitian",
        "checkpoint_dir": MODEL_DIR,
        "model_manifest_sha256": MODEL_MANIFEST,
        "config_sha256": CONFIG_SHA,
        "index_sha256": INDEX_SHA,
        "config_index_header_sha256": COMPOSITE_SHA,
        "authorization_sha256": AUTHORIZATION_SHA,
        "method_source_sha256": methods,
        "metadata_bytes_read": 144024,
        "adapter_call_count": 1,
        "provider_call_count": 1,
        "load_publish_method_call_count": 1,
        "bind_method_call_count": 1,
        "owner_binding_method_call_count": 1,
        "candidate_binding_method_call_count": 1,
        "production_publish_call_count": 1,
        "selected_binding_count": 320,
        "unique_destination_count": 296,
        "binding_hash_count": 320,
        "phase_hash_count": 26,
        "aggregate_hash_verified": True,
        "loader_stats": {
            "assigned_bindings": 320,
            "source_tensors": 320,
            "shard_count": 1,
            "loaded_bytes": 3763655360,
            "peak_source_bytes": 1017118720,
        },
        "dtype": "bfloat16",
        "all_selected_destinations_zero_after_clear": True,
        "non_selected_tensors_unchanged": True,
        "tensor_identity_preserved": True,
        "pool_unchanged": True,
        "all_private_objects_collected": True,
        "cuda_initialized_before": False,
        "cuda_initialized_after": False,
        "model_forward_count": 0,
        "attention_forward_count": 0,
    }
    for name, expected in exact.items():
        checker.require(
            row.get(name) == expected,
            f"producer field mismatch: rank={rank} {name}",
        )
    checker.require(
        isinstance(row.get("process_id"), int)
        and not isinstance(row["process_id"], bool)
        and row["process_id"] > 0,
        f"producer process invalid: rank={rank}",
    )
    layout = row.get("layout_fingerprint")
    checker.require(_is_sha(layout), f"producer layout invalid: rank={rank}")
    _verify_binding_row(
        row.get("method_row"),
        rank,
        MODEL_MANIFEST,
        layout,
        "bfloat16",
        checker,
        f"producer rank={rank}",
    )
    hashes = row.get("binding_destination_sha256")
    checker.require(
        isinstance(hashes, list) and len(hashes) == 320,
        f"binding hash count: rank={rank}",
    )
    for index, value in enumerate(hashes):
        checker.require(
            _is_sha(value),
            f"binding hash invalid: rank={rank} index={index}",
        )
    phases = row.get("phase_destination_sha256")
    checker.require(
        isinstance(phases, dict) and len(phases) == 26,
        f"phase count: rank={rank}",
    )
    for name, value in phases.items():
        checker.require(
            isinstance(name, str) and bool(name) and _is_sha(value),
            f"phase hash invalid: rank={rank}",
        )
    checker.require(
        _is_sha(row.get("aggregate_destination_sha256")),
        f"aggregate invalid: rank={rank}",
    )
    collected = row.get("collected_private_objects")
    checker.require(
        isinstance(collected, dict)
        and len(collected) == 10
        and all(value is True for value in collected.values()),
        f"collection invalid: rank={rank}",
    )
    memory = row.get("memory")
    checker.require(
        isinstance(memory, dict)
        and set(memory) == {
            "before",
            "after_torch",
            "after_metadata",
            "after_pool",
            "after_target",
            "after_clear",
        },
        f"memory points invalid: rank={rank}",
    )
    for name, point in memory.items():
        checker.require(
            isinstance(point, dict)
            and set(point) == {"vmhwm_kib", "vmrss_kib"}
            and all(
                isinstance(value, int) and value >= 0
                for value in point.values()
            ),
            f"memory point invalid: rank={rank} {name}",
        )
    checker.require(
        row.get("total_vmhwm_increment_kib") <= MEMORY_CEILINGS["total"],
        f"total memory ceiling: rank={rank}",
    )
    checker.require(
        row.get("post_torch_vmhwm_increment_kib")
        <= MEMORY_CEILINGS["post_torch"],
        f"post torch memory ceiling: rank={rank}",
    )
    checker.require(
        row.get("post_metadata_vmhwm_increment_kib")
        <= MEMORY_CEILINGS["post_metadata"],
        f"post metadata memory ceiling: rank={rank}",
    )


def _verify_replay(row, mode, oracle_rows, producer_hash, oracle_hash, checker):
    success = mode == REPLAY_MODES[0]
    changed = {
        REPLAY_MODES[0]: None,
        REPLAY_MODES[1]: "model_fingerprint",
        REPLAY_MODES[2]: "layout_fingerprint",
        REPLAY_MODES[3]: "dtype",
    }[mode]
    exact = {
        "schema_version": REPLAY_ROW_SCHEMA,
        "status": "PASS",
        "mode": mode,
        "observed_user": "sitian",
        "provenance": PROVENANCE,
        "claim_boundary": CLAIM_BOUNDARY,
        "producer_rows_sha256": producer_hash,
        "provenance_oracle_sha256": oracle_hash,
        "shared_memory_capacity": 1048576,
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
            f"replay field mismatch: {mode} {name}",
        )
    for name in (
        "read_count_by_rank",
        "executor_count_by_rank",
        "event_set_count_by_rank",
        "event_wait_count_by_rank",
        "event_clear_count_by_rank",
    ):
        checker.require(
            row.get(name) == {"1": 2, "2": 2, "3": 2},
            f"replay count mismatch: {mode} {name}",
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
        f"replay envelopes mismatch: {mode}",
    )
    rows = row.get("oracle_rows")
    checker.require(
        isinstance(rows, list) and len(rows) == 4,
        f"oracle rows invalid: {mode}",
    )
    for rank, observed in enumerate(rows):
        expected = oracle_rows[rank]
        _verify_binding_row(
            observed,
            rank,
            expected["model_fingerprint"],
            expected["layout_fingerprint"],
            expected["dtype"],
            checker,
            f"replay {mode} rank={rank}",
        )
    differences = [
        (rank, name)
        for rank in range(4)
        for name in ("model_fingerprint", "layout_fingerprint", "dtype")
        if rows[rank][name] != oracle_rows[rank][name]
    ]
    checker.require(
        differences == [],
        f"oracle rows mutated unexpectedly: {mode}",
    )
    checker.require(
        isinstance(row.get("process_id"), int) and row["process_id"] > 0,
        f"outer process invalid: {mode}",
    )
    children = row.get("child_process_ids")
    checker.require(
        isinstance(children, dict)
        and tuple(children) == ("1", "2", "3")
        and len(set(children.values())) == 3,
        f"child processes invalid: {mode}",
    )
    checker.require(
        isinstance(row.get("shared_memory_name"), str)
        and row["shared_memory_name"] != "tinyvllm",
        f"shared memory name invalid: {mode}",
    )
    if success:
        checker.require(
            row.get("binding_rows") == rows,
            "success binding rows mismatch",
        )
        checker.require(
            row.get("completion_configuration")
            == [
                MODEL_MANIFEST,
                rows[0]["layout_fingerprint"],
                "bfloat16",
                5.0,
            ],
            "success completion mismatch",
        )
        checker.require(row.get("error_detail") == "", "success error")
    else:
        checker.require(
            row.get("binding_rows") is None,
            f"mismatch binding rows: {mode}",
        )
        checker.require(
            row.get("completion_configuration") is None,
            f"mismatch completion: {mode}",
        )
        checker.require(
            row.get("error_detail")
            == (
                "RuntimeError: loaded checkpoint candidate binding "
                f"mismatch: {changed}"
            ),
            f"mismatch error detail: {mode}",
        )


def _static_safety(source_root, checker):
    source = (
        Path(source_root) / PREFLIGHT_SOURCE
    ).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=PREFLIGHT_SOURCE)
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    modules = {
        alias.name
        for node in imports
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in imports
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
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
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
    worker_source = (
        Path(source_root) / "tools/qwen35_real_checkpoint_load_worker.py"
    ).read_text(encoding="utf-8")
    worker_tree = ast.parse(
        worker_source,
        filename="tools/qwen35_real_checkpoint_load_worker.py",
    )
    hard_rejection = (
        "real checkpoint load worker execution is not implemented; "
        "only the local safety dry-run is authorized"
    )
    hard_rejections = [
        node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.Call)
        if isinstance(node.func, ast.Name)
        and node.func.id == "RuntimeError"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == hard_rejection
    ]
    checker.require(
        len(hard_rejections) == 1,
        "real worker hard rejection changed",
    )


def _verify(run_dir, source_root, checker):
    run_dir = Path(run_dir)
    inventory = sorted(path.name for path in run_dir.iterdir())
    local_inventory = [MANIFEST_NAME, ORACLE_NAME, RESULT_NAME]
    remote_inventory = sorted([
        "model_runner_load_and_publish_preflight.json",
        "model_runner_published_candidate_binding_preflight.json",
        "producer_rank0.json",
        "producer_rank1.json",
        "producer_rank2.json",
        "producer_rank3.json",
        "source",
        MANIFEST_NAME,
        ORACLE_NAME,
        RESULT_NAME,
        "tp4_synthetic_binding_oracle_preflight.json",
    ])
    checker.require(
        inventory in (local_inventory, remote_inventory),
        "run inventory mismatch",
    )
    oracle = _read(run_dir / ORACLE_NAME)
    result = _read(run_dir / RESULT_NAME)
    manifest = _read(run_dir / MANIFEST_NAME)
    methods = _method_hashes(source_root, checker)
    for record, label, schema in (
        (oracle, "oracle", ORACLE_SCHEMA),
        (result, "result", RESULT_SCHEMA),
    ):
        checker.require(
            record.get("schema_version") == schema,
            f"{label} schema mismatch",
        )
        checker.require(record.get("status") == "PASS", f"{label} status")
        checker.require(
            record.get("provenance") == PROVENANCE,
            f"{label} provenance",
        )
        checker.require(
            record.get("claim_boundary") == CLAIM_BOUNDARY,
            f"{label} claim boundary",
        )
    exact_oracle = {
        "checkpoint_dir": MODEL_DIR,
        "model_manifest_sha256": MODEL_MANIFEST,
        "config_sha256": CONFIG_SHA,
        "index_sha256": INDEX_SHA,
        "config_index_header_sha256": COMPOSITE_SHA,
        "authorization_sha256": AUTHORIZATION_SHA,
        "method_source_sha256": methods,
        "producer_contexts": [[4, 0], [4, 1], [4, 2], [4, 3]],
        "all_producers_exited_before_finalization": True,
    }
    for name, expected in exact_oracle.items():
        checker.require(
            oracle.get(name) == expected,
            f"oracle field mismatch: {name}",
        )
    _static_safety(source_root, checker)
    hashes = _verify_sources(
        oracle,
        result,
        manifest,
        source_root,
        checker,
    )
    rows = oracle.get("producer_rows")
    checker.require(
        isinstance(rows, list) and len(rows) == 4,
        "producer row count mismatch",
    )
    for rank, row in enumerate(rows):
        _verify_producer(row, rank, methods, checker)
    if inventory == remote_inventory:
        checker.require(
            (run_dir / "source").is_dir(),
            "remote source directory missing",
        )
        for rank, row in enumerate(rows):
            checker.require(
                _read(run_dir / f"producer_rank{rank}.json") == row,
                f"remote producer row mismatch: rank={rank}",
            )
    producer_ids = [row["process_id"] for row in rows]
    checker.require(
        oracle.get("producer_process_ids") == producer_ids
        and len(set(producer_ids)) == 4,
        "producer process identities invalid",
    )
    producer_hash = _sha(_canonical(rows))
    checker.require(
        oracle.get("producer_rows_sha256") == producer_hash,
        "producer rows hash mismatch",
    )
    oracle_hash = _sha(_canonical(oracle))
    checker.require(
        result.get("provenance_oracle_sha256") == oracle_hash,
        "oracle canonical hash mismatch",
    )
    checker.require(
        result.get("producer_rows_sha256") == producer_hash,
        "result producer hash mismatch",
    )
    checker.require(
        result.get("producer_process_ids") == producer_ids,
        "result producer IDs mismatch",
    )
    replay_rows = result.get("replay_rows")
    checker.require(
        isinstance(replay_rows, list)
        and tuple(row.get("mode") for row in replay_rows) == REPLAY_MODES,
        "replay rows mismatch",
    )
    baseline = [row["method_row"] for row in rows]
    oracle_rows_by_mode = []
    for mode, replay_row in zip(REPLAY_MODES, replay_rows):
        changed = {
            REPLAY_MODES[0]: None,
            REPLAY_MODES[1]: "model_fingerprint",
            REPLAY_MODES[2]: "layout_fingerprint",
            REPLAY_MODES[3]: "dtype",
        }[mode]
        mode_rows = replay_row.get("oracle_rows")
        checker.require(
            isinstance(mode_rows, list) and len(mode_rows) == 4,
            f"replay oracle row count mismatch: {mode}",
        )
        differences = [
            (rank, name)
            for rank in range(4)
            for name in ("model_fingerprint", "layout_fingerprint", "dtype")
            if mode_rows[rank][name] != baseline[rank][name]
        ]
        checker.require(
            differences == (
                [] if changed is None else [(2, changed)]
            ),
            f"replay oracle mismatch scope invalid: {mode}",
        )
        if changed == "dtype":
            checker.require(
                mode_rows[2]["dtype"] == "float32",
                "rank2 dtype mismatch value invalid",
            )
        oracle_rows_by_mode.append(mode_rows)
    for row, mode, mode_rows in zip(
        replay_rows,
        REPLAY_MODES,
        oracle_rows_by_mode,
    ):
        _verify_replay(
            row,
            mode,
            mode_rows,
            producer_hash,
            oracle_hash,
            checker,
        )
    outer = [row["process_id"] for row in replay_rows]
    children = [
        process_id
        for row in replay_rows
        for process_id in row["child_process_ids"].values()
    ]
    all_ids = producer_ids + outer + children
    checker.require(
        result.get("replay_outer_process_ids") == outer,
        "replay outer IDs mismatch",
    )
    checker.require(
        result.get("replay_child_process_ids") == children,
        "replay child IDs mismatch",
    )
    checker.require(
        len(set(all_ids)) == 20,
        "process identity overlap",
    )
    checker.require(
        result.get("all_replay_processes_distinct_from_producers") is True,
        "process separation flag mismatch",
    )
    manifest_exact = {
        "schema_version": RESULT_SCHEMA,
        "run_tag": run_dir.name,
        "remote_target": REMOTE_TARGET,
        "remote_python": REMOTE_PYTHON,
        "source_tree_sha256": SOURCE_TREE,
    }
    for name, expected in manifest_exact.items():
        checker.require(
            manifest.get(name) == expected,
            f"manifest field mismatch: {name}",
        )
    for name in (
        "load_publish_artifact_sha256",
        "published_binding_artifact_sha256",
        "tp4_replay_artifact_sha256",
    ):
        checker.require(
            _is_sha(manifest.get(name)),
            f"manifest prerequisite hash invalid: {name}",
        )
    return {
        "producer_count": 4,
        "replay_count": 4,
        "source_file_count": len(hashes),
        "unique_process_count": len(set(all_ids)),
        "source_tree_sha256": SOURCE_TREE,
        "oracle_sha256": _sha((run_dir / ORACLE_NAME).read_bytes()),
        "result_sha256": _sha((run_dir / RESULT_NAME).read_bytes()),
        "manifest_sha256": _sha((run_dir / MANIFEST_NAME).read_bytes()),
    }


def verify_run(run_dir, *, source_root):
    checker = Checker()
    try:
        summary = _verify(run_dir, source_root, checker)
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
    args = parser.parse_args(argv)
    result = verify_run(args.run_dir, source_root=args.source_root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
