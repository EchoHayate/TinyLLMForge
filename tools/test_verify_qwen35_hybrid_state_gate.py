"""Dependency-light tests for the Qwen3.5 hybrid-state independent verifier.

Run: python3 tools/test_verify_qwen35_hybrid_state_gate.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CONTRACT_PATH = THIS_DIR / "qwen35_hybrid_state_contract.py"
VERIFIER_PATH = THIS_DIR / "verify_qwen35_hybrid_state_gate.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, os.fspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_module(
    "qwen35_hybrid_state_contract_for_verifier_tests",
    CONTRACT_PATH,
)
verifier = _load_module(
    "qwen35_hybrid_state_gate_verifier_under_test",
    VERIFIER_PATH,
)


def _json_bytes(payload):
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode()


def _write_json(path, payload):
    path.write_bytes(_json_bytes(payload))


def _write_jsonl(path, rows):
    path.write_bytes(b"".join(_json_bytes(row) for row in rows))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_entry(path):
    return {
        "path": path.name,
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _refresh_manifest_artifact(run_dir, relative_path):
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target = next(
        entry
        for entry in manifest["artifacts"]
        if entry["path"] == relative_path
    )
    artifact_path = run_dir / relative_path
    target["size"] = artifact_path.stat().st_size
    target["sha256"] = _sha256(artifact_path)
    _write_json(manifest_path, manifest)


def _mutate_json(run_dir, relative_path, mutator):
    path = run_dir / relative_path
    payload = json.loads(path.read_text())
    mutator(payload)
    _write_json(path, payload)
    _refresh_manifest_artifact(run_dir, relative_path)


def _mutate_jsonl(run_dir, relative_path, mutator):
    path = run_dir / relative_path
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutator(rows)
    _write_jsonl(path, rows)
    _refresh_manifest_artifact(run_dir, relative_path)


def _remove_case_row(run_dir):
    _mutate_jsonl(
        run_dir,
        "case_rows.jsonl",
        lambda rows: rows.pop(),
    )


def _duplicate_case_row(run_dir):
    _mutate_jsonl(
        run_dir,
        "case_rows.jsonl",
        lambda rows: rows.append(dict(rows[0])),
    )


def _add_unknown_case(run_dir):
    def mutate(rows):
        row = dict(rows[0])
        row["row_id"] = "row:unknown"
        row["case_id"] = "unknown"
        rows.append(row)

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _alter_source_hash(run_dir):
    def mutate(payload):
        name = sorted(payload["remote_file_sha256"])[0]
        payload["remote_file_sha256"][name] = "f" * 64

    _mutate_json(run_dir, "source_manifest.json", mutate)


def _alter_model_revision(run_dir):
    _mutate_json(
        run_dir,
        "model_manifest.json",
        lambda payload: payload.__setitem__(
            "resolved_revision",
            "c" * 40,
        ),
    )


def _reuse_port(run_dir):
    def mutate_processes(payload):
        row = payload["processes"][0]
        row["master_port"] = row["tinyvllm_dist_port"]

    def mutate_ports(payload):
        row = payload["pairs"][0]
        row["master_port"] = row["tinyvllm_dist_port"]

    _mutate_json(run_dir, "processes.json", mutate_processes)
    _mutate_json(run_dir, "ports.json", mutate_ports)


def _add_unlisted_input(run_dir):
    (run_dir / "unlisted.json").write_text("{}\n")


def _expect_incomplete(base_run, mutator, message):
    run_dir = base_run.parent / mutator.__name__
    shutil.copytree(base_run, run_dir)
    mutator(run_dir)
    result = verifier.verify_run(run_dir)
    assert result["classification"] == "INCOMPLETE"
    assert any(message in reason for reason in result["reasons"])


def _expect_classification(
    base_run,
    mutator,
    classification,
    message,
):
    run_dir = base_run.parent / mutator.__name__
    shutil.copytree(base_run, run_dir)
    mutator(run_dir)
    result = verifier.verify_run(run_dir)
    assert result["classification"] == classification
    assert any(message in reason for reason in result["reasons"])


def _find_case(rows, phase, *, prompt_length=None):
    return next(
        row
        for row in rows
        if row["phase"] == phase
        and (
            prompt_length is None
            or row["prompt_length"] == prompt_length
        )
    )


def _change_cached_token(run_dir):
    def mutate(rows):
        row = _find_case(rows, "one_shot_vs_cached", prompt_length=17)
        row["decoded_token_ids"][0] += 1

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _remove_cached_step(run_dir):
    def mutate(rows):
        row = _find_case(rows, "one_shot_vs_cached", prompt_length=17)
        row["logit_records"].pop()

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _change_cached_full_logit_hash(run_dir):
    def mutate(rows):
        row = _find_case(rows, "one_shot_vs_cached", prompt_length=17)
        row["logit_records"][0]["full_logit_sha256"] = "f" * 64

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _change_cached_oracle_token(run_dir):
    def mutate(rows):
        row = _find_case(rows, "one_shot_vs_cached", prompt_length=17)
        row["logit_records"][0]["position_metadata"][
            "oracle_greedy_token_id"
        ] += 1

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _change_cached_oracle_hash(run_dir):
    def mutate(rows):
        row = _find_case(rows, "one_shot_vs_cached", prompt_length=17)
        row["logit_records"][0]["position_metadata"][
            "oracle_full_logit_sha256"
        ] = "f" * 64

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _raise_repeatability_above_cap(run_dir):
    def mutate(rows):
        row = _find_case(
            rows,
            "same_path_repeatability",
            prompt_length=17,
        )
        row["logit_records"][0]["max_abs_diff"] = (
            contract.MAX_LOGIT_ATOL
        )

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _change_chunk_schedule(run_dir):
    def mutate(rows):
        row = _find_case(rows, "one_shot_vs_chunked")
        row["chunk_schedule"] = [row["prompt_length"]]

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _change_export_token(run_dir):
    def mutate(rows):
        row = _find_case(rows, "state_export_import", prompt_length=17)
        row["decoded_token_ids"][0] += 1
        row["logit_records"][0]["position_metadata"][
            "actual_greedy_token_id"
        ] += 1
        row["logit_records"][0]["position_metadata"][
            "oracle_greedy_token_id"
        ] += 1

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _make_topk_non_finite(run_dir):
    def mutate(rows):
        row = _find_case(rows, "one_shot_vs_cached", prompt_length=17)
        row["logit_records"][0]["topk_logits"][0] = float("nan")

    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate)


def _mutate_component(run_dir, predicate, mutator):
    def mutate(rows):
        row = next(item for item in rows if predicate(item))
        mutator(row)

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate)


def _refresh_state_snapshot_derivatives(run_dir):
    component_path = run_dir / "state_components.jsonl"
    snapshot_path = run_dir / "state_snapshots.jsonl"
    components = [
        json.loads(line)
        for line in component_path.read_text().splitlines()
    ]
    snapshots = [
        json.loads(line)
        for line in snapshot_path.read_text().splitlines()
    ]
    by_epoch = {}
    for component in components:
        by_epoch.setdefault(component["lifetime_epoch"], []).append(component)
    for snapshot in snapshots:
        rows = by_epoch.get(snapshot["lifetime_epoch"], [])
        snapshot["component_count"] = len(rows)
        snapshot["component_sha256"] = contract.canonical_json_sha256(rows)
    _write_jsonl(snapshot_path, snapshots)
    _refresh_manifest_artifact(run_dir, "state_snapshots.jsonl")


def _set_unexplained_role(run_dir):
    _mutate_component(
        run_dir,
        lambda row: row["state_role"] == "linear_recurrent_state",
        lambda row: row.__setitem__(
            "state_role",
            "other_persistent_state",
        ),
    )


def _grow_recurrent_state(run_dir):
    def mutate(row):
        row["shape"][2] += 1
        row["logical_numel"] = (
            row["shape"][0]
            * row["shape"][1]
            * row["shape"][2]
            * row["shape"][3]
        )
        row["logical_bytes"] = row["logical_numel"] * 4
        row["storage_nbytes"] = row["logical_bytes"]

    _mutate_component(
        run_dir,
        lambda row: (
            row["state_role"] == "linear_recurrent_state"
            and row["update_kind"] == "mutated_in_place"
        ),
        mutate,
    )


def _stop_full_kv_growth(run_dir):
    def mutate(rows):
        target = next(
            row
            for row in rows
            if row["state_role"] == "full_attention_key"
            and row["update_kind"] == "grown"
        )
        target["shape"][2] = 1
        target["logical_numel"] = 1 * 16 * 1 * 128
        target["logical_bytes"] = target["logical_numel"] * 4
        target["storage_nbytes"] = target["logical_bytes"]

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate)


def _cross_request_component_mutation(run_dir):
    def mutate(rows):
        target = next(
            row
            for row in rows
            if row["request_id"] == "slot-1"
            and row["state_role"] == "linear_recurrent_state"
        )
        target["request_id"] = "slot-0"

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate)


def _remove_release_component(run_dir):
    def mutate(rows):
        rows[:] = [
            row for row in rows if row["update_kind"] != "released"
        ]

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate)
    _refresh_state_snapshot_derivatives(run_dir)


def _reuse_without_generation_increment(run_dir):
    def mutate_components(rows):
        for row in rows:
            if (
                row["request_id"] == "slot-0"
                and row["request_generation"] == 1
            ):
                row["request_generation"] = 0

    def mutate_snapshots(rows):
        for row in rows:
            if (
                row["request_id"] == "slot-0"
                and row["request_generation"] == 1
            ):
                row["request_generation"] = 0

    def mutate_cases(rows):
        row = _find_case(rows, "completion_release_slot_reuse")
        row["request_generations"][-1] = 0

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate_components)
    _mutate_jsonl(run_dir, "state_snapshots.jsonl", mutate_snapshots)
    _mutate_jsonl(run_dir, "case_rows.jsonl", mutate_cases)
    _refresh_state_snapshot_derivatives(run_dir)


def _reuse_stale_storage_identity(run_dir):
    def mutate(rows):
        released = next(
            row for row in rows if row["update_kind"] == "released"
        )
        replacement = next(
            row
            for row in rows
            if row["request_id"] == "slot-0"
            and row["request_generation"] == 1
        )
        replacement["storage_identity"] = released["storage_identity"]
        replacement["storage_nbytes"] = released["storage_nbytes"]

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate)
    _refresh_state_snapshot_derivatives(run_dir)


def _reuse_stale_content(run_dir):
    def mutate(rows):
        released = next(
            row for row in rows if row["update_kind"] == "released"
        )
        replacement = next(
            row
            for row in rows
            if row["request_id"] == "slot-0"
            and row["request_generation"] == 1
        )
        replacement["content_sha256"] = released["content_sha256"]

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate)
    _refresh_state_snapshot_derivatives(run_dir)


def _make_layer_identity_ambiguous(run_dir):
    _mutate_component(
        run_dir,
        lambda row: row["layer_index"] >= 0,
        lambda row: row.__setitem__("layer_index", -1),
    )


def _set_unsupported_update_kind(run_dir):
    _mutate_component(
        run_dir,
        lambda row: True,
        lambda row: row.__setitem__("update_kind", "teleported"),
    )


def _set_incorrect_logical_bytes(run_dir):
    def mutate(row):
        row["logical_bytes"] += 4

    _mutate_component(run_dir, lambda row: True, mutate)
    _refresh_state_snapshot_derivatives(run_dir)


def _set_conflicting_alias_storage_size(run_dir):
    def mutate(rows):
        source = rows[0]
        alias = dict(source)
        alias["tensor_path"] = source["tensor_path"] + ".alias"
        alias["storage_nbytes"] = source["storage_nbytes"] + 4
        rows.append(alias)

    _mutate_jsonl(run_dir, "state_components.jsonl", mutate)
    _refresh_state_snapshot_derivatives(run_dir)


def _double_count_storage(run_dir):
    def mutate(rows):
        target = next(
            row for row in rows if row["unique_storage_bytes"] > 0
        )
        target["unique_storage_bytes"] *= 2

    _mutate_jsonl(run_dir, "memory_snapshots.jsonl", mutate)


def _remove_memory_epoch(run_dir):
    def mutate(rows):
        rows.pop()

    _mutate_jsonl(run_dir, "memory_snapshots.jsonl", mutate)


def _set_negative_allocator_bytes(run_dir):
    def mutate(rows):
        rows[0]["cuda_allocated_bytes"] = -1

    _mutate_jsonl(run_dir, "memory_snapshots.jsonl", mutate)


def _set_parameter_byte_mismatch(run_dir):
    _mutate_json(
        run_dir,
        "summary.json",
        lambda payload: payload.__setitem__(
            "parameter_bytes",
            payload["parameter_bytes"] + 2,
        ),
    )


def _set_worker_aggregate_disagreement(run_dir):
    _mutate_json(
        run_dir,
        "summary.json",
        lambda payload: payload.__setitem__(
            "state_logical_bytes",
            payload["state_logical_bytes"] + 4,
        ),
    )


def _component(
    *,
    request_id,
    request_generation,
    layer_index,
    layer_type,
    state_role,
    tensor_path,
    lifetime_epoch,
    sequence_length,
    storage_identity,
    update_kind,
):
    if state_role in {"full_attention_key", "full_attention_value"}:
        shape = [1, 16, sequence_length, 128]
    elif state_role == "linear_convolution_state":
        shape = [1, 16, 4, 128]
    else:
        shape = [1, 16, 128, 128]
    logical_numel = 1
    for dimension in shape:
        logical_numel *= dimension
    logical_bytes = logical_numel * 4
    return {
        "request_id": request_id,
        "request_generation": request_generation,
        "layer_index": layer_index,
        "declared_layer_type": layer_type,
        "state_role": state_role,
        "tensor_path": tensor_path,
        "shape": shape,
        "stride": [logical_numel, 1],
        "dtype": "float32",
        "device": "cuda:0",
        "requires_grad": False,
        "logical_numel": logical_numel,
        "logical_bytes": logical_bytes,
        "storage_data_ptr": lifetime_epoch * 4096 + layer_index,
        "storage_offset": 0,
        "storage_nbytes": logical_bytes,
        "storage_identity": storage_identity,
        "lifetime_epoch": lifetime_epoch,
        "sequence_length": sequence_length,
        "update_kind": update_kind,
        "content_sha256": hashlib.sha256(
            f"{storage_identity}:{sequence_length}".encode()
        ).hexdigest(),
    }


def _logit_record(
    *,
    request_id,
    request_generation,
    step_index,
    sequence_length,
):
    full_logit_sha256 = hashlib.sha256(
        f"{request_id}:{request_generation}:{step_index}".encode()
    ).hexdigest()
    return {
        "request_id": request_id,
        "request_generation": request_generation,
        "step_index": step_index,
        "full_logit_sha256": full_logit_sha256,
        "topk_token_ids": list(range(20)),
        "topk_logits": [float(20 - index) for index in range(20)],
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "mean_rel_diff": 0.0,
        "sequence_length": sequence_length,
        "position_metadata": {
            "cache_position": sequence_length,
            "actual_greedy_token_id": 0,
            "oracle_greedy_token_id": 0,
            "actual_full_logit_sha256": full_logit_sha256,
            "oracle_full_logit_sha256": full_logit_sha256,
        },
    }


def write_complete_run(run_dir):
    run_dir.mkdir()
    (run_dir / "stdout").mkdir()
    (run_dir / "stderr").mkdir()
    source_commit = "a" * 40
    model_revision = "b" * 40
    source_hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "tools/qwen35_hybrid_state_contract.py",
            "tools/qwen35_hybrid_state_probe.py",
            "tools/verify_qwen35_hybrid_state_gate.py",
            "tools/run_qwen35_hybrid_state_gate_remote.py",
        )
    }
    source_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "branch": "feat/adaptive-ngram-speculation",
        "commit": source_commit,
        "clean": True,
        "local_file_sha256": source_hashes,
        "remote_file_sha256": source_hashes,
    }
    model_files = {
        "config.json": {
            "size": 128,
            "sha256": hashlib.sha256(b"config").hexdigest(),
        },
        "tokenizer.json": {
            "size": 256,
            "sha256": hashlib.sha256(b"tokenizer").hexdigest(),
        },
        "model.safetensors": {
            "size": 1024,
            "sha256": hashlib.sha256(b"weights").hexdigest(),
        },
    }
    model_manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "repository": contract.MODEL_REPOSITORY,
        "resolved_revision": model_revision,
        "local_path": "/immutable/Qwen3.5-2B/" + model_revision,
        "files": model_files,
        "total_weight_bytes": 1024,
        "config_class": "Qwen3_5Config",
        "model_class": "Qwen3_5ForCausalLM",
        "tokenizer_class": "Qwen2TokenizerFast",
        "tokenizer_vocab_size": 151936,
        "trust_remote_code": False,
        "requested_dtype": "auto",
        "parameter_dtypes": {"bfloat16": 2000000000},
    }
    layer_schedule = {
        str(index): (
            "full_attention"
            if index in {3, 7, 11, 15, 19, 23}
            else "linear_attention"
        )
        for index in range(24)
    }
    environment = {
        "schema_version": contract.SCHEMA_VERSION,
        "host": "10.232.195.203",
        "user": "sitian",
        "gpu_name": "NVIDIA H100",
        "gpu_uuid": "GPU-synthetic",
        "driver_version": "550.54",
        "cuda_runtime_version": "12.1",
        "python_executable": (
            "/data00/home/sitian/sitian-workspace01/tllm/env/bin/python"
        ),
        "python_version": "3.11.9",
        "torch_version": "2.4.1+cu121",
        "transformers_version": "5.8.1",
        "optional_packages": {
            "fla": None,
            "causal_conv1d": None,
            "triton": "3.0.0",
            "flash_attn": "2.6.3",
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "TINYVLLM_DIST_PORT": "40101",
            "MASTER_PORT": "40102",
        },
        "gpu_processes_before": [],
        "gpu_processes_after": [],
    }
    case_rows = []
    state_snapshots = []
    state_components = []
    memory_snapshots = []
    lifetime_epoch = 0
    for case in contract.build_case_matrix():
        request_ids = ["request-0"]
        request_generations = [0]
        if case.phase in {
            "interleaved_multi_request",
            "completion_release_slot_reuse",
        }:
            request_ids = ["slot-0", "slot-1", "slot-2"]
            request_generations = [0, 0, 0]
        if case.phase == "completion_release_slot_reuse":
            request_ids.append("slot-0")
            request_generations.append(1)
        snapshot_ids = []
        memory_ids = []
        lifecycle_steps = None
        if case.phase == "completion_release_slot_reuse":
            lifecycle_steps = [
                ("slot-0", 0, 0, "before_prefill"),
                ("slot-1", 0, 0, "before_prefill"),
                ("slot-2", 0, 0, "before_prefill"),
                ("slot-0", 0, 17, "after_prefill"),
                ("slot-1", 0, 65, "after_prefill"),
                ("slot-2", 0, 257, "after_prefill"),
                *[
                    (
                        "slot-0",
                        0,
                        18 + step,
                        f"after_decode_step_{step}",
                    )
                    for step in range(2)
                ],
                *[
                    (
                        "slot-1",
                        0,
                        66 + step,
                        f"after_decode_step_{step}",
                    )
                    for step in range(8)
                ],
                *[
                    (
                        "slot-2",
                        0,
                        258 + step,
                        f"after_decode_step_{step}",
                    )
                    for step in range(8)
                ],
                ("slot-0", 0, 19, "after_request_release"),
                ("slot-0", 1, 33, "after_slot_reuse"),
                *[
                    (
                        "slot-0",
                        1,
                        34 + step,
                        f"after_decode_step_{step}",
                    )
                    for step in range(8)
                ],
            ]
            assert len(lifecycle_steps) == 34
        for snapshot_index in range(case.expected_state_snapshots):
            if lifecycle_steps is None:
                request_index = min(snapshot_index, len(request_ids) - 1)
                request_id = request_ids[request_index]
                request_generation = request_generations[request_index]
                sequence_length = max(
                    0,
                    case.prompt_length + snapshot_index - 1,
                )
                phase = (
                    "before_prefill"
                    if sequence_length == 0
                    else f"after_decode_step_{snapshot_index - 1}"
                )
            else:
                (
                    request_id,
                    request_generation,
                    sequence_length,
                    phase,
                ) = lifecycle_steps[snapshot_index]
            snapshot_id = (
                f"{case.case_id}:{request_id}:g{request_generation}:"
                f"e{lifetime_epoch}"
            )
            snapshot_components = []
            if sequence_length > 0:
                layer_index = 3
                role = "full_attention_key"
                layer_type = "full_attention"
                if case.phase == "completion_release_slot_reuse":
                    layer_index = 0
                    role = "linear_recurrent_state"
                    layer_type = "linear_attention"
                update_kind = "created" if snapshot_index == 0 else "grown"
                if role == "linear_recurrent_state":
                    if phase in {"after_prefill", "after_slot_reuse"}:
                        update_kind = "created"
                    elif phase == "after_request_release":
                        update_kind = "released"
                    else:
                        update_kind = "mutated_in_place"
                storage_identity = (
                    f"{case.case_id}:{request_id}:g{request_generation}:"
                    f"{layer_index}"
                )
                if role == "full_attention_key":
                    storage_identity += f":e{lifetime_epoch}"
                component = _component(
                    request_id=request_id,
                    request_generation=request_generation,
                    layer_index=layer_index,
                    layer_type=layer_type,
                    state_role=role,
                    tensor_path=f"layers[{layer_index}].state",
                    lifetime_epoch=lifetime_epoch,
                    sequence_length=sequence_length,
                    storage_identity=storage_identity,
                    update_kind=update_kind,
                )
                snapshot_components.append(component)
                state_components.append(component)
            state_snapshots.append({
                "snapshot_id": snapshot_id,
                "request_id": request_id,
                "request_generation": request_generation,
                "lifetime_epoch": lifetime_epoch,
                "sequence_length": sequence_length,
                "component_count": len(snapshot_components),
                "component_sha256": contract.canonical_json_sha256(
                    snapshot_components
                ),
            })
            memory_id = "memory:" + snapshot_id
            memory_components = (
                []
                if phase == "after_request_release"
                else snapshot_components
            )
            memory_snapshots.append({
                "snapshot_id": memory_id,
                "phase": phase,
                "request_id": request_id,
                "request_generation": request_generation,
                "cuda_allocated_bytes": 4096,
                "cuda_reserved_bytes": 8192,
                "logical_state_bytes": sum(
                    item["logical_bytes"] for item in memory_components
                ),
                "unique_storage_bytes": sum(
                    item["storage_nbytes"] for item in memory_components
                ),
            })
            snapshot_ids.append(snapshot_id)
            memory_ids.append(memory_id)
            lifetime_epoch += 1
        if case.phase == "interleaved_multi_request":
            logit_records = [
                _logit_record(
                    request_id=f"slot-{request_index}",
                    request_generation=0,
                    step_index=step_index,
                    sequence_length=prompt_length + step_index,
                )
                for step_index in range(case.decode_steps)
                for request_index, prompt_length in enumerate(
                    contract.MULTI_REQUEST_LENGTHS
                )
            ]
        elif case.phase == "completion_release_slot_reuse":
            request_domains = (
                ("slot-0", 0, contract.MULTI_REQUEST_LENGTHS[0], 2),
                (
                    "slot-1",
                    0,
                    contract.MULTI_REQUEST_LENGTHS[1],
                    case.decode_steps,
                ),
                (
                    "slot-2",
                    0,
                    contract.MULTI_REQUEST_LENGTHS[2],
                    case.decode_steps,
                ),
                (
                    "slot-0",
                    1,
                    contract.SLOT_REUSE_PROMPT_LENGTH,
                    case.decode_steps,
                ),
            )
            logit_records = [
                _logit_record(
                    request_id=request_id,
                    request_generation=request_generation,
                    step_index=step_index,
                    sequence_length=prompt_length + step_index,
                )
                for (
                    request_id,
                    request_generation,
                    prompt_length,
                    step_count,
                ) in request_domains
                for step_index in range(step_count)
            ]
        else:
            logit_records = [
                _logit_record(
                    request_id=request_ids[0],
                    request_generation=request_generations[0],
                    step_index=step_index,
                    sequence_length=case.prompt_length + step_index,
                )
                for step_index in range(case.decode_steps)
            ]
        decoded_token_ids = [
            record["topk_token_ids"][0] for record in logit_records
        ]
        case_rows.append({
            "row_id": "row:" + case.case_id,
            "case_id": case.case_id,
            "phase": case.phase,
            "execution_mode": case.execution_mode,
            "prompt_length": case.prompt_length,
            "chunk_schedule": list(case.chunk_schedule),
            "request_count": case.request_count,
            "decode_steps": case.decode_steps,
            "repeat_index": case.repeat_index,
            "request_ids": request_ids,
            "request_generations": request_generations,
            "decoded_token_ids": decoded_token_ids,
            "logit_records": logit_records,
            "state_snapshot_ids": snapshot_ids,
            "memory_snapshot_ids": memory_ids,
            "complete": True,
            "failure_kind": None,
            "failure_detail": None,
        })
    memory_snapshots = [
        {
            "snapshot_id": "lifecycle:before_model_load",
            "phase": "before_model_load",
            "request_id": "__model__",
            "request_generation": 0,
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 0,
            "logical_state_bytes": 0,
            "unique_storage_bytes": 0,
        },
        {
            "snapshot_id": "lifecycle:after_model_load",
            "phase": "after_model_load",
            "request_id": "__model__",
            "request_generation": 0,
            "cuda_allocated_bytes": 4000000000,
            "cuda_reserved_bytes": 4500000000,
            "logical_state_bytes": 0,
            "unique_storage_bytes": 0,
        },
        *memory_snapshots,
        {
            "snapshot_id": "lifecycle:after_model_release",
            "phase": "after_model_release",
            "request_id": "__model__",
            "request_generation": 0,
            "cuda_allocated_bytes": 0,
            "cuda_reserved_bytes": 4500000000,
            "logical_state_bytes": 0,
            "unique_storage_bytes": 0,
        },
    ]
    for filename, payload in (
        ("source_manifest.json", source_manifest),
        ("model_manifest.json", model_manifest),
        ("environment.json", environment),
        ("processes.json", {
            "processes": [{
                "name": "canonical",
                "attempt": 1,
                "command": ["python", "qwen35_hybrid_state_probe.py"],
                "stdout_path": "stdout/canonical.txt",
                "stderr_path": "stderr/canonical.txt",
                "exit_code": 0,
                "tinyvllm_dist_port": 40101,
                "master_port": 40102,
            }],
        }),
        ("ports.json", {
            "pairs": [{
                "process": "canonical",
                "attempt": 1,
                "tinyvllm_dist_port": 40101,
                "master_port": 40102,
            }],
        }),
        ("summary.json", {
            "schema_version": contract.SCHEMA_VERSION,
            "architecture": {
                "num_hidden_layers": 24,
                "linear_attention_layers": 18,
                "full_attention_layers": 6,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 16,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32",
                "layer_schedule": layer_schedule,
            },
            "case_row_count": len(case_rows),
            "state_snapshot_count": len(state_snapshots),
            "state_component_count": len(state_components),
            "memory_snapshot_count": len(memory_snapshots),
            "parameter_bytes": 4000000000,
            "state_logical_bytes": sum(
                item["logical_bytes"] for item in state_components
            ),
            "state_unique_storage_bytes": contract.unique_storage_bytes(
                state_components
            ),
            "max_memory_allocated": 5000000000,
            "max_memory_reserved": 6000000000,
            "non_state_peak_allocator_observation_bytes": 1000000000,
            "claim_boundary": (
                "Worker aggregate only; compatibility and performance "
                "claims require independent verification."
            ),
        }),
    ):
        _write_json(run_dir / filename, payload)
    _write_jsonl(run_dir / "case_rows.jsonl", case_rows)
    _write_jsonl(run_dir / "state_snapshots.jsonl", state_snapshots)
    _write_jsonl(run_dir / "state_components.jsonl", state_components)
    _write_jsonl(run_dir / "memory_snapshots.jsonl", memory_snapshots)
    (run_dir / "stdout" / "canonical.txt").write_text("ok\n")
    (run_dir / "stderr" / "canonical.txt").write_text("")
    listed_paths = [
        "source_manifest.json",
        "model_manifest.json",
        "environment.json",
        "case_rows.jsonl",
        "state_snapshots.jsonl",
        "state_components.jsonl",
        "memory_snapshots.jsonl",
        "processes.json",
        "ports.json",
        "stdout/canonical.txt",
        "stderr/canonical.txt",
        "summary.json",
    ]
    manifest = {
        "schema_version": contract.SCHEMA_VERSION,
        "run_id": "synthetic-complete",
        "source_commit": source_commit,
        "model_repository": contract.MODEL_REPOSITORY,
        "model_resolved_revision": model_revision,
        "artifacts": [
            {
                "path": relative,
                "size": (run_dir / relative).stat().st_size,
                "sha256": _sha256(run_dir / relative),
            }
            for relative in listed_paths
        ],
    }
    _write_json(run_dir / "manifest.json", manifest)
    return run_dir


def test_complete_synthetic_domain_verifies_go():
    with tempfile.TemporaryDirectory() as temporary:
        run_dir = write_complete_run(Path(temporary) / "run")
        result = verifier.verify_run(run_dir, write_report=True)
        assert result["classification"] == "GO"
        assert result["expected_case_count"] == len(
            contract.build_case_matrix()
        )
        assert (run_dir / "independent_verification.json").is_file()
        assert (run_dir / "report.md").is_file()
        rows = [
            json.loads(line)
            for line in (run_dir / "case_rows.jsonl").read_text().splitlines()
        ]
        interleaved = _find_case(rows, "interleaved_multi_request")
        lifecycle = _find_case(rows, "completion_release_slot_reuse")
        assert len(interleaved["logit_records"]) == 24
        assert len(lifecycle["logit_records"]) == 26


def test_provenance_and_domain_tampering_is_incomplete():
    cases = (
        (_remove_case_row, "missing canonical case"),
        (_duplicate_case_row, "duplicate canonical case"),
        (_add_unknown_case, "unknown canonical case"),
        (_alter_source_hash, "source hash mismatch"),
        (_alter_model_revision, "model revision mismatch"),
        (_reuse_port, "port reuse"),
        (_add_unlisted_input, "unlisted artifact"),
    )
    with tempfile.TemporaryDirectory() as temporary:
        base_run = write_complete_run(Path(temporary) / "complete")
        for mutator, message in cases:
            _expect_incomplete(base_run, mutator, message)


def test_correctness_tampering_separates_no_go_from_incomplete():
    cases = (
        (
            _change_cached_token,
            "INCOMPLETE",
            "decoded token evidence mismatch",
        ),
        (
            _remove_cached_step,
            "INCOMPLETE",
            "missing logit step",
        ),
        (
            _change_cached_full_logit_hash,
            "INCOMPLETE",
            "actual full-logit hash mismatch",
        ),
        (
            _change_cached_oracle_token,
            "NO_GO",
            "oracle greedy token mismatch",
        ),
        (
            _change_cached_oracle_hash,
            "NO_GO",
            "oracle full-logit hash mismatch",
        ),
        (
            _raise_repeatability_above_cap,
            "INCOMPLETE",
            "INCOMPLETE_NUMERICAL_INSTABILITY",
        ),
        (
            _change_chunk_schedule,
            "INCOMPLETE",
            "chunk_schedule",
        ),
        (
            _change_export_token,
            "NO_GO",
            "export/import token mismatch",
        ),
        (
            _make_topk_non_finite,
            "INCOMPLETE",
            "non-finite top-k logit",
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        base_run = write_complete_run(Path(temporary) / "complete")
        for mutator, classification, message in cases:
            _expect_classification(
                base_run,
                mutator,
                classification,
                message,
            )


def test_lifecycle_tampering_is_incomplete():
    cases = (
        (_set_unexplained_role, "unexplained state role"),
        (_grow_recurrent_state, "recurrent state shape grew"),
        (_stop_full_kv_growth, "full-attention state did not grow"),
        (_cross_request_component_mutation, "cross-request state mutation"),
        (_remove_release_component, "missing release"),
        (_reuse_without_generation_increment, "slot generation did not increment"),
        (_reuse_stale_storage_identity, "stale storage identity"),
        (_reuse_stale_content, "stale content"),
        (_make_layer_identity_ambiguous, "ambiguous layer identity"),
        (_set_unsupported_update_kind, "unsupported update kind"),
    )
    with tempfile.TemporaryDirectory() as temporary:
        base_run = write_complete_run(Path(temporary) / "complete")
        for mutator, message in cases:
            _expect_incomplete(base_run, mutator, message)


def test_storage_ledger_tampering_is_incomplete():
    cases = (
        (_set_incorrect_logical_bytes, "incorrect logical bytes"),
        (
            _set_conflicting_alias_storage_size,
            "conflicting storage sizes",
        ),
        (_double_count_storage, "unique storage bytes mismatch"),
        (_remove_memory_epoch, "missing memory epoch"),
        (_set_negative_allocator_bytes, "negative allocator bytes"),
        (_set_parameter_byte_mismatch, "parameter-byte mismatch"),
        (
            _set_worker_aggregate_disagreement,
            "worker aggregate disagreement",
        ),
    )
    with tempfile.TemporaryDirectory() as temporary:
        base_run = write_complete_run(Path(temporary) / "complete")
        for mutator, message in cases:
            _expect_incomplete(base_run, mutator, message)


if __name__ == "__main__":
    test_complete_synthetic_domain_verifies_go()
    test_provenance_and_domain_tampering_is_incomplete()
    test_correctness_tampering_separates_no_go_from_incomplete()
    test_lifecycle_tampering_is_incomplete()
    test_storage_ledger_tampering_is_incomplete()
    print("qwen35 hybrid-state verifier tests passed")
